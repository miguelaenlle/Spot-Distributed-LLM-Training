"""Per-node live-log dashboard: ``spot-orchestrate logs <run_id>``.

A read-only full-screen viewer for a (running or finished) multi-node run: one
tab per (node, attempt) plus an ``orch`` tab for the supervisor's own decision
log, arrow keys to switch, the selected log tailing below. A dead node's tab
flips to ``(dead)`` and its log freezes (still selectable/scrollable); a
replacement or late joiner simply appears as a new tab — so an elastic
kill → shrink → relaunch → grow sequence is watchable node by node.

Discovery prefers the supervisor's ``status.json`` (per-(node, attempt) state +
log keys, rewritten every tick); when it's absent (old or single-node runs) it
falls back to listing ``logs/`` and inferring liveness from each object's
last-modified age. Everything reads through :mod:`spot_train.s3_store` URIs, so
the same code runs against S3 and against a plain local directory (tests, and
the localhost epoch e2e harness).

Split like supervisor.py: a pure-ish data layer (:func:`discover`,
:func:`merge`, :func:`poll`, :func:`render_frame`, :func:`decode_key` — no
terminal, table-testable) under a thin ANSI shell (:func:`run_logs`), following
monitor.py's no-curses convention.
"""

from __future__ import annotations

import json
import os
import re
import select
import shutil
import sys
import time
from dataclasses import dataclass, field

from spot_train import events, s3_store

from .config import OrchestratorConfig

ORCH = -1  # tab id of the orchestrator/supervisor decision log
# status.json is rewritten every supervisor tick (~3s); older than this and the
# control plane itself is presumed wedged/gone.
STALE_AFTER_S = 15.0

# Grid mode: at most this many live panes, laid out with this many columns for a
# given pane count (index = count-1). Wide-and-short beats tall-and-thin for
# reading log tails, so we grow columns before rows.
MAX_GRID = 8
_GRID_COLS = [1, 2, 2, 2, 3, 3, 4, 4]

# boot.log / boot-node2.log / boot-node1-r2.log / seg-3.log / seg-3-node1.log
_LOG_NAME = re.compile(r"^(?:boot|seg-\d+)(?:-node(\d+))?(?:-r(\d+))?\.log$")


@dataclass
class Tab:
    """One log stream: a (node, attempt) box, or the orchestrator (node=ORCH).
    ``fetched`` is the byte offset already downloaded; ``buf`` the full content
    so a frozen (dead) tab stays scrollable."""

    node: int
    attempt: int
    log_uri: str
    state: str = "pending"  # pending | alive | dead
    fetched: int = 0
    buf: bytearray = field(default_factory=bytearray)

    @property
    def label(self) -> str:
        if self.node == ORCH:
            return "orch"
        name = f"node{self.node}"
        return f"{name}·r{self.attempt}" if self.attempt else name


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #
def _log_uri(base: str, log_key: str) -> str:
    """Map a bucket key from status.json to a URI under this run's base. Only
    the basename is trusted — log keys always live at <run>/logs/<name>."""
    return f"{base}/logs/{log_key.rsplit('/', 1)[-1]}"


def _from_status(base: str, doc: dict) -> tuple[dict, list[dict]]:
    meta = {
        "run_id": doc.get("run_id", base.rsplit("/", 1)[-1]),
        "epoch": doc.get("epoch"),
        "members": doc.get("members"),
        "ckpt_step": doc.get("ckpt_step"),
        "updated_at": doc.get("updated_at"),
        "done": bool(doc.get("done")),
        "source": "status",
    }
    infos = []
    orch_key = (doc.get("orchestrator") or {}).get("log_key")
    if orch_key:
        infos.append(
            {"node": ORCH, "attempt": 0, "log_uri": _log_uri(base, orch_key), "state": "alive"}
        )
    for e in doc.get("nodes", []):
        infos.append(
            {
                "node": e["node"],
                "attempt": e.get("attempt", 0),
                "log_uri": _log_uri(base, e["log_key"]),
                "state": e.get("state", "pending"),
            }
        )
    return meta, infos


def _from_listing(base: str, heartbeat_timeout_s: float, now: float) -> tuple[dict, list[dict]]:
    """No status.json (old or single-node run): infer tabs from the logs/ dir.
    Liveness = the object changed within the heartbeat window and the run isn't
    done — the same signal the supervisor itself uses."""
    done = s3_store.exists(f"{base}/metrics.json")
    meta = {
        "run_id": base.rsplit("/", 1)[-1],
        "epoch": None,
        "members": None,
        "updated_at": None,
        "done": done,
        "source": "listing",
    }
    infos = []
    for name in s3_store.list_names(f"{base}/logs/"):
        uri = f"{base}/logs/{name}"
        if name == "orchestrator.log":
            infos.append({"node": ORCH, "attempt": 0, "log_uri": uri, "state": "alive"})
            continue
        m = _LOG_NAME.match(name)
        if not m:
            continue
        lm = s3_store.last_modified(uri)
        alive = not done and lm is not None and (now - lm) <= heartbeat_timeout_s
        infos.append(
            {
                "node": int(m.group(1) or 0),
                "attempt": int(m.group(2) or 0),
                "log_uri": uri,
                "state": "alive" if alive else "dead",
            }
        )
    return meta, infos


def discover(
    run_uri: str, *, heartbeat_timeout_s: float = 90.0, now: float | None = None
) -> tuple[dict, list[dict]]:
    """One poll of the run's control state -> (meta, tab infos)."""
    now = time.time() if now is None else now
    base = run_uri.rstrip("/")
    raw = s3_store.read_bytes(f"{base}/status.json")
    if raw is not None:
        try:
            return _from_status(base, json.loads(raw))
        except (ValueError, KeyError):
            pass  # malformed doc: fall through to listing
    return _from_listing(base, heartbeat_timeout_s, now)


def merge(tabs: dict[tuple[int, int], Tab], infos: list[dict]) -> list[tuple[str, Tab]]:
    """Fold freshly-discovered infos into the tab set. Returns transition
    events: ("new", tab) for a tab that just appeared, ("dead", tab) the one
    time it dies. Death is one-way — a dead tab never resurrects, its log is
    frozen for good."""
    events = []
    for info in infos:
        k = (info["node"], info["attempt"])
        t = tabs.get(k)
        if t is None:
            t = Tab(
                node=info["node"],
                attempt=info["attempt"],
                log_uri=info["log_uri"],
                state=info["state"],
            )
            tabs[k] = t
            events.append(("new", t))
        elif t.state != "dead":
            if info["state"] == "dead":
                t.state = "dead"
                events.append(("dead", t))
            else:
                t.state = info["state"]
    return events


def poll(tab: Tab, *, force: bool = False) -> bytes:
    """Fetch any new bytes of the tab's log (ranged read from the last offset).
    Dead tabs are frozen — never polled — except ``force``: once at the moment
    of death (grab the final traceback) and once on first view of a tab that
    was already dead when discovered (post-mortem)."""
    if tab.state == "dead" and not force:
        return b""
    new = s3_store.read_bytes_from(tab.log_uri, tab.fetched)
    if new:
        tab.fetched += len(new)
        tab.buf.extend(new)
    return new


# --------------------------------------------------------------------------- #
# Rendering (pure: state -> one frame string)
# --------------------------------------------------------------------------- #
def _badge(t: Tab, meta: dict, now: float) -> str:
    if t.node == ORCH:
        upd = meta.get("updated_at")
        if not meta.get("done") and upd is not None and now - upd > STALE_AFTER_S:
            return f" (stale {int(now - upd)}s)"
        return ""
    if t.state == "dead":
        # After a clean finish in listing mode every log looks dead — that's
        # just "the run ended", not a preemption; don't shout (dead).
        if meta.get("done") and meta.get("source") == "listing":
            return ""
        return " (dead)"
    if t.state == "pending":
        return " (joining)"
    return ""


def _tab_bar(tabs: list[Tab], selected: int, cols: int, meta: dict, now: float) -> str:
    cells = [f"[{t.label}{_badge(t, meta, now)}]" for t in tabs]
    # Slide the window right until the selected cell fits on screen.
    start = 0
    while start < selected and sum(len(c) + 2 for c in cells[start : selected + 1]) - 2 > cols:
        start += 1
    out: list[str] = []
    used = 0
    for i in range(start, len(cells)):
        w = len(cells[i]) + (2 if out else 0)
        if used + w > cols and out:
            break
        used += w
        out.append(f"\x1b[7m{cells[i]}\x1b[0m" if i == selected else cells[i])
    return "  ".join(out)


def _status_line(meta: dict, now: float, cols: int) -> str:
    """The run/epoch/members/supervisor-age summary, shared by both views."""
    parts = [f"run: {meta.get('run_id', '?')}"]
    if meta.get("epoch") is not None:
        parts.append(f"epoch {meta['epoch']}")
    if meta.get("members") is not None:
        parts.append("members " + (",".join(map(str, meta["members"])) or "—"))
    if meta.get("updated_at") is not None:
        parts.append(f"supervisor {max(0, int(now - meta['updated_at']))}s ago")
    if meta.get("source") == "listing":
        parts.append("listing mode (no status.json)")
    if meta.get("done"):
        parts.append("RUN COMPLETE")
    return " | ".join(parts)[:cols]


def render_frame(
    tabs: list[Tab],
    selected: int,
    scroll: int,
    size: tuple[int, int],
    meta: dict,
    now: float,
) -> str:
    """One full frame: tab bar, status line, rule, the selected log's visible
    window, footer. Pure — drives the snapshot tests."""
    cols, rows = max(20, size[0]), max(6, size[1])
    body_rows = rows - 4

    bar = _tab_bar(tabs, selected, cols, meta, now) if tabs else "(waiting for logs…)"
    status = _status_line(meta, now, cols)

    lines: list[str] = []
    if tabs:
        lines = tabs[selected].buf.decode("utf-8", errors="replace").splitlines()
    end = max(0, len(lines) - scroll)
    visible = [ln[:cols] for ln in lines[max(0, end - body_rows) : end]]
    visible += [""] * (body_rows - len(visible))

    follow = "[FOLLOW]" if scroll == 0 else f"[SCROLL -{scroll}]"
    help_txt = "←/→ node   ↑/↓ scroll   g grid   t timeline   v events   q quit"
    footer = (help_txt + " " * max(1, cols - len(help_txt) - len(follow)) + follow)[:cols]

    return "\n".join([bar, status, "─" * cols, *visible, footer])


_BADGE = {"alive": "LIVE", "dead": "DEAD", "pending": "JOIN"}


def _cell(tab: Tab, w: int, h: int) -> list[str]:
    """One grid pane: a bold header (label + LIVE/DEAD/JOIN) over the log tail,
    exactly ``h`` lines each ``w`` wide. A dead pane is dimmed in place so a
    node that left is visible, frozen, next to the ones still running."""
    head = f"{tab.label} [{_BADGE.get(tab.state, '?')}]"[:w].ljust(w)
    body_h = max(0, h - 1)
    log_lines = tab.buf.decode("utf-8", errors="replace").splitlines()
    body = [ln[:w].ljust(w) for ln in log_lines[-body_h:]] if body_h else []
    body = [" " * w] * (body_h - len(body)) + body  # bottom-align the tail
    cell = [head, *body]
    if tab.state == "dead":
        return [f"\x1b[2m{ln}\x1b[0m" for ln in cell]  # dim the whole pane
    return [f"\x1b[1m{cell[0]}\x1b[0m", *cell[1:]]  # bold just the header


def render_grid(tabs: list[Tab], size: tuple[int, int], meta: dict, now: float) -> str:
    """All logs at once: a header line over up to :data:`MAX_GRID` live panes,
    one per node (newest attempt) plus the orchestrator. Membership is whatever
    ``tabs`` holds this tick, so a joiner appears as a new pane and a node that
    left stays as a dimmed [DEAD] pane — which joined / which left, in one view.
    Pure, like :func:`render_frame`."""
    cols, rows = max(20, size[0]), max(6, size[1])
    header = _status_line(meta, now, cols)
    show = tabs[:MAX_GRID]
    n = len(show)
    body_rows = rows - 2
    footer = ("grid — 0-9 focus pane   t timeline   g single-view   q quit")[:cols]
    if n == 0:
        blank = [""] * body_rows
        return "\n".join([header, *blank, footer])

    ncols = _GRID_COLS[n - 1]
    nrows = -(-n // ncols)  # ceil
    cell_w = (cols - (ncols - 1)) // ncols  # ncols panes + (ncols-1) separators
    cell_h = max(1, (body_rows - (nrows - 1)) // nrows)  # rows minus inter-row rules

    lines: list[str] = []
    for r in range(nrows):
        blocks = [_cell(t, cell_w, cell_h) for t in show[r * ncols : (r + 1) * ncols]]
        while len(blocks) < ncols:  # pad a short last row with empty panes
            blocks.append([" " * cell_w] * cell_h)
        for i in range(cell_h):
            lines.append("│".join(b[i] for b in blocks))
        if r < nrows - 1:
            lines.append("─" * cols)
    lines = (lines + [""] * body_rows)[:body_rows]  # exact fill
    return "\n".join([header, *lines, footer])


# --------------------------------------------------------------------------- #
# Timeline (Gantt): one row per node, wall-clock x-axis, observed states
# --------------------------------------------------------------------------- #
# The stacked phase bar in the run profile collapses the whole run into one
# global "provisioning/training/degraded" timeline, which mislabels a node that
# died-and-rejoined as blanket "degraded". This view records what the dashboard
# actually observes per node, tick by tick, and draws it as a Gantt — the truth,
# per node. It reflects what the viewer saw, so attach at run start for the full
# history.
# Gantt labels: prov = booting/restoring, reconfig = REALIZED the world changed
# and tearing down the old collective (before provisioning), train = productively
# training, wasted = re-doing steps rolled back to the last checkpoint (lost
# work), stalled = blocked on a down peer, restart = inferred-fallback re-rdzv,
# down = gone/reclaimed/killed.
_GANTT = {
    "prov": "░",
    "train": "▓",
    "wasted": "▨",
    "stalled": "▚",
    "restart": "▒",
    "down": "·",
}
_GANTT_COLORS = {  # for the PNG (order = legend order)
    "prov": "#4C78A8",
    "train": "#59A14F",
    "wasted": "#F58518",
    "stalled": "#E45756",
    "restart": "#EDC948",
    "down": "#BAB0AC",
}
_REALIZED_COLOR = "#B279A2"  # the "realized world changed" marker (a point in time)
# A restart window with no observable checkpoint progress is capped here so a
# missing/never-advancing ckpt_step can't paint a permanent restart bar.
_RESTART_CAP_S = 90.0

# Event-sourced timeline: node lifecycle event `state` -> Gantt label. killed and
# down are the same "gone" bar (distinguished by cause + the ✗ marker). "wasted"
# is derived (a step rollback), not an emitted state.
_EVENT_STATE_MAP = {
    "provisioning": "prov",
    "reconfiguring": "reconfig",
    "training": "train",
    "stalled": "stalled",
    "down": "down",
    "killed": "down",
}
# One-line human descriptions for the events view.
_EVENT_HUMAN = {
    "provisioning": "provisioning",
    "reconfiguring": "REALIZED world changed (tearing down)",
    "training": "training",
    "stalled": "STALLED (blocked on down peer)",
    "down": "DOWN (reclaimed)",
    "killed": "KILLED",
    "epoch": "epoch published",
}


def _compress(samples: list[tuple[float, object]], t0: float, now: float) -> list[tuple]:
    """Merge consecutive equal-value samples into (start_rel, duration, value)."""
    out: list[list] = []
    for i, (w, v) in enumerate(samples):
        end = samples[i + 1][0] if i + 1 < len(samples) else now
        if out and out[-1][2] == v:
            out[-1][1] = end - t0 - out[-1][0]
        else:
            out.append([w - t0, end - w, v])
    return [tuple(r) for r in out]


@dataclass
class TimelineRecorder:
    """Per-node observed history + the world-size (epoch membership) curve,
    accumulated one sample per tick.

    A node's label is driven by MEMBERSHIP, not raw liveness: it's ``train`` only
    while it's an admitted member of the current epoch, ``prov`` while it's
    booting/joining-but-not-yet-a-member, ``down`` once dead. That's why a
    booting node reads as provisioning, never as "training before it started" —
    fresh-boot-log liveness (the listing fallback's only signal) is deliberately
    NOT trusted here; the caller feeds only status.json ticks, which carry the
    epoch's member set. World size = |members|; the full world is the max seen."""

    t0: float | None = None
    # Rows are keyed by a "row id": an int node (inferred fallback) or a
    # (node, attempt) tuple (event-sourced), so a replacement gets its own line.
    samples: dict = field(default_factory=dict)
    kills: list = field(default_factory=list)  # (ts, row_id): a box went down/killed
    realized: list = field(default_factory=list)  # (ts, row_id): node saw the world change
    leaders: list = field(default_factory=list)  # (ts, node): node became rank-0 leader
    world: list[tuple[float, int]] = field(default_factory=list)  # (wall, world size)
    full: int = 0  # full world size (max |members| ever observed)
    wasted: dict = field(default_factory=dict)  # row_id -> steps re-done after a rollback
    _last: dict[int, str] = field(default_factory=dict)
    _epoch: int | None = None
    _prev_members: set[int] = field(default_factory=set)
    _restarting: set[int] = field(default_factory=set)  # survivors re-rendezvousing now
    _restart_at: float = 0.0
    _restart_ckpt: int | None = None  # ckpt step at the epoch change; resumed once it advances

    def _track_restart(self, w: float, members: set[int], epoch: int | None, ckpt: int | None):
        """A survivor (a continuing member across an epoch change) re-rendezvouses
        at the new world size: its torchrun crashed on the peer's NCCL abort and
        relaunches, not training until checkpoint progress resumes. Detect that
        window from observable signals — the epoch bump and checkpoint step."""
        if epoch is not None and self._epoch is not None and epoch != self._epoch:
            survivors = set(members) & self._prev_members  # in both epochs => restarting
            if survivors:
                self._restarting = survivors
                self._restart_at = w
                self._restart_ckpt = ckpt
        if epoch is not None:
            self._epoch = epoch
        self._prev_members = set(members)
        # Resume: checkpoint advanced past the change, OR the safety cap elapsed.
        resumed = self._restart_ckpt is not None and ckpt is not None and ckpt > self._restart_ckpt
        if resumed or (w - self._restart_at) > _RESTART_CAP_S:
            self._restarting = set()
        self._restarting &= set(members)  # a node that left is no longer "restarting"

    def update(
        self,
        w: float,
        tabs: dict[tuple[int, int], Tab],
        members: set[int],
        epoch: int | None = None,
        ckpt_step: int | None = None,
    ) -> None:
        if self.t0 is None:
            self.t0 = w
        self._track_restart(w, members, epoch, ckpt_step)
        for node in sorted({n for (n, _a) in tabs if n != ORCH}):
            newest = max((t for (n, _a), t in tabs.items() if n == node), key=lambda t: t.attempt)
            if newest.state == "dead":
                label = "down"
            elif node in self._restarting:
                label = "restart"  # survivor re-rendezvousing at the new world size
            elif node in members:
                label = "train"  # admitted to the epoch => actually in the group
            else:
                label = "prov"  # registered/booting/joining, not yet a member
            if self._last.get(node) == "train" and label == "down":
                self.kills.append((w, node))
            self.samples.setdefault(node, []).append((w, label))
            self._last[node] = label
        ws = len(members)
        self.world.append((w, ws))
        self.full = max(self.full, ws)

    def runs(self, node: int, now: float) -> list[tuple[float, float, str]]:
        """A node's spans as (start_rel, duration, label)."""
        t0 = self.t0 if self.t0 is not None else now  # not `or`: t0 can be 0.0
        return _compress(self.samples.get(node, []), t0, now)

    def world_runs(self, now: float) -> list[tuple[float, float, int]]:
        """The world-size curve as (start_rel, duration, world_size) spans."""
        t0 = self.t0 if self.t0 is not None else now
        return _compress(self.world, t0, now)

    def degraded(self, now: float) -> dict:
        """Downtime attributable to world-size CHANGE: spans below full world,
        counted only AFTER full world is first reached (the initial boot ramp is
        provisioning, not a shrink). Grouped into recovery windows: ``total``
        seconds + per-window (start_rel, duration, min_ws) — "how long were we
        shrunk by preemption, and for each dip how far?"."""
        windows: list[tuple[float, float, int]] = []
        cur: list | None = None
        reached_full = False
        for s, d, ws in self.world_runs(now):
            if self.full and ws >= self.full:
                reached_full = True
            if reached_full and self.full and ws < self.full:
                if cur is None:
                    cur = [s, d, ws]
                else:
                    cur[1] += d
                    cur[2] = min(cur[2], ws)
            elif cur is not None:
                windows.append(tuple(cur))
                cur = None
        if cur is not None:
            windows.append(tuple(cur))
        return {
            "full": self.full,
            "current": self.world[-1][1] if self.world else 0,
            "total": sum(w[1] for w in windows),
            "windows": windows,
        }

    @classmethod
    def from_events(cls, records: list[dict], now: float) -> TimelineRecorder:
        """Build the timeline from source-stamped lifecycle events instead of
        polled inference: per-node state spans from node events, the world-size
        curve from the orchestrator's ``epoch`` events, kill markers from
        down/killed. Every timestamp is the emitter's own ``ts`` — exact, and
        reconstructable post-mortem from the logs alone. Returns an empty
        recorder (``.samples`` falsy) when there are no usable events, so the
        caller can fall back to the inferred timeline."""
        rec = cls()
        node_recs = [
            r for r in records if r.get("state") in _EVENT_STATE_MAP and r.get("node") is not None
        ]
        world_recs = [
            r for r in records if r.get("state") == "epoch" and r.get("world") is not None
        ]
        if not node_recs and not world_recs:
            return rec
        rec.t0 = min(r["ts"] for r in node_recs + world_recs)

        # One row per (node, attempt): a killed original and its replacement are
        # distinct boxes and get distinct lines.
        by_row: dict[tuple[int, int], list[dict]] = {}
        for r in sorted(node_recs, key=lambda r: r["ts"]):
            row = (r["node"], r.get("attempt", 0))
            by_row.setdefault(row, []).append(r)
            if r["state"] == "reconfiguring":
                # An INSTANT (realized the world changed), not a duration — the
                # teardown->relaunch is usually same-tick, so a segment would be
                # zero-width. Render it as a point marker on the row instead.
                rec.realized.append((r["ts"], row))
                continue
            rec.samples.setdefault(row, []).append((r["ts"], _EVENT_STATE_MAP[r["state"]]))
            if r["state"] in ("killed", "down"):
                rec.kills.append((r["ts"], row))

        rec.world = sorted((r["ts"], int(r["world"])) for r in world_recs)
        rec.full = max((w for _t, w in rec.world), default=0)
        # Leadership (rank-0) handovers from the epoch events, deduped to the
        # moments the leader actually changes.
        prev_leader = None
        for r in sorted(world_recs, key=lambda r: r["ts"]):
            ldr = r.get("leader")
            if ldr is not None and ldr != prev_leader:
                rec.leaders.append((r["ts"], int(ldr)))
                prev_leader = ldr
        rec._overlay_wasted(by_row)
        return rec

    def leader_row(self, node: int, ts: float) -> tuple[int, int]:
        """The (node, attempt) row that was the live box for ``node`` at ``ts`` —
        the highest attempt started by then — so a leadership marker lands on the
        right line even after that node was itself replaced."""
        started = [a for (n, a), s in self.samples.items() if n == node and s and s[0][0] <= ts]
        if started:
            return (node, max(started))
        attempts = sorted(a for (n, a) in self.samples if n == node)
        return (node, attempts[0]) if attempts else (node, 0)

    def current_leader(self, now: float) -> int | None:
        """The node holding rank 0 at ``now`` (the last handover at or before it)."""
        seen = [n for ts, n in self.leaders if ts <= now]
        return seen[-1] if seen else None

    def _overlay_wasted(self, by_row: dict[tuple[int, int], list[dict]]) -> None:
        """Lost work: when a survivor's torchrun crashes and it restores from the
        last checkpoint, the steps trained since that checkpoint are thrown away
        and re-done. Detect the step ROLLBACK (a training event whose step < the
        peak reached before the crash) and paint the re-compute window a distinct
        "wasted" — sized in wall-time by the pre-crash step rate, labelled with the
        exact lost step count."""
        for row, rs in by_row.items():
            seg_start: tuple[float, int] | None = None  # (ts, step) of current train run
            peak: int | None = None  # max step reached in it
            peak_ts = 0.0
            for r in rs:
                st, step = r["state"], r.get("step")
                if st == "training":
                    rolled_back = (
                        peak is not None
                        and step is not None
                        and step < peak
                        and seg_start is not None
                        and seg_start[1] is not None
                    )
                    if rolled_back:
                        lost = peak - step
                        run_steps = peak - seg_start[1]
                        run_secs = peak_ts - seg_start[0]
                        rate = run_steps / run_secs if run_secs > 0 else 0
                        secs = lost / rate if rate > 0 else 0
                        if lost > 0 and secs > 0:
                            self._paint_wasted(row, r["ts"], r["ts"] + secs)
                            self.wasted[row] = self.wasted.get(row, 0) + lost
                    seg_start, peak, peak_ts = (r["ts"], step), step, r["ts"]
                elif st == "stalled" and step is not None:
                    peak = step if peak is None else max(peak, step)
                    peak_ts = r["ts"]

    def _paint_wasted(self, row: tuple[int, int], ws: float, we: float) -> None:
        """Relabel the leading [ws, we) of the post-resume training run as
        "wasted" (the re-compute), clamped to the next real transition."""
        samples = self.samples[row]
        for i, (ts, _lbl) in enumerate(samples):
            if ts == ws:
                nxt = samples[i + 1][0] if i + 1 < len(samples) else float("inf")
                we = min(we, nxt)
                samples[i] = (ws, "wasted")
                if ws < we < nxt:  # re-open training after the re-compute window
                    samples.insert(i + 1, (we, "train"))
                return


def _row_parts(row) -> tuple[int, int]:
    """A row id -> (node, attempt), accepting either an int node (inferred
    fallback) or a (node, attempt) tuple (event-sourced)."""
    return row if isinstance(row, tuple) else (row, 0)


def _row_label(row) -> str:
    node, attempt = _row_parts(row)
    if node == ORCH:
        return "orch"
    return f"n{node}" if not attempt else f"n{node}·r{attempt}"


def render_gantt(
    rec: TimelineRecorder, now: float, size: tuple[int, int], meta: dict, note: str = ""
) -> str:
    """One frame of the per-(node, attempt) Gantt. Pure — drives the snapshots."""
    cols, rows = max(40, size[0]), max(8, size[1])
    rowkeys = sorted(rec.samples, key=_row_parts)
    labelw = max(4, *(len(_row_label(r)) for r in rowkeys)) if rowkeys else 4
    inner = cols - labelw - 1
    t0 = rec.t0 if rec.t0 is not None else now
    span = max(1.0, now - t0)

    title = f"Run timeline — {meta.get('run_id', '?')}   elapsed {int(span)}s"[:cols]
    scale = [" "] * inner
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        txt, xi = f"{int(span * frac)}s", min(inner - 1, int(frac * (inner - 1)))
        xi = max(0, min(xi, inner - len(txt)))
        for j, ch in enumerate(txt):
            scale[xi + j] = ch
    axis = " " * (labelw + 1) + "".join(scale)

    def _at(samples: list, t: float, k: int) -> tuple[object, int]:
        while k + 1 < len(samples) and samples[k + 1][0] <= t:
            k += 1
        return (samples[k][1] if samples else None), k

    body: list[str] = []
    for row in rowkeys:
        s = rec.samples[row]
        chars, k = [], 0
        for x in range(inner):
            lbl, k = _at(s, t0 + (x + 0.5) * span / inner, k)
            chars.append(_GANTT.get(lbl, " "))
        for rw, rn in rec.realized:  # "◆" = this node realized the world changed
            if rn == row:
                xi = int((rw - t0) / span * inner)
                if 0 <= xi < inner:
                    chars[xi] = "◆"
        for lw, ln in rec.leaders:  # "★" = this node became rank-0 leader here
            if rec.leader_row(ln, lw) == row:
                xi = int((lw - t0) / span * inner)
                if 0 <= xi < inner:
                    chars[xi] = "★"
        for kw, kn in rec.kills:  # "✗" drawn last so a kill is never hidden
            if kn == row:
                xi = int((kw - t0) / span * inner)
                if 0 <= xi < inner:
                    chars[xi] = "✗"
        body.append(_row_label(row).center(labelw) + "│" + "".join(chars))

    # World-size strip: the N -> N-1 -> N staircase as a sparkline under the rows.
    blocks = " ▁▂▃▄▅▆▇█"
    ws_chars, k = [], 0
    for x in range(inner):
        ws, k = _at(rec.world, t0 + (x + 0.5) * span / inner, k)
        if not ws or not rec.full:
            ws_chars.append(" ")
        else:
            ws_chars.append(blocks[max(1, round(ws / rec.full * 8))])
    ws_row = "ws".center(labelw) + "│" + "".join(ws_chars)

    deg = rec.degraded(now)
    n_dips = len(deg["windows"])
    lost = sum(rec.wasted.values())
    leader = rec.current_leader(now)
    ldr_txt = f"   leader node{leader}" if leader is not None else ""
    summary = (
        f" world {deg['current']}/{deg['full']}   "
        f"degraded {int(deg['total'])}s ({n_dips} dip{'' if n_dips == 1 else 's'})   "
        f"wasted {lost} steps{ldr_txt}"
    )[:cols]

    rule = "─" * labelw + "┴" + "─" * inner
    legend = " ░prov ▓train ▨wasted ▚stalled ·down ✗gone ◆realized ★leader"[:cols]
    footer = (note or "[e] export   [v] events   [t] tabs   [g] grid   [q] quit")[:cols]
    head = [title, axis]
    foot = [rule, ws_row, summary, legend, footer]
    avail = rows - len(head) - len(foot)
    body = (body + [""] * avail)[:avail]
    return "\n".join([*head, *body, *foot])


def parse_run_events(items: list[tuple[str, str]]) -> list[dict]:
    """Parse ``[event]`` records from a run's raw logs — ``(filename, text)`` per
    log object (as fetched straight from S3, no Tab objects). node/attempt are
    attributed from the filename (``boot-node1-r2.log`` -> node 1, attempt 2);
    ``orchestrator.log`` carries its node in the record. For post-run reporting
    (the Gantt / events.txt export) without attaching the live viewer."""
    recs: list[dict] = []
    for name, text in items:
        if name == "orchestrator.log":
            recs += events.parse(text, default_node=None)
            continue
        m = _LOG_NAME.match(name)
        if not m:
            continue
        recs += events.parse(
            text, default_node=int(m.group(1) or 0), default_attempt=int(m.group(2) or 0)
        )
    return recs


def collect_events(tabs: dict[tuple[int, int], Tab]) -> list[dict]:
    """Parse every ``[event]`` record out of all tab buffers. Node logs carry
    node-lifecycle events (default-attributed to the tab's node); the orchestrator
    log carries epoch / killed / down. Idempotent: each line lives in exactly one
    buffer, so re-parsing the growing buffers never duplicates."""
    recs: list[dict] = []
    for (node, attempt), tab in tabs.items():
        text = tab.buf.decode("utf-8", errors="replace")
        if node == ORCH:  # orch events carry their own node + attempt
            recs += events.parse(text, default_node=None)
        else:  # a box's log => its (node, attempt); replacements land on their row
            recs += events.parse(text, default_node=node, default_attempt=attempt)
    return recs


def _event_line(r: dict, t0: float, cols: int) -> str:
    ts = time.strftime("%H:%M:%S", time.localtime(r["ts"]))
    if r.get("node") is None:
        who = "orch"
    else:
        a = r.get("attempt") or 0
        who = f"node{r['node']}" + (f"·r{a}" if a else "")
    base = _EVENT_HUMAN.get(r["state"], r["state"])
    extra = []
    if r.get("world") is not None:
        extra.append(f"world {r['world']}")
    if r.get("step") is not None:
        extra.append(f"step {r['step']}")
    if r.get("cause"):
        extra.append(str(r["cause"]))
    tail = ("  (" + ", ".join(extra) + ")") if extra else ""
    return f"{ts}  t+{int(r['ts'] - t0):>4}s  {who:>6}: {base}{tail}"[:cols]


def render_events(
    records: list[dict], now: float, size: tuple[int, int], meta: dict, scroll: int = 0
) -> str:
    """The per-node event log: every source-stamped transition, newest at the
    bottom, with absolute (HH:MM:SS) and relative (t+Ns) times. Pure."""
    cols, rows = max(40, size[0]), max(6, size[1])
    recs = sorted(records, key=lambda r: r["ts"])
    t0 = recs[0]["ts"] if recs else now
    title = f"Events — {meta.get('run_id', '?')}   {len(recs)} transitions"[:cols]
    body_rows = rows - 3
    lines = [_event_line(r, t0, cols) for r in recs] or ["(no [event] records yet)"]
    end = max(0, len(lines) - scroll)
    visible = lines[max(0, end - body_rows) : end]
    visible += [""] * (body_rows - len(visible))
    footer = "↑/↓ PgUp/PgDn scroll   t timeline   g grid   e export   q quit"[:cols]
    return "\n".join([title, "─" * cols, *visible, footer])


def export_gantt(
    rec: TimelineRecorder,
    run_id: str,
    now: float,
    *,
    out_dir: str = ".",
    cfg: OrchestratorConfig | None = None,
    local_only: bool = False,
    records: list[dict] | None = None,
) -> list[str]:
    """Render the Gantt to a PNG (and, when ``records`` are given, a companion
    ``events.txt`` of every source-stamped transition) and return where they
    landed — local paths, plus s3:// URIs when a bucket is configured. Lazy
    matplotlib import — same Agg backend the run profile uses."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    t0 = rec.t0 if rec.t0 is not None else now
    rowkeys = sorted(rec.samples, key=_row_parts)
    deg = rec.degraded(now)
    # Two stacked panels sharing the time axis: the per-(node,attempt) Gantt on
    # top, the world-size (membership) staircase below.
    fig, (ax, wax) = plt.subplots(
        2,
        1,
        figsize=(12, 1.6 + 0.7 * max(1, len(rowkeys))),
        gridspec_kw={"height_ratios": [max(1, len(rowkeys)), 1.1], "hspace": 0.32},
        sharex=True,
    )
    for i, row in enumerate(rowkeys):
        runs = rec.runs(row, now)
        ax.broken_barh(
            [(s, d) for s, d, _lbl in runs],
            (i * 10 + 2, 6),
            facecolors=[_GANTT_COLORS.get(lbl, "#ccc") for _s, _d, lbl in runs],
        )
        for s, d, lbl in runs:  # segment-length labels (PNG only)
            if d >= 0.04 * max(1.0, now - t0):  # skip slivers too thin for text
                cap = f"{int(d)}s"
                if lbl == "wasted" and rec.wasted.get(row):
                    cap = f"{rec.wasted[row]} steps lost"  # exact rolled-back count
                ax.text(
                    s + d / 2, i * 10 + 5, cap, ha="center", va="center", fontsize=8, color="white"
                )
        for rw, rn in rec.realized:  # realized the world changed — a point in time
            if rn == row:
                ax.plot(rw - t0, i * 10 + 5, marker="D", color="#B279A2", ms=8, mec="white", mew=1)
        for lw, ln in rec.leaders:  # became rank-0 leader
            if rec.leader_row(ln, lw) == row:
                ax.plot(
                    lw - t0, i * 10 + 5, marker="*", color="#F2C800", ms=16, mec="#7a6000", mew=0.8
                )
        for kw, kn in rec.kills:
            if kn == row:
                ax.plot(kw - t0, i * 10 + 5, marker="x", color="#E45756", ms=11, mew=3)
    ax.set_yticks([i * 10 + 5 for i in range(len(rowkeys))])
    ax.set_yticklabels([_row_label(r).replace("n", "node", 1) for r in rowkeys])
    ax.set_ylim(0, max(1, len(rowkeys)) * 10)
    ax.set_title(f"Run timeline — {run_id}")
    _used = {lbl for r in rowkeys for _s, _d, lbl in rec.runs(r, now)}
    marks = [
        Line2D([], [], marker="x", color="#E45756", ls="", ms=8, mew=2, label="killed/gone"),
    ]
    if rec.realized:
        marks.append(
            Line2D(
                [],
                [],
                marker="D",
                color=_REALIZED_COLOR,
                ls="",
                ms=7,
                mec="white",
                label="realized world change",
            )
        )
    if rec.leaders:
        marks.append(
            Line2D(
                [],
                [],
                marker="*",
                color="#F2C800",
                ls="",
                ms=12,
                mec="#7a6000",
                label="became leader (rank 0)",
            )
        )
    ax.legend(
        handles=[
            mpatches.Patch(color=c, label=lbl) for lbl, c in _GANTT_COLORS.items() if lbl in _used
        ]
        + marks,
        loc="upper right",
        fontsize=8,
    )

    # World-size panel: the membership staircase, with the full line dashed and
    # every below-full (shrunk) window shaded — the downtime due to world change.
    wruns = rec.world_runs(now)
    xs, ys = [0.0], [wruns[0][2] if wruns else 0]
    for s, d, ws in wruns:
        xs += [s, s + d]
        ys += [ws, ws]
    wax.step(xs, ys, where="post", color="#4C78A8", lw=2)
    if rec.full:
        wax.axhline(rec.full, ls="--", lw=1, color="#888")
    for s, d, _mn in deg["windows"]:
        wax.axvspan(s, s + d, color="#E45756", alpha=0.15)
    wax.set_ylim(0, rec.full + 1)
    wax.set_yticks(range(rec.full + 1))
    wax.set_ylabel("world")
    wax.set_xlabel("seconds")
    wax.set_title(
        f"world {deg['current']}/{deg['full']}   "
        f"degraded {int(deg['total'])}s across {len(deg['windows'])} dip(s)",
        fontsize=9,
    )

    path = os.path.abspath(os.path.join(out_dir, f"{run_id}-timeline.png"))
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    local = [path]

    if records:  # a shareable, plain-text record of every source-stamped event
        txt = os.path.abspath(os.path.join(out_dir, f"{run_id}-events.txt"))
        recs = sorted(records, key=lambda r: r["ts"])
        t0 = recs[0]["ts"]
        with open(txt, "w") as f:
            f.write(f"Run events — {run_id}\n")
            for r in recs:
                stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"]))
                f.write(f"{stamp}  {_event_line(r, t0, 200).split('  ', 1)[1]}\n")
        local.append(txt)

    where = list(local)
    if cfg is not None and not local_only and getattr(cfg, "bucket", ""):
        for p in local:
            name = os.path.basename(p)
            uri = f"s3://{cfg.bucket}/{cfg.run_prefix}/{run_id}/{name}"
            try:
                s3_store.put_file(p, uri)
                where.append(uri)
            except Exception as exc:  # noqa: BLE001 — S3 optional; locals already written
                where.append(f"(s3 upload failed for {name}: {exc})")
    return where


def decode_key(raw: bytes) -> str | None:
    """Raw stdin bytes (cbreak mode) -> a semantic key, or None if unmapped."""
    if raw in (b"q", b"Q", b"\x03"):  # Ctrl-C arrives as \x03 in cbreak
        return "quit"
    if raw in (b"g", b"G"):
        return "grid"
    if raw in (b"t", b"T"):
        return "timeline"
    if raw in (b"v", b"V"):
        return "events"
    if raw in (b"e", b"E"):
        return "export"
    if raw.isdigit():
        return raw.decode()
    return {
        b"\x1b[C": "right",
        b"\x1b[D": "left",
        b"\x1b[A": "up",
        b"\x1b[B": "down",
        b"\x1b[5~": "pgup",
        b"\x1b[6~": "pgdn",
        b"n": "right",
        b"p": "left",
    }.get(raw)


# --------------------------------------------------------------------------- #
# Terminal shell
# --------------------------------------------------------------------------- #
def _read_key(fd: int, timeout: float) -> bytes | None:
    if not select.select([fd], [], [], timeout)[0]:
        return None
    raw = os.read(fd, 1)
    if raw == b"\x1b":  # collect the rest of an escape sequence, if any
        while len(raw) < 8 and select.select([fd], [], [], 0.02)[0]:
            raw += os.read(fd, 1)
    return raw


def _paint(frame: str) -> None:
    """Redraw without the flicker/tearing of a full-screen wipe: home the cursor
    and clear each line to its end in place (``\\x1b[K``), then clear anything
    below (``\\x1b[J``, for a shrink). Because the cursor always starts at row 1,
    the tab bar is pinned steady instead of scrolling/lagging as it did under a
    per-frame ``\\x1b[2J``."""
    out = "\x1b[H" + frame.replace("\n", "\x1b[K\r\n") + "\x1b[K\x1b[J"
    sys.stdout.write(out)
    sys.stdout.flush()


def _tick(run_uri: str, tabs: dict[tuple[int, int], Tab], now: float) -> tuple[dict, list]:
    """One discovery pass: merge tab infos, force-poll the final bytes of
    anything that just died."""
    meta, infos = discover(run_uri, now=now)
    events = merge(tabs, infos)
    for kind, t in events:
        if kind == "dead":
            poll(t, force=True)
    return meta, events


def _ordered(tabs: dict[tuple[int, int], Tab]) -> list[Tab]:
    return [tabs[k] for k in sorted(tabs)]  # ORCH = -1 sorts first


def _index_of(order: list[Tab], sel_key: tuple[int, int] | None) -> int:
    return next((i for i, t in enumerate(order) if (t.node, t.attempt) == sel_key), 0)


def _grid_tabs(tabs: dict[tuple[int, int], Tab]) -> list[Tab]:
    """The panes to show in grid mode: the orchestrator, then one pane per node
    index (its newest attempt). A killed-and-replaced node collapses to its live
    replacement pane; a killed-not-replaced node stays as its dead pane — so the
    grid always reflects current membership without unbounded growth."""
    newest: dict[int, Tab] = {}
    for t in _ordered(tabs):
        cur = newest.get(t.node)
        if cur is None or t.attempt >= cur.attempt:
            newest[t.node] = t
    return [newest[k] for k in sorted(newest)]  # ORCH first


def run_logs(
    cfg: OrchestratorConfig,
    run_id: str,
    *,
    uri: str | None = None,
    interval: float | None = None,
    node: int | None = None,
    grid: bool = False,
    plain: bool = False,
) -> None:
    """The ``logs`` subcommand: attach the dashboard to a run (live or done).
    ``grid`` opens the all-panes-at-once view; ``plain`` (or any non-TTY stdout)
    forces the append-tail of one node — the mode a tmux pane wants."""
    if uri:
        run_uri = uri.rstrip("/")
    else:
        cfg.require_bucket()
        run_uri = cfg.run_uri(run_id)
    interval = float(interval if interval is not None else cfg.log_stream_seconds)
    if plain or not (sys.stdin.isatty() and sys.stdout.isatty()):
        _run_plain(run_uri, interval, node)
        return

    tabs: dict[tuple[int, int], Tab] = {}
    meta: dict = {"run_id": run_id}
    sel_key: tuple[int, int] | None = None
    want_node = node  # select this node's tab once it appears (None = first real node)
    scroll = 0
    mode = "grid" if grid else "single"
    timeline = TimelineRecorder()  # inferred fallback (runs without [event] lines)
    records: list[dict] = []  # source-stamped events parsed from the logs
    export_note = ""
    fd = sys.stdin.fileno()

    def active_timeline(at: float) -> TimelineRecorder:
        """Prefer the event-sourced timeline (exact, source-stamped); fall back to
        the inferred one for older runs that emit no [event] lines."""
        evt = TimelineRecorder.from_events(records, at)
        return evt if (evt.samples or evt.world) else timeline

    import termios
    import tty

    old_attrs = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    sys.stdout.write("\x1b[?25l\x1b[2J")  # hide cursor + one full clear at startup
    try:
        next_tick = 0.0
        while True:
            now = time.time()
            dirty = False
            if now >= next_tick:
                meta, _events = _tick(run_uri, tabs, now)
                order = _ordered(tabs)
                # Selection: jump to --node's newest attempt the moment it
                # appears; otherwise default to the first real node tab (orch
                # as a last resort) until the user picks one.
                if order:
                    if want_node is not None:
                        pick = max(
                            (t for t in order if t.node == want_node),
                            key=lambda t: t.attempt,
                            default=None,
                        )
                        if pick is not None:
                            sel_key, scroll, want_node = (pick.node, pick.attempt), 0, None
                    if sel_key not in tabs:
                        pick = next((t for t in order if t.node != ORCH), order[0])
                        sel_key = (pick.node, pick.attempt)
                # timeline/events need every node's events (embedded in its log),
                # so poll ALL tabs there; grid polls its visible panes; single view
                # only the selected tab. Dead panes get one forced fetch.
                if mode in ("timeline", "events"):
                    to_poll = list(tabs.values())
                elif mode == "grid":
                    to_poll = _grid_tabs(tabs)
                else:
                    to_poll = [tabs.get(sel_key)]
                for t in to_poll:
                    if t is not None:
                        poll(t, force=t.state == "dead" and t.fetched == 0)
                # Feed the timeline only status.json ticks: they carry the epoch
                # member set, so a booting node reads as provisioning (not the
                # listing fallback's fresh-log = "alive" = falsely "training").
                if meta.get("source") == "status":
                    timeline.update(
                        meta.get("updated_at") or now,
                        tabs,
                        set(meta.get("members") or []),
                        epoch=meta.get("epoch"),
                        ckpt_step=meta.get("ckpt_step"),
                    )
                records = collect_events(tabs)  # re-parse the (small) event lines
                next_tick = now + interval
                dirty = True

            raw = _read_key(fd, 0.1)
            if raw is not None:
                key = decode_key(raw)
                if key == "quit":
                    return
                if key == "grid":
                    mode, scroll = ("single" if mode == "grid" else "grid"), 0
                    sys.stdout.write("\x1b[2J")  # layout changes shape — clear once
                    dirty = True
                elif key == "timeline":
                    mode = "single" if mode == "timeline" else "timeline"
                    scroll = 0
                    sys.stdout.write("\x1b[2J")
                    dirty = True
                elif key == "events":
                    mode = "single" if mode == "events" else "events"
                    scroll = 0
                    sys.stdout.write("\x1b[2J")
                    dirty = True
                elif key == "export":
                    for t in tabs.values():  # sweep every tab so the export is complete
                        poll(t, force=t.state == "dead" and t.fetched == 0)
                    records = collect_events(tabs)
                    at = meta.get("updated_at") or now
                    where = export_gantt(
                        active_timeline(at),
                        run_id,
                        at,
                        cfg=cfg,
                        local_only=bool(uri),
                        records=records,
                    )
                    export_note = "exported → " + "   ".join(where)
                    dirty = True
                elif mode == "grid":
                    panes = _grid_tabs(tabs)
                    if key and key.isdigit() and int(key) < len(panes):
                        p = panes[int(key)]  # focus a pane -> single view of it
                        sel_key, scroll, mode = (p.node, p.attempt), 0, "single"
                        sys.stdout.write("\x1b[2J")
                        dirty = True
                elif mode == "timeline":
                    pass  # timeline is a read-only overview; e/t/g/v/q handled above
                elif mode == "events":
                    n_lines = max(1, len(records))
                    page = max(1, shutil.get_terminal_size().lines - 3)
                    if key == "up":
                        scroll = min(n_lines - 1, scroll + 1)
                    elif key == "down":
                        scroll = max(0, scroll - 1)
                    elif key == "pgup":
                        scroll = min(n_lines - 1, scroll + page)
                    elif key == "pgdn":
                        scroll = max(0, scroll - page)
                    dirty = True
                else:  # single view
                    order = _ordered(tabs)
                    idx = _index_of(order, sel_key)
                    sel = order[idx] if order else None
                    page = max(1, shutil.get_terminal_size().lines - 4)
                    max_scroll = max(0, (sel.buf.count(b"\n") if sel else 0) - 1)
                    if key in ("right", "left") and order:
                        idx = (idx + (1 if key == "right" else -1)) % len(order)
                        sel_key, scroll = (order[idx].node, order[idx].attempt), 0
                    elif key and key.isdigit() and int(key) < len(order):
                        sel_key, scroll = (order[int(key)].node, order[int(key)].attempt), 0
                    elif key == "up":
                        scroll = min(max_scroll, scroll + 1)
                    elif key == "down":
                        scroll = max(0, scroll - 1)
                    elif key == "pgup":
                        scroll = min(max_scroll, scroll + page)
                    elif key == "pgdn":
                        scroll = max(0, scroll - page)
                    if sel_key in tabs:
                        t = tabs[sel_key]
                        poll(t, force=t.state == "dead" and t.fetched == 0)
                    dirty = True

            if dirty:
                size = shutil.get_terminal_size()
                sz = (size.columns, size.lines)
                if mode == "grid":
                    frame = render_grid(_grid_tabs(tabs), sz, meta, now)
                elif mode == "timeline":
                    frame = render_gantt(active_timeline(now), now, sz, meta, note=export_note)
                elif mode == "events":
                    frame = render_events(records, now, sz, meta, scroll)
                else:
                    order = _ordered(tabs)
                    idx = _index_of(order, sel_key)
                    frame = render_frame(order, idx, scroll, sz, meta, now)
                _paint(frame)
    finally:
        sys.stdout.write("\x1b[?25h\n")
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        if export_note:  # leave the export path in the scrollback after quitting
            print(f"[logs] {export_note}", file=sys.stderr)


def _run_plain(run_uri: str, interval: float, node: int | None) -> None:
    """Non-TTY fallback (pipes/CI): append-tail one node's log, with one-line
    join/death notices for every tab so transitions are still visible."""
    tabs: dict[tuple[int, int], Tab] = {}
    printed: dict[tuple[int, int], int] = {}  # buf bytes already echoed, per tab
    target = 0 if node is None else node
    while True:
        now = time.time()
        meta, evts = _tick(run_uri, tabs, now)
        for kind, t in evts:
            note = "+" if kind == "new" else "DEAD"
            print(f"[logs] {note} {t.label} ({t.state})", file=sys.stderr)
        cands = [t for t in tabs.values() if t.node == target]
        if cands:
            t = max(cands, key=lambda t: t.attempt)
            poll(t, force=t.state == "dead" and t.fetched == 0)
            # Echo from the buffer, not the poll result: the forced final poll
            # at death (in _tick) lands bytes in buf that must still print.
            k = (t.node, t.attempt)
            new = bytes(t.buf[printed.get(k, 0) :])
            if new:
                sys.stdout.write(new.decode("utf-8", errors="replace"))
                sys.stdout.flush()
                printed[k] = len(t.buf)
        if meta.get("done"):
            print("[logs] run complete", file=sys.stderr)
            return
        time.sleep(interval)
