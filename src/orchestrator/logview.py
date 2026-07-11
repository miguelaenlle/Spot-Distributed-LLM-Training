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

from spot_train import s3_store

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
    help_txt = "←/→ node   ↑/↓ PgUp/PgDn scroll   g grid   0-9 jump   q quit"
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
    footer = ("grid — 0-9 focus pane   g single-view   q quit")[:cols]
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


def decode_key(raw: bytes) -> str | None:
    """Raw stdin bytes (cbreak mode) -> a semantic key, or None if unmapped."""
    if raw in (b"q", b"Q", b"\x03"):  # Ctrl-C arrives as \x03 in cbreak
        return "quit"
    if raw in (b"g", b"G"):
        return "grid"
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
    fd = sys.stdin.fileno()
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
                # Grid polls EVERY visible pane; single view only the selected
                # tab. Dead panes get one forced fetch so their final bytes show.
                to_poll = _grid_tabs(tabs) if mode == "grid" else [tabs.get(sel_key)]
                for t in to_poll:
                    if t is not None:
                        poll(t, force=t.state == "dead" and t.fetched == 0)
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
                elif mode == "grid":
                    panes = _grid_tabs(tabs)
                    if key and key.isdigit() and int(key) < len(panes):
                        p = panes[int(key)]  # focus a pane -> single view of it
                        sel_key, scroll, mode = (p.node, p.attempt), 0, "single"
                        sys.stdout.write("\x1b[2J")
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
                if mode == "grid":
                    frame = render_grid(_grid_tabs(tabs), (size.columns, size.lines), meta, now)
                else:
                    order = _ordered(tabs)
                    idx = _index_of(order, sel_key)
                    frame = render_frame(order, idx, scroll, (size.columns, size.lines), meta, now)
                _paint(frame)
    finally:
        sys.stdout.write("\x1b[?25h\n")
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def _run_plain(run_uri: str, interval: float, node: int | None) -> None:
    """Non-TTY fallback (pipes/CI): append-tail one node's log, with one-line
    join/death notices for every tab so transitions are still visible."""
    tabs: dict[tuple[int, int], Tab] = {}
    printed: dict[tuple[int, int], int] = {}  # buf bytes already echoed, per tab
    target = 0 if node is None else node
    while True:
        now = time.time()
        meta, events = _tick(run_uri, tabs, now)
        for kind, t in events:
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
