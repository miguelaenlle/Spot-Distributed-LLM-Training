"""Node lifecycle events — one source-stamped record per state transition.

The Gantt/timeline is built by event-sourcing, not by polling: each actor emits
a timestamped record at the *instant* it changes state, so the recorded time is
derived straight from the source with no inference and no poll latency. Clocks
are assumed synchronized (NTP), so ``ts`` values are directly comparable across
every node and the orchestrator.

Transport is the log pipeline that already exists: an event is one line

    [event] {"ts": 1783746952.417, "node": 0, "state": "training", ...}

printed to stderr, which lands in the box's boot log and is synced to S3 every
few seconds. The 3s upload/parse latency affects only when a reader *sees* the
event, never its ``ts``. No new S3 objects; the viewer parses these lines out of
the per-node logs (and the orchestrator's own log) it already downloads.

Stdlib-only so the trainer (``spot_train``), the sidecar (``orchestrator``), and
the supervisor can all import it without pulling in heavier deps.

States (per node): ``provisioning`` (booting / restoring / re-rendezvousing, not
yet training), ``training`` (in the step loop), ``stalled`` (alive but blocked on
a down peer — the pre-NCCL-timeout hang), ``down`` / ``killed`` (box gone; the two
differ only by ``cause``). ``by`` records the emitter (trainer / sidecar / orch).
"""

from __future__ import annotations

import json
import sys
import time

PREFIX = "[event] "
STATES = ("provisioning", "reconfiguring", "training", "stalled", "down", "killed")
# Field order kept stable for readable log lines; ts/state/by always present.
_FIELDS = ("ts", "node", "attempt", "state", "epoch", "world", "leader", "step", "cause", "by")


def emit(
    state: str,
    *,
    by: str,
    node: int | None = None,
    attempt: int | None = None,
    epoch: int | None = None,
    world: int | None = None,
    leader: int | None = None,
    step: int | None = None,
    cause: str | None = None,
    ts: float | None = None,
    stream=None,
) -> dict:
    """Print one ``[event]`` line to stderr and return the record. ``ts`` defaults
    to now — pass it explicitly to stamp the true onset (e.g. a stall's start is
    the last good step's time, not when the watchdog noticed)."""
    rec = {
        "ts": time.time() if ts is None else ts,
        "node": node,
        "attempt": attempt,
        "state": state,
        "epoch": epoch,
        "world": world,
        "leader": leader,
        "step": step,
        "cause": cause,
        "by": by,
    }
    rec = {k: rec[k] for k in _FIELDS if rec.get(k) is not None}
    print(PREFIX + json.dumps(rec, separators=(",", ":")), file=stream or sys.stderr, flush=True)
    return rec


def parse(
    text: str, *, default_node: int | None = None, default_attempt: int | None = None
) -> list[dict]:
    """Pull every well-formed ``[event]`` record out of a log blob. A record that
    omits ``node``/``attempt`` is attributed to ``default_node``/``default_attempt``
    — the box whose log it came from (each attempt writes its own log file), so a
    replacement's events land on its own row. Malformed lines and any line without
    a numeric ``ts``/``state`` are skipped, not raised on."""
    out: list[dict] = []
    for line in text.splitlines():
        i = line.find(PREFIX)
        if i < 0:
            continue
        try:
            rec = json.loads(line[i + len(PREFIX) :])
        except ValueError:
            continue
        if not isinstance(rec, dict) or "ts" not in rec or "state" not in rec:
            continue
        if not isinstance(rec["ts"], (int | float)):
            continue
        rec.setdefault("node", default_node)
        if default_attempt is not None:
            rec.setdefault("attempt", default_attempt)
        out.append(rec)
    return out
