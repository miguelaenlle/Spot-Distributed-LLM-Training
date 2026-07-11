"""Log-viewer tests — the data layer (discover/merge/poll), the pure frame
renderer, key decoding, and the s3_store read helpers it relies on. Everything
runs against a plain local directory: the same file-store trick as
test_epoch_e2e.py, so the exact code that reads S3 in production is exercised
end to end with no AWS.
"""

from __future__ import annotations

import json
import os
import time

from orchestrator import logview
from orchestrator.logview import (
    ORCH,
    Tab,
    _grid_tabs,
    decode_key,
    discover,
    merge,
    poll,
    render_frame,
    render_grid,
)
from spot_train import s3_store


# --------------------------------------------------------------------------- #
# s3_store read helpers (the viewer's storage primitives)
# --------------------------------------------------------------------------- #
def test_store_last_modified_and_list_names(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    f = d / "boot-node0.log"
    f.write_bytes(b"hello\n")
    assert s3_store.last_modified(str(f)) is not None
    assert s3_store.last_modified(str(d / "missing.log")) is None
    assert s3_store.list_names(str(d)) == ["boot-node0.log"]
    assert s3_store.list_names(str(tmp_path / "nope")) == []


def test_store_read_bytes_from_offsets(tmp_path):
    f = tmp_path / "a.log"
    f.write_bytes(b"one\n")
    assert s3_store.read_bytes_from(str(f), 0) == b"one\n"
    assert s3_store.read_bytes_from(str(f), 4) == b""  # at EOF
    assert s3_store.read_bytes_from(str(f), 99) == b""  # past EOF
    with open(f, "ab") as fh:
        fh.write(b"two\n")
    assert s3_store.read_bytes_from(str(f), 4) == b"two\n"  # only the new tail
    assert s3_store.read_bytes_from(str(tmp_path / "missing"), 0) == b""


# --------------------------------------------------------------------------- #
# discover — status.json path and the listing fallback
# --------------------------------------------------------------------------- #
def _run_dir(tmp_path):
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    return run


def test_discover_from_status_doc(tmp_path):
    run = _run_dir(tmp_path)
    doc = {
        "version": 1,
        "run_id": "r",
        "updated_at": 100.0,
        "epoch": 2,
        "members": [0],
        "done": False,
        "orchestrator": {"log_key": "runs/r/logs/orchestrator.log"},
        "nodes": [
            {"node": 0, "attempt": 0, "log_key": "runs/r/logs/boot-node0.log", "state": "alive"},
            {"node": 1, "attempt": 1, "log_key": "runs/r/logs/boot-node1-r1.log", "state": "dead"},
        ],
    }
    (run / "status.json").write_text(json.dumps(doc))
    meta, infos = discover(str(run), now=100.0)
    assert meta["source"] == "status" and meta["epoch"] == 2 and meta["done"] is False
    by_key = {(i["node"], i["attempt"]): i for i in infos}
    assert by_key[(ORCH, 0)]["log_uri"] == f"{run}/logs/orchestrator.log"
    assert by_key[(0, 0)]["state"] == "alive"
    # log keys are bucket keys — the viewer maps them under ITS run base
    assert by_key[(1, 1)]["log_uri"] == f"{run}/logs/boot-node1-r1.log"
    assert by_key[(1, 1)]["state"] == "dead"


def test_discover_listing_fallback_staleness(tmp_path):
    run = _run_dir(tmp_path)
    now = time.time()
    (run / "logs" / "boot-node0.log").write_bytes(b"a")
    stale = run / "logs" / "boot-node1-r2.log"
    stale.write_bytes(b"b")
    os.utime(stale, (now - 1000, now - 1000))  # went silent long ago -> dead
    (run / "logs" / "orchestrator.log").write_bytes(b"c")
    meta, infos = discover(str(run), heartbeat_timeout_s=90.0, now=now)
    assert meta["source"] == "listing" and meta["done"] is False
    by_key = {(i["node"], i["attempt"]): i["state"] for i in infos}
    assert by_key == {(ORCH, 0): "alive", (0, 0): "alive", (1, 2): "dead"}


def test_discover_listing_done_run(tmp_path):
    run = _run_dir(tmp_path)
    (run / "logs" / "boot-node0.log").write_bytes(b"a")
    (run / "metrics.json").write_text("{}")
    meta, infos = discover(str(run), now=time.time())
    assert meta["done"] is True
    assert all(i["state"] == "dead" for i in infos)  # frozen, viewable post-mortem


# --------------------------------------------------------------------------- #
# merge — join/death transitions, exactly once, never resurrected
# --------------------------------------------------------------------------- #
def _info(node, attempt, state, uri="u"):
    return {"node": node, "attempt": attempt, "log_uri": uri, "state": state}


def test_merge_new_dead_once_and_no_resurrection():
    tabs: dict = {}
    events = merge(tabs, [_info(0, 0, "pending")])
    assert [(k, t.label) for k, t in events] == [("new", "node0")]

    assert merge(tabs, [_info(0, 0, "alive")]) == []  # pending -> alive: no event
    assert tabs[(0, 0)].state == "alive"

    events = merge(tabs, [_info(0, 0, "dead")])
    assert [(k, t.label) for k, t in events] == [("dead", "node0")]
    assert merge(tabs, [_info(0, 0, "dead")]) == []  # dead again: silent
    assert merge(tabs, [_info(0, 0, "alive")]) == []  # never resurrects
    assert tabs[(0, 0)].state == "dead"

    # A replacement is a NEW (node, attempt) — its own tab, its own event.
    events = merge(tabs, [_info(0, 0, "dead"), _info(0, 1, "alive")])
    assert [(k, t.label) for k, t in events] == [("new", "node0·r1")]


# --------------------------------------------------------------------------- #
# poll — incremental tail; dead tabs frozen except the forced final/first fetch
# --------------------------------------------------------------------------- #
def test_poll_incremental_and_dead_freeze(tmp_path):
    f = tmp_path / "n0.log"
    f.write_bytes(b"one\n")
    tab = Tab(node=0, attempt=0, log_uri=str(f), state="alive")
    assert poll(tab) == b"one\n"
    with open(f, "ab") as fh:
        fh.write(b"two\n")
    assert poll(tab) == b"two\n"  # only the appended bytes
    assert poll(tab) == b""
    tab.state = "dead"
    with open(f, "ab") as fh:
        fh.write(b"traceback\n")
    assert poll(tab) == b""  # frozen: no new logs for a dead node
    assert poll(tab, force=True) == b"traceback\n"  # the one-time final grab
    assert bytes(tab.buf) == b"one\ntwo\ntraceback\n"


# --------------------------------------------------------------------------- #
# render_frame — badges, selection highlight, scroll indicator
# --------------------------------------------------------------------------- #
_META = {
    "run_id": "r",
    "epoch": 3,
    "members": [0, 2],
    "updated_at": 98.0,
    "done": False,
    "source": "status",
}


def _tabs():
    tabs = [
        Tab(node=ORCH, attempt=0, log_uri="u", state="alive"),
        Tab(node=0, attempt=0, log_uri="u", state="alive"),
        Tab(node=1, attempt=0, log_uri="u", state="dead"),
        Tab(node=1, attempt=1, log_uri="u", state="pending"),
    ]
    tabs[1].buf.extend(b"step 1\nstep 2\nstep 3\n")
    return tabs


def test_render_frame_badges_and_selection():
    frame = render_frame(_tabs(), 1, 0, (100, 12), _META, now=100.0)
    assert "\x1b[7m[node0]\x1b[0m" in frame  # selected: reverse video, no badge
    assert "[node1 (dead)]" in frame
    assert "[node1·r1 (joining)]" in frame
    assert "[orch]" in frame
    assert "epoch 3" in frame and "members 0,2" in frame and "supervisor 2s ago" in frame
    assert "step 3" in frame
    assert "[FOLLOW]" in frame and "RUN COMPLETE" not in frame


def test_render_frame_scroll_and_done_and_stale():
    frame = render_frame(_tabs(), 1, 2, (100, 12), _META, now=100.0)
    assert "[SCROLL -2]" in frame
    meta = {**_META, "updated_at": 40.0, "done": False}
    frame = render_frame(_tabs(), 0, 0, (100, 12), meta, now=100.0)
    assert "(stale 60s)" in frame  # the control plane itself went silent
    frame = render_frame(_tabs(), 1, 0, (100, 12), {**_META, "done": True}, now=100.0)
    assert "RUN COMPLETE" in frame


def test_grid_tabs_newest_attempt_per_node():
    tabs = {
        (ORCH, 0): Tab(ORCH, 0, "u", "alive"),
        (0, 0): Tab(0, 0, "u", "alive"),
        (1, 0): Tab(1, 0, "u", "dead"),  # killed
        (1, 1): Tab(1, 1, "u", "alive"),  # replacement -> collapses to this pane
        (2, 0): Tab(2, 0, "u", "dead"),  # killed, not replaced -> stays as dead pane
    }
    panes = _grid_tabs(tabs)
    assert [(t.node, t.attempt, t.state) for t in panes] == [
        (ORCH, 0, "alive"),
        (0, 0, "alive"),
        (1, 1, "alive"),  # newest attempt wins
        (2, 0, "dead"),  # who left, still visible
    ]


def test_render_grid_shows_all_panes_and_badges():
    tabs = _grid_tabs(
        {
            (ORCH, 0): Tab(ORCH, 0, "u", "alive"),
            (0, 0): Tab(0, 0, "u", "alive"),
            (1, 0): Tab(1, 0, "u", "dead"),
            (1, 1): Tab(1, 1, "u", "pending"),
        }
    )
    tabs[1].buf.extend(b"n0 step 42\n")  # node0's pane (orch, node0, node1·r1)
    frame = render_grid(tabs, (120, 20), _META, now=100.0)
    # Every live member shows at once, each with its own badge.
    assert "orch [LIVE]" in frame
    assert "node0 [LIVE]" in frame
    assert "node1·r1 [JOIN]" in frame  # a joiner appears as its own pane
    assert "n0 step 42" in frame  # tail rendered inside the pane
    assert "epoch 3" in frame  # header carries the run summary
    # A killed-not-replaced node stays visible as a dimmed dead pane.
    dead = _grid_tabs({(0, 0): Tab(0, 0, "u", "dead")})
    assert "node0 [DEAD]" in render_grid(dead, (80, 12), _META, now=100.0)


def test_render_grid_caps_at_eight_panes():
    tabs = _grid_tabs({(i, 0): Tab(i, 0, "u", "alive") for i in range(12)})
    frame = render_grid(tabs, (160, 24), _META, now=100.0)
    assert "node7 [LIVE]" in frame
    assert "node8 [LIVE]" not in frame  # MAX_GRID = 8


def test_render_grid_empty_is_stable():
    frame = render_grid([], (80, 12), {"run_id": "r"}, now=0.0)
    assert frame.count("\n") == 11  # header + 10 blank body rows + footer == rows


def test_decode_key():
    assert decode_key(b"g") == "grid"
    assert decode_key(b"\x1b[C") == "right"
    assert decode_key(b"\x1b[D") == "left"
    assert decode_key(b"\x1b[A") == "up"
    assert decode_key(b"\x1b[B") == "down"
    assert decode_key(b"\x1b[5~") == "pgup"
    assert decode_key(b"\x1b[6~") == "pgdn"
    assert decode_key(b"3") == "3"
    assert decode_key(b"q") == "quit"
    assert decode_key(b"\x03") == "quit"  # Ctrl-C in cbreak mode
    assert decode_key(b"z") is None


# --------------------------------------------------------------------------- #
# The elastic story end to end (data layer): alive -> kill -> replacement,
# driven through the same _tick the terminal shell runs, over a local dir —
# including the supervisor-side writer via the real status_doc.
# --------------------------------------------------------------------------- #
def test_elastic_kill_and_replace_sequence(tmp_path):
    from orchestrator.supervisor import NodeObs, Observation, Policy, status_doc

    run = _run_dir(tmp_path)
    policy = Policy(replace_on_loss=True, recovery_timeout_s=600)

    def obs(*nodes):
        return Observation(
            node_count=2,
            nodes=tuple(nodes),
            epoch=1,
            members=frozenset({0, 1}),
            metrics_exists=False,
            no_progress_s=None,
        )

    def write(doc):
        (run / "status.json").write_text(json.dumps(doc))

    def sdoc(o, logs, prev, now):
        return status_doc(
            "r",
            o,
            policy,
            epoch=1,
            members=frozenset({0, 1}),
            ips={},
            node_ids={0: "i-0", 1: "i-1"},
            logs=logs,
            orch_log_key="runs/r/logs/orchestrator.log",
            prev=prev,
            now=now,
        )

    n0 = run / "logs" / "boot-node0.log"
    n1 = run / "logs" / "boot-node1.log"
    n0.write_bytes(b"n0 step 1\n")
    n1.write_bytes(b"n1 step 1\n")
    logs = {
        0: {"key": "runs/r/logs/boot-node0.log", "attempt": 0},
        1: {"key": "runs/r/logs/boot-node1.log", "attempt": 0},
    }

    # Tick 1: both alive -> two node tabs (+ orch) appear.
    doc = sdoc(obs(NodeObs(0, "running", True), NodeObs(1, "running", True)), logs, None, 100.0)
    write(doc)
    tabs: dict = {}
    meta, events = logview._tick(str(run), tabs, 100.0)
    assert {k for k, _ in events} == {"new"} and len(events) == 3
    poll(tabs[(1, 0)])
    assert bytes(tabs[(1, 0)].buf) == b"n1 step 1\n"

    # Tick 2: node 1 reclaimed. Its final bytes land, THEN it's observed dead —
    # the dead event must still capture them (forced final poll), after which
    # the tab is frozen.
    with open(n1, "ab") as fh:
        fh.write(b"n1 CUDA error\n")
    doc = sdoc(obs(NodeObs(0, "running", True), NodeObs(1, "terminated", True)), logs, doc, 103.0)
    write(doc)
    meta, events = logview._tick(str(run), tabs, 103.0)
    assert [(k, t.label) for k, t in events] == [("dead", "node1")]
    assert bytes(tabs[(1, 0)].buf).endswith(b"n1 CUDA error\n")
    with open(n1, "ab") as fh:
        fh.write(b"never seen\n")
    assert poll(tabs[(1, 0)]) == b""  # frozen for good

    # Tick 3: the replacement registers as attempt 1 -> a NEW tab joins while
    # node0 keeps streaming; the dead tab stays.
    (run / "logs" / "boot-node1-r1.log").write_bytes(b"n1r1 boot\n")
    logs = {**logs, 1: {"key": "runs/r/logs/boot-node1-r1.log", "attempt": 1}}
    doc = sdoc(obs(NodeObs(0, "running", True), NodeObs(1, "running", True)), logs, doc, 106.0)
    write(doc)
    meta, events = logview._tick(str(run), tabs, 106.0)
    assert [(k, t.label) for k, t in events] == [("new", "node1·r1")]
    assert poll(tabs[(1, 1)]) == b"n1r1 boot\n"
    assert tabs[(1, 0)].state == "dead" and tabs[(1, 1)].state == "alive"
    assert meta["source"] == "status"
