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
    TimelineRecorder,
    _grid_tabs,
    collect_events,
    decode_key,
    discover,
    export_gantt,
    merge,
    poll,
    render_events,
    render_frame,
    render_gantt,
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


# --------------------------------------------------------------------------- #
# Timeline (Gantt): recorder, renderer, export
# --------------------------------------------------------------------------- #
def _tl_tabs(states):  # {node: state} -> tabs dict at their newest attempt
    return {(n, 0): Tab(n, 0, "u", s) for n, s in states.items()}


# A realistic 2-node preempt seen through status.json ticks: both boot (members
# empty), epoch 1 admits [0,1], node1 killed -> members [0], replacement boots
# then rejoins -> members [0,1] again. (w, tabs, members) per tick.
def _seed_preempt(rec):
    rec.update(0.0, _tl_tabs({0: "pending", 1: "pending"}), set())  # booting: not members
    rec.update(10.0, _tl_tabs({0: "alive", 1: "alive"}), {0, 1})  # epoch 1: both train
    rec.update(20.0, {(0, 0): Tab(0, 0, "u", "alive"), (1, 0): Tab(1, 0, "u", "dead")}, {0})
    rec.update(  # replacement booting (attempt 1 pending), still world 1
        30.0,
        {
            (0, 0): Tab(0, 0, "u", "alive"),
            (1, 0): Tab(1, 0, "u", "dead"),
            (1, 1): Tab(1, 1, "u", "pending"),
        },
        {0},
    )
    rec.update(40.0, {(0, 0): Tab(0, 0, "u", "alive"), (1, 1): Tab(1, 1, "u", "alive")}, {0, 1})


def test_timeline_labels_by_membership_never_trains_before_admitted():
    # The bug this guards: a booting node's fresh log made the listing fallback
    # call it "alive", drawing green (training) BEFORE its blue provisioning.
    # Membership-driven labels make a node "train" only once it's an epoch member.
    rec = TimelineRecorder()
    _seed_preempt(rec)
    assert rec.t0 == 0.0
    # node0 provisions (not a member yet) THEN trains — never the reverse.
    assert [lbl for _w, lbl in rec.samples[0]] == ["prov", "train", "train", "train", "train"]
    assert [lbl for _w, lbl in rec.samples[1]] == ["prov", "train", "down", "prov", "train"]
    assert rec.kills == [(20.0, 1)]  # the one alive->down transition
    # ORCH is never a Gantt row
    rec.update(50.0, {(ORCH, 0): Tab(ORCH, 0, "u", "alive")}, {0, 1})
    assert ORCH not in rec.samples


def test_timeline_survivor_restart_on_epoch_change():
    # node0 survives node1's kill, but its torchrun crashes on the NCCL abort and
    # re-rendezvouses at the smaller world — a "restart" span until checkpoint
    # progress resumes. Same on the grow (rejoin). Driven by epoch + ckpt_step.
    rec = TimelineRecorder()
    both = _tl_tabs({0: "alive", 1: "alive"})
    dead1 = {(0, 0): Tab(0, 0, "u", "alive"), (1, 0): Tab(1, 0, "u", "dead")}
    rejoined = {(0, 0): Tab(0, 0, "u", "alive"), (1, 1): Tab(1, 1, "u", "alive")}
    # epoch 1 world 2, ckpt climbing
    rec.update(0.0, both, {0, 1}, epoch=1, ckpt_step=10)
    rec.update(10.0, both, {0, 1}, epoch=1, ckpt_step=20)
    # epoch 2: node1 gone, world 1 — node0 is a survivor -> restart until ckpt moves
    rec.update(20.0, dead1, {0}, epoch=2, ckpt_step=20)
    rec.update(30.0, dead1, {0}, epoch=2, ckpt_step=20)  # still restarting (no progress)
    rec.update(40.0, dead1, {0}, epoch=2, ckpt_step=25)  # ckpt advanced -> resumed
    # epoch 3: replacement rejoins, world 2 — node0 restarts again
    rec.update(50.0, rejoined, {0, 1}, epoch=3, ckpt_step=25)
    rec.update(60.0, rejoined, {0, 1}, epoch=3, ckpt_step=30)  # resumed

    assert [lbl for _w, lbl in rec.samples[0]] == [
        "train",
        "train",  # epoch 1
        "restart",
        "restart",  # shrink: re-rendezvous at world 1
        "train",  # resumed (ckpt advanced)
        "restart",  # grow: re-rendezvous at world 2
        "train",  # resumed
    ]
    runs = rec.runs(0, now=70.0)
    assert "restart" in [lbl for _s, _d, lbl in runs]


def test_timeline_world_curve_and_degraded():
    rec = TimelineRecorder()
    _seed_preempt(rec)
    # World size = |members|: 0 (booting) -> 2 -> 1 (shrunk) -> 2 (regrown).
    assert rec.full == 2
    assert rec.world_runs(now=50.0) == [
        (0.0, 10.0, 0),
        (10.0, 10.0, 2),
        (20.0, 20.0, 1),
        (40.0, 10.0, 2),
    ]
    deg = rec.degraded(now=50.0)
    assert deg["full"] == 2 and deg["current"] == 2
    # Only the POST-startup shrink counts: the 0-10s boot ramp (ws0) is
    # provisioning, not downtime-due-to-world-change; the 20-40s dip to ws1 is.
    assert deg["total"] == 20.0 and len(deg["windows"]) == 1
    assert deg["windows"][0] == (20.0, 20.0, 1)  # 20s shrunk to world 1


def test_timeline_recorder_runs_compresses_spans():
    rec = TimelineRecorder()
    seq = [
        (0.0, "pending", set()),
        (10.0, "alive", {0}),
        (20.0, "alive", {0}),
        (30.0, "dead", set()),
    ]
    for w, s, m in seq:
        rec.update(w, _tl_tabs({0: s}), m)
    assert rec.runs(0, now=40.0) == [
        (0.0, 10.0, "prov"),
        (10.0, 20.0, "train"),
        (30.0, 10.0, "down"),
    ]


def test_render_gantt_rows_world_strip_and_summary():
    rec = TimelineRecorder()
    _seed_preempt(rec)
    frame = render_gantt(rec, now=50.0, size=(80, 14), meta={"run_id": "mn-1"})
    assert "Run timeline — mn-1" in frame and "elapsed 50s" in frame
    assert " n0 │" in frame and " n1 │" in frame  # one row per node
    assert " ws │" in frame  # world-size strip under the gantt
    assert "▓" in frame and "·" in frame and "✗" in frame  # train, down, kill glyphs
    assert "world 2/2" in frame and "degraded 20s" in frame  # post-startup downtime
    assert frame.count("\n") == 13  # exactly fills `rows`


def test_render_gantt_shows_export_note():
    rec = TimelineRecorder()
    rec.update(0.0, _tl_tabs({0: "alive"}), {0})
    frame = render_gantt(
        rec, now=5.0, size=(80, 10), meta={"run_id": "r"}, note="exported → /tmp/r.png"
    )
    assert "exported → /tmp/r.png" in frame


def test_export_gantt_writes_png(tmp_path):
    rec = TimelineRecorder()
    _seed_preempt(rec)
    where = export_gantt(rec, "mn-1", now=50.0, out_dir=str(tmp_path), local_only=True)
    assert len(where) == 1
    png = where[0]
    assert png.endswith("mn-1-timeline.png") and os.path.exists(png)
    assert os.path.getsize(png) > 1000  # a real PNG, not an empty stub


# --------------------------------------------------------------------------- #
# Event-sourced timeline: from_events reducer, events view, collect, export
# --------------------------------------------------------------------------- #
# A realistic 2-node preempt as SOURCE-STAMPED events (what the sidecar/trainer/
# orchestrator emit): both boot+train at world 2, node1 killed, node0 stalls then
# re-rendezvouses and resumes at world 1, replacement boots+joins at world 2.
def _ev(state, ts, **kw):
    return {"ts": float(ts), "state": state, **kw}


_EVENTS = [
    _ev("epoch", 40, world=2, by="orch"),
    _ev("provisioning", 0, node=0, by="sidecar", cause="boot"),
    _ev("provisioning", 2, node=1, by="sidecar", cause="boot"),
    _ev("training", 40, node=0, world=2, step=0, by="trainer"),
    _ev("training", 40, node=1, world=2, step=0, by="trainer"),
    _ev("killed", 152, node=1, by="orch", cause="scheduled-kill"),
    _ev("stalled", 150, node=0, world=2, step=60, by="trainer", cause="peer-stall"),
    _ev("reconfiguring", 156, node=0, world=1, by="sidecar", cause="epoch 1->2"),
    _ev("epoch", 154, world=1, by="orch"),
    _ev("provisioning", 158, node=0, world=1, by="sidecar", cause="torchrun-exit:1"),
    # resumes from step 50 — the 10 steps trained after the last checkpoint (up to
    # the crash at step 60) are lost and re-done ("wasted").
    _ev("training", 176, node=0, world=1, step=50, by="trainer", cause="resumed"),
    _ev("provisioning", 176, node=1, attempt=1, by="sidecar", cause="boot"),
    _ev("epoch", 200, world=2, by="orch"),
    _ev("training", 200, node=1, attempt=1, world=2, step=66, by="trainer"),
]


def test_from_events_rows_per_attempt_and_wasted():
    rec = TimelineRecorder.from_events(_EVENTS, now=238.0)
    assert rec.t0 == 0.0 and rec.full == 2
    # Rows are keyed by (node, attempt): the killed original and its replacement
    # are DISTINCT rows.
    assert set(rec.samples) == {(0, 0), (1, 0), (1, 1)}
    # node0 survives: train -> STALLED -> provisioning -> WASTED (re-doing
    # rolled-back steps) -> training. "realized" is an instant MARKER, not a
    # segment (teardown->relaunch is same-tick), so it's not in the spans.
    assert [lbl for _t, lbl in rec.samples[(0, 0)]] == [
        "prov",
        "train",
        "stalled",
        "prov",
        "wasted",
        "train",
    ]
    assert (156.0, (0, 0)) in rec.realized  # realized-world-change marker on node0
    assert rec.samples[(1, 0)][-1][1] == "down"  # original node1 killed
    assert [lbl for _t, lbl in rec.samples[(1, 1)]] == ["prov", "train"]  # replacement
    assert rec.wasted == {(0, 0): 10}  # exactly the 10 rolled-back steps
    assert (152.0, (1, 0)) in rec.kills  # kill attaches to node1·r0
    # stalled onset stamped at the last good step (t=150), not detection.
    stalled = [(s, d) for s, d, lbl in rec.runs((0, 0), 238.0) if lbl == "stalled"][0]
    assert stalled[0] == 150.0
    # World staircase from epoch events (group-level): 2 -> 1 -> 2.
    assert rec.world_runs(238.0) == [(40.0, 114.0, 2), (154.0, 46.0, 1), (200.0, 38.0, 2)]


def test_from_events_empty_falls_back():
    # No usable events -> empty recorder, so run_logs uses the inferred timeline.
    assert TimelineRecorder.from_events([], now=1.0).samples == {}
    assert TimelineRecorder.from_events([{"ts": 1.0, "state": "noise"}], now=1.0).samples == {}


# Leadership (rank-0) handovers carried on the epoch events.
_LEADER_EVENTS = [
    _ev("provisioning", 0, node=0, by="sidecar"),
    _ev("provisioning", 0, node=1, by="sidecar"),
    _ev("epoch", 10, world=2, leader=0, by="orch"),  # node0 is rank-0 leader
    _ev("training", 10, node=0, world=2, step=0, by="trainer"),
    _ev("training", 10, node=1, world=2, step=0, by="trainer"),
    _ev("killed", 100, node=0, by="orch", cause="scheduled-kill"),  # the master dies
    _ev("epoch", 105, world=1, leader=1, by="orch"),  # node1 takes over
    _ev("training", 110, node=1, world=1, step=30, by="trainer"),
    _ev("provisioning", 130, node=0, attempt=1, by="sidecar"),  # replacement
    _ev("epoch", 150, world=2, leader=1, by="orch"),  # grow back — leader STAYS node1
    _ev("training", 155, node=1, world=2, step=40, by="trainer"),
    _ev("training", 155, node=0, attempt=1, world=2, step=40, by="trainer"),
]


def test_from_events_leader_handovers_and_current():
    rec = TimelineRecorder.from_events(_LEADER_EVENTS, now=180.0)
    # Deduped to the moments leadership actually moves; sticky across the grow.
    assert rec.leaders == [(10.0, 0), (105.0, 1)]
    assert rec.current_leader(now=50.0) == 0  # before the handover
    assert rec.current_leader(now=180.0) == 1  # after node0 died, node1 leads
    assert rec.leader_row(1, 105.0) == (1, 0)  # marker lands on node1's live box


def test_render_gantt_shows_leader():
    rec = TimelineRecorder.from_events(_LEADER_EVENTS, now=180.0)
    frame = render_gantt(rec, now=180.0, size=(90, 14), meta={"run_id": "mn"})
    assert "★" in frame  # the became-leader marker on a row
    assert "leader node1" in frame  # current leader called out in the summary


def test_render_gantt_from_events_shows_stalled_and_wasted():
    rec = TimelineRecorder.from_events(_EVENTS, now=238.0)
    frame = render_gantt(rec, now=238.0, size=(100, 16), meta={"run_id": "mn"})
    assert "n1·r1" in frame  # the replacement has its own row
    assert "▚" in frame  # stalled glyph on node0's row
    assert "▨" in frame  # wasted glyph on node0's row
    assert "◆" in frame  # realized-world-change marker on node0's row
    assert "world 2/2" in frame and "wasted 10 steps" in frame


def test_render_events_lists_stamped_transitions():
    frame = render_events(_EVENTS, now=238.0, size=(100, 22), meta={"run_id": "mn"})
    assert "14 transitions" in frame
    assert "node0: training" in frame
    assert "node1: KILLED" in frame and "scheduled-kill" in frame
    assert "node1·r1: provisioning" in frame  # replacement distinct in the log too
    assert "STALLED" in frame and "REALIZED world changed" in frame
    assert "orch: epoch published" in frame
    assert "t+" in frame  # relative offsets present


def test_collect_events_attributes_node_and_attempt():
    tabs = {
        (ORCH, 0): Tab(ORCH, 0, "u", "alive"),
        (0, 0): Tab(0, 0, "u", "alive"),
        (1, 1): Tab(1, 1, "u", "alive"),  # a replacement's log
    }
    tabs[(ORCH, 0)].buf.extend(b'[event] {"ts": 154.0, "state": "epoch", "world": 1}\n')
    # records missing node/attempt are attributed to the tab they came from
    tabs[(0, 0)].buf.extend(b'step 5: loss 2.0\n[event] {"ts": 40.0, "state": "training"}\n')
    tabs[(1, 1)].buf.extend(b'[event] {"ts": 176.0, "state": "training", "world": 2}\n')
    recs = collect_events(tabs)
    by = {(r.get("node"), r.get("attempt"), r["state"]) for r in recs}
    assert (None, None, "epoch") in by  # orch epoch, no node/attempt
    assert (0, 0, "training") in by  # node 0, attempt 0 (the tab's)
    assert (1, 1, "training") in by  # attributed to the replacement's row
    rec = TimelineRecorder.from_events(recs, now=200.0)
    assert (1, 1) in rec.samples  # replacement is its own row


def test_export_gantt_writes_png_and_events_txt(tmp_path):
    rec = TimelineRecorder.from_events(_EVENTS, now=238.0)
    where = export_gantt(
        rec, "mn", now=238.0, out_dir=str(tmp_path), local_only=True, records=_EVENTS
    )
    assert len(where) == 2
    png, txt = where
    assert png.endswith("mn-timeline.png") and os.path.exists(png) and os.path.getsize(png) > 1000
    assert txt.endswith("mn-events.txt")
    with open(txt) as f:
        body = f.read()
    assert "Run events — mn" in body
    assert "node1: KILLED" in body and "node0: STALLED" in body


def test_decode_key():
    assert decode_key(b"g") == "grid"
    assert decode_key(b"t") == "timeline"
    assert decode_key(b"v") == "events"
    assert decode_key(b"e") == "export"
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
