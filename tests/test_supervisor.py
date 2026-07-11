"""Epoch-supervisor tests — the pure reducer (no AWS, no clock) plus the small
storage/schema helpers. The reducer owns all membership logic, so these tables
are the specification of how the fleet reacts to loss/join/kill/stall.
"""

from __future__ import annotations

from orchestrator.config import OrchestratorConfig
from orchestrator.supervisor import (
    Done,
    LaunchReplacement,
    NodeObs,
    Observation,
    Policy,
    PublishEpoch,
    TerminateNode,
    WholeGroupRestart,
    decide,
    epoch_doc,
    status_doc,
)
from spot_train import s3_store

SHRINK = Policy(replace_on_loss=False, recovery_timeout_s=600)
PREEMPT = Policy(replace_on_loss=True, recovery_timeout_s=600)


def _node(i, state="running", registered=True, log_age=None):
    return NodeObs(node=i, aws_state=state, registered=registered, log_age_s=log_age)


def _obs(nodes, *, epoch, members, node_count=None, metrics=False, no_progress=None, due=()):
    return Observation(
        node_count=node_count if node_count is not None else len(nodes),
        nodes=tuple(nodes),
        epoch=epoch,
        members=frozenset(members),
        metrics_exists=metrics,
        no_progress_s=no_progress,
        due_kills=frozenset(due),
    )


# --------------------------------------------------------------------------- #
# Startup
# --------------------------------------------------------------------------- #
def test_startup_waits_for_all_nodes():
    # Two nodes desired, only one running/registered -> publish nothing yet.
    obs = _obs([_node(0), _node(1, state="pending")], epoch=0, members=[], node_count=2)
    assert decide(obs, SHRINK) == []


def test_startup_publishes_epoch_1_when_all_healthy():
    obs = _obs([_node(0), _node(1)], epoch=0, members=[], node_count=2)
    assert decide(obs, SHRINK) == [PublishEpoch(1, (0, 1))]


# --------------------------------------------------------------------------- #
# Loss -> shrink
# --------------------------------------------------------------------------- #
def test_lost_member_shrinks_without_replacement_under_shrink_policy():
    # node 1 terminated; shrink policy republishes survivors only, no relaunch.
    obs = _obs(
        [_node(0), _node(1, state="terminated")],
        epoch=1,
        members=[0, 1],
        node_count=2,
    )
    assert decide(obs, SHRINK) == [PublishEpoch(2, (0,))]


def test_lost_member_shrinks_and_relaunches_under_preempt_policy():
    obs = _obs(
        [_node(0), _node(1, state="shutting-down")],
        epoch=1,
        members=[0, 1],
        node_count=2,
    )
    assert decide(obs, PREEMPT) == [PublishEpoch(2, (0,)), LaunchReplacement(1)]


def test_scheduled_kill_only_terminates_membership_unchanged():
    # due_kills=1: TERMINATE the box, but do NOT shrink yet — node 1 still reads
    # healthy (AWS lag), so membership is untouched. The shrink is observation-
    # driven and comes a tick or two later (next test).
    obs = _obs([_node(0), _node(1)], epoch=1, members=[0, 1], due=[1])
    assert decide(obs, SHRINK) == [TerminateNode(1)]
    assert decide(obs, PREEMPT) == [TerminateNode(1)]


def test_shrink_happens_when_kill_is_observed_next_tick():
    # After the terminate, once AWS shows node 1 gone (shutting-down), the same
    # reducer that handles a real reclaim shrinks — and, under preempt, replaces.
    # due is empty now (the shell dedups the already-issued kill).
    obs = _obs([_node(0), _node(1, state="shutting-down")], epoch=1, members=[0, 1])
    assert decide(obs, SHRINK) == [PublishEpoch(2, (0,))]
    assert decide(obs, PREEMPT) == [PublishEpoch(2, (0,)), LaunchReplacement(1)]


# --------------------------------------------------------------------------- #
# Join -> grow
# --------------------------------------------------------------------------- #
def test_replacement_registered_grows_group():
    # Running at world 1 (members={0}); node 1's replacement is now healthy and
    # not yet a member -> republish including it.
    obs = _obs([_node(0), _node(1)], epoch=2, members=[0], node_count=2)
    assert decide(obs, PREEMPT) == [PublishEpoch(3, (0, 1))]


def test_no_op_when_membership_matches_healthy():
    obs = _obs([_node(0), _node(1)], epoch=3, members=[0, 1])
    assert decide(obs, PREEMPT) == []


def test_pending_replacement_not_yet_admitted():
    # The replacement box is booting (pending) — not healthy, so no grow yet.
    obs = _obs([_node(0), _node(1, state="pending")], epoch=2, members=[0], node_count=2)
    assert decide(obs, PREEMPT) == []


# --------------------------------------------------------------------------- #
# Floors
# --------------------------------------------------------------------------- #
def test_metrics_exists_is_done():
    obs = _obs([_node(0)], epoch=5, members=[0], metrics=True)
    assert decide(obs, SHRINK) == [Done()]


def test_stall_triggers_whole_group_restart():
    obs = _obs([_node(0), _node(1)], epoch=2, members=[0, 1], no_progress=700)
    assert decide(obs, PREEMPT) == [WholeGroupRestart()]


def test_all_gone_triggers_whole_group_restart():
    obs = _obs(
        [_node(0, state="terminated"), _node(1, state="terminated")],
        epoch=2,
        members=[0, 1],
    )
    assert decide(obs, PREEMPT) == [WholeGroupRestart()]


def test_stale_heartbeat_counts_as_lost():
    # node 1 still "running" per AWS but its log went silent > timeout -> wedged.
    obs = _obs(
        [_node(0, log_age=1.0), _node(1, log_age=999.0)],
        epoch=1,
        members=[0, 1],
        node_count=2,
    )
    assert decide(obs, SHRINK) == [PublishEpoch(2, (0,))]


def test_unregistered_node_is_not_healthy():
    obs = _obs([_node(0), _node(1, registered=False)], epoch=0, members=[], node_count=2)
    assert decide(obs, SHRINK) == []  # only 1 of 2 registered -> keep waiting


# --------------------------------------------------------------------------- #
# epoch_doc schema + config keys + storage
# --------------------------------------------------------------------------- #
def test_epoch_doc_ranks_and_master():
    doc = epoch_doc("r", 3, (0, 2), {0: "10.0.0.1", 2: "10.0.0.2"}, port_base=29400)
    assert doc == {
        "epoch": 3,
        "members": [
            {"node": 0, "ip": "10.0.0.1", "rank": 0},
            {"node": 2, "ip": "10.0.0.2", "rank": 1},
        ],
        "node_count": 2,
        "master_addr": "10.0.0.1",  # lowest-index member
        "master_port": 29403,  # base + epoch
    }


def test_config_epoch_keys():
    cfg = OrchestratorConfig(bucket="b")
    assert cfg.run_epoch_key("r") == "runs/r/epoch.json"
    assert cfg.run_node_key("r", 2) == "runs/r/nodes/node2.json"
    assert cfg.run_nodes_prefix("r") == "runs/r/nodes/"
    assert cfg.run_uri("r") == "s3://b/runs/r"


def test_read_bytes_roundtrip_and_absent(tmp_path):
    uri = str(tmp_path / "doc.json")
    assert s3_store.read_bytes(uri) is None  # absent
    s3_store.put_bytes(b'{"epoch": 1}', uri)
    assert s3_store.read_bytes(uri) == b'{"epoch": 1}'


def test_run_node_uri_is_full_s3_uri():
    # Regression: the supervisor reads registrations via read_bytes, which treats
    # a prefix-less string as a LOCAL path — so it MUST be a full s3:// URI, or
    # every node reads as unregistered and epoch 1 never publishes.
    cfg = OrchestratorConfig(bucket="b")
    assert cfg.run_node_uri("r", 1) == "s3://b/runs/r/nodes/node1.json"


def test_observe_sees_registration_and_publishes_epoch_1(monkeypatch):
    # End-to-end shell test through the real Supervisor._observe: two running,
    # registered nodes must yield PublishEpoch(1). This exercises _node_ip's URI
    # construction (the layer the bare-key bug lived in) that the pure-reducer
    # tables above can't reach.
    from orchestrator import supervisor as sup_mod

    cfg = OrchestratorConfig(bucket="b")
    cfg.node_count = 2
    monkeypatch.setattr(sup_mod.aws, "instance_state", lambda iid: "running")
    monkeypatch.setattr(sup_mod.aws, "object_last_modified", lambda b, k: None)
    monkeypatch.setattr(sup_mod.aws, "max_checkpoint_step", lambda b, p: -1)
    monkeypatch.setattr(sup_mod.aws, "object_exists", lambda b, k: False)
    # Registrations present ONLY at the full s3:// node URIs — a bare key returns
    # None, which is exactly the bug this guards against.
    node_docs = {
        cfg.run_node_uri("r", 0): b'{"ip": "10.0.0.0"}',
        cfg.run_node_uri("r", 1): b'{"ip": "10.0.0.1"}',
    }
    monkeypatch.setattr(sup_mod.s3_store, "read_bytes", lambda uri: node_docs.get(uri))

    from orchestrator.profile import RunProfile

    s = sup_mod.Supervisor(
        cfg,
        RunProfile("r", kind="multinode", market="spot"),
        run_id="r",
        policy=PREEMPT,
        node_ids={0: "i-0", 1: "i-1"},
        logs={0: {"key": "k0", "state": {"printed": 0}}, 1: {"key": "k1", "state": {"printed": 0}}},
        launch_node=lambda n: "i-new",
        pull_logs=lambda: None,
    )
    obs = s._observe(now=0.0, wall=0.0)
    assert all(n.registered for n in obs.nodes)
    assert decide(obs, PREEMPT) == [PublishEpoch(1, (0, 1))]


def test_terminate_does_not_shortcut_membership(monkeypatch):
    # The heart of "observation-driven": after the supervisor terminates a node,
    # its health follows AWS state, NOT the fact that we killed it. While AWS
    # still lags at "running" the node stays healthy (no shrink); only once AWS
    # reports it gone does membership react.
    from orchestrator import supervisor as sup_mod
    from orchestrator.profile import RunProfile

    cfg = OrchestratorConfig(bucket="b")
    cfg.node_count = 2
    state = {"i-0": "running", "i-1": "running"}
    monkeypatch.setattr(sup_mod.aws, "instance_state", lambda iid: state[iid])
    monkeypatch.setattr(sup_mod.aws, "object_last_modified", lambda b, k: None)
    monkeypatch.setattr(sup_mod.aws, "max_checkpoint_step", lambda b, p: 5)
    monkeypatch.setattr(sup_mod.aws, "object_exists", lambda b, k: False)
    monkeypatch.setattr(sup_mod.aws, "terminate", lambda iid: None)  # AWS lag: state unchanged
    docs = {
        cfg.run_node_uri("r", 0): b'{"ip": "10.0.0.0"}',
        cfg.run_node_uri("r", 1): b'{"ip": "10.0.0.1"}',
    }
    monkeypatch.setattr(sup_mod.s3_store, "read_bytes", lambda uri: docs.get(uri))

    s = sup_mod.Supervisor(
        cfg,
        RunProfile("r", kind="multinode-shrink", market="spot"),
        run_id="r",
        policy=SHRINK,
        node_ids={0: "i-0", 1: "i-1"},
        logs={0: {"key": "k0", "state": {"printed": 0}}, 1: {"key": "k1", "state": {"printed": 0}}},
        launch_node=lambda n: "i-new",
        pull_logs=lambda: None,
    )
    s.st.epoch, s.st.members = 1, frozenset({0, 1})

    s._terminate(1)  # kill the box; AWS still shows it running (lag)
    obs = s._observe(now=1.0, wall=1.0)
    assert decide(obs, SHRINK) == []  # NOT shrunk — node 1 still observed healthy

    state["i-1"] = "shutting-down"  # AWS finally reflects the death
    obs = s._observe(now=2.0, wall=2.0)
    assert decide(obs, SHRINK) == [PublishEpoch(2, (0,))]  # now it reacts


# --------------------------------------------------------------------------- #
# status_doc — the observability document the `logs` viewer reads
# --------------------------------------------------------------------------- #
_ORCH_KEY = "runs/r/logs/orchestrator.log"


def _status(obs, *, members, logs, prev=None, now=42.0, done=False, ips=None, node_ids=None):
    return status_doc(
        "r",
        obs,
        SHRINK,
        epoch=obs.epoch,
        members=frozenset(members),
        ips=ips or {},
        node_ids=node_ids or {},
        logs=logs,
        orch_log_key=_ORCH_KEY,
        prev=prev,
        now=now,
        done=done,
    )


def _states(doc):
    return {(e["node"], e["attempt"]): e["state"] for e in doc["nodes"]}


_LOGS2 = {0: {"key": "k0", "attempt": 0}, 1: {"key": "k1", "attempt": 0}}


def test_status_doc_alive_dead_pending():
    # node 0 healthy, node 1 terminated, node 2 still booting (unregistered).
    obs = _obs(
        [_node(0), _node(1, state="terminated"), _node(2, registered=False)],
        epoch=2,
        members=[0, 1],
        node_count=3,
    )
    logs = {**_LOGS2, 2: {"key": "k2", "attempt": 0}}
    doc = _status(obs, members=[0], logs=logs, node_ids={0: "i-0", 1: "i-1", 2: "i-2"})
    assert _states(doc) == {(0, 0): "alive", (1, 0): "dead", (2, 0): "pending"}
    assert doc["updated_at"] == 42.0
    assert doc["members"] == [0]
    assert doc["orchestrator"] == {"log_key": _ORCH_KEY}
    assert doc["done"] is False
    by_node = {e["node"]: e for e in doc["nodes"]}
    assert by_node[1]["aws_state"] == "terminated" and by_node[1]["instance_id"] == "i-1"


def test_status_doc_stale_heartbeat_kills_previously_alive():
    # (1,0) was alive last tick; now AWS still says running but the log went
    # silent past the timeout -> dead. A never-alive stale entry stays pending.
    prev = _status(_obs([_node(0), _node(1)], epoch=1, members=[0, 1]), members=[0, 1], logs=_LOGS2)
    obs = _obs([_node(0, log_age=1.0), _node(1, log_age=999.0)], epoch=1, members=[0, 1])
    doc = _status(obs, members=[0, 1], logs=_LOGS2, prev=prev)
    assert _states(doc) == {(0, 0): "alive", (1, 0): "dead"}


def test_status_doc_dead_is_sticky():
    prev = _status(
        _obs([_node(0), _node(1, state="terminated")], epoch=2, members=[0, 1]),
        members=[0],
        logs=_LOGS2,
    )
    assert _states(prev)[(1, 0)] == "dead"
    # Even if the observation flips healthy again, (1,0) never resurrects.
    obs = _obs([_node(0), _node(1)], epoch=3, members=[0, 1])
    doc = _status(obs, members=[0, 1], logs=_LOGS2, prev=prev)
    assert _states(doc)[(1, 0)] == "dead"


def test_status_doc_replacement_carries_dead_attempt_forward():
    prev = _status(
        _obs([_node(0), _node(1, state="terminated")], epoch=2, members=[0, 1]),
        members=[0],
        logs=_LOGS2,
    )
    # The replacement booted: node 1 now maps to attempt 1 with a fresh log key.
    logs = {0: {"key": "k0", "attempt": 0}, 1: {"key": "k1-r1", "attempt": 1}}
    obs = _obs([_node(0), _node(1)], epoch=3, members=[0])
    doc = _status(obs, members=[0], logs=logs, prev=prev)
    assert _states(doc) == {(0, 0): "alive", (1, 0): "dead", (1, 1): "alive"}
    by_key = {(e["node"], e["attempt"]): e for e in doc["nodes"]}
    assert by_key[(1, 0)]["log_key"] == "k1"  # frozen entry keeps its own log
    assert by_key[(1, 1)]["log_key"] == "k1-r1"


def test_status_doc_done_flag():
    obs = _obs([_node(0)], epoch=5, members=[0], metrics=True)
    doc = _status(obs, members=[0], logs={0: {"key": "k0", "attempt": 0}}, done=True)
    assert doc["done"] is True


def test_supervisor_writes_status_each_tick_and_survives_failure(monkeypatch):
    # The shell hook: _write_status uploads status.json (+ orchestrator.log once
    # events accrued), and a failing put_text is swallowed — observability must
    # never kill the run.
    import json as _json

    from orchestrator import supervisor as sup_mod
    from orchestrator.profile import RunProfile

    cfg = OrchestratorConfig(bucket="b")
    cfg.node_count = 2
    monkeypatch.setattr(sup_mod.aws, "instance_state", lambda iid: "running")
    monkeypatch.setattr(sup_mod.aws, "object_last_modified", lambda b, k: None)
    monkeypatch.setattr(sup_mod.aws, "max_checkpoint_step", lambda b, p: -1)
    monkeypatch.setattr(sup_mod.aws, "object_exists", lambda b, k: False)
    docs = {
        cfg.run_node_uri("r", 0): b'{"ip": "10.0.0.0"}',
        cfg.run_node_uri("r", 1): b'{"ip": "10.0.0.1"}',
    }
    monkeypatch.setattr(sup_mod.s3_store, "read_bytes", lambda uri: docs.get(uri))

    s = sup_mod.Supervisor(
        cfg,
        RunProfile("r", kind="multinode", market="spot"),
        run_id="r",
        policy=PREEMPT,
        node_ids={0: "i-0", 1: "i-1"},
        logs={
            0: {"key": "k0", "attempt": 0, "state": {"printed": 0}},
            1: {"key": "k1", "attempt": 0, "state": {"printed": 0}},
        },
        launch_node=lambda n: "i-new",
        pull_logs=lambda: None,
    )

    # Tick 1: put_text raises -> swallowed, nothing cached.
    def boom(b, k, t):
        raise RuntimeError("s3 down")

    monkeypatch.setattr(sup_mod.aws, "put_text", boom)
    s._write_status(s._observe(now=0.0, wall=100.0), 100.0)
    assert s._last_status is None

    # Tick 2: healthy write; an _event makes orchestrator.log upload too.
    puts: dict[str, str] = {}
    monkeypatch.setattr(sup_mod.aws, "put_text", lambda b, k, t: puts.__setitem__(k, t))
    s._event("terminated node 1 (i-1)")
    s._write_status(s._observe(now=1.0, wall=101.0), 101.0)
    doc = _json.loads(puts[cfg.run_status_key("r")])
    assert doc["updated_at"] == 101.0
    assert _states(doc) == {(0, 0): "alive", (1, 0): "alive"}
    assert "terminated node 1 (i-1)" in puts[cfg.run_orch_log_key("r")]
    assert s._last_status == doc

    # Tick 3: no new events -> status re-uploaded, orchestrator.log NOT.
    puts.clear()
    s._write_status(s._observe(now=2.0, wall=102.0), 102.0)
    assert cfg.run_status_key("r") in puts and cfg.run_orch_log_key("r") not in puts


def test_replacement_attempt_is_not_born_dead(monkeypatch):
    # Regression: on the tick that launches a replacement, _execute bumps node 1
    # to attempt 1 with a fresh instance, but the tick's `obs` still holds the
    # OLD terminated instance. If status is written from that stale obs paired
    # with the new logs, node1·r1 is stamped dead the instant it appears and
    # sticky-dead locks it forever (the box shows [DEAD] while training at ws 2).
    # Writing status BEFORE _execute keeps obs and logs consistent.
    import json as _json

    from orchestrator import supervisor as sup_mod
    from orchestrator.profile import RunProfile

    cfg = OrchestratorConfig(bucket="b")
    cfg.node_count = 2
    aws_state = {"i-0": "running", "i-1": "terminated"}  # node 1 just reclaimed
    monkeypatch.setattr(sup_mod.aws, "instance_state", lambda iid: aws_state.get(iid, "running"))
    monkeypatch.setattr(sup_mod.aws, "object_last_modified", lambda b, k: None)
    monkeypatch.setattr(sup_mod.aws, "max_checkpoint_step", lambda b, p: 5)
    monkeypatch.setattr(sup_mod.aws, "object_exists", lambda b, k: False)
    docs = {
        cfg.run_node_uri("r", 0): b'{"ip": "10.0.0.0"}',
        cfg.run_node_uri("r", 1): b'{"ip": "10.0.0.1"}',
    }
    monkeypatch.setattr(sup_mod.s3_store, "read_bytes", lambda uri: docs.get(uri))
    puts: dict[str, str] = {}
    monkeypatch.setattr(sup_mod.aws, "put_text", lambda b, k, t: puts.__setitem__(k, t))

    logs = {
        0: {"key": "k0", "attempt": 0, "state": {"printed": 0}},
        1: {"key": "k1", "attempt": 0, "state": {"printed": 0}},
    }

    def launch(node):  # a replacement: bump attempt + fresh instance, as the real one does
        logs[node] = {"key": f"k{node}-r1", "attempt": 1, "state": {"printed": 0}}
        aws_state["i-1r"] = "running"
        return "i-1r"

    s = sup_mod.Supervisor(
        cfg,
        RunProfile("r", kind="multinode-preempt", market="spot"),
        run_id="r",
        policy=PREEMPT,
        node_ids={0: "i-0", 1: "i-1"},
        logs=logs,
        launch_node=launch,
        pull_logs=lambda: None,
    )
    s.st.epoch, s.st.members = 1, frozenset({0, 1})
    s.st.ips = {0: "10.0.0.0", 1: "10.0.0.1"}

    # Reproduce ONE loop iteration in the fixed order: observe -> write -> execute.
    obs = s._observe(now=1.0, wall=1.0)
    actions = decide(obs, PREEMPT)
    assert LaunchReplacement(1) in actions  # node 1 observed gone -> replace
    s._write_status(obs, 1.0)  # status written from the consistent (obs, logs) pair
    s._execute(actions)  # only now does logs[1] become attempt 1 + i-1r

    doc = _json.loads(puts[cfg.run_status_key("r")])
    st = {(e["node"], e["attempt"]): e["state"] for e in doc["nodes"]}
    assert st[(1, 0)] == "dead"  # the reclaimed attempt: correctly dead
    assert (1, 1) not in st  # the replacement is NOT yet present, so never born dead

    # Next tick: the replacement is observed running -> it surfaces alive.
    s.st.ips[1] = "10.0.0.1"  # replacement re-registers
    obs2 = s._observe(now=2.0, wall=2.0)
    s._write_status(obs2, 2.0)
    doc2 = _json.loads(puts[cfg.run_status_key("r")])
    st2 = {(e["node"], e["attempt"]): e["state"] for e in doc2["nodes"]}
    assert st2[(1, 1)] == "alive"  # born alive, not dead
    assert st2[(1, 0)] == "dead"  # predecessor carried forward, frozen


def test_scheduled_kill_fires_exactly_once_even_after_replacement(monkeypatch):
    # Regression: the kill schedule is level-triggered on "elapsed >= secs", so
    # without per-ENTRY edge-triggering it re-fires every tick after the due
    # time — and re-kills each replacement the instant it rejoins (an infinite
    # kill loop, observed on AWS as epochs 5,6,7,8,... churning).
    from orchestrator import supervisor as sup_mod
    from orchestrator.profile import RunProfile

    cfg = OrchestratorConfig(bucket="b")
    cfg.node_count = 2
    monkeypatch.setattr(sup_mod.aws, "instance_state", lambda iid: "running")
    monkeypatch.setattr(sup_mod.aws, "object_last_modified", lambda b, k: None)
    monkeypatch.setattr(sup_mod.aws, "max_checkpoint_step", lambda b, p: 5)
    monkeypatch.setattr(sup_mod.aws, "object_exists", lambda b, k: False)
    docs = {
        cfg.run_node_uri("r", 0): b'{"ip": "10.0.0.0"}',
        cfg.run_node_uri("r", 1): b'{"ip": "10.0.0.1"}',
    }
    monkeypatch.setattr(sup_mod.s3_store, "read_bytes", lambda uri: docs.get(uri))

    s = sup_mod.Supervisor(
        cfg,
        RunProfile("r", kind="multinode-preempt", market="spot"),
        run_id="r",
        policy=PREEMPT,
        node_ids={0: "i-0", 1: "i-1"},
        logs={0: {"key": "k0", "state": {"printed": 0}}, 1: {"key": "k1", "state": {"printed": 0}}},
        launch_node=lambda n: "i-new",
        pull_logs=lambda: None,
        kill_schedule=[(100.0, 1)],  # one kill, at 100s after train start
    )
    s._train_start = 0.0

    # Well past the due time: the kill fires on the first observe...
    assert 1 in s._observe(now=200.0, wall=0.0).due_kills
    # ...and NEVER again, even though elapsed is still >> 100 and node 1 has been
    # "replaced" (fresh instance id, as _launch_replacement would set).
    s.node_ids[1] = "i-1-replacement"
    for t in (210.0, 220.0, 300.0):
        assert s._observe(now=t, wall=0.0).due_kills == frozenset()
