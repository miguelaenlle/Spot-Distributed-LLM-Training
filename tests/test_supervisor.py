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
)
from spot_train import s3_store

SHRINK = Policy(replace_on_loss=False, recovery_timeout_s=600)
PREEMPT = Policy(replace_on_loss=True, recovery_timeout_s=600)


def _node(i, state="running", registered=True, terminated=False, log_age=None):
    return NodeObs(
        node=i,
        aws_state=state,
        registered=registered,
        terminated_by_us=terminated,
        log_age_s=log_age,
    )


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


def test_scheduled_kill_publishes_shrink_same_tick():
    # due_kills=1: terminate it AND publish the survivor epoch in the same tick,
    # so survivors' sidecars drop their NCCL-blocked torchrun within one poll.
    obs = _obs([_node(0), _node(1)], epoch=1, members=[0, 1], due=[1])
    assert decide(obs, SHRINK) == [TerminateNode(1), PublishEpoch(2, (0,))]


def test_scheduled_kill_under_preempt_also_relaunches():
    obs = _obs([_node(0), _node(1)], epoch=1, members=[0, 1], due=[1])
    assert decide(obs, PREEMPT) == [
        TerminateNode(1),
        PublishEpoch(2, (0,)),
        LaunchReplacement(1),
    ]


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
