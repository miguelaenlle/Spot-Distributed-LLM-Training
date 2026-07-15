"""Durable ASG-backed orchestrator + the preempting scaling sweep.

Split into: (1) the AWS launch-template/ASG/IAM helpers under ``set_dry_run`` (no
creds, canned returns, idempotent), (2) the pure config/experiment pieces
(_run_id override, _aggressive_victims, _preempt_stats, scaling-compare), (3) the
orchestrator user-data builder, and (4) the F0 fault-injection CORE — a
deterministic stand-in for "the t3.micro dies mid-job" that drives ``run_on_box``
twice against an in-memory fake S3/EC2 and asserts the generation bumps and cold
recovery kills the orphan GPU boxes.
"""

from __future__ import annotations

import json

import pytest

from orchestrator import aws, bootstrap, experiments, remote
from orchestrator.config import OrchestratorConfig
from orchestrator.profile import Event, RunProfile, Sample


@pytest.fixture
def cfg():
    c = OrchestratorConfig()
    c.bucket = "test-bucket"
    return c


# --------------------------------------------------------------------------- #
# (1) AWS helpers under dry-run — no creds, canned returns, idempotent
# --------------------------------------------------------------------------- #
@pytest.fixture
def dry():
    aws.set_dry_run(True)
    yield
    aws.set_dry_run(False)


def test_aws_lookups_canned_under_dry_run(dry):
    assert aws.caller_account_id() == "000000000000"
    assert aws.default_subnet_ids() == ["subnet-DRYRUN"]
    assert aws.describe_asg("spot-orch-x") is None


def test_launch_template_and_asg_helpers_are_noops_under_dry_run(dry):
    name = aws.ensure_launch_template(
        "spot-orch-lt-x",
        ami_id="ami-1",
        instance_type="t3.micro",
        profile_name="spot-orch-profile",
        security_group_id="sg-1",
        user_data="#!/bin/bash\necho hi",
        key_name="",
        tags={"job": "x"},
    )
    assert name == "spot-orch-lt-x"
    # None of these raise or touch boto3 under dry-run.
    aws.ensure_auto_scaling_group(
        "spot-orch-x",
        launch_template_name=name,
        subnet_ids=["subnet-1"],
        min_size=1,
        max_size=1,
        desired=1,
    )
    aws.set_asg_capacity("spot-orch-x", min_size=0, max_size=0, desired=0)
    aws.delete_asg("spot-orch-x", force=True)
    aws.delete_launch_template(name)


def test_ensure_orchestrator_profile_noop_under_dry_run(dry):
    # Idempotent + credential-free under dry-run: just must not raise.
    aws.ensure_orchestrator_profile("spot-orch-role", "spot-orch-profile", "b", "123")
    aws.ensure_orchestrator_profile("spot-orch-role", "spot-orch-profile", "b", "123")


# --------------------------------------------------------------------------- #
# (2) config keys + pure experiment helpers
# --------------------------------------------------------------------------- #
def test_config_names_are_deterministic(cfg):
    assert cfg.orchestrator_asg_name("multinode-9") == "spot-orch-multinode-9"
    assert cfg.orchestrator_lt_name("multinode-9") == "spot-orch-lt-multinode-9"
    assert cfg.sweep_results_key("s1") == "sweeps/s1/results.json"
    assert cfg.orchestrator_generation_key("j") == "orchestrators/j/generation"
    assert cfg.orchestrator_done_key("j") == "orchestrators/j/orchestrator-done.json"


def test_run_id_honors_remote_override(monkeypatch):
    monkeypatch.delenv("REMOTE_RUN_ID", raising=False)
    assert experiments._run_id("multinode").startswith("multinode-")
    monkeypatch.setenv("REMOTE_RUN_ID", "multinode-42")
    assert experiments._run_id("multinode") == "multinode-42"


@pytest.mark.parametrize(
    "n,expected",
    [(2, [1]), (4, [2, 3]), (8, [4, 5, 6, 7]), (16, list(range(8, 16)))],
)
def test_aggressive_victims_halves_the_world(monkeypatch, n, expected):
    monkeypatch.delenv("PREEMPT_VICTIMS", raising=False)
    assert experiments._aggressive_victims(n) == expected
    # every victim leaves a survivor (never kills the whole group)
    assert len(expected) < n


def test_aggressive_victims_env_override(monkeypatch):
    monkeypatch.setenv("PREEMPT_VICTIMS", "0,3")
    assert experiments._aggressive_victims(4) == [0, 3]
    monkeypatch.setenv("PREEMPT_VICTIMS", "9")  # out of range for 4 nodes
    with pytest.raises(SystemExit):
        experiments._aggressive_victims(4)


def test_preempt_stats_from_marks():
    p = RunProfile("r", kind="multinode-preempt", market="spot")
    p.events = [
        Event("kill", 100.0, 1),
        Event("shrink_resume", 110.0, 1),
        Event("full_world", 150.0, 1),
    ]
    p.samples = [
        Sample(step=1, loss=1.0, ms_per_step=80, tok_s=1000, t_rel=0.0, world_size=4),
        Sample(step=2, loss=1.0, ms_per_step=80, tok_s=1000, t_rel=10.0, world_size=2),
        Sample(step=3, loss=1.0, ms_per_step=80, tok_s=1000, t_rel=60.0, world_size=4),
    ]
    s = experiments._preempt_stats(p)
    assert s["killed"] == 1
    assert s["min_world"] == 2
    assert s["recovery_s"] == 50.0  # full_world - kill
    assert s["degraded_s"] == 40.0  # full_world - shrink_resume


def test_scaling_compare_joins_by_node_count(cfg, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for sid, ms2, wall2 in [("clean", 120, 400), ("preempt", 122, 470)]:
        d = tmp_path / "reports" / sid
        d.mkdir(parents=True)
        res = {
            "sweep_id": sid,
            "recipe": {"kind": f"scaling-{sid}", "throughput_only": True, "cap_s": 480},
            "results": [
                {
                    "nodes": 2,
                    "run_id": f"{sid}-2",
                    "analysis": {"reached": False},
                    "ms_per_step": ms2,
                    "run_time_s": wall2,
                    "preempt_stats": {
                        "killed": 1,
                        "min_world": 1,
                        "recovery_s": 55,
                        "degraded_s": 40,
                    },
                }
            ],
        }
        (d / "results.json").write_text(json.dumps(res))
    out = remote_free(lambda: experiments.run_scaling_compare(cfg, "clean", "preempt"))
    with open(out["summary"]) as f:
        summary = f.read()
    assert "runtime overhead +18%" in summary
    assert "recovery 55s" in summary


def remote_free(fn):
    """Run a laptop-side helper with AWS in dry-run so no creds are needed."""
    aws.set_dry_run(True)
    try:
        return fn()
    finally:
        aws.set_dry_run(False)


# --------------------------------------------------------------------------- #
# (3) orchestrator user-data builder
# --------------------------------------------------------------------------- #
def test_build_orchestrator_user_data_runs_remote_entrypoint(cfg):
    ud = bootstrap.build_orchestrator_user_data(
        cfg, job_id="scaling-clean-7", experiment="scaling-clean"
    )
    assert "orchestrator.remote" in ud
    assert 'EXPERIMENT="scaling-clean"' in ud
    assert 'ORCH_ASG_NAME="spot-orch-scaling-clean-7"' in ud
    # a sweep gets REMOTE_SWEEP_ID (not REMOTE_RUN_ID)
    assert 'REMOTE_SWEEP_ID="scaling-clean-7"' in ud
    assert "REMOTE_RUN_ID" not in ud
    # a single run pins the run_id instead
    ud2 = bootstrap.build_orchestrator_user_data(cfg, job_id="multinode-7", experiment="multinode")
    assert 'REMOTE_RUN_ID="multinode-7"' in ud2
    assert "shutdown -h now" in ud2  # self-poweroff backstop


# --------------------------------------------------------------------------- #
# (4) F0 fault-injection CORE — orchestrator dies mid-job, ASG relaunches it
# --------------------------------------------------------------------------- #
class _FakeCloud:
    """In-memory S3 (put/get/head/delete) + EC2 (orphan boxes, terminate) + ASG
    scaling, monkeypatched over orchestrator.aws so run_on_box is deterministic
    and needs no cloud. Persists the generation object ACROSS run_on_box calls —
    exactly what an ASG relaunch onto a fresh box does."""

    def __init__(self):
        self.store: dict[tuple[str, str], str] = {}
        self.orphans: list[dict] = []
        self.terminated: list[str] = []
        self.scaled: list[tuple] = []

    def install(self, monkeypatch):
        monkeypatch.setattr(aws, "object_exists", lambda b, k: (b, k) in self.store)
        monkeypatch.setattr(aws, "get_text", lambda b, k: self.store[(b, k)])
        monkeypatch.setattr(aws, "put_text", lambda b, k, v: self.store.__setitem__((b, k), v))
        monkeypatch.setattr(aws, "delete_object", lambda b, k: self.store.pop((b, k), None))
        def _list_keys(b, prefix):
            return sorted(k for (bb, k) in self.store if bb == b and k.startswith(prefix))

        monkeypatch.setattr(aws, "list_keys", _list_keys)
        monkeypatch.setattr(aws, "instances_by_tag", lambda key, val: list(self.orphans))
        monkeypatch.setattr(aws, "terminate", lambda iid: self.terminated.append(iid))
        monkeypatch.setattr(
            aws,
            "set_asg_capacity",
            lambda name, **kw: self.scaled.append((name, kw["desired"])),
        )


def test_orchestrator_death_bumps_generation_and_cold_recovers(cfg, monkeypatch):
    cloud = _FakeCloud()
    cloud.install(monkeypatch)
    monkeypatch.setenv("EXPERIMENT", "multinode")
    monkeypatch.setenv("ORCH_JOB_ID", "multinode-1")
    monkeypatch.setenv("ORCH_ASG_NAME", "spot-orch-multinode-1")
    monkeypatch.setenv("REMOTE_RUN_ID", "multinode-1")

    dispatched = {"n": 0}

    def _fake_dispatch(_cfg, _exp):
        dispatched["n"] += 1

    monkeypatch.setattr(remote, "_dispatch", _fake_dispatch)

    # --- gen 1: first boot. No prior box => no cold recovery. ---
    rc = remote.run_on_box(cfg)
    assert rc == 0
    assert cloud.store[("test-bucket", cfg.orchestrator_generation_key("multinode-1"))] == "1"
    assert dispatched["n"] == 1
    assert cloud.terminated == []  # gen 1 never cold-recovers
    # done-sentinel written + ASG scaled to 0 on clean finish
    assert ("test-bucket", cfg.orchestrator_done_key("multinode-1")) in cloud.store
    assert cloud.scaled[-1] == ("spot-orch-multinode-1", 0)

    # --- the box "dies" mid-next-job: leave orphan GPU boxes (registered to
    # nodes) + a stale epoch.json ---
    cloud.orphans = [{"id": "i-gpu0", "state": "running"}, {"id": "i-gpu1", "state": "running"}]
    cloud.store[("test-bucket", cfg.run_epoch_key("multinode-1"))] = "{}"
    cloud.store[("test-bucket", cfg.run_node_key("multinode-1", 0))] = '{"instance_id": "i-gpu0"}'
    cloud.store[("test-bucket", cfg.run_node_key("multinode-1", 1))] = '{"instance_id": "i-gpu1"}'

    # --- gen 2: ASG relaunched a fresh box. Cold recovery must fire. ---
    rc = remote.run_on_box(cfg)
    assert rc == 0
    assert cloud.store[("test-bucket", cfg.orchestrator_generation_key("multinode-1"))] == "2"
    # cold recovery terminated BOTH orphan GPU boxes...
    assert set(cloud.terminated) == {"i-gpu0", "i-gpu1"}
    # ...and cleared the stale epoch.json so the fresh supervisor starts clean
    assert ("test-bucket", cfg.run_epoch_key("multinode-1")) not in cloud.store
    # ...and the kill is FIRST-CLASS in the timeline: a killed/orchestrator-restart
    # [event] per node landed in orchestrator.log (which the Gantt parser reads).
    orch_log = cloud.store[("test-bucket", cfg.run_orch_log_key("multinode-1"))]
    assert orch_log.count('"state":"killed"') == 2
    assert "orchestrator-restart" in orch_log
    assert '"node":0' in orch_log and '"node":1' in orch_log
