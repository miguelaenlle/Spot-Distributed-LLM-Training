"""User-data builder tests — hermetic (no AWS; builds strings and runs bash -n).

Pins the boot-script shapes the experiments depend on:

  - single-process and single-node-DDP scripts stay free of multi-node
    artifacts (the multinode work must not leak into the proven 1a/1b paths);
  - the multi-node generation loop has the pieces the pause-and-replace
    failure model needs (ready markers, publish-after-ready ordering,
    per-generation port, budget re-export, done-signal check);
  - every generated script parses (`bash -n`).
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from orchestrator import bootstrap
from orchestrator.config import OrchestratorConfig


def _cfg() -> OrchestratorConfig:
    return OrchestratorConfig(bucket="test-bucket")


def _ud(**kwargs) -> str:
    return bootstrap.build_user_data(
        _cfg(), run_id="run-1", market="on-demand", max_seconds=120, **kwargs
    )


ALL_SHAPES = {
    "single": {},
    "ddp-standalone": {"ddp": True},
    "mn-node0": {"ddp": True, "nodes": 2, "node_index": 0},
    "mn-node1": {"ddp": True, "nodes": 2, "node_index": 1},
    "mn3-node2": {"ddp": True, "nodes": 3, "node_index": 2},
}


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_bash_syntax(shape, tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    path = tmp_path / f"{shape}.sh"
    path.write_text(_ud(**ALL_SHAPES[shape]))
    r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, f"bash -n failed for {shape}:\n{r.stderr}"


def test_single_node_paths_have_no_multinode_artifacts():
    for kwargs in ({}, {"ddp": True}, {"ddp": True, "nproc_per_node": 2}):
        ud = _ud(**kwargs)
        for marker in (
            "rdzv",
            "generation",
            "GEN",
            "ready/",
            "NCCL_TIMEOUT",
            "MN_RC",
            "budget.json",
            "TORCH_NCCL_DUMP_ON_TIMEOUT",
        ):
            assert marker not in ud, f"{marker!r} leaked into single-node user-data"
    assert "--standalone" in _ud(ddp=True)


def test_multinode_master_publishes_only_after_ready_markers():
    ud = _ud(ddp=True, nodes=2, node_index=0)
    # The livelock guard: node 0 waits for the workers' ready markers BEFORE
    # publishing rdzv.json (and only then starts torchrun).
    ready_wait = ud.index("ready/gen$GEN-node")
    publish = ud.index('"generation": $GEN')
    torchrun = ud.index("torch.distributed.run")
    assert ready_wait < publish < torchrun
    # Fresh port per generation dodges TIME_WAIT on the master's own relaunch.
    assert "PORT=$((29400 + GEN))" in ud
    assert '--master_port="$PORT"' in ud
    assert "--max-restarts=0" in ud


def test_multinode_worker_dials_what_is_published():
    ud = _ud(ddp=True, nodes=2, node_index=1)
    # Worker announces readiness for its target generation...
    assert "ready/gen$GEN-node1" in ud
    # ...then joins whatever node 0 actually published (addr/port/gen read back
    # from rdzv.json, not assumed).
    assert "read RDZV_ADDR PORT GEN < /tmp/rdzv_join" in ud
    # A worker never publishes.
    assert '"generation": $GEN' not in ud


def test_multinode_loop_budget_and_done_signal():
    for node_index in (0, 1):
        ud = _ud(ddp=True, nodes=2, node_index=node_index)
        # Budget is orchestrator-authoritative: read budget.json each generation,
        # clamp >= 1 (NEVER exit on exhausted budget — rank 0 must always be able
        # to re-form the group for the eval + metrics.json ending), and export it.
        assert "runs/run-1/budget.json" in ud
        assert '[ "$REMAINING" -ge 1 ] || REMAINING=1' in ud
        assert "export MAX_SECONDS=$REMAINING" in ud
        # The old local wall-clock arithmetic (which billed the crash tail as
        # training and made survivors give up) must be gone.
        assert "ORIG_BUDGET" not in ud
        assert "CONSUMED" not in ud
        # metrics.json is the group-wide done signal; a clean local exit or the
        # done signal are the only RC=0 paths.
        assert "metrics.json" in ud
        assert "MN_RC=0" in ud
        # Survivors abort fast on a dead peer — and skip torch's ~2-minute
        # post-timeout debug dump, which delayed every rejoin.
        assert 'export NCCL_TIMEOUT="60"' in ud
        assert 'export TORCH_NCCL_DUMP_ON_TIMEOUT="0"' in ud


def test_log_key_attempt_suffix():
    cfg = _cfg()
    assert cfg.run_logs_key("r", node=1) == "runs/r/logs/boot-node1.log"
    assert cfg.run_logs_key("r", node=1, attempt=2) == "runs/r/logs/boot-node1-r2.log"
    # attempt=0 must not change existing keys (stream continuity for node 0).
    assert cfg.run_logs_key("r", node=0, attempt=0) == "runs/r/logs/boot-node0.log"


def test_instance_vcpu_count():
    cfg = _cfg()
    assert cfg.instance_vcpu_count() == 4  # g4dn.xlarge default
    cfg.instance_type = "g4dn.12xlarge"
    assert cfg.instance_vcpu_count() == 48
    cfg.instance_vcpus = 7  # explicit override wins
    assert cfg.instance_vcpu_count() == 7
    cfg.instance_vcpus = 0
    cfg.instance_type = "p9.superlarge"
    with pytest.raises(SystemExit, match="INSTANCE_VCPUS"):
        cfg.instance_vcpu_count()
