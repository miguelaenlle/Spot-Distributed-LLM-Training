"""User-data builder tests — hermetic (no AWS; builds strings and runs bash -n).

Pins the boot-script shapes the experiments depend on:

  - single-process and single-node-DDP scripts stay free of multi-node
    artifacts (the multinode work must not leak into the proven 1a/1b paths);
  - the multi-node boot script hands off to the epoch sidecar (registers +
    runs orchestrator.sidecar) and carries the checkpoint-budget/local-tier
    env — with NO torchrun/rendezvous flags of its own (the sidecar owns the
    static torchrun per epoch);
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


def test_provision_updates_existing_clone():
    # Pre-baked AMI support: a boot that finds a baked clone must fast-forward
    # it to the branch tip, not skip the fetch (which would pin bake-time code).
    ud = _ud()
    assert "git -C app fetch --depth 1 origin main" in ud
    assert "git -C app reset --hard FETCH_HEAD" in ud
    assert "git clone --depth 1 -b main" in ud  # fresh-box path still present


def test_bake_user_data_bash_syntax(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    ud = bootstrap.build_bake_user_data(_cfg(), bake_id="b1", base_ami="ami-base")
    path = tmp_path / "bake.sh"
    path.write_text(ud)
    r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, f"bash -n failed for bake user-data:\n{r.stderr}"


def test_bake_user_data_provisions_but_bakes_no_run_state():
    ud = bootstrap.build_bake_user_data(_cfg(), bake_id="b1", base_ami="ami-base")
    # It provisions (clone + submodule + boto3) and reports to the bake keys.
    assert "git clone" in ud
    assert "git submodule update --init" in ud
    assert "bake/b1/status.json" in ud
    assert "bake/b1/bake.log" in ud
    assert '"base_ami": "ami-base"' in ud
    # Nothing run-specific may end up in the image: no training invocation, no
    # trainer env file, no run config — every training boot writes its own.
    for marker in ("spot_train.train", "spot-train.env", "MAX_SECONDS", "CHECKPOINT_URI"):
        assert marker not in ud, f"{marker!r} leaked into bake user-data"
    # The orchestrator stops+images the box; the box must not shut itself down.
    assert "shutdown" not in ud


def test_bake_status_and_log_keys():
    cfg = _cfg()
    assert cfg.bake_status_key("b1") == "bake/b1/status.json"
    assert cfg.bake_log_key("b1") == "bake/b1/bake.log"


def test_single_node_paths_have_no_multinode_artifacts():
    for kwargs in ({}, {"ddp": True}, {"ddp": True, "nproc_per_node": 2}):
        ud = _ud(**kwargs)
        for marker in (
            "rdzv",
            "epoch",
            "sidecar",
            "NCCL_TIMEOUT",
            "MN_RC",
            "TORCH_NCCL_DUMP_ON_TIMEOUT",
            "TRAIN_BUDGET_SECONDS",
            "LOCAL_CHECKPOINT_DIR",
            "--nnodes",
        ):
            assert marker not in ud, f"{marker!r} leaked into single-node user-data"
    assert "--standalone" in _ud(ddp=True)


def test_multinode_hands_off_to_epoch_sidecar():
    # The box registers + runs the sidecar; it does NOT negotiate a rendezvous.
    for node_index in (0, 1, 2):
        ud = _ud(ddp=True, nodes=3, node_index=node_index)
        assert (
            f'-m orchestrator.sidecar --run-uri "s3://test-bucket/runs/run-1" '
            f"--node-index {node_index}" in ud
        )
        assert 'export NPROC_PER_NODE="gpu"' in ud
        # No torchrun / rendezvous flags in the boot script itself — the sidecar
        # owns the static torchrun invocation now.
        for marker in (
            "torch.distributed.run",
            "--rdzv",
            "--node_rank",
            "--nnodes",
            "--max-restarts",
        ):
            assert marker not in ud, f"{marker!r} survived the epoch rewrite"


def test_multinode_budget_env_and_done_signal():
    for node_index in (0, 1):
        ud = _ud(ddp=True, nodes=2, node_index=node_index)
        # Run-level budget rides in the checkpoint; local tier for fast resume.
        assert 'export TRAIN_BUDGET_SECONDS="120"' in ud
        assert 'export LOCAL_CHECKPOINT_DIR="/tmp/spot-ckpt"' in ud
        # NCCL crash is the in-band backstop to the supervisor's epoch bump.
        assert 'export NCCL_TIMEOUT="20"' in ud
        assert 'export TORCH_NCCL_DUMP_ON_TIMEOUT="0"' in ud
        # NCCL net hygiene: pin off docker0/lo (else 4-node hangs on the first
        # collective), disable IB (none on g4dn/g5), WARN debug in the log.
        assert 'export NCCL_SOCKET_IFNAME="^docker0,lo"' in ud
        assert 'export NCCL_IB_DISABLE="1"' in ud
        assert 'export NCCL_DEBUG="WARN"' in ud
        # Parallel TCP sockets: ~2-4x all-reduce bandwidth on the no-EFA path.
        assert 'export NCCL_SOCKET_NTHREADS="4"' in ud
        assert 'export NCCL_NSOCKS_PERTHREAD="2"' in ud
        # The old generation/budget.json rendezvous protocol is gone.
        for marker in ("budget.json", "remaining_seconds", "GEN", "ready/"):
            assert marker not in ud, f"{marker!r} survived the epoch rewrite"


def test_nproc_per_node_forwarded_to_sidecar():
    ud = _ud(ddp=True, nodes=2, node_index=0, nproc_per_node=2)
    assert 'export NPROC_PER_NODE="2"' in ud


def test_config_epoch_keys():
    cfg = _cfg()
    assert cfg.run_epoch_key("run-1") == "runs/run-1/epoch.json"
    assert cfg.run_node_key("run-1", 3) == "runs/run-1/nodes/node3.json"
    assert cfg.run_uri("run-1") == "s3://test-bucket/runs/run-1"


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


def test_trainer_env_exports_sampling_and_prompts_base64():
    import base64
    import json

    ud = _ud()
    assert 'export SAMPLES_URI="s3://test-bucket/runs/run-1/samples.json"' in ud
    assert 'export SAMPLES_PREFIX_URI="s3://test-bucket/runs/run-1/samples/"' in ud
    # SAMPLE_PROMPTS is base64(JSON) — no raw quote can break the export line.
    env = bootstrap._trainer_env(_cfg(), run_id="run-1", market="on-demand", max_seconds=60)
    decoded = json.loads(base64.b64decode(env["SAMPLE_PROMPTS"]).decode())
    assert decoded == ["ROMEO:", "JULIET:", "First Citizen:"]
    assert '"' not in env["SAMPLE_PROMPTS"]


def test_trainer_env_rejects_malformed_prompts():
    cfg = _cfg()
    cfg.sample_prompts = "not-json"
    with pytest.raises(ValueError):
        bootstrap._trainer_env(cfg, run_id="run-1", market="on-demand", max_seconds=60)


def test_trainer_passthrough_only_when_set(monkeypatch):
    for var in ("MAX_STEPS", "LEARNING_RATE", "EVAL_INTERVAL_STEPS"):
        monkeypatch.delenv(var, raising=False)
    env = bootstrap._trainer_env(_cfg(), run_id="run-1", market="on-demand", max_seconds=60)
    assert "MAX_STEPS" not in env  # unset => trainer defaults untouched
    monkeypatch.setenv("MAX_STEPS", "5000")
    monkeypatch.setenv("LEARNING_RATE", "1e-3")
    monkeypatch.setenv("EVAL_INTERVAL_STEPS", "250")
    env = bootstrap._trainer_env(_cfg(), run_id="run-1", market="on-demand", max_seconds=60)
    assert env["MAX_STEPS"] == "5000"
    assert env["LEARNING_RATE"] == "1e-3"
    assert env["EVAL_INTERVAL_STEPS"] == "250"
    ud = _ud()
    assert 'export MAX_STEPS="5000"' in ud


def test_preempt_victim_schedule():
    cfg = _cfg()  # NODES=2, PREEMPT_COUNT=1 defaults
    assert cfg.preempt_victim_schedule() == [1]  # empty => last node (proven path)
    cfg.preempt_count = 3
    assert cfg.preempt_victim_schedule() == [1, 1, 1]
    # the staggered case: node 1 at t1, node 0 (master) at t2
    cfg.preempt_count = 2
    cfg.preempt_victims = "1,0"
    assert cfg.preempt_victim_schedule() == [1, 0]
    cfg.preempt_victims = " 1 , 0 "  # whitespace tolerated
    assert cfg.preempt_victim_schedule() == [1, 0]
    cfg.preempt_victims = "1"
    with pytest.raises(SystemExit, match="PREEMPT_COUNT"):
        cfg.preempt_victim_schedule()
    cfg.preempt_victims = "1,2"
    with pytest.raises(SystemExit, match="node indices"):
        cfg.preempt_victim_schedule()
    cfg.preempt_victims = "1,x"
    with pytest.raises(SystemExit, match="comma-separated"):
        cfg.preempt_victim_schedule()
