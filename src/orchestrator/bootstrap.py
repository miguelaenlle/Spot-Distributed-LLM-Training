"""Build the EC2 user-data script that runs on the box at boot.

The box is an AWS Deep Learning AMI (Ubuntu) with CUDA + PyTorch preinstalled.
Two DLAMI realities shape these scripts:

  1. The PyTorch environment only auto-activates for the ``ubuntu`` **login**
     shell — root user-data would otherwise get a system python without torch.
     So we run everything via ``sudo -u ubuntu -i`` (login shell) and preflight
     ``import torch`` to fail loudly if the env is wrong.
  2. We run from source with ``PYTHONPATH`` instead of ``pip install -e .`` so we
     never try to write into a possibly root-owned framework env. (nanoGPT is
     found by the trainer relative to the package.)

Two builders:

  * ``build_provisioning_user_data`` — clone the repo + nanoGPT submodule and
    install deps, write a ``source``-able env file, then STOP and leave the box
    UP. You ssh in, confirm the code is there, and run training by hand. Used
    while bringing the box up.
  * ``build_user_data`` — the full run: provision, then ``spot_train.train``
    under a wall-clock budget, then self-terminate on success. Used once the box
    is proven. (Parked while we validate provisioning over SSH.)

The trainer pulls the dataset and reads/writes S3 via the instance profile role
— no credentials are passed in user-data.
"""

from __future__ import annotations

from .config import OrchestratorConfig


def _trainer_env(
    cfg: OrchestratorConfig, *, run_id: str, market: str, max_seconds: int
) -> dict[str, str]:
    """Environment the trainer reads via ``TrainConfig.from_env`` (spot_train
    config). Written to a ``source``-able file so you can run training by hand
    after sshing in, and exported inline for the full-run builder."""
    return {
        "PYTHONPATH": "/home/ubuntu/app/src",
        "CHECKPOINT_URI": cfg.run_checkpoint_uri(run_id),
        "METRICS_URI": cfg.run_metrics_uri(run_id),
        "DATA_URI": cfg.data_uri(),
        "DATASET": cfg.dataset,
        "DATA_LOCAL_DIR": f"/home/ubuntu/app/third_party/nanoGPT/data/{cfg.dataset}",
        "MAX_SECONDS": str(max_seconds),
        "CHECKPOINT_INTERVAL_SECONDS": str(cfg.checkpoint_interval_seconds),
        "SMOKE_TEST_EVERY": str(cfg.smoke_test_every),
        "EVAL_ITERS": str(cfg.eval_iters),
        "BATCH_SIZE": str(cfg.batch_size),
        "RUN_ID": run_id,
        "MARKET": market,
        "DEVICE": "auto",  # trainer auto-detects cuda vs cpu on the box
        "DDP_DATA_MODE": cfg.ddp_data_mode,  # only used when launched via torchrun
        "PYTHONUNBUFFERED": "1",  # unbuffered so `tail -f` shows per-step lines live
    }


def _export_block(env: dict[str, str]) -> str:
    return "\n".join(f'export {k}="{v}"' for k, v in env.items())


def _provision_steps(cfg: OrchestratorConfig, exports: str) -> str:
    """Setup run as the ubuntu login shell (torch env active): clone the repo,
    populate the nanoGPT submodule, install deps, write a source-able env file,
    and preflight. Shared by both builders."""
    return f"""set -euxo pipefail
cd /home/ubuntu

# Interpreter: the DLAMI base shell only ships `python3`; plain `python` exists
# only inside the pytorch venv, which the non-interactive login shell does NOT
# reliably auto-activate. Use the venv python by absolute path (it has torch +
# boto3 and always exists on this AMI); fall back to python3 on other images.
VENV_PY=/opt/pytorch/bin/python
[ -x "$VENV_PY" ] || VENV_PY="$(command -v python3)"

# Code: clone the repo (idempotent) and populate the nanoGPT submodule. A plain
# clone leaves third_party/nanoGPT EMPTY, and the trainer imports GPT from there,
# so the submodule init is not optional.
[ -d app ] || git clone --depth 1 -b {cfg.repo_branch} {cfg.repo_url} app
cd app
git submodule update --init --depth 1

# Deps: the DLAMI ships torch + numpy; boto3 is usually missing. Best-effort so a
# broken pip doesn't wedge the whole boot.
"$VENV_PY" -c "import boto3" 2>/dev/null \\
  || "$VENV_PY" -m pip install boto3 \\
  || "$VENV_PY" -m pip install --user boto3 || true

# Env: drop the trainer's config into a file you can `source` before a manual run.
cat > /home/ubuntu/spot-train.env <<'ENV'
{exports}
ENV

# Preflight: fail loudly IN THIS LOG if the torch/boto3 env is wrong.
"$VENV_PY" - <<'PY'
import torch, boto3, numpy  # noqa: F401
print("preflight ok: torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
PY
"""


def build_provisioning_user_data(
    cfg: OrchestratorConfig, *, run_id: str, market: str, max_seconds: int
) -> str:
    """Provision only: get the code + deps onto the box, then leave it UP for SSH.
    Does NOT run training and does NOT shut the box down."""
    env = _trainer_env(cfg, run_id=run_id, market=market, max_seconds=max_seconds)
    steps = _provision_steps(cfg, _export_block(env))
    return f"""#!/bin/bash
set -x
exec > /var/log/spot-train-boot.log 2>&1
echo "[provision] start"

sudo -u ubuntu -i bash <<'BOOT'
{steps}
echo "[provision] DONE"
echo "[provision] code: /home/ubuntu/app   env: /home/ubuntu/spot-train.env"
echo "[provision] to train by hand:"
echo "    cd /home/ubuntu/app && source /home/ubuntu/spot-train.env"
echo "    /opt/pytorch/bin/python -m spot_train.train"
BOOT
echo "[provision] user-data exited rc=$? — box left UP for SSH (no training, no shutdown)"
"""


def build_user_data(
    cfg: OrchestratorConfig,
    *,
    run_id: str,
    market: str,
    max_seconds: int,
    logs_key: str | None = None,
    nproc_per_node: int = 1,
) -> str:
    """Full run: provision, then train under the wall-clock budget while syncing
    the boot log to S3 every ``log_stream_seconds`` so the orchestrator can stream
    it live without SSH. On success the box self-terminates as a cost backstop (the
    orchestrator also terminates it once it sees metrics.json). On failure the box
    stays up for debugging; the orchestrator reaps it on the metrics timeout.

    ``nproc_per_node > 1`` runs the trainer under torchrun (single-node DDP); the
    trainer detects RANK and joins the process group. =1 is the plain single-process
    launch (baseline/preempt) — byte-identical to before."""
    env = _trainer_env(cfg, run_id=run_id, market=market, max_seconds=max_seconds)
    steps = _provision_steps(cfg, _export_block(env))
    bucket = cfg.bucket
    logs_key = logs_key or cfg.run_logs_key(run_id)
    interval = cfg.log_stream_seconds
    if nproc_per_node > 1:
        # OMP_NUM_THREADS=1: N gloo ranks on one box otherwise spawn N×cores threads
        # and thrash. torchrun --standalone sets RANK/LOCAL_RANK/WORLD_SIZE/MASTER_*.
        run_cmd = (
            f'OMP_NUM_THREADS=1 "$VENV_PY" -m torch.distributed.run '
            f"--standalone --nproc_per_node={nproc_per_node} -m spot_train.train"
        )
    else:
        run_cmd = '"$VENV_PY" -u -m spot_train.train'
    return f"""#!/bin/bash
set -x
exec > /var/log/spot-train-boot.log 2>&1
chmod 644 /var/log/spot-train-boot.log   # let the ubuntu log-uploader read it

sudo -u ubuntu -i bash <<'BOOT'
set -x
cd /home/ubuntu

# Interpreter: use the pytorch venv python by absolute path — the base shell only
# has `python3`, and the login shell doesn't reliably auto-activate the venv, so
# a bare `python` fails (command not found). Fall back to python3 on other AMIs.
VENV_PY=/opt/pytorch/bin/python
[ -x "$VENV_PY" ] || VENV_PY="$(command -v python3)"

# Start the S3 log uploader FIRST — before clone/pip — so the *entire* boot
# (clone, submodule, deps, preflight, then training) streams to the orchestrator,
# not just the training tail. boto3 is required for it, so ensure it up front
# (the provision steps below install it too; this is idempotent). Uses the
# instance-profile role via IMDS — no creds in user-data.
"$VENV_PY" -c "import boto3" 2>/dev/null || "$VENV_PY" -m pip install boto3 2>/dev/null \
  || "$VENV_PY" -m pip install --user boto3 2>/dev/null || true
"$VENV_PY" - <<'PY' &
import time, boto3
c = boto3.client("s3")
while True:
    try:
        c.upload_file("/var/log/spot-train-boot.log", "{bucket}", "{logs_key}")
    except Exception:
        pass
    time.sleep({interval})
PY
UPLOADER_PID=$!

# Provision (clone repo + nanoGPT submodule, deps, env file, preflight) — now streamed.
{steps}

# From here we manage exit codes by hand (final log flush + teardown), so drop -e.
set +e
# Load the run config the trainer reads (PYTHONPATH so `spot_train` imports, plus
# the S3 URIs, MAX_SECONDS, etc.). Provisioning wrote this file but doesn't source it.
source /home/ubuntu/spot-train.env
{run_cmd}
RC=$?

# Stop the uploader and push one final copy so the tail of the run is in S3.
kill "$UPLOADER_PID" 2>/dev/null || true
"$VENV_PY" - <<'PY'
import boto3
try:
    boto3.client("s3").upload_file("/var/log/spot-train-boot.log", "{bucket}", "{logs_key}")
except Exception:
    pass
PY
exit "$RC"
BOOT
RC=$?

if [ "$RC" -eq 0 ]; then
  shutdown -h now   # cost backstop; orchestrator also terminates on metrics.json
else
  echo "training exited $RC — leaving instance up for debugging; orchestrator reaps on timeout"
fi
"""
