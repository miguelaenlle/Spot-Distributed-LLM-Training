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
        "EVAL_ITERS": str(cfg.eval_iters),
        "BATCH_SIZE": str(cfg.batch_size),
        "RUN_ID": run_id,
        "MARKET": market,
        "DEVICE": "auto",  # trainer auto-detects cuda vs cpu on the box
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

# Code: clone the repo (idempotent) and populate the nanoGPT submodule. A plain
# clone leaves third_party/nanoGPT EMPTY, and the trainer imports GPT from there,
# so the submodule init is not optional.
[ -d app ] || git clone --depth 1 -b {cfg.repo_branch} {cfg.repo_url} app
cd app
git submodule update --init --depth 1

# Deps: the DLAMI ships torch + numpy; boto3 is usually missing. Best-effort so a
# broken pip doesn't wedge the whole boot.
python -c "import boto3" 2>/dev/null \\
  || python -m pip install boto3 \\
  || python -m pip install --user boto3 || true

# Env: drop the trainer's config into a file you can `source` before a manual run.
cat > /home/ubuntu/spot-train.env <<'ENV'
{exports}
ENV

# Preflight: fail loudly IN THIS LOG if the torch/boto3 env is wrong.
python - <<'PY'
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
echo "    cd /home/ubuntu/app && source /home/ubuntu/spot-train.env && python -m spot_train.train"
BOOT
echo "[provision] user-data exited rc=$? — box left UP for SSH (no training, no shutdown)"
"""


def build_user_data(
    cfg: OrchestratorConfig,
    *,
    run_id: str,
    market: str,
    max_seconds: int,
) -> str:
    """Full run (PARKED while we validate provisioning): provision, train under
    the wall-clock budget, then self-terminate on success
    (InstanceInitiatedShutdownBehavior=terminate). On failure the box stays up
    for debugging; the orchestrator reaps it on the metrics timeout."""
    env = _trainer_env(cfg, run_id=run_id, market=market, max_seconds=max_seconds)
    steps = _provision_steps(cfg, _export_block(env))
    return f"""#!/bin/bash
set -x
exec > /var/log/spot-train-boot.log 2>&1

sudo -u ubuntu -i bash <<'BOOT'
{steps}
python -u -m spot_train.train
BOOT
RC=$?

if [ "$RC" -eq 0 ]; then
  shutdown -h now
else
  echo "training exited $RC — leaving instance up for debugging; orchestrator will reap on timeout"
fi
"""
