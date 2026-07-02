"""Build the EC2 user-data script that runs training on the box.

The box is an AWS Deep Learning AMI (Ubuntu) with CUDA + PyTorch preinstalled.
Two DLAMI realities shape this script:

  1. The PyTorch environment only auto-activates for the ``ubuntu`` **login**
     shell — root user-data would otherwise get a system python without torch.
     So we run the whole job via ``sudo -u ubuntu -i`` (login shell) and
     preflight ``import torch`` to fail loudly if the env is wrong.
  2. We run from source with ``PYTHONPATH`` instead of ``pip install -e .`` so we
     never try to write into a possibly root-owned framework env. (nanoGPT is
     found by the trainer relative to the package.)

The trainer pulls the dataset and reads/writes S3 via the instance profile role
— no credentials are passed in user-data. On success the box shuts down (and,
with InstanceInitiatedShutdownBehavior=terminate, self-terminates). On failure
it stays up so you can SSM in and read the log; the orchestrator terminates it
on its metrics timeout regardless, so nothing is left billing indefinitely.
"""

from __future__ import annotations

from .config import OrchestratorConfig


def build_user_data(
    cfg: OrchestratorConfig,
    *,
    run_id: str,
    market: str,
    max_seconds: int,
) -> str:
    env = {
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
        # the trainer auto-detects cuda vs cpu from the box at runtime
        "DEVICE": "auto",
        # unbuffered stdout so `tail -f` over SSM shows per-step lines live
        "PYTHONUNBUFFERED": "1",
    }
    exports = "\n".join(f'export {k}="{v}"' for k, v in env.items())
    # Inner block runs as the ubuntu login shell (torch env auto-activates).
    return f"""#!/bin/bash
set -x
exec > /var/log/spot-train-boot.log 2>&1

sudo -u ubuntu -i bash <<'BOOT'
set -euxo pipefail
cd /home/ubuntu
[ -d app ] || git clone --depth 1 -b {cfg.repo_branch} {cfg.repo_url} app
cd app
git submodule update --init --depth 1

{exports}

# DLAMI ships torch + numpy; boto3 may be missing. Best-effort install, then
# preflight so a broken env fails loudly in this log instead of mysteriously.
python -c "import boto3" 2>/dev/null \
  || python -m pip install boto3 \
  || python -m pip install --user boto3 || true
python - <<'PY'
import torch, boto3, numpy  # noqa: F401
print("preflight ok: torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
PY

python -u -m spot_train.train
BOOT
RC=$?

if [ "$RC" -eq 0 ]; then
  shutdown -h now
else
  echo "training exited $RC — leaving instance up for debugging; orchestrator will reap on timeout"
fi
"""
