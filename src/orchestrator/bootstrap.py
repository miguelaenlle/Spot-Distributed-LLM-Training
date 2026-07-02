"""Build the EC2 user-data script that runs training on the box.

The box is an AWS Deep Learning AMI (CUDA + PyTorch already installed), so
user-data just fetches our code, installs the package without touching torch,
and launches the trainer with the run's env vars. The trainer pulls the dataset
and reads/writes checkpoints from S3 via the instance profile's role — no creds
are ever passed in user-data. ``shutdown -h now`` at the end terminates the
instance (InstanceInitiatedShutdownBehavior=terminate) so nothing is left
billing if the orchestrator misses it.
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
        "DEVICE": "cuda",
    }
    exports = "\n".join(f'export {k}="{v}"' for k, v in env.items())
    return f"""#!/bin/bash
set -euxo pipefail
exec > /var/log/spot-train-boot.log 2>&1

cd /home/ubuntu
if [ ! -d app ]; then
  git clone --depth 1 -b {cfg.repo_branch} {cfg.repo_url} app
fi
cd app
git submodule update --init --depth 1

# DLAMI already has torch + CUDA; install our package without disturbing them.
pip install --no-deps -e .
pip install boto3 numpy

{exports}

python -m spot_train.train

shutdown -h now
"""
