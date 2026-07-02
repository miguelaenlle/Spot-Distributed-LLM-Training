"""Orchestrator configuration.

All values have defaults except the S3 bucket, which you must set (it's globally
unique). Everything is overridable via environment variables so you can keep the
concrete names in your git-ignored ``.env`` rather than in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


@dataclass
class OrchestratorConfig:
    # --- AWS placement -------------------------------------------------------
    region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))
    instance_type: str = field(default_factory=lambda: _env("INSTANCE_TYPE", "g4dn.xlarge"))
    # Deep Learning AMI, resolved from this public SSM parameter unless AMI_ID is
    # set explicitly. DLAMI ships CUDA + PyTorch so user-data does no GPU setup.
    ami_id: str = field(default_factory=lambda: _env("AMI_ID", ""))
    ami_ssm_param: str = field(
        default_factory=lambda: _env(
            "AMI_SSM_PARAM",
            "/aws/service/deep-learning/ami/x86_64/"
            "base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id",
        )
    )

    # --- names created by `setup` (you own these) ---------------------------
    bucket: str = field(default_factory=lambda: _env("SPOT_TRAIN_BUCKET", ""))
    role_name: str = field(default_factory=lambda: _env("IAM_ROLE", "spot-train-role"))
    instance_profile: str = field(default_factory=lambda: _env("IAM_PROFILE", "spot-train-profile"))
    security_group: str = field(default_factory=lambda: _env("SECURITY_GROUP", "spot-train-sg"))

    # --- S3 key layout -------------------------------------------------------
    run_prefix: str = "runs"
    data_prefix: str = "data"

    # --- code delivery -------------------------------------------------------
    repo_url: str = field(
        default_factory=lambda: _env(
            "REPO_URL", "https://github.com/miguelaenlle/Spot-Distributed-LLM-Training.git"
        )
    )
    repo_branch: str = field(default_factory=lambda: _env("REPO_BRANCH", "main"))

    # --- experiment knobs ----------------------------------------------------
    dataset: str = field(default_factory=lambda: _env("DATASET", "shakespeare_char"))
    baseline_seconds: int = field(default_factory=lambda: _env_int("BASELINE_SECONDS", 300))
    spot_seg1_seconds: int = field(default_factory=lambda: _env_int("SPOT_SEG1_SECONDS", 120))
    spot_seg2_seconds: int = field(default_factory=lambda: _env_int("SPOT_SEG2_SECONDS", 180))
    checkpoint_interval_seconds: int = field(
        default_factory=lambda: _env_int("CHECKPOINT_INTERVAL_SECONDS", 30)
    )
    eval_iters: int = field(default_factory=lambda: _env_int("EVAL_ITERS", 200))
    batch_size: int = field(default_factory=lambda: _env_int("BATCH_SIZE", 12))

    # --- polling -------------------------------------------------------------
    metrics_poll_seconds: int = 15
    metrics_timeout_seconds: int = field(default_factory=lambda: _env_int("METRICS_TIMEOUT", 1800))

    # -- derived S3 locations ------------------------------------------------ #
    def data_uri(self) -> str:
        return f"s3://{self.bucket}/{self.data_prefix}/{self.dataset}/"

    def run_checkpoint_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_prefix}/{run_id}/checkpoints/"

    def run_metrics_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_prefix}/{run_id}/metrics.json"

    def run_metrics_key(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/metrics.json"

    def require_bucket(self) -> None:
        if not self.bucket:
            raise SystemExit(
                "No S3 bucket set. Put SPOT_TRAIN_BUCKET=<name> in your .env "
                "(see .env.example) and run `setup` first."
            )
