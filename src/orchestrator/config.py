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
    # Deep Learning AMI. If AMI_ID is set we use it verbatim; otherwise we resolve
    # the newest Amazon-owned image matching this name filter via DescribeImages.
    # Default targets the PyTorch DLAMI (Ubuntu 22.04) so CUDA + PyTorch are
    # preinstalled and user-data does no GPU/torch setup.
    ami_id: str = field(default_factory=lambda: _env("AMI_ID", ""))
    ami_name_filter: str = field(
        default_factory=lambda: _env(
            "AMI_NAME_FILTER",
            "Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu*",
        )
    )

    # SSH-verification mode: name of an EXISTING EC2 key pair in `region` to
    # attach so you can ssh into the box. Blank = launch without SSH access.
    key_name: str = field(default_factory=lambda: _env("SSH_KEY_NAME", ""))

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

    # Market the spot-style experiments (spot/preempt/ddp-preempt) launch in.
    # MARKET=on-demand runs the same kill/resume mechanics on on-demand capacity —
    # useful when the spot vCPU quota is exhausted. baseline/ddp are always on-demand.
    spot_market: str = field(default_factory=lambda: _env("MARKET", "spot"))

    # --- preemption experiment ----------------------------------------------
    # Total TRAINING seconds to accumulate across all segments (kills don't count).
    train_total_seconds: int = field(default_factory=lambda: _env_int("TRAIN_TOTAL_SECONDS", 180))
    # Number of preemptions to perform. The total training is split evenly across
    # (preempt_count + 1) segments — so 1 => train, kill once, reboot, finish. The
    # node is NOT told the schedule; it only gets its remaining budget as MAX_SECONDS.
    preempt_count: int = field(default_factory=lambda: _env_int("PREEMPT_COUNT", 1))
    # Seconds to wait for the trainer's SIGTERM checkpoint to land before terminating.
    preempt_grace_seconds: int = field(default_factory=lambda: _env_int("PREEMPT_GRACE", 90))
    # Seconds of training before each kill. 0 (default) = split train_total_seconds
    # evenly across segments. Set small (e.g. PREEMPT_AFTER=15) to exercise the
    # kill/resume path fast while debugging; the number of kills stays preempt_count.
    preempt_after_seconds: int = field(default_factory=lambda: _env_int("PREEMPT_AFTER", 0))
    # Small checkpoint interval during preemption so training-start is detectable fast
    # (graceful SIGTERM also checkpoints, so lost work is ~0 regardless).
    preempt_checkpoint_seconds: int = field(
        default_factory=lambda: _env_int("PREEMPT_CHECKPOINT_SECONDS", 5)
    )
    # How often the trainer runs the (noisy) checkpoint verify+smoke test. Set per
    # experiment so frequent preemption checkpoints don't flood the loss output.
    smoke_test_every: int = field(default_factory=lambda: _env_int("SMOKE_TEST_EVERY", 1))

    # --- DDP experiment (spot-orchestrate ddp) ------------------------------
    # Ranks torchrun launches on the box. 0 (default) = auto: one rank per GPU on
    # the machine (torchrun --nproc_per_node=gpu). Set a positive value to force a
    # fixed count — needed to exercise multi-rank DDP on a CPU-only box.
    ddp_nproc_per_node: int = field(default_factory=lambda: _env_int("DDP_NPROC_PER_NODE", 0))
    # "shard" (real data-parallel) | "replicate" (identical data, determinism check).
    ddp_data_mode: str = field(default_factory=lambda: _env("DDP_DATA_MODE", "shard"))

    # --- multi-node experiment (spot-orchestrate multinode) ------------------
    # Nodes in the training group; each runs torchrun with one rank per GPU.
    # Node 0 hosts the c10d rendezvous store and publishes its private IP to S3
    # (runs/<run_id>/rdzv.json); the other nodes poll that key before starting.
    node_count: int = field(default_factory=lambda: _env_int("NODES", 2))
    rdzv_port: int = field(default_factory=lambda: _env_int("RDZV_PORT", 29400))
    # Collective timeout exported to multi-node boxes so survivors' collectives
    # abort fast when a peer node dies (torch's default is 10 minutes).
    nccl_timeout_seconds: int = field(default_factory=lambda: _env_int("NCCL_TIMEOUT", 60))

    # --- polling -------------------------------------------------------------
    metrics_poll_seconds: int = 15
    metrics_timeout_seconds: int = field(default_factory=lambda: _env_int("METRICS_TIMEOUT", 1800))
    # How often the orchestrator pulls the box's boot log from S3 to print new
    # lines. Smaller than the metrics poll — this drives the live view latency.
    log_stream_seconds: int = field(default_factory=lambda: _env_int("LOG_STREAM_SECONDS", 3))

    # --- visualization (optional, Weights & Biases) -------------------------
    # Logging happens on the ORCHESTRATOR only; spot boxes never see the key.
    wandb_project: str = field(default_factory=lambda: _env("WANDB_PROJECT", "spot-train"))
    wandb_entity: str = field(default_factory=lambda: _env("WANDB_ENTITY", ""))

    # -- derived S3 locations ------------------------------------------------ #
    def data_uri(self) -> str:
        return f"s3://{self.bucket}/{self.data_prefix}/{self.dataset}/"

    def run_checkpoint_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_prefix}/{run_id}/checkpoints/"

    def run_metrics_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_prefix}/{run_id}/metrics.json"

    def run_metrics_key(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/metrics.json"

    # The box's boot/training log, synced here every few seconds so the orchestrator
    # can stream it back without SSH. Preemption uses a per-segment key (seg-N.log)
    # so a fresh instance doesn't overwrite the previous segment's log; multi-node
    # adds a per-node suffix so the boxes don't clobber each other.
    def run_logs_key(self, run_id: str, segment: int | None = None, node: int | None = None) -> str:
        name = "boot" if segment is None else f"seg-{segment}"
        if node is not None:
            name += f"-node{node}"
        return f"{self.run_prefix}/{run_id}/logs/{name}.log"

    def run_logs_uri(self, run_id: str, segment: int | None = None) -> str:
        return f"s3://{self.bucket}/{self.run_logs_key(run_id, segment)}"

    # Multi-node rendezvous bootstrap: node 0 writes its private IP here at boot;
    # the other nodes poll it, then all run torchrun against node 0's TCPStore.
    def run_rdzv_key(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/rdzv.json"

    # The tool-agnostic run profile (timeline + loss + merged metrics) the
    # orchestrator writes at end of run. W&B is just a mirror of this.
    def run_profile_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_prefix}/{run_id}/profile.json"

    def run_profile_key(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/profile.json"

    def wandb_enabled(self) -> bool:
        """W&B mirror is on iff an API key is present (loaded from .env) and not
        explicitly disabled. Absent key => S3 profile.json only, no third party."""
        if os.environ.get("WANDB_DISABLED", "") in ("1", "true", "True"):
            return False
        return bool(os.environ.get("WANDB_API_KEY"))

    def require_bucket(self) -> None:
        if not self.bucket:
            raise SystemExit(
                "No S3 bucket set. Put SPOT_TRAIN_BUCKET=<name> in your .env "
                "(see .env.example) and run `setup` first."
            )
