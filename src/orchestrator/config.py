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


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


# Trainer knobs the orchestrator relays verbatim (only when set in ITS
# environment): the convergence recipe + periodic eval/sample cadence. The
# orchestrator never branches on these values, so they stay untyped strings —
# TrainConfig.from_env parses them on the box.
_TRAINER_PASSTHROUGH = (
    "N_LAYER",
    "N_HEAD",
    "N_EMBD",
    "BLOCK_SIZE",
    "TARGET_LOSS",
    "MAX_STEPS",
    "GLOBAL_BATCH_SIZE",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "DROPOUT",
    "DTYPE",
    "DDP_COMM_HOOK",
    "WARMUP_STEPS",
    "LR_DECAY_STEPS",
    "MIN_LR",
    "GRAD_CLIP",
    "CHECKPOINT_ASYNC",
    "LOG_INTERVAL_STEPS",
    "EVAL_INTERVAL_STEPS",
    "SAMPLE_INTERVAL_STEPS",
    "SAMPLE_INTERVAL_PROMPTS",
    "SAMPLE_INTERVAL_TOKENS",
    "SAMPLE_MAX_NEW_TOKENS",
    "SAMPLE_TEMPERATURE",
    "SAMPLE_TOP_K",
    "SAMPLES_PER_PROMPT",
)

# vCPUs per instance type, for the quota-headroom gate. Only the types this
# project plausibly launches; anything else needs INSTANCE_VCPUS set explicitly.
_INSTANCE_VCPUS = {
    "g4dn.xlarge": 4,
    "g4dn.2xlarge": 8,
    "g4dn.4xlarge": 16,
    "g4dn.12xlarge": 48,
    "g5.xlarge": 4,
    "g5.2xlarge": 8,
    "g5.12xlarge": 48,
    "g6.xlarge": 4,
    "g6.12xlarge": 48,
}


# On-demand $/hr (us-east-1, Linux) for the cost ledger. Spot rates are NOT
# listed here — they move hourly and vary per AZ, so they're queried live at
# launch (aws.spot_hourly_rate). Types missing from this table need HOURLY_USD.
ON_DEMAND_HOURLY_USD = {
    "g4dn.xlarge": 0.526,
    "g4dn.2xlarge": 0.752,
    "g4dn.12xlarge": 3.912,
    "g5.xlarge": 1.006,
    "g6.xlarge": 0.805,
}


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

    # --- AMI baking (spot-orchestrate bake-ami) ------------------------------
    # Instance type for the throwaway bake box. pip installs are arch-independent
    # and the DLAMI boots fine without a GPU, so a cheap CPU box is the default —
    # it also keeps the bake entirely off the G-vCPU quota.
    bake_instance_type: str = field(default_factory=lambda: _env("BAKE_INSTANCE_TYPE", "t3.xlarge"))
    # Seconds to wait for the bake box's provisioning to write its status marker.
    bake_timeout_seconds: int = field(default_factory=lambda: _env_int("BAKE_TIMEOUT", 1200))
    # Baked AMIs to retain (newest first); older ones are deregistered and their
    # snapshots deleted after a successful bake. Snapshots bill monthly.
    bake_keep_images: int = field(default_factory=lambda: _env_int("BAKE_KEEP_IMAGES", 2))

    # --- experiment knobs ----------------------------------------------------
    dataset: str = field(default_factory=lambda: _env("DATASET", "shakespeare_char"))
    # $/hr override for the cost ledger: pins the on-demand rate when the
    # instance type isn't in ON_DEMAND_HOURLY_USD (or to correct it for another
    # region). 0 = use the table. Spot rows always use the live queried price.
    hourly_usd: float = field(default_factory=lambda: _env_float("HOURLY_USD", 0.0))
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

    # --- end-of-run + periodic text samples ----------------------------------
    # JSON array of prompts the trainer samples from at the end of a run (and at
    # SAMPLE_INTERVAL_STEPS snapshots). Relayed to the box base64-encoded so the
    # env file's export K="v" quoting can't be broken by quotes/newlines.
    sample_prompts: str = field(
        default_factory=lambda: _env("SAMPLE_PROMPTS", '["ROMEO:", "JULIET:", "First Citizen:"]')
    )

    # --- multinode-preempt victim schedule ------------------------------------
    # Comma-separated node index to hard-kill per preemption round, e.g. "1,0"
    # (kill node 1 first, then node 0). Any node is killable — the epoch after a
    # kill just names a new lowest-index master. Empty = always the last node.
    preempt_victims: str = field(default_factory=lambda: _env("PREEMPT_VICTIMS", ""))

    # --- DDP experiment (spot-orchestrate ddp) ------------------------------
    # Ranks torchrun launches on the box. 0 (default) = auto: one rank per GPU on
    # the machine (torchrun --nproc_per_node=gpu). Set a positive value to force a
    # fixed count — needed to exercise multi-rank DDP on a CPU-only box.
    ddp_nproc_per_node: int = field(default_factory=lambda: _env_int("DDP_NPROC_PER_NODE", 0))
    # "shard" (real data-parallel) | "replicate" (identical data, determinism check).
    ddp_data_mode: str = field(default_factory=lambda: _env("DDP_DATA_MODE", "shard"))

    # --- multi-node experiment (spot-orchestrate multinode) ------------------
    # Nodes in the training group; each runs torchrun with one rank per GPU.
    # The orchestrator owns membership: it publishes runs/<run_id>/epoch.json
    # (who is in the group, their ranks, the master addr/port) and every box's
    # sidecar polls it and runs STATIC torchrun for the current epoch. Node 0 of
    # an epoch is just the lowest live node index — no node hosts a rendezvous
    # store, so any node is killable.
    node_count: int = field(default_factory=lambda: _env_int("NODES", 2))
    # Base for the per-epoch master port: master_port = rdzv_port + epoch, so a
    # relaunched master never fights TIME_WAIT on its own previous socket.
    rdzv_port: int = field(default_factory=lambda: _env_int("RDZV_PORT", 29400))
    # Collective timeout exported to multi-node boxes (torch's default is 10
    # minutes). Under the epoch supervisor a survivor's torchrun is normally
    # killed by its sidecar the moment the shrink epoch lands (~3s), so this is
    # the IN-BAND BACKSTOP: if the supervisor is slow, the survivor's collective
    # still aborts here rather than hanging. 20s keeps >10x margin over the worst
    # legitimate stall at this model size (an async-checkpoint snapshot or a slow
    # TCP allreduce is well under 2s).
    nccl_timeout_seconds: int = field(default_factory=lambda: _env_int("NCCL_TIMEOUT", 20))
    # No checkpoint progress for this long -> the supervisor's whole-group
    # restart floor (terminate all, relaunch, publish a fresh epoch).
    # No-checkpoint-progress this long => the whole group is wedged (e.g. a torchrun
    # rendezvous that can't converge) => whole-group restart. The deadlock-breaker
    # of last resort. Must sit ABOVE the worst-case LEGITIMATE no-progress window
    # (a whole-group reboot: ~45-70s boot + restore + first checkpoint ~= 90s) so it
    # never false-fires mid-recovery, yet well under METRICS_TIMEOUT so a genuine
    # hang is broken within a run instead of stalling to the deadline.
    recovery_timeout_seconds: int = field(default_factory=lambda: _env_int("RECOVERY_TIMEOUT", 150))

    # --- inference fleet (ROADMAP Part 1) ------------------------------------
    # CPU instances by default: the 10M-param model serves fine on CPU, and
    # C/T-family spot draws on the "standard" spot quota, not the G quota.
    fleet_worker_count: int = field(default_factory=lambda: _env_int("FLEET_WORKERS", 4))
    fleet_worker_instance_type: str = field(
        default_factory=lambda: _env("FLEET_WORKER_INSTANCE_TYPE", "c7i.large")
    )
    fleet_router_instance_type: str = field(
        default_factory=lambda: _env("FLEET_ROUTER_INSTANCE_TYPE", "t3.small")
    )
    fleet_market: str = field(default_factory=lambda: _env("FLEET_MARKET", "spot"))
    fleet_router_port: int = field(default_factory=lambda: _env_int("FLEET_ROUTER_PORT", 8000))
    fleet_worker_port: int = field(default_factory=lambda: _env_int("FLEET_WORKER_PORT", 8001))
    # Who may reach the router's public port. Default is open (toy model, short
    # experiments); set FLEET_INGRESS_CIDR=<your-ip>/32 to tighten.
    fleet_ingress_cidr: str = field(default_factory=lambda: _env("FLEET_INGRESS_CIDR", "0.0.0.0/0"))

    # --- vCPU quota gate ------------------------------------------------------
    # The account's "Running On-Demand G and VT instances" vCPU quota. Launches
    # wait until running+pending G/VT usage leaves headroom under this before
    # calling RunInstances (no Service Quotas API — update this if AWS raises
    # your quota).
    vcpu_quota: int = field(default_factory=lambda: _env_int("VCPU_QUOTA", 8))
    # vCPUs of `instance_type`. 0 (default) = look up the builtin table; set
    # explicitly for instance types the table doesn't know.
    instance_vcpus: int = field(default_factory=lambda: _env_int("INSTANCE_VCPUS", 0))

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
    # Optional W&B group for a comparison suite (e.g. shakespeare-convergence);
    # empty keeps the historical group-by-market behavior.
    wandb_group: str = field(default_factory=lambda: _env("WANDB_GROUP", ""))

    # -- derived S3 locations ------------------------------------------------ #
    def data_uri(self) -> str:
        return f"s3://{self.bucket}/{self.data_prefix}/{self.dataset}/"

    def run_checkpoint_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_prefix}/{run_id}/checkpoints/"

    def run_metrics_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_prefix}/{run_id}/metrics.json"

    def run_metrics_key(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/metrics.json"

    # End-of-run consolidated text samples (trainer writes it just before
    # metrics.json, so it's always present when the done-signal appears).
    def run_samples_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_prefix}/{run_id}/samples.json"

    def run_samples_key(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/samples.json"

    # Mid-training inference snapshots (samples/step-<12-digit>.json), written
    # immediately at each SAMPLE_INTERVAL_STEPS gate so they survive preemption.
    def run_samples_prefix_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_prefix}/{run_id}/samples/"

    def run_samples_prefix(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/samples/"

    # The box's boot/training log, synced here every few seconds so the orchestrator
    # can stream it back without SSH. Preemption uses a per-segment key (seg-N.log)
    # so a fresh instance doesn't overwrite the previous segment's log; multi-node
    # adds a per-node suffix so the boxes don't clobber each other, and replacement
    # launches an attempt suffix (-rK) so they don't clobber the dead node's log.
    def run_logs_key(
        self,
        run_id: str,
        segment: int | None = None,
        node: int | None = None,
        attempt: int = 0,
    ) -> str:
        name = "boot" if segment is None else f"seg-{segment}"
        if node is not None:
            name += f"-node{node}"
        if attempt:
            name += f"-r{attempt}"
        return f"{self.run_prefix}/{run_id}/logs/{name}.log"

    def run_logs_uri(self, run_id: str, segment: int | None = None) -> str:
        return f"s3://{self.bucket}/{self.run_logs_key(run_id, segment)}"

    # Epoch protocol (see supervisor.py / sidecar.py). The orchestrator is the
    # ONLY writer of epoch.json — the membership document every box's sidecar
    # polls: {epoch, members:[{node,ip,rank}], node_count, master_addr,
    # master_port}. Each box registers node<i>.json {ip, instance_id} once at
    # boot; that registration is both the ready-marker and the join request
    # (admission = the orchestrator including the node in a published epoch).
    def run_epoch_key(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/epoch.json"

    def run_epoch_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_epoch_key(run_id)}"

    # Observability doc the supervisor rewrites every tick: per-(node, attempt)
    # liveness + log keys, so ANY process (the `logs` viewer) can discover which
    # log belongs to whom and whether that box is alive — without the driver's
    # in-memory state. Same single writer as epoch.json; sidecars never read it.
    def run_status_key(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/status.json"

    def run_status_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_status_key(run_id)}"

    # The supervisor's OWN decision log (published epoch / terminated / relaunch),
    # uploaded next to the box logs so the viewer can show the control plane's
    # narrative as a tab too.
    def run_orch_log_key(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/logs/orchestrator.log"

    def run_logs_prefix(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/logs/"

    def run_nodes_prefix(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/nodes/"

    def run_node_key(self, run_id: str, node: int) -> str:
        return f"{self.run_prefix}/{run_id}/nodes/node{node}.json"

    def run_node_uri(self, run_id: str, node: int) -> str:
        return f"s3://{self.bucket}/{self.run_node_key(run_id, node)}"

    def run_uri(self, run_id: str) -> str:
        """s3://bucket/runs/<run_id> — the base the box sidecar is pointed at."""
        return f"s3://{self.bucket}/{self.run_prefix}/{run_id}"

    # Inference-fleet keys: heartbeat docs the router polls, per-box boot logs,
    # and a state doc recording which instances belong to the fleet.
    def fleet_workers_uri(self, fleet_id: str) -> str:
        return f"s3://{self.bucket}/fleet/{fleet_id}/workers/"

    def fleet_logs_key(self, fleet_id: str, name: str) -> str:
        return f"fleet/{fleet_id}/logs/{name}.log"

    def fleet_state_key(self, fleet_id: str) -> str:
        return f"fleet/{fleet_id}/fleet.json"

    # AMI-bake control keys: the bake box writes status.json (ok/rc/commit) when
    # provisioning finishes and streams its boot log next to it.
    def bake_status_key(self, bake_id: str) -> str:
        return f"bake/{bake_id}/status.json"

    def bake_log_key(self, bake_id: str) -> str:
        return f"bake/{bake_id}/bake.log"

    def on_demand_hourly_usd(self) -> float | None:
        """$/hr for on-demand ledger rows: HOURLY_USD override, else the table.
        None => unknown (the ledger row is kept but flagged, cost sums skip it)."""
        if self.hourly_usd:
            return self.hourly_usd
        return ON_DEMAND_HOURLY_USD.get(self.instance_type)

    # The tool-agnostic run profile (timeline + loss + merged metrics) the
    # orchestrator writes at end of run. W&B is just a mirror of this.
    def run_profile_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_prefix}/{run_id}/profile.json"

    # The cost graph (cumulative $ + loss-per-dollar) rendered at finalize.
    def run_cost_png_uri(self, run_id: str) -> str:
        return f"s3://{self.bucket}/{self.run_prefix}/{run_id}/cost.png"

    def run_profile_key(self, run_id: str) -> str:
        return f"{self.run_prefix}/{run_id}/profile.json"

    def wandb_enabled(self) -> bool:
        """W&B mirror is on iff an API key is present (loaded from .env) and not
        explicitly disabled. Absent key => S3 profile.json only, no third party."""
        if os.environ.get("WANDB_DISABLED", "") in ("1", "true", "True"):
            return False
        return bool(os.environ.get("WANDB_API_KEY"))

    def instance_vcpu_count(self) -> int:
        """vCPUs one `instance_type` box consumes against the G/VT quota."""
        if self.instance_vcpus > 0:
            return self.instance_vcpus
        try:
            return _INSTANCE_VCPUS[self.instance_type]
        except KeyError:
            raise SystemExit(
                f"Unknown vCPU count for instance type {self.instance_type!r} — "
                "set INSTANCE_VCPUS=<n> in your .env so the quota gate can count it."
            ) from None

    def trainer_passthrough(self) -> dict[str, str]:
        """Recipe/cadence env vars relayed to the box verbatim — only the ones
        actually set here, so an unset knob keeps the trainer's own default."""
        return {k: os.environ[k] for k in _TRAINER_PASSTHROUGH if os.environ.get(k)}

    def preempt_victim_schedule(self) -> list[int]:
        """Node index to kill per preemption round. Empty PREEMPT_VICTIMS keeps
        the proven default (always the last node); otherwise one index per round,
        each in [0, node_count) — 0 (the master) is allowed."""
        raw = self.preempt_victims.strip()
        if not raw:
            return [self.node_count - 1] * self.preempt_count
        try:
            victims = [int(v) for v in raw.split(",")]
        except ValueError:
            raise SystemExit(
                f"PREEMPT_VICTIMS={raw!r} — must be comma-separated node indices"
            ) from None
        if len(victims) != self.preempt_count:
            raise SystemExit(
                f"PREEMPT_VICTIMS has {len(victims)} entries but PREEMPT_COUNT is "
                f"{self.preempt_count} — one victim per kill round"
            )
        bad = [v for v in victims if not 0 <= v < self.node_count]
        if bad:
            raise SystemExit(
                f"PREEMPT_VICTIMS contains {bad} — node indices must be in [0, {self.node_count})"
            )
        return victims

    def require_bucket(self) -> None:
        if not self.bucket:
            raise SystemExit(
                "No S3 bucket set. Put SPOT_TRAIN_BUCKET=<name> in your .env "
                "(see .env.example) and run `setup` first."
            )
