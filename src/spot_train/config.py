"""Run configuration for Phase 1a (single node, single GPU/CPU).

The same config drives the local CPU determinism test and the remote spot box.
The box gets its values from environment variables (set by the orchestrator's
user-data script); locally the dataclass defaults are fine. Nothing here touches
AWS credentials — S3 locations are just strings.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field

# Default test prompts for end-of-run/interval sampling (shakespeare_char).
# "\n" stands in for an unconditional sample (GPT.generate needs a non-empty idx).
DEFAULT_SAMPLE_PROMPTS = ["ROMEO:", "JULIET:", "First Citizen:"]


def _env_float(name: str, default: float | None) -> float | None:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


def _env_prompts(name: str, default: list[str]) -> list[str]:
    """Prompt list from the environment: a JSON array, or base64 of one (the
    orchestrator relays base64 so shell quoting can't mangle the text)."""
    v = os.environ.get(name)
    if v in (None, ""):
        return list(default)
    try:
        return list(json.loads(v))
    except (json.JSONDecodeError, TypeError):
        return list(json.loads(base64.b64decode(v).decode()))


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


@dataclass
class TrainConfig:
    # --- model (passed through to nanoGPT's GPTConfig) ---
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.0

    # --- optimization ---
    max_steps: int = 100_000  # effectively "until the time budget runs out"
    batch_size: int = 12
    # Sequences per OPTIMIZER step, independent of how many GPUs are alive.
    # 0 = off (one micro-batch per rank, effective batch = world_size*batch_size,
    # today's behavior). >0 => each rank runs K = ceil(global/(world*batch))
    # micro-batches per step, so an elastic world-size change alters wall-clock
    # per step but NOT the gradient statistics or the LR schedule's validity.
    global_batch_size: int = 0
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    seed: int = 1337
    # LR schedule (nanoGPT-style linear warmup -> cosine decay -> min_lr). All a
    # pure function of the step number, so resume needs no extra state. Defaults
    # keep today's constant-LR behavior: lr_decay_steps == 0 disables the schedule.
    warmup_steps: int = 0
    lr_decay_steps: int = 0
    min_lr: float = 0.0
    # Gradient clipping (global norm); 0 disables (today's behavior).
    grad_clip: float = 0.0

    # --- time budget (the controllable duration) -----------------------------
    # Wall-clock seconds this launch may train before it stops, evaluates, and
    # writes metrics. None => run until max_steps. The orchestrator sets this
    # per launch (e.g. 300 for the baseline, 180 for the second spot segment).
    max_seconds: float | None = None
    # Run-level training budget across ALL launches/elastic restarts. The
    # checkpoint carries the seconds already trained; on resume max_seconds
    # becomes (train_budget_seconds - trained so far), clamped >= 1 so rank 0
    # can always re-form the group to eval + write metrics.json. None => use
    # max_seconds as-is (single-launch semantics).
    train_budget_seconds: float | None = None

    # --- checkpointing -------------------------------------------------------
    # Time-based, not step-based: bounds worst-case *wall-clock* lost work to
    # this interval regardless of how fast a step is. Some spot kills give no
    # warning, so we checkpoint on a clock, not on a signal.
    checkpoint_interval_seconds: float = 30.0
    # Node-local checkpoint tier (elastic multi-node): every node's LOCAL_RANK-0
    # keeps the latest snapshots on its own disk (DDP state is replicated, so
    # this needs no network), letting survivors of an elastic restart resume in
    # milliseconds instead of re-downloading from S3. "" = tier off (single-node
    # behavior unchanged). Rank 0 still uploads to S3 — the durable tier a fresh
    # replacement node resumes from.
    local_checkpoint_dir: str = ""
    # Run the (heavier) restore smoke test on every Nth checkpoint. 1 => always.
    smoke_test_every: int = 1
    # Periodic checkpoints from a background thread: only the point-in-time CPU
    # snapshot (~tens of ms) stays on the training critical path; serialize +
    # upload + verify run off it. Preempt and final checkpoints are always
    # synchronous. CHECKPOINT_ASYNC=0 restores the fully-synchronous behavior.
    checkpoint_async: bool = True
    # Print a per-step loss/throughput line every N steps (0 => silent). Tail the
    # box log over SSM to watch these live.
    log_interval_steps: int = 10
    # Where checkpoints live: a local dir for the CPU test, an s3:// URI on spot.
    checkpoint_uri: str = "checkpoints/"
    # Where the final metrics.json is written (local path or s3:// URI).
    metrics_uri: str = "checkpoints/metrics.json"

    # --- data ----------------------------------------------------------------
    dataset: str = "shakespeare_char"
    # Local dir the prepared bins/meta live in (or are downloaded to).
    data_local_dir: str = "third_party/nanoGPT/data/shakespeare_char"
    # If set (s3:// URI), the loader downloads train.bin/val.bin/meta.pkl from
    # here on first use. Empty => use whatever is already in data_local_dir.
    data_uri: str = ""
    eval_iters: int = 200

    # --- periodic evaluation (Gap E) ------------------------------------------
    # Every N steps, run the deterministic FULL pass over val.bin and print a
    # parseable `eval step S: val_loss X` line. 0 = off (default).
    eval_interval_steps: int = 0

    # --- text sampling ---------------------------------------------------------
    # Prompts sampled at the end of a graceful run (and, if sample_interval_steps
    # is set, at mid-training snapshots). Char-level via meta.pkl stoi/itos;
    # skipped gracefully for datasets without those maps.
    sample_prompts: list[str] = field(default_factory=lambda: list(DEFAULT_SAMPLE_PROMPTS))
    sample_max_new_tokens: int = 200
    sample_temperature: float = 0.8
    sample_top_k: int = 200
    samples_per_prompt: int = 1
    # Where the end-of-run consolidated samples.json goes (local path or s3://).
    samples_uri: str = "checkpoints/samples.json"
    # Mid-training snapshots: every N steps, generate a smaller batch of samples
    # (first sample_interval_prompts prompts × sample_interval_tokens tokens) and
    # write samples/step-<12d>.json under samples_prefix_uri. 0 = off (default).
    sample_interval_steps: int = 0
    sample_interval_prompts: int = 3
    sample_interval_tokens: int = 150
    samples_prefix_uri: str = "checkpoints/samples/"

    # --- eval / provenance ---------------------------------------------------
    run_id: str = "local"
    market: str = "local"  # "on-demand" | "spot" on the box

    # --- device --------------------------------------------------------------
    # "auto" -> cuda if the box has a GPU, else cpu (resolved at runtime, the way
    # ML training scripts normally do it). Set "cuda"/"cpu" to force.
    device: str = "auto"

    # --- DDP (torchrun) ------------------------------------------------------
    # "shard" => each rank seeds differently and draws different batches (real
    # data-parallel, effective batch = world_size*batch_size). "replicate" => all
    # ranks see identical data (bit-exact vs single-process; a plumbing check).
    data_mode: str = "shard"

    @classmethod
    def from_env(cls) -> TrainConfig:
        """Build a config from environment variables (used on the remote box).

        Falls back to the dataclass defaults for anything unset, so this is also
        safe to call locally.
        """
        return cls(
            max_steps=_env_int("MAX_STEPS", 100_000),
            learning_rate=_env_float("LEARNING_RATE", 6e-4),
            weight_decay=_env_float("WEIGHT_DECAY", 1e-1),
            dropout=_env_float("DROPOUT", 0.0),
            warmup_steps=_env_int("WARMUP_STEPS", 0),
            lr_decay_steps=_env_int("LR_DECAY_STEPS", 0),
            min_lr=_env_float("MIN_LR", 0.0),
            grad_clip=_env_float("GRAD_CLIP", 0.0),
            max_seconds=_env_float("MAX_SECONDS", None),
            train_budget_seconds=_env_float("TRAIN_BUDGET_SECONDS", None),
            checkpoint_interval_seconds=_env_float("CHECKPOINT_INTERVAL_SECONDS", 30.0),
            local_checkpoint_dir=_env_str("LOCAL_CHECKPOINT_DIR", ""),
            smoke_test_every=_env_int("SMOKE_TEST_EVERY", 1),
            checkpoint_async=_env_str("CHECKPOINT_ASYNC", "1").lower() not in ("0", "false"),
            log_interval_steps=_env_int("LOG_INTERVAL_STEPS", 10),
            checkpoint_uri=_env_str("CHECKPOINT_URI", "checkpoints/"),
            metrics_uri=_env_str("METRICS_URI", "checkpoints/metrics.json"),
            dataset=_env_str("DATASET", "shakespeare_char"),
            data_local_dir=_env_str("DATA_LOCAL_DIR", "third_party/nanoGPT/data/shakespeare_char"),
            data_uri=_env_str("DATA_URI", ""),
            eval_iters=_env_int("EVAL_ITERS", 200),
            eval_interval_steps=_env_int("EVAL_INTERVAL_STEPS", 0),
            sample_prompts=_env_prompts("SAMPLE_PROMPTS", DEFAULT_SAMPLE_PROMPTS),
            sample_max_new_tokens=_env_int("SAMPLE_MAX_NEW_TOKENS", 200),
            sample_temperature=_env_float("SAMPLE_TEMPERATURE", 0.8),
            sample_top_k=_env_int("SAMPLE_TOP_K", 200),
            samples_per_prompt=_env_int("SAMPLES_PER_PROMPT", 1),
            samples_uri=_env_str("SAMPLES_URI", "checkpoints/samples.json"),
            sample_interval_steps=_env_int("SAMPLE_INTERVAL_STEPS", 0),
            sample_interval_prompts=_env_int("SAMPLE_INTERVAL_PROMPTS", 3),
            sample_interval_tokens=_env_int("SAMPLE_INTERVAL_TOKENS", 150),
            samples_prefix_uri=_env_str("SAMPLES_PREFIX_URI", "checkpoints/samples/"),
            batch_size=_env_int("BATCH_SIZE", 12),
            global_batch_size=_env_int("GLOBAL_BATCH_SIZE", 0),
            run_id=_env_str("RUN_ID", "local"),
            market=_env_str("MARKET", "local"),
            device=_env_str("DEVICE", "auto"),
            data_mode=_env_str("DDP_DATA_MODE", "shard"),
        )
