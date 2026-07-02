"""Run configuration for Phase 1a (single node, single GPU/CPU).

The same config drives the local CPU determinism test and the remote spot box.
The box gets its values from environment variables (set by the orchestrator's
user-data script); locally the dataclass defaults are fine. Nothing here touches
AWS credentials — S3 locations are just strings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float | None) -> float | None:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


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
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    seed: int = 1337

    # --- time budget (the controllable duration) -----------------------------
    # Wall-clock seconds this launch may train before it stops, evaluates, and
    # writes metrics. None => run until max_steps. The orchestrator sets this
    # per launch (e.g. 300 for the baseline, 180 for the second spot segment).
    max_seconds: float | None = None

    # --- checkpointing -------------------------------------------------------
    # Time-based, not step-based: bounds worst-case *wall-clock* lost work to
    # this interval regardless of how fast a step is. Some spot kills give no
    # warning, so we checkpoint on a clock, not on a signal.
    checkpoint_interval_seconds: float = 30.0
    # Run the (heavier) restore smoke test on every Nth checkpoint. 1 => always.
    smoke_test_every: int = 1
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

    # --- eval / provenance ---------------------------------------------------
    run_id: str = "local"
    market: str = "local"  # "on-demand" | "spot" on the box

    # --- device --------------------------------------------------------------
    device: str = "cpu"  # "cuda" on the spot box

    @classmethod
    def from_env(cls) -> TrainConfig:
        """Build a config from environment variables (used on the remote box).

        Falls back to the dataclass defaults for anything unset, so this is also
        safe to call locally.
        """
        return cls(
            max_seconds=_env_float("MAX_SECONDS", None),
            checkpoint_interval_seconds=_env_float("CHECKPOINT_INTERVAL_SECONDS", 30.0),
            log_interval_steps=_env_int("LOG_INTERVAL_STEPS", 10),
            checkpoint_uri=_env_str("CHECKPOINT_URI", "checkpoints/"),
            metrics_uri=_env_str("METRICS_URI", "checkpoints/metrics.json"),
            dataset=_env_str("DATASET", "shakespeare_char"),
            data_local_dir=_env_str("DATA_LOCAL_DIR", "third_party/nanoGPT/data/shakespeare_char"),
            data_uri=_env_str("DATA_URI", ""),
            eval_iters=_env_int("EVAL_ITERS", 200),
            batch_size=_env_int("BATCH_SIZE", 12),
            run_id=_env_str("RUN_ID", "local"),
            market=_env_str("MARKET", "local"),
            device=_env_str("DEVICE", "cpu"),
        )
