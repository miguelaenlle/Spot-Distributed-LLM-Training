"""Run configuration for Phase 1a (single node, single GPU/CPU)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # --- model (passed through to nanoGPT's GPTConfig) ---
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.0

    # --- optimization ---
    max_steps: int = 5000
    batch_size: int = 12
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    seed: int = 1337

    # --- checkpointing ---
    # Save every N steps regardless of preemption signals — some spot kills
    # give no warning. This bounds worst-case lost work.
    checkpoint_interval: int = 100
    # Where checkpoints live. Local dir for the CPU test; an s3:// URI on spot.
    checkpoint_uri: str = "checkpoints/"

    # --- device ---
    device: str = "cpu"  # "cuda" on the spot box
