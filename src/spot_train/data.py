"""Data loader that tracks and restores its position.

The loader's position is part of the training state. If we resume and start
re-reading from a different offset (or a re-shuffled order), the resumed run
sees a different data stream than the uninterrupted one and the loss diverges
even with weights/optimizer/RNG restored correctly.

Phase 1a uses nanoGPT's simple memmap batch scheme (data/<dataset>/train.bin),
so "position" is just the RNG-driven batch index sequence — we make it explicit
and restorable here rather than relying on global RNG alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LoaderState:
    """Everything needed to resume the data stream exactly."""

    step: int = 0  # batches served so far
    epoch: int = 0
    # Extend as needed (e.g. shard index, within-shard offset) for larger data.


class PositionedLoader:
    """Yields batches while tracking a restorable position.

    NOTE: intentionally a scaffold. The kill-and-resume test drives the
    contract: ``state_dict()`` at step N, restore into a fresh loader, and the
    next batch must equal what the uninterrupted loader would have produced.
    """

    def __init__(self, dataset_dir: str, batch_size: int, block_size: int, device: str):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.state = LoaderState()

    def next_batch(self):
        """Return the next (x, y) batch and advance position."""
        raise NotImplementedError("Phase 1a: wire to nanoGPT memmap get_batch")

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.state.step, "epoch": self.state.epoch}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.state = LoaderState(step=sd["step"], epoch=sd["epoch"])
