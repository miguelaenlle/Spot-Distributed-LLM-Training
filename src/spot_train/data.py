"""Data loader over nanoGPT's memmap dataset, with an S3-pull step.

The loader's position is part of the training state. nanoGPT draws each batch
from random offsets via the global torch RNG (``third_party/nanoGPT/train.py``
``get_batch``), so "position" is really the torch RNG stream — which
``rng.py`` captures — plus a step counter we keep here for logging.

On the spot box the prepared bins are pulled once from S3
(``data_uri``); locally they are produced by nanoGPT's ``prepare.py`` and read
straight from ``data_local_dir``. Same ``next_batch`` either way.
"""

from __future__ import annotations

import os
import pickle
import shutil
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from . import s3_store

_FILES = ("train.bin", "val.bin", "meta.pkl")


@dataclass
class LoaderState:
    """Everything needed to resume the data stream (alongside the RNG state)."""

    step: int = 0  # train batches served so far
    epoch: int = 0


class PositionedLoader:
    """Yields (x, y) batches while tracking a restorable position."""

    def __init__(
        self,
        data_local_dir: str,
        batch_size: int,
        block_size: int,
        device: str,
        data_uri: str = "",
    ):
        self.data_local_dir = data_local_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.data_uri = data_uri
        self.state = LoaderState()
        self._ensure_data()
        self.vocab_size = self._read_vocab_size()

    # -- data provisioning -------------------------------------------------- #
    def _ensure_data(self) -> None:
        """Make sure train/val/meta exist locally, pulling from S3 if configured.

        Multi-rank safe: torchrun starts every rank on the box at once and each
        constructs a loader, so on a multi-GPU node N ranks would race to
        download and overwrite the same files. Only LOCAL_RANK 0 downloads —
        each file lands via a same-directory temp + atomic os.replace, so a
        file's *existence* implies it is complete — while the other local ranks
        just wait for the files to appear."""
        os.makedirs(self.data_local_dir, exist_ok=True)
        missing = [f for f in _FILES if not os.path.exists(os.path.join(self.data_local_dir, f))]
        if not missing:
            return
        if not self.data_uri:
            raise FileNotFoundError(
                f"Dataset files {missing} not found in {self.data_local_dir!r} and no "
                f"data_uri set. Run nanoGPT's prepare.py (locally) or `spot-orchestrate "
                f"stage-data` (to S3) first."
            )
        if int(os.environ.get("LOCAL_RANK", "0")) != 0:
            self._wait_for_files(missing)
            return
        for name in missing:
            ref = s3_store._join(self.data_uri.rstrip("/"), name)  # s3://.../<name>
            local = s3_store.download(ref)
            dest = os.path.join(self.data_local_dir, name)
            tmp = f"{dest}.tmp-{os.getpid()}"
            shutil.move(local, tmp)  # may be a cross-fs copy — but into dest's dir
            os.replace(tmp, dest)  # atomic: waiters can trust existence

    def _wait_for_files(self, names: list[str], timeout: float = 600.0) -> None:
        """Non-downloading local ranks block here until rank 0's files land."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(os.path.exists(os.path.join(self.data_local_dir, n)) for n in names):
                return
            time.sleep(1.0)
        raise TimeoutError(
            f"waited {timeout:.0f}s for {names} in {self.data_local_dir!r} "
            "(is the downloading rank 0 on this box stalled?)"
        )

    def _read_vocab_size(self) -> int | None:
        meta = os.path.join(self.data_local_dir, "meta.pkl")
        if not os.path.exists(meta):
            return None
        with open(meta, "rb") as f:
            return pickle.load(f).get("vocab_size")

    # -- batching (mirrors nanoGPT get_batch) ------------------------------- #
    def next_batch(self, split: str = "train"):
        """Return the next (x, y) batch and advance position for train batches."""
        path = os.path.join(self.data_local_dir, f"{split}.bin")
        # Re-open the memmap each call — nanoGPT does this to avoid a leak.
        data = np.memmap(path, dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack(
            [torch.from_numpy(data[i : i + self.block_size].astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy(data[i + 1 : i + 1 + self.block_size].astype(np.int64)) for i in ix]
        )
        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        if split == "train":
            self.state.step += 1
        return x, y

    def iter_eval_batches(self, split: str = "val"):
        """Deterministic FULL pass over ``split``: consecutive non-overlapping
        block_size windows batched batch_size at a time (remainder windows kept
        as a final short batch). Draws NOTHING from any RNG — the training data
        stream is bit-identical whether or not an eval ran — and yields the same
        batches every call, so eval losses are exactly comparable across steps
        and across runs."""
        path = os.path.join(self.data_local_dir, f"{split}.bin")
        data = np.memmap(path, dtype=np.uint16, mode="r")
        n_windows = (len(data) - 1) // self.block_size
        for start in range(0, n_windows, self.batch_size):
            offsets = [
                w * self.block_size for w in range(start, min(start + self.batch_size, n_windows))
            ]
            x = torch.stack(
                [torch.from_numpy(data[i : i + self.block_size].astype(np.int64)) for i in offsets]
            )
            y = torch.stack(
                [
                    torch.from_numpy(data[i + 1 : i + 1 + self.block_size].astype(np.int64))
                    for i in offsets
                ]
            )
            if self.device.startswith("cuda"):
                x = x.pin_memory().to(self.device, non_blocking=True)
                y = y.pin_memory().to(self.device, non_blocking=True)
            else:
                x, y = x.to(self.device), y.to(self.device)
            yield x, y

    # -- resumable position ------------------------------------------------- #
    def state_dict(self) -> dict[str, Any]:
        return {"step": self.state.step, "epoch": self.state.epoch}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.state = LoaderState(step=sd["step"], epoch=sd["epoch"])
