"""Full-state checkpoint: everything that affects the next training step.

Missing any one of these makes resume silently diverge:

    - model weights
    - optimizer state
    - step number
    - all RNG states           (see rng.py)
    - data-loader position     (see data.py)

Save path: serialize to a local temp file, then hand off to the store, which
renames atomically (see s3_store.save_atomic). A mid-write kill can only ever
leave a .tmp behind — never a corrupt "latest".
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Optional

import torch

from . import rng, s3_store


def _ckpt_name(step: int) -> str:
    # zero-padded so lexicographic sort == numeric sort for `latest()`
    return f"{s3_store.CHECKPOINT_PREFIX}{step:012d}.pt"


def save(
    *,
    model,
    optimizer,
    loader,
    step: int,
    uri: str,
) -> str:
    """Atomically persist full training state. Returns the final checkpoint ref."""
    blob: dict[str, Any] = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": rng.capture(),
        "loader": loader.state_dict(),
    }

    fd, tmp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(blob, tmp_path)
    return s3_store.save_atomic(tmp_path, uri, _ckpt_name(step))


def load_latest(uri: str, map_location: str = "cpu") -> Optional[dict[str, Any]]:
    """Return the newest checkpoint blob under ``uri``, or None if none exists."""
    ref = s3_store.latest(uri)
    if ref is None:
        return None
    # For S3 this will download to a temp file first (Phase 1a step 2).
    return torch.load(ref, map_location=map_location)


def restore_into(blob: dict[str, Any], *, model, optimizer, loader) -> int:
    """Restore all state from ``blob``. Returns the step to resume from."""
    model.load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    loader.load_state_dict(blob["loader"])
    rng.restore(blob["rng"])
    return blob["step"]
