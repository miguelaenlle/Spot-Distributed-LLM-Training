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

Two tools answer "is this checkpoint comprehensive and valid?":
  - ``verify(ref)``     — loads it, checks the schema is complete and every
                          tensor is finite (catches NaN/inf and truncation).
  - ``smoke_test(...)`` — restores the weights into a fresh model and runs one
                          forward pass, asserting a finite loss (catches a file
                          that loads but is subtly wrong).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from typing import Any

import torch

from . import rng, s3_store

CKPT_VERSION = 1
_REQUIRED_KEYS = ("version", "step", "model", "optimizer", "rng", "loader")


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is incomplete, corrupt, or non-finite."""


def _ckpt_name(step: int) -> str:
    # zero-padded so lexicographic sort == numeric sort for `latest()`
    return f"{s3_store.CHECKPOINT_PREFIX}{step:012d}.pt"


def save(*, model, optimizer, loader, step: int, uri: str) -> str:
    """Atomically persist full training state. Returns the final checkpoint ref."""
    blob: dict[str, Any] = {
        "version": CKPT_VERSION,
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


def load_latest(uri: str, map_location: str = "cpu") -> dict[str, Any] | None:
    """Return the newest checkpoint blob under ``uri``, or None if none exists."""
    ref = s3_store.latest(uri)
    if ref is None:
        return None
    local = s3_store.download(ref)  # no-op for local refs; downloads+verifies S3
    return torch.load(local, map_location=map_location, weights_only=False)


def restore_into(blob: dict[str, Any], *, model, optimizer, loader) -> int:
    """Restore all state from ``blob``. Returns the step to resume from."""
    model.load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    loader.load_state_dict(blob["loader"])
    rng.restore(blob["rng"])
    return blob["step"]


# --------------------------------------------------------------------------- #
# Validation tools
# --------------------------------------------------------------------------- #
def _all_finite(state: Any) -> bool:
    """Recursively assert every floating-point tensor in a state tree is finite."""
    if isinstance(state, torch.Tensor):
        return (not state.is_floating_point()) or bool(torch.isfinite(state).all())
    if isinstance(state, dict):
        return all(_all_finite(v) for v in state.values())
    if isinstance(state, list | tuple):
        return all(_all_finite(v) for v in state)
    return True


def _verify_blob(blob: dict[str, Any]) -> dict[str, Any]:
    missing = [k for k in _REQUIRED_KEYS if k not in blob]
    if missing:
        raise CheckpointError(f"checkpoint missing keys: {missing}")
    if blob["version"] != CKPT_VERSION:
        raise CheckpointError(f"unsupported checkpoint version {blob['version']}")
    if not _all_finite(blob["model"]):
        raise CheckpointError("model weights contain NaN/inf")
    if not _all_finite(blob["optimizer"].get("state", {})):
        raise CheckpointError("optimizer state contains NaN/inf")
    return blob


def verify(ref: str, map_location: str = "cpu") -> dict[str, Any]:
    """Load ``ref`` and assert it is complete and finite. Returns the blob.

    ``ref`` may be a local path or an ``s3://`` URI (downloaded + checksum-checked).
    Raises :class:`CheckpointError` on any problem; ``torch.load`` itself raises on
    a truncated/corrupt file.
    """
    local = s3_store.download(ref)
    return _verify_blob(torch.load(local, map_location=map_location, weights_only=False))


def smoke_test(
    blob: dict[str, Any],
    build_model: Callable[[], Any],
    sample_batch: tuple,
    device: str,
) -> float:
    """Restore weights into a fresh model and run one forward pass.

    Confirms the saved state actually reconstructs a working model, not just a
    loadable file. Returns the (finite) loss; raises :class:`CheckpointError`
    otherwise.
    """
    model = build_model().to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    with torch.no_grad():
        _, loss = model(*sample_batch)
    if loss is None or not bool(torch.isfinite(loss)):
        raise CheckpointError("smoke test produced a non-finite loss")
    return float(loss.item())
