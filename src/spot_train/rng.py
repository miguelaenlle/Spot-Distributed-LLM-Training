"""Capture and restore *all* RNG states.

This is one of the two things (with data-loader position) that keep a resumed
run from silently diverging from an uninterrupted one. If any RNG source is
missed, the loss curve after resume drifts and the kill-and-resume test fails.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def capture() -> dict[str, Any]:
    """Snapshot every RNG source that influences training."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore(state: dict[str, Any]) -> None:
    """Restore RNG sources captured by :func:`capture`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
