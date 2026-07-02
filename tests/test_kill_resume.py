"""Checkpoint/resume unit tests — hermetic (tiny model, local store, no network).

The full spot kill-and-resume is validated end-to-end against nanoGPT + AWS
separately (see CLAUDE.md verification). These tests pin the pieces that make
resume correct and safe: RNG round-trip, full-state save/restore, checkpoint
validation (`verify`), and the restore smoke test.
"""

from __future__ import annotations

import torch
from torch import nn

from spot_train import checkpoint, rng, s3_store


class Tiny(nn.Module):
    """Minimal model with nanoGPT's ``forward(idx, targets) -> (logits, loss)``."""

    def __init__(self):
        super().__init__()
        self.l = nn.Linear(4, 4)

    def forward(self, x, y=None):
        out = self.l(x)
        loss = ((out - y) ** 2).mean() if y is not None else None
        return out, loss


class FakeLoader:
    def __init__(self):
        self.state = {"step": 0, "epoch": 0}

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, sd):
        self.state = dict(sd)


def _optim(model):
    return torch.optim.SGD(model.parameters(), lr=0.1)


def test_rng_roundtrip():
    """RNG capture/restore is exact — the cheapest half of determinism."""
    import random

    state = rng.capture()
    a = [random.random() for _ in range(5)]
    rng.restore(state)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_save_verify_restore(tmp_path):
    """Full-state round-trip: save → verify → restore into a fresh model matches."""
    uri = str(tmp_path)
    model = Tiny()
    opt = _optim(model)
    loader = FakeLoader()
    loader.state["step"] = 7

    # take an optimizer step so optimizer state is non-trivial
    _, loss = model(torch.randn(3, 4), torch.randn(3, 4))
    loss.backward()
    opt.step()

    ref = checkpoint.save(model=model, optimizer=opt, loader=loader, step=7, uri=uri)
    assert s3_store.latest(uri) == ref

    verified = checkpoint.verify(ref)  # keys + finite; raises on any problem
    assert verified["step"] == 7 and verified["version"] == checkpoint.CKPT_VERSION

    fresh, fresh_opt, fresh_loader = Tiny(), None, FakeLoader()
    fresh_opt = _optim(fresh)
    blob = checkpoint.load_latest(uri)
    step = checkpoint.restore_into(blob, model=fresh, optimizer=fresh_opt, loader=fresh_loader)
    assert step == 7
    assert fresh_loader.state["step"] == 7
    for a, b in zip(model.state_dict().values(), fresh.state_dict().values(), strict=False):
        assert torch.equal(a, b)


def test_verify_rejects_corruption(tmp_path):
    """A truncated checkpoint must not silently load."""
    uri = str(tmp_path)
    model = Tiny()
    ref = checkpoint.save(
        model=model, optimizer=_optim(model), loader=FakeLoader(), step=1, uri=uri
    )

    with open(ref, "r+b") as f:  # nuke the tail
        f.truncate(16)
    try:
        checkpoint.verify(ref)
    except Exception:
        return  # expected — torch.load or CheckpointError
    raise AssertionError("verify() accepted a corrupted checkpoint")


def test_smoke_test(tmp_path):
    """The restore smoke test runs a forward pass and returns a finite loss."""
    uri = str(tmp_path)
    model = Tiny()
    ref = checkpoint.save(
        model=model, optimizer=_optim(model), loader=FakeLoader(), step=1, uri=uri
    )
    blob = checkpoint.verify(ref)
    batch = (torch.randn(3, 4), torch.randn(3, 4))
    loss = checkpoint.smoke_test(blob, Tiny, batch, "cpu")
    assert loss == loss  # finite (not NaN)
