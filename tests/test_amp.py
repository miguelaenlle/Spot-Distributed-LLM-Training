"""Mixed-precision (AMP) parity tests — hermetic, CPU-only.

Two contracts pinned here, both about *not* regressing the fp32/CPU path while
adding nanoGPT's autocast + GradScaler:

  1. ``_resolve_amp`` never turns on autocast off-GPU or under DTYPE=float32, so
     the determinism tests keep running in bit-exact fp32.
  2. The fp16 GradScaler's loss-scale state round-trips through a checkpoint
     (v3), and older/disabled-scaler blobs restore without it — no schema cliff.
"""

from __future__ import annotations

import contextlib

import torch
from torch import nn

from spot_train import checkpoint
from spot_train.train import _resolve_amp


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(4, 4)

    def forward(self, x, y=None):
        out = self.l(x)
        loss = ((out - y) ** 2).mean() if y is not None else None
        return out, loss


class FakeLoader:
    def __init__(self):
        self.state = {"step": 0}

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, sd):
        self.state = dict(sd)


class FakeScaler:
    """Stand-in for torch.amp.GradScaler — a real enabled one needs CUDA, but the
    checkpoint plumbing only touches is_enabled/state_dict/load_state_dict."""

    def __init__(self, enabled, state=None):
        self._enabled = enabled
        self._state = state or {"scale": 65536.0, "_growth_tracker": 3}
        self.loaded = None

    def is_enabled(self):
        return self._enabled

    def state_dict(self):
        return dict(self._state)

    def load_state_dict(self, sd):
        self.loaded = dict(sd)


def _optim(m):
    return torch.optim.SGD(m.parameters(), lr=0.1)


# --- dtype resolution: the CPU/fp32 path must stay a no-op ------------------- #
def test_cpu_is_always_fp32_nullcontext():
    for dtype in ("auto", "float32", "bfloat16", "float16", ""):
        amp_dtype, ctx = _resolve_amp(dtype, "cpu")
        assert amp_dtype is None
        assert isinstance(ctx, contextlib.nullcontext)


def test_float32_on_cuda_is_nullcontext():
    amp_dtype, ctx = _resolve_amp("float32", "cuda")
    assert amp_dtype is None
    assert isinstance(ctx, contextlib.nullcontext)


def test_explicit_cuda_dtypes_map_to_autocast():
    # Constructing autocast needs no GPU; we assert the dtype, never enter it.
    assert _resolve_amp("float16", "cuda")[0] is torch.float16
    assert _resolve_amp("bf16", "cuda")[0] is torch.bfloat16
    assert isinstance(_resolve_amp("float16", "cuda")[1], torch.autocast)


def test_auto_prefers_fp16_on_turing_bf16_on_ampere(monkeypatch):
    """The T4 regression: bf16 has no tensor-core path on Turing (cc 7.5), so
    'auto' must pick fp16 there and only pick bf16 on Ampere+ (cc >= 8.0)."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    assert _resolve_amp("auto", "cuda")[0] is torch.float16  # T4 -> fp16
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 0))
    assert _resolve_amp("auto", "cuda")[0] is torch.bfloat16  # A100 -> bf16


# --- scaler state round-trips through a checkpoint -------------------------- #
def test_enabled_scaler_state_is_saved_and_restored(tmp_path):
    scaler = FakeScaler(enabled=True, state={"scale": 1024.0, "_growth_tracker": 7})
    ref = checkpoint.save(
        model=Tiny(),
        optimizer=_optim(Tiny()),
        loader=FakeLoader(),
        step=5,
        uri=str(tmp_path),
        scaler=scaler,
    )
    blob = torch.load(ref, weights_only=False)
    assert blob["version"] == checkpoint.CKPT_VERSION
    assert blob["scaler"] == {"scale": 1024.0, "_growth_tracker": 7}

    fresh = FakeScaler(enabled=True)
    m = Tiny()
    checkpoint.restore_into(blob, model=m, optimizer=_optim(m), loader=FakeLoader(), scaler=fresh)
    assert fresh.loaded == {"scale": 1024.0, "_growth_tracker": 7}


def test_disabled_scaler_saves_no_state(tmp_path):
    ref = checkpoint.save(
        model=Tiny(),
        optimizer=_optim(Tiny()),
        loader=FakeLoader(),
        step=1,
        uri=str(tmp_path),
        scaler=FakeScaler(enabled=False),
    )
    assert torch.load(ref, weights_only=False)["scaler"] is None


def test_restore_tolerates_missing_scaler_key(tmp_path):
    """A v1/v2 blob (no 'scaler' key) restores into an fp16 run without crashing;
    the scaler just keeps its fresh scale."""
    ref = checkpoint.save(
        model=Tiny(), optimizer=_optim(Tiny()), loader=FakeLoader(), step=2, uri=str(tmp_path)
    )
    blob = torch.load(ref, weights_only=False)
    del blob["scaler"]  # simulate an older checkpoint
    fresh = FakeScaler(enabled=True)
    m = Tiny()
    checkpoint.restore_into(blob, model=m, optimizer=_optim(m), loader=FakeLoader(), scaler=fresh)
    assert fresh.loaded is None  # nothing to load; no crash
