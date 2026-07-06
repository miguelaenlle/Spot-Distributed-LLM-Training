"""Elastic-training unit tests — hermetic (tiny model, local store, no network).

Pins the pieces that let survivors keep training at world size N-1 while a dead
node is replaced:

  - constant-global-batch gradient accumulation (K math + gradient equivalence);
  - the run-level budget carried in the checkpoint (trained_seconds, v2);
  - the node-local checkpoint tier (atomic save + prune) and the group-agreed
    resume step across tiers (survivors restore local, replacements S3).
"""

from __future__ import annotations

import sys

import pytest
import torch
from torch import nn

from spot_train import checkpoint, s3_store
from spot_train.config import TrainConfig
from spot_train.train import grad_accum_steps


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


def _save(uri, step, trained_seconds=0.0, model=None):
    model = model or Tiny()
    return checkpoint.save(
        model=model,
        optimizer=_optim(model),
        loader=FakeLoader(),
        step=step,
        uri=uri,
        trained_seconds=trained_seconds,
    )


# --------------------------------------------------------------------------- #
# Gradient accumulation
# --------------------------------------------------------------------------- #
def test_grad_accum_steps_math():
    assert grad_accum_steps(0, 4, 12) == 1  # disabled => today's behavior
    # The 4-node experiment default: exact at both N and N-1.
    assert grad_accum_steps(144, 4, 12) == 3  # 3 x 4 x 12 = 144
    assert grad_accum_steps(144, 3, 12) == 4  # 4 x 3 x 12 = 144
    # Non-dividing world sizes round UP (effective batch >= target, logged).
    assert grad_accum_steps(144, 7, 12) == 2  # effective 168
    assert grad_accum_steps(144, 8, 12) == 2  # effective 192
    assert grad_accum_steps(1, 4, 12) == 1  # never below one micro-batch


def test_grad_accum_gradient_equivalence():
    """K micro-batches with (loss/K).backward == one backward over the full
    batch — the invariant that makes a world-size change invisible to the
    optimizer. (Mean-reduction losses over equal-size micro-batches.)"""
    torch.manual_seed(0)
    x = torch.randn(8, 4)
    y = torch.randn(8, 4)

    full = Tiny()
    accum = Tiny()
    accum.load_state_dict(full.state_dict())

    _, loss = full(x, y)
    loss.backward()

    for half in (slice(0, 4), slice(4, 8)):
        _, l_micro = accum(x[half], y[half])
        (l_micro / 2).backward()

    for a, b in zip(full.parameters(), accum.parameters(), strict=True):
        assert torch.allclose(a.grad, b.grad, atol=1e-6)


def test_config_reads_elastic_env(monkeypatch):
    monkeypatch.setenv("GLOBAL_BATCH_SIZE", "144")
    monkeypatch.setenv("TRAIN_BUDGET_SECONDS", "600")
    monkeypatch.setenv("LOCAL_CHECKPOINT_DIR", "/tmp/ckpt-local")
    cfg = TrainConfig.from_env()
    assert cfg.global_batch_size == 144
    assert cfg.train_budget_seconds == 600.0
    assert cfg.local_checkpoint_dir == "/tmp/ckpt-local"


# --------------------------------------------------------------------------- #
# Budget in the checkpoint (v2)
# --------------------------------------------------------------------------- #
def test_checkpoint_carries_trained_seconds(tmp_path):
    uri = str(tmp_path)
    _save(uri, step=5, trained_seconds=42.5)
    blob = checkpoint.load_latest(uri)
    assert blob["version"] == 2
    assert blob["trained_seconds"] == 42.5
    checkpoint.verify(s3_store.latest(uri))  # v2 passes verify


def test_v1_checkpoint_still_loads(tmp_path):
    """A pre-elastic checkpoint (no trained_seconds, version 1) must restore —
    one resume path means no schema cliff on upgrade."""
    uri = str(tmp_path)
    ref = _save(uri, step=3)
    blob = torch.load(ref, weights_only=False)
    del blob["trained_seconds"]
    blob["version"] = 1
    torch.save(blob, ref)

    loaded = checkpoint.load_latest(uri)
    assert checkpoint.verify(ref)["version"] == 1
    assert float(loaded.get("trained_seconds", 0.0)) == 0.0
    fresh = Tiny()
    step = checkpoint.restore_into(
        loaded, model=fresh, optimizer=_optim(fresh), loader=FakeLoader()
    )
    assert step == 3


# --------------------------------------------------------------------------- #
# Node-local tier + group-agreed resume
# --------------------------------------------------------------------------- #
def _local_snapshot(local_dir, step, trained_seconds=0.0):
    m = Tiny()
    blob = checkpoint.snapshot(
        model=m,
        optimizer=_optim(m),
        loader=FakeLoader(),
        step=step,
        trained_seconds=trained_seconds,
    )
    return checkpoint.save_local(blob, str(local_dir), step)


def test_save_local_prunes_to_keep(tmp_path):
    for step in (10, 20, 30):
        _local_snapshot(tmp_path, step)
    kept = sorted(p.name for p in tmp_path.glob("ckpt-*.pt"))
    assert kept == ["ckpt-000000000020.pt", "ckpt-000000000030.pt"]
    assert checkpoint.load_latest(str(tmp_path))["step"] == 30


def test_group_latest_prefers_newer_local_tier(tmp_path):
    """Shrink case (single-process degenerate): the local tier is ahead of S3
    (async upload lag) — resume from local, losing nothing."""
    s3 = tmp_path / "s3"
    local = tmp_path / "local"
    _save(str(s3), step=8)
    _local_snapshot(local, step=10)
    blob = checkpoint.load_group_latest(str(s3), str(local))
    assert blob["step"] == 10


def test_group_latest_falls_back_to_s3(tmp_path):
    """Replacement case (degenerate): no local tier -> durable tier wins; and a
    stale local tier loses to a newer S3 checkpoint."""
    s3 = tmp_path / "s3"
    local = tmp_path / "local"
    _save(str(s3), step=12)
    assert checkpoint.load_group_latest(str(s3), "")["step"] == 12
    _local_snapshot(local, step=4)
    assert checkpoint.load_group_latest(str(s3), str(local))["step"] == 12


def test_group_latest_fresh_when_empty(tmp_path):
    assert checkpoint.load_group_latest(str(tmp_path / "s3"), str(tmp_path / "local")) is None


# --------------------------------------------------------------------------- #
# Real 2-rank agreement (gloo): survivor-with-local vs replacement-without
# --------------------------------------------------------------------------- #
def _agree_worker(rank: int, world_size: int, port: int, root: str) -> None:
    import os

    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    from spot_train import distributed

    d = distributed.init("cpu")
    s3 = os.path.join(root, "s3")
    # Rank 0 plays the survivor (local tier ahead at step 10); rank 1 the fresh
    # replacement (no local tier). The group must agree on S3's step 8 so both
    # ranks restore identical state.
    local = os.path.join(root, "local0") if rank == 0 else ""
    blob = checkpoint.load_group_latest(s3, local, d)
    assert blob is not None and blob["step"] == 8, f"rank {rank} got {blob and blob['step']}"
    distributed.shutdown(d)


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="fork-after-torch crashes on macOS (objc); DDP runs on the Linux box/CI",
)
def test_group_agreement_two_ranks(tmp_path):
    import socket

    import torch.multiprocessing as mp

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    _save(str(tmp_path / "s3"), step=8)
    _local_snapshot(tmp_path / "local0", step=10)
    mp.start_processes(
        _agree_worker,
        args=(2, port, str(tmp_path)),
        nprocs=2,
        join=True,
        start_method="fork",
    )
