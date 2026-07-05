"""Async checkpointing tests — hermetic (local fs, tiny torch modules).

Pins the properties the two-phase design rests on: the snapshot is a true
point-in-time copy (later mutation can't leak in), the written artifact is a
valid checkpoint (same schema, restores exactly), only one save is ever in
flight (skip-when-busy), and a background failure is counted + survivable
rather than fatal.
"""

from __future__ import annotations

import threading

import torch

from spot_train import checkpoint


class _StubLoader:
    """Just enough loader for snapshot/restore (state_dict of plain ints)."""

    def __init__(self, step: int = 0, epoch: int = 0):
        self.state = {"step": step, "epoch": epoch}

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, sd):
        self.state = dict(sd)


def _model_opt(seed: int = 0):
    torch.manual_seed(seed)
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # One step so the optimizer has real state tensors to snapshot.
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    opt.step()
    return model, opt


def test_async_write_restores_exactly(tmp_path):
    model, opt = _model_opt()
    writer = checkpoint.AsyncCheckpointer(str(tmp_path) + "/", verify_every=1)
    assert writer.submit(model=model, optimizer=opt, loader=_StubLoader(step=7), step=42)
    writer.flush()
    assert writer.failures == 0

    blob = checkpoint.load_latest(str(tmp_path) + "/")
    fresh_model, fresh_opt = _model_opt(seed=1)  # different init, gets overwritten
    fresh_loader = _StubLoader()
    assert (
        checkpoint.restore_into(blob, model=fresh_model, optimizer=fresh_opt, loader=fresh_loader)
        == 42
    )
    for a, b in zip(model.state_dict().values(), fresh_model.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert fresh_loader.state == {"step": 7, "epoch": 0}


def test_snapshot_is_point_in_time(tmp_path):
    model, opt = _model_opt()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    writer = checkpoint.AsyncCheckpointer(str(tmp_path) + "/", verify_every=0)
    assert writer.submit(model=model, optimizer=opt, loader=_StubLoader(), step=1)
    # Mutate the live weights immediately — the background write must not see it.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    writer.flush()
    blob = checkpoint.load_latest(str(tmp_path) + "/")
    for k, v in blob["model"].items():
        assert torch.equal(v, original[k])


def test_one_in_flight_skips_when_busy(tmp_path, monkeypatch):
    release = threading.Event()
    real_save_atomic = checkpoint.s3_store.save_atomic

    def slow_save_atomic(local_path, uri, name):
        release.wait(timeout=30)
        return real_save_atomic(local_path, uri, name)

    monkeypatch.setattr(checkpoint.s3_store, "save_atomic", slow_save_atomic)
    model, opt = _model_opt()
    writer = checkpoint.AsyncCheckpointer(str(tmp_path) + "/", verify_every=0)
    assert writer.submit(model=model, optimizer=opt, loader=_StubLoader(), step=1)
    # Previous upload still blocked => skip, nothing queued.
    assert not writer.submit(model=model, optimizer=opt, loader=_StubLoader(), step=2)
    release.set()
    writer.flush()
    blob = checkpoint.load_latest(str(tmp_path) + "/")
    assert blob["step"] == 1  # only the first save happened
    # Idle again => a new submit goes through.
    assert writer.submit(model=model, optimizer=opt, loader=_StubLoader(), step=3)
    writer.flush()
    assert checkpoint.load_latest(str(tmp_path) + "/")["step"] == 3


def test_background_failure_is_counted_not_fatal(tmp_path, monkeypatch):
    def boom(local_path, uri, name):
        raise OSError("upload exploded")

    monkeypatch.setattr(checkpoint.s3_store, "save_atomic", boom)
    model, opt = _model_opt()
    logs: list[str] = []
    writer = checkpoint.AsyncCheckpointer(str(tmp_path) + "/", verify_every=0, log=logs.append)
    assert writer.submit(model=model, optimizer=opt, loader=_StubLoader(), step=1)
    writer.flush()
    assert writer.failures == 1
    assert any("FAILED" in m for m in logs)
    assert checkpoint.load_latest(str(tmp_path) + "/") is None  # nothing corrupt left behind


def test_checkpoint_async_env_parse(monkeypatch):
    from spot_train.config import TrainConfig

    monkeypatch.delenv("CHECKPOINT_ASYNC", raising=False)
    assert TrainConfig.from_env().checkpoint_async is True
    monkeypatch.setenv("CHECKPOINT_ASYNC", "0")
    assert TrainConfig.from_env().checkpoint_async is False
    monkeypatch.setenv("CHECKPOINT_ASYNC", "false")
    assert TrainConfig.from_env().checkpoint_async is False
