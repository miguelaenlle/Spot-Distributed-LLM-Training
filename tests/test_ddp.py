"""DDP plumbing tests — hermetic (gloo backend on CPU, no GPU, no network).

Two levels: the single-process `distributed` context is a no-op (guards the
byte-identical fallback), and a real 2-rank gloo group proves gradients are
all-reduced (params converge across ranks) and the coordinated-stop flag agrees.
"""

from __future__ import annotations

import socket
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from spot_train import distributed


class Tiny(nn.Module):
    """nanoGPT's forward(idx, targets) -> (logits, loss) signature, minimal."""

    def __init__(self):
        super().__init__()
        self.l = nn.Linear(4, 4)

    def forward(self, x, y=None):
        out = self.l(x)
        loss = ((out - y) ** 2).mean() if y is not None else None
        return out, loss


# --------------------------------------------------------------------------- #
# Single-process: the context is dormant (RANK unset) => byte-identical path
# --------------------------------------------------------------------------- #
def test_dist_disabled_without_rank(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    d = distributed.init("cpu")
    assert d.enabled is False
    assert d.master is True
    assert d.world_size == 1
    assert d.device == "cpu"


def test_disabled_collectives_passthrough(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    d = distributed.init("cpu")
    assert distributed.all_reduce_stop(d, True) is True
    assert distributed.all_reduce_stop(d, False) is False
    assert distributed.mean_loss(d, 2.5) == 2.5


# --------------------------------------------------------------------------- #
# Real 2-rank gloo group
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _ddp_worker(rank: int, world_size: int, port: int) -> None:
    import os

    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    d = distributed.init("cpu")
    assert d.enabled and d.world_size == world_size and d.rank == rank
    assert d.master == (rank == 0)

    torch.manual_seed(0)  # identical init; DDP also broadcasts rank-0 on construction
    model = Tiny()
    ddp_model = DDP(model, device_ids=None)  # None for gloo/CPU
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    # DIFFERENT data per rank => different local grads. If DDP averages them, both
    # ranks apply the SAME update, so params must be IDENTICAL across ranks after.
    x = torch.randn(4, 4) + rank
    y = torch.randn(4, 4) + rank
    _, loss = ddp_model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    p = next(model.parameters()).detach().clone()
    pmax = p.clone()
    dist.all_reduce(pmax, op=dist.ReduceOp.MAX)
    pmin = p.clone()
    dist.all_reduce(pmin, op=dist.ReduceOp.MIN)
    assert torch.allclose(pmax, pmin, atol=1e-6), "DDP did not sync gradients"

    # coordinated stop: only rank 1 asks to stop; MAX => everyone agrees to stop.
    assert distributed.all_reduce_stop(d, rank == 1) is True
    distributed.shutdown(d)


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="fork-after-torch crashes on macOS (objc); DDP runs on the Linux box/CI",
)
def test_ddp_grads_all_reduced_two_ranks():
    # fork start method: the child inherits sys.path + imports (editable install),
    # avoiding spawn's re-import of the pytest test module. Fork is safe on Linux.
    mp.start_processes(
        _ddp_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
        start_method="fork",
    )
