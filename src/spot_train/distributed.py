"""Single-node DDP context for torchrun — the ONLY DDP-specific module.

Dormant unless ``RANK`` is set (i.e. launched via torchrun), so the single-process
training path is byte-identical to Phase 1a. On CPU we use the ``gloo`` backend
(no GPU needed); on a real GPU box it's ``nccl``. torchrun ``--standalone`` sets
MASTER_ADDR/MASTER_PORT/RANK/LOCAL_RANK/WORLD_SIZE for us.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class Dist:
    enabled: bool  # True only under torchrun (RANK set)
    rank: int
    local_rank: int
    world_size: int
    master: bool  # rank 0 — does all logging / checkpointing / metrics
    device: str


def init(device: str) -> Dist:
    """Detect torchrun and (if present) join the process group. Returns a context;
    ``enabled=False`` (single process) when RANK is unset."""
    rank = int(os.environ.get("RANK", -1))
    if rank == -1:
        return Dist(enabled=False, rank=0, local_rank=0, world_size=1, master=True, device=device)
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if device.startswith("cuda"):
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        # device_id binds the NCCL communicator to this rank's GPU eagerly, so
        # collectives can't lazily pick the wrong device (and shutdown is clean).
        dist.init_process_group(backend="nccl", device_id=torch.device(device))
    else:
        dist.init_process_group(backend="gloo")
    return Dist(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        master=(rank == 0),
        device=device,
    )


def shutdown(d: Dist) -> None:
    if d.enabled and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_stop(d: Dist, local_stop: bool) -> bool:
    """Agree a stop across ranks (MAX): if ANY rank wants to stop, all stop on the
    same iteration. Must be the last collective in the loop body so no rank is left
    blocking on the next backward's gradient all-reduce (deadlock avoidance)."""
    if not d.enabled:
        return local_stop
    # Collective tensors must live on the backend's device: nccl only reduces
    # CUDA tensors (a CPU tensor raises "No backend type associated with device
    # type cpu"); on gloo d.device is "cpu" so this is a no-op.
    t = torch.tensor([1 if local_stop else 0], dtype=torch.int32, device=d.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def mean_loss(d: Dist, loss_value: float) -> float:
    """Global mean loss across ranks for the reported log line (all ranks call it)."""
    if not d.enabled:
        return loss_value
    t = torch.tensor([loss_value], dtype=torch.float32, device=d.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item()) / d.world_size
