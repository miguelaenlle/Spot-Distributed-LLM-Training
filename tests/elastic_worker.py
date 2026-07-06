"""Dummy trainer for the local torchrun-elastic E2E test (test_elastic_e2e.py).

Joins the gloo process group, records (world_size, rank, restart_count) to the
file at $E2E_OUT, then all-reduces in a loop until $E2E_DONE appears. A peer
dying mid-loop makes the all_reduce raise -> this worker crashes -> the elastic
agent re-rendezvouses — exactly the survivor path the real trainer takes.
"""

from __future__ import annotations

import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    out = os.environ["E2E_OUT"]
    done = os.environ["E2E_DONE"]
    dist.init_process_group(backend="gloo", timeout=timedelta(seconds=10))
    world = dist.get_world_size()
    rank = dist.get_rank()
    restart = os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
    with open(out, "a") as f:
        f.write(f"start world={world} rank={rank} restart={restart}\n")
        f.flush()
    for _ in range(600):  # <= 60s of 0.1s heartbeats
        t = torch.zeros(1)
        dist.all_reduce(t)  # raises when a peer dies -> agent restarts us
        if os.path.exists(done):
            break
        time.sleep(0.1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
