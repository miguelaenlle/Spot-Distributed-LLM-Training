"""Local torchrun-elastic E2E: kill one agent, the survivor continues at world 1.

Two real torchrun agents on localhost (c10d rendezvous hosted by agent A, gloo
backend, CPU) — the same launch shape as one spot box each. SIGKILLing agent B's
whole process group simulates a node death: A's worker crashes on the broken
all_reduce, A's elastic agent re-rendezvouses, and (min nodes = 1) training
continues at world 1 WITHOUT B — the exact survivors-keep-training behavior the
multinode-preempt experiment relies on.

Subprocess-based (no fork-after-torch), so it runs on macOS too. It exercises
real sockets and timing, so it's the slowest test in the suite (~30-60s).
Set E2E_ELASTIC=0 to skip.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time

import pytest


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _spawn_agent(port: int, env: dict, min_nodes: int = 1, max_nodes: int = 2):
    here = os.path.dirname(__file__)
    cmd = [
        sys.executable,
        os.path.join(here, "elastic_launcher.py"),  # torchrun + localhost fqdn shim
        f"--nnodes={min_nodes}:{max_nodes}",
        "--nproc_per_node=1",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint=127.0.0.1:{port}",
        "--rdzv_id=e2e-test",
        "--rdzv_conf=last_call_timeout=3",
        "--max-restarts=3",
        "--local-addr=127.0.0.1",
        os.path.join(here, "elastic_worker.py"),
    ]
    # Own process group so killpg takes out the agent AND its worker — a whole
    # "node" dying at once, like TerminateInstances.
    return subprocess.Popen(
        cmd,
        env=env,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _kill_node(agent: subprocess.Popen) -> None:
    """SIGKILL an agent AND its worker processes — a whole "node" dying at once,
    like TerminateInstances. torchrun starts each worker in its own session
    (start_new_session=True in torch's SubprocessHandler), so killing the
    agent's process group alone would ORPHAN the worker and the group would
    keep all-reducing at full world size."""
    kids = subprocess.run(
        ["pgrep", "-P", str(agent.pid)], capture_output=True, text=True
    ).stdout.split()
    import contextlib

    for pid in [agent.pid] + [int(k) for k in kids]:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


def _wait_for(predicate, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


def _lines(out_path: str) -> list[str]:
    if not os.path.exists(out_path):
        return []
    with open(out_path) as f:
        return f.read().splitlines()


@pytest.mark.skipif(
    os.environ.get("E2E_ELASTIC", "1") == "0",
    reason="E2E_ELASTIC=0 — skipping the slow local elastic test",
)
def test_survivor_continues_at_world_one(tmp_path):
    out = str(tmp_path / "events.log")
    done = str(tmp_path / "done")
    env = {
        **os.environ,
        "E2E_OUT": out,
        "E2E_DONE": done,
        "OMP_NUM_THREADS": "1",
    }
    port = _free_port()
    a = b = None
    try:
        # A first (it must host the c10d store — the "node 0" of this test).
        a = _spawn_agent(port, env)
        time.sleep(2)
        b = _spawn_agent(port, env)

        # Both ranks join at world 2.
        _wait_for(
            lambda: sum("world=2" in ln for ln in _lines(out)) >= 2,
            timeout=90,
            what="both workers to start at world 2",
        )

        # "Node" B dies with no warning (agent + worker, like a terminated box).
        _kill_node(b)

        # A's worker crashes on the broken collective; A's agent re-rendezvouses
        # and restarts it WITHOUT B: world 1, restart count > 0.
        _wait_for(
            lambda: any("world=1" in ln and "restart=0" not in ln for ln in _lines(out)),
            timeout=60,
            what="the survivor to resume at world 1",
        )

        # Let the survivor finish cleanly — torchrun must exit 0.
        with open(done, "w") as f:
            f.write("1")
        assert a.wait(timeout=60) == 0
    finally:
        for proc in (a, b):
            if proc is not None and proc.poll() is None:
                _kill_node(proc)
