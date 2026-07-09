"""Localhost E2E of the FULL epoch protocol — the exact code that runs on AWS.

Two real ``orchestrator.sidecar`` processes (one per "node") poll an epoch.json
that the TEST writes (playing the orchestrator), and each runs real STATIC
torchrun (gloo, CPU) on the dummy worker. We then:

  1. publish epoch 1 (both nodes) and assert both workers reach world 2;
  2. hard-kill node 1's whole tree (a node death) and publish the shrink epoch
     (node 0 only) — assert node 0's worker resumes at world 1 (this is exactly
     what FAILED under torchrun's dynamic rendezvous on the DLAMI);
  3. re-register + re-admit node 1 (a replacement) and assert world 2 again.

Static torchrun takes ``--master_addr`` verbatim, so this needs none of the
getfqdn shim the dynamic-rendezvous test required — itself a small vindication
of dropping elastic. Subprocess-based (no fork-after-torch), runs on macOS.
Set E2E_EPOCH=0 to skip.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time

import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _write_epoch(run_dir, epoch, members, port):
    doc = {
        "epoch": epoch,
        "members": [{"node": n, "ip": "127.0.0.1", "rank": r} for r, n in enumerate(members)],
        "node_count": len(members),
        "master_addr": "127.0.0.1",
        "master_port": port,
    }
    (run_dir / "epoch.json").write_text(json.dumps(doc))


def _spawn_sidecar(run_dir, node_index, env):
    """One box: register, then run the real sidecar loop against the local run
    dir, launching the dummy worker (SIDECAR_TRAIN_MODULE) under static torchrun."""
    code = (
        "from orchestrator import sidecar; "
        f"sidecar.register({str(run_dir)!r}, {node_index}, ip='127.0.0.1'); "
        f"raise SystemExit(sidecar.run({str(run_dir)!r}, {node_index}, idle_budget=120))"
    )
    return subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _kill_tree(proc):
    kids = subprocess.run(
        ["pgrep", "-P", str(proc.pid)], capture_output=True, text=True
    ).stdout.split()
    import contextlib

    for pid in [proc.pid, *[int(k) for k in kids]]:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


def _events(run_dir):
    p = run_dir / "events.log"
    return p.read_text().splitlines() if p.exists() else []


def _spin(cond, timeout, what):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.2)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


@pytest.mark.skipif(
    os.environ.get("E2E_EPOCH", "1") == "0",
    reason="E2E_EPOCH=0 — skipping the slow localhost epoch E2E",
)
def test_epoch_shrink_then_grow(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    done = tmp_path / "done"
    env = {
        **os.environ,
        "E2E_OUT": str(run_dir / "events.log"),
        "E2E_DONE": str(done),
        "SIDECAR_TRAIN_MODULE": "elastic_worker",  # dummy worker (tests/ on PYTHONPATH)
        "NPROC_PER_NODE": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": TESTS + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    a = b = b2 = None
    try:
        # Epoch 1: both nodes. Sidecars register then launch static torchrun.
        _write_epoch(run_dir, 1, [0, 1], _free_port())
        a = _spawn_sidecar(run_dir, 0, env)
        b = _spawn_sidecar(run_dir, 1, env)
        _spin(
            lambda: sum("world=2" in ln for ln in _events(run_dir)) >= 2,
            60,
            "both workers at world 2",
        )

        # Node 1 dies with no warning; the orchestrator (this test) then publishes
        # the shrink epoch. Node 0's worker crashes on the collective (backstop)
        # AND sees epoch 2 -> its sidecar relaunches it at world 1.
        _kill_tree(b)
        _write_epoch(run_dir, 2, [0], _free_port())
        _spin(
            lambda: any("world=1" in ln for ln in _events(run_dir)),
            60,
            "survivor to resume at world 1",
        )

        # A replacement for node 1 registers and is admitted at epoch 3.
        b2 = _spawn_sidecar(run_dir, 1, env)
        _write_epoch(run_dir, 3, [0, 1], _free_port())
        _spin(
            lambda: sum("world=2" in ln for ln in _events(run_dir)) >= 4,
            60,
            "group back to world 2 after the replacement joined",
        )

        done.write_text("1")  # let the workers exit cleanly
    finally:
        (run_dir / "metrics.json").write_text("{}")  # sidecars exit their loops
        for proc in (a, b, b2):
            if proc is not None:
                _kill_tree(proc)
