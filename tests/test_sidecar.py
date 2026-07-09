"""Sidecar state-machine tests — hermetic (local dirs as the run store, a fake
launcher instead of real torchrun). Drives the sidecar loop for a handful of
polls per scenario by feeding it epoch docs and asserting what it launched/killed.
"""

from __future__ import annotations

import json
import threading
import time

from orchestrator import sidecar
from spot_train import s3_store


class FakeProc:
    """Stands in for the torchrun Popen: alive until .stop() or kill_tree."""

    def __init__(self, epoch, rank, node_count):
        self.epoch, self.rank, self.node_count = epoch, rank, node_count
        self._rc = None
        self.killed = False

    def poll(self):
        return self._rc

    def stop(self, rc=1):  # simulate torchrun crashing on its own
        self._rc = rc

    def wait(self, timeout=None):
        self._rc = self._rc if self._rc is not None else -9
        return self._rc


class Harness:
    """Records launches; makes kill_tree mark the proc instead of touching the OS."""

    def __init__(self, monkeypatch):
        self.launches = []
        self.live = None
        monkeypatch.setattr(sidecar, "kill_tree", self._kill)

    def launch(self, epoch, rank, node_count, master_addr, master_port):
        p = FakeProc(epoch, rank, node_count)
        self.launches.append(p)
        self.live = p
        return p

    def _kill(self, proc):
        proc.killed = True
        proc._rc = -9
        if self.live is proc:
            self.live = None


def _write_epoch(run_uri, epoch, members):
    doc = {
        "epoch": epoch,
        "members": [{"node": n, "ip": f"10.0.0.{n}", "rank": r} for r, n in enumerate(members)],
        "node_count": len(members),
        "master_addr": f"10.0.0.{members[0]}",
        "master_port": 29400 + epoch,
    }
    s3_store.put_bytes(json.dumps(doc).encode(), sidecar._join(run_uri, "epoch.json"))


def _run_in_thread(run_uri, node_index, launch):
    """Run the sidecar loop in a thread with a tiny poll; return (thread, stop)."""
    result = {}

    def target():
        result["rc"] = sidecar.run(run_uri, node_index, launch=launch, idle_budget=2.0)

    t = threading.Thread(target=target, daemon=True)
    return t, result


def test_register_writes_node_doc(tmp_path):
    run_uri = str(tmp_path)
    sidecar.register(run_uri, 2, ip="10.1.2.3", instance_id="i-abc")
    doc = json.loads(s3_store.read_bytes(sidecar._join(run_uri, "nodes/node2.json")))
    assert doc == {"ip": "10.1.2.3", "instance_id": "i-abc"}


def test_enters_epoch_then_exits_on_metrics(tmp_path, monkeypatch):
    run_uri = str(tmp_path)
    h = Harness(monkeypatch)
    monkeypatch.setattr(sidecar, "POLL_SECONDS", 0.05)
    _write_epoch(run_uri, 1, [0, 1])
    t, result = _run_in_thread(run_uri, 0, h.launch)
    t.start()
    # Give it a couple polls to launch, then drop metrics.json to end the run.
    _spin(lambda: len(h.launches) == 1)
    assert h.launches[0].rank == 0 and h.launches[0].node_count == 2
    s3_store.put_bytes(b"{}", sidecar._join(run_uri, "metrics.json"))
    t.join(timeout=3)
    assert result["rc"] == 0


def test_kills_and_relaunches_on_epoch_bump(tmp_path, monkeypatch):
    run_uri = str(tmp_path)
    h = Harness(monkeypatch)
    monkeypatch.setattr(sidecar, "POLL_SECONDS", 0.05)
    _write_epoch(run_uri, 1, [0, 1])  # node 0 is rank 0 of 2
    t, result = _run_in_thread(run_uri, 0, h.launch)
    t.start()
    _spin(lambda: len(h.launches) == 1)
    # Shrink: epoch 2 keeps node 0 only -> old proc killed, relaunched at world 1.
    _write_epoch(run_uri, 2, [0])
    _spin(lambda: len(h.launches) == 2)
    assert h.launches[0].killed is True
    assert h.launches[1].node_count == 1 and h.launches[1].rank == 0
    s3_store.put_bytes(b"{}", sidecar._join(run_uri, "metrics.json"))
    t.join(timeout=3)


def test_relaunches_after_crash_within_same_epoch(tmp_path, monkeypatch):
    run_uri = str(tmp_path)
    h = Harness(monkeypatch)
    monkeypatch.setattr(sidecar, "POLL_SECONDS", 0.05)
    _write_epoch(run_uri, 1, [0, 1])
    t, result = _run_in_thread(run_uri, 0, h.launch)
    t.start()
    _spin(lambda: len(h.launches) == 1)
    # torchrun crashes on its own (peer death). Same epoch is still published, so
    # after noticing the crash the sidecar relaunches for it.
    h.launches[0].stop(rc=1)
    _spin(lambda: len(h.launches) == 2)
    assert h.launches[1].epoch == 1
    s3_store.put_bytes(b"{}", sidecar._join(run_uri, "metrics.json"))
    t.join(timeout=3)


def test_idles_when_excluded_then_gives_up(tmp_path, monkeypatch):
    run_uri = str(tmp_path)
    h = Harness(monkeypatch)
    monkeypatch.setattr(sidecar, "POLL_SECONDS", 0.05)
    # Epoch names only node 0; node 1 (a shrunk-away node) never launches, and
    # exhausts its short idle budget -> rc 1 (box left up).
    _write_epoch(run_uri, 5, [0])
    rc = sidecar.run(run_uri, 1, launch=h.launch, idle_budget=0.3)
    assert rc == 1
    assert h.launches == []


def _spin(cond, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met within timeout")
