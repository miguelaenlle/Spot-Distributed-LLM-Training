"""Monitor plumbing: worker queue gauge, router scrape/aggregate, rate math."""

import pytest

from inference.service import ServiceStats
from orchestrator.monitor import compute_rates, flatten_metrics


def _sample(ts, workers):
    return {
        "ts": ts,
        "router": {
            "live_workers": len(workers),
            "in_flight": 0,
            "requests": 0,
            "rerouted": 0,
            "failed": 0,
        },
        "workers": workers,
    }


def _worker(wid, requests=0, tokens=0, gen_s=0.0, queued=0, ok=True):
    return {
        "worker_id": wid,
        "ok": ok,
        "requests": requests,
        "completion_tokens": tokens,
        "generate_seconds": gen_s,
        "queued": queued,
        "in_flight": queued + 1 if queued else 0,
    }


def test_stats_gauge_and_queue_depth():
    s = ServiceStats()
    s.enter()
    s.enter()
    s.enter()
    snap = s.snapshot()
    assert snap["in_flight"] == 3
    assert snap["queued"] == 2  # one generating, two waiting on the lock
    s.leave()
    s.leave()
    s.leave()
    assert s.snapshot()["queued"] == 0


def test_rates_from_counter_deltas():
    prev = _sample(100.0, [_worker("w0", requests=10, tokens=320, gen_s=1.0)])
    cur = _sample(102.0, [_worker("w0", requests=14, tokens=448, gen_s=2.6)])
    r = compute_rates(prev, cur)["w0"]
    assert r["rps"] == pytest.approx(2.0)
    assert r["tok_s"] == pytest.approx(64.0)
    assert r["util"] == pytest.approx(0.8)


def test_rates_survive_counter_reset_and_new_workers():
    """A killed/replaced worker resets its counters — one idle tick, no negatives."""
    prev = _sample(100.0, [_worker("w0", requests=50, tokens=1000, gen_s=9.0)])
    cur = _sample(
        102.0,
        [
            _worker("w0", requests=2, tokens=64, gen_s=0.4),  # restarted: counters reset
            _worker("w1", requests=1, tokens=32, gen_s=0.2),  # brand new
        ],
    )
    r = compute_rates(prev, cur)
    assert r["w0"]["rps"] == 0.0  # clamped, not negative
    assert r["w1"] == {"rps": 0.0, "tok_s": 0.0, "util": 0.0}


def test_rates_first_tick_and_down_worker():
    cur = _sample(100.0, [_worker("w0"), _worker("w1", ok=False)])
    r = compute_rates(None, cur)
    assert r["w0"]["rps"] == 0.0
    assert r["w1"]["rps"] == 0.0


def test_flatten_metrics_shapes_wandb_keys():
    cur = _sample(100.0, [_worker("w0", queued=3), _worker("w1", ok=False)])
    flat = flatten_metrics(cur, {"w0": {"rps": 1.5, "tok_s": 48.0, "util": 0.5}})
    assert flat["fleet/queued_total"] == 3
    assert flat["workers/w0/queued"] == 3
    assert flat["workers/w0/rps"] == 1.5
    assert flat["workers/w1/ok"] == 0.0
    assert "workers/w1/queued" not in flat  # down worker reports no gauges
    assert flat["router/live_workers"] == 2


def test_router_scrape_marks_dead_workers():
    pytest.importorskip("fastapi")
    from inference.router import scrape_worker_stats

    def get(addr):
        if addr == "dead:1":
            raise ConnectionError("refused")
        return {"requests": 5, "queued": 1}

    docs = scrape_worker_stats(
        [{"worker_id": "a", "addr": "live:1"}, {"worker_id": "b", "addr": "dead:1"}], get
    )
    assert docs[0]["ok"] and docs[0]["requests"] == 5
    assert not docs[1]["ok"] and "refused" in docs[1]["error"]


def test_router_metrics_endpoint():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from inference.router import RouterSettings, RouterState, create_app

    state = RouterState()
    state.set_worker_stats([_worker("w0", queued=2)])
    app = create_app(RouterSettings(workers_uri="unused"), state)
    # No `with`: skip lifespan so the real scrape thread doesn't overwrite the
    # injected stats with an empty sweep (there are no live workers here).
    client = TestClient(app)
    doc = client.get("/fleet/metrics").json()
    assert doc["router"]["live_workers"] == 0
    assert doc["router"]["in_flight"] == 0
    assert doc["workers"][0]["worker_id"] == "w0"
    assert doc["workers"][0]["queued"] == 2
    assert "ts" in doc
