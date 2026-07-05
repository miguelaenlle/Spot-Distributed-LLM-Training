"""Routing policy: retry-on-failure is the spot-preemption story — verify it
without sockets by injecting the HTTP call."""

import pytest

pytest.importorskip("fastapi")  # router module imports fastapi at module scope

from inference.router import RouteResult, UpstreamError, route_completion  # noqa: E402


def _workers(n):
    return [{"worker_id": f"w{i}", "addr": f"host{i}:8001"} for i in range(n)]


def test_success_first_try():
    def post(addr, body):
        return 200, {"choices": [{"text": "ok"}], "from": addr}

    r = route_completion({"prompt": "x"}, _workers(2), post)
    assert r.status_code == 200
    assert r.attempts == 1
    assert not r.rerouted


def test_reroutes_on_connection_error():
    """A dead worker's request lands on the next one — the headline behavior."""
    calls = []

    def post(addr, body):
        calls.append(addr)
        if addr == "host0:8001":
            raise UpstreamError("connection refused")
        return 200, {"choices": [{"text": "ok"}]}

    r = route_completion({"prompt": "x"}, _workers(2), post, start_index=0, max_attempts=3)
    assert r.status_code == 200
    assert r.rerouted
    assert r.attempts == 2
    assert calls == ["host0:8001", "host1:8001"]


def test_reroutes_on_5xx():
    def post(addr, body):
        if addr == "host0:8001":
            return 500, {}
        return 200, {"choices": []}

    r = route_completion({"prompt": "x"}, _workers(2), post)
    assert r.status_code == 200
    assert r.rerouted


def test_4xx_passes_through_without_retry():
    """A bad request would fail identically everywhere — don't burn workers."""
    calls = []

    def post(addr, body):
        calls.append(addr)
        return 400, {"detail": "prompt contains characters outside the model vocab"}

    r = route_completion({"prompt": "\N{SNOWMAN}"}, _workers(3), post)
    assert r.status_code == 400
    assert "vocab" in r.detail
    assert len(calls) == 1


def test_all_dead_returns_503_with_bounded_attempts():
    def post(addr, body):
        raise UpstreamError("boom")

    r = route_completion({"prompt": "x"}, _workers(5), post, max_attempts=3)
    assert r.status_code == 503
    assert r.attempts == 3


def test_no_workers_is_503():
    r = route_completion({"prompt": "x"}, [], lambda a, b: (200, {}))
    assert r.status_code == 503
    assert r.attempts == 0


def test_round_robin_start_index_spreads_load():
    seen = []

    def post(addr, body):
        seen.append(addr)
        return 200, {}

    workers = _workers(3)
    for i in range(3):
        route_completion({}, workers, post, start_index=i)
    assert seen == ["host0:8001", "host1:8001", "host2:8001"]


def test_result_records_serving_worker():
    def post(addr, body):
        return 200, {}

    r = route_completion({}, _workers(2), post, start_index=1)
    assert isinstance(r, RouteResult)
    assert r.worker_id == "w1"
