"""Fleet router — one public endpoint, N disposable workers behind it.

Keeps a registry snapshot (polled from the heartbeat store), round-robins
completions across live workers, and reroutes on failure: a connection error,
timeout, or 5xx sends the request to the next live worker instead of the
client. That retry is the spot story — a terminated worker's in-flight
requests land somewhere else, and its stale heartbeat drops it from rotation
within the TTL.

Run: ``spot-router --port 8000`` with ``FLEET_WORKERS_URI`` pointing at the
heartbeat prefix. The router is the fleet's one stable (on-demand) component.
"""

from __future__ import annotations

import argparse
import itertools
import os
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import requests
from fastapi import FastAPI, HTTPException

from . import registry


@dataclass
class RouterSettings:
    host: str = "0.0.0.0"
    port: int = 8000
    workers_uri: str = ""
    poll_seconds: float = 3.0
    ttl_seconds: float = registry.DEFAULT_TTL_SECONDS
    request_timeout_seconds: float = 60.0
    # Separate connect timeout: a terminating EC2 box black-holes packets (no
    # RST), and a flat 60s timeout would hold rerouted requests hostage. 3s
    # connect keeps reroute latency bounded while long generations still get
    # the full read window.
    connect_timeout_seconds: float = 3.0
    max_attempts: int = 3
    stats_poll_seconds: float = 2.0  # per-worker /stats scrape cadence (live monitor)

    @classmethod
    def from_env(cls, port: int | None = None) -> RouterSettings:
        return cls(
            host=os.environ.get("HOST", "0.0.0.0"),
            port=port if port is not None else int(os.environ.get("PORT", "8000")),
            workers_uri=os.environ.get("FLEET_WORKERS_URI", ""),
            poll_seconds=float(os.environ.get("ROUTER_POLL_SECONDS", "3")),
            ttl_seconds=float(os.environ.get("WORKER_TTL_SECONDS", "15")),
            request_timeout_seconds=float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "60")),
            connect_timeout_seconds=float(os.environ.get("ROUTER_CONNECT_TIMEOUT_SECONDS", "3")),
            max_attempts=int(os.environ.get("ROUTER_MAX_ATTEMPTS", "3")),
            stats_poll_seconds=float(os.environ.get("ROUTER_STATS_POLL_SECONDS", "2")),
        )


class UpstreamError(Exception):
    """A worker attempt failed in a retryable way (connect/timeout/5xx)."""


@dataclass
class RouteResult:
    response: dict | None  # upstream JSON on success
    status_code: int  # 200 on success; client 4xx passed through; 503 if exhausted
    detail: str = ""
    attempts: int = 0
    rerouted: bool = False  # succeeded on a retry — the headline counter
    worker_id: str = ""


def route_completion(
    body: dict,
    workers: list[dict],
    post: Callable[[str, dict], tuple[int, dict]],
    *,
    start_index: int = 0,
    max_attempts: int = 3,
) -> RouteResult:
    """Try live workers in round-robin order until one answers.

    Pure routing policy (the HTTP call is injected) so tests can drive it
    without sockets. Retryable: transport errors (``post`` raises
    ``UpstreamError``) and 5xx. Not retryable: 4xx — that's the client's
    request, every worker would reject it the same way.
    """
    if not workers:
        return RouteResult(None, 503, detail="no live workers", attempts=0)
    attempts = 0
    last_detail = ""
    order = itertools.islice(
        itertools.cycle(range(len(workers))), start_index, start_index + len(workers)
    )
    for i in order:
        if attempts >= max_attempts:
            break
        worker = workers[i]
        attempts += 1
        try:
            status, payload = post(worker["addr"], body)
        except UpstreamError as e:
            last_detail = f"{worker['worker_id']}: {e}"
            continue
        if 200 <= status < 300:
            return RouteResult(
                payload,
                status,
                attempts=attempts,
                rerouted=attempts > 1,
                worker_id=worker.get("worker_id", ""),
            )
        if 400 <= status < 500:
            detail = payload.get("detail", "") if isinstance(payload, dict) else ""
            return RouteResult(payload, status, detail=detail, attempts=attempts)
        last_detail = f"{worker['worker_id']}: upstream {status}"
    return RouteResult(None, 503, detail=f"all attempts failed ({last_detail})", attempts=attempts)


@dataclass
class RouterState:
    workers: list[dict] = field(default_factory=list)
    rr: int = 0
    requests: int = 0
    rerouted: int = 0
    failed: int = 0
    in_flight: int = 0  # gauge: requests currently being proxied
    last_poll: float = 0.0
    worker_stats: list[dict] = field(default_factory=list)  # last /stats scrape per worker
    stats_ts: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_workers(self, docs: list[dict]) -> None:
        with self._lock:
            self.workers = docs
            self.last_poll = time.time()

    def set_worker_stats(self, stats: list[dict]) -> None:
        with self._lock:
            self.worker_stats = stats
            self.stats_ts = time.time()

    def enter(self) -> None:
        with self._lock:
            self.in_flight += 1

    def leave(self) -> None:
        with self._lock:
            self.in_flight -= 1

    def next_start(self, n: int) -> int:
        with self._lock:
            self.rr = (self.rr + 1) % max(n, 1)
            return self.rr

    def record(self, result: RouteResult) -> None:
        with self._lock:
            self.requests += 1
            if result.rerouted:
                self.rerouted += 1
            if result.status_code >= 500:
                self.failed += 1


def _poll_loop(state: RouterState, settings: RouterSettings, stop: threading.Event):
    while not stop.is_set():
        try:
            docs = registry.list_workers(settings.workers_uri)
            state.set_workers(registry.live_workers(docs, settings.ttl_seconds))
        except Exception as e:
            print(f"[router] registry poll failed: {e}", flush=True)
        stop.wait(settings.poll_seconds)


def scrape_worker_stats(workers: list[dict], get: Callable[[str], dict]) -> list[dict]:
    """One monitoring sweep: ``get`` fetches http://<addr>/stats (injected for
    tests). A worker that errors still appears, with ok=False — the monitor
    shows it dying rather than silently dropping it."""
    out = []
    for w in workers:
        doc = {"worker_id": w.get("worker_id", ""), "addr": w.get("addr", ""), "ok": True}
        try:
            doc.update(get(w["addr"]))
        except Exception as e:
            doc["ok"] = False
            doc["error"] = str(e)[:120]
        out.append(doc)
    return out


def _stats_loop(state: RouterState, settings: RouterSettings, stop: threading.Event):
    def get(addr: str) -> dict:
        return requests.get(f"http://{addr}/stats", timeout=1.5).json()

    while not stop.is_set():
        state.set_worker_stats(scrape_worker_stats(list(state.workers), get))
        stop.wait(settings.stats_poll_seconds)


def create_app(settings: RouterSettings | None = None, state: RouterState | None = None) -> FastAPI:
    settings = settings or RouterSettings.from_env()
    state = state or RouterState()
    stop = threading.Event()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if not settings.workers_uri:
            print("[router] WARNING: FLEET_WORKERS_URI unset — no workers will be found")
        threading.Thread(target=_poll_loop, args=(state, settings, stop), daemon=True).start()
        threading.Thread(target=_stats_loop, args=(state, settings, stop), daemon=True).start()
        yield
        stop.set()

    app = FastAPI(title="spot-train fleet router", lifespan=lifespan)

    def _post(addr: str, body: dict) -> tuple[int, dict]:
        try:
            r = requests.post(
                f"http://{addr}/v1/completions",
                json=body,
                timeout=(settings.connect_timeout_seconds, settings.request_timeout_seconds),
            )
        except requests.RequestException as e:
            raise UpstreamError(str(e)) from e
        try:
            payload = r.json()
        except ValueError:
            payload = {"detail": r.text[:200]}
        return r.status_code, payload

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "live_workers": len(state.workers)}

    @app.get("/fleet/status")
    def fleet_status():
        return {
            "live_workers": len(state.workers),
            "workers": state.workers,
            "requests": state.requests,
            "rerouted": state.rerouted,
            "failed": state.failed,
            "last_poll": state.last_poll,
        }

    @app.get("/fleet/metrics")
    def fleet_metrics():
        """Live aggregate for the fleet monitor: router counters + the latest
        per-worker /stats scrape (queue depth, tokens, in-flight)."""
        return {
            "ts": time.time(),
            "router": {
                "live_workers": len(state.workers),
                "in_flight": state.in_flight,
                "requests": state.requests,
                "rerouted": state.rerouted,
                "failed": state.failed,
            },
            "workers": state.worker_stats,
            "stats_ts": state.stats_ts,
        }

    @app.post("/v1/completions")
    def completions(body: dict):
        workers = list(state.workers)
        start = state.next_start(len(workers)) if workers else 0
        state.enter()
        try:
            result = route_completion(
                body, workers, _post, start_index=start, max_attempts=settings.max_attempts
            )
        finally:
            state.leave()
        state.record(result)
        if result.status_code == 503:
            raise HTTPException(status_code=503, detail=result.detail)
        if result.status_code >= 400:
            raise HTTPException(status_code=result.status_code, detail=result.detail)
        return result.response

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="spot-train fleet router")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default=None)
    args = parser.parse_args()

    settings = RouterSettings.from_env(port=args.port)
    if args.host:
        settings.host = args.host
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
