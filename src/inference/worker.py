"""Inference worker — FastAPI app serving one trained checkpoint.

Loads the newest checkpoint under ``CHECKPOINT_URI`` at boot, then serves an
OpenAI-shaped completions endpoint and heartbeats to ``FLEET_WORKERS_URI`` so
the router can find (and drop) it. A spot kill needs no cleanup: the heartbeat
goes stale and the router reroutes.

Run: ``spot-worker --port 8001`` (env-configured, like the trainer).
"""

from __future__ import annotations

import argparse
import os
import socket
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from spot_train.config import TrainConfig

from . import registry
from .service import ModelService, PromptError


@dataclass
class WorkerSettings:
    host: str = "0.0.0.0"
    port: int = 8001
    worker_id: str = ""
    advertise_addr: str = ""  # "host:port" the router dials
    workers_uri: str = ""  # heartbeat prefix; empty => heartbeat off
    heartbeat_seconds: float = 5.0
    market: str = "local"

    @classmethod
    def from_env(cls, port: int | None = None) -> WorkerSettings:
        port = port if port is not None else int(os.environ.get("PORT", "8001"))
        worker_id = os.environ.get("WORKER_ID", "") or f"{socket.gethostname()}-{port}"
        advertise = os.environ.get("ADVERTISE_ADDR", "") or f"127.0.0.1:{port}"
        if ":" not in advertise:
            advertise = f"{advertise}:{port}"
        return cls(
            host=os.environ.get("HOST", "0.0.0.0"),
            port=port,
            worker_id=worker_id,
            advertise_addr=advertise,
            workers_uri=os.environ.get("FLEET_WORKERS_URI", ""),
            heartbeat_seconds=float(os.environ.get("HEARTBEAT_SECONDS", "5")),
            market=os.environ.get("MARKET", "local"),
        )


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_k: int = Field(default=200, ge=0)
    seed: int | None = None
    model: str | None = None  # informational; one model per worker


def _heartbeat_loop(service: ModelService, settings: WorkerSettings, stop: threading.Event):
    while not stop.is_set():
        try:
            registry.put_worker(
                settings.workers_uri,
                registry.worker_doc(
                    settings.worker_id,
                    settings.advertise_addr,
                    model=service.model_name,
                    market=settings.market,
                    requests_served=service.stats.snapshot()["requests"],
                ),
            )
        except Exception as e:  # never let a flaky store kill serving
            print(f"[worker] heartbeat failed: {e}", flush=True)
        stop.wait(settings.heartbeat_seconds)


def create_app(service: ModelService, settings: WorkerSettings | None = None) -> FastAPI:
    """App factory taking a loaded service — tests inject a stub here."""
    settings = settings or WorkerSettings.from_env()
    stop = threading.Event()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if settings.workers_uri:
            threading.Thread(
                target=_heartbeat_loop, args=(service, settings, stop), daemon=True
            ).start()
        yield
        stop.set()
        if settings.workers_uri:
            registry.remove_worker(settings.workers_uri, settings.worker_id)

    app = FastAPI(title="spot-train inference worker", lifespan=lifespan)

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "worker_id": settings.worker_id, "model": service.model_name}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{"id": service.model_name, "object": "model", "owned_by": "spot-train"}],
        }

    @app.get("/stats")
    def stats():
        return {
            "worker_id": settings.worker_id,
            "model": service.model_name,
            "device": service.device,
            **service.stats.snapshot(),
        }

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        try:
            result = service.complete(
                req.prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                seed=req.seed,
            )
        except PromptError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": service.model_name,
            "worker_id": settings.worker_id,
            "choices": [
                {
                    "text": result["text"],
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            },
        }

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="spot-train inference worker")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default=None)
    args = parser.parse_args()

    settings = WorkerSettings.from_env(port=args.port)
    if args.host:
        settings.host = args.host
    service = ModelService.load(TrainConfig.from_env())
    uvicorn.run(create_app(service, settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
