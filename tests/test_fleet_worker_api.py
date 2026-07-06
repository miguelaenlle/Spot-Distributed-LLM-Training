"""Worker API contract, with a stub model service (no checkpoint needed)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

from inference.service import PromptError, ServiceStats  # noqa: E402
from inference.worker import WorkerSettings, create_app  # noqa: E402


class StubService:
    model_name = "test-run@step42"
    device = "cpu"

    def __init__(self):
        self.stats = ServiceStats()

    def complete(self, prompt, *, max_new_tokens, temperature, top_k, seed=None):
        if "\N{SNOWMAN}" in prompt:
            raise PromptError("prompt contains characters outside the model vocab")
        self.stats.record(len(prompt), max_new_tokens, 0.01)
        return {
            "text": "x" * max_new_tokens,
            "prompt_tokens": len(prompt),
            "completion_tokens": max_new_tokens,
            "generate_seconds": 0.01,
        }


@pytest.fixture
def client():
    settings = WorkerSettings(worker_id="test-w0", workers_uri="")  # heartbeat off
    app = create_app(StubService(), settings)
    with TestClient(app) as c:
        yield c


def test_completions_openai_shape(client):
    r = client.post("/v1/completions", json={"prompt": "ROMEO:", "max_tokens": 8})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "test-run@step42"
    assert body["choices"][0]["text"] == "xxxxxxxx"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 8
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + 8


def test_out_of_vocab_prompt_is_400(client):
    r = client.post("/v1/completions", json={"prompt": "\N{SNOWMAN}"})
    assert r.status_code == 400
    assert "vocab" in r.json()["detail"]


def test_validation_rejects_bad_max_tokens(client):
    r = client.post("/v1/completions", json={"prompt": "x", "max_tokens": 0})
    assert r.status_code == 422


def test_healthz_and_models(client):
    assert client.get("/healthz").json()["ok"] is True
    models = client.get("/v1/models").json()
    assert models["data"][0]["id"] == "test-run@step42"


def test_stats_counts_requests(client):
    client.post("/v1/completions", json={"prompt": "a", "max_tokens": 4})
    client.post("/v1/completions", json={"prompt": "b", "max_tokens": 4})
    stats = client.get("/stats").json()
    assert stats["requests"] == 2
    assert stats["completion_tokens"] == 8
