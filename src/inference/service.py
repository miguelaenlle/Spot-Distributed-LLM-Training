"""Model service for the inference worker — load once, generate serially.

Reuses the trainer's own modules (checkpoint / s3_store / build_model and the
meta.pkl char codec) so the served model is byte-for-byte the artifact a
training run produced. Generation holds a lock: nanoGPT's ``generate()``
recomputes the full context per token, so the MVP serves one request at a time
and lets the router spread load across workers (continuous batching arrives
with vLLM in ROADMAP Part 4).
"""

from __future__ import annotations

import os
import pickle
import shutil
import threading
import time
from dataclasses import dataclass, field

import torch

from spot_train import checkpoint, s3_store
from spot_train.config import TrainConfig
from spot_train.sampling import _char_codec
from spot_train.train import build_model


class ModelNotReady(RuntimeError):
    """The worker cannot serve: missing checkpoint or dataset metadata."""


class PromptError(ValueError):
    """The request prompt cannot be encoded with this model's vocabulary."""


def _ensure_meta(data_local_dir: str, data_uri: str) -> None:
    """Make sure meta.pkl (the char codec + vocab size) exists locally.

    The worker doesn't need train/val bins — only the codec — so this pulls a
    single small file instead of reusing PositionedLoader's full download.
    """
    meta_path = os.path.join(data_local_dir, "meta.pkl")
    if os.path.exists(meta_path):
        return
    if not data_uri:
        raise ModelNotReady(f"no meta.pkl in {data_local_dir} and DATA_URI is unset")
    os.makedirs(data_local_dir, exist_ok=True)
    ref = data_uri.rstrip("/") + "/meta.pkl"
    with s3_store.fetch(ref) as local:
        if local != meta_path:
            shutil.copyfile(local, meta_path)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@dataclass
class ServiceStats:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generate_seconds: float = 0.0
    # Gauge: requests inside complete() right now. Generation is serialized
    # behind a lock, so at most one is generating — the rest are the queue.
    in_flight: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def enter(self) -> None:
        with self._lock:
            self.in_flight += 1

    def leave(self) -> None:
        with self._lock:
            self.in_flight -= 1

    def record(self, prompt_tokens: int, completion_tokens: int, seconds: float) -> None:
        with self._lock:
            self.requests += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.generate_seconds += seconds

    def snapshot(self) -> dict:
        with self._lock:
            tok_s = self.completion_tokens / self.generate_seconds if self.generate_seconds else 0.0
            return {
                "requests": self.requests,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "generate_seconds": round(self.generate_seconds, 3),
                "tokens_per_second": round(tok_s, 1),
                "in_flight": self.in_flight,
                "queued": max(self.in_flight - 1, 0),
            }


class ModelService:
    """A loaded model + codec with a serialized ``complete()``."""

    def __init__(self, model, encode, decode, *, device: str, model_name: str, ckpt_step: int):
        self.model = model
        self.encode = encode
        self.decode = decode
        self.device = device
        self.model_name = model_name
        self.ckpt_step = ckpt_step
        self.stats = ServiceStats()
        self._gen_lock = threading.Lock()

    @classmethod
    def load(cls, cfg: TrainConfig) -> ModelService:
        """Load the newest checkpoint under ``cfg.checkpoint_uri`` for serving."""
        device = resolve_device(cfg.device)
        _ensure_meta(cfg.data_local_dir, cfg.data_uri)
        with open(os.path.join(cfg.data_local_dir, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        vocab_size = int(meta["vocab_size"])
        codec = _char_codec(cfg.data_local_dir)
        if codec is None:
            raise ModelNotReady(f"meta.pkl in {cfg.data_local_dir} has no stoi/itos codec")
        encode, decode = codec

        blob = checkpoint.load_latest(cfg.checkpoint_uri, map_location=device)
        if blob is None:
            raise ModelNotReady(f"no checkpoint found under {cfg.checkpoint_uri}")
        model = build_model(cfg, vocab_size)
        model.load_state_dict(blob["model"])
        model.to(device)
        model.eval()
        step = int(blob["step"])
        print(f"[worker] serving {cfg.run_id} step {step} on {device}", flush=True)
        return cls(
            model,
            encode,
            decode,
            device=device,
            model_name=f"{cfg.run_id}@step{step}",
            ckpt_step=step,
        )

    @torch.no_grad()
    def complete(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        seed: int | None = None,
    ) -> dict:
        """Generate one completion. Returns text + token counts.

        ``temperature`` is clamped to a small positive value (nanoGPT divides
        logits by it, so exactly 0 is undefined; near-0 ≈ greedy).
        """
        text = prompt or "\n"  # generate() requires a non-empty idx
        try:
            ids = self.encode(text)
        except KeyError as e:
            raise PromptError(f"prompt contains characters outside the model vocab: {e}") from e
        idx = torch.tensor(ids, dtype=torch.long, device=self.device)[None, ...]
        temperature = max(float(temperature), 1e-5)

        # Gauge covers the wait on the lock too: in_flight - 1 = queue depth.
        self.stats.enter()
        try:
            start = time.monotonic()
            with self._gen_lock:
                if seed is not None:
                    torch.manual_seed(seed)
                out = self.model.generate(
                    idx, max_new_tokens, temperature=temperature, top_k=top_k or None
                )
            elapsed = time.monotonic() - start
        finally:
            self.stats.leave()

        completion_ids = out[0, len(ids) :].tolist()
        completion = self.decode(completion_ids)
        self.stats.record(len(ids), len(completion_ids), elapsed)
        return {
            "text": completion,
            "prompt_tokens": len(ids),
            "completion_tokens": len(completion_ids),
            "generate_seconds": elapsed,
        }
