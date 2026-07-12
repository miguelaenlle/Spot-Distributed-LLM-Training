"""End-of-run and mid-training text generation for a fixed prompt series.

Two codecs, picked by what the dataset ships: a char-level one from ``meta.pkl``
(``stoi`` / ``itos``, present for shakespeare_char), else the GPT-2 BPE via
``tiktoken`` for the BPE corpora (OpenWebText) that carry no ``meta.pkl`` — the
same tokenizer their ``prepare.py`` used. Only when neither is available (e.g.
tiktoken not installed) does sampling skip gracefully. Generation is wrapped in
RNG capture/restore and seeded from ``(seed, step)``, so it never perturbs the
training data stream (resume determinism) and a re-covered snapshot step after a
preemption regenerates the identical file.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from collections.abc import Callable

import torch

from . import rng, s3_store
from .config import TrainConfig


def _char_codec(
    data_local_dir: str,
) -> tuple[Callable[[str], list[int]], Callable[[list[int]], str]] | None:
    """(encode, decode) from meta.pkl's stoi/itos, or None if unavailable."""
    meta_path = os.path.join(data_local_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta.get("stoi"), meta.get("itos")
    if not stoi or not itos:
        return None

    def encode(text: str) -> list[int]:
        return [stoi[c] for c in text]

    def decode(ids: list[int]) -> str:
        return "".join(itos[i] for i in ids)

    return encode, decode


def _bpe_codec() -> (
    tuple[Callable[[str], list[int]], Callable[[list[int]], str]] | None
):
    """(encode, decode) for the GPT-2 BPE via tiktoken — the codec for BPE
    corpora (OpenWebText) that ship no meta.pkl. None if tiktoken isn't
    installed. decode drops ids in the model's padded-vocab tail (>= 50257,
    since we build with vocab 50304): an undertrained model can still emit them
    and tiktoken has no mapping, so they'd raise — they carry no text anyway."""
    try:
        import tiktoken
    except ImportError:
        return None
    enc = tiktoken.get_encoding("gpt2")

    def encode(text: str) -> list[int]:
        return enc.encode_ordinary(text)

    def decode(ids: list[int]) -> str:
        return enc.decode([i for i in ids if 0 <= i < enc.n_vocab])

    return encode, decode


@torch.no_grad()
def generate_samples(
    raw_model,
    cfg: TrainConfig,
    step: int,
    *,
    prompts: list[str] | None = None,
    max_new_tokens: int | None = None,
) -> dict | None:
    """Generate completions for the prompt series on the UNWRAPPED model.

    Returns the samples document (see schema in CLAUDE-adjacent plan) or None
    when sampling isn't possible (no char codec, no prompts). Restores all RNG
    state on exit, so the caller's training stream is bit-identical whether or
    not sampling ran.
    """
    prompts = cfg.sample_prompts if prompts is None else prompts
    max_new_tokens = cfg.sample_max_new_tokens if max_new_tokens is None else max_new_tokens
    if not prompts:
        print("[sample] skipped: no prompts configured", file=sys.stderr)
        return None
    # Char-level (meta.pkl) if present, else GPT-2 BPE (tiktoken) for OWT & co.
    codec = _char_codec(cfg.data_local_dir) or _bpe_codec()
    if codec is None:
        print(
            f"[sample] skipped: no char codec (meta.pkl) in {cfg.data_local_dir} "
            "and tiktoken not installed for the BPE fallback",
            file=sys.stderr,
        )
        return None
    encode, decode = codec

    saved = rng.capture()
    was_training = raw_model.training
    raw_model.eval()
    samples: list[dict] = []
    try:
        # Seeded per (seed, step): reproducible, and a resumed run re-crossing
        # this step regenerates byte-identical output (idempotent S3 overwrite).
        torch.manual_seed(cfg.seed + step)
        for prompt in prompts:
            text = prompt or "\n"  # generate() requires a non-empty idx
            try:
                ids = encode(text)
            except KeyError:
                print(f"[sample] skipped prompt {prompt!r}: chars outside vocab", file=sys.stderr)
                continue
            idx = torch.tensor(ids, dtype=torch.long, device=cfg.device)[None, ...]
            for k in range(cfg.samples_per_prompt):
                out = raw_model.generate(
                    idx,
                    max_new_tokens,
                    temperature=cfg.sample_temperature,
                    top_k=cfg.sample_top_k,
                )
                samples.append(
                    {
                        "prompt": prompt,
                        "sample_index": k,
                        "completion": decode(out[0, len(ids) :].tolist()),
                    }
                )
    finally:
        if was_training:
            raw_model.train()
        rng.restore(saved)

    if not samples:
        return None
    return {
        "run_id": cfg.run_id,
        "step": step,
        "dataset": cfg.dataset,
        "params": {
            "max_new_tokens": max_new_tokens,
            "temperature": cfg.sample_temperature,
            "top_k": cfg.sample_top_k,
            "samples_per_prompt": cfg.samples_per_prompt,
            "seed": cfg.seed + step,
        },
        "samples": samples,
    }


def write_samples(doc: dict, uri: str) -> None:
    """Serialize a samples document to a local path or s3:// URI."""
    s3_store.put_bytes(json.dumps(doc, indent=2).encode(), uri)


def snapshot_uri(cfg: TrainConfig, step: int) -> str:
    """Deterministic per-step snapshot key (idempotent across resume re-runs)."""
    base = cfg.samples_prefix_uri
    if not base.endswith("/"):
        base += "/"
    return f"{base}step-{step:012d}.json"
