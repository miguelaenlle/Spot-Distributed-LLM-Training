"""Text-sampling + convergence-recipe tests — hermetic (tiny CPU GPT, tmp_path).

Pins: the samples.json document shape, the skip paths (no codec / no prompts /
out-of-vocab), RNG isolation (training stream bit-identical whether or not
sampling ran), prompt-list env parsing (raw JSON and base64), the LR schedule
math, and the deterministic full-pass eval iterator (full coverage, zero RNG).
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
import torch

from spot_train import sampling
from spot_train.config import TrainConfig
from spot_train.data import PositionedLoader
from spot_train.train import get_lr

pytest.importorskip("model")  # nanoGPT submodule (git submodule update --init)


_CHARS = "\n abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:.,"


def _write_char_dataset(dirpath, n_tokens: int = 2000) -> None:
    stoi = {c: i for i, c in enumerate(_CHARS)}
    itos = dict(enumerate(_CHARS))
    meta = {"vocab_size": len(_CHARS), "stoi": stoi, "itos": itos}
    with open(dirpath / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    rng = np.random.default_rng(0)
    for name, n in (("train.bin", n_tokens), ("val.bin", n_tokens // 2)):
        rng.integers(0, len(_CHARS), n, dtype=np.uint16).tofile(dirpath / name)


def _tiny_cfg(tmp_path, **overrides) -> TrainConfig:
    defaults = {
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 16,
        "block_size": 16,
        "batch_size": 4,
        "device": "cpu",
        "data_local_dir": str(tmp_path),
        "samples_uri": str(tmp_path / "samples.json"),
        "samples_prefix_uri": str(tmp_path / "samples/"),
        "sample_max_new_tokens": 8,
        "sample_top_k": 10,
        "run_id": "test-run",
    }
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _tiny_model(cfg: TrainConfig):
    from spot_train.train import build_model

    torch.manual_seed(0)
    return build_model(cfg, vocab_size=len(_CHARS))


def test_generate_samples_document_shape(tmp_path):
    _write_char_dataset(tmp_path)
    cfg = _tiny_cfg(tmp_path, sample_prompts=["JULIET:", ""])
    doc = sampling.generate_samples(_tiny_model(cfg), cfg, step=42)
    assert doc is not None
    assert doc["run_id"] == "test-run" and doc["step"] == 42
    assert doc["params"]["max_new_tokens"] == 8
    assert doc["params"]["seed"] == cfg.seed + 42
    assert len(doc["samples"]) == 2
    s = doc["samples"][0]
    assert s["prompt"] == "JULIET:" and s["sample_index"] == 0
    # Completion excludes the prompt and has exactly max_new_tokens chars.
    assert not s["completion"].startswith("JULIET:")
    assert len(s["completion"]) == 8
    # write + parse round-trip (what train.py does with the doc)
    sampling.write_samples(doc, cfg.samples_uri)
    assert json.loads((tmp_path / "samples.json").read_text())["step"] == 42


def test_generate_samples_deterministic_per_step(tmp_path):
    """Same (seed, step) => byte-identical output — a resumed run re-crossing a
    snapshot step overwrites the S3 key with identical content."""
    _write_char_dataset(tmp_path)
    cfg = _tiny_cfg(tmp_path)
    model = _tiny_model(cfg)
    a = sampling.generate_samples(model, cfg, step=100)
    b = sampling.generate_samples(model, cfg, step=100)
    c = sampling.generate_samples(model, cfg, step=200)
    assert a == b
    assert a != c  # a different step reseeds differently


def test_generate_samples_restores_rng_and_mode(tmp_path):
    _write_char_dataset(tmp_path)
    cfg = _tiny_cfg(tmp_path)
    model = _tiny_model(cfg)
    model.train()
    before = torch.get_rng_state()
    sampling.generate_samples(model, cfg, step=7)
    assert torch.equal(before, torch.get_rng_state())  # training stream untouched
    assert model.training  # train mode restored


def test_generate_samples_skip_paths(tmp_path, monkeypatch):
    _write_char_dataset(tmp_path)
    cfg = _tiny_cfg(tmp_path)
    model = _tiny_model(cfg)
    # no prompts
    assert sampling.generate_samples(model, cfg, 1, prompts=[]) is None
    # prompt with chars outside the vocab is skipped individually
    doc = sampling.generate_samples(model, cfg, 1, prompts=["ROMEO:", "Ünïcode"])
    assert [s["prompt"] for s in doc["samples"]] == ["ROMEO:"]
    # With NO char codec and the BPE fallback unavailable (tiktoken absent), skip.
    monkeypatch.setattr(sampling, "_bpe_codec", lambda: None)
    # meta.pkl without stoi/itos (non-char dataset) => None
    with open(tmp_path / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": 50304}, f)
    assert sampling.generate_samples(model, cfg, 1) is None
    # no meta.pkl at all => None
    (tmp_path / "meta.pkl").unlink()
    assert sampling.generate_samples(model, cfg, 1) is None


def test_bpe_codec_roundtrip_and_padding_filter():
    """GPT-2 BPE codec round-trips text and drops padded-vocab ids (>= 50257)
    that our vocab-50304 model can emit but tiktoken can't map."""
    tiktoken = pytest.importorskip("tiktoken")
    codec = sampling._bpe_codec()
    assert codec is not None
    encode, decode = codec
    assert decode(encode("Hello, world!")) == "Hello, world!"
    n_vocab = tiktoken.get_encoding("gpt2").n_vocab  # 50257
    # ids in the padded tail are silently dropped, not raised on
    assert decode([*encode("hi"), n_vocab, n_vocab + 46]) == "hi"


def test_generate_samples_bpe_fallback_for_owt(tmp_path):
    """A BPE dataset (no meta.pkl) samples via tiktoken against a vocab-50304
    model — the OpenWebText path that previously produced no outputs."""
    pytest.importorskip("tiktoken")
    # BPE bins, deliberately NO meta.pkl (matches openwebtext_300m staging).
    rng = np.random.default_rng(0)
    for name, n in (("train.bin", 2000), ("val.bin", 1000)):
        rng.integers(0, 50304, n, dtype=np.uint16).tofile(tmp_path / name)
    cfg = _tiny_cfg(tmp_path, dataset="openwebtext_300m", sample_prompts=["The"])
    from spot_train.train import build_model

    torch.manual_seed(0)
    model = build_model(cfg, vocab_size=50304)
    doc = sampling.generate_samples(model, cfg, step=3)
    assert doc is not None
    assert doc["samples"][0]["prompt"] == "The"
    assert isinstance(doc["samples"][0]["completion"], str)


def test_snapshot_uri_zero_padded(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    assert sampling.snapshot_uri(cfg, 1000).endswith("samples/step-000000001000.json")
    cfg.samples_prefix_uri = str(tmp_path / "samples")  # no trailing slash
    assert sampling.snapshot_uri(cfg, 5).endswith("samples/step-000000000005.json")


def test_prompts_from_env(monkeypatch):
    import base64

    prompts = ["ROMEO:", "line one\nline two"]
    monkeypatch.setenv("SAMPLE_PROMPTS", json.dumps(prompts))
    assert TrainConfig.from_env().sample_prompts == prompts
    monkeypatch.setenv("SAMPLE_PROMPTS", base64.b64encode(json.dumps(prompts).encode()).decode())
    assert TrainConfig.from_env().sample_prompts == prompts
    monkeypatch.delenv("SAMPLE_PROMPTS")
    assert TrainConfig.from_env().sample_prompts  # defaults are non-empty


def test_get_lr_schedule():
    cfg = TrainConfig(learning_rate=1e-3, warmup_steps=100, lr_decay_steps=5000, min_lr=1e-4)
    assert get_lr(cfg, 0) < 1e-3  # warming up
    assert get_lr(cfg, 99) < 1e-3
    assert get_lr(cfg, 100) == pytest.approx(1e-3, rel=1e-6)  # warmup done
    mid = get_lr(cfg, 2550)
    assert 1e-4 < mid < 1e-3  # cosine interior
    assert get_lr(cfg, 5000) == 1e-4  # floor
    assert get_lr(cfg, 99999) == 1e-4
    # schedule disabled (defaults) => constant LR at every step
    flat = TrainConfig(learning_rate=6e-4)
    assert get_lr(flat, 0) == get_lr(flat, 10_000) == 6e-4


def test_iter_eval_batches_full_coverage_and_no_rng(tmp_path):
    _write_char_dataset(tmp_path, n_tokens=2000)
    loader = PositionedLoader(str(tmp_path), batch_size=4, block_size=16, device="cpu")
    before = torch.get_rng_state()
    batches = list(loader.iter_eval_batches("val"))
    # zero RNG consumed and no loader-position advance
    assert torch.equal(before, torch.get_rng_state())
    assert loader.state.step == 0
    # full coverage: every non-overlapping window exactly once, remainder batch kept
    n_windows = (1000 - 1) // 16
    assert sum(x.shape[0] for x, _ in batches) == n_windows
    # deterministic: a second pass yields identical tensors
    again = list(loader.iter_eval_batches("val"))
    assert all(
        torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        for a, b in zip(batches, again, strict=True)
    )
    # y is x shifted by one
    x, y = batches[0]
    assert torch.equal(x[0, 1:], y[0, :-1])
