"""Training entrypoint — one resume code path, one time-budgeted loop.

Phase 1a: single node, single device (CPU for the local test, one GPU on spot).
The model comes from the nanoGPT submodule; this file owns only the
fault-tolerance loop around it.

Invariants:
  - startup always tries to restore the latest checkpoint and falls back to
    fresh — there is never a separate "resume" branch to drift out of sync;
  - the loop stops on a wall-clock budget (``max_seconds``), then evaluates and
    writes ``metrics.json`` — so a launch is a fixed-duration unit of work the
    orchestrator can schedule and kill.
"""

from __future__ import annotations

import json
import sys
import time

import torch

from . import checkpoint, s3_store
from .config import TrainConfig
from .data import PositionedLoader
from .interruption import InterruptionListener


def _ensure_nanogpt_on_path() -> None:
    """Put the nanoGPT submodule on sys.path so ``from model import GPT`` works.

    Works for an editable install (local or on the box): repo root is two levels
    above this file (src/spot_train/train.py -> repo root), and nanoGPT lives at
    <root>/third_party/nanoGPT. Lets `spot-train` and `python -m spot_train.train`
    run without a manual PYTHONPATH.
    """
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ng = os.path.join(root, "third_party", "nanoGPT")
    if os.path.isdir(ng) and ng not in sys.path:
        sys.path.insert(0, ng)


def build_model(cfg: TrainConfig, vocab_size: int):
    """Instantiate nanoGPT's GPT from the submodule. We import, never rewrite."""
    _ensure_nanogpt_on_path()
    from model import GPT, GPTConfig  # type: ignore

    gpt_cfg = GPTConfig(
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
        vocab_size=vocab_size,
    )
    return GPT(gpt_cfg)


@torch.no_grad()
def estimate_loss(model, loader: PositionedLoader, eval_iters: int) -> dict[str, float]:
    """Mean train/val loss over ``eval_iters`` batches (nanoGPT-style)."""
    model.eval()
    out: dict[str, float] = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = loader.next_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = float(losses.mean().item())
    model.train()
    return out


def _write_metrics(cfg: TrainConfig, metrics: dict) -> None:
    s3_store.put_bytes(json.dumps(metrics, indent=2).encode(), cfg.metrics_uri)
    print(f"[metrics] {json.dumps(metrics)}", file=sys.stderr)


def train(cfg: TrainConfig) -> dict:
    device_type = "cuda" if cfg.device.startswith("cuda") else "cpu"
    torch.manual_seed(cfg.seed)

    loader = PositionedLoader(
        data_local_dir=cfg.data_local_dir,
        batch_size=cfg.batch_size,
        block_size=cfg.block_size,
        device=cfg.device,
        data_uri=cfg.data_uri,
    )
    vocab_size = loader.vocab_size or 50304

    def make_model():
        return build_model(cfg, vocab_size)

    model = make_model().to(cfg.device)
    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (0.9, 0.95), device_type
    )

    # --- the one resume code path --------------------------------------- #
    blob = checkpoint.load_latest(cfg.checkpoint_uri, map_location=cfg.device)
    resumed = blob is not None
    if resumed:
        start_step = checkpoint.restore_into(blob, model=model, optimizer=optimizer, loader=loader)
        print(f"[resume] restored from step {start_step}", file=sys.stderr)
    else:
        start_step = 0
        print("[fresh] no checkpoint found, starting from step 0", file=sys.stderr)

    listener = InterruptionListener().start()

    def do_checkpoint(step: int, ckpt_count: int) -> None:
        ref = checkpoint.save(
            model=model, optimizer=optimizer, loader=loader, step=step, uri=cfg.checkpoint_uri
        )
        # Every Nth checkpoint, prove the written artifact reconstructs a model.
        if cfg.smoke_test_every and ckpt_count % cfg.smoke_test_every == 0:
            good = checkpoint.verify(ref, map_location=cfg.device)
            checkpoint.smoke_test(good, make_model, loader.next_batch("val"), cfg.device)
            print(f"[verify] checkpoint at step {step} passed verify + smoke test", file=sys.stderr)

    start_time = time.monotonic()
    last_ckpt = start_time
    step = start_step
    ckpt_count = 0
    reason = "max_steps"

    while step < cfg.max_steps:
        x, y = loader.next_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1

        now = time.monotonic()
        if now - last_ckpt >= cfg.checkpoint_interval_seconds:
            ckpt_count += 1
            do_checkpoint(step, ckpt_count)
            last_ckpt = now

        if listener.should_stop.is_set():
            reason = "preempt"
            break
        if cfg.max_seconds is not None and now - start_time >= cfg.max_seconds:
            reason = "time_budget"
            break

    # Final checkpoint so no tail work is lost, then evaluate + report.
    ckpt_count += 1
    do_checkpoint(step, ckpt_count)
    listener.stop()

    final_step = step
    losses = estimate_loss(model, loader, cfg.eval_iters)
    metrics = {
        "run_id": cfg.run_id,
        "market": cfg.market,
        "resumed": resumed,
        "steps": final_step,
        "steps_this_launch": final_step - start_step,
        "train_loss": losses["train"],
        "val_loss": losses["val"],
        "wallclock_s": round(time.monotonic() - start_time, 2),
        "stop_reason": reason,
        "device": cfg.device,
        "dataset": cfg.dataset,
    }
    _write_metrics(cfg, metrics)
    return metrics


def main() -> None:
    train(TrainConfig.from_env())


if __name__ == "__main__":
    main()
