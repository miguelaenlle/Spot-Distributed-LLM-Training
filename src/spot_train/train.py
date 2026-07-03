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

from . import checkpoint, distributed, s3_store
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
    torch.manual_seed(cfg.seed)

    # Resolve the device from the actual machine (the usual ML-training pattern):
    # "auto" -> cuda if present else cpu. An explicit "cuda" that isn't available
    # is a hard error (a GPU run silently falling back to CPU is worse).
    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
    if cfg.device == "auto":
        cfg.device = "cuda" if cuda_ok else "cpu"
    if cfg.device.startswith("cuda") and not cuda_ok:
        raise SystemExit(
            "DEVICE=cuda but torch.cuda.is_available() is False — GPU driver / "
            "torch-CUDA build mismatch on this box. Check the AMI has PyTorch+CUDA."
        )

    # DDP context (dormant unless torchrun set RANK). Under DDP on a GPU box the
    # device becomes cuda:<local_rank>; on CPU it stays "cpu" with the gloo backend.
    ddp = distributed.init(cfg.device)
    cfg.device = ddp.device
    device_type = "cuda" if cfg.device.startswith("cuda") else "cpu"

    def log(msg: str) -> None:  # only rank 0 prints
        if ddp.master:
            print(msg, file=sys.stderr, flush=True)

    log(f"[gpu] using cuda: {gpu_name}" if device_type == "cuda" else "[cpu] running on CPU")
    if ddp.enabled:
        log(f"[ddp] world_size={ddp.world_size} rank={ddp.rank} data_mode={cfg.data_mode}")

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

    # Build + configure + resume on the UNWRAPPED model. DDP does not broadcast
    # optimizer state, so every rank loads the checkpoint to get identical state.
    raw_model = make_model().to(cfg.device)
    optimizer = raw_model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (0.9, 0.95), device_type
    )

    # --- the one resume code path --------------------------------------- #
    blob = checkpoint.load_latest(cfg.checkpoint_uri, map_location=cfg.device)
    resumed = blob is not None
    if resumed:
        start_step = checkpoint.restore_into(
            blob, model=raw_model, optimizer=optimizer, loader=loader
        )
        log(f"[resume] restored from step {start_step}")
    else:
        start_step = 0
        log("[fresh] no checkpoint found, starting from step 0")

    # Wrap DDP strictly AFTER resume/configure (device_ids=None on CPU/gloo). A fresh
    # run's DDP construction broadcasts rank-0 weights so all ranks start identical.
    model = raw_model
    if ddp.enabled:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(raw_model, device_ids=[ddp.local_rank] if device_type == "cuda" else None)
        if cfg.data_mode == "shard":  # per-rank data stream (overwrites restored RNG)
            torch.manual_seed(cfg.seed + ddp.rank)

    listener = InterruptionListener().start()

    def do_checkpoint(step: int, ckpt_count: int) -> None:
        if not ddp.master:  # only rank 0 writes to S3
            return
        ref = checkpoint.save(
            model=raw_model, optimizer=optimizer, loader=loader, step=step, uri=cfg.checkpoint_uri
        )
        # Every Nth checkpoint, prove the written artifact reconstructs a model.
        if cfg.smoke_test_every and ckpt_count % cfg.smoke_test_every == 0:
            good = checkpoint.verify(ref, map_location=cfg.device)
            checkpoint.smoke_test(good, make_model, loader.next_batch("val"), cfg.device)
            log(f"[verify] checkpoint at step {step} passed verify + smoke test")

    start_time = time.monotonic()
    # Epoch at loop start: the orchestrator's run profile uses this to split
    # provisioning (launch -> here: boot+clone+pip+dataset) from training (the loop).
    train_started_at = time.time()
    last_ckpt = start_time
    last_log_time = start_time
    last_log_step = start_step
    step = start_step
    ckpt_count = 0
    reason = "max_steps"

    while step < cfg.max_steps:
        x, y = loader.next_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # DDP averages gradients across ranks here
        optimizer.step()
        step += 1

        # per-step progress (nanoGPT-style) — tail the box log to watch this live
        if cfg.log_interval_steps and step % cfg.log_interval_steps == 0:
            gloss = distributed.mean_loss(ddp, loss.item())  # all ranks participate
            if ddp.master:
                t = time.monotonic()
                per_step = (t - last_log_time) / max(1, step - last_log_step)
                toks = ddp.world_size * cfg.batch_size * cfg.block_size
                tok_s = toks / per_step if per_step else 0
                print(
                    f"step {step}: loss {gloss:.4f}, {per_step * 1000:.0f}ms/step, "
                    f"{tok_s:.0f} tok/s",
                    file=sys.stderr,
                    flush=True,
                )
                last_log_time, last_log_step = t, step

        now = time.monotonic()
        if now - last_ckpt >= cfg.checkpoint_interval_seconds:
            ckpt_count += 1
            do_checkpoint(step, ckpt_count)  # rank-0-only body
            last_ckpt = now

        # Coordinated stop — the LAST collective every step, so all ranks break on
        # the same iteration and none is left blocking on the next backward.
        if listener.should_stop.is_set():
            reason = "preempt"
        elif cfg.max_seconds is not None and now - start_time >= cfg.max_seconds:
            reason = "time_budget"
        if distributed.all_reduce_stop(ddp, reason != "max_steps"):
            break

    # Agree the stop REASON across ranks (balanced collective — every rank broke the
    # loop together). If ANY rank saw the SIGTERM, all treat it as preempt so rank 0
    # checkpoints-and-exits rather than finalizing a run that was actually preempted.
    if ddp.enabled:
        if distributed.all_reduce_stop(ddp, reason == "preempt"):
            reason = "preempt"
        elif reason == "max_steps":
            reason = "time_budget"

    # Graceful preemption (SIGTERM from the orchestrator — a stand-in for a Spot
    # reclaim): checkpoint all work up to the signal, then exit FAST. No eval, no
    # metrics.json — metrics.json is reserved for a COMPLETED budget, so the
    # orchestrator can treat its appearance as an unambiguous "run done". Only rank
    # 0 writes; no collective follows, so non-master ranks just shut down and exit.
    if reason == "preempt":
        ckpt_count += 1
        do_checkpoint(step, ckpt_count)
        listener.stop()
        log(f"[preempt] checkpointed at step {step}; exiting for replacement")
        distributed.shutdown(ddp)
        return {"run_id": cfg.run_id, "stop_reason": "preempt", "steps": step, "resumed": resumed}

    # Training loop is done — stamp its wall-clock, then save + evaluate (each timed
    # separately so the run profile shows the real breakdown, not "eval as saves").
    # Everything here is rank-0 only; non-master ranks shut down and exit immediately
    # (freeing the CPU for rank-0's eval); torchrun waits for all ranks to finish.
    train_s = round(time.monotonic() - start_time, 2)
    listener.stop()
    if not ddp.master:
        distributed.shutdown(ddp)
        return {"run_id": cfg.run_id, "stop_reason": reason, "steps": step, "resumed": resumed}

    save_t0 = time.monotonic()
    ckpt_count += 1
    do_checkpoint(step, ckpt_count)
    save_s = round(time.monotonic() - save_t0, 2)

    final_step = step
    eval_t0 = time.monotonic()
    losses = estimate_loss(raw_model, loader, cfg.eval_iters)  # unwrapped: no collective
    eval_s = round(time.monotonic() - eval_t0, 2)

    metrics = {
        "run_id": cfg.run_id,
        "market": cfg.market,
        "resumed": resumed,
        "steps": final_step,
        "steps_this_launch": final_step - start_step,
        "train_loss": losses["train"],
        "val_loss": losses["val"],
        "wallclock_s": round(time.monotonic() - start_time, 2),
        # Exact per-phase wall-clock for the run-profile timeline. train_started_at
        # (epoch) lets the orchestrator derive provisioning = train_started_at - launch.
        "train_started_at": round(train_started_at, 3),
        "phases": {"train_s": train_s, "save_s": save_s, "eval_s": eval_s},
        "stop_reason": reason,
        "device": cfg.device,
        "cuda": cuda_ok,
        "gpu": gpu_name,
        "dataset": cfg.dataset,
        "world_size": ddp.world_size,
    }
    _write_metrics(cfg, metrics)
    distributed.shutdown(ddp)
    return metrics


def main() -> None:
    train(TrainConfig.from_env())


if __name__ == "__main__":
    main()
