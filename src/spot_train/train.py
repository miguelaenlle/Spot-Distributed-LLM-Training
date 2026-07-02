"""Training entrypoint — one resume code path.

Phase 1a: single node, single device (CPU for the determinism test, one GPU on
spot). The model comes from the nanoGPT submodule; this file owns only the
fault-tolerance loop around it.

The invariant the whole project rests on:

    startup always tries to restore the latest checkpoint and falls back to
    fresh — there is never a separate "resume" branch to drift out of sync.
"""

from __future__ import annotations

import sys

import torch

from .checkpoint import load_latest, restore_into, save
from .config import TrainConfig
from .data import PositionedLoader
from .interruption import InterruptionListener


def build_model(cfg: TrainConfig):
    """Instantiate nanoGPT's GPT from the submodule. We import, never rewrite."""
    # third_party/nanoGPT is on sys.path via editable install / conftest.
    from model import GPT, GPTConfig  # type: ignore  # noqa: E402

    gpt_cfg = GPTConfig(
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    )
    return GPT(gpt_cfg)


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)

    model = build_model(cfg).to(cfg.device)
    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (0.9, 0.95), cfg.device
    )
    loader = PositionedLoader(
        dataset_dir="third_party/nanoGPT/data/shakespeare_char",
        batch_size=cfg.batch_size,
        block_size=cfg.block_size,
        device=cfg.device,
    )

    # --- the one resume code path --------------------------------------- #
    blob = load_latest(cfg.checkpoint_uri, map_location=cfg.device)
    start_step = restore_into(blob, model=model, optimizer=optimizer, loader=loader) if blob else 0
    if blob:
        print(f"[resume] restored from step {start_step}", file=sys.stderr)
    else:
        print("[fresh] no checkpoint found, starting from step 0", file=sys.stderr)

    listener = InterruptionListener().start()

    step = start_step
    while step < cfg.max_steps:
        x, y = loader.next_batch()
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1

        # periodic checkpoint — bounds worst-case lost work to the interval
        if step % cfg.checkpoint_interval == 0:
            save(model=model, optimizer=optimizer, loader=loader, step=step,
                 uri=cfg.checkpoint_uri)

        # preemption notice -> final checkpoint + clean exit
        if listener.should_stop.is_set():
            print(f"[preempt] signal at step {step}; final checkpoint", file=sys.stderr)
            save(model=model, optimizer=optimizer, loader=loader, step=step,
                 uri=cfg.checkpoint_uri)
            listener.stop()
            return

    listener.stop()


def main() -> None:
    train(TrainConfig())


if __name__ == "__main__":
    main()
