# Spot-Distributed LLM Training

Training SOTA LLMs on multiple GPU nodes is expensive. This project cuts the
cost by running on **spot instances** and engineering the training stack to
treat preemption as a normal, survivable event.

> **Phase 1:** spot on AWS · **Phase 2:** heterogeneous spot across AWS +
> RunPod.

## Why spot

Spot / interruptible GPUs are 60–90% cheaper than on-demand, but they can be
reclaimed with little or no warning. The bet: if the training loop can
checkpoint and resume **without losing correctness**, and a control plane can
replace lost nodes automatically, the price difference more than pays for the
recovery overhead.

## What "survives preemption" means

A run that is killed and resumed must reach the **same loss** as an
uninterrupted run. That requires checkpointing everything that affects the next
step — not just weights and optimizer state, but **all RNG states** and the
**data-loader position** — and restoring it through a single code path.

## Roadmap

| Phase | Milestone |
|-------|-----------|
| 1a | NanoGPT · 1 node / 1 GPU on AWS spot, survives preemption. |
| 1b | NanoGPT · 1 node / 4 GPUs (DDP + elastic agent). |
| 1c | NanoGPT · multi-node spot + Go control plane; beat the on-demand cost baseline within (1+ε)·T. |
| 1d | Real Llama-arch model on the same system. |
| 2  | Llama on heterogeneous spot (AWS + RunPod), actively waiting for hardware. |

See [`CLAUDE.md`](./CLAUDE.md) for the full plan of record and
[`docs/`](./docs) for architecture and per-phase notes.

## Layout

Minimal by design — only Phase 1a lives here today. Later phases add folders
(`infra/`, `supervisor/`) when they begin.

```
src/spot_train/   fault-tolerance layer (checkpoint, resume, interruption listener)
third_party/      Karpathy's nanoGPT as a pinned submodule — we import the model, not rewrite it
tests/            kill-and-resume determinism tests
```

After cloning: `git submodule update --init` then `pip install -e .`.

## Status

🚧 **Phase 1a — scaffolding.** Get kill-and-resume passing locally on CPU
before moving to spot. No AWS, DDP, or control plane yet.

## License

MIT — see [`LICENSE`](./LICENSE).
