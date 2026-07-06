# Spot-Distributed LLM Training → OptiTrain

Training SOTA LLMs on multiple GPU nodes is expensive. This project cuts the
cost by running on **spot instances** and engineering the training stack to
treat preemption as a normal, survivable event — and is growing into
**OptiTrain**: a unified platform to pretrain, RL-finetune, and serve LLMs
affordably, orchestrated by Go and Kubernetes.

> **Foundation (built):** fault-tolerant spot training on AWS ·
> **Next ([ROADMAP.md](./ROADMAP.md)):** inference fleet, GRPO finetuning,
> and the unified OptiTrain platform.

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
| OptiTrain 1–7 | Inference fleet (Python → Go/K8s, stress-tested, on spot) · GRPO RL finetuning (toy → ~1B → disaggregated) · one platform: pick a model, train, finetune, serve. |

See [`ROADMAP.md`](./ROADMAP.md) for the part-by-part plan of record,
[`CLAUDE.md`](./CLAUDE.md) for working guidance, and [`docs/`](./docs) for
architecture and per-phase notes.

## Layout

Minimal by design — folders appear when their roadmap part begins.

```
src/spot_train/   remote trainer: checkpoint/resume loop + wall-clock budget + eval
src/orchestrator/ local control plane (boto3): setup / experiments / fleet
src/inference/    serving fleet (ROADMAP Part 1): worker + router
loadgen/          Go load generator / stress-test harness (ROADMAP Part 1)
third_party/      Karpathy's nanoGPT as a pinned submodule — we import the model, not rewrite it
docs/iam/         least-privilege IAM policies (controller / worker / setup)
tests/            checkpoint/resume + fleet tests
```

After cloning: `git submodule update --init` then `pip install -e .`.

## Watch a run live

The training box has **no inbound ports** — you attach over SSM Session Manager
(the orchestrator prints the exact command with the instance id when it
launches). `setup` grants the box the SSM role automatically.

```bash
aws ssm start-session --target <instance-id> --region us-east-1
# then, on the box:
sudo tail -f /var/log/spot-train-boot.log   # live per-step loss / tok/s
nvidia-smi                                  # confirm the GPU is actually busy
```

The trainer prints a `[gpu] using cuda: <name>` banner at startup (and fails
fast if CUDA was requested but is missing), logs `step N: loss …, ms/step,
tok/s` every few steps, and records `cuda`/`gpu` in the final `metrics.json`.

## Status

✅ **Training foundation.** Single-node, DDP, and multi-node spot training all
run on AWS with kill/resume proven (controlled preemptions, pause-and-replace
recovery, per-run cost ledger + profiles, async checkpointing, baked AMIs).

🚧 **OptiTrain Part 1** — inference fleet MVP (Python router + spot workers
serving trained checkpoints) and the Go load generator. See
[`ROADMAP.md`](./ROADMAP.md).

## License

MIT — see [`LICENSE`](./LICENSE).
