# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## Problem

Training SOTA LLMs on multiple GPU nodes is expensive.

## Solution

Reduce the cost using **spot instances** — accept preemption as normal and
engineer the training loop + control plane to survive it.

- **Phase 1:** Spot instances on AWS.
- **Phase 2:** Spot instances on AWS, Lambda, and RunPod (heterogeneous,
  spot-distributed training).

## Goals

| Phase | Goal |
|-------|------|
| **1a** | Train NanoGPT on AWS spot — **1 node, 1 GPU** — successfully handling preemption. |
| **1b** | Train NanoGPT on AWS spot — **1 node, 4 GPUs** — successfully handling preemption. |
| **1c** | On-demand baseline: NanoGPT on 4 nodes × 4 GPUs/node takes time **T** and cost **C**. Train on AWS **spot** with N nodes, X GPUs/node in time **T** for **less than C**. |
| **1d** | Same as 1c but a real **Llama-arch** model. |
| **2**  | Train a Llama model on **heterogeneous spot** (AWS + RunPod), actively waiting for hardware availability. |

## Guiding principles

- **Don't write the model.** Start from Karpathy's
  [nanoGPT](https://github.com/karpathy/nanoGPT); we own the fault-tolerance
  layer, not the transformer.
- **One resume code path.** Startup always tries to restore the latest
  checkpoint and falls back to fresh — never two branches.
- **Determinism first.** Get kill-and-resume passing locally on CPU before
  touching a spot instance. Determinism is miserable to debug on a vanishing
  machine.
- **Checkpoint everything that affects the next step:** model weights,
  optimizer state, step number, **all RNG states**, and **data-loader
  position**. The last two are what keep resume from silently diverging.
- **Atomic writes.** Write to a temp key, then atomically rename, so a
  mid-write kill can't corrupt the last good checkpoint.
- **Assume no warning.** Poll IMDS + handle SIGTERM, but also checkpoint
  periodically regardless — some kills give no notice.

---

## Phase 1a — detailed plan (current focus)

**Goal:** train NanoGPT on a single AWS spot GPU that survives preemption.
**Success:** a killed-and-resumed run reaches the same loss as an
uninterrupted one.

**Checkpoint** (`src/spot_train/checkpoint.py`)
- Save model weights, optimizer state, step number, all RNG states, and
  data-loader position.
- Write to a temp key and atomically rename → upload to S3.

**Interruption listener** (`src/spot_train/interruption.py`)
- Background thread polling the IMDS spot-action endpoint every few seconds.
- On notice: write a final checkpoint and exit cleanly.
- SIGTERM handler as backup. Periodic checkpointing regardless.

**Resume** (`src/spot_train/train.py`)
- On startup, restore the latest S3 checkpoint if present, else start fresh —
  one code path for both.

**Infra** (`infra/`)
- One spot GPU (`g4dn`/`g5.xlarge`), an S3 bucket in the **same region**, an
  IAM role linking them.
- ⚠️ **File the G-class spot quota increase today** — fresh accounts sit at
  zero and approval takes days.

**Order of work**
1. Get checkpoint/resume passing the kill-and-resume test **locally on CPU**.
2. Move to spot.
3. Add the interruption listener.
4. Catch a **real** preemption (force one with **AWS FIS**).

**Log:** lost-steps-per-interruption — should never exceed the checkpoint
interval.

**Out of scope for 1a:** DDP, rendezvous, node replacement, the Go supervisor,
async checkpointing — all 1b onward.

---

## Phases 1b / 1c / 1d — high level

- **1b — 1 node, 4 GPUs:** add `torchrun` + DDP; adopt the elastic agent so
  survivors re-rendezvous when one process dies. Rendezvous is local, so no
  coordinator box yet. Same resume test, now across a DDP restart.
- **1c — multi-node spot + Go supervisor:** 4 nodes × 4 GPUs over the network;
  rendezvous store on a dedicated on-demand `t3.micro`. Build the Go control
  plane (**observe / compare / act**: watch health, launch replacement spot
  nodes, re-rendezvous). Add async two-tier checkpointing. **Headline
  experiment:** on-demand baseline vs. spot to the same target loss for under
  **C** within **(1+ε)·T**; report goodput, recovery time, lost work,
  idle-wait.
- **1d — real Llama-arch model:** same 1c system, bigger model. Shows the
  controller is model-agnostic and that η improves with model size, giving a
  cleaner savings story. Validation-at-scale, not new infra — watch memory
  (may need gradient checkpointing / sharding).

---

## Repository layout

Kept **deliberately minimal** — only what Phase 1a (kill-and-resume on CPU)
needs. Later phases add their own folders when they start (see "added later").

```
CLAUDE.md              # this file — the plan of record
README.md              # public overview
pyproject.toml         # src-layout package `spot_train`
src/spot_train/        # the fault-tolerance layer (Python) — OUR code
  checkpoint.py        # atomic save/restore of full training state
  s3_store.py          # S3 upload/download + atomic temp-key rename
  rng.py               # capture/restore all RNG states
  data.py              # data loader that tracks & restores its position
  interruption.py      # IMDS spot poller + SIGTERM handler
  config.py            # training/run configuration
  train.py             # entrypoint: single resume code path
third_party/nanoGPT/   # Karpathy's nanoGPT — git submodule, read-only.
                       #   we import GPT/GPTConfig from model.py; we do NOT
                       #   use their train.py — our loop owns fault tolerance.
tests/                 # kill-and-resume determinism tests

# added later (do not create until the phase begins):
#   infra/       — S3 bucket, IAM role, spot EC2, FIS drills   (1a→spot step)
#   supervisor/  — Go control plane (observe/compare/act)      (Phase 1c)
```

The nanoGPT submodule is pinned; contributors run
`git submodule update --init` after cloning.

## Conventions

- Python package lives under `src/spot_train` (src layout). Install editable:
  `pip install -e .`.
- Prefer running the CPU determinism test before any cloud work:
  `pytest tests/test_kill_resume.py`.
- Keep the Go supervisor untouched until Phase 1c; it has its own module.

### Linting / formatting (ruff)

`ruff` is the linter and formatter; config lives in `pyproject.toml`
(`third_party/` is excluded — nanoGPT is read-only). Two ways to run it
automatically on commit — enable one after cloning:

- **Native git hook (no extra install):** `git config core.hooksPath .githooks`
  — runs `ruff check --fix` + `ruff format` on staged Python via the `ruff`
  already on PATH.
- **pre-commit framework (shareable, pinned):** `pip install pre-commit &&
  pre-commit install` — uses `.pre-commit-config.yaml`.

Run manually: `ruff check --fix . && ruff format .`
