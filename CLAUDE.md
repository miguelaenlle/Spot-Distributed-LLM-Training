# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## Problem

Training SOTA LLMs on multiple GPU nodes is expensive.

## Solution

Reduce the cost using **spot instances** — accept preemption as normal and
engineer the training loop + control plane to survive it.

- **Phase 1 (training foundation):** Spot instances on AWS.
- **OptiTrain (current plan of record: [ROADMAP.md](./ROADMAP.md)):** a unified
  platform — pretrain on spot, RL-finetune, and serve at scale — orchestrated
  by Go/K8s with a static Firebase frontend and a minimal admin backend. This
  supersedes the old "Phase 2" (heterogeneous AWS + Lambda + RunPod), which is
  deferred as possible future work.

## Goals

| Phase | Goal |
|-------|------|
| **1a** | Train NanoGPT on AWS spot — **1 node, 1 GPU** — successfully handling preemption. |
| **1b** | Train NanoGPT on AWS spot — **1 node, 4 GPUs** — successfully handling preemption. |
| **1c** | On-demand baseline: NanoGPT on 4 nodes × 4 GPUs/node takes time **T** and cost **C**. Train on AWS **spot** with N nodes, X GPUs/node in time **T** for **less than C**. |
| **1d** | Same as 1c but a real **Llama-arch** model. |
| **OptiTrain 1–7** | Inference fleet (Python → Go/K8s, spot, stress-tested) · RL finetuning (GRPO toy → ~1B → disaggregated rollout fleet) · unified platform (cloud control plane + Firebase FE + minimal admin backend). Details: [ROADMAP.md](./ROADMAP.md). |

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

## Current focus

**OptiTrain Part 1 — inference fleet MVP (Python) + Go load generator.** See
[ROADMAP.md](./ROADMAP.md) for the full part-by-part plan; Parts 1–2 are at
implementation depth there. The training track below (1a onward) is built and
cloud-proven through multi-node spot preemption (single-node, DDP, and
multinode kill/resume experiments all run; cost ledger + run profiles in
place); the Go supervisor and the 1c headline comparison remain open and are
absorbed into ROADMAP Parts 3/7.

---

## Phase 1a — detailed plan (done; kept for reference)

**Shape:** a **local orchestrator** (`src/orchestrator/`, boto3) drives AWS and
runs two experiments; a **remote trainer** (`src/spot_train/`) runs on the GPU
box. S3 is the transport for dataset, checkpoints, and metrics.

**The two experiments**
1. **Baseline (on-demand):** spin up a regular GPU, train NanoGPT for a
   controllable budget (~5 min), report eval metrics + wallclock + cost.
2. **Spot-with-kill:** spin up a spot GPU, train ~2 min, **kill it**
   (`TerminateInstances`), then a second spot instance resumes from the S3
   checkpoint and trains ~3 more min. **Success:** segment 2 reports
   `resumed=true`, loss keeps dropping from the checkpoint (not reset), and
   total cost < the baseline.

**Remote trainer** (`src/spot_train/`)
- `train.py` — one resume code path (load-latest-or-fresh); a **wall-clock
  budget** (`max_seconds`) ends a launch, then it evaluates and writes
  `metrics.json` to S3.
- `checkpoint.py` — full-state save (weights, optimizer, step, all RNG,
  loader position) + `version`; `verify()` (keys + all-tensors-finite) and a
  post-save **restore smoke test** are the tools that confirm a checkpoint is
  comprehensive and valid.
- `s3_store.py` — local + S3 behind one interface; **atomic** temp-key→rename;
  S3 uploads carry a **SHA-256** checksum, downloads verify it.
- `data.py` — nanoGPT memmap batches; pulls prepared bins from S3 on first use.
- **Checkpointing is time-based** (`checkpoint_interval_seconds`, default 30) so
  worst-case lost work is bounded in wall-clock, synchronous for the MVP.

**Local orchestrator** (`src/orchestrator/`)
- `aws.py` — the **only** module that calls AWS; every mutating call logs first
  and honors `--dry-run`. `setup.py` (bucket + IAM instance profile + SG),
  `dataset.py` (prepare-once → S3), `bootstrap.py` (user-data), `experiments.py`
  (baseline / spot), `__main__.py` (CLI: `setup|stage-data|baseline|spot`).

**Credentials / who runs what:** the user's creds live in a git-ignored `.env`
(or an SSO profile); boto3 resolves them at runtime and no code reads them.
Role-first design — the same `aws.py` works with laptop SSO/keys now and an
attached instance-profile role when the orchestrator becomes a cloud node.
Least-privilege policies for each principal are in `docs/iam/` (controller,
worker, one-time setup). **The user runs every credentialed command**
(`setup`/`stage-data`/`baseline`/`spot`); Claude only writes code and runs local
CPU/lint/test.

**Infra** (created by `spot-orchestrate setup`, idempotent)
- One GPU (`g4dn.xlarge`, us-east-1) on the Deep Learning AMI, an S3 bucket, and
  an IAM instance profile granting the box S3 access.
- ⚠️ **File the G-class quota increase early** — needed for **both** on-demand
  (baseline) and spot (spot run); fresh accounts sit at zero, approval takes days.

**Order of work**
1. **(done)** Trainer + orchestrator implemented; prove locally on CPU that a
   time-budgeted run writes `metrics.json` and a killed+resumed run continues.
2. `spot-orchestrate setup` → `stage-data`.
3. `baseline` (on-demand).
4. `spot` (controlled kill + resume).

**Log:** lost-work-per-interruption — should never exceed the checkpoint interval.

**Out of scope for 1a (moved later):** the IMDS spot listener + AWS FIS
`SendSpotInstanceInterruptions` (real 2-min-notice preemption) — the controlled
kill stands in for now; DDP, rendezvous, node replacement, the Go supervisor, and
async checkpointing are all 1b onward. Bit-exact determinism is relaxed for the
MVP (invariant: loss continues from the checkpoint).

---

## Phases 1b / 1c / 1d — high level

- **1b — 1 node, 4 GPUs:** add `torchrun` + DDP; adopt the elastic agent so
  survivors re-rendezvous when one process dies. Rendezvous is local, so no
  coordinator box yet. Same resume test, now across a DDP restart.
- **1c — multi-node spot, EPOCH SUPERVISOR (current):** N ≤ 8 nodes. A single
  orchestrator owns membership by publishing monotonic **epoch documents**
  (`runs/<run_id>/epoch.json`); every box runs a **sidecar** that obeys them by
  running STATIC torchrun per epoch. **Survivors keep training at world N−1
  while a dead node is replaced** — ~15–30s per membership change. This
  replaced a torchrun-**elastic** design (`--nnodes=N-1:N`, c10d dynamic
  rendezvous) that passed all local tests on torch 2.4 but hung >180s on the
  DLAMI's torch ≥2.8 — the dynamic rendezvous is a version-dependent black box,
  so we moved to central orchestration (one writer, N readers, our code + our
  logs). Kept from the elastic work: constant global batch via gradient
  accumulation (`GLOBAL_BATCH_SIZE`; K recomputed per world size); two-tier
  checkpoints (step-aligned node-local disk for instant survivor restores +
  rank-0 async S3 for replacements; group-MIN agreement picks the resume step);
  budget-in-checkpoint (`TRAIN_BUDGET_SECONDS` − `trained_seconds`, so downtime
  is never billed); the W&B world-size staircase + goodput. The supervisor's
  decision core is a PURE reducer (`decide(Observation, Policy) -> [Action]`,
  table-tested); the whole protocol runs on localhost in `test_epoch_e2e.py`
  (real sidecars + static torchrun). No node hosts a rendezvous store, so any
  node is killable. Details: docs/multinode-design.md. Still open in 1c: promote
  the supervisor onto a dedicated on-demand `t3.micro` (durable control plane),
  the Go control plane (**observe / compare / act**). **Headline experiment:**
  on-demand baseline vs. spot to the same target loss for under **C** within
  **(1+ε)·T**; report goodput, recovery time, lost work, idle-wait.
- **1d — real Llama-arch model:** same 1c system, bigger model. Shows the
  controller is model-agnostic and that η improves with model size, giving a
  cleaner savings story. Validation-at-scale, not new infra — watch memory
  (may need gradient checkpointing / sharding).

---

## Repository layout

Kept **deliberately minimal** — only what Phase 1a needs. Later phases add their
own folders when they start (see "added later").

```
CLAUDE.md              # this file — working guidance + guiding principles
ROADMAP.md             # OptiTrain roadmap — the plan of record for new work
README.md              # public overview
pyproject.toml         # src-layout; scripts: spot-train, spot-orchestrate
.env.example           # AWS creds/bucket template → copy to git-ignored .env
src/spot_train/        # remote trainer — OUR fault-tolerance loop
  train.py             # entrypoint: one resume path + wall-clock budget + eval
  checkpoint.py        # full-state save/restore + verify() + smoke test
  s3_store.py          # local+S3 one interface; atomic rename + SHA-256
  rng.py               # capture/restore all RNG states
  data.py              # nanoGPT memmap batches; pulls dataset from S3
  config.py            # TrainConfig (+ from_env for the box)
  interruption.py      # IMDS poller + SIGTERM handler — PARKED until real
                       #   preemption handling (MVP uses controlled kills)
src/orchestrator/      # local control plane (boto3) — you run this
  aws.py               # the ONLY module that calls AWS; --dry-run + logging
  setup.py             # idempotent: bucket + IAM instance profile + SG
  dataset.py           # prepare-once → upload to S3
  bake.py              # bake-ami: pre-provisioned AMI (repo+deps) → faster boots
  bootstrap.py         # EC2 user-data script builder (multinode → sidecar)
  experiments.py       # run_baseline / run_spot / _run_supervised (multinode)
  supervisor.py        # epoch supervisor: pure decide() reducer + Effects loop
  sidecar.py           # per-box: obey epoch.json, run static torchrun per epoch
  config.py            # OrchestratorConfig (env-overridable)
  __main__.py          # CLI: setup | stage-data | baseline | spot [--dry-run]
third_party/nanoGPT/   # Karpathy's nanoGPT — git submodule, read-only.
                       #   we import GPT/GPTConfig from model.py; we do NOT
                       #   use their train.py — our loop owns fault tolerance.
tests/                 # checkpoint/resume tests

# S3 layout (created at runtime): data/<dataset>/{train,val}.bin,meta.pkl
#                                  runs/<run_id>/checkpoints/  +  metrics.json
# current phase (ROADMAP Part 1) adds: src/inference/ (worker + router),
#   src/orchestrator/fleet.py, loadgen/ (Go load generator, own go.mod)
# added later (do not create until the phase begins):
#   src/spot_train/rl/ + rl_train.py — GRPO finetuning        (ROADMAP Part 2)
#   router-go/, deploy/ — Go router + K8s manifests           (ROADMAP Part 3)
#   supervisor/  — Go control plane (observe/compare/act)     (ROADMAP Parts 3/7)
```

The nanoGPT submodule is pinned; contributors run
`git submodule update --init` after cloning.

## Conventions

- Python package lives under `src/spot_train` (src layout). Install editable:
  `pip install -e .`.
- Prefer running the CPU determinism test before any cloud work:
  `pytest tests/test_kill_resume.py`.
- Go code (`loadgen/` now; `router-go/`, `supervisor/` later) lives in its own
  modules with their own `go.mod`; ruff/pyproject govern Python only.

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
