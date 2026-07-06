# Multi-node DDP — design (elastic: survivors keep training)

**Goal:** train one NanoGPT across N nodes × all-GPUs-per-node (N ≤ 8),
surviving the loss of a node **without stopping**: the survivors re-form at
world size N−1 within ~60s of a kill and keep training while the replacement
boots; the replacement then joins and the group returns to N. This is the
"elastic restart from cheap checkpoints" regime — right for a small fleet with
infrequent failures, where an occasional ~minute restart is far cheaper than
per-step redundancy (Bamboo/Oobleck-style schemes only pay off for pipeline
parallelism).

This supersedes the original **pause-and-replace** design (static rendezvous +
S3 generation protocol), which kept survivors *idle* until the replacement
arrived. Its history is in git; what survives of it is noted below.

---

## What already works (unchanged)

The trainer (`src/spot_train/`) is topology-agnostic:

- `distributed.init()` reads only `RANK`/`LOCAL_RANK`/`WORLD_SIZE` — torchrun
  sets these the same way for multi-node as for `--standalone`.
- Checkpointing is rank-0-only to S3; resume loads on **every** rank; S3 is
  the only shared state: dataset, checkpoints, metrics, logs.
- `all_reduce_stop` (MAX) coordinates budget/preempt stops group-wide.
- Self-referencing security-group rule: all TCP between group members
  (rendezvous store, the worker-group TCPStore on a **random free port**, and
  NCCL/gloo data-plane sockets). Nothing public.
- Quota gates: `wait_vcpu_headroom` before every launch; after a kill only the
  victim's slot is waited on (`wait_quota_released`).

## Design decisions

### 1. Rendezvous: ELASTIC c10d, store on node 0, endpoint discovered via S3

One long-lived torchrun per box:

```
torchrun --nnodes=$MIN:$NODES --nproc_per_node=gpu \
  --rdzv_backend=c10d --rdzv_endpoint=$RDZV_ADDR:$PORT \
  --rdzv_id=$RUN_ID --local_addr=$NODE_IP \
  --rdzv_conf=last_call_timeout=15 --max-restarts=100 \
  -m spot_train.train
```

- `MIN` defaults to N−1 (`NODES_MIN` to override): the group trains with one
  node down, and a rendezvous round closes `last_call_timeout` (15s) after MIN
  nodes are present.
- **Node 0 hosts the c10d store** (its `--local_addr` matches the endpoint) and
  publishes `rdzv.json {addr, port}` to S3 **once** at boot; workers — and
  replacements joining a live group — poll that key and dial. No generations,
  no ready markers.
- Elastic assigns ranks arbitrarily, so global rank 0 (checkpoints, metrics,
  loss lines) may live on any node — the orchestrator streams **all** node
  logs (≤ 8, cheap; per-step dedup keeps it idempotent).
- A thin outer retry loop (bounded, 20 attempts) covers what elastic can't
  absorb: restart budget exhausted or node 0's store lost. Node 0 republishes
  on a bumped port per attempt (the old TIME_WAIT defense, now rare-path only —
  the common path never rebinds because node 0's *agent* survives worker
  restarts).

> **Why elastic is safe now (it wasn't in the first attempt):** the original
> c10d try died because the *worker-group store* landed on the killed node.
> Pinning the rendezvous endpoint to node 0 and killing only workers keeps the
> store alive; node 0's own death is explicitly the fallback path (below).
> torch ≥2.4 puts the worker store on a random free port of the rank-0 agent's
> host — covered by the all-TCP intra-group SG rule.

### 2. The failure timeline (one kill, no orchestrator on the critical path)

1. Victim hard-terminated (no SIGTERM — Spot warns nobody).
2. Survivors' collectives abort after `NCCL_TIMEOUT` (20s;
   `TORCH_NCCL_DUMP_ON_TIMEOUT=0` keeps the crash fast).
3. Each surviving box's **elastic agent** catches its workers' crash and
   re-enters rendezvous; the dead node's agent stops heartbeating and expires.
4. Rendezvous closes with N−1 nodes after `last_call_timeout` → workers
   restart with new RANK/WORLD_SIZE.
5. Every rank resumes via the one resume path — from the **node-local
   checkpoint tier** (instant) since membership only shrank — and training
   continues at N−1. Grad accumulation rescales so the global batch is
   unchanged (below).
6. Meanwhile the orchestrator launches ONE replacement; its boot script reads
   the same `rdzv.json`, its agent joins the live rendezvous, and the next
   round restarts everyone at N — this time resuming from S3-latest (the
   group MIN includes the fresh node).

Downtime per membership change ≈ NCCL timeout + last-call + restore ≈ 40–60s,
twice per kill (shrink + grow), instead of the full replacement boot (~2 min+).

### 3. Constant global batch via gradient accumulation

`GLOBAL_BATCH_SIZE` (sequences per **optimizer step**) is fixed in config;
each (re)start computes `K = ceil(global / (world_size × batch_size))`
micro-batches per rank, with DDP `no_sync()` on all but the K-th backward.
A world-size change then alters wall-clock per step, **not** the gradient
statistics — the LR schedule stays valid and loss-vs-step curves overlay
across N and N−1 stretches. Pick a target that divides both: e.g. 4 nodes ×
1 GPU × batch 12 → `GLOBAL_BATCH_SIZE=144` (K=3 at world 4, K=4 at world 3).
Non-dividing worlds round K up and log the actual effective batch.

### 4. Cheap two-tier checkpoints

- **Node-local tier** (`LOCAL_CHECKPOINT_DIR`): every node's LOCAL_RANK-0
  snapshots to its own disk — DDP state is fully replicated, so this needs no
  network. Trigger is **step-aligned**: rank 0 decides (time-based, async
  submit accepted) and broadcasts the decision (`distributed.broadcast_flag`),
  so all local snapshots are at the same step. Two newest kept.
- **Durable tier** (S3): rank 0's async two-phase upload, unchanged.
- **Group-agreed resume** (`checkpoint.load_group_latest`): each rank offers
  the newest step it can reach; an `all_reduce MIN` picks the step everyone
  restores. Shrink → common local step, ~zero lost work. Replacement present →
  S3-latest (survivors lose ≤ one interval + one in-flight upload, the
  existing durability bound). A rank that can't fetch the agreed step crashes
  loudly → the agent restarts it → the group re-agrees.
- Checkpoint v2 carries `trained_seconds` (see budget, next).

### 5. Budget rides in the checkpoint (no budget.json)

`TRAIN_BUDGET_SECONDS` (the run total) is baked into user-data once. The
checkpoint accumulates in-loop `trained_seconds`; every (re)start computes
`max_seconds = max(1, budget − trained)`. Boot, NCCL stalls, and crash
teardown are never billed — by construction, with zero orchestrator writes
mid-run. The ≥1s clamp keeps rank 0 always able to eval + write
`metrics.json` (the group-wide done signal, unchanged).

### 6. Orchestrator: two-phase watchdog per kill

- **(a) shrink_resume** — a NEW checkpoint past the at-kill baseline within
  `DEGRADED_RECOVERY_TIMEOUT` (180s) of the kill. Survivors only; the
  replacement is *not* required. (The baseline is sampled after the
  stray-async-upload window — one upload can land ~NCCL_TIMEOUT past a kill
  that spared rank 0.)
- **(b) full_world** — a parsed step line reports the pre-kill world size at a
  step past the pre-kill floor, within `RECOVERY_TIMEOUT` (600s).
- Either timeout → **whole-group restart fallback**: delete the stale
  `rdzv.json` (fresh boxes must not dial a dead store), terminate all, wait
  quota, relaunch. Worst case equals the old pause-and-replace behavior.
- Killing node 0 is allowed but forfeits elastic recovery (it hosts the
  store): that experiment exercises the fallback path by design. The 1c
  upgrade — an off-node rendezvous store (t3.micro/etcd) — would remove this
  asymmetry.

### 7. Observability: the world-size staircase

The trainer's step line carries the live world size
(`step 120: loss 1.83, 80ms/step, 15300 tok/s, ws 4`); `profile.py` parses it
into per-sample `world_size`, mirrors it to W&B on the same step/time axes as
the loss curve (the N → N−1 → N staircase), and the timeline gains a
**degraded** phase (kill → shrink_resume = downtime; shrink_resume →
full_world = degraded). Summary adds `goodput` (= `trained_seconds_total` /
wall-clock) and `degraded_s`. Metrics adds `restart_count`,
`grad_accum_steps`, `effective_global_batch`, `trained_seconds_total`.

## Failure modes

| Failure | Behavior |
|---|---|
| Worker node dies | Survivors continue at N−1 in ~40–60s; replacement rejoins → N. |
| Node 0 dies | Store gone; survivors' retries fail → orchestrator whole-group restart (documented scope). |
| Drops below MIN (2+ dead) | Rendezvous can't close; agents idle in join (bounded) until replacements arrive or the watchdog restarts the group. |
| Replacement joins mid-step | Its agent signals the rendezvous; running agents restart workers at the next round — one clean membership change. |
| Async S3 upload in flight at a restart | Atomic temp-key→rename: no corruption, at most one lost upload; group-MIN agreement absorbs S3 lag. |
| Restart budget / retries exhausted | Box exits nonzero, stays up for debugging; watchdog restarts the group. |

## Verification

- `pytest tests/` — includes `test_elastic.py` (K math, gradient equivalence,
  v1/v2 checkpoints, local tier, group agreement) and **`test_elastic_e2e.py`**:
  two real torchrun agents on localhost, one killed (agent + worker — torchrun
  detaches workers into their own sessions), the survivor re-rendezvouses and
  continues at world 1, then exits 0. This validates the entire elastic
  mechanism without AWS.
- On AWS: `multinode` smoke (no kill), then `multinode-preempt` — see
  CLAUDE.md for the run recipe. Success: checkpoints keep advancing at ws N−1
  before the replacement is up; W&B shows the staircase; loss continues from
  the checkpoint; the fallback still fires when node 0 is the victim.

## Scaling notes

- **N ≤ 8 nodes:** `NODES=N` (+ quota). `NODES_MIN` defaults to N−1; set
  `NODES_MIN=$NODES` to pin the old all-or-nothing behavior.
- **Multi-GPU nodes:** `--nproc_per_node=gpu` adapts; pick `GLOBAL_BATCH_SIZE`
  divisible at both N·g and (N−1)·g ranks.
- **Deferred to 1c+:** off-node rendezvous store (t3.micro — makes node 0
  killable), the Go observe/compare/act supervisor, spot-pool bidding across
  AZs; at 1d scale, `torch.distributed.checkpoint` for resharded loads (FSDP)
  and peer-memory checkpoint tiers (Gemini, SOSP '23).
