# Multi-node DDP — design

**Goal:** train one NanoGPT across N nodes × all-GPUs-per-node, surviving the
loss of a node. **First test: 2× g4dn.xlarge (1 GPU each, 4 vCPUs each = 8
vCPUs total — exactly the current on-demand G quota).** The design must scale
to N nodes / multi-GPU nodes with only env changes.

This is a deliberately Python-orchestrated stepping stone to Phase 1c; the Go
supervisor, dedicated rendezvous box, and async checkpointing stay out of scope.

---

## What already works (unchanged)

The trainer (`src/spot_train/`) is topology-agnostic today:

- `distributed.init()` reads only `RANK`/`LOCAL_RANK`/`WORLD_SIZE` — torchrun
  sets these the same way for multi-node as for `--standalone`.
- Data sharding is by **global** rank (`torch.manual_seed(seed + ddp.rank)`),
  checkpointing is rank-0-only, and resume loads from S3 on **every** rank —
  none of it knows or cares about node boundaries.
- `all_reduce_stop` (MAX) already coordinates budget/preempt stops group-wide.
- S3 remains the only shared state: dataset, checkpoints, metrics, logs.

## Design decisions

### 1. Rendezvous: STATIC, master on node 0, endpoint discovered via S3

torchrun replaces `--standalone` with the classic static multi-node form:

```
torchrun --nnodes=$NODE_COUNT --nproc_per_node=gpu --node_rank=$NODE_INDEX \
  --master_addr=$RDZV_ADDR --master_port=29400 --max-restarts=0 \
  -m spot_train.train
```

Static pins the topology: **node 0 is always agent rank 0, hosts the worker
process-group store, and holds global rank 0** — so node 0's log always carries
the losses and "kill a non-master node" is actually enforceable.

> **Why not c10d elastic (what we tried first):** the c10d backend assigns
> agent/worker ranks arbitrarily. In the first real 2-node run, rank 0 *and*
> the worker-group TCPStore landed on the node we killed; the survivor's
> restarted worker then spun on `No route to host` retries dialing the dead
> box's store. In-place elastic rejoin is only robust with the store off-node —
> that's the 1c t3.micro. No new infra needed for the MVP.

**Endpoint discovery** (chicken-and-egg: user-data is built before any node has
an IP): node 0's user-data publishes its private IP to
`s3://<bucket>/runs/<run_id>/rdzv.json`; every other node polls that key before
starting torchrun. S3 is already the transport for everything else, and the
instance role already has the bucket permissions. The orchestrator launches all
N nodes in one go — no launch ordering.

- `NODE_INDEX` (0..N-1) is baked into each node's user-data; it selects
  publish-vs-poll behavior only. torchrun assigns actual ranks at rendezvous.

### 2. Networking: self-referencing security-group rule

Nodes must reach each other: rendezvous :29400, the c10d TCPStore, and the
NCCL/gloo data-plane sockets (ephemeral ports). `ensure_security_group` grows
one idempotent rule: **allow all TCP ingress from the security group itself**.
Nothing new is exposed publicly; SSH ingress stays as is.

g4dn.xlarge has no EFA; NCCL runs over TCP sockets (up to 25 Gbps). Fine at
NanoGPT scale — and the eventual bandwidth ceiling is a 1c/1d measurement, not
a blocker.

### 3. Failure model: one node dies → survivors PAUSE → replace only the victim

A lost node still kills the *job* (synchronous DDP can't step without a full
group), but not the *boxes*: healthy instances are never rebooted. Recovery
replaces exactly the dead node; the group re-forms and resumes from S3.

1. Orchestrator HARD-terminates one **non-master** node — no SIGTERM. (A
   graceful SIGTERM triggers the trainer's coordinated stop, which cleanly
   shuts down the *whole group* — a different, gentler experiment. A real Spot
   reclaim warns nobody.)
2. Survivors' collectives abort quickly — `init_process_group(timeout=60s)`
   via `NCCL_TIMEOUT` (torch default is 10 min, uselessly slow) and NCCL async
   error handling crash the worker rather than hang it.
3. `--max-restarts=0`: the agents exit on the worker crash — but the boot
   script's **generation loop** keeps the box alive, pausing in a cheap S3
   poll. The orchestrator waits for the victim's vCPU quota slot, launches
   **one replacement** with the same `NODE_INDEX`, and the group re-forms at
   the next rendezvous generation (below). Every restarted `train()` is the
   existing one-resume-path: load latest S3 checkpoint, continue. Loss
   continues; the trainer needed zero changes.
4. Lost work per kill ≤ the densified checkpoint interval (5s default here).
5. **Watchdog fallback:** if no new checkpoint appears within
   `RECOVERY_TIMEOUT` (600s) of the replacement launch, the orchestrator
   reverts to the previously-proven whole-group restart (terminate all, wait
   quota, relaunch fresh) — worst case equals the old behavior.

**Generation protocol** (the livelock killer — no node ever restarts the group
on its own, and nobody busy-retries against a box that is still booting):

- `rdzv.json` carries `{addr, port, generation, node_count}`; the master port
  is `RDZV_PORT + generation`, so TIME_WAIT from node 0's previous torchrun
  can't collide when it rebinds on the same box.
- Per generation G (= currently published generation + 1 — survivors and
  fresh replacements compute this independently and agree): each non-master
  node writes a ready marker `ready/gen<G>-node<I>` to S3, then polls
  `rdzv.json` and dials only what node 0 actually publishes. Node 0 publishes
  gen G **only after all N−1 ready markers for G exist**, then immediately
  starts torchrun — its TCPStore comes up seconds before the workers dial,
  well inside the store's client connect window.
- If the *master* dies (real preemption, or a `PREEMPT_VICTIMS` schedule that
  includes node 0), its replacement reads the stale `rdzv.json`, waits for the
  survivors' gen+1 ready markers, and publishes its own new IP — same code
  path, no special case. Orchestrator-side, a node-0 kill additionally switches
  the streamed log to the replacement's attempt key (fresh file ⇒ profile
  segment bump + printed-offset reset).
- Known self-healing wrinkle: ready markers are never deleted, so after a
  whole-group-restart fallback, stale `gen<G+1>` markers from the failed
  attempt can let the fresh master publish before its workers are up; the
  join then times out and everyone converges at G+2 — worst case one wasted
  generation, no intervention needed.
- `metrics.json` appearing in S3 is the group-wide done signal. The budget is
  **orchestrator-authoritative**: `budget.json` (next to `rdzv.json`) holds
  the remaining seconds, recomputed after every kill from *observed* training
  time (first checkpoint → kill), so boot, the NCCL stall, and crash teardown
  are never billed. Boxes read it before each generation, clamp it to ≥ 1,
  and export it as `MAX_SECONDS` — the loop never exits for lack of budget,
  because rank 0 must always be able to re-form the group for the coordinated
  stop → eval → `metrics.json` ending (`all_reduce_stop` MAX keeps mismatched
  budgets coordinated — the first expiring rank stops the group). Multinode
  boxes also set `TORCH_NCCL_DUMP_ON_TIMEOUT=0`: torch's post-timeout debug
  dump added ~2 minutes to every peer-death crash.
- Every wait is bounded (~20 min); on exhaustion a box exits nonzero and is
  left up for debugging, and the orchestrator's watchdog handles recovery.

The off-node rendezvous store (t3.micro) is still the Phase 1c upgrade — it
would let survivors' *processes* idle inside torchrun instead of exiting —
but pause-and-replace already removes the wasted reboots of healthy boxes.

Quota note: every launch is gated by `wait_vcpu_headroom` — the orchestrator
polls `DescribeInstances` (every 15s, one call per poll) until running+pending
G/VT vCPU usage leaves room under `VCPU_QUOTA` (default 8), instead of firing
`RunInstances` into a quota wall. After a kill, only the victim's slot is
waited on (`wait_quota_released`); survivors keep theirs. The separate *spot
instance count* quota is not covered by this gate.

### 4. Static group for the MVP (`--nnodes=N`, not `min:max`)

Elastic `--nnodes=1:2` (survivors continue at reduced world size) changes
effective batch size mid-run and complicates the loss-continuity check. MVP
keeps the group static: training only proceeds at full strength; a lost node
means the group waits in rendezvous until the replacement arrives. Goodput vs.
elastic-shrink is a 1c experiment.

### 5. Logs and metrics

- Per-node log keys: `logs/<run_id>/seg<K>-node<I>.log` (extend
  `run_logs_key` with a node suffix). The orchestrator streams node 0's log
  (rank 0 prints losses) and pulls other nodes' logs on failure/timeout.
- `metrics.json`: unchanged — global rank 0 writes it; the orchestrator polls
  the same key.
- Every box keeps today's self-terminate-on-success backstop; the orchestrator
  reaps all tracked instance ids in its `finally`.

## New orchestrator surface

| Piece | Change |
|---|---|
| `config.py` | `node_count` (`NODES`, default 1), `rdzv_port` (`RDZV_PORT`, 29400), `nccl_timeout_seconds` (`NCCL_TIMEOUT`, 60), `recovery_timeout_seconds` (`RECOVERY_TIMEOUT`, 600), `vcpu_quota` (`VCPU_QUOTA`, 8), `instance_vcpus` (`INSTANCE_VCPUS`, builtin table) |
| `aws.py` | self-referencing SG ingress rule in `ensure_security_group`; `vcpus_in_use` + `wait_vcpu_headroom` (quota gate) |
| `bootstrap.py` | multi-node branch: generation loop (`_multinode_loop`) — ready markers + `rdzv.json` publish/poll + torchrun + pause-and-rejoin; replaces `--standalone` when `NODE_COUNT > 1` |
| `experiments.py` | `run_multinode` (launch N, stream node 0, wait metrics, reap all) and `run_multinode_preempt` (train `PREEMPT_AFTER`, kill non-master node, replace ONLY the victim while survivors pause; whole-group restart as watchdog fallback) |
| `__main__.py` | `multinode` / `multinode-preempt` commands |
| `distributed.py` | pass `timeout=timedelta(seconds=$NCCL_TIMEOUT)` to `init_process_group` (trainer reads it from env via `TrainConfig`) |

## The 2-node test

```bash
# Milestone A — clean multi-node run to budget
NODES=2 MARKET=on-demand BASELINE_SECONDS=120 EVAL_ITERS=20 \
  spot-orchestrate multinode

# Milestone B — kill one node mid-run, replacement rejoins, loss continues
NODES=2 MARKET=on-demand TRAIN_TOTAL_SECONDS=120 PREEMPT_AFTER=20 EVAL_ITERS=20 \
  spot-orchestrate multinode-preempt
```

**Success criteria**

- A: node 0 log shows `[ddp] world_size=2` **and the loss lines** (rank 0 is
  pinned to node 0 by static rendezvous); `metrics.json` lands.
- B: the group trains, the kill lands, and **node 0 is never terminated** —
  its (continuous) log shows the torchrun crash, the `[rdzv]` pause, the gen-2
  publish, then `[resume] restored from step N` with `world_size=2`; loss
  continues from the checkpoint (not reset); `metrics.json` reports
  `resumed=true`. Only 3 boxes are ever launched (2 + 1 replacement), and no
  launch errors on the instance quota (a `[aws] wait for vCPU headroom` line
  appears instead).

**Cost:** 2 × $0.526/hr on-demand; a ~12-min end-to-end run ≈ **$0.21**
(milestone B adds a third short-lived box: ≈ **$0.30**).

## Scaling beyond the test

- **N nodes:** `NODES=N` — discovery, SG rule, and experiment driver are all
  N-ary already. Only the vCPU quota gates it (4 vCPUs per g4dn.xlarge).
- **Multi-GPU nodes:** `--nproc_per_node=gpu` already adapts; `NODES=4` ×
  g4dn.12xlarge is the 1c 4×4 shape with zero code change.
- **Deferred to 1c:** rendezvous store on a dedicated t3.micro (makes node 0
  replaceable), the Go observe/compare/act supervisor, elastic min:max groups,
  async two-tier checkpointing, spot-pool bidding across AZs.
