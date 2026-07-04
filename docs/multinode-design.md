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

### 3. Failure model: one node dies → whole-group restart → same resume path

The standard production model (OPT/BLOOM-style): a lost node kills the job;
recovery is a fresh gang restarting from the last checkpoint.

1. Orchestrator HARD-terminates one **non-master** node — no SIGTERM. (A
   graceful SIGTERM triggers the trainer's coordinated stop, which cleanly
   shuts down the *whole group* — a different, gentler experiment. A real Spot
   reclaim warns nobody.)
2. Survivors' collectives abort quickly — `init_process_group(timeout=60s)`
   via `NCCL_TIMEOUT` (torch default is 10 min, uselessly slow) and NCCL async
   error handling crash the worker rather than hang it.
3. `--max-restarts=0`: the agents exit on the worker crash. The orchestrator
   tears down the remaining boxes, waits for the vCPU quota to release, and
   launches a **fresh full segment** — every node reruns `train()`, which is
   the existing one-resume-path: load latest S3 checkpoint, continue. Loss
   continues; the trainer needed zero changes.
4. Lost work per kill ≤ the densified checkpoint interval (5s default here).

In-place elastic rejoin (survivors idle-wait, replacement slots in) is
deferred to 1c: it needs the rendezvous/worker store off-node (the t3.micro)
so no victim can take the coordination state down with it.

Quota note: `wait_quota_released` (instance left `running`) gates each
segment's launches — at an 8-vCPU quota, two segments' boxes can't coexist.

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
| `config.py` | `node_count` (`NODES`, default 1), `rdzv_port` (`RDZV_PORT`, 29400), `nccl_timeout_seconds` (`NCCL_TIMEOUT`, 60) |
| `aws.py` | self-referencing SG ingress rule in `ensure_security_group` |
| `bootstrap.py` | multi-node branch: export `NODE_INDEX`/`NODE_COUNT`; node 0 publishes `rdzv.json`, others poll; torchrun rendezvous flags replace `--standalone` when `NODE_COUNT > 1` |
| `experiments.py` | `run_multinode` (launch N, stream node 0, wait metrics, reap all) and `run_multinode_preempt` (train `PREEMPT_AFTER`, kill non-master node, terminate→relaunch replacement, verify group recovers and loss continues) |
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
- B: segment 1 trains, the kill lands, the group is torn down; segment 2's
  node 0 log shows `[resume] restored from step N` and `world_size=2`; loss
  continues from the checkpoint (not reset); `metrics.json` reports
  `resumed=true`.

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
