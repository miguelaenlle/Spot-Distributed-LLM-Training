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

### 1. Rendezvous: c10d on node 0, endpoint discovered via S3

torchrun replaces `--standalone` with:

```
torchrun --nnodes=$NODE_COUNT --nproc_per_node=gpu \
  --rdzv_backend=c10d --rdzv_endpoint=$RDZV_ADDR:29400 --rdzv_id=$RUN_ID \
  --max-restarts=3 -m spot_train.train
```

The c10d TCPStore is hosted by the node whose address matches the endpoint —
**node 0**. No new infra (the 1c plan's dedicated t3.micro store is deferred;
it only matters once node 0 itself must be replaceable mid-run).

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

### 3. Failure model: one node dies → elastic restart → same resume path

`--max-restarts=3` puts the elastic agent in charge on every node:

1. Orchestrator SIGTERM-kills / terminates one **non-master** node (the same
   `_preempt_instance` machinery as today).
2. Survivors' collectives abort quickly — we set
   `init_process_group(timeout=60s)` (default is 10 min, far too slow) and rely
   on NCCL async error handling (on by default) to crash the worker rather than
   hang it.
3. Each surviving elastic agent catches its worker's death, tears down, and
   re-enters rendezvous (up to `max_restarts` times), waiting for `nnodes` to
   be satisfied again.
4. The orchestrator launches a **replacement node** (same user-data, same
   `NODE_INDEX`), it joins rendezvous, and every worker restarts `train()` —
   which is just the existing one-resume-path: load latest S3 checkpoint,
   continue. Loss continues from the checkpoint; nothing new to write in the
   trainer.

Killing **node 0** kills the TCPStore and takes the whole group down; the
orchestrator handles that as a full-group relaunch (today's ddp-preempt
semantics, N boxes instead of 1). The dedicated rendezvous box that makes node
0 non-special is exactly the 1c t3.micro — out of scope here.

Quota note: terminate the victim **before** launching its replacement — at an
8-vCPU quota, 3 concurrent g4dn.xlarge (12 vCPUs) would be rejected.

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

- A: node 0 log shows `[ddp] world_size=2`; `metrics.json` lands;
  loss stream is sane.
- B: after the kill, the survivor's agent re-rendezvouses; the replacement
  joins; log shows `[resume] restored from step N` and `world_size=2` again;
  loss continues from the checkpoint (not reset); `metrics.json` reports
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
