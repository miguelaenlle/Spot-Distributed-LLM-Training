# Multi-node DDP — design (epoch supervisor)

**Goal:** train one NanoGPT across N nodes × all-GPUs-per-node (N ≤ 8),
surviving the loss of a node **without stopping**: the survivors re-form at
world size N−1 within ~15–30s of a kill and keep training while a replacement
boots; the replacement then joins and the group returns to N.

Membership is owned by a **single orchestrator** publishing monotonic **epoch
documents**; every box runs a **sidecar** that obeys them by running STATIC
torchrun per epoch. This replaced an earlier torchrun-elastic design — see the
post-mortem at the end for why.

---

## Why not torchrun elastic (post-mortem)

The previous design used torchrun's dynamic c10d rendezvous
(`--nnodes=N-1:N`, store on node 0): a killed node crashed survivors'
collectives (NCCL timeout), their elastic agents re-rendezvoused at N−1, and
training continued. It **passed every local test on torch 2.4** — the exact
"victim held rendezvous rank 0, survivor held the store" layout recovered in
~5s — but on AWS (**DLAMI torch ≥2.8, python 3.13**) the survivors never
resumed within 180s (`multinode-shrink` FAIL). The dynamic rendezvous is a
version-dependent black box: keep-alive TTLs, rank-by-sorted-IP assignment, and
last-call behavior all differ across torch versions we can't see into. Debugging
a peer-to-peer state machine on a vanishing spot box is exactly the misery the
project's guiding principles warn against. So we replaced choreography with
orchestration: **one writer, N readers, monotonic epochs, every recovery step
our code emitting our logs.**

## What already works (unchanged)

The trainer (`src/spot_train/`) is untouched — gradient accumulation (constant
global batch across world sizes), two-tier checkpoints with group-MIN resume,
budget-in-checkpoint (`trained_seconds`), and the `ws N` step-line telemetry all
carry over. Also kept: the self-referencing SG rule (all TCP intra-group),
NCCL_TIMEOUT=20 as an in-band backstop, the vCPU quota gates, the cost ledger,
and the W&B world-size staircase / degraded-phase profile.

## The protocol

**Epoch doc** — `runs/<run_id>/epoch.json`, written ONLY by the orchestrator,
monotonic `epoch`:

```json
{"epoch": 3,
 "members": [{"node": 0, "ip": "172.31.a.b", "rank": 0}, {"node": 2, "ip": "172.31.c.d", "rank": 1}],
 "node_count": 2,
 "master_addr": "172.31.a.b",
 "master_port": 29403}
```

- Master/rank-0 = the lowest live node index (deterministic — no sorted-by-IP
  surprises). No box hosts a rendezvous store, so **any node is killable**,
  including epoch rank 0.
- `master_port = rdzv_port + epoch`, so a relaunched master never fights
  TIME_WAIT on its own previous socket.

**Registration** — each box writes `runs/<run_id>/nodes/node<i>.json`
`{"ip", "instance_id"}` once at boot (IMDSv2, `"unknown"` fallback). Registration
is both the ready-marker and the join request: **admission = the orchestrator
including the node in a published epoch** (this is the "Go" signal).

**Sidecar** (`src/orchestrator/sidecar.py`, runs on every box AND in the
localhost E2E — stdlib + `spot_train.s3_store` only) polls `epoch.json` every
3s:
- an epoch that names me, new or changed → run STATIC torchrun for it
  (`--nnodes=K --node_rank=R --master_addr=A --master_port=P --max-restarts=0
  -m spot_train.train`), killing the previous torchrun tree first (children by
  `pgrep -P`, then the group, then `pkill -f spot_train.train` — torchrun
  detaches workers into their own sessions, learned the hard way);
- torchrun crashed on its own (a peer's death via NCCL_TIMEOUT — the in-band
  backstop) → drop the corpse, relaunch for the still-current epoch;
- not a member → stop any stale torchrun and idle-poll (a replacement awaiting
  admission, or a node the group shrank away from);
- `metrics.json` present → exit 0 (the group-wide done signal).

**Supervisor** (`src/orchestrator/supervisor.py`, runs inside the local
orchestrator process) is observe → decide → act, one tick per
`log_stream_seconds`:
- **observe** — instance states, registrations, log-key `LastModified` (the
  boxes already re-upload logs every 3s → free heartbeat), `max_checkpoint_step`,
  metrics;
- **decide** — a PURE reducer `decide(Observation, Policy) -> [Action]`. The
  whole membership policy, table-tested without AWS. The core is deliberately
  trivial: the membership that *should* be published is just the currently
  observed-healthy set, and an epoch is (re)published whenever that differs from
  what's live. **Membership is observation-driven only.** A scheduled kill (the
  experiments' stand-in for a spot reclaim) emits *only* `TerminateNode` — it
  does **not** itself shrink the group. The shrink happens a tick or two later
  when the terminated box is OBSERVED gone (AWS state flips into `_DEAD_STATES`,
  or its heartbeat goes stale), the identical path a real, un-orchestrated
  reclaim takes. The orchestrator never shortcuts membership with "I know I
  killed it"; the 20s NCCL timeout is the in-band backstop if observation lags;
- **act** — `Effects` executes via existing `aws.*` and emits the profile marks
  the W&B viz consumes (`kill`, `shrink_resume` when checkpoints advance past
  the kill baseline, `relaunch`, `full_world` when a `ws==N` sample returns).

## Failure matrix

| Failure | Handling |
|---|---|
| Worker node dies (real reclaim OR scheduled kill) | Discovered by observation — AWS state flip / stale heartbeat — never by orchestrator foreknowledge. Supervisor then publishes the shrink epoch; survivors' sidecars relaunch at N−1. NCCL_TIMEOUT crash is the in-band backstop if observation lags. |
| Epoch rank-0 dies | Same — the next epoch just names a new lowest-index master; fresh port avoids TIME_WAIT. No node is special (the elastic design's biggest hole, closed). |
| Replacement never registers | It's never admitted (stays out of every epoch); the whole-group-restart floor recovers the run. |
| Two nodes die | Shrink epoch with N−2 members (static K = member count; no min-nodes concept). |
| Everyone lost / no checkpoint progress within RECOVERY_TIMEOUT | Whole-group restart floor: delete epoch.json, terminate all, relaunch, publish a fresh epoch. |
| Orchestrator (laptop) dies mid-run | Boxes finish the current epoch's training to budget, then idle-poll (cheap); no training-budget loss (budget-in-checkpoint). Re-running the experiment command resumes supervision. (t3.micro promotion removes this exposure — see below.) |

## Verification

- `pytest tests/`:
  - `test_supervisor.py` — table-driven reducer (startup, loss→shrink, join→grow,
    scheduled kill, stall/all-gone→restart, stale heartbeat) + epoch-doc schema.
  - `test_sidecar.py` — the state machine with local dirs + a fake launcher
    (enter epoch, kill on bump, relaunch on crash, idle when excluded, exit on
    metrics).
  - `test_epoch_e2e.py` — **the full protocol on localhost**: two real
    `orchestrator.sidecar` processes run real static torchrun (gloo) on a dummy
    worker; a node is hard-killed and a shrink epoch published → the survivor
    resumes at world 1; a replacement registers + is admitted → world 2 again.
    This exercises the exact code that runs on AWS — the coverage gap that let
    the elastic design pass locally yet fail on the DLAMI. Static torchrun takes
    `--master_addr` verbatim, so it needs none of the getfqdn shim the dynamic
    test required — itself a small vindication of the switch.
- AWS: `multinode-shrink` (NODES=2 on-demand) is the acceptance gate — this
  exact experiment FAILED under elastic; PASS (kill→new-checkpoint ≤ 60s, expect
  15–30s) is the bar. Then NODES=4, then `multinode-preempt`.

## Scaling notes

- **N ≤ 8 nodes:** `NODES=N`. Static per-epoch torchrun needs no min-nodes knob.
- **Multi-GPU nodes:** `NPROC_PER_NODE` (forwarded to the sidecar's torchrun);
  pick `GLOBAL_BATCH_SIZE` divisible at both N·g and (N−1)·g ranks.
- **Deferred to 1c+:** promote the supervisor onto a dedicated on-demand
  `t3.micro` (durable control plane — survives laptop sleep; the sidecars
  already poll S3, so only the *writer* moves); the Go observe/compare/act
  supervisor; spot-pool bidding across AZs; at 1d scale,
  `torch.distributed.checkpoint` for resharded FSDP loads and peer-memory
  checkpoint tiers (Gemini, SOSP '23).
