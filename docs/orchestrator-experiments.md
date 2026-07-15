# Durable orchestrator — demonstration experiments

Two runbooks for the ASG-backed t3.micro control plane:

- **E1** — the orchestrator works: a 4- vs 8-node sweep gives comparable, sane
  scaling results, driven entirely off your laptop.
- **E2** — the orchestrator survives its own death: kill the t3.micro mid-run and
  watch a fresh one take over and resume from checkpoint.

> ⚠️ **Billable.** Both launch a `t3.micro` control plane + `g4dn.xlarge` GPU
> boxes. 8 nodes needs a G-class spot quota of **≥32 vCPU** (`VCPU_QUOTA`). Run
> `--dry-run` first; `spot-orchestrate remote-down <job_id>` is the hard stop.

One-time: `spot-orchestrate setup` (creates the bucket + worker **and**
`spot-orch-role`/profile), then `spot-orchestrate stage-data`. Confirm `.env` has
`SPOT_TRAIN_BUCKET`, `AWS_REGION`, `VCPU_QUOTA≥32`, optional `WANDB_API_KEY`.

---

## E1 — the orchestrator works (4- vs 8-node sweep, comparable results)

Goal: prove the durable orchestrator drives a real multi-node sweep and that the
numbers are internally consistent — throughput scales ~linearly at constant global
batch, and a preempting run recovers to the same steady-state ms/step (only wall
clock grows). We run BOTH a clean and a worst-case-preempt sweep at 4 and 8 nodes,
then join them.

```bash
# 1) Clean sweep at 4 and 8 nodes, on the orchestrator (throughput mode).
NODE_COUNTS=4,8 spot-orchestrate remote-up --experiment scaling-clean
#    -> prints CLEAN_JOB_ID (e.g. scaling-clean-1784…)

# 2) Watch it. The sweep runs each point sequentially on the t3.micro.
spot-orchestrate remote-status <CLEAN_JOB_ID>     # which point is running + done-sentinel
spot-orchestrate logs <run_id_from_status>        # live per-node dashboard (unchanged)

# 3) When it's done, worst-case-preempt sweep at the same sizes.
NODE_COUNTS=4,8 spot-orchestrate remote-up --experiment scaling-preempt
#    -> prints PREEMPT_JOB_ID.  At each size it loses floor(N/2) nodes at once
#       (4n -> kill 2, 8n -> kill 4), then recovers to full N.

# 4) Join them into the side-by-side overhead table (runs on the laptop).
spot-orchestrate scaling-compare <CLEAN_JOB_ID> <PREEMPT_JOB_ID>
```

**What "comparable results" means — success criteria** (`reports/scaling-compare-*/summary.txt`):

- **Clean scaling is sane:** 8-node ms/step is meaningfully below 4-node (ideal
  2×; a comms haircut is expected). `scaling-clean`'s own `summary.txt` prints the
  speedup + efficiency vs the baseline.
- **Throughput recovers after preemption:** per node count, the preempt vs clean
  **ms/step delta is ≈ 0** (the trimmed mean drops the recovery spike). This is the
  headline "the system works" result — losing half the world doesn't degrade
  steady-state throughput once it recovers.
- **Overhead is bounded and explained:** the `runtime overhead %` is positive
  (recovery costs wall-clock) but bounded, and the row shows `recovery_s` /
  `degraded_s` / `min_world` = `ceil(N/2)` accounting for it.
- The t3.micro self-scales its ASG to 0 and terminates after each sweep; run
  `remote-down <job_id>` to delete the leftover ASG/template shell.

Tip: for a loss-based comparison instead of throughput, set `TARGET_LOSS=<val>`
(run `spot-orchestrate calibrate` first) — both sweeps then report time-to-target
and the overhead table switches to that metric automatically.

---

## E2 — what happens when the orchestrator gets killed

Goal: a mid-run death of the control plane is invisible to the training result.
Uses a single-run job so the resume is unambiguous (~10–15 min, ~$1).

```bash
# 1) Start a 2-node run with a short budget + dense checkpoints, on the orchestrator.
NODES=2 TRAIN_TOTAL_SECONDS=300 CHECKPOINT_INTERVAL_SECONDS=30 \
  spot-orchestrate remote-up --experiment multinode
#    -> prints JOB_ID (== run_id for single runs, so `logs JOB_ID` works now)

# 2) Wait until training is underway (gen=1, epoch>=1, a checkpoint or two landed).
spot-orchestrate remote-status <JOB_ID>     # look for generation: 1 and a running ASG
spot-orchestrate logs <JOB_ID>              # watch step/loss climb

# 3) INJECT THE FAULT: terminate the live t3.micro (leaves the ASG at desired=1).
spot-orchestrate remote-kill <JOB_ID>

# 4) Observe the self-heal (ASG relaunches a fresh box in ~1–3 min).
watch -n15 "spot-orchestrate remote-status <JOB_ID>"
```

**What you should see — success criteria:**

- **ASG self-healed:** `remote-status` shows `generation: 2` (a fresh control
  plane took over). The `orchestrators/<JOB_ID>/boot.log` shows a `cold-start
  gen=2` banner.
- **Cold recovery:** the fresh box's `orchestrator.log` shows
  `=== orchestrator restarted (previous control plane died); cold recovery ===`
  and terminates the 2 stranded gen-1 GPU boxes.
- **No leaked / double-billed boxes:** exactly one generation of GPU boxes runs
  after recovery (check the AWS console or `aws ec2 describe-instances` on
  `tag:project=spot-train`).
- **Training resumed, not restarted:** the relaunched trainer reports
  `resumed=true` and **loss keeps dropping from the last checkpoint step** (never
  resets to the initial loss). `metrics.json` eventually appears and
  `trained_seconds_total ≈ 300` (recovery downtime is not billed against the
  budget — it rides in the checkpoint).
- **Self-teardown:** on completion the box self-scales its ASG to 0 and
  terminates. Run `spot-orchestrate remote-down <JOB_ID>` to delete the shell.

### How the kill shows up in the timeline

The kill is **first-class** in the run timeline (`spot-orchestrate logs <JOB_ID>`,
or the exported Gantt/`events.txt`):

- Each GPU node gets a **`killed` bar with cause `orchestrator-restart`** at the
  recovery moment — emitted into `orchestrator.log` by cold recovery (mapping each
  orphan instance to its node via `nodes/node<i>.json`). In the Gantt this is the
  ✗ "killed/gone" marker; in `events.txt` it reads `nodeN: KILLED (…,
  orchestrator-restart)`.
- The **`orchestrator.log` narrative is preserved across generations** — the fresh
  supervisor seeds its log buffer from the existing S3 object and appends, so you
  see the pre-death epoch/decision lines, the restart banner, then the gen-2
  narrative in one continuous log (the "orchestrator" tab in `logs`).
- After the ✗, each node's row shows a fresh **provisioning → training** segment
  (the replacement booting and resuming from checkpoint), so the shape reads:
  *training → ✗ killed (orchestrator-restart) → gap → provisioning → training*.
- The `generation` counter (`remote-status`) and `orchestrators/<JOB_ID>/boot.log`
  are the out-of-band confirmation that it was the **control plane** that died,
  not a GPU node.

> Contrast: a GPU **node** kill (the `scaling-preempt` / `multinode-preempt` path)
> shows `killed cause=scheduled-kill` (or `down cause=reclaimed` for a real spot
> reclaim). The `orchestrator-restart` cause is what distinguishes a control-plane
> death from a worker death in the same timeline.

### Optional: kill during a sweep (E1 variant)

`remote-kill` during a `scaling-preempt` sweep restarts the **whole sweep** from
the first node count (the deliberate, simple recovery for sweeps — they don't test
the orchestrator's own fault-tolerance). `remote-status` shows `generation: 2` and
cold recovery terminates the interrupted point's boxes. Use E2's single-run job
when you want the seamless checkpoint-resume behavior.
