# Scaling experiment — operator / handoff guide

A cold-start guide to running the **2- vs 4-node time-to-target-loss** experiment
on AWS spot. Written for an agent taking over mid-setup. Everything here is
implemented and on `main`; what remains is **operating it** (credentialed, some
billable) and interpreting the results.

## What we're trying to prove

Two hypotheses about whether adding nodes speeds up training on preemptible spot:

- **H1:** `time_to_target(4 nodes) < time_to_target(2 nodes)`, no preemptions.
- **H2:** same, **with** preemptions.

**"time to target"** = wall-clock from the **first training step** to the first
periodic eval where `val_loss ≤ TARGET_LOSS`. Boot is excluded (it's ~constant and
not the phenomenon under test); preemption downtime + re-computed steps ARE
included (that's the cost H2 probes).

**Why this and not "time to overfit":** the earlier design measured time-to-overfit
on char-Shakespeare, which overfits in ~15 s — boot dominated and there was no room
to inject a preemption. We moved to a **real corpus** (a 300 M-token OpenWebText
slice) + a **target-loss stop** so runs are compute-bound and 15–30 min long, with a
real pre-target window for preemptions to bite. Fuller scaling context (and the
eventual GPT-2 run + streaming data plane) is in `docs/gpt2-reproduction.md`.

**Controls (what makes the comparison fair):** identical model/data/seed, and a
**constant global batch** (grad-accum keeps it fixed regardless of node count), so
2- and 4-node follow the *same loss trajectory vs step* → the target is hit at ~the
same STEP and the comparison isolates **throughput**. The report prints
`steps_to_target` per run — if those don't match across node counts, the control
failed and it's flagged, not reported as a bogus speedup. Runs are **sequential**
(peak 16 vCPUs; also removes cross-run instance-perf noise).

## The pipeline (4 stages)

```
prepare (local, no GPU)  →  stage-data (upload to S3)  →  calibrate (1 box, sizes TARGET_LOSS)  →  scaling-experiment (4 spot runs)
```

### 0. Prerequisites (do these first)

- **AWS creds** resolvable by boto3 (SSO profile or keys), region us-east-1.
- **A real S3 bucket.** `.env` currently has the PLACEHOLDER
  `SPOT_TRAIN_BUCKET=your-unique-spot-train-bucket` — **change it to a real,
  globally-unique bucket you own.** This is why `stage-data`/`calibrate` currently
  fail ("dataset not staged at s3://your-unique-spot-train-bucket/…"): there's no
  real bucket behind the placeholder.
- **Infra** (bucket + IAM instance profile + security group): `spot-orchestrate setup`
  (idempotent). Run once.
- **Spot G-instance quota ≥ 16 vCPUs** (4× g4dn.xlarge). File the Service Quotas
  increase early if the account is fresh. Pass `VCPU_QUOTA=16` to match.
- The user runs every credentialed/billable command; the agent writes code + runs
  local tests/lint only.

### 1. Prepare the corpus (local, ~3 min, no GPU) — ALREADY DONE

```bash
pip install datasets tiktoken tqdm numpy
python data/openwebtext_300m/prepare.py
```
Streams OpenWebText, tokenizes with GPT-2 BPE, writes `data/openwebtext_300m/train.bin`
(~572 MB, 299.85 M tokens) + `val.bin` (~296 KB, 150 K tokens). **No `meta.pkl`** (BPE
vocab is fixed; the trainer falls back to vocab 50304). Cap via `OWT_TARGET_TOKENS`.

> **Gotcha (benign):** the script finishes its work and prints `[prepare] wrote …`
> but the process may **not return to the shell** — HuggingFace `datasets` streaming
> leaves background threads that block exit after we `break` at the token cap. The
> bins are already written; **Ctrl-C is safe**. (Verify: `ls -la data/openwebtext_300m/`.)

### 2. Stage to S3 (upload, ~minutes)

```bash
DATASET=openwebtext_300m spot-orchestrate stage-data
```
Uploads `train.bin` (+`val.bin`) to `s3://<bucket>/data/openwebtext_300m/`. Idempotent
(skips if already present). **No progress output** — a silent 572 MB upload looks
like a hang for a few minutes; it prints `[stage-data] uploaded […]` when done.
Verify: `aws s3 ls s3://<bucket>/data/openwebtext_300m/`.

### 3. Calibrate (1 on-demand box, ~5–8 min, ~$0.15) — sizes `TARGET_LOSS`

```bash
DATASET=openwebtext_300m spot-orchestrate calibrate
```
Runs GPT-2-small on one box for `CALIBRATE_SECONDS` (default 300), then prints and
writes `reports/calibrate-<ts>/calibration.txt`: measured 1-GPU throughput
(steps/s, tok/s), **projected 2-/4-node step counts** in the per-run cap (per-step
scales ~world-size at constant global batch, 0.85 comms haircut), the loss the
probe reached, and a **suggested `TARGET_LOSS`** (log-extrapolated to land mid-run).
You pick the final `TARGET_LOSS` from this.

> This is also the **de-risking** step: it reveals whether GPT-2-124M actually fits
> and is fast enough on a T4 (16 GB). If it OOMs or is too slow for a clean 15–30 min
> run, drop the config via env (`N_EMBD=512 N_LAYER=8`) or lower `BATCH_SIZE`.

### 4. Run the experiment (4 spot runs, ~15–30 min each, ~$3–6)

```bash
VCPU_QUOTA=16 TARGET_LOSS=<from step 3> DATASET=openwebtext_300m \
  spot-orchestrate scaling-experiment
```
Runs, **sequentially**, `2n-clean → 4n-clean → 2n-preempt → 4n-preempt`. Each stops
at `val_loss ≤ TARGET_LOSS` or a 30-min wall-clock cap. Preempt runs kill a worker
node twice (at `PREEMPT_OFFSETS` = t+600 s, t+1200 s after train start) — the
orchestrator's `TerminateInstances` standing in for a spot reclaim. **`TARGET_LOSS`
is required** — the command errors and points you back to `calibrate` if it's unset.

Watch a run live in another terminal: `spot-orchestrate logs <run_id>` (press `t`
for the Gantt, `v` for the events log).

## Outputs

`reports/scaling-experiment-<ts>/`:
- **`summary.txt`** — recipe, per-run table (`steps_to_target`, `hit_val`,
  **`time_to_target_s`**, cost, wandb link), and the **H1/H2 verdicts** with speedup
  ratios. **Rewritten after every run** (partial results survive a later failure)
  and a failed run is recorded as `FAILED` without sinking the suite.
- Per run: `<run_id>-timeline.png` (event-sourced Gantt — leader ★, kills ✗, world
  staircase, wasted steps), `<run_id>-events.txt`, `<run_id>-valcurve.png`
  (val/train loss with the target line + crossing marked).
- W&B: one run per config under group `scaling-experiment-<ts>`.

## Config knobs (env; defaults shown)

| Env | Default | Meaning |
|---|---|---|
| `SPOT_TRAIN_BUCKET` | (placeholder — **set it**) | your S3 bucket |
| `DATASET` | `shakespeare_char` | set to `openwebtext_300m` |
| `TARGET_LOSS` | — (**required** for the experiment) | stop at `val_loss ≤` this |
| `N_LAYER/N_HEAD/N_EMBD/BLOCK_SIZE` | `12/12/768/1024` | GPT-2-small |
| `GLOBAL_BATCH_SIZE` | `64` | constant global batch (the control) |
| `BATCH_SIZE` | `4` | per-rank micro-batch (T4 memory) |
| `EVAL_INTERVAL_STEPS` | `50` | val cadence = target-detection granularity |
| `SCALING_CAP_SECONDS` | `1800` | per-run wall-clock cap (30 min) |
| `PREEMPT_OFFSETS` | `600,1200` | kill times (s after train start) |
| `CALIBRATE_SECONDS` | `300` | calibrate probe length |
| `VCPU_QUOTA` | `8` | must be ≥16 for 4-node runs |
| `OWT_TARGET_TOKENS` | `300_000_000` | prep corpus cap |

## Key code

- `data/openwebtext_300m/prepare.py` — corpus prep (capped OWT → bins).
- `src/orchestrator/dataset.py` — `stage_data`; finds repo-level `data/<dataset>/prepare.py`;
  `meta.pkl` optional.
- `src/orchestrator/experiments.py` — `run_calibrate`, `run_scaling_experiment`,
  `_analyze_target`, `_calibration_sizing`, `_write_scaling_report`, `_prepare`
  (staged-check keys on `train.bin`).
- `src/spot_train/train.py` — `TARGET_LOSS` early-stop (in the eval block; rides the
  coordinated group stop).
- `src/spot_train/config.py` — model dims + `target_loss` from env.
- `src/orchestrator/config.py` — `_TRAINER_PASSTHROUGH` (relays `N_*`, `TARGET_LOSS`, … to boxes).
- `src/orchestrator/logview.py` — the live dashboard + `export_gantt` used for report PNGs.
- `src/orchestrator/supervisor.py` — the epoch supervisor (elastic membership,
  sticky-survivor master); `docs/multinode-design.md` for the protocol.

## Gotchas / open items

- **Bucket placeholder** — set `SPOT_TRAIN_BUCKET` to a real bucket (root cause of the
  current "not staged" / silent-stage confusion).
- **`stage-data` prints nothing while uploading ~572 MB** — expected; not a hang.
- **`prepare.py` hangs on exit** — HF streaming threads; Ctrl-C safe, bins are written.
  (Nice-to-have: add `os._exit(0)` after the writes so it returns cleanly.)
- **T4 memory/perf for GPT-2-124M** — un-validated until `calibrate` runs; shrink the
  config if it OOMs / is too slow.
- **Real spot reclaims** — the preempt runs schedule 2 kills, but AWS may reclaim a
  box on its own; the supervisor handles it identically (a `down` event, cause
  `reclaimed`, vs the scheduled `killed`). More than 2 ✗ on the Gantt = the market
  reclaimed extra boxes.
- **If a run never hits target** within the cap, that hypothesis prints
  `INCONCLUSIVE` — lower `TARGET_LOSS` (make it easier) or raise `SCALING_CAP_SECONDS`.
