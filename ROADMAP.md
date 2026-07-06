# OptiTrain — Roadmap: Pretrain → Finetune → Serve, One Platform

## Context

**Ultimate goal: OptiTrain** — a unified platform for building LLMs affordably
and serving them at scale. Pretrain your model on spot, RL-finetune it, then
serve it — all on one system, running in the cloud, orchestrated by Go and
Kubernetes, fronted by the lightest possible UI: a static frontend on Firebase
and a minimal admin-password backend on a tiny AWS box. Everything below builds
toward that; the platform layer itself is Part 7.

The repo already proves fault-tolerant **pretraining** on AWS spot (orchestrator
on the laptop, trainer on GPU boxes, S3 as the only transport). The credential
design anticipated this endgame: `aws.py` is role-first, so the same
orchestrator code runs with an attached instance-profile role when the control
plane moves to a cloud node. Two expansion tracks build the missing pieces:

- **Track A — Inference fleet:** a serving system that mimics a real
  completions API. Python first, then Go/K8s. Bootable from the laptop, runs on
  cloud spot instances (preemption handled by rerouting failed requests).
  Includes a stress-test system.
- **Track B — RL finetuning:** GRPO-style post-training. Toy first
  (shakespeare-char, colocated, pure-Python — design complete), then a real
  ~1B LLM, then a disaggregated rollout fleet where the two tracks merge.

Decisions locked:
- Execute **fleet Python MVP first**, then one part at a time.
- Fleet serves **our trained nanoGPT checkpoints first**, graduates to a real
  1B LLM via vLLM later.
- K8s: **kind on laptop → k3s on EC2 spot** (no EKS control-plane cost).
- Stress test: **custom Go load generator** (doubles as the Go learning
  project).
- RL: **shakespeare-char + rule-based verifiable reward**, **colocated GRPO**
  under existing DDP/multinode machinery; disaggregation later.
- Fleet today: 2 nodes × 1 GPU (g4dn.xlarge, 8 G-vCPU quota); 4×4 or 8×1 soon
  — everything must scale by config only.

Execution order: **Part 1 → 2 → 3 → 4 → 5 → 6 → 7** (fleet-Py → RL-toy →
fleet-Go/K8s → fleet-vLLM → RL-1B → disaggregated RL → OptiTrain platform).
Each part gets its own detailed planning session when it starts; Parts 1–2 are
implementable from this document.

---

## Part 1 — Inference fleet MVP (Python) + Go load generator

**Goal:** a router + N spot workers serving our trained shakespeare model
behind one HTTP endpoint; kill a worker under load and the client sees
(near-)zero failed requests. New top-level dirs: `inference/`-shaped package
and `loadgen/`.

### Components

**Worker** (Python, FastAPI + uvicorn)
- Loads a checkpoint from S3 at boot: reuse `spot_train.checkpoint.load_latest`
  + `s3_store` + `train.build_model` verbatim; char codec from `meta.pkl`
  (reuse the encode/decode logic in `src/spot_train/sampling.py`).
- Endpoints: `POST /v1/completions` (prompt, max_tokens, temperature, top_k →
  completion text; OpenAI-response-shaped JSON), `GET /healthz`,
  `GET /v1/models`, `GET /stats` (requests served, tokens/s, queue depth).
- Serially batched inference is fine for MVP (one `model.generate` at a time;
  continuous batching is Part 4 vLLM territory).
- Heartbeat: every ~5s write `fleet/<fleet_id>/workers/<worker_id>.json`
  (private_ip, port, last_seen, model/run_id, requests_served) to S3 — matches
  the repo's S3-as-transport convention, no new IAM.

**Router** (Python, FastAPI)
- Worker registry: poll the S3 heartbeat prefix every few seconds; a worker is
  live if `last_seen` is fresh (~15s TTL). No new AWS permissions needed.
- Load balancing: round-robin over live workers; **retry-on-failure** —
  connection error / 5xx / timeout → requeue the request to the next live
  worker (this is the spot-preemption story: a terminated worker's in-flight
  requests get rerouted, not dropped). Bounded retries + request timeout;
  idempotent by nature (completions).
- Runs on a small **on-demand** box (t3.small — off the G quota); the one
  stable component, like the future rendezvous t3.micro.
- `GET /fleet/status` — live workers, per-worker stats, aggregate.

**Orchestrator** — `src/orchestrator/fleet.py` + CLI verbs in `__main__.py`
- `spot-orchestrate fleet up [--workers N] [--market spot] [--local]`,
  `fleet status`, `fleet down`, `fleet kill-worker`.
- Reuses `aws.py` (launch/terminate/tags), the `bootstrap.py` user-data pattern
  (new builder: start worker or router instead of trainer), the baked AMI, the
  security group (router→workers within the SG — the self-referencing TCP rule
  already exists; public→router :8000).
- `--local`: same code paths, workers + router as local uvicorn processes
  against a local checkpoint dir — the laptop-bootable requirement, mirroring
  how the trainer runs locally on CPU.
- **Cheap-infra knob:** the 10M-param model serves fine on CPU —
  `FLEET_WORKER_INSTANCE_TYPE=c7i.large` (or t3) lets us stress the *infra*
  (routing, rerouting, heartbeats, kills) without touching the G-vCPU quota;
  GPU workers are a config change.

**Load generator** — `loadgen/` (Go module, own `go.mod`; the Go learning
project)
- CLI: `loadgen -url ... -rps 50 -duration 5m -ramp 30s -prompts prompts.txt`.
- Concurrency via goroutines + channels; per-request latency recorded into an
  HDR-style histogram; per-second timeseries (rps, errors, p50/p95/p99) written
  to JSON/CSV + terminal summary.
- **Chaos mode:** `-kill-after 2m -kill-cmd "spot-orchestrate fleet
  kill-worker"` (or manual) — measures error blip, rerouted-request success
  rate, p99 recovery time after a worker dies mid-test.

### Headline experiment (Part 1 success criteria)
Fleet of 2 spot workers + router; loadgen at fixed RPS; hard-kill 1 worker
(`TerminateInstances`) mid-test:
1. Client-visible error rate ≈ 0 (retries absorb the kill; report shows the
   reroute).
2. Router drops the dead worker within heartbeat TTL; p99 recovers within one
   TTL window.
3. Report tokens/s, latency percentiles, and $/1M-tokens from the existing
   cost-ledger pattern (`profile.py`'s per-instance ledger generalizes; fleet
   writes `fleet/<fleet_id>/profile.json`).

### Verification
- Local: `fleet up --local --workers 2`, loadgen against localhost, kill a
  worker process → reroute works with zero cloud spend.
- Unit: router registry TTL logic, retry policy, worker request/response schema
  (pytest, CPU-only).
- Cloud: the headline experiment above on 2 CPU workers first, then 2 spot
  g4dn workers.

### Technologies learned
FastAPI/uvicorn, HTTP LB + health-check/heartbeat patterns, retry/timeout
budgets, **Go fundamentals** (goroutines, channels, context, net/http, flag,
histograms), latency percentile methodology.

---

## Part 2 — RL toy: GRPO on shakespeare-char (Python) — design complete

**Goal:** the RL analogue of the 1a experiment — `rl-preempt` run where
segment 2's reward **continues** from segment 1 (not reset to the init policy),
`resumed=true`, spot cost < on-demand RL baseline. Colocated: every DDP rank
generates its own rollouts, then a normal DDP step. Scales 2×1 → 4×4 / 8×1 by
`NODES`/instance type only.

### New files
```
src/spot_train/rl/
  config.py    # RLConfig(TrainConfig) + from_env; validates prompt_len+max_new_tokens <= block_size, G >= 2
  reward.py    # pure reward registry: (completion ids, itos) -> float; selected by RL_REWARD
  grpo.py      # group_advantages, completion_logprobs, exact-KL, policy loss (pure math, CPU-testable)
  rollout.py   # RolloutBatch + RolloutSource protocol + LocalRolloutSource (the disaggregation seam)
src/spot_train/rl_train.py   # entrypoint mirroring train.py's skeleton; train.py stays byte-identical
```
Reused **verbatim, zero edits**: `checkpoint.py` (same blob schema
`{version:1, step, model, optimizer, rng, loader}`, `AsyncCheckpointer`,
`verify`, `smoke_test`), `s3_store.py`, `rng.py`, `distributed.py`,
`interruption.py`, `data.py` (`PositionedLoader` = prompt sampler),
`sampling.py` (RNG-isolated interval text snapshots); imported from `train.py`:
`build_model`, `eval_full`, `get_lr`, `_write_metrics`.

### The GRPO step (per rank, per step s)
1. `torch.manual_seed(seed + rank*1_000_003 + s)` — **rollouts are a pure
   function of (seed, rank, step)**: resume re-crosses step s and regenerates
   identical rollouts (zero new checkpoint state; a replacement node with the
   same node_index reproduces the dead node's stream). Matches `sampling.py`'s
   (seed, step) precedent.
2. Prompts: `PositionedLoader.next_batch("train")[0][:, :prompt_len]` — real
   Shakespeare contexts.
3. `raw_model.generate(prompts.repeat_interleave(G, 0), max_new_tokens,
   temperature)` — unwrapped model (DDP has no `.generate`), `no_grad`,
   fixed-length (no EOS in char vocab) → rectangular batch, no collectives
   inside generation.
4. Rewards (local, pure) → group-relative advantages `(r − mean_g)/(std_g + ε)`
   per prompt group — no cross-rank collective.
5. Policy forward **with targets** (`model(seq[:,:-1], seq[:,1:])` — nanoGPT
   returns full-sequence logits only when targets≠None); gather
   completion-token logprobs; frozen reference model same forward under
   `no_grad`; **exact KL** over the 65-token vocab on completion positions
   (vocab is tiny; k3 estimator only when vocab grows).
6. `loss = −(adv·logp).mean() + kl_coef·kl.mean()`; backward (DDP allreduce —
   first collective of the step); clip; step.
7. Collective order per step: backward → (every log-interval) cross-rank
   mean-reward via the `mean_loss`-style helper (deterministic gate) →
   `all_reduce_stop` **last**, exactly as `train.py:297-302`.

Starting knobs (T4, no KV cache in nanoGPT's generate → generation dominates):
**P=4 prompts/rank, G=8, prompt_len=32, max_new_tokens=96** ≈ 5–7 s/step.
`RL_KL_COEF=0.05`, LR ~1e-5–3e-5 via existing `LEARNING_RATE`. Worst-case lost
work on hard kill = one RL step + one in-flight upload (checkpoints fire
between steps); accept + document — no partial-rollout checkpointing (would
violate one-resume-path for ≤10 s savings).

### Resume semantics — one path, extended once
```
blob = checkpoint.load_latest(cfg.checkpoint_uri)            # own run's latest
if blob: restore_into(...)                                   # resumed=true
else:    init = checkpoint.verify(s3_store.latest(cfg.init_checkpoint_uri))
         raw_model.load_state_dict(init["model"])            # weights only; fresh optimizer; step 0
         # missing INIT_CHECKPOINT_URI -> hard SystemExit (RL from random weights is meaningless;
         # deliberate deviation from silent-fresh fallback, documented in the module docstring)
ref_model = rebuilt from init URI every boot; .eval(); requires_grad_(False)   # never checkpointed
```
DDP wrap strictly after restore (as `train.py:175-183`). Rank 0 writes
`runs/<run_id>/init.json` sidecar (init URI/step) on first boot; warn on later
mismatch. Blob stays version 1 — orchestrator's `max_checkpoint_step` watchdog,
`verify`, `smoke_test` all work unmodified.

### Reward (registry in `rl/reward.py`, default `line_length`)
Triangular kernel on completed (newline-terminated) lines of the completion:
`score(L)=max(0, 1−|len(L)−target|/target)`, target 40 chars; reward = mean
over completed lines, 0.0 if none. Smooth difficulty gradient (pretrained init
starts mid-range → group variance from step 1), degenerate-proof (all-newlines
and no-newline both score 0), pure function of ids+codec. Alternatives shipped
in the registry: speaker-line format (`^[A-Z][A-Z ]{1,15}:$`), punctuation
cadence, weighted composite.

### Config / env (orchestrator passthrough additions in
`src/orchestrator/config.py`)
`INIT_CHECKPOINT_URI` (required), `RL_GROUP_SIZE=8`, `RL_ROLLOUT_PROMPTS=4`,
`RL_PROMPT_LEN=32`, `RL_MAX_NEW_TOKENS=96`, `RL_GEN_TEMPERATURE=1.0`,
`RL_GEN_TOP_K=0`, `RL_KL_COEF=0.05`, `RL_REWARD=line_length`,
`RL_TARGET_LINE_LEN=40`.

### Orchestrator changes (exhaustive)
1. `bootstrap.py`: `build_user_data(..., train_module="spot_train.train")`;
   thread through `_multinode_loop` and the two single-node run strings;
   `_trainer_env` emits `INIT_CHECKPOINT_URI`.
2. `experiments.py`: `run_multinode`/`run_multinode_preempt`/`_run_single_box`
   gain `kind`/`train_module`/`extra_env` params; new ~15-line wrappers
   `run_rl(cfg, init_run)` / `run_rl_preempt(cfg, init_run)`;
   `_resolve_init_run` refuses if init run's `metrics.json` or checkpoints are
   absent (`aws.object_exists` / `any_object_under`); **fix the pkill pattern**
   — `spot_train.train` does not match `spot_train.rl_train`; parameterize by
   module.
3. `__main__.py`: `rl` / `rl-preempt` subcommands with required
   `--init-run <run_id>` (multinode via `NODES`, as today).
4. `profile.py`: additive regexes for the RL log lines (below); RL lines must
   also stamp `_first_sample_wall`/`_last_sample_wall` so phase attribution
   keeps working; W&B mirrors reward+kl. Budget.json, rendezvous, generation
   loop, watchdog, quota gate: **zero changes** (verified — they key off
   checkpoint names and metrics.json only).

Log line (rank 0): `rl step 120: reward 0.4312, kl 0.0231, loss -0.0187,
2140ms/step, 4780 tok/s` — the `rl ` prefix + `reward` slot can't half-match
the existing `_STEP_RE` (which requires `loss` right after the colon). Sanity
line: existing `eval step S: val_loss X` from `eval_full` — a drifting LM val
loss is the KL-runaway alarm. `metrics.json` keeps `resumed`/`stop_reason`/etc.
and adds `mean_reward_final`, `kl_final`, `init_run`, `rl: true`.

### Tests (CPU, before any cloud spend — the determinism gate)
- `tests/test_reward.py` — kernel properties, degenerate cases, determinism
  (codec built inline).
- `tests/test_grpo.py` — advantages zero-mean/unit-std per group;
  constant-group → zeros; gradient-sign test (positive-advantage completion's
  logprob rises after one SGD step on a tiny model); KL ≥ 0, =0 at policy==ref;
  prompt positions masked out of logprobs.
- `tests/test_rl_kill_resume.py` — mirror of `test_kill_resume.py` with a
  TinyGPT defined in-test: K steps recording rewards, checkpoint at J,
  fresh-process restore, continue → **bit-identical reward trajectory** vs
  uninterrupted run.
- Extend `tests/test_bootstrap.py` (module param; default user-data
  byte-identical) and `tests/test_profile.py` (RL regex round-trip).

### Sequencing
reward.py+tests → grpo.py+tests → rl/config.py → rollout.py+rl_train.py +
local CPU smoke (pretrain ~200 steps locally →
`INIT_CHECKPOINT_URI=checkpoints/ python -m spot_train.rl_train`, watch reward
rise; requires `git submodule update --init`) → kill-resume test →
orchestrator changes + `--dry-run` walk → push to main (boxes track branch
tip) → cloud ladder: 1×1 on-demand `rl` → NODES=2 `rl` → NODES=2 `rl-preempt`
headline.

### Technologies learned
Policy gradients/GRPO for real (advantages, KL control, reward hacking),
RL-specific fault tolerance (rollout determinism), collective-order discipline
under DDP.

---

## Part 3 — Fleet on Go + Kubernetes

**Goal:** rewrite the router as a Go service; run the fleet on K8s — kind on
the laptop, k3s on EC2 spot in the cloud. Same headline experiment as Part 1,
now with K8s-native failure handling. New dirs: `router-go/` (Go), `deploy/`
(manifests).

- **Go router:** `net/http` + `httputil.ReverseProxy`; worker registry (S3 poll
  first; swap to K8s Endpoints watch once in-cluster); per-worker health,
  retries, timeouts via `context`; Prometheus `/metrics` (request counts,
  latency histograms, per-worker state). This is the "Go control plane" the
  repo always planned (the Phase-1c supervisor shares idioms:
  observe/compare/act loop, goroutines, backoff).
- **Containerization:** Dockerfiles for worker (Python) + router (Go,
  multi-stage build, distroless), pushed to ECR.
- **Laptop:** `kind` cluster; `kubectl apply -k deploy/local` runs router +
  2 CPU workers; loadgen from the host.
- **Cloud:** `k3s` — server on the t3.small (router node), agents on spot EC2
  workers; orchestrator gains `fleet k8s-up` (launch nodes, install k3s via
  user-data, join tokens via S3). GPU workers: NVIDIA container toolkit +
  device plugin on g4dn agents.
- **Spot preemption, K8s-style:** a DaemonSet IMDS watcher (Go) that
  cordons+drains the node on spot notice (the productionized
  `interruption.py`); router retries cover hard kills with no notice — measure
  both with loadgen chaos mode.
- **Autoscaling:** start with manual `kubectl scale`; then HPA on a custom
  metric (queue depth / tokens-per-s) via prometheus-adapter; keep optional.
- Observability: Prometheus + Grafana in-cluster (helm chart), one dashboard:
  RPS, p99, worker count, node kills.

**Verify:** kind e2e (kill a pod under load → reroute), cloud e2e (kill a spot
node under load → drain/reroute), same $/1M-tokens report.

**Technologies:** Docker multi-stage builds, kind/k3s, kubectl,
Deployments/Services/DaemonSets, drains/PDBs, Helm/Kustomize,
Prometheus/Grafana, Go services in production shape (context, graceful
shutdown, reverse proxy), ECR.

---

## Part 4 — Real model serving: vLLM on the fleet

**Goal:** swap the toy worker for **vLLM** serving a real small LLM
(Qwen2.5-0.5B-Instruct or Llama-3.2-1B-Instruct) on GPU nodes; the
router/K8s/loadgen layers are unchanged (that's the point of the seam).

- vLLM's OpenAI-compatible server as the worker container; our router now
  fronts a *real* completions API (the "mimics a real API" end state).
- Instance move: g4dn (T4, 16 GB, fp16) works for 0.5–1B; g5.xlarge (A10G,
  24 GB, bf16) preferred — **file the G/VT quota bump early**, as CLAUDE.md
  warns.
- Learn the serving internals the toy hid: continuous batching,
  PagedAttention/KV-cache memory budgeting (`gpu_memory_utilization`,
  `max_num_seqs`), prefill vs decode behavior under load (visible in loadgen
  latency distributions — add TTFT measurement to loadgen).
- Spot: model load takes ~seconds-minutes → drain-on-notice matters more;
  measure recovery time vs Part 3.

**Verify:** loadgen chaos experiment on 2 vLLM spot workers; report
tokens/s/$ vs on-demand.

**Technologies:** vLLM ops, HF model/tokenizer ecosystem, GPU scheduling on
K8s, TTFT/throughput tradeoffs.

---

## Part 5 — RL on a real LLM (~1B)

**Goal:** rerun the Part 2 experiment shape on a 0.5–1B HF model with a real
tokenizer — same fault-tolerance layer, bigger model. Validation-at-scale, not
new infra (mirrors 1d's role).

- Model via HF `transformers` (nanoGPT retires for this track); our GRPO
  loop/checkpoint/orchestrator stay. `RolloutSource` unchanged.
- Rewards: verifiable text tasks appropriate for a 1B instruct model
  (format-following, regex-checkable structure, simple arithmetic) — still
  rule-based, still pure.
- Memory: 1B fp32 AdamW ≈ 16 GB+ — needs bf16 (A10G/g5) and possibly gradient
  checkpointing; T4 (no bf16) is out. Colocated HF `.generate` first (slow but
  correct), which sets up the Part 6 motivation: generation will visibly
  dominate step time.
- KL: k3 estimator now (32k+ vocab — exact KL no longer free).

**Verify:** CPU-tiny variant of the kill-resume test with a small HF model;
cloud `rl-preempt` on g5 spot: reward continues across kill, cost < on-demand.

**Technologies:** HF transformers/tokenizers, mixed precision, gradient
checkpointing, reward design at instruct-model scale.

---

## Part 6 — Capstone: disaggregated RL — the fleet serves the rollouts

**Goal:** the two tracks merge. Rollout generation moves off the learner onto
the Part 4 vLLM fleet; the learner trains on served rollouts; weights sync
back. This is veRL/OpenRLHF's core architecture on our own infra.

- `VLLMRolloutSource` implements the Part 2 `RolloutSource` protocol: requests
  G completions/prompt from the router; fills `behavior_logprobs` +
  `policy_step` (already in the `RolloutBatch` schema from day one) so
  off-by-K-steps rollouts can be importance-corrected.
- **Weight sync:** the atomic, step-keyed S3 checkpoint (`ckpt-<step>.pt`) is
  already the handle — fleet workers poll `s3_store.latest()`, hot-reload
  weights (vLLM sleep/wake or worker restart via K8s rolling update; measure
  staleness). Staleness policy: accept rollouts within K policy steps, else
  IS-correct or drop.
- Async RL: learner never waits for generation (the AReaL idea); goodput metric
  becomes learner-GPU utilization.
- Headline: same reward target as Part 5, wall-clock and $ vs colocated — show
  the disaggregation win, on spot, with preemption on *both* fleets.

**Technologies:** weight synchronization, async/off-policy correction,
multi-fleet orchestration — the exact systems problems RL-infra roles
interview on.

---

## Part 7 — OptiTrain: the unified platform

**Goal:** one system, one URL. Log in with an admin password, launch a
pretrain / RL-finetune / serve job, watch its loss/reward/cost live, and hit
the model's completions endpoint when it's done. The laptop orchestrator
retires; the control plane lives in the cloud.

### Control plane (Go, on the k3s cluster)
- The Go supervisor that Parts 3–6 built up (observe/compare/act: launch nodes,
  watch health, replace preempted boxes) becomes a long-running **controller
  service** on the k3s server node, running with an instance-profile role — no
  laptop credentials anywhere (`aws.py`'s role-first design was built for
  exactly this; port its behaviors to Go incrementally, shelling into the
  Python orchestrator as a bridge until parity).
- Job model, house style: a **job spec is a JSON document in S3**
  (`jobs/<job_id>/spec.json`: kind = pretrain | rl | fleet, config knobs,
  budget cap). The controller watches the prefix, executes, and writes status
  back (`status.json`, plus the existing `profile.json`/`metrics.json`
  artifacts). S3-as-queue keeps the backend stateless and the whole platform
  inspectable with `aws s3 ls`.
- Hard **budget caps per job** (the controller refuses/kills over-budget jobs)
  — an affordability platform must enforce its own headline.

### Backend — the lightest one possible
- One **t4g.nano/t3.micro on-demand box** (~$3–8/mo) running a single Go
  binary: static-token auth (admin password → bearer token), ~6 endpoints:
  `POST /jobs`, `GET /jobs`, `GET /jobs/<id>` (proxies status/profile from S3),
  `DELETE /jobs/<id>`, `GET /fleet/status`, `POST /fleet/scale`. TLS via Caddy
  sidecar (Let's Encrypt) or an ALB if already convenient.
- It does **no orchestration itself** — it only writes/reads S3 job documents.
  The controller does the work. That's what keeps it "the lightest backend
  ever": stateless, restartable, nothing lost if it dies.
- Alternative considered: Lambda + API Gateway (even cheaper at idle). Rejected
  for v1 — a tiny always-on box is simpler to debug, and the S3-queue design
  means either can be swapped in later without touching the controller.

### Frontend — static, on Firebase
- A minimal SPA: login form → job list with status/cost → launch-job form
  (kind + a few knobs) → run detail page rendering `profile.json`
  (loss/reward curve, cost curve, text samples) → fleet page with a "try the
  model" completions box hitting the router.
- No server-side rendering, no database — the backend proxies S3; Firebase
  Hosting serves the bundle.

### Headline demo (the platform's proof)
From the browser: launch a pretrain job → when it completes, launch an RL job
with `--init-run` pointing at it → deploy the result to the serving fleet →
run the loadgen chaos test against it — pretrain→finetune→serve, end to end,
on spot, with the total dollar cost of the whole pipeline on screen.

**Verify:** controller unit tests (job-spec state machine); e2e on cloud:
submit each job kind via the API and confirm artifacts match
laptop-orchestrated runs; auth test (no token → 401); budget-cap kill test.

**Technologies:** Go services with AWS SDK (role-based auth),
S3-as-queue/state-machine design, minimal API auth, Caddy/TLS, Firebase
Hosting, cost governance.

---

## Cross-cutting notes

- **CLAUDE.md** gets updated at the start of each part (repo convention:
  folders/docs appear when a phase begins).
- **Quota:** current 8 G-vCPUs = 2× g4dn.xlarge. Part 1 can develop on CPU
  workers (off quota). File the g5 quota increase when Part 4 starts (approval
  takes days).
- **Credentials discipline unchanged:** the user runs every credentialed
  command; Claude writes code + local tests.
- **One part at a time:** when a part begins, we re-plan it at implementation
  depth (Parts 1–2 are already at that depth; 3–7 are architecture-level on
  purpose).
- **Record demos incrementally:** each part's headline experiment gets a short
  screen recording when it lands; the final OptiTrain video is an edit job,
  not a milestone.

## Suggested learning path (mapped to parts)

| Part | Learn |
|------|-------|
| 1 | FastAPI, LB/heartbeat/retry patterns, Go basics via loadgen (goroutines, channels, context), latency percentiles |
| 2 | GRPO/policy gradients, KL control, rollout determinism, DDP collective discipline |
| 3 | Docker, kind/k3s, kubectl, drains/PDBs, Helm, Prometheus/Grafana, production Go services |
| 4 | vLLM internals (continuous batching, PagedAttention), TTFT/throughput, GPU-on-K8s |
| 5 | HF transformers, mixed precision, reward design for instruct models |
| 6 | Weight sync, async RL/importance correction, multi-fleet orchestration |
| 7 | Go + AWS SDK (role auth), S3-as-queue design, API auth/TLS (Caddy), Firebase Hosting, cost governance |

Reading list: Lambert's RLHF book (rlhfbook.com), DeepSeekMath §4 (GRPO),
Schulman's KL-approximation note, veRL/HybridFlow paper, Kipply's "Transformer
Inference Arithmetic", the vLLM paper, llm-d architecture docs.
