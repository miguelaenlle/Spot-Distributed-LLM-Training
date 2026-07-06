# Plan: nanochat-class training on the spot fault-tolerance stack

**Status: plan only — nothing here is implemented yet.**

## Goal

Train a nanochat-class chatbot (~560M-param Llama-arch, ~11B FineWeb-Edu
tokens, then midtrain + SFT) entirely on interruptible spot capacity, with
every stage surviving preemption, for **~$100 all-in** — publishing the full
cost ledger.

**Headline:** nanochat's "$100 ChatGPT" assumes 4 pristine hours on a rented
8×H100 node. Ours runs ~2.5 days on 4× g5.xlarge spot (1× A10G each), eats
real preemptions, and costs the same — because spot A10Gs deliver almost
exactly the same FLOPs-per-dollar as rented H100s. The delta is wall-clock,
and surviving that wall-clock on dying hardware is the product.

**Positioning vs. nanochat:** we take its *recipe* (model shape, Muon,
tokenizer artifact, data mixtures, stage structure) and reject its *harness*
(linear single-node scripts, no checkpoint/resume story). Same rule as
nanoGPT today: we own the fault tolerance, not the transformer.

RL (DPO, then GRPO) is deliberately **out of scope** here — see §8.

---

## 1. What already works (do not rebuild)

Inventory from the current tree — all of this carries over unchanged:

| Capability | Where |
|---|---|
| One resume path (group-agreed latest, S3 + node-local tier, fresh fallback) | `checkpoint.load_group_latest`, `train.py` |
| Atomic S3 writes + SHA-256, verify() + restore smoke test | `s3_store.py`, `checkpoint.py` |
| Async checkpointing (CPU snapshot ~tens of ms, background upload) | `checkpoint.AsyncCheckpointer` |
| Elastic DDP: stop-consensus, world-size changes, per-rank shard streams | `distributed.py`, `train.py` |
| Constant-global-batch grad accumulation with `no_sync` | `train.py` (`grad_accum_steps`) |
| Wall-clock budgets, run-level `train_budget_seconds`, metrics.json as done-signal | `train.py` |
| Orchestrator: launch/kill/resume, watchdog, baked AMI, cost ledger + live spot rates | `src/orchestrator/` |
| Sampling snapshots mid-run + end-of-run | `sampling.py` |
| CPU kill-resume test doctrine | `tests/` |

Notably: **gradient accumulation and comm-hiding already exist**, so the
2.5 Gbps baseline network on g5.xlarge is already handled by raising
`global_batch_size` — no new code.

## 2. What's missing (the plan)

Five workstreams, ordered. Each lands with CPU kill-resume tests before any
cloud time. Effort tags: S (≤1 day), M (2–4 days), L (~1 week).

---

### WS1 — bf16 autocast (S) — *gating prerequisite*

`train.py` runs fp32 today; on A10G that ~4×s the cost of every run.

- Add `precision: "fp32" | "bf16"` to `TrainConfig` (default `fp32` so CPU
  tests are untouched; runs set `bf16`). Wrap forward/backward in
  `torch.autocast(device_type, dtype=torch.bfloat16)`.
- **bf16-only.** No fp16/GradScaler path: A10G (Ampere) has bf16, and this
  track doesn't target T4s. If a T4 path is ever needed, fp16+scaler (and
  scaler state in the checkpoint) becomes its own small workstream.
- Params/optimizer state stay fp32 (autocast only touches compute), so the
  checkpoint format doesn't change.
- No `torch.compile`, no custom kernels — plain PyTorch throughout (explicit
  project constraint).

### WS2 — vendored nanochat model + Muon (M)

- **Vendor, don't submodule**: copy nanochat's model definition and Muon
  optimizer (2 files, MIT) into `src/spot_train/models/nanochat/` with a
  header noting the pinned upstream commit. Rationale: nanochat is top-level
  scripts, not a library, and moves fast — vendoring pins exactly what our
  checkpoint key layout depends on. (Keep an eye on upstream; a bump is a
  deliberate re-vendor + resume-compat check.)
- **Arch registry** in `train.py`: `cfg.arch: "nanogpt" | "nanochat"` selects
  the model builder. Existing experiments/tests keep `nanogpt`; nothing is
  removed. This is also the Phase-1d Llama-arch box ticked.
- Strip any H100-specific defaults during vendoring (compile flags, dtype
  assumptions); the module must run on CPU for tests.
- **Dual optimizer (Muon for matrices + AdamW for embeddings/scalars)** behind
  a `MultiOptimizer` shim exposing `state_dict / load_state_dict /
  param_groups / zero_grad / step`, so `checkpoint.save(optimizer=...)` and
  the LR loop are untouched. One wrinkle: the schedule must scale each group
  relative to its *own* base LR (store `base_lr` per group; `get_lr` returns
  a multiplier) since Muon and AdamW run different LRs.
- **Muon ships in v1 and is the default for the nanochat arch.** A
  `optimizer: "muon" | "adamw"` switch stays as a debugging fallback only
  (AdamW-only needs ~1.3× the tokens for the same loss — ≈ $115–125 instead
  of ~$100 — so it is never the plan of record).
- Tests: tiny-config nanochat model through the existing kill-resume suite;
  Muon momentum buffers restored bit-exact; loss continues after restart.

### WS3 — tokenizer + FineWeb-Edu at scale (M)

- **Decided: nanochat's published tokenizer artifact** (65,536 tokens —
  exactly uint16's range, so `.bin` format survives) via HF `tokenizers`.
  It pairs with the vendored model's embedding config and chat special
  tokens, so no dimension or template remapping. We do NOT train a
  tokenizer (rustbpe stays out of the dependency tree). Contingency only —
  if the artifact can't be loaded standalone: SmolLM2's tokenizer (49,152,
  also uint16-safe), accepting a template remap.
- Extend `stage-data`: `--dataset fineweb-edu --tokens 11.2B` streams
  parquet shards from the HF hub, tokenizes locally/on a cheap CPU box,
  writes `train.bin`/`val.bin` (uint16 memmap, same format as today) +
  `meta.json` (tokenizer id/hash, vocab) + the tokenizer files, uploads all
  to `data/fineweb-edu/`. Staging is resumable (per-shard progress marker);
  heavy deps (`datasets`) live on the staging side only, not on the box.
- **No loader rewrite**: 11.2B tokens ≈ 22GB — `np.memmap` random-offset
  sampling (today's `PositionedLoader`) handles that as-is; it never loads
  the file into RAM. Box needs ~60GB EBS; one-time S3 pull is ~2–4 min
  in-region (already amortized by the existing `_ensure_data` rank-0 gate).
- `sampling.py`: encode/decode via the tokenizer files instead of
  `meta.pkl` char maps when `meta.json` is present.

### WS4 — fault-tolerant chat stages: midtrain + SFT (L) — *the big one*

Key design call: **precompute packed rows at staging time, keep the trainer's
loader shape.** Conversations are rendered with the chat template, tokenized,
and packed into fixed `block_size` rows *offline*, emitting three memmap
files: `train.bin` (uint16 tokens), `train.mask.bin` (uint8 — 1 where loss
applies, i.e. assistant tokens), plus val equivalents. Then:

- The loader samples **row-aligned** indices instead of arbitrary offsets —
  a ~10-line variant of `next_batch` that returns `(x, y, mask)`; resume
  remains the existing RNG-stream mechanism. No epoch/cursor machinery
  needed beyond what exists.
- The loop computes masked cross-entropy when a mask file is present.
  Presence of `*.mask.bin` in the data dir is the switch — no
  `objective` flag, no second code path.
- **Midtrain vs. SFT is the same code with different data**: two staged
  datasets built from nanochat's published mixtures (midtrain: chat-format
  + multiple-choice mixture; SFT: higher-quality conversations). The
  fault-tolerance loop doesn't know which stage it's running.
- **Stage initialization**: new `init_weights_uri` config. Startup order
  (still one path, a fallback chain): (1) resume this stage's own latest
  checkpoint; (2) else load *weights only* from `init_weights_uri` (the
  previous stage's final checkpoint), fresh optimizer, step 0; (3) else
  random init. Guard: checkpoint records arch/config + tokenizer hash;
  mismatch on load is a hard error, not a silent divergence.
- `eval_full` gets a masked variant over fixed rows so stage val losses are
  exactly comparable across restarts (same property as today).
- End-of-SFT sampling uses the chat template so `samples.json` shows real
  conversations.
- **Also enabled by this stage (secondary goal): finetuning existing
  bases** — an HF-safetensors → vendored-model weight converter makes
  `init_weights_uri` point at, e.g., SmolLM2-135M. Full finetune only for
  now; LoRA is deliberately deferred until a >1B base matters.
- Tests: mask correctness (loss ignores prompt tokens), kill-resume mid-SFT
  (loss continues, mask stream identical), stage-init from a pretrain
  checkpoint, tokenizer-hash mismatch fails loudly.

### WS5 — pipeline orchestration + spend cap (M)

- `experiments.py` grows `run_pipeline`: an ordered stage list
  (pretrain → midtrain → SFT), each stage with its own run_id,
  `checkpoint_uri`, `data_uri`, budget, and `init_weights_uri` chained from
  the previous stage's final checkpoint. A stage is complete when its
  `metrics.json` exists — which makes the *pipeline* resumable the same way
  a run is (completed stages are skipped on re-invoke).
- Per-stage ledger entries + a pipeline-level roll-up: the "$100" artifact
  is one table covering GPU spot, EBS, S3, and the rendezvous box.
- **Spend cap**: orchestrator kills the fleet when the live ledger crosses a
  configured ceiling — the second safety net under the checkpoint system.
- CLI: `spot-orchestrate pipeline --profile nanochat-d20` (stage recipe in a
  checked-in profile file).

---

## 3. The runs

### Validation ladder (~$15 total; each tier gates the next)

| Tier | What | Cost |
|---|---|---|
| 0 | CPU kill-resume tests for WS1–WS4 (repo doctrine: determinism before cloud) | $0 |
| 1 | 1× g5.xlarge spot: ~20M model, ~150M tokens, **full pipeline** end-to-end (stage-data → pretrain w/ controlled kill → midtrain → SFT → chat samples) | ~$2 |
| 2 | 4× g5.xlarge, real 560M config, ~1h budget: measure actual tok/s + MFU (validates the $100 math), deliberately preempt one node, confirm re-rendezvous + ledger goodput accounting | ~$3 |
| 3 | ~1B-token pilot at full size (~4h): pin LR/warmup/batch against the expected loss curve before committing | ~$7 |

Tier 2's measured tokens/sec re-prices the flagship before any commitment.
A bug at hour 30 of the flagship costs one checkpoint interval, not the run
— the system under test is its own insurance.

### Flagship: the $100 run

- **Cluster:** 4× g5.xlarge spot us-east-1 (~$0.40/hr each) + t3.micro
  rendezvous. 16 G-vCPUs — fits current quota, no Activate needed.
- **Model:** vendored nanochat d20 (~560M), Muon+AdamW, bf16 autocast.
- **Pretrain:** ~11.2B FineWeb-Edu tokens, global batch ~0.5M tokens,
  30s checkpoint interval → ~2.4 days at ~200 TFLOPS effective (incl. ~10%
  preemption goodput loss) ≈ **$93**.
- **Midtrain + SFT:** ~2–3h on the same fleet ≈ **$4**.
- **Overhead:** EBS ×4 + S3 + rendezvous ≈ **$4–5**.
- **Report:** full ledger vs. the ~$250 on-demand equivalent; preemption
  count, goodput, recovery time, lost-work-per-interruption (bound: the
  checkpoint interval); chat transcripts.

Budget-scaling notes: same pipeline at ~$160 upgrades to ~30B tokens
(better model, same headline structure); with Activate + 256 vCPUs the
identical run on 4× g5.12xlarge finishes in ~14h, or scales to ~1B params /
~45B tokens for ~$800.

---

## 4. Explicitly rejected / deferred

| Item | Why |
|---|---|
| nanochat as codebase/submodule | Linear single-node scripts, no interruption story; retrofitting our FT layer into it re-tests everything we've proven. Vendor 2 files instead. |
| nanochat's midtrain/SFT/eval scripts | Second training loop that dies on preemption — breaks the "every stage survives a kill" invariant. Reuse the *mixtures* only. |
| Tokenizer training (rustbpe) | Rust build dep for no headline value; published artifact is fine. |
| CORE/ChatCORE eval harness | Val loss + samples + one cheap zero-shot task suffices for the headline. |
| torch.compile / custom kernels / fp16-on-T4 | Project constraint: no kernel goofiness. bf16 on Ampere only. |
| LoRA | Not needed ≤1B full-finetune on 24GB; revisit with bigger bases. |
| FSDP/ZeRO | DDP ceiling is ~1B params on A10G; that's beyond this plan's model. |

## 5. New dependencies

`tokenizers` (box + staging), `huggingface_hub` (staging), `safetensors`
(only when the HF-base-import path lands), `datasets` (staging only — never
on the box). All pinned in `pyproject.toml`; box footprint stays minimal so
the baked AMI stays small.

## 6. Order of work

1. WS1 (bf16) — unblocks all pricing math.
2. WS2 (model + Muon) — through CPU kill-resume tests.
3. WS3 (tokenizer + data) — through a small staged corpus.
4. Tier-1 validation run (~$2) as soon as WS1–WS3 land (pretrain only).
5. WS4 (midtrain/SFT) — through CPU tests, then re-run Tier 1 full-pipeline.
6. WS5 (pipeline + spend cap).
7. Tiers 2–3, then the flagship.

## 7. Decisions (previously open questions — all resolved)

- **Muon: in v1, default.** It's the difference between $100 and ~$120 at
  fixed quality, its state (momentum buffers) is simpler to checkpoint than
  AdamW's, and it's pure PyTorch. AdamW remains only as a debug switch.
- **Tokenizer: nanochat's published artifact** (see WS3) — it's what the
  vendored model shape and chat template assume. SmolLM2's is contingency
  only.
- **Midtrain/SFT mixtures: adopt nanochat's published mixtures verbatim**,
  pinned by HF dataset revision in the `nanochat-d20` profile file at
  vendoring time. No mixture design of our own until the flagship has run —
  recipe deviations would make quality-vs-nanochat comparisons murky.

## 8. RL — acknowledged, excluded

The natural next step after this plan is **DPO**: offline, SFT-shaped (one
extra frozen reference forward pass, no generation loop), so it inherits the
checkpoint/resume layer almost for free. WS4's design keeps that door open
deliberately — packed-rows-with-mask extends to chosen/rejected pair files,
and `init_weights_uri` already expresses "start from the SFT checkpoint."
Online RL (GRPO — nanochat's optional final stage) is a genuinely new
subsystem (generation loop, rollout buffers as resumable state) and stays
out until DPO has passed a kill-resume test. Neither is part of this plan.
