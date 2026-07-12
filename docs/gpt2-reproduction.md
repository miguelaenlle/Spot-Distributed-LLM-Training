# Scaling to a GPT-2-class reproduction on spot

Guidance for taking the current spot/elastic trainer from the Shakespeare
fixture to a **GPT-2-class pretraining run** on a real corpus. Kept general — the
exact model size and token budget depend on how much compute we're willing to
buy — with **GPT-2-124M on OpenWebText as the running example**. The one piece
that is *not* optional at this scale is a **sharded, streaming data plane**; the
rest is sizing and hardening what we already have.

## What is (and isn't) new

| Concern | Status | Why |
|---|---|---|
| **Model** | done — config only | nanoGPT's `GPT`/`GPTConfig` *is* the GPT-2 architecture (loads OpenAI's weights). "GPT-2 small/medium/large" = bigger `n_layer/n_head/n_embd/block_size`. We already import it and own only the fault-tolerance layer. Nothing to write. |
| **Elastic control plane** | mostly done | The epoch supervisor, sticky-survivor master, two-tier checkpoints, budget-in-checkpoint, and the observability stack (event-sourced Gantt, cost ledger) carry over. Scaling changes *degree*, not design. |
| **Data plane** | **must build** | A monolithic `train.bin` downloaded in full to every node blocks step 1 for minutes and is re-downloaded on every preemption replacement. At GB–TB scale that dominates boot and recovery. This is the centerpiece. |
| **Memory** | must add | GPT-2-medium+ won't fit a single T4 with optimizer state at a useful batch. Needs activation checkpointing and/or parameter/optimizer sharding (FSDP). |
| **Checkpoints** | must scale | Checkpoints grow from MB to GB; the rank-0 S3 write and the resume must stay off the critical path (async, and eventually sharded). |
| **Cost** | must budget | A full GPT-2 run is GPU-days. We will likely validate the *pipeline* at a smaller size / token budget, not train to the literature loss. |

## Goal, stated as a loss + a budget

"Reproduce GPT-2" concretely = train nanoGPT at a chosen config on a real corpus
to a **target validation loss** within a **fixed compute budget**. For the 124M
example the literature target on OWT-val is ~2.85 (nanoGPT reaches it in ~4 days
on 8×A100). We do **not** assume that budget. Instead:

- Pick the **largest config whose pipeline we can afford to validate** (may be
  124M, may be smaller), and a **token budget** (not "until convergence").
- Success = "the elastic system trains this config to its budget's target loss,
  on spot, for less than the on-demand cost, surviving preemptions" — the same
  1c headline, at real-model scale. The *number* matters less than proving the
  system holds up as model, data, and world size all grow.

Rough compute estimate to size the budget (Chinchilla-style):
`training_FLOPs ≈ 6 · N_params · N_tokens`; `GPU-hours ≈ FLOPs / (GPU_FLOPs · MFU)`.
124M over ~3B tokens ≈ 6·1.2e8·3e9 ≈ 2.2e18 FLOPs; a T4 at ~40 TFLOP/s × ~30% MFU
≈ 1.2e13 eff FLOP/s → ~50 GPU-hours → ~13 h on 4 nodes. Scale this table before
committing spend; the driver already records $/run so we can measure vs predict.

## The data plane (the required piece)

**Principle:** never download the whole corpus before step 1; **stream shards,
prefetch ahead, overlap I/O with compute, cache locally with a bound, and
checkpoint the stream position.** The sustained rate is tiny (a T4 consumes
~20–40 KB/s of raw token bytes; even a large cluster is ~100 MB/s aggregate — S3
handles it easily), so this is a *latency + don't-block* problem, not a bandwidth
one. This is what MosaicML StreamingDataset / WebDataset / tf.data all do, and
it's the cloud-native counterpart to the HPC pattern (mmap an indexed dataset on
a shared Lustre/FSx filesystem).

Concretely, on top of what we have:

1. **Prep to shards, not a monolith.** Tokenize the corpus (GPT-2 BPE) into many
   ~128–256 MB shards written to `s3://<bucket>/data/<dataset>/shards/NNNNNN.bin`
   plus a small `manifest.json` (shard sizes, token counts, order seed). One-time
   offline job; replaces the single `train.bin`. `stage-data` already "prepare
   once → S3"; this generalizes the layout.
2. **Streaming loader** (replaces `data.py`'s "download the whole bin" in
   `_ensure_data`). Each node opens the manifest, streams the shards assigned to
   it, **prefetches K shards ahead** on a background thread, and keeps a
   **bounded local LRU cache** (cap disk at a few GB). Step 1 begins after shard 1
   lands (~seconds), independent of corpus size.
3. **Per-rank sharding.** Assign shards to ranks (Megatron-style document/shard
   sharding) so each node streams ~1/N of the data, not the whole corpus — today
   every node grabs the entire bin, which is `N×` wasted at scale.
4. **Resume from `(shard_id, offset)`.** We *already* checkpoint the data-loader
   position (a guiding principle). Promote it from a flat index to
   `(shard_id, offset)` so a resumed or **replacement** node re-enters the stream
   where it left off and fetches only the shards around that point — this is what
   removes the full re-download from preemption recovery (it prefetches during the
   ~13 s rendezvous instead). Determinism across a world-size change: seed the
   shard shuffle by `(epoch_seed, rank, world_size)` as the training RNG already
   is, so the stream is reproducible after an elastic reshape.

**Alternative (keep the mmap simplicity):** mount **FSx for Lustre / EFS** on all
nodes holding the dataset → zero per-node download, POSIX mmap unchanged. Simpler,
but adds standing infra + cost and a shared-throughput ceiling. For a pure-spot
thesis, stream-from-S3 is the cheaper, more elastic default; FSx is the pragmatic
shortcut if loader complexity isn't worth it for a one-off run.

## Model & training config at scale

- **Size = config.** Set `N_LAYER/N_HEAD/N_EMBD/BLOCK_SIZE` (add as env knobs +
  relay to boxes) for the target GPT-2 size; vocab 50257 (GPT-2 BPE) with a real
  corpus. `block_size` 1024 for GPT-2 proper.
- **Constant global batch** via the existing grad-accum machinery so the loss
  trajectory is world-size-invariant (the control that makes the 2- vs 4- vs
  N-node comparison meaningful).
- **Memory:** enable **activation (gradient) checkpointing** for medium+; for
  configs that don't fit even so, move to **FSDP** (shard params + optimizer
  state across ranks). Mixed precision (bf16 on newer GPUs, fp16+GradScaler on
  T4). This is the biggest *code* change beyond the data plane.
- **LR schedule:** GPT-2-style warmup + cosine decay to `min_lr`, sized to the
  token budget (not step count) so it's world-size-invariant.

## Checkpointing at scale

- Checkpoints go from MB → GB. Keep the **async S3 writer** (already built) so the
  loop only pays the CPU snapshot; the two-tier (node-local disk for instant
  survivor restore + rank-0 S3 for replacements) still applies.
- Beyond a few hundred MB, adopt **sharded checkpoints** (`torch.distributed.
  checkpoint` / FSDP sharded state dict): each rank writes its own shard, so save
  and restore scale with world size instead of bottlenecking rank 0. The
  group-MIN resume-step agreement stays.
- Dense-enough checkpoint interval that lost work per preemption stays bounded to
  a fraction of an epoch (the "wasted steps" the Gantt already surfaces).

## Elastic system: what changes with scale

Design is unchanged; magnitudes grow:
- **World size** up to the spot pool we can hold; the sticky-survivor master and
  observation-driven shrink/grow are already world-size-agnostic.
- **Recovery time** stays ~boot-bounded *only if* the data plane streams (item 4
  above) — otherwise every replacement re-downloads GBs. This is why the data
  plane is a prerequisite, not a nice-to-have, for GPT-2.
- **Bigger boxes** (A10/L4/A100 rather than T4) change $/token and MFU; the cost
  ledger + run profile already capture this for the on-demand-vs-spot verdict.

## Phasing (validate small, scale deliberately)

1. **Now — 15–30 min series** on a bounded **~300 M-token corpus** (~300–600 MB
   bin, downloads in seconds → *no data-plane change yet*): confirm the
   throughput (H1) and preemption-resilience (H2) hypotheses with a target-loss
   stop. Proves the elastic + observability + cost machinery on a real corpus.
2. **Data plane** — build the streaming sharded loader (items 1–4). Re-run the
   300 M series unchanged to prove boot ≈ constant and recovery no longer carries
   a download; this is the gate to any multi-GB corpus.
3. **Memory** — add activation checkpointing (+ FSDP if the target config needs
   it), validated at a medium config on a few GPUs.
4. **Scale the run** — pick the largest config whose token budget we can afford,
   set the LR schedule to that budget, and run to the target loss on spot,
   reporting the on-demand-vs-spot cost verdict + goodput + preemption timeline.

## Success criteria

- The chosen config trains to its budget's **target val loss** on spot.
- **Boot and preemption-recovery time are independent of corpus size** (proves
  the streaming data plane).
- **Spot cost < on-demand cost** for the same target, within `(1+ε)·T` wall-clock
  (the 1c headline, now at real-model scale).
- The event-sourced Gantt + cost ledger tell the whole story per run (leader
  hand-offs, world-size dips, wasted steps, $).

## Open questions / risks

- **Budget** — a full GPT-2-124M run is ~tens of GPU-hours; larger sizes are
  GPU-days. We will likely validate the pipeline below the literature target and
  extrapolate, rather than pay for full convergence.
- **Streaming loader correctness under elastic reshape** — the shard→rank
  assignment must stay deterministic across world-size changes so no data is
  skipped/duplicated after a shrink/grow; needs the same care the training RNG got.
- **FSDP × the epoch protocol** — sharded params/optimizer interact with the
  restore path (a replacement must reconstruct its shard); design when we actually
  need a config that doesn't fit with activation checkpointing alone.
- **Tokenizer/data prep cost** — tokenizing OWT-scale corpora is itself a
  multi-hour offline job; budget it separately from training.
