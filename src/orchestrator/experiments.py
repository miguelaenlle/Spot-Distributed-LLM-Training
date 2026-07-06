"""The two Phase 1a experiments.

  baseline — one on-demand GPU trains for ``baseline_seconds``, writes
             metrics.json; we read and print it.
  spot     — a spot GPU trains until we KILL it mid-run, then a second spot
             instance (same run prefix) resumes from the S3 checkpoint and
             finishes its ``spot_seg2_seconds`` budget. Demonstrates that a kill
             costs at most one checkpoint interval, not the whole run.

Timing note: we don't kill on a fixed sleep from launch — boot + clone + pip
takes minutes before training starts. Instead we wait for the first checkpoint
to appear in S3 (training is underway), then let it run ``spot_seg1_seconds``
before terminating. Instances are always terminated in a ``finally`` block.
"""

from __future__ import annotations

import json
import math
import sys
import time

from . import aws, bootstrap
from .config import OrchestratorConfig
from .profile import RunProfile

# seg1's trainer must not self-stop before we kill it — give it a huge budget.
_SEG1_TRAINER_BUDGET = 24 * 3600


def _run_id(kind: str) -> str:
    return f"{kind}-{int(time.time())}"


def _poll_metrics(cfg: OrchestratorConfig, run_id: str) -> dict:
    key = cfg.run_metrics_key(run_id)
    deadline = time.monotonic() + cfg.metrics_timeout_seconds
    while time.monotonic() < deadline:
        if aws.object_exists(cfg.bucket, key):
            return json.loads(aws.get_text(cfg.bucket, key))
        time.sleep(cfg.metrics_poll_seconds)
    raise TimeoutError(f"metrics.json for {run_id} did not appear within timeout")


def _launch_gated(cfg: OrchestratorConfig, do_launch):
    """Launch behind the vCPU-quota gate: wait until one box's vCPUs fit under
    VCPU_QUOTA, then RunInstances. If AWS still rejects on quota (external
    launches can grab the slack between check and call, or VCPU_QUOTA overstates
    the real quota), keep waiting and retrying — each retry sits in the 15s
    headroom poll, not a hot loop, so patience costs no API spam. The attempt
    cap is a runaway backstop, not an expected exit."""
    needed = cfg.instance_vcpu_count()
    last: Exception | None = None
    attempts = 1000
    for attempt in range(1, attempts + 1):
        aws.wait_vcpu_headroom(needed, cfg.vcpu_quota)
        try:
            return do_launch()
        except Exception as e:  # noqa: BLE001 — boto ClientError; match on the code
            if "VcpuLimitExceeded" not in str(e) and "MaxSpotInstanceCountExceeded" not in str(e):
                raise
            last = e
            print(
                f"[quota] RunInstances rejected on quota (attempt {attempt}/{attempts})",
                file=sys.stderr,
            )
            time.sleep(15)
    raise SystemExit(
        f"RunInstances rejected on quota {attempts} times — VCPU_QUOTA={cfg.vcpu_quota} "
        f"probably overstates the real account quota. Last error: {last}"
    )


def _record_instance(cfg: OrchestratorConfig, profile: RunProfile, iid: str, market: str) -> None:
    """Open a cost-ledger row for a box that just reached ``running``: its AZ
    plus the rate it's actually billed at — the live per-AZ spot price, or the
    on-demand table / HOURLY_USD override."""
    az = aws.instance_az(iid)
    rate = (
        aws.spot_hourly_rate(cfg.instance_type, az)
        if market == "spot"
        else (cfg.on_demand_hourly_usd())
    )
    profile.instance_started(iid, market, az, rate)
    if rate is not None:
        print(f"[cost] {iid} ({market}, {az}) @ ${rate:.4f}/hr", file=sys.stderr)
    else:
        print(
            f"[cost] {iid} ({market}, {az}) rate unknown — set HOURLY_USD for "
            f"{cfg.instance_type}",
            file=sys.stderr,
        )


def _launch(
    cfg: OrchestratorConfig,
    ami: str,
    sg_id: str,
    run_id: str,
    market: str,
    budget: int,
    logs_key: str | None = None,
    ddp: bool = False,
    nproc_per_node: int = 0,
    profile: RunProfile | None = None,
) -> str:
    # Run mode: provision, then train under the wall-clock budget while the box
    # syncs its (per-segment) log to S3 — so we can stream it here without SSH.
    # ddp=True => the box runs the trainer under torchrun (single-node DDP).
    ud = bootstrap.build_user_data(
        cfg,
        run_id=run_id,
        market=market,
        max_seconds=budget,
        logs_key=logs_key,
        ddp=ddp,
        nproc_per_node=nproc_per_node,
    )
    iid = _launch_gated(
        cfg,
        lambda: aws.launch(
            ami_id=ami,
            instance_type=cfg.instance_type,
            profile_name=cfg.instance_profile,
            security_group_id=sg_id,
            user_data=ud,
            market=market,
            run_id=run_id,
            key_name=cfg.key_name,
        ),
    )
    aws.wait_running(iid)
    if profile is not None:
        _record_instance(cfg, profile, iid, market)
    print(
        f"[launch] instance {iid} ({market}) — training; streaming its log below "
        f"(boot + clone take ~1-2 min before the first lines appear).",
        file=sys.stderr,
    )
    return iid


def _pull_log(cfg: OrchestratorConfig, logs_key: str, profile: RunProfile, state: dict) -> bool:
    """Pull the current segment log from S3, feed per-step samples to ``profile``,
    and print any NEW bytes. ``state = {"printed": int}``; returns True if new bytes
    were printed. Shared by the baseline stream and the preemption loop."""
    if not aws.object_exists(cfg.bucket, logs_key):
        return False
    text = aws.get_text(cfg.bucket, logs_key)
    profile.ingest_log(text)
    if len(text) > state["printed"]:
        sys.stdout.write(text[state["printed"] :])
        sys.stdout.flush()
        state["printed"] = len(text)
        return True
    return False


def _pull_logs(cfg: OrchestratorConfig, logs: dict[int, dict], profile: RunProfile) -> None:
    """Pull EVERY node's current log stream. Elastic rendezvous assigns ranks
    arbitrarily, so after a restart the loss-printing rank 0 can live on any
    node — all logs feed the profile; per-step dedup keeps re-reads idempotent.
    ``logs`` maps node index -> {"key": s3 key, "state": {"printed": int}}."""
    for entry in logs.values():
        _pull_log(cfg, entry["key"], profile, entry["state"])


def _pull_samples(cfg: OrchestratorConfig, run_id: str, profile: RunProfile) -> None:
    """Collect the trainer's text samples once the run is done: every mid-training
    snapshot under runs/<run_id>/samples/ plus the final samples.json (written just
    before metrics.json, so it's guaranteed present here). Attaches them to the
    profile (-> profile.json + W&B table) and prints the final document's texts."""
    docs: list[dict] = []
    for key in aws.list_keys(cfg.bucket, cfg.run_samples_prefix(run_id)):
        if key.endswith(".json"):
            try:
                docs.append(json.loads(aws.get_text(cfg.bucket, key)))
            except ValueError:
                print(f"[samples] unreadable snapshot {key} — skipping", file=sys.stderr)
    final_key = cfg.run_samples_key(run_id)
    final_doc = None
    if aws.object_exists(cfg.bucket, final_key):
        final_doc = json.loads(aws.get_text(cfg.bucket, final_key))
        docs.append(final_doc)
    for doc in docs:
        profile.from_samples(doc)
    if final_doc:
        print(f"\n[samples] final outputs (step {final_doc.get('step')}):")
        for s in final_doc.get("samples", []):
            print(f"\n--- prompt {s['prompt']!r} ---\n{s['prompt']}{s['completion']}")


def _stream_until_metrics(
    cfg: OrchestratorConfig,
    run_id: str,
    profile: RunProfile,
    logs_key: str | None = None,
    state: dict | None = None,
) -> dict | None:
    """Stream the box log until ``metrics.json`` appears (trainer writes it last =>
    done) or ``metrics_timeout_seconds`` elapses. Returns parsed metrics, or None.
    Pass ``state`` to continue a stream whose head was already printed elsewhere
    (otherwise the log is re-echoed from byte 0)."""
    logs_key = logs_key or cfg.run_logs_key(run_id)
    metrics_key = cfg.run_metrics_key(run_id)
    state = state if state is not None else {"printed": 0}
    start = time.monotonic()
    deadline = start + cfg.metrics_timeout_seconds
    last_heartbeat = start
    marked_first = False

    while True:
        if _pull_log(cfg, logs_key, profile, state) and not marked_first:
            profile.mark("first_log")
            marked_first = True
        if aws.object_exists(cfg.bucket, metrics_key):
            _pull_log(cfg, logs_key, profile, state)
            _pull_samples(cfg, run_id, profile)
            return json.loads(aws.get_text(cfg.bucket, metrics_key))
        now = time.monotonic()
        if now > deadline:
            print("\n[run] timeout waiting for metrics.json", file=sys.stderr)
            return None
        if state["printed"] == 0 and now - last_heartbeat >= 15:
            print(
                f"[run] waiting for the box to start logging… ({int(now - start)}s)",
                file=sys.stderr,
            )
            last_heartbeat = now
        time.sleep(cfg.log_stream_seconds)


def _stream_until_metrics_multi(
    cfg: OrchestratorConfig,
    run_id: str,
    profile: RunProfile,
    logs: dict[int, dict],
) -> dict | None:
    """Multi-log variant of :func:`_stream_until_metrics` for elastic groups:
    stream every node's log until metrics.json appears or the timeout."""
    metrics_key = cfg.run_metrics_key(run_id)
    deadline = time.monotonic() + cfg.metrics_timeout_seconds
    while True:
        _pull_logs(cfg, logs, profile)
        if aws.object_exists(cfg.bucket, metrics_key):
            _pull_logs(cfg, logs, profile)
            _pull_samples(cfg, run_id, profile)
            return json.loads(aws.get_text(cfg.bucket, metrics_key))
        if time.monotonic() > deadline:
            print("\n[run] timeout waiting for metrics.json", file=sys.stderr)
            return None
        time.sleep(cfg.log_stream_seconds)


def _prepare(cfg: OrchestratorConfig) -> tuple[str, str]:
    cfg.require_bucket()
    aws.set_region(cfg.region)
    if not aws.is_dry_run() and not aws.object_exists(
        cfg.bucket, f"{cfg.data_prefix}/{cfg.dataset}/meta.pkl"
    ):
        raise SystemExit(
            f"dataset not staged at {cfg.data_uri()} — run `spot-orchestrate stage-data` first"
        )
    ami = aws.resolve_ami(cfg.ami_id, cfg.ami_name_filter)
    sg_id = aws.ensure_security_group(cfg.security_group, cfg.region)
    return ami, sg_id


def _run_single_box(
    cfg: OrchestratorConfig,
    *,
    kind: str,
    market: str,
    budget: int,
    ddp: bool = False,
    nproc_per_node: int = 0,
) -> dict | None:
    """One box, run to its wall-clock budget, stream the log, collect the run
    profile, then terminate. Shared by `baseline` (1 process) and `ddp` (torchrun,
    N processes) — the only difference is the ddp/torchrun launch."""
    ami, sg_id = _prepare(cfg)
    run_id = _run_id(kind)
    extra = f" nproc_per_node={nproc_per_node or 'auto(gpu)'}" if ddp else ""
    print(f"[{kind}] run_id={run_id} budget={budget}s{extra}", file=sys.stderr)
    # Collect a run profile (timeline + loss) and mirror to W&B if configured.
    profile = RunProfile(run_id, kind=kind, market=market)
    if not aws.is_dry_run():  # dry-run must not create a real W&B run
        profile.wandb_start(cfg)
    iid = _launch(
        cfg,
        ami,
        sg_id,
        run_id,
        market,
        budget,
        ddp=ddp,
        nproc_per_node=nproc_per_node,
        profile=profile,
    )
    profile.mark("launch")
    try:
        if aws.is_dry_run():
            print(f"[{kind}] dry-run: skipping stream/terminate", file=sys.stderr)
            return None
        metrics = _stream_until_metrics(cfg, run_id, profile)
        profile.mark("metrics" if metrics is not None else "timeout")
        profile.from_metrics(metrics)
        profile.instance_stopped(iid)  # ledger stop ~= the terminate call below
        profile.finalize(cfg)  # write profile.json to S3 + finish the W&B run
        print(f"[{kind}] profile: {cfg.run_profile_uri(run_id)}", file=sys.stderr)
        if metrics is not None:
            print(f"\n[{kind}] metrics: {json.dumps(metrics, indent=2)}")
        return metrics
    finally:
        # Destroy the box when training finishes (also on timeout or Ctrl-C).
        aws.terminate(iid)


def run_baseline(cfg: OrchestratorConfig) -> dict | None:
    return _run_single_box(cfg, kind="baseline", market="on-demand", budget=cfg.baseline_seconds)


def run_resume(
    cfg: OrchestratorConfig,
    run_id: str,
    budget: int | None = None,
    market: str | None = None,
) -> dict | None:
    """Salvage a crashed/interrupted run: launch ONE fresh box against the same
    run prefix and let the trainer's one resume path pick up the latest S3
    checkpoint. Refuses if the run already completed (metrics.json exists) or
    never checkpointed (nothing to resume from). The new box logs to a fresh
    ``boot-rK`` key so the crashed segment's log survives in S3. profile.json /
    W&B are NOT rewritten — this is a salvage tool, not a profiled experiment.
    """
    kind = run_id.split("-", 1)[0]
    if kind == "multinode":
        raise SystemExit(
            "resume handles single-box runs; multinode runs restart their whole "
            "group via the multinode-preempt machinery"
        )
    ami, sg_id = _prepare(cfg)
    budget = budget or cfg.baseline_seconds
    market = market or ("on-demand" if kind in ("baseline", "ddp") else cfg.spot_market)
    ckpt_prefix = f"{cfg.run_prefix}/{run_id}/checkpoints/"
    if not aws.is_dry_run():
        if aws.object_exists(cfg.bucket, cfg.run_metrics_key(run_id)):
            raise SystemExit(f"{run_id} already wrote metrics.json — nothing to resume")
        if not aws.any_object_under(cfg.bucket, ckpt_prefix):
            raise SystemExit(
                f"no checkpoints under s3://{cfg.bucket}/{ckpt_prefix} — nothing to resume from"
            )
    attempt = 1
    while not aws.is_dry_run() and aws.object_exists(
        cfg.bucket, cfg.run_logs_key(run_id, attempt=attempt)
    ):
        attempt += 1
    logs_key = cfg.run_logs_key(run_id, attempt=attempt)
    ddp = kind == "ddp"
    print(f"[resume] run_id={run_id} budget={budget}s market={market}", file=sys.stderr)
    # Profile object only feeds log ingestion/streaming; never started or finalized.
    profile = RunProfile(run_id, kind=kind, market=market)
    iid = _launch(
        cfg,
        ami,
        sg_id,
        run_id,
        market,
        budget,
        logs_key=logs_key,
        ddp=ddp,
        nproc_per_node=cfg.ddp_nproc_per_node if ddp else 0,
        profile=profile,  # prints the [cost] rate line; profile is never finalized
    )
    try:
        if aws.is_dry_run():
            print("[resume] dry-run: skipping stream/terminate", file=sys.stderr)
            return None
        metrics = _stream_until_metrics(cfg, run_id, profile, logs_key=logs_key)
        if metrics is not None:
            if not metrics.get("resumed"):
                print("[resume] WARNING: trainer did not report resumed=true", file=sys.stderr)
            print(f"\n[resume] metrics: {json.dumps(metrics, indent=2)}")
        return metrics
    finally:
        aws.terminate(iid)


def run_ddp(cfg: OrchestratorConfig) -> dict | None:
    """Single-node, multi-process DDP via torchrun (Phase 1b). Same machinery as
    baseline; the box runs the trainer under torchrun with ddp_nproc_per_node ranks."""
    return _run_single_box(
        cfg,
        kind="ddp",
        market="on-demand",
        budget=cfg.baseline_seconds,
        ddp=True,
        nproc_per_node=cfg.ddp_nproc_per_node,
    )


def _launch_node(
    cfg: OrchestratorConfig,
    ami: str,
    sg_id: str,
    run_id: str,
    market: str,
    budget: int,
    *,
    node_index: int,
    logs_key: str,
) -> str:
    """Launch one member of an N-node DDP group (see bootstrap._multinode_loop
    for how the nodes find each other and rejoin after a peer dies). Does not
    wait for 'running'."""
    ud = bootstrap.build_user_data(
        cfg,
        run_id=run_id,
        market=market,
        max_seconds=budget,
        logs_key=logs_key,
        ddp=True,
        nproc_per_node=cfg.ddp_nproc_per_node,
        nodes=cfg.node_count,
        node_index=node_index,
    )
    return _launch_gated(
        cfg,
        lambda: aws.launch(
            ami_id=ami,
            instance_type=cfg.instance_type,
            profile_name=cfg.instance_profile,
            security_group_id=sg_id,
            user_data=ud,
            market=market,
            run_id=run_id,
            key_name=cfg.key_name,
        ),
    )


def run_multinode(cfg: OrchestratorConfig) -> dict | None:
    """N nodes × one-rank-per-GPU DDP via torchrun's elastic c10d rendezvous,
    run to the wall-clock budget. Node 0 hosts the rendezvous store; elastic
    assigns ranks, so the orchestrator streams EVERY node's log (rank 0's loss
    lines can come from any of them)."""
    ami, sg_id = _prepare(cfg)
    run_id = _run_id("multinode")
    budget = cfg.baseline_seconds
    market = cfg.spot_market
    print(
        f"[multinode] run_id={run_id} nodes={cfg.node_count} budget={budget}s market={market}",
        file=sys.stderr,
    )
    profile = RunProfile(run_id, kind="multinode", market=market)
    if not aws.is_dry_run():  # dry-run must not create a real W&B run
        profile.wandb_start(cfg)
    iids: list[str] = []
    logs: dict[int, dict] = {}
    try:
        # One up-front gate for the whole group (the per-launch gate inside
        # _launch_node then passes instantly) so we never fire N RunInstances
        # into a quota wall.
        aws.wait_vcpu_headroom(cfg.node_count * cfg.instance_vcpu_count(), cfg.vcpu_quota)
        for i in range(cfg.node_count):
            logs[i] = {"key": cfg.run_logs_key(run_id, node=i), "state": {"printed": 0}}
            iids.append(
                _launch_node(
                    cfg,
                    ami,
                    sg_id,
                    run_id,
                    market,
                    budget,
                    node_index=i,
                    logs_key=logs[i]["key"],
                )
            )
        for iid in iids:
            aws.wait_running(iid)
        for iid in iids:
            _record_instance(cfg, profile, iid, market)
        profile.mark("launch")
        if aws.is_dry_run():
            print("[multinode] dry-run: skipping stream/terminate", file=sys.stderr)
            return None
        metrics = _stream_until_metrics_multi(cfg, run_id, profile, logs)
        profile.mark("metrics" if metrics is not None else "timeout")
        profile.from_metrics(metrics)
        profile.finalize(cfg)
        print(f"[multinode] profile: {cfg.run_profile_uri(run_id)}", file=sys.stderr)
        if metrics is not None:
            print(f"\n[multinode] metrics: {json.dumps(metrics, indent=2)}")
        return metrics
    finally:
        for iid in iids:
            aws.terminate(iid)


def run_multinode_preempt(cfg: OrchestratorConfig) -> dict | None:
    """Multi-node preemption with ELASTIC degraded-mode recovery. Launch the
    group ONCE; per preemption: HARD-terminate one node (no SIGTERM — a graceful
    stop would coordinate a clean group shutdown, which is not the failure we
    want; a real Spot reclaim gives none of the peers any warning either). The
    survivors' collectives abort after NCCL_TIMEOUT, their elastic agents
    re-rendezvous at world N-1, and training CONTINUES from the node-local
    checkpoint tier — before the replacement even boots. The orchestrator
    launches ONE replacement; it dials the live rendezvous and the group scales
    back to N. The orchestrator is still the only launcher of boxes; torchrun
    only ever restarts worker processes on boxes that are already up.

    Two watchdog phases verify each kill:

      (a) shrink_resume — the SURVIVORS produce a new checkpoint within
          ``degraded_recovery_timeout_seconds`` of the kill (training at N-1,
          replacement not required);
      (b) full_world — a parsed step line reports the pre-kill world size again
          within ``recovery_timeout_seconds`` (the replacement joined).

    Either phase timing out falls back to the previously-proven whole-group
    restart (terminate everything, delete the stale rdzv.json so fresh boxes
    can't dial a dead store, relaunch) — worst case equals the old
    pause-and-replace behavior.

    The run-level budget rides in the checkpoint (TRAIN_BUDGET_SECONDS +
    trained_seconds), so there is no budget.json to recompute after kills —
    downtime is never billed by construction. The victim per round comes from
    PREEMPT_VICTIMS (default: always the last node). Killing node 0 is allowed
    but forfeits elastic recovery — it hosts the rendezvous store, so that kill
    exercises the whole-group-restart fallback instead. Lost work per kill is
    bounded by the densified checkpoint interval."""
    if cfg.node_count < 2:
        raise SystemExit("multinode-preempt needs NODES >= 2")
    victims = cfg.preempt_victim_schedule()  # validate before spending anything
    ami, sg_id = _prepare(cfg)
    run_id = _run_id("multinode-preempt")
    total = cfg.train_total_seconds
    budget = max(1, math.ceil(total))  # constant: the trainer subtracts trained_seconds
    market = cfg.spot_market
    interval = cfg.preempt_after_seconds or math.ceil(total / (cfg.preempt_count + 1))
    node_vcpus = cfg.instance_vcpu_count()
    ckpt_prefix = f"{cfg.run_prefix}/{run_id}/checkpoints/"
    metrics_key = cfg.run_metrics_key(run_id)
    rdzv_key = cfg.run_rdzv_key(run_id)
    # Dense checkpoints: a hard kill gives no warning, so lost work is bounded by
    # this interval (same knobs as run_preempt).
    cfg.checkpoint_interval_seconds = min(
        cfg.checkpoint_interval_seconds, cfg.preempt_checkpoint_seconds
    )
    cfg.smoke_test_every = max(1, round(30 / cfg.checkpoint_interval_seconds))
    print(
        f"[multinode-preempt] run_id={run_id} nodes={cfg.node_count} "
        f"(min {cfg.nodes_min_count()}) total_train={total}s "
        f"preemptions={cfg.preempt_count} interval={interval}s market={market} — hard-kill "
        f"victims={victims}; survivors keep training at reduced world size while one "
        f"replacement joins (whole-group restart only as fallback)",
        file=sys.stderr,
    )
    profile = RunProfile(run_id, kind="multinode-preempt", market=market)
    if not aws.is_dry_run():  # dry-run must not create a real W&B run
        profile.wandb_start(cfg)

    metrics: dict | None = None
    live: dict[int, str] = {}
    seq = 0  # counts launch events; suffixes log keys so relaunches never clobber
    logs: dict[int, dict] = {}  # node -> {"key", "state"}: every live log stream

    def _track_log(i: int) -> str:
        logs[i] = {
            "key": cfg.run_logs_key(run_id, node=i, attempt=seq),
            "state": {"printed": 0},
        }
        return logs[i]["key"]

    def _launch_group(mark: str | None) -> None:
        """Launch all N nodes (one up-front headroom gate for the whole group;
        the per-launch gate inside _launch_node then passes instantly)."""
        if mark != "launch":
            # Fresh log files re-parsed from byte 0, and the resumed rank 0
            # re-covers a few pre-restart steps — bump the profile's dedup
            # segment so only THIS group's samples read as new.
            profile.segment += 1
        aws.wait_vcpu_headroom(cfg.node_count * node_vcpus, cfg.vcpu_quota)
        for i in range(cfg.node_count):
            live[i] = _launch_node(
                cfg,
                ami,
                sg_id,
                run_id,
                market,
                budget,
                node_index=i,
                logs_key=_track_log(i),
            )
        for iid in live.values():
            aws.wait_running(iid)
        for iid in live.values():
            _record_instance(cfg, profile, iid, market)
        if mark:
            profile.mark(mark)

    def _launch_replacement(victim: int) -> None:
        """One fresh box for the victim's node index; survivors — and their log
        streams — are untouched. The replacement gets a fresh attempt-suffixed
        log key so it never clobbers the dead node's log."""
        live[victim] = _launch_node(
            cfg,
            ami,
            sg_id,
            run_id,
            market,
            budget,
            node_index=victim,
            logs_key=_track_log(victim),
        )
        aws.wait_running(live[victim])
        _record_instance(cfg, profile, live[victim], market)
        profile.mark("relaunch")

    def _fallback_restart() -> None:
        """The pre-elastic worst case: terminate everything and relaunch the
        whole group. Deletes rdzv.json first — fresh boxes must never dial the
        dead group's store address."""
        nonlocal seq
        aws.delete_object(cfg.bucket, rdzv_key)
        for iid in live.values():
            aws.terminate(iid)
            profile.instance_stopped(iid)
        for iid in live.values():
            aws.wait_quota_released(iid)
        live.clear()
        seq += 1
        fallback_base = aws.max_checkpoint_step(cfg.bucket, ckpt_prefix)
        _launch_group(mark=None)
        _wait_train_start(cfg, ckpt_prefix, fallback_base, logs, profile, metrics_key)

    kills = 0
    try:
        _launch_group(mark="launch")

        if aws.is_dry_run():
            # Walk the real control flow, minus the waiting: per planned kill,
            # terminate that round's victim, free its quota slot, launch one
            # replacement.
            for victim in victims:
                aws.terminate(live[victim])
                aws.wait_quota_released(live[victim])
                seq += 1
                aws.wait_vcpu_headroom(node_vcpus, cfg.vcpu_quota)
                _launch_replacement(victim)
            print("[multinode-preempt] dry-run: skipping stream/kill timing", file=sys.stderr)
            return None

        base_step = aws.max_checkpoint_step(cfg.bucket, ckpt_prefix)
        _wait_train_start(cfg, ckpt_prefix, base_step, logs, profile, metrics_key)
        profile.mark("train_start")
        t0 = time.monotonic()

        while kills < cfg.preempt_count:
            victim = victims[kills]
            # Train one hidden interval, then deliver the "Spot reclaim".
            while time.monotonic() - t0 < interval:
                _pull_logs(cfg, logs, profile)
                if aws.object_exists(cfg.bucket, metrics_key):
                    break  # the budget beat us to it — nothing left to kill
                time.sleep(cfg.log_stream_seconds)
            if aws.object_exists(cfg.bucket, metrics_key):
                break
            trained = time.monotonic() - t0
            kills += 1
            print(
                f"[multinode-preempt] kill {kills}/{cfg.preempt_count}: hard-terminating "
                f"node {victim}{' (the master)' if victim == 0 else ''} after ~{trained:.0f}s "
                f"training (no warning); survivors continue at world "
                f"{cfg.node_count - 1} of {cfg.node_count} nodes",
                file=sys.stderr,
            )
            # Phase-(b) reference: the world size and step the group ran at just
            # before the kill (from the trainer's `ws N` step-line suffix).
            full_ws = next((s.world_size for s in reversed(profile.samples) if s.world_size), None)
            step_floor = max((s.step for s in profile.samples), default=0)
            aws.terminate(live[victim])
            kill_t = time.monotonic()
            profile.instance_stopped(live[victim])
            profile.mark("kill")
            # Only the victim's quota slot frees — the survivors keep theirs
            # (that's the point) — so the replacement needs exactly one node's
            # worth of headroom. Survivors re-rendezvous on their own meanwhile.
            aws.wait_quota_released(live[victim])
            seq += 1
            aws.wait_vcpu_headroom(node_vcpus, cfg.vcpu_quota)
            _launch_replacement(victim)
            # Phase-(a) baseline — sampled here, bounded BELOW by the stray-upload
            # window: when the kill spares rank 0, its async checkpoint writer can
            # land one last upload up to ~NCCL_TIMEOUT past the kill, which would
            # otherwise satisfy "new checkpoint appeared" instantly (fake 0s
            # recovery). Sampling after the replacement launch makes the floor
            # nearly free (boot overlaps it); genuine survivor progress past this
            # point is exactly what phase (a) is meant to observe.
            settle = (cfg.nccl_timeout_seconds + 15) - (time.monotonic() - kill_t)
            if settle > 0:
                time.sleep(settle)
            pre_kill_step = aws.max_checkpoint_step(cfg.bucket, ckpt_prefix)
            try:
                # (a) survivors checkpointing again at N-1 — the whole point of
                # elastic: this must NOT need the replacement. Deadline anchored
                # at the kill.
                remaining = cfg.degraded_recovery_timeout_seconds - (time.monotonic() - kill_t)
                _wait_train_start(
                    cfg,
                    ckpt_prefix,
                    pre_kill_step,
                    logs,
                    profile,
                    metrics_key,
                    timeout=max(30, math.ceil(remaining)),
                )
                profile.mark("shrink_resume")
                print(
                    f"[multinode-preempt] survivors training at reduced world size "
                    f"~{time.monotonic() - kill_t:.0f}s after the kill",
                    file=sys.stderr,
                )
                # (b) replacement joined: a post-kill step line reports the
                # pre-kill world size at a step past the pre-kill floor.
                _wait_full_world(
                    cfg,
                    logs,
                    profile,
                    metrics_key,
                    full_ws=full_ws,
                    step_floor=step_floor,
                    timeout=cfg.recovery_timeout_seconds,
                )
                profile.mark("full_world")
                print(
                    "[multinode-preempt] replacement joined; group back to full world",
                    file=sys.stderr,
                )
            except TimeoutError as e:
                print(
                    f"[multinode-preempt] {e} — falling back to whole-group restart",
                    file=sys.stderr,
                )
                _fallback_restart()
            profile.mark("train_start")
            t0 = time.monotonic()

        # Final stretch: run out the remaining budget, collect metrics.
        metrics = _stream_until_metrics_multi(cfg, run_id, profile, logs)
        profile.mark("metrics" if metrics is not None else "timeout")
        profile.from_metrics(metrics)
        for iid in live.values():
            aws.terminate(iid)
            profile.instance_stopped(iid)
        live.clear()

        profile.finalize(cfg)
        print(f"[multinode-preempt] profile: {cfg.run_profile_uri(run_id)}", file=sys.stderr)
        if metrics is not None:
            print(f"\n[multinode-preempt] metrics: {json.dumps(metrics, indent=2)}")
        return metrics
    finally:
        for iid in live.values():
            aws.terminate(iid)


def _wait_train_start(
    cfg: OrchestratorConfig,
    ckpt_prefix: str,
    base_step: int,
    logs: dict[int, dict],
    profile: RunProfile,
    metrics_key: str,
    timeout: int | None = None,
) -> None:
    """Block (while streaming every node's log) until a NEW checkpoint appears
    past ``base_step`` — training is underway — or metrics.json shows up (a very
    short final segment). ``timeout`` overrides the metrics timeout (the elastic
    watchdogs wait their own bounds instead)."""
    deadline = time.monotonic() + (timeout or cfg.metrics_timeout_seconds)
    while True:
        _pull_logs(cfg, logs, profile)
        if aws.max_checkpoint_step(cfg.bucket, ckpt_prefix) > base_step:
            return
        if aws.object_exists(cfg.bucket, metrics_key):
            return
        if time.monotonic() > deadline:
            raise TimeoutError("training never started (no new checkpoint appeared)")
        time.sleep(cfg.log_stream_seconds)


def _wait_full_world(
    cfg: OrchestratorConfig,
    logs: dict[int, dict],
    profile: RunProfile,
    metrics_key: str,
    *,
    full_ws: int | None,
    step_floor: int,
    timeout: int,
) -> None:
    """Block until a step line reports the pre-kill world size again at a step
    PAST the pre-kill floor (both guards matter: a lag-parsed pre-kill line
    carries the old ws at an old step, and degraded-mode lines carry a smaller
    ws). ``full_ws`` None means the trainer never reported ws (pre-elastic log)
    — phase (a) already proved progress, so just note it and return."""
    if full_ws is None:
        print(
            "[multinode-preempt] no `ws` in step lines before the kill — skipping the "
            "full-world check (old trainer?)",
            file=sys.stderr,
        )
        return
    deadline = time.monotonic() + timeout
    while True:
        _pull_logs(cfg, logs, profile)
        if any(s.world_size == full_ws and s.step > step_floor for s in profile.samples):
            return
        if aws.object_exists(cfg.bucket, metrics_key):
            return  # run finished while degraded — nothing left to verify
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"world size never returned to {full_ws} within {timeout}s of the kill"
            )
        time.sleep(cfg.log_stream_seconds)


def _preempt_instance(cfg: OrchestratorConfig, iid: str, ckpt_prefix: str) -> None:
    """Deliver a Spot-style shutdown: SIGTERM the trainer via SSM so its interruption
    handler checkpoints and exits, wait for that checkpoint to land, then terminate
    the box. Falls back to a hard terminate if the SSM agent isn't reachable."""
    before = aws.max_checkpoint_step(cfg.bucket, ckpt_prefix)
    if aws.ssm_online(iid):
        aws.ssm_send(iid, ["pkill -TERM -f spot_train.train || true"])
        deadline = time.monotonic() + cfg.preempt_grace_seconds
        saved = False
        while time.monotonic() < deadline:
            if aws.max_checkpoint_step(cfg.bucket, ckpt_prefix) > before:
                print("[preempt] graceful checkpoint saved", file=sys.stderr)
                saved = True
                break
            time.sleep(cfg.log_stream_seconds)
        if not saved:
            print("[preempt] grace window elapsed; terminating anyway", file=sys.stderr)
    else:
        print(
            "[preempt] SSM agent not online; hard-terminating "
            "(lost work bounded by the checkpoint interval)",
            file=sys.stderr,
        )
    aws.terminate(iid)


def run_preempt(cfg: OrchestratorConfig, *, ddp: bool = False) -> dict | None:
    """Orchestrator-driven preemption: play the role of AWS Spot by killing the
    training instance at a FIXED interval the node isn't told about, accumulating
    ``train_total_seconds`` of TRAINING across segments. Each kill is unrecoverable —
    a FRESH instance is provisioned that resumes from the S3 checkpoint.

    ``ddp=True`` runs each segment under torchrun (single-node DDP), one rank per
    GPU on the box (or ``ddp_nproc_per_node`` if forced): a preemption kills the
    whole box (all ranks together), and the replacement box brings up a fresh
    N-rank group that resumes from rank-0's S3 checkpoint. The trainer's coordinated
    stop + all-ranks-resume make this deadlock-free."""
    ami, sg_id = _prepare(cfg)
    nproc_per_node = cfg.ddp_nproc_per_node if ddp else 1
    kind = "ddp-preempt" if ddp else "preempt"
    run_id = _run_id(kind)
    total = cfg.train_total_seconds
    # Train-before-kill time: PREEMPT_AFTER if set (fast kills for debugging),
    # otherwise split the total evenly across (preempt_count + 1) segments, so
    # preempt_count=1 => train `interval`, kill once, reboot, finish the rest.
    interval = cfg.preempt_after_seconds or math.ceil(total / (cfg.preempt_count + 1))
    ckpt_prefix = f"{cfg.run_prefix}/{run_id}/checkpoints/"
    metrics_key = cfg.run_metrics_key(run_id)
    # Dense checkpoints so training-start is detectable quickly; graceful SIGTERM
    # also checkpoints, so lost work is ~0 regardless of the interval.
    cfg.checkpoint_interval_seconds = min(
        cfg.checkpoint_interval_seconds, cfg.preempt_checkpoint_seconds
    )
    # Keep the checkpoint verify+smoke cadence ~30s (like baseline) so the frequent
    # preemption checkpoints don't flood the streamed loss lines.
    cfg.smoke_test_every = max(1, round(30 / cfg.checkpoint_interval_seconds))
    ddp_note = f" nproc_per_node={nproc_per_node or 'auto(gpu)'}" if ddp else ""
    print(
        f"[{kind}] run_id={run_id} total_train={total}s preemptions={cfg.preempt_count} "
        f"interval={interval}s ckpt_every={cfg.checkpoint_interval_seconds}s{ddp_note} — fresh "
        f"instance per segment, node not told the schedule",
        file=sys.stderr,
    )

    profile = RunProfile(run_id, kind=kind, market=cfg.spot_market)
    if not aws.is_dry_run():  # dry-run must not create a real W&B run
        profile.wandb_start(cfg)

    accumulated = 0.0
    seg = 1
    metrics: dict | None = None
    iid: str | None = None
    try:
        while True:
            remaining = total - accumulated
            budget = max(1, math.ceil(remaining))
            # Final once all preemptions are spent (keeps the kill count at
            # preempt_count even when PREEMPT_AFTER shrinks the interval) or when
            # the remaining budget wouldn't outlast another interval anyway.
            final = seg > cfg.preempt_count or remaining <= interval
            logs_key = cfg.run_logs_key(run_id, segment=seg)
            profile.segment = seg
            print(
                f"[preempt] segment {seg}: launch "
                f"({'final' if final else 'will preempt'}), MAX_SECONDS={budget}, "
                f"~{accumulated:.0f}/{total}s trained so far",
                file=sys.stderr,
            )
            iid = _launch(
                cfg,
                ami,
                sg_id,
                run_id,
                cfg.spot_market,
                budget,
                logs_key=logs_key,
                ddp=ddp,
                nproc_per_node=nproc_per_node,
                profile=profile,
            )
            profile.mark("launch" if seg == 1 else "relaunch")

            if aws.is_dry_run():
                print("[preempt] dry-run: not waiting/killing", file=sys.stderr)
                aws.terminate(iid)
                iid = None
                if final:
                    return None
                accumulated += interval
                seg += 1
                continue

            base_step = aws.max_checkpoint_step(cfg.bucket, ckpt_prefix)
            state = {"printed": 0}
            _wait_train_start(cfg, ckpt_prefix, base_step, logs_key, profile, state, metrics_key)
            profile.mark("train_start")
            t_start = time.monotonic()

            if final:
                # Let this segment finish its remaining budget and write metrics.
                metrics = _stream_until_metrics(cfg, run_id, profile, logs_key=logs_key)
                profile.mark("metrics" if metrics is not None else "timeout")
                profile.from_metrics(metrics)
                done, iid = iid, None
                profile.instance_stopped(done)
                aws.terminate(done)
                break

            # Non-final: let it train one hidden interval, then send the Spot signal.
            while time.monotonic() - t_start < interval:
                _pull_log(cfg, logs_key, profile, state)
                time.sleep(cfg.log_stream_seconds)
            trained = time.monotonic() - t_start
            print(
                f"[preempt] segment {seg}: PREEMPT after ~{trained:.0f}s training",
                file=sys.stderr,
            )
            _preempt_instance(cfg, iid, ckpt_prefix)
            profile.mark("kill")
            profile.instance_stopped(iid)
            iid = None
            accumulated += trained
            seg += 1

        profile.finalize(cfg)
        print(f"[preempt] profile: {cfg.run_profile_uri(run_id)}", file=sys.stderr)
        if metrics is not None:
            print(f"\n[preempt] metrics: {json.dumps(metrics, indent=2)}")
        return metrics
    finally:
        if iid is not None:
            aws.terminate(iid)


def run_spot(cfg: OrchestratorConfig) -> dict | None:
    ami, sg_id = _prepare(cfg)
    run_id = _run_id("spot")
    ckpt_prefix = f"{cfg.run_prefix}/{run_id}/checkpoints/"
    print(
        f"[spot] run_id={run_id} seg1={cfg.spot_seg1_seconds}s seg2={cfg.spot_seg2_seconds}s",
        file=sys.stderr,
    )

    # --- segment 1: train, then kill mid-run ------------------------------ #
    iid1 = _launch(cfg, ami, sg_id, run_id, cfg.spot_market, _SEG1_TRAINER_BUDGET)
    try:
        if aws.is_dry_run():
            print("[spot] dry-run: skipping wait/kill/resume", file=sys.stderr)
            return None
        print("[spot] waiting for first checkpoint (training underway)...", file=sys.stderr)
        deadline = time.monotonic() + cfg.metrics_timeout_seconds
        while not aws.any_object_under(cfg.bucket, ckpt_prefix):
            if time.monotonic() > deadline:
                raise TimeoutError("no checkpoint appeared; training never started")
            time.sleep(cfg.metrics_poll_seconds)
        print(
            f"[spot] checkpoint seen; training {cfg.spot_seg1_seconds}s more, then KILL",
            file=sys.stderr,
        )
        time.sleep(cfg.spot_seg1_seconds)
    finally:
        aws.terminate(iid1)  # the "preemption"
    print("[spot] segment 1 instance terminated (simulated preemption)", file=sys.stderr)

    # --- segment 2: resume from S3 and finish ----------------------------- #
    iid2 = _launch(cfg, ami, sg_id, run_id, cfg.spot_market, cfg.spot_seg2_seconds)
    try:
        metrics = _poll_metrics(cfg, run_id)
        print(f"[spot] metrics: {json.dumps(metrics, indent=2)}")
        if not metrics.get("resumed"):
            print("[spot] WARNING: segment 2 did not report resumed=true", file=sys.stderr)
        return metrics
    finally:
        aws.terminate(iid2)
