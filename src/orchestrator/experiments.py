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

import contextlib
import json
import math
import os
import sys
import time

from . import aws, bootstrap
from .config import OrchestratorConfig
from .profile import RunProfile

# seg1's trainer must not self-stop before we kill it — give it a huge budget.
_SEG1_TRAINER_BUDGET = 24 * 3600


def _run_id(kind: str) -> str:
    return f"{kind}-{int(time.time())}"


def _logs_hint(run_id: str) -> None:
    """Point at the live per-node dashboard — attachable from another terminal
    the moment the run exists (and after it finishes)."""
    print(f"[logs] dashboard:  spot-orchestrate logs {run_id}", file=sys.stderr)


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
    _logs_hint(run_id)
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
    _logs_hint(run_id)
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


def _make_launch_node(cfg, ami, sg_id, run_id, market, budget, profile, logs):
    """Return ``launch(node_index) -> instance_id`` for the supervisor: allocate
    a fresh (attempt-suffixed) log key so a replacement never clobbers the dead
    node's log, launch behind the vCPU gate, wait running, open a cost row."""
    attempts: dict[int, int] = {}

    def launch(node_index: int) -> str:
        attempts[node_index] = attempts.get(node_index, 0) + 1
        key = cfg.run_logs_key(run_id, node=node_index, attempt=attempts[node_index] - 1)
        logs[node_index] = {
            "key": key,
            "attempt": attempts[node_index] - 1,
            "state": {"printed": 0},
        }
        iid = _launch_node(
            cfg, ami, sg_id, run_id, market, budget, node_index=node_index, logs_key=key
        )
        aws.wait_running(iid)
        _record_instance(cfg, profile, iid, market)
        return iid

    return launch


def _run_supervised(
    cfg: OrchestratorConfig,
    *,
    kind: str,
    budget: int,
    replace_on_loss: bool,
    kill_schedule: list[tuple[float, int]],
    verdict: bool = False,
    return_profile: bool = False,
):
    """Shared driver for every multi-node experiment: launch N boxes, hand
    membership to the epoch :class:`~orchestrator.supervisor.Supervisor`, and let
    it drive to metrics.json. ``multinode`` passes no kills; ``multinode-shrink``
    one kill with ``replace_on_loss=False`` (+ a PASS/FAIL verdict);
    ``multinode-preempt`` a schedule with ``replace_on_loss=True``. All three
    share one code path, so the W&B world-size staircase / degraded phase / cost
    ledger behave identically across them."""
    from .supervisor import Policy, Supervisor

    if cfg.node_count < 2:
        raise SystemExit(f"{kind} needs NODES >= 2")
    ami, sg_id = _prepare(cfg)
    run_id = _run_id(kind)
    market = cfg.spot_market
    if kill_schedule:
        # Dense checkpoints: a hard kill gives no warning, so lost work (and the
        # time to observe a shrink resume) is bounded by this interval.
        cfg.checkpoint_interval_seconds = min(
            cfg.checkpoint_interval_seconds, cfg.preempt_checkpoint_seconds
        )
        cfg.smoke_test_every = max(1, round(30 / cfg.checkpoint_interval_seconds))
    print(
        f"[{kind}] run_id={run_id} nodes={cfg.node_count} budget={budget}s market={market} "
        f"kills={kill_schedule} replace={replace_on_loss}",
        file=sys.stderr,
    )
    _logs_hint(run_id)
    profile = RunProfile(run_id, kind=kind, market=market)
    if not aws.is_dry_run():
        profile.wandb_start(cfg)

    node_ids: dict[int, str] = {}
    logs: dict[int, dict] = {}
    launch_node = _make_launch_node(cfg, ami, sg_id, run_id, market, budget, profile, logs)

    try:
        aws.wait_vcpu_headroom(cfg.node_count * cfg.instance_vcpu_count(), cfg.vcpu_quota)
        for i in range(cfg.node_count):
            node_ids[i] = launch_node(i)
        profile.mark("launch")

        if aws.is_dry_run():
            # Walk the kill/replace control flow minus the waiting.
            for _secs, victim in kill_schedule:
                aws.terminate(node_ids[victim])
                if replace_on_loss:
                    aws.wait_quota_released(node_ids[victim])
                    node_ids[victim] = launch_node(victim)
            print(f"[{kind}] dry-run: skipping supervision", file=sys.stderr)
            return (profile, None) if return_profile else None

        policy = Policy(
            replace_on_loss=replace_on_loss,
            recovery_timeout_s=cfg.recovery_timeout_seconds,
        )
        sup = Supervisor(
            cfg,
            profile,
            run_id=run_id,
            policy=policy,
            node_ids=node_ids,
            logs=logs,
            launch_node=launch_node,
            pull_logs=lambda: _pull_logs(cfg, logs, profile),
            kill_schedule=kill_schedule,
        )
        metrics = sup.run(deadline_s=cfg.metrics_timeout_seconds)
        profile.mark("metrics" if metrics is not None else "timeout")
        profile.from_metrics(metrics)
        if metrics is not None:
            _pull_samples(cfg, run_id, profile)

        if verdict:
            _shrink_verdict(cfg, run_id, profile, sup, metrics)

        profile.finalize(cfg)
        print(f"[{kind}] profile: {cfg.run_profile_uri(run_id)}", file=sys.stderr)
        if metrics is not None:
            print(f"\n[{kind}] metrics: {json.dumps(metrics, indent=2)}")
        return (profile, metrics) if return_profile else metrics
    finally:
        for iid in node_ids.values():
            aws.terminate(iid)


def run_multinode(cfg: OrchestratorConfig) -> dict | None:
    """N nodes x one-rank-per-GPU DDP under the epoch supervisor, run to the
    wall-clock budget with no kills. Proves a clean multi-node run end to end."""
    return _run_supervised(
        cfg,
        kind="multinode",
        budget=cfg.baseline_seconds,
        replace_on_loss=False,
        kill_schedule=[],
    )


def run_multinode_shrink(cfg: OrchestratorConfig) -> dict | None:
    """The minimal elastic validation: ONE kill, NO replacement. The supervisor
    publishes a shrink epoch; survivors must re-form at N-1 and finish the run on
    their own. Ends with an explicit PASS/FAIL verdict (survivors checkpointed
    again, step lines at the shrunken world size, metrics with that world size +
    resumed). The victim is the last node — with no rendezvous store on any box,
    that's no longer special, but it keeps the experiment simple."""
    total = cfg.train_total_seconds
    kill_after = cfg.preempt_after_seconds or 120
    return _run_supervised(
        cfg,
        kind="multinode-shrink",
        budget=max(1, math.ceil(total)),
        replace_on_loss=False,
        kill_schedule=[(kill_after, cfg.node_count - 1)],
        verdict=True,
    )


def run_multinode_preempt(cfg: OrchestratorConfig) -> dict | None:
    """Multi-node preemption under the epoch supervisor: a schedule of hard kills,
    each followed by a replacement that rejoins at the next epoch. Same profile
    marks as before (kill -> shrink_resume -> relaunch -> full_world), so the W&B
    world-size staircase, degraded phase, and goodput carry over unchanged."""
    victims = cfg.preempt_victim_schedule()
    total = cfg.train_total_seconds
    interval = cfg.preempt_after_seconds or math.ceil(total / (cfg.preempt_count + 1))
    # Kill i fires `interval` seconds after the PREVIOUS one resumed; the
    # supervisor clock is seconds-since-train-start, so space them by interval.
    schedule = [((k + 1) * interval, victims[k]) for k in range(cfg.preempt_count)]
    return _run_supervised(
        cfg,
        kind="multinode-preempt",
        budget=max(1, math.ceil(total)),
        replace_on_loss=True,
        kill_schedule=schedule,
    )


# --------------------------------------------------------------------------- #
# Overfit scaling experiment: does adding nodes reduce time-to-overfit?
# --------------------------------------------------------------------------- #
def _analyze_overfit(profile: RunProfile) -> dict:
    """Find the overfit point rigorously — the step of MINIMUM validation loss
    (overfitting = val loss bottoms out then rises while train loss keeps
    falling), and the wall-clock time to reach it FROM the first training step
    (so the ~constant boot is excluded, but preemption downtime/re-compute is
    included). ``reached`` is False if the run didn't clearly turn up (a real
    interior minimum) — then the comparison is inconclusive, not a bogus argmin
    at the last step."""
    vals = sorted(profile.val_samples, key=lambda v: v.step)
    steps = sorted(profile.samples, key=lambda s: s.t_rel)
    if len(vals) < 3 or not steps:
        return {"reached": False, "why": "too few eval / step samples"}
    best = min(vals, key=lambda v: v.loss)
    last = vals[-1]
    reached = best.step < last.step and last.loss > best.loss + 1e-3  # a rise after the min

    first_train = steps[0].t_rel  # wall of the first training step (seconds since launch)
    overfit_wall = next((s.t_rel for s in steps if s.step >= best.step), steps[-1].t_rel)
    return {
        "reached": reached,
        "overfit_step": best.step,
        "best_val": round(best.loss, 4),
        "final_val": round(last.loss, 4),
        "last_step": last.step,
        "steps_to_overfit": best.step,  # ~equal across node counts validates the control
        "time_to_overfit_s": round(overfit_wall - first_train, 1),
        "total_train_s": round(steps[-1].t_rel - first_train, 1),
    }


def _fetch_run_events(cfg: OrchestratorConfig, run_id: str) -> list[dict]:
    """All ``[event]`` records for a finished run, parsed from its S3 logs."""
    from . import logview

    prefix = cfg.run_logs_prefix(run_id)
    items = []
    for key in aws.list_keys(cfg.bucket, prefix):
        name = key.rsplit("/", 1)[-1]
        with contextlib.suppress(Exception):
            items.append((name, aws.get_text(cfg.bucket, key)))
    return logview.parse_run_events(items)


def _render_run_timeline(cfg: OrchestratorConfig, run_id: str, out_dir: str) -> dict:
    """Export the event-sourced Gantt PNG + events.txt for a finished run."""
    from .logview import TimelineRecorder, export_gantt

    records = _fetch_run_events(cfg, run_id)
    if not records:
        return {"png": None, "events": None}
    now = max(r["ts"] for r in records)
    rec = TimelineRecorder.from_events(records, now)
    where = export_gantt(rec, run_id, now, out_dir=out_dir, local_only=True, records=records)
    return {"png": where[0], "events": where[1] if len(where) > 1 else None}


def _val_curve_png(profile: RunProfile, analysis: dict, path: str) -> str | None:
    """Val-loss (and train-loss) vs step with the overfit point marked — makes
    the 'done' call auditable."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = sorted(profile.val_samples, key=lambda v: v.step)
    if not vals:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    tr = sorted(profile.samples, key=lambda s: s.step)
    if tr:
        ax.plot([s.step for s in tr], [s.loss for s in tr], color="#BAB0AC", lw=1, label="train")
    ax.plot([v.step for v in vals], [v.loss for v in vals], color="#4C78A8", lw=2, label="val")
    if analysis.get("overfit_step") is not None:
        ax.axvline(analysis["overfit_step"], ls="--", color="#E45756", lw=1)
        ax.plot(
            [analysis["overfit_step"]],
            [analysis["best_val"]],
            marker="*",
            color="#F2C800",
            ms=16,
            mec="#7a6000",
            label=f"overfit (val {analysis['best_val']})",
        )
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"{profile.run_id} — val-loss overfit point")
    ax.legend(fontsize=8)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return os.path.abspath(path)


def _write_overfit_report(path: str, results: list[dict], recipe: dict) -> None:
    """The human summary: recipe, per-run table, and the H1/H2 verdicts."""

    def t(label: str) -> float | None:
        r = next((x for x in results if x["label"] == label), None)
        return (
            r["analysis"].get("time_to_overfit_s") if r and r["analysis"].get("reached") else None
        )

    def verdict(name: str, faster: str, slower: str, claim: str) -> str:
        a, b = t(faster), t(slower)
        if a is None or b is None:
            return (
                f"{name}: INCONCLUSIVE — a run did not clearly overfit ({faster}={a}, {slower}={b})"
            )
        ok = a < b
        ratio = b / a if a else float("inf")
        return (
            f"{name}: {'TRUE' if ok else 'FALSE'} — {claim}\n"
            f"     {faster} = {a}s   vs   {slower} = {b}s   ({ratio:.2f}x "
            f"{'speedup' if ok else 'SLOWER'})"
        )

    lines = [
        f"Overfit scaling experiment — {recipe['stamp']}",
        "=" * 72,
        "",
        "Hypotheses (time-to-overfit = wall-clock from first training step to the",
        "step of MINIMUM validation loss; overfit = val loss bottoms out then rises):",
        "  H1: time_to_overfit(4 nodes) < time_to_overfit(2 nodes), no preemptions",
        "  H2: same, WITH preemptions",
        "",
        "Controls: identical model/data/seed, CONSTANT global batch "
        f"(GLOBAL_BATCH_SIZE={recipe['global_batch']}) so 2- and 4-node follow the",
        "same trajectory vs step -> the comparison isolates throughput. Sequential",
        f"runs on {recipe['market']}. MAX_STEPS={recipe['max_steps']}, "
        f"EVAL_INTERVAL_STEPS={recipe['eval_interval']}, DROPOUT={recipe['dropout']}.",
        f"Preemptions: 2 worker kills at t+{recipe['offsets']}s after train start.",
        "",
        "VERDICTS",
        "-" * 72,
        verdict("H1 (clean)", "4n-clean", "2n-clean", "more nodes overfit faster"),
        "",
        verdict("H2 (preempt)", "4n-preempt", "2n-preempt", "more nodes win despite preemption"),
        "",
        "PER-RUN",
        "-" * 72,
    ]
    for r in results:
        a = r["analysis"]
        if a.get("reached"):
            detail = (
                f"  step {a['overfit_step']}  best_val {a['best_val']}  "
                f"time_to_overfit {a['time_to_overfit_s']}s  "
                f"(total train {a['total_train_s']}s)"
            )
        else:
            detail = f"  ({a.get('why', 'val did not turn up — raise MAX_STEPS')})"
        lines += [
            f"[{r['label']}]  run_id={r['run_id']}  nodes={r['nodes']}  preempt={r['preempt']}",
            f"    overfit: {'YES' if a.get('reached') else 'NOT REACHED'}{detail}",
            f"    cost: ${r['cost']}    wandb: {r['wandb'] or '(disabled)'}",
            f"    gantt: {r['png']}    events: {r['events']}    valcurve: {r['valcurve']}",
            "",
        ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_overfit_experiment(cfg: OrchestratorConfig) -> list[dict]:
    """ONE command: 2- vs 4-node time-to-overfit, clean and preempted, on spot.
    Runs the four configs sequentially, determines each run's overfit point from
    its validation-loss curve, and compiles reports/overfit-experiment-<ts>/ with
    per-run Gantt + events + val-curve and the H1/H2 verdicts."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # --- fixed recipe (env-overridable) — constant global batch is the control ---
    recipe = {
        "max_steps": os.environ.setdefault("MAX_STEPS", "5000"),
        "global_batch": os.environ.setdefault("GLOBAL_BATCH_SIZE", "64"),
        "eval_interval": os.environ.setdefault("EVAL_INTERVAL_STEPS", "100"),
        "dropout": os.environ.setdefault("DROPOUT", "0.0"),  # overfit clearly: measure the point
        "market": "spot",
        "stamp": stamp,
    }
    for k, v in {
        "LEARNING_RATE": "1e-3",
        "LR_DECAY_STEPS": "5000",
        "MIN_LR": "1e-4",
        "WARMUP_STEPS": "100",
        "GRAD_CLIP": "1.0",
        "SAMPLE_INTERVAL_STEPS": "0",
    }.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("WANDB_GROUP", f"overfit-experiment-{stamp}")
    cfg.spot_market = "spot"
    cfg.batch_size = int(os.environ.get("BATCH_SIZE", "16"))  # per-rank micro; global stays 64
    budget = int(os.environ.get("OVERFIT_BUDGET", "3600"))  # generous: MAX_STEPS is the stop
    offsets = [float(x) for x in os.environ.get("PREEMPT_OFFSETS", "60,150").split(",")]
    recipe["offsets"] = ",".join(str(int(o)) for o in offsets)

    out_dir = os.path.abspath(f"reports/overfit-experiment-{stamp}")
    os.makedirs(f"{out_dir}/runs", exist_ok=True)
    print(
        "\n\033[1m⚠️  BILLABLE: this launches four SEQUENTIAL spot runs "
        "(2x2-node + 2x4-node). Peak 16 vCPUs.\033[0m\n"
        f"[overfit-experiment] recipe: {recipe}\n[overfit-experiment] report dir: {out_dir}",
        file=sys.stderr,
    )

    plan = [
        ("2n-clean", 2, []),
        ("4n-clean", 4, []),
        ("2n-preempt", 2, [(offsets[0], 1), (offsets[1], 1)]),  # kill worker node1 twice
        ("4n-preempt", 4, [(offsets[0], 3), (offsets[1], 3)]),  # kill worker node3 twice
    ]
    results: list[dict] = []
    for label, nodes, kills in plan:
        cfg.node_count = nodes
        print(
            f"\n[overfit-experiment] === {label} (nodes={nodes}, kills={kills}) ===",
            file=sys.stderr,
        )
        profile, _metrics = _run_supervised(
            cfg,
            kind="multinode-preempt" if kills else "multinode",
            budget=budget,
            replace_on_loss=bool(kills),
            kill_schedule=kills,
            return_profile=True,
        )
        analysis = _analyze_overfit(profile)
        art = _render_run_timeline(cfg, profile.run_id, f"{out_dir}/runs")
        valcurve = _val_curve_png(
            profile, analysis, f"{out_dir}/runs/{profile.run_id}-valcurve.png"
        )
        results.append(
            {
                "label": label,
                "nodes": nodes,
                "preempt": bool(kills),
                "run_id": profile.run_id,
                "analysis": analysis,
                "cost": round(profile.cost_now(), 4),
                "wandb": getattr(profile._wb, "url", None) if profile._wb else None,
                "png": art["png"],
                "events": art["events"],
                "valcurve": valcurve,
            }
        )

    _write_overfit_report(f"{out_dir}/summary.txt", results, recipe)
    print(f"\n\033[1m[overfit-experiment] DONE → {out_dir}/summary.txt\033[0m", file=sys.stderr)
    with open(f"{out_dir}/summary.txt") as f:
        print(f.read())
    return results


def _shrink_verdict(cfg, run_id, profile, sup, metrics) -> None:
    """Turn the observed run into the three PASS/FAIL checks the shrink
    experiment exists to answer, then print the verdict."""
    full_ws = sup.st.full_ws
    shrunk_ws = None
    if full_ws is not None and full_ws % cfg.node_count == 0:
        shrunk_ws = full_ws // cfg.node_count * (cfg.node_count - 1)
    marked = {e.event for e in profile.events}
    checks: list[tuple[str, bool, str]] = []

    resumed_mark = "shrink_resume" in marked
    checks.append(
        (
            "survivors checkpointing again",
            resumed_mark,
            "shrink_resume mark emitted" if resumed_mark else "no new checkpoint after the kill",
        )
    )
    ws_seen = shrunk_ws is not None and any(s.world_size == shrunk_ws for s in profile.samples)
    checks.append(
        (
            "step lines at shrunken world size",
            bool(ws_seen),
            f"ws {shrunk_ws} observed" if ws_seen else f"no ws {shrunk_ws} step line",
        )
    )
    m_ok = bool(
        metrics
        and (shrunk_ws is None or metrics.get("world_size") == shrunk_ws)
        and metrics.get("resumed")
    )
    checks.append(
        (
            "metrics.json from the shrunken group",
            m_ok,
            f"world_size={metrics.get('world_size') if metrics else None} "
            f"resumed={metrics.get('resumed') if metrics else None}",
        )
    )
    _print_shrink_verdict(cfg, run_id, checks)


def _print_shrink_verdict(
    cfg: OrchestratorConfig, run_id: str, checks: list[tuple[str, bool, str]]
) -> None:
    print("\n================ ELASTIC SHRINK VERDICT ================", file=sys.stderr)
    if not checks:
        print("  no checks ran (killed before training started?)", file=sys.stderr)
    for name, passed, detail in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name} — {detail}", file=sys.stderr)
    ok = bool(checks) and all(p for _, p, _ in checks)
    summary = "PASS — survivors kept training without the dead node" if ok else "FAIL"
    print(f"  VERDICT: {summary}", file=sys.stderr)
    if not ok:
        print(
            "  evidence: aws s3 cp "
            f"s3://{cfg.bucket}/{cfg.run_prefix}/{run_id}/logs/boot-node<N>.log - "
            "| grep -nE '\\[epoch\\]|torchrun|\\[resume\\]|Traceback'",
            file=sys.stderr,
        )
    print("========================================================\n", file=sys.stderr)


def _wait_train_start(
    cfg: OrchestratorConfig,
    ckpt_prefix: str,
    base_step: int,
    logs_key: str,
    profile: RunProfile,
    state: dict,
    metrics_key: str,
    timeout: int | None = None,
) -> None:
    """Single-box variant used by ``run_preempt``: block (streaming one box's log)
    until a NEW checkpoint appears past ``base_step`` or metrics.json shows up."""
    deadline = time.monotonic() + (timeout or cfg.metrics_timeout_seconds)
    while True:
        _pull_log(cfg, logs_key, profile, state)
        if aws.max_checkpoint_step(cfg.bucket, ckpt_prefix) > base_step:
            return
        if aws.object_exists(cfg.bucket, metrics_key):
            return
        if time.monotonic() > deadline:
            raise TimeoutError("training never started (no new checkpoint appeared)")
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
    _logs_hint(run_id)

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
    _logs_hint(run_id)

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
