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
    iid = aws.launch(
        ami_id=ami,
        instance_type=cfg.instance_type,
        profile_name=cfg.instance_profile,
        security_group_id=sg_id,
        user_data=ud,
        market=market,
        run_id=run_id,
        key_name=cfg.key_name,
    )
    aws.wait_running(iid)
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


def _stream_until_metrics(
    cfg: OrchestratorConfig, run_id: str, profile: RunProfile, logs_key: str | None = None
) -> dict | None:
    """Stream the box log until ``metrics.json`` appears (trainer writes it last =>
    done) or ``metrics_timeout_seconds`` elapses. Returns parsed metrics, or None."""
    logs_key = logs_key or cfg.run_logs_key(run_id)
    metrics_key = cfg.run_metrics_key(run_id)
    state = {"printed": 0}
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
    profile.wandb_start(cfg)
    iid = _launch(cfg, ami, sg_id, run_id, market, budget, ddp=ddp, nproc_per_node=nproc_per_node)
    profile.mark("launch")
    try:
        if aws.is_dry_run():
            print(f"[{kind}] dry-run: skipping stream/terminate", file=sys.stderr)
            return None
        metrics = _stream_until_metrics(cfg, run_id, profile)
        profile.mark("metrics" if metrics is not None else "timeout")
        profile.from_metrics(metrics)
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


def _wait_train_start(
    cfg: OrchestratorConfig,
    ckpt_prefix: str,
    base_step: int,
    logs_key: str,
    profile: RunProfile,
    state: dict,
    metrics_key: str,
) -> None:
    """Block (while streaming the log) until a NEW checkpoint appears past
    ``base_step`` — training is underway on this instance — or metrics.json shows up
    (a very short final segment)."""
    deadline = time.monotonic() + cfg.metrics_timeout_seconds
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
    # Split the total training evenly across (preempt_count + 1) segments, so
    # preempt_count=1 => train `interval`, kill once, reboot, finish `interval`.
    interval = math.ceil(total / (cfg.preempt_count + 1))
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

    profile = RunProfile(run_id, kind=kind, market="spot")
    profile.wandb_start(cfg)

    accumulated = 0.0
    seg = 1
    metrics: dict | None = None
    iid: str | None = None
    try:
        while True:
            remaining = total - accumulated
            budget = max(1, math.ceil(remaining))
            final = remaining <= interval
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
                "spot",
                budget,
                logs_key=logs_key,
                ddp=ddp,
                nproc_per_node=nproc_per_node,
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
    iid1 = _launch(cfg, ami, sg_id, run_id, "spot", _SEG1_TRAINER_BUDGET)
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
    iid2 = _launch(cfg, ami, sg_id, run_id, "spot", cfg.spot_seg2_seconds)
    try:
        metrics = _poll_metrics(cfg, run_id)
        print(f"[spot] metrics: {json.dumps(metrics, indent=2)}")
        if not metrics.get("resumed"):
            print("[spot] WARNING: segment 2 did not report resumed=true", file=sys.stderr)
        return metrics
    finally:
        aws.terminate(iid2)
