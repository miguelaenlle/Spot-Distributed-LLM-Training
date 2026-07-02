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
import sys
import time

from . import aws, bootstrap
from .config import OrchestratorConfig

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
    cfg: OrchestratorConfig, ami: str, sg_id: str, run_id: str, market: str, budget: int
) -> str:
    ud = bootstrap.build_user_data(cfg, run_id=run_id, market=market, max_seconds=budget)
    iid = aws.launch(
        ami_id=ami,
        instance_type=cfg.instance_type,
        profile_name=cfg.instance_profile,
        security_group_id=sg_id,
        user_data=ud,
        market=market,
        run_id=run_id,
    )
    aws.wait_running(iid)
    return iid


def _prepare(cfg: OrchestratorConfig) -> tuple[str, str]:
    cfg.require_bucket()
    aws.set_region(cfg.region)
    if not aws.is_dry_run() and not aws.object_exists(
        cfg.bucket, f"{cfg.data_prefix}/{cfg.dataset}/meta.pkl"
    ):
        raise SystemExit(
            f"dataset not staged at {cfg.data_uri()} — run `spot-orchestrate stage-data` first"
        )
    ami = aws.resolve_ami(cfg.ami_id, cfg.ami_ssm_param)
    sg_id = aws.ensure_security_group(cfg.security_group, cfg.region)
    return ami, sg_id


def run_baseline(cfg: OrchestratorConfig) -> dict | None:
    ami, sg_id = _prepare(cfg)
    run_id = _run_id("baseline")
    print(f"[baseline] run_id={run_id} budget={cfg.baseline_seconds}s", file=sys.stderr)
    iid = _launch(cfg, ami, sg_id, run_id, "on-demand", cfg.baseline_seconds)
    try:
        if aws.is_dry_run():
            print("[baseline] dry-run: skipping metrics poll", file=sys.stderr)
            return None
        metrics = _poll_metrics(cfg, run_id)
        print(f"[baseline] metrics: {json.dumps(metrics, indent=2)}")
        return metrics
    finally:
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
