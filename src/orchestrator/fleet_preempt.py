"""`spot-orchestrate fleet preempt` — the serving preemption experiment.

The serving analogue of the training preempt runs: put the fleet under
moderate-heavy load, hard-kill one worker mid-test (TerminateInstances on
cloud, SIGKILL locally), and prove requests reroute seamlessly. Output is a
latency report split around the kill:

  phase        window        reqs   ok  err    p50ms    p99ms
  before       [0,60s)        ...
  disruption   [60,75s)       ...   <- kill to heartbeat-TTL expiry
  recovered    [75s,end)      ...

Success = zero client-visible errors, rerouted > 0 on the router, and p99
back inside the pre-kill band within the disruption window. The raw loadgen
report + verdict are saved as a JSON artifact.

Load is auto-calibrated: a few warmup requests measure per-request latency,
and the offered rate is set to ~70% of measured fleet capacity so the test is
"moderate-heavy" on any hardware (CPU or GPU workers) without tuning flags.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import time

from . import fleet
from .config import OrchestratorConfig

DEFAULT_TTL_SECONDS = 15.0  # keep in sync with inference.registry.DEFAULT_TTL_SECONDS
MAX_TOKENS = 32


# --------------------------------------------------------------------------- #
# Analysis (pure — unit-tested)
# --------------------------------------------------------------------------- #
def analyze(report: dict, kill_rel: float, ttl: float = DEFAULT_TTL_SECONDS) -> dict:
    """Split the loadgen per-second series into before/disruption/recovered
    windows around ``kill_rel`` (seconds since loadgen start) and compute
    per-window stats + recovery time."""
    buckets = report.get("per_second", [])
    windows = {
        "before": [b for b in buckets if b["t"] < kill_rel],
        "disruption": [b for b in buckets if kill_rel <= b["t"] < kill_rel + ttl],
        "recovered": [b for b in buckets if b["t"] >= kill_rel + ttl],
    }

    def stats(bs: list[dict]) -> dict:
        oks = [b for b in bs if b["ok"]]
        return {
            "requests": sum(b["sent"] for b in bs),
            "ok": sum(b["ok"] for b in bs),
            "errors": sum(b["errors"] for b in bs),
            "p50_ms": round(statistics.median(b["mean_ms"] for b in oks), 1) if oks else 0.0,
            "p99_ms": round(max(b["p99_ms"] for b in oks), 1) if oks else 0.0,
        }

    out = {name: stats(bs) for name, bs in windows.items()}

    # Recovery: first post-kill second whose p99 is back inside 2x the pre-kill
    # median p99 (and error-free), sustained for 3 consecutive seconds.
    pre_p99s = [b["p99_ms"] for b in windows["before"] if b["ok"]]
    baseline = statistics.median(pre_p99s) if pre_p99s else 0.0
    recovery_s = None
    post = [b for b in buckets if b["t"] >= kill_rel]
    for i, b in enumerate(post):
        window = post[i : i + 3]
        if len(window) == 3 and all(
            w["errors"] == 0 and w["ok"] > 0 and w["p99_ms"] <= max(2 * baseline, 1.0)
            for w in window
        ):
            recovery_s = round(b["t"] - kill_rel, 1)
            break

    out["kill_rel_s"] = round(kill_rel, 1)
    out["baseline_p99_ms"] = round(baseline, 1)
    out["recovery_s"] = recovery_s
    out["total_failed"] = report.get("failed", 0)
    out["total_dropped"] = report.get("dropped", 0)
    out["passed"] = bool(
        report.get("failed", 0) == 0 and recovery_s is not None and recovery_s <= ttl + 10
    )
    return out


def render(analysis: dict, rerouted: int) -> str:
    kill = analysis["kill_rel_s"]
    ttl_end = kill + DEFAULT_TTL_SECONDS
    rows = [
        ("before", f"[0,{kill:.0f}s)", analysis["before"]),
        ("disruption", f"[{kill:.0f},{ttl_end:.0f}s)", analysis["disruption"]),
        ("recovered", f"[{ttl_end:.0f}s,end)", analysis["recovered"]),
    ]
    lines = [
        "=== fleet preempt report ===",
        f"{'phase':<12} {'window':<12} {'reqs':>6} {'ok':>6} {'err':>5} {'p50ms':>8} {'p99ms':>8}",
    ]
    for name, window, s in rows:
        lines.append(
            f"{name:<12} {window:<12} {s['requests']:>6} {s['ok']:>6} {s['errors']:>5} "
            f"{s['p50_ms']:>8.1f} {s['p99_ms']:>8.1f}"
        )
    rec = analysis["recovery_s"]
    rec_text = f"{rec:.1f}s after kill" if rec is not None else "NOT DETECTED"
    lines.append(
        f"kill at t={kill:.0f}s | rerouted {rerouted} | failed {analysis['total_failed']} | "
        f"recovery {rec_text}"
    )
    lines.append(f"RESULT: {'PASS' if analysis['passed'] else 'FAIL'}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Experiment driver
# --------------------------------------------------------------------------- #
def _loadgen_binary() -> str:
    """Build the Go loadgen once into .fleet/ and return the path."""
    if shutil.which("go") is None:
        raise SystemExit("fleet preempt needs the Go toolchain on PATH (builds loadgen/)")
    out = os.path.abspath(".fleet/loadgen-bin")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run(["go", "build", "-o", out, "."], cwd="loadgen", check=True)
    return out


def _router_status(url: str) -> dict:
    import requests

    try:
        return requests.get(f"{url}/fleet/status", timeout=5).json()
    except requests.RequestException:
        return {}


def _calibrate(url: str, live_workers: int) -> float:
    """Offered rps = 70% of CONCURRENT throughput, measured end-to-end.

    Sequential probes overestimate badly when workers contend for shared cores
    (the local fleet) — one request at a time hides the contention that real
    load creates. So: warm up, then push ``2x live_workers`` requests with
    ``live_workers`` in flight and take measured completions/second as
    capacity."""
    from concurrent.futures import ThreadPoolExecutor

    import requests

    def one() -> None:
        requests.post(
            f"{url}/v1/completions",
            json={"prompt": "ROMEO:", "max_tokens": MAX_TOKENS},
            timeout=60,
        ).raise_for_status()

    # Warm EVERY worker before timing: round-robin spreads sequential requests
    # across the fleet, and each worker's first generate pays one-time init
    # (CUDA context, kernel autotune). 2x live makes it robust to a retry
    # skewing the rotation. Without this, cold workers deflate the measured
    # capacity and the "moderate-heavy" load ends up light (observed: a T4
    # fleet calibrated at ~2 rps vs ~12 rps actual).
    live = max(live_workers, 1)
    for _ in range(2 * live):
        one()
    n = 4 * live
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=live) as pool:
        for f in [pool.submit(one) for _ in range(n)]:
            f.result()
    elapsed = max(time.monotonic() - t0, 1e-3)
    capacity = n / elapsed
    rps = max(round(0.7 * capacity, 1), 0.5)
    print(
        f"[preempt] calibration: {n} concurrent reqs in {elapsed:.1f}s (after {2 * live} warmup) "
        f"=> capacity ~{capacity:.1f} rps => offering {rps} rps (70%)"
    )
    return rps


def run_fleet_preempt(
    cfg: OrchestratorConfig,
    *,
    local: bool,
    run_id: str = "",
    workers: int | None = None,
    duration: int = 150,
    kill_after: int = 60,
    rps: float = 0.0,
    keep: bool = False,
) -> None:
    from . import aws

    # --- ensure a fleet (reuse a running one; boot otherwise) ---------------
    booted = False
    if local:
        url = fleet.router_url_local()
        if url is None:
            fleet.up_local(workers=workers or 2)
            url = fleet.router_url_local()
            booted = True
    else:
        url = fleet.router_url_cloud(cfg)
        if url is None:
            fleet.up_cloud(cfg, workers=workers or cfg.fleet_worker_count, run_id=run_id)
            url = fleet.router_url_cloud(cfg)
            booted = True
    if url is None or aws.is_dry_run():
        if aws.is_dry_run():
            print("[preempt] dry-run: fleet plan shown above; experiment not executed")
            return
        raise SystemExit("[preempt] no router endpoint — fleet boot failed")

    try:
        status = _router_status(url)
        live = status.get("live_workers", 0)
        if live < 2:
            raise SystemExit(f"[preempt] need >= 2 live workers to reroute (router sees {live})")
        rerouted_before = status.get("rerouted", 0)
        offered = rps or _calibrate(url, live)

        # --- loadgen (subprocess) + kill at T --------------------------------
        binary = _loadgen_binary()
        report_path = os.path.abspath(".fleet/preempt-report.json")
        cmd = [
            binary,
            "-url",
            url,
            "-rps",
            str(offered),
            "-duration",
            f"{duration}s",
            "-max-tokens",
            str(MAX_TOKENS),
            "-timeout",
            "30s",
            "-out",
            report_path,
        ]
        print(f"[preempt] load: {offered} rps for {duration}s; kill at t={kill_after}s")
        proc = subprocess.Popen(cmd)
        time.sleep(kill_after)
        if local:
            fleet.kill_worker_local()
        else:
            fleet.kill_worker_cloud(cfg)
        kill_unix = time.time()
        proc.wait(timeout=duration + 120)

        # --- analyze ----------------------------------------------------------
        with open(report_path) as f:
            report = json.load(f)
        kill_rel = kill_unix - report["start_unix"]
        analysis = analyze(report, kill_rel)
        rerouted = _router_status(url).get("rerouted", 0) - rerouted_before
        print()
        print(render(analysis, rerouted))

        artifact = {
            "kind": "fleet-preempt",
            "url": url,
            "offered_rps": offered,
            "duration_s": duration,
            "rerouted": rerouted,
            "analysis": analysis,
            "loadgen": report,
        }
        out_path = os.path.abspath(".fleet/preempt-analysis.json")
        with open(out_path, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"[preempt] artifact: {out_path}")
        if not local and cfg.bucket:
            key = f"fleet/preempt-{time.strftime('%Y%m%d-%H%M%S')}.json"
            aws.put_text(cfg.bucket, key, json.dumps(artifact))
            print(f"[preempt] artifact: s3://{cfg.bucket}/{key}")
    finally:
        if booted and not keep:
            print("[preempt] tearing down the fleet this experiment booted (--keep to skip)")
            if local:
                fleet.down_local()
            else:
                fleet.down_cloud(cfg)
        elif not booted:
            print("[preempt] fleet left up (it was running before the experiment)")
