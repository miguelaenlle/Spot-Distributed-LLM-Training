"""Live fleet monitor — `spot-orchestrate fleet monitor [--local] [--wandb]`.

Polls the router's public ``/fleet/metrics`` every few seconds and redraws a
terminal table: per-worker queue depth, in-flight, req/s, tok/s, and
utilization (fraction of the interval spent generating), plus router totals.
Rates are computed here from counter deltas, so the boxes stay dumb and a
worker that restarts (counters reset) just shows a zero-rate tick.

``--wandb`` mirrors every tick to Weights & Biases (orchestrator-side only,
same discipline as run profiles: the key never reaches the boxes).
"""

from __future__ import annotations

import sys
import time

from .config import OrchestratorConfig


def compute_rates(prev: dict | None, cur: dict) -> dict[str, dict]:
    """Per-worker rates from two /fleet/metrics samples, keyed by worker_id.

    Counter deltas are clamped at 0 so a restarted worker (or a worker id that
    just appeared) reads as idle for one tick instead of a huge negative rate.
    """
    rates: dict[str, dict] = {}
    dt = max(cur["ts"] - prev["ts"], 1e-6) if prev else 0.0
    prev_workers = {w["worker_id"]: w for w in (prev or {}).get("workers", [])}
    for w in cur.get("workers", []):
        wid = w["worker_id"]
        p = prev_workers.get(wid)
        if not prev or p is None or not w.get("ok") or not p.get("ok"):
            rates[wid] = {"rps": 0.0, "tok_s": 0.0, "util": 0.0}
            continue
        d_req = max(w.get("requests", 0) - p.get("requests", 0), 0)
        d_tok = max(w.get("completion_tokens", 0) - p.get("completion_tokens", 0), 0)
        d_gen = max(w.get("generate_seconds", 0.0) - p.get("generate_seconds", 0.0), 0.0)
        rates[wid] = {
            "rps": d_req / dt,
            "tok_s": d_tok / dt,
            "util": min(d_gen / dt, 1.0),
        }
    return rates


def flatten_metrics(cur: dict, rates: dict[str, dict]) -> dict[str, float]:
    """One flat dict per tick for W&B: router/* and workers/<id>/*."""
    r = cur["router"]
    out = {
        "router/live_workers": r["live_workers"],
        "router/in_flight": r["in_flight"],
        "router/requests_total": r["requests"],
        "router/rerouted_total": r["rerouted"],
        "router/failed_total": r["failed"],
        "fleet/queued_total": sum(w.get("queued", 0) for w in cur["workers"] if w.get("ok")),
    }
    for w in cur["workers"]:
        wid, rate = w["worker_id"], rates.get(w["worker_id"], {})
        out[f"workers/{wid}/ok"] = 1.0 if w.get("ok") else 0.0
        if w.get("ok"):
            out[f"workers/{wid}/queued"] = w.get("queued", 0)
            out[f"workers/{wid}/in_flight"] = w.get("in_flight", 0)
            out[f"workers/{wid}/rps"] = round(rate.get("rps", 0.0), 3)
            out[f"workers/{wid}/tok_s"] = round(rate.get("tok_s", 0.0), 1)
            out[f"workers/{wid}/util"] = round(rate.get("util", 0.0), 3)
    return out


def render_frame(url: str, cur: dict, rates: dict[str, dict], interval: float) -> str:
    r = cur["router"]
    lines = [
        f"fleet monitor — {url}  (refresh {interval:.0f}s, Ctrl-C to stop)",
        (
            f"router: {r['live_workers']} live | in-flight {r['in_flight']} | "
            f"{r['requests']} reqs | {r['rerouted']} rerouted | {r['failed']} failed"
        ),
        "",
        f"{'worker':<28} {'state':<6} {'queued':>6} {'inflt':>5} "
        f"{'req/s':>6} {'tok/s':>7} {'util':>5}",
    ]
    for w in cur["workers"]:
        wid = w["worker_id"]
        if not w.get("ok"):
            lines.append(f"{wid:<28} {'DOWN':<6} {'-':>6} {'-':>5} {'-':>6} {'-':>7} {'-':>5}")
            continue
        rate = rates.get(wid, {})
        lines.append(
            f"{wid:<28} {'ok':<6} {w.get('queued', 0):>6} {w.get('in_flight', 0):>5} "
            f"{rate.get('rps', 0.0):>6.1f} {rate.get('tok_s', 0.0):>7.1f} "
            f"{rate.get('util', 0.0) * 100:>4.0f}%"
        )
    if not cur["workers"]:
        lines.append("(no worker stats yet — first scrape lands within a few seconds)")
    return "\n".join(lines)


def run_monitor(
    cfg: OrchestratorConfig, url: str, *, interval: float = 2.0, use_wandb: bool = False
) -> None:
    import requests

    wandb_run = None
    if use_wandb:
        if not cfg.wandb_enabled():
            print("[monitor] WANDB_API_KEY not set — terminal only", file=sys.stderr)
        else:
            import wandb

            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity or None,
                name=f"fleet-monitor-{time.strftime('%Y%m%d-%H%M%S')}",
                job_type="fleet-monitor",
                config={"url": url, "interval": interval},
            )
            print(f"[monitor] mirroring to W&B: {wandb_run.url}")

    prev = None
    try:
        while True:
            try:
                cur = requests.get(f"{url}/fleet/metrics", timeout=3).json()
            except requests.RequestException as e:
                print(f"[monitor] router unreachable: {e} — retrying", file=sys.stderr)
                time.sleep(interval)
                continue
            rates = compute_rates(prev, cur)
            sys.stdout.write("\x1b[2J\x1b[H")  # clear + home
            print(render_frame(url, cur, rates, interval), flush=True)
            if wandb_run is not None:
                wandb_run.log(flatten_metrics(cur, rates))
            prev = cur
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[monitor] stopped")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
