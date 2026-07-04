"""Side-by-side comparison of finished runs (`spot-orchestrate compare`).

Pure S3 reads — no instances, safe to run anytime after the runs finish. Pulls
each run's ``profile.json`` (source of truth: loss/val curves with wall-clock
offsets, materialized timeline segments, text samples, metrics) and renders:

  * ``loss.png``       — train + val loss overlaid, vs wall-clock AND vs step
                         (the wall-clock panel makes preemption gaps visible)
  * ``timelines.png``  — every run's phase timeline stacked on one time axis
  * ``report.md``      — metrics/durations table + prompt outputs side by side

Artifacts land in a local ``reports/<compare_id>/`` directory and are uploaded
to ``s3://<bucket>/reports/<compare_id>/``. PNGs need matplotlib (the ``viz``
extra); without it the markdown report is still produced.
"""

from __future__ import annotations

import json
import os
import sys
import time

from . import aws
from .config import ON_DEMAND_HOURLY_USD, OrchestratorConfig
from .profile import render_multi_timeline_png


def _fetch_profile(cfg: OrchestratorConfig, run_id: str) -> dict:
    key = cfg.run_profile_key(run_id)
    if not aws.object_exists(cfg.bucket, key):
        raise SystemExit(
            f"no profile.json for {run_id!r} (s3://{cfg.bucket}/{key}) — run finished?"
        )
    return json.loads(aws.get_text(cfg.bucket, key))


def _cost_estimate(cfg: OrchestratorConfig, profile: dict) -> float | None:
    """The run's RECORDED cost when the profile carries a ledger (per-instance,
    actual spot rates — see RunProfile.cost_dict); otherwise the old rough
    estimate: total wall-clock x nodes x on-demand hourly rate. Multinode
    estimates bill every live node for the full run (replacements overlap the
    victims' teardown by at most minutes — inside the estimate's precision)."""
    recorded = (profile.get("cost") or {}).get("total_usd")
    if recorded is not None:
        return recorded
    rate = ON_DEMAND_HOURLY_USD.get(cfg.instance_type)
    total = (profile.get("durations") or {}).get("total_s")
    if rate is None or not total:
        return None
    nodes = (profile.get("metrics") or {}).get("world_size") or 1
    return round(total / 3600 * nodes * rate, 3)


def _loss_png(profiles: list[dict], path: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for p in profiles:
        label = p["run_id"]
        samples = p.get("loss_samples") or []
        vals = p.get("val_samples") or []
        if samples:
            ax_t.plot(
                [s.get("t_rel", 0) for s in samples],
                [s["loss"] for s in samples],
                label=label,
                linewidth=1,
            )
            ax_s.plot(
                [s["step"] for s in samples], [s["loss"] for s in samples], label=label, linewidth=1
            )
        if vals:
            ax_t.plot(
                [v.get("t_rel", 0) for v in vals],
                [v["loss"] for v in vals],
                linestyle="--",
                linewidth=1.5,
            )
            ax_s.plot(
                [v["step"] for v in vals], [v["loss"] for v in vals], linestyle="--", linewidth=1.5
            )
    ax_t.set_xlabel("seconds since launch (gaps = downtime)")
    ax_s.set_xlabel("step")
    ax_t.set_ylabel("loss (solid=train, dashed=val)")
    ax_t.legend(fontsize=8)
    fig.suptitle("Loss curves")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def _report_md(cfg: OrchestratorConfig, profiles: list[dict], images: list[str]) -> str:
    lines = ["# Run comparison", ""]
    lines.append(
        "| run | kind | steps | train_loss | val_loss | resumed | training_s | "
        "downtime_s | recovery_s | total_s | est. cost |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in profiles:
        m = p.get("metrics") or {}
        d = p.get("durations") or {}
        cost = _cost_estimate(cfg, p)
        lines.append(
            f"| {p['run_id']} | {p.get('kind')} | {_fmt(m.get('steps'))} | "
            f"{_fmt(m.get('train_loss'))} | {_fmt(m.get('val_loss'))} | "
            f"{_fmt(m.get('resumed'))} | {_fmt(d.get('training_s'))} | "
            f"{_fmt(d.get('downtime_s'))} | {_fmt(d.get('preemption_recovery_s'))} | "
            f"{_fmt(d.get('total_s'))} | {'$' + _fmt(cost) if cost is not None else '—'} |"
        )
    lines.append("")
    for img in images:
        lines.append(f"![{img}]({img})")
    lines.append("")

    # Prompt outputs side by side: prompt -> run -> snapshot steps (early..final).
    prompts: list[str] = []
    for p in profiles:
        for doc in p.get("text_samples") or []:
            for s in doc.get("samples", []):
                if s["prompt"] not in prompts:
                    prompts.append(s["prompt"])
    for prompt in prompts:
        lines.append(f"## Prompt: `{prompt!r}`")
        for p in profiles:
            lines.append(f"\n### {p['run_id']}")
            docs = p.get("text_samples") or []
            hits = [
                (doc.get("step", 0), s)
                for doc in docs
                for s in doc.get("samples", [])
                if s["prompt"] == prompt and s.get("sample_index", 0) == 0
            ]
            if not hits:
                lines.append("_no samples_")
            for step, s in hits:
                lines.append(f"\n**step {step}:**\n")
                lines.append("```\n" + prompt + s["completion"] + "\n```")
        lines.append("")
    return "\n".join(lines)


def run_compare(cfg: OrchestratorConfig, run_ids: list[str]) -> str:
    """Build the comparison artifacts for ``run_ids``; returns the local dir."""
    cfg.require_bucket()
    aws.set_region(cfg.region)
    profiles = [_fetch_profile(cfg, rid) for rid in run_ids]

    compare_id = f"compare-{int(time.time())}"
    out_dir = os.path.join("reports", compare_id)
    os.makedirs(out_dir, exist_ok=True)
    images: list[str] = []

    if _loss_png(profiles, os.path.join(out_dir, "loss.png")):
        images.append("loss.png")
    rows = [(p["run_id"], p.get("segments") or []) for p in profiles]
    if render_multi_timeline_png("Run timelines", rows, os.path.join(out_dir, "timelines.png")):
        images.append("timelines.png")
    if not images:
        print(
            "[compare] matplotlib not installed — markdown only " "(pip install -e '.[viz]')",
            file=sys.stderr,
        )

    report = _report_md(cfg, profiles, images)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(report)

    for name in [*images, "report.md"]:
        aws.upload_file(os.path.join(out_dir, name), cfg.bucket, f"reports/{compare_id}/{name}")
    print(
        f"[compare] wrote {out_dir}/ (report.md{', ' + ', '.join(images) if images else ''}) "
        f"and s3://{cfg.bucket}/reports/{compare_id}/",
        file=sys.stderr,
    )
    return out_dir
