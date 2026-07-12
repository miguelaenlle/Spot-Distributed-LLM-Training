"""Run-profile collector: a tool-agnostic timeline + loss curve built entirely on
the orchestrator side, with an optional Weights & Biases mirror.

No trainer/box changes: everything here is derived from data the orchestrator
already has — control-plane transitions (wall-clock ``mark`` calls) and the boot
log it already streams from S3 (per-step loss lines). We write
``runs/<run_id>/profile.json`` as the source of truth; W&B is a mirror that
no-ops cleanly when the package or an API key is absent, so ``profile.json`` can
feed Grafana or anything else later.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field

# Matches the trainer's per-step line (spot_train/train.py):
#   "step 20: loss 2.9876, 80ms/step, 15300 tok/s, ws 4"
# The trailing world size is optional so pre-elastic logs still parse; it is
# what renders the N -> N-1 -> N staircase next to the loss curve.
_STEP_RE = re.compile(
    r"step\s+(\d+):\s+loss\s+([0-9.]+),\s+(\d+)ms/step,\s+(\d+)\s+tok/s(?:,\s+ws\s+(\d+))?"
)

# Matches the trainer's periodic exact-eval line (Gap E):
#   "eval step 250: val_loss 2.1034"
_EVAL_RE = re.compile(r"eval\s+step\s+(\d+):\s+val_loss\s+([0-9.]+)")

# The timeline is a sequence of phases running between ordered milestones. Crucially
# train_start / train_end come from the FIRST / LAST per-step line actually observed
# (not "first log byte"), so the training phase reflects the loop (~max_seconds) and
# the boot/clone/pip/dataset work lands in provisioning while eval+checkpoint+metrics
# land in final_saves. Each consecutive milestone pair maps to a named phase.
_SEGMENT_LABEL: dict[tuple[str, str], str] = {
    ("launch", "train_start"): "provisioning",
    ("train_start", "train_end"): "training",
    ("train_end", "end"): "final_saves",
    # degenerate paths (kept so a partial run still yields a sane bar):
    ("launch", "end"): "provisioning",  # never reached the first training step
    ("launch", "train_end"): "provisioning",  # exactly one step observed
    ("train_start", "end"): "training",  # metrics immediately after the last step
}

# Preemption timeline is built from control-plane MARKS (not per-step lines): each
# consecutive (from_mark, to_mark) pair maps to a phase. Repeats for each segment.
# Elastic runs add shrink_resume (survivors checkpointing again at N-1) and
# full_world (world size back to N): kill -> shrink_resume is the true downtime,
# shrink_resume -> full_world is DEGRADED training (fewer GPUs, still learning).
# relaunch (the replacement's launch) can land on either side of shrink_resume.
_PREEMPT_MARKS = {
    "launch",
    "relaunch",
    "train_start",
    "kill",
    "shrink_resume",
    "full_world",
    "metrics",
    "timeout",
}
_MARK_PHASE: dict[tuple[str, str], str] = {
    ("launch", "train_start"): "provisioning",
    ("relaunch", "train_start"): "preemption_recovery",
    ("train_start", "kill"): "training",
    ("kill", "relaunch"): "downtime",
    ("train_start", "metrics"): "training",  # final segment (split via stamps below)
    ("train_start", "timeout"): "training",
    # elastic (degraded-mode) sequences
    ("kill", "shrink_resume"): "downtime",
    ("relaunch", "shrink_resume"): "downtime",
    ("shrink_resume", "relaunch"): "degraded",
    ("shrink_resume", "full_world"): "degraded",
    ("relaunch", "full_world"): "degraded",
    ("shrink_resume", "kill"): "degraded",
    ("shrink_resume", "metrics"): "degraded",
    ("shrink_resume", "timeout"): "degraded",
    ("full_world", "kill"): "training",
    ("full_world", "metrics"): "training",
    ("full_world", "timeout"): "training",
}

# Stable colors per phase (used by the stacked timeline bar).
_PHASE_COLORS: dict[str, str] = {
    "provisioning": "#4C78A8",
    "training": "#54A24B",
    "degraded": "#EECA3B",  # training at reduced world size (one node down)
    "downtime": "#9E9E9E",
    "preemption_recovery": "#F58518",
    "evaluation": "#72B7B2",
    "final_saves": "#B279A2",
    "sampling": "#E45756",
}


@dataclass
class Event:
    # event vocab (baseline): launch | first_log | metrics | timeout
    # (spot later adds: first_checkpoint | kill | relaunch)
    event: str
    t_wall: float
    segment: int


@dataclass
class Sample:
    step: int
    loss: float
    ms_per_step: int
    tok_s: int
    # Seconds since the run's first event, stamped when the line was OBSERVED
    # (log-poll granularity, ~3s). Lets loss curves overlay on a wall-clock axis
    # where preemption gaps are visible.
    t_rel: float = 0.0
    # Live world size from the trainer's `ws N` suffix (None on pre-elastic
    # logs) — the per-step record of the N -> N-1 -> N staircase.
    world_size: int | None = None


@dataclass
class ValSample:
    """A periodic exact-eval point (`eval step N: val_loss X`)."""

    step: int
    loss: float
    t_rel: float = 0.0


@dataclass
class InstanceCost:
    """One cost-ledger row: an EC2 box billed per-second from ``running`` to
    terminate — each instance separately, matching AWS billing. ``hourly_usd``
    is the actual spot price at launch (per-AZ) or the on-demand rate; None
    means unknown (sums skip the row; cost_dict counts it so nothing hides)."""

    instance_id: str
    market: str
    az: str
    hourly_usd: float | None
    started_at: float
    stopped_at: float | None = None


@dataclass
class RunProfile:
    """Accumulates a run's timeline + loss samples, writes profile.json, and
    optionally mirrors to W&B. One instance per run (per run_id)."""

    run_id: str
    kind: str  # "baseline" | "spot"
    market: str  # "on-demand" | "spot"
    segment: int = 1  # bump to 2 before a spot relaunch (future); baseline stays 1
    events: list[Event] = field(default_factory=list)
    samples: list[Sample] = field(default_factory=list)
    val_samples: list[ValSample] = field(default_factory=list)
    # Text-sample documents (end-of-run samples.json + per-step snapshots), each
    # {"step", "params", "samples": [{"prompt", "sample_index", "completion"}]}.
    text_samples: list[dict] = field(default_factory=list)
    metrics: dict | None = None
    # Cost ledger: one row per EC2 box this run launched (see InstanceCost).
    instances: list[InstanceCost] = field(default_factory=list)
    _seen: set[tuple[int, int]] = field(default_factory=set)  # (segment, step) dedup
    _seen_val: set[tuple[int, int]] = field(default_factory=set)
    _wb: object | None = field(default=None, repr=False)  # wandb run handle or None
    _wb_step: int = 0  # monotonic step for W&B (survives spot step-number resets)
    # Wall-clock of the first/last per-step line observed => training-phase bounds.
    _first_sample_wall: float | None = None
    _last_sample_wall: float | None = None

    # -- timeline ---------------------------------------------------------- #
    def mark(self, event: str) -> None:
        """Record a control-plane transition at the current wall clock."""
        self.events.append(Event(event, time.time(), self.segment))

    def _t_rel_now(self, now: float) -> float:
        """Seconds since the run's first event (0 if nothing marked yet)."""
        return round(now - self.events[0].t_wall, 2) if self.events else 0.0

    def ingest_log(self, text: str) -> None:
        """Parse per-step lines from the FULL boot log, dedup on (segment, step),
        and log newly-seen steps to W&B. Called with the whole log each poll —
        dedup makes repeated full reads idempotent; the regex requires a complete
        line, so mid-write partial tails are ignored until finished."""
        for m in _STEP_RE.finditer(text):
            step = int(m.group(1))
            key = (self.segment, step)
            if key in self._seen:
                continue
            self._seen.add(key)
            now = time.time()  # observation time bounds the training phase
            s = Sample(
                step,
                float(m.group(2)),
                int(m.group(3)),
                int(m.group(4)),
                t_rel=self._t_rel_now(now),
                world_size=int(m.group(5)) if m.group(5) else None,
            )
            self.samples.append(s)
            if self._first_sample_wall is None:
                self._first_sample_wall = now
            self._last_sample_wall = now
            self._wb_log_step(s)
        for m in _EVAL_RE.finditer(text):
            step = int(m.group(1))
            key = (self.segment, step)
            if key in self._seen_val:
                continue
            self._seen_val.add(key)
            v = ValSample(step, float(m.group(2)), t_rel=self._t_rel_now(time.time()))
            self.val_samples.append(v)
            self._wb_log_val(v)

    def from_metrics(self, metrics: dict | None) -> None:
        self.metrics = metrics

    def from_samples(self, doc: dict | None) -> None:
        """Attach a text-samples document (final samples.json or a per-step
        snapshot). Dedup by step — a resumed run rewrites a snapshot key with
        byte-identical content, so first-seen wins."""
        if not doc or not doc.get("samples"):
            return
        if any(d.get("step") == doc.get("step") for d in self.text_samples):
            return
        self.text_samples.append(doc)
        self.text_samples.sort(key=lambda d: d.get("step", 0))

    # -- cost ledger --------------------------------------------------------- #
    def instance_started(
        self,
        instance_id: str,
        market: str,
        az: str,
        hourly_usd: float | None,
        t: float | None = None,
    ) -> None:
        """Open a ledger row when a box reaches ``running`` (billing starts)."""
        self.instances.append(
            InstanceCost(instance_id, market, az, hourly_usd, t if t is not None else time.time())
        )
        self._wb_config_cost()

    def instance_stopped(self, instance_id: str, t: float | None = None) -> None:
        """Close the open row for ``instance_id`` (no-op if unknown/closed)."""
        for row in self.instances:
            if row.instance_id == instance_id and row.stopped_at is None:
                row.stopped_at = t if t is not None else time.time()
                return

    def close_open_instances(self) -> None:
        """Stamp any still-open rows at now — called at finalize, where the
        remaining boxes are terminated immediately after."""
        now = time.time()
        for row in self.instances:
            if row.stopped_at is None:
                row.stopped_at = now

    def cost_at(self, t_wall: float) -> float:
        """Cumulative USD across the ledger at wall-clock ``t_wall``. Rows with
        an unknown rate contribute 0 (cost_dict flags how many were skipped)."""
        total = 0.0
        for row in self.instances:
            if row.hourly_usd is None:
                continue
            end = row.stopped_at if row.stopped_at is not None else t_wall
            total += max(0.0, min(end, t_wall) - row.started_at) * row.hourly_usd / 3600.0
        return total

    def cost_now(self) -> float:
        return self.cost_at(time.time())

    def hourly_rate_now(self) -> float | None:
        """Rate of the newest still-open row (the box currently training)."""
        for row in reversed(self.instances):
            if row.stopped_at is None:
                return row.hourly_usd
        return None

    def cost_dict(self) -> dict | None:
        """The ``cost`` section of profile.json, or None with an empty ledger."""
        if not self.instances:
            return None
        now = time.time()
        items = []
        for row in self.instances:
            end = row.stopped_at if row.stopped_at is not None else now
            seconds = round(max(0.0, end - row.started_at), 2)
            usd = None
            if row.hourly_usd is not None:
                usd = round(seconds * row.hourly_usd / 3600.0, 6)
            items.append(
                {
                    "instance_id": row.instance_id,
                    "market": row.market,
                    "az": row.az,
                    "hourly_usd": row.hourly_usd,
                    "started_at": round(row.started_at, 3),
                    "stopped_at": round(row.stopped_at, 3) if row.stopped_at is not None else None,
                    "billed_seconds": seconds,
                    "usd": usd,
                }
            )
        return {
            "instances": items,
            "total_usd": round(self.cost_at(now), 6),
            "rate_unknown_count": sum(1 for r in self.instances if r.hourly_usd is None),
        }

    def cost_curve(self) -> tuple[list[float], list[float]]:
        """Piecewise-linear cumulative cost: (t_rel seconds, USD) sampled at
        every ledger breakpoint (each box's start/stop), relative to the run's
        first event. Exact — cost is linear between breakpoints."""
        if not self.instances:
            return [], []
        t0 = self.events[0].t_wall if self.events else min(r.started_at for r in self.instances)
        pts = {t0}
        for r in self.instances:
            pts.add(r.started_at)
            pts.add(r.stopped_at if r.stopped_at is not None else time.time())
        xs = sorted(pts)
        return [round(x - t0, 2) for x in xs], [round(self.cost_at(x), 6) for x in xs]

    # -- derived (pure) ---------------------------------------------------- #
    def _t(self, name: str, segment: int | None = None) -> float | None:
        for e in self.events:
            if e.event == name and (segment is None or e.segment == segment):
                return e.t_wall
        return None

    def _milestones(self) -> list[tuple[str, float]]:
        """Ordered (name, t_wall) phase boundaries. train_start/train_end are the
        first/last per-step line observed, so training reflects the loop — not the
        boot/clone/pip before it or the eval/checkpoint/metrics after it."""
        pts: list[tuple[str, float]] = []
        launch = self._t("launch")
        if launch is not None:
            pts.append(("launch", launch))
        if self._first_sample_wall is not None:
            pts.append(("train_start", self._first_sample_wall))
        if self._last_sample_wall is not None:
            pts.append(("train_end", self._last_sample_wall))
        end = self._t("metrics") or self._t("timeout")
        if end is None and self.events:
            end = self.events[-1].t_wall
        if end is not None:
            pts.append(("end", end))
        pts.sort(key=lambda p: p[1])
        return pts

    def durations(self) -> dict:
        """Phase durations (seconds, 2dp), summed per phase, plus total_s."""
        out: dict[str, float] = {}
        for s in self.segments():
            key = f"{s['phase']}_s"
            out[key] = round(out.get(key, 0.0) + s["seconds"], 2)
        pts = self._milestones()
        if len(pts) >= 2:
            out["total_s"] = round(pts[-1][1] - pts[0][1], 2)
        return out

    def to_dict(self) -> dict:
        first = self.events[0].t_wall if self.events else time.time()
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "market": self.market,
            "schema_version": 1,
            "created_at": round(time.time(), 3),
            "durations": self.durations(),
            # Materialized so `spot-orchestrate compare` can rebuild timelines
            # from profile.json alone (no live RunProfile needed).
            "segments": self.segments(),
            "events": [
                {
                    "event": e.event,
                    "t_wall": round(e.t_wall, 3),
                    "t_rel": round(e.t_wall - first, 3),
                    "segment": e.segment,
                }
                for e in self.events
            ],
            "loss_samples": [vars(s) for s in self.samples],
            "val_samples": [vars(v) for v in self.val_samples],
            "text_samples": self.text_samples,
            "metrics": self.metrics,
            "cost": self.cost_dict(),
        }

    # -- outputs ----------------------------------------------------------- #
    def write(self, cfg) -> None:
        """Serialize to runs/<run_id>/profile.json (S3 or local, via s3_store)."""
        from spot_train import s3_store

        s3_store.put_bytes(
            json.dumps(self.to_dict(), indent=2).encode(), cfg.run_profile_uri(self.run_id)
        )

    def finalize(self, cfg) -> None:
        """Close the cost ledger, write the S3 artifacts (profile.json +
        cost.png), push the W&B summary, and finish the W&B run."""
        self.close_open_instances()
        self.write(cfg)
        png = os.path.join(tempfile.gettempdir(), f"{self.run_id}-cost.png")
        if self.render_cost_png(png):
            from spot_train import s3_store

            s3_store.put_file(png, cfg.run_cost_png_uri(self.run_id))
            print(f"[profile] cost graph: {cfg.run_cost_png_uri(self.run_id)}", file=sys.stderr)
        self._wb_finish()

    # -- optional W&B mirror (all no-op when self._wb is None) ------------- #
    def wandb_start(self, cfg) -> None:
        """Init a single W&B run for this run_id iff enabled + importable + keyed.
        One run per run_id, so future spot segments share it (kill gap visible)."""
        if not cfg.wandb_enabled():
            return
        try:
            import wandb
        except ImportError:
            print(
                "[profile] wandb not installed; skipping viz (pip install -e '.[viz]')",
                file=sys.stderr,
            )
            return
        self._wb = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            name=self.run_id,
            # A comparison suite groups its runs explicitly (WANDB_GROUP);
            # otherwise keep the historical group-by-market behavior.
            group=cfg.wandb_group or self.market,
            config={
                "kind": self.kind,
                "market": self.market,
                "dataset": cfg.dataset,
                "instance_type": cfg.instance_type,
            },
        )
        # Chart against the REAL training step. We log with a monotonic internal
        # step (self._wb_step) so a spot step-number reset never makes W&B's
        # implicit step go backwards — but that counter advances once per logged
        # line (every LOG_INTERVAL_STEPS steps), which made the x-axis read
        # step/10. define_metric points every chart's x-axis at the logged
        # `train_step` instead, so it reads in true training steps.
        self._wb.define_metric("train_step")
        self._wb.define_metric("*", step_metric="train_step")

    def _wb_config_cost(self) -> None:
        """Mirror the newest box's rate into the run config (Overview panel)."""
        if self._wb is None or not self.instances:
            return
        row = self.instances[-1]
        if row.hourly_usd is not None:
            self._wb.config.update(
                {"hourly_usd": row.hourly_usd, "az": row.az}, allow_val_change=True
            )

    def _wb_log_step(self, s: Sample) -> None:
        if self._wb is None:
            return
        payload = {
            # t_rel/train_step let the W&B x-axis switch to wall-clock (gaps
            # visible) or the trainer's own step (resume overlaps visible).
            "loss": s.loss,
            "ms_per_step": s.ms_per_step,
            "tok_s": s.tok_s,
            "t_rel": s.t_rel,
            "train_step": s.step,
        }
        if s.world_size is not None:
            # Charted against the same axes as loss: the world-size staircase
            # (N -> N-1 while a replacement boots -> N) under the loss curve.
            payload["world_size"] = s.world_size
        if self.instances:
            payload["cost/cumulative_usd"] = round(self.cost_now(), 6)
            rate = self.hourly_rate_now()
            if rate is not None:
                payload["cost/hourly_usd"] = rate
        self._wb.log(payload, step=self._wb_step)
        self._wb_step += 1

    def _wb_log_val(self, v: ValSample) -> None:
        if self._wb is None:
            return
        # Same monotonic W&B step counter, no increment: the val point merges
        # into the current x position between two train samples.
        self._wb.log(
            {"val_loss": v.loss, "t_rel": v.t_rel, "train_step": v.step}, step=self._wb_step
        )

    def _stamp_phases(self) -> list[dict]:
        """training/final_saves/evaluation from the trainer's exact stamps, or []."""
        ph = self.metrics.get("phases") if isinstance(self.metrics, dict) else None
        if not isinstance(ph, dict):
            return []
        out = []
        for phase, key in (
            ("training", "train_s"),
            ("final_saves", "save_s"),
            ("evaluation", "eval_s"),
            ("sampling", "sample_s"),
        ):
            secs = ph.get(key)
            if secs:
                out.append({"phase": phase, "seconds": round(secs, 2)})
        return out

    def _segments_from_marks(self) -> list[dict]:
        """Preemption timeline: walk the control-plane marks into the repeated
        provisioning/training/downtime/preemption_recovery sequence. The final
        training block is split via the trainer's stamps."""
        evs = [e for e in self.events if e.event in _PREEMPT_MARKS]
        out: list[dict] = []
        for a, b in zip(evs, evs[1:], strict=False):
            phase = _MARK_PHASE.get((a.event, b.event))
            if phase is None:
                continue
            if phase == "training" and b.event in ("metrics", "timeout"):
                split = self._stamp_phases()
                if split:
                    out.extend(split)
                    continue
            out.append({"phase": phase, "seconds": round(b.t_wall - a.t_wall, 2)})
        return out

    # -- phase segments (ordered, non-overlapping) ------------------------- #
    def segments(self) -> list[dict]:
        """The timeline as an ordered list of phases [{"phase", "seconds"}].

        - Preemption runs (any "kill" mark): walk the control-plane marks into the
          repeated provisioning/training/downtime/recovery sequence.
        - Otherwise prefer the trainer's EXACT wall-clock stamps in metrics.json.
        - Else fall back to the per-step-line milestone proxy (timeout / old trainer).
        """
        if any(e.event == "kill" for e in self.events):
            return self._segments_from_marks()
        m = self.metrics
        launch = self._t("launch")
        if (
            m
            and launch is not None
            and m.get("train_started_at")
            and isinstance(m.get("phases"), dict)
        ):
            ph = m["phases"]
            out: list[dict] = []
            prov = round(m["train_started_at"] - launch, 2)
            if prov > 0:
                out.append({"phase": "provisioning", "seconds": prov})
            # order matches the trainer: loop -> final checkpoint -> eval -> sample
            for phase, key in (
                ("training", "train_s"),
                ("final_saves", "save_s"),
                ("evaluation", "eval_s"),
                ("sampling", "sample_s"),
            ):
                secs = ph.get(key)
                if secs:
                    out.append({"phase": phase, "seconds": round(secs, 2)})
            return out

        # fallback: derive from the first/last per-step line observed
        out = []
        pts = self._milestones()
        for (na, ta), (nb, tb) in zip(pts, pts[1:], strict=False):
            phase = _SEGMENT_LABEL.get((na, nb))
            if phase is None:
                continue
            out.append({"phase": phase, "seconds": round(tb - ta, 2)})
        return out

    # -- table row builders (pure; unit-testable without wandb) ------------ #
    def segment_rows(self) -> list[list]:
        """Rows for the stacked timeline: [phase, seconds, start_s] in time order."""
        rows, start = [], 0.0
        for s in self.segments():
            rows.append([s["phase"], s["seconds"], round(start, 2)])
            start += s["seconds"]
        return rows

    def duration_rows(self) -> list[list]:
        """Rows for the durations table/bar chart: [phase, seconds]."""
        return [[k, v] for k, v in self.durations().items()]

    def sample_rows(self) -> list[list]:
        """Rows for the text-samples table: [step, prompt, sample_index, completion],
        one per generated sample across all snapshots + the final document."""
        return [
            [doc.get("step", 0), s["prompt"], s.get("sample_index", 0), s["completion"]]
            for doc in self.text_samples
            for s in doc.get("samples", [])
        ]

    def val_rows(self) -> list[list]:
        """Rows for the periodic exact-eval table: [step, val_loss, t_rel_s]."""
        return [[v.step, v.loss, v.t_rel] for v in self.val_samples]

    def cost_rows(self) -> list[list]:
        """Rows for the cost table: [instance_id, market, az, hourly_usd,
        billed_seconds, usd] — one per launched box."""
        cost = self.cost_dict()
        if cost is None:
            return []
        return [
            [r["instance_id"], r["market"], r["az"], r["hourly_usd"], r["billed_seconds"], r["usd"]]
            for r in cost["instances"]
        ]

    def timeline_rows(self) -> list[list]:
        """Rows for the timeline table: [event, segment, t_rel_s, t_wall]."""
        if not self.events:
            return []
        t0 = self.events[0].t_wall
        return [
            [e.event, e.segment, round(e.t_wall - t0, 2), round(e.t_wall, 3)] for e in self.events
        ]

    def render_timeline_png(self, path: str) -> bool:
        """Render this run's phases as a stacked timeline bar to ``path``."""
        return render_segments_png(f"Run timeline — {self.run_id}", self.segments(), path)

    def render_cost_png(self, path: str) -> bool:
        """Render the two-panel cost graph to ``path``: cumulative $ vs
        wall-clock (kill/relaunch markers) over loss vs cumulative $ (the
        loss-per-dollar curve). False without matplotlib, ledger rows, or any
        known rate."""
        xs, ys = self.cost_curve()
        if not xs or all(r.hourly_usd is None for r in self.instances):
            return False
        try:
            import matplotlib

            matplotlib.use("Agg")  # headless
            import matplotlib.pyplot as plt
        except ImportError:
            return False

        t0 = self.events[0].t_wall if self.events else min(r.started_at for r in self.instances)
        # Exact val evals when present, else the (noisier) per-step train loss.
        series: list = self.val_samples or self.samples
        fig, axes = plt.subplots(
            2 if series else 1, 1, figsize=(10.5, 6.5 if series else 3.2), squeeze=False
        )
        ax0 = axes[0][0]
        ax0.plot(xs, ys, color=_PHASE_COLORS["provisioning"], linewidth=1.5)
        for e in self.events:
            if e.event in ("kill", "relaunch"):
                ax0.axvline(
                    e.t_wall - t0,
                    color=_PHASE_COLORS["downtime" if e.event == "kill" else "preemption_recovery"],
                    linestyle="--",
                    linewidth=0.9,
                )
        ax0.set_xlabel("seconds since launch")
        ax0.set_ylabel("cumulative USD")
        ax0.set_title(f"Cost — {self.run_id} (total ${ys[-1]:.4f})")

        if series:
            ax1 = axes[1][0]
            cx = [self.cost_at(t0 + p.t_rel) for p in series]
            cy = [p.loss for p in series]
            is_val = bool(self.val_samples)
            ax1.plot(
                cx,
                cy,
                marker="o" if is_val else None,
                markersize=3,
                linewidth=1.0,
                color=_PHASE_COLORS["training"],
                label="val_loss" if is_val else "train loss",
            )
            ax1.set_xlabel("cumulative USD")
            ax1.set_ylabel("loss")
            ax1.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return True

    def _wb_finish(self) -> None:
        if self._wb is None:
            return
        import wandb

        # Comparable scalar columns in the runs table.
        for k, v in self.durations().items():
            self._wb.summary[k] = v
        cost = self.cost_dict()
        if cost is not None:
            self._wb.summary["total_usd"] = cost["total_usd"]
        if self.metrics:
            for k in (
                "train_loss",
                "val_loss",
                "steps",
                "stop_reason",
                "resumed",
                "world_size",
                "restart_count",
                "effective_global_batch",
                "trained_seconds_total",
            ):
                if k in self.metrics:
                    self._wb.summary[k] = self.metrics[k]
            # Goodput: fraction of the run's wall-clock spent actually training
            # (trained_seconds is checkpoint-carried, so downtime never counts).
            trained = self.metrics.get("trained_seconds_total")
            total = self.durations().get("total_s")
            if trained and total:
                self._wb.summary["goodput"] = round(trained / total, 4)

        # The timeline as (1) a per-event table and (2) a stacked segment bar image,
        # both built programmatically — they render as panels with no UI setup.
        log: dict[str, object] = {
            "profile/timeline": wandb.Table(
                columns=["event", "segment", "t_rel_s", "t_wall"], data=self.timeline_rows()
            ),
            "profile/segments": wandb.Table(
                columns=["phase", "seconds", "start_s"], data=self.segment_rows()
            ),
        }
        sample_rows = self.sample_rows()
        if sample_rows:
            log["samples/text"] = wandb.Table(
                columns=["step", "prompt", "sample_index", "completion"], data=sample_rows
            )
        val_rows = self.val_rows()
        if val_rows:
            log["eval/val_table"] = wandb.Table(
                columns=["step", "val_loss", "t_rel_s"], data=val_rows
            )
        if self.instances:
            log["profile/cost"] = wandb.Table(
                columns=["instance_id", "market", "az", "hourly_usd", "billed_seconds", "usd"],
                data=self.cost_rows(),
            )
        png = os.path.join(tempfile.gettempdir(), f"{self.run_id}-timeline.png")
        if self.render_timeline_png(png):
            log["profile/timeline_bar"] = wandb.Image(png)
        cost_png = os.path.join(tempfile.gettempdir(), f"{self.run_id}-cost.png")
        # finalize() renders this before uploading to S3; render here only if
        # _wb_finish is reached some other way.
        if os.path.exists(cost_png) or self.render_cost_png(cost_png):
            log["cost/graph"] = wandb.Image(cost_png)
        self._wb.log(log)
        self._wb.finish()


# --------------------------------------------------------------------------- #
# Rendering (module-level so it can preview arbitrary phase sequences — a single
# run now, a multi-node Gantt later)
# --------------------------------------------------------------------------- #
def render_multi_timeline_png(title: str, rows: list[tuple[str, list[dict]]], path: str) -> bool:
    """Render several runs' phase sequences as stacked horizontal bars sharing
    one time axis — the side-by-side timeline for `spot-orchestrate compare`.
    ``rows`` is [(label, segments), ...] top-to-bottom. Returns False if
    matplotlib is absent or there is nothing to draw."""
    rows = [(label, segs) for label, segs in rows if segs]
    if not rows:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(10.5, 0.9 * len(rows) + 1.2))
    seen: set[str] = set()
    for y, (_label, segments) in enumerate(reversed(rows)):
        left = 0.0
        for s in segments:
            color = _PHASE_COLORS.get(s["phase"], "#777777")
            legend_label = None if s["phase"] in seen else s["phase"]
            seen.add(s["phase"])
            ax.barh(y, s["seconds"], left=left, color=color, edgecolor="white", label=legend_label)
            if s["seconds"] > 0:
                ax.text(
                    left + s["seconds"] / 2,
                    y,
                    f"{s['seconds']:.0f}s",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
                )
            left += s["seconds"]
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in reversed(rows)], fontsize=8)
    ax.set_xlabel("seconds")
    ax.set_title(title)
    legend = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.close(fig)
    return True


def render_segments_png(title: str, segments: list[dict], path: str) -> bool:
    """Render an ordered list of ``{"phase", "seconds"}`` as one horizontal
    STACKED bar (segment width = seconds, colored per phase) to ``path``. Returns
    False if matplotlib is absent or there are no segments."""
    if not segments:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(10.5, 2.0))
    left = 0.0
    seen: set[str] = set()
    for s in segments:
        color = _PHASE_COLORS.get(s["phase"], "#777777")
        label = None if s["phase"] in seen else s["phase"]
        seen.add(s["phase"])
        ax.barh(0, s["seconds"], left=left, color=color, edgecolor="white", label=label)
        if s["seconds"] > 0:
            ax.text(
                left + s["seconds"] / 2,
                0,
                f"{s['seconds']:.0f}s",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
        left += s["seconds"]
    ax.set_yticks([])
    ax.set_xlabel("seconds")
    ax.set_title(title)
    # Legend on the right so it never collides with the "seconds" x-axis label.
    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
    )
    fig.savefig(path, dpi=120, bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.close(fig)
    return True
