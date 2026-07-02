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
import re
import sys
import time
from dataclasses import dataclass, field

# Matches the trainer's per-step line (spot_train/train.py):
#   "step 20: loss 2.9876, 80ms/step, 15300 tok/s"
_STEP_RE = re.compile(r"step\s+(\d+):\s+loss\s+([0-9.]+),\s+(\d+)ms/step,\s+(\d+)\s+tok/s")


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
    metrics: dict | None = None
    _seen: set[tuple[int, int]] = field(default_factory=set)  # (segment, step) dedup
    _wb: object | None = field(default=None, repr=False)  # wandb run handle or None
    _wb_step: int = 0  # monotonic step for W&B (survives spot step-number resets)

    # -- timeline ---------------------------------------------------------- #
    def mark(self, event: str) -> None:
        """Record a control-plane transition at the current wall clock."""
        self.events.append(Event(event, time.time(), self.segment))

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
            s = Sample(step, float(m.group(2)), int(m.group(3)), int(m.group(4)))
            self.samples.append(s)
            self._wb_log_step(s)

    def from_metrics(self, metrics: dict | None) -> None:
        self.metrics = metrics

    # -- derived (pure) ---------------------------------------------------- #
    def _t(self, name: str, segment: int | None = None) -> float | None:
        for e in self.events:
            if e.event == name and (segment is None or e.segment == segment):
                return e.t_wall
        return None

    def durations(self) -> dict:
        """Phase durations (seconds, 2dp) derived from events. Robust to missing
        events (timeout path). End = metrics if present, else the last event."""
        launch = self._t("launch")
        first_log = self._t("first_log")
        end = self._t("metrics") or self._t("timeout")
        if end is None and self.events:
            end = self.events[-1].t_wall
        out: dict[str, float] = {}
        if launch is not None and first_log is not None:
            out["provision_s"] = round(first_log - launch, 2)
        if first_log is not None and end is not None:
            out["train_s"] = round(end - first_log, 2)
        if self.events and end is not None:
            out["total_s"] = round(end - self.events[0].t_wall, 2)
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
            "metrics": self.metrics,
        }

    # -- outputs ----------------------------------------------------------- #
    def write(self, cfg) -> None:
        """Serialize to runs/<run_id>/profile.json (S3 or local, via s3_store)."""
        from spot_train import s3_store

        s3_store.put_bytes(
            json.dumps(self.to_dict(), indent=2).encode(), cfg.run_profile_uri(self.run_id)
        )

    def finalize(self, cfg) -> None:
        """Write the S3 artifact, push the W&B summary, and finish the W&B run."""
        self.write(cfg)
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
            group=self.market,
            config={
                "kind": self.kind,
                "market": self.market,
                "dataset": cfg.dataset,
                "instance_type": cfg.instance_type,
            },
        )

    def _wb_log_step(self, s: Sample) -> None:
        if self._wb is None:
            return
        self._wb.log(
            {"loss": s.loss, "ms_per_step": s.ms_per_step, "tok_s": s.tok_s},
            step=self._wb_step,
        )
        self._wb_step += 1

    # -- table row builders (pure; unit-testable without wandb) ------------ #
    def duration_rows(self) -> list[list]:
        """Rows for the durations table/bar chart: [phase, seconds]."""
        return [[k, v] for k, v in self.durations().items()]

    def timeline_rows(self) -> list[list]:
        """Rows for the timeline table: [event, segment, t_rel_s, t_wall]."""
        if not self.events:
            return []
        t0 = self.events[0].t_wall
        return [
            [e.event, e.segment, round(e.t_wall - t0, 2), round(e.t_wall, 3)] for e in self.events
        ]

    def _wb_finish(self) -> None:
        if self._wb is None:
            return
        import wandb

        # Comparable scalar columns in the runs table.
        for k, v in self.durations().items():
            self._wb.summary[k] = v
        if self.metrics:
            for k in ("train_loss", "val_loss", "steps", "stop_reason", "resumed"):
                if k in self.metrics:
                    self._wb.summary[k] = self.metrics[k]

        # Profiling tables + a durations bar chart, built programmatically (they
        # render as panels in the run automatically — no UI setup). Use distinct
        # Table objects: a wandb.Table can only be consumed by one log target.
        timeline = wandb.Table(
            columns=["event", "segment", "t_rel_s", "t_wall"], data=self.timeline_rows()
        )
        durations = wandb.Table(columns=["phase", "seconds"], data=self.duration_rows())
        self._wb.log(
            {
                "profile/timeline": timeline,
                "profile/durations": wandb.plot.bar(
                    durations, "phase", "seconds", title="Run profile (seconds)"
                ),
            }
        )
        self._wb.finish()
