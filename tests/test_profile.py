"""Run-profile collector tests — hermetic (no network/AWS/W&B).

Pins the pieces the timeline correctness depends on: the per-step log parser
(exact trainer format, typing, dedup across repeated full-log reads, partial-line
safety), the phase-duration math, the timeout path, the local write, and the
W&B-disabled no-op.
"""

from __future__ import annotations

import json
import types

from orchestrator.profile import Event, RunProfile

# Mirrors the trainer's stdout, including noise lines the parser must ignore.
LOG = """\
+ /opt/pytorch/bin/python -u -m spot_train.train
[cpu] running on CPU
number of parameters: 10.67M
[fresh] no checkpoint found, starting from step 0
step 10: loss 3.3405, 3085ms/step, 996 tok/s
step 20: loss 2.9876, 80ms/step, 15300 tok/s
"""


def test_ingest_parses_and_types():
    p = RunProfile("baseline-1", "baseline", "on-demand")
    p.ingest_log(LOG)
    assert [s.step for s in p.samples] == [10, 20]
    s = p.samples[0]
    assert isinstance(s.loss, float) and s.loss == 3.3405
    assert isinstance(s.ms_per_step, int) and s.ms_per_step == 3085
    assert isinstance(s.tok_s, int) and s.tok_s == 996


def test_ingest_dedup_on_repeated_full_reads():
    p = RunProfile("baseline-1", "baseline", "on-demand")
    p.ingest_log(LOG)
    # Re-feeding the whole log (as the poll loop does) plus one new line must not
    # duplicate 10/20 and must pick up 30.
    p.ingest_log(LOG + "step 30: loss 2.5000, 70ms/step, 16000 tok/s\n")
    assert [s.step for s in p.samples] == [10, 20, 30]


def test_ingest_ignores_partial_tail():
    p = RunProfile("baseline-1", "baseline", "on-demand")
    p.ingest_log("step 40: loss 2.1")  # mid-write truncated line
    assert p.samples == []
    p.ingest_log("step 40: loss 2.1234, 60ms/step, 17000 tok/s\n")
    assert [s.step for s in p.samples] == [40]


def test_durations_baseline():
    p = RunProfile("baseline-1", "baseline", "on-demand")
    p.events = [
        Event("launch", 1000.0, 1),
        Event("first_log", 1090.0, 1),
        Event("metrics", 1390.0, 1),
    ]
    d = p.durations()
    assert d == {"provision_s": 90.0, "train_s": 300.0, "total_s": 390.0}


def test_to_dict_timeout_path():
    p = RunProfile("baseline-1", "baseline", "on-demand")
    p.events = [
        Event("launch", 1000.0, 1),
        Event("first_log", 1050.0, 1),
        Event("timeout", 1200.0, 1),
    ]
    p.from_metrics(None)
    d = p.to_dict()
    assert d["metrics"] is None
    assert d["durations"] == {"provision_s": 50.0, "train_s": 150.0, "total_s": 200.0}
    assert d["events"][0]["t_rel"] == 0.0
    assert d["events"][-1]["t_rel"] == 200.0


def test_write_local(tmp_path):
    p = RunProfile("baseline-1", "baseline", "on-demand")
    p.ingest_log(LOG)
    p.events = [
        Event("launch", 1000.0, 1),
        Event("first_log", 1090.0, 1),
        Event("metrics", 1390.0, 1),
    ]
    p.from_metrics({"train_loss": 1.2, "val_loss": 1.3, "steps": 20})
    out = tmp_path / "profile.json"
    cfg = types.SimpleNamespace(run_profile_uri=lambda rid: str(out))
    p.write(cfg)
    data = json.loads(out.read_text())
    assert data["run_id"] == "baseline-1"
    assert data["metrics"]["train_loss"] == 1.2
    assert len(data["loss_samples"]) == 2
    assert data["durations"]["train_s"] == 300.0


def test_table_rows():
    p = RunProfile("baseline-1", "baseline", "on-demand")
    p.events = [
        Event("launch", 1000.0, 1),
        Event("first_log", 1090.0, 1),
        Event("metrics", 1390.0, 1),
    ]
    assert p.duration_rows() == [
        ["provision_s", 90.0],
        ["train_s", 300.0],
        ["total_s", 390.0],
    ]
    assert p.timeline_rows() == [
        ["launch", 1, 0.0, 1000.0],
        ["first_log", 1, 90.0, 1090.0],
        ["metrics", 1, 390.0, 1390.0],
    ]


def test_table_rows_empty():
    p = RunProfile("baseline-1", "baseline", "on-demand")
    assert p.timeline_rows() == []
    assert p.duration_rows() == []


def test_wandb_disabled_is_noop():
    p = RunProfile("baseline-1", "baseline", "on-demand")
    cfg = types.SimpleNamespace(wandb_enabled=lambda: False)
    p.wandb_start(cfg)  # must not import wandb or raise
    assert p._wb is None
    p.ingest_log(LOG)  # _wb_log_step must no-op
    p._wb_finish()  # must no-op
