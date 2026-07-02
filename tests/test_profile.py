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


# Baseline phase boundaries: launch=1000, first step observed at 1090 (provision
# 90), last step at 1150 (training 60 == the budget), metrics at 1165 (final_saves
# 15 = eval + checkpoint + metrics write).
def _baseline_profile() -> RunProfile:
    p = RunProfile("baseline-1", "baseline", "on-demand")
    p.events = [Event("launch", 1000.0, 1), Event("metrics", 1165.0, 1)]
    p._first_sample_wall = 1090.0
    p._last_sample_wall = 1150.0
    return p


def test_durations_baseline():
    d = _baseline_profile().durations()
    assert d == {
        "provisioning_s": 90.0,
        "training_s": 60.0,  # == max_seconds, NOT boot+clone+eval
        "final_saves_s": 15.0,
        "total_s": 165.0,
    }


def test_to_dict_timeout_path():
    # No per-step samples (died before/at first step) => whole span is provisioning.
    p = RunProfile("baseline-1", "baseline", "on-demand")
    p.events = [
        Event("launch", 1000.0, 1),
        Event("first_log", 1050.0, 1),
        Event("timeout", 1200.0, 1),
    ]
    p.from_metrics(None)
    d = p.to_dict()
    assert d["metrics"] is None
    assert d["durations"] == {"provisioning_s": 200.0, "total_s": 200.0}
    assert d["events"][0]["t_rel"] == 0.0
    assert d["events"][-1]["t_rel"] == 200.0


def test_write_local(tmp_path):
    p = _baseline_profile()
    p.ingest_log(LOG)  # 2 loss samples (walls overwritten by _baseline_profile setup)
    p._first_sample_wall = 1090.0
    p._last_sample_wall = 1150.0
    p.from_metrics({"train_loss": 1.2, "val_loss": 1.3, "steps": 20})
    out = tmp_path / "profile.json"
    cfg = types.SimpleNamespace(run_profile_uri=lambda rid: str(out))
    p.write(cfg)
    data = json.loads(out.read_text())
    assert data["run_id"] == "baseline-1"
    assert data["metrics"]["train_loss"] == 1.2
    assert len(data["loss_samples"]) == 2
    assert data["durations"]["training_s"] == 60.0


def test_table_rows():
    p = _baseline_profile()
    assert p.duration_rows() == [
        ["provisioning_s", 90.0],
        ["training_s", 60.0],
        ["final_saves_s", 15.0],
        ["total_s", 165.0],
    ]
    assert p.timeline_rows() == [
        ["launch", 1, 0.0, 1000.0],
        ["metrics", 1, 165.0, 1165.0],
    ]


def test_table_rows_empty():
    p = RunProfile("baseline-1", "baseline", "on-demand")
    assert p.timeline_rows() == []
    assert p.duration_rows() == []


def test_segments_baseline():
    p = _baseline_profile()
    assert p.segments() == [
        {"phase": "provisioning", "seconds": 90.0},
        {"phase": "training", "seconds": 60.0},
        {"phase": "final_saves", "seconds": 15.0},
    ]
    # stacked-bar rows carry the running start offset
    assert p.segment_rows() == [
        ["provisioning", 90.0, 0.0],
        ["training", 60.0, 90.0],
        ["final_saves", 15.0, 150.0],
    ]


def test_segments_no_samples_is_provisioning():
    # A run that never reached the first training step reads as all provisioning.
    p = RunProfile("baseline-1", "baseline", "on-demand")
    p.events = [Event("launch", 1000.0, 1), Event("timeout", 1120.0, 1)]
    assert p.segments() == [{"phase": "provisioning", "seconds": 120.0}]


def test_wandb_disabled_is_noop():
    p = RunProfile("baseline-1", "baseline", "on-demand")
    cfg = types.SimpleNamespace(wandb_enabled=lambda: False)
    p.wandb_start(cfg)  # must not import wandb or raise
    assert p._wb is None
    p.ingest_log(LOG)  # _wb_log_step must no-op
    p._wb_finish()  # must no-op
