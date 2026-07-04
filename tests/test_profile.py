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


def test_segments_from_trainer_stamps():
    # Exact wall-clock phases from the trainer take precedence over the per-step
    # proxy: training reads the real 60s loop, and eval is split from saves.
    p = RunProfile("baseline-1", "baseline", "on-demand")
    p.events = [Event("launch", 1000.0, 1), Event("metrics", 1170.0, 1)]
    p.from_metrics(
        {
            "train_started_at": 1060.0,  # provisioning = 60
            "phases": {"train_s": 60.0, "save_s": 2.0, "eval_s": 44.0},
        }
    )
    assert p.segments() == [
        {"phase": "provisioning", "seconds": 60.0},
        {"phase": "training", "seconds": 60.0},
        {"phase": "final_saves", "seconds": 2.0},
        {"phase": "evaluation", "seconds": 44.0},
    ]
    # durations() sums those + total_s from the launch->metrics span
    d = p.durations()
    assert d["training_s"] == 60.0
    assert d["evaluation_s"] == 44.0
    assert d["total_s"] == 170.0


def test_segments_preempt_multi_segment():
    # 3-segment preemption: provisioning/training/downtime/recovery repeat; the
    # final training block is split via the trainer's stamps.
    p = RunProfile("preempt-1", "preempt", "spot")
    p.events = [
        Event("launch", 0.0, 1),
        Event("first_log", 40.0, 1),  # ignored by the mark walk
        Event("train_start", 45.0, 1),
        Event("kill", 65.0, 1),  # seg1 training 20
        Event("relaunch", 70.0, 2),  # downtime 5
        Event("train_start", 115.0, 2),  # recovery 45
        Event("kill", 135.0, 2),  # seg2 training 20
        Event("relaunch", 140.0, 3),  # downtime 5
        Event("train_start", 185.0, 3),  # recovery 45
        Event("metrics", 250.0, 3),  # final training block -> split via stamps
    ]
    p.from_metrics({"phases": {"train_s": 20.0, "save_s": 1.0, "eval_s": 44.0}})
    assert [s["phase"] for s in p.segments()] == [
        "provisioning",
        "training",
        "downtime",
        "preemption_recovery",
        "training",
        "downtime",
        "preemption_recovery",
        "training",
        "final_saves",
        "evaluation",
    ]
    secs = [s["seconds"] for s in p.segments()]
    assert secs[:8] == [45.0, 20.0, 5.0, 45.0, 20.0, 5.0, 45.0, 20.0]
    assert secs[8:] == [1.0, 44.0]  # from stamps


def test_wandb_disabled_is_noop():
    p = RunProfile("baseline-1", "baseline", "on-demand")
    cfg = types.SimpleNamespace(wandb_enabled=lambda: False)
    p.wandb_start(cfg)  # must not import wandb or raise
    assert p._wb is None
    p.ingest_log(LOG)  # _wb_log_step must no-op
    p._wb_finish()  # must no-op


# ---- val-eval lines, t_rel stamps, and text samples (Gaps C/E) ------------- #

VAL_LOG = LOG + "eval step 250: val_loss 2.1034\neval step 500: val_loss 1.9020\n"


def test_ingest_val_lines_and_dedup():
    p = RunProfile("baseline-1", "baseline", "on-demand")
    p.ingest_log(VAL_LOG)
    assert [(v.step, v.loss) for v in p.val_samples] == [(250, 2.1034), (500, 1.902)]
    # re-feeding the full log must not duplicate; train samples unaffected
    p.ingest_log(VAL_LOG)
    assert len(p.val_samples) == 2 and len(p.samples) == 2
    # a truncated val line is ignored until complete
    p.ingest_log("eval step 750: val_loss 1.8")
    assert len(p.val_samples) == 3  # 1.8 parses — the regex needs digits, has them
    assert p.val_samples[-1].loss == 1.8


def test_samples_t_rel_is_relative_to_first_event():
    import time as _time

    p = RunProfile("baseline-1", "baseline", "on-demand")
    p.events = [Event("launch", _time.time() - 100.0, 1)]
    p.ingest_log(VAL_LOG)
    assert all(99.0 < s.t_rel < 102.0 for s in p.samples)
    assert all(99.0 < v.t_rel < 102.0 for v in p.val_samples)
    # without any event the stamp degrades to 0, never raises
    q = RunProfile("baseline-2", "baseline", "on-demand")
    q.ingest_log(LOG)
    assert all(s.t_rel == 0.0 for s in q.samples)


def _samples_doc(step: int) -> dict:
    return {
        "run_id": "r",
        "step": step,
        "dataset": "shakespeare_char",
        "params": {"max_new_tokens": 8, "temperature": 0.8},
        "samples": [{"prompt": "ROMEO:", "sample_index": 0, "completion": " hi"}],
    }


def test_from_samples_dedup_and_order():
    p = RunProfile("baseline-1", "baseline", "on-demand")
    p.from_samples(_samples_doc(2000))
    p.from_samples(_samples_doc(1000))
    p.from_samples(_samples_doc(2000))  # resumed-run rewrite: first-seen wins
    p.from_samples(None)  # tolerated
    p.from_samples({"step": 3, "samples": []})  # empty doc ignored
    assert [d["step"] for d in p.text_samples] == [1000, 2000]
    assert p.sample_rows() == [
        [1000, "ROMEO:", 0, " hi"],
        [2000, "ROMEO:", 0, " hi"],
    ]


def test_to_dict_includes_new_fields(tmp_path):
    p = RunProfile("baseline-1", "baseline", "on-demand")
    p.events = [Event("launch", 1000.0, 1)]
    p.ingest_log(VAL_LOG)
    p.from_samples(_samples_doc(500))
    d = p.to_dict()
    assert [v["step"] for v in d["val_samples"]] == [250, 500]
    assert d["text_samples"][0]["step"] == 500
    assert "t_rel" in d["loss_samples"][0]
    assert isinstance(d["segments"], list)  # materialized for `compare`
    json.dumps(d)  # stays serializable


def test_segments_include_sampling_stamp():
    p = RunProfile("baseline-1", "baseline", "on-demand")
    p.events = [Event("launch", 1000.0, 1), Event("metrics", 1175.0, 1)]
    p.from_metrics(
        {
            "train_started_at": 1060.0,
            "phases": {"train_s": 60.0, "save_s": 2.0, "eval_s": 44.0, "sample_s": 9.0},
        }
    )
    assert p.segments()[-1] == {"phase": "sampling", "seconds": 9.0}
    assert p.durations()["sampling_s"] == 9.0
    # old metrics.json without sample_s stays untouched
    p.from_metrics({"train_started_at": 1060.0, "phases": {"train_s": 60.0}})
    assert all(s["phase"] != "sampling" for s in p.segments())


def test_render_multi_timeline(tmp_path):
    from orchestrator.profile import render_multi_timeline_png

    rows = [
        (
            "baseline",
            [{"phase": "provisioning", "seconds": 60}, {"phase": "training", "seconds": 300}],
        ),
        ("preempt", [{"phase": "training", "seconds": 100}, {"phase": "downtime", "seconds": 30}]),
        ("empty", []),  # dropped, not drawn
    ]
    out = tmp_path / "tl.png"
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        assert render_multi_timeline_png("t", rows, str(out)) is False
        return
    assert render_multi_timeline_png("t", rows, str(out)) is True
    assert out.stat().st_size > 0
    assert render_multi_timeline_png("t", [("e", [])], str(out)) is False
