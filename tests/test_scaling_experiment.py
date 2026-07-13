"""Scaling experiment — the pure analysis + report + calibration pieces (no AWS)."""

from __future__ import annotations

import pytest

from orchestrator import experiments, logview
from orchestrator.profile import RunProfile, Sample, ValSample


def _profile_with_curve(run_id, val_by_step, step_walls):
    """A RunProfile carrying a synthetic val curve + per-step wall-clocks."""
    p = RunProfile(run_id, kind="multinode", market="spot")
    p.val_samples = [ValSample(step=s, loss=v) for s, v in val_by_step]
    p.samples = [
        Sample(step=s, loss=1.0, ms_per_step=80, tok_s=1000, t_rel=w) for s, w in step_walls
    ]
    return p


def test_analyze_target_time_to_first_crossing():
    # val descends past 1.6 at step 300; first training step at t_rel=10s; step 300
    # first reached at t_rel=40s -> 30s to target.
    val = [(100, 2.0), (200, 1.7), (300, 1.55), (400, 1.45)]
    walls = [(0, 10.0), (100, 20.0), (200, 30.0), (300, 40.0), (400, 50.0)]
    a = experiments._analyze_target(_profile_with_curve("r", val, walls), target=1.6)
    assert a["reached"] is True
    assert a["target_step"] == 300 and a["hit_val"] == 1.55
    assert a["time_to_target_s"] == 30.0 and a["steps_to_target"] == 300


def test_analyze_target_not_reached():
    val = [(100, 2.0), (200, 1.9), (300, 1.85)]  # never gets to 1.5
    walls = [(0, 5.0), (100, 10.0), (200, 15.0), (300, 20.0)]
    a = experiments._analyze_target(_profile_with_curve("r", val, walls), target=1.5)
    assert a["reached"] is False and a["best_val"] == 1.85


def test_report_verdicts_true_false_and_inconclusive(tmp_path):
    def result(label, nodes, preempt, t, reached=True):
        return {
            "label": label,
            "nodes": nodes,
            "preempt": preempt,
            "run_id": f"{label}-id",
            "analysis": {
                "reached": reached,
                "target": 3.5,
                "target_step": 800,
                "hit_val": 3.49,
                "time_to_target_s": t,
                "total_train_s": t + 100,
            },
            "cost": 0.5,
            "wandb": None,
            "png": "a.png",
            "events": "a.txt",
            "valcurve": "v.png",
        }

    recipe = {
        "stamp": "x",
        "target": 3.5,
        "global_batch": "64",
        "market": "spot",
        "model": "12L-768d-1024ctx",
        "dataset": "openwebtext_300m",
        "eval_interval": "50",
        "cap_s": 1800,
        "offsets": "600,1200",
    }
    results = [
        result("2n-clean", 2, False, 900.0),
        result("4n-clean", 4, False, 500.0),  # H1 TRUE
        result("2n-preempt", 2, True, 1000.0),
        result("4n-preempt", 4, True, 1200.0),  # H2 FALSE
    ]
    path = str(tmp_path / "summary.txt")
    experiments._write_scaling_report(path, results, recipe)
    with open(path) as f:
        body = f.read()
    assert "H1 (clean): TRUE" in body and "1.80x speedup" in body
    assert "H2 (preempt): FALSE" in body
    assert "target val_loss <= 3.5" in body and "run_id=4n-clean-id" in body

    results[1]["analysis"]["reached"] = False  # a run that missed target
    experiments._write_scaling_report(path, results, recipe)
    with open(path) as f:
        assert "H1 (clean): INCONCLUSIVE" in f.read()


def _clean_result(label, nodes, t, reached=True, steps=800, ms=None):
    return {
        "label": label,
        "nodes": nodes,
        "run_id": f"{label}-id",
        "instance": "g5.xlarge",
        "market": "spot",
        "ms_per_step": ms if ms is not None else round(2000 / nodes, 1),
        "tok_per_s": 30000 * nodes,
        "analysis": {
            "reached": reached,
            "target": 5.0,
            "target_step": steps,
            "hit_val": 4.98,
            "steps_to_target": steps,
            "time_to_target_s": t,
            "total_train_s": t + 30,
        },
        "cost": 0.3,
        "wandb": None,
        "png": "g.png",
        "events": "e.txt",
        "valcurve": "v.png",
    }


_CLEAN_RECIPE = {
    "stamp": "x",
    "target": 5.0,
    "market": "spot",
    "instance": "g5.xlarge",
    "model": "12L-768d-1024ctx",
    "dataset": "openwebtext_300m",
    "global_batch": "64",
    "eval_interval": "25",
    "cap_s": 480,
    "node_counts": "1,2,4",
}


def test_scaling_clean_report_speedup_and_efficiency(tmp_path):
    # 1n=400s baseline, 2n=220s (1.82x), 4n=130s (3.08x) — sub-linear, as expected.
    results = [
        _clean_result("1n", 1, 400.0),
        _clean_result("2n", 2, 220.0),
        _clean_result("4n", 4, 130.0),
    ]
    path = str(tmp_path / "summary.txt")
    experiments._write_scaling_clean_report(path, results, _CLEAN_RECIPE)
    with open(path) as f:
        body = f.read()
    assert "SPEEDUP vs 1 node(s)" in body
    assert "2n: 220.0s   1.82x vs 1n" in body
    assert "4n: 130.0s   3.08x vs 1n" in body
    assert "scaling efficiency" in body
    assert "run_id=4n-id" in body
    # per-run hardware + throughput + duration line
    assert "hardware: 4x g5.xlarge (spot)" in body
    assert "ms/step: 500.0" in body  # 2000/4
    assert "run time: 160.0s" in body  # total_train_s = 130 + 30
    # steps_to_target match across runs -> no control warning
    assert "CONTROL CHECK" not in body


def test_scaling_clean_report_flags_control_mismatch(tmp_path):
    # Different steps_to_target across node counts -> constant-batch control broke.
    results = [
        _clean_result("1n", 1, 400.0, steps=800),
        _clean_result("2n", 2, 220.0, steps=825),  # different step -> flagged
    ]
    path = str(tmp_path / "summary.txt")
    experiments._write_scaling_clean_report(path, results, _CLEAN_RECIPE)
    with open(path) as f:
        assert "CONTROL CHECK" in f.read()


def test_scaling_clean_report_inconclusive_when_target_missed(tmp_path):
    results = [
        _clean_result("1n", 1, 400.0),
        _clean_result("2n", 2, 0.0, reached=False),
    ]
    path = str(tmp_path / "summary.txt")
    experiments._write_scaling_clean_report(path, results, _CLEAN_RECIPE)
    with open(path) as f:
        body = f.read()
    assert "2n: INCONCLUSIVE" in body


def test_scaling_clean_requires_target_loss(monkeypatch):
    from orchestrator.config import OrchestratorConfig

    monkeypatch.delenv("TARGET_LOSS", raising=False)
    with pytest.raises(SystemExit, match="TARGET_LOSS is required"):
        experiments.run_scaling_clean(OrchestratorConfig())


def test_scaling_clean_vcpu_guard(monkeypatch):
    from orchestrator.config import OrchestratorConfig

    monkeypatch.setenv("TARGET_LOSS", "5.0")
    monkeypatch.setenv("NODE_COUNTS", "1,2,4")
    # default vcpu_quota=8; 4 nodes x 4 vCPU (g4dn/g5.xlarge) = 16 > 8 -> guard trips
    cfg = OrchestratorConfig()
    assert cfg.instance_vcpu_count() == 4 and cfg.vcpu_quota == 8
    with pytest.raises(SystemExit, match="> VCPU_QUOTA="):
        experiments.run_scaling_clean(cfg)


def test_calibration_sizing_projects_and_suggests():
    p = RunProfile("cal", kind="calibrate", market="on-demand")
    # 200 ms/step single GPU -> 5 steps/s; a descending val curve for the log fit.
    p.samples = [
        Sample(step=s, loss=1.0, ms_per_step=200, tok_s=20000, t_rel=s * 0.2) for s in range(1, 200)
    ]
    p.val_samples = [ValSample(step=s, loss=6.0 - 0.4 * (s / 25)) for s in range(25, 200, 25)]
    z = experiments._calibration_sizing(p, cap_s=1800, global_batch=64, block=1024)
    assert z["ok"] and z["steps_per_s_1gpu"] == 5.0
    assert z["proj_steps_at_cap"][4] == int(5.0 * 4 * 0.85 * 1800)  # 4-node ~ 4x x 0.85 x cap
    assert z["proj_steps_at_cap"][2] < z["proj_steps_at_cap"][4]
    assert z["suggested_target_loss"] is not None


def test_calibration_sizing_too_short():
    p = RunProfile("cal", kind="calibrate", market="on-demand")
    p.samples = [Sample(step=1, loss=1.0, ms_per_step=200, tok_s=100, t_rel=0.0)]
    assert experiments._calibration_sizing(p, 1800, 64, 1024)["ok"] is False


def test_parse_run_events_attributes_by_filename():
    items = [
        ("orchestrator.log", '[event] {"ts": 5.0, "state": "epoch", "world": 2, "leader": 0}'),
        ("boot-node0.log", '[event] {"ts": 1.0, "state": "training"}'),
        ("boot-node1-r1.log", '[event] {"ts": 8.0, "state": "training"}'),
        ("not-a-log.txt", "noise"),
    ]
    by = {(r.get("node"), r.get("attempt"), r["state"]) for r in logview.parse_run_events(items)}
    assert (None, None, "epoch") in by
    assert (0, 0, "training") in by
    assert (1, 1, "training") in by
