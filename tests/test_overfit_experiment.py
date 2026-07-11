"""Overfit scaling experiment — the pure analysis + report pieces (no AWS)."""

from __future__ import annotations

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


def test_analyze_overfit_finds_val_minimum():
    # val bottoms at step 300 then RISES -> overfit at 300. First train step at
    # t_rel=10s (after boot); step 300 first reached at t_rel=40s -> 30s to overfit.
    val = [(100, 2.0), (200, 1.6), (300, 1.5), (400, 1.55), (500, 1.7)]
    walls = [(100, 20.0), (200, 30.0), (300, 40.0), (400, 50.0), (500, 60.0)]
    walls = [(0, 10.0), *walls]  # first training step at 10s
    a = experiments._analyze_overfit(_profile_with_curve("r", val, walls))
    assert a["reached"] is True
    assert a["overfit_step"] == 300 and a["best_val"] == 1.5
    assert a["time_to_overfit_s"] == 30.0  # 40s (reached step 300) - 10s (first train)
    assert a["steps_to_overfit"] == 300


def test_analyze_overfit_not_reached_when_still_descending():
    # Monotonically decreasing val -> min is the LAST point -> not a real overfit.
    val = [(100, 2.0), (200, 1.7), (300, 1.5)]
    walls = [(0, 5.0), (100, 10.0), (200, 15.0), (300, 20.0)]
    a = experiments._analyze_overfit(_profile_with_curve("r", val, walls))
    assert a["reached"] is False


def test_analyze_overfit_too_few_points():
    a = experiments._analyze_overfit(_profile_with_curve("r", [(1, 2.0)], [(0, 0.0)]))
    assert a["reached"] is False


def test_report_verdicts_true_false_and_inconclusive(tmp_path):
    def result(label, nodes, preempt, t, reached=True):
        return {
            "label": label,
            "nodes": nodes,
            "preempt": preempt,
            "run_id": f"{label}-id",
            "analysis": {
                "reached": reached,
                "overfit_step": 300,
                "best_val": 1.5,
                "time_to_overfit_s": t,
                "total_train_s": t + 50,
            },
            "cost": 0.5,
            "wandb": None,
            "png": "a.png",
            "events": "a.txt",
            "valcurve": "v.png",
        }

    recipe = {
        "stamp": "x",
        "global_batch": "64",
        "market": "spot",
        "max_steps": "5000",
        "eval_interval": "100",
        "dropout": "0.0",
        "offsets": "60,150",
    }
    # H1 true (4n faster), H2 false (4n slower under preemption), plus an inconclusive case
    results = [
        result("2n-clean", 2, False, 200.0),
        result("4n-clean", 4, False, 120.0),  # H1 TRUE
        result("2n-preempt", 2, True, 180.0),
        result("4n-preempt", 4, True, 220.0),  # H2 FALSE (slower)
    ]
    path = str(tmp_path / "summary.txt")
    experiments._write_overfit_report(path, results, recipe)
    with open(path) as f:
        body = f.read()
    assert "H1 (clean): TRUE" in body and "1.67x speedup" in body
    assert "H2 (preempt): FALSE" in body
    assert "wandb: (disabled)" in body and "run_id=4n-clean-id" in body

    # A run that didn't overfit -> that hypothesis is INCONCLUSIVE.
    results[1]["analysis"]["reached"] = False
    experiments._write_overfit_report(path, results, recipe)
    with open(path) as f:
        assert "H1 (clean): INCONCLUSIVE" in f.read()


def test_parse_run_events_attributes_by_filename():
    items = [
        ("orchestrator.log", '[event] {"ts": 5.0, "state": "epoch", "world": 2, "leader": 0}'),
        ("boot-node0.log", '[event] {"ts": 1.0, "state": "training"}'),  # -> node0, attempt0
        ("boot-node1-r1.log", '[event] {"ts": 8.0, "state": "training"}'),  # -> node1, attempt1
        ("not-a-log.txt", "noise"),
    ]
    recs = logview.parse_run_events(items)
    by = {(r.get("node"), r.get("attempt"), r["state"]) for r in recs}
    assert (None, None, "epoch") in by
    assert (0, 0, "training") in by
    assert (1, 1, "training") in by
