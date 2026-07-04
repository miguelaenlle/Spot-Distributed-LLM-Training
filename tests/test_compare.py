"""Compare-report tests — hermetic (fixture profiles, no AWS)."""

from __future__ import annotations

import types

from orchestrator import compare


def _profile(run_id: str, kind: str) -> dict:
    return {
        "run_id": run_id,
        "kind": kind,
        "durations": {"training_s": 300.0, "total_s": 400.0, "downtime_s": 12.0},
        "segments": [
            {"phase": "provisioning", "seconds": 60.0},
            {"phase": "training", "seconds": 300.0},
        ],
        "loss_samples": [{"step": 10, "loss": 3.0, "t_rel": 61.0}],
        "val_samples": [{"step": 250, "loss": 2.0, "t_rel": 200.0}],
        "text_samples": [
            {
                "step": 1000,
                "samples": [{"prompt": "ROMEO:", "sample_index": 0, "completion": " away"}],
            },
            {
                "step": 5000,
                "samples": [{"prompt": "ROMEO:", "sample_index": 0, "completion": " my lord"}],
            },
        ],
        "metrics": {
            "steps": 5000,
            "train_loss": 1.2,
            "val_loss": 1.47,
            "resumed": kind != "baseline",
            "world_size": 2,
        },
    }


def _cfg():
    return types.SimpleNamespace(instance_type="g4dn.xlarge")


def test_report_md_table_and_samples():
    profiles = [_profile("baseline-1", "baseline"), _profile("mp-1", "multinode-preempt")]
    md = compare._report_md(_cfg(), profiles, ["loss.png"])
    # one table row per run, with metrics + durations + a cost estimate
    assert "| baseline-1 | baseline | 5000 | 1.2000 | 1.4700 |" in md
    assert "| mp-1 | multinode-preempt |" in md
    assert "$0.117" in md  # 400s * 2 nodes * $0.526/hr
    assert "![loss.png](loss.png)" in md
    # per-prompt sections show snapshot progression per run
    assert md.count("### baseline-1") == 1
    assert "**step 1000:**" in md and "**step 5000:**" in md
    assert "ROMEO: my lord" in md


def test_cost_estimate_unknown_type_or_missing_total():
    p = _profile("r", "baseline")
    assert compare._cost_estimate(types.SimpleNamespace(instance_type="weird.type"), p) is None
    p["durations"] = {}
    assert compare._cost_estimate(_cfg(), p) is None
