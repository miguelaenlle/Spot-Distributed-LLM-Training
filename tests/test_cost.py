"""Cost-ledger tests: per-second per-instance billing math, the profile.json
``cost`` schema, the cost curve, the rendered graph, and compare's preference
for recorded cost over the wallclock estimate. All hermetic — rows are opened
and closed with explicit timestamps; no AWS."""

import os

import pytest

from orchestrator import compare
from orchestrator.config import OrchestratorConfig
from orchestrator.profile import RunProfile, ValSample


def _p(kind: str = "preempt", market: str = "spot") -> RunProfile:
    return RunProfile(f"{kind}-1", kind, market)


def test_two_closed_instances_sum_per_second():
    p = _p()
    p.instance_started("i-1", "spot", "us-east-1a", 0.18, t=1000.0)
    p.instance_stopped("i-1", t=1000.0 + 3600)
    p.instance_started("i-2", "spot", "us-east-1b", 0.36, t=6000.0)
    p.instance_stopped("i-2", t=6000.0 + 1800)
    assert p.cost_at(99_999.0) == pytest.approx(0.18 + 0.18)
    d = p.cost_dict()
    assert d["total_usd"] == pytest.approx(0.36)
    assert [r["usd"] for r in d["instances"]] == [pytest.approx(0.18), pytest.approx(0.18)]
    assert d["rate_unknown_count"] == 0


def test_open_row_accrues_to_query_time_only():
    p = _p()
    p.instance_started("i-1", "spot", "us-east-1a", 0.36, t=0.0)
    assert p.cost_at(1800.0) == pytest.approx(0.18)
    assert p.cost_at(3600.0) == pytest.approx(0.36)
    # never negative before the box started
    assert p.cost_at(-5.0) == 0.0


def test_unknown_rate_contributes_zero_and_is_flagged():
    p = _p("baseline", "on-demand")
    p.instance_started("i-1", "on-demand", "us-east-1a", None, t=0.0)
    p.instance_stopped("i-1", t=3600.0)
    assert p.cost_at(3600.0) == 0.0
    d = p.cost_dict()
    assert d["rate_unknown_count"] == 1
    assert d["instances"][0]["usd"] is None
    assert d["instances"][0]["billed_seconds"] == 3600.0


def test_stop_is_idempotent_and_ignores_unknown_ids():
    p = _p()
    p.instance_started("i-1", "spot", "us-east-1a", 0.5, t=0.0)
    p.instance_stopped("i-1", t=100.0)
    p.instance_stopped("i-1", t=999.0)  # second stop must not move the stamp
    p.instance_stopped("i-nope", t=50.0)  # unknown id: no-op, no raise
    assert p.instances[0].stopped_at == 100.0


def test_close_open_instances_stamps_only_open_rows():
    p = _p()
    p.instance_started("i-1", "spot", "us-east-1a", 0.5, t=0.0)
    p.instance_stopped("i-1", t=60.0)
    p.instance_started("i-2", "spot", "us-east-1a", 0.5, t=30.0)
    p.close_open_instances()
    assert p.instances[0].stopped_at == 60.0
    assert p.instances[1].stopped_at is not None


def test_profile_json_cost_schema():
    p = _p()
    p.instance_started("i-1", "spot", "us-east-1a", 0.5, t=0.0)
    p.instance_stopped("i-1", t=60.0)
    cost = p.to_dict()["cost"]
    row = cost["instances"][0]
    assert set(row) == {
        "instance_id",
        "market",
        "az",
        "hourly_usd",
        "started_at",
        "stopped_at",
        "billed_seconds",
        "usd",
    }
    assert row["billed_seconds"] == 60.0
    assert cost["total_usd"] == pytest.approx(60 * 0.5 / 3600, abs=1e-5)


def test_no_instances_means_no_cost_section_and_no_rows():
    p = _p()
    assert p.to_dict()["cost"] is None
    assert p.cost_rows() == []
    assert p.cost_curve() == ([], [])
    assert p.hourly_rate_now() is None


def test_cost_curve_hits_every_breakpoint():
    p = _p()
    p.instance_started("i-1", "spot", "us-east-1a", 0.36, t=100.0)
    p.instance_stopped("i-1", t=1900.0)
    p.instance_started("i-2", "spot", "us-east-1b", 0.36, t=1000.0)
    p.instance_stopped("i-2", t=2800.0)
    xs, ys = p.cost_curve()
    assert xs[0] == 0.0 and ys[0] == 0.0  # relative to the earliest start
    assert xs[-1] == 2700.0  # 2800 - 100
    # 1800s + 1800s at $0.36/hr
    assert ys[-1] == pytest.approx(0.36)
    # monotone non-decreasing
    assert all(b >= a for a, b in zip(ys, ys[1:], strict=False))


def test_hourly_rate_now_tracks_newest_open_row():
    p = _p()
    p.instance_started("i-1", "spot", "us-east-1a", 0.18, t=0.0)
    p.instance_stopped("i-1", t=10.0)
    p.instance_started("i-2", "spot", "us-east-1b", 0.44, t=20.0)
    assert p.hourly_rate_now() == 0.44


def test_render_cost_png_writes_file(tmp_path):
    pytest.importorskip("matplotlib")
    p = _p()
    p.instance_started("i-1", "spot", "us-east-1a", 0.36, t=1000.0)
    p.instance_stopped("i-1", t=1300.0)
    p.instance_started("i-2", "spot", "us-east-1a", 0.36, t=1400.0)
    p.instance_stopped("i-2", t=1700.0)
    p.val_samples.append(ValSample(step=100, loss=2.0, t_rel=200.0))
    p.val_samples.append(ValSample(step=200, loss=1.5, t_rel=600.0))
    out = tmp_path / "cost.png"
    assert p.render_cost_png(str(out))
    assert out.exists() and os.path.getsize(out) > 0


def test_render_cost_png_refuses_without_rates(tmp_path):
    p = _p()
    p.instance_started("i-1", "on-demand", "us-east-1a", None, t=0.0)
    assert not p.render_cost_png(str(tmp_path / "cost.png"))


def test_compare_prefers_recorded_cost_over_estimate():
    cfg = OrchestratorConfig()
    recorded = {"cost": {"total_usd": 1.23}, "durations": {"total_s": 999999}}
    assert compare._cost_estimate(cfg, recorded) == 1.23
    # no ledger -> falls back to wallclock x rate (g4dn.xlarge default type)
    legacy = {"durations": {"total_s": 3600}, "metrics": {}}
    est = compare._cost_estimate(cfg, legacy)
    assert est == pytest.approx(0.526, abs=0.01)
