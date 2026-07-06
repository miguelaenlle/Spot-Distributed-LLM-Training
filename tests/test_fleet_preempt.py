"""fleet preempt analysis: window splits, recovery detection, verdicts."""

from orchestrator.fleet_preempt import analyze, render


def _bucket(t, sent=4, ok=4, errors=0, p99=200.0, mean=130.0):
    return {"t": t, "sent": sent, "ok": ok, "errors": errors, "p99_ms": p99, "mean_ms": mean}


def _report(buckets, failed=0, dropped=0):
    return {"per_second": buckets, "failed": failed, "dropped": dropped, "start_unix": 0.0}


def _clean_run(kill_at=30, end=90, blip_p99=1500.0, blip_len=5):
    """Steady 200ms p99, a latency blip (no errors) right after the kill."""
    buckets = []
    for t in range(end):
        p99 = blip_p99 if kill_at <= t < kill_at + blip_len else 200.0
        buckets.append(_bucket(t, p99=p99))
    return _report(buckets)


def test_windows_split_around_kill():
    a = analyze(_clean_run(kill_at=30, end=90), kill_rel=30.0, ttl=15.0)
    assert a["before"]["requests"] == 30 * 4
    assert a["disruption"]["requests"] == 15 * 4
    assert a["recovered"]["requests"] == 45 * 4
    assert a["disruption"]["p99_ms"] == 1500.0
    assert a["before"]["p99_ms"] == 200.0


def test_clean_reroute_passes_with_recovery_time():
    a = analyze(_clean_run(kill_at=30, blip_len=5), kill_rel=30.0, ttl=15.0)
    assert a["total_failed"] == 0
    # p99 back inside 2x pre-kill baseline at t=35, sustained 3s.
    assert a["recovery_s"] == 5.0
    assert a["passed"]


def test_client_visible_errors_fail():
    buckets = [_bucket(t) for t in range(60)]
    buckets[31] = _bucket(31, sent=4, ok=2, errors=2)
    a = analyze(_report(buckets, failed=2), kill_rel=30.0, ttl=15.0)
    assert a["disruption"]["errors"] == 2
    assert not a["passed"]


def test_no_recovery_fails():
    """p99 never returns to the pre-kill band => recovery not detected => FAIL."""
    buckets = [_bucket(t, p99=200.0 if t < 30 else 5000.0) for t in range(90)]
    a = analyze(_report(buckets), kill_rel=30.0, ttl=15.0)
    assert a["recovery_s"] is None
    assert not a["passed"]


def test_late_recovery_fails():
    """Recovering long after the heartbeat TTL means rerouting was too slow."""
    a = analyze(_clean_run(kill_at=30, end=120, blip_len=40), kill_rel=30.0, ttl=15.0)
    assert a["recovery_s"] == 40.0
    assert not a["passed"]  # > ttl + 10


def test_render_contains_verdict_and_windows():
    a = analyze(_clean_run(), kill_rel=30.0)
    text = render(a, rerouted=7)
    assert "RESULT: PASS" in text
    assert "rerouted 7" in text
    assert "before" in text and "disruption" in text and "recovered" in text
