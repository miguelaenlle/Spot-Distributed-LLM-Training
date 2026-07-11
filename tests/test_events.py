"""Event emit/parse — the source-stamped records the event-sourced timeline is
built from. Pure stdlib, no AWS."""

from __future__ import annotations

import io
import json

from spot_train import events


def test_emit_roundtrips_through_parse():
    buf = io.StringIO()
    rec = events.emit(
        "training", by="trainer", node=0, epoch=3, world=2, step=806, ts=1783746952.4, stream=buf
    )
    line = buf.getvalue()
    assert line.startswith("[event] ")
    parsed = events.parse(line)
    assert parsed == [rec]
    assert parsed[0] == {
        "ts": 1783746952.4,
        "node": 0,
        "state": "training",
        "epoch": 3,
        "world": 2,
        "step": 806,
        "by": "trainer",
    }


def test_emit_omits_none_fields_and_defaults_ts():
    buf = io.StringIO()
    rec = events.emit("provisioning", by="sidecar", node=1, stream=buf)
    assert "epoch" not in rec and "world" not in rec and "cause" not in rec
    assert "ts" in rec and isinstance(rec["ts"], float)  # stamped now


def test_parse_skips_noise_and_malformed():
    text = "\n".join(
        [
            "step 20: loss 2.9, 80ms/step, 15300 tok/s, ws 2",  # a normal log line
            "[event] {not json}",  # malformed
            '[event] {"state": "training"}',  # no ts
            '[event] {"ts": "nope", "state": "x"}',  # non-numeric ts
            '[event] {"ts": 5.0, "state": "down", "node": 1, "cause": "reclaimed"}',  # good
            'prefix junk [event] {"ts": 6.0, "state": "training", "node": 0}',  # embedded, ok
        ]
    )
    got = events.parse(text)
    assert [r["ts"] for r in got] == [5.0, 6.0]
    assert got[0]["state"] == "down" and got[0]["node"] == 1


def test_parse_default_node_attribution():
    # A record missing node is attributed to the log it came from.
    line = '[event] {"ts": 1.0, "state": "training"}'
    assert events.parse(line, default_node=2)[0]["node"] == 2
    # An explicit node wins over the default.
    line2 = '[event] {"ts": 1.0, "state": "training", "node": 5}'
    assert events.parse(line2, default_node=2)[0]["node"] == 5


def test_emit_line_is_single_json_object():
    buf = io.StringIO()
    events.emit("stalled", by="trainer", node=0, ts=1.0, cause="peer-stall", stream=buf)
    payload = buf.getvalue()[len("[event] ") :].strip()
    assert json.loads(payload)["state"] == "stalled"  # exactly one JSON doc per line
    assert "\n" not in payload
