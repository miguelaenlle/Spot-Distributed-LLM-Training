"""Registry: heartbeat docs in a local dir behave like the S3 prefix will."""

import time

from inference import registry


def test_put_then_list_roundtrip(tmp_path):
    uri = str(tmp_path / "workers")
    doc = registry.worker_doc("w0", "127.0.0.1:8001", model="run@step5", requests_served=3)
    registry.put_worker(uri, doc)

    docs = registry.list_workers(uri)
    assert len(docs) == 1
    assert docs[0]["worker_id"] == "w0"
    assert docs[0]["addr"] == "127.0.0.1:8001"
    assert docs[0]["requests_served"] == 3


def test_heartbeat_overwrites_not_appends(tmp_path):
    uri = str(tmp_path / "workers")
    for served in (1, 2, 3):
        registry.put_worker(
            uri, registry.worker_doc("w0", "127.0.0.1:8001", requests_served=served)
        )
    docs = registry.list_workers(uri)
    assert len(docs) == 1
    assert docs[0]["requests_served"] == 3


def test_live_filter_drops_stale_workers():
    now = time.time()
    fresh = registry.worker_doc("fresh", "a:1")
    stale = registry.worker_doc("stale", "b:2")
    stale["last_seen"] = now - 60
    live = registry.live_workers([fresh, stale], ttl_seconds=15, now=now)
    assert [d["worker_id"] for d in live] == ["fresh"]


def test_live_filter_requires_addr():
    doc = registry.worker_doc("w0", "")
    assert registry.live_workers([doc], ttl_seconds=15) == []


def test_list_skips_torn_docs(tmp_path):
    uri = tmp_path / "workers"
    uri.mkdir()
    (uri / "bad.json").write_text("{not json")
    registry.put_worker(str(uri), registry.worker_doc("w0", "a:1"))
    docs = registry.list_workers(str(uri))
    assert [d["worker_id"] for d in docs] == ["w0"]


def test_remove_worker(tmp_path):
    uri = str(tmp_path / "workers")
    registry.put_worker(uri, registry.worker_doc("w0", "a:1"))
    registry.remove_worker(uri, "w0")
    assert registry.list_workers(uri) == []


def test_missing_dir_lists_empty(tmp_path):
    assert registry.list_workers(str(tmp_path / "nope")) == []
