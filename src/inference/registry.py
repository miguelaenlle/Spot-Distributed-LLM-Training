"""Worker registry — heartbeat documents in the object store (S3 or local dir).

Workers overwrite ``<workers_uri>/<worker_id>.json`` every few seconds; the
router lists the prefix and treats a worker as live while its ``last_seen`` is
within the TTL. S3-as-transport, same convention as checkpoints/rdzv.json — no
new IAM permissions, and the whole fleet state is inspectable with `aws s3 ls`.

Clocks: heartbeats use the writer's wall clock; EC2 boxes are NTP-synced, and
the TTL (~3x the heartbeat interval) absorbs normal skew.
"""

from __future__ import annotations

import json
import os
import time

from spot_train import s3_store

DEFAULT_TTL_SECONDS = 15.0


def worker_doc(
    worker_id: str,
    addr: str,
    *,
    model: str = "",
    market: str = "local",
    requests_served: int = 0,
    extra: dict | None = None,
) -> dict:
    doc = {
        "worker_id": worker_id,
        "addr": addr,  # "host:port" the router dials
        "model": model,
        "market": market,
        "requests_served": requests_served,
        "last_seen": time.time(),
    }
    if extra:
        doc.update(extra)
    return doc


def _doc_uri(workers_uri: str, worker_id: str) -> str:
    return workers_uri.rstrip("/") + f"/{worker_id}.json"


def put_worker(workers_uri: str, doc: dict) -> None:
    """Write (overwrite) a worker's heartbeat document."""
    s3_store.put_bytes(json.dumps(doc).encode(), _doc_uri(workers_uri, doc["worker_id"]))


def remove_worker(workers_uri: str, worker_id: str) -> None:
    """Best-effort delete on graceful shutdown (a TTL expiry covers hard kills)."""
    uri = _doc_uri(workers_uri, worker_id)
    try:
        if s3_store.is_s3(uri):
            import boto3  # lazy, mirrors s3_store

            bucket, _, key = uri[len("s3://") :].partition("/")
            boto3.client("s3").delete_object(Bucket=bucket, Key=key)
        elif os.path.exists(uri):
            os.remove(uri)
    except Exception:
        pass  # heartbeat TTL is the real liveness mechanism


def list_workers(workers_uri: str) -> list[dict]:
    """All heartbeat documents under the prefix (live and stale alike)."""
    docs: list[dict] = []
    if s3_store.is_s3(workers_uri):
        import boto3  # lazy, mirrors s3_store

        bucket, _, prefix = workers_uri[len("s3://") :].partition("/")
        c = boto3.client("s3")
        paginator = c.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith(".json"):
                    continue
                try:
                    body = c.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                    docs.append(json.loads(body))
                except Exception:
                    continue  # torn write / non-doc object: skip, next poll heals
    else:
        if not os.path.isdir(workers_uri):
            return []
        for name in sorted(os.listdir(workers_uri)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(workers_uri, name)) as f:
                    docs.append(json.load(f))
            except Exception:
                continue
    return docs


def live_workers(
    docs: list[dict], ttl_seconds: float = DEFAULT_TTL_SECONDS, now: float | None = None
) -> list[dict]:
    """Filter to workers whose heartbeat is fresher than the TTL."""
    now = time.time() if now is None else now
    return [d for d in docs if now - float(d.get("last_seen", 0)) <= ttl_seconds and d.get("addr")]
