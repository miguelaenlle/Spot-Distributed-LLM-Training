"""Object store with atomic writes — local filesystem or S3, one interface.

Both backends share a code path so the CPU test and the spot box use the same
trainer. A URI is either a local path/dir or an ``s3://bucket/prefix/`` string.

Atomicity is the whole point: write to a temp key, then rename. A kill during
the write must never corrupt the last good checkpoint. On a local FS that's
``os.replace`` (atomic on POSIX). On S3 there is no true rename, so we upload to
``<key>.tmp`` then server-side copy to ``<key>`` then delete the ``.tmp`` — a
reader sees either the old complete object or the new one, never a partial.

Integrity: S3 uploads request a SHA-256 checksum (``ChecksumAlgorithm=SHA256``)
which S3 validates on write; downloads request ``ChecksumMode=ENABLED`` so the
client validates the bytes on read. No credentials are referenced here — boto3
resolves them from the ambient environment/instance profile at call time, and
``boto3`` is imported lazily so the local path never needs it installed.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile

CHECKPOINT_PREFIX = "ckpt-"
_TMP_SUFFIX = ".tmp"


def is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


def _split(uri: str) -> tuple[str, str]:
    """s3://bucket/a/b -> ("bucket", "a/b")."""
    rest = uri[len("s3://") :]
    bucket, _, key = rest.partition("/")
    return bucket, key


def _join(prefix_key: str, name: str) -> str:
    if not prefix_key:
        return name
    return prefix_key.rstrip("/") + "/" + name


def _client():
    import boto3  # lazy: only needed on the S3 path

    return boto3.client("s3")


# --------------------------------------------------------------------------- #
# Local filesystem backend
# --------------------------------------------------------------------------- #
def _local_save(local_path: str, dest_dir: str, name: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    tmp = os.path.join(dest_dir, name + _TMP_SUFFIX)
    final = os.path.join(dest_dir, name)
    os.replace(local_path, tmp)  # caller's bytes -> temp
    os.replace(tmp, final)  # atomic on POSIX
    return final


def _local_latest(dest_dir: str) -> str | None:
    if not os.path.isdir(dest_dir):
        return None
    cks = sorted(
        f
        for f in os.listdir(dest_dir)
        if f.startswith(CHECKPOINT_PREFIX) and not f.endswith(_TMP_SUFFIX)
    )
    return os.path.join(dest_dir, cks[-1]) if cks else None


# --------------------------------------------------------------------------- #
# S3 backend
# --------------------------------------------------------------------------- #
def _s3_save(local_path: str, uri: str, name: str) -> str:
    bucket, prefix = _split(uri)
    key = _join(prefix, name)
    tmp_key = key + _TMP_SUFFIX
    c = _client()
    c.upload_file(local_path, bucket, tmp_key, ExtraArgs={"ChecksumAlgorithm": "SHA256"})
    c.copy(
        {"Bucket": bucket, "Key": tmp_key}, bucket, key, ExtraArgs={"ChecksumAlgorithm": "SHA256"}
    )
    c.delete_object(Bucket=bucket, Key=tmp_key)
    return f"s3://{bucket}/{key}"


def _s3_latest(uri: str) -> str | None:
    bucket, prefix = _split(uri)
    c = _client()
    best: str | None = None
    paginator = c.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            base = obj["Key"].rsplit("/", 1)[-1]
            is_ckpt = base.startswith(CHECKPOINT_PREFIX) and not base.endswith(_TMP_SUFFIX)
            if is_ckpt and (best is None or obj["Key"] > best):
                best = obj["Key"]
    return f"s3://{bucket}/{best}" if best else None


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #
def save_atomic(local_path: str, uri: str, name: str) -> str:
    """Move bytes at ``local_path`` into the store atomically. Returns final ref."""
    return _s3_save(local_path, uri, name) if is_s3(uri) else _local_save(local_path, uri, name)


def latest(uri: str) -> str | None:
    """Return a ref to the newest checkpoint under ``uri``, or None if empty."""
    return _s3_latest(uri) if is_s3(uri) else _local_latest(uri)


def ref_for(uri: str, name: str) -> str:
    """The ref of object ``name`` under prefix/dir ``uri`` (existence not checked)."""
    if is_s3(uri):
        return uri.rstrip("/") + "/" + name
    return os.path.join(uri, name)


def download(ref: str, verify: bool = True) -> str:
    """Return a local path for ``ref``. For S3, download to a temp file (and let
    S3/boto3 validate the SHA-256 when ``verify``). For local, ``ref`` already is
    a path, so return it unchanged.

    The caller OWNS the returned temp file and must remove (or move) it — a
    30-second checkpoint loop that forgets will fill the disk. Prefer
    :func:`fetch` unless you are taking ownership of the bytes."""
    if not is_s3(ref):
        return ref
    bucket, key = _split(ref)
    fd, local = tempfile.mkstemp(suffix="-" + key.rsplit("/", 1)[-1])
    os.close(fd)
    extra = {"ChecksumMode": "ENABLED"} if verify else {}
    _client().download_file(bucket, key, local, ExtraArgs=extra)
    return local


@contextlib.contextmanager
def fetch(ref: str, verify: bool = True):
    """Context manager around :func:`download` that deletes the temp copy on
    exit. Local refs are yielded unchanged and never deleted."""
    local = download(ref, verify=verify)
    try:
        yield local
    finally:
        if local != ref:
            with contextlib.suppress(OSError):
                os.remove(local)


def put_file(local_path: str, uri: str) -> None:
    """Upload/copy a single file to a full destination URI (not atomic — for
    read-only artifacts like dataset bins and metrics.json)."""
    if is_s3(uri):
        bucket, key = _split(uri)
        _client().upload_file(local_path, bucket, key, ExtraArgs={"ChecksumAlgorithm": "SHA256"})
    else:
        os.makedirs(os.path.dirname(uri) or ".", exist_ok=True)
        shutil.copyfile(local_path, uri)


def put_bytes(data: bytes, uri: str) -> None:
    if is_s3(uri):
        bucket, key = _split(uri)
        _client().put_object(Bucket=bucket, Key=key, Body=data, ChecksumAlgorithm="SHA256")
    else:
        os.makedirs(os.path.dirname(uri) or ".", exist_ok=True)
        with open(uri, "wb") as f:
            f.write(data)


def read_bytes(uri: str) -> bytes | None:
    """Return the bytes at ``uri`` (S3 object or local file), or None if it does
    not exist. The read side of :func:`put_bytes` — used for the small control
    docs the epoch protocol polls (epoch.json, node<i>.json). No checksum
    verification: these are tiny JSON docs rewritten in place, not checkpoints."""
    if is_s3(uri):
        bucket, key = _split(uri)
        try:
            return _client().get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception:  # noqa: BLE001 — NoSuchKey (and any transient error) => absent
            return None
    try:
        with open(uri, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None


def exists(uri: str) -> bool:
    if is_s3(uri):
        bucket, key = _split(uri)
        try:
            _client().head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False
    return os.path.exists(uri)
