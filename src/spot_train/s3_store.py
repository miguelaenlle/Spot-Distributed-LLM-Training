"""Checkpoint store with atomic writes.

Supports two backends behind one interface so the CPU test and the spot box
share a code path:

- local filesystem  ("checkpoints/")
- S3                ("s3://bucket/prefix/")

Atomicity is the whole point: write to a temp key, then rename. A kill during
the write must never corrupt the last good checkpoint. On a local FS this is
``os.replace`` (atomic on POSIX). On S3 it is upload-to-temp-key then
server-side copy + delete (S3 has no true rename, but a reader either sees the
old object or the new one — never a partial one).
"""

from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlparse

CHECKPOINT_PREFIX = "ckpt-"
_TMP_SUFFIX = ".tmp"


def is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


# --------------------------------------------------------------------------- #
# Local filesystem backend
# --------------------------------------------------------------------------- #
def _local_save(local_path: str, dest_dir: str, name: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    tmp = os.path.join(dest_dir, name + _TMP_SUFFIX)
    final = os.path.join(dest_dir, name)
    # caller has already written bytes to local_path; move into place atomically
    os.replace(local_path, tmp)
    os.replace(tmp, final)  # atomic on POSIX
    return final


def _local_latest(dest_dir: str) -> Optional[str]:
    if not os.path.isdir(dest_dir):
        return None
    cks = sorted(
        f for f in os.listdir(dest_dir)
        if f.startswith(CHECKPOINT_PREFIX) and not f.endswith(_TMP_SUFFIX)
    )
    return os.path.join(dest_dir, cks[-1]) if cks else None


# --------------------------------------------------------------------------- #
# S3 backend (implemented when we move off CPU — Phase 1a step 2)
# --------------------------------------------------------------------------- #
def _s3_save(local_path: str, uri: str, name: str) -> str:
    raise NotImplementedError(
        "S3 backend lands when we move to spot. Contract: upload to "
        "<prefix>/<name>.tmp, then server-side copy to <prefix>/<name>, "
        "then delete the .tmp key."
    )


def _s3_latest(uri: str) -> Optional[str]:
    raise NotImplementedError("list_objects_v2 under prefix, return max ckpt- key")


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #
def save_atomic(local_path: str, uri: str, name: str) -> str:
    """Move bytes at ``local_path`` into the store atomically. Returns final ref."""
    if is_s3(uri):
        return _s3_save(local_path, uri, name)
    return _local_save(local_path, uri, name)


def latest(uri: str) -> Optional[str]:
    """Return a ref to the newest checkpoint under ``uri``, or None if empty."""
    return _s3_latest(uri) if is_s3(uri) else _local_latest(uri)
