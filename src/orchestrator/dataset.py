"""Stage the dataset to S3 once, so every training box pulls identical bytes.

Runs nanoGPT's ``prepare.py`` locally (if the bins aren't already present), then
uploads ``train.bin``/``val.bin``/``meta.pkl`` to ``s3://<bucket>/data/<dataset>/``.
Idempotent: if the objects already exist in S3, it does nothing. shakespeare_char
is tiny; the same flow works for OpenWebText later (just prepared offline).
"""

from __future__ import annotations

import subprocess
import sys

from . import aws
from .config import OrchestratorConfig

_FILES = ("train.bin", "val.bin", "meta.pkl")


def _local_dir(cfg: OrchestratorConfig) -> str:
    # Run from the repo root; nanoGPT's prepare.py lives alongside its data dir.
    return f"third_party/nanoGPT/data/{cfg.dataset}"


def stage_data(cfg: OrchestratorConfig) -> None:
    cfg.require_bucket()
    prefix = f"{cfg.data_prefix}/{cfg.dataset}"

    if all(aws.object_exists(cfg.bucket, f"{prefix}/{f}") for f in _FILES):
        print(f"[stage-data] {cfg.data_uri()} already present — nothing to do", file=sys.stderr)
        return

    import os

    data_dir = _local_dir(cfg)
    if not all(os.path.exists(os.path.join(data_dir, f)) for f in _FILES):
        print(f"[stage-data] running nanoGPT prepare.py in {data_dir}", file=sys.stderr)
        subprocess.run([sys.executable, "prepare.py"], cwd=data_dir, check=True)

    for f in _FILES:
        aws.upload_file(os.path.join(data_dir, f), cfg.bucket, f"{prefix}/{f}")
    print(f"[stage-data] uploaded {list(_FILES)} to {cfg.data_uri()}", file=sys.stderr)
