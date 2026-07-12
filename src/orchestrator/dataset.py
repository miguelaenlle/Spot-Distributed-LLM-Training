"""Stage the dataset to S3 once, so every training box pulls identical bytes.

Runs nanoGPT's ``prepare.py`` locally (if the bins aren't already present), then
uploads ``train.bin``/``val.bin``/``meta.pkl`` to ``s3://<bucket>/data/<dataset>/``.
Idempotent: if the objects already exist in S3, it does nothing. shakespeare_char
is tiny; the same flow works for OpenWebText later (just prepared offline).
"""

from __future__ import annotations

import os
import subprocess
import sys

from . import aws
from .config import OrchestratorConfig

# train/val bins are required; meta.pkl is only for char-level datasets (the BPE
# corpora like OpenWebText have a fixed vocab and ship no meta).
_REQUIRED = ("train.bin", "val.bin")
_OPTIONAL = ("meta.pkl",)


def _local_dir(cfg: OrchestratorConfig) -> str:
    """Where this dataset's ``prepare.py`` + bins live. Prefer a repo-level
    ``data/<dataset>/`` (our own preps, e.g. the capped OpenWebText slice) over
    nanoGPT's ``third_party/nanoGPT/data/<dataset>/`` (the submodule fixtures)."""
    ours = f"data/{cfg.dataset}"
    if os.path.exists(os.path.join(ours, "prepare.py")):
        return ours
    return f"third_party/nanoGPT/data/{cfg.dataset}"


def stage_data(cfg: OrchestratorConfig) -> None:
    cfg.require_bucket()
    prefix = f"{cfg.data_prefix}/{cfg.dataset}"

    if all(aws.object_exists(cfg.bucket, f"{prefix}/{f}") for f in _REQUIRED):
        print(f"[stage-data] {cfg.data_uri()} already present — nothing to do", file=sys.stderr)
        return

    data_dir = _local_dir(cfg)
    if not all(os.path.exists(os.path.join(data_dir, f)) for f in _REQUIRED):
        print(f"[stage-data] running prepare.py in {data_dir}", file=sys.stderr)
        subprocess.run([sys.executable, "prepare.py"], cwd=data_dir, check=True)

    uploaded = []
    for f in (*_REQUIRED, *_OPTIONAL):
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            aws.upload_file(path, cfg.bucket, f"{prefix}/{f}")
            uploaded.append(f)
    print(f"[stage-data] uploaded {uploaded} to {cfg.data_uri()}", file=sys.stderr)
