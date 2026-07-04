"""Regression: the checkpoint save/verify cycle must not leak temp files.

A spot run checkpoints every 30 seconds; before this test existed, each S3
cycle left ~2x the checkpoint size in /tmp (the save's local temp .pt plus
verify's re-download) until the box died with ENOSPC mid-run. The S3 backend
is exercised hermetically via a fake boto3 client backed by a dict.
"""

import os
import tempfile

import pytest
import torch

from spot_train import checkpoint, s3_store


class _FakeS3Client:
    """In-memory stand-in for the handful of boto3 S3 calls s3_store makes."""

    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}

    def upload_file(self, local, bucket, key, ExtraArgs=None):
        with open(local, "rb") as f:
            self.objects[(bucket, key)] = f.read()

    def copy(self, src, bucket, key, ExtraArgs=None):
        self.objects[(bucket, key)] = self.objects[(src["Bucket"], src["Key"])]

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)

    def download_file(self, bucket, key, local, ExtraArgs=None):
        with open(local, "wb") as f:
            f.write(self.objects[(bucket, key)])

    def get_paginator(self, name):
        objects = self.objects

        class _Paginator:
            def paginate(self, Bucket, Prefix):
                keys = [k for (b, k) in objects if b == Bucket and k.startswith(Prefix)]
                yield {"Contents": [{"Key": k} for k in sorted(keys)]}

        return _Paginator()


class _StubLoader:
    def state_dict(self):
        return {"pos": 0}

    def load_state_dict(self, state):
        pass


@pytest.fixture
def isolated_tmpdir(tmp_path, monkeypatch):
    """Redirect tempfile.mkstemp into an empty dir we can assert is left empty."""
    d = tmp_path / "scratch"
    d.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(d))
    return d


@pytest.fixture
def fake_s3(monkeypatch):
    client = _FakeS3Client()
    monkeypatch.setattr(s3_store, "_client", lambda: client)
    return client


def _save_verify_load(uri: str) -> None:
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ref = checkpoint.save(model=model, optimizer=optimizer, loader=_StubLoader(), step=1, uri=uri)
    checkpoint.verify(ref)
    blob = checkpoint.load_latest(uri)
    assert blob is not None and blob["step"] == 1


def test_s3_cycle_leaves_no_temp_files(isolated_tmpdir, fake_s3):
    _save_verify_load("s3://test-bucket/runs/r1/checkpoints")
    assert os.listdir(isolated_tmpdir) == []
    # exactly the final checkpoint object remains — no .tmp upload key either
    keys = [k for _, k in fake_s3.objects]
    assert keys == ["runs/r1/checkpoints/ckpt-000000000001.pt"]


def test_local_cycle_leaves_no_temp_files(isolated_tmpdir, tmp_path):
    _save_verify_load(str(tmp_path / "ckpts"))
    assert os.listdir(isolated_tmpdir) == []
