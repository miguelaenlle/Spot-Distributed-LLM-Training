"""Dataset-provisioning tests — hermetic (no S3; download is monkeypatched).

Pins the multi-rank download contract on a box: only LOCAL_RANK 0 pulls from
S3 and lands each file atomically (temp + os.replace), while other local ranks
wait for the files to appear instead of racing to overwrite them.
"""

from __future__ import annotations

import os

import pytest

from spot_train import data as data_mod


def _make_loader(tmp_path, monkeypatch, local_rank: str | None):
    """Build a PositionedLoader far enough to run _ensure_data, no more."""
    if local_rank is None:
        monkeypatch.delenv("LOCAL_RANK", raising=False)
    else:
        monkeypatch.setenv("LOCAL_RANK", local_rank)
    loader = data_mod.PositionedLoader.__new__(data_mod.PositionedLoader)
    loader.data_local_dir = str(tmp_path / "ds")
    loader.data_uri = "s3://bucket/data/ds"
    return loader


def test_rank0_downloads_atomically(tmp_path, monkeypatch):
    staged = tmp_path / "staged"
    staged.mkdir()

    def fake_download(ref):
        name = ref.rsplit("/", 1)[-1]
        p = staged / name
        p.write_bytes(b"payload:" + name.encode())
        return str(p)

    monkeypatch.setattr(data_mod.s3_store, "download", fake_download)
    monkeypatch.setattr(data_mod.s3_store, "exists", lambda ref: True)
    loader = _make_loader(tmp_path, monkeypatch, "0")
    loader._ensure_data()
    for name in data_mod._FILES:
        dest = os.path.join(loader.data_local_dir, name)
        assert os.path.exists(dest)
        with open(dest, "rb") as f:
            assert f.read() == b"payload:" + name.encode()
    # No temp debris left behind (everything was os.replace'd into place).
    assert all(".tmp-" not in n for n in os.listdir(loader.data_local_dir))


def test_nonzero_local_rank_waits_instead_of_downloading(tmp_path, monkeypatch):
    def must_not_download(ref):
        raise AssertionError("non-zero local rank must never download")

    monkeypatch.setattr(data_mod.s3_store, "download", must_not_download)
    loader = _make_loader(tmp_path, monkeypatch, "1")
    os.makedirs(loader.data_local_dir)

    # Files already present (rank 0 finished first): returns without downloading.
    for name in data_mod._FILES:
        with open(os.path.join(loader.data_local_dir, name), "wb") as f:
            f.write(b"x")
    loader._ensure_data()

    # Files absent and rank 0 never delivers: bounded wait, then a clear error.
    for name in data_mod._FILES:
        os.unlink(os.path.join(loader.data_local_dir, name))
    with pytest.raises(TimeoutError, match="rank 0"):
        loader._wait_for_files(list(data_mod._FILES), timeout=0.1)


def test_single_process_still_downloads_without_local_rank(tmp_path, monkeypatch):
    calls = []

    def fake_download(ref):
        name = ref.rsplit("/", 1)[-1]
        calls.append(name)
        p = tmp_path / name
        p.write_bytes(b"x")
        return str(p)

    monkeypatch.setattr(data_mod.s3_store, "download", fake_download)
    monkeypatch.setattr(data_mod.s3_store, "exists", lambda ref: True)
    loader = _make_loader(tmp_path, monkeypatch, None)  # LOCAL_RANK unset
    loader._ensure_data()
    assert sorted(calls) == sorted(data_mod._FILES)


def test_missing_optional_meta_is_skipped_not_downloaded(tmp_path, monkeypatch):
    """BPE datasets (OpenWebText) ship no meta.pkl. The box must fetch the
    required bins and skip the un-staged meta.pkl cleanly — not 404 on it."""
    calls = []

    def fake_download(ref):
        name = ref.rsplit("/", 1)[-1]
        calls.append(name)
        p = tmp_path / name
        p.write_bytes(b"x")
        return str(p)

    # meta.pkl is not in S3; train/val are.
    def fake_exists(ref):
        return not ref.endswith("meta.pkl")

    monkeypatch.setattr(data_mod.s3_store, "download", fake_download)
    monkeypatch.setattr(data_mod.s3_store, "exists", fake_exists)
    loader = _make_loader(tmp_path, monkeypatch, "0")
    loader._ensure_data()  # must not raise

    assert sorted(calls) == sorted(data_mod._REQUIRED)  # meta.pkl never fetched
    for name in data_mod._REQUIRED:
        assert os.path.exists(os.path.join(loader.data_local_dir, name))
    assert not os.path.exists(os.path.join(loader.data_local_dir, "meta.pkl"))
