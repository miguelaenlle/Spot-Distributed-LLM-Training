"""Full-state checkpoint: everything that affects the next training step.

Missing any one of these makes resume silently diverge:

    - model weights
    - optimizer state
    - step number
    - all RNG states           (see rng.py)
    - data-loader position     (see data.py)

Save path: serialize to a local temp file, then hand off to the store, which
renames atomically (see s3_store.save_atomic). A mid-write kill can only ever
leave a .tmp behind — never a corrupt "latest".

Two tools answer "is this checkpoint comprehensive and valid?":
  - ``verify(ref)``     — loads it, checks the schema is complete and every
                          tensor is finite (catches NaN/inf and truncation).
  - ``smoke_test(...)`` — restores the weights into a fresh model and runs one
                          forward pass, asserting a finite loss (catches a file
                          that loads but is subtly wrong).
"""

from __future__ import annotations

import contextlib
import copy
import os
import shutil
import sys
import tempfile
import threading
from collections.abc import Callable
from typing import Any

import torch

from . import distributed, rng, s3_store

# v2 adds trained_seconds (cumulative in-loop wall-clock, the run-level budget's
# progress meter). v3 adds an OPTIONAL "scaler" (the fp16 GradScaler's loss-scale
# state — restoring it keeps a preempt/resume bit-exact instead of re-warming the
# scale from 2**16). Older blobs still load: trained_seconds -> 0, scaler -> None.
CKPT_VERSION = 3
_KNOWN_VERSIONS = (1, 2, 3)
_REQUIRED_KEYS = ("version", "step", "model", "optimizer", "rng", "loader")


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is incomplete, corrupt, or non-finite."""


def _ckpt_name(step: int) -> str:
    # zero-padded so lexicographic sort == numeric sort for `latest()`
    return f"{s3_store.CHECKPOINT_PREFIX}{step:012d}.pt"


def _step_of(ref: str | None) -> int:
    """Step number encoded in a checkpoint ref's basename, or -1 for None."""
    if ref is None:
        return -1
    base = ref.rsplit("/", 1)[-1]
    return int(base[len(s3_store.CHECKPOINT_PREFIX) : -len(".pt")])


def save(
    *, model, optimizer, loader, step: int, uri: str, trained_seconds: float = 0.0, scaler=None
) -> str:
    """Atomically persist full training state. Returns the final checkpoint ref."""
    blob: dict[str, Any] = {
        "version": CKPT_VERSION,
        "step": step,
        "trained_seconds": trained_seconds,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": rng.capture(),
        "loader": loader.state_dict(),
        "scaler": _scaler_state(scaler),
    }
    fd, tmp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(blob, tmp_path)
        _warn_if_low_disk(os.path.getsize(tmp_path))
        return s3_store.save_atomic(tmp_path, uri, _ckpt_name(step))
    finally:
        # The local backend renames tmp_path away; the S3 backend uploads a
        # copy and leaves it — without this, every save leaks a checkpoint
        # into /tmp until the disk fills.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _scaler_state(scaler) -> dict | None:
    """The GradScaler's state, or None when there's no scaler / it's disabled
    (bf16, fp32, CPU) — a disabled scaler's state_dict is empty and carries no
    information worth persisting, so None keeps blobs uniform across dtypes."""
    if scaler is None or not scaler.is_enabled():
        return None
    return scaler.state_dict()


def _cpu_copy(obj: Any) -> Any:
    """Deep copy a state tree with every tensor moved to CPU. The copy is the
    point: the optimizer keeps mutating the live tensors while the background
    writer serializes, so the snapshot must not alias them."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _cpu_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpu_copy(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cpu_copy(v) for v in obj)
    return copy.deepcopy(obj)


def snapshot(
    *, model, optimizer, loader, step: int, trained_seconds: float = 0.0, scaler=None
) -> dict[str, Any]:
    """Point-in-time CPU copy of the full training state (same schema as
    ``save`` writes). This is the only part of an async checkpoint that must
    run on the training critical path — ~tens of ms for a NanoGPT-sized model.
    RNG and loader state are captured in the same instant as the weights, the
    invariant that keeps resume from silently diverging."""
    return {
        "version": CKPT_VERSION,
        "step": step,
        "trained_seconds": trained_seconds,
        "model": _cpu_copy(model.state_dict()),
        "optimizer": _cpu_copy(optimizer.state_dict()),
        "rng": rng.capture(),  # capture() already returns copies
        "loader": dict(loader.state_dict()),
        "scaler": _cpu_copy(_scaler_state(scaler)),
    }


def save_local(blob: dict[str, Any], local_dir: str, step: int, keep: int = 2) -> str:
    """Write a snapshot blob to the node-local tier (atomic same-dir rename) and
    prune to the ``keep`` newest. Keeping two absorbs one interval of skew when
    a crash lands mid-save somewhere in the group — the group-MIN agreement in
    :func:`load_group_latest` can then still find a step everyone has."""
    os.makedirs(local_dir, exist_ok=True)
    # Temp file IN the destination dir so os.replace never crosses filesystems.
    fd, tmp_path = tempfile.mkstemp(suffix=".pt", dir=local_dir)
    os.close(fd)
    try:
        torch.save(blob, tmp_path)
        final = os.path.join(local_dir, _ckpt_name(step))
        os.replace(tmp_path, final)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    kept = sorted(
        f
        for f in os.listdir(local_dir)
        if f.startswith(s3_store.CHECKPOINT_PREFIX) and f.endswith(".pt")
    )
    for old in kept[:-keep]:
        with contextlib.suppress(OSError):
            os.remove(os.path.join(local_dir, old))
    return final


class AsyncCheckpointer:
    """Two-phase checkpointing: the caller snapshots on its own thread
    (:func:`snapshot`, ~tens of ms), then a daemon thread serializes, uploads
    (same atomic temp-key -> rename), and verify/smoke-tests off the critical
    path.

    One save in flight at a time: ``submit`` returns False while the previous
    write is still running (the caller retries on a later loop iteration), so
    memory is bounded at one snapshot and a slow S3 day can't queue-pile.
    Durability shifts accordingly: worst-case lost work on a hard kill is the
    checkpoint interval PLUS one upload, not the interval alone. Preempt and
    final checkpoints stay synchronous in the trainer — call ``flush`` first so
    the writer never races them.

    Verify/smoke run on CPU in the background thread (a GPU forward there would
    contend with training); a failed background save is logged and counted, and
    training continues on the previous good checkpoint."""

    def __init__(
        self,
        uri: str,
        *,
        verify_every: int = 1,
        build_model: Callable[[], Any] | None = None,
        sample_batch: tuple | None = None,
        log: Callable[[str], None] | None = None,
    ):
        self._uri = uri
        self._verify_every = verify_every
        self._build_model = build_model
        self._sample_batch = sample_batch  # CPU tensors (see trainer: RNG-free eval batch)
        self._log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
        self._thread: threading.Thread | None = None
        self._count = 0
        self.failures = 0

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(
        self, *, model, optimizer, loader, step: int, trained_seconds: float = 0.0, scaler=None
    ) -> bool:
        """Snapshot now and hand off to the writer. False = previous save still
        in flight (skipped; nothing was snapshotted)."""
        if self.busy():
            return False
        blob = snapshot(
            model=model,
            optimizer=optimizer,
            loader=loader,
            step=step,
            trained_seconds=trained_seconds,
            scaler=scaler,
        )
        self._count += 1
        self._thread = threading.Thread(
            target=self._write, args=(blob, step, self._count), name="ckpt-writer", daemon=True
        )
        self._thread.start()
        return True

    def _write(self, blob: dict[str, Any], step: int, count: int) -> None:
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".pt")
            os.close(fd)
            try:
                torch.save(blob, tmp_path)
                _warn_if_low_disk(os.path.getsize(tmp_path))
                ref = s3_store.save_atomic(tmp_path, self._uri, _ckpt_name(step))
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            if self._verify_every and count % self._verify_every == 0:
                good = verify(ref, map_location="cpu")
                if self._build_model is not None and self._sample_batch is not None:
                    smoke_test(good, self._build_model, self._sample_batch, "cpu")
                self._log(f"[verify] checkpoint at step {step} passed verify + smoke test (async)")
        except Exception as e:  # noqa: BLE001 — background thread: log, count, keep training
            self.failures += 1
            self._log(
                f"[checkpoint] ASYNC save at step {step} FAILED: {e!r} — "
                "training continues on the previous checkpoint"
            )

    def flush(self, timeout: float = 300.0) -> None:
        """Wait for the in-flight save (if any) to finish. Called before the
        synchronous preempt/final checkpoints and at shutdown."""
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout)


def _warn_if_low_disk(ckpt_bytes: int) -> None:
    """One save+verify cycle needs ~2x the checkpoint transiently; warn at 4x
    so the operator sees the disk shrinking before a write fails mid-run."""
    free = shutil.disk_usage(tempfile.gettempdir()).free
    if free < 4 * ckpt_bytes:
        print(
            f"[checkpoint] WARNING: {free / 1e9:.1f} GB free on "
            f"{tempfile.gettempdir()} < 4x checkpoint size "
            f"({ckpt_bytes / 1e9:.2f} GB) — the next save/verify may fail",
            file=sys.stderr,
            flush=True,
        )


def load_latest(uri: str, map_location: str = "cpu") -> dict[str, Any] | None:
    """Return the newest checkpoint blob under ``uri``, or None if none exists."""
    ref = s3_store.latest(uri)
    if ref is None:
        return None
    with s3_store.fetch(ref) as local:  # no-op for local refs; downloads+verifies S3
        return torch.load(local, map_location=map_location, weights_only=False)


def load_group_latest(
    uri: str,
    local_dir: str = "",
    dist_ctx: distributed.Dist | None = None,
    map_location: str = "cpu",
) -> dict[str, Any] | None:
    """The one resume path, now across two tiers and N ranks.

    Every rank offers the newest step it can reach (node-local tier if present,
    else the durable S3 tier); the group takes the MIN so all ranks restore the
    SAME step, then each loads it from the cheapest tier that has it. The two
    interesting cases fall out without any membership detection:

      - group shrank (survivors only): everyone holds the same step-aligned
        local snapshot -> instant disk restore, ~zero lost work;
      - a fresh replacement is present: its best is S3-latest, which becomes the
        group MIN -> everyone restores S3-latest (survivors lose at most one
        interval + one in-flight upload — the existing durability bound).

    Degrades exactly to :func:`load_latest` semantics for a single process with
    no local tier, so single-node paths are unchanged.
    """
    local_ref = s3_store.latest(local_dir) if local_dir else None
    s3_ref = s3_store.latest(uri)
    best = max(_step_of(local_ref), _step_of(s3_ref))
    group_step = distributed.all_reduce_min(dist_ctx, best) if dist_ctx is not None else best
    if group_step < 0:
        return None  # no checkpoint anywhere in the group -> fresh
    name = _ckpt_name(group_step)
    local_path = os.path.join(local_dir, name) if local_dir else ""
    if local_path and os.path.exists(local_path):
        ref = local_path
    else:
        ref = s3_store.ref_for(uri, name)
        if not s3_store.exists(ref):
            # A peer holds a local step the durable tier never received (its
            # upload failed). Crash loudly: the elastic agent restarts us, the
            # group re-agrees, and S3 has usually caught up by then.
            raise CheckpointError(
                f"group agreed on step {group_step} but this rank has no tier "
                f"holding it (local={local_path or 'off'}, s3={ref})"
            )
    with s3_store.fetch(ref) as local:
        return torch.load(local, map_location=map_location, weights_only=False)


def restore_into(blob: dict[str, Any], *, model, optimizer, loader, scaler=None) -> int:
    """Restore all state from ``blob``. Returns the step to resume from.

    ``scaler`` is optional: only fp16 runs carry one, and only a v3+ blob written
    by an enabled scaler holds its state. When either is absent the scaler simply
    keeps its fresh loss-scale (a brief re-warm, never a divergence)."""
    model.load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    loader.load_state_dict(blob["loader"])
    rng.restore(blob["rng"])
    saved_scaler = blob.get("scaler")
    if scaler is not None and scaler.is_enabled() and saved_scaler:
        scaler.load_state_dict(saved_scaler)
    return blob["step"]


# --------------------------------------------------------------------------- #
# Validation tools
# --------------------------------------------------------------------------- #
def _all_finite(state: Any) -> bool:
    """Recursively assert every floating-point tensor in a state tree is finite."""
    if isinstance(state, torch.Tensor):
        return (not state.is_floating_point()) or bool(torch.isfinite(state).all())
    if isinstance(state, dict):
        return all(_all_finite(v) for v in state.values())
    if isinstance(state, list | tuple):
        return all(_all_finite(v) for v in state)
    return True


def _verify_blob(blob: dict[str, Any]) -> dict[str, Any]:
    missing = [k for k in _REQUIRED_KEYS if k not in blob]
    if missing:
        raise CheckpointError(f"checkpoint missing keys: {missing}")
    if blob["version"] not in _KNOWN_VERSIONS:
        raise CheckpointError(f"unsupported checkpoint version {blob['version']}")
    if not _all_finite(blob["model"]):
        raise CheckpointError("model weights contain NaN/inf")
    if not _all_finite(blob["optimizer"].get("state", {})):
        raise CheckpointError("optimizer state contains NaN/inf")
    return blob


def verify(ref: str, map_location: str = "cpu") -> dict[str, Any]:
    """Load ``ref`` and assert it is complete and finite. Returns the blob.

    ``ref`` may be a local path or an ``s3://`` URI (downloaded + checksum-checked).
    Raises :class:`CheckpointError` on any problem; ``torch.load`` itself raises on
    a truncated/corrupt file.
    """
    with s3_store.fetch(ref) as local:
        return _verify_blob(torch.load(local, map_location=map_location, weights_only=False))


def smoke_test(
    blob: dict[str, Any],
    build_model: Callable[[], Any],
    sample_batch: tuple,
    device: str,
) -> float:
    """Restore weights into a fresh model and run one forward pass.

    Confirms the saved state actually reconstructs a working model, not just a
    loadable file. Returns the (finite) loss; raises :class:`CheckpointError`
    otherwise.
    """
    model = build_model().to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    with torch.no_grad():
        _, loss = model(*sample_batch)
    if loss is None or not bool(torch.isfinite(loss)):
        raise CheckpointError("smoke test produced a non-finite loss")
    return float(loss.item())
