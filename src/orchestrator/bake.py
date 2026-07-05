"""Bake a pre-provisioned AMI (``spot-orchestrate bake-ami``).

How baking works: an AMI is an EBS root-volume snapshot plus launch metadata.
We launch a throwaway box from the CURRENT stock DLAMI (always re-resolved via
AMI_NAME_FILTER — never from a previous bake, so layers can't accumulate), let
user-data run the same provisioning every training boot performs (repo clone +
nanoGPT submodule + boto3 into the torch venv), stop the box, CreateImage, wait
for the snapshot to finish, and terminate the box. Launching from the result
skips clone+pip at boot and removes GitHub/PyPI as boot-time dependencies —
at 4-8 nodes one flaky clone otherwise stalls a whole rendezvous generation.

Boxes launched from the baked AMI still track the branch tip: the shared
provisioning steps fast-forward an existing clone at every boot (~2s).

GPU not required: pip installs are arch-independent and the DLAMI boots fine on
a CPU instance, so the bake defaults to a cheap t3 — off the G-vCPU quota.

Use: run ``spot-orchestrate bake-ami`` (needs ``setup`` done once), then put the
printed ``AMI_ID=ami-...`` in your .env — ``resolve_ami`` honors it verbatim, so
no other config changes. Delete AMI_ID to fall back to the stock DLAMI. Rebake
only when dependencies change (the boot preflight failing loudly is the
signal); new models, datasets, and code changes never need a rebake.
"""

from __future__ import annotations

import json
import sys
import time

from . import aws, bootstrap
from .config import OrchestratorConfig

_AMI_NAME_PREFIX = "spot-train-baked-"


def _print_new_log(cfg: OrchestratorConfig, log_key: str, printed: int) -> int:
    """Print any bake-log bytes past ``printed``; returns the new offset. The
    log lands in S3 only every ~10s, and not at all until boto3 is up — silence
    here is normal early in the boot."""
    try:
        text = aws.get_text(cfg.bucket, log_key)
    except Exception:  # noqa: BLE001 — log not uploaded yet
        return printed
    if len(text) > printed:
        sys.stderr.write(text[printed:])
        sys.stderr.flush()
    return max(printed, len(text))


def _wait_status(cfg: OrchestratorConfig, bake_id: str, iid: str) -> dict:
    """Stream the bake log until status.json appears (or the box dies/times out)."""
    status_key = cfg.bake_status_key(bake_id)
    log_key = cfg.bake_log_key(bake_id)
    printed = 0
    deadline = time.monotonic() + cfg.bake_timeout_seconds
    while time.monotonic() < deadline:
        printed = _print_new_log(cfg, log_key, printed)
        if aws.object_exists(cfg.bucket, status_key):
            _print_new_log(cfg, log_key, printed)  # final flush
            return json.loads(aws.get_text(cfg.bucket, status_key))
        if aws.instance_state(iid) not in ("pending", "running"):
            raise SystemExit(
                f"[bake] box {iid} died before writing status — "
                f"log: s3://{cfg.bucket}/{log_key}"
            )
        time.sleep(10)
    raise TimeoutError(
        f"[bake] no status after {cfg.bake_timeout_seconds}s — " f"log: s3://{cfg.bucket}/{log_key}"
    )


def _prune_old_images(cfg: OrchestratorConfig) -> None:
    """Keep the newest ``bake_keep_images`` baked AMIs; deregister the rest and
    delete their snapshots (a deregistered AMI's snapshot keeps billing)."""
    keep = cfg.bake_keep_images
    if keep <= 0:
        return
    images = aws.list_baked_images(_AMI_NAME_PREFIX)  # oldest first
    for img in images[:-keep]:
        print(f"[bake] pruning old image {img['id']} ({img['name']})", file=sys.stderr)
        aws.deregister_image(img["id"], img["snapshot_ids"])


def bake_ami(cfg: OrchestratorConfig) -> str | None:
    """Provision a throwaway box, image it, terminate it. Returns the AMI id."""
    cfg.require_bucket()
    bake_id = time.strftime("%Y%m%d-%H%M%S")
    # Always bake from the stock DLAMI (ignore any AMI_ID in the environment):
    # rebaking on top of a previous bake would stack layers and never pick up a
    # new DLAMI release.
    base_ami = aws.resolve_ami("", cfg.ami_name_filter)
    sg_id = aws.ensure_security_group(cfg.security_group, cfg.region)
    user_data = bootstrap.build_bake_user_data(cfg, bake_id=bake_id, base_ami=base_ami)
    print(
        f"[bake] {bake_id}: provisioning {cfg.bake_instance_type} from {base_ami}",
        file=sys.stderr,
    )
    iid = aws.launch(
        ami_id=base_ami,
        instance_type=cfg.bake_instance_type,
        profile_name=cfg.instance_profile,
        security_group_id=sg_id,
        user_data=user_data,
        market="on-demand",
        run_id=f"bake-{bake_id}",
        key_name=cfg.key_name,
    )
    if aws.is_dry_run():
        print("[bake] dry-run: skipping provision-wait/image/terminate", file=sys.stderr)
        return None
    image_id = None
    try:
        aws.wait_running(iid)
        status = _wait_status(cfg, bake_id, iid)
        if not status.get("ok"):
            raise SystemExit(
                f"[bake] provisioning failed (rc={status.get('rc')}) — "
                f"log: s3://{cfg.bucket}/{cfg.bake_log_key(bake_id)}"
            )
        commit = status.get("commit", "")
        print(f"[bake] provisioned at commit {commit or '<unknown>'}; imaging", file=sys.stderr)
        # Stop before imaging so the snapshot captures a quiesced filesystem.
        aws.stop_instance(iid)
        aws.wait_stopped(iid)
        name = f"{_AMI_NAME_PREFIX}{bake_id}"
        tags = {"Name": name, "project": "spot-train", "base_ami": base_ami}
        if commit:
            tags["repo_commit"] = commit
        image_id = aws.create_image(iid, name, tags)
        print(f"[bake] {image_id} registered; waiting for snapshot (minutes)", file=sys.stderr)
        aws.wait_image_available(image_id)
    finally:
        # The bake box is disposable either way — the log survives in S3.
        aws.terminate(iid)
    _prune_old_images(cfg)
    print(f"\n[bake] AMI ready: {image_id}")
    print(f"[bake] add to your .env:\n\n    AMI_ID={image_id}\n")
    print(
        "[bake] remove AMI_ID to fall back to the stock DLAMI; "
        "rebake only when dependencies change."
    )
    return image_id
