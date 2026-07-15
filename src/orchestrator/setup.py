"""One-time, idempotent AWS infra setup: bucket + instance profile + SG.

Run once (``spot-orchestrate setup``). Needs your local creds to have S3, IAM,
and EC2 permissions. Everything here is safe to re-run. Use ``--dry-run`` to see
exactly what it would create before granting it those permissions.
"""

from __future__ import annotations

import sys

from . import aws
from .config import OrchestratorConfig


def ensure_infra(cfg: OrchestratorConfig) -> None:
    cfg.require_bucket()
    aws.set_region(cfg.region)

    aws.ensure_bucket(cfg.bucket, cfg.region)
    aws.ensure_instance_profile(cfg.role_name, cfg.instance_profile, cfg.bucket)
    # The durable orchestrator's controller role (used by `remote-up` boxes): a
    # distinct, higher-privilege profile than the worker's — see
    # docs/iam/orchestrator-policy.json. Idempotent, so `setup` provisions both.
    aws.ensure_orchestrator_profile(
        cfg.orchestrator_role_name,
        cfg.orchestrator_instance_profile,
        cfg.bucket,
        aws.caller_account_id(),
    )
    sg_id = aws.ensure_security_group(cfg.security_group, cfg.region)

    print(
        f"[setup] ready: bucket={cfg.bucket} profile={cfg.instance_profile} "
        f"orch_profile={cfg.orchestrator_instance_profile} sg={sg_id} region={cfg.region}",
        file=sys.stderr,
    )
    print(
        "[setup] note: a freshly-created IAM instance profile can take ~10s to "
        "propagate before the first launch succeeds.",
        file=sys.stderr,
    )
