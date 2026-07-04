"""The ONLY module that talks to AWS.

Every credentialed call lives here so the surface is auditable in one place.
Design rules:

  - Credentials are never referenced in code — boto3 resolves them from the
    ambient environment/profile at call time.
  - Every *mutating* call logs a plain-English line before it fires.
  - ``set_dry_run(True)`` makes every function log what it *would* do and call
    nothing — so ``--dry-run`` provably touches no AWS API and needs no creds.

The orchestrator's other modules (setup, experiments, dataset) call these
functions; they never import boto3 themselves.
"""

from __future__ import annotations

import sys
import time
from typing import Any

_DRY_RUN = False
_clients: dict[str, Any] = {}
_region = "us-east-1"


def set_dry_run(flag: bool) -> None:
    global _DRY_RUN
    _DRY_RUN = flag


def is_dry_run() -> bool:
    return _DRY_RUN


def set_region(region: str) -> None:
    global _region
    _region = region
    _clients.clear()


def _client(service: str):
    import boto3  # lazy: only imported when a real call is made

    if service not in _clients:
        _clients[service] = boto3.client(service, region_name=_region)
    return _clients[service]


def _log(msg: str) -> None:
    prefix = "[aws:dry-run] would" if _DRY_RUN else "[aws]"
    print(f"{prefix} {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Read-only lookups
# --------------------------------------------------------------------------- #
def resolve_ami(ami_id: str, name_filter: str) -> str:
    """Return an explicit AMI id, or the newest Amazon-owned image whose name
    matches ``name_filter`` (via DescribeImages — no SSM public parameters)."""
    if ami_id:
        return ami_id
    if _DRY_RUN:
        _log(f"resolve AMI via DescribeImages name~={name_filter!r}")
        return "ami-DRYRUN"
    r = _client("ec2").describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name", "Values": [name_filter]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": ["x86_64"]},
        ],
    )
    images = sorted(r.get("Images", []), key=lambda i: i["CreationDate"])
    if not images:
        raise SystemExit(
            f"No AMI matched {name_filter!r} in this region. Set AMI_ID explicitly "
            f"(see README) or adjust AMI_NAME_FILTER."
        )
    chosen = images[-1]
    _log(f"resolved AMI {chosen['ImageId']} ({chosen['Name']})")
    return chosen["ImageId"]


def object_exists(bucket: str, key: str) -> bool:
    if _DRY_RUN:
        _log(f"head s3://{bucket}/{key}")
        return False
    try:
        _client("s3").head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def any_object_under(bucket: str, prefix: str) -> bool:
    """True if at least one object exists under ``prefix`` (e.g. first checkpoint)."""
    if _DRY_RUN:
        _log(f"list s3://{bucket}/{prefix} (MaxKeys=1)")
        return False
    r = _client("s3").list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return r.get("KeyCount", 0) > 0


def get_text(bucket: str, key: str) -> str:
    if _DRY_RUN:
        _log(f"get s3://{bucket}/{key}")
        return "{}"
    return _client("s3").get_object(Bucket=bucket, Key=key)["Body"].read().decode()


def max_checkpoint_step(bucket: str, prefix: str) -> int:
    """Highest checkpoint step under ``prefix`` (ckpt-<step>.pt), or -1 if none.
    Used to detect training-start (step advances past the resume point) and to
    confirm the graceful SIGTERM checkpoint landed before we terminate the box."""
    if _DRY_RUN:
        _log(f"list checkpoints s3://{bucket}/{prefix}")
        return -1
    import contextlib

    best = -1
    paginator = _client("s3").get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            base = obj["Key"].rsplit("/", 1)[-1]
            if base.startswith("ckpt-") and base.endswith(".pt"):
                with contextlib.suppress(ValueError):
                    best = max(best, int(base[len("ckpt-") : -len(".pt")]))
    return best


def ssm_online(instance_id: str) -> bool:
    """True if the SSM agent on the instance is registered and online (so we can
    send it a command). Boxes get AmazonSSMManagedInstanceCore via the instance
    profile and outbound HTTPS via the public IP."""
    if _DRY_RUN:
        _log(f"ssm describe-instance-information {instance_id}")
        return True
    r = _client("ssm").describe_instance_information(
        Filters=[{"Key": "InstanceIds", "Values": [instance_id]}]
    )
    info = r.get("InstanceInformationList", [])
    return bool(info) and info[0].get("PingStatus") == "Online"


def ssm_send(instance_id: str, commands: list[str]) -> str:
    """Run shell commands on the instance via SSM RunCommand; returns command id.
    This is how the orchestrator delivers the 'Spot' shutdown signal (SIGTERM to
    the trainer) without SSH."""
    _log(f"ssm send-command {instance_id}: {' && '.join(commands)}")
    if _DRY_RUN:
        return "cmd-DRYRUN"
    r = _client("ssm").send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": commands},
    )
    return r["Command"]["CommandId"]


def instance_state(instance_id: str) -> str:
    if _DRY_RUN:
        _log(f"describe {instance_id}")
        return "running"
    r = _client("ec2").describe_instances(InstanceIds=[instance_id])
    return r["Reservations"][0]["Instances"][0]["State"]["Name"]


def public_ip(instance_id: str) -> str:
    """Public IPv4 of the instance, or "" if it has none. (SSH-verification mode.)"""
    if _DRY_RUN:
        _log(f"describe {instance_id} (public ip)")
        return "203.0.113.10"
    r = _client("ec2").describe_instances(InstanceIds=[instance_id])
    return r["Reservations"][0]["Instances"][0].get("PublicIpAddress", "")


# --------------------------------------------------------------------------- #
# Mutating: S3
# --------------------------------------------------------------------------- #
def ensure_bucket(bucket: str, region: str) -> None:
    _log(f"create S3 bucket {bucket} in {region} (idempotent)")
    if _DRY_RUN:
        return
    s3 = _client("s3")
    if object_exists_bucket(s3, bucket):
        return
    kwargs: dict[str, Any] = {"Bucket": bucket}
    if region != "us-east-1":  # us-east-1 rejects an explicit LocationConstraint
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**kwargs)


def object_exists_bucket(s3, bucket: str) -> bool:
    try:
        s3.head_bucket(Bucket=bucket)
        return True
    except Exception:
        return False


def upload_file(local_path: str, bucket: str, key: str) -> None:
    _log(f"upload {local_path} -> s3://{bucket}/{key}")
    if _DRY_RUN:
        return
    _client("s3").upload_file(local_path, bucket, key, ExtraArgs={"ChecksumAlgorithm": "SHA256"})


# --------------------------------------------------------------------------- #
# Mutating: IAM (instance profile granting the box S3 access)
# --------------------------------------------------------------------------- #
def ensure_instance_profile(role_name: str, profile_name: str, bucket: str) -> None:
    """Create a role the EC2 box assumes, scoped to read/write ``bucket``, and an
    instance profile wrapping it. Idempotent."""
    import json

    _log(f"create IAM role {role_name} + instance profile {profile_name} for s3://{bucket}")
    if _DRY_RUN:
        return
    iam = _client("iam")
    assume = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    _ignore_exists(
        lambda: iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(assume))
    )
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                # DeleteObject is required: the atomic checkpoint writes a .tmp
                # key, copies it to the final key, then DELETES the .tmp
                # (s3_store._s3_save). Without it, checkpointing fails AccessDenied.
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
            }
        ],
    }
    iam.put_role_policy(
        RoleName=role_name, PolicyName="spot-train-s3", PolicyDocument=json.dumps(policy)
    )
    # SSM Session Manager: lets you attach a shell to the box (no inbound ports)
    # to `tail -f /var/log/spot-train-boot.log` and run nvidia-smi.
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
    )
    _ignore_exists(lambda: iam.create_instance_profile(InstanceProfileName=profile_name))
    # An instance profile holds at most one role; on a re-run the role is already
    # attached and AddRoleToInstanceProfile raises LimitExceeded. Add only if the
    # role isn't already in the profile (idempotent).
    attached = [
        r["RoleName"]
        for r in iam.get_instance_profile(InstanceProfileName=profile_name)["InstanceProfile"][
            "Roles"
        ]
    ]
    if role_name not in attached:
        iam.add_role_to_instance_profile(InstanceProfileName=profile_name, RoleName=role_name)


def ensure_security_group(name: str, region: str) -> str:
    """Create the security group and ensure it allows inbound SSH (port 22).
    Returns the group id.

    SSH-verification mode: the group used to be egress-only (user-data mode needs
    no inbound). We now open TCP 22 so you can ssh into a bare box. Idempotent —
    AWS raises InvalidPermission.Duplicate if the rule already exists.
    """
    _log(f"ensure security group {name} (inbound SSH :22) in {region}")
    if _DRY_RUN:
        return "sg-DRYRUN"
    ec2 = _client("ec2")
    existing = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [name]}])[
        "SecurityGroups"
    ]
    gid = (
        existing[0]["GroupId"]
        if existing
        else ec2.create_security_group(GroupName=name, Description="spot-train (SSH verify)")[
            "GroupId"
        ]
    )
    try:
        ec2.authorize_security_group_ingress(
            GroupId=gid,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    # TEMP: open to the world for a quick SSH test. Tighten to your
                    # own IP (e.g. "<your-ip>/32") if the box stays up any length of
                    # time, and revert this whole block when done verifying.
                    "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH (temp verify)"}],
                }
            ],
        )
    except Exception as e:  # noqa: BLE001 — boto ClientError; duplicate rule is fine
        if "InvalidPermission.Duplicate" not in str(e):
            raise
    # Multi-node DDP: allow ALL TCP between instances in this group (the c10d
    # rendezvous TCPStore on node 0 plus the NCCL/gloo data-plane sockets, which
    # use ephemeral ports). Self-referencing, so nothing new is exposed publicly.
    try:
        ec2.authorize_security_group_ingress(
            GroupId=gid,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 0,
                    "ToPort": 65535,
                    "UserIdGroupPairs": [
                        {"GroupId": gid, "Description": "intra-group DDP (rendezvous + NCCL)"}
                    ],
                }
            ],
        )
    except Exception as e:  # noqa: BLE001 — boto ClientError; duplicate rule is fine
        if "InvalidPermission.Duplicate" not in str(e):
            raise
    return gid


# --------------------------------------------------------------------------- #
# Mutating: EC2 lifecycle
# --------------------------------------------------------------------------- #
def launch(
    *,
    ami_id: str,
    instance_type: str,
    profile_name: str,
    security_group_id: str,
    user_data: str,
    market: str,
    run_id: str,
    key_name: str = "",
) -> str:
    """Launch one instance (on-demand or spot). Returns the instance id."""
    _log(
        f"RunInstances type={instance_type} market={market} ami={ami_id} "
        f"run_id={run_id} key={key_name or '<none>'} "
        f"user-data={'yes' if user_data else 'none'} (public IP + SSH ingress)"
    )
    if _DRY_RUN:
        return "i-DRYRUN"
    kwargs: dict[str, Any] = {
        "ImageId": ami_id,
        "InstanceType": instance_type,
        "MinCount": 1,
        "MaxCount": 1,
        "IamInstanceProfile": {"Name": profile_name},
        # SSH-verification mode: give the box a public IP so you can reach it, and
        # attach the SG via the interface. NOTE: when you pass NetworkInterfaces you
        # must NOT also set top-level "SecurityGroupIds" — the group goes in here.
        "NetworkInterfaces": [
            {
                "DeviceIndex": 0,
                "AssociatePublicIpAddress": True,
                "Groups": [security_group_id],
            }
        ],
        # --- ORIGINAL (SG without public IP) — restore when done SSH-testing ---
        # "SecurityGroupIds": [security_group_id],
        "InstanceInitiatedShutdownBehavior": "terminate",
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"spot-train-{run_id}"},
                    {"Key": "project", "Value": "spot-train"},
                    {"Key": "market", "Value": market},
                ],
            }
        ],
    }
    if market == "spot":
        kwargs["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time"},
        }
    if key_name:  # SSH-verification mode: attach a key pair so you can ssh in
        kwargs["KeyName"] = key_name
    if user_data:  # the boot script (provisioning); empty => bare boot, no user-data
        kwargs["UserData"] = user_data
    r = _client("ec2").run_instances(**kwargs)
    return r["Instances"][0]["InstanceId"]


def wait_running(instance_id: str) -> None:
    _log(f"wait until running: {instance_id}")
    if _DRY_RUN:
        return
    _client("ec2").get_waiter("instance_running").wait(InstanceIds=[instance_id])


def terminate(instance_id: str) -> None:
    _log(f"TerminateInstances {instance_id}")
    if _DRY_RUN:
        return
    _client("ec2").terminate_instances(InstanceIds=[instance_id])


def wait_quota_released(instance_id: str) -> None:
    """Block until the instance leaves pending/running — the point at which it
    stops counting against the vCPU quota, so a replacement can launch. Do NOT
    wait for full 'terminated': shutting-down can linger for minutes (a hung OS
    shutdown holds it until AWS force-kills) and the quota is already free."""
    _log(f"wait until instance stops counting against quota: {instance_id}")
    if _DRY_RUN:
        return
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if instance_state(instance_id) not in ("pending", "running"):
            return
        time.sleep(5)
    raise TimeoutError(f"{instance_id} still running 300s after TerminateInstances")


def vcpus_in_use() -> int:
    """vCPUs currently counting against the "Running On-Demand G and VT
    instances" quota: every pending/running G- or VT-family instance in the
    region, whoever launched it (external instances eat the same quota, so
    counting only our own would overshoot)."""
    if _DRY_RUN:
        return 0
    total = 0
    paginator = _client("ec2").get_paginator("describe_instances")
    pages = paginator.paginate(
        Filters=[{"Name": "instance-state-name", "Values": ["pending", "running"]}]
    )
    for page in pages:
        for res in page.get("Reservations", []):
            for inst in res.get("Instances", []):
                family = inst.get("InstanceType", "").split(".")[0]
                if family.startswith("g") or family.startswith("vt"):
                    cpu = inst.get("CpuOptions", {})
                    total += cpu.get("CoreCount", 0) * cpu.get("ThreadsPerCore", 1)
    return total


def wait_vcpu_headroom(needed: int, quota: int, timeout: int = 900) -> None:
    """Block until `needed` vCPUs fit under `quota` alongside current usage, so
    RunInstances isn't fired into a quota wall. Polls DescribeInstances every
    15s (one API call per poll — no spam); logs once when it has to wait."""
    if needed > quota:
        raise SystemExit(
            f"Launch needs {needed} vCPUs but VCPU_QUOTA={quota} — it can never fit. "
            "Raise the quota (Service Quotas console) and update VCPU_QUOTA."
        )
    _log(f"wait for vCPU headroom: need {needed} of {quota} quota")
    if _DRY_RUN:
        return
    waiting_logged = False
    deadline = time.monotonic() + timeout
    while True:
        used = vcpus_in_use()
        if used + needed <= quota:
            if waiting_logged:
                _log(f"vCPU headroom available ({used} used + {needed} needed <= {quota})")
            return
        if not waiting_logged:
            _log(f"quota full ({used} used + {needed} needed > {quota}); polling every 15s")
            waiting_logged = True
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"no vCPU headroom after {timeout}s ({used} used + {needed} needed > {quota})"
            )
        time.sleep(15)


def _ignore_exists(fn) -> None:
    """Run an idempotent IAM create, swallowing 'already exists' errors."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001 — boto ClientError; treat EntityAlreadyExists as ok
        if "EntityAlreadyExists" not in str(e) and "already exists" not in str(e):
            raise
