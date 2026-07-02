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


def instance_state(instance_id: str) -> str:
    if _DRY_RUN:
        _log(f"describe {instance_id}")
        return "running"
    r = _client("ec2").describe_instances(InstanceIds=[instance_id])
    return r["Reservations"][0]["Instances"][0]["State"]["Name"]


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
                "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
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
    _ignore_exists(
        lambda: iam.add_role_to_instance_profile(
            InstanceProfileName=profile_name, RoleName=role_name
        )
    )


def ensure_security_group(name: str, region: str) -> str:
    """Create an egress-only security group (no inbound needed — user-data mode).
    Returns the group id."""
    _log(f"ensure security group {name} (egress only) in {region}")
    if _DRY_RUN:
        return "sg-DRYRUN"
    ec2 = _client("ec2")
    existing = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [name]}])[
        "SecurityGroups"
    ]
    if existing:
        return existing[0]["GroupId"]
    gid = ec2.create_security_group(GroupName=name, Description="spot-train egress only")["GroupId"]
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
) -> str:
    """Launch one instance (on-demand or spot). Returns the instance id."""
    _log(
        f"RunInstances type={instance_type} market={market} ami={ami_id} "
        f"run_id={run_id} (auto-terminate on shutdown)"
    )
    if _DRY_RUN:
        return "i-DRYRUN"
    kwargs: dict[str, Any] = {
        "ImageId": ami_id,
        "InstanceType": instance_type,
        "MinCount": 1,
        "MaxCount": 1,
        "IamInstanceProfile": {"Name": profile_name},
        "SecurityGroupIds": [security_group_id],
        "UserData": user_data,
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


def _ignore_exists(fn) -> None:
    """Run an idempotent IAM create, swallowing 'already exists' errors."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001 — boto ClientError; treat EntityAlreadyExists as ok
        if "EntityAlreadyExists" not in str(e) and "already exists" not in str(e):
            raise
