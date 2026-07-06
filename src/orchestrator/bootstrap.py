"""Build the EC2 user-data script that runs on the box at boot.

The box is an AWS Deep Learning AMI (Ubuntu) with CUDA + PyTorch preinstalled.
Two DLAMI realities shape these scripts:

  1. The PyTorch environment only auto-activates for the ``ubuntu`` **login**
     shell — root user-data would otherwise get a system python without torch.
     So we run everything via ``sudo -u ubuntu -i`` (login shell) and preflight
     ``import torch`` to fail loudly if the env is wrong.
  2. We run from source with ``PYTHONPATH`` instead of ``pip install -e .`` so we
     never try to write into a possibly root-owned framework env. (nanoGPT is
     found by the trainer relative to the package.)

Three builders:

  * ``build_provisioning_user_data`` — clone the repo + nanoGPT submodule and
    install deps, write a ``source``-able env file, then STOP and leave the box
    UP. You ssh in, confirm the code is there, and run training by hand. Used
    while bringing the box up.
  * ``build_user_data`` — the full run: provision, then ``spot_train.train``
    under a wall-clock budget, then self-terminate on success. Used once the box
    is proven. (Parked while we validate provisioning over SSH.)
  * ``build_bake_user_data`` — provision only (no training, no env file), then
    write a status marker to S3; ``spot-orchestrate bake-ami`` images the box.

The trainer pulls the dataset and reads/writes S3 via the instance profile role
— no credentials are passed in user-data.
"""

from __future__ import annotations

import base64
import json

from .config import OrchestratorConfig


def _trainer_env(
    cfg: OrchestratorConfig, *, run_id: str, market: str, max_seconds: int
) -> dict[str, str]:
    """Environment the trainer reads via ``TrainConfig.from_env`` (spot_train
    config). Written to a ``source``-able file so you can run training by hand
    after sshing in, and exported inline for the full-run builder."""
    env = {
        "PYTHONPATH": "/home/ubuntu/app/src",
        "CHECKPOINT_URI": cfg.run_checkpoint_uri(run_id),
        "METRICS_URI": cfg.run_metrics_uri(run_id),
        "SAMPLES_URI": cfg.run_samples_uri(run_id),
        "SAMPLES_PREFIX_URI": cfg.run_samples_prefix_uri(run_id),
        # base64(JSON array): the env file is `export K="v"` lines, and prompt
        # text (quotes, spaces, newlines) must survive that quoting untouched.
        # The loads/dumps round-trip fails fast here on a malformed .env value —
        # before any instance is launched.
        "SAMPLE_PROMPTS": base64.b64encode(
            json.dumps(json.loads(cfg.sample_prompts)).encode()
        ).decode(),
        "DATA_URI": cfg.data_uri(),
        "DATASET": cfg.dataset,
        "DATA_LOCAL_DIR": f"/home/ubuntu/app/third_party/nanoGPT/data/{cfg.dataset}",
        "MAX_SECONDS": str(max_seconds),
        "CHECKPOINT_INTERVAL_SECONDS": str(cfg.checkpoint_interval_seconds),
        "SMOKE_TEST_EVERY": str(cfg.smoke_test_every),
        "EVAL_ITERS": str(cfg.eval_iters),
        "BATCH_SIZE": str(cfg.batch_size),
        "RUN_ID": run_id,
        "MARKET": market,
        "DEVICE": "auto",  # trainer auto-detects cuda vs cpu on the box
        "DDP_DATA_MODE": cfg.ddp_data_mode,  # only used when launched via torchrun
        "PYTHONUNBUFFERED": "1",  # unbuffered so `tail -f` shows per-step lines live
    }
    # Recipe/cadence knobs (MAX_STEPS, LR schedule, EVAL/SAMPLE intervals, …)
    # relay verbatim, and only when set — unset keeps the trainer's defaults.
    env.update(cfg.trainer_passthrough())
    return env


def _export_block(env: dict[str, str]) -> str:
    return "\n".join(f'export {k}="{v}"' for k, v in env.items())


def _provision_steps(cfg: OrchestratorConfig, exports: str) -> str:
    """Setup run as the ubuntu login shell (torch env active): clone the repo,
    populate the nanoGPT submodule, install deps, write a source-able env file,
    and preflight. Shared by both builders."""
    return f"""set -euxo pipefail
cd /home/ubuntu

# Interpreter: the DLAMI base shell only ships `python3`; plain `python` exists
# only inside the pytorch venv, which the non-interactive login shell does NOT
# reliably auto-activate. Use the venv python by absolute path (it has torch +
# boto3 and always exists on this AMI); fall back to python3 on other images.
VENV_PY=/opt/pytorch/bin/python
[ -x "$VENV_PY" ] || VENV_PY="$(command -v python3)"

# Code: clone the repo, or — on a pre-baked AMI where the clone already exists —
# fast-forward it to the branch tip so baked boxes still track {cfg.repo_branch}
# (a bare `[ -d app ] || clone` would pin baked boxes at bake-time code forever).
# Then populate the nanoGPT submodule: a plain clone leaves third_party/nanoGPT
# EMPTY, and the trainer imports GPT from there, so the init is not optional.
if [ -d app/.git ]; then
  git -C app fetch --depth 1 origin {cfg.repo_branch}
  git -C app reset --hard FETCH_HEAD
else
  git clone --depth 1 -b {cfg.repo_branch} {cfg.repo_url} app
fi
cd app
git submodule update --init --depth 1

# Deps: the DLAMI ships torch + numpy; boto3 is usually missing. Best-effort so a
# broken pip doesn't wedge the whole boot.
"$VENV_PY" -c "import boto3" 2>/dev/null \\
  || "$VENV_PY" -m pip install boto3 \\
  || "$VENV_PY" -m pip install --user boto3 || true

# Env: drop the trainer's config into a file you can `source` before a manual run.
cat > /home/ubuntu/spot-train.env <<'ENV'
{exports}
ENV

# Preflight: fail loudly IN THIS LOG if the torch/boto3 env is wrong.
"$VENV_PY" - <<'PY'
import torch, boto3, numpy  # noqa: F401
print("preflight ok: torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
PY
"""


def _fleet_env(
    cfg: OrchestratorConfig, *, fleet_id: str, role: str, worker_id: str, run_id: str, port: int
) -> dict[str, str]:
    """Environment for a fleet box (inference worker or router). Workers need
    the checkpoint + codec locations; the router only needs the heartbeat
    prefix. ADVERTISE_ADDR is exported at boot from IMDS (private IP).

    ``fleet_id=""`` = standalone serve box: no heartbeat prefix, so the worker
    skips registration entirely (there is no router to find it)."""
    env = {
        "PYTHONPATH": "/home/ubuntu/app/src",
        "FLEET_WORKERS_URI": cfg.fleet_workers_uri(fleet_id) if fleet_id else "",
        "PORT": str(port),
        "HOST": "0.0.0.0",
        "MARKET": cfg.fleet_market if role == "worker" else "on-demand",
        "PYTHONUNBUFFERED": "1",
    }
    if role == "worker":
        env.update(
            {
                "WORKER_ID": worker_id,
                "CHECKPOINT_URI": cfg.run_checkpoint_uri(run_id),
                "DATA_URI": cfg.data_uri(),
                "DATASET": cfg.dataset,
                "DATA_LOCAL_DIR": f"/home/ubuntu/app/third_party/nanoGPT/data/{cfg.dataset}",
                "RUN_ID": run_id,
                "DEVICE": "auto",
            }
        )
    return env


def build_fleet_user_data(
    cfg: OrchestratorConfig,
    *,
    fleet_id: str,
    role: str,  # "worker" | "router"
    worker_id: str = "",
    run_id: str = "",
    logs_key: str,
    port: int,
) -> str:
    """Boot one fleet box: provision code (same clone/fast-forward + submodule
    steps as the trainer), pip-install the fleet deps, then run the worker or
    router service in the foreground while streaming the boot log to S3.

    Unlike training boxes there is no self-terminate: serving runs until
    `fleet down` (or a spot reclaim) terminates the instance. A nonzero service
    exit leaves the box up for debugging — the heartbeat TTL removes a wedged
    worker from rotation either way."""
    module = "inference.worker" if role == "worker" else "inference.router"
    env = _fleet_env(
        cfg, fleet_id=fleet_id, role=role, worker_id=worker_id, run_id=run_id, port=port
    )
    steps = _provision_steps(cfg, _export_block(env))
    bucket = cfg.bucket
    interval = cfg.log_stream_seconds
    return f"""#!/bin/bash
set -x
exec > /var/log/spot-train-boot.log 2>&1
chmod 644 /var/log/spot-train-boot.log

sudo -u ubuntu -i bash <<'BOOT'
set -x
cd /home/ubuntu

VENV_PY=/opt/pytorch/bin/python
[ -x "$VENV_PY" ] || VENV_PY="$(command -v python3)"

# S3 log uploader first, so the whole boot streams to the orchestrator.
"$VENV_PY" -c "import boto3" 2>/dev/null || "$VENV_PY" -m pip install boto3 2>/dev/null \
  || "$VENV_PY" -m pip install --user boto3 2>/dev/null || true
"$VENV_PY" - <<'PY' &
import time, boto3
c = boto3.client("s3")
while True:
    try:
        c.upload_file("/var/log/spot-train-boot.log", "{bucket}", "{logs_key}")
    except Exception:
        pass
    time.sleep({interval})
PY
UPLOADER_PID=$!

{steps}

# Fleet deps (not in the DLAMI/baked image): FastAPI + uvicorn.
"$VENV_PY" -c "import fastapi, uvicorn" 2>/dev/null \\
  || "$VENV_PY" -m pip install fastapi uvicorn \\
  || "$VENV_PY" -m pip install --user fastapi uvicorn

set +e
source /home/ubuntu/spot-train.env

# Advertise the private IP so the router (same SG) can dial this box directly.
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \\
  -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
PRIVATE_IP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \\
  http://169.254.169.254/latest/meta-data/local-ipv4)
export ADVERTISE_ADDR="${{PRIVATE_IP}}:{port}"

"$VENV_PY" -u -m {module} --port {port}
RC=$?

kill "$UPLOADER_PID" 2>/dev/null || true
"$VENV_PY" - <<'PY'
import boto3
try:
    boto3.client("s3").upload_file("/var/log/spot-train-boot.log", "{bucket}", "{logs_key}")
except Exception:
    pass
PY
exit "$RC"
BOOT
echo "fleet {role} exited rc=$? — box left up; run 'fleet down' to terminate it"
"""


def build_provisioning_user_data(
    cfg: OrchestratorConfig, *, run_id: str, market: str, max_seconds: int
) -> str:
    """Provision only: get the code + deps onto the box, then leave it UP for SSH.
    Does NOT run training and does NOT shut the box down."""
    env = _trainer_env(cfg, run_id=run_id, market=market, max_seconds=max_seconds)
    steps = _provision_steps(cfg, _export_block(env))
    return f"""#!/bin/bash
set -x
exec > /var/log/spot-train-boot.log 2>&1
echo "[provision] start"

sudo -u ubuntu -i bash <<'BOOT'
{steps}
echo "[provision] DONE"
echo "[provision] code: /home/ubuntu/app   env: /home/ubuntu/spot-train.env"
echo "[provision] to train by hand:"
echo "    cd /home/ubuntu/app && source /home/ubuntu/spot-train.env"
echo "    /opt/pytorch/bin/python -m spot_train.train"
BOOT
echo "[provision] user-data exited rc=$? — box left UP for SSH (no training, no shutdown)"
"""


def build_bake_user_data(cfg: OrchestratorConfig, *, bake_id: str, base_ami: str) -> str:
    """Provision a box for IMAGING (``spot-orchestrate bake-ami``): clone the repo
    + nanoGPT submodule and install boto3 — the same steps every training boot
    performs, minus anything run-specific — then write a status marker to S3 so
    the orchestrator knows to stop the box and call CreateImage.

    Deliberately bakes NO run state: no spot-train.env (each training boot writes
    its own), no dataset (stays in S3 — EBS lazy-restore would make baked bins
    first-touch-slow anyway), no credentials (the instance profile is IMDS-only,
    nothing lands on disk). The clone IS baked, and stays current because
    ``_provision_steps`` fast-forwards an existing clone at every boot."""
    bucket = cfg.bucket
    status_key = cfg.bake_status_key(bake_id)
    log_key = cfg.bake_log_key(bake_id)
    return f"""#!/bin/bash
set -x
exec > /var/log/spot-train-bake.log 2>&1
chmod 644 /var/log/spot-train-bake.log   # let the ubuntu log-uploader read it

sudo -u ubuntu -i bash <<'BOOT'
set -x
cd /home/ubuntu

# Interpreter: venv python by absolute path (same DLAMI reality as training boots).
VENV_PY=/opt/pytorch/bin/python
[ -x "$VENV_PY" ] || VENV_PY="$(command -v python3)"

# boto3 FIRST — the log uploader and the status marker need it even if the
# provisioning below fails. (Also the one pip dep the bake exists to pre-install.)
"$VENV_PY" -c "import boto3" 2>/dev/null || "$VENV_PY" -m pip install boto3 2>/dev/null \\
  || "$VENV_PY" -m pip install --user boto3 2>/dev/null || true
"$VENV_PY" - <<'PY' &
import time, boto3
c = boto3.client("s3")
while True:
    try:
        c.upload_file("/var/log/spot-train-bake.log", "{bucket}", "{log_key}")
    except Exception:
        pass
    time.sleep(10)
PY
UPLOADER_PID=$!

# Provision in a subshell so a failure is captured as RC, not a silent exit —
# the status marker below must ALWAYS be written.
(
  set -euxo pipefail
  if [ -d app/.git ]; then
    git -C app fetch --depth 1 origin {cfg.repo_branch}
    git -C app reset --hard FETCH_HEAD
  else
    git clone --depth 1 -b {cfg.repo_branch} {cfg.repo_url} app
  fi
  cd app
  git submodule update --init --depth 1
  git rev-parse HEAD > /home/ubuntu/bake-commit
  "$VENV_PY" - <<'PY'
import torch, boto3, numpy  # noqa: F401
print("bake preflight ok: torch", torch.__version__, flush=True)
PY
)
RC=$?

kill "$UPLOADER_PID" 2>/dev/null || true
"$VENV_PY" - <<PY
import json
import boto3
c = boto3.client("s3")
try:
    c.upload_file("/var/log/spot-train-bake.log", "{bucket}", "{log_key}")
except Exception:
    pass
try:
    commit = open("/home/ubuntu/bake-commit").read().strip()
except Exception:
    commit = ""
c.put_object(
    Bucket="{bucket}",
    Key="{status_key}",
    Body=json.dumps(
        {{"ok": $RC == 0, "rc": $RC, "commit": commit, "base_ami": "{base_ami}"}}
    ).encode(),
)
PY
exit "$RC"
BOOT
echo "bake user-data exited rc=$? — box left UP for the orchestrator (stop + CreateImage on ok)"
"""


def _multinode_loop(
    cfg: OrchestratorConfig, *, run_id: str, nodes: int, node_index: int, nproc: str
) -> str:
    """Bash for the multi-node ELASTIC loop: one long-lived torchrun per box.

    ``--nnodes=MIN:N`` (c10d rendezvous hosted by node 0's agent) is what lets
    survivors of a peer death keep training: the dead node's collectives abort
    on every survivor after NCCL_TIMEOUT, each box's elastic agent catches the
    worker crash and re-rendezvouses, and the round closes with whoever is
    present (>= MIN) after ``last_call_timeout`` — the group continues at N-1
    while the orchestrator's replacement boots. The replacement runs this same
    script, dials the same rendezvous, and its arrival triggers one more
    restart back to N. Every worker (re)start runs the one proven resume path;
    the node-local checkpoint tier makes a survivor's restore near-instant.

    Rendezvous discovery stays S3-based but is publish-ONCE: node 0 writes
    rdzv.json {addr, port} at boot; workers and replacements poll the same key.
    No generations, no ready markers, no per-generation budget reads — the
    run-level budget rides in the checkpoint itself (TRAIN_BUDGET_SECONDS).

    The outer while is a thin retry for the paths elastic can't absorb (restart
    budget exhausted, node 0's store lost): each attempt re-reads rdzv.json,
    and node 0 republishes on a bumped port so a TIME_WAIT socket from its own
    dead agent can't wedge the rebind. Attempts are bounded; on exhaustion the
    box exits nonzero and stays up for the orchestrator's whole-group-restart
    watchdog. metrics.json in S3 is the group-wide done signal."""
    bucket = cfg.bucket
    rdzv_key = cfg.run_rdzv_key(run_id)
    metrics_key = cfg.run_metrics_key(run_id)
    min_nodes = cfg.nodes_min_count(nodes)
    polls = 400  # x3s = ~20 min bound on the worker's rdzv.json wait
    max_attempts = 20
    if node_index == 0:
        rendezvous = f"""  # Node 0 hosts the c10d store: publish this box's address ONCE per
  # attempt (fresh port per attempt so a relaunch never fights TIME_WAIT).
  NODE_IP=$(hostname -I | awk '{{print $1}}')
  RDZV_ADDR=$NODE_IP
  PORT=$(({cfg.rdzv_port} + ATTEMPT - 1))
  "$VENV_PY" - <<PY
import json
import boto3
boto3.client("s3").put_object(
    Bucket="{bucket}", Key="{rdzv_key}",
    Body=json.dumps({{"addr": "$RDZV_ADDR", "port": $PORT}}).encode(),
)
PY
  echo "[rdzv] node 0/{nodes}: hosting elastic rendezvous at $RDZV_ADDR:$PORT"
"""
    else:
        rendezvous = f"""  # Worker: dial whatever node 0 actually published (addr/port read back
  # from rdzv.json, never assumed) — a replacement joining a LIVE group takes
  # this exact path. Exit 2 = the run finished while we waited.
  NODE_IP=$(hostname -I | awk '{{print $1}}')
  "$VENV_PY" - <<'PY' > /tmp/rdzv_join
import json, sys, time
import boto3
s3 = boto3.client("s3")
for _ in range({polls}):
    try:
        s3.head_object(Bucket="{bucket}", Key="{metrics_key}")
        sys.exit(2)
    except Exception:
        pass
    try:
        doc = json.loads(s3.get_object(Bucket="{bucket}", Key="{rdzv_key}")["Body"].read())
        print(doc["addr"], doc["port"])
        sys.exit(0)
    except Exception:
        pass
    time.sleep(3)
sys.exit(1)
PY
  WRC=$?
  if [ "$WRC" -eq 2 ]; then continue; fi
  if [ "$WRC" -ne 0 ]; then
    echo "[rdzv] node {node_index}: rdzv.json never appeared"
    break
  fi
  read RDZV_ADDR PORT < /tmp/rdzv_join
  echo "[rdzv] node {node_index}/{nodes}: dialing elastic rendezvous at $RDZV_ADDR:$PORT"
"""
    return f"""
# --- multi-node: elastic rendezvous (survivors keep training at N-1) ---------
# One torchrun rides out kills AND rejoins via --max-restarts; this outer loop
# only retries the rare failures elastic can't absorb. metrics.json is the done
# signal; MN_RC=0 only then or on a clean local exit.
MN_RC=1
ATTEMPT=0
while :; do
  "$VENV_PY" - <<'PY'
import sys
import boto3
try:
    boto3.client("s3").head_object(Bucket="{bucket}", Key="{metrics_key}")
except Exception:
    sys.exit(1)
PY
  if [ $? -eq 0 ]; then
    echo "[rdzv] metrics.json present — run complete"
    MN_RC=0
    break
  fi
  ATTEMPT=$((ATTEMPT + 1))
  if [ "$ATTEMPT" -gt {max_attempts} ]; then
    echo "[rdzv] out of torchrun attempts — leaving the box up for the watchdog"
    break
  fi
{rendezvous}  OMP_NUM_THREADS=1 "$VENV_PY" -m torch.distributed.run \\
    --nnodes={min_nodes}:{nodes} --nproc_per_node={nproc} \\
    --rdzv_backend=c10d --rdzv_endpoint="$RDZV_ADDR:$PORT" \\
    --rdzv_id={run_id} --local_addr="$NODE_IP" \\
    --rdzv_conf="last_call_timeout={cfg.rdzv_last_call_seconds}" \\
    --max-restarts={cfg.max_restarts} -m spot_train.train
  TRC=$?
  if [ "$TRC" -eq 0 ]; then
    MN_RC=0
    break
  fi
  echo "[rdzv] torchrun exited $TRC (attempt $ATTEMPT/{max_attempts}) — retrying"
  sleep 5
done
(exit "$MN_RC")
"""


def build_user_data(
    cfg: OrchestratorConfig,
    *,
    run_id: str,
    market: str,
    max_seconds: int,
    logs_key: str | None = None,
    ddp: bool = False,
    nproc_per_node: int = 0,
    nodes: int = 1,
    node_index: int = 0,
) -> str:
    """Full run: provision, then train under the wall-clock budget while syncing
    the boot log to S3 every ``log_stream_seconds`` so the orchestrator can stream
    it live without SSH. On success the box self-terminates as a cost backstop (the
    orchestrator also terminates it once it sees metrics.json). On failure the box
    stays up for debugging; the orchestrator reaps it on the metrics timeout.

    ``ddp=True`` runs the trainer under torchrun (single-node DDP); the trainer
    detects RANK and joins the process group. ``nproc_per_node <= 0`` auto-detects
    one rank per GPU on the box (torchrun --nproc_per_node=gpu); a positive value
    forces that count. ``ddp=False`` is the plain single-process launch
    (baseline/preempt) — byte-identical to before.

    ``nodes > 1`` (implies ddp) joins an N-node group via torchrun's ELASTIC
    c10d rendezvous (see ``_multinode_loop``): node 0 hosts the store and
    publishes its private IP + port to S3 (rdzv.json) once; the others —
    including replacements joining a live group — poll that key and dial.
    ``--nnodes=MIN:N`` with ``--max-restarts>0`` means a peer death only
    restarts the WORKERS: survivors re-rendezvous and keep training at N-1
    while the orchestrator's replacement boots and scales the group back to N.
    Every worker (re)start resumes via the one proven resume path — from the
    node-local checkpoint tier when this box has the agreed step, else from S3.
    Killing node 0 still downs the store: the orchestrator's whole-group
    restart watchdog is the documented fallback for that."""
    env = _trainer_env(cfg, run_id=run_id, market=market, max_seconds=max_seconds)
    if nodes > 1:
        # Short collective timeout so survivors of a node kill abort fast (see
        # distributed.init); single-node keeps torch's default.
        env["NCCL_TIMEOUT"] = str(cfg.nccl_timeout_seconds)
        # Skip torch's post-timeout debug-info dump: it added ~2 minutes to every
        # peer-death crash (observed 18:26:51 -> 18:29:10), delaying the rejoin.
        env["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "0"
        # Run-level budget: rides in the checkpoint (trained_seconds), so every
        # elastic restart computes its own remaining time — no budget.json, and
        # boot/NCCL-stall/teardown are never billed. MAX_SECONDS still gets a
        # value but the trainer overrides it from this once resumed.
        env["TRAIN_BUDGET_SECONDS"] = str(max_seconds)
        # Node-local checkpoint tier: survivors of an elastic restart resume
        # from their own disk instead of re-downloading from S3.
        env["LOCAL_CHECKPOINT_DIR"] = "/tmp/spot-ckpt"
    steps = _provision_steps(cfg, _export_block(env))
    bucket = cfg.bucket
    logs_key = logs_key or cfg.run_logs_key(run_id)
    interval = cfg.log_stream_seconds
    rdzv_block = ""
    if nodes > 1:
        nproc = "gpu" if nproc_per_node <= 0 else str(nproc_per_node)
        run_cmd = _multinode_loop(
            cfg, run_id=run_id, nodes=nodes, node_index=node_index, nproc=nproc
        )
    elif ddp:
        nproc = "gpu" if nproc_per_node <= 0 else str(nproc_per_node)
        # nproc=gpu => torchrun uses torch.cuda.device_count() (one rank per GPU).
        # OMP_NUM_THREADS=1: N gloo ranks on one CPU box otherwise spawn N×cores
        # threads and thrash. torchrun --standalone sets RANK/LOCAL_RANK/WORLD_SIZE/MASTER_*.
        run_cmd = (
            f'OMP_NUM_THREADS=1 "$VENV_PY" -m torch.distributed.run '
            f"--standalone --nproc_per_node={nproc} -m spot_train.train"
        )
    else:
        run_cmd = '"$VENV_PY" -u -m spot_train.train'
    return f"""#!/bin/bash
set -x
exec > /var/log/spot-train-boot.log 2>&1
chmod 644 /var/log/spot-train-boot.log   # let the ubuntu log-uploader read it

sudo -u ubuntu -i bash <<'BOOT'
set -x
cd /home/ubuntu

# Interpreter: use the pytorch venv python by absolute path — the base shell only
# has `python3`, and the login shell doesn't reliably auto-activate the venv, so
# a bare `python` fails (command not found). Fall back to python3 on other AMIs.
VENV_PY=/opt/pytorch/bin/python
[ -x "$VENV_PY" ] || VENV_PY="$(command -v python3)"

# Start the S3 log uploader FIRST — before clone/pip — so the *entire* boot
# (clone, submodule, deps, preflight, then training) streams to the orchestrator,
# not just the training tail. boto3 is required for it, so ensure it up front
# (the provision steps below install it too; this is idempotent). Uses the
# instance-profile role via IMDS — no creds in user-data.
"$VENV_PY" -c "import boto3" 2>/dev/null || "$VENV_PY" -m pip install boto3 2>/dev/null \
  || "$VENV_PY" -m pip install --user boto3 2>/dev/null || true
"$VENV_PY" - <<'PY' &
import time, boto3
c = boto3.client("s3")
while True:
    try:
        c.upload_file("/var/log/spot-train-boot.log", "{bucket}", "{logs_key}")
    except Exception:
        pass
    time.sleep({interval})
PY
UPLOADER_PID=$!

# Provision (clone repo + nanoGPT submodule, deps, env file, preflight) — now streamed.
{steps}

# From here we manage exit codes by hand (final log flush + teardown), so drop -e.
set +e
# Load the run config the trainer reads (PYTHONPATH so `spot_train` imports, plus
# the S3 URIs, MAX_SECONDS, etc.). Provisioning wrote this file but doesn't source it.
source /home/ubuntu/spot-train.env
{rdzv_block}
{run_cmd}
RC=$?

# Stop the uploader and push one final copy so the tail of the run is in S3.
kill "$UPLOADER_PID" 2>/dev/null || true
"$VENV_PY" - <<'PY'
import boto3
try:
    boto3.client("s3").upload_file("/var/log/spot-train-boot.log", "{bucket}", "{logs_key}")
except Exception:
    pass
PY
exit "$RC"
BOOT
RC=$?

if [ "$RC" -eq 0 ]; then
  shutdown -h now   # cost backstop; orchestrator also terminates on metrics.json
else
  echo "training exited $RC — leaving instance up for debugging; orchestrator reaps on timeout"
fi
"""
