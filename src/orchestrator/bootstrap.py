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

Two builders:

  * ``build_provisioning_user_data`` — clone the repo + nanoGPT submodule and
    install deps, write a ``source``-able env file, then STOP and leave the box
    UP. You ssh in, confirm the code is there, and run training by hand. Used
    while bringing the box up.
  * ``build_user_data`` — the full run: provision, then ``spot_train.train``
    under a wall-clock budget, then self-terminate on success. Used once the box
    is proven. (Parked while we validate provisioning over SSH.)

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

# Code: clone the repo (idempotent) and populate the nanoGPT submodule. A plain
# clone leaves third_party/nanoGPT EMPTY, and the trainer imports GPT from there,
# so the submodule init is not optional.
[ -d app ] || git clone --depth 1 -b {cfg.repo_branch} {cfg.repo_url} app
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


def _multinode_loop(
    cfg: OrchestratorConfig, *, run_id: str, nodes: int, node_index: int, nproc: str
) -> str:
    """Bash for the multi-node GENERATION loop: rendezvous, torchrun, and rejoin
    after a peer dies. The surviving instance never reboots — only its process
    restarts; between generations it pauses in a cheap S3 poll (GPU idle).

    Each *generation* is one attempt of the full group, coordinated through S3:
    non-master nodes write a per-generation ready marker and dial only what node 0
    has actually published; node 0 publishes rdzv.json (fresh port = base+gen, so
    TIME_WAIT on its own relaunch can't bite) only once every worker's marker for
    that generation exists — so its TCPStore comes up seconds before the workers
    dial, and nobody busy-retries against a box that is still booting (the
    timeout-retry livelock this design exists to prevent). A crash sends every
    node back to the top of the loop, where all of them — survivors reading their
    own last generation, replacements reading the dead group's stale rdzv.json —
    independently converge on generation+1. metrics.json appearing in S3 is the
    group-wide done signal.

    The training budget is ORCHESTRATOR-authoritative: before each generation the
    box reads runs/<run_id>/budget.json (recomputed by the orchestrator after
    every kill from observed training time) and exports it as MAX_SECONDS —
    boot, the NCCL stall, and crash teardown are never billed, and there is no
    local wall-clock arithmetic to drift. The value is clamped to >= 1 and the
    loop NEVER exits for lack of budget: rank 0 must always be able to re-form
    the group so the coordinated stop -> eval -> metrics.json can land. Every
    wait is bounded (~20 min): on exhaustion the loop exits nonzero and the box
    is left up for debugging, and the orchestrator's recovery watchdog falls
    back to a whole-group restart."""
    bucket = cfg.bucket
    rdzv_key = cfg.run_rdzv_key(run_id)
    metrics_key = cfg.run_metrics_key(run_id)
    ready_prefix = cfg.run_ready_prefix(run_id)
    budget_key = cfg.run_budget_key(run_id)
    polls = 400  # x3s = ~20 min bound on any single wait
    if node_index == 0:
        rendezvous = f"""  # Master: wait for every worker's gen-ready marker (exit 2 = the run
  # finished while we waited; 1 = bound exhausted), then publish and host.
  "$VENV_PY" - <<PY
import sys, time
import boto3
s3 = boto3.client("s3")
for _ in range({polls}):
    try:
        s3.head_object(Bucket="{bucket}", Key="{metrics_key}")
        sys.exit(2)
    except Exception:
        pass
    got = 0
    for i in range(1, {nodes}):
        try:
            s3.head_object(Bucket="{bucket}", Key="{ready_prefix}gen$GEN-node%d" % i)
            got += 1
        except Exception:
            pass
    if got >= {nodes} - 1:
        sys.exit(0)
    time.sleep(3)
sys.exit(1)
PY
  WRC=$?
  if [ "$WRC" -eq 2 ]; then continue; fi
  if [ "$WRC" -ne 0 ]; then
    echo "[rdzv] node 0: gave up waiting for gen $GEN ready markers"
    break
  fi
  RDZV_ADDR=$(hostname -I | awk '{{print $1}}')
  "$VENV_PY" - <<PY
import json
import boto3
boto3.client("s3").put_object(
    Bucket="{bucket}", Key="{rdzv_key}",
    Body=json.dumps(
        {{"addr": "$RDZV_ADDR", "port": $PORT, "generation": $GEN, "node_count": {nodes}}}
    ).encode(),
)
PY
  echo "[rdzv] node 0/{nodes}: published gen $GEN at $RDZV_ADDR:$PORT"
"""
    else:
        rendezvous = f"""  # Worker: announce readiness for this generation, then dial whatever
  # node 0 actually publishes (its addr/port/gen — never assumptions).
  "$VENV_PY" - <<PY
import boto3
boto3.client("s3").put_object(
    Bucket="{bucket}", Key="{ready_prefix}gen$GEN-node{node_index}", Body=b"1"
)
PY
  echo "[rdzv] node {node_index}/{nodes}: ready for gen $GEN, waiting for publication"
  "$VENV_PY" - <<PY > /tmp/rdzv_join
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
        if int(doc.get("generation", 0)) >= $GEN:
            print(doc["addr"], doc["port"], doc["generation"])
            sys.exit(0)
    except Exception:
        pass
    time.sleep(3)
sys.exit(1)
PY
  WRC=$?
  if [ "$WRC" -eq 2 ]; then continue; fi
  if [ "$WRC" -ne 0 ]; then echo "[rdzv] node {node_index}: never saw gen $GEN published"; break; fi
  read RDZV_ADDR PORT GEN < /tmp/rdzv_join
  echo "[rdzv] node {node_index}/{nodes}: joining gen $GEN at $RDZV_ADDR:$PORT"
"""
    return f"""
# --- multi-node: generation rendezvous + rejoin loop -------------------------
# A dead peer crashes torchrun on every survivor (NCCL_TIMEOUT); this loop then
# PAUSES the box until the orchestrator's replacement node is ready and rejoins
# at the next generation. metrics.json is the done signal; MN_RC=0 only then or
# on a clean local exit.
MN_RC=1
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
  # Budget: read the orchestrator's authoritative remaining seconds. 0/unreadable
  # falls back to the last known MAX_SECONDS; clamp >= 1 — NEVER exit for lack of
  # budget, or rank 0 could no longer re-form the group to eval + write metrics.
  REMAINING=$("$VENV_PY" - <<'PY'
import json
import boto3
try:
    body = boto3.client("s3").get_object(Bucket="{bucket}", Key="{budget_key}")["Body"].read()
    print(int(json.loads(body)["remaining_seconds"]))
except Exception:
    print(0)
PY
)
  [ "$REMAINING" -ge 1 ] || REMAINING=$MAX_SECONDS
  [ "$REMAINING" -ge 1 ] || REMAINING=1
  export MAX_SECONDS=$REMAINING
  echo "[rdzv] budget: $MAX_SECONDS seconds remain"
  # Next generation = one past whatever is currently published (0 if nothing) —
  # survivors and fresh replacements independently agree on this number.
  G_PUB=$("$VENV_PY" - <<'PY'
import json
import boto3
try:
    body = boto3.client("s3").get_object(Bucket="{bucket}", Key="{rdzv_key}")["Body"].read()
    print(int(json.loads(body).get("generation", 0)))
except Exception:
    print(0)
PY
)
  GEN=$((G_PUB + 1))
  PORT=$(({cfg.rdzv_port} + GEN))
{rendezvous}  OMP_NUM_THREADS=1 "$VENV_PY" -m torch.distributed.run \\
    --nnodes={nodes} --nproc_per_node={nproc} --node_rank={node_index} \\
    --master_addr="$RDZV_ADDR" --master_port="$PORT" \\
    --max-restarts=0 -m spot_train.train
  TRC=$?
  if [ "$TRC" -eq 0 ]; then
    MN_RC=0
    break
  fi
  echo "[rdzv] torchrun exited $TRC at gen $GEN — pausing to regroup"
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

    ``nodes > 1`` (implies ddp) joins an N-node group via torchrun's STATIC
    rendezvous inside the generation loop built by ``_multinode_loop``: node 0
    publishes its private IP + a generation number to S3 (rdzv.json), the others
    poll it, and every node runs with ``--node_rank`` + ``--master_addr`` so
    global rank 0 and the worker-group store are always on node 0 (c10d elastic
    assigned both arbitrarily — killing the wrong node stranded survivors dialing
    a dead store). ``--max-restarts=0``: a peer death crashes the survivors'
    collectives after NCCL_TIMEOUT and the agents exit — but the loop keeps the
    surviving BOX alive, pausing until the orchestrator's replacement node is up
    and rejoining at the next generation; every restart of train() resumes from
    the S3 checkpoint (the one proven resume path)."""
    env = _trainer_env(cfg, run_id=run_id, market=market, max_seconds=max_seconds)
    if nodes > 1:
        # Short collective timeout so survivors of a node kill abort fast (see
        # distributed.init); single-node keeps torch's default.
        env["NCCL_TIMEOUT"] = str(cfg.nccl_timeout_seconds)
        # Skip torch's post-timeout debug-info dump: it added ~2 minutes to every
        # peer-death crash (observed 18:26:51 -> 18:29:10), delaying the rejoin.
        env["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "0"
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
