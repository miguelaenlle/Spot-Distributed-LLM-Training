"""Fleet lifecycle (ROADMAP Part 1) — `spot-orchestrate fleet up|status|down|kill-worker`.

Local mode (``--local``) boots the whole fleet as OS processes on this machine:
N workers + 1 router, with a local directory standing in for the S3 heartbeat
store. Same code the cloud runs — uvicorn, registry, retries — so the reroute
experiment works with zero cloud spend. ``kill-worker`` SIGKILLs a worker
(the local stand-in for a spot reclaim): its heartbeat goes stale, the router
drops it, in-flight requests reroute.

Cloud mode launches N **spot** workers + 1 on-demand router on EC2 (same AMI +
user-data pattern as training boxes; heartbeats live under
``s3://bucket/fleet/<fleet_id>/workers/``). ``kill-worker`` is a real
``TerminateInstances`` — the controlled stand-in for a spot reclaim, exactly
like the training experiments. Discovery is by EC2 tags (``fleet`` /
``fleet_role``), so ``status``/``down``/``kill-worker`` need no local state.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

from .config import OrchestratorConfig

STATE_DIR = ".fleet/local"
FLEET_TAG = "fleet"
ROLE_TAG = "fleet_role"


def _state_path(name: str) -> str:
    return os.path.join(STATE_DIR, name)


def _read_state() -> dict | None:
    try:
        with open(_state_path("fleet.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _spawn(module: str, port: int, env: dict, log_name: str) -> int:
    log_path = _state_path(os.path.join("logs", log_name))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "ab")  # noqa: SIM115 — handed to the child process
    proc = subprocess.Popen(
        [sys.executable, "-m", module, "--port", str(port)],
        env={**os.environ, **env},
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # our SIGTERM/SIGKILL, not ctrl-C, ends them
    )
    log.close()
    return proc.pid


def up_local(
    workers: int = 2,
    router_port: int = 8000,
    base_worker_port: int = 8001,
    checkpoint_uri: str = "checkpoints/",
    data_local_dir: str = "third_party/nanoGPT/data/shakespeare_char",
) -> None:
    if _read_state() is not None:
        print("[fleet] a local fleet is already up — run `fleet down --local` first")
        return
    workers_uri = os.path.abspath(_state_path("workers"))
    os.makedirs(workers_uri, exist_ok=True)

    state: dict = {"router": {}, "workers": [], "workers_uri": workers_uri}
    for i in range(workers):
        port = base_worker_port + i
        worker_id = f"local-w{i}"
        pid = _spawn(
            "inference.worker",
            port,
            {
                "WORKER_ID": worker_id,
                "ADVERTISE_ADDR": f"127.0.0.1:{port}",
                "FLEET_WORKERS_URI": workers_uri,
                "CHECKPOINT_URI": checkpoint_uri,
                "DATA_LOCAL_DIR": data_local_dir,
                "HOST": "127.0.0.1",
                # N workers share this machine's cores; unbounded torch threads
                # make them fight and *lower* total throughput. Cloud workers
                # (one box each) don't inherit this.
                "OMP_NUM_THREADS": "2",
            },
            f"worker-{i}.log",
        )
        state["workers"].append({"worker_id": worker_id, "port": port, "pid": pid})
        print(f"[fleet] worker {worker_id} pid={pid} port={port}")

    router_pid = _spawn(
        "inference.router",
        router_port,
        {"FLEET_WORKERS_URI": workers_uri, "HOST": "127.0.0.1"},
        "router.log",
    )
    state["router"] = {"pid": router_pid, "port": router_port}
    print(f"[fleet] router pid={router_pid} port={router_port}")

    with open(_state_path("fleet.json"), "w") as f:
        json.dump(state, f, indent=2)

    _wait_router(router_port, expect_workers=workers)
    print(f"[fleet] up — try: curl -s localhost:{router_port}/fleet/status")
    print(
        f"[fleet]        curl -s localhost:{router_port}/v1/completions "
        f'-H "content-type: application/json" '
        f'-d \'{{"prompt": "ROMEO:", "max_tokens": 64}}\''
    )


def _wait_router(port: int, expect_workers: int, timeout: float = 120.0) -> None:
    """Wait for the router to come up and see every worker's heartbeat.

    Workers can take a while on first boot (checkpoint download / model init),
    so this is generous; logs are in .fleet/local/logs/ if it times out.
    """
    import requests

    deadline = time.time() + timeout
    seen = -1
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/fleet/status", timeout=2)
            live = r.json().get("live_workers", 0)
            if live != seen:
                print(f"[fleet] router sees {live}/{expect_workers} workers")
                seen = live
            if live >= expect_workers:
                return
        except requests.RequestException:
            pass
        time.sleep(1.0)
    print(
        f"[fleet] WARNING: router did not see {expect_workers} workers within {timeout:.0f}s "
        f"— check {_state_path('logs')}/",
        file=sys.stderr,
    )


def status_local() -> None:
    state = _read_state()
    if state is None:
        print("[fleet] no local fleet is up")
        return
    import requests

    port = state["router"]["port"]
    for w in state["workers"]:
        run = "up" if _alive(w["pid"]) else "DEAD"
        print(f"[fleet] worker {w['worker_id']} pid={w['pid']} port={w['port']} [{run}]")
    try:
        r = requests.get(f"http://127.0.0.1:{port}/fleet/status", timeout=3)
        print(json.dumps(r.json(), indent=2))
    except requests.RequestException as e:
        print(f"[fleet] router unreachable on :{port}: {e}")


def kill_worker_local(worker_id: str | None = None) -> None:
    """SIGKILL one worker — the local stand-in for a spot reclaim (no warning,
    no cleanup; the router must notice via the stale heartbeat)."""
    state = _read_state()
    if state is None:
        print("[fleet] no local fleet is up")
        return
    candidates = [w for w in state["workers"] if _alive(w["pid"])]
    if worker_id:
        candidates = [w for w in candidates if w["worker_id"] == worker_id]
    if not candidates:
        print(f"[fleet] no live worker to kill (worker_id={worker_id or 'any'})")
        return
    victim = candidates[-1]
    os.kill(victim["pid"], signal.SIGKILL)
    print(f"[fleet] killed {victim['worker_id']} (pid={victim['pid']}) — watch the router reroute")


def down_local() -> None:
    state = _read_state()
    if state is None:
        print("[fleet] no local fleet is up")
        return
    pids = [state["router"]["pid"]] + [w["pid"] for w in state["workers"]]
    for pid in pids:
        if _alive(pid):
            os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 10
    while time.time() < deadline and any(_alive(p) for p in pids):
        time.sleep(0.2)
    for pid in pids:
        if _alive(pid):
            os.kill(pid, signal.SIGKILL)
    # Clear registry docs + state so the next `up` starts clean.
    workers_dir = state.get("workers_uri", "")
    if workers_dir and os.path.isdir(workers_dir):
        for name in os.listdir(workers_dir):
            if name.endswith(".json"):
                os.remove(os.path.join(workers_dir, name))
    os.remove(_state_path("fleet.json"))
    print("[fleet] down")


# --------------------------------------------------------------------------- #
# Cloud mode (EC2 spot workers + on-demand router)
# --------------------------------------------------------------------------- #
def _discover(cfg: OrchestratorConfig) -> tuple[list[dict], list[dict]]:
    """(workers, routers) — all non-terminated fleet instances, by tag."""
    from . import aws

    return (
        aws.instances_by_tag(ROLE_TAG, "worker"),
        aws.instances_by_tag(ROLE_TAG, "router"),
    )


def up_cloud(cfg: OrchestratorConfig, *, workers: int, run_id: str) -> None:
    """Launch the serving fleet: 1 on-demand router + N spot workers serving
    ``run_id``'s latest checkpoint. Prints the router's public endpoint."""
    from . import aws, bootstrap

    cfg.require_bucket()
    if not run_id:
        raise SystemExit("fleet up (cloud) needs --run <run_id> — the checkpoint to serve")
    ckpt_prefix = f"{cfg.run_prefix}/{run_id}/checkpoints/"
    if not aws.is_dry_run() and not aws.any_object_under(cfg.bucket, ckpt_prefix):
        raise SystemExit(
            f"run {run_id!r} has no checkpoints under s3://{cfg.bucket}/{ckpt_prefix} — "
            "serve a completed training run"
        )
    existing_workers, existing_routers = _discover(cfg)
    if existing_workers or existing_routers:
        raise SystemExit(
            f"a fleet is already up ({len(existing_workers)} workers, "
            f"{len(existing_routers)} routers) — run `fleet down` first"
        )

    ami = aws.resolve_ami(cfg.ami_id, cfg.ami_name_filter)
    sg = aws.ensure_security_group(cfg.security_group, cfg.region)
    aws.authorize_port(
        sg, cfg.fleet_router_port, cfg.fleet_ingress_cidr, "fleet router (completions API)"
    )
    fleet_id = time.strftime("fleet-%Y%m%d-%H%M%S")
    print(
        f"[fleet] id={fleet_id} run={run_id} workers={workers}x{cfg.fleet_worker_instance_type}"
        f" ({cfg.fleet_market}) router=1x{cfg.fleet_router_instance_type} (on-demand)"
    )

    router_id = aws.launch(
        ami_id=ami,
        instance_type=cfg.fleet_router_instance_type,
        profile_name=cfg.instance_profile,
        security_group_id=sg,
        user_data=bootstrap.build_fleet_user_data(
            cfg,
            fleet_id=fleet_id,
            role="router",
            logs_key=cfg.fleet_logs_key(fleet_id, "router"),
            port=cfg.fleet_router_port,
        ),
        market="on-demand",
        run_id=f"{fleet_id}-router",
        key_name=cfg.key_name,
        extra_tags={FLEET_TAG: fleet_id, ROLE_TAG: "router"},
    )
    worker_ids = []
    for i in range(workers):
        wid = f"{fleet_id}-w{i}"
        worker_ids.append(
            aws.launch(
                ami_id=ami,
                instance_type=cfg.fleet_worker_instance_type,
                profile_name=cfg.instance_profile,
                security_group_id=sg,
                user_data=bootstrap.build_fleet_user_data(
                    cfg,
                    fleet_id=fleet_id,
                    role="worker",
                    worker_id=wid,
                    run_id=run_id,
                    logs_key=cfg.fleet_logs_key(fleet_id, f"worker-{i}"),
                    port=cfg.fleet_worker_port,
                ),
                market=cfg.fleet_market,
                run_id=wid,
                key_name=cfg.key_name,
                extra_tags={FLEET_TAG: fleet_id, ROLE_TAG: "worker", "worker_id": wid},
            )
        )
    for iid in [router_id, *worker_ids]:
        aws.wait_running(iid)
    aws.put_text(
        cfg.bucket,
        cfg.fleet_state_key(fleet_id),
        json.dumps(
            {"fleet_id": fleet_id, "run_id": run_id, "router": router_id, "workers": worker_ids},
            indent=2,
        ),
    )

    router_ip = aws.public_ip(router_id)
    endpoint = f"http://{router_ip}:{cfg.fleet_router_port}"
    print(f"[fleet] router {router_id} at {endpoint}")
    if aws.is_dry_run():
        return
    _wait_cloud_ready(endpoint, expect_workers=workers)
    print(f"[fleet] up — status:  curl -s {endpoint}/fleet/status")
    print(
        f"[fleet] completions:  curl -s {endpoint}/v1/completions "
        "-H 'content-type: application/json' "
        '-d \'{"prompt": "ROMEO:", "max_tokens": 64}\''
    )
    print(f"[fleet] loadgen:      cd loadgen && go run . -url {endpoint} -rps 4 -duration 60s")


def _wait_cloud_ready(endpoint: str, expect_workers: int, timeout: float = 900.0) -> None:
    """Poll the router's public /fleet/status until every worker heartbeats.
    Boot = instance start + clone + pip + checkpoint download; be generous."""
    import requests

    print(f"[fleet] waiting for {expect_workers} workers to heartbeat (boot takes ~3-8 min)...")
    deadline = time.time() + timeout
    seen = -1
    while time.time() < deadline:
        try:
            live = requests.get(f"{endpoint}/fleet/status", timeout=3).json().get("live_workers", 0)
            if live != seen:
                print(f"[fleet] router sees {live}/{expect_workers} workers")
                seen = live
            if live >= expect_workers:
                return
        except Exception:
            pass
        time.sleep(5.0)
    print(
        f"[fleet] WARNING: not all workers ready within {timeout:.0f}s — check the "
        "boot logs under s3://<bucket>/fleet/<fleet_id>/logs/",
        file=sys.stderr,
    )


def status_cloud(cfg: OrchestratorConfig) -> None:
    workers, routers = _discover(cfg)
    if not workers and not routers:
        print("[fleet] no cloud fleet instances found (tags fleet_role=worker|router)")
        return
    for r in routers:
        endpoint = f"http://{r['public_ip']}:{cfg.fleet_router_port}" if r["public_ip"] else "?"
        print(f"[fleet] router {r['id']} [{r['state']}] {r['type']} {endpoint}")
    for w in workers:
        print(
            f"[fleet] worker {w['tags'].get('worker_id', w['id'])} {w['id']} "
            f"[{w['state']}] {w['type']} {w['private_ip']}"
        )
    for r in routers:
        if r["state"] == "running" and r["public_ip"]:
            import requests

            try:
                doc = requests.get(
                    f"http://{r['public_ip']}:{cfg.fleet_router_port}/fleet/status", timeout=5
                ).json()
                print(json.dumps(doc, indent=2))
            except Exception as e:
                print(f"[fleet] router API not reachable yet: {e}")


def kill_worker_cloud(cfg: OrchestratorConfig, worker_id: str | None = None) -> None:
    """TerminateInstances one spot worker — the controlled spot-reclaim stand-in
    (same mechanism as the training preemption experiments)."""
    from . import aws

    workers, _ = _discover(cfg)
    candidates = [w for w in workers if w["state"] == "running"]
    if worker_id:
        candidates = [w for w in candidates if w["tags"].get("worker_id") == worker_id]
    if not candidates:
        print(f"[fleet] no running worker to kill (worker_id={worker_id or 'any'})")
        return
    victim = candidates[-1]
    aws.terminate(victim["id"])
    print(
        f"[fleet] terminated {victim['tags'].get('worker_id', victim['id'])} "
        f"({victim['id']}) — the router drops it within the heartbeat TTL"
    )


def down_cloud(cfg: OrchestratorConfig) -> None:
    from . import aws

    workers, routers = _discover(cfg)
    if not workers and not routers:
        print("[fleet] nothing to terminate")
        return
    for inst in [*workers, *routers]:
        aws.terminate(inst["id"])
    print(f"[fleet] terminated {len(workers)} workers + {len(routers)} router(s)")
