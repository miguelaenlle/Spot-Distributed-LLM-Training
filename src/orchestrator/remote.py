"""Durable orchestrator: run the epoch supervisor / a sweep OFF the laptop, on a
t3.micro kept alive by a desired=1 Auto Scaling Group for the job's duration.

Two halves live here:

  * **Laptop side** — the python scripts you run: ``launch`` (create the IAM
    profile + launch template + ASG), ``teardown`` (delete them), ``status``
    (observe), and ``kill_orchestrator`` (fault injector: terminate the live box,
    leave the ASG so it self-heals). These call only :mod:`orchestrator.aws`, so
    ``--dry-run`` touches no AWS API.
  * **Box side** — ``run_on_box`` / ``main``: the entrypoint the t3.micro runs. It
    bumps a generation marker (gen>1 ⇒ the ASG relaunched a fresh box), does a
    cold recovery if it's a relaunch-into-an-interrupted-job, dispatches to the
    same :mod:`orchestrator.experiments` code the laptop would run, and — on
    completion — writes a done-sentinel and self-scales its own ASG to 0 so it
    won't relaunch. The box uses its instance-profile role; no creds in user-data.

Why this is cheap: the system is already S3-centric — GPU-box sidecars poll
``epoch.json`` in S3 and the ``logs`` dashboard polls S3 by run_id, so only the
*writer* moves. ``logs``/``compare`` keep working unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

from . import aws, bootstrap
from .config import OrchestratorConfig

# Experiments the durable orchestrator can drive. Single-run ones pin the run_id
# to the job id (so `logs <job_id>` works immediately); sweeps generate their own
# per-point run_ids and are addressed by the sweep id.
_SINGLE_RUN = ("multinode", "multinode-shrink", "multinode-preempt")
_SWEEPS = ("scaling-clean", "scaling-preempt")
_EXPERIMENTS = _SINGLE_RUN + _SWEEPS


def _log(msg: str) -> None:
    print(f"[remote] {msg}", file=sys.stderr)


def _validate(experiment: str) -> None:
    if experiment not in _EXPERIMENTS:
        raise SystemExit(
            f"remote: unknown experiment {experiment!r} — choose one of {', '.join(_EXPERIMENTS)}"
        )


# --------------------------------------------------------------------------- #
# Laptop side — launch / teardown / status / fault-inject
# --------------------------------------------------------------------------- #
def launch(cfg: OrchestratorConfig, experiment: str) -> str:
    """Provision the durable orchestrator for one job and return its job id. The
    ASG boots a t3.micro that runs ``run_on_box`` for ``experiment``."""
    _validate(experiment)
    cfg.require_bucket()
    job_id = f"{experiment}-{int(time.time())}"
    tags = {"Name": f"spot-orch-{job_id}", "project": "spot-orch", "job": job_id}

    account = aws.caller_account_id()
    aws.ensure_orchestrator_profile(
        cfg.orchestrator_role_name, cfg.orchestrator_instance_profile, cfg.bucket, account
    )
    sg_id = aws.ensure_security_group(cfg.security_group, cfg.region)
    ami = aws.resolve_ami(cfg.ami_id, cfg.ami_name_filter)
    user_data = bootstrap.build_orchestrator_user_data(cfg, job_id=job_id, experiment=experiment)

    lt = cfg.orchestrator_lt_name(job_id)
    aws.ensure_launch_template(
        lt,
        ami_id=ami,
        instance_type=cfg.orchestrator_instance_type,
        profile_name=cfg.orchestrator_instance_profile,
        security_group_id=sg_id,
        user_data=user_data,
        key_name=cfg.key_name,
        tags=tags,
    )
    asg = cfg.orchestrator_asg_name(job_id)
    aws.ensure_auto_scaling_group(
        asg,
        launch_template_name=lt,
        subnet_ids=aws.default_subnet_ids(),
        min_size=1,
        max_size=1,
        desired=1,
        tags=tags,
    )
    _log(f"launched job {job_id} on {cfg.orchestrator_instance_type} (ASG {asg}, desired=1)")
    _log("a freshly-created instance profile can take ~10s to propagate; the ASG retries.")
    watch = job_id if experiment in _SINGLE_RUN else f"<run_id from `remote-status {job_id}`>"
    _log(f"watch:   spot-orchestrate remote-status {job_id}")
    _log(f"         spot-orchestrate logs {watch}")
    _log(f"teardown: spot-orchestrate remote-down {job_id}   (also your abort switch)")
    return job_id


def teardown(cfg: OrchestratorConfig, job_id: str) -> None:
    """Delete the ASG (force-terminating any live box) + launch template.
    Idempotent — safe after the box already self-scaled to 0."""
    asg = cfg.orchestrator_asg_name(job_id)
    lt = cfg.orchestrator_lt_name(job_id)
    aws.delete_asg(asg, force=True)
    aws.delete_launch_template(lt)
    _log(f"torn down job {job_id}: ASG {asg} + launch template {lt} deleted")


def kill_orchestrator(cfg: OrchestratorConfig, job_id: str) -> None:
    """FAULT INJECTOR: terminate the live orchestrator box but leave the ASG at
    desired=1 so it self-heals with a fresh box. Proves the durable control plane
    survives its own death (see the fault-injection experiments)."""
    asg = cfg.orchestrator_asg_name(job_id)
    info = aws.describe_asg(asg)
    if not info or not info["instances"]:
        _log(f"no live instance in ASG {asg} — nothing to kill")
        return
    for inst in info["instances"]:
        aws.terminate(inst["id"])
    _log(
        f"killed {len(info['instances'])} orchestrator box(es) in {asg}; "
        "ASG desired=1 will relaunch a fresh one (watch generation bump)."
    )


def status(cfg: OrchestratorConfig, job_id: str) -> None:
    """Print the ASG state, the control-plane generation, and — for sweeps — which
    point is running, plus whether the job is done."""
    asg = cfg.orchestrator_asg_name(job_id)
    info = aws.describe_asg(asg)
    print(f"job: {job_id}")
    if info is None:
        print(f"  ASG {asg}: (absent — not launched or already torn down)")
    else:
        print(f"  ASG {asg}: desired={info['desired']} min={info['min']} max={info['max']}")
        for i in info["instances"]:
            print(f"    box {i['id']}  {i['state']}  health={i['health']}")
    gen_key = cfg.orchestrator_generation_key(job_id)
    if aws.object_exists(cfg.bucket, gen_key):
        print(f"  generation: {aws.get_text(cfg.bucket, gen_key).strip()}  (>1 ⇒ ASG relaunched)")
    manifest_key = cfg.sweep_manifest_key(job_id)
    if aws.object_exists(cfg.bucket, manifest_key):
        try:
            man = json.loads(aws.get_text(cfg.bucket, manifest_key))
            print(f"  sweep points: {json.dumps(man.get('points', []))}")
        except Exception:  # noqa: BLE001 — manifest may be mid-write
            pass
    done_key = cfg.orchestrator_done_key(job_id)
    if aws.object_exists(cfg.bucket, done_key):
        print(f"  DONE: {aws.get_text(cfg.bucket, done_key).strip()}")
    else:
        print("  status: running (no done-sentinel yet)")


# --------------------------------------------------------------------------- #
# Box side — the entrypoint the t3.micro runs
# --------------------------------------------------------------------------- #
def _bump_generation(cfg: OrchestratorConfig, job_id: str) -> int:
    """Increment and return the control-plane generation. One box runs at a time
    (ASG desired=1), so a read-modify-write is safe enough — the counter only has
    to prove a relaunch happened, not be transactionally exact."""
    key = cfg.orchestrator_generation_key(job_id)
    prev = 0
    if aws.object_exists(cfg.bucket, key):
        try:
            prev = int(aws.get_text(cfg.bucket, key).strip())
        except Exception:  # noqa: BLE001 — treat unreadable/malformed as gen 0
            prev = 0
    gen = prev + 1
    aws.put_text(cfg.bucket, key, str(gen))
    return gen


def _mark_orphans_killed(cfg: OrchestratorConfig, run_id: str, orphans: list[dict]) -> None:
    """Make the orchestrator's death FIRST-CLASS in the run timeline: for each
    orphan GPU box, emit a ``killed cause=orchestrator-restart`` [event] into
    orchestrator.log (which the Gantt/events parser reads), attributed to its node
    via nodes/node<i>.json. The events are APPENDED to the existing orchestrator.log
    so the pre-death narrative survives the fresh supervisor's writes — combine with
    the seeding in Supervisor.__init__. Best-effort: observability never blocks
    recovery."""
    import io

    from spot_train import events

    try:
        id2node: dict[str, int] = {}
        for key in aws.list_keys(cfg.bucket, cfg.run_nodes_prefix(run_id)):
            with contextlib.suppress(Exception):
                doc = json.loads(aws.get_text(cfg.bucket, key))
                if doc.get("instance_id"):
                    # runs/<run_id>/nodes/node<i>.json — the last "node" is the filename's
                    id2node[doc["instance_id"]] = int(key.rsplit("node", 1)[-1].split(".")[0])
        lines: list[str] = []
        for b in orphans:
            node = id2node.get(b["id"])
            if node is None:
                continue
            buf = io.StringIO()
            events.emit("killed", by="orch", node=node, cause="orchestrator-restart", stream=buf)
            lines.append(buf.getvalue().rstrip("\n"))
        if not lines:
            return
        banner = (
            f"[{time.strftime('%H:%M:%S')}] [supervisor] === orchestrator restarted "
            "(previous control plane died); cold recovery ==="
        )
        key = cfg.run_orch_log_key(run_id)
        prior = aws.get_text(cfg.bucket, key) if aws.object_exists(cfg.bucket, key) else ""
        head = prior.rstrip("\n") + "\n" if prior.strip() else ""
        aws.put_text(cfg.bucket, key, head + banner + "\n" + "\n".join(lines) + "\n")
        _log(f"cold recovery: recorded {len(lines)} killed event(s) in the timeline")
    except Exception:  # noqa: BLE001 — a timeline annotation must never block recovery
        pass


def _cold_recovery(cfg: OrchestratorConfig, experiment: str) -> None:
    """A fresh box relaunched into an interrupted job: the dead box's `finally`
    never ran, so its GPU boxes may still be up. Terminate them (the orchestrator
    runs one job at a time, so every live spot-train box is this job's orphan) and,
    for a single run, clear the stale epoch/status so the fresh supervisor starts
    clean. Single runs then resume from the last S3 checkpoint (trainer's one
    resume path); sweeps restart with fresh per-point run_ids."""
    orphans = [
        b
        for b in aws.instances_by_tag("project", "spot-train")
        if b["state"] in ("pending", "running")
    ]
    run_id = os.environ.get("REMOTE_RUN_ID", "")
    # Annotate the timeline BEFORE terminating, so the kill marker lands at the
    # recovery moment on each node's row (single runs only — sweeps restart fresh).
    if experiment in _SINGLE_RUN and run_id and orphans:
        _mark_orphans_killed(cfg, run_id, orphans)
    for b in orphans:
        aws.terminate(b["id"])
    _log(f"cold recovery: terminated {len(orphans)} orphan GPU box(es)")
    if experiment in _SINGLE_RUN and run_id:
        aws.delete_object(cfg.bucket, cfg.run_epoch_key(run_id))
        aws.delete_object(cfg.bucket, cfg.run_status_key(run_id))
        _log(f"cold recovery: cleared stale epoch/status for {run_id} (resume from checkpoint)")


def _dispatch(cfg: OrchestratorConfig, experiment: str) -> None:
    from . import experiments

    fn = {
        "multinode": experiments.run_multinode,
        "multinode-shrink": experiments.run_multinode_shrink,
        "multinode-preempt": experiments.run_multinode_preempt,
        "scaling-clean": experiments.run_scaling_clean,
        "scaling-preempt": experiments.run_scaling_preempt,
    }[experiment]
    fn(cfg)


def run_on_box(cfg: OrchestratorConfig) -> int:
    """Drive the whole job from the durable box. Bump the generation, cold-recover
    if this is a relaunch, dispatch to the experiment, then ALWAYS write the
    done-sentinel and self-scale this ASG to 0. A genuine box death (kill -9 /
    instance terminate) never reaches the `finally`, so only real death leaves the
    ASG at desired=1 to self-heal; a completed or failed job scales to 0 and stops."""
    experiment = os.environ.get("EXPERIMENT", "")
    _validate(experiment)
    job_id = os.environ.get("ORCH_JOB_ID", "")
    asg = os.environ.get("ORCH_ASG_NAME", "")
    gen = _bump_generation(cfg, job_id)
    _log(f"cold-start gen={gen} experiment={experiment} job={job_id}")
    if gen > 1:
        _cold_recovery(cfg, experiment)

    rc = 0
    try:
        _dispatch(cfg, experiment)
    except SystemExit as e:
        rc = int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)
    except Exception:  # noqa: BLE001 — record the failure, still tear down cleanly
        import traceback

        traceback.print_exc()
        rc = 1
    finally:
        if job_id:
            with contextlib.suppress(Exception):  # done-sentinel is best-effort
                aws.put_text(
                    cfg.bucket,
                    cfg.orchestrator_done_key(job_id),
                    json.dumps({"job_id": job_id, "rc": rc, "generation": gen}),
                )
        if asg:
            with contextlib.suppress(Exception):  # teardown also deletes the ASG
                aws.set_asg_capacity(asg, min_size=0, max_size=0, desired=0)
    _log(f"job {job_id} finished rc={rc}; ASG scaled to 0")
    return rc


def main() -> None:
    p = argparse.ArgumentParser(prog="orchestrator.remote")
    p.add_argument("--experiment", default=os.environ.get("EXPERIMENT", ""))
    args = p.parse_args()
    if args.experiment:
        os.environ["EXPERIMENT"] = args.experiment
    cfg = OrchestratorConfig()
    aws.set_region(cfg.region)
    sys.exit(run_on_box(cfg))


if __name__ == "__main__":
    main()
