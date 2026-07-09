"""The epoch supervisor: single-writer membership for multi-node training.

Replaces torchrun's dynamic c10d rendezvous (a version-dependent black box that
recovered in 5s locally on torch 2.4 but hung >180s on the DLAMI's torch) with
explicit central orchestration. The orchestrator is the ONLY writer of the
membership document ``runs/<run_id>/epoch.json``; every box's sidecar
(``sidecar.py``) polls it and runs STATIC torchrun for the current epoch. One
writer, N readers, monotonic epochs — every recovery step is our code emitting
our logs, and nothing waits on a timeout to infer what the supervisor decided.

This module splits cleanly into:
  - :func:`decide` — a PURE reducer (Observation, Policy) -> list[Action]. All
    the membership logic, table-testable without AWS.
  - :class:`Supervisor` — the imperative shell: builds an Observation each tick
    (AWS + S3), executes the actions, and emits the profile marks the W&B
    world-size staircase / degraded phase already consume.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field

from spot_train import s3_store

from . import aws
from .config import OrchestratorConfig
from .profile import RunProfile

# --------------------------------------------------------------------------- #
# Observation (what the supervisor sees) and Policy (how it should react)
# --------------------------------------------------------------------------- #

# AWS instance states that mean a node is gone (or going), not a live member.
_DEAD_STATES = frozenset({"shutting-down", "terminated", "stopping", "stopped"})


@dataclass(frozen=True)
class NodeObs:
    """One node's health, as the supervisor can observe it from outside the box."""

    node: int
    aws_state: str  # DescribeInstances state, or "unknown"
    registered: bool  # node<i>.json present in S3 (the box announced its IP)
    terminated_by_us: bool = False  # we called TerminateInstances — dead NOW, not on lag
    log_age_s: float | None = None  # seconds since the log key last changed (heartbeat)


@dataclass(frozen=True)
class Observation:
    """Everything ``decide`` needs for one tick — a pure snapshot."""

    node_count: int  # desired full group size
    nodes: tuple[NodeObs, ...]
    epoch: int  # current published epoch (0 = none yet)
    members: frozenset[int]  # node indices in the current epoch
    metrics_exists: bool  # run's done signal landed
    no_progress_s: float | None  # seconds since the checkpoint step last advanced (None = n/a yet)
    due_kills: frozenset[int] = frozenset()  # scheduled victims whose time has arrived


@dataclass(frozen=True)
class Policy:
    """The knobs that make one supervisor a shrink experiment and another a
    preempt experiment — passed to the pure reducer so behavior is explicit."""

    replace_on_loss: bool  # relaunch a lost node (preempt) vs let the group shrink (shrink)
    recovery_timeout_s: float  # no checkpoint progress this long -> whole-group restart
    heartbeat_timeout_s: float = 90.0  # log key stale this long -> node presumed dead


def _healthy(n: NodeObs, policy: Policy) -> bool:
    """A node counts toward membership iff it announced itself, AWS still shows it
    running, we didn't just kill it, and its log heartbeat isn't stale."""
    if n.terminated_by_us or not n.registered or n.aws_state in _DEAD_STATES:
        return False
    if n.aws_state != "running":
        return False  # pending/unknown: not yet a member
    # Fresh boot has no log yet (age None) — treat as alive; only a stale, once-
    # live log means wedged-but-alive.
    return not (n.log_age_s is not None and n.log_age_s > policy.heartbeat_timeout_s)


# --------------------------------------------------------------------------- #
# Actions (what the reducer decides; the shell executes)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PublishEpoch:
    epoch: int
    members: tuple[int, ...]  # node indices, sorted; rank = position, master = members[0]


@dataclass(frozen=True)
class TerminateNode:
    node: int


@dataclass(frozen=True)
class LaunchReplacement:
    node: int


@dataclass(frozen=True)
class WholeGroupRestart:
    pass


@dataclass(frozen=True)
class Done:
    pass


Action = PublishEpoch | TerminateNode | LaunchReplacement | WholeGroupRestart | Done


# --------------------------------------------------------------------------- #
# The pure reducer
# --------------------------------------------------------------------------- #
def decide(obs: Observation, policy: Policy) -> list[Action]:
    """Observed state -> the actions that reconcile it toward "N healthy members
    training". Pure: no AWS, no clock, no I/O — every branch is table-testable.

    The heart is trivial by design (that's the point of central orchestration):
    the membership that SHOULD be published is just the currently-healthy set,
    and an epoch is (re)published whenever that set differs from what's live.
    """
    if obs.metrics_exists:
        return [Done()]

    healthy = frozenset(n.node for n in obs.nodes if _healthy(n, policy))

    # Whole-group restart floor: the group stopped making progress and can't be
    # fixed by a membership change (also the only recourse when everyone's gone).
    if obs.epoch > 0 and (
        (obs.no_progress_s is not None and obs.no_progress_s > policy.recovery_timeout_s)
        or not (healthy - obs.due_kills)
    ):
        return [WholeGroupRestart()]

    # Startup: don't begin training degraded — wait for the whole group to
    # register and reach running, then publish epoch 1.
    if obs.epoch == 0:
        if len(healthy) >= obs.node_count:
            return [PublishEpoch(1, tuple(sorted(healthy))[: obs.node_count])]
        return []

    actions: list[Action] = []
    for v in sorted(obs.due_kills):
        actions.append(TerminateNode(v))

    # Target membership excludes anyone we're killing this very tick, so the
    # shrink epoch goes out SAME tick as the kill — survivors' sidecars drop
    # their NCCL-blocked torchrun within one poll (~3s), no 20s timeout wait.
    target = healthy - obs.due_kills
    if target and frozenset(target) != obs.members:
        actions.append(PublishEpoch(obs.epoch + 1, tuple(sorted(target))))

    if policy.replace_on_loss:
        for v in sorted(obs.members - target):  # members that left (killed or died)
            actions.append(LaunchReplacement(v))

    return actions


# --------------------------------------------------------------------------- #
# The imperative shell
# --------------------------------------------------------------------------- #
def epoch_doc(
    run_id: str, epoch: int, members: tuple[int, ...], ips: dict[int, str], port_base: int
) -> dict:
    """The membership document, built from the node->ip map the boxes registered.
    rank = position in the sorted member list; master = the lowest-index member."""
    ranked = [{"node": n, "ip": ips[n], "rank": r} for r, n in enumerate(members)]
    return {
        "epoch": epoch,
        "members": ranked,
        "node_count": len(members),
        "master_addr": ips[members[0]],
        "master_port": port_base + epoch,
    }


@dataclass
class SupervisorState:
    """Cross-tick memory the pure reducer deliberately doesn't hold."""

    epoch: int = 0
    members: frozenset[int] = frozenset()
    terminated_by_us: set[int] = field(default_factory=set)  # dead NOW, ignore AWS lag
    replacing: set[int] = field(default_factory=set)  # a LaunchReplacement is in flight
    ips: dict[int, str] = field(default_factory=dict)  # node -> private IP (from registration)
    ckpt_step: int = -1
    ckpt_changed_at: float = 0.0
    shrink_baseline: int | None = None  # ckpt step at the last kill (for shrink_resume)
    full_ws: int | None = None  # world size before the last kill (for full_world)
    marks: set[str] = field(default_factory=set)  # one-shot marks already emitted per epoch cycle


class Supervisor:
    """The imperative shell around :func:`decide`: one observe->decide->act tick
    per ``log_stream_seconds``. Startup (launch N, wait for registrations) is the
    experiment driver's job; this owns the steady state — publishing epoch 1 once
    everyone's healthy, reacting to losses, streaming every node's log so the
    console is never blind, and emitting the profile marks the W&B world-size
    staircase / degraded phase consume. Effectful bits that live in the
    experiment (launching a box) are injected as callbacks to avoid an import
    cycle."""

    def __init__(
        self,
        cfg: OrchestratorConfig,
        profile: RunProfile,
        *,
        run_id: str,
        policy: Policy,
        node_ids: dict[int, str],  # node index -> instance id (mutated on replacement)
        logs: dict[int, dict],  # node -> {"key", "state"}: the live log streams
        launch_node,  # (node_index) -> instance_id (waits running, records cost)
        pull_logs,  # () -> None: pull every node's log into the profile
        kill_schedule: list[tuple[float, int]] | None = None,  # (secs after train start, victim)
    ):
        self.cfg = cfg
        self.profile = profile
        self.run_id = run_id
        self.policy = policy
        self.node_ids = node_ids
        self.logs = logs
        self._launch_node = launch_node
        self._pull_logs = pull_logs
        self.kill_schedule = list(kill_schedule or [])
        self.st = SupervisorState()
        self.ckpt_prefix = f"{cfg.run_prefix}/{run_id}/checkpoints/"
        self.metrics_key = cfg.run_metrics_key(run_id)
        self._train_start: float | None = None
        self._killed_at: dict[int, float] = {}  # victim -> monotonic time we terminated it
        self.metrics: dict | None = None

    # -- observation ------------------------------------------------------- #
    def _node_ip(self, node: int) -> str | None:
        """Private IP the box registered (node<i>.json), cached once seen."""
        if node in self.st.ips:
            return self.st.ips[node]
        # Full s3:// URI, not the bare key: s3_store.read_bytes treats a
        # prefix-less string as a LOCAL path, so a bare key silently reads as
        # "absent" and the node never counts as registered (epoch 1 never fires).
        raw = s3_store.read_bytes(self.cfg.run_node_uri(self.run_id, node))
        if raw is None:
            return None
        try:
            ip = json.loads(raw)["ip"]
        except (ValueError, KeyError):
            return None
        self.st.ips[node] = ip
        return ip

    def _observe(self, now: float, wall: float) -> Observation:
        nodes = []
        for node, iid in self.node_ids.items():
            state = aws.instance_state(iid) if iid else "unknown"
            log_lm = aws.object_last_modified(self.cfg.bucket, self.logs[node]["key"])
            nodes.append(
                NodeObs(
                    node=node,
                    aws_state=state,
                    registered=self._node_ip(node) is not None,
                    terminated_by_us=node in self.st.terminated_by_us,
                    log_age_s=(wall - log_lm) if log_lm is not None else None,
                )
            )
        # Checkpoint progress: track when the max step last advanced.
        step = aws.max_checkpoint_step(self.cfg.bucket, self.ckpt_prefix)
        if step > self.st.ckpt_step:
            self.st.ckpt_step, self.st.ckpt_changed_at = step, now
        no_progress = (now - self.st.ckpt_changed_at) if self._train_start is not None else None

        due = set()
        if self._train_start is not None:
            elapsed = now - self._train_start
            for secs, victim in self.kill_schedule:
                if elapsed >= secs and victim not in self.st.terminated_by_us:
                    due.add(victim)

        return Observation(
            node_count=self.cfg.node_count,
            nodes=tuple(nodes),
            epoch=self.st.epoch,
            members=self.st.members,
            metrics_exists=aws.object_exists(self.cfg.bucket, self.metrics_key),
            no_progress_s=no_progress,
            due_kills=frozenset(due),
        )

    # -- current world size (for full_world), read from the profile stream -- #
    def _latest_ws(self) -> int | None:
        for s in reversed(self.profile.samples):
            if s.world_size:
                return s.world_size
        return None

    # -- effects ----------------------------------------------------------- #
    def _publish_epoch(self, epoch: int, members: tuple[int, ...]) -> None:
        shrinking = self.st.members and len(members) < len(self.st.members)
        doc = epoch_doc(self.run_id, epoch, members, self.st.ips, self.cfg.rdzv_port)
        aws.put_text(self.cfg.bucket, self.cfg.run_epoch_key(self.run_id), json.dumps(doc))
        self.st.epoch, self.st.members = epoch, frozenset(members)
        print(
            f"[supervisor] published epoch {epoch}: members {sorted(members)} "
            f"(master {doc['master_addr']}:{doc['master_port']})",
            file=sys.stderr,
        )
        if shrinking:
            # The kill mark + baselines were captured in _terminate; a grow resets
            # the shrink markers so the next kill re-arms them.
            pass
        elif len(members) == self.cfg.node_count and self.st.shrink_baseline is not None:
            self.st.shrink_baseline = None  # back to full; full_world handled in _emit_marks

    def _terminate(self, node: int) -> None:
        if node in self.st.terminated_by_us:
            return
        self.st.full_ws = self._latest_ws()
        self.st.shrink_baseline = self.st.ckpt_step
        self.st.marks.discard("shrink_resume")
        self.st.marks.discard("full_world")
        aws.terminate(self.node_ids[node])
        self.st.terminated_by_us.add(node)
        self._killed_at[node] = time.monotonic()
        self.profile.instance_stopped(self.node_ids[node])
        self.profile.mark("kill")
        print(f"[supervisor] terminated node {node} ({self.node_ids[node]})", file=sys.stderr)

    def _launch_replacement(self, node: int) -> None:
        if node in self.st.replacing:
            return
        self.st.replacing.add(node)
        aws.wait_quota_released(self.node_ids[node])
        aws.wait_vcpu_headroom(self.cfg.instance_vcpu_count(), self.cfg.vcpu_quota)
        self.st.terminated_by_us.discard(node)
        self.st.ips.pop(node, None)  # force re-read of the replacement's fresh registration
        self.node_ids[node] = self._launch_node(node)
        self.profile.mark("relaunch")

    def _whole_group_restart(self) -> None:
        import sys

        print("[supervisor] whole-group restart (floor)", file=sys.stderr)
        aws.delete_object(self.cfg.bucket, self.cfg.run_epoch_key(self.run_id))
        for iid in self.node_ids.values():
            aws.terminate(iid)
            self.profile.instance_stopped(iid)
        for iid in self.node_ids.values():
            aws.wait_quota_released(iid)
        self.st = SupervisorState()
        aws.wait_vcpu_headroom(
            self.cfg.node_count * self.cfg.instance_vcpu_count(), self.cfg.vcpu_quota
        )
        for node in list(self.node_ids):
            self.node_ids[node] = self._launch_node(node)

    def _execute(self, actions: list[Action]) -> None:
        for a in actions:
            if isinstance(a, TerminateNode):
                self._terminate(a.node)
            elif isinstance(a, PublishEpoch):
                if all(n in self.st.ips or self._node_ip(n) is not None for n in a.members):
                    self._publish_epoch(a.epoch, a.members)
            elif isinstance(a, LaunchReplacement):
                self._launch_replacement(a.node)
            elif isinstance(a, WholeGroupRestart):
                self._whole_group_restart()
            elif isinstance(a, Done):
                pass

    def _emit_marks(self) -> None:
        """Observational profile marks (the ones that depend on downstream state
        crossing a threshold, not on a decision): survivors checkpointing again,
        and the world returning to full."""
        base = self.st.shrink_baseline
        if base is not None and self.st.ckpt_step > base and "shrink_resume" not in self.st.marks:
            self.profile.mark("shrink_resume")
            self.st.marks.add("shrink_resume")
        if (
            self.st.full_ws is not None
            and "full_world" not in self.st.marks
            and any(
                s.world_size == self.st.full_ws and s.step > (base or -1)
                for s in self.profile.samples
            )
        ):
            self.profile.mark("full_world")
            self.st.marks.add("full_world")

    # -- loop -------------------------------------------------------------- #
    def run(self, *, deadline_s: float) -> dict | None:
        """Drive the run to completion (metrics.json) or the deadline. Returns the
        parsed metrics, or None on timeout."""
        import sys

        end = time.monotonic() + deadline_s
        while time.monotonic() < end:
            self._pull_logs()
            now, wall = time.monotonic(), time.time()
            obs = self._observe(now, wall)
            # First checkpoint => training has started; arm the kill schedule clock.
            if self._train_start is None and self.st.ckpt_step >= 0 and self.st.epoch > 0:
                self._train_start = now
                self.profile.mark("train_start")
            actions = decide(obs, self.policy)
            if any(isinstance(a, Done) for a in actions):
                self._pull_logs()
                self.metrics = json.loads(aws.get_text(self.cfg.bucket, self.metrics_key))
                return self.metrics
            self._execute(actions)
            self._emit_marks()
            time.sleep(self.cfg.log_stream_seconds)
        print("[supervisor] deadline reached without metrics.json", file=sys.stderr)
        return None
