"""Spot-interruption listener.

Three layers of defense, because spot kills are unreliable:

1. IMDS poll  — background thread hitting the instance metadata spot-action
   endpoint every few seconds. AWS usually gives ~2 minutes notice here.
2. SIGTERM    — backup signal handler, in case the notice is delivered as a
   termination signal instead.
3. Periodic checkpointing (in the train loop, not here) — because some kills
   give no warning at all.

On any signal we set a flag; the train loop checks it, writes a final
checkpoint, and exits cleanly. We never checkpoint from inside the signal
handler / poller thread itself — that work happens on the main loop where the
model and optimizer live.

On the CPU test box IMDS is absent, so the poller no-ops and only SIGTERM is
active. Same code path on spot, where IMDS responds.
"""

from __future__ import annotations

import signal
import threading

# 169.254.169.254 is the EC2 link-local IMDS address.
IMDS_TOKEN_URL = "http://169.254.169.254/latest/api/token"
IMDS_SPOT_ACTION_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"
POLL_SECONDS = 5


class InterruptionListener:
    """Sets ``.should_stop`` when a preemption signal is observed."""

    def __init__(self, poll_seconds: int = POLL_SECONDS):
        self._stop = threading.Event()  # tells the poller thread to exit
        self.should_stop = threading.Event()  # tells the train loop to wind down
        self._poll_seconds = poll_seconds
        self._thread: threading.Thread | None = None

    def start(self) -> InterruptionListener:
        signal.signal(signal.SIGTERM, self._on_sigterm)
        self._thread = threading.Thread(target=self._poll_imds, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    # -- signal path -------------------------------------------------------- #
    def _on_sigterm(self, signum, frame) -> None:
        self.should_stop.set()

    # -- IMDS path ---------------------------------------------------------- #
    def _poll_imds(self) -> None:
        """Poll the spot-action endpoint until told to stop.

        Scaffold: real impl fetches an IMDSv2 token, then GETs the
        spot/instance-action endpoint. A 200 (with action=terminate/stop)
        means preemption is imminent -> set should_stop. A 404 is the normal
        "no action pending" response. Any connection error (no IMDS, e.g. the
        CPU test box) is treated as "not on spot" and the poller quietly idles.
        """
        while not self._stop.wait(self._poll_seconds):
            action = self._check_spot_action()
            if action is not None:
                self.should_stop.set()
                return

    def _check_spot_action(self) -> str | None:
        # PARKED for Phase 1a: the MVP uses controlled kills, not real preemption.
        # The real impl fetches an IMDSv2 token and GETs the spot/instance-action
        # endpoint (a 200 => preemption imminent). Until then we no-op so the
        # poller "quietly idles" as the docstring promises, instead of raising and
        # crashing the daemon thread on every poll (noisy traceback in the log).
        return None
