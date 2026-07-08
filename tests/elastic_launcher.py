"""torchrun wrapper for the local elastic E2E test.

torch 2.4's ``RendezvousStoreInfo.build`` derives the worker-group store address
from ``socket.getfqdn()`` (ignores ``--local-addr``). On a localhost test that
must be 127.0.0.1 — and on macOS getfqdn() returns an unresolvable
``...ip6.arpa`` reverse-DNS name outright — so pin it before torchrun starts.
On the EC2 boxes getfqdn() is the in-VPC-resolvable internal hostname and no
shim is needed; this wrapper exists only for tests.
"""

from __future__ import annotations

import socket
import sys
from datetime import timedelta

socket.getfqdn = lambda *a, **kw: "127.0.0.1"

# Speed shim #2: torch 2.4 hardcodes the rendezvous keep-alive at 5s x 3
# attempts (dynamic_rendezvous.py, from_backend) and does not expose it via
# --rdzv_conf — so after a node dies, the survivor's next round stalls ~15s
# waiting for the ghost to expire. 1s x 2 keeps the local test snappy without
# touching the code under test. Production accepts the 15s: it overlaps the
# 20s NCCL_TIMEOUT crash window, so it costs little on a real box.
import torch.distributed.elastic.rendezvous.dynamic_rendezvous as _dr  # noqa: E402

_orig_settings = _dr.RendezvousSettings


def _fast_settings(*args, **kwargs):
    kwargs["keep_alive_interval"] = timedelta(seconds=1)
    kwargs["keep_alive_max_attempt"] = 2
    return _orig_settings(*args, **kwargs)


_dr.RendezvousSettings = _fast_settings

from torch.distributed.run import main  # noqa: E402 — patches must precede import

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
