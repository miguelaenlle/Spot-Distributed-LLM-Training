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

socket.getfqdn = lambda *a, **kw: "127.0.0.1"

from torch.distributed.run import main  # noqa: E402 — patch must precede import

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
