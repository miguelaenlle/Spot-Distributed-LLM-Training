"""Make our packages and the nanoGPT submodule importable in tests.

Adds ``src/`` (so ``import spot_train`` / ``orchestrator`` work without an
editable install) and the nanoGPT checkout (so ``from model import GPT`` works;
nanoGPT has no package metadata) to ``sys.path``.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "third_party", "nanoGPT")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
