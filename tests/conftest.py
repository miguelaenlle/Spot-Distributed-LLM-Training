"""Make the nanoGPT submodule importable in tests (``from model import GPT``).

The package `spot_train` itself is importable via `pip install -e .` (src
layout). nanoGPT is a plain checkout with no package metadata, so we add its
directory to sys.path here rather than installing it.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NANOGPT = os.path.join(_ROOT, "third_party", "nanoGPT")
if _NANOGPT not in sys.path:
    sys.path.insert(0, _NANOGPT)
