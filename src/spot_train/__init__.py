"""spot_train — a fault-tolerance layer for training LLMs on spot instances.

We do not own the model (see ``third_party/nanoGPT``). We own everything that
lets a training run be killed at any moment and resume to the *same loss*:
full-state checkpointing, an S3 store with atomic writes, RNG + data-loader
state capture, and a spot-interruption listener.
"""

__version__ = "0.0.1"
