"""The test that gates all cloud work.

Success criterion for Phase 1a: a killed-and-resumed run reaches the SAME loss
as an uninterrupted one. We prove this on CPU first, where it's deterministic
and debuggable, before ever touching a spot instance.

Structure (to be filled in as the loop lands):

    1. Run uninterrupted for N steps; record loss at step N.
    2. Fresh run: train N/2 steps, checkpoint, throw away the process.
    3. Resume from the checkpoint; train to N.
    4. Assert the resumed loss == the uninterrupted loss (bitwise / within a
       tight tolerance). Divergence means an RNG source or the data-loader
       position wasn't captured.

These are marked xfail until the loop + nanoGPT wiring exist, so CI stays green
during scaffolding but the intent is committed.
"""

import pytest

from spot_train import rng


def test_rng_roundtrip():
    """RNG capture/restore is exact — the cheapest half of determinism."""
    import random

    rng_state = rng.capture()
    a = [random.random() for _ in range(5)]
    rng.restore(rng_state)
    b = [random.random() for _ in range(5)]
    assert a == b


@pytest.mark.xfail(reason="Phase 1a: train loop + nanoGPT wiring not implemented yet")
def test_kill_and_resume_matches_uninterrupted():
    from spot_train.config import TrainConfig
    from spot_train.train import train

    cfg = TrainConfig(max_steps=20, checkpoint_interval=10, device="cpu")

    # 1. uninterrupted reference run.
    train(cfg)
    # TODO: 2. kill at step 10, 3. resume to 20, 4. assert losses match.
    #       Divergence => an RNG source or the loader position wasn't captured.
    raise NotImplementedError
