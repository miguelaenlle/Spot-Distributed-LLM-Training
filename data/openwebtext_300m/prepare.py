"""Prepare a CAPPED OpenWebText slice (~300M tokens) as nanoGPT-style bins.

A bounded corpus for the scaling series: big enough that GPT-2-small won't
overfit within a 15-30 min run (a run processes ~50-100M tokens, so <1 pass),
but small enough (~600MB bin) to download to each spot node in seconds — the
full 17GB OpenWebText would add minutes to every boot AND every preemption
replacement (see docs/gpt2-reproduction.md for the streaming data plane that
lifts that limit for the eventual full run).

Streams OpenWebText from HuggingFace, tokenizes with the GPT-2 BPE (tiktoken,
uint16), and writes ``train.bin`` / ``val.bin`` — no ``meta.pkl`` (BPE vocab is
fixed, so the trainer uses vocab 50304). Run once, locally:

    pip install datasets tiktoken tqdm numpy
    python data/openwebtext_300m/prepare.py

Then ``DATASET=openwebtext_300m spot-orchestrate stage-data`` uploads the bins.
Deterministic (fixed cap + split) so every stage produces identical bytes.
"""

from __future__ import annotations

import os

import numpy as np

TARGET_TOKENS = int(os.environ.get("OWT_TARGET_TOKENS", 300_000_000))
VAL_FRACTION = 0.0005  # ~150k val tokens — enough for a stable eval, tiny to ship
_HERE = os.path.dirname(os.path.abspath(__file__))


def _write_bin(path: str, tokens: np.ndarray) -> None:
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(len(tokens),))
    arr[:] = tokens
    arr.flush()


def main() -> None:
    import tiktoken
    from datasets import load_dataset
    from tqdm import tqdm

    train_path = os.path.join(_HERE, "train.bin")
    val_path = os.path.join(_HERE, "val.bin")
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"[prepare] bins already present in {_HERE} — nothing to do")
        return

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # separate documents with the end-of-text token
    # Stream so we never materialize the full 40GB dataset; stop at the cap.
    stream = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    buf: list[int] = []
    pbar = tqdm(total=TARGET_TOKENS, unit="tok", desc="tokenizing OWT (capped)")
    for row in stream:
        ids = enc.encode_ordinary(row["text"])
        ids.append(eot)
        buf.extend(ids)
        pbar.update(len(ids))
        if len(buf) >= TARGET_TOKENS:
            break
    pbar.close()

    toks = np.array(buf[:TARGET_TOKENS], dtype=np.uint16)
    n_val = max(1, int(len(toks) * VAL_FRACTION))
    _write_bin(val_path, toks[:n_val])
    _write_bin(train_path, toks[n_val:])
    print(
        f"[prepare] wrote {len(toks) - n_val:,} train + {n_val:,} val tokens "
        f"(uint16) to {_HERE}"
    )


if __name__ == "__main__":
    main()
