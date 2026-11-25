"""Export the aligned block 24 + norm of an alignment checkpoint: ``{block: state dict, step, epoch}``.

The classifier initializes its trainable block 24 from this file (its ``block`` entry; keys ``block.*`` and ``norm.*``).

    python -m fora_align.export_block24 OUT/epoch_07.pt aligned_block24.pt
"""
from __future__ import annotations

import argparse
import json

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    a = ap.parse_args()
    s = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    torch.save(dict(block=s["block"], step=s["step"], epoch=s["epoch"]), a.out)
    print(json.dumps(dict(source=a.checkpoint, step=s["step"], epoch=s["epoch"], tensors=len(s["block"]))), flush=True)


if __name__ == "__main__":
    main()
