"""Export the EMA teacher backbone of a pre-training checkpoint for the downstream stages and inference.

    python -m fora_pretrain.export_teacher --checkpoint OUT/checkpoints/step_00005000 --out teacher.pt \
        [--block24 block24_state.pt]

Reads only the ``model.teacher.backbone.*`` tensors of the DCP checkpoint (no process group, CPU), checks that they load
strictly into ``fora.encoder.build_backbone('large')`` with matching RoPE periods, and writes
``{format, backbone, metadata, step}``. ``--block24`` additionally writes (block 24, final norm) of the teacher as fp32,
the initialization of the trainable block in sentence alignment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from fora.encoder.backbone import TEACHER_FORMAT, build_backbone
from fora.encoder.blocks import block24_state

PREFIX = "model.teacher.backbone."


def export(checkpoint: Path, out: Path, block24: Path | None = None) -> dict:
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    if metadata["args"].get("arch", "large") != "large":
        raise ValueError("expected a ViT-L checkpoint")
    stored = dcp.FileSystemReader(str(checkpoint)).read_metadata().state_dict_metadata
    state = {k: torch.empty(v.size, dtype=v.properties.dtype) for k, v in stored.items() if k.startswith(PREFIX)}
    if len(state) <= 300:
        raise ValueError(f"{checkpoint}: only {len(state)} teacher tensors found")
    dcp.load(state, checkpoint_id=str(checkpoint), no_dist=True)
    weights = {k[len(PREFIX):]: v for k, v in state.items()}
    backbone = build_backbone("large")
    rebuilt = backbone.rope_embed._make_periods(device=None)
    backbone.load_state_dict(weights, strict=True)
    if not torch.allclose(weights["rope_embed.periods"].float(), rebuilt.float(), rtol=1e-5):
        raise ValueError("checkpoint RoPE periods differ from the encoder geometry")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    torch.save(dict(format=TEACHER_FORMAT, backbone=weights, metadata=metadata, step=int(metadata["step"])), tmp)
    tmp.replace(out)
    info = dict(checkpoint=str(checkpoint), step=int(metadata["step"]), stage=metadata.get("stage"),
                modality=metadata["args"].get("modality", "ct"), tensors=len(weights))
    if block24 is not None:
        b24 = dict(block24_state(backbone), step=int(metadata["step"]))
        tmp = block24.with_suffix(".partial")
        torch.save(b24, tmp)
        tmp.replace(block24)
        info["block24"] = str(block24)
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--block24", type=Path, default=None)
    a = p.parse_args()
    print(json.dumps(export(a.checkpoint, a.out, a.block24)), flush=True)


if __name__ == "__main__":
    main()
