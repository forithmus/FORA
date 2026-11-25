"""Build the FORA image encoder (the same network for pre-training, downstream stages and inference) and load an exported teacher.

Configuration (Methods, "Image encoder"): DINOv3 ViT-L blocks, 4 register tokens, patch 2 x 16 x 16 voxels, drop path 0.3,
LayerScale 1e-5, 3-D RoPE in millimetres on the CT grid of 1.5 x 0.5 x 0.5 mm with periods from 6 mm to 1,200 mm.

The X-ray encoder uses the same geometry: its RoPE treats one pixel as 0.5 mm (and the two slices as 1.5 mm) with
periods from 6 mm, although films are used at native resolution (about 0.15 mm per pixel).
``CXR_VOXEL_SPACING_MM`` and ``CXR_ROPE_MIN_PERIOD_MM`` therefore equal the CT values.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .model import DinoVisionTransformer3D, vit_large_3d

PATCH_SIZE = (2, 16, 16)
CT_VOXEL_SPACING_MM = (1.5, 0.5, 0.5)
ROPE_MIN_PERIOD_MM = 6.0
ROPE_MAX_PERIOD_MM = 1200.0
VOLUME_SIZE = (64, 384, 384)      # nominal size of the patch-embedding module; any multiple of the patch works
N_STORAGE_TOKENS = 4
PREFIX_TOKENS = 1 + N_STORAGE_TOKENS
EMBED_DIM = 1024
DEPTH = 24
DROP_PATH_RATE = 0.3
LAYERSCALE_INIT = 1e-5

# X-ray encoder geometry (equal to the CT values; see the module docstring).
CXR_VOXEL_SPACING_MM = CT_VOXEL_SPACING_MM
CXR_ROPE_MIN_PERIOD_MM = ROPE_MIN_PERIOD_MM

TEACHER_FORMAT = "fora_teacher"


def build_backbone(arch: str = "large", *, patch_size=PATCH_SIZE, voxel_spacing_mm=CT_VOXEL_SPACING_MM,
                   rope_min_period_mm: float = ROPE_MIN_PERIOD_MM, drop_path_rate: float = DROP_PATH_RATE,
                   device=None) -> DinoVisionTransformer3D:
    """The 3-D ViT-L of the paper. Parameters are uninitialized until ``init_weights()`` or a state dict is loaded."""
    if arch != "large":
        raise ValueError(f"Only the ViT-L ('large') encoder is part of FORA, got {arch!r}")
    return vit_large_3d(volume_size=VOLUME_SIZE, patch_size=tuple(patch_size), voxel_spacing_mm=tuple(voxel_spacing_mm),
                        rope_min_period_mm=float(rope_min_period_mm), rope_max_period_mm=ROPE_MAX_PERIOD_MM,
                        in_chans=1, n_storage_tokens=N_STORAGE_TOKENS, drop_path_rate=drop_path_rate,
                        layerscale_init=LAYERSCALE_INIT, device=device)


def read_teacher(path: str | Path) -> dict:
    """Exported teacher file -> {'backbone': state dict, 'metadata': dict, 'step': int}.

    The state dict is stored under ``backbone`` (``encoder-pretraining/fora_pretrain/export_teacher.py``); files that
    store it under ``state`` are read as well."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob.get("backbone", blob.get("state"))
    if state is None:
        raise ValueError(f"{path}: no 'backbone' or 'state' entry; not an exported FORA teacher")
    return dict(backbone=state, metadata=blob.get("metadata", {}), step=int(blob.get("step", -1)))


def load_teacher(path: str | Path, device="cpu", *, voxel_spacing_mm=CT_VOXEL_SPACING_MM,
                 rope_min_period_mm: float = ROPE_MIN_PERIOD_MM) -> DinoVisionTransformer3D:
    """Build the ViT-L, load an exported EMA-teacher backbone (strict) and return it in eval mode without gradients.

    The stored RoPE periods must equal the ones rebuilt from ``voxel_spacing_mm`` / ``rope_min_period_mm``; a mismatch
    means the encoder geometry differs from the one the checkpoint was trained with."""
    blob = read_teacher(path)
    backbone = build_backbone("large", voxel_spacing_mm=voxel_spacing_mm, rope_min_period_mm=rope_min_period_mm)
    rebuilt = backbone.rope_embed._make_periods(device=None)
    backbone.load_state_dict(blob["backbone"], strict=True)
    if not torch.allclose(backbone.rope_embed.periods.float(), rebuilt.float(), rtol=1e-5):
        raise ValueError("Teacher RoPE periods differ from the requested encoder geometry")
    backbone.teacher_step = blob["step"]
    return backbone.to(device).eval().requires_grad_(False)
