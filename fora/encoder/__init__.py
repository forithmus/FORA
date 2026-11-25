"""The FORA image encoder: 3-D DINOv3 ViT-L with RoPE in millimetres, teacher loading, the block-23 / block-24 split,
the block-23 cache reader and X-ray tiling.

Modules
  model      DinoVisionTransformer3D, 3-D patch embedding, physical 3-D RoPE
  backbone   build_backbone, load_teacher, geometry constants (CT and X-ray)
  blocks     block23_tokens, final_block, Block24 (cached tokens), build_block24, block24_state, crop_layout
  positions  Fourier features of the token position in the scan / film
  cache      reader of the int8 block-23 cache (fora_int8_tokens)
  cxr        X-ray film tiling with token ownership, CXRSplitEncoder
"""
from .backbone import (CT_VOXEL_SPACING_MM, CXR_ROPE_MIN_PERIOD_MM, CXR_VOXEL_SPACING_MM, PATCH_SIZE, PREFIX_TOKENS,
                       ROPE_MIN_PERIOD_MM, build_backbone, load_teacher, read_teacher)
from .blocks import Block24, block23_tokens, block24_state, build_block24, crop_layout, final_block, split_top

__all__ = [
    "CT_VOXEL_SPACING_MM", "CXR_ROPE_MIN_PERIOD_MM", "CXR_VOXEL_SPACING_MM", "PATCH_SIZE", "PREFIX_TOKENS",
    "ROPE_MIN_PERIOD_MM", "build_backbone", "load_teacher", "read_teacher",
    "Block24", "block23_tokens", "block24_state", "build_block24", "crop_layout", "final_block", "split_top",
]
