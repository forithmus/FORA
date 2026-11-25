"""3-D DINOv3 vision transformer with a native 3-D patch embedding and RoPE in millimetres (Methods, "Image encoder").

The transformer blocks are the unmodified DINOv3 ``SelfAttentionBlock`` (facebookresearch/dinov3, commit 6876159). Only the
image-specific parts are our own:

* ``PatchEmbed3D``: a Conv3d with kernel = stride = patch size (2, 16, 16 voxels = 3 x 8 x 8 mm on the CT grid);
* ``PhysicalRopePositionEmbedding3D``: axial 3-D rotary embedding whose token coordinates are measured in millimetres, so a
  crop of any size at the same voxel spacing gets the same relative geometry;
* 1 CLS token + 4 register ("storage") tokens, an untied layer norm for the CLS token of local crops during training.

An X-ray is fed as a two-slice volume (2, H, W), which the (2, 16, 16) patch turns into a single token layer.
"""
from __future__ import annotations

import math
from functools import partial
from typing import Iterable

import torch
from torch import Tensor, nn

from dinov3.layers import LayerScale, Mlp, SelfAttentionBlock
from dinov3.utils import named_apply


def _triple(x: int | Iterable[int]) -> tuple[int, int, int]:
    if isinstance(x, int):
        return (x, x, x)
    out = tuple(int(v) for v in x)
    if len(out) != 3:
        raise ValueError(f"Expected a 3-tuple, got {out}")
    return out


class PatchEmbed3D(nn.Module):
    """(B, C, D, H, W) -> (B, D', H', W', embed_dim)."""

    def __init__(self, volume_size=(64, 384, 384), patch_size=(2, 16, 16), in_chans: int = 1, embed_dim: int = 768) -> None:
        super().__init__()
        self.volume_size = _triple(volume_size)
        self.patch_size = _triple(patch_size)
        self.patches_resolution = tuple(v // p for v, p in zip(self.volume_size, self.patch_size))
        self.num_patches = math.prod(self.patches_resolution)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"PatchEmbed3D expects B,C,D,H,W, got {tuple(x.shape)}")
        return self.proj(x).permute(0, 2, 3, 4, 1).contiguous()

    def reset_parameters(self) -> None:
        bound = math.sqrt(1.0 / (self.in_chans * math.prod(self.patch_size)))
        nn.init.uniform_(self.proj.weight, -bound, bound)
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -bound, bound)


class PhysicalRopePositionEmbedding3D(nn.Module):
    """Axial 3-D RoPE with coordinates in millimetres.

    A 64-dimensional head has 32 rotary pairs, allocated as 11 (z), 11 (y) and 10 (x). Periods are log-spaced between
    ``min_period_mm`` and ``max_period_mm`` per axis. Token centres sit at (i + 0.5 - n / 2) * patch * spacing, i.e. the
    crop centre is the origin. The output layout matches DINOv3's ``rope_rotate_half``.
    """

    def __init__(self, embed_dim: int, num_heads: int, patch_size=(2, 16, 16), voxel_spacing_mm=(1.5, 0.5, 0.5),
                 min_period_mm: float = 6.0, max_period_mm: float = 1200.0, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.head_dim = embed_dim // num_heads
        if self.head_dim % 2:
            raise ValueError("3-D RoPE requires an even attention head dimension")
        base, rem = divmod(self.head_dim // 2, 3)
        self.axis_pairs = (base + (rem > 0), base + (rem > 1), base)
        self.patch_size = _triple(patch_size)
        self.voxel_spacing_mm = tuple(float(v) for v in voxel_spacing_mm)
        self.min_period_mm = float(min_period_mm)
        self.max_period_mm = float(max_period_mm)
        self.dtype = dtype
        self.register_buffer("periods", self._make_periods(device=None), persistent=True)

    def _make_periods(self, device) -> Tensor:
        periods = []
        for n in self.axis_pairs:
            if n == 1:
                periods.append(torch.tensor([self.min_period_mm], dtype=self.dtype, device=device))
            else:
                periods.append(torch.logspace(math.log10(self.min_period_mm), math.log10(self.max_period_mm), n,
                                              dtype=self.dtype, device=device))
        return torch.cat(periods)

    def _axis_coords(self, n: int, axis: int, device) -> Tensor:
        step = self.patch_size[axis] * self.voxel_spacing_mm[axis]
        return (torch.arange(n, device=device, dtype=self.dtype) + 0.5 - n / 2.0) * step

    def forward(self, *, D: int, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        coords = torch.stack(torch.meshgrid(self._axis_coords(D, 0, device), self._axis_coords(H, 1, device),
                                            self._axis_coords(W, 2, device), indexing="ij"), dim=-1).flatten(0, 2)
        angles, start = [], 0
        for axis, n_freq in enumerate(self.axis_pairs):
            p = self.periods[start:start + n_freq]
            angles.append(2.0 * math.pi * coords[:, axis, None] / p[None, :])
            start += n_freq
        half = torch.cat(angles, dim=-1)
        full = torch.cat((half, half), dim=-1)
        return torch.sin(full), torch.cos(full)

    def _init_weights(self) -> None:
        # FSDP2 builds the model on the meta device; recreate the deterministic periods after materialization.
        self.periods.copy_(self._make_periods(self.periods.device))


def _init_weights(module: nn.Module, name: str = "") -> None:
    del name
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, LayerScale, PatchEmbed3D)):
        module.reset_parameters()


class DinoVisionTransformer3D(nn.Module):
    """Volumetric counterpart of DINOv3's ``DinoVisionTransformer`` (LayerNorm eps 1e-6, MLP FFN, QKV bias)."""

    def __init__(self, *, volume_size=(64, 384, 384), patch_size=(2, 16, 16), voxel_spacing_mm=(1.5, 0.5, 0.5),
                 in_chans: int = 1, embed_dim: int = 768, depth: int = 12, num_heads: int = 12, ffn_ratio: float = 4.0,
                 drop_path_rate: float = 0.0, layerscale_init: float | None = 1e-5, n_storage_tokens: int = 4,
                 untie_global_and_local_cls_norm: bool = True, rope_min_period_mm: float = 6.0,
                 rope_max_period_mm: float = 1200.0, device=None) -> None:
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = _triple(patch_size)
        self.voxel_spacing_mm = tuple(float(v) for v in voxel_spacing_mm)
        norm_cls = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed3D(volume_size, self.patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))
        self.n_storage_tokens = int(n_storage_tokens)
        if self.n_storage_tokens:
            self.storage_tokens = nn.Parameter(torch.empty(1, self.n_storage_tokens, embed_dim, device=device))
        self.rope_embed = PhysicalRopePositionEmbedding3D(embed_dim, num_heads, patch_size=self.patch_size,
                                                          voxel_spacing_mm=self.voxel_spacing_mm,
                                                          min_period_mm=rope_min_period_mm, max_period_mm=rope_max_period_mm)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(dim=embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio, qkv_bias=True, proj_bias=True,
                               ffn_bias=True, drop_path=drop_path_rate, norm_layer=norm_cls, act_layer=nn.GELU,
                               ffn_layer=Mlp, init_values=layerscale_init, mask_k_bias=False, device=device)
            for _ in range(depth)])
        self.norm = norm_cls(embed_dim)
        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        self.local_cls_norm = norm_cls(embed_dim) if untie_global_and_local_cls_norm else None
        self.head = nn.Identity()

    @property
    def prefix_tokens(self) -> int:
        return 1 + self.n_storage_tokens

    def init_weights(self) -> None:
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.zeros_(self.mask_token)
        if self.n_storage_tokens:
            nn.init.normal_(self.storage_tokens, std=0.02)
        named_apply(_init_weights, self)

    def prepare_tokens_with_masks(self, x: Tensor, masks: Tensor | None = None):
        x = self.patch_embed(x)
        B, D, H, W, _ = x.shape
        x = x.flatten(1, 3)
        if masks is not None:
            if masks.shape != x.shape[:2]:
                raise ValueError(f"Mask {tuple(masks.shape)} does not match patch grid {tuple(x.shape[:2])}")
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls = self.cls_token
        else:
            cls = self.cls_token + 0 * self.mask_token
        storage = (self.storage_tokens if self.n_storage_tokens
                   else torch.empty(1, 0, self.embed_dim, device=x.device, dtype=x.dtype))
        return torch.cat((cls.expand(B, -1, -1), storage.expand(B, -1, -1), x), dim=1), (D, H, W)

    def rope_for(self, grid, like: Tensor) -> tuple[Tensor, Tensor]:
        """RoPE (sin, cos) for a token grid, cast to the dtype and device of ``like``."""
        d, h, w = grid
        return tuple(c.to(device=like.device, dtype=like.dtype) for c in self.rope_embed(D=d, H=h, W=w))

    def forward_features_list(self, x_list: list[Tensor], masks_list: list[Tensor | None]):
        xs, grids = [], []
        for image, masks in zip(x_list, masks_list):
            tokens, grid = self.prepare_tokens_with_masks(image, masks)
            xs.append(tokens)
            grids.append(grid)
        ropes = [self.rope_for(g, x) for x, g in zip(xs, grids)]
        for block in self.blocks:
            xs = block(xs, ropes)
        outputs = []
        prefix = self.prefix_tokens
        for idx, (x, masks) in enumerate(zip(xs, masks_list)):
            if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                cls_reg = self.local_cls_norm(x[:, :prefix])
                patch = self.norm(x[:, prefix:])
            else:
                y = self.norm(x)
                cls_reg, patch = y[:, :prefix], y[:, prefix:]
            outputs.append({"x_norm_clstoken": cls_reg[:, 0], "x_storage_tokens": cls_reg[:, 1:],
                            "x_norm_patchtokens": patch, "x_prenorm": x, "masks": masks, "patch_grid": grids[idx]})
        return outputs

    def forward_features(self, x, masks=None):
        if isinstance(x, Tensor):
            return self.forward_features_list([x], [masks])[0]
        if masks is None:
            masks = [None] * len(x)
        return self.forward_features_list(x, masks)

    def forward(self, x, masks=None, is_training: bool = False):
        out = self.forward_features(x, masks)
        return out if is_training else self.head(out["x_norm_clstoken"])


def vit_large_3d(**kwargs) -> DinoVisionTransformer3D:
    """ViT-L: width 1024, 24 blocks, 16 heads, MLP ratio 4 (302,894,080 backbone parameters with the default patch)."""
    return DinoVisionTransformer3D(embed_dim=1024, depth=24, num_heads=16, ffn_ratio=4, **kwargs)
