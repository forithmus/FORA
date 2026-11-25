"""Split the encoder at block 23 (Methods, "Sentence alignment" and "Finding classifier").

Downstream stages keep blocks 1-23 frozen and train block 24 + the final norm:

* ``block23_tokens`` runs the patch embedding and blocks 1-23 of a backbone (prefix + patch tokens, before any norm);
* ``final_block`` applies a (block 24, norm) pair to such tokens with the full-crop RoPE; together they reproduce
  ``backbone.forward_features(x)['x_norm_patchtokens']`` exactly (tested);
* ``Block24`` implements block 24 + norm for the CT block-23 *cache*, where each 64 x 192 x 192 crop keeps only its
  body tokens: crops are batched with a key-padding mask and every token keeps the RoPE of its position in the
  32 x 12 x 12 crop grid;
* ``block24_state`` / ``build_block24`` move (block 24, norm) between a backbone and a plain state file.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .backbone import CT_VOXEL_SPACING_MM, PREFIX_TOKENS, ROPE_MIN_PERIOD_MM, build_backbone

PREFIX = PREFIX_TOKENS                 # CLS + 4 registers
CROP_GRID = (32, 12, 12)               # token grid of one 64 x 192 x 192 voxel crop = 4,608 tokens
LAST = 23                              # index of block 24


def block23_tokens(backbone, x: torch.Tensor, rope=None):
    """Blocks 1-23 of ``backbone`` on images x (B, 1, D, H, W) -> (tokens [B, 5 + N, C], rope for the grid).

    Pass ``rope`` to reuse the (sin, cos) of an identical grid. Autocast / no_grad are the caller's choice."""
    tokens, grid = backbone.prepare_tokens_with_masks(x)
    if rope is None:
        rope = backbone.rope_for(grid, tokens)
    for block in backbone.blocks[:LAST]:
        tokens = block(tokens, rope)
    return tokens, rope


def final_block(block: nn.Module, norm: nn.Module, h23: torch.Tensor, rope) -> torch.Tensor:
    """(block 24, norm) on block-23 tokens -> normalized patch tokens [B, N, C] (prefix removed)."""
    return norm(block(h23, rope))[:, PREFIX:]


def block24_state(backbone) -> dict:
    """{'block': blocks[23] state, 'norm': norm state} as fp32 CPU tensors (FSDP DTensors are gathered)."""
    full = lambda t: (t.full_tensor() if hasattr(t, "full_tensor") else t).detach().float().cpu().clone()
    return dict(block={k: full(v) for k, v in backbone.blocks[LAST].state_dict().items()},
                norm={k: full(v) for k, v in backbone.norm.state_dict().items()})


def split_top(backbone, state: dict | None = None) -> nn.ModuleDict:
    """Detach (block 24, norm) of a backbone into ``ModuleDict(block=..., norm=...)``; optionally load ``state``."""
    top = nn.ModuleDict(dict(block=copy.deepcopy(backbone.blocks[LAST]), norm=copy.deepcopy(backbone.norm)))
    if state is not None:
        top.block.load_state_dict(state["block"])
        top.norm.load_state_dict(state["norm"])
    return top


class Block24(nn.Module):
    """Block 24 + final norm over cached block-23 tokens, batched over crops with a key-padding mask.

    Input x [B, 5 + L, C] (prefix first), positions [B, L] local token ids in the 32 x 12 x 12 crop grid, valid [B, 5 + L].
    The prefix tokens carry no RoPE, as in DINOv3. The state dict keys are ``block.*`` and ``norm.*``."""

    def __init__(self, block, norm, rope):
        super().__init__()
        self.block = block
        self.norm = norm
        self.heads = block.attn.num_heads
        rope.eval()
        with torch.no_grad():
            rope._init_weights()
            sin, cos = rope(D=CROP_GRID[0], H=CROP_GRID[1], W=CROP_GRID[2])
        self.register_buffer("sin", sin.float(), persistent=False)
        self.register_buffer("cos", cos.float(), persistent=False)

    @staticmethod
    def _rotate(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, positions, valid):
        blk = self.block
        B, L, C = x.shape
        h = blk.norm1(x)
        qkv = blk.attn.qkv(h).reshape(B, L, 3, self.heads, C // self.heads)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
        sin = self.sin[positions].to(q.dtype)[:, None]
        cos = self.cos[positions].to(q.dtype)[:, None]
        q = torch.cat((q[:, :, :PREFIX], q[:, :, PREFIX:] * cos + self._rotate(q[:, :, PREFIX:]) * sin), 2)
        k = torch.cat((k[:, :, :PREFIX], k[:, :, PREFIX:] * cos + self._rotate(k[:, :, PREFIX:]) * sin), 2)
        mask = None if bool(valid.all()) else valid[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        a = blk.attn.proj(a.transpose(1, 2).reshape(B, L, C))
        x = x + blk.ls1(a)
        x = x + blk.ls2(blk.mlp(blk.norm2(x)))
        return self.norm(x)


def build_block24(state: dict | None = None, *, voxel_spacing_mm=CT_VOXEL_SPACING_MM,
                  rope_min_period_mm: float = ROPE_MIN_PERIOD_MM) -> Block24:
    """Plain (non-FSDP) Block24 with the training-time configuration, optionally loading a ``block24_state`` dict."""
    backbone = build_backbone("large", voxel_spacing_mm=voxel_spacing_mm, rope_min_period_mm=rope_min_period_mm)
    block, norm, rope = backbone.blocks[LAST], backbone.norm, backbone.rope_embed
    if state is not None:
        block.load_state_dict(state["block"])
        norm.load_state_dict(state["norm"])
    del backbone
    return Block24(block, norm, rope)


def crop_layout(flat: np.ndarray, grid, extent=CROP_GRID):
    """Crop-major order of the body tokens of a view.

    flat: body-token indices in the (gz, gy, gx) grid of the whole view. Returns (order, crop ids, crop starts into the
    ordered tokens with a final end entry, local position of each ordered token in its crop, number of crops)."""
    gz, gy, gx = grid
    z, rem = np.divmod(flat, gy * gx)
    y, x = np.divmod(rem, gx)
    cz, cy, cx = (-(-g // e) for g, e in zip(grid, extent))
    crop = ((z // extent[0]) * cy + (y // extent[1])) * cx + (x // extent[2])
    local = ((z % extent[0]) * extent[1] + (y % extent[1])) * extent[2] + (x % extent[2])
    order = np.argsort(crop, kind="stable")
    ids, starts = np.unique(crop[order], return_index=True)
    return order, ids, np.append(starts, len(order)), local[order], cz * cy * cx
