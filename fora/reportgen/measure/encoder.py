"""Frozen 3-D ViT-L of the classifier (blocks 1-23 of the CT encoder + the classifier's fine-tuned block 24 / final norm, token
normalisation (y - mean) / rms) run on lesion-centred, upsampled crops for the lesion calipers (paper: Methods, "Measurements").

A crop view is a DINO input grid of IN_VOX = (32, 384, 384) voxels = (16, 24, 24) tokens of patch (2, 16, 16); token size tok_mm
(isotropic) = FOV / 24, voxel spacing = (tok/2, tok/16, tok/16) mm.  The physical RoPE uses the true voxel spacing of the sampled grid
(`rope_embed.voxel_spacing_mm` set per call).
"""
import numpy as np
import torch

TOK = (16, 24, 24); PATCH = (2, 16, 16); IN_VOX = tuple(t * p for t, p in zip(TOK, PATCH)); CACHE_SP = np.array([1.5, 0.5, 0.5])


def load_encoder(device, encoder_ckpt, mil_ckpt, normalization, mil_block=True):
    """-> (backbone, token mean [1024], token rms).  encoder_ckpt: teacher backbone state (`backbone`); mil_ckpt: classifier checkpoint
    whose `block` entry holds `block.*` (block 24) and `norm.*` (final norm); normalization: {'mean', 'rms'} of the block-24 tokens."""
    from fora.encoder.backbone import build_backbone
    bb = build_backbone('large'); st = torch.load(encoder_ckpt, map_location='cpu', weights_only=False)['backbone']
    bb.load_state_dict(st, strict=True)
    if mil_block:
        ck = torch.load(mil_ckpt, map_location='cpu', weights_only=False)['block']
        bb.blocks[23].load_state_dict({k[6:]: v for k, v in ck.items() if k.startswith('block.')})
        bb.norm.load_state_dict({k[5:]: v for k, v in ck.items() if k.startswith('norm.')})
    bb.eval().requires_grad_(False).to(device)
    n = torch.load(normalization, map_location='cpu', weights_only=False)
    return bb, n['mean'].float().to(device), float(n['rms'])


def window(x_hu, lo, hi):   # == the encoder's brick window normalisation (before the bf16 cast)
    return ((x_hu.clamp(lo, hi) - lo) / (hi - lo)) * 2. - 1.


def encode(bb, mean, rms, img, spacing_mm):
    """img [B,1,D,H,W] normalised; spacing_mm (z, y, x) voxel spacing for the physical RoPE. -> [B, N, 1024] normalised block-24 tokens."""
    bb.rope_embed.voxel_spacing_mm = tuple(float(s) for s in spacing_mm)
    dev = img.device
    ctx = torch.autocast('cuda', dtype=torch.bfloat16) if dev.type == 'cuda' else torch.autocast('cpu', dtype=torch.bfloat16)
    with torch.no_grad(), ctx:
        out = bb(img.to(torch.bfloat16), is_training=True)
    y = out['x_norm_patchtokens'].float()
    return (y - mean) / rms
