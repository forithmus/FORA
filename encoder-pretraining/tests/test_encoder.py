"""fora.encoder: geometry of the ViT-L, the block-23 / block-24 split, Block24 on cached crops, cache dequantization."""
import numpy as np
import torch

from conftest import tiny_backbone


def test_vitl_geometry_and_periods():
    from fora.encoder.backbone import build_backbone
    with torch.device("meta"):
        bb = build_backbone("large")
    assert sum(p.numel() for p in bb.parameters()) == 302_894_080
    periods = bb.rope_embed._make_periods(device=None)
    assert bb.rope_embed.axis_pairs == (11, 11, 10)
    assert abs(float(periods[0]) - 6.0) < 1e-4 and abs(float(periods[10]) - 1200.0) < 1e-2
    assert bb.n_storage_tokens == 4 and bb.patch_size == (2, 16, 16)


def test_split_reproduces_forward():
    from fora.encoder.blocks import LAST, block23_tokens, final_block
    bb = tiny_backbone(depth=LAST + 1).eval()
    x = torch.randn(2, 1, 4, 32, 48)
    with torch.no_grad():
        ref = bb.forward_features(x)["x_norm_patchtokens"]
        h23, rope = block23_tokens(bb, x)
        got = final_block(bb.blocks[LAST], bb.norm, h23, rope)
    assert torch.allclose(got, ref, atol=1e-5)


def test_block24_matches_full_crop():
    from fora.encoder.blocks import LAST, PREFIX, Block24, block23_tokens, final_block
    bb = tiny_backbone(depth=LAST + 1).eval()
    x = torch.randn(1, 1, 64, 192, 192)
    with torch.no_grad():
        h23, rope = block23_tokens(bb, x)
        ref = final_block(bb.blocks[LAST], bb.norm, h23, rope)
        blk = Block24(bb.blocks[LAST], bb.norm, bb.rope_embed).eval()
        n = h23.shape[1] - PREFIX
        y = blk(h23, torch.arange(n)[None], torch.ones(1, PREFIX + n, dtype=torch.bool))[:, PREFIX:]
        # body-only subset with padding: drop half the tokens, results for kept tokens must use the reduced context
        keep = torch.arange(0, n, 2)
        xs = torch.cat([h23[:, :PREFIX], h23[:, PREFIX + keep]], 1)
        pad = torch.cat([xs, torch.zeros(1, 7, xs.shape[2])], 1)
        valid = torch.ones(1, pad.shape[1], dtype=torch.bool)
        valid[:, -7:] = False
        pos = torch.cat([keep, torch.zeros(7, dtype=torch.long)])[None]
        a = blk(xs, keep[None], torch.ones(1, xs.shape[1], dtype=torch.bool))
        b = blk(pad, pos, valid)[:, :xs.shape[1]]
    assert torch.allclose(y, ref, atol=1e-4)
    assert torch.allclose(a, b, atol=1e-5)


def test_crop_layout_and_dequantize():
    from fora.encoder.blocks import crop_layout
    from fora.encoder.cache import dequantize
    grid = (40, 15, 13)
    flat = np.sort(np.random.default_rng(0).choice(np.prod(grid), 3000, replace=False))
    order, ids, starts, local, n_crops = crop_layout(flat, grid)
    assert n_crops == 2 * 2 * 2 and starts[-1] == len(flat) and len(ids) == len(starts) - 1
    assert local.max() < 32 * 12 * 12
    x = torch.randn(len(flat), 16)
    rows = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    scale = torch.zeros(len(starts) - 1, 16).index_reduce_(0, torch.from_numpy(rows), x.abs(), "amax") / 127
    q = torch.round(x / scale[rows]).clamp(-127, 127).to(torch.int8)
    oidx = np.array([3, 7])
    view = dict(q=q.numpy(), scale=scale.half().numpy(), oidx=oidx, starts=starts,
                outlier=x[:, oidx].to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))
    y = dequantize(view)
    assert np.abs(y - x.numpy()).max() < 0.05


def test_cxr_tiles_partition():
    from fora.encoder.cxr import film_tiles
    u = np.random.default_rng(0).uniform(-1, 1, size=(1500, 1100)).astype(np.float32)
    tiles, sel, pf, grid = film_tiles(u, 1024)
    assert tiles.shape == (4, 1024, 1024) and grid == (94, 69)
    assert sum(len(s) for s in sel) == 94 * 69 == len(pf)
