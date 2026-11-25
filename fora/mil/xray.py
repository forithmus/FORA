"""Chest X-ray MIL: head build and token factory over online block-24 tokens (Methods, "Chest X-ray").

The X-ray classifier uses the CT head (``fora.mil.model``) on the X-ray encoder: frozen teacher blocks 1-23 run once per
step on the 1,024-px tiles of every film of a report, the trainable (block 24, norm) pair runs inside the token factory.
A view-kind embedding (0 frontal, 1 lateral, 2 other; zero-initialised) is added to every patch token of a tile before
block 24. X-ray has no organ regions: every token is in every route, so the soft route penalty never applies.

A prepared study (``prepare_study``) holds the tiles, the owned token selection of every tile, film-grid position features,
the view kind of every token and the film grids, in film order.
"""
from __future__ import annotations

import numpy as np
import torch

from fora.encoder.blocks import PREFIX
from fora.encoder.cxr import film_tiles
from fora.geometry.xray import content_bbox, to_unit
from fora.mil.model import build_head

VIEW_KIND = {"frontal": 0, "lateral": 1, "other": 2}


def unit_film(img: np.ndarray) -> np.ndarray:
    """Raw 2-D film -> content box -> [-1, 1] (percentile stretch)."""
    r0, r1, c0, c1 = content_bbox(img)
    return to_unit(img[r0:r1, c0:c1])


def prepare_study(study_id: str, films, tile: int = 1024, freqs: int = 4) -> dict:
    """films: list of (prepared [-1, 1] film, view kind) -> tiles + owned-token selections + position features + kinds."""
    tiles, sel, pf, kind, grids = [], [], [], [], []
    for u, k in films:
        t, s, p, g = film_tiles(u, tile, freqs)
        tiles.append(t)
        sel += s
        pf.append(p)
        kind.append(np.full(len(p), k, np.int64))
        grids.append(g)
    return dict(id=study_id, tiles=torch.cat(tiles), sel=sel, pos=torch.from_numpy(np.concatenate(pf)),
                kind=torch.from_numpy(np.concatenate(kind)), grids=grids)


def build_xray_model(spec: dict, n_classes: int, top: torch.nn.ModuleDict, device, dim: int = 1048) -> dict:
    """Head + view embedding + AdamW groups (head, route penalty, block-24 matrices, block-24 vectors, view embedding)."""
    head = build_head(n_classes, spec.get("soft_init", 5.), dim).to(device).train()
    rest = [p for n, p in head.named_parameters() if n != "route_bias_raw"]
    lr_route = spec.get("lr_route", 1e-2)
    groups = [dict(params=rest, lr=spec["lr_head"], weight_decay=spec["wd_head"], base=spec["lr_head"]),
              dict(params=[head.route_bias_raw], lr=lr_route, weight_decay=0., base=lr_route)]
    decay = [p for p in top.parameters() if p.ndim > 1]
    other = [p for p in top.parameters() if p.ndim <= 1]
    groups += [dict(params=decay, lr=spec["lr_block"], weight_decay=spec["wd_block"], base=spec["lr_block"]),
               dict(params=other, lr=spec["lr_block"], weight_decay=0., base=spec["lr_block"])]
    view = torch.nn.Embedding(len(VIEW_KIND), 1024).to(device)
    torch.nn.init.zeros_(view.weight)
    groups += [dict(params=[view.weight], lr=spec["lr_head"], weight_decay=spec["wd_head"], base=spec["lr_head"])]
    params = [p for g in groups for p in g["params"]]
    return dict(spec=spec, block=top, head=head, view=view, params=params, optimizer=torch.optim.AdamW(groups),
                ema=[p.detach().clone() for p in params] if spec.get("ema") else None)


def make_xray_factory(enc, h23, bag, model, mean, rms, tiles_per_chunk, device, n_classes, seed=None):
    """(x, all-true route mask) chunks in a fixed tile order; training drops token rows (0.2) and feature entries (0.1)
    of the standardised features with a seeded generator (identical in both passes)."""
    spec = model["spec"]
    drop_t = spec.get("token_drop", 0.) if seed is not None else 0.
    drop_f = spec.get("feature_drop", 0.) if seed is not None else 0.
    ps = spec.get("position_scale", 1.)

    def factory():
        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(int(seed))
        o = 0
        starts = np.cumsum([0] + [len(s) for s in bag["sel"][:-1]])
        tk = bag["kind"][torch.as_tensor(starts)].to(device)          # view kind of each tile
        for i in range(0, len(h23), tiles_per_chunk):
            h = h23[i:i + tiles_per_chunk]
            ve = model["view"](tk[i:i + tiles_per_chunk])[:, None, :].expand(-1, h.shape[1] - PREFIX, -1)
            h = torch.cat((h[:, :PREFIX], h[:, PREFIX:] + ve), 1)
            y = enc.b24(h)
            sel = bag["sel"][i:i + tiles_per_chunk]
            f = torch.cat([y[b][torch.as_tensor(s, device=device)] for b, s in enumerate(sel)]).float()
            n = len(f)
            pf = bag["pos"][o:o + n].to(device).float()
            o += n
            x = (f - mean) / rms
            if drop_t:
                keep = torch.rand(len(x), generator=g, device=device) >= drop_t
                x, pf = x[keep], pf[keep]
                if len(x) == 0:
                    continue
            if drop_f:
                x = x * ((torch.rand(x.shape, generator=g, device=device) >= drop_f).float() / (1 - drop_f))
            yield torch.cat((x, pf * ps), 1), torch.ones(len(x), n_classes, dtype=torch.bool, device=device)
    return factory
