"""Frozen X-ray MIL visual stream for the chest X-ray writer (paper section "Grounded report generation", X-ray arm).

The stream is exactly what the X-ray MIL head sees at evaluation (study bag = all films of the report):
    films -> content box + percentile stretch -> 1024-px tiles (owned tokens) -> frozen DINO teacher blocks 1-23
    -> + view-kind embedding (before block 24) -> MIL block 24 + norm -> x = [(y - mean) / rms, Fourier film-grid
    position features] (1048-d) -> writer resampler (all tokens of the study, about 25k).
The same pass runs the frozen MIL head on the tokens:
  * per-class attention weights w[t, c] = softmax_t(score[t, c]) -> pooled finding feature f[c] = sum_t w[t, c] x[t]
    (1048-d), the X-ray analogue of the CT attention-pooled finding token;
  * bag logits sum_t w * token_logit * scale + bias, used only as a consistency check against the precomputed logits
    (the plan uses the precomputed study-bag predictions the calibration was fitted on);
  * optionally the leave-one-out location statistics of ``fora.reportgen.cxr.location``.
The writer uses the MIL checkpoint with its raw (non-EMA) weights.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

N_VIEW_KINDS = 4       # 0 frontal, 1 lateral, 2 other (3 unused; same table size as CT)
MAX_SERIES = 32        # film ordinal within the study (clamped)


def h5_film_loader(case: dict, h5_root: str):
    """Film loader of the internal X-ray shards: <h5_root>/<h5>.h5 [patient][report][acc][sid]['data'] -> [-1, 1] film."""
    from fora.geometry.xray import read_h5_film

    def load_film(film):
        return read_h5_film(f"{h5_root}/{case['h5']}.h5", case['patient'], case['report'], film[0], film[1])
    return load_film


class CXRVisualStream:
    """Frozen X-ray MIL (teacher 1-23 + block 24/view/head) -> 1048-d study tokens + per-class pooled features."""

    def __init__(self, mil_config: str, mil_checkpoint: str, device, variant: str = 'raw'):
        from fora.encoder.cxr import CXRSplitEncoder
        from fora.mil.xray import build_xray_model
        m = json.loads(Path(mil_config).read_text())
        self.m = m
        self.cfg = cfg = m['training']
        spec = cfg['model']
        self.device = torch.device(device)
        src = torch.load(m['normalization'], map_location=self.device, weights_only=False)
        self.mean = src['mean'].to(self.device).float()
        self.rms = float(src['rms'])
        self.enc = CXRSplitEncoder(m['teacher_backbone'], torch.load(m['block24_state'], map_location='cpu', weights_only=False),
                                   self.device)
        s = torch.load(mil_checkpoint, map_location=self.device, weights_only=False)
        self.classes = list(s['classes'])
        self.C = len(self.classes)
        if int(s['teacher_step']) != int(self.enc.step):
            raise RuntimeError(f"snapshot teacher step {s['teacher_step']} != exported teacher {self.enc.step}")
        self.dim = 1024 + 6 * int(cfg.get('position_freqs', 4))
        model = build_xray_model(spec, self.C, self.enc.top, self.device, self.dim)
        model['head'].load_state_dict(s['head'])
        self.enc.top.load_state_dict(s['block'])
        model['view'].load_state_dict(s['view'])
        if variant == 'ema':
            with torch.no_grad():
                for p, e in zip(model['params'], s['ema']):
                    p.copy_(e)
        del s
        model.pop('optimizer', None)
        model.pop('ema', None)
        for p in model['params']:
            p.requires_grad_(False)
        model['head'].eval()
        self.model = model
        self.head = model['head']
        self.allc = torch.arange(self.C, device=self.device)
        self.fb = int(cfg.get('frozen_batch', 8))
        self.variant = variant

    def prepare(self, case: dict, load_film=None, h5_root: str | None = None) -> dict:
        """Host side (loader threads): all films (<= max_films) -> tiles/selections/position features/view kinds.

        ``load_film(film) -> [-1, 1] film`` reads one film of the case (default: the internal H5 shards under
        ``h5_root``); a film that fails to load is skipped."""
        from fora.mil.xray import prepare_study
        load_film = load_film or h5_film_loader(case, h5_root)
        films = []
        for film in case['films'][:self.cfg.get('max_films', 8)]:
            try:
                films.append((load_film(film), int(film[2])))
            except Exception as e:
                print(json.dumps(dict(event='film_skip', case=case['id'], error=repr(e)[:200])), flush=True)
        if not films:
            raise RuntimeError(f"no loadable film in {case['id']}")
        return prepare_study(case['id'], films, self.cfg.get('tile', 1024), self.cfg.get('position_freqs', 4))

    @torch.no_grad()
    def encode(self, bag: dict, location: bool = False, keep_scores: bool = False) -> SimpleNamespace:
        """All tokens of the study bag through the frozen MIL; ``keep_scores`` also returns the token logits
        (``token_logits`` [N, C]) and the attention softmax (``weights`` [N, C]) for the evidence maps."""
        from fora.mil.xray import make_xray_factory
        h23 = self.enc.h23(bag['tiles'], self.fb)
        factory = make_xray_factory(self.enc, h23, bag, self.model, self.mean, self.rms, self.cfg.get('tiles_per_chunk', 8),
                                    self.device, self.C)
        xs, scores, token_logits = [], [], []
        for x, _mask in factory():
            sc, lg = self.head._chunk_outputs(x, self.allc)
            xs.append(x.to(torch.bfloat16))
            scores.append(sc.float())
            token_logits.append(lg.float())
        del h23
        tokens = torch.cat(xs)
        S = torch.cat(scores)
        L = torch.cat(token_logits)
        w = torch.softmax(S, 0)
        logits = (w * L).sum(0) * self.head.output_scale.float() + self.head.output_bias.float()
        pooled = w.t() @ tokens.float()                                  # [C, 1048]
        n = int(tokens.shape[0])
        counts = [int(gy) * int(gx) for gy, gx in bag['grids']]
        if sum(counts) != n or int(bag['kind'].numel()) != n:
            raise RuntimeError(f"{bag.get('id')}: {n} tokens, grids {sum(counts)}, kinds {bag['kind'].numel()}")
        film = torch.repeat_interleave(torch.arange(len(counts), device=self.device),
                                       torch.tensor(counts, device=self.device)).clamp_(max=MAX_SERIES - 1)
        kind = bag['kind'].to(self.device).long().clamp_(0, N_VIEW_KINDS - 1)
        loc = self.location_stats(S, L, w, bag) if location else None
        out = SimpleNamespace(tokens=tokens, view_kind=kind, view_series=film, pooled=pooled, logits=logits,
                              total_tokens=n, films=len(counts), location=loc)
        if keep_scores:
            out.token_logits, out.weights = L, w
        return out

    def film_geometry(self, bag):
        """Per bag token: (row, col, gy, gx) on its content-cropped film grid, in the bag order (films in order, tiles in
        order, owned tokens row-major)."""
        from fora.encoder.cxr import PATCH_PX, axis_tiles
        tile = int(self.cfg.get('tile', 1024))
        rows, cols, GY, GX = [], [], [], []
        for gy, gx in bag['grids']:
            _ys, oy, g1 = axis_tiles(int(gy) * PATCH_PX, tile)
            _xs, ox, g2 = axis_tiles(int(gx) * PATCH_PX, tile)
            if (g1, g2) != (gy, gx):
                raise RuntimeError('film grid mismatch')
            for (ylo, yhi) in oy:
                for (xlo, xhi) in ox:
                    ty, tx = np.meshgrid(np.arange(ylo, yhi), np.arange(xlo, xhi), indexing='ij')
                    rows.append(ty.ravel())
                    cols.append(tx.ravel())
                    GY.append(np.full(ty.size, gy))
                    GX.append(np.full(ty.size, gx))
        return [torch.as_tensor(np.concatenate(a), device=self.device, dtype=torch.float32) for a in (rows, cols, GY, GX)]

    @torch.no_grad()
    def location_stats(self, S, L, w, bag):
        """Positive leave-one-out evidence on frontal films -> [C, 5] (mass, frac_right, upper, mid, lower)."""
        return location_statistics(L, w, self.head.output_scale.float(), self.film_geometry(bag),
                                   bag['kind'].to(self.device))


def location_statistics(L, w, scale, geometry, kind):
    """[T, C] token logits L and attention w, per-class output scale, per-token film geometry and view kind ->
    [C, 5] numpy (positive LOO mass on frontal films, frac patient-right, upper / mid / lower zone shares)."""
    sl = L * scale[None]
    mu = (w * sl).sum(0, keepdim=True)
    loo = w * (sl - mu) / (1 - w).clamp_min(1e-6)
    e = loo.clamp_min(0)
    row, col, gy, gx = geometry
    if len(row) != len(e):
        raise RuntimeError('location geometry / token count mismatch')
    frontal = (kind == 0).float()
    cc = col + 0.5
    right = torch.where(cc < gx / 2, 1.0, torch.where(cc > gx / 2, 0.0, 0.5)) * frontal   # image left = patient right
    zi = torch.clamp(((row + 0.5) * 3 / gy).floor(), 0, 2)
    ef = e * frontal[:, None]
    mass = ef.sum(0)
    den = mass.clamp_min(1e-12)
    out = [mass, (e * right[:, None]).sum(0) / den]
    for k in range(3):
        out.append((ef * (zi == k).float()[:, None]).sum(0) / den)
    return torch.stack(out, 1).cpu().numpy()   # [C, 5]
