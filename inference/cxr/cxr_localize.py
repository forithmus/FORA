"""Signed-evidence localization of planned chest X-ray findings (paper section "Localization").

Per planned finding and film: E = max(w (sL - b_c), 0) on the film's token grid, with w the study-bag attention
softmax, sL the token logit times the output scale and b_c the normal-tissue level of the class (token-weighted mean
sL over 2,000 training studies negative for it). The map is smoothed with a 4 mm Gaussian (token size = 16 px x
430 mm / long side of the content box). Foci are 8-connected components of the smoothed map >= 0.3 x its 99.5 %
quantile. The point is the peak of the focus with the largest mass, and the HDR80 area share is the fraction of the
grid holding 80 % of the smoothed mass. A film without positive evidence has no focus. Attention alone is never used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def localize(token_logits, weights, output_scale, film_geometry, grids, films, boxes, classes, b_ref, labels,
             work: Path, cfg: dict) -> list[dict]:
    """token_logits / weights: [N, C] tensors of the bag; film_geometry: (row, col, gy, gx) per token;
    grids: (gy, gx) per loaded film in bag order; films: case film list; boxes: content boxes by film id."""
    import torch
    from fora.mil.evidence import xray_film_focus, xray_token_evidence
    scale = output_scale.float()
    counts = [int(gy) * int(gx) for gy, gx in grids]
    row, col = [t.long().cpu().numpy() for t in film_geometry[:2]]
    starts = np.cumsum([0] + counts)
    loaded = [f[0] for f in films if f[0] in boxes]
    maps, out = {}, []
    for c in classes:
        b = torch.tensor([float(b_ref[labels[c]])], device=weights.device)
        e = xray_token_evidence(weights[:, [c]], token_logits[:, [c]].float(), scale[[c]], b)[:, 0].cpu().numpy()
        for k, (gy, gx) in enumerate(grids):
            a0, a1 = starts[k], starts[k + 1]
            E = np.zeros((gy, gx), np.float64)
            E[row[a0:a1], col[a0:a1]] = e[a0:a1]
            rec = xray_film_focus(E, boxes[loaded[k]]["content_box_rc"], quantile=cfg["quantile"],
                                  rel=cfg["focus_fraction_of_q995"])
            sm = rec.pop("map", None)
            if sm is not None:
                maps[f"{labels[c]}|film{loaded[k]}"] = sm.astype(np.float16)
            out.append(dict(index=int(c), name=labels[c], film_index=int(loaded[k]),
                            view_kind=int(films[int(loaded[k])][2]), **rec))
    if maps:
        np.savez_compressed(Path(work) / "evidence_maps.npz", **{k.replace("/", "_"): v for k, v in maps.items()})
    return out
