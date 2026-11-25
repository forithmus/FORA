"""HU windows, window normalization, the 10 mm lung slab MIP and the per-series view rule.

Implements the "views" of the CT preprocessing (Methods). Every usable series is encoded once per view:
  * FULL: the HU window [-1000, 1000];
  * every distinct native DICOM window stored with the series (WindowCenter / WindowWidth pairs);
  * a 10 mm rolling slab maximum-intensity projection along z, rendered in the deepest-centre native lung window,
    when the series has a native lung window (window centre below -250 HU).

The lung-MIP gate differs between the training data and the deployed inference bundle:
  * mip_rule="training" (default; the rule that produced the training cache): a native lung window is enough, there is
    no protocol gate;
  * mip_rule="inference" (the deployed bundle): the study protocol must also be a lung-CT protocol
    (fora.geometry.protocols.LUNG_CT_CLASSES).
"""
from __future__ import annotations

import json
import re

import numpy as np
import torch

FULL_WINDOW = (-1000, 1000)
LUNG_WINDOW_MAX_CENTER = -250      # a native window centred below this is a lung window
MIP_SLAB_MM = 10.0
MIP_RULES = ("training", "inference")


def windows_from_meta(meta: dict) -> list[tuple[int, int]]:
    """Distinct HU windows (lo, hi) = (int(WC - WW/2), int(WC + WW/2)) from paired WindowCenter / WindowWidth values.

    Accepts multi-valued fields ('\\'-separated PACS values, '[..]' dcm2niix lists, commas, numbers or lists).
    """
    def vals(v):
        if v is None:
            return []
        return [float(x) for x in re.findall(r"-?\d+\.?\d*", str(v).replace("\\", " "))]

    wc, ww = vals(meta.get("WindowCenter")), vals(meta.get("WindowWidth"))
    out, seen = [], set()
    for c, w in zip(wc, ww):
        if w > 0:
            lo, hi = int(c - w / 2), int(c + w / 2)
            if hi > lo and (lo, hi) not in seen:
                seen.add((lo, hi))
                out.append((lo, hi))
    return out


def _dec(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else x


def native_windows(series_group) -> list[list[int]]:
    """Native windows of one CT H5 series group: `pacs_metadata` first, else `dcm2niix_metadata` (first non-empty)."""
    for key in ('pacs_metadata', 'dcm2niix_metadata'):
        if key in series_group:
            try:
                w = windows_from_meta(json.loads(_dec(series_group[key][()])))
                if w:
                    return [list(x) for x in w]
            except Exception:
                pass
    return []


def _centre(w) -> float:
    return (w[0] + w[1]) / 2


def plan_views(natives, *, lung_protocol: bool | None = None, mip_rule: str = "training"):
    """Views of one series -> (windows, mip_window).

    windows    [FULL] + the distinct native windows other than FULL, as [lo, hi] lists (FULL first);
    mip_window the deepest-centre native lung window ([lo, hi]) when a lung MIP view is added, else None.
    mip_rule   "training" adds the MIP whenever a native lung window exists; "inference" additionally requires
               lung_protocol=True (the study protocol is a lung-CT protocol).
    """
    if mip_rule not in MIP_RULES:
        raise ValueError(f"mip_rule must be one of {MIP_RULES}")
    nat = [list(w) for w in natives]
    views = [list(FULL_WINDOW)] + [w for w in nat if w != list(FULL_WINDOW)]
    lung = [w for w in nat if _centre(w) < LUNG_WINDOW_MAX_CENTER]
    if mip_rule == "inference" and not lung_protocol:
        lung = []
    mip = min(lung, key=_centre) if lung else None
    return views, mip


def view_list(windows, mip):
    """Plan entry -> ordered view dicts: kind 'full' (index 0), 'native', then 'lung_mip_10mm'."""
    views = [dict(kind='full' if k == 0 else 'native', lo=float(w[0]), hi=float(w[1])) for k, w in enumerate(windows)]
    if mip is not None:
        views.append(dict(kind='lung_mip_10mm', lo=float(mip[0]), hi=float(mip[1])))
    return views


def window_normalize(crop_hu: np.ndarray, hu_min: float, hu_max: float) -> torch.Tensor:
    """Clip a HU crop to [hu_min, hu_max] and map it linearly to [-1, 1] -> (1, D, H, W) bf16.

    Operation order (float32 clamp, subtract, divide, scale, bf16 cast) is the one used for the training cache.
    """
    if hu_max <= hu_min:
        raise ValueError("HU window maximum must be greater than its minimum")
    norm = torch.from_numpy(np.array(crop_hu, dtype=np.float32, copy=True))
    norm.clamp_(float(hu_min), float(hu_max))
    norm.sub_(float(hu_min)).div_(float(hu_max - hu_min)).mul_(2.0).sub_(1.0)
    return norm.unsqueeze(0).to(torch.bfloat16)


def slab_mip(vol: np.ndarray, slab_mm: float = MIP_SLAB_MM, z_spacing: float = 1.5) -> np.ndarray:
    """Rolling slab maximum-intensity projection along z: (D, H, W) -> (D, H, W).

    Each slice becomes the maximum over a slab_mm thick window centred on it (kernel round(slab_mm / z_spacing),
    'nearest' at the edges), so the projected volume keeps the geometry of the input.
    """
    from scipy import ndimage
    win = max(1, int(round(slab_mm / z_spacing)))
    return ndimage.maximum_filter1d(vol, size=win, axis=0, mode='nearest')
