"""Chest X-ray study input contract: film reading and the study case (paper section "Chest radiograph model").

A study is one JSON with all films of one report. Each film is read as a 2-D float32 raw grayscale image in the
convention of the training store (MONOCHROME2, no window or VOI LUT, original bit depth). Content cropping and
percentile normalization are done by `fora.geometry.xray` when the visual stream loads the film.

Validation status: the internal H5 entry was checked end to end against stored model outputs. DICOM, NPY and
PNG/TIFF/JPEG use the same preprocessing but were not checked against stored outputs. 8-bit windowed exports are a
known domain shift (higher scores, larger plans).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

VIEW = {"frontal": 0, "pa": 0, "ap": 0, "lateral": 1, "other": 2}
MAX_FILMS = 64


def read_pixels(item: dict, base: Path) -> np.ndarray:
    """One `films[]` entry -> 2-D float32 film."""
    if "h5" in item:
        import h5py
        p = Path(item["h5"])
        p = p if p.is_absolute() else base / p
        with h5py.File(p, "r") as f:
            d = f["/".join(item["key"])]["data"][()]
        return np.asarray(d[:, :, 0] if d.ndim == 3 else d, dtype=np.float32)
    p = Path(item["path"])
    p = p if p.is_absolute() else base / p
    suf = p.suffix.lower()
    if suf == ".npy":
        a = np.load(p)
    elif suf in (".dcm", "") or item.get("format") == "dicom":
        import pydicom
        ds = pydicom.dcmread(str(p))
        a = ds.pixel_array.astype(np.float32)
        if a.ndim == 3:
            a = a[0] if a.shape[-1] not in (3, 4) else a[..., 0]
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            a = a.max() - a
    else:
        from PIL import Image
        a = np.asarray(Image.open(p))
        if a.ndim == 3:
            a = a[..., 0]
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"{p}: expected a 2-D grayscale film, got {a.shape}")
    return a


def make_case(study: dict, base: Path) -> dict:
    """Study JSON -> visual-stream case (films as [film id, film id, view kind, 0, 0]) plus the film sources."""
    if "study_id" not in study:
        raise ValueError("study_id is required")
    films, sources = [], {}
    for i, f in enumerate(study.get("films", [])):
        view = str(f.get("view", "other")).lower()
        if view not in VIEW:
            raise ValueError(f"film {i}: view must be one of {sorted(VIEW)}")
        if "h5" not in f and "path" not in f:
            raise ValueError(f"film {i}: path (or h5 + key) is required")
        films.append([str(i), str(i), VIEW[view], 0, 0])
        sources[str(i)] = f
    if not films:
        raise ValueError("study.films is empty")
    if len(films) > MAX_FILMS:
        raise ValueError(f"At most {MAX_FILMS} films per study are supported")
    return dict(id=str(study["study_id"]), h5="", patient="", report="", films=films, _sources=sources, _base=str(base))


def film_loader(case: dict, boxes: dict):
    """Film loader for the visual stream: read, content-crop and normalize; records the content box of every film."""
    from fora.geometry.xray import content_bbox, to_unit

    def load_film(film, max_px=0):
        img = read_pixels(case["_sources"][film[0]], Path(case["_base"]))
        r0, r1, c0, c1 = content_bbox(img)
        boxes[film[0]] = dict(content_box_rc=[int(r0), int(r1), int(c0), int(c1)], shape=list(img.shape))
        return to_unit(img[r0:r1, c0:c1])
    return load_film


def load_study(path: Path) -> tuple[dict, dict]:
    study = json.loads(path.read_text())
    return study, make_case(study, path.resolve().parent)
