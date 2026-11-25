"""Zoomed evidence images -> Gemma-4 vision-tower inputs (paper section "Grounded report generation", evidence images).

Each planned finding has up to six JPEG views (axial / coronal / sagittal at the evidence peak, then axial views of further
sites; ``fora.mil.render``). Only existing views are encoded; the writer pools each image to one token.
The Gemma-4 image processor runs with ``max_soft_tokens=140`` (the value of the paper runs, training and inference).
"""
from __future__ import annotations

FIMG_SOFT = 140


def make_fimg_processor(fimg_soft: int = FIMG_SOFT):
    """Gemma-4 image processor used for the finding images."""
    from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
    return Gemma4ImageProcessor(max_soft_tokens=int(fimg_soft))


def fimg_batch(processor, images):
    """[(view JPEG bytes, ...)] per finding (plan order) -> (pixel_values, image_position_ids, mask [findings, views] bool)."""
    import io

    import torch as _t
    from PIL import Image
    pil, mask = [], []
    if not images:   # no planned finding: one black image with an empty mask so every rank still runs the vision tower (FSDP)
        out = processor(images=[Image.new("RGB", (448, 448))], return_tensors="pt")
        return out["pixel_values"], out["image_position_ids"], _t.zeros((0, 0), dtype=_t.bool)
    for views in images:   # only existing views are encoded; a finding without any image gets one black image
        row = [bool(b) for b in views]
        if not any(row):
            row[0] = True; views = (None,) + tuple(views[1:])
        mask.append(row)
        for b, keep in zip(views, row):
            if keep:
                pil.append(Image.open(io.BytesIO(b)).convert("RGB") if b else Image.new("RGB", (448, 448)))
    out = processor(images=pil, return_tensors="pt")
    return out["pixel_values"], out["image_position_ids"], _t.tensor(mask, dtype=_t.bool)
