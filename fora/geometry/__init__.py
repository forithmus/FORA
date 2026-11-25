"""CT and X-ray geometry shared by preprocessing, training and inference.

Modules
  tiling     RAS reorientation, resampling to 1.5 x 0.5 x 0.5 mm, body crop (Methods, "CT preprocessing").
  windows    HU windows from DICOM metadata, window normalization, the 10 mm lung slab MIP, the view rule.
  protocols  the model-side protocol taxonomy (29 ids) and the lung-CT protocol set.
  series     selection of usable series in the CT H5 shards (derivative preference, frames, sane affine).
  tokens     token grid of the (2, 16, 16) patch, token mean HU and the body-token rule.
  world      world (patient mm) coordinates of cached tokens.
  xray       X-ray film reading: content crop and percentile normalization.
"""
from .tiling import TARGET_SPACING, body_frame, crop_to_body, resample_series, to_ras
from .tokens import PATCH, body_mask, token_grid, token_mean_hu
from .windows import FULL_WINDOW, LUNG_WINDOW_MAX_CENTER, MIP_SLAB_MM, plan_views, slab_mip, window_normalize

__all__ = [
    "TARGET_SPACING", "body_frame", "crop_to_body", "resample_series", "to_ras",
    "PATCH", "body_mask", "token_grid", "token_mean_hu",
    "FULL_WINDOW", "LUNG_WINDOW_MAX_CENTER", "MIP_SLAB_MM", "plan_views", "slab_mip", "window_normalize",
]
