"""Token grid of the CT encoder and the body-token rule.

The encoder patch is (2, 16, 16) voxels = 3 x 8 x 8 mm at 1.5 x 0.5 x 0.5 mm. A token is a body token if its mean HU
(air-padded volume) is above -500 HU, or it lies inside the per-slice hole-filled mask of such tokens. Only body tokens
are cached and seen by the MIL head (Methods, "Block-23 token cache").
"""
from __future__ import annotations

import numpy as np

PATCH = (2, 16, 16)
BODY_TOKEN_HU = -500.0
BODY_RULE = "token mean HU > -500 OR per-z-slice binary_fill_holes(token mean HU > -500)"


def token_grid(volume_shape, patch=PATCH):
    """(gz, gy, gx) token grid covering a (D, H, W) volume (ceil division)."""
    z, y, x = (int(v) for v in volume_shape)
    return (z + patch[0] - 1) // patch[0], (y + patch[1] - 1) // patch[1], (x + patch[2] - 1) // patch[2]


def token_mean_hu(vol: np.ndarray, air_hu: float = -1000.0, patch=PATCH) -> np.ndarray:
    """Mean HU per token of the air-padded volume -> (gz, gy, gx) float32."""
    gz, gy, gx = token_grid(vol.shape, patch)
    padded = np.full((gz * patch[0], gy * patch[1], gx * patch[2]), air_hu, np.float32)
    padded[:vol.shape[0], :vol.shape[1], :vol.shape[2]] = vol
    return padded.reshape(gz, patch[0], gy, patch[1], gx, patch[2]).mean(axis=(1, 3, 5))


def body_mask(mean_hu: np.ndarray, threshold: float = BODY_TOKEN_HU) -> np.ndarray:
    """Body token mask on the token grid (see BODY_RULE)."""
    from scipy.ndimage import binary_fill_holes
    solid = mean_hu > threshold
    body = solid.copy()
    for z in range(solid.shape[0]):
        body[z] = binary_fill_holes(solid[z]) | solid[z]
    return body
