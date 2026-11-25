"""Fourier features of a token's position in its scan (Methods, "Sentence alignment" and "Finding classifier").

Block 24 only sees positions inside its own crop. The heads after it additionally receive where the token lies in the whole
scanned volume (CT) or film (X-ray): each axis is normalized to 0..1 over the grid and encoded with ``freqs`` sine/cosine
octaves, giving 6 x freqs features (24 with the default 4).
"""
from __future__ import annotations

import numpy as np


def position_features(flat: np.ndarray, grid, freqs: int = 4) -> np.ndarray:
    """flat token indices in a (gz, gy, gx) grid -> float16 [N, 6 * freqs] (sin block of 3 x freqs, then cos block)."""
    gz, gy, gx = grid
    z, rem = np.divmod(flat, gy * gx)
    y, x = np.divmod(rem, gx)
    u = np.stack([(z + .5) / gz, (y + .5) / gy, (x + .5) / gx], 1).astype(np.float32)
    ang = u[:, :, None] * (np.pi * 2.0 ** np.arange(freqs))[None, None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], 2).reshape(len(flat), -1).astype(np.float16)
