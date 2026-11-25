"""Class-region bits: which token regions each finding may attend to (Methods, "Organ routing").

A class-region table is a JSON object {class name: int bitmask over REGION_NAMES} (Python ints, may exceed 2**63).
Classes missing from the table, and the classes set to "no routing", use ALL_BITS (every token). A token routes to
class c if its region bits intersect class_bits[c]. The routing tables themselves are built in the classifier stage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .definitions import REGION_NAMES

ALL_BITS = (1 << 64) - 1
PELVIS_BIT = REGION_NAMES.index('pelvis')
HIP_BONES_BIT = REGION_NAMES.index('pelvic_bones_hips')


def load_class_region_bits(path, classes) -> np.ndarray:
    """uint64 [C] bitmask per class in `classes` order (missing classes -> ALL_BITS)."""
    rb = json.loads(Path(path).read_text())
    return np.array([int(rb.get(n, ALL_BITS)) for n in classes], dtype=np.uint64)


def bits_of(regions) -> int:
    """Bitmask of a list of region names."""
    out = 0
    for r in regions:
        out |= 1 << REGION_NAMES.index(r)
    return out


def clear_pelvis_without_hips(bits: np.ndarray) -> np.ndarray:
    """Clear the 'pelvis' bit of a series whose tokens carry no hip-bone bit.

    Without hip bones in the field of view the 'pelvis' landmark (hip top) is missing and the region degenerates to the
    whole upper peritoneum; the MIL trainer and inference clear it. bits: int64 or uint64 per-token array of one view.
    """
    b = np.asarray(bits)
    u = b.view(np.uint64) if b.dtype == np.int64 else b.astype(np.uint64, copy=False)
    if not bool(((u >> np.uint64(HIP_BONES_BIT)) & np.uint64(1)).any()):
        u = u & ~np.uint64(1 << PELVIS_BIT)
        return u.view(np.int64) if b.dtype == np.int64 else u
    return b
