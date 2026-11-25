"""CADS-derived token regions for organ routing and evidence restriction (Methods, "Organ routing").

  definitions  REGIONS (CADS label unions and operations), REGION_NAMES (the 63 emitted bits, fixed order), MIN_TOKENS
  builder      Builder (region masks on the 1.5 x 2 x 2 mm work grid), token_cover (>= 25 % coverage)
  token_bits   token_region_bits(series file) -> per-token uint64 bits; work-grid geometry and CADS sampling
  class_bits   class-region bit tables, ALL_BITS, the pelvis-without-hip-bones fix
  families     family_of(class name) -> anatomical family used to restrict evidence maps
"""
from .builder import COVER, WORK_MM, Builder, token_cover
from .class_bits import ALL_BITS, bits_of, clear_pelvis_without_hips, load_class_region_bits
from .definitions import HELPER_REGIONS, MIN_TOKENS, REGION_NAMES, REGIONS
from .families import family_bits, family_of
from .token_bits import token_region_bits, work_axes

__all__ = [
    "COVER", "WORK_MM", "Builder", "token_cover",
    "ALL_BITS", "bits_of", "clear_pelvis_without_hips", "load_class_region_bits",
    "HELPER_REGIONS", "MIN_TOKENS", "REGION_NAMES", "REGIONS",
    "family_bits", "family_of",
    "token_region_bits", "work_axes",
]
