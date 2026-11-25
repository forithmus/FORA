"""Per-(report, finding) label states of the FORA CT label database (Methods, report labels).

The label DB stores one int8 state per report and hierarchy node. Training targets treat asserted and uncertain as
positive and denied and unmentioned as negative; masked entries carry no loss.
"""
from __future__ import annotations

import numpy as np

DENIED = 0          # explicitly negated in the report (or a descendant of a negated node)
ASSERTED = 1        # stated as present (or an ancestor of an asserted node)
UNCERTAIN = 2       # hedged (or an ancestor of an uncertain node), never overriding an asserted state
UNMENTIONED = 3     # not mentioned: implicit negative with full weight (no field-of-view rule)
MASKED = -1         # unknown; no loss

POSITIVE_STATES = (ASSERTED, UNCERTAIN)
NEGATIVE_STATES = (DENIED, UNMENTIONED)


def is_positive(state: np.ndarray) -> np.ndarray:
    return (state == ASSERTED) | (state == UNCERTAIN)


def is_negative(state: np.ndarray) -> np.ndarray:
    return (state == DENIED) | (state == UNMENTIONED)


def to_target(state: np.ndarray) -> np.ndarray:
    """float32 targets: 1 for asserted/uncertain, 0 for denied/unmentioned, NaN for masked."""
    state = np.asarray(state)
    return np.where(state < 0, np.nan, is_positive(state).astype(np.float32)).astype(np.float32)
