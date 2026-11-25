"""Apply the train-fitted calibration and cut-offs of the classifier (Methods, "Finding classifier", operating points).

Calibration JSON (written by ``classifier/fit_calibration.py``): per finding, in the order of ``labels``,
  class_slope / class_intercept        Platt curve p = sigmoid(slope * logit + intercept), fitted on training studies;
  class_raw_plan_threshold             "probable" tier on the raw sigmoid: calibrated p >= 0.5, never below the train Youden cut;
  class_raw_possible_threshold         "possible" tier on the raw sigmoid: train Youden cut, never above the probable cut;
  class_plan_threshold / class_possible_threshold   the same cut-offs on the calibrated scale.

Rule A (``classifier/fit_rule_a.py``, the operating point of the reported balanced accuracy): one cut-off on the raw logit
per finding and scan type, fitted on training studies of that scan type; a finding is positive if logit >= cut-off.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

SCAN_TYPES = ("chest", "abdomen", "headneck", "CAP", "other")


def scan_type(protocol_titles) -> str:
    """Scan type of a study from its protocol titles (ordered rules; CAP = chest, abdomen and pelvis)."""
    p = " | ".join(protocol_titles or [])
    if "Chest, Abdomen and Pelvis" in p:
        return "CAP"
    if re.search(r"CT Chest - |CTA Chest|Coronary CTA", p):
        return "chest"
    if re.search(r"Abdomen|Urography", p):
        return "abdomen"
    if re.search(r"CT Head|CTA Head|Sinus|Temporal|CT Neck|CTA Neck|Dental", p):
        return "headneck"
    return "other"


def sigmoid(x):
    return 1. / (1. + np.exp(-np.asarray(x, np.float64)))


class Calibration:
    """Platt curves and the probable / possible tiers of one calibration JSON."""

    def __init__(self, path):
        c = json.loads(Path(path).read_text())
        self.labels = list(c["labels"])
        self.slope = np.asarray(c["class_slope"], np.float64)
        self.intercept = np.asarray(c["class_intercept"], np.float64)
        self.raw_plan = np.asarray(c["class_raw_plan_threshold"], np.float64)
        self.raw_possible = np.asarray(c["class_raw_possible_threshold"], np.float64)
        self.base_positive = np.asarray(c["class_val_positive"], np.float64)
        self.n_studies = int(c["n_calibration_studies"])

    def probability(self, logits) -> np.ndarray:
        """Calibrated probabilities of logits [..., C]."""
        return sigmoid(self.slope * np.asarray(logits, np.float64) + self.intercept)

    def base_rate(self) -> np.ndarray:
        """Training prevalence of every finding."""
        return self.base_positive / max(self.n_studies, 1)

    def tiers(self, logits):
        """(probable [..., C] bool, possible [..., C] bool) on the raw sigmoid of logits."""
        p = sigmoid(logits)
        probable = p >= self.raw_plan
        return probable, (p >= self.raw_possible) & ~probable


class RuleA:
    """Rule-A cut-offs {finding: {scan type: cut}} (``fora_per_scan_type`` of the cut-off JSON)."""

    def __init__(self, path, labels):
        table = json.loads(Path(path).read_text())["fora_per_scan_type"]
        self.cut = np.array([[table[n][s] for s in SCAN_TYPES] for n in labels], np.float32)

    def decide(self, logits, scan_types) -> np.ndarray:
        """logits [N, C] and one scan type per study -> positive decisions [N, C] (logit >= cut, float32)."""
        cols = np.array([SCAN_TYPES.index(s) for s in scan_types])
        return np.asarray(logits, np.float32) >= self.cut[:, cols].T
