"""Model-side CT protocol taxonomy: the 29 protocol ids (28 classes + "other") and the lung-CT protocol set.

The ordering below is the study-type id order used by the models and the inference bundle. It is NOT the numbering
n = 1..29 of the protocol classifier (data-preprocessing/protocol_classification/protocol_classes.json); map between
the two by title. "CT Liver - Four-Phase" is merged into "CT Abdomen and Pelvis - Multiphase" by the classifier
post-processing and has no id here. Unknown titles map to UNK_STUDY_TYPE = 29.
"""
from __future__ import annotations

import re

PROTOCOL_CLASSES = [
    "CT Chest - Non-Contrast (incl. HRCT / Low-Dose Screening)",
    "CT Head - Non-Contrast",
    "CT Abdomen and Pelvis - IV Contrast",
    "CT Sinus / Facial Bones / Orbits",
    "CT Chest - IV Contrast",
    "CT Abdomen - Renal Stone Protocol",
    "CT Abdomen and Pelvis - Multiphase",
    "CT Abdomen and Pelvis - Non-Contrast",
    "Coronary CTA / Calcium Score",
    "CT Extremity",
    "CTA Chest / PE / Aorta",
    "CT Lumbar Spine",
    "other",
    "CT Chest, Abdomen and Pelvis - IV Contrast",
    "CTA Abdomen",
    "CT Temporal Bones",
    "CT Cervical Spine",
    "CT Urography",
    "CTA Head and Neck",
    "CT Thoracic Spine",
    "CTA Extremity Runoff",
    "CT Chest, Abdomen and Pelvis - Non-Contrast",
    "CT Neck - IV Contrast",
    "CT Chest, Abdomen and Pelvis - Multiphase",
    "CT Neck - Non-Contrast",
    "CTA Neck",
    "CTA Head",
    "CT Head - IV Contrast",
    "Dental Cone-Beam CT",
]


def norm_class(s: str) -> str:
    """Normalize a protocol title for matching (dash / space / case tolerant)."""
    return re.sub(r"\s+", " ", str(s).lower().replace("–", "-").replace("—", "-")).strip()


_CLASS_TO_IDX = {norm_class(c): i for i, c in enumerate(PROTOCOL_CLASSES)}
UNK_STUDY_TYPE = len(PROTOCOL_CLASSES)          # 29


def study_type_index(title: str) -> int:
    return _CLASS_TO_IDX.get(norm_class(title), UNK_STUDY_TYPE)


# Protocols whose field of view contains lung parenchyma (chest classes, coronary CTA, thoracic spine, the abdomen and
# pelvis family and CTA abdomen for the lung bases). Used only by the inference lung-MIP rule.
LUNG_CT_CLASSES = {norm_class(c) for c in (
    "CT Chest - Non-Contrast (incl. HRCT / Low-Dose Screening)",
    "CT Chest - IV Contrast",
    "CTA Chest / PE / Aorta",
    "CT Chest, Abdomen and Pelvis - IV Contrast",
    "CT Chest, Abdomen and Pelvis - Non-Contrast",
    "CT Chest, Abdomen and Pelvis - Multiphase",
    "Coronary CTA / Calcium Score",
    "CT Thoracic Spine",
    "CT Abdomen and Pelvis - IV Contrast",
    "CT Abdomen and Pelvis - Multiphase",
    "CT Abdomen and Pelvis - Non-Contrast",
    "CTA Abdomen",
)}


def is_lung_study(study_type_ids) -> bool:
    """True if any study-type id of a study is a lung-CT protocol (the inference bundle's gate)."""
    return any(i < len(PROTOCOL_CLASSES) and norm_class(PROTOCOL_CLASSES[i]) in LUNG_CT_CLASSES for i in study_type_ids)
