"""Conservative anatomy-scoped normality extraction for X-ray reports (Methods, X-ray labels).

This module does not infer absence from silence.  It recognizes an explicit
normal statement, assigns that statement to a small anatomical scope, and then
expands the scope through an auditable label allow-list.  Every expansion keeps
the source sentence and rule identifier.

The rules are deliberately high precision and constitute weak/biased-negative
evidence, not clinical ground truth.  Direct label assertions, uncertainty, and
direct denials are resolved by the index builder before scoped evidence is used.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ScopeRule:
    rule_id: str
    scope: str
    pattern: re.Pattern[str]
    weight: float
    qualifier: str = "general"


@dataclass(frozen=True)
class ScopedEvidence:
    label_index: int
    label: str
    scope: str
    quote: str
    rule_id: str
    weight: float
    qualifier: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _rx(expression: str) -> re.Pattern[str]:
    return re.compile(expression, re.IGNORECASE)


RULES: tuple[ScopeRule, ...] = (
    ScopeRule(
        "visible_skeleton_normal",
        "visible_skeleton",
        _rx(
            r"\b(?:bones?|bony structures?|osseous structures?|visuali[sz]ed "
            r"osseous structures?|bony thorax|skeletal structures?)\b.{0,100}"
            r"\b(?:normal|unremarkable|intact|within normal limits|natural)\b"
        ),
        0.50,
    ),
    ScopeRule(
        "visible_skeleton_no_abnormality",
        "visible_skeleton",
        _rx(
            r"\bno (?:significant )?(?:osseous|bony|skeletal) "
            r"(?:abnormalit(?:y|ies)|finding|lesion)s?\b"
        ),
        0.50,
    ),
    ScopeRule(
        "visible_skeleton_no_acute_abnormality",
        "acute_visible_skeleton",
        _rx(
            r"\bno acute (?:osseous|bony|skeletal) "
            r"(?:abnormalit(?:y|ies)|finding|injur(?:y|ies))\b"
        ),
        0.60,
        "acute_only",
    ),
    ScopeRule(
        "chest_wall_soft_tissue_normal",
        "chest_wall_soft_tissue",
        _rx(
            r"\b(?:chest wall |visuali[sz]ed )?soft tissues?\b.{0,80}"
            r"\b(?:normal|unremarkable|within normal limits)\b"
        ),
        0.40,
    ),
    ScopeRule(
        "heart_size_normal",
        "heart_size",
        _rx(
            r"\b(?:heart size|cardiac silhouette|cardiomediastinal silhouette|"
            r"cardiothoracic (?:ratio|index))\b.{0,80}\b(?:normal|within normal "
            r"limits|not enlarged|normal in size)\b"
        ),
        0.50,
    ),
    ScopeRule(
        "mediastinum_normal",
        "mediastinum",
        _rx(
            r"\bmediastin(?:um|al (?:contours?|silhouette))\b.{0,80}"
            r"\b(?:normal|unremarkable|not widened|within normal limits)\b"
        ),
        0.40,
    ),
    ScopeRule(
        "hila_normal",
        "hila",
        _rx(
            r"\b(?:bilateral )?(?:hila|hilar structures?|hilar contours?)\b"
            r".{0,80}\b(?:normal|unremarkable|not enlarged|within normal limits)\b"
        ),
        0.50,
    ),
    ScopeRule(
        "diaphragm_normal",
        "diaphragm",
        _rx(
            r"\b(?:hemi)?diaphragm(?:atic)?(?: contours?)?\b.{0,80}"
            r"\b(?:normal|regular|unremarkable|intact)\b"
        ),
        0.40,
    ),
    ScopeRule(
        "costophrenic_recesses_clear",
        "pleural_fluid",
        _rx(
            r"\b(?:costophrenic (?:angles?|recesses?)|(?:bilateral )?sinuses?)\b"
            r".{0,80}\b(?:clear|sharp|open|free|normal|unremarkable)\b"
        ),
        0.60,
    ),
    ScopeRule(
        "pleural_spaces_clear",
        "pleural_space",
        _rx(
            r"\bpleural (?:spaces?|cavities)\b.{0,80}"
            r"\b(?:clear|free|normal|unremarkable)\b"
        ),
        0.50,
    ),
    ScopeRule(
        "lungs_no_active_lesion",
        "active_lung_lesion",
        _rx(
            r"(?:\b(?:lungs?|lung fields?|lung parenchyma|pulmonary parenchyma)\b"
            r".{0,120}\bno (?:active|focal) (?:specific )?"
            r"(?:lesions?|abnormalit(?:y|ies)|opacit(?:y|ies)|infiltrates?)\b|"
            r"\bno (?:active|focal) (?:specific )?"
            r"(?:lesions?|abnormalit(?:y|ies)|opacit(?:y|ies)|infiltrates?)\b"
            r".{0,120}\b(?:lungs?|lung fields?|lung parenchyma|pulmonary parenchyma)\b)"
        ),
        0.25,
        "active_or_focal_only",
    ),
)


# These names are checked against the immutable 311-label order when the
# artifact is built.  Explicit allow-lists prevent a broad sentence such as
# "no acute cardiopulmonary abnormality" from silently denying unrelated or
# chronic findings.
SCOPE_LABEL_NAMES: dict[str, frozenset[str]] = {
    "visible_skeleton": frozenset(
        {
            "Bone island (enostosis)",
            "Cervical rib",
            "Costal cartilage calcification / costochondral junction hypertrophy",
            "Pectus carinatum",
            "Pectus excavatum",
            "Rib companion shadow (normal variant)",
            "Thoracic cage deformation (generic/other)",
            "Bone metastasis",
            "Clavicle fracture",
            "DISH (diffuse idiopathic skeletal hyperostosis)",
            "Flail chest (segmental rib fractures)",
            "Glenohumeral (shoulder) osteoarthritis",
            "Kyphosis (hyperkyphosis)",
            "Lytic bone lesion",
            "Osteopenia / osteoporosis",
            "Proximal humeral fracture",
            "Rib fracture, acute",
            "Rib fracture, healed (with callus)",
            "Rib notching, inferior",
            "Rib notching, superior",
            "Rotator cuff arthropathy / subacromial space narrowing",
            "Scapular fracture",
            "Sclerotic (blastic) bone lesion",
            "Scoliosis",
            "Shoulder (glenohumeral) dislocation",
            "Sternal fracture",
            "Sternoclavicular joint degenerative change / hypertrophy",
            "Vertebral body wedging / old compression deformity",
            "Vertebral compression fracture (acute)",
            "Vertebral degenerative changes / spondylosis",
            "Rib splaying / horizontal ribs / widened intercostal spaces",
            "Cerclage wires (rib/sternal fracture fixation)",
            "Osteosynthesis material (fracture fixation hardware)",
            "Shoulder prosthesis / arthroplasty",
            "Spinal hardware / fusion instrumentation",
            "Sternal dehiscence / sternal wire fracture",
            "Thoracoplasty / rib resection (post-surgical)",
            "Clavicle fracture, healed (with callus)",
        }
    ),
    "acute_visible_skeleton": frozenset(
        {
            "Clavicle fracture",
            "Flail chest (segmental rib fractures)",
            "Proximal humeral fracture",
            "Rib fracture, acute",
            "Scapular fracture",
            "Shoulder (glenohumeral) dislocation",
            "Sternal fracture",
            "Vertebral compression fracture (acute)",
        }
    ),
    "chest_wall_soft_tissue": frozenset(
        {
            "Soft tissue gas (deep, non-emphysematous)",
            "Soft tissue mass / chest wall mass",
            "Subcutaneous emphysema",
        }
    ),
    "heart_size": frozenset(
        {
            "Cardiomegaly",
            "Enlarged cardiomediastinum",
            "Left atrial (LA) enlargement",
            "Left ventricular (LV) enlargement",
            "Right atrial (RA) enlargement",
            "Right ventricular (RV) enlargement",
        }
    ),
    "mediastinum": frozenset(
        {
            "Anterior mediastinal mass",
            "Mediastinal lipomatosis",
            "Mediastinal lymphadenopathy",
            "Mediastinal shift",
            "Mediastinal widening",
            "Middle mediastinal mass",
            "Pneumomediastinum",
            "Posterior mediastinal mass",
            "Superior mediastinal mass / enlargement",
        }
    ),
    "hila": frozenset(
        {
            "Hilar congestion",
            "Hilar enlargement",
            "Hilar lymphadenopathy",
        }
    ),
    "diaphragm": frozenset(
        {
            "Congenital diaphragmatic hernia, Bochdalek type",
            "Congenital diaphragmatic hernia, Morgagni type",
            "Diaphragmatic eventration",
            "Diaphragmatic paralysis",
            "Elevated hemidiaphragm",
            "Traumatic diaphragmatic hernia / rupture",
        }
    ),
    "pleural_fluid": frozenset(
        {
            "Blunting of the costophrenic angle",
            "Chylothorax",
            "Empyema (pleural)",
            "Fissural fluid / interlobar pseudotumor",
            "Hemothorax",
            "Hydropneumothorax",
            "Hydrothorax (transudative, e.g. hepatic hydrothorax)",
            "Large / massive pleural effusion",
            "Loculated pleural effusion",
            "Malignant pleural effusion",
            "Meniscus sign (Damoiseau-Ellis line)",
            "Parapneumonic effusion",
            "Pleural effusion (simple/free-flowing)",
            "Small pleural effusion",
            "Subpulmonic (infrapulmonary) pleural effusion",
            "Pleural effusion",
        }
    ),
    "pleural_space": frozenset(
        {
            "Apical (small) pneumothorax",
            "Deep sulcus sign (supine pneumothorax)",
            "Hydropneumothorax",
            "Loculated pneumothorax",
            "Pneumothorax (simple/spontaneous)",
            "Pneumothorax ex vacuo",
            "Tension pneumothorax",
        }
    ),
    "active_lung_lesion": frozenset(
        {
            "Consolidation",
            "Ground-glass opacity (GGO)",
            "Lung Opacity (airspace opacity, NOS)",
            "Pneumonia",
        }
    ),
}


def validate_scope_labels(labels: Sequence[str]) -> None:
    available = set(labels)
    missing = {
        name
        for names in SCOPE_LABEL_NAMES.values()
        for name in names
        if name not in available
    }
    if missing:
        raise ValueError(f"Scoped-normality labels absent from vocabulary: {sorted(missing)}")


def extract_scoped_evidence(
    sentences: Iterable[str], labels: Sequence[str]
) -> list[ScopedEvidence]:
    """Return auditable scope-expanded negative candidates.

    Conflict resolution against asserted/uncertain/direct-denied states is the
    responsibility of the index builder because those states come from a
    separate extraction artifact.
    """
    validate_scope_labels(labels)
    label_to_index = {label: index for index, label in enumerate(labels)}
    evidence: list[ScopedEvidence] = []
    seen: set[tuple[int, str, str]] = set()
    for raw_sentence in sentences:
        sentence = " ".join(str(raw_sentence).split())
        if not sentence:
            continue
        for rule in RULES:
            if not rule.pattern.search(sentence):
                continue
            for label in sorted(SCOPE_LABEL_NAMES[rule.scope]):
                index = label_to_index[label]
                key = (index, rule.rule_id, sentence)
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(
                    ScopedEvidence(
                        label_index=index,
                        label=label,
                        scope=rule.scope,
                        quote=sentence,
                        rule_id=rule.rule_id,
                        weight=rule.weight,
                        qualifier=rule.qualifier,
                    )
                )
    return evidence


def rule_contract() -> dict[str, object]:
    return {
        "policy": "explicit normal scope only; silence never expanded",
        "conflict_priority": [
            "asserted",
            "uncertain",
            "direct_denied",
            "scoped_denied",
            "unmentioned",
        ],
        "rules": [
            {
                "rule_id": rule.rule_id,
                "scope": rule.scope,
                "weight": rule.weight,
                "qualifier": rule.qualifier,
                "pattern": rule.pattern.pattern,
                "labels": sorted(SCOPE_LABEL_NAMES[rule.scope]),
            }
            for rule in RULES
        ],
    }
