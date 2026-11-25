"""Inference-time plan rules of the chest X-ray writer (paper section "Grounded report generation", X-ray decoding).

Applied at generation only (the writer was trained with the plain plan):
  * device rule: for a fixed set of device / hardware classes (support devices and hardware-type post-surgical classes,
    chosen from the vocabulary, malposition classes excluded) the probable threshold is lowered to the class's train
    Youden ('possible') threshold, so a device past Youden is a probable plan item;
  * high-AUROC promotion: after the plan selection, possible items whose class is a device or has internal-validation
    AUROC >= 0.90 with >= 10 validation positives (``highauc_classes.json``) are promoted to probable, at most K = 8 per
    study, highest calibrated probability first. Thresholds are unchanged (Youden still defines 'possible').
"""
from __future__ import annotations

import json
from pathlib import Path

DEVICES = frozenset({
    'Cardiac pacemaker', 'Central venous catheter (CVC), normal position', 'Central venous catheter via jugular vein',
    'Central venous catheter via umbilical vein', 'Chest tube (thoracostomy tube)', 'Endotracheal tube (ETT), normal position',
    'Implantable port / Port-a-Cath', 'Mediastinal drain', 'Nasogastric / orogastric (NG/OG) tube, normal position',
    'Nasogastric / orogastric (NG) tube, normal position', 'Prosthetic heart valve', 'Support Devices (generic umbrella)',
    'Tracheostomy tube', 'Breast implant / mammary prosthesis',
    'Median sternotomy / sternal wires (normal)', 'Cerclage wires (rib/sternal fracture fixation)',
    'Osteosynthesis material (fracture fixation hardware)', 'Spinal hardware / fusion instrumentation',
    'Surgical clips / vascular clips / suture material', 'Thoracic aortic stent graft (TEVAR)',
})
HIGHAUC_FILE = Path(__file__).resolve().parent / 'highauc_classes.json'
PROMOTE_K = 8


def high_auroc_classes(path: str | Path = HIGHAUC_FILE) -> frozenset:
    return frozenset(json.loads(Path(path).read_text())['high'])


def apply_device_rule(builder, devices=DEVICES) -> list[int]:
    """Probable threshold := possible (train Youden) threshold for the device classes. Returns the class indices."""
    idx = [i for i, label in enumerate(builder.labels) if label in devices]
    for i in idx:
        builder.class_threshold[i] = builder.class_possible_threshold[i]
    return idx


def promote(selection, labels, promotable, k: int, plan_selection_class):
    """Possible items of promotable classes -> probable (at most k, highest calibrated probability first)."""
    p = selection.probability
    cand = sorted((i for i in selection.possible if labels[i] in promotable), key=lambda i: -float(p[i]))[:k]
    if not cand:
        return selection
    cs = set(cand)
    return plan_selection_class(probability=selection.probability,
                                source_class_indices=selection.source_class_indices,
                                confident=tuple(selection.confident) + tuple(cand),
                                possible=tuple(i for i in selection.possible if i not in cs))


def inference_plan_builder_class(highauc_file: str | Path = HIGHAUC_FILE, devices=DEVICES, promote_k: int = PROMOTE_K):
    """DiagnosticPlanBuilder with the device rule (constructor) and high-AUROC promotion (plan_selection); the defaults
    are the paper rules (``highauc_classes.json``, the 20 device classes, K = 8)."""
    from fora.reportgen.plan import diagnostic_plan as DP
    device_set = frozenset(devices)

    class CXRInferencePlanBuilder(DP.DiagnosticPlanBuilder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.promote_k = int(promote_k)
            self.device_classes = apply_device_rule(self, device_set)
            self.promotable = high_auroc_classes(highauc_file) | device_set
            print(f'[plan] {len(self.device_classes)} device classes probable at Youden; '
                  f'high-AUROC promotion K={self.promote_k}', flush=True)

        def plan_selection(self, logits):
            s = super().plan_selection(logits)
            if self.mode != 'all_classes':
                return s
            return promote(s, self.labels, self.promotable, self.promote_k, DP.PlanSelection)

    return CXRInferencePlanBuilder
