"""CPU tests of the X-ray inference plan rules (device rule + high-AUROC promotion) on synthetic probabilities."""

import hashlib
import json

import torch

from fora.reportgen.cxr import plan_rules as PR
from fora.reportgen.cxr.model import make_plan_builder


class CharTokenizer:
    def apply_chat_template(self, messages, **_):
        return {"input_ids": [ord(c) for c in messages[0]["content"]]}


def logit(p):
    p = torch.tensor(p, dtype=torch.float32)
    return torch.log(p / (1 - p))


def write_calibration(path, labels, plan=0.5, possible=0.1):
    C = len(labels)
    cal = dict(labels=labels, canonical_plan_labels=labels,
               label_order_sha256=hashlib.sha256("\n".join(labels).encode()).hexdigest(),
               class_slope=[1.0] * C, class_intercept=[0.0] * C,
               class_plan_threshold=[plan] * C, class_possible_threshold=[possible] * C,
               class_val_positive=[100] * C, n_calibration_studies=1000)
    path.write_text(json.dumps(cal))


def test_highauc_file_has_classes():
    high = PR.high_auroc_classes()
    assert 'Cardiomegaly' in high and len(high) > 50


def test_device_rule_and_promotion(tmp_path):
    labels = ['Cardiac pacemaker', 'Cardiomegaly', 'Atelectasis (general)', 'Rare thing'] + \
             [f'Device-free finding {i}' for i in range(10)]
    write_calibration(tmp_path / 'c.json', labels)
    plain = make_plan_builder(tmp_path / 'c.json', labels, CharTokenizer())
    rules = make_plan_builder(tmp_path / 'c.json', labels, CharTokenizer(), PR.inference_plan_builder_class())
    p = [0.2, 0.3, 0.4, 0.3] + [0.15] * 10
    sel0 = plain.plan_selection(logit(p))
    assert sel0.confident == () and 0 in sel0.possible
    sel = rules.plan_selection(logit(p))
    assert rules.device_classes == [0]
    assert float(rules.class_threshold[0]) == float(rules.class_possible_threshold[0])
    high = PR.high_auroc_classes()
    promoted = {i for i in sel.confident}
    assert 0 in promoted                                     # device past Youden -> probable
    assert (1 in promoted) == ('Cardiomegaly' in high)
    assert 3 not in promoted and 3 in sel.possible            # not high-AUROC, stays possible
    assert set(sel.confident) | set(sel.possible) == set(sel0.confident) | set(sel0.possible)


def test_promotion_cap_and_order():
    class Sel:
        def __init__(self, probability, confident, possible):
            self.probability, self.source_class_indices = probability, torch.arange(len(probability))
            self.confident, self.possible = confident, possible

    p = torch.linspace(0.1, 0.9, 12)
    labels = [f'c{i}' for i in range(12)]
    s = PR.promote(Sel(p, (), tuple(range(12))), labels, set(labels), 8,
                   lambda **kw: Sel(kw['probability'], kw['confident'], kw['possible']))
    assert s.confident == tuple(range(11, 3, -1))           # top 8 by p, highest first
    assert s.possible == (0, 1, 2, 3)
