"""Report-generation scoring and judge parsing."""
import json

import numpy as np
import pytest

from evaluation.report_generation.finding_metrics import looped, score_models, summarise
from evaluation.report_generation.judge.prompts import parse, report_score


def test_summarise_hand_example():
    Y = np.array([[1, 0], [1, 0], [0, 1], [0, 0]], bool); P = np.array([[1, 0], [0, 0], [0, 1], [1, 0]], bool)
    m = summarise(P, Y, np.arange(2), np.ones(4))
    assert m['macro_recall'] == pytest.approx((0.5 + 1) / 2)
    assert m['macro_specificity'] == pytest.approx((0.5 + 1) / 2)
    assert m['macro_precision'] == pytest.approx((0.5 + 1) / 2) and m['n_classes'] == 2


def test_ge10_and_intervals():
    rng = np.random.default_rng(0); Y = rng.random((200, 5)) < np.array([.3, .3, .02, .3, .0]); P = Y ^ (rng.random((200, 5)) < .1)
    r = score_models(Y, {'a': P, 'b': rng.random((200, 5)) < .2}, None, boot=50, paired_draws=50)
    assert r['models']['a']['ge10']['n_classes'] == 3
    lo, hi = r['models']['a']['ge10']['macro_BA']['ci95']
    assert lo <= r['models']['a']['ge10']['macro_BA']['value'] <= hi
    assert all('p_holm' in c for c in r['comparisons'])


def test_loop_rule():
    assert looped('Normal heart. Normal heart. Normal heart.') and not looped('One finding here. Another one there.')


def test_judge_score_rule():
    assert report_score(3, 1) == 0.75 and report_score(0, 0) == 1.0 and report_score(0, 2) == 0.0
    v = {'matched_findings': 0, 'significant': {}, 'insignificant': {}}
    assert parse({'kind': 'single'}, json.dumps(v))['score'] == 1.0
    assert parse({'kind': 'pair'}, '```json\n{"better": "b", "reason": "x"}\n```')['better'] == 'B'
    assert parse({'kind': 'single'}, 'not json') is None
