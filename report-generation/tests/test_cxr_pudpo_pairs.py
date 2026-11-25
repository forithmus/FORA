"""CPU tests of the PU-DPO pair rules (a/b/c pair types, verification, length filter, weights) on toy data."""

import random

import numpy as np

from fora_report.cxr import pudpo_pairs as PP
from fora_report.cxr.pudpo_edit import clean, load_prompts, messages
from fora_report.cxr.train_pudpo import sample_pairs


def items_for(g, ref, pi, plan=0.5, possible=0.1):
    pi = np.asarray(pi)
    prob = pi >= plan
    poss = (pi >= possible) & ~prob
    C = len(pi)
    freq = np.arange(C, 0, -1) * 10.0
    return PP.study_items(set(g), set(ref), pi, prob, poss, np.ones(C, bool), freq, np.sqrt(freq), random.Random(0))


def test_pair_types():
    # class 0,1 labelled; base mentions 1 and 5; class 2 probable, 3 possible, 4/6/7 unsupported
    items = items_for(g={1, 5}, ref={0, 1}, pi=[.9, .9, .8, .2, .01, .01, .01, .01])
    by = {}
    for t, j, op, mode in items:
        by.setdefault(t, []).append((j, op, mode))
    assert by['a_insert'] == [(0, 'insert', 'plain')]
    assert by['a_keep'] == [(1, 'remove', '')]
    assert (2, 'insert', 'plain') in by['b_insert'] and (3, 'insert', 'hedged') in by['b_insert']
    assert by['c_insert'][0][0] in (4, 6, 7) and by['c_insert'][0][1] == 'insert'
    assert by['c_remove'] == [(5, 'remove', '')]
    assert 'b_keep' not in by


def test_orientation():
    assert PP.orient('a_insert', 'E', 'B') == ('E', 'B')
    assert PP.orient('c_remove', 'E', 'B') == ('E', 'B')
    for t in ('a_keep', 'b_keep', 'c_insert'):
        assert PP.orient(t, 'E', 'B') == ('B', 'E')


def test_verification_and_length_filter():
    base = 'x' * 400
    r_ins = dict(j=3, op='insert')
    assert PP.verify_edit(r_ins, base + ' y' * 50, base, {3}, set()) is None            # +100 chars <= 0.5 * 400
    assert PP.verify_edit(r_ins, base + 'y' * 201, base, {3}, set()) == 'length'
    short = 'x' * 50                                                                    # tau * max(len, 200) = 100
    assert PP.verify_edit(r_ins, short + 'y' * 100, short, {3}, set()) is None
    assert PP.verify_edit(r_ins, short + 'y' * 101, short, {3}, set()) == 'length'
    assert PP.verify_edit(r_ins, base, base, {3}, set()) == 'unchanged'
    assert PP.verify_edit(r_ins, '', base, {3}, set()) == 'no_edit'
    assert PP.verify_edit(r_ins, base + ' y', base, None, set()) == 'no_label'
    assert PP.verify_edit(r_ins, base + ' y', base, set(), set()) == 'j_not_added'
    assert PP.verify_edit(dict(j=3, op='remove'), base[:-5], base, {3}, {3}) == 'j_not_removed'
    assert PP.verify_edit(r_ins, base + ' y', base, {3, 7, 8}, set()) == 'side_effects'
    assert PP.verify_edit(r_ins, base + ' y', base, {3, 7}, set()) is None             # one other change allowed


def test_weights_and_repeats():
    assert PP.pair_weight(248, 248) == 1.0
    assert PP.pair_weight(1, 248) == 4.0 and PP.pair_weight(10 ** 6, 248) == 0.5
    rep = PP.study_repeats([('u1', 'a', 'c', 2.6), ('u1', 'b', 'c', 1.0), ('u2', 'a', 'c', 0.5)])
    assert rep == {'u1': 3, 'u2': 1}


def test_weighted_sampling_deterministic():
    rows = [(i, 'a_insert', 'f', w, 'c', 'r') for i, w in enumerate([0.5, 4.0, 1.0])]
    a = sample_pairs(rows, 2, 'k')
    assert a == sample_pairs(rows, 2, 'k') and len(a) == 2 and len({r[0] for r in a}) == 2


def test_editor_prompt():
    P = load_prompts()
    r = dict(base='Heart size normal.', op='insert', name='Pleural effusion', syn='effusion', location='left lower zone',
             mode='hedged')
    m = messages(r, P)
    assert m[0]['role'] == 'system' and m[0]['content'].startswith('You are a radiologist')
    user = m[1]['content']
    assert user.startswith('Findings section:\n<<<\nHeart size normal.\n>>>\n\nRequested change: the report must state')
    assert 'Pleural effusion (also called: effusion). The image evidence places it at: left lower zone;' in user
    assert "locations. State it with uncertainty" in user
    rm = messages(dict(base='B', op='remove', name='X', syn='', mode=''), P)[1]['content']
    assert rm.endswith('keep every other sentence word for word.') and 'remove all mention of X so that' in rm
    assert clean('```\nFindings: text\n```') == 'text'
