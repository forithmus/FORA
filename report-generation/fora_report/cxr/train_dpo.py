#!/usr/bin/env python3
"""Chest X-ray writer, offline DPO stage (paper section "Grounded report generation", X-ray DPO).

Policy and frozen reference = the SFT checkpoint. Studies = TRAIN studies with at least one synthetic pair
(``fora_report.cxr.dpo_pairs``; the chosen text is the study's own report, its sha256 is re-checked against the target).
Per visit at most K = 2 pairs (deterministic cycle over the study's pairs, ``select_pairs``). Objective per paired study:
    dpo_weight * mean_k -log sigmoid(beta * margin_k) + chosen_ce_weight * token-mean CE(chosen),   beta = 0.1
with the sequence score = sum log p of 'Findings:\\n' + text + EOS (``fsdp_dpo_study_backward``). A study without
pairs on every rank of its FSDP shard group takes a plain CE step. 600 steps; the paper model uses step 200.
"""
from __future__ import annotations

import json
import random
import sqlite3

from fora.reportgen.cxr.model import target_ids
from fora_report.cxr.preference import parse_preference_args, plain_ce_step, run_preference

OBJECTIVE = 'offline_dpo_sum_logprob'
STORE_SCHEMA = 'cxr_report_preference_index_v1'


class CXRPairStore:
    """Pair store written by ``fora_report.cxr.dpo_pairs``: pairs(pair_id, report_uid, pair_type, attribute_type,
    chosen_sha256, rejected_text)."""

    def __init__(self, path: str):
        from fora_report.ct.dpo_loss import PAIR_TYPES
        self.con = sqlite3.connect(f'file:{path}?mode=ro', uri=True, check_same_thread=False)
        self.summary = json.loads(self.con.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()[0])
        if self.summary.get('schema') != STORE_SCHEMA:
            raise ValueError(f'not an X-ray preference store: {path}')
        if set(self.summary['pairs_by_type']) - set(PAIR_TYPES):
            raise ValueError('unknown pair types in store')

    def pairs(self, uid: str):
        return self.con.execute('SELECT pair_id, pair_type, attribute_type, chosen_sha256, rejected_text FROM pairs '
                                'WHERE report_uid=? ORDER BY pair_id', (str(uid),)).fetchall()

    def uids(self) -> set:
        return {r[0] for r in self.con.execute('SELECT DISTINCT report_uid FROM pairs')}


def dpo_step(env):
    """One study: select pairs, agree the slot count in the shard group, DPO backward (or plain CE)."""
    from fora_report.ct.dpo_loss import select_pairs, sha256_text
    from fora_report.writer import fsdp_dpo as reportgen_fsdp_dpo
    a = env.args
    rejected = []
    rows = env.store.pairs(env.uid)
    epoch_of_step = env.step // max(1, env.steps_per_epoch)
    if rows and random.Random(f'{a.seed}|dpo|{epoch_of_step}|{env.uid}').random() < a.dpo_mix:
        csha = sha256_text(env.text)
        for _pid, ptype, _attr, psha, rtext in select_pairs(rows, uid=env.uid, epoch=epoch_of_step, seed=a.seed,
                                                            k=a.dpo_pairs_per_visit):
            if psha != csha:
                env.stats.misc[2] += 1
                continue
            rids = target_ids(env.tokenizer, rtext, env.max_tok + 1, env.device)
            if rids.numel() > env.max_tok or env.chosen.numel() >= env.max_tok:
                env.stats.misc[2] += 1
                continue
            rejected.append((ptype, rids))
    slots = reportgen_fsdp_dpo.agree_slots(len(rejected), env.slot_group, env.device)
    if not slots:
        return plain_ce_step(env), len(rejected)
    value, _dpo_value, _out, _paired = reportgen_fsdp_dpo.fsdp_dpo_study_backward(
        score=env.score, chosen_ids=env.chosen, rejected=rejected, slots=slots, reference=env.reference,
        ddp_context=env.sync, beta=a.dpo_beta, dpo_weight=a.dpo_weight, chosen_ce_weight=a.dpo_chosen_ce_weight,
        unpaired_ce_weight=a.dpo_unpaired_ce_weight, accumulation=1, stats=env.stats,
        recompute_tolerance=a.dpo_recompute_tolerance, mismatch_abort=a.dpo_recompute_abort)
    return value, len(rejected)


def main(argv=None):
    args = parse_preference_args(argv)
    store = CXRPairStore(args.dpo_pairs)
    paired = store.uids()

    def select_cases(cases):
        cases = [c for c in cases if c['id'] in paired]
        return cases[: args.limit_train] if args.limit_train else cases

    return run_preference(args, store=store, select_cases=select_cases, objective=OBJECTIVE, train_step=dpo_step)


if __name__ == '__main__':
    raise SystemExit(main())
