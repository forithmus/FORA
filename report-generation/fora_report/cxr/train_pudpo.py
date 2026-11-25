#!/usr/bin/env python3
"""Chest X-ray writer, PU-DPO stage (paper section "Grounded report generation", X-ray PU-DPO).

Pairs are edited versions of the writer's own reports (``fora_report.cxr.pudpo_pairs``): every pair carries its own
chosen and rejected text. Per visit K = 2 pairs are drawn without replacement with probability proportional to the
pair weight (rare-class balance), and studies are repeated by their `repeat` column. Objective per pair = the DPO term
(beta 0.1) + chosen CE on that pair's chosen text, averaged over the K pairs. Policy and frozen reference = the DPO
checkpoint used as the start. 600 steps; the paper model uses step 200.
"""
from __future__ import annotations

import json
import random
import sqlite3

import torch
from torch.nn import functional as F

from fora.reportgen.cxr.model import target_ids
from fora_report.cxr.preference import parse_preference_args, plain_ce_step, run_preference

OBJECTIVE = 'pudpo_per_pair_chosen_dpo_sum_logprob'
PU_TYPES = ('a_insert', 'a_keep', 'b_insert', 'b_keep', 'c_insert', 'c_remove')
STORE_SCHEMA = 'pudpo_pairs_v1'


class PUPairStore:
    """pairs(pair_id, report_uid, pair_type, finding, weight, chosen_text, rejected_text, mode, p);
    studies(report_uid, repeat)."""

    def __init__(self, path: str):
        self.con = sqlite3.connect(f'file:{path}?mode=ro', uri=True, check_same_thread=False)
        self.summary = json.loads(self.con.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()[0])
        if self.summary.get('schema') != STORE_SCHEMA:
            raise ValueError(f'not a PU-DPO pair store: {path}')
        if set(self.summary['pairs_by_type']) - set(PU_TYPES):
            raise ValueError('unknown pair types in store')

    def pairs(self, uid: str):
        return self.con.execute('SELECT pair_id, pair_type, finding, weight, chosen_text, rejected_text FROM pairs '
                                'WHERE report_uid=? ORDER BY pair_id', (str(uid),)).fetchall()

    def repeats(self) -> dict:
        return {r[0]: int(r[1]) for r in self.con.execute('SELECT report_uid, repeat FROM studies')}


def sample_pairs(rows, k: int, key: str):
    """K pairs without replacement, probability proportional to the pair weight; deterministic in key."""
    rr = random.Random(key)
    rows = list(rows)
    out = []
    while rows and len(out) < k:
        w = [max(1e-6, float(r[3])) for r in rows]
        x = rr.random() * sum(w)
        acc = 0.0
        for i, wi in enumerate(w):
            acc += wi
            if acc >= x:
                break
        out.append(rows.pop(i))
    return out


def pu_study_backward(*, score, pairs, slots, reference, ddp_context, beta, dpo_weight, chosen_ce_weight, stats,
                      dummy, recompute_tolerance=0.5, mismatch_abort=20.0):
    """FSDP DPO with a separate chosen text per pair (exact sequential-graph chain rule).

    pairs: list of (pair_type, chosen_ids, rejected_ids), K = len(pairs) <= slots. Every rank of the shard group runs
    the same schedule per slot: 2 reference no-grad + 1 policy no-grad forwards, then 2 grad passes (chosen, rejected);
    missing slots run the same passes on `dummy` with zero loss. Loss = (1/K) sum_k [dpo_weight * dpo_k +
    chosen_ce_weight * CE(chosen_k)]; the chosen pass uses the rejected score as a constant and vice versa.
    """
    from fora_report.ct.dpo_loss import dpo_terms
    K = len(pairs)
    if K > slots or slots < 1:
        raise ValueError(f'invalid schedule K={K} slots={slots}')
    with torch.no_grad():
        with reference.swapped():
            ref = [(score('reference', c)[0].detach().float(), score('reference', r)[0].detach().float())
                   for _t, c, r in pairs]
            for _ in range(slots - K):
                score('reference', dummy)
                score('reference', dummy)
        pol_r = [score('policy', r)[0].detach().float() for _t, _c, r in pairs]
        for _ in range(slots - K):
            score('policy', dummy)
    total_value = 0.0
    n_pass = 2 * slots
    j = 0
    for k in range(slots):
        if k < K:
            ptype, cids, rids = pairs[k]
            ref_c, ref_r = ref[k]
            with ddp_context(j == n_pass - 1):
                pc_live, out_c = score('train', cids)
                pc_live = pc_live.float()
                loss_k, _ = dpo_terms(pc_live, pol_r[k], ref_c, ref_r, beta)
                ce = -pc_live / int(cids.numel())
                obj = (dpo_weight * loss_k + chosen_ce_weight * ce) / K
                if not torch.isfinite(obj).item():
                    raise FloatingPointError('non-finite PU-DPO chosen objective')
                obj.backward()
            j += 1
            pc = pc_live.detach()
            total_value += float(obj.detach())
            ce_v = float(ce.detach())
            del pc_live, out_c
            with ddp_context(j == n_pass - 1):
                pr_live, _o = score('train', rids)
                pr_live = pr_live.float()
                error = abs(float(pr_live.detach()) - float(pol_r[k]))
                if error > mismatch_abort:
                    raise RuntimeError(f'rejected policy score changed on recompute by {error:.3f} nats')
                if error > recompute_tolerance:
                    stats.misc[3] += 1
                loss_k2, _ = dpo_terms(pc, pr_live, ref_c, ref_r, beta)
                (dpo_weight * loss_k2 / K).backward()
            j += 1
            margin = float((pc - pol_r[k]) - (ref_c - ref_r))
            stats.add(ptype, dpo_loss=float(-F.logsigmoid(torch.tensor(beta * margin))), correct=float(margin > 0),
                      margin=margin, policy_margin=float(pc - pol_r[k]), reference_margin=float(ref_c - ref_r),
                      chosen_ce=ce_v, recompute_abs_error=error)
            del pr_live, _o
        else:
            for _ in range(2):
                with ddp_context(j == n_pass - 1):
                    d_live, _o = score('train', dummy)
                    (d_live.float() * 0.0).backward()
                j += 1
                del d_live, _o
    stats.misc[0 if K else 1] += 1
    return total_value


def pudpo_step(env):
    from fora_report.writer import fsdp_dpo as reportgen_fsdp_dpo
    a = env.args
    pairs = []
    key = f'{a.seed}|pudpo|{env.step}|{env.rank}|{env.uid}'
    for _pid, ptype, _f, _w, ctext, rtext in sample_pairs(env.store.pairs(env.uid), a.dpo_pairs_per_visit, key):
        cids = target_ids(env.tokenizer, ctext, env.max_tok + 1, env.device)
        rids = target_ids(env.tokenizer, rtext, env.max_tok + 1, env.device)
        if rids.numel() > env.max_tok or cids.numel() > env.max_tok:
            env.stats.misc[2] += 1
            continue
        pairs.append((ptype, cids, rids))
    slots = reportgen_fsdp_dpo.agree_slots(len(pairs), env.slot_group, env.device)
    if not slots:
        return plain_ce_step(env), len(pairs)
    value = pu_study_backward(
        score=env.score, pairs=pairs, slots=slots, reference=env.reference, ddp_context=env.sync, beta=a.dpo_beta,
        dpo_weight=a.dpo_weight, chosen_ce_weight=a.dpo_chosen_ce_weight, stats=env.stats,
        dummy=env.chosen[: max(1, min(16, int(env.chosen.numel())))],
        recompute_tolerance=a.dpo_recompute_tolerance, mismatch_abort=a.dpo_recompute_abort)
    return value, len(pairs)


def main(argv=None):
    args = parse_preference_args(argv)
    store = PUPairStore(args.dpo_pairs)
    repeats = store.repeats()

    def select_cases(cases):
        base = [c for c in cases if c['id'] in repeats]
        if args.limit_train:
            base = base[: args.limit_train]
        return [c for c in base for _ in range(max(1, repeats[c['id']]))]

    return run_preference(args, store=store, select_cases=select_cases, objective=OBJECTIVE, train_step=pudpo_step,
                          stats_types=PU_TYPES)


if __name__ == '__main__':
    raise SystemExit(main())
