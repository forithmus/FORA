"""Plan-based CT DPO pairs on TRAIN reports (paper: "Preference optimization", final plan tiers).

Reuses the base builder's sentence/quote parsing and screens (``base.py``, ``text.py``) and the findings sqlite
of ``base findings``. chosen = the real report (unprefixed Findings; the trainer prepends "Findings:\\n").
Rejected types (all synthetic, relative to the documented report, not image truth):

  insertion_plan_fp  a frozen-MIL plan item (confident p >= class_plan_threshold, or possible p >=
                     class_possible_threshold with the possible tier capped at --possible_cap items ranked by p,
                     exactly as the writer's plan) that the report does not assert or hedge is asserted by a donor
                     sentence of that class from another TRAIN patient (sentences with a measurement preferred),
                     inserted after the last sentence of the class's organ (unmentioned) or replacing a unary
                     whole-sentence denial of that class (denied).
  insertion_random   the same insertion for a class absent from the report, drawn from the donor pool of organs the
                     report describes, regardless of MIL.
  normalised         1..all qualified abnormal sentences of one organ replaced by one normal-template sentence of
                     that organ (frequent train sentence chosen by co-occurrence with the report's own sentences).
  multi_omission     2-4 qualified abnormal sentences deleted (base omission filters; dependency of the next
                     surviving sentence re-checked for the whole deletion set).

Incidental / degenerative classes (``configs/pairs_insertion_exclusions.json``, 43 classes) are excluded from both
insertion types and screened from donor sentences. ``insertion_plan_fp`` is capped per class at
``plan_class_cap_frac`` of all plan pairs (fixed-point iteration, then a strict trim of the lowest-p excess).

  python -m fora_report.pairs.plan_pairs --index INDEX --findings findings.sqlite --dense dense_predictions.npz \
      --calibration calibration_train_platt.json --split split.json --heads heads.json --hierarchy hierarchy.json \
      --base_store base_all.sqlite --out_dir OUT
writes OUT/plan_pairs.sqlite (with provenance), OUT/dpo_pairs.sqlite (= base store + new rows, the DPO pool),
OUT/plan_pairs_examples.jsonl and OUT/plan_pairs_summary.json.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np

from .base import analyse_study, load_studies, make_pair, omission_candidates, unary_sentences
from .lexicon import (ANAT_WORDS, GENERIC, NORMAL_RE, anatomy_terms, delete_rows, hk, insert_after, jaccard,
                      label_stems, organs_of, replace_span, spell, words)
from .merge import create_store, merge_stores
from .text import (ADDITIVE_REFERENCE, DONOR_EXCLUDE, FINITE_VERB, HEDGE_RE, NEXT_DEPENDENCY, NONCURRENT, SIZE_RE,
                   TARGET_PREFIX, jdump)

NEW_TYPES = ("insertion_plan_fp", "insertion_random", "normalised", "multi_omission")
POLICY = "ct_plan_insertion_normalisation_multiomission"
DEFAULT_EXCLUSIONS = Path(__file__).resolve().parents[2] / "configs" / "pairs_insertion_exclusions.json"


def plan_candidates(p: np.ndarray, thr_plan: np.ndarray, thr_pos: np.ndarray, possible_cap: int) -> np.ndarray:
    """Plan items of one study: confident (p >= plan threshold) plus the possible tier (p >= possible threshold),
    the possible tier capped at ``possible_cap`` items ranked by p (ties by class id); 0 = no cap."""
    cand = np.nonzero(p >= thr_pos)[0]
    if possible_cap > 0:
        conf_ = p >= thr_plan
        poss_ = [c for c in cand.tolist() if not conf_[c]]
        if len(poss_) > possible_cap:
            poss_ = sorted(poss_, key=lambda c: (-float(p[c]), int(c)))[:possible_cap]
        cand = np.array(sorted(set(np.nonzero(conf_)[0].tolist()) | set(poss_)), dtype=np.int64)
    return cand


def build(args):
    began = time.time()
    out = Path(args.out_dir)
    log = lambda m: print(f"[plan_pairs {time.time()-began:6.0f}s] {m}", flush=True)  # noqa: E731

    # ---------------- load ----------------
    train, labels = load_studies(args.index, args.findings, "train")
    if args.limit:
        train = train[: args.limit]
    log(f"train studies={len(train)} labels={len(labels)}")
    cal = json.loads(Path(args.calibration).read_text())
    assert cal["labels"] == labels, "calibration label order != index label order"
    slope = np.asarray(cal["class_slope"], np.float32)
    icpt = np.asarray(cal["class_intercept"], np.float32)
    thr_plan = np.asarray(cal["class_plan_threshold"], np.float32)
    thr_pos = np.asarray(cal["class_possible_threshold"], np.float32)
    z = np.load(args.dense, mmap_mode="r")
    assert list(z["label"]) == labels, "dense label order differs"
    uid_row = {u: i for i, u in enumerate(z["report_uid"].tolist())}
    fora_ids = z["fora_id"]
    split_map = json.loads(Path(args.split).read_text())
    issues = Counter()
    keep = []
    for s in train:
        r = uid_row.get(s["report_uid"])
        if r is None:
            issues["no_dense_mil_prediction"] += 1
            continue
        fid = str(fora_ids[r])
        if split_map.get(fid) != "train":
            issues["not_train_in_split"] += 1
            continue
        s["row"] = r
        s["fora_id"] = fid
        keep.append(s)
    train = keep
    log(f"train studies with MIL + train split: {len(train)} issues={dict(issues)}")
    logits = np.asarray(z["logit"])   # (N,1028) float32, ~1.4 GB
    heads = json.loads(Path(args.heads).read_text())
    name_to_id = {n: i for i, n in enumerate(labels)}
    related = defaultdict(set)
    for rec in heads.get("attached", []):
        a, b = name_to_id.get(rec["name"]), name_to_id.get(rec.get("parent"))
        if a is not None and b is not None:
            related[a].add(b); related[b].add(a)
    hier = json.loads(Path(args.hierarchy).read_text())["nodes"]
    exclusions = json.loads(Path(args.exclusions).read_text())
    excl_names = set(exclusions["exact"])
    for n_, v_ in hier.items():
        if set(v_.get("ancestors", [])) & set(exclusions["with_descendants"]):
            excl_names.add(n_)
    excl_names |= set(exclusions["with_descendants"])
    excluded_plan = {name_to_id[n_] for n_ in excl_names if n_ in name_to_id}
    excluded_plan_names = sorted(labels[c] for c in excluded_plan)
    log(f"insertion_plan_fp excluded classes ({len(excluded_plan)}): {excluded_plan_names}")
    normal_classes = {i for i, n in enumerate(labels) if "normal" in n.casefold() and "abnormal" not in n.casefold()}

    # ---------------- pass 1: per-study analysis, organs, donors, templates, co-occurrence ----------------
    per = {}
    type_n, type_class = Counter(), Counter()
    stems = [label_stems(l) for l in labels]
    class_organ_votes = defaultdict(Counter)
    pos_count = Counter()
    pair_count = Counter()
    sent_reports = Counter()          # sentence text -> number of train reports containing it
    for n, s in enumerate(train):
        sentences, rows, records, local = analyse_study(s, labels)
        for row in rows:
            row["organs"] = organs_of(row["text"])
            for r in row["inside"]:
                if r["state"] in ("asserted", "uncertain"):
                    for o in row["organs"]:
                        class_organ_votes[r["class_id"]][o] += 1
        oms = omission_candidates(s, rows, Counter())
        per[s["report_uid"]] = (rows, records, oms)
        pos = sorted(s["asserted"] | s["uncertain"])
        pos_count.update(pos)
        for t in s["study_type_ids"]:
            type_n[t] += 1
            for c in pos:
                type_class[(t, c)] += 1
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                pair_count[(pos[i], pos[j])] += 1
        for t in {row["text"] for row in rows}:
            sent_reports[t] += 1
        if (n + 1) % 50000 == 0:
            log(f"pass1 {n+1}/{len(train)}")
    class_organ = {}
    for c, votes in class_organ_votes.items():
        organ, k = votes.most_common(1)[0]
        class_organ[c] = organ
    log(f"class organs assigned: {len(class_organ)} / {len(labels)}")

    def plausible_for_study(c, study):
        return any(type_class[(t, c)] >= 10 and type_class[(t, c)] / max(1, type_n[t]) >= 0.002
                   for t in study["study_type_ids"])

    def already_described(c, rows, donor_text=None):
        """Any report sentence names the class by a label stem (asserted, hedged or negated: the class is then
        documented or denied in words even if the extractor did not map it) or resembles the donor."""
        st = stems[c]
        dw = words(donor_text) if donor_text else None
        for row in rows:
            if st:
                low = spell(row["text"])
                if any(re.search(r"\b" + re.escape(x), low) for x in st):
                    return True
            if dw is not None and jaccard(dw, words(row["text"])) >= 0.5:
                return True
        return False

    def cooc_related(c, positives):
        if c in normal_classes:
            return True
        for d in positives:
            if d == c or d in related[c]:
                return True
            k = pair_count[(min(c, d), max(c, d))]
            limit = args.cooc_exclude_same_organ if class_organ.get(c) == class_organ.get(d) else args.cooc_exclude
            if k and (k / max(1, pos_count[d]) >= limit or k / max(1, pos_count[c]) >= limit):
                return True
        return False

    # donors: one asserted class touched, self-contained, organ of the class mentioned
    donors = defaultdict(list)    # class -> [(uid, patient, text, has_size)]
    donor_reasons = Counter()
    for s in train:
        rows, records, _ = per[s["report_uid"]]
        for row in rows:
            ins = row["inside"]
            if not ins:
                continue
            touched = {r["class_id"] for r in ins + row["covering"]}
            states = {r["state"] for r in ins + row["covering"]}
            if len(touched) != 1 or states != {"asserted"}:
                continue
            c = next(iter(touched))
            t = row["text"]
            reason = None
            if not t.endswith(".") or not (4 <= len(t.split()) <= 40) or not FINITE_VERB.search(t):
                reason = "not_complete_sentence"
            elif NEXT_DEPENDENCY.search(t) or ADDITIVE_REFERENCE.search(t):
                reason = "anaphoric"
            elif NONCURRENT.search(t):
                reason = "noncurrent"
            elif DONOR_EXCLUDE.search(t) or HEDGE_RE.search(t):
                reason = "interpretive_or_hedged"
            elif ";" in t or ":" in t or "\n" in t:
                reason = "multiclause_punctuation"
            elif class_organ.get(c) not in row["organs"]:
                reason = "class_organ_not_in_sentence"
            elif len(row["organs"]) > 2:
                reason = "too_many_organs"
            elif s["findings"].count(t) != 1:
                reason = "repeated_in_source"
            if reason:
                donor_reasons[reason] += 1
                continue
            donors[c].append((s["report_uid"], s["patient_id"], t, bool(SIZE_RE.search(t))))
    for c in donors:
        donors[c].sort(key=lambda d: hk("donor", args.seed, d[0], d[2]))
    donor_sizes = sum(1 for v in donors.values() for d in v if d[3])
    log(f"donor classes={len(donors)} donor sentences={sum(map(len, donors.values()))} with_size={donor_sizes}")
    # donors by organ for insertion_random (natural class frequency)
    donors_by_organ = defaultdict(list)
    for c, opts in donors.items():
        for k in range(len(opts)):
            donors_by_organ[class_organ[c]].append((c, k))
    for o in donors_by_organ:
        donors_by_organ[o].sort(key=lambda x: hk("dorg", args.seed, x))

    # normal templates per organ
    tmpl_by_organ = defaultdict(list)
    for t, k in sent_reports.items():
        if k < args.template_min_reports:
            continue
        os_ = organs_of(t)
        if len(os_) != 1 or not t.endswith(".") or len(t.split()) < 4:
            continue
        if not NORMAL_RE.search(t):
            continue
        if NONCURRENT.search(t) or HEDGE_RE.search(t) or NEXT_DEPENDENCY.search(t) or SIZE_RE.search(t):
            continue
        tmpl_by_organ[next(iter(os_))].append((k, t))
    template_set = {}
    # remove templates that ever carry a positive quote (i.e. not purely normal)
    tainted = set()
    for s in train:
        rows, _, _ = per[s["report_uid"]]
        for row in rows:
            if any(r["state"] in ("asserted", "uncertain") for r in row["inside"] + row["covering"]):
                tainted.add(row["text"])
    for o, lst in tmpl_by_organ.items():
        lst = [x for x in sorted(lst, reverse=True) if x[1] not in tainted][: args.templates_per_organ]
        tmpl_by_organ[o] = lst
        for k, t in lst:
            template_set[t] = o
    frequent = {t for t, k in sent_reports.items() if k >= args.template_min_reports}
    cooc = defaultdict(Counter)       # template -> frequent sentence -> reports
    for s in train:
        rows, _, _ = per[s["report_uid"]]
        texts = {row["text"] for row in rows}
        tp = [t for t in texts if t in template_set]
        fr = [t for t in texts if t in frequent]
        for t in tp:
            c = cooc[t]
            for f in fr:
                if f != t:
                    c[f] += 1
    log(f"templates: organs={len(tmpl_by_organ)} total={len(template_set)} frequent_sentences={len(frequent)}")

    def pick_template(organ, rows, target, replaced):
        anat = set().union(*[anatomy_terms(r["text"]) for r in replaced])
        rw = set().union(*[words(r["text"]) for r in replaced])
        cands = tmpl_by_organ.get(organ) or []
        present = [row["text"] for row in rows if row["text"] in frequent]
        report_words = [words(row["text"]) for row in rows]
        best, best_score = None, -1.0
        for k, t in cands:
            if t in target:
                continue
            wt = words(t)
            if any(jaccard(wt, w) >= 0.5 for w in report_words):
                continue
            if not (anatomy_terms(t) & anat):
                continue
            c = cooc.get(t, {})
            style = sum(c.get(f, 0) / k for f in present) / max(1, len(present))
            score = style + 2.0 * jaccard(wt, rw) + 1e-7 * k
            if score > best_score:
                best, best_score = t, score
        return best, best_score

    # Donor screen for insertion_plan_fp: a donor sentence of a kept class must not describe an excluded
    # (incidental/degenerative) finding, e.g. "Hernia" donors that say "hiatal hernia", CAD donors with coronary
    # calcification.  Pattern = every content word of the excluded label (suffix-tolerant) + its organ in the sentence;
    # skipped when the kept class's own label contains the same words (e.g. Colonic diverticulitis vs Diverticulum).
    def label_word_res(label):
        out = []
        for w in re.findall(r"[a-z]+", spell(label)):
            if len(w) < 4 or w in GENERIC or w in ANAT_WORDS:
                continue
            out.append((w, re.compile(r"\b(?:fibro)?" + re.escape(w[:max(4, min(len(w) - 2, 6))]) + r"\w*")))
        return out
    excl_patterns = []
    for e in sorted(excluded_plan):
        res = label_word_res(labels[e])
        if res:
            excl_patterns.append((e, [r for _w, r in res], class_organ.get(e), {w for w, _r in res}))
    donor_screen_cache = {}

    def plan_donor_ok(c, text):
        key = (c, text)
        if key not in donor_screen_cache:
            own = spell(labels[c])
            own_words = {w for w, _r in label_word_res(labels[c])}
            low = spell(text)
            org = organs_of(text)
            bad = False
            for e, res, organ, ewords in excl_patterns:
                # exempt only a MORE SPECIFIC kept class with its own word (Colonic diverticulitis vs Diverticulum);
                # a generic parent (Cyst vs Renal cyst, Hernia vs Hiatal hernia) is screened
                if all(r.search(own) for r in res) and (own_words - ewords):
                    continue
                if all(r.search(low) for r in res) and (organ is None or organ in org):
                    bad = True
                    break
            donor_screen_cache[key] = not bad
        return donor_screen_cache[key]

    # ---------------- pass 2: pairs (re-runnable for the per-class plan cap) ----------------
    base_issues = Counter(issues)
    def run_pass2(plan_cap):
        issues = Counter(base_issues)
        plan_class = Counter()
        used = Counter()
        pointer = Counter()
        type_counts, covered = Counter(), Counter()
        tier_counts, state_counts, size_counts = Counter(), Counter(), Counter()
        norm_counts, multi_k = Counter(), Counter()
        all_pairs = []

        def take_donor(c, patient, target, rows, screen=False):
            opts = donors.get(c) or []
            if not opts:
                return None
            sized = [i for i in range(min(len(opts), 4096)) if opts[(pointer[c] + i) % len(opts)][3]]
            order = sized[:256] + list(range(min(len(opts), 256)))
            for step in order:
                d = opts[(pointer[c] + step) % len(opts)]
                cap = sent_reports[d[2]] + 2
                if screen and not plan_donor_ok(c, d[2]):
                    continue
                if (d[1] != patient and used[d[2]] < cap and d[2] not in target
                        and not any(jaccard(words(d[2]), words(r["text"])) >= 0.5 for r in rows)):
                    used[d[2]] += 1
                    pointer[c] = (pointer[c] + step + 1) % len(opts)
                    return d
            return None

        def anchor_row(rows, organ):
            hits = [k for k, row in enumerate(rows) if organ in row["organs"]]
            for k in reversed(hits):
                if k + 1 >= len(rows) or not NEXT_DEPENDENCY.search(rows[k + 1]["text"]):
                    return rows[k]
            return None

        for n, s in enumerate(train):
            uid, target = s["report_uid"], s["findings"]
            rows, records, oms = per[uid]
            local = []
            if len(target) > args.max_chosen_chars:
                issues["chosen_too_long_for_max_report_tokens"] += 1
                covered["ce_only"] += 1
                continue
            if s["verdict"] != "pass":
                issues["extraction_verdict_not_pass"] += 1
                covered["ce_only"] += 1
                continue
            positives = s["asserted"] | s["uncertain"]
            report_organs = set().union(*[row["organs"] for row in rows]) if rows else set()
            rng = random.Random(f"{args.seed}|{uid}")
            unary_denials = {}
            for row, cid, reason in unary_sentences(rows, records, "denied"):
                if reason is None and target.count(row["text"]) == 1 and cid not in positives:
                    unary_denials.setdefault(cid, row)

            # ---- insertion_plan_fp ----
            if args.cap_plan_fp:
                p = 1.0 / (1.0 + np.exp(-(logits[s["row"]] * slope + icpt)))
                cand = plan_candidates(p, thr_plan, thr_pos, args.possible_cap)
                ranked = sorted(cand.tolist(), key=lambda c: -float(p[c]))
                made = 0
                for c in ranked:
                    if made >= args.cap_plan_fp:
                        break
                    if c in positives:
                        continue
                    reason = None
                    if c in excluded_plan:
                        issues["plan_fp_class_excluded_incidental_or_degenerative"] += 1
                        continue
                    if plan_class[c] >= plan_cap:
                        issues["plan_fp_class_at_per_class_cap"] += 1
                        continue
                    organ = class_organ.get(c)
                    if cooc_related(c, positives):
                        reason = "plan_fp_class_related_to_documented_finding"
                    elif already_described(c, rows):
                        reason = "plan_fp_class_named_in_report_sentence"
                    elif c not in donors:
                        reason = "plan_fp_no_donor_for_class"
                    elif c in s["denied"] and c not in unary_denials:
                        reason = "plan_fp_denied_in_nonunary_sentence"
                    elif c not in s["denied"] and anchor_row(rows, organ) is None:
                        reason = "plan_fp_organ_not_described_in_report"
                    if reason:
                        issues[reason] += 1
                        continue
                    d = take_donor(c, s["patient_id"], target, rows, screen=True)
                    if d is None:
                        issues["plan_fp_no_independent_donor_with_capacity_or_screen"] += 1
                        continue
                    if c in s["denied"]:
                        row = unary_denials[c]
                        rejected = replace_span(target, row, d[2]); mode = "replace_unary_denial"; changed = row["text"]
                    else:
                        row = anchor_row(rows, organ)
                        rejected = insert_after(target, row, d[2]); mode = "insert_after_organ_sentence"; changed = None
                    tier = "confident" if p[c] >= thr_plan[c] else "possible"
                    pair = make_pair(s, "insertion_plan_fp", rejected, dict(
                        rule="assert_unreported_mil_plan_item_with_train_donor_sentence", mode=mode,
                        calibrated_p=round(float(p[c]), 4), plan_tier=tier,
                        plan_threshold=round(float(thr_plan[c]), 4), possible_threshold=round(float(thr_pos[c]), 4),
                        report_state="denied" if c in s["denied"] else "unmentioned", organ=organ,
                        anchor_sentence=row["text"], inserted_sentence=d[2], donor_report_uid=d[0],
                        donor_patient_id=d[1], donor_has_size=d[3]),
                        changed=changed or d[2], labels=labels, class_ids=[c])
                    if pair:
                        local.append(pair); made += 1
                        plan_class[c] += 1
                        tier_counts[tier] += 1; state_counts["denied" if c in s["denied"] else "unmentioned"] += 1
                        size_counts[("insertion_plan_fp", d[3])] += 1
                if not made:
                    issues["plan_fp_none_for_report" if len(ranked) else "plan_fp_no_plan_items"] += 1

            # ---- insertion_random ----
            plan_made = any(pr["pair_type"] == "insertion_plan_fp" for pr in local)
            if args.cap_random and plan_made and rng.random() >= args.random_keep_with_plan:
                issues["random_skipped_for_balance_report_has_plan_fp"] += 1
            elif args.cap_random:
                organs_here = sorted(o for o in report_organs if donors_by_organ.get(o))
                made, tries = 0, 0
                while organs_here and made < args.cap_random and tries < 40:
                    tries += 1
                    pool = donors_by_organ[organs_here[rng.randrange(len(organs_here))]]
                    c, k = pool[rng.randrange(len(pool))]
                    if c in excluded_plan:
                        issues["random_draw_rejected_excluded_incidental_or_degenerative"] += 1
                        continue
                    if c in positives or c in s["denied"] or cooc_related(c, positives):
                        issues["random_draw_rejected_related_or_documented"] += 1
                        continue
                    if not plausible_for_study(c, s):
                        issues["random_draw_rejected_implausible_for_study_type"] += 1
                        continue
                    if already_described(c, rows):
                        issues["random_draw_rejected_class_named_in_report"] += 1
                        continue
                    row = anchor_row(rows, class_organ[c])
                    if row is None:
                        continue
                    d = take_donor(c, s["patient_id"], target, rows, screen=True)
                    if d is None:
                        continue
                    rejected = insert_after(target, row, d[2])
                    pair = make_pair(s, "insertion_random", rejected, dict(
                        rule="insert_train_donor_sentence_of_absent_class_after_organ_sentence",
                        organ=class_organ[c], anchor_sentence=row["text"], inserted_sentence=d[2],
                        donor_report_uid=d[0], donor_patient_id=d[1], donor_has_size=d[3],
                        calibrated_p=round(float(1.0 / (1.0 + math.exp(-(float(logits[s["row"], c]) * float(slope[c]) + float(icpt[c]))))), 4)),
                        changed=d[2], labels=labels, class_ids=[c])
                    if pair:
                        local.append(pair); made += 1
                        size_counts[("insertion_random", d[3])] += 1
                if not made:
                    issues["random_no_valid_absent_class_donor"] += 1

            # ---- normalised ----
            if args.cap_normalised:
                by_organ = defaultdict(list)
                for k, row in enumerate(rows):
                    positive = [r for r in row["inside"] + row["covering"] if r["state"] in ("asserted", "uncertain")]
                    if not positive or len(row["organs"]) != 1:
                        continue
                    t = row["text"]
                    if not t.endswith(".") or len(t.split()) < 3 or target.count(t) != 1 or NONCURRENT.search(t):
                        issues["normalised_sentence_filtered"] += 1
                        continue
                    if k + 1 < len(rows) and NEXT_DEPENDENCY.search(rows[k + 1]["text"]):
                        issues["normalised_following_sentence_may_depend_on_it"] += 1
                        continue
                    by_organ[next(iter(row["organs"]))].append(row)
                organs_ab = sorted(by_organ, key=lambda o: hk(args.seed, uid, o))
                made = 0
                for o in organs_ab:
                    if made >= args.cap_normalised:
                        break
                    group = by_organ[o]
                    m = rng.randint(1, len(group))
                    chosen_rows = sorted(rng.sample(group, m), key=lambda r: r["start"])
                    tmpl, score = pick_template(o, rows, target, chosen_rows)
                    if tmpl is None:
                        issues["normalised_no_matching_template_for_organ"] += 1
                        continue
                    if len(rows) - m < 1:
                        issues["normalised_would_empty_report"] += 1
                        continue
                    first, rest = chosen_rows[0], chosen_rows[1:]
                    rejected = delete_rows(target, rest)
                    # first row's offsets are unchanged only if all deleted rows lie after it (sorted -> yes)
                    rejected = rejected[:first["start"]] + tmpl + rejected[first["end"]:]
                    cls = sorted({r["class_id"] for row in chosen_rows for r in row["inside"] + row["covering"]
                                  if r["state"] in ("asserted", "uncertain")})
                    pair = make_pair(s, "normalised", rejected, dict(
                        rule="replace_abnormal_sentences_of_one_organ_with_organ_normal_template",
                        organ=o, replaced_sentences=[r["text"] for r in chosen_rows], n_replaced=m,
                        n_abnormal_in_organ=len(group), template=tmpl, template_style_score=round(score, 4),
                        template_train_reports=sent_reports[tmpl]),
                        changed=" ".join(r["text"] for r in chosen_rows), labels=labels, class_ids=cls)
                    if pair:
                        local.append(pair); made += 1
                        norm_counts["all" if m == len(group) else "partial"] += 1
                if not by_organ:
                    issues["normalised_no_qualified_abnormal_sentence"] += 1

            # ---- multi_omission ----
            if args.cap_multi:
                if len(oms) < 2:
                    issues["multi_omission_fewer_than_2_candidates"] += 1
                else:
                    index_of = {row["index"]: row for row in rows}
                    made, tries = 0, 0
                    while made < args.cap_multi and tries < 20:
                        tries += 1
                        k = rng.randint(2, min(4, len(oms)))
                        pick = sorted(rng.sample(oms, k), key=lambda c: c["index"])
                        gone = {c["index"] for c in pick}
                        ok = True
                        for c in pick:
                            nxt = c["index"] + 1
                            while nxt in gone:
                                nxt += 1
                            if nxt < len(rows) and NEXT_DEPENDENCY.search(index_of[nxt]["text"]):
                                ok = False
                                break
                        if not ok or len(rows) - k < 2:
                            continue
                        rejected = delete_rows(target, [index_of[i] for i in gone])
                        cls = sorted({x for c in pick for x in c["class_ids"]})
                        pair = make_pair(s, "multi_omission", rejected, dict(
                            rule="delete_2_to_4_whole_sentences_with_qualified_positive_quotes",
                            n_deleted=k, sentence_indices=sorted(gone), source_states=[c["state"] for c in pick],
                            deleted_sentences=[c["text"] for c in pick]),
                            changed=" ".join(c["text"] for c in pick), labels=labels, class_ids=cls)
                        if pair:
                            local.append(pair); made += 1; multi_k[k] += 1
                    if not made:
                        issues["multi_omission_dependency_or_length_filter"] += 1
            for pr in local:
                type_counts[pr["pair_type"]] += 1
            for kind in {pr["pair_type"] for pr in local}:
                covered[kind] += 1
            covered["any_pair" if local else "ce_only"] += 1
            all_pairs.extend(local)
            if (n + 1) % 50000 == 0:
                log(f"pass2 {n+1}/{len(train)} pairs={dict(type_counts)}")
        log(f"pairs={dict(type_counts)} covered={dict(covered)}")
        return dict(used=used, pointer=pointer, type_counts=type_counts, covered=covered, tier_counts=tier_counts,
                    state_counts=state_counts, size_counts=size_counts, norm_counts=norm_counts, multi_k=multi_k,
                    all_pairs=all_pairs, issues=issues, plan_class=plan_class)

    caps_saved = (args.cap_random, args.cap_normalised, args.cap_multi)
    args.cap_random = args.cap_normalised = args.cap_multi = 0
    plan_cap, history = 10 ** 9, []
    for it in range(6):   # fixed point: cap = floor(frac * N_plan) (plan-only passes)
        r = run_pass2(plan_cap)
        n_plan = r['type_counts']['insertion_plan_fp']
        new_cap = int(math.floor(args.plan_class_cap_frac * n_plan * 0.995))
        history.append(dict(cap=plan_cap, n_plan=n_plan, top=r['plan_class'].most_common(1)))
        log(f'plan-cap iteration {it}: cap={plan_cap} n_plan={n_plan} top={r["plan_class"].most_common(3)}')
        if plan_cap != 10 ** 9 and r['plan_class'].most_common(1)[0][1] <= args.plan_class_cap_frac * n_plan:
            break
        plan_cap = new_cap
    args.cap_random, args.cap_normalised, args.cap_multi = caps_saved
    r = run_pass2(plan_cap)
    (used, pointer, type_counts, covered, tier_counts, state_counts, size_counts, norm_counts, multi_k, all_pairs,
     issues, plan_class) = (r[k] for k in ('used', 'pointer', 'type_counts', 'covered', 'tier_counts', 'state_counts',
                                         'size_counts', 'norm_counts', 'multi_k', 'all_pairs', 'issues', 'plan_class'))
    # strict share: the full pass consumes donors with the other types, so trim the lowest-p excess of capped classes
    plan_pairs = [pr for pr in all_pairs if pr["pair_type"] == "insertion_plan_fp"]
    by_cls = defaultdict(list)
    for pr in plan_pairs:
        by_cls[pr["class_ids"][0]].append(pr)
    n_plan = len(plan_pairs)
    while True:
        lim = int(math.floor(args.plan_class_cap_frac * n_plan))
        excess = sum(max(0, len(v) - lim) for v in by_cls.values())
        if excess == 0:
            break
        n_plan -= excess
        for cls_, v in by_cls.items():
            if len(v) > lim:
                v.sort(key=lambda pr: -pr["provenance"]["calibrated_p"])
                del v[lim:]
    keep_ids = {pr["pair_id"] for v in by_cls.values() for pr in v}
    trimmed = [pr for pr in plan_pairs if pr["pair_id"] not in keep_ids]
    all_pairs = [pr for pr in all_pairs if pr["pair_type"] != "insertion_plan_fp" or pr["pair_id"] in keep_ids]
    issues["plan_fp_trimmed_to_strict_class_share"] += len(trimmed)
    for pr in trimmed:
        plan_class[pr["class_ids"][0]] -= 1
        tier_counts[pr["provenance"]["plan_tier"]] -= 1
        state_counts[pr["provenance"]["report_state"]] -= 1
        size_counts[("insertion_plan_fp", pr["provenance"]["donor_has_size"])] -= 1
    type_counts = Counter(pr["pair_type"] for pr in all_pairs)
    per_report = defaultdict(set)
    for pr in all_pairs:
        per_report[pr["report_uid"]].add(pr["pair_type"])
    n_ce = covered["any_pair"] + covered["ce_only"]
    covered = Counter()
    for kinds in per_report.values():
        for kd in kinds:
            covered[kd] += 1
    covered["any_pair"] = len(per_report)
    covered["ce_only"] = n_ce - len(per_report)
    n_plan = type_counts['insertion_plan_fp']
    log(f'trimmed {len(trimmed)} plan pairs to strict share; pairs={dict(type_counts)}')
    log(f'final plan cap={plan_cap} n_plan={n_plan} max_share={plan_class.most_common(1)[0][1] / max(1, n_plan):.4f}')

    summary = dict(
        schema="ct_report_preference_index_v1", builder_schema="ct_report_preferences_plan_summary",
        policy=POLICY, created=time.strftime("%F %T"), index=args.index, findings=args.findings,
        dense_mil=args.dense, calibration=args.calibration, split=args.split, seed=args.seed,
        target_prefix=TARGET_PREFIX, chosen_text_function="SFT dataset report target (full writer) == findings sqlite of `base findings`",
        train_reports=len(train), val_used_for_pairs=False,
        caps_per_report=dict(insertion_plan_fp=args.cap_plan_fp, insertion_random=args.cap_random,
                             random_keep_with_plan=args.random_keep_with_plan,
                             normalised=args.cap_normalised, multi_omission=args.cap_multi),
        pairs_total=sum(type_counts.values()), pairs_by_type=dict(type_counts),
        reports_covered=dict(covered),
        insertion_plan_fp_tiers=dict(tier_counts), insertion_plan_fp_report_state=dict(state_counts),
        insertion_donor_has_size={f"{k[0]}|{k[1]}": v for k, v in size_counts.items()},
        normalised_extent=dict(norm_counts), multi_omission_sentences_deleted={str(k): v for k, v in sorted(multi_k.items())},
        donor_classes=len(donors), donor_sentences=sum(map(len, donors.values())),
        donor_exclusions=dict(donor_reasons), template_organs={o: len(v) for o, v in tmpl_by_organ.items()},
        class_organs_assigned=len(class_organ),
        insertion_plan_fp_excluded_classes=excluded_plan_names,
        insertion_random_uses_same_exclusion_and_donor_screen=True,
        insertion_plan_fp_class_cap=dict(fraction=args.plan_class_cap_frac, absolute=plan_cap, iterations=history),
        insertion_plan_fp_top_classes=[(labels[c], k, round(k / max(1, n_plan), 4)) for c, k in plan_class.most_common(15)],
        insertion_random_top_classes=Counter(lb for pr in all_pairs if pr["pair_type"] == "insertion_random" for lb in pr["labels"]).most_common(15), exclusion_reasons=dict(sorted(issues.items())),
        thresholds=dict(cooc_exclude=args.cooc_exclude, cooc_exclude_same_organ=args.cooc_exclude_same_organ, template_min_reports=args.template_min_reports,
                        max_chosen_chars=args.max_chosen_chars),
        unmentioned_is_negative="only for insertion_* pairs: inserting an assertion of an undocumented class is "
                                "dispreferred relative to the documented report; classes strongly co-reported with "
                                "a documented finding (cooc >= cooc_exclude) or parent/child heads are excluded",
        limitations=[
            "Preferences are relative to the documented Findings, not image truth (train MIL is in-sample, so plan "
            "FPs on train are rarer and some unmentioned plan items may be true but unreported).",
            "Organ placement and normal templates use a keyword organ lexicon; donors must mention the class's "
            "majority organ.",
            "Normalised: partial replacement may leave other abnormal sentences of the organ, which can then "
            "contradict the template.",
        ],
        seconds=None)

    # ---------------- write stores ----------------
    out.mkdir(parents=True, exist_ok=True)
    new_store = out / "plan_pairs.sqlite"
    merged_store = out / "dpo_pairs.sqlite"
    for path in (new_store, merged_store) if args.base_store else (new_store,):
        if path.exists():
            raise FileExistsError(path)
    db, tmp = create_store(new_store)
    db.executemany("INSERT OR IGNORE INTO pairs VALUES(?,?,?,?,?,?)",
                   ((p["pair_id"], p["report_uid"], p["pair_type"], None, p["chosen_sha256"], p["rejected_text"])
                    for p in all_pairs))
    db.execute("CREATE INDEX pairs_uid ON pairs(report_uid)")
    # provenance table (extra; the trainer only reads pairs + metadata)
    db.execute("CREATE TABLE provenance(pair_id TEXT PRIMARY KEY, record TEXT NOT NULL)")
    db.executemany("INSERT OR IGNORE INTO provenance VALUES(?,?)",
                   ((p["pair_id"], jdump({k: v for k, v in p.items() if k not in ("chosen_text", "rejected_text")}))
                    for p in all_pairs))
    n_new = db.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
    summary["new_store"] = new_store.name
    summary["seconds"] = round(time.time() - began, 1)
    db.execute("INSERT INTO metadata VALUES(?,?)", ("summary", jdump(dict(
        schema="ct_report_preference_index_v1", sources=["fora_report.pairs.plan_pairs"],
        pairs_by_type=dict(type_counts), reports=covered["any_pair"], target_prefix=TARGET_PREFIX,
        policy=POLICY))))
    db.execute("INSERT INTO metadata VALUES(?,?)", ("build_summary", jdump(summary)))
    db.commit()
    db.close()
    os.replace(tmp, new_store)
    log(f"wrote {new_store} rows={n_new}")

    # examples: 20 random insertion_plan_fp pairs
    ex_rng = random.Random(args.seed + 1)
    by_t = defaultdict(list)
    for p in all_pairs:
        by_t[p["pair_type"]].append(p)
    with open(out / "plan_pairs_examples.jsonl", "w") as fh:
        for t in ("insertion_plan_fp",):
            for p in ex_rng.sample(by_t[t], min(20, len(by_t[t]))):
                fh.write(json.dumps(dict(pair_type=t, report_uid=p["report_uid"], class_labels=p["labels"],
                                         provenance=p["provenance"], chosen_text=p["chosen_text"],
                                         rejected_text=p["rejected_text"]), ensure_ascii=False) + "\n")
    if args.base_store:
        rows = ((p["pair_id"], p["report_uid"], p["pair_type"], None, p["chosen_sha256"], p["rejected_text"])
                for p in all_pairs)
        merged = merge_stores(args.base_store, rows, new_store.name, merged_store, POLICY)
        summary["merged_store"] = merged_store.name
        summary["merged_pairs_by_type"] = merged["pairs_by_type"]
        summary["merged_reports"] = merged["reports"]
        log(f"wrote {merged_store} by_type={merged['pairs_by_type']} reports={merged['reports']}")

    summary["seconds"] = round(time.time() - began, 1)
    (out / "plan_pairs_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log("done")
    return summary


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="report index sqlite (samples + metadata['global'].labels)")
    ap.add_argument("--findings", required=True, help="findings sqlite of `base findings`")
    ap.add_argument("--dense", required=True, help="frozen-MIL dense logits npz (report_uid, fora_id, label, logit)")
    ap.add_argument("--calibration", required=True, help="tier calibration json (train Platt + Youden)")
    ap.add_argument("--split", required=True, help="patient-level split json (patient id -> split)")
    ap.add_argument("--heads", required=True, help="label heads json ('attached': [{name, parent}])")
    ap.add_argument("--hierarchy", required=True, help="hierarchy json ('nodes': {name: {ancestors}})")
    ap.add_argument("--exclusions", default=str(DEFAULT_EXCLUSIONS))
    ap.add_argument("--base_store", default="", help="base + attribute pair store to merge with (empty = no merge)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--possible_cap", type=int, default=20, help="possible-tier cap of the writer's plan")
    ap.add_argument("--cap_plan_fp", type=int, default=1)
    ap.add_argument("--cap_random", type=int, default=1)
    ap.add_argument("--cap_normalised", type=int, default=1)
    ap.add_argument("--cap_multi", type=int, default=1)
    ap.add_argument("--random_keep_with_plan", type=float, default=0.3,
                    help="keep insertion_random in a report that already has insertion_plan_fp with this probability")
    ap.add_argument("--cooc_exclude", type=float, default=0.25)
    ap.add_argument("--cooc_exclude_same_organ", type=float, default=0.10)
    ap.add_argument("--plan_class_cap_frac", type=float, default=0.03)
    ap.add_argument("--template_min_reports", type=int, default=30)
    ap.add_argument("--templates_per_organ", type=int, default=40)
    ap.add_argument("--max_chosen_chars", type=int, default=7000,
                    help="skip reports whose chosen target may exceed max_report_tokens=2048 after the edit")
    return ap.parse_args(argv)


def main(argv=None):
    build(parse_args(argv))


if __name__ == "__main__":
    main()
