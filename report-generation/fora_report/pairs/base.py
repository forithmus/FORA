"""CT DPO base pairs: omission, verbatim repetition and explicit-denial controls (paper: "Preference optimization").

chosen   = the complete source Findings target used by SFT (H5 ``report/structured/findings`` with the same fallbacks
           as the SFT dataset), exported once to a findings sqlite (``findings`` subcommand).
omission = the same target with ONE whole sentence removed that contains a qualified asserted/uncertain extractor
           quote (nothing is rewritten; at most --max_omissions per report, deterministic hash order).
repetition control = target + one verbatim repeated qualifying sentence (the non-repetitive source is preferred).
explicit-denial control = one unary explicit denial sentence replaced by a unary asserted sentence of the SAME class
           from another TRAIN patient (donor use capped by the donor sentence's train-report support). An unmentioned
           class is never treated as negative.

The ``build`` subcommand also writes the attribute-corruption candidates consumed by ``gemma_verifier.py``.
Validation reports (``val`` split) are never used for pairs; the builder fails on any train/val overlap.

Input index: sqlite with table ``samples(report_uid, patient_uid, h5_id, split, protocol_class, study_type_ids_json,
asserted_json, uncertain_json, denied_json, asserted_evidence_json, uncertain_evidence_json, denied_evidence_json,
extraction_verdict)`` and ``metadata(key='global')`` holding ``{"labels": [...]}`` (written by data-preprocessing).

  python -m fora_report.pairs.base findings --index INDEX --h5_root H5 --output findings.sqlite
  python -m fora_report.pairs.base build --index INDEX --findings findings.sqlite --output_dir OUT
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import os
from pathlib import Path
import re
import sqlite3
import time

from .text import (ADDITIVE_REFERENCE, DONOR_EXCLUDE, EMPTY_FINDINGS, NEXT_DEPENDENCY, NONCURRENT, PAIR_SCHEMA,
                   SCOPE_LIMITED, TARGET_PREFIX, attribute_spans, hkey, jdump, norm, sha, split_sentences,
                   unary_exclusion)

POLICY = "ct_source_sentence_preferences"


def _read_findings_shard(job):
    import h5py
    h5_root, h5_id, items = job
    out = []
    path = os.path.join(h5_root, f"{h5_id}.h5")
    with h5py.File(path, "r") as handle:
        for uid, patient in items:
            findings, source = "", "missing"
            try:
                report = handle[patient][uid]["report"]
                structured = report.get("structured")
                if structured is not None and "findings" in structured:
                    value = structured["findings"][()]
                    findings = (value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)).strip()
                    source = "structured_findings"
                if not findings:
                    for key in ("extracted_sentences", "there_is_statements"):
                        if key in report:
                            value = report[key][()]
                            value = value.decode("utf-8", "replace") if isinstance(value, bytes) else value
                            try:
                                parsed = json.loads(value)
                            except Exception:
                                parsed = []
                            parts = [str(x).strip() for x in parsed if str(x).strip()] if isinstance(parsed, list) else []
                            if parts:
                                findings, source = " ".join(parts), key
                                break
            except KeyError:
                source = "missing_group"
            if not findings:
                findings, source = EMPTY_FINDINGS, source + "_empty_fallback"
            out.append((uid, findings, source))
    return out


def cmd_findings(args):
    from concurrent.futures import ProcessPoolExecutor
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    src = sqlite3.connect(f"file:{args.index}?mode=ro", uri=True)
    rows = src.execute("SELECT report_uid, patient_uid, h5_id, split FROM samples WHERE protocol_class=0").fetchall()
    meta = {uid: (patient, h5, split) for uid, patient, h5, split in rows}
    groups = defaultdict(list)
    for uid, patient, h5, split in rows:
        groups[h5].append((uid, patient))
    tmp = out.with_suffix(".partial")
    if tmp.exists():
        tmp.unlink()
    db = sqlite3.connect(tmp)
    db.execute("CREATE TABLE findings(report_uid TEXT PRIMARY KEY, patient_uid TEXT, h5_id TEXT, split TEXT,"
               " findings TEXT NOT NULL, source TEXT NOT NULL, sha256 TEXT NOT NULL)")
    db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
    began, done, sources = time.time(), 0, Counter()
    jobs = [(args.h5_root, h5, items) for h5, items in sorted(groups.items())]
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(_read_findings_shard, jobs, chunksize=4):
            for uid, findings, source in result:
                patient, h5, split = meta[uid]
                db.execute("INSERT INTO findings VALUES(?,?,?,?,?,?,?)",
                           (uid, patient, h5, split, findings, source, sha(findings)))
                sources[source] += 1
            done += 1
            if done % 250 == 0:
                print(f"[findings] shards {done}/{len(jobs)} {time.time()-began:.0f}s", flush=True)
    db.execute("INSERT INTO metadata VALUES(?,?)", ("summary", jdump(dict(
        index=args.index, h5_root=args.h5_root, reports=len(rows), shards=len(jobs), sources=dict(sources),
        policy="identical to the SFT dataset report target (full writer)"))))
    db.execute("CREATE INDEX findings_split ON findings(split)")
    db.commit(); db.close()
    os.replace(tmp, out)
    print(f"[findings] wrote {out} reports={len(rows)} sources={dict(sources)} {time.time()-began:.0f}s")

def load_studies(index_path, findings_path, split):
    idx = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    fnd = sqlite3.connect(f"file:{findings_path}?mode=ro", uri=True)
    text = dict(fnd.execute("SELECT report_uid, findings FROM findings WHERE split=?", (split,)))
    labels = json.loads(idx.execute("SELECT value FROM metadata WHERE key='global'").fetchone()[0])["labels"]
    studies = []
    for (uid, patient, h5, types, asserted, uncertain, denied, a_ev, u_ev, d_ev, verdict) in idx.execute(
            "SELECT report_uid, patient_uid, h5_id, study_type_ids_json, asserted_json, uncertain_json, "
            "denied_json, asserted_evidence_json, uncertain_evidence_json, denied_evidence_json, "
            "extraction_verdict FROM samples WHERE split=? AND protocol_class=0 ORDER BY report_uid", (split,)):
        if uid not in text:
            raise ValueError("findings missing for " + uid)
        studies.append(dict(report_uid=uid, patient_id=patient, h5_id=h5, split=split,
                            study_type_ids=json.loads(types), findings=text[uid], verdict=verdict,
                            asserted=set(json.loads(asserted)), uncertain=set(json.loads(uncertain)),
                            denied=set(json.loads(denied)),
                            evidence={"asserted": json.loads(a_ev), "uncertain": json.loads(u_ev),
                                      "denied": json.loads(d_ev)}))
    return studies, labels


def quote_records(study):
    """Qualified quotes: pass/partial verdict rows only; state consistent with the class lists."""
    out = []
    for state, members in (("asserted", study["asserted"]), ("uncertain", study["uncertain"]),
                           ("denied", study["denied"])):
        for raw, quotes in study["evidence"][state].items():
            cid = int(raw)
            if cid not in members:
                continue  # e.g. denied evidence of a class that is positive (positive wins)
            for quote in quotes:
                q = norm(quote)
                if len(q) >= 3:
                    out.append(dict(class_id=cid, state=state, quote=str(quote).strip(), nquote=q))
    return out


def analyse_study(study, labels):
    """Sentence table with quote support, omission eligibility and issues."""
    target = study["findings"]
    sentences = split_sentences(target)
    records = quote_records(study) if study["verdict"] == "pass" else []
    rows, issues = [], Counter()
    for i, (text, start, end) in enumerate(sentences):
        ntext = norm(text)
        inside = [r for r in records if r["nquote"] in ntext]
        covering = [r for r in records if ntext and ntext in r["nquote"] and r not in inside]
        rows.append(dict(index=i, text=text, start=start, end=end, ntext=ntext, inside=inside,
                         covering=covering))
    return sentences, rows, records, issues


def omission_candidates(study, rows, issues):
    target = study["findings"]
    out = []
    for k, row in enumerate(rows):
        text = row["text"]
        positive = [r for r in row["inside"] if r["state"] in ("asserted", "uncertain")]
        reason = None
        if not positive:
            reason = "no_qualified_positive_quote_in_sentence"
        elif not text.endswith(".") or len(text.split()) < 3 or not re.search(r"[A-Za-z]", text):
            reason = "not_independent_complete_sentence"
        elif target.count(text) != 1:
            reason = "source_sentence_repeated"
        elif NONCURRENT.search(text):
            reason = "noncurrent_or_comparison_language"
        elif k + 1 < len(rows) and NEXT_DEPENDENCY.search(rows[k + 1]["text"]):
            reason = "following_sentence_may_depend_on_it"
        elif len(rows) < 2:
            reason = "deletion_would_empty_report"
        if reason:
            if positive:
                issues[reason] += 1
            continue
        state = "asserted" if any(r["state"] == "asserted" for r in positive) else "uncertain"
        out.append(dict(index=row["index"], text=text, state=state,
                        class_ids=sorted({r["class_id"] for r in positive}),
                        quotes=sorted({r["quote"] for r in positive})))
    return out


def edited_target(target, sentence, replacement):
    if target.count(sentence) != 1:
        raise ValueError("edit sentence must occur exactly once")
    left, right = target.split(sentence)
    if replacement:
        return left + replacement + right
    return left.rstrip() + (" " if left.strip() and right.strip() else "") + right.lstrip()


def make_pair(study, pair_type, rejected, provenance, *, changed=None, attribute_type=None,
              labels=None, class_ids=()):
    chosen = study["findings"]
    if not rejected.strip() or rejected == chosen:
        return None
    rec = dict(schema=PAIR_SCHEMA, report_uid=study["report_uid"], patient_id=study["patient_id"],
               h5_id=study["h5_id"], split=study["split"], pair_type=pair_type,
               chosen_text=chosen, rejected_text=rejected, chosen_sha256=sha(chosen),
               changed_sentence=changed, attribute_type=attribute_type,
               class_ids=list(class_ids), labels=[labels[c] for c in class_ids] if labels else [],
               provenance=provenance, target_prefix=TARGET_PREFIX,
               preference_basis="documented_source_relative_not_independent_image_truth",
               clinical_validity_verified=False, synthetic=True)
    rec["pair_id"] = hkey(rec)[:32]
    return rec


def unary_sentences(rows, records, state):
    """Sentences equal (modulo case/period) to one quote of ``state`` and touching one class only."""
    out = []
    for row in rows:
        match = [r for r in row["inside"] if r["state"] == state and r["nquote"] == row["ntext"]]
        if not match:
            continue
        touching = {r["class_id"] for r in row["inside"] + row["covering"]}
        reason = unary_exclusion(row["text"])
        if len(touching) != 1:
            reason = reason or "multiple_classes_in_sentence"
        cid = match[0]["class_id"]
        if any(r["class_id"] == cid and r["nquote"] != row["ntext"] and r["nquote"] not in row["ntext"]
               for r in records):
            reason = reason or "class_documented_elsewhere"
        out.append((row, cid, reason))
    return out


def cmd_build(args):
    began = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs_path = out / "base_pairs.jsonl.gz"
    cand_path = out / "attribute_candidates.jsonl.gz"
    for path in (pairs_path, cand_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
    train, labels = load_studies(args.index, args.findings, "train")
    val_uids = {u for (u,) in sqlite3.connect(f"file:{args.index}?mode=ro", uri=True).execute(
        "SELECT report_uid FROM samples WHERE split='val'")}
    val_patients = {p for (p,) in sqlite3.connect(f"file:{args.index}?mode=ro", uri=True).execute(
        "SELECT DISTINCT patient_uid FROM samples WHERE split='val'")}
    if args.limit:
        train = train[: args.limit]
    print(f"[build] train studies={len(train)} labels={len(labels)} ({time.time()-began:.0f}s)", flush=True)
    issues = Counter()
    per_study = {}
    donors = defaultdict(list)
    denial_candidates = {}
    for n, study in enumerate(train):
        if study["report_uid"] in val_uids or study["patient_id"] in val_patients:
            raise ValueError("train study overlaps the val split: " + study["report_uid"])
        sentences, rows, records, local_issues = analyse_study(study, labels)
        if study["verdict"] != "pass":
            issues["extraction_verdict_not_pass"] += 1
        omissions = omission_candidates(study, rows, local_issues)
        issues.update(local_issues)
        per_study[study["report_uid"]] = (rows, records, omissions)
        for row, cid, reason in unary_sentences(rows, records, "asserted"):
            if reason is None and not ADDITIVE_REFERENCE.search(row["text"]) and not DONOR_EXCLUDE.search(row["text"]):
                donors[cid].append((study["report_uid"], study["patient_id"], row["text"]))
        dens = []
        for row, cid, reason in unary_sentences(rows, records, "denied"):
            if reason is None and SCOPE_LIMITED.search(row["text"]):
                reason = "denial_scope_limited_not_whole_study"
            if reason is None and (cid in study["asserted"] or cid in study["uncertain"]):
                reason = "class_also_positive"
            if reason is None and study["findings"].count(row["text"]) != 1:
                reason = "source_sentence_repeated"
            if reason:
                issues["denial_" + reason] += 1
                continue
            dens.append((row, cid))
        denial_candidates[study["report_uid"]] = dens
        if (n + 1) % 50000 == 0:
            print(f"[build] analysed {n+1}/{len(train)} ({time.time()-began:.0f}s)", flush=True)
    # donor support = number of distinct TRAIN reports containing the exact donor sentence
    donor_text = {t for options in donors.values() for _u, _p, t in options}
    support = Counter()
    for study in train:
        target = study["findings"]
        for text, _s, _e in split_sentences(target):
            if text in donor_text:
                support[text] += 1
    for options in donors.values():
        options.sort(key=lambda d: hkey("donor", d[0], d[2]))
    used = Counter()
    pointer = Counter()
    by_uid = {s["report_uid"]: s for s in train}
    pairs_written, type_counts, covered = 0, Counter(), Counter()
    omission_state = Counter()
    cand_counts, cand_counts_capped, cand_reports = Counter(), Counter(), Counter()
    with gzip.open(pairs_path, "wt", encoding="utf-8", compresslevel=3) as pairs_out, \
            gzip.open(cand_path, "wt", encoding="utf-8", compresslevel=3) as cand_out:
        for study in train:
            uid = study["report_uid"]
            rows, records, omissions = per_study[uid]
            target = study["findings"]
            local = []
            ordered = sorted(omissions, key=lambda c: hkey(args.seed, uid, c["index"], c["text"]))
            for cand in ordered[: args.max_omissions]:
                pair = make_pair(study, "omission", edited_target(target, cand["text"], ""),
                                 dict(rule="delete_one_whole_sentence_with_qualified_positive_quote",
                                      source_state=cand["state"], quotes=cand["quotes"],
                                      sentence_index=cand["index"]),
                                 changed=cand["text"], labels=labels, class_ids=cand["class_ids"])
                if pair:
                    local.append(pair)
                    omission_state[cand["state"]] += 1
            if len(omissions) > args.max_omissions:
                issues["omission_candidates_over_cap"] += len(omissions) - args.max_omissions
            if ordered:
                cand = ordered[0]
                pair = make_pair(study, "repetition", target.rstrip() + " " + cand["text"],
                                 dict(rule="verbatim_repetition_shorter_preferred", sentence_index=cand["index"]),
                                 changed=cand["text"], labels=labels, class_ids=cand["class_ids"])
                if pair:
                    local.append(pair)
            for row, cid in sorted(denial_candidates[uid], key=lambda x: hkey(args.seed, uid, x[0]["text"])):
                # Rotating pointer over the hashed donor list of this class: the
                # first donor with remaining capacity from another patient
                # (amortised O(1); capacity = distinct train reports with the sentence).
                options = donors[cid]
                donor = None
                for step in range(min(len(options), 256)):
                    candidate = options[(pointer[cid] + step) % len(options)]
                    if (candidate[1] != study["patient_id"] and used[candidate[2]] < support[candidate[2]]
                            and candidate[2] != row["text"]):
                        donor = candidate
                        pointer[cid] = (pointer[cid] + step + 1) % len(options)
                        break
                if donor is None:
                    issues["denial_no_independent_train_donor_with_capacity"] += 1
                    continue
                used[donor[2]] += 1
                pair = make_pair(study, "explicit_denial", edited_target(target, row["text"], donor[2]),
                                 dict(rule="explicit_denial_replaced_by_same_class_asserted_train_donor",
                                      denial_sentence=row["text"], donor_report_uid=donor[0],
                                      donor_patient_id=donor[1], donor_sentence=donor[2],
                                      donor_support_reports=support[donor[2]]),
                                 changed=row["text"], labels=labels, class_ids=[cid])
                if pair:
                    local.append(pair)
                break
            for pair in local:
                pairs_out.write(jdump(pair) + "\n")
                type_counts[pair["pair_type"]] += 1
            pairs_written += len(local)
            for kind in {p["pair_type"] for p in local}:
                covered[kind] += 1
            covered["any_pair" if local else "ce_only"] += 1
            # ---- attribute candidates (finding sentences only) ----
            per_type = Counter()
            seen_rows = []
            for row in rows:
                positive = [r for r in row["inside"] if r["state"] in ("asserted", "uncertain")]
                text = row["text"]
                if (not positive or target.count(text) != 1 or NONCURRENT.search(text)
                        or not text.endswith(".") or len(text.split()) < 3):
                    continue
                state = "asserted" if any(r["state"] == "asserted" for r in positive) else "uncertain"
                found = attribute_spans(text, state)
                for kind, spans in found.items():
                    cand_counts[kind] += 1
                    seen_rows.append((kind, row, spans, state, positive))
            ordered_c = sorted(seen_rows, key=lambda x: hkey(args.seed, uid, x[0], x[1]["index"]))
            kept = 0
            for kind, row, spans, state, positive in ordered_c:
                if per_type[kind] >= args.max_attr_per_type or kept >= args.max_attr_per_report:
                    continue
                per_type[kind] += 1
                kept += 1
                cand_counts_capped[kind] += 1
                rec = dict(report_uid=uid, split="train", sentence_index=row["index"], sentence=row["text"],
                           attribute_type=kind, attribute_spans=spans, source_state=state,
                           quotes=sorted({r["quote"] for r in positive}),
                           class_ids=sorted({r["class_id"] for r in positive}),
                           sentence_sha256=sha(row["text"]), chosen_sha256=sha(target))
                rec["cand_id"] = hkey(rec)[:24]
                cand_out.write(jdump(rec) + "\n")
            if kept:
                cand_reports["reports_with_candidates"] += 1
    summary = dict(
        schema="ct_report_preferences_summary", policy=POLICY, created=time.strftime("%F %T"),
        index=args.index, findings=args.findings, seed=args.seed, max_omissions=args.max_omissions,
        train_reports=len(train), val_reports_excluded=len(val_uids),
        pairs_total=pairs_written, pairs_by_type=dict(type_counts),
        reports_covered=dict(covered), omission_source_states=dict(omission_state),
        donor_classes=len(donors), donor_sentences=len(donor_text),
        exclusion_reasons=dict(sorted(issues.items())),
        attribute_candidates_before_cap=dict(cand_counts),
        attribute_candidates_written=dict(cand_counts_capped),
        attribute_candidate_reports=cand_reports["reports_with_candidates"],
        max_attr_per_type=args.max_attr_per_type, max_attr_per_report=args.max_attr_per_report,
        outputs=dict(pairs=str(pairs_path), attribute_candidates=str(cand_path)),
        unmentioned_is_negative=False, val_used_for_pairs=False,
        limitations=[
            "Preferences are relative to documented Findings and LLM extraction quotes, not image truth.",
            "Whole-sentence deletion can remove several facts; uncertainty is never rewritten.",
            "Explicit-denial controls insert a foreign asserted sentence; no whole-study denial flag exists for CT, "
            "scope-limited denials (side/segment words) are excluded instead.",
            "Unmentioned findings are never negatives.",
        ],
        seconds=round(time.time() - began, 1))
    (out / "base_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("pairs_total", "pairs_by_type", "reports_covered",
                                              "attribute_candidates_written", "attribute_candidate_reports")},
                     indent=1))



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("findings", help="H5 -> one sqlite with every train/val Findings target")
    f.add_argument("--index", required=True)
    f.add_argument("--h5_root", required=True)
    f.add_argument("--output", required=True)
    f.add_argument("--workers", type=int, default=16)
    b = sub.add_parser("build", help="omission / repetition / denial pairs + attribute candidates")
    b.add_argument("--index", required=True)
    b.add_argument("--findings", required=True)
    b.add_argument("--output_dir", required=True)
    b.add_argument("--max_omissions", type=int, default=3)
    b.add_argument("--max_attr_per_type", type=int, default=1)
    b.add_argument("--max_attr_per_report", type=int, default=4)
    b.add_argument("--seed", type=int, default=20261011)
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    {"findings": cmd_findings, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
