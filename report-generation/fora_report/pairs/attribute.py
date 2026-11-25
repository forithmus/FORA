"""CT DPO attribute pairs: CPU verifier of the Gemma minimal attribute corruptions (paper: "Preference optimization").

ONE attribute (laterality, location, severity, size, number or certainty) of ONE finding sentence is corrupted by
Gemma-4-31B-it (``gemma_verifier.py corrupt``), screened by the Gemma judge (``gemma_verifier.py judge`` +
``filter``), and accepted here only if ``verify_attribute_edit`` finds exactly one local attribute edit with the rest
of the sentence unchanged. Accepted edits become ``attribute`` pairs (chosen = source Findings, rejected = the same
Findings with the corrupted sentence).

  python -m fora_report.pairs.attribute --candidates attribute_candidates.jsonl.gz \
      --gemma_outputs 'gemma_attr_judged/rank_*.jsonl' --findings findings.sqlite --output attribute_pairs.jsonl.gz
"""
from __future__ import annotations

import argparse
from collections import Counter
import difflib
import glob
import gzip
import json
from pathlib import Path
import re
import sqlite3

from .base import edited_target, make_pair
from .text import (AGREEMENT_WORDS, HEDGE_RE, HEDGE_WORDS, INVALID_ANATOMY, LOCATION_WORDS, NUMBER_WORDS,
                   PROTECTED_SIDE, SEVERITY_WORDS, SIDE_WORDS, jdump, sha, tokens)


def _number(token):
    try:
        return float(token.replace(",", "."))
    except ValueError:
        return None


def _plural_variant(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    return a != b and (a + "s" == b or b + "s" == a or a + "es" == b or b + "es" == a
                       or (a.endswith("y") and a[:-1] + "ies" == b) or (b.endswith("y") and b[:-1] + "ies" == a))


def verify_attribute_edit(original: str, corrupted: str, attribute_type: str, max_changed_tokens=6):
    """Return (ok, reason, detail).  Exactly one attribute span changed, rest identical."""
    if not isinstance(corrupted, str) or not corrupted.strip():
        return False, "empty", None
    corrupted = corrupted.strip()
    if corrupted == original:
        return False, "no_op", None
    if "  " in corrupted or "\n" in corrupted:
        return False, "whitespace_artifact", None
    if corrupted[-1:] != original[-1:]:
        return False, "terminal_punctuation_changed", None
    if corrupted[:1].isupper() != original[:1].isupper():
        return False, "initial_case_changed", None
    if INVALID_ANATOMY.search(corrupted) and not INVALID_ANATOMY.search(original):
        return False, "invalid_anatomy_created", None
    a, b = tokens(original), tokens(corrupted)
    ops = [op for op in difflib.SequenceMatcher(a=[t.lower() for t in a], b=[t.lower() for t in b],
                                                autojunk=False).get_opcodes() if op[0] != "equal"]
    if not ops:
        return False, "case_only_change", None
    removed = [t for _tag, i1, i2, _j1, _j2 in ops for t in a[i1:i2]]
    added = [t for _tag, _i1, _i2, j1, j2 in ops for t in b[j1:j2]]
    span = (ops[0][1], ops[-1][2])
    detail = dict(removed=removed, added=added, regions=len(ops))
    if len(removed) > max_changed_tokens or len(added) > max_changed_tokens:
        return False, "too_many_tokens_changed", detail
    if span[1] - span[0] > max_changed_tokens + 3:
        return False, "edit_not_local", detail
    lower_r, lower_a = [t.lower() for t in removed], [t.lower() for t in added]

    def core(words, vocab):
        """Tokens that are neither in the type vocabulary nor agreement/plural adjustments."""
        rest = []
        for t in words:
            if t in vocab or t in AGREEMENT_WORDS:
                continue
            rest.append(t)
        return rest

    def plural_only(xs, ys):
        xs, ys = list(xs), list(ys)
        for x in list(xs):
            for y in ys:
                if _plural_variant(x, y):
                    xs.remove(x); ys.remove(y); break
        return xs, ys

    if attribute_type == "size":
        nums_r = [_number(t) for t in removed if _number(t) is not None]
        nums_a = [_number(t) for t in added if _number(t) is not None]
        other_r = [t for t in lower_r if _number(t) is None]
        other_a = [t for t in lower_a if _number(t) is None]
        units = {"mm", "cm", "ml", "cc"}
        if not nums_r and not (set(other_r) | set(other_a)) <= units:
            return False, "size_no_number_changed", detail
        if (set(other_r) | set(other_a)) - units:
            return False, "size_edit_touches_non_numeric_words", detail
        if len(nums_r) != len(nums_a):
            return False, "size_dimension_count_changed", detail
        for x, y in zip(nums_r, nums_a):
            if x <= 0 or y <= 0:
                return False, "size_nonpositive", detail
            if max(x, y) / min(x, y) < 1.5:
                return False, "size_change_too_small_to_be_clinically_wrong", detail
        if not nums_r and not (set(other_r) <= units and set(other_a) <= units):
            return False, "size_unit_edit_invalid", detail
        return True, "ok", detail
    vocab = {"laterality": SIDE_WORDS | {"the"}, "location": LOCATION_WORDS | SIDE_WORDS, "severity": SEVERITY_WORDS,
             "number": NUMBER_WORDS, "certainty": HEDGE_WORDS}.get(attribute_type)
    if vocab is None:
        return False, "unknown_attribute_type", detail
    if attribute_type == "number":
        numeric_r = [t for t in lower_r if t.isdigit()]
        numeric_a = [t for t in lower_a if t.isdigit()]
        lower_r = [t for t in lower_r if not t.isdigit()]
        lower_a = [t for t in lower_a if not t.isdigit()]
        if numeric_r or numeric_a:
            if len(numeric_r) != len(numeric_a):
                return False, "number_count_token_mismatch", detail
    rest_r, rest_a = plural_only(core(lower_r, vocab), core(lower_a, vocab))
    if attribute_type == "location":
        # segment / vertebral-level identifiers may change ("segment 6" -> "segment 2", "L4" -> "L1")
        rest_r = [t for t in rest_r if not re.fullmatch(r"[ctlsd]?\d{1,2}", t)]
        rest_a = [t for t in rest_a if not re.fullmatch(r"[ctlsd]?\d{1,2}", t)]
    if rest_r or rest_a:
        return False, "edit_outside_attribute_vocabulary", dict(detail, outside=rest_r + rest_a)
    if attribute_type == "laterality":
        if not ({t for t in lower_r} & SIDE_WORDS and {t for t in lower_a} & SIDE_WORDS):
            return False, "laterality_word_not_swapped", detail
        if PROTECTED_SIDE.search(original) and not PROTECTED_SIDE.search(corrupted):
            return False, "protected_fixed_side_anatomy_changed", detail
    if attribute_type == "severity" and not (set(lower_r) | set(lower_a)) & (SEVERITY_WORDS - AGREEMENT_WORDS - {"a", "an", "to"}):
        return False, "severity_word_not_changed", detail
    if attribute_type == "certainty" and bool(HEDGE_RE.search(original)) == bool(HEDGE_RE.search(corrupted)) \
            and not (set(lower_r) | set(lower_a)) & {"definite", "definitely", "possible", "possibly", "probable",
                                                    "probably", "likely", "suspicious", "suspected", "consistent",
                                                    "compatible", "questionable", "equivocal", "excluded"}:
        return False, "certainty_not_changed", detail
    if attribute_type == "number" and not ((set(lower_r) | set(lower_a)) & (NUMBER_WORDS - {"a", "an", "of"})
                                           or numeric_r):
        return False, "number_word_not_changed", detail
    return True, "ok", detail


def cmd_attributes(args):
    """Merge Gemma corruption outputs, verify, write attribute pairs jsonl."""
    out = Path(args.output)
    if out.exists() and not args.overwrite:
        raise FileExistsError(out)
    cands = {}
    with gzip.open(args.candidates, "rt", encoding="utf-8") as stream:
        for line in stream:
            rec = json.loads(line)
            cands[rec["cand_id"]] = rec
    fnd = sqlite3.connect(f"file:{args.findings}?mode=ro", uri=True)
    reasons, kinds, reports = Counter(), Counter(), set()
    seen = set()
    files = sorted(glob.glob(args.gemma_outputs))
    if not files:
        raise FileNotFoundError(args.gemma_outputs)
    examples = []
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=3) as stream:
        for path in files:
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    res = json.loads(line)
                    cand = cands.get(res.get("cand_id"))
                    if cand is None or res["cand_id"] in seen:
                        reasons["unknown_or_duplicate_candidate"] += 1
                        continue
                    seen.add(res["cand_id"])
                    if not res.get("applicable", False):
                        reasons["llm_not_applicable"] += 1
                        continue
                    corrupted = res.get("corrupted_sentence")
                    ok, reason, detail = verify_attribute_edit(cand["sentence"], corrupted, cand["attribute_type"])
                    reasons[reason] += 1
                    if not ok:
                        continue
                    row = fnd.execute("SELECT findings, patient_uid, h5_id, split FROM findings WHERE report_uid=?",
                                      (cand["report_uid"],)).fetchone()
                    if row is None or row[3] != "train" or sha(row[0]) != cand["chosen_sha256"]:
                        reasons["source_changed_or_not_train"] += 1
                        continue
                    study = dict(report_uid=cand["report_uid"], patient_id=row[1], h5_id=row[2], split="train",
                                 findings=row[0])
                    pair = make_pair(study, "attribute", edited_target(row[0], cand["sentence"], corrupted.strip()),
                                     dict(rule="llm_minimal_single_attribute_corruption_cpu_verified",
                                          model=res.get("model", "gemma-4-31b-it"), cand_id=cand["cand_id"],
                                          original_sentence=cand["sentence"], corrupted_sentence=corrupted.strip(),
                                          llm_original_span=res.get("original_span"),
                                          llm_new_span=res.get("new_span"), diff=detail,
                                          source_state=cand["source_state"]),
                                     changed=cand["sentence"], attribute_type=cand["attribute_type"],
                                     class_ids=cand["class_ids"])
                    if pair is None:
                        reasons["no_op_pair"] += 1
                        continue
                    stream.write(jdump(pair) + "\n")
                    kinds[cand["attribute_type"]] += 1
                    reports.add(cand["report_uid"])
                    if len(examples) < 20:
                        examples.append((cand["attribute_type"], cand["sentence"], corrupted))
    summary = dict(candidates=len(cands), gemma_results=len(seen), accepted_by_type=dict(kinds),
                   accepted_total=sum(kinds.values()), reports_covered=len(reports),
                   verifier_reasons=dict(reasons.most_common()), examples=examples)
    Path(str(out).replace(".jsonl.gz", "") + "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=1))



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True, help="attribute_candidates.jsonl.gz from `base build`")
    ap.add_argument("--gemma_outputs", required=True, help="glob of judge-filtered Gemma corruption rank files")
    ap.add_argument("--findings", required=True, help="findings sqlite from `base findings`")
    ap.add_argument("--output", required=True)
    ap.add_argument("--overwrite", action="store_true")
    cmd_attributes(ap.parse_args(argv))


if __name__ == "__main__":
    main()
