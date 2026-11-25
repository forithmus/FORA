"""Recall pass: a second look for positive findings the three-pass extraction missed (Methods, report labels).

Per report, two generators run on the global vocabulary:
  A ("missing", prompts/ct_recall/missing_*): the report plus the labels it already has -> findings not yet covered;
  B ("relist", prompts/ct_recall/relist_*): every abnormal statement of the report with all labels it supports.
Their proposals are unioned (first occurrence of a label wins) and filtered: the label must be a vocabulary name,
polarity asserted|uncertain, not already positive, and the quote must occur in the report (case/punctuation normalised).
Survivors go to one strict VERIFY call (prompts/ct_recall/verify_*); unparsed or unjudged proposals are rejected.
The pass only adds positives.

"Already labelled" = state 1/2 rows of the label DB built from the extraction alone (``build_label_db`` without
recall/malignancy/adjudication inputs). Eligible rows: findings >= 20 characters, protocol class 1..29.
Output: one ``REC\t{json}`` line per report (ReportUID, protocol_class, status ok|unparsed|too_long, add[{leaf, polarity,
quote}], dropped_n counts). ``merge_recall`` combines a first run with a retry run.

    python -m labels.ct.recall_pass --config labels/configs/ct_recall.json --input global_input.csv \
        --labels labels_extraction_only/labels.npz --model <snapshot> --output recall_rank00.recs
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from labels.ct.three_pass import Protocols
from labels.llm import (CONTEXT_LENGTH_SAFETY_TOKENS, GREEDY_TEMPERATURE, GREEDY_TOP_P, inferer_from_config,
                        load_config, parse_first_last_brace, rank_world, read_prompt, vocab_lines)

MIN_FINDINGS_CHARS = 20


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def clean_label(x) -> str:
    return str(x or "").split(" — ")[0].strip()


def screen(items, names: set[str], have: set[str], text: str, seen: set[str]) -> tuple[list, list]:
    """Deterministic filter of statement-level proposals [{quote, labels[{label, polarity}]}]."""
    add, drop = [], []
    for st in items:
        if not isinstance(st, dict):
            continue
        q = str(st.get("quote") or "")
        for x in st.get("labels") if isinstance(st.get("labels"), list) else []:
            if not isinstance(x, dict):
                continue
            lab, pol = clean_label(x.get("label")), x.get("polarity")
            if lab in seen:
                continue
            seen.add(lab)
            why = ("not_in_vocab" if lab not in names else "bad_polarity" if pol not in ("asserted", "uncertain") else
                   "already" if lab in have else "quote_not_verbatim" if len(norm(q)) < 3 or norm(q) not in text else None)
            (drop if why else add).append({"leaf": lab, "polarity": pol, "quote": q, **({"why": why} if why else {})})
    return add, drop


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/ct_recall.json")
    ap.add_argument("--input", required=True)
    ap.add_argument("--labels", required=True, help="labels.npz of the extraction-only label DB")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--only-reports", default="", help="file with one ReportUID per line (retry run)")
    ap.add_argument("--max-tokens", type=int, help="override (retry run: 6144)")
    ap.add_argument("--rank", type=int)
    ap.add_argument("--world", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config)
    max_tokens = a.max_tokens or cfg["max_tokens"]
    rank, world = rank_world(a.rank, a.world)
    log = lambda m: print(f"[rank {rank}] {m}", file=sys.stderr, flush=True)  # noqa: E731
    sys_b, user_b = read_prompt(cfg["prompts"]["relist_system"]), read_prompt(cfg["prompts"]["relist_user"])
    sys_a, user_a = read_prompt(cfg["prompts"]["missing_system"]), read_prompt(cfg["prompts"]["missing_user"])
    verify, verify_user = read_prompt(cfg["prompts"]["verify_system"]), read_prompt(cfg["prompts"]["verify_user"])
    df = pd.read_csv(a.input, low_memory=False)
    pc = pd.to_numeric(df.protocol_class, errors="coerce")
    df = df[(df.findings.astype(str).str.strip().str.len() >= MIN_FINDINGS_CHARS) & pc.between(1, 29)].reset_index(drop=True)
    if a.only_reports:
        keep = set(Path(a.only_reports).read_text().split())
        df = df[df.ReportUID.isin(keep)].reset_index(drop=True)
    rows = list(df.iloc[rank::world].itertuples(index=False))
    z = np.load(a.labels)
    nodes = [str(n) for n in z["nodes"]]
    rowof = {str(r): i for i, r in enumerate(z["reports"])}
    st = z["state"]
    protocols = Protocols(cfg["protocols"], cfg["vocabulary"])
    leaves = protocols.global_leaves
    names = {lf["name"] for lf in leaves}
    syn = {lf["name"]: (lf.get("syn") or "").strip() for lf in leaves}
    system_a = sys_a + vocab_lines(leaves)
    system_b = sys_b + vocab_lines(leaves)
    log(f"{len(rows)} rows of {len(df)}; {len(leaves)} labels")
    inf = inferer_from_config(a.model, cfg, log)
    samp = inf.make_sampling(temperature=GREEDY_TEMPERATURE, top_p=GREEDY_TOP_P, max_tokens=max_tokens)
    fits = lambda m: inf.prompt_fits(m, max_tokens, safety_tokens=CONTEXT_LENGTH_SAFETY_TOKENS)[0]  # noqa: E731
    out = open(a.output, "w", encoding="utf-8")
    t0 = time.time()
    n_add = n_drop = n_fail = 0
    for b in range(0, len(rows), cfg["batch"]):
        msgs_b, msgs_a, meta = [], [], []
        for r in rows[b:b + cfg["batch"]]:
            mod = protocols.load(int(float(r.protocol_class)))
            i = rowof.get(str(r.ReportUID))
            have = sorted(nodes[k] for k in np.flatnonzero(np.isin(st[i], (1, 2)))) if i is not None else []
            imp = r.impression if isinstance(r.impression, str) else ""
            mb = [{"role": "system", "content": system_b},
                  {"role": "user", "content": user_b.format(title=mod["title"], region=mod["region"], findings=r.findings,
                                                            impression=imp or "(none)")}]
            ma = [{"role": "system", "content": system_a},
                  {"role": "user", "content": user_a.format(title=mod["title"], region=mod["region"], findings=r.findings,
                                                            impression=imp or "(none)", have="; ".join(have) or "(none)")}]
            if fits(mb) and fits(ma):
                msgs_b.append(mb)
                msgs_a.append(ma)
                meta.append((r, set(have), norm(str(r.findings) + " " + imp)))
            else:
                out.write("REC\t" + json.dumps({"ReportUID": r.ReportUID, "protocol_class": int(float(r.protocol_class)),
                                                "add": [], "dropped": [], "status": "too_long"}) + "\n")
        outs_b = inf.chat(msgs_b, samp) if msgs_b else []
        outs_a = inf.chat(msgs_a, samp) if msgs_a else []
        done = []
        for (r, have, text), raw_b, raw_a in zip(meta, outs_b, outs_a):
            pb, pa = parse_first_last_brace(raw_b), parse_first_last_brace(raw_a)
            items_a = pa.get("missing") if isinstance(pa, dict) else None   # generator A: one statement per item
            extra = [{"quote": x.get("quote"), "labels": [x]} for x in (items_a if isinstance(items_a, list) else [])
                     if isinstance(x, dict)]
            items_b = pb.get("statements") if isinstance(pb, dict) else None
            add, drop = screen((items_b if isinstance(items_b, list) else []) + extra, names, have, text, set())
            n_drop += len(drop)
            n_fail += (pb is None) + (pa is None)
            done.append((r, add, drop, (pb is not None) and (pa is not None)))
        vmsgs, vidx = [], []
        for k, (r, add, drop, ok) in enumerate(done):
            if add:
                imp = r.impression if isinstance(r.impression, str) else ""
                props = "\n".join(f"- {x['leaf']}" + (f" — {syn[x['leaf']]}" if syn.get(x['leaf']) else "")
                                  + f" | {x['polarity']} | \"{x['quote']}\"" for x in add)
                vmsgs.append([{"role": "system", "content": verify},
                              {"role": "user", "content": verify_user.format(findings=r.findings, impression=imp or "(none)",
                                                                             props=props)}])
                vidx.append(k)
        vouts = inf.chat(vmsgs, samp) if vmsgs else []
        for k, raw in zip(vidx, vouts):
            r, add, drop, ok = done[k]
            v = parse_first_last_brace(raw)
            keep = {}
            for x in (v.get("verdicts") if isinstance(v, dict) and isinstance(v.get("verdicts"), list) else []):
                if isinstance(x, dict):
                    keep[clean_label(x.get("label", ""))] = (x.get("keep") is True, str(x.get("reason", "")))
            kept = []
            for x in add:   # unparsed verdict or label not judged -> reject
                kp, why = keep.get(x["leaf"], (False, "not_judged" if v is not None else "verify_unparsed"))
                (kept if kp else drop).append(x if kp else {**x, "why": "verify: " + why})
            done[k] = (r, kept, drop, ok)
        for r, add, drop, ok in done:
            n_add += len(add)
            out.write("REC\t" + json.dumps({"ReportUID": r.ReportUID, "protocol_class": int(float(r.protocol_class)),
                                            "status": "ok" if ok else "unparsed",
                                            "dropped_n": dict(Counter(d.get("why", "")[:12] for d in drop)),
                                            "add": [{**x, "quote": x["quote"][:200]} for x in add]},
                                           ensure_ascii=False) + "\n")
        out.flush()
        log(f"{min(b + cfg['batch'], len(rows))}/{len(rows)} rows | +{n_add} added, {n_drop} dropped, {n_fail} unparsed | "
            f"{time.time() - t0:.0f}s")
    out.close()


if __name__ == "__main__":
    main()
