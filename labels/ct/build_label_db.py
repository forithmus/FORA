"""Build the CT label database: one int8 state per (report, hierarchy node) (Methods, report labels).

Sources, in order:
1. global-vocabulary extraction shards (``--shards``); retry runs (``--retry-shards``, in order) replace rows of the same
   (ReportUID, protocol_class). Rows with verdict pass, p3_unparsed, p3_repaired or p1_repaired are used.
   Every emitted leaf goes through ``canonicalizer`` (hierarchy + curated alias table). With the alias table, a leaf named
   "Cardiomegaly denied" is read as a denial of Cardiomegaly. Out-of-vocabulary ``other`` findings whose name is a class
   become asserted. With ``--adjudication``, MAP verdicts replace the label and polarity of the named leaf.
2. per-protocol-vocabulary extraction shards (``--union-shards``): their asserted/uncertain labels are added to reports of
   source 1 (their denials are not used).
3. recall pass (``--recall``): added positives.
4. malignancy check (``--malignancy``): history_only / not_supported malignancy labels are removed from every positive
   source of the report.
5. adjudication JUDGE verdicts (``--adjudication``): absent removes the class and its descendants from every positive
   source; possible turns asserted into uncertain.
Then ``fora.labels.report_state`` propagates (asserted/uncertain up, denied down, positive wins; unmentioned = 3).
Remaining positive/denial conflicts resolve to positive. ``--dump-items`` writes the MAP and JUDGE items for
``adjudicate.py`` instead of using verdicts.

Split and case ids come from ``--cases`` (JSONL with "id" and "report"; split = train when the case id starts with
"train", else val). Reports without a case are skipped.

Outputs: labels.npz (reports, cases, split, nodes, state[int8]), heads.json (head tiers, ``fora.labels.assign_heads``),
summary.json (counts).
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

from fora.labels import assign_heads, canonicalizer, load_alias_table, load_hierarchy, report_state

USED_VERDICTS = ("pass", "p3_unparsed", "p3_repaired", "p1_repaired")
UNION_VERDICTS = ("pass", "p3_unparsed")
POLARITIES = ("asserted", "uncertain", "denied")


def empty() -> dict[str, set[int]]:
    return {"asserted": set(), "uncertain": set(), "denied": set()}


def read_cases(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    split, case_of = {}, {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        case_of[d["report"]] = d["id"]
        split[d["report"]] = "train" if d["id"].startswith("train") else "val"
    return split, case_of


def read_tab_lines(path: Path | None, kind: str):
    if not path:
        return
    for line in open(path, encoding="utf-8"):
        if not line.startswith(kind + "\t"):
            continue
        try:
            yield json.loads(line[len(kind) + 1:])
        except Exception:
            continue


def build(a: argparse.Namespace) -> dict:
    csv.field_size_limit(10 ** 9)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    H = load_hierarchy(a.hierarchy)
    alias = load_alias_table(a.aliases)
    canon = canonicalizer(H, alias)
    names, idx, K = H.names, H.index, len(H)
    split, case_of = read_cases(Path(a.cases))
    stat, unm = collections.Counter(), collections.Counter()
    dump = open(a.dump_items, "w", encoding="utf-8") if a.dump_items else None
    adj_map, adj_judge = {}, {}
    for d in read_tab_lines(a.adjudication, "MAP"):
        adj_map[(d["ReportUID"], d["name"])] = (d.get("label"), d.get("polarity"))
    for d in read_tab_lines(a.adjudication, "JUDGE"):
        adj_judge[(d["ReportUID"], d["label"])] = d.get("verdict")

    rows_all = {}
    for k, d in enumerate([a.shards] + list(a.retry_shards)):
        for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
            for r in csv.DictReader(open(f, encoding="utf-8")):
                if k:
                    stat["retry_rows"] += 1
                    stat["retry_verdict_" + r["pathology_extraction_verdict"]] += 1
                rows_all[(r["ReportUID"], r["protocol_class"])] = r
    rows = collections.defaultdict(list)
    for r in rows_all.values():
        stat["rows"] += 1
        stat["verdict_" + r["pathology_extraction_verdict"]] += 1
        u = r["ReportUID"]
        if u not in split:
            stat["rows_without_cache"] += 1
            continue
        if r["pathology_extraction_verdict"] not in USED_VERDICTS:
            continue
        L = empty()
        for k in POLARITIES:
            for x in json.loads(r["pathology_" + k] or "[]"):
                lf = x.get("leaf", "").strip()
                if (u, lf) in adj_map:
                    lab, pol = adj_map[(u, lf)]
                    c = canon(lab) if lab else None
                    if c is not None and pol in POLARITIES:
                        L[pol].add(idx[c])
                        stat["adj_mapped_" + pol] += 1
                    else:
                        stat["adj_mapped_none"] += 1
                    continue
                if dump is not None and lf and (lf in alias or lf == "Cardiomegaly denied" or canon(lf) is None):
                    dump.write("MAP\t" + json.dumps({"ReportUID": u, "name": lf, "polarity": k,
                                                     "quote": (x.get("quote") or "")[:300]}, ensure_ascii=False) + "\n")
                if alias and lf == "Cardiomegaly denied":
                    L["denied"].add(idx["Cardiomegaly"])
                    stat["flip_cardiomegaly_denied"] += 1
                    continue
                c = canon(lf)
                if c is None:
                    unm[lf] += 1
                    continue
                if lf in alias:
                    stat["alias_mapped"] += 1
                L[k].add(idx[c])
        for x in json.loads(r.get("pathology_other") or "[]"):
            c = canon(str(x.get("finding", "")))
            if c is not None:
                L["asserted"].add(idx[c])
                stat["other_recovered"] += 1
        rows[u].append(L)

    if a.union_shards:
        for f in sorted(glob.glob(os.path.join(a.union_shards, "*.csv"))):
            for r in csv.DictReader(open(f, encoding="utf-8")):
                u = r["ReportUID"]
                if u not in rows or r["pathology_extraction_verdict"] not in UNION_VERDICTS:
                    continue
                L = empty()
                for k in ("asserted", "uncertain"):
                    for x in json.loads(r["pathology_" + k] or "[]"):
                        c = canon(x.get("leaf", ""))
                        if c is not None:
                            L[k].add(idx[c])
                rows[u].append(L)
                stat["union_rows"] += 1

    for d in read_tab_lines(a.recall, "REC"):
        u = d["ReportUID"]
        if u not in rows:
            continue
        L = empty()
        for x in d["add"]:
            c = canon(x["leaf"])
            if c is not None:
                L["asserted" if x["polarity"] == "asserted" else "uncertain"].add(idx[c])
                stat["recall_added_" + x["polarity"]] += 1
        if L["asserted"] or L["uncertain"]:
            rows[u].append(L)
            stat["recall_reports"] += 1

    if a.malignancy:
        drop = collections.defaultdict(set)
        for d in read_tab_lines(a.malignancy, "CAN"):
            if d["verdict"] in ("history_only", "not_supported") and d["label"] in idx:
                drop[d["ReportUID"]].add(idx[d["label"]])
                stat["drop_" + d["verdict"]] += 1
        for u, ds in drop.items():
            for L in rows.get(u, []):
                for k in ("asserted", "uncertain"):
                    n0 = len(L[k])
                    L[k] -= ds
                    stat["dropped_source_labels"] += n0 - len(L[k])

    for (u, lab), v in adj_judge.items():
        if u not in rows or lab not in idx:
            continue
        i = idx[lab]
        kill = {i} | H.descendants[i]
        for L in rows[u]:
            if v == "absent":
                for k in ("asserted", "uncertain"):
                    n0 = len(L[k])
                    L[k] -= kill
                    stat["judge_absent_removed"] += n0 - len(L[k])
            elif v == "possible" and i in L["asserted"]:
                L["asserted"].discard(i)
                L["uncertain"].add(i)
                stat["judge_to_uncertain"] += 1
        stat["judge_" + str(v)] += 1

    rep = sorted(rows)
    S = np.empty((len(rep), K), np.int8)
    conf = 0
    for ri, u in enumerate(rep):
        S[ri], c = report_state(rows[u], H.ancestors, H.descendants, K)
        conf += len(c)
        if dump is not None:
            for d in c:
                dump.write("JUDGE\t" + json.dumps({"ReportUID": u, "label": names[d], "why": "conflict"}) + "\n")
    if dump is not None:
        dump.close()

    tr = np.array([split[u] == "train" for u in rep])
    va = ~tr
    pos = lambda m: ((S[m] == 1) | (S[m] == 2)).sum(0)  # noqa: E731
    neg = lambda m: ((S[m] == 0) | (S[m] == 3)).sum(0)  # noqa: E731
    heads = assign_heads(names, {n: H.kind(n) for n in names}, {n: H.node_ancestors(n) for n in names},
                         pos(tr), neg(tr), pos(va), neg(va))
    np.savez_compressed(out / "labels.npz", reports=np.array(rep), cases=np.array([case_of[u] for u in rep]),
                        split=np.array([split[u] for u in rep]), nodes=np.array(names), state=S)
    json.dump(heads, open(out / "heads.json", "w"), indent=0)
    summary = dict(stat, reports=len(rep), train=int(tr.sum()), val=int(va.sum()), nodes=K, own_heads=len(heads["own"]),
                   attached=len(heads["attached"]), rare_orphans=len(heads["rare_orphans"]), no_head=len(heads["no_head"]),
                   pos_denied_conflicts=conf, unmapped_labels=sum(unm.values()), unmapped_top=unm.most_common(15),
                   mean_pos_per_report=float(((S == 1) | (S == 2)).sum(1).mean()),
                   mean_neg_per_report=float(((S == 0) | (S == 3)).sum(1).mean()),
                   mean_explicit_neg_per_report=float((S == 0).sum(1).mean()))
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True, help="global-vocabulary extraction shard dir")
    ap.add_argument("--retry-shards", nargs="*", default=[], help="retry shard dirs, later replaces earlier")
    ap.add_argument("--union-shards", help="per-protocol-vocabulary extraction shard dir (positives only)")
    ap.add_argument("--hierarchy", required=True)
    ap.add_argument("--aliases", help="curated alias table (data/ct/unmapped_aliases.json)")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--recall", help="merged recall-pass REC lines")
    ap.add_argument("--malignancy", help="malignancy-check CAN lines")
    ap.add_argument("--adjudication", help="adjudication JUDGE/MAP lines")
    ap.add_argument("--dump-items", help="write MAP/JUDGE items for adjudicate.py")
    ap.add_argument("--output", required=True)
    s = build(ap.parse_args())
    print(json.dumps(s, indent=0)[:2500])


if __name__ == "__main__":
    main()
