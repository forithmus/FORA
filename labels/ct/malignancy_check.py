"""LLM check of every malignancy label (Methods, report labels).

For each report whose label DB row has a positive malignancy class (name matches ``MALIGNANT``), Gemma-4-31B-it reads
the report and judges each such label: ``current`` (tumour, metastasis, recurrence or residual described or suspected
now, or a known cancer mentioned without being stated as removed), ``history_only`` (stated as operated / removed /
treated, no current tumour) or ``not_supported``. Unparsed or unjudged labels are kept. The label build removes
``history_only`` and ``not_supported`` labels from every positive source of that report.

Input DB: the label DB built from extraction + alias table + recall pass. Output: one ``CAN\t{ReportUID, label,
verdict, reason}`` line per (report, label).

    python -m labels.ct.malignancy_check --input global_input.csv --labels labels_with_recall/labels.npz \
        --model <snapshot> --output malignancy_rank00.can
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from labels.llm import (CONTEXT_LENGTH_SAFETY_TOKENS, GREEDY_TEMPERATURE, GREEDY_TOP_P, inferer_from_config,
                        load_config, parse_first_last_brace, rank_world, read_prompt)

MALIGNANT = re.compile(r"cancer|carcinom|malignan|lymphoma|sarcoma|melanoma|myeloma|leuk|metasta|mesothelioma|blastoma|"
                       r"stromal tumour|neuroendocrine|choriocarcinoma|seminoma|germ cell", re.I)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/ct_malignancy.json")
    ap.add_argument("--input", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--only-reports", default="")
    ap.add_argument("--rank", type=int)
    ap.add_argument("--world", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config)
    rank, world = rank_world(a.rank, a.world)
    log = lambda m: print(f"[rank {rank}] {m}", file=sys.stderr, flush=True)  # noqa: E731
    system, user = read_prompt(cfg["prompts"]["system"]), read_prompt(cfg["prompts"]["user"])
    z = np.load(a.labels)
    nodes = [str(n) for n in z["nodes"]]
    st = z["state"]
    reports = [str(x) for x in z["reports"]]
    mcols = [k for k, n in enumerate(nodes) if MALIGNANT.search(n)]
    pos = np.isin(st[:, mcols], (1, 2))
    rows_with = np.flatnonzero(pos.any(1))
    reps = [reports[i] for i in rows_with]
    labs = {reports[i]: [nodes[mcols[k]] for k in np.flatnonzero(pos[i])] for i in rows_with}
    df = pd.read_csv(a.input, low_memory=False).set_index("ReportUID")
    if a.only_reports:
        keep = set(Path(a.only_reports).read_text().split())
        reps = [u for u in reps if u in keep]
    mine = [u for u in sorted(reps) if u in df.index][rank::world]
    log(f"{len(mine)} reports of {len(reps)} with malignancy labels ({len(mcols)} malignancy classes)")
    inf = inferer_from_config(a.model, cfg, log)
    samp = inf.make_sampling(temperature=GREEDY_TEMPERATURE, top_p=GREEDY_TOP_P, max_tokens=cfg["max_tokens"])
    out = open(a.output, "w", encoding="utf-8")
    t0 = time.time()
    cnt: dict[str, int] = {}
    for b in range(0, len(mine), cfg["batch"]):
        msgs, meta = [], []
        for u in mine[b:b + cfg["batch"]]:
            r = df.loc[u]
            imp = r.impression if isinstance(r.impression, str) else ""
            m = [{"role": "system", "content": system},
                 {"role": "user", "content": user.format(findings=r.findings, impression=imp or "(none)",
                                                         labels="\n".join("- " + x for x in labs[u]))}]
            if inf.prompt_fits(m, cfg["max_tokens"], safety_tokens=CONTEXT_LENGTH_SAFETY_TOKENS)[0]:
                msgs.append(m)
                meta.append(u)
            else:
                for x in labs[u]:
                    out.write("CAN\t" + json.dumps({"ReportUID": u, "label": x, "verdict": "too_long"}) + "\n")
        outs = inf.chat(msgs, samp) if msgs else []
        for u, raw in zip(meta, outs):
            p = parse_first_last_brace(raw)
            got = {}
            for v in (p.get("verdicts") if isinstance(p, dict) and isinstance(p.get("verdicts"), list) else []):
                if isinstance(v, dict):
                    got[str(v.get("label", "")).strip()] = (str(v.get("verdict", "")), str(v.get("reason", ""))[:120])
            for x in labs[u]:   # only explicit verdicts remove a label
                vd, why = got.get(x, ("unjudged" if p is not None else "unparsed", ""))
                cnt[vd] = cnt.get(vd, 0) + 1
                out.write("CAN\t" + json.dumps({"ReportUID": u, "label": x, "verdict": vd, "reason": why},
                                               ensure_ascii=False) + "\n")
        out.flush()
        log(f"{min(b + cfg['batch'], len(mine))}/{len(mine)} reports | {cnt} | {time.time() - t0:.0f}s")
    out.close()


if __name__ == "__main__":
    main()
