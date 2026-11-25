"""LLM adjudication (JUDGE) and mapping (MAP) of label-build items (Methods, report labels).

Items (written by ``build_label_db --dump-items``):
  JUDGE {ReportUID, label}: a class that is both positive and explicitly denied in the report, or a recall-pass
        "Nasal cavity abnormality" item. Gemma-4-31B-it answers present / possible / absent.
  MAP   {ReportUID, name, polarity, quote}: a label name the extraction emitted that is not a vocabulary class or was
        resolved only by the curated alias table. Gemma picks the vocabulary label (or none) and the polarity from the
        report; every mapped label is then checked independently with the JUDGE prompt and kept only if the verdict agrees
        with the mapped polarity (asserted: present; uncertain: possible or present; denied: absent).
Output lines: ``JUDGE\t{ReportUID, label, verdict}`` and ``MAP\t{ReportUID, name, label, polarity, check}``.
Unparsed answers produce no line; the build then keeps its rule-based behaviour for that item.

    python -m labels.ct.adjudicate --items adj_items.txt --input global_input.csv --model <snapshot> --output adj_rank00.out
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from labels.ct.three_pass import Protocols
from labels.llm import (CONTEXT_LENGTH_SAFETY_TOKENS, GREEDY_TEMPERATURE, GREEDY_TOP_P, inferer_from_config,
                        load_config, rank_world, read_prompt)

OK_FOR = {"asserted": ("present",), "uncertain": ("possible", "present"), "denied": ("absent",)}


def parse(raw: str):
    """First complete JSON object in the reply."""
    s = raw.strip()
    i = s.find("{")
    if i < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(s[i:])[0]
    except Exception:
        return None


def clean(x) -> str:
    return str(x or "").split(" — ")[0].split(" | ")[0].strip()


def read_items(path: str) -> tuple[dict, dict]:
    J, M = defaultdict(list), defaultdict(list)
    for line in open(path, encoding="utf-8"):
        t, d = line.split("\t", 1)
        d = json.loads(d)
        if t == "JUDGE" and d["label"] not in J[d["ReportUID"]]:
            J[d["ReportUID"]].append(d["label"])
        elif t == "MAP" and all(x["name"] != d["name"] for x in M[d["ReportUID"]]):
            M[d["ReportUID"]].append(d)
    return J, M


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/ct_adjudication.json")
    ap.add_argument("--items", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--only-reports", default="")
    ap.add_argument("--rank", type=int)
    ap.add_argument("--world", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config)
    rank, world = rank_world(a.rank, a.world)
    log = lambda m: print(f"[rank {rank}] {m}", file=sys.stderr, flush=True)  # noqa: E731
    P = {k: read_prompt(v) for k, v in cfg["prompts"].items()}
    J, M = read_items(a.items)
    reps = sorted(set(J) | set(M))
    if a.only_reports:
        keep = set(Path(a.only_reports).read_text().split())
        reps = [u for u in reps if u in keep]
    df = pd.read_csv(a.input, low_memory=False).set_index("ReportUID")
    mine = [u for u in reps if u in df.index][rank::world]
    leaves = Protocols(cfg["protocols"], cfg["vocabulary"]).global_leaves
    names = {lf["name"] for lf in leaves}
    syn = {lf["name"]: (lf.get("syn") or "").strip() for lf in leaves}
    map_sys = P["map_system"] + "\n".join(f"- {lf['name']}" + (f" — {syn[lf['name']]}" if syn[lf['name']] else "")
                                          for lf in leaves)
    defn = lambda x: f"- {x}" + (f" — {syn[x]}" if syn.get(x) else "")  # noqa: E731
    log(f"{len(mine)} reports of {len(reps)} ({sum(len(J[u]) for u in mine)} judge, {sum(len(M[u]) for u in mine)} map)")
    inf = inferer_from_config(a.model, cfg, log)
    samp = inf.make_sampling(temperature=GREEDY_TEMPERATURE, top_p=GREEDY_TOP_P, max_tokens=cfg["max_tokens"])
    report = lambda u: (df.loc[u].findings, (df.loc[u].impression if isinstance(df.loc[u].impression, str) else "") or "(none)")  # noqa: E731
    msgs, meta = [], []
    for u in mine:
        f, imp = report(u)
        if J[u]:
            msgs.append([{"role": "system", "content": P["judge_system"]},
                         {"role": "user", "content": P["judge_user"].format(findings=f, impression=imp,
                                                                            labels="\n".join(defn(x) for x in J[u]))}])
            meta.append(("J", u))
        if M[u]:
            items = "\n".join(f"- {x['name']} | {x['polarity']} | \"{x['quote']}\"" for x in M[u])
            msgs.append([{"role": "system", "content": map_sys},
                         {"role": "user", "content": P["map_user"].format(findings=f, impression=imp, items=items)}])
            meta.append(("M", u))
    keep = [k for k, m in enumerate(msgs) if inf.prompt_fits(m, cfg["max_tokens"], safety_tokens=CONTEXT_LENGTH_SAFETY_TOKENS)[0]]
    msgs, meta = [msgs[k] for k in keep], [meta[k] for k in keep]
    out = open(a.output, "w", encoding="utf-8")
    t0 = time.time()
    outs = inf.chat(msgs, samp) if msgs else []
    cnt: dict[str, int] = defaultdict(int)
    pending = []
    for (t, u), raw in zip(meta, outs):
        p = parse(raw)
        if not isinstance(p, dict):
            cnt["unparsed"] += 1
            continue
        if t == "J":
            for v in p.get("verdicts") or []:
                if isinstance(v, dict) and clean(v.get("label")) in J[u] and v.get("verdict") in ("present", "possible", "absent"):
                    cnt["J_" + v["verdict"]] += 1
                    out.write("JUDGE\t" + json.dumps({"ReportUID": u, "label": clean(v["label"]), "verdict": v["verdict"]}) + "\n")
        else:
            want = {x["name"] for x in M[u]}
            for v in p.get("mapped") or []:
                if not isinstance(v, dict) or clean(v.get("name")) not in want:
                    continue
                lab, pol = clean(v.get("label") or "none"), v.get("polarity")
                lab = lab if lab in names else None
                if lab is None or pol not in ("asserted", "uncertain", "denied"):
                    cnt["M_none"] += 1
                    out.write("MAP\t" + json.dumps({"ReportUID": u, "name": clean(v["name"]), "label": None, "polarity": pol},
                                                   ensure_ascii=False) + "\n")
                else:
                    pending.append((u, clean(v["name"]), lab, pol))
    vm = []
    for u, _, lab, _ in pending:
        f, imp = report(u)
        vm.append([{"role": "system", "content": P["judge_system"]},
                   {"role": "user", "content": P["judge_user"].format(findings=f, impression=imp, labels=defn(lab))}])
    vouts = inf.chat(vm, samp) if vm else []
    for (u, nm, lab, pol), raw in zip(pending, vouts):
        p = parse(raw)
        vd = None
        for v in (p.get("verdicts") or []) if isinstance(p, dict) else []:
            if isinstance(v, dict) and clean(v.get("label")) == lab:
                vd = v.get("verdict")
        agree = vd in OK_FOR[pol]
        cnt["M_" + pol + ("_verified" if agree else "_rejected")] += 1
        out.write("MAP\t" + json.dumps({"ReportUID": u, "name": nm, "label": lab if agree else None, "polarity": pol,
                                        "check": vd}, ensure_ascii=False) + "\n")
    out.close()
    log(f"done {len(msgs)} prompts | {dict(cnt)} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
