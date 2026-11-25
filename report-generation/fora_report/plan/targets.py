"""(case, class) evidence-extraction targets of the final CT plan (paper: "Evidence tokens").

Applies the calibration tiers and the possible-tier cap of :class:`fora.reportgen.plan.diagnostic_plan.DiagnosticPlanBuilder`
(probable never capped; possible top-K by calibrated p or by fold change p / base rate) to every train and val study
and writes the list of planned findings for which evidence tokens and zoomed images must be extracted.

Outputs in ``--out-dir``:
  plan_targets.json     {"targets": {case: [cls, ...]}, "case2rep": {case: report_uid}} for every planned finding
  missing_targets.json  same format, restricted to findings not already in ``--existing-store`` (all of them when no
                        store is given); this is the ``--targets`` input of the evidence extractor
  targets_summary.json  counts (pairs / probable / possible / missing per split, measurable classes without a
                        lesion measurement when ``--measurable-classes`` and ``--measurements`` are given)
CPU only.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def capped_tiers(X, slope, intercept, raw_plan, raw_possible, prevalence, cap: int, rank: str):
    """Boolean (probable, possible) matrices for logits ``X`` [n, C]."""
    raw = 1 / (1 + np.exp(-X)); p = 1 / (1 + np.exp(-(X * slope + intercept)))
    conf = raw >= raw_plan; poss = (raw >= raw_possible) & ~conf
    if cap > 0:
        score = np.where(poss, (p / prevalence[None]) if rank == "fold" else p, -np.inf)
        order = np.argsort(-score, axis=1, kind="stable")[:, :cap]
        keep = np.zeros_like(poss); np.put_along_axis(keep, order, True, axis=1); poss = poss & keep
    return conf, poss


def load_store_keys(path: Path) -> dict[str, set[int]]:
    """report_uid -> classes present in a finding store npz (report_uid, row_start, cls)."""
    z = np.load(path, mmap_mode="r"); su = z["report_uid"].astype(str); rs = np.asarray(z["row_start"]); sc = np.asarray(z["cls"])
    return {u: set(sc[rs[i]:rs[i + 1]].tolist()) for i, u in enumerate(su.tolist())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--dense", type=Path, required=True, help="dense MIL logits npz (report_uid, logit, fora_id)")
    ap.add_argument("--case-map", type=Path, default=None,
                    help="npz (report_uid, fora_id) mapping reports to case ids; default: the fora_id of --dense")
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--plan-possible-cap", type=int, default=20)
    ap.add_argument("--plan-cap-rank", choices=("prob", "fold"), default="prob")
    ap.add_argument("--existing-store", type=Path, default=None, help="finding store npz whose (uid, cls) are skipped")
    ap.add_argument("--measurable-classes", type=Path, default=None, help="json {measurable: [...], vessel: [...]}")
    ap.add_argument("--measurements", type=Path, default=None, help="json {'case|cls': text} lesion measurements")
    ap.add_argument("--chunk", type=int, default=20000)
    a = ap.parse_args()

    cal = json.loads(a.calibration.read_text())
    slope = np.array(cal["class_slope"], np.float32); intercept = np.array(cal["class_intercept"], np.float32)
    rp = np.array(cal["class_raw_plan_threshold"], np.float32); rq = np.array(cal["class_raw_possible_threshold"], np.float32)
    prev = (np.array(cal["class_val_positive"], np.float64) + .5) / (cal["n_calibration_studies"] + 1.)
    d = np.load(a.dense); uid = d["report_uid"].astype(str); X = d["logit"]
    cm = np.load(a.case_map) if a.case_map else d
    case_of = dict(zip(cm["report_uid"].astype(str).tolist(), cm["fora_id"].astype(str).tolist()))
    split = json.loads(a.split.read_text())
    store = load_store_keys(a.existing_store) if a.existing_store else {}
    measurable: set[int] = set()
    meas: dict = {}
    if a.measurable_classes and a.measurements:
        mc = json.loads(a.measurable_classes.read_text())
        measurable = set(map(int, mc["measurable"])) | set(map(int, mc["vessel"]))
        meas = json.loads(a.measurements.read_text())

    targets, case2rep, missing = {}, {}, {}
    cnt: collections.Counter = collections.Counter()
    for s0 in range(0, len(uid), a.chunk):
        Xs = X[s0:s0 + a.chunk]
        conf, poss = capped_tiers(Xs, slope, intercept, rp, rq, prev, a.plan_possible_cap, a.plan_cap_rank)
        for j in range(Xs.shape[0]):
            u = uid[s0 + j]; c = case_of.get(u)
            if c is None or split.get(c) not in ("train", "val"):
                cnt["skipped_no_case_or_split"] += 1; continue
            cl = np.flatnonzero(conf[j] | poss[j]).tolist()
            if not cl:
                continue
            targets[c] = cl; case2rep[c] = u; have = store.get(u, set()); sp = split[c]
            m = [x for x in cl if x not in have]
            cnt[f"{sp}_pairs"] += len(cl); cnt[f"{sp}_probable"] += int(conf[j].sum()); cnt[f"{sp}_possible"] += int(poss[j].sum())
            cnt[f"{sp}_missing"] += len(m)
            if measurable:
                cnt[f"{sp}_measurable_pairs"] += sum(1 for x in cl if x in measurable)
                cnt[f"{sp}_measurable_without_measurement"] += sum(1 for x in cl if x in measurable and f"{c}|{x}" not in meas)
            if m:
                missing[c] = m
        print(s0, dict(cnt), flush=True)
    cnt["studies"] = len(targets); cnt["studies_with_missing"] = len(missing)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "plan_targets.json").write_text(json.dumps(dict(targets=targets, case2rep=case2rep)))
    (a.out_dir / "missing_targets.json").write_text(
        json.dumps(dict(targets=missing, case2rep={c: case2rep[c] for c in missing})))
    (a.out_dir / "targets_summary.json").write_text(json.dumps(
        dict(calibration=str(a.calibration), cap=a.plan_possible_cap, rank=a.plan_cap_rank, **cnt), indent=1))
    print(json.dumps(cnt), flush=True)


if __name__ == "__main__":
    main()
