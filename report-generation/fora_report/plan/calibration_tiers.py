"""Calibration and plan tiers for the report writers (paper: "Diagnostic plan", Methods "Calibration").

Shared by the CT and chest X-ray writers:
  * per-class Platt scaling p = sigmoid(a * logit + b) fitted on TRAIN study-bag logits, ridge prior toward
    (a, b) = (1, 0) with weight 1.0, Newton steps with backtracking, slope floor 0.05;
  * probable cutoff = calibrated p >= 0.5, never below the train Youden cutoff;
  * possible cutoff = per-class Youden cutoff on the TRAIN raw sigmoid, never above the probable cutoff
    (0.5 for classes without train positives);
  * base rates = train positives / train studies.
Cutoffs are chosen in raw-probability space and mapped through the class Platt curve (monotone, so the plan selects
exactly the studies whose raw probability passes the raw cutoff).

The CLI builds the CT calibration (``calibration_train_platt.json``) from dense study-bag
logits, the label state matrix and the patient-level split. CPU only.

Calibration JSON schema (read by :class:`fora.reportgen.plan.diagnostic_plan.DiagnosticPlanBuilder`):
  labels, canonical_plan_labels, label_order_sha256      class order (sha256 of the newline-joined labels)
  class_slope, class_intercept                           Platt curve per class
  class_plan_threshold, class_possible_threshold         calibrated-probability cutoffs (probable / possible tier)
  class_raw_plan_threshold, class_raw_possible_threshold raw-sigmoid cutoffs (same tiers)
  class_raw_youden_threshold                             train Youden cutoff (CT file only)
  class_val_positive, n_calibration_studies              TRAIN positives per class and TRAIN study count (key names
                                                         read by the plan builder and the released assets)
  source_predictions                                     dense logits file (the trainer checks it)
  calibration_report_uids [] / exclude_from_report_training false, format, platt, threshold_rule,
  threshold_policy, target_policy, mil_source, mil_checkpoint, summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CT_TARGET_POLICY = "label state asserted(1) or uncertain(2) = 1; denied(0) or unmentioned(3) = 0; -1 masked"


def label_order_sha256(labels) -> str:
    return hashlib.sha256("\n".join(map(str, labels)).encode("utf-8")).hexdigest()


def fit_platt(logits, targets, mask, l2: float = 1.0, iterations: int = 50):
    """Per-class Platt fit, vectorised Newton with backtracking, ridge toward (1, 0); returns (slope, intercept)."""
    x = logits.astype(np.float64); y = targets.astype(np.float64); w = mask.astype(np.float64)
    C = x.shape[1]; a = np.ones(C); b = np.zeros(C)

    def objective(a_, b_):
        z = x * a_ + b_
        loss = (np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))) * w
        return loss.sum(0) + 0.5 * l2 * ((a_ - 1.0) ** 2 + b_ ** 2)

    current = objective(a, b)
    for _ in range(iterations):
        z = x * a + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        r = (p - y) * w; s = p * (1 - p) * w
        ga = (r * x).sum(0) + l2 * (a - 1.0); gb = r.sum(0) + l2 * b
        haa = (s * x * x).sum(0) + l2; hab = (s * x).sum(0); hbb = s.sum(0) + l2
        det = haa * hbb - hab * hab; det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        da = (hbb * ga - hab * gb) / det; db = (haa * gb - hab * ga) / det
        step = np.ones(C); accepted = np.zeros(C, bool)
        for _h in range(30):
            trial = objective(a - step * da, b - step * db)
            better = (trial <= current) & ~accepted
            a = np.where(better, a - step * da, a); b = np.where(better, b - step * db, b)
            current = np.where(better, trial, current); accepted |= better
            if accepted.all():
                break
            step = np.where(accepted, step, step * 0.5)
        if max(np.abs(step * da).max(), np.abs(step * db).max()) < 1e-7:
            break
    return np.maximum(a, 0.05), b


def youden(s, y) -> float:
    """Youden cutoff (rule A: predict positive if score >= cut), evaluated at distinct-score boundaries."""
    o = np.argsort(-s, kind="mergesort"); s = s[o]; y = y[o]; tp = np.cumsum(y); fp = np.cumsum(1 - y)
    last = np.r_[s[1:] != s[:-1], True]
    j = tp / y.sum() - fp / (len(y) - y.sum()); j[~last] = -np.inf; k = int(np.argmax(j)); return float(s[k])


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def class_youden(raw_prob, target, mask=None, require_negative: bool = True):
    """Per-class Youden cutoff on raw probabilities (masked rows excluded) and the positive count per class.

    Classes without positives get 0.5. With ``require_negative`` (CT builder) a class whose unmasked rows are all
    positive also gets 0.5; the X-ray builder used ``require_negative=False`` (``Y.any()``).
    """
    C = raw_prob.shape[1]; cut = np.empty(C); npos = np.zeros(C, int)
    for j in range(C):
        m = mask[:, j] if mask is not None else slice(None)
        yy = target[m, j].astype(np.float64); npos[j] = int(yy.sum())
        ok = 0 < yy.sum() < len(yy) if require_negative else yy.sum() > 0
        cut[j] = youden(raw_prob[m, j], yy) if ok else 0.5
    return cut, npos


def tier_thresholds(slope, intercept, youden_raw):
    """Probable = calibrated p >= 0.5 never below Youden; possible = Youden never above probable.

    Returns raw cutoffs (raw_plan, raw_possible) and calibrated cutoffs (plan, possible)."""
    raw_p50 = 1 / (1 + np.exp(intercept / np.maximum(slope, 1e-6)))
    raw_plan = np.maximum(raw_p50, youden_raw)
    raw_possible = np.minimum(raw_plan, youden_raw)
    plan = 1 / (1 + np.exp(-(slope * logit(raw_plan) + intercept)))
    possible = np.minimum(1 / (1 + np.exp(-(slope * logit(raw_possible) + intercept))), plan)
    return dict(raw_plan=raw_plan, raw_possible=raw_possible, plan=plan, possible=possible, raw_p50=raw_p50)


def tier_stats(raw, raw_plan, raw_poss, target, mask, name):
    """Per-study tier sizes and micro precision/recall against report labels (masked entries excluded)."""
    conf = raw >= raw_plan; poss = (raw >= raw_poss) & ~conf; union = conf | poss
    y = target & mask
    out = dict(rule=name, n_studies=int(raw.shape[0]), label_positives_per_study=float(y.sum(1).mean()))
    for k, sel in (("probable", conf), ("possible", poss), ("union", union)):
        n = sel.sum(1); tp = int((sel & y).sum()); np_ = int((sel & mask).sum())
        out[k] = dict(mean_per_study=float(n.mean()), median=float(np.median(n)), p95=float(np.percentile(n, 95)),
                      zero_frac=float((n == 0).mean()),
                      micro_precision=tp / max(np_, 1), micro_recall=tp / max(int(y.sum()), 1), true_positives=tp)
    return out


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp"); tmp.write_text(json.dumps(payload) + "\n"); tmp.replace(path)


def build_ct_calibration(dense: Path, labels_npz: Path, split: Path, *, platt_l2: float = 1.0, chunk: int = 128,
                         mil_source: str = "precomputed", mil_checkpoint: str = "",
                         target_policy: str = CT_TARGET_POLICY) -> tuple[dict, dict]:
    """CT tier-2 calibration from dense study-bag logits (npz: report_uid, fora_id, logit [n, C] f32, label [C]),
    the label state matrix (npz: nodes [K], cases [M], state [M, K] int; 1/2 -> positive, 0/3 -> negative, -1
    masked) and the split json (case id -> "train" | "val" | ...). Returns (calibration, summary)."""
    d = np.load(dense, allow_pickle=False)
    classes = list(map(str, d["label"].tolist())); C = len(classes)
    fid = d["fora_id"].astype(str); X = np.asarray(d["logit"], np.float32)
    split_of = json.loads(Path(split).read_text())
    Z = np.load(labels_npz, allow_pickle=False)
    node_col = {n: j for j, n in enumerate(Z["nodes"].tolist())}
    cols = np.array([node_col[c] for c in classes])
    case_row = {c: j for j, c in enumerate(Z["cases"].tolist())}
    rows = {s: np.array([i for i, f in enumerate(fid.tolist()) if split_of.get(f) == s and f in case_row])
            for s in ("train", "val")}
    print({s: len(v) for s, v in rows.items()}, flush=True)
    state = {s: Z["state"][[case_row[f] for f in fid[k]]][:, cols] for s, k in rows.items()}
    Xs = {s: X[k] for s, k in rows.items()}
    tgt = {s: np.isin(state[s], (1, 2)) for s in rows}; msk = {s: state[s] >= 0 for s in rows}

    slope = np.ones(C); intercept = np.zeros(C)
    for c0 in range(0, C, chunk):   # class chunks bound memory
        sl = slice(c0, min(C, c0 + chunk))
        slope[sl], intercept[sl] = fit_platt(Xs["train"][:, sl], tgt["train"][:, sl], msk["train"][:, sl], l2=platt_l2)
        print(f"platt {sl.stop}/{C}", flush=True)
    ptr = 1 / (1 + np.exp(-Xs["train"].astype(np.float64)))
    yd, ntr_pos = class_youden(ptr, tgt["train"], msk["train"], require_negative=True)
    del ptr
    t = tier_thresholds(slope, intercept, yd)
    n_train = int(len(rows["train"]))

    pva = 1 / (1 + np.exp(-Xs["val"].astype(np.float64)))
    val_stats = tier_stats(pva, t["raw_plan"], t["raw_possible"], tgt["val"], msk["val"], "platt50_train + youden_train")
    ptr32 = 1 / (1 + np.exp(-Xs["train"]))
    trc = ptr32 >= t["raw_plan"].astype(np.float32); trp = (ptr32 >= t["raw_possible"].astype(np.float32)) & ~trc
    train_sizes = dict(probable=float(trc.sum(1).mean()), possible=float(trp.sum(1).mean()),
                       union_p95=float(np.percentile((trc | trp).sum(1), 95)))
    summary = dict(
        val_plan_sizes=dict(confident_mean=val_stats["probable"]["mean_per_study"],
                            possible_mean=val_stats["possible"]["mean_per_study"],
                            confident_p95=val_stats["probable"]["p95"], possible_p95=val_stats["possible"]["p95"]),
        val_tiers=val_stats, train_plan_sizes=train_sizes,
        n_classes_youden_equals_plan=int((t["raw_possible"] >= t["raw_plan"]).sum()),
        n_classes_plan_raised_to_youden=int((yd > t["raw_p50"]).sum()),
        n_classes_no_train_positive=int((ntr_pos == 0).sum()))
    calibration = {
        "format": "ct_frozen_mil_train_platt", "format_version": 1,
        "mil_source": mil_source, "mil_checkpoint": mil_checkpoint,
        "source_predictions": str(dense), "calibration_logits_source": str(dense),
        "threshold_train_logits_source": str(dense),
        "source_split": "train", "n_calibration_studies": n_train, "calibration_report_uids": [],
        "exclude_from_report_training": False,
        "target_policy": target_policy,
        "platt": {"l2_toward_identity": platt_l2, "min_slope": 0.05},
        "threshold_rule": {"plan": "platt50_train", "possible": "youden_train"},
        "threshold_policy": {
            "plan": "calibrated probability (Platt fitted on TRAIN study-bag logits) >= 0.5; never below the train Youden cutoff",
            "possible": "per-class Youden cutoff on the TRAIN raw sigmoid (masked rows excluded; 0.5 if no train positive), never above the probable cutoff; mapped through Platt",
            "base_rate": "class_val_positive / n_calibration_studies = TRAIN positives / TRAIN studies"},
        "labels": classes, "label_order_sha256": label_order_sha256(classes), "canonical_plan_labels": classes,
        "class_val_positive": ntr_pos.astype(int).tolist(),
        "class_slope": slope.tolist(), "class_intercept": intercept.tolist(),
        "class_raw_youden_threshold": yd.tolist(),
        "class_raw_plan_threshold": t["raw_plan"].tolist(), "class_raw_possible_threshold": t["raw_possible"].tolist(),
        "class_plan_threshold": t["plan"].tolist(), "class_possible_threshold": t["possible"].tolist(),
        "summary": summary,
    }
    return calibration, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the CT report-generation calibration (train Platt + train Youden tiers).")
    ap.add_argument("--dense", type=Path, required=True, help="dense study-bag MIL logits (npz)")
    ap.add_argument("--labels-npz", type=Path, required=True, help="label state matrix (npz: nodes, cases, state)")
    ap.add_argument("--split", type=Path, required=True, help="patient-level split json (case id -> split)")
    ap.add_argument("--mil-checkpoint", default="", help="recorded in the output for provenance")
    ap.add_argument("--platt-l2", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True)
    a = ap.parse_args()
    calibration, summary = build_ct_calibration(a.dense, a.labels_npz, a.split, platt_l2=a.platt_l2, chunk=a.chunk,
                                                mil_checkpoint=a.mil_checkpoint)
    print(json.dumps(summary, indent=1), flush=True)
    write_json_atomic(a.out, calibration)
    a.stats.parent.mkdir(parents=True, exist_ok=True)
    a.stats.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {a.out} {a.stats}", flush=True)


if __name__ == "__main__":
    main()
