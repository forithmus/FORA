"""Per-class scale calibration of a lesion caliper on the training split (paper: Methods, "Measurements").

The caliper is run on every candidate (MIL local maxima, rank < k) of every training / validation finding; the measurement of the
top MIL peak (rank 0) is compared with the report size: offset = median log(caliper / report) per (class, measurement type) with
>= 30 training findings, else per measurement type (`t<mt>`), using only pairs with |log ratio| < 1.  At inference the measurement is
divided by exp(offset) (lesion_measure.py, merge.py).  Midline-shift classes (336, 663) are left out.
Primary value per target: nodules (ycmp 3) mean of long + short axis, ycmp 1 long, ycmp 2 short, ycmp 4 max(long, 3-D extent),
otherwise long axis (lesion type) or inscribed diameter (thickness / vessel / lumen).
usage: python -m fora.reportgen.measure.scale_calibration CAL_DIR --cand cand.npz --targets targets.npz --out caliper_scale_<group>.json [--k 16]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

CALF = ['long', 'short', 'ins', 'ext3d', 'vol', 'empty', 'zoom', 'border', 'route', 'side', 'dcen', 'hin', 'hsh', 'msum', 'tok', 'touch0', 'touch1', 'expand']


def load_cal(d, nrows, C):
    """caliper rows keyed by the stable (target, rank) pair."""
    out = {k: np.full(nrows, np.nan, np.float32) for k in CALF}; have = np.zeros(nrows, bool)
    key = C['t'].astype(np.int64) * 64 + C['rank']; ordr = np.argsort(key); sk = key[ordr]
    for f in sorted(glob.glob(str(Path(d) / 'cal_r*_c*.npz'))):
        z = np.load(f)
        q = z['t'].astype(np.int64) * 64 + z['prank']; pos = np.clip(np.searchsorted(sk, q), 0, len(sk) - 1); okm = sk[pos] == q
        r = ordr[pos[okm]]; z = {k: z[k][okm] for k in z.files}; have[r] = True
        for k in CALF:
            if k in z: out[k][r] = z[k]
    return out, have


def fit_cal(pred, yv, mask_, cls, mt):
    lr_ = np.log(pred / yv); ok = mask_ & np.isfinite(lr_) & (np.abs(lr_) < 1.0); calib = {}
    for t_ in range(4):
        kt = ok & (mt == t_); calib[f't{t_}'] = float(np.median(lr_[kt])) if kt.sum() >= 30 else 0.
    for c in np.unique(cls):
        for t_ in range(4):
            kc = ok & (cls == c) & (mt == t_)
            if kc.sum() >= 30: calib[f'{c}_{t_}'] = float(np.median(lr_[kc]))
    return calib


def top_peak_calibration(cal_dir, C, T, K=16):
    T = dict(T)
    if 'keep' in T: T['split'] = np.where(T['keep'], T['split'], 'drop')
    nT = len(T['key']); nrows = len(C['row'])
    cal, have = load_cal(cal_dir, nrows, C)
    mt_row = T['mt'][C['t']]; P_row = np.where(mt_row == 0, cal['long'], cal['ins'])
    if 'ycmp' in T:
        ycr = T['ycmp'][C['t']]; Lr = cal['long'].astype(np.float64); Sr = cal['short'].astype(np.float64)
        Er = cal['ext3d'].astype(np.float64); P_row = np.where(ycr == 3, (Lr + Sr) / 2, np.where(ycr == 1, Lr, np.where(ycr == 2, Sr, np.where(ycr == 4, np.fmax(Lr, Er), P_row))))
    keep = (C['rank'] < K) & have
    R = np.full((nT, K), -1, np.int64)
    ki = np.flatnonzero(keep); R[C['t'][ki], C['rank'][ki]] = ki
    M = R >= 0; Rc = np.clip(R, 0, None)
    P = np.where(M, P_row[Rc], np.nan); hasc = M.any(1)
    top = np.where(hasc, P[:, 0], np.nan)
    ex = np.isin(T['cls'], [336, 663]); tr = (T['split'] == 'train') & ~ex
    return fit_cal(top, T['y_long'], tr, T['cls'], T['mt'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cal'); ap.add_argument('--cand', required=True); ap.add_argument('--targets', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--k', type=int, default=16)
    a = ap.parse_args()
    C = dict(np.load(a.cand, allow_pickle=True)); T = dict(np.load(a.targets, allow_pickle=True))
    calib = top_peak_calibration(a.cal, C, T, a.k)
    Path(a.out).write_text(json.dumps(calib, indent=2) + '\n'); print(json.dumps(calib))


if __name__ == '__main__':
    main()
