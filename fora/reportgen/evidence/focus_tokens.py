"""Focus-token store of the report writer (paper section "Grounded report generation", evidence tokens).

Reads the extractor chunks (``extent_r*_p*_c*.npz`` from ``fora.reportgen.evidence.extract``) and writes one row per
(report, class), keyed ``"<report_uid>|<cls:04d>"`` and sorted by key:

    keys.npy       [n] unicode keys
    row_ids.npy    [n] 0..n-1
    x.npy          [n, 16, 1048] float16  focus-token features (MIL head input of the 16 evidence tokens)
    aux.npy        [n, 16, 16]   float16  per-token side features (layout below)
    n_valid.npy    [n] int16              valid focus tokens per row
    fimg_blob.npy  uint8                   concatenated JPEG evidence images
    fimg_off.npy   [n, 7] int64           byte offsets of the 6 image views per row
    desc.npy       [n] S600                finding description (location, pattern, measurement; fora.reportgen.evidence.describe)
    loc_part.npz   per-row localization fields (uid, cls, prob, reg3, frac3, left, right, zfrac, view, feat = pooled 1048-d feature)

aux per token: [0] 1 (focus token), [1:4] organ-relative position S/R/A (0 if unknown), [4] position valid, [5:8] offset from its
focus centroid / 50 mm, [8:11] offset from the global evidence peak / 100 mm, [11] evidence / peak, [12:16] focus one-hot (foci >= 3
share the last slot).

CLI: ``python -m fora.reportgen.evidence.focus_tokens EXTENT_DIR OUT_DIR --labels labels.json --region-names regions.json
[--measurements measurement_text.json]`` where measurement_text maps ``"<case>|<cls>"`` to a measurement phrase.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from fora.reportgen.evidence.describe import describe_finding

N_TOK, FEAT_DIM, AUX_DIM, N_VIEWS, DESC_BYTES = 16, 1048, 16, 6, 600
FOCI_KEYS = ('tally', 'cov', 'gshare', 'gshare80', 'left80', 'ncell', 'restricted', 'left_all', 'n5', 'share', 'ctr', 'regall', 'left', 'rel', 'ml',
             'bdist', 'lbdist', 'inv10', 'neff')
LOC_FIELDS = ('uid', 'cls', 'prob', 'reg3', 'frac3', 'left', 'right', 'zfrac', 'view', 'feat')


def token_aux(q, i):
    """aux [k,16] and the valid-token mask of pair i of an extractor chunk q."""
    va = q['fo_tva'][i].astype(bool); k = int(va.sum()); a = np.zeros((k, AUX_DIM), np.float32)
    if k:
        a[:, 0] = 1
        rel = q['fo_trel'][i][va].astype(np.float32); ok = np.isfinite(rel).all(1); a[ok, 1:4] = rel[ok]; a[:, 4] = ok
        a[:, 5:8] = q['fo_toff'][i][va] / 50.; a[:, 8:11] = q['fo_tgo'][i][va] / 100.; a[:, 11] = q['fo_tev'][i][va]
        fid = q['fo_tfo'][i][va].astype(int); a[np.arange(k), 12 + np.clip(fid, 0, 3)] = 1
    return a, va


def build_store(ext_dir, out_dir, labels, region_names, measurements=None, log=print):
    """Build the focus-token store of all extractor chunks in ``ext_dir`` into ``out_dir`` (see module docstring)."""
    ext_dir, out_dir = str(ext_dir), str(out_dir); os.makedirs(out_dir, exist_ok=True); meas = measurements or {}
    files = sorted(glob.glob(ext_dir + '/extent_r*_p*_c*.npz')); n = 0
    for f in files:
        with np.load(f, allow_pickle=True) as q: n += len(q['cls'])
    X = np.lib.format.open_memmap(out_dir + '/x_unsorted.npy', 'w+', np.float16, (n, N_TOK, FEAT_DIM)); AUX = np.zeros((n, N_TOK, AUX_DIM), np.float16)
    IMG = [None] * n; KEY = np.empty(n, object); LP = {k: [] for k in LOC_FIELDS}; NV = np.zeros(n, np.int16); TXT = np.empty(n, object); o = 0
    for f in files:
        with np.load(f, allow_pickle=True) as qz: q = {k: qz[k] for k in qz.files}   # load each array once (npz access decompresses the whole array)
        for i in range(len(q['cls'])):
            u = str(q['pair_rep'][i]); c = int(q['cls'][i]); case = str(q['pair_case'][i])
            KEY[o] = f'{u}|{c:04d}'; fo = {k: q['fo_' + k][i] for k in FOCI_KEYS}
            for k_, v_ in (('uid', u), ('cls', c), ('prob', float(q['prob'][i])), ('reg3', q['reg3'][i]), ('frac3', q['frac3'][i]), ('left', float(q['left'][i])),
                           ('right', float(q['right'][i])), ('zfrac', float(q['zfrac'][i])), ('view', int(q['view'][i]))): LP[k_].append(v_)
            LP['feat'].append(q['feat'][i])
            if 'img_off' in q:
                of_ = q['img_off'][i]; bl_ = q['img_blob']; IMG[o] = tuple(bl_[of_[k]:of_[k + 1]].tobytes() for k in range(len(of_) - 1))
            TXT[o] = describe_finding(labels[c], fo, region_names, measurement=meas.get(f'{case}|{c}', ''), fallback='')
            a, va = token_aux(q, i); k = len(a); NV[o] = k
            if k: X[o, :k] = q['fo_tx'][i][va]; AUX[o, :k] = a
            o += 1
        log(f'{f} {o}')
    KS = np.array([KEY[i_] for i_ in range(n)]); oo = np.argsort(KS, kind='stable')
    ks = KS[oo]; _, first = np.unique(ks, return_index=True); order = oo[first]; ks = ks[first]   # first occurrence of a duplicate key wins
    np.save(out_dir + '/keys.npy', ks.astype('U80'))
    np.savez(out_dir + '/loc_part.npz', **{k_: (np.array([LP[k_][j_] for j_ in order]) if k_ != 'feat' else np.stack([LP['feat'][j_] for j_ in order]).astype(np.float16))
                                          for k_ in LP})
    np.save(out_dir + '/row_ids.npy', np.arange(len(order))); np.save(out_dir + '/n_valid.npy', NV[order]); np.save(out_dir + '/aux.npy', AUX[order])
    Xs = np.lib.format.open_memmap(out_dir + '/x.npy', 'w+', np.float16, (len(order), N_TOK, FEAT_DIM))
    for s0 in range(0, len(order), 4096): Xs[s0:s0 + 4096] = X[order[s0:s0 + 4096]]
    Xs.flush(); del X, Xs; os.remove(out_dir + '/x_unsorted.npy')
    parts = [IMG[o_] or (b'',) * N_VIEWS for o_ in order]; FO = np.zeros((len(order), N_VIEWS + 1), np.int64); pos = 0
    for i_, t in enumerate(parts):
        FO[i_, 0] = pos
        for k in range(N_VIEWS): pos += len(t[k]); FO[i_, k + 1] = pos
    np.save(out_dir + '/fimg_blob.npy', np.frombuffer(b''.join(b for t in parts for b in t), np.uint8)); np.save(out_dir + '/fimg_off.npy', FO)
    np.save(out_dir + '/desc.npy', np.array([t.encode('utf-8')[:DESC_BYTES] for t in TXT[order]], dtype=f'S{DESC_BYTES}'))
    summary = dict(pairs=int(n), rows=int(len(order)), mean_tokens=float(NV[order].mean()) if len(order) else 0.)
    Path(out_dir, 'summary.json').write_text(json.dumps(summary)); return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description='Build the focus-token store from evidence extractor chunks.')
    ap.add_argument('extent_dir'); ap.add_argument('out_dir')
    ap.add_argument('--labels', type=Path, required=True, help='JSON list of class names in MIL vocabulary order')
    ap.add_argument('--region-names', type=Path, required=True, help='JSON list of CADS region names (bit order)')
    ap.add_argument('--measurements', type=Path, default=None, help='JSON {"<case>|<cls>": measurement phrase}')
    a = ap.parse_args(argv)
    meas = json.loads(a.measurements.read_text()) if a.measurements else {}
    print(json.dumps(build_store(a.extent_dir, a.out_dir, json.loads(a.labels.read_text()), json.loads(a.region_names.read_text()), meas)))


if __name__ == '__main__':
    main()
