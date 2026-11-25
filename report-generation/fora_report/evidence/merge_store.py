"""Merge per-node focus-token stores into the global localization store of the report writer
(paper section "Grounded report generation", data preparation).

Subcommands:
  parts  PARTS OUT_STORE OUT_DESC --ref-store REF   global store from the per-node parts (PARTS/node_*/{keys.npy, loc_part.npz, desc.npy})
  union  PARTS OUT_STORE OUT_DESC --old-store S --old-desc D
                                                  union of an existing global store and new per-node parts (existing rows win)
  tokens OLD NEW OUT                              union of two node-local focus-token stores (OLD wins on a duplicate key)

The global store (npz) has report_uid, row_start (rows grouped by report, sorted uid, classes ascending), cls, prob, reg3, frac3,
left, right, zfrac, view, feat (pooled 1048-d), region_names, labels; the description array (S600) is aligned with its rows.
The trainer stages it with ``fora_report.writer.dense_cache --kind loc`` and reads it through ``FindingLocStore``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import numpy as np

FIELDS = ('cls', 'prob', 'reg3', 'frac3', 'left', 'right', 'zfrac', 'view', 'feat')


def _save_store(path, uids, row_start, F, region_names, labels, uid_dtype):
    np.savez(path, report_uid=uids.astype(uid_dtype), row_start=row_start, cls=F['cls'].astype(np.int16), prob=F['prob'].astype(np.float32),
             reg3=F['reg3'].astype(np.int16), frac3=F['frac3'].astype(np.float16), left=F['left'].astype(np.float16), right=F['right'].astype(np.float16),
             zfrac=F['zfrac'].astype(np.float16), view=F['view'].astype(np.int16), region_names=region_names, labels=labels, feat=F['feat'].astype(np.float16))


def merge_parts(parts_dir, out_store, out_desc, ref_store):
    """Global store over all per-node parts; region_names / labels copied from ``ref_store``."""
    ref = np.load(ref_store, allow_pickle=True)
    K, D, F = [], [], {k: [] for k in ('uid',) + FIELDS}
    for nd in sorted(glob.glob(str(parts_dir) + '/node_*')):
        k = np.load(nd + '/keys.npy'); d = np.load(nd + '/desc.npy'); z = np.load(nd + '/loc_part.npz')
        assert len(k) == len(d) == len(z['cls']), nd
        K.append(k); D.append(d)
        for f in F: F[f].append(z[f])
    K = np.concatenate(K); D = np.concatenate(D); F = {f: np.concatenate(v) for f, v in F.items()}
    o = np.argsort(K, kind='stable'); K = K[o]; D = D[o]; F = {f: v[o] for f, v in F.items()}
    _, first = np.unique(K, return_index=True); K, D = K[first], D[first]; F = {f: v[first] for f, v in F.items()}
    uids, starts = np.unique(F['uid'], return_index=True)
    row_start = np.r_[starts, len(K)].astype(np.int64)
    _save_store(out_store, uids, row_start, F, ref['region_names'], ref['labels'], f'U{max(11, uids.dtype.itemsize // 4)}')
    np.save(out_desc, D.astype('S600'))
    return dict(reports=int(len(uids)), rows=int(len(K)), per_report=round(len(K) / max(len(uids), 1), 2))


def merge_union(parts_dir, out_store, out_desc, old_store, old_desc):
    """Union of an existing global store (+ descriptions) and new per-node parts; existing rows win on duplicates."""
    z = np.load(old_store)
    uid_o = z['report_uid']; rs = z['row_start']; n_o = int(rs[-1])
    F = {'uid': [np.repeat(uid_o.astype('U11'), np.diff(rs))]}
    for f in FIELDS: F[f] = [z[f]]
    D = [np.load(old_desc, mmap_mode='r')[:]]; src = [np.zeros(n_o, np.int8)]
    for nd in sorted(glob.glob(str(parts_dir) + '/node_*')):
        q = np.load(nd + '/loc_part.npz'); d = np.load(nd + '/desc.npy'); k = np.load(nd + '/keys.npy')
        assert len(k) == len(d) == len(q['cls']), nd
        F['uid'].append(q['uid'].astype('U11'))
        for f in FIELDS: F[f].append(q[f])
        D.append(d); src.append(np.ones(len(k), np.int8))
    F = {f: np.concatenate(v) for f, v in F.items()}; D = np.concatenate(D); src = np.concatenate(src)
    o = np.lexsort((src, F['cls'].astype(np.int64), F['uid']))
    F = {f: v[o] for f, v in F.items()}; D = D[o]; src = src[o]
    key_u, key_c = F['uid'], F['cls'].astype(np.int64)
    first = np.r_[True, (key_u[1:] != key_u[:-1]) | (key_c[1:] != key_c[:-1])]
    F = {f: v[first] for f, v in F.items()}; D = D[first]; src = src[first]
    uids, starts = np.unique(F['uid'], return_index=True)
    row_start = np.r_[starts, len(F['cls'])].astype(np.int64)
    _save_store(out_store, uids, row_start, F, z['region_names'], z['labels'], 'U11')
    np.save(out_desc, D.astype('S600'))
    return dict(reports=int(len(uids)), rows=int(len(F['cls'])), old_rows=int((src == 0).sum()), new_rows=int((src == 1).sum()))


def merge_tokens(old, new, out):
    """Union of two node-local focus-token stores (keys, x, aux, n_valid, fimg); OLD wins on a duplicate key."""
    tmp = out + '.part'; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    src = []
    for d in (old, new):
        if not os.path.exists(d + '/keys.npy'): continue
        src.append(dict(keys=np.load(d + '/keys.npy'), x=np.load(d + '/x.npy', mmap_mode='r'), aux=np.load(d + '/aux.npy', mmap_mode='r'), nv=np.load(d + '/n_valid.npy'),
                        off=np.load(d + '/fimg_off.npy'), blob=np.load(d + '/fimg_blob.npy', mmap_mode='r')))
    keys = np.concatenate([s['keys'].astype('U24') for s in src]); which = np.concatenate([np.full(len(s['keys']), i) for i, s in enumerate(src)])
    pos = np.concatenate([np.arange(len(s['keys'])) for s in src])
    o = np.lexsort((which, keys)); keys, which, pos = keys[o], which[o], pos[o]
    first = np.r_[True, keys[1:] != keys[:-1]]; keys, which, pos = keys[first], which[first], pos[first]
    n = len(keys)
    np.save(tmp + '/keys.npy', keys); np.save(tmp + '/row_ids.npy', np.arange(n))
    X = np.lib.format.open_memmap(tmp + '/x.npy', 'w+', np.float16, (n, 16, 1048)); A = np.zeros((n, 16, 16), np.float16); NV = np.zeros(n, np.int16)
    for i, s in enumerate(src):
        sel = np.flatnonzero(which == i); p = pos[sel]
        for a in range(0, len(sel), 8192): X[sel[a:a + 8192]] = s['x'][p[a:a + 8192]]
        A[sel] = s['aux'][p]; NV[sel] = s['nv'][p]
    X.flush(); del X
    np.save(tmp + '/aux.npy', A); np.save(tmp + '/n_valid.npy', NV)
    nview = src[0]['off'].shape[1] - 1
    lens = np.zeros((n, nview), np.int64)
    for i, s in enumerate(src):
        sel = np.flatnonzero(which == i); lens[sel] = np.diff(s['off'][pos[sel]], axis=1)
    FO = np.zeros((n, nview + 1), np.int64); FO[:, 1:] = np.cumsum(lens, axis=1); start = np.r_[0, np.cumsum(FO[:, -1])[:-1]]; FO += start[:, None]
    B = np.lib.format.open_memmap(tmp + '/fimg_blob.npy', 'w+', np.uint8, (int(FO[-1, -1]) if n else 0,))
    for j in range(n):
        s = src[which[j]]; a0, a1 = s['off'][pos[j], 0], s['off'][pos[j], -1]
        B[FO[j, 0]:FO[j, -1]] = s['blob'][a0:a1]
    B.flush(); del B
    np.save(tmp + '/fimg_off.npy', FO)
    summ = dict(rows=int(n), from_old=int((which == 0).sum()), from_new=int((which == 1).sum()) if len(src) > 1 else 0)
    with open(tmp + '/summary.json', 'w') as f: json.dump(summ, f)
    shutil.rmtree(out, ignore_errors=True); os.rename(tmp, out)
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0]); sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('parts'); p.add_argument('parts'); p.add_argument('out_store'); p.add_argument('out_desc'); p.add_argument('--ref-store', required=True)
    u = sp.add_parser('union'); u.add_argument('parts'); u.add_argument('out_store'); u.add_argument('out_desc'); u.add_argument('--old-store', required=True); u.add_argument('--old-desc', required=True)
    t = sp.add_parser('tokens'); t.add_argument('old'); t.add_argument('new'); t.add_argument('out')
    a = ap.parse_args(argv)
    if a.cmd == 'parts': r = merge_parts(a.parts, a.out_store, a.out_desc, a.ref_store)
    elif a.cmd == 'union': r = merge_union(a.parts, a.out_store, a.out_desc, a.old_store, a.old_desc)
    else: r = merge_tokens(a.old, a.new, a.out)
    print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
