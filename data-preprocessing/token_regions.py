"""Per-token region bits for every cached series (Methods, "Organ routing").

compute  For the series of the selected ranks (assign.json) compute fora.regions.token_region_bits and write
         <local>/regions/<relative_file>.npz with `bits` (uint64 per body token, order of flat_indices). Series are
         read from <local>/<relative_file> when staged, otherwise directly from their pack.
pack     Concatenate the per-series bits of the selected ranks into one durable H5 (`bits`, `offsets`,
         `relative_files`, `cads_source`, `info_json`, attr `region_names`).
unpack   Write the per-series .npz files of a pack file back to <local>/regions/ (what the trainers read).

Usage
  python token_regions.py compute --cases cases.jsonl --assign assign.json --ranks 0-3 --local LOCAL \
         --h5-dir H5 --cads-dir CADS [--procs 72]
  python token_regions.py pack    --cases cases.jsonl --assign assign.json --ranks 0-3 --local LOCAL --out regions_00.h5
  python token_regions.py unpack  --pack regions_00.h5 --local LOCAL
LOCAL is the node-local cache root of one fingerprint (the directory that holds <split>/<report id>/series_NNN.h5).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback

import h5py
import numpy as np

from fora.regions import REGION_NAMES, token_region_bits


def parse_ranks(s: str) -> list[int]:
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return out


def series_jobs(cases_path: str, assign_path: str, ranks) -> list[tuple]:
    """(pack file, offset, file_size, relative_file) of every series of the reports of the given ranks, pack order."""
    A = json.load(open(assign_path))
    mine = set()
    for split in ('train', 'val'):
        for rk in ranks:
            mine.update(A['assignments'][split][rk])
    jobs = []
    with open(cases_path) as f:
        for line in f:
            c = json.loads(line)
            if c['id'] in mine:
                jobs += [(s['file'], s['offset'], s['file_size'], s['relative_file']) for s in c['series']]
    jobs.sort()
    return jobs


def _work(args):
    src, dst, h5_dir, cads_dir = args
    try:
        bits, _, info = token_region_bits(src, h5_dir, cads_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + '.tmp.npz'
        np.savez(tmp, bits=bits, info=np.array(json.dumps(info)))
        os.replace(tmp, dst)
        return dst, dict(ok=True, seconds=info['seconds'], src=info['cads_source'])
    except Exception as e:
        return dst, dict(ok=False, err=f'{type(e).__name__}: {e}', tb=traceback.format_exc()[-1500:])


def compute(a):
    jobs = series_jobs(a.cases, a.assign, parse_ranks(a.ranks))
    todo = []
    for pack, off, size, rel in jobs:
        dst = f'{a.local}/regions/{rel}.npz'
        if os.path.exists(dst):
            continue
        local = f'{a.local}/{rel}'
        src = local if os.path.exists(local) and os.path.getsize(local) == size else (pack, off, size)
        todo.append((src, dst, a.h5_dir, a.cads_dir))
    print(json.dumps(dict(event='start', files=len(jobs), pending=len(todo), procs=a.procs)), flush=True)
    t0 = time.time()
    done = failed = 0
    with mp.get_context('fork').Pool(a.procs, maxtasksperchild=50) as pool:
        for dst, r in pool.imap_unordered(_work, todo, chunksize=4):
            if r['ok']:
                done += 1
            else:
                failed += 1
                print(json.dumps(dict(event='error', file=dst, err=r['err'], tb=r['tb'])), flush=True)
    print(json.dumps(dict(event='computed', done=done, failed=failed, minutes=round((time.time() - t0) / 60, 1))), flush=True)


def pack(a):
    jobs = series_jobs(a.cases, a.assign, parse_ranks(a.ranks))
    rels, offs, srcs, infos = [], [0], [], []
    tmp = a.out + '.tmp'
    with h5py.File(tmp, 'w') as h:
        d = h.create_dataset('bits', shape=(0,), maxshape=(None,), dtype=np.uint64, chunks=(1 << 20,), shuffle=True,
                             compression='gzip', compression_opts=1)
        buf = []
        nbuf = 0

        def flush():
            nonlocal buf, nbuf
            if not buf:
                return
            x = np.concatenate(buf)
            n0 = d.shape[0]
            d.resize((n0 + len(x),))
            d[n0:] = x
            buf = []
            nbuf = 0

        for _, _, _, rel in jobs:
            fn = f'{a.local}/regions/{rel}.npz'
            if not os.path.exists(fn):
                continue
            z = np.load(fn)
            b = z['bits']
            info = json.loads(str(z['info']))
            buf.append(b)
            nbuf += len(b)
            rels.append(rel)
            offs.append(offs[-1] + len(b))
            srcs.append(info['cads_source'])
            infos.append({k: info.get(k) for k in ('cads_source', 'mapping', 'cads_sid', 'geom_ok', 'n_tokens', 'borrow_cover')})
            if nbuf >= (64 << 20):
                flush()
        flush()
        h.create_dataset('offsets', data=np.asarray(offs, np.int64))
        h.create_dataset('relative_files', data=np.asarray(rels, dtype=h5py.string_dtype()))
        h.create_dataset('cads_source', data=np.asarray(srcs, dtype=h5py.string_dtype()))
        h.create_dataset('info_json', data=np.asarray([json.dumps(x) for x in infos], dtype=h5py.string_dtype()))
        h.attrs['region_names'] = json.dumps(REGION_NAMES)
        h.attrs['fingerprint'] = json.load(open(a.assign))['fingerprint']
        h.attrs['bit_order'] = 'bit i of bits[t] = region_names[i]; token order = series flat_indices'
    os.replace(tmp, a.out)
    print(json.dumps(dict(files=len(rels), tokens=int(offs[-1]), missing=len(jobs) - len(rels),
                          no_cads=srcs.count('none'), borrowed=srcs.count('borrowed'))), flush=True)


def unpack(a):
    t0 = time.time()
    n = 0
    with h5py.File(a.pack, 'r') as f:
        assert json.loads(f.attrs['region_names']) == REGION_NAMES, 'region order mismatch'
        rel = [x.decode() if isinstance(x, bytes) else x for x in f['relative_files'][:]]
        off = f['offsets'][:]
        B = f['bits']
        CH = 200
        for s in range(0, len(rel), CH):
            e = min(len(rel), s + CH)
            block = B[off[s]:off[e]]
            for k in range(s, e):
                d = f'{a.local}/regions/{rel[k]}.npz'
                os.makedirs(os.path.dirname(d), exist_ok=True)
                np.savez(d, bits=block[off[k] - off[s]:off[k + 1] - off[s]].astype(np.uint64))
                n += 1
    print(json.dumps(dict(files=n, seconds=round(time.time() - t0))), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('compute', 'pack'):
        p = sub.add_parser(name)
        p.add_argument('--cases', required=True)
        p.add_argument('--assign', required=True)
        p.add_argument('--ranks', required=True, help='rank list, e.g. 0-3 or 0,1,2,3')
        p.add_argument('--local', required=True)
    sub.choices['compute'].add_argument('--h5-dir', required=True)
    sub.choices['compute'].add_argument('--cads-dir', required=True)
    sub.choices['compute'].add_argument('--procs', type=int, default=72)
    sub.choices['pack'].add_argument('--out', required=True)
    p = sub.add_parser('unpack')
    p.add_argument('--pack', required=True)
    p.add_argument('--local', required=True)
    a = ap.parse_args()
    dict(compute=compute, pack=pack, unpack=unpack)[a.cmd](a)


if __name__ == '__main__':
    main()
