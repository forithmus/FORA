"""Copy the cache series files of selected ranks from the durable packs to node-local storage.

Each series file of the reports assigned to the given ranks (assign.json) is copied from its pack (file, offset,
file_size in cases.jsonl) to <local>/<relative_file>. Reads are sorted by (pack, offset), run in a few threads and are
capped at --mbps in total. Files already present with the right size are skipped, so the copy can be re-run. A
STAGED.json marker is written at the end. The trainers read the same <local> root and restore any missing file from
its pack.

Usage
  python stage_local.py --cases cases.jsonl --assign assign.json --ranks 0-3 --local LOCAL [--mbps 700 --threads 6]
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from token_regions import parse_ranks, series_jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--assign', required=True)
    ap.add_argument('--ranks', required=True)
    ap.add_argument('--local', required=True)
    ap.add_argument('--mbps', type=float, default=700)
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    jobs = series_jobs(a.cases, a.assign, parse_ranks(a.ranks))
    rate = a.mbps * 1e6
    total = sum(j[2] for j in jobs)
    lock = threading.Lock()
    state = dict(bytes=0, t0=time.time(), done=0, skip=0)

    def throttle(n):
        with lock:
            state['bytes'] += n
            ahead = state['bytes'] / rate - (time.time() - state['t0'])
        if ahead > 0:
            time.sleep(ahead)

    def copy(j):
        src, off, size, rel = j
        dst = f'{a.local}/{rel}'
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            with lock:
                state['skip'] += 1
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + '.part'
        with open(src, 'rb') as fi, open(tmp, 'wb') as fo:
            fi.seek(off)
            left = size
            while left:
                b = fi.read(min(left, 16 << 20))
                if not b:
                    raise IOError(f'short read {src}@{off}')
                fo.write(b)
                left -= len(b)
                throttle(len(b))
        os.replace(tmp, dst)
        with lock:
            state['done'] += 1

    print(json.dumps(dict(event='start', files=len(jobs), TB=round(total / 1e12, 3))), flush=True)
    last = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        for k, _ in enumerate(ex.map(copy, jobs)):
            if time.time() - last > 120:
                last = time.time()
                print(json.dumps(dict(files=k + 1, of=len(jobs), GB=round(state['bytes'] / 1e9, 1), skipped=state['skip'])), flush=True)
    os.makedirs(a.local, exist_ok=True)
    json.dump(dict(ranks=a.ranks, files=len(jobs), bytes=total, seconds=time.time() - state['t0']), open(f'{a.local}/STAGED.json', 'w'))
    print(json.dumps(dict(event='done', seconds=round(time.time() - state['t0']))), flush=True)


if __name__ == '__main__':
    main()
