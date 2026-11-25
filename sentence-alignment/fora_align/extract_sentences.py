"""Collect the sentence statements of every CT report used for alignment (Methods, "Sentence alignment").

Each report group of the CT H5 shards stores ``report/there_is_statements``: a JSON list of normalized statements of the
form "There is ..." / "There is no ...". This script reads them for the reports listed in the plan files (JSON lines with
``id``, ``h5``, ``patient``, ``report``) and writes ``{case id: [statements]}``, keeping statements of at least 4
characters after stripping.

    python -m fora_align.extract_sentences --reports plan/reports_train.jsonl plan/reports_val.jsonl \
        --h5-folder H5 --out sentences.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from functools import partial
from multiprocessing import Pool

import h5py


def work(item, h5_folder: str):
    h5, reports = item
    out, missing = {}, 0
    try:
        with h5py.File(f"{h5_folder}/{h5}.h5", "r") as f:
            for cid, patient, report in reports:
                try:
                    s = f[f"{patient}/{report}/report/there_is_statements"][()]
                    s = json.loads(s.decode() if isinstance(s, bytes) else s)
                    out[cid] = [x.strip() for x in s if isinstance(x, str) and len(x.strip()) >= 4]
                except Exception:
                    missing += 1
    except Exception:
        return {}, len(reports)
    return out, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", nargs="+", required=True)
    ap.add_argument("--h5-folder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    by_shard: dict[str, list] = {}
    for path in a.reports:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                by_shard.setdefault(r["h5"], []).append((r["id"], r["patient"], r["report"]))
    result, missing = {}, 0
    with Pool(a.workers) as pool:
        for out, miss in pool.imap_unordered(partial(work, h5_folder=a.h5_folder), sorted(by_shard.items()), chunksize=4):
            result.update(out)
            missing += miss
    with open(a.out, "w") as f:
        json.dump(result, f)
    n = [len(v) for v in result.values()]
    print(json.dumps(dict(cases=len(result), missing=missing, median=statistics.median(n) if n else 0,
                          max=max(n, default=0), empty=sum(x == 0 for x in n))), flush=True)


if __name__ == "__main__":
    main()
