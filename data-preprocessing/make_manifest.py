"""Extraction manifest: encoder checkpoint, encoding settings, plan files and output directories.

The manifest fingerprint (sha256 of the manifest without `dirs`) identifies the cache. It is stored in every series
file, in the study-bag manifest and in the rank assignment, and it names the node-local cache directory.

Usage
  python make_manifest.py --plan PLAN_DIR --checkpoint ENCODER_CKPT --store STORE_DIR --out manifest.json
         [--encoding configs/encoding.json] [--workers 128]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(plan: str, checkpoint: str, store: str, encoding: dict, workers: int) -> dict:
    plan_files = {sp: dict(reports=f'{plan}/reports_{sp}.jsonl', assign=f'{plan}/assign_{sp}.json',
                           reports_bytes=os.path.getsize(f'{plan}/reports_{sp}.jsonl')) for sp in ['train', 'val']}
    enc = {k: v for k, v in encoding.items() if k not in ('inference_brick_batch', 'encoder_step', 'patch')}
    m = dict(workers=workers,
             encoder=dict(checkpoint=str(checkpoint), step=int(encoding['encoder_step']), patch=list(encoding['patch'])),
             encoding=enc, execution=dict(inference_brick_batch=int(encoding['inference_brick_batch'])),
             splits=['train', 'val'], plan=plan_files)
    m['fingerprint'] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
    m['dirs'] = dict(store=str(store))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--checkpoint', required=True, help='exported teacher checkpoint of the CT encoder')
    ap.add_argument('--store', required=True, help='durable directory for packs and journals')
    ap.add_argument('--encoding', default=str(HERE / 'configs/encoding.json'))
    ap.add_argument('--workers', type=int, default=128)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    m = build(a.plan, a.checkpoint, a.store, json.loads(Path(a.encoding).read_text()), a.workers)
    Path(a.out).write_text(json.dumps(m, indent=1))
    print(m['fingerprint'])


if __name__ == '__main__':
    main()
