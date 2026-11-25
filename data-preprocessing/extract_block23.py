"""Block-23 token cache of every planned series and view (Methods, "Block-23 token cache").

Per report of this worker, every planned series is read from the CT H5 shard, reoriented to RAS, resampled to
1.5 x 0.5 x 0.5 mm, cropped in-plane to the body box (fora.geometry.tiling) and encoded for every view (FULL, native
windows, and the 10 mm lung slab MIP when planned) with the frozen teacher (fora.cache.encode). Each series becomes one
H5 file in the `fora_int8_tokens` format (fora.cache.format) holding all its views and the provenance needed to
project CADS masks onto the token grid later (affine, orig_shape, body_crop_yxyx, resampled_body_shape).

Storage: the series file is written node-locally (--local-dir/<fingerprint>/<split>/<report id>/series_NNN.h5), then
appended to this worker's durable pack `pack_worker_NNN.bin` in the manifest's store directory; the JSON journal
`index_worker_NNN.json` records every pack entry (file, offset, file_size, identity) and every completed report with
its series and views. The run resumes from the journal. A series that fails to load or encode is recorded under
journal['failed'] and skipped.

Usage (one process per GPU)
  python extract_block23.py --manifest manifest.json --worker W --local-dir /local/fora_cache [--device cuda]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import torch

from fora.cache.encode import Block23Encoder
from fora.encoder.blocks import block24_state
from fora.cache.format import FORMAT, quantize, valid_series_file, write_group, write_json_atomic
from fora.encoder.cache import restore_from_pack
from fora.geometry.series import read_series
from fora.geometry.tiling import TARGET_SPACING, crop_to_body, resample_series
from fora.geometry.windows import MIP_SLAB_MM, slab_mip, view_list


def load_series(h5_dir: str, rep: dict, s: dict):
    """One planned series -> (body-cropped volume, lung MIP volume or None, views, provenance)."""
    with h5py.File(f"{h5_dir}/{rep['h5']}.h5", 'r') as f:
        data, affine = read_series(f, rep['patient'], rep['report'], s['acc'], s['sid'], s['sub'], s['frame'])
    vol, _ = resample_series(data, affine)
    orig_shape = list(np.shape(data))
    del data
    vol, crop = crop_to_body(vol)
    views = view_list(s['windows'], s['mip'])
    mip = slab_mip(vol, slab_mm=MIP_SLAB_MM, z_spacing=TARGET_SPACING[0]) if s['mip'] is not None else None
    prov = dict(h5=rep['h5'], patient=rep['patient'], report=rep['report'], acc=s['acc'], sid=s['sid'], sub=s['sub'],
                frame=s['frame'], affine=np.asarray(affine).tolist(), orig_shape=orig_shape, body_crop_yxyx=crop,
                resampled_body_shape=list(vol.shape), target_spacing=list(TARGET_SPACING))
    return vol, mip, views, prov


def encode_series(encoder: Block23Encoder, encoding: dict, vol, mip, views, prov, out: Path, identity: str, fingerprint: str):
    """Encode all views of one series into a new cache series file (written to a .partial file, then renamed)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.partial')
    flat = None
    with h5py.File(tmp, 'w') as hf:
        hf.attrs['identity'] = identity
        hf.attrs['fingerprint'] = fingerprint
        hf.attrs['format'] = FORMAT
        hf.attrs['provenance_json'] = json.dumps(prov)
        for vi, w in enumerate(views):
            v = mip if w['kind'] == 'lung_mip_10mm' else vol
            h23, crop_ids, indices, meta = encoder.encode(v, (w['lo'], w['hi']), vol, encoding)
            if flat is None:
                flat = indices
            else:
                assert np.array_equal(flat, indices)
            c = quantize(h23, flat, meta['grid'])
            assert np.array_equal(c['crop_ids'], crop_ids)
            del h23
            write_group(hf, f'v{vi:02d}', c, meta, w)
        hf.create_dataset('flat_indices', data=flat)
        hf.attrs['complete'] = 1
        hf.flush()
    tmp.replace(out)
    return len(flat)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--h5-dir', required=True, help='directory of CT H5 shards')
    ap.add_argument('--worker', required=True, type=int)
    ap.add_argument('--local-dir', required=True, type=Path, help='node-local scratch for series files')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-cases', type=int, default=0, help='stop after this many new reports (smoke tests)')
    a = ap.parse_args()
    m = json.loads(a.manifest.read_text())
    fp = m['fingerprint']
    local = a.local_dir / fp
    store = Path(m['dirs']['store'])
    store.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    torch.set_num_threads(4)
    encoder = None
    start = time.time()
    finished = done = 0
    last_flush = time.time()
    journalpath = store / f'index_worker_{a.worker:03d}.json'
    packpath = store / f'pack_worker_{a.worker:03d}.bin'
    journal = (json.loads(journalpath.read_text()) if journalpath.exists()
               else dict(fingerprint=fp, end=0, reconstructions={}, cases={}, failed={}))
    assert journal['fingerprint'] == fp
    journal.setdefault('failed', {})
    if journal['end']:
        assert packpath.exists() and packpath.stat().st_size >= journal['end'], 'Committed feature pack missing/truncated'
    with packpath.open('a+b') as packed:
        packed.truncate(journal['end'])
    for split in m['splits']:
        mine = set(json.loads(Path(m['plan'][split]['assign']).read_text())['assignments'][a.worker])
        reports = [r for r in (json.loads(line) for line in open(m['plan'][split]['reports'])) if r['id'] in mine] if mine else []
        reports.sort(key=lambda r: r['id'])
        for rep in reports:
            if rep['id'] in journal['cases']:
                finished += 1
                continue
            if a.max_cases and done >= a.max_cases:
                break
            done += 1
            result = dict(id=rep['id'], split=split, report=rep['report'], fingerprint=fp, views=[], series=[])
            for si, s in enumerate(rep['series']):
                rel = Path(split) / rep['id'] / f'series_{si:03d}.h5'
                nodefile = local / rel
                key = f"{rep['id']}/{si}"
                identity = hashlib.sha256(json.dumps(dict(protocol=fp, report=rep['id'], series=s), sort_keys=True).encode()).hexdigest()
                archived = journal['reconstructions'].get(key)
                if key in journal['failed']:
                    continue
                if not valid_series_file(nodefile, identity) and archived:
                    restore_from_pack(archived, nodefile)
                if not valid_series_file(nodefile, identity):
                    try:
                        began = time.time()
                        vol, mip, views, prov = load_series(a.h5_dir, rep, s)
                        if encoder is None:
                            from fora.encoder.backbone import load_teacher
                            backbone = load_teacher(m['encoder']['checkpoint'], device=device)
                            encoder = Block23Encoder(backbone, device, brick_batch=int(m['execution'].get('inference_brick_batch', 4)))
                            sf = store / 'block24_state.pt'
                            if a.worker == 0 and not sf.exists():
                                t = sf.with_suffix('.tmp')
                                torch.save(block24_state(backbone), t)
                                t.replace(sf)
                        n = encode_series(encoder, m['encoding'], vol, mip, views, prov, nodefile, identity, fp)
                        del vol, mip
                        print(json.dumps(dict(worker=a.worker, split=split, report=rep['id'], series=si, views=len(views),
                                              tokens=n, seconds=round(time.time() - began, 2), finished_reports=finished,
                                              elapsed=round(time.time() - start))), flush=True)
                    except Exception as e:
                        journal['failed'][key] = dict(error=repr(e)[:500], trace=traceback.format_exc()[-800:])
                        write_json_atomic(journalpath, journal)
                        print(json.dumps(dict(event='series_failed', worker=a.worker, key=key, error=repr(e)[:300])), flush=True)
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                if not archived:
                    offset = journal['end']
                    size = nodefile.stat().st_size
                    with packpath.open('r+b') as packed, nodefile.open('rb') as source:
                        packed.seek(offset)
                        shutil.copyfileobj(source, packed, length=8 * 1024 ** 2)
                        packed.flush()
                        os.fsync(packed.fileno())
                        assert packed.tell() == offset + size
                    archived = dict(file=str(packpath), offset=offset, file_size=size, identity=identity)
                    journal['reconstructions'][key] = archived
                    journal['end'] = offset + size
                with h5py.File(nodefile, 'r') as hf:
                    grid = json.loads(hf['v00'].attrs['metadata_json'])['grid']
                    n = len(hf['flat_indices'])
                    result['series'].append(dict(index=si, acc=s['acc'], sid=s['sid'], relative_file=str(rel), **archived))
                    for g in sorted(k for k in hf.keys() if k.startswith('v')):
                        result['views'].append(dict(**archived, relative_file=str(rel), series=si, group=g, n_tokens=n,
                                                    n_crops=len(hf[g]['crop_ids']), grid=grid,
                                                    window=json.loads(hf[g].attrs['window_json'])))
            result['complete'] = True
            journal['cases'][rep['id']] = result
            finished += 1
            if finished % 10 == 0 or time.time() - last_flush > 300:
                write_json_atomic(journalpath, journal)
                last_flush = time.time()
    write_json_atomic(journalpath, journal)
    print(json.dumps(dict(worker=a.worker, complete=True, fingerprint=fp, finished_reports=finished,
                          failed_series=len(journal['failed']), seconds=round(time.time() - start))), flush=True)


if __name__ == '__main__':
    main()
