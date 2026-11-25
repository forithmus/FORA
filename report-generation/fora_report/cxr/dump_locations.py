#!/usr/bin/env python3
"""Per (study, class) location statistics for the X-ray writer plan (paper section "Grounded report generation",
X-ray plan location text; rules in ``fora.reportgen.cxr.location``). torchrun; one npz per split, gathered on rank 0.

    torchrun ... -m fora_report.cxr.dump_locations --cases cases_val.jsonl --out locations_val.npz \\
        --mil_config mil_config.json --mil_checkpoint mil.pt
npz: ids (n,), classes (C,), mass (n, C) f16 positive LOO evidence on frontal films, frac_right (n, C) f16, zone (n, C, 3)
f16 (upper, mid, lower shares), logit (n, C) f32 online bag logit, frontal_films (n,), films (n,).
"""
import argparse
import json
import os
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist

from fora_report.cxr.data import Prefetcher, read_cases
from fora.reportgen.cxr.stream import CXRVisualStream


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--h5_root', required=True)
    ap.add_argument('--mil_config', required=True)
    ap.add_argument('--mil_checkpoint', required=True)
    ap.add_argument('--mil_variant', default='raw')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(argv)
    rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE']); local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(hours=4))
    stream = CXRVisualStream(a.mil_config, a.mil_checkpoint, device, variant=a.mil_variant)
    cases = read_cases(a.cases)
    if a.limit:
        cases = cases[: a.limit]
    order = list(range(rank, len(cases), world))
    loader = Prefetcher(stream, cases, order, a.h5_root, depth=4, workers=3)
    R = dict(ids=[], stats=[], logit=[], frontal=[], films=[])
    t0 = time.time()
    for k in range(len(order)):
        case, bag, _text = loader.get(k)
        nfront = int(sum(1 for f in case['films'][:64] if int(f[2]) == 0))
        enc = stream.encode(bag, location=True)
        R['ids'].append(case['id'])
        R['stats'].append(enc.location.astype(np.float16))
        R['logit'].append(enc.logits.float().cpu().numpy())
        R['frontal'].append(nfront)
        R['films'].append(enc.films)
        del bag, enc
        if rank == 0 and k % 100 == 0:
            print(json.dumps(dict(event='progress', done=k + 1, of=len(order), s=round(time.time() - t0, 1))), flush=True)
    loader.close()
    parts = [None] * world if rank == 0 else None
    dist.gather_object(R, parts, dst=0)
    if rank == 0:
        ids = np.array([u for p in parts for u in p['ids']])
        o = np.argsort(ids, kind='stable')
        st = np.concatenate([np.stack(p['stats']) for p in parts if p['stats']])[o]
        if len(set(ids.tolist())) != len(ids):
            print(json.dumps(dict(event='warning', duplicates=int(len(ids) - len(set(ids.tolist()))))), flush=True)
        tmp = a.out + '.tmp.npz'
        np.savez(tmp, ids=ids[o], classes=np.array(stream.classes), mass=st[:, :, 0], frac_right=st[:, :, 1],
                 zone=st[:, :, 2:5], logit=np.concatenate([np.stack(p['logit']) for p in parts if p['logit']])[o],
                 frontal_films=np.array([x for p in parts for x in p['frontal']])[o],
                 films=np.array([x for p in parts for x in p['films']])[o], mil_checkpoint=np.array(a.mil_checkpoint))
        os.replace(tmp, a.out)
        print(json.dumps(dict(event='done', studies=len(ids), out=a.out, seconds=round(time.time() - t0, 1))), flush=True)
    dist.barrier(device_ids=[local])
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
