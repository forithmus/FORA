#!/usr/bin/env python3
"""Greedy Findings generation with a chest X-ray writer checkpoint (paper section "Grounded report generation", X-ray
decoding). torchrun; every rank holds the whole Gemma (no FSDP).

Visual prefix from the study bag, frozen-Platt label tokens, plan text + finding tokens, greedy decoding (1 beam,
up to 1,024 new tokens) with the Gemma-4 stop ids. --plan_rules (default on) applies the device rule and the high-AUROC
promotion of ``fora.reportgen.cxr.plan_rules``; the PU-DPO base reports are generated with --no-plan_rules.
Output: one JSONL (rank 0 merges), one record per study.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from fora.reportgen.cxr import location as cxr_location
from fora_report.cxr.data import DenseLogits, findings_text, read_cases
from fora.reportgen.cxr.model import (STUDY_TYPE_CXR, build_report_model, load_llm, load_trainable, make_plan_builder,
                                      plan_inputs, stop_ids)
from fora.reportgen.cxr.plan_rules import inference_plan_builder_class
from fora.reportgen.cxr.stream import CXRVisualStream
from fora.reportgen.writer.model import generation_inputs

PATH_KEYS = ('calibration', 'dense', 'mil_config', 'mil_checkpoint', 'label_db', 'locations')


def main(argv=None):
    import os
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--llm_path', required=True)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--h5_root', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max_new_tokens', type=int, default=1024)
    ap.add_argument('--out', required=True)
    ap.add_argument('--plan_rules', action=argparse.BooleanOptionalAction, default=True)
    for key in PATH_KEYS:
        ap.add_argument(f'--{key}', default=None, help=f'override the {key} recorded in the checkpoint')
    a = ap.parse_args(argv)
    rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE']); local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(hours=4))

    package = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    meta = package['metadata']
    targs = argparse.Namespace(**meta['arguments'])
    for key in PATH_KEYS:
        if getattr(a, key) is not None:
            setattr(targs, key, getattr(a, key))
    if not meta.get('plan_probability', True):
        raise RuntimeError('the X-ray writer is trained with probability plan text')
    calibration = json.loads(Path(targs.calibration).read_text())
    labels = list(calibration['labels'])
    dense = DenseLogits(targs.dense, labels, calibration)
    locations = getattr(targs, 'locations', '')
    if locations and not getattr(targs, 'label_db', None):
        raise SystemExit('the checkpoint uses location text: pass --label_db')
    locstore = cxr_location.open_store(locations, labels, getattr(targs, 'label_db', None))
    stream = CXRVisualStream(targs.mil_config, targs.mil_checkpoint, device, variant=targs.mil_variant)
    llm, tokenizer, llm_dim = load_llm(a.llm_path, device, targs.lora_r, targs.lora_alpha)
    builder_class = inference_plan_builder_class() if a.plan_rules else None
    plan_builder = make_plan_builder(targs.calibration, labels, tokenizer, builder_class)
    model = build_report_model(llm, tokenizer, llm_dim, labels, plan_builder, device, targs)
    load_trainable(model, package['trainable_model'], device)
    del package
    model.eval()
    model.llm.config.use_cache = True
    stops = stop_ids(tokenizer, a.llm_path)
    study_types = torch.tensor([STUDY_TYPE_CXR], dtype=torch.long, device=device)

    cases = read_cases(a.cases)
    if a.limit:
        cases = cases[: a.limit]
    records = []
    for case in cases[rank::world]:
        t0 = time.time()
        bag = stream.prepare(case, h5_root=a.h5_root)
        enc = stream.encode(bag)
        del bag
        logits = dense.get(case['id'], device)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            prefix = model.visual_prefix_from_bag(enc, study_types)
            plan, features = plan_inputs(plan_builder, logits, enc.pooled, device, targs.finding_tokens,
                                         locstore.texts(case['id']) if locstore else None)
            embeddings, attention, token_types = generation_inputs(model, prefix, logits, plan, features)
            embedding = model.llm.get_input_embeddings()
            ids = model.llm.generate(inputs_embeds=embeddings.to(embedding.weight.dtype), attention_mask=attention,
                                     token_type_ids=token_types, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     num_beams=1, use_cache=True, eos_token_id=stops, pad_token_id=stops[0],
                                     logits_to_keep=1)
        selection = plan_builder.plan_selection(logits)
        records.append(dict(id=case['id'], generated=tokenizer.decode(ids[0], skip_special_tokens=True).strip(),
                            reference=findings_text(case, a.h5_root), new_tokens=int(ids.shape[-1]),
                            confident=[labels[i] for i in selection.confident],
                            possible=[labels[i] for i in selection.possible], films=enc.films,
                            visual_tokens=enc.total_tokens, seconds=round(time.time() - t0, 1)))
        print(json.dumps(dict(event='generated', rank=rank, id=case['id'], seconds=records[-1]['seconds'],
                              new_tokens=records[-1]['new_tokens'])), flush=True)
    gathered = [None] * world
    dist.all_gather_object(gathered, records)
    if rank == 0:
        out = Path(a.out)
        tmp = out.with_name(out.name + '.tmp')
        with open(tmp, 'w') as f:
            for part in gathered:
                for r in part:
                    f.write(json.dumps(r) + '\n')
        tmp.replace(out)
        print(json.dumps(dict(event='done', records=sum(len(p) for p in gathered), out=str(out))), flush=True)
    dist.barrier(device_ids=[local])
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
