#!/usr/bin/env python3
"""Chest X-ray writer, supervised fine-tuning (paper section "Grounded report generation", X-ray arm, SFT stage).

Gemma-4-31B-it + LoRA r16/a32 (FSDP2/HSDP), trained from scratch on the frozen X-ray MIL token stream
(``fora.reportgen.cxr.stream.CXRVisualStream``). One study (all films) per GPU per optimizer step; global batch = world size.
Recipe (as for CT): full writer (Findings only, 'Findings:\\n' prefix + EOS, max 2,048 report tokens), predicted
diagnostic plan (all classes, frozen Platt label conditioner, plan text with calibrated probability and fold change vs
the train base rate, one evidence token per planned finding = attention-pooled 1048-d MIL feature, location text from
``fora.reportgen.cxr.location``), resampler 512 latents (depth 2, 8 heads, width 1024), connector lr 1e-4, LoRA lr 2e-5,
AdamW wd 0.05, cosine with 5% warm-up, grad-norm clip 1.0, all auxiliary loss weights 0, LoRA dropout off, Gemma
gradient checkpointing on, seed 42, 2 epochs. The epoch order is a seeded permutation with size-bucketed steps
(``fora_report.cxr.data.epoch_schedule``).

Checkpoints (DDP format: full trainable tensors on CPU + plain optimizer state): latest.pt every --save_every_steps /
--save_every_minutes and on SIGUSR1/SIGTERM, checkpoint_epoch00E.pt at each epoch end. Exact resume with --resume.
Launch with torchrun on nodes x 4 GPUs (see README).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from fora.reportgen.cxr import location as cxr_location
from fora_report.cxr.data import DenseLogits, Prefetcher, case_tokens, epoch_schedule, read_cases
from fora.reportgen.cxr.model import (STUDY_TYPE_CXR, build_report_model, cosine_scheduler, load_llm, make_plan_builder,
                                 plan_inputs, split_parameters, target_ids)
from fora.reportgen.cxr.stream import CXRVisualStream


def add_common_arguments(p: argparse.ArgumentParser) -> None:
    """Arguments shared by SFT, DPO and PU-DPO (DPO checks the model ones against its SFT checkpoint)."""
    p.add_argument('--mil_config', required=True, help='X-ray MIL config json (classifier stage)')
    p.add_argument('--mil_checkpoint', required=True, help='X-ray MIL checkpoint (raw weights, see --mil_variant)')
    p.add_argument('--mil_variant', choices=('ema', 'raw'), default='raw')
    p.add_argument('--labels', required=True, help='labels npz (classes, train_ids, train_Y) of the X-ray MIL')
    p.add_argument('--cases_train', required=True, help='train study list jsonl (one case per line)')
    p.add_argument('--h5_root', required=True, help='X-ray H5 directory (data stage)')
    p.add_argument('--label_db', required=True, help='X-ray label sqlite (labels stage; class groups for locations)')
    p.add_argument('--calibration', required=True)
    p.add_argument('--dense', required=True)
    p.add_argument('--llm_path', required=True, help='local snapshot of google/gemma-4-31B-it')
    p.add_argument('--results_dir', required=True)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--target_global_batch', type=int, default=0)
    p.add_argument('--connector_lr', type=float, default=1e-4)
    p.add_argument('--llm_lr', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_fraction', type=float, default=0.05)
    p.add_argument('--lora_r', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--max_report_tokens', type=int, default=2048)
    p.add_argument('--num_visual_tokens', type=int, default=512)
    p.add_argument('--resampler_depth', type=int, default=2)
    p.add_argument('--resampler_heads', type=int, default=8)
    p.add_argument('--dino_resampler_dim', type=int, default=1024)
    p.add_argument('--dino_study_types', type=int, default=32)
    p.add_argument('--finding_tokens', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--locations', default='', help='comma-separated locations_<split>.npz; empty = no location text')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--bucket', type=int, default=8)
    p.add_argument('--save_every_steps', type=int, default=100)
    p.add_argument('--save_every_minutes', type=float, default=30)
    p.add_argument('--stop_after_steps', type=int, default=0, help='smoke test: stop (and save latest.pt) after N steps')
    p.add_argument('--limit_train', type=int, default=0, help='smoke test: use only the first N train cases')
    p.add_argument('--resume', default='')
    p.add_argument('--loader_depth', type=int, default=4)
    p.add_argument('--loader_workers', type=int, default=3)
    p.add_argument('--pg_timeout_s', type=int, default=36000)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(p)
    return p.parse_args(argv)


def init_distributed(pg_timeout_s: int):
    rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE']); local = int(os.environ['LOCAL_RANK'])
    local_world = int(os.environ.get('LOCAL_WORLD_SIZE', '4'))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(seconds=pg_timeout_s))
    return rank, world, local, local_world, device


class StopFlag:
    """SIGUSR1 / SIGTERM -> save latest.pt at the next step boundary and exit 0 (agreed over all ranks)."""

    def __init__(self, device):
        self.flag = False
        self.device = device
        signal.signal(signal.SIGUSR1, self._set)
        signal.signal(signal.SIGTERM, self._set)

    def _set(self, *_):
        self.flag = True

    def requested(self) -> bool:
        t = torch.tensor([1.0 if self.flag else 0.0], device=self.device)
        dist.all_reduce(t)
        return bool(t.item() > 0)


def load_labels(path: str, labels: list[str]):
    L = np.load(path, allow_pickle=False)
    if [str(c) for c in L['classes'].tolist()] != labels:
        raise RuntimeError('label npz class order != calibration labels')
    return {str(i): np.flatnonzero(y) for i, y in zip(L['train_ids'].tolist(), L['train_Y'])}


def setup_writer(args, labels, device, world, local_world, log):
    """LLM + LoRA, plan builder, X-ray writer, FSDP sharding (Gemma-4 path only)."""
    from fora_report.writer import fsdp as reportgen_fsdp
    from fora.reportgen.writer.llm import language_layers
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    llm, tokenizer, llm_dim = load_llm(args.llm_path, device, args.lora_r, args.lora_alpha)
    family = llm.fora_llm_family
    if not reportgen_fsdp.fsdp_requested(family):
        raise RuntimeError('the X-ray writer supports the FSDP (Gemma-4) path only')
    report = reportgen_fsdp.train_mode_for_checkpointing(llm, False)   # LoRA dropout off
    if not bool(llm.get_base_model().is_gradient_checkpointing):
        raise RuntimeError('Gemma gradient checkpointing is not active after PEFT wrapping')
    log(event='llm', family=family, hidden=llm_dim, train_mode=report)
    plan_builder = make_plan_builder(args.calibration, labels, tokenizer)
    model = build_report_model(llm, tokenizer, llm_dim, labels, plan_builder, device, args)
    mesh = reportgen_fsdp.build_mesh('cuda', world, local_world)
    reportgen_fsdp.shard_report_model(model, language_layers(llm), mesh)
    torch.cuda.empty_cache()
    log(event='fsdp', mesh=list(mesh.shape), memory=reportgen_fsdp.describe_memory(device))
    return llm, tokenizer, plan_builder, model, mesh


def main(argv=None):
    from fora_report.writer import fsdp as reportgen_fsdp
    args = parse_args(argv)
    rank, world, local, local_world, device = init_distributed(args.pg_timeout_s)
    main_rank = rank == 0
    results = Path(args.results_dir)
    if main_rank:
        results.mkdir(parents=True, exist_ok=True)
    target = args.target_global_batch or world
    if target != world:
        raise ValueError(f'one study per GPU per step: target_global_batch must equal world ({world}), got {target}')

    def log(**kw):
        if main_rank:
            print(json.dumps(kw), flush=True)

    calibration = json.loads(Path(args.calibration).read_text())
    labels = list(calibration['labels'])
    Y = load_labels(args.labels, labels)
    dense = DenseLogits(args.dense, labels, calibration)
    locstore = cxr_location.open_store(args.locations, labels, args.label_db)
    cases = read_cases(args.cases_train)
    if args.limit_train:
        cases = cases[: args.limit_train]
    sizes = np.array([case_tokens(c) for c in cases], np.int64)
    steps_per_epoch = math.ceil(len(cases) / world)
    total_steps = args.epochs * steps_per_epoch
    log(event='data', train_cases=len(cases), world=world, steps_per_epoch=steps_per_epoch, total_steps=total_steps,
        token_p50=int(np.median(sizes)), token_p99=int(np.percentile(sizes, 99)), token_max=int(sizes.max()))

    stream = CXRVisualStream(args.mil_config, args.mil_checkpoint, device, variant=args.mil_variant)
    if stream.classes != labels:
        raise RuntimeError('MIL checkpoint classes != calibration labels')
    llm, tokenizer, plan_builder, model, _mesh = setup_writer(args, labels, device, world, local_world, log)

    connector, adapters = split_parameters(model)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and name.startswith('llm.') and 'lora_' not in name:
            raise RuntimeError(f'unexpected trainable base parameter {name}')
    optimizer = torch.optim.AdamW([{'params': connector, 'lr': args.connector_lr},
                                   {'params': adapters, 'lr': args.llm_lr}],
                                  weight_decay=args.weight_decay, foreach=True)
    scheduler = cosine_scheduler(optimizer, total_steps, args.warmup_fraction)
    log(event='params', connector=sum(p.numel() for p in connector), lora=sum(p.numel() for p in adapters))

    schedule_key = dict(train_cases=len(cases), world=world, epochs=args.epochs, seed=args.seed, bucket=args.bucket)
    epoch, start_step, optimizer_step = 0, 0, 0
    if args.resume:
        package = torch.load(args.resume, map_location='cpu', weights_only=False)
        if package['metadata']['schedule'] != schedule_key:
            raise RuntimeError(f"resume schedule differs: {package['metadata']['schedule']}")
        reportgen_fsdp.load_full_trainable_state(model, package['trainable_model'])
        reportgen_fsdp.load_full_optimizer_state(optimizer, package['optimizer'])
        scheduler.load_state_dict(package['scheduler'])
        epoch, start_step, optimizer_step = int(package['epoch']), int(package['next_step']), int(package['optimizer_step'])
        del package
        log(event='resumed', path=args.resume, epoch=epoch, next_step=start_step, optimizer_step=optimizer_step)

    metadata = dict(arguments={**vars(args), 'visual_stream': 'cxr_mil_b24', 'mil_source': 'precomputed_cxr_mil',
                               'report_writer_mode': 'full', 'dataset_kind': 'cxr'},
                    mil_calibration=str(Path(args.calibration).resolve()), diagnostic_plan_mode='all_classes',
                    writer=dict(finding_feature_dim=1048 if args.finding_tokens else 0), labels=labels,
                    plan_probability=True, schedule=schedule_key)

    def save(name, next_epoch, next_step):
        t0 = time.time()
        state = reportgen_fsdp.full_trainable_state(model, main_rank)
        opt = reportgen_fsdp.full_optimizer_state(optimizer, main_rank)
        if main_rank:
            package = dict(trainable_model=state, optimizer=opt, scheduler=scheduler.state_dict(), epoch=next_epoch,
                           next_step=next_step, optimizer_step=optimizer_step, metadata=metadata)
            tmp = results / f'{name}.tmp'
            torch.save(package, tmp)
            tmp.replace(results / name)
            (results / 'latest_step.json').write_text(json.dumps(dict(file=name, epoch=next_epoch, next_step=next_step,
                                                                      optimizer_step=optimizer_step, time=time.time())))
        dist.barrier(device_ids=[local])
        log(event='saved', file=name, epoch=next_epoch, next_step=next_step, optimizer_step=optimizer_step,
            seconds=round(time.time() - t0, 1))

    stopper = StopFlag(device)
    model.train()
    study_types = torch.tensor([STUDY_TYPE_CXR], dtype=torch.long, device=device)
    modality_empty = torch.empty((0, args.dino_resampler_dim), dtype=torch.bfloat16, device=device)
    last_save = time.monotonic()
    steps_this_launch = 0
    stop = False
    while epoch < args.epochs:
        schedule = epoch_schedule(len(cases), sizes, world, epoch, args.seed, args.bucket)
        assert schedule.shape[0] == steps_per_epoch
        order = schedule[start_step:, rank]
        loader = Prefetcher(stream, cases, order, args.h5_root, depth=args.loader_depth, workers=args.loader_workers)
        window = []
        for k in range(len(order)):
            step = start_step + k
            t0 = time.time()
            case, bag, text = loader.get(k)
            t_load = time.time() - t0
            enc = stream.encode(bag)
            del bag
            t_enc = time.time() - t0 - t_load
            uid = case['id']
            logits = dense.get(uid, device)
            positives = torch.as_tensor(Y.get(uid, np.zeros(0, np.int64)), dtype=torch.long, device=device)
            plan, features = plan_inputs(plan_builder, logits, enc.pooled, device, args.finding_tokens,
                                         locstore.texts(uid) if locstore else None)
            targets = target_ids(tokenizer, text, args.max_report_tokens, device)
            consistency = plan_builder.consistency_targets(positives, device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(enc.tokens, modality_empty, plan, targets, positives, logits, plan, consistency,
                            finding_features=features, dino_view_kind=enc.view_kind, dino_view_series=enc.view_series,
                            dino_study_types=study_types)
            out['loss'].backward()
            grad_norm = reportgen_fsdp.clip_grad_norm([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            steps_this_launch += 1
            replay = float((enc.logits - logits.reshape(-1)).abs().max())
            stats = torch.tensor([float(out['loss'].detach()), float(out['report_loss']), float(targets.numel()),
                                  float(enc.total_tokens), float(plan.numel()), replay, time.time() - t0, t_load, t_enc,
                                  torch.cuda.max_memory_allocated(device) / 2**30], device=device)
            mean = stats.clone(); dist.all_reduce(mean); mean /= world
            peak = stats.clone(); dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            window.append(mean[6].item())
            if main_rank and (optimizer_step <= 20 or optimizer_step % 10 == 0):
                log(event='step', epoch=epoch, step=step, opt_step=optimizer_step, loss=round(mean[0].item(), 4),
                    report_loss=round(mean[1].item(), 4), target_tokens=int(mean[2].item()),
                    visual_tokens=int(mean[3].item()), plan_tokens=int(mean[4].item()),
                    replay_max=round(peak[5].item(), 4), s_step=round(mean[6].item(), 2),
                    s_step_max=round(peak[6].item(), 2), s_load=round(mean[7].item(), 2),
                    s_enc=round(mean[8].item(), 2), grad_norm=round(float(grad_norm), 3),
                    lr=[g['lr'] for g in optimizer.param_groups], peak_gib=round(peak[9].item(), 1),
                    s_step_avg10=round(float(np.mean(window[-10:])), 2))
            del out, enc, features, plan, targets
            done_epoch = step + 1 == steps_per_epoch
            stop = stopper.requested() or bool(args.stop_after_steps and steps_this_launch >= args.stop_after_steps)
            if done_epoch:
                break
            if stop:
                loader.close()
                save('latest.pt', epoch, step + 1)
                log(event='stopped', epoch=epoch, next_step=step + 1, optimizer_step=optimizer_step)
                dist.destroy_process_group()
                return 0
            if (args.save_every_steps and optimizer_step % args.save_every_steps == 0) or \
                    (time.monotonic() - last_save) / 60 >= args.save_every_minutes:
                save('latest.pt', epoch, step + 1)
                last_save = time.monotonic()
        loader.close()
        epoch += 1
        start_step = 0
        save(f'checkpoint_epoch{epoch:03d}.pt', epoch, 0)
        if main_rank:   # latest.pt = the epoch checkpoint (hard link)
            tmp = results / 'latest.pt.lnk'
            tmp.unlink(missing_ok=True)
            os.link(results / f'checkpoint_epoch{epoch:03d}.pt', tmp)
            tmp.replace(results / 'latest.pt')
        dist.barrier(device_ids=[local])
        last_save = time.monotonic()
        if stop:
            log(event='stopped', epoch=epoch, next_step=0, optimizer_step=optimizer_step)
            dist.destroy_process_group()
            return 0
    if main_rank:
        (results / 'COMPLETE.json').write_text(json.dumps(dict(epochs=args.epochs, optimizer_step=optimizer_step,
                                                               time=time.time())))
    log(event='complete', optimizer_step=optimizer_step)
    dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
