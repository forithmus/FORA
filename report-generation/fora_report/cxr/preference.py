"""Shared preference-optimisation loop of the chest X-ray writer (paper section "Grounded report generation",
X-ray DPO and PU-DPO stages).

The policy starts from an SFT (or DPO) checkpoint, which is also the frozen reference (``FSDPReferenceWeights``: the
start tensors as local FSDP shards, swapped in under no_grad for the reference scores; the start is checked to be
exact, max|policy - reference| == 0). Sequence score = sum log p over 'Findings:\\n' + text + EOS. Defaults: connector
lr 2e-5, LoRA lr 5e-6, warm-up 2%, cosine over --max_steps, AdamW wd 0.05, clip 1.0, one study per GPU per step,
dropout off, step_000N00.pt every 100 steps (latest.pt = hard link), exact resume with --resume.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from fora.reportgen.cxr import location as cxr_location
from fora_report.cxr import train_sft as S
from fora_report.cxr.data import DenseLogits, Prefetcher, case_tokens, epoch_schedule
from fora.reportgen.cxr.model import STUDY_TYPE_CXR, cosine_scheduler, plan_inputs, split_parameters, target_ids
from fora.reportgen.cxr.stream import CXRVisualStream

PREFERENCE_DEFAULTS = dict(connector_lr=2e-5, llm_lr=5e-6, warmup_fraction=0.02, save_every_steps=100)
CHECKED_KEYS = ('dense', 'mil_config', 'mil_checkpoint', 'mil_variant', 'lora_r', 'lora_alpha', 'num_visual_tokens',
                'resampler_depth', 'resampler_heads', 'dino_resampler_dim', 'dino_study_types', 'finding_tokens',
                'max_report_tokens', 'locations')


def parse_preference_args(argv=None, add_extra=None):
    """Common writer arguments (``train_sft``) with the preference defaults + the DPO arguments."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--dpo_pairs', required=True)
    p.add_argument('--sft', required=True, help='start checkpoint = warm start = frozen reference')
    p.add_argument('--max_steps', type=int, default=600)
    p.add_argument('--dpo_beta', type=float, default=0.1)
    p.add_argument('--dpo_weight', type=float, default=1.0)
    p.add_argument('--dpo_chosen_ce_weight', type=float, default=1.0)
    p.add_argument('--dpo_unpaired_ce_weight', type=float, default=1.0)
    p.add_argument('--dpo_pairs_per_visit', type=int, default=2)
    p.add_argument('--dpo_mix', type=float, default=1.0)
    p.add_argument('--dpo_recompute_tolerance', type=float, default=0.5)
    p.add_argument('--dpo_recompute_abort', type=float, default=20.0)
    p.add_argument('--log_every_steps', type=int, default=5)
    if add_extra:
        add_extra(p)
    d, rest = p.parse_known_args(argv)
    rest_keys = {r.split('=')[0] for r in rest if r.startswith('--')}
    extra = []
    for k, v in PREFERENCE_DEFAULTS.items():
        if f'--{k}' not in rest_keys:
            extra += [f'--{k}', str(v)]
    args = S.parse_args(rest + extra)
    for k, v in vars(d).items():
        setattr(args, k, v)
    return args


def run_preference(args, *, store, select_cases, objective: str, train_step, stats_types=None):
    """Run the preference loop.

    select_cases(all_train_cases) -> the visited case list (with repeats); train_step(env) -> (value, n_pairs) does the
    backward of one study (env carries uid, text, chosen ids, forward/score/sync closures, reference, stats, ...)."""
    from fora_report.ct.dpo_loss import DPOStatistics, disable_preference_dropout
    from fora_report.writer import fsdp as reportgen_fsdp
    from fora_report.writer import fsdp_dpo as reportgen_fsdp_dpo

    rank, world, local, local_world, device = S.init_distributed(args.pg_timeout_s)
    main_rank = rank == 0
    results = Path(args.results_dir)
    if main_rank:
        results.mkdir(parents=True, exist_ok=True)

    def log(**kw):
        if main_rank:
            print(json.dumps(kw), flush=True)

    sft = torch.load(args.sft, map_location='cpu', weights_only=False, mmap=True)
    sft_meta = sft['metadata']
    sft_args = sft_meta['arguments']
    if Path(sft_args['calibration']).resolve() != Path(args.calibration).resolve():
        raise RuntimeError(f"start calibration {sft_args['calibration']} != preference calibration {args.calibration}")
    for key in CHECKED_KEYS:
        if str(sft_args[key]) != str(getattr(args, key)):
            raise RuntimeError(f'{key}: start checkpoint {sft_args[key]} != {getattr(args, key)}')
    reference_state = sft['trainable_model']
    sft_step = sft.get('optimizer_step')

    calibration = json.loads(Path(args.calibration).read_text())
    labels = list(calibration['labels'])
    Y = S.load_labels(args.labels, labels)
    dense = DenseLogits(args.dense, labels, calibration)
    locstore = cxr_location.open_store(args.locations, labels, args.label_db)
    from fora_report.cxr.data import read_cases
    cases = select_cases(read_cases(args.cases_train))
    sizes = np.array([case_tokens(c) for c in cases], np.int64)
    steps_per_epoch = len(cases) // world
    total_steps = args.max_steps
    log(event='data', train_cases=len(cases), world=world, steps_per_epoch=steps_per_epoch, max_steps=total_steps,
        pairs=store.summary['pairs_by_type'])

    stream = CXRVisualStream(args.mil_config, args.mil_checkpoint, device, variant=args.mil_variant)
    llm, tokenizer, plan_builder, model, mesh = S.setup_writer(args, labels, device, world, local_world, log)
    missing = reportgen_fsdp_dpo.load_exact_warm_start(model, reference_state)
    if missing:
        raise RuntimeError(f'start checkpoint lacks trainable tensors: {missing[:5]}')
    dropout_changes = disable_preference_dropout(model)
    reference = reportgen_fsdp_dpo.FSDPReferenceWeights(model, reference_state, device)
    del reference_state, sft
    slot_group = reportgen_fsdp_dpo.slot_group(mesh)
    adapter = reportgen_fsdp.FSDPSyncAdapter(model)

    connector, adapters = split_parameters(model)
    optimizer = torch.optim.AdamW([{'params': connector, 'lr': args.connector_lr},
                                   {'params': adapters, 'lr': args.llm_lr}],
                                  weight_decay=args.weight_decay, foreach=True)
    scheduler = cosine_scheduler(optimizer, total_steps, args.warmup_fraction)
    schedule_key = dict(paired_cases=len(cases), world=world, max_steps=total_steps, seed=args.seed, bucket=args.bucket,
                        pairs=str(Path(args.dpo_pairs).resolve()), sft=str(Path(args.sft).resolve()))
    start = 0
    if args.resume:
        package = torch.load(args.resume, map_location='cpu', weights_only=False)
        if package['metadata']['dpo']['schedule'] != schedule_key:
            raise RuntimeError(f"resume schedule differs: {package['metadata']['dpo']['schedule']}")
        reportgen_fsdp.load_full_trainable_state(model, package['trainable_model'])
        reportgen_fsdp.load_full_optimizer_state(optimizer, package['optimizer'])
        scheduler.load_state_dict(package['scheduler'])
        start = int(package['optimizer_step'])
        del package
        log(event='resumed', path=args.resume, optimizer_step=start)
    else:
        diff = reference.max_abs_difference()
        if diff != 0.0:
            raise RuntimeError(f'preference training must start exactly from the reference: max|diff|={diff:g}')
    log(event='dpo', beta=args.dpo_beta, k=args.dpo_pairs_per_visit, mix=args.dpo_mix,
        lr=[args.connector_lr, args.llm_lr], warmup=args.warmup_fraction, ref_tensors=len(reference.items),
        ref_numel=reference.numel, dropout_disabled=len(dropout_changes), sft=args.sft, sft_step=sft_step,
        memory=reportgen_fsdp.describe_memory(device))

    metadata = json.loads(json.dumps(sft_meta, default=str))
    metadata['arguments'] = {**metadata['arguments'], **{k: v for k, v in vars(args).items() if k.startswith('dpo_')},
                             'connector_lr': args.connector_lr, 'llm_lr': args.llm_lr}
    metadata['dpo'] = dict(schedule=schedule_key, objective=objective, pairs_summary=store.summary,
                           warmup_fraction=args.warmup_fraction)

    def save(step_done):
        t0 = time.time()
        state = reportgen_fsdp.full_trainable_state(model, main_rank)
        opt = reportgen_fsdp.full_optimizer_state(optimizer, main_rank)
        name = f'step_{step_done:06d}.pt'
        if main_rank:
            package = dict(trainable_model=state, optimizer=opt, scheduler=scheduler.state_dict(), epoch=0,
                           next_step=step_done, optimizer_step=step_done, metadata=metadata)
            tmp = results / f'{name}.tmp'
            torch.save(package, tmp)
            tmp.replace(results / name)
            lnk = results / 'latest.pt.lnk'
            lnk.unlink(missing_ok=True)
            os.link(results / name, lnk)
            lnk.replace(results / 'latest.pt')
            (results / 'latest_step.json').write_text(json.dumps(dict(file=name, optimizer_step=step_done,
                                                                      time=time.time())))
        dist.barrier(device_ids=[local])
        log(event='saved', file=name, optimizer_step=step_done, seconds=round(time.time() - t0, 1))

    stopper = S.StopFlag(device)
    model.train()
    stats = DPOStatistics(device)
    if stats_types is not None:
        stats.types = stats_types
        stats.values = torch.zeros(len(stats_types), len(DPOStatistics.FIELDS), device=device, dtype=torch.float64)
    study_types = torch.tensor([STUDY_TYPE_CXR], dtype=torch.long, device=device)
    modality_empty = torch.empty((0, args.dino_resampler_dim), dtype=torch.bfloat16, device=device)
    schedule = np.concatenate([epoch_schedule(len(cases), sizes, world, e, args.seed, args.bucket)
                               for e in range(-(-total_steps // max(1, steps_per_epoch)))])[:total_steps]
    order = schedule[start:, rank]
    loader = Prefetcher(stream, cases, order, args.h5_root, depth=args.loader_depth, workers=args.loader_workers)
    max_tok = args.max_report_tokens
    window, launched = [], 0
    for k in range(len(order)):
        step = start + k
        t0 = time.time()
        case, bag, text = loader.get(k)
        enc = stream.encode(bag)
        del bag
        t_enc = time.time() - t0
        uid = case['id']
        logits = dense.get(uid, device)
        positives = torch.as_tensor(Y.get(uid, np.zeros(0, np.int64)), dtype=torch.long, device=device)
        plan, features = plan_inputs(plan_builder, logits, enc.pooled, device, args.finding_tokens,
                                     locstore.texts(uid) if locstore else None)
        consistency = plan_builder.consistency_targets(positives, device)
        chosen = target_ids(tokenizer, text, max_tok + 1, device)

        def forward(target):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return adapter(enc.tokens, modality_empty, plan, target, positives, logits, plan, consistency,
                               finding_features=features, dino_view_kind=enc.view_kind,
                               dino_view_series=enc.view_series, dino_study_types=study_types)

        def score(kind, target):
            out = forward(target)
            if not torch.isclose(out['loss'].detach().float(), out['report_loss'].float(), rtol=1e-4, atol=1e-5):
                raise RuntimeError('preference training needs loss == token-mean report NLL')
            return -out['loss'].float() * target.numel(), out

        def sync(is_last):
            return contextlib.nullcontext() if is_last else adapter.no_sync()

        env = SimpleNamespace(args=args, step=step, rank=rank, uid=uid, text=text, chosen=chosen, max_tok=max_tok,
                              tokenizer=tokenizer, device=device, store=store, stats=stats, reference=reference,
                              slot_group=slot_group, forward=forward, score=score, sync=sync,
                              steps_per_epoch=steps_per_epoch)
        value, n_pairs = train_step(env)
        grad_norm = reportgen_fsdp.clip_grad_norm([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        done = step + 1
        launched += 1
        st = torch.tensor([value, float(n_pairs), time.time() - t0, t_enc,
                           torch.cuda.max_memory_allocated(device) / 2**30], device=device)
        mean = st.clone(); dist.all_reduce(mean); mean /= world
        peak = st.clone(); dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        window.append(mean[2].item())
        if done <= 10 or done % args.log_every_steps == 0:
            line = stats.line(done, lambda t: dist.all_reduce(t))
            log(event='step', opt_step=done, loss=round(mean[0].item(), 4), pairs_per_study=round(mean[1].item(), 2),
                s_step=round(mean[2].item(), 2), s_step_max=round(peak[2].item(), 2), s_enc=round(mean[3].item(), 2),
                s_step_avg10=round(float(np.mean(window[-10:])), 2), grad_norm=round(float(grad_norm), 3),
                lr=[g['lr'] for g in optimizer.param_groups], peak_gib=round(peak[4].item(), 1))
            if main_rank:
                print(line, flush=True)
        del enc, features, plan, chosen, env
        stop = stopper.requested() or bool(args.stop_after_steps and launched >= args.stop_after_steps)
        if done == total_steps or (args.save_every_steps and done % args.save_every_steps == 0) or stop:
            save(done)
        if stop and done < total_steps:
            loader.close()
            log(event='stopped', optimizer_step=done)
            dist.destroy_process_group()
            return 0
    loader.close()
    if main_rank:
        (results / 'COMPLETE.json').write_text(json.dumps(dict(max_steps=total_steps, time=time.time())))
    log(event='complete', optimizer_step=total_steps)
    dist.destroy_process_group()
    return 0


def plain_ce_step(env):
    """No pair on any rank of the shard group: one plain CE step on the reference report."""
    out = env.forward(env.chosen)
    (env.args.dpo_unpaired_ce_weight * out['loss']).backward()
    env.stats.misc[1] += 1
    return float(out['loss'].detach())
