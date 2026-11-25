"""Encoder pre-training entry point: CT phase 1 (DINO + iBOT + KoLeo + cross-crop), CT phase 2 (``--stage gram``) and
X-ray pre-training (``--modality cxr``) (Methods, "Image encoder pre-training" and "Chest X-ray encoder").

Launch with torchrun, 4 GPUs per node, e.g.

    torchrun --nnodes 32 --nproc-per-node 4 ... -m fora_pretrain.train --output-dir OUT \
        --h5-folder H5 --splits-csv SPLITS.csv $(cat configs/ct_phase1.args)

A run resumes from OUT/checkpoints/latest.txt. Schedules (cosine lr / weight decay / EMA momentum) always span ``--steps``;
``--stop-after-step`` ends the run earlier without changing them. The peak learning rate is ``--lr`` x sqrt(B / 1024)
for a global batch B.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from fora.encoder.backbone import PATCH_SIZE, build_backbone

from . import distributed as D
from .objective import METRIC_KEYS, DINO3DLearner, LossWeights, normalize_prototypes
from .optim import build_optimizer, cancel_last_layer_gradients, schedules
from .views import DEFAULT_GLOBAL_OVERLAP, CropSpec, InfiniteReportSampler, collate

RECIPE = "fora_dino3d_vitl"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5-folder", required=True)
    p.add_argument("--splits-csv", required=True, help="CSV with columns h5_id, split; split == train shards are used")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--assignment-cache", default="", help="JSON cache of per-shard reconstruction counts (CT)")
    p.add_argument("--staging-dir", default="", help="node-local directory the assigned shards are copied to first")
    p.add_argument("--modality", choices=("ct", "cxr"), default="ct")
    p.add_argument("--stage", choices=("pretrain", "gram"), default="pretrain")
    p.add_argument("--arch", choices=("large",), default="large")
    p.add_argument("--steps", type=int, required=True, help="schedule horizon")
    p.add_argument("--stop-after-step", type=int, default=None, help="save and stop at this step (schedules unchanged)")
    p.add_argument("--batch-size", type=int, default=8, help="samples per GPU")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--global-shape", type=int, nargs=3, default=[64, 192, 192], help="CT global crop (voxels)")
    p.add_argument("--local-shape", type=int, nargs=3, default=[32, 96, 96], help="CT local crop (voxels)")
    p.add_argument("--local-crops", type=int, default=8)
    p.add_argument("--global-overlap", type=float, default=DEFAULT_GLOBAL_OVERLAP)
    p.add_argument("--mip-probability", type=float, default=0.25)
    p.add_argument("--mip-slab-mm", type=float, default=10.0)
    p.add_argument("--cxr-global", type=int, default=1024, help="X-ray global crop side (px)")
    p.add_argument("--cxr-local", type=int, default=512, help="X-ray local crop side (px)")
    p.add_argument("--cxr-spacing-mm", type=float, default=0.0, help="<= 0: native pixel spacing")
    p.add_argument("--cxr-student-windows", default="", help="'lo:hi,...' intensity windows for X-ray student views")
    p.add_argument("--dino-prototypes", type=int, default=16384)
    p.add_argument("--ibot-prototypes", type=int, default=16384)
    p.add_argument("--dino-head-hidden-dim", type=int, default=2048)
    p.add_argument("--ibot-head-hidden-dim", type=int, default=2048)
    p.add_argument("--dino-bottleneck-dim", type=int, default=256)
    p.add_argument("--ibot-bottleneck-dim", type=int, default=256)
    p.add_argument("--position-bins", type=int, nargs=3, default=[2, 2, 2], help="spatial bins of the patch Sinkhorn")
    p.add_argument("--cross-view-weight", type=float, default=1.0)
    p.add_argument("--ibot-weight", type=float, default=1.0)
    p.add_argument("--koleo-weight", type=float, default=0.1)
    p.add_argument("--gram-weight", type=float, default=0.0, help="used with --stage gram")
    p.add_argument("--gram-max-tokens", type=int, default=1024)
    p.add_argument("--ibot-loss-chunk-size", type=int, default=2048)
    p.add_argument("--mask-min", type=float, default=0.1)
    p.add_argument("--mask-max", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3, help="peak lr at global batch 1024 (sqrt scaling)")
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=2500)
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--weight-decay-end", type=float, default=0.4)
    p.add_argument("--adamw-beta1", type=float, default=0.9)
    p.add_argument("--adamw-beta2", type=float, default=0.999)
    p.add_argument("--layerwise-decay", type=float, default=0.9)
    p.add_argument("--patch-embed-lr-mult", type=float, default=0.2)
    p.add_argument("--freeze-last-layer-steps", type=int, default=1250)
    p.add_argument("--teacher-momentum", type=float, default=0.992)
    p.add_argument("--teacher-temperature", type=float, default=0.07)
    p.add_argument("--teacher-warmup-temperature", type=float, default=0.04)
    p.add_argument("--teacher-warmup-steps", type=int, default=10000)
    p.add_argument("--clip-grad", type=float, default=3.0, help="per module (backbone, DINO head, iBOT head)")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--keep-checkpoints", type=int, default=6)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--resume", default="latest", help="latest, none, or a checkpoint directory (e.g. phase 1 for gram)")
    p.add_argument("--resume-allow-partial", action="store_true",
                   help="non-strict resume (tensors absent from the checkpoint, e.g. optimizer state of a still-frozen "
                        "prototype layer, keep their fresh values)")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)
    if not 0 < args.freeze_last_layer_steps < args.warmup_steps < args.steps:
        raise ValueError("Require 0 < freeze_last_layer_steps < warmup_steps < steps")
    if not 0 < args.teacher_warmup_steps <= args.steps:
        raise ValueError("Require 0 < teacher_warmup_steps <= steps")
    if args.stop_after_step is not None and not 0 < args.stop_after_step <= args.steps:
        raise ValueError("stop-after-step must be positive and no greater than steps")
    args.recipe = RECIPE
    return args


def crop_spec_for(args) -> CropSpec:
    if args.modality == "cxr":
        return CropSpec((2, args.cxr_global, args.cxr_global), (2, args.cxr_local, args.cxr_local), int(args.local_crops))
    return CropSpec(tuple(args.global_shape), tuple(args.local_shape), int(args.local_crops))


def build_dataset(args, rank: int, world: int):
    files = D.read_train_files(args.splits_csv, args.h5_folder)
    spec = crop_spec_for(args)
    if args.modality == "cxr":
        # Fewer X-ray shards than ranks: shards are balanced over NODES, staged once per node, and the node's films are
        # split over its ranks.
        from .data_cxr import CXRDinoDataset, parse_windows
        lw = int(os.environ.get("LOCAL_WORLD_SIZE", "4"))
        node, lrank = rank // lw, rank % lw
        node_files = D.balanced_file_assignment(files, world // lw)[node]
        if lrank == 0:
            D.stage_files(node_files, args.staging_dir, node * lw)
        dist.barrier()
        assigned = ([str(Path(args.staging_dir) / f"rank_{node * lw}" / Path(f).name) for f in node_files]
                    if args.staging_dir else node_files)
        dataset = CXRDinoDataset(assigned, spec, global_overlap=args.global_overlap, spacing_mm=args.cxr_spacing_mm,
                                 student_windows=parse_windows(args.cxr_student_windows), seed=args.seed,
                                 verbose=rank == 0)
        dataset.index = dataset.index[lrank::lw]
        return dataset, spec
    from .data_ct import CTDinoDataset, count_reconstructions
    assigned = D.reconstruction_balanced_file_assignment(files, world, rank, args.assignment_cache, count_reconstructions)[rank]
    assigned = D.stage_files(assigned, args.staging_dir, rank)
    dataset = CTDinoDataset(assigned, spec, global_overlap=args.global_overlap, mip_probability=args.mip_probability,
                            mip_slab_mm=args.mip_slab_mm, seed=args.seed, patch=PATCH_SIZE, verbose=rank == 0)
    return dataset, spec


def build_learner(args, grid, device) -> DINO3DLearner:
    gram = args.stage == "gram"
    with torch.device("meta"):
        learner = DINO3DLearner(
            build_backbone(args.arch), dino_prototypes=args.dino_prototypes, ibot_prototypes=args.ibot_prototypes,
            dino_head_hidden_dim=args.dino_head_hidden_dim, ibot_head_hidden_dim=args.ibot_head_hidden_dim,
            dino_bottleneck_dim=args.dino_bottleneck_dim, ibot_bottleneck_dim=args.ibot_bottleneck_dim,
            position_grid=grid, position_bins=args.position_bins,
            loss_weights=LossWeights(dino=1.0, ibot=args.ibot_weight, koleo=args.koleo_weight,
                                     gram=args.gram_weight if gram else 0.0, cross=args.cross_view_weight),
            gram_max_tokens=args.gram_max_tokens, ibot_loss_chunk_size=args.ibot_loss_chunk_size,
            with_gram_anchor=gram, initialize_weights=False)
    D.fsdp_wrap(learner, device, args.compile)
    normalize_prototypes(learner)
    return learner


def main(argv=None) -> int:
    args = parse_args(argv)
    rank, world, _, device = D.setup_distributed()
    if world < 2:
        raise RuntimeError("FSDP2 training requires WORLD_SIZE >= 2")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed + rank)

    output = Path(args.output_dir)
    if rank == 0:
        D.checkpoint_root(output).mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    dist.barrier()

    dataset, spec = build_dataset(args, rank, world)
    totals = torch.tensor([len(dataset), dataset.n_reports], device=device, dtype=torch.long)
    dist.all_reduce(totals)
    total_items, total_reports = int(totals[0]), int(totals[1])
    sampler = InfiniteReportSampler(len(dataset), seed=args.seed + rank * 7919)

    grid = [g // p for g, p in zip(spec.global_shape, PATCH_SIZE)]
    learner = build_learner(args, grid, device)
    optimizer = build_optimizer(learner, args)
    start = D.load_checkpoint(D.resolve_checkpoint(output, args.resume), learner, optimizer, sampler, args, rank)
    consumed = sampler.offset

    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                        persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers > 0 else None,
                        pin_memory=False, drop_last=True,
                        collate_fn=functools.partial(collate, patch_size=PATCH_SIZE, mask_ratio=(args.mask_min, args.mask_max)),
                        generator=torch.Generator().manual_seed(args.seed + rank * 104729))
    iterator = iter(loader)

    global_batch = world * args.batch_size
    peak_lr = args.lr * math.sqrt(global_batch / 1024.0)
    balance = total_items / total_reports        # importance weight = balance / (items of the sample's report)
    if rank == 0:
        print(json.dumps(dict(event="start", modality=args.modality, stage=args.stage, world=world,
                              global_batch=global_batch, reports=total_reports, items=total_items,
                              global_crop=list(spec.global_shape), tokens=math.prod(grid), peak_lr=peak_lr,
                              backbone_params=sum(p.numel() for p in learner.student.backbone.parameters()),
                              start_step=start, steps=args.steps, stop_after_step=args.stop_after_step)), flush=True)

    stopper = D.StopController()
    learner.train()
    last_time = time.time()
    stop_step = args.stop_after_step if args.stop_after_step is not None else args.steps
    if stop_step <= start:
        raise ValueError(f"Requested stop {stop_step} must exceed the resumed step {start}")
    for step in range(start, stop_step):
        if stopper.should_stop(device):
            D.save_checkpoint(output, learner, optimizer, args, step, consumed, rank)
            dist.destroy_process_group()
            return 75
        s = schedules(args, step, peak_lr)
        for group in optimizer.param_groups:
            group["lr"], group["weight_decay"] = s["lr"], s["weight_decay"]
        optimizer.zero_grad(set_to_none=True)
        batch = next(iterator)
        consumed += args.batch_size
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch["loss_weights"] = batch["sample_weights"] * balance
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = learner(batch, teacher_temperature=s["teacher_temperature"], step=step)
        finite = torch.isfinite(loss.detach()).to(torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError(f"Non-finite loss at step {step}: { {k: float(v) for k, v in metrics.items()} }")
        loss.backward()
        frozen = cancel_last_layer_gradients(learner.student) if step < args.freeze_last_layer_steps else 0
        grad_norms = [torch.nn.utils.clip_grad_norm_(m.parameters(), args.clip_grad)
                      for m in (learner.student.backbone, learner.student.dino_head, learner.student.ibot_head)]
        optimizer.step()
        learner.update_teacher(s["teacher_momentum"])

        if (step + 1) % args.log_every == 0 or step == start:
            values = torch.stack([metrics[k].float() for k in METRIC_KEYS] + [loss.detach().float()])
            dist.all_reduce(values)
            values /= world
            now = time.time()
            if rank == 0:
                record = {"step": step + 1, "stage": args.stage, "epoch": (step + 1) * global_batch / total_items}
                record.update({k: float(v) for k, v in zip(METRIC_KEYS, values)})
                record.update(objective_total=float(values[-1]), last_layer_frozen=bool(frozen),
                              grad_norm_max=max(float(g.full_tensor() if hasattr(g, "full_tensor") else g) for g in grad_norms),
                              steps_per_second=args.log_every / max(1e-9, now - last_time),
                              peak_gpu_memory_gib=torch.cuda.max_memory_allocated(device) / 2 ** 30, **s)
                with open(output / "metrics.jsonl", "a") as f:
                    f.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            last_time = now
        if (step + 1) % args.save_every == 0:
            D.save_checkpoint(output, learner, optimizer, args, step + 1, consumed, rank)

    if stop_step % args.save_every:
        D.save_checkpoint(output, learner, optimizer, args, stop_step, consumed, rank)
    if rank == 0:
        (output / "RUN_COMPLETE.json").write_text(json.dumps({"start_step": start, "stop_step": stop_step}))
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
