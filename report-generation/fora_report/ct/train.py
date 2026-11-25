"""CT report writer training: supervised fine-tuning and offline DPO (paper: "Grounded report generation").

One optimizer step = one study per GPU (global batch 128 on 32 nodes x 4 GPUs). Per study:

1. visual stream: node-local block-23 DINO cache -> frozen classifier block 24 -> 1048-d tokens, uniformly
   subsampled to ``--dino_token_cap`` tokens (deterministic in seed, study and epoch);
2. diagnostic plan from the frozen classifier logits and the plan calibration (confident tier, possible tier capped
   by calibrated probability, probabilities relative to the base rate), with each planned finding's evidence
   description, focus tokens (pooled + up to 16 foci) and up to six zoomed evidence images;
3. writer = Gemma-4-31B-it + LoRA, 512-latent query resampler, study-type tokens, Gemma-4 vision tower pooled per
   finding; HSDP over the node's GPUs (replicate across nodes);
4. SFT: token-mean cross-entropy on ``"Findings:\\n" + Findings``. DPO (``--dpo_pairs``): sigmoid DPO on summed
   log-probabilities against the frozen stage-2 SFT reference, K pairs per study, chosen-response CE, CE for studies
   without a pair.

Launch with torchrun (``reportgen/ct/run_node.sh``); configs in ``configs/ct_sft1.json``, ``configs/ct_sft2.json``,
``configs/ct_dpo.json``.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from fora_report.ct.args import parse_args
from fora_report.ct.checkpoint import (
    init_runtime,
    restore_rng_state,
    save_checkpoint,
    synchronize_checkpoint_decision,
    worker_init,
)
from fora_report.ct.dpo_loss import (
    PREFERENCE_OBJECTIVE,
    DPOStatistics,
    PreferencePairStore,
    disable_preference_dropout,
    load_reference_state,
    select_pairs,
    sha256_text,
)
from fora_report.ct.mil_logits import DenseMILLogitStore
from fora_report.ct.prompts import prompt_ids, target_ids
from fora_report.ct.report_data import DinoReportDataset, collate_one, read_index_metadata, shard_counts
from fora_report.ct.schedule import TokenBalancedEpochSampler, cosine_scheduler

FIMG_PLACEHOLDER_TOKEN = "<|audio|>"
# Volatile paths and run-control flags that must not change the resume identity.
UNSTABLE_ARGUMENTS = {
    "config", "dpo_reference", "resume", "warm_start", "llm_path", "results_dir", "dino_cache_root",
    "stop_after_steps", "stop_request_file", "save_every_minutes", "keep_checkpoint_every", "log_every_steps",
    "pg_timeout_s", "dpo_recompute_abort", "finding_store", "precomputed_mil_predictions", "focus_tokens",
}


def _print(runtime, *parts):
    if runtime.main:
        print(*parts, flush=True)


def build_dataset(args, runtime, labels):
    """This rank's studies: the train reports whose DINO cache belongs to its data rank on this node."""
    from fora.reportgen.writer.dino_stream import (check_unique_data_ranks, data_rank_for, detect_data_node,
                                              local_report_cases, read_index_splits)
    dino_config = json.loads(Path(args.dino_mil_config).read_text())
    dino_assign = json.loads(Path(dino_config["assign"]).read_text())
    gpus_per_node = int(dino_assign.get("ranks_per_node", 4))
    if runtime.local_world_size != gpus_per_node:
        raise ValueError(f"DINO cache placement needs {gpus_per_node} GPUs per node, got {runtime.local_world_size}")
    node = detect_data_node(dino_config["fingerprint"], Path(args.dino_cache_root))
    data_rank = data_rank_for(node, runtime.local_rank, gpus_per_node)
    all_ranks = [None] * runtime.world_size
    dist.all_gather_object(all_ranks, data_rank)
    check_unique_data_ranks(all_ranks)
    cases = local_report_cases(dino_config, data_rank, read_index_splits(args.index), "train", assign=dino_assign)
    dataset = DinoReportDataset(cases_by_uid=cases, mil_config=dino_config, cache_root=args.dino_cache_root,
                                index_path=args.index, h5_root=args.h5_root, split="train", n_classes=len(labels))
    if len(dataset.index) != len(cases):
        raise RuntimeError(f"rank {runtime.rank}: {len(cases) - len(dataset.index)} node-local train reports "
                           "are missing from the index")
    if runtime.local_rank == 0:
        print(f"[dino-data] rank={runtime.rank} data_node={node} data_rank={data_rank} "
              f"train_reports={len(dataset.index)} tokens_total={sum(dataset.token_counts())}", flush=True)
    return dataset, dino_config


def build_model(args, runtime, labels, dino_config, plan_builder, finding_store, tokenizer, llm, llm_dim):
    from fora.reportgen.writer.conditioning import semantic_label_embeddings
    from fora.reportgen.writer.dino_stream import DinoCTReportGenerator, token_dim
    semantics = semantic_label_embeddings(llm, tokenizer, labels, runtime.device)
    placeholder_id = int(tokenizer.convert_tokens_to_ids(plan_builder_placeholder()))
    probe = tokenizer(plan_builder_placeholder(), add_special_tokens=False).input_ids
    if probe != [placeholder_id] or placeholder_id in (tokenizer.unk_token_id, None):
        raise RuntimeError(f"{plan_builder_placeholder()} is not a single reserved token: {probe}")
    model = DinoCTReportGenerator(
        llm,
        resampler_dim=args.dino_resampler_dim,
        n_study_types=args.dino_study_types,
        visual_dim=token_dim(dino_config),
        llm_dim=llm_dim,
        num_visual_tokens=args.num_visual_tokens,
        resampler_depth=args.resampler_depth,
        resampler_heads=args.resampler_heads,
        semantic_embeddings=semantics,
        eligible_class_indices=torch.arange(len(labels), dtype=torch.long, device=runtime.device),
        finding_feature_dim=finding_store.feature_dim,
        finding_placeholder_id=placeholder_id,
        calibration_slope=plan_builder.slope.to(runtime.device),
        calibration_intercept=plan_builder.intercept.to(runtime.device),
        diagnostic_consistency_classes=len(labels),
        fimg_tokens=args.fimg_tokens,
        fimg_placeholder_id=int(tokenizer.convert_tokens_to_ids(FIMG_PLACEHOLDER_TOKEN)),
    ).to(runtime.device, dtype=torch.bfloat16)
    fimg_processor = None
    if getattr(model, "fimg_pool", None) is not None:
        from fora.reportgen.evidence.zoom_images import make_fimg_processor
        fimg_processor = make_fimg_processor(args.fimg_soft_tokens)
        vision = sum(p.numel() for n, p in model.named_parameters() if ".vision_tower." in n or ".embed_vision." in n)
        _print(runtime, f"[fimg] {model.fimg_queries} tokens/finding, placeholder id {model.fimg_placeholder_id}, "
                        f"vision params {vision / 1e6:.0f}M trainable")
    return model, fimg_processor


def plan_builder_placeholder() -> str:
    from fora.reportgen.plan import diagnostic_plan
    return diagnostic_plan.FINDING_PLACEHOLDER


def optimizer_groups(model, args):
    """Connector (everything outside the LLM), LoRA, and Gemma-4 vision tower parameter groups."""
    connector, lora, vision = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".vision_tower." in name or ".embed_vision." in name:
            vision.append(parameter)
        elif not name.startswith("llm."):
            connector.append(parameter)
        elif "lora_" in name:
            lora.append(parameter)
        else:
            raise RuntimeError(f"unexpected trainable base LLM parameter {name}")
    groups = [{"params": connector, "lr": args.connector_lr}, {"params": lora, "lr": args.llm_lr}]
    if vision:
        groups.append({"params": vision, "lr": args.vision_lr})
    return groups


def main(argv=None):
    args = parse_args(argv)
    runtime = init_runtime(args.pg_timeout_s)
    from fora_report.writer import fsdp as writer_fsdp
    from fora_report.writer import fsdp_dpo
    from fora.reportgen.writer.dino_stream import VISUAL_STREAM, DinoVisualStream, is_dino_checkpoint
    from fora.reportgen.writer.llm import build_gemma_lora, language_layers, llm_family_id
    from fora.reportgen.plan.diagnostic_plan import DiagnosticPlanBuilder, set_finding_placeholder
    from fora.reportgen.evidence.finding_store import FindingLocStore
    from fora.reportgen.evidence.zoom_images import fimg_batch

    dpo_enabled = bool(args.dpo_pairs)
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)
    checkpoint_vote = torch.zeros(2, dtype=torch.uint8, device=runtime.device)
    seed = args.seed + runtime.rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    args.num_workers = max(1, int(args.num_workers))
    if runtime.world_size > 1 and args.target_global_batch % runtime.world_size:
        args.target_global_batch = runtime.world_size

    results = Path(args.results_dir)
    if runtime.main:
        results.mkdir(parents=True, exist_ok=True)
    dist.barrier(device_ids=[runtime.local_rank])

    labels = list(read_index_metadata(args.index)["global"]["labels"])
    counts = shard_counts(args.index, split="train", protocol_class=0)
    dataset, dino_config = build_dataset(args, runtime, labels)
    calibration_package = json.loads(Path(args.mil_calibration).read_text())
    if args.limit_train_per_rank:
        dataset.index = dataset.index[: args.limit_train_per_rank]
    local_size = torch.tensor(len(dataset), device=runtime.device)
    dist.all_reduce(local_size, op=dist.ReduceOp.MAX)
    microsteps_per_epoch = int(local_size.item())
    accumulation = max(1, math.ceil(args.target_global_batch / runtime.world_size))
    updates_per_epoch = math.ceil(microsteps_per_epoch / accumulation)
    _print(runtime, f"[data] reports=~{sum(counts.values())} world={runtime.world_size} "
                    f"microsteps/epoch={microsteps_per_epoch} accumulation={accumulation} updates/epoch={updates_per_epoch}")

    dino_stream = DinoVisualStream(args.dino_mil_config, args.dino_mil_checkpoint, runtime.device,
                                   crops_per_chunk=args.dino_crops_per_chunk, gpu_bag_gib=args.dino_gpu_bag_gib)
    visual_source_id = f"dino_b24:{Path(args.dino_mil_config).resolve()}:{Path(args.dino_mil_checkpoint).resolve()}"
    _print(runtime, f"[visual] token_dim={dino_stream.dim} cap_train={args.dino_token_cap} "
                    f"crops_per_chunk={args.dino_crops_per_chunk} resampler_dim={args.dino_resampler_dim}")

    # Identical initialisation of every trainable module on every rank; rank-specific RNG is restored afterwards.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    llm, tokenizer, llm_dim = build_gemma_lora(args.llm_path, runtime.device, lora_r=args.lora_r,
                                               lora_alpha=args.lora_alpha, fimg_tokens=args.fimg_tokens)
    family_id = llm_family_id(args.llm_path)
    set_finding_placeholder("gemma4")
    train_mode_report = writer_fsdp.train_mode_for_checkpointing(llm, False)
    if not bool(llm.get_base_model().is_gradient_checkpointing):
        raise RuntimeError("Gemma gradient checkpointing is not active after PEFT wrapping")
    _print(runtime, f"[llm] id={family_id} hidden={llm_dim} gradient_checkpointing=1 {train_mode_report}")

    plan_builder = DiagnosticPlanBuilder(args.mil_calibration, labels, tokenizer, "all_classes", dataset_kind="ct",
                                         plan_possible_cap=args.plan_possible_cap, plan_cap_rank=args.plan_cap_rank,
                                         plan_prob=True, fimg_tokens=args.fimg_tokens,
                                         evidence_descriptions=bool(args.finding_descriptions))
    mil_store = DenseMILLogitStore(args.precomputed_mil_predictions, labels, calibration=calibration_package)
    mil_store.require_coverage([str(s["report_uid"]) for s in dataset.index])
    token_counts = dataset.token_counts()
    sampling_summary = TokenBalancedEpochSampler(token_counts, microsteps_per_epoch, 0, 0, args.seed,
                                                 local_rank=runtime.local_rank,
                                                 local_world_size=runtime.local_world_size).summary()
    _print(runtime, "[writer-sampling] " + json.dumps(sampling_summary, sort_keys=True))
    finding_store = FindingLocStore(args.finding_store, labels, descriptions=args.finding_descriptions or None,
                                    ftok=args.focus_tokens or None, study_meas=args.study_measurements or None)
    local_uids = [str(s["report_uid"]) for s in dataset.index]
    covered = torch.tensor([sum(finding_store.contains(u) for u in local_uids), len(local_uids)],
                           dtype=torch.long, device=runtime.device)
    dist.all_reduce(covered)
    _print(runtime, f"[finding-store] reports={finding_store.uids.size} feature_dim={finding_store.feature_dim} "
                    f"train_coverage={int(covered[0])}/{int(covered[1])}")

    model, fimg_processor = build_model(args, runtime, labels, dino_config, plan_builder, finding_store,
                                        tokenizer, llm, llm_dim)
    mesh = writer_fsdp.build_mesh(runtime.device.type, runtime.world_size, runtime.local_world_size)
    writer_fsdp.shard_report_model(model, language_layers(llm), mesh)
    torch.cuda.empty_cache()
    _print(runtime, f"[fsdp] HSDP mesh={tuple(mesh.shape)} units={len(language_layers(llm))}+root "
                    f"{writer_fsdp.describe_memory(runtime.device)}")

    optimizer = torch.optim.AdamW(optimizer_groups(model, args), weight_decay=args.weight_decay, foreach=True)
    total_steps = args.epochs * updates_per_epoch
    horizon = dict(initial_horizon_steps=args.initial_horizon_epochs * updates_per_epoch,
                   initial_horizon_until_step=args.initial_horizon_until_step)
    scheduler = cosine_scheduler(optimizer, total_steps, args.warmup_fraction, **horizon)

    stable_arguments = {k: v for k, v in vars(args).items() if k not in UNSTABLE_ARGUMENTS}
    fingerprint = hashlib.sha256(json.dumps({
        "visual": visual_source_id, "llm": family_id, "world_size": runtime.world_size,
        "arguments": stable_arguments}, sort_keys=True).encode()).hexdigest()
    run_metadata = {
        "visual_stream": "dino_b24", "visual_source": visual_source_id, "llm": family_id,
        "llm_adaptation": {"lora_rank": args.lora_r, "lora_alpha": args.lora_alpha,
                           "lora_target": "all_attention_and_mlp_projections"},
        "label_policy": "asserted_or_uncertain_positive; denied_negative; unmentioned_masked",
        "mil_calibration": str(Path(args.mil_calibration).resolve()),
        "mil_logits_source": mil_store.manifest.get("source"),
        "finding_store_source": finding_store.manifest.get("source"),
        "finding_feature_dim": finding_store.feature_dim,
        "generation_target": "report/structured/findings_only",
        "writer_sampling": {"mode": "natural", "ordering": sampling_summary},
        "world_size": runtime.world_size,
        "arguments": dict(vars(args), visual_stream=VISUAL_STREAM),
        "stable_arguments": stable_arguments,
        "fingerprint": fingerprint,
    }
    if runtime.main:
        (results / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n")

    start_epoch = start_microstep = optimizer_step = 0
    resume = Path(args.resume) if args.resume else results / "latest.pt"
    resumed = False
    if resume.is_file():
        package = torch.load(resume, map_location="cpu", weights_only=False)
        if package.get("metadata", {}).get("fingerprint") != fingerprint:
            raise RuntimeError("resume checkpoint configuration fingerprint differs")
        writer_fsdp.load_full_trainable_state(model, package["trainable_model"])
        writer_fsdp.load_full_optimizer_state(optimizer, package["optimizer"])
        scheduler.load_state_dict(package["scheduler"])
        start_epoch, start_microstep = int(package["epoch"]), int(package["microstep"])
        optimizer_step = int(package["optimizer_step"])
        restore_rng_state(package["rng_states"][runtime.rank], runtime)
        _print(runtime, f"[resume] {resume} epoch={start_epoch + 1} microstep={start_microstep}")
        resumed = True
    elif args.warm_start:
        package = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        if not is_dino_checkpoint(package):
            raise RuntimeError("warm start must be a writer checkpoint with the DINO visual stream")
        missing = fsdp_dpo.load_exact_warm_start(model, package["trainable_model"])
        if missing:
            raise RuntimeError(f"warm start must load every trainable tensor; newly initialized: {missing[:10]}")
        _print(runtime, f"[warm-start] {args.warm_start} source_optimizer_step={package.get('optimizer_step')} "
                        "optimizer=reset scheduler=reset")
        del package
    if not resumed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    with torch.no_grad():
        checksum = writer_fsdp.trainable_checksum(model, runtime.device)
        low, high = checksum.clone(), checksum.clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN)
        dist.all_reduce(high, op=dist.ReduceOp.MAX)
        if not torch.equal(low, high):
            raise RuntimeError("trainable parameter initialization differs across ranks")

    dpo_store = dpo_reference = dpo_stats = slot_group = None
    if dpo_enabled:
        dropout_changes = disable_preference_dropout(model)
        reference_path = args.dpo_reference or args.warm_start
        reference_state, reference_meta = load_reference_state(reference_path)
        dpo_reference = fsdp_dpo.FSDPReferenceWeights(model, reference_state, runtime.device)
        slot_group = fsdp_dpo.slot_group(mesh, args.dpo_slot_scope)
        del reference_state
        if not resumed and dpo_reference.max_abs_difference() != 0.0:
            raise RuntimeError("DPO must start from the reference (SFT) weights exactly; pass it as --warm_start")
        dpo_store = PreferencePairStore(args.dpo_pairs, [t for t in args.dpo_pair_types.split(",") if t] or None)
        dpo_stats = DPOStatistics(runtime.device)
        _print(runtime, f"[dpo] objective={PREFERENCE_OBJECTIVE} pairs={dpo_store.summary.get('pairs_by_type')} "
                        f"ref_step={reference_meta.get('optimizer_step')} beta={args.dpo_beta} k={args.dpo_pairs_per_visit}")
        if runtime.main:
            (results / "dpo_metadata.json").write_text(json.dumps({
                "objective": PREFERENCE_OBJECTIVE, "pairs_summary": dpo_store.summary,
                "reference": str(Path(reference_path).resolve()), "reference_meta": reference_meta,
                "dropout_disabled": dropout_changes, "resumed": resumed}, indent=2, default=str) + "\n")

    ddp = writer_fsdp.FSDPSyncAdapter(model)
    fixed_prompt = prompt_ids(tokenizer, runtime.device)
    last_save = time.monotonic()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs):
        begin = start_microstep if epoch == start_epoch else 0
        sampler = TokenBalancedEpochSampler(token_counts, microsteps_per_epoch, epoch, begin, args.seed,
                                            local_rank=runtime.local_rank, local_world_size=runtime.local_world_size)
        loader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=collate_one,
                            num_workers=args.num_workers, pin_memory=False, persistent_workers=True,
                            prefetch_factor=1, worker_init_fn=worker_init(args.worker_torch_threads),
                            multiprocessing_context="spawn")
        for microstep, batch in enumerate(loader, start=begin):
            image, _mask, _coords, study_types, positives, negatives, report, uid = batch
            bag = dino_stream.encode(image, cap=args.dino_token_cap, seed_key=f"{args.seed}|{uid}|{epoch}")
            del image
            writer_kwargs = {
                "dino_view_kind": bag.view_kind,
                "dino_view_series": bag.view_series,
                "dino_study_types": study_types.to(device=runtime.device, dtype=torch.long),
            }
            # Study-type tokens are built inside the writer from dino_study_types.
            modality_tokens = bag.tokens.new_empty((0, args.dino_resampler_dim))
            targets = target_ids(tokenizer, report, args.max_report_tokens, runtime.device)
            positives = positives.to(runtime.device)
            negatives = negatives.to(runtime.device)
            mil_logits = mil_store.get(uid, runtime.device)
            finding_rows = finding_store.rows(uid)
            descriptions = {c: finding_store.describe_row(row) for c, row in finding_rows.items()}
            if finding_store.study_meas:
                descriptions[-1] = finding_store.study_line(uid)
            token_classes = finding_store.token_classes(finding_rows)
            order = plan_builder.planned_finding_order(mil_logits, token_classes)
            finding_features = finding_store.features([finding_rows[i] for i in order], runtime.device)
            if fimg_processor is not None:
                model.set_finding_images(*fimg_batch(fimg_processor,
                                                     finding_store.images([finding_rows[i] for i in order])),
                                         len(order))
            plan = plan_builder.plan_ids(mil_logits, runtime.device, location_descriptions=descriptions,
                                         finding_token_classes=token_classes)
            consistency = plan_builder.consistency_targets(positives, runtime.device,
                                                           negative_class_indices=negatives)
            boundary = ((microstep + 1) % accumulation == 0) or (microstep + 1 == microsteps_per_epoch)
            visual_tokens = bag.tokens.detach()
            del bag
            writer_kwargs["finding_features"] = finding_features

            def writer_inputs(target_tensor):
                return dict(visual_tokens=visual_tokens, modality_tokens=modality_tokens, prompt_ids=fixed_prompt,
                            target_ids=target_tensor, positive_class_indices=positives,
                            precomputed_mil_logits=mil_logits, diagnostic_plan_ids=plan,
                            diagnostic_consistency_targets=consistency, **writer_kwargs)

            rejected = []
            if dpo_store is not None:
                rows = dpo_store.pairs(str(uid))
                if rows and random.Random(f"{args.seed}|dpo|{epoch}|{uid}").random() < args.dpo_mix:
                    chosen_sha = sha256_text(str(report))
                    for _pid, pair_type, _attr, pair_sha, rejected_text in select_pairs(
                            rows, uid=str(uid), epoch=epoch, seed=args.seed, k=args.dpo_pairs_per_visit):
                        if pair_sha != chosen_sha:
                            dpo_stats.misc[2] += 1
                            continue
                        rejected_ids = target_ids(tokenizer, rejected_text, args.max_report_tokens + 1,
                                                  runtime.device)
                        # Preference targets are never truncated.
                        if rejected_ids.numel() > args.max_report_tokens or targets.numel() >= args.max_report_tokens:
                            dpo_stats.misc[2] += 1
                            continue
                        rejected.append((pair_type, rejected_ids))
            slots = fsdp_dpo.agree_slots(len(rejected), slot_group, runtime.device) if dpo_store is not None else 0
            if slots:
                def score(kind, target_tensor):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        result = (ddp if kind == "train" else model)(**writer_inputs(target_tensor))
                    if not torch.isclose(result["loss"].detach().float(), result["report_loss"].float(),
                                         rtol=1e-4, atol=1e-5):
                        raise RuntimeError("DPO needs loss == token-mean report NLL; an auxiliary loss is active")
                    return -result["loss"].float() * target_tensor.numel(), result

                def sync(is_last):
                    return contextlib.nullcontext() if (boundary and is_last) else ddp.no_sync()

                value, _part, output, paired = fsdp_dpo.fsdp_dpo_study_backward(
                    score=score, chosen_ids=targets, rejected=rejected, slots=slots, reference=dpo_reference,
                    ddp_context=sync, beta=args.dpo_beta, dpo_weight=args.dpo_weight,
                    chosen_ce_weight=args.dpo_chosen_ce_weight, unpaired_ce_weight=args.dpo_unpaired_ce_weight,
                    accumulation=accumulation, stats=dpo_stats, recompute_tolerance=args.dpo_recompute_tolerance,
                    mismatch_abort=args.dpo_recompute_abort)
                if paired:
                    output = dict(output)
                    output["loss"] = output["loss"].detach().new_tensor(value)
            else:
                weight = args.dpo_unpaired_ce_weight if dpo_store is not None else 1.0
                if dpo_store is not None:
                    dpo_stats.misc[1] += 1
                with contextlib.nullcontext() if boundary else ddp.no_sync():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = ddp(**writer_inputs(targets))
                    (output["loss"] * weight / accumulation).backward()
            del visual_tokens
            if not boundary:
                continue

            writer_fsdp.clip_grad_norm([p for p in model.parameters() if p.requires_grad and p.grad is not None], 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            if optimizer_step % args.log_every_steps == 0:
                values = torch.stack((output["loss"].detach().float(), output["report_loss"].detach().float()))
                dist.all_reduce(values)
                values /= runtime.world_size
                if dpo_stats is not None:
                    line = dpo_stats.line(optimizer_step, dist.all_reduce)
                    _print(runtime, line)
                lrs = " ".join(f"lr{i}={g['lr']:.3e}" for i, g in enumerate(optimizer.param_groups))
                _print(runtime, f"[train] epoch={epoch + 1} micro={microstep + 1}/{microsteps_per_epoch} "
                                f"step={optimizer_step} loss={values[0]:.4f} report={values[1]:.4f} {lrs} uid={uid}")
            local_due = (optimizer_step % args.save_every_steps == 0
                         or time.monotonic() - last_save >= args.save_every_minutes * 60)
            due, stop = synchronize_checkpoint_decision(
                local_due,
                stop_requested or (bool(args.stop_request_file) and os.path.isfile(args.stop_request_file)),
                checkpoint_vote)
            limit = bool(args.stop_after_steps and optimizer_step >= args.stop_after_steps)
            if due or stop or limit:
                save_checkpoint(results / "latest.pt", model, optimizer, scheduler, epoch, microstep + 1,
                                optimizer_step, runtime, run_metadata)
                last_save = time.monotonic()
                if runtime.main and args.keep_checkpoint_every and optimizer_step % args.keep_checkpoint_every == 0:
                    shutil.copy2(results / "latest.pt", results / f"step_{optimizer_step:06d}.pt")
            if limit or stop:
                dist.barrier(device_ids=[runtime.local_rank])
                return
        start_microstep = 0
        save_checkpoint(results / "latest.pt", model, optimizer, scheduler, epoch + 1, 0, optimizer_step, runtime,
                        run_metadata)
        if runtime.main:
            shutil.copy2(results / "latest.pt", results / f"checkpoint_epoch{epoch + 1:03d}.pt")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
