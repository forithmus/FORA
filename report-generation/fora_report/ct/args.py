"""Command line of the CT writer trainer (SFT and DPO share one entry point).

``--config configs/ct_*.json`` supplies defaults (JSON object: argument name -> value; keys starting with ``_`` are
comments); explicit flags override it.
Defaults below are the values shared by every paper run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CT report writer training (SFT / offline DPO)")
    p.add_argument("--config", default="", help="JSON file with argument defaults")
    # data
    p.add_argument("--index", required=False, default="", help="report-generation index (sqlite)")
    p.add_argument("--h5_root", default="", help="directory of the study HDF5 shards (Findings text)")
    p.add_argument("--dino_mil_config", default="", help="classifier config JSON (cases, assignment, block-24 state)")
    p.add_argument("--dino_mil_checkpoint", default="", help="classifier checkpoint holding the trained block 24")
    p.add_argument("--dino_cache_root", default="", help="root of the node-local block-23 caches")
    p.add_argument("--dino_token_cap", type=int, default=524288,
                   help="uniform subsample of each study's tokens after block 24 (training)")
    p.add_argument("--dino_crops_per_chunk", type=int, default=32)
    p.add_argument("--dino_gpu_bag_gib", type=float, default=8.0)
    p.add_argument("--mil_calibration", default="", help="plan calibration JSON (Platt + tier cutoffs)")
    p.add_argument("--precomputed_mil_predictions", default="", help="node-local dense MIL logit cache")
    p.add_argument("--finding_store", default="", help="node-local finding store cache (evidence rows)")
    p.add_argument("--finding_descriptions", default="", help="per-row evidence description array (.npy)")
    p.add_argument("--focus_tokens", default="", help="node-local focus-token store directory")
    p.add_argument("--study_measurements", default="", help="per-study automatic measurement lines (JSON)")
    p.add_argument("--plan_possible_cap", type=int, default=20, help="possible tier: keep top K by calibrated p; 0 = all")
    p.add_argument("--plan_cap_rank", default="prob", choices=("prob", "fold"))
    p.add_argument("--limit_train_per_rank", type=int, default=0,
                   help="keep the first N studies (report-uid order) of every rank; 0 = all")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--worker_torch_threads", type=int, default=4)
    # model
    p.add_argument("--llm_path", default="", help="local Gemma-4-31B-it snapshot")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--num_visual_tokens", type=int, default=512)
    p.add_argument("--resampler_depth", type=int, default=2)
    p.add_argument("--resampler_heads", type=int, default=8)
    p.add_argument("--dino_resampler_dim", type=int, default=1024)
    p.add_argument("--dino_study_types", type=int, default=32)
    p.add_argument("--max_report_tokens", type=int, default=2048)
    p.add_argument("--fimg_tokens", type=int, default=6, help="pooled tokens per zoomed evidence image set")
    p.add_argument("--fimg_soft_tokens", type=int, default=140, help="Gemma-4 image processor max_soft_tokens")
    # optimisation
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--target_global_batch", type=int, default=128)
    p.add_argument("--connector_lr", type=float, default=1e-4)
    p.add_argument("--llm_lr", type=float, default=2e-5)
    p.add_argument("--vision_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--initial_horizon_epochs", type=int, default=0,
                   help="cosine horizon in epochs before --initial_horizon_until_step (0 = none)")
    p.add_argument("--initial_horizon_until_step", type=int, default=0,
                   help="optimizer step from which the final horizon applies")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warm_start", default="", help="initialise every trainable tensor from this checkpoint")
    p.add_argument("--resume", default="", help="exact resume (default: <results_dir>/latest.pt if present)")
    # DPO
    p.add_argument("--dpo_pairs", default="", help="node-local pair store; enables DPO")
    p.add_argument("--dpo_reference", default="", help="frozen reference checkpoint (stage-2 SFT)")
    p.add_argument("--dpo_beta", type=float, default=0.1)
    p.add_argument("--dpo_weight", type=float, default=1.0)
    p.add_argument("--dpo_chosen_ce_weight", type=float, default=1.0)
    p.add_argument("--dpo_unpaired_ce_weight", type=float, default=1.0)
    p.add_argument("--dpo_pairs_per_visit", type=int, default=2)
    p.add_argument("--dpo_mix", type=float, default=1.0)
    p.add_argument("--dpo_pair_types", default="", help="comma list; empty = all")
    p.add_argument("--dpo_recompute_tolerance", type=float, default=0.5)
    p.add_argument("--dpo_recompute_abort", type=float, default=20.0)
    p.add_argument("--dpo_slot_scope", default="shard", choices=("shard", "world"))
    # run control
    p.add_argument("--results_dir", default="")
    p.add_argument("--save_every_steps", type=int, default=100)
    p.add_argument("--save_every_minutes", type=float, default=30.0)
    p.add_argument("--keep_checkpoint_every", type=int, default=0, help="copy latest.pt to step_N.pt every N steps")
    p.add_argument("--log_every_steps", type=int, default=5)
    p.add_argument("--stop_after_steps", type=int, default=0)
    p.add_argument("--stop_request_file", default="", help="save and exit at the next step once this file exists")
    p.add_argument("--pg_timeout_s", type=int, default=36000)
    return p


REQUIRED = ("index", "h5_root", "dino_mil_config", "dino_mil_checkpoint", "dino_cache_root", "mil_calibration",
            "precomputed_mil_predictions", "finding_store", "llm_path", "results_dir")


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    first, _ = parser.parse_known_args(argv)
    if first.config:
        values = {k: v for k, v in json.loads(Path(first.config).read_text()).items() if not k.startswith("_")}
        unknown = sorted(set(values) - {a.dest for a in parser._actions})
        if unknown:
            raise SystemExit(f"unknown config keys: {unknown}")
        parser.set_defaults(**values)
    args = parser.parse_args(argv)
    missing = [name for name in REQUIRED if not getattr(args, name)]
    if missing:
        parser.error("missing required arguments: " + ", ".join("--" + m for m in missing))
    if args.dpo_pairs and not (args.dpo_reference or args.warm_start or args.resume):
        parser.error("DPO needs --dpo_reference (the SFT checkpoint) or --warm_start")
    if args.dpo_pairs and (args.dpo_beta <= 0 or args.dpo_pairs_per_visit < 1 or not 0.0 <= args.dpo_mix <= 1.0):
        parser.error("invalid DPO hyper-parameters")
    if args.resume and args.warm_start:
        parser.error("--resume and --warm_start are mutually exclusive")
    return args
