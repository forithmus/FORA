"""
SFT Training Entry Point for FORA Report Generation
====================================================

Builds visual encoder (RADRATE + poolers + perceiver + projection),
LLM with LoRA (Qwen3.5-9B), dataset, and starts SFT training.

Usage:
    # Single GPU
    python train.py --config config.yaml

    # Multi-GPU with accelerate
    accelerate launch --mixed_precision bf16 train.py --config config.yaml

    # Multi-node via SLURM
    sbatch submit_train.sh
"""

import os
import sys
import argparse
import logging

# Path setup — report_generation/ must come BEFORE scripts/ so that
# "from data import SFTReportDataset" finds our data.py, not scripts/data.py
script_dir = os.path.dirname(os.path.abspath(__file__))
fora_root = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
for subdir in ["rad_rate", "vision_encoder", "scripts"]:
    path = os.path.join(fora_root, subdir)
    if path not in sys.path:
        sys.path.append(path)

import yaml
import torch

from model import build_visual_encoder, build_llm_with_lora
from data import SFTReportDataset, sft_collate_fn, CTReportDataset
from sft_trainer import SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SFT training for FORA report generation")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML config (relative to script dir)")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(script_dir, args.config)
    print("=" * 60)
    print(f"Loading config: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Disable mem_efficient attention (can break with dynamic sequence lengths)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} (LOCAL_RANK={local_rank})")

    # ------------------------------------------------------------------
    # 1. Build dataset
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Stage 1/4: Building dataset...")
    data_cfg = config["data"]

    # Build kwargs — only pass non-None values so CTReportDataset uses its own defaults
    ct_kwargs = dict(
        data_folder=data_cfg.get("ctrate_folder"),
        jsonl_file=data_cfg.get("ctrate_jsonl"),
        meta_file=data_cfg.get("ctrate_meta"),
        include_ctrate=data_cfg.get("include_ctrate", True),
        include_merlin=data_cfg.get("include_merlin", False),
        include_mimic=data_cfg.get("include_mimic", False),
        include_chexpert=data_cfg.get("include_chexpert", False),
        include_padchest=data_cfg.get("include_padchest", False),
    )
    # Optional overrides for non-CT-RATE sources
    for key, ct_key in [
        ("merlin_data_folder", "merlin_data_folder"),
        ("merlin_jsonl", "merlin_jsonl_file"),
        ("merlin_csv", "merlin_csv_file"),
    ]:
        val = data_cfg.get(key)
        if val:
            ct_kwargs[ct_key] = val

    ct_dataset = CTReportDataset(**ct_kwargs)

    single_source = data_cfg.get("single_source")
    reports_csv = data_cfg.get("reports_csv")
    sft_dataset = SFTReportDataset(
        ct_dataset,
        reports_csv=reports_csv,
        single_source=single_source,
    )

    print(f"SFT dataset: {len(sft_dataset)} samples")
    if single_source:
        print(f"  Single source mode: {single_source}")
    else:
        print(f"  Multi-source mode (balanced round-robin)")

    # ------------------------------------------------------------------
    # 2. Build visual encoder
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Stage 2/4: Building visual encoder...")
    visual_encoder = build_visual_encoder(config, device)

    freeze_visual = config["model"].get("freeze_visual", False)
    if not freeze_visual:
        visual_encoder.unfreeze_trainable()
        trainable_v = sum(p.numel() for p in visual_encoder.parameters() if p.requires_grad)
        total_v = sum(p.numel() for p in visual_encoder.parameters())
        print(f"Visual encoder: {trainable_v:,} / {total_v:,} trainable params")
    else:
        visual_encoder.freeze()
        print("Visual encoder: FROZEN")

    # ------------------------------------------------------------------
    # 3. Build LLM with LoRA
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Stage 3/4: Building LLM with LoRA...")
    llm, tokenizer = build_llm_with_lora(config, device)
    print(f"Tokenizer: vocab={len(tokenizer)}, "
          f"pad='{tokenizer.pad_token}', eos='{tokenizer.eos_token}'")

    # ------------------------------------------------------------------
    # 4. Create trainer and start training
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Stage 4/4: Creating SFT trainer...")
    trainer = SFTTrainer(
        visual_encoder=visual_encoder,
        llm=llm,
        tokenizer=tokenizer,
        train_dataset=sft_dataset,
        collate_fn=sft_collate_fn,
        config=config,
    )

    print("=" * 60)
    print("Starting SFT training...")
    trainer.train()
    print("Done!")


if __name__ == "__main__":
    main()
