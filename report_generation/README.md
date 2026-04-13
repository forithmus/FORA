# FORA Report Generation (SFT)

Supervised fine-tuning for autoregressive radiology report generation using FORA's frozen contrastive visual encoder as the backbone.

## Architecture

```
Input: CT volume or X-ray [B, 4, 1, D, H, W] + real_volume_mask [B, 4]
    |
    v
RADRATE visual_transformer (frozen, from contrastive checkpoint)
    |  per-window tokens [B, N, 1408]
    v
WindowAttentionPooler x4 (trainable, learnable queries)
    |  [B, 256, 1408] per window
    v
Concatenate -> [B, 1024, 1408]
    |
    v
PerceiverResampler (trainable, 6 layers, 16 heads)
    |  compress 1024 -> 512 tokens, project to 4096-dim
    v
VisualProjection (trainable)
    |  Linear -> LayerNorm -> GELU -> Linear + start/end tokens
    v
[B, 514, 4096] prepended to text token embeddings
    |
    v
Qwen3.5-9B + LoRA (trainable)
    |  labels: -100 for visual tokens, report token ids for text
    v
Cross-entropy loss (causal LM)
```

## Data Sources

All 5 FORA data sources are supported:

| Source | Type | Windows |
|--------|------|---------|
| CT-RATE | Chest CT | full, mediastinal, lung, bone |
| Merlin | Abdominal CT | full, soft_tissue, liver, bone |
| MIMIC-CXR | Chest X-ray | single view, padded to 4 |
| CheXpert | Chest X-ray | single view, padded to 4 |
| PadChest | Chest X-ray | single view, padded to 4 |

For CT-RATE, ground-truth reports are read from a reports CSV (`Findings_EN` + `Impressions_EN` columns). For other sources, reports are constructed by joining the extracted sentence list.

Training can use all sources jointly (balanced round-robin) or a single source for domain-specific fine-tuning. Set `data.single_source` in `config.yaml`.

## Quick Start

1. **Edit `config.yaml`** with your data paths, RADRATE checkpoint path, and desired hyperparameters.

2. **Single GPU**:
   ```bash
   python train.py --config config.yaml
   ```

3. **Multi-GPU** (single node):
   ```bash
   accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
       train.py --config config.yaml
   ```

4. **SLURM** (multi-node):
   ```bash
   # Edit submit_train.sh: set your account, environment activation, etc.
   sbatch submit_train.sh
   ```

## Configuration

Key settings in `config.yaml`:

### Data
- `ctrate_folder` / `ctrate_jsonl` / `ctrate_meta`: CT-RATE dataset paths
- `reports_csv`: Full-text reports CSV for SFT targets (Findings + Impressions)
- `include_ctrate` / `include_merlin` / `include_mimic` / etc.: Enable/disable sources
- `single_source`: `null` for all sources, or `"ctrate"` / `"merlin"` / etc. for single-source training

### Model
- `radrate_checkpoint`: Path to the frozen RADRATE/FORA contrastive checkpoint
- `llm_model_name`: LLM to use (default: `Qwen/Qwen3.5-9B`)
- `freeze_visual`: If `true`, freeze all visual components (poolers, perceiver, projection)
- `pretrained_checkpoint`: Path to a previous SFT checkpoint to resume from or fine-tune

### Training
- `max_steps`: Total training steps
- `learning_rate`: LLM LoRA learning rate
- `visual_lr`: Visual encoder components learning rate (lower than LLM)
- `gradient_accumulation_steps`: Gradient accumulation (effective batch = local_bs x accum x num_gpus)
- `save_steps`: Checkpoint save interval
- `generate_every`: Sample report generation interval for W&B logging
- `resume`: Auto-resume from latest checkpoint in `output_dir`

### W&B
- `project`: W&B project name (set to `null` to disable)
- `run_name`: Custom run name (auto-generated if null)

## Checkpoint Format

Checkpoints are saved as `checkpoint_{step}.pt` containing:

```python
{
    "model_state_dict": {
        "visual_encoder.*": ...,  # poolers, perceiver, projection (not RADRATE)
        "llm.*": ...,             # Qwen + LoRA weights
    },
    "optim": ...,
    "scheduler": ...,
    "step": int,
}
```

This format is compatible with downstream GRPO fine-tuning via the `pretrained_checkpoint` config field.

## File Overview

| File | Purpose |
|------|---------|
| `perceiver_resampler.py` | Learnable-query cross-attention compressor |
| `model.py` | WindowAttentionPooler, MultiWindowEncoder, VisualProjection, VisualEncoder, factory builders |
| `data.py` | SFTReportDataset wrapper over FORA's CTReportDataset, reports CSV loading |
| `sft_trainer.py` | Accelerate-based trainer with W&B, checkpointing, EOS handling |
| `train.py` | CLI entry point |
| `config.yaml` | Default configuration |
| `submit_train.sh` | SLURM submission script |

## Typical Workflow

1. **Joint pretraining**: Train on all sources with `single_source: null`
2. **Domain fine-tuning**: Set `single_source: "ctrate"` and `pretrained_checkpoint` to the joint checkpoint
3. **GRPO (optional)**: Use the SFT checkpoint as initialization for RL-based report refinement
