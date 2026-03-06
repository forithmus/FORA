# FORA: Foundation for Omnimodal Radiology Alignment

[![Tests](https://github.com/forithmus/RAD-RATE/actions/workflows/tests.yml/badge.svg)](https://github.com/forithmus/RAD-RATE/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/forithmus/RAD-RATE/branch/main/graph/badge.svg)](https://codecov.io/gh/forithmus/RAD-RATE)

Contrastive vision-language model for medical imaging that aligns CT volumes and radiology reports using VL-CABS loss. Supports multi-source training across CT and X-ray datasets with multiple fusion and pooling strategies.

## Architecture

**Image Encoder**: [VJEPA2](https://huggingface.co/facebook/vjepa2-vitg-fpc64-384) (ViT-G) with LoRA fine-tuning and a temporal CNN for depth downsampling.

**Text Encoder**: [BiomedVLP-CXR-BERT-specialized](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized).

**Fusion Modes** for combining multiple CT window reconstructions:

| Mode | Strategy |
|------|----------|
| `early` | Stack windows into channels before the encoder |
| `mid_cnn` | Process separately through CNN, merge features, then transformer |
| `late` | Siamese processing, merge at token level via masked average |
| `late_attn` | Siamese processing with learned attention-based pooling |

**Pooling Strategies** (for `late_attn` mode):

| Strategy | Description |
|----------|-------------|
| `simple_attn` | Learned attention weights over reconstructions |
| `cross_attn` | Text-guided cross-attention pooling |
| `gated` | Gated attention with text conditioning |

## Repository Structure

```
RAD-RATE/
├── rad_rate/                  # Core model package
│   ├── rad_rate/
│   │   └── rad_rate.py        # RADRATE model, pooling modules, VL-CABS loss
│   └── setup.py
├── vision_encoder/            # Vision encoder package
│   ├── vision_encoder/
│   │   ├── vjepa_encoder.py   # VJEPA2Encoder with LoRA + temporal CNN
│   │   └── optimizer.py       # Optimizer utilities
│   └── setup.py
├── scripts/                   # Training, inference, and evaluation
│   ├── run_train.py           # Training entry point
│   ├── rad_rate_trainer.py    # Distributed trainer class
│   ├── data.py                # Multi-source dataset loader
│   ├── fast_inference.py      # Zero-shot inference
│   ├── fast_inference_new.py  # Inference with configurable pooling
│   ├── bootstrap_values.py    # Bootstrap confidence intervals
│   ├── eval.py                # Evaluation metrics (AUROC, F1, etc.)
│   └── data_inference_nii_fixed.py  # Inference dataset loader
├── tests/                     # Unit tests (90 tests, 95% core coverage)
├── requirements.txt           # All dependencies
└── pyproject.toml             # Pytest + coverage configuration
```

## Installation

```bash
git clone https://github.com/<your-org>/RAD-RATE.git
cd RAD-RATE

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install local packages in editable mode
pip install -e rad_rate/ -e vision_encoder/
```

## Training

RAD-RATE uses [Accelerate](https://huggingface.co/docs/accelerate) for distributed training across multiple GPUs/nodes.

### Quick Start

```bash
# Single-node multi-GPU
accelerate launch --num_processes 4 scripts/run_train.py \
    --fusion_mode late_attn \
    --pooling_strategy simple_attn
```

### Multi-Node Distributed Training

```bash
# Using accelerate (recommended)
accelerate launch \
    --multi_gpu \
    --num_machines <NUM_NODES> \
    --num_processes <TOTAL_GPUS> \
    --machine_rank <NODE_RANK> \
    --main_process_ip <MASTER_ADDR> \
    --main_process_port <MASTER_PORT> \
    scripts/run_train.py \
        --fusion_mode late_attn \
        --pooling_strategy simple_attn
```

### SLURM Cluster

```bash
#!/bin/bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=450G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

srun accelerate launch \
    --num_processes $((SLURM_NNODES * 4)) \
    --num_machines $SLURM_NNODES \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    scripts/run_train.py \
        --fusion_mode late_attn \
        --pooling_strategy simple_attn
```

### Training Arguments

| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--fusion_mode` | `early`, `mid_cnn`, `late`, `late_attn` | required | How to combine multi-window CT reconstructions |
| `--pooling_strategy` | `simple_attn`, `cross_attn`, `gated` | `simple_attn` | Reconstruction pooling (only used with `late_attn`) |

### Training Configuration

Key parameters in `scripts/run_train.py` (edit before training):

```python
# Model
dim_text = 768          # BiomedVLP-CXR-BERT hidden size
dim_latent = 512        # Shared latent dimension
lora_r = 32             # LoRA rank
lora_alpha = 64         # LoRA alpha

# Training
batch_size = 1          # Per-GPU batch size (effective = batch × GPUs × grad_accum)
lr = 1e-5               # Learning rate
warmup_steps = 500      # Linear warmup steps
num_train_steps = 100001
save_model_every = 500  # Checkpoint frequency
```

### Data Sources

The trainer supports 5 medical imaging datasets with balanced round-robin sampling:

| Dataset | Modality | Windows |
|---------|----------|---------|
| [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) | Chest CT | full, mediastinal, lung, bone |
| [Merlin](https://github.com/StanfordMIMI/Merlin) | Abdominal CT | full, soft tissue, liver, bone |
| [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/) | Chest X-ray | N/A |
| [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) | Chest X-ray | N/A |
| [PadChest](https://bimcv.cipf.es/bimcv-projects/padchest/) | Chest X-ray | N/A |

Configure dataset paths and inclusion flags in `run_train.py` and `rad_rate_trainer.py`.

## Inference

Zero-shot pathology classification on the CT-RATE validation set.

### Basic Inference

```bash
python scripts/fast_inference.py \
    --fusion_mode late_attn \
    --weights_path /path/to/RadRate.2000.pt \
    --batch_size 1 \
    --results_folder ./results
```

### Inference with Pooling Strategy

```bash
python scripts/fast_inference_new.py \
    --fusion_mode late_attn \
    --pooling_strategy cross_attn \
    --weights_path /path/to/RadRate.2000.pt \
    --batch_size 1 \
    --results_folder ./results \
    --data_folder /path/to/validation/data \
    --reports_file /path/to/reports.jsonl \
    --meta_file /path/to/metadata.csv \
    --labels_file /path/to/labels.csv
```

### Inference Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--fusion_mode` | required | Fusion mode used during training |
| `--pooling_strategy` | `simple_attn` | Pooling strategy (fast_inference_new.py only) |
| `--weights_path` | required | Path to model checkpoint |
| `--batch_size` | `1` | Inference batch size |
| `--results_folder` | `./inference_results` | Output directory |
| `--data_folder` | — | Path to CT validation data |
| `--reports_file` | — | Path to reports JSONL |
| `--meta_file` | — | Path to metadata CSV |
| `--labels_file` | — | Path to ground truth labels CSV |

### Evaluated Pathologies (18)

Medical material, Arterial wall calcification, Cardiomegaly, Pericardial effusion, Coronary artery wall calcification, Hiatal hernia, Lymphadenopathy, Emphysema, Atelectasis, Lung nodule, Lung opacity, Pulmonary fibrotic sequela, Pleural effusion, Mosaic attenuation pattern, Peribronchial thickening, Consolidation, Bronchiectasis, Interlobular septal thickening

### Outputs

Inference produces the following files in `results_folder`:

| File | Content |
|------|---------|
| `predicted_weights.npz` | Raw prediction scores per pathology |
| `labels_weights.npz` | Ground truth labels |
| `accessions.txt` | Scan IDs processed |
| `aurocs.xlsx` | Per-pathology AUROC scores |

## Evaluation

### Bootstrap Confidence Intervals

After running inference:

```bash
python scripts/bootstrap_values.py
```

Edit the paths inside the script to point to your `predicted_weights.npz` and `labels_weights.npz`. Produces:
- `aurocs_bootstrap.xlsx` — AUROC with 95% CI
- `f1_bootstrap.xlsx` — F1 with 95% CI
- `acc_bootstrap.xlsx` — Accuracy with 95% CI
- `precision_bootstrap.xlsx` — Precision with 95% CI

## Testing

```bash
# Run all tests with coverage
python -m pytest

# Run specific test files
python -m pytest tests/test_rad_rate_model.py -v
python -m pytest tests/test_fusion_modes.py -v

# Run without coverage (faster)
python -m pytest --no-cov
```

Coverage report is generated as HTML in `htmlcov/`.

### Test Suite (90 tests)

| File | Tests | Coverage |
|------|-------|----------|
| `test_imports.py` | Dependency + package import verification | All imports |
| `test_pooling.py` | SimpleAttnPool, CrossAttnPool, GatedAttnPool | Shapes, masking, gradients |
| `test_rad_rate_model.py` | RADRATE model init, forward, loss, serialization | 95% of core model |
| `test_fusion_modes.py` | All 4 fusion modes × all pooling strategies | End-to-end forward pass |
| `test_vision_encoder.py` | ResidualTemporalDownsample, VJEPA2 preprocessing | CNN shapes, gradients |

