# Encoder pre-training

Self-supervised pre-training of the FORA image encoder, a 3-D ViT-L (DINOv3 blocks, 4 registers, patch 2 x 16 x 16 voxels,
RoPE in millimetres) on CT and, separately, on chest X-rays (Methods, "Image encoder pre-training" and "Chest X-ray
encoder"). CT training has two phases: DINO + iBOT + KoLeo with a cross-crop patch term on the 1.5 x 0.5 x 0.5 mm grid,
then Gram anchoring to the phase-1 teacher. The X-ray encoder uses the same trainer with a film as a two-slice volume.
The exported EMA teacher is the frozen encoder of every later stage.

The network itself is in `fora/encoder` (shared with sentence alignment, the classifier and inference).

## Inputs and outputs

| | Format |
|---|---|
| CT shards | H5 `{patient}/{report}/{accession}/{series}/{data (X,Y,Z[,T]) HU, affine, pacs_metadata, dcm2niix_metadata}` and `{patient}/{report}/report/there_is_statements` (JSON list). Reports without statements are skipped. |
| X-ray shards | H5 `{patient}/{report}/{accession}/{film}/{data (H,W,1) uint16, dcm2niix_metadata}` |
| Split CSV | columns `h5_id`, `split`; shards with `split == train` are used |
| Checkpoints | `OUT/checkpoints/step_XXXXXXXX/` (PyTorch DCP: model + optimizer, per-rank sampler offset and RNG states, `metadata.json`), `OUT/metrics.jsonl` |
| Exported teacher | `teacher.pt` = `{format, backbone: state dict, metadata, step}`; optional `block24_state.pt` = `{block, norm, step}` |

## Code

| File | Content |
|---|---|
| `fora_pretrain/train.py` | entry point and training loop |
| `fora_pretrain/objective.py` | student / EMA teacher / Gram anchor, DINO, iBOT with position-binned Sinkhorn, KoLeo, Gram, cross-crop term, cosine prototypes, target monitor |
| `fora_pretrain/optim.py` | layer-wise lr decay groups, prototype-layer freezing, schedules |
| `fora_pretrain/views.py` | crop placement, student distortion, 3-D block masks, batching with twin-token pairs |
| `fora_pretrain/data_ct.py`, `data_cxr.py` | CT and X-ray sample rendering |
| `fora_pretrain/distributed.py` | FSDP2 wrapping, shard assignment and staging, DCP checkpoints |
| `fora_pretrain/export_teacher.py` | DCP checkpoint -> teacher backbone (+ block 24) |
| `scripts/launch_node.sh` | one task per node, 4 ranks |

## Run

Dependencies: PyTorch with FSDP2, `dinov3` (facebookresearch/dinov3 at commit 6876159, unmodified, on `PYTHONPATH`),
h5py, nibabel, scipy. The paper runs used 32 nodes x 4 H200 (global batch 1,024). Run from this folder (or put it on
`PYTHONPATH`) with the repository root importable.

```bash
export FORA_H5=/data/ct/h5 FORA_SPLITS=/data/ct/splits.csv FORA_STAGING=/local/stage FORA_COUNTS=/data/ct/recon_counts.json
# on every node i of N (MASTER = node 0):
scripts/launch_node.sh N i MASTER 29500 runs/ct_phase1 configs/ct_phase1.args
scripts/launch_node.sh N i MASTER 29500 runs/ct_gram   configs/ct_phase2_gram.args --resume runs/ct_phase1/checkpoints/step_00020000
python -m fora_pretrain.export_teacher --checkpoint runs/ct_gram/checkpoints/step_00005000 \
    --out ct_teacher.pt --block24 ct_block24_state.pt

export FORA_H5=/data/cxr/h5 FORA_SPLITS=/data/cxr/cxr_splits.csv
scripts/launch_node.sh N i MASTER 29500 runs/cxr configs/cxr.args
python -m fora_pretrain.export_teacher --checkpoint runs/cxr/checkpoints/step_00030000 \
    --out cxr_teacher.pt --block24 cxr_block24_state.pt
```

A run resumes from its own `latest` checkpoint. SIGTERM / SIGUSR1 save a checkpoint and exit with code 75.
Keep `checkpoints/step_00020000` of phase 1 (`--keep-checkpoints` rotates old ones away) or copy it elsewhere before
phase 2.

## Configs that reproduce the paper

| Config | Run |
|---|---|
| `configs/ct_phase1.args` | CT phase 1: 128 GPUs x 8, lr 1e-3 (2,500 warm-up steps), wd 0.04 -> 0.4, EMA 0.992 -> 1, teacher temperature 0.04 -> 0.07 over 10,000 steps, schedules over 125,000 steps, stop at 20,000 |
| `configs/ct_phase2_gram.args` | CT phase 2: resume from phase-1 step 20,000 with `--stage gram`; Gram weight 1, lr 5e-4 (250 warm-up), wd 0.1 -> 0.4, EMA 0.996, teacher temperature 0.07, schedules over 10,000 steps, stop at 5,000. The step-5,000 teacher is the released CT encoder |
| `configs/cxr.args` | X-ray: global 1,024 px / local 512 px at native resolution, position bins 1 x 3 x 3, student windows `0:1,0:0.65,0.2:0.8,0.45:1`, 30,000 steps |

Phase 2 starts from the phase-1 checkpoint including its AdamW moments, sampler position and RNG states; the step counter
and all schedules restart at 0 and the Gram anchor is the phase-1 EMA teacher.
