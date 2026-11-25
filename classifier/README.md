# Finding classifier

This stage trains the organ-routed multiple-instance classifier that recognises 1,028 CT findings (749 own heads and
279 rare heads attached to a parent) from a whole study bag, and a 129-class chest X-ray classifier
(Methods, "Finding classifier" and "Chest X-ray"). It also fits the operating points: Platt calibration and the plan tiers
used by report generation, and the rule-A cut-offs behind the reported balanced accuracy. Last, it exports the signed
evidence, foci and evidence images that the report writer reads (Methods, "Finding localization").

The model code shared with report generation and inference lives in `fora/mil/`:

| module | content |
|---|---|
| `fora.mil.model` | gated-attention head (hidden 512), token MLP (hidden 384, GELU), per-finding scale and bias, soft routing penalty softplus(r), exact two-pass streamed forward / backward over a study bag |
| `fora.mil.bag` | CT study bags from the block-23 cache, per-token region bits, route masks (< 8 routed tokens in a view = whole view), token factory (standardisation, token / feature drop, position features) |
| `fora.mil.xray` | X-ray head with the view-kind embedding and its token factory |
| `fora.mil.evidence` | CT signed evidence, foci (world grid, 4 mm Gaussian, family restriction, HDR50, 26-connected components); X-ray referenced evidence and film focus |
| `fora.mil.render` | axial / coronal / sagittal evidence images |
| `fora.mil.calibration` | apply Platt curves, plan tiers and rule-A cut-offs; scan-type rules |

## Inputs and outputs

| file | format | produced by |
|---|---|---|
| `cases.jsonl` | one line per report: `id`, `report`, `views` (pack `file`, `offset`, `file_size`, `relative_file`, `group`, `n_tokens`, `series`, `window`) | data-preprocessing |
| `assign_128ranks.json` | `{"ranks": 128, "assignments": {"train": [[ids] x 128], "val": [...]}}` (size-balanced) | data-preprocessing |
| `split.json` | `{report id: "train" / "val"}` (patient level) | data-preprocessing (`make_split.py`) |
| block-23 cache, region bits | series HDF5 files under `cache_root`, `<region_root>/<relative_file>.npz` (`bits`, uint64 per token) | data-preprocessing |
| `normalization.pt`, `block24_state.pt` | `{"mean": [1024], "rms": float}`, `{"block": ..., "norm": ...}` | data-preprocessing / encoder-pretraining |
| `aligned_block24.pt` | `{block, step, epoch}` (keys `block.*`, `norm.*`), read by `init_block_from` | sentence-alignment `export_block24` |
| `labels.npz`, `heads.json` | `nodes`, `cases`, `state` [reports, nodes] (0 denied, 1 asserted, 2 uncertain, 3 unmentioned, -1 unknown); `own` / `attached` heads (`heads.json` = `labels/data/ct/heads_1028.json`) | labels |
| `routes/` | routing tables (`class_routes_table*.json`, `class_region_bits_table*.json`) | this stage |
| `model_epoch_XX.pt` | `head`, `block`, `ema`, `classes` | `train_ct.py` |
| `dense_predictions.npz` | `fora_id`, `report_uid`, `logit` [N, 1028], `label` | `predict_ct.py merge` |
| calibration JSON | Platt slope / intercept, raw and calibrated tier cut-offs, base rates per finding | `fit_calibration.py` |
| `ruleA_cutoffs.json` | `fora_per_scan_type` {finding: {scan type: cut}}, `external` | `fit_rule_a.py` |
| `extent_r*_p0_c*.npz` | per (report, finding): outputs, pooled feature, regions, foci (`fo_*`), JPEG evidence images | `export_evidence_ct.py` |

Config paths use `${FORA_DATA}` and `${FORA_RUNS}`; the routing-table paths are relative to the repository root. Run the
scripts from the repository root with `classifier/` on `PYTHONPATH` (or from `classifier/`), and DINOv3 importable.

## Steps

```bash
# 1. routing tables (table1 is shipped; table2 and table3 are rebuilt from it)
python classifier/build_routes.py table2 --hierarchy labels/hierarchy.json --out classifier/routes
python classifier/build_routes.py table3 --heads $FORA_DATA/labels/heads.json --out classifier/routes

# 2. CT classifier, 128 ranks (32 nodes x 4 GPUs), one output directory per stage
torchrun --nnodes 32 --nproc-per-node 4 ... classifier/train_ct.py --config classifier/configs/ct_stage1a.json --output $FORA_RUNS/ct_stage1
torchrun ... classifier/train_ct.py --config classifier/configs/ct_stage1b.json --output $FORA_RUNS/ct_stage1   # resumes latest.pt
torchrun ... classifier/train_ct.py --config classifier/configs/ct_stage2.json  --output $FORA_RUNS/ct_stage2
torchrun ... classifier/train_ct.py --config classifier/configs/ct_stage3.json  --output $FORA_RUNS/ct_stage3
# final CT model: $FORA_RUNS/ct_stage3/model_epoch_04.pt, raw weights (keys head, block)

# 3. study-bag logits of all reports, calibration, rule A
python classifier/predict_ct.py run   --config classifier/configs/ct_stage3.json --ckpt $FORA_RUNS/ct_stage3/model_epoch_04.pt --out preds --shard $i --shards 128
python classifier/predict_ct.py merge --config classifier/configs/ct_stage3.json --out preds --dense dense_predictions.npz
python classifier/fit_calibration.py ct --dense dense_predictions.npz --labels $FORA_DATA/labels/labels.npz --split $FORA_DATA/splits/split.json --out calibration_ct.json
python classifier/fit_rule_a.py --dense dense_predictions.npz --labels $FORA_DATA/labels/labels.npz --split $FORA_DATA/splits/split.json \
       --protocols protocols.json --external external_mapping.json --out ruleA_cutoffs.json

# 4. evidence, foci and evidence images for the report writer's (report, finding) pairs
python classifier/export_evidence_ct.py --config classifier/configs/ct_stage3.json --ckpt $FORA_RUNS/ct_stage3/model_epoch_04.pt \
       --targets targets.json --reports plan_reports.jsonl --h5-root $CT_H5 --out evidence --shard $i --shards 128 \
       --crops 32 --chunk 200 --render 1

# 5. X-ray classifier (128 ranks), calibration and evidence reference
torchrun ... classifier/train_xray.py --config classifier/configs/xray.json --output $FORA_RUNS/xray
python classifier/predict_xray.py run --config classifier/configs/xray.json --model $FORA_RUNS/xray/model_best_raw.pt --cases $FORA_DATA/xray/cases_train.jsonl --out preds_xtr --shard $i --shards 16
python classifier/predict_xray.py merge --out preds_xtr --npz preds_train.npz          # same for the validation cases
python classifier/fit_calibration.py xray --preds-train preds_train.npz --preds-val preds_val.npz --labels $FORA_DATA/xray/labels.npz --out calibration_xray.json
python classifier/xray_evidence_reference.py run   --config classifier/configs/xray.json --model $FORA_RUNS/xray/model_best_raw.pt --out xref --shard $i --shards 8
python classifier/xray_evidence_reference.py merge --config classifier/configs/xray.json --out xref --json localization_reference_bc.json
```

## Configs that reproduce the paper

| config | init | routing table | schedule | lr head / block | stop |
|---|---|---|---|---|---|
| `ct_stage1a.json` | block 24 from CT sentence alignment, fresh head | table1 | 12-epoch cosine, warm-up 0.25 epoch | 1e-3 / 2e-5 | step 24,800 |
| `ct_stage1b.json` | resumes stage 1 (`latest.pt`) | table2 | same schedule | same | step 26,356 (end of epoch 11) |
| `ct_stage2.json` | stage 1 `model_epoch_11.pt` | table2 | 2-epoch cosine, warm-up 0.1 | 5e-4 / 1e-5 | step 2,800 (`latest.pt`) |
| `ct_stage3.json` | stage 2 `latest.pt`, route penalty 1.0 for 79 rare routed findings | table3 | 4-epoch cosine, warm-up 0.1 | 5e-4 / 1e-5 | step 9,584 (complete) |
| `xray.json` | block 24 from X-ray sentence alignment | none (every token routed) | 6-epoch cosine, warm-up 0.25 | 1e-3 / 2e-5 | complete (12,078 steps) |

Common CT settings: 128 ranks x 1 study bag per step (2,396 steps per epoch), AdamW weight decay 0.05 (0 for vectors and the
route penalty), route lr 1e-2, route penalty initialised at 5, token drop 0.2, feature drop 0.1, EMA 0.998,
attach decay 0.5, gradient clip 1.0, seed 0. Total CT steps: 26,356 + 2,800 + 9,584 = 38,740.

## Tests

```bash
cd classifier && python -m pytest -q tests
```
