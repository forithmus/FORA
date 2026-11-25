# Report generation

This stage turns the frozen CT classifier into a report writer (paper section "Grounded report generation", Methods
"Report generation", Supplementary Fig. 11). For each study a diagnostic plan is built from the calibrated classifier
probabilities. Each planned finding carries a description of its evidence (distribution, sites, measurement), evidence
focus tokens and zoomed evidence images. A Gemma-4-31B-it writer with LoRA reads the study's encoder tokens through a
512-latent resampler together with the plan and writes the Findings section. The CT writer is trained in two SFT stages
and one offline DPO stage. The chest X-ray writer uses the same model and is trained with SFT, DPO and PU-DPO.

## Layout

The runtime used by both this stage and `inference/` is the shared package `fora.reportgen`; this folder holds the
training, pair-synthesis, calibration-tier and batch-decoding command lines (package `fora_report`), their configs and
data.

| location | content |
|---|---|
| `fora/reportgen/plan/` | diagnostic plan builder and its fixed prompt text (`plan_templates.txt`) |
| `fora/reportgen/evidence/` | evidence extractor, finding descriptions, focus-token store, finding store, evidence-image batching |
| `fora/reportgen/measure/` | lesion calipers and CADS anatomy measurements used in the finding descriptions |
| `fora/reportgen/writer/` | writer model (resampler, conditioning tokens, Gemma-4 + LoRA, visual stream) |
| `fora/reportgen/decoding/` | per-study seed, forced opener, phrase-ban lists (`ban_lists/`), writer construction and decoding |
| `fora/reportgen/cxr/` | chest X-ray visual stream, writer model, plan rules, location text, promotion list |
| `fora/mil/` | signed evidence, foci, extent statistics and evidence images (shared with the classifier stage) |
| `fora_report/plan/` | plan calibration (train Platt + Youden tiers), evidence targets |
| `fora_report/evidence/` | merging of per-node evidence stores |
| `fora_report/writer/` | FSDP2/HSDP training, DPO under FSDP, node-local logit cache |
| `fora_report/ct/` | CT SFT and DPO trainer |
| `fora_report/pairs/` | CT DPO pair synthesis |
| `fora_report/decoding/` | CT report generation CLI (cohorts, records) |
| `fora_report/cxr/` | chest X-ray writer: calibration, locations, SFT, DPO, PU-DPO, generation with plan rules |
| `configs/` | arguments of the paper runs |
| `prompts/` | fixed prompt text (writer instruction, Gemma pair-synthesis and editor prompts) |
| `data/` | caliper groups and scale calibration, CADS measurement calibration, measurement schema |
| `tests/` | CPU smoke tests |

Modules run as `python -m fora_report.<sub>.<module>` from this folder (or with this folder on `PYTHONPATH`) and as
`python -m fora.reportgen.<sub>.<module>` for the runtime. The shared package `fora` must be installed
(`pip install -e ..`). External models: `google/gemma-4-31B-it`, `facebook/dinov2-large` (calipers) and CADS
segmentations. Tests: `python -m pytest tests`.

## CT pipeline order

1. Classifier stage: frozen MIL study-bag logits for all studies (dense npz) and the block-23 caches.
2. `fora_report.plan.calibration_tiers` gives the plan calibration, `fora_report.plan.targets` the planned findings.
3. `fora.reportgen.measure` gives the per-finding measurement text and the study measurement line.
4. `fora.reportgen.evidence.extract`, `focus_tokens` and `merge_store` give the finding store, descriptions, focus tokens and images.
5. `fora_report.ct` SFT stage 1, then SFT stage 2.
6. `fora_report.pairs` builds the DPO pair pool, then `fora_report.ct` DPO starts from SFT stage 2.
7. `fora_report.decoding.run` writes the reports for FORA val, CT-RATE and Merlin.

## Diagnostic plan (`fora/reportgen/plan/`, `fora_report/plan/`)

The writer never sees the classifier logits directly. Each study's MIL logits pass through per-class Platt curves
fitted on the training studies. A class enters the **confident** tier when its calibrated probability is at least
0.5 (never below the train Youden cutoff). It enters the **possible** tier when its raw probability passes the train
Youden cutoff but not the confident cutoff. For CT the possible tier keeps at most 20 findings per study, ranked by
calibrated probability. The confident tier is never capped. The X-ray possible tier is uncapped. Each planned finding is written as
`<name> (calibrated probability p, Nx the base rate of X%; <evidence description>) <evidence tokens><image tokens>`,
where the base rate is (train positives + 0.5) / (train studies + 1). The fixed prompt text (Supplementary Fig. 11)
is in `fora/reportgen/plan/plan_templates.txt`.

| module | role |
|---|---|
| `diagnostic_plan.py` | `DiagnosticPlanBuilder`: tiers, possible-tier cap, plan text, prompt token ids, placeholder order for evidence tokens |
| `calibration_tiers.py` | Platt fit (ridge toward (1, 0), weight 1.0, slope floor 0.05), Youden cutoffs, tier cutoffs; CLI builds the CT calibration JSON |
| `targets.py` | lists every planned (case, class) for the evidence extractor (`--targets` input) |

Inputs: dense study-bag MIL logits (npz: `report_uid`, `fora_id`, `logit` [n, C] float32, `label` [C]), the label
state matrix (npz: `nodes`, `cases`, `state` with 1/2 = asserted/uncertain -> positive, 0/3 = denied/unmentioned ->
negative, -1 = masked), and the patient-level split json (case id -> split).

```bash
# 1. calibration (CPU)
python -m fora_report.plan.calibration_tiers --dense dense_logits.npz --labels-npz labels.npz --split split.json \
    --out calibration_train_platt.json --stats calibration_train_platt_stats.json
# 2. planned findings that need evidence tokens / zoomed images (CPU)
python -m fora_report.plan.targets --calibration calibration_train_platt.json --dense dense_logits.npz \
    --split split.json --out-dir targets/ --plan-possible-cap 20 --plan-cap-rank prob \
    [--existing-store finding_store.npz] [--measurable-classes measurable.json --measurements lesion_meas.json]
```

Configs: `configs/plan_ct.json` (SFT stage 2, DPO, decoding), `configs/plan_ct_stage1.json` (SFT stage 1),
`configs/plan_cxr.json` (X-ray writer).

SFT stage 1 uses its own plan calibration: Platt fitted on val, a best-F1 confident cutoff and a val recall-0.9
possible cutoff, no cap. SFT stage 2, DPO and decoding use the final tiers above. In the calibration JSON the keys
`class_val_positive` and `n_calibration_studies` hold the train positive counts and the train study count.

## Evidence tokens, descriptions and zoomed evidence images (`fora/reportgen/evidence/`)

For every planned finding the writer receives a description, one pooled evidence token, up to 16 focus tokens and up to six
zoomed evidence images. All four come from the frozen CT MIL classifier. Block 24 and the organ-routed head run once over all
body tokens of all views; the signed evidence of a class is `attention x token_logit x output_scale` (attention = softmax over all
tokens of all views), and only its positive part is used. The positive evidence is smoothed with a 4 mm Gaussian on a world token
grid, restricted to the finding's anatomical family when that family holds evidence, cut to the region holding 50 % of the mass,
and split into 26-connected foci. Sixteen evidence tokens are spread over the foci by farthest-point sampling. The description
names the distribution (focal / multifocal / diffuse), the main site and its position inside the organ, the measurement, a tally of
sites by region and further sites. The images are axial, coronal and sagittal slices through the evidence peak plus axial slices
through up to three further sites, with the positive evidence overlaid as in the paper figures.

| module | role |
|---|---|
| `extract.py` | streaming evidence pass, foci, peak blocks, renders; writes `extent_r<rank>_p<part>_c<chunk>.npz` (training stores and inference) |
| `fora.mil.evidence` | signed evidence pass (`study_evidence`), foci decomposition and 16 focus tokens (`foci`) |
| `fora.mil.extents` | extent statistics and the token blocks around the evidence peak |
| `fora.mil.render` | zoomed evidence JPEGs |
| `describe.py` | location / pattern / measurement wording, anatomical families |
| `focus_tokens.py` | extractor chunks -> focus-token store (`keys, x [n,16,1048] f16, aux [n,16,16] f16, n_valid, fimg_blob, fimg_off [n,7], desc S600, loc_part.npz`) |
| `fora_report/evidence/merge_store.py` | per-node stores -> global localization store (`parts`, `union`) and focus-token store union (`tokens`) |
| `finding_store.py` | `FindingLocStore` read by the writer; `write_study_loc_dir` for one study at inference |
| `zoom_images.py` | images -> Gemma-4 image-processor tensors (`max_soft_tokens=140`) |

Inputs: CT classifier config JSON (normalization, block24_state, class_region_bits, cases, assign, cache_root, region_root),
classifier checkpoint, block-23 cache (or its node-local staged copy, `--cache-root`), targets JSON `{"targets": {case: [cls...]}, "case2rep": {case: report_uid}}` (from `fora_report.plan.targets`), CT h5 + study
plan JSONL (training renders) or `ct_<series>.npy` volumes (inference), measurement text JSON `{"<case>|<cls>": phrase}` (from
`fora.reportgen.measure`).

Commands (training store, per data rank, 3 parts per GPU):
```
python -m fora.reportgen.evidence.extract --config mil_config.json --ckpt mil.pt --targets plan_targets.json --out EXT \
    --rank R --part P --parts 3 --resume-rank 1 --cache-root STAGED_CACHE --crops 32 --block -1 --render 1 --chunk 200 \
    --plan-reports plan_train.jsonl plan_val.jsonl --h5-root H5
python -m fora.reportgen.evidence.focus_tokens EXT PARTS/node_NN --labels labels.json --region-names regions.json --measurements meas_by_case_cls.json
python -m fora_report.evidence.merge_store tokens OLD_NODE_STORE PARTS/node_NN FTOK_NODE     # union with an existing store (existing rows are kept)
python -m fora_report.evidence.merge_store union PARTS loc_store.npz desc.npy --old-store OLD.npz --old-desc OLD_desc.npy
```
Inference (one study): `fora.reportgen.evidence.extract.extract_ct_evidence(..., cases=case_ids.txt, block=2, chunk=1,
ct_dir=WORK)`, then `focus_tokens.build_store`, then `finding_store.write_study_loc_dir`.

The training store is built in two passes (planned findings of the stage-1 plan, then the findings added by the final
plan tiers) and merged with `merge_store union` / `merge_store tokens`; rows already present are kept.

## Measurements (`fora/reportgen/measure/`)

Size phrases in the finding descriptions come from two image-only measurers (paper: Methods, "Measurements").
Lesion, wall-thickness and lumen sizes are measured by segment-then-measure calipers on the strongest MIL evidence focus of the finding; the size is measured geometrically on the predicted mask (RECIST 1.1), never regressed.
Organ, vessel, midline-shift, Evans-index and cardiothoracic-ratio values are measured on CADS anatomical segmentations without learning.

**Lesion calipers.** There are seven finding-group calipers: air, cyst, node, nodule, solid, stone and thick (`data/measure/caliper_groups.json` lists each group's (class, type) pairs).
For each candidate the model reads a lesion-centred crop and runs the frozen classifier ViT-L, a frozen class-conditioned 3-D mask head, and a 2-D decoder on the axial RECIST slice.
The 3-D head uses the frozen-MIL class evidence of the candidate's 5x5x5 token block. The 2-D decoder takes frozen DINOv2-L tokens of the slice as input.
The raw value is divided by the group's per-class scale calibration, exp(median log(caliper / report)) on the training split, stored in `data/measure/caliper_scale_<group>.json`.

| module | what |
|---|---|
| `geometry.py` | RECIST long / short axis, 3-D extent, component pick |
| `crops.py` | crop reader (H5 series → cache frame, organ route from token-region bits) |
| `encoder.py` | frozen 3-D ViT-L on upsampled crops |
| `caliper_model.py` | 3-D head, 2-D decoder, view geometry, 2-D measurement |
| `caliper.py` | `infer` / `train` CLI |
| `scale_calibration.py` | per-class scale offsets |
| `tables.py` | candidate / target tables (class schema, training peak shards, inference extent files) |
| `lesion_measure.py` | inference driver |
| `merge.py` | training-cohort `{case|cls: text}` |

**CADS anatomy** (`fora/reportgen/measure/cads/`):

| module | what |
|---|---|
| `caselist.py` | studies and predicted measurement classes |
| `measure.py` | per-study raw values, calibrated records and reference values |
| `vessel.py` | aorta / pulmonary-trunk centreline cross-sections |
| `organ.py` | liver / spleen / kidney geometry |
| `midline.py` | mid-sagittal symmetry plane + plane-frame resampling |
| `study.py` | study measurement line + Evans / CTR finding texts |
| `classes.py` | measurement classes |

The calibration file is `data/measure/cads_calibration.json`.

**Inputs**
- CT-FORA H5 volumes (`--h5-root`).
- CADS label files `cads_<h5>.h5` (`--cads-root`).
- Staged case lines (JSONL `{id, views:[{relative_file, file, offset, file_size, window}]}`).
- Per-view token-region bits (`--regions-dir` with `<relative_file>.npz`, or `--region-packs`).
- Classifier checkpoint + config.
- CT encoder teacher backbone + token normalisation.
- DINOv2-L (`facebook/dinov2-large`).
- Caliper checkpoints `caliper_<group>.pt` (artifacts; keys `head`, `dec2d`, `b24`, `nk`).

**Outputs**
- `cal_r*_c*.npz` per caliper run.
- `measurements.json` + `{case|cls: text}`.
- CADS JSONL records.
- `study_measurements.json` `{report_uid: line}`.

**Commands**

```bash
# CADS anatomy (CPU)
python -m fora.reportgen.measure.cads.caselist --dense DENSE.npz --calibration MIL_CAL.json --index INDEX.sqlite --all --out cases_all.jsonl
python -m fora.reportgen.measure.cads.measure --cases cases_all.jsonl --out cads_0.jsonl --calib data/measure/cads_calibration.json \
    --h5-root H5 --cads-root CADS --procs 64 [--node-idx i --nnodes n]
python -m fora.reportgen.measure.cads.study --cases cases_all.jsonl --raw "cads_*.jsonl" --calib data/measure/cads_calibration.json \
    --out-study study_measurements.json --out-findings study_finding_measurements.json

# lesion calipers, training-cohort text (per node: peak shards -> tables -> 7 groups -> merge)
python -m fora.reportgen.measure.tables peaks --schema data/measure/measurement_schema.json PEAK_DIR NODE_DIR
torchrun --nproc_per_node 4 -m fora.reportgen.measure.caliper infer --ckpt caliper_<group>.pt --cand NODE_DIR/cand_local.npz \
    --targets NODE_DIR/targets_local.npz --exclude "<group exclude>" --k 1 --split all --out NODE_DIR/cal_<group> <common inputs>
python -m fora.reportgen.measure.merge PULL_DIR --scales data/measure --extra study_finding_measurements.json --out finding_measurements.json

# inference (one study, after the evidence extractor)
python -m fora.reportgen.measure.lesion_measure --extent-dir EXT --out OUT --calipers MODELS --data data/measure <common inputs> --ln-short-axis
# (CADS anatomy of one study: fora.reportgen.measure.anatomy.measure_anatomy, used by inference/ct)

# caliper training (one group) and its scale calibration
torchrun --nproc_per_node 4 -m fora.reportgen.measure.caliper train --init INIT.pt --cand CAND.npz --targets TARGETS.npz --info INFO.npz \
    --ln-short-axis --exclude "<group exclude>" --k 8 --epochs 2 <common inputs>
python -m fora.reportgen.measure.caliper infer --ckpt head_ep1.pt ... --k 16
python -m fora.reportgen.measure.scale_calibration OUT --cand CAND.npz --targets TARGETS.npz --k 16 --out caliper_scale_<group>.json
```

Here `<common inputs>` stands for:

```
--cases --h5-root --regions-dir --encoder-ckpt --mil-ckpt --mil-config --normalization --dino2d
```

The full caliper recipe is in `configs/measure_calipers.json`.

The caliper training command trains the 2-D decoder; the 3-D head and block 24 are loaded frozen from `--init`.
The training-cohort measurement text uses the calipers at `--k 1` on the top MIL peak per (study, predicted class).

## Writer model (`fora/reportgen/writer/`, training in `fora_report/writer/`)

The writer turns one study into a report with Gemma-4-31B-it. Its input sequence is a bidirectional prefix followed
by the report:

1. **Visual prefix.** The study's encoder tokens are produced exactly as the finding classifier sees them: the
   block-23 cache of every view, the classifier's frozen block 24, normalisation, and Fourier position features. This
   gives 1,048-d tokens (`dino_stream.DinoVisualStream`). Zero-initialised view embeddings (window kind and series
   ordinal) are added to the tokens. A query resampler (`resampler.QueryResampler`) has 512 latents, width 1024,
   depth 2 and 8 heads, and reads all tokens of the study. For very large bags it uses exact chunked online-softmax
   attention. The latents and the trainable study-type tokens are projected to the LLM width and placed between
   learned `image_start` and `image_end` tokens. During training the study's tokens are subsampled uniformly to at
   most 524,288 after block 24. Inference uses all tokens.
2. **Classifier tokens.** There is one token per vocabulary class. It is built from the class-name embedding and
   the classifier's frozen Platt-calibrated probability (`conditioning.FrozenPlattLabelConditioner`).
3. **Diagnostic plan.** The prompt and plan text come from `fora.reportgen.plan`. Each planned finding carries
   placeholder tokens:
   - Placeholders for its pooled evidence token and up to 16 focus tokens. These are 1,064-d evidence features,
     projected by `conditioning.FindingTokenProjector`.
   - Optionally one placeholder per zoomed evidence image, up to 6 views. Each image is encoded by the Gemma-4 vision
     tower and attention-pooled to one token (`conditioning.FindingImagePool`). Images are encoded 32 at a time.
     During training at most 64 images per study keep vision-tower gradients.
4. **Report.** Training uses the token-mean cross-entropy over the report, computed in checkpointed fp32 chunks of
   64 positions. The checkpoint also holds a diagnostic-consistency head on the mean report hidden state, with loss
   weight 0.

LoRA with rank 16 and alpha 32 is attached to every attention and MLP projection of the language model
(`llm.build_gemma_lora`). LoRA dropout is 0 during training (`fsdp.train_mode_for_checkpointing`). The vision tower
and its projector are trained fully when evidence images are used. Training runs under FSDP2/HSDP, with a
(nodes, GPUs per node) mesh, one shard unit per decoder layer and the writer as the root unit (`fsdp.py`).
Checkpoints store full trainable tensors and a plain optimizer state dict. DPO under FSDP keeps the frozen reference
as local parameter shards and runs a fixed per-shard-group pass schedule (`fsdp_dpo.py`).

Node-local caches of the classifier logits and the evidence store are created once per node:

    python -m fora_report.writer.dense_cache --source dense_predictions.npz --output CACHE/dense
    python -m fora_report.writer.dense_cache --kind loc --source evidence_store.npz --output CACHE/loc

Inputs:
- dense logits npz (`report_uid`, `full_token_count`, `logit` float32 [studies, classes], `label`, `class_index`)
- evidence store npz (`report_uid`, `row_start`, `cls`, `prob`, `reg3`, `frac3`, `left`, `right`, `zfrac`, `view`,
  `feat`, `region_names`, `labels`)
- classifier config JSON (`fingerprint`, `cases`, `assign`, `normalization`, `block24_state`,
  `training.position_freqs`) and the classifier checkpoint (`block` = block-24 state)
- block-23 cache under `<local_root>/<fingerprint>/`

Fixed settings used by the paper runs, all as constructor defaults or explicit arguments:
- resampler: kv chunk 32,768, streaming threshold 524,288, streaming chunk 8,192
- activations of sequences of at least 6,000 positions are kept in host RAM if they are at least 16 MiB
- CE chunk 64
- DPO slot scope `shard`
- CT: `fimg_tokens=6`, `fimg_views=6`. X-ray: `fimg_tokens=0`. X-ray finding features are 1,048-d (no auxiliary
  features).

## CT writer training (SFT stage 1, SFT stage 2, offline DPO)

`fora_report/ct/` trains the CT report writer (paper: "Grounded report generation"). One optimizer step is one study
per GPU (global batch 128 = 32 nodes x 4 GPUs). Each study's input is its block-23 DINO token cache passed through the
frozen classifier block 24 (subsampled to 524,288 tokens), the diagnostic plan built from the frozen classifier
logits with evidence descriptions, focus tokens and zoomed evidence images, and the fixed writer instruction
(`prompts/ct_writer_instruction.txt`). The target is `Findings:\n` + the Findings section of the report.
The writer is sharded with FSDP2/HSDP (shard over the 4 GPUs of a node, replicate across nodes).

Inputs
- report-generation index (sqlite, `samples` + `metadata` tables) and the study HDF5 shards (Findings text);
- classifier config + checkpoint (block 24, case list, rank assignment) and the staged node-local block-23 caches;
- frozen classifier logits of all studies (`.npz`) and the plan calibration JSON (`fora_report.plan.calibration_tiers`);
- finding store (`.npz`), evidence descriptions (`desc.npy`), node-local focus-token store, per-study measurement
  lines (`fora.reportgen.evidence`);
- DPO only: the merged pair store (`fora_report.pairs.merge`).

Outputs (results dir): `run_metadata.json`, `latest.pt`, `step_NNNNNN.pt`, `checkpoint_epochNNN.pt`
(full trainable tensors: LoRA, connector, vision tower; optimizer; scheduler; per-rank RNG), DPO: `dpo_metadata.json`.

Run (once per node, e.g. under `srun --ntasks-per-node=1`):

```bash
export NNODES=32 NODE_RANK=<node index> MASTER_ADDR=<host> MASTER_PORT=29500 GPUS_PER_NODE=4
export DENSE_NPZ=dense_logits.npz FINDING_NPZ=finding_store.npz LOCAL_CACHE=/local/rg DINO_CACHE_ROOT=/local/block23
COMMON="--index report_index.sqlite --h5_root h5/ --dino_mil_config mil_config.json --dino_mil_checkpoint mil.pt \
  --llm_path gemma-4-31B-it/ --finding_descriptions desc.npy --focus_tokens /local/focus_tokens \
  --study_measurements study_measurements.json"
# stage 1 (from scratch, 2 epochs, validation-split calibration)
bash fora_report/ct/run_node.sh configs/ct_sft1.json $COMMON --mil_calibration calibration_val_platt_bestf1.json \
  --results_dir runs/ct_sft1
# stage 2 (final plan tiers, 1,750 steps)
bash fora_report/ct/run_node.sh configs/ct_sft2.json $COMMON --mil_calibration calibration_train_platt.json \
  --warm_start runs/ct_sft1/checkpoint_epoch002.pt --results_dir runs/ct_sft2
# offline DPO (300 steps); PAIRS_SQLITE triggers the node-local pair filter
PAIRS_SQLITE=pairs_all.sqlite MIL_CONFIG=mil_config.json INDEX=report_index.sqlite \
bash fora_report/ct/run_node.sh configs/ct_dpo.json $COMMON --mil_calibration calibration_train_platt.json \
  --warm_start runs/ct_sft2/checkpoint_epoch001.pt --dpo_reference runs/ct_sft2/checkpoint_epoch001.pt \
  --results_dir runs/ct_dpo
```
The stage-1/2 evidence stores differ (stage 1 covers the plan items of its calibration, stage 2/DPO the final tiers).
A run resumes exactly from `results_dir/latest.pt`; SIGTERM/SIGUSR1 or `--stop_request_file` saves and exits.

Configs: `configs/ct_sft1.json` (2 epochs = 4,666 steps, connector LR 1e-4, LoRA 2e-5, vision 1e-5, warm-up 5 %,
possible tier uncapped), `configs/ct_sft2.json` (1,750 steps, 5e-5 / 1e-5 / 5e-6, warm-up 5 %, possible tier capped
at 20 by calibrated p), `configs/ct_dpo.json` (300 steps, beta 0.1, 2 pairs per study, chosen CE 1.0, unpaired CE 1.0,
warm-up 2 %, 5e-5 / 1e-5 / 5e-6). Weight decay 0.05, gradient clipping 1.0, seed 42 throughout.

DPO: sigmoid DPO on summed token log-probabilities, beta 0.1, frozen stage-2 reference held as tensors and swapped in
for reference scores, K = 2 pairs per study chosen by a seeded type-interleaving cycle, chosen-response CE weight 1,
CE weight 1 for studies without a usable pair, dropout disabled, every rejected score recomputed with gradient and
checked against its no-grad value.

Schedules:
- Stage 1 uses the stage-1 plan calibration (Platt on the validation split, per-class best-F1 confident cutoffs,
  recall-0.9 possible cutoffs, possible tier uncapped). Stage 2 and DPO use the final tiers.
- The stage-1 LR schedule is cosine over a 3-epoch horizon (warm-up 349 steps) up to optimizer step 1,358 and cosine
  over a 2-epoch horizon (same warm-up) afterwards, set by `initial_horizon_epochs` / `initial_horizon_until_step` in
  `configs/ct_sft1.json`.
- Stage 2 and DPO train on the first 1,750 / 300 studies of every rank in report-uid order (one pass each).
- 300 DPO steps x 128 ranks x 2 pairs gives about 77k pairs (fewer when a study has no usable pair).

## DPO pair synthesis (CT)

The DPO stage prefers the documented report (chosen) to a synthetic, minimally edited copy (rejected). All pairs come
from TRAIN reports only (the builder fails on any train/val overlap). Preferences are relative to the documented
Findings, not image truth. Eight pair types make up the 1,636,197-pair pool over 290,033 reports:

| type | count | rejected report |
|---|---|---|
| omission | 417,967 | one whole sentence with an asserted or uncertain extractor quote deleted (at most 3 per report) |
| repetition | 216,232 | one qualifying sentence repeated verbatim |
| explicit_denial | 46,409 | a unary whole-study denial replaced by an asserted sentence of the same class from another train patient |
| attribute | 417,428 | one attribute (laterality 109,870, location 66,509, size 72,668, certainty 110,906, severity 52,922, number 4,553) of one finding sentence corrupted by Gemma-4-31B-it, screened by a Gemma judge and a CPU verifier |
| insertion_plan_fp | 126,821 | an unreported item of the writer's plan (confident, or possible with the possible tier capped at 20 by p) asserted with a donor sentence from another train patient |
| insertion_random | 102,163 | a donor sentence of a class absent from the report, from an organ the report describes |
| normalised | 178,642 | 1..all abnormal sentences of one organ replaced by a frequent normal sentence of that organ |
| multi_omission | 130,535 | 2-4 qualifying abnormal sentences deleted |

43 incidental or degenerative classes (`configs/pairs_insertion_exclusions.json`) are never inserted, and each class
may supply at most 3% of the insertion_plan_fp pairs.

Inputs: the report index sqlite (`samples` table with per-class asserted/uncertain/denied lists and evidence quotes,
`metadata['global'].labels`), the H5 reports (Findings export), frozen-MIL dense logits (npz with `report_uid`,
`fora_id`, `label`, `logit`), the train Platt/Youden calibration json (`class_slope`, `class_intercept`,
`class_plan_threshold`, `class_possible_threshold`, `labels`), the patient split json, the label heads json and the
hierarchy json, and the class-name list for the Gemma prompts.

Output: `dpo_pairs.sqlite`, read by the DPO trainer:
```
pairs(pair_id TEXT PRIMARY KEY, report_uid TEXT NOT NULL, pair_type TEXT NOT NULL, attribute_type TEXT,
      chosen_sha256 TEXT NOT NULL, rejected_text TEXT NOT NULL)        INDEX pairs_uid ON pairs(report_uid)
metadata(key TEXT PRIMARY KEY, value TEXT)    'summary' = {"schema": "ct_report_preference_index_v1", "sources",
      "pairs_by_type", "reports", "target_prefix": "Findings:\n", "policy"}
```
The chosen text is not stored; the trainer rebuilds it from the report and checks `chosen_sha256`. `plan_pairs.sqlite`
additionally has `provenance(pair_id, record)` with the full JSON record of each plan-based pair.

Commands (`configs/pairs_ct.json` lists every argument of the paper run):
```bash
# 1 CPU: Findings export, base pairs + attribute candidates
python -m fora_report.pairs.base findings --index INDEX --h5_root H5 --output OUT/findings.sqlite --workers 24
python -m fora_report.pairs.base build --index INDEX --findings OUT/findings.sqlite --output_dir OUT
# 2 GPU (one process per GPU, r = 0..W-1): Gemma corruption, judge, filter
python -m fora_report.pairs.gemma_verifier corrupt --input OUT/attribute_candidates.jsonl.gz --class_names CLASSES \
    --out_dir OUT/gemma_attr --rank r --world W
python -m fora_report.pairs.gemma_verifier judge-input --candidates OUT/attribute_candidates.jsonl.gz \
    --gemma_dir OUT/gemma_attr --output OUT/judge_input.jsonl.gz
python -m fora_report.pairs.gemma_verifier judge --input OUT/judge_input.jsonl.gz --class_names CLASSES \
    --out_dir OUT/gemma_judge --rank r --world W
python -m fora_report.pairs.gemma_verifier filter --gemma_dir OUT/gemma_attr --judge_dir OUT/gemma_judge \
    --out_dir OUT/gemma_attr_judged
# 3 CPU: verify attribute edits, index, plan-based pairs + merge
python -m fora_report.pairs.attribute --candidates OUT/attribute_candidates.jsonl.gz \
    --gemma_outputs 'OUT/gemma_attr_judged/rank_*.jsonl' --findings OUT/findings.sqlite --output OUT/attribute_pairs.jsonl.gz
python -m fora_report.pairs.merge index --pairs OUT/base_pairs.jsonl.gz OUT/attribute_pairs.jsonl.gz --output OUT/base_all.sqlite
python -m fora_report.pairs.plan_pairs --index INDEX --findings OUT/findings.sqlite --dense DENSE \
    --calibration calibration_train_platt.json --split SPLIT --heads HEADS --hierarchy HIERARCHY \
    --base_store OUT/base_all.sqlite --out_dir OUT          # -> OUT/dpo_pairs.sqlite (~35 min CPU)
```
Gemma runs use vLLM, greedy decoding, bf16, max_model_len 16,384, 512 new tokens. Prompts:
`prompts/pairs_attribute_{corruption,verifier}_{system,user}.txt`. Of 655,736 attribute candidates, 462,954
corruptions passed the judge (all five criteria true) and 417,428 passed the CPU verifier.

## CT report decoding (`fora/reportgen/decoding/`, CLI `fora_report/decoding/`)

The trained CT writer (after DPO) generates one report per study. For each study the code encodes all block-24 tokens
into the 512-latent visual prefix, builds the diagnostic plan from the frozen classifier logits with its location and
measurement text, evidence focus tokens and zoomed evidence images, and runs one `generate` call. Decoding is nucleus
sampling with temperature 0.8, top-p 0.9 and no top-k. The torch seed is set per study, right before `generate`, to
`int.from_bytes(sha256("temperature_v1|<uid>|42")[:8], "big") % 2**31`. This makes each report independent of how
studies are spread over GPUs. Reports are capped at 2,048 new tokens.

For the external cohorts, the report opener is fixed and phrases that describe a prior examination are banned. The
opener tokens are appended to the prompt as embeddings, and the generation budget shrinks by the same number of
tokens. The ban uses HF `bad_words_ids`.

| set | config | opener | phrase ban |
|---|---|---|---|
| FORA val (all studies) | `configs/decode_ct_fora_val.json` | none | none |
| CT-RATE val (1,564) | `configs/decode_ct_ctrate.json` | `chest`: "Findings:\nThe trachea and both main bronchi" | `old_imaging` (1,491 studies); `prior_comparison` (the other 73) |
| Merlin test (5,125) | `configs/decode_ct_merlin.json` | `abdomen`: "Findings:\nThe contour, size," | `comparative` |

Ban lists (`fora/reportgen/decoding/ban_lists/`):
- `prior_comparison_phrases.json`: prior-examination and comparison phrases (626 token sequences).
- `prior_comparison_phrases_old_imaging.json`: the first list plus old-imaging and external-centre phrases (702).
- `prior_comparison_phrases_comparative.json`: the second list plus comparative-examination and size-change phrases (734).

Each phrase is banned in four forms: with or without a leading space, and with the first letter upper or lower case.
The lists also ban 30 report openers directly after "Findings:\n" and the years 1990-2029 as digit-token sequences.

Inputs:
- the DPO writer checkpoint and the Gemma-4-31B-it snapshot;
- the MIL config and the block-24 checkpoint;
- the dense MIL logits cache;
- the evidence store (descriptions, focus tokens, zoomed images, study measurement line);
- the plan calibration (train Platt + Youden tiers);
- a study manifest (jsonl, see `fora_report/decoding/cohorts.py`).

Outputs: `OUTPUT/records_rank<shard>.jsonl`, plus `predictions.jsonl` and `summary.json` after the merge.

```bash
# study manifests
python -m fora_report.decoding.cohorts fora   --index INDEX.sqlite --mil-config MIL.json --h5-root H5 --cache-root CACHE --output fora_val.jsonl
python -m fora_report.decoding.cohorts ctrate --cohort-json CTRATE_COHORT.json --cases CASES.jsonl --output ctrate_val.jsonl
python -m fora_report.decoding.cohorts merlin --reports-csv REPORTS.csv --protocol-csv PROTOCOLS.csv --cases CASES.jsonl --output merlin_test.jsonl
# one process per GPU (shard = RANK, GPU = LOCAL_RANK)
torchrun --nproc-per-node 4 -m fora_report.decoding.run --config configs/decode_ct_ctrate.json \
  --checkpoint DPO.pt --llm-path GEMMA4 --manifest ctrate_val.jsonl --cache-root CACHE --mil-config MIL.json \
  --dense DENSE --finding-store LOC --descriptions DESC.npy --ftok FTOK --study-meas STUDY_MEAS.json \
  --calibration CAL.json --output OUT
# the 73 CT-RATE studies decoded with the prior_comparison list
torchrun ... --ban-list prior_comparison --only-uids uids73.json --output OUT_73
python -m fora_report.decoding.run --config configs/decode_ct_ctrate.json --manifest ctrate_val.jsonl --output OUT --merge-only --num-shards 4
# check that the shipped ban lists rebuild exactly from their phrases
python -m fora.reportgen.decoding.bans verify --tokenizer google/gemma-4-31B-it prior_comparison old_imaging comparative
```

## Chest X-ray writer (`fora/reportgen/cxr/`, `fora_report/cxr/`)

The X-ray writer uses the same model as CT (`fora.reportgen.writer.dino_stream.DinoCTReportGenerator`: Gemma-4-31B-it,
LoRA r16/a32 on the language model's attention and MLP projections, 512-latent resampler, study-type token,
frozen-Platt label tokens, one attention-pooled evidence token per planned finding). The visual tokens are the frozen
X-ray MIL stream (DINO teacher blocks 1-23 + MIL block 24, 1048-d, all films of the study; raw MIL weights). There
are no zoomed evidence images and no measurements. Plan items carry location text (side and lung zone) computed from
positive leave-one-out MIL evidence on frontal films (`fora.reportgen.cxr.location`). Training is SFT -> DPO -> PU-DPO, and
decoding is greedy with two plan rules: device classes become probable at the train Youden cutoff, and possible items
of high-AUROC classes or devices are promoted to probable, at most 8 per study.

Inputs: X-ray H5 studies and study lists (data stage), train labels npz and label sqlite (labels stage), X-ray MIL
config + checkpoint and its study-bag predictions on train/val (classifier stage), Gemma-4-31B-it snapshot.

| step | command (torchrun = `torchrun --nnodes N --nproc-per-node 4 ... -m`) | output |
|---|---|---|
| 1 calibration | `python -m fora_report.cxr.calibration --preds-train P_tr.npz --preds-val P_va.npz --labels labels.npz --cases cases_train.jsonl cases_val.jsonl --mil-checkpoint mil.pt --out calibration_train_platt.json --dense dense_predictions.npz` | plan tiers + dense logits |
| 2 locations | `torchrun fora_report.cxr.dump_locations --cases cases_{train,val}.jsonl --out locations_{train,val}.npz --h5_root H5 --mil_config C --mil_checkpoint M` | per (study, class) LOO evidence stats |
| 3 SFT | `torchrun fora_report.cxr.train_sft --calibration ... --dense ... --mil_config C --mil_checkpoint M --labels labels.npz --cases_train cases_train.jsonl --h5_root H5 --label_db labels.sqlite --locations locations_train.npz,locations_val.npz --llm_path GEMMA --results_dir sft --epochs 2 --save_every_steps 0 --save_every_minutes 60` | `checkpoint_epoch002.pt` |
| 4 DPO pairs | `python -m fora_report.cxr.dpo_pairs --label_db ... --cases cases_train.jsonl --h5_root H5 --calibration ... --out pairs_dpo.sqlite` | 202,000 synthetic pairs |
| 5 DPO | `torchrun fora_report.cxr.train_dpo <SFT args> --dpo_pairs pairs_dpo.sqlite --sft sft/checkpoint_epoch002.pt --results_dir dpo --max_steps 600` | `step_000200.pt` used |
| 6 PU-DPO data | `pudpo_subset` -> `torchrun fora_report.cxr.generate --no-plan_rules --checkpoint dpo/step_000200.pt --cases cases_pudpo.jsonl` -> `pudpo_pairs gen_csv` -> labels-stage X-ray extractor -> `pudpo_pairs requests` -> `pudpo_edit` (one process per GPU, vLLM) -> `pudpo_pairs label_csv` -> extractor -> `pudpo_pairs finalize` | `pairs_pudpo.sqlite` |
| 7 PU-DPO | `torchrun fora_report.cxr.train_pudpo <SFT args> --dpo_pairs pairs_pudpo.sqlite --sft dpo/step_000200.pt --results_dir pudpo --max_steps 600` | `step_000200.pt` used |
| 8 generation | `torchrun fora_report.cxr.generate --checkpoint pudpo/step_000200.pt --cases cases_val.jsonl --h5_root H5 --llm_path GEMMA --out gen.jsonl` | Findings JSONL |

Configs: `configs/cxr_calibration.json`, `cxr_sft.json`, `cxr_dpo_pairs.json`, `cxr_dpo.json`, `cxr_pudpo.json`,
`cxr_infer.json`; editor prompt `prompts/cxr_pudpo_editor.txt`; promotion list `fora/reportgen/cxr/highauc_classes.json`.
The epoch schedules depend on the world size: SFT and DPO run on 128 GPUs, PU-DPO on 40 GPUs.
`generate` reads calibration / dense / MIL / location paths from the checkpoint metadata; override them with
`--calibration --dense --mil_config --mil_checkpoint --label_db --locations` when the files are elsewhere.

Procedure:
- The X-ray SFT starts from the base Gemma-4-31B-it.
- DPO and PU-DPO each run 600 steps; the released model uses step 200 of each.
- The plan rules (device rule, high-AUROC promotion) are applied only at generation; training uses the plain plan.
- PU-DPO base reports are generated without the plan rules.
