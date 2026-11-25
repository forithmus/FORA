# Evaluation

Scoring code for the paper's finding recognition (Methods 4.9), localization (Methods 4.10) and report generation
(Methods 4.11). Every model, FORA and the comparison models alike, is scored by the same code from prediction files
in the formats below. Running the comparison models themselves is not part of this folder.

All modules run on CPU. Install the repository (`pip install -e .`) and call the modules with `python -m`.

```
evaluation/
  classification/   metrics, training Youden cut-offs, held-out cells, calibration, subgroups
  benchmarks/       CT-RATE, Merlin, RAD-ChestCT, KiTS23, RibFrac readers and scoring; X-ray benchmarks
  localization/     CT evidence regions, hit rate, evidence overlap, chance, equal-size comparison; X-ray boxes
  report_generation/ finding-level scoring of generated reports, X-ray label resolution, LLM judge
  statistics/       linearization intervals, bootstraps, Holm correction, exact binomial intervals
  tests/            CPU unit tests
```

## File formats

| file | content |
|---|---|
| predictions `.npz` | `case_ids` (str), `findings` (str), `scores` (float, n x K), optional `decision` (bool, n x K) |
| held-out labels `.npz` | `case_ids`, `findings`, `labels` (int8: 1 asserted or uncertain, 0 absent, -1 unknown), `exam_type` (12 CT types, from `data-preprocessing/exam_types.py`), `scan_type` (cut-off group), optional `patient_ids` |
| training predictions `.npz` (cut-offs) | the held-out label fields plus `scores` for the training split |
| benchmark truth `.npz` | `case_ids`, `patient_ids` (resampling clusters), `findings`, `labels` (`benchmarks/ct_readers.py`) |
| X-ray truth `labels.json` | `{names, ids, Y}` (`benchmarks/xray_readers.py`) |
| CT evidence `<case>.npz` | per finding either evidence points (`e_j`, `world_j` in S, R, A mm) or regions (`cells_j`, `score_j`, `core_j`, `hdr80_j`, `focus_j`) |
| report labels `.jsonl` | `{source: "ref" or "gen:<model>", uid, positive_classes: [...]}` |
| judge items `.jsonl` | `{item_id, kind, dataset, modality, region, uid, model, ref, gen}` and pairwise `{..., a_model, b_model, a, b}` |

## Finding recognition (Methods 4.9)

Cut-offs. FORA decides with the training Youden cut-off per finding and scan type (chest, abdomen, head and neck,
chest-abdomen-pelvis, other; fallback to all training studies when a scan type has fewer than 50 positives or 50
negatives). Benchmark findings that map to several FORA classes use the maximum member logit, and their cut-off is
fitted on the scan types of `benchmarks/configs/ct_mappings.json`.

```
python -m evaluation.classification.thresholds ct --train fora_train.npz \
    --mappings evaluation/benchmarks/configs/ct_mappings.json --out youden_cutoffs.json
```

Held-out set. AUROC, AUPRC and balanced accuracy are averaged over (finding, CT type) cells with at least 10
positives and 50 negatives, and CRG pools every decision. The 12 CT types are the 11 examination types plus
"mixed": a study takes the type whose protocol classes contain all of its protocol classes, otherwise it is mixed
(`data-preprocessing/exam_types.py`). On the FORA validation set this gives 2,132 cells over 854 findings. FORA's
decisions use the cut-off of the study's scan type (chest, abdomen, head and neck, chest-abdomen-pelvis, other).
Other models use their native decisions.

```
python -m evaluation.classification.heldout --labels val_labels.npz --pred FORA=fora_val.npz \
    --cutoffs FORA=youden_cutoffs.json --pred NV-Reason-CT=nv_val.npz --out heldout/ --ci --draws 200000
python -m evaluation.classification.calibration --pred fora_val.npz --labels val_labels.npz --platt platt.json --out calibration/
python -m evaluation.classification.subgroups --pred fora_val.npz --labels val_labels.npz --cutoffs youden_cutoffs.json \
    --meta val_meta.csv --out subgroups.json
```

Calibration uses findings with at least 10 validation positives and the expected calibration error over 15
equal-count bins. Subgroups take the macro over cells with at least 10 positives and 50 negatives in every level.

CT benchmarks. Truth files are built from the public label files, then every model is scored on the full finding
set (18 CT-RATE findings on 1,564 study bags, 30 Merlin findings on 5,125 test studies, 16 RAD-ChestCT findings on
3,630 scans, KiTS23 renal cyst on 489 scans, RibFrac rib fracture on 80 scans).

```
python -m evaluation.benchmarks.ct_readers ctrate --input valid_predicted_labels.csv --out ctrate_truth.npz
python -m evaluation.benchmarks.ct_readers merlin --input zero_shot_findings_disease_cls.csv --ids merlin_test_ids.txt --out merlin_truth.npz
python -m evaluation.benchmarks.ct_readers radchestct --input inferred_external_{train,valid,test}.csv --out rad_truth.npz
python -m evaluation.benchmarks.ct_readers kits23 --input kits23/dataset --out kits_truth.npz
python -m evaluation.benchmarks.ct_readers ribfrac --input ribfrac-val-info.csv --out ribfrac_truth.npz
python -m evaluation.benchmarks.ct_score --truth ctrate_truth.npz --dataset CT-RATE --pred FORA=fora_ctrate.npz \
    --fora-cutoffs youden_cutoffs.json --pred NV-Reason-CT=nv.npz --native-zero NV-Reason-CT --draws 200000 --out ctrate.json
python -m evaluation.benchmarks.ct_score --truth kits_truth.npz --dataset KiTS23 ... --draws 2000 --test normal --out kits.json
```

CT-RATE resamples patients, the other sets resample studies or cases.

X-ray benchmarks. ChestX-ray14, ChestX-Det10, RSNA Pneumonia, PadChest-GR (24 findings), Open-I (study bag score on
every image) and the internal validation set; mappings in `benchmarks/configs/xray_mappings.json`.

```
python -m evaluation.benchmarks.xray_score --truth cxr14_labels.json --mapping chestxray14 \
    --fora fora_cxr14.npz --fora-train fora_train.npz --pred CARZero=carzero.npz --out cxr14.json
```

## Localization (Methods 4.10)

CT. Evidence is reduced to 3 x 8 x 8 mm world cells (maximum per cell), smoothed with a 4 mm Gaussian, and the
smallest cell sets holding 50% and 80% of the smoothed mass give the evidence regions. The top-focus hit uses the
centroid of the highest-mass 26-connected focus. Chance is the probability that a uniformly drawn token centre in
the finding's route lies inside the annotation. Hit-rate intervals are exact binomial intervals. For comparisons
with a model whose regions are much larger, `--match-size` keeps as many of its highest cells as FORA's 80% region
of the same instance and counts cells tied at the cut-off at their expected value.

```
python -m evaluation.localization.score_ct --cohort kits23 --annotations kits23/dataset --maps fora_maps/ --out loc_kits/
python -m evaluation.localization.score_ct --cohort kits23 --annotations kits23/dataset --maps other_maps/ \
    --match-size loc_kits/per_instance.jsonl --out loc_kits_other/
python -m evaluation.localization.score_ct --cohort rexg --annotations dataset.json --segmentations segmentations/ --maps fora_maps/ --out loc_rexg/
```

Dense maps of other models are converted with `ct_regions.dense_map_to_cells` and stored in the points form.
ReXGroundingCT intervals in the paper come from study-bootstrap resamples (`statistics.bootstrap.bootstrap_ci`
with the study as cluster).

X-ray. `localization/xray.py` scores the peak of the highest-mass focus inside any box of the finding and the
region Dice, averaged over instances per finding and then over findings, with 2,000 image resamples.

## Report generation (Methods 4.11)

Labels. Run the report extractor of `labels/report_scoring` on the reference Findings and on every generated
Findings text, with the same vocabulary, and write the report label file (`source`, `uid`, `positive_classes`).
For X-ray the extractor CSV is resolved here (`report_generation/xray_labels.py`: vocabulary aliases, ancestor
credit, roll-up of unscored subtypes, sternal wire rule, synonym groups from `report_generation/configs/`).

```
python -m evaluation.report_generation.finding_metrics --studies ctrate_studies.jsonl --labels ct_labels.jsonl \
    --classes ct_classes.json --first FORA --paired-draws 2000 --out rg_ctrate.json
python -m evaluation.report_generation.xray_labels --csv 'labels_*.csv' --vocabulary cxr_vocabulary.json \
    --classes cxr_classes.json --studies val_studies.jsonl --first FORA --paired-draws 2000 --out rg_xray_val.json
```

Macro recall, specificity, balanced accuracy and precision are averaged over findings with at least 10 reference
positives and CRG pools the same findings. Intervals: 1,000 resamples of patients on CT-RATE and of studies
elsewhere; paired tests from 2,000 resamples with Holm correction over the comparisons of one call (one figure panel).

LLM judge. Gemma-4-31B-it compares each generated report with the reference on abnormal findings. The score of a
report is matched / (matched + significant errors), and a report with no matched finding scores 1 when it has no
significant error and 0 otherwise. The pairwise prompt is run in both presentation orders.

```
python -m evaluation.report_generation.judge.build_items --dataset ct_ctrate --modality CT --region chest \
    --references refs.jsonl --gens FORA=fora.jsonl --gens NV-Reason-CT=nv.jsonl --ours FORA --opponent NV-Reason-CT --out judge_in/
python -m evaluation.report_generation.judge.run_vllm --model google/gemma-4-31B-it --items judge_in/items.jsonl \
    --items judge_in/pairs.jsonl --shard 0 --nshards 8 --out judge_out/
python -m evaluation.report_generation.judge.metrics --items judge_in/items.jsonl --pairs judge_in/pairs.jsonl \
    --outputs 'judge_out/shard_*.jsonl' --ours FORA --out judge.json
```

## Statistics (Methods 4.9-4.11)

`statistics/` implements the procedures described in Methods 4.9-4.11:

* `linearization.py`: first-order linearization of AUROC, AUPRC, balanced accuracy, sensitivity, specificity,
  precision and pooled CRG (exact weight derivatives), macro means over cells, patient-level Gaussian multiplier
  draws (200,000 by default), paired differences on shared draws and two-sided paired p-values.
* `bootstrap.py`: paired cluster bootstrap (patients or studies), case bootstrap with a two-sided normal
  approximation test, percentile intervals, vectorized weighted AUROC and balanced accuracy.
* `multiple_testing.py`: Holm correction and exact binomial (Clopper-Pearson) intervals.

Defaults: Gaussian multipliers, one multiplier per patient, 200,000 draws, seed 0, percentile intervals of the
draws, macro means linearized as the mean of the per-cell influence values, paired p-value
2 x min(P(D* <= 0), P(D* >= 0)) on shared draws. X-ray classification intervals use 1,000 case resamples
(`benchmarks/xray_score.py`), report-generation and judge intervals use 1,000 study or patient resamples.

## Tests

```
python -m pytest evaluation/tests
```

The tests compare the metrics with scikit-learn, check the influence functions against finite differences, check
interval coverage on synthetic data, Holm on a hand example, exact binomial intervals against SciPy, and the
localization measures on toy grids.
