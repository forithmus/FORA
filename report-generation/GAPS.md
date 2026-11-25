# Known gaps

Steps of the paper procedure whose code or intermediate data is not available in this repository.

## Plan
- The calibration used by SFT stage 1 (Platt on the validation split, per-class best-F1 confident cutoff, recall-0.9
  possible cutoff) has no builder here. Stage 1 can only be reproduced with the released stage-1 calibration file.
- In the paper run the evidence targets mapped report uids to case ids through a separate case-map file. Here this is
  the optional `--case-map` argument of `fora_report.plan.targets`; the default reads `fora_id` from the dense logits.

## Evidence tokens
- The focus-token store used to train SFT stage 2 and DPO (training studies, final plan tiers) was not kept. It must be
  rebuilt with `fora.reportgen.evidence.extract`, `focus_tokens` and `merge_store tokens` (about 2.5 h on 32 GPU nodes)
  before stage 2 or DPO can be rerun. The store for the stage-1 plan and the validation store exist as artifacts.
- The global finding stores and description arrays are data artifacts and are not part of the repository.
- The per-study CT volumes read by the inference renders are written by the inference stage; only the reader is here.

## Measurements
- Only the final lesion-caliper stage (2-D decoder on the axial slice) can be trained here. The training code of the
  class-conditioned 3-D mask head, its earlier warm-start stages and the builders of its inputs (measurement targets,
  out-of-fold candidates with size priors, report information tables) are not included. Retraining needs `--init`
  pointing at a released group checkpoint.
- The per-group scale-calibration files in `data/measure/` are the released copies. The files written by the original
  fit are lost, and `fora.reportgen.measure.scale_calibration` was not re-run against them.
- The training-time candidate extraction (a separate MIL streaming pass that wrote per-rank peak shards) is not
  included. At inference the evidence extractor's extent files are used instead; the two were not verified to give
  identical candidates.
- The study measurement line of the training cohort merged three CADS measurement runs; the original merge script is
  lost. `fora.reportgen.measure.cads.study` takes any list of raw record files with per-task override.
- The step that combined lesion texts and Evans-index / cardiothoracic-ratio texts into the training measurement file
  was not found. The data match the lesion texts plus the study-level finding texts with no overlapping keys, which
  `fora.reportgen.measure.merge --extra` reproduces.
- About 477k planned findings added by the final plan tiers have no size in the training text because the measurement
  pass was not re-run for them.

## CT writer training
- The vision-tower learning rate of the DPO run is not recorded; `configs/ct_dpo.json` uses 5e-6, the value of SFT
  stage 2.
- The full FSDP training step has no CPU test (it needs CUDA, NCCL and the Gemma-4 weights).

## DPO pair synthesis
- The step that turned the Gemma judge outputs into the accepted attribute corruptions is lost. `gemma_verifier filter`
  keeps a corruption when all five judge criteria are true; on the archived judge outputs (650,466 judgements) this
  reproduces the 462,954 accepted corruptions exactly.
- The pair counts in `configs/pairs_ct.json` are those of the paper run; the pool was not regenerated with this code.
- Gemma generations depend on the vLLM version and model snapshot, so a rerun of the corruption and judge steps can
  produce different edits.

## CT decoding
- The phrase-ban lists ship as data. The selection of their phrases measured candidates on private generated reports
  (dropping any candidate that fired on more than 0.5 % of single-examination reports) and the additions of the second
  and third list were made by hand; neither step is included. `fora.reportgen.decoding.bans verify` rebuilds the token
  sequences of the shipped lists exactly from their phrase fields.
- Evidence stores, dense logits and calibration files of the external cohorts are built by the evidence and evaluation
  stages.

## Cache layout
- The writer trainer, the batch decoding CLI and `fora_report.ct.dpo_pairs_local` read the block-23 cache from
  `<cache root>/<fingerprint>/` and find the node's classifier data ranks from `STAGED_node<n>.json` markers; the
  data stage (`stage_local.py`) writes the staged files to `<local>/` with one `STAGED.json`. Point `--cache-root` /
  `--dino_cache_root` at a directory with that layout (or pass `--data_node`).

## Chest X-ray writer
- The X-ray label extractor that labels base and edited reports for PU-DPO belongs to the labels stage; no launcher is
  included here. `fora_report.cxr.pudpo_pairs` reads and writes its CSV format.
- PU-DPO pairs cannot be reproduced bit-exactly: the editor is an LLM, and the order of the edit requests differs from
  the paper run.
- The DPO and PU-DPO trainers check that the data paths recorded in the start checkpoint equal the CLI paths; released
  checkpoints need the same path strings or a metadata rewrite.
