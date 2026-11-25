# Steps not included in this repository

1. **DICOM to NIfTI conversion.** The CT series were converted with dcm2niix before ingestion (the shards keep the
   dcm2niix sidecar and its derivatives `Tilt_Eq_1`, `Tilt_1`, `Eq_1`, `ROI*`). The conversion script, the dcm2niix
   version and its flags are not available.
2. **Ingestion into H5 shards and enrichment.** The NIfTI tree, PACS metadata and reports were joined into a manifest
   (series deduplication, non-CT and PET/CT removal), written into `batch_NNNNN.h5` (50 patients per shard) and
   enriched with the report group and the `classified_protocols` attribute. This code depends on the hospital PACS and
   EHR exports and is not released; the resulting format is described in [docs/h5_format.md](docs/h5_format.md).
3. **Report anonymization, translation and structuring.** Reports were pseudonymized, anonymized, translated from
   Turkish to English and structured into five sections with Gemma-4-31B-it. These steps need the private
   re-identification table and are not released.
4. **The shard-level split file** (`splits.csv`, shard -> train / val; 4,361 / 531 shards). The code that created it
   is not available. `make_split.py` starts from it.
5. **CADS mask production.** The nine CADS models were run on every in-scope series and four post-processing rules
   were applied (see README, "Differences from the paper text"). The production scripts for these masks are not
   available; only the mask format is documented. At inference CADS is run by the inference stage.
6. **Protocol-classification job wrappers.** `classify.py` reads its rank from `SLURM_PROCID` / `SLURM_NTASKS`; the
   cluster submission scripts and their exact arguments are not included.
7. **The report order of the label set.** `make_split.py` draws the validation patients moved back to training in a
   seeded random order that depends on the order of the input reports (`--labeled`). The released split file should be
   used to reproduce the paper split exactly.

# Known data issues

- Without hip bones in the field of view the `pelvis` region degenerates; the MIL trainer and inference clear the
  `pelvis` bit for such series (`fora.regions.clear_pelvis_without_hips`).
- 614 studies whose CADS masks were mapped through world coordinates from their own parent series have
  misplaced region bits.
