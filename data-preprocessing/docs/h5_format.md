# Data formats

## CT H5 shards (`batch_NNNNN.h5`)

One shard holds about 50 patients; a patient never spans two shards. Group layout:

```
/<PatientUID>/<ReportUID>/<AccessionNumber>/<series>/
    data                (X, Y, Z) or (X, Y, Z, T)   int16 / int32 (smallest lossless integer type), float32 otherwise; HU; lzf
    affine              (4, 4) float64              NIfTI voxel -> world (RAS mm) affine
    header              NIfTI header binary block
    pacs_metadata       JSON string                 curated DICOM attributes of the series (PACS export)
    dcm2niix_metadata   JSON string                 the dcm2niix sidecar of the series
    derivative_<tag>/   data, affine, ...           dcm2niix derivatives: Tilt_Eq_1, Tilt_1, Eq_1, ROI1, ...
    attrs: derivatives (list of tags), segmentation_status, classified_protocols (JSON list of protocol titles of the study)
/<PatientUID>/<ReportUID>/report/
    structured, extracted_sentences, there_is_statements      JSON strings (the processed report)
    attrs: kind = "report", ReportUID
root attrs: enrichment_done
```

Series names carry the acquisition date and time (`..._YYYYMMDDhhmmss_...`). Window settings are read from
`WindowCenter` / `WindowWidth` in `pacs_metadata`, else in `dcm2niix_metadata`.

The shard-level split is a CSV with columns `h5_id` (shard name without `.h5`) and `split` (`train` / `val`).

## CADS masks (`cads_batch_NNNNN.h5`)

One file per CT shard, at the same group path as the CT series. Per series group: `part_551` ... `part_559`
(uint8 label maps of the nine CADS models, in the array order of the CT `data`; an absent part is all zeros),
`image_affine` and the attribute `image_shape`. Label ids follow each CADS model's `dataset.json`. Series without a mask
are listed in the root attribute `excluded_series`.

## Extraction plan (`reports_{train,val}.jsonl`, `assign_{train,val}.json`)

One JSON line per report: `id` (`train_NNNNNN` / `val_NNNNNN`), `h5`, `patient`, `report`, `split` and `series[]` with
`acc`, `sid`, `sub` (derivative subgroup or ""), `frame` (null or the 4-D timepoint), `windows` (list of [lo, hi], FULL
first), `mip` ([lo, hi] or null), `cost`, `protocols`. The assignment files hold `workers`, `assignments` (report ids per
worker) and `load`.

## Block-23 cache series file (`fora_int8_tokens`)

One H5 per (report, series), named `<split>/<report id>/series_NNN.h5`:

```
attrs: identity (sha256 of fingerprint + report id + plan entry), fingerprint, format, provenance_json, complete
flat_indices   [N] int64       body tokens of the series in (gz, gy, gx) grid order (shared by all views)
vNN/           one group per view, in plan order: v00 FULL, then the native windows, then the lung MIP
    prefix       [K, 5, 1024]  uint16   bf16 bits of CLS + 4 registers per kept crop
    q            [N, 1024]     int8     body tokens, crop-major order
    scale        [K, 1024]     float16  per crop and channel absmax / 127
    outlier_idx  [8]           int16    the view's 8 channels with the largest |x|
    outlier      [N, 8]        uint16   bf16 bits of those channels
    crop_ids     [K]           int32    z-major id of each kept 64 x 192 x 192 crop
    attrs: metadata_json (grid, volume_shape, n_grid_tokens, n_tokens, crops, kept_crops, seconds),
           window_json (kind: full | native | lung_mip_10mm, lo, hi), complete
```

`provenance_json` holds `h5`, `patient`, `report`, `acc`, `sid`, `sub`, `frame`, `affine`, `orig_shape`,
`body_crop_yxyx` ([y0, y1, x0, x1] of the in-plane body crop in the resampled frame), `resampled_body_shape` and
`target_spacing`. Series files are appended to per-worker packs `pack_worker_NNN.bin`; the journal
`index_worker_NNN.json` lists every pack entry (`file`, `offset`, `file_size`, `identity`) and every completed report.

## Study bags (`cases.jsonl`, `assign.json`)

`cases.jsonl` has one line per report (study bag): `id`, `split`, `report`, `fingerprint`, `complete`,
`series[]` (`index`, `acc`, `sid`, `relative_file`, `file`, `offset`, `file_size`, `identity`) and `views[]` (the same
pack fields plus `series`, `group`, `n_tokens`, `n_crops`, `grid`, `window`). `assign.json` holds `fingerprint`,
`ranks`, `ranks_per_node` and `assignments` = {train, val: report ids per rank}.

## Split (`split.json`)

{report id: `train` | `val`} at patient level, with every CT-RATE validation study in `val` (`make_split.py`).

## Region bits

Per series `<local>/regions/<relative_file>.npz` with `bits` (uint64 [N], token order of `flat_indices`); bit i =
`fora.regions.REGION_NAMES[i]`. Pack files hold `bits` (all series concatenated), `offsets` [files + 1],
`relative_files`, `cads_source` (own / borrowed / none), `info_json` and the attribute `region_names`.
