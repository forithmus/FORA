# Data preprocessing

This stage turns the CT H5 shards into the training inputs of FORA (Methods, "Data" and "CT preprocessing"): protocol
classes per study, an extraction plan over every usable series of every report, a cache of frozen-encoder block-23
tokens for every view of every series, per-token CADS region bits for organ routing, study-bag manifests and the
patient-level split. The views that MIL and sentence alignment train on are these cached tokens, not voxels.
It also documents the public chest X-ray release and how films are read (`fora.geometry.xray`).

Shared code lives in the package `fora`:

| module | content |
|---|---|
| `fora.geometry.tiling` | RAS reorientation, trilinear resampling to 1.5 x 0.5 x 0.5 mm, in-plane body crop (HU > -300) |
| `fora.geometry.windows` | native windows from DICOM metadata, window normalization, 10 mm lung slab MIP, the view rule |
| `fora.geometry.series` | usable series in the H5 shards (derivative preference, >= 16 slices, 4-D frames, sane affine) |
| `fora.geometry.tokens` | (2, 16, 16) token grid, body tokens (token mean HU > -500, hole-filled per slice) |
| `fora.geometry.protocols` | model-side protocol ids and the lung-CT protocol set |
| `fora.geometry.world` | world coordinates of cached tokens |
| `fora.geometry.xray` | X-ray content crop and [0.5, 99.5] percentile normalization; public-release image reader |
| `fora.cache` | block-23 cache format (int8 body tokens + 8 exact outlier channels) and the encoder pass |
| `fora.regions` | the 63 CADS regions, token-region bits (>= 25 % coverage), class-region bits, anatomical families |

File formats are described in [docs/h5_format.md](docs/h5_format.md).

## Steps

All scripts take `--help`. Paths below are placeholders.

### 1. Protocol classes

Each CT study is classified from the metadata of all its series into a list of protocol classes
(`protocol_classification/protocol_classes.json`: 29 configured classes plus `other`) by Gemma-4-31B-it on vLLM with
greedy decoding and one QC pass. The post-processing merges "CT Liver - Four-Phase" into "CT Abdomen and Pelvis -
Multiphase", which leaves 28 classes.

```bash
cd protocol_classification
python prepare_metadata.py -i FILTERED_METADATA_DIR -o classification_metadata.csv
python classify.py --input classification_metadata.csv --output-dir shards --log-dir logs     # one process per GPU,
                                                                                               # ranks from SLURM_PROCID / SLURM_NTASKS
python merge_shards.py --input classification_metadata.csv --shard-dir shards --output classified_protocols.csv
python post_process.py --input classified_protocols.csv --output classified_protocols_post_processed.csv
```

The study labels are written into the H5 series attribute `classified_protocols` and the ingestion manifest. Then

```bash
python report_protocol_classes.py --manifest ingestion_manifest.csv --reports structured_reports.csv --output report_protocols.csv
python exam_types.py --report-protocols report_protocols.csv --output exam_types.csv
```

give one row per (report, protocol class) and the 11 merged exam types (plus `mixed`) used in the dataset tables.

### 2. Extraction plan

```bash
python plan_extraction.py --h5-dir H5 --splits splits.csv --out plan --workers 128
python make_manifest.py --plan plan --checkpoint TEACHER.pt --store STORE --out manifest.json
```

Every report with a report group, every usable series (one per timepoint of a 4-D series) and its views: FULL
[-1000, 1000] HU, every distinct native DICOM window, and a 10 mm slab MIP in the deepest-centre native lung window.
`configs/encoding.json` holds the encoding settings (64 x 192 x 192 crops, stride = crop, encoder step 5,000,
brick batch 4).

**Lung-MIP rule.** The training cache adds the lung MIP whenever a series has a native window with centre below
-250 HU, without a protocol gate (`--mip-rule training`, the default). The inference bundle additionally requires a
lung-CT protocol for the study (`fora.geometry.protocols.LUNG_CT_CLASSES`); `--mip-rule inference` reproduces that rule
(`fora.geometry.windows.plan_views(..., mip_rule="inference")`). A non-lung study with a lung window therefore has one
view more in training than at inference.

### 3. Block-23 cache

```bash
python extract_block23.py --manifest manifest.json --h5-dir H5 --worker W --local-dir /local/fora_cache   # W = 0..127
```

One process per GPU (the paper data used 128 workers on 32 nodes with 4 GPUs). Each series is resampled, body-cropped and
encoded for all its views by the frozen teacher (blocks 1-23); body tokens are stored as int8 with per-crop scales and
8 exact outlier channels. Runs resume from the per-worker journal. Worker 0 also writes `block24_state.pt`
(block 24 + final norm of the teacher), the initialization of the trainable block 24 downstream.

### 4. Study bags

```bash
python make_study_bags.py --manifest manifest.json --store STORE --out bags          # cases.jsonl + assign.json
```

### 5. Split

```bash
python match_ctrate.py --h5-dir H5 --plan plan --ctrate-metadata validation_metadata.csv --out ctrate_match
python make_split.py --cases bags/cases.jsonl --labeled labeled_ids.txt --report-protocols report_protocols.csv \
    --match ctrate_match/match.json --out split
```

All reports of patients with a study in the CT-RATE validation set go to validation; the same number of reports moves
back to training as whole validation patients with only non-contrast chest CTs (seed 0). Training split: 304,962
reports, 856,181 series, 2,332,070 views; validation: 36,619 reports, 102,186 series, 275,856 views.

### 6. Node-local staging and region bits

```bash
python stage_local.py --cases bags/cases.jsonl --assign bags/assign.json --ranks 0-3 --local /local/fora_cache/FINGERPRINT
python token_regions.py compute --cases bags/cases.jsonl --assign bags/assign.json --ranks 0-3 \
    --local /local/fora_cache/FINGERPRINT --h5-dir H5 --cads-dir CADS --procs 72
python token_regions.py pack   --cases bags/cases.jsonl --assign bags/assign.json --ranks 0-3 \
    --local /local/fora_cache/FINGERPRINT --out regions_node_00.h5
python token_regions.py unpack --pack regions_node_00.h5 --local /local/fora_cache/FINGERPRINT
```

Staging copies the series files of one node's ranks (4 ranks per node) from the packs, capped at 700 MB/s. Region bits
are computed once and kept as one pack per node; each training allocation unpacks them next to the staged series.
A token is in a region if at least 25 % of its volume is covered on a 1.5 x 2 x 2 mm work grid. Series without their
own CADS mask use the mask of the same-accession series with the best coverage (>= 10 %) through world coordinates;
series without any CADS mask only get the `body` bits.

## Chest X-ray data

The X-ray models were trained on X-ray H5 shards with the same group layout as the CT shards (one 2-D uint16 `data`
per film and a `dcm2niix_metadata` JSON). Films are read at native resolution, cropped to their content box and
stretched once from the [0.5, 99.5] percentile range to [-1, 1] (`fora.geometry.xray.prepare_film`).

The public release is on Hugging Face as `Forithmus/FORA` (gated): lossless JPEG-LS 16-bit images (higher values are
brighter) in per-batch zip archives (`images/batchNN/batchNN_partK.zip`, one folder per study), `metadata/batchNN_metadata.csv`
(one row per image; `file` = `<zip path>::<member>`), `reports/batchNN_reports.csv` (one row per study), `labels/` and
`splits.csv` (patient-level). A film from the release goes through the same preparation:

```python
from huggingface_hub import snapshot_download
import pandas as pd
from fora.geometry.xray import read_release_film        # needs `pip install imagecodecs`

root = snapshot_download("Forithmus/FORA", repo_type="dataset",
                         allow_patterns=["splits.csv", "labels/*", "metadata/*", "reports/*", "images/batch01/*"])
meta = pd.read_csv(f"{root}/metadata/batch01_metadata.csv")
film = read_release_film(root, meta.iloc[0]["file"])    # float [-1, 1], content-cropped, native resolution
```

## Tests

```bash
python -m pytest data-preprocessing/tests
```

CPU smoke tests on a synthetic shard: geometry, windows and the MIP rule, the cache codec, a tiny plan -> cache (stub
encoder) -> region bits -> bags -> pack round trip, the split rule, exam types and the protocol-classifier helpers.
