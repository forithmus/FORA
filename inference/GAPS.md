# Gaps

## Not validated

* CT DICOM and NPZ input and X-ray DICOM / image input were checked on synthetic data only.
* GPUs smaller than 141 GB were not validated for the CT writer.
* `--opener auto` on mixed-protocol studies was not evaluated.

## Open questions

* FORA validation report rows: generated without a forced opener; whether the phrase ban was applied is recorded
  inconsistently in the run notes. The table in README.md gives only `--opener none`.
* The `facebook/dinov2-large` revision used by the calipers is not recorded; `weights.json` gives the sha256 of the
  files instead.
