# Gaps

- **CT normalization statistics.** `normalization.pt` (per-dimension mean, scalar rms = sqrt(mean over dimensions of the
  variance)) was computed from block-24 tokens of the step-5,000 CT encoder on CT-RATE training scans: 2,048 random
  scans, 2,048 consecutive tokens each, from a separate CT-RATE token cache. That computation is not part of this
  repository. The file itself is required as input.
- **Statement generation.** The `there_is_statements` stored in the H5 shards were produced upstream by an LLM report
  processing step (normalized "There is / There is no" statements). That step is not part of this repository.
- **First 3 CT epochs.** They ran with the same training code except for `--init` and `epochs_schedule`, which they do
  not use. Intermediate step checkpoints of those epochs are not available; `epoch_01..03.pt` are.
- **Discarded steps.** In the CT continuation, steps 11,841-11,910 were run with the 7-epoch horizon and then repeated
  from the step-11,840 checkpoint with the 10-epoch horizon. In the X-ray run, steps 9,001-9,050 were run with the
  100-epoch horizon and repeated from the step-9,000 checkpoint. The configs reproduce the kept trajectory.
