# Gaps

- **X-ray student windows from step 0.** The X-ray recipe sets student windows `0:1,0:0.65,0.2:0.8,0.45:1`. They are
  confirmed for the later part of the run (from about step 20,000 to 30,000). For the first part of the run the launch
  command is not available, so it cannot be shown from the files that the windows were on from step 0. The recipe assumes
  that they were.
- **X-ray partial resume.** The X-ray resumes ran non-strict (`--resume-allow-partial`). Which tensors, if any, were
  missing at those resumes is not recorded.
- **CT phase 1, steps 0-2,500.** The log of the first segment is not available. Config and metrics (from step 1) are.
  One interruption at step 9,000 resumed from the step-8,500 checkpoint, so steps 8,501-9,000 were run twice. The recipe
  did not change.
- **Shard staging and counts.** The paper runs staged shards to node-local disk and balanced CT shards by a cached count
  of reconstructions per shard. The cache file is recomputed when `--assignment-cache` is empty or stale. Its content
  only changes how shards are distributed over ranks, not the samples.
