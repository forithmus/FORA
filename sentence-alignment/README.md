# Sentence alignment

Contrastive alignment of the encoder's last block to the sentence statements of the reports (Methods, "Sentence
alignment"). Blocks 1-23 of the pre-trained teacher stay frozen; block 24 + the final norm, a projection and CXR-BERT are
trained so that each statement ("There is ...", "There is no ...") scores highest against its own report. A report is a
bag of all its views (CT: every usable series in the full window, each distinct native window and, with a lung window,
the 10 mm lung MIP; X-ray: all films). A statement scores a report through VL-CABS attention pooling over all of the
report's tokens; every statement of a report is a positive. The aligned block 24 initializes the block 24 of the
finding classifier.

## Inputs and outputs

| | Format |
|---|---|
| CT block-23 cache | series files of format `fora_int8_tokens` (see `fora/encoder/cache.py`), written by `data-preprocessing`; packed series are restored to `cache_root` on first use |
| CT `cases.jsonl` | one report per line: `id`, `views: [{relative_file, file_size, file, offset, group, n_tokens, window: {kind}}]` |
| CT `assign.json` (optional) | `{ranks, assignments: {train: [[ids of rank r]], val: [...]}}`: reports per rank (data staged node-locally); without it reports are spread round-robin |
| `split.json` | `{case id: "train" or "val"}`; only `train` reports are used |
| `sentences.json` | `{case id: [statements]}` (`fora_align.extract_sentences`) |
| `normalization.pt` | `{mean [1024], rms}` of block-24 tokens of the initial encoder |
| `block24_state.pt`, `teacher.pt` | from `encoder-pretraining` (`fora_pretrain.export_teacher`) |
| X-ray | H5 shards, split CSV; `cases_train.jsonl` from `train_cxr index` |
| Output | `OUT/latest.pt` (every 200 steps), `step_*.pt`, `epoch_XX.pt` = `{training, block, vproj, text, temp, pool_temp, optimizer, step, epoch, ...}`, `progress.json`, `COMPLETE.json` |

## Run

Text model: CXR-BERT (`microsoft/BiomedVLP-CXR-BERT-specialized`, a local copy or the HF id in `text_model`). Training
runs on 32 nodes x 4 GPUs = 128 reports per step. Run from this folder with the repository root on `PYTHONPATH`.

```bash
# CT statements of the planned reports
python -m fora_align.extract_sentences --reports plan/reports_train.jsonl plan/reports_val.jsonl --h5-folder H5 --out data/ct/sentences.json
# CT: 3 epochs, then the continuation started from epoch 3 (epoch 7 of the continuation is used)
torchrun --nnodes 32 --nproc-per-node 4 ... -m fora_align.train_ct --config configs/ct_alignment.json --output runs/ct
torchrun --nnodes 32 --nproc-per-node 4 ... -m fora_align.train_ct --config configs/ct_alignment_continuation.json \
    --output runs/ct_cont --init runs/ct/epoch_03.pt
python -m fora_align.export_block24 runs/ct_cont/epoch_07.pt ct_aligned_block24.pt

# X-ray
python -m fora_align.train_cxr index --config configs/cxr_alignment.json
torchrun --nproc-per-node 4 -m fora_align.train_cxr normalization --config configs/cxr_alignment.json --films 200
torchrun --nnodes 32 --nproc-per-node 4 ... -m fora_align.train_cxr train --config configs/cxr_alignment.json --output runs/cxr
python -m fora_align.export_block24 runs/cxr/epoch_10.pt cxr_aligned_block24.pt
```

SIGUSR1 saves `latest.pt` and exits; restarting with the same command resumes.

## Configs that reproduce the paper

| Config | Run |
|---|---|
| `configs/ct_alignment.json` | CT, 3 epochs (2,368 steps per epoch), warm-up 300, lr block 2e-5 / projections 1e-4 / BERT 1e-5 / temperatures 1e-3, wd 0.05, seed 0 |
| `configs/ct_alignment_continuation.json` | CT continuation from `epoch_03.pt` with half the learning rates, warm-up 200, seed 1; cosine horizon 2 epochs for steps 0-999, 7 epochs for steps 1,000-11,839, 10 epochs from step 11,840 (`epochs_schedule`); stops after epoch 7 (step 16,576) |
| `configs/cxr_alignment.json` | X-ray, 2,006 steps per epoch; cosine horizon 100 epochs for steps 0-8,999 and 10 epochs from step 9,000 (`epochs_schedule`); 10 epochs (20,060 steps) |
