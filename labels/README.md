# labels: report labels, finding hierarchy, classifier targets, report-scoring labeler

This stage turns free-text reports into finding labels (Methods, "Report labels" and "X-ray labels"; Supplementary Data
1-3, Box S3). CT reports are labelled by Gemma-4-31B-it with a three-pass closed-vocabulary extraction, extended by a
recall pass, a malignancy check and an adjudication/mapping pass, and propagated over a 1,459-node finding hierarchy into
a label database with four states per report and finding. Head assignment selects the 1,028 CT classifier classes. X-ray
reports go through the same three-pass scheme with a 304-leaf vocabulary into a SQLite label index and 129 classifier
targets. `report_scoring/` holds the labeler used to score generated reports (Methods, "Report evaluation").

All LLM passes run Gemma-4-31B-it (Hugging Face `google/gemma-4-31B-it`, snapshot `518276fb`) under vLLM, one process
per GPU (tensor parallel 1), greedy decoding, prefix caching. Shard a run over N GPUs by starting N processes with
`--rank i --world N` (or under Slurm, where `SLURM_PROCID`/`SLURM_NTASKS` are read). Run every command from the
repository root (`python -m labels....`) with the repository on `PYTHONPATH`; relative paths in configs resolve against
`labels/`.

## Shared code (`fora/labels`)

| module | content |
|---|---|
| `states.py` | state codes: 1 asserted, 2 uncertain, 0 denied, 3 unmentioned (negative), -1 masked; `to_target` (1/2 -> 1, 0/3 -> 0) |
| `hierarchy.py` | `load_hierarchy`, `Hierarchy` (sorted node order = DB column order, ancestors, descendants), `canonicalizer` (name/alias/redirect, curated alias table, spelling variants) |
| `propagation.py` | `report_state`: asserted and uncertain propagate to ancestors, denials to descendants, a positive always wins |
| `heads.py` | `assign_heads` (>= 200 train positives and negatives: own head; 20-199: attached to the own-head ancestor with the fewest train positives; else none), `exclude_heads`, `load_heads` -> (classes, parent_of) |

## Data files

| file | content |
|---|---|
| `data/ct/hierarchy.json` | final hierarchy (1,286 leaves + 173 umbrellas); used for the final label DB |
| `data/ct/hierarchy_base.json` | the same class list without 15 structural fixes; used for the intermediate DBs (recall, malignancy and adjudication inputs), for head tiering and by the report scorer |
| `data/ct/global_vocabulary.json` | the 1,459-entry extraction vocabulary (specific leaves first, umbrella "GENERAL:" entries with definitions last) |
| `data/ct/protocols.json` | per-protocol framing (title, region, contrast note, hints, second look) and per-protocol leaf vocabularies for the 29 protocol classes |
| `data/ct/unmapped_aliases.json` | curated extractor-name -> hierarchy-name table |
| `data/ct/heads_1028.json` | final CT classifier heads (749 own + 279 attached), consumed by the classifier |
| `data/cxr/vocabulary.json` | X-ray vocabulary: 304 leaves (name, synonyms, group, type), hints, second-look list |
| `report_scoring/data/ct_protocols_scoring.json` | per-protocol vocabularies of the CT scoring labeler (1,106 labels with the extras) |
| `report_scoring/data/ct_scoring_extras.json` | extractor-extra label names of the CT scoring vocabulary |

Vocabulary `syn` fields and hierarchy `aliases` contain alternative concept names only. Definitions exist for the 173
umbrella nodes and were written for this hierarchy. All prompts are plain text files under `prompts/`; templates use
Python `str.format` slots (`{findings}`, `{vocabulary}`, ...; literal braces doubled). The recall, malignancy and mapping
system prompts are used as written with the vocabulary block appended.

## CT pipeline (Methods, report labels)

Inputs: structured reports CSV (ReportUID, PatientUID, findings, impression), the ingestion manifest (one row per series
with ReportUID, StudyInstanceUID, h5_id, enriched_with_report, classified_protocols) and the case list of the CT
classifier (`cases.jsonl`, lines with `id` and `report`; split = train when the id starts with `train`).

```bash
M=<gemma-4-31B-it snapshot dir>
# 0. input rows, one per (report, protocol class)
python -m labels.ct.prepare_input --manifest manifest.csv --reports structured_reports.csv --output per_protocol_input.csv
# global_input.csv: one row per report (346,441 reports; see GAPS.md)

# 1a. per-protocol-vocabulary extraction (positives are unioned in step 5)
python -m labels.ct.extract_findings --config configs/ct_extract_per_protocol.json --input per_protocol_input.csv \
    --output-dir per_protocol/shards --log-dir per_protocol/logs --model $M --rank $i --world 256
# 1b. global-vocabulary extraction, then two retries of the reports with any non-pass row
python -m labels.ct.extract_findings --config configs/ct_extract_global.json --input global_input.csv \
    --output-dir global/shards --log-dir global/logs --model $M --rank $i --world 256
python -m labels.ct.prepare_input --retry-from global/shards --retry-input global_input.csv --retry-output retry1_input.csv
python -m labels.ct.extract_findings --config configs/ct_extract_global_retry1.json --input retry1_input.csv \
    --output-dir retry1/shards --log-dir retry1/logs --model $M --rank $i --world 128
python -m labels.ct.prepare_input --retry-from retry1/shards --retry-input retry1_input.csv --retry-output retry2_input.csv
python -m labels.ct.extract_findings --config configs/ct_extract_global_retry2.json --input retry2_input.csv \
    --output-dir retry2/shards --log-dir retry2/logs --model $M --rank $i --world 128

EXTR="--shards global/shards --retry-shards retry1/shards retry2/shards --union-shards per_protocol/shards --cases cases.jsonl"
# 2. recall pass on the extraction-only DB (no alias table)
python -m labels.ct.build_label_db $EXTR --hierarchy labels/data/ct/hierarchy_base.json --output db_extraction
python -m labels.ct.recall_pass --input global_input.csv --labels db_extraction/labels.npz --model $M \
    --output recall/first_$i.recs --rank $i --world 128
python -m labels.ct.merge_recall --input global_input.csv --first recall/first_*.recs --retry-ids recall/retry_ids.txt
python -m labels.ct.recall_pass --input global_input.csv --labels db_extraction/labels.npz --model $M \
    --only-reports recall/retry_ids.txt --max-tokens 6144 --output recall/retry_$i.recs --rank $i --world 128
python -m labels.ct.merge_recall --first recall/first_*.recs --retry recall/retry_*.recs --output recall_merged.recs
# 3. malignancy check on extraction + aliases + recall
A="--aliases labels/data/ct/unmapped_aliases.json"
python -m labels.ct.build_label_db $EXTR $A --hierarchy labels/data/ct/hierarchy_base.json --recall recall_merged.recs --output db_recall
python -m labels.ct.malignancy_check --input global_input.csv --labels db_recall/labels.npz --model $M \
    --output malignancy/$i.can --rank $i --world 128
cat malignancy/*.can > malignancy.can
# 4. adjudication / mapping items, then the LLM pass
python -m labels.ct.build_label_db $EXTR $A --hierarchy labels/data/ct/hierarchy_base.json --recall recall_merged.recs \
    --malignancy malignancy.can --dump-items adj_items.txt --output db_dump
python -m labels.ct.adjudicate --items adj_items.txt --input global_input.csv --model $M --output adj/$i.out --rank $i --world 128
cat adj/*.out > adjudication.out
# 5. final label DB (final hierarchy) and heads
python -m labels.ct.build_label_db $EXTR $A --hierarchy labels/data/ct/hierarchy.json --recall recall_merged.recs \
    --malignancy malignancy.can --adjudication adjudication.out --output db_final
python -m labels.ct.build_label_db $EXTR $A --hierarchy labels/data/ct/hierarchy_base.json --recall recall_merged.recs \
    --malignancy malignancy.can --adjudication adjudication.out --output db_base
python -m labels.ct.assign_heads --labels db_base/labels.npz --hierarchy labels/data/ct/hierarchy_base.json \
    --config configs/ct_heads.json --output heads.json
```

`--world` values in the step-0 to step-4 commands are the GPU counts of the paper runs and do not change results. Step 5 was rerun with this code from the recorded pass outputs: `db_final/labels.npz` is identical
to the label DB used by the classifier, `db_base/labels.npz` plus `assign_heads` reproduces `data/ct/heads_1028.json`
exactly, and the extraction-only DB of step 2 reproduces the recorded counts of the recall-input DB.

Outputs: `labels.npz` (reports, cases, split, nodes[1459], state int8 [reports, 1459]), `heads.json`, `summary.json`.
Label DB of the paper: 341,581 reports (304,962 train / 36,619 val), 9.97 positives per report on average; recall pass added
197,769 asserted + 10,732 uncertain labels in 145,484 reports; malignancy check removed 10,804 history-only and 952
unsupported labels; 22,854 remaining positive/denial conflicts resolved to positive.

Heads (`configs/ct_heads.json`): own >= 200 train positives and >= 200 negatives, attached 20-199. Crystalluria and
Gastroesophageal reflux are excluded (not visible on CT) by the `exclude` list. The classifier reads `classes = own +
attached` and the per-class `train_pos`/`train_neg` of `heads_1028.json` (positive weights).

## X-ray pipeline (Methods, X-ray labels)

```bash
python -m labels.cxr.prepare_input --source cxr_structured_reports.csv --out cxr_input.csv
python -m labels.cxr.extract_findings --input cxr_input.csv --output-dir cxr/shards --log-dir cxr/logs --model $M --rank $i --world 16
python -m labels.cxr.merge_shards --shard-dir cxr/shards --out cxr_extracted.csv --expect-ranks 16
python -m labels.cxr.build_index --manifest cxr_manifest.csv --labels cxr_extracted.csv --splits cxr_splits.csv \
    --vocabulary labels/data/cxr/vocabulary.json --h5_root <cxr h5 dir> --output cxr_labels_index.sqlite --include_empty_targets
python -m labels.cxr.build_scoped_supervision --base-index cxr_labels_index.sqlite --h5-root <cxr h5 dir> \
    --output cxr_labels_scoped.sqlite --audit-output cxr_labels_scoped_audit.json
python -m labels.cxr.select_targets --label-db cxr_labels_scoped.sqlite --cases-train cases_train.jsonl \
    --cases-val cases_val.jsonl --out-dir cxr_targets
```

The index commands are reconstructed from the arguments recorded in the SQLite metadata. The index holds 311 labels: the
304 vocabulary leaves plus 7 extractor spellings that resolve to no leaf, kept as explicit labels. The scoped index adds
anatomy-scoped negatives from explicit normal statements (high precision, weighted); they do not change the classifier
targets. `select_targets` (target = asserted or uncertain; >= 50 train positives; "No Finding / Normal study" dropped)
reproduces the 129 X-ray classes and target matrices of the classifier exactly from the paper's label index and case
lists. The case lists (reports with at least one valid film) come from the X-ray case index of the classifier stage.

## Report scoring (`report_scoring/`, Methods, report evaluation)

CT: generated and reference reports are labelled with the scoring extractor (same three-pass engine, scoring prompts,
closed per-protocol vocabularies, 24,576-token context) on findings text, then mapped to the 1,028 classes.

```bash
python -m labels.report_scoring.make_tasks --studies studies.jsonl --generations gen_*.jsonl --output tasks.jsonl
python -m labels.report_scoring.label_reports --tasks tasks.jsonl --output scoring/labels_$i.jsonl --model $M --rank $i --world 64
```

then group the rows of one (source, cohort, uid) and call `ct_states.Resolver().report_states(rows)`. Protocol classes:
FORA studies use all protocol classes of the report, Merlin its labelled class, CT-RATE class 6 or 7 from the Technique
field (`make_tasks.ctrate_protocol`). The extras list is rebuilt from the reference-report extraction CSV with
`build_extras.py`; the shipped list reproduces the 1,106-label vocabulary and resolver counts of the scored index.

X-ray: generated and reference reports are labelled with `labels.cxr.extract_findings --chunk-size 2000` (input rows
ReportUID = `<source>|<study id>`, findings = text without a leading "Findings:" header, empty impression), merged with
`labels.cxr.merge_shards`, and mapped with `cxr_states.CXRResolver(label_db, classes)`. The metrics are computed in
`evaluation/`.

## Tests

```bash
python -m pytest labels/tests
```
CPU only: shared label code on toy data and the shipped hierarchy/heads, the three-pass engines with a fake LLM for every
prompt set, the label-DB build on a synthetic corpus, X-ray target selection, scoped normality and both scoring resolvers.
