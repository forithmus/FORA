# Gaps

- **Global extraction input.** The global-vocabulary extraction ran on one row per report (346,441 reports). The step
  that reduced the per-protocol input (`prepare_input.py`, one row per report and protocol class) to one row per report,
  and which protocol class it kept for the prompt framing, is not available. `prepare_input.py` produces the
  per-protocol input only.
- **Recall-pass "Nasal cavity abnormality" items.** The adjudication item file contains, besides the items written by
  `build_label_db --dump-items`, JUDGE items for recall-pass additions of "Nasal cavity abnormality" (from septal
  deviation). The code that selected and appended them is not available.
- **Hierarchy and vocabulary generators.** `hierarchy.json`, `hierarchy_base.json`, `global_vocabulary.json` and the
  per-protocol vocabularies are shipped as data. The code that built them (taxonomy import, curated merges, umbrella
  definitions and membership rules, audit fixes) is not part of this release; the 15 structural fixes that turn
  `hierarchy_base.json` into `hierarchy.json` were applied by hand.
- **X-ray taxonomy source.** `data/cxr/vocabulary.json` is shipped as data; the spreadsheet it was generated from is not
  available.
- **Label audit.** The double-blind label audit (two judges + adjudicator, 2,403 items over 215 reports) was run as an
  agent workflow whose script is not available; only its results are reported in the paper.
- **X-ray report-scoring variant.** Two X-ray scorers exist downstream of `cxr_states.py`: plain class matching, and
  matching with subtype roll-up plus synonym-group merging. Which one the paper tables use is decided in `evaluation/`.
- **Scoring-labeler model revision.** The CT scoring labeler loaded a local Gemma-4-31B-it copy whose snapshot revision
  was not recorded; the X-ray scorer and all training-label passes used snapshot `518276fb`. Gemma thinking mode was not
  enabled explicitly (chat-template default) in any pass.
- **Embedded authoring note.** The umbrella definition of "Intracranial mass" in the hierarchy and in the extraction
  vocabulary contains an editing note instead of a definition. It is kept verbatim because the extraction prompts used
  this text.
