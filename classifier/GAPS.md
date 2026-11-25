# Gaps

- **Selection of the findings without routing (routing table 3).** The analysis that compared the classifier trained
  with and without routing per finding (AUPRC difference with bootstrap confidence intervals on validation) is not
  available. Its result, the 30 findings set to "no routing", is stored as data in `build_routes.py`
  (`NO_ROUTING_TABLE3`).
- **Merge of the route audit.** The script that merged the two-reviewer LLM audit batches into
  `routes/route_table.json` is not available; the merged table is shipped.
- **Routing table 1 hierarchy.** Table 1 was built from `routes/route_table.json` with the finding hierarchy before the
  15 structural fixes of the final hierarchy, which is not shipped. The resulting routes are shipped as
  `routes/class_routes_table1.json` / `class_region_bits_table1.json`; `build_routes.py table1` rebuilds them only with
  that hierarchy. Tables 2 and 3 rebuild exactly from table 1 with the final hierarchy and the heads file.
- **Rule-A cut-off file writer.** The script that wrote the paper's `ruleA_cutoffs.json` is not available.
  `fit_rule_a.py` implements the rule-A cut-off function used in the paper analysis; it reproduces the stored cut-offs
  exactly where the all-training fallback applies (checked on 15 findings). It needs the protocol titles per report
  (`protocols.json`) and, for external benchmarks, the benchmark label -> finding mapping of the evaluation stage.
- **Assignment of reports to ranks.** `assign_128ranks.json` (size-balanced assignment of reports to 128 ranks) is an input
  from data-preprocessing; the CT trainer requires a world size equal to its number of ranks.
