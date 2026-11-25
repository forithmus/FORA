# Gaps

1. **Statistics procedures.** `statistics/` implements the interval and test procedures described in Methods
   4.9-4.11. It reproduces every interval and p-value printed in the paper text (README, Statistics); intervals that
   appear only in figures were not checked number by number.
2. **ReXGroundingCT localization scorer.** The scoring script used for the ReXGroundingCT numbers (414 studies,
   1,139 evaluable instances) is not available. `localization/ct_readers.rexg` reads the annotations (it reproduces
   the 414 studies, 1,152 instances and 1,139 evaluable instances) and `localization/score_ct.py` applies the same
   generic measures as for FUMPE, KiTS23 and RibFrac, but the resulting ReXGroundingCT numbers have not been checked
   against the paper. The segmentation file's affine is used as image affine. The route-uniform chance for
   ReXGroundingCT needs the per-token route and centre-in-annotation arrays from the inference stage.
3. **Chance inputs.** `ct_metrics.uniform_route_chance` needs, per token, whether its centre lies in the
   annotation and whether it is in the finding's route. Producing these arrays requires the token geometry and
   CADS region bits of the inference stage; no CLI here builds them.
4. **PadChest-GR internal development splits, MIMIC-CXR.** Readers are provided only for the test sets used in the
   paper tables.
5. **Laterality agreement** of generated CT reports (a secondary report statistic) is not included; it depends on
   the report-generation sentence attribution tools.
6. **GLINT-CT localization comparison** (threshold sweep of its similarity map) is not included.
