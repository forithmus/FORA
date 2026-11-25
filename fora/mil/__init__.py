"""Organ-routed attention MIL shared by the classifier, report generation and inference (Methods, "Finding classifier").

Modules:
  model        gated-attention MIL head, soft organ routing, exact two-pass streamed forward/backward over a study bag
  bag          CT study bags from the block-23 cache with per-token region bits and per-finding route masks
  xray         X-ray head build (view embedding) and token factory over online block-24 tokens
  evidence     signed evidence, evidence maps and foci (CT), referenced evidence and foci (X-ray)
  render       axial / coronal / sagittal evidence images for the report writer
  calibration  apply Platt calibration and train-fitted cut-offs
"""
