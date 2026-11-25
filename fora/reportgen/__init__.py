"""Report-generation runtime shared by report-generation training and inference (Methods, "Grounded report generation").

Modules:
  plan       diagnostic plan builder (calibrated tiers, plan prompt text)
  evidence   evidence extraction per planned finding, finding descriptions, focus tokens, finding store, evidence images
  measure    CADS anatomy measurements and lesion calipers
  writer     Gemma-4 writer model: visual resampler, label / evidence tokens, LoRA, visual stream
  decoding   per-study seed, forced opener, phrase-ban lists, writer construction and decoding
  cxr        chest X-ray visual stream, writer model, plan rules and location text
"""
