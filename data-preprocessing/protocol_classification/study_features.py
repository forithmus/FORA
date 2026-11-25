"""Study-level derived features injected into the protocol-classification prompt.

The language model is good at SEMANTICS (reading
through mojibake / typos, judging whether "omnipque" or "UNDEFINED" is a real
contrast agent) but bad at ARITHMETIC and at seeing EVERY series in a long or
truncated study. So code does only the counting / full-study aggregation and
hands the model a short, trustworthy block; the model still judges meaning.

``render_derived_features`` is called by ``build_study_prompt`` with the study's
full list of per-series dicts (keyed by SERIES_METADATA_COLS). It computes, over
the CT diagnostic series only:

  * contrast-agent values recorded (verbatim, deduped — the LLM decides which are
    real agents vs placeholders like UNDEFINED / NO ENTRY)
  * distinct Contrast/BolusStartTime count  (the multiphase signal code counts
    reliably; the tag is blank ~93% of series so the model cannot tally it well)
  * distinct AcquisitionNumber count

It returns "" when the study has no CT series (nothing useful to add).
"""
from __future__ import annotations

# Column keys as they appear in each series_rows dict (SERIES_METADATA_COLS).
_MODALITY = "Modality"
_AGENT = "Contrast/BolusAgent"
_BOLUS_TIME = "Contrast/BolusStartTime"
_ACQ = "AcquisitionNumber"
_CT = "CT"

# How many distinct agent strings to surface (keeps the block short).
_MAX_AGENTS_LISTED = 8


def _blank(v) -> bool:
    s = ("" if v is None else str(v)).strip()
    return s == "" or s.lower() == "nan"


def _distinct_nonblank(rows, key) -> list[str]:
    """Order-preserving unique of non-blank values for ``key`` across rows."""
    out: list[str] = []
    seen: set[str] = set()
    for r in rows:
        v = r.get(key, "")
        if _blank(v):
            continue
        s = str(v).strip()
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def compute_features(series_rows) -> dict | None:
    """Return the derived-feature dict for one study, or None if no CT series."""
    ct = [r for r in series_rows if str(r.get(_MODALITY, "")).strip().upper() == _CT]
    if not ct:
        return None
    return {
        "agents_seen": _distinct_nonblank(ct, _AGENT),
        "n_bolus_times": len(_distinct_nonblank(ct, _BOLUS_TIME)),
        "n_acq": len(_distinct_nonblank(ct, _ACQ)),
        "n_ct_series": len(ct),
    }


def render_derived_features(series_rows) -> str:
    """Render the DERIVED STUDY FEATURES block for a study's series rows.

    Returns "" when there is no CT series (the block would carry no signal)."""
    feat = compute_features(series_rows)
    if feat is None:
        return ""
    agents = feat["agents_seen"]
    shown = agents[:_MAX_AGENTS_LISTED]
    extra = len(agents) - len(shown)
    if shown:
        agents_str = ", ".join(shown) + (f"  (+{extra} more)" if extra > 0 else "")
    else:
        agents_str = "(none recorded)"
    n_bolus = feat["n_bolus_times"]
    if n_bolus == 0:
        phase_hint = " (no bolus time recorded → judge phases from SeriesDescriptions)"
    elif n_bolus == 1:
        phase_hint = " (single bolus time → single-phase unless SeriesDescriptions show distinct phases)"
    else:
        phase_hint = " (≥2 bolus times → multiphase)"
    return (
        "== DERIVED STUDY FEATURES (computed in code across ALL series — trust these counts) ==\n"
        f"- contrast-agent values recorded across series: {agents_str}  "
        "[judge which are REAL agents vs placeholders: UNDEFINED / NO ENTRY / UNKNOWN ≠ contrast]\n"
        f"- distinct contrast bolus-start times: {n_bolus}{phase_hint}\n"
        f"- distinct acquisition numbers: {feat['n_acq']}"
    )
