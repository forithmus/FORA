"""CADS anatomy measurements of one study at inference (paper: Methods, "Measurements").

The CADS measurer (``cads/measure.py``) runs on the study's CT series and CADS label volumes (``<measure_dir>/study.h5`` and
``<measure_dir>/cads_study.h5``, case-list ``h5`` = "study"). From its records the writer inputs are built exactly as for the
training cohort: the study measurement line and the Evans-index / cardiothoracic-ratio texts (``cads/study.py``), then the
vessel, organ and midline texts of the records (``merge.cads_record_text``), which replace a text of the same finding.
"""
from __future__ import annotations

import json
from pathlib import Path

from .cads import io, measure as M
from .cads.study import study_texts
from .merge import cads_record_text


def measure_anatomy(measure_dir, case: dict, calibration: dict, mid_none: float = 2.5):
    """case: case-list line (case, report, h5, patient, series, classes); calibration: cads_calibration.json content.
    -> (measurer result {records, ref, raw, err, ...}, finding texts {case|cls: text}, study line {report: text})."""
    root = str(Path(measure_dir))
    io.set_roots(root, root)
    result = M.work((json.dumps(case), calibration, 1, None, (root, root)))
    study, text, _, _ = study_texts({case['case']: case}, {case['case']: result['raw']}, calibration, mid_none)
    for rec in result.get('records', []):
        value = cads_record_text(rec)
        if value is not None:
            text[case['case'] + '|' + str(rec['cls'])] = value
    return result, text, study
