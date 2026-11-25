"""Write the extractor-extra label list of the CT scoring vocabulary from the reference-report extraction CSV.

The input is the merged per-report output of the scoring extractor over the reference reports (column
``pathology_labels``: JSON list of {protocol_class, asserted, uncertain, denied}). The output lists label names only.

    python -m labels.report_scoring.build_extras --labels reference_extraction.csv \
        --output labels/report_scoring/data/ct_scoring_extras.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from labels.report_scoring.ct_vocab import DEFAULT_PROTOCOLS, load_vocabularies, observed_extras, scoring_vocabulary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--protocols", default=DEFAULT_PROTOCOLS)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    vocabs = load_vocabularies(a.protocols)
    extras, audit = observed_extras(a.labels, vocabs)
    n = len(scoring_vocabulary(vocabs, extras))
    json.dump({"extras": {str(k): v for k, v in sorted(extras.items())}, "scoring_vocabulary_size": n,
               "resolver_audit": dict(audit)}, open(a.output, "w"), indent=1, ensure_ascii=False)
    print(n, dict(audit))


if __name__ == "__main__":
    main()
