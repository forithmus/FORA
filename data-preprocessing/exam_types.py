"""Exam types: the 28 protocol classes grouped into 11 CT types, or "mixed" (Methods, "Dataset"; dataset tables).

A report gets the type whose class set contains all of its protocol classes (classifier numbering n = 1..29); a report
whose classes fall into more than one type, or into a class outside every type (dental cone-beam CT), is "mixed".

Usage
  python exam_types.py --report-protocols report_protocols.csv --output exam_types.csv
"""
from __future__ import annotations

import argparse
import collections
import csv

EXAM_TYPES = {
    'Chest CT': {6, 7},
    'Abdomen-pelvis CT': {1, 2, 3, 4, 21},
    'Chest-abdomen-pelvis CT': {8, 9, 10},
    'Head CT': {12, 13},
    'Sinus / orbit CT': {18},
    'Temporal bone CT': {19},
    'Neck CT': {16, 17},
    'Spine CT': {5, 15, 20},
    'Extremity CT': {11},
    'CT angiography': {22, 23, 24, 25, 26, 27},
    'Coronary CTA': {28},
}


def exam_type(classes) -> str | None:
    s = set(classes)
    for g, m in EXAM_TYPES.items():
        if s and s <= m:
            return g
    return 'mixed' if s else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--report-protocols', required=True, help='CSV with ReportUID, protocol_class (report_protocol_classes.py)')
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    csv.field_size_limit(10 ** 9)
    prot = collections.defaultdict(set)
    for r in csv.DictReader(open(a.report_protocols)):
        prot[r['ReportUID']].add(int(r['protocol_class']))
    with open(a.output, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ReportUID', 'protocol_classes', 'exam_type'])
        for u, s in prot.items():
            w.writerow([u, ' '.join(map(str, sorted(s))), exam_type(s)])
    print(collections.Counter(exam_type(s) for s in prot.values()).most_common())


if __name__ == '__main__':
    main()
