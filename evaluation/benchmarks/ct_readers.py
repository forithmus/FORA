"""Readers for the public CT classification benchmarks (paper Methods 4.9). Each returns a ``Truth``:
case ids, cluster ids for resampling, finding names and an int8 label matrix (1 positive, 0 negative, -1 unknown).

* CT-RATE validation (18 findings, 1,564 studies of 1,304 patients): ``multi_abnormality_labels`` CSV with one row
  per reconstruction (``VolumeName`` = ``valid_<patient>_<study>_<recon>.nii.gz``). A study is positive when any of
  its reconstructions is positive. Clusters = patients.
* Merlin test (30 findings, 5,125 studies): ``zero_shot_findings_disease_cls.csv`` (1 / 0 / -1 per finding),
  restricted to the given test study ids. Clusters = studies.
* RAD-ChestCT (16 findings, 3,630 scans): the supplied 18-label CSVs (train, valid, test; a row present in two
  files must agree). Arterial wall and coronary artery wall calcification are merged by OR into "Calcification"
  and the mosaic attenuation pattern is removed. Clusters = studies (scans).
* KiTS23 renal cyst (489 public training cases): positive when the segmentation contains any label-3 voxel.
* RibFrac validation (80 scans, RibFrac421-500): positive when ``ribfrac-val-info.csv`` lists any fracture
  instance (label_id > 0, including label_code -1).
"""
from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass

import numpy as np

CTRATE_FINDINGS = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion',
                   'Coronary artery wall calcification', 'Hiatal hernia', 'Lymphadenopathy', 'Emphysema', 'Atelectasis',
                   'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern',
                   'Peribronchial thickening', 'Consolidation', 'Bronchiectasis', 'Interlobular septal thickening']
RAD_FINDINGS = ['Medical material', 'Cardiomegaly', 'Pericardial effusion', 'Hiatal hernia', 'Lymphadenopathy', 'Emphysema',
                'Atelectasis', 'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Peribronchial thickening',
                'Consolidation', 'Bronchiectasis', 'Interlobular septal thickening', 'Calcification']


@dataclass
class Truth:
    case_ids: list
    clusters: list
    findings: list
    labels: np.ndarray

    def save(self, path: str) -> None:
        np.savez_compressed(path, case_ids=np.array(self.case_ids), patient_ids=np.array(self.clusters),
                            findings=np.array(self.findings), labels=self.labels.astype(np.int8))


def ctrate(labels_csv: str) -> Truth:
    rows = list(csv.DictReader(open(labels_csv)))
    by_study: dict = {}
    for r in rows:
        name = os.path.basename(r['VolumeName']).split('.')[0]
        study = name.rsplit('_', 1)[0]
        y = np.array([int(float(r[f])) for f in CTRATE_FINDINGS])
        by_study[study] = np.maximum(by_study.get(study, y), y)
    ids = sorted(by_study, key=lambda s: (int(s.split('_')[1]), s))
    return Truth(ids, ['_'.join(s.split('_')[:2]) for s in ids], list(CTRATE_FINDINGS), np.stack([by_study[s] for s in ids]))


def merlin(labels_csv: str, test_ids: list[str], findings: list[str]) -> Truth:
    raw = {r['study id']: r for r in csv.DictReader(open(labels_csv))}
    Y = np.array([[int(float(raw[u][f])) if u in raw else -1 for f in findings] for u in test_ids])
    return Truth(list(test_ids), list(test_ids), list(findings), Y)


def radchestct(label_csvs: list[str], case_ids: list[str] | None = None) -> Truth:
    labels: dict = {}
    for f in label_csvs:
        for r in csv.DictReader(open(f)):
            u = r.pop('AccessionNo'); v = {k: int(float(x)) for k, x in r.items()}
            if u in labels:
                assert labels[u] == v, u
            labels[u] = v
    ids = sorted(case_ids if case_ids is not None else labels)
    Y = []
    for u in ids:
        y = dict(labels[u])
        y['Calcification'] = max(y.pop('Arterial wall calcification'), y.pop('Coronary artery wall calcification'))
        y.pop('Mosaic attenuation pattern')
        Y.append([y[f] for f in RAD_FINDINGS])
    return Truth(ids, ids, list(RAD_FINDINGS), np.array(Y))


def kits23(root: str) -> Truth:
    import nibabel as nib
    ids, Y = [], []
    for d in sorted(glob.glob(os.path.join(root, 'case_*'))):
        lab = np.asarray(nib.load(os.path.join(d, 'segmentation.nii.gz')).dataobj)
        ids.append(os.path.basename(d)); Y.append([int((lab == 3).any())])
    return Truth(ids, ids, ['Renal cyst'], np.array(Y))


def ribfrac(info_csv: str) -> Truth:
    info: dict = {}
    for r in csv.DictReader(open(info_csv)):
        info.setdefault(r['public_id'], []).append(int(r['label_id']))
    ids = sorted(info, key=lambda u: int(u.replace('RibFrac', '')))
    return Truth(ids, ids, ['Rib fracture'], np.array([[int(any(k > 0 for k in info[u]))] for u in ids]))


def main() -> None:
    import argparse
    import json
    ap = argparse.ArgumentParser(description='Write a benchmark truth file (.npz: case_ids, patient_ids, findings, labels).')
    ap.add_argument('dataset', choices=['ctrate', 'merlin', 'radchestct', 'kits23', 'ribfrac'])
    ap.add_argument('--input', nargs='+', required=True, help='CT-RATE labels CSV | Merlin CSV | RAD-ChestCT label CSVs | KiTS23 root | RibFrac info CSV')
    ap.add_argument('--ids', help='text file with one case id per line (Merlin test studies; optional RAD-ChestCT subset)')
    ap.add_argument('--mappings', default=os.path.join(os.path.dirname(__file__), 'configs', 'ct_mappings.json'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ids = [l.strip() for l in open(a.ids) if l.strip()] if a.ids else None
    if a.dataset == 'ctrate':
        T = ctrate(a.input[0])
    elif a.dataset == 'merlin':
        T = merlin(a.input[0], ids, list(json.load(open(a.mappings))['Merlin']['findings']))
    elif a.dataset == 'radchestct':
        T = radchestct(a.input, ids)
    elif a.dataset == 'kits23':
        T = kits23(a.input[0])
    else:
        T = ribfrac(a.input[0])
    T.save(a.out)
    print(a.dataset, len(T.case_ids), 'cases', len(T.findings), 'findings', 'positives', (T.labels == 1).sum(0).tolist())


if __name__ == '__main__':
    main()
