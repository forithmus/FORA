"""Readers for the chest X-ray classification benchmarks (paper Methods 4.9, Fig. 2e, Supplementary Fig. 5).

Split and label files are those distributed with the RadZero / CARZero zero-shot benchmarks (and the MedKLIP
RSNA test list). Each reader returns ``dict(names=[...], ids=[...], Y=int array)``; cases are images.

* ChestX-ray14: ``test_list.txt`` (22,433 images, 14 labels, image name followed by 14 0/1 values).
* ChestX-Det10: ``test.json`` (542 images); label k is positive when it is in the image's ``syms``.
* RSNA Pneumonia: ``test.csv`` (5,337 images, ``ID`` and ``classes`` 1 = pneumonia opacity).
* Open-I: ``custom.csv`` (7,470 frontal and lateral images); label k is positive when ``labels_automatic``
  contains the lower-case label name or one of its synonyms. The study id is the file name before ``_IM``.
* PadChest-GR test: the HF parquet shards; the label set of an image is the set of its ``label`` rows (including
  "Normal"). Case ids are ``padchest_gr/<sha1(image bytes)[:20]>``. ``padchest_gr24`` turns the label sets into the
  24-finding protocol (a finding is positive when any of its PadChest labels is present).
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os

import numpy as np

CXR14 = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
         'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
CXD10 = ['Atelectasis', 'Calcification', 'Consolidation', 'Effusion', 'Emphysema', 'Fibrosis', 'Fracture', 'Mass', 'Nodule', 'Pneumothorax']
OPENI = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Edema', 'Emphysema',
         'Fibrosis', 'Pleural_Thickening', 'Hernia', 'Fracture', 'Opacity', 'Lesion', 'Calcified Granuloma', 'Granuloma']
OPENI_SYNONYMS = {'Pleural_Thickening': ['pleural thickening'], 'Infiltration': ['Infiltrate'], 'Atelectasis': ['Atelectases']}


def chestxray14(test_list: str) -> dict:
    rows = [l.split() for l in open(test_list) if l.strip()]
    ids = ['chestxray14/' + os.path.basename(r[0])[:-4] for r in rows]
    return dict(names=list(CXR14), ids=ids, Y=np.array([[int(x) for x in r[1:]] for r in rows]))


def chestxdet10(test_json: str) -> dict:
    d = json.load(open(test_json))
    return dict(names=list(CXD10), ids=['chestxdet10/' + x['file_name'][:-4] for x in d],
                Y=np.array([[int(c in x['syms']) for c in CXD10] for x in d]))


def rsna(test_csv: str) -> dict:
    rows = list(csv.DictReader(open(test_csv)))
    return dict(names=['Pneumonia'], ids=['rsna/' + r['ID'] for r in rows], Y=np.array([[int(r['classes'])] for r in rows]))


def openi(custom_csv: str) -> dict:
    import pandas as pd
    c = pd.read_csv(custom_csv).replace(np.nan, '-1')
    gt = []
    for p in OPENI:
        m = c['labels_automatic'].str.contains(p.lower())
        for s in OPENI_SYNONYMS.get(p, []):
            m |= c['labels_automatic'].str.contains(s.lower())
        gt.append(m.values)
    ids = ['openi/' + f[:-8] for f in c['file_name']]
    return dict(names=list(OPENI), ids=ids, Y=np.asarray(gt).T.astype(int))


def openi_study(case_id: str) -> str:
    return case_id.split('/')[-1].split('_IM')[0]


def padchest_gr_label_sets(parquet_dir: str) -> dict:
    import pyarrow.parquet as pq
    sets: dict = {}
    for p in sorted(glob.glob(os.path.join(parquet_dir, '*.parquet'))):
        f = pq.ParquetFile(p)
        for g in range(f.num_row_groups):
            for r in f.read_row_group(g, columns=['image', 'label']).to_pylist():
                b = r['image']; b = b['bytes'] if isinstance(b, dict) else b
                sets.setdefault('padchest_gr/' + hashlib.sha1(b).hexdigest()[:20], set()).add(r['label'])
    return {k: sorted(v) for k, v in sets.items()}


def padchest_gr24(label_sets: dict, mapping24: dict, ids: list[str] | None = None) -> dict:
    names = list(mapping24); ids = sorted(label_sets) if ids is None else ids
    Y = np.array([[int(any(l in mapping24[k]['padchest_labels'] for l in label_sets[i])) for k in names] for i in ids])
    return dict(names=names, ids=ids, Y=Y)
