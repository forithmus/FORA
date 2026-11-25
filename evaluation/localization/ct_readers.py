"""Annotation readers for CT localization (paper Methods 4.10). Each yields ``(case_id, finding, instance, mask,
affine)`` with the mask on the image grid and the image's RAS affine.

* FUMPE (35 scans): pulmonary embolism = any non-zero label voxel.
* KiTS23 (489 cases): renal cyst = label 3; renal mass = labels 2 or 3 (``segmentation.nii.gz`` per case folder).
* RibFrac validation (80 scans): rib fracture = any non-zero instance label (``ribfrac-val-labels.zip``).
* ReXGroundingCT, CT-RATE validation scans (414 scans named ``valid_*``, all in the ReXGroundingCT training
  split of ``dataset.json``): ``dataset.json`` lists per scan the findings with a category; the
  segmentation file ``segmentations/<scan>.nii.gz`` has one channel per finding. An instance is evaluable when its
  category maps to a CT-RATE finding (``REXG_CATEGORY_TO_CTRATE``); categories 2f, 2g and 2h have no CT-RATE class.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import zipfile

import numpy as np

REXG_CATEGORY_TO_CTRATE = {'1a': 'Peribronchial thickening', '1b': 'Bronchiectasis', '1c': 'Emphysema', '1d': 'Interlobular septal thickening',
                           '1e': 'Lung nodule', '1f': 'Mosaic attenuation pattern', '2a': 'Pulmonary fibrotic sequela', '2b': 'Consolidation',
                           '2c': 'Lung opacity', '2d': 'Lung nodule', '2e': 'Pleural effusion'}


def _nib():
    import nibabel as nib
    return nib


def fumpe(truth_dir: str):
    nib = _nib()
    for f in sorted(glob.glob(os.path.join(truth_dir, '*.nii.gz'))):
        im = nib.load(f); lab = np.asarray(im.dataobj)
        yield os.path.basename(f)[:-7], 'Pulmonary embolism', '0', lab > 0, im.affine


def kits23(root: str, findings=('Renal cyst', 'Renal mass')):
    nib = _nib()
    for d in sorted(glob.glob(os.path.join(root, 'case_*'))):
        im = nib.load(os.path.join(d, 'segmentation.nii.gz')); lab = np.asarray(im.dataobj)
        for f in findings:
            yield os.path.basename(d), f, '0', (lab == 3) if f == 'Renal cyst' else ((lab == 2) | (lab == 3)), im.affine


def ribfrac(labels_zip: str):
    nib = _nib()
    with zipfile.ZipFile(labels_zip) as z:
        for name in sorted(n for n in z.namelist() if n.endswith('-label.nii.gz')):
            im = nib.Nifti1Image.from_bytes(gzip.decompress(z.read(name)))
            yield os.path.basename(name)[:-len('-label.nii.gz')], 'Rib fracture', '0', np.asarray(im.dataobj) > 0, im.affine


def rexg_instances(dataset_json: str, prefix: str = 'valid_') -> list[dict]:
    """Instance list of the scans whose name starts with ``prefix`` (any split): scan, study, finding index
    (= segmentation channel), category, CT-RATE class, evaluable."""
    d = [e for v in json.load(open(dataset_json)).values() for e in v if e['name'].startswith(prefix)]; out = []
    for e in d:
        scan = e['name'].split('.')[0]
        for k, cat in e['categories'].items():
            cls = REXG_CATEGORY_TO_CTRATE.get(cat)
            out.append(dict(scan=scan, study=scan.rsplit('_', 1)[0], finding=k, channel=int(k), category=cat, text=e['findings'].get(k),
                            CTRATE_class=cls, evaluable=cls is not None))
    return out


def rexg(dataset_json: str, seg_dir: str, prefix: str = 'valid_'):
    nib = _nib(); inst: dict = {}
    for r in rexg_instances(dataset_json, prefix):
        if r['evaluable']:
            inst.setdefault(r['scan'], []).append(r)
    for scan, rows in sorted(inst.items()):
        im = nib.load(os.path.join(seg_dir, scan + '.nii.gz')); seg = np.asarray(im.dataobj)
        for r in rows:
            yield scan, r['CTRATE_class'], r['finding'], seg[r['channel']] > 0, im.affine
