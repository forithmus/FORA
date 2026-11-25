"""Plan location text for the chest X-ray writer (paper section "Grounded report generation", X-ray plan).

Evidence is the MIL leave-one-out token contribution to the bag logit, never the attention weights alone:
    loo[t, c] = w[t, c] * (s_c L[t, c] - mu_c) / (1 - w[t, c]),   mu_c = sum_t w[t, c] s_c L[t, c]
Per study and class (``CXRVisualStream.encode(..., location=True)``):
  * positive mass E = sum of max(loo, 0) over the tokens of frontal films (view kind 0) only;
  * frac_right = share of E in the image-left half of each content-cropped frontal grid (image left = patient right on
    PA/AP films; the middle column of an odd grid counts half);
  * zone shares = share of E in the upper / middle / lower third of the content-cropped grid rows.
Rules:
  * E < MIN_MASS or no frontal film -> no location;
  * side = right if frac_right >= 0.7, left if frac_right <= 0.3, else bilateral;
  * zones = the fewest zones (largest first) holding >= 60% of E; all three -> 'all zones';
  * style per class: 'side_zone' -> '<side> <zones>', 'side' -> 'right side' / 'left side' / 'bilateral',
    'none' (global, midline and non-lateral classes, devices, post-surgical, technical, variants) -> no location.
The rules were fixed before any outcome was looked at.
"""
from __future__ import annotations

import json
import sqlite3

import numpy as np

MIN_MASS = 0.05
SIDE_CUT = 0.7
ZONE_COVER = 0.6
ZONES = ('upper', 'mid', 'lower')

GROUP_STYLE = {
    'Pulmonary parenchyma & airspace': 'side_zone',
    'Pulmonary edema & vascular congestion': 'side_zone',
    'Airways & chronic lung disease': 'side_zone',
    'Pleura': 'side_zone',
    'Diaphragm & upper abdomen': 'none',
    'Mediastinum & hila': 'none',
    'Cardiac & pericardial': 'none',
    'Great vessels & pulmonary vasculature': 'none',
    'Bones & chest wall': 'none',
    'Soft tissues & breast': 'side',
    'Support devices, tubes & lines': 'none',
    'Post-surgical & treatment changes': 'none',
    'Technical & image quality': 'none',
    'Anatomical variants': 'none',
    'Extractor extras': 'none',
}
CLASS_STYLE = {   # overrides of the group default
    'Silhouette sign': 'side',
    'COPD (chronic obstructive pulmonary disease)': 'none',
    'Pulmonary hyperinflation (hyperexpansion)': 'none',
    'Rib splaying / horizontal ribs / widened intercostal spaces': 'none',
    'Flattened hemidiaphragm(s)': 'side',
    'Blunting of the costophrenic angle': 'side',
    'Apical pleural cap': 'side',
    'Apical (small) pneumothorax': 'side',
    'Elevated hemidiaphragm': 'side',
    'Diaphragmatic eventration': 'side',
    'Hilar enlargement': 'side',
    'Hilar lymphadenopathy': 'side',
    'Hilar congestion': 'side',
    'Cardiophrenic angle mass': 'side',
    'Clavicle fracture': 'side',
    'Rib fracture, acute': 'side',
    'Rib fracture, healed (with callus)': 'side',
    'Proximal humeral fracture': 'side',
    'Lytic bone lesion': 'side',
    'Sclerotic (blastic) bone lesion': 'side',
    'Bone metastasis': 'side',
    'Glenohumeral (shoulder) osteoarthritis': 'side',
}


def style_table(classes, groups):
    return {c: CLASS_STYLE.get(c, GROUP_STYLE.get(groups.get(c, ''), 'none')) for c in classes}


def side_of(frac_right):
    return 'right' if frac_right >= SIDE_CUT else 'left' if frac_right <= 1 - SIDE_CUT else 'bilateral'


def zones_of(z):
    order = sorted(range(3), key=lambda k: -z[k])
    keep, cum = [], 0.0
    for k in order:
        keep.append(k)
        cum += z[k]
        if cum >= ZONE_COVER:
            break
    return sorted(keep)


def describe(style, mass, frac_right, z):
    """Location text or None. z = zone shares (upper, mid, lower)."""
    if style == 'none' or not (mass >= MIN_MASS):
        return None
    side = side_of(frac_right)
    if style == 'side':
        return {'right': 'right side', 'left': 'left side', 'bilateral': 'bilateral'}[side]
    ks = zones_of(z)
    if len(ks) == 3:
        ztxt = 'all zones'
    else:
        names = [ZONES[k] for k in ks]
        ztxt = (' and '.join(names)) + (' zones' if len(names) > 1 else ' zone')
    if side == 'bilateral':
        return 'bilateral, all zones' if len(ks) == 3 else f'bilateral, mainly {ztxt}'
    return f'{side} {ztxt}'


class LocationStore:
    """locations_<split>.npz files (``fora_report.cxr.dump_locations``) -> per study {class index: text}."""

    def __init__(self, paths, classes, groups):
        self.classes = list(classes)
        st = style_table(self.classes, groups)
        self.style = [st[c] for c in self.classes]
        self.row, self.parts = {}, []
        for p in [x for x in str(paths).split(',') if x]:
            z = np.load(p, allow_pickle=False)
            if [str(c) for c in z['classes'].tolist()] != self.classes:
                raise ValueError(f'{p}: class order differs')
            part = dict(mass=np.asarray(z['mass'], np.float32), fr=np.asarray(z['frac_right'], np.float32),
                        zone=np.asarray(z['zone'], np.float32))
            k = len(self.parts)
            self.parts.append(part)
            for i, u in enumerate(z['ids'].tolist()):
                self.row[str(u)] = (k, i)

    def texts(self, uid, indices=None):
        if str(uid) not in self.row:
            raise KeyError(f'no location descriptor for {uid}')
        k, i = self.row[str(uid)]
        p = self.parts[k]
        out = {}
        for c in (range(len(self.classes)) if indices is None else indices):
            t = describe(self.style[c], float(p['mass'][i, c]), float(p['fr'][i, c]), p['zone'][i, c].tolist())
            if t:
                out[int(c)] = t
        return out


def load_groups(label_db: str) -> dict:
    """Class -> anatomical group from the X-ray label database (labels stage, metadata key 'vocabulary')."""
    con = sqlite3.connect(f'file:{label_db}?mode=ro', uri=True)
    return json.loads(con.execute("select value from metadata where key='vocabulary'").fetchone()[0])['groups']


def open_store(paths, classes, label_db: str):
    """Comma-separated npz list -> LocationStore, or None when no location files are given."""
    return LocationStore(paths, classes, load_groups(label_db)) if paths else None
