"""Anatomical families of findings, used to restrict evidence maps and foci (Methods, "Evidence maps and foci").

`family_of(class name)` returns (family, region keys) or (None, None). When a finding has a family and that family's
regions hold evidence, the focus search is restricted to tokens in those regions (lung findings -> lung tokens, lymph
nodes -> nodal stations, organ findings -> the organ and its neighbours). Matching is by regular expression on the
lower-case class name, first match wins. The lymph-node rule (names with lymph / nodal / node but not nodule) comes
first, then ORGAN_OF, then the lung rule.
"""
from __future__ import annotations

import re

LN_STATION = {'hila': 'hilar', 'mediastinum': 'mediastinal', 'neck': 'cervical', 'axilla': 'axillary', 'groin': 'inguinal',
              'abdominal_vessels': 'para-aortic / retroperitoneal', 'pelvis': 'pelvic', 'peritoneum': 'mesenteric',
              'portal_venous': 'periportal'}
BONE_KEYS = ['ribs', 'spine_cervical', 'spine_thoracic', 'spine_lumbosacral', 'pelvic_bones_hips', 'skull', 'shoulder',
             'temporal_bone', 'spine', 'bones']
LUNG_KEYS = ['lung_rul', 'lung_rml', 'lung_rll', 'lung_lul', 'lung_lll', 'lungs']
ORGAN_KEYS = {
    'kidneys': ['kidneys'], 'ureters': ['ureters'], 'adrenals': ['adrenals'], 'gallbladder': ['gallbladder', 'biliary'],
    'liver': ['liver', 'biliary', 'portal_venous'], 'periportal': ['portal_venous', 'biliary', 'liver'], 'spleen': ['spleen'],
    'thyroid': ['thyroid'], 'breast': ['breast'], 'axilla': ['axilla'], 'groin': ['groin'],
    'neck': ['neck', 'neck_vessels', 'pharynx_larynx', 'salivary'], 'hila': ['hila'],
    'mediastinum': ['mediastinum', 'hila', 'esophagus', 'airway'], 'pleura': ['pleura'], 'pericardium': ['heart'],
    'bladder': ['bladder'], 'prostate': ['prostate'], 'pancreas': ['pancreas'],
    'peritoneum': ['peritoneum', 'bowel', 'colon', 'small_bowel'], 'paraaortic': ['abdominal_vessels'],
    'aorta': ['aorta_thoracic', 'abdominal_vessels'], 'uterus': ['pelvis', 'bladder'], 'adnexa': ['pelvis'], 'bones': BONE_KEYS,
}
ORGAN_OF = [  # (regex on the lower-case class name, organ keys); first match wins
    (r'adrenal', ['adrenals']), (r'hydroureteronephrosis', ['kidneys', 'ureters']), (r'ureter', ['ureters']),
    (r'renal|kidney|nephr|pyel|angiomyolipoma|perirenal|calyc|caliec', ['kidneys']), (r'adrenal', ['adrenals']),
    (r'gallbladder|cholecyst|cholelith', ['gallbladder']), (r'periportal|porta hepatis', ['periportal']),
    (r'hepat|liver|portal vein', ['liver']),
    (r'splen', ['spleen']), (r'thyroid', ['thyroid']), (r'breast|gynecomastia', ['breast']), (r'axilla', ['axilla']),
    (r'inguinal|groin', ['groin']), (r'cervical lymph|neck', ['neck']), (r'hilar', ['hila']),
    (r'mediastin|paratracheal|subcarinal', ['mediastinum']),
    (r'pericard', ['pericardium']), (r'pleura|hydrothorax|pneumothorax|empyema|effusion', ['pleura']),
    (r'bladder|vesical|urachal', ['bladder']), (r'prostat', ['prostate']), (r'pancrea', ['pancreas']),
    (r'\bbone|osseous|lytic|sclerotic|blastic|fracture|rib\b|vertebr|spondyl|spine|disc\b|osteo|arthr|joint|scolio|kyph', ['bones']),
    (r'ascites|peritone', ['peritoneum']), (r'para.?aortic|retroperitoneal lymph', ['paraaortic']), (r'aort', ['aorta']),
    (r'adnex|ovar', ['adnexa']), (r'endometri|uterus|uterine|myoma|fibroid', ['uterus']),
]
LUNG_RE = r'lung|pulmonary|lobe|nodule|consolidation|atelectasis|ground|emphysema|bronch|opacity|lymphangitis|septal|fibro|infiltrat'


def family_of(cls_name: str):
    """(family name, region keys) of a finding, or (None, None) if it has no anatomical family."""
    n = str(cls_name).lower()
    if re.search(r'lymph|nodal|node', n) and not re.search(r'nodule', n):
        return 'ln', list(LN_STATION)
    for pat, org in ORGAN_OF:
        if re.search(pat, n):
            if org and org[0] in ORGAN_KEYS:
                return org[0], [k for o in org for k in ORGAN_KEYS.get(o, [o])]
            return None, None
    if re.search(LUNG_RE, n):
        return 'lung', LUNG_KEYS
    return None, None


def family_bits(cls_name: str, region_names) -> int:
    """OR of the region bits of a finding's family (0 if it has no family)."""
    _, keys = family_of(cls_name)
    out = 0
    for k in keys or []:
        if k in region_names:
            out |= 1 << list(region_names).index(k)
    return out
