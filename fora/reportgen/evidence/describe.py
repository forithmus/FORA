"""Finding descriptions for the diagnostic plan (paper section "Grounded report generation", evidence wording).

Each planned finding is written as ``<finding> (<description>)`` where the description is built from the
evidence foci of the finding (see ``fora.mil.evidence``): distribution pattern, main site with the
position inside the organ, an image-derived measurement, a tally of sites by region when there are many,
and further sites. The anatomical family of a finding (``fora.regions.family_of``) selects the wording of its site.

Location wording: the organ named by the finding itself is used as the location, lung findings keep the
attended lobe(s), side words only on paired / lateral structures. Patient side from the evidence share on the
patient-left half (>= 0.7 one side, both >= 0.25 for a paired organ = bilateral).

Measurement wording: nodules 'about L x S mm (mean diameter M mm)', lymph nodes 'short axis about X mm',
other lesions 'about L x S mm', thickness classes 'thickness about X mm', lumen 'diameter about X mm',
vessels '<segment> diameter about X mm'.
"""
from __future__ import annotations

import math
import re

import numpy as np

from fora.regions.families import BONE_KEYS, LN_STATION, LUNG_KEYS, LUNG_RE, ORGAN_KEYS, ORGAN_OF, family_of  # noqa: F401

PAIRED = {'kidneys': 'kidney', 'adrenals': 'adrenal gland', 'pleura': 'pleural space', 'hila': 'hilum', 'ureters': 'ureter',
          'breast': 'breast', 'axilla': 'axilla', 'groin': 'groin', 'orbit': 'orbit', 'temporal_bone': 'temporal bone',
          'salivary': 'salivary gland', 'shoulder': 'shoulder', 'ribs': 'ribs', 'chest_wall': 'chest wall', 'pelvic_bones_hips': 'hip / pelvic bones',
          'lungs': 'lung', 'neck_vessels': 'neck vessels'}
LOBES = {'lung_rul': 'right upper lobe', 'lung_rml': 'right middle lobe', 'lung_rll': 'right lower lobe', 'lung_lul': 'left upper lobe',
         'lung_lll': 'left lower lobe'}
PLAIN = {'abdominal_vessels': 'abdominal vessels', 'abdominal_wall': 'abdominal wall', 'airway': 'airway', 'aorta_thoracic': 'thoracic aorta',
         'appendix_rlq': 'appendix / right lower quadrant', 'biliary': 'biliary tree', 'bladder': 'urinary bladder', 'bones': 'bones',
         'bowel': 'bowel', 'colon': 'colon', 'duodenum': 'duodenum', 'esophagus': 'esophagus', 'gallbladder': 'gallbladder', 'heart': 'heart',
         'intracranial': 'intracranial', 'liver': 'liver', 'mediastinum': 'mediastinum', 'neck': 'neck', 'oral_dental': 'oral cavity / teeth',
         'pancreas': 'pancreas', 'pelvis': 'pelvis', 'peritoneum': 'peritoneum', 'pharynx_larynx': 'pharynx / larynx', 'portal_venous': 'portal venous system',
         'prostate': 'prostate', 'pulmonary_arteries': 'pulmonary arteries', 'sinonasal': 'paranasal sinuses / nasal cavity', 'skull': 'skull',
         'small_bowel': 'small bowel', 'soft_tissue': 'soft tissues', 'spinal_canal': 'spinal canal', 'spine': 'spine', 'spine_cervical': 'cervical spine',
         'spine_lumbosacral': 'lumbosacral spine', 'spine_thoracic': 'thoracic spine', 'spleen': 'spleen', 'stomach': 'stomach', 'thyroid': 'thyroid',
         'vessels': 'vessels', 'body': None, 'body_minus_lungs': None,
         'pericardium': 'pericardium', 'paraaortic': 'para-aortic region', 'periportal': 'periportal region', 'aorta': 'aorta', 'uterus': 'uterus'}


def side_of(left: float, right: float) -> str:
    if right >= 0.7: return 'right'
    if left >= 0.7: return 'left'
    if left >= 0.25 and right >= 0.25: return 'bilateral'
    return ''


def fmt_mm(v: float) -> str:
    return f'{v:.1f}' if v < 10 else f'{v:.0f}'


def measurement_phrase(mt: int, ycmp: int, is_node: bool, long_mm: float, short_mm: float, ins_mm: float, *, vessel_segment: str = '') -> str:
    """Measurement phrase. mt: 0 lesion, 1 thickness, 2 vessel, 3 lumen; ycmp == 3: report both axes + mean diameter."""
    ok = lambda x: x is not None and math.isfinite(x) and x > 0  # noqa: E731
    if mt == 2 and ok(long_mm):
        return f'{vessel_segment + " " if vessel_segment else ""}diameter about {fmt_mm(long_mm)} mm'
    if mt == 1 and ok(ins_mm): return f'thickness about {fmt_mm(ins_mm)} mm'
    if mt == 3 and ok(ins_mm): return f'diameter about {fmt_mm(ins_mm)} mm'
    if mt != 0 or not ok(long_mm): return ''
    if ycmp == 3 and ok(short_mm): return f'about {fmt_mm(long_mm)} x {fmt_mm(short_mm)} mm (mean diameter {fmt_mm((long_mm + short_mm) / 2)} mm)'
    if is_node and ok(short_mm): return f'short axis about {fmt_mm(short_mm)} mm'
    if ok(short_mm) and 2 <= short_mm < long_mm * 0.95: return f'about {fmt_mm(long_mm)} x {fmt_mm(short_mm)} mm'
    return f'about {fmt_mm(long_mm)} mm'


def measurement_label(cls_name: str, txt: str) -> str:
    """Name what was measured where the class name alone is ambiguous."""
    n = str(cls_name).lower()
    if not txt: return txt
    if 'obstructive uropathy' in n: return 'obstructing calculus ' + txt
    if txt.startswith('diameter') and 'hydronephrosis' in n: return 'renal pelvis ' + txt
    if txt.startswith('diameter') and 'hydroureter' in n: return 'ureter ' + txt
    return txt


# ---- location wording -------------------------------------------------------------------------------------------------
SIDED_PLAIN = {'soft_tissue': 'soft tissues', 'abdominal_wall': 'abdominal wall', 'neck': 'neck', 'pelvis': 'pelvis',
               'pulmonary_arteries': 'pulmonary artery', 'intracranial': 'cerebral hemisphere', 'thyroid': 'thyroid lobe',
               'colon': 'hemicolon'}
NO_SIDE = {'pericardium', 'aorta', 'periportal region', 'liver', 'spleen', 'gallbladder', 'heart', 'pancreas', 'stomach', 'urinary bladder', 'prostate', 'peritoneum', 'uterus'}
PAIRED_SIDED = dict(PAIRED, adnexa='adnexa')
PLURAL = {'axilla': 'axillae', 'hilum': 'hila', 'adnexa': 'adnexa', 'ribs': 'ribs', 'neck vessels': 'neck vessels'}


def location_phrase(cls_name, region_keys, region_shares, left, right, *, min_share=0.3, max_regions=2) -> str:
    name = str(cls_name).lower(); side = side_of(float(left), float(right))
    if name.startswith('bilateral'): side = 'bilateral'
    keys = [str(k) for k, s in zip(region_keys, region_shares) if float(s) >= min_share]
    if 'liver' in keys and 'biliary' in keys: keys.remove('biliary')
    lobes = [k for k in keys if k in LOBES]
    if 'pleura' in name or 'effusion' in name and 'pericard' not in name: lobes = []      # pleural findings: no lobe
    organs = None
    for pat, org in ORGAN_OF:
        if re.search(pat, name): organs = org; break
    if organs is None and re.search(LUNG_RE, name) and (lobes or 'lungs' in keys): organs = []; keys = lobes[:max_regions] or ['lungs']
    elif organs is not None:
        keys = list(organs)
    parts, sided = [], False
    for key in keys:
        if key == 'lungs' and lobes: continue
        if key in LOBES:
            txt = LOBES[key]; sided = True
        elif key in PAIRED_SIDED:
            organ = PAIRED_SIDED[key]
            if side == 'bilateral': txt = f"bilateral {PLURAL.get(organ, organ + 's')}"; sided = True
            elif side: txt = f'{side} {organ}'; sided = True
            else: txt = organ
        elif key in SIDED_PLAIN and side in ('right', 'left'):
            txt = f'{side} {SIDED_PLAIN[key]}'; sided = True
        else:
            txt = SIDED_PLAIN.get(key) if key in SIDED_PLAIN else PLAIN.get(key, key.replace('_', ' '))
            if key == 'colon': txt = 'colon'
            if txt is None: continue
        if txt not in parts: parts.append(txt)
        if len(parts) >= max_regions: break
    if side and not sided and parts and not all(p in NO_SIDE for p in parts) and (side != 'bilateral' or name.startswith('bilateral')):
        parts.insert(0, f'{side}-sided' if side != 'bilateral' else 'bilateral')
    elif side and not parts and side != 'bilateral':
        parts = [f'{side}-sided']
    return ', '.join(parts)


def _within(organ_key, rel, ml, bdist, lbdist):
    """Position words inside the organ (empty when central / uninformative)."""
    if rel is None or any(math.isnan(x) for x in rel): return []
    S, Rr, A = rel; w = []
    zone = 'upper' if S > 0.67 else 'lower' if S < 0.33 else 'mid'
    if organ_key in ('kidneys', 'spleen'):
        w.append({'upper': 'upper pole', 'lower': 'lower pole', 'mid': 'interpolar region'}[zone])
    elif organ_key == 'lungs':
        w.append(f'{zone} zone')
    elif organ_key.startswith('lung_'):
        upper_lobe = organ_key in ('lung_rul', 'lung_lul')
        if zone == 'upper': w.append('apical' if upper_lobe else 'superior' if organ_key != 'lung_rml' else '')
        elif zone == 'lower': w.append('inferior' if upper_lobe else 'basal' if organ_key != 'lung_rml' else '')
        w = [x for x in w if x]
    elif organ_key == 'liver':
        if S > 0.8: w.append('dome')
        elif S < 0.2: w.append('inferior tip')
    elif zone != 'mid' and organ_key not in ('pleura', 'peritoneum', 'bones', 'soft_tissue', 'body', 'vessels', 'abdominal_vessels'):
        w.append(f'{zone} part')
    if A > 0.67: w.append('anterior')
    elif A < 0.33: w.append('posterior')
    if ml is not None and not math.isnan(ml) and organ_key in ('kidneys', 'adrenals', 'breast', 'axilla') or (organ_key.startswith('lung_') or organ_key == 'lungs'):
        if ml is not None and not math.isnan(ml):
            if ml > 0.75: w.append('lateral')
            elif ml < 0.25: w.append('medial' if not organ_key.startswith('lung') and organ_key != 'lungs' else 'paramediastinal')
    if (organ_key.startswith('lung_') or organ_key == 'lungs') and lbdist is not None and not math.isnan(lbdist):
        if lbdist <= 10: w.append('subpleural')
        elif lbdist >= 35: w.append('central')
    if organ_key == 'liver' and bdist is not None and not math.isnan(bdist) and bdist <= 8: w.append('subcapsular')
    return w


# ---- anatomical families (lymph-node stations, bones, lungs, organs) -------------------------------------------------------------
LN_SIDED = {'hila', 'neck', 'axilla', 'groin', 'pelvis', 'mediastinum'}
BONE_NAME = {'ribs': 'ribs', 'spine_cervical': 'cervical spine', 'spine_thoracic': 'thoracic spine', 'spine_lumbosacral': 'lumbosacral spine',
             'pelvic_bones_hips': 'pelvic bones / hip', 'skull': 'skull', 'shoulder': 'shoulder girdle', 'temporal_bone': 'temporal bone', 'spine': 'spine', 'bones': 'bones'}
GENERIC = {'lungs', 'bones', 'spine', 'vessels', 'soft_tissue', 'body', 'body_minus_lungs', 'peritoneum', 'pelvis', 'chest_wall', 'abdominal_wall', 'mediastinum'}


def pick_key(fam_keys, regall, rnames, min_share=0.1):
    ra = {k: float(regall[j]) for j, k in enumerate(rnames)}
    cand = [k for k in (fam_keys or [k for k in rnames if k not in ('body', 'body_minus_lungs')]) if ra.get(k, 0) >= (min_share if fam_keys else 0.3)]
    if not cand: return None
    return sorted(cand, key=lambda k: (k in GENERIC, -round(ra[k], 1)))[0]


def name_key(fam, key, left, cls_name):
    side = 'left' if left >= 0.7 else 'right' if left <= 0.3 else ''
    if fam == 'ln': st = LN_STATION.get(key, key); return f'{side} {st}'.strip() if key in LN_SIDED and side else st
    if key in BONE_NAME: b = BONE_NAME[key]; return f'{side} {b}' if side and key in ('ribs', 'pelvic_bones_hips', 'shoulder', 'temporal_bone') else b
    return location_phrase(cls_name, [key], [1.0], left, 1 - left, max_regions=1)


# ---- descriptions from the paper-style evidence region ------------------------------------------------------------------------
REGION_NAME = {'lung_rul': 'right upper lobe', 'lung_rml': 'right middle lobe', 'lung_rll': 'right lower lobe', 'lung_lul': 'left upper lobe', 'lung_lll': 'left lower lobe'}


def describe_finding(cls_name, fo, rnames, measurement='', fallback=''):
    """Pattern (HDR50 region size); main site (+ position); measured lesion; sites by region; other sites; regions involved from the
    HDR80 evidence mass. ``fo`` is one foci record of ``fora.mil.evidence.foci`` (keys share, ctr, regall, left, rel, ml,
    bdist, lbdist, gshare80, left80, ncell, n5, tally)."""
    name = str(cls_name).lower(); share = np.asarray(fo['share'], float); nf = int((share > 0).sum())
    if nf == 0: return '; '.join(x for x in (fallback, measurement) if x)
    fam, fkeys = family_of(cls_name); g80 = np.asarray(fo['gshare80'], float)
    inv = [(rnames[j], g80[j]) for j in range(len(rnames)) if g80[j] >= 0.1 and (not fkeys or rnames[j] in fkeys) and rnames[j] not in ('body', 'body_minus_lungs')]
    if any(k.startswith('lung_') for k, _ in inv): inv = [(k, c) for k, c in inv if k.startswith('lung_')]
    inv.sort(key=lambda t: -t[1])
    l80 = float(fo['left80']) if not math.isnan(float(fo['left80'])) else 0.5
    lobe_l = sum(c for k, c in inv if k in ('lung_lul', 'lung_lll')); lobe_r = sum(c for k, c in inv if k in ('lung_rul', 'lung_rml', 'lung_rll'))
    bil = name.startswith('bilateral') or (lobe_l >= 0.15 and lobe_r >= 0.15) or (not inv or not inv[0][0].startswith('lung_')) and 0.15 <= l80 <= 0.85
    ncell = int(fo['ncell']); n_sites = int((share >= 0.1).sum()); n_all = max(n_sites, int(fo['n5']))
    if ncell >= 150: pattern = 'diffuse' + (', bilateral' if bil else '')
    elif n_sites >= 2 or n_all >= 3: pattern = f'multifocal, {n_all} sites' + (', bilateral' if bil else '')
    else: pattern = 'focal' + (', bilateral' if name.startswith('bilateral') else '')
    no_pos = pattern.startswith('diffuse') or fam in ('pleura', 'peritoneum', 'pericardium')

    def site_text(i):
        k = pick_key(fkeys, np.asarray(fo['regall'][i], float), rnames)
        if k is None: return ''
        t = name_key(fam, k, float(fo['left'][i]), cls_name)
        if no_pos: return t
        w = _within(k, list(map(float, fo['rel'][i])), float(fo['ml'][i]), float(fo['bdist'][i]), float(fo['lbdist'][i]))
        return ', '.join([t] + w)

    order = list(np.argsort(-share)[:nf]); main = site_text(order[0]) or fallback
    parts = [pattern]
    if pattern.startswith('diffuse') and len(inv) >= 2:
        nm = lambda k: REGION_NAME.get(k, name_key(fam, k, l80, cls_name))  # noqa: E731
        parts.append(f'mainly {nm(inv[0][0])}; also ' + ', '.join(nm(k) for k, _ in inv[1:4]))
    else:
        parts.append(main)
    if measurement:
        plain = measurement.startswith(('thickness', 'diameter', 'renal pelvis', 'ureter', 'main', 'thoracic', 'ascending', 'aortic', 'abdominal', 'descending', 'midline', 'right kidney', 'left kidney', 'liver', 'spleen'))
        parts.append(measurement if plain else 'measured lesion ' + measurement)
    tally = np.asarray(fo.get('tally', np.zeros(len(rnames))), int)
    if n_all >= 4 and tally.sum() >= 4:   # many sites: where ALL of them are (every site >= 2 % of the evidence)
        tl = sorted(((rnames[j], int(tally[j])) for j in np.flatnonzero(tally)), key=lambda t: -t[1])[:6]
        parts.append('sites by region: ' + ', '.join(f"{REGION_NAME.get(k, name_key(fam, k, 0.5, cls_name))} x{n}" for k, n in tl))
    if not pattern.startswith('diffuse'):
        others = []
        for i in order[1:4]:
            if share[i] < 0.1: continue
            t = site_text(i)
            if t and t != main and t not in others: others.append(t)
        if others: parts.append('also ' + '; '.join(others[:2]))
    return '; '.join(p for p in parts if p)
