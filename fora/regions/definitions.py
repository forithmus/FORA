"""The 63 routed CADS regions (Methods, "Organ routing").

Every region is a union of CADS labels ('551:5' = CADS task 551, label 5), optionally combined with other regions,
then dilated by `dilate_mm` (world mm, in-plane and z) on a 1.5 x 2 x 2 mm work grid before projection onto the
3 x 8 x 8 mm token grid. Derived regions use simple operations: `minus` / `and_` other regions, z slabs bounded by
landmarks (`z_above`, `z_below`, `z_between`), `erode_mm`, `shell_mm`, `within='body'`, left/right and
anterior/posterior halves. The semantics are implemented by `fora.regions.builder.Builder`.

REGION_NAMES is the emitted bit order (bit i of a token's uint64 = REGION_NAMES[i]). It is the sorted set of regions
referenced by the class routes; the 7 helper regions in HELPER_REGIONS are only computed internally. The order is
fixed: the MIL checkpoints, the class-region bit tables and the inference bundle depend on it.

Fallback at routing time (per study, per class; fora.mil): a class with fewer than MIN_TOKENS routed tokens in a view
uses the whole view.
"""

MIN_TOKENS = 8

# ---- CADS label ids (task:label) ----
LUNG_LOBES={'lung_rul':'551:15','lung_rml':'551:16','lung_rll':'551:17','lung_lul':'551:13','lung_lll':'551:14'}
VERT_C=[f'552:{k}' for k in range(18,25)];VERT_T=[f'552:{k}' for k in range(6,18)];VERT_L=[f'552:{k}' for k in range(1,6)]
RIBS=[f'555:{k}' for k in range(1,25)]
HEART=['553:3','553:4','553:5','553:6','553:7','556:3']

REGIONS={
 # head
 'intracranial':dict(cads=['553:9','557:1','557:2','557:3','557:8','558:5','558:24','558:29'],dilate_mm=6),
 'skull':dict(cads=['557:6','557:7','557:4'],dilate_mm=6),
 'orbit':dict(cads=['557:5','558:12','558:13','558:14','558:15','558:16','558:17','558:25','558:26'],dilate_mm=16),
 'sinonasal':dict(cads=['553:18','558:7','557:5','558:12','558:13'],dilate_mm=24,minus=['intracranial']),   # face+oral+eyes: maxillary, ethmoid, frontal, sphenoid
 'temporal_bone':dict(cads=['558:8','558:9'],dilate_mm=30,minus=['intracranial']),
 'oral_dental':dict(cads=['558:4','558:6','558:7','558:23'],dilate_mm=12),
 'pharynx_larynx':dict(cads=['556:2','558:3','558:10','558:11','558:21','558:22'],dilate_mm=12),
 'salivary':dict(cads=['558:18','558:19','558:27','558:28'],dilate_mm=8),
 'thyroid':dict(cads=['558:20'],dilate_mm=12),
 'neck':dict(slab_of=VERT_C,within='body',minus=['intracranial']),          # all body tokens at C1-C7 level
 'neck_vessels':dict(cads=['558:1','558:2']+VERT_C,dilate_mm=12),
 # chest
 **{k:dict(cads=[v],dilate_mm=8) for k,v in LUNG_LOBES.items()},
 'lungs':dict(cads=list(LUNG_LOBES.values()),dilate_mm=4),
 'pleura':dict(cads=['559:4'],regions=['lungs_shell'],minus=['lungs_core','mediastinum','heart']),
 'lungs_core':dict(cads=list(LUNG_LOBES.values()),erode_mm=8),                 # lung interior; pleura = cavity/shell minus this
 'lungs_shell':dict(cads=list(LUNG_LOBES.values()),shell_mm=(12,8)),         # dilate 12 mm minus erode 8 mm
 'airway':dict(cads=['553:2'],dilate_mm=8),
 'mediastinum':dict(cads=['559:9','559:7','553:1','553:2','553:8'],regions=['aorta_thoracic'],dilate_mm=8),
 'hila':dict(cads=['553:8'],dilate_mm=20,and_=['lungs_or_mediastinum']),
 'lungs_or_mediastinum':dict(cads=list(LUNG_LOBES.values())+['559:9'],dilate_mm=8),
 'heart':dict(cads=HEART+['559:7'],dilate_mm=8),
 'aorta_thoracic':dict(cads=['551:7'],dilate_mm=8,z_above='lungs_base'),
 'aorta_abdominal':dict(cads=['551:7'],dilate_mm=8,z_below='lungs_base'),
 'pulmonary_arteries':dict(cads=['553:8'],dilate_mm=8),
 'esophagus':dict(cads=['553:1','558:11'],dilate_mm=8),
 'breast':dict(cads=['556:9','556:10','559:8'],dilate_mm=12),
 'chest_wall':dict(cads=RIBS+['556:11','554:3','554:4','554:5','554:6'],dilate_mm=12),
 'axilla':dict(cads=['554:1','554:2','554:3','554:4'],dilate_mm=40,within='body',minus=['lungs','pleura']),
 # abdomen
 'liver':dict(cads=['551:5'],dilate_mm=8),
 'gallbladder':dict(cads=['551:4'],dilate_mm=8),
 'biliary':dict(cads=['551:5','551:4','551:10','553:15'],dilate_mm=8),
 'spleen':dict(cads=['551:1'],dilate_mm=8),
 'pancreas':dict(cads=['551:10'],dilate_mm=12),
 'adrenals':dict(cads=['551:11','551:12'],dilate_mm=12),
 'kidneys':dict(cads=['551:2','551:3'],dilate_mm=12),
 'ureters':dict(cads=['556:12','556:13','554:20','554:21'],dilate_mm=16,and_=['abdominopelvic']),  # psoas corridor
 'bladder':dict(cads=['553:17'],dilate_mm=8),
 'prostate':dict(cads=['556:7','556:8'],dilate_mm=8),
 'stomach':dict(cads=['551:6'],dilate_mm=8),
 'duodenum':dict(cads=['553:15'],dilate_mm=8),
 'small_bowel':dict(cads=['553:14','553:15'],dilate_mm=8),
 'colon':dict(cads=['553:16','556:5','556:6'],dilate_mm=8),
 'rectum':dict(cads=['556:6','556:5'],dilate_mm=12),
 'bowel':dict(cads=['553:14','553:15','553:16','556:4','556:5','556:6'],dilate_mm=8),
 'appendix_rlq':dict(cads=['553:16','553:14'],dilate_mm=12,right_half=True,z_below='kidney_right_centre'),
 'peritoneum':dict(cads=['559:3'],dilate_mm=4),
 'abdominopelvic':dict(cads=['559:3'],dilate_mm=24),
 'pelvis':dict(cads=['559:3','553:17','556:5','556:6','556:7','556:8'],dilate_mm=8,z_below='hip_top'),
 'abdominal_vessels':dict(cads=['551:7','551:8','551:9','553:10','553:11','553:12','553:13'],dilate_mm=12,z_below='lungs_base'),
 'portal_venous':dict(cads=['551:9'],dilate_mm=12),
 'abdominal_wall':dict(cads=['556:14','556:15','559:1','559:2'],dilate_mm=8,z_between=('lowest_rib','pubis'),anterior_half=True),
 'groin':dict(special='groin',minus=['bladder','rectum']),   # extraperitoneal lower anterior wall in front of pubis/femoral heads (whole inguinal/femoral canal path) + scrotal/labial strip; routes with pelvis
 # musculoskeletal
 'spine':dict(cads=VERT_C+VERT_T+VERT_L+['554:11','556:1','559:10'],dilate_mm=8),
 'spine_cervical':dict(cads=VERT_C,dilate_mm=8),
 'spine_thoracic':dict(cads=VERT_T,dilate_mm=8),
 'spine_lumbosacral':dict(cads=VERT_L+['554:11'],dilate_mm=8),
 'spinal_canal':dict(cads=['556:1','559:10'],dilate_mm=6),
 'ribs':dict(cads=RIBS,dilate_mm=8),
 'sternum':dict(cads=['556:11'],dilate_mm=8),
 'shoulder':dict(cads=['554:1','554:2','554:3','554:4','554:5','554:6'],dilate_mm=16),
 'pelvic_bones_hips':dict(cads=['554:7','554:8','554:9','554:10','554:11'],dilate_mm=16),
 'bones':dict(cads=['559:5']+VERT_C+VERT_T+VERT_L+RIBS+['554:1','554:2','554:3','554:4','554:5','554:6','554:7','554:8','554:9','554:10','554:11','556:11','557:6','557:7','558:4'],dilate_mm=4),
 'soft_tissue':dict(cads=['559:1','559:2','554:12','554:13','554:14','554:15','554:16','554:17','554:18','554:19','554:20','554:21'],dilate_mm=4),
 'vessels':dict(cads=['551:7','551:8','551:9','553:8','553:10','553:11','553:12','553:13','558:1','558:2'],dilate_mm=8),
 # generic
 'body':dict(all_body=True),
 'body_minus_lungs':dict(all_body=True,minus=['lungs']),
}


REGION_NAMES = [
    "abdominal_vessels",
    "abdominal_wall",
    "adrenals",
    "airway",
    "aorta_thoracic",
    "appendix_rlq",
    "axilla",
    "biliary",
    "bladder",
    "body",
    "body_minus_lungs",
    "bones",
    "bowel",
    "breast",
    "chest_wall",
    "colon",
    "duodenum",
    "esophagus",
    "gallbladder",
    "groin",
    "heart",
    "hila",
    "intracranial",
    "kidneys",
    "liver",
    "lung_lll",
    "lung_lul",
    "lung_rll",
    "lung_rml",
    "lung_rul",
    "lungs",
    "mediastinum",
    "neck",
    "neck_vessels",
    "oral_dental",
    "orbit",
    "pancreas",
    "pelvic_bones_hips",
    "pelvis",
    "peritoneum",
    "pharynx_larynx",
    "pleura",
    "portal_venous",
    "prostate",
    "pulmonary_arteries",
    "ribs",
    "salivary",
    "shoulder",
    "sinonasal",
    "skull",
    "small_bowel",
    "soft_tissue",
    "spinal_canal",
    "spine",
    "spine_cervical",
    "spine_lumbosacral",
    "spine_thoracic",
    "spleen",
    "stomach",
    "temporal_bone",
    "thyroid",
    "ureters",
    "vessels",
]
HELPER_REGIONS = sorted(set(REGIONS) - set(REGION_NAMES))
assert len(REGION_NAMES) == 63 and set(REGION_NAMES) <= set(REGIONS) and len(HELPER_REGIONS) == 7
