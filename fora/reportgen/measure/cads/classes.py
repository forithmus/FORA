"""Measurement classes for the report-generation CADS measurer (indices into the CT finding vocabulary).

kind:
  vessel  (segment, calibration class): segment = the aorta part the class refers to
          asc = ascending (valve -> arch start), root = first 30 mm of the ascending, thor = max(asc, arch, desc thoracic),
          abd = abdominal (below the lung bases), all = max over the aorta, pa = pulmonary trunk (mid-trunk window)
          calibration class = the training-split choice (measure variant + mult/add calibration) used for it
  liver / spleen / kidney / midline / evans (Evans index) / ctr (cardiothoracic ratio)
"""
VESSEL = {
    55: ('asc', '55'),     # Ascending aorta dilatation
    56: ('asc', '56'),     # Ascending aortic aneurysm
    43: ('root', '43'),    # Aortic root dilatation
    678: ('thor', '678'),  # Thoracic aortic dilatation
    677: ('thor', '677'),  # Thoracic aortic aneurysm (own calibration entry chosen on the training split)
    40: ('all', '678'),    # Aortic dilatation (unspecified segment): max over the aorta, thoracic-dilatation calibration
    39: ('all', '39'),     # Aortic aneurysm
    1: ('abd', '1'),       # Abdominal aortic aneurysm
    549: ('pa', '549'),    # Pulmonary trunk dilatation
    529: ('pa', '549'),    # Pulmonary artery enlargement
    528: ('pa', '549'),    # Pulmonary artery ectasia
}
LIVER = {279: 'Hepatomegaly', 267: 'Hepatic atrophy', 280: 'Hepatosplenomegaly'}
SPLEEN = {633: 'Splenomegaly', 280: 'Hepatosplenomegaly'}
KIDNEY = {561: 'Renal atrophy'}
MIDLINE = {336: 'Intracranial mass effect', 663: 'Subfalcine herniation', 334: 'Intracranial herniation', 1006: 'Transtentorial herniation'}
EVANS = {737: 'Ventriculomegaly', 293: 'Hydrocephalus', 428: 'Obstructive hydrocephalus', 786: 'Communicating hydrocephalus',
         89: 'Brain atrophy', 120: 'Cerebral atrophy'}          # Evans index
CTR = {110: 'Cardiomegaly'}                                      # cardiothoracic ratio
RG_CLASSES = set(VESSEL) | set(LIVER) | set(SPLEEN) | set(KIDNEY) | set(MIDLINE) | set(EVANS) | set(CTR)
