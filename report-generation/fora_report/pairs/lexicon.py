"""Keyword organ lexicon and text edits for the plan-based CT DPO pairs (``plan_pairs.py``).

A sentence maps to the set of organs whose pattern it matches; label stems, anatomy terms, word sets and the
insertion / replacement / deletion edits on sentence offsets are shared by the four plan-based pair types.
"""
from __future__ import annotations

import hashlib
import json
import re

ORGANS = {
    "brain": r"cerebr\w*|cerebell\w*|brain\w*|lateral ventricles?|third ventricle|3rd ventricle|4th ventricle|fourth ventricle|"
             r"ventricular system|sulc\w*|gyr\w*|basal ganglia|thalam\w*|pons|bulbus|mesencephal\w*|midbrain|brainstem|"
             r"intracranial|white matter|corona radiata|cisterns?|vermis|posterior fossa|sella\w*|pituitar\w*|"
             r"supratentorial|infratentorial|periventricular|hemispher\w*|centrum semiovale|internal capsule",
    "sinonasal": r"paranasal|(?:maxillary|frontal|sphenoid|ethmoid)\w* sinus\w*|sinus(?:es)?\b(?! rhythm)|nasal|concha\w*|turbinate\w*|"
                 r"ostiomeatal|nasopharyn\w*|adenoid\w*",
    "ear": r"mastoid\w*|middle ear|tympan\w*|ossic\w*|external auditory|cochlea\w*|labyrinth\w*|internal auditory",
    "orbit": r"orbit\w*|globe|optic nerve|ocular|lens\b|retrobulbar",
    "thyroid": r"thyroid\w*",
    "neck": r"parotid\w*|submandibular|salivary|laryn\w*|pharyn\w*(?<!nasopharynx)|vocal cord\w*|epiglott\w*|vallecula\w*|piriform|"
            r"cervical lymph|carotid space|parapharyngeal|tonsil\w*",
    "airway": r"trachea\w*|main bronch\w*|carina\w*|tracheobronch\w*",
    "lung": r"lungs?\b|pulmonary(?! arter\w*| trunk| vein\w*| embol\w*)|pneum\w*|emphysem\w*|bronchiect\w*|atelecta\w*|consolidat\w*|"
            r"ground[- ]glass|interstiti\w*|lingula\w*|(?:upper|middle|lower) lobes?\b(?! of the liver)|parenchyma window|"
            r"peribronch\w*|bulla\w*|bullous",
    "pleura": r"pleura\w*|hydrothorax|pneumothorax",
    "mediastinum": r"mediastin\w*|hil(?:ar|um|a)\b|paratracheal|subcarinal|prevascular|aortopulmonary|thym\w*",
    "heart": r"heart|cardi(?:ac|o\w*|omegaly)|pericard\w*|coronar\w*|atri(?:um|al|a)\b|ventricle\b|valv\w*",
    "aorta": r"aort\w*",
    "pulmonary_vessels": r"pulmonary arter\w*|pulmonary trunk|pulmonary embol\w*|pulmonary vein\w*",
    "vessels": r"vascular|(?<!pulmonary )arter(?:y|ies|ial)|(?<!pulmonary )\bveins?\b|venous|vena cava|aneurysm\w*|"
               r"atheroscl\w*|atherom\w*|thrombo\w*|portal vein|splenic vein|mesenteric (?:artery|vein)",
    "breast": r"breast\w*|axilla\w*|mammar\w*",
    "liver": r"liver|hepat(?!ic flexure)\w*|segment (?:[1-8]|i{1,3}|iv|v|vi{1,3})\b(?! of the lung)",
    "gallbladder": r"gall ?bladder|cholecyst\w*|cholelith\w*",
    "biliary": r"bile ducts?|biliary|choledoch\w*|cystic duct|cbd\b",
    "pancreas": r"pancrea\w*|wirsung\w*",
    "spleen": r"splen\w*|spleen",
    "adrenal": r"adrenal\w*|surrenal\w*",
    "kidney": r"kidneys?|renal|nephr\w*|pelvicalyc\w*|calyc\w*|calice\w*|ureteropelvic|perirenal",
    "ureter": r"ureter\w*",
    "bladder": r"(?<!gall )\bbladder|vesical\b|cystitis",
    "prostate": r"prostat\w*|seminal vesic\w*",
    "female_pelvis": r"uter(?:us|ine)|ovar\w*|adnex\w*|endometri\w*|cervix|vagina\w*|myom\w*",
    "gi": r"stomach|gastr\w*|duoden\w*|jejun\w*|ile(?:um|al)\b|bowel|intestin\w*|colon\w*|sigmoid|rect(?:um|al)\b|c[a]?ec(?:um|al)\b|"
          r"appendi\w*|diverticul\w*|esophag\w*|oesophag\w*|hiatal|hiatus|anal\b|perirectal",
    "peritoneum": r"periton\w*|ascit\w*|mesenter\w*|omentum|omental|free fluid|retroperiton\w*|intra-?abdominal fluid",
    "lymph": r"lymph\w*",
    "bone": r"bone\w*|osseous|vertebr\w*|spine|spinal|discs?\b|intervertebral|ribs?\b|stern\w*|clavic\w*|scapul\w*|sacr\w*|"
            r"iliac bone|femo(?:ral|r)\w* head|femur|humer\w*|osteo\w*|spondyl\w*|lytic|sclerotic|fractur\w*|joint\w*|facet\w*|"
            r"calvar\w*|cranial bone|skull|acetabul\w*|sacroiliac|endplate\w*|schmorl|hemangioma of the vertebra",
    "softtissue": r"soft tissue\w*|subcutaneous|abdominal wall|muscle\w*|inguinal|umbilical|gynecomast\w*",
}
ORGAN_RE = {k: re.compile(r"\b(?:" + v + r")", re.I) for k, v in ORGANS.items()}
NORMAL_RE = re.compile(r"\b(?:normal|natural|unremarkable|within normal limits|regular|homogeneous|clear|preserved|patent|"
                       r"symmetric\w*|no\b|not (?:detected|observed|seen|present)|were not|was not)\b", re.I)


GENERIC = {"lesion", "lesions", "finding", "findings", "disease", "diseases", "abnormal", "abnormality", "change", "changes",
           "syndrome", "left", "right", "bilateral", "other", "acute", "chronic", "with", "without", "type", "related",
           "increased", "decreased", "status", "post", "after", "the", "and", "of", "in", "or", "mass", "masses"}


# pure anatomy / location words (dropped from label stems; disease words such as emphysema are kept)
ANAT_WORDS = {
    "renal", "kidney", "kidneys", "liver", "hepatic", "lung", "lungs", "pulmonary", "cerebral", "cerebellar", "spinal",
    "vertebral", "vertebra", "cardiac", "heart", "aortic", "aorta", "arterial", "artery", "arteries", "venous", "vein",
    "thoracic", "abdominal", "pelvic", "pleural", "mediastinal", "hilar", "splenic", "spleen", "pancreatic", "pancreas",
    "adrenal", "biliary", "bile", "duct", "ducts", "gallbladder", "bladder", "urinary", "ureteral", "ureteric", "prostate",
    "prostatic", "uterine", "ovarian", "gastric", "colonic", "bowel", "intestinal", "esophageal", "oesophageal",
    "thyroid", "nasal", "sinus", "maxillary", "frontal", "ethmoid", "sphenoid", "mastoid", "orbital", "intracranial",
    "bone", "bones", "osseous", "skeletal", "soft", "tissue", "wall", "lobe", "segment", "coronary", "carotid", "breast",
    "lymph", "node", "nodes", "cervical", "lumbar", "sacral", "femoral", "iliac", "inguinal", "portal", "cerebrum",
    "brain", "parenchymal", "cortical", "subcutaneous", "muscle", "chest", "abdomen", "pelvis", "neck", "head",
}


def spell(text: str) -> str:
    return text.casefold().replace("ae", "e").replace("oe", "e")


def label_stems(label: str) -> set:
    out = set()
    for w in re.findall(r"[a-z]+", spell(label)):
        if len(w) < 4 or w in GENERIC or w in ANAT_WORDS:
            continue
        out.add(w[:5])
    return out


def anatomy_terms(text: str) -> set:
    return {m.group(0).casefold()[:7] for r in ORGAN_RE.values() for m in r.finditer(text)}


def organs_of(text: str) -> frozenset:
    return frozenset(k for k, r in ORGAN_RE.items() if r.search(text))


def hk(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def insert_after(target: str, row: dict, sentence: str) -> str:
    end = row["end"]
    return target[:end] + " " + sentence + target[end:]


def replace_span(target: str, row: dict, sentence: str) -> str:
    return target[:row["start"]] + sentence + target[row["end"]:]


def delete_rows(target: str, rows_to_delete: list) -> str:
    out = target
    for row in sorted(rows_to_delete, key=lambda r: -r["start"]):
        left, right = out[:row["start"]], out[row["end"]:]
        out = left.rstrip() + (" " if left.strip() and right.strip() else "") + right.lstrip()
    return out

