"""Shared text helpers for the CT DPO pair builders (paper: "Preference optimization", synthetic rejected reports).

Sentence splitter, normalisation, deterministic hashing, and the regular-expression screens and attribute vocabularies
used by every pair type (base, attribute and plan-based pairs). Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import re

PAIR_SCHEMA = "ct_report_preference_pair_v1"
EMPTY_FINDINGS = "No findings were reported."
TARGET_PREFIX = "Findings:\n"

# Same splitter as the CT SFT dataset (sentence record spans of the report target);
# kept here so the pair builders stay importable without torch or h5py.
_SENTENCE = re.compile(r"\S(?:.*?\S)?(?:[.!?](?=\s|$)|(?=\n)|$)", re.DOTALL)


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    text = str(text)
    out = [(m.group(0).strip(), m.start(), m.end()) for m in _SENTENCE.finditer(text)
           if m.group(0).strip()]
    return out or ([(text.strip(), 0, len(text))] if text.strip() else [])


def norm(value: str) -> str:
    return " ".join(str(value).casefold().split()).rstrip(" .;,")


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hkey(*parts) -> str:
    return sha(json.dumps(parts, ensure_ascii=False, sort_keys=True))


def jdump(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# Generic language screens (never create a diagnosis or a negative label).
# Dependency is checked where it matters: the sentence FOLLOWING a deleted
# sentence must not refer back to it.
NEXT_DEPENDENCY = re.compile(
    r"\b(?:it|its|this|these|those|they|their|them|such|former|latter|respectively|"
    r"aforementioned|aforedescribed|above[- ]mentioned|mentioned above|described above|"
    r"the same|also|as well|similar(?:ly)?|likewise|another|other|additional(?:ly)?|"
    r"further(?:more)?|moreover|in addition|remaining|besides|again)\b", re.I)
NONCURRENT = re.compile(
    r"\b(?:compared (?:to|with)|comparison|previous(?:ly)?|prior (?:exam\w*|stud\w*|ct|imaging|scan)|"
    r"in the interval|interval\w*|regress\w*|progress\w*|(?:in|de)creased in (?:size|number|extent)|"
    r"unchanged|stable|no significant change|newly|new(?:ly)? (?:developed|appeared)|has developed|"
    r"have developed|history of|underwent|recommend\w*|follow[- ]?up|clinical correlation|"
    r"correlat\w+ (?:is|with)|suggested to|advised)\b", re.I)
COORDINATION = re.compile(r"\b(?:and|or|but|whereas|while|although|however|except|excluding|with|"
                          r"associated|because|versus|vs|due to|secondary to)\b", re.I)
FINITE_VERB = re.compile(r"\b(?:is|are|was|were|has|have|had|shows?|showed|demonstrates?|appears?|"
                         r"seen|observed|noted|detected|identified|present|can|cannot|may|might|could)\b", re.I)
ADDITIVE_REFERENCE = re.compile(r"\b(?:also|additionally|furthermore|similarly|likewise|too|either|"
                                r"other|another|same)\b", re.I)
# Donor sentences must be self-contained observations, not interpretations of "the findings".
DONOR_EXCLUDE = re.compile(r"\b(?:findings?|described|etiology|aetiology|considered|differential|diagnos\w*|"
                           r"favou?r\w*|primarily|first|evaluated|interpreted|recommend\w*)\b", re.I)
# Denials restricted to a side/part are not whole-study denials.
SCOPE_LIMITED = re.compile(
    r"\b(?:right|left|upper|lower|middle|segment\w*|lobe\w*|pole|apical|basal|anterior|posterior|"
    r"medial|lateral|proximal|distal|on the|in the [a-z]+ (?:part|portion|aspect))\b", re.I)


def unary_exclusion(text: str) -> str | None:
    """Strict single-clause gate: reason string if the sentence is not one independent current-image clause."""
    if not text.endswith(".") or len(text.split()) < 3 or not FINITE_VERB.search(text):
        return "not_independent_complete_sentence"
    if any(c in text for c in ";,:()\n") or COORDINATION.search(text):
        return "coordinated_or_multiclause_sentence"
    if NEXT_DEPENDENCY.search(text):
        return "anaphoric_or_context_dependent_sentence"
    if NONCURRENT.search(text):
        return "noncurrent_or_nonimage_language"
    return None


# Attribute vocabularies (candidate pre-selection and the CPU verifier).
ATTRIBUTE_TYPES = ("laterality", "location", "severity", "size", "number", "certainty")
SIDE_WORDS = {"right", "left", "bilateral", "bilaterally", "both", "unilateral", "unilaterally",
              "either", "sided", "side", "sides", "contralateral", "ipsilateral"}
LOCATION_WORDS = {
    "upper", "middle", "lower", "lobe", "lobes", "lingula", "lingular", "segment", "segments",
    "segmental", "subsegmental", "apical", "apex", "apicoposterior", "anterior", "posterior",
    "superior", "inferior", "medial", "lateral", "basal", "base", "bases", "anteromedial",
    "laterobasal", "posterobasal", "mediobasal", "anterobasal", "anterolateral", "posterolateral",
    "caudate", "dome", "hilum", "hilar", "perihilar", "pole", "interpolar", "cortex", "cortical",
    "medulla", "medullary", "pelvis", "head", "neck", "body", "tail", "uncinate", "proximal",
    "distal", "mid", "central", "peripheral", "periphery", "subpleural", "paramediastinal",
    "paravertebral", "fundus", "antrum", "cardia", "pylorus", "frontal", "parietal", "temporal",
    "occipital", "insular", "cerebellar", "thalamic", "paratracheal", "subcarinal", "prevascular",
    "aortopulmonary", "axillary", "supraclavicular", "paraaortic", "para-aortic", "mesenteric",
    "inguinal", "iliac", "retroperitoneal", "periportal", "perigastric", "retrocrural",
    "ascending", "descending", "transverse", "sigmoid", "cecum", "rectum", "duodenum", "jejunum",
    "ileum", "zone", "region", "aspect", "portion", "part", "wall", "margin", "level",
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "of", "the", "in", "at",
}
SEVERITY_WORDS = {"mild", "mildly", "moderate", "moderately", "severe", "severely", "marked",
                  "markedly", "massive", "minimal", "minimally", "slight", "slightly", "significant",
                  "significantly", "prominent", "extensive", "trace", "small", "large", "huge",
                  "subtle", "gross", "grossly", "pronounced", "advanced", "early", "minor",
                  "considerable", "substantial", "grade", "i", "ii", "iii", "iv", "1", "2", "3", "4",
                  "a", "an", "amount", "degree", "very", "much", "limited", "diffuse", "focal",
                  "widespread", "mild-moderate", "moderate-severe", "mild-to-moderate",
                  "moderate-to-severe", "to"}
NUMBER_WORDS = {"single", "solitary", "a", "an", "one", "two", "three", "four", "five", "six",
                "seven", "eight", "nine", "ten", "several", "multiple", "numerous", "few", "many",
                "some", "couple", "countless", "innumerable", "occasional", "isolated", "few", "a few",
                "number", "of", "plural", "multifocal", "unifocal", "scattered"}
HEDGE_WORDS = {"possible", "possibly", "probable", "probably", "likely", "unlikely", "suspicious",
               "suspected", "suspicion", "suggestive", "suggest", "suggests", "suggesting",
               "consistent", "compatible", "may", "might", "could", "can", "cannot", "be", "excluded",
               "ruled", "out", "questionable", "equivocal", "favor", "favour", "favored", "favoured",
               "represent", "represents", "representing", "indeterminate", "presumably", "presumed",
               "considered", "thought", "definite", "definitely", "evident", "clearly", "clear",
               "diagnostic", "characteristic", "typical", "certainly", "certain", "for", "of", "with",
               "to", "is", "are", "be", "not", "concerning", "appearance", "appears", "appear",
               "potential", "potentially", "doubtful", "uncertain", "undetermined", "differential",
               "includes", "include", "and", "the", "a", "an", "as", "seen", "evaluated",
               "interpreted", "observed", "noted", "consistent", "finding", "findings", "which", "that",
               "there", "it", "this", "suspect"}
AGREEMENT_WORDS = {"is", "are", "was", "were", "has", "have", "a", "an", "one", "this", "these",
                   "that", "those", "it", "they", "its", "their", "measures", "measure", "measuring",
                   "each", "both"}
SIDE_RE = re.compile(r"\b(?:right|left|bilateral(?:ly)?|both|unilateral(?:ly)?)\b", re.I)
LOCATION_RE = re.compile(
    r"\b(?:(?:upper|middle|lower)\s+(?:lobe|pole|zone|lobes|poles|zones)|lingula\w*|"
    r"segment\s*(?:\d{1,2}|[ivx]{1,4})\b|(?:apical|apicoposterior|posterior|anterior|superior|inferior|"
    r"medial|lateral|laterobasal|posterobasal|mediobasal|anterobasal)\s+(?:basal\s+)?segment|"
    r"caudate|(?:head|neck|body|tail|uncinate process)\s+of\s+the\s+pancreas|pancreatic\s+(?:head|neck|body|tail)|"
    r"(?:frontal|parietal|temporal|occipital|insular)\s+lobe|"
    r"(?:para|sub)?tracheal|subcarinal|prevascular|aortopulmonary|supraclavicular|axillary|"
    r"para-?aortic|retrocrural|periportal|inguinal|(?:[ctlsd]\d{1,2})(?:\s*-\s*[ctlsd]?\d{1,2})?)\b", re.I)
SEVERITY_RE = re.compile(
    r"\b(?:mild(?:ly)?|moderate(?:ly)?|sever(?:e|ely)|marked(?:ly)?|massive|minimal(?:ly)?|slight(?:ly)?|"
    r"significant(?:ly)?|extensive|trace|subtle|pronounced|advanced|grade\s+(?:[1-4]|i{1,3}v?|iv))\b", re.I)
SIZE_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?(?:\s*(?:x|×|\*)\s*\d+(?:[.,]\d+)?){0,2}\s*(?:mm|cm|ml|cc|millimet(?:er|re)s?|"
    r"centimet(?:er|re)s?)\b", re.I)
NUMBER_RE = re.compile(
    r"\b(?:single|solitary|several|multiple|numerous|few|a few|countless|innumerable|"
    r"two|three|four|five|six|seven|eight|nine|ten|a couple of|multifocal|"
    r"\d+\s+(?:nodules?|lesions?|cysts?|lymph nodes?|nodes?|stones?|calculi|foci|masses|metastases|"
    r"fractures?|polyps?|diverticul\w+|plaques?|hemangiomas?))\b", re.I)
HEDGE_RE = re.compile(
    r"\b(?:possibl[ey]|probabl[ey]|likely|suspicious|suspected|suggestive|suggest(?:s|ing)?|"
    r"consistent with|compatible with|cannot be (?:ruled out|excluded)|could not be excluded|"
    r"questionable|equivocal|favou?r(?:ed|s)?|may (?:be|represent|indicate|reflect|correspond)|"
    r"might|could (?:be|represent)|indeterminate|presum(?:ed|ably)|potential(?:ly)?|"
    r"doubtful|concerning for|not excluded)\b", re.I)
# Fixed-side anatomy: changing the side word gives a non-existent structure.
INVALID_ANATOMY = re.compile(
    r"\b(?:left\s+middle\s+lobe|left\s+(?:minor|horizontal)\s+fissure|right\s+lingul\w*|"
    r"right\s+(?:anterior\s+descending|circumflex)|right\s+main\s+coronary|"
    r"right\s+(?:spleen|splenic\s+flexure)|left\s+(?:gallbladder|appendix|c[a]?ecum|ascending\s+colon|hepatic\s+flexure)|"
    r"right\s+(?:sigmoid|descending\s+colon)|"
    r"bilateral\s+(?:liver|spleen|heart|aorta|pancreas|stomach|bladder|urinary\s+bladder|uterus|prostate|"
    r"trachea|esophagus|oesophagus|gallbladder|sigmoid|rectum|cecum|caecum|appendix|thyroid\s+isthmus)|"
    r"(?:right|left)\s+(?:trachea|esophagus|oesophagus|thyroid\s+isthmus|vermis|pons|sella)|"
    r"segment\s*(?:9|1[0-9]|ix|x)\s+of\s+the\s+liver|"
    r"(?:right|left)\s+(?:hepatic|liver)\s+lobe\s+segment\s*(?:9|1[0-9]))\b", re.I)
PROTECTED_SIDE = re.compile(
    r"\b(?:left\s+anterior\s+descending|left\s+circumflex|left\s+main|right\s+coronary\s+artery|"
    r"right\s+middle\s+lobe|right\s+(?:minor|horizontal)\s+fissure|left\s+lingul\w*)\b", re.I)
_TOKEN = re.compile(r"\d+(?:[.,]\d+)?|[^\W\d_]+(?:-[^\W\d_]+)*|[^\w\s]|_")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


def attribute_spans(text: str, state: str) -> dict[str, list[str]]:
    found = {}
    protected = [m.span() for m in PROTECTED_SIDE.finditer(text)]
    free_sides = [m.group(0) for m in SIDE_RE.finditer(text)
                  if not any(a <= m.start() < b for a, b in protected)]
    if free_sides:
        found["laterality"] = free_sides
    loc = [m.group(0) for m in LOCATION_RE.finditer(text)]
    if loc:
        found["location"] = loc
    sev = [m.group(0) for m in SEVERITY_RE.finditer(text)]
    if sev:
        found["severity"] = sev
    size = [m.group(0) for m in SIZE_RE.finditer(text)]
    if size:
        found["size"] = size
    num = [m.group(0) for m in NUMBER_RE.finditer(text)]
    if num:
        found["number"] = num
    hedges = [m.group(0) for m in HEDGE_RE.finditer(text)]
    if hedges and state == "uncertain":
        found["certainty"] = hedges  # hedged -> definite corruption
    elif not hedges and state == "asserted":
        found["certainty"] = ["<definite statement>"]  # definite -> hedged corruption
    return found

