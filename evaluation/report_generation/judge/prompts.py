"""LLM-judge prompts and verdict parsing (paper Methods 4.11, Supplementary Fig. 13a).

single: reference + one candidate. The judge compares abnormal findings only (diseases, lesions, pathological
        changes, devices, post-surgical changes) and returns the number of matched findings and, for six error kinds,
        the numbers of clinically significant and insignificant errors: false finding, missed finding, wrong
        location, wrong severity, false comparison, omitted comparison.
        score = matched / (matched + significant errors); a report with no matched finding scores 1 when it has
        no significant error and 0 otherwise.
pair:   reference + candidate A + candidate B -> which is clinically more accurate (A / B / tie); run in both orders.
No model names appear in any prompt.
"""
import json
import re

MAX_CHARS = 12000   # longer reports are degenerate loops; cut so prompts fit
CATS = ['false_finding', 'missed_finding', 'wrong_location', 'wrong_severity', 'false_comparison', 'omitted_comparison']

SYSTEM = ("You are an expert radiologist evaluating the clinical accuracy of a candidate radiology report against the "
          "reference report written by the radiologist who read the same examination. Judge clinical content only. Ignore "
          "writing style, report length, section order, template phrasing and synonyms that mean the same thing.")

SINGLE = """Modality: {modality} ({region}).

REFERENCE REPORT (ground truth):
<<<
{ref}
>>>

CANDIDATE REPORT:
<<<
{gen}
>>>

Compare the candidate with the reference on ABNORMAL findings only: diseases, lesions, pathological changes, devices, post-surgical changes. Ignore every normal statement and every statement that something is absent; they are neither matched findings nor errors. Exception: when the reference reports an abnormal finding and the candidate states it is absent or normal, count one missed_finding.

Error categories (abnormal findings only):
(a) false_finding: the candidate reports an abnormal finding that is not in the reference.
(b) missed_finding: an abnormal finding in the reference is absent from the candidate or is called normal or absent by it.
(c) wrong_location: an abnormal finding is reported in both but with the wrong anatomical location, side or position.
(d) wrong_severity: an abnormal finding is reported in both but with the wrong severity, size or extent.
(e) false_comparison: the candidate states a change from a prior study (new, increased, stable, resolved) that the reference does not state.
(f) omitted_comparison: the reference states a change from a prior study that the candidate omits.

Count each reference finding at most once. For each category give the number of clinically SIGNIFICANT errors (could change diagnosis or management) and the number of clinically INSIGNIFICANT errors. Also give matched_findings, the number of abnormal findings correctly reported in both (0 when neither report has an abnormal finding). List each significant error in at most 12 words. An empty candidate misses every abnormal reference finding.

Answer with JSON only, exactly in this form:
{{"matched_findings": 0, "significant": {{"false_finding": 0, "missed_finding": 0, "wrong_location": 0, "wrong_severity": 0, "false_comparison": 0, "omitted_comparison": 0}}, "insignificant": {{"false_finding": 0, "missed_finding": 0, "wrong_location": 0, "wrong_severity": 0, "false_comparison": 0, "omitted_comparison": 0}}, "significant_errors": ["..."]}}"""

PAIR = """Modality: {modality} ({region}).

REFERENCE REPORT (ground truth):
<<<
{ref}
>>>

CANDIDATE REPORT A:
<<<
{a}
>>>

CANDIDATE REPORT B:
<<<
{b}
>>>

Which candidate agrees better with the reference on clinically important content: findings that are present, findings that are absent, their location and severity? Hallucinated findings and missed findings both count against a report. Ignore style, length and template wording. Answer "tie" only when both are clinically equivalent.

Answer with JSON only: {{"better": "A" | "B" | "tie", "reason": "<at most 20 words>"}}"""


def _cut(t):
    t = (t or '').strip()
    return (t[:MAX_CHARS] + ' [...]') if len(t) > MAX_CHARS else (t or '(empty report)')


def messages(item: dict) -> list[dict]:
    if item['kind'] == 'single':
        u = SINGLE.format(modality=item['modality'], region=item['region'], ref=_cut(item['ref']), gen=_cut(item['gen']))
    else:
        u = PAIR.format(modality=item['modality'], region=item['region'], ref=_cut(item['ref']), a=_cut(item['a']), b=_cut(item['b']))
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': u}]


def _json(text):
    t = re.sub(r'^.*?</think>', '', text or '', flags=re.S).strip()
    t = re.sub(r'^```(?:json)?\s*|\s*```$', '', t.strip())
    i, j = t.find('{'), t.rfind('}')
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return None


def _int(x):
    try:
        return max(0, int(round(float(x))))
    except (TypeError, ValueError):
        return None


def report_score(matched: int, significant: int) -> float:
    if matched > 0:
        return matched / (matched + significant)
    return 1.0 if significant == 0 else 0.0


def parse(item: dict, text: str):
    """Parsed verdict or None (unparsable)."""
    d = _json(text)
    if not isinstance(d, dict):
        return None
    if item['kind'] == 'pair':
        b = {'A': 'A', 'B': 'B', 'TIE': 'tie'}.get(str(d.get('better', '')).strip().upper())
        return None if b is None else {'better': b, 'reason': str(d.get('reason', ''))[:300]}
    m = _int(d.get('matched_findings'))
    sig, ins = d.get('significant') or {}, d.get('insignificant') or {}
    if m is None or not isinstance(sig, dict) or not isinstance(ins, dict):
        return None
    s = {c: _int(sig.get(c, 0)) for c in CATS}; n = {c: _int(ins.get(c, 0)) for c in CATS}
    if None in s.values() or None in n.values():
        return None
    tot = sum(s.values())
    return {'matched': m, 'sig': s, 'insig': n, 'sig_total': tot, 'score': report_score(m, tot),
            'sig_list': [str(x)[:200] for x in (d.get('significant_errors') or [])][:30]}
