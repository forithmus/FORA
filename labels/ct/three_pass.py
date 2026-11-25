"""Three-pass closed-vocabulary finding extraction with Gemma-4-31B-it (Methods, report labels; Box S3).

Per report and protocol class: (1) EXTRACTION against the vocabulary with verbatim quotes, (2) VERIFICATION of every
labelled leaf against the report, (3) RESOLUTION: apply the QC and do one targeted second look. The final lists come from
pass 3; if pass 3 does not parse, pass-1 lists are used with verdict ``p3_unparsed``. With ``repair_truncated`` (final
retry run only) a pass-3 / pass-1 reply cut at the token cap keeps its complete list items (``p3_repaired`` /
``p1_repaired``). Out-of-vocabulary findings (``other``) come from pass 1 only.

Used by the CT training-label extraction (``extract_findings.py``) and by the report-scoring labeler
(``labels/report_scoring``), which differ only in prompt set, vocabularies and context length (config files).
"""
from __future__ import annotations

import json
from pathlib import Path

from labels.llm import (CONTEXT_LENGTH_SAFETY_TOKENS, GREEDY_TEMPERATURE, GREEDY_TOP_P, STATUS_FAIL, STATUS_PASS,
                        STATUS_TOO_LONG, ChatMessages, read_prompt, stage_path, vocab_lines)

STATUS_P3_UNPARSED = "p3_unparsed"
STATUS_P3_REPAIRED = "p3_repaired"
STATUS_P1_REPAIRED = "p1_repaired"
KEYS_P1 = ("asserted", "denied", "uncertain", "other")
KEYS_P3 = ("final_asserted", "final_denied", "final_uncertain")
# Class 14 (CT Liver four-phase) is never routed: protocol classification maps it to class 3.
N_PROTOCOLS = 29


class PromptSet:
    def __init__(self, prompt_dir: str | Path):
        d = stage_path(prompt_dir)
        self.p1_system, self.p1_user = read_prompt(d / "p1_system.txt"), read_prompt(d / "p1_user.txt")
        self.p2_system, self.p2_user = read_prompt(d / "p2_system.txt"), read_prompt(d / "p2_user.txt")
        self.p3_system, self.p3_user = read_prompt(d / "p3_system.txt"), read_prompt(d / "p3_user.txt")


class Protocols:
    """Protocol framing (+ per-protocol leaves); a global vocabulary file replaces the leaves of every protocol."""

    def __init__(self, protocols_path: str | Path, vocabulary_path: str | Path | None = None):
        self.by_class = {int(n): m for n, m in json.loads(stage_path(protocols_path).read_text("utf-8"))["protocols"].items()}
        self.global_leaves = json.loads(stage_path(vocabulary_path).read_text("utf-8")) if vocabulary_path else None

    def load(self, n: int) -> dict:
        m = dict(self.by_class[n])
        if self.global_leaves is not None:
            m["leaves"] = self.global_leaves
        return m


def _hints_block(hints: list[str]) -> str:
    return "\n".join(f"- {h}" for h in hints) if hints else "- (none)"


def _fmt(tmpl: str, mod: dict, vocab: str, hints: str) -> str:
    return tmpl.format(title=mod["title"], region=mod["region"], contrast_note=mod["contrast_note"], hints=hints,
                       vocabulary=vocab)


def p1_messages(P: PromptSet, mod, vocab, hints, findings, impression) -> ChatMessages:
    return [{"role": "system", "content": _fmt(P.p1_system, mod, vocab, hints)},
            {"role": "user", "content": P.p1_user.format(title=mod["title"], findings=findings,
                                                         impression=impression or "(none)")}]


def p2_messages(P: PromptSet, mod, vocab, hints, findings, impression, labelling_json) -> ChatMessages:
    return [{"role": "system", "content": _fmt(P.p2_system, mod, vocab, hints)},
            {"role": "user", "content": P.p2_user.format(findings=findings, impression=impression or "(none)",
                                                         labelling=labelling_json)}]


def p3_messages(P: PromptSet, mod, vocab, hints, findings, impression, p1_json, p2_json) -> ChatMessages:
    # Pass 3's {hints} slot carries the protocol's second-look list (falls back to the hints when empty).
    hints2 = "\n".join(f"- {h}" for h in mod.get("second_look", [])) or hints
    sysmsg = P.p3_system.format(title=mod["title"], region=mod["region"], hints=hints2, vocabulary=vocab)
    user = P.p3_user.format(findings=findings, impression=impression or "(none)", p1_json=p1_json, p2_json=p2_json)
    return [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]


def repair_truncated(s: str):
    """Truncated JSON (generation hit the token cap mid-list): keep every complete list item, close the brackets."""
    i = s.find("{")
    if i < 0:
        return None
    stack, in_str, esc, best = [], False, False, None
    for pos in range(i, len(s)):
        ch = s[pos]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                break
            stack.pop()
            if ch == "}" and len(stack) == 2 and stack[-1] == "[":   # closed one item inside a top-level list
                best = (pos, list(stack))
    if best is None:
        return None
    pos, st = best
    tail = "".join("]" if c == "[" else "}" for c in reversed(st))
    try:
        return json.loads(s[i:pos + 1] + tail)
    except Exception:
        return None


def parse_json(raw: str, repair: bool = False):
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):]
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < 0:
        return None
    try:
        return json.loads(s[i:j + 1])
    except Exception:
        return repair_truncated(s) if repair else None


def _lists(parsed, keys) -> dict:
    out = {}
    for k in keys:
        v = parsed.get(k) if isinstance(parsed, dict) else None
        out[k] = v if isinstance(v, list) else []
    return out


class ThreePass:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.prompts = PromptSet(cfg["prompt_dir"])
        self.protocols = Protocols(cfg["protocols"], cfg.get("vocabulary"))
        self.t1, self.t2, self.t3 = cfg["p1_max_tokens"], cfg["p2_max_tokens"], cfg["p3_max_tokens"]
        self.repair = bool(cfg.get("repair_truncated", False))
        # Pass 3's prompt is pass 1's plus both JSON outputs and its system prompt is never larger than pass 1's, so
        # gating pass 1 on the whole chain's output budget guarantees that pass 3 fits as well.
        self.chain_reserve = self.t1 + self.t2 + self.t3

    def process_modality(self, inferer, mod: dict, rows: list[tuple]) -> dict:
        """rows: [(idx, findings, impression)] of one protocol class -> {idx: result dict}."""
        P = self.prompts
        vocab = vocab_lines(mod["leaves"])
        hints = _hints_block(mod["hints"])
        mk = lambda t: inferer.make_sampling(temperature=GREEDY_TEMPERATURE, top_p=GREEDY_TOP_P, max_tokens=t)  # noqa: E731
        s1, s2, s3 = mk(self.t1), mk(self.t2), mk(self.t3)
        results, p1_msgs, live = {}, [], []
        for idx, find, imp in rows:
            msgs = p1_messages(P, mod, vocab, hints, find, imp)
            if not inferer.prompt_fits(msgs, self.chain_reserve, safety_tokens=CONTEXT_LENGTH_SAFETY_TOKENS)[0]:
                results[idx] = {"verdict": STATUS_TOO_LONG}
            else:
                p1_msgs.append(msgs)
                live.append((idx, find, imp))
        if not live:
            return results
        p1_raw = inferer.chat(p1_msgs, s1)
        p2_raw = inferer.chat([p2_messages(P, mod, vocab, hints, f, i, p1) for (_, f, i), p1 in zip(live, p1_raw)], s2)
        p3_raw = inferer.chat([p3_messages(P, mod, vocab, hints, f, i, p1, p2)
                               for (_, f, i), p1, p2 in zip(live, p1_raw, p2_raw)], s3)
        for (idx, _, _), p1, p3 in zip(live, p1_raw, p3_raw):
            results[idx] = self.resolve(p1, p3)
        return results

    def resolve(self, p1: str, p3: str) -> dict:
        p3p, p1p = parse_json(p3), parse_json(p1)
        other = p1p.get("other", []) if isinstance(p1p, dict) else []
        if p3p is not None:
            f = _lists(p3p, KEYS_P3)
            return {"verdict": STATUS_PASS, "asserted": f["final_asserted"], "denied": f["final_denied"],
                    "uncertain": f["final_uncertain"], "other": other}
        if self.repair and (p3r := parse_json(p3, repair=True)) is not None:
            f = _lists(p3r, KEYS_P3)
            return {"verdict": STATUS_P3_REPAIRED, "asserted": f["final_asserted"], "denied": f["final_denied"],
                    "uncertain": f["final_uncertain"], "other": other}
        if p1p is not None:
            f = _lists(p1p, KEYS_P1)
            return {"verdict": STATUS_P3_UNPARSED, "asserted": f["asserted"], "denied": f["denied"],
                    "uncertain": f["uncertain"], "other": other}
        if self.repair and (p1r := parse_json(p1, repair=True)) is not None:
            f = _lists(p1r, KEYS_P1)
            return {"verdict": STATUS_P1_REPAIRED, "asserted": f["asserted"], "denied": f["denied"],
                    "uncertain": f["uncertain"], "other": p1r.get("other", []) if isinstance(p1r, dict) else []}
        return {"verdict": STATUS_FAIL, "asserted": [], "denied": [], "uncertain": [], "other": []}
