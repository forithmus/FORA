"""Prior-examination phrase ban for CT report decoding (paper section "Grounded report generation", decoding paragraph).

The writer never sees a prior scan, yet FORA training reports are often follow-up reports, so the writer sometimes
describes comparisons with an earlier examination. At decoding time these token sequences are forbidden through the
HF `generate(bad_words_ids=...)` processor (-inf on the last token of a banned sequence when its prefix was just
generated; all other logits unchanged).

A ban list (``ban_lists/*.json`` next to this module) holds
  * `phrases`: words / phrases, each banned in four surface forms (leading space or not, first letter upper or lower),
  * `openers_after_header`: report openers banned only directly after the header `"Findings:\\n"`,
  * `years`: the calendar years [first, last], banned as their digit-token sequences (Gemma-4 writes digits as single tokens),
  * `bad_words_ids`: the resulting Gemma-4 token sequences (sorted, unique).
`build_bad_words_ids` rebuilds `bad_words_ids` from the other fields; with the Gemma-4-31B-it tokenizer the rebuild
reproduces all shipped lists exactly (`python -m fora.reportgen.decoding.bans verify prior_comparison old_imaging comparative`).

Shipped lists and where the paper uses them:
  prior_comparison  (prior_comparison_phrases.json)              CT-RATE val, 73 of 1,564 studies
  old_imaging       (prior_comparison_phrases_old_imaging.json)  CT-RATE val, the other 1,491 studies
  comparative       (prior_comparison_phrases_comparative.json)  Merlin test
FORA val is decoded without a ban.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

BAN_DIR = Path(__file__).resolve().parent / "ban_lists"


def variants(phrase: str) -> list[str]:
    """The four surface forms of a phrase: with / without a leading space x first letter upper / lower."""
    out = set()
    for first in {phrase[0].upper(), phrase[0].lower()}:
        form = first + phrase[1:]
        out.add(" " + form)
        out.add(form)
    return sorted(out)


def build_bad_words_ids(phrases: Iterable[str], openers: Iterable[str], header: str, years: Sequence,
                        encode: Callable[[str], list[int]]) -> list[list[int]]:
    """Token sequences of all phrase variants + header-anchored openers + years, sorted and unique.
    `encode` = tokenizer.encode(text, add_special_tokens=False)."""
    seqs = {tuple(encode(v)) for phrase in phrases for v in variants(phrase)}
    seqs |= {tuple(encode(header + opener)) for opener in openers}
    seqs |= {tuple(encode(str(year))) for year in range(int(years[0]), int(years[1]) + 1)}
    return [list(s) for s in sorted(seqs)]


SHIPPED = {"prior_comparison": "prior_comparison_phrases.json",
           "old_imaging": "prior_comparison_phrases_old_imaging.json",
           "comparative": "prior_comparison_phrases_comparative.json"}


def resolve(name_or_path: str) -> Path | None:
    """'none' -> None; a shipped list name (see SHIPPED) -> its file; anything else is a path."""
    if not name_or_path or name_or_path == "none":
        return None
    if name_or_path in SHIPPED:
        return BAN_DIR / SHIPPED[name_or_path]
    return Path(name_or_path)


def load_ban_list(name_or_path: str) -> tuple[list[list[int]] | None, dict | None]:
    """(bad_words_ids, info) for a ban list, or (None, None) for 'none'. info is stored in every record."""
    path = resolve(name_or_path)
    if path is None:
        return None, None
    raw = path.read_bytes()
    data = json.loads(raw)
    ids = [[int(t) for t in seq] for seq in data["bad_words_ids"]]
    if not ids or any(not seq or min(seq) <= 0 for seq in ids):
        raise RuntimeError(f"{path}: bad_words_ids must be non-empty lists of positive token ids")
    info = {"list": path.name, "sha256": hashlib.sha256(raw).hexdigest()[:16], "sequences": len(ids),
            "phrases": len(data.get("phrases", []))}
    return ids, info


def _verify(args) -> int:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)   # noqa: E731
    status = 0
    for name in args.lists:
        data = json.loads(resolve(name).read_text())
        rebuilt = build_bad_words_ids(data["phrases"], data["openers_after_header"], data["header"], data["years"], encode)
        same = rebuilt == data["bad_words_ids"]
        print(f"{name}: {len(rebuilt)} sequences, identical to stored list: {same}")
        status |= int(not same)
    return status


def _build(args) -> int:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)   # noqa: E731
    data = json.loads(Path(args.source).read_text())
    data["bad_words_ids"] = build_bad_words_ids(data["phrases"], data["openers_after_header"], data["header"],
                                                data["years"], encode)
    Path(args.output).write_text(json.dumps(data, indent=1))
    print(f"{args.output}: {len(data['bad_words_ids'])} sequences")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify", help="rebuild bad_words_ids of shipped lists and compare")
    v.add_argument("--tokenizer", default="google/gemma-4-31B-it")
    v.add_argument("lists", nargs="+", help="shipped list names or paths")
    b = sub.add_parser("build", help="(re)build bad_words_ids of a phrase file")
    b.add_argument("--tokenizer", default="google/gemma-4-31B-it")
    b.add_argument("source")
    b.add_argument("output")
    args = parser.parse_args(argv)
    return _verify(args) if args.command == "verify" else _build(args)


if __name__ == "__main__":
    raise SystemExit(main())
