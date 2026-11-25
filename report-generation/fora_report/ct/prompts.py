"""CT writer instruction prompt and training targets (paper: Supplementary Fig. 11, writer prompt).

The writer input is: [visual prefix] + instruction prompt (this module) + diagnostic plan prompt
(``fora.reportgen.plan.diagnostic_plan.DiagnosticPlanBuilder.plan_ids``). The target is ``"Findings:\\n" + Findings
text`` followed by EOS. Both texts are stored verbatim in ``prompts/ct_writer_instruction.txt`` and
``prompts/ct_writer_target_prefix.txt``.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


@lru_cache(maxsize=None)
def _read(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def writer_instruction() -> str:
    return _read("ct_writer_instruction.txt")


def target_prefix() -> str:
    return _read("ct_writer_target_prefix.txt")


def prompt_ids(tokenizer, device, dataset_kind: str = "ct", report_writer_mode: str = "full",
               writer_supported_plan: bool = False) -> torch.Tensor:
    """Chat-templated instruction token ids (one user turn + generation prompt).

    Only the CT full-Findings writer exists; the extra arguments are accepted for call compatibility and must keep
    their defaults."""
    if dataset_kind != "ct" or report_writer_mode != "full" or writer_supported_plan:
        raise ValueError("only the CT full-Findings writer prompt is defined")
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": writer_instruction()}],
        tokenize=True, add_generation_prompt=True, return_dict=True,
    )
    ids = encoded["input_ids"] if hasattr(encoded, "__getitem__") else encoded
    return torch.tensor(ids, dtype=torch.long, device=device)


def target_ids(tokenizer, text: str, maximum: int, device) -> torch.Tensor:
    """``"Findings:\\n" + text`` tokenized, truncated to ``maximum - 1`` tokens, EOS appended."""
    ids = tokenizer(target_prefix() + text, add_special_tokens=False).input_ids
    eos = tokenizer.eos_token_id
    if isinstance(eos, (list, tuple)):
        eos = eos[0]
    ids = ids[: max(1, maximum - 1)]
    if not ids or ids[-1] != eos:
        ids.append(int(eos))
    return torch.tensor(ids, dtype=torch.long, device=device)
