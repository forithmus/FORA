"""Forced report opener for external CT cohorts (paper section "Grounded report generation", decoding paragraph).

On the external cohorts (CT-RATE: preset 'chest', Merlin: preset 'abdomen') the first words of the report are fixed so that the writer starts in the single-examination
style of the cohort: CT-RATE (chest) "Findings:\\nThe trachea and both main bronchi", Merlin (abdomen)
"Findings:\\nThe contour, size,". FORA val uses no opener. ``auto_opener`` maps the protocols of a new study to these
presets (chest-only protocols -> 'chest', abdomen / pelvis-only protocols -> 'abdomen', otherwise 'none'); the paper did not
evaluate this mapping on mixed protocols.

Mechanics: the opener tokens (Gemma-4 tokenizer, no special tokens = exactly the tokens the writer emits after the
turn header) are appended to the prompt as text embeddings (attention 1, token type 0), the writer generates the
continuation with `max_new_tokens` reduced by the opener length (same total cap), and the opener ids are prepended to
the output so the decoded report starts with the opener.
"""
from __future__ import annotations

import torch

OPENERS = {
    "none": None,
    "chest": "Findings:\nThe trachea and both main bronchi",
    "abdomen": "Findings:\nThe contour, size,",
}


CHEST_TYPES = frozenset({0, 4, 10})                  # chest-only protocols of the 29-class protocol taxonomy
ABDOMEN_TYPES = frozenset({2, 5, 6, 7, 14, 17})      # abdomen / pelvis-only protocols


def auto_opener(study_type_ids) -> str:
    """Opener preset of a study from its protocol classes."""
    types = set(int(t) for t in study_type_ids)
    if types and types <= CHEST_TYPES:
        return "chest"
    if types and types <= ABDOMEN_TYPES:
        return "abdomen"
    return "none"


def opener_text(name_or_text: str | None) -> str | None:
    """Preset name ('none' | 'chest' | 'abdomen') or a literal opener text."""
    if name_or_text is None:
        return None
    return OPENERS.get(name_or_text, name_or_text) or None


def append_opener(embeddings: torch.Tensor, attention: torch.Tensor, token_types: torch.Tensor | None,
                  opener_ids: list[int], embedding_table: torch.nn.Embedding):
    """Append the opener embeddings to the prompt; returns (embeddings, attention, token_types)."""
    ids = torch.tensor([opener_ids], dtype=torch.long, device=embedding_table.weight.device)
    with torch.autocast(device_type=embeddings.device.type, dtype=torch.bfloat16, enabled=embeddings.device.type == "cuda"):
        extra = embedding_table(ids)
    embeddings = torch.cat((embeddings, extra.to(device=embeddings.device, dtype=embeddings.dtype)), dim=1)
    n = len(opener_ids)
    attention = torch.cat((attention, torch.ones((attention.shape[0], n), dtype=attention.dtype, device=attention.device)), 1)
    if token_types is not None:
        token_types = torch.cat(
            (token_types, torch.zeros((token_types.shape[0], n), dtype=token_types.dtype, device=token_types.device)), 1)
    return embeddings, attention, token_types


def prepend_opener(output_ids: torch.Tensor, opener_ids: list[int]) -> torch.Tensor:
    """Generated ids (new tokens only) -> opener ids + generated ids, per row."""
    head = torch.tensor([opener_ids] * output_ids.shape[0], dtype=output_ids.dtype, device=output_ids.device)
    return torch.cat((head, output_ids), dim=1)
