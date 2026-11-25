"""Generation records: one JSON line per study, sharded per worker, resumable, merged into predictions.jsonl.

Record keys: uid, checkpoint, optimizer_step, generation_max_new_tokens, reference_findings, generated_findings, the
plan tiers with probabilities, token counts, and the decoding settings (decode_settings, forced_prefix, phrase_ban).
Reports are scored with the finding-level labeller in `evaluation/`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def plan_entries(plan_builder, selection, descriptions, indices) -> list[dict]:
    out = []
    for i in indices:
        label = plan_builder.names[i]
        if descriptions and descriptions.get(i):
            label += f" ({descriptions[i]})"
        out.append({"class_index": int(i), "label": label, "probability": float(selection.probability[i])})
    return out


def build_record(*, uid, checkpoint_id, optimizer_step, max_new_tokens, reference_findings, generated, plan_builder,
                 selection, descriptions, n_findings, visual_tokens, modality_tokens, extra=None) -> dict:
    top = selection.probability.topk(min(30, selection.probability.numel())).indices.tolist()
    record = {
        "uid": uid, "checkpoint": checkpoint_id, "optimizer_step": optimizer_step,
        "generation_max_new_tokens": max_new_tokens, "reference_findings": reference_findings,
        "writer_mode": "full", "generated_findings": generated,
        "mil_confident": plan_entries(plan_builder, selection, descriptions, selection.confident),
        "mil_possible": plan_entries(plan_builder, selection, descriptions, selection.possible),
        "highest_mil": plan_entries(plan_builder, selection, descriptions, top),
        "finding_tokens_used": int(n_findings), "visual_tokens": int(visual_tokens), "modality_tokens": int(modality_tokens),
    }
    if extra:
        record.update(extra)
    return record


def shard_path(output: Path, shard: int) -> Path:
    return Path(output) / f"records_rank{int(shard):03d}.jsonl"


def read_records(path: Path, checkpoint_id: str, max_new_tokens: int) -> dict:
    """uid -> record of a shard for this checkpoint / max_new_tokens (a torn last line is ignored)."""
    out = {}
    if not Path(path).is_file():
        return out
    with open(path) as handle:
        for line in handle:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("checkpoint") == checkpoint_id and int(r.get("generation_max_new_tokens", -1)) == max_new_tokens:
                out[r["uid"]] = r
    return out


def atomic_write(path: Path, text: str) -> None:
    tmp = Path(f"{path}.partial.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def merge_outputs(output: Path, order: list[str], num_shards: int, checkpoint_id: str, max_new_tokens: int,
                  extra_summary=None):
    """predictions.jsonl (in `order`) + summary.json from all shards; returns (summary, missing uids)."""
    records = {}
    for shard in range(num_shards):
        records.update(read_records(shard_path(output, shard), checkpoint_id, max_new_tokens))
    ordered = [records[u] for u in order if u in records]
    missing = [u for u in order if u not in records]
    with open(Path(output) / "predictions.jsonl.partial", "w") as handle:
        for r in ordered:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(Path(output) / "predictions.jsonl.partial", Path(output) / "predictions.jsonl")
    summary = {
        "count": len(ordered), "selected": len(order), "missing": len(missing),
        "optimizer_step": ordered[0]["optimizer_step"] if ordered else None, "checkpoint": checkpoint_id,
        "generation_max_new_tokens": max_new_tokens, "mil_source": "precomputed_logits", "visual_stream": "dino_b24",
        "dino_token_caps_used": sorted({int(r.get("dino_token_cap", 0)) for r in ordered}),
        "truncated": sum(1 for r in ordered if r.get("truncated")),
    }
    summary.update(extra_summary or {})
    atomic_write(Path(output) / "summary.json", json.dumps(summary, indent=2) + "\n")
    return summary, missing
