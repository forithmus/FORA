"""Visual stream of the report writer: the classifier's own encoder tokens (Methods, "Report generation").

The visual tokens are produced exactly as the finding classifier sees them:

    block-23 token cache of all views of the study (int8, data-preprocessing)
      -> frozen block 24 of the classifier checkpoint (bf16, chunked over crops, no grad)
      -> x = cat((y - mean) / rms, position_features * position_scale)                       [n, 1024 + 6 * freqs]

This is also the space of the per-finding evidence tokens. ``DinoCTReportGenerator`` resamples these tokens with
512 latent queries (resampler width 1024), adds zero-initialised per-token view embeddings (window kind and series
ordinal) and appends trainable study-type tokens.

Placement: the cache of a study lives on the node that staged its classifier data rank (ranks 4n..4n+3 on node n,
marker ``STAGED_node<n>.json``). A rank on node n with local GPU g owns data rank 4n+g; its study set is the
classifier assignment of that rank filtered by the split column of the report index.

Training subsamples a study's tokens uniformly without replacement to ``cap`` (524,288) after block 24 has run on
all crops; the draw is a deterministic function of (seed, study uid, epoch). Inference uses all tokens (cap 0).
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .model import CTReportGenerator
from .resampler import QueryResampler

VIEW_KINDS = ("full", "native", "lung_mip_10mm")  # anything else -> index len(VIEW_KINDS) ("other")
N_VIEW_KINDS = len(VIEW_KINDS) + 1
MAX_SERIES = 32
DEFAULT_TRAIN_TOKEN_CAP = 524288
VISUAL_STREAM = "dino_b24"   # value of the checkpoint metadata key ``arguments.visual_stream``


def view_kind_id(kind: str) -> int:
    return VIEW_KINDS.index(kind) if kind in VIEW_KINDS else len(VIEW_KINDS)


def token_dim(config: dict) -> int:
    return 1024 + 6 * int(config["training"].get("position_freqs", 4))


# ----------------------------------------------------------------------------------------- placement
def detect_data_node(fingerprint: str, local_root: Path, data_node: str = "") -> int:
    """Node index n whose classifier ranks 4n..4n+3 are staged on this host (``data_node`` overrides)."""
    if str(data_node).strip():
        return int(data_node)
    marks = sorted(glob.glob(str(Path(local_root) / fingerprint / "STAGED_node*.json")))
    nodes = set()
    for path in marks:
        with open(path) as handle:
            nodes.add(int(json.load(handle)["node"]))
    if len(nodes) != 1:
        raise RuntimeError(
            f"cannot identify the staged token cache on this host: markers={marks} (expected exactly one "
            f"STAGED_node<n>.json under {Path(local_root) / fingerprint}; pass data_node to override)")
    return nodes.pop()


def data_rank_for(node: int, local_rank: int, gpus_per_node: int = 4) -> int:
    if not 0 <= local_rank < gpus_per_node:
        raise ValueError(f"local_rank {local_rank} outside [0,{gpus_per_node})")
    return int(node) * int(gpus_per_node) + int(local_rank)


def check_unique_data_ranks(all_data_ranks) -> None:
    ranks = list(map(int, all_data_ranks))
    if len(set(ranks)) != len(ranks):
        dup = sorted({r for r in ranks if ranks.count(r) > 1})
        raise RuntimeError(f"two trainer ranks own the same data rank(s) {dup}: node markers collide")


def read_index_splits(index_path: str, uids=None) -> dict:
    """report_uid -> split ('train'/'val') of the report index (protocol_class 0)."""
    import sqlite3
    connection = sqlite3.connect(f"file:{Path(index_path).resolve()}?mode=ro&immutable=1", uri=True)
    try:
        rows = connection.execute("SELECT report_uid, split FROM samples WHERE protocol_class=0").fetchall()
    finally:
        connection.close()
    wanted = None if uids is None else set(map(str, uids))
    return {str(u): str(s) for u, s in rows if wanted is None or str(u) in wanted}


def load_cases(cases_path: str, case_ids) -> dict:
    """case_id -> case dict from the cases jsonl (one JSON object per line with an ``"id"`` field)."""
    need = set(case_ids)
    cases = {}
    with open(cases_path) as handle:
        for line in handle:
            i = line.find('"id": "')
            cid = line[i + 7: line.index('"', i + 7)]
            if cid in need:
                cases[cid] = json.loads(line)
                if len(cases) == len(need):
                    break
    missing = need - set(cases)
    if missing:
        raise RuntimeError(f"{len(missing)} assigned cases missing from {cases_path}: {sorted(missing)[:3]}")
    return cases


def local_report_cases(config: dict, data_rank: int, index_splits: dict, split: str, assign=None) -> dict:
    """report_uid -> case for this data rank's node-local studies whose index split == ``split``.

    The classifier assignment's train and val lists are pooled; the report index decides the split.
    """
    A = assign if assign is not None else json.loads(Path(config["assign"]).read_text())
    if A["fingerprint"] != config["fingerprint"]:
        raise RuntimeError("assignment fingerprint differs from the classifier config")
    if not 0 <= data_rank < int(A["ranks"]):
        raise RuntimeError(f"data rank {data_rank} outside the {A['ranks']}-rank assignment")
    ids = list(A["assignments"]["train"][data_rank]) + list(A["assignments"]["val"][data_rank])
    cases = load_cases(config["cases"], ids)
    out = {}
    for cid in ids:
        case = cases[cid]
        uid = str(case["report"])
        if index_splits.get(uid) == split:
            out[uid] = case
    return out


def case_token_count(case: dict) -> int:
    return int(sum(int(v["n_tokens"]) for v in case["views"]))


# ----------------------------------------------------------------------------------------- host side
def to_shared(views: list[dict]) -> list[dict]:
    """numpy views -> torch tensors (shared memory across the DataLoader boundary); uint16 travels as int16."""
    out = []
    for v in views:
        d = {}
        for k, x in v.items():
            if isinstance(x, np.ndarray):
                d[k] = ("u16" if x.dtype == np.uint16 else "arr",
                        torch.from_numpy(np.ascontiguousarray(x.view(np.int16) if x.dtype == np.uint16 else x)))
            else:
                d[k] = ("obj", x)
        out.append(d)
    return out


def from_shared(views: list[dict]) -> list[dict]:
    return [{k: (t.numpy().view(np.uint16) if kind == "u16" else t.numpy() if kind == "arr" else t)
             for k, (kind, t) in v.items()} for v in views]


def prepare_dino_image(case: dict, mil_config: dict, cache_root: Path) -> dict:
    """Host-side image of one study: all cached views (crop-major tokens + position features) + view ids.
    ``cache_root`` is the directory holding the study's block-23 series files."""
    from fora.encoder.cache import load_case
    views = load_case(case, Path(cache_root), int(mil_config["training"].get("position_freqs", 4)))
    return {
        "views": to_shared(views),
        "view_kind": torch.tensor([view_kind_id(v["window"]["kind"]) for v in case["views"]], dtype=torch.long),
        "view_series": torch.tensor([min(int(v.get("series", 0)), MAX_SERIES - 1) for v in case["views"]],
                                    dtype=torch.long),
        "case_id": str(case["id"]),
        "n_tokens": case_token_count(case),
    }


# ----------------------------------------------------------------------------------------- GPU stream
def subsample_mask(total: int, cap: int, seed_key: str, device) -> torch.Tensor | None:
    """Uniform without-replacement keep mask of exactly ``cap`` of ``total`` tokens (None = keep all)."""
    if cap <= 0 or total <= cap:
        return None
    seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "little") & ((1 << 63) - 1)
    generator = torch.Generator().manual_seed(seed)
    keep = torch.randperm(total, generator=generator)[:cap]
    mask = torch.zeros(total, dtype=torch.bool)
    mask[keep] = True
    return mask.to(device)


class DinoVisualStream:
    """Frozen classifier block 24 over the block-23 cache -> classifier input tokens [n, token_dim]."""

    def __init__(self, mil_config_path: str, mil_checkpoint: str, device, *, crops_per_chunk: int = 32,
                 gpu_bag_gib: float = 8.0):
        from fora.encoder.blocks import build_block24
        self.config = json.loads(Path(mil_config_path).read_text())
        self.device = torch.device(device)
        norm = torch.load(self.config["normalization"], map_location="cpu", weights_only=False)
        self.mean = norm["mean"].to(self.device).float()
        self.rms = float(norm["rms"])
        block = build_block24(torch.load(self.config["block24_state"], map_location="cpu", weights_only=False))
        package = torch.load(mil_checkpoint, map_location="cpu", weights_only=False)
        block.load_state_dict(package["block"])
        self.block = block.to(self.device).eval().requires_grad_(False)
        self.mil_classes = len(package.get("classes", []))
        del package
        self.position_scale = float(self.config["training"].get("model", {}).get("position_scale", 1.0))
        self.dim = token_dim(self.config)
        self.crops_per_chunk = int(crops_per_chunk)
        self.gpu_bag_gib = float(gpu_bag_gib)

    @torch.no_grad()
    def _encode(self, views, crops, keep):
        from fora.encoder.cache import CachedBag
        bag = CachedBag(views, self.device, self.gpu_bag_gib)
        pieces, owners, offset = [], [], 0
        for k in range(len(bag.views)):
            for y, pf in bag.view_chunks(k, self.block, crops):
                n = int(y.shape[0])
                x = torch.cat(((y - self.mean) / self.rms, pf.float() * self.position_scale), 1)
                if keep is not None:
                    x = x[keep[offset: offset + n]]
                offset += n
                if x.shape[0]:
                    pieces.append(x.to(torch.bfloat16))
                    owners.append(torch.full((x.shape[0],), k, dtype=torch.long, device=self.device))
        del bag
        return pieces, owners, offset

    @torch.no_grad()
    def encode(self, image: dict, *, cap: int = 0, seed_key: str = "") -> SimpleNamespace:
        """All tokens of one study (or a deterministic ``cap`` subsample) + per-token view kind / series ids."""
        views = from_shared(image["views"])
        total = int(sum(int(v["starts"][-1]) for v in views))
        keep = subsample_mask(total, int(cap), seed_key, self.device)
        result = None
        for crops in dict.fromkeys((self.crops_per_chunk, 8, 2)):
            try:
                result = self._encode(views, crops, keep)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
        if result is None:
            raise torch.OutOfMemoryError(f"block-24 stream OOM for {image.get('case_id')}")
        pieces, owners, seen = result
        if seen != total:
            raise RuntimeError(f"{image.get('case_id')}: streamed {seen} tokens, expected {total}")
        owner = torch.cat(owners)
        view_kind = image["view_kind"].to(self.device)[owner]
        view_series = image["view_series"].to(self.device)[owner]
        return SimpleNamespace(tokens=torch.cat(pieces), view_kind=view_kind, view_series=view_series,
                               total_tokens=total, tokens_per_brick=0)


# ----------------------------------------------------------------------------------------- model
class DinoCTReportGenerator(CTReportGenerator):
    """Report writer over encoder tokens (CT and chest X-ray).

    * resampler: QueryResampler(input ``visual_dim`` -> ``resampler_dim`` (1024 = 8 heads x 128)),
      ``num_visual_tokens`` latents, then LayerNorm + Linear to the LLM width;
    * per-token view embedding (window kind + series ordinal, zero-initialised) added before the resampler, so at
      initialisation the resampler sees exactly the classifier input;
    * study-type tokens: trainable embedding in the resampler width, projected like the latents.
    """

    def __init__(self, llm, *, visual_dim: int, llm_dim: int, num_visual_tokens: int, resampler_depth: int,
                 resampler_heads: int, resampler_dim: int = 1024, n_study_types: int = 32,
                 n_view_kinds: int = N_VIEW_KINDS, max_series: int = MAX_SERIES, **kwargs):
        super().__init__(llm, visual_dim=visual_dim, llm_dim=llm_dim, num_visual_tokens=num_visual_tokens,
                         resampler_depth=resampler_depth, resampler_heads=resampler_heads, **kwargs)
        self.dino_token_dim = int(visual_dim)
        self.resampler_dim = int(resampler_dim)
        self.max_series = int(max_series)
        self.resampler = QueryResampler(input_dim=visual_dim, dim=self.resampler_dim, num_queries=num_visual_tokens,
                                        depth=resampler_depth, heads=resampler_heads, **self.resampler_options)
        self.visual_projection = nn.Sequential(nn.LayerNorm(self.resampler_dim), nn.Linear(self.resampler_dim, llm_dim))
        self.dino_view_kind_embed = nn.Embedding(int(n_view_kinds), visual_dim)
        self.dino_view_series_embed = nn.Embedding(self.max_series, visual_dim)
        nn.init.zeros_(self.dino_view_kind_embed.weight)
        nn.init.zeros_(self.dino_view_series_embed.weight)
        self.dino_study_type_embed = nn.Embedding(int(n_study_types), self.resampler_dim)
        nn.init.normal_(self.dino_study_type_embed.weight, std=0.02)

    def dino_inputs(self, tokens, view_kind, view_series, study_types):
        dtype = self.image_start.dtype
        if tokens.ndim != 2 or tokens.shape[1] != self.dino_token_dim:
            raise ValueError(f"visual tokens must be [N,{self.dino_token_dim}]")
        if view_kind.shape != tokens.shape[:1] or view_series.shape != tokens.shape[:1]:
            raise ValueError("view ids must be one per token")
        embed = self.dino_view_kind_embed(view_kind.to(tokens.device)) + self.dino_view_series_embed(
            view_series.to(tokens.device).clamp(0, self.max_series - 1))
        visual = tokens.to(dtype) + embed.to(dtype)
        study_types = study_types.to(device=self.dino_study_type_embed.weight.device, dtype=torch.long).reshape(-1)
        if study_types.numel() == 0:
            raise ValueError("a study needs at least one study type id")
        modality = self.dino_study_type_embed(study_types).to(dtype)
        return visual, modality

    def _visual_prefix(self, visual_tokens, modality_tokens):
        visual_tokens = visual_tokens.to(self.image_start.dtype)
        modality_tokens = modality_tokens.to(self.image_start.dtype)
        if visual_tokens.ndim != 2 or visual_tokens.shape[-1] != self.dino_token_dim:
            raise ValueError("visual tokens must be [N, dino_token_dim]")
        if modality_tokens.ndim != 2 or modality_tokens.shape[-1] != self.resampler_dim:
            raise ValueError("modality tokens must be [K, resampler_dim]")
        latents = self.resampler(visual_tokens)
        visual_content = self.visual_projection(latents)
        modality_prefix = self.visual_projection(modality_tokens.unsqueeze(0))
        return torch.cat((self.image_start, visual_content, modality_prefix, self.image_end), dim=1)

    def forward(self, visual_tokens, modality_tokens, *args, dino_view_kind=None, dino_view_series=None,
                dino_study_types=None, **kwargs):
        """As ``CTReportGenerator.forward``; ``modality_tokens`` is ignored (study types come from
        ``dino_study_types``)."""
        if dino_view_kind is None or dino_view_series is None or dino_study_types is None:
            raise ValueError("DinoCTReportGenerator.forward needs dino_view_kind/series/study_types")
        visual_tokens, modality_tokens = self.dino_inputs(visual_tokens, dino_view_kind, dino_view_series,
                                                          dino_study_types)
        return super().forward(visual_tokens, modality_tokens, *args, **kwargs)

    def visual_prefix_from_bag(self, bag, study_types):
        """Inference helper: the exact training prefix for one encoded study."""
        visual, modality = self.dino_inputs(bag.tokens, bag.view_kind, bag.view_series, study_types)
        return self._visual_prefix(visual, modality)


def is_dino_checkpoint(package: dict) -> bool:
    return package.get("metadata", {}).get("arguments", {}).get("visual_stream") == VISUAL_STREAM
