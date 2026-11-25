"""Frozen classifier logits per study for the diagnostic plan (paper: "Diagnostic plan").

The organ-routed MIL classifier is not run inside the writer trainer: its study-bag logits for every study are
precomputed once and converted to a node-local memory-mapped cache by ``fora_report.writer.dense_cache``
(files ``manifest.json``, ``report_uid.npy``, ``logit.npy`` [studies, classes] float32, ``full_token_count.npy``,
``label.npy``, ``class_index.npy``). The cache must be the exact source the plan calibration was fitted on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

CACHE_FORMAT = "fora_dense_mil_mmap_cache"


class DenseMILLogitStore:
    """Read-only UID-indexed frozen MIL logits from a node-local mmap cache."""

    def __init__(self, root: str | Path, labels: list[str], *, calibration: dict) -> None:
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") != CACHE_FORMAT:
            raise ValueError("precomputed MIL cache has an unknown format")
        load = lambda name: np.load(self.root / name, mmap_mode="r", allow_pickle=False)  # noqa: E731
        self.uids = load("report_uid.npy")
        self.logits = load("logit.npy")
        self.full_token_counts = load("full_token_count.npy")
        cached_labels = load("label.npy")
        cached_indices = load("class_index.npy")
        if self.logits.shape != (self.uids.size, len(labels)):
            raise ValueError("precomputed MIL logits do not match study/class dimensions")
        if self.full_token_counts.shape != self.uids.shape:
            raise ValueError("precomputed MIL token counts do not match study dimensions")
        if not np.issubdtype(self.full_token_counts.dtype, np.integer) or np.any(self.full_token_counts <= 0):
            raise ValueError("precomputed MIL token counts must be positive integers")
        if self.logits.dtype != np.float32:
            raise ValueError("precomputed MIL logits must be float32")
        if list(map(str, cached_labels.tolist())) != list(map(str, labels)):
            raise ValueError("precomputed MIL vocabulary does not exactly match report labels")
        if not np.array_equal(cached_indices, np.arange(len(labels), dtype=cached_indices.dtype)):
            raise ValueError("precomputed MIL class order is incomplete")
        source = Path(str(self.manifest.get("source", ""))).resolve()
        calibration_source = Path(str(calibration.get("source_predictions", ""))).resolve()
        if source != calibration_source:
            raise ValueError("precomputed MIL logits are not the source predictions used by the frozen calibration")
        self.row_by_uid = {str(uid): row for row, uid in enumerate(self.uids.tolist())}
        if len(self.row_by_uid) != self.uids.size:
            raise ValueError("precomputed MIL cache contains duplicate UIDs")

    def require_coverage(self, uids: list[str]) -> None:
        missing = [uid for uid in uids if str(uid) not in self.row_by_uid]
        if missing:
            raise ValueError(f"precomputed MIL cache misses {len(missing)} local reports; first={missing[0]}")

    def get(self, uid: str, device) -> torch.Tensor:
        """[1, classes] float32 logits of one study (one row copied out of the shared mmap)."""
        try:
            row = self.row_by_uid[str(uid)]
        except KeyError as error:
            raise KeyError(f"no precomputed MIL logits for {uid}") from error
        return torch.tensor(np.array(self.logits[row], dtype=np.float32, copy=True), device=device).unsqueeze(0)
