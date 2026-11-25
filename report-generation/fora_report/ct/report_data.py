"""CT report-writer training data (paper: "Grounded report generation", writer training data).

One sample = one study (report). The image is the study's node-local block-23 DINO cache (all series and views,
prepared by ``fora.reportgen.writer.dino_stream.prepare_dino_image``); the target is the Findings section of the report,
read from the study's HDF5 report group. Labels (asserted/uncertain = positive, explicit denial = negative,
unmentioned = masked) come from the report-generation index and are used only as output-side targets.

The report-generation index is a SQLite file with a ``metadata(key, value)`` table (``global`` -> {"labels": [...]})
and a ``samples`` table with columns ``report_uid, patient_uid, h5_id, split, protocol_class, series_json,
study_type_ids_json, positive_json, negative_json``.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

FALLBACK_FINDINGS = "No findings were reported."


def open_index(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def read_index_metadata(path: str | Path) -> dict:
    with open_index(path) as connection:
        return {row["key"]: json.loads(row["value"]) for row in connection.execute("SELECT key, value FROM metadata")}


def shard_counts(path: str | Path, *, split: str, protocol_class: int = 0) -> dict[str, int]:
    """h5_id -> number of indexed studies of ``split`` (used for logging the cohort size)."""
    with open_index(path) as connection:
        rows = connection.execute(
            "SELECT h5_id, COUNT(*) AS n FROM samples WHERE split = ? AND protocol_class = ? GROUP BY h5_id",
            (split, int(protocol_class)))
        return {str(row["h5_id"]): int(row["n"]) for row in rows}


def _chunks(values: list[str], size: int = 500) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start: start + size]


def _decode(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _json_string_list(value) -> list[str]:
    try:
        parsed = json.loads(_decode(value))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def read_findings(h5_path: str, patient: str, report_uid: str) -> str:
    """Findings section only (impression, technique, comparison and clinical information are excluded)."""
    import h5py
    findings = ""
    with h5py.File(h5_path, "r") as handle:
        report = handle[patient][report_uid]["report"]
        structured = report.get("structured")
        if structured is not None and "findings" in structured:
            findings = _decode(structured["findings"][()]).strip()
        if not findings:
            for key in ("extracted_sentences", "there_is_statements"):
                if key in report:
                    sentences = _json_string_list(report[key][()])
                    if sentences:
                        findings = " ".join(sentences)
                        break
    return findings or FALLBACK_FINDINGS


class CTReportIndexDataset(Dataset):
    """Studies of ``split`` in the given H5 shards, in (h5_id, report_uid) order; ``render_report`` gives the image."""

    def __init__(self, *, index_path: str | Path, h5_root: str | Path, split: str, h5_ids: list[str],
                 protocol_class: int = 0, n_classes: int | None = None) -> None:
        self.n_classes = int(n_classes) if n_classes is not None else None
        self.index: list[dict] = []
        h5_root = Path(h5_root)
        with open_index(index_path) as connection:
            for group in _chunks(sorted(h5_ids)):
                placeholders = ",".join("?" for _ in group)
                query = f"""
                    SELECT report_uid, patient_uid, h5_id, series_json,
                           study_type_ids_json, positive_json, negative_json
                    FROM samples
                    WHERE split = ? AND protocol_class = ? AND h5_id IN ({placeholders})
                    ORDER BY h5_id, report_uid
                """
                for row in connection.execute(query, [split, int(protocol_class), *group]):
                    self.index.append({
                        "h5_path": str(h5_root / f"{row['h5_id']}.h5"),
                        "patient": str(row["patient_uid"]),
                        "report_uid": str(row["report_uid"]),
                        "series": json.loads(row["series_json"]),
                        "study_type_ids": json.loads(row["study_type_ids_json"]),
                        "positive_indices": json.loads(row["positive_json"]),
                        "negative_indices": json.loads(row["negative_json"]),
                    })

    def __len__(self) -> int:
        return len(self.index)

    def render_report(self, sample):
        raise NotImplementedError

    def report_text(self, sample: dict) -> str:
        return read_findings(sample["h5_path"], sample["patient"], sample["report_uid"])

    _report_text = report_text

    def __getitem__(self, index: int):
        sample = self.index[index]
        image, real_mask, tile_coords = self.render_report(sample)
        return (
            image,
            real_mask,
            tile_coords,
            torch.tensor(sample["study_type_ids"], dtype=torch.long),
            torch.tensor(sample["positive_indices"], dtype=torch.long),
            torch.tensor(sample["negative_indices"], dtype=torch.long),
            self.report_text(sample),
            str(sample["report_uid"]),
        )


class DinoReportDataset(CTReportIndexDataset):
    """Studies whose node-local DINO cache is on this host; the image is the prepared cache of all views."""

    def __init__(self, *, cases_by_uid: dict, mil_config: dict, cache_root: str | Path, index_path: str,
                 h5_root: str, split: str, n_classes: int | None = None):
        uids = sorted(cases_by_uid)
        with open_index(index_path) as connection:
            h5_ids = sorted({
                str(row[0]) for start in range(0, len(uids), 500)
                for row in connection.execute(
                    "SELECT h5_id FROM samples WHERE protocol_class=0 AND split=? AND report_uid IN (%s)"
                    % ",".join("?" * len(uids[start:start + 500])), [split, *uids[start:start + 500]])
            })
        super().__init__(index_path=index_path, h5_root=h5_root, split=split, h5_ids=h5_ids, n_classes=n_classes)
        wanted = set(uids)
        self.index = [s for s in self.index if str(s["report_uid"]) in wanted]
        self.index.sort(key=lambda s: str(s["report_uid"]))
        self.cases = {u: cases_by_uid[u] for u in (str(s["report_uid"]) for s in self.index)}
        self.mil_config = mil_config
        self.cache_root = Path(cache_root)

    def token_counts(self) -> list[int]:
        from fora.reportgen.writer.dino_stream import case_token_count
        return [case_token_count(self.cases[str(s["report_uid"])]) for s in self.index]

    def render_report(self, sample):
        from fora.reportgen.writer.dino_stream import prepare_dino_image
        return prepare_dino_image(self.cases[str(sample["report_uid"])], self.mil_config,
                                  Path(self.cache_root) / self.mil_config["fingerprint"]), None, None


def collate_one(batch):
    if len(batch) != 1:
        raise ValueError("variable-size study bags require per-rank batch_size=1")
    return batch[0]
