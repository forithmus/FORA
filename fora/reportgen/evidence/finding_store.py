"""Per-finding inputs of the report writer: descriptions, pooled + focus evidence tokens, zoomed evidence images and the study
measurement line (paper section "Grounded report generation").

``FindingLocStore`` reads a localization-store directory (one ``.npy`` per field + ``manifest.json``; written by
``fora_report.writer.dense_cache --kind loc`` for training, by ``write_study_loc_dir`` at inference) together with the
focus-token store of ``fora.reportgen.evidence.focus_tokens``. Each planned finding gets 1 + n tokens of 1048 + 16 dims: the
attention-pooled MIL feature (aux all zero) followed by up to 16 focus tokens (aux layout in ``focus_tokens``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

LOC_FORMAT = "fora_evidence_store_cache"   # written by fora_report.writer.dense_cache --kind loc and write_study_loc_dir
LOC_MEMBERS = (
    "report_uid", "row_start", "cls", "prob", "reg3", "frac3", "left",
    "right", "zfrac", "view", "feat", "region_names",
)
AUX_DIM = 16


class FindingLocStore:
    """UID-indexed localization rows with descriptions, focus tokens and evidence images.

    root: localization-store directory; labels: MIL vocabulary (must match the manifest);
    descriptions: S600 array file aligned with the store rows; ftok: focus-token store directory;
    study_meas: JSON {report_uid: measurement line}; fimg_views: image views per finding."""

    FTOK_SAFETY = 1024   # every finding gets 1 pooled + up to 16 focus tokens unless the study would exceed this many evidence tokens

    def __init__(self, root: str | Path, labels: Sequence[str], *, descriptions: str | Path | None = None, ftok: str | Path | None = None,
                 study_meas: str | Path | None = None, fimg_views: int = 6) -> None:
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") != LOC_FORMAT:
            raise ValueError(f"{self.root}: not a {LOC_FORMAT} directory")
        load = lambda name: np.load(self.root / f"{name}.npy", mmap_mode="r", allow_pickle=False)  # noqa: E731
        self.uids = load("report_uid")
        self.row_start = np.asarray(load("row_start"), dtype=np.int64)
        self.cls = load("cls")
        self.prob = load("prob")
        self.reg3 = load("reg3")
        self.frac3 = load("frac3")
        self.left = load("left")
        self.right = load("right")
        self.zfrac = load("zfrac")
        self.feat = load("feat")
        self.region_names = [str(x) for x in load("region_names").tolist()]
        self.fimg_views = int(fimg_views)
        cached_labels = self.manifest.get("labels")
        if cached_labels is not None and list(cached_labels) != list(labels):
            raise ValueError("localization store vocabulary differs from labels")
        if self.row_start.shape != (self.uids.size + 1,):
            raise ValueError("localization row_start has the wrong shape")
        if int(self.row_start[-1]) != int(self.cls.shape[0]):
            raise ValueError("localization row_start does not cover all rows")
        self.feature_dim = int(self.feat.shape[1])
        self.row_by_uid = {str(u): i for i, u in enumerate(self.uids.tolist())}
        self.desc = np.load(descriptions, mmap_mode="r") if descriptions else None
        if self.desc is not None and self.desc.shape[0] != self.cls.shape[0]:
            raise ValueError("descriptions do not match the localization store rows")
        self.ftok = None
        self.fimg = None
        if ftok:
            fr = Path(ftok)
            self.ftok = dict(keys=np.load(fr / "keys.npy"), x=np.load(fr / "x.npy", mmap_mode="r"),
                             aux=np.load(fr / "aux.npy", mmap_mode="r"), n=np.load(fr / "n_valid.npy"))
            self.feature_dim = self.feature_dim + AUX_DIM
            if (fr / "fimg_off.npy").exists():
                self.fimg = dict(off=np.load(fr / "fimg_off.npy"), blob=np.load(fr / "fimg_blob.npy", mmap_mode="r"))
        self.study_meas = json.loads(Path(study_meas).read_text()) if study_meas and Path(study_meas).exists() else None
        print(f"[finding-store] study_meas={'on ' + str(len(self.study_meas)) if self.study_meas else 'off'} "
              f"desc={'on' if self.desc is not None else 'off'} ftok={'on rows=' + str(len(self.ftok['keys'])) if self.ftok is not None else 'off'} "
              f"feature_dim={self.feature_dim}", flush=True)

    def study_line(self, uid: str) -> str:
        return (self.study_meas or {}).get(str(uid), "")

    def contains(self, uid: str) -> bool:
        return str(uid) in self.row_by_uid

    def rows(self, uid: str) -> dict[int, int]:
        """class index -> global row for one report (empty if absent)."""
        index = self.row_by_uid.get(str(uid))
        if index is None:
            return {}
        start, stop = int(self.row_start[index]), int(self.row_start[index + 1])
        return {int(c): start + offset for offset, c in enumerate(np.asarray(self.cls[start:stop]).tolist())}

    def describe_row(self, row: int) -> str:
        if self.desc is None:
            raise ValueError("FindingLocStore was built without descriptions")
        v = self.desc[row]
        return v.decode("utf-8", "replace") if isinstance(v, (bytes, np.bytes_)) else str(v)

    def descriptions(self, uid: str) -> dict[int, str]:
        return {c: self.describe_row(r) for c, r in self.rows(uid).items()}

    def images(self, rows: Sequence[int]):
        """[(view_0, ..., view_5) JPEG bytes] per row (b'' when missing)."""
        out = []
        for r in rows:
            j = self._ftok_index(int(r)) if self.ftok is not None else -1
            if j < 0 or self.fimg is None:
                out.append((b"",) * self.fimg_views); continue
            o = self.fimg["off"][j]
            out.append(tuple(bytes(self.fimg["blob"][o[k]:o[k + 1]]) for k in range(len(o) - 1)))
        return out

    def _ftok_index(self, row: int) -> int:
        """focus-token store index of a global store row (key "<report_uid>|<cls:04d>"), -1 if absent."""
        i = int(np.searchsorted(self.row_start, int(row), side="right")) - 1
        k = f"{self.uids[i]}|{int(self.cls[row]):04d}"
        keys = self.ftok["keys"]; j = int(np.searchsorted(keys, k))
        return j if j < len(keys) and keys[j] == k else -1

    def _per_finding(self, n: int) -> int:
        return 16 if n * 17 <= self.FTOK_SAFETY else max(1, (self.FTOK_SAFETY - n) // max(n, 1))

    def token_count(self, row: int, per: int = 16) -> int:
        if self.ftok is None:
            return 1
        j = self._ftok_index(row)
        return 1 + (min(per, int(self.ftok["n"][j])) if j >= 0 else 0)

    def token_classes(self, finding_rows: dict[int, int]):
        """class -> number of evidence tokens (dict when focus tokens are enabled, else the class set). The per-finding focus
        allowance is fixed here for the study and reused by the next features() call (same sample)."""
        if self.ftok is None:
            return frozenset(finding_rows)
        self._per = self._per_finding(len(finding_rows))
        return {int(c): self.token_count(r, self._per) for c, r in finding_rows.items()}

    def features(self, rows: Sequence[int], device) -> torch.Tensor:
        if not rows:
            return torch.zeros((0, self.feature_dim), dtype=torch.float32, device=device)
        array = np.asarray(self.feat[np.asarray(rows, dtype=np.int64)], dtype=np.float32)
        if self.ftok is None:
            return torch.from_numpy(array).to(device)
        out = []
        for r, pooled in zip(rows, array):
            out.append(np.concatenate([pooled, np.zeros(AUX_DIM, np.float32)])[None])   # aux all 0 = pooled token
            j = self._ftok_index(int(r))
            per = getattr(self, "_per", 16)
            if j >= 0 and min(per, int(self.ftok["n"][j])) > 0:
                k = min(per, int(self.ftok["n"][j]))
                out.append(np.concatenate([np.asarray(self.ftok["x"][j, :k], np.float32), np.asarray(self.ftok["aux"][j, :k], np.float32)], 1))
        return torch.from_numpy(np.concatenate(out, 0)).to(device)


def write_study_loc_dir(loc_dir: str | Path, ftok_dir: str | Path, uid: str, n_targets: int, labels: Sequence[str], region_names: Sequence[str]) -> None:
    """Inference: localization-store directory of one study from its focus-token store (``ftok_dir/loc_part.npz``). With no
    targets, empty arrays are written (and an empty focus-token store) so the writer runs with an empty plan."""
    loc, ft = Path(loc_dir), Path(ftok_dir); loc.mkdir(parents=True, exist_ok=True)
    if n_targets:
        with np.load(ft / "loc_part.npz") as z:
            for k in ("cls", "prob", "reg3", "frac3", "left", "right", "zfrac", "view", "feat"): np.save(loc / f"{k}.npy", z[k])
    else:
        for k, shape, dtype in [("cls", (0,), np.int16), ("prob", (0,), np.float32), ("reg3", (0, 3), np.int16), ("frac3", (0, 3), np.float16),
                                ("left", (0,), np.float16), ("right", (0,), np.float16), ("zfrac", (0,), np.float16), ("view", (0,), np.int16),
                                ("feat", (0, 1048), np.float16)]: np.save(loc / f"{k}.npy", np.empty(shape, dtype))
        ft.mkdir(parents=True, exist_ok=True)
        for k, v in dict(row_ids=np.empty(0, np.int64), keys=np.empty(0, "U80"), n_valid=np.empty(0, np.int16),
                         x=np.empty((0, 16, 1048), np.float16), aux=np.empty((0, 16, 16), np.float16), fimg_off=np.empty((0, 7), np.int64),
                         fimg_blob=np.empty(0, np.uint8), desc=np.empty(0, "S600")).items(): np.save(ft / f"{k}.npy", v)
    np.save(loc / "report_uid.npy", np.array([uid])); np.save(loc / "row_start.npy", np.array([0, n_targets]))
    np.save(loc / "region_names.npy", np.array(list(region_names)))
    (loc / "manifest.json").write_text(json.dumps(dict(format=LOC_FORMAT, labels=list(labels))))
