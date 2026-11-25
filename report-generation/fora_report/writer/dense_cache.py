"""Extract a classifier prediction archive or an evidence store into a validated node-local mmap cache.

Inputs of report-generation training and inference (Methods, "Report generation"):

* ``--kind dense``: the classifier's raw logits for every study (``report_uid``, ``full_token_count``, ``logit``
  [studies, classes] float32, ``label``, ``class_index``), an uncompressed-or-deflated ``npz``;
* ``--kind loc``: the per-(study, class) evidence store (pooled evidence feature, location fields).

Run once per node (e.g. under ``flock``); it extracts the members to local storage, validates their schema and
byte sizes and writes ``manifest.json`` last. An existing valid cache of the same source is reused.

    python -m fora_report.writer.dense_cache --source dense_predictions.npz --output CACHE/dense
    python -m fora_report.writer.dense_cache --kind loc --source evidence_store.npz --output CACHE/loc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np


REQUIRED_MEMBERS = (
    "report_uid.npy",
    "full_token_count.npy",
    "logit.npy",
    "label.npy",
    "class_index.npy",
)
# Evidence store (one row per planned (study, class)).
LOC_MEMBERS = (
    "report_uid.npy",
    "row_start.npy",
    "cls.npy",
    "prob.npy",
    "reg3.npy",
    "frac3.npy",
    "left.npy",
    "right.npy",
    "zfrac.npy",
    "view.npy",
    "feat.npy",
    "region_names.npy",
    "labels.npy",
)


def validate_loc_arrays(output: Path, **_unused) -> dict:
    load = lambda name: np.load(output / name, mmap_mode="r", allow_pickle=False)  # noqa: E731
    uids = load("report_uid.npy")
    row_start = load("row_start.npy")
    cls = load("cls.npy")
    feat = load("feat.npy")
    labels = load("labels.npy")
    if uids.ndim != 1 or row_start.shape != (uids.size + 1,):
        raise ValueError("loc store report_uid/row_start shapes disagree")
    if int(row_start[0]) != 0 or int(row_start[-1]) != cls.shape[0]:
        raise ValueError("loc store row_start does not span all rows")
    if np.any(np.diff(row_start) < 0):
        raise ValueError("loc store row_start is not monotonic")
    for name in ("prob.npy", "reg3.npy", "frac3.npy", "left.npy", "right.npy", "zfrac.npy", "view.npy"):
        if load(name).shape[0] != cls.shape[0]:
            raise ValueError(f"loc store {name} has the wrong row count")
    if feat.ndim != 2 or feat.shape[0] != cls.shape[0]:
        raise ValueError("loc store feat has the wrong shape")
    if cls.size and (int(cls.min()) < 0 or int(cls.max()) >= labels.size):
        raise ValueError("loc store class index out of range")
    if np.unique(uids).size != uids.size:
        raise ValueError("loc store report_uid contains duplicates")
    return {
        "n_studies": int(uids.size),
        "n_rows": int(cls.shape[0]),
        "n_classes": int(labels.size),
        "feature_dim": int(feat.shape[1]),
        "labels": [str(x) for x in labels.tolist()],
        "label_sha256": hashlib.sha256(
            "\n".join(map(str, labels.tolist())).encode()
        ).hexdigest(),
        "files": {name: int((output / name).stat().st_size) for name in LOC_MEMBERS},
    }


def file_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(f"{path.suffix}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_arrays(
    output: Path,
    *,
    expected_studies: int,
    expected_classes: int,
) -> dict:
    uids = np.load(output / "report_uid.npy", mmap_mode="r", allow_pickle=False)
    token_counts = np.load(
        output / "full_token_count.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    logits = np.load(output / "logit.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(output / "label.npy", mmap_mode="r", allow_pickle=False)
    indices = np.load(output / "class_index.npy", mmap_mode="r", allow_pickle=False)
    if uids.ndim != 1:
        raise ValueError(f"report_uid must be rank one, got {uids.shape}")
    if token_counts.shape != uids.shape:
        raise ValueError(
            "full_token_count must have one value per study, got "
            f"{token_counts.shape} for {uids.shape}"
        )
    if not np.issubdtype(token_counts.dtype, np.integer):
        raise ValueError(
            f"full_token_count must be integral, got {token_counts.dtype}"
        )
    if np.any(token_counts <= 0):
        raise ValueError("full_token_count values must all be positive")
    if logits.shape != (uids.size, labels.size):
        raise ValueError(
            "dense MIL shapes disagree: "
            f"uids={uids.shape} logits={logits.shape} labels={labels.shape}"
        )
    if logits.dtype != np.float32:
        raise ValueError(f"dense MIL logits must be float32, got {logits.dtype}")
    if indices.shape != (labels.size,) or not np.array_equal(
        indices, np.arange(labels.size, dtype=indices.dtype)
    ):
        raise ValueError("dense MIL class_index is not the complete ordered vocabulary")
    if expected_studies and uids.size != expected_studies:
        raise ValueError(
            f"expected {expected_studies} studies, found {uids.size}"
        )
    if expected_classes and labels.size != expected_classes:
        raise ValueError(
            f"expected {expected_classes} classes, found {labels.size}"
        )
    if np.unique(uids).size != uids.size:
        raise ValueError("dense MIL report_uid contains duplicates")
    return {
        "n_studies": int(uids.size),
        "n_classes": int(labels.size),
        "logit_dtype": str(logits.dtype),
        "uid_dtype": str(uids.dtype),
        "token_count_dtype": str(token_counts.dtype),
        "minimum_token_count": int(token_counts.min()),
        "maximum_token_count": int(token_counts.max()),
        "label_sha256": hashlib.sha256(
            "\n".join(map(str, labels.tolist())).encode()
        ).hexdigest(),
        "files": {
            name: int((output / name).stat().st_size)
            for name in REQUIRED_MEMBERS
        },
    }


def reusable(
    source: Path,
    output: Path,
    *,
    expected_studies: int,
    expected_classes: int,
    source_metrics: Path | None,
    kind: str = "dense",
) -> dict | None:
    marker = output / "manifest.json"
    if not marker.is_file():
        return None
    try:
        manifest = json.loads(marker.read_text())
        stat = source.stat()
        if (
            Path(manifest["source"]).resolve() != source.resolve()
            or int(manifest["source_size"]) != stat.st_size
            or int(manifest["source_mtime_ns"]) != stat.st_mtime_ns
        ):
            return None
        if source_metrics is not None and (
            Path(str(manifest.get("source_metrics", ""))).resolve()
            != source_metrics.resolve()
            or manifest.get("source_metrics_sha256")
            != file_sha256(source_metrics)
        ):
            return None
        if manifest.get("kind", "dense") != kind:
            return None
        validation = (validate_loc_arrays if kind == "loc" else validate_arrays)(
            output,
            expected_studies=expected_studies,
            expected_classes=expected_classes,
        )
        if validation["files"] != manifest.get("files"):
            return None
        return manifest
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-studies", type=int, default=0)
    parser.add_argument("--expected-classes", type=int, default=0)
    parser.add_argument(
        "--source-metrics",
        type=Path,
        default=None,
        help="Optional classifier evaluation metrics JSON (recorded in the manifest for provenance).",
    )
    parser.add_argument(
        "--kind",
        choices=("dense", "loc"),
        default="dense",
        help="dense = classifier logits archive; loc = evidence store.",
    )
    args = parser.parse_args()
    members = LOC_MEMBERS if args.kind == "loc" else REQUIRED_MEMBERS
    validator = validate_loc_arrays if args.kind == "loc" else validate_arrays
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)

    manifest = reusable(
        source,
        output,
        expected_studies=args.expected_studies,
        expected_classes=args.expected_classes,
        source_metrics=(
            args.source_metrics.resolve()
            if args.source_metrics is not None
            else None
        ),
        kind=args.kind,
    )
    if manifest is not None:
        print(
            f"[dense-mil-cache] reuse studies={manifest['n_studies']} "
            f"classes={manifest['n_classes']} output={output}",
            flush=True,
        )
        return

    (output / "manifest.json").unlink(missing_ok=True)
    for partial in output.glob("*.partial.*"):
        partial.unlink(missing_ok=True)
    with zipfile.ZipFile(source) as archive:
        available = set(archive.namelist())
        missing = set(members) - available
        if missing:
            raise ValueError(f"{args.kind} archive is missing {sorted(missing)}")
        for name in members:
            destination = output / name
            temporary = output / f"{name}.partial.{os.getpid()}"
            with archive.open(name) as reader, temporary.open("wb") as writer:
                shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, destination)

    validation = validator(
        output,
        expected_studies=args.expected_studies,
        expected_classes=args.expected_classes,
    )
    source_metrics = {}
    if args.source_metrics is not None:
        source_metrics_path = args.source_metrics.resolve()
        source_metrics = json.loads(source_metrics_path.read_text())
        source_metrics = {
            "source_metrics": str(source_metrics_path),
            "mil_checkpoint": source_metrics.get("checkpoint"),
            "mil_optimizer_step": source_metrics.get(
                "optimizer_step", source_metrics.get("mil_optimizer_step")
            ),
            "source_metrics_sha256": file_sha256(source_metrics_path),
        }
    stat = source.stat()
    manifest = {
        "format": (
            "fora_evidence_store_cache"
            if args.kind == "loc"
            else "fora_dense_mil_mmap_cache"
        ),
        "kind": args.kind,
        "format_version": 1,
        "source": str(source),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        **source_metrics,
        **validation,
    }
    atomic_json(manifest, output / "manifest.json")
    print(
        f"[dense-mil-cache] created studies={manifest['n_studies']} "
        f"classes={manifest['n_classes']} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
