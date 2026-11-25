"""CT study input contract: study JSON validation, series readers (NIfTI, NPZ, DICOM directory) and CADS mask lookup.

All readers return HU arrays (X, Y, Z) with a RAS-world affine in mm (paper section "Cohort and data preparation").
The DICOM reader applies RescaleSlope/Intercept, sorts slices geometrically and converts LPS to RAS. Enhanced or
multiframe DICOM, irregular slice spacing and gantry-tilted (sheared) series are rejected.

Validation status: NIfTI was checked end to end against stored model outputs on a CT-RATE study. NPZ and DICOM were
checked on synthetic data only (HU conversion, slice sorting, LPS to RAS).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

STUDY_ID_PATTERN = r"[A-Za-z0-9_.-]{1,64}"
MAX_SERIES = 32
CADS_TASKS = tuple(range(551, 560))


@dataclass(frozen=True)
class LoadedSeries:
    data: np.ndarray
    affine: np.ndarray
    windows: tuple[tuple[float, float], ...]
    source: str
    warnings: tuple[str, ...] = ()


def _full_window() -> tuple[float, float]:
    from fora.geometry.windows import FULL_WINDOW
    return tuple(map(float, FULL_WINDOW))


def _number_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.replace("\\", ",").split(",")
    if np.isscalar(value):
        value = [value]
    result = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            pass
    return result


def _explicit_windows(item: dict[str, Any]) -> tuple[tuple[float, float], ...]:
    if "windows" in item:
        result = []
        for window in item["windows"]:
            values = _number_list(window)
            if len(values) != 2 or values[1] <= values[0]:
                raise ValueError(f"invalid HU window: {window!r}")
            result.append((values[0], values[1]))
        return tuple(result)
    centers = _number_list(item.get("window_center"))
    widths = _number_list(item.get("window_width"))
    return tuple((c - w / 2.0, c + w / 2.0) for c, w in zip(centers, widths) if w > 0)


def _with_full_window(windows: Iterable[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    ordered = [_full_window()]
    for window in windows:
        normalized = tuple(map(float, window))
        if normalized not in ordered:
            ordered.append(normalized)
    return tuple(ordered)


def _load_nifti(path: Path, item: dict[str, Any]) -> LoadedSeries:
    import nibabel as nib
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D NIfTI, got {data.shape}")
    windows = _with_full_window(_explicit_windows(item))
    warning = () if len(windows) > 1 else (
        "no native DICOM window supplied for NIfTI; rendered with the full HU window only",)
    return LoadedSeries(data, np.asarray(image.affine), windows, str(path), warning)


def _load_npz(path: Path, item: dict[str, Any]) -> LoadedSeries:
    with np.load(path, allow_pickle=False) as package:
        if "data" not in package or "affine" not in package:
            raise ValueError(f"{path}: NPZ requires data and affine arrays")
        data = np.array(package["data"], copy=True)
        affine = np.array(package["affine"], copy=True)
        embedded = package["windows"].tolist() if "windows" in package else ()
    if data.ndim != 3:
        raise ValueError(f"{path}: expected 3-D data, got {data.shape}")
    windows = _with_full_window(_explicit_windows(item) or embedded)
    return LoadedSeries(data, affine, windows, str(path))


def _dicom_window(dataset: Any) -> tuple[tuple[float, float], ...]:
    centers = _number_list(getattr(dataset, "WindowCenter", None))
    widths = _number_list(getattr(dataset, "WindowWidth", None))
    return tuple((c - w / 2.0, c + w / 2.0) for c, w in zip(centers, widths) if w > 0)


def _load_dicom_directory(path: Path, item: dict[str, Any]) -> LoadedSeries:
    import pydicom
    records = []
    requested_uid = str(item.get("series_instance_uid", "")).strip()
    for candidate in sorted(c for c in path.rglob("*") if c.is_file()):
        try:
            dataset = pydicom.dcmread(candidate, stop_before_pixels=True, force=False)
        except Exception:
            continue
        if str(getattr(dataset, "Modality", "")).upper() != "CT":
            continue
        uid = str(getattr(dataset, "SeriesInstanceUID", ""))
        if requested_uid and uid != requested_uid:
            continue
        if not hasattr(dataset, "ImagePositionPatient") or not hasattr(dataset, "ImageOrientationPatient"):
            continue
        records.append((candidate, uid, dataset))
    if not records:
        raise ValueError(f"{path}: no geometry-complete CT DICOM slices found")
    found_uids = sorted({uid for _, uid, _ in records})
    if len(found_uids) != 1:
        raise ValueError(f"{path}: contains {len(found_uids)} CT series; provide series_instance_uid "
                         "or one series directory per manifest entry")
    first = records[0][2]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=np.float64)
    if orientation.shape != (6,):
        raise ValueError(f"{path}: invalid ImageOrientationPatient")
    column_axis, row_axis = orientation[:3], orientation[3:]
    normal = np.cross(column_axis, row_axis)
    norm = np.linalg.norm(normal)
    if not np.isfinite(norm) or norm < 0.9:
        raise ValueError(f"{path}: invalid DICOM orientation vectors")
    normal /= norm
    positioned = []
    for candidate, _, header in records:
        current = np.asarray(header.ImageOrientationPatient, dtype=np.float64)
        if not np.allclose(current, orientation, atol=1e-4):
            raise ValueError(f"{path}: gantry orientation varies within the series")
        position = np.asarray(header.ImagePositionPatient, dtype=np.float64)
        positioned.append((float(position @ normal), candidate, position, header))
    positioned.sort(key=lambda value: value[0])
    coordinates = np.asarray([record[0] for record in positioned])
    if len(coordinates) < 16:
        raise ValueError(f"{path}: CT series has fewer than 16 slices")
    gaps = np.diff(coordinates)
    slice_spacing = float(np.median(gaps))
    if slice_spacing <= 0 or not np.allclose(gaps, slice_spacing, rtol=0.05, atol=0.15):
        raise ValueError(f"{path}: irregular or duplicate slice geometry")
    delta = np.diff(np.stack([r[2] for r in positioned]), axis=0)
    if not np.allclose(delta, normal[None] * slice_spacing, rtol=.05, atol=.15):
        raise ValueError(f"{path}: gantry tilt / sheared slice positions require geometry-corrected HU NIfTI")
    arrays, native_windows = [], []
    rows = columns = None
    for _, candidate, _, _header in positioned:
        dataset = pydicom.dcmread(candidate)
        pixels = np.asarray(dataset.pixel_array)
        if pixels.ndim != 2:
            raise ValueError(f"{candidate}: enhanced/multiframe DICOM is not supported")
        if rows is None:
            rows, columns = pixels.shape
        elif pixels.shape != (rows, columns):
            raise ValueError(f"{path}: slice matrix changes within the series")
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        arrays.append(pixels.astype(np.float32) * slope + intercept)
        native_windows.extend(_dicom_window(dataset))
    # NIfTI-style (X, Y, Z): X is the DICOM column, Y the DICOM row.
    data = np.stack(arrays, axis=-1).transpose(1, 0, 2)
    spacing = _number_list(getattr(first, "PixelSpacing", None))
    if len(spacing) < 2 or min(spacing[:2]) <= 0:
        raise ValueError(f"{path}: invalid PixelSpacing")
    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, 0] = column_axis * spacing[1]
    affine_lps[:3, 1] = row_axis * spacing[0]
    affine_lps[:3, 2] = normal * slice_spacing
    affine_lps[:3, 3] = positioned[0][2]
    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps
    windows = _with_full_window(_explicit_windows(item) or native_windows)
    return LoadedSeries(data, affine, windows, f"{path}#{found_uids[0]}")


def load_series(item: dict[str, Any], base: Path) -> LoadedSeries:
    """One `series[]` entry -> HU volume, RAS affine and the ordered view windows (full window first)."""
    from fora.geometry.series import sane_affine
    if not isinstance(item, dict) or not item.get("path"):
        raise ValueError("every series[] entry must contain a path")
    path = Path(str(item["path"])).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    kind = str(item.get("format", "auto")).strip().lower()
    if path.is_dir() or kind == "dicom":
        if not path.is_dir():
            raise ValueError("DICOM input path must be a directory")
        result = _load_dicom_directory(path, item)
    elif kind == "npz" or path.suffix.lower() == ".npz":
        result = _load_npz(path, item)
    elif kind in {"auto", "nifti", "nii"} and path.name.lower().endswith((".nii", ".nii.gz")):
        result = _load_nifti(path, item)
    else:
        raise ValueError(f"unsupported CT series format for {path}")
    if not sane_affine(result.affine):
        raise ValueError(f"{result.source}: affine is corrupt or physically implausible")
    return result


def study_type_ids(study: dict[str, Any]) -> tuple[list[int], list[str]]:
    """`study_types` (names or integer ids of the 29-class protocol taxonomy) -> sorted ids + warnings."""
    from fora.geometry.protocols import UNK_STUDY_TYPE, study_type_index
    raw = study.get("study_types", study.get("study_type"))
    if raw is None:
        return [UNK_STUDY_TYPE], ["study_type absent; used the learned unknown-modality token"]
    values = raw if isinstance(raw, list) else [raw]
    indices, warnings = [], []
    for value in values:
        if isinstance(value, int):
            index = int(value)
            if not 0 <= index <= UNK_STUDY_TYPE:
                raise ValueError(f"study type ID must be in [0,{UNK_STUDY_TYPE}]")
        else:
            index = study_type_index(str(value))
            if index == UNK_STUDY_TYPE:
                warnings.append(f"unrecognized study_type {value!r}; used unknown token")
        indices.append(index)
    return sorted(set(indices)) or [UNK_STUDY_TYPE], warnings


def validate_study(data: dict[str, Any]) -> str:
    uid = str(data.get("study_id", ""))
    if not re.fullmatch(STUDY_ID_PATTERN, uid):
        raise ValueError("study_id must contain 1-64 letters, digits, _, . or -")
    if not isinstance(data.get("series"), list) or not data["series"]:
        raise ValueError("A nonempty series list is required")
    if len(data["series"]) > MAX_SERIES:
        raise ValueError(f"At most {MAX_SERIES} reconstructions per study are supported")
    return uid


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare_input(study: Path, work: Path) -> dict:
    """Read every series, write it as a float32 HU NIfTI into `work`, copy supplied CADS masks or stage the volume for
    CADS, and write `work/input.json` (study id, protocol ids, per-series windows and mask paths)."""
    import nibabel as nib
    data = json.loads(study.read_text())
    uid = validate_study(data)
    types, warnings = study_type_ids(data)
    records, needs_cads = [], False
    (work / "cads_in").mkdir(exist_ok=True)
    for i, item in enumerate(data["series"]):
        loaded = load_series(item, study.parent)
        if not np.isfinite(loaded.affine).all():
            raise ValueError("Nonfinite affine")
        raw = work / ("raw_%03d.nii.gz" % i)
        image = nib.Nifti1Image(np.asarray(loaded.data, dtype=np.float32), loaded.affine)
        image.header.set_xyzt_units("mm")
        nib.save(image, raw)
        cads = item.get("cads_parts")
        masks = {}
        if cads is not None:
            if set(map(int, cads)) != set(CADS_TASKS):
                raise ValueError("cads_parts must contain all task IDs 551-559")
            for task in CADS_TASKS:
                src = Path(cads[str(task)])
                src = src if src.is_absolute() else study.parent / src
                seg = nib.load(src)
                if seg.shape != image.shape or not np.allclose(seg.affine, image.affine, rtol=0, atol=1e-3):
                    raise ValueError("CADS masks must match the native CT voxel grid and RAS affine")
                arr = np.asarray(seg.dataobj)
                if not np.isfinite(arr).all() or not np.array_equal(arr, arr.astype(np.uint8)):
                    raise ValueError("CADS mask labels must be uint8 IDs")
                dst = work / ("mask_%03d_%d.nii.gz" % (i, task))
                nib.save(nib.Nifti1Image(arr.astype(np.uint8), image.affine), dst)
                masks[str(task)] = str(dst)
        else:
            needs_cads = True
            # Names make the CADS output lookup unambiguous for multiple reconstructions.
            (work / "cads_in" / ("series%03d.nii.gz" % i)).symlink_to(raw)
        records.append(dict(index=i, raw=str(raw), windows=[list(w) for w in loaded.windows],
                            add_mip=bool(item.get("add_mip", True)), cads_parts=masks))
        warnings.extend(loaded.warnings)
    normalized = dict(study_id=uid, study_type_ids=types, warnings=warnings, series=records, needs_cads=needs_cads)
    save_json(work / "input.json", normalized)
    return normalized


def load_cads_parts(item: dict, work: Path, shape, affine) -> dict[int, np.ndarray]:
    """CADS label maps of one series (supplied masks, else the CADS output under work/cads_out), checked against the CT."""
    import nibabel as nib
    parts = {}
    for task in CADS_TASKS:
        p = item["cads_parts"].get(str(task))
        if p is None:
            found = list((work / "cads_out").rglob("*series%03d*part_%d*.nii.gz" % (item["index"], task)))
            if not found:
                found = list((work / "cads_out").rglob("*series%03d*%d*.nii.gz" % (item["index"], task)))
            if len(found) != 1:
                raise RuntimeError("Missing/ambiguous CADS output for series %d task %d" % (item["index"], task))
            p = found[0]
        seg = nib.load(p)
        if seg.shape != tuple(shape) or not np.allclose(seg.affine, affine, rtol=0, atol=1e-3):
            raise ValueError("CADS output geometry does not match its CT")
        parts[task] = np.asarray(seg.dataobj, dtype=np.uint8)
    return parts
