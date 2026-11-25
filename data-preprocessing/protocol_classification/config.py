"""Constants of the CT protocol classification (Methods, "Protocol classes").

Each CT study (StudyInstanceUID) is classified from the DICOM metadata of all its series into a LIST of protocol
classes (protocol_classes.json, n = 1..29, plus "other" = 0) by Gemma-4-31B-it with greedy decoding, followed by one
QC pass with the same model. The system prompts are in prompts/.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Columns extracted from the filtered metadata parquet shards (absent columns are filled with NaN).
CLASSIFICATION_COLUMNS: list[str] = [
    "AccessionNumber", "StudyInstanceUID", "SeriesInstanceUID", "ImageTypeText", "Modality", "ProtocolName",
    "SeriesDescription", "StudyDescription", "BodyPartExamined", "RequestedProcedureDescription",
    "ScanOptions", "[AcquisitionType]", "ImagePositionPatient",
    "ConvolutionKernel", "ReconstructionDiameter",
    "KVP", "Exposure", "CTDIvol",
    "Contrast/BolusAgent", "Contrast/BolusStartTime", "AcquisitionNumber", "SeriesNumber", "SeriesTime",
]

# Per-rank shard / log file names.
CLS_SHARD_FILENAME_TEMPLATE = "classified_shard_rank_{rank:02d}.csv"
CLS_LOG_FILENAME_TEMPLATE = "classification_rank_{rank:02d}.log"
CLS_SHARD_GLOB = "classified_shard_rank_*.csv"

STUDY_COL = "StudyInstanceUID"
PROTOCOL_COL = "classified_protocols"        # JSON list of canonical titles, e.g. '["CT Head - Non-Contrast"]'
VERDICT_COL = "classification_verdict"       # pass / fail / too_long / pet_ct / no_ct
QC_ISSUES_COL = "classification_qc_issues"


def encode_protocols(titles: list[str]) -> str:
    return json.dumps(list(titles), ensure_ascii=False)


def decode_protocols(cell) -> list[str]:
    """Parse a PROTOCOL_COL cell (empty -> [], a bare title -> [title])."""
    if cell is None:
        return []
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return []
    try:
        val = json.loads(s)
        return [str(v) for v in val] if isinstance(val, list) else [str(val)]
    except (ValueError, TypeError):
        return [s]


# Study-level fields shown once per study; per-series fields enumerated per distinct series tuple.
STUDY_METADATA_COLS: list[str] = ["StudyDescription", "RequestedProcedureDescription"]
SERIES_METADATA_COLS: list[str] = [
    "Modality", "SeriesDescription", "ProtocolName", "ImageTypeText", "BodyPartExamined",
    "ScanOptions", "[AcquisitionType]", "ImagePositionPatient",
    "ConvolutionKernel", "ReconstructionDiameter",
    "KVP", "Exposure", "CTDIvol",
    "Contrast/BolusAgent", "Contrast/BolusStartTime", "AcquisitionNumber", "SeriesNumber", "SeriesTime",
]

STATUS_OTHER = "other"
OTHER_N = 0
STATUS_TOO_LONG = "too_long"
QC_STATUS_PASS = "pass"
QC_STATUS_FAIL = "fail"
STATUS_PET = "pet_ct"        # studies with a PT series or 'pet' in Series/Study/ProtocolName (whole study + accession)
STATUS_NO_CT = "no_ct"       # studies without any CT series
PET_MODALITY = "PT"
PET_KEYWORD = "pet"
PET_TEXT_COLS: list[str] = ["SeriesDescription", "StudyDescription", "ProtocolName"]
CT_MODALITY = "CT"

PROTOCOL_CLASSES_JSON = str(HERE / "protocol_classes.json")

# vLLM / model
DEFAULT_MODEL = "google/gemma-4-31B-it"
DEFAULT_MAX_MODEL_LEN = 131072
DEFAULT_GPU_MEMORY_UTIL = 0.85
DEFAULT_SEED = 42
DEFAULT_DTYPE = "auto"
CONTEXT_LENGTH_SAFETY_TOKENS = 64

# Sampling: classification and QC are greedy.
CLS_TEMPERATURE = 0.0
CLS_TOP_P = 1.0
CLS_MAX_TOKENS = 768
CLS_BATCH_SIZE = 256
CLS_RAW_TRUNCATE_CHARS = 2048
CLS_QC_TEMPERATURE = 0.0
CLS_QC_TOP_P = 1.0
CLS_QC_MAX_TOKENS = 512
CLS_QC_BATCH_SIZE = 256

# Preambles stripped before JSON parsing.
CLS_OUTPUT_PREFIXES: tuple[str, ...] = ("Here is the classification:", "Classification:", "Answer:", "Output:")

# System prompts. The classification prompt is a str.format template with the {class_menu} placeholder.
CLS_SYSTEM_PROMPT_TEMPLATE = (HERE / "prompts/classify_system.txt").read_text(encoding="utf-8")
CLS_QC_SYSTEM_PROMPT = (HERE / "prompts/qc_system.txt").read_text(encoding="utf-8")
