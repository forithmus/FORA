#!/usr/bin/env python3
"""Per-study CT protocol classification into a list of protocol classes (Methods, "Protocol classes").

A CT study can bundle more than one exam (for example a head and a chest CT), so every study gets the ordered list of
distinct protocols it contains. Per study: PET/CT studies (a PT series or 'pet' in the series / study / protocol name,
propagated over the study and its accession) are labelled pet_ct, studies without a CT series no_ct; all other studies
are rendered into one prompt (code-computed contrast / phase counts, study fields once, distinct series tuples with a
(xN) count), classified by the model into a JSON array of {n, title}, and checked by one QC pass (PASS / FAIL).
Studies whose prompt does not fit the context are labelled other / too_long. Unique studies are sharded over ranks
(SLURM_PROCID / SLURM_NTASKS); merge_shards.py joins the shards back onto every series row.

Usage (one process per GPU)
  python classify.py --input classification_metadata.csv --output-dir SHARDS --log-dir LOGS [--model google/gemma-4-31B-it]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import pandas as pd

import config as cfg
from study_features import render_derived_features
from vllm_inferer import ChatMessages, VLLMInferer


# ─────────────────────────────────────────────────────────────────────────────
# Per-study machinery: protocol menu + prompt builders, PET/no-CT filters, study
# grouping, logging, output cleaning (constants from ``config``).
# ─────────────────────────────────────────────────────────────────────────────
def load_keep_classes(path: str = cfg.PROTOCOL_CLASSES_JSON) -> list[dict]:
    """Return the protocol classes as a flat list ``[{n, title, region}, ...]``."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_class_menu(classes: list[dict]) -> str:
    """Render the keep classes as a numbered, region-grouped menu for the prompt."""
    by_region: "OrderedDict[str, list[dict]]" = OrderedDict()
    for c in classes:
        by_region.setdefault(c["region"], []).append(c)
    lines: list[str] = []
    for region, items in by_region.items():
        lines.append(f"## {region}")
        for c in items:
            lines.append(f"  {c['n']}. {c['title']}")
    return "\n".join(lines)


def build_study_prompt(study_level: dict, series_rows: list[dict]) -> str:
    """Render the per-study user message.

    ``study_level`` maps each ``STUDY_METADATA_COLS`` name to a (cleaned) string;
    ``series_rows`` is a list of dicts keyed by ``SERIES_METADATA_COLS``. Values
    are assumed already normalized to plain strings ("" for empty/NaN) by the
    caller. Identical series tuples are collapsed and annotated with a "(×N)"
    count to keep prompts compact for studies with many redundant series.
    """
    lines: list[str] = []
    # Code-computed counts / agent set across ALL series, prepended so the model
    # trusts them over its own per-row tally (see study_features).
    feat_block = render_derived_features(series_rows)
    if feat_block:
        lines.append(feat_block)
        lines.append("")

    lines.append("STUDY-LEVEL METADATA:")
    for col in cfg.STUDY_METADATA_COLS:
        val = study_level.get(col, "")
        lines.append(f"  {col}: {val if val else '(empty)'}")

    counts: "OrderedDict[tuple, int]" = OrderedDict()
    for r in series_rows:
        key = tuple(r.get(c, "") for c in cfg.SERIES_METADATA_COLS)
        counts[key] = counts.get(key, 0) + 1

    lines.append("")
    lines.append(
        f"SERIES IN THIS STUDY ({len(series_rows)} series, "
        f"{len(counts)} distinct):"
    )
    for idx, (tup, n) in enumerate(counts.items(), start=1):
        parts = ", ".join(
            f"{c}={v if v else '-'}" for c, v in zip(cfg.SERIES_METADATA_COLS, tup)
        )
        suffix = f"  (×{n})" if n > 1 else ""
        lines.append(f"  [{idx}] {parts}{suffix}")
    return "\n".join(lines)


def pet_study_uids(df: pd.DataFrame) -> set:
    """Return the set of ``StudyInstanceUID``s to drop as PET/CT.

    Flag any
    row whose ``Modality`` is PT or whose Series/Study/ProtocolName text contains
    'pet', then propagate that flag across the WHOLE study (StudyInstanceUID) and
    across any study sharing a non-empty ``AccessionNumber`` with a flagged row.
    """
    modality_mask = (
        df["Modality"].astype(str).str.strip().str.upper() == cfg.PET_MODALITY
    )
    text_mask = pd.Series(False, index=df.index)
    for col in cfg.PET_TEXT_COLS:
        if col in df.columns:
            text_mask |= (
                df[col].astype(str).str.lower().str.contains(cfg.PET_KEYWORD, na=False)
            )

    flagged = df[modality_mask | text_mask]
    pet_uids = set(flagged[cfg.STUDY_COL].dropna().unique())

    if "AccessionNumber" in df.columns:
        flagged_accs = flagged["AccessionNumber"].dropna().astype(str).str.strip()
        flagged_accs = set(flagged_accs[flagged_accs != ""].unique())
        if flagged_accs:
            acc = df["AccessionNumber"].astype(str).str.strip()
            shares_acc = (acc != "") & acc.isin(flagged_accs)
            pet_uids |= set(df.loc[shares_acc, cfg.STUDY_COL].dropna().unique())

    return pet_uids


def no_ct_study_uids(df: pd.DataFrame) -> set:
    """Return the set of ``StudyInstanceUID``s that contain NO CT image series.

    A study is classifiable only if at least one of its series has
    ``Modality == "CT"``; studies made up entirely of non-CT objects (e.g.
    SR-only dose-report studies) have nothing to classify and are labeled
    ``no_ct`` instead of being sent to the model.
    """
    is_ct = df["Modality"].astype(str).str.strip().str.upper() == cfg.CT_MODALITY
    ct_uids = set(df.loc[is_ct, cfg.STUDY_COL].dropna().unique())
    all_uids = set(df[cfg.STUDY_COL].dropna().unique())
    return all_uids - ct_uids


def _make_logger(log_dir: Path, rank: int, world: int) -> logging.Logger:
    """Per-rank logger that writes to file + stdout."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / cfg.CLS_LOG_FILENAME_TEMPLATE.format(rank=rank)

    logger = logging.getLogger(f"classify_protocols.rank{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt=f"%(asctime)s [rank {rank}/{world}] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"Logging to {log_path}")
    return logger


def _clean(value) -> str:
    """Normalize a CSV cell to a plain stripped string ("" for NaN/None)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    return "" if s.lower() == "nan" else s


def _group_studies(
    df: pd.DataFrame,
) -> "dict[str, tuple[dict, list[dict]]]":
    """Group per-series rows into ``{study_uid: (study_level, series_rows)}``.

    Study-level fields take the first non-empty value seen across the study's
    series; series rows keep only ``SERIES_METADATA_COLS``. Rows with an empty
    ``StudyInstanceUID`` are skipped (handled as "" at merge time).
    """
    # Resolve column positions up front. We iterate with name=None (plain
    # positional tuples) rather than ._asdict() because several metadata columns
    # (e.g. "Contrast/BolusAgent", "[AcquisitionType]") are not valid Python
    # identifiers and would be silently renamed to "_<n>" by named itertuples.
    needed = [cfg.STUDY_COL] + cfg.STUDY_METADATA_COLS + cfg.SERIES_METADATA_COLS
    pos = {c: i for i, c in enumerate(df.columns) if c in needed}
    if cfg.STUDY_COL not in pos:
        raise SystemExit(f"Input CSV must contain '{cfg.STUDY_COL}' column.")
    study_pos = cfg.STUDY_METADATA_COLS  # iterate names, look up via pos
    uid_i = pos[cfg.STUDY_COL]

    studies: "dict[str, tuple[dict, list[dict]]]" = {}
    for tup in df.itertuples(index=False, name=None):
        uid = _clean(tup[uid_i])
        if not uid:
            continue
        if uid not in studies:
            studies[uid] = ({c: "" for c in cfg.STUDY_METADATA_COLS}, [])
        study_level, series_rows = studies[uid]
        for c in study_pos:
            if not study_level[c] and c in pos:
                study_level[c] = _clean(tup[pos[c]])
        series_rows.append(
            {c: (_clean(tup[pos[c]]) if c in pos else "") for c in cfg.SERIES_METADATA_COLS}
        )
    return studies


def _strip_preamble(raw: str) -> str:
    text = (raw or "").strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    for prefix in cfg.CLS_OUTPUT_PREFIXES:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            break
    return text


def _truncate(s: str | None, n: int = cfg.CLS_RAW_TRUNCATE_CHARS) -> str:
    if not s:
        return ""
    if len(s) <= n:
        return s
    return s[:n] + " …[truncated]"


def _batched(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ─────────────────────────────────────────────────────────────────────────────
# Output parsing — JSON ARRAY of {n, title} -> list of canonical titles
# ─────────────────────────────────────────────────────────────────────────────
def parse_protocols(raw: str, title_by_n: dict[int, str], title_set: set[str]) -> list[str]:
    """Resolve the model output to a de-duplicated list of canonical titles.

    Accepts ``[{"n": <int>, "title": "<title>"}, ...]``. Per element the class
    number is authoritative (immune to title typos); an exact-match title is the
    fallback. Unknown elements are dropped. ``other`` mixed with real protocols
    is dropped (a real list wins); an empty / unparseable result -> ["other"].
    """
    text = _strip_preamble(raw)
    out: list[str] = []
    seen: set[str] = set()
    try:
        a = text.find("[")
        b = text.rfind("]") + 1
        if a != -1 and b > a:
            arr = json.loads(text[a:b])
            if isinstance(arr, dict):  # tolerate a single object
                arr = [arr]
            for item in arr:
                title = None
                if isinstance(item, dict):
                    try:
                        n = int(item.get("n"))
                    except (TypeError, ValueError):
                        n = None
                    if n is not None and n in title_by_n:
                        title = title_by_n[n]
                    elif n == cfg.OTHER_N:
                        title = cfg.STATUS_OTHER
                    else:
                        t = str(item.get("title", "")).strip()
                        if t in title_set:
                            title = t
                        elif t.lower() == cfg.STATUS_OTHER:
                            title = cfg.STATUS_OTHER
                elif isinstance(item, str) and item.strip() in title_set:
                    title = item.strip()
                if title and title not in seen:
                    seen.add(title)
                    out.append(title)
    except (json.JSONDecodeError, ValueError):
        pass
    if not out:
        return [cfg.STATUS_OTHER]
    real = [t for t in out if t != cfg.STATUS_OTHER]
    return real if real else [cfg.STATUS_OTHER]


def parse_qc_verdict(raw: str) -> tuple[str, str]:
    """Conservative PASS / FAIL / unknown parser of the QC answer."""
    text = (raw or "").strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    issues_str = ""
    try:
        a = text.find("{")
        b = text.rfind("}") + 1
        if a != -1 and b > a:
            parsed = json.loads(text[a:b])
            verdict_raw = str(parsed.get("verdict", "")).strip().upper()
            issues = parsed.get("issues", [])
            issues_str = " | ".join(str(x) for x in issues) if isinstance(issues, list) else str(issues)
            if verdict_raw == "PASS":
                return cfg.QC_STATUS_PASS, issues_str
            if verdict_raw == "FAIL":
                return cfg.QC_STATUS_FAIL, issues_str
    except (json.JSONDecodeError, ValueError):
        pass
    upper = (raw or "").upper()
    if '"PASS"' in upper or "'PASS'" in upper:
        return cfg.QC_STATUS_PASS, issues_str
    if '"FAIL"' in upper or "'FAIL'" in upper:
        return cfg.QC_STATUS_FAIL, issues_str
    return "unknown", issues_str


# ─────────────────────────────────────────────────────────────────────────────
# Message builders
# ─────────────────────────────────────────────────────────────────────────────
def _classify_messages(system_prompt: str, study_prompt: str) -> ChatMessages:
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": study_prompt}]


def _qc_messages(study_prompt: str, protocols_encoded: str) -> ChatMessages:
    user = f"{study_prompt}\n\n---\n\nPROPOSED PROTOCOL LIST: {protocols_encoded}"
    return [{"role": "system", "content": cfg.CLS_QC_SYSTEM_PROMPT},
            {"role": "user", "content": user}]


# ─────────────────────────────────────────────────────────────────────────────
# Batched inference
# ─────────────────────────────────────────────────────────────────────────────
def _classify_batch(inferer, system_prompt, study_prompts, title_by_n, title_set, logger):
    sampling = inferer.make_sampling(
        temperature=cfg.CLS_TEMPERATURE, top_p=cfg.CLS_TOP_P, max_tokens=cfg.CLS_MAX_TOKENS,
    )
    encoded: list[str] = []
    raw_list: list[str] = []
    n_chunks = (len(study_prompts) + cfg.CLS_BATCH_SIZE - 1) // cfg.CLS_BATCH_SIZE
    for idx, chunk in enumerate(_batched(study_prompts, cfg.CLS_BATCH_SIZE), start=1):
        t0 = time.time()
        raws = inferer.chat([_classify_messages(system_prompt, p) for p in chunk], sampling)
        for raw in raws:
            titles = parse_protocols(raw, title_by_n, title_set)
            encoded.append(cfg.encode_protocols(titles))
            raw_list.append(_truncate(raw))
        logger.info(f"  classify chunk {idx}/{n_chunks} ({len(chunk)} studies) in {time.time()-t0:.1f}s")
    return encoded, raw_list


def _qc_batch(inferer, pairs, logger):
    sampling = inferer.make_sampling(
        temperature=cfg.CLS_QC_TEMPERATURE, top_p=cfg.CLS_QC_TOP_P, max_tokens=cfg.CLS_QC_MAX_TOKENS,
    )
    verdicts, issues_list, raw_list = [], [], []
    n_chunks = (len(pairs) + cfg.CLS_QC_BATCH_SIZE - 1) // cfg.CLS_QC_BATCH_SIZE
    for idx, chunk in enumerate(_batched(pairs, cfg.CLS_QC_BATCH_SIZE), start=1):
        t0 = time.time()
        raws = inferer.chat([_qc_messages(p, proto) for (p, proto) in chunk], sampling)
        for raw in raws:
            v, iss = parse_qc_verdict(raw)
            verdicts.append(v)
            issues_list.append(iss)
            raw_list.append(_truncate(raw))
        logger.info(f"  qc chunk {idx}/{n_chunks} ({len(chunk)} studies) in {time.time()-t0:.1f}s")
    return verdicts, issues_list, raw_list


def _write_shard(out_path, shard_uids, protocol, verdict, qc_issues, cls_raw, qc_raw):
    pd.DataFrame({
        cfg.STUDY_COL: shard_uids,
        cfg.PROTOCOL_COL: [protocol[uid] for uid in shard_uids],
        cfg.VERDICT_COL: [verdict[uid] for uid in shard_uids],
        cfg.QC_ISSUES_COL: [
            qc_issues[uid] if verdict[uid] in (cfg.QC_STATUS_FAIL, cfg.STATUS_TOO_LONG) else ""
            for uid in shard_uids
        ],
        "classification_raw_truncated": [cls_raw.get(uid, "") for uid in shard_uids],
        "qc_raw_truncated": [qc_raw.get(uid, "") for uid in shard_uids],
    }).to_csv(out_path, index=False, encoding="utf-8-sig")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--log-dir", required=True, type=Path)
    ap.add_argument("--model", default=cfg.DEFAULT_MODEL)
    ap.add_argument("--max-model-len", type=int, default=cfg.DEFAULT_MAX_MODEL_LEN)
    ap.add_argument("--gpu-memory-utilization", type=float, default=cfg.DEFAULT_GPU_MEMORY_UTIL)
    ap.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    ap.add_argument("--dtype", default=cfg.DEFAULT_DTYPE)
    args = ap.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = _make_logger(args.log_dir, rank, world)
    log.info(f"input={args.input} output_dir={args.output_dir} model={args.model}")

    keep_classes = load_keep_classes()
    class_menu = build_class_menu(keep_classes)
    system_prompt = cfg.CLS_SYSTEM_PROMPT_TEMPLATE.format(class_menu=class_menu)
    title_by_n = {int(c["n"]): c["title"] for c in keep_classes}
    title_set = {c["title"] for c in keep_classes}
    log.info(f"Loaded {len(keep_classes)} keep protocol classes.")

    t_start = time.time()
    df = pd.read_csv(args.input, low_memory=False)
    if cfg.STUDY_COL not in df.columns:
        raise SystemExit(f"Input CSV must contain '{cfg.STUDY_COL}'; got {list(df.columns)}")
    pet_uids = pet_study_uids(df)
    noct_uids = no_ct_study_uids(df) - pet_uids
    studies = _group_studies(df)
    del df
    unique = sorted(studies)
    log.info(f"{len(unique)} studies; pet={sum(u in pet_uids for u in unique)} "
             f"no_ct={sum(u in noct_uids for u in unique)}")

    shard_uids = unique[rank::world]
    out_cols = [cfg.STUDY_COL, cfg.PROTOCOL_COL, cfg.VERDICT_COL, cfg.QC_ISSUES_COL,
                "classification_raw_truncated", "qc_raw_truncated"]
    out_path = args.output_dir / cfg.CLS_SHARD_FILENAME_TEMPLATE.format(rank=rank)
    if not shard_uids:
        pd.DataFrame(columns=out_cols).to_csv(out_path, index=False, encoding="utf-8-sig")
        log.info(f"Empty shard -> {out_path}")
        return

    protocol, verdict = {}, {}
    qc_issues = {u: "" for u in shard_uids}
    cls_raw = {u: "" for u in shard_uids}
    qc_raw = {u: "" for u in shard_uids}
    for uid in shard_uids:
        if uid in pet_uids:
            protocol[uid] = cfg.encode_protocols([cfg.STATUS_PET]); verdict[uid] = cfg.STATUS_PET
        elif uid in noct_uids:
            protocol[uid] = cfg.encode_protocols([cfg.STATUS_NO_CT]); verdict[uid] = cfg.STATUS_NO_CT
    classify_uids = [u for u in shard_uids if u not in pet_uids and u not in noct_uids]
    log.info(f"{len(classify_uids)} studies to classify (shard {len(shard_uids)}).")
    if not classify_uids:
        _write_shard(out_path, shard_uids, protocol, verdict, qc_issues, cls_raw, qc_raw)
        return

    study_prompt = {uid: build_study_prompt(*studies[uid]) for uid in classify_uids}

    inferer = VLLMInferer(
        model_path=args.model, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed,
        dtype=args.dtype, log_fn=log.info,
    )

    # context guard (classification)
    to_classify = []
    n_too_long = 0
    for uid in classify_uids:
        fits, _ = inferer.prompt_fits(
            _classify_messages(system_prompt, study_prompt[uid]),
            cfg.CLS_MAX_TOKENS, safety_tokens=cfg.CONTEXT_LENGTH_SAFETY_TOKENS)
        if fits:
            to_classify.append(uid)
        else:
            protocol[uid] = cfg.encode_protocols([cfg.STATUS_OTHER])
            verdict[uid] = cfg.STATUS_TOO_LONG
            n_too_long += 1
    if n_too_long:
        log.warning(f"{n_too_long} studies too_long (classification).")

    if to_classify:
        log.info(f"Classifying {len(to_classify)} studies ...")
        enc, raws = _classify_batch(
            inferer, system_prompt, [study_prompt[u] for u in to_classify],
            title_by_n, title_set, log)
        for uid, e, raw in zip(to_classify, enc, raws):
            protocol[uid] = e
            cls_raw[uid] = raw

    # QC pass over the proposed lists
    qc_uids, n_qc_too_long = [], 0
    for uid in to_classify:
        fits, _ = inferer.prompt_fits(
            _qc_messages(study_prompt[uid], protocol[uid]),
            cfg.CLS_QC_MAX_TOKENS, safety_tokens=cfg.CONTEXT_LENGTH_SAFETY_TOKENS)
        if fits:
            qc_uids.append(uid)
        else:
            verdict[uid] = cfg.STATUS_TOO_LONG
            qc_issues[uid] = "QC: prompt exceeds limit"
            n_qc_too_long += 1
    if qc_uids:
        log.info(f"QC on {len(qc_uids)} classifications ...")
        verdicts, issues_list, raws = _qc_batch(
            inferer, [(study_prompt[u], protocol[u]) for u in qc_uids], log)
        for uid, v, iss, raw in zip(qc_uids, verdicts, issues_list, raws):
            qc_raw[uid] = raw
            if v == cfg.QC_STATUS_PASS:
                verdict[uid] = cfg.QC_STATUS_PASS
            else:
                verdict[uid] = cfg.QC_STATUS_FAIL
                qc_issues[uid] = iss

    _write_shard(out_path, shard_uids, protocol, verdict, qc_issues, cls_raw, qc_raw)
    vc = Counter(verdict[u] for u in shard_uids)
    log.info(f"FINAL shard={len(shard_uids)} pass={vc.get(cfg.QC_STATUS_PASS,0)} "
             f"fail={vc.get(cfg.QC_STATUS_FAIL,0)} too_long={vc.get(cfg.STATUS_TOO_LONG,0)} "
             f"pet={vc.get(cfg.STATUS_PET,0)} no_ct={vc.get(cfg.STATUS_NO_CT,0)}")
    log.info(f"shard written: {out_path}  ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[classify_protocols] FATAL: {exc!r}", file=sys.stderr, flush=True)
        raise
