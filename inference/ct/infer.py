#!/usr/bin/env python3
"""End-to-end CT inference with the FORA model.

One study (all reconstructions of one CT examination) in, one JSON out: 1,028 hierarchical finding logits, train-Platt
probabilities and plan decisions, signed-evidence foci with their locations, the report plan and the generated
Findings text.

    python inference/ct/infer.py --study study.json --output result.json --work WORK --weights WEIGHTS \
        [--cads-command "python /opt/CADS/cads/scripts/predict_images.py"] [--measurements on|off] [--no-report]

Pipeline (paper sections "Image encoder", "Organ-routed multiple-instance classifier", "Grounded report generation"):
native CT -> CADS organ masks (one call for all series, skipped when masks are supplied) -> 3-D DINO ViT-L blocks
1-23 on every series and intensity view -> organ-routed MIL with the learned block 24 over all tokens of all views ->
plan tiers -> signed evidence, foci and evidence images of every planned finding (-> CADS anatomy measurements and
lesion calipers) -> Gemma-4-31B writer. Intermediates stay in --work, which must be a new directory for each study.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ct_inputs  # noqa: E402
import ct_pipeline  # noqa: E402

SCHEMA_VERSION = "fora-ct-inference/1"
DEFAULT_CONFIG = HERE.parent / "configs" / "ct.json"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", type=Path, required=True, help="study JSON (input contract in inference/README.md)")
    p.add_argument("--output", type=Path, required=True, help="result JSON")
    p.add_argument("--work", type=Path, required=True, help="new, empty work directory for this study")
    p.add_argument("--weights", type=Path, required=True, help="directory with the artifacts listed in inference/weights.json")
    p.add_argument("--gemma", type=Path, default=None, help="Gemma-4-31B-it snapshot (default WEIGHTS/gemma-4-31B-it)")
    p.add_argument("--dinov2", type=Path, default=None, help="DINOv2-L snapshot for the lesion calipers (default WEIGHTS/dinov2-large)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="model and decoding configuration")
    p.add_argument("--cads-command", default="", help="CADS predict command; '-in DIR -out DIR -task all -np 4 -ns 4' is appended. "
                                                     "Needed unless every series supplies cads_parts")
    p.add_argument("--measurements", choices=["on", "off"], default="on",
                   help="CADS anatomy measurements + lesion calipers as writer input (off for the paper CT-RATE report rows)")
    p.add_argument("--no-report", action="store_true", help="skip the writer (classification, plan and evidence foci only)")
    p.add_argument("--mil-only", action="store_true", help="classification and plan only (no evidence, measurements or writer)")
    p.add_argument("--opener", choices=list(ct_pipeline.OPENER_CHOICES), default="auto")
    p.add_argument("--ban", choices=list(ct_pipeline.BAN_CHOICES), default="default")
    p.add_argument("--token-cap", type=int, default=0, help="visual-token cap of the writer (0 = all body tokens)")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args(argv)


def check_work(work: Path) -> Path:
    work = work.resolve()
    if work.exists() and any(work.iterdir()):
        raise ValueError("Choose a new or empty --work directory")
    work.mkdir(parents=True, exist_ok=True)
    return work


def select_device(name: str):
    import torch
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required (or pass --device cpu for tests)")
        torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    return device


def finding_records(labels, logits, probs, selection, calibration) -> list[dict]:
    confident, possible = set(selection.confident), set(selection.possible)
    return [dict(index=i, name=n, raw_logit=float(logits[i]), probability=float(probs[i]),
                 threshold=float(calibration["class_plan_threshold"][i]),
                 possible_threshold=float(calibration["class_possible_threshold"][i]),
                 confident=i in confident, possible=i in possible) for i, n in enumerate(labels)]


def predict(a: argparse.Namespace) -> dict:
    import torch
    t0 = time.time()
    config = json.loads(a.config.read_text())
    assets = ct_pipeline.Assets(a.weights, config, gemma=a.gemma, dinov2=a.dinov2)
    work = check_work(a.work)
    data = ct_inputs.prepare_input(a.study.resolve(), work)
    if data["needs_cads"]:
        if not a.cads_command:
            raise ValueError("--cads-command is required when a series has no cads_parts")
        ct_pipeline.run_cads(work, shlex.split(a.cads_command), config["cads"]["args"])
    device = select_device(a.device)
    ct_pipeline.set_numerics("classifier", assets)
    config_path, mil_config = ct_pipeline.mil_config(assets, work)
    case = ct_pipeline.extract(data, work, assets, mil_config, device)
    logits, labels = ct_pipeline.classify(case, work, assets, mil_config, device)
    calibration = json.loads(assets.calibration.read_text())
    assert labels == calibration["labels"], "Calibration / MIL label order mismatch"
    selection, probs = ct_pipeline.plan_selection(assets, labels, logits)
    result = dict(schema_version=SCHEMA_VERSION, study_id=data["study_id"], model=config["model"], models=config["models"],
                  measurements=None if a.mil_only else a.measurements, study_type_ids=data["study_type_ids"],
                  warnings=data["warnings"],
                  windows=[dict(series=v["series"], **v["window"], body_tokens=v["n_tokens"]) for v in case["views"]],
                  findings=finding_records(labels, logits, probs, selection, calibration),
                  plan=dict(confident=[labels[i] for i in selection.confident],
                            possible=[labels[i] for i in selection.possible]))
    if not a.mil_only:
        uid = data["study_id"]
        targets = sorted(set(selection.confident) | set(selection.possible))
        anatomy = ct_pipeline.anatomy_measurements(work, assets, targets) if a.measurements == "on" else None
        if anatomy is None:
            ct_pipeline.no_measurements(work, uid)
        ct_pipeline.evidence(work, config_path, assets, targets, labels, uid, device, with_measurements=a.measurements == "on")
        ct_pipeline.cleanup_gpu()
        region_names = json.loads(assets.region_names.read_text())
        result["evidence"] = ct_pipeline.evidence_foci(work, labels, region_names, case["views"])
        lesion_path = work / "calipers/measurements.json"
        result["measurements_output"] = None if anatomy is None else dict(
            records=anatomy["records"], reference=anatomy.get("ref", {}), errors=anatomy["err"],
            lesions=json.loads(lesion_path.read_text()) if lesion_path.exists() else [])
        if not a.no_report:
            ct_pipeline.set_numerics("writer", assets)
            with torch.inference_mode():
                result["report"] = ct_pipeline.report(data, case, work, config_path, assets, logits, labels, device,
                                                      a.token_cap, a.opener, a.ban)
            if result["report"]["optimizer_step"] != config["models"]["writer_dpo_step"]:
                raise RuntimeError("Writer checkpoint step differs from the configuration")
    result["seconds"] = round(time.time() - t0, 2)
    ct_inputs.save_json(work / "result.json", result)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main(argv=None) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    predict(parse_args(argv))


if __name__ == "__main__":
    main()
