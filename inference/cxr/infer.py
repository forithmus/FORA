#!/usr/bin/env python3
"""End-to-end chest X-ray inference with the FORA chest radiograph model (paper section "Chest radiograph model").

One study (all films of one report) in, one JSON out: 129 finding logits, train-Platt probabilities and plan
decisions with the location text of each finding, the report plan, signed-evidence localization of every planned
finding and the generated Findings text.

    python inference/cxr/infer.py --study study.json --output result.json --work WORK --weights WEIGHTS [--no-report]

Pipeline: films -> X-ray DINO ViT-L blocks 1-23 on 1,024 px tiles at native resolution -> MIL with the learned block
24 and view embedding over all tokens of all films jointly -> plan tiers (device rule and high-AUROC promotion) ->
location text and signed-evidence localization -> Gemma-4-31B writer with greedy decoding (1,024 new tokens).
Runs on one GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cxr_inputs  # noqa: E402
import cxr_localize  # noqa: E402

SCHEMA_VERSION = "fora-cxr-inference/1"
DEFAULT_CONFIG = HERE.parent / "configs" / "cxr.json"
ENV = dict(TOKENIZERS_PARALLELISM="false", HDF5_USE_FILE_LOCKING="FALSE",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", type=Path, required=True, help="study JSON (input contract in inference/README.md)")
    p.add_argument("--output", type=Path, required=True, help="result JSON")
    p.add_argument("--work", type=Path, required=True, help="work directory (evidence maps)")
    p.add_argument("--weights", type=Path, required=True, help="directory with the artifacts listed in inference/weights.json")
    p.add_argument("--gemma", type=Path, default=None, help="Gemma-4-31B-it snapshot (default WEIGHTS/gemma-4-31B-it)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="model and decoding configuration")
    p.add_argument("--no-report", action="store_true", help="skip the writer (classification, plan and localization only)")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args(argv)


class Assets:
    def __init__(self, weights: Path, config: dict, gemma: Path | None = None):
        self.root, self.config, self.art = Path(weights), config, config["artifacts"]
        self.gemma = Path(gemma) if gemma else self.root / self.art["gemma"]

    def __getattr__(self, key):
        art = self.__dict__.get("art", {})
        if key in art:
            return self.root / art[key]
        raise AttributeError(key)


def mil_config(assets: Assets, work: Path) -> Path:
    cfg = json.loads(assets.mil_config.read_text())
    cfg.update(teacher_backbone=str(assets.teacher), block24_state=str(assets.block24_state),
               normalization=str(assets.normalization), classes=str(assets.labels))
    path = work / "mil_config.json"
    path.write_text(json.dumps(cfg))
    return path


def plan_builder_class(assets: Assets):
    """Plan tiers: confident = train-Platt p >= 0.5 (device classes at train Youden), possible = train Youden, then
    possible items of high-AUROC or device classes promoted to confident (at most K = 8 per study, highest p first)."""
    from fora.reportgen.cxr.plan_rules import inference_plan_builder_class
    return inference_plan_builder_class(assets.high_auroc_classes, json.loads(assets.device_classes.read_text()),
                                        int(assets.config["plan"]["promote_k"]))


def plan_builder(assets: Assets, labels, tokenizer):
    from fora.reportgen.cxr.model import make_plan_builder
    return make_plan_builder(str(assets.calibration), labels, tokenizer, plan_builder_class(assets))


def location_texts(labels, groups, locstats) -> dict[int, str]:
    """Side / zone text of every class from the positive leave-one-out evidence on frontal films (writer input)."""
    import numpy as np
    from fora.reportgen.cxr.location import describe, style_table
    table = style_table(labels, groups)
    l16 = np.asarray(locstats).astype(np.float16).astype(np.float32)   # the precision of the training location store
    out = {}
    for c, name in enumerate(labels):
        text = describe(table[name], float(l16[c, 0]), float(l16[c, 1]), l16[c, 2:5].tolist())
        if text:
            out[c] = text
    return out


def write_report(assets: Assets, labels, logits, enc, loc_texts, device):
    """Gemma-4-31B + LoRA writer (adapters attached, not merged) with greedy decoding -> (report, plan builder)."""
    import torch
    from fora.reportgen.cxr.model import STUDY_TYPE_CXR, build_writer, plan_inputs, stop_ids
    dec = assets.config["decoding"]
    model, tok, step, train_args, pb = build_writer(assets.writer, assets.gemma, labels, device, assets.calibration,
                                                    plan_builder_class(assets))
    stops = stop_ids(tok, str(assets.gemma))
    study_types = torch.tensor([STUDY_TYPE_CXR], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        prefix = model.visual_prefix_from_bag(enc, study_types)
        plan, features = plan_inputs(pb, logits, enc.pooled, device, bool(train_args.finding_tokens), loc_texts)
        label_tokens = model.label_conditioner(logits.float()).to(prefix.dtype)
        label_tokens = model.interleave_pathology_evidence(label_tokens, None, return_details=True)["tokens"]
        embedding = model.llm.get_input_embeddings()
        prompt = embedding(plan).unsqueeze(0)
        if model.finding_token_projection is not None:
            prompt = model.insert_finding_tokens(prompt, plan, features)
        embeddings = torch.cat((prefix, label_tokens, prompt.to(prefix.dtype)), dim=1)
        attention = torch.ones(embeddings.shape[:2], dtype=torch.long, device=device)
        token_types = torch.zeros_like(attention)
        token_types[:, : prefix.shape[1] + label_tokens.shape[1]] = 1
        ids = model.llm.generate(inputs_embeds=embeddings.to(embedding.weight.dtype), attention_mask=attention,
                                 token_type_ids=token_types, max_new_tokens=int(dec["max_new_tokens"]), do_sample=False,
                                 num_beams=int(dec["num_beams"]), use_cache=True, eos_token_id=stops,
                                 pad_token_id=stops[0], logits_to_keep=1)
    return dict(text=tok.decode(ids[0], skip_special_tokens=True).strip(), new_tokens=int(ids.shape[-1]),
                truncated=int(ids.shape[-1]) >= int(dec["max_new_tokens"]), optimizer_step=step,
                decoding=dict(greedy=True, num_beams=int(dec["num_beams"]), max_new_tokens=int(dec["max_new_tokens"])),
                prompt_text=tok.decode(plan.tolist(), skip_special_tokens=False)), pb


def predict(a: argparse.Namespace) -> dict:
    import torch
    from fora.reportgen.cxr.stream import CXRVisualStream
    t0 = time.time()
    for k, v in ENV.items():
        os.environ[k] = v
    config = json.loads(a.config.read_text())
    assets = Assets(a.weights, config, gemma=a.gemma)
    study, case = cxr_inputs.load_study(a.study)
    work = a.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    stream = CXRVisualStream(str(mil_config(assets, work)), str(assets.mil), device, variant="raw")
    calibration = json.loads(assets.calibration.read_text())
    labels = list(calibration["labels"])
    assert stream.classes == labels, "Calibration / MIL label order mismatch"
    boxes: dict = {}
    bag = stream.prepare(case, load_film=cxr_inputs.film_loader(case, boxes))
    enc = stream.encode(bag, location=True, keep_scores=True)
    logits = enc.logits.float().reshape(1, -1)
    groups = json.loads(assets.class_groups.read_text())
    loc_texts = location_texts(labels, groups, enc.location)

    result = dict(schema_version=SCHEMA_VERSION, study_id=case["id"], model=config["model"], models=config["models"],
                  films=[dict(index=int(k), **v) for k, v in sorted(boxes.items(), key=lambda kv: int(kv[0]))],
                  visual_tokens=int(enc.total_tokens))
    report = None
    if a.no_report:
        class _NoTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return {"input_ids": [0]}
        pb = plan_builder(assets, labels, _NoTokenizer())
    else:
        report, pb = write_report(assets, labels, logits, enc, loc_texts, device)
    selection = pb.plan_selection(logits)
    probs = pb.probabilities(logits).reshape(-1).tolist()
    conf, poss = set(selection.confident), set(selection.possible)
    result["findings"] = [dict(index=i, name=labels[i], raw_logit=float(enc.logits[i]), probability=float(probs[i]),
                               youden_threshold_prob=float(pb.class_possible_threshold[i]),
                               confident_threshold_prob=float(pb.class_threshold[i]), confident=i in conf,
                               possible=i in poss, location_text=loc_texts.get(i)) for i in range(len(labels))]
    result["plan"] = dict(confident=[labels[i] for i in selection.confident], possible=[labels[i] for i in selection.possible])
    b_ref = json.loads(assets.localization_reference.read_text())["b_mean"]
    planned = [int(i) for i in list(selection.confident) + list(selection.possible)]
    result["localization"] = cxr_localize.localize(
        enc.token_logits, enc.weights, stream.head.output_scale, stream.film_geometry(bag), bag["grids"], case["films"],
        boxes, planned, b_ref, labels, work, config["localization"]) if planned else []
    if report is not None:
        result["report"] = report
        if report["optimizer_step"] != config["models"]["writer_pudpo_step"]:
            raise RuntimeError("Writer checkpoint step differs from the configuration")
    result["seconds"] = round(time.time() - t0, 1)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    return result


def main(argv=None) -> None:
    predict(parse_args(argv))


if __name__ == "__main__":
    main()
