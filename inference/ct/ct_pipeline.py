"""CT inference stages: CADS -> encoder (blocks 1-23) -> organ-routed MIL -> plan -> evidence
foci and renders -> measurements -> Gemma-4-31B writer (paper sections "Image encoder", "Organ-routed
multiple-instance classifier", "Grounded report generation").

Model code is imported from the shared package (``fora.mil``, ``fora.reportgen``). This file moves data between the
stages and keeps the intermediate files of one study inside its work directory.
"""
from __future__ import annotations

import gc
import json
import subprocess
from pathlib import Path

import numpy as np

from ct_inputs import load_cads_parts, save_json

# Phrase-ban lists of the CLI (fora/reportgen/decoding/ban_lists): default = the Merlin list, ctrate / ctrate_alt = the two
# lists of the CT-RATE report rows (1,491 / 73 studies).
BAN_LISTS = {"default": "comparative", "ctrate": "old_imaging", "ctrate_alt": "prior_comparison"}
BAN_CHOICES = (*BAN_LISTS, "none")
OPENER_CHOICES = ("auto", "chest", "abdomen", "none")


class Assets:
    """Resolves the artifact names of configs/ct.json against the weights directory."""

    def __init__(self, weights: Path, config: dict, gemma: Path | None = None, dinov2: Path | None = None):
        self.root = Path(weights)
        self.config = config
        self.art = config["artifacts"]
        self.gemma = Path(gemma) if gemma else self.root / self.art["gemma"]
        self.dinov2 = Path(dinov2) if dinov2 else self.root / self.art["dinov2"]

    def __getattr__(self, key):
        art = self.__dict__.get("art", {})
        if key in art and isinstance(art[key], str):
            return self.root / art[key]
        raise AttributeError(key)

    def caliper_head(self, group: str) -> Path:
        return self.root / self.art["caliper_heads"].format(group=group)

    def caliper_scale(self, group: str) -> Path:
        return self.root / self.art["caliper_scales"].format(group=group)


def mil_config(assets: Assets, work: Path) -> tuple[Path, dict]:
    """Classifier config with the artifact paths resolved and the one-study case and assignment files."""
    config = json.loads(assets.mil_config.read_text())
    for key in ("normalization", "block24_state", "heads", "class_region_bits"):
        config[key] = str(getattr(assets, key))
    config.update(cases=str(work / "cases.jsonl"), assign=str(work / "assign.json"))
    path = work / "config.json"
    path.write_text(json.dumps(config))
    return path, config


def run_cads(work: Path, command: list[str], args: list[str]) -> None:
    """CADS (external, task IDs 551-559) on every staged series: `command -in work/cads_in -out work/cads_out ARGS`."""
    out = work / "cads_out"
    out.mkdir(exist_ok=True)
    subprocess.run([*command, "-in", str(work / "cads_in"), "-out", str(out), *args], check=True)


def _series_views(item: dict, lung_protocol: bool, mip_rule: str) -> list[dict]:
    from fora.geometry.windows import plan_views, view_list
    windows, mip = plan_views(item["windows"][1:], lung_protocol=lung_protocol, mip_rule=mip_rule)
    return view_list(windows, mip if item["add_mip"] else None)


def extract(data: dict, work: Path, assets: Assets, config: dict, device) -> dict:
    """Every series and view -> block-23 tokens (int8 + 8 outliers cache, the classifier's training input) + per-token
    CADS region bits. Also writes the native CT and CADS parts for the measurement stage. Returns the MIL case."""
    import h5py
    import nibabel as nib
    import torch
    from fora.cache.encode import Block23Encoder
    from fora.cache.format import FORMAT, quantize, write_group
    from fora.encoder.backbone import load_teacher
    from fora.geometry.protocols import is_lung_study
    from fora.geometry.tiling import TARGET_SPACING, body_frame, resample_series
    from fora.geometry.windows import slab_mip
    from fora.regions import REGION_NAMES
    from fora.regions.builder import Builder, work_ratio
    from fora.regions.token_bits import bits_from_builder, sample_identity, token_body_work, work_axes

    enc_cfg = assets.config["encoder"]
    encoding = json.loads(assets.encoding.read_text())
    assert list(REGION_NAMES) == json.loads(assets.region_names.read_text()), "Region order mismatch"
    backbone = load_teacher(assets.encoder, device)
    if torch.device(device).type == "cuda":
        # Parameters in bf16 (the precision the token cache was computed in), RoPE buffers stay FP32.
        for parameter in backbone.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
    encoder = Block23Encoder(backbone, device, brick_batch=enc_cfg["brick_batch"])
    uid = data["study_id"]
    case = dict(id=uid, report=uid, views=[])
    cache = work / "cache"
    cache.mkdir(parents=True)
    mcase = dict(case=uid, report=uid, split="test", h5="study", patient=uid, series=[], classes=[])
    (work / "measure").mkdir(exist_ok=True)
    lung = is_lung_study(data["study_type_ids"])
    mip_rule = assets.config["views"]["mip_rule"]
    r = work_ratio()
    with h5py.File(work / "measure/cads_study.h5", "w") as cadsh, h5py.File(work / "measure/study.h5", "w") as cth:
        for item in data["series"]:
            si = item["index"]
            raw = nib.load(item["raw"])
            orig = np.asarray(raw.dataobj)
            aff = raw.affine
            parts = load_cads_parts(item, work, orig.shape, aff)
            sid = "series%03d" % si
            key = "/".join([uid, uid, uid, sid])
            cg = cadsh.create_group(key)
            cg.create_dataset("image_affine", data=aff)
            cg.attrs["image_shape"] = orig.shape
            for task, arr in parts.items():
                cg.create_dataset("part_%d" % task, data=arr, compression="lzf")
            ctg = cth.create_group(key)
            ctg.create_dataset("data", data=orig[..., None], compression="lzf")
            ctg.create_dataset("affine", data=aff)
            mcase["series"].append([uid, sid])
            vol, _ = resample_series(orig, aff)
            y0, y1, x0, x1, _, _ = body_frame(vol)
            vol = np.ascontiguousarray(vol[:, y0:y1, x0:x1], dtype=np.float32)
            np.save(work / ("ct_%03d.npy" % si), np.clip(vol, -1024, 3071).astype(np.int16))
            prov = dict(affine=aff.tolist(), orig_shape=list(orig.shape), body_crop_yxyx=[int(y0), int(y1), int(x0), int(x1)],
                        resampled_body_shape=list(vol.shape), target_spacing=list(TARGET_SPACING), reconstruction_index=si,
                        h5="study", patient=uid, report=uid, acc=uid, sid=sid, sub="", frame=0)
            windows = _series_views(item, lung, mip_rule)
            file = cache / ("series_%03d.h5" % si)
            flat = grid = None
            with h5py.File(file, "w") as f:
                f.attrs["provenance_json"] = json.dumps(prov)
                f.attrs["format"] = FORMAT
                for vi, w in enumerate(windows):
                    v = slab_mip(vol, assets.config["views"]["lung_mip_mm"], TARGET_SPACING[0]) \
                        if w["kind"] == "lung_mip_10mm" else vol
                    h, _crops, idx, meta = encoder.encode(v, (w["lo"], w["hi"]), vol, encoding, patch=tuple(enc_cfg["patch"]))
                    if flat is None:
                        flat, grid = idx, meta["grid"]
                    else:
                        assert np.array_equal(flat, idx)
                    q = quantize(h, flat, grid)
                    del h
                    write_group(f, "v%02d" % vi, q, meta, w)
                f.create_dataset("flat_indices", data=flat)
                f.attrs["complete"] = 1
            coords, ras2orig, wsp, ok = work_axes(prov, grid)
            if not ok:
                raise ValueError("CADS region geometry check failed")
            _, body = token_body_work(flat, grid, r)
            sampled = {t: sample_identity(a, coords, ras2orig) for t, a in parts.items()}
            bits, _ = bits_from_builder(Builder(sampled, wsp, body), flat, r)
            target = work / "regions" / (file.name + ".npz")
            target.parent.mkdir(exist_ok=True)
            np.savez(target, bits=bits)
            for vi, w in enumerate(windows):
                case["views"].append(dict(relative_file=file.name, file=str(file), file_size=file.stat().st_size, offset=0,
                                          group="v%02d" % vi, series=si, window=w, n_tokens=len(flat), grid=grid))
            del parts, orig, vol, sampled
    del encoder, backbone
    cleanup_gpu()
    save_json(work / "case.json", case)
    (work / "cases.jsonl").write_text(json.dumps(case) + "\n")
    save_json(work / "assign.json", dict(assignments=dict(train=[[]], val=[[case["id"]]])))
    (work / "measure/cases_all.jsonl").write_text(json.dumps(mcase) + "\n")
    return case


def classify(case: dict, work: Path, assets: Assets, config: dict, device):
    """Study bag (all tokens of all series and views jointly) -> 1,028 raw MIL logits."""
    import torch
    from fora.encoder.blocks import build_block24
    from fora.mil.bag import RoutedBag, load_study, make_factory
    from fora.mil.model import build_head, class_bits, forward
    ck = torch.load(assets.mil, map_location="cpu", weights_only=False)
    labels = list(ck["classes"])
    assert labels == json.loads(assets.labels.read_text())
    block = build_block24(torch.load(assets.block24_state, map_location="cpu", weights_only=False))
    block.load_state_dict(ck["block"])
    block = block.to(device).eval()
    training = config["training"]
    head = build_head(len(labels), soft_init=training["model"]["soft_init"])
    head.load_state_dict(ck["head"])
    head = head.to(device).eval()
    del ck
    norm = torch.load(config["normalization"], map_location="cpu", weights_only=False)
    routes = class_bits(json.loads(Path(config["class_region_bits"]).read_text()), labels, device)
    views = load_study(case, work / "cache", work / "regions", training["position_freqs"])
    bag = RoutedBag(views, device, routes, training["min_route_tokens"], assets.config["classifier"]["gpu_bag_gib"])
    factory = make_factory(bag, block, training["model"], norm["mean"].to(device).float(), float(norm["rms"]),
                           training["crops_per_chunk"])
    with torch.inference_mode():
        logits = forward(head, factory)[0].float().cpu()
    del block, head, bag, views
    cleanup_gpu()
    return logits, labels


def plan_selection(assets: Assets, labels: list[str], logits):
    """Calibrated probabilities and the two plan tiers (no tokenizer needed for the selection). The CT defaults of the
    plan builder are the paper settings: possible tier capped at 20 findings ranked by calibrated probability."""
    from fora.reportgen.plan.diagnostic_plan import DiagnosticPlanBuilder
    pb = DiagnosticPlanBuilder(str(assets.calibration), labels, None, assets.config["plan"]["mode"], dataset_kind="ct")
    selection = pb.plan_selection(logits.reshape(1, -1))
    probs = pb.probabilities(logits.reshape(1, -1)).reshape(-1).tolist()
    return selection, probs


def anatomy_measurements(work: Path, assets: Assets, targets: list[int]) -> dict:
    """CADS organ / vessel / midline measurements and their writer text (work/measurement_text.json, study_meas.json)."""
    from fora.reportgen.measure.anatomy import measure_anatomy
    case = json.loads((work / "measure/cases_all.jsonl").read_text())
    case["classes"] = targets
    (work / "measure/cases_all.jsonl").write_text(json.dumps(case) + "\n")
    calibration = json.loads(assets.anatomy_measurement_calibration.read_text())
    result, text, study_meas = measure_anatomy(work / "measure", case, calibration)
    save_json(work / "measurement_text.json", text)
    save_json(work / "study_meas.json", study_meas)
    return result


def no_measurements(work: Path, uid: str) -> None:
    """No image measurements: empty measurement text and study line (the setting of the paper CT-RATE report rows)."""
    save_json(work / "measurement_text.json", {})
    save_json(work / "study_meas.json", {uid: ""})


def evidence(work: Path, config_path: Path, assets: Assets, targets: list[int], labels: list[str], uid: str, device,
             with_measurements: bool) -> None:
    """Signed evidence, foci and renders of every planned finding (work/extent), optional lesion calipers on the
    strongest evidence site, then the writer's finding store (work/ftok, work/loc)."""
    from fora.reportgen.evidence.extract import extract_ct_evidence
    from fora.reportgen.evidence.finding_store import write_study_loc_dir
    from fora.reportgen.evidence.focus_tokens import build_store
    region_names = json.loads(assets.region_names.read_text())
    save_json(work / "targets.json", dict(targets={uid: targets} if targets else {}, case2rep={uid: uid}))
    (work / "case_ids.txt").write_text(uid + "\n")
    extract_ct_evidence(config_path, assets.mil, work / "targets.json", work / "extent", cases=work / "case_ids.txt",
                        cases_jsonl=str(work / "cases.jsonl"), cache_root=work / "cache", region_root=work / "regions",
                        ct_dir=work, device=str(device), **assets.config["evidence"])
    if targets and with_measurements:
        from fora.reportgen.measure.lesion_measure import run as lesion_calipers
        mcfg = assets.config["measurements"]["calipers"]
        groups = json.loads(assets.caliper_groups.read_text())
        common = ["--cases", work / "cases.jsonl", "--h5-root", work / "measure", "--regions-dir", work / "regions",
                  "--encoder-ckpt", assets.encoder, "--mil-ckpt", assets.mil, "--mil-config", config_path,
                  "--normalization", assets.normalization, "--dino2d", assets.dinov2]
        if mcfg["ln_short_axis"]:
            common.append("--ln-short-axis")
        lesion_text = lesion_calipers(
            work / "extent", work / "calipers", schema=json.loads(assets.measurement_schema.read_text()), groups=groups,
            heads={g: assets.caliper_head(g) for g in groups},
            scales={g: json.loads(assets.caliper_scale(g).read_text()) for g in groups}, common_args=common,
            workers=mcfg["workers"])
        anatomy_text = json.loads((work / "measurement_text.json").read_text())
        # CADS anatomy measurements take precedence over lesion sizes for the same finding.
        save_json(work / "measurement_text.json", dict(lesion_text, **anatomy_text))
    loc, ft = work / "loc", work / "ftok"
    if targets:
        build_store(work / "extent", ft, labels, region_names, json.loads((work / "measurement_text.json").read_text()))
    write_study_loc_dir(loc, ft, uid, len(targets), labels, region_names)


def evidence_foci(work: Path, labels: list[str], region_names: list[str], views: list[dict]) -> list[dict]:
    """Foci of every planned finding from the evidence files: mass share, evidence-weighted centroid and extent in world
    mm (S, R, A), the top three CADS regions with their evidence fractions, the left-side share and the source view."""
    out = []
    for fn in sorted((work / "extent").glob("extent_*.npz")):
        with np.load(fn) as z:
            for j, c in enumerate(z["cls"]):
                foci = []
                for f in range(z["fo_share"].shape[1]):
                    share = float(z["fo_share"][j, f])
                    if share <= 0:
                        continue
                    view = int(z["fo_view"][j, f])
                    regions = [dict(name=region_names[int(r)], fraction=float(fr))
                               for r, fr in zip(z["fo_reg"][j, f], z["fo_regf"][j, f]) if int(r) >= 0]
                    left = float(z["fo_left"][j, f])
                    foci.append(dict(rank=f, mass_share=share, tokens=int(z["fo_ntok"][j, f]),
                                     centroid_mm_sra=[float(v) for v in z["fo_ctr"][j, f]],
                                     extent_mm_sra=[float(v) for v in z["fo_ext"][j, f]], regions=regions,
                                     left_share=left if np.isfinite(left) else None, view_index=view,
                                     series=views[view]["series"] if 0 <= view < len(views) else None))
                out.append(dict(index=int(c), name=labels[int(c)], probability=float(z["prob"][j]), foci=foci))
    return out


def report(data: dict, case: dict, work: Path, config_path: Path, assets: Assets, logits, labels: list[str], device,
           token_cap: int, opener: str, ban: str) -> dict:
    """Gemma-4-31B writer on the visual prefix, the label tokens, the plan with the
    evidence locations, the focus tokens and six evidence images per planned finding. Call under
    torch.inference_mode()."""
    import torch
    from fora.reportgen.decoding import bans
    from fora.reportgen.decoding.generator import (build_generator, decode, dino_prefix, generation_inputs,
                                                   generation_stop_ids, set_finding_images)
    from fora.reportgen.decoding.opener import OPENERS, auto_opener
    from fora.reportgen.decoding.sampling import SamplingConfig
    from fora.reportgen.evidence.finding_store import FindingLocStore
    from fora.reportgen.writer.dino_stream import DinoVisualStream, prepare_dino_image

    wcfg, dec = assets.config["writer"], assets.config["decoding"]
    model, tok, step, pb, _, _, _ = build_generator(str(assets.writer), str(assets.gemma), labels, device,
                                                    str(assets.calibration))
    config = json.loads(config_path.read_text())
    uid = data["study_id"]
    store = FindingLocStore(work / "loc", labels, descriptions=work / "ftok/desc.npy", ftok=work / "ftok",
                            study_meas=work / "study_meas.json")
    rows = store.rows(uid)
    logits = logits.to(device).reshape(1, -1)
    descriptions = store.descriptions(uid)
    descriptions[-1] = store.study_line(uid)
    tc = store.token_classes(rows)
    order = pb.planned_finding_order(logits, tc)
    features = store.features([rows[c] for c in order], device)
    set_finding_images(model, store, rows, order, wcfg["finding_image_soft_tokens"])
    ids = pb.plan_ids(logits, device, location_descriptions=descriptions, finding_token_classes=tc)
    stream = DinoVisualStream(str(config_path), str(assets.mil), device, crops_per_chunk=wcfg["crops_per_chunk"],
                              gpu_bag_gib=wcfg["gpu_bag_gib"])
    image_dir = work / "evidence_images"
    image_dir.mkdir(exist_ok=True)
    localized = []
    for c in order:
        r = rows[c]
        images = []
        for vi, jpeg in enumerate(store.images([r])[0]):
            if jpeg:
                name = "%04d_%d.jpg" % (c, vi)
                (image_dir / name).write_bytes(jpeg)
                images.append(dict(view_index=vi, path="evidence_images/" + name))
        localized.append(dict(index=c, name=labels[c], description=descriptions[c],
                              regions=[store.region_names[int(v)] for v in store.reg3[r]], images=images))
    bag = stream.encode(prepare_dino_image(case, config, work / "cache"), cap=token_cap,
                        seed_key=wcfg["visual_seed_key"].format(study_id=uid))
    total, used = int(bag.total_tokens), len(bag.tokens)
    prefix = dino_prefix(model, bag, torch.tensor(data["study_type_ids"], device=device))
    del bag
    emb, attention, types = generation_inputs(model, prefix, logits, ids, features)
    stops = generation_stop_ids(tok, str(assets.gemma))
    opener_name = auto_opener(data["study_type_ids"]) if opener == "auto" else opener
    opener_text = OPENERS[opener_name]
    opener_ids = tok.encode(opener_text, add_special_tokens=False) if opener_text else []
    bad, ban_info = bans.load_ban_list(BAN_LISTS[ban]) if ban != "none" else (None, None)
    sampling = SamplingConfig(temperature=dec["temperature"], top_p=dec["top_p"], top_k=dec["top_k"],
                              seed_index=dec["seed_index"])
    maximum = int(dec["max_new_tokens_total"])
    generated = decode(model, emb, attention, types, max_new_tokens=maximum, stop_ids=stops, uid=uid, sampling=sampling,
                       opener_ids=opener_ids, bad_words_ids=bad)
    new = int(generated.shape[-1]) - len(opener_ids)
    return dict(text=tok.decode(generated[0], skip_special_tokens=True).strip(), new_tokens=new,
                truncated=new >= maximum - len(opener_ids), optimizer_step=step,
                decoding=dict(temperature=dec["temperature"], top_p=dec["top_p"], top_k=dec["top_k"],
                              seed=sampling.seed_for(uid), seed_index=dec["seed_index"], opener=opener_name,
                              opener_text=opener_text, phrase_ban=None if ban_info is None else dict(name=ban, **ban_info)),
                visual_tokens=used, total_visual_tokens=total, token_cap=token_cap,
                prompt_text=tok.decode(ids.tolist(), skip_special_tokens=False), study_measurements=store.study_line(uid),
                localization_descriptions={labels[c]: descriptions[c] for c in rows}, localizations=localized)


def set_numerics(stage: str, assets: Assets) -> None:
    """TF32 off for encoder, classifier and evidence, on ('high') for the writer."""
    import torch
    if stage == "classifier":
        torch.backends.cuda.matmul.allow_tf32 = bool(assets.config["classifier"]["allow_tf32"])
        torch.backends.cuda.enable_cudnn_sdp(bool(assets.config["classifier"]["cudnn_sdp"]))
    else:
        torch.backends.cuda.matmul.allow_tf32 = bool(assets.config["writer"]["allow_tf32"])
        torch.set_float32_matmul_precision(assets.config["writer"]["float32_matmul_precision"])


def cleanup_gpu() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

