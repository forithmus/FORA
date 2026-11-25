"""CT report generation CLI (paper section "Grounded report generation").

One process per GPU. Each process takes every `num_shards`-th study of the sorted manifest (shard = RANK, GPU =
LOCAL_RANK under torchrun), builds the trained writer once and, per study,
  1. encodes all block-24 tokens of the study (DINO block-23 cache -> MIL block 24 -> 512-latent resampler prefix;
     on CUDA OOM the study is retried with a uniform token subsample, `--fallback-caps`),
  2. builds the diagnostic plan from the frozen MIL logits (confident p >= 0.5, possible >= train Youden, possible tier
     capped at 20 by p, probability shown relative to the base rate), its location / measurement descriptions,
     evidence focus tokens and zoomed evidence images,
  3. decodes one report (`decode`: T0.8 / top-p 0.9 / per-study seed, optional forced opener, optional phrase ban).
Records go to OUTPUT/records_rank<shard>.jsonl (resumable by uid); `--merge-only` writes predictions.jsonl and
summary.json in manifest order.

Paper settings (configs/decode_ct_*.json): FORA val T0.8, no opener, no ban; CT-RATE val T0.8 + opener 'chest' + ban
list 'old_imaging' (1,491 studies; the other 73 studies use 'prior_comparison'); Merlin test T0.8 + opener 'abdomen' +
ban list 'comparative'.

    python -m fora_report.decoding.run --config configs/decode_ct_ctrate.json --checkpoint DPO.pt --llm-path GEMMA4 \
        --manifest ctrate_val.jsonl --cache-root CACHE --mil-config MIL.json --dense DENSE --finding-store LOC \
        --descriptions DESC.npy --ftok FTOK --study-meas STUDY_MEAS.json --calibration CAL.json --output OUT
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch

from fora.reportgen.decoding import bans
from fora.reportgen.decoding.generator import (DEFAULT_FALLBACK_CAPS, FINAL_SWITCHES, build_generator, caps_to_try, decode,
                                              dino_prefix, generation_inputs, generation_stop_ids, set_finding_images)
from fora.reportgen.decoding.opener import opener_text
from fora.reportgen.decoding.sampling import SamplingConfig

from . import cohorts, records

PATH_KEYS = ("checkpoint", "llm_path", "manifest", "cache_root", "mil_config", "mil_checkpoint", "dense", "finding_store",
             "descriptions", "ftok", "study_meas", "calibration", "output")


def compress_placeholders(text: str) -> str:
    import re
    return re.sub(r"((?:<\|video\|>|<\|audio\|>))(?:\1)+",
                  lambda m: f"{m.group(1)}x{len(m.group(0)) // len(m.group(1))}", text)


def arguments(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    known, _ = pre.parse_known_args(argv)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[pre])
    for key in PATH_KEYS:
        p.add_argument("--" + key.replace("_", "-"), default="")
    p.add_argument("--checkpoint-id", default="", help="provenance recorded in records (default: resolved --checkpoint)")
    p.add_argument("--sampling", default="t08", choices=("t08", "greedy"))
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed-index", type=int, default=42, help="per-study seed = sha256('temperature_v1|uid|<index>')")
    p.add_argument("--opener", default="none", help="none | chest | abdomen | literal opener text")
    p.add_argument("--ban-list", default="none", help="none | prior_comparison | old_imaging | comparative | path to a ban-list json")
    p.add_argument("--only-uids", default="", help="optional json list of uids (e.g. the 73 CT-RATE studies decoded with the prior_comparison list)")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42, help="seeds the fallback-cap token subsample only")
    p.add_argument("--dino-token-cap", type=int, default=0, help="0 = all block-24 tokens")
    p.add_argument("--fallback-caps", default=",".join(map(str, DEFAULT_FALLBACK_CAPS)))
    p.add_argument("--dino-crops-per-chunk", type=int, default=32)
    p.add_argument("--dino-gpu-bag-gib", type=float, default=8.0)
    p.add_argument("--restrict-store-to-plan", type=int, default=0,
                   help="1: use only finding-store rows of planned classes (external stores hold the union of several plans)")
    p.add_argument("--save-prompt", type=int, default=0, help="store the decoded plan prompt (placeholders run-length coded)")
    p.add_argument("--plan-possible-cap", type=int, default=FINAL_SWITCHES["plan_possible_cap"])
    p.add_argument("--plan-cap-rank", default=FINAL_SWITCHES["plan_cap_rank"], choices=("prob", "fold"))
    p.add_argument("--plan-prob", type=int, default=int(FINAL_SWITCHES["plan_prob"]))
    p.add_argument("--fimg-tokens", type=int, default=FINAL_SWITCHES["fimg_tokens"])
    p.add_argument("--fimg-views", type=int, default=FINAL_SWITCHES["fimg_views"])
    p.add_argument("--fimg-soft", type=int, default=FINAL_SWITCHES["fimg_soft"])
    p.add_argument("--num-shards", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("RANK", "0")))
    p.add_argument("--merge-only", action="store_true", help="CPU: rebuild predictions.jsonl / summary.json from the shards")
    if known.config:
        config = {k: v for k, v in json.loads(Path(known.config).read_text()).items() if not k.startswith("_")}
        unknown = set(config) - {a.dest for a in p._actions}
        if unknown:
            raise SystemExit(f"{known.config}: unknown keys {sorted(unknown)}")
        p.set_defaults(**config)
    return p.parse_args(argv)


def sampling_from(args) -> SamplingConfig:
    if args.sampling == "greedy":
        return SamplingConfig(do_sample=False)
    return SamplingConfig(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, seed_index=args.seed_index)


def selected_uids(args) -> tuple[list[dict], list[str]]:
    studies = cohorts.read_manifest(args.manifest)
    if args.only_uids:
        keep = set(json.loads(Path(args.only_uids).read_text()))
        studies = [s for s in studies if s["uid"] in keep]
    studies.sort(key=lambda s: str(s["uid"]))
    return studies, [str(s["uid"]) for s in studies]


def main(argv=None) -> int:
    args = arguments(argv)
    output = Path(args.output)
    checkpoint_id = args.checkpoint_id or str(Path(args.checkpoint).resolve())
    studies, order = selected_uids(args)
    if args.merge_only:
        summary, missing = records.merge_outputs(output, order, args.num_shards, checkpoint_id, args.max_new_tokens)
        print(f"[merge] {json.dumps(summary, sort_keys=True)}", flush=True)
        return 1 if missing else 0

    from fora_report.ct.mil_logits import DenseMILLogitStore
    from fora.reportgen.evidence.finding_store import FindingLocStore
    from fora.reportgen.writer.dino_stream import DinoVisualStream, prepare_dino_image, token_dim

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    mine = studies[args.shard_index::args.num_shards]
    output.mkdir(parents=True, exist_ok=True)

    config = json.loads(Path(args.mil_config).read_text())
    calibration = json.loads(Path(args.calibration).read_text())
    labels = list(calibration["labels"])
    switches = {k: getattr(args, k) for k in FINAL_SWITCHES}
    switches["plan_prob"] = bool(switches["plan_prob"])
    package = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, tokenizer, optimizer_step, plan_builder, meta, train_args, calibration_path = build_generator(
        args.checkpoint, args.llm_path, labels, device, args.calibration, visual_dim=token_dim(config), package=package,
        switches=switches)
    del package
    if not meta.get("finding_feature_dim"):
        raise RuntimeError("expected a writer trained with finding tokens")
    stop_ids = generation_stop_ids(tokenizer, args.llm_path)
    stream = DinoVisualStream(args.mil_config, args.mil_checkpoint or train_args.get("dino_mil_checkpoint")
                              or train_args.get("mil_checkpoint"), device,
                              crops_per_chunk=args.dino_crops_per_chunk, gpu_bag_gib=args.dino_gpu_bag_gib)
    if stream.dim != token_dim(config) or stream.dim != model.dino_token_dim:
        raise RuntimeError("DINO token width mismatch between stream, MIL config and checkpoint")
    finding_store = FindingLocStore(args.finding_store, labels, descriptions=args.descriptions, ftok=args.ftok,
                                    study_meas=args.study_meas, fimg_views=args.fimg_views)
    dense = DenseMILLogitStore(args.dense, labels, calibration=calibration)
    sampling = sampling_from(args)
    bad_words_ids, ban_info = bans.load_ban_list(args.ban_list)
    opener = opener_text(args.opener)
    opener_ids = tokenizer.encode(opener, add_special_tokens=False) if opener else None
    fallback = [int(x) for x in str(args.fallback_caps).split(",") if x.strip()]

    shard = records.shard_path(output, args.shard_index)
    done = set(records.read_records(shard, checkpoint_id, args.max_new_tokens))
    todo = [s for s in mine if s["uid"] not in done]
    print(f"[ready] host={socket.gethostname()} shard={args.shard_index}/{args.num_shards} step={optimizer_step} "
          f"selected={len(mine)} todo={len(todo)} sampling={sampling} opener={opener!r} ban={ban_info} "
          f"stop_ids={stop_ids}", flush=True)
    dense.require_coverage([s["uid"] for s in todo])

    for count, study in enumerate(todo, 1):
        started = time.time()
        uid, case = str(study["uid"]), study["case"]
        cohorts.verify_staged(case, Path(args.cache_root))
        study_types = torch.tensor(study["study_type_ids"], dtype=torch.long)
        image = prepare_dino_image(case, config, Path(args.cache_root) / config["fingerprint"])
        logits = dense.get(uid, device)
        selection = plan_builder.plan_selection(logits)
        rows = finding_store.rows(uid)
        if args.restrict_store_to_plan:
            plan = set(selection.confident) | set(selection.possible)
            rows = {c: r for c, r in rows.items() if c in plan}
        descriptions = {c: finding_store.describe_row(r) for c, r in rows.items()}
        descriptions[-1] = finding_store.study_line(uid)
        token_classes = finding_store.token_classes(rows)
        order_ = plan_builder.planned_finding_order(logits, token_classes)
        features = finding_store.features([rows[c] for c in order_], device)
        n_images = set_finding_images(model, finding_store, rows, order_, args.fimg_soft)
        plan_ids = plan_builder.plan_ids(logits, device, location_descriptions=descriptions,
                                         finding_token_classes=token_classes)
        result = None
        for cap in caps_to_try(args.dino_token_cap, fallback):
            bag = prefix = None
            try:
                bag = stream.encode(image, cap=cap, seed_key=f"{args.seed}|{uid}|infer")
                total, used = int(bag.total_tokens), int(bag.tokens.shape[0])
                prefix = dino_prefix(model, bag, study_types)
                del bag
                bag = None
                embeddings, attention, token_types = generation_inputs(model, prefix, logits, plan_ids, features)
                ids = decode(model, embeddings, attention, token_types, max_new_tokens=args.max_new_tokens,
                             stop_ids=stop_ids, uid=uid, sampling=sampling, opener_ids=opener_ids,
                             bad_words_ids=bad_words_ids)
                result = (cap, total, used, ids, int(embeddings.shape[1]))
                break
            except torch.OutOfMemoryError:
                print(f"[oom] uid={uid} cap={cap}: retrying with a smaller cap", flush=True)
                del bag, prefix
                torch.cuda.empty_cache()
        if result is None:
            raise torch.OutOfMemoryError(f"{uid}: OOM at every cap {caps_to_try(args.dino_token_cap, fallback)}")
        cap, total, used, ids, prompt_tokens = result
        new_tokens = int(ids.shape[-1]) - len(opener_ids or [])
        extra = {
            "visual_stream": "dino_b24", "dino_token_cap": int(cap), "dino_tokens_total": total,
            "generated_new_tokens": {"full": new_tokens}, "shard": args.shard_index, "case_id": str(case["id"]),
            "stop_token_ids": stop_ids, "decode_settings": sampling.record(uid),
            "plan_size": len(selection.confident) + len(selection.possible),
            "evidence_findings": len(order_), "evidence_tokens": int(features.shape[0]), "finding_images": int(n_images),
            "study_line": descriptions[-1], "prompt_tokens": prompt_tokens, "plan_text_tokens": int(plan_ids.numel()),
            "truncated": new_tokens >= args.max_new_tokens - len(opener_ids or []),
            "seconds": round(time.time() - started, 1),
        }
        if "protocol_class" in study:
            extra["protocol_class"] = study["protocol_class"]
        if opener:
            extra["forced_prefix"] = opener
        if ban_info:
            extra["phrase_ban"] = ban_info
        if args.save_prompt:
            extra["prompt_text"] = compress_placeholders(tokenizer.decode(plan_ids.tolist(), skip_special_tokens=False))
        record = records.build_record(
            uid=uid, checkpoint_id=checkpoint_id, optimizer_step=optimizer_step, max_new_tokens=args.max_new_tokens,
            reference_findings=study.get("reference_findings"),
            generated=tokenizer.decode(ids[0], skip_special_tokens=True).strip(), plan_builder=plan_builder,
            selection=selection, descriptions=descriptions, n_findings=len(order_), visual_tokens=used,
            modality_tokens=int(study_types.numel()), extra=extra)
        with shard.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        del image, prefix
        torch.cuda.empty_cache()
        print(f"[infer] shard={args.shard_index} {count}/{len(todo)} uid={uid} tokens={used}/{total} new={new_tokens} "
              f"{time.time() - started:.0f}s", flush=True)
    print(f"[done] shard={args.shard_index}; run with --merge-only once every shard has finished "
          f"(calibration {calibration_path})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
