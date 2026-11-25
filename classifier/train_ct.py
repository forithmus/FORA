"""Train the organ-routed CT finding classifier (Methods, "Finding classifier").

Model: trainable block 24 (+ final norm) over the cached block-23 tokens of every view of a report, fixed standardisation,
Fourier position features, and the organ-routed attention MIL head of ``fora.mil.model`` with soft routing.

Labels: four-state report labels (labels npz: ``nodes``, ``cases``, ``state`` [reports, nodes]); asserted (1) and
uncertain (2) -> 1, denied (0) and unmentioned (3) -> 0, unknown (-1) -> no loss. Loss = sum of BCE over the known classes
divided by the number of head classes.

Heads (heads JSON): ``own`` findings (>= 200 train positives) and ``attached`` rare findings (20-199) with a ``parent``.
A rare head's per-finding rows start as a copy of its parent's and are pulled back every step by a decoupled L2 on the
difference (``attach_decay`` x lr_head x lr fraction).

Optimisation: one study bag per rank per step, exact streamed gradient (``fora.mil.model.backward``), manual gradient
all-reduce, global grad-norm clip 1.0, AdamW (head, route penalty, block-24 matrices, block-24 vectors), linear warm-up
then cosine, EMA of all trainable parameters. Raw and EMA weights are evaluated on the validation reports of every rank
after each epoch (macro AUROC over own heads).

Initialisation (``training``): ``init_block_from`` loads block 24 only (``{block, step, epoch}`` written by
sentence-alignment ``fora_align.export_block24``, key ``block``);
``init_from`` warm-starts block 24 and the head from a classifier checkpoint (per-finding rows matched by name) and
optionally resets the route penalty of listed findings (``route_penalty_init``).

Resume: ``<output>/latest.pt`` (written every 200 steps) is resumed when present; its ``training`` dict must equal the
config's. ``stop_after_step`` (top-level config key) ends the run at that global step: inside an epoch latest.pt is
written and the run exits; at an epoch end the epoch evaluation and snapshot run first.

  torchrun --nproc-per-node 4 ... train_ct.py --config configs/ct_stage3.json --output runs/ct_stage3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from common import (allreduce_gradients, bucketed_order, init_distributed, load_config, lr_fraction, save_npz,
                    save_torch, swap_ema, write_json)
from fora.encoder.blocks import build_block24
from fora.mil.bag import Loader, RoutedBag, load_study, make_factory
from fora.mil.model import (CLASS_ROWS, HEAD_DIM, auroc, backward, build_head, class_bits, forward, masked_bce,
                            softplus_inverse)


def build_model(cfg, state, device, n_classes, parent_of):
    """Block 24 + head + AdamW groups (+ EMA copies) with rare heads initialised as their parents."""
    spec = cfg["model"]
    block = build_block24(state).to(device).train()
    head = build_head(n_classes, spec.get("soft_init", 5.), HEAD_DIM).to(device).train()
    child = torch.tensor(list(parent_of), dtype=torch.long, device=device)
    par = torch.tensor(list(parent_of.values()), dtype=torch.long, device=device)
    with torch.no_grad():
        for n, p in head.named_parameters():
            if n in CLASS_ROWS and len(child):
                p[child] = p[par]
    rest = [p for n, p in head.named_parameters() if n != "route_bias_raw"]
    lr_route = spec.get("lr_route", 1e-2)
    groups = [dict(params=rest, lr=spec["lr_head"], weight_decay=spec["wd_head"], base=spec["lr_head"]),
              dict(params=[head.route_bias_raw], lr=lr_route, weight_decay=0., base=lr_route)]
    decay = [p for p in block.parameters() if p.ndim > 1]
    other = [p for p in block.parameters() if p.ndim <= 1]
    groups += [dict(params=decay, lr=spec["lr_block"], weight_decay=spec["wd_block"], base=spec["lr_block"]),
               dict(params=other, lr=spec["lr_block"], weight_decay=0., base=spec["lr_block"])]
    params = [p for g in groups for p in g["params"]]
    return dict(spec=spec, block=block, head=head, params=params, optimizer=torch.optim.AdamW(groups),
                ema=[p.detach().clone() for p in params] if spec.get("ema") else None, child=child, par=par)


@torch.no_grad()
def attach_decay(model, rate):
    """Decoupled L2 on (rare row - parent row): rare heads = parent + a shrunk delta."""
    if not len(model["child"]) or rate <= 0:
        return
    for n, p in model["head"].named_parameters():
        if n in CLASS_ROWS:
            p[model["child"]] -= rate * (p[model["child"]] - p[model["par"]])


def warm_start(model, src, classes, cidx, parent_of, cfg, rank):
    """init_from: block 24 + shared head layers; per-finding rows copied by finding name; new rare heads = parent."""
    model["block"].load_state_dict(src["block"])
    old = {n: i for i, n in enumerate(src["classes"])}
    hs = model["head"].state_dict()
    with torch.no_grad():
        for k, v in src["head"].items():
            if k not in hs:
                continue
            if k in CLASS_ROWS:
                for n, i in old.items():
                    if n in cidx:
                        hs[k][cidx[n]] = v[i]
            elif hs[k].shape == v.shape:
                hs[k].copy_(v)
    model["head"].load_state_dict(hs)
    with torch.no_grad():
        fresh = [(c, p) for c, p in parent_of.items() if classes[c] not in old]
        if fresh:
            ch = torch.tensor([c for c, _ in fresh], device=model["head"].route_bias_raw.device)
            pa = torch.tensor([p for _, p in fresh], device=ch.device)
            for k, prm in model["head"].named_parameters():
                if k in CLASS_ROWS:
                    prm[ch] = prm[pa]
        for n, v in cfg.get("route_penalty_init", {}).items():
            if n in cidx:
                model["head"].route_bias_raw[cidx[n]] = softplus_inverse(v)
    if rank == 0:
        print(json.dumps(dict(event="warm_start", classes_copied=sum(n in cidx for n in old),
                              route_penalty_reset=sum(n in cidx for n in cfg.get("route_penalty_init", {})))), flush=True)


def read_cases(path, need):
    cases = {}
    with open(path) as f:
        for line in f:
            i = line.find('"id": "')
            cid = line[i + 7:line.index('"', i + 7)]
            if cid in need:
                cases[cid] = json.loads(line)
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--eval-limit", type=int, default=0, help="evaluate only this many validation reports per rank")
    a = ap.parse_args()
    m = load_config(a.config)
    cfg = m["training"]
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    rank, world, device, barrier = init_distributed()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(cfg["seed"])
    src = torch.load(m["normalization"], map_location=device, weights_only=False)
    mean, rms = src["mean"].to(device).float(), float(src["rms"])
    state = torch.load(m["block24_state"], map_location="cpu", weights_only=False)

    H = json.loads(Path(m["heads"]).read_text())
    own = [h["name"] for h in H["own"]]
    rare = H["attached"]
    classes = own + [h["name"] for h in rare]
    C = len(classes)
    cidx = {n: i for i, n in enumerate(classes)}
    parent_of = {cidx[h["name"]]: cidx[h["parent"]] for h in rare}
    cb = class_bits(json.loads(Path(m["class_region_bits"]).read_text()), classes, device)
    L = dict(np.load(m["labels"]))
    nodes = list(L["nodes"])
    col = np.array([nodes.index(n) for n in classes])
    row = {c: i for i, c in enumerate(L["cases"])}

    A = json.loads(Path(m["assign"]).read_text())
    if A["ranks"] != world:
        raise SystemExit(f"assignment is for {A['ranks']} ranks, world size is {world}")
    mine = list(A["assignments"]["train"][rank]) + list(A["assignments"]["val"][rank])
    sp = json.loads(Path(m["split"]).read_text())
    owned = {s: [c for c in mine if sp.get(c) == s] for s in ("train", "val")}
    cases = read_cases(m["cases"], set(mine))

    def target(cid):
        if cid not in row:
            return None
        s = L["state"][row[cid]][col].astype(np.float32)
        return np.where(s < 0, np.nan, ((s == 1) | (s == 2)).astype(np.float32))
    tgt = {cid: target(cid) for cid in cases}
    usable = lambda c: tgt.get(c) is not None and np.isfinite(tgt[c]).any()
    train = [c for c in owned["train"] if usable(c)]
    val = [c for c in owned["val"] if usable(c)]
    if a.eval_limit:
        val = val[:a.eval_limit]
    counts = [None] * world
    dist.all_gather_object(counts, len(train))
    steps = max(counts)

    loss_fn = masked_bce(C)
    model = build_model(cfg, state, device, C, parent_of)
    del state
    pf = cfg.get("position_freqs", 4)
    loader = Loader(m["cache_root"], m["region_root"], pf, cfg.get("prefetch_depth", 4), cfg.get("loader_workers", 3))
    cmin, crops, gib = cfg.get("min_route_tokens", 8), cfg["crops_per_chunk"], cfg.get("gpu_bag_gib", 20.)
    stop_after = int(m.get("stop_after_step") or 0)

    statepath = out / "latest.pt"
    epoch0 = batch0 = step = 0
    if statepath.exists():
        s = torch.load(statepath, map_location=device, weights_only=False)
        if not (s["training"] == cfg and s["world"] == world and s["classes"] == classes):
            raise SystemExit("latest.pt is from a different recipe, world size or class list")
        model["head"].load_state_dict(s["head"])
        model["block"].load_state_dict(s["block"])
        model["optimizer"].load_state_dict(s["optimizer"])
        if model["ema"] is not None:
            for e, v in zip(model["ema"], s["ema"]):
                e.copy_(v)
        epoch0, batch0, step = s["epoch"], s["batch"], s["step"]
        del s
    elif cfg.get("init_from") or cfg.get("init_block_from"):
        src = torch.load(cfg.get("init_from") or cfg["init_block_from"], map_location=device, weights_only=False)
        if cfg.get("init_from"):
            warm_start(model, src, classes, cidx, parent_of, cfg, rank)
        else:
            model["block"].load_state_dict(src["block"])
        if model["ema"] is not None:
            for e, p in zip(model["ema"], model["params"]):
                e.copy_(p.detach())
        del src

    def save(epoch, batch):
        if rank == 0:
            save_torch(statepath, dict(training=cfg, world=world, classes=classes, epoch=epoch, batch=batch, step=step,
                                       head=model["head"].state_dict(), block=model["block"].state_dict(),
                                       optimizer=model["optimizer"].state_dict(), ema=model["ema"]))
        barrier()

    def bag_of(views):
        return RoutedBag(views, device, cb, cmin, gib)

    def evaluate(epoch):
        """All views of every validation report; raw + EMA logits; macro AUROC over own heads (>= 10 pos and neg)."""
        preds = {}
        model["head"].eval()
        model["block"].eval()
        for cid in val:
            bag = bag_of(load_study(cases[cid], m["cache_root"], m["region_root"], pf))
            f = make_factory(bag, model["block"], model["spec"], mean, rms, crops)
            preds.setdefault("raw", {})[cid] = forward(model["head"], f)[0].float().cpu().numpy()
            if model["ema"] is not None:
                swap_ema(model["params"], model["ema"])
                preds.setdefault("ema", {})[cid] = forward(model["head"], f)[0].float().cpu().numpy()
                swap_ema(model["params"], model["ema"])
            del bag
        model["head"].train()
        model["block"].train()
        gathered = [None] * world
        dist.all_gather_object(gathered, (preds, {c: tgt[c] for c in val}))
        if rank == 0:
            T = {c: t for _, tp in gathered for c, t in tp.items()}
            summary, arrays = {}, {}
            for var in sorted({k for p, _ in gathered for k in p}):
                merged = {c: v for p, _ in gathered for c, v in p.get(var, {}).items()}
                ids = sorted(merged)
                X = np.stack([merged[c] for c in ids]).astype(np.float64)
                Y = np.stack([T[c] for c in ids])
                per, macro = {}, []
                for k, name in enumerate(classes):
                    y = Y[:, k]
                    ok = np.isfinite(y)
                    yp, yn = int((y[ok] == 1).sum()), int((y[ok] == 0).sum())
                    e = dict(val_pos=yp, val_neg=yn, tier="own" if k < len(own) else "rare")
                    if yp >= 10 and yn >= 10:
                        e["auroc"] = auroc(y[ok], X[ok, k])
                        if k < len(own):
                            macro.append(e["auroc"])
                    per[name] = e
                rk = list(range(len(own), C))
                pooled = None
                if rk:
                    yy, xx = Y[:, rk].ravel(), X[:, rk].ravel()
                    ok = np.isfinite(yy)
                    if ok.any() and 0 < yy[ok].sum() < ok.sum():
                        pooled = auroc(yy[ok], xx[ok])
                summary[var] = dict(n_reports=len(ids), macro_auroc_own=float(np.mean(macro)) if macro else None,
                                    n_own_scored=len(macro), rare_pooled_auroc=pooled, per_class=per)
                arrays[f"{var}__logits"] = X.astype(np.float32)
                arrays[f"{var}__ids"] = np.array(ids)
                arrays[f"{var}__targets"] = Y
            save_npz(out / f"predictions_epoch_{epoch:02d}.npz", **arrays)
            save_torch(out / f"model_epoch_{epoch:02d}.pt", dict(head=model["head"].state_dict(), block=model["block"].state_dict(),
                                                                 ema=model["ema"], classes=classes))
            res = out / "results.json"
            hist = json.loads(res.read_text())["by_epoch"] if res.exists() else []
            write_json(res, dict(by_epoch=[h for h in hist if h["epoch"] != epoch] + [dict(epoch=epoch, **summary)]))
            print(json.dumps(dict(event="epoch_complete", epoch=epoch, **{v: dict(macro_auroc_own=d["macro_auroc_own"],
                  rare_pooled_auroc=d["rare_pooled_auroc"]) for v, d in summary.items()})), flush=True)
        barrier()

    total, warmup = steps * cfg["epochs"], int(steps * cfg["warmup_epochs"])
    if rank == 0:
        write_json(out / "setup.json", dict(classes=C, own=len(own), rare=len(rare), train_reports=sum(counts),
                                            steps_per_epoch=steps, epochs=cfg["epochs"], stop_after_step=stop_after))
    size = lambda c: sum(v["n_tokens"] for v in cases[c]["views"])
    start = time.time()
    for epoch in range(epoch0, cfg["epochs"]):
        order = bucketed_order(train, size, cfg["seed"], epoch, rank, cfg["bucket_size"], min(counts))
        first = batch0 if epoch == epoch0 else 0
        D = loader.depth
        for b in range(first, min(first + D, len(order))):
            loader.submit(cases[order[b]])
        for batch in range(first, steps):
            active = batch < len(order)
            count = sum(batch < n for n in counts)
            frac = lr_fraction(step, warmup, total)
            model["optimizer"].zero_grad(set_to_none=True)
            for g in model["optimizer"].param_groups:
                g["lr"] = g["base"] * frac
            row_loss = 0.
            if active:
                bag = bag_of(loader.get())
                if batch + D < len(order):
                    loader.submit(cases[order[batch + D]])
                t = torch.from_numpy(tgt[order[batch]]).to(device)
                f = make_factory(bag, model["block"], model["spec"], mean, rms, crops,
                                 seed=(cfg["seed"] * 1000003 + step) * 64 + rank * 7919)
                loss, _ = backward(model["head"], f, t, loss_scale=world / count, loss_fn=loss_fn)
                row_loss = float(loss) * count / world
                del bag
            allreduce_gradients(model["params"], world)
            norm = float(torch.nn.utils.clip_grad_norm_(model["params"], 1., error_if_nonfinite=True))
            model["optimizer"].step()
            attach_decay(model, cfg.get("attach_decay", 0.5) * model["spec"]["lr_head"] * frac)
            if model["ema"] is not None:
                with torch.no_grad():
                    torch._foreach_lerp_(model["ema"], [p.detach() for p in model["params"]], 1 - model["spec"]["ema"])
            step += 1
            meter = torch.tensor([row_loss, float(active)], device=device, dtype=torch.float64)
            dist.all_reduce(meter)
            if rank == 0 and (step % 10 == 0 or step <= 20):
                r = dict(epoch=epoch + 1, batch=batch + 1, batches=steps, step=step, lr_fraction=frac,
                         loss=float(meter[0] / max(1., float(meter[1]))), gradient_norm=norm, elapsed=time.time() - start)
                write_json(out / "progress.json", r)
                print(json.dumps(r), flush=True)
            stop = bool(stop_after and step >= stop_after)
            if step % 200 == 0 or (stop and batch + 1 < steps):
                save(epoch, batch + 1)
            if stop and batch + 1 < steps:
                loader.close()
                dist.destroy_process_group()
                return
        evaluate(epoch + 1)
        save(epoch + 1, 0)
        if stop_after and step >= stop_after:
            break
    if rank == 0 and step >= total:
        write_json(out / "COMPLETE.json", dict(epochs=cfg["epochs"], step=step, finished=time.time()))
    loader.close()
    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
