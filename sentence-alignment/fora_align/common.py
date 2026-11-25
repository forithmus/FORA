"""Parts shared by the CT and X-ray alignment trainers (Methods, "Sentence alignment").

Model: trainable block 24 + final norm -> (y - mean) / rms ++ Fourier position features (24) -> Linear(1048, 512) + a
zero-initialized view-kind embedding. Text: CXR-BERT CLS -> Linear(768, 512) -> l2norm. Learned logit temperature
(init 1 / 0.07) and pooling temperature (init 1 / 0.07), both exp of a parameter clamped to [0, 4].

One step = one report per GPU, at most 34 sentences per report (sampled, padded with masked empty rows). The global loss is
computed identically on every rank; gradients are summed by the gather backward and averaged manually over ranks, then
clipped to norm 1. AdamW groups: block 24 (weights with decay, 1-D without), projection, BERT, text projection,
temperatures (no decay). Learning rate: linear warm-up then cosine to 0 over epochs x steps per epoch.
"""
from __future__ import annotations

import json
import math
import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .loss import GatherLayer, StreamPool, contrastive_ce


class VisualProjector(torch.nn.Module):
    def __init__(self, dim_in: int, dim: int, n_kinds: int):
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim)
        self.view = torch.nn.Embedding(n_kinds, dim)
        torch.nn.init.zeros_(self.view.weight)

    def forward(self, x, kind):
        return self.proj(x) + self.view.weight[kind]


class TextEncoder(torch.nn.Module):
    """CXR-BERT (``microsoft/BiomedVLP-CXR-BERT-specialized``) CLS token -> Linear -> l2norm."""

    def __init__(self, path: str, dim: int):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(path)
        self.proj = torch.nn.Linear(self.bert.config.hidden_size, dim)

    def forward(self, ids, att):
        return F.normalize(self.proj(self.bert(ids, attention_mask=att).last_hidden_state[:, 0]).float(), dim=-1)


def init_distributed():
    """NCCL on CUDA; gloo on CPU (CPU runs are for tests)."""
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=30))
        barrier = lambda: dist.barrier(device_ids=[local])
    else:
        device = torch.device("cpu")
        dist.init_process_group("gloo", timeout=timedelta(minutes=30))
        barrier = dist.barrier
    return rank, world, device, barrier


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def build_model(block, spec: dict, position_freqs: int, n_kinds: int, device):
    """(vproj, text, tokenizer, temp, pool_temp, optimizer, params) for a trainable ``block`` (block 24 + norm)."""
    from transformers import BertTokenizer
    vproj = VisualProjector(1024 + 6 * position_freqs, spec["dim"], n_kinds).to(device).train()
    text = TextEncoder(spec["text_model"], spec["dim"]).to(device).train()
    tok = BertTokenizer.from_pretrained(spec["text_model"], do_lower_case=True)
    temp = torch.nn.Parameter(torch.tensor(float(np.log(1 / 0.07)), device=device))
    pool_temp = torch.nn.Parameter(torch.tensor(float(np.log(spec.get("pool_temperature_init", 1 / 0.07))), device=device))
    groups = [dict(params=[p for p in block.parameters() if p.ndim > 1], lr=spec["lr_block"], weight_decay=spec["wd"], base=spec["lr_block"]),
              dict(params=[p for p in block.parameters() if p.ndim <= 1], lr=spec["lr_block"], weight_decay=0., base=spec["lr_block"]),
              dict(params=list(vproj.parameters()), lr=spec["lr_proj"], weight_decay=spec["wd"], base=spec["lr_proj"]),
              dict(params=list(text.bert.parameters()), lr=spec["lr_text"], weight_decay=spec["wd"], base=spec["lr_text"]),
              dict(params=list(text.proj.parameters()), lr=spec["lr_proj"], weight_decay=spec["wd"], base=spec["lr_proj"]),
              dict(params=[temp, pool_temp], lr=spec["lr_temp"], weight_decay=0., base=spec["lr_temp"])]
    params = [p for g in groups for p in g["params"]]
    return vproj, text, tok, temp, pool_temp, torch.optim.AdamW(groups), params


RESUME_FREE = ("epochs", "epochs_schedule", "stop_after_epoch", "continuation")


def horizon_epochs(cfg: dict, step: int) -> int:
    """Epoch count that defines the cosine horizon at ``step``: ``epochs``, or the last ``epochs_schedule`` entry
    [start_step, epochs] with start_step <= step."""
    epochs = int(cfg["epochs"])
    for start, e in cfg.get("epochs_schedule", []):
        if step >= start:
            epochs = int(e)
    return epochs


def lr_fraction(step: int, warmup: int, total: int) -> float:
    return min(1., (step + 1) / max(1, warmup)) * (.5 + .5 * math.cos(math.pi * max(0, step - warmup) / max(1, total - warmup)))


def size_bucketed_order(train: list, size, steps: int, seed: int, epoch: int, rank: int, bucket: int) -> list:
    """Per-epoch order: reports sorted by size into bins of ``bucket``; the bin order is shared by all ranks (so a step's
    reports have similar sizes on every GPU), the order inside a bin is per rank."""
    rng = np.random.default_rng(seed + 100 + epoch + 10000 * rank)
    ordered = sorted(train, key=lambda c: (-size(c), c))
    bins = steps // bucket
    bin_order = np.random.default_rng(seed + epoch).permutation(bins)
    return ([c for b in bin_order for c in rng.permutation(ordered[b * bucket:(b + 1) * bucket]).tolist()]
            + rng.permutation(ordered[bins * bucket:]).tolist())[:steps]


class ThreadLoader:
    """Prefetch ``fn(item)`` in threads (h5py / numpy release the GIL); a failed or slow load is retried synchronously."""

    def __init__(self, fn, depth: int = 4, workers: int = 3, timeout: float = 900.):
        from concurrent.futures import ThreadPoolExecutor
        self.fn, self.depth, self.timeout = fn, depth, timeout
        self.pool = ThreadPoolExecutor(workers)
        self.pending = []

    def submit(self, item):
        self.pending.append((item, self.pool.submit(self.fn, item)))

    def get(self):
        item, fut = self.pending.pop(0)
        try:
            return fut.result(timeout=self.timeout)
        except Exception as e:
            print(json.dumps(dict(event="loader_retry", case=item.get("id"), error=repr(e)[:200])), flush=True)
            return self.fn(item)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


def alignment_step(chunks, sentences: list[str], *, model, cfg: dict, sentence_seed: int, world: int, device):
    """One optimizer step on one report per rank.

    chunks(): iterator over (x [n, 1048], kind [n]) of the report's tokens in a fixed order (called twice).
    model: dict(vproj, text, tok, temp, pool_temp, optimizer, params). Returns a metrics dict."""
    vproj, text, tok, temp, pool_temp = (model[k] for k in ("vproj", "text", "tok", "temp", "pool_temp"))
    MS = cfg["max_sentences"]
    r_ = random.Random(sentence_seed)
    sents = r_.sample(sentences, MS) if len(sentences) > MS else list(sentences)
    n = len(sents)
    sents = sents + [""] * (MS - n)
    smask = torch.tensor([1] * n + [0] * (MS - n), device=device, dtype=torch.bool)
    enc = tok(sents, return_tensors="pt", padding=True, truncation=True, max_length=cfg.get("max_tokens", 128)).to(device)
    # pass 1 (no grad): text, gathered sentences, streamed attention pooling of every token of the report
    tl = text(enc.input_ids, enc.attention_mask)
    tl = torch.where(smask[:, None], tl, torch.zeros_like(tl))
    t_all = torch.cat(GatherLayer.apply(tl), 0)
    m_all = torch.cat(GatherLayer.apply(smask.float()[:, None]), 0)[:, 0].bool()
    T_, PT = temp.clamp(0., 4.).exp(), pool_temp.clamp(0., 4.).exp()
    sp, NT = StreamPool(t_all, PT), 0
    with torch.no_grad():
        for x, k in chunks():
            zc = vproj(x, k).float()
            sp.add(zc)
            NT += len(zc)
    p_leaf = sp.pooled().requires_grad_(True)
    col = (T_ * (F.normalize(p_leaf, dim=-1) * t_all).sum(1)).clamp(-100., 100.)[:, None]   # [N_T, 1]
    logits = torch.cat(GatherLayer.apply(col), 1)                                            # [N_T, world]
    loss = contrastive_ce(logits, MS, m_all)
    # Every rank computes the same global loss and the gather backward sums all ranks' slices, so each local gradient
    # is world x its share: the average over ranks below is the exact gradient.
    loss.backward(retain_graph=True)
    g, p_ = p_leaf.grad.detach(), p_leaf.detach()
    del p_leaf
    # pass 2: recompute block 24 + projection chunk by chunk; the surrogate gives the exact attention-pooling gradients
    t_leaf = t_all.detach().requires_grad_(True)
    pt_leaf = PT.detach().requires_grad_(True)
    seen = 0
    for x, k in chunks():
        zc = vproj(x, k).float()
        sp.surrogate(t_leaf, pt_leaf, zc, p_, g).backward()
        seen += len(zc)
    if seen != NT:
        raise AssertionError((seen, NT))
    torch.autograd.backward([t_all, PT], [t_leaf.grad, pt_leaf.grad])
    params = model["params"]
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    flat = torch.cat([p.grad.reshape(-1) for p in params])
    dist.all_reduce(flat)
    flat.div_(world)
    o = 0
    for p in params:
        p.grad.copy_(flat[o:o + p.numel()].reshape_as(p))
        o += p.numel()
    norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
    model["optimizer"].step()
    with torch.no_grad():   # in-batch retrieval: a sentence's own report should be the argmax column
        tgt = torch.arange(world, device=device).repeat_interleave(MS)
        acc = ((logits.argmax(1) == tgt) & m_all).sum() / m_all.sum().clamp(min=1)
    return dict(loss=float(loss), sent_to_report_acc=float(acc), chance=1 / world, temp=float(T_), pool_temp=float(PT),
                gradient_norm=norm, tokens=NT)


def save_state(path: Path, **state) -> None:
    tmp = Path(path).with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


class Stopper:
    """SIGUSR1 requests a checkpoint and a clean exit at the next step boundary."""

    def __init__(self):
        import signal
        self.requested = False
        signal.signal(signal.SIGUSR1, self._set)

    def _set(self, *_):
        self.requested = True

    def agreed(self, extra: bool, device) -> bool:
        flag = torch.tensor(int(self.requested or extra), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())
