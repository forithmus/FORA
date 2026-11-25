"""Chest X-ray writer data: report targets, study lists, dense MIL logits, epoch schedule and a threaded prefetcher.

Paper section "Grounded report generation" (X-ray arm). The target is the Findings section of the study's report
(structured findings, else extracted sentences, else there-is statements, else 'No findings were reported.'),
as for the CT writer.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode('utf-8', 'replace')
    if isinstance(value, np.ndarray):
        value = value.tolist()
        if isinstance(value, bytes):
            return value.decode('utf-8', 'replace')
    return str(value)


def _json_list(value) -> list[str]:
    text = _decode(value).strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(data, str):
        return [data]
    return [str(x).strip() for x in data if str(x).strip()]


def findings_from(report) -> str:
    """Findings target from an open H5 report group."""
    findings = ''
    structured = report.get('structured')
    if structured is not None and 'findings' in structured:
        findings = _decode(structured['findings'][()]).strip()
    if not findings:
        for key in ('extracted_sentences', 'there_is_statements'):
            if key in report:
                sentences = _json_list(report[key][()])
                if sentences:
                    findings = ' '.join(sentences)
                    break
    return findings or 'No findings were reported.'


def findings_text(case: dict, h5_root: str) -> str:
    """Findings target of one study case (H5 layout of the X-ray data stage: <h5>.h5 / patient / report / report)."""
    with h5py.File(f"{h5_root}/{case['h5']}.h5", 'r') as handle:
        return findings_from(handle[case['patient']][case['report']]['report'])


def read_cases(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def case_tokens(case: dict, max_films: int = 64) -> int:
    """Number of 16-px patch tokens of a study's films (film = [.., .., kind, height, width, ...])."""
    return int(sum(-(-int(f[3]) // 16) * -(-int(f[4]) // 16) for f in case['films'][:max_films]))


def epoch_schedule(n_items: int, sizes: np.ndarray, world: int, epoch: int, seed: int, bucket: int = 8) -> np.ndarray:
    """[steps, world] item indices for one epoch.

    A seeded permutation is cut into blocks of world*bucket items; inside a block the items are sorted by size and dealt
    so that one step holds `world` items of similar size (synchronous FSDP steps wait for the slowest rank); block steps
    are shuffled. The tail is padded by wrapping to the epoch start."""
    rng = np.random.default_rng([int(seed), int(epoch), 7919])
    perm = rng.permutation(n_items)
    steps = math.ceil(n_items / world)
    total = steps * world
    perm = np.concatenate([perm, perm[: total - n_items]])
    out = []
    span = world * bucket
    for start in range(0, total, span):
        block = perm[start:start + span]
        block = block[np.argsort(sizes[block], kind='stable')]
        rows = block.reshape(-1, world)
        rows = rows[rng.permutation(len(rows))]
        for row in rows:
            out.append(row[rng.permutation(world)])
    return np.stack(out)


class DenseLogits:
    """Study-bag MIL logits (dense npz written by ``fora_report.cxr.calibration``), checked against the calibration."""

    def __init__(self, path: str, labels: list[str], calibration: dict):
        if Path(str(calibration.get('source_predictions', ''))).resolve() != Path(path).resolve():
            raise ValueError('dense logits are not the source predictions of the calibration')
        z = np.load(path, allow_pickle=False)
        if [str(c) for c in z['label'].tolist()] != list(labels):
            raise ValueError('dense logits vocabulary differs from the calibration labels')
        self.logits = np.asarray(z['logit'], np.float32)
        self.row = {str(u): k for k, u in enumerate(z['report_uid'].tolist())}

    def get(self, uid: str, device) -> torch.Tensor:
        return torch.tensor(self.logits[self.row[str(uid)]], device=device).unsqueeze(0)


class Prefetcher:
    """Thread prefetch of (case, prepared film bag, findings text) for this rank's schedule (h5py/numpy release the GIL)."""

    def __init__(self, stream, cases: list[dict], order, h5_root: str, depth: int = 4, workers: int = 3):
        from concurrent.futures import ThreadPoolExecutor
        self.stream, self.cases, self.order, self.h5_root = stream, cases, list(order), h5_root
        self.pool = ThreadPoolExecutor(workers)
        self.depth = depth
        self.pending = {}
        self.next_submit = 0
        for _ in range(depth):
            self._submit()

    def _load(self, index):
        """A case without any loadable film is replaced by the next case of the list (logged), so every rank still
        runs one forward per step (FSDP collectives); its own report text goes with the replacement images."""
        for attempt in range(8):
            case = self.cases[(int(index) + attempt) % len(self.cases)]
            try:
                return case, self.stream.prepare(case, h5_root=self.h5_root), findings_text(case, self.h5_root)
            except Exception as error:
                print(json.dumps(dict(event='case_skip', case=case['id'], error=repr(error)[:200])), flush=True)
        raise RuntimeError(f'8 consecutive unloadable cases from item {index}')

    def _submit(self):
        if self.next_submit < len(self.order):
            k = self.next_submit
            self.pending[k] = self.pool.submit(self._load, self.order[k])
            self.next_submit += 1

    def get(self, k):
        future = self.pending.pop(k)
        self._submit()
        try:
            return future.result(timeout=900)
        except Exception as error:   # one synchronous retry (file-system hiccup)
            print(json.dumps(dict(event='loader_retry', item=int(self.order[k]), error=repr(error)[:200])), flush=True)
            return self._load(self.order[k])

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
