"""CPU tests of the CT DPO objective, pair selection and the node-local pair store."""
from __future__ import annotations

import contextlib
import json
import math
import sqlite3

import pytest
import torch

from fora_report.ct import dpo_loss as D
from fora_report.ct.dpo_pairs_local import filter_pair_store, read_filter_metadata


def test_dpo_terms_matches_hand_computation():
    pc, pr, rc, rr, beta = torch.tensor(-10.0), torch.tensor(-14.0), torch.tensor(-11.0), torch.tensor(-13.0), 0.1
    loss, margin = D.dpo_terms(pc, pr, rc, rr, beta)
    expected_margin = (-10 + 14) - (-11 + 13)
    assert float(margin) == pytest.approx(expected_margin)
    assert float(loss) == pytest.approx(math.log1p(math.exp(-beta * expected_margin)))


def _toy_setup():
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(5))
    features = {"c": torch.randn(5), "r1": torch.randn(5), "r2": torch.randn(5)}
    lengths = {"c": 4, "r1": 3, "r2": 6}
    return weight, features, lengths


def test_dpo_study_backward_equals_direct_autograd():
    weight, features, lengths = _toy_setup()
    ids = {k: torch.zeros(n, dtype=torch.long) for k, n in lengths.items()}
    key_of = {id(v): k for k, v in ids.items()}
    reference_weight = weight.detach() + 0.3

    def logp(w, key):
        return -(w * features[key]).sum().abs() - lengths[key]

    class Reference:
        @contextlib.contextmanager
        def swapped(self):
            original = weight.data.clone()
            weight.data.copy_(reference_weight)
            try:
                yield
            finally:
                weight.data.copy_(original)

    def score(_kind, target):
        value = logp(weight, key_of[id(target)])
        return value, {"loss": -value / target.numel()}

    stats = D.DPOStatistics(torch.device("cpu"))
    beta, dpo_weight, ce_weight = 0.1, 1.0, 1.0
    combined, _dpo, _out = D.dpo_study_backward(
        score=score, chosen_ids=ids["c"], rejected=[("omission", ids["r1"]), ("attribute", ids["r2"])],
        reference=Reference(), ddp_context=lambda last: contextlib.nullcontext(), beta=beta,
        dpo_weight=dpo_weight, chosen_ce_weight=ce_weight, accumulation=1, stats=stats)
    grad = weight.grad.clone()

    w = weight.detach().clone().requires_grad_(True)
    ref = {k: logp(reference_weight, k).detach() for k in ids}
    losses = [D.dpo_terms(logp(w, "c"), logp(w, k), ref["c"], ref[k], beta)[0] for k in ("r1", "r2")]
    total = dpo_weight * torch.stack(losses).mean() + ce_weight * (-logp(w, "c") / lengths["c"])
    total.backward()
    assert combined == pytest.approx(float(total), rel=1e-6)
    assert torch.allclose(grad, w.grad, atol=1e-6)
    assert int(stats.misc[0]) == 1 and int(stats.values[:, 0].sum()) == 2


def test_select_pairs_interleaves_types_and_cycles():
    rows = [(i, t, None, "sha", f"text{i}") for i, t in enumerate(["omission"] * 4 + ["repetition"] * 2)]
    first = D.select_pairs(rows, uid="u1", epoch=0, seed=42, k=2)
    assert first == D.select_pairs(rows, uid="u1", epoch=0, seed=42, k=2)
    assert {r[1] for r in first} == {"omission", "repetition"}
    seen = [r[0] for e in range(3) for r in D.select_pairs(rows, uid="u1", epoch=e, seed=42, k=2)]
    assert sorted(seen) == list(range(6))
    assert D.select_pairs([], uid="u1", epoch=0, seed=42, k=2) == []


def _pair_store(path, uids):
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE pairs(pair_id INTEGER PRIMARY KEY, report_uid TEXT, pair_type TEXT, "
                       "attribute_type TEXT, chosen_sha256 TEXT, rejected_text TEXT)")
    connection.execute("CREATE INDEX pairs_uid ON pairs(report_uid)")
    connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
    pid = 0
    for uid in uids:
        for kind in ("omission", "explicit_denial"):
            connection.execute("INSERT INTO pairs VALUES (?,?,?,?,?,?)",
                               (pid, uid, kind, None, D.sha256_text(f"report {uid}"), f"rejected {uid} {kind}"))
            pid += 1
    summary = {"schema": D.PREFERENCE_SCHEMA, "pairs_by_type": {"omission": len(uids), "explicit_denial": len(uids)}}
    connection.execute("INSERT INTO metadata VALUES ('summary', ?)", (json.dumps(summary),))
    connection.commit()
    connection.close()


def test_node_local_filter_keeps_exactly_local_pairs(tmp_path):
    source = tmp_path / "pairs.sqlite"
    _pair_store(source, ["a", "b", "c", "d"])
    out = tmp_path / "local.sqlite"
    meta = filter_pair_store(source, out, ["b", "d", "zz"], provenance={"node": 3})
    store = D.PreferencePairStore(str(out))
    everything = D.PreferencePairStore(str(source))
    assert store.pairs("b") == everything.pairs("b")
    assert store.pairs("a") == [] and store.count_uids(["a", "b", "c", "d"]) == 2
    assert meta["pairs"] == 4 and read_filter_metadata(out)["node"] == 3
    assert store.summary["pairs_by_type"] == {"explicit_denial": 2, "omission": 2}
    only = D.PreferencePairStore(str(out), ["omission"])
    assert [r[1] for r in only.pairs("d")] == ["omission"]
