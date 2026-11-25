"""The streamed two-pass VL-CABS pooling gives the same loss and gradients as plain autograd; loss sanity."""
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F


@pytest.fixture(scope="module", autouse=True)
def single_process_group():
    if not dist.is_initialized():
        f = tempfile.NamedTemporaryFile(delete=False)
        dist.init_process_group("gloo", init_method=f"file://{f.name}", rank=0, world_size=1)
    yield


def test_streamed_pooling_matches_autograd():
    from fora_align.loss import StreamPool, pooled_logits_reference
    torch.manual_seed(0)
    t = F.normalize(torch.randn(6, 8, dtype=torch.float64), dim=-1).requires_grad_(True)
    z = torch.randn(50, 8, dtype=torch.float64, requires_grad=True)
    pt, temp = torch.tensor(3.0, dtype=torch.float64, requires_grad=True), 5.0
    w = torch.randn(6, dtype=torch.float64)
    (pooled_logits_reference(t, z, pt, temp) * w).sum().backward()
    ref = (t.grad.clone(), z.grad.clone(), pt.grad.clone())
    t.grad = z.grad = pt.grad = None
    sp = StreamPool(t, pt)
    for c in z.split(7):
        sp.add(c.detach())
    p_leaf = sp.pooled().requires_grad_(True)
    (temp * (F.normalize(p_leaf, dim=-1) * t).sum(1) * w).sum().backward()
    g, p_ = p_leaf.grad, p_leaf.detach()
    t_leaf, pt_leaf = t.detach().requires_grad_(True), pt.detach().requires_grad_(True)
    zg = []
    for c in z.detach().split(7):
        c = c.requires_grad_(True)
        sp.surrogate(t_leaf, pt_leaf, c, p_, g).backward()
        zg.append(c.grad)
    assert torch.allclose(t.grad + t_leaf.grad, ref[0]) and torch.allclose(torch.cat(zg), ref[1])
    assert torch.allclose(pt_leaf.grad, ref[2])


def test_contrastive_ce_multi_positive():
    from fora_align.loss import contrastive_ce
    logits = torch.full((6, 3), -5.0)
    for i in range(6):
        logits[i, i // 2] = 5.0
    mask = torch.tensor([1, 1, 1, 0, 1, 1], dtype=torch.bool)
    good = contrastive_ce(logits, 2, mask)
    bad = contrastive_ce(-logits, 2, mask)
    assert good < 0.01 < bad
    # the masked row does not change the loss
    logits2 = logits.clone()
    logits2[3] = torch.randn(3)
    assert torch.allclose(contrastive_ce(logits2, 2, mask), good)
