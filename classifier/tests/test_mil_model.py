"""Streamed MIL forward / backward against direct autograd, routing penalty and bag routing on toy shapes."""
import numpy as np
import torch

from fora.mil.model import backward, build_head, class_bits, forward, masked_bce


def _toy(n=300, C=5, dim=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    head = build_head(C, dim=dim).double()
    head.token_chunk_size = 64
    with torch.no_grad():
        head.route_bias_raw.copy_(torch.randn(C, generator=g, dtype=torch.float64))
        head.output_scale.copy_(1 + 0.3 * torch.randn(C, generator=g, dtype=torch.float64))
    x = torch.randn(n, dim, generator=g, dtype=torch.float64)
    mask = torch.rand(n, C, generator=g) > 0.4
    return head, x, mask


def _factory(x, mask, chunk=70):
    def f():
        for i in range(0, len(x), chunk):
            yield x[i:i + chunk], mask[i:i + chunk]
    return f


def _direct(head, x, mask):
    classes = torch.arange(head.n_classes)
    s, l = head._chunk_outputs(x, classes)
    s = s + (~mask).double() * (-torch.nn.functional.softplus(head.route_bias_raw))
    return ((torch.softmax(s, 0) * l).sum(0) * head.output_scale + head.output_bias)[None]


def test_forward_matches_direct_and_routed_head():
    head, x, mask = _toy()
    ref = _direct(head, x, mask)
    assert torch.allclose(forward(head, _factory(x, mask)).double(), ref, atol=1e-6)
    assert torch.allclose(head(x, torch.arange(head.n_classes), mask).double(), ref, atol=1e-6)


def test_streamed_backward_is_exact():
    head, x, mask = _toy()
    target = torch.tensor([1., 0., float("nan"), 1., 0.], dtype=torch.float64)
    loss_fn = masked_bce(head.n_classes)
    head.zero_grad()
    loss, _ = backward(head, _factory(x, mask), target, loss_fn=loss_fn)
    streamed = {n: p.grad.clone() for n, p in head.named_parameters()}
    head.zero_grad()
    ref = loss_fn(_direct(head, x, mask), target.reshape(1, -1))
    ref.backward()
    assert torch.allclose(loss.double(), ref.detach(), atol=1e-9)
    for n, p in head.named_parameters():
        assert torch.allclose(streamed[n], p.grad, atol=1e-7), n


def test_class_bits_default_all():
    t = class_bits({"a": 3}, ["a", "b"], "cpu")
    assert int(t[0]) == 3 and int(t[1]) == -1


def test_routed_bag_masks():
    from fora.mil.bag import RoutedBag
    from fora.regions import REGION_NAMES
    n = 20
    starts = np.array([0, 12, n])
    bits = np.zeros(n, np.int64)
    bits[:10] = 1 << REGION_NAMES.index("lungs")
    bits[10:12] = 1 << REGION_NAMES.index("liver")
    view = dict(prefix=np.zeros((2, 5, 1024), np.uint16), q=np.zeros((n, 1024), np.int8), scale=np.ones((2, 1024), np.float16),
                oidx=np.arange(8, dtype=np.int16), outlier=np.zeros((n, 8), np.uint16), starts=starts,
                local=np.arange(n), pos=np.zeros((n, 24), np.float16), bits=bits)
    cb = torch.tensor([1 << REGION_NAMES.index("lungs"), 1 << REGION_NAMES.index("liver")], dtype=torch.int64)
    bag = RoutedBag([view], "cpu", cb, min_route=8)
    m = bag.masks[0]
    assert m[:, 0].sum() == 10 and bool(m[:, 1].all())     # liver: 2 < 8 tokens -> whole view
