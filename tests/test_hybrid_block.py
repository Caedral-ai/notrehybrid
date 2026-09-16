from __future__ import annotations

import pytest
import torch

from notre.layers.hybrid_block import DummySoftmax, unpack_gdn


def test_unpack_gdn() -> None:
    t = torch.zeros(1)
    assert unpack_gdn(t) is t
    assert unpack_gdn((t, None, None)) is t


def test_dummy_softmax_forward_backward() -> None:
    m = DummySoftmax(hidden_size=64, num_heads=4)
    x = torch.randn(2, 8, 64, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def _cuda_ok() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 7


@pytest.mark.skipif(not _cuda_ok(), reason="needs CUDA capability >= 7 (never P100)")
def test_hybrid_block_forward_backward() -> None:
    pytest.importorskip("fla.layers")
    from notre.layers.hybrid_block import HybridBlock

    device = torch.device("cuda")
    m = HybridBlock(hidden_size=128, num_heads=4, head_dim=32, mode="chunk").to(device)
    x = torch.randn(2, 64, 128, device=device, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape
    y.float().pow(2).mean().backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(g is None or torch.isfinite(g).all() for g in grads)
