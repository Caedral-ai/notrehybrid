from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from notre.layers.notre_linear import (
    CollisionCache,
    apply_collision_cache,
    collision_error,
    gated_delta_scan,
    ring_readout,
)


def test_error_is_normalized_delta() -> None:
    delta = torch.tensor([[3.0, 4.0]])
    value = torch.tensor([[0.0, 2.0]])
    err = collision_error(delta, value, eps=1e-6)
    assert torch.allclose(err, torch.tensor([5.0 / 2.0]), atol=1e-5)


def test_below_tau_reads_nothing() -> None:
    cache = CollisionCache(slots=4, tau=0.5)
    q = torch.randn(1, 3, 1, 4)
    k = torch.randn(1, 3, 1, 4)
    v = torch.randn(1, 3, 1, 4)
    err = torch.full((1, 3, 1), 0.1)
    out = cache(q, k, v, err)
    assert torch.count_nonzero(out) == 0


def test_promoted_key_is_readable_next() -> None:
    cache = CollisionCache(slots=4, tau=0.5)
    q = torch.zeros(1, 2, 1, 4)
    k = torch.zeros(1, 2, 1, 4)
    v = torch.zeros(1, 2, 1, 4)
    q[0, 1, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    k[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    v[0, 0, 0] = torch.tensor([0.0, 5.0, 0.0, 0.0])
    err = torch.tensor([[[0.9], [0.0]]])
    out = cache(q, k, v, err)
    assert torch.allclose(out[0, 1, 0], v[0, 0, 0], atol=1e-5)


def test_future_write_does_not_change_the_past() -> None:
    cache = CollisionCache(slots=4, tau=0.5)
    q = torch.randn(1, 3, 2, 4)
    k = torch.randn(1, 3, 2, 4)
    v = torch.randn(1, 3, 2, 4)
    err = torch.zeros(1, 3, 2)
    err[0, 0] = 0.9
    first = cache(q, k, v, err)
    err2 = err.clone()
    err2[0, 2] = 5.0
    v2 = v.clone()
    v2[0, 2] = 9.0
    second = cache(q, k, v2, err2)
    assert torch.allclose(first[:, :2], second[:, :2])


def test_ring_overwrites_the_oldest_slot() -> None:
    cache = CollisionCache(slots=2, tau=0.5)
    q = torch.zeros(1, 4, 1, 2)
    k = torch.zeros(1, 4, 1, 2)
    v = torch.zeros(1, 4, 1, 2)
    eye = torch.eye(2)
    for t in range(3):
        k[0, t, 0] = eye[t % 2]
        v[0, t, 0, 0] = float(t + 1)
    q[0, 3, 0] = eye[0]
    err = torch.tensor([[[0.9], [0.9], [0.9], [0.0]]])
    out = cache(q, k, v, err)
    # slots: t0 writes 0, t1 writes 1, t2 overwrites 0. Query matches k[0], which is gone.
    assert not torch.allclose(out[0, 3, 0, 0], torch.tensor(1.0), atol=1e-4)


def test_query_groups_share_one_kv_ring() -> None:
    cache = CollisionCache(slots=2, tau=0.5)
    q = torch.zeros(1, 2, 2, 2)
    k = torch.zeros(1, 2, 1, 2)
    v = torch.zeros(1, 2, 1, 2)
    q[0, 1, :, 0] = 1.0
    k[0, 0, 0, 0] = 1.0
    v[0, 0, 0, 1] = 3.0
    err = torch.tensor([[[0.8], [0.0]]])
    out = cache(q, k, v, err)
    assert out.shape == (1, 2, 2, 2)
    assert torch.allclose(out[0, 1, 0], out[0, 1, 1])


def test_disabled_cache_returns_the_linear_output() -> None:
    cache = CollisionCache(slots=4, tau=0.5)
    q = torch.randn(1, 2, 1, 4)
    k = torch.randn(1, 2, 1, 4)
    v = torch.randn(1, 2, 1, 4)
    err = torch.full((1, 2, 1), 0.9)
    o_lin = torch.randn(1, 2, 1, 4)
    out = cache(q, k, v, err, o_lin=o_lin, enabled=False)
    assert torch.equal(out, o_lin)


def test_threshold_is_detached_and_values_get_grad() -> None:
    cache = CollisionCache(slots=4, tau=0.5)
    q = torch.zeros(1, 2, 1, 2)
    k = torch.zeros(1, 2, 1, 2)
    v = torch.zeros(1, 2, 1, 2, requires_grad=True)
    q[0, 1, 0, 0] = 1.0
    k[0, 0, 0, 0] = 1.0
    err = torch.tensor([[[0.2], [0.0]]], requires_grad=True)
    out = cache(q, k, v, err)
    out.sum().backward()
    assert err.grad is None or torch.count_nonzero(err.grad) == 0
    assert v.grad is not None
    assert torch.count_nonzero(v.grad) == 0

    v2 = torch.zeros(1, 2, 1, 2, requires_grad=True)
    err2 = torch.tensor([[[0.9], [0.0]]])
    out2 = cache(q, k, v2, err2)
    out2.sum().backward()
    assert v2.grad is not None
    assert v2.grad[0, 0].abs().sum() > 0


def test_scan_error_is_high_when_the_state_misses() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 4, 1, 4)
    k = torch.randn(1, 4, 1, 4)
    v = torch.randn(1, 4, 1, 4)
    beta = torch.ones(1, 4, 1)
    alpha = torch.ones(1, 4, 1)
    o_lin, err, q_hat, k_hat = gated_delta_scan(q, k, v, beta, alpha)
    assert o_lin.shape == q.shape
    assert err.shape == (1, 4, 1)
    assert torch.isfinite(err).all()
    assert q_hat.shape == q.shape and k_hat.shape == k.shape


class _TinyGDN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.use_short_conv = False
        self.head_dim = 4
        self.q_proj = torch.nn.Linear(8, 4, bias=False)
        self.k_proj = torch.nn.Linear(8, 4, bias=False)
        self.v_proj = torch.nn.Linear(8, 4, bias=False)
        self.a_proj = torch.nn.Linear(8, 1, bias=False)
        self.b_proj = torch.nn.Linear(8, 1, bias=False)
        self.A_log = torch.nn.Parameter(torch.zeros(1))
        self.dt_bias = torch.nn.Parameter(torch.zeros(1))
        self.o_proj = torch.nn.Linear(4, 8, bias=False)


def test_ring_readout_matches_sequential_cache() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 6, 2, 4)
    k = torch.randn(2, 6, 2, 4)
    v = torch.randn(2, 6, 2, 4)
    err = torch.rand(2, 6, 2)
    ref = CollisionCache(slots=3, tau=0.4)(q, k, v, err)
    out = ring_readout(q, k, v, err, slots=3, tau=0.4)
    assert torch.allclose(out, ref, atol=1e-5)


def test_disabled_cache_leaves_student_output() -> None:
    module = _TinyGDN()
    hidden = torch.randn(1, 3, 8)
    student = torch.randn(1, 3, 8)
    cache = CollisionCache(slots=4, tau=0.0)
    out = apply_collision_cache(module, hidden, student, cache, enabled=False)
    assert torch.equal(out, student)


def test_enabled_cache_changes_student_output() -> None:
    torch.manual_seed(0)
    module = _TinyGDN()
    with torch.no_grad():
        module.o_proj.weight.copy_(torch.eye(8, 4))
        module.v_proj.weight.copy_(torch.eye(4, 8))
    hidden = torch.randn(1, 4, 8)
    student = torch.zeros(1, 4, 8)
    cache = CollisionCache(slots=4, tau=0.0)
    out = apply_collision_cache(module, hidden, student, cache, enabled=True)
    assert out.shape == student.shape
    assert not torch.allclose(out, student)
