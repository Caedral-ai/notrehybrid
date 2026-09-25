"""One-kernel fp32 delta scan. Returns ``err_t`` only; the threshold is detached later."""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _err_kernel(
        K,
        V,
        BETA,
        ALPHA,
        ERR,
        STATE,
        T,
        tau_unused,
        eps,
        D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, D)
        state = STATE + pid * D * D
        for d in range(D):
            tl.store(state + d * D + offs, tl.zeros([D], dtype=tl.float32))
        k_base = K + pid * T * D
        v_base = V + pid * T * D
        for t in range(T):
            alpha = tl.load(ALPHA + pid * T + t)
            beta = tl.load(BETA + pid * T + t)
            for d in range(D):
                row = tl.load(state + d * D + offs) * alpha
                tl.store(state + d * D + offs, row)
            pred = tl.zeros([D], dtype=tl.float32)
            for e in range(D):
                ke = tl.load(k_base + t * D + e)
                pred += tl.load(state + offs * D + e) * ke
            vv = tl.load(v_base + t * D + offs)
            delta = beta * (vv - pred)
            for d in range(D):
                kd = tl.load(k_base + t * D + d)
                row = tl.load(state + d * D + offs)
                tl.store(state + d * D + offs, row + kd * delta)
            err = tl.sqrt(tl.sum(delta * delta, axis=0)) / (
                tl.sqrt(tl.sum(vv * vv, axis=0)) + eps
            )
            tl.store(ERR + pid * T + t, err)


def triton_delta_error(
    k_hat: Tensor,
    v: Tensor,
    beta: Tensor,
    alpha: Tensor,
    eps: float,
) -> Tensor:
    """``err`` shaped ``[B, T, H]``. ``k_hat`` and ``v`` are ``[B, T, H, D]``."""
    if triton is None or not k_hat.is_cuda:
        raise RuntimeError("triton delta scan needs CUDA")
    batch, steps, heads, dim = k_hat.shape
    flat = batch * heads
    k = k_hat.permute(0, 2, 1, 3).contiguous().view(flat, steps, dim)
    vv = v.permute(0, 2, 1, 3).contiguous().view(flat, steps, dim)
    beta_f = beta.permute(0, 2, 1).contiguous().view(flat, steps)
    alpha_f = alpha.permute(0, 2, 1).contiguous().view(flat, steps)
    err = torch.empty(flat, steps, device=k.device, dtype=torch.float32)
    state = torch.empty(flat, dim, dim, device=k.device, dtype=torch.float32)
    _err_kernel[(flat,)](k, vv, beta_f, alpha_f, err, state, steps, 0.0, eps, D=dim)
    return err.view(batch, heads, steps).permute(0, 2, 1).contiguous()
