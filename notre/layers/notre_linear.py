"""Trainable collision cache on top of a Gated DeltaNet mixer.

The push is a hard threshold on ``err_t``. That decision is detached.
Gradients flow through the written keys and values and the softmax read-out.
The cache has no parameters, so the no-cache twin has the same trainable set.

``q_hat`` and ``k_hat`` are the normalized vectors from the plan. Read-out
scores are ``q_hat · k / sqrt(d)``. State is one ring per KV head. Query
heads share that ring in groups of ``n_q / n_kv``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def collision_error(delta: Tensor, value: Tensor, eps: float = 1e-6) -> Tensor:
    """``err_t = ||δ_t|| / (||v_t|| + ε)`` over the head dimension."""
    return delta.norm(dim=-1) / (value.norm(dim=-1) + eps)


def gated_delta_scan(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    alpha: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Reference fp32 delta rule. Returns ``o_lin, err, q_hat, k_hat``.

    Shapes are ``[B, T, H, D]`` for ``q, k, v`` and ``[B, T, H]`` for the gates.
    This is the trigger's definition, not the FLA chunk kernel.
    """
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must share shape [B, T, H, D]")
    if beta.shape != q.shape[:-1] or alpha.shape != q.shape[:-1]:
        raise ValueError("beta and alpha must be [B, T, H]")

    q32 = q.float()
    k32 = k.float()
    v32 = v.float()
    beta32 = beta.float()
    alpha32 = alpha.float()
    scale = q32.size(-1) ** -0.5
    q_hat = q32 / q32.norm(dim=-1, keepdim=True).clamp_min(eps) * scale
    k_hat = k32 / k32.norm(dim=-1, keepdim=True).clamp_min(eps)

    batch, steps, heads, dim = q32.shape
    state = q32.new_zeros(batch, heads, dim, dim)
    outputs: list[Tensor] = []
    errors: list[Tensor] = []
    for t in range(steps):
        decayed = alpha32[:, t, :, None, None] * state
        predicted = torch.einsum("bhde,bhe->bhd", decayed, k_hat[:, t])
        delta = beta32[:, t, :, None] * (v32[:, t] - predicted)
        state = decayed + torch.einsum("bhd,bhe->bhde", k_hat[:, t], delta)
        outputs.append(torch.einsum("bhde,bhd->bhe", state, q_hat[:, t]))
        errors.append(collision_error(delta, v32[:, t], eps))
    o_lin = torch.stack(outputs, dim=1)
    err = torch.stack(errors, dim=1)
    return o_lin, err, q_hat, k_hat


class CollisionCache(nn.Module):
    """K-slot ring. Push when ``err_t > tau``, then read with causal softmax."""

    def __init__(self, slots: int = 32, tau: float = 0.5, eps: float = 1e-6) -> None:
        super().__init__()
        if slots < 1:
            raise ValueError("slots must be >= 1")
        self.slots = int(slots)
        self.tau = float(tau)
        self.eps = float(eps)

    def forward(
        self,
        q_hat: Tensor,
        k_hat: Tensor,
        v: Tensor,
        err: Tensor,
        o_lin: Tensor | None = None,
        enabled: bool = True,
    ) -> Tensor:
        if q_hat.ndim != 4 or k_hat.ndim != 4 or v.ndim != 4 or err.ndim != 3:
            raise ValueError("expected q/k/v [B,T,H,D] and err [B,T,H]")
        if not enabled:
            if o_lin is not None:
                return o_lin
            return v.new_zeros(q_hat.shape[:-1] + (v.size(-1),))
        readout = self._readout(q_hat, k_hat, v, err)
        if o_lin is None:
            return readout
        return o_lin + readout

    def _readout(self, q_hat: Tensor, k_hat: Tensor, v: Tensor, err: Tensor) -> Tensor:
        batch, steps, n_q, dim = q_hat.shape
        n_kv = k_hat.size(2)
        if n_q % n_kv != 0:
            raise ValueError(f"query heads {n_q} must be divisible by KV heads {n_kv}")
        if k_hat.shape[:2] != (batch, steps) or v.shape[:3] != (batch, steps, n_kv):
            raise ValueError("k and v must match batch, time, and KV heads")
        if err.shape != (batch, steps, n_kv):
            raise ValueError("err must be [B, T, n_kv]")
        groups = n_q // n_kv
        slots = self.slots
        keys = k_hat.new_zeros(batch, n_kv, slots, dim)
        vals = v.new_zeros(batch, n_kv, slots, v.size(-1))
        filled = k_hat.new_zeros(batch, n_kv, slots)
        ptr = torch.zeros(batch, n_kv, dtype=torch.long, device=k_hat.device)
        scale = dim ** -0.5
        outputs: list[Tensor] = []
        for t in range(steps):
            gate = (err[:, t] > self.tau).to(dtype=k_hat.dtype).detach()
            choose = torch.nn.functional.one_hot(ptr, slots).to(dtype=k_hat.dtype)
            choose = choose * gate.unsqueeze(-1)
            keep = 1.0 - choose
            keys = keep.unsqueeze(-1) * keys + choose.unsqueeze(-1) * k_hat[:, t].unsqueeze(2)
            vals = keep.unsqueeze(-1) * vals + choose.unsqueeze(-1) * v[:, t].unsqueeze(2)
            filled = keep * filled + choose
            ptr = torch.where(gate.bool(), (ptr + 1) % slots, ptr)

            q_t = q_hat[:, t].reshape(batch, n_kv, groups, dim)
            scores = torch.einsum("bhgd,bhkd->bhgk", q_t, keys) * scale
            empty = filled.sum(dim=-1) == 0
            scores = scores.masked_fill(filled.unsqueeze(2) == 0, float("-inf"))
            scores = scores.masked_fill(empty.unsqueeze(-1).unsqueeze(-1), 0.0)
            attn = torch.softmax(scores, dim=-1)
            attn = attn.masked_fill(empty.unsqueeze(-1).unsqueeze(-1), 0.0)
            out = torch.einsum("bhgk,bhkd->bhgd", attn, vals)
            outputs.append(out.reshape(batch, n_q, v.size(-1)))
        return torch.stack(outputs, dim=1)


def _heads(x: Tensor, head_dim: int) -> Tensor:
    return x.reshape(*x.shape[:-1], -1, head_dim)


def project_qkv_gates(module: nn.Module, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """q, k, v, beta, alpha from a Gated DeltaNet-style mixer.

    Short conv is used when the module has it, matching the FLA forward.
    Decay matches the kernel: ``exp(-exp(A_log) * softplus(a + dt_bias))``.
    """
    if getattr(module, "use_short_conv", False):
        q, _ = module.q_conv1d(
            x=module.q_proj(hidden), cache=None, output_final_state=False, cu_seqlens=None
        )
        k, _ = module.k_conv1d(
            x=module.k_proj(hidden), cache=None, output_final_state=False, cu_seqlens=None
        )
        v, _ = module.v_conv1d(
            x=module.v_proj(hidden), cache=None, output_final_state=False, cu_seqlens=None
        )
    else:
        q = torch.nn.functional.silu(module.q_proj(hidden))
        k = torch.nn.functional.silu(module.k_proj(hidden))
        v = torch.nn.functional.silu(module.v_proj(hidden))

    head_k = int(getattr(module, "head_k_dim", module.head_dim))
    head_v = int(getattr(module, "head_v_dim", module.head_dim))
    q = _heads(q, head_k)
    k = _heads(k, head_k)
    v = _heads(v, head_v)
    if q.size(2) != k.size(2) or q.size(2) != v.size(2):
        raise ValueError(
            f"cache v0 needs equal q/k/v heads, got {q.size(2)}, {k.size(2)}, {v.size(2)}"
        )

    beta = torch.sigmoid(module.b_proj(hidden).float())
    if getattr(module, "allow_neg_eigval", False):
        beta = beta * 2
    raw = module.a_proj(hidden).float()
    dt_bias = getattr(module, "dt_bias", None)
    if dt_bias is not None:
        raw = raw + dt_bias.float()
    alpha = torch.exp(-torch.exp(module.A_log.float()) * torch.nn.functional.softplus(raw))
    return q, k, v, beta, alpha


def apply_collision_cache(
    module: nn.Module,
    hidden: Tensor,
    student_out: Tensor,
    cache: CollisionCache,
    enabled: bool,
) -> Tensor:
    """Add the cache read-out through ``o_proj``. Disabled returns ``student_out``."""
    if not enabled:
        return student_out
    q, k, v, beta, alpha = project_qkv_gates(module, hidden)
    _o_lin, err, q_hat, k_hat = gated_delta_scan(q, k, v, beta, alpha, eps=cache.eps)
    readout = cache(q_hat, k_hat, v, err, enabled=True)
    flat = readout.reshape(readout.shape[0], readout.shape[1], -1).to(dtype=module.o_proj.weight.dtype)
    return student_out + module.o_proj(flat)
