"""SDPA stand-in for flash-attn.

FLA softmax layers import flash-attn in ``Attention.__init__``. Kaggle T4 has
no nvcc, so that wheel does not build. Register this before importing ``fla``.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types

import torch
import torch.nn.functional as F


def _as_bhld(t: torch.Tensor) -> torch.Tensor:
    """flash-attn layout (B, L, H, D) → SDPA layout (B, H, L, D)."""
    return t.transpose(1, 2)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    **kwargs,
) -> torch.Tensor:
    del dropout_p, kwargs
    q_s, k_s, v_s = _as_bhld(q), _as_bhld(k), _as_bhld(v)
    enable_gqa = q_s.size(1) != k_s.size(1)
    attn_mask = None
    left = window_size[0] if window_size else -1
    if left is not None and left >= 0:
        seq = q_s.size(2)
        idx = torch.arange(seq, device=q.device)
        band = (idx[None, :] - idx[:, None]) <= left
        causal_mask = idx[None, :] <= idx[:, None]
        attn_mask = (band & causal_mask) if causal else band
        causal = False
    out = F.scaled_dot_product_attention(
        q_s,
        k_s,
        v_s,
        attn_mask=attn_mask,
        is_causal=causal,
        scale=softmax_scale,
        enable_gqa=enable_gqa,
    )
    return out.transpose(1, 2)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    **kwargs,
) -> torch.Tensor:
    """Packed sequences of equal length, which is the Gate A batch shape."""
    del cu_seqlens_k, max_seqlen_k, kwargs
    batch = int(cu_seqlens_q.numel() - 1)
    heads_q, dim = q.shape[-2], q.shape[-1]
    heads_k = k.shape[-2]
    q_b = q.view(batch, max_seqlen_q, heads_q, dim)
    k_b = k.view(batch, max_seqlen_q, heads_k, dim)
    v_b = v.view(batch, max_seqlen_q, heads_k, dim)
    out = flash_attn_func(
        q_b,
        k_b,
        v_b,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
    )
    return out.reshape(-1, heads_q, dim)


def install() -> None:
    """Expose SDPA as flash_attn unless a real install is already present.

    Transformers calls ``find_spec('flash_attn')`` while FLA imports. A module
    with ``__spec__ is None`` raises. Let that check run first, then register a
    spec-backed stand-in so FLA softmax layers can be constructed on T4.
    """
    try:
        import importlib.metadata as metadata

        metadata.version("flash_attn")
        return
    except metadata.PackageNotFoundError:
        pass
    sys.modules.pop("flash_attn", None)
    import transformers.modeling_flash_attention_utils  # noqa: F401

    mod = types.ModuleType("flash_attn")
    mod.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn", loader=None, origin="notre-sdpa-fallback"
    )
    mod.__file__ = "notre-sdpa-fallback"
    mod.__package__ = "flash_attn"
    mod.flash_attn_func = flash_attn_func
    mod.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = mod
    attn = sys.modules.get("fla.layers.attn")
    if attn is not None and getattr(attn, "flash_attn_func", None) is None:
        attn.flash_attn_func = flash_attn_func
        attn.flash_attn_varlen_func = flash_attn_varlen_func


install()
