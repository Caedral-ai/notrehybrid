"""Week 0 hybrid stack: 3× FLA GatedDeltaNet + 1× dummy softmax.

Collision cache / err_t is Week 2. Inherited teacher GQA is Week 1 surgery.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

try:
    from fla.layers import GatedDeltaNet
except ImportError:  # pragma: no cover - tests skip if FLA missing
    GatedDeltaNet = None


def unpack_gdn(out: Tensor | tuple) -> Tensor:
    if isinstance(out, tuple):
        return out[0]
    return out


class DummySoftmax(nn.Module):
    """Stand-in for the 1/4 inherited teacher attention. Not GQA or RoPE."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        return x + attn_out


class HybridBlock(nn.Module):
    """Three GDN mixers then one dummy softmax. Forward + backward only."""

    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 8,
        head_dim: int = 64,
        mode: str = "chunk",
    ) -> None:
        super().__init__()
        if GatedDeltaNet is None:
            raise ImportError("flash-linear-attention is required for HybridBlock")
        self.gdn = nn.ModuleList(
            [
                GatedDeltaNet(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    mode=mode,
                )
                for _ in range(3)
            ]
        )
        self.softmax = DummySoftmax(hidden_size, num_heads)

    def forward(self, x: Tensor) -> Tensor:
        for gdn in self.gdn:
            x = unpack_gdn(gdn(x))
        return self.softmax(x)
