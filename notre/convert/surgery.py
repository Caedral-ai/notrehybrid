"""3:1 hybrid layer map for Gate A. No cache."""

from __future__ import annotations


SMOLLM2_LAYERS = 32
SMOLLM2_HF = "HuggingFaceTB/SmolLM2-360M"


def keep_softmax_layers(n_layers: int = SMOLLM2_LAYERS) -> list[int]:
    """Uniform every 4th layer → 1/4 teacher softmax, 3/4 GDN."""
    if n_layers % 4 != 0:
        raise ValueError(f"n_layers={n_layers} is not divisible by 4")
    return list(range(0, n_layers, 4))
