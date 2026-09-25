"""Convert HuggingFace SmolLM2 (Llama) → FLA Transformer checkpoint.

Adapted from FutureMLS-Lab/Taylor-Calibrate convert/convert_from_llama3.2.py.
T4: fp16 only.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import notre.convert.flash_attn_sdpa  # noqa: F401  before fla
import fla  # noqa: F401
from fla.models.transformer.configuration_transformer import TransformerConfig
from fla.models.transformer.modeling_transformer import TransformerForCausalLM

from notre.convert.paths import coerce_tied_keys, rope_theta_from
from notre.convert.surgery import SMOLLM2_HF, SMOLLM2_LAYERS

coerce_tied_keys(TransformerForCausalLM)

AutoConfig.register("transformer", TransformerConfig, exist_ok=True)
AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM, exist_ok=True)

SMOLLM2_FLA = dict(
    attention_bias=False,
    bos_token_id=0,
    eos_token_id=0,
    fuse_cross_entropy=False,
    fuse_norm=True,
    hidden_act="swish",
    hidden_size=960,
    initializer_range=0.02,
    intermediate_size=2560,
    model_type="transformer",
    norm_eps=1e-5,
    num_heads=15,
    num_hidden_layers=SMOLLM2_LAYERS,
    num_kv_heads=5,
    rope_theta=100000.0,
    tie_word_embeddings=True,
    use_cache=True,
    vocab_size=49152,
    max_position_embeddings=8192,
    qkv_bias=False,
)


def _copy_linear(dst, src) -> None:
    dst.weight.data.copy_(src.weight)
    if dst.bias is not None and src.bias is not None:
        dst.bias.data.copy_(src.bias)


def convert(hf_id: str, output: Path, precision: str = "float16") -> None:
    dtype = torch.float16 if precision in {"float16", "fp16"} else torch.float32
    if precision in {"bfloat16", "bf16"}:
        raise SystemExit("Never bf16 on T4. Use fp16.")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    print(f"converting {hf_id} → {output}", flush=True)
    AutoTokenizer.from_pretrained(hf_id).save_pretrained(output)
    llama = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=dtype, low_cpu_mem_usage=False
    )
    cfg = TransformerConfig(**SMOLLM2_FLA)
    cfg.torch_dtype = dtype
    model = AutoModelForCausalLM.from_config(cfg).to(dtype)

    vocab = min(model.model.embeddings.weight.shape[0], llama.model.embed_tokens.weight.shape[0])
    if model.model.embeddings.weight.shape[0] != llama.model.embed_tokens.weight.shape[0]:
        warnings.warn("vocab size mismatch; copying the overlapping rows")
    model.model.embeddings.weight.data[:vocab].copy_(llama.model.embed_tokens.weight[:vocab])

    for i in range(cfg.num_hidden_layers):
        sl, tl = model.model.layers[i], llama.model.layers[i]
        if hasattr(sl, "attn_norm") and sl.attn_norm.weight is not None:
            sl.attn_norm.weight.data.copy_(tl.input_layernorm.weight)
        _copy_linear(sl.attn.q_proj, tl.self_attn.q_proj)
        _copy_linear(sl.attn.k_proj, tl.self_attn.k_proj)
        _copy_linear(sl.attn.v_proj, tl.self_attn.v_proj)
        sl.attn.o_proj.weight.data.copy_(tl.self_attn.o_proj.weight)
        if hasattr(sl, "mlp_norm") and sl.mlp_norm.weight is not None:
            sl.mlp_norm.weight.data.copy_(tl.post_attention_layernorm.weight)
        sl.mlp.gate_proj.weight.data.copy_(tl.mlp.gate_proj.weight)
        sl.mlp.up_proj.weight.data.copy_(tl.mlp.up_proj.weight)
        sl.mlp.down_proj.weight.data.copy_(tl.mlp.down_proj.weight)

    if model.model.norm.weight is not None:
        model.model.norm.weight.data.copy_(llama.model.norm.weight)
    if not model.config.tie_word_embeddings:
        model.lm_head.weight.data[:vocab].copy_(llama.lm_head.weight[:vocab])
    model.config.rope_theta = rope_theta_from(llama.config, default=float(model.config.rope_theta))
    model.save_pretrained(output)
    print(f"saved FLA teacher → {output}")


def main() -> None:
    p = argparse.ArgumentParser(description="SmolLM2 HF → FLA Transformer")
    p.add_argument("--hf", default=SMOLLM2_HF)
    p.add_argument("--out", type=Path, default=Path("./teachers/SmolLM2-360M"))
    p.add_argument("--precision", default="float16")
    args = p.parse_args()
    convert(args.hf, args.out, args.precision)
    os._exit(0)


if __name__ == "__main__":
    main()
