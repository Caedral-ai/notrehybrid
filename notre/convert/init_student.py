"""Copy-only student: Taylor-Calibrate init_from_teacher in fp16 (no bf16 on T4)."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from notre.convert.paths import register_hf_classes
from notre.convert.surgery import keep_softmax_layers


def _copy_if_present(dst_module, src_module, name: str) -> None:
    if not (hasattr(dst_module, name) and hasattr(src_module, name)):
        return
    dst, src = getattr(dst_module, name), getattr(src_module, name)
    if hasattr(dst, "weight") and hasattr(src, "weight") and dst.weight is not None and src.weight is not None:
        dst.weight.data.copy_(src.weight.data)
    if hasattr(dst, "bias") and hasattr(src, "bias") and dst.bias is not None and src.bias is not None:
        dst.bias.data.copy_(src.bias.data)
    if hasattr(dst, "eps") and hasattr(src, "eps"):
        dst.eps = src.eps
    if hasattr(dst, "variance_epsilon") and hasattr(src, "variance_epsilon") and hasattr(dst, "eps"):
        dst.eps = src.variance_epsilon


def build_student_from_teacher(cfg: dict, dtype: torch.dtype = torch.float32):
    register_hf_classes()
    from distill_model.config_distilled_student import StudentConfig

    teacher_name = cfg["teacher_model"]["name"]
    student_name = cfg["student_model"]["name"]
    keep_layers = cfg["student_model"].get(
        "keep_full_attention_layers", keep_softmax_layers()
    )

    print(f"building student from {teacher_name}", flush=True)
    teacher_config = AutoConfig.from_pretrained(teacher_name, local_files_only=True)
    config_dict = teacher_config.to_dict()
    config_dict["name"] = "student"
    config_dict["student_name"] = student_name
    config_dict["keep_full_attention_layers"] = list(keep_layers)
    student_config = StudentConfig(**config_dict)

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=dtype, low_cpu_mem_usage=False, local_files_only=True
    )
    print("teacher module ready", flush=True)
    student = AutoModelForCausalLM.from_config(student_config, torch_dtype=dtype)

    student.model.embeddings.weight.data.copy_(teacher.model.embeddings.weight.data)
    if hasattr(student.model, "norm") and hasattr(teacher.model, "norm"):
        _copy_if_present(student.model, teacher.model, "norm")
    if hasattr(student, "lm_head") and hasattr(teacher, "lm_head"):
        student.lm_head.weight.data.copy_(teacher.lm_head.weight.data)
        if getattr(student.lm_head, "bias", None) is not None and getattr(teacher.lm_head, "bias", None) is not None:
            student.lm_head.bias.data.copy_(teacher.lm_head.bias.data)

    for idx, (student_layer, teacher_layer) in enumerate(
        zip(student.model.layers, teacher.model.layers)
    ):
        _copy_if_present(student_layer, teacher_layer, "attn_norm")
        _copy_if_present(student_layer, teacher_layer, "mlp_norm")
        if idx in keep_layers:
            student_layer.attn.load_state_dict(teacher_layer.attn.state_dict(), strict=False)
        else:
            student_layer.attn.init_from_teacher(teacher_layer.attn)
        student_layer.mlp.load_state_dict(teacher_layer.mlp.state_dict(), strict=False)
        teacher.model.layers[idx] = None

    del teacher
    gc.collect()
    return student


def main() -> None:
    p = argparse.ArgumentParser(description="Copy-only GDN student (default gates)")
    p.add_argument("--cfg", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--teacher", type=Path, default=None, help="FLA teacher dir (overrides yaml)")
    args = p.parse_args()
    cfg = yaml.safe_load(args.cfg.read_text())
    if args.teacher:
        cfg["teacher_model"]["name"] = str(args.teacher)
    args.output.mkdir(parents=True, exist_ok=True)
    model = build_student_from_teacher(cfg)
    tok = AutoTokenizer.from_pretrained(cfg["teacher_model"]["name"])
    model.save_pretrained(args.output, safe_serialization=True)
    tok.save_pretrained(args.output)
    print(f"copy-only student → {args.output}")


if __name__ == "__main__":
    main()
