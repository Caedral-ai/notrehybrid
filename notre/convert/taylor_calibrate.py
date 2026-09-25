"""Taylor-Calibrate student init in fp16. Strategy taylor_calibrate from the paper repo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from notre.convert.init_student import build_student_from_teacher
from notre.convert.paths import assert_t4_or_newer, ensure_taylor_on_path
from notre.convert.surgery import SMOLLM2_HF


def run_taylor_calibrate(cfg: dict, output: Path, hf_teacher: str) -> None:
    ensure_taylor_on_path()
    from init_ckpt_calibrated import apply_taylor_calibrate

    name = assert_t4_or_newer()
    print(f"Taylor-Calibrate fp16 on {name}")
    device = "cuda"
    # fp32: their per-layer AdamW step overflows fp16 weights and the PPL is NaN.
    student = build_student_from_teacher(cfg, dtype=torch.float32)
    student = student.eval().cpu()
    torch.cuda.empty_cache()
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_teacher, torch_dtype=torch.float16, attn_implementation="eager"
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_teacher)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    apply_taylor_calibrate(student, hf_model, tokenizer, cfg, device)
    del hf_model
    torch.cuda.empty_cache()
    output.mkdir(parents=True, exist_ok=True)
    student.cpu().save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(cfg["teacher_model"]["name"]).save_pretrained(output)
    print(f"taylor-calibrated student → {output}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hf-teacher", default=SMOLLM2_HF)
    p.add_argument("--teacher", type=Path, default=None, help="FLA teacher dir (overrides yaml)")
    args = p.parse_args()
    cfg = yaml.safe_load(args.cfg.read_text())
    if args.teacher:
        cfg["teacher_model"]["name"] = str(args.teacher)
    run_taylor_calibrate(cfg, args.output, args.hf_teacher)
    # Non-daemon threads in the HF/torch stack can keep the process alive after
    # the last print, which stalls the next notebook cell.
    os._exit(0)


if __name__ == "__main__":
    main()
