"""Locate the cloned Taylor-Calibrate tree (setup_kaggle.sh)."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def taylor_root() -> Path:
    path = repo_root() / "third_party" / "Taylor-Calibrate"
    if not (path / "src").is_dir():
        raise FileNotFoundError(
            f"Taylor-Calibrate not at {path}. Run bash setup_kaggle.sh"
        )
    return path


def ensure_taylor_on_path() -> Path:
    src = taylor_root() / "src"
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)
    scripts = taylor_root() / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return src


def assert_t4_or_newer() -> str:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required. Use Kaggle T4×2.")
    major, _ = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    if major < 7:
        raise SystemExit(f"Refuse {name} (capability {major}). Never P100.")
    return name


def register_hf_classes() -> None:
    """FLA Transformer teacher + Taylor-Calibrate StudentForCausalLM."""
    import notre.convert.flash_attn_sdpa  # noqa: F401  before fla

    ensure_taylor_on_path()
    import fla  # noqa: F401
    from transformers import AutoConfig, AutoModelForCausalLM
    from fla.models.transformer.configuration_transformer import TransformerConfig
    from fla.models.transformer.modeling_transformer import TransformerForCausalLM
    from distill_model.config_distilled_student import StudentConfig
    from distill_model.modeling_distilled_student import StudentForCausalLM

    AutoConfig.register("transformer", TransformerConfig, exist_ok=True)
    AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM, exist_ok=True)
    AutoConfig.register("student", StudentConfig, exist_ok=True)
    AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)
