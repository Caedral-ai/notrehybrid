#!/usr/bin/env bash
# Week 0 Kaggle / Colab setup. T4 or better (capability >= 7). Never P100.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python - <<'PY'
import sys

try:
    import torch
except ImportError:
    sys.exit("torch missing — pip install the Kaggle GPU torch wheel first")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required. Select a GPU runtime (Kaggle T4×2).")
major, minor = torch.cuda.get_device_capability()
name = torch.cuda.get_device_name()
print(f"GPU: {name} capability {major}.{minor}")
if major < 7:
    raise SystemExit(
        f"Refuse {name} (capability {major}.{minor}). Never P100. Use T4 or newer."
    )
print("GPU check OK — keeping preinstalled torch", torch.__version__)
PY

# Keep Kaggle's CUDA torch. Do not pip install torch or causal_conv1d:
# Kaggle has no nvcc, and there is often no wheel for this torch/CUDA pair.
# FLA short-conv falls back to Triton. Zoology is cloned, not pip-installed
# (its setup pulls causal_conv1d and breaks Week 0).
pip install -q einops datasets huggingface_hub peft pytest
pip install -q "flash-linear-attention[cuda]"
pip install -e "$ROOT" --no-deps

mkdir -p third_party
if [[ ! -d third_party/lolcats ]]; then
  git clone --depth 1 https://github.com/HazyResearch/lolcats third_party/lolcats
fi
if [[ ! -d third_party/Taylor-Calibrate ]]; then
  git clone --depth 1 https://github.com/FutureMLS-Lab/Taylor-Calibrate third_party/Taylor-Calibrate
fi
if [[ ! -d third_party/zoology ]]; then
  git clone --depth 1 https://github.com/HazyResearch/zoology third_party/zoology
fi

echo "setup_kaggle.sh done"
