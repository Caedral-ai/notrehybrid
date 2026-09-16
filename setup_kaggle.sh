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

# Do not `pip install torch`: Kaggle already ships a CUDA build; a CPU wheel would kill FLA.
pip install -q flash-linear-attention transformers datasets huggingface_hub peft pytest
pip install -q "git+https://github.com/HazyResearch/zoology"

mkdir -p third_party
if [[ ! -d third_party/lolcats ]]; then
  git clone --depth 1 https://github.com/HazyResearch/lolcats third_party/lolcats
fi
if [[ ! -d third_party/Taylor-Calibrate ]]; then
  git clone --depth 1 https://github.com/FutureMLS-Lab/Taylor-Calibrate third_party/Taylor-Calibrate
fi

pip install -e "$ROOT"

echo "setup_kaggle.sh done"
