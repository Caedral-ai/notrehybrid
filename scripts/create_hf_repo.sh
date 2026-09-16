#!/usr/bin/env bash
# Private Hub for checkpoints. Run after `huggingface-cli login`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "$ROOT/scripts/create_hf_repo.py"
