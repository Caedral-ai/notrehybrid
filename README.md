<p align="center">
  <a href="https://caedral.com">
    <img src="docs/assets/caedral-mark.png" alt="Caedral" width="96" height="96">
  </a>
</p>

<h1 align="center">NotreHybrid</h1>

<p align="center">
  <strong><code>notrehybrid</code></strong> · Caedral research on a trainable collision cache for GDN linearization.
</p>

Does a small K-slot buffer, triggered by Gated DeltaNet delta error, reduce
attention-transfer MSE enough to write a methods paper? Converted models are
vehicles for that measurement. This repository is **not** a product, not an
API, and not something to install or sell.

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="notebooks/kaggle_week0_smoke.ipynb"><img src="https://img.shields.io/badge/compute-Kaggle%20T4%C3%972-20BEFF?logo=kaggle&logoColor=white" alt="Compute"></a>
  <a href="#status"><img src="https://img.shields.io/badge/status-week%200%20setup-orange" alt="Status"></a>
  <a href="https://caedral.com"><img src="https://img.shields.io/badge/Caedral-research-111111" alt="Caedral"></a>
</p>

*A project of [Caedral](https://caedral.com). The public Caedral surface is a
subscription OpenAI-compatible API (chat, embeddings, rerank). NotreHybrid
is separate internal research.*

---

> **Status: week 0 — setup.** FLA smoke on Kaggle T4×2, dummy 3:1 hybrid
> block, checkpoint `--resume auto`. No SmolLM2, no Qwen, no cache yet.
> See [docs/PLAN.md](docs/PLAN.md).

## Why

Linear attention (Gated DeltaNet) is cheap to run but loses the tokens the
delta rule does not absorb. LoLA already uses an inference-only buffer with
a different trigger. The bet here is narrower:

**Promote the residual the GDN state failed to write, train that buffer
during attention transfer, and see if imitation MSE actually moves.**

If it does not, the project ends. There is no fallback of “publish the
conversion” and no plan to ship a chat model.

## The hypothesis

Linearize a small Transformer (LoLCATs + Taylor-Calibrate → GDN, **3:1**
hybrid). On linear layers:

```
err_t = ||δ_t||₂ / (||v_t||₂ + ε)
if err_t > τ_promote:  cache.push(k̂_t, v_t)     # K=32, ring
o_t = o_lin + softmax(q̂_t · k_cache / √d) · v_cache
```

Compare transfer MSE **with** vs **without** the cache (same seed, same
trainable budget). ΔMSE is the paper. Everything else is scaffolding.

## What decides the paper

| Gate | Signal | Pass | Fail |
|---|---|---|---|
| **A** | SmolLM2 transfer → GDN (no cache) | MSE falls and stabilizes; calibrated PPL ≪ copy-only | abandon |
| **B** | collision cache vs identical twin | `MSE_with < MSE_without` on a majority of layers | **end of project** |
| **C** | same ΔMSE on Qwen2.5-0.5B | replication | pilot-only paper, or skip |

MQAR is **support**. If MQAR wins and MSE does not → abandon (synthetic
false positive). Decode `ms/token`, Rust, and tok/s vs `llama.cpp` are
**not** kill criteria and not the headline.

```
Week 0    setup + FLA smoke
Week 1    Gate A — SmolLM2
Weeks 2–3 Gate B — cache (kill hard)
Weeks 4–6 Gate C — Qwen, only if B passed
```

## Status

| Piece | Badge | Notes |
|---|---|---|
| Repo scaffold | ![](https://img.shields.io/badge/status-working-brightgreen) | Apache-2.0, `pyproject.toml`, Kaggle setup |
| Dummy 3:1 `HybridBlock` | ![](https://img.shields.io/badge/status-working-brightgreen) | 3× FLA GDN + dummy softmax; CPU unit test |
| FLA T4 smoke | ![](https://img.shields.io/badge/status-run%20on%20Kaggle-lightgrey) | print `FLA OK`; never P100 |
| Resume harness | ![](https://img.shields.io/badge/status-run%20on%20Kaggle-lightgrey) | 10 min train / kill / `--resume auto` |
| Collision cache | ![](https://img.shields.io/badge/status-week%202-orange) | `notre_linear.py` — not started |
| Gate A / B / C | ![](https://img.shields.io/badge/status-not%20started-lightgrey) | templates in `docs/decisions/` |

Long-form notes live in local `internal-docs/` (gitignored, not on remotes).

## Layout

```
README.md · LICENSE · pyproject.toml · setup_kaggle.sh
docs/PLAN.md · docs/decisions/gate-A.md · gate-B.md
notebooks/kaggle_week0_smoke.ipynb
notre/
  layers/hybrid_block.py     # Week 0: 3× GDN + dummy softmax
  train/smoke_resume.py      # local checkpoint resume
tests/test_hybrid_block.py
internal-docs/               # gitignored — canonical research plan
```

Week 1+ (not in tree yet): `notre_linear.py`, surgery, Taylor-Calibrate,
transfer, MSE / MQAR eval.

## Quick start (Week 0)

**GPU:** Kaggle **T4×2**. fp16 + GradScaler. Capability ≥ 7. **Never P100.
Never bf16 on T4.**

```sh
bash setup_kaggle.sh

python -c "
from fla.layers import GatedDeltaNet
import torch
assert torch.cuda.get_device_capability()[0] >= 7
m = GatedDeltaNet(hidden_size=512, num_heads=8, mode='chunk').cuda().half()
x = torch.randn(2, 1024, 512, device='cuda', dtype=torch.float16)
assert m(x)[0].shape == x.shape
print('FLA OK', torch.cuda.get_device_name())
"

python -m pytest tests/test_hybrid_block.py -q

python -m notre.train.smoke_resume --minutes 10 --ckpt-dir ./checkpoints/week0
python -m notre.train.smoke_resume --minutes 10 --ckpt-dir ./checkpoints/week0 --resume auto
```

Notebook: [notebooks/kaggle_week0_smoke.ipynb](notebooks/kaggle_week0_smoke.ipynb).

Private Hub (after `huggingface-cli login`):

```sh
bash scripts/create_hf_repo.sh
```

Weights stay private unless a later gate actually earns a paper release.

## Principles

- **The cache is the experiment.** The linearized model is a vehicle.
- **Kill hard at Gate B.** No tech report if ΔMSE ≤ 0 after a short K/τ sweep.
- **Measure the right thing.** Transfer MSE decides; MQAR reports; decode is appendix.
- **Free T4 only.** Kaggle / Colab / Lightning. Never P100 (no Triton).
- **Do not skip gates.** No Qwen, no cache, no paper scaffolding before A then B.

## Non-goals

Caedral API features, a chat model users install, FoX / ACP, QAT, Notre-1B,
`notre-cpu`, Rust, or speed vs `ik_llama.cpp`.

---

Part of the [Caedral](https://caedral.com) ecosystem —
subscription AI infrastructure for automation agencies.

Apache-2.0 — see [LICENSE](LICENSE).
How we commit and merge: [CONTRIBUTING.md](CONTRIBUTING.md).
