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
  <a href="#status"><img src="https://img.shields.io/badge/status-ended%20at%20gate%20B-red" alt="Status"></a>
  <a href="https://caedral.com"><img src="https://img.shields.io/badge/Caedral-research-111111" alt="Caedral"></a>
</p>

*A project of [Caedral](https://caedral.com). The public Caedral surface is a
subscription OpenAI-compatible API (chat, embeddings, rerank). NotreHybrid
is separate internal research.*

---

> **Status: ended 2026-09-25 (Gate B failed).** Week 0 passed 2026-09-17.
> Gate A passed 2026-09-24. The collision cache did not reduce transfer MSE.
> Log: [docs/decisions/gate-B.md](docs/decisions/gate-B.md).
> No Gate C. No tech report.

## Why

Linear attention (Gated DeltaNet) is cheap to run but loses the tokens the
delta rule does not absorb. LoLA already uses an inference-only buffer with
a different trigger. The bet here is narrower:

**Promote the residual the GDN state failed to write, train that buffer
during attention transfer, and see if imitation MSE actually moves.**

If it does not, the project ends. It did not. Gate B failed on 2026-09-25.
There is no fallback of “publish the conversion” and no plan to ship a chat model.

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
| **A** | SmolLM2 transfer → GDN (no cache) | **passed** 2026-09-24 | abandon |
| **B** | collision cache vs identical twin | `MSE_with < MSE_without` on a majority of layers | **failed** 2026-09-25 — end of project |
| **C** | same ΔMSE on Qwen2.5-0.5B | replication | will not run |

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
| FLA T4 smoke | ![](https://img.shields.io/badge/status-passed-brightgreen) | `FLA OK Tesla T4` (2026-09-17) |
| Resume harness | ![](https://img.shields.io/badge/status-passed-brightgreen) | loaded 3 → 21687 → 25871; see [docs/decisions/week-0.md](docs/decisions/week-0.md) |
| Collision cache | ![](https://img.shields.io/badge/status-no%20gain-red) | measured; mean MSE worse than the twin at step 500 |
| Gate A convert / transfer | ![](https://img.shields.io/badge/status-passed-brightgreen) | PPL 1156 → 346; MSE 1.555 → 0.798; [docs/decisions/gate-A.md](docs/decisions/gate-A.md) |
| Gate B | ![](https://img.shields.io/badge/status-failed-red) | ΔMSE −1.3% to −3.0% at step 500; [docs/decisions/gate-B.md](docs/decisions/gate-B.md) |
| Gate C | ![](https://img.shields.io/badge/status-will%20not%20run-lightgrey) | Qwen stays unstarted |

Long-form notes live in local `internal-docs/` (gitignored, not on remotes).

## Layout

```
README.md · LICENSE · pyproject.toml · setup_kaggle.sh
docs/PLAN.md · docs/decisions/week-0.md · gate-A.md · gate-B.md
notebooks/kaggle_week0_smoke.ipynb · kaggle_gate_a.ipynb · kaggle_gate_b.ipynb
configs/smollm2_360m/gate_a.yaml
notre/
  layers/hybrid_block.py     # Week 0: 3× GDN + dummy softmax
  layers/notre_linear.py     # collision cache, err_t ring
  layers/cache_scan.py       # Triton delta-error scan
  train/smoke_resume.py      # local checkpoint resume
  convert/                   # HF→FLA, 3:1, Taylor init, transfer ± cache
  eval/ppl.py                # WikiText-2 PPL (fp32)
tests/test_hybrid_block.py · test_surgery.py · test_notre_linear.py
internal-docs/               # gitignored — canonical research plan
```

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

## Gate A (Week 1)

**No cache.** Convert SmolLM2 ourselves (Taylor-Calibrate has no SmolLM2
converter). Call their Taylor init. Run transfer in **fp16** on T4. Never bf16.

```sh
bash setup_kaggle.sh

python -m pytest tests/test_surgery.py tests/test_hybrid_block.py -q

python -m notre.convert.convert_smollm2 \
  --hf HuggingFaceTB/SmolLM2-360M \
  --out ./teachers/SmolLM2-360M

python -m notre.convert.init_student \
  --cfg configs/smollm2_360m/gate_a.yaml \
  --output ./checkpoints/gate-a/init-copy \
  --teacher ./teachers/SmolLM2-360M

python -m notre.convert.taylor_calibrate \
  --cfg configs/smollm2_360m/gate_a.yaml \
  --output ./checkpoints/gate-a/init-taylor \
  --teacher ./teachers/SmolLM2-360M

python -m notre.eval.ppl --ckpt ./checkpoints/gate-a/init-copy \
  --tokenizer ./teachers/SmolLM2-360M
python -m notre.eval.ppl --ckpt ./checkpoints/gate-a/init-taylor \
  --tokenizer ./teachers/SmolLM2-360M

python -m notre.convert.transfer --cfg configs/smollm2_360m/gate_a.yaml \
  --teacher ./teachers/SmolLM2-360M \
  --student-init ./checkpoints/gate-a/init-taylor \
  --ckpt-dir ./checkpoints/gate-a --resume auto
```

Notebook: [notebooks/kaggle_gate_a.ipynb](notebooks/kaggle_gate_a.ipynb).
Recorded in [docs/decisions/gate-A.md](docs/decisions/gate-A.md) — **passed** 2026-09-24.

## Gate B (Weeks 2–3) — ended

Cache on, same student, same seed, ring off as the twin. ΔMSE at matched step 500 was negative on every measured (K, τ): about **−1.3%** (K=8, τ=0.3) to **−3.0%** (K=32, τ=0.7). Larger K and higher τ were worse. K=64 at τ=0.5 and τ=0.7 were not run.

MQAR and the trigger ablations were not run. They were required only if ΔMSE was positive.

Notebook: [notebooks/kaggle_gate_b.ipynb](notebooks/kaggle_gate_b.ipynb).
Recorded in [docs/decisions/gate-B.md](docs/decisions/gate-B.md) — **END OF PROJECT**, 2026-09-25.

Private Hub (after `huggingface-cli login`):

```sh
bash scripts/create_hf_repo.sh
```

Weights stay private unless a later gate actually earns a paper release.

## Principles

- **The cache is the experiment.** The linearized model is a vehicle.
- **Kill hard at Gate B.** That rule fired on 2026-09-25. No tech report.
- **Measure the right thing.** Transfer MSE decided; MQAR was support and was not run.
- **Free T4 only.** Kaggle / Colab / Lightning. Never P100 (no Triton).
- **Do not continue past B.** No Qwen, no paper scaffolding, no tech report.

## Non-goals

Caedral API features, a chat model users install, FoX / ACP, QAT, Notre-1B,
`notre-cpu`, Rust, or speed vs `ik_llama.cpp`.

---

Part of the [Caedral](https://caedral.com) ecosystem —
subscription AI infrastructure for automation agencies.

Apache-2.0 — see [LICENSE](LICENSE).
How we commit and merge: [CONTRIBUTING.md](CONTRIBUTING.md).
