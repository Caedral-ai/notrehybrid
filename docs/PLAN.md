# Execution funnel

Public schedule for NotreHybrid. This was **Caedral research** toward a methods paper, not a product roadmap. The schedule is closed.

The long-form plan stays in local `internal-docs/` (gitignored; not on remotes). Hypothesis, ownership, and Week 0 smoke: [README](../README.md).

> **Ended 2026-09-25.** Gate B failed ([decisions/gate-B.md](decisions/gate-B.md)). Mean transfer MSE with the cache was higher than the twin at every measured (K, τ). No Gate C. No tech report.

## Funnel

| When | Work | Artifact | Kill / next |
|---|---|---|---|
| **Week 0** | Setup + FLA smoke + dummy `HybridBlock` + resume | [decisions/week-0.md](decisions/week-0.md) — **passed** 2026-09-17 | Then Gate A |
| **Week 1** | Gate A — SmolLM2 3:1 surgery, Taylor-Calibrate, transfer probe, **no cache** | [decisions/gate-A.md](decisions/gate-A.md) — **passed** 2026-09-24 | PPL still thousands or MSE does not fall → abandon |
| **Weeks 2–3** | Gate B — collision cache vs twin; ΔMSE **decides** | [decisions/gate-B.md](decisions/gate-B.md) — **failed** 2026-09-25 | ΔMSE ≤ 0 after the K/τ sweep → **end** |
| **Weeks 4–6** | Gate C — Qwen2.5-0.5B, only if B passed | will not run | |
| **Week 7+** | Optional appendix + draft | will not run | |

```
A fails     → abandon
B fails     → END. No tech report.
C MSE fails → pilot-only, or skip replication
A+B+C MSE   → methods workshop / arXiv
appendix    → extra figure, not the claim
```

## Week 0 exit (before Gate A)

All of:

1. `print('FLA OK', …)` on T4-class GPU (capability ≥ 7), fp16, not bf16.
2. `python -m pytest tests/test_hybrid_block.py -q`
3. Train ~10 min, kill the session, `--resume auto` continues `curve.csv`.

Then stop. Do not download SmolLM2 or Qwen to “get ahead.”

**Passed 2026-09-17** — log: [decisions/week-0.md](decisions/week-0.md).

Week 0 notebook: [notebooks/kaggle_week0_smoke.ipynb](../notebooks/kaggle_week0_smoke.ipynb).
Gate A notebook: [notebooks/kaggle_gate_a.ipynb](../notebooks/kaggle_gate_a.ipynb) — clone [Caedral-ai/notrehybrid](https://github.com/Caedral-ai/notrehybrid), T4×2, Internet on. Private Hub: [hf-hub.md](hf-hub.md).

## Gate A commands

Taylor-Calibrate has no SmolLM2 converter and its trainer is bf16/8-GPU. Gate A converts SmolLM2 locally, calls their Taylor init, and runs transfer in fp16 on T4. **No cache.**

```sh
python -m notre.convert.convert_smollm2 --out ./teachers/SmolLM2-360M
python -m notre.convert.init_student --cfg configs/smollm2_360m/gate_a.yaml \
  --output ./checkpoints/gate-a/init-copy --teacher ./teachers/SmolLM2-360M
python -m notre.convert.taylor_calibrate --cfg configs/smollm2_360m/gate_a.yaml \
  --output ./checkpoints/gate-a/init-taylor --teacher ./teachers/SmolLM2-360M
python -m notre.eval.ppl --ckpt ./checkpoints/gate-a/init-copy --tokenizer ./teachers/SmolLM2-360M
python -m notre.eval.ppl --ckpt ./checkpoints/gate-a/init-taylor --tokenizer ./teachers/SmolLM2-360M
python -m notre.convert.transfer --cfg configs/smollm2_360m/gate_a.yaml \
  --teacher ./teachers/SmolLM2-360M --student-init ./checkpoints/gate-a/init-taylor \
  --ckpt-dir ./checkpoints/gate-a --resume auto
```

Pass: calibrated zero-shot PPL ≪ copy-only; transfer MSE falls and stabilizes; no NaN in fp16.

**Passed 2026-09-24** — log: [decisions/gate-A.md](decisions/gate-A.md). Copy-only PPL 1156, Taylor PPL 346, probe MSE 1.555 → 0.798.

## Gate B result

Same SmolLM2 student, seed 0, Taylor init. Cache on versus the ring off. The decision step is step 500, where the 20-minute twin has MSE **0.94699**.

| K | τ | MSE @ 500 | ΔMSE |
|---|---|---|---|
| 8 | 0.3 | 0.95972 | −1.34% |
| 8 | 0.5 | 0.96274 | −1.66% |
| 8 | 0.7 | 0.96713 | −2.13% |
| 32 | 0.3 | 0.96978 | −2.41% |
| 32 | 0.5 | 0.97057 | −2.49% |
| 32 | 0.7 | 0.97557 | −3.02% |
| 64 | 0.3 | 0.97235 | −2.68% |
| 64 | 0.5 | | not run |
| 64 | 0.7 | | not run |

**Failed 2026-09-25** — log: [decisions/gate-B.md](decisions/gate-B.md). MQAR and trigger ablations were not run.

## What is in vs out

| Measured | Out of this project |
|---|---|
| Trainable collision cache, `err_t` trigger, transfer ΔMSE on SmolLM2 | A methods paper, a tech report, Qwen |
| The K/τ cells in the table above | LoLA / teacher top-K / random trigger ablations |
| | Shipping a chat model or a Caedral API feature |
| | Rust, FoX, ACP, QAT, Notre-1B, `notre-cpu`, tok/s vs llama.cpp |

Converted SmolLM2 was the vehicle. The cache did not move MSE in the direction the paper needed.

## Compute

Kaggle T4×2 (30 h/week), Colab T4 overflow, Lightning for short dev. Checkpoint every 25–30 min once Hub exists. FineWeb-Edu streaming when transfer starts — do not download the corpus to disk.

## Decisions

The runs are recorded:

- [week-0.md](decisions/week-0.md) — **GO A** (2026-09-17)
- [gate-A.md](decisions/gate-A.md) — **GO B** (2026-09-24)
- [gate-B.md](decisions/gate-B.md) — **END OF PROJECT** (2026-09-25)
