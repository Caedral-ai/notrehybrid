# Execution funnel

Public schedule for NotreHybrid. This is **Caedral research** toward a methods paper, not a product roadmap. Do not skip gates.

The long-form plan stays in local `internal-docs/` (gitignored; not on remotes). Hypothesis, ownership, and Week 0 smoke: [README](../README.md).

> **Now: Gate A (Week 1).** Week 0 smoke passed ([decisions/week-0.md](decisions/week-0.md)). SmolLM2 surgery + Taylor-Calibrate + transfer, **no cache**.

## Funnel

| When | Work | Artifact | Kill / next |
|---|---|---|---|
| **Week 0** | Setup + FLA smoke + dummy `HybridBlock` + resume | [decisions/week-0.md](decisions/week-0.md) — **passed** 2026-09-17 | Then Gate A |
| **Week 1** | Gate A — SmolLM2 3:1 surgery, Taylor-Calibrate, transfer ~5M tokens, **no cache** | [decisions/gate-A.md](decisions/gate-A.md) | PPL still thousands or MSE does not fall → abandon |
| **Weeks 2–3** | Gate B — collision cache vs twin; ΔMSE **decides**; MQAR reports | [decisions/gate-B.md](decisions/gate-B.md) + ΔMSE table | ΔMSE ≤ 0 after K/τ sweep → **end**. MQAR+ and MSE− → **end** |
| **Weeks 4–6** | Gate C — Qwen2.5-0.5B ± twin, only if B passed | Qwen checkpoint + twin + PPL | MSE fail → SmolLM2-only (pilot) paper, or skip replication |
| **Week 7+** | Optional appendix + draft | figures in the paper, not the README headline | Decode / OpenVINO only after ΔMSE on Qwen |

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

Notebook: [notebooks/kaggle_week0_smoke.ipynb](../notebooks/kaggle_week0_smoke.ipynb) — clone [Caedral-ai/notrehybrid](https://github.com/Caedral-ai/notrehybrid), T4×2, Internet on. Private Hub: [hf-hub.md](hf-hub.md).

## What is in vs out

| In the paper if gates pass | Out of this project |
|---|---|
| Trainable collision cache, `err_t` trigger, transfer ΔMSE | Shipping a chat model or Caedral API feature |
| Ablations: LoLA trigger, teacher top-K, random slots | Rust, FoX, ACP, QAT, Notre-1B, `notre-cpu` |
| Qwen replication after B | tok/s vs llama.cpp as success |

Converted SmolLM2 / Qwen are **vehicles**. The experiment is whether the cache moves MSE.

## Compute

Kaggle T4×2 (30 h/week), Colab T4 overflow, Lightning for short dev. Checkpoint every 25–30 min once Hub exists. FineWeb-Edu streaming when transfer starts — do not download the corpus to disk.

## Decisions

Fill the templates when the run finishes, not before:

- [week-0.md](decisions/week-0.md) — **GO A** (2026-09-17)
- [gate-A.md](decisions/gate-A.md) — GO B / ABANDON / RETRY (LoLCATs)
- [gate-B.md](decisions/gate-B.md) — GO C / END OF PROJECT
