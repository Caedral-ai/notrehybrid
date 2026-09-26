# Week 0 — FLA smoke (setup)

Date: 2026-09-17
Decision: **GO A** (Week 0 exit passed). Do not start SmolLM2 until Gate A work begins.

## Environment

- Service: Kaggle notebook, accelerator **GPU T4 × 2**, Internet on
- GPU: Tesla T4, CUDA capability **7.5** (never P100)
- Torch: Kaggle preinstall `2.10.0+cu128` (not pip-replaced)
- Notebook: `notebooks/kaggle_week0_smoke.ipynb` against `github.com/Caedral-ai/notrehybrid`

## Exit checks

| Check | Result |
|---|---|
| `FLA OK` | `FLA OK Tesla T4` |
| pytest `tests/test_hybrid_block.py` | `[100%]` (PyTorch/SWIG deprecation warnings only) |
| Dry run | `done step=3` (`step_000001.pt` … `step_000003.pt`) |
| `--resume auto` | loaded step **3**, trained to **21687** |
| Second `--resume auto` | loaded **21687**, trained to **25871** |
| Dummy loss | ~1e-9 (`y²` on random tensors; **not** language-model PPL) |

This is **runtime smoke**, not Gate A and not the collision-cache hypothesis.

## What we built (repo)

- Scaffold: Apache-2.0, `pyproject.toml`, Caedral README, `CONTRIBUTING.md` (Conventional Commits)
- `notre/layers/hybrid_block.py`: 3× FLA `GatedDeltaNet` + dummy softmax
- `notre/train/smoke_resume.py`: fp16 + GradScaler, `--resume auto`
- `setup_kaggle.sh`: keep Kaggle CUDA torch; clone lolcats / Taylor-Calibrate / zoology; **do not pip-install zoology**
- Private Hub script: `scripts/create_hf_repo.sh` (not created this week; optional)

## Incidents (first Kaggle session)

1. **`causal_conv1d` wheel build failed.** Pip tried to compile a CUDA extension (no `nvcc`, no wheel for Torch 2.10). Pulled in via `pip install zoology`. Fix: clone zoology only; install `flash-linear-attention[cuda]`; FLA short-conv uses Triton fallback.
2. **Disk full / truncated `.pt`.** Saving every 20 steps filled `/kaggle/working` (~57 MB × hundreds of files). Last resume loaded a corrupt zip. First session still proved resume (step 1 → 4620). Fix: atomic save, `--save-every 100`, `--keep-last 2`, skip truncated files.
3. **Dry-run stopped at step 1** once: first Triton compile exceeded `--minutes 1`. Fix: `--max-steps` ignores the wall clock.

Clean re-run (same day) completed all cells without disk errors.

## Out of Week 0

SmolLM2, Qwen, collision cache, MQAR training, Taylor-Calibrate *runs*, private Hub weights.

## Next

Gate A — [gate-A.md](gate-A.md): **passed 2026-09-24** (GO B). 3:1 surgery, Taylor-Calibrate, transfer probe, **no cache**.

Gate B — [gate-B.md](gate-B.md): **ended 2026-09-25**. The collision cache did not reduce transfer MSE. No Gate C.
