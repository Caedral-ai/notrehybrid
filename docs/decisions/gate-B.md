# Gate B — collision cache (KILL HARD)

Date: 2026-09-25
Decision: **END OF PROJECT**

Mean ΔMSE at matched step 500: **−1.34% to −3.02%** on the seven measured cells. Best cell: K=8, τ=0.3. Every cell is negative.
Layers with ΔMSE>0: **not recovered** (`mse_layers.csv` stayed on the Kaggle VM)
MQAR: not run
Triggers: not run (required only if ΔMSE > 0)

The cache made mean transfer MSE worse than the identical twin. A few K=8 arms were slightly better at steps 100 and 200, then worse by step 500. That is not stable, and it is not a pass. K=64 at τ=0.5 and τ=0.7 were not run. At K=8 and K=32, a higher τ was worse, so those two cells were stopped. No tech report. No Qwen. No Gate C.

## Definition

`ΔMSE = (MSE_without − MSE_with) / MSE_without` on the mean across converted layers. Positive would mean the cache imitates the teacher more closely. The logged `mse` is that mean (`torch.stack(per_layer).mean()`). Per-layer ratios were written beside it and were not copied off the VM.

Same seed (0), same Taylor init, same trainable parameters. The cache has no parameters. The twin is the same student with the ring off. Compare matched steps. A 15-minute arm ends near step 580; the twin’s later steps are not the comparison.

## Environment

- Service: Kaggle notebook, accelerator **NvidiaTeslaT4**, Internet on, 6-hour cap
- GPU: Tesla T4 (never P100, never bf16)
- Training: fp16 + GradScaler. Trainable student and GDN state stay fp32. Frozen teacher stays fp16. GDN forward runs outside autocast
- Notebook: `notebooks/kaggle_gate_b.ipynb` against `github.com/Caedral-ai/notrehybrid` at `95172d8`
- Deciding run: kernel `tyoungnoone/notrehybrid-gate-b` **version 16** (2026-09-25)
- Model: SmolLM2-360M, same 3:1 map as Gate A. Softmax kept at `[0,4,8,12,16,20,24,28]` (8 softmax + 24 GDN)
- Data: FineWeb-Edu, seq 1024, seed 0

## Twin (no cache)

20-minute probe, ring off.

| Step | MSE |
|---|---|
| 1 | 1.55507 |
| 100 | 1.19778 |
| 200 | 1.10245 |
| 500 | **0.94699** |
| 650 | 0.87326 |
| done | step 1678, tokens 1,718,272, elapsed 1210s |

## Measured cells

ΔMSE uses the twin row above. The K=32, τ=0.5 cell is an earlier 20-minute probe (Triton scan). The other cells are 15-minute arms on version 16. K=32, τ=0.5 was not repeated.

| K | τ | Steps | MSE @ 100 | Δ @ 100 | MSE @ 200 | Δ @ 200 | MSE @ 500 | Δ @ 500 |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.3 | 560 | 1.19124 | +0.55% | 1.09284 | +0.87% | 0.95972 | **−1.34%** |
| 8 | 0.5 | 581 | 1.19203 | +0.48% | 1.09410 | +0.76% | 0.96274 | **−1.66%** |
| 8 | 0.7 | 583 | 1.19302 | +0.40% | 1.09604 | +0.58% | 0.96713 | **−2.13%** |
| 32 | 0.3 | 582 | 1.19854 | −0.06% | 1.10316 | −0.06% | 0.96978 | **−2.41%** |
| 32 | 0.5 | 651 | 1.19904 | −0.11% | 1.10397 | −0.14% | 0.97057 | **−2.49%** |
| 32 | 0.7 | 583 | 1.20006 | −0.19% | 1.10573 | −0.30% | 0.97557 | **−3.02%** |
| 64 | 0.3 | 582 | 1.20104 | −0.27% | 1.10547 | −0.27% | 0.97235 | **−2.68%** |
| 64 | 0.5 | — | | | | | | not run |
| 64 | 0.7 | — | | | | | | not run |

K=32, τ=0.5 at step 650: cache 0.89297 vs twin 0.87326, **−2.26%**. The sign does not flip later in that longer probe.

Larger K and higher τ made the step-500 gap worse. The early K=8 gains were gone by the decision step.

## What we built (repo)

- `notre/layers/notre_linear.py`: ring per KV head, `err_t` push (decision detached), softmax read-out added through `o_proj`. No cache parameters
- `notre/layers/cache_scan.py`: Triton delta-error scan on CUDA. Reference recurrence stays `gated_delta_scan` (fp32). Read-out stays in PyTorch
- `notre/convert/transfer.py`: `--cache`, `--slots`, `--tau`. Without `--cache` the ring is off
- `notebooks/kaggle_gate_b.ipynb`: convert, Taylor-Calibrate, then the K/τ arms
- `tests/test_notre_linear.py`: CPU tests of the recurrence and the ring

## Incidents

Versions 1–15 did not finish the grid. Version 16 did, through K=64, τ=0.3.

The failures that mattered were process hangs, not a bad MSE. `%%bash` cells never started. Hugging Face `from_pretrained` stalled while materializing weights. The teacher and the FLA student had to be built from config and loaded from local safetensors. Version 15 then died on a missing `AutoConfig` import (`95172d8`). Version 16 is the curve above.

The default K=32, τ=0.5 probe was already negative before that sweep. One cell is not enough to end the project. The sweep is.

## Out of Gate B

MQAR, trigger ablations (`err_t` vs LoLA vs teacher top-K vs random), the two unrun K=64 cells, the full ~5M-token transfer, per-layer counts, Qwen, a tech report.

## Next

None. The project ends here.
