# Gate A — SmolLM2 transfer→GDN

Date: 2026-09-24
Decision: **GO B**

Zero-shot WikiText-2 PPL: copy-only = **1156.37** | Taylor-calibrated = **345.87**
Transfer MSE (no cache): initial = **1.555** (step 1) | final = **0.798** (step 1740) | stable? **yes** (noisy band ~0.7–0.85)

Calibrated PPL is finite and well below copy-only. MSE fell. No NaN. That is the pass bar. The collision cache was not trained.

## Environment

- Service: Kaggle notebook, accelerator **GPU T4 × 2**, Internet on, 6-hour cap
- GPU: Tesla T4 (never P100, never bf16)
- Torch: Kaggle preinstall `2.10.0+cu128`
- Notebook: `notebooks/kaggle_gate_a.ipynb` against `github.com/Caedral-ai/notrehybrid`
- Deciding run: kernel `tyoungnoone/notrehybrid-gate-a` **version 8** (2026-09-24)
- Model: SmolLM2-360M, 32 layers, 3:1 hybrid. Softmax kept every 4th layer `[0,4,8,12,16,20,24,28]` (8 softmax + 24 GDN)

## What ran (version 8)

| Step | Result |
|---|---|
| FLA teacher | saved |
| Copy-only student | saved |
| Copy-only WikiText-2 PPL | **1156.37** (fp32) |
| Taylor-Calibrate | all 24 GDN layers, finite losses |
| Taylor WikiText-2 PPL | **345.87** (fp32) |
| Transfer dry run | `done step=3`, finite MSE |
| 20-minute transfer probe | `done step=1748` tokens=1,789,952 elapsed=1210s |
| MSE | 1.555 → 0.798, no NaN |
| Full ~5M tokens | not run |

The probe is the measurement. The full 5M-token transfer waits; it is not required to pass Gate A once MSE is finite and falling.

## What we built (repo)

- `notre/convert/surgery.py`: 3:1 keep-every-4th map
- `notre/convert/convert_smollm2.py`: HF Llama → FLA teacher, fp16
- `notre/convert/init_student.py`: copy-only student, fp32
- `notre/convert/taylor_calibrate.py`: their `apply_taylor_calibrate`, student built in fp32
- `notre/convert/transfer.py`: FineWeb-Edu attention-transfer MSE, no cache. Frozen teacher fp16, trainable GDN fp32, GDN forward outside autocast, GradScaler on the fp32 weights
- `notre/eval/ppl.py`: WikiText-2 PPL loaded in fp32
- `notre/convert/flash_attn_sdpa.py`: SDPA stand-in. Kaggle has no `nvcc`, so flash-attn is not built
- `configs/smollm2_360m/gate_a.yaml`: target 5M tokens, batch 2, micro-batch 1, seq 1024

## Incidents

Versions 1–7 did not produce a usable MSE curve. Version 8 did.

1. **flash-attn.** FLA attention requires `flash_attn_func`. No compiler on the T4 image. Fix: SDPA stand-in registered before `import fla`, with a real module spec.
2. **Tied weights.** Current Transformers expects `_tied_weights_keys` to be a dict. FLA ships a list. Fix: `coerce_tied_keys`.
3. **RoPE theta.** SmolLM2 on current Transformers stores it under `rope_parameters`, not `config.rope_theta`. Fix: `rope_theta_from`.
4. **fp16 Taylor init NaN.** Phase 2 AdamW in fp16 overflowed. Step-0 loss was finite; later steps and both PPLs were NaN. Their loop can save the post-step weights. Fix: build the student and run Taylor and PPL in fp32. Version 7 then reported copy-only PPL 1056 and Taylor PPL 349, both finite.
5. **GradScaler.** Version 7 still cast the trainable GDN to fp16, then died with `Attempting to unscale FP16 gradients` before any MSE. Fix: teacher stays fp16, student weights stay fp32 and run outside autocast.

## Out of Gate A

Collision cache, MQAR, Qwen, the full 5M-token transfer, LoLCATs as a retry.

## Next

Gate B — [gate-B.md](gate-B.md): collision cache vs an identical twin. ΔMSE decides. Do not start it from this note alone.
