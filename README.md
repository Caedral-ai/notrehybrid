# NotreHybrid

**The product is the collision cache, not the converted model.** Linearize a small Transformer (LoLCATs + Taylor-Calibrate → Gated DeltaNet, 3:1) and test whether a trainable K-slot buffer, triggered by GDN delta error `err_t`, reduces attention-transfer MSE. If Gate B fails, the project ends.

Public funnel: [docs/PLAN.md](docs/PLAN.md). The long-form plan is local-only under `internal-docs/` (gitignored).

Week 0 is setup only: FLA on Kaggle T4×2, a dummy 3:1 block, checkpoint resume. No SmolLM2, no Qwen, no cache.

## Week 0 smoke (Kaggle T4×2, never P100)

```bash
# GPU runtime: T4×2. fp16. Never bf16 on T4.
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

# ~10 min train, then kill the session and re-run with --resume auto
python -m notre.train.smoke_resume --minutes 10 --ckpt-dir ./checkpoints/week0
python -m notre.train.smoke_resume --minutes 10 --ckpt-dir ./checkpoints/week0 --resume auto
```

Launch notebook: [notebooks/kaggle_week0_smoke.ipynb](notebooks/kaggle_week0_smoke.ipynb).

Private Hub (after `huggingface-cli login`):

```bash
bash scripts/create_hf_repo.sh
```

## Out of Week 0

SmolLM2 / Qwen, collision cache, MQAR, FoX, Rust, OpenVINO, tok/s vs llama.cpp.

## License

Apache-2.0.
