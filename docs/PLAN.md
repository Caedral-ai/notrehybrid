# Execution funnel

This repo follows the local-only long-form plan in `internal-docs/` (gitignored; not on remotes). Do not skip gates. The public schedule is this funnel.

| When | Work | Kill rule |
|------|------|-----------|
| **Week 0** (now) | FLA smoke, dummy 3:1 `HybridBlock`, local `--resume auto` | Change GPU if not T4-class. Never P100. |
| Week 1 | Gate A: SmolLM2 surgery + Taylor-Calibrate + transfer, **no cache** | PPL still thousands or MSE does not fall → abandon |
| Weeks 2–3 | Gate B: cache vs twin; ΔMSE decides; MQAR support | ΔMSE ≤ 0 after K/τ sweep → **end project** |
| Weeks 4–6 | Gate C only if B passed | MSE fail → pilot-only paper |

Decisions: [decisions/gate-A.md](decisions/gate-A.md), [decisions/gate-B.md](decisions/gate-B.md).

**Product:** trainable collision cache. **Vehicle:** converted Qwen/SmolLM2. CPU decode, Rust, FoX/ACP, Notre-1B are out of MVP.
