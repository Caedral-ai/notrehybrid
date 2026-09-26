# Contributing to NotreHybrid

Rules for this repository. They exist to protect the only claim this project
was allowed to make: **transfer ΔMSE with the collision cache**, under the
gates in [docs/PLAN.md](docs/PLAN.md). This is Caedral **research**, not a
product to ship.

**The project ended 2026-09-25.** Gate B failed. Mean transfer MSE with the
cache was higher than the twin. Do not open Gate C, a tech report, or a
paper draft. The log is [docs/decisions/gate-B.md](docs/decisions/gate-B.md).

Same commit language as [caedral-notre-engine](https://github.com/Caedral-ai/caedral-notre-engine/blob/main/CONTRIBUTING.md).

---

## 1. Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <imperative summary, max 72 chars>

[optional body: wrap at ~100 cols, explain WHY not WHAT]

[optional footer: BREAKING CHANGE:, Refs:, Gate:]
```

**Types**

| Type | Use for |
|---|---|
| `feat` | new method code (cache, surgery, transfer loop) |
| `fix` | bug fixes |
| `docs` | README, PLAN, gate write-ups |
| `test` | tests only |
| `refactor` | no behavior change |
| `chore` | tooling, housekeeping, metadata |
| `ci` | CI pipelines (when they exist) |

**Scopes**: `layers`, `train`, `eval`, `convert`, `docs`, `kaggle`.

Examples:

```
feat(layers): train K=32 ring on err_t during transfer
docs(plan): record Gate B ΔMSE table
fix(train): keep GDN state in fp32 under GradScaler
chore: set Caedral package metadata
```

**Rules**

- One logical change per commit. Never mix a refactor with a behavior change.
- Never rewrite published history on `main`.
- Experiment commits that close a gate use `Gate: A` / `Gate: B` / `Gate: C`
  in the footer.
- Do not commit weights, `.pt` dumps, Hub tokens, or FineWeb on disk.

## 2. Branches

```
feat/<topic>   fix/<topic>   docs/<topic>   exp/<gate>-<topic>
```

Keep branches short-lived; rebase onto `main` before merge. Squash-merge WIP.

Do not open a branch to continue the cache, Qwen, or a write-up. Gate B is the end.

## 3. Engineering invariants

Violating any of these is out of scope, not a style nit:

1. **The cache is the experiment.** Converted SmolLM2 / Qwen are vehicles.
2. **Kill hard at Gate B.** That rule fired on 2026-09-25. No tech report.
3. **Do not continue past B.** No Qwen, no MQAR follow-up, no paper scaffolding.
4. **T4 or newer.** Never P100. Training is fp16 + GradScaler; GDN state fp32.
   Never bf16 on T4.
5. **No secrets, no weights** in git. Checkpoints go to the private Hub.
6. **Do not add the unrun follow-ups** (`eval/mqar.py`, trigger ablations,
   Qwen conversion). Gate B did not earn them.

## 4. Code

- Python 3.10+. Comments explain *why*; English only.
- Public schedule lives in `docs/PLAN.md`. The long-form plan stays in
  local `internal-docs/` (gitignored).
- [docs/decisions/gate-A.md](docs/decisions/gate-A.md) and
  [gate-B.md](docs/decisions/gate-B.md) are filled. Do not reopen them
  with a new decision.

Apache-2.0 — see [LICENSE](LICENSE). A project of [Caedral](https://caedral.com).
