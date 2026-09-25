"""Gate A attention-transfer MSE. FineWeb-Edu streaming, no cache.

Frozen teacher stays fp16. Trainable GDN weights stay fp32 so GradScaler can
unscale, and the GDN forward runs outside autocast. The loss is fp16 autocast
on the teacher path only. Their Stage-1 trainer is bf16 / 8-GPU / DeepSpeed;
this loop is the T4 substitute.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from transformers import AutoModelForCausalLM, AutoTokenizer

from notre.convert.paths import assert_t4_or_newer, register_hf_classes
from notre.convert.surgery import keep_softmax_layers
from notre.layers.notre_linear import CollisionCache, apply_collision_cache
from notre.train.smoke_resume import latest_ckpt, save_ckpt


_DISTILL_LOSSES: list[tuple[int, torch.Tensor]] = []
_COLLECTING = False


class AttentionDistillationWrapper(nn.Module):
    """Teacher-forced per-layer MSE. Residual stays the teacher hidden."""

    def __init__(
        self,
        teacher_attn,
        student_cls,
        config,
        layer_idx: int,
        cache: CollisionCache | None = None,
    ):
        super().__init__()
        self.teacher_attn = teacher_attn.eval()
        for p in self.teacher_attn.parameters():
            p.requires_grad_(False)
        self.student_attn = student_cls(config, layer_idx)
        self.student_attn.init_from_teacher(self.teacher_attn)
        self.layer_idx = layer_idx
        self.cache = cache

    def forward(self, *args, **kwargs):
        kwargs["output_attentions"] = False
        kwargs["use_cache"] = False
        with torch.no_grad():
            t_hidden, _, _ = self.teacher_attn(*args, **kwargs)
        # GDN state stays fp32. Autocast would cast the recurrent part to fp16.
        student_args = tuple(
            a.float() if torch.is_tensor(a) and a.is_floating_point() else a for a in args
        )
        student_kwargs = {
            k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in kwargs.items()
        }
        hidden = student_args[0] if student_args else student_kwargs.get("hidden_states")
        with autocast("cuda", enabled=False):
            s_hidden, _, _ = self.student_attn(*student_args, **student_kwargs)
            if self.cache is not None and hidden is not None:
                s_hidden = apply_collision_cache(
                    self.student_attn, hidden, s_hidden, self.cache, enabled=True
                )
        distill_loss = torch.linalg.vector_norm(
            t_hidden.float() - s_hidden.float(), dim=-1
        ).mean() * (t_hidden.size(-1) ** -0.5)
        if _COLLECTING:
            _DISTILL_LOSSES.append((self.layer_idx, distill_loss))
        return t_hidden, None, None


def _load_student_sd(ckpt_dir: Path) -> dict | None:
    from safetensors.torch import load_file

    single = ckpt_dir / "model.safetensors"
    if single.exists():
        return load_file(str(single))
    shards = sorted(ckpt_dir.glob("model-*.safetensors"))
    if not shards:
        return None
    out: dict = {}
    for shard in shards:
        out.update(load_file(str(shard)))
    return out


def build_wrapped_teacher(
    cfg: dict,
    device: torch.device,
    cache: CollisionCache | None = None,
) -> tuple[nn.Module, list[int]]:
    register_hf_classes()
    from distill_model.modeling_distilled_student import get_student_attention_class

    teacher_name = cfg["teacher_model"]["name"]
    keep_layers = list(
        cfg["student_model"].get("keep_full_attention_layers", keep_softmax_layers())
    )
    student_cls = get_student_attention_class(cfg["student_model"]["name"])
    model = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=torch.float16
    )
    model.config.use_cache = False
    for idx, layer in enumerate(model.model.layers):
        if idx in keep_layers:
            for p in layer.attn.parameters():
                p.requires_grad_(False)
            continue
        layer.attn = AttentionDistillationWrapper(
            layer.attn, student_cls, model.config, idx, cache=cache
        )

    # Teacher weights are fp16. Student weights are loaded after that cast so the
    # fp32 Taylor checkpoint is not rounded away, then left in fp32 for GradScaler.
    model = model.to(device=device, dtype=torch.float16)
    for layer in model.model.layers:
        student = getattr(getattr(layer, "attn", None), "student_attn", None)
        if student is not None:
            student.float()

    init_dir = cfg["train"].get("student_init_ckpt")
    if init_dir:
        init_path = Path(init_dir)
        sd = _load_student_sd(init_path)
        if sd is None:
            print(f"no student weights at {init_path}; using init_from_teacher")
        else:
            loaded = 0
            for idx, layer in enumerate(model.model.layers):
                if idx in keep_layers or not hasattr(layer.attn, "student_attn"):
                    continue
                prefix = f"model.layers.{idx}.attn."
                piece = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
                if piece:
                    layer.attn.student_attn.load_state_dict(piece, strict=False)
                    loaded += 1
            print(f"loaded student_attn for {loaded} GDN layers from {init_path}")

    for name, p in model.named_parameters():
        p.requires_grad_(".student_attn." in name)
        if p.requires_grad and p.dtype != torch.float32:
            raise RuntimeError(f"trainable param {name} is {p.dtype}, want float32")
    return model, keep_layers


def iter_fineweb_chunks(tokenizer, seq_len: int, subset: str, seed: int):
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=subset,
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    buf: list[int] = []
    for row in ds:
        text = row.get("text") or ""
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        buf.extend(ids)
        while len(buf) >= seq_len:
            chunk, buf = buf[:seq_len], buf[seq_len:]
            yield torch.tensor(chunk, dtype=torch.long)


def log_mse(path: Path, row: dict, layers: dict[int, float] | None = None) -> None:
    new = not path.exists()
    fields = ["step", "tokens", "mse"]
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in fields})
    if not layers:
        return
    layer_path = path.with_name("mse_layers.csv")
    layer_new = not layer_path.exists()
    keys = ["step", "tokens", *[f"l{idx}" for idx in sorted(layers)]]
    with layer_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if layer_new:
            w.writeheader()
        payload = {"step": row["step"], "tokens": row["tokens"]}
        payload.update({f"l{idx}": f"{value:.6f}" for idx, value in layers.items()})
        w.writerow(payload)


def export_student(wrapped, init_dir: Path, out_dir: Path, keep_layers: list[int]) -> None:
    register_hf_classes()
    student = AutoModelForCausalLM.from_pretrained(str(init_dir), torch_dtype=torch.float16)
    for idx, layer in enumerate(wrapped.model.layers):
        if idx in keep_layers or not hasattr(layer.attn, "student_attn"):
            continue
        student.model.layers[idx].attn.load_state_dict(
            layer.attn.student_attn.state_dict()
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    student.cpu().save_pretrained(out_dir, safe_serialization=True)
    tok_dir = Path(init_dir)
    if (tok_dir / "tokenizer.json").exists() or (tok_dir / "tokenizer_config.json").exists():
        AutoTokenizer.from_pretrained(str(tok_dir)).save_pretrained(out_dir)
    print(f"exported student → {out_dir}")


def trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def run_transfer(cfg: dict, args: argparse.Namespace) -> None:
    global _COLLECTING
    gpu = assert_t4_or_newer()
    device = torch.device("cuda")
    cache_on = bool(getattr(args, "cache", False))
    slots = int(getattr(args, "slots", 32) or 32)
    tau = float(getattr(args, "tau", 0.5))
    cache = CollisionCache(slots=slots, tau=tau) if cache_on else None
    print(
        f"transfer fp32 student / fp16 teacher on {gpu} — "
        f"cache {'on' if cache_on else 'off'} slots={slots} tau={tau}"
    )

    seq_len = int(cfg["train"]["train_seq_len"])
    micro = int(cfg["train"].get("micro_batch_size", 1))
    target = int(cfg["train"]["target_tokens"])
    lr = float(cfg["train"].get("lr_attn", cfg["train"]["lr"]))
    save_every = int(args.save_every or cfg["train"].get("save_steps", 100))
    keep = int(args.keep_last or cfg["train"].get("save_total_limit", 2))
    subset = cfg.get("data", {}).get("subset", "sample-10BT")
    run_name = "transfer-cache" if cache_on else "transfer"
    ckpt_dir = Path(args.ckpt_dir or cfg["train"]["output_dir"]) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ckpt_dir / "mse.csv"

    tokenizer = AutoTokenizer.from_pretrained(cfg["teacher_model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, keep_layers = build_wrapped_teacher(cfg, device, cache=cache)
    params = trainable_params(model)
    try:
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), fused=True)
    except TypeError:
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95))
    scaler = GradScaler("cuda")

    step = 0
    tokens = 0
    if args.resume == "auto":
        found = latest_ckpt(ckpt_dir, device)
        if found is not None:
            path, payload = found
            result = model.load_state_dict(payload["student"], strict=False)
            student_missing = [k for k in result.missing_keys if ".student_attn." in k]
            print("resume", path, "student_attn missing", len(student_missing))
            opt.load_state_dict(payload["opt"])
            scaler.load_state_dict(payload["scaler"])
            step = int(payload["step"])
            tokens = int(payload["tokens"])
            print(f"resumed step={step} tokens={tokens}")

    deadline = time.time() + args.minutes * 60 if args.minutes > 0 else None
    skipped = 0
    model.train()
    t0 = time.time()
    if tokens >= target and not args.max_steps:
        print(f"already at target_tokens={target}")
    else:
        for chunk in iter_fineweb_chunks(tokenizer, seq_len, subset, seed=0):
            if skipped < tokens:
                skipped += seq_len
                continue
            if args.max_steps and (step >= args.max_steps):
                break
            if deadline is not None and time.time() >= deadline and not args.max_steps:
                break
            if tokens >= target:
                break

            ids = chunk.unsqueeze(0).to(device)
            if micro != 1:
                raise SystemExit("micro_batch_size > 1 not implemented; pack later")

            opt.zero_grad(set_to_none=True)
            _DISTILL_LOSSES.clear()
            _COLLECTING = True
            with autocast("cuda", dtype=torch.float16):
                model(input_ids=ids, use_cache=False)
            _COLLECTING = False
            if not _DISTILL_LOSSES:
                raise RuntimeError("wrapper collected no layer MSE")
            losses = [item[1] for item in _DISTILL_LOSSES]
            layer_mse = {idx: float(item.detach()) for idx, item in _DISTILL_LOSSES}
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise SystemExit(f"NaN/Inf MSE at step {step}: {loss}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()

            step += 1
            tokens += seq_len
            mse = float(loss.detach())
            log_mse(csv_path, {"step": step, "tokens": tokens, "mse": f"{mse:.6f}"}, layer_mse)
            if step == 1 or step % 10 == 0:
                print(f"step {step} tokens={tokens} mse={mse:.5f}")

            if step % save_every == 0:
                student = {k: v.detach().cpu() for k, v in model.state_dict().items() if ".student_attn." in k}
                save_ckpt(
                    ckpt_dir / f"step_{step:08d}.pt",
                    {
                        "student": student,
                        "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(),
                        "step": step,
                        "tokens": tokens,
                    },
                    ckpt_dir,
                    keep,
                )

    student = {k: v.detach().cpu() for k, v in model.state_dict().items() if ".student_attn." in k}
    save_ckpt(
        ckpt_dir / f"step_{step:08d}.pt",
        {"student": student, "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step, "tokens": tokens},
        ckpt_dir,
        keep,
    )
    print(f"done step={step} tokens={tokens} elapsed={time.time() - t0:.0f}s → {csv_path}")

    if tokens >= target and cfg["train"].get("student_init_ckpt"):
        export_student(
            model,
            Path(cfg["train"]["student_init_ckpt"]),
            ckpt_dir / "student-final",
            keep_layers,
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    # The FineWeb stream thread aborts the interpreter during normal shutdown.
    os._exit(0)


def main() -> None:
    p = argparse.ArgumentParser(description="Gate A transfer MSE (no cache)")
    p.add_argument("--cfg", type=Path, required=True)
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--resume", choices=["none", "auto"], default="none")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--minutes", type=float, default=0.0, help="0 = run to target_tokens")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--keep-last", type=int, default=0)
    p.add_argument("--teacher", type=Path, default=None)
    p.add_argument("--student-init", type=Path, default=None)
    p.add_argument("--cache", action="store_true", help="enable the err_t collision cache")
    p.add_argument("--slots", type=int, default=32)
    p.add_argument("--tau", type=float, default=0.5)
    args = p.parse_args()
    cfg = yaml.safe_load(args.cfg.read_text())
    if args.teacher:
        cfg["teacher_model"]["name"] = str(args.teacher)
    if args.student_init:
        cfg["train"]["student_init_ckpt"] = str(args.student_init)
    run_transfer(cfg, args)


if __name__ == "__main__":
    main()
