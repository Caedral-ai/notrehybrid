"""10-minute train / kill / --resume auto smoke. Local files; Hub comes after login."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast

from notre.layers.hybrid_block import HybridBlock


def assert_t4_or_newer() -> str:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required. Use Kaggle T4×2.")
    major, _ = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    if major < 7:
        raise SystemExit(f"Refuse {name} (capability {major}). Never P100.")
    return name


def latest_ckpt(ckpt_dir: Path) -> Path | None:
    files = sorted(ckpt_dir.glob("step_*.pt"))
    return files[-1] if files else None


def log_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss"])
        if new:
            w.writeheader()
        w.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 0 HybridBlock resume smoke")
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/week0"))
    p.add_argument("--resume", choices=["none", "auto"], default="none")
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--max-steps", type=int, default=0, help="0 = time-limited only")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    name = assert_t4_or_newer()
    device = torch.device("cuda")
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.ckpt_dir / "curve.csv"

    model = HybridBlock(
        hidden_size=args.hidden, num_heads=args.heads, head_dim=64, mode="chunk"
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    step = 0

    if args.resume == "auto":
        ckpt = latest_ckpt(args.ckpt_dir)
        if ckpt is None:
            print("resume auto: no checkpoint, starting fresh")
        else:
            payload = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(payload["model"])
            opt.load_state_dict(payload["opt"])
            scaler.load_state_dict(payload["scaler"])
            step = int(payload["step"])
            print(f"resume auto: loaded {ckpt} at step {step}")

    print(f"GPU {name} fp16+GradScaler  start_step={step}")
    deadline = time.time() + args.minutes * 60.0
    model.train()

    while time.time() < deadline:
        if args.max_steps and step >= args.max_steps:
            break
        x = torch.randn(args.batch, args.seq, args.hidden, device=device)
        opt.zero_grad(set_to_none=True)
        with autocast(dtype=torch.float16):
            y = model(x)
            loss = y.float().pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        step += 1
        log_csv(csv_path, {"step": step, "loss": float(loss.detach().cpu())})
        if step % args.save_every == 0:
            path = args.ckpt_dir / f"step_{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                path,
            )
            print(f"saved {path} loss={float(loss):.6f}")

    path = args.ckpt_dir / f"step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
        },
        path,
    )
    print(f"done step={step} last={path}")


if __name__ == "__main__":
    main()
