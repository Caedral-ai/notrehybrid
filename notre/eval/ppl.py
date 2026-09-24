"""WikiText-2 PPL. fp16. Works for FLA teacher, student, or HF Llama."""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from notre.convert.paths import assert_t4_or_newer, register_hf_classes


def load_model(ckpt: str, device: torch.device):
    register_hf_classes()
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32)
    return model.to(device).eval()


@torch.no_grad()
def wikitext2_ppl(model, tokenizer, seqlen: int, device: torch.device) -> float:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    n_chunks = input_ids.size(1) // seqlen
    total_nll = 0.0
    total_tokens = 0
    for i in range(n_chunks):
        chunk = input_ids[:, i * seqlen : (i + 1) * seqlen]
        logits = model(chunk).logits[:, :-1, :].contiguous()
        targets = chunk[:, 1:].contiguous()
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            targets.view(-1),
            reduction="sum",
        )
        total_nll += float(loss)
        total_tokens += targets.numel()
    return math.exp(total_nll / max(total_tokens, 1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--seqlen", type=int, default=1024)
    args = p.parse_args()
    assert_t4_or_newer()
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.ckpt)
    model = load_model(args.ckpt, device)
    ppl = wikitext2_ppl(model, tok, args.seqlen, device)
    print(f"WikiText-2 PPL = {ppl:.2f}")


if __name__ == "__main__":
    main()
