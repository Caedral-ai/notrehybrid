from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from notre.convert.surgery import SMOLLM2_LAYERS, keep_softmax_layers

ROOT = Path(__file__).resolve().parents[1]


def test_keep_softmax_every_fourth() -> None:
    keep = keep_softmax_layers(SMOLLM2_LAYERS)
    assert keep == [0, 4, 8, 12, 16, 20, 24, 28]
    assert len(keep) == 8
    assert len(keep) * 4 == SMOLLM2_LAYERS


def test_qwen_ratio_also_divisible() -> None:
    keep = keep_softmax_layers(24)
    assert keep == [0, 4, 8, 12, 16, 20]
    assert len(keep) == 6


def test_reject_non_multiple_of_four() -> None:
    with pytest.raises(ValueError, match="divisible by 4"):
        keep_softmax_layers(30)


def test_gate_a_yaml_matches_surgery() -> None:
    cfg = yaml.safe_load((ROOT / "configs/smollm2_360m/gate_a.yaml").read_text())
    assert cfg["student_model"]["name"] == "gdn_v4"
    assert cfg["student_model"]["keep_full_attention_layers"] == keep_softmax_layers()
    assert cfg["train"]["target_tokens"] == 5000000
    assert cfg["train"]["quantize_frozen"] is False


def test_rope_theta_from_parameters() -> None:
    from notre.convert.paths import rope_theta_from

    class Legacy:
        rope_theta = 10000.0

    class Current:
        rope_parameters = {"rope_theta": 10000.0, "rope_type": "default"}

    class Missing:
        pass

    assert rope_theta_from(Legacy()) == 10000.0
    assert rope_theta_from(Current()) == 10000.0
    assert rope_theta_from(Missing(), default=100000.0) == 100000.0


def test_coerce_tied_keys_list_to_dict() -> None:
    from notre.convert.paths import coerce_tied_keys

    class Model:
        _tied_weights_keys = ["lm_head.weight"]

    coerce_tied_keys(Model)
    assert Model._tied_weights_keys == {"lm_head.weight": "model.embeddings.weight"}


def test_sdpa_matches_causal_attention() -> None:
    torch = pytest.importorskip("torch")
    from notre.convert.flash_attn_sdpa import flash_attn_func

    torch.manual_seed(0)
    q = torch.randn(2, 5, 4, 8)
    k = torch.randn(2, 5, 4, 8)
    v = torch.randn(2, 5, 4, 8)
    out = flash_attn_func(q, k, v, causal=True)
    scale = 8 ** -0.5
    scores = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-1, -2)) * scale
    mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    ref = torch.matmul(torch.softmax(scores, dim=-1), v.transpose(1, 2)).transpose(1, 2)
    assert out.shape == q.shape
    assert torch.allclose(out, ref, atol=1e-5)


def test_taylor_root_requires_clone(tmp_path, monkeypatch) -> None:
    from notre.convert import paths

    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="setup_kaggle"):
        paths.taylor_root()
