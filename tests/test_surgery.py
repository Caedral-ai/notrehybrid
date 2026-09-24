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


def test_taylor_root_requires_clone(tmp_path, monkeypatch) -> None:
    from notre.convert import paths

    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="setup_kaggle"):
        paths.taylor_root()
