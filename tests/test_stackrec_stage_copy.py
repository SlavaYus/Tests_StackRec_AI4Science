"""
Проверка копирования весов между этапами StackRec.

Что проверяется:
  1. В конфиге этап 2 копирует слои 4->2 и 5->3, этап 3 — 2->0 и 3->1.
  2. В рантайме после этапа 2 слои 2,3 действительно равны слоям 4,5
     (веса предыдущего этапа).
  3. После этапа 3 слои 0,1 действительно равны слоям 2,3, то есть веса
     копируются именно с предыдущего этапа, а не с исходных слоёв 4,5.

Запуск:
    python3 tests/test_stackrec_stage_copy.py
    pytest -q tests/test_stackrec_stage_copy.py
"""

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_stackrec import make_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "beauty_sasrec_stackrec.yaml"


def _copy_pairs(stage):
    return [[int(src), int(dst)] for src, dst in stage.get("copy_layers", [])]


def test_config_copy_pairs():
    cfg = OmegaConf.load(CONFIG_PATH)
    stages = cfg.stackrec.stages

    assert len(stages) == 3
    assert _copy_pairs(stages[0]) == []
    assert _copy_pairs(stages[1]) == [[4, 2], [5, 3]]
    assert _copy_pairs(stages[2]) == [[2, 0], [3, 1]]


def _state(layer):
    return {key: value.clone() for key, value in layer.state_dict().items()}


def _assert_layers_equal(model, src_idx, dst_idx):
    src = _state(model._encoder.layers[src_idx])
    dst = _state(model._encoder.layers[dst_idx])
    assert src.keys() == dst.keys()
    for key in src:
        assert torch.allclose(src[key], dst[key]), (
            f"layer {dst_idx} != layer {src_idx} for {key}"
        )


def test_stage_2_and_3_copy_from_previous_stage():
    cfg = OmegaConf.load(CONFIG_PATH)
    stages = cfg.stackrec.stages
    model = make_model(num_layers=6)

    # Делаем все слои различимыми.
    with torch.no_grad():
        for idx, layer in enumerate(model._encoder.layers):
            for p in layer.parameters():
                p.add_(float(idx + 1))

    # Этап 1: слои 4,5 — "предыдущий этап". Этап 2 копирует их в 2,3.
    for src, dst in _copy_pairs(stages[1]):
        model.copy_layer_weights(src, dst)
    _assert_layers_equal(model, 4, 2)
    _assert_layers_equal(model, 5, 3)

    # Имитируем обучение на этапе 2: слои 2,3 изменились.
    with torch.no_grad():
        for idx in (2, 3):
            for p in model._encoder.layers[idx].parameters():
                p.add_(100.0)

    # Этап 3 должен копировать уже обновлённые слои 2,3 в 0,1.
    for src, dst in _copy_pairs(stages[2]):
        model.copy_layer_weights(src, dst)
    _assert_layers_equal(model, 2, 0)
    _assert_layers_equal(model, 3, 1)

    # Убеждаемся, что скопировано именно с предыдущего этапа (2,3),
    # а не со слоёв 4,5 первого этапа.
    with torch.no_grad():
        assert not torch.allclose(
            model._encoder.layers[0].self_attn.in_proj_weight,
            model._encoder.layers[4].self_attn.in_proj_weight,
        )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main(["-q", "-s", __file__]))
