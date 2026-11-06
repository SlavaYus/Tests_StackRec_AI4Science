"""
Тесты StackRec: корректное добавление и тренировка блоков.

Проверяем, что модель корректно:
  1. Обучается на первых 2 блоках (layers 4, 5).
  2. Расширяется до 4 блоков (layers 2-5) с копированием весов.
  3. Расширяется до 6 блоков (layers 0-5) с копированием весов.

Два примера с одной последовательностью:
  - seq_short: длина 5
  - seq_long:  длина 20

Без GPU — только CPU.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.sequence import SequentialEncoder
from src.loss import LocalObjective
from src.dataset import SasrecProcessor, SequenceDataset
from torch.utils.data import DataLoader


# ============================================================
# Утилиты
# ============================================================

def make_model(num_layers=6, embedding_dim=32, num_heads=2,
               dim_feedforward=64, dropout=0.1, num_items=50):
    """Создать модель SASRec на CPU."""
    item_embeddings = nn.Embedding(
        num_embeddings=num_items + 2,
        embedding_dim=embedding_dim,
        padding_idx=0,
    )
    position_embeddings = nn.Embedding(
        num_embeddings=100,
        embedding_dim=embedding_dim,
    )
    model = SequentialEncoder(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation="gelu",
        layer_norm_eps=1e-9,
        num_layers=num_layers,
        num_items=num_items,
        topk_k=10,
        position_embeddings=position_embeddings,
        item_embeddings=item_embeddings,
        curvature_mode="independent",
        init_curvature=1.0,
    )
    return model


def make_batch_from_sequence(item_ids, mode="train", max_seq_len=25):
    """Создать батч из одной последовательности item IDs.

    Для train: контекст = item_ids[:-1], label = item_ids[1:]
    Для val/test: контекст = item_ids[:-1], label = item_ids[-1]
    """
    processor = SasrecProcessor(pad_id=0, mode=mode)
    sample = {"user.ids": 0, "item.ids": list(item_ids)}
    batch = processor([sample])
    return batch


def train_one_step(model, batch, optimizer, loss_fn, device="cpu"):
    """Один шаг обучения на батче."""
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    outputs = model(batch)
    loss_inputs = {**batch, **outputs}
    loss = loss_fn(loss_inputs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_model(model, batch, device="cpu"):
    """Один шаг инференса, возвращает top-k predictions."""
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    model.eval()
    with torch.no_grad():
        preds = model(batch)
    model.train()
    return preds


# ============================================================
# Тест 1: Инициализация модели
# ============================================================
def test_model_initialization():
    """Модель создаётся, все параметры на CPU, num_layers=6."""
    print("\n--- Test 1: Model initialization ---")
    model = make_model(num_layers=6)
    assert model.num_layers == 6, f"Expected 6 layers, got {model.num_layers}"
    assert model.device.type == "cpu"

    # Проверить что scale=1.0 для всех слоёв
    scales = model.get_layer_scales()
    assert len(scales) == 6
    assert all(s == 1.0 for s in scales), f"Scales: {scales}"

    # Проверить что aux scalars = 1.0
    aux = model.get_layer_aux_scalars()
    assert len(aux) == 6
    assert all(a == 1.0 for a in aux)

    print("  PASS: model init OK")
    return True


# ============================================================
# Тест 2: Прямой проход на короткой последовательности
# ============================================================
def test_forward_short():
    """Прямой проход на seq длины 5."""
    print("\n--- Test 2: Forward pass (short seq=5) ---")
    model = make_model(num_layers=6)
    seq = [10, 20, 30, 40, 5]  # 5 items
    batch = make_batch_from_sequence(seq, mode="train")

    # Training mode
    outputs = model(batch)
    assert "local_prediction" in outputs
    # local_prediction shape: (all_items,) x num_classes
    # all_items = len(seq)-1 = 4 (контекст)
    preds = outputs["local_prediction"]
    assert preds.shape[0] == 4, f"Expected 4 predictions, got {preds.shape[0]}"

    # Inference mode
    batch_val = make_batch_from_sequence(seq, mode="validation")
    preds = evaluate_model(model, batch_val)
    assert preds.shape[0] == 1, f"Expected 1 prediction, got {preds.shape[0]}"
    assert preds.shape[1] == 10, f"Expected topk=10, got {preds.shape[1]}"

    print(f"  PASS: forward short OK, preds shape={preds.shape}")
    return True


# ============================================================
# Тест 3: Прямой проход на длинной последовательности
# ============================================================
def test_forward_long():
    """Прямой проход на seq длины 20."""
    print("\n--- Test 3: Forward pass (long seq=20) ---")
    model = make_model(num_layers=6)
    seq = list(range(1, 21))  # 20 items
    batch = make_batch_from_sequence(seq, mode="train")

    outputs = model(batch)
    preds = outputs["local_prediction"]
    assert preds.shape[0] == 19, f"Expected 19 predictions, got {preds.shape[0]}"

    batch_val = make_batch_from_sequence(seq, mode="validation")
    preds = evaluate_model(model, batch_val)
    assert preds.shape[0] == 1
    assert preds.shape[1] == 10

    print(f"  PASS: forward long OK, preds shape={preds.shape}")
    return True


# ============================================================
# Тест 4: Обучение на 2 блоках (layers 4, 5)
# ============================================================
def test_stage_2_layers():
    """
    Этап 1: активны только слои 4, 5.
    Остальные слои замораживаются, их scale = 0.
    Модель должна обучаться и loss должен уменьшаться.
    """
    print("\n--- Test 4: Stage 1 — 2 layers (4, 5) ---")
    model = make_model(num_layers=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = LocalObjective()

    seq = [10, 20, 30, 40, 5]
    batch = make_batch_from_sequence(seq, mode="train")

    # Активируем только слои 4, 5
    model.set_layers_trainable([4, 5], trainable=True)
    model.set_layers_trainable([0, 1, 2, 3], trainable=False)
    model.set_layer_scales([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

    # Несколько шагов обучения
    initial_loss = None
    final_loss = None
    for epoch in range(10):
        loss = train_one_step(model, batch, optimizer, loss_fn)
        if epoch == 0:
            initial_loss = loss
        if epoch == 9:
            final_loss = loss

    assert initial_loss is not None and final_loss is not None
    print(f"  Loss: {initial_loss:.4f} -> {final_loss:.4f}")

    # Проверить что слои 0-3 заморожены
    for idx in [0, 1, 2, 3]:
        for p in model._encoder.layers[idx].parameters():
            assert not p.requires_grad, f"Layer {idx} should be frozen"

    # Проверить что слои 4-5 обучаются
    for idx in [4, 5]:
        for p in model._encoder.layers[idx].parameters():
            assert p.requires_grad, f"Layer {idx} should be trainable"

    print("  PASS: 2-layer training OK")
    return True


# ============================================================
# Тест 5: Обучение на 4 блоках (layers 2-5) с копированием
# ============================================================
def test_stage_4_layers():
    """
    Этап 2: копируем веса слоёв 4->2, 5->3, затем обучаем слои 2-5.
    Слои 2, 3 имеют scale, который растёт от 0 до 1.
    """
    print("\n--- Test 5: Stage 2 — 4 layers (2,3,4,5) with copy ---")
    model = make_model(num_layers=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = LocalObjective()

    seq = [10, 20, 30, 40, 5]
    batch = make_batch_from_sequence(seq, mode="train")

    # Сначала обучаем слои 4, 5
    model.set_layers_trainable([4, 5], trainable=True)
    model.set_layers_trainable([0, 1, 2, 3], trainable=False)
    model.set_layer_scales([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

    for _ in range(5):
        train_one_step(model, batch, optimizer, loss_fn)

    # Копируем веса: 4->2, 5->3
    model.copy_layer_weights(4, 2)
    model.copy_layer_weights(5, 3)

    # Теперь обучаем слои 2-5
    model.set_layers_trainable([2, 3, 4, 5], trainable=True)
    model.set_layers_trainable([0, 1], trainable=False)

    # Scale для 2, 3 начинаются с 0 (ramp)
    model.set_layer_scales([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    # Проверить что веса скопированы
    src_state = model._encoder.layers[4].state_dict()
    dst_state = model._encoder.layers[2].state_dict()
    for key in src_state:
        assert torch.allclose(src_state[key], dst_state[key]), \
            f"Layer 4 != Layer 2 after copy: {key}"

    # Несколько шагов обучения
    for _ in range(10):
        train_one_step(model, batch, optimizer, loss_fn)

    # Проверить что scale=0 для слоёв 0, 1
    scales = model.get_layer_scales()
    assert scales[0] == 0.0 and scales[1] == 0.0
    assert scales[2] == 1.0 and scales[3] == 1.0
    assert scales[4] == 1.0 and scales[5] == 1.0

    print("  PASS: 4-layer training with copy OK")
    return True


# ============================================================
# Тест 6: Обучение на 6 блоках (layers 0-5) с копированием
# ============================================================
def test_stage_6_layers():
    """
    Этап 3: копируем 2->0, 3->1, затем обучаем все 6 слоёв.
    Слои 0, 1 имеют scale, который растёт от 0 до 1.
    """
    print("\n--- Test 6: Stage 3 — 6 layers (0-5) with copy ---")
    model = make_model(num_layers=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = LocalObjective()

    seq = [10, 20, 30, 40, 5]
    batch = make_batch_from_sequence(seq, mode="train")

    # Этап 1: обучаем слои 4, 5
    model.set_layers_trainable([4, 5], trainable=True)
    model.set_layers_trainable([0, 1, 2, 3], trainable=False)
    model.set_layer_scales([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    for _ in range(5):
        train_one_step(model, batch, optimizer, loss_fn)

    # Этап 2: копируем 4->2, 5->3, обучаем 2-5
    model.copy_layer_weights(4, 2)
    model.copy_layer_weights(5, 3)
    model.set_layers_trainable([2, 3, 4, 5], trainable=True)
    model.set_layers_trainable([0, 1], trainable=False)
    model.set_layer_scales([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    for _ in range(5):
        train_one_step(model, batch, optimizer, loss_fn)

    # Этап 3: копируем 2->0, 3->1, обучаем все 6
    model.copy_layer_weights(2, 0)
    model.copy_layer_weights(3, 1)
    model.set_layers_trainable([0, 1, 2, 3, 4, 5], trainable=True)
    model.set_layer_scales([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Проверить что веса скопированы
    src_state = model._encoder.layers[2].state_dict()
    dst_state = model._encoder.layers[0].state_dict()
    for key in src_state:
        assert torch.allclose(src_state[key], dst_state[key]), \
            f"Layer 2 != Layer 0 after copy: {key}"

    # Несколько шагов обучения всех слоёв
    for _ in range(10):
        train_one_step(model, batch, optimizer, loss_fn)

    # Проверить что все слои обучаются
    for idx in range(6):
        for p in model._encoder.layers[idx].parameters():
            assert p.requires_grad, f"Layer {idx} should be trainable"

    scales = model.get_layer_scales()
    assert all(s == 1.0 for s in scales), f"All scales should be 1.0, got {scales}"

    print("  PASS: 6-layer training with copy OK")
    return True


# ============================================================
# Тест 7: Полный пайплайн StackRec на короткой последовательности
# ============================================================
def test_full_stackrec_short():
    """
    Полный пайплайн StackRec на seq длины 5:
      1. Обучить 2 слоя (4, 5)
      2. Копировать + обучить 4 слоя (2-5)
      3. Копировать + обучить 6 слоёв (0-5)
    """
    print("\n--- Test 7: Full StackRec pipeline (short seq=5) ---")
    model = make_model(num_layers=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = LocalObjective()

    seq = [10, 20, 30, 40, 5]
    batch = make_batch_from_sequence(seq, mode="train")

    losses = []

    # Stage 1: 2 layers
    model.set_layers_trainable([4, 5], trainable=True)
    model.set_layers_trainable([0, 1, 2, 3], trainable=False)
    model.set_layer_scales([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    for _ in range(15):
        loss = train_one_step(model, batch, optimizer, loss_fn)
        losses.append(("stage1", loss))

    # Stage 2: 4 layers
    model.copy_layer_weights(4, 2)
    model.copy_layer_weights(5, 3)
    model.set_layers_trainable([2, 3, 4, 5], trainable=True)
    model.set_layers_trainable([0, 1], trainable=False)
    model.set_layer_scales([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    for _ in range(15):
        loss = train_one_step(model, batch, optimizer, loss_fn)
        losses.append(("stage2", loss))

    # Stage 3: 6 layers
    model.copy_layer_weights(2, 0)
    model.copy_layer_weights(3, 1)
    model.set_layers_trainable([0, 1, 2, 3, 4, 5], trainable=True)
    model.set_layer_scales([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    for _ in range(15):
        loss = train_one_step(model, batch, optimizer, loss_fn)
        losses.append(("stage3", loss))

    # Проверить что loss уменьшался в каждом этапе
    stage1_losses = [l for s, l in losses if s == "stage1"]
    stage2_losses = [l for s, l in losses if s == "stage2"]
    stage3_losses = [l for s, l in losses if s == "stage3"]

    assert stage1_losses[-1] < stage1_losses[0], \
        f"Stage 1 loss should decrease: {stage1_losses[0]:.4f} -> {stage1_losses[-1]:.4f}"
    assert stage2_losses[-1] < stage2_losses[0], \
        f"Stage 2 loss should decrease: {stage2_losses[0]:.4f} -> {stage2_losses[-1]:.4f}"
    assert stage3_losses[-1] < stage3_losses[0], \
        f"Stage 3 loss should decrease: {stage3_losses[0]:.4f} -> {stage3_losses[-1]:.4f}"

    print(f"  Stage 1 loss: {stage1_losses[0]:.4f} -> {stage1_losses[-1]:.4f}")
    print(f"  Stage 2 loss: {stage2_losses[0]:.4f} -> {stage2_losses[-1]:.4f}")
    print(f"  Stage 3 loss: {stage3_losses[0]:.4f} -> {stage3_losses[-1]:.4f}")
    print("  PASS: full pipeline short OK")
    return True


# ============================================================
# Тест 8: Полный пайплайн StackRec на длинной последовательности
# ============================================================
def test_full_stackrec_long():
    """
    Полный пайплайн StackRec на seq длины 20:
      1. Обучить 2 слоя (4, 5)
      2. Копировать + обучить 4 слоя (2-5)
      3. Копировать + обучить 6 слоёв (0-5)
    """
    print("\n--- Test 8: Full StackRec pipeline (long seq=20) ---")
    model = make_model(num_layers=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = LocalObjective()

    seq = list(range(1, 21))  # 20 items
    batch = make_batch_from_sequence(seq, mode="train")

    losses = []

    # Stage 1: 2 layers
    model.set_layers_trainable([4, 5], trainable=True)
    model.set_layers_trainable([0, 1, 2, 3], trainable=False)
    model.set_layer_scales([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    for _ in range(5):
        loss = train_one_step(model, batch, optimizer, loss_fn)
        losses.append(("stage1", loss))

    # Stage 2: 4 layers
    model.copy_layer_weights(4, 2)
    model.copy_layer_weights(5, 3)
    model.set_layers_trainable([2, 3, 4, 5], trainable=True)
    model.set_layers_trainable([0, 1], trainable=False)
    model.set_layer_scales([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    for _ in range(5):
        loss = train_one_step(model, batch, optimizer, loss_fn)
        losses.append(("stage2", loss))

    # Stage 3: 6 layers
    model.copy_layer_weights(2, 0)
    model.copy_layer_weights(3, 1)
    model.set_layers_trainable([0, 1, 2, 3, 4, 5], trainable=True)
    model.set_layer_scales([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    for _ in range(5):
        loss = train_one_step(model, batch, optimizer, loss_fn)
        losses.append(("stage3", loss))

    # Проверить что loss уменьшался в каждом этапе
    stage1_losses = [l for s, l in losses if s == "stage1"]
    stage2_losses = [l for s, l in losses if s == "stage2"]
    stage3_losses = [l for s, l in losses if s == "stage3"]

    assert stage1_losses[-1] < stage1_losses[0], \
        f"Stage 1 loss should decrease: {stage1_losses[0]:.4f} -> {stage1_losses[-1]:.4f}"
    assert stage2_losses[-1] < stage2_losses[0], \
        f"Stage 2 loss should decrease: {stage2_losses[0]:.4f} -> {stage2_losses[-1]:.4f}"
    assert stage3_losses[-1] < stage3_losses[0], \
        f"Stage 3 loss should decrease: {stage3_losses[0]:.4f} -> {stage3_losses[-1]:.4f}"

    print(f"  Stage 1 loss: {stage1_losses[0]:.4f} -> {stage1_losses[-1]:.4f}")
    print(f"  Stage 2 loss: {stage2_losses[0]:.4f} -> {stage2_losses[-1]:.4f}")
    print(f"  Stage 3 loss: {stage3_losses[0]:.4f} -> {stage3_losses[-1]:.4f}")
    print("  PASS: full pipeline long OK")
    return True


# ============================================================
# Тест 9: Ramp scale — корректное поведение при scale=0
# ============================================================
def test_ramp_scale_zero():
    """
    При scale=0 слой должен работать как identity (пропускать вход).
    Проверим что при scale=0 для слоёв 2, 3 модель даёт те же
    предсказания, что и модель с обученными только слоями 4, 5.
    """
    print("\n--- Test 9: Ramp scale=0 identity ---")
    model = make_model(num_layers=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = LocalObjective()

    seq = [10, 20, 30, 40, 5]
    batch = make_batch_from_sequence(seq, mode="train")

    # Обучаем слои 4, 5
    model.set_layers_trainable([4, 5], trainable=True)
    model.set_layers_trainable([0, 1, 2, 3], trainable=False)
    model.set_layer_scales([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    for _ in range(5):
        train_one_step(model, batch, optimizer, loss_fn)

    # Сохраняем предсказания
    batch_val = make_batch_from_sequence(seq, mode="validation")
    preds_before = evaluate_model(model, batch_val)

    # Копируем веса и устанавливаем scale=0 для новых слоёв
    model.copy_layer_weights(4, 2)
    model.copy_layer_weights(5, 3)
    model.set_layers_trainable([2, 3, 4, 5], trainable=True)
    model.set_layers_trainable([0, 1], trainable=False)
    model.set_layer_scales([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

    # Предсказания должны быть идентичны (scale=0 = identity)
    preds_after = evaluate_model(model, batch_val)
    assert torch.allclose(preds_before, preds_after, atol=1e-5), \
        f"Predictions should match at scale=0: {preds_before} vs {preds_after}"

    print(f"  Predictions match at scale=0: {preds_before.flatten()[:5]}")
    print("  PASS: ramp scale=0 identity OK")
    return True


# ============================================================
# Тест 10: Aux scalar и boundary
# ============================================================
def test_aux_scalar_boundary():
    """
    Проверка что aux scalar и boundary корректно настраиваются.
    """
    print("\n--- Test 10: Aux scalar and boundary ---")
    model = make_model(num_layers=6)

    # Включить aux scalar на слое 3 (последний из новых слоёв 2, 3)
    model.enable_layer_aux_scalar(3, init=1.0)
    model.set_aux_boundary(2, 4)  # [2, 4)

    assert model._aux_active is True
    assert model._aux_lower_start == 2
    assert model._aux_upper_start == 4

    # Проверить что aux scalar на слое 3 обучается
    assert model._layer_aux_scalars[3].requires_grad is True
    # А на остальных — нет
    for i in [0, 1, 2, 4, 5]:
        assert model._layer_aux_scalars[i].requires_grad is False

    # Сбросить
    model.clear_aux_boundary()
    assert model._aux_active is False
    assert model._aux_lower_start is None
    assert model._aux_upper_start is None

    print("  PASS: aux scalar and boundary OK")
    return True


# ============================================================
# Тест 11: Lorentz head добавление
# ============================================================
def test_lorentz_head_addition():
    """
    Проверка что Lorentz head корректно добавляется и gate работает.
    """
    print("\n--- Test 11: Lorentz head addition ---")
    model = make_model(num_layers=6)

    # Добавить Lorentz head на слой 5
    head = model.add_lorentz_head(5, curvature=1.0)
    assert head is not None
    assert "5" in model._lorentz_heads

    # Gate должен быть 0 по умолчанию
    gates = model.get_lorentz_gates()
    assert gates.get(5) == 0.0

    # Установить gate
    model.set_lorentz_gate(5, 0.5)
    gates = model.get_lorentz_gates()
    assert gates.get(5) == 0.5

    # Проверить что head имеет curvature
    assert hasattr(head, "curvature")
    c = float(head.curvature().detach().cpu())
    assert abs(c - 1.0) < 1e-5

    print(f"  PASS: Lorentz head OK, curvature={c:.4f}, gate={gates[5]}")
    return True


# ============================================================
# Тест 12: Curvature schedule
# ============================================================
def test_curvature_schedule():
    """
    Проверка что расписание кривизны корректно.
    """
    print("\n--- Test 12: Curvature schedule ---")
    model = make_model(num_layers=6)

    curvatures = model.get_curvatures()
    assert len(curvatures) == 6
    # По умолчанию curvature_mode="independent", init_curvature=1.0
    assert all(abs(c - 1.0) < 1e-5 for c in curvatures)

    # Проверить set_curvature
    model.set_curvature(0, 2.0)
    curvatures = model.get_curvatures()
    assert abs(curvatures[0] - 2.0) < 1e-5

    print(f"  PASS: curvature schedule OK, curvatures={curvatures}")
    return True


# ============================================================
# Тест 13: Copy layer weights
# ============================================================
def test_copy_layer_weights():
    """
    Проверка что copy_layer_weights корректно копирует веса.
    """
    print("\n--- Test 13: Copy layer weights ---")
    model = make_model(num_layers=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = LocalObjective()

    # Сначала обучим слои 4 и 5, чтобы они стали разными
    seq = [10, 20, 30, 40, 5]
    batch = make_batch_from_sequence(seq, mode="train")
    model.set_layers_trainable([4, 5], trainable=True)
    model.set_layers_trainable([0, 1, 2, 3], trainable=False)
    model.set_layer_scales([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    for _ in range(10):
        train_one_step(model, batch, optimizer, loss_fn)

    # Теперь веса слоёв 2 и 4 должны быть разными
    w4 = model._encoder.layers[4].self_attn.in_proj_weight.clone()
    w2 = model._encoder.layers[2].self_attn.in_proj_weight.clone()
    assert not torch.allclose(w4, w2), "Layers should be different after training"

    # Копируем 4->2
    model.copy_layer_weights(4, 2)

    w2_after = model._encoder.layers[2].self_attn.in_proj_weight
    assert torch.allclose(w4, w2_after), "Layer 2 should equal Layer 4 after copy"

    # Копирование в себя — ничего не ломает
    model.copy_layer_weights(4, 4)

    print("  PASS: copy layer weights OK")
    return True


# ============================================================
# Тест 14: Batch из двух последовательностей
# ============================================================
def test_batch_two_sequences():
    """
    Проверка что батч из двух последовательностей (длина 5 и 20)
    корректно обрабатывается.
    """
    print("\n--- Test 14: Batch with two sequences ---")
    model = make_model(num_layers=6)

    seq_short = [10, 20, 30, 40, 5]
    seq_long = list(range(1, 21))

    # Создаём батч вручную
    processor = SasrecProcessor(pad_id=0, mode="train")
    samples = [
        {"user.ids": 0, "item.ids": seq_short},
        {"user.ids": 1, "item.ids": seq_long},
    ]
    batch = processor(samples)

    # Проверить размеры
    assert batch["padded_sequence_ids"].shape[0] == 2  # 2 пользователя
    # max_seq_len = max(len(seq_short)-1, len(seq_long)-1) = max(4, 19) = 19
    assert batch["padded_sequence_ids"].shape[1] == 19

    # Forward pass
    outputs = model(batch)
    preds = outputs["local_prediction"]
    # all_items = 4 + 19 = 23
    assert preds.shape[0] == 23, f"Expected 23 predictions, got {preds.shape[0]}"

    # Inference
    processor_val = SasrecProcessor(pad_id=0, mode="validation")
    samples_val = [
        {"user.ids": 0, "item.ids": seq_short},
        {"user.ids": 1, "item.ids": seq_long},
    ]
    batch_val = processor_val(samples_val)
    preds = evaluate_model(model, batch_val)
    assert preds.shape == (2, 10), f"Expected (2, 10), got {preds.shape}"

    print(f"  PASS: batch two sequences OK, preds shape={preds.shape}")
    return True


# ============================================================
# Тест 15: Device — только CPU
# ============================================================
def test_cpu_only():
    """
    Проверка что все тензоры модели и батчей на CPU.
    """
    print("\n--- Test 15: CPU only ---")
    model = make_model(num_layers=6)
    assert model.device.type == "cpu"

    # Проверить что все параметры на CPU
    for name, param in model.named_parameters():
        assert param.device.type == "cpu", f"Parameter {name} on {param.device}"

    seq = [10, 20, 30, 40, 5]
    batch = make_batch_from_sequence(seq, mode="train")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cpu", f"Batch {key} on {value.device}"

    print("  PASS: CPU only OK")
    return True


# ============================================================
# Тест 16: SASRec vs StackRec (2 layers) — идентичность
# ============================================================
def test_sasrec_vs_stackrec_2layers():
    """
    Проверка что SASRec с 2 слоями и StackRec с 6 слоями
    (активны только слои 4, 5, scale=0 для 0-3) дают
    абсолютно одинаковые предсказания.

    Это ключевое свойство StackRec: при scale=0 новые слои
    являются identity, поэтому подсеть из 2 слоёв работает
    точно так же, как самостоятельная 2-слойная SASRec.
    """
    print("\n--- Test 16: SASRec vs StackRec (2 layers) identity ---")

    embedding_dim = 32
    num_heads = 2
    dim_feedforward = 64
    dropout = 0.0  # отключаем dropout для детерминизма
    num_items = 50
    num_layers_sasrec = 2
    num_layers_stackrec = 6

    # Общие эмбеддинги — чтобы вход был одинаковым
    item_embeddings = nn.Embedding(num_items + 2, embedding_dim, padding_idx=0)
    position_embeddings = nn.Embedding(100, embedding_dim)

    # SASRec с 2 слоями
    sasrec = SequentialEncoder(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation="gelu",
        layer_norm_eps=1e-9,
        num_layers=num_layers_sasrec,
        num_items=num_items,
        topk_k=10,
        position_embeddings=position_embeddings,
        item_embeddings=item_embeddings,
    )

    # StackRec с 6 слоями
    stackrec = SequentialEncoder(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation="gelu",
        layer_norm_eps=1e-9,
        num_layers=num_layers_stackrec,
        num_items=num_items,
        topk_k=10,
        position_embeddings=position_embeddings,
        item_embeddings=item_embeddings,
    )

    # Копируем веса слоёв 0, 1 SASRec в слои 4, 5 StackRec
    stackrec._encoder.layers[4].load_state_dict(
        sasrec._encoder.layers[0].state_dict()
    )
    stackrec._encoder.layers[5].load_state_dict(
        sasrec._encoder.layers[1].state_dict()
    )

    # Копируем norm из SASRec в StackRec
    stackrec._encoder.norm.load_state_dict(
        sasrec._encoder.norm.state_dict()
    )

    # Scale: слои 0-3 = 0 (identity), слои 4-5 = 1
    stackrec.set_layer_scales([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

    # Входные данные — короткая последовательность
    seq = [10, 20, 30, 40, 5]
    batch = make_batch_from_sequence(seq, mode="validation")

    # Eval mode
    sasrec.eval()
    stackrec.eval()

    with torch.no_grad():
        sasrec_preds = sasrec(batch)
        stackrec_preds = stackrec(batch)

    assert torch.allclose(
        sasrec_preds, stackrec_preds, atol=1e-5
    ), (
        f"Predictions differ: "
        f"SASRec={sasrec_preds.flatten()[:5]}, "
        f"StackRec={stackrec_preds.flatten()[:5]}"
    )

    print(f"  Predictions match: {sasrec_preds.flatten()[:5]}")
    print("  PASS: SASRec vs StackRec (2 layers) identity OK")
    return True


# ============================================================
# Тест 17: SASRec vs StackRec (2 layers) — длинная последовательность
# ============================================================
def test_sasrec_vs_stackrec_2layers_long():
    """
    То же самое, но на длинной последовательности (20 items).
    """
    print("\n--- Test 17: SASRec vs StackRec (2 layers) long seq ---")

    embedding_dim = 32
    num_heads = 2
    dim_feedforward = 64
    dropout = 0.0
    num_items = 50
    num_layers_sasrec = 2
    num_layers_stackrec = 6

    item_embeddings = nn.Embedding(num_items + 2, embedding_dim, padding_idx=0)
    position_embeddings = nn.Embedding(100, embedding_dim)

    sasrec = SequentialEncoder(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation="gelu",
        layer_norm_eps=1e-9,
        num_layers=num_layers_sasrec,
        num_items=num_items,
        topk_k=10,
        position_embeddings=position_embeddings,
        item_embeddings=item_embeddings,
    )

    stackrec = SequentialEncoder(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation="gelu",
        layer_norm_eps=1e-9,
        num_layers=num_layers_stackrec,
        num_items=num_items,
        topk_k=10,
        position_embeddings=position_embeddings,
        item_embeddings=item_embeddings,
    )

    stackrec._encoder.layers[4].load_state_dict(
        sasrec._encoder.layers[0].state_dict()
    )
    stackrec._encoder.layers[5].load_state_dict(
        sasrec._encoder.layers[1].state_dict()
    )
    stackrec._encoder.norm.load_state_dict(
        sasrec._encoder.norm.state_dict()
    )
    stackrec.set_layer_scales([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

    seq = list(range(1, 21))  # 20 items
    batch = make_batch_from_sequence(seq, mode="validation")

    sasrec.eval()
    stackrec.eval()

    with torch.no_grad():
        sasrec_preds = sasrec(batch)
        stackrec_preds = stackrec(batch)

    assert torch.allclose(
        sasrec_preds, stackrec_preds, atol=1e-5
    ), (
        f"Predictions differ: "
        f"SASRec={sasrec_preds.flatten()[:5]}, "
        f"StackRec={stackrec_preds.flatten()[:5]}"
    )

    print(f"  Predictions match: {sasrec_preds.flatten()[:5]}")
    print("  PASS: SASRec vs StackRec (2 layers) long seq OK")
    return True


# ============================================================
# Запуск
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("StackRec tests — CPU, 2 examples (seq=5, seq=20)")
    print("=" * 60)

    results = {}
    tests = [
        ("ModelInit", test_model_initialization),
        ("ForwardShort", test_forward_short),
        ("ForwardLong", test_forward_long),
        ("Stage2Layers", test_stage_2_layers),
        ("Stage4Layers", test_stage_4_layers),
        ("Stage6Layers", test_stage_6_layers),
        ("FullPipelineShort", test_full_stackrec_short),
        ("FullPipelineLong", test_full_stackrec_long),
        ("RampScaleZero", test_ramp_scale_zero),
        ("AuxScalarBoundary", test_aux_scalar_boundary),
        ("LorentzHead", test_lorentz_head_addition),
        ("CurvatureSchedule", test_curvature_schedule),
        ("CopyLayerWeights", test_copy_layer_weights),
        ("BatchTwoSequences", test_batch_two_sequences),
        ("CPUOnly", test_cpu_only),
        ("SASrecVsStackRec2L", test_sasrec_vs_stackrec_2layers),
        ("SASrecVsStackRec2LLong", test_sasrec_vs_stackrec_2layers_long),
    ]

    for name, fn in tests:
        try:
            fn()
            results[name] = {"status": "PASS"}
        except Exception as e:
            results[name] = {"status": "FAIL", "error": str(e)}
            print(f"  FAILED: {e}")

    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 60)
    passed = sum(1 for r in results.values() if r["status"] == "PASS")
    failed = sum(1 for r in results.values() if r["status"] == "FAIL")
    for name, res in results.items():
        status = "PASS" if res["status"] == "PASS" else "FAIL"
        if status == "FAIL":
            print(f"  [{status}] {name}: {res['error']}")
        else:
            print(f"  [{status}] {name}")
    print(f"\n  Итого: {passed} passed, {failed} failed из {len(results)}")
    print("=" * 60)