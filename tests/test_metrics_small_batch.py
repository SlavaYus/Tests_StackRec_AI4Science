"""
Проверка всех используемых метрик на малых батчах: 1 и 2 пользователя.

Для каждого числа пользователей значения считаются вручную и сравниваются
с реализацией из src/metrics.py. Дополнительно проверяются:
  - групповые метрики по популярности (ndcg/recall/coverage/mean_rank groupN);
  - метрики Novelty и ILD;
  - compute_popularity_groups;
  - обнаружение намеренно внесённых ошибок (mutation testing):
    для каждой "плохой" метрики проверка с корректным ожиданием обязана
    упасть с AssertionError.

Запуск:
    python3 tests/test_metrics_small_batch.py
    pytest -q tests/test_metrics_small_batch.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import (
    CoverageMetric,
    ILDMetric,
    MeanRankMetric,
    NDCGMetric,
    NoveltyMetric,
    RecallMetric,
    compute_popularity_groups,
)
from src.utils import inference

# ============================================================
# Малый синтетический датасет
# ============================================================
K = 3
NUM_ITEMS = 5

# Расстояния dist(i, j) = |i - j| — легко проверить вручную.
DIST_MATRIX = torch.tensor(
    [[abs(i - j) for j in range(NUM_ITEMS)] for i in range(NUM_ITEMS)],
    dtype=torch.float,
)

# Популярность для Novelty и для групповых метрик.
ITEM2NUM = {0: 1, 1: 0, 2: 0, 3: 0, 4: 0}
NUM_USERS_TOTAL = 2

# Для групповых метрик: 2 группы, граница проходит между item1 и item2.
ITEM2NUM_GROUPS = {0: 10, 1: 8, 2: 6, 3: 4, 4: 2}
EXPECTED_GROUPS = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}


def make_batch(logits, labels):
    return {
        "logits": torch.tensor(logits, dtype=torch.long),
        "labels.ids": torch.tensor(labels, dtype=torch.long),
    }


# ============================================================
# Ручные ожидания для 1 и 2 пользователей
# ============================================================
CASES = {
    1: {
        # user 0: hit на позиции 2
        "batch": make_batch([[0, 1, 2]], [1]),
        "ndcg": [1.0 / np.log2(3.0)],
        "recall": [1.0],
        "coverage": 3.0 / NUM_ITEMS,
        "mean_rank": [2.0],
        # items 0,1,2: пары (0,1)=1, (0,2)=2, (1,2)=1
        "ild": [4.0 / 3.0],
        # recommended counts: 0->1, 1->1, 2->1
        # novelty: item0 skipped/0, item1=0.5, item2=0.5
        "novelty": (0.0 + 0.5 + 0.5) / 3.0,
    },
    2: {
        # user 0: hit на позиции 2, user 1: hit на позиции 3
        "batch": make_batch([[0, 1, 2], [2, 3, 4]], [1, 4]),
        "ndcg": [1.0 / np.log2(3.0), 1.0 / np.log2(4.0)],
        "recall": [1.0, 1.0],
        "coverage": 1.0,  # items 0..4
        "mean_rank": [2.0, 3.0],
        "ild": [4.0 / 3.0, 4.0 / 3.0],
        # counts: 0->1,1->1,2->2,3->1,4->1
        # novelty: 0.0, 0.5, 0.0, 0.5, 0.5
        "novelty": (0.0 + 0.5 + 0.0 + 0.5 + 0.5) / 5.0,
    },
}


# ============================================================
# Базовые метрики
# ============================================================
@pytest.mark.parametrize("n_users", [1, 2])
def test_ndcg(n_users):
    case = CASES[n_users]
    values = NDCGMetric(k=K)(case["batch"])["ndcg"]
    np.testing.assert_allclose(values, case["ndcg"], atol=1e-6)


@pytest.mark.parametrize("n_users", [1, 2])
def test_recall(n_users):
    case = CASES[n_users]
    values = RecallMetric(k=K)(case["batch"])["recall"]
    np.testing.assert_allclose(values, case["recall"], atol=1e-6)


@pytest.mark.parametrize("n_users", [1, 2])
def test_coverage(n_users):
    case = CASES[n_users]
    metric = CoverageMetric(k=K, num_items=NUM_ITEMS)
    raw = metric(case["batch"])["coverage"]
    assert len(raw) == n_users * K
    assert abs(metric.reduce(raw) - case["coverage"]) < 1e-6


@pytest.mark.parametrize("n_users", [1, 2])
def test_mean_rank(n_users):
    case = CASES[n_users]
    values = MeanRankMetric(k=K)(case["batch"])["mean_rank"]
    np.testing.assert_allclose(values, case["mean_rank"], atol=1e-6)


@pytest.mark.parametrize("n_users", [1, 2])
def test_ild(n_users):
    case = CASES[n_users]
    values = ILDMetric(k=K, dist_matrix=DIST_MATRIX)(case["batch"])
    np.testing.assert_allclose(values, case["ild"], atol=1e-6)


@pytest.mark.parametrize("n_users", [1, 2])
def test_novelty(n_users):
    case = CASES[n_users]
    metric = NoveltyMetric(
        k=K,
        item2num_iteractions=ITEM2NUM,
        num_users=NUM_USERS_TOTAL,
    )
    raw = metric(case["batch"])
    assert len(raw) == n_users * K
    assert abs(metric.reduce(raw) - case["novelty"]) < 1e-6


# ============================================================
# Групповые метрики по популярности
# ============================================================
def test_popularity_groups_small():
    groups = compute_popularity_groups(ITEM2NUM_GROUPS, NUM_ITEMS, n_groups=2)
    assert groups == EXPECTED_GROUPS


def test_group_metrics_two_users():
    batch = CASES[2]["batch"]
    kwargs = dict(
        k=K,
        num_items=NUM_ITEMS,
        item2num_interactions=ITEM2NUM_GROUPS,
        n_groups=2,
    )

    ndcg = NDCGMetric(**kwargs)(batch)
    # label user0 = 1 -> group0, label user1 = 4 -> group1
    np.testing.assert_allclose(ndcg["ndcg_group0"], [CASES[2]["ndcg"][0]], atol=1e-6)
    np.testing.assert_allclose(ndcg["ndcg_group1"], [CASES[2]["ndcg"][1]], atol=1e-6)

    recall = RecallMetric(**kwargs)(batch)
    np.testing.assert_allclose(recall["recall_group0"], [1.0], atol=1e-6)
    np.testing.assert_allclose(recall["recall_group1"], [1.0], atol=1e-6)

    mean_rank = MeanRankMetric(**kwargs)(batch)
    np.testing.assert_allclose(mean_rank["mean_rank_group0"], [2.0], atol=1e-6)
    np.testing.assert_allclose(mean_rank["mean_rank_group1"], [3.0], atol=1e-6)

    coverage_metric = CoverageMetric(**kwargs)
    coverage = coverage_metric(batch)
    # user0 -> items {0,1,2}, user1 -> items {2,3,4}
    assert abs(coverage_metric.reduce(coverage["coverage_group0"]) - 3.0 / NUM_ITEMS) < 1e-6
    assert abs(coverage_metric.reduce(coverage["coverage_group1"]) - 3.0 / NUM_ITEMS) < 1e-6


def test_group_metrics_one_user():
    batch = CASES[1]["batch"]
    kwargs = dict(
        k=K,
        num_items=NUM_ITEMS,
        item2num_interactions=ITEM2NUM_GROUPS,
        n_groups=2,
    )
    # label user0 = 1 -> group0, group1 пустая
    ndcg = NDCGMetric(**kwargs)(batch)
    assert len(ndcg["ndcg_group0"]) == 1
    assert ndcg["ndcg_group1"] == []

    recall = RecallMetric(**kwargs)(batch)
    assert len(recall["recall_group0"]) == 1
    assert recall["recall_group1"] == []

    mean_rank = MeanRankMetric(**kwargs)(batch)
    assert len(mean_rank["mean_rank_group0"]) == 1
    assert mean_rank["mean_rank_group1"] == []


# ============================================================
# End-to-end: inference() на 2 пользователях
# ============================================================
class _FixedModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self._logits = logits

    def forward(self, batch):
        return self._logits.clone()


def test_inference_end_to_end_two_users():
    case = CASES[2]
    batch = case["batch"]
    model = _FixedModel(batch["logits"])
    dataloader = [{"labels.ids": batch["labels.ids"]}]
    metrics = {
        "ndcg@3": NDCGMetric(k=K),
        "recall@3": RecallMetric(k=K),
        "coverage@3": CoverageMetric(k=K, num_items=NUM_ITEMS),
        "meanrank@3": MeanRankMetric(k=K),
    }

    results = inference(dataloader, model, metrics, device="cpu")

    assert abs(results["ndcg@3"] - np.mean(case["ndcg"])) < 1e-6
    assert abs(results["recall@3"] - np.mean(case["recall"])) < 1e-6
    assert abs(results["coverage@3"] - case["coverage"]) < 1e-6
    assert abs(results["meanrank@3"] - np.mean(case["mean_rank"])) < 1e-6


def test_inference_group_metrics_two_users():
    case = CASES[2]
    batch = case["batch"]
    model = _FixedModel(batch["logits"])
    dataloader = [{"labels.ids": batch["labels.ids"]}]
    kwargs = dict(
        k=K,
        num_items=NUM_ITEMS,
        item2num_interactions=ITEM2NUM_GROUPS,
        n_groups=2,
    )
    metrics = {
        "ndcg@3": NDCGMetric(**kwargs),
        "recall@3": RecallMetric(**kwargs),
        "coverage@3": CoverageMetric(**kwargs),
        "meanrank@3": MeanRankMetric(**kwargs),
    }

    results = inference(dataloader, model, metrics, device="cpu")

    assert abs(results["ndcg@3_group0"] - case["ndcg"][0]) < 1e-6
    assert abs(results["ndcg@3_group1"] - case["ndcg"][1]) < 1e-6
    assert abs(results["recall@3_group0"] - 1.0) < 1e-6
    assert abs(results["recall@3_group1"] - 1.0) < 1e-6
    assert abs(results["coverage@3_group0"] - 3.0 / NUM_ITEMS) < 1e-6
    assert abs(results["coverage@3_group1"] - 3.0 / NUM_ITEMS) < 1e-6
    assert abs(results["meanrank@3_group0"] - 2.0) < 1e-6
    assert abs(results["meanrank@3_group1"] - 3.0) < 1e-6


# ============================================================
# Mutation testing: "плохие" метрики должны ловиться проверками
# ============================================================
def _halve(value):
    if isinstance(value, dict):
        return {key: _halve(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_halve(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.number)):
        return float(value) / 2.0
    return value


class HalvedNDCG(NDCGMetric):
    def __call__(self, inputs):
        return _halve(super().__call__(inputs))


class WrongDiscountNDCG(NDCGMetric):
    """discount = 1 / log2(rank + 2) вместо 1 / log2(rank + 1)."""

    def __call__(self, inputs):
        predictions = inputs["logits"][:, : self._k].float()
        labels = inputs["labels.ids"].float()
        hits = torch.eq(predictions, labels[..., None]).float()
        discount = 1.0 / torch.log2(torch.arange(1, self._k + 1).float() + 2.0)
        return {"ndcg": (hits @ discount).tolist()}


class HalvedRecall(RecallMetric):
    def __call__(self, inputs):
        return _halve(super().__call__(inputs))


class HalvedCoverage(CoverageMetric):
    def reduce(self, values):
        return super().reduce(values) / 2.0


class FirstUserCoverage(CoverageMetric):
    """Coverage учитывает только первого пользователя."""

    def __call__(self, inputs):
        predictions = inputs["logits"][:1, : self._k]
        return {"coverage": predictions.reshape(-1).cpu().detach().tolist()}


class HalvedNovelty(NoveltyMetric):
    def reduce(self, recommended_items):
        return super().reduce(recommended_items) / 2.0


class HalvedMeanRank(MeanRankMetric):
    def __call__(self, inputs):
        return _halve(super().__call__(inputs))


class HalvedILD(ILDMetric):
    def __call__(self, inputs):
        return _halve(super().__call__(inputs))


def _assert_close(actual, expected, atol=1e-6):
    if not np.allclose(actual, expected, atol=atol):
        raise AssertionError(f"expected {expected}, got {actual}")


def _expect_error(description, check):
    """Проверка обязана упасть: именно так обнаруживается ошибка в метрике."""
    try:
        check()
    except AssertionError:
        print(f"  [DETECTED] {description}: получен AssertionError")
        return
    raise AssertionError(f"Ошибка НЕ обнаружена: {description}")


def _check_metric_values(n_users):
    case = CASES[n_users]
    batch = case["batch"]

    _expect_error(
        f"NDCG halved ({n_users} user)",
        lambda: _assert_close(HalvedNDCG(k=K)(batch)["ndcg"], case["ndcg"]),
    )
    _expect_error(
        f"NDCG wrong discount ({n_users} user)",
        lambda: _assert_close(WrongDiscountNDCG(k=K)(batch)["ndcg"], case["ndcg"]),
    )
    _expect_error(
        f"Recall halved ({n_users} user)",
        lambda: _assert_close(HalvedRecall(k=K)(batch)["recall"], case["recall"]),
    )

    def check_halved_coverage():
        metric = HalvedCoverage(k=K, num_items=NUM_ITEMS)
        _assert_close(metric.reduce(metric(batch)["coverage"]), case["coverage"])

    _expect_error(f"Coverage halved ({n_users} user)", check_halved_coverage)

    def check_halved_novelty():
        metric = HalvedNovelty(
            k=K,
            item2num_iteractions=ITEM2NUM,
            num_users=NUM_USERS_TOTAL,
        )
        _assert_close(metric.reduce(metric(batch)), case["novelty"])

    _expect_error(f"Novelty halved ({n_users} user)", check_halved_novelty)

    _expect_error(
        f"MeanRank halved ({n_users} user)",
        lambda: _assert_close(HalvedMeanRank(k=K)(batch)["mean_rank"], case["mean_rank"]),
    )
    _expect_error(
        f"ILD halved ({n_users} user)",
        lambda: _assert_close(
            HalvedILD(k=K, dist_matrix=DIST_MATRIX)(batch), case["ild"]
        ),
    )


def test_mutations_detected_one_user():
    _check_metric_values(1)


def test_mutations_detected_two_users():
    _check_metric_values(2)


def test_first_user_coverage_error_detected():
    """На 2 пользователях корректный Coverage = 1.0, у плохой метрики = 0.6."""
    batch = CASES[2]["batch"]
    correct_metric = CoverageMetric(k=K, num_items=NUM_ITEMS)
    correct = correct_metric.reduce(correct_metric(batch)["coverage"])

    buggy_metric = FirstUserCoverage(k=K, num_items=NUM_ITEMS)
    raw = buggy_metric(batch)["coverage"]
    buggy = buggy_metric.reduce(raw)

    assert abs(correct - 1.0) < 1e-6
    assert len(raw) == 1 * K
    assert abs(buggy - 3.0 / NUM_ITEMS) < 1e-6

    _expect_error(
        "Coverage only first user (2 users)",
        lambda: _assert_close(buggy, correct),
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", __file__]))
