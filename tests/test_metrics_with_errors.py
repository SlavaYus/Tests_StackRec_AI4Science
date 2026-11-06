"""
Тесты с намеренно внесёнными ошибками в метрики (fault injection / mutation).

Часть 1: Тесты, которые проверяют, что ошибки обнаруживаются (PASS).
Часть 2: Тесты, которые используют ошибочные метрики и ожидают корректные значения (FAIL).

Запуск:
    python3 tests/test_metrics_with_errors.py
    pytest tests/test_metrics_with_errors.py -v
"""

import numbers
import sys
from pathlib import Path

import numpy as np
import torch

# Добавляем корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import (
    NDCGMetric,
    RecallMetric,
    CoverageMetric,
    NoveltyMetric,
    MeanRankMetric,
    ILDMetric,
)


# ============================================================
# Синтетические данные: 20 пользователей, 20 товаров
# ============================================================
NUM_USERS = 20
NUM_ITEMS = 20
K = 5

# Первые 10 пользователей получают рекомендации из items 0-9,
# последние 10 — из items 10-19. Поэтому корректный Coverage = 1.0.
logits = torch.tensor([
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
], dtype=torch.long)

labels = torch.tensor([
    2, 7, 0, 9, 5, 6, 10, 8, 3, 5,
    12, 17, 10, 19, 15, 16, 20, 18, 13, 15,
], dtype=torch.long)

item2num_interactions = {i: (i % 7) + 1 for i in range(NUM_ITEMS)}
num_users_total = 100

dist_matrix = torch.rand(NUM_ITEMS, NUM_ITEMS)
dist_matrix = (dist_matrix + dist_matrix.T) / 2
np.fill_diagonal(dist_matrix.numpy(), 0)


def make_batch(logits, labels):
    return {
        "logits": logits,
        "labels.ids": labels,
    }


def _halve(value):
    """Рекурсивно делит все числовые значения результата метрики на 2."""
    if isinstance(value, dict):
        return {key: _halve(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_halve(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Real):
        return value / 2.0
    return value


def buggy_dcg(predictions, labels, k):
    """
    НЕВЕРНЫЙ NDCG/DCG:
      - discount считается как 1 / log2(rank + 2);
      - отсутствует нормализация на IDCG (как и в исходной реализации);
      - результат дополнительно делится на 2.
    """
    hits = torch.eq(
        predictions[:, :k].float(),
        labels[..., None].float(),
    ).float()

    # ОШИБКА: rank + 2 вместо rank + 1
    ranks = torch.arange(1, k + 1, dtype=torch.float)
    discount = 1.0 / torch.log2(ranks + 2.0)

    dcg = hits @ discount
    return dcg / 2.0


# ============================================================
# "Плохие" метрики
# ============================================================
class HalvedNDCGMetric(NDCGMetric):
    """NDCG, в котором все значения просто поделены на 2."""

    def __call__(self, inputs):
        return _halve(super().__call__(inputs))


class BuggyNDCGMetric(NDCGMetric):
    """NDCG с неверной формулой discount и делением на 2."""

    def __call__(self, inputs):
        dcg = buggy_dcg(inputs["logits"], inputs["labels.ids"], self._k)
        return {"ndcg": dcg.cpu().tolist()}


class HalvedRecallMetric(RecallMetric):
    def __call__(self, inputs):
        return _halve(super().__call__(inputs))


class HalvedCoverageMetric(CoverageMetric):
    def reduce(self, values):
        return super().reduce(values) / 2.0


class BuggyCoverageMetric(CoverageMetric):
    """
    Coverage с двумя ошибками:
      - учитываются только первые 10 пользователей;
      - результат делится на 2.
    """

    FIRST_USERS = 10

    def __call__(self, inputs):
        predictions = inputs["logits"][:, : self._k]
        # ОШИБКА: только первые 10 пользователей
        all_items = (
            predictions[: self.FIRST_USERS]
            .reshape(-1)
            .cpu()
            .detach()
            .tolist()
        )
        return {"coverage": all_items}

    def reduce(self, values):
        # ОШИБКА: результат делится на 2
        return super().reduce(values) / 2.0


class HalvedNoveltyMetric(NoveltyMetric):
    def reduce(self, recommended_items):
        return super().reduce(recommended_items) / 2.0


class HalvedMeanRankMetric(MeanRankMetric):
    def __call__(self, inputs):
        return _halve(super().__call__(inputs))


class HalvedILDMetric(ILDMetric):
    def __call__(self, inputs):
        return _halve(super().__call__(inputs))


# ============================================================
# ЧАСТЬ 1: Тесты, которые проверяют, что ошибки обнаруживаются (PASS)
# ============================================================

def test_buggy_ndcg_formula():
    """Проверяет, что BuggyNDCG даёт другое значение, чем корректный NDCG."""
    batch = make_batch(logits, labels)

    correct = NDCGMetric(k=K)(batch)["ndcg"]
    buggy = BuggyNDCGMetric(k=K)(batch)["ndcg"]

    # user 0: hit на позиции 3.
    # Корректно: 1 / log2(4) = 0.5.
    # С ошибкой: (1 / log2(5)) / 2.
    expected_buggy_0 = (1.0 / np.log2(5.0)) / 2.0
    assert abs(buggy[0] - expected_buggy_0) < 1e-6, (
        f"User 0: expected {expected_buggy_0}, got {buggy[0]}"
    )
    assert abs(buggy[0] - correct[0]) > 1e-6, (
        "Ошибка NDCG не обнаружена: значения совпали с корректными"
    )

    print(f"  NDCG correct[0] = {correct[0]:.6f}, buggy[0] = {buggy[0]:.6f}")
    return "OK"


def test_all_metrics_are_halved():
    """Проверяет, что все halved-метрики дают значения в 2 раза меньше."""
    batch = make_batch(logits, labels)
    checks = {}

    # NDCG (только деление на 2)
    correct = NDCGMetric(k=K)(batch)["ndcg"]
    halved = HalvedNDCGMetric(k=K)(batch)["ndcg"]
    checks["ndcg"] = (np.mean(correct), np.mean(halved))

    # Recall
    correct = RecallMetric(k=K)(batch)["recall"]
    halved = HalvedRecallMetric(k=K)(batch)["recall"]
    checks["recall"] = (np.mean(correct), np.mean(halved))

    # Coverage
    correct_metric = CoverageMetric(k=K, num_items=NUM_ITEMS)
    correct = correct_metric.reduce(correct_metric(batch)["coverage"])
    halved_metric = HalvedCoverageMetric(k=K, num_items=NUM_ITEMS)
    halved = halved_metric.reduce(halved_metric(batch)["coverage"])
    checks["coverage"] = (correct, halved)

    # Novelty
    correct_metric = NoveltyMetric(
        k=K,
        item2num_iteractions=item2num_interactions,
        num_users=num_users_total,
    )
    correct = correct_metric.reduce(correct_metric(batch))
    halved_metric = HalvedNoveltyMetric(
        k=K,
        item2num_iteractions=item2num_interactions,
        num_users=num_users_total,
    )
    halved = halved_metric.reduce(halved_metric(batch))
    checks["novelty"] = (correct, halved)

    # MeanRank
    correct = MeanRankMetric(k=K)(batch)["mean_rank"]
    halved = HalvedMeanRankMetric(k=K)(batch)["mean_rank"]
    checks["mean_rank"] = (np.mean(correct), np.mean(halved))

    # ILD
    correct = ILDMetric(k=K, dist_matrix=dist_matrix)(batch)
    halved = HalvedILDMetric(k=K, dist_matrix=dist_matrix)(batch)
    checks["ild"] = (np.mean(correct), np.mean(halved))

    for name, (correct_value, halved_value) in checks.items():
        assert abs(halved_value - correct_value / 2.0) < 1e-6, (
            f"{name}: expected {correct_value / 2.0}, got {halved_value}"
        )
        print(
            f"  {name}: correct={correct_value:.6f}, "
            f"halved={halved_value:.6f}"
        )

    return "OK"


def test_buggy_coverage_first_10_users():
    """Проверяет, что BuggyCoverage учитывает только первые 10 пользователей."""
    batch = make_batch(logits, labels)

    correct_metric = CoverageMetric(k=K, num_items=NUM_ITEMS)
    correct = correct_metric.reduce(correct_metric(batch)["coverage"])
    assert abs(correct - 1.0) < 1e-6, f"Correct coverage: {correct}"

    buggy_metric = BuggyCoverageMetric(k=K, num_items=NUM_ITEMS)
    raw_items = buggy_metric(batch)["coverage"]
    buggy = buggy_metric.reduce(raw_items)

    # Только первые 10 пользователей: 10 * K рекомендаций
    assert len(raw_items) == BuggyCoverageMetric.FIRST_USERS * K, (
        f"Expected {BuggyCoverageMetric.FIRST_USERS * K} items, "
        f"got {len(raw_items)}"
    )

    expected_buggy = (
        len(set(raw_items)) / NUM_ITEMS / 2.0
    )
    assert abs(buggy - expected_buggy) < 1e-6, (
        f"Expected {expected_buggy}, got {buggy}"
    )
    assert abs(buggy - correct / 2.0) > 1e-6, (
        "Ошибка Coverage (первые 10 пользователей) не обнаружена"
    )

    print(
        f"  coverage: correct={correct:.6f}, buggy={buggy:.6f}, "
        f"items_used={len(raw_items)}"
    )
    return "OK"


def test_correct_expectations_detect_mutations():
    """Проверяет, что корректные проверки обнаруживают внесённые ошибки."""
    batch = make_batch(logits, labels)

    # NDCG: user 9 — hit на позиции 1, корректное значение 1.0.
    correct_ndcg = NDCGMetric(k=K)(batch)["ndcg"]
    buggy_ndcg = BuggyNDCGMetric(k=K)(batch)["ndcg"]
    assert abs(correct_ndcg[9] - 1.0) < 1e-6
    assert abs(buggy_ndcg[9] - 1.0) > 1e-6, (
        "Тест NDCG не смог обнаружить ошибку: user 9 всё ещё равен 1.0"
    )

    # Coverage: корректно 1.0, с ошибкой — 0.25.
    correct_metric = CoverageMetric(k=K, num_items=NUM_ITEMS)
    correct_coverage = correct_metric.reduce(
        correct_metric(batch)["coverage"]
    )
    buggy_metric = BuggyCoverageMetric(k=K, num_items=NUM_ITEMS)
    buggy_coverage = buggy_metric.reduce(buggy_metric(batch)["coverage"])
    assert abs(correct_coverage - 1.0) < 1e-6
    assert abs(buggy_coverage - 0.25) < 1e-6, (
        f"Expected buggy coverage 0.25, got {buggy_coverage}"
    )

    # Все метрики / 2: сравнение с корректным значением не проходит.
    correct_recall = np.mean(RecallMetric(k=K)(batch)["recall"])
    halved_recall = np.mean(HalvedRecallMetric(k=K)(batch)["recall"])
    assert abs(halved_recall - correct_recall) > 1e-6

    print(
        f"  detected: ndcg[9]={buggy_ndcg[9]:.6f} (correct 1.0), "
        f"coverage={buggy_coverage:.6f} (correct 1.0), "
        f"recall={halved_recall:.6f} (correct {correct_recall:.6f})"
    )
    return "OK"


# ============================================================
# ЧАСТЬ 2: Тесты, которые используют ошибочные метрики и ожидают
# корректные значения — они должны FAIL
# ============================================================

def test_buggy_ndcg_should_fail():
    """
    BUG: BuggyNDCGMetric ожидает корректное значение NDCG.
    Поскольку метрика ошибочна, тест FAIL.
    """
    batch = make_batch(logits, labels)

    # Корректное значение для user 9: hit на позиции 1 → NDCG = 1.0
    correct_ndcg = NDCGMetric(k=K)(batch)["ndcg"]
    assert abs(correct_ndcg[9] - 1.0) < 1e-6, "База: корректный NDCG[9] = 1.0"

    # Ошибочная метрика: должна дать 1.0, но даст ~0.315
    buggy_ndcg = BuggyNDCGMetric(k=K)(batch)["ndcg"]
    assert abs(buggy_ndcg[9] - 1.0) < 1e-6, (
        f"BuggyNDCGMetric FAILED: expected 1.0, got {buggy_ndcg[9]:.6f}"
    )


def test_halved_ndcg_should_fail():
    """
    BUG: HalvedNDCGMetric ожидает корректное значение NDCG.
    Поскольку метрика делит на 2, тест FAIL.
    """
    batch = make_batch(logits, labels)

    correct_ndcg = NDCGMetric(k=K)(batch)["ndcg"]
    halved_ndcg = HalvedNDCGMetric(k=K)(batch)["ndcg"]

    # Ожидаем корректное среднее, но получаем поделенное на 2
    expected_mean = np.mean(correct_ndcg)
    assert abs(np.mean(halved_ndcg) - expected_mean) < 1e-6, (
        f"HalvedNDCGMetric FAILED: expected {expected_mean:.6f}, "
        f"got {np.mean(halved_ndcg):.6f}"
    )


def test_halved_recall_should_fail():
    """
    BUG: HalvedRecallMetric ожидает корректное значение Recall.
    Поскольку метрика делит на 2, тест FAIL.
    """
    batch = make_batch(logits, labels)

    correct_recall = np.mean(RecallMetric(k=K)(batch)["recall"])
    halved_recall = np.mean(HalvedRecallMetric(k=K)(batch)["recall"])

    assert abs(halved_recall - correct_recall) < 1e-6, (
        f"HalvedRecallMetric FAILED: expected {correct_recall:.6f}, "
        f"got {halved_recall:.6f}"
    )


def test_buggy_coverage_should_fail():
    """
    BUG: BuggyCoverageMetric ожидает корректное значение Coverage.
    Поскольку метрика учитывает только первых 10 пользователей и делит на 2,
    тест FAIL.
    """
    batch = make_batch(logits, labels)

    correct_metric = CoverageMetric(k=K, num_items=NUM_ITEMS)
    correct_coverage = correct_metric.reduce(correct_metric(batch)["coverage"])
    assert abs(correct_coverage - 1.0) < 1e-6, "База: корректный Coverage = 1.0"

    buggy_metric = BuggyCoverageMetric(k=K, num_items=NUM_ITEMS)
    buggy_coverage = buggy_metric.reduce(buggy_metric(batch)["coverage"])

    assert abs(buggy_coverage - correct_coverage) < 1e-6, (
        f"BuggyCoverageMetric FAILED: expected {correct_coverage:.6f}, "
        f"got {buggy_coverage:.6f}"
    )


def test_halved_novelty_should_fail():
    """
    BUG: HalvedNoveltyMetric ожидает корректное значение Novelty.
    Поскольку метрика делит на 2, тест FAIL.
    """
    batch = make_batch(logits, labels)

    correct_metric = NoveltyMetric(
        k=K,
        item2num_iteractions=item2num_interactions,
        num_users=num_users_total,
    )
    correct_novelty = correct_metric.reduce(correct_metric(batch))
    halved_metric = HalvedNoveltyMetric(
        k=K,
        item2num_iteractions=item2num_interactions,
        num_users=num_users_total,
    )
    halved_novelty = halved_metric.reduce(halved_metric(batch))

    assert abs(halved_novelty - correct_novelty) < 1e-6, (
        f"HalvedNoveltyMetric FAILED: expected {correct_novelty:.6f}, "
        f"got {halved_novelty:.6f}"
    )


def test_halved_meanrank_should_fail():
    """
    BUG: HalvedMeanRankMetric ожидает корректное значение MeanRank.
    Поскольку метрика делит на 2, тест FAIL.
    """
    batch = make_batch(logits, labels)

    correct_mr = np.mean(MeanRankMetric(k=K)(batch)["mean_rank"])
    halved_mr = np.mean(HalvedMeanRankMetric(k=K)(batch)["mean_rank"])

    assert abs(halved_mr - correct_mr) < 1e-6, (
        f"HalvedMeanRankMetric FAILED: expected {correct_mr:.6f}, "
        f"got {halved_mr:.6f}"
    )


def test_halved_ild_should_fail():
    """
    BUG: HalvedILDMetric ожидает корректное значение ILD.
    Поскольку метрика делит на 2, тест FAIL.
    """
    batch = make_batch(logits, labels)

    correct_ild = np.mean(ILDMetric(k=K, dist_matrix=dist_matrix)(batch))
    halved_ild = np.mean(HalvedILDMetric(k=K, dist_matrix=dist_matrix)(batch))

    assert abs(halved_ild - correct_ild) < 1e-6, (
        f"HalvedILDMetric FAILED: expected {correct_ild:.6f}, "
        f"got {halved_ild:.6f}"
    )


# ============================================================
# Запуск
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Метрики с намеренными ошибками — проверка (CPU)")
    print("=" * 60)

    results = {}
    tests = [
        # Часть 1: ошибки обнаруживаются (PASS)
        ("BuggyNDCGFormula", test_buggy_ndcg_formula),
        ("AllMetricsHalved", test_all_metrics_are_halved),
        ("BuggyCoverageFirst10Users", test_buggy_coverage_first_10_users),
        ("CorrectExpectationsDetectMutations", test_correct_expectations_detect_mutations),
        # Часть 2: ошибочные метрики ожидают корректные значения (FAIL)
        ("BuggyNDCGShouldFail", test_buggy_ndcg_should_fail),
        ("HalvedNDCGShouldFail", test_halved_ndcg_should_fail),
        ("HalvedRecallShouldFail", test_halved_recall_should_fail),
        ("BuggyCoverageShouldFail", test_buggy_coverage_should_fail),
        ("HalvedNoveltyShouldFail", test_halved_novelty_should_fail),
        ("HalvedMeanRankShouldFail", test_halved_meanrank_should_fail),
        ("HalvedILDShouldFail", test_halved_ild_should_fail),
    ]

    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            value = fn()
            results[name] = {"status": "PASS", "value": value}
        except Exception as e:
            results[name] = {"status": "FAIL", "value": str(e)}
            print(f"  FAILED: {e}")

    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 60)
    for name, res in results.items():
        print(f"  [{res['status']}] {name}: {res['value']}")
    passed = sum(1 for r in results.values() if r["status"] == "PASS")
    failed = sum(1 for r in results.values() if r["status"] == "FAIL")
    print(f"\n  Итого: {passed} passed, {failed} failed")
    print("=" * 60)