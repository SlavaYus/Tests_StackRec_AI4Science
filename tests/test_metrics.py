"""
Тесты на метрики ранжирования (без GPU, без модели).
Данные: 10 пользователей, 10 товаров.
Каждый тест — детерминированный synthetic batch, проверяем корректность формул.
"""

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
    compute_popularity_groups,
)

# ============================================================
# Синтетические данные: 10 пользователей, 10 товаров
# ============================================================
NUM_USERS = 10
NUM_ITEMS = 10
K = 5  # top-k для метрик

# logits: (batch_size=10, top_k=5) — предсказанные item IDs
# labels: (batch_size=10,) — ground truth item ID
np.random.seed(42)

logits = torch.tensor([
    [0, 1, 2, 3, 4],   # user 0: предсказал 0,1,2,3,4; label=2 (hit@3)
    [5, 6, 7, 8, 9],   # user 1: предсказал 5,6,7,8,9; label=7 (hit@3)
    [0, 1, 2, 3, 4],   # user 2: label=0 (hit@1)
    [5, 6, 7, 8, 9],   # user 3: label=9 (hit@5)
    [0, 1, 2, 3, 4],   # user 4: label=5 (no hit)
    [5, 6, 7, 8, 9],   # user 5: label=6 (hit@2)
    [0, 1, 2, 3, 4],   # user 6: label=10 (no hit, out of range)
    [5, 6, 7, 8, 9],   # user 7: label=8 (hit@4)
    [0, 1, 2, 3, 4],   # user 8: label=3 (hit@4)
    [5, 6, 7, 8, 9],   # user 9: label=5 (hit@1)
], dtype=torch.long)

labels = torch.tensor([2, 7, 0, 9, 5, 6, 10, 8, 3, 5], dtype=torch.long)

# item2num_interactions для Coverage/Novelty
item2num_interactions = {i: np.random.randint(1, 20) for i in range(NUM_ITEMS)}
num_users_total = 100  # общее число пользователей для novelty

# distance matrix для ILD
dist_matrix = torch.rand(NUM_ITEMS, NUM_ITEMS)
dist_matrix = (dist_matrix + dist_matrix.T) / 2  # symmetric
np.fill_diagonal(dist_matrix.numpy(), 0)


def make_batch(logits, labels):
    return {
        "logits": logits,
        "labels.ids": labels,
    }


# ============================================================
# Тест 1: NDCG
# ============================================================
def test_ndcg():
    """
    NDCG@k = DCG / IDCG
    DCG = sum_{i=1}^{k} rel_i / log2(i+1)
    IDCG = sum_{i=1}^{k} 1 / log2(i+1)  (если есть хотя бы один hit)
    
    Для каждого пользователя:
      - hit при позиции 1: rel=1, discount=1/log2(2)=1.0
      - hit при позиции 2: rel=1, discount=1/log2(3)=0.6309
      - hit при позиции 3: rel=1, discount=1/log2(4)=0.5
      - hit при позиции 4: rel=1, discount=1/log2(5)=0.3869
      - hit при позиции 5: rel=1, discount=1/log2(6)=0.3219
    """
    metric = NDCGMetric(k=K)
    batch = make_batch(logits, labels)
    results = metric(batch)

    # Проверяем что вернул dict с ключом "ndcg"
    assert "ndcg" in results, f"Expected 'ndcg' key, got {results.keys()}"
    ndcg_values = results["ndcg"]
    assert len(ndcg_values) == NUM_USERS, f"Expected {NUM_USERS} values, got {len(ndcg_values)}"

    # Ручной расчёт для user 0: logits=[0,1,2,3,4], label=2
    # hit at position 3 (0-indexed: 2)
    # DCG = 1/log2(4) = 0.5
    # IDCG = 1/log2(2) = 1.0 (лучший случай — hit на позиции 1)
    # NDCG = 0.5 / 1.0 = 0.5
    expected_ndcg_0 = 0.5 / 1.0
    assert abs(ndcg_values[0] - expected_ndcg_0) < 1e-6, \
        f"User 0: expected {expected_ndcg_0}, got {ndcg_values[0]}"

    # User 4: label=5, logits=[0,1,2,3,4] — no hit
    # DCG = 0, NDCG = 0
    assert ndcg_values[4] == 0.0, f"User 4: expected 0.0, got {ndcg_values[4]}"

    # User 6: label=10, logits=[0,1,2,3,4] — no hit (item 10 out of range)
    assert ndcg_values[6] == 0.0, f"User 6: expected 0.0, got {ndcg_values[6]}"

    # User 9: label=5, logits=[5,6,7,8,9] — hit at position 1
    # DCG = 1/log2(2) = 1.0, IDCG = 1.0, NDCG = 1.0
    assert abs(ndcg_values[9] - 1.0) < 1e-6, f"User 9: expected 1.0, got {ndcg_values[9]}"

    # Средний NDCG
    mean_ndcg = np.mean(ndcg_values)
    print(f"  NDCG@{K} per user: {[round(v, 4) for v in ndcg_values]}")
    print(f"  Mean NDCG@{K}: {mean_ndcg:.6f}")
    return mean_ndcg


# ============================================================
# Тест 2: Recall
# ============================================================
def test_recall():
    """
    Recall@k = |pred ∩ true| / |true|
    Здесь |true| = 1 (один label на пользователя),
    значит recall = 1 если hit, 0 иначе.
    """
    metric = RecallMetric(k=K)
    batch = make_batch(logits, labels)
    results = metric(batch)

    assert "recall" in results
    recall_values = results["recall"]
    assert len(recall_values) == NUM_USERS

    # User 0: hit at pos 3 → recall=1
    assert recall_values[0] == 1.0
    # User 4: no hit → recall=0
    assert recall_values[4] == 0.0
    # User 6: label=10 out of range → recall=0
    assert recall_values[6] == 0.0

    mean_recall = np.mean(recall_values)
    print(f"  Recall@{K} per user: {[round(v, 4) for v in recall_values]}")
    print(f"  Mean Recall@{K}: {mean_recall:.6f}")
    return mean_recall


# ============================================================
# Тест 3: Coverage
# ============================================================
def test_coverage():
    """
    Coverage@k = |{recommended items}| / |all items|
    """
    metric = CoverageMetric(k=K, num_items=NUM_ITEMS)
    batch = make_batch(logits, labels)
    results = metric(batch)

    all_recommended = results["coverage"]
    assert len(all_recommended) == NUM_USERS * K

    unique_items = set(all_recommended)
    coverage = len(unique_items) / NUM_ITEMS

    # logits содержит только items 0-4 и 5-9 → все 10 items покрыты
    assert coverage == 1.0, f"Expected coverage=1.0, got {coverage}"

    print(f"  Coverage@{K}: {coverage:.6f} ({len(unique_items)}/{NUM_ITEMS} items)")
    return coverage


# ============================================================
# Тест 4: Novelty
# ============================================================
def test_novelty():
    """
    Novelty(i) = 1 - (recommended_times_i / (num_users - interactions_i))
    """
    metric = NoveltyMetric(k=K, item2num_iteractions=item2num_interactions, num_users=num_users_total)
    batch = make_batch(logits, labels)
    results = metric(batch)

    all_recommended = results
    novelty = metric.reduce(all_recommended)

    print(f"  Novelty@{K}: {novelty:.6f}")
    return novelty


# ============================================================
# Тест 5: MeanRank
# ============================================================
def test_mean_rank():
    """
    Mean Rank: позиция первого hit в recommendations.
    Если нет hit — rank = k+1.
    """
    metric = MeanRankMetric(k=K)
    batch = make_batch(logits, labels)
    results = metric(batch)

    assert "mean_rank" in results
    ranks = results["mean_rank"]
    assert len(ranks) == NUM_USERS

    # User 0: label=2, logits=[0,1,2,3,4] → hit at pos 3 → rank=3
    assert ranks[0] == 3.0, f"User 0: expected 3.0, got {ranks[0]}"
    # User 2: label=0, logits=[0,1,2,3,4] → hit at pos 1 → rank=1
    assert ranks[2] == 1.0, f"User 2: expected 1.0, got {ranks[2]}"
    # User 4: label=5, logits=[0,1,2,3,4] → no hit → rank=6 (k+1)
    assert ranks[4] == 6.0, f"User 4: expected 6.0, got {ranks[4]}"
    # User 6: label=10, logits=[0,1,2,3,4] → no hit → rank=6
    assert ranks[6] == 6.0, f"User 6: expected 6.0, got {ranks[6]}"
    # User 9: label=5, logits=[5,6,7,8,9] → hit at pos 1 → rank=1
    assert ranks[9] == 1.0, f"User 9: expected 1.0, got {ranks[9]}"

    mean_rank = np.mean(ranks)
    print(f"  MeanRank@{K} per user: {[round(v, 4) for v in ranks]}")
    print(f"  Mean MeanRank@{K}: {mean_rank:.6f}")
    return mean_rank


# ============================================================
# Тест 6: ILD (Intra-List Diversity)
# ============================================================
def test_ild():
    """
    ILD(u) = mean over pairs (i,j) in top-k of dist(item_i, item_j)
    """
    metric = ILDMetric(k=K, dist_matrix=dist_matrix)
    batch = make_batch(logits, labels)
    ild_values = metric(batch)

    assert len(ild_values) == NUM_USERS

    # User 0: logits=[0,1,2,3,4]
    # Pairs: (0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4) = 10 pairs
    expected_ild_0 = np.mean([
        dist_matrix[0, 1], dist_matrix[0, 2], dist_matrix[0, 3], dist_matrix[0, 4],
        dist_matrix[1, 2], dist_matrix[1, 3], dist_matrix[1, 4],
        dist_matrix[2, 3], dist_matrix[2, 4],
        dist_matrix[3, 4],
    ])
    assert abs(ild_values[0] - expected_ild_0) < 1e-6, \
        f"User 0 ILD: expected {expected_ild_0}, got {ild_values[0]}"

    mean_ild = np.mean(ild_values)
    print(f"  ILD@{K} per user: {[round(v, 4) for v in ild_values]}")
    print(f"  Mean ILD@{K}: {mean_ild:.6f}")
    return mean_ild


# ============================================================
# Тест 7: compute_popularity_groups
# ============================================================
def test_popularity_groups():
    """
    Проверяем что группы формируются корректно.
    """
    item2group = compute_popularity_groups(item2num_interactions, NUM_ITEMS, n_groups=3)

    assert len(item2group) == NUM_ITEMS + 2  # +2 для fallback (pad_id=0 и beyond)
    groups = set(item2group.values())
    assert groups <= {0, 1, 2}, f"Unexpected groups: {groups}"

    # Все items должны быть в какой-то группе
    for i in range(NUM_ITEMS):
        assert i in item2group, f"Item {i} not in groups"

    print(f"  Groups: {dict(sorted(item2group.items()))}")
    return True


# ============================================================
# Тест 8: Edge cases
# ============================================================
def test_edge_cases():
    """
    Все hits, no hits.
    Примечание: NDCGMetric в данном коде вычисляет DCG (без деления на IDCG).
    """
    # --- Все hits на позиции 1 (уникальные items в logits) ---
    logits_all_hit = torch.zeros((NUM_USERS, K), dtype=torch.long)
    for i in range(NUM_USERS):
        logits_all_hit[i, 0] = i  # hit на позиции 1
        # остальные позиции — другие items
        for j in range(1, K):
            logits_all_hit[i, j] = (i + j + 1) % NUM_ITEMS
    labels_all_hit = torch.arange(NUM_USERS, dtype=torch.long)
    batch = make_batch(logits_all_hit, labels_all_hit)

    ndcg = NDCGMetric(k=K)(batch)["ndcg"]
    # DCG при hit на позиции 1 = 1/log2(2) = 1.0
    assert all(abs(v - 1.0) < 1e-6 for v in ndcg), f"All-hit DCG: {ndcg}"

    recall = RecallMetric(k=K)(batch)["recall"]
    assert all(v == 1.0 for v in recall), f"All-hit Recall: {recall}"

    # --- No hits ---
    # items 100-104 не пересекаются с labels [0..9]
    logits_no_hit = torch.full((NUM_USERS, K), 100, dtype=torch.long)
    labels_no_hit = torch.arange(NUM_USERS, dtype=torch.long)
    batch = make_batch(logits_no_hit, labels_no_hit)

    ndcg = NDCGMetric(k=K)(batch)["ndcg"]
    assert all(v == 0.0 for v in ndcg), f"No-hit DCG: {ndcg}"

    recall = RecallMetric(k=K)(batch)["recall"]
    assert all(v == 0.0 for v in recall), f"No-hit Recall: {recall}"

    print("  Edge cases: ALL-HIT and NO-HIT passed")
    return True


# ============================================================
# Тест 9: Coverage с частичным покрытием
# ============================================================
def test_coverage_partial():
    """
    Все рекомендации — один и тот же item → coverage = 1/NUM_ITEMS.
    """
    logits_single = torch.full((NUM_USERS, K), 5, dtype=torch.long)
    labels_single = torch.arange(NUM_USERS, dtype=torch.long)
    batch = make_batch(logits_single, labels_single)

    metric = CoverageMetric(k=K, num_items=NUM_ITEMS)
    results = metric(batch)
    coverage = metric.reduce(results["coverage"])

    assert coverage == 1.0 / NUM_ITEMS, f"Expected {1.0/NUM_ITEMS}, got {coverage}"
    print(f"  Coverage@{K} (single item): {coverage:.6f}")
    return coverage


# ============================================================
# Тест 10: MeanRank с частичным покрытием
# ============================================================
def test_mean_rank_partial():
    """
    Все рекомендации — один item, label совпадает для всех → rank=1 для всех.
    """
    logits_single = torch.full((NUM_USERS, K), 5, dtype=torch.long)
    labels_single = torch.full((NUM_USERS,), 5, dtype=torch.long)
    batch = make_batch(logits_single, labels_single)

    metric = MeanRankMetric(k=K)
    results = metric(batch)
    ranks = results["mean_rank"]

    assert all(r == 1.0 for r in ranks), f"Expected all ranks=1.0, got {ranks}"
    print(f"  MeanRank@{K} (all hit at pos 1): {np.mean(ranks):.6f}")
    return np.mean(ranks)


# ============================================================
# Запуск
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Метрики ранжирования — тесты (CPU, 10 users, 10 items)")
    print("=" * 60)

    results = {}
    tests = [
        ("NDCG@5", test_ndcg),
        ("Recall@5", test_recall),
        ("Coverage@5", test_coverage),
        ("Novelty@5", test_novelty),
        ("MeanRank@5", test_mean_rank),
        ("ILD@5", test_ild),
        ("PopularityGroups", test_popularity_groups),
        ("EdgeCases", test_edge_cases),
        ("CoveragePartial", test_coverage_partial),
        ("MeanRankPartial", test_mean_rank_partial),
    ]

    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            val = fn()
            results[name] = {"status": "PASS", "value": val}
        except Exception as e:
            results[name] = {"status": "FAIL", "value": str(e)}
            print(f"  FAILED: {e}")

    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 60)
    for name, res in results.items():
        status = "PASS" if res["status"] == "PASS" else "FAIL"
        val = res["value"]
        if isinstance(val, float):
            print(f"  [{status}] {name}: {val:.6f}")
        else:
            print(f"  [{status}] {name}: {val}")
    print("=" * 60)