"""
Проверка temporal split датасета.

Сплиты в data/global_split/<dataset>/ являются накопительными:
    train ⊂ validation ⊂ test (по префиксу истории каждого пользователя).

Поэтому корректная проверка "Train раньше Test" такая:
  - для каждого пользователя train-события идут строго раньше
    test-only событий (тех, которых нет в train);
  - train является префиксом test для этого пользователя;
  - аналогично для validation -> test.

Запуск:
    python3 tests/test_dataset_split.py
    pytest -q tests/test_dataset_split.py
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "global_split" / "reviews_Beauty_5"

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"


def _load(path):
    if not path.exists():
        pytest.skip(f"Нет файла {path}")
    df = pd.read_csv(path)
    required = {"user_id", "item_id", "timestamp"}
    assert required <= set(df.columns), f"{path}: нет колонок {required}"
    return df.sort_values(["user_id", "timestamp"], kind="mergesort")


def _group(df):
    return {
        int(user_id): group.reset_index(drop=True)
        for user_id, group in df.groupby("user_id", sort=True)
    }


def _rows(group):
    return list(zip(group["item_id"].tolist(), group["timestamp"].tolist()))


def _check_split(earlier, later, earlier_name, later_name):
    """earlier — префикс later; все earlier-события строго раньше later-only."""
    g_earlier = _group(earlier)
    g_later = _group(later)

    common_users = sorted(set(g_earlier) & set(g_later))
    assert common_users, f"Нет общих пользователей между {earlier_name} и {later_name}"

    checked_with_new_events = 0
    for user_id in common_users:
        e = g_earlier[user_id]
        l = g_later[user_id]

        # 1. Ранняя история — точный префикс более поздней истории.
        assert _rows(l)[: len(e)] == _rows(e), (
            f"user {user_id}: {earlier_name} не является префиксом {later_name}"
        )

        # 2. Всё, что добавлено в later, идёт строго позже earlier.
        later_only = l.iloc[len(e):]
        if len(later_only) == 0:
            continue
        checked_with_new_events += 1
        earlier_max = int(e["timestamp"].max())
        later_only_min = int(later_only["timestamp"].min())
        assert earlier_max < later_only_min, (
            f"user {user_id}: {earlier_name} timestamp {earlier_max} "
            f">= {later_name}-only timestamp {later_only_min}"
        )

    print(
        f"  {earlier_name} -> {later_name}: users={len(common_users)}, "
        f"с новыми событиями={checked_with_new_events}, нарушений=0"
    )
    return len(common_users)


def test_train_is_earlier_than_test():
    """Ключевая проверка: Train содержит только более ранние события, чем Test."""
    train = _load(TRAIN_PATH)
    test = _load(TEST_PATH)
    _check_split(train, test, "train", "test")


def test_train_is_earlier_than_validation():
    train = _load(TRAIN_PATH)
    val = _load(VAL_PATH)
    _check_split(train, val, "train", "validation")


def test_validation_is_earlier_than_test():
    val = _load(VAL_PATH)
    test = _load(TEST_PATH)
    _check_split(val, test, "validation", "test")


def test_split_files_are_sorted_by_time():
    for path in (TRAIN_PATH, VAL_PATH, TEST_PATH):
        if not path.exists():
            pytest.skip(f"Нет файла {path}")
        df = pd.read_csv(path)
        assert df["timestamp"].is_monotonic_increasing, f"{path} не отсортирован по времени"


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", "-s", __file__]))
