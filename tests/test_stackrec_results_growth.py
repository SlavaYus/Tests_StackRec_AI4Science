"""
Проверка роста метрик StackRec после второго этапа.

Сравниваются этапы:
  2_layers (этап 1) -> 4_layers (этап 2).

Основная метрика выбора чекпоинта — ndcg@10. Дополнительно проверяются
recall@10, ndcg@100 и recall@100. Для них после второго этапа значения
должны быть строго больше, чем после первого.

Запуск:
    python3 tests/test_stackrec_results_growth.py
    pytest -q -s tests/test_stackrec_results_growth.py
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULT_FILES = [
    PROJECT_ROOT / "results" / "sources" / "sasrec_stackrec_results.json",
    PROJECT_ROOT / "outputs" / "stackrec_adamw" / "stackrec_results.json",
]

KEY_METRICS = ["ndcg@10", "recall@10", "ndcg@100", "recall@100"]


def _load(path):
    if not path.exists():
        pytest.skip(f"Нет файла результатов: {path}")
    with open(path) as f:
        return json.load(f)


def _key_metrics(stages, stage_name):
    stage = stages[stage_name]
    km = stage.get("best_complete_key_metrics")
    if km is None:
        km = {
            "val": stage["best_complete_val_metrics"],
            "test": stage["best_complete_test_metrics"],
        }
    return km


@pytest.mark.parametrize("path", RESULT_FILES, ids=lambda p: p.parent.name)
def test_stage_2_metric_grows(path):
    data = _load(path)
    stages = data["stages"]

    stage_1 = _key_metrics(stages, "2_layers")
    stage_2 = _key_metrics(stages, "4_layers")

    print(f"\n{path}")
    for split in ("val", "test"):
        for metric in KEY_METRICS:
            before = stage_1[split][metric]
            after = stage_2[split][metric]
            print(
                f"  {split:4s} {metric:10s}: "
                f"{before:.6f} -> {after:.6f} "
                f"({'UP' if after > before else 'DOWN'})"
            )
            assert after > before, (
                f"{path}: {split}/{metric} не вырос после второго этапа: "
                f"{before:.6f} -> {after:.6f}"
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", "-s", __file__]))
