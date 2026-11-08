# StackRec: масштабирование SASRec и тесты

**Автор репозитория: Вячеслав Юсупов.**

Исследовательский репозиторий, посвящённый прогрессивному масштабированию
последовательных рекомендательных моделей на архитектуре SASRec: StackRec,
оптимизаторы (AdamW, Muon), гиперболические головы и вспомогательные
регуляризации. Эксперименты проведены на датасете Amazon Beauty
(`reviews_Beauty_5`).

## Что внутри

- `run_model.py` — обучение обычного SASRec с заданным числом слоёв.
- `run_stackrec.py` — обучение StackRec: одна модель с 6 слоями постепенно
  раскрывается как 2 → 4 → 6 активных слоёв с копированием весов и ramp-коэффициентами.
- `src/` — модели, датасет, метрики, функции потерь и оптимизаторы.
- `configs/` — Hydra-конфигурации экспериментов.
- `tests/` — тесты датасета, метрик и механики StackRec.
- `results/` — объединённые результаты экспериментов (`all_results.json` и `sources/`).
- `report.md` / `report.pdf` — отчёт по методам и результатам.
- `diary.md`, `AGENT.md`, `RUN.txt` — журнал работы и инструкции по запуску.

## Установка

Требуется Python 3.10+ и пакеты:

```bash
pip install torch pandas numpy hydra-core omegaconf tqdm matplotlib seaborn pytest
```

Данные должны находиться в `data/` (файл `reviews_Beauty_5.csv` и готовые
сплиты в `data/global_split/reviews_Beauty_5/`).

## Запуск экспериментов

Обычный SASRec (число слоёв задаётся через `model.num_layers`):

```bash
python run_model.py -cn beauty_sasrec model.num_layers=4 \
    hydra.run.dir=outputs/sasrec_l4 hydra.job.chdir=false
```

StackRec с AdamW (этапы 2 → 4 → 6 слоёв):

```bash
python run_stackrec.py -cn beauty_sasrec_stackrec
```

StackRec с Muon:

```bash
python run_stackrec.py -cn beauty_sasrec_stackrec_muon
```

StackRec с гиперболическими головами / Lorentz-остатками:

```bash
python run_stackrec.py -cn beauty_sasrec_stackrec_hyperbolic
python run_stackrec.py -cn beauty_sasrec_stackrec_hyperbolic_residual
```

Доступные конфигурации: `configs/beauty_sasrec*.yaml`. Устройство задаётся
параметром `device` в `configs/general.yaml` (по умолчанию `cuda`).

## Тесты

```bash
pytest -q tests/
```

Отдельные проверки (без GPU):

```bash
python3 tests/test_dataset_split.py
python3 tests/test_metrics.py
python3 tests/test_stackrec.py
python3 tests/test_stackrec_stage_copy.py
```

## Результаты

Кратко (Amazon Beauty, seed 42, test split): StackRec с AdamW показывает
test NDCG@10 = 0.013299 для 6 слоёв против 0.005645 у обычного SASRec, а для
4 слоёв — 0.012964 против 0.004797. Подробные таблицы, описание методов и
настроек — в `report.md` и `results/all_results.json`.
