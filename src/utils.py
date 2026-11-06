import argparse
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from tqdm import trange

sns.set_theme()

from .metrics import StatefullMetric




def create_logger(
    name,
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
):
    logging.basicConfig(level=level, format=format, datefmt=datefmt)
    logger = logging.getLogger(name)
    return logger


logger = create_logger(name=__name__)


def move_batch(batch, device):
    for key, value in batch.items():
        batch[key] = value.to(device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)

    return params


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_masked_tensor(data, lengths, device):
    batch_size = lengths.shape[0]
    max_sequence_length = lengths.max().item()

    padded_embeddings = torch.zeros(
        batch_size,
        max_sequence_length,
        data.shape[-1],
        dtype=torch.float,
        device=device,
    )  # (batch_size, max_seq_len, emb_dim)

    mask = (
        torch.arange(end=max_sequence_length, device=device)[None].tile([batch_size, 1])
        < lengths[:, None]
    )  # (batch_size, max_seq_len)

    padded_embeddings[mask] = data

    return padded_embeddings, mask


def get_activation_function(name, **kwargs):
    if name == "relu":
        return torch.nn.ReLU()
    elif name == "gelu":
        return torch.nn.GELU()
    elif name == "elu":
        return torch.nn.ELU(alpha=float(kwargs.get("alpha", 1.0)))
    elif name == "leaky":
        return torch.nn.LeakyReLU(
            negative_slope=float(kwargs.get("negative_slope", 1e-2))
        )
    elif name == "sigmoid":
        return torch.nn.Sigmoid()
    elif name == "tanh":
        return torch.nn.Tanh()
    elif name == "softmax":
        return torch.nn.Softmax()
    elif name == "softplus":
        return torch.nn.Softplus(
            beta=int(kwargs.get("beta", 1.0)),
            threshold=int(kwargs.get("threshold", 20)),
        )
    elif name == "softmax_logit":
        return torch.nn.LogSoftmax()
    else:
        raise ValueError("Unknown activation function name `{}`".format(name))


# def inference(dataloader, model, metrics, device):
#     running_metrics = {}
#     for metric_name, metric_function in metrics.items():
#         running_metrics[metric_name] = []

#     model.eval()

#     with torch.no_grad():
#         for idx, batch in enumerate(dataloader):
#             for key, value in batch.items():
#                 batch[key] = value.to(device)
#             batch["logits"] = model(batch)

#             for key, values in batch.items():
#                 batch[key] = values.cpu()

#             for metric_name, metric_function in metrics.items():
#                 running_metrics[metric_name].extend(metric_function(inputs=batch))
#                 metric_output = metric_function(inputs=batch)


#                 if isinstance(metric_output, dict):
#                     for sub_name, sub_values in metric_output.items():
#                         # ИСПРАВЛЕНИЕ: правильное формирование имени метрики
#                         if sub_name == metric_name.split("@")[0]:  # общая метрика (например "ndcg")
#                             # Сохраняем как есть: ndcg@10
#                             full_name = metric_name
#                         else:  # групповая метрика (например "ndcg_group0")
#                             # Формируем: ndcg@10_group0
#                             base_name = metric_name.split("@")[0]  # "ndcg"
#                             k_value = metric_name.split("@")[1] if "@" in metric_name else ""  # "10"
#                             group_suffix = sub_name.replace(base_name, "")  # "_group0"
#                             full_name = f"{base_name}@{k_value}{group_suffix}"  # "ndcg@10_group0"
                        
#                         if full_name not in running_metrics:
#                             running_metrics[full_name] = []
#                         running_metrics[full_name].extend(sub_values)
#                 else:
#                     # Старое поведение: метрика возвращает список значений
#                     running_metrics[metric_name].extend(metric_output)

#     # print("Inference procedure has been finished!")
#     # print("Metrics are the following:")
#     results = {}
#     for metric_name, metric_value in running_metrics.items():
#         results[metric_name] = np.mean(metric_value)
#         # print("{}: {}".format(metric_name, np.mean(metric_value)))
#     # print("Metrics finished!")
#     model.train()
#     return results

def inference(dataloader, model, metrics, device):
    running_metrics = {}
    for metric_name, metric_function in metrics.items():
        running_metrics[metric_name] = []

    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            for key, value in batch.items():
                batch[key] = value.to(device)
            batch["logits"] = model(batch)

            for key, values in batch.items():
                batch[key] = values.cpu()

            for metric_name, metric_function in metrics.items():
                # Вызываем метрику только один раз
                metric_output = metric_function(inputs=batch)
                
                # Обработка: метрика может вернуть dict с несколькими значениями
                if isinstance(metric_output, dict):
                    for sub_name, sub_values in metric_output.items():
                        # Формируем правильное имя метрики.
                        # Сравниваем имена без underscore, чтобы "meanrank"
                        # из конфига и "mean_rank" из метрики совпадали.
                        base_name = metric_name.split("@")[0]  # "ndcg"
                        base_norm = base_name.replace("_", "")
                        sub_norm = sub_name.replace("_", "")
                        if sub_norm == base_norm:
                            full_name = metric_name  # "ndcg@10"
                        elif "_group" in sub_name:
                            # sub_name = "ndcg_group0" / "mean_rank_group0"
                            suffix = sub_name[sub_name.index("_group"):]
                            full_name = f"{metric_name}{suffix}"  # "ndcg@10_group0"
                        else:
                            suffix = sub_name.replace(base_name, "")
                            full_name = f"{metric_name}{suffix}"
                        
                        if full_name not in running_metrics:
                            running_metrics[full_name] = []
                        # Добавляем только если значения — числа
                        if isinstance(sub_values, (list, tuple)) and len(sub_values) > 0:
                            if isinstance(sub_values[0], (int, float, np.number)):
                                running_metrics[full_name].extend(sub_values)
                else:
                    # Старое поведение: метрика возвращает список значений
                    if isinstance(metric_output, (list, tuple)) and len(metric_output) > 0:
                        if isinstance(metric_output[0], (int, float, np.number)):
                            running_metrics[metric_name].extend(metric_output)

        # Reduce для StatefullMetric
        for metric_name, metric_function in metrics.items():
            if isinstance(metric_function, StatefullMetric):
                # ИСПРАВЛЕНИЕ 1: проверяем, что значение ещё список перед len()
                if metric_name in running_metrics and isinstance(running_metrics[metric_name], list) and len(running_metrics[metric_name]) > 0:
                    running_metrics[metric_name] = metric_function.reduce(
                        running_metrics[metric_name]
                    )
                # Для групповых метрик
                for key in list(running_metrics.keys()):
                    if key.startswith(metric_name.split("@")[0] + "@") and key != metric_name:
                        # ИСПРАВЛЕНИЕ 2: проверяем тип перед len()
                        if isinstance(running_metrics[key], list) and len(running_metrics[key]) > 0:
                            running_metrics[key] = metric_function.reduce(
                                running_metrics[key]
                            )

    results = {}
    for metric_name, metric_value in running_metrics.items():
        # ИСПРАВЛЕНИЕ 3: безопасное вычисление среднего
        if isinstance(metric_value, (list, tuple)) and len(metric_value) > 0:
            if isinstance(metric_value[0], (int, float, np.number)):
                results[metric_name] = float(np.mean(metric_value))
            else:
                results[metric_name] = 0.0
        elif isinstance(metric_value, (int, float)):
            # Уже reduced значение — используем как есть
            results[metric_name] = float(metric_value)
        else:
            results[metric_name] = 0.0
    
    model.train()
    return results


def train(
    train_dataloader,
    val_dataloader,
    test_dataloader,
    model,
    optimizer,
    loss_function,
    num_epochs,
    early_stopping_rounds,
    device,
    metrics,
):
    logger.debug("Start training...")

    train_start = time.time()

    best_epoch = 0
    best_val_metric = 0.0
    best_metrics = {}
    history = []

    for epoch_num in trange(num_epochs):
        logger.debug(f"Start epoch {epoch_num}")

        for step, batch in enumerate(train_dataloader):
            model.train()
            move_batch(batch, device)
            out = model(batch)
            loss_inp = {**batch, **out, "epoch_num": epoch_num, "step": step}
            loss = loss_function(loss_inp)
            optimizer.step(loss)

        test_metrics = inference(test_dataloader, model, metrics, device)

        epoch_metrics = {}

        if val_dataloader is not None:
            val_metrics = inference(val_dataloader, model, metrics, device)

            epoch_metrics.update(
                {
                    f"val/{metric_name}": metric_value
                    for metric_name, metric_value in val_metrics.items()
                }
            )

            val_ndcg = val_metrics["ndcg@10"]
            if val_ndcg > best_val_metric:
                best_metrics = epoch_metrics
                best_val_metric = val_ndcg
                best_epoch = epoch_num
            elif epoch_num - best_epoch > early_stopping_rounds:
                print(f"No more improve in {early_stopping_rounds} epoch")
                break

        epoch_metrics.update(
            {
                f"test/{metric_name}": metric_value
                for metric_name, metric_value in test_metrics.items()
            }
        )
        history.append(epoch_metrics)

    train_end = time.time()

    best_metrics = best_metrics.copy()
    best_metrics["train_time"] = train_end - train_start
    best_metrics["best_epoch"] = best_epoch

    print("Total time:", best_metrics["train_time"])
    print(f"Best metrics: \n{best_metrics}")

    return history, best_metrics


# def save_metrics(all_metrics: list[dict], output_dir: str | Path):
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     metrics_per_split = defaultdict(lambda: defaultdict(list))

#     for epoch_metrics in all_metrics:
#         for key, value in epoch_metrics.items():
#             split, metric_name = key.split("/", 1)
#             metrics_per_split[split][metric_name].append(value)

#     metric_names = sorted(
#         {name for split in metrics_per_split.values() for name in split}
#     )

#     def plot_metric(metric_name):
#         plt.figure(figsize=(6, 4))
#         for split, metrics in metrics_per_split.items():
#             if metric_name in metrics:
#                 plt.plot(
#                     range(1, len(metrics[metric_name]) + 1),
#                     metrics[metric_name],
#                     marker="o",
#                     label=split,
#                 )
#         plt.title(metric_name)
#         plt.xlabel("Epoch")
#         plt.ylabel(metric_name)
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()
#         save_path = output_dir / f"{metric_name.replace('@', '_at_')}.png"
#         plt.savefig(save_path, dpi=200)
#         plt.close()

#     for name in metric_names:
#         plot_metric(name)

#     print(f"✅ Saved {len(metric_names) + 1} plots to {output_dir.resolve()}")

def save_metrics(all_metrics: list[dict], output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Проверка на пустой список
    if len(all_metrics) == 0:
        logger.warning("No metrics to save, creating empty directory")
        return

    metrics_per_split = defaultdict(lambda: defaultdict(list))

    for epoch_metrics in all_metrics:
        for key, value in epoch_metrics.items():
            split, metric_name = key.split("/", 1)
            metrics_per_split[split][metric_name].append(value)

    metric_names = sorted(
        {name for split in metrics_per_split.values() for name in split}
    )

    def plot_metric(metric_name):
        plt.figure(figsize=(6, 4))
        for split, metrics in metrics_per_split.items():
            if metric_name in metrics:
                plt.plot(
                    range(1, len(metrics[metric_name]) + 1),
                    metrics[metric_name],
                    marker="o",
                    label=split,
                )
        plt.title(metric_name)
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # ИСПРАВЛЕНИЕ: заменяем @ и / на безопасные символы
        safe_name = metric_name.replace('@', '_at_').replace('/', '_')
        save_path = output_dir / f"{safe_name}.png"
        plt.savefig(save_path, dpi=200)
        plt.close()

    for name in metric_names:
        plot_metric(name)

    print(f"✅ Saved {len(metric_names) + 1} plots to {output_dir.resolve()}")


def compute_item_distance_matrix(df, num_items, num_users):
    """
    df must contain columns: item_id, user_id, rating (0/1 or rating)
    Returns: (num_items × num_items) cosine distance matrix
    """

    mat = torch.zeros((num_items, num_users), dtype=torch.float32)
    mat[df["item_id"], df["user_id"]] = torch.tensor(
        df["rating"].values, dtype=torch.float32
    )

    mat = torch.nn.functional.normalize(mat, p=2, dim=1)  # (I × U)

    sim = mat @ mat.T

    dist = 1 - sim

    return dist


def unpack_dataset(dataset_config):
    data_folder = Path(dataset_config["path_to_data_dir"])
    dataset_name = dataset_config["name"]
    all_data = pd.read_csv(data_folder / f"{dataset_name}.csv")
    split_folder = data_folder / "global_split" / dataset_name
    train_path = split_folder / "train.csv"
    val_path = split_folder / "validation.csv"
    train_val_path = split_folder / "train_validation.csv"
    test_path = split_folder / "test.csv"

    return all_data, train_path, val_path, train_val_path, test_path
