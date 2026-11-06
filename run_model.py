import json
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
import pandas as pd

from src.dataset import get_dataloaders
from src.models.sequence import SequentialEncoder
from src.optimizer import BasicOptimizer
from src.utils import (
    create_logger,
    fix_random_seed,
    save_metrics,
    train,
    unpack_dataset,
)

logger = create_logger(name=__name__)


def prepare_data(cfg, stage):
    all_data, train_path, val_path, train_val_path, test_path = unpack_dataset(
        cfg["dataset"]
    )

    dataset_meta = {
        "num_users": all_data["user_id"].max(),
        "num_items": all_data["item_id"].max(),
        "max_sequence_length": cfg["dataset"]["max_sequence_length"],
    }

    if stage == "select_epoch":
        dataloaders = get_dataloaders(
            cfg, dataset_meta, train_path, val_path, test_path
        )
    else:
        dataloaders = get_dataloaders(
            cfg, dataset_meta, train_val_path, val_path, test_path
        )
        dataloaders["validation"] = None

    return dataloaders, dataset_meta


def build_sasrec(cfg, dataset_meta):
    _embedding_dim = cfg["model"]["embedding_dim"]
    _num_items = dataset_meta["num_items"]
    _max_sequence_length = dataset_meta["max_sequence_length"]

    item_embeddings = nn.Embedding(
        num_embeddings=_num_items + 2,
        embedding_dim=_embedding_dim,
        padding_idx=0,
    )
    position_embeddings = nn.Embedding(
        num_embeddings=_max_sequence_length
        + 1,  # in order to include `max_sequence_length` value
        embedding_dim=_embedding_dim,
    )
    model = SequentialEncoder(
        **cfg["model"],
        position_embeddings=position_embeddings,
        item_embeddings=item_embeddings,
        num_items=_num_items,
    )

    return model


def build_metrics(cfg, dataset_meta, item2num_interactions=None):
    metrics = {}
    for k in cfg["metrics_ks"]:
        for name, metric_cfg in cfg["metrics"].items():
            metrics[f"{name}@{k}"] = instantiate(
                metric_cfg,
                k=k,
                num_items=dataset_meta["num_items"],
                num_users=dataset_meta["num_users"],
                item2num_interactions=item2num_interactions,
            )
    return metrics


def build_model(model_name, config, dataset_meta):
    assert model_name == "sasrec"
    return build_sasrec(config, dataset_meta)


@hydra.main(version_base=None, config_path="configs", config_name="beauty_sasrec")
def main(cfg):
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    config = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.save(config, output_dir / "config.yaml")
    logger.info(f"Training config:\n{OmegaConf.to_yaml(config)}\n")

    if "best_epoch" not in cfg:
        logger.info(f"SELECT_EPOCH STEP")
        history, best = run_train(config, stage="select_epoch")
        with open(output_dir / "select_epoch_best_metrics.json", "w") as f:
            json.dump(best, f, indent=2)
        with open(output_dir / "select_epoch_all_metrics.json", "w") as f:
            json.dump(history, f, indent=2)
        (output_dir / "only_train").mkdir()
        save_metrics(history, output_dir / "only_train")

        best_epoch = best_epoch = best["best_epoch"]
    else:
        best_epoch = cfg["best_epoch"]

    logger.info(f"REFIT STEP")
    history2, best2 = run_train(config, stage="refit", best_epoch=best_epoch)
    best2 |= history2[-1]
    with open(output_dir / "refit_best_metrics.json", "w") as f:
        json.dump(best2, f, indent=2)
    with open(output_dir / "refit_all_metrics.json", "w") as f:
        json.dump(history2, f, indent=2)
    (output_dir / "train_val").mkdir()
    save_metrics(history2, output_dir / "train_val")


def run_train(config, stage=None, best_epoch=None):
    assert stage in ["refit", "select_epoch"]
    if stage == "refit":
        assert best_epoch is not None, "There isn't best_epoch"

    model_name = config["model_name"]
    assert model_name == "sasrec"

    fix_random_seed(config.get("seed", 42))

    config["model"]["topk_k"] = max(config["metrics_ks"])
    device = config["device"]

    logger.info(f"Current DEVICE: {device}")

    dataloaders, dataset_meta = prepare_data(config, stage)

    item2num_interactions = None
    from pathlib import Path
    data_dir = Path(config["dataset"]["path_to_data_dir"])
    dataset_name = config["dataset"]["name"]
    train_path = data_dir / "global_split" / dataset_name / "train.csv"
    if train_path.exists():
        train_df = pd.read_csv(train_path)
        item2num_interactions = train_df.groupby("item_id").size().to_dict()

    model = build_model(model_name, config, dataset_meta)
    model = model.to(device)

    loss_function = instantiate(config["loss"])
    optimizer = BasicOptimizer.create_from_config(
        config["optimizer"],
        model=model,
        num_layers=config["model"]["num_layers"],
    )
    logger.info(f"Num params: {sum(p.numel() for p in model.parameters())}")
    logger.debug("Everything is ready for training process!")

    metrics = build_metrics(config, dataset_meta, item2num_interactions)

    # Train process
    all_metrics_list, metrics = train(
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders["validation"],
        test_dataloader=dataloaders["test"],
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        num_epochs=config["num_epochs"] if best_epoch is None else best_epoch,
        early_stopping_rounds=config["early_stopping_rounds"],
        device=device,
        metrics=metrics,
    )

    return all_metrics_list, metrics


if __name__ == "__main__":
    main()
