import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE_NUM_LAYERS = 4
ADAMW_LR = 0.0021002963516611
MUON_LR = 0.02
LAYERS = [2, 4, 6]

VARIANTS = {
    "adamw": {
        "output_pattern": "outputs/adamw_l{num_layers}",
        "muon": False,
        "description": "AdamW for all parameters.",
    },
    "muon": {
        "output_pattern": "outputs/muon_l{num_layers}",
        "muon": True,
        "description": (
            "Muon for 2D hidden weights; AdamW for embeddings and 1D weights."
        ),
    },
    "muon_all2d": {
        "output_pattern": "outputs/muon_all2d_l{num_layers}",
        "muon": True,
        "description": "Muon for every 2D weight; AdamW for 1D weights.",
    },
}

SUMMARY_METRICS = [
    "test/ndcg@10",
    "test/recall@10",
    "test/ndcg@100",
    "test/recall@100",
]


def lr_scale(num_layers):
    return math.sqrt(REFERENCE_NUM_LAYERS / num_layers)


def collect():
    results = {
        "description": (
            "SASRec on Beauty. Only num_layers (2, 4, 6) and the optimizer "
            "were varied. Learning rate is scaled as "
            "lr(L) = lr_ref * sqrt(4 / L)."
        ),
        "fixed_hyperparameters": {
            "dataset": "beauty",
            "dataset_name": "reviews_Beauty_5",
            "model_name": "sasrec",
            "seed": 42,
            "embedding_dim": 64,
            "num_heads": 1,
            "dim_feedforward": 64,
            "dropout": 0.5,
            "activation": "gelu",
            "layer_norm_eps": 1e-09,
            "max_sequence_length": 10,
            "clip_grad_threshold": 10.0,
            "loss": "src.loss.LocalObjective",
            "train_batch_size": 128,
            "num_epochs": 200,
            "early_stopping_rounds": 30,
            "metrics_ks": [10, 100],
            "lr_scaling": "lr(L) = lr_ref * sqrt(4 / L)",
            "adamw_lr_ref": ADAMW_LR,
            "muon_lr_ref": MUON_LR,
        },
        "experiments": {},
        "summary": {},
    }

    for variant_name, variant in VARIANTS.items():
        results["experiments"][variant_name] = {}
        results["summary"][variant_name] = {}
        for num_layers in LAYERS:
            output_dir = ROOT / variant["output_pattern"].format(
                num_layers=num_layers
            )
            with open(output_dir / "select_epoch_best_metrics.json") as f:
                select_metrics = json.load(f)
            with open(output_dir / "refit_best_metrics.json") as f:
                refit_metrics = json.load(f)

            scale = lr_scale(num_layers)
            lrs = {"adamw_lr": ADAMW_LR * scale}
            if variant["muon"]:
                lrs["muon_lr"] = MUON_LR * scale

            experiment = {
                "num_layers": num_layers,
                "optimizer": variant_name,
                "optimizer_description": variant["description"],
                "output_dir": str(output_dir.relative_to(ROOT)),
                "effective_lrs": lrs,
                "selected_best_epoch": select_metrics["best_epoch"],
                "select_train_time_sec": select_metrics["train_time"],
                "validation_metrics": select_metrics,
                "refit_train_time_sec": refit_metrics["train_time"],
                "test_metrics": refit_metrics,
            }
            results["experiments"][variant_name][str(num_layers)] = experiment

            summary = {
                "num_layers": num_layers,
                "effective_lrs": lrs,
                "best_epoch": select_metrics["best_epoch"],
                "select_train_time_sec": select_metrics["train_time"],
                "refit_train_time_sec": refit_metrics["train_time"],
            }
            for metric_name in SUMMARY_METRICS:
                summary[f"val/{metric_name.split('/', 1)[1]}"] = select_metrics[
                    metric_name.replace("test/", "val/")
                ]
                summary[metric_name] = refit_metrics[metric_name]
            results["summary"][variant_name][str(num_layers)] = summary

    with open(ROOT / "sasrec_depth_lr_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    collect()
