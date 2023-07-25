"Collect data for analysis."
from typing import List
from pathlib import Path

import pandas as pd

from genpred.utils.paths import PROJECT_ROOT, DATA_ROOT


HUMANIZE = {
    "accuracy": "Accuracy",
    "roc_auc": "AUROC",
    "f1_weighted": "Weighted F1",
    "matthews_corrcoef": "MCC",
}
# pylint:disable=logging-fstring-interpolation


def load_gene_list(filename: str) -> List[str]:
    """Loads a list of genes."""
    genes_folder = DATA_ROOT / "genes"
    with open(genes_folder / filename, "r", encoding="utf-8") as file:
        genes = file.read().split("\n")
    return genes[:-1] if genes[-1] == "" else genes


def get_num_features(exp_root: Path) -> int:
    """Gets number of features from feature importance file."""
    return pd.read_csv(exp_root / "importances.csv").shape[0]


def collect_feature_scan_data(
    dataset: str,
    vocab_size: int,
    strategy: str,
    metric: str = "accuracy",
) -> pd.DataFrame:
    """Collects feature scan data."""
    exp_root = PROJECT_ROOT / "experiments" / dataset / f"{vocab_size}" / strategy

    rows = []
    base = pd.read_csv(exp_root / "metrics.csv").iloc[0].to_dict()
    for stage in ["train", "val", "test"]:
        rows.append(
            {
                "Dataset": dataset,
                "Vocab size": vocab_size,
                "Strategy": strategy,
                "Num. Features": get_num_features(exp_root),
                "Stage": stage.capitalize(),
                HUMANIZE[metric]: base[f"{stage}_{metric}"],
            }
        )

    for fold_name in [2**i for i in range(1, 15)]:
        fs_fold = exp_root / "feature_scan" / f"{fold_name}"
        fs_data = pd.read_csv(fs_fold / "metrics.csv").iloc[0].to_dict()
        for stage in ["train", "val", "test"]:
            rows.append(
                {
                    "Dataset": dataset.capitalize(),
                    "Vocab size": vocab_size,
                    "Strategy": strategy.capitalize(),
                    "Num. Features": fold_name,
                    "Stage": stage.capitalize(),
                    HUMANIZE[metric]: fs_data[f"{stage}_{metric}"],
                }
            )
    data = pd.DataFrame(rows).sort_values("Num. Features")
    return data.reset_index(drop=True)


def collect_vocab_sens_analysis(
    dataset: str,
    strategy: str,
    metric: str = "accuracy",
) -> pd.DataFrame:
    """Collects results for sensitivity analysis of vocab_size."""
    rows = []
    for vocab_size in [4000, 8000, 16000, 32000]:
        exp_root = PROJECT_ROOT / "experiments" / dataset / f"{vocab_size}" / strategy
        base = pd.read_csv(exp_root / "metrics.csv").iloc[0].to_dict()
        for stage in ["train", "val", "test"]:
            rows.append(
                {
                    "Dataset": dataset,
                    "Vocab size": vocab_size,
                    "Strategy": strategy,
                    "Num. Features": get_num_features(exp_root),
                    "Stage": stage.capitalize(),
                    HUMANIZE[metric]: base[f"{stage}_{metric}"],
                }
            )

    return pd.DataFrame(rows)
