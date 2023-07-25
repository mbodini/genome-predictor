"""Utils to preprocess genomes."""
from typing import Callable

import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

import genpred
from genpred.utils.sequences import read_fasta

# pylint:disable=invalid-name
# pylint:disable=redefined-outer-name


def process_disease(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Process disease dataset."""
    labels = [
        "carrier",
        "invasive (unspecified/other)",
        "meningitis",
        "septicaemia",
        "meningitis and septicaemia",
    ]
    filtered_df = filtered_df.copy()
    filtered_df = filtered_df[filtered_df.disease.isin(labels)]
    filtered_df["Label"] = filtered_df.disease.apply(lambda x: int(x != "carrier"))
    return filtered_df[["Path", "Label"]]


def process_capsule(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Process capsule dataset."""
    filtered_df = filtered_df.copy()
    filtered_df = filtered_df[filtered_df.capsule_group != "discrepancy"]
    filtered_df["Label"] = filtered_df.capsule_group.apply(lambda x: int(x == "B"))
    return filtered_df[["Path", "Label"]]


PREPROCESS_FUNCS = {"capsule": process_capsule, "disease": process_disease}


def filter_dataset(
    genomes_df: pd.DataFrame,
    process_func: Callable,
) -> pd.DataFrame:
    """Applies common filters to the dataset."""
    filtered_df = genomes_df[genomes_df.species == "Neisseria meningitidis"]
    filtered_df = filtered_df[filtered_df.Complete == 0]
    return process_func(filtered_df)


def _helper(
    genome_path: str,
    num: int = 10,
) -> dict:
    """
    Calculates the fraction of a genome (accessible with `genome_path`)
    covered by the `num` longest contigs.
    """
    try:
        contigs = [str(c.seq) for c in read_fasta(genome_path)]
        sorted_lengths = sorted([len(cont) for cont in contigs], reverse=True)
        numerator = sum(sorted_lengths[:num])
        denominator = sum(sorted_lengths)
        return {
            "Path": genome_path,
            f"FractionCovered{num}": numerator / denominator,
        }
    except Exception:
        return {}


def calculate_fraction_covered(
    genomes_df: pd.DataFrame,
    num: int = 10,
) -> pd.DataFrame:
    """
    Adds a column to the geneomes in the DataFrame `genomes_file` containing the genome
    fraction covered by the `num` longest contigs.
    """
    results = Parallel(verbose=1, n_jobs=-1)(delayed(_helper)(p, num=num) for p in genomes_df.Path.tolist())
    return pd.merge(genomes_df, pd.DataFrame(results), on="Path")


def balance_quality(
    genomes_df: pd.DataFrame,
    num: int = 10,
    n_clusters: int = 512,
) -> pd.DataFrame:
    """Balances dataset quality."""
    X = genomes_df[f"FractionCovered{num}"].values.reshape(-1, 1)
    genomes_df["Cluster"] = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0).fit_predict(X)

    def sample_indices(data, label, cluster):
        no_points_0 = data[(data.Label == 0) & (data.Cluster == cluster)].shape[0]
        no_points_1 = data[(data.Label == 1) & (data.Cluster == cluster)].shape[0]
        population = data[(data.Label == label) & (data.Cluster == cluster)]
        return population.sample(
            n=min(no_points_0, no_points_1),
            replace=False,
            random_state=0,
        ).index.values

    indices = []
    for cluster in range(n_clusters):
        indices.append(sample_indices(genomes_df, label=0, cluster=cluster))
        indices.append(sample_indices(genomes_df, label=1, cluster=cluster))

    indices = np.concatenate(indices)
    genomes_df["Keep"] = False
    genomes_df.loc[indices, "Keep"] = True
    genomes_df = genomes_df[["Path", "Label", "Keep"]]
    genomes_df = genomes_df[genomes_df.Keep == True]  # noqa: E712

    return genomes_df.drop("Keep", axis=1).reset_index(drop=True)


def split_data(
    genomes_df: pd.DataFrame,
    num_val: int = 500,
    num_test: int = 1000,
) -> pd.DataFrame:
    """Splits data."""
    indices = np.arange(genomes_df.shape[0])
    labels = genomes_df.Label.values
    dev_idx, test_idx = train_test_split(
        indices,
        stratify=labels,
        test_size=num_test,
        random_state=0,
    )
    train_idx, val_idx = train_test_split(
        dev_idx,
        stratify=labels[dev_idx],
        test_size=num_val,
        random_state=0,
    )
    genomes_df["Partition"] = None
    genomes_df.loc[train_idx, "Partition"] = "training"
    genomes_df.loc[val_idx, "Partition"] = "validation"
    genomes_df.loc[test_idx, "Partition"] = "test"
    return genomes_df


@hydra.main(
    config_path=genpred.CFG_ROOT.as_posix(),
    config_name="preprocess.yaml",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    """preprocess function."""
    data = pd.read_csv(genpred.DATA_ROOT / "genomes.csv", low_memory=False)

    if len(data) <= (cfg.num_val + cfg.num_test):
        raise ValueError("Ensure that (num_val + num_test) is strictly less than len(data).")

    data = filter_dataset(data, process_func=PREPROCESS_FUNCS[cfg.dataset.name])
    data = calculate_fraction_covered(data, num=cfg.fraction_covered)
    data = balance_quality(data, num=cfg.fraction_covered, n_clusters=cfg.n_clusters)
    data = split_data(data, num_val=cfg.num_val, num_test=cfg.num_test)

    output_dir = genpred.DATA_ROOT / cfg.dataset.name
    if not (output_dir / "genomes.csv").exists():
        output_dir.mkdir(exist_ok=True, parents=True)
        data.to_csv(output_dir / "genomes.csv", index=False)


if __name__ == "__main__":
    main()
