"""Plots gene knockout."""
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from genpred.utils.paths import EXPS_ROOT, NB_ROOT, DATA_ROOT

# pylint:disable=invalid-name

COLORMAP = sns.color_palette("crest", as_cmap=True)
EXPS_ROOT = Path("/hpc/projects/upt/Genome_classification/experiments")


def _process(df: pd.DataFrame) -> pd.DataFrame:
    gene = df.Gene.unique()[0]
    return pd.DataFrame(
        {"Genome": df.Genome, gene: (df.Original - df.Removed.round(3))}  # / df.Length}
    )


def fetch_data_paths(
    dataset: str,
    vectorizer: str = "sentencepiece",
    vocab_size: Optional[int] = None,
):
    """Fetch dataset paths."""
    root = EXPS_ROOT / "knockout" / dataset / vectorizer / f"{vocab_size}"
    print(root)
    return sorted(root.glob("*NEIS*/predictions.csv"))


def fetch_data(
    dataset: str,
    vectorizer: str = "sentencepiece",
    vocab_size: Optional[int] = None,
):
    """Fetch datasets."""
    nb_path = NB_ROOT / "knockout" / dataset / vectorizer / f"{vocab_size}" / "data.csv"

    if not nb_path.exists():
        nb_path.parent.mkdir(exist_ok=True, parents=True)
        data_paths = fetch_data_paths(dataset, vectorizer, vocab_size)

        data = _process(pd.read_csv(data_paths[0]))
        for path in data_paths[1:]:
            if not data.empty:
                data1 = _process(pd.read_csv(path))
                data = pd.merge(data, data1, on="Genome", how="left")

        data = data.set_index("Genome").transpose().astype(float)
        print(f"Found {data.shape[0]} genes.")
        data.to_csv(nb_path)

    return pd.read_csv(nb_path, index_col=0)


def plot_delta_heatmap(
    data: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 10),
    filename: str = "plot.png",
):
    """Plot heatmap."""
    # most_important = data.mean(1).sort_values(ascending=False)[:200].index.values
    with open("data/genes/all_genes.txt", "r", encoding="utf-8") as file:
        genes = file.read().split("\n")

    data = data.loc[[g for g in genes if g in data.index], :]

    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data, cmap=COLORMAP, ax=ax)

    plt.savefig(filename)
    plt.clf()

    return data


def plot_delta_dendogram(
    data: pd.DataFrame,
    method: str = "average",
    row_cluster: bool = True,
    filename: str = "dendogram.png",
    figsize: Tuple[int, int] = (20, 40),
    col_colors: Optional[List] = None,
    vmin: float = 0,
) -> None:
    """Plot dendogram of prediction deltas."""
    sns.clustermap(
        data,
        method=method,
        row_cluster=row_cluster,
        figsize=figsize,
        cmap=COLORMAP,
        col_colors=col_colors,
        vmin=vmin,
    )

    plt.savefig(filename)
    plt.clf()


def process_data(
    data: pd.DataFrame, na_thresh: float = 0.8, k: Optional[int] = None
) -> pd.DataFrame:
    """Process data for plot."""
    data[data < 0] = 0

    threshold = int(len(data.columns) * na_thresh)
    data = data.dropna(axis=0, thresh=threshold)

    data = data.loc[:, data.sum(0) > 0]

    def scale(df: pd.DataFrame) -> pd.DataFrame:
        return (df - df.min()) / (df.max() - df.min() + 1e-12)

    data = scale(data)

    if k is not None:
        notk = -int(k)
        important = data.mean(1).dropna().sort_values()[notk:].index
        data = data.loc[important, :]

    return data


def get_colors(data: pd.DataFrame):
    """Colors for clonal complex."""
    genomes = [int(c.split("_")[0]) for c in data.columns]
    cc = pd.read_csv(DATA_ROOT / "cc.csv")
    cc = cc[cc.id.isin(genomes)]

    def assign_col(val):
        palette = sns.color_palette(as_cmap=True)
        if val == "ST-11 complex":
            return palette[0]
        if val == "ST-41/44 complex":
            return palette[1]
        if val == "ST-32 complex":
            return palette[2]
        if val == "ST-23 complex":
            return palette[3]
        if val == "ST-213 complex":
            return palette[4]
        return "k"

    return cc["clonal_complex (MLST)"].map(assign_col).values


def fetch_deltas(
    dataset: str,
    vectorizer: str = "sentencepiece",
    vocab_size: int = 32000,
):
    """Fetches deltas from experiments folder."""
    root = EXPS_ROOT / "knockout" / dataset / vectorizer / f"{vocab_size}"

    dfs, columns = [], []
    for path in sorted(root.glob("*NEIS*/predictions.csv")):
        df = pd.read_csv(path)[["Genome", "Delta"]].set_index("Genome")
        dfs.append(df)
        columns.append(path.parts[-2])

    df = pd.concat(dfs, axis=1, ignore_index=False)
    df.columns = columns
    return df
