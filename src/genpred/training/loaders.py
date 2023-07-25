"""Utils to load data."""

import random
import itertools
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.utils import Bunch


from Bio import SeqIO
from joblib import Parallel, delayed
from compress_pickle import load

from genpred.utils.paths import PROJECT_ROOT, DATA_ROOT
from genpred.utils.sequences import clean_seq

# pylint:disable=invalid-name


def load_paths(
    name: str,
    vectorizer: str,
    vocab_size: int = 32000,
    partition: Optional[str] = None,
) -> Bunch:
    """
    Loads dataframe converting paths into Path objects. Optionally
    selects genomes from a partition (training, validation, or test).
    """
    path = DATA_ROOT / name / "genomes.csv"
    dataframe = pd.read_csv(path)

    assert "Path" in dataframe.columns
    dataframe.Path = dataframe.Path.apply(Path)

    if partition is not None:
        assert "Partition" in dataframe.columns
        dataframe = dataframe[dataframe.Partition == partition].reset_index(drop=True)

    save_dir = DATA_ROOT / name / vectorizer / f"{vocab_size}"
    return Bunch(paths=dataframe.Path.tolist(), save_dir=save_dir)


def _process_contigs(path: Path) -> List:
    result = []

    for contig in SeqIO.parse(path, "fasta"):
        seq = str(contig.seq)

        if len(seq) < 500:
            continue

        length = np.random.choice(range(500, min(len(seq) + 1, 1001)))
        start = np.random.choice(range(len(seq) + 1 - length))
        subseq = seq[start : start + length]
        result.append(clean_seq(subseq))

    return result


def load_corpus(paths: List[Path], num_sentences: int = 1_000_000) -> List:
    """Loads corpus to train the tokenizer."""
    contigs_list = Parallel(verbose=1, n_jobs=-1)(
        delayed(_process_contigs)(p) for p in paths
    )
    contigs = list(itertools.chain.from_iterable(contigs_list))
    return random.sample(contigs, k=num_sentences)


def _split_data(num_train, num_val) -> List:
    split_train = np.arange(num_train)
    split_val = np.arange(num_train, num_train + num_val)
    return [(split_train, split_val)]


def _load_array(data_path, partition) -> Tuple:
    array = load(data_path / f"{partition}.gz")
    return array, array.shape[0]


def _load_genomes(frame, partition) -> np.ndarray:
    data = frame[frame.Partition == partition]
    return data.Path.apply(lambda x: Path(x).stem).values


def _load_labels(frame, partition) -> np.ndarray:
    data = frame[frame.Partition == partition]
    return data.Label.values


def load_training_data(data_path: Path, frame: pd.DataFrame) -> Tuple:
    """Load training data."""
    X_train, num_train = _load_array(data_path, "training")
    X_val, num_val = _load_array(data_path, "validation")
    X_develop = sp.vstack([X_train, X_val])

    y_train = _load_labels(frame, "training")
    y_val = _load_labels(frame, "validation")
    y_develop = np.hstack([y_train, y_val])

    G_train = _load_genomes(frame, "training")
    G_val = _load_genomes(frame, "validation")
    G_develop = np.hstack([G_train, G_val])

    splits = _split_data(num_train, num_val)
    return X_develop, y_develop, G_develop, splits


def load_test_data(data_path: Path, frame: pd.DataFrame) -> Tuple:
    """Load test data."""
    X_test, _ = _load_array(data_path, "test")
    y_test = _load_labels(frame, "test")
    G_test = _load_genomes(frame, "test")
    return X_test, y_test, G_test


def load_dataset(name: str, vectorizer: str, vocab_size: int) -> Bunch:
    """Loads dataset."""
    data_path = PROJECT_ROOT / "data" / name / vectorizer / f"{vocab_size}"
    frame = pd.read_csv(PROJECT_ROOT / "data" / name / "genomes.csv")

    X_develop, y_develop, G_develop, splits = load_training_data(data_path, frame)
    X_test, y_test, G_test = load_test_data(data_path, frame)

    return Bunch(
        X_train=X_develop,
        y_train=y_develop,
        G_train=G_develop,
        X_test=X_test,
        y_test=y_test,
        G_test=G_test,
        split=splits,
    )
