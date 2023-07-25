"Gene Knockout."
import os
from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from joblib import Parallel, delayed

import genpred
from genpred.utils.sequences import (
    remove_gene,
    write_fasta,
    read_fasta,
    read_gff3,
    filter_predictions,
)

# pylint:disable=logging-fstring-interpolation
# pylint:disable=invalid-name


def _helper(vectorizer, model, out_dir, row, genes, control):
    genome = row.Fasta.stem
    fasta = read_fasta(row.Fasta)
    gff3 = read_gff3(genpred.GFF3_ROOT / f"{genome}.gff3")

    length, removed_prediction = 0, row.Probability
    for gene in genes:
        fasta, l = remove_gene(fasta, gff3, gene, padding=20, control=control)  # noqa: E741
        length += l

    if length > 0:
        path_removed = out_dir / f"{row.Fasta.stem}-knockout.fasta"
        write_fasta(fasta, path_removed)
        removed_hist = vectorizer.transform([path_removed])
        removed_prediction = model.predict_proba(removed_hist)[:, 1][0]
        os.system(f"rm {path_removed}")

    return genome, row.Probability, removed_prediction, length


def _get_exp_path(cfg: DictConfig) -> Path:
    data = cfg.dataset.name
    vect = cfg.dataset.vectorizer
    vocab_size = cfg.dataset.vocab_size
    exp_root = genpred.PROJECT_ROOT / "experiments"
    exp_root = exp_root / data / vect / f"{vocab_size}"
    return exp_root


def _get_data_path(cfg: DictConfig) -> Path:
    data = cfg.dataset.name
    vect = cfg.dataset.vectorizer
    vocab_size = cfg.dataset.vocab_size
    exp_root = genpred.DATA_ROOT / data / vect / f"{vocab_size}"
    return exp_root


def _get_out_path(cfg: DictConfig, gene: str) -> Path:
    data = cfg.dataset.name
    vect = cfg.dataset.vectorizer
    vocab_size = cfg.dataset.vocab_size
    out_dir = genpred.EXPS_ROOT / "knockout"
    out_dir = out_dir / data / vect / f"{vocab_size}" / gene
    out_dir.mkdir(exist_ok=True, parents=True)
    return out_dir


@hydra.main(
    config_path=genpred.CFG_ROOT.as_posix(),
    config_name="knockout.yaml",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    "Performs knockout."
    exp_root = _get_exp_path(cfg)
    data_root = _get_data_path(cfg)
    out_dir = _get_out_path(cfg, cfg.gene)

    preds = pd.read_csv(exp_root / "predictions.csv")
    preds = filter_predictions(preds, "invasive")
    print(f"Working with {preds.shape[0]} genomes.")

    vectorizer = joblib.load(data_root / "vectorizer.pkl")
    model = joblib.load(exp_root / "model.pkl")

    if not (out_dir / "predictions.csv").exists():
        data = Parallel(verbose=1, n_jobs=-1)(
            delayed(_helper)(vectorizer, model, out_dir, row, [cfg.gene], cfg.control) for row in preds.itertuples()
        )

        genomes, predictions, removed, lengths = zip(*data)
        deltas = np.array(predictions) - np.array(removed)
        data = pd.DataFrame(
            {
                "Genome": genomes,
                "Gene": [cfg.gene] * len(genomes),
                "Original": predictions,
                "Removed": removed,
                "Delta": deltas,
                "Length": lengths,
            }
        ).to_csv(out_dir / "predictions.csv", index=False)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
