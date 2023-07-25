"Capsule Knockout."
import hydra
import joblib

import pandas as pd
from omegaconf import DictConfig
from joblib import Parallel, delayed


import genpred
from genpred.utils.sequences import (
    remove_capsule,
    write_fasta,
    read_fasta,
    read_gff3,
    filter_predictions,
)

# pylint:disable=logging-fstring-interpolation
# pylint:disable=invalid-name


def _helper(vectorizer, model, out_dir, row, control):
    genome = row.Fasta.stem
    fasta = read_fasta(row.Fasta)
    gff3 = read_gff3(genpred.GFF3_ROOT / f"{genome}.gff3")

    new_fasta, length = remove_capsule(fasta, gff3, control)
    removed_prediction = None

    if length > 0:
        path_removed = out_dir / f"{row.Fasta.stem}-modified.fasta"
        write_fasta(new_fasta, path_removed)
        removed_hist = vectorizer.transform([path_removed])
        removed_prediction = model.predict_proba(removed_hist)[:, 1][0]

    return genome, row.Probability, removed_prediction, length


def _get_exp_path(cfg: DictConfig):
    data = cfg.dataset.name
    vect = cfg.dataset.vectorizer
    vocab_size = cfg.dataset.vocab_size
    exp_root = genpred.EXPS_ROOT / data / vect / f"{vocab_size}"
    return exp_root


def _get_data_path(cfg: DictConfig):
    data = cfg.dataset.name
    vect = cfg.dataset.vectorizer
    vocab_size = cfg.dataset.vocab_size
    data_root = genpred.DATA_ROOT / data / vect / f"{vocab_size}"
    return data_root


def _get_out_path(cfg: DictConfig):
    data = cfg.dataset.name
    vect = cfg.dataset.vectorizer
    vocab_size = cfg.dataset.vocab_size
    out_dir = genpred.EXPS_ROOT / "knockout"
    name = "capsule" if cfg.control is False else "capsule-control"
    out_dir = out_dir / data / vect / f"{vocab_size}" / name
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
    out_dir = _get_out_path(cfg)

    preds = pd.read_csv(exp_root / "predictions.csv")
    preds = filter_predictions(preds, "capsule")
    print(f"Working with {preds.shape[0]} genomes.")

    vectorizer = joblib.load(data_root / "vectorizer.pkl")
    model = joblib.load(exp_root / "model.pkl")

    data = Parallel(verbose=1, n_jobs=-1)(
        delayed(_helper)(vectorizer, model, out_dir, row, cfg.control) for row in preds.itertuples()
    )

    genomes, predictions, removed, lengths = zip(*data)
    name = "capsule" if cfg.control is False else "capsule-control"

    pd.DataFrame(
        {
            "Genome": genomes,
            "Group": [name] * len(genomes),
            "Before": predictions,
            "After": removed,
            "Length": lengths,
        }
    ).dropna().to_csv(out_dir / "predictions.csv", index=False)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
