"""Trains a vectorizer on the corpus."""
import os

import hydra
import joblib
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from compress_pickle import dump

import genpred


os.environ["TOKENIZERS_PARALLELISM"] = "true"

NGRAM_RANGES = {
    64: (3, 4),
    320: (3, 5),
    1344: (3, 6),
    5440: (3, 7),
    21824: (3, 8),
    87360: (3, 9),
}


def transform_data(cfg, vectorizer, partition):
    """Transforms documents into vectors."""
    dataset = hydra.utils.instantiate(cfg.dataset, partition=partition)
    assert dataset.save_dir.exists()

    print(f"Processing {partition} data...", end=" ")
    if not (dataset.save_dir / f"{partition}.npz").exists():
        data = vectorizer.transform(dataset.paths)
        dump(data, dataset.save_dir / f"{partition}.gz")
    print("Done.")


@hydra.main(
    config_path=genpred.CFG_ROOT.as_posix(),
    config_name="vectorize.yaml",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    """Trains a vectorizers with a given config."""
    vect = HydraConfig.get().runtime.choices["vectorizer"]
    dataset = hydra.utils.instantiate(cfg.dataset, vectorizer=vect, partition="training")
    dataset.save_dir.mkdir(exist_ok=True, parents=True)

    if not (dataset.save_dir / "vectorizer.pkl").exists():
        print(
            f"Training vectorizer with vocab size: [{cfg.dataset.vocab_size}]...",
            end=" ",
        )

        if cfg.dataset.vectorizer == "kmer":
            ngram_range = NGRAM_RANGES[cfg.dataset.vocab_size]
            vectorizer = hydra.utils.instantiate(cfg.vectorizer, ngram_range=ngram_range)
        else:
            vectorizer = hydra.utils.instantiate(cfg.vectorizer)

        vectorizer = vectorizer.fit(dataset.paths)
        joblib.dump(vectorizer, dataset.save_dir / "vectorizer.pkl")
        print("Done.")

    vectorizer = joblib.load(dataset.save_dir / "vectorizer.pkl")

    transform_data(cfg, vectorizer, "training")
    transform_data(cfg, vectorizer, "validation")
    transform_data(cfg, vectorizer, "test")


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
