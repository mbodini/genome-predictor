"""Trains a tokenizer on the corpus."""
import os

import hydra
from omegaconf import DictConfig

import genpred
from genpred.training.loaders import load_corpus
from genpred.training.tokenizer import Tokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "true"


@hydra.main(
    config_path=genpred.CFG_ROOT.as_posix(),
    config_name="tokenize.yaml",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    """Trains a tokenizer with a given config."""

    dataset = hydra.utils.instantiate(cfg.dataset, partition="training")
    dataset.save_dir.mkdir(exist_ok=True, parents=True)

    if not (dataset.save_dir / "tokenizer.json").exists():
        print(f"Training tokenizer with vocab size: [{cfg.dataset.vocab_size}].")

        corpus = load_corpus(dataset.paths, num_sentences=cfg.num_sentences)
        print(f"Training corpus size: [{len(corpus)}].")

        Tokenizer().train(
            corpus=corpus,
            vocab_size=cfg.dataset.vocab_size,
            path=dataset.save_dir / "tokenizer.json",
        )


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
