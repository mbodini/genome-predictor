"""Tokenizer wrapper class."""
from pathlib import Path
from typing import Optional, List

from tokenizers import SentencePieceBPETokenizer, Tokenizer as T

from genpred.utils.paths import DATA_ROOT
from genpred.utils.sequences import seq2rc


class Tokenizer:
    """A wrapper on a HuggingFace Tokenizer."""

    def __init__(
        self,
        tokenizer: Optional[SentencePieceBPETokenizer] = None,
    ):
        if tokenizer is None:
            tokenizer = SentencePieceBPETokenizer(add_prefix_space=False)

        self.tokenizer = tokenizer

    def _tokenize(self, seq):
        assert self.tokenizer is not None
        return [
            tok
            for tok in self.tokenizer.encode(seq).tokens
            if len(tok) > 2 and "X" not in tok
        ]

    def __call__(self, seq: str) -> List[str]:
        return self._tokenize(seq) + self._tokenize(seq2rc(seq))

    def train(self, corpus: List, vocab_size: int, path: Path) -> None:
        """Trains a tokenizer on an iterator."""
        assert self.tokenizer is not None

        def batch_iterator(batch_size=10000):
            for start in range(0, len(corpus), batch_size):
                end = min(start + batch_size, len(corpus))
                yield corpus[start:end]

        self.tokenizer.train_from_iterator(
            batch_iterator(),
            show_progress=False,
            initial_alphabet=list("ACGT"),
            vocab_size=vocab_size,
            special_tokens=["X"],
            min_frequency=5,
        )
        self.tokenizer.save(path.as_posix())
        print(f"Saved to {path}.")


def load_tokenizer(name: str, vocab_size: int) -> Tokenizer:
    """Loads a previously saved Tokenizer."""
    path = DATA_ROOT / name / "sentencepiece" / f"{vocab_size}" / "tokenizer.json"
    tokenizer = T.from_file(path.as_posix())
    return Tokenizer(tokenizer)
