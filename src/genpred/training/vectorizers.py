"""Preprocessors."""
import itertools
import numpy as np

from Bio import SeqIO
from sklearn.feature_extraction.text import TfidfVectorizer

from genpred.utils.sequences import clean_seq


class SentencePieceTfidfVectorizer(TfidfVectorizer):
    """Custom Tfidf vectorizer for sentencepiece."""

    def __init__(
        self,
        *,
        tokenizer=None,
        max_df=1.0,
        min_df=1,
        binary=False,
    ):
        super().__init__(
            input="filename",
            lowercase=False,
            preprocessor=None,
            tokenizer=None,
            token_pattern=None,
            analyzer=str.split,
            max_df=max_df,
            min_df=min_df,
            binary=binary,
            dtype=np.float32,
        )
        self._tok = tokenizer

    def decode(self, doc):
        """Decode the input into a string of unicode symbols.
        The decoding strategy depends on the vectorizer parameters.
        Parameters
        ----------
        doc : bytes or str
            The string to decode.
        Returns
        -------
        doc: str
            A string of unicode symbols.
        """
        assert self.input == "filename"
        assert self._tok is not None

        words = " ".join(
            itertools.chain.from_iterable(
                self._tok(clean_seq(str(c.seq))) for c in SeqIO.parse(doc, "fasta")
            )
        )
        return words


class KMerTfidfVectorizer(TfidfVectorizer):
    """Custom Tfidf vectorizer for kmers."""

    def __init__(
        self,
        *,
        ngram_range=(3, 8),
        max_df=1.0,
        min_df=1,
    ):
        super().__init__(
            input="filename",
            lowercase=False,
            preprocessor=clean_seq,
            analyzer="char",
            ngram_range=tuple(ngram_range),
            max_df=max_df,
            min_df=min_df,
            dtype=np.float32,
        )

        bases = ["A", "T", "C", "G"]
        self.ngrams_set = set(
            "".join(t)
            for i in range(*ngram_range)
            for t in itertools.product(bases, repeat=i)
        )

    def decode(self, doc):
        """Decode the input into a string of unicode symbols.
        The decoding strategy depends on the vectorizer parameters.
        Parameters
        ----------
        doc : bytes or str
            The string to decode.
        Returns
        -------
        doc: str
            A string of unicode symbols.
        """
        assert self.input == "filename"

        records = SeqIO.parse(doc, "fasta")
        doc = "\n".join([str(s.seq) for s in records])
        return doc

    def _tokenize(self, text):
        text_len = len(text)
        min_n, max_n = self.ngram_range
        if min_n == 1:
            # no need to do any slicing for unigrams
            # iterate through the string
            ngrams = list(text)
            min_n += 1
        else:
            ngrams = []

        # bind method outside of loop to reduce overhead
        ngrams_append = ngrams.append

        for n in range(min_n, min(max_n + 1, text_len + 1)):
            for i in range(text_len - n + 1):
                ngram = text[i : i + n]
                if ngram in self.ngrams_set:
                    ngrams_append(ngram)
        return ngrams

    def _char_ngrams(self, text_document):
        """Tokenize text_document into a sequence of character n-grams"""
        return list(
            itertools.chain.from_iterable(
                self._tokenize(ch) for ch in text_document.split("\n")
            )
        )
