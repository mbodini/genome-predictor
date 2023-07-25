"""Knockout utils."""
from typing import Dict, Union
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from gff3_parser import parse_gff3

from genpred.utils.paths import FASTA_ROOT, GFF3_ROOT
from genpred.training.tokenizer import Tokenizer


def print_to_file(
    tokenizer: Tokenizer,
    content: Dict,
    filename: Union[Path, str],
) -> None:
    """Prints tokens to file."""
    with open(filename, "w", encoding="utf-8") as file:
        for key in content:
            tokens = tokenizer(content[key])
            print(" ".join(tokens), file=file)


def read_fasta_file(path: Path):
    """Reads contigs and puts them into a dictionary."""
    contigs = list(SeqIO.parse(path, "fasta"))
    return {c.id: str(c.seq) for c in contigs}


def read_gff3_file(path: Path):
    """Reads gff3 file."""
    gff3 = parse_gff3(path, parse_attributes=True, verbose=False)
    gff3 = gff3[["Seqid", "Start", "End", "locus_tag"]]
    gff3.Start = gff3.Start.astype(int)
    gff3.End = gff3.End.astype(int)
    return gff3.copy()


def filter_predictions(data: pd.DataFrame, task: str):
    """Filters predictions."""

    filters = {
        "capsule": (data.Probability >= 0.9),
        "invasive": (data.Probability >= 0.9),
        "carriage": (data.Probability < 0.1),
    }

    data["Fasta"] = data.Genome.apply(lambda g: FASTA_ROOT / f"{g}.fasta")
    data["Gff3"] = data.Genome.apply(lambda g: GFF3_ROOT / f"{g}.gff3")
    return data[filters[task]].copy()
