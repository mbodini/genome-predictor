"""Utils to work on sequences."""

import re
import string
from copy import deepcopy
from typing import Union, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from gff3_parser import parse_gff3

from genpred.utils.paths import DATA_ROOT, FASTA_ROOT, GFF3_ROOT


NONSTANDARD_BASES = "".join(set(string.ascii_uppercase).difference(set("ACGT")))


def load_genes_list(filename: str) -> List[str]:
    """Loads a list of genes."""
    genes_folder = DATA_ROOT / "genes"
    with open(genes_folder / filename, "r", encoding="utf-8") as file:
        genes = file.read().split("\n")
    return genes[:-1] if genes[-1] == "" else genes


def read_gff3(path: Union[Path, str]) -> pd.DataFrame:
    """Reads gff3 file."""
    gff3 = parse_gff3(path, parse_attributes=True, verbose=False)
    gff3 = gff3[["Seqid", "Start", "End", "locus_tag"]]
    gff3.Seqid = gff3.Seqid.astype(str)
    gff3.Start = gff3.Start.astype(int)
    gff3.End = gff3.End.astype(int)
    return gff3


def read_fasta(path: Union[Path, str]) -> List[SeqRecord]:
    """Loads BioPython SeqRecords given the path to a fasta file."""
    return list(SeqIO.parse(path, "fasta"))


def write_fasta(records: List[SeqRecord], path: Union[Path, str]) -> None:
    """Saves a list of BioPython SeqRecords to a given fasta file path."""
    with open(path, "w", encoding="utf-8") as file:
        SeqIO.write(records, file, "fasta")


def clean_seq(seq: str) -> str:
    """Replaces non-standard bases with X."""
    seq, _ = re.subn(f"[{NONSTANDARD_BASES}]", "X", seq)
    cleaned, _ = re.subn("X+", "X", seq)
    return cleaned


def seq2rc(seq: str) -> str:
    """Transforms a sequence into its reverse complement."""
    return str(Seq(seq).reverse_complement())


def _retrieve_contig(fasta: List[SeqRecord], cid: str) -> int:
    contig = [(i, contig) for (i, contig) in enumerate(fasta) if contig.id == cid]
    assert len(contig) == 1
    return contig[0][0]


def _remove_bases(record: SeqRecord, start: int, end: int, padding: int) -> SeqRecord:
    """Removes a subset of consecutive bases from start to end."""
    sequence = str(record.seq)
    padding = padding // 2
    start = max(start - padding, 0)
    end = min(end + padding, len(sequence))
    replacement = "".join(["X"] * (end - start))
    return SeqRecord(
        id=record.id,
        name=record.name,
        description=record.description,
        seq=Seq(sequence.replace(sequence[start:end], replacement)),
    )


def remove_gene(
    fasta: List[SeqRecord],
    gff3: pd.DataFrame,
    gene: str,
    padding: int = 0,
    control: bool = False,
) -> Tuple[List[SeqRecord], int]:
    """Removes gene from genome."""
    gff3 = gff3[gff3.locus_tag == gene]

    if gff3.shape[0] != 1:
        return fasta, 0

    locus = gff3.iloc[0]
    contig_idx = _retrieve_contig(fasta, locus.Seqid)
    start, end = locus.Start, locus.End + 1

    if control:
        contig_idx = np.random.choice(len(fasta))

    new_fasta = deepcopy(fasta)
    new_fasta[contig_idx] = _remove_bases(
        fasta[contig_idx],
        start,
        end,
        padding=padding,
    )

    return new_fasta, end - start


def remove_capsule(
    fasta: List[SeqRecord],
    gff3: pd.DataFrame,
    control: bool = False,
) -> Tuple[List[SeqRecord], int]:
    """Removes capsule from genome."""
    gff3 = gff3[gff3.locus_tag.isin(["NEIS0044", "NEIS0068"])]

    if len(gff3) != 2 or len(gff3.Seqid.unique()) != 1:
        return fasta, 0

    contig_idx = _retrieve_contig(fasta, gff3.iloc[0].Seqid)
    start = gff3.loc[gff3.Start.idxmin()].End + 1
    end = gff3.loc[gff3.End.idxmax()].Start
    new_fasta = deepcopy(fasta)

    if control:
        contig_idx = np.random.choice(len(fasta))

    new_fasta[contig_idx] = _remove_bases(fasta[contig_idx], start, end, padding=0)
    return new_fasta, end - start


def genomes_not_equal(fasta1: List[SeqRecord], fasta2: List[SeqRecord]) -> bool:
    """Returns true if the two genomes are equal in sequence across contigs."""
    return any(str(c1.seq) != str(c2.seq) for c1, c2 in zip(fasta1, fasta2))


def filter_predictions(data: pd.DataFrame, task: str, threshold: float = 0.0):
    """Filters predictions."""

    filters = {
        "capsule": (data.Probability >= threshold),
        "invasive": (data.Probability >= threshold),
        "carriage": (data.Probability < 1 - threshold),
    }

    data["Fasta"] = data.Genome.apply(lambda g: FASTA_ROOT / f"{g}.fasta")
    data["Gff3"] = data.Genome.apply(lambda g: GFF3_ROOT / f"{g}.gff3")
    return data[filters[task]].copy()
