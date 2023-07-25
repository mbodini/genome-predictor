"""Common paths."""

import pyrootutils

PROJECT_ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".gitignore"],
    pythonpath=True,
    dotenv=True,
)

CFG_ROOT = PROJECT_ROOT / "config"
DATA_ROOT = PROJECT_ROOT / "data"
EXPS_ROOT = PROJECT_ROOT / "experiments"

DATA_RAW_ROOT = DATA_ROOT / "raw"
FASTA_ROOT = DATA_RAW_ROOT / "fasta"
GFF3_ROOT = DATA_RAW_ROOT / "gff3"
NB_ROOT = PROJECT_ROOT / "notebooks"
