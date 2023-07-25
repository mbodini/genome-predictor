"""Common paths."""

import pyrootutils

PROJECT_ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".gitignore"],
    pythonpath=True,
    dotenv=True,
)

# pylint:disable=wrong-import-order
# pylint:disable=wrong-import-position
# pylint:disable=logging-fstring-interpolation

DATA_ROOT = PROJECT_ROOT / "data"
EXPS_ROOT = PROJECT_ROOT / "experiments"

# pylint:disable=line-too-long
FASTA_ROOT = DATA_ROOT / "raw" / "fasta"
GFF3_ROOT = DATA_ROOT / "raw" / "gff3"
NB_ROOT = PROJECT_ROOT / "notebooks"
