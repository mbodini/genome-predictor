"""Download scripts."""
from pathlib import Path
from urllib import request

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

import genpred


FASTA_URL = "https://pubmlst.org/bigsdb?db=pubmlst_neisseria_isolates&page=downloadSeqbin&isolate_id={}"
GFF3_URL = "https://pubmlst.org/bigsdb?db=pubmlst_neisseria_isolates&page=gff&isolate_id={}"


def _download(url: str, dst_path: Path) -> Path:
    if not dst_path.exists():
        with request.urlopen(url) as src, open(dst_path, "w", encoding="utf-8") as dst:
            dst.write(src.read().decode("utf-8"))
    return dst_path


def download_fasta(gid: int) -> Path:
    url = FASTA_URL.format(gid)
    dst_path = genpred.FASTA_ROOT / f"{gid}.fasta"
    return _download(url, dst_path)


def download_gff3(gid: int) -> Path:
    url = GFF3_URL.format(gid)
    dst_path = genpred.GFF3_ROOT / f"{gid}.gff3"
    return _download(url, dst_path)


@hydra.main(
    config_path=genpred.CFG_ROOT.as_posix(),
    config_name="download.yaml",
    version_base="1.2",
)
def main(cfg: DictConfig):
    """Download genomes from PubMLST."""
    genpred.FASTA_ROOT.mkdir(exist_ok=True, parents=True)
    genpred.GFF3_ROOT.mkdir(exist_ok=True, parents=True)
    data = pd.read_csv(genpred.DATA_RAW_ROOT / "info.csv", low_memory=False)
    genome_ids = data.id.tolist()

    max_num = cfg.max_num or len(genome_ids)

    downloaded_ids, dst_paths = [], []
    for gid in tqdm(genome_ids[:max_num]):
        try:
            dst_path = download_fasta(gid)
            _ = download_gff3(gid)
            dst_paths.append(dst_path)
            downloaded_ids.append(gid)
        except Exception:
            print(f"Couldn't download genome with id={gid}")

    data = data[data.id.isin(downloaded_ids)]
    data["Path"] = dst_paths
    data.to_csv(genpred.DATA_ROOT / "genomes.csv", index=False)


if __name__ == "__main__":
    main()
