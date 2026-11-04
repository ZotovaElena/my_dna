"""Shared helpers: config and raw-file loading."""

import argparse
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
AUTOSOMES = [str(c) for c in range(1, 23)]
PATH_KEYS = ("raw_file", "out_dir", "ref_dir")


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    """Load the YAML config and resolve paths against the project root."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k in PATH_KEYS:
        cfg[k] = (ROOT / cfg[k]).resolve()
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    return cfg


def config_arg(description: str) -> argparse.ArgumentParser:
    """Argument parser with the common --config option."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def read_header(path: Path) -> dict[str, str]:
    """Parse '##key=value' metadata lines at the top of a MyHeritage file."""
    meta = {}
    with open(path, encoding="ascii") as f:
        for line in f:
            if not line.startswith("##"):
                break
            k, _, v = line[2:].strip().partition("=")
            meta[k] = v
    return meta


def load_myheritage(path: Path, drop_nocalls: bool = True) -> pd.DataFrame:
    """Load a MyHeritage raw file (ASCII, GRCh37, forward strand).

    Returns columns rsid, chrom, pos, genotype. Hemizygous calls may be one letter.
    """
    df = pd.read_csv(
        path,
        comment="#",
        dtype={"RSID": str, "CHROMOSOME": str, "POSITION": int, "RESULT": str},
    )
    df.columns = ["rsid", "chrom", "pos", "genotype"]
    if drop_nocalls:
        df = df[df["genotype"].str.fullmatch(r"[ACGT]{1,2}", na=False)]
    return df.reset_index(drop=True)


def is_het(genotype: pd.Series) -> pd.Series:
    return (genotype.str.len() == 2) & (genotype.str[0] != genotype.str[1])


if __name__ == "__main__":
    # Lets shell scripts read config values: python scripts/common.py out_dir
    parser = config_arg("Print a config value.")
    parser.add_argument("key")
    args = parser.parse_args()
    value = load_config(args.config)[args.key]
    print(" ".join(map(str, value)) if isinstance(value, list) else value)
