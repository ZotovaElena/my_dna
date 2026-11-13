"""Shared helpers: config, raw-file loading, genotype utilities and reference-panel access."""

import argparse
import shlex
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
AUTOSOMES = [str(c) for c in range(1, 23)]
CHROM_RANK = {c: i for i, c in enumerate(AUTOSOMES + ["X", "Y", "MT"])}
PATH_KEYS = ("raw_file", "out_dir", "ref_dir", "cram", "reference_fasta", "ancient_dir")
KG_PANEL = "integrated_call_samples_v3.20130502.ALL.panel"

# Shell variable -> config key, exported by `python scripts/common.py --shell` (see scripts/env.sh)
SHELL_VARS = {
    "OUT": "out_dir", "REF": "ref_dir", "THREADS": "threads", "SAMPLE": "sample_id",
    "REFS": "admixture_refs", "CRAM": "cram", "FASTA": "reference_fasta", "ANCIENT_DIR": "ancient_dir",
}


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    """Load the YAML config and resolve paths against the project root."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k in PATH_KEYS:
        if k in cfg:
            cfg[k] = (ROOT / cfg[k]).resolve()
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    cfg["panel"] = cfg["ref_dir"] / "1kg" / KG_PANEL
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


def is_called(genotype: pd.Series) -> pd.Series:
    """True for real calls (one or two of A/C/G/T); False for no-calls such as '--'."""
    return genotype.str.fullmatch(r"[ACGT]{1,2}", na=False)


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
        df = df[is_called(df["genotype"])]
    return df.reset_index(drop=True)


def is_het(genotype: pd.Series) -> pd.Series:
    return (genotype.str.len() == 2) & (genotype.str[0] != genotype.str[1])


def alt_copies(genotype: str, ref: str, alt: str) -> int | None:
    """ALT allele count in a called genotype ('AG', or 'A' if hemizygous); None if not made of REF/ALT."""
    if not set(genotype) <= {ref, alt}:
        return None
    return genotype.count(alt)


def chrom_key(col: pd.Series) -> pd.Series:
    """`sort_values(key=...)` helper: natural chromosome order (1..22, X, Y, MT); other columns as is."""
    return col.map(CHROM_RANK).fillna(len(CHROM_RANK)) if col.name == "chrom" else col


def load_panel(cfg: dict) -> pd.DataFrame:
    """1000 Genomes phase 3 sample panel: sample, pop, super_pop, gender."""
    panel = pd.read_csv(cfg["panel"], sep="\t")
    return panel.loc[:, ~panel.columns.str.startswith("Unnamed")]  # the file has trailing tabs


def shell_exports(cfg: dict) -> str:
    """Config values as shell assignments; lists become space-separated strings."""
    lines = []
    for var, key in {**SHELL_VARS, "PANEL": "panel"}.items():
        if key in cfg:
            value = cfg[key]
            lines.append(f"{var}={shlex.quote(' '.join(map(str, value)) if isinstance(value, list) else str(value))}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Lets shell scripts read config values:
    #   python scripts/common.py out_dir   -> one value
    #   python scripts/common.py --shell   -> OUT=... REF=... lines for `eval` (used by scripts/env.sh)
    parser = config_arg("Print config values.")
    parser.add_argument("key", nargs="?")
    parser.add_argument("--shell", action="store_true", help="print all values as shell assignments")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.shell:
        print(shell_exports(cfg))
    elif args.key:
        value = cfg[args.key]
        print(" ".join(map(str, value)) if isinstance(value, list) else value)
    else:
        parser.error("give a config key or --shell")
