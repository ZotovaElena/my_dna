"""Polygenic scores from PGS Catalog, scored with plink2 and placed against a 1000G superpopulation.

Scores trained mostly on Europeans transfer poorly to other ancestries; a percentile is
only meaningful against an ancestry-matched reference (`prs_superpop`).
"""

import gzip
import subprocess
import time
import urllib.request
from itertools import takewhile

import pandas as pd

from common import AUTOSOMES, config_arg, load_config

PGS_URL = "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/{id}/ScoringFiles/Harmonized/{id}_hmPOS_GRCh37.txt.gz"


def fetch(pgs_id: str, dest_dir) -> tuple[pd.DataFrame, dict[str, str]]:
    """Download (once) a harmonised GRCh37 scoring file; return weights and header metadata."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{pgs_id}_hmPOS_GRCh37.txt.gz"
    for attempt in range(3):  # EBI FTP resets connections intermittently
        if path.exists():
            break
        try:
            urllib.request.urlretrieve(PGS_URL.format(id=pgs_id), path)
        except OSError:
            path.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(5)
    with gzip.open(path, "rt") as f:
        header = takewhile(lambda line: line.startswith("#"), f)
        meta = dict(line.lstrip("#").strip().split("=", 1) for line in header if "=" in line)
    df = pd.read_csv(path, sep="\t", comment="#", dtype={"hm_chr": str}, low_memory=False)
    return df, meta


def to_plink_score(df: pd.DataFrame) -> pd.DataFrame:
    """IDs as chrom:pos to match the merged pgen; autosomal SNVs only."""
    df = df.dropna(subset=["hm_chr", "hm_pos", "effect_weight"])
    df = df[df["hm_chr"].isin(AUTOSOMES) & (df["effect_allele"].str.len() == 1)]
    score = pd.DataFrame({
        "ID": df["hm_chr"] + ":" + df["hm_pos"].astype(int).astype(str),
        "A1": df["effect_allele"],
        "WEIGHT": df["effect_weight"],
    })
    return score.drop_duplicates("ID", keep=False)


def main() -> None:
    args = config_arg(__doc__).parse_args()
    cfg = load_config(args.config)
    out = cfg["out_dir"] / "prs"
    out.mkdir(exist_ok=True)
    panel = pd.read_csv(cfg["ref_dir"] / "1kg" / "integrated_call_samples_v3.20130502.ALL.panel",
                        sep="\t", usecols=["sample", "super_pop"])
    ref_ids = set(panel.loc[panel["super_pop"] == cfg["prs_superpop"], "sample"])

    rows = []
    for pgs_id in cfg["pgs_ids"]:
        weights, meta = fetch(pgs_id, cfg["ref_dir"] / "pgs")
        score = to_plink_score(weights)
        score_file = out / f"{pgs_id}.score.tsv"
        score.to_csv(score_file, sep="\t", index=False)

        subprocess.run([
            "plink2", "--pfile", str(cfg["out_dir"] / "merged"),
            "--score", str(score_file), "1", "2", "3", "header", "cols=+scoresums", "list-variants",
            "--threads", str(cfg["threads"]), "--out", str(out / pgs_id),
        ], check=True)

        res = pd.read_csv(out / f"{pgs_id}.sscore", sep="\t").rename(columns=lambda c: c.lstrip("#"))
        n_used = sum(1 for _ in open(out / f"{pgs_id}.sscore.vars"))
        col = next(c for c in res.columns if c.endswith("_SUM") and "DOSAGE" not in c)
        me = res.loc[res["IID"] == cfg["sample_id"], col].iloc[0]
        ref = res.loc[res["IID"].isin(ref_ids), col]
        rows.append({
            "pgs_id": pgs_id,
            "trait": meta.get("trait_reported", ""),
            "variants_in_score": len(weights),
            "variants_used": n_used,
            "coverage": round(n_used / len(weights), 3),
            "z": round((me - ref.mean()) / ref.std(), 2),
            f"percentile_vs_{cfg['prs_superpop']}": round((ref < me).mean() * 100, 1),
        })

    table = pd.DataFrame(rows)
    table.to_csv(out / "prs_summary.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
