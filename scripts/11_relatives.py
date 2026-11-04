"""Compare with a relative's raw file: KING-robust kinship, IBS0 and half-identical segments.

The other file can come from any vendor `snps` reads (23andMe, AncestryDNA, FTDNA, MyHeritage...).
Segments use physical length (no genetic map), so they are a rough proxy for shared cM.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from common import AUTOSOMES, config_arg, load_config, load_myheritage

# KING thresholds (Manichaikul et al. 2010)
DEGREES = [(0.354, "duplicate / identical twin"), (0.177, "1st degree"), (0.0884, "2nd degree"),
           (0.0442, "3rd degree")]


def load_other(path: Path) -> pd.DataFrame:
    from snps import SNPs

    s = SNPs(str(path))
    print(f"Other file: {s.source}, build {s.build}, {s.count} SNPs")
    return s.snps.reset_index().dropna(subset=["genotype"])[["rsid", "chrom", "pos", "genotype"]]


def compare(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Merge on rsID (build-independent); keep diploid autosomal biallelic sites, positions from `a`."""
    m = a.merge(b[["rsid", "genotype"]], on="rsid", suffixes=("_a", "_b"))
    m = m[m["chrom"].isin(AUTOSOMES) & (m["genotype_a"].str.len() == 2) & (m["genotype_b"].str.len() == 2)]
    m = m[[len(set(x + y)) <= 2 for x, y in zip(m["genotype_a"], m["genotype_b"])]]
    ga, gb = m["genotype_a"], m["genotype_b"]
    hom_a, hom_b = ga.str[0] == ga.str[1], gb.str[0] == gb.str[1]
    return m.assign(het_a=~hom_a, het_b=~hom_b, opp_hom=hom_a & hom_b & (ga.str[0] != gb.str[0]))


def kinship(m: pd.DataFrame) -> dict[str, float]:
    """KING-robust: (N_Aa,Aa − 2·N_AA,aa) / (N_Aa(a) + N_Aa(b))."""
    n_hethet = (m["het_a"] & m["het_b"]).sum()
    n_opp = m["opp_hom"].sum()
    phi = (n_hethet - 2 * n_opp) / (m["het_a"].sum() + m["het_b"].sum())
    return {"n_snps": len(m), "kinship": round(phi, 4), "ibs0": round(n_opp / len(m), 5)}


def relationship(phi: float, ibs0: float) -> str:
    for threshold, label in DEGREES:
        if phi > threshold:
            if label == "1st degree":
                return "parent/child" if ibs0 < 0.002 else "full siblings"
            return label
    return "unrelated or distant"


def half_identical_segments(m: pd.DataFrame, min_snps: int, min_mb: float) -> pd.DataFrame:
    """Runs without opposite homozygotes, i.e. stretches where at least one haplotype may be shared."""
    segs = []
    for chrom, g in m.sort_values("pos").groupby("chrom", sort=False):
        pos = g["pos"].to_numpy()
        bounds = np.concatenate([[-1], np.flatnonzero(g["opp_hom"].to_numpy()), [len(g)]])
        for s, e in zip(bounds[:-1] + 1, bounds[1:]):
            if e - s >= min_snps and (mb := (pos[e - 1] - pos[s]) / 1e6) >= min_mb:
                segs.append({"chrom": chrom, "start": pos[s], "end": pos[e - 1], "mb": round(mb, 2), "snps": e - s})
    return pd.DataFrame(segs, columns=["chrom", "start", "end", "mb", "snps"])


def main() -> None:
    parser = config_arg(__doc__)
    parser.add_argument("--other", type=Path, required=True, help="Relative's raw DNA file")
    parser.add_argument("--name", default="relative")
    parser.add_argument("--min_snps", type=int, default=500)
    parser.add_argument("--min_mb", type=float, default=7.0)
    args = parser.parse_args()
    cfg = load_config(args.config)

    m = compare(load_myheritage(cfg["raw_file"]), load_other(args.other))
    k = kinship(m)
    segs = half_identical_segments(m, args.min_snps, args.min_mb)
    k |= {"relationship": relationship(k["kinship"], k["ibs0"]), "n_segments": len(segs),
          "total_mb": round(segs["mb"].sum(), 1), "largest_mb": segs["mb"].max() if len(segs) else 0}

    segs.to_csv(cfg["out_dir"] / f"segments_{args.name}.csv", index=False)
    for key, v in k.items():
        print(f"{key:>14}: {v}")


if __name__ == "__main__":
    main()
