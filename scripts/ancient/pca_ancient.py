"""West Eurasian PCA from modern Human Origins samples, with ancient genomes and you projected onto it.

The axes come from modern AADR HO samples only (config pca_ancient.modern, at most max_per_pop each). Ancient
genomes and you are then placed by least squares on the SNPs each one has (what smartpca does with
lsqproject: YES), so missing data and one-read-per-SNP genotypes neither shrink them toward the centre nor
shift the axes. Pseudo-haploid samples count as homozygous at their one read, like smartpca.

Output in out_dir/pca_ancient/: moderns.csv, projected.csv (PC1-PC4 per sample) and eigen.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py and the module folders
from common import config_arg, load_config

from ancient import aadr_prepare as prep
K = 4
CHUNK = 40_000


def main() -> None:
    args = config_arg(__doc__).parse_args()
    cfg = load_config(args.config)
    o = cfg["pca_ancient"]
    out = cfg["out_dir"] / "pca_ancient"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    f = prep.aadr_files(cfg)
    ind = prep.read_ind(f["ind"])
    snp = pd.read_csv(f["snp"], sep=r"\s+", header=None, usecols=[0], names=["id"])["id"]
    aadr = cfg["out_dir"] / "aadr"
    merged_ids = pd.read_csv(aadr / "merged.snp", sep=r"\s+", header=None, usecols=[0], names=["id"])["id"]
    full_i = pd.Series(np.arange(len(snp)), index=snp).reindex(merged_ids).to_numpy()
    if np.isnan(full_i).any():
        raise SystemExit("merged.snp has SNPs that are not in the AADR .snp")
    full_i = full_i.astype(np.int64)
    L = len(full_i)

    fmt, nind, nsnp = prep.geno_format(f["geno"])
    if fmt != "TGENO":
        raise SystemExit(f"{f['geno']}: {fmt} layout; expected the AADR v66 TGENO")
    rlen = max(48, -(-nsnp // 4))
    head = prep.header_bytes(f["geno"], rlen, nind)
    fh = open(f["geno"], "rb")

    def person(i: int) -> np.ndarray:  # copies of allele 1 at the merged SNPs, 3 = missing
        fh.seek(head + rlen * int(i))
        return prep.unpack(np.frombuffer(fh.read(rlen), np.uint8), full_i)

    # Moderns that define the axes
    grp = ind["group"].to_numpy()
    mod = []
    for pop in o["modern"]:
        pool = np.flatnonzero(grp == pop)
        if len(pool) == 0:
            print(f"  {pop}: not in the AADR .ind, skipped")
            continue
        mod += list(rng.choice(pool, size=min(o["max_per_pop"], len(pool)), replace=False))
    mod = np.array(sorted(mod))
    print(f"Reading {len(mod)} modern samples × {L:,} SNPs...", flush=True)
    G = np.empty((len(mod), L), np.uint8)
    for r, i in enumerate(mod):
        G[r] = person(i)

    # SNPs: called in >= 95% of the moderns, minor allele frequency >= 1%
    ok = G != 3
    called = ok.mean(0)
    p = np.where(ok, G, 0).sum(0) / np.maximum(2 * ok.sum(0), 1)
    keep = (called >= 0.95) & (np.minimum(p, 1 - p) >= 0.01)
    G, p, cols = G[:, keep], p[keep], np.flatnonzero(keep)
    sd = np.sqrt(p * (1 - p))
    M = len(cols)
    print(f"  {M:,} SNPs kept (called in ≥ 95% of moderns, MAF ≥ 1%)", flush=True)

    def standard(g: np.ndarray, sl: slice) -> np.ndarray:
        x = (g.astype(np.float32) - 2 * p[sl]) / sd[sl]
        return np.where(g == 3, 0, x).astype(np.float32)

    # GRM and eigenvectors, then SNP axes W = X^T U / s (so scores = X W, and projection = least squares on W)
    grm = np.zeros((len(mod), len(mod)), np.float64)
    for s0 in range(0, M, CHUNK):
        X = standard(G[:, s0:s0 + CHUNK], slice(s0, s0 + CHUNK))
        grm += X @ X.T
    lam, U = np.linalg.eigh(grm)
    lam, U = lam[::-1], U[:, ::-1]
    s = np.sqrt(lam[:K])
    W = np.zeros((M, K), np.float32)
    for s0 in range(0, M, CHUNK):
        X = standard(G[:, s0:s0 + CHUNK], slice(s0, s0 + CHUNK))
        W[s0:s0 + CHUNK] = (X.T @ U[:, :K]) / s
    scores = U[:, :K] * s
    share = lam[:K] / lam[lam > 0].sum()
    pd.DataFrame({"pc": [f"PC{k + 1}" for k in range(K)], "eigenvalue": lam[:K] / M, "variance_share": share}
                 ).to_csv(out / "eigen.csv", index=False)
    mods = pd.DataFrame(scores, columns=[f"PC{k + 1}" for k in range(K)])
    mods.insert(0, "pop", grp[mod])
    mods.insert(0, "id", ind["id"].to_numpy()[mod])
    mods.to_csv(out / "moderns.csv", index=False)
    del G

    def project(g: np.ndarray) -> tuple[np.ndarray, int]:
        """Least-squares position from the called SNPs (g: copies of allele 1 at the kept SNPs, 3 = missing)."""
        obs = g != 3
        x = (g[obs].astype(np.float32) - 2 * p[obs]) / sd[obs]
        Wo = W[obs]
        return np.linalg.solve(Wo.T @ Wo, Wo.T @ x), int(obs.sum())

    rows = []
    # You: me_cram (one read per SNP, coded 0/2) and me_csv, from the merged dataset's own files
    me = np.frombuffer(b"".join(l.strip() for l in open(aadr / "me_cram.geno", "rb")), np.uint8) - ord("0")
    mine = {"you (CRAM)": me}
    m_ind = pd.read_csv(aadr / "merged.ind", sep=r"\s+", header=None, names=["id", "sex", "group"])
    if (m_ind["group"] == "me_csv").any():
        j = np.flatnonzero(m_ind["group"].to_numpy() == "me_csv")
        g = np.concatenate(list(prep.aadr_genotypes(aadr / "merged.geno", len(m_ind), L, j, chunk=50_000)))[:, 0]
        mine["you (MyHeritage CSV)"] = g
    for name, g in mine.items():
        g = np.where(g[cols] == 9, 3, g[cols]).astype(np.uint8)
        pc, nobs = project(g)
        rows.append({"id": name, "group": "you", "label": "", "snps": nobs, **{f"PC{k + 1}": pc[k] for k in range(K)}})

    # Ancient genomes, grouped by display name
    anno = prep.annotation_summary(f["anno"])
    for name, labels in o["ancient"].items():
        pool = np.flatnonzero(np.isin(grp, labels))
        if len(pool) == 0:
            print(f"  {name}: no samples with labels {labels}")
            continue
        n_ok = 0
        for i in rng.permutation(pool):
            if n_ok >= o["max_per_ancient"]:
                break
            g = person(i)[cols]
            if (g != 3).sum() < o["min_snps"]:
                continue
            g = np.where(g == 1, np.where(rng.random(len(g)) < 0.5, 0, 2), g).astype(np.uint8)  # het: random read
            pc, nobs = project(g)
            rows.append({"id": ind.at[i, "id"], "group": name, "label": grp[i], "snps": nobs,
                         "date_bp": anno.at[grp[i], "mean_BP"] if anno is not None and grp[i] in anno.index else np.nan,
                         **{f"PC{k + 1}": pc[k] for k in range(K)}})
            n_ok += 1
        print(f"  {name}: {n_ok} projected", flush=True)
    proj = pd.DataFrame(rows)
    proj.to_csv(out / "projected.csv", index=False)

    c = mods.groupby("pop")[["PC1", "PC2"]].mean()
    you = proj[proj["group"] == "you"].set_index("id")[["PC1", "PC2"]]
    print(f"\nPC1 {share[0]:.1%}, PC2 {share[1]:.1%} of the variance")
    for name, r in you.iterrows():
        d = np.hypot(c["PC1"] - r["PC1"], c["PC2"] - r["PC2"]).sort_values()
        print(f"{name}: PC1 {r['PC1']:.2f}, PC2 {r['PC2']:.2f}; nearest modern centroids: "
              + ", ".join(f"{k} ({v:.2f})" for k, v in d.head(5).items()))
    print(proj.groupby("group")[["PC1", "PC2"]].mean().round(2).to_string())
    print(f"\n{out}")


if __name__ == "__main__":
    main()
