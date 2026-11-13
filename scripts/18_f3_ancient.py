"""Outgroup f3(Mbuti; you, X) for every ancient individual in the AADR: how much genetic drift you share with
each ancient person since both lineages split from an African outgroup. It is the valid version of
GEDmatch's "Ancient Calculator" (shared %). Higher = closer. Values differ only in the third decimal, so
the standard errors (5 cM block jackknife) matter as much as the ranking.

You enter as `me_cram` (one random read per SNP, from 12_aadr_prepare.py), like the pseudo-haploid ancient
genomes, so no imputation is involved and no bias term is needed between the two. The AADR genotype file
is read one person at a time (TGENO layout), so memory stays small with thousands of individuals.

Output: out_dir/f3/outgroup_f3_ancient.csv (one row per ancient individual, with date, place, haplogroups).
With config f4.controls, the same f3 for AADR Russians/Belarusians/Ukrainians (one random allele per SNP,
like you) goes into f3_ctrl, and f3_diff = yours minus theirs: what is specific to you.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import config_arg, load_config

sys.path.insert(0, str(Path(__file__).parent))
prep = importlib.import_module("12_aadr_prepare")

BLOCK_CM = 5.0
MIN_SNPS = 30_000        # SNPs shared with you; fewer gives a large SE
OUTGROUP = "Mbuti"


def read_anno(path: Path) -> pd.DataFrame:
    a = pd.read_csv(path, sep="\t", low_memory=False)
    c = a.columns
    num = lambda i: pd.to_numeric(a[c[i]], errors="coerce")
    return pd.DataFrame({
        "id": a[c[0]], "iid": a[c[2]].astype(str), "pub": a[c[6]], "date_bp": num(10), "group": a[c[14]],
        "locality": a[c[15]], "country": a[c[16]], "lat": num(17), "lon": num(18),
        "y_hg": a[c[35]].astype(str), "mt_hg": a[c[38]].astype(str), "assessment": a[c[47]].astype(str),
        "ho_snps": num(27),
    })


def main() -> None:
    parser = config_arg(__doc__)
    parser.add_argument("--min-snps", type=int, default=MIN_SNPS, help="minimum SNPs shared with you")
    args = parser.parse_args()
    cfg = load_config(args.config)
    f = prep.aadr_files(cfg)
    out = cfg["out_dir"] / "f3"
    out.mkdir(parents=True, exist_ok=True)

    ind = prep.read_ind(f["ind"])
    ind.columns = ["id", "sex", "grp"]
    snp = pd.read_csv(f["snp"], sep=r"\s+", header=None, names=["id", "chrom", "gpos", "pos", "a1", "a2"],
                      dtype={"chrom": str})
    anno = read_anno(f["anno"])

    # You: pseudo-haploid calls on the AADR autosomal SNPs (same allele coding as the AADR)
    aadr = cfg["out_dir"] / "aadr"
    my_snp = pd.read_csv(aadr / "me_cram.snp", sep=r"\s+", header=None, usecols=[0], names=["id"])["id"]
    me = np.frombuffer(b"".join(l.strip() for l in open(aadr / "me_cram.geno", "rb")), np.uint8) - ord("0")
    full_i = pd.Series(np.arange(len(snp)), index=snp["id"]).reindex(my_snp).to_numpy()
    if np.isnan(full_i.astype(float)).any():
        raise SystemExit("me_cram.snp has SNPs that are not in the AADR .snp; rebuild with 12_aadr_prepare.py")
    order = np.argsort(full_i)
    keep_i, me = full_i[order].astype(np.int64), me[order]
    sub = snp.iloc[keep_i]
    block = pd.factorize(sub["chrom"] + ":" + (sub["gpos"] * 100 // BLOCK_CM).astype(int).astype(str))[0]
    nb = block.max() + 1
    me_called = me != 9
    a_me = np.where(me_called, me / 2.0, 0.0)  # frequency of the AADR first allele in you (0 or 1)
    print(f"You (me_cram): {me_called.sum():,} of {len(me):,} SNPs called; {nb} jackknife blocks of {BLOCK_CM:g} cM")

    # Ancient individuals: dated, not failed QC, one entry per person (most SNPs)
    anc = anno[(anno["date_bp"] > 0) & ~anno["assessment"].str.contains("FAIL|CRITICAL|Ignore", case=False)
               & anno["id"].isin(ind["id"])].sort_values("ho_snps", ascending=False).drop_duplicates("iid")
    anc = anc[anc["ho_snps"] >= args.min_snps]
    pos_of = pd.Series(np.arange(len(ind)), index=ind["id"])
    out_idx = pos_of[ind.loc[ind["grp"] == OUTGROUP, "id"]].to_numpy()
    anc_idx = pos_of[anc["id"]].to_numpy()
    order = np.argsort(anc_idx)                      # file order: sequential reads
    anc, anc_idx = anc.iloc[order], anc_idx[order]
    n_a = len(anc_idx)
    fmt, nind, nsnp = prep.geno_format(f["geno"])
    if fmt != "TGENO":
        raise SystemExit(f"{f['geno']}: {fmt} layout; this script reads the AADR v66 TGENO (one record per person)")
    rlen = max(48, -(-nsnp // 4))
    head = prep.header_bytes(f["geno"], rlen, nind)

    def person(fh, i: int) -> np.ndarray:  # copies of the first allele at your SNPs, 3 = missing
        fh.seek(head + rlen * int(i))
        return prep.unpack(np.frombuffer(fh.read(rlen), np.uint8), keep_i)

    print(f"Outgroup {OUTGROUP}: {len(out_idx)} people; ancient individuals: {n_a:,}", flush=True)
    S = np.zeros((nb, n_a))
    N = np.zeros((nb, n_a))
    with open(f["geno"], "rb") as fh:
        go = np.stack([person(fh, i) for i in out_idx])
        oc = go != 3
        n_hap = 2 * oc.sum(0)
        c = np.where(oc, go, 0).sum(0) / np.maximum(n_hap, 1)   # Mbuti frequency of the first allele
        ok = me_called & (n_hap > 1)
        base = c - a_me
        corr = c * (1 - c) / np.maximum(n_hap - 1, 1)          # Mbuti sampling term, same for every X
        # Controls: AADR moderns (config f4.controls) as pseudo-haploid genomes, one random allele per SNP
        rng = np.random.default_rng(7)
        ctrl, ctrl_pop = [], []
        for pop, kk in cfg.get("f4", {}).get("controls", {}).items():
            pool = np.flatnonzero((ind["grp"] == pop).to_numpy())
            for i in rng.choice(pool, size=min(kk, len(pool)), replace=False):
                g = person(fh, i).astype(np.float32)
                ctrl.append(np.where(g == 3, np.nan, np.where(g == 1, rng.integers(0, 2, len(g)), g / 2)))
                ctrl_pop.append(pop)
        if ctrl:
            Ct = np.stack(ctrl, axis=1)                              # SNPs × controls
            okc = np.isfinite(Ct) & (n_hap > 1)[:, None]
            Bc = np.where(okc, c[:, None] - np.nan_to_num(Ct), 0.0)
            Sc = np.zeros((n_a, len(ctrl)))
            Nc = np.zeros((n_a, len(ctrl)))
            print(f"Controls: {len(ctrl)} ({', '.join(f'{p} {ctrl_pop.count(p)}' for p in dict.fromkeys(ctrl_pop))})",
                  flush=True)
        for k, i in enumerate(anc_idx):
            x = person(fh, i)
            m = ok & (x != 3)
            term = base[m] * (c[m] - x[m] / 2.0) - corr[m]
            S[:, k] = np.bincount(block[m], weights=term, minlength=nb)
            N[:, k] = np.bincount(block[m], minlength=nb)
            if ctrl:
                xc = x != 3
                d = np.where(xc, c - x / 2.0, 0.0)
                Sc[k] = d @ Bc - (corr * xc) @ okc
                Nc[k] = xc.astype(np.float32) @ okc
            if (k + 1) % 1000 == 0:
                print(f"  {k + 1:,} / {n_a:,} individuals", flush=True)

    # Weighted block jackknife (Busing et al. 1999), vectorised over individuals
    Nt, St = N.sum(0), S.sum(0)
    theta = St / np.maximum(Nt, 1)
    loo = (St - S) / np.maximum(Nt - N, 1)
    used = N > 0
    g_b = used.sum(0)
    h = np.where(used, Nt / np.maximum(N, 1), 1.0)
    tj = g_b * theta - np.where(used, (1 - N / np.maximum(Nt, 1)) * loo, 0).sum(0)
    tau = h * theta - (h - 1) * loo
    var = np.where(used, (tau - tj) ** 2 / np.maximum(h - 1, 1e-9), 0).sum(0) / np.maximum(g_b, 1)
    se = np.sqrt(var)

    res = anc.assign(f3=theta, se=se, snps_shared=Nt.astype(int)).drop(columns=["assessment"])
    if ctrl:
        fc = Sc / np.maximum(Nc, 1)
        res["f3_ctrl"] = fc.mean(1)                        # what a typical Eastern European gets
        for pop in dict.fromkeys(ctrl_pop):
            res[f"f3_{pop}"] = fc[:, [q == pop for q in ctrl_pop]].mean(1)
        res["f3_diff"] = res["f3"] - res["f3_ctrl"]       # what is specific to you (SE ≈ your SE)
    res = res[res["snps_shared"] >= args.min_snps].sort_values("f3", ascending=False)
    res["rank"] = np.arange(1, len(res) + 1)
    path = out / "outgroup_f3_ancient.csv"
    res.to_csv(path, index=False)
    show = ["rank", "iid", "group", "date_bp", "country", "f3", "se", "snps_shared"]
    print(f"\nClosest ancient individuals, f3({OUTGROUP}; you, X), {len(res):,} with ≥ {args.min_snps:,} shared SNPs:")
    print(res[show].head(25).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n{path}")


if __name__ == "__main__":
    main()
