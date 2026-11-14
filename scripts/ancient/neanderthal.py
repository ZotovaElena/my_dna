"""Genome-wide Neanderthal ancestry: f4-ratio (Petr et al. 2019) on the AADR Human Origins SNPs.

  alpha = f4(Altai, Chimp; test, Yoruba) / f4(Altai, Chimp; Vindija, Yoruba)

The introgressing Neanderthals were closer to Vindija (Croatia) than to Altai, so the denominator is what a
test would give at 100% Neanderthal ancestry. Yoruba stand for no Neanderthal ancestry. All archaic genomes
are the high-coverage .DG versions in the AADR. Tests: you (me_cram, one read per SNP) and control individuals
made pseudo-haploid the same way (config neanderthal.controls). The HO SNP panel was ascertained in present-day
people, which biases f4-ratios a little; the controls on the same panel show where you stand regardless.

Also a Denisovan check: f4(Denisova, Chimp; test, Yoruba) / f4(Denisova, Chimp; Papuan, Yoruba) is near
0 for Europeans; it is shown with Papuans (~4-6% Denisovan) as the reference. Output: out_dir/neanderthal/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py and the module folders
from common import config_arg, load_config

from ancient import aadr_prepare as prep
BLOCK_CM = 5.0
ARCHAIC = {"Altai": "AltaiNeanderthal.DG", "Vindija": "Vindija.DG", "Chagyrskaya": "Chagyrskaya8.DG",
           "Denisova": "Denisova3.DG", "Chimp": "Chimp.REF"}
CONTROLS = {"Russian": 10, "French": 10, "Han": 10, "Papuan": 5, "Mende": 5, "Yoruba": 5}


def main() -> None:
    args = config_arg(__doc__).parse_args()
    cfg = load_config(args.config)
    controls = cfg.get("neanderthal", {}).get("controls", CONTROLS)
    out = cfg["out_dir"] / "neanderthal"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    f = prep.aadr_files(cfg)
    ind = prep.read_ind(f["ind"])
    snp = pd.read_csv(f["snp"], sep=r"\s+", header=None, names=["id", "chrom", "gpos", "pos", "a1", "a2"],
                      dtype={"chrom": str})
    aadr = cfg["out_dir"] / "aadr"
    merged = pd.read_csv(aadr / "merged.snp", sep=r"\s+", header=None, usecols=[0], names=["id"])["id"]
    keep_i = pd.Series(np.arange(len(snp)), index=snp["id"]).reindex(merged).to_numpy().astype(np.int64)
    me = np.frombuffer(b"".join(l.strip() for l in open(aadr / "me_cram.geno", "rb")), np.uint8) - ord("0")
    sub = snp.iloc[keep_i]
    block = pd.factorize(sub["chrom"] + ":" + (sub["gpos"] * 100 // BLOCK_CM).astype(int).astype(str))[0]
    starts = np.r_[0, np.flatnonzero(block[1:] != block[:-1]) + 1]
    L = len(keep_i)

    fmt, nind, nsnp = prep.geno_format(f["geno"])
    rlen = max(48, -(-nsnp // 4))
    head = prep.header_bytes(f["geno"], rlen, nind)
    fh = open(f["geno"], "rb")
    pos_of = {i: k for k, i in enumerate(ind["id"])}

    def person(i: int) -> np.ndarray:
        fh.seek(head + rlen * int(i))
        return prep.unpack(np.frombuffer(fh.read(rlen), np.uint8), keep_i)

    def freq1(i: int) -> np.ndarray:            # one diploid or haploid genome as an allele frequency
        g = person(i).astype(np.float64)
        return np.where(g == 3, np.nan, g / 2)

    def hap(i: int) -> np.ndarray:              # one random allele per SNP, like your data
        g = person(i).astype(np.float64)
        h = np.where(g == 1, rng.integers(0, 2, L), g / 2)
        return np.where(g == 3, np.nan, h)

    def group_freq(label: str, skip=()) -> np.ndarray:
        a = np.zeros(L)
        n = np.zeros(L)
        for i in np.flatnonzero(ind["group"].to_numpy() == label):
            if i in skip:
                continue
            g = person(i)
            ok = g != 3
            a += np.where(ok, g, 0)
            n += ok * 2
        return np.where(n > 0, a / np.maximum(n, 1), np.nan)

    arch = {k: freq1(pos_of[v]) for k, v in ARCHAIC.items()}
    tests = [("you", "you", np.where(me == 9, np.nan, me / 2.0))]
    picked = set()
    for pop, k in controls.items():
        pool = np.flatnonzero(ind["group"].to_numpy() == pop)
        for i in rng.choice(pool, size=min(k, len(pool)), replace=False):
            tests.append((ind.at[i, "id"], pop, hap(i)))
            picked.add(int(i))
    yri = group_freq("Yoruba", skip=picked)     # controls left out of the baseline
    papuan = group_freq("Papuan", skip=picked)

    def ratio(num_x: np.ndarray, den_x: np.ndarray, a: np.ndarray, base: np.ndarray):
        """f4(a, Chimp; x, base) / f4(a, Chimp; ref, base) with a weighted block jackknife over 5 cM blocks."""
        ch = arch["Chimp"]
        vn = (a - ch) * (num_x - base)
        vd = (a - ch) * (den_x - base)
        ok = np.isfinite(vn) & np.isfinite(vd)
        sn = np.add.reduceat(np.where(ok, vn, 0), starts)
        sd = np.add.reduceat(np.where(ok, vd, 0), starts)
        nb = np.add.reduceat(ok.astype(float), starts)
        used = nb > 0
        sn, sd, nb = sn[used], sd[used], nb[used]
        est = sn.sum() / sd.sum()
        loo = (sn.sum() - sn) / (sd.sum() - sd)
        g, N = len(nb), nb.sum()
        h = N / nb
        tj = g * est - ((1 - nb / N) * loo).sum()
        tau = h * est - (h - 1) * loo
        se = np.sqrt(((tau - tj) ** 2 / (h - 1)).sum() / g)
        return est, se, int(N)

    rows = []
    for name, pop, x in tests:
        nea, nse, n = ratio(x, arch["Vindija"], arch["Altai"], yri)
        den, dse, _ = ratio(x, papuan, arch["Denisova"], yri)
        rows.append({"test": name, "pop": pop, "neanderthal": nea, "neanderthal_se": nse, "snps": n,
                     "denisovan_vs_papuan": den, "denisovan_se": dse})
    d = pd.DataFrame(rows)
    d.to_csv(out / "neanderthal.csv", index=False)
    pd.set_option("display.width", 200)
    print("Neanderthal ancestry, f4(Altai, Chimp; X, Yoruba) / f4(Altai, Chimp; Vindija, Yoruba)")
    print(d[d["test"] == "you"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(d[d["test"] != "you"].groupby("pop")[["neanderthal", "denisovan_vs_papuan"]].agg(["mean", "min", "max"])
          .to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\n{out / 'neanderthal.csv'}")


if __name__ == "__main__":
    main()
