"""f4(Mbuti, X; Ukrainian, test) for the Uralic / Siberian question.

Positive when the test genome shares more drift with population X than Ukrainians do. If you carry real
Uralic or Siberian ancestry beyond what Ukrainians have, Siberian/Uralic X come out positive and Slavic X stay
near zero. The tests are you (`me_cram`, one read per SNP) and AADR moderns (config f4.controls) tested in
your place, with one random allele per SNP like your data and left out of the baseline when they are part of
it. Their spread shows what a typical Eastern European gets, so your values can be read against it.

Groups in config `f4` (AADR .ind labels). Output: out_dir/f4/f4_uralic.csv.
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


def main() -> None:
    parser = config_arg(__doc__)
    args = parser.parse_args()
    cfg = load_config(args.config)
    o = cfg["f4"]
    f = prep.aadr_files(cfg)
    out = cfg["out_dir"] / "f4"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    ind = prep.read_ind(f["ind"])
    ind.columns = ["id", "sex", "grp"]
    snp = pd.read_csv(f["snp"], sep=r"\s+", header=None, names=["id", "chrom", "gpos", "pos", "a1", "a2"],
                      dtype={"chrom": str})
    aadr = cfg["out_dir"] / "aadr"
    my_snp = pd.read_csv(aadr / "me_cram.snp", sep=r"\s+", header=None, usecols=[0], names=["id"])["id"]
    me = np.frombuffer(b"".join(l.strip() for l in open(aadr / "me_cram.geno", "rb")), np.uint8) - ord("0")
    full_i = pd.Series(np.arange(len(snp)), index=snp["id"]).reindex(my_snp).to_numpy()
    order = np.argsort(full_i)
    keep_i, me = full_i[order].astype(np.int64), me[order]
    sub = snp.iloc[keep_i]
    block = pd.factorize(sub["chrom"] + ":" + (sub["gpos"] * 100 // BLOCK_CM).astype(int).astype(str))[0]
    nb, L = block.max() + 1, len(keep_i)

    fmt, nind, nsnp = prep.geno_format(f["geno"])
    if fmt != "TGENO":
        raise SystemExit(f"{f['geno']}: {fmt} layout; expected the AADR v66 TGENO")
    rlen = max(48, -(-nsnp // 4))
    head = prep.header_bytes(f["geno"], rlen, nind)
    fh = open(f["geno"], "rb")

    def person(i: int) -> np.ndarray:  # copies of the first allele at your SNPs, 3 = missing
        fh.seek(head + rlen * int(i))
        return prep.unpack(np.frombuffer(fh.read(rlen), np.uint8), keep_i)

    def members(labels) -> np.ndarray:
        return np.flatnonzero(ind["grp"].isin(labels).to_numpy())

    def freq(idx) -> np.ndarray:
        """Allele frequency; pseudo-haploid people (no heterozygous calls) count one allele, others two."""
        a = np.zeros(L, np.float32)
        n = np.zeros(L, np.float32)
        for i in idx:
            g = person(i)
            ok = g != 3
            hap = not (g == 1).any()
            a += np.where(ok, g / 2.0 if hap else g, 0)
            n += ok * (1 if hap else 2)
        return np.where(n > 0, a / np.maximum(n, 1), np.nan)

    # Controls first, so they can be left out of every X group (otherwise a control is compared with itself)
    tests, tinfo = [np.where(me == 9, np.nan, me / 2.0)], [{"test": "you", "pop": "you"}]
    for pop, k in o.get("controls", {}).items():
        pool = members([pop])
        for i in rng.choice(pool, size=min(k, len(pool)), replace=False):
            g = person(i).astype(np.float32)
            hap = np.where(g == 1, rng.integers(0, 2, L), g / 2)  # one random allele at heterozygous sites
            tests.append(np.where(g == 3, np.nan, hap))
            tinfo.append({"test": ind.at[i, "id"], "pop": pop, "idx": int(i)})
    ctrl_idx = {t["idx"] for t in tinfo[1:]}

    # Left populations (outgroup, and a Slavic reference that cancels drift all Europeans share) and X groups
    lefts = {"Mbuti": freq(members(o["outgroup"]))}
    ref_name = o.get("contrast", {}).get("name")
    if ref_name:
        lefts[ref_name] = freq([i for i in members(o["contrast"]["labels"]) if i not in ctrl_idx])
    base_idx = members(o["baseline"])
    xs = [(cat, name, labels) for cat, d in o["tests"].items() for name, labels in d.items()]
    X, xinfo = [], []
    for cat, name, labels in xs:
        idx = [i for i in members(labels) if i not in ctrl_idx]
        if not idx:
            print(f"  {name}: no individuals with labels {labels}, skipped")
            continue
        X.append(freq(idx))
        xinfo.append({"category": cat, "X": name, "n": len(idx)})
        print(f"  X {name}: {len(idx)} people", flush=True)
    X = np.stack(X, axis=1)                                  # SNPs × populations
    base_sum = None
    # Baseline frequency, leaving out a control that belongs to it
    def base_freq(skip: int | None) -> np.ndarray:
        nonlocal base_sum
        if base_sum is None:
            a = np.zeros(L, np.float32)
            n = np.zeros(L, np.float32)
            per = {}
            for i in base_idx:
                g = person(i)
                ok = g != 3
                per[int(i)] = (np.where(ok, g, 0).astype(np.float32), ok * 2.0)
                a += per[int(i)][0]
                n += per[int(i)][1]
            base_sum = (a, n, per)
        a, n, per = base_sum
        if skip in per:
            a, n = a - per[skip][0], n - per[skip][1]
        return np.where(n > 0, a / np.maximum(n, 1), np.nan)

    print(f"Tests: you + {len(tests) - 1} controls; baseline {o['baseline']} ({len(base_idx)} people)", flush=True)
    u_all = base_freq(None)
    T = np.stack(tests, axis=1)                              # SNPs × tests
    U = np.stack([base_freq(t.get("idx")) if t.get("idx") in base_sum[2] else u_all for t in tinfo], axis=1)

    B = (U - T)                                              # SNPs × tests
    okB = np.isfinite(B)
    B0 = np.where(okB, B, 0)
    starts = np.r_[0, np.flatnonzero(block[1:] != block[:-1]) + 1, L]
    tables = []
    for left_name, lf in lefts.items():
        # f4 per block: sum over SNPs of (left - x)(u - t), all four called
        A = (lf[:, None] - X)                                # SNPs × X
        okA = np.isfinite(A)
        A0 = np.where(okA, A, 0)
        S = np.zeros((nb, X.shape[1], T.shape[1]))
        N = np.zeros_like(S)
        for b0, b1 in zip(starts[:-1], starts[1:]):
            k = block[b0]
            S[k] += A0[b0:b1].T @ B0[b0:b1]
            N[k] += okA[b0:b1].T.astype(np.float32) @ okB[b0:b1].astype(np.float32)
        theta, se = jackknife(S, N)
        z = theta / np.where(se > 0, se, np.nan)
        rows = [{**info, "left": left_name, "test": t["test"], "test_pop": t["pop"], "f4": theta[xi, ti],
                 "se": se[xi, ti], "z": z[xi, ti], "snps": int(N.sum(0)[xi, ti])}
                for xi, info in enumerate(xinfo) for ti, t in enumerate(tinfo)]
        tables.append(pd.DataFrame(rows))
    long = pd.concat(tables, ignore_index=True)
    long.to_csv(out / "f4_uralic_all_tests.csv", index=False)

    summ = []
    for left_name, d in long.groupby("left", sort=False):
        you = d[d["test"] == "you"].set_index("X")
        ctrl = d[d["test"] != "you"].groupby("X")["z"]
        s_ = you[["category", "left", "n", "f4", "se", "z", "snps"]].join(
            ctrl.mean().rename("ctrl_z_mean")).join(ctrl.std().rename("ctrl_z_sd")).join(
            ctrl.min().rename("ctrl_z_min")).join(ctrl.max().rename("ctrl_z_max"))
        s_["you_vs_ctrl_sd"] = (s_["z"] - s_["ctrl_z_mean"]) / s_["ctrl_z_sd"]
        summ.append(s_.reset_index())
    summ = pd.concat(summ, ignore_index=True)
    summ.to_csv(out / "f4_uralic.csv", index=False)
    pd.set_option("display.width", 200)
    for left_name, d in summ.groupby("left", sort=False):
        print(f"\nf4({left_name}, X; Ukrainian, test): Z > 3 = you share more with X than Ukrainians do")
        print(d[["category", "X", "n", "f4", "z", "ctrl_z_mean", "ctrl_z_min", "ctrl_z_max", "you_vs_ctrl_sd"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\n{out / 'f4_uralic.csv'}")


def jackknife(S: np.ndarray, N: np.ndarray):
    """Weighted block jackknife (Busing et al. 1999) over the first axis."""
    St, Nt = S.sum(0), N.sum(0)
    theta = St / np.maximum(Nt, 1)
    loo = (St - S) / np.maximum(Nt - N, 1)
    used = N > 0
    g_b = used.sum(0)
    h = np.where(used, Nt / np.maximum(N, 1), 1.0)
    tj = g_b * theta - np.where(used, (1 - N / np.maximum(Nt, 1)) * loo, 0).sum(0)
    tau = h * theta - (h - 1) * loo
    se = np.sqrt(np.where(used, (tau - tj) ** 2 / np.maximum(h - 1, 1e-9), 0).sum(0) / np.maximum(g_b, 1))
    return theta, se


if __name__ == "__main__":
    main()
