"""Farmer ancestry seen directly: f4(Mbuti, Anatolia_N; baseline, test) and admixture f3 for populations.

f4(Mbuti, Anatolia_N; B, test) is positive when the test genome shares more drift with Anatolian farmers than
baseline B does. With B = EHG or WHG (hunter-gatherers) or Yamnaya (steppe herders), all without Anatolian
farmer ancestry, a clearly positive value means the test carries farmer ancestry on top of B. The test sits in a
single position of the statistic, so one random read per SNP (your data) gives an unbiased value.

Tests: you (me_cram, me_csv), whole populations (config farmer.test_groups) and single control individuals made
pseudo-haploid like you (farmer.control_individuals): Sardinians (much farmer ancestry) and Nganasans (none)
bracket the scale, Russians and Ukrainians show where an Eastern European normally falls.

Admixture f3(target; A, B) < 0 shows the target is a mix of A and B. It needs a heterozygosity correction for
the target, which is not possible for a pseudo-haploid genome, so only diploid populations are targets.

Output: out_dir/farmer/f4_farmer.csv (one row per baseline and test) and admix_f3.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py and the module folders
from common import config_arg, load_config

from ancient import aadr_prepare as prep
from ancient import qpadm_fast as fast


def block_sums(v: np.ndarray, ok: np.ndarray, starts: np.ndarray):
    """Per jackknife block: sum of v and number of SNPs where ok (both 1-D, SNP order)."""
    return np.add.reduceat(np.where(ok, v, 0.0), starts), np.add.reduceat(ok.astype(np.float64), starts)


def jackknife(S: np.ndarray, N: np.ndarray) -> tuple[float, float]:
    """Weighted block jackknife (Busing et al. 1999) of sum(S) / sum(N)."""
    used = N > 0
    S, N = S[used], N[used]
    St, Nt = S.sum(), N.sum()
    theta = St / Nt
    loo = (St - S) / (Nt - N)
    g = len(S)
    h = Nt / N
    tj = g * theta - ((1 - N / Nt) * loo).sum()
    tau = h * theta - (h - 1) * loo
    return float(theta), float(np.sqrt(((tau - tj) ** 2 / (h - 1)).sum() / g))


def main() -> None:
    args = config_arg(__doc__).parse_args()
    cfg = load_config(args.config)
    o = cfg["farmer"]
    out = cfg["out_dir"] / "farmer"
    out.mkdir(parents=True, exist_ok=True)
    data = cfg["out_dir"] / "aadr" / "merged"
    counts = fast.load_counts(data, cfg["out_dir"] / "qpadm" / "fast" / "counts.npz", False)
    groups = [str(g) for g in counts["groups"]]
    gi = {g: i for i, g in enumerate(groups)}
    a, n, starts = counts["a"], counts["n"], counts["starts"]

    def freq(g: str) -> np.ndarray:
        k = gi[g]
        return np.where(n[k] > 0, a[k] / np.maximum(n[k], 1).astype(np.float64), np.nan)

    missing = [g for g in [o["outgroup"], o["farmer"], *o["baselines"], *o["test_groups"]] if g not in gi]
    if missing:
        raise SystemExit(f"Not in {data}.ind: {missing} (add them to aadr.groups and rerun ancient/aadr_prepare.py)")

    # Tests: you, whole populations, and control individuals with one random allele per SNP
    tests = [("you", "you (CRAM)", freq("me_cram"))]
    if "me_csv" in gi:
        tests.append(("you_csv", "you (MyHeritage CSV)", freq("me_csv")))
    tests += [(f"pop:{g}", g, freq(g)) for g in o["test_groups"]]
    ind = pd.read_csv(f"{data}.ind", sep=r"\s+", header=None, names=["id", "sex", "group"])
    snp_n = sum(1 for _ in open(f"{data}.snp"))
    rng = np.random.default_rng(11)
    picks = []
    for pop, k in o.get("control_individuals", {}).items():
        pool = np.flatnonzero(ind["group"].to_numpy() == pop)
        picks += [(int(i), pop) for i in rng.choice(pool, size=min(k, len(pool)), replace=False)]
    if picks:
        sel = np.array([i for i, _ in picks])
        G = np.concatenate(list(prep.aadr_genotypes(Path(f"{data}.geno"), len(ind), snp_n, sel, chunk=50_000)))
        for j, (i, pop) in enumerate(picks):
            g = G[:, j].astype(np.float64)
            hap = np.where(g == 1, rng.integers(0, 2, len(g)), g / 2)
            tests.append((f"ind:{pop}", f"{ind.at[i, 'id']} ({pop})", np.where(g == 9, np.nan, hap)))
        del G

    O, F = freq(o["outgroup"]), freq(o["farmer"])
    left = O - F
    rows = []
    for base in o["baselines"]:
        B = freq(base)
        for kind, name, T in tests:
            if kind == f"pop:{base}":
                continue
            v = left * (B - T)
            S, N = block_sums(v, np.isfinite(v), starts)
            f4, se = jackknife(S, N)
            rows.append({"baseline": base, "kind": kind.split(":")[0], "pop": kind.split(":")[-1], "test": name,
                         "f4": f4, "se": se, "z": f4 / se, "snps": int(N.sum())})
    f4 = pd.DataFrame(rows)
    f4.to_csv(out / "f4_farmer.csv", index=False)

    # Admixture f3 with the heterozygosity correction: (c-a)(c-b) - c(1-c)/(n_c-1), n_c alleles in the target
    rows = []
    for t in o["admix_f3"]["targets"]:
        if t not in gi:
            continue
        k = gi[t]
        nc = n[k].astype(np.float64)
        C = np.where(nc >= 2, a[k] / np.maximum(nc, 1), np.nan)
        for s1, s2 in o["admix_f3"]["pairs"]:
            A, B = freq(s1), freq(s2)
            v = (C - A) * (C - B) - C * (1 - C) / np.maximum(nc - 1, 1)
            S, N = block_sums(v, np.isfinite(v), starts)
            f3, se = jackknife(S, N)
            rows.append({"target": t, "source1": s1, "source2": s2, "f3": f3, "se": se, "z": f3 / se,
                         "snps": int(N.sum())})
    f3 = pd.DataFrame(rows)
    f3.to_csv(out / "admix_f3.csv", index=False)

    pd.set_option("display.width", 200)
    for base, d in f4.groupby("baseline", sort=False):
        print(f"\nf4({o['outgroup']}, {o['farmer']}; {base}, test): > 0 = more farmer-related than {base}")
        ind_ = d[d["kind"] == "ind"].groupby("pop")["z"].agg(["mean", "min", "max"])
        print(d[d["kind"] != "ind"][["test", "f4", "se", "z"]].to_string(index=False, float_format=lambda x: f"{x:.5g}"))
        print("  control individuals, Z:", ", ".join(f"{p} {r['mean']:.1f} [{r['min']:.1f}..{r['max']:.1f}]"
                                                   for p, r in ind_.iterrows()))
    print("\nAdmixture f3(target; A, B), Z < -3 = target is a mix of A and B")
    print(f3.to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    print(f"\n{out / 'f4_farmer.csv'}\n{out / 'admix_f3.csv'}")


if __name__ == "__main__":
    main()
