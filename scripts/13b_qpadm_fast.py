"""Fast qpAdm in NumPy: the same models as 13_qpadm.py in seconds instead of hours, checked against ADMIXTOOLS.

ADMIXTOOLS' qpAdm/qpfstats are single-threaded and recompute f-statistics SNP by SNP for every model.
This script reads the genotypes once, caches per-group allele counts (out_dir/qpadm/fast/counts.npz), and
evaluates every model with block matrix products on all CPU cores. The method follows ADMIXTOOLS:

  * f4(target, source_i; right_0, right_j) for every source i and right group j. Each statistic uses the
    SNPs covered in its own 4 groups (ADMIXTOOLS `allsnps: YES`). f4 needs no bias correction.
  * Pseudo-haploid samples (no heterozygous calls: ancient genomes, me_cram) count one allele per SNP.
  * Weighted block jackknife (Busing et al. 1999) over 5 cM blocks (ADMIXTOOLS `blgsize: 0.05`) for the
    covariance of the f4 matrix.
  * Rank test: the target is a mix of the k sources iff the k × m f4 matrix has rank k-1. The fit
    minimises the chi-square of (X - A·B) by alternating generalised least squares; p = chi-square tail
    with (k - r)(m - r) degrees of freedom. Weights are the null vector of A, summing to 1; their standard
    errors come from refitting every leave-one-block-out matrix.

Writes out_dir/qpadm/qpadm_fast_results.csv (same columns as 13_qpadm.py) and, where ADMIXTOOLS results
exist, out_dir/qpadm/fast_vs_admixtools.csv plus a printed agreement summary.
Use the results in the report with: python scripts/14_report.py --qpadm-results qpadm_fast_results.csv
"""

import hashlib
import importlib
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from common import config_arg, load_config

sys.path.insert(0, str(Path(__file__).parent))
prep = importlib.import_module("12_aadr_prepare")   # genotype reader
slow = importlib.import_module("13_qpadm")          # model list, rotation, report

BLOCK_MORGANS = 0.05

# ---------------------------------------------------------------- data: allele counts per group

def dataset_key(data: Path) -> str:
    h = hashlib.sha1()
    for ext in ("geno", "snp", "ind"):
        st = Path(f"{data}.{ext}").stat()
        h.update(f"{ext}:{st.st_size}:{st.st_mtime_ns};".encode())
    return h.hexdigest()[:16]


def load_counts(data: Path, cache: Path, force: bool) -> dict:
    """Per group and SNP: alleles observed (n) and copies of allele 1 (a). Cached by dataset size and date."""
    key = dataset_key(data)
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            return {k: z[k] for k in z.files}
    t0 = time.time()
    ind = pd.read_csv(f"{data}.ind", sep=r"\s+", header=None, names=["id", "sex", "group"])
    snp = pd.read_csv(f"{data}.snp", sep=r"\s+", header=None, names=["id", "chrom", "gen", "pos", "a1", "a2"])
    groups = np.array(sorted(ind["group"].unique()))
    gi = np.searchsorted(groups, ind["group"].to_numpy())
    member = np.zeros((len(ind), len(groups)), np.float32)
    member[np.arange(len(ind)), gi] = 1
    n_ind, n_snp = len(ind), len(snp)
    print(f"Reading {n_ind} samples × {n_snp:,} SNPs once (cached afterwards)...", flush=True)

    # First pass: which samples are pseudo-haploid (no heterozygous call anywhere)
    het = np.zeros(n_ind, np.int64)
    called = np.zeros(n_ind, np.int64)
    for g in prep.aadr_genotypes(Path(f"{data}.geno"), n_ind, n_snp, np.arange(n_ind), chunk=20_000):
        het += (g == 1).sum(axis=0)
        called += (g != 9).sum(axis=0)
    haploid = (het == 0) & (called > 0)
    ploidy = np.where(haploid, 1, 2).astype(np.float32)

    a = np.zeros((len(groups), n_snp), np.uint16)
    n = np.zeros((len(groups), n_snp), np.uint16)
    pos = 0
    for g in prep.aadr_genotypes(Path(f"{data}.geno"), n_ind, n_snp, np.arange(n_ind), chunk=20_000):
        ok = (g != 9) & ~((g == 1) & haploid)             # a stray het in a haploid sample: drop
        alleles = np.where(ok, ploidy, 0).astype(np.float32)
        copies = np.where(ok, np.where(haploid, g / 2, g), 0).astype(np.float32)
        k = len(g)
        n[:, pos:pos + k] = (alleles @ member).T.round().astype(np.uint16)
        a[:, pos:pos + k] = (copies @ member).T.round().astype(np.uint16)
        pos += k

    # 5 cM jackknife blocks within chromosomes (SNPs are sorted by chromosome and position)
    chrom = snp["chrom"].astype(str).to_numpy()
    blk = np.floor(snp["gen"].to_numpy() / BLOCK_MORGANS).astype(np.int64)
    change = np.r_[True, (chrom[1:] != chrom[:-1]) | (blk[1:] != blk[:-1])]
    starts = np.flatnonzero(change)
    out = {"key": np.array(key), "groups": groups, "a": a, "n": n, "starts": starts,
           "haploid_groups": np.array(sorted(set(ind["group"][haploid])))}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    print(f"  {len(groups)} groups, {len(starts)} jackknife blocks, {haploid.sum()} pseudo-haploid samples "
          f"({time.time() - t0:.0f} s)", flush=True)
    return out


# ---------------------------------------------------------------- statistics

def chi2_sf(x: float, k: int) -> float:
    """Upper tail of the chi-square distribution with integer k degrees of freedom (no SciPy needed)."""
    if not np.isfinite(x):
        return math.nan
    if x <= 0:
        return 1.0
    if k % 2 == 0:
        term = math.exp(-x / 2)
        s = term
        for i in range(1, k // 2):
            term *= (x / 2) / i
            s += term
    else:
        s = math.erfc(math.sqrt(x / 2))
        term = math.sqrt(2 * x / math.pi) * math.exp(-x / 2)
        for i in range(1, (k - 1) // 2 + 1):
            s += term
            term *= x / (2 * i + 1)
    return min(1.0, s)


def gls(M: np.ndarray, Qi: np.ndarray, x: np.ndarray) -> np.ndarray:
    MtQi = M.T @ Qi
    return np.linalg.lstsq(MtQi @ M, MtQi @ x, rcond=None)[0]


def fit_rank(X: np.ndarray, Qi: np.ndarray, r: int, A: np.ndarray | None = None, iters: int = 100):
    """Minimise chi2 = (x - vec(A B))' Qi (x - vec(A B)) with A k×r, B r×m (row-major vec)."""
    k, m = X.shape
    x = X.ravel()
    if r == 0:
        return float(x @ Qi @ x), np.zeros((k, 0))
    if A is None:
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        A = U[:, :r] * s[:r]
    Ik, Im = np.eye(k), np.eye(m)
    prev = math.inf
    for _ in range(iters):
        B = gls(np.kron(A, Im), Qi, x).reshape(r, m)
        A = gls(np.kron(Ik, B.T), Qi, x).reshape(k, r)
        res = x - (A @ B).ravel()
        chi2 = float(res @ Qi @ res)
        if prev - chi2 < 1e-9 * max(1.0, chi2):
            break
        prev = chi2
    return chi2, A


def weights_from(A: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return np.ones(1)
    w = np.linalg.svd(A.T)[2][-1]          # null vector of A': w' A = 0
    return w / w.sum()


def jackknife(theta: np.ndarray, loo: np.ndarray, wb: np.ndarray):
    """Weighted block jackknife: covariance of theta from leave-one-block-out estimates."""
    h = 1 / wb
    B = len(wb)
    tau = h[:, None] * theta[None] - (h - 1)[:, None] * loo
    theta_j = B * theta - ((1 - wb)[:, None] * loo).sum(axis=0)
    d = tau - theta_j[None]
    return (d.T * (1 / (h - 1))) @ d / B


# ---------------------------------------------------------------- one model (runs in worker processes)

_G = {}


def _init(freq: np.ndarray, starts: np.ndarray, index: dict):
    _G.update(freq=freq, starts=starts, index=index)


def evaluate(task: tuple) -> dict:
    name, target, sources, right = task
    P, starts, idx = _G["freq"], _G["starts"], _G["index"]
    missing = [g for g in [target, *sources, *right] if g not in idx]
    if missing:
        return {"p": math.nan, "weights": [math.nan] * len(sources), "se": [math.nan] * len(sources),
                "status": f"not in dataset: {', '.join(missing)}"}
    k, m = len(sources), len(right) - 1
    t, r0 = P[idx[target]], P[idx[right[0]]]
    DL = t[None] - P[[idx[s] for s in sources]]                 # k × snps
    DR = r0[None] - P[[idx[r] for r in right[1:]]]              # m × snps
    ML, MR = ~np.isnan(DL), ~np.isnan(DR)
    DL, DR = np.where(ML, DL, 0), np.where(MR, DR, 0)
    ML, MR = ML.astype(np.float32), MR.astype(np.float32)  # counts: exact in float32
    bounds = np.r_[starts, DL.shape[1]]
    S = np.empty((len(starts), k, m))
    N = np.empty((len(starts), k, m))
    for b in range(len(starts)):
        sl = slice(bounds[b], bounds[b + 1])
        S[b] = DL[:, sl] @ DR[:, sl].T
        N[b] = ML[:, sl] @ MR[:, sl].T  # SNPs shared by all 4 groups
    S, N = S.reshape(len(starts), -1), N.reshape(len(starts), -1)
    nb = N.mean(axis=1)
    keep = nb > 0
    S, N, nb = S[keep], N[keep], nb[keep]
    Stot, Ntot = S.sum(axis=0), N.sum(axis=0)
    if (Ntot == 0).any():
        return {"p": math.nan, "weights": [math.nan] * k, "se": [math.nan] * k,
                "status": "no shared SNPs for some f4-statistic"}
    theta = Stot / Ntot
    with np.errstate(divide="ignore", invalid="ignore"):
        loo = (Stot[None] - S) / (Ntot[None] - N)
    loo = np.where(np.isfinite(loo), loo, theta[None])
    wb = nb / nb.sum()
    Q = jackknife(theta, loo, wb)
    Qi = np.linalg.pinv(Q, hermitian=True)

    X = theta.reshape(k, m)
    r = k - 1
    chi2, A = fit_rank(X, Qi, r)
    dof = (k - r) * (m - r)
    w = weights_from(A, k)
    if k == 1:
        se = np.zeros(1)
    else:
        wl = np.empty((len(loo), k))
        for b in range(len(loo)):
            _, Ab = fit_rank(loo[b].reshape(k, m), Qi, r, A=A.copy(), iters=8)
            wl[b] = weights_from(Ab, k)
        se = np.sqrt(np.diag(jackknife(w, wl, wb)))
    return {"p": chi2_sf(chi2, dof), "weights": w.tolist(), "se": se.tolist(), "status": "ok",
            "chisq": chi2, "dof": dof}


# ---------------------------------------------------------------- compare with ADMIXTOOLS

def compare(fast: pd.DataFrame, ref_path: Path, p_min: float, out: Path) -> None:
    if not ref_path.exists():
        print("\nNo ADMIXTOOLS results to compare with.")
        return
    ref = pd.read_csv(ref_path)
    key = ["analysis", "target", "sources"]
    f = fast.assign(w=fast["weights"].map(lambda v: [float(x) for x in v]))
    ref["w"] = ref["weights"].map(lambda s: [float(x) for x in str(s).split()])
    one = ref["n_sources"] == 1
    ref.loc[one, "w"] = pd.Series([[1.0]] * one.sum(), index=ref.index[one], dtype=object)
    both = f.merge(ref[key + ["p", "w", "fits"]], on=key, suffixes=("_fast", "_admixtools"))
    if both.empty:
        print("\nNo models in common with the ADMIXTOOLS results.")
        return
    lp = lambda p: np.log10(np.clip(p.astype(float), 1e-300, 1))
    both["log10p_diff"] = lp(both["p_fast"]) - lp(both["p_admixtools"])
    both["max_weight_diff"] = [max(abs(a - b) for a, b in zip(x, y)) if len(x) == len(y) else math.nan
                               for x, y in zip(both["w_fast"], both["w_admixtools"])]
    both[key + ["p_fast", "p_admixtools", "log10p_diff", "max_weight_diff", "fits_fast", "fits_admixtools"]] \
        .to_csv(out, index=False)
    ok = np.isfinite(both["log10p_diff"])
    r = np.corrcoef(lp(both.loc[ok, "p_fast"]), lp(both.loc[ok, "p_admixtools"]))[0, 1]
    agree = (both["fits_fast"] == both["fits_admixtools"]).mean()
    print(f"\n== Agreement with ADMIXTOOLS over {len(both)} models ==")
    print(f"  log10 p: correlation {r:.4f}, median |difference| {both['log10p_diff'].abs().median():.3f}")
    print(f"  weights: median max |difference| {both['max_weight_diff'].median():.4f}, "
          f"95th percentile {both['max_weight_diff'].quantile(0.95):.4f}")
    print(f"  same fit / no-fit verdict (p ≥ {p_min}, weights in [0,1]): {agree:.1%}")
    dis = both[both["fits_fast"] != both["fits_admixtools"]]
    if len(dis):
        print(f"  {len(dis)} models change verdict, e.g.:")
        print(dis.head(8)[key + ["p_fast", "p_admixtools"]].to_string(index=False,
              float_format=lambda v: f"{v:.3g}"))
    print(f"  Details: {out}")


# ---------------------------------------------------------------- main

def main() -> None:
    parser = config_arg(__doc__)
    parser.add_argument("--analysis", nargs="+", help="subset of qpadm.analyses (default: all)")
    parser.add_argument("--target", nargs="+", help="subset of qpadm.targets (default: all)")
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1),
                        help="worker processes (default 6: each needs ~400 MB with ~30 outgroups)")
    parser.add_argument("--force", action="store_true", help="re-read the genotypes instead of the cache")
    args = parser.parse_args()
    cfg = load_config(args.config)
    q = cfg["qpadm"]
    data = cfg["out_dir"] / "aadr" / "merged"
    qdir = cfg["out_dir"] / "qpadm"
    if not Path(f"{data}.geno").exists():
        raise SystemExit(f"{data}.geno not found: run `python scripts/12_aadr_prepare.py build` first")

    t0 = time.time()
    counts = load_counts(data, qdir / "fast" / "counts.npz", args.force)
    groups = [str(g) for g in counts["groups"]]
    with np.errstate(invalid="ignore", divide="ignore"):
        freq = np.where(counts["n"] > 0, counts["a"] / counts["n"].astype(np.float64), np.nan)

    analyses = {k: v for k, v in q["analyses"].items() if not args.analysis or k in args.analysis}
    tasks = []
    for name, a in analyses.items():
        targets = a.get("targets", q["targets"])
        if args.target:
            targets = [t for t in targets if t in args.target]
        tasks += [(name, t, s, slow.right_for(a, s)) for t in targets for s in slow.models(a)]
    print(f"{len(tasks)} qpAdm models on {args.jobs} cores...", flush=True)

    t1 = time.time()
    with ProcessPoolExecutor(args.jobs, initializer=_init,
                             initargs=(freq, counts["starts"], {g: i for i, g in enumerate(groups)})) as pool:
        results = list(pool.map(evaluate, tasks, chunksize=4))
    print(f"  done in {time.time() - t1:.0f} s (total {time.time() - t0:.0f} s)")

    rows = []
    for (name, target, sources, _), res in zip(tasks, results):
        fits = res["status"] == "ok" and res["p"] >= q["p_min"] and all(0 <= w <= 1 for w in res["weights"])
        rows.append({"analysis": name, "target": target, "n_sources": len(sources),
                     "sources": " + ".join(sources), "p": res["p"], "weights": res["weights"], "se": res["se"],
                     "status": res["status"], "fits": fits})
    df = pd.DataFrame(rows)
    out = qdir / "qpadm_fast_results.csv"
    df.assign(weights=df["weights"].map(lambda w: " ".join(f"{x:.4f}" for x in w)),
              se=df["se"].map(lambda s: " ".join(f"{x:.4f}" for x in s))).to_csv(out, index=False)
    slow.report(df, q["p_min"])
    print(f"\nAll models: {out}")
    compare(df, qdir / "qpadm_results.csv", q["p_min"], qdir / "fast_vs_admixtools.csv")


if __name__ == "__main__":
    main()
