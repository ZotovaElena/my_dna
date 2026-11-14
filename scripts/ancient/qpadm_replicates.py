"""Repeat the pseudo-haploid draw: how much do qpAdm results move when another random read is picked?

`me_cram` uses one random read per SNP, a single draw; qpAdm's jackknife SE does not include that sampling
noise. This script piles up the CRAM once and keeps every read's allele at each AADR SNP, then draws R
pseudo-haploid replicates (one random read per SNP, a different seed each time) and reruns every me_cram
model of config `qpadm` with the fast NumPy engine (ancient/qpadm_fast.py) for each.

  python scripts/ancient/qpadm_replicates.py pileup          # once, a few minutes: read counts per SNP
  python scripts/ancient/qpadm_replicates.py run -r 30       # R replicates, ~70 s each on 6 cores

Same read filters as ancient/aadr_prepare.py (mapping quality >= 25, no BAQ). Differs from pileupCaller in one
detail: a read showing neither of the SNP's two alleles is skipped, rather than making the site missing.

Output in out_dir/qpadm/replicates/: replicates.csv (every model in every replicate) and summary.csv (per
model: mean, SD, min and max of each weight, p-value quantiles, share of replicates in which it fits).
"""

import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py and the module folders
from common import config_arg, load_config

from ancient import aadr_prepare as prep
from ancient import qpadm_fast as fast
from ancient import qpadm as slow

INDEL = re.compile(r"[+-](\d+)")


def count_bases(bases: str, ref: str) -> dict[str, int]:
    """Allele counts from an mpileup read-base column ('.' ',' = reference; skips ^Q, $, indels, '*')."""
    out = {"A": 0, "C": 0, "G": 0, "T": 0}
    i, n = 0, len(bases)
    while i < n:
        c = bases[i]
        if c == "^":
            i += 2
            continue
        if c in "+-":
            m = INDEL.match(bases, i)
            i = m.end() + int(m.group(1))
            continue
        if c in ".,":
            out[ref] = out.get(ref, 0) + 1
        else:
            u = c.upper()
            if u in out:
                out[u] += 1
        i += 1
    return out


def pileup(cfg: dict, work: Path) -> None:
    aadr = cfg["out_dir"] / "aadr"
    snp = pd.read_csv(aadr / "merged.snp", sep=r"\s+", header=None, names=["id", "chrom", "gen", "pos", "a1", "a2"],
                      dtype={"chrom": str})
    key = {(c, p): i for i, (c, p) in enumerate(zip(snp["chrom"], snp["pos"]))}
    n1 = np.zeros(len(snp), np.uint16)
    n2 = np.zeros(len(snp), np.uint16)
    q = lambda p: shlex.quote(str(p))  # noqa: E731
    cmd = (f"samtools mpileup -B -q {prep.MIN_MAPQ} -Q 0 -l {q(aadr / 'auto.pos')} -f {q(cfg['reference_fasta'])} "
           f"{q(cfg['cram'])}")
    print("Piling up the CRAM at the AADR SNPs (one pass)...", flush=True)
    t0, lines = time.time(), 0
    a1, a2 = snp["a1"].to_numpy(), snp["a2"].to_numpy()
    with subprocess.Popen(["bash", "-o", "pipefail", "-c", cmd], stdout=subprocess.PIPE, text=True,
                          stderr=subprocess.DEVNULL) as p:
        for line in p.stdout:
            chrom, pos, ref, _, bases, *_ = line.rstrip("\n").split("\t")
            i = key.get((chrom, int(pos)))
            if i is None:
                continue
            c = count_bases(bases, ref.upper())
            n1[i], n2[i] = c.get(a1[i], 0), c.get(a2[i], 0)
            lines += 1
    if p.returncode:
        raise SystemExit(f"samtools mpileup failed ({p.returncode})")
    np.savez(work / "reads.npz", n1=n1, n2=n2)
    cov = (n1 + n2) > 0
    print(f"  {lines:,} SNPs with reads, {cov.sum():,} with a read for one of the two alleles; "
          f"depth there: mean {(n1 + n2)[cov].mean():.2f} ({time.time() - t0:.0f} s) -> {work / 'reads.npz'}")
    # Check against the original single draw: where only one allele was seen, both must agree
    me = np.frombuffer(b"".join(l.strip() for l in open(aadr / "me_cram.geno", "rb")), np.uint8) - ord("0")
    one = cov & ((n1 == 0) | (n2 == 0)) & (me != 9)
    agree = np.where(n1[one] > 0, 2, 0) == me[one]
    print(f"  agreement with me_cram where only one allele was seen: {agree.mean():.4%} of {one.sum():,} SNPs")


def draw(n1: np.ndarray, n2: np.ndarray, seed: int) -> np.ndarray:
    """One random read per SNP: frequency of allele 1 (0 or 1), NaN without reads."""
    rng = np.random.default_rng(seed)
    tot = n1.astype(np.int64) + n2
    pick = rng.random(len(tot)) * tot < n1                 # P(allele 1) = n1 / (n1 + n2)
    return np.where(tot > 0, pick.astype(np.float64), np.nan)


def run(cfg: dict, work: Path, reps: int, jobs: int) -> None:
    z = np.load(work / "reads.npz")
    n1, n2 = z["n1"], z["n2"]
    q = cfg["qpadm"]
    qdir = cfg["out_dir"] / "qpadm"
    counts = fast.load_counts(cfg["out_dir"] / "aadr" / "merged", qdir / "fast" / "counts.npz", False)
    groups = [str(g) for g in counts["groups"]]
    index = {g: i for i, g in enumerate(groups)}
    with np.errstate(invalid="ignore", divide="ignore"):
        freq = np.where(counts["n"] > 0, counts["a"] / counts["n"].astype(np.float64), np.nan)
    tasks = [(name, "me_cram", s, slow.right_for(a, s)) for name, a in q["analyses"].items()
             if "me_cram" in a.get("targets", q["targets"]) for s in slow.models(a)]
    print(f"{reps} replicates × {len(tasks)} me_cram models on {jobs} cores", flush=True)
    out_long = work / "replicates.csv"
    done = pd.read_csv(out_long) if out_long.exists() else pd.DataFrame(columns=["rep"])
    rows = [done] if len(done) else []
    for r in range(1, reps + 1):
        if r in set(done["rep"]):
            continue
        t0 = time.time()
        freq[index["me_cram"]] = draw(n1, n2, seed=r)
        with ProcessPoolExecutor(jobs, initializer=fast._init, initargs=(freq, counts["starts"], index)) as pool:
            res = list(pool.map(fast.evaluate, tasks, chunksize=4))
        rep = pd.DataFrame([{"rep": r, "analysis": n, "n_sources": len(s), "sources": " + ".join(s), "p": x["p"],
                             "weights": " ".join(f"{w:.4f}" for w in x["weights"]),
                             "se": " ".join(f"{e:.4f}" for e in x["se"]), "status": x["status"],
                             "fits": x["status"] == "ok" and x["p"] >= q["p_min"]
                             and all(0 <= w <= 1 for w in x["weights"])}
                            for (n, _, s, _), x in zip(tasks, res)])
        rows.append(rep)
        pd.concat(rows, ignore_index=True).to_csv(out_long, index=False)   # saved after every replicate
        print(f"  replicate {r}/{reps}: {rep['fits'].sum()} of {len(rep)} models fit ({time.time() - t0:.0f} s)",
              flush=True)
    summarise(pd.concat(rows, ignore_index=True), work, q["p_min"])


def summarise(df: pd.DataFrame, work: Path, p_min: float) -> None:
    out = []
    for (analysis, sources), d in df.groupby(["analysis", "sources"], sort=False):
        W = np.array([[float(x) for x in w.split()] for w in d["weights"]])
        SE = np.array([[float(x) for x in s.split()] for s in d["se"]])
        p = d["p"].to_numpy()
        row = {"analysis": analysis, "sources": sources, "n_sources": int(d["n_sources"].iloc[0]), "replicates": len(d),
               "fit_share": d["fits"].mean(), "p_median": np.nanmedian(p), "p_q05": np.nanquantile(p, 0.05),
               "p_q95": np.nanquantile(p, 0.95), "p_min": np.nanmin(p), "p_max": np.nanmax(p)}
        for j, s in enumerate(sources.split(" + ")):
            row[f"w{j + 1}_mean"] = W[:, j].mean()
            row[f"w{j + 1}_sd"] = W[:, j].std(ddof=1) if len(W) > 1 else np.nan   # between draws
            row[f"w{j + 1}_min"], row[f"w{j + 1}_max"] = W[:, j].min(), W[:, j].max()
            row[f"w{j + 1}_jk_se"] = SE[:, j].mean()                              # within a draw
        out.append(row)
    s = pd.DataFrame(out)
    s.to_csv(work / "summary.csv", index=False)
    show = s[s["fit_share"] > 0].sort_values(["analysis", "n_sources", "fit_share", "p_median"],
                                             ascending=[True, True, False, False])
    pd.set_option("display.width", 220)
    for analysis, d in show.groupby("analysis", sort=False):
        print(f"\n== {analysis}: models that fit in at least one replicate (p ≥ {p_min}, weights in [0, 1])")
        for _, r in d.head(8).iterrows():
            ws = " + ".join(f"{src} {r[f'w{j + 1}_mean']:.3f}±{r[f'w{j + 1}_sd']:.3f} (jk {r[f'w{j + 1}_jk_se']:.3f})"
                            for j, src in enumerate(r["sources"].split(" + ")))
            print(f"  fits {r['fit_share']:5.0%}  p {r['p_median']:.3g} [{r['p_q05']:.2g}–{r['p_q95']:.2g}]  {ws}")
    print(f"\n{work / 'summary.csv'}")


def main() -> None:
    parser = config_arg(__doc__)
    parser.add_argument("step", choices=["pileup", "run", "summary"])
    parser.add_argument("-r", "--replicates", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()
    cfg = load_config(args.config)
    work = cfg["out_dir"] / "qpadm" / "replicates"
    work.mkdir(parents=True, exist_ok=True)
    if args.step == "pileup":
        pileup(cfg, work)
    elif args.step == "run":
        run(cfg, work, args.replicates, args.jobs)
    else:
        summarise(pd.read_csv(work / "replicates.csv"), work, cfg["qpadm"]["p_min"])


if __name__ == "__main__":
    main()
