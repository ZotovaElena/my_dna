"""qpAdm (ADMIXTOOLS) model rotation: you = a mix of 1..max_sources sources, per target and analysis.

Reads out_dir/aadr/merged.* (from ancient/aadr_prepare.py) and the `qpadm` config section. Each model runs
in its own folder under out_dir/qpadm/runs/ (par file, left/right lists, log). A finished model is reused
on rerun as long as its sources, outgroups and settings are unchanged; shared f-statistics are cached in
runs/_fstats/ by group list. After rebuilding the data with *different genotypes* for an existing group
(e.g. `ancient/aadr_prepare.py build --redo-cram`), rerun with --force. A model fits when p >= p_min and every
weight is within [0, 1].

p is qpAdm's tail probability for "the sources explain the target's f4-statistics against the right
populations". A low p rejects the model; a high p means it is not rejected, not that it is proven.
Prefer the simplest fitting model, and compare me_cram with me_csv.
"""

import hashlib
import itertools
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py and the module folders
from common import config_arg, load_config


def models(analysis: dict):
    for k in range(1, analysis["max_sources"] + 1):
        yield from itertools.combinations(analysis["sources"], k)


def right_for(analysis: dict, sources: tuple) -> list:
    """Outgroups of one model. With `rotate: true`, the candidates not used as sources join the right set,
    so a model fits only if it explains the target better than every other candidate (Harney et al. 2021)."""
    if not analysis.get("rotate"):
        return analysis["right"]
    return analysis["right"] + [c for c in analysis["sources"] if c not in sources]


def parse(text: str, n: int) -> dict:
    """p-value, weights and standard errors from a qpAdm log ('summ:' and 'std. errors:' lines)."""
    nan = [math.nan] * n
    summ = re.search(r"^summ:\s+(.+)$", text, re.M)
    if not summ:
        last = [line for line in text.splitlines() if line.strip()][-1:] or ["empty log"]
        return {"p": math.nan, "weights": nan, "se": nan, "status": f"no result: {last[0][:120]}"}
    tok = summ.group(1).split()  # target, n_sources, tail prob (6 decimals), weights..., covariances...
    se = re.search(r"std\. errors:\s+(.+)$", text, re.M)
    # Full-precision p: the rank n-1 test ("f4rank: n-1 dof: .. chisq: .. tail: p")
    rank = re.search(rf"^f4rank:\s+{n - 1}\s.*?\stail:\s+(\S+)", text, re.M)
    weights = [float(x) for x in tok[3:3 + n]]
    errors = [float(x) for x in se.group(1).split()[:n]] if se else nan
    if n == 1:  # ADMIXTOOLS 8 prints 0.000 for a single source; its weight is 1 by definition
        weights, errors = [1.0], [0.0]
    return {
        "p": float(rank.group(1)) if rank else float(tok[2]),
        "weights": weights,
        "se": errors,
        "status": "ok",
    }


def data_par(data: Path) -> str:
    return (f"genotypename: {data}.geno\nsnpname: {data}.snp\nindivname: {data}.ind\n"
            "allsnps: YES\n"  # each f4 uses all SNPs covered in its 4 groups (standard with ancient data)
            "inbreed: NO\n")  # pseudo-haploid data; ADMIXTOOLS 8 otherwise rejects single-sample groups


def shared_fstats(data: Path, runs: Path, pops: list, force: bool) -> Path:
    """Compute f-statistics for a set of populations once (qpfstats), for every model to reuse.
    Without this each qpAdm run computes its own: with ~30 groups that is >10 min and ~2 GB per model.

    Cached in runs/_fstats/<key>/, where the key is a hash of the population list and the settings, so any
    analysis with the same groups reuses it and a changed group list gets a fresh file."""
    poplist = "\n".join(pops) + "\n"
    key = hashlib.sha1((poplist + data_par(data)).encode()).hexdigest()[:12]
    cdir = runs / "_fstats" / key
    out = cdir / "fstats.txt"
    if not out.exists():  # files from before this cache layout: runs/<analysis>/<target>/fstats.txt
        for old in runs.glob("*/*/fstats.txt"):
            if (old.parent / "poplist").exists() and (old.parent / "poplist").read_text() == poplist:
                cdir.mkdir(parents=True, exist_ok=True)
                for name in ("fstats.txt", "poplist", "qpfstats.par", "qpfstats.log"):
                    if (old.parent / name).exists():
                        (old.parent / name).rename(cdir / name)
                break
    if out.exists() and out.stat().st_size and not force:
        return out
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "poplist").write_text(poplist)
    tmp = cdir / "fstats.tmp"
    (cdir / "qpfstats.par").write_text(data_par(data) + f"poplistname: {cdir / 'poplist'}\nfstatsoutname: {tmp}\n")
    print(f"Computing shared f-statistics for {len(pops)} groups in {cdir} (one run, can take an hour)...",
          flush=True)
    with open(cdir / "qpfstats.log", "w") as f:
        subprocess.run(["qpfstats", "-p", str(cdir / "qpfstats.par")], stdout=f, stderr=subprocess.STDOUT)
    if not tmp.exists() or not tmp.stat().st_size:
        raise SystemExit(f"qpfstats failed, see {cdir / 'qpfstats.log'}")
    tmp.rename(out)
    return out


def is_cached(run_dir: Path, files: dict[str, str]) -> bool:
    """A finished result whose inputs (left, right, par) are the same as now. Par lines are compared as a set,
    so results from older versions of this script that wrote them in another order still count."""
    log = run_dir / "qpadm.log"
    if not log.exists() or not re.search(r"^summ:", log.read_text(errors="replace"), re.M):
        return False
    for name, text in files.items():
        path = run_dir / name
        if not path.exists():
            return False
        old = path.read_text()
        if (set(old.splitlines()) != set(text.splitlines())) if name == "par" else old != text:
            return False
    return True


def run_model(data: Path, run_dir: Path, target: str, sources: tuple, right: list, force: bool,
              fstats: Path | None = None) -> dict:
    log = run_dir / "qpadm.log"
    files = {
        "left": "\n".join([target, *sources]) + "\n",
        "right": "\n".join(right) + "\n",
        "par": data_par(data) + f"popleft: {run_dir / 'left'}\npopright: {run_dir / 'right'}\n"
               + (f"fstatsname: {fstats}\n" if fstats else ""),
    }
    if force or not is_cached(run_dir, files):
        run_dir.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (run_dir / name).write_text(text)
        with open(log, "w") as f:
            subprocess.run(["qpAdm", "-p", str(run_dir / "par")], stdout=f, stderr=subprocess.STDOUT)
    return parse(log.read_text(errors="replace"), len(sources))


def fmt_model(row) -> str:
    parts = [f"{s} {w:.2f}±{e:.2f}" for s, w, e in zip(row["sources"].split(" + "), row["weights"], row["se"])]
    return " + ".join(parts)


def report(df: pd.DataFrame, p_min: float, top: int = 12) -> None:
    for analysis, a in df.groupby("analysis", sort=False):
        for target, t in a.groupby("target", sort=False):
            fits = t[t["fits"]].sort_values(["n_sources", "p"], ascending=[True, False])
            print(f"\n== {analysis} | {target}: {len(fits)} of {len(t)} models fit (p ≥ {p_min}, weights in [0, 1])")
            for _, r in fits.head(top).iterrows():
                print(f"  p={r['p']:.3g}  {fmt_model(r)}")
            failed = t[t["status"] != "ok"]
            if len(failed):
                print(f"  ({len(failed)} runs gave no result, e.g. {failed['status'].iloc[0]})")
        # Same model, both inputs: does the imputed CSV tell the same story as the raw reads?
        wide = a.pivot_table(index="sources", columns="target", values="p", aggfunc="first")
        fit_any = a.groupby("sources")["fits"].any()
        wide = wide[fit_any.reindex(wide.index, fill_value=False)]
        if len(wide) and wide.shape[1] > 1:
            print(f"\n   {analysis}: p for models that fit with either input")
            print(wide.sort_values(wide.columns[0], ascending=False).head(top)
                  .to_string(float_format=lambda v: f"{v:.3g}"))


def main() -> None:
    parser = config_arg(__doc__)
    parser.add_argument("--analysis", nargs="+", help="subset of qpadm.analyses (default: all)")
    parser.add_argument("--target", nargs="+", help="subset of qpadm.targets (default: all)")
    parser.add_argument("--jobs", type=int, help="parallel qpAdm runs (default: qpadm.jobs)")
    parser.add_argument("--force", action="store_true", help="rerun models that already have a result")
    args = parser.parse_args()
    cfg = load_config(args.config)
    q = cfg["qpadm"]
    data = cfg["out_dir"] / "aadr" / "merged"
    if not Path(f"{data}.geno").exists():
        raise SystemExit(f"{data}.geno not found: run `python scripts/ancient/aadr_prepare.py build` first")

    analyses = {k: v for k, v in q["analyses"].items() if not args.analysis or k in args.analysis}
    def targets_of(a: dict) -> list:  # an analysis can limit its targets; --target narrows further
        ts = a.get("targets", q["targets"])
        return [t for t in ts if t in args.target] if args.target else ts

    tasks = [(name, target, sources) for name, a in analyses.items() for target in targets_of(a)
             for sources in models(a)]
    all_targets = list(dict.fromkeys(t for _, t, _ in tasks))
    print(f"{len(tasks)} qpAdm models ({', '.join(analyses)} × {', '.join(all_targets)})")

    runs = cfg["out_dir"] / "qpadm" / "runs"
    # Analyses with `shared_fstats: true` (default for rotating ones) compute f-statistics once per target
    fstats = {}
    for name, target in dict.fromkeys((n, t) for n, t, _ in tasks):
        a = analyses[name]
        if a.get("shared_fstats", a.get("rotate", False)):
            # The first group is qpfstats' base: it must be diploid (inbreed: NO), so outgroups go first
            pops = list(dict.fromkeys([*a["right"], *a["sources"], target]))
            fstats[name, target] = shared_fstats(data, runs, pops, args.force)
    rows = [None] * len(tasks)  # filled as runs finish, kept in task order
    with ThreadPoolExecutor(max_workers=args.jobs or q.get("jobs", 4)) as pool:
        futures = {
            pool.submit(run_model, data, runs / name / target / "+".join(sources),
                        target, sources, right_for(analyses[name], sources), args.force,
                        fstats.get((name, target))): i
            for i, (name, target, sources) in enumerate(tasks)
        }
        for done, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            name, target, sources = tasks[i]
            res = fut.result()
            fits = (res["status"] == "ok" and res["p"] >= q["p_min"]
                    and all(0 <= w <= 1 for w in res["weights"]))
            rows[i] = {"analysis": name, "target": target, "n_sources": len(sources),
                       "sources": " + ".join(sources), **res, "fits": fits}
            print(f"[{done}/{len(tasks)}] {name} {target} {'+'.join(sources)}: "
                  f"{'p=%.3g' % res['p'] if res['status'] == 'ok' else res['status']}{'  FITS' if fits else ''}",
                  flush=True)

    df = pd.DataFrame(rows)
    out = cfg["out_dir"] / "qpadm" / "qpadm_results.csv"
    table = df.assign(weights=df["weights"].map(lambda w: " ".join(f"{x:.4f}" for x in w)),
                      se=df["se"].map(lambda s: " ".join(f"{x:.4f}" for x in s)))
    if out.exists():  # a subset run (--analysis/--target) keeps the other results in the file
        old = pd.read_csv(out)
        done = set(zip(table["analysis"], table["target"]))
        old = old[[(a, t) not in done for a, t in zip(old["analysis"], old["target"])]]
        old = old[old["analysis"].isin(q["analyses"])]  # drop analyses removed from the config
        table = pd.concat([old, table], ignore_index=True)
    table.to_csv(out, index=False)
    report(df, q["p_min"])
    print(f"\nAll models: {out}")


if __name__ == "__main__":
    main()
