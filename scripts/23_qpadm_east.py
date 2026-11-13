"""Which population carries the East Eurasian (Nganasan-related) component? qpAdm model comparison.

Nganasan is a convenient modern proxy, not a historical source. Here every candidate in config
`east.candidates` (Uralic-speaking groups old and new, deep Siberian groups, medieval steppe/Volga peoples)
takes Nganasan's place:

  1. Single models: base + candidate (bases: EarlySlav, EarlySlav + EHG), against the proximal outgroups.
     Fit, p-value and the candidate's weight. Medieval groups that are themselves mostly European get a much
     larger weight than Nganasan, so compare fit and p, not weight.
  2. Head to head (rotating design, Harney et al. 2021): for every ordered pair, compete_base + A with B added
     to the outgroups. It fits only if B shares no extra drift with the target beyond what A provides.
     A candidate that survives every competitor is the best-supported source; if A survives B and B fails
     with A as outgroup, A is preferred over B.
     Only candidates that pass an outgroup check count as B: a European without East Eurasian ancestry
     (east.null_target, Czech) must still fit compete_base with B added to the outgroups. Medieval groups with
     much West Eurasian ancestry of their own (Mongols, Avars, ...) fail it: as outgroups they reject every
     model, even for Czechs, so their "wins" would be an artifact of the design.
  3. With --replicates R: steps 1-2 for me_cram are repeated on R pseudo-haploid redraws of your reads
     (outputs/qpadm/replicates/reads.npz from 20_qpadm_replicates.py), so verdicts near p = 0.01 are counted
     as shares of draws, not as one coin flip.

Uses the fast NumPy engine (13b_qpadm_fast.py). Output in out_dir/qpadm/east/: models.csv (every model, every
replicate), single.csv and compete.csv (summaries).
"""

import importlib
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from common import config_arg, load_config

sys.path.insert(0, str(Path(__file__).parent))
fast = importlib.import_module("13b_qpadm_fast")
reps_mod = importlib.import_module("20_qpadm_replicates")


def tasks_for(e: dict, targets: list[str]) -> list[tuple]:
    cands = [c for group in e["candidates"].values() for c in group]
    out = []
    for t in targets:
        for base in e["bases"]:
            label = " + ".join(base)
            out.append((f"single|{label}", t, tuple(base), e["right"]))           # no East Eurasian source
            out += [(f"single|{label}", t, tuple(base) + (c,), e["right"]) for c in cands]
        cb = tuple(e["compete_base"])
        out += [(f"compete|{b}", t, cb + (a,), e["right"] + [b]) for a, b in itertools.permutations(cands, 2)]
    return out


def run_tasks(tasks, freq, starts, index, jobs, rep, p_min) -> pd.DataFrame:
    with ProcessPoolExecutor(jobs, initializer=fast._init, initargs=(freq, starts, index)) as pool:
        res = list(pool.map(fast.evaluate, tasks, chunksize=4))
    rows = []
    for (kind, t, s, _), x in zip(tasks, res):
        mode, extra = kind.split("|")
        rows.append({"rep": rep, "mode": mode, "target": t, "base" if mode == "single" else "outgroup_added": extra,
                     "sources": " + ".join(s), "candidate": s[-1] if mode == "compete" or len(s) > len(extra.split(" + "))
                     else "(none)", "p": x["p"], "weights": " ".join(f"{w:.4f}" for w in x["weights"]),
                     "se": " ".join(f"{v:.4f}" for v in x["se"]), "status": x["status"],
                     "fits": bool(x["status"] == "ok" and x["p"] >= p_min and all(0 <= w <= 1 for w in x["weights"]))})
    return pd.DataFrame(rows)


def outgroup_check(e: dict, freq, starts, index, p_min: float, work: Path) -> pd.DataFrame:
    """Candidate B is a usable extra outgroup if null_target ~ compete_base still fits with B on the right."""
    fast._init(freq, starts, index)
    t, base = e.get("null_target", "Czech"), tuple(e["compete_base"])
    ref = fast.evaluate(("check", t, base, e["right"]))["p"]
    rows = [{"candidate": c, "null_target": t, "p": fast.evaluate(("check", t, base, e["right"] + [c]))["p"],
             "p_without": ref} for g in e["candidates"].values() for c in g]
    d = pd.DataFrame(rows)
    d["usable_outgroup"] = d["p"] >= p_min
    d.to_csv(work / "outgroup_check.csv", index=False)
    return d


def summarise(df: pd.DataFrame, e: dict, work: Path) -> None:
    cat = {c: k for k, group in e["candidates"].items() for c in group}
    cands = list(cat)
    chk = pd.read_csv(work / "outgroup_check.csv") if (work / "outgroup_check.csv").exists() else None
    usable = set(chk.loc[chk["usable_outgroup"], "candidate"]) if chk is not None else set(cands)
    R = df["rep"].nunique()
    # 1. single models: p and last weight over replicates (rep 0 = the original draw)
    s = df[df["mode"] == "single"].copy()
    s["w_last"] = s["weights"].map(lambda w: float(w.split()[-1]))
    s["se_last"] = s["se"].map(lambda w: float(w.split()[-1]))
    single = s.groupby(["target", "base", "candidate"], sort=False).agg(
        p=("p", lambda v: v.iloc[0]), p_median=("p", "median"), fit_share=("fits", "mean"),
        weight=("w_last", lambda v: v.iloc[0]), weight_se=("se_last", lambda v: v.iloc[0]),
        weight_sd=("w_last", "std"), reps=("rep", "nunique")).reset_index()
    single["category"] = single["candidate"].map(cat).fillna("")
    single.to_csv(work / "single.csv", index=False)
    # 2. head to head: share of draws in which A survives B as an extra outgroup
    c = df[df["mode"] == "compete"]
    wins = c.groupby(["target", "candidate", "outgroup_added"])["fits"].mean().rename("survives").reset_index()
    rows = []
    for t, d in wins.groupby("target", sort=False):
        m = d.pivot(index="candidate", columns="outgroup_added", values="survives").reindex(index=cands, columns=cands)
        for a in cands:
            others = [b for b in cands if b != a and b in usable]
            surv = m.loc[a, others]
            beats = [b for b in others if m.loc[a, b] >= 0.5 and (a not in usable or m.loc[b, a] < 0.5)]
            lost = [b for b in others if m.loc[a, b] < 0.5 and (a not in usable or m.loc[b, a] >= 0.5)]
            rows.append({"target": t, "candidate": a, "category": cat[a], "survives_all": bool((surv >= 0.5).all()),
                         "survived": int((surv >= 0.5).sum()), "of": len(others), "min_share": float(surv.min()),
                         "beats": " ".join(beats), "n_beats": len(beats), "loses_to": " ".join(lost)})
        m.to_csv(work / f"compete_matrix_{t}.csv")
    comp = pd.DataFrame(rows).sort_values(["target", "survived", "n_beats"], ascending=[True, False, False])
    comp.to_csv(work / "compete.csv", index=False)

    pd.set_option("display.width", 220)
    for t in dict.fromkeys(single["target"]):
        for base in dict.fromkeys(single["base"]):
            d = single[(single["target"] == t) & (single["base"] == base)].sort_values("p_median", ascending=False)
            print(f"\n== {t}: {base} + candidate ({R} draw{'s' if R > 1 else ''})")
            print(d[["category", "candidate", "p", "p_median", "fit_share", "weight", "weight_se"]].to_string(
                index=False, float_format=lambda v: f"{v:.3g}"))
        d = comp[comp["target"] == t]
        print(f"\n== {t}: head to head, {' + '.join(e['compete_base'])} + A with B as an extra outgroup "
              f"(B = the {len(usable)} candidates that pass the outgroup check)")
        print(d[["category", "candidate", "survived", "of", "n_beats", "loses_to"]].to_string(index=False))
    print(f"\n{work}")


def main() -> None:
    parser = config_arg(__doc__)
    parser.add_argument("--replicates", type=int, default=0, help="pseudo-haploid redraws of me_cram (default 0)")
    parser.add_argument("--jobs", type=int, default=min(5, os.cpu_count() or 1))
    parser.add_argument("--summary", action="store_true", help="only re-summarise models.csv")
    args = parser.parse_args()
    cfg = load_config(args.config)
    e, p_min = cfg["east"], cfg["qpadm"]["p_min"]
    work = cfg["out_dir"] / "qpadm" / "east"
    work.mkdir(parents=True, exist_ok=True)
    out = work / "models.csv"
    counts = fast.load_counts(cfg["out_dir"] / "aadr" / "merged", cfg["out_dir"] / "qpadm" / "fast" / "counts.npz",
                              False)
    groups = [str(g) for g in counts["groups"]]
    index = {g: i for i, g in enumerate(groups)}
    need = set(e["right"]) | {c for g in e["candidates"].values() for c in g} | set(e["targets"]) | \
        {s for b in e["bases"] for s in b}
    missing = sorted(need - set(groups))
    if missing:
        raise SystemExit(f"Not in the merged dataset: {missing}. Add them to aadr.groups and rerun "
                         "`python scripts/12_aadr_prepare.py build`.")
    with np.errstate(invalid="ignore", divide="ignore"):
        freq = np.where(counts["n"] > 0, counts["a"] / counts["n"].astype(np.float64), np.nan)

    chk = outgroup_check(e, freq, counts["starts"], index, p_min, work)
    print("Outgroup check (" + chk["null_target"].iloc[0] + f" ~ {' + '.join(e['compete_base'])}, p without = "
          f"{chk['p_without'].iloc[0]:.3g}); unusable as outgroups: "
          + (", ".join(chk.loc[~chk["usable_outgroup"], "candidate"]) or "none"), flush=True)
    if args.summary:
        summarise(pd.read_csv(out), e, work)
        return
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["rep"])
    parts = [done] if len(done) else []
    if 0 not in set(done["rep"]):
        tasks = tasks_for(e, e["targets"])
        t0 = time.time()
        print(f"{len(tasks)} models (all targets) on {args.jobs} cores...", flush=True)
        parts.append(run_tasks(tasks, freq, counts["starts"], index, args.jobs, 0, p_min))
        pd.concat(parts, ignore_index=True).to_csv(out, index=False)
        print(f"  done in {time.time() - t0:.0f} s", flush=True)
    if args.replicates:
        z = np.load(cfg["out_dir"] / "qpadm" / "replicates" / "reads.npz")
        tasks = tasks_for(e, ["me_cram"])
        original = freq[index["me_cram"]].copy()
        for r in range(1, args.replicates + 1):
            if r in set(done["rep"]):
                continue
            t0 = time.time()
            freq[index["me_cram"]] = reps_mod.draw(z["n1"], z["n2"], seed=r)
            parts.append(run_tasks(tasks, freq, counts["starts"], index, args.jobs, r, p_min))
            pd.concat(parts, ignore_index=True).to_csv(out, index=False)
            print(f"  replicate {r}/{args.replicates} ({time.time() - t0:.0f} s)", flush=True)
        freq[index["me_cram"]] = original
    summarise(pd.concat(parts, ignore_index=True), e, work)


if __name__ == "__main__":
    main()
