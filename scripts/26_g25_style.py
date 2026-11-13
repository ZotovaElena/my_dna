"""G25-style coordinates: a 25-dimensional scaled PCA of present-day people worldwide, with you and every ancient
genome projected onto it, plus Vahaduo-style distances and mixture fits.

Global25 (Eurogenes) coordinates are scaled principal components: each PC multiplied by the square root of its
eigenvalue, so the Euclidean distance between two coordinate rows approximates genetic distance. The real G25
axes come from a private SNP weighting, so these numbers are NOT compatible with Global25 datasheets. They live
in their own space, built here the same way:

  1. Axes: present-day AADR genomes (date 0 BP, outlier/relative labels removed), at most axes_per_pop per
     population so large samples (Spanish, Han) don't dominate. SNPs called in >= 95% of them, MAF >= 1%.
  2. Scaled coordinates: PC scores = U * sqrt(lambda), divided by sqrt(#SNPs) so values look like G25's.
  3. Projection (least squares on each genome's own SNPs, like smartpca lsqproject): you (CRAM, one read per
     SNP; MyHeritage CSV), every other present-day genome (out of sample), and every ancient genome in the
     outgroup-f3 table. Population rows are averages of out-of-sample genomes, so you (also out of sample) are
     compared like with like: in-sample PCA scores are inflated on the later axes.
  4. Controls: present-day Russians left out of the axes, cut to your CRAM's SNPs and one random allele per SNP,
     then projected like you. Their distance to the Russian average is what a "Russian with your data" scores.
  5. Distances to every population and individual; mixture fits (non-negative weights summing to 1 minimising
     the distance, the Vahaduo method) for preset source sets and for open sets where the fit picks the sources.

Output in out_dir/g25/: you.txt, modern_populations.txt, ancient_populations.txt, ancient_individuals.txt
(G25 text format: name,PC1,...,PC25, ready to paste into Vahaduo's Target / Source boxes), distances.csv,
fits.csv, controls.csv, eigen.csv, summary.json.
"""

import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import config_arg, load_config

sys.path.insert(0, str(Path(__file__).parent))
prep = importlib.import_module("12_aadr_prepare")
K = 25
CHUNK = 40_000
SKIP = re.compile(r"(-o|_o$|_o_|Ignore|outlier|Relative|relative|_lc|contam|_dup|Chimp|Gorilla|Href|Altai|Vindija|"
                  r"Denisova|Chagyrskaya|Neanderthal|Archaic)")
DEFAULTS = {
    "axes_per_pop": 4,
    "min_pop": 2,
    "controls": {"Russian": 10},
    "models": {
        "Deep: hunter-gatherers, farmers, steppe": ["WHG", "EHG", "Anatolia_N", "Yamnaya", "Nganasan"],
        "Deep without steppe": ["WHG", "EHG", "Anatolia_N", "Iran_N", "Nganasan"],
        "Proximal: early medieval": ["EarlySlav", "EHG", "Nganasan"],
        "Proximal: Bronze Age": ["CordedWare", "Anatolia_N", "Nganasan"],
    },
    "open_modern_region": ["Russian", "Ukrainian", "Belarusian", "Polish", "Czech", "Lithuanian", "Latvian", "Estonian",
                           "Finnish", "Karelian", "Veps", "Mordovian", "Udmurt", "Komi_Zyrian", "Mari", "Chuvash",
                           "Tatar_Kazan", "Tatar_Mishar", "Bashkir", "Hungarian", "Croatian", "Bulgarian", "Romanian",
                           "German", "Norwegian", "Swedish", "Nganasan", "Selkup", "Khanty", "Mansi", "Yakut"],
}


def nnls(A: np.ndarray, b: np.ndarray, iters: int = 500) -> np.ndarray:
    """Lawson-Hanson non-negative least squares: min ||A x - b|| subject to x >= 0."""
    m, n = A.shape
    x = np.zeros(n)
    P = np.zeros(n, bool)
    w = A.T @ (b - A @ x)
    for _ in range(iters):
        if P.all() or (w[~P] <= 1e-12).all():
            break
        j = np.flatnonzero(~P)[np.argmax(w[~P])]
        P[j] = True
        while True:
            z = np.zeros(n)
            z[P] = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
            if (z[P] > 0).all():
                x = z
                break
            neg = P & (z <= 0)
            alpha = np.min(x[neg] / (x[neg] - z[neg]))
            x = x + alpha * (z - x)
            P &= x > 1e-12
            x[~P] = 0
        w = A.T @ (b - A @ x)
    return x


def mix(target: np.ndarray, sources: pd.DataFrame) -> tuple[pd.Series, float]:
    """Weights >= 0 summing to 1 that bring the weighted source average closest to target (Vahaduo)."""
    S = sources.to_numpy().T                       # K x n
    big = 1e3 * max(1.0, np.abs(S).max())          # sum-to-one row, weighted heavily
    A = np.vstack([S, np.full(S.shape[1], big)])
    b = np.r_[target, big]
    w = nnls(A, b)
    w = w / w.sum()
    d = float(np.linalg.norm(S @ w - target))
    return pd.Series(w, index=sources.index), d


def g25_text(df: pd.DataFrame) -> str:
    return "\n".join(f"{name}," + ",".join(f"{v:.6f}" for v in row) for name, row in
                     zip(df.index, df.to_numpy())) + "\n"


def main() -> None:
    args = config_arg(__doc__).parse_args()
    cfg = load_config(args.config)
    o = {**DEFAULTS, **cfg.get("g25", {})}
    out = cfg["out_dir"] / "g25"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(25)
    f = prep.aadr_files(cfg)
    ind = prep.read_ind(f["ind"])
    snp = pd.read_csv(f["snp"], sep=r"\s+", header=None, usecols=[0], names=["id"])["id"]
    aadr = cfg["out_dir"] / "aadr"
    merged_ids = pd.read_csv(aadr / "merged.snp", sep=r"\s+", header=None, usecols=[0], names=["id"])["id"]
    full_i = pd.Series(np.arange(len(snp)), index=snp).reindex(merged_ids).to_numpy().astype(np.int64)
    L = len(full_i)
    fmt, nind, nsnp = prep.geno_format(f["geno"])
    rlen = max(48, -(-nsnp // 4))
    head = prep.header_bytes(f["geno"], rlen, nind)
    fh = open(f["geno"], "rb")

    def person(i: int) -> np.ndarray:
        fh.seek(head + rlen * int(i))
        return prep.unpack(np.frombuffer(fh.read(rlen), np.uint8), full_i)

    # Present-day genomes: date 0 in the annotation
    anno = pd.read_csv(f["anno"], sep="\t", dtype=str, low_memory=False)
    dcol = next(c for c in anno.columns if c.lstrip('"').lower().startswith("date mean in bp"))
    date = pd.to_numeric(anno.set_index(anno.columns[0])[dcol], errors="coerce")
    ind["bp"] = ind["id"].map(date)
    grp = ind["group"].to_numpy()
    modern = (ind["bp"] == 0) & ~ind["group"].str.contains(SKIP)
    sizes = ind[modern].groupby("group").size()
    pops = sizes[sizes >= o["min_pop"]].index
    axes, oos = [], []
    for p in pops:
        pool = rng.permutation(np.flatnonzero(modern.to_numpy() & (grp == p)))
        axes += list(pool[:o["axes_per_pop"]])
        oos += list(pool[o["axes_per_pop"]:])
    axes = np.array(sorted(axes))
    print(f"Axes: {len(axes):,} present-day genomes from {len(pops)} populations; {len(oos):,} more projected", flush=True)

    G = np.empty((len(axes), L), np.uint8)
    for r, i in enumerate(axes):
        G[r] = person(i)
    ok = G != 3
    p = np.where(ok, G, 0).sum(0) / np.maximum(2 * ok.sum(0), 1)
    keep = (ok.mean(0) >= 0.95) & (np.minimum(p, 1 - p) >= 0.01)
    G, p, cols = G[:, keep], p[keep], np.flatnonzero(keep)
    sd = np.sqrt(p * (1 - p))
    M = len(cols)
    print(f"  {M:,} SNPs (called in >= 95%, MAF >= 1%)", flush=True)

    def standard(g, sl):
        x = (g.astype(np.float32) - 2 * p[sl]) / sd[sl]
        return np.where(g == 3, 0, x).astype(np.float32)

    grm = np.zeros((len(axes), len(axes)))
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
    del G
    scale = 1 / np.sqrt(M)
    pd.DataFrame({"pc": [f"PC{k + 1}" for k in range(K)], "eigenvalue": lam[:K] / M,
                  "variance_share": lam[:K] / lam[lam > 0].sum()}).to_csv(out / "eigen.csv", index=False)
    WtW = (W.T.astype(np.float64) @ W.astype(np.float64))

    def project(g: np.ndarray) -> tuple[np.ndarray, int]:
        """g: copies of allele 1 at the kept SNPs, 3 = missing. Least squares on the observed SNPs."""
        obs = g != 3
        x = np.where(obs, (g.astype(np.float64) - 2 * p) / sd, 0)
        miss = ~obs
        if miss.mean() < 0.3:
            Wm = W[miss]
            A = WtW - (Wm.T @ Wm).astype(np.float64)
        else:
            Wo = W[obs]
            A = (Wo.T @ Wo).astype(np.float64)
        return np.linalg.solve(A, W.T @ x.astype(np.float32)) * scale, int(obs.sum())

    pcs = [f"PC{k + 1}" for k in range(K)]
    # Present-day: out-of-sample genomes; populations with none left out of sample use their axis genomes
    rows = []
    for i in oos:
        c, n = project(person(i)[cols])
        rows.append({"id": ind.at[i, "id"], "pop": grp[i], "in_axes": False, "snps": n, **dict(zip(pcs, c))})
    for r, i in enumerate(axes):
        rows.append({"id": ind.at[i, "id"], "pop": grp[i], "in_axes": True, "snps": M,
                     **dict(zip(pcs, U[r, :K] * s * scale))})
    mod = pd.DataFrame(rows)
    has_oos = mod[~mod["in_axes"]].groupby("pop").size()
    use = mod[~mod["in_axes"] | ~mod["pop"].isin(has_oos[has_oos >= 2].index)]
    modern_pops = use.groupby("pop")[pcs].mean()
    modern_n = use.groupby("pop").size()
    print(f"  {len(modern_pops)} present-day populations", flush=True)

    # You
    me = np.frombuffer(b"".join(l.strip() for l in open(aadr / "me_cram.geno", "rb")), np.uint8) - ord("0")
    me_g = np.where(me[cols] == 9, 3, me[cols]).astype(np.uint8)
    you = {}
    you["you_CRAM"], n_cram = project(me_g)
    m_ind = pd.read_csv(aadr / "merged.ind", sep=r"\s+", header=None, names=["id", "sex", "group"])
    if (m_ind["group"] == "me_csv").any():
        j = np.flatnonzero(m_ind["group"].to_numpy() == "me_csv")
        g = np.concatenate(list(prep.aadr_genotypes(aadr / "merged.geno", len(m_ind), L, j, chunk=50_000)))[:, 0]
        you["you_MyHeritage_CSV"], n_csv = project(np.where(g[cols] == 9, 3, g[cols]).astype(np.uint8))
    you = pd.DataFrame(you, index=pcs).T

    # Controls: out-of-sample individuals, masked to your CRAM SNPs, one random allele per SNP
    ctrl = []
    for pop, k in o["controls"].items():
        pool = mod[(mod["pop"] == pop) & ~mod["in_axes"]]["id"].to_numpy()
        for iid in rng.choice(pool, size=min(k, len(pool)), replace=False):
            i = int(np.flatnonzero(ind["id"].to_numpy() == iid)[0])
            g = person(i)[cols]
            g = np.where(me_g == 3, 3, g)
            g = np.where(g == 1, np.where(rng.random(M) < 0.5, 0, 2), g).astype(np.uint8)
            c, n = project(g)
            others = use[(use["pop"] == pop) & (use["id"] != iid)][pcs].mean().to_numpy()
            full_c = mod.loc[mod["id"] == iid, pcs].to_numpy()[0]
            ctrl.append({"id": iid, "pop": pop, "snps": n, "dist_to_pop": float(np.linalg.norm(c - others)),
                         "dist_full_data_to_pop": float(np.linalg.norm(full_c - others)),
                         "shift_from_data_loss": float(np.linalg.norm(c - full_c)), **dict(zip(pcs, c))})
    ctrl = pd.DataFrame(ctrl)
    ctrl.to_csv(out / "controls.csv", index=False)

    # Ancient individuals: those in the outgroup-f3 table (>= 30k SNPs with you)
    f3p = cfg["out_dir"] / "f3" / "outgroup_f3_ancient.csv"
    f3 = pd.read_csv(f3p) if f3p.exists() else pd.DataFrame(columns=["id", "group"])
    pos_of = pd.Series(np.arange(len(ind)), index=ind["id"])
    arows = []
    print(f"Projecting {len(f3):,} ancient genomes...", flush=True)
    for k, r in enumerate(f3.itertuples()):
        if r.id not in pos_of.index:
            continue
        g = person(pos_of[r.id])[cols]
        if (g != 3).sum() < 15_000:
            continue
        g = np.where(g == 1, np.where(rng.random(M) < 0.5, 0, 2), g).astype(np.uint8)
        c, n = project(g)
        arows.append({"id": r.id, "iid": r.iid, "group": r.group, "date_bp": r.date_bp, "country": r.country,
                      "snps": n, **dict(zip(pcs, c))})
        if (k + 1) % 3000 == 0:
            print(f"  {k + 1:,}", flush=True)
    anc = pd.DataFrame(arows)
    anc_pops = anc.groupby("group")[pcs].mean()
    anc_n = anc.groupby("group").size()
    anc_date = anc.groupby("group")["date_bp"].median()

    # Model groups: config short names (aadr.groups) -> pooled AADR labels, ancient or present-day
    short = cfg.get("aadr", {}).get("groups", {})

    def source(name: str) -> np.ndarray | None:
        labels = short.get(name, [name])
        parts = []
        for lab in labels:
            if lab in anc_pops.index:
                parts.append((anc_pops.loc[lab].to_numpy(), anc_n[lab]))
            elif lab in modern_pops.index:
                parts.append((modern_pops.loc[lab].to_numpy(), modern_n[lab]))
        if not parts:
            return None
        return sum(v * n for v, n in parts) / sum(n for _, n in parts)

    # Distances
    drows = []
    for t, tv in you.iterrows():
        tv = tv.to_numpy()
        for kind, tab, nn in [("modern population", modern_pops, modern_n), ("ancient population", anc_pops, anc_n)]:
            d = np.linalg.norm(tab.to_numpy() - tv, axis=1)
            for name, dist in zip(tab.index, d):
                drows.append({"target": t, "kind": kind, "name": name, "n": int(nn[name]), "distance": float(dist),
                              "date_bp": float(anc_date.get(name, 0)) if kind.startswith("ancient") else 0.0})
        d = np.linalg.norm(anc[pcs].to_numpy() - tv, axis=1)
        for r, dist in zip(anc.itertuples(), d):
            drows.append({"target": t, "kind": "ancient individual", "name": r.iid, "group": r.group, "n": 1,
                          "distance": float(dist), "date_bp": float(r.date_bp), "country": r.country})
    dist = pd.DataFrame(drows).sort_values(["target", "kind", "distance"])
    dist["rank"] = dist.groupby(["target", "kind"]).cumcount() + 1
    dist.to_csv(out / "distances.csv", index=False)

    # Mixture fits
    frows = []

    def record(t, model, kind, w, d):
        w = w[w > 0.0005].sort_values(ascending=False)
        frows.append({"target": t, "model": model, "kind": kind, "distance": d,
                      "weights": "; ".join(f"{k} {v:.3f}" for k, v in w.items())})

    for t, tv in you.iterrows():
        tv = tv.to_numpy()
        for model, names in o["models"].items():
            S = {nm: source(nm) for nm in names}
            S = pd.DataFrame({k: v for k, v in S.items() if v is not None}, index=pcs).T
            if len(S) < 2:
                continue
            w, d = mix(tv, S)
            record(t, model, "preset", w, d)
        reg = [x for x in o["open_modern_region"] if x in modern_pops.index]
        w, d = mix(tv, modern_pops.loc[reg])
        record(t, "Open: Eastern Europe, Volga-Urals and Siberia today", "open", w, d)
        w, d = mix(tv, modern_pops[modern_n >= 3])
        record(t, "Open: all present-day populations", "open", w, d)
        big = anc_n[anc_n >= 3].index
        w, d = mix(tv, anc_pops.loc[big])
        record(t, "Open: all ancient populations (>= 3 people)", "open", w, d)
        old = [g for g in big if anc_date.get(g, 0) >= 3500]
        w, d = mix(tv, anc_pops.loc[old])
        record(t, "Open: ancient populations older than 3,500 years", "open", w, d)
        med = [g for g in big if anc_date.get(g, 1e9) <= 1500]
        w, d = mix(tv, anc_pops.loc[med])
        record(t, "Open: medieval and later populations", "open", w, d)
    # The same fits for the controls, so a "good" distance has a reference
    for r in ctrl.itertuples():
        tv = np.array([getattr(r, c) for c in pcs])
        for model in ["Deep: hunter-gatherers, farmers, steppe", "Proximal: early medieval"]:
            names = o["models"].get(model, [])
            S = pd.DataFrame({nm: source(nm) for nm in names if source(nm) is not None}, index=pcs).T
            if len(S) >= 2:
                w, d = mix(tv, S)
                record(f"control:{r.id} ({r.pop})", model, "preset", w, d)
    fits = pd.DataFrame(frows)
    fits.to_csv(out / "fits.csv", index=False)

    # G25-format sheets
    (out / "you.txt").write_text(g25_text(you))
    (out / "modern_populations.txt").write_text(g25_text(modern_pops))
    (out / "ancient_populations.txt").write_text(g25_text(anc_pops))
    (out / "ancient_individuals.txt").write_text(g25_text(anc.set_index(anc["group"] + ":" + anc["iid"])[pcs]))
    eig = pd.read_csv(out / "eigen.csv")
    summ = {"axes_genomes": int(len(axes)), "axes_populations": int(len(pops)), "snps": int(M),
            "modern_populations": int(len(modern_pops)), "ancient_individuals": int(len(anc)),
            "ancient_populations": int(len(anc_pops)), "you_snps": {"CRAM": n_cram, "CSV": n_csv if len(you) > 1 else 0},
            "cram_vs_csv_distance": float(np.linalg.norm(you.iloc[0] - you.iloc[-1])) if len(you) > 1 else None,
            "variance_share_25": float(eig["variance_share"].sum())}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))

    pd.set_option("display.width", 220)
    print(f"\n{g25_text(you)}")
    print(json.dumps(summ, indent=2))
    for t in you.index:
        for kind in ["modern population", "ancient population", "ancient individual"]:
            d = dist[(dist["target"] == t) & (dist["kind"] == kind)].head(8)
            print(f"\n{t}, closest {kind}s:", ", ".join(f"{r.name} {r.distance:.4f}" for r in d.itertuples()))
    print("\nControls (Russians, your data quality): distance to the Russian average",
          f"{ctrl['dist_to_pop'].min():.4f}-{ctrl['dist_to_pop'].max():.4f} (full data "
          f"{ctrl['dist_full_data_to_pop'].min():.4f}-{ctrl['dist_full_data_to_pop'].max():.4f})")
    print(fits.to_string(index=False, max_colwidth=120))
    print(f"\n{out}")


if __name__ == "__main__":
    main()
