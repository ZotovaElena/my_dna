"""Which variants do you share with ancient populations, and are they informative about ancestry?

Most alleles you share with an ancient genome are simply common everywhere, so the raw count of shared
alleles says nothing. Informative alleles are those that are rare in present-day Europeans but common in an
ancient group: "diagnostic" alleles. For each ancient (or Siberian) source G:

  1. Diagnostic set D_G: HO SNPs where the allele is at <= ref_max in a pooled present-day European reference
     and >= src_min in G (at least src_n allele copies observed in G).
  2. Sharing rate: the share of D_G alleles a genome carries, one random read per SNP for you (me_cram) and for
     control individuals drawn from the AADR the same way (their own alleles are removed from the reference
     and the source counts first). Expected rate if you were an average present-day European = mean reference
     frequency over D_G. Block-jackknife SE over 5 cM blocks.
  3. Variant list: diagnostic alleles you carry with read support (CRAM read shows the allele) and, where the
     imputed CSV has the SNP, the CSV genotype too. Each is annotated with the gene (span estimated from ClinVar
     records of that gene) and any ClinVar record at that exact position.

Step 2 is the test of informativeness: a source matters only if you carry its diagnostic alleles more often
than present-day Europeans do. A single shared allele is weak evidence: at 1.9x one read per SNP can be a
sequencing or mapping error, and a rare-in-Europe allele can be common elsewhere.

The Human Origins SNPs were chosen as common variants for population genetics; known trait variants (LCT,
HERC2, SLC24A5, SLC45A2, ...) are not on this panel, so ancient frequencies for them need the 1240k AADR.

Output in out_dir/shared_alleles/: rates.csv (per source and genome), variants.csv (your carried diagnostic
alleles with annotation), summary.json.
"""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import config_arg, load_config

sys.path.insert(0, str(Path(__file__).parent))
prep = importlib.import_module("12_aadr_prepare")
fast = importlib.import_module("13b_qpadm_fast")
farmer = importlib.import_module("21_farmer_f4")

DEFAULTS = {
    "reference": ["Russian", "Ukrainian", "Belarusian", "Czech", "Lithuanian", "Hungarian", "Norwegian", "Estonian",
                  "Finnish", "Basque", "Sardinian"],
    "sources": ["WHG", "EHG", "Anatolia_N", "Yamnaya", "CordedWare", "Iran_N", "BOO", "Nganasan", "Han"],
    "controls": {"Russian": 10, "Ukrainian": 10, "Sardinian": 5, "Finnish": 5, "Nganasan": 5},
    "ref_max": 0.05, "src_min": 0.5, "src_n": 4,
}


def clinvar_annotation(path: Path, snp: pd.DataFrame) -> pd.DataFrame:
    """Gene spans (min-max position of each gene's ClinVar records) and exact-position ClinVar records."""
    want = set(zip(snp["chrom"], snp["pos"].astype(int)))
    span: dict[tuple[str, str], list[int]] = {}
    exact = {}
    proc = subprocess.Popen(["bcftools", "query", "-f", r"%CHROM\t%POS\t%INFO/GENEINFO\t%INFO/CLNSIG\t%INFO/MC\t%INFO/CLNDN\n",
                             str(path)], stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:                    # streamed: ClinVar has millions of records
        c, p, gene, sig, mc, dn = line.rstrip("\n").split("\t")
        p = int(p)
        gene = gene.split("|")[0].split(":")[0]
        if gene != ".":
            s = span.get((c, gene))
            if s is None:
                span[(c, gene)] = [p, p]
            else:
                s[0], s[1] = min(s[0], p), max(s[1], p)
        if (c, p) in want and (c, p) not in exact:
            exact[(c, p)] = (sig, mc, dn)
    proc.wait()
    g = pd.DataFrame([(c, gn, lo, hi) for (c, gn), (lo, hi) in span.items()], columns=["chrom", "gene", "min", "max"])
    g = g[(g["max"] - g["min"]) < 2_500_000]                       # drop odd multi-gene records
    cv = pd.DataFrame([(c, p, *v) for (c, p), v in exact.items()], columns=["chrom", "pos", "sig", "mc", "dn"])
    if cv.empty:
        cv = pd.DataFrame(columns=["chrom", "pos", "sig", "mc", "dn"])
    out = pd.DataFrame(index=snp.index)
    out["gene"] = ""
    for c, d in snp.groupby("chrom"):
        gg = g[g["chrom"] == c]
        if gg.empty:
            continue
        lo, hi, name = gg["min"].to_numpy(), gg["max"].to_numpy(), gg["gene"].to_numpy()
        for i, p in zip(d.index, d["pos"].to_numpy()):
            hit = name[(lo <= p) & (p <= hi)]
            out.at[i, "gene"] = ",".join(sorted(set(hit))[:3])
    exact = cv.drop_duplicates(["chrom", "pos"]).set_index(["chrom", "pos"])
    key = pd.MultiIndex.from_arrays([snp["chrom"], snp["pos"]])
    hit = exact.reindex(key)
    out["clinvar"] = np.where(hit["sig"].notna().to_numpy(), hit["sig"].fillna("").to_numpy(), "")
    out["consequence"] = hit["mc"].fillna("").str.split("|").str[-1].str.replace(".", "", regex=False).to_numpy()
    out["condition"] = hit["dn"].fillna("").str.replace("_", " ").str.split("|").str[0].to_numpy()
    return out


def main() -> None:
    args = config_arg(__doc__).parse_args()
    cfg = load_config(args.config)
    o = {**DEFAULTS, **cfg.get("shared_alleles", {})}
    out = cfg["out_dir"] / "shared_alleles"
    out.mkdir(parents=True, exist_ok=True)
    data = cfg["out_dir"] / "aadr" / "merged"
    counts = fast.load_counts(data, cfg["out_dir"] / "qpadm" / "fast" / "counts.npz", False)
    gi = {str(g): i for i, g in enumerate(counts["groups"])}
    a, n, starts = counts["a"].astype(np.float64), counts["n"].astype(np.float64), counts["starts"]
    hap = set(str(g) for g in counts["haploid_groups"])
    snp = pd.read_csv(f"{data}.snp", sep=r"\s+", header=None, names=["id", "chrom", "gpos", "pos", "a1", "a2"],
                      dtype={"chrom": str})
    L = len(snp)
    missing = [g for g in o["reference"] + o["sources"] if g not in gi]
    if missing:
        raise SystemExit(f"Not in {data}.ind: {missing}")

    # Control individuals: one random read per SNP; remove them from their group's counts
    ind = pd.read_csv(f"{data}.ind", sep=r"\s+", header=None, names=["id", "sex", "group"])
    rng = np.random.default_rng(17)
    picks = []
    for pop, k in o["controls"].items():
        pool = np.flatnonzero(ind["group"].to_numpy() == pop)
        picks += [(int(i), pop) for i in rng.choice(pool, size=min(k, len(pool)), replace=False)]
    tests = []
    me = np.frombuffer(b"".join(l.strip() for l in open(f"{data.parent}/me_cram.geno", "rb")), np.uint8) - ord("0")
    tests.append(("you", "you", np.where(me == 9, np.nan, me / 2.0)))
    if picks:
        sel = np.array([i for i, _ in picks])
        G = np.concatenate(list(prep.aadr_genotypes(Path(f"{data}.geno"), len(ind), L, sel, chunk=50_000)))
        for j, (i, pop) in enumerate(picks):
            g = G[:, j].astype(np.float64)
            ok = g != 9
            k = gi[pop]
            ploidy = 1 if pop in hap else 2
            a[k] -= np.where(ok, g / 2 * ploidy, 0)
            n[k] -= ok * ploidy
            h = np.where(g == 1, rng.integers(0, 2, L), g / 2)
            tests.append((ind.at[i, "id"], pop, np.where(ok, h, np.nan)))
        del G
    csv = None
    if "me_csv" in gi:
        k = gi["me_csv"]
        csv = np.where(n[k] > 0, a[k] / np.maximum(n[k], 1), np.nan)   # 0, 0.5, 1 copies of allele 1 / 2

    def pooled(groups):
        A = sum(a[gi[g]] for g in groups)
        N = sum(n[gi[g]] for g in groups)
        return np.where(N > 0, A / np.maximum(N, 1), np.nan), N

    ref, ref_n = pooled(o["reference"])
    flip = ref > 0.5                                    # count the allele that is rarer in the reference
    minor = lambda x: np.where(flip, 1 - x, x)
    ref_m = minor(ref)

    rows, sets = [], {}
    for src in o["sources"]:
        k = gi[src]
        sf = minor(np.where(n[k] > 0, a[k] / np.maximum(n[k], 1), np.nan))
        D = (ref_n >= 100) & (ref_m <= o["ref_max"]) & (n[k] >= o["src_n"]) & (sf >= o["src_min"])
        sets[src] = (D, sf)
        exp_S, exp_N = farmer.block_sums(ref_m, D, starts)
        expected, _ = farmer.jackknife(exp_S, exp_N)
        for name, pop, x in tests:
            xm = minor(x)
            ok = D & np.isfinite(xm)
            S, N = farmer.block_sums(np.where(ok, xm, 0), ok, starts)
            rate, se = farmer.jackknife(S, N)
            rows.append({"source": src, "test": name, "pop": pop, "rate": rate, "se": se, "snps": int(N.sum()),
                         "diagnostic": int(D.sum()), "expected": expected, "src_freq": float(np.nanmean(sf[D]))})
    rates = pd.DataFrame(rows)
    ctrl = rates[rates["test"] != "you"].groupby(["source", "pop"])["rate"].mean().unstack()
    you = rates[rates["test"] == "you"].set_index("source")
    rates.to_csv(out / "rates.csv", index=False)

    # Variant list: diagnostic alleles you carry with read support
    print("Annotating with ClinVar...", flush=True)
    xm_me = minor(np.where(me == 9, np.nan, me / 2.0))
    csv_m = None if csv is None else minor(csv)
    carried = np.zeros(L, bool)
    for D, _ in sets.values():
        carried |= D & (xm_me == 1)
    idx = np.flatnonzero(carried)
    ann = clinvar_annotation(cfg["ref_dir"] / "clinvar.vcf.gz", snp.iloc[idx])
    vrows = []
    for i in idx:
        alt_is_a1 = not flip[i]
        allele = snp.at[i, "a1"] if alt_is_a1 else snp.at[i, "a2"]
        srcs = {s: sets[s][1][i] for s in o["sources"] if sets[s][0][i]}
        c = np.nan if csv_m is None or not np.isfinite(csv_m[i]) else float(csv_m[i] * 2)
        vrows.append({"rsid": snp.at[i, "id"], "chrom": snp.at[i, "chrom"], "pos": int(snp.at[i, "pos"]),
                      "allele": allele, "ref_freq": float(ref_m[i]),
                      "sources": " ".join(srcs), "best_source": max(srcs, key=srcs.get),
                      "best_freq": float(max(srcs.values())),
                      **{f"f_{s}": float(sets[s][1][i]) for s in o["sources"]},
                      "csv_copies": c, "gene": ann.at[i, "gene"], "clinvar": ann.at[i, "clinvar"],
                      "consequence": ann.at[i, "consequence"], "condition": ann.at[i, "condition"]})
    var = pd.DataFrame(vrows).sort_values(["best_freq", "ref_freq"], ascending=[False, True])
    var.to_csv(out / "variants.csv", index=False)

    summ = {"settings": {k: o[k] for k in ("reference", "ref_max", "src_min", "src_n")},
            "you_reads": int(np.isfinite(xm_me).sum()), "carried": len(var),
            "carried_csv_confirmed": int((var["csv_copies"] >= 1).sum()),
            "in_genes": int((var["gene"] != "").sum()), "clinvar_exact": int((var["clinvar"] != "").sum())}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))

    pd.set_option("display.width", 220)
    print("\nShare of diagnostic alleles carried (one read per SNP); expected = average present-day European")
    t = you[["diagnostic", "snps", "expected", "rate", "se"]].join(ctrl)
    print(t.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nYour carried diagnostic alleles: {len(var)} ({summ['carried_csv_confirmed']} also in the imputed CSV), "
          f"{summ['in_genes']} inside a ClinVar gene span, {summ['clinvar_exact']} with a ClinVar record")
    print(var.groupby("best_source").size().to_string())
    print(var[var["clinvar"] != ""][["rsid", "best_source", "gene", "clinvar", "consequence"]].head(20).to_string(index=False))
    print(f"\n{out}")


if __name__ == "__main__":
    main()
