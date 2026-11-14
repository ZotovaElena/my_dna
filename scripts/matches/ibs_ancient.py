"""Half-identical segments between you and ancient genomes, the way GEDmatch's "archaic matches" chart
finds them, plus the same scan for 1000 Genomes / AADR moderns of other ancestries as a control.

A segment is a run of SNPs with no opposite homozygote (you AA, them BB), at least `ibs.min_cm` long on the
genetic map and with at least `ibs.min_snps` SNPs called in both. Missing SNPs are skipped, not counted as
mismatches, so an ancient genome with fewer SNPs has fewer chances to break a run. Most ancient genomes
are pseudo-haploid (one read per SNP, so always "homozygous"), which matters as well. If Han Chinese or
Yoruba people "match" Kostenki14 as much as you do, the chart measures the ancient genome's data, not
relatedness: real shared segments decay to nothing within a few dozen generations.

Inputs: AADR (aadr.prefix, read directly, so any individual can be listed), outputs/merged.pgen (you +
1000 Genomes) and the `ibs` config section. Outputs in out_dir/ibs/: summary.csv (one row per modern ×
ancient pair), segments.csv, and the genotype cache. Rerun with --force after changing the config lists.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py and the module folders
from common import AUTOSOMES, config_arg, load_config, load_panel

from ancient import aadr_prepare as prep


def read_snp(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=["id", "chrom", "gpos", "pos", "a1", "a2"],
                       dtype={"chrom": str})


def pick_ancients(cfg: dict, ind: pd.DataFrame, anno: pd.DataFrame | None) -> pd.DataFrame:
    """Rows of the AADR .ind for the configured groups and IDs, with date and group from the annotation."""
    ib, groups = cfg["ibs"], cfg["aadr"]["groups"]
    labels = {lab: short for short in ib.get("ancient_groups", []) for lab in groups.get(short, [])}
    sel = ind[ind["grp"].isin(labels) | ind["id"].isin(ib.get("ancient_ids", []))].copy()
    missing = set(ib.get("ancient_ids", [])) - set(sel["id"])
    if missing:
        print(f"Not in the AADR .ind (check the spelling): {', '.join(sorted(missing))}")
    sel["set"] = [labels.get(g, "named") if i not in ib.get("ancient_ids", []) else "named"
                  for i, g in zip(sel["id"], sel["grp"])]
    if anno is not None:
        sel = sel.merge(anno, on="id", how="left")
    return sel


def read_anno(cfg: dict) -> pd.DataFrame | None:
    files = sorted((cfg["ref_dir"] / Path(cfg["aadr"]["prefix"]).parent).glob("*.anno"))
    if not files:
        return None
    a = pd.read_csv(files[0], sep="\t", low_memory=False)
    c = a.columns
    return pd.DataFrame({"id": a[c[0]], "date_bp": pd.to_numeric(a[c[10]], errors="coerce"),
                         "locality": a[c[15]], "country": a[c[16]]})


def modern_1kg(cfg: dict, work: Path) -> tuple[list[str], list[str], pd.DataFrame, np.ndarray]:
    """You + sampled 1000G controls from outputs/merged.pgen: ids, populations, .bim and allele counts."""
    panel = load_panel(cfg)
    rng = np.random.default_rng(1)
    ids, pops = [cfg["sample_id"]], ["you"]
    for pop, n in cfg["ibs"].get("controls_1kg", {}).items():
        pool = panel.loc[panel["pop"] == pop, "sample"].to_numpy()
        for s in rng.choice(pool, size=min(n, len(pool)), replace=False):
            ids.append(str(s))
            pops.append(pop)
    keep = work / "modern.keep"
    keep.write_text("".join(f"{i}\n" for i in ids))
    stem = work / "modern"
    subprocess.run(["plink2", "--pfile", str(cfg["out_dir"] / "merged"), "--keep", str(keep),
                    "--snps-only", "just-acgt", "--max-alleles", "2", "--make-bed", "--out", str(stem)],
                   check=True, stdout=subprocess.DEVNULL)
    bim = pd.read_csv(f"{stem}.bim", sep="\t", header=None, names=["chrom", "id", "cm", "pos", "a1", "a2"],
                      dtype={"chrom": str})
    order = pd.read_csv(f"{stem}.fam", sep=r"\s+", header=None)[1].astype(str).tolist()
    n = len(order)
    raw = np.fromfile(f"{stem}.bed", np.uint8)[3:].reshape(len(bim), -1)
    codes = np.stack([(raw >> (2 * k)) & 3 for k in range(4)], axis=2).reshape(len(bim), -1)[:, :n]
    a1 = np.choose(codes, [2, 9, 1, 0]).astype(np.uint8)  # bed: 00 hom a1, 01 missing, 10 het, 11 hom a2
    pop_of = dict(zip(ids, pops))
    return order, [pop_of[i] for i in order], bim, a1


def load(cfg: dict, work: Path, force: bool) -> dict:
    cache = work / "genotypes.npz"
    if cache.exists() and not force:
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    f = prep.aadr_files(cfg)
    ind = prep.read_ind(f["ind"])
    ind.columns = ["id", "sex", "grp"]
    snp = read_snp(f["snp"])
    anc = pick_ancients(cfg, ind, read_anno(cfg))
    ctrl = cfg["ibs"].get("controls_aadr", {})
    labels = {lab: short for short in ctrl for lab in cfg["aadr"]["groups"].get(short, [short])}
    rng = np.random.default_rng(2)
    am = [i for lab, short in labels.items()
          for i in rng.permutation(ind.index[ind["grp"] == lab].to_numpy())[:ctrl[short]]]
    am_pop = [labels[ind.at[i, "grp"]] for i in am]

    # Shared SNPs: AADR autosomal SNPs that are in merged.pgen with the same two alleles
    order, pops, bim, g1k = modern_1kg(cfg, work)
    snp["i"] = np.arange(len(snp))
    m = snp[snp["chrom"].isin(AUTOSOMES)].merge(bim.reset_index(names="j"), on=["chrom", "pos"],
                                                 suffixes=("", "_b"))
    m = m.drop_duplicates(["chrom", "pos"], keep=False)
    same = (m["a1"] == m["a1_b"]) & (m["a2"] == m["a2_b"])
    swap = (m["a1"] == m["a2_b"]) & (m["a2"] == m["a1_b"])
    m = m[same | swap].assign(swap=swap[same | swap]).sort_values("i")
    print(f"Shared SNPs (AADR HO ∩ your merged 1000G file, same alleles): {len(m):,}")
    mod = g1k[m["j"].to_numpy()]
    sw = m["swap"].to_numpy()
    called = mod != 9
    mod[sw[:, None] & called] = 2 - mod[sw[:, None] & called]  # counts of the AADR first allele

    # One pass over the AADR genotype file for the selected individuals
    sel = np.concatenate([anc.index.to_numpy(), np.array(am, dtype=int)])
    keep_i = m["i"].to_numpy()
    out = np.empty((len(keep_i), len(sel)), np.uint8)
    pos = 0
    print(f"Reading {len(sel)} individuals from {f['geno'].name} (one pass, a few minutes)...", flush=True)
    for block in prep.aadr_genotypes(f["geno"], len(ind), len(snp), sel):
        lo, hi = pos, pos + len(block)
        a, b = np.searchsorted(keep_i, [lo, hi])
        out[a:b] = block[keep_i[a:b] - lo]
        pos = hi
    na = len(anc)
    data = {
        "chrom": m["chrom"].astype(int).to_numpy(np.int8), "cm": (m["gpos"].to_numpy() * 100).astype(np.float32),
        "anc": out[:, :na].T.copy(), "anc_id": anc["id"].to_numpy(), "anc_grp": anc["grp"].to_numpy(),
        "anc_set": anc["set"].to_numpy(), "anc_date": anc.get("date_bp", pd.Series(np.nan, anc.index)).to_numpy(float),
        "anc_place": (anc.get("locality", "").astype(str) + ", " + anc.get("country", "").astype(str)).to_numpy(),
        "mod": np.concatenate([mod.T, out[:, na:].T]), "mod_id": np.array(order + [ind.at[i, "id"] for i in am]),
        "mod_pop": np.array(pops + am_pop),
    }
    np.savez_compressed(cache, **data)
    return data


def segments(m: np.ndarray, a: np.ndarray, chrom: np.ndarray, cm: np.ndarray, min_cm: float, min_snps: int):
    """Half-identical runs between one diploid genotype vector and one other (0/1/2 copies, 9 = missing)."""
    idx = np.flatnonzero((m != 9) & (a != 9))
    if len(idx) == 0:
        return []
    mi, ai, ch = m[idx], a[idx], chrom[idx]
    opp = ((mi == 0) & (ai == 2)) | ((mi == 2) & (ai == 0))
    # A run starts after each opposite homozygote and at each chromosome start; the mismatch itself is excluded
    new = np.r_[True, ch[1:] != ch[:-1]] | np.r_[False, opp[:-1]]
    run = np.cumsum(new) - 1
    ok = ~opp
    r, c = run[ok], cm[idx][ok]
    if len(r) == 0:
        return []
    starts = np.r_[0, np.flatnonzero(r[1:] != r[:-1]) + 1]
    n = np.diff(np.r_[starts, len(r)])
    lo, hi = c[starts], c[np.r_[starts[1:], len(r)] - 1]
    keep = (n >= min_snps) & (hi - lo >= min_cm)
    chs = ch[ok][starts]
    return [(int(chs[k]), float(lo[k]), float(hi[k]), int(n[k])) for k in np.flatnonzero(keep)]


def main() -> None:
    parser = config_arg(__doc__)
    parser.add_argument("--force", action="store_true", help="reread genotypes (after changing the lists)")
    parser.add_argument("--min-cm", type=float, help="override ibs.min_cm")
    parser.add_argument("--min-snps", type=int, help="override ibs.min_snps")
    args = parser.parse_args()
    cfg = load_config(args.config)
    work = cfg["out_dir"] / "ibs"
    work.mkdir(parents=True, exist_ok=True)
    min_cm = args.min_cm or cfg["ibs"]["min_cm"]
    min_snps = args.min_snps or cfg["ibs"]["min_snps"]
    d = load(cfg, work, args.force)
    anc, mod, chrom, cm = d["anc"], d["mod"], d["chrom"], d["cm"]

    het = ((anc == 1).sum(1) / np.maximum((anc != 9).sum(1), 1))
    print(f"{len(d['mod_id'])} moderns × {len(d['anc_id'])} ancients, {anc.shape[1]:,} SNPs, "
          f"≥{min_cm} cM and ≥{min_snps} SNPs")
    rows, segs = [], []
    for mi in range(len(mod)):
        for ai in range(len(anc)):
            runs = segments(mod[mi], anc[ai], chrom, cm, 0, 1)  # every half-identical run, then two filters
            strict = [r for r in runs if r[2] - r[1] >= min_cm and r[3] >= min_snps]
            by_cm = [r for r in runs if r[2] - r[1] >= min_cm]  # cM only: what a sparse genome gets credited
            both = int(((mod[mi] != 9) & (anc[ai] != 9)).sum())
            rows.append({"modern": d["mod_id"][mi], "modern_pop": d["mod_pop"][mi], "ancient": d["anc_id"][ai],
                         "ancient_group": d["anc_grp"][ai], "set": d["anc_set"][ai], "date_bp": d["anc_date"][ai],
                         "place": d["anc_place"][ai], "ancient_snps": int((anc[ai] != 9).sum()),
                         "ancient_het": float(het[ai]), "shared_snps": both,
                         "n_segments": len(strict), "total_cm": sum(h - l for _, l, h, _ in strict),
                         "n_segments_cm_only": len(by_cm), "total_cm_cm_only": sum(h - l for _, l, h, _ in by_cm),
                         "longest_cm": max((h - l for _, l, h, _ in runs), default=0.0),
                         "longest_snps": max((n for *_, n in runs), default=0)})
            segs += [(d["mod_id"][mi], d["anc_id"][ai], c, l, h, n, n >= min_snps) for c, l, h, n in by_cm]
        last = rows[-len(anc):]
        print(f"  {d['mod_id'][mi]} ({d['mod_pop'][mi]}): {sum(r['n_segments'] for r in last)} segments "
              f"(≥{min_cm} cM and ≥{min_snps} SNPs), {sum(r['n_segments_cm_only'] for r in last)} by cM only",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(work / "summary.csv", index=False)
    pd.DataFrame(segs, columns=["modern", "ancient", "chrom", "start_cm", "end_cm", "snps", "passes_snp_min"]).to_csv(
        work / "segments.csv", index=False)
    (work / "settings.json").write_text(json.dumps({"min_cm": min_cm, "min_snps": min_snps}))
    report(df)
    print(f"\n{work / 'summary.csv'}\n{work / 'segments.csv'}")


def report(df: pd.DataFrame) -> None:
    sup = {"FIN": "EUR", "GBR": "EUR", "TSI": "EUR", "IBS": "EUR", "CEU": "EUR", "Russian": "EUR",
           "CHB": "EAS", "JPT": "EAS", "Han": "EAS", "PJL": "SAS", "PEL": "AMR",
           "YRI": "AFR", "LWK": "AFR", "Mbuti": "AFR", "you": "you"}
    df = df.assign(super=df["modern_pop"].map(sup).fillna("other"))
    per = df.groupby(["super", "ancient"])["n_segments_cm_only"].mean().unstack(0)
    info = df.drop_duplicates("ancient").set_index("ancient")[["ancient_snps", "ancient_het", "date_bp"]]
    per = info.join(per).sort_values("ancient_snps", ascending=False)
    cols = [c for c in ["you", "EUR", "SAS", "AMR", "EAS", "AFR"] if c in per]
    print(f"\nSegments ≥ min_cm (any SNP count) per ancient individual, mean per group (most SNPs first). "
          f"With the SNP minimum too: {df['n_segments'].sum()} segments in all {len(df):,} pairs.")
    print(per[["ancient_snps", "ancient_het", *cols]].head(40).to_string(float_format=lambda v: f"{v:.1f}"))
    r = per[cols].corrwith(np.log10(per["ancient_snps"].clip(lower=1)))
    print("\nCorrelation of segment count with log10(ancient SNPs):", r.round(2).to_dict())
    if len(cols):
        print("Mean over ancients:", per[cols].mean().round(1).to_dict())


if __name__ == "__main__":
    main()
