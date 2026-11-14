"""Ancestrify-style IBS scan: your genome against every ancient individual in the AADR, with controls.

Ancient-DNA "match" services compare an uploaded genome with ~15,000 ancient individuals and report
half-identical (IBS) segments: runs of SNPs with no opposite homozygote (you AA, them BB), above a length
(cM) and a marker-count minimum. This repeats that scan on the AADR v66 Human Origins panel:

  you        GLIMPSE2-imputed genome (genome/glimpse.sh run `ibs_all.run`): best-guess diploid genotypes at the
             panel SNPs, a call kept only if its genotype probability is >= ibs_all.min_gp. Only chromosomes
             that are imputed so far are scanned (for everyone, so all are compared on the same SNPs).
             The MyHeritage CSV shares only ~57k SNPs with this panel, too few for a marker minimum.
  ancients   every individual of the outgroup-f3 ranking (outputs/f3/outgroup_f3_ancient.csv, 14,743)
  controls   present-day AADR people (ibs_all.controls) scanned the same way. If a Russian, a Han Chinese
             or a Yoruba "matches" the same ancient individuals as often as you do, a match says nothing
             about your ancestry: it reflects the ancient genome's data (missing SNPs, one read per SNP).

Thresholds: ibs_all.min_cm and ibs_all.min_snps (Ancestrify-like defaults 7 cM, 200 SNPs).
Output in out_dir/ibs_all/: per_ancient.csv (one row per ancient: your count + control means),
segments.csv (every passing segment, all people), counts.npz (ancient × person matrices), settings.json.
Runs in parallel (--jobs); about 10-30 minutes for all chromosomes.
"""

import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py and the module folders
from common import AUTOSOMES, config_arg, load_config

from ancient import aadr_prepare as prep

DEFAULTS = {
    "run": "me", "min_gp": 0.9, "min_cm": 7, "min_snps": 200,
    "controls": {"Russian": 5, "Ukrainian": 3, "Belarusian": 2, "Finnish": 3, "English": 2, "Sardinian": 2,
                 "Han": 3, "Yoruba": 3, "Mbuti": 2},
}
SUPER = {"Russian": "EUR", "Ukrainian": "EUR", "Belarusian": "EUR", "Finnish": "EUR", "English": "EUR",
         "Sardinian": "EUR", "French": "EUR", "Han": "EAS", "Japanese": "EAS", "Yoruba": "AFR", "Mbuti": "AFR",
         "Mende": "AFR"}
G: dict = {}


def your_genotypes(cfg: dict, s: dict, snp: pd.DataFrame, work: Path) -> tuple[np.ndarray, list[str]]:
    """Copies of the .snp first allele (0/1/2, 9 = missing or uncertain) from the GLIMPSE run, per panel SNP."""
    run = cfg["ref_dir"] / "glimpse" / "runs" / s["run"]
    done = [c for c in AUTOSOMES if (run / f"chr{c}.bcf.csi").exists()]
    if not done:
        raise SystemExit(f"No imputed chromosomes in {run}: run bash scripts/genome/glimpse.sh first")
    sites = work / "panel_sites.tsv"
    snp[snp["chrom"].isin(done)][["chrom", "pos"]].to_csv(sites, sep="\t", header=False, index=False)
    parts = []
    for c in done:
        q = subprocess.run(["bcftools", "query", "-T", str(sites), "-f", "%POS\t%REF\t%ALT[\t%GT\t%GP]\n",
                            str(run / f"chr{c}.bcf")], capture_output=True, text=True, check=True).stdout
        d = pd.read_csv(io.StringIO(q), sep="\t", header=None, names=["pos", "ref", "alt", "gt", "gp"])
        parts.append(d.assign(chrom=c))
    v = pd.concat(parts, ignore_index=True).drop_duplicates(["chrom", "pos"], keep=False)
    v = v[v["gt"].str.len() == 3]                           # diploid "a|b" calls only
    v["alt_n"] = v["gt"].str[0].astype(int) + v["gt"].str[2].astype(int)
    v["max_gp"] = v["gp"].str.split(",", expand=True).astype(float).max(axis=1)
    m = snp.reset_index(names="k").merge(v, on=["chrom", "pos"])
    same = (m["a1"] == m["ref"]) & (m["a2"] == m["alt"])
    swap = (m["a1"] == m["alt"]) & (m["a2"] == m["ref"])
    ok = ((same | swap) & (m["max_gp"] >= s["min_gp"])).to_numpy()
    a1 = np.where(same, 2 - m["alt_n"], m["alt_n"])       # copies of the .snp first allele, like the .geno
    g = np.full(len(snp), 9, np.uint8)
    g[m["k"].to_numpy()[ok]] = a1[ok]
    return g, done


def runs(m: np.ndarray, a: np.ndarray, chrom: np.ndarray, cm: np.ndarray):
    """Half-identical runs between two genotype vectors on the SNPs called in both: (chrom, start, end cM, SNPs)."""
    idx = np.flatnonzero((m != 9) & (a != 9))
    if len(idx) == 0:
        return np.zeros(0, np.int8), np.zeros(0), np.zeros(0), np.zeros(0, int)
    mi, ai, ch = m[idx], a[idx], chrom[idx]
    opp = ((mi == 0) & (ai == 2)) | ((mi == 2) & (ai == 0))
    new = np.r_[True, ch[1:] != ch[:-1]] | np.r_[False, opp[:-1]]
    run = np.cumsum(new) - 1
    ok = ~opp
    r, c, chs = run[ok], cm[idx][ok], ch[ok]
    if len(r) == 0:
        return np.zeros(0, np.int8), np.zeros(0), np.zeros(0), np.zeros(0, int)
    st = np.r_[0, np.flatnonzero(r[1:] != r[:-1]) + 1]
    en = np.r_[st[1:], len(r)] - 1
    return chs[st], c[st], c[en], en - st + 1


def _init(mods, keep, chrom, cm, geno, head, rlen, min_cm, min_snps):
    G.update(mods=mods, keep=keep, chrom=chrom, cm=cm, geno=geno, head=head, rlen=rlen, min_cm=min_cm,
             min_snps=min_snps)


def _scan(batch: list[tuple[int, int]]):
    """For each (row, AADR index): segment counts and lengths against every modern, plus passing segments."""
    nm = len(G["mods"])
    out = np.zeros((len(batch), nm, 5), np.float32)   # strict n, strict cM, cM-only n, longest cM, shared SNPs
    called = np.zeros(len(batch), np.int64)
    segs = []
    with open(G["geno"], "rb") as fh:
        for b, (row, i) in enumerate(batch):
            fh.seek(G["head"] + G["rlen"] * i)
            a = prep.unpack(np.frombuffer(fh.read(G["rlen"]), np.uint8), G["keep"])
            a[a == 3] = 9
            called[b] = int((a != 9).sum())
            for j in range(nm):
                ch, lo, hi, n = runs(G["mods"][j], a, G["chrom"], G["cm"])
                ln = hi - lo
                by_cm = ln >= G["min_cm"]
                strict = by_cm & (n >= G["min_snps"])
                out[b, j] = (strict.sum(), ln[strict].sum(), by_cm.sum(), ln.max() if len(ln) else 0,
                             int(((G["mods"][j] != 9) & (a != 9)).sum()))
                segs += [(row, j, int(ch[k]), float(lo[k]), float(hi[k]), int(n[k])) for k in np.flatnonzero(strict)]
    return [r for r, _ in batch], out, called, segs


def main() -> None:
    parser = config_arg(__doc__)
    parser.add_argument("--jobs", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--min-cm", type=float)
    parser.add_argument("--min-snps", type=int)
    parser.add_argument("--limit", type=int, help="only the first N ancients (testing)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    s = {**DEFAULTS, **cfg.get("ibs_all", {})}
    s["min_cm"] = args.min_cm or s["min_cm"]
    s["min_snps"] = args.min_snps or s["min_snps"]
    work = cfg["out_dir"] / "ibs_all"
    work.mkdir(parents=True, exist_ok=True)
    f = prep.aadr_files(cfg)
    ind = prep.read_ind(f["ind"])
    snp = pd.read_csv(f["snp"], sep=r"\s+", header=None, names=["id", "chrom", "gpos", "pos", "a1", "a2"],
                      dtype={"chrom": str})

    t0 = time.time()
    you, done = your_genotypes(cfg, s, snp, work)
    keep = np.flatnonzero(snp["chrom"].isin(done).to_numpy())
    print(f"You: {int((you[keep] != 9).sum()):,} of {len(keep):,} panel SNPs on chromosomes {','.join(done)} "
          f"(GP >= {s['min_gp']})", flush=True)

    fmt, nind, nsnp = prep.geno_format(f["geno"])
    if fmt != "TGENO":
        raise SystemExit("Expects the AADR v66 TGENO file (one record per person)")
    rlen = max(48, -(-nsnp // 4))
    head = prep.header_bytes(f["geno"], rlen, nind)
    pos_of = {v: k for k, v in enumerate(ind["id"])}
    rng = np.random.default_rng(5)
    mods, mod_id, mod_pop = [you[keep]], ["you"], ["you"]
    with open(f["geno"], "rb") as fh:
        for pop, k in s["controls"].items():
            pool = np.flatnonzero(ind["group"].to_numpy() == pop)
            for i in rng.choice(pool, size=min(k, len(pool)), replace=False):
                fh.seek(head + rlen * int(i))
                g = prep.unpack(np.frombuffer(fh.read(rlen), np.uint8), keep)
                g[g == 3] = 9
                mods.append(g)
                mod_id.append(ind.at[i, "id"])
                mod_pop.append(pop)
    mods = np.stack(mods)

    f3 = pd.read_csv(cfg["out_dir"] / "f3" / "outgroup_f3_ancient.csv")
    f3 = f3[f3["id"].isin(pos_of)].reset_index(drop=True)
    if args.limit:
        f3 = f3.head(args.limit)
    idx = f3["id"].map(pos_of).to_numpy()
    chrom = snp["chrom"].to_numpy()[keep].astype(np.int8)
    cm = (snp["gpos"].to_numpy()[keep] * 100).astype(np.float64)
    print(f"{len(f3):,} ancients x {len(mods)} people (you + {len(mods) - 1} controls), {len(keep):,} SNPs, "
          f">= {s['min_cm']} cM and >= {s['min_snps']} SNPs, {args.jobs} processes", flush=True)

    order = np.argsort(idx)                               # read the file front to back
    batches = [[(int(r), int(idx[r])) for r in order[k:k + 100]] for k in range(0, len(order), 100)]
    res = np.zeros((len(f3), len(mods), 5), np.float32)
    called = np.zeros(len(f3), np.int64)
    segs = []
    with ProcessPoolExecutor(args.jobs, initializer=_init,
                             initargs=(mods, keep, chrom, cm, f["geno"], head, rlen, s["min_cm"], s["min_snps"])) as ex:
        for k, (rows, out, cl, sg) in enumerate(ex.map(_scan, batches)):
            res[rows], called[rows] = out, cl
            segs += sg
            if (k + 1) % 20 == 0 or k + 1 == len(batches):
                print(f"  {min((k + 1) * 100, len(f3)):,}/{len(f3):,} ancients ({time.time() - t0:.0f} s)", flush=True)

    sup = np.array([SUPER.get(p, "you" if p == "you" else "other") for p in mod_pop])
    np.savez_compressed(work / "counts.npz", strict_n=res[..., 0], strict_cm=res[..., 1], cm_only_n=res[..., 2],
                        longest_cm=res[..., 3], shared=res[..., 4], ancient=f3["id"].to_numpy(),
                        person=np.array(mod_id), pop=np.array(mod_pop), super=sup)
    seg = pd.DataFrame(segs, columns=["row", "person", "chrom", "start_cm", "end_cm", "snps"])
    seg = seg.assign(ancient=f3["id"].to_numpy()[seg["row"]], person_id=np.array(mod_id)[seg["person"]],
                     pop=np.array(mod_pop)[seg["person"]], length_cm=seg["end_cm"] - seg["start_cm"])
    seg = seg.drop(columns=["row", "person"])
    seg.to_csv(work / "segments.csv", index=False)

    per = f3[["id", "iid", "group", "date_bp", "country", "lat", "lon", "mt_hg", "rank", "f3"]].copy()
    per["ancient_snps"] = called
    per["you_n"] = res[:, 0, 0].astype(int)
    per["you_cm"] = res[:, 0, 1]
    per["you_cm_only_n"] = res[:, 0, 2].astype(int)
    per["you_longest_cm"] = res[:, 0, 3]
    per["you_shared_snps"] = res[:, 0, 4].astype(int)
    ctrl = sup != "you"
    per["controls_with_segment"] = (res[:, ctrl, 0] > 0).sum(1)
    per["controls"] = int(ctrl.sum())
    for g in ["EUR", "EAS", "AFR"]:
        if (sup == g).any():
            per[f"{g}_n_mean"] = res[:, sup == g, 0].mean(1)
            per[f"{g}_cm_only_mean"] = res[:, sup == g, 2].mean(1)
    per.to_csv(work / "per_ancient.csv", index=False)

    tot = pd.DataFrame({"person": mod_id, "pop": mod_pop, "super": sup, "segments": res[:, :, 0].sum(0).astype(int),
                        "ancients_matched": (res[:, :, 0] > 0).sum(0), "total_cm": res[:, :, 1].sum(0),
                        "segments_cm_only": res[:, :, 2].sum(0).astype(int)})
    tot.to_csv(work / "per_person.csv", index=False)
    (work / "settings.json").write_text(json.dumps({
        "min_cm": s["min_cm"], "min_snps": s["min_snps"], "min_gp": s["min_gp"], "chromosomes": done,
        "snps": int(len(keep)), "you_snps": int((you[keep] != 9).sum()), "ancients": int(len(f3)),
        "controls": int(ctrl.sum()), "limit": args.limit}))
    pd.set_option("display.width", 200)
    print("\nPer person (segments passing both thresholds, over all ancients):")
    print(tot.to_string(index=False, float_format=lambda v: f"{v:.0f}"))
    top = per[per["you_n"] > 0].sort_values(["you_cm", "you_longest_cm"], ascending=False)
    print(f"\nYour matches: {len(top)} ancients")
    print(top.head(20)[["iid", "group", "date_bp", "country", "rank", "ancient_snps", "you_n", "you_cm",
                        "controls_with_segment"]].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"\n{work / 'per_ancient.csv'}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
