"""Real IBD between you and ancient genomes with ancIBD (Ringbauer et al. 2024).

Steps (the downloads are separate shell scripts you run yourself):
  select    pick ancient shotgun genomes from the AADR annotation (config `ancibd`) -> ancients.tsv
            then: bash scripts/setup/download_ancient_bams.sh manifest   (ENA file lists)
  match     match the ENA file lists to the selected individuals -> manifest.tsv (what fetch downloads)
            then: bash scripts/setup/download_ancient_bams.sh fetch      (BAMs to ancient_dir/bam/)
            then: bash scripts/genome/glimpse.sh outputs/ancibd/glimpse_samples.tsv ancient  (imputation)
  run       per chromosome: the GLIMPSE runs (ancibd.runs) cut to the 1240k sites and merged, then ancIBD-run
            (conda env "ancibd", support files from setup/download_ancibd.sh) on you x every ancient and, as a
            calibration, every ancient x ancient pair -> ibd/ancibd.ch<N>.tsv. Done chromosomes are skipped.
            --chrom limits the chromosomes (e.g. --chrom 1 to test while the other imputations run).
  summary   segments >= ancibd.min_cm per pair -> you_vs_ancient.csv, ancient_pairs.csv, segments.csv

Work in out_dir/ancibd/.
"""

import itertools
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py and the module folders
from common import config_arg, load_config


def read_anno(cfg: dict) -> pd.DataFrame:
    prefix = cfg["ref_dir"] / cfg["aadr"]["prefix"]
    path = Path(f"{prefix}.anno")
    if not path.exists():
        path = Path(f"{prefix}.anno".replace(".patch", ""))
    a = pd.read_csv(path, sep="\t", low_memory=False)
    c = a.columns
    num = lambda i: pd.to_numeric(a[c[i]], errors="coerce")
    return pd.DataFrame({
        "id": a[c[0]], "iid": a[c[2]].astype(str), "pub": a[c[6]], "repo": a[c[8]].astype(str),
        "date_bp": num(10), "group": a[c[14]], "locality": a[c[15]], "country": a[c[16]],
        "lat": num(17), "lon": num(18), "depth": pd.concat([num(23), num(24)], axis=1).max(axis=1),
        "assessment": a[c[47]].astype(str),
    })


def select(cfg: dict, work: Path) -> pd.DataFrame:
    o = cfg["ancibd"]
    d = read_anno(cfg)
    d = d[d["id"].str.contains(r"\.SG$", regex=True) & (d["date_bp"] > 0) & (d["date_bp"] <= o["max_age_bp"])
          & (d["depth"] >= o["min_depth"]) & d["repo"].str.contains("ENA:", regex=False)
          & ~d["assessment"].str.contains("QUESTIONABLE|CRITICAL|FAIL", case=False)]
    box = o.get("box", {})
    d = d[(d["lat"] >= box.get("min_lat", -90)) & d["lon"].between(box.get("min_lon", -180), box.get("max_lon", 180))]
    d = d.drop_duplicates("iid")
    pick = [d[d["country"] == country].sort_values(["date_bp", "depth"], ascending=[True, False]).head(n)
            for country, n in o["per_country"].items()]
    sel = pd.concat(pick, ignore_index=True)
    sel["projects"] = sel["repo"].map(lambda r: " ".join(re.findall(r"ENA:(\w+)", r)))
    sel.to_csv(work / "ancients.tsv", sep="\t", index=False)
    print(sel[["iid", "country", "locality", "date_bp", "depth", "pub", "projects"]].to_string(index=False))
    print(f"\n{len(sel)} individuals -> {work / 'ancients.tsv'}")
    print("Next: bash scripts/setup/download_ancient_bams.sh manifest")
    return sel


def match(cfg: dict, work: Path) -> None:
    """ENA run lists (one per project, from setup/download_ancient_bams.sh manifest) -> files per individual."""
    sel = pd.read_csv(work / "ancients.tsv", sep="\t")
    ena_dir = cfg.get("ancient_dir", cfg["ref_dir"] / "ancient") / "ena"
    runs = pd.concat([pd.read_csv(p, sep="\t", dtype=str).assign(project=p.stem) for p in sorted(ena_dir.glob("*.tsv"))
                      if p.stat().st_size], ignore_index=True).fillna("")
    rows, missing = [], []
    for _, s in sel.iterrows():
        key = re.escape(s["iid"])
        hit = runs[runs["project"].isin(s["projects"].split())
                   & (runs["sample_alias"].str.fullmatch(key, case=False)
                      | runs["sample_title"].str.fullmatch(key, case=False)
                      | runs["sample_alias"].str.contains(rf"(?:^|[_\-.]){key}(?:$|[_\-.])", case=False)
                      | runs["sample_title"].str.contains(rf"(?:^|[_\-.]){key}(?:$|[_\-.])", case=False))]
        bams = [(u, b) for _, r in hit.iterrows() for u, b in zip(r["submitted_ftp"].split(";"), r["submitted_bytes"].split(";"))
                if u.endswith(".bam")]
        if not bams:
            missing.append((s["iid"], "no runs" if hit.empty else "runs but no BAM (FASTQ only)"))
            continue
        rows += [{"iid": s["iid"], "url": u if u.startswith("ftp") else f"ftp://{u}", "bytes": int(b or 0)}
                 for u, b in bams]
    man = pd.DataFrame(rows)
    man.to_csv(work / "manifest.tsv", sep="\t", index=False)
    if len(man):
        per = man.groupby("iid")["bytes"].sum()
        print(f"{len(per)} individuals with BAMs, {len(man)} files, {per.sum() / 1e9:.1f} GB in total "
              f"(largest {per.max() / 1e9:.1f} GB)")
    for iid, why in missing:
        print(f"  {iid}: {why}")
    print(f"-> {work / 'manifest.tsv'}\nNext: bash scripts/setup/download_ancient_bams.sh fetch")


def ancibd_files(cfg: dict) -> dict:
    d = cfg["ref_dir"] / "ancibd_data" / "data"
    f = {"filters": d / "filters", "map": d / "map" / "v51.1_1240k.snp", "afs": d / "afs"}
    if not f["map"].exists():
        raise SystemExit(f"Missing {f['map']}: run bash scripts/setup/download_ancibd.sh")
    return f


def run(cfg: dict, work: Path, chroms: list[int]) -> None:
    a = cfg["ancibd"]
    f = ancibd_files(cfg)
    runs = cfg["ref_dir"] / "glimpse" / "runs"
    me = cfg.get("sample_id", "me")
    env = dict(os.environ, PATH=f"{Path(sys.executable).parent}:{os.environ['PATH']}")   # ancIBD calls bcftools
    (work / "vcf").mkdir(exist_ok=True)
    (work / "ibd").mkdir(exist_ok=True)
    for c in chroms:
        out = work / "ibd" / f"ancibd.ch{c}.tsv"
        if out.exists():
            print(f"chr{c}: done", flush=True)
            continue
        bcfs = [runs / r / f"chr{c}.bcf" for r in a.get("runs", ["me", "ancient"])]
        missing = [str(b) for b in bcfs if not b.exists()]
        if missing:
            print(f"chr{c}: not imputed yet: {' '.join(missing)}", flush=True)
            continue
        sites = f["filters"] / f"snps_bcftools_ch{c}.csv"
        vcf = work / "vcf" / f"chr{c}.vcf.gz"
        cut = []
        for b in bcfs:                                  # 1240k sites only: ~5% of the imputed sites
            t = work / "vcf" / f"{b.parent.name}.chr{c}.bcf"
            subprocess.run(["bcftools", "view", "-T", str(sites), "-m2", "-M2", "-v", "snps", "-Ob", "-o", str(t),
                            "--write-index", str(b)], check=True)
            cut.append(str(t))
        if len(cut) > 1:
            subprocess.run(["bcftools", "merge", "-Oz", "-o", str(vcf), "--write-index", *cut], check=True)
        else:
            subprocess.run(["bcftools", "view", "-Oz", "-o", str(vcf), "--write-index", cut[0]], check=True)
        for t in cut:
            Path(t).unlink(missing_ok=True)
            Path(t + ".csi").unlink(missing_ok=True)
        names = subprocess.run(["bcftools", "query", "-l", str(vcf)], capture_output=True, text=True,
                               check=True).stdout.split()
        others = [n for n in names if n != me]
        pairs = [(me, n) for n in others] if me in names else []
        if a.get("ancient_pairs", True):
            pairs += list(itertools.combinations(others, 2))
        if not pairs:
            pairs = [(me, me)]                          # one genome only: a self-pair just to test the setup
        pf = work / "ibd" / "pairs.txt"
        pf.write_text("".join(f"{x} {y}\n" for x, y in pairs))
        print(f"chr{c}: {len(names)} genomes, {len(pairs)} pairs", flush=True)
        subprocess.run(["conda", "run", "--no-capture-output", "-n", a.get("env", "ancibd"), "ancIBD-run",
                        "--vcf", str(vcf), "--ch", str(c), "--pair", str(pf),
                        "--marker_path", str(sites), "--map_path", str(f["map"]),
                        "--af_path", str(f["afs"] / f"v51.1_1240k_AF_ch{c}.tsv"),
                        "--out", str(work / "ibd"), "--prefix", "ancibd", "--min", str(a.get("min_cm_call", 6))],
                       check=True, env=env)
        (work / "ibd" / f"ancibd.ch{c}.h5").unlink(missing_ok=True)   # intermediate, rebuilt from the VCF


def summary(cfg: dict, work: Path) -> None:
    a = cfg["ancibd"]
    me = cfg.get("sample_id", "me")
    files = sorted((work / "ibd").glob("ancibd.ch*.tsv"), key=lambda p: int(re.search(r"ch(\d+)", p.name)[1]))
    if not files:
        raise SystemExit("No ancIBD output yet: run the run step first")
    seg = pd.concat([pd.read_csv(p, sep="\t") for p in files], ignore_index=True)
    done = sorted(int(re.search(r"ch(\d+)", p.name)[1]) for p in files)
    seg = seg[seg["iid1"] != seg["iid2"]]
    seg.to_csv(work / "segments.csv", index=False)
    anc = pd.read_csv(work / "ancients.tsv", sep="\t").set_index("iid")
    cuts = [a["min_cm"], 12, 16, 20]
    def per_pair(d: pd.DataFrame) -> pd.Series:
        cm = d["lengthM"] * 100
        r = {f"n_ge{c}": int((cm >= c).sum()) for c in cuts}
        r[f"total_cm_ge{a['min_cm']}"] = float(cm[cm >= a["min_cm"]].sum())
        r["longest_cm"] = float(cm.max()) if len(cm) else 0.0
        return pd.Series(r)
    seg = seg.assign(other=seg["iid2"].where(seg["iid1"] == me, seg["iid1"]))
    mine = seg[(seg["iid1"] == me) | (seg["iid2"] == me)]
    you = (mine.groupby("other").apply(per_pair, include_groups=False)
           .reindex(anc.index.intersection(pd.read_csv(work / "ibd" / "pairs.txt", sep=" ", header=None)[1]))
           .fillna(0).join(anc[["group", "date_bp", "locality", "country", "depth"]]))
    you = you.sort_values([f"n_ge{a['min_cm']}", "longest_cm"], ascending=False)
    you.index.name = "ancient"
    you.to_csv(work / "you_vs_ancient.csv")
    pairs = seg[(seg["iid1"] != me) & (seg["iid2"] != me)]
    ap = pairs.groupby(["iid1", "iid2"]).apply(per_pair, include_groups=False).reset_index()
    ap = ap[ap[f"n_ge{a['min_cm']}"] > 0].sort_values(f"total_cm_ge{a['min_cm']}", ascending=False)
    ap.to_csv(work / "ancient_pairs.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"Chromosomes done: {done}")
    print(f"You vs each ancient (segments >= {a['min_cm']} cM):")
    print(you.head(30).to_string(float_format=lambda v: f"{v:.1f}"))
    print(f"\nAncient pairs sharing >= {a['min_cm']} cM (calibration): {len(ap)}")
    print(ap.head(15).to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"\n{work / 'you_vs_ancient.csv'}")


def main() -> None:
    parser = config_arg(__doc__)
    parser.add_argument("step", choices=["select", "match", "run", "summary"])
    parser.add_argument("--chrom", type=int, nargs="*", default=list(range(1, 23)))
    args = parser.parse_args()
    cfg = load_config(args.config)
    work = cfg["out_dir"] / "ancibd"
    work.mkdir(parents=True, exist_ok=True)
    if args.step == "select":
        select(cfg, work)
    elif args.step == "match":
        match(cfg, work)
    elif args.step == "run":
        run(cfg, work, args.chrom)
    else:
        summary(cfg, work)


if __name__ == "__main__":
    main()
