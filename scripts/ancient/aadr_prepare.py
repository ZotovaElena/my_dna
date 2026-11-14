"""Build the qpAdm dataset: AADR groups from the config plus you, as two samples.

  find WORD...   search AADR group labels (case-insensitive substring) with sample counts
  check          verify every label in `aadr.groups` exists and every model population is defined
  build          extract the groups, add me_cram and me_csv, run QC; write out_dir/aadr/merged.*

me_cram: one random read per AADR SNP from the CRAM ("pseudo-haploid", as for ancient DNA). Not affected
         by imputation; covers the SNPs with at least one read.
me_csv:  the imputed MyHeritage genotypes at the same SNPs. Imputation copies haplotypes from its
         reference panel, which can pull genotypes toward the populations in that panel.
Agreement between the two is the main robustness check for the qpAdm results.
"""

import difflib
import itertools
import math
import shlex
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py and the module folders
from common import AUTOSOMES, config_arg, load_config, load_myheritage

SNP_COLS = ["id", "chrom", "cm", "pos", "a1", "a2"]  # EIGENSTRAT .snp; geno = copies of a1
COMPLEMENT = str.maketrans("ACGT", "TGCA")
PALINDROMIC = ({"A", "T"}, {"C", "G"})
ME = ("me_cram", "me_csv")
MIN_MAPQ = 25  # reads; the CRAM stores no base qualities, so there is no base-quality filter


# ---------- AADR files and labels ----------

def aadr_files(cfg: dict) -> dict[str, Path]:
    base = cfg["ref_dir"] / cfg["aadr"]["prefix"]
    files = {ext: Path(f"{base}.{ext}") for ext in ("geno", "snp", "ind", "anno")}
    if not files["anno"].exists():  # v66: the .anno has no ".patch" in its name
        files["anno"] = Path(f"{base}.anno".replace(".patch", ""))
    return files


def read_ind(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=["id", "sex", "group"], dtype=str)


def group_counts(cfg: dict) -> pd.Series:
    path = aadr_files(cfg)["ind"]
    if not path.exists():
        raise SystemExit(f"{path} not found: run `bash scripts/setup/download_aadr.sh` first")
    return read_ind(path)["group"].value_counts()


def label_map(cfg: dict) -> dict[str, str]:
    """AADR label -> short group name."""
    mapping = {}
    for short, labels in cfg["aadr"]["groups"].items():
        for label in labels:
            if label in mapping:
                raise SystemExit(f"AADR label {label} is listed under both {mapping[label]} and {short}")
            mapping[label] = short
    return mapping


def find(cfg: dict, words: list[str]) -> None:
    counts = group_counts(cfg)
    anno = annotation_summary(aadr_files(cfg)["anno"])
    for word in words:
        hits = counts[counts.index.str.contains(word, case=False, regex=False)]
        print(f"\n== '{word}': {len(hits)} group(s)")
        table = hits.rename("n").to_frame()
        if anno is not None:
            table = table.join(anno)
        print(table.head(60).to_string() if len(table) else "  (none)")


def annotation_summary(path: Path) -> pd.DataFrame | None:
    """Per group: mean date (years BP), country and median HO SNPs covered, from the .anno columns."""
    if not path.exists():
        return None
    anno = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    # Headers are long descriptions; match on how they start (other headers mention "group" etc. in passing)
    col = lambda start: next((c for c in anno.columns if c.lstrip('"').lower().startswith(start)), None)  # noqa: E731
    group, date, place = col("group id"), col("date mean in bp"), col("political entity")
    snps = next((c for c in anno.columns if "snps hit" in c.lower() and c.lower().rstrip().endswith("on ho snpset)")), None)
    if not group:
        return None
    by = anno[group]
    out = pd.DataFrame(index=by.unique())
    if date:
        out["mean_BP"] = pd.to_numeric(anno[date], errors="coerce").groupby(by).mean().round(-1)
    if place:
        out["country"] = anno.groupby(group)[place].agg(lambda s: ", ".join(sorted(set(s.dropna()))[:2]))
    if snps:
        out["HO_snps"] = pd.to_numeric(anno[snps], errors="coerce").groupby(by).median()
    return out


def check(cfg: dict) -> bool:
    counts = group_counts(cfg)
    ok = True
    print(f"{'group':<12} {'n':>4}  AADR labels")
    for short, labels in cfg["aadr"]["groups"].items():
        n = sum(int(counts.get(label, 0)) for label in labels)
        print(f"{short:<12} {n:>4}  {', '.join(labels)}")
        for label in labels:
            if label not in counts.index:
                ok = False
                close = difflib.get_close_matches(label, counts.index, n=5, cutoff=0.5)
                print(f"{'':<18}MISSING {label}; close: {', '.join(close) or '-'}")
    defined = set(cfg["aadr"]["groups"])
    for name, a in cfg["qpadm"]["analyses"].items():
        undefined = (set(a["sources"]) | set(a["right"])) - defined
        overlap = set(a["sources"]) & set(a["right"])
        if undefined:
            ok = False
            print(f"analysis {name}: not in aadr.groups: {', '.join(sorted(undefined))}")
        if overlap:
            ok = False
            print(f"analysis {name}: both source and right: {', '.join(sorted(overlap))}")
    print("\nAll labels found." if ok else "\nFix the config entries above, then rerun `check`.")
    return ok


# ---------- your genotypes ----------

def csv_genotypes(cfg: dict, snp: pd.DataFrame) -> tuple[list[str], Counter]:
    """Imputed CSV genotypes as EIGENSTRAT codes (copies of a1; 9 = missing) in `snp` order."""
    me = load_myheritage(cfg["raw_file"])
    me = me[me["chrom"].isin(AUTOSOMES) & (me["genotype"].str.len() == 2)]
    me = me.drop_duplicates(["chrom", "pos"], keep=False)
    m = snp[["chrom", "pos", "a1", "a2"]].merge(me[["chrom", "pos", "genotype"]], on=["chrom", "pos"], how="left")
    assert len(m) == len(snp)
    codes, stats = [], Counter()
    for g, a1, a2 in zip(m["genotype"], m["a1"], m["a2"]):
        alleles = {a1, a2}
        if not isinstance(g, str):
            codes.append("9"); stats["not in CSV"] += 1
        elif set(g) <= alleles:
            codes.append(str(g.count(a1))); stats["matched"] += 1
        elif alleles not in PALINDROMIC and set(g.translate(COMPLEMENT)) <= alleles:
            codes.append(str(g.translate(COMPLEMENT).count(a1))); stats["matched after strand flip"] += 1
        else:
            codes.append("9"); stats["allele mismatch (set missing)"] += 1
    return codes, stats


def cram_genotypes(cfg: dict, work: Path, snp: pd.DataFrame, redo: bool) -> list[str]:
    """Pseudo-haploid calls from the CRAM (samtools mpileup | pileupCaller), in `snp` order."""
    prefix = work / "me_cram"
    if redo or not Path(f"{prefix}.geno").exists():
        snp.to_csv(work / "auto.snp", sep="\t", header=False, index=False)
        snp[["chrom", "pos"]].to_csv(work / "auto.pos", sep="\t", header=False, index=False)
        q = lambda p: shlex.quote(str(p))  # noqa: E731
        cmd = (f"samtools mpileup -B -q {MIN_MAPQ} -Q 0 -l {q(work / 'auto.pos')} -f {q(cfg['reference_fasta'])} "
               f"{q(cfg['cram'])} | pileupCaller --randomHaploid --sampleNames me_cram "
               f"--samplePopName me_cram -f {q(work / 'auto.snp')} -e {q(prefix)}")
        print("Calling pseudo-haploid genotypes from the CRAM (one pass over the reads, ~30+ min)...")
        subprocess.run(["bash", "-o", "pipefail", "-c", cmd], check=True)
    ids = pd.read_csv(f"{prefix}.snp", sep=r"\s+", header=None, usecols=[0], dtype=str)[0]
    calls = Path(f"{prefix}.geno").read_text().split()
    assert len(ids) == len(calls), "pileupCaller .snp and .geno disagree"
    by_id = dict(zip(ids, calls))
    return [by_id.get(i, "9") for i in snp["id"]]


# ---------- AADR genotypes (streamed: only the selected samples are decoded) ----------

def geno_format(path: Path) -> tuple[str, int, int]:
    """'GENO' (packed, one record per SNP), 'TGENO' (packed, one record per sample) or 'TEXT' (EIGENSTRAT)."""
    with open(path, "rb") as f:
        head = f.read(48).split()
    if head and head[0] in (b"GENO", b"TGENO"):
        return head[0].decode(), int(head[1]), int(head[2])
    return "TEXT", -1, -1


def unpack(packed: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """2-bit codes at positions idx of packed rows: 4 per byte, first in the high bits; 3 = missing."""
    return (packed[..., idx // 4] >> (6 - 2 * (idx % 4)).astype(np.uint8)) & 3


def header_bytes(path: Path, rlen: int, n_records: int) -> int:
    """Header length from the file size: one full record (classic layout) or 48 bytes (AADR v66 TGENO)."""
    h = path.stat().st_size - rlen * n_records
    if h not in (48, rlen):
        raise SystemExit(f"{path}: size doesn't fit {n_records} records of {rlen} bytes plus a header "
                         f"(left over: {h} bytes); the file is truncated or in an unknown layout")
    return h


def aadr_genotypes(path: Path, n_ind: int, n_snp: int, sel: np.ndarray, chunk: int = 10_000):
    """Yield (snps, len(sel)) uint8 blocks of a1 copies (0/1/2, 9 = missing) in SNP order."""
    fmt, nind, nsnp = geno_format(path)
    if fmt != "TEXT" and (nind, nsnp) != (n_ind, n_snp):
        raise SystemExit(f"{path}: header says {nind} samples × {nsnp} SNPs, but .ind/.snp have {n_ind} × {n_snp}")
    if fmt == "GENO":
        rlen = max(48, math.ceil(n_ind / 4))
        with open(path, "rb") as f:
            f.seek(header_bytes(path, rlen, n_snp))
            for start in range(0, n_snp, chunk):
                k = min(chunk, n_snp - start)
                g = unpack(np.frombuffer(f.read(k * rlen), np.uint8).reshape(k, rlen), sel)
                g[g == 3] = 9
                yield g
    elif fmt == "TGENO":
        rlen = max(48, math.ceil(n_snp / 4))
        head = header_bytes(path, rlen, n_ind)
        snps = np.arange(n_snp)
        mat = np.empty((len(sel), n_snp), np.uint8)
        with open(path, "rb") as f:
            for r, i in enumerate(sel):
                f.seek(head + rlen * int(i))
                mat[r] = unpack(np.frombuffer(f.read(rlen), np.uint8), snps)
        mat[mat == 3] = 9
        for start in range(0, n_snp, chunk):
            yield mat[:, start:start + chunk].T
    else:
        with open(path, "rb") as f:
            while lines := list(itertools.islice(f, chunk)):
                rows = np.frombuffer(b"".join(x.rstrip(b"\r\n") for x in lines), np.uint8)
                yield rows.reshape(len(lines), -1)[:, sel] - ord("0")


def write_merged(blocks, keep: np.ndarray, extra: list[list[str]], geno_out: Path) -> dict:
    """EIGENSTRAT text of the kept SNPs: AADR columns + one column per extra sample.

    Returns per-SNP AADR mean dosage and per-sample missing/het counts for QC.
    """
    extra_arr = np.array([[ord(c) - ord("0") for c in col] for col in extra], np.uint8).T
    means, miss, het = [], 0, 0
    pos = j = 0
    with open(geno_out, "wb") as out:
        for g in blocks:
            k = len(g)
            g = g[keep[pos:pos + k]]
            pos += k
            if not len(g):
                continue
            rows = np.concatenate([g, extra_arr[j:j + len(g)], np.full((len(g), 1), 9, np.uint8)], axis=1)
            rows = rows + ord("0")
            rows[:, -1] = ord("\n")
            out.write(rows.tobytes())
            j += len(g)
            called = g != 9
            n = called.sum(axis=1)
            means.append(np.where(called, g, 0).sum(axis=1) / np.where(n == 0, np.nan, n))
            miss = miss + (~called).sum(axis=0)
            het = het + (g == 1).sum(axis=0)
    if pos != len(keep) or j != len(extra_arr):
        raise SystemExit(f"SNP count mismatch while merging ({pos} read, {len(keep)} expected)")
    return {"mean": np.concatenate(means), "missing": miss, "het": het, "n_snps": j}


def convertf(params: dict, stem: Path) -> None:
    par = stem.with_suffix(".par")
    par.write_text("".join(f"{k}: {v}\n" for k, v in params.items()))
    with open(stem.with_suffix(".log"), "w") as log:
        subprocess.run(["convertf", "-p", str(par)], stdout=log, stderr=subprocess.STDOUT, check=True)


# ---------- QC ----------

def group_qc(ind: pd.DataFrame, stats: dict) -> None:
    """Per group missingness and heterozygosity: a check that samples were decoded and labelled correctly."""
    n = stats["n_snps"]
    per = ind.assign(missing=stats["missing"] / n,
                     het=stats["het"] / np.maximum(n - stats["missing"], 1))
    print("\nAADR samples per group (expect: modern groups ~0-3% missing with het ~0.2-0.3;"
          "\nancient groups are pseudo-haploid in AADR: often 15-65% missing with het 0):")
    print(per.groupby("group", sort=False).agg(n=("id", "size"), missing=("missing", "mean"), het=("het", "mean"))
          .to_string(float_format=lambda v: f"{v:.3f}"))


def qc(codes: dict[str, list[str]], aadr_mean: np.ndarray, csv_stats: Counter) -> bool:
    """Print QC; False if your genotypes don't line up with AADR (then qpAdm results would be meaningless)."""
    n = len(aadr_mean)
    arr = {k: np.array([int(c) for c in v]) for k, v in codes.items()}
    ok = True
    print(f"\nAutosomal AADR SNPs: {n:,}")
    for k, a in arr.items():
        called = a != 9
        both = called & ~np.isnan(aadr_mean)
        r = np.corrcoef(a[both], aadr_mean[both])[0, 1]
        # Correct coding: r ≈ 0.4-0.7. Inverted alleles give r < 0; SNPs out of register give r ≈ 0.
        flag = ("OK" if r > 0.2 else "STOP: allele coding inverted" if r < -0.2
                else "STOP: your SNPs and the AADR SNPs are out of register")
        ok &= flag == "OK"
        print(f"{k}: {called.sum():,} called ({called.mean():.1%}) | r with AADR mean dosage {r:.2f} {flag}")
    print("me_csv allele matching:", dict(csv_stats))
    cram, csv = arr["me_cram"], arr["me_csv"]
    hom = (cram != 9) & np.isin(csv, (0, 2))
    disc = (cram[hom] != csv[hom]).mean()
    print(f"CRAM vs CSV at {hom.sum():,} CSV-homozygous sites: {disc:.2%} disagree "
          "(read errors + imputation errors; a few % is normal, >10% means a coding problem)")
    ok &= disc <= 0.10
    return ok


def build(cfg: dict, redo_cram: bool) -> None:
    if not check(cfg):
        raise SystemExit(1)
    files = aadr_files(cfg)
    work = cfg["out_dir"] / "aadr"
    work.mkdir(parents=True, exist_ok=True)
    labels = label_map(cfg)

    all_ind = read_ind(files["ind"])
    sel = np.flatnonzero(all_ind["group"].isin(labels).to_numpy())
    all_snp = pd.read_csv(files["snp"], sep=r"\s+", header=None, names=SNP_COLS, dtype={"id": str, "chrom": str})
    keep = all_snp["chrom"].isin(AUTOSOMES).to_numpy()
    snp = all_snp[keep].reset_index(drop=True)
    print(f"\nAADR: {len(all_ind):,} samples, {len(all_snp):,} SNPs; using {len(sel)} samples, {len(snp):,} autosomal SNPs")

    csv_codes, csv_stats = csv_genotypes(cfg, snp)
    cram_codes = cram_genotypes(cfg, work, snp, redo_cram)
    codes = {"me_cram": cram_codes, "me_csv": csv_codes}

    print("Extracting the configured groups from AADR...")
    text = work / "merged_text"
    blocks = aadr_genotypes(files["geno"], len(all_ind), len(all_snp), sel)
    stats = write_merged(blocks, keep, [cram_codes, csv_codes], Path(f"{text}.geno"))
    snp.to_csv(f"{text}.snp", sep="\t", header=False, index=False)
    ind = all_ind.iloc[sel].assign(group=lambda d: d["group"].map(labels)).reset_index(drop=True)
    group_qc(ind, stats)
    ind = pd.concat([ind, pd.DataFrame({"id": ME, "sex": "U", "group": ME})], ignore_index=True)
    ind.to_csv(f"{text}.ind", sep="\t", header=False, index=False)

    print("Packing the merged dataset (convertf)...")
    convertf({
        "genotypename": f"{text}.geno", "snpname": f"{text}.snp", "indivname": f"{text}.ind",
        "outputformat": "PACKEDANCESTRYMAP", "genotypeoutname": work / "merged.geno",
        "snpoutname": work / "merged.snp", "indivoutname": work / "merged.ind", "hashcheck": "NO",
    }, work / "convertf_pack")
    for ext in ("geno", "snp", "ind"):
        Path(f"{text}.{ext}").unlink()

    if not qc(codes, stats["mean"], csv_stats):
        raise SystemExit("\nQC failed: don't run qpAdm on this dataset (see the STOP lines above).")
    print(f"\nDataset: {work / 'merged'}.{{geno,snp,ind}}  →  next: python scripts/ancient/qpadm.py")


def main() -> None:
    parser = config_arg(__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("find", help="search AADR group labels").add_argument("words", nargs="+")
    sub.add_parser("check", help="verify configured labels")
    b = sub.add_parser("build", help="build out_dir/aadr/merged.*")
    b.add_argument("--redo-cram", action="store_true", help="call the CRAM again instead of reusing me_cram.*")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "find":
        find(cfg, args.words)
    elif args.cmd == "check":
        raise SystemExit(0 if check(cfg) else 1)
    else:
        build(cfg, args.redo_cram)


if __name__ == "__main__":
    main()
