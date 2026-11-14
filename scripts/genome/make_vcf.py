"""Write sample sites and VCFs.

sites   autosomal positions, used to subset 1000 Genomes
grch37  VCF on 1000G REF/ALT alleles (for plink: ROH, PCA, ADMIXTURE, PRS)
pgx     GRCh38 VCF at PharmCAT's positions, matched by rsID (for scripts/drugs/pharmcat.sh)
"""

import gzip
import json
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py and the module folders
from common import AUTOSOMES, alt_copies, chrom_key, config_arg, load_config, load_myheritage

GT = ("0/0", "0/1", "1/1")


def to_gt(genotype: str, ref: str, alt: str) -> str | None:
    """Diploid genotype string → VCF GT against REF/ALT; None if not diploid or alleles don't match."""
    n = alt_copies(genotype, ref, alt) if len(genotype) == 2 else None
    return None if n is None else GT[n]


def write_vcf(df: pd.DataFrame, sample_id: str, path) -> None:
    header = [
        "##fileformat=VCFv4.2",
        "##reference=GRCh37",
        *[f"##contig=<ID={c}>" for c in AUTOSOMES],
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
    ]
    body = pd.DataFrame({
        "#CHROM": df["chrom"], "POS": df["pos"], "ID": df["rsid"], "REF": df["ref"], "ALT": df["alt"],
        "QUAL": ".", "FILTER": ".", "INFO": ".", "FORMAT": "GT", sample_id: df["gt"],
    })
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        body.to_csv(f, sep="\t", index=False)


def grch37(cfg: dict, sample: pd.DataFrame) -> None:
    out = cfg["out_dir"]
    sites = pd.read_csv(out / "1kg_sites.tsv", sep="\t", names=["chrom", "pos", "ref", "alt"], dtype={"chrom": str})
    sites = sites.drop_duplicates(["chrom", "pos"], keep=False)  # ambiguous split multi-allelics
    m = sample.merge(sites, on=["chrom", "pos"])
    m["gt"] = [to_gt(g, r, a) for g, r, a in zip(m["genotype"], m["ref"], m["alt"])]
    print(f"Sample sites in 1000G: {len(m)} | allele mismatches dropped: {m['gt'].isna().sum()}")
    m = m.dropna(subset=["gt"])
    m = m.sort_values(["chrom", "pos"], key=chrom_key)
    write_vcf(m, cfg["sample_id"], out / "sample.grch37.vcf")


def pharmcat_positions(cfg: dict) -> Path:
    """PharmCAT's own list of the GRCh38 positions it reads (from scripts/setup/download_pharmcat.sh)."""
    files = sorted(p for p in (cfg["ref_dir"] / "pharmcat").glob("pharmcat_positions*")
                   if p.name.endswith((".vcf", ".vcf.gz", ".vcf.bgz")) and "uniallelic" not in p.name)
    if not files:
        raise SystemExit(f"No pharmcat_positions*.vcf* in {cfg['ref_dir'] / 'pharmcat'}: "
                         "run `bash scripts/setup/download_pharmcat.sh` first")
    return files[-1]


def read_vcf_rows(path: Path):
    with (gzip.open(path, "rt") if path.suffix in (".gz", ".bgz") else open(path)) as f:
        for line in f:
            if not line.startswith("#"):
                yield line.rstrip("\n").split("\t")


def pgx(cfg: dict) -> None:
    """GRCh38 VCF of your genotypes at PharmCAT's positions, matched by rsID.

    rsIDs are the same in every genome build and both files are on the forward strand, so no liftover
    (and no Ensembl or genome download) is needed. Only SNVs: the raw file has no indels. Positions
    PharmCAT reads but the raw file lacks stay missing, which PharmCAT reports as such.
    """
    pos_file = pharmcat_positions(cfg)
    sample = load_myheritage(cfg["raw_file"])
    calls = dict(zip(sample["rsid"], sample["genotype"]))
    records, stats = [], {"positions": 0, "snv": 0, "found": 0, "mismatch": 0}
    for row in read_vcf_rows(pos_file):
        chrom, pos, ids, ref, alts = row[:5]
        stats["positions"] += 1
        alleles = [ref, *alts.split(",")]
        if any(len(a) != 1 or a not in "ACGT" for a in alleles):
            continue  # indel or symbolic allele
        stats["snv"] += 1
        g = next((calls[i] for i in ids.split(";") if i in calls), None)
        if g is None:
            continue
        stats["found"] += 1
        if not set(g) <= set(alleles):
            stats["mismatch"] += 1
            continue
        idx = sorted(alleles.index(a) for a in g)
        info = next((x for x in row[7].split(";") if x.startswith("PX=")), "") if len(row) > 7 else ""
        records.append((chrom, int(pos), ids, ref, alts, info or ".", "/".join(map(str, idx))))
    print(f"PharmCAT positions: {stats['positions']:,} ({stats['snv']:,} SNVs) | in your raw file: "
          f"{stats['found']:,} | allele mismatches dropped: {stats['mismatch']:,} | written: {len(records):,}")
    rank = {c: i for i, c in enumerate(f"chr{c}" for c in [*AUTOSOMES, "X", "Y", "M"])}
    records.sort(key=lambda r: (rank.get(r[0], 99), r[1]))
    chroms = list(dict.fromkeys(r[0] for r in records))
    out = cfg["out_dir"] / "pharmcat" / "sample.pgx.grch38.vcf"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("##fileformat=VCFv4.2\n##reference=GRCh38\n")
        f.write("".join(f"##contig=<ID={c},assembly=GRCh38>\n" for c in chroms))
        f.write('##INFO=<ID=PX,Number=.,Type=String,Description="PGx gene (from PharmCAT positions)">\n')
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{cfg['sample_id']}\n")
        for chrom, pos, ids, ref, alts, info, gt in records:
            f.write(f"{chrom}\t{pos}\t{ids}\t{ref}\t{alts}\t.\tPASS\t{info}\tGT\t{gt}\n")
    (out.parent / "pgx_input_stats.json").write_text(json.dumps({**stats, "written": len(records),
                                                                  "positions_file": pos_file.name}, indent=1))
    print("Wrote", out)


def main() -> None:
    parser = config_arg(__doc__)
    parser.add_argument("step", choices=["sites", "grch37", "pgx"])
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.step == "pgx":
        return pgx(cfg)
    sample = load_myheritage(cfg["raw_file"])
    sample = sample[sample["chrom"].isin(AUTOSOMES)]
    if args.step == "sites":
        sample[["chrom", "pos"]].to_csv(cfg["out_dir"] / "sample_sites.tsv", sep="\t", header=False, index=False)
    else:
        grch37(cfg, sample)


if __name__ == "__main__":
    main()
