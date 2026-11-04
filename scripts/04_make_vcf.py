"""Write sample sites and VCFs.

sites   autosomal positions, used to subset 1000 Genomes
grch37  VCF on 1000G REF/ALT alleles (for plink: ROH, PCA, ADMIXTURE, PRS)
grch38  lifted-over VCF via `snps` with Ensembl reference alleles (for PharmCAT)
"""

import pandas as pd

from common import AUTOSOMES, config_arg, load_config, load_myheritage

GT = ("0/0", "0/1", "1/1")


def to_gt(genotype: str, ref: str, alt: str) -> str | None:
    """Diploid genotype string → VCF GT against REF/ALT; None if alleles don't match."""
    if len(genotype) != 2 or not set(genotype) <= {ref, alt}:
        return None
    return GT[genotype.count(alt)]


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
    m = m.sort_values(["chrom", "pos"], key=lambda s: s.astype(int) if s.name == "chrom" else s)
    write_vcf(m, cfg["sample_id"], out / "sample.grch37.vcf")


def grch38(cfg: dict) -> None:
    from snps import SNPs

    s = SNPs(str(cfg["raw_file"]), output_dir=str(cfg["out_dir"]), resources_dir=str(cfg["ref_dir"] / "snps"))
    print(s.source, s.build, s.count)
    s.remap(38)
    print("Wrote", s.to_vcf("sample.grch38.vcf", chrom_prefix="chr"))


def main() -> None:
    parser = config_arg(__doc__)
    parser.add_argument("step", choices=["sites", "grch37", "grch38"])
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.step == "grch38":
        return grch38(cfg)
    sample = load_myheritage(cfg["raw_file"])
    sample = sample[sample["chrom"].isin(AUTOSOMES)]
    if args.step == "sites":
        sample[["chrom", "pos"]].to_csv(cfg["out_dir"] / "sample_sites.tsv", sep="\t", header=False, index=False)
    else:
        grch37(cfg, sample)


if __name__ == "__main__":
    main()
