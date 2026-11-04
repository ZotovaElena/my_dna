"""Annotate genotypes with ClinVar (GRCh37) by position + allele.

Low-pass/imputed and array calls of rare variants are often false positives:
treat every pathogenic hit as a hypothesis to confirm with a clinical test.
"""

import pandas as pd

from common import config_arg, load_config, load_myheritage

INFO_KEYS = ("CLNSIG", "CLNDN", "CLNREVSTAT", "GENEINFO")
STARS = {
    "practice_guideline": 4,
    "reviewed_by_expert_panel": 3,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "criteria_provided,_single_submitter": 1,
    "criteria_provided,_conflicting_classifications": 1,
}
PATHOGENIC = r"^(Pathogenic|Likely_pathogenic|Pathogenic/Likely_pathogenic)"


def parse_info(info: str) -> dict[str, str]:
    fields = dict(kv.split("=", 1) for kv in info.split(";") if "=" in kv)
    return {k: fields.get(k, "") for k in INFO_KEYS}


def load_clinvar_at(path, sites: pd.DataFrame, chunksize: int = 500_000) -> pd.DataFrame:
    """Stream ClinVar VCF and keep SNVs at the sample's positions."""
    cols = ["chrom", "pos", "clinvar_id", "ref", "alt", "info"]
    keep = []
    for chunk in pd.read_csv(path, sep="\t", comment="#", header=None, usecols=[0, 1, 2, 3, 4, 7],
                             names=cols, dtype={"chrom": str, "pos": int}, chunksize=chunksize):
        chunk = chunk[(chunk["ref"].str.len() == 1) & (chunk["alt"].str.len() == 1)]
        keep.append(chunk.merge(sites, on=["chrom", "pos"]))
    return pd.concat(keep, ignore_index=True)


def main() -> None:
    args = config_arg(__doc__).parse_args()
    cfg = load_config(args.config)

    sample = load_myheritage(cfg["raw_file"])
    hits = load_clinvar_at(cfg["ref_dir"] / "clinvar.vcf.gz", sample[["chrom", "pos", "rsid", "genotype"]])
    hits["alt_copies"] = [g.count(a) for g, a in zip(hits["genotype"], hits["alt"])]
    hits["allele_mismatch"] = [not set(g) <= {r, a} for g, r, a in zip(hits["genotype"], hits["ref"], hits["alt"])]
    carriers = hits[(hits["alt_copies"] > 0) & ~hits["allele_mismatch"]].copy()

    info = pd.DataFrame([parse_info(i) for i in carriers["info"]], index=carriers.index)
    carriers = carriers.drop(columns="info").join(info)
    carriers["stars"] = carriers["CLNREVSTAT"].map(STARS).fillna(0).astype(int)
    carriers["gene"] = carriers["GENEINFO"].str.split(":").str[0]
    carriers.sort_values(["stars", "chrom", "pos"], ascending=[False, True, True]).to_csv(
        cfg["out_dir"] / "clinvar_carriers.csv", index=False)

    print(f"ClinVar SNVs at sample sites: {len(hits)} | carrying ALT: {len(carriers)} "
          f"| allele mismatches: {hits['allele_mismatch'].sum()}")
    print(carriers["CLNSIG"].value_counts().head(12).to_string())

    show = ["rsid", "gene", "genotype", "alt", "alt_copies", "CLNSIG", "stars", "CLNDN"]
    for label, mask in [
        ("Pathogenic / likely pathogenic", carriers["CLNSIG"].str.match(PATHOGENIC)),
        ("Drug response", carriers["CLNSIG"].str.contains("drug_response")),
        ("Risk factor", carriers["CLNSIG"].str.contains("risk_factor")),
    ]:
        sub = carriers[mask].sort_values("stars", ascending=False)
        print(f"\n== {label}: {len(sub)} ==")
        with pd.option_context("display.max_colwidth", 50, "display.width", 220):
            print(sub[show].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
