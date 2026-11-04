import pandas as pd
from snps import SNPs

df = pd.read_csv(
    "MyHeritage_raw_dna_data.csv",
    comment="#",
    dtype={"RSID": str, "CHROMOSOME": str, "POSITION": int, "RESULT": str},
)
df = df[df["RESULT"] != "--"]          # drop no-calls
print(df.shape, df["CHROMOSOME"].value_counts().head())

# call rate and heterozygosity
het = df["RESULT"].str[0] != df["RESULT"].str[1]
print(f"Heterozygosity: {het.mean():.3f}")

snps_of_interest = {
    "rs4988235": "Lactase persistence (LCT/MCM6)",
    "rs12913832": "Blue/brown eye colour (HERC2/OCA2)",
    "rs1815739": "ACTN3 'sprinter gene' (R577X)",
    "rs762551": "Caffeine metabolism (CYP1A2)",
    "rs713598": "Bitter taste perception (TAS2R38)",
    "rs17822931": "Earwax type / body odour (ABCC11)",
    "rs671": "Alcohol flush (ALDH2)",
}
lookup = df.set_index("RSID").loc[lambda d: d.index.isin(snps_of_interest)]
lookup["trait"] = lookup.index.map(snps_of_interest)
print(lookup[["RESULT", "trait"]])

# s = SNPs("MyHeritage_raw_dna_data.csv")
# print(s.source, s.build, s.count)
# print(s.determine_sex())   # inferred from X/Y calls

# s.remap(38)                # lift over GRCh37 → GRCh38 if needed
# s.to_vcf()                 # export VCF for bioinformatics tools