#!/usr/bin/env bash
# Subset 1000G to the sample's sites, encode the sample on 1000G alleles, merge into one plink2 fileset.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$(python scripts/common.py out_dir)
REF=$(python scripts/common.py ref_dir)
THREADS=$(python scripts/common.py threads)
mkdir -p "$OUT/1kg"

python scripts/04_make_vcf.py sites
for c in {1..22}; do
  bcftools view -R "$OUT/sample_sites.tsv" -m2 -M2 -v snps --threads "$THREADS" \
    -Oz -o "$OUT/1kg/chr$c.vcf.gz" "$REF"/1kg/ALL.chr$c.phase3_*.vcf.gz
done
bcftools concat --threads "$THREADS" -Oz -o "$OUT/1kg_subset.vcf.gz" "$OUT"/1kg/chr{1..22}.vcf.gz
bcftools index -t "$OUT/1kg_subset.vcf.gz"
bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\n' "$OUT/1kg_subset.vcf.gz" > "$OUT/1kg_sites.tsv"

python scripts/04_make_vcf.py grch37
bcftools view -Oz -o "$OUT/sample.grch37.vcf.gz" "$OUT/sample.grch37.vcf"
bcftools index -t "$OUT/sample.grch37.vcf.gz"

bcftools merge --threads "$THREADS" -Oz -o "$OUT/merged.vcf.gz" "$OUT/sample.grch37.vcf.gz" "$OUT/1kg_subset.vcf.gz"
plink2 --vcf "$OUT/merged.vcf.gz" --set-all-var-ids '@:#' --rm-dup exclude-all \
  --geno 0.02 --threads "$THREADS" --make-pgen --out "$OUT/merged"
