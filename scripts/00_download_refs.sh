#!/usr/bin/env bash
# Download GRCh37 references: 1000 Genomes phase 3 autosomes (~15 GB) and ClinVar (~150 MB).
set -euo pipefail
cd "$(dirname "$0")/.."
REF=$(python scripts/common.py ref_dir)

KG=https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502
mkdir -p "$REF/1kg"
wget -c -P "$REF/1kg" "$KG/integrated_call_samples_v3.20130502.ALL.panel"
for c in {1..22}; do
  f=ALL.chr$c.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz
  wget -c -P "$REF/1kg" "$KG/$f" "$KG/$f.tbi"
done

CV=https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh37
wget -c -P "$REF" "$CV/clinvar.vcf.gz" "$CV/clinvar.vcf.gz.tbi"
