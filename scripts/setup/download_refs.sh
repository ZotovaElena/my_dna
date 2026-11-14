#!/usr/bin/env bash
# Download GRCh37 references: 1000 Genomes phase 3 panel + autosomes (~15 GB) and ClinVar (~150 MB).
# Safe to rerun: valid files are kept, missing or corrupt ones are fetched again.
source "$(dirname "$0")/../env.sh"

bash scripts/setup/download_1kg.sh

CV=https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh37
wget -c -P "$REF" "$CV/clinvar.vcf.gz" "$CV/clinvar.vcf.gz.tbi"
