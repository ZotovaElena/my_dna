#!/usr/bin/env bash
# Runs of homozygosity (plink 1.9). F_ROH ≈ 1/16 for first-cousin parents, ≈ 1/64 for second cousins.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$(python scripts/common.py out_dir)

plink --vcf "$OUT/sample.grch37.vcf.gz" --double-id --chr 1-22 \
  --homozyg --homozyg-kb 1000 --homozyg-snp 100 --out "$OUT/roh"

# KB column of .hom.indiv; 2,881,033 kb = GRCh37 autosomal length
awk 'NR==2 {printf "ROH segments: %d | total: %.1f Mb | F_ROH: %.4f\n", $4, $5/1000, $5/2881033}' "$OUT/roh.hom.indiv"
