#!/usr/bin/env bash
# Pharmacogenomics with PharmCAT (needs GRCh38). CYP2D6 is not callable from SNP data.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$(python scripts/common.py out_dir)
IMAGE=$(python scripts/common.py pharmcat_image)

python scripts/04_make_vcf.py grch38
docker run --rm -v "$OUT":/pharmcat/data "$IMAGE" \
  pharmcat_pipeline data/sample.grch38.vcf -o data/pharmcat

echo "Report: $OUT/pharmcat/"
