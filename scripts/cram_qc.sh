#!/usr/bin/env bash
# CRAM alignment QC for the exploratory notebook, written to out_dir:
#   idxstats.tsv          per-contig mapped/unmapped read counts
#   coverage.tsv          per-contig breadth and mean depth (samtools coverage)
#   chr1.depth_hist.tsv   depth → number of chr1 bases (a histogram, not 249M per-base rows)
# Builds the FASTA and CRAM indexes first if missing. Run with the my_analysis env active.
source "$(dirname "$0")/env.sh"

[ -f "$FASTA.fai" ] || samtools faidx "$FASTA"
[ -f "$CRAM.crai" ] || samtools index "$CRAM"

echo "idxstats..."
samtools idxstats "$CRAM" > "$OUT/idxstats.tsv.tmp"
mv "$OUT/idxstats.tsv.tmp" "$OUT/idxstats.tsv"

echo "coverage (reads the whole CRAM, takes several minutes)..."
samtools coverage --reference "$FASTA" "$CRAM" > "$OUT/coverage.tsv.tmp"
mv "$OUT/coverage.tsv.tmp" "$OUT/coverage.tsv"

echo "chr1 depth histogram (all positions, including zero depth)..."
samtools depth -a -r 1 --reference "$FASTA" "$CRAM" \
  | awk -F'\t' '{h[$3]++} END {for (d in h) print d "\t" h[d]}' \
  | sort -n > "$OUT/chr1.depth_hist.tsv.tmp"
mv "$OUT/chr1.depth_hist.tsv.tmp" "$OUT/chr1.depth_hist.tsv"

echo "Done: $OUT/{idxstats,coverage,chr1.depth_hist}.tsv"
