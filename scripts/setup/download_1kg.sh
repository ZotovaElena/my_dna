#!/usr/bin/env bash
# Download 1000 Genomes phase 3 (GRCh37): sample panel + autosomal VCFs, fetching only missing or corrupt files.
#
# Usage:
#   bash scripts/setup/download_1kg.sh              # check ref_dir/1kg with bgzip -t, fetch every missing/BAD chr
#   bash scripts/setup/download_1kg.sh 4 10 12      # fetch only these chromosomes (skips the check)
#
# Downloads from EBI with aria2c (parallel connections, resumable) into a staging
# directory on the Linux filesystem, verifies each file, then moves it into ref_dir/1kg and verifies again.
# Override defaults with env vars: STAGE=~/somewhere CONN=16 JOBS=4 bash scripts/setup/download_1kg.sh
source "$(dirname "$0")/../env.sh"

DEST="$REF/1kg"
STAGE="${STAGE:-$HOME/1kg_staging}"
CONN="${CONN:-8}"   # connections per file
JOBS="${JOBS:-4}"   # files downloaded at the same time
# KG_BASE and kg_vcf() come from env.sh. EBI supports HTTP range requests, so aria2c's parallel
# connections speed it up (AWS S3 has only the older v5a release, NCBI has neither).
ok() { [[ -s "$1" ]] && bgzip -t "$1" 2>/dev/null; }

mkdir -p "$DEST" "$STAGE"

[[ -s "$PANEL" ]] || wget -c -O "$PANEL" "$KG_BASE/$(basename "$PANEL")"

if (($#)); then
  chroms=("$@")
else
  echo "Scanning $DEST (reads ~15 GB, takes a while)..."
  chroms=()
  for c in {1..22}; do
    if ok "$DEST/$(kg_vcf "$c")"; then echo "  OK  chr$c"; else echo "  BAD chr$c"; chroms+=("$c"); fi
  done
fi
((${#chroms[@]})) || { echo "Nothing to download, all chromosomes OK."; exit 0; }
echo "To download: chr${chroms[*]}"

# Skip the staging copy if it is already complete and valid (e.g. a rerun after an interruption).
urls="$STAGE/urls.txt"
: > "$urls"
for c in "${chroms[@]}"; do
  f=$(kg_vcf "$c")
  if ok "$STAGE/$f" && [[ ! -e "$STAGE/$f.aria2" ]]; then
    echo "  chr$c already in staging and valid, skipping download"
  else
    rm -f "$STAGE/$f"*  # a corrupt staged file would otherwise be resumed as-is
    printf '%s\n  out=%s\n%s\n  out=%s\n' "$KG_BASE/$f" "$f" "$KG_BASE/$f.tbi" "$f.tbi" >> "$urls"
  fi
done

if [[ -s "$urls" ]]; then
  if command -v aria2c >/dev/null; then
    aria2c -i "$urls" -d "$STAGE" -x "$CONN" -s "$CONN" -j "$JOBS" -c \
      --max-tries=10 --retry-wait=10 --auto-file-renaming=false \
      --summary-interval=60 --console-log-level=warn
  else
    echo "aria2c not found (sudo apt install aria2), falling back to slower single-connection wget"
    grep -v '^  out=' "$urls" | wget -c -P "$STAGE" -i -
  fi
fi

failed=()
for c in "${chroms[@]}"; do
  f=$(kg_vcf "$c")
  if ! ok "$STAGE/$f"; then echo "BAD after download: chr$c (rerun the script to retry)"; failed+=("$c"); continue; fi
  echo "chr$c verified, moving to $DEST"
  mv -f "$STAGE/$f" "$STAGE/$f.tbi" "$DEST/"
  touch "$DEST/$f.tbi"  # index must be newer than data, avoids htslib warning
  ok "$DEST/$f" || { echo "BAD after move to $DEST: chr$c, the destination drive is corrupting files"; failed+=("$c"); }
done

if ((${#failed[@]})); then
  echo "Failed: chr${failed[*]}"
  exit 1
fi
echo "All done: chr${chroms[*]} OK in $DEST"
