#!/usr/bin/env bash
# Ancient BAMs from ENA for the ancIBD step (individuals chosen by `python scripts/matches/ancibd.py select`).
#
#   bash scripts/setup/download_ancient_bams.sh manifest   # ENA file lists of the projects (a few MB), then matches them
#   bash scripts/setup/download_ancient_bams.sh fetch      # the BAMs, one at a time, kept as CRAM in ref_dir/ancient/bam/
#
# fetch checks free space per file, keeps chromosomes 1-22 as CRAM against the b37 reference (about half
# the size), merges several files of one individual into <iid>.cram, and
# writes outputs/ancibd/glimpse_samples.tsv for scripts/genome/glimpse.sh.
source "$(dirname "$0")/../env.sh"

work="$OUT/ancibd"
dest="${ANCIENT_DIR:-$REF/ancient}"   # ancient_dir in config.yaml (exported by env.sh); default ref_dir/ancient
mkdir -p "$dest/ena" "$dest/bam"
[[ -s "$work/ancients.tsv" ]] || { echo "Run python scripts/matches/ancibd.py select first" >&2; exit 1; }

# ENA closes slow connections mid-file (curl error 18) and sometimes stalls. Every attempt resumes where the
# last one stopped (-C -); an attempt slower than 20 kB/s for 2 minutes is dropped and restarted.
# ENA_PROTO=ftp uses ENA's FTP server instead of HTTPS (often faster from home connections).
fetch_resume() {
  local f=$1 url=$2 bytes=$3 try have
  local src="${ENA_PROTO:-https}://${url#*://}"
  [[ "$url" == *://* ]] || src="${ENA_PROTO:-https}://$url"
  for try in $(seq 1 50); do
    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    (( have >= bytes )) && return 0
    echo "  attempt $try: $((have / 1000000)) of $((bytes / 1000000)) MB, $(date +%H:%M)"
    curl -fsSL -C - --connect-timeout 60 --speed-limit 20000 --speed-time 120 -o "$f" "$src" && return 0
    sleep $(( try < 10 ? 10 * try : 120 ))
  done
  echo "Giving up on $f after 50 attempts" >&2; return 1
}

case "${1:-}" in
manifest)
  fields=run_accession,sample_alias,sample_title,submitted_ftp,submitted_bytes,submitted_format,fastq_ftp,fastq_bytes
  projects=$(python -c "import pandas as pd; print(' '.join(sorted({p for x in pd.read_csv('$work/ancients.tsv', sep='\t')['projects'] for p in str(x).split()})))")
  for p in $projects; do
    echo "ENA file list: $p"
    curl -fsSL --retry 3 -o "$dest/ena/$p.tsv.tmp" \
      "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$p&result=read_run&fields=$fields&format=tsv&limit=0" \
      && mv "$dest/ena/$p.tsv.tmp" "$dest/ena/$p.tsv"
  done
  python scripts/matches/ancibd.py match
  ;;
fetch)
  man="$work/manifest.tsv"
  [[ -s "$man" ]] || { echo "Run: bash scripts/setup/download_ancient_bams.sh manifest" >&2; exit 1; }
  total=$(awk -F'\t' 'NR > 1 {s += $3} END {printf "%.0f", s / 1e9}' "$man")
  echo "BAMs: ~${total} GB to download; kept as CRAM of chromosomes 1-22 (~40-50%), one file at a time"
  regions=$(seq -s ' ' 1 22)
  : > "$work/glimpse_samples.tsv.tmp"
  for iid in $(awk -F'\t' 'NR > 1 {print $1}' "$man" | sort -u); do
    final="$dest/bam/$iid.cram"
    if [[ ! -s "$final.crai" ]]; then
      mkdir -p "$dest/bam/$iid"
      parts=()
      while IFS=$'\t' read -r id url bytes; do
        [[ "$id" == "$iid" ]] || continue
        f="$dest/bam/$iid/$(basename "$url")"
        free=$(df -BG --output=avail "$dest" | tail -1 | tr -dc 0-9)
        (( free * 1000000000 > bytes + 10000000000 )) \
          || { echo "Not enough space for $(basename "$url") ($((bytes / 1000000000)) GB, ${free} GB free)" >&2; exit 1; }
        echo "$iid: $(basename "$url") ($((bytes / 1000000)) MB)"
        fetch_resume "$f" "$url" "$bytes"
        samtools quickcheck "$f" || { echo "$f is truncated or not a BAM" >&2; exit 1; }
        # Same build and contig names as the b37 reference? (chromosome 1 = "1", 249,250,621 bp)
        samtools view -H "$f" | grep -qP '^@SQ\tSN:1\tLN:249250621' \
          || { echo "$f: not GRCh37 with 1,2,.. contig names; first contigs:" >&2
               samtools view -H "$f" | grep '^@SQ' | head -3 >&2; exit 1; }
        samtools index -@ "$THREADS" "$f"
        samtools view -C -T "$FASTA" -@ "$THREADS" -o "$f.cram" "$f" $regions
        rm -f "$f" "$f.bai"
        parts+=("$f.cram")
      done < <(tail -n +2 "$man")
      if (( ${#parts[@]} > 1 )); then
        samtools merge -f -@ "$THREADS" --reference "$FASTA" -O cram -o "$final.tmp" "${parts[@]}"
      else
        mv "${parts[0]}" "$final.tmp"
      fi
      mv "$final.tmp" "$final"; samtools index "$final"; rm -rf "$dest/bam/$iid"
    fi
    echo "$iid: ready ($(du -h "$final" | cut -f1))"
    printf '%s\t%s\n' "$final" "$iid" >> "$work/glimpse_samples.tsv.tmp"
  done
  mv "$work/glimpse_samples.tsv.tmp" "$work/glimpse_samples.tsv"
  echo "Next: bash scripts/genome/glimpse.sh $work/glimpse_samples.tsv ancient"
  ;;
*)
  echo "usage: $0 manifest|fetch" >&2; exit 1 ;;
esac
