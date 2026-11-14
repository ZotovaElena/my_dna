#!/usr/bin/env bash
# GLIMPSE2 imputation + phasing of low-coverage genomes against 1000 Genomes phase 3 (GRCh37, local files).
# Used for your CRAM and for the ancient BAMs of the ancIBD step (matches/ancibd.py).
#
#   bash scripts/genome/glimpse.sh SAMPLES.tsv RUN_NAME [chrom ...]     (default: chromosomes 1-22)
#
# SAMPLES.tsv: "<BAM/CRAM path><TAB><sample name>" per line, all imputed together.
# The genetic map comes from the AADR .snp file (the GRCh37 map ancIBD uses), so nothing is downloaded.
# Work files go to ref_dir/glimpse/ (D:): panel/ and split/ are shared by all runs, runs/<RUN_NAME>/ holds
# the result chr<N>.bcf (phased GT + GP + DS). Every step is skipped when its output exists, so it resumes.
# With KEEP_SITES=<file of "chrom<TAB>pos"> the result is cut to those sites (e.g. 1240k) to save disk.
# Chromosomes can run in parallel as separate calls, e.g. THREADS=4 ... me 10 13 & THREADS=4 ... me 11 14 &
# With CLEAN_SPLIT=1 each chromosome's binary reference chunks are deleted after use (they total ~35 GB).
source "$(dirname "$0")/../env.sh"

samples=${1:?usage: genome/glimpse.sh SAMPLES.tsv RUN_NAME [chrom ...]}
name=${2:?usage: genome/glimpse.sh SAMPLES.tsv RUN_NAME [chrom ...]}
shift 2
chroms=("${@:-}")
[[ -n "${chroms[0]}" ]] || chroms=($(seq 1 22))

need() { [[ -s "$1" ]] || { echo "Missing or empty: $1 (see the .log next to it)" >&2; exit 1; }; }

G="$REF/glimpse"
run="$G/runs/$name"
mkdir -p "$G/maps" "$G/panel" "$G/split" "$run/chunks"
# Written via a temp file + rename, so parallel runs of the same RUN_NAME (different chromosomes) never
# see a half-written list
awk -F'\t' 'NF >= 2 {print $1" "$2}' "$samples" > "$run/bams.txt.$$" && mv "$run/bams.txt.$$" "$run/bams.txt"
cp "$samples" "$run/samples.tsv.$$" && mv "$run/samples.tsv.$$" "$run/samples.tsv"
echo "Run $name: $(wc -l < "$run/bams.txt") genome(s), chromosomes ${chroms[*]}"

# Genetic maps (pos chr cM) from the AADR .snp (Morgans, column 3)
if [[ ! -s "$G/maps/chr22.gmap" ]]; then
  python - "$REF" "$G/maps" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from common import load_config
cfg = load_config()
snp = Path(f"{cfg['ref_dir'] / cfg['aadr']['prefix']}.snp")
out = Path(sys.argv[2])
rows = {}
with open(snp) as f:
    for line in f:
        _, c, m, p, *_ = line.split()
        if c.isdigit() and 1 <= int(c) <= 22:
            rows.setdefault(c, []).append((int(p), float(m) * 100))
for c, r in rows.items():
    r.sort()
    keep, last = [], (-1, 0.0)  # GLIMPSE2 needs cM >= 0 (AADR chr17 starts below 0)
    for p, cm in r:  # strictly increasing position, non-decreasing cM
        if p > last[0] and cm >= last[1]:
            keep.append((p, cm)); last = (p, cm)
    with open(out / f"chr{c}.gmap", "w") as f:
        f.write("pos\tchr\tcM\n" + "".join(f"{p}\t{c}\t{cm:.6f}\n" for p, cm in keep))
print(f"Genetic maps from {snp.name}: {', '.join(f'chr{c} {len(r):,}' for c, r in list(rows.items())[:3])}, ...")
EOF
fi

for c in "${chroms[@]}"; do
  map="$G/maps/chr$c.gmap"
  panel="$G/panel/chr$c.bcf"
  out="$run/chr$c.bcf"
  if [[ -s "$out.csi" ]]; then echo "chr$c: done ($out)"; continue; fi   # before the panel: it may be cleaned up
  # 1. Reference panel: biallelic SNPs of 1000G phase 3
  if [[ ! -s "$G/split/chunks.chr$c.txt" ]]; then
    if [[ ! -s "$panel.csi" ]]; then
      echo "chr$c: reference panel"
      bcftools norm -m -any "$REF/1kg/$(kg_vcf "$c")" -Ou --threads "$THREADS" \
        | bcftools view -m2 -M2 -v snps --threads "$THREADS" -Ob -o "$panel.tmp"
      need "$panel.tmp"; mv "$panel.tmp" "$panel"; bcftools index -f "$panel"
    fi
    # 2. Chunks and binary reference per chunk
    GLIMPSE2_chunk --input "$panel" --region "$c" --map "$map" --sequential \
      --output "$G/split/chunks.chr$c.tmp" --log "$G/split/chunks.chr$c.log" >/dev/null
    while read -r id _ irg org _; do
      GLIMPSE2_split_reference --reference "$panel" --map "$map" --input-region "$irg" --output-region "$org" \
        --output "$G/split/1000G_chr$c" --threads "$THREADS" --log "$G/split/split.chr$c.$id.log" >/dev/null
    done < "$G/split/chunks.chr$c.tmp"
    mv "$G/split/chunks.chr$c.tmp" "$G/split/chunks.chr$c.txt"
    rm -f "$panel" "$panel.csi"   # the binary chunks are all that is needed from here on
  fi
  # 3. Impute + phase every chunk, 4. ligate
  if [[ ! -s "$out.csi" ]]; then
    : > "$run/chunks/list.chr$c.txt"
    # GLIMPSE2_ligate refuses positions covered by three files. Around large gaps (centromeres) a chunk's
    # buffer can reach past the next chunk into the one after it; such a chunk is cut to end before that
    # one starts (ligation only needs neighbouring chunks to overlap).
    mapfile -t cstart < <(awk '{split($3, a, /[:-]/); print a[2]}' "$G/split/chunks.chr$c.txt")
    mapfile -t cend < <(awk '{split($3, a, /[:-]/); print a[3]}' "$G/split/chunks.chr$c.txt")
    k=0
    while read -r id _ irg org _; do
      s=${irg#*:}; bin="$G/split/1000G_chr${c}_${irg%%:*}_${s%-*}_${s#*-}.bin"   # named by the buffered region
      part="$run/chunks/chr${c}_$id.bcf"
      if [[ ! -s "$part.csi" ]]; then
        GLIMPSE2_phase --bam-list "$run/bams.txt" --reference "$bin" --fasta "$FASTA" \
          --threads "$THREADS" --output "$part" --log "$part.log" >/dev/null
        need "$part"; bcftools index -f "$part"
      fi
      lim=${cstart[$((k + 2))]:-}
      if [[ -n "$lim" ]] && (( cend[k] >= lim )); then
        bcftools view -r "$c:1-$((lim - 1))" -Ob -o "$part.trim.bcf" --write-index "$part"
        echo "chr$c: chunk $id cut at $((lim - 1)) (overlapped chunk $((k + 2)))"
        echo "$part.trim.bcf" >> "$run/chunks/list.chr$c.txt"
      else
        echo "$part" >> "$run/chunks/list.chr$c.txt"
      fi
      k=$((k + 1))
    done < "$G/split/chunks.chr$c.txt"
    # GLIMPSE2_ligate has no --log and exits 0 even on errors, so its output is checked explicitly
    GLIMPSE2_ligate --input "$run/chunks/list.chr$c.txt" --output "$out.tmp.bcf" --threads "$THREADS" \
      > "$run/ligate.chr$c.log" 2>&1
    need "$out.tmp.bcf"
    if [[ -n "${KEEP_SITES:-}" ]]; then
      bcftools view -T "$KEEP_SITES" "$out.tmp.bcf" -Ob -o "$out.cut.bcf"; need "$out.cut.bcf"; mv "$out.cut.bcf" "$out.tmp.bcf"
    fi
    mv "$out.tmp.bcf" "$out"; rm -f "$out.tmp.bcf.csi"; bcftools index -f "$out"
    rm -f "$run/chunks/chr${c}_"*.bcf "$run/chunks/chr${c}_"*.bcf.csi
    if [[ "${CLEAN_SPLIT:-0}" == 1 ]]; then   # binary panel is ~35 GB for all chromosomes; rebuilt when needed
      rm -f "$G/split/1000G_chr${c}_"*.bin "$G/split/chunks.chr$c.txt"
    fi
    echo "chr$c: $(bcftools index -n "$out") sites -> $out"
  fi
done
echo "Done: $run/chr*.bcf"
