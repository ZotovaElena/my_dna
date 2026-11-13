#!/usr/bin/env bash
# PCA with 1000G and supervised ADMIXTURE on LD-pruned common SNPs.
source "$(dirname "$0")/env.sh"
K=$(wc -w <<< "$REFS")

plink2 --pfile "$OUT/merged" --maf 0.05 --indep-pairwise 200 50 0.2 --threads "$THREADS" --out "$OUT/prune"
plink2 --pfile "$OUT/merged" --extract "$OUT/prune.prune.in" --pca 10 --threads "$THREADS" --out "$OUT/ancestry"
plink2 --pfile "$OUT/merged" --extract "$OUT/prune.prune.in" --make-bed --out "$OUT/ancestry"

# .pop: superpop label for reference samples in $REFS, '-' (estimate) for everyone else
awk -v refs="$REFS" 'BEGIN {n = split(refs, r, " "); for (i = 1; i <= n; i++) ok[r[i]] = 1}
  NR == FNR {if (FNR > 1) sp[$1] = $3; next}
  {label = ($2 in sp && sp[$2] in ok) ? sp[$2] : "-"; print label}' \
  "$PANEL" "$OUT/ancestry.fam" > "$OUT/ancestry.pop"

(cd "$OUT" && admixture --supervised -j"$THREADS" ancestry.bed "$K" | tee admixture.log)

python scripts/08_plot_ancestry.py
