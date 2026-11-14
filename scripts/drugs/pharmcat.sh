#!/usr/bin/env bash
# Pharmacogenomics with PharmCAT: gene diplotypes and phenotypes -> CPIC / DPWG drug recommendations.
# Needs scripts/setup/download_pharmcat.sh and Java 17+ (conda env 'java' or on PATH) (no Docker, no liftover: positions are matched by rsID).
# CYP2D6 and HLA are not callable from SNP data; PharmCAT reports them as "no call".
source "$(dirname "$0")/../env.sh"

jar=$(ls "$REF"/pharmcat/pharmcat*.jar 2>/dev/null | sort -V | tail -1)
[[ -n "$jar" ]] || { echo "No PharmCAT jar in $REF/pharmcat: run bash scripts/setup/download_pharmcat.sh" >&2; exit 1; }
# Java comes from its own conda env ("java"), so it can't clash with the packages in my_analysis
java=$(command -v java || true)
base=$(conda info --base 2>/dev/null || true)
[[ -z "$java" && -n "$base" && -x "$base/envs/java/bin/java" ]] && java="$base/envs/java/bin/java"
[[ -n "$java" ]] || { echo 'No java: conda create -n java -c conda-forge --override-channels openjdk=21' >&2; exit 1; }
"$java" -version 2>&1 | head -1

python scripts/genome/make_vcf.py pgx
dir="$OUT/pharmcat"
rm -f "$dir"/sample.{match,phenotype,report}.* 2>/dev/null || true
# The JSON report feeds report/build_report.py; the HTML is PharmCAT's own full report. Older releases lack some
# output flags, so fall back to fewer of them.
ok=
for flags in "-reporterJson -reporterHtml" "-reporterJson" ""; do
  # shellcheck disable=SC2086
  if "$java" -jar "$jar" -vcf "$dir/sample.pgx.grch38.vcf" -o "$dir" -bf sample $flags >"$dir/pharmcat.log" 2>&1; then
    ok=1; break
  fi
  echo "PharmCAT with flags '$flags' failed (see $dir/pharmcat.log), retrying with fewer" >&2
done
cat "$dir/pharmcat.log"
[[ -n "$ok" ]] || { echo "PharmCAT failed, see $dir/pharmcat.log" >&2; exit 1; }
ls -l "$dir"
echo "Report: $dir/sample.report.html (also in outputs/report/ after python scripts/report/build_report.py)"
