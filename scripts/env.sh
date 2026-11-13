# Sourced by the pipeline's shell scripts: strict mode, project root as cwd, config values, shared names.
#   source "$(dirname "$0")/env.sh"
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# OUT REF THREADS SAMPLE REFS CRAM FASTA ANCIENT_DIR PANEL (see SHELL_VARS in common.py).
# Assigned first so a config error stops the script; `eval "$(...)"` would hide it.
_threads=${THREADS:-}   # THREADS=N on the command line overrides config threads
_cfg=$(python scripts/common.py --shell)
eval "$_cfg"
unset _cfg
THREADS=${_threads:-$THREADS}
unset _threads

# 1000 Genomes phase 3 (GRCh37). EBI is the only mirror with the v5b release.
KG_BASE=https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502
kg_vcf() { echo "ALL.chr$1.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"; }
