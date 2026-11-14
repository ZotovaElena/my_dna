#!/usr/bin/env bash
# ancIBD (Ringbauer et al. 2024) in its own conda env "ancibd", plus its 1240k support files (marker lists per
# chromosome, the v51.1 1240k .snp with the genetic map, allele frequencies) into ref_dir/ancibd_data/.
# The support files are the authors' Dropbox folder linked from https://ancibd.readthedocs.io
source "$(dirname "$0")/../env.sh"

# ancIBD builds a C extension with pip, so the env gets conda-forge's C compiler (WSL has no gcc by default)
if ! conda run -n ancibd ancIBD-run -h >/dev/null 2>&1; then
  conda env list | grep -q '^ancibd ' || conda create -y -n ancibd -c conda-forge --override-channels python=3.11 pip
  conda install -y -n ancibd -c conda-forge --override-channels c-compiler
  conda run -n ancibd pip install --no-cache-dir ancIBD matplotlib "pandas<3"   # matplotlib: undeclared dependency; pandas 3 postdates ancIBD 0.8
fi
conda run -n ancibd ancIBD-run -h | head -3

dir="$REF/ancibd_data"
mkdir -p "$dir"
zip="$dir/ancibd_data.zip"
if [[ ! -e "$dir/.unpacked" ]]; then
  if [[ ! -s "$zip" ]]; then
    curl -fL --retry 3 -o "$zip.part" "https://www.dropbox.com/sh/q18yyrffbdj1yv1/AAC1apifYB_oKB8SNrmQQ-26a?dl=1"
    mv "$zip.part" "$zip"
  fi
  # Python instead of unzip (not installed on WSL by default); skips the "/" entry Dropbox folder zips have
  # and anything that would land outside $dir. The zip is deleted only after a clean unpack.
  python -I - "$zip" "$dir" <<'EOF2' && touch "$dir/.unpacked" && rm -f "$zip"
import os, sys, zipfile
src, dest = sys.argv[1], os.path.realpath(sys.argv[2])
with zipfile.ZipFile(src) as z:
    for m in z.infolist():
        target = os.path.realpath(os.path.join(dest, m.filename))
        if m.filename.strip("/") and target.startswith(dest + os.sep):
            z.extract(m, dest)
    print(f"unpacked {len(z.infolist())} entries into {dest}")
EOF2
fi
find "$dir" -maxdepth 3 \( -name "snps_bcftools_ch1.csv" -o -name "*1240k.snp" -o -name "*AF_ch1.tsv" \) -print
