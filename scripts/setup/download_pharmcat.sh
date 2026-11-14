#!/usr/bin/env bash
# PharmCAT (latest GitHub release): the Java program plus its list of GRCh38 positions, into ref_dir/pharmcat/.
# About 50 MB. Needs Java 17+ to run: `conda create -n java -c conda-forge --override-channels openjdk=21`.
source "$(dirname "$0")/../env.sh"

dir="$REF/pharmcat"
mkdir -p "$dir"
api=https://api.github.com/repos/PharmGKB/PharmCAT/releases/latest
urls=$(curl -fsSL "$api" | python -c '
import json, sys
rel = json.load(sys.stdin)
print("release", rel["tag_name"], file=sys.stderr)
for a in rel["assets"]:
    n = a["name"]
    if n.endswith("-all.jar") or (n.startswith("pharmcat_positions") and "uniallelic" not in n):
        print(a["browser_download_url"])')
[[ -n "$urls" ]] || { echo "No PharmCAT assets found at $api" >&2; exit 1; }
for u in $urls; do
  echo "Downloading $(basename "$u")"
  curl -fL --retry 3 -C - -o "$dir/$(basename "$u")" "$u"
done
ls -l "$dir"
