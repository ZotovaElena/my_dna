#!/usr/bin/env bash
# Download the AADR v66.p1 Human Origins dataset (~4.1 GB) into ref_dir/aadr and verify MD5 checksums.
#
#   bash scripts/download_aadr.sh           # HO panel (~600k SNPs; default, matches config aadr.prefix)
#   bash scripts/download_aadr.sh compat    # "compatibility" HO panel: same samples, about half the SNPs
#
# Small files (.ind/.snp/.anno, README) come first, so `python scripts/12_aadr_prepare.py find/check`
# works while the large .geno is still downloading. Valid files are skipped on rerun; a file failing its
# checksum is deleted so the next run fetches it cleanly instead of resuming a corrupt copy.
# Source: Allen Ancient DNA Resource (Reich lab), Harvard Dataverse doi:10.7910/DVN/FFIDCW, CC0.
source "$(dirname "$0")/env.sh"

DEST="$REF/aadr"
API=https://dataverse.harvard.edu/api/access/datafile
mkdir -p "$DEST"

# dataverse-id  md5  filename   (md5 "-" = not published, size check only)
case "${1:-ho}" in
  ho) FILES="
13994526 a14985ffae6dd33a00f4fd5bfefe58c5 v66.p1_HO.aadr.patch.PUB.ind
13994527 fbd2cfe09ec6bad794f7607768b57a3a v66.p1_HO.aadr.patch.PUB.snp
13994528 548b011b4adbcd46f0ff66b957c59c3f v66.p1_HO.aadr.PUB.anno
13994530 - aadr_v66.p1__README.docx
13994808 9f9f0629f803835985bc38856e989995 v66.p1_HO.aadr.patch.PUB.geno" ;;
  compat) FILES="
13994523 a14985ffae6dd33a00f4fd5bfefe58c5 v66.p1_compatibility_HO.aadr.patch.PUB.ind
13994524 900b317b81b249593b20b2fe15dcbf70 v66.p1_compatibility_HO.aadr.patch.PUB.snp
13994525 548b011b4adbcd46f0ff66b957c59c3f v66.p1_compatibility_HO.aadr.PUB.anno
13994530 - aadr_v66.p1__README.docx
13994522 880194e511ec97848d72c4778c307d9f v66.p1_compatibility_HO.aadr.patch.PUB.geno" ;;
  *) echo "usage: $0 [ho|compat]"; exit 2 ;;
esac

ok() {  # file exists and matches its md5 (or is non-empty when no md5 is published)
  [[ -s "$1" ]] || return 1
  [[ "$2" == - ]] || [[ "$(md5sum < "$1" | cut -d' ' -f1)" == "$2" ]]
}

failed=0
while read -r id md5 name; do
  [[ -n "$id" ]] || continue
  f="$DEST/$name"
  if ok "$f" "$md5"; then echo "OK   $name"; continue; fi
  echo "GET  $name"
  if command -v aria2c >/dev/null; then
    aria2c -x 8 -s 8 -c --auto-file-renaming=false --max-tries=10 --retry-wait=10 \
      --summary-interval=60 --console-log-level=warn -d "$DEST" -o "$name" "$API/$id" || true
  else
    wget -c -O "$f" "$API/$id" || true
  fi
  if ok "$f" "$md5"; then echo "OK   $name"; else echo "BAD  $name (checksum mismatch or incomplete; deleted, rerun)"; rm -f "$f" "$f.aria2"; failed=1; fi
done <<< "$FILES"

((failed == 0)) || exit 1
echo "AADR ready in $DEST"
