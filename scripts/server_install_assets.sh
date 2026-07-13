#!/usr/bin/env bash
set -euo pipefail
cd /home/sophgo13/cjl
DATA_ROOT=/home/sophgo13/cjl/storage/parameter-importance
ARCHIVE="$DATA_ROOT/tmp/minimum-loop-assets.tar"
STAGE="$DATA_ROOT/tmp/minimum-loop-assets"
[[ -f $ARCHIVE ]] || { echo "ERROR: missing $ARCHIVE" >&2; exit 2; }
rm -rf -- "$STAGE"
mkdir -p "$STAGE"
tar -xf "$ARCHIVE" -C "$STAGE"

while IFS=$'\t' read -r expected size relative; do
  file="$STAGE/$relative"
  [[ -f $file && $(stat -c '%s' "$file") == "$size" ]] || { echo "ERROR: asset size mismatch: $relative" >&2; exit 3; }
  actual=$(sha256sum "$file" | awk '{print $1}')
  [[ $actual == "$expected" ]] || { echo "ERROR: asset hash mismatch: $relative" >&2; exit 3; }
done < "$STAGE/manifests/asset-sha256.tsv"

for dir in models datasets source; do
  mkdir -p "$DATA_ROOT/$dir"
  cp -a "$STAGE/$dir/." "$DATA_ROOT/$dir/"
done
for archive in "$DATA_ROOT"/source/pythia-*.tar.gz; do
  [[ -e $archive ]] || continue
  top=$(basename "$archive" .tar.gz)
  [[ -d "$DATA_ROOT/source/$top" ]] || tar -xzf "$archive" -C "$DATA_ROOT/source"
done
cp -a "$STAGE/manifests/." "$DATA_ROOT/manifests/"
rm -rf -- "$STAGE"
rm -f -- "$ARCHIVE"
echo "OK: minimum-loop assets installed"
