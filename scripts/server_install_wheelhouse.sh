#!/usr/bin/env bash
set -euo pipefail
cd /home/sophgo13/cjl
DATA_ROOT=/home/sophgo13/cjl/storage/parameter-importance
ARCHIVE="$DATA_ROOT/tmp/wheelhouse.tar"
STAGE="$DATA_ROOT/tmp/wheelhouse-extract"

[[ -f $ARCHIVE ]] || { echo "ERROR: missing $ARCHIVE" >&2; exit 2; }
rm -rf -- "$STAGE"
mkdir -p "$STAGE" "$DATA_ROOT/manifests" "$DATA_ROOT/envs"
tar -xf "$ARCHIVE" -C "$STAGE"

while IFS=$'\t' read -r expected size name; do
  file="$STAGE/wheelhouse/$name"
  [[ -f $file && $(stat -c '%s' "$file") == "$size" ]] || { echo "ERROR: wheel size mismatch: $name" >&2; exit 3; }
  actual=$(sha256sum "$file" | awk '{print $1}')
  [[ $actual == "$expected" ]] || { echo "ERROR: wheel hash mismatch: $name" >&2; exit 3; }
done < "$STAGE/wheelhouse-sha256.tsv"

rm -rf -- "$DATA_ROOT/wheelhouse"
mv "$STAGE/wheelhouse" "$DATA_ROOT/wheelhouse"
cp "$STAGE/wheelhouse-sha256.tsv" "$DATA_ROOT/manifests/"
cp "$STAGE/resolution-report.json" "$DATA_ROOT/manifests/"
cp "$STAGE/requirements.lock" /home/sophgo13/cjl/parameter-importance/environment/requirements.lock

python3 -m venv "$DATA_ROOT/envs/parameter-importance"
PY="$DATA_ROOT/envs/parameter-importance/bin/python"
"$PY" -m pip install --no-index --find-links "$DATA_ROOT/wheelhouse" \
  --requirement /home/sophgo13/cjl/parameter-importance/environment/requirements.lock
"$PY" -m pip check
"$PY" -m pip freeze > "$DATA_ROOT/manifests/pip-freeze.txt"
rm -rf -- "$STAGE"
rm -f -- "$ARCHIVE"
echo "OK: offline venv ready at $DATA_ROOT/envs/parameter-importance"
