#!/usr/bin/env bash
set -euo pipefail
cd /home/sophgo13/cjl
DATA_ROOT=/home/sophgo13/cjl/storage/parameter-importance
REPO=/home/sophgo13/cjl/parameter-importance
ARCHIVE="$DATA_ROOT/tmp/linux-runtime-wheels.tar"
STAGE="$DATA_ROOT/tmp/linux-runtime-extract"
WHEELHOUSE="$DATA_ROOT/wheelhouse"
MANIFEST="$STAGE/wheelhouse-sha256.tsv"

[[ -f $ARCHIVE ]] || { echo "ERROR: missing $ARCHIVE" >&2; exit 2; }
[[ -d $WHEELHOUSE ]] || { echo "ERROR: missing base wheelhouse" >&2; exit 2; }
rm -rf -- "$STAGE"
mkdir -p "$STAGE" "$DATA_ROOT/manifests" "$DATA_ROOT/envs"
tar -xf "$ARCHIVE" -C "$STAGE"

runtime_count=$(find "$STAGE" -maxdepth 1 -type f -name '*.whl' | wc -l)
[[ $runtime_count == 18 ]] || { echo "ERROR: expected 18 runtime wheels, found $runtime_count" >&2; exit 3; }

while IFS=$'\t' read -r expected size name; do
  name=${name%$'\r'}
  if [[ -f $STAGE/$name ]]; then
    file="$STAGE/$name"
  else
    file="$WHEELHOUSE/$name"
  fi
  [[ -f $file && $(stat -c '%s' "$file") == "$size" ]] || { echo "ERROR: wheel size mismatch: $name" >&2; exit 3; }
  actual=$(sha256sum "$file" | awk '{print $1}')
  [[ $actual == "$expected" ]] || { echo "ERROR: wheel hash mismatch: $name" >&2; exit 3; }
done < "$MANIFEST"

find "$STAGE" -maxdepth 1 -type f -name '*.whl' -exec cp -f -- {} "$WHEELHOUSE/" \;
[[ $(find "$WHEELHOUSE" -maxdepth 1 -type f -name '*.whl' | wc -l) == 88 ]] || {
  echo 'ERROR: merged wheelhouse does not contain 88 wheels' >&2
  exit 3
}
cp "$MANIFEST" "$DATA_ROOT/manifests/wheelhouse-sha256.tsv"

python3 -m venv --clear "$DATA_ROOT/envs/parameter-importance"
PY="$DATA_ROOT/envs/parameter-importance/bin/python"
"$PY" -m pip install --no-index --find-links "$WHEELHOUSE" --requirement "$REPO/environment/requirements.lock"
"$PY" -m pip check
"$PY" -m pip freeze > "$DATA_ROOT/manifests/pip-freeze.txt"
rm -rf -- "$STAGE"
rm -f -- "$ARCHIVE" "$DATA_ROOT/tmp/wheelhouse.tar"
echo "OK: complete offline venv ready at $DATA_ROOT/envs/parameter-importance"
