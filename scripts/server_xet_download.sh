#!/usr/bin/env bash
set -euo pipefail
umask 027

usage() {
  echo "usage: server_xet_download.sh DEST EXPECTED_SIZE EXPECTED_SHA256" >&2
  exit 2
}

[[ $# == 3 ]] || usage
dest=$1
expected_size=$2
expected_sha=${3,,}
[[ $expected_size =~ ^[0-9]+$ ]] || usage
[[ $expected_sha =~ ^[0-9a-f]{64}$ ]] || usage

part="${dest}.part"
meta="${part}.meta"
lock="${part}.lock"
mkdir -p -- "$(dirname -- "$dest")"
command -v flock >/dev/null || { echo "ERROR: flock is required" >&2; exit 2; }
exec 9>"$lock"
flock -w 3900 9 || { echo "ERROR: timed out waiting for the object download lock" >&2; exit 8; }

if [[ -f $dest ]]; then
  final_size=$(stat -c '%s' -- "$dest")
  [[ $final_size == "$expected_size" ]] || { echo "ERROR: completed object has the wrong size" >&2; exit 4; }
  final_sha=$(sha256sum -- "$dest" | awk '{print $1}')
  [[ $final_sha == "$expected_sha" ]] || { echo "ERROR: completed object has the wrong SHA-256" >&2; exit 7; }
  echo "OK: completed object already exists and is verified"
  exit 0
fi

IFS= read -r signed_url
# Windows PowerShell writes CRLF to a native-process stdin pipe.
signed_url=${signed_url%$'\r'}
[[ $signed_url == https://* ]] || { echo "ERROR: stdin did not contain an HTTPS URL" >&2; exit 2; }
[[ $signed_url != *[$'\n\r"\\']* ]] || { echo "ERROR: URL contains an unsafe curl-config character" >&2; exit 2; }

# Never enable xtrace in this script: signed_url must not enter logs or process arguments.
host=${signed_url#https://}
host=${host%%/*}
host=${host%%:*}
[[ $host =~ ^[A-Za-z0-9.-]+$ ]] || { echo "ERROR: invalid URL host" >&2; exit 2; }

resolve() {
  dig +time=4 +tries=1 +short "@$1" "$host" A 2>/dev/null | grep -E '^[0-9]+(\.[0-9]+){3}$' | sort -u
}

mapfile -t a < <(resolve 223.5.5.5)
mapfile -t b < <(resolve 119.29.29.29)
mapfile -t c < <(resolve 114.114.114.114)
nonempty=0
(( ${#a[@]} > 0 )) && (( nonempty += 1 ))
(( ${#b[@]} > 0 )) && (( nonempty += 1 ))
(( ${#c[@]} > 0 )) && (( nonempty += 1 ))
(( nonempty >= 2 )) || { echo "ERROR: fewer than two public DNS resolvers answered for $host" >&2; exit 3; }

ip=''
for pair in 'a b' 'a c' 'b c'; do
  read -r left_name right_name <<< "$pair"
  declare -n left=$left_name right=$right_name
  for candidate in "${left[@]}"; do
    for other in "${right[@]}"; do
      [[ $candidate == "$other" ]] && ip=$candidate && break 3
    done
  done
done
[[ -n $ip ]] || { echo "ERROR: no address was corroborated by two public DNS resolvers for $host" >&2; exit 3; }

current=0
[[ -f $part ]] && current=$(stat -c '%s' -- "$part")
(( current <= expected_size )) || { echo "ERROR: partial file is larger than expected" >&2; exit 4; }

if [[ -f $meta ]]; then
  read -r old_size old_sha < "$meta"
  [[ $old_size == "$expected_size" && $old_sha == "$expected_sha" ]] || {
    echo "ERROR: .part metadata belongs to a different object" >&2
    exit 4
  }
else
  printf '%s %s\n' "$expected_size" "$expected_sha" > "$meta"
fi

if (( current < expected_size )); then
  headers=$(mktemp "${TMPDIR:-/tmp}/xet-headers.XXXXXX")
  trap 'rm -f -- "$headers"' EXIT
  if (( current == 0 )); then
    transfer_mode=(--range '0-')
  else
    transfer_mode=(--continue-at -)
  fi
  # Read the URL through curl's config stdin so it never appears in curl's argv.
  code=$(printf 'url = "%s"\n' "$signed_url" | curl -q --config - \
    --noproxy '*' --fail --silent --show-error --location \
    --connect-timeout 15 --retry 3 --retry-delay 3 \
    --resolve "$host:443:$ip" \
    "${transfer_mode[@]}" \
    --dump-header "$headers" --output "$part" \
    --write-out '%{http_code}') || rc=$?
  rc=${rc:-0}
  [[ $rc == 0 ]] || { echo "ERROR: transfer failed (curl=$rc); refresh URL and resume" >&2; exit "$rc"; }
  [[ $code == 206 ]] || { truncate -s "$current" -- "$part"; echo "ERROR: expected HTTP 206, got $code" >&2; exit 5; }
  range_start=$(awk 'BEGIN{IGNORECASE=1} /^content-range:/ {gsub("\r",""); split($3,x,"-"); print x[1]}' "$headers" | tail -1)
  [[ $range_start == "$current" ]] || { truncate -s "$current" -- "$part"; echo "ERROR: Content-Range does not start at local size" >&2; exit 5; }
fi

actual_size=$(stat -c '%s' -- "$part")
[[ $actual_size == "$expected_size" ]] || { echo "INCOMPLETE: $actual_size/$expected_size bytes; refresh URL and resume" >&2; exit 6; }
actual_sha=$(sha256sum -- "$part" | awk '{print $1}')
[[ $actual_sha == "$expected_sha" ]] || { echo "ERROR: SHA-256 mismatch" >&2; exit 7; }
mv -- "$part" "$dest"
rm -f -- "$meta"
echo "OK: $dest ($actual_size bytes, sha256=$actual_sha)"
