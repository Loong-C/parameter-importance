#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
unset BASH_ENV ENV CDPATH CUDA_VISIBLE_DEVICES
export PATH="/usr/sbin:/usr/bin:/sbin:/bin"
exec 2>&1

# Read-only, fail-closed verification of the persistent NVIDIA UUID exclusion
# applied for the approved Stage 0 four-GPU path.  This script writes no report
# file and performs no service, driver, PCI, permission, or configuration change.

readonly EXPECTED_HOST="sophgo13"
readonly PROJECT_USER="sophgo13"
readonly PROJECT_PYTHON="/home/sophgo13/cjl/storage/parameter-importance/envs/parameter-importance-stage0-1bd963c65f75/bin/python"
readonly OLD_BOOT_ID="227d26bb-ef3c-420e-bc57-7aa186eddb87"
readonly EXPECTED_DRIVER="575.57.08"
readonly EXPECTED_MODEL="NVIDIA A100-SXM4-80GB"
readonly COMMAND_TIMEOUT_SECONDS=20
readonly PYTORCH_TIMEOUT_SECONDS=120

readonly -a EXPECTED_KERNELS=("6.8.0-134-generic" "6.8.0-136-generic")
readonly -a ALL_BDFS=(
  "0000:4f:00.0" "0000:50:00.0" "0000:53:00.0" "0000:57:00.0"
  "0000:9c:00.0" "0000:9d:00.0" "0000:a0:00.0" "0000:a4:00.0"
)
readonly -a EXCLUDED_BDFS=(
  "0000:4f:00.0" "0000:50:00.0" "0000:53:00.0" "0000:57:00.0"
)
readonly -a EXCLUDED_UUIDS=(
  "GPU-6ff7389b-eaf8-aefd-b2c6-1611be41fa5d"
  "GPU-dc6cfc60-41dd-7bcf-ed09-b7deb5be342c"
  "GPU-180ff767-885a-7dc9-c8a9-921d65a01bbd"
  "GPU-d0ce0b43-7e46-6bca-b078-5aa7043928d7"
)
readonly -a ALLOWED_BDFS=(
  "0000:9c:00.0" "0000:9d:00.0" "0000:a0:00.0" "0000:a4:00.0"
)
readonly -a ALLOWED_UUIDS=(
  "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267"
  "GPU-e78c55cd-db97-b761-f559-dc6eae3be81d"
  "GPU-9b2b2a3b-3547-187f-ca29-2c02624e2e4f"
  "GPU-5a81500d-5e9c-b0d7-5607-fdfdaab65ff4"
)
readonly -a MASKED_INACTIVE_UNITS=(
  docker.service docker.socket containerd.service nvidia-fabricmanager.service
)
readonly -a DISABLED_INACTIVE_UNITS=(
  snap.lxd.activate.service
  snap.lxd.daemon.unix.socket
  snap.lxd.user-daemon.unix.socket
)
readonly -a STATIC_INACTIVE_UNITS=(
  snap.lxd.daemon.service snap.lxd.user-daemon.service
)
readonly -a NOT_FOUND_INACTIVE_UNITS=(lxd.service lxd.socket)

declare -Ar EXPECTED_UUID_BY_BDF=(
  ["0000:4f:00.0"]="GPU-6ff7389b-eaf8-aefd-b2c6-1611be41fa5d"
  ["0000:50:00.0"]="GPU-dc6cfc60-41dd-7bcf-ed09-b7deb5be342c"
  ["0000:53:00.0"]="GPU-180ff767-885a-7dc9-c8a9-921d65a01bbd"
  ["0000:57:00.0"]="GPU-d0ce0b43-7e46-6bca-b078-5aa7043928d7"
  ["0000:9c:00.0"]="GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267"
  ["0000:9d:00.0"]="GPU-e78c55cd-db97-b761-f559-dc6eae3be81d"
  ["0000:a0:00.0"]="GPU-9b2b2a3b-3547-187f-ca29-2c02624e2e4f"
  ["0000:a4:00.0"]="GPU-5a81500d-5e9c-b0d7-5607-fdfdaab65ff4"
)

CURRENT_CHECK="startup"
on_exit() {
  local rc=$?
  trap - EXIT
  if (( rc == 0 )); then
    printf 'RESULT|PASS|all_post_reboot_gpu_exclusion_checks_passed\n'
  else
    printf 'RESULT|FAIL|check=%s|exit_code=%s\n' "${CURRENT_CHECK}" "${rc}"
  fi
  exit "${rc}"
}
trap on_exit EXIT

begin_check() {
  CURRENT_CHECK="$1"
  printf 'CHECK|%s\n' "${CURRENT_CHECK}"
}

pass() {
  printf 'PASS|%s|%s\n' "${CURRENT_CHECK}" "$1"
}

fail() {
  printf 'FAIL|%s|%s\n' "${CURRENT_CHECK}" "$1"
  exit 1
}

contains_exact() {
  local wanted="$1"
  shift
  local item
  for item in "$@"; do
    [[ ${item} == "${wanted}" ]] && return 0
  done
  return 1
}

BOUNDED_OUTPUT=""
run_bounded_for() {
  local timeout_seconds="$1"
  local label="$2"
  shift 2
  local rc
  set +e
  BOUNDED_OUTPUT="$(timeout --signal=TERM --kill-after=2s "${timeout_seconds}s" "$@" 2>&1)"
  rc=$?
  set -e
  if (( rc != 0 )); then
    printf '%s\n' "${BOUNDED_OUTPUT}"
    fail "bounded command failed: ${label}; rc=${rc}"
  fi
}

run_bounded() {
  run_bounded_for "${COMMAND_TIMEOUT_SECONDS}" "$@"
}

for required in awk basename grep hostname id journalctl nvidia-smi pgrep readlink runuser sed stat systemctl timeout uname /usr/bin/python3; do
  command -v "${required}" >/dev/null 2>&1 || fail "required command unavailable: ${required}"
done
[[ -x ${PROJECT_PYTHON} ]] || fail "project Python is unavailable: ${PROJECT_PYTHON}"

if (( EUID == 0 )); then
  readonly -a USER_ENV_PREFIX=(
    "$(command -v runuser)" -u "${PROJECT_USER}" -- /usr/bin/env -i
    HOME=/nonexistent USER="${PROJECT_USER}" LOGNAME="${PROJECT_USER}"
    PATH=/usr/sbin:/usr/bin:/sbin:/bin PYTHONDONTWRITEBYTECODE=1
    PYTHONNOUSERSITE=1 CUDA_CACHE_DISABLE=1
  )
else
  readonly -a USER_ENV_PREFIX=(
    /usr/bin/env -i HOME=/nonexistent USER="$(id -un)" LOGNAME="$(id -un)"
    PATH=/usr/sbin:/usr/bin:/sbin:/bin PYTHONDONTWRITEBYTECODE=1
    PYTHONNOUSERSITE=1 CUDA_CACHE_DISABLE=1
  )
fi

begin_check host
[[ $(hostname -s) == "${EXPECTED_HOST}" ]] || fail "expected ${EXPECTED_HOST}, observed $(hostname -s)"
pass "hostname=${EXPECTED_HOST}"

begin_check boot_and_versions
BOOT_ID="$(< /proc/sys/kernel/random/boot_id)"
[[ ${BOOT_ID} != "${OLD_BOOT_ID}" ]] || fail "boot ID did not change from the pre-reboot value"
[[ ${BOOT_ID} =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] || fail "invalid current boot ID: ${BOOT_ID}"
KERNEL_RELEASE="$(uname -r)"
contains_exact "${KERNEL_RELEASE}" "${EXPECTED_KERNELS[@]}" || fail "unexpected kernel release: ${KERNEL_RELEASE}"
[[ -r /proc/driver/nvidia/version ]] || fail "/proc/driver/nvidia/version is unreadable"
DRIVER_VERSION="$(sed -n 's/^NVRM version:.*  \([0-9][0-9.]*\)  .*/\1/p' /proc/driver/nvidia/version)"
[[ ${DRIVER_VERSION} == "${EXPECTED_DRIVER}" ]] || fail "NVIDIA driver is not ${EXPECTED_DRIVER}: ${DRIVER_VERSION}"
UPTIME_SECONDS="$(awk '{print int($1)}' /proc/uptime)"
(( UPTIME_SECONDS >= 900 )) || fail "post-reboot observation is only ${UPTIME_SECONDS}s; at least 900s required"
pass "boot_id=${BOOT_ID}|kernel=${KERNEL_RELEASE}|driver=${EXPECTED_DRIVER}|uptime_seconds=${UPTIME_SECONDS}"

begin_check pci_inventory
for bdf in "${ALL_BDFS[@]}"; do
  [[ -d /sys/bus/pci/devices/${bdf} ]] || fail "missing PCI function: ${bdf}"
  [[ $(< /sys/bus/pci/devices/${bdf}/vendor) == 0x10de ]] || fail "PCI function is not NVIDIA: ${bdf}"
done
pass "all_8_expected_pci_functions_present"

begin_check excluded_gpus_parameter
[[ -r /proc/driver/nvidia/params ]] || fail "/proc/driver/nvidia/params is unreadable"
PARAM_LINES="$(sed -n 's/^ExcludedGpus:[[:space:]]*//p' /proc/driver/nvidia/params)"
EXPECTED_EXCLUDED_SET="$(IFS=,; printf '%s' "${EXCLUDED_UUIDS[*]}")"
set +e
PARAM_RESULT="$(/usr/bin/env -i EXPECTED="${EXPECTED_EXCLUDED_SET}" ACTUAL="${PARAM_LINES}" /usr/bin/python3 -I -c '
import os
expected = os.environ["EXPECTED"].split(",")
raw = os.environ["ACTUAL"].strip()
if "\n" in raw or not raw:
    raise SystemExit("missing or duplicate ExcludedGpus parameter")
if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"\047":
    raw = raw[1:-1]
actual = [item.strip() for item in raw.split(",") if item.strip()]
if len(actual) != 4 or len(set(actual)) != 4 or actual != expected:
    raise SystemExit(f"ExcludedGpus mismatch: {actual!r}")
print(",".join(actual))
' 2>&1)"
PARAM_RC=$?
set -e
(( PARAM_RC == 0 )) || fail "${PARAM_RESULT}"
pass "exact_excluded_uuids=${PARAM_RESULT}"

begin_check allowed_driver_identity
declare -A ALLOWED_MINOR_BY_BDF=()
declare -A EXPECTED_NUMERIC_NODE=()
for bdf in "${ALLOWED_BDFS[@]}"; do
  driver="/sys/bus/pci/devices/${bdf}/driver"
  info="/proc/driver/nvidia/gpus/${bdf}/information"
  [[ -L ${driver} && $(basename "$(readlink -f "${driver}")") == nvidia ]] || fail "allowed BDF is not bound to nvidia: ${bdf}"
  [[ -r ${info} ]] || fail "allowed GPU information is unreadable: ${bdf}"
  model="$(sed -n 's/^Model:[[:space:]]*//p' "${info}")"
  uuid="$(sed -n 's/^GPU UUID:[[:space:]]*//p' "${info}")"
  minor="$(sed -n 's/^Device Minor:[[:space:]]*//p' "${info}")"
  [[ ${model} == "${EXPECTED_MODEL}" ]] || fail "model mismatch at ${bdf}: ${model}"
  [[ ${uuid} == "${EXPECTED_UUID_BY_BDF[${bdf}]}" ]] || fail "UUID mismatch at ${bdf}: ${uuid}"
  [[ ${minor} =~ ^[0-9]+$ ]] || fail "invalid device minor at ${bdf}: ${minor}"
  [[ -z ${EXPECTED_NUMERIC_NODE[${minor}]+x} ]] || fail "duplicate allowed device minor: ${minor}"
  ALLOWED_MINOR_BY_BDF["${bdf}"]="${minor}"
  EXPECTED_NUMERIC_NODE["${minor}"]=1
done
pass "four_allowed_bdfs_bound_with_exact_uuid_and_model"

verify_device_nodes() {
  local node basename_value minor_value metadata rdev major_hex minor_hex
  local -a numeric_nodes=() nodes=()
  shopt -s nullglob
  numeric_nodes=(/dev/nvidia[0-9]*)
  nodes=(
    "${numeric_nodes[@]}" /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools
    /dev/nvidia-modeset /dev/nvidia-nvswitchctl /dev/nvidia-caps/*
  )
  shopt -u nullglob
  (( ${#numeric_nodes[@]} == 4 )) || fail "expected 4 numeric NVIDIA nodes, observed ${#numeric_nodes[@]}"
  [[ -c /dev/nvidiactl ]] || fail "required shared node is missing: /dev/nvidiactl"
  for node in "${numeric_nodes[@]}"; do
    basename_value="${node##*/}"
    minor_value="${basename_value#nvidia}"
    [[ ${minor_value} =~ ^[0-9]+$ && -n ${EXPECTED_NUMERIC_NODE[${minor_value}]+x} ]] || fail "unexpected numeric NVIDIA node: ${node}"
    rdev="$(stat -c '%t:%T' -- "${node}")"
    major_hex="${rdev%%:*}"
    minor_hex="${rdev##*:}"
    (( 16#${major_hex} == 195 && 16#${minor_hex} == 10#${minor_value} )) \
      || fail "numeric node device number mismatch: ${node}=${rdev}"
  done
  for minor_value in "${!EXPECTED_NUMERIC_NODE[@]}"; do
    [[ -c /dev/nvidia${minor_value} ]] || fail "missing numeric node for allowed minor ${minor_value}"
  done
  for node in "${nodes[@]}"; do
    [[ -c ${node} ]] || fail "NVIDIA path is not a character device: ${node}"
    metadata="$(stat -c '%u:%g:%a' -- "${node}")"
    [[ ${metadata} == 0:0:666 ]] || fail "unsafe/unrestored node metadata: ${node}=${metadata}, expected 0:0:666"
  done
}

begin_check device_nodes_initial
verify_device_nodes
pass "numeric_nodes_exactly_map_allowed_minors_and_all_nodes_are_root_root_0666"

verify_service_barrier() {
  local unit active enabled load_state
  for unit in "${MASKED_INACTIVE_UNITS[@]}"; do
    active="$(systemctl is-active "${unit}" 2>/dev/null || true)"
    enabled="$(systemctl is-enabled "${unit}" 2>/dev/null || true)"
    [[ ${active} == inactive ]] || fail "unit must be inactive: ${unit}=${active}"
    [[ ${enabled} == masked ]] || fail "unit must be persistently masked: ${unit}=${enabled}"
  done
  for unit in "${DISABLED_INACTIVE_UNITS[@]}"; do
    active="$(systemctl is-active "${unit}" 2>/dev/null || true)"
    enabled="$(systemctl is-enabled "${unit}" 2>/dev/null || true)"
    [[ ${active} == inactive && ${enabled} == disabled ]] \
      || fail "Snap LXD unit must be disabled/inactive: ${unit}, enabled=${enabled}, active=${active}"
  done
  for unit in "${STATIC_INACTIVE_UNITS[@]}"; do
    active="$(systemctl is-active "${unit}" 2>/dev/null || true)"
    enabled="$(systemctl is-enabled "${unit}" 2>/dev/null || true)"
    [[ ${active} == inactive && ${enabled} == static ]] \
      || fail "Snap LXD service must be static/inactive: ${unit}, enabled=${enabled}, active=${active}"
  done
  for unit in "${NOT_FOUND_INACTIVE_UNITS[@]}"; do
    active="$(systemctl is-active "${unit}" 2>/dev/null || true)"
    load_state="$(systemctl show --property=LoadState --value "${unit}" 2>/dev/null || true)"
    [[ ${active} == inactive && ${load_state} == not-found ]] \
      || fail "legacy LXD unit must be not-found/inactive: ${unit}, load=${load_state}, active=${active}"
  done
  [[ $(systemctl is-active nvidia-persistenced.service 2>/dev/null || true) == active ]] || fail "nvidia-persistenced.service is not active"
  if pgrep -f '(^|/)(containerd-shim|dockerd|containerd|lxd|lxc-start|nv-fabricmanager|nvidia-fabricmanager)([[:space:]]|$)' >/dev/null; then
    fail "allocator, LXD, or Fabric Manager process exists while the reboot barrier must be active"
  fi
}

begin_check service_barrier_initial
verify_service_barrier
pass "allocator_and_fabric_units_masked;LXD_units_disabled_static_or_not_found;all_inactive;persistenced_active"

EXCLUDED_MAP_VALUE="$(for index in "${!EXCLUDED_BDFS[@]}"; do printf '%s=%s\n' "${EXCLUDED_BDFS[${index}]}" "${EXCLUDED_UUIDS[${index}]}"; done)"
ALLOWED_MAP_VALUE="$(for index in "${!ALLOWED_BDFS[@]}"; do printf '%s=%s\n' "${ALLOWED_BDFS[${index}]}" "${ALLOWED_UUIDS[${index}]}"; done)"

validate_excluded_listing() {
  local listing="$1"
  /usr/bin/env -i EXPECTED_MAP="${EXCLUDED_MAP_VALUE}" LISTING="${listing}" /usr/bin/python3 -I -c '
import os, re
def pci(value):
    parts = value.lower().split(":")
    if len(parts) != 3:
        raise ValueError(value)
    return f"{parts[0][-4:]}:{parts[1]}:{parts[2]}"
expected = dict(line.split("=", 1) for line in os.environ["EXPECTED_MAP"].splitlines())
observed = {}
uuid_re = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
pci_re = re.compile(r"(?i)(?:[0-9a-f]{4,8}):[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
for line in os.environ["LISTING"].splitlines():
    uuids = uuid_re.findall(line)
    bdfs = pci_re.findall(line)
    if uuids or bdfs:
        if len(uuids) != 1 or len(bdfs) != 1:
            raise SystemExit(f"unparseable excluded-GPU line: {line!r}")
        key = pci(bdfs[0])
        if key in observed:
            raise SystemExit(f"duplicate excluded BDF: {key}")
        observed[key] = uuids[0]
if observed != expected:
    raise SystemExit(f"excluded listing mismatch: {observed!r}")
'
}

begin_check nvidia_smi_excluded_listings
run_bounded "nvidia-smi -B" "${USER_ENV_PREFIX[@]}" /usr/bin/nvidia-smi -B
EXCLUDED_SHORT="${BOUNDED_OUTPUT}"
validate_excluded_listing "${EXCLUDED_SHORT}" || fail "nvidia-smi -B did not match the exact excluded BDF/UUID map"
run_bounded "nvidia-smi --list-excluded-gpus" "${USER_ENV_PREFIX[@]}" /usr/bin/nvidia-smi --list-excluded-gpus
EXCLUDED_LONG="${BOUNDED_OUTPUT}"
validate_excluded_listing "${EXCLUDED_LONG}" || fail "nvidia-smi --list-excluded-gpus did not match the exact excluded BDF/UUID map"
pass "both_bounded_aliases_report_exactly_four_excluded_gpus"

begin_check nvml_health
run_bounded "NVML identity/ECC query" "${USER_ENV_PREFIX[@]}" /usr/bin/nvidia-smi \
  --query-gpu=pci.bus_id,uuid,driver_version,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total \
  --format=csv,noheader,nounits
NVML_CSV="${BOUNDED_OUTPUT}"
run_bounded "row-remapper query" "${USER_ENV_PREFIX[@]}" /usr/bin/nvidia-smi -q -d ROW_REMAPPER
ROW_REMAPPER="${BOUNDED_OUTPUT}"
run_bounded "compute-app query" "${USER_ENV_PREFIX[@]}" /usr/bin/nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader
COMPUTE_APPS="${BOUNDED_OUTPUT}"
[[ -z ${COMPUTE_APPS//[[:space:]]/} ]] || fail "active compute applications reported: ${COMPUTE_APPS}"
/usr/bin/env -i ALLOWED_MAP="${ALLOWED_MAP_VALUE}" NVML_CSV="${NVML_CSV}" ROW_REMAPPER="${ROW_REMAPPER}" EXPECTED_DRIVER="${EXPECTED_DRIVER}" /usr/bin/python3 -I -c '
import csv, io, os, re
def pci(value):
    fields = value.strip().lower().split(":")
    if len(fields) != 3:
        raise ValueError(value)
    return f"{fields[0][-4:]}:{fields[1]}:{fields[2]}"
allowed = dict(line.split("=", 1) for line in os.environ["ALLOWED_MAP"].splitlines())
rows = list(csv.reader(io.StringIO(os.environ["NVML_CSV"])))
if len(rows) != 4:
    raise SystemExit(f"NVML expected 4 GPUs, observed {len(rows)}")
observed = {}
for row in rows:
    if len(row) != 5:
        raise SystemExit(f"unexpected NVML row: {row!r}")
    bdf, uuid, driver, volatile_uce, aggregate_uce = (x.strip() for x in row)
    bdf = pci(bdf)
    if bdf in observed:
        raise SystemExit(f"duplicate NVML BDF: {bdf}")
    observed[bdf] = uuid
    if driver != os.environ["EXPECTED_DRIVER"]:
        raise SystemExit(f"driver mismatch at {bdf}: {driver}")
    if volatile_uce != "0" or aggregate_uce != "0":
        raise SystemExit(f"uncorrectable ECC at {bdf}: volatile={volatile_uce}, aggregate={aggregate_uce}")
if observed != allowed:
    raise SystemExit(f"NVML mapping mismatch: {observed!r}")
blocks = re.split(r"(?=^GPU\s+[0-9A-Fa-f:.]+\s*$)", os.environ["ROW_REMAPPER"], flags=re.M)
states = {}
for block in blocks:
    match = re.search(r"^GPU\s+([0-9A-Fa-f:.]+)\s*$", block, flags=re.M)
    if not match:
        continue
    bdf = pci(match.group(1))
    pending = re.search(r"^\s*Pending\s*:\s*(\S+)\s*$", block, flags=re.M)
    failure = re.search(r"^\s*Remapping Failure Occurred\s*:\s*(\S+)\s*$", block, flags=re.M)
    if not pending or not failure:
        raise SystemExit(f"incomplete row-remap state at {bdf}")
    states[bdf] = (pending.group(1), failure.group(1))
if set(states) != set(allowed):
    raise SystemExit(f"row-remap BDF mismatch: {sorted(states)!r}")
for bdf, state in states.items():
    if state != ("No", "No"):
        raise SystemExit(f"row-remap gate failed at {bdf}: {state!r}")
' || fail "basic NVML health contract failed"
pass "nvml_exact_four_allowed;driver_and_ecc_clean;row_remap_clean;no_compute"

begin_check pytorch_enumeration
EXPECTED_ALLOWED_ORDER="$(IFS=,; printf '%s' "${ALLOWED_UUIDS[*]}")"
run_bounded_for "${PYTORCH_TIMEOUT_SECONDS}" "project PyTorch enumeration" "${USER_ENV_PREFIX[@]}" EXPECTED_UUIDS="${EXPECTED_ALLOWED_ORDER}" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID "${PROJECT_PYTHON}" -I -c '
import json, os
if "CUDA_VISIBLE_DEVICES" in os.environ:
    raise SystemExit("CUDA_VISIBLE_DEVICES must be absent")
import torch
expected = os.environ["EXPECTED_UUIDS"].split(",")
if torch.cuda.device_count() != 4:
    raise SystemExit(f"PyTorch expected 4 devices, observed {torch.cuda.device_count()}")
observed = []
for index in range(4):
    value = getattr(torch.cuda.get_device_properties(index), "uuid", None)
    if isinstance(value, bytes):
        value = value.decode("ascii")
    observed.append(str(value or ""))
if observed != expected:
    raise SystemExit(f"PyTorch UUID order mismatch: {observed!r}")
print(json.dumps({"device_count": 4, "uuids": observed}, separators=(",", ":")))
'
pass "CUDA_VISIBLE_DEVICES_absent|${BOUNDED_OUTPUT}"

begin_check post_probe_safety
run_bounded "post-PyTorch compute-app query" "${USER_ENV_PREFIX[@]}" /usr/bin/nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader
[[ -z ${BOUNDED_OUTPUT//[[:space:]]/} ]] || fail "compute application remained after PyTorch exited: ${BOUNDED_OUTPUT}"
verify_device_nodes
verify_service_barrier
pass "no_compute_contexts;device_permissions_and_service_barrier_unchanged"

begin_check current_boot_kernel_log
run_bounded "current-boot kernel journal" /usr/bin/journalctl -k -b "${BOOT_ID}" --no-pager --quiet -o short-monotonic
KERNEL_LOG="${BOUNDED_OUTPUT}"
[[ -n ${KERNEL_LOG//[[:space:]]/} && ${KERNEL_LOG} == *"Linux version"* ]] || fail "current-boot kernel journal is empty, incomplete, or unreadable"
ALLOWED_BDF_RE='0000:(9c|9d|a0|a4):00(\.0)?'
ERROR_RE='NVRM:.*Xid|RmInitAdapter|rm_init_adapter|AER:|PCIe Bus Error'
if printf '%s\n' "${KERNEL_LOG}" | grep -Eiq "(${ALLOWED_BDF_RE}).*(${ERROR_RE})|(${ERROR_RE}).*(${ALLOWED_BDF_RE})"; then
  printf '%s\n' "${KERNEL_LOG}" | grep -Ei "(${ALLOWED_BDF_RE}).*(${ERROR_RE})|(${ERROR_RE}).*(${ALLOWED_BDF_RE})" || true
  fail "current boot contains an allowed-GPU Xid/RmInit/AER record"
fi
pass "no_allowed_gpu_xid_rminit_or_aer_in_current_boot"

CURRENT_CHECK="complete"
