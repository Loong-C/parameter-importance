#!/usr/bin/env bash
set -Eeuo pipefail
umask 022
unset BASH_ENV ENV CDPATH CUDA_VISIBLE_DEVICES
export PATH="/usr/sbin:/usr/bin:/sbin:/bin"

# Finalize the persistent Stage 0 UUID-exclusion path after a reboot.  This
# script is intentionally self-contained: it creates fresh root-owned evidence,
# and it will not trust a caller-supplied PASS file or GPU visibility mask.

readonly EXPECTED_HOST="sophgo13"
readonly PROJECT_USER="sophgo13"
readonly CANDIDATE_PYTHON="/home/sophgo13/cjl/storage/parameter-importance/envs/parameter-importance-stage0-1bd963c65f75/bin/python"
readonly EXPECTED_DRIVER="575.57.08"
readonly ADMIN_ROOT="/var/lib/parameter-importance/stage0/g0-g-uuid-exclusion/service-finalize"
readonly LOCK_DIR="/run/lock/parameter-importance"
readonly LOCK_PATH="${LOCK_DIR}/gpu-uuid-exclusion.lock"
readonly FABRIC_UNIT="nvidia-fabricmanager.service"
readonly SNAP_LXD_ACTIVATE="snap.lxd.activate.service"
readonly SNAP_LXD_DAEMON_SOCKET="snap.lxd.daemon.unix.socket"
readonly SNAP_LXD_USER_SOCKET="snap.lxd.user-daemon.unix.socket"
readonly SNAP_LXD_DAEMON_SERVICE="snap.lxd.daemon.service"
readonly SNAP_LXD_USER_SERVICE="snap.lxd.user-daemon.service"

readonly -a ALLOWED_BDFS=(
  "0000:9c:00.0" "0000:9d:00.0" "0000:a0:00.0" "0000:a4:00.0"
)
readonly -a ALLOWED_UUIDS=(
  "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267"
  "GPU-e78c55cd-db97-b761-f559-dc6eae3be81d"
  "GPU-9b2b2a3b-3547-187f-ca29-2c02624e2e4f"
  "GPU-5a81500d-5e9c-b0d7-5607-fdfdaab65ff4"
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
readonly -a MASKED_ALLOCATOR_UNITS=(
  "containerd.service"
  "docker.socket"
  "docker.service"
)
readonly -a SNAP_LXD_ENTRY_UNITS=(
  "${SNAP_LXD_ACTIVATE}"
  "${SNAP_LXD_DAEMON_SOCKET}"
  "${SNAP_LXD_USER_SOCKET}"
)
readonly -a SNAP_LXD_STATIC_SERVICES=(
  "${SNAP_LXD_DAEMON_SERVICE}"
  "${SNAP_LXD_USER_SERVICE}"
)
readonly EXPECTED_EXCLUDED_CSV="$(IFS=,; printf '%s' "${EXCLUDED_UUIDS[*]}")"

if [[ ${EUID} -ne 0 ]]; then
  printf 'ERROR: root is required.\n' >&2
  exit 2
fi
if [[ $(hostname -s) != "${EXPECTED_HOST}" ]]; then
  printf 'ERROR: expected host %s.\n' "${EXPECTED_HOST}" >&2
  exit 3
fi
if (( $# != 0 )); then
  printf 'ERROR: this finalizer accepts no arguments.\n' >&2
  exit 4
fi
for required in flock fuser lspci nvidia-smi runuser timeout journalctl \
  docker ctr lxc /usr/bin/python3; do
  command -v "${required}" >/dev/null 2>&1 || {
    printf 'ERROR: required command is unavailable: %s\n' "${required}" >&2
    exit 5
  }
done
[[ -x ${CANDIDATE_PYTHON} ]] || {
  printf 'ERROR: candidate Python is unavailable.\n' >&2
  exit 6
}

if [[ -e ${LOCK_DIR} || -L ${LOCK_DIR} ]]; then
  [[ -d ${LOCK_DIR} && ! -L ${LOCK_DIR} \
      && $(stat -c '%u:%g:%a' -- "${LOCK_DIR}") == "0:0:700" ]] || {
    printf 'ERROR: untrusted lock directory.\n' >&2
    exit 7
  }
else
  mkdir -m 0700 "${LOCK_DIR}"
  chown root:root "${LOCK_DIR}"
fi
exec 9>"${LOCK_PATH}"
flock -n 9 || {
  printf 'ERROR: another GPU exclusion operation holds the lock.\n' >&2
  exit 8
}

install -d -o root -g root -m 0755 "${ADMIN_ROOT}"
readonly RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$(mktemp -d -p "${ADMIN_ROOT}" "finalize-${RUN_ID}.XXXXXXXX")"
readonly RUN_DIR
chmod 0755 "${RUN_DIR}"
readonly LOG_PATH="${RUN_DIR}/admin-finalize.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

PRECONDITIONS_VERIFIED=0
MUTATION_STARTED=0
FINAL_PASS=0
SAFE_HOLD_VERIFIED=0
FAILED_LINE=""
FAILED_COMMAND=""
FABRIC_OUTCOME="NOT_ATTEMPTED"
GPU_NODES=()

record_error() {
  FAILED_LINE="$1"
  FAILED_COMMAND="$2"
}

mask_and_stop_unit() {
  local unit="$1"
  systemctl stop "${unit}" >/dev/null 2>&1 || true
  systemctl mask "${unit}" >/dev/null 2>&1 || true
}

enter_safe_hold() {
  local node unit ok=1
  set +e
  for unit in "${SNAP_LXD_ENTRY_UNITS[@]}"; do
    systemctl stop "${unit}" >/dev/null 2>&1 || true
    systemctl disable "${unit}" >/dev/null 2>&1 || ok=0
  done
  for unit in "${SNAP_LXD_STATIC_SERVICES[@]}"; do
    systemctl stop "${unit}" >/dev/null 2>&1 || true
  done
  for ((i=${#MASKED_ALLOCATOR_UNITS[@]} - 1; i >= 0; i--)); do
    mask_and_stop_unit "${MASKED_ALLOCATOR_UNITS[${i}]}"
  done
  mask_and_stop_unit "${FABRIC_UNIT}"
  systemctl stop nvidia-persistenced.service >/dev/null 2>&1 || true
  shopt -s nullglob
  GPU_NODES=(/dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm \
    /dev/nvidia-uvm-tools /dev/nvidia-modeset /dev/nvidia-nvswitchctl \
    /dev/nvidia-caps/*)
  shopt -u nullglob
  for node in "${GPU_NODES[@]}"; do
    chmod 0600 -- "${node}" || ok=0
    [[ $(stat -c '%a:%u:%g' -- "${node}" 2>/dev/null) == "600:0:0" ]] || ok=0
  done
  for unit in "${MASKED_ALLOCATOR_UNITS[@]}" "${FABRIC_UNIT}"; do
    systemctl is-active --quiet "${unit}" && ok=0
    state="$(systemctl is-enabled "${unit}" 2>/dev/null || true)"
    [[ ${state} == masked || ${state} == masked-runtime ]] || ok=0
  done
  for unit in "${SNAP_LXD_ENTRY_UNITS[@]}"; do
    systemctl is-active --quiet "${unit}" && ok=0
    [[ $(systemctl is-enabled "${unit}" 2>/dev/null || true) == disabled ]] || ok=0
  done
  for unit in "${SNAP_LXD_STATIC_SERVICES[@]}"; do
    systemctl is-active --quiet "${unit}" && ok=0
    [[ $(systemctl is-enabled "${unit}" 2>/dev/null || true) == static ]] || ok=0
  done
  SAFE_HOLD_VERIFIED=${ok}
  set -e
  [[ ${ok} -eq 1 ]]
}

publish_failure() {
  local rc="$1" tmp
  tmp="$(mktemp -p "${RUN_DIR}" .failure.XXXXXXXX)"
  {
    printf 'status=FAILED\n'
    printf 'exit_code=%s\n' "${rc}"
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'failed_line=%s\n' "${FAILED_LINE}"
    printf 'failed_command=%s\n' "${FAILED_COMMAND}"
    printf 'fabric_manager=%s\n' "${FABRIC_OUTCOME}"
    printf 'safe_hold_verified=%s\n' "${SAFE_HOLD_VERIFIED}"
  } > "${tmp}"
  chmod 0444 "${tmp}"
  mv -f "${tmp}" "${RUN_DIR}/FAILURE"
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ ${rc} -ne 0 && ${FINAL_PASS} -eq 0 ]]; then
    if [[ ${PRECONDITIONS_VERIFIED} -eq 1 || ${MUTATION_STARTED} -eq 1 ]]; then
      enter_safe_hold || true
    fi
    publish_failure "${rc}" || true
  fi
  exit "${rc}"
}

trap 'record_error "${LINENO}" "${BASH_COMMAND}"' ERR
trap on_exit EXIT

critical_kernel_errors() {
  local path="$1"
  grep -Ei \
    'NVRM:.*Xid|RmInitAdapter|AER:|PCIe Bus Error|GPU has fallen off the bus|Could not exclude GPU|Failed to exclude GPU|GSP.*(fail|fatal)' \
    "${path}"
}

assert_masked_inactive() {
  local unit="$1" state
  state="$(systemctl is-enabled "${unit}" 2>/dev/null || true)"
  [[ ${state} == masked || ${state} == masked-runtime ]] || {
    printf 'ERROR: unit is not masked: %s (%s)\n' "${unit}" "${state}" >&2
    return 1
  }
  if systemctl is-active --quiet "${unit}"; then
    printf 'ERROR: masked unit is still active: %s\n' "${unit}" >&2
    return 1
  fi
}

validate_module_contract() {
  local output_path="$1"
  env -i PATH=/usr/bin:/bin EXPECTED="${EXPECTED_EXCLUDED_CSV}" \
    /usr/bin/python3 -I - /proc/driver/nvidia/params "${output_path}" <<'PY'
import pathlib
import sys
import os

text = pathlib.Path(sys.argv[1]).read_text()
line = next((x for x in text.splitlines() if x.startswith("ExcludedGpus:")), None)
if line is None:
    raise SystemExit("ExcludedGpus is absent from driver params")
value = line.split(":", 1)[1].strip().strip('"').replace(" ", "")
if value != os.environ["EXPECTED"]:
    raise SystemExit(f"ExcludedGpus mismatch: {value!r}")
pathlib.Path(sys.argv[2]).write_text(line + "\n")
PY
  [[ $(sed -n 's/^NVRM version:.*  \([0-9][0-9.]*\)  .*/\1/p' /proc/driver/nvidia/version) \
      == "${EXPECTED_DRIVER}" ]] || {
    printf 'ERROR: NVIDIA driver version mismatch.\n' >&2
    return 1
  }
}

validate_proc_and_excluded_inventory() {
  local output_path="$1" bdf
  timeout 30 nvidia-smi -B > "${RUN_DIR}/excluded-gpus-${output_path}.txt"
  ALLOWED_MAP="$(for i in "${!ALLOWED_BDFS[@]}"; do printf '%s=%s\n' "${ALLOWED_BDFS[${i}]}" "${ALLOWED_UUIDS[${i}]}"; done)"
  EXCLUDED_MAP="$(for i in "${!EXCLUDED_BDFS[@]}"; do printf '%s=%s\n' "${EXCLUDED_BDFS[${i}]}" "${EXCLUDED_UUIDS[${i}]}"; done)"
  env -i PATH=/usr/bin:/bin ALLOWED_MAP="${ALLOWED_MAP}" EXCLUDED_MAP="${EXCLUDED_MAP}" \
    /usr/bin/python3 -I - "${RUN_DIR}/excluded-gpus-${output_path}.txt" \
      "${RUN_DIR}/proc-inventory-${output_path}.txt" <<'PY'
import os
import pathlib
import re
import sys

allowed = dict(x.split("=", 1) for x in os.environ["ALLOWED_MAP"].splitlines())
excluded = dict(x.split("=", 1) for x in os.environ["EXCLUDED_MAP"].splitlines())
listed = set(re.findall(r"GPU-[0-9A-Fa-f-]{36}", pathlib.Path(sys.argv[1]).read_text()))
if listed != set(excluded.values()):
    raise SystemExit(f"excluded-GPU listing mismatch: {sorted(listed)!r}")
rows = []
for bdf, expected_uuid in {**excluded, **allowed}.items():
    pci = pathlib.Path("/sys/bus/pci/devices") / bdf
    driver = pci / "driver"
    info = pathlib.Path("/proc/driver/nvidia/gpus") / bdf / "information"
    if not pci.exists() or not driver.is_symlink() or driver.resolve().name != "nvidia" or not info.is_file():
        raise SystemExit(f"missing NVIDIA binding/proc entry: {bdf}")
    text = info.read_text()
    uuid = re.search(r"^GPU UUID:\s*(\S+)", text, re.MULTILINE)
    state = re.search(r"^GPU Excluded:\s*(Yes|No)", text, re.MULTILINE)
    if not uuid or uuid.group(1) != expected_uuid or not state:
        raise SystemExit(f"identity/exclusion entry mismatch: {bdf}")
    expected_state = "Yes" if bdf in excluded else "No"
    if state.group(1) != expected_state:
        raise SystemExit(f"wrong exclusion state for {bdf}: {state.group(1)}")
    rows.append(f"{bdf}|{expected_uuid}|{state.group(1)}")
pathlib.Path(sys.argv[2]).write_text("\n".join(rows) + "\n")
PY
}

validate_health_snapshot() {
  local prefix="$1"
  timeout 30 nvidia-smi \
    --query-gpu=pci.bus_id,uuid,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total,memory.used,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits > "${RUN_DIR}/nvml-${prefix}.csv"
  timeout 30 nvidia-smi -q -d ROW_REMAPPER > "${RUN_DIR}/row-remapper-${prefix}.txt"
  timeout 30 nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader \
    > "${RUN_DIR}/compute-apps-${prefix}.csv"
  [[ ! -s ${RUN_DIR}/compute-apps-${prefix}.csv ]] || {
    printf 'ERROR: compute application detected during %s.\n' "${prefix}" >&2
    return 1
  }
  ALLOWED_MAP="$(for i in "${!ALLOWED_BDFS[@]}"; do printf '%s=%s\n' "${ALLOWED_BDFS[${i}]}" "${ALLOWED_UUIDS[${i}]}"; done)"
  env -i PATH=/usr/bin:/bin ALLOWED_MAP="${ALLOWED_MAP}" \
    /usr/bin/python3 -I - "${RUN_DIR}/nvml-${prefix}.csv" \
      "${RUN_DIR}/row-remapper-${prefix}.txt" <<'PY'
import csv
import os
import pathlib
import re
import sys

def pci(v):
    parts = v.strip().lower().split(":")
    if len(parts) != 3:
        raise SystemExit(f"bad PCI address: {v!r}")
    return f"{parts[0][-4:]}:{parts[1]}:{parts[2]}"

allowed = dict(x.split("=", 1) for x in os.environ["ALLOWED_MAP"].splitlines())
rows = list(csv.reader(pathlib.Path(sys.argv[1]).read_text().splitlines()))
if len(rows) != 4:
    raise SystemExit(f"NVML expected 4 GPUs, got {len(rows)}")
seen = {}
for row in rows:
    if len(row) != 7:
        raise SystemExit(f"bad NVML row: {row!r}")
    bdf, uuid, volatile, aggregate, memory, util, temp = (x.strip() for x in row)
    bdf = pci(bdf)
    seen[bdf] = uuid
    if volatile != "0" or aggregate != "0":
        raise SystemExit(f"ECC failure: {bdf} {volatile}/{aggregate}")
    if memory != "0" or util != "0":
        raise SystemExit(f"activity failure: {bdf} {memory}/{util}")
    if not temp.isdigit() or int(temp) >= 85:
        raise SystemExit(f"temperature failure: {bdf} {temp}")
if seen != allowed:
    raise SystemExit(f"allowed mapping mismatch: {seen!r}")

text = pathlib.Path(sys.argv[2]).read_text()
blocks = re.split(r"(?=^GPU\s+[0-9A-Fa-f:.]+\s*$)", text, flags=re.MULTILINE)
states = {}
for block in blocks:
    match = re.search(r"^GPU\s+([0-9A-Fa-f:.]+)\s*$", block, flags=re.MULTILINE)
    if not match:
        continue
    bdf = pci(match.group(1))
    pending = re.search(r"^\s*Pending\s*:\s*(\S+)", block, flags=re.MULTILINE)
    failure = re.search(r"^\s*Remapping Failure Occurred\s*:\s*(\S+)", block, flags=re.MULTILINE)
    if not pending or not failure:
        raise SystemExit(f"incomplete row-remap state: {bdf}")
    states[bdf] = (pending.group(1), failure.group(1))
if set(states) != set(allowed):
    raise SystemExit(f"row-remap set mismatch: {sorted(states)!r}")
if any(v != ("No", "No") for v in states.values()):
    raise SystemExit(f"row-remap failure: {states!r}")
PY
}

validate_pytorch() {
  local prefix="$1"
  timeout 120 runuser -u "${PROJECT_USER}" -- env -i \
    HOME="/home/${PROJECT_USER}" USER="${PROJECT_USER}" LOGNAME="${PROJECT_USER}" \
    PATH="$(dirname "${CANDIDATE_PYTHON}"):/usr/bin:/bin" \
    PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    EXPECTED_UUIDS="$(IFS=,; printf '%s' "${ALLOWED_UUIDS[*]}")" \
    "${CANDIDATE_PYTHON}" -I - > "${RUN_DIR}/pytorch-${prefix}.json" <<'PY'
import json
import os
import torch

expected = os.environ["EXPECTED_UUIDS"].split(",")
if torch.cuda.device_count() != 4:
    raise SystemExit(f"PyTorch expected 4 GPUs, got {torch.cuda.device_count()}")
rows = []
for index in range(4):
    p = torch.cuda.get_device_properties(index)
    uuid = getattr(p, "uuid", "")
    if isinstance(uuid, bytes):
        uuid = uuid.decode("ascii")
    rows.append({"index": index, "uuid": str(uuid), "name": p.name, "memory": p.total_memory})
if [x["uuid"] for x in rows] != expected:
    raise SystemExit(f"PyTorch UUID mismatch: {rows!r}")
print(json.dumps({"device_count": 4, "devices": rows}, indent=2))
PY
}

validate_gpu_clients() {
  local prefix="$1" allow_fabric="$2"
  shopt -s nullglob
  GPU_NODES=(/dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm \
    /dev/nvidia-uvm-tools /dev/nvidia-modeset /dev/nvidia-nvswitchctl \
    /dev/nvidia-caps/*)
  shopt -u nullglob
  NODES="$(printf '%s\n' "${GPU_NODES[@]}")" ALLOW_FABRIC="${allow_fabric}" \
    /usr/bin/python3 -I - "${RUN_DIR}/gpu-clients-${prefix}.json" <<'PY'
import json
import os
import pathlib
import sys

nodes = set(os.environ["NODES"].splitlines())
clients = []
for proc in pathlib.Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        comm = (proc / "comm").read_text().strip()
        fds = list((proc / "fd").iterdir())
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    matches = set()
    for fd in fds:
        try:
            target = os.readlink(fd)
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if target in nodes:
            matches.add(target)
    matches = sorted(matches)
    if matches:
        clients.append({"pid": int(proc.name), "comm": comm, "nodes": matches})
allowed = []
for item in clients:
    if item["comm"].startswith("nvidia-persiste"):
        allowed.append(item)
    elif os.environ["ALLOW_FABRIC"] == "1" and item["comm"].startswith("nv-fabric"):
        allowed.append(item)
    else:
        raise SystemExit(f"unexpected GPU client: {item!r}")
pathlib.Path(sys.argv[1]).write_text(json.dumps({"clients": clients}, indent=2) + "\n")
PY
}

run_exact_four_gate() {
  local prefix="$1" allow_fabric="$2"
  validate_module_contract "${RUN_DIR}/module-params-${prefix}.txt"
  validate_proc_and_excluded_inventory "${prefix}"
  validate_health_snapshot "${prefix}"
  validate_pytorch "${prefix}"
  validate_health_snapshot "${prefix}-after-pytorch"
  validate_gpu_clients "${prefix}" "${allow_fabric}"
}

printf 'Stage 0 UUID-exclusion service finalizer: %s\n' "$(date --iso-8601=seconds)"
printf 'Run directory: %s\n' "${RUN_DIR}"

uptime_seconds="$(awk '{print int($1)}' /proc/uptime)"
if (( uptime_seconds < 900 )); then
  printf 'ERROR: postboot observation is only %s seconds; 900 required.\n' "${uptime_seconds}" >&2
  exit 20
fi

for unit in "${MASKED_ALLOCATOR_UNITS[@]}" "${FABRIC_UNIT}"; do
  assert_masked_inactive "${unit}"
done
for unit in "${SNAP_LXD_ENTRY_UNITS[@]}"; do
  [[ $(systemctl is-enabled "${unit}" 2>/dev/null || true) == disabled ]] || {
    printf 'ERROR: Snap LXD entry must be disabled before finalization: %s\n' "${unit}" >&2
    exit 20
  }
  if systemctl is-active --quiet "${unit}"; then
    printf 'ERROR: Snap LXD entry must be inactive before finalization: %s\n' "${unit}" >&2
    exit 20
  fi
done
for unit in "${SNAP_LXD_STATIC_SERVICES[@]}"; do
  [[ $(systemctl is-enabled "${unit}" 2>/dev/null || true) == static ]] || {
    printf 'ERROR: Snap LXD service must retain static unit state: %s\n' "${unit}" >&2
    exit 20
  }
  if systemctl is-active --quiet "${unit}"; then
    printf 'ERROR: Snap LXD static service must be inactive before finalization: %s\n' "${unit}" >&2
    exit 20
  fi
done
systemctl is-active --quiet nvidia-persistenced.service || {
  printf 'ERROR: nvidia-persistenced is not active.\n' >&2
  exit 21
}

journalctl -k -b --no-pager > "${RUN_DIR}/kernel-current-boot.txt"
if critical_kernel_errors "${RUN_DIR}/kernel-current-boot.txt"; then
  printf 'ERROR: critical GPU/PCI error exists in the current boot.\n' >&2
  exit 22
fi
JOURNAL_CURSOR="$(journalctl -k -n 0 --show-cursor --no-pager | sed -n 's/^-- cursor: //p')"
[[ -n ${JOURNAL_CURSOR} ]] || {
  printf 'ERROR: unable to establish kernel journal cursor.\n' >&2
  exit 23
}

run_exact_four_gate "pre-restore" 0
PRECONDITIONS_VERIFIED=1
MUTATION_STARTED=1

# Fabric Manager was enabled+failed before this maintenance.  Try once while
# all workload allocators are still masked.  A clean failure is preserved as a
# documented degraded state; it must not perturb the exact-four GPU contract.
systemctl unmask "${FABRIC_UNIT}"
systemctl enable "${FABRIC_UNIT}"
set +e
timeout 60 systemctl start "${FABRIC_UNIT}"
fabric_rc=$?
set -e
if [[ ${fabric_rc} -eq 0 ]] && systemctl is-active --quiet "${FABRIC_UNIT}"; then
  FABRIC_OUTCOME="ACTIVE"
  allow_fabric=1
else
  systemctl status "${FABRIC_UNIT}" --no-pager -l > "${RUN_DIR}/fabric-manager-failure.txt" 2>&1 || true
  journalctl -u "${FABRIC_UNIT}" -b --no-pager > "${RUN_DIR}/fabric-manager-journal.txt" 2>&1 || true
  systemctl stop "${FABRIC_UNIT}" >/dev/null 2>&1 || true
  systemctl mask "${FABRIC_UNIT}"
  assert_masked_inactive "${FABRIC_UNIT}"
  FABRIC_OUTCOME="DEGRADED_MASKED"
  allow_fabric=0
fi
sleep 3
run_exact_four_gate "post-fabric-manager" "${allow_fabric}"
journalctl -k --after-cursor="${JOURNAL_CURSOR}" --no-pager \
  > "${RUN_DIR}/kernel-delta-post-fabric-manager.txt"
if critical_kernel_errors "${RUN_DIR}/kernel-delta-post-fabric-manager.txt"; then
  printf 'ERROR: Fabric Manager attempt produced a critical GPU/PCI error.\n' >&2
  exit 29
fi

for unit in "${SNAP_LXD_ENTRY_UNITS[@]}"; do
  systemctl enable "${unit}"
done
timeout 60 systemctl start "${SNAP_LXD_DAEMON_SOCKET}"
timeout 60 systemctl start "${SNAP_LXD_USER_SOCKET}"
if systemctl is-active --quiet "${SNAP_LXD_ACTIVATE}"; then
  printf 'ERROR: Snap LXD activate service unexpectedly became active.\n' >&2
  exit 30
fi
for unit in "${SNAP_LXD_DAEMON_SOCKET}" "${SNAP_LXD_USER_SOCKET}"; do
  systemctl is-active --quiet "${unit}" || {
    printf 'ERROR: restored Snap LXD socket is not active: %s\n' "${unit}" >&2
    exit 30
  }
done
for unit in "${SNAP_LXD_ENTRY_UNITS[@]}"; do
  [[ $(systemctl is-enabled "${unit}" 2>/dev/null || true) == enabled ]] || {
    printf 'ERROR: Snap LXD entry did not return to enabled state: %s\n' "${unit}" >&2
    exit 30
  }
done
for unit in "${SNAP_LXD_STATIC_SERVICES[@]}"; do
  [[ $(systemctl is-enabled "${unit}" 2>/dev/null || true) == static ]] || {
    printf 'ERROR: Snap LXD service lost static unit state: %s\n' "${unit}" >&2
    exit 30
  }
done

for unit in "${MASKED_ALLOCATOR_UNITS[@]}"; do
  systemctl unmask "${unit}"
  systemctl enable "${unit}"
  timeout 60 systemctl start "${unit}"
  systemctl is-active --quiet "${unit}" || {
    printf 'ERROR: restored unit is not active: %s\n' "${unit}" >&2
    exit 30
  }
done
sleep 5

docker ps -q > "${RUN_DIR}/docker-containers-release.txt"
[[ ! -s ${RUN_DIR}/docker-containers-release.txt ]] || {
  printf 'ERROR: a Docker container started during restoration.\n' >&2
  exit 31
}
ctr namespaces list -q > "${RUN_DIR}/containerd-namespaces-release.txt"
: > "${RUN_DIR}/containerd-tasks-release.txt"
while IFS= read -r namespace; do
  [[ -z ${namespace} ]] && continue
  ctr -n "${namespace}" tasks list -q >> "${RUN_DIR}/containerd-tasks-release.txt"
done < "${RUN_DIR}/containerd-namespaces-release.txt"
[[ ! -s ${RUN_DIR}/containerd-tasks-release.txt ]] || {
  printf 'ERROR: a containerd task started during restoration.\n' >&2
  exit 32
}
command -v lxc >/dev/null 2>&1 || {
  printf 'ERROR: lxc is unavailable after restoring its socket.\n' >&2
  exit 33
}
timeout 15 lxc list --format csv -c ns > "${RUN_DIR}/lxd-instances-release.txt"
if grep -E ',RUNNING$' "${RUN_DIR}/lxd-instances-release.txt" >/dev/null; then
  printf 'ERROR: an LXD instance started during restoration.\n' >&2
  exit 34
fi
if systemctl is-active --quiet "${SNAP_LXD_ACTIVATE}"; then
  printf 'ERROR: Snap LXD activate service did not preserve inactive state.\n' >&2
  exit 34
fi
for unit in "${SNAP_LXD_DAEMON_SOCKET}" "${SNAP_LXD_USER_SOCKET}"; do
  systemctl is-active --quiet "${unit}" || {
    printf 'ERROR: Snap LXD socket lost restored active state: %s\n' "${unit}" >&2
    exit 34
  }
done
for unit in "${SNAP_LXD_ENTRY_UNITS[@]}"; do
  [[ $(systemctl is-enabled "${unit}" 2>/dev/null || true) == enabled ]] || {
    printf 'ERROR: Snap LXD entry lost restored enablement: %s\n' "${unit}" >&2
    exit 34
  }
done
for unit in "${SNAP_LXD_STATIC_SERVICES[@]}"; do
  [[ $(systemctl is-enabled "${unit}" 2>/dev/null || true) == static ]] || {
    printf 'ERROR: Snap LXD service lost static unit state: %s\n' "${unit}" >&2
    exit 34
  }
done

run_exact_four_gate "release" "${allow_fabric}"
journalctl -k --after-cursor="${JOURNAL_CURSOR}" --no-pager > "${RUN_DIR}/kernel-delta-release.txt"
if critical_kernel_errors "${RUN_DIR}/kernel-delta-release.txt"; then
  printf 'ERROR: a critical GPU/PCI error appeared during restoration.\n' >&2
  exit 35
fi

BOOT_ID="$(cat /proc/sys/kernel/random/boot_id)"
env -i PATH=/usr/bin:/bin \
  /usr/bin/python3 -I - "${RUN_DIR}/administrator-response-final.json" \
    "${RUN_ID}" "${BOOT_ID}" "${FABRIC_OUTCOME}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

payload = {
    "schema_version": "stage0.gpu-uuid-exclusion-service-finalize.v1",
    "status": "PASS" if sys.argv[4] == "ACTIVE" else "PASS_WITH_FABRIC_MANAGER_DEGRADED",
    "run_id": sys.argv[2],
    "boot_id": sys.argv[3],
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "driver_version": "575.57.08",
    "fabric_manager": sys.argv[4],
    "allocator_services_restored": [
        "snap.lxd.activate.service", "snap.lxd.daemon.unix.socket",
        "snap.lxd.user-daemon.unix.socket", "containerd.service",
        "docker.socket", "docker.service",
    ],
    "gpu_contract": "four allowed GPUs visible; four exact UUIDs excluded",
    "minimum_postboot_uptime_seconds": 900,
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n")
PY

(
  cd "${RUN_DIR}"
  find . -maxdepth 1 -type f \
    ! -name 'admin-finalize.log' ! -name 'SHA256SUMS' \
    ! -name 'SUCCESS' ! -name 'FAILURE' -print0 \
    | sort -z | xargs -0 sha256sum
) > "${RUN_DIR}/SHA256SUMS"
find "${RUN_DIR}" -maxdepth 1 -type f ! -name 'admin-finalize.log' -exec chmod 0444 {} +
success_tmp="$(mktemp -p "${RUN_DIR}" .success.XXXXXXXX)"
{
  printf 'status=G0_G_UUID_EXCLUSION_SERVICE_FINALIZE_PASS\n'
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'boot_id=%s\n' "${BOOT_ID}"
  printf 'fabric_manager=%s\n' "${FABRIC_OUTCOME}"
  printf 'evidence=%s\n' "${RUN_DIR}"
} > "${success_tmp}"
chmod 0444 "${success_tmp}"
mv "${success_tmp}" "${RUN_DIR}/SUCCESS"
FINAL_PASS=1

printf 'G0_G_UUID_EXCLUSION_SERVICE_FINALIZE_PASS\n'
printf 'Fabric Manager outcome: %s\n' "${FABRIC_OUTCOME}"
printf 'Evidence: %s\n' "${RUN_DIR}"
