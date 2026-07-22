#!/usr/bin/env bash
set -Eeuo pipefail
umask 022
unset BASH_ENV ENV CDPATH
export PATH="/usr/sbin:/usr/bin:/sbin:/bin"

# Administrator-only, fail-closed implementation of the approved Stage 0
# four-GPU path B. The four excluded PCI devices are detached from the NVIDIA
# driver. Docker/containerd/LXD allocators and nvidia-persistenced are stopped
# during the maintenance window and restored before release. No user workload
# is terminated and no reset, reboot, driver reload, ECC clear, or PCI remove
# is performed.

readonly EXPECTED_HOST="sophgo13"
readonly PROJECT_USER="sophgo13"
readonly CANDIDATE_PYTHON="/home/sophgo13/cjl/storage/parameter-importance/envs/parameter-importance-stage0-1bd963c65f75/bin/python"
readonly ADMIN_ROOT="/var/lib/parameter-importance/stage0/g0-g-path-b"
readonly LOCK_DIR="/run/lock/parameter-importance"
readonly LOCK_PATH="${LOCK_DIR}/gpu-path-b.lock"
readonly EXPECTED_MODEL="NVIDIA A100-SXM4-80GB"

readonly -a ALL_BDFS=(
  "0000:4f:00.0" "0000:50:00.0" "0000:53:00.0" "0000:57:00.0"
  "0000:9c:00.0" "0000:9d:00.0" "0000:a0:00.0" "0000:a4:00.0"
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
readonly -a EXCLUDED_BDFS=(
  "0000:4f:00.0" "0000:50:00.0" "0000:53:00.0" "0000:57:00.0"
)
readonly -a SELECTED_DEVICE_NODES=(
  "/dev/nvidia4" "/dev/nvidia5" "/dev/nvidia6" "/dev/nvidia7"
)

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
declare -Ar EXPECTED_MINOR_BY_BDF=(
  ["0000:4f:00.0"]="0" ["0000:50:00.0"]="1"
  ["0000:53:00.0"]="2" ["0000:57:00.0"]="3"
  ["0000:9c:00.0"]="4" ["0000:9d:00.0"]="5"
  ["0000:a0:00.0"]="6" ["0000:a4:00.0"]="7"
)

RESUME_FROM=""
if (( $# == 0 )); then
  :
elif (( $# == 2 )) && [[ $1 == "--resume-from" ]]; then
  RESUME_FROM="$2"
  shift 2
else
  printf 'ERROR: usage: %s [--resume-from ROOT_OWNED_FAILED_RUN_DIR]\n' "$0" >&2
  exit 1
fi

if [[ ${EUID} -ne 0 ]]; then
  printf 'ERROR: interactive sudo/root is required.\n' >&2
  exit 2
fi
if [[ $(hostname -s) != "${EXPECTED_HOST}" ]]; then
  printf 'ERROR: expected host %s, got %s.\n' "${EXPECTED_HOST}" "$(hostname -s)" >&2
  exit 3
fi
for required in flock fuser lspci nvidia-smi runuser /usr/bin/python3; do
  if ! command -v "${required}" >/dev/null 2>&1; then
    printf 'ERROR: required command is unavailable: %s\n' "${required}" >&2
    exit 4
  fi
done
if [[ ! -x ${CANDIDATE_PYTHON} ]]; then
  printf 'ERROR: candidate Python is unavailable.\n' >&2
  exit 5
fi

if [[ -e ${LOCK_DIR} || -L ${LOCK_DIR} ]]; then
  if [[ ! -d ${LOCK_DIR} || -L ${LOCK_DIR} \
        || $(stat -c '%u:%g:%a' -- "${LOCK_DIR}") != "0:0:700" ]]; then
    printf 'ERROR: maintenance lock directory is not a trusted root-owned directory.\n' >&2
    exit 6
  fi
else
  mkdir -m 0700 "${LOCK_DIR}"
  chown root:root "${LOCK_DIR}"
fi
if [[ -e ${LOCK_PATH} || -L ${LOCK_PATH} ]]; then
  if [[ ! -f ${LOCK_PATH} || -L ${LOCK_PATH} \
        || $(stat -c '%u:%g' -- "${LOCK_PATH}") != "0:0" ]]; then
    printf 'ERROR: maintenance lock file is not a trusted root-owned regular file.\n' >&2
    exit 7
  fi
fi
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  printf 'ERROR: another GPU maintenance process holds %s.\n' "${LOCK_PATH}" >&2
  exit 8
fi

if [[ -n ${RESUME_FROM} ]]; then
  if [[ -L ${RESUME_FROM} || ! -d ${RESUME_FROM} ]]; then
    printf 'ERROR: resume source must be an existing non-symlink directory.\n' >&2
    exit 9
  fi
  RESUME_FROM="$(readlink -e -- "${RESUME_FROM}")"
  if [[ $(dirname -- "${RESUME_FROM}") != "${ADMIN_ROOT}" \
        || $(stat -c '%u:%g:%a' -- "${RESUME_FROM}") != "0:0:755" ]]; then
    printf 'ERROR: resume source is outside the trusted root evidence directory or has unsafe ownership/mode.\n' >&2
    exit 10
  fi
  for source_file in pre-change-identity.txt pre-change-device-nodes.txt FAILURE; do
    source_path="${RESUME_FROM}/${source_file}"
    if [[ -L ${source_path} || ! -f ${source_path} \
          || $(stat -c '%u:%g' -- "${source_path}") != "0:0" ]]; then
      printf 'ERROR: trusted resume evidence is missing or unsafe: %s\n' "${source_path}" >&2
      exit 11
    fi
  done
fi
readonly RESUME_FROM

install -d -o root -g root -m 0755 "${ADMIN_ROOT}"
readonly RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$(mktemp -d -p "${ADMIN_ROOT}" "admin-${RUN_ID}.XXXXXXXX")"
readonly RUN_DIR
chmod 0755 "${RUN_DIR}"
readonly LOG_PATH="${RUN_DIR}/admin-maintenance.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

declare -A NODE_MODE=()
declare -A NODE_UID=()
declare -A NODE_GID=()
GPU_NODES=()
PERSISTENCED_WAS_ACTIVE=0
PERSISTENCED_RESTORED=0
INFRA_SERVICES_RESTORED=0
PERMISSIONS_CAPTURED=0
PERMISSIONS_RESTORED=0
MUTATION_STARTED=0
ISOLATION_VERIFIED=0
VALIDATION_PASSED=0
READY_TO_RELEASE=0
SAFE_HOLD_VERIFIED=0
PRECHANGE_RESTORE_VERIFIED=0
FAILED_LINE=""
FAILED_COMMAND=""
INFRA_SERVICES_WERE_ACTIVE=()

record_error() {
  FAILED_LINE="$1"
  FAILED_COMMAND="$2"
}

restore_node() {
  local node="$1"
  if [[ -e ${node} && -n ${NODE_MODE[${node}]+x} ]]; then
    chown "${NODE_UID[${node}]}:${NODE_GID[${node}]}" -- "${node}"
    chmod "${NODE_MODE[${node}]}" -- "${node}"
  fi
}

restore_selected_access() {
  local node
  for node in \
    /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools \
    /dev/nvidia-modeset /dev/nvidia-nvswitchctl; do
    restore_node "${node}"
  done
  for node in "${SELECTED_DEVICE_NODES[@]}"; do
    restore_node "${node}"
  done
  if [[ -d /dev/nvidia-caps ]]; then
    for node in /dev/nvidia-caps/*; do
      [[ -e ${node} ]] && restore_node "${node}"
    done
  fi
  PERMISSIONS_RESTORED=1
}

lxd_daemon_active() {
  systemctl is-active --quiet lxd.service \
    || systemctl is-active --quiet snap.lxd.daemon.service \
    || pgrep -f '(^|/)(lxd|lxcfs)([[:space:]]|$)' >/dev/null
}

restore_all_access() {
  local node
  for node in "${GPU_NODES[@]}"; do
    restore_node "${node}"
  done
  PERMISSIONS_RESTORED=1
}

stop_infrastructure_service_if_active() {
  local unit="$1"
  if systemctl is-active --quiet "${unit}"; then
    INFRA_SERVICES_WERE_ACTIVE+=("${unit}")
    systemctl stop "${unit}"
    if systemctl is-active --quiet "${unit}"; then
      printf 'ERROR: infrastructure service did not stop: %s\n' "${unit}" >&2
      return 1
    fi
  fi
}

restore_infrastructure_services() {
  local index unit
  for ((index=${#INFRA_SERVICES_WERE_ACTIVE[@]} - 1; index >= 0; index--)); do
    unit="${INFRA_SERVICES_WERE_ACTIVE[${index}]}"
    systemctl start "${unit}"
    if ! systemctl is-active --quiet "${unit}"; then
      printf 'ERROR: infrastructure service did not restart: %s\n' "${unit}" >&2
      return 1
    fi
  done
  INFRA_SERVICES_RESTORED=1
}

enter_gpu_safe_hold() {
  local node unit ok=1
  for unit in "${INFRA_SERVICES_WERE_ACTIVE[@]}"; do
    systemctl stop "${unit}" >/dev/null 2>&1 || ok=0
    systemctl is-active --quiet "${unit}" && ok=0
  done
  systemctl stop nvidia-persistenced >/dev/null 2>&1 || true
  systemctl is-active --quiet nvidia-persistenced && ok=0
  for node in "${GPU_NODES[@]}"; do
    if [[ -e ${node} ]]; then
      chmod 0600 -- "${node}" || ok=0
      [[ $(stat -c '%a' -- "${node}" 2>/dev/null) == 600 ]] || ok=0
    fi
  done
  SAFE_HOLD_VERIFIED=${ok}
  [[ ${ok} -eq 1 ]]
}

verify_prechange_state() {
  local node unit ok=1
  for node in "${GPU_NODES[@]}"; do
    if [[ -e ${node} ]]; then
      [[ $(stat -c '%a:%u:%g' -- "${node}" 2>/dev/null) \
          == "${NODE_MODE[${node}]}:${NODE_UID[${node}]}:${NODE_GID[${node}]}" ]] || ok=0
    fi
  done
  if [[ ${PERSISTENCED_WAS_ACTIVE} -eq 1 ]]; then
    systemctl is-active --quiet nvidia-persistenced || ok=0
  else
    systemctl is-active --quiet nvidia-persistenced && ok=0
  fi
  for unit in "${INFRA_SERVICES_WERE_ACTIVE[@]}"; do
    systemctl is-active --quiet "${unit}" || ok=0
  done
  PRECHANGE_RESTORE_VERIFIED=${ok}
  [[ ${ok} -eq 1 ]]
}

publish_failure() {
  local rc="$1"
  rm -f "${RUN_DIR}/SUCCESS"
  local tmp
  tmp="$(mktemp -p "${RUN_DIR}" .failure.XXXXXXXX)"
  {
    printf 'status=FAILED\n'
    printf 'exit_code=%s\n' "${rc}"
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'failed_line=%s\n' "${FAILED_LINE}"
    printf 'failed_command=%s\n' "${FAILED_COMMAND}"
    if [[ ${MUTATION_STARTED} -eq 1 && ${READY_TO_RELEASE} -eq 0 \
          && ${SAFE_HOLD_VERIFIED} -eq 1 ]]; then
      printf 'safety_state=SAFE_HOLD_ALL_GPU_DEVICE_NODES_ROOT_ONLY_PERSISTENCED_STOPPED\n'
    elif [[ ${MUTATION_STARTED} -eq 1 && ${READY_TO_RELEASE} -eq 0 ]]; then
      printf 'safety_state=SAFE_HOLD_ATTEMPTED_BUT_NOT_VERIFIED\n'
    elif [[ ${READY_TO_RELEASE} -eq 1 ]]; then
      printf 'safety_state=EXCLUDED_DEVICES_UNBOUND_SELECTED_DEVICES_VALIDATED\n'
    elif [[ ${PRECHANGE_RESTORE_VERIFIED} -eq 1 ]]; then
      printf 'safety_state=PRECHANGE_ACCESS_RESTORED\n'
    else
      printf 'safety_state=PRECHANGE_RESTORE_ATTEMPTED_BUT_NOT_VERIFIED\n'
    fi
  } > "${tmp}"
  chmod 0444 "${tmp}"
  mv -f "${tmp}" "${RUN_DIR}/FAILURE"
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ ${rc} -ne 0 ]]; then
    if [[ ${MUTATION_STARTED} -eq 0 && ${PERMISSIONS_CAPTURED} -eq 1 ]]; then
      restore_all_access || true
      if [[ ${PERSISTENCED_WAS_ACTIVE} -eq 1 ]]; then
        systemctl start nvidia-persistenced || true
      fi
      restore_infrastructure_services || true
      verify_prechange_state || true
    elif [[ ${MUTATION_STARTED} -eq 1 && ${READY_TO_RELEASE} -eq 0 ]]; then
      enter_gpu_safe_hold || true
    elif [[ ${READY_TO_RELEASE} -eq 1 ]]; then
      [[ ${PERMISSIONS_RESTORED} -eq 1 ]] || restore_selected_access || true
      [[ ${INFRA_SERVICES_RESTORED} -eq 1 ]] || restore_infrastructure_services || true
    fi
    publish_failure "${rc}" || true
  fi
  exit "${rc}"
}

trap 'record_error "${LINENO}" "${BASH_COMMAND}"' ERR
trap on_exit EXIT

printf 'Stage 0 path B maintenance start: %s\n' "$(date --iso-8601=seconds)"
printf 'Run directory: %s\n' "${RUN_DIR}"
printf 'Executor: %s\n' "$(id)"
if [[ -n ${RESUME_FROM} ]]; then
  printf 'Resume source: %s\n' "${RESUME_FROM}"
fi

# Establish the complete immutable BDF -> UUID -> minor -> model contract before
# changing device permissions, services, or driver bindings.
{
  printf 'captured_at=%s\n' "$(date --iso-8601=seconds)"
  lspci -Dnn | grep -i NVIDIA || true
} > "${RUN_DIR}/pre-change-pci.txt"

if [[ -n ${RESUME_FROM} ]]; then
  if [[ $(wc -l < "${RESUME_FROM}/pre-change-identity.txt") -ne ${#ALL_BDFS[@]} ]]; then
    printf 'ERROR: resume identity evidence does not contain exactly eight records.\n' >&2
    exit 19
  fi
  for bdf in "${ALL_BDFS[@]}"; do
    expected_identity="${bdf}|${EXPECTED_UUID_BY_BDF[${bdf}]}|${EXPECTED_MINOR_BY_BDF[${bdf}]}|${EXPECTED_MODEL}"
    if ! grep -Fqx -- "${expected_identity}" "${RESUME_FROM}/pre-change-identity.txt"; then
      printf 'ERROR: resume identity evidence does not match the approved contract: %s\n' "${bdf}" >&2
      exit 19
    fi
  done
  install -o root -g root -m 0444 \
    "${RESUME_FROM}/pre-change-identity.txt" \
    "${RUN_DIR}/resume-source-identity.txt"
fi

for bdf in "${ALL_BDFS[@]}"; do
  info="/proc/driver/nvidia/gpus/${bdf}/information"
  driver="/sys/bus/pci/devices/${bdf}/driver"
  if [[ ! -e "/sys/bus/pci/devices/${bdf}" ]]; then
    printf 'ERROR: expected PCI inventory entry is missing: %s\n' "${bdf}" >&2
    exit 20
  fi
  if [[ ! -L ${driver} ]]; then
    if [[ -n ${RESUME_FROM} && ( ${bdf} == "0000:4f:00.0" || ${bdf} == "0000:50:00.0" ) ]]; then
      printf '%s|%s|%s|%s\n' "${bdf}" "${EXPECTED_UUID_BY_BDF[${bdf}]}" \
        "${EXPECTED_MINOR_BY_BDF[${bdf}]}" "${EXPECTED_MODEL}" \
        >> "${RUN_DIR}/pre-change-identity.txt"
      continue
    fi
    printf 'ERROR: expected bound NVIDIA inventory entry is missing: %s\n' "${bdf}" >&2
    exit 20
  fi
  if [[ ! -r ${info} ]]; then
    printf 'ERROR: bound NVIDIA identity record is missing: %s\n' "${bdf}" >&2
    exit 20
  fi
  if [[ $(basename "$(readlink -f "${driver}")") != nvidia ]]; then
    printf 'ERROR: %s is not bound to the NVIDIA driver.\n' "${bdf}" >&2
    exit 21
  fi
  actual_model="$(sed -n 's/^Model:[[:space:]]*//p' "${info}")"
  actual_uuid="$(sed -n 's/^GPU UUID:[[:space:]]*//p' "${info}")"
  actual_minor="$(sed -n 's/^Device Minor:[[:space:]]*//p' "${info}")"
  if [[ ${actual_model} != "${EXPECTED_MODEL}" \
        || ${actual_uuid} != "${EXPECTED_UUID_BY_BDF[${bdf}]}" \
        || ${actual_minor} != "${EXPECTED_MINOR_BY_BDF[${bdf}]}" ]]; then
    printf 'ERROR: identity contract mismatch for %s.\n' "${bdf}" >&2
    exit 22
  fi
  printf '%s|%s|%s|%s\n' "${bdf}" "${actual_uuid}" "${actual_minor}" "${actual_model}" \
    >> "${RUN_DIR}/pre-change-identity.txt"
done

if [[ -n ${RESUME_FROM} ]]; then
  for bdf in "0000:4f:00.0" "0000:50:00.0"; do
    if [[ -L "/sys/bus/pci/devices/${bdf}/driver" ]]; then
      printf 'ERROR: resume accepts only the state where %s is already unbound.\n' "${bdf}" >&2
      exit 22
    fi
  done
  for bdf in "0000:53:00.0" "0000:57:00.0" "${ALLOWED_BDFS[@]}"; do
    if [[ ! -L "/sys/bus/pci/devices/${bdf}/driver" \
          || $(basename "$(readlink -f "/sys/bus/pci/devices/${bdf}/driver")") != nvidia ]]; then
      printf 'ERROR: resume accepts only the state where %s remains bound to NVIDIA.\n' "${bdf}" >&2
      exit 22
    fi
  done
fi

if [[ -z ${RESUME_FROM} ]] && command -v docker >/dev/null 2>&1; then
  if ! docker ps -q > "${RUN_DIR}/docker-containers.txt"; then
    printf 'ERROR: Docker inventory failed.\n' >&2
    exit 23
  fi
  if [[ -s "${RUN_DIR}/docker-containers.txt" ]]; then
    printf 'ERROR: running Docker containers were detected.\n' >&2
    exit 24
  fi
fi
if pgrep -f '(^|/)nvidia-cuda-mps-control([[:space:]]|$)' > "${RUN_DIR}/mps-pids.txt"; then
  printf 'ERROR: NVIDIA MPS is active.\n' >&2
  exit 25
fi
if pgrep -x nv-hostengine > "${RUN_DIR}/dcgm-pids.txt"; then
  printf 'ERROR: DCGM host engine is active.\n' >&2
  exit 26
fi
if systemctl is-active --quiet nvidia-fabricmanager \
  || pgrep -f '(^|/)(nv-fabricmanager|nvidia-fabricmanager)([[:space:]]|$)' \
    > "${RUN_DIR}/fabric-manager-pids.txt"; then
  printf 'ERROR: NVIDIA Fabric Manager is active; this runbook does not alter a live fabric.\n' >&2
  exit 27
fi
if [[ -z ${RESUME_FROM} ]] && command -v ctr >/dev/null 2>&1; then
  if ! ctr namespaces list -q > "${RUN_DIR}/containerd-namespaces.txt"; then
    printf 'ERROR: containerd namespace inventory failed.\n' >&2
    exit 28
  fi
  while IFS= read -r namespace; do
    [[ -z ${namespace} ]] && continue
    if ! ctr -n "${namespace}" tasks list -q >> "${RUN_DIR}/containerd-tasks.txt"; then
      printf 'ERROR: containerd task inventory failed.\n' >&2
      exit 29
    fi
  done < "${RUN_DIR}/containerd-namespaces.txt"
  if [[ -s "${RUN_DIR}/containerd-tasks.txt" ]]; then
    printf 'ERROR: running containerd tasks were detected.\n' >&2
    exit 30
  fi
elif [[ -n ${RESUME_FROM} ]]; then
  printf 'SKIPPED_RESUME_SAFE_HOLD\n' > "${RUN_DIR}/containerd-namespaces.txt"
  : > "${RUN_DIR}/containerd-tasks.txt"
fi
if [[ -n ${RESUME_FROM} ]]; then
  printf 'SKIPPED_RESUME_SAFE_HOLD\n' > "${RUN_DIR}/lxd-instances.txt"
elif lxd_daemon_active; then
  if ! command -v lxc >/dev/null 2>&1; then
    printf 'ERROR: an LXD daemon is active but the lxc client is unavailable.\n' >&2
    exit 31
  fi
  if ! timeout 10 lxc list --format csv -c ns > "${RUN_DIR}/lxd-instances.txt"; then
    printf 'ERROR: LXD instance inventory failed.\n' >&2
    exit 32
  fi
  if grep -E ',RUNNING$' "${RUN_DIR}/lxd-instances.txt" >/dev/null; then
    printf 'ERROR: a running LXD instance was detected.\n' >&2
    exit 33
  fi
else
  printf 'SKIPPED_NO_ACTIVE_LXD_DAEMON\n' > "${RUN_DIR}/lxd-instances.txt"
fi

shopt -s nullglob
GPU_NODES=(
  /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools
  /dev/nvidia-modeset /dev/nvidia-nvswitchctl /dev/nvidia-caps/*
)
shopt -u nullglob
if (( ${#GPU_NODES[@]} == 0 )); then
  printf 'ERROR: no NVIDIA device nodes were found.\n' >&2
  exit 33
fi
for node in "${GPU_NODES[@]}"; do
  NODE_MODE["${node}"]="$(stat -c '%a' -- "${node}")"
  NODE_UID["${node}"]="$(stat -c '%u' -- "${node}")"
  NODE_GID["${node}"]="$(stat -c '%g' -- "${node}")"
  printf '%s|%s|%s|%s\n' "${node}" "${NODE_MODE[${node}]}" "${NODE_UID[${node}]}" "${NODE_GID[${node}]}" \
    >> "${RUN_DIR}/pre-change-device-nodes.txt"
done
PERMISSIONS_CAPTURED=1

# Deny new user-space GPU opens before checking for existing clients. This is
# the authoritative maintenance barrier in the absence of a site scheduler.
for node in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools "${GPU_NODES[@]}"; do
  [[ -e ${node} ]] && chmod 0600 -- "${node}"
done

# Keep container allocators stopped for the entire mutation and validation
# window so that no privileged daemon can race the device-client check.
stop_infrastructure_service_if_active docker.service
stop_infrastructure_service_if_active docker.socket
stop_infrastructure_service_if_active containerd.service
stop_infrastructure_service_if_active lxd.service
stop_infrastructure_service_if_active lxd.socket
stop_infrastructure_service_if_active snap.lxd.daemon.service
stop_infrastructure_service_if_active snap.lxd.daemon.unix.socket

if systemctl is-active --quiet nvidia-persistenced; then
  PERSISTENCED_WAS_ACTIVE=1
  systemctl stop nvidia-persistenced
fi
sleep 2

set +e
fuser "${GPU_NODES[@]}" > "${RUN_DIR}/fuser.stdout" 2> "${RUN_DIR}/fuser.stderr"
fuser_rc=$?
set -e
if [[ ${fuser_rc} -eq 0 ]]; then
  printf 'ERROR: GPU device clients remain after the maintenance barrier.\n' >&2
  cat "${RUN_DIR}/fuser.stdout" >&2
  exit 34
elif [[ ${fuser_rc} -ne 1 || -s "${RUN_DIR}/fuser.stdout" || -s "${RUN_DIR}/fuser.stderr" ]]; then
  printf 'ERROR: fuser failed or returned an unexpected result (rc=%s).\n' "${fuser_rc}" >&2
  cat "${RUN_DIR}/fuser.stderr" >&2
  exit 35
fi

MUTATION_CURSOR="$(journalctl -k -n 0 --show-cursor --no-pager | sed -n 's/^-- cursor: //p')"
if [[ -z ${MUTATION_CURSOR} ]]; then
  printf 'ERROR: unable to establish the pre-mutation kernel journal cursor.\n' >&2
  exit 36
fi

MUTATION_STARTED=1
for bdf in "${EXCLUDED_BDFS[@]}"; do
  printf 'Detaching excluded PCI device: %s (%s)\n' "${bdf}" "${EXPECTED_UUID_BY_BDF[${bdf}]}"
  printf '%s\n' "${bdf}" > /sys/bus/pci/drivers/nvidia/unbind
  if [[ -L "/sys/bus/pci/devices/${bdf}/driver" ]]; then
    printf 'ERROR: excluded device remained bound: %s\n' "${bdf}" >&2
    exit 37
  fi
done

for bdf in "${ALLOWED_BDFS[@]}"; do
  driver="/sys/bus/pci/devices/${bdf}/driver"
  if [[ ! -L ${driver} || $(basename "$(readlink -f "${driver}")") != nvidia ]]; then
    printf 'ERROR: approved device lost its NVIDIA binding: %s\n' "${bdf}" >&2
    exit 38
  fi
done
for bdf in "${EXCLUDED_BDFS[@]}"; do
  if [[ -L "/sys/bus/pci/devices/${bdf}/driver" ]]; then
    printf 'ERROR: excluded device is not isolated: %s\n' "${bdf}" >&2
    exit 39
  fi
done
ISOLATION_VERIFIED=1

journalctl -k --after-cursor="${MUTATION_CURSOR}" --no-pager \
  > "${RUN_DIR}/kernel-delta-isolation.txt"
if grep -Ei 'AER|NVRM:.*(9c:00|9d:00|a0:00|a4:00)' "${RUN_DIR}/kernel-delta-isolation.txt"; then
  printf 'ERROR: the isolation change affected an approved GPU or produced PCI AER.\n' >&2
  exit 40
fi
JOURNAL_CURSOR="$(journalctl -k -n 0 --show-cursor --no-pager | sed -n 's/^-- cursor: //p')"
if [[ -z ${JOURNAL_CURSOR} ]]; then
  printf 'ERROR: unable to establish the validation journal cursor.\n' >&2
  exit 41
fi
OBSERVATION_STARTED_AT_EPOCH="$(date +%s)"

restore_selected_access
if [[ ${PERSISTENCED_WAS_ACTIVE} -eq 1 ]]; then
  systemctl start nvidia-persistenced
  if ! systemctl is-active --quiet nvidia-persistenced; then
    printf 'ERROR: nvidia-persistenced did not return to active state.\n' >&2
    exit 42
  fi
  PERSISTENCED_RESTORED=1
fi
sleep 3

nvidia-smi \
  --query-gpu=pci.bus_id,uuid,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total,memory.used,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits > "${RUN_DIR}/nvml-first.csv"
nvidia-smi -q -d ROW_REMAPPER > "${RUN_DIR}/row-remapper-first.txt"
nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader \
  > "${RUN_DIR}/compute-apps-first.csv"
nvidia-smi -q -x > "${RUN_DIR}/nvidia-smi-first.xml"

if [[ -s "${RUN_DIR}/compute-apps-first.csv" ]]; then
  printf 'ERROR: a compute process appeared during first-snapshot validation.\n' >&2
  exit 43
fi

ALLOWED_MAP_VALUE="$(
  for index in "${!ALLOWED_BDFS[@]}"; do
    printf '%s=%s\n' "${ALLOWED_BDFS[${index}]}" "${ALLOWED_UUIDS[${index}]}"
  done
)"

validate_health_snapshot() {
  local nvml_csv="$1"
  local row_remapper="$2"
  env -i PATH=/usr/bin:/bin ALLOWED_MAP="${ALLOWED_MAP_VALUE}" \
    /usr/bin/python3 -I - "${nvml_csv}" "${row_remapper}" <<'PY'
import csv
import os
import pathlib
import re
import sys


def normalize_pci(value: str) -> str:
    fields = value.strip().lower().split(":")
    if len(fields) != 3:
        raise ValueError(f"invalid PCI address: {value!r}")
    return f"{fields[0][-4:]}:{fields[1]}:{fields[2]}"


allowed = dict(line.split("=", 1) for line in os.environ["ALLOWED_MAP"].splitlines())
rows = list(csv.reader(pathlib.Path(sys.argv[1]).read_text().splitlines()))
if len(rows) != 4:
    raise SystemExit(f"NVML expected 4 devices, observed {len(rows)}")
observed: dict[str, str] = {}
for row in rows:
    if len(row) != 7:
        raise SystemExit(f"unexpected NVML row: {row!r}")
    pci, uuid, volatile_uce, aggregate_uce, memory_used, utilization, temperature = (
        item.strip() for item in row
    )
    pci = normalize_pci(pci)
    if pci in observed:
        raise SystemExit(f"duplicate NVML PCI address: {pci}")
    observed[pci] = uuid
    if volatile_uce != "0" or aggregate_uce != "0":
        raise SystemExit(f"ECC gate failed for {pci}: volatile={volatile_uce}, aggregate={aggregate_uce}")
    if memory_used != "0" or utilization != "0":
        raise SystemExit(f"activity gate failed for {pci}: memory={memory_used}, utilization={utilization}")
    if not temperature.isdigit() or int(temperature) >= 85:
        raise SystemExit(f"temperature gate failed for {pci}: {temperature}")
if observed != allowed:
    raise SystemExit(f"NVML PCI/UUID mapping mismatch: {observed!r}")

text = pathlib.Path(sys.argv[2]).read_text()
blocks = re.split(r"(?=^GPU\s+[0-9A-Fa-f:.]+\s*$)", text, flags=re.MULTILINE)
row_state: dict[str, tuple[str, str]] = {}
for block in blocks:
    match = re.search(r"^GPU\s+([0-9A-Fa-f:.]+)\s*$", block, flags=re.MULTILINE)
    if not match:
        continue
    pci = normalize_pci(match.group(1))
    pending = re.search(r"^\s*Pending\s*:\s*(\S+)\s*$", block, flags=re.MULTILINE)
    failure = re.search(r"^\s*Remapping Failure Occurred\s*:\s*(\S+)\s*$", block, flags=re.MULTILINE)
    if not pending or not failure:
        raise SystemExit(f"incomplete row-remap state for {pci}")
    row_state[pci] = (pending.group(1), failure.group(1))
if set(row_state) != set(allowed):
    raise SystemExit(f"row-remap PCI set mismatch: {sorted(row_state)!r}")
for pci, state in row_state.items():
    if state != ("No", "No"):
        raise SystemExit(f"row-remap gate failed for {pci}: {state!r}")
PY
}

validate_health_snapshot "${RUN_DIR}/nvml-first.csv" "${RUN_DIR}/row-remapper-first.txt"

validate_pytorch_enumeration() {
  local output_path="$1"
  runuser -u "${PROJECT_USER}" -- env -i \
    HOME="/home/${PROJECT_USER}" USER="${PROJECT_USER}" LOGNAME="${PROJECT_USER}" \
    PATH="$(dirname "${CANDIDATE_PYTHON}"):/usr/bin:/bin" \
    PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    EXPECTED_UUIDS="$(IFS=,; printf '%s' "${ALLOWED_UUIDS[*]}")" \
    "${CANDIDATE_PYTHON}" -I - > "${output_path}" <<'PY'
import json
import os
import torch

allowed = os.environ["EXPECTED_UUIDS"].split(",")
count = torch.cuda.device_count()
if count != 4:
    raise SystemExit(f"PyTorch expected 4 CUDA devices, observed {count}")
devices = []
observed = []
for index in range(count):
    properties = torch.cuda.get_device_properties(index)
    raw_uuid = getattr(properties, "uuid", None)
    if isinstance(raw_uuid, bytes):
        raw_uuid = raw_uuid.decode("ascii")
    uuid = str(raw_uuid or "")
    observed.append(uuid)
    devices.append(
        {
            "index": index,
            "name": properties.name,
            "total_memory": properties.total_memory,
            "uuid": uuid,
        }
    )
if observed != allowed:
    raise SystemExit(f"PyTorch UUID mapping mismatch: {observed!r}")
print(json.dumps({"device_count": count, "devices": devices}, indent=2))
PY
}

capture_health_snapshot() {
  local prefix="$1"
  nvidia-smi \
    --query-gpu=pci.bus_id,uuid,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total,memory.used,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits > "${RUN_DIR}/nvml-${prefix}.csv"
  nvidia-smi -q -d ROW_REMAPPER > "${RUN_DIR}/row-remapper-${prefix}.txt"
  nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader \
    > "${RUN_DIR}/compute-apps-${prefix}.csv"
  nvidia-smi -q -x > "${RUN_DIR}/nvidia-smi-${prefix}.xml"
  if [[ -s "${RUN_DIR}/compute-apps-${prefix}.csv" ]]; then
    printf 'ERROR: a compute process appeared during snapshot %s.\n' "${prefix}" >&2
    return 1
  fi
  validate_health_snapshot \
    "${RUN_DIR}/nvml-${prefix}.csv" \
    "${RUN_DIR}/row-remapper-${prefix}.txt"
}

validate_final_bindings() {
  local output_path="$1"
  local bdf driver info
  {
    printf 'captured_at=%s\n' "$(date --iso-8601=seconds)"
    for bdf in "${ALL_BDFS[@]}"; do
      printf '%s|' "${bdf}"
      if [[ -L "/sys/bus/pci/devices/${bdf}/driver" ]]; then
        basename "$(readlink -f "/sys/bus/pci/devices/${bdf}/driver")"
      else
        printf 'UNBOUND\n'
      fi
    done
  } > "${output_path}"
  for bdf in "${ALLOWED_BDFS[@]}"; do
    driver="/sys/bus/pci/devices/${bdf}/driver"
    info="/proc/driver/nvidia/gpus/${bdf}/information"
    if [[ ! -L ${driver} || $(basename "$(readlink -f "${driver}")") != nvidia \
          || ! -r ${info} \
          || $(sed -n 's/^GPU UUID:[[:space:]]*//p' "${info}") != "${EXPECTED_UUID_BY_BDF[${bdf}]}" ]]; then
      printf 'ERROR: approved-device identity check failed: %s\n' "${bdf}" >&2
      return 1
    fi
  done
  for bdf in "${EXCLUDED_BDFS[@]}"; do
    if [[ -L "/sys/bus/pci/devices/${bdf}/driver" ]]; then
      printf 'ERROR: an excluded device was rebound: %s\n' "${bdf}" >&2
      return 1
    fi
  done
}

validate_pytorch_enumeration "${RUN_DIR}/pytorch-first.json"

capture_health_snapshot "after-pytorch"

journalctl -k --after-cursor="${JOURNAL_CURSOR}" --no-pager \
  > "${RUN_DIR}/kernel-delta-first.txt"
if grep -E 'NVRM:.*Xid|RmInitAdapter|AER' "${RUN_DIR}/kernel-delta-first.txt"; then
  printf 'ERROR: a new GPU/PCI kernel error appeared during first-snapshot validation.\n' >&2
  exit 43
fi
validate_final_bindings "${RUN_DIR}/driver-bindings-first.txt"

# Complete the mandatory 15-minute observation while privileged allocators
# remain stopped. The elapsed calculation includes the first/PyTorch snapshots.
elapsed_seconds=$(( $(date +%s) - OBSERVATION_STARTED_AT_EPOCH ))
remaining_seconds=$(( 900 - elapsed_seconds ))
if (( remaining_seconds > 0 )); then
  printf 'Observation window active; waiting %s seconds before the final snapshot.\n' "${remaining_seconds}"
  sleep "${remaining_seconds}"
fi

capture_health_snapshot "observation-final"
validate_pytorch_enumeration "${RUN_DIR}/pytorch-observation-final.json"
capture_health_snapshot "observation-final-after-pytorch"
validate_final_bindings "${RUN_DIR}/driver-bindings-observation-final.txt"
journalctl -k --after-cursor="${JOURNAL_CURSOR}" --no-pager \
  > "${RUN_DIR}/kernel-delta-observation.txt"
if grep -E 'NVRM:.*Xid|RmInitAdapter|AER' "${RUN_DIR}/kernel-delta-observation.txt"; then
  printf 'ERROR: a new GPU/PCI kernel error appeared during the observation window.\n' >&2
  exit 44
fi

# Restore allocators, then run one final release snapshot so restart-policy or
# privileged-daemon side effects cannot escape validation.
POST_RESTORE_CURSOR="$(journalctl -k -n 0 --show-cursor --no-pager | sed -n 's/^-- cursor: //p')"
if [[ -z ${POST_RESTORE_CURSOR} ]]; then
  printf 'ERROR: unable to establish the service-restore journal cursor.\n' >&2
  exit 45
fi
restore_infrastructure_services
sleep 3

if command -v docker >/dev/null 2>&1; then
  docker ps -q > "${RUN_DIR}/docker-containers-release.txt"
  if [[ -s "${RUN_DIR}/docker-containers-release.txt" ]]; then
    printf 'ERROR: a Docker container started during service restoration.\n' >&2
    exit 46
  fi
fi
if command -v ctr >/dev/null 2>&1; then
  ctr namespaces list -q > "${RUN_DIR}/containerd-namespaces-release.txt"
  while IFS= read -r namespace; do
    [[ -z ${namespace} ]] && continue
    ctr -n "${namespace}" tasks list -q >> "${RUN_DIR}/containerd-tasks-release.txt"
  done < "${RUN_DIR}/containerd-namespaces-release.txt"
  if [[ -s "${RUN_DIR}/containerd-tasks-release.txt" ]]; then
    printf 'ERROR: a containerd task started during service restoration.\n' >&2
    exit 47
  fi
fi
if lxd_daemon_active; then
  if ! command -v lxc >/dev/null 2>&1; then
    printf 'ERROR: an LXD daemon became active without an available lxc client.\n' >&2
    exit 48
  fi
  timeout 10 lxc list --format csv -c ns > "${RUN_DIR}/lxd-instances-release.txt"
  if grep -E ',RUNNING$' "${RUN_DIR}/lxd-instances-release.txt" >/dev/null; then
    printf 'ERROR: an LXD instance started during service restoration.\n' >&2
    exit 49
  fi
else
  printf 'SKIPPED_NO_ACTIVE_LXD_DAEMON\n' > "${RUN_DIR}/lxd-instances-release.txt"
fi

capture_health_snapshot "release"
validate_pytorch_enumeration "${RUN_DIR}/pytorch-release.json"
capture_health_snapshot "release-after-pytorch"
validate_final_bindings "${RUN_DIR}/driver-bindings-release.txt"

set +e
fuser "${GPU_NODES[@]}" > "${RUN_DIR}/fuser-release.stdout" 2> "${RUN_DIR}/fuser-release.stderr"
fuser_release_rc=$?
set -e
if [[ ${fuser_release_rc} -eq 0 ]]; then
  for token in $(cat "${RUN_DIR}/fuser-release.stdout"); do
    pid="${token//[^0-9]/}"
    if [[ -z ${pid} ]]; then
      printf 'ERROR: unable to parse final fuser token: %s\n' "${token}" >&2
      exit 49
    fi
    comm="$(ps -o comm= -p "${pid}" | xargs)"
    if [[ ${comm} != nvidia-persiste* ]]; then
      printf 'ERROR: unexpected GPU device client after service restoration: pid=%s comm=%s\n' "${pid}" "${comm}" >&2
      exit 50
    fi
  done
elif [[ ${fuser_release_rc} -ne 1 \
        || -s "${RUN_DIR}/fuser-release.stdout" \
        || -s "${RUN_DIR}/fuser-release.stderr" ]]; then
  printf 'ERROR: final fuser verification failed (rc=%s).\n' "${fuser_release_rc}" >&2
  exit 51
fi

journalctl -k --after-cursor="${POST_RESTORE_CURSOR}" --no-pager \
  > "${RUN_DIR}/kernel-delta-release.txt"
if grep -E 'NVRM:.*Xid|RmInitAdapter|AER' "${RUN_DIR}/kernel-delta-release.txt"; then
  printf 'ERROR: a new GPU/PCI kernel error appeared during service restoration.\n' >&2
  exit 52
fi

VALIDATION_PASSED=1
READY_TO_RELEASE=1

env -i PATH=/usr/bin:/bin \
  /usr/bin/python3 -I - "${RUN_DIR}/administrator-response-final.json" "${RUN_ID}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

payload = {
    "schema_version": "stage0.gpu-admin-response.v1",
    "request_id": "g0-g-admin-request-20260719",
    "change_reference": f"g0-g-path-b-{sys.argv[2]}",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "response_status": "PASS",
    "selected_path": "B",
    "administrator_execution": "interactive sudo supplied by host account owner",
    "expected_device_count": 4,
    "allowed_devices": [
        {"pci": "0000:9c:00.0", "uuid": "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267"},
        {"pci": "0000:9d:00.0", "uuid": "GPU-e78c55cd-db97-b761-f559-dc6eae3be81d"},
        {"pci": "0000:a0:00.0", "uuid": "GPU-9b2b2a3b-3547-187f-ca29-2c02624e2e4f"},
        {"pci": "0000:a4:00.0", "uuid": "GPU-5a81500d-5e9c-b0d7-5607-fdfdaab65ff4"},
    ],
    "excluded_devices": [
        {"pci": "0000:4f:00.0", "reason": "repeated RM initialization failure; quarantined"},
        {"pci": "0000:50:00.0", "reason": "uncorrectable ECC, pending row-remap, Xid 95; quarantined"},
        {"pci": "0000:53:00.0", "reason": "healthy spare excluded to enforce exact four-card scope"},
        {"pci": "0000:57:00.0", "reason": "healthy spare excluded to enforce exact four-card scope"},
    ],
    "isolation_enforcement": "root-controlled exact PCI unbind from the NVIDIA driver; excluded device nodes remain root-only",
    "invalidation_conditions": [
        "node reboot", "NVIDIA driver reload", "manual PCI rebind", "kernel change",
        "hardware or topology change", "PCI or UUID mapping change",
    ],
    "minimum_observation_minutes": 15,
    "g0_g_status": "PASS",
    "administrator_conclusion": "Approved four-GPU path B is enforced and passed identity, ECC, row-remap, process, PyTorch, journal, binding, observation-window, and post-service-restore checks.",
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n")
PY

(
  cd "${RUN_DIR}"
  find . -maxdepth 1 -type f \
    ! -name 'admin-maintenance.log' \
    ! -name 'SHA256SUMS' \
    ! -name 'SUCCESS' \
    ! -name 'FAILURE' \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum
) > "${RUN_DIR}/SHA256SUMS"

find "${RUN_DIR}" -maxdepth 1 -type f ! -name 'admin-maintenance.log' -exec chmod 0444 {} +
success_tmp="$(mktemp -p "${RUN_DIR}" .success.XXXXXXXX)"
{
  printf 'status=G0_G_PATH_B_PASS\n'
  printf 'g0_g_status=PASS\n'
  printf 'minimum_observation_minutes=15\n'
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'evidence=%s\n' "${RUN_DIR}"
} > "${success_tmp}"
chmod 0444 "${success_tmp}"
mv "${success_tmp}" "${RUN_DIR}/SUCCESS"

printf 'G0_G_PATH_B_PASS\n'
printf 'G0-G administrator isolation and observation passed.\n'
printf 'Evidence: %s\n' "${RUN_DIR}"
printf 'Stage 0 path B maintenance completed: %s\n' "$(date --iso-8601=seconds)"
