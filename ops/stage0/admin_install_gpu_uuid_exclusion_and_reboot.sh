#!/usr/bin/env bash
set -Eeuo pipefail
umask 022
unset BASH_ENV ENV CDPATH
export PATH=/usr/sbin:/usr/bin:/sbin:/bin

readonly EXPECTED_HOST="sophgo13"
readonly EXPECTED_BOOT_ID="227d26bb-ef3c-420e-bc57-7aa186eddb87"
readonly EXPECTED_DRIVER="575.57.08"
readonly EXPECTED_BOOT_KERNEL="6.8.0-136-generic"
readonly STUCK_PID="989519"
readonly STUCK_SCRIPT="/root/stage0-gpu-admin.EDD5RzEz"
readonly PARENT_SCRIPT_SHA256="fe63376201d9ec3e8a379a724a16fc4d8531282938182d6974d2f76d6cf7af96"
readonly PARENT_RUN="/var/lib/parameter-importance/stage0/g0-g-path-b/admin-20260719T121157Z.MtJwlbzw"
readonly ADMIN_ROOT="/var/lib/parameter-importance/stage0/g0-g-uuid-exclusion"
readonly CONFIG_PATH="/etc/modprobe.d/parameter-importance-stage0-gpu-exclusion.conf"
readonly CONFIG_LINE="options nvidia NVreg_ExcludedGpus=GPU-6ff7389b-eaf8-aefd-b2c6-1611be41fa5d,GPU-dc6cfc60-41dd-7bcf-ed09-b7deb5be342c,GPU-180ff767-885a-7dc9-c8a9-921d65a01bbd,GPU-d0ce0b43-7e46-6bca-b078-5aa7043928d7"
readonly DOWNLOAD_PART="/home/sophgo13/cjl/storage/parameter-importance/datasets/pile-deduped-pythia-preshuffled/document-00009-of-00020.bin.part"
readonly DOWNLOAD_META="${DOWNLOAD_PART}.meta"
readonly DOWNLOAD_SCRIPT="/home/sophgo13/cjl/storage/parameter-importance/tmp/pile-full-download/server_xet_download.sh"

readonly -a REQUIRED_KERNELS=("6.8.0-134-generic" "6.8.0-136-generic")
readonly -a ALL_BDFS=(
  "0000:4f:00.0" "0000:50:00.0" "0000:53:00.0" "0000:57:00.0"
  "0000:9c:00.0" "0000:9d:00.0" "0000:a0:00.0" "0000:a4:00.0"
)
readonly -a ALREADY_UNBOUND_BDFS=("0000:4f:00.0" "0000:50:00.0")
readonly -a STILL_BOUND_BDFS=(
  "0000:53:00.0" "0000:57:00.0"
  "0000:9c:00.0" "0000:9d:00.0" "0000:a0:00.0" "0000:a4:00.0"
)
readonly -a MASK_UNITS=(
  "docker.service" "docker.socket" "containerd.service" "nvidia-fabricmanager.service"
)
readonly -a DISABLE_UNITS=(
  "snap.lxd.activate.service"
  "snap.lxd.daemon.unix.socket"
  "snap.lxd.user-daemon.unix.socket"
)

declare -Ar PARENT_HASHES=(
  ["pre-change-identity.txt"]="c16ea7dea0ca3876320e879cc834f6620ab080679f096b48fbe7ea373639e604"
  ["pre-change-device-nodes.txt"]="a911d82635df52dbcd28ef239b1db2b331fbd76445a9a11fd43985ec9549d905"
  ["pre-change-pci.txt"]="860a0a37ae9f9f832ade468967ce8cd02f92c42c1217bea3d9e0017fc68344e2"
)

if [[ ${EUID} -ne 0 ]]; then
  printf 'ERROR: root is required.\n' >&2
  exit 2
fi
if [[ $(hostname -s) != "${EXPECTED_HOST}" ]]; then
  printf 'ERROR: expected host %s.\n' "${EXPECTED_HOST}" >&2
  exit 3
fi
for required in flock fuser modinfo modprobe update-initramfs lsinitramfs unmkinitramfs systemd-run sha256sum; do
  command -v "${required}" >/dev/null 2>&1 || {
    printf 'ERROR: required command unavailable: %s\n' "${required}" >&2
    exit 4
  }
done

install -d -o root -g root -m 0700 /run/lock/parameter-importance
exec 9>/run/lock/parameter-importance/gpu-uuid-exclusion-reboot.lock
flock -n 9 || {
  printf 'ERROR: another UUID-exclusion recovery process is active.\n' >&2
  exit 5
}

install -d -o root -g root -m 0755 "${ADMIN_ROOT}"
RUN_DIR="$(mktemp -d -p "${ADMIN_ROOT}" "preboot-$(date -u +%Y%m%dT%H%M%SZ).XXXXXXXX")"
readonly RUN_DIR
chmod 0755 "${RUN_DIR}"
readonly LOG_PATH="${RUN_DIR}/admin-preboot.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

CONFIG_INSTALLED=0
MASKS_APPLIED=0
REBOOT_COMMITTED=0
ROLLBACK_OK=1
ROLLBACK_INITRAMFS_VERIFIED=0
CONFIG_TMP=""

restore_unit_enablement() {
  local unit enabled current prior_active
  [[ -f ${RUN_DIR}/unit-prestate.txt ]] || return 0
  while IFS='|' read -r unit enabled prior_active; do
    systemctl unmask "${unit}" >/dev/null 2>&1 || ROLLBACK_OK=0
    case "${enabled}" in
      enabled|enabled-runtime|linked|linked-runtime|alias)
        systemctl enable "${unit}" >/dev/null 2>&1 || ROLLBACK_OK=0
        ;;
      disabled)
        systemctl disable "${unit}" >/dev/null 2>&1 || ROLLBACK_OK=0
        ;;
      static|indirect|generated|transient|not-found)
        ;;
      masked|masked-runtime)
        systemctl mask "${unit}" >/dev/null 2>&1 || ROLLBACK_OK=0
        ;;
      *)
        printf 'ROLLBACK WARNING: unknown prior enablement for %s: %s\n' "${unit}" "${enabled}" >&2
        ROLLBACK_OK=0
        ;;
    esac
    if [[ ${prior_active} == active ]]; then
      systemctl start "${unit}" >/dev/null 2>&1 || ROLLBACK_OK=0
    fi
  done < "${RUN_DIR}/unit-prestate.txt"
  while IFS='|' read -r unit enabled prior_active; do
    current="$(systemctl is-enabled "${unit}" 2>/dev/null || true)"
    case "${enabled}" in
      enabled|enabled-runtime|linked|linked-runtime|alias)
        [[ ${current} == enabled || ${current} == enabled-runtime \
            || ${current} == linked || ${current} == linked-runtime \
            || ${current} == alias ]] || ROLLBACK_OK=0
        ;;
      disabled|static|indirect|generated|transient|not-found)
        [[ ${current} == "${enabled}" ]] || ROLLBACK_OK=0
        ;;
      masked|masked-runtime)
        [[ ${current} == masked || ${current} == masked-runtime ]] || ROLLBACK_OK=0
        ;;
      *)
        ROLLBACK_OK=0
        ;;
    esac
    if [[ ${prior_active} == active \
          && $(systemctl is-active "${unit}" 2>/dev/null || true) != active ]]; then
      ROLLBACK_OK=0
    fi
    printf '%s|expected=%s|observed=%s\n' "${unit}" "${enabled}" "${current}" \
      >> "${RUN_DIR}/rollback-unit-verification.txt"
  done < "${RUN_DIR}/unit-prestate.txt"
}

rollback_preboot() {
  local kernel image
  set +e
  if [[ -n ${CONFIG_TMP} ]]; then
    rm -f -- "${CONFIG_TMP}" || ROLLBACK_OK=0
  fi
  if [[ ${CONFIG_INSTALLED} -eq 1 ]]; then
    rm -f -- "${CONFIG_PATH}"
    if [[ -e ${CONFIG_PATH} || -L ${CONFIG_PATH} ]]; then
      ROLLBACK_OK=0
    elif update-initramfs -u -k all > "${RUN_DIR}/rollback-update-initramfs.log" 2>&1; then
      ROLLBACK_INITRAMFS_VERIFIED=1
      for kernel in "${REQUIRED_KERNELS[@]}"; do
        image="/boot/initrd.img-${kernel}"
        if ! lsinitramfs "${image}" \
          > "${RUN_DIR}/rollback-initrd-${kernel}-files.txt" 2>/dev/null; then
          ROLLBACK_INITRAMFS_VERIFIED=0
          ROLLBACK_OK=0
        elif grep -Fqx 'etc/modprobe.d/parameter-importance-stage0-gpu-exclusion.conf' \
          "${RUN_DIR}/rollback-initrd-${kernel}-files.txt"; then
          ROLLBACK_INITRAMFS_VERIFIED=0
          ROLLBACK_OK=0
        fi
      done
    else
      ROLLBACK_OK=0
    fi
  fi
  if [[ ${MASKS_APPLIED} -eq 1 ]]; then
    if [[ ${CONFIG_INSTALLED} -eq 0 || ${ROLLBACK_INITRAMFS_VERIFIED} -eq 1 ]]; then
      restore_unit_enablement
      systemctl daemon-reload >/dev/null 2>&1 || ROLLBACK_OK=0
    else
      printf 'ROLLBACK HOLD: allocator/Fabric Manager units remain masked because initramfs rollback was not verified.\n' >&2
      ROLLBACK_OK=0
    fi
  fi
  sync
  set -e
}

assert_no_gpu_clients() {
  local output_prefix="$1" rc
  set +e
  fuser "${GPU_NODES[@]}" \
    > "${RUN_DIR}/${output_prefix}.stdout" \
    2> "${RUN_DIR}/${output_prefix}.stderr"
  rc=$?
  set -e
  [[ ${rc} -eq 1 \
      && ! -s ${RUN_DIR}/${output_prefix}.stdout \
      && ! -s ${RUN_DIR}/${output_prefix}.stderr ]]
}

assert_package_quiescent() {
  local process_name rc unattended_pid unattended_cmd unattended_state
  local -a unattended_pids=()
  for process_name in apt apt-get dpkg update-initramfs dkms; do
    if pgrep -x "${process_name}" >/dev/null; then
      printf 'ERROR: package/initramfs process is active: %s\n' "${process_name}" >&2
      return 1
    fi
  done
  unattended_pid="$(systemctl show -p MainPID --value unattended-upgrades.service 2>/dev/null || true)"
  [[ ${unattended_pid} =~ ^[0-9]+$ && ${unattended_pid} -gt 1 \
      && -d /proc/${unattended_pid} ]] || {
    printf 'ERROR: unattended-upgrades shutdown helper identity is unavailable.\n' >&2
    return 1
  }
  mapfile -t unattended_pids < <(pgrep -x unattended-upgr || true)
  [[ ${#unattended_pids[@]} -eq 1 && ${unattended_pids[0]} == "${unattended_pid}" ]] || {
    printf 'ERROR: an additional unattended-upgrades worker is present.\n' >&2
    return 1
  }
  unattended_cmd="$(tr '\0' ' ' < "/proc/${unattended_pid}/cmdline")"
  unattended_state="$(awk '/^State:/ {print $2}' "/proc/${unattended_pid}/status")"
  [[ ${unattended_cmd} \
      == "/usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal " \
      && ${unattended_state} == S \
      && $(awk '/^PPid:/ {print $2}' "/proc/${unattended_pid}/status") == 1 ]] || {
    printf 'ERROR: unattended-upgrades is not the approved idle shutdown helper.\n' >&2
    return 1
  }
  printf 'pid=%s|state=%s|cmdline_sha256=%s\n' \
    "${unattended_pid}" "${unattended_state}" \
    "$(printf '%s' "${unattended_cmd}" | sha256sum | awk '{print $1}')" \
    > "${RUN_DIR}/unattended-upgrades-shutdown-helper.txt"
  for unit in apt-daily.service apt-daily-upgrade.service; do
    if [[ $(systemctl is-active "${unit}" 2>/dev/null || true) != inactive ]]; then
      printf 'ERROR: package maintenance unit is not inactive: %s\n' "${unit}" >&2
      return 1
    fi
  done
  if systemctl list-jobs --no-legend 2>/dev/null | grep -Ei 'apt|dpkg|initramfs|dkms'; then
    printf 'ERROR: a package/initramfs systemd job is queued.\n' >&2
    return 1
  fi
  set +e
  fuser /var/lib/dpkg/lock /var/lib/dpkg/lock-frontend /var/cache/apt/archives/lock \
    > "${RUN_DIR}/package-lock-fuser.stdout" \
    2> "${RUN_DIR}/package-lock-fuser.stderr"
  rc=$?
  set -e
  if [[ ${rc} -ne 1 || -s ${RUN_DIR}/package-lock-fuser.stdout \
        || -s ${RUN_DIR}/package-lock-fuser.stderr ]]; then
    printf 'ERROR: package-manager lock is held or fuser failed (rc=%s).\n' "${rc}" >&2
    return 1
  fi
  set +e
  dpkg --audit > "${RUN_DIR}/dpkg-audit.txt" 2>&1
  rc=$?
  set -e
  if [[ ${rc} -ne 0 || -s ${RUN_DIR}/dpkg-audit.txt ]]; then
    printf 'ERROR: dpkg audit is not clean.\n' >&2
    return 1
  fi
}

on_exit() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [[ ${rc} -ne 0 && ${REBOOT_COMMITTED} -eq 0 ]]; then
    rm -f -- "${RUN_DIR}/READY_FOR_REBOOT" "${RUN_DIR}/REBOOT_REQUESTED"
    rollback_preboot
    failure_tmp="$(mktemp -p "${RUN_DIR}" .failure.XXXXXXXX)"
    {
      printf 'status=FAILED\n'
      printf 'exit_code=%s\n' "${rc}"
      printf 'rollback_ok=%s\n' "${ROLLBACK_OK}"
      printf 'rollback_initramfs_verified=%s\n' "${ROLLBACK_INITRAMFS_VERIFIED}"
      printf 'reboot_scheduled=0\n'
    } > "${failure_tmp}"
    chmod 0444 "${failure_tmp}"
    mv -f "${failure_tmp}" "${RUN_DIR}/FAILURE"
  fi
  exit "${rc}"
}
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
trap on_exit EXIT

printf 'GPU UUID exclusion preboot start: %s\n' "$(date --iso-8601=seconds)"
printf 'Run directory: %s\n' "${RUN_DIR}"
printf 'Script SHA-256: '
sha256sum -- "$0"

[[ $(cat /proc/sys/kernel/random/boot_id) == "${EXPECTED_BOOT_ID}" ]] || {
  printf 'ERROR: boot ID changed; the hung-run recovery contract is stale.\n' >&2
  exit 10
}
[[ $(uname -r) == "6.8.0-134-generic" ]] || {
  printf 'ERROR: running kernel changed from the approved preboot state.\n' >&2
  exit 10
}
grep -Eq '^GRUB_DEFAULT=0([[:space:]]*(#.*)?)?$' /etc/default/grub || {
  printf 'ERROR: GRUB_DEFAULT is not the approved first-entry policy.\n' >&2
  exit 10
}
first_grub_linux="$(grep -m1 -E '^[[:space:]]+linux[[:space:]]+' /boot/grub/grub.cfg || true)"
[[ ${first_grub_linux} == *"vmlinuz-${EXPECTED_BOOT_KERNEL}"* ]] || {
  printf 'ERROR: GRUB first Linux entry is not the expected %s.\n' "${EXPECTED_BOOT_KERNEL}" >&2
  exit 10
}
printf '%s\n' "${EXPECTED_BOOT_ID}" > "${RUN_DIR}/preboot-boot-id.txt"
uname -a > "${RUN_DIR}/preboot-uname.txt"
printf '%s\n' "${first_grub_linux}" > "${RUN_DIR}/grub-first-linux-entry.txt"

for source_file in "${!PARENT_HASHES[@]}"; do
  source_path="${PARENT_RUN}/${source_file}"
  [[ -f ${source_path} && ! -L ${source_path} \
      && $(stat -c '%u:%g' -- "${source_path}") == "0:0" ]] || {
    printf 'ERROR: parent evidence is missing or unsafe: %s\n' "${source_path}" >&2
    exit 11
  }
  actual_hash="$(sha256sum -- "${source_path}" | awk '{print $1}')"
  [[ ${actual_hash} == "${PARENT_HASHES[${source_file}]}" ]] || {
    printf 'ERROR: parent evidence hash mismatch: %s\n' "${source_file}" >&2
    exit 12
  }
  printf '%s  %s\n' "${actual_hash}" "${source_path}" >> "${RUN_DIR}/parent-evidence-sha256.txt"
done
chmod 0444 "${RUN_DIR}/parent-evidence-sha256.txt"

if [[ -d /proc/${STUCK_PID} ]]; then
  [[ $(stat -c '%u' "/proc/${STUCK_PID}") == 0 ]] || {
    printf 'ERROR: expected stuck PID is not root-owned.\n' >&2
    exit 13
  }
  stuck_cmd="$(tr '\0' ' ' < "/proc/${STUCK_PID}/cmdline")"
  [[ ${stuck_cmd} == *"${STUCK_SCRIPT}"* ]] || {
    printf 'ERROR: PID %s is not the approved stuck maintenance script.\n' "${STUCK_PID}" >&2
    exit 14
  }
  [[ $(sha256sum -- "${STUCK_SCRIPT}" | awk '{print $1}') == "${PARENT_SCRIPT_SHA256}" ]] || {
    printf 'ERROR: stuck administrator script hash mismatch.\n' >&2
    exit 14
  }
  stuck_state="$(awk '/^State:/ {print $2}' "/proc/${STUCK_PID}/status")"
  [[ ${stuck_state} == R || ${stuck_state} == S ]] || {
    printf 'ERROR: stuck process is in an unsupported state: %s\n' "${stuck_state}" >&2
    exit 14
  }
  unbind_fd_found=0
  for fd_path in /proc/${STUCK_PID}/fd/*; do
    if [[ $(readlink "${fd_path}" 2>/dev/null || true) == /sys/bus/pci/drivers/nvidia/unbind ]]; then
      unbind_fd_found=1
      break
    fi
  done
  [[ ${unbind_fd_found} -eq 1 ]] || {
    printf 'ERROR: stuck process no longer holds the NVIDIA unbind sysfs file.\n' >&2
    exit 14
  }
  [[ $(readlink /proc/${STUCK_PID}/fd/9 2>/dev/null || true) \
      == /run/lock/parameter-importance/gpu-path-b.lock ]] || {
    printf 'ERROR: stuck process no longer owns the approved maintenance lock descriptor.\n' >&2
    exit 14
  }
  exec 8>/run/lock/parameter-importance/gpu-path-b.lock
  if flock -n 8; then
    printf 'ERROR: the original maintenance lock is unexpectedly available.\n' >&2
    exit 14
  fi
  exec 8>&-
  pending_hex="$(awk '/^ShdPnd:/ {print $2}' "/proc/${STUCK_PID}/status")"
  (( (16#${pending_hex} & 2) != 0 )) || {
    printf 'ERROR: the single approved SIGINT is not pending on the stuck process.\n' >&2
    exit 15
  }
  ps -p "${STUCK_PID}" -o pid,ppid,user,state,pcpu,etime,args > "${RUN_DIR}/stuck-process.txt"
  cat "/proc/${STUCK_PID}/wchan" > "${RUN_DIR}/stuck-process-wchan.txt" || true
elif [[ -f ${PARENT_RUN}/FAILURE ]]; then
  install -o root -g root -m 0444 "${PARENT_RUN}/FAILURE" "${RUN_DIR}/parent-failure.txt"
else
  printf 'ERROR: neither the approved stuck process nor a parent FAILURE marker exists.\n' >&2
  exit 16
fi

for bdf in "${ALL_BDFS[@]}"; do
  [[ -e /sys/bus/pci/devices/${bdf} \
      && $(cat "/sys/bus/pci/devices/${bdf}/vendor") == "0x10de" ]] || {
    printf 'ERROR: expected NVIDIA PCI function missing: %s\n' "${bdf}" >&2
    exit 17
  }
done
for bdf in "${ALREADY_UNBOUND_BDFS[@]}"; do
  [[ ! -L /sys/bus/pci/devices/${bdf}/driver ]] || {
    printf 'ERROR: expected partially isolated device rebound: %s\n' "${bdf}" >&2
    exit 18
  }
done
for bdf in "${STILL_BOUND_BDFS[@]}"; do
  driver="/sys/bus/pci/devices/${bdf}/driver"
  [[ -L ${driver} && $(basename "$(readlink -f "${driver}")") == nvidia ]] || {
    printf 'ERROR: expected NVIDIA binding missing: %s\n' "${bdf}" >&2
    exit 19
  }
done

shopt -s nullglob
GPU_NODES=(/dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset /dev/nvidia-nvswitchctl /dev/nvidia-caps/*)
shopt -u nullglob
(( ${#GPU_NODES[@]} > 0 )) || {
  printf 'ERROR: no NVIDIA device nodes found.\n' >&2
  exit 20
}
for node in "${GPU_NODES[@]}"; do
  [[ $(stat -c '%a:%u:%g' -- "${node}") == "600:0:0" ]] || {
    printf 'ERROR: safe-hold node is not root-only: %s\n' "${node}" >&2
    exit 21
  }
  stat -c '%n|%a|%u|%g|%t:%T' -- "${node}" >> "${RUN_DIR}/safe-hold-device-nodes.txt"
done

assert_no_gpu_clients "gpu-fuser-before-service-isolation" || {
  printf 'ERROR: GPU device clients exist or fuser failed.\n' >&2
  exit 22
}

for unit in docker.service docker.socket containerd.service \
  snap.lxd.activate.service snap.lxd.daemon.unix.socket \
  snap.lxd.daemon.service snap.lxd.user-daemon.service nvidia-persistenced.service; do
  unit_state="$(systemctl is-active "${unit}" 2>/dev/null || true)"
  [[ ${unit_state} == inactive ]] || {
    printf 'ERROR: service must be exactly inactive before recovery reboot: %s (%s)\n' \
      "${unit}" "${unit_state}" >&2
    exit 23
  }
done
user_lxd_socket_state="$(systemctl is-active snap.lxd.user-daemon.unix.socket 2>/dev/null || true)"
[[ ${user_lxd_socket_state} == active ]] || {
  printf 'ERROR: LXD user-daemon socket prestate changed from active: %s\n' \
    "${user_lxd_socket_state}" >&2
  exit 23
}
for legacy_unit in lxd.service lxd.socket; do
  legacy_active="$(systemctl is-active "${legacy_unit}" 2>/dev/null || true)"
  legacy_enabled="$(systemctl is-enabled "${legacy_unit}" 2>/dev/null || true)"
  printf '%s|%s|%s\n' "${legacy_unit}" "${legacy_enabled:-not-found}" \
    "${legacy_active:-inactive}" >> "${RUN_DIR}/legacy-lxd-prestate.txt"
  [[ ${legacy_active} == inactive && ( -z ${legacy_enabled} || ${legacy_enabled} == not-found ) ]] || {
    printf 'ERROR: an unexpected legacy LXD unit is installed or active: %s\n' "${legacy_unit}" >&2
    exit 23
  }
done
fm_state="$(systemctl is-active nvidia-fabricmanager.service 2>/dev/null || true)"
[[ ${fm_state} == failed ]] || {
  printf 'ERROR: Fabric Manager prestate changed from the approved failed state: %s\n' "${fm_state}" >&2
  exit 23
}

assert_package_quiescent || exit 24

available_boot_bytes="$(df --output=avail -B1 /boot | tail -n 1 | tr -d ' ')"
(( available_boot_bytes >= 536870912 )) || {
  printf 'ERROR: /boot has less than 512 MiB free.\n' >&2
  exit 26
}
df -h / /boot /var > "${RUN_DIR}/filesystem-space.txt"

for kernel in "${REQUIRED_KERNELS[@]}"; do
  [[ -f /boot/initrd.img-${kernel} ]] || {
    printf 'ERROR: required initrd missing: %s\n' "${kernel}" >&2
    exit 27
  }
  [[ $(modinfo -k "${kernel}" -F version nvidia) == "${EXPECTED_DRIVER}" ]] || {
    printf 'ERROR: NVIDIA driver version mismatch for %s.\n' "${kernel}" >&2
    exit 28
  }
  modinfo -k "${kernel}" -p nvidia | grep -Fqx 'NVreg_ExcludedGpus: (charp)' || {
    printf 'ERROR: NVreg_ExcludedGpus unavailable for %s.\n' "${kernel}" >&2
    exit 29
  }
  sha256sum "/boot/initrd.img-${kernel}" >> "${RUN_DIR}/initrd-before-sha256.txt"
done

[[ ! -e ${CONFIG_PATH} && ! -L ${CONFIG_PATH} ]] || {
  printf 'ERROR: managed modprobe configuration already exists.\n' >&2
  exit 30
}
if modprobe --showconfig | grep -Eq '^options[[:space:]]+nvidia[[:space:]].*NVreg_(ExcludedGpus|GpuBlacklist)='; then
  printf 'ERROR: an existing NVIDIA exclusion option would conflict.\n' >&2
  exit 31
fi

for unit in "${MASK_UNITS[@]}" "${DISABLE_UNITS[@]}"; do
  enabled_state="$(systemctl is-enabled "${unit}" 2>/dev/null || true)"
  active_state="$(systemctl is-active "${unit}" 2>/dev/null || true)"
  printf '%s|%s|%s\n' "${unit}" "${enabled_state:-not-found}" "${active_state:-unknown}" \
    >> "${RUN_DIR}/unit-prestate.txt"
  [[ ${enabled_state} != masked && ${enabled_state} != masked-runtime ]] || {
    printf 'ERROR: unit was already masked before this run: %s\n' "${unit}" >&2
    exit 32
  }
done
printf 'nvidia-persistenced.service|%s|%s\n' \
  "$(systemctl is-enabled nvidia-persistenced.service 2>/dev/null || true)" \
  "$(systemctl is-active nvidia-persistenced.service 2>/dev/null || true)" \
  > "${RUN_DIR}/gpu-service-prestate.txt"
MASKS_APPLIED=1
for unit in "${MASK_UNITS[@]}"; do
  systemctl mask --now "${unit}"
  [[ $(systemctl is-enabled "${unit}" 2>/dev/null || true) == masked ]] || {
    printf 'ERROR: failed to persistently mask unit: %s\n' "${unit}" >&2
    exit 33
  }
done
for unit in "${DISABLE_UNITS[@]}"; do
  systemctl disable --now "${unit}"
  [[ $(systemctl is-enabled "${unit}" 2>/dev/null || true) == disabled ]] || {
    printf 'ERROR: failed to persistently disable unit: %s\n' "${unit}" >&2
    exit 33
  }
done
systemctl daemon-reload
for unit in "${MASK_UNITS[@]}" "${DISABLE_UNITS[@]}"; do
  unit_state="$(systemctl is-active "${unit}" 2>/dev/null || true)"
  if [[ ${unit} == nvidia-fabricmanager.service ]]; then
    expected_isolated_state=failed
  else
    expected_isolated_state=inactive
  fi
  [[ ${unit_state} == "${expected_isolated_state}" ]] || {
    printf 'ERROR: isolated service is not inactive after mask/disable: %s (%s)\n' \
      "${unit}" "${unit_state}" >&2
    exit 33
  }
done
assert_no_gpu_clients "gpu-fuser-after-service-isolation" || {
  printf 'ERROR: GPU client appeared during service isolation.\n' >&2
  exit 33
}

CONFIG_TMP="$(mktemp /etc/modprobe.d/.parameter-importance-gpu-exclusion.XXXXXXXX)"
printf '%s\n' "${CONFIG_LINE}" > "${CONFIG_TMP}"
chown root:root "${CONFIG_TMP}"
chmod 0644 "${CONFIG_TMP}"
CONFIG_INSTALLED=1
mv -f "${CONFIG_TMP}" "${CONFIG_PATH}"
CONFIG_TMP=""
[[ $(stat -c '%u:%g:%a' -- "${CONFIG_PATH}") == "0:0:644" ]] || exit 34
[[ $(wc -l < "${CONFIG_PATH}") -eq 1 ]] || exit 34
grep -Fqx -- "${CONFIG_LINE}" "${CONFIG_PATH}" || exit 34

mapfile -t configured_exclusions < <(
  modprobe --showconfig \
    | grep -oE 'NVreg_ExcludedGpus=[^[:space:]]+' \
    || true
)
[[ ${#configured_exclusions[@]} -eq 1 \
    && ${configured_exclusions[0]} == "NVreg_ExcludedGpus=${CONFIG_LINE#*NVreg_ExcludedGpus=}" ]] || {
  printf 'ERROR: modprobe did not resolve exactly the approved exclusion value.\n' >&2
  exit 35
}
modprobe --showconfig > "${RUN_DIR}/modprobe-showconfig.txt"

assert_package_quiescent || exit 35
update-initramfs -u -k all > "${RUN_DIR}/update-initramfs.log" 2>&1
cat "${RUN_DIR}/update-initramfs.log"

for kernel in "${REQUIRED_KERNELS[@]}"; do
  image="/boot/initrd.img-${kernel}"
  sha256sum "${image}" >> "${RUN_DIR}/initrd-after-sha256.txt"
  lsinitramfs "${image}" > "${RUN_DIR}/initrd-${kernel}-files.txt"
  grep -Fqx 'etc/modprobe.d/parameter-importance-stage0-gpu-exclusion.conf' \
    "${RUN_DIR}/initrd-${kernel}-files.txt" || {
    printf 'ERROR: exclusion config missing from initrd %s.\n' "${kernel}" >&2
    exit 36
  }
  extract_dir="$(mktemp -d)"
  unmkinitramfs "${image}" "${extract_dir}"
  mapfile -t embedded_configs < <(
    find "${extract_dir}" -path '*/etc/modprobe.d/parameter-importance-stage0-gpu-exclusion.conf' -type f -print
  )
  if [[ ${#embedded_configs[@]} -ne 1 ]] || ! cmp -s "${CONFIG_PATH}" "${embedded_configs[0]}"; then
    rm -rf -- "${extract_dir}"
    printf 'ERROR: embedded initramfs config mismatch for %s.\n' "${kernel}" >&2
    exit 37
  fi
  rm -rf -- "${extract_dir}"
done

install -o root -g root -m 0444 "${CONFIG_PATH}" "${RUN_DIR}/installed-modprobe-config.txt"
sha256sum "${CONFIG_PATH}" > "${RUN_DIR}/installed-modprobe-config.sha256"
journalctl -k --since '2026-07-19 12:11:45 UTC' --no-pager > "${RUN_DIR}/kernel-hung-unbind.txt"
systemctl status "${MASK_UNITS[@]}" "${DISABLE_UNITS[@]}" \
  snap.lxd.daemon.service nvidia-persistenced.service --no-pager \
  > "${RUN_DIR}/service-status-preboot.txt" 2>&1 || true
if [[ -f ${DOWNLOAD_PART} ]]; then
  stat -c 'before|%n|%s|%y' "${DOWNLOAD_PART}" > "${RUN_DIR}/download-part-state.txt"
  [[ -f ${DOWNLOAD_META} ]] && sha256sum "${DOWNLOAD_META}" >> "${RUN_DIR}/download-part-state.txt"
  [[ -f ${DOWNLOAD_SCRIPT} ]] && sha256sum "${DOWNLOAD_SCRIPT}" >> "${RUN_DIR}/download-part-state.txt"
fi

# Stop only the uniquely identified resumable transfer after every boot-image
# verification has passed. The partial and metadata files are retained.
if [[ -d /proc/971897 ]]; then
  download_cmd="$(tr '\0' ' ' < /proc/971897/cmdline)"
  [[ $(readlink -f /proc/971897/exe) == /usr/bin/curl \
      && ${download_cmd} == *"--output ${DOWNLOAD_PART}"* ]] || {
    printf 'ERROR: PID 971897 no longer matches the approved resumable curl transfer.\n' >&2
    exit 38
  }
  {
    printf 'pid=971897\n'
    printf 'exe=/usr/bin/curl\n'
    printf 'output=%s\n' "${DOWNLOAD_PART}"
    printf 'cmdline_sha256=%s\n' "$(printf '%s' "${download_cmd}" | sha256sum | awk '{print $1}')"
    printf 'signed_url_recorded=0\n'
  } > "${RUN_DIR}/download-command-sanitized.txt"
  kill -TERM 971897
fi
for _attempt in $(seq 1 30); do
  if ! pgrep -f -- "${DOWNLOAD_PART}" >/dev/null; then
    break
  fi
  sleep 1
done
if pgrep -f -- "${DOWNLOAD_PART}" > "${RUN_DIR}/download-pids-after-stop.txt"; then
  printf 'ERROR: the resumable transfer did not stop cleanly.\n' >&2
  exit 39
fi
set +e
fuser "${DOWNLOAD_PART}" "${DOWNLOAD_PART}.lock" \
  > "${RUN_DIR}/download-fuser-after-stop.stdout" \
  2> "${RUN_DIR}/download-fuser-after-stop.stderr"
download_fuser_rc=$?
set -e
[[ ${download_fuser_rc} -eq 1 \
    && ! -s ${RUN_DIR}/download-fuser-after-stop.stdout \
    && ! -s ${RUN_DIR}/download-fuser-after-stop.stderr ]] || {
  printf 'ERROR: the resumable transfer files remain open or fuser failed (rc=%s).\n' "${download_fuser_rc}" >&2
  exit 40
}
if [[ -f ${DOWNLOAD_PART} ]]; then
  stat -c 'after|%n|%s|%y' "${DOWNLOAD_PART}" >> "${RUN_DIR}/download-part-state.txt"
fi

{
  printf 'status=READY_FOR_CONTROLLED_REBOOT\n'
  printf 'old_boot_id=%s\n' "${EXPECTED_BOOT_ID}"
  printf 'expected_boot_kernel=%s\n' "${EXPECTED_BOOT_KERNEL}"
  printf 'driver=%s\n' "${EXPECTED_DRIVER}"
  printf 'config=%s\n' "${CONFIG_PATH}"
  printf 'excluded_uuid_count=4\n'
  printf 'allocator_and_fabric_units_masked=1\n'
  printf 'lxd_activation_socket_disabled=1\n'
  printf 'resumable_transfer_stopped=1\n'
  printf 'reboot_submission=direct_systemd_request\n'
} > "${RUN_DIR}/READY_FOR_REBOOT"
chmod 0444 "${RUN_DIR}/READY_FOR_REBOOT"
{
  printf 'status=REBOOT_REQUEST_READY_TO_SUBMIT\n'
  printf 'requested_at=%s\n' "$(date --iso-8601=seconds)"
} > "${RUN_DIR}/REBOOT_REQUESTED"
chmod 0444 "${RUN_DIR}/REBOOT_REQUESTED"
sync

# No fallible evidence I/O occurs after entering this critical section. Signals
# are ignored only across the immediate systemd request so cleanup cannot race
# an already accepted reboot.
trap '' INT TERM HUP
REBOOT_COMMITTED=1
set +e
systemctl reboot
reboot_rc=$?
set -e
if [[ ${reboot_rc} -ne 0 ]]; then
  REBOOT_COMMITTED=0
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  printf 'ERROR: systemd rejected the controlled reboot request (rc=%s).\n' "${reboot_rc}" >&2
  exit 38
fi

trap - EXIT
printf 'Controlled reboot request accepted by systemd.\n'
exit 0
