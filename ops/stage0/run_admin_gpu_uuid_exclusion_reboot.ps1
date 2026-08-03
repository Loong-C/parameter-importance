$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "Stage 0 GPU UUID exclusion - controlled reboot"

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$adminScript = Join-Path $projectRoot "ops\stage0\admin_install_gpu_uuid_exclusion_and_reboot.sh"
$sshPath = Join-Path $env:WINDIR "System32\OpenSSH\ssh.exe"
if (-not (Test-Path -LiteralPath $sshPath -PathType Leaf)) {
    throw "Trusted Windows OpenSSH client is unavailable at $sshPath."
}

$expectedSha256 = "9233bde65b31714b189d6d89dd6f2da48f93ae87d9c8a01e94dacc235edcf1ec"
$actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $adminScript).Hash.ToLowerInvariant()
if ($actualSha256 -ne $expectedSha256) {
    throw "Local administrator script hash mismatch. Expected $expectedSha256, got $actualSha256."
}

Write-Host "Stage 0 persistent GPU UUID exclusion" -ForegroundColor Cyan
Write-Host "Enter the sudo password only at the remote terminal prompt." -ForegroundColor Yellow
Write-Host "This installs an exact four-UUID NVIDIA exclusion and rebuilds both initramfs images."
Write-Host "Docker/containerd/LXD/Fabric Manager remain masked for post-boot validation."
Write-Host "A controlled reboot is scheduled only after every configuration check passes." -ForegroundColor Yellow
Write-Host "The resumable 30 GB dataset transfer will stop; its .part file is preserved."
Write-Host "Do not close this window until the script reports that the reboot is scheduled."
Write-Host ""

$remoteTemplate = @'
set -euo pipefail
unset BASH_ENV ENV CDPATH
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
source_path=/home/sophgo13/cjl/parameter-importance/ops/stage0/admin_install_gpu_uuid_exclusion_and_reboot.sh
root_copy=$(/usr/bin/mktemp /root/stage0-gpu-uuid-reboot.XXXXXXXX)
cleanup() { /usr/bin/rm -f "$root_copy"; }
trap cleanup EXIT
/usr/bin/install -o root -g root -m 0700 "$source_path" "$root_copy"
printf '%s  %s\n' '__EXPECTED_SHA256__' "$root_copy" | /usr/bin/sha256sum -c -
/usr/bin/env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin /bin/bash --noprofile --norc "$root_copy"
'@
$remoteRootScript = $remoteTemplate.Replace("__EXPECTED_SHA256__", $expectedSha256)
$encodedRemoteRootScript = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remoteRootScript))
$remoteCommand = "set -o pipefail; printf '%s' '$encodedRemoteRootScript' | /usr/bin/base64 -d | /usr/bin/sudo -k /usr/bin/env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin /bin/bash --noprofile --norc"

& $sshPath -tt sophgo13-via-lab $remoteCommand
$maintenanceExitCode = $LASTEXITCODE

Write-Host ""
if ($maintenanceExitCode -eq 0) {
    Write-Host "Configuration verified; controlled reboot was scheduled." -ForegroundColor Green
} else {
    Write-Host "Preboot recovery stopped with exit code $maintenanceExitCode; reboot was not authorized by the script." -ForegroundColor Red
}
Write-Host "Leave this window open while Codex observes the reboot."
