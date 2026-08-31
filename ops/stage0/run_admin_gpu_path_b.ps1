$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "Stage 0 GPU maintenance - password entry"

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$adminScript = Join-Path $projectRoot "ops\stage0\admin_apply_gpu_path_b.sh"
$sshPath = Join-Path $env:WINDIR "System32\OpenSSH\ssh.exe"
if (-not (Test-Path -LiteralPath $sshPath -PathType Leaf)) {
    throw "Trusted Windows OpenSSH client is unavailable at $sshPath."
}
$expectedSha256 = "792e58a00cc598cd5853f76593423099fccd61315e428cfb1f4f4fc287c05227"
$actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $adminScript).Hash.ToLowerInvariant()
if ($actualSha256 -ne $expectedSha256) {
    throw "Local administrator script hash mismatch. Expected $expectedSha256, got $actualSha256."
}

Write-Host "Stage 0 GPU path B maintenance" -ForegroundColor Cyan
Write-Host "Enter the sudo password only at the remote terminal prompt." -ForegroundColor Yellow
Write-Host "Docker, containerd, LXD allocators, and nvidia-persistenced will be stopped and restored." -ForegroundColor Yellow
Write-Host "No user workload will be terminated. No reset, reboot, driver reload, or ECC clear will occur."
Write-Host "The health observation lasts at least 15 minutes before services are released."
Write-Host "Excluded PCI devices: 4f:00.0, 50:00.0, 53:00.0, 57:00.0"
Write-Host "Approved PCI devices: 9c:00.0, 9d:00.0, a0:00.0, a4:00.0"
Write-Host ""

$remoteTemplate = @'
set -euo pipefail
unset BASH_ENV ENV CDPATH
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
source_path=/home/sophgo13/cjl/parameter-importance/ops/stage0/admin_apply_gpu_path_b.sh
root_copy=$(/usr/bin/mktemp /root/stage0-gpu-admin.XXXXXXXX)
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
    Write-Host "Administrator isolation and observation finished successfully." -ForegroundColor Green
} else {
    Write-Host "Maintenance stopped with exit code $maintenanceExitCode." -ForegroundColor Red
}
Write-Host "Leave this window open while Codex verifies the result."
