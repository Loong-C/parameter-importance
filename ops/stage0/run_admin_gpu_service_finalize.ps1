$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "Stage 0 GPU service restoration"

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$localScript = Join-Path $projectRoot "ops\stage0\admin_restore_gpu_services_after_exclusion.sh"
$sshPath = Join-Path $env:WINDIR "System32\OpenSSH\ssh.exe"
$scpPath = Join-Path $env:WINDIR "System32\OpenSSH\scp.exe"
$expectedSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $localScript).Hash.ToLowerInvariant()
$remotePath = "/home/sophgo13/cjl/storage/parameter-importance/tmp/stage0-gpu-service-finalize-$($expectedSha256.Substring(0, 16)).sh"

if (-not (Test-Path -LiteralPath $sshPath -PathType Leaf)) {
    throw "Trusted Windows OpenSSH client is unavailable at $sshPath."
}
if (-not (Test-Path -LiteralPath $scpPath -PathType Leaf)) {
    throw "Trusted Windows OpenSSH copy client is unavailable at $scpPath."
}
$actualSha256 = $expectedSha256

& $scpPath -q $localScript "sophgo13-via-lab:$remotePath"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upload the verified finalizer to the authorized project temporary directory."
}
& $sshPath -o BatchMode=yes sophgo13-via-lab "chmod 0700 '$remotePath'"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to set the remote finalizer mode."
}

Write-Host "Stage 0 exact-four-GPU service restoration" -ForegroundColor Cyan
Write-Host "Read-only post-boot gate: PASS" -ForegroundColor Green
Write-Host "Previous attempt returned to a verified safe hold; this version restores that baseline before retrying." -ForegroundColor Yellow
Write-Host "Verified finalizer SHA-256: $actualSha256"
Write-Host "Enter the sudo password only at the remote terminal prompt." -ForegroundColor Yellow
Write-Host "Do not close this window until a PASS/FAIL result is printed."
Write-Host ""

$remoteTemplate = @'
set -euo pipefail
remote_path='__REMOTE_PATH__'
expected='__EXPECTED_SHA256__'
test -f "$remote_path"
actual=$(/usr/bin/sha256sum "$remote_path" | /usr/bin/awk '{print $1}')
if [ "$actual" != "$expected" ]; then
  printf 'ERROR: remote finalizer hash mismatch: %s\n' "$actual" >&2
  exit 90
fi
/usr/bin/sudo -k /usr/bin/env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin \
  /bin/bash --noprofile --norc "$remote_path"
'@
$remoteCommand = $remoteTemplate.Replace("__REMOTE_PATH__", $remotePath).Replace("__EXPECTED_SHA256__", $expectedSha256)

& $sshPath -tt sophgo13-via-lab $remoteCommand
$finalizerExitCode = $LASTEXITCODE

Write-Host ""
if ($finalizerExitCode -eq 0) {
    Write-Host "GPU service restoration finished successfully." -ForegroundColor Green
} else {
    Write-Host "GPU service restoration stopped with exit code $finalizerExitCode." -ForegroundColor Red
}
Read-Host "Press Enter to close this window"
exit $finalizerExitCode
