#requires -Version 5.1
<#
.SYNOPSIS
    探测服务器 sophgo13 的 GPU 空闲状态与 S0.9 解锁条件，并记录历史。
.DESCRIPTION
    - 通过 ssh sophgo13-via-lab 只读检查：GPU 计算进程数/显存占用、verl 进程、
      G0-G 管理员 finalizer SUCCESS 数量、boot_id、Xid 日志、S0.9 链日志尾。
    - 结果写入 ops/monitor/runtime/（CSV 历史 + latest.json + 状态变化/就绪标记）。
    - 当 GPU 空闲且最新一次 G0-G finalizer 为 SUCCESS（且其 boot_id 等于当前 boot）
      时写 S0_9_READY.flag 并尝试通知。
.NOTES
    依赖本机 SSH 别名 sophgo13-via-lab（见 Agent/remote_access.md）。
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$base = $PSScriptRoot
$runtime = Join-Path $base 'runtime'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null

$remote = @'
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BOOT=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)
UPTIME=$(awk '{print int($1)}' /proc/uptime 2>/dev/null)
APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
V=$(pgrep -fc "verl.trainer.main_ppo" 2>/dev/null || true); [ -n "$V" ] || V=0
FS=$(ls /var/lib/parameter-importance/stage0/g0-g-uuid-exclusion/service-finalize/finalize-*/SUCCESS 2>/dev/null | wc -l)
LATEST=$(ls -dt /var/lib/parameter-importance/stage0/g0-g-uuid-exclusion/service-finalize/finalize-*/ 2>/dev/null | head -1)
RESULT=NO_DIR
LATEST_BOOT=
if [ -n "$LATEST" ]; then
  if [ -f "$LATEST/SUCCESS" ]; then
    RESULT=SUCCESS
    LATEST_BOOT=$(grep '^boot_id=' "$LATEST/SUCCESS" 2>/dev/null | head -1 | cut -d= -f2)
  elif [ -f "$LATEST/FAILURE" ]; then
    RESULT=FAILURE
  else
    RESULT=NO_MARKER
  fi
fi
XID=$(dmesg 2>/dev/null | grep -i 'Xid' | tail -1 | cut -c1-200)
CHAIN=$(tail -1 /home/sophgo13/cjl/storage/parameter-importance/tmp/g7-recovery-chain-975f83c.log 2>/dev/null | cut -c1-200)
printf 'TS=%s\n' "$TS"
printf 'BOOT=%s\n' "$BOOT"
printf 'UPTIME_S=%s\n' "$UPTIME"
printf 'GPU_APPS=%s\n' "$APPS"
printf 'GPU_MEM_MIB=%s\n' "$MEM"
printf 'GPU_UTIL_MAX=%s\n' "$UTIL"
printf 'VERL_COUNT=%s\n' "$V"
printf 'FINALIZER_SUCCESS=%s\n' "$FS"
printf 'FINALIZER_RESULT=%s\n' "$RESULT"
printf 'FINALIZER_BOOT=%s\n' "$LATEST_BOOT"
printf 'FINALIZER_LATEST=%s\n' "$LATEST"
printf 'XID_TAIL=%s\n' "$XID"
printf 'CHAIN_TAIL=%s\n' "$CHAIN"
'@

$sshOut = & ssh -o BatchMode=yes -o ConnectTimeout=20 sophgo13-via-lab $remote 2>&1
if ($LASTEXITCODE -ne 0) {
    $err = '{0} SSH_FAILED rc={1}: {2}' -f (Get-Date -Format o), $LASTEXITCODE, ($sshOut -join ' ')
    Add-Content -Path (Join-Path $runtime 'errors.log') -Value $err
    Write-Error $err
    exit 1
}

$props = @{}
foreach ($line in $sshOut) {
    if ($line -match '^([A-Z_]+)=(.*)$') { $props[$matches[1]] = $matches[2] }
}

$obj = [PSCustomObject]@{
    checked_at_local  = (Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz')
    ts_utc            = $props['TS']
    boot_id           = $props['BOOT']
    uptime_s          = $props['UPTIME_S']
    gpu_apps          = [int]$props['GPU_APPS']
    gpu_mem_mib       = [int]$props['GPU_MEM_MIB']
    gpu_util_max      = [int]$props['GPU_UTIL_MAX']
    verl_count        = [int]$props['VERL_COUNT']
    finalizer_success = [int]$props['FINALIZER_SUCCESS']
    finalizer_result  = $props['FINALIZER_RESULT']
    finalizer_boot    = $props['FINALIZER_BOOT']
    finalizer_latest  = $props['FINALIZER_LATEST']
    xid_tail          = $props['XID_TAIL']
    chain_tail        = $props['CHAIN_TAIL']
}

$csv = Join-Path $runtime 'gpu_idle_history.csv'
if (-not (Test-Path $csv)) {
    $obj | Export-Csv -Path $csv -NoTypeInformation -Encoding UTF8
} else {
    $obj | Export-Csv -Path $csv -Append -NoTypeInformation -Encoding UTF8
}

$latest = Join-Path $runtime 'latest.json'
$prev = $null
if (Test-Path $latest) {
    try { $prev = Get-Content -Raw $latest | ConvertFrom-Json } catch { $prev = $null }
}
$obj | ConvertTo-Json | Set-Content -Path $latest -Encoding UTF8

$busyNow = $obj.gpu_apps -gt 0
$busyPrev = $busyNow
$fsPrev = $obj.finalizer_success
if ($null -ne $prev) {
    $busyPrev = ([int]$prev.gpu_apps) -gt 0
    $fsPrev = [int]$prev.finalizer_success
}

$changes = Join-Path $runtime 'state_changes.txt'
if ($busyPrev -ne $busyNow) {
    $state = if ($busyNow) { 'GPU_BUSY' } else { 'GPU_IDLE' }
    Add-Content -Path $changes -Value ('{0} {1} (apps={2}, mem={3} MiB)' -f (Get-Date -Format o), $state, $obj.gpu_apps, $obj.gpu_mem_mib)
}
if ($obj.finalizer_success -gt $fsPrev) {
    Add-Content -Path $changes -Value ('{0} FINALIZER_SUCCESS_COUNT {1} -> {2}' -f (Get-Date -Format o), $fsPrev, $obj.finalizer_success)
}

$readyNow = (-not $busyNow) -and ($obj.finalizer_result -eq 'SUCCESS') -and ($obj.finalizer_boot -eq $obj.boot_id)
$flag = Join-Path $runtime 'S0_9_READY.flag'
if ($readyNow -and -not (Test-Path $flag)) {
    ('READY at {0} | boot={1} | finalizer_result={2} | finalizer_boot={3}' -f (Get-Date -Format o), $obj.boot_id, $obj.finalizer_result, $obj.finalizer_boot) | Set-Content -Path $flag -Encoding UTF8
    if (Get-Command New-BurntToastNotification -ErrorAction SilentlyContinue) {
        try { New-BurntToastNotification -Text 'S0.9 解锁：GPU 空闲且 G0-G finalizer 通过' } catch { }
    } elseif (Get-Command msg.exe -ErrorAction SilentlyContinue) {
        try { & msg.exe $env:USERNAME 'S0.9 解锁：GPU 空闲且 G0-G finalizer 通过' 2>$null } catch { }
    }
} elseif (-not $readyNow -and (Test-Path $flag)) {
    Remove-Item -LiteralPath $flag -Force
}

$state = if ($busyNow) { 'BUSY' } elseif ($readyNow) { 'READY' } else { 'IDLE_WAIT_FINALIZER' }
Write-Output ('STATE={0} | gpu_apps={1} | gpu_mem_mib={2} | finalizer_result={3} | finalizer_boot={4} | boot={5}' -f $state, $obj.gpu_apps, $obj.gpu_mem_mib, $obj.finalizer_result, $obj.finalizer_boot, $obj.boot_id)
