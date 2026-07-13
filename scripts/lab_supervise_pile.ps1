param(
  [string]$TaskName = 'CjlPilePrefix',
  [string]$LogFile = 'C:\Users\cjl\parameter-importance-tools\pile-supervisor.log',
  [int]$MaxHours = 18
)
$ErrorActionPreference = 'Continue'
$deadline = (Get-Date).AddHours($MaxHours)
$remote = '/home/sophgo13/cjl/storage/parameter-importance/datasets/pile-deduped-pythia-preshuffled'
New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null
function Log([string]$Message) { Add-Content -LiteralPath $LogFile -Encoding UTF8 -Value "$(Get-Date -Format o) $Message" }

function Test-RemoteState([string]$RemoteCommand, [int]$TimeoutSeconds = 45) {
  $arguments = @(
    '-o', 'BatchMode=yes',
    '-o', 'ConnectTimeout=20',
    '-o', 'ServerAliveInterval=15',
    '-o', 'ServerAliveCountMax=2',
    'sophgo13', "`"$RemoteCommand`""
  )
  $start = New-Object System.Diagnostics.ProcessStartInfo
  $start.FileName = 'ssh.exe'
  $start.Arguments = ($arguments -join ' ')
  $start.UseShellExecute = $false
  $start.CreateNoWindow = $true
  $process = New-Object System.Diagnostics.Process
  $process.StartInfo = $start
  [void]$process.Start()
  if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
    try { $process.Kill() } catch {}
    $process.Dispose()
    return $false
  }
  $success = $process.ExitCode -eq 0
  $process.Dispose()
  return $success
}

Log "supervise $TaskName for at most $MaxHours hours"
while ((Get-Date) -lt $deadline) {
  if (Test-RemoteState "test -f '$remote/document-00000-of-00020.bin' -a -f '$remote/document.idx'") {
    Log 'Pile shard0 and idx are complete; start final offline validation'
    & ssh.exe -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 `
      sophgo13 bash /home/sophgo13/cjl/parameter-importance/scripts/server_finalize.sh 2>&1 |
      ForEach-Object { Add-Content -LiteralPath $LogFile -Encoding UTF8 -Value $_ }
    if ($LASTEXITCODE -eq 0) {
      Log 'final offline validation complete'
      exit 0
    }
    Log "final offline validation failed with exit code $LASTEXITCODE"
    exit 3
  }

  try {
    $state = (Get-ScheduledTask -TaskName $TaskName).State
    if ($state -ne 'Running') {
      Log "restart $TaskName from state $state"
      Start-ScheduledTask -TaskName $TaskName
    }
  } catch {
    Log "task inspection/restart failed: $($_.Exception.Message)"
  }
  Start-Sleep -Seconds 60
}
Log 'supervisor deadline reached before both Pile objects completed'
exit 2
