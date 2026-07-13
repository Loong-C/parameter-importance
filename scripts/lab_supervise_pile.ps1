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

Log "supervise $TaskName for at most $MaxHours hours"
while ((Get-Date) -lt $deadline) {
  & ssh.exe -o BatchMode=yes sophgo13 "test -f '$remote/document-00000-of-00020.bin' -a -f '$remote/document.idx'"
  if ($LASTEXITCODE -eq 0) {
    Log 'Pile shard0 and idx are complete; start final offline validation'
    & ssh.exe -o BatchMode=yes sophgo13 bash /home/sophgo13/cjl/parameter-importance/scripts/server_finalize.sh 2>&1 |
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
