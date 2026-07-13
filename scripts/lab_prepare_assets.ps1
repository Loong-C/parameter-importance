param(
  [string]$Root = 'C:\Users\cjl\parameter-importance-assets',
  [string]$Python = 'C:\Users\cjl\Apps\Python312\python.exe',
  [string]$Script = 'C:\Users\cjl\parameter-importance-assets\lab_cache_assets.py'
)
$ErrorActionPreference = 'Stop'
$log = Join-Path $Root 'prepare.log'
New-Item -ItemType Directory -Force $Root | Out-Null
function Log([string]$Message) { Add-Content -LiteralPath $log -Encoding UTF8 -Value "$(Get-Date -Format o) $Message" }

Log 'start fixed-revision model and dataset cache'
& $Python $Script
if ($LASTEXITCODE -ne 0) { throw 'asset cache failed' }
$archive = Join-Path $Root 'minimum-loop-assets.tar'
if (-not (Test-Path $archive)) { throw 'asset archive was not created' }
Log "archive ready: $((Get-Item $archive).Length) bytes"

& scp.exe -O -o BatchMode=yes $archive 'sophgo13:/home/sophgo13/cjl/storage/parameter-importance/tmp/minimum-loop-assets.tar'
if ($LASTEXITCODE -ne 0) { throw 'asset SCP failed' }
Log 'archive transferred; install on server'
& ssh.exe -o BatchMode=yes sophgo13 bash /home/sophgo13/cjl/parameter-importance/scripts/server_install_assets.sh
if ($LASTEXITCODE -ne 0) { throw 'server asset installation failed' }
Log 'model, dataset and source assets complete'
