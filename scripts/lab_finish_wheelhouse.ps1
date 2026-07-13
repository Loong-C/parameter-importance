param([string]$Root = 'C:\Users\cjl\parameter-importance-wheelhouse')
$ErrorActionPreference = 'Stop'
$wheelhouse = Join-Path $Root 'wheelhouse'
$lock = Join-Path $Root 'requirements.lock'
$report = Join-Path $Root 'resolution-report.json'
$manifest = Join-Path $Root 'wheelhouse-sha256.tsv'
$archive = Join-Path $Root 'wheelhouse.tar'
$log = Join-Path $Root 'finish.log'
function Log([string]$Message) { Add-Content -LiteralPath $log -Encoding UTF8 -Value "$(Get-Date -Format o) $Message" }

$wheels = Get-ChildItem -File $wheelhouse
if ($wheels.Count -ne 88) { throw "expected 88 wheels, found $($wheels.Count)" }
Log 'hash wheelhouse'
$wheels | Sort-Object Name | ForEach-Object {
  $h = Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName
  "$($h.Hash.ToLowerInvariant())`t$($_.Length)`t$($_.Name)"
} | Set-Content -LiteralPath $manifest -Encoding Ascii

if (Test-Path $archive) { Remove-Item -LiteralPath $archive -Force }
Log 'create uncompressed tar archive'
& tar.exe -cf $archive -C $Root wheelhouse requirements.lock resolution-report.json wheelhouse-sha256.tsv
if ($LASTEXITCODE -ne 0) { throw 'tar creation failed' }
Log "archive size $((Get-Item $archive).Length)"

Log 'transfer archive to server'
& scp.exe -O -o BatchMode=yes $archive 'sophgo13:/home/sophgo13/cjl/storage/parameter-importance/tmp/wheelhouse.tar'
if ($LASTEXITCODE -ne 0) { throw 'wheelhouse SCP failed' }
& scp.exe -O -o BatchMode=yes $lock 'sophgo13:/home/sophgo13/cjl/parameter-importance/environment/requirements.lock'
if ($LASTEXITCODE -ne 0) { throw 'lock SCP failed' }

Log 'verify archive and install with --no-index on server'
& ssh.exe -o BatchMode=yes sophgo13 bash /home/sophgo13/cjl/parameter-importance/scripts/server_install_wheelhouse.sh
if ($LASTEXITCODE -ne 0) { throw 'server offline install failed' }
Log 'wheelhouse and server venv complete'
