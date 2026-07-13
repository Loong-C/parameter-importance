param([string]$Root = 'C:\Users\cjl\parameter-importance-wheelhouse')
$ErrorActionPreference = 'Stop'
$wheelhouse = Join-Path $Root 'wheelhouse'
$lock = Join-Path $Root 'requirements.lock'
$manifest = Join-Path $Root 'wheelhouse-sha256.tsv'
$archive = Join-Path $Root 'linux-runtime-wheels.tar'
$log = Join-Path $Root 'linux-runtime-finish.log'
function Log([string]$Message) { Add-Content -LiteralPath $log -Encoding UTF8 -Value "$(Get-Date -Format o) $Message" }

$wheels = @(Get-ChildItem -File -LiteralPath $wheelhouse -Filter '*.whl' | Sort-Object Name)
if ($wheels.Count -ne 88) { throw "expected 88 wheels, found $($wheels.Count)" }
$runtime = @($wheels | Where-Object { $_.Name -match '^(cuda_|nvidia_|triton-)' })
if ($runtime.Count -ne 18) { throw "expected 18 Linux runtime wheels, found $($runtime.Count)" }

Log "hash all $($wheels.Count) wheels"
$wheels | ForEach-Object {
  $h = Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName
  "$($h.Hash.ToLowerInvariant())`t$($_.Length)`t$($_.Name)"
} | Set-Content -LiteralPath $manifest -Encoding Ascii

if (Test-Path $archive) { Remove-Item -LiteralPath $archive -Force }
Log "archive $($runtime.Count) incremental Linux runtime wheels"
& tar.exe -cf $archive -C $wheelhouse @($runtime.Name)
if ($LASTEXITCODE -ne 0) { throw 'runtime tar creation failed' }
& tar.exe -rf $archive -C $Root wheelhouse-sha256.tsv
if ($LASTEXITCODE -ne 0) { throw 'manifest tar append failed' }
Log "runtime archive size $((Get-Item $archive).Length)"

Log 'transfer incremental runtime archive and exact lock to server'
& scp.exe -O -o BatchMode=yes $archive 'sophgo13:/home/sophgo13/cjl/storage/parameter-importance/tmp/linux-runtime-wheels.tar'
if ($LASTEXITCODE -ne 0) { throw 'runtime wheel SCP failed' }
& scp.exe -O -o BatchMode=yes $lock 'sophgo13:/home/sophgo13/cjl/parameter-importance/environment/requirements.lock'
if ($LASTEXITCODE -ne 0) { throw 'lock SCP failed' }

Log 'install complete wheelhouse with --no-index on server'
& ssh.exe -o BatchMode=yes sophgo13 bash /home/sophgo13/cjl/parameter-importance/scripts/server_install_linux_runtime.sh
if ($LASTEXITCODE -ne 0) { throw 'server offline install failed' }
Log 'incremental runtime transfer and server venv complete'
