param([string]$Root = 'C:\Users\cjl\parameter-importance-wheelhouse')
$ErrorActionPreference = 'Stop'
$wheelhouse = Join-Path $Root 'wheelhouse'
$lock = Join-Path $Root 'requirements.lock'
$manifest = Join-Path $Root 'wheelhouse-sha256.tsv'
$archive = Join-Path $Root 'cufile-wheel.tar'
$log = Join-Path $Root 'cufile-finish.log'
function Log([string]$Message) { Add-Content -LiteralPath $log -Encoding UTF8 -Value "$(Get-Date -Format o) $Message" }

Log 'wait for the 18-wheel runtime transfer task to finish'
while ((Get-ScheduledTask -TaskName 'CjlLinuxRuntime').State -eq 'Running') {
  Start-Sleep -Seconds 15
}

$wheels = @(Get-ChildItem -File -LiteralPath $wheelhouse -Filter '*.whl' | Sort-Object Name)
if ($wheels.Count -ne 89) { throw "expected 89 wheels, found $($wheels.Count)" }
$cufile = @($wheels | Where-Object { $_.Name -match '^nvidia_cufile_cu12-' })
if ($cufile.Count -ne 1) { throw "expected one cuFile wheel, found $($cufile.Count)" }
$cufileName = $cufile[0].Name

Log 'hash the final 89-wheel set'
$wheels | ForEach-Object {
  $h = Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName
  "$($h.Hash.ToLowerInvariant())`t$($_.Length)`t$($_.Name)"
} | Set-Content -LiteralPath $manifest -Encoding Ascii

if (Test-Path $archive) { Remove-Item -LiteralPath $archive -Force }
& tar.exe -cf $archive -C $wheelhouse $cufileName
if ($LASTEXITCODE -ne 0) { throw 'cuFile tar creation failed' }
& tar.exe -rf $archive -C $Root wheelhouse-sha256.tsv
if ($LASTEXITCODE -ne 0) { throw 'manifest tar append failed' }
Log "cuFile archive size $((Get-Item $archive).Length)"

& scp.exe -O -o BatchMode=yes $archive 'sophgo13:/home/sophgo13/cjl/storage/parameter-importance/tmp/cufile-wheel.tar'
if ($LASTEXITCODE -ne 0) { throw 'cuFile SCP failed' }
& scp.exe -O -o BatchMode=yes $lock 'sophgo13:/home/sophgo13/cjl/parameter-importance/environment/requirements.lock'
if ($LASTEXITCODE -ne 0) { throw '89-package lock SCP failed' }
Log 'cuFile transferred; complete the offline environment install'
& ssh.exe -o BatchMode=yes sophgo13 bash /home/sophgo13/cjl/parameter-importance/scripts/server_install_cufile.sh
if ($LASTEXITCODE -ne 0) { throw 'final server offline install failed' }
Log 'final 89-wheel server venv complete'
