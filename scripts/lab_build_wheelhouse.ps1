param(
  [string]$Root = 'C:\Users\cjl\parameter-importance-wheelhouse',
  [string]$Python = 'C:\Users\cjl\Apps\Python312\python.exe',
  [string]$LockScript = 'C:\Users\cjl\parameter-importance-wheelhouse\lock_from_pip_report.py'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$log = Join-Path $Root 'build.log'
$inputFile = Join-Path $Root 'requirements.in'
$report = Join-Path $Root 'resolution-report.json'
$lock = Join-Path $Root 'requirements.lock'
$wheelhouse = Join-Path $Root 'wheelhouse'
$manifest = Join-Path $Root 'wheelhouse-sha256.tsv'
$archive = Join-Path $Root 'wheelhouse.tar'
$torchIndex = 'https://download.pytorch.org/whl/cu126'

New-Item -ItemType Directory -Force $Root,$wheelhouse | Out-Null
function Log([string]$Message) { Add-Content -LiteralPath $log -Encoding UTF8 -Value "$(Get-Date -Format o) $Message" }

Log 'resolve Linux CPython 3.12 dependency graph'
& $Python -m pip install --dry-run --ignore-installed --only-binary=:all: `
  --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 `
  --python-version 3.12 --implementation cp --abi cp312 `
  --extra-index-url $torchIndex --report $report --requirement $inputFile *>> $log
if ($LASTEXITCODE -ne 0) { throw 'pip resolution failed; see build.log' }

& $Python $LockScript --report $report --output $lock
if ($LASTEXITCODE -ne 0) { throw 'lock generation failed' }
$pinCount = (Get-Content $lock | Where-Object { $_ -and -not $_.StartsWith('--') }).Count
Log "resolved $pinCount pinned distributions"

Log 'download wheelhouse'
& $Python -m pip download --only-binary=:all: `
  --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 `
  --python-version 3.12 --implementation cp --abi cp312 `
  --extra-index-url $torchIndex --dest $wheelhouse --requirement $lock *>> $log
if ($LASTEXITCODE -ne 0) { throw 'wheel download failed; see build.log' }

Get-ChildItem -File $wheelhouse | Sort-Object Name | ForEach-Object {
  $h = Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName
  "$($h.Hash.ToLowerInvariant())`t$($_.Length)`t$($_.Name)"
} | Set-Content -LiteralPath $manifest -Encoding Ascii
Log "hashed $((Get-ChildItem -File $wheelhouse).Count) wheels"

if (Test-Path $archive) { Remove-Item -LiteralPath $archive -Force }
& tar.exe -cf $archive -C $Root wheelhouse requirements.lock resolution-report.json wheelhouse-sha256.tsv
if ($LASTEXITCODE -ne 0) { throw 'tar creation failed' }
Log "archive size $((Get-Item $archive).Length)"

Log 'transfer wheelhouse archive to server'
& scp.exe -O -o BatchMode=yes $archive 'sophgo13:/home/sophgo13/cjl/storage/parameter-importance/tmp/wheelhouse.tar' *>> $log
if ($LASTEXITCODE -ne 0) { throw 'wheelhouse SCP failed' }
& scp.exe -O -o BatchMode=yes $lock 'sophgo13:/home/sophgo13/cjl/parameter-importance/environment/requirements.lock' *>> $log
if ($LASTEXITCODE -ne 0) { throw 'requirements.lock SCP failed' }

Log 'extract and install offline environment on server'
& ssh.exe -o BatchMode=yes sophgo13 bash /home/sophgo13/cjl/parameter-importance/scripts/server_install_wheelhouse.sh *>> $log
if ($LASTEXITCODE -ne 0) { throw 'remote offline install failed' }
Log 'wheelhouse and offline venv complete'
