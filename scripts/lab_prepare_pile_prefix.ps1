param(
  [string]$BrokerScript = 'C:\Users\cjl\parameter-importance-tools\lab_hf_broker.ps1',
  [string]$LogFile = 'C:\Users\cjl\parameter-importance-tools\pile-prefix.log'
)
$ErrorActionPreference = 'Stop'
$revision = '4647773ea142ab1ff5694602fa104bbf49088408'
$base = "https://hf-mirror.com/datasets/EleutherAI/pile-deduped-pythia-preshuffled/resolve/$revision"
$remote = '/home/sophgo13/cjl/storage/parameter-importance/datasets/pile-deduped-pythia-preshuffled'

$objects = @(
  @{
    Name='document-00000-of-00020.bin'; Size=30000000000L
    Sha='1ce355bd2683627d0ff689f8578115cf3df84bd1edf3410e6aca9705d31fc6ea'
  },
  @{
    Name='document.idx'; Size=1757184042L
    Sha='1d9fdd760295eb2007a4874440b27c559ca722239fa2814aa8a2ee6724b7852f'
  }
)

New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null
function Write-Log([string]$Message) {
  $line = "$(Get-Date -Format o) $Message"
  Add-Content -LiteralPath $LogFile -Value $line -Encoding UTF8
}

foreach ($obj in $objects) {
  $dest = "$remote/$($obj.Name)"
  & ssh.exe -o BatchMode=yes sophgo13 "mkdir -p '$remote' && test -f '$dest'"
  if ($LASTEXITCODE -eq 0) { Write-Log "already complete: $($obj.Name)"; continue }

  for ($attempt=1; $attempt -le 100; $attempt++) {
    Write-Log "start/refresh $($obj.Name), attempt $attempt"
    try {
      & $BrokerScript -ResolveUrl "$base/$($obj.Name)" -RemoteDestination $dest `
        -ExpectedSize $obj.Size -ExpectedSha256 $obj.Sha 2>&1 |
        ForEach-Object { Add-Content -LiteralPath $LogFile -Value $_ -Encoding UTF8 }
    } catch {
      Write-Log "segment ended: $($_.Exception.Message)"
    }
    & ssh.exe -o BatchMode=yes sophgo13 "test -f '$dest'"
    if ($LASTEXITCODE -eq 0) { Write-Log "complete: $($obj.Name)"; break }
    if ($attempt -eq 100) { throw "download did not finish after 100 URL refreshes: $($obj.Name)" }
    Start-Sleep -Seconds 5
  }
}
Write-Log 'Pile prefix complete.'
