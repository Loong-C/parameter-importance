param(
  [Parameter(Mandatory=$true)][string]$ResolveUrl,
  [Parameter(Mandatory=$true)][string]$RemoteDestination,
  [Parameter(Mandatory=$true)][Int64]$ExpectedSize,
  [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$ExpectedSha256
)
$ErrorActionPreference = 'Stop'

# ResolveUrl is a public, revision-pinned HF URL. The resulting signed URL stays in memory.
$signed = & curl.exe -q --noproxy '*' -fsSIL -o NUL -w '%{url_effective}' $ResolveUrl
if ($LASTEXITCODE -ne 0 -or -not $signed.StartsWith('https://')) {
  throw 'Could not obtain a signed HTTPS URL from the public resolve URL.'
}

# The signed URL is sent on stdin, never as a remote command argument.
$signed | ssh.exe -o BatchMode=yes sophgo13 `
  bash /home/sophgo13/cjl/parameter-importance/scripts/server_xet_download.sh `
  $RemoteDestination $ExpectedSize $ExpectedSha256.ToLowerInvariant()
if ($LASTEXITCODE -ne 0) {
  throw 'Remote segment did not finish. Re-run the same command to refresh the URL and resume.'
}
