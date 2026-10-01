$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$directory = Join-Path $repo 'artifacts/deps/mediamtx'
New-Item -ItemType Directory -Force -Path $directory | Out-Null
$archive = Join-Path $directory 'mediamtx.zip'
# Official v1.21.1 release checksum, pinned rather than trusting a mutable latest tag.
$expected = 'faa97974861eb75a68b5aa326c78e7e7a6f670b5ef191bace78e715130381f23'
if (-not (Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest 'https://github.com/bluenviron/mediamtx/releases/download/v1.21.1/mediamtx_v1.21.1_windows_amd64.zip' -OutFile $archive
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw 'MediaMTX checksum mismatch' }
Expand-Archive -LiteralPath $archive -DestinationPath $directory -Force
Write-Host "Verified local RTSP test relay: $directory/mediamtx.exe (no system install)"
