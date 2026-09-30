# Local, version-pinned Windows x64 development dependencies. No system PATH edits.
param([switch]$WithQdrant)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$dependencyRoot = Join-Path $projectRoot 'artifacts/deps'
New-Item -ItemType Directory -Force $dependencyRoot | Out-Null
function Get-VerifiedArchive($Name, $Url, $Sha256, $Folder) {
    $archive = Join-Path $dependencyRoot "$Name.zip"
    if (-not (Test-Path -LiteralPath $archive)) {
        & curl.exe -fL $Url -o $archive
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $Name" }
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $Sha256) {
        throw "SHA256 mismatch: $archive (left untouched)"
    }
    $destination = Join-Path $dependencyRoot $Folder
    if (-not (Test-Path -LiteralPath $destination)) { Expand-Archive -LiteralPath $archive -DestinationPath $destination }
}
Get-VerifiedArchive 'ort' 'https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-1.23.2.zip' '0B38DF9AF21834E41E73D602D90DB5CB06DBD1CA618948B8F1D66D607AC9F3CD' 'ort'
Get-VerifiedArchive 'icu' 'https://github.com/unicode-org/icu/releases/download/release-77-1/icu4c-77_1-Win64-MSVC2022.zip' '6B62471ED2895959D6A85C64C58572AC677734547FE8383E6E3FD706A06DC3FA' 'icu'
if ($WithQdrant) {
    Get-VerifiedArchive 'qdrant' 'https://github.com/qdrant/qdrant/releases/download/v1.12.5/qdrant-x86_64-pc-windows-msvc.zip' 'DD1918F38BF587550C85A56D9EE02A43FA639AA3F2EED14ABC458F8D143CBD0A' 'qdrant'
}
Write-Output "Dependencies ready under $dependencyRoot"
