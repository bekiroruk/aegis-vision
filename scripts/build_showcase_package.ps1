[CmdletBinding()]
param(
  [string]$Presentation = "outputs/sunum-paketi-20261019/AegisVision.pptx",
  [string]$PresentationAssets = "outputs/sunum-paketi-20261019/assets",
  [string]$PeopleClip = "outputs/presentation-20261018-200635/1791700823339-16-comparison.mp4",
  [string]$PeopleTracks = "outputs/presentation-20261018-200635/1791700823339-16-tracks.csv",
  [string]$TrafficClip = "outputs/presentation-20261018-200635/1791700823339-15-comparison.mp4",
  [string]$TrafficTracks = "outputs/presentation-20261018-200635/1791700823339-15-tracks.csv",
  [string]$SecurityClip = "outputs/presentation-20261023-000906-e3dc16/1791700823339-17-comparison.mp4",
  [string]$SecurityTracks = "outputs/presentation-20261023-000906-e3dc16/1791700823339-17-tracks.csv",
  [string]$Showcase = "outputs/AegisVision-showcase-20261023.mp4",
  [string]$OutputDirectory = "outputs/AegisVision-vitrin-paketi",
  [string]$Archive = "outputs/AegisVision-vitrin-20261023.zip"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$inputs = @($Presentation, "docs/assets/showcase.jpg", $PeopleClip, $PeopleTracks, $TrafficClip, $TrafficTracks,
  $SecurityClip, $SecurityTracks, $Showcase)
foreach ($inputPath in $inputs) {
  if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
    throw "Required file not found: $inputPath"
  }
}
if (-not (Test-Path -LiteralPath $PresentationAssets -PathType Container)) {
  throw "Presentation assets not found: $PresentationAssets"
}
if (Test-Path -LiteralPath $OutputDirectory) {
  throw "Output directory already exists: $OutputDirectory"
}
if (Test-Path -LiteralPath $Archive) {
  throw "Archive already exists: $Archive"
}

New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
Copy-Item -LiteralPath "docs/sunum/index.html", "docs/sunum/KULLANIM.md" -Destination $OutputDirectory
Copy-Item -LiteralPath $PresentationAssets -Destination $OutputDirectory -Recurse
Copy-Item -LiteralPath "docs/assets/showcase.jpg" -Destination (Join-Path $OutputDirectory "assets/showcase.jpg")
Copy-Item -LiteralPath $Presentation -Destination (Join-Path $OutputDirectory "AegisVision.pptx")

$copies = @(
  @($PeopleClip, "insanlar.mp4"), @($PeopleTracks, "insanlar-takip.csv"),
  @($TrafficClip, "trafik.mp4"), @($TrafficTracks, "trafik-takip.csv"),
  @($SecurityClip, "guvenlik.mp4"), @($SecurityTracks, "guvenlik-takip.csv"),
  @($Showcase, "showcase.mp4")
)
foreach ($copy in $copies) {
  Copy-Item -LiteralPath $copy[0] -Destination (Join-Path $OutputDirectory $copy[1])
}

$manifest = foreach ($copy in $copies) {
  $target = Join-Path $OutputDirectory $copy[1]
  $item = Get-Item -LiteralPath $target
  [ordered]@{
    file = $copy[1]
    bytes = $item.Length
    sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToLowerInvariant()
    source = $copy[0]
  }
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutputDirectory "manifest.json") -Encoding utf8
Compress-Archive -Path (Join-Path $OutputDirectory "*") -DestinationPath $Archive -CompressionLevel Optimal

Write-Host "Showcase package ready: $OutputDirectory"
Write-Host "Portable archive: $Archive"
Get-ChildItem -LiteralPath $OutputDirectory | Select-Object Name, Length
