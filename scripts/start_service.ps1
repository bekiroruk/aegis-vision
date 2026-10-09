param(
    [string]$MediaDirectory = "artifacts/media",
    [ValidateRange(1, 65535)][int]$Port = 8090,
    [string]$JobDatabase = "artifacts/service/jobs.sqlite",
    [string]$LiveUrl = "",
    [string]$LiveConfig = "configs/live-preview.toml",
    [string]$Ffmpeg = "ffmpeg",
    [string]$SegmentationModel = ""
)
$ErrorActionPreference = "Stop"
$repoDirectory = Split-Path $PSScriptRoot -Parent
Push-Location $repoDirectory
try {
    foreach ($required in @("build/search/Release/aegisvision_server.exe", "artifacts/models/yolov8n.onnx", "artifacts/models/clip-vit-b32/manifest.json")) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Eksik dosya: $required. Kurulum icin docs/service.md dosyasina bakin."
        }
    }
    if (-not (Test-Path -LiteralPath $MediaDirectory -PathType Container)) {
        New-Item -ItemType Directory -Path $MediaDirectory | Out-Null
    }
    try { Invoke-RestMethod -Uri http://127.0.0.1:6333/healthz -TimeoutSec 3 | Out-Null }
    catch { throw "Qdrant calismiyor. Ayri terminalde artifacts/deps/qdrant/qdrant.exe --config-path configs/qdrant-local.yaml komutunu baslatin." }
    $serverArguments = @('configs/service-search.toml','configs/image.toml',$MediaDirectory,'web',$Port,$JobDatabase)
    if ($LiveUrl) { $serverArguments += @($LiveConfig,$LiveUrl,$Ffmpeg) }
    if ($SegmentationModel) {
        if (-not (Test-Path -LiteralPath $SegmentationModel -PathType Leaf)) { throw "Segmentation modeli bulunamadi: $SegmentationModel" }
        $serverArguments += @('--segment-model',$SegmentationModel)
    }
    & ./build/search/Release/aegisvision_server.exe @serverArguments
    if ($LASTEXITCODE -ne 0) { throw "Video servisi hata koduyla kapandi: $LASTEXITCODE" }
}
finally { Pop-Location }
