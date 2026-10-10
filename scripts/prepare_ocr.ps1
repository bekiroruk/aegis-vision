param([string]$Directory = 'artifacts/models/ocr-en', [switch]$WithSample)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$destination = if ([IO.Path]::IsPathRooted($Directory)) { $Directory } else { Join-Path $repo $Directory }
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$revision = '47534e27c9851bb1128ccc0102f1145e27f23f98'
$models = @(
    @{name='text_detection_en_ppocrv3_2023may.onnx'; folder='text_detection_ppocr'; sha256='03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587'},
    @{name='text_recognition_CRNN_EN_2021sep.onnx'; folder='text_recognition_crnn'; sha256='a84b1f6e11a65c2d733cb0cc1f014aae3f99051e3f11447dc282faa678eee544'}
)
foreach ($model in $models) {
    $path = Join-Path $destination $model.name
    $model.url = "https://media.githubusercontent.com/media/opencv/opencv_zoo/$revision/models/$($model.folder)/$($model.name)"
    if (-not (Test-Path -LiteralPath $path)) { Invoke-WebRequest $model.url -OutFile $path -TimeoutSec 180 }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $model.sha256) {
        throw "OCR model checksum mismatch: $path (not overwritten)"
    }
    $license = Join-Path $destination "$($model.folder)-LICENSE"
    if (-not (Test-Path -LiteralPath $license)) {
        Invoke-WebRequest "https://raw.githubusercontent.com/opencv/opencv_zoo/$revision/models/$($model.folder)/LICENSE" -OutFile $license -TimeoutSec 30
    }
}
@{schema_version=1; upstream_revision=$revision; models=$models; recognition_charset='0123456789abcdefghijklmnopqrstuvwxyz'; runtime='C++ OpenCV DNN CPU FP32'} |
    ConvertTo-Json -Depth 5 | Out-File -LiteralPath (Join-Path $destination 'manifest.json') -Encoding utf8
Write-Host "Verified OCR models: $destination (English alphanumeric recognition; no Turkish alphabet)"
if ($WithSample) {
    $sampleDirectory = Join-Path $repo 'artifacts/media/ocr'
    New-Item -ItemType Directory -Force -Path $sampleDirectory | Out-Null
    $sample = Join-Path $sampleDirectory 'text_det_test2.jpg'
    $sampleUrl = 'https://raw.githubusercontent.com/opencv/opencv_extra/22b4a7bc7cf5e4ddd3a0426eafc88bae28d3dfc3/testdata/dnn/text_det_test2.jpg'
    if (-not (Test-Path -LiteralPath $sample)) { Invoke-WebRequest $sampleUrl -OutFile $sample -TimeoutSec 30 }
    if ((Get-FileHash -LiteralPath $sample -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'a8e1910727ff5043e65f3039cdb22dbf23509dfd73293c81805fabca327fadc0') {
        throw 'OCR sample checksum mismatch (not overwritten)'
    }
    Write-Host "Verified upstream Canon test photo: $sample. Evaluation sample only; not redistributed by this repository."
}
