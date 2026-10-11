param([string]$Directory='artifacts/models/ocr-multi-v6')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$destination=if([IO.Path]::IsPathRooted($Directory)){$Directory}else{Join-Path $repo $Directory}
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$revision='2612ab37152ae0a677521bae4e1e3d4fb4cf7c30'
$base="https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx/resolve/$revision"
$files=@(
    @{name='inference.onnx';url="$base/inference.onnx";sha256='9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6'},
    @{name='inference.yml';url="$base/inference.yml";sha256='66170210bad538e83fff3c4a3867e547d6bf20b50d64b20347c4b913f3034ea1'},
    @{name='text_detection_en_ppocrv3_2023may.onnx';url='https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx';sha256='03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587'}
)
foreach($file in $files) {
    $path=Join-Path $destination $file.name
    if(-not(Test-Path -LiteralPath $path)){Invoke-WebRequest $file.url -OutFile $path -TimeoutSec 180}
    if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256){throw "OCR checksum mismatch: $path (not overwritten)"}
}
foreach($item in @(
    @{name='MODEL_CARD.md';url="$base/README.md"},
    @{name='DETECTOR_LICENSE';url='https://raw.githubusercontent.com/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/text_detection_ppocr/LICENSE'}
)) {
    $path=Join-Path $destination $item.name
    if(-not(Test-Path -LiteralPath $path)){Invoke-WebRequest $item.url -OutFile $path -TimeoutSec 30}
}
@{schema_version=1;recognizer_revision=$revision;files=$files;recognizer_license='Apache-2.0 (official pinned model card)';runtime='C++ CPU ONNX Runtime + OpenCV';dictionary='Pinned inference.yml + appended space; blank=0'} |
    ConvertTo-Json -Depth 5 | Out-File -LiteralPath (Join-Path $destination 'manifest.json') -Encoding utf8
Write-Host "Verified multilingual OCR bundle: $destination"
