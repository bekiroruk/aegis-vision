param([string]$Directory='artifacts/models/smolvlm-500m', [switch]$WithRuntime)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$destination=if([IO.Path]::IsPathRooted($Directory)){$Directory}else{Join-Path $repo $Directory}
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$revision='72e986006ef53e37cdd3f6d4241c90b0f01df376'
$base="https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/$revision"
$files=@(
    @{name='SmolVLM-500M-Instruct-Q8_0.gguf';sha256='9d4612de6a42214499e301494a3ecc2be0abdd9de44e663bda63f1152fad1bf4'},
    @{name='mmproj-SmolVLM-500M-Instruct-Q8_0.gguf';sha256='d1eb8b6b23979205fdf63703ed10f788131a3f812c7b1f72e0119d5d81295150'}
)
foreach($file in $files){
    $file.url="$base/$($file.name)";$path=Join-Path $destination $file.name
    if(-not(Test-Path -LiteralPath $path)){Invoke-WebRequest $file.url -OutFile $path -TimeoutSec 600}
    if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256){throw "VLM checksum mismatch: $path (not overwritten)"}
}
$card=Join-Path $destination 'MODEL_CARD.md'
if(-not(Test-Path -LiteralPath $card)){Invoke-WebRequest "$base/README.md" -OutFile $card -TimeoutSec 30}
@{revision=$revision;files=$files;license='Apache-2.0';runtime='llama.cpp b11552 CPU';alias='aegis-smolvlm-500m'} | ConvertTo-Json -Depth 5 | Out-File (Join-Path $destination 'manifest.json') -Encoding utf8
if($WithRuntime){
    New-Item -ItemType Directory -Force -Path (Join-Path $repo 'artifacts/deps') | Out-Null
    $runtime=Join-Path $repo 'artifacts/deps/llama-b11552'
    $zip=Join-Path $repo 'artifacts/deps/llama-b11552-bin-win-cpu-x64.zip'
    if(-not(Test-Path -LiteralPath $zip)){Invoke-WebRequest 'https://github.com/ggml-org/llama.cpp/releases/download/b11552/llama-b11552-bin-win-cpu-x64.zip' -OutFile $zip -TimeoutSec 180}
    if((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'c00741179811ff903b515eae28bdeef1f47af6e81c49920c4e68b62954e056d6'){throw 'llama.cpp archive checksum mismatch'}
    if(-not(Test-Path -LiteralPath $runtime)){Expand-Archive -LiteralPath $zip -DestinationPath $runtime}
    Write-Host "CPU runtime: $runtime"
}
Write-Host "Verified VLM bundle: $destination"
