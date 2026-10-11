param([ValidateRange(1,65535)][int]$Port=8092)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
Push-Location $repo
try {
    # Verify the pinned local model bytes before starting; does not download implicitly.
    $models=@{
        'SmolVLM-500M-Instruct-Q8_0.gguf'='9d4612de6a42214499e301494a3ecc2be0abdd9de44e663bda63f1152fad1bf4'
        'mmproj-SmolVLM-500M-Instruct-Q8_0.gguf'='d1eb8b6b23979205fdf63703ed10f788131a3f812c7b1f72e0119d5d81295150'
    }
    foreach($name in $models.Keys){
        if((Get-FileHash -LiteralPath "artifacts/models/smolvlm-500m/$name" -Algorithm SHA256).Hash.ToLowerInvariant() -ne $models[$name]){throw "VLM hash mismatch: $name"}
    }
    & ./artifacts/deps/llama-b11552/llama-server.exe -m artifacts/models/smolvlm-500m/SmolVLM-500M-Instruct-Q8_0.gguf --mmproj artifacts/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf --alias aegis-smolvlm-500m --host 127.0.0.1 --port $Port -c 2048 -t 4 -np 1 -ngl 0 --no-mmproj-offload --no-webui --no-agent --cors-origins "http://127.0.0.1:$Port" --no-cors-credentials
    if($LASTEXITCODE -ne 0){throw "VLM server exited: $LASTEXITCODE"}
} finally {Pop-Location}
