param([switch]$Stop, [switch]$OpenBrowser, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$stateDirectory = Join-Path $repo 'artifacts/service/demo-launcher'
$stateFile = Join-Path $stateDirectory 'processes.clixml'
$owned = @()
$started = @()
$lock = $null
$requiredFiles = @('build/live-seg/Release/aegisvision_server.exe','artifacts/deps/qdrant/qdrant.exe',
    'artifacts/deps/llama-b11552/llama-server.exe','configs/analysis.toml','configs/service-search.toml',
    'configs/image.toml','configs/qdrant-local.yaml','web/index.html','artifacts/media/pedestrians.mp4',
    'artifacts/models/yolov8n.onnx','artifacts/models/yolov8n-seg/yolov8n-seg.onnx',
    'artifacts/models/clip-vit-b32/manifest.json','artifacts/models/ocr-multi-v6/inference.onnx',
    'artifacts/models/ocr-multi-v6/inference.yml','artifacts/models/ocr-multi-v6/text_detection_en_ppocrv3_2023may.onnx',
    'artifacts/models/smolvlm-500m/SmolVLM-500M-Instruct-Q8_0.gguf',
    'artifacts/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf')
function Get-MissingDemoFiles {
    @($requiredFiles | Where-Object { -not (Test-Path -LiteralPath (Join-Path $repo $_) -PathType Leaf) })
}
if ($CheckOnly) {
    if ($Stop -or $OpenBrowser) { throw '-CheckOnly cannot be combined with -Stop or -OpenBrowser.' }
    $missing = @(Get-MissingDemoFiles)
    [pscustomobject]@{
        FilesPresent = ($missing.Count -eq 0)
        MissingFiles = $missing
        ProjectDirectory = $repo
        Note = 'File presence only; does not validate models, DLLs, services or GPU. Nothing was started or changed.'
    }
    return
}

function Get-OwnedProcess($record) {
    $process = Get-Process -Id $record.Id -ErrorAction SilentlyContinue
    if ($process -and $process.StartTime.ToUniversalTime().Ticks -eq $record.StartTicks -and
        $process.Path -eq $record.Path) { return $process }
    return $null
}
function Save-State {
    $script:owned | Export-Clixml -LiteralPath $stateFile
}
function Stop-Owned($records) {
    foreach ($record in @($records)[($records.Count-1)..0]) {
        if ($null -eq $record) { continue }
        $process = Get-OwnedProcess $record
        if ($process) {
            Stop-Process -InputObject $process -ErrorAction Stop
            $process.WaitForExit(10000) | Out-Null
            Write-Host "Durduruldu: $($record.Name)"
        }
    }
}
function Start-Component($name, $relativeExe, $arguments, $port, $url, $validate) {
    $exe = (Resolve-Path -LiteralPath (Join-Path $repo $relativeExe)).Path
    $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue)
    if ($listeners.Count) {
        foreach ($listener in $listeners) {
            $existing = Get-Process -Id $listener.OwningProcess -ErrorAction Stop
            if ($existing.Path -ne $exe -or $listener.LocalAddress -ne '127.0.0.1') {
                throw "$name portu ($port) baska bir uygulama veya adres tarafindan kullaniliyor."
            }
        }
        $status = Invoke-RestMethod -Uri $url -TimeoutSec 3
        if (-not (& $validate $status)) { throw "$name acik fakat beklenen yetenekler hazir degil." }
        Write-Host "Zaten hazir: $name"
        return
    }
    $process = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $stateDirectory "$name.stdout.log") `
        -RedirectStandardError (Join-Path $stateDirectory "$name.stderr.log")
    $record = [pscustomobject]@{Name=$name; Id=$process.Id; StartTicks=$process.StartTime.ToUniversalTime().Ticks; Path=$exe}
    $script:started += $record
    $script:owned += $record
    Save-State
    $deadline = [DateTime]::UtcNow.AddSeconds(120)
    do {
        if ($process.HasExited) { throw "$name kapandi. Gunluk: $stateDirectory/$name.stderr.log" }
        try {
            $status = Invoke-RestMethod -Uri $url -TimeoutSec 2
            if (& $validate $status) { Write-Host "Hazir: $name"; return }
        } catch { }
        Start-Sleep -Milliseconds 500
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "$name 120 saniye icinde hazir olmadi. Gunluk: $stateDirectory"
}

Push-Location $repo
try {
    New-Item -ItemType Directory -Path $stateDirectory -Force | Out-Null
    # One launcher at a time, including stop. The lock handle is released on exit.
    $lock = [IO.File]::Open((Join-Path $stateDirectory 'launcher.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
    if (Test-Path -LiteralPath $stateFile) { $owned = @(Import-Clixml -LiteralPath $stateFile | Where-Object { Get-OwnedProcess $_ }) }
    if ($Stop) {
        if ($owned.Count) { Stop-Owned $owned }
        $owned = @(); Save-State
        Write-Host 'Demo baslaticisinin actigi servisler durduruldu. Daha once acik servisler korunur.'
        return
    }
    $missing = @(Get-MissingDemoFiles)
    if ($missing.Count) { throw "Eksik dosyalar: $($missing -join ', '). docs/demo.md kurulum adimlarini kullanin." }
    $hashes = @{
        'SmolVLM-500M-Instruct-Q8_0.gguf'='9d4612de6a42214499e301494a3ecc2be0abdd9de44e663bda63f1152fad1bf4'
        'mmproj-SmolVLM-500M-Instruct-Q8_0.gguf'='d1eb8b6b23979205fdf63703ed10f788131a3f812c7b1f72e0119d5d81295150'
    }
    foreach ($name in $hashes.Keys) {
        if ((Get-FileHash -LiteralPath "artifacts/models/smolvlm-500m/$name" -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hashes[$name]) {
            throw "VLM dosya dogrulamasi basarisiz: $name"
        }
    }
    Start-Component 'qdrant' 'artifacts/deps/qdrant/qdrant.exe' '--config-path configs/qdrant-local.yaml' 6333 'http://127.0.0.1:6333/healthz' { param($value) $null -ne $value }
    Start-Component 'vlm' 'artifacts/deps/llama-b11552/llama-server.exe' '-m artifacts/models/smolvlm-500m/SmolVLM-500M-Instruct-Q8_0.gguf --mmproj artifacts/models/smolvlm-500m/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf --alias aegis-smolvlm-500m --host 127.0.0.1 --port 8092 -c 2048 -t 4 -np 1 -ngl 0 --no-mmproj-offload --no-webui --no-agent --cors-origins http://127.0.0.1:8092 --no-cors-credentials' 8092 'http://127.0.0.1:8092/v1/models' { param($value) @($value.data | Where-Object id -eq 'aegis-smolvlm-500m').Count -eq 1 }
    Start-Component 'aegisvision' 'build/live-seg/Release/aegisvision_server.exe' 'configs/service-search.toml configs/image.toml artifacts/media web 8090 artifacts/service/vlm-demo-jobs.sqlite --analysis-config configs/analysis.toml' 8090 'http://127.0.0.1:8090/api/health' { param($value) $value.status -eq 'ready' -and $value.segmentation_enabled -eq $true -and $value.ocr_unicode -eq $true -and $value.vlm_enabled -eq $true }
    Write-Host 'AegisVision hazir: http://127.0.0.1:8090'
    Write-Host 'Durdurmak icin: .\scripts\demo.ps1 -Stop'
    if ($OpenBrowser) { Start-Process 'http://127.0.0.1:8090' }
} catch {
    if ($started.Count) {
        Stop-Owned $started
        $owned = @($owned | Where-Object { Get-OwnedProcess $_ }); Save-State
    }
    throw
} finally {
    if ($lock) { $lock.Dispose() }
    Pop-Location
}
