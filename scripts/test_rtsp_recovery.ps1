param(
    [string]$Output = "outputs/rtsp-recovery",
    [string]$BuildDir = "build/search/Release",
    [string]$MediaMtx = "artifacts/deps/mediamtx/mediamtx.exe",
    [string]$Ffmpeg = "ffmpeg",
    [string]$Video = "artifacts/media/pedestrians.mp4"
)
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$outputPath = [IO.Path]::GetFullPath($Output, $repo)
if (Test-Path -LiteralPath $outputPath) { throw "Use a new output directory: $outputPath" }
$cli = [IO.Path]::GetFullPath("$BuildDir/aegisvision_stream.exe", $repo)
$relayExe = [IO.Path]::GetFullPath($MediaMtx, $repo)
$videoPath = [IO.Path]::GetFullPath($Video, $repo)
foreach ($path in @($cli, $relayExe, $videoPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing file: $path" }
}
$ffmpegExe = (Get-Command $Ffmpeg -ErrorAction Stop).Source
New-Item -ItemType Directory -Path $outputPath | Out-Null
$resultPath = Join-Path $outputPath "result"
$url = "rtsp://127.0.0.1:8554/pedestrians"
$children = [Collections.Generic.List[Diagnostics.Process]]::new()
function Spawn([string]$exe, [string[]]$argv, [string]$name) {
    # Quote each argument so local paths containing spaces stay intact.
    $quoted = $argv | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }
    $process = Start-Process -FilePath $exe -ArgumentList ($quoted -join ' ') -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $outputPath "$name.out.log") -RedirectStandardError (Join-Path $outputPath "$name.err.log")
    $children.Add($process)
    return $process
}
function Publisher([string]$name) {
    return Spawn $ffmpegExe @('-hide_banner','-loglevel','warning','-re','-stream_loop','-1','-i',$videoPath,
        '-an','-c:v','libx264','-preset','ultrafast','-tune','zerolatency','-g','10','-keyint_min','10',
        '-pix_fmt','yuv420p','-f','rtsp','-rtsp_transport','tcp',$url) $name
}
try {
    # Do not reuse or stop someone else's server on this port.
    if (Get-NetTCPConnection -LocalPort 8554 -State Listen -ErrorAction SilentlyContinue) {
        throw "Port 8554 is in use; stop your test relay before running this isolated test"
    }
    $relay = Spawn $relayExe @((Join-Path $repo 'configs/rtsp-local.yml')) 'relay'
    $deadline = [DateTime]::UtcNow.AddSeconds(10)
    while (-not (Get-NetTCPConnection -LocalPort 8554 -State Listen -ErrorAction SilentlyContinue)) {
        if ($relay.HasExited -or [DateTime]::UtcNow -gt $deadline) { throw 'Relay did not start' }
        Start-Sleep -Milliseconds 200
    }
    $first = Publisher 'publisher-before'
    Start-Sleep -Seconds 2
    $analyzer = Spawn $cli @((Join-Path $repo 'configs/stream.toml'), $url, $resultPath) 'analyzer'
    $deadline = [DateTime]::UtcNow.AddSeconds(15)
    while (-not (Test-Path -LiteralPath (Join-Path $resultPath 'preview.jpg'))) {
        if ($analyzer.HasExited -or $first.HasExited -or [DateTime]::UtcNow -gt $deadline) { throw 'No analyzed RTSP frame' }
        Start-Sleep -Milliseconds 100
    }
    Start-Sleep -Seconds 5
    $first.Kill(); $first.WaitForExit()
    Start-Sleep -Seconds 5
    $second = Publisher 'publisher-after'
    if (-not $analyzer.WaitForExit(55000)) { throw 'Analyzer did not respect bounded duration/timeouts' }
    if ($analyzer.ExitCode -ne 0) { throw "Analyzer failed; inspect $outputPath logs" }
    $summary = Get-Content -LiteralPath (Join-Path $resultPath 'summary.json') -Raw | ConvertFrom-Json
    if ($summary.sessions -lt 2 -or $summary.read_failures -lt 1 -or $summary.tracking_epochs -lt 2) {
        throw 'Disconnect/reconnect or tracking reset was not observed'
    }
    if ($summary.queue_high_watermark -gt $summary.queue_capacity) { throw 'Queue exceeded its bound' }
    $timeline = Import-Csv -LiteralPath (Join-Path $resultPath 'frames.csv')
    $sessions = @($timeline.source_session | Sort-Object -Unique)
    if ($sessions.Count -lt 2 -or $timeline.Count -ne $summary.processed_frames) { throw 'Analyzed frames missing from recovered session' }
    & $ffmpegExe -hide_banner -v error -i (Join-Path $resultPath 'tracked.avi') -f null -
    if ($LASTEXITCODE -ne 0) { throw 'Recording is not fully decodable' }
    Write-Host "PASS: RTSP publisher killed/restarted; sessions=$($summary.sessions), analyzed=$($summary.processed_frames), decoded=$($summary.decoded_frames), queue=$($summary.queue_high_watermark)"
    Write-Host "Outputs: $resultPath"
} finally {
    # Only processes created by this invocation; no name-based or broad cleanup.
    foreach ($child in $children) { if (-not $child.HasExited) { $child.Kill(); $child.WaitForExit() } }
}
