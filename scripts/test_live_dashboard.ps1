param([int]$Port = 8090, [string]$Output = 'outputs/live-dashboard-test')
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$destination = [IO.Path]::GetFullPath($Output, $repo)
if (Test-Path -LiteralPath $destination) { throw 'Use a new output directory' }
$base = "http://127.0.0.1:$Port"
$current = (Invoke-RestMethod "$base/api/live").session
if ($current -and $current.active) { throw 'Stop the existing live session first; the test will not interrupt it' }
$presets = Invoke-RestMethod "$base/api/live/sources"
if (-not ($presets.sources.id -contains 'local-pedestrians') -or $presets.duration_seconds -lt 60) {
    throw 'Start the service with configs/live-preview.toml and the local pedestrians RTSP URL'
}
New-Item -ItemType Directory -Path $destination | Out-Null
$ffmpeg = (Get-Command ffmpeg -ErrorAction Stop).Source
$video = Join-Path $repo 'artifacts/media/pedestrians.mp4'
$children = [Collections.Generic.List[Diagnostics.Process]]::new()
$sessionId = ''
function Publish([string]$name) {
    $arguments = @('-hide_banner','-loglevel','warning','-re','-stream_loop','-1','-i',$video,'-an',
        '-c:v','libx264','-preset','ultrafast','-tune','zerolatency','-g','10','-keyint_min','10','-pix_fmt','yuv420p',
        '-f','rtsp','-rtsp_transport','tcp','rtsp://127.0.0.1:8554/pedestrians')
    $quoted = $arguments | ForEach-Object { '"' + $_.Replace('"','\"') + '"' }
    $child = Start-Process -FilePath $ffmpeg -ArgumentList ($quoted -join ' ') -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $destination "$name.out.log") -RedirectStandardError (Join-Path $destination "$name.err.log")
    $children.Add($child); return $child
}
function WaitSession([scriptblock]$predicate, [int]$seconds = 20) {
    $end = [DateTime]::UtcNow.AddSeconds($seconds)
    do {
        $snapshot = (Invoke-RestMethod "$base/api/live").session
        if (& $predicate $snapshot) { return $snapshot }
        if ($snapshot -and -not $snapshot.active) { throw "Live worker stopped: $($snapshot.state) $($snapshot.error)" }
        Start-Sleep -Milliseconds 100
    } while ([DateTime]::UtcNow -lt $end)
    throw 'Live HTTP state timed out'
}
try {
    $beforePublisher = Publish 'before'
    Start-Sleep -Seconds 1
    $accepted = Invoke-RestMethod "$base/api/live/start" -Method Post -ContentType application/json -Body '{"source_id":"local-pedestrians"}'
    $sessionId = $accepted.id
    $first = WaitSession { param($s) $s.has_preview }
    Invoke-WebRequest "$base/api/live/$sessionId/preview.jpg" -OutFile (Join-Path $destination 'before.jpg')
    Start-Sleep -Seconds 2
    $beforePublisher.Kill(); $beforePublisher.WaitForExit()
    $disconnected = WaitSession { param($s) $s.connection_state -eq 'reconnecting' -and -not $s.has_preview } 10
    $empty = Invoke-WebRequest "$base/api/live/$sessionId/preview.jpg"
    if ($empty.StatusCode -ne 204) { throw 'Disconnected preview still served an image' }
    $afterPublisher = Publish 'after'
    $recovered = WaitSession { param($s) $s.has_preview -and $s.sessions -gt $first.sessions }
    if ($recovered.tracking_epochs -le $first.tracking_epochs -or $recovered.queue_high_watermark -gt 1) {
        throw 'Tracking reset or bounded queue failed'
    }
    Invoke-WebRequest "$base/api/live/$sessionId/preview.jpg" -OutFile (Join-Path $destination 'recovered.jpg')
    $job = Invoke-RestMethod "$base/api/jobs" -Method Post -ContentType application/json -Body '{"type":"search","query":"a person walking on the street","limit":4}'
    $end = [DateTime]::UtcNow.AddSeconds(15)
    do {
        $search = Invoke-RestMethod "$base/api/jobs/$($job.id)"
        if ($search.state -eq 'succeeded') { break }
        if ($search.state -eq 'failed' -or [DateTime]::UtcNow -gt $end) { throw 'Concurrent semantic search failed' }
        Start-Sleep -Milliseconds 100
    } while ($true)
    @{before=$first;disconnected=$disconnected;recovered=$recovered;search_job=$search.id;search_results=$search.result.results.Count} |
        ConvertTo-Json -Depth 8 | Out-File -LiteralPath (Join-Path $destination 'report.json') -Encoding utf8
    Write-Host "PASS: live HTTP decoded/recovered, stale JPEG removed, epoch reset, search ran concurrently; results: $destination"
} finally {
    if ($sessionId) { Invoke-RestMethod "$base/api/live/$sessionId/stop" -Method Post -ContentType application/json -Body '{}' | Out-Null }
    foreach ($child in $children) { if (-not $child.HasExited) { $child.Kill(); $child.WaitForExit() } }
}
