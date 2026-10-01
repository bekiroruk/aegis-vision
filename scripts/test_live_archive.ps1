param([int]$Port = 8090, [string]$Output = 'outputs/live-archive-test')
$ErrorActionPreference = 'Stop'
$repoDirectory = Split-Path $PSScriptRoot -Parent
$destination = [IO.Path]::GetFullPath($Output,$repoDirectory)
if (Test-Path -LiteralPath $destination) { throw 'Use a new output directory' }
$base = "http://127.0.0.1:$Port"
$current = (Invoke-RestMethod "$base/api/live").session
if ($current -and $current.active) { throw 'Existing live session is active; this test will not interrupt it' }
$presets = Invoke-RestMethod "$base/api/live/sources"
if (-not $presets.archive_available -or -not ($presets.sources.id -contains 'local-pedestrians')) {
    throw 'Start the live-enabled service with the local pedestrians RTSP source'
}
$existing = Get-CimInstance Win32_Process -Filter "Name='ffmpeg.exe'" | Where-Object { $_.CommandLine -like '*rtsp://127.0.0.1:8554/pedestrians*' }
if ($existing) { throw 'An existing publisher uses this RTSP URL; test refuses to interrupt it' }
$ffmpeg = (Get-Command ffmpeg -ErrorAction Stop).Source
$video = Join-Path $repoDirectory 'artifacts/media/pedestrians.mp4'
if (-not (Test-Path -LiteralPath $video -PathType Leaf)) { throw 'Local pedestrians video is missing' }
New-Item -ItemType Directory -Path $destination | Out-Null
$arguments = @('-hide_banner','-loglevel','warning','-re','-stream_loop','-1','-i',$video,'-an',
    '-c:v','libx264','-preset','ultrafast','-tune','zerolatency','-g','10','-keyint_min','10','-pix_fmt','yuv420p',
    '-f','rtsp','-rtsp_transport','tcp','rtsp://127.0.0.1:8554/pedestrians')
$quoted = $arguments | ForEach-Object { '"' + $_.Replace('"','\"') + '"' }
$publisher = $null
$sessionId = ''
function WaitValue([scriptblock]$read, [scriptblock]$predicate, [int]$seconds = 45) {
    $deadline = [DateTime]::UtcNow.AddSeconds($seconds)
    do {
        $value = & $read
        if (& $predicate $value) { return $value }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)
    throw 'Live archive integration state timed out'
}
try {
    $publisher = Start-Process -FilePath $ffmpeg -ArgumentList ($quoted -join ' ') -WorkingDirectory $repoDirectory -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $destination 'publisher.out.log') -RedirectStandardError (Join-Path $destination 'publisher.err.log')
    $accepted = Invoke-RestMethod "$base/api/live/start" -Method Post -ContentType application/json -Body '{"source_id":"local-pedestrians","archive":true}'
    $sessionId = $accepted.id
    $preview = WaitValue { (Invoke-RestMethod "$base/api/live").session } { param($s) $s.has_preview }
    Invoke-WebRequest "$base/api/live/$sessionId/preview.jpg" -OutFile (Join-Path $destination 'live-preview.jpg')
    $first = WaitValue { Invoke-RestMethod "$base/api/live/archive" } { param($a) @($a.segments | Where-Object session_id -eq $sessionId).Count -ge 1 }
    Invoke-RestMethod "$base/api/live/$sessionId/stop" -Method Post -ContentType application/json -Body '{}' | Out-Null
    $stopped = WaitValue { (Invoke-RestMethod "$base/api/live").session } { param($s) -not $s.active }
    $finished = WaitValue { Invoke-RestMethod "$base/api/live/archive" } {
        param($a)
        $mine = @($a.segments | Where-Object session_id -eq $sessionId)
        $mine.Count -ge 1 -and @($mine | Where-Object index_state -ne 'succeeded').Count -eq 0
    } 90
    $segments = @($finished.segments | Where-Object session_id -eq $sessionId)
    $indexJobs = @($segments | ForEach-Object { Invoke-RestMethod "$base/api/jobs/$($_.job_id)" })
    $indexedItems = ($indexJobs | ForEach-Object { $_.result.indexed_items } | Measure-Object -Sum).Sum
    foreach ($segment in $segments) {
        if (-not $segment.media_path -or $segment.frames -lt 1) { throw 'Finalized MP4 is missing' }
        $range = Invoke-WebRequest "$base/media/$($segment.media_path)" -Headers @{Range='bytes=0-99'}
        if ($range.StatusCode -ne 206) { throw 'Archive MP4 byte range failed' }
    }
    $search = Invoke-RestMethod "$base/api/jobs" -Method Post -ContentType application/json -Body '{"type":"search","query":"a person walking on the street","scope":"live","limit":8}'
    $found = WaitValue { Invoke-RestMethod "$base/api/jobs/$($search.id)" } { param($j) $j.state -in @('succeeded','failed','cancelled') }
    if ($found.state -ne 'succeeded' -or $found.result.results.Count -eq 0) { throw 'Real CLIP live archive search failed' }
    foreach ($match in $found.result.results) {
        if ($match.metadata.origin -ne 'live_archive' -or -not $match.media_path) { throw 'Result escaped live archive scope' }
    }
    Invoke-WebRequest "$base/api/preview/$($search.id)/0.jpg" -OutFile (Join-Path $destination 'search-preview.jpg')
    $found | ConvertTo-Json -Depth 20 | Out-File (Join-Path $destination 'search.json') -Encoding utf8
    @{session=$stopped;segments=$segments;index_jobs=$indexJobs;indexed_items=$indexedItems;search_job=$search.id;search_results=$found.result.results.Count} |
        ConvertTo-Json -Depth 20 | Out-File (Join-Path $destination 'report.json') -Encoding utf8
    Write-Output "Live archive PASS: $($segments.Count) encoded/indexed clips; $indexedItems indexed objects; $($found.result.results.Count) live-scoped results; evidence=$destination"
}
finally {
    if ($sessionId) { try { Invoke-RestMethod "$base/api/live/$sessionId/stop" -Method Post -ContentType application/json -Body '{}' | Out-Null } catch {} }
    if ($publisher -and -not $publisher.HasExited) { $publisher.Kill(); $publisher.WaitForExit() }
}
