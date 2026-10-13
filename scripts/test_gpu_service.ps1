param([int]$Port=8091,[string]$Output='outputs/gpu-service-test-v1')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$destination=[IO.Path]::GetFullPath($Output,$repo)
if(Test-Path -LiteralPath $destination){throw 'Use a new output directory'}
$base="http://127.0.0.1:$Port"
function RequireGpu($backend) {
    if($backend.engine -ne 'tensorrt' -or $backend.device -ne 'cuda:0' -or $backend.precision -ne 'fp16-mixed'){
        throw 'Expected native TensorRT FP16, not CPU or INT8'
    }
}
$health=Invoke-RestMethod "$base/api/health"
RequireGpu $health.detector_backend
$presets=Invoke-RestMethod "$base/api/live/sources"
RequireGpu $presets.backend
if($presets.analysis_mode -ne 'detection'){throw 'This test requires live detection, not CPU live masks'}
if((Invoke-RestMethod "$base/api/live").session.active){throw 'Existing live session will not be interrupted'}
New-Item -ItemType Directory -Path $destination | Out-Null
$sessionId=''
try {
    $live=Invoke-RestMethod "$base/api/live/start" -Method Post -ContentType application/json -Body '{"source_id":"local-pedestrians","archive":false}'
    $sessionId=$live.id
    $deadline=[DateTime]::UtcNow.AddSeconds(35)
    do {
        $first=(Invoke-RestMethod "$base/api/live").session
        if($first.has_preview){break}
        if($first.state -eq 'failed' -or [DateTime]::UtcNow -gt $deadline){throw 'GPU live preview failed'}
        Start-Sleep -Milliseconds 200
    } while($true)
    RequireGpu $first.backend
    Invoke-WebRequest "$base/api/live/$sessionId/preview.jpg" -OutFile (Join-Path $destination 'gpu-live.jpg')
    $job=Invoke-RestMethod "$base/api/jobs" -Method Post -ContentType application/json -Body '{"type":"index_video","path":"pedestrians.mp4","stride":10,"max_frames":30}'
    $deadline=[DateTime]::UtcNow.AddSeconds(90)
    do {
        $result=Invoke-RestMethod "$base/api/jobs/$($job.id)"
        if($result.state -eq 'succeeded'){break}
        if($result.state -in @('failed','cancelled') -or [DateTime]::UtcNow -gt $deadline){throw "GPU file job failed: $($result.error)"}
        Start-Sleep -Milliseconds 200
    } while($true)
    if($result.result.indexed_items -lt 1){throw 'GPU file detector indexed no objects'}
    $after=(Invoke-RestMethod "$base/api/live").session
    if(-not $after.has_preview -or $after.processed_frames -le $first.processed_frames -or $after.queue_high_watermark -gt 1){
        throw 'Live worker did not progress alongside file GPU worker'
    }
    RequireGpu $after.backend
    $search=Invoke-RestMethod "$base/api/jobs" -Method Post -ContentType application/json -Body '{"type":"search","query":"a person walking on the street","limit":4}'
    $deadline=[DateTime]::UtcNow.AddSeconds(30)
    do {
        $found=Invoke-RestMethod "$base/api/jobs/$($search.id)"
        if($found.state -eq 'succeeded'){break}
        if($found.state -eq 'failed' -or [DateTime]::UtcNow -gt $deadline){throw 'Search failed'}
        Start-Sleep -Milliseconds 200
    } while($true)
    if($found.result.results.Count -lt 1){throw 'Search found no GPU-indexed objects'}
    $wrongSignature=@($found.result.results | Where-Object {$_.metadata.detector_signature -notlike 'tensorrt-fp16-v1:*'})
    if($wrongSignature.Count){throw 'Search result detector identity is not GPU-specific'}
    Invoke-RestMethod "$base/api/live/$sessionId/stop" -Method Post -ContentType application/json -Body '{}' | Out-Null
    $deadline=[DateTime]::UtcNow.AddSeconds(15)
    do {
        $stopped=(Invoke-RestMethod "$base/api/live").session
        if(-not $stopped.active){break}
        if([DateTime]::UtcNow -gt $deadline){throw 'GPU live stop timed out'}
        Start-Sleep -Milliseconds 100
    } while($true)
    if($stopped.has_preview -or $stopped.state -ne 'stopped'){throw 'GPU preview remained stale after stop'}
    $sessionId=''
    @{health=$health;first=$first;after=$after;index_job=$result;search_job=$found;stopped=$stopped;status='passed'} |
        ConvertTo-Json -Depth 16 | Out-File -LiteralPath (Join-Path $destination 'report.json') -Encoding utf8
    Write-Host "PASS: concurrent native GPU file/live detection, CLIP search, distinct cache identity, clean stop: $destination"
} finally {
    if($sessionId){Invoke-RestMethod "$base/api/live/$sessionId/stop" -Method Post -ContentType application/json -Body '{}' | Out-Null}
}
