param(
    [string]$BaseUrl = 'http://127.0.0.1:8090',
    [string]$Video = 'pedestrians.mp4',
    [ValidateRange(0,10000)][int]$FrameIndex = 0,
    [Parameter(Mandatory=$true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Output directory must not exist' }
$health = Invoke-RestMethod "$BaseUrl/api/health" -TimeoutSec 5
if (-not $health.segmentation_enabled) { throw 'Service segmentation is disabled' }
$body = @{type='segment_frame';path=$Video;frame_index=$FrameIndex} | ConvertTo-Json -Compress
$job = Invoke-RestMethod "$BaseUrl/api/jobs" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10
$deadline = (Get-Date).AddSeconds(90)
while ($job.state -in @('queued','running')) {
    if ((Get-Date) -gt $deadline) {
        Invoke-RestMethod "$BaseUrl/api/jobs/$($job.id)/cancel" -Method Post -ContentType 'application/json' -Body '{}' -TimeoutSec 5 | Out-Null
        throw 'Segmentation timed out; cancellation requested'
    }
    Start-Sleep -Milliseconds 200
    $job = Invoke-RestMethod "$BaseUrl/api/jobs/$($job.id)" -TimeoutSec 10
}
if ($job.state -ne 'succeeded') { throw "Segmentation failed: $($job.error)" }
$result = $job.result
if ($result.frame_index -ne $FrameIndex -or $result.tracking -or $result.width -lt 1 -or $result.height -lt 1) {
    throw 'Unexpected single-frame result'
}
foreach ($instance in $result.instances) {
    if ($instance.segmentation.size[0] -ne $result.height -or $instance.segmentation.size[1] -ne $result.width) { throw 'Mask dimensions differ' }
    $total = 0L; $positive = 0L; $offset = 0
    foreach ($run in $instance.segmentation.counts) {
        if ($run -lt 0 -or $run -ne [math]::Floor($run)) { throw 'Invalid RLE count' }
        $total += $run
        if ($offset % 2 -eq 1) { $positive += $run }
        $offset++
    }
    if ($total -ne $result.width*$result.height -or $positive -ne $instance.mask_pixels) { throw 'RLE coverage/area mismatch' }
}
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$job | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'job.json') -Encoding UTF8
Write-Output "PASS: HTTP job=$($job.id) frame=$FrameIndex masks=$($result.instances.Count) size=$($result.width)x$($result.height)"
