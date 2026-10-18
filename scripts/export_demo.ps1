param([Parameter(Mandatory=$true)][string[]]$JobIds)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$ffmpeg = (Get-Command ffmpeg -ErrorAction Stop).Source
$ffprobe = (Get-Command ffprobe -ErrorAction Stop).Source
$output = Join-Path $repo ('outputs/presentation-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0,6))
New-Item -ItemType Directory -Path $output | Out-Null
$font = (Join-Path $env:WINDIR 'Fonts/arial.ttf').Replace('\','/').Replace(':','\:')
if (-not (Test-Path -LiteralPath (Join-Path $env:WINDIR 'Fonts/arial.ttf'))) { throw 'Arial font not found.' }
$manifest = @()
foreach ($id in $JobIds) {
    if ($id -notmatch '^[0-9]+-[0-9]+$') { throw 'Invalid job ID.' }
    $job = Invoke-RestMethod "http://127.0.0.1:8090/api/jobs/$id"
    if ($job.state -ne 'succeeded' -or $job.request.type -ne 'segment_video') { throw "Video job $id is not complete." }
    # Use a catalog entry, never an arbitrary path from a response.
    $catalog = (Invoke-RestMethod 'http://127.0.0.1:8090/api/media').videos
    if ($job.result.path -notin $catalog.path -or [IO.Path]::GetFileName($job.result.path) -ne $job.result.path) { throw 'Source is not a top-level catalog video.' }
    $source = Join-Path $repo ('artifacts/media/' + $job.result.path)
    $base = Join-Path $output $id
    Invoke-WebRequest "http://127.0.0.1:8090/api/jobs/$id/video.mp4" -OutFile "$base-analysis.mp4" -UseBasicParsing
    Invoke-WebRequest "http://127.0.0.1:8090/api/jobs/$id/tracks.csv" -OutFile "$base-tracks.csv" -UseBasicParsing
    Copy-Item -LiteralPath $source -Destination "$base-source.mp4"
    $filter = "[0:v]scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1[a];[1:v]scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1[b];[a][b]hstack=shortest=1,pad=1280:450:0:50:color=0x0c121b,drawtext=fontfile='$font':text='AEGISVISION  /  SOURCE':x=20:y=16:fontsize=20:fontcolor=0xe9f0f8,drawtext=fontfile='$font':text='SEGMENTATION + TRACKING':x=660:y=16:fontsize=20:fontcolor=0x69e0c0,drawtext=fontfile='$font':text='C++  |  YOLOv8-seg  |  Offline CPU  |  Track IDs are not unique person counts':x=20:y=423:fontsize=14:fontcolor=0x9eafc5"
    & $ffmpeg -hide_banner -loglevel error -nostdin -n -i $source -i "$base-analysis.mp4" -filter_complex $filter -an -c:v libx264 -crf 19 -preset fast -pix_fmt yuv420p -movflags +faststart "$base-comparison.mp4"
    if ($LASTEXITCODE -ne 0) { throw 'Comparison video export failed.' }
    $probe = & $ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=codec_name,width,height,nb_read_frames -of json "$base-comparison.mp4"
    if ($LASTEXITCODE -ne 0) { throw 'Comparison verification failed.' }
    $stream = ($probe | ConvertFrom-Json).streams[0]
    if ([int]$stream.nb_read_frames -ne $job.result.frames -or $stream.width -ne 1280 -or $stream.height -ne 450) { throw 'Comparison frame count or dimensions do not match.' }
    & $ffmpeg -hide_banner -loglevel error -nostdin -n -i "$base-comparison.mp4" -frames:v 1 -update 1 "$base-preview.jpg"
    if ($LASTEXITCODE -ne 0) { throw 'Preview export failed.' }
    $manifest += [pscustomobject]@{job_id=$id;source=$job.result.path;result=$job.result;comparison="$id-comparison.mp4";sha256=(Get-FileHash "$base-comparison.mp4" -Algorithm SHA256).Hash.ToLowerInvariant();probe=$stream}
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output 'manifest.json') -Encoding UTF8
Write-Host "Presentation videos, originals, analysis videos and CSV reports: $output"
