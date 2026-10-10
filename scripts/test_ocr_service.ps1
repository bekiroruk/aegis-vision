param(
    [int]$Port=8090,
    [string]$Video='ocr-canon.mp4',
    [ValidateRange(0,10000)][int]$FrameIndex=0,
    [string]$ExpectedText='',
    [string]$Output='outputs/ocr-http-v1'
)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$destination=if ([IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $repo $Output }
if (Test-Path -LiteralPath $destination) { throw 'Use a new output directory' }
$base="http://127.0.0.1:$Port"
if (-not (Invoke-RestMethod "$base/api/health" -TimeoutSec 5).ocr_enabled) { throw 'Start service with -OcrModels first' }
New-Item -ItemType Directory -Path $destination | Out-Null
$request=@{type='ocr_frame';path=$Video;frame_index=$FrameIndex} | ConvertTo-Json
$accepted=Invoke-RestMethod "$base/api/jobs" -Method Post -ContentType application/json -Body $request -TimeoutSec 5
$deadline=[DateTime]::UtcNow.AddSeconds(60)
do {
    $job=Invoke-RestMethod "$base/api/jobs/$($accepted.id)" -TimeoutSec 5
    if ($job.state -in @('succeeded','failed','cancelled')) { break }
    if ([DateTime]::UtcNow -gt $deadline) {
        Invoke-RestMethod "$base/api/jobs/$($accepted.id)/cancel" -Method Post -ContentType application/json -Body '{}' -TimeoutSec 5 | Out-Null
        throw 'OCR HTTP deadline exceeded; cancellation requested'
    }
    Start-Sleep -Milliseconds 100
} while ($true)
$job | ConvertTo-Json -Depth 12 | Out-File -LiteralPath (Join-Path $destination 'job.json') -Encoding utf8
if ($job.state -ne 'succeeded') { throw "OCR job failed: $($job.error)" }
if ($job.result.path -ne $Video -or $job.result.frame_index -ne $FrameIndex) { throw 'OCR source/frame mismatch' }
if ($ExpectedText -and -not ($job.result.regions.text -ccontains $ExpectedText)) { throw 'Expected reference text missing' }
$prefix='data:image/jpeg;base64,'
if (-not $job.result.preview_data_url.StartsWith($prefix)) { throw 'OCR JPEG preview missing' }
$jpeg=[Convert]::FromBase64String($job.result.preview_data_url.Substring($prefix.Length))
if ($jpeg.Length -lt 4 -or $jpeg.Length -gt 524288 -or $jpeg[0] -ne 255 -or $jpeg[1] -ne 216) { throw 'Invalid JPEG payload' }
[IO.File]::WriteAllBytes((Join-Path $destination 'preview.jpg'),$jpeg)
Write-Host "PASS: OCR HTTP frame=$FrameIndex regions=$($job.result.regions.Count) texts=$($job.result.regions.text -join ', ') output=$destination"
