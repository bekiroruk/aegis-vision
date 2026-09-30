param(
    [string]$Cli = './build/search/Release/aegisvision_search_cli.exe',
    [string]$Bundle = './artifacts/models/clip-vit-b32',
    [string]$DetectorConfig = './configs/image.toml',
    [string]$Images = './artifacts/samples',
    [string]$Video = './artifacts/samples/moving-bus.avi',
    [string]$Collection = 'aegis_indexing_demo',
    [string]$Output = './outputs/indexing-demo',
    [int]$Port = 6333
)
$ErrorActionPreference = 'Stop'
if ($Collection -notmatch '^[A-Za-z0-9_-]{1,64}$' -or $Port -lt 1 -or $Port -gt 65535) {
    throw 'Invalid collection or port'
}
$baseUri = "http://127.0.0.1:$Port"
$collections = Invoke-RestMethod "$baseUri/collections"
if ($Collection -in $collections.result.collections.name) {
    throw 'Smoke test requires a new collection; choose a different -Collection (existing data is retained).'
}
if ((Test-Path -LiteralPath $Output) -and
    (!(Test-Path -LiteralPath $Output -PathType Container) -or (Get-ChildItem -LiteralPath $Output -Force))) {
    throw 'Smoke test output directory must be new or empty'
}
function Search-Command([string[]]$CommandArguments) {
    $searchArgs = @($Bundle, "$Port", $Collection) + $CommandArguments
    $raw = & $Cli @searchArgs
    if ($LASTEXITCODE -ne 0) { throw "Search CLI failed: $CommandArguments" }
    return (($raw -join "`n") | ConvertFrom-Json)
}
function Point-Count {
    return (Invoke-RestMethod "$baseUri/collections/$Collection").result.points_count
}

$null = New-Item -ItemType Directory -Force -Path $Output
$initialized = Search-Command @('init')
$directory = Search-Command @('index-directory', $Images)
$directoryCount = Point-Count
$directoryRetry = Search-Command @('index-directory', $Images)
if ((Point-Count) -ne $directoryCount -or $directoryCount -ne $directory.indexed_items) {
    throw 'Directory retry did not preserve the point count'
}
$videoResult = Search-Command @('index-video', $DetectorConfig, $Video, '10', '30')
$combinedCount = Point-Count
$videoRetry = Search-Command @('index-video', $DetectorConfig, $Video, '10', '30')
if ((Point-Count) -ne $combinedCount -or
    $combinedCount -ne ($directoryCount + $videoResult.indexed_items)) {
    throw 'Video retry did not preserve the point count'
}
$query = Search-Command @('text', 'a photo of a bus', '10')
if (!$query.results -or !($query.results | Where-Object { $_.metadata.kind -eq 'video_crop' })) {
    throw 'Query did not return an indexed video crop'
}
$report = [ordered]@{
    collection = $Collection
    space_id = $initialized.space_id
    directory = $directory
    directory_retry = $directoryRetry
    video = $videoResult
    video_retry = $videoRetry
    point_count = $combinedCount
    retries_preserved_point_count = $true
    query = $query
}
$reportPath = Join-Path $Output 'report.json'
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $reportPath -Encoding utf8
Write-Output "Indexing smoke passed: $combinedCount records, retry counts unchanged. Report: $reportPath"
