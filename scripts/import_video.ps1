param(
    [string]$VideoPath,
    [ValidateRange(0,86400)][double]$StartSeconds = 0,
    [ValidateRange(1,60)][int]$Seconds = 15,
    [switch]$NoOpen
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if (-not $VideoPath) {
    Add-Type -AssemblyName System.Windows.Forms
    $picker = New-Object System.Windows.Forms.OpenFileDialog
    $picker.Title = 'AegisVision - analiz edilecek videoyu secin'
    $picker.Filter = 'Video|*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v|Tum dosyalar|*.*'
    try {
        if ($picker.ShowDialog() -ne 'OK') { return }
        $VideoPath = $picker.FileName
    } finally { $picker.Dispose() }
}
$source = Get-Item -LiteralPath $VideoPath
if ($source.PSIsContainer) { throw 'Bir video dosyasi secin.' }
$ffmpeg = (Get-Command ffmpeg -ErrorAction Stop).Source
$ffprobe = (Get-Command ffprobe -ErrorAction Stop).Source
$mediaRoot = Join-Path $repo 'artifacts/media'
New-Item -ItemType Directory -Path $mediaRoot -Force | Out-Null
# Publish only after validation; original video is never changed. ASCII output
# names also work with the native C++ service on Windows.
$label = [regex]::Replace($source.BaseName, '[^a-zA-Z0-9_-]', '-').Trim('-')
if (-not $label) { $label = 'video' }
if ($label.Length -gt 48) { $label = $label.Substring(0,48) }
$name = 'demo-' + $label + '-' + [guid]::NewGuid().ToString('N').Substring(0,8) + '.mp4'
$target = Join-Path $mediaRoot $name
$temporary = Join-Path ([IO.Path]::GetTempPath()) ('aegis-' + [guid]::NewGuid().ToString('N') + '.mp4')
try {
    Write-Host "Video hazirlaniyor: $Seconds saniyeye kadar, 10 FPS, en fazla 960x540. Orijinal korunur."
    $start = $StartSeconds.ToString([Globalization.CultureInfo]::InvariantCulture)
    & $ffmpeg -hide_banner -loglevel error -nostdin -n -ss $start -i $source.FullName -t $Seconds -map 0:v:0 -an -sn -dn -vf "scale=960:540:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1,fps=10" -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p -movflags +faststart $temporary
    if ($LASTEXITCODE -ne 0) { throw 'Video donusturulemedi. Kaynak dosyasini kontrol edin.' }
    $raw = & $ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=codec_name,width,height,nb_read_frames -of json $temporary
    if ($LASTEXITCODE -ne 0) { throw 'Hazirlanan video okunamadi.' }
    $stream = ($raw | ConvertFrom-Json).streams | Select-Object -First 1
    if (-not $stream -or $stream.codec_name -ne 'h264' -or [int]$stream.nb_read_frames -lt 1 -or [int]$stream.nb_read_frames -gt 600 -or $stream.width -gt 960 -or $stream.height -gt 540 -or ($stream.width % 2) -or ($stream.height % 2)) {
        throw 'Video demo sinirlarini karsilamiyor veya secilen baslangicta kare yok.'
    }
    Move-Item -LiteralPath $temporary -Destination $target
    Write-Host "Hazir: $name ($($stream.nb_read_frames) kare). Panelde Videoyu analiz et dugmesine basin."
    if (-not $NoOpen) {
        & (Join-Path $PSScriptRoot 'demo.ps1')
        Start-Process ('http://127.0.0.1:8090/?video=' + [Uri]::EscapeDataString($name))
    }
    [pscustomobject]@{ Path=$target; MediaPath=$name; Frames=[int]$stream.nb_read_frames }
} finally {
    if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
}
