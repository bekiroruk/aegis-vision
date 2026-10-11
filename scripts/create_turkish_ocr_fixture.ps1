param([string]$Output='outputs/ocr-turkish-fixture')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$destination=if([IO.Path]::IsPathRooted($Output)){$Output}else{Join-Path $repo $Output}
if(Test-Path -LiteralPath $destination){throw 'Output must be a new directory'}
Add-Type -AssemblyName System.Drawing
New-Item -ItemType Directory -Path $destination | Out-Null
$bitmap=[Drawing.Bitmap]::new(1000,300)
$graphics=[Drawing.Graphics]::FromImage($bitmap)
$font=[Drawing.Font]::new('Arial',40,[Drawing.FontStyle]::Regular,[Drawing.GraphicsUnit]::Pixel)
try {
    $graphics.Clear([Drawing.Color]::White)
    $graphics.TextRenderingHint=[Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    # ASCII source also works in Windows PowerShell 5.1 without a UTF-8 BOM.
    $lines=ConvertFrom-Json '["T\u00fcrk\u00e7e \u0130stanbul","\u00c7\u0131\u011f \u015e\u00d6LEN \u0131\u0130 \u00f6\u00fc 123"]'
    for($index=0;$index -lt $lines.Count;$index++) {
        $graphics.DrawString($lines[$index],$font,[Drawing.Brushes]::Black,[Drawing.PointF]::new(45,40+100*$index))
    }
    $bitmap.Save((Join-Path $destination 'turkish.png'),[Drawing.Imaging.ImageFormat]::Png)
    @{synthetic=$true;description='Generated typography smoke, not a real-world accuracy benchmark';font=$font.Name;size_pixels=40;lines=$lines} |
        ConvertTo-Json | Out-File -LiteralPath (Join-Path $destination 'reference.json') -Encoding utf8
} finally {$font.Dispose();$graphics.Dispose();$bitmap.Dispose()}
Write-Host "Synthetic Turkish OCR fixture: $destination"
