$ErrorActionPreference = 'Stop'
$temporaryParent = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\','/')
$testRoot = Join-Path $temporaryParent ('aegis-preflight-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path (Join-Path $testRoot 'scripts') | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'demo.ps1') -Destination (Join-Path $testRoot 'scripts/demo.ps1')
function Assert($condition, $message) { if (-not $condition) { throw $message } }
Push-Location $temporaryParent
try {
    $launcher = Join-Path $testRoot 'scripts/demo.ps1'
    $empty = & $launcher -CheckOnly
    Assert (-not $empty.FilesPresent) 'Empty checkout reported complete'
    Assert ($empty.MissingFiles.Count -eq 17) 'Missing prerequisite inventory changed; review this test'
    Assert (-not (Test-Path (Join-Path $testRoot 'artifacts'))) 'Preflight mutated empty checkout'
    foreach ($relative in $empty.MissingFiles) {
        $target = Join-Path $testRoot $relative
        New-Item -ItemType Directory -Path (Split-Path $target -Parent) -Force | Out-Null
        [IO.File]::WriteAllText($target, 'not a real model or binary')
    }
    $complete = & $launcher -CheckOnly
    Assert $complete.FilesPresent 'Relative paths are not anchored to project'
    Assert ($complete.MissingFiles.Count -eq 0) 'Unexpected missing file'
    Assert ($complete.Note -match 'File presence only') 'Presence check must not claim runtime readiness'
    $missingPath = Join-Path $testRoot 'web/index.html'
    Remove-Item -LiteralPath $missingPath
    New-Item -ItemType Directory -Path $missingPath | Out-Null
    $directory = & $launcher -CheckOnly
    Assert (-not $directory.FilesPresent -and $directory.MissingFiles.Count -eq 1) 'Directory accepted as a file'
    foreach ($option in @('Stop','OpenBrowser')) {
        $rejected = $false
        $flags = @{$option=$true}
        try { & $launcher -CheckOnly @flags | Out-Null } catch { $rejected = $true }
        Assert $rejected 'Mutating option combined with CheckOnly'
    }
    Assert (-not (Test-Path (Join-Path $testRoot 'artifacts/service'))) 'Preflight created service state'
    Write-Host 'Demo preflight: empty/complete/missing/directory/foreign-cwd/no-side-effect checks passed.'
} finally {
    Pop-Location
    $resolved = [IO.Path]::GetFullPath($testRoot)
    if ((Split-Path $resolved -Parent) -ne $temporaryParent -or
        (Split-Path $resolved -Leaf) -notmatch '^aegis-preflight-[0-9a-f]{32}$') {
        throw 'Refusing cleanup outside the exact temporary test directory'
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
