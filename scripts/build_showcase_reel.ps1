param(
    [Parameter(Mandatory=$true)][string]$PeopleClip,
    [Parameter(Mandatory=$true)][string]$TrafficClip,
    [Parameter(Mandatory=$true)][string]$SecurityClip,
    [string]$Output = 'outputs/aegisvision-showcase.mp4'
)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$ffmpeg=(Get-Command ffmpeg -ErrorAction Stop).Source
$ffprobe=(Get-Command ffprobe -ErrorAction Stop).Source
$font=(Join-Path $env:WINDIR 'Fonts/arial.ttf').Replace('\','/').Replace(':','\:')
$inputs=@($PeopleClip,$TrafficClip,$SecurityClip) | ForEach-Object {(Resolve-Path -LiteralPath (Join-Path $repo $_)).Path}
$target=[IO.Path]::GetFullPath((Join-Path $repo $Output))
New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($target)) -Force | Out-Null
if(Test-Path -LiteralPath $target){throw "Output already exists: $target"}
$filter="color=c=0x0c121b:s=1280x720:r=10:d=3,drawtext=fontfile='$font':text='AEGISVISION':x=70:y=230:fontsize=78:fontcolor=0xe9f0f8,drawtext=fontfile='$font':text='C++ VIDEO INTELLIGENCE':x=74:y=330:fontsize=32:fontcolor=0x69e0c0,drawtext=fontfile='$font':text='Segmentation  /  Tracking  /  Search':x=76:y=395:fontsize=22:fontcolor=0x9eafc5[intro];"+
"[0:v]trim=duration=7.2,setpts=PTS-STARTPTS,scale=1280:450,pad=1280:720:0:86:color=0x0c121b,drawtext=fontfile='$font':text='01  PEDESTRIAN STREET':x=54:y=30:fontsize=25:fontcolor=0x69e0c0,drawtext=fontfile='$font':text='72 frames  /  447 mask instances  /  26 video-local track IDs':x=54:y=660:fontsize=19:fontcolor=0xe9f0f8[p];"+
"[1:v]trim=duration=8,setpts=PTS-STARTPTS,scale=1280:450,pad=1280:720:0:86:color=0x0c121b,drawtext=fontfile='$font':text='02  MOVING VEHICLE CAMERA':x=54:y=30:fontsize=25:fontcolor=0x69e0c0,drawtext=fontfile='$font':text='120 frames  /  832 mask instances  /  36 video-local track IDs':x=54:y=660:fontsize=19:fontcolor=0xe9f0f8[t];"+
"[2:v]trim=duration=8,setpts=PTS-STARTPTS,scale=1280:450,pad=1280:720:0:86:color=0x0c121b,drawtext=fontfile='$font':text='03  SECURITY CAMERA':x=54:y=30:fontsize=25:fontcolor=0x69e0c0,drawtext=fontfile='$font':text='120 frames  /  1090 mask instances  /  20 video-local track IDs':x=54:y=660:fontsize=19:fontcolor=0xe9f0f8[s];"+
"[intro][p][t][s]concat=n=4:v=1:a=0[out]"
& $ffmpeg -hide_banner -loglevel error -nostdin -i $inputs[0] -i $inputs[1] -i $inputs[2] -filter_complex $filter -map '[out]' -an -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p -movflags +faststart $target
if($LASTEXITCODE-ne 0){throw 'Showcase video export failed.'}
$probe=(& $ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=codec_name,width,height,nb_read_frames -show_entries format=duration -of json $target)|ConvertFrom-Json
if($LASTEXITCODE-ne 0 -or $probe.streams[0].codec_name-ne'h264' -or $probe.streams[0].width-ne1280 -or $probe.streams[0].height-ne720){throw 'Showcase verification failed.'}
[pscustomobject]@{Path=$target;DurationSeconds=[math]::Round([double]$probe.format.duration,1);Frames=[int]$probe.streams[0].nb_read_frames;Sha256=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()}
