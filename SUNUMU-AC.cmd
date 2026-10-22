@echo off
if not exist "%~dp0outputs\AegisVision-vitrin-paketi\index.html" (
  echo Sunum paketi bu bilgisayarda bulunamadi.
  echo Yerel teslim: outputs\AegisVision-vitrin-paketi
  echo Sunum ve veri dosyalari Git deposuna dahil degildir.
  pause
  exit /b 1
)
start "" "%~dp0outputs\AegisVision-vitrin-paketi\index.html"
