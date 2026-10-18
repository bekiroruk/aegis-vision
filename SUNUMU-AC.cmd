@echo off
if not exist "%~dp0outputs\sunum-paketi-20261019\index.html" (
  echo Sunum paketi bu bilgisayarda bulunamadi.
  echo Yerel teslim: outputs\sunum-paketi-20261019
  echo Sunum ve veri dosyalari Git deposuna dahil degildir.
  pause
  exit /b 1
)
start "" "%~dp0outputs\sunum-paketi-20261019\index.html"
