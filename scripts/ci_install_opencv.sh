#!/usr/bin/env bash
# Only called on disposable Ubuntu GitHub runners, never on the user's Windows host.
set -euo pipefail
# The Azure mirror delivered the previous run at ~76 kB/s (>50 minutes).
# Keep Ubuntu's signed repositories/keys; change only its package mirror.
for source_file in /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list; do
    if [[ -f "$source_file" ]]; then
        sudo sed -i 's#http[s]*://azure\.archive\.ubuntu\.com/ubuntu#https://archive.ubuntu.com/ubuntu#g' "$source_file"
    fi
done
sudo apt-get update -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30
sudo apt-get install -y --no-install-recommends libopencv-dev ffmpeg "$@"
