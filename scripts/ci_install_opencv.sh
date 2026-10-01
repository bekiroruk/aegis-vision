#!/usr/bin/env bash
# Only called on disposable Ubuntu GitHub runners, never on the user's Windows host.
set -euo pipefail
# The Azure mirror delivered the previous run at ~76 kB/s (>50 minutes).
# Keep Ubuntu's signed repositories/keys; change only its package mirror.
for source_file in /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list; do
    if [[ -f "$source_file" ]]; then
        # Current runner images use mirror+file:, older images used a direct URI.
        # Ref: actions/runner-images/images/ubuntu/scripts/build/configure-apt-sources.sh
        sudo sed -i \
            -e 's#mirror+file:/etc/apt/apt-mirrors\.txt#https://archive.ubuntu.com/ubuntu/#g' \
            -e 's#http[s]*://azure\.archive\.ubuntu\.com/ubuntu#https://archive.ubuntu.com/ubuntu#g' "$source_file"
    fi
done
sudo apt-get update -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30
sudo apt-get install -y --no-install-recommends \
    -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 \
    libopencv-dev ffmpeg "$@"
