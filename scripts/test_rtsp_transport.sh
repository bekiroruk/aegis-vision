#!/usr/bin/env bash
# CI transport test: generated moving test pattern, no downloaded model, no accuracy claim.
set -euo pipefail
build_dir=${1:-build}
relay_exe=${2:?Provide the verified MediaMTX binary}
scratch=$(mktemp -d)
relay_pid= publisher_pid= probe_pid=
cleanup() {
    for child in "$probe_pid" "$publisher_pid" "$relay_pid"; do
        if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then
            kill "$child" 2>/dev/null || true
            wait "$child" 2>/dev/null || true
        fi
    done
    # Preserve logs for diagnosis; do not remove an unresolved directory.
    echo "RTSP transport logs: $scratch"
}
trap cleanup EXIT
"$relay_exe" configs/rtsp-local.yml >"$scratch/relay.log" 2>&1 & relay_pid=$!
sleep 1
publish() {
    ffmpeg -hide_banner -loglevel warning -re -f lavfi -i testsrc2=size=320x240:rate=10 \
        -an -c:v libx264 -preset ultrafast -tune zerolatency -g 10 -pix_fmt yuv420p \
        -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/pedestrians >"$scratch/publisher-$1.log" 2>&1 &
    publisher_pid=$!
}
publish before
sleep 1
"$build_dir/aegisvision_live_tests" --rtsp rtsp://127.0.0.1:8554/pedestrians >"$scratch/probe.log" 2>&1 & probe_pid=$!
ready=false
for ((attempt=0; attempt<60; ++attempt)); do
    if grep -q RTSP_SESSION=1 "$scratch/probe.log"; then ready=true; break; fi
    kill -0 "$probe_pid" || { cat "$scratch/probe.log"; exit 1; }
    sleep 0.1
done
if [[ "$ready" != true ]]; then cat "$scratch/probe.log"; exit 1; fi
sleep 3
kill "$publisher_pid"; wait "$publisher_pid" || true; publisher_pid=
sleep 3
publish after
wait "$probe_pid" || { cat "$scratch/probe.log"; exit 1; }; probe_pid=
cat "$scratch/probe.log"
