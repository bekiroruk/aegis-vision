#!/usr/bin/env bash
# Disposable Ubuntu CI only. Does not install over the system OpenCV.
# Options: https://docs.opencv.org/4.12.0/db/d05/tutorial_config_reference.html
set -euo pipefail
task_root="$PWD/work/ci-opencv"
task_prefix="$task_root/install"
if [[ -f "$task_prefix/lib/cmake/opencv4/OpenCVConfig.cmake" ]]; then
    echo "Using cached private OpenCV 4.12.0 image runtime"
    exit 0
fi
mkdir -p "$task_root"
curl --fail --location --retry 3 \
    https://codeload.github.com/opencv/opencv/tar.gz/refs/tags/4.12.0 \
    -o "$task_root/source.tgz"
echo "44c106d5bb47efec04e531fd93008b3fcd1d27138985c5baf4eafac0e1ec9e9d  $task_root/source.tgz" | sha256sum --check
tar -xzf "$task_root/source.tgz" -C "$task_root"
cmake -S "$task_root/opencv-4.12.0" -B "$task_root/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$task_prefix" \
    -DBUILD_LIST=core,imgproc,imgcodecs,features2d,calib3d,dnn,videoio \
    -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
    -DBUILD_JAVA=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF \
    -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_OPENCL=OFF -DWITH_LAPACK=OFF \
    -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_V4L=OFF \
    -DWITH_OPENEXR=OFF -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_JASPER=OFF \
    -DWITH_OPENJPEG=OFF -DWITH_AVIF=OFF -DCPU_DISPATCH= \
    -DBUILD_PROTOBUF=ON -DPROTOBUF_UPDATE_FILES=OFF
cmake --build "$task_root/build" --parallel 4
cmake --install "$task_root/build"
