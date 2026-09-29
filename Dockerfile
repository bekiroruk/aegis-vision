FROM gcc:14 AS build

RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY include ./include
COPY src ./src
COPY apps ./apps
COPY tests ./tests

RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build \
    && ctest --test-dir build --output-on-failure

FROM debian:bookworm-slim AS runtime
COPY --from=build /src/build/aegisvision_demo /usr/local/bin/aegisvision_demo
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/aegisvision_demo"]

