FROM rocm/dev-ubuntu-24.04:7.2-complete AS build
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY src/ src/
COPY include/ include/
COPY third_party/ third_party/
RUN cmake -S . -B /build -G Ninja -DSTRATA_BUILD_TESTS=OFF \
      -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx1100;gfx1151" -DGPU_TARGETS="gfx1100;gfx1151" \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm && \
    cmake --build /build --target tintero --parallel 2

FROM scratch AS bundle
COPY --from=build /build/tintero /build-ik2/tintero
COPY VERSION LICENSE* maya.py maya.sh setup.sh /
COPY third_party/ggml/LICENSE /LICENSE.ggml
COPY third_party/ciru/LICENSE /LICENSE.ciru
COPY third_party/ciru/NOTICE /NOTICE.ciru
COPY serve/ /serve/
COPY tools/ /tools/
COPY ref/ /ref/
COPY tests/ /tests/
COPY data/ /data/
COPY docs/inkling2-mini/ /docs/inkling2-mini/
COPY docs/PRESETS.md /docs/
COPY docker/requirements.txt docker/entrypoint.sh /

FROM rocm/dev-ubuntu-24.04:7.2
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv libgomp1 hipblas && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/tintero
COPY --from=bundle / ./
ENV PATH="/opt/venv/bin:$PATH" \
    TINTERO_CONFIG=docs/inkling2-mini/ik2-mini-gfx1100.json
RUN python3 -m venv /opt/venv && pip install --no-cache-dir -r requirements.txt && python3 serve/server.py --help >/dev/null
EXPOSE 8080
ENTRYPOINT ["/opt/tintero/entrypoint.sh"]
