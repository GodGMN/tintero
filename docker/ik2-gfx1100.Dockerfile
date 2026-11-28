FROM ghcr.io/ggml-org/llama.cpp:server-rocm-b11176
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    python3-jinja2 python3-jsonschema python3-numpy python3-pil python3-regex python3-requests python3-yaml \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/maya
COPY LICENSE LICENSE.ggml LICENSE.ciru NOTICE.ciru VERSION ./
COPY tintero build-ik2/tintero
COPY serve/ serve/
COPY tools/ tools/
COPY tests/ tests/
COPY ref/ ref/
COPY tokenizer/ ik2/tokenizer/
COPY vision/ vision/
COPY ablate/ ablate/
COPY ik2-mini-gfx1100*.json ./
COPY start-ik2-gfx1100.sh ./
ENTRYPOINT ["/bin/sh", "start-ik2-gfx1100.sh"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
