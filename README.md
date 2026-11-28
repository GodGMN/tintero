<h1><picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/tintero-wordmark-dark.png">
  <img src="assets/tintero-wordmark.png" alt="Tintero" width="300">
</picture></h1>

Tintero is a local Linux inference server for [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) supporting **AMD (HIP)** and **NVIDIA (CUDA)** GPUs. It is tailored to keep the full quantized model resident on a single **24 GB+ VRAM** card, with FP8/FP4 KV context scaling to whatever memory you have available and optional zero-retraining runtime abliteration (`IK2_ABLATE_MODE=on`).

### Tested hardware

| GPU | Workload | Throughput (tok/s) |
| --- | --- | ---: |
| RX 7900 XTX, gfx1100 | IQ4_NL, FP8, one stream, 4,096 input tokens, median of three | 3,459.46 prefill / 111.01 decode |
| DGX Spark GB10, sm_121a | IQ4_NL quality preset, one stream, 512 input tokens | 45.04 median decode |
| DGX Spark GB10, sm_121a | UD-Q4_K_XL + Q8 DFlash2 preset, eight active sampled coding streams | 121.07 and 128.71 whole-run aggregate |
| A100-SXM4-80GB, sm_80 (historical) | IQ4_NL, FP8, one stream | 98.8 decode / 2,155–2,184 prefill across 4K–32K inputs |

See [preset measurements](docs/PRESETS.md#measurements) for context reservations, workload sizes, timing intervals and historical comparisons. The [source-build examples](docs/PRESETS.md#source-only-presets) cover DFlash2 on Spark and the experimental Strix Halo target.

Both backends support continuous batching, FP8/FP4-global KV cache, sparse long-context decode, graphed MTP speculative decoding, and runtime abliteration. CUDA covers `sm_80` through `sm_121a` (Ampere through Blackwell, x86_64 and ARM64). See [presets](docs/PRESETS.md) and [settings](docs/SETTINGS.md).

## Quick start

> [!TIP]
> Using a coding agent? Send it this README and ask: “Set up Tintero on this machine and verify it with the test request below.”

You need Linux, a GPU with **24 GB+ VRAM**, and Docker (with the NVIDIA Container Toolkit for CUDA, or ROCm drivers for AMD). The `IQ4_NL` model download is about 18 GB.

**1. Download the model and export the tokenizer.**

```sh
mkdir -p models ik2
hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-IQ4_NL.gguf --local-dir models
docker run --rm --entrypoint python3 \
  -v "$PWD/models:/opt/tintero/models:ro" -v "$PWD/ik2:/opt/tintero/ik2" \
  ghcr.io/godgmn/tintero:v0.1.0-cuda \
  tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-IQ4_NL.gguf \
  --out ik2 --chat-template tools/ik2/chat_template.jinja
```

*(On AMD, use `ghcr.io/godgmn/tintero:v0.1.0-hip-gfx1100` in the tokenizer command above.)*

**2. Start the server.**

```sh
# NVIDIA (RTX 30/40/50, A100, H100, DGX Spark — x86_64 & ARM64)
docker run -d --name tintero --gpus all -p 127.0.0.1:8085:8080 \
  -v "$PWD/models:/opt/tintero/models:ro" -v "$PWD/ik2:/opt/tintero/ik2:ro" \
  ghcr.io/godgmn/tintero:v0.1.0-cuda

# AMD RX 7900 XTX (RDNA3 gfx1100)
docker run -d --name tintero --device=/dev/kfd --device=/dev/dri -p 127.0.0.1:8085:8080 \
  -v "$PWD/models:/opt/tintero/models:ro" -v "$PWD/ik2:/opt/tintero/ik2:ro" \
  ghcr.io/godgmn/tintero:v0.1.0-hip-gfx1100
```

Each image starts with a 24 GB preset by default. For an 80 GB card or a DGX Spark (single-stream, 32-stream NVFP4 batching, or 1M FP4-global context), pass `-e TINTERO_CONFIG=docs/inkling2-mini/<preset>.json` from [docs/PRESETS.md](docs/PRESETS.md). Pass `-e TINTERO_MODEL=models/<file>.gguf` to switch GGUF files, and `-e IK2_ABLATE_MODE=on` for runtime abliteration.

<details>
<summary>Without Docker: prebuilt release tarball or build from source</summary>

**Prebuilt Ubuntu 24.04 tarballs** (`x86_64-cuda`, `aarch64-cuda`, `x86_64-hip-gfx1100`) are attached to each [GitHub Release](https://github.com/GodGMN/tintero/releases). Extract into an empty directory on a host with matching CUDA 13.1 or ROCm 7.2 runtime libraries, install `requirements.txt` in a Python `venv`, export the tokenizer, and run:

```sh
TINTERO_CONFIG=docs/inkling2-mini/cuda-24gb-fp4.json TINTERO_HOST=127.0.0.1 TINTERO_PORT=8085 ./entrypoint.sh
```

**Build from source** (requires a C++20 compiler, CMake 3.24+, Ninja, Git, Python 3.10+ with `venv`, and either ROCm under `/opt/rocm` or the NVIDIA CUDA Toolkit):

```sh
git clone https://github.com/GodGMN/tintero
cd tintero
```

AMD HIP (`gfx1100`, tested with ROCm 7.2.1):

```sh
cmake -S . -B build-ik2 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGPU_TARGETS=gfx1100
cmake --build build-ik2 --target tintero -j4
```

NVIDIA CUDA (`sm_80+` — RTX 30/40/50 series, A100, H100, DGX Spark):

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target tintero ik2_kernels_test -j4
./build/ik2_kernels_test
```

Install Python dependencies, download the model, export the tokenizer, and start a preset from [docs/PRESETS.md](docs/PRESETS.md):

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r docker/requirements.txt huggingface_hub
hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-IQ4_NL.gguf --local-dir models
python tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-IQ4_NL.gguf \
  --out ik2 --chat-template tools/ik2/chat_template.jinja
python serve/server.py --engine maya --config docs/inkling2-mini/cuda-24gb-fp4.json \
  --host 127.0.0.1 --port 8085
```

</details>

**3. Send a message.** Once the model has loaded, run this in another terminal:

```sh
curl -fsS http://127.0.0.1:8085/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"What is 17 + 25? Answer with the number only."}],"reasoning_effort":"none","temperature":0,"max_tokens":48}'
```

The answer is in `choices[0].message.content`. To connect a client, set its base URL to `http://127.0.0.1:8085/v1` and its model to `inkling-2-mini`.

## Configuration and docs

- [Hardware presets](docs/PRESETS.md) list ready-to-run configs for 24 GB CUDA cards, 80 GB GPUs, DGX Spark GB10, and RX 7900 XTX.
- [Speed benchmark](docs/BENCHMARK.md) explains how to measure prefill and decode throughput and submit a speed report.
- [Runtime settings](docs/SETTINGS.md) cover context size, KV precision (`fp8` / `fp4-global`), concurrent batching, speculative decoding, and runtime abliteration (`IK2_ABLATE*`).
- [Vision](docs/HIP-PORT.md#image-encoder-on-the-ryzen-5-3600) requires the image encoder and `mmproj-F16.gguf`.
- [HIP port & benchmarks](docs/HIP-PORT.md) document the 393K FP8/FP4 context measurements and kernel parity tests.

## Credits

Tintero builds on [Maya’s Inkling engine](https://github.com/alesha-pro/maya-inkling-mini). [Project Maya](https://github.com/mw00/project-maya) grew out of [Strata](https://github.com/Niko1221/Strata). Selected RDNA 3/3.5 Wave32 HIP kernels are adapted from [CIRU](third_party/ciru/NOTICE) (Apache-2.0). Thinking Machines released the model, [incoai](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) released the DFlash2 drafter, and [ggml-org](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) and [Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) provide the GGUF weights. [MIT license](LICENSE)
