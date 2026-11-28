# Inkling-2-Mini presets

Choose a JSON file from `docs/inkling2-mini/`.

| GPU | Workload | Preset | GGUF | KV cache | Context budget |
| --- | --- | --- | --- | --- | ---: |
| RX 7900 XTX (`gfx1100`) | Single conversation | `ik2-mini-gfx1100.json` | IQ4_NL | FP8 | 393,216 |
| Strix Halo (`gfx1151`, experimental) | Long context on unified RAM | `strix-halo-gfx1151.json` | IQ4_NL | Paged FP4 | 524,288 |
| RTX 4070 / 12 GB CUDA card | Single conversation (tiered VRAM / RAM / disk) | `cuda-12gb-tiered.json` | IQ4_NL | FP4 global | 32,768 |
| RTX 3090, 4090 or 5090 with 24 GB+ | Long chat on one card | `cuda-24gb-fp4.json` | IQ4_NL | FP4 global | 262,144 |
| A100 or H100 with 80 GB | Long chat with FP8 KV | `cuda-80gb.json` | IQ4_NL | FP8 | 393,216 |
| DGX Spark GB10, 121 GiB | Single user | `dgx-spark-quality.json` | IQ4_NL | FP8 | 262,144 |
| DGX Spark GB10, 121 GiB | Coding with DFlash2 (1–8 streams) | `dgx-spark-dflash2.json` | UD-Q4_K_XL | FP8 | 262,144 total (32,768 per slot) |
| DGX Spark GB10, 121 GiB | 32 concurrent streams | `dgx-spark-batched-nvfp4.json` | UD-Q4_K_XL | FP8 | 131,072 total (4,096 per stream) |
| DGX Spark GB10, 121 GiB | One very long conversation | `dgx-spark-longctx-fp4.json` | IQ4_NL | FP4 global | 1,048,576 |

Context budgets are configured reservations. All four Spark presets started on GB10 and answered a short HTTP request. The 1,048,576-position preset was tested for startup and short generation. Strix Halo is a source-build target. Its 524,288-position setting has no device capacity or throughput measurement.

## Measurements

### DGX Spark GB10

These runs used CUDA 13.4.92 on GB10 with stock weights and the combined engine from revisions `9a9e1143` and `77756ac4` (Git tree `5548e452`). The DFlash2 preset loads UD-Q4_K_XL, quantizes the drafter to Q8 at load time, reserves eight slots, uses a 2,048-token chunk and keeps the default adaptive policy. The two preset columns are consecutive timed passes after warmup. Rates use the native serving protocol and exclude HTTP overhead.

| Workload | Active streams | Timing interval | Preset run 1 (tok/s) | Preset run 2 | Historical fixed width 8 |
| --- | ---: | --- | ---: | ---: | ---: |
| Sampled coding, four requests × 256 tokens | 1 | Generation time summed over requests | 47.94 | 50.86 | 55.60 |
| Sampled coding, four prompts twice × 256 tokens | 8 | First admission to final completion | 121.07 | 128.71 | 119.69 |
| Greedy mixed, 16 requests × 96 tokens | 8 | First admission to final completion | 113.84 | 112.94 | 113.63 |
| Same greedy mixed run | 8 | First wave, from all eight streams emitting to first completion | 176.75 | 169.04 | 162.73 |

The four coding inputs had 45, 45, 48 and 50 tokens. The mixed set ranged from 45 to 105 tokens. Sampled coding used temperature 0.7. The single-stream driver set seed 5845. The historical single-slot run used 4,096 positions and a 1,024-token chunk. The historical eight-slot runs used 8,192 total positions and a 256-token chunk. Both disabled adaptation. [Historical DFlash2 measurements](https://github.com/GodGMN/tintero/pull/16)

Without a drafter, the IQ4_NL quality preset measured 45.04 median decode tok/s at 512 input tokens (three uncached requests, 128 timed steps, 262,144 reserved). With UD-Q4_K_XL, the plain 32-slot preset measured 106.22–106.24 whole-run tok/s with distinct prompts and 128.99 with a shared synthetic prompt. Each greedy stream had 256 input and 128 output tokens. The historical 130.3 UD-Q4_K_XL and 119.0 IQ4_NL decode figures used an undocumented decode interval. The IQ4_NL result also used group mode 2, while the preset selects mode 1. [Plain UD-Q4_K_XL measurements](https://github.com/GodGMN/tintero/pull/5) · [IQ4_NL measurements](https://github.com/GodGMN/tintero/pull/8)

Plain native NVFP4 prefill with UD-Q4_K_XL measured 1,998.6–2,011.9 tok/s at 4,096 input tokens and 2,017.1–2,027.4 at 16,384. Each depth used three fresh processes, a synthetic input stream, a 2,048-token chunk, context set to depth plus 1,024, and one output token. The earlier 2,089 tok/s was the high end of the 4K prefill measurements. [Historical prefill measurements](https://github.com/GodGMN/tintero/pull/11)

### RX 7900 XTX and A100

The RX 7900 XTX ran stock IQ4_NL on ROCm 7.2.1 with all experts resident, one greedy stream, one cache slot, no drafter, 393,216 positions and a 1,024-token chunk. Each row gives the median of three uncached requests generating 129 tokens (128 timed decode steps). The 512-token series followed a one-token warmup. The 4K series followed the 512-token series. [Runner, token fixture and individual timings](https://github.com/GodGMN/tintero/pull/19)

| KV | KV allocation | Input tokens | Prefill (tok/s) | Decode (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| FP8 | 6.16 GiB | 512 | 2,585.86 | 115.63 |
| FP8 | 6.16 GiB | 4,096 | 3,459.46 | 111.01 |
| FP4 | 3.46 GiB | 512 | 2,767.57 | 112.87 |
| FP4 | 3.46 GiB | 4,096 | 3,480.03 | 107.47 |

The historical A100-SXM4-80GB run used CUDA 13.0, stock IQ4_NL, one stream and 393,216 FP8 positions. It reported 98.8 decode tok/s and 2,155–2,184 prefill tok/s across 4K–32K inputs. The report gives no per-depth mapping. [Original A100 measurement](https://github.com/GodGMN/tintero/commit/99c2668e)

### RTX 4070 12 GB (Windows 10, CUDA 13.1)

Measured on a desktop Windows 10 machine with an RTX 4070 12 GB (`sm_89`, ~10.8 GiB free VRAM with desktop apps open), 32 GB system RAM (~16.6 GiB available), and the 17.6 GB `IQ4_NL` GGUF stored on a 7200 RPM SATA HDD (`~100–150 MB/s`). All runs used `IK2_KV=fp4-global`, `--max-context 32768`, `--chunk 1024`, a 97-token coding prompt (plus an 80-token history prompt for the topic-shift check), and greedy decoding for 64 tokens (63 timed decode steps). Every mode produced bit-identical 64/64 greedy token IDs.

| Mode | VRAM / RAM expert budget | Prefill (tok/s) | Decode (tok/s) | Decode tier breakdown |
| --- | --- | ---: | ---: | --- |
| `IK2_GPU_SERVED=1` (profile-warmed) | 45 hot/layer + reserves / 14.7 GiB pinned (`--ram-gb 16`) | 28.5–33.5 | 25.32–29.22 | 85.2% VRAM hits, 14.8% mapped RAM |
| 3-tier auto-sized (cold) | 6.5 GiB VRAM (1,356 slots) / 11.8 GiB RAM (2,472 slots) | 4.0 | 2.62 | 88.4% VRAM, 10.0% RAM, 1.6% disk (HDD) |
| 3-tier auto-sized (warmed + surplus RAM backing) | 6.5 GiB VRAM + 11.7 GiB RAM (disjoint + surplus backups) | 62.3 | 3.94 | 89.6% VRAM, 9.4% RAM, 1.0% disk (HDD) |
| 3-tier constrained RAM (`--ram-gb 4`, warmed) | 6.5 GiB VRAM / 4.0 GiB RAM (835 slots) + Direct-I/O disk | 1.8 | 0.56 | 89.6% VRAM, 2.4% RAM, 8.1% disk (HDD) |
| 3-tier topic shift (`--ram-gb 6`, coding profile -> history prompt) | 6.5 GiB VRAM / 6.0 GiB RAM (1,253 slots) + Direct-I/O disk | 1.9 | 0.81 | 89.4% VRAM, 5.1% RAM, 5.5% disk (HDD) |

Because disk reads on that test system came from a spinning HDD, disk misses dominated wall time whenever host RAM was capped below the 14.7 GiB expert footprint. Storing the GGUF on an NVMe SSD removes that disk-tier bottleneck, and when ~15+ GiB of host RAM is available `IK2_GPU_SERVED=1` keeps all cold experts in pinned RAM and reaches 25–29 tok/s decode on the 12 GB card. Live HTTP serving via `serve/server.py` with `cuda-12gb-tiered.json` was also verified end-to-end on Windows 10.

## Run a preset

Put the matching GGUF in `models/` and export its tokenizer from the package root.

```sh
python3 tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-IQ4_NL.gguf \
  --out ik2 --chat-template tools/ik2/chat_template.jinja
```

The single-user DGX Spark preset uses adaptive MTP. Download the drafter beside the GGUF before starting this preset.

```sh
hf download thinkingmachines/Inkling-2-Mini mtp.safetensors --local-dir models
```

The default verification window has room for the current token and up to four guesses. Set `IK2_DRAFT_WIDTH` (2–8) to change the maximum window. For plain decoding, use a copy of the preset with `--draft` and its filename removed from `args`.

For CUDA containers, mount both directories under `/opt/tintero`.

```sh
docker run --rm --gpus all -p 127.0.0.1:8080:8080 \
  -v "$PWD/models:/opt/tintero/models:ro" -v "$PWD/ik2:/opt/tintero/ik2:ro" \
  -e TINTERO_CONFIG=docs/inkling2-mini/dgx-spark-quality.json \
  ghcr.io/godgmn/tintero:v0.1.0-cuda
```

The HIP image uses `v0.1.0-hip-gfx1100` and defaults to the RX 7900 XTX preset. Pass `--device=/dev/kfd --device=/dev/dri` for ROCm. Set `TINTERO_MODEL` to override the GGUF path inside the container. Explicit `IK2_*` environment settings override preset values. The server listens on `TINTERO_HOST` (default `0.0.0.0`) and `TINTERO_PORT` (default `8080`).

For a release tarball, extract it into an empty directory on Ubuntu 24.04 with the matching CUDA 13.1 or ROCm 7.2 runtime libraries. From that directory, activate a Python virtual environment, run `pip install -r requirements.txt`, export the tokenizer, and start the server.

```sh
TINTERO_CONFIG=docs/inkling2-mini/cuda-24gb-fp4.json TINTERO_HOST=127.0.0.1 ./entrypoint.sh
```

## Source-only presets

Run these examples from a checkout containing the preset, with the Python environment and build tools listed in the [Quick start](../README.md#quick-start). The published v0.1.0 images and tarballs predate both presets.

### DGX Spark DFlash2

Build the CUDA executable for GB10. Download the stock UD-Q4_K_XL target and the DFlash2 checkpoint into the paths configured by the preset.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121-real
cmake --build build --target tintero -j4
hf download unsloth/Inkling-2-Mini-GGUF Inkling-2-Mini-UD-Q4_K_XL.gguf --local-dir models
hf download incoai/Inkling-2-Mini-DFlash2 model.safetensors --local-dir models/dflash2
python3 tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-UD-Q4_K_XL.gguf --out ik2 --chat-template tools/ik2/chat_template.jinja
python3 serve/server.py --engine maya --config docs/inkling2-mini/dgx-spark-dflash2.json --host 127.0.0.1 --port 8085
```

### 12 GB CUDA cards (Linux or Windows)

`docs/inkling2-mini/cuda-12gb-tiered.json` enables `IK2_KV=fp4-global` and `IK2_GPU_SERVED=1` with a 32,768-token context. On hosts with enough free RAM for the 14.7 GiB expert footprint (e.g. 32 GB RAM), cold experts stay pinned in host memory and hot VRAM slots warm from the saved expert profile. On tighter-RAM hosts (or when `--ram-gb N` / `IK2_RAM_GB=N` is set), the engine automatically falls back to the 3-tier VRAM / pinned RAM / Direct-I/O disk store.

```powershell
# Windows (Developer PowerShell for VS)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target tintero -j4
python serve\server.py --engine maya --config docs\inkling2-mini\cuda-12gb-tiered.json --host 127.0.0.1 --port 8085
```

### Strix Halo (experimental)

Build for gfx1151 on Linux with ROCm and use the IQ4_NL model.

```sh
cmake -S . -B build-ik2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DGPU_TARGETS=gfx1151
cmake --build build-ik2 --target tintero -j4
hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-IQ4_NL.gguf --local-dir models
python3 tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-IQ4_NL.gguf --out ik2 --chat-template tools/ik2/chat_template.jinja
python3 serve/server.py --engine maya --config docs/inkling2-mini/strix-halo-gfx1151.json --host 127.0.0.1 --port 8085
```
