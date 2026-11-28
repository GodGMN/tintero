# Inkling-2-Mini presets

Choose a JSON file from `docs/inkling2-mini/`.

| GPU | Workload | Preset | GGUF | KV cache | Context budget |
| --- | --- | --- | --- | --- | ---: |
| RX 7900 XTX (`gfx1100`) | Single conversation | `ik2-mini-gfx1100.json` | IQ4_NL | FP8 | 393,216 |
| RX 7900 XTX (`gfx1100`) | Coding in one conversation, MTP drafter | `ik2-mini-gfx1100-mtp.json` | IQ4_NL | FP8 | 262,144 |
| Strix Halo (`gfx1151`, experimental) | Long context on unified RAM | `strix-halo-gfx1151.json` | IQ4_NL | Paged FP4 | 524,288 |
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

### RX 7900 XTX with the MTP drafter

This sweep ran stock IQ4_NL and the official `mtp.safetensors` on ROCm 7.2.1 with all experts resident, one stream, FP8 KV, 32,768 positions and a 1,024-token chunk. Each configuration loaded on an idle card three times, in a different shuffled order each round. Every load answered the same five coding prompts at temperature 0 with up to 2,048 output tokens. Three prompts were saved turns from a coding agent with 1,268, 11,971 and 17,314 input tokens. The other two asked for new code and for an edit. Rates are generated tokens over the engine's decode time, so prefill is excluded. The table gives the median of the three rounds. Fixed windows varied by under 1% between rounds and adaptive ones by up to 5.4%.

| Maximum window | Decode (tok/s) | Versus plain | Drafts accepted | Tokens per window |
| --- | ---: | ---: | ---: | ---: |
| Plain (no drafter) | 104.9 | — | — | — |
| 2 | 107.4 | +2% | 98% | 1.98 |
| 3 | 138.1 | +32% | 94% | 2.88 |
| 4 | 164.3 | +57% | 92% | 3.75 |
| 5 | 181.3 | +73% | 88% | 4.49 |
| 6 | 195.7 | +87% | 86% | 5.11 |
| 7 | 199.2 | +90% | 81% | 5.50 |
| 8 | 213.4 | +103% | 81% | 6.32 |

These rows use the adaptive controller. With `IK2_DRAFT_ADAPT=0`, every width was within 3% of its adaptive result. At width 8 the five prompts decoded at 182–248 tok/s, against 98–116 without the drafter. Most drafted answers differed from the plain greedy answers, so this sweep measures speed, not answer quality. Prose was not part of the sweep. An earlier single 257-token story slowed from 116 to 89 tok/s with the drafter, which is why the general preset stays plain.

`ik2-mini-gfx1100-mtp.json` applies width 8 at 262,144 positions with a 65,536-position drafter cache, using 23.8 GB of VRAM after loading. At that reservation it decoded the new-code, 17K-token agent and edit prompts at 205, 207 and 227 tok/s. The drafter does not fit beside the 393,216-position reservation of the general preset.

## Run a preset

Put the matching GGUF in `models/` and export its tokenizer from the package root.

```sh
python3 tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-IQ4_NL.gguf \
  --out ik2 --chat-template tools/ik2/chat_template.jinja
```

The single-user DGX Spark preset and the RX 7900 XTX coding preset use adaptive MTP. Download the drafter beside the GGUF before starting either preset.

```sh
hf download thinkingmachines/Inkling-2-Mini mtp.safetensors --local-dir models
```

The default verification window has room for the current token and up to four guesses. The RX 7900 XTX coding preset raises it to eight positions with `IK2_DRAFT_WIDTH=8`. Set `IK2_DRAFT_WIDTH` (2–8) to change the maximum window. For plain decoding, use a copy of the preset with `--draft` and its filename removed from `args`.

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

### Strix Halo (experimental)

Build for gfx1151 on Linux with ROCm and use the IQ4_NL model.

```sh
cmake -S . -B build-ik2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DGPU_TARGETS=gfx1151
cmake --build build-ik2 --target tintero -j4
hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-IQ4_NL.gguf --local-dir models
python3 tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-IQ4_NL.gguf --out ik2 --chat-template tools/ik2/chat_template.jinja
python3 serve/server.py --engine maya --config docs/inkling2-mini/strix-halo-gfx1151.json --host 127.0.0.1 --port 8085
```
