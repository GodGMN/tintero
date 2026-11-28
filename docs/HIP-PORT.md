# Inkling 2 Mini on gfx1100

`tintero` runs Inkling-2-Mini on AMD RDNA 3 and RDNA 3.5 GPUs through HIP. The 24 GiB profile keeps the quantized text weights and every routed expert resident, then scales the KV cache from FP8 down to FP4 to fit long conversations.

For benchmark results and the commands used to collect them, see the [measurement history](#measurement-history).

## Which GPUs and models can run it?

The HIP executable is built for Wave32 AMD targets. CMake accepts `gfx1100`, `gfx1101`, `gfx1102`, `gfx1103`, `gfx1150` and `gfx1151`, and rejects other architectures. The 24 GiB full-context profile was measured on an RX 7900 XTX (`gfx1100`). The 20 GiB RX 7900 XT and 16 GiB RX 7900 GRE share that target but do not fit the measured 23.64 GiB image at 393,216 positions; a smaller context or cache profile needs its own fit and correctness checks. The RX 7800 XT is `gfx1101` and needs an explicit build for that target.

Both the HIP and CUDA builds read the same GGUF text weights. The FP8 profile stores attention keys and values from previously read tokens in E4M3. The FP4 profiles use block-scaled E2M1 for the global cache and E4M3 for sliding-window attention. On `gfx1100`, Maya unpacks E4M3 in software for attention because the card has no native FP8 math. A separate image encoder reads `mmproj-F16.gguf` with either backend. Changed KV precision can change numerical results even when the text weights are identical.

## Build the HIP engine

You need Linux, ROCm with HIP and hipBLAS (the release image uses ROCm 7.2), CMake 3.24 or newer, Ninja, a C++20 compiler, and Python 3.10 or newer for the tokenizer and tests. Compiling needs the ROCm toolchain but not a GPU. The device tests do need a supported card.

```sh
git clone https://github.com/GodGMN/tintero
cd tintero
export ROCM="${ROCM:-/opt/rocm}"
cmake -S . -B build-ik2 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_PREFIX_PATH="$ROCM" \
  -DCMAKE_HIP_COMPILER="$ROCM/llvm/bin/clang++" \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-ik2 --target tintero -j"$(nproc)"
```

The server executable is `build-ik2/tintero`. To build the numerical tests as well, configure with `-DSTRATA_BUILD_TESTS=ON` and add the test targets:

```sh
cmake -S . -B build-ik2 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_PREFIX_PATH="$ROCM" \
  -DCMAKE_HIP_COMPILER="$ROCM/llvm/bin/clang++" \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-ik2 --target tintero ik2_kernels_test ik2_host_ref_test ik2_served_policy_test ik2_dflash2_test -j"$(nproc)"
```

`CMAKE_HIP_ARCHITECTURES` takes a semicolon-separated list. A build for an RX 7900 XTX and a Strix Halo machine in one executable uses `-DCMAKE_HIP_ARCHITECTURES="gfx1100;gfx1151"`. Select exactly one GPU backend per build directory: `STRATA_ENABLE_HIP=ON` and `STRATA_ENABLE_CUDA=ON` together fail at configure time.

The container build compiles both tested HIP targets and does not need the host's ROCm installation:

```sh
docker build -f docker/Dockerfile.hip -t tintero:hip .
```

## Verification

Check the linked executable's code objects and runtime dependencies:

```sh
"$ROCM"/llvm/bin/llvm-objdump --offloading build-ik2/tintero
ldd build-ik2/tintero
```

`llvm-objdump` must list a `gfx1100` code object when that is the configured target. `ldd` must resolve the HIP runtime and hipBLAS from the same ROCm installation.

The host-reference test needs no GPU. It checks packed NVFP4 decoding and the CPU expert path against the independent NumPy/PyTorch reference:

```sh
python3 tests/ik2_host_ref_fixture.py > ik2-host-ref.bin
./build-ik2/ik2_host_ref_test < ik2-host-ref.bin
python3 -m unittest tools.test_hip_compat
```

The fixture needs `numpy` and `torch`; the compatibility test only checks that every CUDA name compiled by the HIP sources has a mapping in `include/strata/hip_compat/`.

The device suite needs a supported GPU. It covers scalar expert, routing, FP8 and FP16 attention, WMMA tiles, paged KV and NVFP4 paths against numerical references:

```sh
ctest --test-dir build-ik2 -R '^ik2_kernels_test$' --output-on-failure
```

You can also run the binary directly with the deployment environment, for example `IK2_KV=fp8 ./build-ik2/ik2_kernels_test`.

### Forced decode comparison

Use identical prompt and forced token IDs in each run. The CLI writes its first raw logit row after consuming the first forced token. For `--gen N` it writes N−1 rows and omits the prompt prediction. The comparator checks expected byte counts and reports top-1 agreement, mean/median/max KL, and the position with the largest KL. It reads one vocabulary row at a time.

This example exercises token alignment only. For quality measurements, use tokenized real text and the same cache format, prompt and forced file in both runs.

```sh
MODEL=models/Inkling-2-Mini-IQ4_NL.gguf  # Set this to your downloaded GGUF.
printf '%s\n' '101,102,103' > force.csv
for graph in 0 1; do
  IK2_KV=fp8 IK2_GRAPH="$graph" IK2_FORCE=force.csv IK2_LOGITS_OUT="graph-$graph.f32" \
    ./build-ik2/tintero --model "$MODEL" --tokens 1,2 --gen 4 --ctx 256
done
python3 tools/ik2_compare_decode.py graph-1.f32 --baseline graph-0.f32 \
  --prompt-length 2 --gen 4 --vocab 200058
```

For a comparison against the independent model reference, run the reference on a machine with enough RAM for the GGUF forward pass. Pass the prompt followed by the first N−1 forced tokens. `--logprobs-out` saves only the final array, without intermediate layer dumps.

```sh
python3 ref/inkling2.py "$MODEL" --tokens 1,2,101,102,103 --logprobs-out ref.npy
python3 tools/ik2_compare_decode.py graph-1.f32 --ref ref.npy \
  --prompt-length 2 --gen 4 --vocab 200058
```

Run throughput measurements separately with logit export disabled. Exporting the full vocabulary adds a device-to-host copy per step.

## Download the model and export the tokenizer

The text engine expects the GGUF that contains the language model. Vision also needs the matching `mmproj` file.

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r docker/requirements.txt huggingface_hub
mkdir -p models ik2
hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-IQ4_NL.gguf --local-dir models
python3 tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-IQ4_NL.gguf \
  --out ik2 --chat-template tools/ik2/chat_template.jinja --check
```

The tokenizer is written to `ik2/tokenizer`. The preset in `docs/inkling2-mini/ik2-mini-gfx1100.json` already points at that directory and at `build-ik2/tintero`.

## Serve the IQ4_NL model on gfx1100

Run the server from the repository root so the preset's relative paths resolve:

```sh
python3 serve/server.py --engine maya --config docs/inkling2-mini/ik2-mini-gfx1100.json \
  --host 127.0.0.1 --port 18085
```

If the model or executable lives elsewhere, copy the preset and change `exe` and the `--model` value in `args`. The preset reserves 393,216 positions for one FP8 conversation with 1,024-token prefill chunks.

Wait for the model to load, then check the OpenAI-compatible endpoint:

```sh
curl -fsS http://127.0.0.1:18085/v1/models
curl -fsS http://127.0.0.1:18085/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"What is the capital of France? Reply in one word."}],"temperature":0,"reasoning_effort":"none","max_tokens":512}'
curl -NsS http://127.0.0.1:18085/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"What is 17 plus 25? Reply with digits only."}],"temperature":0,"reasoning_effort":"none","max_tokens":512,"stream":true}'
```

### Named FP4 profiles

The native `--max-context` argument is the sum of all resident slots. The HTTP `/health` field `max_context` is the limit on one conversation. Set `IK2_CACHE_SLOTS=N` or pass `--cache-slots N`; the engine divides the total context and the global KV budget across those slots.

The default FP8 profile uses 1,024-token prefill chunks. For two resident conversations, keep `IK2_KV=fp8` and set the slot count:

```sh
python3 - <<'PY'
import json
from pathlib import Path
source = Path("docs/inkling2-mini/ik2-mini-gfx1100.json")
config = json.loads(source.read_text())
args = config["args"]
args[args.index("--max-context") + 1] = "393216"
config["env"]["IK2_CACHE_SLOTS"] = "2"
Path("ik2-two-slots.json").write_text(json.dumps(config, indent=2) + "\n")
PY
python3 serve/server.py --engine maya --config ik2-two-slots.json --host 127.0.0.1 --port 18085
```

That gives two 196,608-token conversations. Three or four slots divide the same 393,216 positions evenly. Changing the slot count or the cache profile discards the previous resident caches; check `cache_slots`, `max_context` and `images` on `/health` after a reload.

The packaged gfx1100 image also ships two FP4 profiles that set the slot count, per-slot context and 512-token chunk together. They are generated by `docker/package-ik2-gfx1100.sh`, not stored in the source tree:

| Packaged preset | Slots | Context per slot | Aggregate positions | Allocated per cache |
|---|---:|---:|---:|---:|
| `ik2-mini-gfx1100-fp4-2.json` | 2 | 360,448 | 720,896 | 3,291.7 MiB |
| `ik2-mini-gfx1100-fp4-4.json` | 4 | 176,128 | 704,512 | 1,671.7 MiB |

In the packaged image, select one with `MAYA_IK2_PRESET`. To build an equivalent source preset, set `env.IK2_KV` to `fp4-global`, set `--max-context` to the aggregate positions, and add `--cache-slots N`. The measured capacities and their trade-offs are in the [named-profile measurements](#named-fp4-profile-measurements).

### Concurrent decode for short chat requests

Set `IK2_BATCH` to the maximum number of active requests, from 2 to 32, and use the gfx1100 Maya server. The native `--max-context` is the total across those slots. For 32 conversations of 4,096 tokens each, use `--max-context 131072`. A 256-token prefill chunk keeps the sliding-window allocation small.

```sh
python3 - <<'PY'
import json
from pathlib import Path
source = Path("docs/inkling2-mini/ik2-mini-gfx1100.json")
config = json.loads(source.read_text())
args = config["args"]
args[args.index("--max-context") + 1] = "131072"
args[args.index("--chunk") + 1] = "256"
config["env"]["IK2_BATCH"] = "32"
Path("ik2-batch.json").write_text(json.dumps(config, indent=2) + "\n")
PY
python3 serve/server.py --engine maya --config ik2-batch.json --host 127.0.0.1 --port 8088
```

The engine advertises `READY 4096 stop multi=32`. The HTTP server queues excess requests. Between decode steps the engine reads at most one prefill chunk, then predicts one token for every ready request. Each sequence keeps its own sampler, position, KV and convolution histories for all 48 visits. Completed and cancelled requests release their slot. Follow-up turns read their full prompt. Leave `IK2_BATCH` unset or set it to 1 for the single-request decoder with prefix reuse.

Concurrent decode requires a single HIP GPU with resident experts and planar Q8 attention projections and output head. Speculative decoding with a drafter uses the single-request server. For the concurrent path, leave `IK2_CACHE_SLOTS` unset or use the same count as `IK2_BATCH`. Increasing the slot count reserves a sliding ring for each slot as well as its global KV. The measured throughput and correctness checks are in the [concurrent decode measurements](#concurrent-decode-measurements).

## Image encoder

Images go through `maya-vision`, a small program built on llama.cpp's `mtmd` library. The encoder can run on the CPU or on the GPU. Upstream llama.cpp has the Inkling 2 image tower since tag `b12590`.

```sh
git clone --depth 1 -b b12590 https://github.com/ggml-org/llama.cpp third_party/llama.cpp-b12590
cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_DIR=third_party/llama.cpp-b12590
cmake --build build-vision --target maya-vision -j"$(nproc)"
```

The binary is `build-vision/bin/maya-vision`. For the HIP encoder, add the HIP options:

```sh
cmake -S tools/vision -B build-vision-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_DIR=third_party/llama.cpp-b12590 \
  -DSTRATA_VISION_HIP=ON -DSTRATA_VISION_HIP_TARGETS=gfx1100 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_HIP_NO_VMM=ON
cmake --build build-vision-hip --target maya-vision -j"$(nproc)"
```

To serve images, add a `vision` block to the preset and point `exe` at the encoder:

```json
"vision": {
  "exe": "build-vision-hip/bin/maya-vision",
  "model": "models/Inkling-2-Mini-IQ4_NL.gguf",
  "mmproj": "models/mmproj-F16.gguf",
  "gpu": true,
  "resident": true
}
```

Download `mmproj-F16.gguf` from the same GGUF repository. On a 24 GiB card the tested resident-HIP configuration uses 256-token prefill chunks; with 1,024-token chunks every expert no longer stays resident at the full context. The CPU encoder keeps the 1,024-token chunk and leaves the GPU to the language model. Vision measurements and encoder comparisons are in the [image encoder history](#image-encoder-on-the-ryzen-5-3600).

## CUDA and HIP differences in the initial scalar port

| Area | Phase 1 implementation |
| --- | --- |
| Wave size | The lane index, reductions, and broadcasts assume 32 lanes. The initial port restricted the target to gfx1100; CMake now also accepts the other Wave32 targets listed above. Strata's HIP intrinsics header supplies CUDA-compatible byte selection and warp operations. |
| Decode and expert MMA | `mma.sync.m16n8k16`, packed-half PTX and Hopper `wgmma` use NVIDIA-specific register layouts. HIP omits those kernels and dispatches the existing scalar products. The CUDA switches keep their original behavior on NVIDIA. |
| Prefill | The CUDA flash kernel uses `ldmatrix` and pipelined `cp.async`. Its dynamic shared-memory request is 88,064 bytes, above the 65,536-byte gfx1100 workgroup limit. HIP uses the existing tiled logits, softmax and strided batched GEMM path. With FP8 KV it unpacks cache chunks to FP16. This requires more scratch than CUDA flash. |
| Cache and weights | gfx1100 has no native FP8 or FP4 math. KV uses the software E4M3FN conversion, including subnormals. NVFP4 uses E2M1 nibbles and unsigned block scales. Device compilation selects device-resident IQ lookup tables. |
| Graphs | Global stream capture maps to HIP global capture. Instantiate and replay use the existing hipGraph wrappers. `IK2_GRAPH=0` runs individual launches for comparison. Replay and captured host/device copies need a card test. |
| Memory | The compatibility header maps managed allocation, advice, prefetch and the CPU device ID to HIP. Managed KV remains opt-in and checks the device's concurrent managed access attribute. Ordinary resident decode uses device allocations. |
| Speculative block verification | Block attention and its dense Q8 tile require CUDA MMA. HIP disables the block path. Serving rejects a drafter because the server verifies its proposals with that path. Explicit nonzero requests for `IK2_BLOCK`, `IK2_Q8_TILE`, `IK2_GROUP_MMA`, `IK2_EXPERT_MMA`, `IK2_ATTN_MMA`, `IK2_FLASH` or `IK2_WGMMA` fail at startup. |

## Measurement history

The sections below are the measurement record for the HIP port. They keep the dates, commit hashes, image digests, file hashes, benchmark commands, test-machine paths and hardware notes reported during development. Some commands use paths such as `/work`, `/weight.gguf` and `~/ai/...`, and some use the historical binary name `maya-ik2`; current source builds produce `build-ik2/tintero`. Use the portable guide above for a new deployment. No figure below was re-measured for this document.

### Build and host verification record

The initial HIP build ran in a ROCm 7.2 container limited to four CPUs and 6 GiB of RAM with swap disabled; after the build `/work` reported 21G free. `llvm-objdump --offloading` found gfx1100 code objects in the linked executable, and `llvm-readelf --notes` reported a wavefront size of 32. `ldd` resolved the ROCm 7.2.1 HIP runtime and hipBLAS.

The CPU host-reference fixture matched every packed NVFP4 weight exactly. The C++ expert's maximum absolute difference from the one-expert `Ref.moe` output was `1.90734863e-06`. `python3 -m unittest tools.test_hip_compat` passed. The CUDA 13 managed-memory shims are excluded from HIP.

The first loopback service check used a 32,768-token context and 1,024-token chunks. `GET /v1/models` returned `inkling-2-mini`. The non-streaming chat request returned “Paris”; the streaming request returned “42”, a stop reason and the final SSE `[DONE]`.

### Build on Prometheus without a GPU

The source tree is under `~/ai/build/maya-ik2-hip/src`. The build container mounts its parent at `/work`. The following commands reproduce the environment from a synced source tree.

```sh
cd ~/ai/build/maya-ik2-hip
docker run -d --name maya-ik2-hip-build --memory 6g --memory-swap 6g --cpus 4 \
  -v "$PWD:/work" -w /work --entrypoint sleep ghcr.io/ggml-org/llama.cpp:server-rocm-b11176 infinity
docker exec -it maya-ik2-hip-build bash
```

Run the following inside that container.

```sh
apt-get update -qq
apt-get install -y --no-install-recommends build-essential cmake ninja-build git python3 python3-pip libomp-dev
cmake -S src -B build-ik2 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-ik2 -j4 --target tintero ik2_kernels_test ik2_host_ref_test
```

The text engine links its own kernels and hipBLAS directly.

### Live service on Prometheus

On 2026-11-23, `hyperion/maya-ik2:gfx1100-service-r4` replaced r3 on port 8085. The image ID is `sha256:b66e209a0315be451a02ccce3f0819f9dee938cec763cb6a4f388285d6493042`. Hyperion still selects `inkling-2-mini`. The default is the stock ggml-org IQ4_NL file, one 393,216-token FP8 cache, GPU vision and medium reasoning. Two and four larger resident FP4 profiles are available at startup. [Profile selection](#named-fp4-profiles) · [Why FP8 remains the default](#same-slots-different-kv-precision)

On port 8085, all four OpenAI and Anthropic parallel tool conversations returned both correct desk codes, including streaming and a second turn with tool results. The three fresh screenshots returned 39%, 43% and 55% in 1.384, 0.861 and 0.650 seconds. The first was the first inference after startup. The strict OpenAI Python SDK passed through the LAN Responses router on port 8079. Stored-response retrieval by ID uses Maya’s direct port 8085.

The median across nine short streamed requests was **111.54 tok/s** end to end on r4 and **111.61 tok/s** in the same test on r3. The running native engine had SHA-256 `58fe3173ea7c0c1aa1197a2b9599f3c39fad40bd11571f04ea03f21c4ddb15a1`. Its actual arguments and environment confirmed one slot, 393,216 positions, chunk 1,024, FP8 KV and all experts resident. The live container was healthy with zero restarts and zero OOM kills under its 10 GiB memory and swap limits. Its server, model configuration reader and loop guard matched the checkout by SHA-256. The stock text and image weight files retained their original inodes, sizes and modification times.

```sh
python3 tests/ik2_image_answers.py http://192.168.100.111:8085 --screenshots
python3 tests/ik2_tool_conversations.py http://192.168.100.111:8085
python3 tests/ik2_responses_route.py http://192.168.100.111:8079 --direct-url http://192.168.100.111:8085 --sdk
python3 tools/ik2_http_bench.py --url http://192.168.100.111:8085 --max-tokens 129 --repeats 3
```

The stock release evidence is in `~/ai/build/maya-ik2-hip/fp4-overnight/results/` on Prometheus. `sha256sum --quiet -c evidence.sha256` verified all 478 listed files, including the replay inputs, raw responses, client timings and memory samples. The system disk had 9.04 GiB free after the copy.

The previous r3 service used image ID `sha256:41b5796fba7d9d09e515dc966ca962e3acd59180d9567be0d5a46729e0a2fecc`. It ran the same native text engine as r1, with 393,216 positions and one slot. GPU vision with host-resident projector weights returned 39%, 43% and 55% on three fresh screenshots in 1.557, 0.881 and 0.656 s. The first was the first inference request after the server became ready. A cold 8,220-token parcel lookup returned H317, taking 2.601 s for prefill (3,160 tok/s). This was one live request. The music scores below were collected on r1. At Sergio's request, the music benchmark was run only once and the remaining long timing runs were stopped.

All four OpenAI Chat Completions and Anthropic tool conversations passed on r3. Through the Hyperion LAN router, OpenAI Python SDK 3.57.0 with strict validation streamed both function calls and used their results in a stored follow-up. Check with `python3 tests/ik2_responses_route.py http://192.168.100.111:8079 --direct-url http://192.168.100.111:8085 --sdk`. Run `python3 ik-r3-short-check.py` from the Prometheus build directory for the parcel lookup.

On 2026-11-22, `hyperion/maya-ik2:gfx1100-service-r1` replaced `hyperion/llama.cpp:server-rocm-inkling-1427-r2` on port 8085. The first Maya image ID was `sha256:9f2df85e9fde0a5d4450c7f4cb87e4247bdb651d63ba5c4168d754dec9ed89d9`. Its native engine SHA-256 is `81cbe5f004ff5c0e64db8ead4fe13f1a7f410acbe26f5d01bdf6b33d9aa959b3`. Compose kept the controller's service name, port, device access, restart policy and 10 GiB memory and swap limits. The live health response reported one slot, 393,216 positions, images enabled and medium default reasoning effort.

The r1 image passed all 19 Inkling frontend tests. On port 8085 at that time, all four tool conversations returned Larch's VX-731 and Birch's PK-284, and all four image answers matched the incumbent. The CPU-encoded screenshot request took 13.965 seconds. The incumbent had its own 8,192-token reasoning budget. Maya's Inkling backend accepts a reasoning effort and a total output limit. Its separate thinking-budget endpoint reports `supported: false`.

```sh
cd ~/ai/build/maya-ik2-hip
docker inspect -f '{{.Config.Image}} {{.State.Health.Status}} {{.RestartCount}} {{.State.OOMKilled}}' inkling-2-mini
curl -fsS http://127.0.0.1:8085/health
curl -fsS http://127.0.0.1:8085/v1/models
curl -fsS http://127.0.0.1:8085/api/thinking-budget
python3 tests/ik2_tool_conversations.py http://127.0.0.1:8085
python3 tests/ik2_image_answers.py http://127.0.0.1:8085
```

In Hyperion select `inkling-2-mini`. The on-node control daemon starts the Compose project in `~/ai/services/inkling-2-mini/`. The separate Hyperion repository still contains its original llama.cpp recipe. Copying that recipe back over the service directory would restore llama.cpp.

Through the LAN router at `http://192.168.100.111:8079/v1`, a non-streaming Responses request called two functions. A streamed follow-up with `previous_response_id` and the results in reverse order returned both correct desk codes. Retrieving that stored response with GET returned HTTP 200 on Maya's own port 8085 and HTTP 502 on the router, which sends that GET to its default UI backend. The public gateway at `https://prometheus.gimeno.dev/v1` served Chat Completions during the music run and returned HTTP 404 for Responses. Use the direct port for stored-response retrieval.

Run `python3 tests/ik2_responses_route.py http://192.168.100.111:8079 --direct-url http://192.168.100.111:8085`. With OpenAI Python SDK 3.57.0, `python3 -m unittest serve.test_responses serve.test_responses_tools serve.test_responses_schema` passed all 55 tests. A streamed function completion now includes its name for strict validators in SDK 2.24.0. The current SDK also requires a description on returned namespaces and accepts only supported namespace members.

To restore the previous Maya r3 release, use its saved Compose configuration and controller description. If you selected an r4 FP4 preset in `.env`, clear that selection before starting r3.

```sh
cd ~/ai/services/inkling-2-mini
cp docker-compose.yml.pre-fp4-r4 docker-compose.yml
cp service.json.pre-fp4-r4 service.json
docker compose up -d --no-deps --pull never llama
```

The earlier llama.cpp image also remains on Prometheus. Its configuration and controller description have separate backups.

```sh
cd ~/ai/services/inkling-2-mini
cp docker-compose.yml.pre-maya-1429 docker-compose.yml
cp service.json.pre-maya-1429 service.json
docker compose up -d --no-deps --pull never llama
```

#### Keygen Bench

The live service ran the FastTracker II composition benchmark from the same source revision (`f86fa76a3de035078d09b8460ac553336aa4a413`), over the same public Chat Completions route and model alias as the prior llama.cpp run. Both used `xhigh` and a 16,384-token output limit. The native tool-readiness pilot passed before three independent attempts were reserved. The sandbox, visualizer, listener and trusted FT2 renderer were already installed on the controller host. The offline prerequisite check initially lacked the pinned Debian image manifest. Pulling its digest reused the existing base layer.

| Attempt | Maya listener score | Prior llama.cpp score | Maya wall time | Maya audio length | Largest Maya prompt | Public transport retries |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 61.2 | 29.0 (recovered XM) | 24m 52s | 29.99 s | 126,504 | 1 |
| 2 | 69.8 | 68.6 | 29m 02s | 25.60 s | 151,115 | 1 |
| 3 | 56.1 | HTTP 524, no XM | 1h 21m 28s | 7.99 s | 329,289 | 4 |

All three Maya attempts produced an XM that the trusted renderer could play. The best listener score was 69.8, compared with 68.6 on the prior llama.cpp run. Attempt 3 used 204 tool commands and generated 225,855 tokens across 205 measured responses. On 194 of 204 follow-ups it reused every preceding token. Four transport retries had unknown usage. The local bundle scanner withheld the bundle with `Secret or environment dump detected`. The individual render and listener results were available independently. No benchmark result or bundle was sent upstream.

The measured private run started at `20261122T142933Z`. In 276 samples taken about 30 seconds apart, the same server process stayed healthy with zero restarts and OOM kills. Its highest sampled anonymous RAM was 2.20 GiB, and device-wide VRAM was 23.08 GiB. The cgroup charge reached the 10 GiB limit, including file-backed model pages. Reproduce the per-attempt measurements with the commands below and the cache counts with the trajectory command under Growing conversations.

Run from the pinned Keygen checkout on the controller host. The saved provider config names `https://prometheus.gimeno.dev/v1` and `inkling-2-mini-26.4b-iq4-nl`. Supply the controller credential through `KEYGEN_CONTRIB_API_KEY`. The work and bundle paths must be new for each run.

```sh
state="$HOME/.local/share/keygen-bench-inkling"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
work="$state/maya-hip-1429-$stamp"
export KEYGEN_LISTENER_PYTHON="$state/listener-venv/bin/python" KEYGEN_FT2_ANALYSIS="$state/ft2-analysis"
export HF_HUB_OFFLINE=1 LITELLM_LOCAL_MODEL_COST_MAP=True
"$state/venv/bin/python" benchmark/contrib/cli.py doctor --config "$state/prometheus.json" --infrastructure-only --no-input
"$state/venv/bin/python" -u benchmark/contrib/cli.py run --config "$state/prometheus.json" \
    --work "$work" --out "$state/maya-hip-1429-bundle-$stamp" --yes --no-input
"$state/venv/bin/python" benchmark/score.py profile --out "$work" --workers 1
```

### Growing conversations

The single-slot FP8 image profile reserves 393,216 positions with a 1,024-token prefill chunk. The web client now sends Inkling's earlier reasoning back. API clients should retain the complete assistant message, including reasoning and tool calls, and preserve tool IDs when sending results. Changing the system prompt, tools, their order, or the reasoning level can change tokens near the beginning of the rendered history. A rewritten history starts a new prefix.

The first long chat found a direct-answer bug at 70,024 tokens. The engine reused all 70,003 available tokens, then generated the correct seal, switched to thinking and repeated the seal. The direct-answer path now ends generation at the first text segment's closing token. Reasoning and tool conversations still run to the end of their whole turn.

On build `27a63d9`, every append reused the preceding input and every generated token that the engine had processed. The sampled stop token is read when the next turn arrives. Two short follow-ups added only 22 tokens. After replacing the whole chat with a summary, the engine started at zero and resumed reuse on the following turn. Each answer returned the requested seal. All 12 matched their independent zero-cache replay on the complete response, finish reason and generated token count. The cold 300,000-token request took 259.365 seconds. VRAM stayed at 23.08 GiB after the first long prefill. Anonymous host memory reached 1.28 GiB at 300K and 1.29 GiB after the final cold controls. The remaining cgroup charge was mostly file-backed model pages. No OOM kills occurred.

An identical request retried after generation behaves differently from a new conversation turn. During the live Keygen run, a 27,812-token request reached its 16,384-token output cap in 178 seconds. The public controller sent the identical input hash a second time. The repeated request restored 25,736 tokens and reread 2,076. The native engine saves convolution and sliding KV at prompt boundaries. After it has decoded past the prompt, replaying that exact prompt needs the preceding checkpoint to regenerate its next-token logits. The following chat turn resumed normally and reused 31,035 of 31,035 available tokens. Compare `attempt-1/transport.jsonl` with `docker logs --since 2026-11-22T14:33:00Z inkling-2-mini` on Prometheus.

Two more turns in that attempt illustrate a template limitation. The client retained the returned reasoning, content and tool calls. Rendering assistant turn 19 with the packaged tokenizer produced 8,071 tokens from an 8,055-token generation. Turn 44 rendered in 7,925 tokens from a 7,926-token generation. A structured API message carries the text and parsed arguments but loses the original token boundaries and JSON spelling. The engine rolled back to the prior prompt boundary in these cases. Of 77 follow-ups, 74 reused every preceding token. The two re-renderings and the identical retry above account for all three misses. Reproduce the counts from the private run with `python3 tools/ik2_keygen_cache.py --tokenizer TOKENIZER_DIR "$work/attempt-1/trajectory.json"`. The audit emits counts only.

Keep the reasoning effort constant on image turns in a long conversation. On a 344,768-token FP8 archive, every medium-effort text branch restored the archive. A direct-answer screenshot changed the effort from medium to none. It returned the correct 39% but reused zero tokens and took 356.732 seconds for a 345,975-token request. The packaged tokenizer puts the effort in the first system message; for this archive, the first nine tokens match before `0.7` and `0` diverge. In the paired quality runner, use `--reasoning-effort medium --screenshots --screenshot-effort medium` to keep the prefix. The intentional effort-change control uses `--screenshot-effort none`.

A separate medium-effort chat added 17 and 26, then removed four. The first reply contained 415 characters of reasoning. Feeding back the complete message reused 174 of 174 available tokens. Omitting `reasoning_content` reused the 52-token prompt checkpoint. Both variants answered 39. See `retained-reasoning.jsonl` in the Prometheus build directory, reproduced with `python3 ik-reason-history.py` there.

| Turn | Prompt tokens | Cached | New | Wall seconds | Anonymous RAM, MiB |
|---|---:|---:|---:|---:|---:|
| 1 | 20,000 | 0 | 20,000 | 6.477 | 828 |
| 2 | 20,024 | 20,002 | 22 | 0.172 | 870 |
| 3 | 39,999 | 20,026 | 19,973 | 8.044 | 914 |
| 4 | 70,000 | 40,001 | 29,999 | 14.991 | 960 |
| 5 | 70,024 | 70,002 | 22 | 0.292 | 1037 |
| 6 | 114,999 | 70,026 | 44,973 | 28.907 | 1120 |
| 7 | 160,000 | 115,001 | 44,999 | 36.935 | 1208 |
| 8 | 204,999 | 160,002 | 44,997 | 44.724 | 1283 |
| 9 | 250,000 | 205,001 | 44,999 | 52.387 | 1294 |
| 10 | 300,000 | 250,002 | 49,998 | 67.577 | 1311 |
| 11 (compacted) | 48 | 0 | 48 | 0.117 | 1311 |
| 12 | 74 | 51 | 23 | 0.101 | 1311 |

Run from the Prometheus build directory with the full-context server on port 18085.

```sh
pid=$(docker inspect -f '{{.State.Pid}}' maya-ik2-api)
python3 tests/ik2_chat_growth.py http://127.0.0.1:18085 --server-pid "$pid" >growth-direct-full.jsonl
```

### FP4 global KV experiment on gfx1100

`IK2_KV=fp4-global` stores 32 consecutive channels as sixteen E2M1 bytes and one FP16 scale. It applies to keys and values of the eight full-attention visits. The 40 sliding-window visits keep E4M3. The text weights are the same stock IQ4_NL GGUF. At 393,216 tokens the global KV payload is 3.375 GiB per conversation, including scales. Conversation state and windowed KV require additional memory.

With the gfx1100 preset, the concurrent decoder batches FP4 global attention when the rows share a scalar attention bucket below position 8,192 and enough scratch space is available. Three kernel launches append the keys and values, compute attention and combine the partial results for the whole batch. Each row reads its own cache slot through its own position, including when several rows share a slot.

Run `IK2_DECODE_SUBWAVE=1 IK2_SHARED_COMBINE=1 ./build-ik2/ik2_kernels_test --fp4-batch` to compare with per-row execution on deterministic synthetic caches. The test checks the complete packed K and V buffers, including an unused slot, and compares the FP32 attention outputs bit for bit. Its GPU-event timings cover a single layer's cache append, attention and gated output. Each measurement averages 32 graph replays after five warm-ups and repeats with the order reversed.

The table gives the median of six measurements per path on an RX 7900 XTX with shared combine enabled (three runs in both orders). Both configurations use two cache slots. The two-row case advances each slot by one position. The eight-row case tests four consecutive positions in each slot. The second slot starts 19 positions earlier than the first.

| First request position | Rows | Per-row execution (µs) | Batched execution (µs) |
| ---: | ---: | ---: | ---: |
| 256 | 2 | 55.4 | 38.9 |
| 256 | 8 | 200.0 | 95.6 |
| 2,048 | 2 | 214.3 | 161.5 |
| 2,048 | 8 | 819.5 | 575.8 |

Build and check the FP4 kernels with the existing Prometheus build containers:

```sh
docker exec -w /work maya-ik2-hip-build cmake --build build-ik2 -j4 --target tintero ik2_kernels_test
docker exec -w /work -e IK2_FLASH_WAVES=4 -e IK2_ATTN_STAGED=1 -e IK2_DECODE_SUBWAVE=1 \
    maya-ik2-gfx1100 ./build-ik2/ik2_kernels_test
```

The binary built from `4d384de` passed the Radeon suite. The independent CPU tests cover clipped K, RMS-normalized V, empty blocks, nibble decoding, causal prefill, scalar decode, WMMA decode and block verification. The GPU vision profile kept the 1,024-image-token host-staged encoder loaded. With two caches, 393,216 tokens per slot failed the resident-expert check at both 512- and 256-token prefill chunks. At chunk 512, 385,024, 376,832 and 368,640 also failed. The server started at 360,448 per slot with 3,291.7 MiB per cache and a 4 MiB exchange buffer. The three screenshot requests returned 39%, 43%, and 55% in 0.785, 0.844, and 0.741 s on this configuration.

The extended kernel suite also passed FP4 causal prefill and decode at 392,957 keys against a double-precision sparse-attention calculation. Its four marked keys lay near 5%, 50%, 95% and the final position. The test filled the three trailing padded rows with large values to expose an incorrect attention mask. Across 768 prefill outputs the largest error used 5% of the test tolerance. The 384 decode outputs also passed. Run the same `ik2_kernels_test` command above.

The trial used the service-r3 image with the new executable mounted over `/opt/maya/build-ik2/tintero`. Set `IK2_CACHE_SLOTS=2` in the container environment. In a copy of the host-staged GPU vision preset, set `env.IK2_KV` to `fp4-global`, `--max-context` to twice the per-slot capacity, and `--chunk` to the tested chunk. Start with `MAYA_IK2_PRESET` pointing to the mounted preset. The established all-resident expert requirement and 256 MiB loader reserve remain in force.

```sh
python3 tests/ik2_image_answers.py "$URL" --screenshots
python3 tests/ik2_kv_quality.py --url "$URL" --mmlu mmlu32.jsonl
python3 tests/ik2_coding.py --url "$URL" --data humanevalplus.jsonl > coding.jsonl
python3 tests/ik2_coding.py --data humanevalplus.jsonl --score coding.jsonl --image ik2-evalplus:local
```

The coding test samples 16 task IDs with seed 1429 from [EvalPlus HumanEval+](https://huggingface.co/datasets/evalplus/humanevalplus/blob/d32357cf319e50e9c8d8dab5ea876c72b0fd321b/test.jsonl). The scorer runs each first answer against the dataset's supplied tests in an isolated, network-disabled CPU container. Build the local grader image with:

```sh
docker build -t ik2-evalplus:local - <<'DOCKER'
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.2.6
DOCKER
```

#### Serving an attention output LoRA

The native engine accepts `--lora FILE.gguf` for GGUF LoRAs with F32 A and B tensors on `blk.N.attn_output.weight`. It checks the base architecture, tensor dimensions, and each A/B pair. Other targets fail at startup. Each visit to an adapted layer adds the low-rank projection before the attention output convolution. The original quantized weights stay on disk and in VRAM. The adapter takes effect at process startup.

For an agent-facing service, copy the stock Maya server preset, append `["--lora", "/opt/maya/adapters/adapter.gguf"]` to its native `args`, and set a distinct `model_name`. Mount the adapter read-only. Register this model ID and the new service port in Hyperion's router, then sync its client configs. Both model IDs share the normal `/v1` endpoint. Stop the active full-context service before switching on a 24 GB card.

The regression fixture uses a rank-two adapter with a non-unit scale. A separate Q8_0 model expresses the same projection by copying two rows and flipping the sign of one. The test compares prompt and teacher-forced decode logits and repeats the requests with the adapter disabled. Only the changed matrix is written to a test shard. Run it with the production image's NumPy and a checkout mounted at `/source`:

```sh
python3 /source/tests/ik2_lora_reference.py /weights/Inkling-2-Mini-IQ4_NL.gguf /opt/maya/build-ik2/tintero
python3 /source/tests/ik2_lora_reference.py /weights/Inkling-2-Mini-IQ4_NL.gguf /opt/maya/build-ik2/tintero --adapter /adapter.gguf
python3 tests/ik2_responses_route.py "$ROUTER" --model "$MODEL_ID" --sdk
python3 tests/ik2_tool_conversations.py "$ROUTER" --model "$MODEL_ID"
```

With the final physical layer replaced, the native adapter and the independent weight edit agreed on the top token at all 12 prompt positions and three teacher-forced decode steps. The largest per-position logit RMS errors were 0.000986 and 0.0000651, respectively. Disabling the adapter increased RMS error to at least 0.1445. An earlier version of the test reshaped three decode rows as four. An early-layer version agreed on every top token but exceeded its logit tolerance after downstream layers. The final-layer fixture isolates the projection, and `--adapter` checks the actual adapter against CPU matrix products at every adapted visit. For a rank-16 adapter on 20 physical layers, the test compared 128 output channels on four input tokens across all 44 adapted visits. The worst relative L2 error was 0.0000322 against the full CPU projection and 0.000638 relative to the adapter contribution alone.

With the same 393,216-token FP8 preset on both services, the base and an attention-output rank-16 adapter each answered GSM8K test rows 128 through 135 correctly (8/8). Their median native decode rates on those requests were 120.15 and 70.50 tok/s. The two additional FP32 products run on every adapted layer visit. This sample measured arithmetic and decode speed. It was too small to establish general capability parity.

On Prometheus, the comparison driver and per-request timings are in `~/ai/build/maya-ik2-hip/unbound-test/`. With the respective model active, run:

```sh
python3 paired.py 8085 inkling-2-mini
python3 paired.py 8086 "$MODEL_ID"
```

Through the LAN router, all four streaming and non-streaming OpenAI and Anthropic tool conversations completed both calls and used both results. Pi's read-only agent fetched a random code from a local fixture and returned it exactly. A two-color image request also passed. The Responses API continued a stored response with `previous_response_id`. A separate GET by response ID returned HTTP 502 on the shared router and 200 at the service port on the host.

#### Adapter-free caption trial on Prometheus

The trial starts from stock IQ4_NL. The running Unbound service at handoff added a separate rank-16 adapter. Stock, the adapter, and the offline weight edits are distinct model configurations.

`IK2_CAPTURE_PREFILL_RESIDUALS=/path/to/new.f32` appends the last token's residual after each layer visit on the final multi-token prefill chunk. The record is three native int32s (visit, absolute token position, hidden width), followed by that many FP32 values. Set it only on a dedicated capture process. The first instrumentation missed the fused HIP prefill path and wrote no records. Moving capture to the layer-visit boundary produced all 48 visits for every one of the 80 training prompts. Each capture was 35,435,520 bytes. An independent reader checked every visit index, width and absolute position against the tokenized prompts.

Each prompt supplies an original, harmless sentence containing profanity and asks the model to transcribe it. The score requires the whole supplied sentence in the response, ignoring case, curly apostrophes and extra whitespace. The numerical fit uses only the 80 training prompts. The evaluation uses 28 other sentences, each in two wordings. Fitting and evaluation artifacts are under `~/ai/build/maya-ik2-hip/direction-trial/` on Prometheus. Each candidate uses a small second GGUF shard containing edited Q8 attention projections. The original GGUF is the first shard, and decode runs the stock executable.

```sh
python3 direction-trial/quotes.py 8085 inkling-2-mini stock
python3 direction-trial/confirm.py 8085 inkling-2-mini stock
# Repeat both sets against each service with its port, model ID and output label.
python3 direction-trial/quotes.py 8087 inkling-2-mini-direction caption80
python3 direction-trial/confirm.py 8087 inkling-2-mini-direction caption80
python3 tests/ik2_gsm8k.py --chat-url http://127.0.0.1:8087 --model-id inkling-2-mini-direction \
    --data fp4-overnight/results/gsm8k-test.jsonl --offset 128 --count 8 --max-tokens 1024
python3 tests/ik2_coding.py --url http://127.0.0.1:8087 --model-id inkling-2-mini-direction \
    --data fp4-overnight/results/humanevalplus.jsonl --count 4
python3 tests/ik2_tool_conversations.py http://127.0.0.1:8087 --model inkling-2-mini-direction
```

The selected edit (`caption80`) changes eight attention-output matrices and adds a 60,162,592-byte shard. Relative to stock it corrected three captions and lost one that stock had copied. Moving the fit to the start of the answer lost five stock successes and gained none. Applying one shared direction to sixteen matrices tied the selected edit at 22/56, with four gains and two losses. The eight-matrix edit preserves one more stock success and uses half the disk. Unbound copied all 56. These prompts measure one kind of over-refusal. They provide little evidence about how the edits handle other requests.

| Configuration | Captions (sets of 24 + 32) | GSM8K | HumanEval+ | Median native decode on GSM8K (tok/s) |
|---|---|---|---|---:|
| Stock | 7 + 13 | 8/8 | 4/4 | 119.75 |
| Unbound rank-16 adapter, active at handoff | 24 + 32 | 8/8 | 4/4 | 69.80 |
| Eight edited matrices, prompt boundary (selected) | 8 + 14 | 8/8 | 4/4 | 119.55 |
| Eight edited matrices, start of answer | 7 + 8 | 8/8 | 4/4 | 119.55 |
| Sixteen edited matrices, shared direction | 8 + 14 | 8/8 | 4/4 | 119.70 |

All five configurations answered the same short math and coding samples correctly. Math used greedy decoding with medium reasoning. HumanEval+ scored the first generated program against the problem's tests in a network-disabled CPU container. Decode is the median of the engine's eight per-request `predicted_per_second` values. The fractions cover only these twelve problems. In a separate eight-question yes/no check, stock and each offline edit matched all expected answers. Unbound matched five. The prompts requested no instructions.

```sh
# On the development machine, from the checkout, after copying the JSONL generations.
trial="$HOME/.local/share/maya-ik2/direction-trial"
for variant in stock unbound caption80 text80 shared; do
    python3 tests/ik2_coding.py --data "$trial/humanevalplus.jsonl" --score "$trial/$variant-code.jsonl" \
        > "$trial/$variant-code-score.jsonl"
done
python3 "$trial/summary.py"
```

Hyperion serves the selected model on its usual LAN endpoint under `inkling-2-mini-direction`. Pi and OpenCode list it as `prometheus/inkling-2-mini-direction`. The live native process opened `/trial/caption80/weights-00001-of-00002.gguf` with one 393,216-token FP8 cache and GPU vision. At handoff Docker reported healthy, zero restarts and no OOM. The router reported this variant running and both stock and Unbound stopped.

All four combinations of OpenAI Chat Completions or Anthropic Messages with streaming on or off completed both tool calls and used both results. Through the LAN router, a Pi agent read a new random code from a local file and returned it exactly. The model also identified the red half of a new two-color image. The strict OpenAI Responses SDK stream and follow-up with tool results passed. GET by response ID returned HTTP 502 on the shared router and 200 on the service's loopback port. The temporary container initially logged a read-only token-counter path. The registered service mounts its preset and token ledger in a writable state directory and mounts both weight shards read-only.

```sh
# Development machine
python3 tests/ik2_responses_route.py http://192.168.100.111:8079 --model inkling-2-mini-direction --sdk
python3 "$trial/agent-verify.py"
pi --offline --list-models inkling
opencode models | rg prometheus/inkling
# Prometheus, from ~/ai/build/maya-ik2-hip
python3 tests/ik2_responses_route.py http://127.0.0.1:8079 --direct-url http://127.0.0.1:8087 \
    --model inkling-2-mini-direction
docker top inkling-2-mini-direction -eo pid,args
docker inspect -f '{{.State.Health.Status}} restarts={{.RestartCount}} oom={{.State.OOMKilled}}' inkling-2-mini-direction
```

#### Runtime refusal ablation (experimental)

The caption trial edited eight attention-output matrices and barely moved refusals. Inkling cannot be abliterated
by editing weights, as measured below. Each attention and MoE output passes through a per-channel time convolution
before it joins the residual. Each layer then reads a learned mix of earlier block states, and the loop gates
rescale channels. `maya-ik2` therefore removes a direction at runtime, after every write has joined the residual
state: `y <- y - s r (r . y - c)`. That covers the normed embedding (pictures included) and each layer visit's
attention and MoE states, on the prefill, resident-token, block and batch paths. The block bank, the loop's kept
stream and its messages are copies of those states. The conv histories keep the raw outputs. Without
`IK2_ABLATE` nothing launches. One GPU only.

| Variable | Default | Meaning |
|---|---|---|
| `IK2_ABLATE=FILE` | off | the direction: the hidden size (2304) of raw little-endian floats, made unit length at load |
| `IK2_ABLATE_SCALE` | 1 | `s`; 0 or more; with `IK2_ABLATE_VISIT_SCALES` it applies to the embedding only |
| `IK2_ABLATE_VISITS=LO:HI` | all | only the writes of visits `LO .. HI-1`; the embedding only when `LO` is 0 |
| `IK2_ABLATE_VISIT_SCALES=LO:HI:S,...` | none | a strength per range of visits; visits not listed are left alone |
| `IK2_ABLATE_MEAN=FILE` | none | `1 + 2 n_visit` floats: the target `c` of the embedding, then each visit's attention and MoE state |
| `IK2_ABLATE_KEEP_NORM=1` | off | rescale each state to the length it had |
| `IK2_ABLATE_PREFILL_ONLY=1` | off | project only while a prompt is read |
| `IK2_ABLATE_DECODE_TOKENS=N` | all | batch server: project only a request's first `N` generated tokens |
| `IK2_CAPTURE_ATTN_RESIDUALS=FILE` | off | appends each visit's post-attention state at the last prompt token, in the `IK2_CAPTURE_PREFILL_RESIDUALS` record format |

`k::project_out` is checked in `ik2_kernels_test` against a double reference. The checks cover scales 0.8, 1 and
1.3, a nonzero target, per-row strengths, norm keeping, and that a row's bits do not depend on `T`. With a random
direction, the component left in every one of the 48 captured prefill states was about 1e-9 of the state,
against about 1e-2 without the projection. With batching on, four requests under a synthetic direction matched
all 128 serial predictions each.

**Procedure.** The scripts are in `tools/ik2_ablate/` and ran from `~/ai/build/maya-ik2-hip/ablate-lab/` on
Prometheus, with `tools/ablate_fit.py`, `tools/gguf_reader.py` and `tools/gguf_writer.py` copied beside them.
`lab.sh` restarts a lab server on 127.0.0.1:8087 from a `preset.py` preset: no vision, extra engine env,
`IK2_BATCH=32` with 4,096 tokens per request. `py` runs a script in the service image, which has numpy. The prompt
sets are `tools/ablate_fit.py prompts`: AdvBench harmful and Alpaca harmless, 416 train and 104 test each.
`label.py` sends every prompt once to stock with capture on, greedy with reasoning off. It stores only a refused
flag (`ablate_fit.refused`, a phrase match) and the capture's byte range, never prompt or answer text. Stock
refused 415/416 harmful and 15/416 harmless training prompts. `fit.py` takes, at each visit, the mean post-visit
state over refused harmful prompts minus the mean over answered harmless ones. `v21` is visit 21's direction.
`bench-q.sh` (with `evalq.py`) is the quick check, about 8 minutes through the batch server. It runs GSM8K's
first 128 with medium reasoning and 2,048 tokens, MMLU's first 256 with reasoning off, and both held-out refusal
sets with medium reasoning. `refnone.py` repeats the refusal sets with reasoning off. `final-bench.sh` adds all
164 HumanEval+ tasks (`he_all.py`, scored in the network-disabled sandbox), the tool conversations and the
single-request decode speed. Through the batch server, stock scored as it did serially.

**Results.** "Refused" counts the 104 held-out harmful prompts. "At limit" counts GSM8K answers still reasoning
at 2,048 tokens.

| Setting | Refused, reasoning off / medium | GSM8K /128 (at limit) | MMLU /256 |
|---|---|---|---|
| Stock | 103 / 104 | 125 (0) | 209 |
| `v21`, 1.0, visits 22-47 (A) | 42 / 3 | 120 (5) | 204 |
| `v21`, 1.0, 22-47, target 25% of the harmless mean (B) | - / 15 | 123 (2) | 206 |
| `v21`, 1.0, 22-47, and 1-21 at 0.3 | 16 / 0 | 113 (11) | 185 |
| `v21`, 1.0, 22-47, and 1-21 at 0.5 | 4 / 0 | 107 (16) | 175 |
| `v21`, 1.0, 22-47, and 1-21 at 0.7 | 1 / 0 | 107 (17) | 164 |
| `p22`, 1.3, all visits | 2 / 0 | 95 (23) | 151 |

| Setting | HumanEval+ /164 (at 4,096 tokens) | Tool conversations | Decode tok/s, one request |
|---|---|---|---|
| Stock | 153 (5) | 4/4 | 118.6 |
| A | 132 (17) | 4/4 | 114.7 |
| B | 139 (14) | 4/4 | 115.3 |

What did not help, all on `v21` at visits 22-47 unless noted:

- Medium-reasoning fits: 90-104 refused.
- The full harmless mean as the target: 98-103 refused, quality at stock.
- Projecting only the prompt: 103 refused.
- Projecting a request's first 32, 128 or 512 generated tokens: 102, 68 and 5 refused. The last scored 121 on
  GSM8K, as without the cutoff.
- Norm keeping: 2 refused, GSM8K 121, MMLU 199. Wider ranges were worse.

Refusal with reasoning on is decided in visits 22-47. Removing it costs about 5 GSM8K points, 5 MMLU points
and 21 HumanEval+ tasks, mostly reasoning that never ends. With reasoning off the decision also uses visits 1-21,
and projecting there costs much more. The early visits carry most of the damage.

**Weights.** `bake.py` projects the outputs of `attn_output` and every expert's `ffn_down_exps` off the span of
the post conv's four tap-weighted copies of the direction. The conv then cannot bring the direction back. It
writes an override shard that `maya-ik2` reads after the base file, or a whole GGUF with `--full`. Its
dequantization matches llama.cpp's `gguf-py` exactly. `--keep-type` re-quantizes IQ4_NL as IQ4_NL, which leaves
about a third of the removed component in the experts; Q8_0 leaves 0.6%. Neither moved refusals: baked layers
10-23 in IQ4_NL left 103/104 refused with medium reasoning, and layers 16-23 in exact Q8_0 also left 103/104.
A weight edit only stops new writes along the direction. It cannot remove what earlier writes put into the
state, which the runtime projection does at every visit. A portable abliterated Inkling needs a fine-tune that
learns the runtime behavior, not a weight projection.


#### Named FP4 profile measurements

The packaged host-staged vision build includes two explicit `MAYA_IK2_PRESET` values. Each preset sets the native slot count, per-request context budget and 512-token prefill chunk together. The native `--max-context` argument is the sum of all slots. The HTTP `/health` field `max_context` is the limit on one conversation.

| Preset | Slots | Context per slot | Aggregate positions | Allocated per cache |
|---|---:|---:|---:|---:|
| `ik2-mini-gfx1100-fp4-2.json` | 2 | 360,448 | 720,896 | 3,291.7 MiB |
| `ik2-mini-gfx1100-fp4-4.json` | 4 | 176,128 | 704,512 | 1,671.7 MiB |

Set `MAYA_IK2_PRESET` in `~/ai/services/inkling-2-mini/.env` to one of the filenames above and run `docker compose up -d --no-deps --pull never llama` from that directory. The new container starts with empty caches. Check `cache_slots`, `max_context` and `images` on `/health`. To explicitly select a single 393,216-token FP8 cache, set `MAYA_IK2_PRESET=ik2-mini-gfx1100-hip-host-vision.json` and `IK2_CACHE_SLOTS=1`.

For two resident conversations at the original precision, leave `MAYA_IK2_PRESET` empty and set `IK2_CACHE_SLOTS=2`. The release image served two 196,608-token FP8 slots with GPU vision. Both chats grew to 196,365 input tokens. All ten early, middle, late, image and sampled answers were correct. The screenshots took 1.615 and 1.818 seconds, with 195,463 tokens already cached in each conversation. The two sampled turns processed 21 new tokens apiece. Independent cold replays at 195,461 tokens matched both warm answers and took 128.998 and 129.040 seconds.

```sh
python3 tests/ik2_slot_retrieval.py --url "$URL" --slots 2 --depth 195584 --screenshots --sampled-turn --cold
```

The running dashboard's context setting is per conversation. Previously a request for 49,152 on two caches passed 49,152 total to the engine, leaving 24,576 per conversation. The reload now passes 98,304 total and saves that value to the config. Failure restores the prior per-slot capacity. The server also reads the training limit from the exact text GGUF passed with `--model`. The stock header reports 1,048,576. The tested GPU capacities are the ones in the table above.

On the packaged two-slot server, posting 8,192 to `/api/context` saved 16,384 native positions and finished reloading in 66.3 seconds. A request for 8,200 output tokens returned HTTP 400 with `context_length_exceeded`. Both existing desks then reused their full 237- and 236-token histories and returned their own codes. Restoring 360,448 per conversation saved 720,896 native positions and took 70.0 seconds. The trained limit and model path came from the stock IQ4_NL file in both configurations. The HTTP reload check is archived as `fp4-overnight/results/check-resize.py`.

```sh
python3 -m unittest serve.test_dashboard.Dashboard.test_inkling_context_reload_and_rollback_with_two_caches
curl -fsS "$URL/api/context"
```

At four slots, per-slot capacities 196,608, 192,512, 188,416, 184,320 and 180,224 all failed the resident-expert requirement. At 176,128 the server started and read three 1480×900 screenshots correctly in 1.385, 0.888 and 0.749 seconds. The minimum free VRAM sampled every 100 ms during those requests was 257.7 MiB. Both profiles use a shared 4 MiB slot-exchange buffer.

Four independent registers started at 174,934 tokens apiece. All twelve early, middle and late lookups were correct, along with all four color questions. The eight text follow-ups reused 174,936 or 174,959 tokens and processed 21 each in 0.273–0.483 seconds. The image follow-ups reused 174,982 tokens and processed 62 each in 0.358–0.377 seconds. Four independent 174,980-token cold replays matched the respective warm answers, lengths and finish reasons in 120.956–121.210 seconds. Device memory peaked at 23.734 GiB, leaving 256.7 MiB. The container reported zero OOM kills. The two-slot run also reported zero OOM kills.

```sh
python3 tests/ik2_slot_retrieval.py --url "$URL" --slots 4 --depth 175104 --cold > four-registers.jsonl
pid=$(docker inspect -f '{{.State.Pid}}' ik2-fp4-trial)
cg=/sys/fs/cgroup$(cut -d: -f3 "/proc/$pid/cgroup")
cat "$cg/memory.events" "$cg/memory.peak"
cat /sys/class/drm/card0/device/mem_info_vram_{total,used}
```

The packaged r4 image (ID `sha256:b47adcb1799042529b2079f83c083facf8f67d08dd832af54a3783d29ebe2815`) repeated the four-way workload with 1480×900 screenshots in place of color panels. All twelve register lookups and four image answers were correct. The image turns reused 174,982 tokens each and read 878–879 new tokens. Complete requests took 1.590, 1.766, 1.573 and 1.293 seconds. Device-wide free VRAM stayed at or above 256.7 MiB in 5,887 samples taken at 100 ms intervals. The container recorded zero OOM kills and zero restarts.

OpenAI Chat Completions and Anthropic Messages each passed streaming and non-streaming parallel tool calls with their results fed back in reverse order. The strict OpenAI Python SDK 3.57.0 streamed both calls, continued the stored response, and read both desk codes. Direct stored-response retrieval returned HTTP 200. After a rewritten history, the next turn reused the new prefix. All five warm answers matched independent cold replays after filling every cache with an unrelated request.

```sh
python3 tests/ik2_slot_retrieval.py --url "$URL" --slots 4 --depth 175104 --screenshots
python3 tests/ik2_tool_conversations.py "$URL"
python3 tests/ik2_responses_route.py "$URL" --direct-url "$URL" --sdk
python3 tests/ik2_chat_growth.py "$URL" --depths 2048,0,8192
```

The 10 GiB host-memory limit includes reclaimable file pages from the mapped weights. Across the two- and four-slot trials, sampled anonymous RAM peaked at 2.52 and 2.44 GiB respectively. The container cgroups reclaimed file pages at their limit. These host-memory numbers are separate from the device-wide VRAM readings.

The corrected r4 image also served four chats through a mix of greedy decoding, screenshots and seeded sampling. All 20 answers matched at the 8K input setting and all 20 matched near the limit. Each long chat started with 174,934 tokens. Its screenshot reused 174,982 tokens and took 1.304–1.311 seconds. Each sampled follow-up added 21 tokens and took 0.305–0.315 seconds. The longest request used 175,884 of the 176,128 positions. Across startup and these requests, 62,961 device-memory readings taken about 10 ms apart found a minimum of 256.008 MiB free. There were no cgroup OOMs or container restarts.

For concurrent HTTP admission, two clients at 8K and four at 4K started each turn together. All twelve follow-ups processed only 20 new tokens each. All six final replies matched a separate zero-cache replay in prompt length, generated tokens, answer and finish reason. The endpoint still has one decoding queue. `/health.cache_slots` counts resident histories while `/props.total_slots` reports simultaneous decodes.

```sh
python3 tests/ik2_slot_chats.py "$URL" --slots 2 --depth 8192 --concurrent
python3 tests/ik2_slot_chats.py "$URL" --slots 4 --depth 4096 --concurrent
python3 tests/ik2_slot_retrieval.py --url "$URL" --slots 4 --depth 8192 --screenshots --sampled-turn
python3 tests/ik2_slot_retrieval.py --url "$URL" --slots 4 --depth 175104 --screenshots --sampled-turn
# On Prometheus, run alongside the test with a fresh marker path. Touch that path to finish.
python3 fp4-overnight/peak-sampler.py /tmp/ik2-finish-profile-001
```

Four clients also submitted **175,022-token cold conversations simultaneously** to the release image. Maya read one at a time. The four clients received their correct answers after 124.172, 245.723, 367.362 and 489.159 seconds, including queue time. The same four clients then sent two simultaneous follow-up rounds. All eight replies were correct. Each request processed 20 new tokens and reused its entire 175,024- or 175,046-token prefix. Client latency across the eight warm requests was 0.279–1.354 seconds.

All four independent cold replays processed 175,066 tokens, matched their own warm answer, generated length and finish reason, and took 121.552–121.724 seconds. The server passed the image and tool checks afterward. Across 98,104 device-wide readings about 10 ms apart, minimum free VRAM was 256.63 MiB. The container recorded zero OOM kills and zero restarts. Results are in `fp4-four-final-{chats,screenshots,tools}.jsonl` and `fp4-four-final-vram-10ms.tsv`.

```sh
python3 tests/ik2_slot_chats.py "$URL" --slots 4 --depth 175104 --concurrent --images
```

```sh
# From ~/ai/build/maya-ik2-hip on Prometheus
sh docker/package-ik2-gfx1100.sh build-ik2/tintero ik2/tokenizer \
    hyperion/maya-ik2:gfx1100-service-r4 vision-cpu/bin/maya-vision \
    vision-resident/maya-vision vision-hip/bin/maya-vision
docker run --rm --network none --memory 768m --memory-swap 768m --cpus 2 --entrypoint python3 \
    hyperion/maya-ik2:gfx1100-service-r4 -m unittest serve.test_inkling2_frontend \
    serve.test_responses serve.test_responses_tools serve.test_responses_schema
```

The first packaged FP4 trial used engine SHA-256 `f08cbeb153f26ea7e1fcf219a1f61fdca50aba7c737b19541d77dadbac822454`. The live release digest is recorded at the top of this log. The packaged Python suites ran 74 tests with three skips and zero failures. All 74 passed on the development host with OpenAI SDK 3.57.0, jsonschema 4.26.0 and Jinja2 3.1.2. The shipped image includes jsonschema and skips the three optional SDK tests. An initial host run used SDK 2.24.0 without jsonschema and failed structured-output and namespace checks. A fresh Python 3.14 environment also lacked Jinja2. The complete run used the existing Python 3.11 SDK environment.


#### Repeating the paired checks

Save each server's JSONL output separately. Compare question IDs, expected answers, input token counts and available request-body SHA256s with the following commands. The report lists the specific gains and losses. For MRCR it averages the published prefix-sensitive string similarity. HumanEval+ requires executing the returned programs against the supplied tests before comparing scores.

```sh
python3 tests/ik2_compare_quality.py fp8-mmlu256.jsonl fp4-mmlu256.jsonl
python3 tests/ik2_compare_quality.py fp8-medium.jsonl fp4-medium.jsonl
python3 tests/ik2_compare_quality.py fp8-long-code-score.jsonl fp4-long-code-score.jsonl
```

The first FP4 direct-answer archive run predates request hashing. Its generator used the same message bodies, output limits and JSON encoding, and all paired input token counts matched. Later runs report how many hashes matched.

#### Growing stored Responses conversations

`tests/ik2_responses_growth.py` sends only each new batch and the previous response ID. It repeats the same instructions on every request. Each chat receives a lot-to-depot mapping and, in a later batch, a depot-to-number mapping. The final question requires both. It then tests normal-temperature sampling, a same-effort screenshot and a rewritten summary. The cold control explicitly evicts all native slots and replays the complete input, including the model's earlier reasoning. Short runs can repeat each cold control to check reproducibility.

```sh
python3 tests/ik2_responses_growth.py "$URL" --batches 512,1024,1536 --serial --repeat-cold
python3 tests/ik2_responses_growth.py "$URL" --batches 20000,80000,150000,95000 --serial
# Omit --serial to interleave both stored conversations on the two-slot profile.
```

On the short FP8 control both final numbers, both screenshots and both post-compaction numbers were correct. The visible answers matched their cold replays. For the 3,490-token lookup, warm and cold generation took 1,661 and 480 tokens. At 3,693 tokens they took 194 and 199. Repeating each cold input after eviction reproduced its entire reasoning text and output count exactly. A 512-token output cap in the first pilot exhausted the reasoning budget before the first number. The recorded comparison uses 4,096. This difference in reasoning between incremental and cold execution already occurs with the stock FP8 cache.

The two-slot FP4 smoke run also kept both answers correct and fully cached through sampling and image turns. Its two warm/cold lookups generated 406/355 and 1,046/358 tokens. Both independently repeated cold runs reproduced their respective reasoning and visible answers. Add `--save-controls DIRECTORY` to retain the exact cold Responses request bodies for cross-format replays.

In the full run, two independent Responses chains grew past 345K. FP8 ran them in sequence on one cache. FP4 alternated between two resident caches on every turn. Each new request reused every preceding token that had reached the engine. The service generated different reasoning lengths between formats, so their absolute prefix lengths diverged even though the user batches were identical.

| FP4 request | Chat A cached | Chat B cached | New input in each chat |
|---|---:|---:|---:|
| First batch | 0 | 0 | 20,127 |
| Second batch | 21,540 | 20,872 | 80,099 |
| Third batch | 102,029 | 101,148 | 150,012 |
| Join lot and depot records | 252,122 | 251,243 | 95,027 |
| Sampled follow-up | 347,400 | 346,532 | 13 |
| 1480×900 screenshot | 347,531 | 346,697 | 876 |
| Rewritten summary | 0 | 0 | 60 |
| Follow-up after compaction | 164 | 195 | 20 |

Both profiles answered the two-hop questions, read 39% and 43% from the screenshots, and recovered both codes after compaction. The two warm image requests took 3.829 and 4.377 seconds for FP8, and 4.881 and 5.248 seconds for FP4, including medium-effort reasoning. Cold replays of the FP8 lookups at 345,717 and 345,685 tokens produced the same final answers in 360.553 and 357.783 seconds. Both FP4 replays also matched, at 347,149 and 346,270 tokens in 376.699 and 380.167 seconds. Warm and cold reasoning differed on all four comparisons. On FP4, the first lookup generated 252 tokens warm and 345 cold, the second 263 warm and 481 cold.

#### Teacher-forced logits on the stock IQ4_NL weights

The CPU oracle used `ref/inkling2.py` on the same GGUF and the same 256-token passage. Each Radeon run used the environment in `ik2-mini-gfx1100.json` with the indicated override. The decode test feeds 32 known continuations after a 224-token prompt. Its 32 predictions cover the tail of the prefill passage, so the two rows are overlapping samples.

| Input | CPU top-1 matches, FP8 | CPU top-1 matches, FP4 | Mean KL from CPU, FP8 | Mean KL from CPU, FP4 | FP8/FP4 top-1 matches |
|---|---:|---:|---:|---:|---:|
| 256 prefill positions | 250/256 | 252/256 | 0.002041 | 0.004161 | 250/256 |
| 32 forced decode positions | 30/32 | 32/32 | 0.002114 | 0.004454 | 30/32 |

FP4 put more probability on a different part of the vocabulary even when the most likely token agreed. Over the 255 next-token targets in the full passage, perplexity was 3.9677 on the CPU, 3.9678 with FP8 and 3.9419 with FP4. On the 31 scored targets in the forced tail, those values were 2.5071, 2.5316 and 2.6072. This single passage measures numerical drift and supplies no independent estimate of benchmark accuracy.

Use the existing `heldout-teacher.{all,prompt,force}` and `heldout-teacher-ref.npy` fixture with the deployed Radeon environment in the GPU container, from `/work`:

```sh
for kv in fp8 fp4-global; do
    IK2_KV="$kv" IK2_PREFILL_LOGITS_OUT="$kv-pp.f32" ./build-ik2/tintero \
        --model /weight.gguf --ctx 1024 --chunk 64 --tokens @heldout-teacher.all --gen 0
    IK2_KV="$kv" IK2_FORCE=heldout-teacher.force IK2_LOGITS_OUT="$kv-tg.f32" ./build-ik2/tintero \
        --model /weight.gguf --ctx 1024 --chunk 128 --tokens @heldout-teacher.prompt --gen 33
    oracle-env/bin/python tools/ik2_compare_decode.py "$kv-pp.f32" --ref heldout-teacher-ref.npy \
        --tokens heldout-teacher.all --vocab 200058 --prompt-length 256 --prefill
    oracle-env/bin/python tools/ik2_compare_decode.py "$kv-tg.f32" --ref heldout-teacher-ref.npy \
        --tokens heldout-teacher.all --vocab 200058 --prompt-length 224 --gen 33
done
```

#### Paired short and long prompts

The deployment decision uses the stock FP8 model as the control. A two-slot default needs correct cold replays, prefix reuse, working vision, at least 256 MiB of sampled device headroom, and comparable answers on the same short and deep prompts. A reproducible loss of a long-distance record that FP8 recovers is grounds to keep FP8 as the default. The scored tasks below are local paired samples, including when they use public benchmark questions.

All 16 short HumanEval+ tasks passed on both service-r3 FP8 and the two-slot FP4 build. Both used medium reasoning, greedy sampling and a 4,096-token limit. The 16 canonical functions passed the same grader, and replacing each function with one that raises an exception failed all 16. The two KV formats generated 10,565 and 11,750 tokens respectively. The longest short coding prompt fit in one prefill chunk in both configurations.

Two untruncated eight-needle conversations from [OpenAI MRCR](https://huggingface.co/datasets/openai/mrcr/tree/f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d) contained 344,768 and 348,465 input tokens. Their published answers require 328 and 730 tokens in the Inkling tokenizer. The direct-answer test allows 768. Each follow-up uses the published answer in the history so the input stays identical across formats. The separate register test appends the model's own answer, and checks the number of tokens reused on every turn.

```sh
python3 tests/ik2_kv_quality.py --url "$URL" --mrcr mrcr-mid-0.json mrcr-mid-1.json \
    --mmlu mmlu32.jsonl --code humanevalplus.jsonl > long.jsonl
python3 tests/ik2_coding.py --data humanevalplus.jsonl --score long.jsonl
python3 tests/ik2_slot_retrieval.py --url "$URL" --slots 2 --depth 359424 --cold > registers.jsonl
```

The MRCR files correspond to default train rows 1850 and 1870. The FP4 match ratios were 0.0180 and 0.0558. The model reproduced the eighth poem and the eighth diary entry in the respective histories. The requests asked for the seventh and fourth. FP8 returned exactly the same answer text on both rows, including both ordinal mistakes. On the 32 short MMLU questions each format scored 26/32 on precisely the same items. One incorrect letter changed to another incorrect letter.

| Public eight-needle input | Cold prefill, FP8 / FP4 | Generated tokens in both | Decode tok/s, FP8 / FP4 |
|---|---:|---:|---:|
| 344,768 tokens | 353.1 / 370.0 s | 374 | 39.0 / 27.4 |
| 348,465 tokens | 359.8 / 376.8 s | 536 | 38.8 / 27.1 |

The FP8 server ran with one slot and `--serial`. The FP4 server alternated two slots. Both used the same native binary, chunk 512, per-slot maximum of 360,448, host-staged GPU vision and the same chat template. Each retrieval request reported zero cached tokens. The table uses server-side timings from `tests/ik2_kv_quality.py`. The two published rows alone cannot estimate full-benchmark accuracy.

Two independent parcel registers each occupied 359,254 input tokens before the first answer. FP4 retrieved all six randomly chosen verification numbers at 5%, 50% and 95% of the respective archive text. Alternating the two chats reused 359,256 and 359,279 tokens on the text follow-ups. Each read 21 new tokens in 0.431–0.716 seconds. The final image questions each reused 359,302 tokens and processed 62 more. Both colors were correct, in 0.518 and 0.536 seconds. Each final text query then replayed from zero cached tokens. The cold requests took 384.256 and 384.627 seconds and matched the warm replies, generated token counts and finish reasons exactly. The highest device-wide memory sample during these turns was 23.613 GiB, leaving 380.6 MiB on the card.

On the direct-answer MRCR follow-ups FP4 scored 23/32 on MMLU and 1/4 on the short coding set embedded in those histories. Two multiple-choice answers and one coding response reached their output caps. The coding failures were a missing `typing.List` import, an unfinished answer, and a time-out in the supplied tests. The separate 16-task short coding run used medium reasoning. A comparison of short and long coding scores must account for both the archive and the reasoning setting.

On nine short HTTP requests with 129 generated tokens each, median end-to-end throughput was 111.64 tok/s for one FP8 slot and 110.37 for two FP4 slots. Both servers reserved 360,448 positions per conversation and used 512-token prefill chunks. Across the six requests after the first pass through the three prompts, median time to first text was 83.66 ms and 86.48 ms respectively. At 345K the decode rates in the table above show a larger difference.

```sh
python3 tools/ik2_http_bench.py --url "$URL" --max-tokens 129 --repeats 3
```

##### Matched capability scores

| Task and setting | FP8 | FP4 | Items that changed correctness |
|---|---:|---:|---|
| 256 MMLU, short, direct answer | 207/256 | 206/256 | 3 gains, 4 losses |
| First 128 GSM8K, short, medium reasoning | 125/128 | 123/128 | 2 losses |
| 16 HumanEval+, short, medium reasoning | 16/16 | 16/16 | None |
| 32 MMLU on the 345K and 348K archives, direct answer | 24/32 | 23/32 | 1 loss |
| 4 HumanEval+ on the same archives, direct answer | 1/4 | 1/4 | None |
| 16 feasible/impossible programming specifications, direct answer | 9/16 | 9/16 | None |

On GSM8K test item 12 both models computed that the tree breaks even after 12 years. FP8 gave the first profitable year, 13, and FP4 returned 12. Neither answer hit its cap. On item 106, FP8 computed the inner volume of three boxes correctly in 551 tokens. FP4 exhausted its 4,096-token limit without a final answer. All 128 request hashes and input lengths matched. On the long college mathematics item 0, FP8 returned the correct letter. FP4 began reasoning in the answer and reached the eight-token cap. The feasibility questions are a local check of whether the model can recognize missing information. Both direct-answer runs recognized all eight impossible tasks and only one of the eight feasible tasks.

At medium reasoning effort and a 4,096-token output limit, both formats returned the requested fourth entry from the 348,465-token archive. The official string match ratio was 0.998739 for FP8 (2,364 output tokens) and 0.998379 for FP4 (3,536 tokens). Relative to the reference, FP8 inserted seven characters and FP4 inserted seven and omitted two. Both exhausted the output budget without a final answer on the 344,768-token archive that asked for the seventh entry. All four retrieval requests started cold. On the 16 feasible and impossible specifications, both formats answered 16/16 correctly with medium effort. All corresponding request hashes matched.

```sh
python3 tests/ik2_kv_quality.py --url "$URL" --mrcr mrcr-mid-0.json mrcr-mid-1.json \
    --mmlu mmlu16.jsonl --code humanevalplus.jsonl --screenshots --reasoning-effort medium
python3 tests/ik2_feasibility.py --url "$URL" --reasoning-effort medium
```

Eight further eight-needle MRCR rows contained 23,365 to 60,974 input tokens. Their mean direct-answer match ratios were 0.0600 on FP8 and 0.0773 on FP4. Neither model reproduced the requested answer on any row. In five cases the generated text was identical. The rows were 1714, 1725, 1750, 1797, 1920, 1959, 1979 and 1992 in the default train split. Results and the exact inputs are in `~/ai/build/maya-ik2-hip/fp4-overnight/results/` on Prometheus.

On the first medium-effort run of row 1714 (23,365 tokens), FP8 copied the requested seventh article verbatim. FP4 copied the eighth verbatim. Both emitted the correct identifying prefix and finished within 4,096 output tokens. The evaluator records each exact match against the previous articles as well as the official string-similarity score. Use `--require-cached-archive` for the multi-turn MMLU and coding checks to fail immediately if a follow-up rereads its archive.

##### Same slots, different KV precision

The deployment comparison uses one FP8 slot and two FP4 slots. To separate precision from slot count, the release image also ran both formats with **two 65,536-token slots**, 512-token prefill chunks, the same stock IQ4_NL file, GPU vision, greedy sampling, medium effort and a 4,096-token output limit. The two configuration files matched after removing `env.IK2_KV`, with SHA-256 `9fd0f224192f6540c5ea12511d3776b85a6f694ad19e5de46b46e72de92c3714` over the remaining settings. All four request hashes matched.

| Cold request | FP8 | FP4 global |
|---|---|---|
| MRCR row 1714, 23,365 tokens, asked for the seventh entry | Copied the seventh verbatim | Copied the eighth verbatim |
| MRCR row 1797, 34,253 tokens, asked for the fourth entry | Copied the fourth verbatim | Reached 4,096 output tokens without a final answer |
| GSM8K item 12, expected 13 | 13 | 12 |
| GSM8K item 106, expected 72 | 72 | Reached 4,096 output tokens without a final answer |

On the larger two-slot profile, a separate FP4 retry of row 1797 still had no final answer at 8,192 tokens. At that budget GSM8K item 106 returned 108 after 7,418 tokens. Across the eight 23K–61K MRCR questions, each format reached 0.99 string similarity on four questions. The successful questions differed. The repeated loss of records that FP8 recovered is the reason FP8 remains the default.

From the checkout root, `python3 "$E/crossover.py"` runs the two isolated configurations on an idle Prometheus with the live service stopped. Set `E` to the archived `results` directory or a working copy of it. The paired responses are in `crossover-{fp8,fp4}-{retrieval,gsm}.jsonl`. The broader comparison uses `python3 "$E/final-pairs.py" stock`.

```sh
python3 tests/ik2_kv_quality.py --url "$URL" --mmlu mmlu256.jsonl
python3 tests/ik2_gsm8k.py --chat-url "$URL" --data gsm8k-test.jsonl --count 32 --max-tokens 4096
python3 tests/ik2_gsm8k.py --chat-url "$URL" --data gsm8k-test.jsonl --offset 32 --count 96 --max-tokens 4096
python3 tests/ik2_feasibility.py --url "$URL"
python3 tests/ik2_kv_quality.py --url "$URL" --mrcr mrcr-short/row-*.json
```

### Separate conversation caches

#### Distinguishing reasoning loops from answer loops

During the first FP4 medium-effort HumanEval/130 run at 348K, the server classified three copies of a 153-token reasoning block as a loop in the final answer. Its next retry stopped with a server notice and no Python function. Inkling marks thinking, text and tool JSON with distinct vocabulary tokens. The server's generic guard had only looked for a closing `</think>` and XML-style tool calls.

The Inkling guard now follows those segment tokens, including a tool header already present in the request. If a persistent reasoning loop needs to be closed, the retry prompt ends with `<|end_message|><|message_model|><|content_text|>`. An answer loop can still stop the response. The native sampler currently ignores the generic repetition penalty setting. The server can still trim unsent repeats, retry and close reasoning.

Three scripted requests failed before the change. They cover a reasoning passage that naturally resolves, a persistent reasoning loop followed by a final answer, and repeated legitimate tool calls with identical JSON. A fourth checks that visible answer loops still stop. The following command passed all 144 tests on the development host. The packaged image ran the same suite with three optional SDK skips.

With both fixes in the image, all 20 follow-ups on the 344,768- and 348,465-token archives retained their base prefix in both formats. FP4 scored 16/16 on deep MMLU and FP8 scored 15/16. Both produced final letters on all 16 questions. FP8 missed college mathematics item 1. On HumanEval/130, FP4 repeated a 153-token block four times during reasoning. After two retries the server closed thinking at token 2,775. The model returned a Python function. Its memoized implementation exhausted the 256 MiB CPU grader, giving 3/4 deep coding passes. FP8 passed all four on both server versions. All 22 request-body hashes matched across the corrected runs.

The published test calls the function repeatedly with inputs around one million. The FP4-generated inner function captures its own memo dictionary and references itself. The same unmodified test also triggered a cgroup OOM kill at 512 MiB and 1 GiB. As a diagnostic, forcing Python to collect cycles **between** test inputs let the function pass every assertion at a 192.6 MiB peak. FP8's iterative function passed the original test at a 97.5 MiB peak in the 256 MiB grader. The diagnostic does not count toward the score.

Two additional clients submitted the full archives at the same time on the fixed FP4 server. They returned `ALDER-17` and `ASPEN-29` after 4.876 and 10.271 seconds, with 344,768 and 348,465 cached input tokens. The exact requests and diagnostic commands are in `fp4-overnight/results/` on Prometheus.

After the request accounting change, both FP8 and FP4 reproduced all 22 of their previous deep answers on the release image. Each profile reported zero cache hits on the two cold inputs and retained the full archive on all 20 follow-ups. A second pair of concurrent FP4 clients recovered the same two codes in 4.844 and 10.273 seconds. The server passed the four OpenAI/Anthropic tool conversations and the strict OpenAI Responses SDK check, with zero container restarts and zero cgroup OOM kills. The raw records are `fp4-accounting-{deep,concurrent,tools,responses}.jsonl`. The two-client replay uses `python3 "$E/concurrent-archives.py"` from the checkout root with the test archives resident. Set `E` to the archived `results` directory.

The 8,192-output-token FP8 control exposed a separate reporting error. It read the full 344,768-token prompt from scratch, then reported 344,395 cache hits. On the final native stop token the client-facing generator closed before its retry statistics were merged. The last retry's cache count and prefill time replaced the original read. Maya now saves each native completion record while its generator drains and calculates usage once for the whole HTTP request. A scripted cold request previously reported 47 cache hits and a 1 ms prefill. The corrected response reports zero hits and the first native call's 12,345 ms fixture time, and sums all four native decode times. The combined local regression run covered 386 tests with one skip. The packaged request and SDK suite covered 145 tests with three optional SDK skips.

The corrected FP8 image replayed the same cold request on Prometheus. It reported 344,768 new tokens, zero cached, 354.784 s of prefill and 7,191 generated tokens. At the higher budget it returned four lines of verse with a 0.2115 official match ratio. FP4 exhausted all 8,192 output tokens without a final answer. Neither recovered the requested poem. From the checkout root, run `python3 "$E/long-budget.py"` against each matched profile. The measured responses are `fp8-guard-deep-8192.jsonl` and `fp4-guard-deep-8192.jsonl`.

```sh
python -m unittest serve.test_loop_guard serve.test_inkling2_frontend \
    serve.test_responses serve.test_responses_tools serve.test_responses_schema
python -m unittest serve.test_inkling2_frontend.Serving.test_guard_retries_report_the_first_native_prefill
python3 tests/ik2_coding.py --data humanevalplus.jsonl --score fp4-guard-deep.jsonl
python3 grade-tri-memory.py
python3 grade-tri-memory.py --collect
```

The broader CPU regression run covered 527 tests with 22 skips and zero failures, including fair queueing, stop tokens, streaming and stored responses. The command uses Bash brace expansion and the development machine’s OpenAI SDK test environment.

```sh
python -m unittest serve.test_{engine_contract,fair,stop,stream_jobs,response_jobs,response_store,savings,think_budget,tools,logprobs,prompt_memo,server,model_api,multi,inkling2_frontend,loop_guard,responses,responses_tools,responses_schema}
```

#### Slot selection

Set `IK2_CACHE_SLOTS` in Compose or pass `--cache-slots N` to the native server. With the default decoder, one request runs at a time. Each slot retains its token and image history, KV, convolution state and prefix checkpoints. The engine selects the longest prefix it can restore. A new conversation uses an empty slot first, then the least recently used one. A restart clears all slots. With a single slot the engine allocates no exchange buffer.

An early r4 build evicted chat A when the loop guard retried chat B from a checkpoint. The next HumanEval/62 request had to read all 345,238 input tokens again in 370.542 seconds. A divergent continuation now forks into an empty slot if available. When all slots are occupied it rewinds its matching chat in place. A cancelled prefill releases its slot for the next new chat.

On the 7900 XTX the old executable lost all 1,799 live tokens of the other chat on a rewind and all 1,795 on a prefill cancellation. With the fix, both chats survived the rewind and each matched eight independent cold predictions in FP8, FP4, and FP4 with the official MTP drafter. The fixed cancellation test retained all 1,795 tokens and matched all eight cold predictions.

```sh
# From /work in maya-ik2-gfx1100, with the usual gfx1100 preset environment
IK2_KV=fp4-global python3 tests/ik2_prefix_cache.py --engine ./build-ik2/tintero \
    --model /weight.gguf --tokens water.force --fork-slots
IK2_KV=fp8 python3 tests/ik2_prefix_cache.py --engine ./build-ik2/tintero \
    --model /weight.gguf --tokens water.force --fork-slots
IK2_KV=fp4-global IK2_DRAFT_ADAPT=0 IK2_DRAFT_WIDTH=2 python3 tests/ik2_prefix_cache.py \
    --engine ./build-ik2/tintero --model /weight.gguf --tokens water.force --draft mtp.safetensors --fork-slots
IK2_KV=fp4-global python3 tests/ik2_prefix_cache.py --engine ./build-ik2/tintero \
    --model /weight.gguf --tokens water.force --cancel-prefill
```

All 18 follow-ups across the three settings reused every previously processed token. Each needed 20 new tokens. The nine final replies matched their independent zero-cache controls on prompt length, assistant message, finish reason and generated token count. No cgroup OOM kills occurred. Memory below is the highest device-wide reading sampled after a request. All runs used 1,024-token prefill chunks. The two-, three- and four-slot servers included the CPU vision encoder. The one-slot row uses the timings from the growing conversation and the device reading from the full-context CPU vision profile.

| Slots | Context per slot | Final prompt per chat | Warm follow-up, seconds | Final cold control, seconds | Device VRAM, GiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 393,216 | 300,000 | See growing conversation | 259.365 | 23.08 |
| 2 | 196,608 | 196,314 | 0.279 to 0.327 | 129.967 to 129.974 | 23.24 |
| 3 | 131,072 | 24,538 | 0.136 to 0.181 | 8.011 to 8.023 | 23.40 |
| 4 | 98,304 | 97,962 | 0.194 to 0.307 | 46.122 to 46.177 | 23.56 |

The 196K two-slot run used `202ace2`. The three- and four-slot runs used the deployed binary, which also passed the two-slot fork test below. `GET /health` reports the configured cache count and per-conversation context. The decoding queue admits one request at a time.

For two, three and four slots the engine allocated 3,235.7, 2,211.7 and 1,699.7 MiB per cache plus a shared 4 MiB exchange buffer. Each conversation needs its own sliding-window cache even though the global context budget is divided among the slots. The highest anonymous RAM readings in those runs were 1.96, 1.82 and 2.04 GiB.

Two image chats had identical text and different pictures. Each first request read 65 tokens from scratch. On alternating follow-ups both reused 66 of the 66 available tokens, read 19 more and correctly identified the other half of their own image. The first parcel fixture had instructed the model to keep a code, and it declined to reveal it on its initial zero-cache request. The test now uses the parcel-register prompt from the single-chat run and records a cold control for any wrong answer.

From the Prometheus build directory, start a fresh instance with the indicated slot count, then run the corresponding line.

```sh
pid=$(docker inspect -f '{{.State.Pid}}' maya-ik2-slots-2)
python3 tests/ik2_slot_chats.py http://127.0.0.1:18085 --server-pid "$pid" --slots 2 --depth 196352
pid=$(docker inspect -f '{{.State.Pid}}' maya-ik2-slots-3)
python3 tests/ik2_slot_chats.py http://127.0.0.1:18085 --server-pid "$pid" --slots 3 --depth 24576
python3 tests/ik2_slot_chats.py http://127.0.0.1:18085 --server-pid "$pid" --slots 3 --depth 768 --images
pid=$(docker inspect -f '{{.State.Pid}}' maya-ik2-slots-4)
python3 tests/ik2_slot_chats.py http://127.0.0.1:18085 --server-pid "$pid" --slots 4 --depth 98000
```

A separate native test forked one shared history at 1,792 tokens, then resumed both descendants. The first implementation restored only 1,792 of the first descendant's 1,827 live tokens. The engine now preserves that descendant when it restores a checkpoint into an unused or older slot. Both descendants reused all 1,827 tokens and matched eight cold predictions apiece, with plain decode and with the official DSpark drafter. The first cold-control fixture accidentally shared one token with its eviction prompt. The regression now requires zero cached tokens before comparing predictions.

```sh
radeon python3 tests/ik2_prefix_cache.py --engine ./cache-slots-fork --model /weight.gguf --tokens water.force --fork-slots
radeon env IK2_DRAFT_ADAPT=0 IK2_DRAFT_WIDTH=2 python3 tests/ik2_prefix_cache.py --engine ./cache-slots-fork \
    --model /weight.gguf --tokens water.force --fork-slots --draft mtp.safetensors
```

Median single-slot decode before and after introducing slots measured 116.89 and 117.00 tok/s after 512 tokens. At 8,192 tokens both measured 107.56 tok/s. Each median is from three independent greedy requests with 128 timed decode steps.

```sh
for exe in full-direct-stop cache-slots-api; do
    radeon python3 tools/ik2_bench.py --engine "./$exe" --model /weight.gguf --depths 512,8192 \
        --ctx 393216 --chunk 1024 --gen 129 --repeats 3 --tokens water.force
done
```


#### Concurrent decode measurements
On Prometheus, the unmodified `dd139eea` server and the batch server ran the same IQ4_NL weights with FP8 KV, medium reasoning and greedy decoding. Each HTTP request had a short prompt and generated exactly 512 tokens. The client sent one simultaneous group at each concurrency, in the order shown. Aggregate tok/s divides all generated tokens by wall time from the first request until the last response. The last two columns are the median of each request's tokens divided by its own elapsed time, including queueing and prefill. Both servers had a 4,096-token per-request limit and a 256-token prefill chunk. The batch server reserved all 32 slots throughout.

| Concurrent requests | Serial aggregate tok/s | Batch aggregate tok/s | Serial median request tok/s | Batch median request tok/s |
|---:|---:|---:|---:|---:|
| 1 | 97.43 | 84.30 | 97.43 | 84.30 |
| 4 | 115.22 | 188.41 | 48.15 | 47.20 |
| 8 | 115.98 | 247.14 | 26.07 | 31.02 |
| 16 | 116.11 | 294.61 | 13.70 | 18.54 |
| 32 | 116.14 | 327.84 | 7.04 | 10.36 |

At concurrency 32 the client received 16,384 tokens in 49.975 s, compared with 141.067 s on the serial server. The single-request row was the first call after each server started and includes first-use costs. At this memory budget the serial server retained 868 MiB of FP16 prefill projections. The batch server used per-call dequantization. Changes in the active slot list, attention-depth buckets or sampling mode can require graph recapture.

Across 32 different chat prompts of 33 to 47 tokens, the new binary with batching off matched all 4,096 original greedy token IDs. With 32 slots enabled, concurrency 1, 8 and 32 each matched the same 4,096 IDs, including all 128 tokens of every reply. Cancelling a request and reusing its slot preserved the other replies. Eight interleaved greedy and seeded sampling requests also reproduced their serial outputs. With a synthetic residual-projection direction enabled, four requests at prompt depths 39, 1,008, 1,269 and 1,357 matched all 128 serial predictions across the attention bucket change and sliding-ring wrap.

The original server and the batch server each answered 32/32 GSM8K questions correctly with a 2,048-token limit. All final answers and finish reasons matched. An eight-worker GSM8K run also scored 32/32 while tool calls ran alongside it. OpenAI and Anthropic tool conversations passed with and without streaming in all three runs.

```sh
python3 tests/ik2_batch_decode.py --baseline /path/to/original/maya-ik2 --engine build-ik2/tintero \
    --model models/Inkling-2-Mini-IQ4_NL.gguf --tokenizer ik2/tokenizer \
    --preset docs/inkling2-mini/ik2-mini-gfx1100.json
python3 tools/ik2_batch_bench.py --url http://127.0.0.1:8088
python3 tests/ik2_gsm8k.py --chat-url http://127.0.0.1:8088 --data gsm8k-test.jsonl \
    --count 32 --concurrency 8 --max-tokens 2048
python3 tests/ik2_tool_conversations.py http://127.0.0.1:8088
```

### Multi-turn tool calls on Prometheus

Run `python3 tests/ik2_tool_conversations.py http://127.0.0.1:PORT` against each listening server. The packaged gfx1100 Maya image (`gfx1100-08aad41`) and the incumbent llama.cpp image (`server-rocm-inkling-1427-r2`, port 8085) each passed all four conversations. On both the OpenAI and Anthropic APIs, in streaming and non-streaming mode, the first assistant turn called the same tool twice with distinct IDs and desk arguments. The client returned results in reverse order. Every final answer associated Larch with VX-731 and Birch with PK-284. The llama.cpp answers sometimes added a heading. Raw responses are in `maya-tools.jsonl` and `llama-tools.jsonl` in the Prometheus build directory.

### Image encoder on the Ryzen 5 3600

The installed `mmproj-F16.gguf` works with the node's llama.cpp `b12600` source. The older ggml tree used for the language engine predates the Inkling vision tower. From the build directory on Prometheus, copy the needed source from the existing checkout, then build in the CPU container.

```sh
mkdir -p vision-llama-b12600
git -C ../llama.cpp-b12600 archive b12600 -- CMakeLists.txt cmake src include common ggml tools vendor LICENSE convert_hf_to_gguf.py | tar -x -C vision-llama-b12600
docker exec -w /work maya-ik2-hip-build cmake -S tools/vision -B vision-cpu -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=/work/vision-llama-b12600
docker exec -w /work maya-ik2-hip-build cmake --build vision-cpu --target maya-vision -j4
python3 tests/ik2_image_answers.py --write-images vision-fixtures
docker run --rm -i --memory 2304m --memory-swap 2304m --cpus 4 -v "$PWD:/work" \
    -v "$HOME/ai/models/inkling-2-mini:/models:ro" --entrypoint /work/vision-cpu/bin/maya-vision \
    ghcr.io/ggml-org/llama.cpp:server-rocm-b11176 --threads 4 \
    --model /models/Inkling-2-Mini-IQ4_NL.gguf --mmproj /models/mmproj-F16.gguf <<'ENCODER'
ENC /work/vision-fixtures/red-blue.png /work/vision-fixtures/cpu-red-blue.sve
ENC /work/vision-fixtures/blue-red.png /work/vision-fixtures/cpu-blue-red.sve
QUIT
ENCODER
```

The encoder returned 35 embeddings of width 2,304 for each 256×192 image in 376 and 377 ms. This command ran without GPU devices. The host used four threads.

#### Image answers and GPU memory

CPU and HIP mtmd and the incumbent llama.cpp server all identified the three left-hand colors after swapping the images under an identical prompt. All three read **39%** from the repository's setup screenshot. The CPU profile used four threads and the full 393,216-token context. Wall times include encoding, prefill and generation. Each row contains one request per server.

| Image | CPU encoder + Maya | HIP encoder + Maya, 393,216 context | llama.cpp |
|---|---:|---:|---:|
| Red/blue | 1.232 s | 1.716 s | 1.508 s |
| Blue/red | 0.497 s | 0.186 s | 1.140 s |
| Green/blue | 0.498 s | 0.166 s | 1.129 s |
| Setup screenshot | 14.058 s | 0.820 s | 2.131 s |

The resident HIP encoder used 984 MiB above its 36 MiB device baseline after warming up at 1,024 image tokens. With 1,024-token prefill chunks, the engine passed at 360,448 positions. It could not keep every expert resident at 376,832 or 393,216. A 512-token chunk also failed at 393,216. **The packaged GPU-vision profile uses 256-token chunks and keeps all 393,216 positions.** The card reported 23.64 GiB used after the image requests. The CPU profile keeps 1,024-token chunks and leaves the GPU to the language model.

Build the encoder in the same CPU-only build container as the CPU version. ROCm's static host objects need PIC on this distribution. The original link failed with `R_X86_64_32 ... can not be used when making a PIE object`. The HIP target now builds with `-fPIC`.

```sh
docker exec -w /work maya-ik2-hip-build cmake -S tools/vision -B vision-hip -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=/work/vision-llama-b12600 -DSTRATA_VISION_HIP=ON \
    -DSTRATA_VISION_HIP_TARGETS=gfx1100 -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_HIP_NO_VMM=ON
docker exec -w /work maya-ik2-hip-build cmake --build vision-hip --target maya-vision -j4
# From the Prometheus build directory. The footprint measurement runs with the GPU idle.
bash ik-vision-footprint.sh
python3 ik-compare-image-rows.py
python3 tests/ik2_image_answers.py http://127.0.0.1:18085
python3 tests/ik2_image_answers.py http://127.0.0.1:8085
```

Comparing all 80,640 embedding values for each color-swap fixture gave CPU/HIP cosine similarities of 0.999370 and 0.999403. Pairing each CPU image with the opposite GPU image gave 0.913765 and 0.912137.

#### Image latency on port 8085

The three screenshot fixtures are 1,480 × 900 pixels and have different image hashes. On a fresh one-slot service (`gfx1100-service-r2`) with a resident HIP encoder, the first screenshot returned 39% in 1.394 s. The next two returned 43% in 0.903 s and 55% in 0.807 s. The CPU service took 13.813 and 13.928 s on the latter two. Its 0.323 s reply to the first used an image embedding retained from the earlier run. Text-prefix cache hits were zero in all six requests.

Reducing the prefill chunk to 256 had a cost for text. In the same HTTP test with 8,207 actual prompt tokens, median cold prefill fell from 3,251.3 tok/s with CPU vision to 1,804.5 with resident HIP vision. Decode measured 109.1 and 107.8 tok/s, respectively. The prompt corpus, chat template, GGUF, generation cap and context reservation were identical. Each median covers three requests with zero cached tokens after a warm-up.

```sh
python3 tests/ik2_image_answers.py http://127.0.0.1:8085 --screenshots
python3 tools/strata/bench_http.py --url http://127.0.0.1:8085 --api openai \
    --tokenizer ik2/tokenizer --ctx 641,8321 --prompt-lens 8192 --gen 129 --runs 3 --warmup 1
```

An OpenBLAS CPU experiment encoded the same three images in 27.993, 27.719 and 27.880 s with four threads. The original CPU kernel took 13.855, 13.873 and 14.135 s. These are encoder times before any language-model work. The services keep the original CPU encoder. Run `bash ik-vision-blas-check.sh` in the Prometheus build directory.

The r3 encoder keeps the FP16 projector weights in host memory. mtmd's scheduler transfers weights for HIP operations when reading a picture. At the same 1,024-image-token limit it measured 404 MiB of additional GPU memory, down from 984 MiB. On the three screenshots, its embeddings had cosine similarities of 0.99872, 0.99876 and 0.99819 against CPU, and at least 0.99970 against the resident HIP encoder. Each comparison covered all 1,960,704 values. Run `bash ik-measure-host-vision.sh` with the GPU otherwise idle and `python3 tests/ik2_vision_compare.py vision-fixtures/cpu-compile.sve vision-fixtures/host-compile.sve --min-cosine 0.998` from the Prometheus build directory.

Four slots needed a 512-token prefill chunk to fit every expert and the encoder. All four cold controls at 97,962 tokens matched their warm replies. Eight alternating follow-ups read 20 new tokens each in 0.197–0.345 s. Cold replays took 51.015–51.114 s. The three screenshots took 1.403, 0.935 and 0.752 s. Device-wide VRAM peaked at 23.67 GiB in the post-request samples, with zero OOM kills. Two slots loaded at 196,608 positions each with 1,024-token chunks and answered the same screenshots in 1.116, 0.916 and 0.662 s. The full two-slot replay was stopped. Three slots use the four-slot chunk size; the earlier three-slot cold controls used CPU vision. On the deployed single-slot profile, the device reported 23.50 GiB after the image checks.

```sh
python3 tests/ik2_image_answers.py http://127.0.0.1:8085 --screenshots
# The four-slot result used a separate listener in the same build directory.
python3 tests/ik2_slot_chats.py http://127.0.0.1:18085 --slots 4 --depth 98000 --images --server-pid "$pid"
```

### Full-model measurements at 393,216 reserved tokens

The RX 7900 XTX ran commit `9a0f164` with the installed ggml-org IQ4_NL GGUF, a 1,024-token prefill chunk and E4M3 KV. Every expert stayed on the card. The runner alternated prompt starts, checked that zero input tokens were reused and timed 128 decode steps after the first prediction. Each cell is the median of three requests. Startup is excluded.

| Input tokens | Maya prefill, tok/s | Maya decode, tok/s |
| ---: | ---: | ---: |
| 32 | 438.36 | 119.18 |
| 512 | 2,438.10 | 117.11 |
| 2,048 | 3,155.62 | 114.18 |
| 8,192 | 3,365.65 | 107.47 |
| 32,768 | 2,953.14 | 93.70 |

The table cycles the water-passage token IDs. Using contiguous archive text instead measured 116.79 and 107.56 tok/s decode at 512 and 8,192 inputs. On the water passage, temperature 0.7 and top-p 0.95 measured 115.73 and 106.40 tok/s.

```sh
# From the Prometheus build directory inside the ROCm container.
radeon() {
    python3 -c 'import json, os, sys
cfg = json.load(open("docs/inkling2-mini/ik2-mini-gfx1100.json"))
os.execvpe(sys.argv[1], sys.argv[1:], {**os.environ, **cfg["env"]})' "$@"
}
bench() {
    radeon python3 tools/ik2_bench.py --engine ./build-ik2/tintero --model /weight.gguf \
        --ctx 393216 --chunk 1024 --gen 129 --repeats 3 "$@"
}
bench --tokens water.force --depths 32,512,2048,8192,32768
bench --tokens water.force --depths 512,8192 --temperature 0.7 --top-p 0.95
python3 tests/ik2_retrieval.py --tokenizer ik2/tokenizer --data gsm8k-test.jsonl --depths 32768 >needle-32768.request
python3 -c 'from pathlib import Path; ids=Path("needle-32768.request").read_text().splitlines()[0].split()[-1].split(","); Path("natural-8k.ids").write_text(",".join(ids[:8193]))'
bench --tokens natural-8k.ids --depths 512,8192
```

On the first 192 GSM8K test questions, this build scored **184/192** with a 4,096-token generation cap. One response reached the cap. The earlier Radeon build scored 187/192. The current build answered **207/256** multiple-choice questions correctly across eight subjects, and all 256 answers were valid letters. With a shared 256-token archive, it answered 24/32 and reused the full prefix on all 32 questions. The question sets and preparation commands are in the [GSM8K checks](#gsm8k) and [multiple-choice checks](#direct-multiple-choice-answers).

```sh
for offset in 0 96; do
    args="--model /weight.gguf --data gsm8k-test.jsonl --count 96 --offset $offset --max-tokens 4096"
    oracle-env/bin/python tests/ik2_gsm8k.py $args >"gsm-$offset.requests"
    radeon ./build-ik2/tintero --serve-ik2 --model /weight.gguf --ctx 8192 --chunk 1024 \
        <"gsm-$offset.requests" >"gsm-$offset.log"
    oracle-env/bin/python tests/ik2_gsm8k.py $args --score "gsm-$offset.log"
done
quiz() {
    oracle-env/bin/python tests/ik2_multiple_choice.py --data mmlu256.jsonl --tokenizer ik2/tokenizer "$@"
}
quiz | radeon ./build-ik2/tintero --serve-ik2 --model /weight.gguf --ctx 4096 --chunk 512 >mmlu.log
quiz --score mmlu.log
```

#### Sliding-window KV precision

`IK2_KV=fp8-global` keeps the sliding layers' keys and values in FP16 and the global layers' cache in E4M3. At 393,216 reserved positions, the engine reported 6.31 GiB of KV in this mode and 6.16 GiB with E4M3 throughout. All 3,072 expert slots remained on the card in both runs.

| Input tokens | Mixed KV prefill, tok/s | Mixed KV decode, tok/s |
| ---: | ---: | ---: |
| 32 | 432.43 | 119.40 |
| 512 | 2,023.72 | 117.65 |
| 2,048 | 2,824.83 | 115.11 |
| 8,192 | 3,258.55 | 107.20 |
| 32,768 | 2,956.60 | 94.67 |

On the same 192 GSM8K questions the mixed cache answered **188/192** correctly. All responses finished within the 4,096-token limit. The four changed scores were all gains. On MMLU the mixed cache scored **206/256** with all letters valid. Two answers changed from wrong to right and three from right to wrong. The all-E4M3 preset had the higher measured prefill rate at 512 and 2,048 tokens.

Run the throughput command below. To repeat the quality comparison, use `radeon env IK2_KV=fp8-global ./build-ik2/tintero` for both engine invocations in the preceding GSM8K and MMLU commands.

```sh
radeon env IK2_KV=fp8-global python3 tools/ik2_bench.py --engine ./build-ik2/tintero \
    --model /weight.gguf --ctx 393216 --chunk 1024 --tokens water.force \
    --depths 32,512,2048,8192,32768 --gen 129 --repeats 3
```

#### Agreement with the CPU reference

The engine and `ref/inkling2.py` read the same IQ4_NL weights and the [same 256 held-out token IDs](#held-out-distribution-fixture). We compared all 200,058 vocabulary entries at each position. The prefill used four 64-token chunks. A separate graph-decode run started at 224 input tokens and consumed the final 32 known tokens. The two sets of positions overlap.

| KV precision | Prefill top-1 | Mean prefill KL | Prefill perplexity on 255 labels | Decode top-1 | Mean decode KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| E4M3 | 250/256 | 0.002041 | 3.96780 | 30/32 | 0.002114 |
| FP16 sliding, E4M3 global | 252/256 | 0.001504 | 3.98305 | 30/32 | 0.002062 |
| FP16 | 256/256 | 0.000825 | 3.97048 | 32/32 | 0.000813 |

The CPU reference's perplexity was 3.96770 on those 255 labels. With FP16 throughout, a separate 8,321-position reservation measured 118.74 tok/s decode after 512 inputs and 108.38 after 8,192. Reserving 393,216 positions would roughly double the global KV memory relative to E4M3.

```sh
# Create heldout-teacher.{all,prompt,force} and heldout-teacher-ref.npy using the fixture command.
for kv in fp8 fp8-global f16; do
    radeon env IK2_KV="$kv" IK2_PREFILL_LOGITS_OUT="$kv-pp.f32" ./build-ik2/tintero \
        --model /weight.gguf --ctx 1024 --chunk 64 --tokens @heldout-teacher.all --gen 0
    radeon env IK2_KV="$kv" IK2_GRAPH=1 IK2_FORCE=heldout-teacher.force IK2_LOGITS_OUT="$kv-tg.f32" \
        ./build-ik2/tintero --model /weight.gguf --ctx 1024 --chunk 128 --tokens @heldout-teacher.prompt --gen 33
    oracle-env/bin/python tools/ik2_compare_decode.py "$kv-pp.f32" --ref heldout-teacher-ref.npy \
        --tokens heldout-teacher.all --vocab 200058 --prefill --prompt-length 256
    oracle-env/bin/python tools/ik2_compare_decode.py "$kv-tg.f32" --ref heldout-teacher-ref.npy \
        --tokens heldout-teacher.all --vocab 200058 --gen 33 --prompt-length 224
done
radeon env IK2_KV=f16 python3 tools/ik2_bench.py --engine ./build-ik2/tintero \
    --model /weight.gguf --chunk 1024 --tokens water.force --depths 512,8192 --gen 129 --repeats 3
```

#### Retrieval at 392,960 input tokens

The all-E4M3 preset returned the exact four-word phrase on all three direct-answer prompts. Each phrase had a different position in the archive, and every prefill started with zero cached tokens. A second request extended the same input by one token, reused all 392,960 previous tokens and timed 128 decode steps with stopping disabled.

| Phrase position | Exact answer | Cold prefill, s | Decode, tok/s |
| ---: | --- | ---: | ---: |
| 5% | velvet falcon tundra juniper | 415.490 | 35.86 |
| 50% | falcon juniper quartz ember | 415.744 | 35.81 |
| 95% | ember meadow lantern harbor | 415.985 | 35.88 |

```sh
for position in 5 50 95; do
    needle() {
        python3 tests/ik2_retrieval.py --tokenizer ik2/tokenizer --data gsm8k-test.jsonl \
            --depths 392960 --percent "$position" --direct-answer --max-tokens 200 --decode-steps 128 "$@"
    }
    needle | radeon ./build-ik2/tintero --serve-ik2 --model /weight.gguf --ctx 393216 --chunk 1024 >"needle-$position.log"
    needle --score "needle-$position.log"
done
```

#### Completed archive and drafter runs

With 392,576 background tokens the paged E4M3 server answered 23/32 MMLU questions correctly. All 32 answers were valid and all 32 requests reused the entire archive. The 256-token control scored 24/32. The first archive length of 392,704 left too little room for question 15. The CPU preflight now checks the complete tokenized question and its eight output tokens. Generate and score the requests with `tests/ik2_multiple_choice.py --max-context 393216 --prefix-tokens 392576 --background gsm8k-test.jsonl --data mmlu32.jsonl --tokenizer ik2/tokenizer`. Feed them to `radeon env IK2_PAGED_KV=1 ./build-ik2/tintero --serve-ik2 --model /weight.gguf --ctx 393216 --chunk 1024`. Add `--score SERVER.log --min-cached 392576` to the generator for scoring.

The paged cache returned 986 on the three-record inventory after 392,192 tokens and the exact midpoint phrase after 392,960. It measured 45.12 tok/s on the 128-step midpoint replay. The accuracy checks support retaining the 393,216-position ceiling.

The official Thinking Machines drafter at revision `b85c3cfa615fc413cc9af3379f491b963f3b586a` has SHA-256 `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`. On the released engine, adaptive drafting raised median decode from 117.22 to 137.49 tok/s after 512 tokens and from 107.47 to 127.49 after 8,192. For a 257-token story, a single run fell from 116.26 to 89.51 tok/s. The packaged server uses plain decode. The drafter remains available through `--draft`.

```sh
for drafter in plain adaptive; do
    args=()
    if test "$drafter" = adaptive; then args=(--draft mtp.safetensors); fi
    radeon env IK2_DRAFT_CTX=8192 python3 tools/ik2_bench.py --engine ./gfx1100-9a0f164 --model /weight.gguf \
        --ctx 32768 --chunk 1024 --gen 129 --repeats 3 --tokens water.force --depths 512,8192 "${args[@]}"
done
```


### Prometheus GPU validation

The installed ggml-org IQ4_NL GGUF can be mounted read-only at `/weight.gguf`. It stores 24 physical layers under `inkling2_loop`. The three passes over layers 6–17 produce 48 visits. Each core layer stores its four message queries in q, k, v, FFN order. Twenty layers use IQ4_NL experts and four use Q8_0. Both the engine and CPU oracle read this layout directly.

The gfx1100 run of `build-ik2/ik2_kernels_test` passed, including the new IQ4_NL comparisons. HIP FP contraction across statements changed the MoE convolution sum. The kernel target now contracts only within a source expression. KV unpacking preserves the E4M3 sign bit through rescaling, including negative zero. The tests exercise scalar and packed reads at inverse scales 0.5, 1 and 2. The CUDA-only MMA and flash tests are excluded from this build.

The five-token prompt below is “The capital of France is”. On the three forced positions the scalar graph and eager logits agreed exactly by the comparator. The float64 reference and FP8-KV engine picked the same top token at all three positions, with mean KL 0.00443123 and max KL 0.00697867. This short check exercises the loop wiring and token alignment.

```sh
cd /work
export IK2_HOST=mmap IK2_KV=fp8 IK2_GATE_BF16=1 IK2_GATE_TILE=64 IK2_Q8_UNROLL=1 IK2_BATCH_PREFILL=1
printf '%s\n' '152717,11,149880' > paris-force.csv
for graph in 0 1; do
  IK2_GRAPH=$graph IK2_FORCE=paris-force.csv IK2_LOGITS_OUT=paris-$graph.f32 \
    build-ik2/tintero --model /weight.gguf --ctx 1024 --chunk 128 --tokens 371,151451,149879,151194,149884 --gen 4
done
oracle-env/bin/python ref/inkling2.py /weight.gguf --tokens 371,151451,149879,151194,149884,152717,11,149880 --logprobs-out paris-ref.npy
oracle-env/bin/python tools/ik2_compare_decode.py paris-1.f32 --baseline paris-0.f32 --prompt-length 5 --gen 4 --vocab 200058
oracle-env/bin/python tools/ik2_compare_decode.py paris-0.f32 --ref paris-ref.npy --prompt-length 5 --gen 4 --vocab 200058
```

#### Expert decode baseline

With the environment above and graph replay enabled, `build-ik2/tintero --model /weight.gguf --ctx 1024 --chunk 128 --tokens 371,151451,149879,151194,149884 --gen 65` measured 32.65 tok/s for 64 scalar decode steps. The prompt predicted the first generated token. On the same prompt, `IK2_GRAPH=0 rocprofv3 --kernel-trace --stats -f csv -d /work/profile-scalar -- /work/scalar-baseline --model /weight.gguf --ctx 1024 --chunk 128 --tokens 371,151451,149879,151194,149884 --gen 33` attributed 43.83% of the traced kernel time to single-token gate/up and 21.88% to down projections.

The vectorized IQ4_NL path reads four packed bytes per lane and uses gfx1100 byte-permute instructions to decode eight weights. Input activations and dot accumulators remain FP32. The GPU kernel suite passed. For a text check, tokenize “Water freezes at” and the continuation “ zero degrees Celsius and boils at one hundred degrees Celsius at sea level. At higher altitudes the boiling point falls because atmospheric pressure is lower.” with `tools/strata_tokenizer.py`. On this GGUF they yield 4 prompt and 30 continuation tokens. Force the continuation with `--gen 31`, export logits and run the oracle on all 34 tokens. `oracle-env/bin/python tools/ik2_compare_decode.py v1-water.f32 --ref water-ref.npy --prompt-length 4 --gen 31 --vocab 200058` measured 30/30 top-1, mean KL 0.0018280142 and max KL 0.0081613966.

#### RDNA3 expert and attention kernels

`IK2_EXPERT_LIMB=60` uses signed integer dot products for IQ4_NL and Q8_0 experts during decode. Each group of sixteen activations shares a scale and uses two signed bytes per value. `IK2_BATCH_PREFILL=1 IK2_PREFILL_WMMA=1` groups routed tokens for native 16×16×16 FP16 matrix instructions with FP32 accumulation. `IK2_ATTN_MMA=1` enables WMMA decode attention on every visit, and `2` selects full-attention visits only. `IK2_FLASH=1` enables WMMA prefill attention. Both attention kernels accept FP16 and software-decoded E4M3 KV, including ring buffers, relative bias and the sink. At this stage HIP still rejected the CUDA block verifier. A later revision added native RDNA3 block attention.

These are plain greedy rates, using three uncached requests per depth in one resident process. The server predicts the first output token during prefill. Each decode timing covers the following 128 tokens. For this measurement, attention used the HIP scalar decode and hipBLAS prefill paths. The reported 64 tok/s for the existing llama.cpp service came from a separate run.

| Prompt tokens | Decode tok/s, three runs | Uncached prefill tok/s, three runs |
| ---: | --- | --- |
| 32 | 95.10, 95.10, 96.24 | 12.90, 48.71, 49.00 |
| 512 | 91.69, 91.69, 91.56 | 890.43, 888.89, 890.43 |
| 2,048 | 87.55, 87.55, 87.55 | 818.87, 825.14, 819.86 |

Run inside the ROCm container, with the installed IQ4_NL GGUF mounted at `/weight.gguf`. The benchmark alternates the first prompt token, checks the server's cached-token count, and requests greedy decoding without a stop token. The water continuation described above supplies these input IDs.

```sh
cd /work
echo '151712,156358,179692,149880,167442,82,149922,150011,156258,156358,179692,149922,151333,150628,13,150847,151197,178091,82,149876,167442,279,150279,159383,150301,186867,152023,149884,151214,13' > water.force
IK2_HOST=mmap IK2_KV=fp8 IK2_GATE_BF16=1 IK2_GATE_TILE=64 IK2_Q8_UNROLL=1 \
IK2_BATCH_PREFILL=1 IK2_PREFILL_WMMA=1 IK2_EXPERT_LIMB=60 IK2_ATTN_MMA=0 IK2_FLASH=0 \
python3 tools/ik2_bench.py --engine ./build-ik2/tintero --model /weight.gguf \
    --tokens water.force --depths 32,512,2048 --gen 129 --repeats 3 --chunk 128
```

The four full-vocabulary checks below used the same WMMA attention and integer-dot decode configuration. Each row compares every forced step with the independent float64 forward, including the prediction after the final forced token. The three chat checks used terminal token `<|content_model_end_sampling|>` and the normal chat template at effort `none`. They returned Paris, `12 - 3 - 4 = 5 apples`, and an explanation of ice's lower density and open hydrogen-bonded lattice.

| Passage | Prompt | Forced positions | Top-1 matches | Mean KL | Max KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Water | 4 | 30 | 30 | 0.00147747 | 0.00451225 |
| Fibonacci | 8 | 33 | 33 | 0.00048531 | 0.00675873 |
| Apples | 7 | 33 | 33 | 0.00152825 | 0.00606835 |
| Library | 114 | 13 | 11 | 0.01083507 | 0.05335118 |

For the library check, repeat “The town library keeps its reading room open on Saturdays. Visitors can borrow a book for three weeks and renew it online. ” four times and append “A visitor borrows a book on Saturday.” Force the continuation “ When is the book due? It is due three weeks later.” Starting from the benchmark environment above, enable `IK2_FLASH=1 IK2_ATTN_MMA=1`. Run the CLI with `--tokens @long.prompt --gen 14 --ctx 1024 --chunk 128`, `IK2_FORCE=long.force`, and `IK2_LOGITS_OUT=all-flash-long.f32`. Compare using `python3 tools/ik2_compare_decode.py all-flash-long.f32 --ref long-ref.npy --prompt-length 114 --gen 14 --vocab 200058`. The two top-1 changes were at positions 114 and 119. These are distribution checks on four passages.

The Loom bundle uses the original Unsloth quant and an external drafter on Strix Halo. Its report identifies speculative acceptance separately for each context. Here the weights are ggml-org IQ4_NL and the decoder predicts one token per step. The Loom code identifies two relevant operations for the port, grouping routed experts for WMMA and accumulating relative-position attention with an explicit sink. [Loom measurement report](https://huggingface.co/loom-fun/unsloth-Inkling-2-Mini-loom/blob/a78779814f545e254cffa9b346e85f236d54ea7c/REPORT.md)

The upstream Mini pull request describes the CUDA engine this fork ports, including graph replay and the DSpark verifier. The other SM80 pull request targets NVIDIA Ampere with E5M2 KV and tensor parallelism. Prometheus uses one gfx1100 device and the port's tested E4M3 cache. [Mini engine source](https://github.com/alesha-pro/maya-inkling-mini) · [Project Maya #1095](https://github.com/mw00/project-maya/pull/1095) · [SM80 #1](https://github.com/snailbrainx/inkling-2-mini-sm80/pull/1)

The inference optimization playbook studies larger MoE models whose experts spill into system RAM. Its advice on measuring prefix reuse applies to the two servers here. All experts of the installed Mini checkpoint fit in VRAM, which changes the value of its CPU and PCIe strategies. [Inference optimization playbook](https://github.com/alesha-pro/inference-optimization-playbook)

#### FP32 expert decode and packed E4M3 reads

The IQ4_NL decode kernel reads eight GGUF blocks per wave and reuses each FP32 activation across gate and up rows. It also reads the FP16 down input directly from registers after conversion. The 32- and 128-row FP16 projections use four waves per output row. For attention, a pair of E4M3 bytes is expanded to FP16 in two lanes. Scaling the packed FP16 result by 256 preserves E4M3 subnormals. The WMMA prefill kernel stages Q and transposes V in LDS. The device suite checks the packed conversions, signed zero, small dense products, fused SwiGLU, down projections, and both attention formats against numerical references.

The following rates are medians over three uncached prompts, 128 timed greedy decode steps each. A single resident process served all four depths. The scalar attention columns use the same packed KV conversion and FP32 expert inputs. The experiment changed one expert or attention switch per comparison.

| Prompt tokens | Direct IQ4 + scalar attention | Previous IQ4 + scalar attention | Direct IQ4 + WMMA attention | Direct IQ4 uncached prefill |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 105.52 | 101.27 | 99.07 | 323.23 |
| 512 | 101.27 | 97.19 | 88.52 | 1,003.92 |
| 2,048 | 97.12 | 93.23 | 84.04 | 973.85 |
| 8,192 | 92.69 | 89.20 | 80.30 | 949.03 |

All throughput columns are in tokens per second. Run the following from `/work` with `water.force` as above. Set `IK2_IQ4_MV=0` for the previous expert kernel or `IK2_ATTN_MMA=1` for WMMA decode attention.

```sh
export IK2_HOST=mmap IK2_KV=fp8 IK2_GATE_BF16=1 IK2_GATE_TILE=64 IK2_Q8_UNROLL=1
export IK2_BATCH_PREFILL=1 IK2_PREFILL_WMMA=1 IK2_FLASH=1 IK2_Q8_TILE=1
export IK2_EXPERT_LIMB=0 IK2_IQ4_MV=1 IK2_F16_MV=1 IK2_ATTN_MMA=0
python3 tools/ik2_bench.py --engine ./build-ik2/tintero --model /weight.gguf \
    --tokens water.force --depths 32,512,2048,8192 --gen 129 --repeats 3 --chunk 128
```

#### Comparison with the production llama.cpp service

The same `ik2_bench.py` replayed the same input IDs, request order and 129-token output length against the patched llama.cpp HTTP service. Both decoders ran alone on the 7900 XTX. The service retained its deployed 393,216-token context, Q4_0 K/V, batch 2,048 and microbatch 256. Maya reserved 8,321 positions for this run, used E4M3 K/V and a 128-token chunk. The benchmark rejects a cached prompt or incomplete output and divides the generation time by the 128 steps after prefill.

```sh
python3 tools/ik2_bench.py --llama-url http://127.0.0.1:8085 --tokens water.force \
    --depths 32,512,2048,8192 --gen 129 --repeats 3
```

| Prompt | Maya decode | llama.cpp decode | Maya prefill | llama.cpp prefill |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 105.52 | 64.14 | 323.23 | 247.14 |
| 512 | 101.27 | 63.96 | 1,003.92 | 1,304.86 |
| 2,048 | 97.12 | 62.89 | 973.85 | 1,511.03 |
| 8,192 | 92.69 | 61.97 | 949.03 | 1,488.44 |

Medians in tok/s. The wider comparison is one GPU, one stream and one installed GGUF. Each runtime uses its own KV representation and greedy trajectories can diverge.

#### GSM8K

The first 96 GSM8K test questions use the official Inkling chat template with reasoning effort `none`, greedy sampling, and up to 1,024 generated tokens. The scorer reads the final content channel and compares its last number to the labeled answer. Missing final answers count as incorrect. With the FP32 expert configuration above, Maya scored **90/96** and the production llama.cpp service scored **93/96**. Both missed questions 12, 40 and 93 (zero-based). Maya reached the output limit before the final answer on questions 41, 86 and 87. The service reached the limit on 12 and 40. Both engines returned a final answer on the other 91 questions. This sample measures arithmetic and instruction following. [OpenAI GSM8K dataset](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data)

```sh
cd /work
curl -fLsS https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl -o gsm8k-test.jsonl
oracle-env/bin/python tests/ik2_gsm8k.py --data gsm8k-test.jsonl --model /weight.gguf --count 96 >gsm96-requests.txt
# Use the environment from the FP32 expert benchmark.
./build-ik2/tintero --serve-ik2 --model /weight.gguf --ctx 2048 --chunk 128 <gsm96-requests.txt >gsm96.log
oracle-env/bin/python tests/ik2_gsm8k.py --data gsm8k-test.jsonl --model /weight.gguf --count 96 --score gsm96.log
oracle-env/bin/python tests/ik2_gsm8k.py --data gsm8k-test.jsonl --model /weight.gguf --count 96 \
    --llama-url http://192.168.100.111:8085
```

At a 4,096-token cap, the cooperative RDNA3 configuration answered **93/96** on questions 0–95 and **94/96** on 96–191. All 192 requests finished. The three misses on the first set were questions 12, 40 and 93. On the second set they were 100 and 119. Across the two runs, generation averaged 108.41 and 108.15 tok/s respectively, computed as the sum of generated tokens after the first token divided by the sum of decode seconds. Neither run reused a prompt prefix. These raw token requests let the model emit a reasoning channel.

```sh
# Use the cooperative prefill environment below with IK2_EXPERT_TILE_LDS=1.
for offset in 0 96; do
    oracle-env/bin/python tests/ik2_gsm8k.py --data gsm8k-test.jsonl --model /weight.gguf \
        --count 96 --offset "$offset" --max-tokens 4096 >"gsm-$offset.requests"
    ./build-ik2/tintero --serve-ik2 --model /weight.gguf --ctx 8192 --chunk 1024 \
        <"gsm-$offset.requests" >"gsm-$offset.log"
    oracle-env/bin/python tests/ik2_gsm8k.py --data gsm8k-test.jsonl --model /weight.gguf \
        --count 96 --offset "$offset" --score "gsm-$offset.log"
done
```

#### Larger prefill chunks and exact router accumulation

On the installed IQ4_NL model, the 56 attention projection matrices occupy 1,228 MiB when dequantized to FP16. `IK2_PREFILL_DENSE=1` retains those matrices when VRAM has room after the expert cache and headroom. Decode continues to read the original planar Q8. The grouped expert kernel expands signed byte pairs and applies the original FP16 block scale. It can reuse one loaded weight tile across 16, 32 or 64 routed tokens. The table uses the 16-token tile.

| Prompt | Chunk | Uncached prefill, both runs (tok/s) | Decode, both runs (tok/s) |
| ---: | ---: | ---: | ---: |
| 512 | 512 | 1,625.40, 1,635.78 | 101.91, 101.99 |
| 8,192 | 512 | 1,485.94, 1,485.94 | 93.36, 93.43 |
| 8,192 | 1,024 | 1,589.14, 1,586.67 | 93.50, 93.50 |
| 8,192 | 2,048 | 1,666.40, 1,663.35 | 93.64, 93.57 |
| 32,768 | 2,048 | 1,339.38, 1,339.82 | 79.06, 79.11 |

The production comparison above used identical tokens at 8,192 and reached 1,488.44 tok/s prefill and 61.97 tok/s decode. The 2,048-token chunk improves prefill, costs workspace and adds startup latency on the first matrix product.

```sh
# Start with the FP32 expert environment above.
export IK2_PREFILL_DENSE=1 IK2_PREFILL_TILE=16 IK2_Q8_MV=1 IK2_F16_MV=2
for chunk in 512 1024; do
    python3 tools/ik2_bench.py --engine ./build-ik2/tintero --model /weight.gguf \
        --tokens water.force --depths 32,512,2048,8192,32768 --gen 129 --repeats 2 --chunk "$chunk"
done
python3 tools/ik2_bench.py --engine ./build-ik2/tintero --model /weight.gguf \
    --tokens water.force --depths 2048,8192,32768 --gen 129 --repeats 2 --chunk 2048
```

`IK2_F16_MV=2` preserves the scalar router's FP32 FMA sequence within each lane and its wave reduction order. The gfx1100 device test compares its output bit for bit to this CPU computation. The library passage with E4M3 KV matched 12/13 CPU top-1 predictions (mean KL 0.02426905, maximum 0.22447712). With FP16 KV and uncompressed loop-gate weights it matched 9/13, with a lower mean KL of 0.00308744 and maximum 0.01372127. Both are full distributions over 200,058 tokens. Reproduce both with the library command above, `--chunk 128`, `IK2_ATTN_MMA=0 IK2_EXPERT_LIMB=0 IK2_IQ4_MV=1 IK2_Q8_MV=1 IK2_F16_MV=2 IK2_PREFILL_DENSE=1`. For the second run additionally set `IK2_KV=f16 IK2_GATE_BF16=0`.

#### Cooperative prefill and depth-dependent decode attention

A 64×64 WMMA projection tile replaced the hipBLAS FP16 products in large prefill batches. Four wave32 groups now share the K and transposed V for 64 query tokens. In decode, eight lanes cooperate on one key for sliding attention and short global attention. Global visits switch to WMMA at 8,192. The engine recaptures the graph when the split count or attention kernel changes. Expert matvecs specialize the 2,304- and 1,152-wide matrices of the installed model.

| Prompt | Maya prefill | Maya decode | Deployed llama.cpp prefill | Deployed llama.cpp decode |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 376.47 | 111.21 | 247.14 | 64.14 |
| 512 | 2,370.37 | 109.50 | 1,304.86 | 63.96 |
| 2,048 | 3,362.89 | 107.29 | 1,511.03 | 62.89 |
| 8,192 | 3,558.64 | 100.63 | 1,488.44 | 61.97 |
| 32,768 | 3,010.93 | 87.67 | | |

All five rows use the fixed asynchronous input transfer. A fast resident prefill could previously overwrite pinned embeddings or positions while their copy was pending. Reproduce the warm vs cold continuation check with `python3 tests/ik2_prefix_cache.py --engine ./build-ik2/tintero --model /weight.gguf --tokens water.force`.

Medians in tok/s for three uncached requests per depth. Each request generates 129 tokens with 128 timed decode steps. Maya reserved 32,897 positions with FP8 KV. The service retained its deployed 393,216 positions with Q4_0 KV. On the first 32-token request the initial matrix-library setup took the prefill rate down to 18.52 tok/s. The other two measured 376.47 tok/s.

```sh
export IK2_HOST=mmap IK2_KV=fp8 IK2_GATE_BF16=1 IK2_GATE_TILE=64 IK2_Q8_UNROLL=1
export IK2_BATCH_PREFILL=1 IK2_PREFILL_WMMA=1 IK2_PREFILL_TILE=64 IK2_EXPERT_TILE_LDS=1 IK2_SHARED_COMBINE=1
export IK2_F16_MV=2 IK2_IQ4_MV=1 IK2_Q8_MV=1 IK2_Q8_TILE=1 IK2_EXPERT_LIMB=0
export IK2_PREFILL_DENSE=1 IK2_DENSE_WMMA=1 IK2_FLASH=1 IK2_FLASH_WAVES=4
export IK2_ATTN_MMA=3 IK2_DECODE_SUBWAVE=1 IK2_ATTN_STAGED=1 IK2_RESIDENT_ONLY=1
python3 tools/ik2_bench.py --engine ./build-ik2/tintero --model /weight.gguf --tokens water.force \
    --depths 32,512,2048,8192,32768 --gen 129 --repeats 3 --chunk 2048
```

The gfx1100 device suite checks WMMA against FP64 on partial tiles and padded outputs. The attention tests cover E4M3 and FP16, sink and relative bias, ring wraparound, causal masking and multi-token block verification. Run `IK2_EXPERT_TILE_LDS=1 ./build-ik2/ik2_kernels_test` with the environment above. Two output rows sharing each Q8 input load slowed decode at every measured depth, so the engine keeps one row per wave.

For long-context retrieval, `tests/ik2_retrieval.py` places a seeded four-word phrase among GSM8K distractor questions at an exact prompt depth. The score records the final content channel, the number of cached tokens and the stop reason. To reproduce the midpoint test, pass `--tokenizer ik2/tokenizer --data gsm8k-test.jsonl --depths 32768,131072,392960`. Feed stdout to a resident server with `--ctx 393216 --chunk 1024`, then run the same Python command with `--score SERVER.log`.

#### Copy and edit requests without a drafter

Maya can propose a continuation from the request's own tokens and check it against the target model. On Prometheus, a 64-token verification window copied an 81-box inventory at **271.15 tok/s** and made a single requested edit at **196.72 tok/s**. Plain decoding produced the same two final answers at 105.23 and 106.21 tok/s. Each figure is one uncached greedy request with a 3,072-token output limit. Timing excludes prefill and the first token. The edit took 1,586 generated tokens with verification and 1,601 without it, including the model's reasoning. These examples measure repeated text.

| Verification window | Verbatim copy | Single edit | Exact final answers |
| ---: | ---: | ---: | ---: |
| Off | 105.23 | 106.21 | 2/2 |
| 8 | 183.02 | 177.24 | 2/2 |
| 64 | 271.15 | 196.72 | 2/2 |

Use the preceding cooperative prefill environment with `IK2_EXPERT_TILE_LDS=1`. The GGUF and tokenizer are the same ones used for the throughput runs.

```sh
oracle-env/bin/python tests/ik2_copy_edit.py --tokenizer ik2/tokenizer >copy.requests
for window in 0 8 64; do
    IK2_SUFFIX=$window ./build-ik2/tintero --serve-ik2 --model /weight.gguf \
        --ctx 4096 --chunk 512 <copy.requests >"copy-$window.log"
    oracle-env/bin/python tests/ik2_copy_edit.py --tokenizer ik2/tokenizer --score "copy-$window.log"
done
```

The direct-answer HTTP template prefills the final content channel. With that same prompt ending, the Python task renames one function in `tests/ik2_compare_decode_test.py` and asks for the complete file. The inventory tasks use the same text as above. The table gives median decode tok/s over three uncached runs. Every final response matched the expected string, and every setting generated the same number of tokens for each task.

| Verification window | Inventory copy, 582 tokens | Inventory edit, 583 tokens | Python rename, 886 tokens |
| ---: | ---: | ---: | ---: |
| Off | 108.40 | 108.36 | 107.83 |
| 64 | 523.42 | 412.18 | 409.91 |
| 128 | 750.65 | 482.59 | 445.84 |

```sh
# Use the preceding cooperative prefill environment with IK2_EXPERT_TILE_LDS=1.
for task in inventory python; do
    for window in 0 64 128; do
        oracle-env/bin/python tests/ik2_copy_edit.py --tokenizer ik2/tokenizer \
            --direct-answer --workload "$task" --repeats 3 >"copy-$task.requests"
        IK2_SUFFIX=$window ./build-ik2/tintero --serve-ik2 --model /weight.gguf \
            --ctx 8192 --chunk 512 <"copy-$task.requests" >"copy-$task-$window.log"
        oracle-env/bin/python tests/ik2_copy_edit.py --tokenizer ik2/tokenizer \
            --direct-answer --workload "$task" --repeats 3 --score "copy-$task-$window.log"
    done
done
```

#### Exact phrase retrieval at long context

In the run before the host input copy fix, with `--ctx 393216 --chunk 1024`, the resident IQ4_NL engine returned the exact four-word phrase at 32,768 and 131,072 tokens, placed halfway through each uncached prompt. At 392,960 input tokens the engine finished prefill in 470.020 seconds and used all 200 generation tokens without reaching its final answer. That request scored incorrect. All 3,072 experts remained on the card.

```sh
oracle-env/bin/python tests/ik2_retrieval.py --tokenizer ik2/tokenizer --data gsm8k-test.jsonl \
    --depths 32768,131072,392960 >retrieval.requests
./build-ik2/tintero --serve-ik2 --model /weight.gguf --ctx 393216 --chunk 1024 <retrieval.requests >retrieval.log
oracle-env/bin/python tests/ik2_retrieval.py --tokenizer ik2/tokenizer --data gsm8k-test.jsonl \
    --depths 32768,131072,392960 --score retrieval.log
```

With the pinned-input fix, Maya again returned the exact midpoint phrase on the two shorter prompts. The same process reserved 393,216 positions and kept all experts on the GPU. Each retrieval started with an empty prefix. A second request extended the prompt by one token and generated 129 tokens with stopping disabled. Its decode time covers the last 128 tokens.

| Input tokens | Expected phrase and actual final answer | Uncached prefill | Tokens before stop | Fixed-length decode |
| ---: | --- | ---: | ---: | ---: |
| 32,768 | falcon saffron cobalt meadow | 14.056 s | 119 | 87.13 tok/s |
| 131,072 | falcon meadow lantern glacier | 78.134 s | 80 | 59.76 tok/s |

These are single runs with token-major E4M3. The longer prompt fully recovered its 131,072-token prefix for the fixed-length request. To reproduce, add `--decode-steps 128` to both Python commands above and use `--depths 32768,131072`. Set `IK2_EXPERT_TILE_LDS=1` and `IK2_SHARED_COMBINE=1` with the cooperative prefill environment.


#### CUDA build check

CUDA 12.8 compiled and linked the engine and its device tests for sm_86 with the following commands. The host used for compilation has no NVIDIA GPU.

```sh
cmake -S . -B build-cuda -G Ninja -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_TESTING=ON
cmake --build build-cuda --target tintero ik2_kernels_test -j4
```

#### Held-out distribution fixture

Use the first 256 tokenizer IDs from GSM8K test questions and solutions 192 through 199, numbered from zero. The first 224 IDs form the decode prompt and the remaining 32 are forced in order. The arithmetic evaluation above used questions 0 through 191.

```sh
PYTHONPATH=src/tools oracle-env/bin/python - <<'PYDATA'
import json
from pathlib import Path
from strata_tokenizer import Tokenizer
rows = [json.loads(line) for line in Path("gsm8k-test.jsonl").read_text().splitlines()]
text = "\n\n".join(r["question"] + "\n" + r["answer"] for r in rows[192:200])
ids = Tokenizer.from_dir("ik2/tokenizer").encode(text)[:256]
for name, seq in [("all", ids), ("prompt", ids[:224]), ("force", ids[224:])]:
    Path("heldout-teacher." + name).write_text(",".join(map(str, seq)))
PYDATA
OPENBLAS_NUM_THREADS=1 oracle-env/bin/python ref/inkling2.py /weight.gguf \
    --tokens "$(cat heldout-teacher.all)" --logprobs-out heldout-teacher-ref.npy
```

#### Global E4M3 pages

`IK2_PAGED_KV=1` groups 16 positions per KV head and transposes V inside each page. Sliding attention retains its ring. A 131,072-token midpoint retrieval with the direct page reader returned the exact same four-word phrase as the token-major layout. The 128-step decode measured 65.07 tok/s with pages and 59.76 with token-major KV. Cold prefill took 101.342 and 78.134 seconds respectively. These are single runs with `--ctx 393216 --chunk 1024` and `--decode-steps 128` in the retrieval command above.

The paged prefill agreed with the FP64 reference on 251 of 256 held-out next-token predictions. Mean KL was 0.0022271 across the full vocabulary. Next-token perplexity on the 255 labels was 3.9611 for Maya and 3.9677 for the reference. The device tests cover page boundaries, partial pages, transposed V, causal masking and the stored K/V bytes.

```sh
export IK2_PAGED_KV=1
./build-ik2/ik2_kernels_test
IK2_PREFILL_LOGITS_OUT=teacher-pages.f32 ./build-ik2/tintero --model /weight.gguf \
    --ctx 1024 --chunk 64 --tokens @heldout-teacher.all --gen 0
oracle-env/bin/python tools/ik2_compare_decode.py teacher-pages.f32 --ref heldout-teacher-ref.npy \
    --tokens heldout-teacher.all --vocab 200058 --prefill --prompt-length 256
```

#### Fused decode projections

The Radeon preset computes the FP16 attention sink and dynamic convolution taps alongside their Q8 projection. For short prompts this removes 16 launches per generated token. Across 32 forced steps the two paths produced byte-identical full-vocabulary logits. With the installed IQ4_NL GGUF, three 256-step greedy measurements gave the following medians in tok/s.

| Input tokens | Separate projections | Fused projections |
| ---: | ---: | ---: |
| 32 | 111.06 | 117.06 |
| 512 | 109.35 | 115.16 |
| 8,192 | 100.95 | 105.74 |

```sh
# Export the cooperative prefill environment above.
for flag in 0 1; do
    IK2_DECODE_PROJ=$flag python3 tools/ik2_bench.py --engine ./build-ik2/tintero --model /weight.gguf \
        --tokens water.force --gen 257 --depths 32,512,8192 --repeats 3 --chunk 2048
    IK2_DECODE_PROJ=$flag IK2_FORCE=heldout-teacher.force IK2_LOGITS_OUT="projection-$flag.f32" ./build-ik2/tintero --model /weight.gguf \
        --ctx 1024 --chunk 128 --tokens @heldout-teacher.prompt --gen 33
done
cmp projection-0.f32 projection-1.f32
```


#### Direct multiple-choice answers

On 256 questions from eight MMLU subjects, the paged E4M3 engine with the cancellation fix answered **207 correctly**. Every response supplied a valid letter. The first 32 test examples from each subject were selected before running either engine. Each chat requested one letter in the final content channel, with greedy decoding and an eight-token limit. The scorer accepts the letter with optional parentheses or a final period. [Pinned CAIS dataset](https://huggingface.co/datasets/cais/mmlu/tree/c30699e8356da336a370243923dbaf21066bb9fe)

Prepare the small JSONL with PyArrow. This command downloads the eight evaluation splits.

```sh
python3 - <<'PYDATA'
import json
from urllib.request import urlopen
import pyarrow as pa
import pyarrow.parquet as pq
subjects = ("astronomy", "college_mathematics", "conceptual_physics", "high_school_european_history",
            "machine_learning", "philosophy", "professional_psychology", "world_religions")
revision = "c30699e8356da336a370243923dbaf21066bb9fe"
with open("mmlu256.jsonl", "w") as out:
    for subject in subjects:
        url = f"https://huggingface.co/datasets/cais/mmlu/resolve/{revision}/{subject}/test-00000-of-00001.parquet"
        with urlopen(url) as response:
            rows = pq.read_table(pa.BufferReader(response.read())).slice(0, 32).to_pylist()
        for i, r in enumerate(rows):
            print(json.dumps(dict(subject=subject, index=i, question=r["question"], choices=r["choices"], answer=r["answer"])), file=out)
PYDATA
```

Use the cooperative prefill environment with fused projections and paged KV. Both engines receive the same token IDs.

```sh
oracle-env/bin/python tests/ik2_multiple_choice.py --data mmlu256.jsonl --tokenizer ik2/tokenizer >mmlu.requests
IK2_DECODE_PROJ=1 IK2_PAGED_KV=1 ./build-ik2/tintero --model /weight.gguf --serve-ik2 --ctx 4096 --chunk 512 \
    <mmlu.requests >mmlu-maya.log
oracle-env/bin/python tests/ik2_multiple_choice.py --data mmlu256.jsonl --tokenizer ik2/tokenizer --score mmlu-maya.log
oracle-env/bin/python tests/ik2_multiple_choice.py --data mmlu256.jsonl --tokenizer ik2/tokenizer --llama-url http://127.0.0.1:8085
```

#### Attention workgroups and KV layout

The Radeon kernel splits the keys according to the number of query workgroups. Each wave covers 16 queries. Larger decode graphs use additional key splits at 32K, 128K and 256K positions. The paged decode kernel reads global KV directly, while prefill shares each page between waves. Four-wave measurements on the installed IQ4_NL model gave the following medians in tok/s. Each depth had three uncached requests and 128 timed decode steps.

| Input tokens | Token-major prefill | Paged prefill | Token-major decode | Paged decode |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 2,393 | 2,327 | 113.98 | 112.87 |
| 8,192 | 3,660 | 3,674 | 104.58 | 102.48 |
| 32,768 | 3,159 | 3,233 | 91.76 | 93.77 |

The Radeon server preset uses token-major KV. The page layout takes the same bytes per token. For the two- and eight-wave boundary cases, the device tests checked results against the CPU attention reference.

```sh
# Use the cooperative prefill environment above.
export IK2_DECODE_PROJ=1 IK2_FLASH_WAVES=4
for page in 0 1; do
    IK2_PAGED_KV=$page python3 tools/ik2_bench.py --engine ./build-ik2/tintero --model /weight.gguf \
        --tokens water.force --depths 512,8192,32768 --gen 129 --repeats 3 --chunk 2048
done
for waves in 2 8; do
    IK2_FLASH_WAVES=$waves IK2_PAGED_KV=1 ./build-ik2/ik2_kernels_test
done
```

#### Looped-layer message gates

On a single decode token, 128 threads now read four input channels each for the two BF16 projections. One rank computes the input norms. The next kernel applies the two output projections to 64 channels per workgroup. The device test checks three simultaneous in-place gates against FP64 across all 6,912 outputs.

With `IK2_GATE_TILE=256`, median decode at 512 tokens rose from 113.93 to **117.38 tok/s**. At 8,192 it rose from 105.05 to **107.61 tok/s**. Across 32 forced held-out steps, both variants agreed with the CPU reference on 30 top tokens. Mean full-vocabulary KL changed from 0.002206 to 0.002220. The Radeon preset enables the new gate.

```sh
# Use the cooperative prefill environment above with IK2_DECODE_PROJ=1.
for tile in 64 256; do
    IK2_GATE_TILE=$tile python3 tools/ik2_bench.py --engine ./build-ik2/tintero --model /weight.gguf \
        --tokens water.force --depths 32,512,8192 --gen 257 --repeats 3 --chunk 2048
    IK2_GATE_TILE=$tile IK2_FORCE=heldout-teacher.force IK2_LOGITS_OUT="gate-$tile.f32" \
        ./build-ik2/tintero --model /weight.gguf --ctx 1024 --chunk 128 --tokens @heldout-teacher.prompt --gen 33
    oracle-env/bin/python tools/ik2_compare_decode.py "gate-$tile.f32" --ref heldout-teacher-ref.npy \
        --tokens heldout-teacher.all --vocab 200058 --gen 33 --prompt-length 224
done
IK2_GATE_TILE=256 ./build-ik2/ik2_kernels_test
```

#### Repeated requests against a shared prefix

The server has eight snapshots of the sliding KV and convolution state. Previously, changing the end of a request invalidated newer snapshots, but the next save could overwrite the shared prefix while empty slots were available. In a replay with a 256-token prefix the ninth branch recomputed all 256 tokens. The server now chooses an empty slot first. All 12 branches reused the full prefix and the last one produced the same four predictions as a cold replay.

```sh
# Use the Radeon preset environment.
python3 tests/ik2_prefix_cache.py --engine ./build-ik2/tintero --model /weight.gguf \
    --tokens water.force --branches 12
```

#### Host RAM on the production comparison server

Prometheus gives the `inkling-2-mini` container 10 GiB of host RAM and the same combined RAM-plus-swap limit. The RX 7900 XTX has a separate 24 GB of VRAM. The service image defaults to a prompt cache of up to 8,192 MiB and 32 context checkpoints per slot. A checkpoint at 229,211 tokens occupied 48.670 MiB in the server log. Thirty-two checkpoints of that size would occupy 1.52 GiB. The crash did not retain an allocator-level breakdown.

The deployed Compose command now sets `--cache-ram 0 --ctx-checkpoints 2`. It keeps one slot, 393,216 context positions, Q4_0 K/V and the 10 GiB cgroup limit. This bounds the two host-side caches while preserving prefix reuse in the active slot. The host kernel also charges file-backed pages to the cgroup, so the resident-memory total includes reclaimable file data.

```sh
docker run --rm --memory 1g --memory-swap 1g --cpus 1 \
    hyperion/llama.cpp:server-rocm-inkling-1427-r2 --help 2>&1 | grep -A3 -E -- '--cache-ram|--ctx-checkpoints'
docker inspect -f '{{.HostConfig.Memory}} {{.HostConfig.MemorySwap}} {{json .Config.Cmd}}' inkling-2-mini
docker logs inkling-2-mini 2>&1 | grep 'erasing old context checkpoint' | tail -1
```

#### Fused expert activation in prefill

An 8,192-token natural prompt spent 809 ms of GPU work in IQ4_NL expert products and 44 ms in the separate SwiGLU kernel. With `IK2_PREFILL_SWIGLU=1`, a wave computes adjacent gate and up tiles and writes the scaled, capped FP16 activation. The device tests compare each stored half to the separate kernels and to an FP64 calculation for IQ4_NL and Q8_0. Across all 256 positions and 200,058 vocabulary entries of the held-out fixture, the fused and separate paths produced identical FP32 logits.

| Input tokens | Separate prefill, tok/s | Fused prefill, tok/s |
| ---: | ---: | ---: |
| 2,048 | 3,419 | 3,501 |
| 8,192 | 3,674 | 3,732 |
| 32,768 | 3,167 | 3,214 |

The table uses three uncached requests per depth, `--chunk 2048` and the installed IQ4_NL GGUF. At 512 tokens the fused run took 1.660, 0.294 and 0.210 seconds. The separate run took 1.664, 0.213 and 0.213 seconds.

```sh
# Export the cooperative prefill environment, IK2_DECODE_PROJ=1 and IK2_GATE_TILE=256.
export IK2_WMMA_K=32
for flag in 0 1; do
    IK2_PREFILL_SWIGLU=$flag IK2_PREFILL_LOGITS_OUT="swiglu-$flag.f32" ./build-ik2/tintero \
        --model /weight.gguf --ctx 1024 --chunk 64 --tokens @heldout-teacher.all --gen 0
    IK2_PREFILL_SWIGLU=$flag python3 tools/ik2_bench.py --engine ./build-ik2/tintero --model /weight.gguf \
        --tokens water.force --depths 512,2048,8192,32768 --gen 129 --repeats 3 --chunk 2048
done
cmp swiglu-0.f32 swiglu-1.f32
for k in 32 64; do IK2_WMMA_K=$k IK2_PREFILL_SWIGLU=1 ./build-ik2/ik2_kernels_test; done
```

#### Q8 attention output projection

The 3,072-input Q8 projection handles eight output rows per workgroup. It reads the original quantized weights and accumulates each row in the same order. A 37-row device test covers the partially filled last workgroup against FP64. For 32 held-out decode steps, the full-vocabulary FP32 logits matched the prior build byte for byte.

With the loop-gate optimization enabled, the median of three 256-step requests changed from 117.38 to 118.14 tok/s at 512 tokens and from 107.61 to 108.61 tok/s at 8,192 tokens. The command uses the same input IDs on each build.

```sh
# Run on the parent of the projection commit, then on the commit itself.
export IK2_GATE_TILE=256 IK2_Q8_UNROLL=1 IK2_DECODE_PROJ=1
python3 tools/ik2_bench.py --engine ./build-ik2/tintero --model /weight.gguf \
    --tokens water.force --depths 32,512,8192 --gen 257 --repeats 3 --chunk 2048
IK2_Q8_UNROLL=1 ./build-ik2/ik2_kernels_test
IK2_FORCE=heldout-teacher.force IK2_LOGITS_OUT=projection.f32 ./build-ik2/tintero \
    --model /weight.gguf --ctx 1024 --chunk 128 --tokens @heldout-teacher.prompt --gen 33
sha256sum projection.f32
```

#### Quality checks with background text

The multiple-choice runner can place the same archive before each question. It sends one warm-up request and checks how much of the prefix each later request actually reused. With 256 archive tokens, the Radeon engine answered 24 of 32 questions correctly and returned a valid letter every time. Each question reused all 256 tokens. The sample contains indices 0 through 3 from each of the eight subjects in the 256-question fixture above.

A separate inventory task places three shipment records at 5%, 50% and 95% of the input. The prompt asks for the two deliveries minus the damaged boxes. At 32,768 tokens the direct-answer run returned 1,008. The expected answer was 986. This task checks whether the model can retrieve and combine the records.

```sh
# Use the Radeon preset environment and the MMLU and GSM8K data above.
python3 -c 'import json; print("".join(s for s in open("mmlu256.jsonl") if json.loads(s)["index"] < 4), end="")' >mmlu32.jsonl
quiz() {
    oracle-env/bin/python tests/ik2_multiple_choice.py --data mmlu32.jsonl \
        --tokenizer ik2/tokenizer --prefix-tokens 256 --background gsm8k-test.jsonl "$@"
}
quiz | ./build-ik2/tintero --serve-ik2 --model /weight.gguf --ctx 4096 --chunk 512 >quiz.log
quiz --min-cached 256 --score quiz.log
ledger() {
    oracle-env/bin/python tests/ik2_retrieval.py --data gsm8k-test.jsonl \
        --tokenizer ik2/tokenizer --depths 32768 --inventory --direct-answer --max-tokens 32 "$@"
}
ledger | IK2_PAGED_KV=1 ./build-ik2/tintero --serve-ik2 --model /weight.gguf --ctx 393216 --chunk 1024 >ledger.log
ledger --score ledger.log
```

#### Larger copy and edit windows

The HIP verifier can check up to 512 proposed inputs per target pass. With a window of at least 32 it computes up to 64 vocabulary rows together for greedy verification. The window is bounded by the prefill chunk size. Plain generation is the default.

| Verification window | Inventory copy, tok/s | Inventory edit, tok/s | Python rename, tok/s |
| ---: | ---: | ---: | ---: |
| 128 | 833.57 | 527.65 | 479.67 |
| 256 | 881.64 | 553.23 | 693.57 |
| 512 | 996.57 | 488.66 | 637.15 |

These are medians of three direct-answer requests per cell. All 27 final answers exactly matched the requested text. On the longer, 27-line inventory, the client sent STOP after reading 750 tokens. The server ended the verified batch after emitting 1,164 and reused 2,939 tokens on the next request. Its eight predictions after resumption matched a cold replay.

```sh
# Use the preceding copy/edit commands with a 512-token chunk and windows 128, 256 and 512.
for task in inventory python; do
    oracle-env/bin/python tests/ik2_copy_edit.py --tokenizer ik2/tokenizer \
        --direct-answer --workload "$task" --repeats 3 >"$task.requests"
    for w in 128 256 512; do
        IK2_SUFFIX=$w ./build-ik2/tintero --serve-ik2 --model /weight.gguf --ctx 8192 \
            --chunk 512 <"$task.requests" >"$task-$w.log"
        oracle-env/bin/python tests/ik2_copy_edit.py --tokenizer ik2/tokenizer \
            --direct-answer --workload "$task" --repeats 3 --score "$task-$w.log"
    done
done
oracle-env/bin/python tests/ik2_copy_edit.py --tokenizer ik2/tokenizer \
    --direct-answer --inventory-lines 27 --max-tokens 2048 >long-copy.requests
IK2_SUFFIX=512 python3 tests/ik2_prefix_cache.py --engine ./build-ik2/tintero --model /weight.gguf \
    --tokens water.force --chunk 512 --cancel-after 750 --cancel-request long-copy.requests
```

#### Precision of split E4M3 attention

Two keys can have exactly equal dot products and opposite values. In a gfx1100 regression test, permuting the entries of one key changed its FP32 reduction order. The scalar and resident kernels returned −0.0000153 and −0.0000305 on a channel whose exact result is zero. The scalar E4M3 path now forms the score in FP64. The eight-lane resident kernel refines a score when its magnitude reaches 16, where FP32 spacing reaches 2⁻¹⁹. Both kernels returned zero on all 128 channels of the device regression. The separate 16,640-key stress case also passed at its original tolerance.

On 32 forced held-out steps, mean full-vocabulary KL was 0.002114 and top-1 agreement was 30/32. Before the score change the figures were 0.002220 and 30/32. With Q8 output projection enabled, median decode after 8,192 tokens was 108.11 tok/s on the corrected build and 108.61 on the prior build. These runs used the short-context reservation and three 256-step repetitions. The corrected speed run used a prefill K tile of 64. Repeating the forced-logit check with a tile of 32 produced the same FP32 file.

```sh
# Export the Radeon preset environment, including IK2_DECODE_SUBWAVE=1 and IK2_GATE_TILE=256.
./build-ik2/ik2_kernels_test
IK2_FORCE=heldout-teacher.force IK2_LOGITS_OUT=attention.f32 ./build-ik2/tintero \
    --model /weight.gguf --ctx 1024 --chunk 128 --tokens @heldout-teacher.prompt --gen 33
oracle-env/bin/python tools/ik2_compare_decode.py attention.f32 --ref heldout-teacher-ref.npy \
    --tokens heldout-teacher.all --vocab 200058 --gen 33 --prompt-length 224
IK2_WMMA_K=64 python3 tools/ik2_bench.py --engine ./build-ik2/tintero --model /weight.gguf \
    --tokens water.force --depths 32,512,8192 --gen 257 --repeats 3 --chunk 2048
```

#### Dense WMMA tile depth

The gfx1100 test suite exercises the 3,072-input by 2,304-output attention projection with 513 tokens. It checks the FP32 output against FP64 at the first and last channels and around tile boundaries. A 64-channel K tile accelerated this projection in isolation. Selecting it only for this shape gave mixed results on uncached natural prompts. The preset keeps the 32-channel depth.

```sh
for k in 32 64; do IK2_WMMA_K=$k ./build-ik2/ik2_kernels_test; done
```

#### Short prompt tiles and inventory control

The Radeon preset chooses the MoE tile height from the number of routed rows per expert. On this model it uses 16 rows through 128 input tokens, 32 rows through 256, and 64 rows above that. The 64-row case uses fused SwiGLU. Median prefill time for 16, 32, 64 and 128 new tokens fell from 63, 88, 101 and 116 ms to 57, 80, 91 and 108 ms. At 256, 512, 1,024 and 2,048 tokens both settings measured 141, 180, 290 and 513 ms. At chunk sizes 64 and 256, all 256 held-out prefill rows matched the fixed-height run byte for byte across the full vocabulary.

```sh
# Use the Radeon preset and the GSM8K test file and tokenizer from the retrieval check.
oracle-env/bin/python tests/ik2_retrieval.py --tokenizer ik2/tokenizer --data gsm8k-test.jsonl \
    --depths 32768 >needle-32768.request
python3 -c 'from pathlib import Path; ids=Path("needle-32768.request").read_text().splitlines()[0].split()[-1].split(","); Path("profile-pp-ids").write_text(",".join(ids[:8192]))'
for rows in auto 64; do
    IK2_WMMA_K=32 IK2_PREFILL_TILE=$rows python3 tools/ik2_bench.py --engine ./build-ik2/tintero \
        --model /weight.gguf --tokens profile-pp-ids --depths 16,32,64,128,256,512,1024,2048 --chunk 2048 --gen 3 --repeats 5
    for chunk in 64 256; do
        IK2_WMMA_K=32 IK2_PREFILL_TILE=$rows IK2_PREFILL_LOGITS_OUT="tile-$chunk-$rows.f32" ./build-ik2/tintero \
            --model /weight.gguf --ctx 1024 --chunk "$chunk" --tokens @heldout-teacher.all --gen 0
    done
done
for chunk in 64 256; do cmp "tile-$chunk-auto.f32" "tile-$chunk-64.f32"; done
```

On the three-record inventory, the direct-answer prompt returned the correct 986 at 256 and 2,048 input tokens and 1,008 at 32,768. Allowing the model to reason before answering produced 986 at all three depths. The model used 355, 169 and 488 generated tokens respectively. Switching all KV entries to FP16 did not recover the 32,768-token direct answer. Run the ledger command above with `--depths 256,2048,32768`, then remove `--direct-answer` and use `--max-tokens 512` for the reasoning control.

#### NVFP4 expert WMMA on RDNA3

The Radeon kernel unpacks GGUF NVFP4 in registers. Each 36-byte block holds 64 E2M1 weights and four unsigned E4M3 scales. Batched prefill can group resident experts and apply the same capped SwiGLU as the IQ4_NL path. For single-token NVFP4 decode, `IK2_EXPERT_MMA=1` enables a WMMA gate/up product with a two-half representation of the FP32 input and a down projection over the FP16 hidden state. The default for single-token NVFP4 on HIP remains scalar.

In the first grouped implementation, adding each matrix tile directly to its WMMA accumulator gave −2.19387 for an FP64 reference of −2.19205. One output of 4,512 exceeded its test tolerance. Adding independently rounded 16-channel dot products in FP32 passed the original test. The complete Radeon suite passed with register and LDS tiles, including tile heights 16, 32 and 64 and both 32- and 64-channel K depths. The single-token test compares all 640 gate/up outputs and 1,280 down outputs to independent FP64 evaluations of the packed bytes. These NVFP4 results use generated expert weights. All full-model performance and quality results for this port use the installed IQ4_NL checkpoint.

```sh
for k in 32 64; do
    IK2_EXPERT_MMA=1 IK2_WMMA_K=$k IK2_EXPERT_TILE_LDS=1 ./build-ik2/ik2_kernels_test
done
IK2_EXPERT_MMA=1 IK2_WMMA_K=32 IK2_EXPERT_TILE_LDS=0 ./build-ik2/ik2_kernels_test
```

#### Resident prefill workspace and projection cache

The grouped expert gate/up and down projections use one FP32 workspace in sequence. SwiGLU places its FP16 output in a separate buffer. With an 8,192-token chunk this saves 576 MiB of expert scratch, enough for the engine to keep all 56 dense attention projections in FP16 under the existing VRAM headroom setting. On 8,192 and 32,768 input tokens, median prefill fell from 2.175 and 9.756 seconds to 2.078 and 9.612 seconds respectively. At 512 tokens it went from 0.215 to 0.221 seconds. All 256 rows of held-out prefill logits matched the previous allocation byte for byte, both with a 64-token chunk and with a 256-token chunk.

The Radeon preset uses `IK2_PREFILL_DENSE=2` to keep just the 24 fused QKV projections in FP16 (868 MiB). Q8 projections go directly through wave32 WMMA on batches of 32 through 511 tokens. At a chunk size of 2,048, the selective cache took 0.151, 2.214 and 10.333 seconds for 256, 8,192 and 32,768 input tokens. The 56-matrix cache took 0.160, 2.219 and 10.368 seconds. The uncached, selective and full caches produced identical full-vocabulary logits on the 256-token passage with chunk size 128. All times are medians of three requests.

```sh
# Use the preset environment and profile-pp-ids from the short-prompt test above.
# Compare the workspace on commit 3b2e4ac and the compact build using the same 56-matrix cache.
IK2_PREFILL_DENSE=1 python3 tools/ik2_bench.py --engine ./build-ik2/tintero --model /weight.gguf \
    --tokens profile-pp-ids --depths 512,8192,32768 --chunk 8192 --gen 3 --repeats 3

for cache in 1 0 2; do
    IK2_PREFILL_DENSE=$cache python3 tools/ik2_bench.py --engine ./build-ik2/tintero --model /weight.gguf \
        --tokens profile-pp-ids --depths 64,256,512,2048,8192,32768 --chunk 2048 --gen 3 --repeats 3
    IK2_PREFILL_DENSE=$cache IK2_PREFILL_LOGITS_OUT="cache-$cache.f32" ./build-ik2/tintero \
        --model /weight.gguf --ctx 1024 --chunk 128 --tokens @heldout-teacher.all --gen 0
done
cmp cache-0.f32 cache-1.f32 && cmp cache-1.f32 cache-2.f32
```

#### Alternating prompt lengths

The preset keeps the two most recently used decode graphs for each output mode. When successive requests cross an attention depth boundary, the engine can replay the previous graph. On 24 alternating uncached prompts of 128 and 2,048 tokens, both configurations returned the same 120 greedy token IDs. After the first two requests of each length, the median time for four decode steps fell from 38 to 34 ms at 128 tokens and from 40 to 36 ms at 2,048. The second graph used another 2 MiB of reported device memory at a context reservation of 393,216. A separate run visited depths 128, 2,048 and 8,192 in the order A, B, C, A, C, B, first greedy and then with temperature 0.7, top-p 0.95 and seed 1. All 60 generated token IDs matched with the cache enabled and disabled. Expert placement in managed or host memory retains its existing graph lifecycle.

```sh
python3 - <<'PY'
from pathlib import Path
ids = Path("profile-pp-ids").read_text().strip().split(",")
with Path("mixed.requests").open("w") as out:
    for r in range(12):
        for j, n in enumerate((128, 2048)):
            prompt = [str(1000 + 2 * r + j)] + ids[r:r+n-1]
            print("GEN 5 0 1 1 -", ",".join(prompt), file=out)
    print("QUIT", file=out)
PY
for pair in 0 1; do
    IK2_GRAPH_PAIR=$pair ./build-ik2/tintero --model /weight.gguf --serve-ik2 --ctx 393216 --chunk 1024 \
        <mixed.requests >"mixed-$pair.log"
    grep '^@TOK ' "mixed-$pair.log" >"mixed-$pair.ids"
done
cmp mixed-0.ids mixed-1.ids
python3 - <<'PY'
from statistics import median
for pair in (0, 1):
    rows = [l.split() for l in open(f"mixed-{pair}.log") if l.startswith("@END ")][4:]
    for n in (128, 2048):
        times = [float(r[5]) for r in rows if int(r[2]) == n]
        assert len(times) == 10
        print(pair, n, 1000 * median(times), "ms")
PY
```

#### Single-token NVFP4 timings

For single-token WMMA, the E2M1 lookup now constructs the four FP16 values directly from their high bytes. Sixteen waves split the gate/up dot products, and four split each down projection. At model width, all 1,152 activated gate/up outputs and 2,304 down outputs passed the FP64 check with the same tolerances as the smaller case. The device suite passed at both K depths and with register tiles.

With eight of 128 generated experts active and width 2,304, the median gate/up kernel time fell from 60.455 to 42.411 microseconds. The down projection went from 25.720 to 27.730 microseconds. Scalar gate/up and down measured 37.715 and 23.225 microseconds on the new binary. These are device-event times across 96 expert selections per repetition. They exclude routing and input conversion. The scalar default still wins for single-token NVFP4 on this fixture.

```sh
# From /work in the CPU build container. For the earlier comparison, link the same tool against the previous library.
hipcc --offload-arch=gfx1100 -O3 -DSTRATA_USE_HIP=1 -Isrc/include/strata/hip_compat -Isrc/src/ik2 \
    tools/ik2_nv_bench.hip -x none build-ik2/libik2_kernels.a -o ik2-nv-bench
# In the GPU container.
IK2_EXPERT_MMA=1 ./ik2-nv-bench
```

#### Other gfx1100 experiments

Prometheus kept four waves for global prefill. Using eight after 8,192 cached positions reduced the time of an isolated deep attention kernel but increased full-model prefill at both 32K and 64K. Double buffering the input and weight tiles in LDS also increased full-model prefill time at 2K, 8K and 32K. Fully unrolling the 2,304-input Q8 projections changed median decode by less than 0.5 tok/s at both 512 and 8K. The preset keeps the existing unroll.

The resident loader also tried copying each mapped tensor in file order across experts. Both orders produced byte-identical held-out decode logits. Their relative startup times changed across runs with different file-cache contents, so the loader keeps its existing order.

These commands use the retained experiment binaries in the Prometheus build directory. Export the Radeon preset first, then set the listed overrides to reproduce the comparison.

```sh
for waves in 4 auto; do
    IK2_FLASH_WAVES=$waves IK2_PREFILL_DENSE=1 python3 tools/ik2_bench.py --engine ./flash-depth \
        --model /weight.gguf --tokens profile-pp-ids --depths 32768,65536 --gen 3 --ctx 393216 --repeats 3 --chunk 1024
done
for pipeline in 0 1; do
    IK2_PREFILL_PIPELINED=$pipeline IK2_WMMA_K=32 IK2_PREFILL_TILE=64 IK2_PREFILL_DENSE=1 \
        python3 tools/ik2_bench.py --engine ./pipeline --model /weight.gguf --tokens profile-pp-ids \
        --depths 2048,8192,32768 --gen 3 --repeats 3 --chunk 2048
done
for unroll in 4 18; do
    IK2_PROJ_UNROLL=$unroll python3 tools/ik2_bench.py --engine ./proj-width --model /weight.gguf \
        --tokens water.force --depths 512,8192 --gen 257 --repeats 3 --chunk 2048
done
for order in 1 0; do
    IK2_ORDERED_LOAD=$order IK2_FORCE=heldout-teacher.force IK2_LOGITS_OUT="order-$order.f32" ./ordered \
        --model /weight.gguf --ctx 1024 --chunk 128 --tokens @heldout-teacher.prompt --gen 33
done
cmp order-0.f32 order-1.f32
```

#### Global attention at 392,957 keys

The device suite checks the global attention computation at the production context depth independently of a language task. Four keys at 5%, 50%, 95% and the last position carry known scores and value patterns. All other keys and the attention sink have score zero. This gives a closed-form FP64 reference for every output channel. The first query must exclude the last key, the second must include it, and unused positions in the final 16-token page carry large sentinel values. On gfx1100 both cache layouts passed all 768 prefill outputs and all 384 decode outputs at 512 decode splits. Run `./build-ik2/ik2_kernels_test` with the Radeon preset.
