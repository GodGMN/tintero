# Tintero settings

The [gfx1100 preset](inkling2-mini/ik2-mini-gfx1100.json) supplies the Radeon defaults. See the [HIP guide](HIP-PORT.md) for slot counts, memory use and tested cache sizes.

## When the engine stops an answer

An answer ends with `finish_reason` "length" (Anthropic: `stop_reason` "max_tokens"; Responses: status "incomplete"
with reason "max_output_tokens") only when it reached its token limit or the context. When the engine stops an
answer itself because of an error - a GPU's fault, a failed decode step, numbers gone non-finite - the request ends
with that API's own error instead, never with a finish reason a client would take for the limit:

| API | Streamed | Not streamed |
|---|---|---|
| OpenAI chat completions | the text already sent stands; then a `data: {"error": {"type": "server_error", ...}}` chunk and `data: [DONE]` (no `finish_reason` chunk) | HTTP 422, `{"error": {"type": "engine_error", "code": "engine_error", "message": ..., "partial_text": ...}}` |
| Anthropic messages | the text already sent stands; then `event: error` (`api_error`), no `message_delta` with a `stop_reason` | HTTP 422, `{"type": "error", "error": {"type": "engine_error", "message": ..., "partial_text": ...}}` |
| Responses | `response.failed` with `error.code` "server_error" | HTTP 422, the response with status "failed", `error.code` "engine_error" and `error.partial_text` |

The documented finish reasons have no "error" value (OpenAI's are stop / length / tool_calls / content_filter /
function_call, Anthropic's end_turn / max_tokens / stop_sequence / tool_use / pause_turn / refusal), so a new one
would be read as a normal end by some clients; the error forms above are the ones every client handles (the OpenAI and
Anthropic SDKs raise an error on them). A request that is not streamed gets 422, not a 5xx: the SDKs send a 5xx,
408, 409 or 429 again by themselves, which would run the same long answer once more without the client seeing the
error. Its error carries what the answer had written before it: `partial_text` and `partial_reasoning` when it was still thinking, less the last few tokens the loop guard still held
back. The message says how many tokens the answer had and why it stopped: after a GPU's fault on DeepSeek, the call
that failed and the position in the context (what the engine prints on the console as it ends); otherwise that the
reason is in the engine's log. After a GPU's fault the engine ends, and the next request starts a fresh one.

## The thinking level and the thinking budget

Two config settings shape the thinking, and only one of them counts tokens:

| Setting | What it is |
|---|---|
| `"thinking_budget"` | The token cap for reasoning. Inkling-2-Mini defaults to 0, which leaves reasoning uncapped. |
| `"reasoning_effort"` | The thinking level when a request leaves it out. Inkling accepts `none`, `auto`, `low`, `medium`, `high`, `xhigh` and `max`. |

Set `thinking_budget` in the preset or POST a JSON body such as `{"budget":2048}` to `/api/thinking-budget`.
Chat Completions, Responses and Messages also accept a per-request override. Zero disables the cap for that request.
Inkling counts reasoning tokens
and opens an answer segment at the cap. If the output limit binds first, the server reserves
a quarter of it (up to 1024 tokens) for the answer. Inkling reads the cap from the top-level JSON setting, outside `env`.
Anthropic's `thinking.budget_tokens` selects an effort level. The hard cap uses `thinking_budget`.

An invalid `reasoning_effort` in the preset stops startup before the model loads.

## The loop guard

A quantized model, greedy above all, can fall into a loop: the same sentence, number sequence or block of code again
and again until `max_tokens`. The server watches every answer for one, in the thinking and in the answer, on every
model (GLM, DeepSeek, MiniMax) - beside the GLM engine's own check for short patterns (`MAYA_GLM_LOOP_THINK`,
`MAYA_GLM_LOOP_ANSWER`).

- **What counts as a loop:** the end of what was written is one block of 8-2048 tokens repeated exactly, back to back
  (4 times for a block under 32 tokens, 3 times for a longer one), over 96 tokens at least. In the thinking it takes
  more: 5 and 4 times, over 192 tokens at least (reasoning that drafts a plan again is not a loop). A table's rows,
  an array, a name used over and over or code lines alike are not exact copies of one block back to back, so they go
  on untouched. The check costs a few microseconds a token.
- **The first loop** (the first and second in the thinking): the repeats not sent yet are dropped and the model goes
  on from there with a mild repetition penalty (1.15 over the last 2 x the block's length, at least 128 tokens) for
  the next 2 x the block's length + 128 tokens; then the request's own sampling again.
- **A third loop in the thinking:** the thinking is closed (the model's own end-of-thinking token) and it answers,
  with the same short penalty at first. The thinking is never ended any other way: the answer always follows.
- **A second loop in the answer, or a fourth in all:** the answer ends there (`finish_reason` "stop"). When no answer
  text was written yet (a model without an end-of-thinking token, looping while it reasons), the answer says so in
  one line instead of ending empty.
- **Streaming:** the stream is at most 64 tokens behind the model. Tokens that repeat what came before them are held
  back only for that short tail; a block the model copies from earlier in its output (a draft written again, a
  restated plan) streams on as it is written. What was sent stays; a loop's repeats past the tail are cut from what
  follows.
- **Nothing changes without a loop:** the same engine calls with the same sampling, the same tokens.
- **Across turns (agents):** an agent can loop with every answer short and fine - the same tool call, the same result,
  the same call again. When a request ends with tool results and its last call (name, arguments and result all the
  same) is the 3rd of its kind among the last 12 calls since the user's last message, a short message follows that
  result, marked as the server's: the call gave nothing new, so use what is there, try a different approach or answer.
  Only that prompt changes; the model goes on. A command run again after an edit gives another result, so it does not
  count. The note is worded differently at the 3rd, 4th and 5th run; the last one says a block comes next. A call
  made again after those three notes is **blocked**: it never reaches the client, the server answers it in the prompt
  ("not run") with a firmer note and the model goes on in the same answer, its thinking started for it a step away from
  the call (worded differently each time). It is never stopped: if it writes the blocked call again, the same happens,
  a little hotter (temperature +0.15 a time from the 2nd, at most 1.5), until it does something else. A user's bare
  "continue" / "proceed" / "ok" does not start the count over, and the model is told at once which calls stay blocked.
  Within a row of the same call, a result that differs only in numbers or garbled bytes counts as the same (not for a
  wait with `sleep`). Blocks count in `/metrics`
  `loop_guard.tool_blocked` and `maya:loop_guard_tool_blocked_total`.
- **A conversation that repeats itself:** once one piece of the model's own text turns up in 3 of its messages (or
  one call with the same result has run 3 times), the prompt leaves out what the model already wrote: a sentence of
  its reasoning or a paragraph of its answer text that it wrote before, and a tool round whose calls and results all
  repeat earlier ones with nothing new said. The model then reads what it found once, not dozens of copies of the
  same thoughts to go on from (GLM-5.3 had one sentence 543 times in an 88K prompt). Only the prompt changes, and each
  message depends only on the ones before it, so the next request's prompt starts the same and the cache holds.
  Counted in `loop_guard.collapsed` and `maya:loop_guard_collapsed_total`.
- **An answer that ends the turn with nothing new:** in a conversation that repeats itself a model can end its turn
  with its own words about the loop ("OK - the loop guard is right. Break out: use what I have") or with what it
  already said - and the agent stops there while the work is not done. Such an answer goes on: a short user message,
  marked as the server's, follows it ("do not stop here, the task is not done: take the next concrete step") and the
  model continues in the same answer. An answer with anything new in it - a finding, an answer to the user, one line
  of it - ends the turn as always, however short it is, and so does any answer after 3 such notes in one answer. An
  answer to the user is never kept going: when the request ends with a user message, not a tool result, the model is
  answering the user, and a sentence said again across user turns is the user asking the same kind of thing again,
  not a loop (#921). The reasoning is not re-opened: it was closed before the answer went out, so the model goes on
  in the answer's own channel (a client sees the answer continue, not its thinking rewritten, with a paragraph break
  before the continuation). Counted in `loop_guard.kept_going` and `maya:loop_guard_kept_going_total`.
- **A turn-end marker inside the thinking:** a model that quotes these markers, or writes about them, wrote them in its
  thinking - and there the server ended the turn (or the engine itself, at its EOS token): the thinking was cut off
  and the answer never came. Inside the thinking such a marker is the model's text, not the turn's end: the server
  skips the marker's token and the thinking goes on; at the engine's own end-of-sequence token it goes on from the
  quote with a short note as the server's (at most twice), and the thinking continues in its own channel. In the
  answer a marker other than the engine's own EOS token, written as text - quoted in inline code or a code fence, or
  inside a tool call's arguments - is text too; a marker spelled out of ordinary pieces in the answer's prose still
  ends it, and so does the engine's EOS token there (a call cut off by it is incomplete, not failed).

The server window gets a line for each loop broken (`[maya] loop guard: a 19-token block 5 times in the answer at
token 1,763: repeats dropped, a short repetition penalty`); `/metrics` marks that request (Loop guard,
on hover what it did) and counts the loops since the start, as do `/metrics` (`loop_guard`) and
`/metrics/prometheus` (`maya:loop_guard_*_total`; the notes across turns: `[maya] loop guard: `bash` 3 times with the
same arguments and result: a note after it`, `maya:loop_guard_tool_repeats_total`).

Off for the server: `MAYA_LOOP_GUARD=0`, or `"loop_guard": false` in the run config. A request can choose for itself
with `"loop_guard": false` (or `true`) beside its sampling fields (Chat Completions, Anthropic Messages).

## Inkling-2-Small

thinkingmachines' [Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) (594B, ~17B active) on
its own engine, `tintero` (`src/ik2`; CUDA on Linux, Ampere on), from unsloth's GGUFs
([unsloth/Inkling-2-Small-GGUF](https://huggingface.co/unsloth/Inkling-2-Small-GGUF): UD-Q4_K_XL, UD-Q3_K_XL) read in
place. The routed experts sit in pinned host memory (~323 GiB for UD-Q4_K_XL, ~260 GiB for UD-Q3_K_XL) with a cache of
them in VRAM. It serves one text request at a time. See [Inkling-2-Small](INKLING2.md#6-phase-2-the-engine-tintero) for the
tokenizer and run config.

**Defaults**: temperature 1.0, top-p 1.0 (its card and files name none); the thinking level Auto, the template's own
(0.9 - "high"). The template is thinkingmachines' (`tools/ik2/chat_template.jinja`), which takes every level by name:
none, low (0.2), medium (0.7), high (0.9), xhigh and max (0.99); Off is none.

The engine sizes the expert cache after allocating weights, KV for the requested context and scratch buffers.
Streaming reserves another 2 GiB. On one GPU, `--vram-gb N` simulates a device budget in GiB. The cache is
limited by both actual free memory and the simulated budget after device allocations. Pinned host experts
are outside the simulated budget.

The engine saves a binary profile for each canonical model path under
`$XDG_CACHE_HOME/tintero` (default `~/.cache/tintero` on Unix, `%LOCALAPPDATA%/tintero` on Windows).
On the next launch, each pool loads its most frequently selected experts with an initial eviction frequency
of at most two. Counts accumulate across launches and halve together when a counter would overflow.
Concurrent sessions publish complete snapshots; the last clean shutdown supplies the next profile.

Engine flags: `--max-context N` (262144), `--cache-gb N` (optional expert-cache ceiling in GiB),
`--gpus 1|2`, `--threads N` (8: the loading and the CPU lane), `--eos a,b`. Environment:

| Variable | Default | What it does |
| --- | --- | --- |
| `IK2_FUSED_ROUTER` | `0` | HIP only. Fuse FP16 router projection, biased sigmoid top-8 selection and weight normalization for 128 experts and 16–64 batched rows. Single-token decode keeps the wider `mv_f16` + `route_token` path. |
| `IK2_KV` | `f16` | `fp8`: the KV cache in FP8 E4M3, half the bytes (the full layers' 32 KB a token -> 16 KB); against the reference it changed 5 of 48 top-1 tokens where FP16 changed none (docs/INKLING2.md 6.1). On CUDA (SM 80 and newer) and HIP (RDNA 3/3.5, including gfx1100 and gfx1151), `fp4-global` stores global keys and values in E2M1 blocks and keeps sliding layers in E4M3. Each 128-channel global head uses 72 bytes for K and 72 for V, including scales. On HIP, `fp4` also packs the sliding-window ring in E2M1. Both modes require 128-wide heads and dense attention. HIP accepts `IK2_PAGED_KV=1` for 16-token global pages. |
| `IK2_WARM` | model profile | Read expert counts from this file at startup. An empty value starts cold. |
| `IK2_POP_OUT` | model profile | Save the updated counts here on clean shutdown. An empty value disables saving. The resident decoder normally counts on the GPU. An explicit output path selects the existing host profiling path. |
| `IK2_EP` | off | `1` with `--gpus 2`: expert parallel - each layer's experts dealt to both GPUs by how often they are picked, each GPU its own cache and link; attention and the KV cache on GPU 0. Decode 22 -> 42 tok/s on two RTX PRO 6000s. Off, `--gpus 2` splits the layers |
| `IK2_EP_SPLIT` | `0.55` | GPU 1's share of the picks under `IK2_EP` (GPU 0 also holds the dense weights and the cache of the KV) |
| `IK2_TP` | off | `1` with `--gpus 2`: the experts tensor parallel instead (each GPU half of every expert): decode 42.8 vs EP's 41.3 tok/s, prompts slower |
| `IK2_CPU` | off | `1`: in decode, a missing NVFP4 expert computed on the CPU (AVX-512) when that is sooner than its copy over PCIe (H200 node, with `IK2_CPU_LAZY=0`: 34.5 -> 36.2 tok/s); one GPU only. `IK2_CPU_LAZY=0`: such an expert is not copied into the cache afterwards either (the link carries less) |
| `IK2_SUP` | 16384 (4096 when VRAM is short) | the tokens of one layer-major prompt pass (each pass copies every expert it misses once); a pass's scratch (~280 KB a token) comes out of the expert cache |
| `IK2_PF_P` | `0.999` | the prompt prefetch: an expert is copied ahead when the chance that some token of the layer picks it is at least this |
| `IK2_WGMMA` | on | `0`: the mma.sync attention on Hopper too (the wgmma one needs a build with sm_90a) |
| `IK2_ALLOC` | mmap + register | `hostalloc`: the experts' host memory by `cudaHostAlloc`, which pins ~3 GiB/s (load 110-190 s instead of ~50-75 s) |
| `IK2_HOST` | pinned; `mmap` when every expert stays in VRAM | `mmap`: the experts read through the page cache instead of pinned (for a host without the RAM: unusably slow from network storage, 0.34 tok/s). With every expert in VRAM they are copied there straight from the mapped files and no pinned copy is kept (Inkling-2-Mini loads in 3 s from the page cache) |
| `IK2_FAST_LOAD` | on (CUDA resident mmap) | Stage resident expert mmap uploads through a temporary double-buffered `2 × 64 MiB` pinned ring packed across CPU threads, then free the staging buffers after load (DGX Spark GB10: 14.7 GiB in 0.6 s vs 43.3 s unpinned mmap). Set `0` to copy each expert directly from mmap. |
| `IK2_BATCH_GROUP_MOE` | off | CUDA continuous batching, up to 512 routed slots. Speculative verification always groups routed slots. `0` runs per token. `1` groups rows by expert and uses scalar products for IQ4_NL and Q8_0. `2` additionally uses Tensor-Core MMA when gate, up and down share either format and both dimensions are multiples of 32. MMA rounding can change token IDs. |
| `IK2_BLACKWELL_NVFP4` | off | On SM 120a and 121a, `1` quantizes grouped expert activations to E2M1 with a UE4M3 scale per 16 values and uses block-scaled FP4 MMA for gate/up and down. This can change logits. The single-token decode path uses its existing precision. |
| `IK2_GPU_SERVED` | off | Single-GPU CUDA token decode. Use `--cache-gb N` to budget the prompt cache, resident expert slots and per-visit reserves together. Expert weights require pinned mapped host memory. The CLI reports GPU copies, refilled slots and repeat picks served by copies from an earlier loop pass. |
| `IK2_FAST` | on when every expert fits in VRAM | the resident token path (docs/INKLING2.md 8): every expert in VRAM, a token's launches with nothing from the host between its embedding and its logits. `0`: tokens take the cache path |
| `IK2_GRAPH` | on | `0`: the resident path's launches one by one instead of a CUDA graph a token (for profiling: nsys does not see inside a graph) |
| `IK2_ATTN_MMA` | `1` | a token's attention on the tensor cores, a warp straight from the KV cache (65,536 tokens deep on an RTX 3090: 63 -> 77 tok/s with an FP16 cache, 53 -> 84 with FP8). `0`: the scalar kernels; `2`: the layers without a window only |
| `IK2_EXPERT_MMA`, `IK2_EXPERT_MMA_SPLIT`, `IK2_EXPERT_MMA_SPLIT_DOWN` | `1`, `8`, `4` | a token's NVFP4 experts on the tensor cores straight from their blocks (resident path): the gate / up input as two FP16 limbs in two columns, a code times its scale exact in FP16, FP32 sums. A group of 16 rows is a block of `SPLIT` warps, each a stride of the 64-value blocks (the weights in flight). One RTX 3090, Inkling-2-Mini: 121 -> 127 tok/s. `0`: the scalar kernels (also what Q8_0 experts take) |
| `IK2_DFLASH2_QUANT` | `q8` | DFlash2 attention and MLP weights. Choose `q8` or `q4` for signed per-channel quantization at load, or `bf16` for checkpoint precision. The tap projection stays BF16. |
| `IK2_DFLASH2_SAMPLE` | `prob` | At positive temperature, sample from the 16 corrected and softcapped candidate logits at each path position. `greedy` proposes the highest-scoring candidate. |
| `IK2_DRAFT_GRAPH` | `1` | Replay the Q8-head MTP proposal graph on CUDA and HIP. Batched DFlash2 caches a graph for each active count and window width. Set to `0` for eager launches. |
| `IK2_GROUP_MMA` | `0` | CUDA NVFP4 block verification uses FP16 grouped MMA for gate/up with bit 1 and down with bit 2. `IK2_BLACKWELL_NVFP4=1` selects both on supported Blackwell devices. |
| `IK2_DRAFT_WIDTH` | `5` single, `7` batched | Verification window from 2 to 8 tokens, including the anchor. DFlash2 serves up to eight concurrent draft requests with `IK2_BATCH=2..8`. A lone active request uses the single-stream controller. |
| `IK2_DRAFT_ADAPT`, `IK2_DRAFT_PROBE`, `IK2_DRAFT_DECAY` | `1`, `8`, `0.85` | the controller: a window only when its expected tokens over its measured time beat a plain token's; after `PROBE` plain tokens one window to look at the acceptance again; `DECAY` how fast the acceptance is forgotten. `IK2_DRAFT_ADAPT=0`: always `IK2_DRAFT_WIDTH` wide |
| `IK2_DRAFT_CTX` | the context | the positions the drafter's layer without a window sees (the last ones; its cache FP16, 4 KB a position): `65536` keeps 262144 resident on a 24 GB card |
| `IK2_BLOCK`, `IK2_BLOCK_CHECK` | `1`, `0` | `0`: a verify window through forward() instead of the block path; `n`: the first n windows through both, their logits compared |
| `IK2_RG_SLOTS` | 96 tokens' picks | with every expert in VRAM, a prompt of up to this many slots (tokens x picks) has its experts read by groups - an expert's blocks once for up to 8 of its tokens - instead of decoded to FP16 for a GEMM (a chat turn of 12 new tokens 118 -> 56 ms). `0`: never |
| `IK2_CAND` | on | sampling takes the largest logits found on the device (a thousand and a few more) instead of the vocabulary's logits every token (100 -> 110 tok/s at temperature 1; the same tokens for the same seed). `0`: the logits |
| `IK2_FLASH`, `IK2_MV`, `IK2_GROUP`, `IK2_OVERLAP`, `IK2_PREFETCH`, `IK2_SCAN`, `IK2_RELB`, `IK2_BORROW` | on | A/B switches (`0`): the paths these replaced - the attention through logits in memory, few tokens' experts through the GEMMs, no grouping, copies on the compute stream, no prefetch, plain LRU in prompt passes, no relative-bias table, no scratch lent from the cache for long prompts |
| `IK2_DENSE` | FP16; `q8` when every expert stays in VRAM | `q8`: the dense weights kept as the GGUF's Q8_0 blocks (half an FP16 copy's bytes), a token's products straight over them, planar (`mv_q8p`); a prompt chunk decodes a matrix to FP16 for its GEMM. `f16`: the FP16 copy |
| `IK2_PROF`, `IK2_STATS` | off | each request's time by phase; where its experts came from (cached, prefetched, copied on a miss, the CPU) |
| `IK2_TRACE`, `IK2_FORCE`, `IK2_LOGITS_OUT`, `IK2_MARGIN`, `IK2_VERIFY_CHECK`, `IK2_UP_MS`, `IK2_ALLOC_THREADS` | off | measurement: every layer's picks to a file; fixed tokens fed (`--tokens` runs: two builds compared on the same text, and how often each one's pick agrees); every generated token's logits to a file (with `IK2_FORCE`: two builds' logits over the same tokens); the drafter's checks (`--draft`); the CPU lane's estimate of an expert's copy time; threads allocating host memory |
