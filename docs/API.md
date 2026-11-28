# Tintero HTTP API

Tintero serves OpenAI Chat Completions, OpenAI Responses and Anthropic Messages from one
process, plus llama.cpp-style tokenizer, status and model-state routes. It serves one model,
so there is nothing to select: point a client's base URL at `http://127.0.0.1:8085/v1` (the port
used in the [quick start](../README.md#quick-start)) and send any model name.

The examples below assume:

```sh
BASE=http://127.0.0.1:8085
KEY=your-key   # leave out the Authorization header when the server has no key
```

Text generation requires a loaded model. The [quick start](../README.md#quick-start) uses a
24 GB GPU. The HTTP server can continue answering status requests after a model unload.
To exercise the API with canned responses, run
`python3 serve/server.py --engine mock --port 8085`.

## Routes

| Method | Path | Key needed | Needs the GPU | Purpose |
|---|---|---|---|---|
| GET | `/health` | no | no | [Readiness](#readiness) and basic facts |
| GET | `/v1/models`, `/models` | yes | no | The loaded model and its context size |
| POST | `/v1/chat/completions` | yes | yes | [Chat Completions](#chat-completions), streamed or not |
| POST | `/v1/responses` | yes | yes | [Responses](#responses), see [RESPONSES.md](RESPONSES.md) |
| GET, DELETE | `/v1/responses/{id}` | yes | no | Retrieve or delete a stored response |
| GET | `/v1/responses/{id}/input_items` | yes | no | List the input items for a response |
| POST | `/v1/responses/{id}/cancel` | yes | no | Cancel a background response |
| POST | `/v1/messages` | yes | yes | [Anthropic Messages](#anthropic-messages) |
| POST | `/tokenize`, `/detokenize` | yes | no | [Token ids](#tokenize-and-detokenize) |
| GET | `/v1/status` | yes | no | Uptime, context, concurrency, Responses cache and job counters |
| GET | `/metrics` | yes | no | Monitor-style JSON: live request, recent requests, totals, hardware |
| GET | `/metrics/prometheus` | yes | no | [Prometheus text format](#metrics) |
| GET | `/props`, `/slots` | yes | no | llama.cpp-compatible server properties and slots |
| GET | `/mcp` | yes | no | [MCP](#mcp-tools) servers and their tools |
| POST | `/models/load`, `/models/unload`; GET `/models/sse` | yes | loads the model | [Model state](#model-state) |
| GET, POST | `/api/context`, `/api/thinking-budget`; GET `/api/update` | yes | reload | [Runtime controls](#runtime-controls) |

"Key needed" applies only when the server has an API key. Without one, every route is open.
Unknown paths return 404 (`{"error": {"message": "not found"}}`); for POST and DELETE the key is
checked first, so a missing key on an unknown path is a 401.

Browser-page and dashboard routes that the server also answers are not part of this API.

## Authentication

Set a key with `--api-key`, the `MAYA_API_KEY` environment variable, or `"api_key"` in the
config JSON. The flag, then the variable, then the config decide which one is used. A flag
shows up in the process list; prefer the variable or the config.

```sh
docker run ... -e MAYA_API_KEY=your-key ghcr.io/godgmn/tintero:v0.1.0-cuda       # Docker image
MAYA_API_KEY=your-key python3 serve/server.py --engine maya --config docs/inkling2-mini/cuda-24gb-fp4.json
```

Several keys: separate them with commas (`k1,k2`), or use a list in the config
(`"api_key": ["k1", "k2"]`), which also lets a key contain a comma.

A client sends the key in one of two headers:

```sh
curl -fsS "$BASE/v1/models" -H "Authorization: Bearer $KEY"
curl -fsS "$BASE/v1/models" -H "x-api-key: $KEY"
```

Precedence, as the server implements it:

1. A header that starts with `Bearer ` (any case) is the key. `x-api-key` is then ignored, so a
   wrong Bearer value with a correct `x-api-key` is rejected.
2. Any other `Authorization` value (`Basic ...`, or the bare key with no scheme) is ignored, and
   `x-api-key` is used.
3. A key in the query string is never read.

A missing or wrong key returns HTTP 401 with the same body on every route, including
`/v1/messages`:

```sh
curl -s -w ' %{http_code}\n' "$BASE/v1/models"
# {"error": {"type": "authentication_error", "message": "missing or wrong API key"}} 401
```

`/health` is the only public API route. It reports `"api_key": true` when a key is set.

Stored Responses are isolated by key: a response saved with one key returns 404 for another.
`Authorization: Bearer` and `x-api-key` with the same key share their responses. Without a key,
all callers share one store.

## Binding beyond loopback

The server listens on `127.0.0.1` by default, which only this machine can reach. To change that:

| How you run it | Setting |
|---|---|
| `serve/server.py` | `--host 0.0.0.0`, or `"host": "0.0.0.0"` in the config. The flag wins over the config. |
| `docker/entrypoint.sh` (the Docker image's entrypoint) | `TINTERO_HOST` (default `0.0.0.0`) and `TINTERO_PORT` (default `8080`); arguments after the script, such as `--host`, are appended and win. |
| Docker image | The container listens on `0.0.0.0:8080`, so what the host exposes is the `-p` mapping. The README's `-p 127.0.0.1:8085:8080` is loopback only; `-p 8085:8080` publishes the port on every host interface. |

Before exposing the port, set an API key. On a non-loopback `--host`, the server prints the
address other devices should use and a warning when no key is set. The server does not speak
TLS; put a reverse proxy in front for anything beyond a trusted network. A port that is already
taken stops the start before the model loads.

Browsers are a separate case. A page from another site cannot call the server unless the config
lists it:

```json
{ "cors_origins": ["http://localhost:3000"] }
```

- Only `/v1/*` gets CORS headers. A preflight (`OPTIONS`) needs no key and returns 204 for a
  listed origin, 403 otherwise. `["*"]` allows every origin.
- A POST or DELETE carrying an `Origin` that is neither the server's own host nor listed is
  refused with 403, even with a valid key. `curl` and the SDKs send no `Origin` and are not
  affected.

Request bodies over 256 MiB return 413 (`MAYA_MAX_BODY_MIB` changes the limit). A connection
that waits 300 seconds on one socket read or write is dropped (`MAYA_REQUEST_TIMEOUT_S`).

## Readiness

The port stays closed while the model loads (a minute or two on the first start), so a client
sees connection refused rather than an error. Once the port answers, `/health` needs no key:

```sh
until curl -fsS "$BASE/health" >/dev/null; do sleep 2; done
curl -fsS "$BASE/health"
```

```json
{"status": "ok", "max_context": 32768, "model": "inkling-2-mini", "version": "1.1.70",
 "api_key": true, "images": false, "videos": false, "unloaded": false, "mcp_tools": false,
 "defaults": {}, "history_thinking": true, "ablate": {"available": false, "default": false}}
```

(`max_context` is the preset's context, set by `--max-context` in its `args`. Fields vary a
little with the model and config.)

- `status` is `ok`, `reloading` (the model is restarting after a context change) or `updating`.
  The HTTP status stays 200 in all three.
- `unloaded: true` means idle unload freed the GPU. The next generation request loads the model
  again, which takes about as long as a start.
- While `reloading` or `updating`, `/v1/chat/completions`, `/v1/responses` and `/v1/messages`
  return 503 with `Retry-After: 30` and `{"error": {"type": "overloaded_error", ...}}`.

## Models

```sh
curl -fsS "$BASE/v1/models" -H "Authorization: Bearer $KEY"
```

Returns one entry: `id` is the config's `model_name` (`inkling-2-mini` in every preset),
`status.value` is `loaded` or `unloaded`, and `meta.n_ctx` / `context_length` is the context
size. The `model` field of a request is not matched against it; any value is accepted and the
response echoes the loaded model's id.

## Chat Completions

```sh
curl -fsS "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"What is 17 + 25? Answer with the number only."}],"reasoning_effort":"none","temperature":0,"max_tokens":48}'
```

The answer is `choices[0].message.content`. When the model thinks, its reasoning is returned
separately as `choices[0].message.reasoning_content`. Stream with `"stream": true`; each event
is a `data:` line of JSON and the stream ends with `data: [DONE]`. Add
`"stream_options": {"include_usage": true}` for token counts in the last chunk:

```sh
curl -fsSN "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello."}],"stream":true,"stream_options":{"include_usage":true},"max_tokens":64}'
```

Settings worth knowing (the model-specific ones are in [SETTINGS.md](SETTINGS.md)):

- `reasoning_effort`: `none` (also `off` and `minimal`) turns thinking off, `low`, `medium`,
  `high`, `xhigh` and `max` pick a level, and `auto` uses the model's default. Any other word is a
  400 that lists the accepted ones. Without the field the preset's default applies.
- `thinking_budget`: a per-request cap on thinking tokens (0 removes the cap).
- `max_tokens` / `max_completion_tokens`: unset, 0 or -1 means "whatever the context has left".
- `tools` and `tool_choice`, and image parts when the preset has vision (`"images": true` in
  `/health`).
- A prompt plus `max_tokens` that does not fit the context is rejected with 400 and
  `"code": "context_length_exceeded"`. Nothing is ever truncated.

Errors keep the OpenAI shape, `{"error": {"type": ..., "message": ...}}`: 400 for a bad body or
parameter, 401, 413, 503 while reloading. An engine failure is 422 `engine_error` (not a 5xx, so
SDKs do not retry a long generation); [SETTINGS.md](SETTINGS.md#when-the-engine-stops-an-answer)
has the streamed forms.

Requests queue behind the engine: one answer at a time, or as many as the preset's batch slots
(`docs/PRESETS.md`). A client that disconnects stops its own generation.

## Responses

`POST /v1/responses` takes the same key and the same engine as Chat Completions. The full
reference, with tools, structured output, `previous_response_id`, background tasks and the
retention limits, is [RESPONSES.md](RESPONSES.md). Responses are not stored unless `store` is
true:

```sh
curl -fsS "$BASE/v1/responses" \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"input":"Remember the number 73.","store":true,"max_output_tokens":64}'
```

The saved response can be read, listed and deleted by its `id`:

```sh
ID=$(curl -fsS "$BASE/v1/responses" -H "Authorization: Bearer $KEY" \
  -d '{"input":"Remember the number 73.","store":true,"max_output_tokens":64}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
curl -fsS "$BASE/v1/responses/$ID" -H "Authorization: Bearer $KEY" >/dev/null
curl -fsS "$BASE/v1/responses/$ID/input_items" -H "Authorization: Bearer $KEY"
curl -fsS -X DELETE "$BASE/v1/responses/$ID" -H "Authorization: Bearer $KEY"
```

Unknown, expired, deleted and other-key ids all return 404. Unsupported parameters return 400
before generation (`conversation` and `strata_mcp` among them).

## Anthropic Messages

```sh
curl -fsS "$BASE/v1/messages" \
  -H "x-api-key: $KEY" -H 'anthropic-version: 2023-06-01' -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","max_tokens":64,"messages":[{"role":"user","content":"Say hello."}]}'
```

Streaming, tools and images follow Anthropic's format. The server does not check the
`anthropic-version` header, and a trailing query such as `?beta=true` is ignored. Both key
headers work here. A 401 still uses the OpenAI-style error body shown above.

## Tokenize and detokenize

```sh
curl -fsS "$BASE/tokenize" -H "Authorization: Bearer $KEY" -d '{"content":"Hello"}'
curl -fsS "$BASE/tokenize" -H "Authorization: Bearer $KEY" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
curl -fsS "$BASE/detokenize" -H "Authorization: Bearer $KEY" -d '{"tokens":[72,101]}'
```

`/tokenize` takes `content` (or `prompt`) for plain text, or `messages` (with optional `tools`) to
count a whole chat prompt as the model sees it. It returns `tokens`, `count` and `max_model_len`.
`/detokenize` returns the text as both `content` and `prompt`. The ids above are the mock
engine's byte tokenizer; the real model uses its own vocabulary.

## Status and metrics

`GET /v1/status` returns the server's name and model, uptime, context, concurrency and counters for
the Responses cache and background jobs, plus the list of dialects served. `GET /metrics` returns
the live request and recent request timings as JSON (`?requests=all` keeps every request); its
shape follows the Monitor and can change between releases. `GET /props` and `GET /slots`
mirror llama.cpp's: context size, template and slot activity.

### Metrics

`GET /metrics/prometheus` is the stable scraping route (`text/plain; version=0.0.4`). It carries
llama.cpp's counters (`llamacpp:prompt_tokens_total`, `llamacpp:tokens_predicted_total`,
`llamacpp:predicted_tokens_seconds`, `llamacpp:requests_processing`, ...) and Tintero's own
(`maya:requests_total`, `maya:engine_alive`, `maya:loop_guard_*`, `maya:responses_cache_*`,
`maya:responses_jobs_*`). None contain prompts or key material. A scrape job with a key:

```yaml
scrape_configs:
  - job_name: tintero
    metrics_path: /metrics/prometheus
    authorization: { credentials: your-key }   # sent as "Authorization: Bearer your-key"
    static_configs: [{ targets: ["127.0.0.1:8085"] }]
```

```sh
curl -fsS "$BASE/metrics/prometheus" -H "Authorization: Bearer $KEY" | grep '^llamacpp:requests_processing'
```

## MCP tools

The server can run tools from [MCP](https://modelcontextprotocol.io) servers on the model's
behalf. List servers under `"mcp_servers"` in the config or with `--mcp-config file.json`
(Claude Desktop's `{"mcpServers": {...}}` format), as stdio commands or Streamable HTTP URLs.
The server starts them in the background and names each tool `<server>__<tool>`.

```sh
curl -fsS "$BASE/mcp" -H "Authorization: Bearer $KEY"
```

Tools are offered only when a Chat Completions request asks for them with `"strata_mcp": true`;
every other request behaves as if no MCP server existed. The server then runs the tool loop
itself (at most 8 rounds, tool output capped at 20,000 characters) and the response message
carries a `strata_mcp` list describing each call.

```sh
curl -fsS "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Add 2 and 3."}],"strata_mcp":true,"max_tokens":128}'
```

Scope and limits:

- Chat Completions only. `/v1/responses` rejects `strata_mcp` with 400; `/v1/messages` ignores it.
- The request must be `application/json` (415 otherwise) and carry no foreign `Origin` (403), so a
  web page elsewhere cannot trigger tools.
- Tools run on the host with the server's own permissions. Anyone who can reach the port with a
  valid key can use them, so set a key before binding beyond loopback.

## Model state

Enabled by `"model_management"` in the config or `MAYA_MODEL_MANAGEMENT=1`, in llama.cpp's form:

- `POST /models/unload` frees the GPU and RAM (409 while a request is running). The next
  generation request reloads the model.
- `POST /models/load` loads it again, and returns when it is ready.
- `GET /models/sse` streams `{"object": "list", "data": [{"id": ..., "status": {"value": "loaded"}}]}`
  every five seconds, with `loaded`, `unloaded` or `loading`.

When it is off these return 404. `--idle-unload SECONDS` (or `"idle_unload_s"`) does the same
automatically after a quiet period.

## Runtime controls

- `GET /api/context` reports the context size and its range; `POST /api/context` with
  `{"max_context": N}` reloads the model at a new size, for every client.
- `GET /api/thinking-budget` reports the budget; `POST` with `{"budget": N}` changes it without a
  restart (`"supported": false` when the engine has no cap).
- `GET /api/update` reports the running version and whether a newer release exists
  (`"enabled": false` when `MAYA_UPDATE_CHECK=0`).

Both POSTs need `application/json` and no foreign `Origin`, the same rule as MCP. They change the
server for everyone, so protect them with a key.
