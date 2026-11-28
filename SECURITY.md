# Security

## Reporting a vulnerability

Please do not open a public issue for a security problem. Open a private security advisory instead: on the
repository's **Security** tab, choose **Report a vulnerability**. Say which version you ran, what you did and what
happened. We will answer there.

## Supported versions

Only the latest release gets security fixes. Use the latest published release or update your checkout before reporting.

## What is in scope

The engine, HTTP server, native launchers and quantization tools. The model weights are out of scope: they come from their authors and from
[Hugging Face](https://huggingface.co/thinkingmachines); report a problem in a model to its authors.

## How Tintero behaves by default

- The server listens on `127.0.0.1` by default. Set `--host 0.0.0.0` or `"host"` in the config to listen on other devices. Set an API key when exposing the server.
- Set `--api-key` or `"api_key"` in the config to require `Authorization: Bearer KEY` or `x-api-key: KEY`. Health remains public.
- The server grants browser CORS on `/v1/*` only to origins listed in `"cors_origins"`.
- `GET /api/update` checks GitHub for the latest release. Set `MAYA_UPDATE_CHECK=0` to disable the lookup.
- The server connects to HTTP image URLs from clients and to MCP servers named in its config.
