"""Exit successfully when the server reports that the model is ready."""
import json
import os
import sys
from urllib.request import ProxyHandler, build_opener

host = os.getenv("TINTERO_HOST") or "0.0.0.0"
if host == "0.0.0.0":
    host = "127.0.0.1"
if host == "::":
    host = "::1"
if ":" in host:
    host = f"[{host}]"
port = os.getenv("TINTERO_PORT") or "8080"

try:
    with build_opener(ProxyHandler({})).open(f"http://{host}:{port}/health", timeout=2) as response:
        payload = json.load(response)
        ready = isinstance(payload, dict) and payload.get("ready") is True
except (OSError, ValueError):
    ready = False
sys.exit(0 if ready else 1)
