"""serve/test_api_guide.py - docs/API.md against the real handler and the mock engine (no GPU, no model).

    python -m unittest serve.test_api_guide -v

The guide's route table, authentication rules, CORS and MCP scope are checked here, and every `sh` block of the
guide that calls `curl "$BASE/..."` is run against the server with BASE and KEY set.  The fixtures are benign
(canned mock text, byte tokenizer, serve/mcp_fake_server.py).
"""
from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.mcp import McpHub  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GUIDE = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
KEY = "guide-test-key"
ORIGIN = "http://localhost:3000"
SCRIPT = "Thinking about it.</think>\n\nHello from the mock engine."

# the guide's route table: path -> (needs a key, a method that reaches it)
ROUTES = {
    "/health": (False, "GET"), "/v1/models": (True, "GET"), "/models": (True, "GET"),
    "/v1/chat/completions": (True, "POST"), "/v1/responses": (True, "POST"), "/v1/messages": (True, "POST"),
    "/tokenize": (True, "POST"), "/detokenize": (True, "POST"), "/v1/status": (True, "GET"),
    "/metrics": (True, "GET"), "/metrics/prometheus": (True, "GET"), "/props": (True, "GET"),
    "/slots": (True, "GET"), "/mcp": (True, "GET"), "/models/load": (True, "POST"),
    "/models/unload": (True, "POST"), "/models/sse": (True, "GET"), "/api/context": (True, "GET"),
    "/api/thinking-budget": (True, "GET"), "/api/update": (True, "GET"),
    "/v1/responses/{id}": (True, "GET"), "/v1/responses/{id}/input_items": (True, "GET"),
    "/v1/responses/{id}/cancel": (True, "POST"),
}


def guide_table_paths() -> set[str]:
    section = GUIDE.split("## Routes", 1)[1].split("\n## ", 1)[0]
    paths = set()
    for row in section.splitlines():
        if row.startswith("|") and not row.startswith("|---") and not row.startswith("| Method"):
            paths |= set(re.findall(r"`(/[^`]*)`", row.split("|")[2]))
            paths |= set(re.findall(r"; (?:GET|POST) `(/[^`]*)`", row.split("|")[2]))
    return paths


class Guide(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, SCRIPT), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"),
                          model_name="inkling-2-mini")
        cls.svc.api_key = KEY
        cls.svc.cors_origins = [ORIGIN]
        cls.svc.model_management = True
        cls.svc.mcp = McpHub({"fake": {"command": sys.executable, "args": [str(ROOT / "serve/mcp_fake_server.py")]}})
        cls.svc.mcp.start()
        cls.svc.mcp.wait(10)
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.svc.mcp.close()

    def call(self, method, path, headers=None, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            res = conn.getresponse()
            return res.status, res.read(), res
        finally:
            conn.close()

    def bearer(self, key=KEY):
        return {"Authorization": f"Bearer {key}"}

    # ---------------------------------------------------------------------------------------- the route table
    def test_route_table_lists_exactly_the_tested_paths(self):
        self.assertEqual(guide_table_paths(), set(ROUTES))

    def test_only_health_is_public(self):
        for path, (needs_key, method) in ROUTES.items():
            if "{" in path:
                path = path.replace("{id}", "resp_unknown")
            with self.subTest(path=path):
                status, body, _ = self.call(method, path, body=b"{}" if method == "POST" else None,
                                            headers={"Content-Type": "application/json"} if method == "POST" else {})
                if needs_key:
                    self.assertEqual(status, 401, body)
                    self.assertEqual(json.loads(body), {"error": {"type": "authentication_error",
                                                                  "message": "missing or wrong API key"}})
                else:
                    self.assertEqual(status, 200)

    def test_unknown_paths(self):
        self.assertEqual(self.call("GET", "/nope")[0], 404)               # no key needed to learn that
        self.assertEqual(self.call("POST", "/nope", body=b"{}")[0], 401)  # the key comes first
        self.assertEqual(self.call("POST", "/nope", self.bearer(), b"{}")[0], 404)
        self.assertEqual(self.call("DELETE", "/v1/models", self.bearer())[0], 404)

    def test_health_is_readable_without_a_key(self):
        status, body, _ = self.call("GET", "/health?x=1")
        health = json.loads(body)
        self.assertEqual((status, health["status"], health["api_key"], health["model"]), (200, "ok", True, "inkling-2-mini"))
        for field in ("max_context", "version", "images", "videos", "unloaded", "mcp_tools"):
            self.assertIn(field, health)

    # ------------------------------------------------------------------------------------------ authentication
    def test_header_precedence(self):
        ok = lambda headers: self.call("GET", "/v1/models", headers)[0]  # noqa: E731
        self.assertEqual(ok({"Authorization": f"Bearer {KEY}"}), 200)
        self.assertEqual(ok({"Authorization": f"bearer {KEY}"}), 200)
        self.assertEqual(ok({"x-api-key": KEY}), 200)
        self.assertEqual(ok({"Authorization": "Bearer wrong", "x-api-key": KEY}), 401)     # Bearer decides
        self.assertEqual(ok({"Authorization": f"Bearer {KEY}", "x-api-key": "wrong"}), 200)
        self.assertEqual(ok({"Authorization": "Basic abc", "x-api-key": KEY}), 200)        # not Bearer: ignored
        self.assertEqual(ok({"Authorization": KEY}), 401)                                  # no scheme
        self.assertEqual(self.call("GET", f"/v1/models?api_key={KEY}")[0], 401)           # never the query

    def test_several_keys(self):
        self.svc.api_key = "k1,k2"
        try:
            self.assertEqual([self.call("GET", "/v1/models", self.bearer(k))[0] for k in ("k1", "k2", "k1,k2")],
                             [200, 200, 401])
            self.svc.api_key = ["a,b", "c"]
            self.assertEqual([self.call("GET", "/v1/models", self.bearer(k))[0] for k in ("a,b", "c", "a")],
                             [200, 200, 401])
        finally:
            self.svc.api_key = KEY

    def test_responses_are_isolated_by_key_and_shared_across_headers(self):
        self.svc.api_key = "k1,k2"
        try:
            body = json.dumps({"input": "hi", "store": True, "max_output_tokens": 16}).encode()
            _, raw, _ = self.call("POST", "/v1/responses", self.bearer("k1"), body)
            rid = json.loads(raw)["id"]
            self.assertEqual(self.call("GET", f"/v1/responses/{rid}", {"x-api-key": "k1"})[0], 200)
            self.assertEqual(self.call("GET", f"/v1/responses/{rid}", self.bearer("k2"))[0], 404)
        finally:
            self.svc.api_key = KEY

    # --------------------------------------------------------------------------------------------------- CORS
    def test_cors_is_for_v1_and_listed_origins_only(self):
        status, _, res = self.call("OPTIONS", "/v1/chat/completions", {"Origin": ORIGIN})
        self.assertEqual((status, res.getheader("Access-Control-Allow-Origin")), (204, ORIGIN))
        self.assertEqual(self.call("OPTIONS", "/v1/chat/completions", {"Origin": "http://other.example"})[0], 403)
        self.assertEqual(self.call("OPTIONS", "/health", {"Origin": ORIGIN})[0], 403)
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}).encode()
        self.assertEqual(self.call("POST", "/v1/chat/completions", {**self.bearer(), "Origin": "http://other.example"}, body)[0], 403)
        status, _, res = self.call("POST", "/v1/chat/completions", {**self.bearer(), "Origin": ORIGIN}, body)
        self.assertEqual((status, res.getheader("Access-Control-Allow-Origin")), (200, ORIGIN))

    # ---------------------------------------------------------------------------------------------- the routes
    def test_models_ignore_the_requested_model(self):
        _, raw, _ = self.call("GET", "/v1/models", self.bearer())
        entry = json.loads(raw)["data"][0]
        self.assertEqual((entry["id"], entry["status"]["value"], entry["meta"]["n_ctx"]), ("inkling-2-mini", "loaded", 32768))
        body = json.dumps({"model": "something-else", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}).encode()
        status, raw, _ = self.call("POST", "/v1/chat/completions", self.bearer(), body)
        self.assertEqual((status, json.loads(raw)["model"]), (200, "inkling-2-mini"))

    def test_chat_errors(self):
        status, raw, _ = self.call("POST", "/v1/chat/completions", self.bearer(),
                                   json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 999999}).encode())
        self.assertEqual((status, json.loads(raw)["error"]["code"]), (400, "context_length_exceeded"))
        status, raw, _ = self.call("POST", "/v1/chat/completions", self.bearer(),
                                   json.dumps({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "bogus"}).encode())
        self.assertEqual(status, 400)
        self.assertIn("use none, auto, low, medium, high, xhigh or max", json.loads(raw)["error"]["message"])

    def test_reasoning_effort_words(self):
        for word in ("none", "off", "minimal", "low", "medium", "high", "xhigh", "max", "auto"):
            with self.subTest(word=word):
                body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": word, "max_tokens": 64}).encode()
                self.assertEqual(self.call("POST", "/v1/chat/completions", self.bearer(), body)[0], 200)

    def test_messages_take_either_key_header_and_ignore_the_version_header(self):
        body = json.dumps({"max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}).encode()
        for headers in ({"x-api-key": KEY}, self.bearer()):
            self.assertEqual(self.call("POST", "/v1/messages?beta=true", headers, body)[0], 200)
        self.assertEqual(self.call("POST", "/v1/messages", {}, body)[0], 401)

    def test_background_without_store_and_unsupported_parameters(self):
        for extra in ({"background": True}, {"conversation": "x"}, {"strata_mcp": True}):
            status, raw, _ = self.call("POST", "/v1/responses", self.bearer(), json.dumps({"input": "hi", **extra}).encode())
            self.assertEqual(status, 400, raw)

    def test_tokenize_round_trip(self):
        _, raw, _ = self.call("POST", "/tokenize", self.bearer(), b'{"content":"Hello"}')
        out = json.loads(raw)
        self.assertEqual((out["tokens"], out["count"], out["max_model_len"]), ([72, 101, 108, 108, 111], 5, 32768))
        _, raw, _ = self.call("POST", "/detokenize", self.bearer(), b'{"tokens":[72,101]}')
        self.assertEqual(json.loads(raw), {"content": "He", "prompt": "He"})
        status, raw, _ = self.call("POST", "/tokenize", self.bearer(), b'{"messages":[{"role":"user","content":"Hello"}]}')
        self.assertEqual(status, 200)
        self.assertGreater(json.loads(raw)["count"], 5)

    def test_prometheus_names_and_type(self):
        status, raw, res = self.call("GET", "/metrics/prometheus", self.bearer())
        text = raw.decode()
        self.assertEqual((status, res.getheader("Content-Type")), (200, "text/plain; version=0.0.4; charset=utf-8"))
        for name in ("llamacpp:prompt_tokens_total", "llamacpp:tokens_predicted_total", "llamacpp:predicted_tokens_seconds",
                     "llamacpp:requests_processing", "maya:requests_total", "maya:engine_alive",
                     "maya:loop_guard_detected_total", "maya:responses_cache_hits", "maya:responses_jobs_running"):
            self.assertIn(f"# TYPE {name} ", text)

    def test_status_dialects(self):
        _, raw, _ = self.call("GET", "/v1/status", self.bearer())
        self.assertEqual(json.loads(raw)["dialects"], ["/v1/chat/completions", "/v1/responses", "/v1/messages"])

    # ----------------------------------------------------------------------------------------------- MCP scope
    def test_mcp_status_lists_tools_namespaced_by_server(self):
        _, raw, _ = self.call("GET", "/mcp", self.bearer())
        mcp = json.loads(raw)
        self.assertEqual(mcp["servers"][0]["name"], "fake")
        self.assertIn("fake__add", [t["name"] for t in mcp["servers"][0]["tools"]])

    def test_mcp_is_opt_in_and_chat_completions_only(self):
        msg = [{"role": "user", "content": "hi"}]
        asked = json.dumps({"messages": msg, "strata_mcp": True, "max_tokens": 64}).encode()
        json_headers = {**self.bearer(), "Content-Type": "application/json"}
        self.assertEqual(self.call("POST", "/v1/chat/completions", self.bearer(), asked)[0], 415)    # curl's default type
        self.assertEqual(self.call("POST", "/v1/chat/completions", json_headers, asked)[0], 200)
        self.assertEqual(self.call("POST", "/v1/chat/completions", {**json_headers, "Origin": "http://other.example"}, asked)[0], 403)
        self.assertEqual(self.call("POST", "/v1/messages", json_headers,
                                   json.dumps({"messages": msg, "max_tokens": 64, "strata_mcp": True}).encode())[0], 200)

    # --------------------------------------------------------------------------------------------- model state
    def test_model_state_routes(self):
        headers = {**self.bearer(), "Content-Type": "application/json"}
        self.assertEqual(self.call("POST", "/models/load", headers, b"{}")[0], 200)
        status, raw, _ = self.call("POST", "/models/unload", headers, b"{}")
        self.assertEqual((status, "cannot unload" in raw.decode()), (400, True))            # the mock engine cannot
        self.svc.model_management = False
        try:
            self.assertEqual(self.call("POST", "/models/load", headers, b"{}")[0], 404)
        finally:
            self.svc.model_management = True

    # ----------------------------------------------------------------------------------- the guide's own examples
    @unittest.skipUnless(shutil.which("curl") and shutil.which("bash"), "needs curl and bash")
    def test_every_documented_curl_example_runs(self):
        every = [b for b in re.findall(r"```sh\n(.*?)```", GUIDE, re.S) if "curl" in b]
        blocks = [b for b in every if '"$BASE' in b]
        self.assertEqual(len(blocks), len(every), "a curl example that does not use $BASE would not be run")
        self.assertGreaterEqual(len(blocks), 12)
        env = {**os.environ, "BASE": self.base, "KEY": KEY}
        for i, block in enumerate(blocks):
            with self.subTest(block=i, first=block.strip().splitlines()[0][:70]):
                done = subprocess.run(["bash", "-eo", "pipefail", "-c", block], env=env, capture_output=True,
                                      text=True, timeout=60)
                self.assertEqual(done.returncode, 0, done.stderr)

    @unittest.skipUnless(shutil.which("curl"), "needs curl")
    def test_documented_401_example(self):
        done = subprocess.run(["curl", "-s", "-w", " %{http_code}\\n", f"{self.base}/v1/models"], capture_output=True, text=True)
        self.assertEqual(done.stdout.strip(),
                         '{"error": {"type": "authentication_error", "message": "missing or wrong API key"}} 401')
        self.assertIn(done.stdout.strip(), GUIDE.replace("# ", "").replace("\n", " "))


if __name__ == "__main__":
    unittest.main()
