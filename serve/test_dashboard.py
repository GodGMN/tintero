"""HTTP tests for context changes and metrics with a scripted engine."""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import CONTEXT_MIN, MAYA_VERSION, ByteTokenizer, MockEngine, Service, StrataEngine, serve

ROOT = Path(__file__).resolve().parents[1]


class Dashboard(unittest.TestCase):
    def setUp(self):
        os.environ["MAYA_MOCK_RELOAD_S"] = "0.3"
        tok = ByteTokenizer()
        self.engine = MockEngine(tok, "</think>\n\nhello", max_context=32768)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Path(self.tmp.name) / "maya-test.json"
        self.cfg.write_text(json.dumps({"exe": "strata", "args": ["--glm-pack", "/x/pack", "--max-context", "32768"],
                                        "env": {"MAYA_GLM_RAM_GB": "20", "MY_API_KEY": "secret"}}, indent=1))
        self.svc.config_path = str(self.cfg)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def req(self, method, path, body=None, headers=None):
        h = {"Content-Type": "application/json"} if body is not None else {}
        h.update(headers or {})
        r = urllib.request.Request(self.base + path, method=method, headers=h,
                                   data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                raw = resp.read()
                return resp.status, resp.headers, raw
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.headers, e.read()

    def wait_reload(self):
        for _ in range(100):
            s, _, b = self.req("GET", "/api/context")
            rl = json.loads(b)["reload"]
            if rl and rl["state"] in ("done", "failed"):
                return json.loads(b)
            time.sleep(0.05)
        self.fail("the reload did not end")

    def test_health_has_the_version(self):
        s, _, b = self.req("GET", "/health")
        self.assertEqual(json.loads(b)["version"], MAYA_VERSION)

    def test_health_history_thinking(self):
        """A Qwen3.x template keeps past thinking: the web chat is told to send it back (the GLM-5 one: not)."""
        s, _, b = self.req("GET", "/health")
        self.assertIs(json.loads(b)["history_thinking"], True)
        self.svc.template = ChatTemplate(ROOT / "serve/glm5_chat_template.jinja")
        s, _, b = self.req("GET", "/health")
        self.assertIs(json.loads(b)["history_thinking"], False)

    def test_context_range(self):
        s, _, b = self.req("GET", "/api/context")
        j = json.loads(b)
        self.assertEqual((s, j["context"], j["min"]), (200, 32768, CONTEXT_MIN))
        self.assertGreaterEqual(j["max"], 32768)
        self.assertEqual(j["tested"], 262144)                   # (the GLM engine's: the dashboard warns past it)

    def test_reload_changes_the_context_and_saves_it(self):
        s, _, b = self.req("POST", "/api/context", {"max_context": 65536})
        self.assertEqual(s, 202)
        j = self.wait_reload()
        self.assertEqual((j["reload"]["state"], j["context"]), ("done", 65536))
        args = json.loads(self.cfg.read_text())["args"]
        self.assertEqual(args[args.index("--max-context") + 1], "65536")
        self.assertEqual(json.loads(self.cfg.read_text())["env"]["MAYA_GLM_RAM_GB"], "20")   # the rest kept
        self.assertGreaterEqual(j["load_s"], 0.3)       # (how long that load took: the page's estimate of the next)
        self.req("POST", "/api/context", {"max_context": 16384})
        self.wait_reload()
        self.assertEqual(json.loads(self.cfg.read_text())["args"][-1], "16384")

    def test_inkling_context_reload_and_rollback_with_two_caches(self):
        slots = 2
        self.cfg.write_text(json.dumps({"exe": "tintero", "args": ["--max-context", "65536", "--cache-slots", "2"]}))
        self.engine.info.update(engine_kind="ik2", cache_slots=slots)
        self.engine.stop = lambda: None
        self.engine.restart = lambda: None

        def apply_total(total):
            if total > 131072:
                raise RuntimeError("context exceeds device memory")
            self.engine.max_context = total // slots

        self.engine.set_context = apply_total
        status, _, _ = self.req("POST", "/api/context", {"max_context": 49152})
        self.assertEqual(status, 202)
        self.assertEqual(self.wait_reload()["context"], 49152)
        args = json.loads(self.cfg.read_text())["args"]
        self.assertEqual(args, ["--max-context", "98304", "--cache-slots", "2"])
        self.req("POST", "/api/context", {"max_context": 131072})
        result = self.wait_reload()
        self.assertEqual((result["reload"]["state"], result["context"]), ("failed", 49152))
        self.assertEqual(json.loads(self.cfg.read_text())["args"], args)

    def test_metrics_list_the_drafters_that_run(self):
        # the Monitor's Speculation: the drafters this engine runs with, per model engine; empty = drafting is off
        from serve.server import request_spec_off, speculation_modes
        cases = [
            ({"engine_kind": "glm-fast", "mtp": 1, "dflash": 0}, {}, ["MTP"]),
            ({"engine_kind": "glm-fast", "mtp": 1, "dflash": 1}, {}, ["DFlash2"]),      # (DFlash decode takes the request first)
            ({"engine_kind": "glm-fast", "mtp": 0, "dflash": 1}, {}, ["DFlash2"]),
            ({"engine_kind": "glm-fast", "mtp": 0, "dflash": 0}, {}, []),
            ({"engine_kind": "qwen", "mtp": 1, "lookup": 3}, {}, ["MTP", "Lookup"]),
            ({"engine_kind": "qwen", "mtp": 0, "lookup": 0}, {}, []),
            ({"engine_kind": "mm3", "dspark": 1}, {}, ["DSpark"]),
            ({"engine_kind": "mm3", "dspark": 0}, {}, []),
            ({"engine_kind": "hy4", "dspark": 0, "mtp": 0}, {}, []),                   # (maya-hy4: drafting off)
            ({"engine_kind": "hy4", "dspark": 0, "mtp": 0, "dfly2": 1}, {}, ["DFly2"]),
            ({"engine_kind": "hy4", "dspark": 0, "mtp": 0, "dfly2": 0}, {}, []),
            ({"engine_kind": "ds4"}, {"dspark": True, "lookup": True}, ["DSpark", "Lookup"]),
            ({"engine_kind": "ds4"}, {"dspark": False, "lookup": True}, ["Lookup"]),
            ({"engine_kind": "ds4"}, {"dspark": False, "lookup": False}, []),
        ]
        for info, kw, want in cases:
            with self.subTest(info=info, kw=kw):
                self.assertEqual(speculation_modes(info, **kw), want)
        self.assertTrue(request_spec_off({"strata_tune": {"spec": 1}}))
        self.assertFalse(request_spec_off({"strata_tune": {"spec": 4}}))
        self.assertFalse(request_spec_off({}))
        self.assertFalse(request_spec_off(None))
        s, _, b = self.req("GET", "/metrics")
        self.assertEqual(json.loads(b)["engine"]["speculation"], [])               # (the mock engine drafts nothing)
        self.engine.info.update(mtp=1)
        self.assertEqual(json.loads(self.req("GET", "/metrics")[2])["engine"]["speculation"], ["MTP"])

    def test_stat_lines_carry_the_draft_counts_and_the_ssd_bytes(self):
        # the engines' STAT lines (GLM, CYBER-FROST, DeepSeek, MiniMax): live draft_acc / draft_off and disk_gb reach the
        # Monitor's tiers.now as numbers
        class Stub:
            pass
        eng = Stub()
        StrataEngine._parse_stat(eng, "STAT tok_s=31.20 vram_hit=0.9100 vram_used=300 ram_used=500 ram_gb=20.3 "
                                      "vram_gb=12.5 disk_gb=31.4 draft_acc=124 draft_off=200")
        self.assertEqual((eng.stat["draft_acc"], eng.stat["draft_off"], eng.stat["disk_gb"]), (124.0, 200.0, 31.4))
        self.engine.stat = dict(eng.stat)
        now = json.loads(self.req("GET", "/metrics")[2])["tiers"]["now"]
        self.assertEqual((now["draft_acc"], now["draft_off"], now["disk_gb"]), (124.0, 200.0, 31.4))

    def test_requests_carry_the_drafts_offered_and_kept(self):
        # the page's last-answer acceptance: the engine's DONE counts reach the request row, None when it reported none
        self.engine.last = None
        self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        row = json.loads(self.req("GET", "/metrics")[2])["requests"][0]
        self.assertIn("drafts_offered", row)
        self.assertIn("spec_off", row)
        self.assertFalse(row["spec_off"])

    def test_chat_gets_503_while_reloading(self):
        os.environ["MAYA_MOCK_RELOAD_S"] = "1.5"
        self.req("POST", "/api/context", {"max_context": 16384})
        s, h, b = self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(s, 503)
        self.assertEqual(h.get("Retry-After"), "30")
        self.assertIn("reloading", json.loads(b)["error"]["message"])
        s, _, b = self.req("GET", "/health")
        self.assertEqual(json.loads(b)["status"], "reloading")
        self.wait_reload()
        s, _, _ = self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(s, 200)

    def test_out_of_range_and_foreign_pages_are_refused(self):
        s, _, _ = self.req("POST", "/api/context", {"max_context": 100})
        self.assertEqual(s, 400)
        s, _, _ = self.req("POST", "/api/context", {"max_context": 65536}, {"Origin": "http://evil.example"})
        self.assertEqual(s, 403)
        r = urllib.request.Request(self.base + "/api/context", method="POST", data=b"max_context=65536",
                                   headers={"Content-Type": "application/x-www-form-urlencoded"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(r, timeout=10)
        self.assertEqual(cm.exception.code, 415)
        cm.exception.close()

    def test_token_count_matches_the_prompt(self):
        msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hello there"},
                {"role": "assistant", "content": "Hi! How can I help?"}, {"role": "user", "content": "count me"}]
        s, _, b = self.req("POST", "/tokenize", {"messages": msgs, "reasoning_effort": "high"})
        j = json.loads(b)
        from serve.frontend import openai_to_messages
        m, tools, kw = openai_to_messages({"messages": msgs, "reasoning_effort": "high"})
        ids, _, _ = self.svc.prepare(m, tools, kw)
        self.assertEqual((s, j["tokens"]), (200, ids))

    def test_tokenize_and_detokenize(self):
        """POST /tokenize and /detokenize in llama.cpp's and vLLM's forms: a text's ids (its special tokens' text as
        those tokens unless parse_special is false), a chat request's prompt as prepare() makes it, and the text back."""
        tok = ByteTokenizer()
        s, _, b = self.req("POST", "/tokenize", {"content": "hi<|im_end|>"})
        self.assertEqual((s, json.loads(b)["tokens"]), (200, tok.encode("hi<|im_end|>", parse_special=True)))
        s, _, b = self.req("POST", "/tokenize", {"model": "m", "prompt": "hi<|im_end|>", "parse_special": False})
        j = json.loads(b)
        self.assertEqual((j["tokens"], j["count"]), (tok.encode("hi<|im_end|>"), len("hi<|im_end|>")))
        self.assertEqual(j["max_model_len"], self.svc.engine.max_context)
        msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "count me"}]
        s, _, b = self.req("POST", "/tokenize", {"messages": msgs, "reasoning_effort": "high"})
        from serve.frontend import openai_to_messages
        m, tools, kw = openai_to_messages({"messages": msgs, "reasoning_effort": "high"})
        ids, _, _ = self.svc.prepare(m, tools, kw)
        self.assertEqual((s, json.loads(b)["tokens"]), (200, ids))
        s, _, b = self.req("POST", "/detokenize", {"tokens": ids})
        j = json.loads(b)
        self.assertEqual((s, j["content"]), (200, tok.decode(ids)))
        self.assertEqual(j["prompt"], j["content"])
        for path, body in (("/tokenize", {"content": 3}), ("/tokenize", {}), ("/detokenize", {"tokens": [1, -1]}),
                           ("/detokenize", {"tokens": "1,2"})):
            s, _, _ = self.req("POST", path, body)
            self.assertEqual(s, 400, (path, body))

    def test_peak_decode_speed(self):
        # the Monitor's peak decode: the fastest answer since the server started, of PEAK_MIN_TOKENS tokens or more
        from serve.server import PEAK_MIN_TOKENS
        s, _, b = self.req("GET", "/metrics")
        self.assertIsNone(json.loads(b)["totals"]["peak_decode"])
        end = self.engine.script[-1:]                                  # (its end-of-turn token)
        self.engine.script = self.engine.scripts[0] = ByteTokenizer().encode("</think>\n\n" + "a long enough answer " * 3) + end
        self.engine.tok_s = 4000
        rows = []
        for _ in range(3):
            self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
            rows.append(json.loads(self.req("GET", "/metrics")[2])["requests"][0])
        peak = json.loads(self.req("GET", "/metrics")[2])["totals"]["peak_decode"]
        top = max(rows, key=lambda r: r["decode_tok_s"])
        self.assertEqual((peak["tok_s"], peak["tokens"], peak["time"]), (top["decode_tok_s"], top["output_tokens"], top["time"]))
        self.assertGreaterEqual(peak["tokens"], PEAK_MIN_TOKENS)
        _, _, b = self.req("GET", "/metrics/prometheus")
        self.assertIn(f"maya:decode_peak_tokens_seconds {peak['tok_s']}", b.decode())
        # a short answer, however fast, is not the peak
        self.engine.tok_s = 400000
        self.req("POST", "/v1/chat/completions", {"model": "m", "max_tokens": 5,
                                                  "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(json.loads(self.req("GET", "/metrics")[2])["totals"]["peak_decode"], peak)

    def test_peak_decode_follows_the_live_figure(self):
        # a live Decode reading above the peak is the peak in the same /metrics answer (the owner's screenshot: 30.0
        # live, peak 27.8), rounded as shown; a reading before PEAK_MIN_TOKENS tokens is not
        from serve.server import PEAK_MIN_TOKENS, RequestStat
        svc, now = self.svc, time.time()
        svc.totals["peak_decode"] = {"tok_s": 27.8, "tokens": 400, "time": now - 60}
        r = RequestStat()
        r.first_token = now - 10
        with svc.status_lock:
            svc.status.update(busy=True, generated=PEAK_MIN_TOKENS - 1)
            svc.current_requests.add(r)
            svc.latest_request = r
        try:
            r.base = (now - 1, 0); r.samples.append((now, PEAK_MIN_TOKENS - 1))   # 31 tok/s, but only 31 tokens
            m = json.loads(self.req("GET", "/metrics")[2])
            self.assertEqual(m["totals"]["peak_decode"]["tok_s"], 27.8)
            for k, n in enumerate((300, 330, 310)):                   # 300, 330, 310 tok/s live, one a reading
                r.samples.clear(); r.base = (time.time() - 1, 1000 * k); r.samples.append((time.time(), 1000 * k + n))
                m = json.loads(self.req("GET", "/metrics")[2])
                self.assertGreaterEqual(m["totals"]["peak_decode"]["tok_s"], m["live"]["tok_s"])
            self.assertAlmostEqual(m["totals"]["peak_decode"]["tok_s"], 330, delta=2)
            self.assertEqual(m["totals"]["peak_decode"]["tokens"], 1330)
            self.assertEqual(round(m["totals"]["peak_decode"]["tok_s"], 1), m["totals"]["peak_decode"]["tok_s"])
        finally:
            with svc.status_lock:
                svc.status.update(busy=False)
                svc.current_requests.discard(r)

    def test_requests_record_their_api(self):
        self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        s, _, b = self.req("GET", "/metrics")
        self.assertEqual(json.loads(b)["requests"][0]["api"], "openai")


class ContextCost(unittest.TestCase):
    """The GLM engine's start lines give what its context costs in VRAM, summed over its GPUs."""

    def test_parsed_from_the_start_log(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.write_text("old start\nglm fast: CUDA0 VRAM before the expert pool: 4.0 of 31.7 GB in use - dense weights 3.1, "
                           "state/KV (16384 ctx) 0.40, activations 0.1\n"
                           "this start\nglm fast: CUDA0 VRAM before the expert pool: 4.4 of 31.7 GB in use - dense weights 3.1, "
                           "state/KV (32768 ctx) 0.66, activations 0.1\n"
                           "glm fast: CUDA1 VRAM before the expert pool: 4.3 of 31.7 GB in use - dense weights 2.9, "
                           "state/KV (32768 ctx) 0.65, activations 0.1\n")
            eng = StrataEngine.__new__(StrataEngine)
            eng.log_path, eng.info = str(log), {}
            eng._context_cost(log.read_text().index("this start"))
            self.assertEqual((eng.info["kv_ctx"], eng.info["kv_gb"]), (32768, 1.31))

    def test_set_context_rewrites_the_command(self):
        eng = StrataEngine.__new__(StrataEngine)
        eng.spawn = ("strata", ["--glm-pack", "p", "--max-context", "32768"], None, None, None)
        eng.set_context(131072)
        self.assertEqual(eng.spawn[1], ["--glm-pack", "p", "--max-context", "131072"])


GB = 2**30
# the 3090 alone with EXL3-4.05bpw and DFlash2, as its start measured it (glm fast: CUDA0 VRAM before the expert pool ...)
LINE = ("VRAM dev=0 pci=0000:01:00.0 total=25296896000 free=751828992 weights=6678575104 kv=665719603 kv_ram=0 "
        "act=107374182 drafter=751828992 window=246415360 pool=15536793108 tail=7945689497 borrow=7945689497 "
        "vision=0 slots=1231 ctx=32768 layers=0-45\n")


class GpuMap(unittest.TestCase):
    """What fills each GPU: the engine's VRAM lines, joined with NVML's cards and processes (serve/gpu_map.py)."""

    def setUp(self):
        from serve import gpu_map
        self.gm = gpu_map
        self.r = gpu_map.parse(LINE)

    def card(self, used, procs, index=0, pci="0000:01:00"):
        return {"index": index, "pci": pci, "name": "NVIDIA GeForce RTX 3090", "used": used, "total": 25296896000,
                "procs": procs}

    def test_parse(self):
        self.assertEqual((self.r["dev"], self.r["pci"], self.r["layers"], self.r["slots"]), (0, "0000:01:00", [0, 45], 1231))
        self.assertEqual(self.r["pool"], 15536793108)
        self.assertIsNone(self.gm.parse("INFO context=32768"))
        self.assertEqual(self.gm.pci_key("00000000:0A:00.0"), "0000:0a:00")

    def test_helper_pool_is_accounted_separately_from_drafter_and_rest(self):
        helper = 20_000_000_000
        r = self.gm.parse(f"VRAM dev=2 pci=0000:0f:00.0 total={32 * GB} free={5 * GB} "
                          f"weights=0 kv=0 act=0 drafter={4 * GB} pool=0 helper={helper} slots=0 layers=-1--1")
        (g,) = self.gm.view([r], [], engine_pid=100)
        self.assertEqual(g["parts"]["helper"], helper)
        self.assertEqual(g["parts"]["drafter"], 4 * GB)
        self.assertEqual(g["parts"]["rest"], 23 * GB - helper)
        self.assertIsNone(g["layers"])
        self.assertEqual(sum(g["parts"].values()) + g["free"], g["total"])
        merged = self.gm.merge([r, dict(dev=2, helper=1024, free=5 * GB)])
        self.assertEqual(merged[0]["helper"], helper + 1024)

    def test_helper_pool_and_primary_pool_on_one_card_do_not_overlap(self):
        r = dict(self.r, helper=2 * GB)
        engine = sum(r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool", "helper"))
        card = dict(self.card(engine + GB, {100: engine + GB}), total=32 * GB)
        (g,) = self.gm.view([r], [card], engine_pid=100)
        self.assertEqual(g["parts"]["helper"], 2 * GB)
        self.assertEqual(g["parts"]["pool"], r["pool"])
        self.assertEqual(g["parts"]["rest"], GB)
        self.assertEqual(sum(g["parts"].values()) + g["free"], g["total"])

    def test_nvml_tells_the_engine_from_other_programs(self):
        engine = sum(self.r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool"))
        own, other = engine + 400 * 2**20, 600 * 2**20
        (g,) = self.gm.view([self.r], [self.card(own + other, {100: own, 200: other})], engine_pid=100)
        self.assertEqual((g["index"], g["measured"]), (0, "nvml"))
        self.assertEqual(g["parts"]["rest"], 400 * 2**20)        # the CUDA context and the small buffers
        self.assertEqual(g["parts"]["other"], other)
        self.assertEqual(g["parts"]["drafter"], self.r["drafter"] + self.r["window"])
        self.assertEqual(g["free"], g["total"] - own - other)
        self.assertEqual(sum(g["parts"].values()) + g["free"], g["total"])

    def test_the_drivers_reserve_is_not_another_program(self):
        # (9 GPUs, the engine alone on each: NVML's used memory ~0.58 GB beyond its process - the driver's reserve)
        engine = sum(self.r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool"))
        own, reserve = engine + 500 * 2**20, 590 * 2**20
        (g,) = self.gm.view([self.r], [self.card(own + reserve, {100: own})], engine_pid=100)
        self.assertEqual((g["parts"]["other"], g["parts"]["rest"]), (0, 500 * 2**20 + reserve))

    def test_found_by_pci_not_by_place(self):
        cards = [self.card(1 * GB, {}, index=0, pci="0000:02:00"), self.card(23 * GB, {100: 22 * GB}, index=1)]
        (g,) = self.gm.view([self.r], cards, engine_pid=100)
        self.assertEqual(g["index"], 1)

    def test_without_nvml_the_start_measurement(self):
        (g,) = self.gm.view([self.r], [], engine_pid=100)
        start = self.r["total"] - self.r["free"]
        engine = sum(self.r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool"))
        self.assertEqual((g["measured"], g["parts"]["rest"], g["parts"]["other"]), ("start", start - engine, 0))
        self.assertEqual(g["free"], self.r["free"])

    def test_the_vision_encoder_has_the_tail(self):
        r = dict(self.r, vision=self.r["tail"])
        engine = sum(r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool")) - r["tail"]
        own, enc = engine + 400 * 2**20, r["tail"] - 50 * 2**20
        (g,) = self.gm.view([r], [self.card(own + enc, {100: own, 300: enc})], 100, vision_pid=300, vision_lent=True)
        self.assertTrue(g["vision_lent"])
        self.assertEqual((g["parts"]["pool"], g["parts"]["vision"]), (r["pool"] - r["tail"], enc))
        self.assertEqual(g["parts"]["other"], 0)

    def test_a_vision_tower_in_the_engines_process(self):
        # (strata-ds4: the tower on the first GPU inside the engine - vision_own, not a pool tail lent to another process)
        r = dict(self.r, vision_own=318 * 2**20)
        engine = sum(r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool"))
        own = engine + r["vision_own"] + 400 * 2**20
        (g,) = self.gm.view([r], [self.card(own, {100: own})], engine_pid=100)
        self.assertEqual((g["parts"]["vision"], g["vision_own"]), (r["vision_own"], r["vision_own"]))
        self.assertEqual((g["parts"]["rest"], g["parts"]["other"]), (400 * 2**20, 0))   # the CUDA context, not vision
        (s,) = self.gm.view([r], [], engine_pid=100)             # (no NVML: the start's measurement)
        self.assertEqual(s["parts"]["vision"], r["vision_own"])
        self.assertEqual(sum(s["parts"].values()) + s["free"], s["total"])

    def test_an_engine_run_through_a_launcher(self):
        # (the engine's command a script that runs `nsys launch` (pid 63), which starts the engine (85) as a grandchild:
        # NVML lists 85 - found among the launcher's tree; the launcher's pid alone found nothing, and the engine's own
        # growth since the start's measurement was drawn as other programs)
        engine = sum(self.r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool"))
        own, reserve, other = engine + 500 * 2**20, 128 * 2**20, 600 * 2**20
        tree = {63, 84, 85}
        (g,) = self.gm.view([self.r], [self.card(own + reserve, {85: own})], engine_pid=tree)
        self.assertEqual((g["measured"], g["parts"]["other"], g["parts"]["rest"]), ("nvml", 0, 500 * 2**20 + reserve))
        (g,) = self.gm.view([self.r], [self.card(own + reserve + other, {85: own, 200: other})], engine_pid=tree)
        self.assertEqual((g["parts"]["other"], g["parts"]["rest"]), (other, 500 * 2**20 + reserve))   # a real one
        r = dict(self.r, vision=self.r["tail"])                  # (the on-demand encoder, run through one too)
        enc = r["tail"] - 50 * 2**20
        own = engine - r["tail"] + 400 * 2**20
        (g,) = self.gm.view([r], [self.card(own + enc, {85: own, 301: enc})], tree, vision_pid={300, 301},
                            vision_lent=True)
        self.assertEqual((g["parts"]["vision"], g["parts"]["other"], g["parts"]["rest"]), (enc, 0, 400 * 2**20))

    def test_the_engine_not_listed_the_start_measurement(self):
        # (NVML lists the engine under a pid the server can't name - a pid namespace it does not share: the card is
        # drawn from the start's measurement, as without NVML)
        engine = sum(self.r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool"))
        (g,) = self.gm.view([self.r], [self.card(engine + GB, {450462: engine + GB})], engine_pid=65)
        start = self.r["total"] - self.r["free"]
        self.assertEqual((g["measured"], g["parts"]["rest"]), ("start", start - engine))

    def test_never_more_than_the_card(self):
        (g,) = self.gm.view([self.r], [self.card(26 * GB, {100: 25 * GB, 200: 2 * GB})], engine_pid=100)
        self.assertLessEqual(sum(g["parts"].values()), g["total"])
        self.assertEqual(g["free"], 0)

    def test_strata_ds4s_prompt_buffers_and_tensor_parallel(self):
        # (strata-ds4 at --tp 8: a 5060 Ti of the group of 8 - its prompt chunks' buffers and tensor parallel's copies
        # told apart, the GLM engine's lines without them)
        r = self.gm.parse("VRAM dev=3 pci=0000:05:00.0 total=16610951168 free=301989888 weights=268435456 kv=33554432 "
                          "kv_ram=0 act=134217728 drafter=0 window=0 pool=13421772800 tail=0 borrow=0 vision=0 "
                          "vision_own=0 prompt=1610612736 tp=805306368 tp_n=8 slots=3343 ctx=262144 layers=3-39")
        (g,) = self.gm.view([r], [], engine_pid=100)
        self.assertEqual((g["parts"]["prompt"], g["parts"]["tp"], g["tp_n"]), (1610612736, 805306368, 8))
        self.assertEqual(sum(g["parts"].values()) + g["free"], g["total"])
        (h,) = self.gm.view([self.r], [], engine_pid=100)
        self.assertEqual((h["parts"]["prompt"], h["parts"]["tp"], h["tp_n"]), (0, 0, 0))

    def test_tensor_parallel_room_is_the_groups_least_free(self):
        # (a --tp 2 pair holds the same expert slots: the 8 GB GPU 1 has beyond GPU 0's 1 GB give no expert more)
        def line(dev, free, layers):
            return self.gm.parse(f"VRAM dev={dev} pci=0000:0{dev + 1}:00.0 total={24 * GB} free={free} weights={GB} "
                                 f"kv=0 kv_ram=0 act=0 drafter=0 window=0 pool={10 * GB} tail=0 borrow=0 vision=0 "
                                 f"vision_own=0 prompt=0 tp=0 tp_n=2 slots=2000 ctx=65536 layers={layers}")
        a, b, c, d = self.gm.view([line(0, GB, "0-20"), line(1, 9 * GB, "0-20"), line(2, 5 * GB, "21-42"),
                                   line(3, 3 * GB, "21-42")], [], engine_pid=100)
        self.assertEqual((a["tp_group"], b["tp_group"], c["tp_group"], d["tp_group"]), (0, 0, 2, 2))
        self.assertEqual((a["tp_room"], b["tp_room"]), (a["free"], a["free"]))
        self.assertGreater(b["free"] - b["tp_room"], 7 * GB)
        self.assertEqual((c["tp_room"], d["tp_room"]), (d["free"], d["free"]))
        (h,) = self.gm.view([self.r], [], engine_pid=100)        # (no group: as before)
        self.assertNotIn("tp_room", h)

    def test_two_parts_on_one_card_add_up(self):
        a = dict(self.r, layers=[0, 20], slots=500)
        b = dict(self.r, layers=[20, 45], slots=700, free=self.r["free"] - GB)
        (m,) = self.gm.merge([a, b])
        self.assertEqual((m["layers"], m["slots"], m["pool"], m["free"]), ([0, 45], 1200, 2 * self.r["pool"], b["free"]))

    def test_metrics_carry_it(self):
        from types import SimpleNamespace
        tok = ByteTokenizer()
        eng = MockEngine(tok, "hi", max_context=32768)
        eng.vram, eng.proc = [self.r], SimpleNamespace(pid=100)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        card = self.card(24 * GB, {100: 23 * GB})
        svc.telemetry = SimpleNamespace(snapshot=lambda: {"now": {"gpu_mem_used": 24 * GB, "vram": [card]},
                                                          "history": {}, "static": {}})
        m = svc.metrics()
        self.assertNotIn("vram", m["hardware"])                  # (the processes stay in the server)
        self.assertEqual(m["gpu_map"][0]["slots"], 1231)
        json.dumps(m)
        eng.vram = []
        self.assertIsNone(svc.metrics()["gpu_map"])             # (an engine without VRAM lines: no card)

    def test_later_lines_follow_a_picture(self):
        """maya-ds4 prints its VRAM lines again when a picture's room is lent to its vision tower and when it is given
        back: the map of that GPU follows them, the others stay."""
        from types import SimpleNamespace
        from serve.server import StrataEngine
        other = dict(self.r, dev=1, pci="0000:02:00")
        eng = SimpleNamespace(vram=[self.r, other])
        lent = LINE.replace("vision=0", "vision=0 vision_own=334495744").replace("pool=15536793108", "pool=15202297364")
        StrataEngine._vram_line(eng, lent)
        self.assertEqual((eng.vram[0]["vision_own"], eng.vram[0]["pool"], eng.vram[1]), (334495744, 15202297364, other))
        StrataEngine._vram_line(eng, LINE)
        self.assertEqual(eng.vram, [self.r, other])
        StrataEngine._vram_line(eng, "VRAM nonsense\n")      # (not a GPU's line: nothing changes)
        self.assertEqual(eng.vram, [self.r, other])


class GpuRelease(unittest.TestCase):
    """A reload starts the new engine only once the GPUs have dropped the old one (its memory is freed after it ends)."""

    def test_waits_until_no_gpu_lists_the_engine(self):
        from unittest import mock
        from serve import server as SV, telemetry as T
        seen = [{4242, 7}, {4242, 7}, {7}]
        with mock.patch.object(T, "gpu_pids", side_effect=lambda: seen.pop(0) if len(seen) > 1 else seen[0]), \
                mock.patch.object(SV.time, "sleep") as sleep, \
                mock.patch.object(SV.sys, "platform", "linux"):
            SV.wait_gpu_release(4242, timeout=30)
        self.assertEqual(seen, [{7}])
        self.assertEqual(sleep.call_count, 2)

    def test_without_nvml_a_short_wait(self):
        from unittest import mock
        from serve import server as SV, telemetry as T
        with mock.patch.object(T, "gpu_pids", return_value=None), mock.patch.object(SV.time, "sleep") as sleep, \
             mock.patch.object(SV.sys, "platform", "linux"):
            SV.wait_gpu_release(4242)
        sleep.assert_called_once()

    def test_mac_waits_until_free_memory_stops_rising(self):
        from unittest import mock
        from serve import server as SV
        gb = 1 << 30
        free = [4 * gb, 20 * gb, 40 * gb, 40 * gb, 90 * gb]
        with mock.patch.object(SV, "mac_free_bytes", side_effect=lambda: free.pop(0)), \
                mock.patch.object(SV.time, "sleep") as sleep, mock.patch.object(SV.sys, "platform", "darwin"):
            SV.wait_gpu_release(4242)
        self.assertEqual(sleep.call_count, 3)                   # rising, rising, settled: started
        self.assertEqual(free, [90 * gb])

    def test_mac_free_bytes_reads_vm_stat(self):
        from unittest import mock
        from serve import server as SV
        out = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:      10.\n"
               "Pages active:   99.\nPages inactive:  20.\nPages speculative:  3.\nPages purgeable:  2.\n")
        with mock.patch.object(SV.subprocess, "run", return_value=mock.Mock(stdout=out)):
            self.assertEqual(SV.mac_free_bytes(), 35 * 16384)
        with mock.patch.object(SV.subprocess, "run", return_value=mock.Mock(stdout="")):
            self.assertIsNone(SV.mac_free_bytes())


class NvmlPids(unittest.TestCase):
    """The GPU map finds an engine run through a launcher: NVML lists a process the launcher started, not it."""

    def test_its_own_pid_when_listed_or_nothing_is(self):
        from unittest import mock
        from serve import server as SV
        with mock.patch.dict(sys.modules, {"psutil": None}):     # (nothing is looked up then)
            self.assertEqual(SV.nvml_pids(85, {85, 200}), {85})
            self.assertEqual(SV.nvml_pids(63, set()), {63})
            self.assertEqual(SV.nvml_pids(None, {85}), set())
            self.assertEqual(SV.nvml_pids(63, {85}), {63})        # (no psutil: the pid alone)

    def test_the_processes_a_launcher_started(self):
        try:
            import psutil
        except ImportError:
            self.skipTest("psutil is not installed")
        import subprocess
        from serve import server as SV
        # (a launcher that does not exec: python starting python, which sleeps)
        launcher = subprocess.Popen([sys.executable, "-c", "import subprocess, sys; subprocess.run([sys.executable, "
                                     "'-c', 'import sys, time; print(flush=True); time.sleep(60)'])"],
                                    stdout=subprocess.PIPE, text=True)
        try:
            launcher.stdout.readline()                           # (the grandchild runs)
            pids = SV.nvml_pids(launcher.pid, {1 << 30})
            self.assertIn(launcher.pid, pids)
            self.assertGreater(len(pids), 1)                     # (and the grandchild: a venv's on Windows, more)
        finally:
            for p in psutil.Process(launcher.pid).children(recursive=True):
                p.kill()
            launcher.kill()
            launcher.wait()
            launcher.stdout.close()


if __name__ == "__main__":
    unittest.main()
