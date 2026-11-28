"""serve/test_inkling2_frontend.py - Inkling-2-Small in the frontend and the server: thinkingmachines' own chat
template detected and rendered at every level a client names (its effort_map's names, thinking off as "none"), the
messages it refuses as clients send them (a tool call's arguments as a JSON string, None content) taken, its answers
- typed segments, not tags - read into reasoning, answer and tool calls however the stream is cut, its end of turn
a stop, a required tool call opened in its own form, and pictures refused in plain words (tintero has no vision
tower yet).

    python -m unittest serve.test_inkling2_frontend

Offline, no GPU: the template is tools/ik2/chat_template.jinja, the tokenizer a small stand-in with the template's
marks as single tokens, the engine a scripted one.
"""
from __future__ import annotations

import hashlib
import json
import threading
import unittest
import urllib.request
from pathlib import Path

from serve import server
from serve.frontend import ChatTemplate, InklingRewriter, OutputParser, effort_kwargs, inkling_messages, \
    openai_to_messages
from serve.server import ByteTokenizer, Service

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "tools" / "ik2" / "chat_template.jinja"
PUBLIC_SHA256 = "e9cfd8e75d1272633c2e72b3cc76e99567e2bc8912b960483aed483ad35d54c7"
USER, MODEL, SYSTEM, TOOL = "<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>"
THINK, TEXT, CALL, END = "<|content_thinking|>", "<|content_text|>", "<|content_invoke_tool_json|>", "<|end_message|>"
EOS = "<|content_model_end_sampling|>"
EFFORT = "<|message_system|><|content_text|>Thinking effort level: "
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "The weather of a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}]
ASK = [{"role": "user", "content": "Weather in Paris?"}]
PICTURE = {"role": "user", "content": [{"type": "text", "text": "What is this?"},
                                       {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}


def public() -> ChatTemplate:
    return ChatTemplate(PUBLIC)


def call(name, args) -> str:
    """A tool call as the model writes it (and its template renders a past one)."""
    return MODEL + name + CALL + json.dumps({"name": name, "args": args}, separators=(",", ":")) + END


class InkTok(ByteTokenizer):
    """ByteTokenizer with the template's marks as one token each; its eos the end of sampling (the GGUF's)."""
    SPECIALS = ByteTokenizer.SPECIALS + [USER, MODEL, SYSTEM, TOOL, THINK, TEXT, CALL, END, EOS, "<|content_xml|>",
                                         "<|content_image|>"]
    special_tokens = {s: 256 + k for k, s in enumerate(SPECIALS)}
    special_ids = {"eos": special_tokens[EOS]}


class Scripted:
    """Request k's answer is scripts[k] (the last one again after that), then the end of sampling.  Records every
    prompt.  Like tintero: no logprobs, no logit bias, one conversation."""

    def __init__(self, tok, *scripts):
        self.tok, self.max_context, self.scripts = tok, 32768, scripts
        self.prompts, self.sampling, self.generated = [], [], []
        self.info = {"engine": "ik2", "engine_kind": "ik2"}

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        self.sampling.append(dict(sampling or {}))
        text = self.scripts[min(len(self.prompts), len(self.scripts)) - 1]
        self.generated.append([])
        for t in (self.tok.encode(text, parse_special=True) + [self.tok.special_tokens[EOS]])[:max_new]:
            self.generated[-1].append(t)
            yield t


def service(*scripts, model_name="inkling-2-small"):
    tok = InkTok()
    eng = Scripted(tok, *scripts)
    return Service(eng, tok, public(), model_name=model_name), eng, tok


def answer(svc, req):
    msgs, tools, kw = openai_to_messages(req)
    ids, thinking, max_new = svc.prepare(msgs, tools, kw, req.get("max_tokens", 500))
    sampling = {"thinking_budget": req.get("thinking_budget")}
    sampling.update({k: kw.get(k) for k in ("_prefill", "_blocked", "_think_resume")})
    events, done = [], None
    for kind, x in svc.run(ids, thinking, tools, max_new, sampling, threading.Event()):
        if kind == "event":
            events.append(x)
        elif kind == "done":
            done = x
    return {"content": "".join(e.text for e in events if e.kind == "content"),
            "reasoning": "".join(e.text for e in events if e.kind == "reasoning"),
            "calls": [(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"], "done": done}


class Template(unittest.TestCase):
    def test_public_copy(self):   # (line endings aside: a checkout may rewrite them)
        raw = PUBLIC.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(raw).hexdigest(), PUBLIC_SHA256)

    def test_detected(self):
        t = public()
        self.assertTrue(t.inkling)
        self.assertTrue(t.keeps_thinking)
        self.assertFalse(t.minimax or t.deepseek or t.qwen or t.hy4 or t.glm_effort or t.glm_dsa)

    def test_other_templates_are_not(self):
        for path in ("serve/chat_template.jinja", "serve/glm5_chat_template.jinja"):
            self.assertFalse(ChatTemplate(ROOT / path).inkling, path)

    def test_levels(self):
        # its own names reach it as they are; thinking off is its "none" (0); none named: its default, 0.9
        t = public()
        for word, num in (("none", "0"), (False, "0"), ("off", "0"), ("minimal", "0"), ("low", "0.2"),
                          ("medium", "0.7"), ("high", "0.9"), ("xhigh", "0.99"), ("max", "0.99"), ("auto", "0.9"),
                          (None, "0.9")):
            with self.subTest(word=word):
                out = t.render(ASK, **effort_kwargs(word))
                self.assertIn(EFFORT + num + END, out)
                self.assertFalse(t.opens_thinking(out))   # (the model opens its own segments)

    def test_render(self):
        out = public().render([{"role": "system", "content": "Be terse."}] + ASK)
        self.assertEqual(out, SYSTEM + TEXT + "Be terse." + END + EFFORT + "0.9" + END + USER + TEXT +
                         "Weather in Paris?" + END)

    def test_history_as_clients_send_it(self):
        # a past call's arguments as a JSON string (OpenAI's form) and its content None: the template refuses both
        # as given - they go in as an object and ""; the same prompt as the canonical form
        t = public()
        flat = [x["function"] for x in TOOLS]
        tc = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"days": 2, "city": "Paris"}'}}
        given = ASK + [{"role": "assistant", "content": None, "reasoning_content": "Look it up.", "tool_calls": [tc]},
                       {"role": "tool", "tool_call_id": "c1", "content": '{"temp": 21}'}]
        canon = json.loads(json.dumps(given))
        canon[1]["content"] = ""
        canon[1]["tool_calls"][0]["function"]["arguments"] = {"city": "Paris", "days": 2}
        out = t.render(given, tools=flat)
        self.assertEqual(out, t.template.render(messages=canon, tools=TOOLS, add_generation_prompt=True))
        self.assertIn('<|message_system|>tool_declare<|content_xml|>[{"description":"The weather of a city."', out)
        self.assertIn(MODEL + THINK + "Look it up." + END + call("get_weather", {"city": "Paris", "days": 2}) + EOS,
                      out)
        self.assertIn(TOOL + "get_weather" + TEXT + '{"temp": 21}' + END, out)

    def test_messages(self):
        got = inkling_messages([{"role": "developer", "content": "x"},
                                {"role": "assistant", "content": None, "reasoning": "r", "tool_calls": [
                                    {"function": {"name": "f", "arguments": ""}},
                                    {"function": {"name": "g", "arguments": "not json"}}]}])
        self.assertEqual(got[0], {"role": "system", "content": "x"})
        self.assertEqual((got[1]["content"], got[1]["reasoning_content"]), ("", "r"))
        self.assertEqual([c["function"]["arguments"] for c in got[1]["tool_calls"]], [{}, {"_raw": "not json"}])


class Parser(unittest.TestCase):
    OUT = {
        "think+text": (MODEL + THINK + "Let me see." + END + MODEL + TEXT + "The answer is 4." + END + EOS,
                       "Let me see.", "The answer is 4.", []),
        "text only": (MODEL + TEXT + "Hi there!" + END + EOS, "", "Hi there!", []),
        "plain": ("Just words.", "", "Just words.", []),
        "think+call": (MODEL + THINK + "Need it." + END +
                       call("get_weather", {"city": "a <b> \"c\"", "days": 3, "opts": {"x": [1, 2]}}) + EOS,
                       "Need it.", "", [("get_weather", {"city": "a <b> \"c\"", "days": 3, "opts": {"x": [1, 2]}})]),
        "text+call": (MODEL + TEXT + "Looking." + END + call("get_weather", {"city": "Rome"}) + EOS,
                      "", "Looking.", [("get_weather", {"city": "Rome"})]),
    }

    def feed(self, text, thinking, step):
        p, events = OutputParser(thinking=thinking, tools=[t["function"] for t in TOOLS], stream_tools=True,
                                 inkling=True), []
        for i in range(0, len(text), step):
            events += p.feed(text[i:i + step])
        events += p.finish()
        return ("".join(e.text for e in events if e.kind == "reasoning"),
                "".join(e.text for e in events if e.kind == "content").strip(),
                [(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"])

    def test_however_cut(self):
        for name, (text, reasoning, content, calls) in self.OUT.items():
            for thinking in (False, True):
                for step in (1, 2, 3, 7, 1000):
                    with self.subTest(name=name, thinking=thinking, step=step):
                        self.assertEqual(self.feed(text, thinking, step), (reasoning, content, calls))

    def test_recover_call_envelope(self):
        args = {"city": 'Paris, "args": {}', "days": 0, "opts": [None, False, {"x": "<b>"}]}
        value = json.dumps(args)
        for body in (
            '{"name":"get_weather""args":' + value + '}',
            '{"name":"get_weather" ,,"args":' + value + '}',
            '{"name":"get_weather", "args"' + value + '}',
            '{"name":"get_weather","args"::' + value + '}',
            '{"name":"get_weather","arguments":' + value + '} trailing',
            '{"name":"get_weather","arguments":' + json.dumps(value) + '}',
            '{"name":"get_weather","args":' + value,
            '{"name":"get_weather","args":' + value + ',}',
            value,
        ):
            with self.subTest(body=body):
                self.assertEqual(self.feed(MODEL + "get_weather" + CALL + body + END, False, 2),
                                 ("", "", [("get_weather", args)]))

    def test_invalid_arguments_stay_text(self):
        for body in ('{"name":"get_weather"}', 'null', '{"args":false}', '{"args":[]}', '{"args":null}',
                     '{"name":"get_weather""args":{"city":}}', '{"arguments":"{"}',
                     '{"args":{},"arguments":{}}', '{"args":{},"args":{"city":"Paris"}}',
                     '{"args":{}}{"args":{"city":"Paris"}}'):
            with self.subTest(body=body):
                self.assertEqual(self.feed(MODEL + "get_weather" + CALL + body + END, True, 1),
                                 ("", body, []))

    def test_unfinished_call_is_text(self):
        r = InklingRewriter(thinking=False)
        out = r.feed(MODEL + "get_weather" + CALL + '{"name":"get_weather","args":{"ci') + r.finish()
        self.assertEqual(out, '{"name":"get_weather","args":{"ci')


class Serving(unittest.TestCase):
    def test_budget_counts_payload_across_segments(self):
        head = MODEL + THINK + "a" + END + MODEL + THINK + "bc"
        for guard in (True, False):
            with self.subTest(guard=guard):
                svc, eng, tok = service(head + "def", "OK" + END, model_name="inkling-2-mini")
                svc.loop_guard = guard
                svc.think_budget = 3
                self.assertTrue(svc.think_budget_info()["supported"])
                got = answer(svc, {"messages": ASK})
                self.assertEqual((got["reasoning"], got["content"]), ("abc", "OK"))
                self.assertEqual(eng.prompts[1], eng.prompts[0] + tok.encode(head + END + MODEL + TEXT,
                                                                            parse_special=True))

    def test_budget_preserves_natural_transition_and_disabled_requests(self):
        script = MODEL + THINK + "why" + END + MODEL + TEXT + "OK" + END
        for fields, default in (({}, 0), ({"thinking_budget": None}, 0), ({"thinking_budget": 4}, 1),
                                ({"thinking_budget": 0}, 1)):
            with self.subTest(fields=fields):
                svc, eng, _ = service(script)
                svc.think_budget = default
                got = answer(svc, {"messages": ASK, **fields})
                self.assertEqual((got["reasoning"], got["content"]), ("why", "OK"))
                self.assertEqual(len(eng.prompts), 1)
        svc, eng, _ = service("OK" + END)
        svc.think_budget = 1
        got = answer(svc, {"messages": ASK, "reasoning_effort": "none"})
        self.assertEqual((got["reasoning"], got["content"], len(eng.prompts)), ("", "OK", 1))

    def test_budget_reserves_room_for_answer(self):
        svc, eng, _ = service(MODEL + THINK + "abcdefghijklmnopqrstuvwxyz", "OK" + END)
        got = answer(svc, {"messages": ASK, "thinking_budget": 1000, "max_tokens": 20})
        self.assertEqual((got["reasoning"], got["content"]), ("abcdefghij", "OK"))
        self.assertEqual(got["done"]["reasoning_tokens"], 10)
        self.assertLessEqual(got["done"]["completion_tokens"], 20)
        self.assertEqual(len(eng.prompts), 2)

    def test_budget_over_http(self):
        for path in ("chat/completions", "messages", "responses"):
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    svc, eng, _ = service(MODEL + THINK + "abcdef", "OK" + END, model_name="inkling-2-mini")
                    httpd = server.serve(svc, port=0)
                    try:
                        body = {"input": "Hi", "max_output_tokens": 20} if path == "responses" else {
                            "messages": ASK, "max_tokens": 20}
                        body.update(thinking_budget=3, stream=stream)
                        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_port}/v1/{path}",
                                                     json.dumps(body).encode(), {"Content-Type": "application/json"})
                        with urllib.request.urlopen(req, timeout=10) as response:
                            wire = response.read().decode()
                        if stream:
                            frames = [json.loads(line[6:]) for line in wire.splitlines()
                                      if line.startswith("data: ") and line != "data: [DONE]"]
                        if path == "chat/completions":
                            parts = [f["choices"][0]["delta"] for f in frames if f["choices"]] if stream else [
                                json.loads(wire)["choices"][0]["message"]]
                            reasoning, content = ("".join(p.get(key) or "" for p in parts)
                                                  for key in ("reasoning_content", "content"))
                        elif path == "messages":
                            parts = [f["delta"] for f in frames if f["type"] == "content_block_delta"] if stream else (
                                json.loads(wire)["content"])
                            reasoning, content = ("".join(p.get(key, "") for p in parts) for key in ("thinking", "text"))
                        else:
                            result = frames[-1]["response"] if stream else json.loads(wire)
                            reasoning, content = [item["content"][0]["text"] for item in result["output"]]
                        self.assertEqual((reasoning, content), ("abc", "OK"))
                        self.assertEqual(len(eng.prompts), 2)
                    finally:
                        httpd.shutdown()
                        httpd.server_close()

    def test_disabled_reasoning_over_http(self):
        for path in ("chat/completions", "messages", "responses"):
            for stream in (False, True):
                for with_tools in (False, True):
                    with self.subTest(path=path, stream=stream, tools=with_tools):
                        fn = TOOLS[0]["function"]
                        follow = call(fn["name"], {"city": "Paris"}) if with_tools else "OK" + END
                        svc, eng, tok = service(MODEL + THINK + "secret draft", follow, model_name="inkling-2-mini")
                        httpd = server.serve(svc, port=0)
                        try:
                            if path == "responses":
                                body = {"input": "Hi", "reasoning": {"effort": 0.0}}
                                tools = [{"type": "function", **fn}]
                            elif path == "messages":
                                body = {"messages": ASK, "max_tokens": 200, "thinking": {"type": "disabled"}}
                                tools = [{"name": fn["name"], "input_schema": fn["parameters"]}]
                            else:
                                body = {"messages": ASK, "reasoning_effort": "none"} if stream else {
                                    "messages": ASK, "chat_template_kwargs": {"enable_thinking": False}}
                                tools = TOOLS
                            if with_tools:
                                body["tools"] = tools
                            body.update(stream=stream, thinking_budget=5)
                            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_port}/v1/{path}",
                                                         json.dumps(body).encode(), {"Content-Type": "application/json"})
                            with urllib.request.urlopen(req, timeout=10) as response:
                                wire = response.read().decode()
                            self.assertNotIn("secret", wire)
                            reasoning_field = {"responses": '"type": "reasoning"', "messages": '"thinking_delta"',
                                               "chat/completions": '"reasoning_content"'}[path]
                            self.assertNotIn(reasoning_field, wire)
                            if stream and not with_tools and path != "responses":
                                frames = [json.loads(line[6:]) for line in wire.splitlines()
                                          if line.startswith("data: ") and line != "data: [DONE]"]
                                if path == "messages":
                                    content = [f["delta"]["text"] for f in frames if f["type"] == "content_block_delta"]
                                else:
                                    content = [f["choices"][0]["delta"].get("content", "") for f in frames if f["choices"]]
                                self.assertEqual("".join(content), "OK")
                            else:
                                self.assertIn("get_weather" if with_tools else "OK", wire)
                            self.assertEqual(tok.decode(eng.generated[0]), MODEL + THINK)
                            if with_tools:
                                self.assertIn({"responses": "function_call", "messages": "tool_use",
                                               "chat/completions": "tool_calls"}[path], wire)
                                self.assertIn("Paris", wire)
                            self.assertEqual(len(eng.prompts), 2)
                            close = END if with_tools else END + MODEL + TEXT
                            self.assertEqual(tok.decode(eng.prompts[1][len(eng.prompts[0]):]), MODEL + THINK + close)
                        finally:
                            httpd.shutdown()
                            httpd.server_close()

    def test_numeric_zero_and_disabled_tools(self):
        for effort in (0, 0.0):
            svc, eng, tok = service(MODEL + THINK + "secret", "OK" + END)
            svc.loop_guard = False
            got = answer(svc, {"messages": ASK, "tools": TOOLS, "tool_choice": "none", "reasoning_effort": effort})
            self.assertEqual((got["reasoning"], got["done"]["reasoning_tokens"], got["content"]), ("", 0, "OK"))
            self.assertTrue(tok.decode(eng.prompts[0]).endswith(MODEL + TEXT))

    def test_repeated_reasoning_can_finish_with_its_own_answer(self):
        draft = "Check hours at the end of each month. " * 3
        svc, eng, _ = service(MODEL + THINK + draft + END + MODEL + TEXT + "July." + END)
        got = answer(svc, {"messages": ASK})
        self.assertEqual((got["reasoning"], got["content"]), (draft, "July."))
        self.assertEqual(len(eng.prompts), 1)

    def test_persistent_thinking_loop_resumes_in_a_text_segment(self):
        draft = "Compare calendar invitations in both time zones. " * 5
        svc, eng, tok = service(MODEL + THINK + draft, draft, draft, "July." + END)
        got = answer(svc, {"messages": ASK})
        self.assertEqual(got["content"], "July.")
        self.assertTrue(tok.decode(eng.prompts[-1]).endswith(END + MODEL + TEXT))
        self.assertEqual(svc.loop_guard_counts["thinking_closed"], 1)

    def test_guard_retries_report_the_first_native_prefill(self):
        block = "Compare calendar invitations in both time zones. " * 5
        svc, eng, _ = service(MODEL + THINK + block, block, block, "July." + END)
        generate, reports = eng.generate, []

        def timed(ids, max_new, sampling, cancel, embeddings=None):
            generated = 0
            try:
                for token in generate(ids, max_new, sampling, cancel, embeddings):
                    generated += 1
                    yield token
            finally:
                eng.last = dict(prompt_tokens=len(ids), generated=generated, decode_ms=2 * generated,
                                reused=len(ids) - 2 if reports else 0, prompt_ms=1 if reports else 12345)
                reports.append(eng.last)

        eng.generate = timed
        got = answer(svc, {"messages": ASK})
        self.assertEqual(got["content"], "July.")
        self.assertEqual(len(reports), 4)
        timings = got["done"]["timings"]
        self.assertEqual((timings["cache_n"], timings["prompt_ms"]), (0, 12345))
        self.assertEqual(timings["predicted_ms"], sum(r["decode_ms"] for r in reports))

    def test_repeated_typed_calls_keep_all_arguments(self):
        args = {"city": "San Cristobal de La Laguna", "days": 2}
        svc, eng, _ = service(call("get_weather", args) * 3)
        got = answer(svc, {"messages": ASK, "tools": TOOLS})
        self.assertEqual(got["calls"], [("get_weather", args)] * 3)
        self.assertEqual(len(eng.prompts), 1)

    def test_loop_in_visible_answer_still_stops(self):
        block = "Check hours at the end of each month. "
        svc, _, _ = service(MODEL + TEXT + block * 6, block * 6)
        got = answer(svc, {"messages": ASK})
        self.assertEqual(got["done"]["finish"], "stop")
        self.assertEqual(svc.loop_guard_counts["stopped"], 1)
        self.assertEqual(svc.loop_guard_counts["thinking_closed"], 0)

    def test_stop(self):
        svc, _, tok = service("x")
        self.assertIn(tok.special_tokens[EOS], svc.stop_ids)
        self.assertEqual(svc.eos_ids, {tok.special_tokens[EOS]})

    def test_thinking_and_answer(self):
        svc, eng, tok = service(MODEL + THINK + "Thinking." + END + MODEL + TEXT + "Hello." + END)
        got = answer(svc, {"messages": ASK})
        self.assertEqual((got["reasoning"], got["content"], got["calls"]), ("Thinking.", "Hello.", []))
        self.assertTrue(tok.decode(eng.prompts[0]).endswith(USER + TEXT + "Weather in Paris?" + END))

    def test_effort(self):
        svc, eng, tok = service(MODEL + TEXT + "Hello." + END)
        answer(svc, {"messages": ASK, "reasoning_effort": "none"})
        answer(svc, {"messages": ASK, "reasoning_effort": "low"})
        svc.default_effort = "xhigh"
        answer(svc, {"messages": ASK})
        said = [tok.decode(p) for p in eng.prompts]
        for text, num in zip(said, ("0", "0.2", "0.99")):
            self.assertIn(EFFORT + num + END, text)

    def test_direct_answer_ends_at_its_first_segment(self):
        svc, eng, tok = service("H609" + END + MODEL + THINK + "Consider again." + END + MODEL + TEXT + "H609" + END)
        got = answer(svc, {"messages": ASK, "reasoning_effort": "none"})
        self.assertEqual((got["content"], got["reasoning"], got["done"]["finish"]), ("H609", "", "stop"))
        self.assertIn(f" stop_token={tok.special_tokens[END]}", server.StrataEngine.sampling_keys(eng.sampling[0]))

    def test_tool_call(self):
        svc, _, _ = service(MODEL + THINK + "Look it up." + END + call("get_weather", {"city": "Paris", "days": 2}))
        got = answer(svc, {"messages": ASK, "tools": TOOLS})
        self.assertEqual(got["calls"], [("get_weather", {"city": "Paris", "days": 2})])
        self.assertEqual(got["reasoning"], "Look it up.")

    def test_required_call_opens_in_its_own_form(self):
        svc, eng, tok = service('{"name":"get_weather","args":{"city":"Rome"}}' + END)
        got = answer(svc, {"messages": ASK, "tools": TOOLS,
                           "tool_choice": {"type": "function", "function": {"name": "get_weather"}}})
        want = tok.encode(MODEL + "get_weather" + CALL, parse_special=True)
        self.assertEqual(eng.prompts[0][-len(want):], want)
        self.assertIn(EFFORT + "0" + END, tok.decode(eng.prompts[0]))   # (thinking off for a required call)
        self.assertEqual(got["calls"], [("get_weather", {"city": "Rome"})])

    def test_pictures_are_refused_in_plain_words(self):
        for name in ("inkling-2-small", "inkling-2-mini"):
            with self.subTest(name=name):
                svc, eng, _ = service("x", model_name=name)
                msgs, tools, kw = openai_to_messages({"messages": [PICTURE]})
                with self.assertRaises(ValueError) as e:
                    svc.prepare(msgs, tools, kw, 100)
                self.assertIn(f"{name} on tintero reads text only", str(e.exception))
                self.assertEqual(eng.prompts, [])

    def test_logprobs_and_bias_are_refused(self):   # (an engine whose INFO names neither)
        svc, _, _ = service("x")
        with self.assertRaises(ValueError):
            svc.check_request({"messages": ASK, "logprobs": True})
        with self.assertRaises(ValueError):
            svc.check_request({"messages": ASK, "logit_bias": {"65": 5}})


class Defaults(unittest.TestCase):
    CFG = {"exe": "/opt/maya/build/tintero", "args": ["--model", "/models/Inkling-2-Small-UD-Q3_K_XL-00001-of-00005.gguf",
                                                       "--max-context", "32768"], "model_name": "inkling-2-small"}

    def test_config_effort(self):
        cfg = server.checked_config_effort(dict(self.CFG), public(), True)
        self.assertEqual(cfg["reasoning_effort"], "auto")


if __name__ == "__main__":
    unittest.main()
