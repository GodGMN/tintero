"""Responses tool contracts over both model tool syntaxes and the real HTTP/service pipeline."""
import json
import unittest

from serve.test_responses import CALL, TOOLS, ROOT, ResponseCase
from serve.frontend import ChatTemplate, OutputParser

QWEN = '<tool_call><function=lookup><parameter=key>alpha</parameter></function></tool_call>'
DSML = ('<\uff5cDSML\uff5c calls>\n<\uff5cDSML\uff5c invoke name="lookup">\n'
        '<\uff5cDSML\uff5c parameter name="key" string="true">alpha</\uff5cDSML\uff5c parameter>\n'
        '</\uff5cDSML\uff5c invoke>\n</\uff5cDSML\uff5c calls>')


class ToolContracts(ResponseCase):
    def test_single_tool_stops_at_a_complete_call_and_keeps_real_usage(self):
        for script in (CALL, QWEN):
            with self.subTest(script=script):
                text = script + script.replace("alpha", "beta") + "unwanted tail"
                self.script(text)
                events = self.stream(tools=TOOLS, parallel_tool_calls=False)
                out = events[-1]["response"]
                self.assertEqual(out["status"], "completed")
                self.assertFalse(out["parallel_tool_calls"])
                self.assertEqual(len(out["output"]), 1)
                self.assertEqual(json.loads(out["output"][0]["arguments"]), {"key": "alpha"})
                self.assertEqual(out["usage"]["output_tokens"], len(self.tok.encode(script, parse_special=True)))
                self.assertEqual(self.svc.history[-1]["finish"], "stop")
                self.assertFalse(self.engine.cancel.is_set())
                self.assertTrue(self.engine.closed)
                self.assertEqual(self.svc.active, 0)
                self.script("Next request works")
                self.assertEqual(self.post()[0], 200)

    def test_truncated_calls_never_publish_completed_arguments(self):
        for text in (CALL, QWEN):
            # Cut inside the string, its closing tag, the outer closing tag, or the function name.
            cuts = [text.index("alpha") + 2, text.index("alpha") + 5, len(text) - 2, 13]
            for cut in cuts:
                with self.subTest(text=text, cut=cut):
                    self.script(text)
                    events = self.stream(tools=TOOLS, max_output_tokens=cut)
                    self.assertEqual(events[-1]["type"], "response.incomplete")
                    self.assertNotIn("response.function_call_arguments.done", [e["type"] for e in events])
                    calls = [item for item in events[-1]["response"]["output"] if item["type"] == "function_call"]
                    self.assertTrue(all(item["status"] == "incomplete" for item in calls))

    def test_early_end_of_turn_inside_a_call_fails(self):
        for text in (CALL, QWEN):
            self.script(text[:text.index("alpha") + 2])
            events = self.stream(tools=TOOLS)
            self.assertEqual(events[-1]["type"], "response.failed")
            self.assertNotIn("response.function_call_arguments.done", [e["type"] for e in events])

    def test_explicit_outer_close_does_not_fix_missing_parameter_close(self):
        for text in (CALL.replace("</arg_value>", ""), QWEN.replace("</parameter>", "")):
            self.script(text)
            events = self.stream(tools=TOOLS)
            self.assertEqual(events[-1]["type"], "response.failed")
            self.assertNotIn("response.function_call_arguments.done", [e["type"] for e in events])

    def test_trailing_garbage_inside_a_call_is_not_a_complete_call(self):
        for text in (CALL, QWEN):
            self.script(text.replace("</tool_call>", "unexpected</tool_call>"))
            events = self.stream(tools=TOOLS)
            self.assertEqual(events[-1]["type"], "response.failed")
            self.assertNotIn("response.function_call_arguments.done", [e["type"] for e in events])

    def test_no_executable_call_is_published_after_a_malformed_call(self):
        self.script(CALL.replace("</arg_value>", "") + CALL)
        events = self.stream(tools=TOOLS)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertNotIn("response.function_call_arguments.done", [e["type"] for e in events])

    def test_non_finite_json_arguments_are_rejected(self):
        tools = [{**TOOLS[0], "parameters": {"type": "object", "properties": {"key": {"type": "number"}}}}]
        self.script(CALL.replace("alpha", "NaN"))
        events = self.stream(tools=tools)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertNotIn("response.function_call_arguments.done", [e["type"] for e in events])

    def test_single_tool_parser_stops_when_multiple_calls_arrive_in_one_token(self):
        for text in (CALL, QWEN):
            parser = OutputParser(thinking=False, tools=[TOOLS[0]], stream_tools=True,
                                  strict_tools=True, single_tool=True)
            events = parser.feed(text + text)
            self.assertEqual(len([e for e in events if e.kind == "tool_call"]), 1)
            self.assertEqual(parser.feed(text) + parser.finish(), [])

    def test_tool_selection_violations_fail_before_an_executable_item(self):
        cases = [(CALL, {"tools": TOOLS, "tool_choice": "none"}),
                 (CALL.replace("lookup", "undeclared"), {"tools": TOOLS})]
        for script, fields in cases:
            self.script(script)
            events = self.stream(**fields)
            self.assertEqual(events[-1]["type"], "response.failed")
            self.assertNotIn("response.function_call_arguments.done", [e["type"] for e in events])
            self.assertNotIn("response.output_item.added", [e["type"] for e in events])

    def test_required_without_a_call_fails_but_budget_exhaustion_is_incomplete(self):
        # Forced tool prefill is tested by existing tests. A generated malformed/incomplete
        # continuation must not fulfill that requirement merely because its name was prefixed.
        self.script("not a valid function body")
        events = self.stream(tools=TOOLS, tool_choice="required")
        self.assertEqual(events[-1]["type"], "response.failed")
        self.script("a" * 100)
        events = self.stream(tools=TOOLS, tool_choice="required", max_output_tokens=3)
        self.assertEqual(events[-1]["type"], "response.incomplete")

    def test_the_rejection_names_the_include(self):
        status, out = self.post(include=["file_search_call.results"])
        self.assertEqual(status, 400, out)
        self.assertIn("file_search_call.results", json.dumps(out))

    def test_codex_built_in_tools_are_left_out(self):
        # Codex CLI 0.160.0 sends web_search next to its functions: Maya cannot run it, the model never sees it
        self.script(CALL)
        status, out = self.post(tools=[*TOOLS, {"type": "web_search", "external_web_access": False},
                                       {"type": "local_shell"}], reasoning={"effort": "none", "summary": "auto"})
        self.assertEqual(status, 200, out)
        self.assertEqual(out["output"][0]["name"], "lookup")
        self.assertEqual([t["type"] for t in out["tools"]], ["function", "web_search", "local_shell"])
        prompt = self.tok.decode(self.engine.last_prompt)
        self.assertNotIn("web_search", prompt)
        self.assertNotIn("local_shell", prompt)

    def test_codex_namespace_tools_round_trip(self):
        # Codex CLI groups its sub-agent tools in a namespace; the model sees "ns::name", the client gets
        # name + namespace back and sends the call that way in the next turn's history
        space = {"type": "namespace", "name": "agents", "description": "Sub-agents.",
                 "tools": [{**TOOLS[0], "name": "wait"}, {"type": "web_search"}]}
        for script in (CALL, QWEN):
            with self.subTest(script=script):
                call_text = script.replace("lookup", "agents::wait")
                self.script([call_text, "waited"])
                events = self.stream(tools=[*TOOLS, space])
                out = events[-1]["response"]
                self.assertEqual(out["status"], "completed", out)
                self.assertEqual(out["tools"][-1], {**space, "tools": [space["tools"][0]]})
                call = out["output"][0]
                self.assertEqual((call["name"], call["namespace"]), ("wait", "agents"))
                added = next(e for e in events if e["type"] == "response.output_item.added")
                self.assertEqual((added["item"]["name"], added["item"]["namespace"]), ("wait", "agents"))
                prompt = self.tok.decode(self.engine.last_prompt)
                self.assertIn("agents::wait", prompt)
                self.assertIn("Sub-agents.", prompt)
                history = [{"role": "user", "content": "wait"}, *out["output"],
                           {"type": "function_call_output", "call_id": call["call_id"], "output": "none left"}]
                status, second = self.post(input=history, tools=[*TOOLS, space])
                self.assertEqual(status, 200, second)
                self.assertEqual(second["output"][0]["content"][0]["text"], "waited")
                prompt = self.tok.decode(self.engine.last_prompt)
                self.assertIn("agents::wait", prompt.split("Sub-agents.")[-1])   # the replayed call, not the tool list
                self.assertIn("none left", prompt)
        self.script('<parameter=key>alpha</parameter>\n</function>\n</tool_call>')   # the forced opening is the prompt's
        status, out = self.post(tools=[*TOOLS, space],
                                tool_choice={"type": "function", "name": "wait", "namespace": "agents"})
        self.assertEqual((status, out["output"][0]["namespace"]), (200, "agents"), out)
        self.assertIn("<function=agents::wait>", self.tok.decode(self.engine.last_prompt))
        # a call to the namespace's function under its bare name is not a declared function
        self.script(CALL.replace("lookup", "wait"))
        self.assertEqual(self.stream(tools=[space])[-1]["type"], "response.failed")

    def test_codex_encrypted_reasoning_round_trip(self):
        # include reasoning.encrypted_content with store=false: the reasoning comes back as an opaque blob too;
        # the next turn may send the text and the blob (Codex does), or the blob alone, or another server's blob
        self.script(["check the sum</think>\n\n42", "ok", "ok", "ok"])
        events = self.stream(reasoning={"effort": "high", "summary": "auto"}, include=["reasoning.encrypted_content"],
                             store=False)
        out = events[-1]["response"]
        reasoning = out["output"][0]
        self.assertEqual(reasoning["type"], "reasoning")
        self.assertTrue(reasoning["encrypted_content"].startswith("maya-r1:"))
        done = [e["item"] for e in events if e["type"] == "response.output_item.done"][0]
        self.assertEqual(done["encrypted_content"], reasoning["encrypted_content"])
        self.assertEqual(out["reasoning"]["summary"], "auto")
        blob_only = {"type": "reasoning", "summary": [], "encrypted_content": reasoning["encrypted_content"]}
        foreign = {"type": "reasoning", "summary": [], "encrypted_content": "gAAAAB-not-ours"}
        for item, seen in ((reasoning, True), (blob_only, True), (foreign, False)):
            with self.subTest(item=item):
                history = [{"role": "user", "content": "What is the sum?"}, item, *out["output"][1:],
                           {"role": "user", "content": "Check it"}]
                status, second = self.post(input=history, include=["reasoning.encrypted_content"],
                                           reasoning={"effort": "high"})
                self.assertEqual(status, 200, second)
                prompt = self.tok.decode(self.engine.last_prompt)
                self.assertIn("42", prompt)
                self.assertEqual("check the sum" in prompt, seen)
        # without include, no blob
        self.script("check the sum</think>\n\n42")
        out = self.stream(reasoning={"effort": "high"})[-1]["response"]
        self.assertNotIn("encrypted_content", out["output"][0])

    def test_codex_hints_and_reasoning_disabled_projection(self):
        status, out = self.post(prompt_cache_key="test-session", client_metadata={"session_id": "fixture"},
                                include=["reasoning.encrypted_content"])
        self.assertEqual(status, 200, out)
        self.assertEqual(out["prompt_cache_key"], "test-session")
        self.assertNotIn("client_metadata", out)
        for fields in ({"prompt_cache_key": {}}, {"prompt_cache_key": "a" * 257},
                       {"client_metadata": []}, {"client_metadata": {"session": 1}}):
            self.assertEqual(self.post(**fields)[0], 400)


if __name__ == "__main__":
    unittest.main()
