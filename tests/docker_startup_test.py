"""GPU-free acceptance tests for the Docker entrypoint and health probe."""
import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from gguf import GGUFWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from strata_tokenizer import bytes_to_unicode  # noqa: E402

ENTRYPOINT = ROOT / "docker/entrypoint.sh"
if not ENTRYPOINT.exists():  # Release bundles put the entrypoint at their root.
    ENTRYPOINT = ROOT / "entrypoint.sh"

ENGINE = """#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

Path("engine-pid").write_text(str(os.getpid()))
while not Path("engine-ready").exists():
    if Path("engine-fail").exists():
        sys.exit(7)
    time.sleep(0.01)
print("READY 256", flush=True)
for line in sys.stdin:
    if line.strip() == "QUIT":
        break
"""


class DockerStartupTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for directory in ("docker", "serve"):
            (self.root / directory).symlink_to(ROOT / directory, target_is_directory=True)
        for directory in ("models", "tmp"):
            (self.root / directory).mkdir()
        self.model = self.root / "models/fixture.gguf"
        writer = GGUFWriter(self.model, "inkling2")
        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("inkling2")
        tokens = [*bytes_to_unicode().values(), "ab"]
        writer.add_token_list(tokens)
        writer.add_token_merges(["a b"])
        writer.add_token_types([1] * len(tokens))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()

        engine = self.root / "fixture-engine"
        engine.write_text(ENGINE, encoding="utf-8")
        engine.chmod(0o755)
        self.original = self.root / "preset.json"
        self.original.write_text(json.dumps({
            "exe": str(engine),
            "args": ["--model", "models/unused.gguf"],
            "env": {"IK2_KV": "fp8"},
            "tokenizer": "ik2/tokenizer",
            "model_management": True,
            "model_name": "inkling-2-mini",
            "api_key": "fixture-key",
        }), encoding="utf-8")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.env = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
                    "TMPDIR": str(self.root / "tmp"), "TINTERO_CONFIG": str(self.original),
                    "TINTERO_MODEL": "models/fixture.gguf", "IK2_KV": "fp4-global", "MAYA_API_KEY": "",
                    "TINTERO_HOST": "127.0.0.1", "TINTERO_PORT": str(self.port),
                    "TINTERO_TOKENIZER_CACHE": str(self.root / "cache")}
        self.gate = self.root / "engine-ready"
        self.pid_file = self.root / "engine-pid"

    def prepare(self):
        return subprocess.run(
            [sys.executable, ROOT / "docker/prepare-config.py", self.original],
            cwd=self.root, env=self.env, check=True, capture_output=True, text=True, timeout=20,
        )

    def start(self, *args):
        output = (self.root / "startup.log").open("w+", encoding="utf-8")
        self.addCleanup(output.close)
        process = subprocess.Popen([ENTRYPOINT, *args], cwd=self.root, env=self.env,
                                   stdout=output, stderr=output, start_new_session=True)
        self.addCleanup(self.stop, process)
        return process

    def stop(self, process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)

    def eventually(self, condition):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(0.025)
        logfile = self.root / "startup.log"
        self.fail("Timed out waiting for fixture state\n" + (logfile.read_text() if logfile.exists() else ""))

    def request(self, path, post=False):
        headers = {"Authorization": "Bearer fixture-key"} if post else {}
        req = Request(f"http://127.0.0.1:{self.port}{path}", data=b"{}" if post else None, headers=headers)
        with urlopen(req, timeout=15) as response:
            return json.load(response)

    def probe(self):
        return subprocess.run([sys.executable, ROOT / "docker/healthcheck.py"], cwd=self.root,
                              env=self.env, check=False, timeout=5).returncode

    def test_first_start_exports_and_restart_reuses_it(self):
        original = self.original.read_bytes()
        self.gate.touch()
        process = self.start()
        self.eventually(lambda: self.probe() == 0)
        generated, = self.root.glob("cache/*/tokenizer")
        vocab = json.loads((generated / "vocab.json").read_text())
        self.assertEqual(vocab["ab"], 256)
        self.assertEqual(len(vocab), 257)
        template = generated / "chat_template.jinja"
        self.assertEqual(template.read_bytes(), (ROOT / "tools/ik2/chat_template.jinja").read_bytes())
        self.assertEqual(self.original.read_bytes(), original)
        mtime = template.stat().st_mtime_ns
        self.stop(process)
        self.start()
        self.eventually(lambda: self.probe() == 0)
        self.assertIn("using cached tokenizer", (self.root / "startup.log").read_text())
        self.assertEqual(template.stat().st_mtime_ns, mtime)

    def test_bootstraps_the_preset_given_on_the_command_line(self):
        fallback = self.root / "fallback.json"
        fallback.write_text("{}", encoding="utf-8")
        self.env["TINTERO_CONFIG"] = str(fallback)
        self.gate.touch()
        self.start("--config", str(self.original))
        self.eventually(lambda: self.probe() == 0)
        self.assertEqual(len(list(self.root.glob("cache/*/tokenizer"))), 1)

    def test_recovers_partial_cache_and_keeps_user_files(self):
        first = self.prepare()
        effective = json.loads(Path(first.stdout.strip()).read_text())
        self.assertEqual(effective["args"][-1], "models/fixture.gguf")
        self.assertEqual(effective["env"]["IK2_KV"], "fp4-global")
        generated = Path(effective["tokenizer"])
        user = self.root / "ik2/tokenizer"
        user.mkdir(parents=True)
        vocab = user / "vocab.json"
        vocab.write_text('{"personal":0}', encoding="utf-8")
        staged = generated.parent.with_name(f".{generated.parent.name}.tmp")
        staged.mkdir()
        (staged / "abandoned").touch()
        (generated / "merges.txt").unlink()

        recovered = self.prepare()
        self.assertIn("exporting tokenizer", recovered.stderr)
        self.assertEqual(Path(json.loads(Path(recovered.stdout.strip()).read_text())["tokenizer"]), generated)
        self.assertEqual((generated / "merges.txt").read_text(), "a b")
        self.assertFalse(staged.exists())
        self.assertEqual(vocab.read_text(), '{"personal":0}')

        for name in ("vocab.json", "merges.txt", "tokenizer.json", "token_type.json"):
            (user / name).write_bytes((generated / name).read_bytes())
        custom = "{{ messages[0].content }}\n"
        (user / "chat_template.jinja").write_text(custom, encoding="utf-8")
        restored = self.prepare()
        self.assertIn("using configured tokenizer", restored.stderr)
        self.assertEqual(json.loads(Path(restored.stdout.strip()).read_text())["tokenizer"], "ik2/tokenizer")
        self.assertEqual((user / "chat_template.jinja").read_text(), custom)

    def test_parallel_starts_publish_one_complete_cache(self):
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            runs = list(pool.map(lambda _: self.prepare(), range(2)))
        self.assertEqual(sum("exporting tokenizer" in run.stderr for run in runs), 1)
        self.assertEqual(sum("using cached tokenizer" in run.stderr for run in runs), 1)
        generated, = self.root.glob("cache/*/tokenizer")
        self.assertEqual((generated / "merges.txt").read_text(), "a b")

    def test_readiness_through_load_unload_and_engine_failure(self):
        self.start()
        self.eventually(self.pid_file.exists)
        self.assertEqual(self.probe(), 1)
        self.gate.touch()
        self.eventually(lambda: self.probe() == 0)
        self.assertTrue(self.request("/health")["ready"])

        self.assertEqual(self.request("/models/unload", post=True)["status"]["value"], "unloaded")
        self.assertTrue(self.request("/health")["unloaded"])
        self.assertEqual(self.probe(), 1)
        self.assertEqual(self.request("/models/load", post=True)["status"]["value"], "loaded")
        self.assertEqual(self.probe(), 0)

        os.kill(int(self.pid_file.read_text()), signal.SIGKILL)
        self.eventually(lambda: self.probe() == 1)
        self.assertFalse(self.request("/health")["ready"])
        self.assertFalse(self.request("/health")["unloaded"])
        self.pid_file.unlink()
        self.gate.unlink()
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            reloading = pool.submit(self.request, "/models/load", post=True)
            try:
                self.eventually(self.pid_file.exists)
                self.assertFalse(self.request("/health")["ready"])
                self.assertEqual(self.probe(), 1)
            finally:
                (self.root / "engine-fail").touch()
            with self.assertRaises(HTTPError) as failure:
                reloading.result(timeout=15)
        self.assertEqual(failure.exception.code, 503)
        self.assertEqual(self.probe(), 1)

    def test_failure_before_ready_exits_nonzero(self):
        (self.root / "engine-fail").touch()
        process = self.start()
        self.assertNotEqual(process.wait(timeout=15), 0)
        self.assertIn("engine exited before it was ready", (self.root / "startup.log").read_text())
        self.assertEqual(self.probe(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
