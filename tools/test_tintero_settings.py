"""Exercise setting aliases through the server, CMake and the gfx1100 launcher."""
import contextlib
import http.client
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def setting_env(settings):
    inherited = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("TINTERO_", "MAYA_", "STRATA_", "IK2_"))
    }
    return {**inherited, **settings}


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(port, path, *, key=None, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        conn.request("GET" if body is None else "POST", path, body, headers)
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


class ConfigSettings(unittest.TestCase):
    def test_round_trip_prefers_the_public_name_and_preserves_other_keys(self):
        from tools import maya_names

        cfg = {"env": {
            "TINTERO_SERVER_LOG": "", "MAYA_SERVER_LOG": "old.log", "STRATA_SERVER_LOG": "older.log",
            "IK2_KV": "fp8", "MAYA_DEBUG": "1", "TINTERO_DEBUG": "custom",
        }}
        maya_names.env_in(cfg)
        self.assertEqual(cfg["env"], {
            "MAYA_SERVER_LOG": "", "IK2_KV": "fp8", "MAYA_DEBUG": "1", "TINTERO_DEBUG": "custom",
        })
        self.assertEqual(maya_names.env_out(cfg)["env"], {
            "TINTERO_SERVER_LOG": "", "IK2_KV": "fp8", "MAYA_DEBUG": "1", "TINTERO_DEBUG": "custom",
        })


class ServerSettings(unittest.TestCase):
    @contextlib.contextmanager
    def server(self, env, config=None, args=()):
        with tempfile.TemporaryDirectory() as tmp:
            port = free_port()
            cmd = [
                sys.executable, "-m", "serve.server", "--engine", "mock",
                "--port", str(port), "--tokenizer", str(Path(tmp) / "no-tokenizer"),
            ]
            if config is not None:
                path = Path(tmp) / "server.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                cmd += ["--config", str(path)]
            log = Path(tmp) / "server.log"
            with log.open("w", encoding="utf-8") as stream, subprocess.Popen(
                [*cmd, *args], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                env=setting_env(env)
            ) as process:
                try:
                    deadline = time.monotonic() + 10
                    while process.poll() is None and time.monotonic() < deadline:
                        try:
                            request(port, "/v1/models")
                            break
                        except ConnectionRefusedError:
                            time.sleep(0.05)
                    else:
                        self.fail(log.read_text(encoding="utf-8"))
                    yield port
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()

    def test_api_key(self):
        cases = [
            ("default", {"TINTERO_MOCK_TOK_S": "private-name"}, None, (), None, None),
            ("legacy", {"MAYA_API_KEY": "old"}, None, (), "old", None),
            ("public", {"TINTERO_API_KEY": "new"}, None, (), "new", None),
            ("conflict", {"TINTERO_API_KEY": "new", "MAYA_API_KEY": "old"}, None, (), "new", "old"),
            ("empty", {"TINTERO_API_KEY": "", "MAYA_API_KEY": "old"}, None, (), None, None),
            ("top-level fallback", {"TINTERO_API_KEY": "", "MAYA_API_KEY": "old"},
             {"api_key": "file"}, (), "file", "old"),
            ("CLI", {"TINTERO_API_KEY": "new"}, None, ("--api-key", "flag"), "flag", "new"),
        ]
        for label, env, config, args, accepted, rejected in cases:
            with self.subTest(case=label), self.server(env, config, args) as port:
                self.assertEqual(request(port, "/v1/models")[0], 401 if accepted else 200)
                if accepted:
                    self.assertEqual(request(port, "/v1/models", key=accepted)[0], 200)
                if rejected:
                    self.assertEqual(request(port, "/v1/models", key=rejected)[0], 401)

    def test_request_body_limit(self):
        cases = [
            ("default", {}, {}, 400),
            ("legacy process key", {"MAYA_MAX_BODY_MIB": "1"}, {}, 413),
            ("public process key", {"TINTERO_MAX_BODY_MIB": "1"}, {}, 413),
            ("empty public key", {"TINTERO_MAX_BODY_MIB": "", "MAYA_MAX_BODY_MIB": "1"}, {}, 400),
            ("public JSON key wins", {}, {"TINTERO_MAX_BODY_MIB": "1", "MAYA_MAX_BODY_MIB": "2"}, 413),
            ("JSON overrides process", {"TINTERO_MAX_BODY_MIB": "2"}, {"MAYA_MAX_BODY_MIB": "1"}, 413),
        ]
        body = b" " * (1 << 20) + b"{"
        for label, env, config_env, expected in cases:
            with self.subTest(case=label), self.server(env, {"env": config_env}) as port:
                self.assertEqual(request(port, "/v1/chat/completions", body=body)[0], expected)

    def test_update_check_with_conflicting_json_keys(self):
        env = {"HTTPS_PROXY": "http://127.0.0.1:1", "https_proxy": "http://127.0.0.1:1", "NO_PROXY": "", "no_proxy": ""}
        config = {"env": {"TINTERO_UPDATE_CHECK": "0", "MAYA_UPDATE_CHECK": "1"}}
        with self.server(env, config) as port:
            status, result = request(port, "/api/update")
            self.assertEqual(status, 200)
            self.assertIs(result["enabled"], False)
            self.assertIsNone(result["checked"])
            self.assertIsNone(result["error"])

    def test_responses_limit_from_json_reaches_service_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "server.json"
            config.write_text(json.dumps({"env": {
                "TINTERO_GLM_RESPONSES_JOB_WORKERS": "33",
                "MAYA_GLM_RESPONSES_JOB_WORKERS": "2",
            }}), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "serve.server", "--config", str(config), "--port", "0"],
                cwd=ROOT, env=setting_env({}), capture_output=True, text=True, timeout=5,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Responses supports at most 32 background workers and 4096 job records", result.stderr)


@unittest.skipUnless(shutil.which("cmake"), "CMake is unavailable")
class BuildSettings(unittest.TestCase):
    def test_backend_selection(self):
        cases = [
            ("default", [], 0),
            ("legacy", ["-DMAYA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=ON"], 1),
            ("public HIP", ["-DTINTERO_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=ON"], 1),
            ("public CUDA", ["-DTINTERO_ENABLE_CUDA=ON", "-DSTRATA_ENABLE_HIP=ON"], 1),
            ("conflict", ["-DTINTERO_ENABLE_HIP=OFF", "-DMAYA_ENABLE_HIP=ON"], 0),
            ("empty", ["-DTINTERO_ENABLE_HIP=", "-DMAYA_ENABLE_HIP=ON"], 0),
        ]
        for label, args, expected in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmp:
                result = subprocess.run(
                    ["cmake", "-S", str(ROOT), "-B", tmp, *args],
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                if expected:
                    self.assertIn("Select one GPU backend per build directory", result.stderr)


@unittest.skipUnless(os.name == "posix", "The gfx1100 launcher uses POSIX shell")
class PackageSettings(unittest.TestCase):
    def test_preset_selection(self):
        cases = [
            ("default", {}, "ik2-mini-gfx1100.json"),
            ("legacy", {"MAYA_IK2_PRESET": "old.json"}, "old.json"),
            ("public", {"TINTERO_IK2_PRESET": "new.json"}, "new.json"),
            ("conflict", {"TINTERO_IK2_PRESET": "new.json", "MAYA_IK2_PRESET": "old.json"}, "new.json"),
            ("empty", {"TINTERO_IK2_PRESET": "", "MAYA_IK2_PRESET": "old.json"}, "ik2-mini-gfx1100.json"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            python = Path(tmp) / "python3"
            python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n', encoding="utf-8")
            python.chmod(0o755)
            for label, env, expected in cases:
                with self.subTest(case=label):
                    env = setting_env({"PATH": tmp + os.pathsep + os.environ.get("PATH", ""), **env})
                    result = subprocess.run(
                        ["sh", str(ROOT / "docker/start-ik2-gfx1100.sh"), "--port", "19095"],
                        cwd=tmp, env=env, capture_output=True, text=True, timeout=5, check=True,
                    )
                    self.assertEqual(result.stdout.splitlines(), [
                        "serve/server.py", "--engine", "maya", "--config", expected, "--port", "19095",
                    ])


if __name__ == "__main__":
    unittest.main()
