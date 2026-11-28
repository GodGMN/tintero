"""Process-level tests for server shutdown, parent-pipe handling and the native launcher."""
from __future__ import annotations

import io
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import server as SV  # noqa: E402

try:
    import psutil
except ImportError:                                     # (the CI's and setup's environments have it)
    psutil = None
try:
    import regex  # noqa: F401  (the GLM tokenizer's, which --engine maya loads: else the mock engine, no process)
    FAKE_ENGINE = True
except ImportError:
    FAKE_ENGINE = False
WIN = os.name == "nt"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def fake_config(d: Path) -> Path:
    """A config whose engine is serve/fake_multi_engine.py (one conversation) and a tokenizer of single bytes."""
    tok = d / "tokenizer"
    tok.mkdir()
    bs = [*range(ord("!"), ord("~") + 1), *range(ord("\xa1"), ord("\xac") + 1), *range(ord("\xae"), ord("\xff") + 1)]
    cs = bs[:] + [256 + n for n in range(256 - len(bs))]
    bs += [b for b in range(256) if b not in bs]
    vocab = {chr(c): b for b, c in zip(bs, cs)}         # (byte-level BPE's 256 byte tokens, as GPT-2 maps them)
    vocab.update({"<|im_end|>": 256, "</think>": 257})
    (tok / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (tok / "merges.txt").write_text("a b", encoding="utf-8")
    engine = ROOT / "serve" / "fake_multi_engine.py"
    if WIN:                                             # (Windows starts no .py by itself: a .bat around it)
        exe = d / "fake_engine.bat"
        exe.write_text(f'@"{sys.executable}" "{engine}" %*\r\n', encoding="utf-8")
    else:
        exe = engine
    cfg = d / "maya-fake.json"
    cfg.write_text(json.dumps({"exe": str(exe), "args": ["1"], "tokenizer": str(tok), "model_name": "fake",
                               "log": str(d / "engine.log")}), encoding="utf-8")
    return cfg


def server_cmd(cfg: Path, port: int) -> list:
    engine = ["--engine", "maya", "--config", str(cfg)] if FAKE_ENGINE else ["--engine", "mock"]
    return [sys.executable, str(ROOT / "serve" / "server.py"), *engine, "--port", str(port)]


def hup_default() -> None:
    """(a child's preexec_fn) SIGHUP at its default, as in a terminal: under nohup (a release run's `setsid nohup`)
    the tests inherit it ignored, and the server and maya.py keep an ignored SIGHUP ignored (by design) - the SIGHUP
    tests then waited 60 s for an end that never came."""
    signal.signal(signal.SIGHUP, signal.SIG_DFL)


class Proc:
    """A process with its output read line by line (until() waits for a line)."""

    def __init__(self, cmd, **kw):
        if not WIN:
            kw.setdefault("preexec_fn", hup_default)
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  encoding="utf-8", errors="replace", bufsize=1, **kw)
        self.lines: queue.Queue = queue.Queue()
        self.seen: list = []
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.p.stdout:
            self.lines.put(line.rstrip())
        self.lines.put(None)

    def until(self, text: str, timeout: float = 60) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                line = self.lines.get(timeout=max(0.05, end - time.monotonic()))
            except queue.Empty:
                break
            if line is None:
                return False
            self.seen.append(line)
            if text in line:
                return True
        return False

    def close(self):
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait(timeout=30)
        for f in (self.p.stdin, self.p.stdout):
            if f is not None:
                try:
                    f.close()
                except OSError:
                    pass

    def rest(self, timeout: float = 5) -> str:
        """The output that is left once the process ended."""
        self.until("\0never", timeout)
        return "\n".join(self.seen)


def gone(pids, timeout: float = 30) -> list:
    """The processes of `pids` still running after `timeout` seconds (zombies count as ended)."""
    def alive(pid):
        try:
            return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False
    end = time.monotonic() + timeout
    left = list(pids)
    while left and time.monotonic() < end:
        left = [p for p in left if alive(p)]
        if left:
            time.sleep(0.2)
    return left


def kill_all(pids) -> None:
    for q in pids:
        try:
            psutil.Process(q).kill()
        except psutil.Error:
            pass


def descendants(pid: int) -> list:
    try:
        return [c.pid for c in psutil.Process(pid).children(recursive=True)]
    except psutil.Error:
        return []


class WatchParent(unittest.TestCase):
    """The server's watch of its parent (--parent-pipe), in-process."""

    def test_eof_on_the_pipe_stops(self):
        got = []
        SV.watch_parent(got.append, stdin=io.BytesIO(b"x" * 10000), getppid=lambda: 1234)
        end = time.monotonic() + 5
        while not got and time.monotonic() < end:
            time.sleep(0.01)
        self.assertEqual(got[:1], ["the program that started it ended"])

    def test_a_new_parent_stops(self):
        got, ppid = [], [1234]
        quiet = SimpleNamespace(read=lambda n: time.sleep(3600))   # (a pipe that stays open: the EOF watch waits)
        SV.watch_parent(got.append, stdin=quiet, getppid=lambda: ppid[0], poll=0.01)
        time.sleep(0.1)
        self.assertEqual(got, [])
        ppid[0] = 1                                     # (the parent ended: init took the server)
        end = time.monotonic() + 5
        while not got and time.monotonic() < end:
            time.sleep(0.01)
        self.assertEqual(got, ["its parent process ended"])

    def test_stop_once(self):
        stop = threading.Event()
        SV.request_stop(stop, "a")
        SV.request_stop(stop, "b")
        self.assertTrue(stop.is_set())

    def test_the_main_wait_never_holds_the_events_lock(self):
        """main()'s wait polls the flag: Event.wait held the Event's lock while a signal handler (same thread) called
        stop.set() on it - a SIGTERM / Ctrl+C at that moment hung the server for good."""
        class NoWait(threading.Event):
            def wait(self, timeout=None):
                raise AssertionError("main()'s stop wait must not call Event.wait")
        stop = NoWait()
        threading.Timer(0.05, lambda: SV.request_stop(stop, "test")).start()
        t0 = time.monotonic()
        SV.wait_for_stop(stop, step=0.01)
        self.assertTrue(stop.is_set())
        self.assertLess(time.monotonic() - t0, 2.0)

    @unittest.skipIf(WIN, "POSIX signals")
    def test_a_signal_handler_stops_the_main_wait(self):
        """The real handler path: SIGALRM's handler calls request_stop on the main thread while it waits."""
        stop = threading.Event()
        old = signal.signal(signal.SIGALRM, lambda s, f: SV.request_stop(stop, "SIGALRM"))
        self.addCleanup(signal.signal, signal.SIGALRM, old)
        signal.setitimer(signal.ITIMER_REAL, 0.05)
        self.addCleanup(signal.setitimer, signal.ITIMER_REAL, 0)
        t0 = time.monotonic()
        SV.wait_for_stop(stop, step=0.01)
        self.assertTrue(stop.is_set())
        self.assertLess(time.monotonic() - t0, 2.0)


@unittest.skipIf(psutil is None, "psutil is not installed")
class ServerStops(unittest.TestCase):
    """The real server with the fake engine: each way of stopping it ends it and its engine, after QUIT."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = fake_config(Path(self.tmp.name))

    def start(self, extra=(), **kw) -> Proc:
        p = self.serve(server_cmd(self.cfg, free_port()) + list(extra), **kw)
        self.assertTrue(self.engine or not FAKE_ENGINE, "no engine process")
        return p

    def serve(self, cmd, **kw) -> Proc:
        """The server, started and ready; self.engine: its engine's processes."""
        p = Proc(cmd, **kw)
        # (after p.close, before the folder goes: a server the test leaves running is killed there, and its engine -
        # no QUIT then - ends only on its input's EOF a moment later; on Windows it still held engine.log)
        self.engine = []
        self.addCleanup(lambda: (kill_all(self.engine), gone(self.engine, 15)))
        self.addCleanup(p.close)
        self.assertTrue(p.until("ready:"), "the server did not start:\n" + "\n".join(p.seen))
        self.engine = descendants(p.p.pid)
        return p

    def ended(self, p: Proc, why: str):
        self.assertEqual(p.p.wait(timeout=60), 0)
        out = p.rest()
        self.assertIn(f"[maya] stopping ({why}): the engine ends", out)
        self.assertEqual(gone(self.engine), [], "the engine outlived the server")

    @unittest.skipUnless(FAKE_ENGINE, "the tokenizer's regex module")
    def test_old_engine_name(self):
        """--engine strata (the old name, in older run scripts) starts the configured engine, not the mock one."""
        cmd = server_cmd(self.cfg, free_port())
        cmd[cmd.index("maya")] = "strata"
        self.serve(cmd, stdin=subprocess.DEVNULL)
        self.assertTrue(self.engine, "no engine process: --engine strata ran the mock engine")

    def test_parent_pipe_closed(self):
        p = self.start(["--parent-pipe"], stdin=subprocess.PIPE)
        p.p.stdin.close()
        self.ended(p, "the program that started it ended")

    def test_stdin_closed_without_parent_pipe_changes_nothing(self):
        p = self.start(stdin=subprocess.PIPE)
        p.p.stdin.close()
        time.sleep(2)
        self.assertIsNone(p.p.poll())                   # (llama-swap, systemd, a run script: as before)

    @unittest.skipIf(WIN, "POSIX signals")
    def test_sigterm(self):
        p = self.start(stdin=subprocess.DEVNULL)
        p.p.send_signal(signal.SIGTERM)
        self.ended(p, "SIGTERM")

    @unittest.skipIf(WIN, "POSIX signals")
    def test_sighup(self):
        p = self.start(stdin=subprocess.DEVNULL)
        p.p.send_signal(signal.SIGHUP)
        self.ended(p, "SIGHUP")

    @unittest.skipIf(WIN, "POSIX signals")
    def test_sighup_under_nohup_is_ignored(self):
        """A server started under nohup (SIGHUP ignored) keeps running on SIGHUP; SIGTERM still stops it."""
        p = self.start(stdin=subprocess.DEVNULL, preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN))
        p.p.send_signal(signal.SIGHUP)
        time.sleep(2)
        self.assertIsNone(p.p.poll())
        p.p.send_signal(signal.SIGTERM)
        self.ended(p, "SIGTERM")


@unittest.skipUnless(FAKE_ENGINE and psutil, "the fake engine needs regex and psutil")
class NativeLauncher(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="tintero launch ")
        self.addCleanup(temp.cleanup)
        self.cfg = fake_config(Path(temp.name))
        cfg = json.loads(self.cfg.read_text(encoding="utf-8"))
        cfg["port"] = self.port = free_port()
        self.cfg.write_text(json.dumps(cfg), encoding="utf-8")
        cmd = [sys.executable, str(ROOT / "maya.py"), "--config", str(self.cfg)]
        self.launcher = Proc(cmd, stdin=subprocess.DEVNULL)
        self.addCleanup(self.launcher.close)
        self.assertTrue(self.launcher.until("ready:"), "the launcher did not start:\n" + "\n".join(self.launcher.seen))
        self.tree = descendants(self.launcher.p.pid)
        self.addCleanup(lambda: kill_all(self.tree))
        self.assertTrue(self.tree, "the launcher did not start an engine")

    def test_serves_the_configured_model_on_the_configured_port(self):
        from urllib.request import urlopen

        with urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=10) as reply:
            model = json.load(reply)["data"][0]
        self.assertEqual(model["id"], "fake")
        self.assertEqual(model["meta"]["n_ctx"], 4096)

    @unittest.skipIf(WIN, "POSIX signals")
    def test_sigterm_stops_the_engine(self):
        self.launcher.p.send_signal(signal.SIGTERM)
        self.assertEqual(self.launcher.p.wait(timeout=15), 0)
        self.assertEqual(gone(self.tree), [])

    def test_terminating_the_launcher_closes_the_engine_pipe(self):
        self.launcher.p.kill()
        self.launcher.p.wait(timeout=15)
        self.assertEqual(gone(self.tree), [])


if __name__ == "__main__":
    unittest.main()
