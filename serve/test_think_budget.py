"""Live and persisted thinking budget settings."""
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "serve"), str(ROOT / "tools"), str(ROOT)]
import server  # noqa: E402

CONFIG = {"exe": "build/tintero", "args": ["--model", "model.gguf"]}


class Field(unittest.TestCase):
    def test_validation(self):
        for ok in (0, 256, 8192, 262144, 8192.0):
            self.assertEqual(self.svc(None).set_think_budget(ok)["budget"], int(ok))
        for bad in (1, 255, 262145, -5, 1.5, "8192", True, None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.svc(None).set_think_budget(bad)

    def svc(self, cfg_path, end=5):
        s = object.__new__(server.Service)
        s._config_lock = threading.Lock()
        s.think_budget, s.think_end_id, s.config_path = 8192, end, cfg_path
        return s

    def test_apply_live_and_save(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "maya-x.json"
            p.write_text(json.dumps(CONFIG))
            s = self.svc(str(p))
            self.assertEqual(s.think_budget_info(), {"budget": 8192, "supported": True, "saved": True})
            self.assertEqual(s.set_think_budget(2048)["budget"], 2048)
            self.assertEqual(s.think_budget, 2048)                               # live: read per request
            cfg = json.loads(p.read_text())
            self.assertEqual(cfg["thinking_budget"], 2048)
            self.assertEqual(cfg["args"], CONFIG["args"])
            with self.assertRaises(ValueError):
                s.set_think_budget(100)
            self.assertEqual(s.think_budget, 2048)                               # a bad value changes nothing

    def test_unsupported_model(self):
        s = self.svc(None, end=None)
        self.assertFalse(s.think_budget_info()["supported"])
        with self.assertRaises(ValueError):
            s.set_think_budget(1024)


if __name__ == "__main__":
    unittest.main()
