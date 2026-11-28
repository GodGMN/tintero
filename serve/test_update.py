"""serve/test_update.py - About > Updates: the version compare, the release summary, which folders may update themselves,
and the update itself (a fast-forward to the release's tag, then the server's exit for maya.py) - against a throwaway
git "origin" with two releases, nothing from the network.

    python -m unittest serve.test_update -v
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import update as U  # noqa: E402

HAVE_GIT = shutil.which("git") is not None


def sh(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "init.defaultBranch=main", *args],
                   cwd=cwd, check=True, capture_output=True)


class Pieces(unittest.TestCase):
    def test_versions(self):
        self.assertEqual(U.vtuple("v1.0.18"), (1, 0, 18))
        self.assertEqual(U.vtuple("1.0.9"), (1, 0, 9))
        self.assertGreater(U.vtuple("v1.0.10"), U.vtuple("1.0.9"))     # numbers, not text
        self.assertIsNone(U.vtuple("--upload-pack=x"))                  # never handed to git
        self.assertIsNone(U.vtuple(None))

    def test_summary_is_the_first_sentence(self):
        notes = "**Maya-L is out: the closest to FP8.**\nUpdate: `git pull`\n\n## What's new\n- a"
        self.assertEqual(U.summary(notes), "Maya-L is out: the closest to FP8.")
        self.assertEqual(U.summary("## What's new\n- only a list"), "- only a list")

    def test_newer(self):
        u = U.Updater("1.0.17")
        u.latest = {"version": "1.0.18", "tag": "v1.0.18"}
        self.assertTrue(u.newer())
        u.latest = {"version": "1.0.17", "tag": "v1.0.17"}
        self.assertFalse(u.newer())

    def test_github_unreachable_is_said_and_retried_later(self):
        u = U.Updater("1.0.17")
        with mock.patch.object(U.urllib.request, "urlopen", side_effect=OSError("offline")):
            info = u.check()
        self.assertIn("GitHub could not be reached", info["error"])
        self.assertFalse(info["newer"])
        with mock.patch.object(U.urllib.request, "urlopen") as again:
            u.check()                                   # within the retry pause: not asked again
            again.assert_not_called()

    def test_check_off(self):
        with mock.patch.dict(os.environ, {"MAYA_UPDATE_CHECK": "0"}), \
                mock.patch.object(U.urllib.request, "urlopen") as net:
            self.assertFalse(U.Updater("1.0.17").check(force=True)["enabled"])
            net.assert_not_called()


REPO_ROOT = Path(__file__).resolve().parents[1]


def fake_release(tag):
    """What urlopen returns for the releases/latest request: a context manager whose read() is the API's JSON."""
    body = json.dumps({"tag_name": tag, "html_url": f"https://github.com/GodGMN/tintero/releases/tag/{tag}",
                       "body": "**Notes.**", "published_at": "2026-11-27T04:26:45Z"}).encode()
    resp = mock.MagicMock()
    resp.__enter__.return_value.read.return_value = body
    return resp


class ReleaseIdentity(unittest.TestCase):
    """An installed distribution (a Docker image or a release tarball) has no .git: its identity is this repository's
    VERSION file and the Tintero releases, never another project's."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.pkg = self.tmp / "opt" / "tintero"           # the layout of docker/Dockerfile.*: VERSION and serve/ side by side
        (self.pkg / "serve").mkdir(parents=True)
        shutil.copy(REPO_ROOT / "VERSION", self.pkg / "VERSION")
        shutil.copy(REPO_ROOT / "serve" / "update.py", self.pkg / "serve" / "update.py")
        env = {k: v for k, v in os.environ.items() if k not in U.UPDATE_CHECK_ENV}
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def version(self, root=None):
        return (Path(root or self.pkg) / "VERSION").read_text(encoding="utf-8").strip()

    def check(self, tag, root=None, current=None):
        u = U.Updater(current or self.version(root), root=root or self.pkg)
        with mock.patch.object(U.urllib.request, "urlopen", return_value=fake_release(tag)) as net:
            info = u.check(force=True)
        return info, net.call_args.args[0]

    def test_version_file_is_the_latest_released_version(self):
        version = self.version(REPO_ROOT)
        self.assertRegex(version, r"^\d+\.\d+\.\d+$", "VERSION must be MAJOR.MINOR.PATCH")
        self.assertIsNotNone(U.vtuple(version))
        headings = re.findall(r"^## v(\d+\.\d+\.\d+) - ", (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), re.M)
        self.assertTrue(headings, "CHANGELOG.md lists no release")
        self.assertEqual(version, headings[0], "VERSION is the newest release in CHANGELOG.md (Unreleased is not one)")
        self.assertFalse(any(U.vtuple(h) > U.vtuple(version) for h in headings))

    def test_the_fallback_repository_is_tintero(self):
        self.assertEqual(U.REPO, "GodGMN/tintero")

    def test_packaged_install_asks_tintero_releases(self):
        self.assertFalse((self.pkg / ".git").exists())
        info, req = self.check("v" + self.version())
        self.assertEqual(req.full_url, "https://api.github.com/repos/GodGMN/tintero/releases/latest")
        self.assertEqual(req.get_header("User-agent"), f"tintero/{self.version()}")
        self.assertEqual(info["latest"]["repo"], "GodGMN/tintero")
        self.assertIsNone(info["error"])

    def test_packaged_install_at_the_latest_release_is_up_to_date(self):
        info, _ = self.check("v" + self.version())
        self.assertFalse(info["newer"])
        self.assertEqual(info["current"], self.version())

    def test_packaged_install_sees_a_newer_release_but_cannot_update_itself(self):
        major, minor, patch = U.vtuple(self.version())
        with mock.patch.dict(os.environ, {U.SUPERVISED: "1"}):          # even when started by maya.py
            info, _ = self.check(f"v{major}.{minor}.{patch + 1}")
        self.assertTrue(info["newer"])
        self.assertFalse(info["can_update"])
        self.assertIn("not a git checkout", info["blocker"])
        self.assertIn("download", info["by_hand"])

    def test_the_release_check_can_be_turned_off_under_either_name(self):
        for key in ("TINTERO_UPDATE_CHECK", "MAYA_UPDATE_CHECK"):
            with self.subTest(key=key), mock.patch.dict(os.environ, {key: "0"}), \
                    mock.patch.object(U.urllib.request, "urlopen") as net:
                self.assertFalse(U.Updater(self.version(), root=self.pkg).check(force=True)["enabled"])
                net.assert_not_called()

    def test_tintero_name_wins_over_the_maya_name(self):
        with mock.patch.dict(os.environ, {"TINTERO_UPDATE_CHECK": "1", "MAYA_UPDATE_CHECK": "0"}):
            self.assertTrue(U.Updater("0.1.0").enabled())
        with mock.patch.dict(os.environ, {"TINTERO_UPDATE_CHECK": "0", "MAYA_UPDATE_CHECK": "1"}):
            self.assertFalse(U.Updater("0.1.0").enabled())

    @unittest.skipUnless(HAVE_GIT, "git is not installed")
    def test_a_checkout_keeps_asking_its_own_origin(self):
        for url, repo in (("git@github.com:SomeFork/tintero.git", "SomeFork/tintero"),
                          ("https://github.com/GodGMN/tintero", "GodGMN/tintero"),
                          ("https://example.org/mirror/tintero.git", U.REPO)):          # not GitHub: the fallback
            with self.subTest(origin=url):
                co = self.tmp / "co" / hex(abs(hash(url)))
                co.mkdir(parents=True, exist_ok=True)
                sh(co, "init")
                sh(co, "remote", "add", "origin", url)
                (co / "VERSION").write_text(self.version() + "\n")
                info, req = self.check("v" + self.version(), root=co)
                self.assertEqual(req.full_url, f"https://api.github.com/repos/{repo}/releases/latest")
                self.assertEqual(info["latest"]["repo"], repo)

    def test_a_tag_that_is_not_a_version_is_refused(self):
        info, _ = self.check("nightly")
        self.assertIn("could not be reached", info["error"])
        self.assertFalse(info["newer"])


@unittest.skipUnless(HAVE_GIT, "git is not installed")
class Updating(unittest.TestCase):
    """An origin with v1.0.0 and v1.0.1, a clone at v1.0.0, and the server's side of the update."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        src = self.tmp / "src"
        src.mkdir()
        sh(src, "init")
        (src / "VERSION").write_text("1.0.0\n")
        sh(src, "add", "VERSION")
        sh(src, "commit", "-m", "1.0.0")
        sh(src, "tag", "-a", "v1.0.0", "-m", "v1.0.0")
        sh(self.tmp, "clone", "-q", str(src), "clone")
        (src / "VERSION").write_text("1.0.1\n")
        sh(src, "commit", "-am", "1.0.1")
        sh(src, "tag", "-a", "v1.0.1", "-m", "v1.0.1")
        self.clone = self.tmp / "clone"
        self.u = U.Updater("1.0.0", root=self.clone)
        self.u.latest = {"version": "1.0.1", "tag": "v1.0.1", "url": None, "summary": ""}
        self.env = mock.patch.dict(os.environ, {U.SUPERVISED: "1"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def svc(self):
        s = mock.Mock()
        s.fifo = threading.Lock()
        return s

    def run_update(self):
        svc = self.svc()
        with mock.patch.object(U.os, "_exit") as ex:
            self.u.start(svc)
            self.u.thread.join(60)
        return svc, ex

    def test_update_moves_to_the_tag_and_ends_for_maya_py(self):
        info = self.u.info()
        self.assertTrue(info["can_update"], info["blocker"])
        svc, ex = self.run_update()
        self.assertEqual((self.clone / "VERSION").read_text().strip(), "1.0.1")
        svc.close_for_restart.assert_called_once()
        ex.assert_called_once_with(U.UPDATE_EXIT)

    def test_local_changes_block_it(self):
        (self.clone / "VERSION").write_text("mine\n")
        info = self.u.info()
        self.assertFalse(info["can_update"])
        self.assertIn("changed by hand", info["blocker"])
        with self.assertRaises(ValueError):
            self.u.start(self.svc())

    def test_not_started_by_maya_sh(self):
        del os.environ[U.SUPERVISED]
        self.assertIn("setup.sh", self.u.info()["blocker"])

    def test_not_a_checkout(self):
        shutil.rmtree(self.clone / ".git", onerror=lambda f, p, e: (os.chmod(p, 0o700), f(p)))
        self.assertIn("not a git checkout", self.u.info()["blocker"])
        self.assertIn("download", self.u.info()["by_hand"])

    def test_a_diverged_folder_fails_and_keeps_serving(self):
        sh(self.clone, "commit", "--allow-empty", "-m", "a commit of my own")
        svc, ex = self.run_update()
        ex.assert_not_called()
        svc.close_for_restart.assert_not_called()
        self.assertEqual(self.u.state["state"], "failed")
        self.assertIn("git pull", self.u.state["error"])
        self.assertEqual((self.clone / "VERSION").read_text().strip(), "1.0.0")


if __name__ == "__main__":
    unittest.main()
