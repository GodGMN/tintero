"""serve/test_pdf.py - the chat page's PDF attachments (serve/pdf.py, #837): a hand-written PDF's text layer comes
back as text, a scanned one comes back as PNG pages, and a machine without poppler is told the command to install
it.  The extraction tests run against the real pdftotext and pdftoppm and skip when they are not installed; the
route tests only need the mock engine (no GPU, no pack).

    python -m unittest serve.test_pdf -v
"""
from __future__ import annotations

import base64
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import pdf as pdfmod    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POPPLER = shutil.which("pdftotext") and shutil.which("pdftoppm")


def _pdf(streams: list[bytes]) -> bytes:
    """A minimal PDF, one page per content stream (an empty stream: a page with nothing readable on it, which is
    what a scanned one looks like to pdftotext).  Written by hand so the test knows exactly which text layer
    poppler should find; the xref is computed as the objects go down."""
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(len(streams)))
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [{kids}] /Count {len(streams)} >>".encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for i, s in enumerate(streams):
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >>"
                    f" /Contents {5 + 2 * i} 0 R >>".encode())
        objs.append(b"<< /Length " + str(len(s)).encode() + b" >>\nstream\n" + s + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").encode()
    return bytes(out)


def _lines(texts: list[str]) -> bytes:
    """A page's content stream: each text on its own Helvetica line."""
    return b"\n".join(f"BT /F1 12 Tf 72 {740 - 16 * k} Td ({t}) Tj ET".encode() for k, t in enumerate(texts))


WORDS = [f"The quick brown fox jumps over the lazy dog, line {k}, and the pale moon sets behind the hill."
         for k in range(6)]            # well past the 200-character floor, so this one is "a PDF with text"
LIMIT = 512 * 1024                     # the cap the chat page's health advertises at a small context


@unittest.skipUnless(POPPLER, "poppler is not installed")
class Extract(unittest.TestCase):
    def test_text_layer_becomes_text(self):
        r = pdfmod.extract(_pdf([_lines(WORDS)]), want_images=True, text_limit=LIMIT)
        self.assertEqual(r["kind"], "text")
        self.assertIn("The quick brown fox jumps over the lazy dog, line 0", r["text"])
        self.assertIn("line 5", r["text"])
        self.assertFalse(r["truncated"])
        self.assertNotIn("note", r)

    def test_long_text_cut_at_the_pages_limit(self):
        r = pdfmod.extract(_pdf([_lines(WORDS)]), want_images=True, text_limit=300)
        self.assertEqual(r["kind"], "text")                    # (the floor is what a page is worth, not 300)
        self.assertEqual(len(r["text"]), 300)
        self.assertTrue(r["truncated"])
        self.assertIn("cut at 300", r["note"])

    def test_scanned_pdf_becomes_pages(self):
        r = pdfmod.extract(_pdf([b""]), want_images=True, text_limit=LIMIT)
        self.assertEqual(r["kind"], "images")
        self.assertEqual(r["pages"], 1)
        self.assertTrue(r["images"][0]["data"].startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(r["images"][0]["data"].split(",", 1)[1])[:8], b"\x89PNG\r\n\x1a\n")

    def test_scanned_pages_capped_at_six(self):
        r = pdfmod.extract(_pdf([b""] * 8), want_images=True, text_limit=LIMIT)
        self.assertEqual(r["kind"], "images")
        self.assertEqual(len(r["images"]), 6)
        self.assertEqual(r["pages"], 6)
        self.assertIn("first 6 of 8", r["note"])               # (when pdfinfo is there to know the 8)

    def test_scanned_pdf_without_images_says_so(self):
        r = pdfmod.extract(_pdf([b""]), want_images=False, text_limit=LIMIT)
        self.assertIn("no text layer", r["error"]["message"])

    def test_not_a_pdf(self):
        r = pdfmod.extract(b"hello, not a PDF at all", want_images=True, text_limit=LIMIT)
        self.assertIn("not a PDF", r["error"]["message"])


@unittest.skipUnless(POPPLER, "poppler is not installed")
class RealPdfExtras(unittest.TestCase):
    def test_page_count_comes_from_pdfinfo(self):
        """(the count only decorates the toast, so its absence is never an error - but with poppler installed,
        pdfinfo is one of its packages and the text result carries the count)"""
        if shutil.which("pdfinfo") is None:
            self.skipTest("pdfinfo is not installed")
        r = pdfmod.extract(_pdf([_lines(WORDS), b"", b""]), want_images=True, text_limit=LIMIT)
        self.assertEqual((r["kind"], r["pages"]), ("text", 3))


class MissingPoppler(unittest.TestCase):
    """The machine without poppler - a PATH where no tool answers, however the rest of the test happens to look
    (most Windows and macOS PCs): the pure-Python reader (serve/pdf_text.py) reads the text layer, and only a PDF
    nothing came out of names poppler, apt and pacman both."""

    def test_text_layer_read_without_poppler(self):
        with mock.patch.object(shutil, "which", return_value=None):
            r = pdfmod.extract(_pdf([_lines(WORDS), b""]), want_images=True, text_limit=LIMIT)
        self.assertEqual((r["kind"], r["pages"]), ("text", 2))
        self.assertIn("The quick brown fox jumps over the lazy dog, line 0", r["text"])
        self.assertIn("line 5", r["text"])

    def test_install_command_named_when_nothing_came_out(self):
        with mock.patch.object(shutil, "which", return_value=None):
            r = pdfmod.extract(_pdf([b""]), want_images=True, text_limit=LIMIT)
        m = r["error"]["message"]
        self.assertIn("sudo apt install poppler-utils", m)
        self.assertIn("sudo pacman -S poppler", m)

    def test_no_pictures_wanted_says_so(self):
        with mock.patch.object(shutil, "which", return_value=None):
            r = pdfmod.extract(_pdf([b""]), want_images=False, text_limit=LIMIT)
        self.assertIn("no text layer", r["error"]["message"])


if __name__ == "__main__":
    unittest.main()
