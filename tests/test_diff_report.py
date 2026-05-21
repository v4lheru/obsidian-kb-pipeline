"""
tests/test_diff_report.py -- Diff-mode unified-diff renderer unit tests.

Covers empty diff (no changes), new-page diff (/dev/null header), unchanged
page (returns ''), and 200-line truncation.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src.diff_report import build_diff, print_diff_report
from src.synthesizer import WikiPage
from src.wiki_writer import _render_page


def _page(path: str, body: str = "# Test\n\nA line.\n", is_new: bool = True) -> WikiPage:
    return WikiPage(
        path=path,
        title="Test",
        frontmatter={"type": "resource", "tags": ["wiki"], "status": "draft",
                     "created": "2026-04-06", "modified": "2026-04-06", "area": "work"},
        body=body,
        wikilinks=[],
        extraction_ids=[],
        word_count=len(body.split()),
        is_new=is_new,
    )


class TestBuildDiff(unittest.TestCase):
    def test_unchanged_page_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            page = _page("Patterns/x.md")
            rendered = _render_page(page)
            target = root / "Patterns" / "x.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rendered, encoding="utf-8")
            self.assertEqual(build_diff(page, root), "")

    def test_new_page_diff_has_dev_null_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            page = _page("Patterns/new.md")
            diff = build_diff(page, root)
            self.assertIn("/dev/null", diff)
            self.assertIn("+# Test", diff)

    def test_changed_page_shows_both_sides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            page = _page("Patterns/x.md", body="# Test\n\nNew line.\n")
            target = root / "Patterns" / "x.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                _render_page(_page("Patterns/x.md", body="# Test\n\nOld line.\n")),
                encoding="utf-8",
            )
            diff = build_diff(page, root)
            self.assertIn("-Old line.", diff)
            self.assertIn("+New line.", diff)

    def test_huge_diff_truncated_at_200_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            huge_body = "# Test\n\n" + "\n".join(f"line {i}" for i in range(500)) + "\n"
            page = _page("Patterns/huge.md", body=huge_body)
            diff = build_diff(page, root)
            lines = diff.splitlines()
            self.assertLessEqual(len(lines), 202)  # 200 cap + truncation footer + slack
            self.assertTrue(any("diff truncated" in ln for ln in lines))


class TestPrintDiffReport(unittest.TestCase):
    def test_summary_line_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pages = [_page("a.md"), _page("b.md")]
            buf = io.StringIO()
            with redirect_stdout(buf):
                print_diff_report(pages, root)
            output = buf.getvalue()
            self.assertIn("DRY RUN (diff mode)", output)
            self.assertIn("Summary:", output)
            self.assertIn("No writes performed", output)


if __name__ == "__main__":
    unittest.main()
