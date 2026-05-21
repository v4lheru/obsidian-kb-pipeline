"""
src/diff_report.py -- Unified-diff renderer for `--dry-run --dry-run-mode=diff`.

Builds a unified diff between a synthesized WikiPage and the on-disk file it
would overwrite. New pages diff against `/dev/null` so all content shows as
added lines. Unchanged output reports `no changes`.

Uses stdlib `difflib.unified_diff` only -- zero new dependencies. Diffs are
truncated at 200 lines per page with a footer so a huge page rewrite doesn't
swamp stdout.

Used by `pipeline._emit_diff_report` in diff-mode dry-runs.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from .synthesizer import WikiPage
from .wiki_writer import _render_page

_DIFF_LINE_CAP = 200
_DIFF_CONTEXT = 3


def build_diff(page: WikiPage, vault_root: Path) -> str:
    """Build a unified-diff string for `page` against its on-disk counterpart.

    Returns '' when the rendered page is byte-identical to the file on disk.
    For new pages (no file exists), the diff shows the whole new content as
    additions against /dev/null.

    The diff is truncated at 200 body lines; if the cap is hit, a footer
    `... (diff truncated, N lines omitted)` is appended.
    """
    rendered = _render_page(page)
    target_path = vault_root / page.path
    existing: str
    is_new_file = not target_path.exists()
    if is_new_file:
        existing = ""
    else:
        try:
            existing = target_path.read_text(encoding="utf-8")
        except OSError:
            existing = ""

    if existing == rendered:
        return ""

    from_file = "/dev/null" if is_new_file else str(target_path)
    to_file = str(target_path)

    diff_iter = difflib.unified_diff(
        existing.splitlines(keepends=True),
        rendered.splitlines(keepends=True),
        fromfile=from_file,
        tofile=to_file,
        n=_DIFF_CONTEXT,
    )

    lines = list(diff_iter)
    if not lines:
        return ""

    if len(lines) > _DIFF_LINE_CAP:
        omitted = len(lines) - _DIFF_LINE_CAP
        truncated = lines[:_DIFF_LINE_CAP]
        truncated.append(f"... (diff truncated, {omitted} lines omitted)\n")
        return "".join(truncated)

    return "".join(lines)


def print_diff_report(pages: list[WikiPage], vault_root: Path) -> None:
    """Print a unified diff per page to stdout, then a summary footer.

    Per-page output:
      - non-empty diff: a `=== <relative-path> ===` header then the diff body.
      - empty diff: a single `<relative-path>: no changes` line.

    Summary footer counts pages changed, added lines, removed lines (excluding
    diff hunk headers / file headers).
    """
    changed = 0
    added_lines = 0
    removed_lines = 0

    print(f"\n{'=' * 60}")
    print(f"DRY RUN (diff mode) -- {len(pages)} pages synthesized")
    print(f"{'=' * 60}")

    for page in pages:
        diff_text = build_diff(page, vault_root)
        if not diff_text:
            print(f"{page.path}: no changes")
            continue

        changed += 1
        print(f"\n=== {page.path} ===")
        print(diff_text, end="" if diff_text.endswith("\n") else "\n")

        for line in diff_text.splitlines():
            if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
                continue
            if line.startswith("+"):
                added_lines += 1
            elif line.startswith("-"):
                removed_lines += 1

    print(f"\n{'=' * 60}")
    print(
        f"Summary: {changed} of {len(pages)} pages would change "
        f"(+{added_lines} -{removed_lines} lines). No writes performed."
    )
    print(f"{'=' * 60}\n")
