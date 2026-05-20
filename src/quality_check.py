"""
src/quality_check.py -- Wiki page quality scanner + auto-fixer.

Scans all wiki pages in the Obsidian Coding-Notes vault and reports:
  - Pages over a word threshold (default 5000, flag for splitting)
  - Pages under a word threshold (default 100, flag as stubs)
  - Orphaned pages (no inbound [[wikilinks]] from any other page)
  - Broken [[wikilinks]] (link targets that don't exist as files)
  - Pages missing required frontmatter fields (type, tags, status)

Also exports `quality_fix()` for safe auto-repair of orphans, broken
wikilinks, and missing frontmatter. Oversized pages are FLAGGED ONLY;
auto-splitting curated content without semantic understanding produces
incoherent topical fragments.

Used by pipeline.py via the `quality-check` CLI subcommand and as a
post-write step in run_memory_pipeline / run_session_pipeline.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

# Matches [[link]] and [[link|display text]]
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:[|#][^\]]*?)?\]\]")

# Required frontmatter fields
_REQUIRED_FRONTMATTER = {"type", "tags", "status"}


@dataclass
class QualityReport:
    """Results of a quality scan across the wiki vault."""

    oversized: list[tuple[str, int]] = field(default_factory=list)
    stubs: list[tuple[str, int]] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    broken_links: list[tuple[str, str]] = field(default_factory=list)
    missing_frontmatter: list[tuple[str, set[str]]] = field(default_factory=list)
    total_pages: int = 0
    max_words: int = 5000
    min_words: int = 100


def run_quality_check(
    vault_root: Path,
    max_words: int = 5000,
    min_words: int = 100,
) -> QualityReport:
    """Scan all markdown files in vault_root and produce a quality report.

    Args:
        vault_root: Path to the Coding-Notes directory.
        max_words: Word count above which a page is flagged as oversized.
        min_words: Word count below which a page is flagged as a stub.

    Returns:
        QualityReport with all findings.
    """
    report = QualityReport(max_words=max_words, min_words=min_words)

    md_files = sorted(vault_root.rglob("*.md"))
    report.total_pages = len(md_files)

    # Build a set of valid page stems (filenames without .md)
    valid_stems: set[str] = set()
    for path in md_files:
        valid_stems.add(path.stem)

    # Track inbound links per page stem
    inbound_links: dict[str, int] = {stem: 0 for stem in valid_stems}

    # First pass: scan all files
    for path in md_files:
        rel_path = path.relative_to(vault_root)
        content = path.read_text(encoding="utf-8", errors="replace")

        # Word count (exclude frontmatter)
        body = _strip_frontmatter(content)
        word_count = len(body.split())

        if word_count > max_words:
            report.oversized.append((str(rel_path), word_count))
        elif word_count < min_words:
            report.stubs.append((str(rel_path), word_count))

        # Frontmatter check
        missing = _check_frontmatter(content)
        if missing:
            report.missing_frontmatter.append((str(rel_path), missing))

        # Extract wikilinks
        links = _WIKILINK_RE.findall(content)
        for link_target in links:
            target_stem = link_target.strip()
            if target_stem in inbound_links:
                inbound_links[target_stem] += 1
            elif target_stem not in valid_stems:
                report.broken_links.append((str(rel_path), target_stem))

    # Orphan detection: pages with zero inbound links
    # Exclude the MOC itself — it's the entry point, not expected to have inbound links
    moc_stem = "coding-knowledge-map"
    for stem, count in inbound_links.items():
        if count == 0 and stem != moc_stem:
            report.orphans.append(stem)

    report.orphans.sort()
    report.oversized.sort(key=lambda x: -x[1])
    report.broken_links.sort()
    report.stubs.sort(key=lambda x: x[1])

    return report


def print_report(report: QualityReport) -> None:
    """Print a formatted quality report to stdout."""
    print(f"\n{'=' * 60}")
    print(f"WIKI QUALITY CHECK — {report.total_pages} pages scanned")
    print(f"{'=' * 60}")

    # Oversized pages
    if report.oversized:
        print(f"\n  OVERSIZED PAGES ({len(report.oversized)} over {report.max_words} words):")
        for path, wc in report.oversized:
            print(f"    {path} — {wc:,} words")
    else:
        print("\n  OVERSIZED PAGES: None")

    # Stub pages
    if report.stubs:
        print(f"\n  STUB PAGES ({len(report.stubs)} under {report.min_words} words):")
        for path, wc in report.stubs:
            print(f"    {path} — {wc} words")
    else:
        print("\n  STUB PAGES: None")

    # Orphaned pages
    if report.orphans:
        print(f"\n  ORPHANED PAGES ({len(report.orphans)} with no inbound links):")
        for stem in report.orphans:
            print(f"    {stem}")
    else:
        print("\n  ORPHANED PAGES: None")

    # Broken links
    if report.broken_links:
        print(f"\n  BROKEN WIKILINKS ({len(report.broken_links)} broken):")
        for source, target in report.broken_links:
            print(f"    {source} → [[{target}]]")
    else:
        print("\n  BROKEN WIKILINKS: None")

    # Missing frontmatter
    if report.missing_frontmatter:
        print(f"\n  MISSING FRONTMATTER ({len(report.missing_frontmatter)} pages):")
        for path, fields in report.missing_frontmatter:
            print(f"    {path} — missing: {', '.join(sorted(fields))}")
    else:
        print("\n  MISSING FRONTMATTER: None")

    # Summary
    total_issues = (
        len(report.oversized)
        + len(report.stubs)
        + len(report.orphans)
        + len(report.broken_links)
        + len(report.missing_frontmatter)
    )
    print(f"\n{'=' * 60}")
    if total_issues == 0:
        print("  All clear — no quality issues found.")
    else:
        print(f"  {total_issues} total issues found.")
    print(f"{'=' * 60}\n")


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter delimited by --- from content."""
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            return content[end + 3:]
    return content


def _check_frontmatter(content: str) -> set[str]:
    """Check if content has the required frontmatter fields.

    Returns set of missing field names, or empty set if all present.
    """
    if not content.startswith("---"):
        return _REQUIRED_FRONTMATTER.copy()

    end = content.find("---", 3)
    if end == -1:
        return _REQUIRED_FRONTMATTER.copy()

    fm_text = content[3:end]
    present: set[str] = set()
    for line in fm_text.split("\n"):
        stripped = line.strip()
        if ":" in stripped:
            key = stripped.split(":", 1)[0].strip()
            if key in _REQUIRED_FRONTMATTER:
                present.add(key)

    return _REQUIRED_FRONTMATTER - present


# ---------------------------------------------------------------------------
# Auto-fix
# ---------------------------------------------------------------------------

@dataclass
class QualityFixReport:
    """Counts of fixes applied during a quality_fix() run."""

    moc_entries_added: list[str] = field(default_factory=list)
    broken_links_removed: list[tuple[str, str]] = field(default_factory=list)
    frontmatter_repaired: list[str] = field(default_factory=list)
    oversized_flagged: list[tuple[str, int]] = field(default_factory=list)


def quality_fix(
    vault_root: Path,
    max_words: int = 5000,
    min_words: int = 100,
) -> QualityFixReport:
    """Auto-repair safe quality issues in the vault.

    Behavior:
      - Orphans → add an entry under the appropriate MOC section.
      - Broken wikilinks → remove the link, leaving its display text in place
        (or the bare target if there was no display text).
      - Missing frontmatter → prepend a stub frontmatter generated from the
        filename and best-effort content inspection.
      - Oversized pages → FLAG ONLY (return in the report). Auto-splitting
        curated handwritten content without semantic understanding produces
        incoherent fragments; flagging defers to human judgment.

    Returns a QualityFixReport summarizing what was repaired.
    """
    fix_report = QualityFixReport()

    # Re-scan the vault to know the current state of things.
    scan = run_quality_check(vault_root, max_words=max_words, min_words=min_words)

    # 1. Repair missing frontmatter (do this BEFORE orphan detection so newly
    # repaired pages can match valid stems if any were misclassified).
    for rel_path, missing_fields in scan.missing_frontmatter:
        full_path = vault_root / rel_path
        if _repair_frontmatter(full_path, missing_fields):
            fix_report.frontmatter_repaired.append(rel_path)

    # 2. Remove broken wikilinks
    broken_by_source: dict[str, list[str]] = {}
    for source, target in scan.broken_links:
        broken_by_source.setdefault(source, []).append(target)
    for source_path, targets in broken_by_source.items():
        full_path = vault_root / source_path
        for target in targets:
            if _remove_broken_link(full_path, target):
                fix_report.broken_links_removed.append((source_path, target))

    # 3. Add orphans to MOC
    moc_path = vault_root / "coding-knowledge-map.md"
    if moc_path.exists() and scan.orphans:
        added = _add_orphans_to_moc(moc_path, vault_root, scan.orphans)
        fix_report.moc_entries_added.extend(added)

    # 4. Flag oversized (no auto-split — curated content must be split by hand)
    fix_report.oversized_flagged = list(scan.oversized)
    if fix_report.oversized_flagged:
        for path, wc in fix_report.oversized_flagged:
            logger.warning(
                "Oversized page (manual split needed): %s — %d words", path, wc,
            )

    return fix_report


def print_fix_report(report: QualityFixReport) -> None:
    """Print a formatted quality_fix() report to stdout."""
    print(f"\n{'=' * 60}")
    print("WIKI QUALITY AUTO-FIX")
    print(f"{'=' * 60}")

    if report.frontmatter_repaired:
        print(f"\n  FRONTMATTER REPAIRED ({len(report.frontmatter_repaired)}):")
        for path in report.frontmatter_repaired:
            print(f"    {path}")
    else:
        print("\n  FRONTMATTER REPAIRED: None")

    if report.broken_links_removed:
        print(f"\n  BROKEN LINKS REMOVED ({len(report.broken_links_removed)}):")
        for source, target in report.broken_links_removed:
            print(f"    {source} → [[{target}]]")
    else:
        print("\n  BROKEN LINKS REMOVED: None")

    if report.moc_entries_added:
        print(f"\n  ORPHANS ADDED TO MOC ({len(report.moc_entries_added)}):")
        for entry in report.moc_entries_added:
            print(f"    {entry}")
    else:
        print("\n  ORPHANS ADDED TO MOC: None")

    if report.oversized_flagged:
        print(f"\n  OVERSIZED FLAGGED ({len(report.oversized_flagged)}, manual split):")
        for path, wc in report.oversized_flagged:
            print(f"    {path} — {wc:,} words")
    else:
        print("\n  OVERSIZED FLAGGED: None")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Auto-fix helpers
# ---------------------------------------------------------------------------

# Map subfolder to MOC section header (mirrors wiki_writer._SUBFOLDER_TO_SECTION)
_SUBFOLDER_TO_MOC_SECTION: dict[str, str] = {
    "n8n": "## n8n Workflow Automation",
    "APIs": "## API Integration References",
    "Architecture": "## Architecture Decisions",
    "Patterns": "## Cross-Cutting Patterns",
    "Gotchas": "## Gotchas & Debugging",
    "Workflows": "## Workflows & Processes",
    "Comparisons": "## Technology Comparisons",
}


def _repair_frontmatter(path: Path, missing_fields: set[str]) -> bool:
    """Prepend or extend frontmatter so required fields exist.

    Conservative: if the page already starts with `---`, only adds the
    missing keys before the closing `---`. Otherwise prepends a fresh
    frontmatter block. Returns True if the file was modified.
    """
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        logger.warning("Could not read %s for frontmatter repair", path)
        return False

    today = date.today().isoformat()
    stem = path.stem

    # Build the fields we need to write (only the ones that are missing).
    defaults = {
        "type": "resource",
        "tags": f"\n  - wiki\n  - {stem.replace('_', '-')}",
        "status": "draft",
    }
    new_lines = [f"{k}: {defaults[k]}" for k in sorted(missing_fields)
                 if k in defaults]

    if not new_lines:
        return False

    if content.startswith("---"):
        end = content.find("---", 3)
        if end == -1:
            # Malformed frontmatter — prepend a fresh block instead.
            new_fm = _build_stub_frontmatter(stem, today)
            new_content = new_fm + content
        else:
            # Insert missing keys right before the closing ---.
            insertion = "\n".join(new_lines) + "\n"
            new_content = content[:end] + insertion + content[end:]
    else:
        new_fm = _build_stub_frontmatter(stem, today)
        new_content = new_fm + content

    _atomic_write(path, new_content)
    logger.info("Repaired frontmatter on %s (added: %s)",
                path.name, ", ".join(sorted(missing_fields)))
    return True


def _build_stub_frontmatter(stem: str, today: str) -> str:
    """Generate a minimal valid frontmatter block."""
    tag_slug = stem.replace("_", "-")
    return (
        "---\n"
        "type: resource\n"
        "tags:\n"
        "  - wiki\n"
        f"  - {tag_slug}\n"
        "status: draft\n"
        f"created: {today}\n"
        f"modified: {today}\n"
        "area: work\n"
        "wiki-source: auto-repair\n"
        "---\n\n"
    )


def _remove_broken_link(path: Path, broken_target: str) -> bool:
    """Strip [[broken_target]] from a page, preserving display text if any.

    [[broken|display]] → display
    [[broken]] → broken (the user can re-link or remove the bare text)

    Returns True if the file was modified.
    """
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False

    # Match the exact broken target inside [[...]] with optional |display
    # Escaping the target since it may contain regex metachars.
    target_re = re.compile(
        r"\[\[" + re.escape(broken_target) + r"(?:\|([^\]]+?))?(?:#[^\]]*?)?\]\]"
    )

    def _replacement(match: re.Match) -> str:
        display = match.group(1)
        return display if display else broken_target

    new_content, n = target_re.subn(_replacement, content)
    if n == 0:
        return False

    _atomic_write(path, new_content)
    logger.info("Removed %d broken link(s) to [[%s]] from %s",
                n, broken_target, path.name)
    return True


def _add_orphans_to_moc(
    moc_path: Path, vault_root: Path, orphans: list[str]
) -> list[str]:
    """Add orphan stems under the appropriate MOC section.

    Returns the list of stems actually added (skips any that landed in the
    MOC between scan and write).
    """
    try:
        moc_text = moc_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    added: list[str] = []
    original = moc_text

    for stem in orphans:
        if f"[[{stem}]]" in moc_text:
            continue  # raced — already there

        # Resolve which subfolder this orphan lives in to choose its section.
        candidates = list(vault_root.rglob(f"{stem}.md"))
        if not candidates:
            continue
        rel_parts = candidates[0].relative_to(vault_root).parts
        subfolder = rel_parts[0] if len(rel_parts) > 1 else ""
        section = _SUBFOLDER_TO_MOC_SECTION.get(subfolder, "## Other")

        # Try to extract a display title from the page body (first H1).
        try:
            page_text = candidates[0].read_text(encoding="utf-8", errors="replace")
            title = _first_h1(page_text) or stem.replace("-", " ").title()
        except OSError:
            title = stem.replace("-", " ").title()

        entry = f"- [[{stem}]] -- {title}"
        moc_text = _insert_under_section(moc_text, section, entry)
        added.append(entry)

    if moc_text != original:
        _atomic_write(moc_path, moc_text)
        logger.info("Added %d orphan(s) to MOC", len(added))

    return added


def _first_h1(text: str) -> str:
    """Return the text of the first H1 heading, or empty string if none."""
    match = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _insert_under_section(moc_text: str, section: str, entry: str) -> str:
    """Insert an entry under a section header in the MOC.

    Mirrors wiki_writer._insert_moc_entry. Creates the section at the end
    if it doesn't exist.
    """
    idx = moc_text.find(section)
    if idx == -1:
        return moc_text.rstrip() + f"\n\n{section}\n\n{entry}\n"

    next_section = moc_text.find("\n## ", idx + len(section))
    if next_section == -1:
        return moc_text.rstrip() + f"\n{entry}\n"

    return moc_text[:next_section] + f"\n{entry}" + moc_text[next_section:]


def _atomic_write(path: Path, content: str) -> None:
    """Atomic write via temp file + os.replace (Obsidian-safe)."""
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".md.tmp")
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        os.replace(tmp_path, str(path))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
