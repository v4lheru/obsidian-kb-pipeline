"""
src/wiki_writer.py — Writes WikiPage objects to the Obsidian vault.

Handles:
  - Rendering YAML frontmatter
  - Atomic file writes (temp file → rename) to prevent corruption
  - Creating subdirectories as needed (Gotchas/, Workflows/, Comparisons/)
  - Updating the MOC (coding-knowledge-map.md)

Used by pipeline.py as the final stage of the Phase 1 pipeline.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path

from .synthesizer import WikiPage

logger = logging.getLogger(__name__)

# Claude model used for MOC description generation.
_LLM_MODEL = "claude-haiku-4-5-20251001"
_LLM_SYSTEM_PROMPT = (
    "Generate a 15-25 word description for a wiki index entry. "
    "Lead with the project name and technology stack (e.g. 'Google Apps Script partner invoicing', "
    "'Next.js + Supabase dashboard'), then name 2-3 specific gotchas or patterns. "
    "An agent must be able to tell from your description alone whether this page is relevant to their task. "
    "No generic labels like 'Gotchas' or 'Patterns'. Output only the description, nothing else."
)


class WikiWriter:
    """Writes wiki pages to the Obsidian vault with atomic file operations."""

    def __init__(self, vault_root: Path) -> None:
        self._vault_root = vault_root

    def write_page(self, page: WikiPage) -> Path:
        """Write a single wiki page to disk (atomic write).

        Returns the absolute path of the written file.
        """
        target = self._vault_root / page.path
        target.parent.mkdir(parents=True, exist_ok=True)

        content = _render_page(page)

        # Atomic write: temp file in same directory, then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(target.parent), suffix=".md.tmp"
        )
        try:
            os.write(fd, content.encode("utf-8"))
            os.close(fd)
            os.replace(tmp_path, str(target))
        except Exception:
            os.close(fd) if not os.get_inheritable(fd) else None
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        action = "Created" if page.is_new else "Updated"
        logger.info("%s wiki page: %s (%d words)", action, page.path, page.word_count)
        return target

    def write_pages(self, pages: list[WikiPage]) -> list[Path]:
        """Write multiple pages. Returns list of written paths."""
        written = []
        for page in pages:
            path = self.write_page(page)
            written.append(path)
        return written

    def update_moc(self, pages: list[WikiPage]) -> None:
        """Update the coding-knowledge-map.md MOC with new page entries.

        Preserves existing entries.  Adds new entries under the appropriate
        section based on the page's subfolder.
        """
        moc_path = self._vault_root / "coding-knowledge-map.md"
        if not moc_path.exists():
            logger.warning("MOC not found at %s — skipping update", moc_path)
            return

        moc_text = moc_path.read_text(encoding="utf-8", errors="replace")
        original_text = moc_text

        for page in pages:
            if not page.is_new:
                continue  # Only add MOC entries for new pages

            stem = page.path.rsplit("/", 1)[-1].replace(".md", "")
            summary = _llm_summarize_page_for_moc(page.body, page.title)
            if not summary:
                summary = _summarize_page_for_moc(page.body)
            entry = f"- [[{stem}]] — {summary}" if summary else f"- [[{stem}]] — {page.title}"

            # Skip if already in MOC
            if f"[[{stem}]]" in moc_text:
                continue

            # Determine which section to add under
            section = _moc_section_for_page(page)
            moc_text = _insert_moc_entry(moc_text, section, entry)

        if moc_text != original_text:
            # Atomic write for MOC too
            fd, tmp_path = tempfile.mkstemp(
                dir=str(moc_path.parent), suffix=".md.tmp"
            )
            try:
                os.write(fd, moc_text.encode("utf-8"))
                os.close(fd)
                os.replace(tmp_path, str(moc_path))
            except Exception:
                os.close(fd) if not os.get_inheritable(fd) else None
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
            logger.info("Updated MOC with %d new entries",
                        sum(1 for p in pages if p.is_new))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_page(page: WikiPage) -> str:
    """Render a WikiPage to a complete markdown string with frontmatter."""
    parts = ["---\n"]
    parts.append(_render_frontmatter(page.frontmatter))
    parts.append("---\n\n")
    parts.append(page.body.strip())
    parts.append("\n")
    return "".join(parts)


def _render_frontmatter(fm: dict[str, object]) -> str:
    """Render a frontmatter dict to YAML string (simple, no PyYAML dep)."""
    lines: list[str] = []

    # Render in a stable key order
    key_order = [
        "type", "tags", "status", "created", "modified", "area", "tool",
        "wiki-source", "wiki-confidence", "wiki-sources",
    ]

    rendered_keys: set[str] = set()
    for key in key_order:
        if key in fm:
            _render_fm_value(lines, key, fm[key])
            rendered_keys.add(key)

    # Render any remaining keys
    for key, value in fm.items():
        if key not in rendered_keys:
            _render_fm_value(lines, key, value)

    return "\n".join(lines) + "\n"


def _render_fm_value(lines: list[str], key: str, value: object) -> None:
    """Render a single frontmatter key-value pair."""
    if isinstance(value, list):
        lines.append(f"{key}:")
        for item in value:
            lines.append(f"  - {item}")
    elif isinstance(value, bool):
        lines.append(f"{key}: {'true' if value else 'false'}")
    elif isinstance(value, (int, float)):
        lines.append(f"{key}: {value}")
    else:
        lines.append(f"{key}: {value}")


# ---------------------------------------------------------------------------
# MOC helpers
# ---------------------------------------------------------------------------

# Map subfolder to MOC section header
_SUBFOLDER_TO_SECTION: dict[str, str] = {
    "n8n": "## n8n Workflow Automation",
    "APIs": "## API Integration References",
    "Architecture": "## Architecture Decisions",
    "Patterns": "## Cross-Cutting Patterns",
    "Gotchas": "## Gotchas & Debugging",
    "Workflows": "## Workflows & Processes",
    "Comparisons": "## Technology Comparisons",
    "Documentation": "## Security & Research",
}


def _moc_section_for_page(page: WikiPage) -> str:
    """Determine which MOC section a page belongs to."""
    # Extract subfolder from path
    parts = page.path.split("/")
    if len(parts) >= 2:
        subfolder = parts[0]
        section = _SUBFOLDER_TO_SECTION.get(subfolder)
        if section:
            return section

    return "## Other"


def _insert_moc_entry(moc_text: str, section: str, entry: str) -> str:
    """Insert a MOC entry under the given section header.

    If the section doesn't exist, creates it at the end.
    """
    idx = moc_text.find(section)
    if idx == -1:
        # Create new section at the end
        return moc_text.rstrip() + f"\n\n{section}\n\n{entry}\n"

    # Find the end of this section (next ## or end of file)
    next_section = moc_text.find("\n## ", idx + len(section))
    if next_section == -1:
        # Append at end of file
        return moc_text.rstrip() + f"\n{entry}\n"

    # Insert before the next section
    return moc_text[:next_section] + f"\n{entry}" + moc_text[next_section:]


def _llm_summarize_page_for_moc(body: str, title: str) -> str:
    """Generate a MOC description via Claude Haiku.

    Returns an empty string if the `anthropic` package is missing, the
    `ANTHROPIC_API_KEY` env var is unset, or the API call fails. Callers
    should fall back to the heuristic summarizer in those cases.
    """
    if not body:
        return ""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return ""

    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError:
        return ""

    try:
        client = anthropic.Anthropic()
        user_content = f"Title: {title}\n\n{body}"
        response = client.messages.create(
            model=_LLM_MODEL,
            max_tokens=60,
            temperature=0,
            system=_LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                return text.strip()
        return ""
    except Exception as exc:
        logger.warning("LLM MOC summarization failed for %r: %s", title, exc)
        return ""


# Headings that don't add meaning to a MOC summary
_SKIP_HEADINGS = {
    "related", "notes", "references", "see also", "links",
    "key components", "key decisions", "lessons learned",
    "patterns", "gotchas", "api quirks", "workflows",
    "design reasoning",
}

_HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_BULLET_RE = re.compile(r"^[-*]\s+(.+)$", re.MULTILINE)


def _summarize_page_for_moc(body: str) -> str:
    """Build a concise MOC description from page body content.

    Extracts key topics from ## headings and leading bullet points to produce
    a comma-separated summary (max ~120 chars) that tells an agent what the
    page contains.
    """
    if not body:
        return ""

    topics: list[str] = []

    # 1. Collect meaningful ## headings
    for m in _HEADING_RE.finditer(body):
        heading = m.group(1).strip()
        if heading.lower() not in _SKIP_HEADINGS:
            topics.append(heading)

    # 2. If few headings, pull leading bullets for more context
    if len(topics) < 3:
        for m in _BULLET_RE.finditer(body):
            line = m.group(1).strip()
            # Take the first clause before any parenthetical or long detail
            short = re.split(r"[;(—]", line)[0].strip().rstrip(".,:")
            if len(short) > 10 and short not in topics:
                topics.append(short)
            if len(topics) >= 6:
                break

    if not topics:
        return ""

    # Build summary: join topics, truncate to ~120 chars
    summary = ", ".join(topics)
    if len(summary) > 120:
        # Cut at last comma before 120 chars
        cut = summary[:120].rfind(",")
        if cut > 30:
            summary = summary[:cut]
        else:
            summary = summary[:117] + "..."

    return summary
