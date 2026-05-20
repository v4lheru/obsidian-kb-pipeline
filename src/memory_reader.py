"""
src/memory_reader.py — Reads pre-distilled memory sources.

Parses three kinds of Claude Code memory:
  1. pact-memory SQLite DB (structured rows with JSON fields)
  2. Agent persistent memory (MEMORY.md indexes + linked markdown files)
  3. Per-project memory (MEMORY.md indexes + linked markdown files)

Each source is converted into a list of Extraction objects that feed into
the clusterer.  Used by pipeline.py during the Phase 1 memory-only run.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path

from .config import Paths
from .domains import get_domain_config
from .scrubber import scrub_text
from .state_store import Extraction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Credential patterns to scrub (safety net — memories shouldn't have creds)
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


# ---------------------------------------------------------------------------
# pact-memory reader
# ---------------------------------------------------------------------------

def _parse_json_field(raw: str | None) -> list[dict] | list[str]:
    """Safely parse a JSON column that may be a list of dicts or strings."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        return []
    except (json.JSONDecodeError, TypeError):
        return []


def read_pact_memory(paths: Paths) -> list[Extraction]:
    """Read all rows from the pact-memory SQLite database.

    Each row may produce multiple Extractions — one per non-empty knowledge
    field (lessons_learned, decisions, entities, reasoning_chains).
    """
    db_path = paths.pact_memory_db
    if not db_path.exists():
        logger.warning("pact-memory DB not found at %s", db_path)
        return []

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA query_only = ON;")
    rows = conn.execute(
        "SELECT id, context, goal, lessons_learned, decisions, entities, "
        "project_id, reasoning_chains FROM memories"
    ).fetchall()
    conn.close()

    extractions: list[Extraction] = []

    for row in rows:
        mem_id, context, goal, lessons_raw, decisions_raw, entities_raw, \
            project_id, reasoning_raw = row

        project_name = _project_id_to_name(project_id or "")
        base_context = scrub_text(context or "")

        # Lessons learned → gotcha / pattern extractions
        for item in _parse_json_field(lessons_raw):
            text = item if isinstance(item, str) else str(item)
            text = scrub_text(text)
            if len(text) < 10:
                continue
            topic = _infer_topic(text, project_name)
            extractions.append(Extraction(
                source_id=mem_id,
                source_type="pact_memory",
                extraction_type="lesson",
                topic=topic,
                title=text[:80],
                content=text,
                confidence=0.9,
                context=project_name,
            ))

        # Decisions → architecture / decision extractions
        for item in _parse_json_field(decisions_raw):
            if isinstance(item, dict):
                decision = scrub_text(item.get("decision", ""))
                rationale = scrub_text(item.get("rationale", ""))
                text = f"{decision}. Rationale: {rationale}" if rationale else decision
            else:
                text = scrub_text(str(item))
            if len(text) < 10:
                continue
            topic = _infer_topic(text, project_name)
            extractions.append(Extraction(
                source_id=mem_id,
                source_type="pact_memory",
                extraction_type="decision",
                topic=topic,
                title=text[:80],
                content=text,
                confidence=0.95,
                context=project_name,
            ))

        # Entities → entity fact extractions
        for item in _parse_json_field(entities_raw):
            if isinstance(item, dict):
                name = item.get("name", "")
                etype = item.get("type", "")
                notes = item.get("notes", "")
                text = scrub_text(f"{name} ({etype}): {notes}" if notes else f"{name} ({etype})")
            else:
                text = scrub_text(str(item))
            if len(text) < 10:
                continue
            topic = _infer_topic(text, project_name)
            extractions.append(Extraction(
                source_id=mem_id,
                source_type="pact_memory",
                extraction_type="entity",
                topic=topic,
                title=text[:80],
                content=text,
                confidence=0.9,
                context=project_name,
            ))

        # Reasoning chains → synthesis extractions (if non-empty)
        for item in _parse_json_field(reasoning_raw):
            text = item if isinstance(item, str) else str(item)
            text = scrub_text(text)
            if len(text) < 20:
                continue
            topic = _infer_topic(text, project_name)
            extractions.append(Extraction(
                source_id=mem_id,
                source_type="pact_memory",
                extraction_type="reasoning",
                topic=topic,
                title=text[:80],
                content=text,
                confidence=0.85,
                context=project_name,
            ))

    logger.info("Read %d extractions from pact-memory (%d rows)", len(extractions), len(rows))
    return extractions


# ---------------------------------------------------------------------------
# Agent memory reader
# ---------------------------------------------------------------------------

def read_agent_memory(paths: Paths) -> list[Extraction]:
    """Walk ~/.claude/agent-memory/*/ and parse MEMORY.md + linked files."""
    root = paths.agent_memory_root
    if not root.exists():
        logger.warning("Agent memory root not found at %s", root)
        return []

    extractions: list[Extraction] = []

    for agent_dir in sorted(root.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent_name = agent_dir.name
        memory_index = agent_dir / "MEMORY.md"
        if not memory_index.exists():
            continue

        # Parse the MEMORY.md index for inline content
        index_text = memory_index.read_text(encoding="utf-8", errors="replace")
        inline_extractions = _parse_memory_index(
            index_text, agent_name, source_type="agent_memory"
        )
        extractions.extend(inline_extractions)

        # Parse linked markdown files in the same directory
        for md_file in sorted(agent_dir.glob("*.md")):
            if md_file.name == "MEMORY.md":
                continue
            file_extractions = _parse_memory_file(
                md_file, agent_name, source_type="agent_memory"
            )
            extractions.extend(file_extractions)

    logger.info("Read %d extractions from agent memory", len(extractions))
    return extractions


# ---------------------------------------------------------------------------
# Project memory reader
# ---------------------------------------------------------------------------

def read_project_memory(paths: Paths) -> list[Extraction]:
    """Walk ~/.claude/projects/*/memory/ and parse MEMORY.md + linked files."""
    root = paths.project_memory_root
    if not root.exists():
        logger.warning("Project memory root not found at %s", root)
        return []

    extractions: list[Extraction] = []

    for project_dir in sorted(root.iterdir()):
        if not project_dir.is_dir():
            continue
        memory_dir = project_dir / "memory"
        if not memory_dir.exists():
            continue
        memory_index = memory_dir / "MEMORY.md"
        if not memory_index.exists():
            continue

        project_name = _project_id_to_name(project_dir.name)

        # Parse index
        index_text = memory_index.read_text(encoding="utf-8", errors="replace")
        inline_extractions = _parse_memory_index(
            index_text, project_name, source_type="project_memory"
        )
        extractions.extend(inline_extractions)

        # Parse linked files
        for md_file in sorted(memory_dir.glob("*.md")):
            if md_file.name == "MEMORY.md":
                continue
            file_extractions = _parse_memory_file(
                md_file, project_name, source_type="project_memory"
            )
            extractions.extend(file_extractions)

    logger.info("Read %d extractions from project memory", len(extractions))
    return extractions


# ---------------------------------------------------------------------------
# Shared parsing helpers
# ---------------------------------------------------------------------------

def _parse_memory_index(text: str, context_name: str, source_type: str) -> list[Extraction]:
    """Extract inline knowledge from a MEMORY.md index file.

    MEMORY.md files contain a mix of:
      - Markdown headers (## Section)
      - Bullet-point inline knowledge (- **Bold**: description)
      - Link entries (- [Title](file.md) -- description)  ← these point to files parsed separately
    """
    extractions: list[Extraction] = []
    current_section = ""

    for line in text.splitlines():
        line = line.strip()

        # Track section headers
        if line.startswith("## "):
            current_section = line[3:].strip()
            continue
        if line.startswith("# "):
            continue

        # Skip link-only entries (those are parsed from their files)
        if re.match(r"^-\s*\[.+\]\(.+\.md\)", line):
            continue

        # Inline knowledge bullets
        if line.startswith("- **") and "**:" in line:
            # Pattern: - **Bold Label**: Description text
            content = scrub_text(line[2:])  # strip leading "- "
            if len(content) < 15:
                continue
            topic = _infer_topic(content, current_section or context_name)
            extractions.append(Extraction(
                source_id=f"{source_type}:{context_name}:index",
                source_type=source_type,
                extraction_type="lesson",
                topic=topic,
                title=content[:80],
                content=content,
                confidence=0.85,
                context=context_name,
            ))

    return extractions


def _parse_memory_file(file_path: Path, context_name: str, source_type: str) -> list[Extraction]:
    """Parse a linked memory file with optional YAML frontmatter."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        logger.warning("Could not read %s", file_path)
        return []

    # Extract frontmatter metadata
    fm_name = ""
    fm_type = ""
    body = text

    fm_match = _FRONTMATTER_RE.match(text)
    if fm_match:
        body = text[fm_match.end():]
        for fm_line in fm_match.group(1).splitlines():
            fm_line = fm_line.strip()
            if fm_line.startswith("name:"):
                fm_name = fm_line[5:].strip().strip('"').strip("'")
            elif fm_line.startswith("type:"):
                fm_type = fm_line[5:].strip().strip('"').strip("'")

    body = scrub_text(body.strip())
    if len(body) < 20:
        return []

    # Map frontmatter type to extraction type
    ext_type_map = {
        "feedback": "lesson",
        "user": "entity",
        "project": "decision",
        "reference": "entity",
    }
    extraction_type = ext_type_map.get(fm_type, "lesson")
    title = fm_name or file_path.stem.replace("_", " ").title()
    topic = _infer_topic(body, context_name)

    return [Extraction(
        source_id=f"{source_type}:{context_name}:{file_path.name}",
        source_type=source_type,
        extraction_type=extraction_type,
        topic=topic,
        title=title[:80],
        content=body,
        confidence=0.9,
        context=context_name,
    )]


# ---------------------------------------------------------------------------
# Topic inference
# ---------------------------------------------------------------------------

# Built-in keyword -> canonical topic. Generic technologies only; user
# additions merge in via cfg.topic_keywords (checked first).
_BUILTIN_TOPIC_KEYWORDS: list[tuple[str, str]] = [
    ("n8n", "n8n"),
    ("supabase", "supabase"),
    ("okta", "okta-auth"),
    ("saml", "okta-auth"),
    ("gcp", "gcp"),
    ("kubernetes", "gcp"),
    ("gke", "gcp"),
    ("cloudsql", "gcp"),
    ("docker", "devops"),
    ("ci/cd", "devops"),
    ("railway", "devops"),
    ("launchd", "devops"),
    ("apify", "apify"),
    ("impact.com", "impact-api"),
    ("impact ", "impact-api"),
    ("monday", "monday-api"),
    ("slack", "slack"),
    ("react", "frontend"),
    ("next.js", "nextjs"),
    ("nextjs", "nextjs"),
    ("next js", "nextjs"),
    ("tailwind", "frontend"),
    ("typescript", "typescript"),
    ("drizzle", "database"),
    ("postgresql", "database"),
    ("postgres", "database"),
    ("rls", "supabase"),
    ("migration", "database"),
    ("schema", "database"),
    ("auth", "auth"),
    ("jwt", "auth"),
    ("oauth", "auth"),
    ("security", "security"),
    ("credential", "security"),
    ("mcp", "mcp-sdk"),
    ("fastmcp", "mcp-sdk"),
    ("apps script", "google-apps-script"),
    ("google apps", "google-apps-script"),
    ("tiptap", "frontend"),
    ("bullmq", "backend"),
    ("redis", "backend"),
    ("fastify", "backend"),
    ("pact", "pact-framework"),
    ("claude", "claude-ai"),
    ("anthropic", "claude-ai"),
    ("openai", "openai"),
    ("linkedin", "social-media"),
    ("instagram", "social-media"),
    ("facebook", "social-media"),
    ("twitter", "social-media"),
    ("reddit", "reddit"),
    ("seo", "seo"),
    ("monorepo", "monorepo"),
    ("turborepo", "monorepo"),
    ("pnpm", "monorepo"),
    ("npm workspace", "monorepo"),
]


def _infer_topic(text: str, fallback: str) -> str:
    """Infer a canonical topic from text content via keyword matching.

    User-config keywords are checked first so user terminology wins on overlap.
    """
    cfg = get_domain_config()
    lower = text.lower()
    for tk in cfg.topic_keywords:
        if tk.keyword in lower:
            return tk.topic
    for keyword, topic in _BUILTIN_TOPIC_KEYWORDS:
        if keyword in lower:
            return topic
    # Fall back to the context/project name, cleaned up
    return _clean_topic(fallback)


def _clean_topic(raw: str) -> str:
    """Normalize a raw topic string to a clean kebab-case slug."""
    cleaned = raw.lower().strip()
    cleaned = re.sub(r"[^a-z0-9\s-]", "", cleaned)
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = cleaned.strip("-")
    return cleaned or "general"


def _project_id_to_name(project_id: str) -> str:
    """Convert a Claude project ID path to a human-readable name.

    Input:  '-Users-alice-Documents-projects-my-app'
    Output: 'my-app'

    The skip set combines built-in generics with cfg.path_skip_prefixes
    (where users place their macOS short username and personal folder names).
    """
    # Take the last meaningful segment
    parts = project_id.replace("-", " ").split()
    # Skip common path components (built-in + user-config)
    skip = {"users", "documents", "personal", "projects", "github", "repos"} | get_domain_config().path_skip_prefixes
    meaningful = [p for p in parts if p.lower() not in skip]
    if not meaningful:
        return project_id
    # Take the last 2-4 meaningful words
    name_parts = meaningful[-4:]
    return " ".join(word.title() for word in name_parts if word)
