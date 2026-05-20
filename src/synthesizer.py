"""
src/synthesizer.py — Template-based wiki page generator (Phase 1).

Converts PageCandidate objects (clusters of extractions) into WikiPage
objects with Obsidian-compatible markdown.  For Phase 1, synthesis is
purely template-based — memories are already distilled knowledge that
just needs formatting, deduplication, and cross-referencing.

Phase 2 will add LLM-based synthesis via Opus for richer page generation.
Used by pipeline.py after clustering.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date

from .clusterer import PageCandidate
from .domains import get_domain_config
from .state_store import Extraction

logger = logging.getLogger(__name__)

_TODAY = date.today().isoformat()

# Default page-size cap. Mirrors SynthesisConfig.max_page_words but is kept
# here as a literal so existing call-sites (and tests) that don't pass
# max_page_words still get the guard.
_DEFAULT_MAX_PAGE_WORDS = 3000


@dataclass
class WikiPage:
    """A fully rendered wiki page ready for writing to disk."""

    path: str              # relative path within Coding-Notes/
    title: str
    frontmatter: dict[str, object] = field(default_factory=dict)
    body: str = ""
    wikilinks: list[str] = field(default_factory=list)
    extraction_ids: list[str] = field(default_factory=list)
    word_count: int = 0
    is_new: bool = True


def synthesize_page(
    candidate: PageCandidate,
    max_page_words: int = _DEFAULT_MAX_PAGE_WORDS,
) -> WikiPage:
    """Generate a WikiPage from a PageCandidate.

    For updates: integrates new extractions into the existing page body
    (additive only — no existing content is removed). If the existing body
    is already at or above max_page_words, new extractions are skipped and
    logged so they surface in the next quality-check report.
    For creates: generates a full page from scratch.
    """
    if candidate.action == "update" and candidate.existing_content:
        return _synthesize_update(candidate, max_page_words=max_page_words)
    return _synthesize_new(candidate, max_page_words=max_page_words)


def _synthesize_new(
    candidate: PageCandidate,
    max_page_words: int = _DEFAULT_MAX_PAGE_WORDS,
) -> WikiPage:
    """Build a brand-new wiki page from extractions."""
    sections = _group_extractions_by_type(candidate.extractions)
    body_parts: list[str] = []
    all_wikilinks: list[str] = []

    header = f"# {candidate.title}\n"
    body_parts.append(header)
    words_used = len(header.split())
    skipped_count = 0

    for section_title, items in sections:
        section_header = f"\n## {section_title}\n"
        section_words = len(section_header.split())
        if words_used + section_words > max_page_words:
            skipped_count += len(items)
            continue
        body_parts.append(section_header)
        words_used += section_words
        for ext in items:
            formatted = _format_extraction(ext)
            ext_words = len(formatted.split())
            if words_used + ext_words > max_page_words:
                skipped_count += 1
                continue
            body_parts.append(formatted)
            words_used += ext_words
            all_wikilinks.extend(_extract_wikilinks(ext.content))

    if skipped_count > 0:
        logger.warning(
            "Dropped %d extractions creating %s: would exceed %d-word cap.",
            skipped_count, candidate.target_path, max_page_words,
        )

    # Related section
    related = _build_related_section(candidate, all_wikilinks)
    if related:
        body_parts.append(f"\n## Related\n\n{related}")

    body = "\n".join(body_parts)
    word_count = len(body.split())

    frontmatter = _build_frontmatter(
        candidate, wiki_source="auto", is_new=True
    )

    return WikiPage(
        path=candidate.target_path,
        title=candidate.title,
        frontmatter=frontmatter,
        body=body,
        wikilinks=sorted(set(all_wikilinks)),
        extraction_ids=[e.id for e in candidate.extractions],
        word_count=word_count,
        is_new=True,
    )


def _synthesize_update(
    candidate: PageCandidate,
    max_page_words: int = _DEFAULT_MAX_PAGE_WORDS,
) -> WikiPage:
    """Integrate new extractions into an existing page (additive only).

    Page-size guard: if existing body is already at or above max_page_words,
    refuse to grow it. The new extractions are skipped (NOT silently dropped —
    they're logged, and the existing body is returned unchanged). The next
    quality-check run will flag the page as oversized for manual split.
    """
    existing = candidate.existing_content
    all_wikilinks: list[str] = []

    # Separate frontmatter from body
    existing_fm, existing_body = _split_frontmatter(existing)

    # Page-size guard: if already at the cap, refuse to grow.
    existing_word_count = len(existing_body.split())
    if existing_word_count >= max_page_words:
        logger.warning(
            "Skipping %d extractions for %s: page already at %d words "
            "(cap %d). Page will be flagged in next quality-check.",
            len(candidate.extractions), candidate.target_path,
            existing_word_count, max_page_words,
        )
        return WikiPage(
            path=candidate.target_path,
            title=candidate.title,
            frontmatter=_parse_existing_frontmatter(existing_fm, candidate),
            body=existing_body,
            wikilinks=[],
            extraction_ids=[],
            word_count=existing_word_count,
            is_new=False,
        )

    # Deduplicate: skip extractions whose key content is already in the page
    new_extractions = _deduplicate_extractions(candidate.extractions, existing_body)

    if not new_extractions:
        # Nothing new to add — return existing page as-is
        return WikiPage(
            path=candidate.target_path,
            title=candidate.title,
            frontmatter=_parse_existing_frontmatter(existing_fm, candidate),
            body=existing_body,
            wikilinks=[],
            extraction_ids=[],
            word_count=len(existing_body.split()),
            is_new=False,
        )

    # Budget: how many words we can still add before hitting the cap.
    word_budget = max(0, max_page_words - existing_word_count)

    # Build an addendum section with new extractions, stopping when budget
    # is exhausted so a single synthesis pass can't blow past the cap.
    addendum_parts: list[str] = []
    words_added = 0
    sections = _group_extractions_by_type(new_extractions)
    skipped_count = 0

    for section_title, items in sections:
        section_header = f"## {section_title}"
        if section_header in existing_body:
            budget_items: list[str] = []
            for ext in items:
                formatted = _format_extraction(ext)
                ext_words = len(formatted.split())
                if words_added + ext_words > word_budget:
                    skipped_count += 1
                    continue
                budget_items.append(formatted)
                words_added += ext_words
            if budget_items:
                insertion = "\n".join(budget_items)
                existing_body = _insert_under_section(existing_body, section_title, insertion)
        else:
            section_parts: list[str] = []
            for ext in items:
                formatted = _format_extraction(ext)
                ext_words = len(formatted.split())
                if words_added + ext_words > word_budget:
                    skipped_count += 1
                    continue
                section_parts.append(formatted)
                words_added += ext_words
                all_wikilinks.extend(_extract_wikilinks(ext.content))
            if section_parts:
                addendum_parts.append(f"\n## {section_title}\n")
                addendum_parts.extend(section_parts)

    if skipped_count > 0:
        logger.warning(
            "Dropped %d extractions for %s: page would exceed %d-word cap.",
            skipped_count, candidate.target_path, max_page_words,
        )

    # Append new sections before the Related section (if it exists)
    if addendum_parts:
        addendum = "\n".join(addendum_parts)
        existing_body = _insert_before_related(existing_body, addendum)

    # Update Related section
    new_related_links = _build_related_links(candidate, all_wikilinks)
    if new_related_links:
        existing_body = _merge_related_section(existing_body, new_related_links)

    frontmatter = _parse_existing_frontmatter(existing_fm, candidate)
    frontmatter["modified"] = _TODAY
    if "wiki-source" not in frontmatter:
        frontmatter["wiki-source"] = "seed"

    word_count = len(existing_body.split())

    return WikiPage(
        path=candidate.target_path,
        title=candidate.title,
        frontmatter=frontmatter,
        body=existing_body,
        wikilinks=sorted(set(all_wikilinks)),
        extraction_ids=[e.id for e in new_extractions],
        word_count=word_count,
        is_new=False,
    )


# ---------------------------------------------------------------------------
# Extraction formatting
# ---------------------------------------------------------------------------

def _group_extractions_by_type(
    extractions: list[Extraction],
) -> list[tuple[str, list[Extraction]]]:
    """Group extractions into section buckets by type."""
    type_to_section = {
        "lesson": "Lessons Learned",
        "decision": "Key Decisions",
        "entity": "Key Components",
        "reasoning": "Design Reasoning",
        "pattern": "Patterns",
        "gotcha": "Gotchas",
        "api_quirk": "API Quirks",
        "process_step": "Workflows",
    }
    groups: dict[str, list[Extraction]] = {}
    for ext in extractions:
        section = type_to_section.get(ext.extraction_type, "Notes")
        groups.setdefault(section, []).append(ext)

    # Return in a stable order
    section_order = [
        "Key Decisions", "Key Components", "Lessons Learned",
        "Patterns", "Gotchas", "API Quirks", "Workflows",
        "Design Reasoning", "Notes",
    ]
    result = []
    for section in section_order:
        if section in groups:
            result.append((section, groups[section]))
    return result


def _format_extraction(ext: Extraction) -> str:
    """Format a single extraction as a markdown bullet or block."""
    content = ext.content.strip()

    # If the content is already multi-line with structure, use it as-is
    if "\n" in content and len(content) > 200:
        context_tag = f" *(from {ext.context})*" if ext.context else ""
        return f"\n### {ext.title}{context_tag}\n\n{content}\n"

    # Short content: format as a bullet point
    context_tag = f" *(from {ext.context})*" if ext.context else ""
    return f"- {content}{context_tag}\n"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _normalize_for_compare(text: str) -> str:
    """Lowercase and collapse whitespace for similarity comparison.

    Idempotency depends on the same content producing the same signature
    even when whitespace, capitalization, or punctuation drift between runs.
    """
    return re.sub(r"\s+", " ", text.lower()).strip()


def _deduplicate_extractions(
    new: list[Extraction], existing_body: str
) -> list[Extraction]:
    """Remove extractions whose key content is already in the page.

    Stronger than first-60-chars: normalizes whitespace + uses 120-char
    signature, falls back to title-match for long titles. Catches
    near-duplicates from re-runs where minor whitespace differs.
    """
    existing_norm = _normalize_for_compare(existing_body)
    deduplicated = []

    for ext in new:
        content = ext.content.strip()
        if not content:
            continue

        # Primary signature: first 120 chars (whitespace-normalized).
        # Wider window than the original 60 catches paraphrases where the
        # opening sentence drifts slightly.
        norm_content = _normalize_for_compare(content)
        sig = norm_content[:120]
        if sig and sig in existing_norm:
            continue

        # Secondary signature: title (only if long enough to be specific).
        # Short titles like "T", "RLS", "DB" would false-positive.
        title_norm = _normalize_for_compare(ext.title)
        if len(title_norm) >= 12 and title_norm in existing_norm:
            continue

        deduplicated.append(ext)

    return deduplicated


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------

def _build_frontmatter(
    candidate: PageCandidate, wiki_source: str, is_new: bool
) -> dict[str, object]:
    """Build frontmatter dict for a new wiki page."""
    # Collect tags from extractions
    tags = {"wiki", candidate.page_type}
    for ext in candidate.extractions:
        if ext.topic:
            tags.add(ext.topic)

    return {
        "type": "resource",
        "tags": sorted(tags),
        "status": "draft" if is_new else "reference",
        "created": _TODAY,
        "modified": _TODAY,
        "area": "work",
        "wiki-source": wiki_source,
        "wiki-confidence": "high",
    }


def _parse_existing_frontmatter(
    fm_text: str, candidate: PageCandidate
) -> dict[str, object]:
    """Parse YAML frontmatter from an existing page into a dict.

    Uses simple line-by-line parsing to avoid a PyYAML dependency.
    """
    result: dict[str, object] = {}
    if not fm_text.strip():
        return _build_frontmatter(candidate, wiki_source="seed", is_new=False)

    current_key = ""
    current_list: list[str] = []

    for line in fm_text.splitlines():
        line_stripped = line.strip()
        if line_stripped in ("---", ""):
            continue

        if line_stripped.startswith("- ") and current_key:
            current_list.append(line_stripped[2:].strip())
            continue

        # Flush any pending list
        if current_key and current_list:
            result[current_key] = current_list
            current_list = []

        if ":" in line_stripped:
            key, _, value = line_stripped.partition(":")
            key = key.strip()
            value = value.strip()
            current_key = key
            if value:
                result[key] = value
            else:
                current_list = []

    # Flush final list
    if current_key and current_list:
        result[current_key] = current_list

    return result


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split a markdown file into (frontmatter_text, body_text)."""
    if not text.startswith("---"):
        return "", text

    end = text.find("\n---", 3)
    if end == -1:
        return "", text

    # Find the end of the closing ---
    body_start = text.index("\n", end + 1) + 1 if end + 4 < len(text) else end + 4
    return text[: end + 4], text[body_start:]


# ---------------------------------------------------------------------------
# Section manipulation
# ---------------------------------------------------------------------------

def _insert_under_section(body: str, section_title: str, content: str) -> str:
    """Append content at the end of an existing section."""
    header = f"## {section_title}"
    idx = body.find(header)
    if idx == -1:
        return body + "\n" + content

    # Find the end of this section (next ## or end of file)
    next_section = body.find("\n## ", idx + len(header))
    if next_section == -1:
        # Check for Related section
        related_idx = body.find("\n## Related", idx + len(header))
        if related_idx != -1:
            return body[:related_idx] + "\n" + content + body[related_idx:]
        return body + "\n" + content

    return body[:next_section] + "\n" + content + body[next_section:]


def _insert_before_related(body: str, content: str) -> str:
    """Insert content before the ## Related section, or append at end."""
    related_idx = body.find("\n## Related")
    if related_idx != -1:
        return body[:related_idx] + "\n" + content + body[related_idx:]
    return body + "\n" + content


def _merge_related_section(body: str, new_links: list[str]) -> str:
    """Add new links to the existing Related section."""
    related_idx = body.find("## Related")
    if related_idx == -1:
        return body + "\n\n## Related\n\n" + "\n".join(new_links) + "\n"

    # Find the end of the Related section
    next_section = body.find("\n## ", related_idx + 10)
    if next_section == -1:
        existing_related = body[related_idx:]
    else:
        existing_related = body[related_idx:next_section]

    # Add only links not already present
    for link in new_links:
        link_target = link.split("]]")[0] if "]]" in link else link
        if link_target not in existing_related:
            existing_related = existing_related.rstrip() + "\n" + link

    if next_section == -1:
        return body[:related_idx] + existing_related + "\n"
    return body[:related_idx] + existing_related + body[next_section:]


# ---------------------------------------------------------------------------
# Wikilinks
# ---------------------------------------------------------------------------

# Built-in generic wikilink targets. User config (cfg.wikilink_targets) wins
# on key collision via _merged_wikilink_targets().
_BUILTIN_WIKILINK_TARGETS: dict[str, str] = {
    "n8n": "n8n-workflow-patterns",
    "supabase": "database-schema-patterns",
    "apify": "apify-platform-reference",
    "impact": "impact-com-api-reference",
    "gcp": "monorepo-and-devops-patterns",
    "docker": "monorepo-and-devops-patterns",
    "okta": "arch-okta-auth-pattern",
    "react": "frontend-patterns",
    "next.js": "frontend-patterns",
    "tailwind": "frontend-patterns",
    "drizzle": "database-schema-patterns",
    "postgresql": "database-schema-patterns",
    "bullmq": "backend-patterns",
    "fastify": "backend-patterns",
    "redis": "backend-patterns",
    "mcp": "backend-patterns",
    "apps script": "backend-patterns",
    "reddit": "reddit-shadow-ban-research",
}


def _merged_wikilink_targets() -> dict[str, str]:
    """Built-in defaults overlaid with user config (user wins on collision)."""
    return {**_BUILTIN_WIKILINK_TARGETS, **get_domain_config().wikilink_targets}


def _extract_wikilinks(text: str) -> list[str]:
    """Find existing [[wikilink]] references in text."""
    return re.findall(r"\[\[([^\]]+)\]\]", text)


def _build_related_section(
    candidate: PageCandidate, extra_links: list[str]
) -> str:
    """Build the ## Related section content."""
    links = _build_related_links(candidate, extra_links)
    if not links:
        return ""
    return "\n".join(links) + "\n"


def _build_related_links(
    candidate: PageCandidate, extra_links: list[str]
) -> list[str]:
    """Determine which wikilinks should appear in the Related section."""
    # Start with any explicit wikilinks found in extraction content
    related: set[str] = set(extra_links)

    # Add topic-based cross-references
    seen_topics: set[str] = set()
    targets = _merged_wikilink_targets()
    for ext in candidate.extractions:
        lower = ext.content.lower()
        for keyword, target in targets.items():
            if keyword in lower and target not in candidate.target_path:
                seen_topics.add(target)

    related.update(seen_topics)

    # Remove self-references
    self_stem = candidate.target_path.rsplit("/", 1)[-1].replace(".md", "")
    related.discard(self_stem)

    return [f"- [[{link}]]" for link in sorted(related)]
