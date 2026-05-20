"""
src/clusterer.py — Groups extractions by topic into wiki page candidates.

Takes a flat list of Extraction objects and groups them by canonical topic,
then maps each group to an existing wiki page (update) or proposes a new
page (create).  Used by pipeline.py between the memory reader and synthesizer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import ClusterConfig
from .domains import get_domain_config
from .state_store import Extraction

logger = logging.getLogger(__name__)


@dataclass
class PageCandidate:
    """A proposed wiki page (new or update) with its source extractions."""

    action: str                          # "create" | "update"
    page_type: str                       # entity, pattern, gotcha, architecture, etc.
    topic: str                           # canonical topic slug
    title: str                           # human-readable page title
    target_path: str                     # relative path within Coding-Notes/
    extractions: list[Extraction] = field(default_factory=list)
    existing_content: str = ""           # current page body if action == "update"


# Maps canonical topics to (page_type, subfolder, filename, title).
#
# Routing principle: each major topic gets its OWN focused page so a single
# catch-all (e.g. backend-patterns.md) does not grow unbounded. Topics that
# share a logical surface (supabase + database + drizzle + postgres) still
# merge onto one page; truly distinct topics route to their own page.
#
# Unknown topics fall through to Gotchas/{topic}.md (handled in
# cluster_extractions below). User overrides merge over this built-in via
# get_domain_config().topic_page_map.
_BUILTIN_TOPIC_PAGE_MAP: dict[str, tuple[str, str, str, str]] = {
    # n8n + workflow automation
    "n8n": ("pattern", "n8n", "n8n-workflow-patterns.md", "n8n Workflow Patterns"),
    "apify": ("entity", "n8n", "n8n-apify-integration.md", "n8n + Apify Integration"),

    # Database surface (intentionally co-located on one page)
    "supabase": ("pattern", "Patterns", "database-schema-patterns.md",
                 "Database Schema Patterns"),
    "database": ("pattern", "Patterns", "database-schema-patterns.md",
                 "Database Schema Patterns"),

    # Frontend surface (intentionally co-located on one page)
    "nextjs": ("pattern", "Patterns", "frontend-patterns.md", "Frontend Patterns"),
    "frontend": ("pattern", "Patterns", "frontend-patterns.md", "Frontend Patterns"),

    # Backend — focused per-topic pages (was: all → backend-patterns.md catch-all)
    "backend": ("pattern", "Patterns", "backend-patterns.md", "Backend Coding Patterns"),
    "typescript": ("pattern", "Patterns", "typescript-patterns.md",
                   "TypeScript Patterns"),
    "google-apps-script": ("pattern", "Patterns", "google-apps-script-patterns.md",
                           "Google Apps Script Patterns"),
    "mcp-sdk": ("pattern", "Patterns", "mcp-sdk-patterns.md", "MCP SDK Patterns"),
    "auth": ("pattern", "Patterns", "auth-patterns.md", "Authentication Patterns"),
    "security": ("pattern", "Patterns", "security-patterns.md", "Security Patterns"),
    "seo": ("pattern", "Patterns", "seo-patterns.md", "SEO Patterns"),
    "pact-framework": ("pattern", "Patterns", "pact-framework-patterns.md",
                       "PACT Framework Patterns"),

    # DevOps surface (intentionally co-located on one page)
    "devops": ("pattern", "Patterns", "monorepo-and-devops-patterns.md",
               "Monorepo & DevOps Patterns"),
    "monorepo": ("pattern", "Patterns", "monorepo-and-devops-patterns.md",
                 "Monorepo & DevOps Patterns"),
    "gcp": ("pattern", "Patterns", "monorepo-and-devops-patterns.md",
            "Monorepo & DevOps Patterns"),

    # External APIs
    "impact-api": ("entity", "APIs", "impact-com-api-reference.md",
                   "Impact.com API Reference"),
    "monday-api": ("entity", "APIs", "third-party-api-misc.md",
                   "Third-Party API Miscellaneous"),
    "slack": ("entity", "APIs", "third-party-api-misc.md",
              "Third-Party API Miscellaneous"),
    "claude-ai": ("entity", "APIs", "third-party-api-misc.md",
                  "Third-Party API Miscellaneous"),
    "openai": ("entity", "APIs", "third-party-api-misc.md",
               "Third-Party API Miscellaneous"),
    "social-media": ("entity", "APIs", "apify-platform-reference.md",
                     "Apify Platform Reference"),

    # Architecture
    "okta-auth": ("architecture", "Architecture", "arch-okta-auth-pattern.md",
                  "Okta Auth Pattern"),

    # Domain research
    "reddit": ("entity", "", "reddit-shadow-ban-research.md", "Reddit Shadow Ban Research"),
}


def _resolve_page(topic: str) -> tuple[str, str, str, str] | None:
    """Look up a topic in the user config first, then the built-in map."""
    cfg = get_domain_config()
    if topic in cfg.topic_page_map:
        d = cfg.topic_page_map[topic]
        return (d.page_type, d.subfolder, d.filename, d.title)
    return _BUILTIN_TOPIC_PAGE_MAP.get(topic)


def cluster_extractions(
    extractions: list[Extraction],
    vault_root: Path,
    config: ClusterConfig,
) -> list[PageCandidate]:
    """Group extractions by topic and map to wiki page candidates.

    1. Group by canonical topic
    2. Map each group to an existing page (update) or new page (create)
    3. Apply threshold: new pages need >= min_extractions_for_new_page
    """
    # Step 1: group by topic
    topic_groups: dict[str, list[Extraction]] = {}
    for ext in extractions:
        topic = ext.topic
        topic_groups.setdefault(topic, []).append(ext)

    logger.info("Clustered %d extractions into %d topic groups",
                len(extractions), len(topic_groups))

    # Step 2: map each group to a page candidate
    candidates: list[PageCandidate] = []

    for topic, group in sorted(topic_groups.items()):
        mapping = _resolve_page(topic)

        if mapping:
            page_type, subfolder, filename, title = mapping
            rel_path = f"{subfolder}/{filename}" if subfolder else filename
        else:
            # Unknown topic — create a new page in Gotchas/ if enough extractions
            page_type = "gotcha"
            filename = f"gotcha-{topic}.md"
            rel_path = f"Gotchas/{filename}"
            title = f"{topic.replace('-', ' ').title()} Gotchas"

        # Check if the target page already exists on disk
        full_path = vault_root / rel_path
        existing_content = ""
        action = "create"

        if full_path.exists():
            action = "update"
            try:
                existing_content = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                existing_content = ""

        # Threshold check for new pages only
        if action == "create" and len(group) < config.min_extractions_for_new_page:
            logger.debug(
                "Skipping topic '%s' — only %d extractions (need %d for new page)",
                topic, len(group), config.min_extractions_for_new_page,
            )
            continue

        candidates.append(PageCandidate(
            action=action,
            page_type=page_type,
            topic=topic,
            title=title,
            target_path=rel_path,
            extractions=group,
            existing_content=existing_content,
        ))

    # Merge candidates targeting the same page
    candidates = _merge_same_page_candidates(candidates)

    logger.info("Produced %d page candidates (%d create, %d update)",
                len(candidates),
                sum(1 for c in candidates if c.action == "create"),
                sum(1 for c in candidates if c.action == "update"))

    return candidates


def _merge_same_page_candidates(candidates: list[PageCandidate]) -> list[PageCandidate]:
    """Merge candidates that target the same file path."""
    by_path: dict[str, PageCandidate] = {}

    for c in candidates:
        if c.target_path in by_path:
            existing = by_path[c.target_path]
            existing.extractions.extend(c.extractions)
        else:
            by_path[c.target_path] = c

    return list(by_path.values())
