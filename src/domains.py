"""
src/domains.py -- Taxonomy loader for the obsidian-kb-pipeline.

Defines the DomainConfig dataclass family that holds user + built-in
taxonomy (project patterns, topic canonicalization, page-routing, etc.),
the YAML loader that merges user overrides over built-in generic defaults,
and the module-level singleton that the rest of the pipeline reads from.

Used by classifier, clusterer, session_extractor, memory_reader, and
synthesizer via `from .domains import get_domain_config`. Pipeline.main()
calls load_domain_config + set_domain_config at startup so all downstream
stages see the same config.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    """Raised when an explicit config override is missing or malformed."""


@dataclass(frozen=True)
class DomainPattern:
    """Single keyword->domain entry from project_domain_patterns."""
    keyword: str
    domain: str


@dataclass(frozen=True)
class TopicKeyword:
    """Single keyword->topic entry from topic_keywords."""
    keyword: str
    topic: str


@dataclass(frozen=True)
class TopicPageDescriptor:
    """Single value object from topic_page_map."""
    page_type: str
    subfolder: str
    filename: str
    title: str


@dataclass(frozen=True)
class DomainConfig:
    """Loaded + merged taxonomy. Modules read sections from this object."""
    project_domain_patterns: tuple[DomainPattern, ...] = ()
    specific_domain_topics: frozenset[str] = frozenset()
    topic_canonicalization: dict[str, str] = field(default_factory=dict)
    topic_keywords: tuple[TopicKeyword, ...] = ()
    topic_page_map: dict[str, TopicPageDescriptor] = field(default_factory=dict)
    wikilink_targets: dict[str, str] = field(default_factory=dict)
    path_skip_prefixes: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Built-in defaults (generic taxonomy only — no personal entries)
# ---------------------------------------------------------------------------

BUILTIN_DOMAIN_CONFIG = DomainConfig(
    project_domain_patterns=(
        DomainPattern("n8n", "n8n"),
        DomainPattern("slack", "slack"),
        DomainPattern("hubspot", "hubspot"),
        DomainPattern("monday", "monday"),
        DomainPattern("linkedin", "social-media"),
        DomainPattern("reddit", "reddit"),
        DomainPattern("claude", "claude-ai"),
        DomainPattern("supabase", "supabase"),
        DomainPattern("github", "devops"),
        DomainPattern("docker", "devops"),
        DomainPattern("gcp", "gcp"),
        DomainPattern("kubernetes", "devops"),
    ),
    specific_domain_topics=frozenset({
        "n8n", "supabase", "reddit", "gcp", "slack", "monday",
    }),
    topic_canonicalization={
        "api-integration": "backend",
        "frontend": "frontend",
        "backend": "backend",
        "database": "database",
        "devops": "devops",
        "auth": "auth",
        "security": "security",
        "testing": "backend",
        "mcp": "mcp-sdk",
        "scraping": "social-media",
        "seo": "seo",
        "n8n": "n8n",
        "supabase": "supabase",
        "pact-framework": "pact-framework",
        "claude-ai": "claude-ai",
        "slack": "slack",
        "monday": "monday-api",
        "hubspot": "backend",
        "reddit": "reddit",
        "gcp": "gcp",
        "general": "backend",
    },
    topic_keywords=(
        TopicKeyword("n8n", "n8n"),
        TopicKeyword("supabase", "supabase"),
        TopicKeyword("okta", "okta-auth"),
        TopicKeyword("saml", "okta-auth"),
        TopicKeyword("gcp", "gcp"),
        TopicKeyword("kubernetes", "gcp"),
        TopicKeyword("gke", "gcp"),
        TopicKeyword("cloudsql", "gcp"),
        TopicKeyword("docker", "devops"),
        TopicKeyword("ci/cd", "devops"),
        TopicKeyword("railway", "devops"),
        TopicKeyword("launchd", "devops"),
        TopicKeyword("apify", "apify"),
        TopicKeyword("impact.com", "impact-api"),
        TopicKeyword("impact ", "impact-api"),
        TopicKeyword("monday", "monday-api"),
        TopicKeyword("slack", "slack"),
        TopicKeyword("react", "frontend"),
        TopicKeyword("next.js", "nextjs"),
        TopicKeyword("nextjs", "nextjs"),
        TopicKeyword("next js", "nextjs"),
        TopicKeyword("tailwind", "frontend"),
        TopicKeyword("typescript", "typescript"),
        TopicKeyword("drizzle", "database"),
        TopicKeyword("postgresql", "database"),
        TopicKeyword("postgres", "database"),
        TopicKeyword("rls", "supabase"),
        TopicKeyword("migration", "database"),
        TopicKeyword("schema", "database"),
        TopicKeyword("auth", "auth"),
        TopicKeyword("jwt", "auth"),
        TopicKeyword("oauth", "auth"),
        TopicKeyword("security", "security"),
        TopicKeyword("credential", "security"),
        TopicKeyword("mcp", "mcp-sdk"),
        TopicKeyword("fastmcp", "mcp-sdk"),
        TopicKeyword("apps script", "google-apps-script"),
        TopicKeyword("google apps", "google-apps-script"),
        TopicKeyword("tiptap", "frontend"),
        TopicKeyword("bullmq", "backend"),
        TopicKeyword("redis", "backend"),
        TopicKeyword("fastify", "backend"),
        TopicKeyword("pact", "pact-framework"),
        TopicKeyword("claude", "claude-ai"),
        TopicKeyword("anthropic", "claude-ai"),
        TopicKeyword("openai", "openai"),
        TopicKeyword("linkedin", "social-media"),
        TopicKeyword("instagram", "social-media"),
        TopicKeyword("facebook", "social-media"),
        TopicKeyword("twitter", "social-media"),
        TopicKeyword("reddit", "reddit"),
        TopicKeyword("seo", "seo"),
        TopicKeyword("monorepo", "monorepo"),
        TopicKeyword("turborepo", "monorepo"),
        TopicKeyword("pnpm", "monorepo"),
        TopicKeyword("npm workspace", "monorepo"),
    ),
    topic_page_map={
        "n8n": TopicPageDescriptor("pattern", "n8n", "n8n-workflow-patterns.md", "n8n Workflow Patterns"),
        "apify": TopicPageDescriptor("entity", "n8n", "n8n-apify-integration.md", "n8n + Apify Integration"),
        "supabase": TopicPageDescriptor("pattern", "Patterns", "database-schema-patterns.md", "Database Schema Patterns"),
        "database": TopicPageDescriptor("pattern", "Patterns", "database-schema-patterns.md", "Database Schema Patterns"),
        "nextjs": TopicPageDescriptor("pattern", "Patterns", "frontend-patterns.md", "Frontend Patterns"),
        "frontend": TopicPageDescriptor("pattern", "Patterns", "frontend-patterns.md", "Frontend Patterns"),
        "backend": TopicPageDescriptor("pattern", "Patterns", "backend-patterns.md", "Backend Coding Patterns"),
        "typescript": TopicPageDescriptor("pattern", "Patterns", "typescript-patterns.md", "TypeScript Patterns"),
        "google-apps-script": TopicPageDescriptor("pattern", "Patterns", "google-apps-script-patterns.md", "Google Apps Script Patterns"),
        "mcp-sdk": TopicPageDescriptor("pattern", "Patterns", "mcp-sdk-patterns.md", "MCP SDK Patterns"),
        "auth": TopicPageDescriptor("pattern", "Patterns", "auth-patterns.md", "Authentication Patterns"),
        "security": TopicPageDescriptor("pattern", "Patterns", "security-patterns.md", "Security Patterns"),
        "seo": TopicPageDescriptor("pattern", "Patterns", "seo-patterns.md", "SEO Patterns"),
        "pact-framework": TopicPageDescriptor("pattern", "Patterns", "pact-framework-patterns.md", "PACT Framework Patterns"),
        "devops": TopicPageDescriptor("pattern", "Patterns", "monorepo-and-devops-patterns.md", "Monorepo & DevOps Patterns"),
        "monorepo": TopicPageDescriptor("pattern", "Patterns", "monorepo-and-devops-patterns.md", "Monorepo & DevOps Patterns"),
        "gcp": TopicPageDescriptor("pattern", "Patterns", "monorepo-and-devops-patterns.md", "Monorepo & DevOps Patterns"),
        "impact-api": TopicPageDescriptor("entity", "APIs", "impact-com-api-reference.md", "Impact.com API Reference"),
        "monday-api": TopicPageDescriptor("entity", "APIs", "third-party-api-misc.md", "Third-Party API Miscellaneous"),
        "slack": TopicPageDescriptor("entity", "APIs", "third-party-api-misc.md", "Third-Party API Miscellaneous"),
        "claude-ai": TopicPageDescriptor("entity", "APIs", "third-party-api-misc.md", "Third-Party API Miscellaneous"),
        "openai": TopicPageDescriptor("entity", "APIs", "third-party-api-misc.md", "Third-Party API Miscellaneous"),
        "social-media": TopicPageDescriptor("entity", "APIs", "apify-platform-reference.md", "Apify Platform Reference"),
        "okta-auth": TopicPageDescriptor("architecture", "Architecture", "arch-okta-auth-pattern.md", "Okta Auth Pattern"),
        "reddit": TopicPageDescriptor("entity", "", "reddit-shadow-ban-research.md", "Reddit Shadow Ban Research"),
    },
    wikilink_targets={
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
    },
    path_skip_prefixes=frozenset({
        "users", "documents", "personal", "projects", "github", "repos",
    }),
)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_ALLOWED_PAGE_TYPES = frozenset({"architecture", "patterns", "pattern", "entity", "gotcha"})


def _coerce_user_config(raw: object) -> DomainConfig:
    """Build a DomainConfig from a loaded YAML dict.

    Tolerant of missing sections (each defaults to empty). Bad rows inside a
    section are dropped with a WARNING; the rest of the section still loads.
    """
    if raw is None:
        return DomainConfig()
    if not isinstance(raw, dict):
        raise ConfigError(f"Top-level YAML must be a mapping, got {type(raw).__name__}")

    known_keys = {
        "project_domain_patterns", "specific_domain_topics",
        "topic_canonicalization", "topic_keywords",
        "topic_page_map", "wikilink_targets", "path_skip_prefixes",
    }
    for key in raw.keys():
        if key not in known_keys:
            log.warning("Unknown domains.yaml key %r; ignoring.", key)

    patterns: list[DomainPattern] = []
    for row in raw.get("project_domain_patterns") or []:
        if isinstance(row, dict) and "keyword" in row and "domain" in row:
            patterns.append(DomainPattern(str(row["keyword"]), str(row["domain"])))
        else:
            log.warning("Skipping malformed project_domain_patterns row: %r", row)

    specific = raw.get("specific_domain_topics") or []
    if not isinstance(specific, list):
        log.warning("specific_domain_topics must be a list; got %r", type(specific).__name__)
        specific = []

    canon = raw.get("topic_canonicalization") or {}
    if not isinstance(canon, dict):
        log.warning("topic_canonicalization must be a mapping; got %r", type(canon).__name__)
        canon = {}

    keywords: list[TopicKeyword] = []
    for row in raw.get("topic_keywords") or []:
        if isinstance(row, dict) and "keyword" in row and "topic" in row:
            keywords.append(TopicKeyword(str(row["keyword"]), str(row["topic"])))
        else:
            log.warning("Skipping malformed topic_keywords row: %r", row)

    page_map: dict[str, TopicPageDescriptor] = {}
    raw_page_map = raw.get("topic_page_map") or {}
    if not isinstance(raw_page_map, dict):
        log.warning("topic_page_map must be a mapping; got %r", type(raw_page_map).__name__)
        raw_page_map = {}
    for topic, descriptor in raw_page_map.items():
        if not isinstance(descriptor, dict):
            log.warning("Skipping malformed topic_page_map row for %r: %r", topic, descriptor)
            continue
        required = {"page_type", "subfolder", "filename", "title"}
        missing = required - descriptor.keys()
        if missing:
            log.warning("topic_page_map[%r] missing fields %s; skipping.", topic, sorted(missing))
            continue
        page_type = str(descriptor["page_type"])
        if page_type not in _ALLOWED_PAGE_TYPES:
            log.warning("topic_page_map[%r] has unknown page_type=%r; skipping.", topic, page_type)
            continue
        subfolder = str(descriptor["subfolder"])
        if "/" in subfolder:
            log.warning("topic_page_map[%r] subfolder must be a single path component; skipping.", topic)
            continue
        filename = str(descriptor["filename"])
        if not filename.endswith(".md"):
            log.warning("topic_page_map[%r] filename must end with .md; skipping.", topic)
            continue
        title = str(descriptor["title"])
        if not title.strip():
            log.warning("topic_page_map[%r] title must be non-empty; skipping.", topic)
            continue
        page_map[str(topic)] = TopicPageDescriptor(page_type, subfolder, filename, title)

    wiki = raw.get("wikilink_targets") or {}
    if not isinstance(wiki, dict):
        log.warning("wikilink_targets must be a mapping; got %r", type(wiki).__name__)
        wiki = {}

    skip = raw.get("path_skip_prefixes") or []
    if not isinstance(skip, list):
        log.warning("path_skip_prefixes must be a list; got %r", type(skip).__name__)
        skip = []

    return DomainConfig(
        project_domain_patterns=tuple(patterns),
        specific_domain_topics=frozenset(str(s).lower() for s in specific),
        topic_canonicalization={str(k): str(v) for k, v in canon.items()},
        topic_keywords=tuple(keywords),
        topic_page_map=page_map,
        wikilink_targets={str(k): str(v) for k, v in wiki.items()},
        path_skip_prefixes=frozenset(str(s).lower() for s in skip),
    )


def _merge(user: DomainConfig, builtin: DomainConfig) -> DomainConfig:
    """Merge user over built-in. User wins on key collisions; user entries come FIRST in tuples."""
    return DomainConfig(
        project_domain_patterns=tuple(user.project_domain_patterns) + tuple(builtin.project_domain_patterns),
        specific_domain_topics=user.specific_domain_topics | builtin.specific_domain_topics,
        topic_canonicalization={**builtin.topic_canonicalization, **user.topic_canonicalization},
        topic_keywords=tuple(user.topic_keywords) + tuple(builtin.topic_keywords),
        topic_page_map={**builtin.topic_page_map, **user.topic_page_map},
        wikilink_targets={**builtin.wikilink_targets, **user.wikilink_targets},
        path_skip_prefixes=user.path_skip_prefixes | builtin.path_skip_prefixes,
    )


def _default_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "domains.yaml"


def load_domain_config(path: Optional[Path] = None) -> DomainConfig:
    """Load user config from YAML; merge over built-in defaults; return DomainConfig.

    Resolution order:
      1. Explicit `path` argument (if provided).
      2. `KB_BRAIN_DOMAINS_FILE` env var.
      3. Package-relative default `<repo>/config/domains.yaml`.

    Missing-file behavior:
      - If the env var was set and points at a missing file -> raise ConfigError.
      - Otherwise (argument or default path missing) -> log INFO and return
        BUILTIN_DOMAIN_CONFIG. This keeps tests that pass a tempdir path clean.
    """
    env_path_str = os.environ.get("KB_BRAIN_DOMAINS_FILE")
    env_override = path is None and env_path_str is not None

    if path is None:
        if env_path_str:
            path = Path(env_path_str).expanduser()
        else:
            path = _default_path()

    if not path.exists():
        if env_override:
            raise ConfigError(f"Domains file not found: {path}")
        log.info("No config/domains.yaml found; using built-in defaults only.")
        return BUILTIN_DOMAIN_CONFIG

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse {path}: {e}") from e
    except OSError as e:
        raise ConfigError(f"Failed to read {path}: {e}") from e

    user_cfg = _coerce_user_config(raw)
    return _merge(user_cfg, BUILTIN_DOMAIN_CONFIG)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_domain_config: Optional[DomainConfig] = None


def set_domain_config(cfg: Optional[DomainConfig]) -> None:
    """Install the active DomainConfig. Called from pipeline.main() and tests.

    Passing None resets the singleton so the next get_domain_config() auto-loads.
    """
    global _domain_config
    _domain_config = cfg


def get_domain_config() -> DomainConfig:
    """Return the active DomainConfig, auto-loading from the default path if unset.

    Safety net for modules used outside of pipeline.main() (tests that import
    a single module directly): never silently uses stale state.
    """
    global _domain_config
    if _domain_config is None:
        _domain_config = load_domain_config()
    return _domain_config
