"""
src/classifier.py -- Heuristic session classifier (no LLM calls).

Scores filtered sessions on a 1-5 value scale using project path
analysis, message statistics, and keyword detection.  Sessions scoring
>= 3 are passed to the session extractor for text extraction.

Used by pipeline.py to decide which sessions to process in Phase 2.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .domains import get_domain_config
from .prefilter import FilteredSession

logger = logging.getLogger(__name__)


@dataclass
class SessionClassification:
    """Classification result for a single session."""

    session_id: str
    project_dir: str
    value_score: int             # 1-5, higher = more valuable
    topics: list[str] = field(default_factory=list)
    session_type: str = "main"   # "main" or "subagent"
    domain: str = ""             # inferred domain (e.g., "frontend", "backend")
    engagement_level: str = ""   # "low", "medium", "high", "very-high"
    message_count: int = 0
    text_chars: int = 0
    has_decisions: bool = False
    has_architecture: bool = False


# ---------------------------------------------------------------------------
# Project path -> domain mapping (built-in generic only; user adds via config)
# ---------------------------------------------------------------------------

_BUILTIN_DOMAIN_PATTERNS: list[tuple[str, str]] = [
    ("n8n", "n8n"),
    ("slack", "slack"),
    ("hubspot", "hubspot"),
    ("monday", "monday"),
    ("linkedin", "social-media"),
    ("reddit", "reddit"),
    ("claude", "claude-ai"),
    ("supabase", "supabase"),
    ("github", "devops"),
    ("docker", "devops"),
    ("gcp", "gcp"),
    ("kubernetes", "devops"),
]


def _infer_domain(project_dir: str) -> str:
    """Infer a domain from the project directory name.

    User-config entries (from domains.yaml) are checked first; built-in
    generic entries are the fallback. Returns "general" if no match.
    """
    cfg = get_domain_config()
    haystack = project_dir.lower().replace("-", " ")
    for entry in cfg.project_domain_patterns:
        if entry.keyword.lower() in haystack:
            return entry.domain
    for keyword, domain in _BUILTIN_DOMAIN_PATTERNS:
        if keyword in haystack:
            return domain
    return "general"


# ---------------------------------------------------------------------------
# Topic detection keywords
# ---------------------------------------------------------------------------

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "n8n": ["n8n", "workflow", "webhook", "node configuration"],
    "supabase": ["supabase", "rls", "edge function", "postgres"],
    "api-integration": ["api", "endpoint", "rest", "graphql", "webhook"],
    "frontend": ["react", "next.js", "nextjs", "tailwind", "component", "css"],
    "backend": ["express", "fastify", "middleware", "server", "route"],
    "database": ["schema", "migration", "query", "index", "sql", "table"],
    "devops": ["docker", "ci/cd", "pipeline", "deploy", "kubernetes", "gcp"],
    "auth": ["authentication", "authorization", "oauth", "jwt", "token", "okta"],
    "security": ["vulnerability", "injection", "xss", "csrf", "sanitize"],
    "testing": ["test", "jest", "pytest", "coverage", "mock"],
    "mcp": ["mcp", "model context protocol", "tool_use"],
    "scraping": ["scraper", "apify", "crawl", "selenium", "playwright"],
    "seo": ["seo", "keyword", "serp", "ranking", "backlink"],
}

# Keywords that indicate high-value decision/architecture content
_DECISION_KEYWORDS = [
    "decided", "decision", "chose", "because", "trade-off", "tradeoff",
    "instead of", "rather than", "approach", "strategy", "rationale",
    "why we", "opted for", "considered", "alternative",
]

_ARCHITECTURE_KEYWORDS = [
    "architecture", "design", "pattern", "schema", "interface",
    "component", "module", "service", "layer", "separation",
    "dependency", "coupling", "cohesion", "abstraction",
    "data flow", "pipeline", "middleware", "orchestrat",
]

_GOTCHA_KEYWORDS = [
    "gotcha", "quirk", "workaround", "bug", "issue", "fix",
    "doesn't work", "broke", "broken", "unexpected", "careful",
    "watch out", "trap", "pitfall", "caveat", "surprising",
]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_session(session: FilteredSession) -> SessionClassification:
    """Classify a filtered session by value, topics, and domain.

    Scoring rubric:
      - Base score: 1 (every session gets at least 1)
      - Engagement bonus: +1 for medium, +2 for high/very-high
      - Decision/architecture keywords: +1 each (max +2)
      - Gotcha keywords: +0.5 (counted toward the +2 cap above)
      - Cap at 5
    """
    domain = _infer_domain(session.project_dir)
    topics = _detect_topics(session)
    engagement = _assess_engagement(session)
    msg_count = len(session.messages)

    # Sample text for keyword analysis (first 200K chars)
    sample_text = _sample_text(session, max_chars=200_000).lower()

    has_decisions = any(kw in sample_text for kw in _DECISION_KEYWORDS)
    has_architecture = any(kw in sample_text for kw in _ARCHITECTURE_KEYWORDS)
    has_gotchas = any(kw in sample_text for kw in _GOTCHA_KEYWORDS)

    # Calculate score
    score = 1

    if engagement in ("medium",):
        score += 1
    elif engagement in ("high", "very-high"):
        score += 2

    bonus = 0
    if has_decisions:
        bonus += 1
    if has_architecture:
        bonus += 1
    if has_gotchas and bonus < 2:
        bonus += 1
    score += min(bonus, 2)

    score = min(score, 5)

    return SessionClassification(
        session_id=session.session_id,
        project_dir=session.project_dir,
        value_score=score,
        topics=topics,
        session_type="main",
        domain=domain,
        engagement_level=engagement,
        message_count=msg_count,
        text_chars=session.total_text_chars,
        has_decisions=has_decisions,
        has_architecture=has_architecture,
    )


def _detect_topics(session: FilteredSession) -> list[str]:
    """Detect topics present in a session based on keyword matching."""
    sample = _sample_text(session, max_chars=100_000).lower()
    found: list[str] = []

    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(kw in sample for kw in keywords):
            found.append(topic)

    return sorted(found)


def _assess_engagement(session: FilteredSession) -> str:
    """Assess engagement level based on message count and text volume."""
    msg_count = len(session.messages)
    text_kb = session.total_text_chars / 1024

    if msg_count >= 40 or text_kb >= 100:
        return "very-high"
    if msg_count >= 20 or text_kb >= 50:
        return "high"
    if msg_count >= 10 or text_kb >= 20:
        return "medium"
    return "low"


def _sample_text(session: FilteredSession, max_chars: int) -> str:
    """Concatenate message text up to max_chars for analysis."""
    parts: list[str] = []
    total = 0

    for msg in session.messages:
        if total >= max_chars:
            break
        remaining = max_chars - total
        text = msg.text[:remaining]
        parts.append(text)
        total += len(text)

    return "\n".join(parts)
