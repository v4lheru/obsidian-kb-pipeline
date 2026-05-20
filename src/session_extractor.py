"""
src/session_extractor.py -- Extracts distilled text from high-value sessions.

For sessions scoring >= 3 in the classifier, extracts user prompts and
assistant text responses (no tool outputs), chunked at conversation
boundaries.  Each chunk becomes an Extraction that feeds into the
existing clusterer -> synthesizer -> wiki_writer chain.

Used by pipeline.py as the fourth stage of Phase 2 JSONL processing.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field

from .classifier import SessionClassification
from .domains import get_domain_config
from .prefilter import FilteredMessage, FilteredSession
from .state_store import Extraction

logger = logging.getLogger(__name__)

# Maximum characters per text chunk
MAX_CHUNK_CHARS = 50_000

# Minimum characters for a chunk to be worth extracting
MIN_CHUNK_CHARS = 200


@dataclass
class TextChunk:
    """A chunk of session text, bounded by conversation turns."""

    text: str
    message_count: int
    start_index: int    # index of first message in this chunk
    end_index: int      # index of last message in this chunk


def extract_session(
    session: FilteredSession,
    classification: SessionClassification,
) -> list[Extraction]:
    """Extract knowledge from a high-value session as Extraction objects.

    1. Concatenates user prompts + assistant text (no tool results)
    2. Chunks at conversation boundaries (user prompt = new chunk boundary)
    3. Converts each chunk to an Extraction with session metadata

    Returns a list of Extraction objects ready for the clusterer.
    """
    chunks = _chunk_session(session)

    if not chunks:
        logger.debug("No extractable chunks from session %s", session.session_id)
        return []

    extractions: list[Extraction] = []

    for i, chunk in enumerate(chunks):
        # Generate a stable ID from session + chunk index
        chunk_id = _stable_id(session.session_id, i)

        # Determine extraction type based on content
        extraction_type = _infer_extraction_type(chunk.text)

        # Prefer domain when it has a specific clusterer mapping (e.g., "frontend"
        # maps to a frontend architecture page).  Fall back to first detected topic.
        topic = _select_topic(classification)

        # Build a title from the first user message in the chunk
        title = _extract_title(chunk.text, session.session_id, i)

        # Build context string
        context = _build_context(session, classification)

        extractions.append(Extraction(
            id=chunk_id,
            source_id=session.session_id,
            source_type="session",
            extraction_type=extraction_type,
            topic=topic,
            title=title,
            content=chunk.text,
            confidence=_score_to_confidence(classification.value_score),
            context=context,
        ))

    logger.debug("Extracted %d chunks from session %s (score=%d)",
                 len(extractions), session.session_id, classification.value_score)

    return extractions


def _chunk_session(session: FilteredSession) -> list[TextChunk]:
    """Split session messages into text chunks at conversation boundaries.

    A new chunk starts at each user message.  Chunks are capped at
    MAX_CHUNK_CHARS.  Chunks smaller than MIN_CHUNK_CHARS are dropped.
    """
    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_chars = 0
    current_msg_count = 0
    chunk_start = 0

    for i, msg in enumerate(session.messages):
        # User message starts a potential new chunk boundary
        if msg.role == "user" and current_chars > 0:
            # If current chunk is big enough, finalize it
            if current_chars >= MIN_CHUNK_CHARS:
                text = "\n\n".join(current_parts)
                chunks.append(TextChunk(
                    text=text,
                    message_count=current_msg_count,
                    start_index=chunk_start,
                    end_index=i - 1,
                ))
                current_parts = []
                current_chars = 0
                current_msg_count = 0
                chunk_start = i

        # Format the message
        prefix = "User:" if msg.role == "user" else "Assistant:"
        formatted = f"{prefix} {msg.text}"

        # Check if adding this message would exceed chunk size
        if current_chars + len(formatted) > MAX_CHUNK_CHARS and current_chars > 0:
            # Finalize current chunk
            if current_chars >= MIN_CHUNK_CHARS:
                text = "\n\n".join(current_parts)
                chunks.append(TextChunk(
                    text=text,
                    message_count=current_msg_count,
                    start_index=chunk_start,
                    end_index=i - 1,
                ))
            current_parts = []
            current_chars = 0
            current_msg_count = 0
            chunk_start = i

        current_parts.append(formatted)
        current_chars += len(formatted)
        current_msg_count += 1

    # Finalize the last chunk
    if current_parts and current_chars >= MIN_CHUNK_CHARS:
        text = "\n\n".join(current_parts)
        chunks.append(TextChunk(
            text=text,
            message_count=current_msg_count,
            start_index=chunk_start,
            end_index=len(session.messages) - 1,
        ))

    return chunks


def _stable_id(session_id: str, chunk_index: int) -> str:
    """Generate a stable, deterministic ID for a session chunk.

    Using a hash ensures the same session + chunk always produces the
    same ID, supporting idempotent re-runs.
    """
    raw = f"{session_id}:chunk:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _infer_extraction_type(text: str) -> str:
    """Infer the extraction type from chunk content."""
    lower = text.lower()

    # Check for decision-related content
    decision_signals = ["decided", "decision", "chose", "because we", "trade-off",
                        "instead of", "opted for", "rationale"]
    if sum(1 for s in decision_signals if s in lower) >= 2:
        return "decision"

    # Check for architecture content
    arch_signals = ["architecture", "design pattern", "component", "schema",
                    "interface", "data flow", "pipeline design"]
    if sum(1 for s in arch_signals if s in lower) >= 2:
        return "reasoning"

    # Check for gotcha/debugging content
    gotcha_signals = ["gotcha", "workaround", "bug", "doesn't work", "broke",
                      "unexpected", "careful", "watch out", "pitfall"]
    if sum(1 for s in gotcha_signals if s in lower) >= 2:
        return "gotcha"

    # Check for process/workflow content
    process_signals = ["step 1", "step 2", "first,", "then,", "next,",
                       "finally,", "workflow", "process"]
    if sum(1 for s in process_signals if s in lower) >= 2:
        return "process_step"

    # Default: pattern (most session content is pattern-like)
    return "pattern"


# Built-in: domains that have specific (non-generic) clusterer page mappings.
# User can extend via cfg.specific_domain_topics or by adding topic_page_map entries.
_BUILTIN_SPECIFIC_DOMAIN_TOPICS = frozenset({
    "n8n", "supabase", "reddit", "gcp", "slack", "monday",
})


def _specific_domain_topics() -> frozenset[str]:
    """Combined set: user topic_page_map keys + user specific_domain_topics + built-in."""
    cfg = get_domain_config()
    return (
        frozenset(cfg.topic_page_map.keys())
        | cfg.specific_domain_topics
        | _BUILTIN_SPECIFIC_DOMAIN_TOPICS
    )


def _select_topic(classification: SessionClassification) -> str:
    """Choose the best canonical topic for routing to the clusterer.

    Prefers the domain when it has a specific page mapping. Falls back to
    the first detected topic.
    """
    domain = classification.domain

    # If domain has a specific page in the clusterer, use it
    if domain in _specific_domain_topics():
        canonical = _canonicalize_topic(domain)
        if canonical:
            return canonical

    # Otherwise use the first detected topic
    if classification.topics:
        for topic in classification.topics:
            canonical = _canonicalize_topic(topic)
            if canonical:
                return canonical

    # Final fallback
    return _canonicalize_topic(domain) or "backend"


# Built-in topic canonicalization (generic only). User entries merge via config.
_BUILTIN_TOPIC_CANONICALIZATION: dict[str, str] = {
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
}


def _canonicalize_topic(topic: str) -> str:
    """Map a topic to canonical clusterer topic.

    User config entries (from domains.yaml) win on key collision; otherwise
    fall back to the built-in generic map. Returns empty string if no mapping.
    """
    cfg = get_domain_config()
    if topic in cfg.topic_canonicalization:
        return cfg.topic_canonicalization[topic]
    return _BUILTIN_TOPIC_CANONICALIZATION.get(topic, "")


def _extract_title(chunk_text: str, session_id: str, chunk_index: int) -> str:
    """Extract a meaningful title from the first user message in a chunk."""
    # Find the first "User:" line
    for line in chunk_text.split("\n"):
        if line.startswith("User:"):
            prompt = line[5:].strip()
            # Truncate to a reasonable title length
            if len(prompt) > 100:
                prompt = prompt[:97] + "..."
            if prompt:
                return prompt
            break

    return f"Session {session_id[:8]} chunk {chunk_index + 1}"


def _build_context(
    session: FilteredSession,
    classification: SessionClassification,
) -> str:
    """Build a context string for the extraction."""
    parts = []

    # Decode project path to a human-readable name
    project_name = _project_dir_to_name(session.project_dir)
    if project_name:
        parts.append(project_name)

    if classification.domain and classification.domain != "general":
        parts.append(classification.domain)

    return " / ".join(parts) if parts else "Claude Code session"


def _project_dir_to_name(project_dir: str) -> str:
    """Convert a project directory name to a human-readable name.

    Input:  "-Users-alice-Documents-projects-my-app"
    Output: "projects / my-app"

    The macOS short username + any per-machine folder names are taken from
    cfg.path_skip_prefixes so each user can adapt without code changes.
    """
    # Strip leading dash and split by dash
    parts = project_dir.lstrip("-").split("-")

    # Drop common path prefixes (built-in generic + user-config)
    skip_prefixes = {"users", "documents"} | get_domain_config().path_skip_prefixes
    filtered = []
    skipping = True
    for part in parts:
        if skipping and part.lower() in skip_prefixes:
            continue
        skipping = False
        filtered.append(part)

    if not filtered:
        return project_dir

    # Join with spaces, capitalize each part
    return " ".join(filtered)


def _score_to_confidence(value_score: int) -> float:
    """Map value score (1-5) to confidence (0.0-1.0)."""
    return {1: 0.3, 2: 0.4, 3: 0.6, 4: 0.8, 5: 0.9}.get(value_score, 0.5)
