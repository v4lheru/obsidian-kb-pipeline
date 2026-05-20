"""
src/scrubber.py — Credential and sensitive data scrubbing.

Safety net for memory sources.  Memories should not contain credentials,
but scrubbing runs anyway to catch leaks before wiki pages are written.
Used by memory_reader.py on every text field.
"""

from __future__ import annotations

import re

# Patterns that match common credential formats
_CREDENTIAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # AWS access keys
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_KEY]"),
    # Anthropic API keys
    (re.compile(r"sk-ant-[a-zA-Z0-9_-]{20,}"), "[REDACTED_ANTHROPIC_KEY]"),
    # OpenAI API keys
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "[REDACTED_OPENAI_KEY]"),
    # GitHub tokens
    (re.compile(r"ghp_[a-zA-Z0-9]{36,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"gho_[a-zA-Z0-9]{36,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"github_pat_[a-zA-Z0-9_]{20,}"), "[REDACTED_GITHUB_TOKEN]"),
    # Generic Bearer tokens
    (re.compile(r"Bearer\s+[a-zA-Z0-9._-]{20,}"), "Bearer [REDACTED_TOKEN]"),
    # Supabase keys (long JWT-like strings after known prefixes)
    (re.compile(r"(eyJ[a-zA-Z0-9_-]{50,}\.[a-zA-Z0-9_-]{50,}\.[a-zA-Z0-9_-]{20,})"),
     "[REDACTED_JWT]"),
    # Generic password assignments
    (re.compile(r'(?i)(password|passwd|secret|token)\s*[=:]\s*["\']?[^\s"\']{8,}'),
     r"\1=[REDACTED]"),
    # .env file patterns
    (re.compile(r'(?m)^[A-Z_]+(KEY|SECRET|TOKEN|PASSWORD|PASS)\s*=\s*.+$'),
     "[REDACTED_ENV_VAR]"),
]


def scrub_text(text: str) -> str:
    """Remove credential patterns from text.

    Returns the scrubbed text.  This is a safety net — memory sources
    should already be credential-free, but defense in depth.
    """
    result = text
    for pattern, replacement in _CREDENTIAL_PATTERNS:
        result = pattern.sub(replacement, result)
    return result
