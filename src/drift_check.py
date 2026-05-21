"""
src/drift_check.py -- LLM-backed drift detection for page merges.

When the synthesizer is about to merge new extractions into an existing wiki
page, this module asks Claude whether any new extraction CONTRADICTS the
existing page content. Contradictions are not auto-resolved; they are returned
as `DriftFinding` objects so the synthesizer can annotate the merged body
with `<!-- DRIFT: ... -->` HTML comments for human review.

The drift check is purely additive enhancement. EVERY failure mode -- disabled
flag, missing `ANTHROPIC_API_KEY`, missing `anthropic` package, API timeout
or 5xx, unparseable response -- returns `DriftResult(findings=[], skipped=True,
skip_reason=...)`. This function NEVER raises; the pipeline must remain green
when the LLM is unavailable.

Used by `synthesizer._synthesize_update` after dedup and before addendum build.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from .state_store import Extraction

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 1024
_EXISTING_BODY_TRUNCATE = 8000
_EXTRACTION_TRUNCATE = 1000
_SNIPPET_MAX = 200
_MIN_BODY_CHARS = 200

DRIFT_SYSTEM_PROMPT = """You audit a coding-knowledge wiki for internal contradictions. The user will give you:
1. The existing content of one wiki page (as markdown).
2. A list of new findings about to be merged into that page.

Your job: find ONLY direct contradictions where a new finding asserts something incompatible with the existing page. Not minor differences, not additions, not elaborations. ONLY contradictions where both statements cannot be true at the same time about the same subject.

Output a single JSON object with one key "findings" whose value is an array. Each element is an object with three string fields:
- "existing_snippet": a verbatim quote from the existing page (max 200 chars) that contains the claim being contradicted.
- "new_snippet": a verbatim quote from the new finding (max 200 chars) that contains the contradicting claim.
- "note": a one-sentence description of the contradiction in plain English (max 200 chars).

If there are no contradictions, output {"findings": []}.

Do not output anything other than the JSON object. No prose, no markdown fences, no commentary. The JSON must parse with json.loads."""


@dataclass(frozen=True)
class DriftFinding:
    """A single contradiction between existing page content and a new extraction."""
    existing_snippet: str
    new_snippet: str
    note: str


@dataclass(frozen=True)
class DriftResult:
    """Outcome of a drift check. Empty findings = no contradictions OR check skipped."""
    findings: list[DriftFinding] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


def check_drift(
    existing_body: str,
    new_extractions: list[Extraction],
    enabled: bool = True,
    timeout_seconds: float | None = None,
) -> DriftResult:
    """Ask Claude whether any new_extractions contradict existing_body.

    Returns DriftResult with findings=[] and skipped=True on every failure
    mode -- this function never raises. Drift is enhancement, not gate.
    """
    if not enabled:
        return DriftResult(skipped=True, skip_reason="disabled")

    if not new_extractions or len(existing_body) < _MIN_BODY_CHARS:
        return DriftResult(skipped=True, skip_reason="trivial_input")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.info("Drift check skipped: ANTHROPIC_API_KEY not set.")
        return DriftResult(skipped=True, skip_reason="no_api_key")

    try:
        import anthropic
    except ImportError:
        logger.info(
            "Drift check skipped: anthropic package not installed. "
            "Install with `pip install obsidian-kb-pipeline[llm]`."
        )
        return DriftResult(skipped=True, skip_reason="no_anthropic_package")

    if timeout_seconds is None:
        try:
            timeout_seconds = float(os.environ.get("KB_BRAIN_DRIFT_TIMEOUT", "30"))
        except ValueError:
            timeout_seconds = 30.0

    user_message = _build_user_message(existing_body, new_extractions)

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            temperature=0,
            timeout=timeout_seconds,
            system=DRIFT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except Exception as exc:  # noqa: BLE001 -- SDK raises many distinct types
        logger.warning("Drift check API call failed: %s", exc)
        return DriftResult(skipped=True, skip_reason="api_error")

    raw_text = _extract_response_text(response)
    findings = _parse_findings(raw_text)
    if findings is None:
        logger.warning("Drift check response did not parse as expected JSON: %r", raw_text[:200])
        return DriftResult(skipped=True, skip_reason="parse_error")

    return DriftResult(findings=findings, skipped=False)


def _build_user_message(existing_body: str, new_extractions: list[Extraction]) -> str:
    """Build the per-call user-message body from existing page + new findings."""
    truncated_body = existing_body[-_EXISTING_BODY_TRUNCATE:] if len(existing_body) > _EXISTING_BODY_TRUNCATE else existing_body

    findings_lines = []
    for i, ext in enumerate(new_extractions, start=1):
        content = ext.content[:_EXTRACTION_TRUNCATE]
        findings_lines.append(f"{i}. {ext.title}: {content}")
    findings_formatted = "\n".join(findings_lines)

    return (
        "Existing page content:\n---\n"
        f"{truncated_body}\n---\n\n"
        "New findings (each on its own line, prefixed with index):\n"
        f"{findings_formatted}"
    )


def _extract_response_text(response: Any) -> str:
    """Pull text out of an anthropic Messages response. Tolerant of mock shapes."""
    try:
        content = response.content
        if not content:
            return ""
        block = content[0]
        return getattr(block, "text", "") or ""
    except (AttributeError, IndexError, TypeError):
        return ""


def _parse_findings(raw_text: str) -> list[DriftFinding] | None:
    """Parse the JSON-only model response into DriftFinding list.

    Returns None on any structural failure (not-JSON, top-level not dict,
    `findings` not list). Per-item failures skip that item but keep the rest.
    """
    if not raw_text or not raw_text.strip():
        return None
    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    items = data.get("findings")
    if not isinstance(items, list):
        return None

    findings: list[DriftFinding] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        existing = item.get("existing_snippet")
        new = item.get("new_snippet")
        note = item.get("note")
        if not (isinstance(existing, str) and isinstance(new, str) and isinstance(note, str)):
            continue
        findings.append(DriftFinding(
            existing_snippet=_sanitize_snippet(existing),
            new_snippet=_sanitize_snippet(new),
            note=_sanitize_snippet(note),
        ))
    return findings


def _sanitize_snippet(text: str) -> str:
    """Clamp to 200 chars and strip any `-->` substring so the snippet cannot
    break out of the wrapping `<!-- ... -->` HTML comment.
    """
    cleaned = text.replace("-->", "--")
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > _SNIPPET_MAX:
        cleaned = cleaned[:_SNIPPET_MAX]
    return cleaned


def build_drift_comment(findings: list[DriftFinding]) -> str:
    """Render a list of findings into a single multi-line HTML comment block,
    one line per finding. Returns empty string when findings is empty.
    """
    if not findings:
        return ""
    lines = ["<!-- DRIFT detected (auto-generated; review and resolve manually):"]
    for f in findings:
        lines.append(
            f'     {f.note} | existing: "{f.existing_snippet}" | new: "{f.new_snippet}"'
        )
    lines.append("-->")
    return "\n".join(lines)
