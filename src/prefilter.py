"""
src/prefilter.py -- Pre-filters JSONL session messages to strip noise.

Takes the raw stream from jsonl_parser and removes non-knowledge-bearing
content: system records, tool results, thinking blocks, image blocks,
and credentials.  Also drops sessions with fewer than 5 messages after
filtering.

Used by classifier.py and session_extractor.py as the second stage
of Phase 2 JSONL processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .jsonl_parser import ContentBlock, SessionFile, SessionMessage, parse_session
from .scrubber import scrub_text

logger = logging.getLogger(__name__)

# Record types to drop entirely
_DROP_RECORD_TYPES = frozenset({
    "permission-mode",
    "file-history-snapshot",
    "last-prompt",
    "system",
    "attachment",
})

# Content block types to strip from messages
_STRIP_BLOCK_TYPES = frozenset({
    "tool_result",
    "thinking",
    "image",
})

# Minimum messages after filtering to keep a session
MIN_MESSAGES = 5


@dataclass
class FilteredMessage:
    """A message after noise stripping, ready for classification."""

    role: str                           # "user" or "assistant"
    text: str                           # concatenated text content (scrubbed)
    tool_names: list[str] = field(default_factory=list)  # tools used (context only)
    uuid: str = ""
    timestamp: str = ""
    is_sidechain: bool = False


@dataclass
class FilteredSession:
    """A session after filtering, with metadata for classification."""

    session_id: str
    project_dir: str
    file_path: str
    file_size: int
    messages: list[FilteredMessage] = field(default_factory=list)
    total_text_chars: int = 0
    user_message_count: int = 0
    assistant_message_count: int = 0


def filter_session(session: SessionFile) -> FilteredSession | None:
    """Filter a single session, stripping noise and scrubbing credentials.

    Returns None if the session has fewer than MIN_MESSAGES after filtering.
    """
    filtered_messages: list[FilteredMessage] = []
    total_text_chars = 0
    user_count = 0
    assistant_count = 0

    for msg in parse_session(session):
        # Drop non-message record types
        if msg.record_type in _DROP_RECORD_TYPES:
            continue

        # Only process user and assistant messages
        if msg.record_type not in ("user", "assistant"):
            continue

        # Skip sidechain messages (alternate conversation branches)
        if msg.is_sidechain:
            continue

        filtered = _filter_message(msg)
        if filtered is None:
            continue

        filtered_messages.append(filtered)
        total_text_chars += len(filtered.text)

        if filtered.role == "user":
            user_count += 1
        else:
            assistant_count += 1

    # Drop sessions with too few messages
    if len(filtered_messages) < MIN_MESSAGES:
        logger.debug("Dropping session %s — only %d messages after filtering",
                     session.session_id, len(filtered_messages))
        return None

    return FilteredSession(
        session_id=session.session_id,
        project_dir=session.project_dir,
        file_path=str(session.file_path),
        file_size=session.file_size,
        messages=filtered_messages,
        total_text_chars=total_text_chars,
        user_message_count=user_count,
        assistant_message_count=assistant_count,
    )


def _filter_message(msg: SessionMessage) -> FilteredMessage | None:
    """Filter a single message: strip noisy blocks, scrub credentials.

    Returns None if the message has no text content after filtering.
    """
    text_parts: list[str] = []
    tool_names: list[str] = []

    for block in msg.content_blocks:
        if block.block_type in _STRIP_BLOCK_TYPES:
            continue

        if block.block_type == "text" and block.text.strip():
            text_parts.append(block.text)
        elif block.block_type == "tool_use" and block.tool_name:
            tool_names.append(block.tool_name)

    if not text_parts:
        return None

    # Concatenate and scrub
    combined_text = "\n".join(text_parts)
    scrubbed_text = scrub_text(combined_text)

    return FilteredMessage(
        role=msg.role,
        text=scrubbed_text,
        tool_names=tool_names,
        uuid=msg.uuid,
        timestamp=msg.timestamp,
        is_sidechain=msg.is_sidechain,
    )
