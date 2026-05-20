"""
src/jsonl_parser.py -- Streaming parser for Claude Code session JSONL files.

Reads JSONL session files from ~/.claude/projects/ line-by-line (never
loading full files into memory -- some are 147MB).  Extracts user and
assistant message records with their text content blocks.

Used by prefilter.py as the first stage of Phase 2 JSONL processing.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ContentBlock:
    """A single content block from a message."""

    block_type: str          # "text", "tool_use", "tool_result", "thinking", "image"
    text: str = ""           # text content (for text blocks)
    tool_name: str = ""      # tool name (for tool_use blocks)
    tool_id: str = ""        # tool call ID (for tool_use/tool_result)


@dataclass
class SessionMessage:
    """A parsed message from a JSONL session file."""

    record_type: str         # "user", "assistant", "system", etc.
    role: str = ""           # "user" or "assistant"
    content_blocks: list[ContentBlock] = field(default_factory=list)
    uuid: str = ""
    parent_uuid: str = ""
    timestamp: str = ""
    session_id: str = ""
    is_sidechain: bool = False


@dataclass
class SessionFile:
    """Metadata about a discovered JSONL session file."""

    file_path: Path
    project_dir: str         # e.g., "-Users-alice-Documents-projects-my-app"
    session_id: str          # UUID from filename
    file_size: int = 0
    is_subagent: bool = False


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

def discover_sessions(projects_root: Path) -> list[SessionFile]:
    """Find all main JSONL session files under ~/.claude/projects/.

    Skips subagent sessions (those nested under session-uuid/subagents/).
    Returns a list of SessionFile objects sorted by file size (largest first).
    """
    sessions: list[SessionFile] = []

    if not projects_root.exists():
        logger.warning("Projects root not found: %s", projects_root)
        return sessions

    for project_dir in sorted(projects_root.iterdir()):
        if not project_dir.is_dir():
            continue

        project_name = project_dir.name

        for item in project_dir.iterdir():
            if not item.name.endswith(".jsonl"):
                continue

            # Skip files inside subagent directories
            # Subagent paths look like: {session-uuid}/subagents/{sub-uuid}.jsonl
            if "subagents" in str(item):
                continue

            session_id = item.stem  # filename without .jsonl
            file_size = item.stat().st_size

            sessions.append(SessionFile(
                file_path=item,
                project_dir=project_name,
                session_id=session_id,
                file_size=file_size,
                is_subagent=False,
            ))

    sessions.sort(key=lambda s: s.file_size, reverse=True)
    logger.info("Discovered %d main sessions across %d project dirs (%.1f MB total)",
                len(sessions),
                len(set(s.project_dir for s in sessions)),
                sum(s.file_size for s in sessions) / 1024 / 1024)

    return sessions


# ---------------------------------------------------------------------------
# Streaming parser
# ---------------------------------------------------------------------------

def parse_session(session: SessionFile) -> Generator[SessionMessage, None, None]:
    """Stream-parse a JSONL session file, yielding SessionMessage objects.

    Reads one line at a time to handle large files without loading
    the entire file into memory.
    """
    line_count = 0
    error_count = 0

    with open(session.file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                error_count += 1
                if error_count <= 3:
                    logger.debug("JSON parse error at line %d in %s",
                                 line_count, session.file_path.name)
                continue

            record_type = record.get("type", "")

            # Only parse user and assistant messages into full objects
            if record_type in ("user", "assistant"):
                yield _parse_message_record(record, record_type)
            else:
                # Yield a minimal record for filtering decisions
                yield SessionMessage(
                    record_type=record_type,
                    session_id=record.get("sessionId", ""),
                )

    if error_count > 3:
        logger.warning("Session %s had %d JSON parse errors",
                       session.session_id, error_count)


def _parse_message_record(record: dict, record_type: str) -> SessionMessage:
    """Parse a user or assistant JSONL record into a SessionMessage."""
    message = record.get("message", {})
    role = message.get("role", record_type)
    raw_content = message.get("content", "")

    content_blocks = _parse_content(raw_content)

    return SessionMessage(
        record_type=record_type,
        role=role,
        content_blocks=content_blocks,
        uuid=record.get("uuid", ""),
        parent_uuid=record.get("parentUuid", ""),
        timestamp=record.get("timestamp", ""),
        session_id=record.get("sessionId", ""),
        is_sidechain=record.get("isSidechain", False),
    )


def _parse_content(raw_content: str | list) -> list[ContentBlock]:
    """Parse message content into ContentBlock objects.

    Content can be either:
      - A plain string (user messages often)
      - A list of content block dicts (assistant messages, some user messages)
    """
    if isinstance(raw_content, str):
        if raw_content.strip():
            return [ContentBlock(block_type="text", text=raw_content)]
        return []

    if not isinstance(raw_content, list):
        return []

    blocks: list[ContentBlock] = []
    for block in raw_content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type", "unknown")

        if block_type == "text":
            text = block.get("text", "")
            if text.strip():
                blocks.append(ContentBlock(block_type="text", text=text))

        elif block_type == "tool_use":
            blocks.append(ContentBlock(
                block_type="tool_use",
                tool_name=block.get("name", ""),
                tool_id=block.get("id", ""),
            ))

        elif block_type == "tool_result":
            blocks.append(ContentBlock(
                block_type="tool_result",
                tool_id=block.get("tool_use_id", ""),
            ))

        elif block_type == "thinking":
            blocks.append(ContentBlock(block_type="thinking"))

        elif block_type == "image":
            blocks.append(ContentBlock(block_type="image"))

        else:
            blocks.append(ContentBlock(block_type=block_type))

    return blocks
