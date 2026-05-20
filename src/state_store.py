"""
src/state_store.py — SQLite state management for the wiki pipeline.

Tracks processed sessions, extracted knowledge, generated wiki pages, and
pipeline run history.  Used by pipeline.py to avoid reprocessing and to
maintain lineage from source → extraction → wiki page.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Extraction:
    """A single knowledge unit extracted from a memory source."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source_id: str = ""          # session or memory entry identifier
    source_type: str = ""        # "pact_memory" | "agent_memory" | "project_memory"
    extraction_type: str = ""    # entity, pattern, gotcha, decision, etc.
    topic: str = ""
    title: str = ""
    content: str = ""
    confidence: float = 1.0
    context: str = ""            # project / tool this relates to
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class WikiPageRecord:
    """Metadata about a generated wiki page stored in the state DB."""

    page_path: str = ""
    page_type: str = ""
    title: str = ""
    extraction_ids: list[str] = field(default_factory=list)
    last_generated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    word_count: int = 0
    needs_review: bool = False


@dataclass
class PipelineRun:
    """Audit record for a single pipeline execution."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str | None = None
    sessions_processed: int = 0
    extractions_created: int = 0
    pages_created: int = 0
    pages_updated: int = 0
    config: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS processed_sessions (
    session_id TEXT PRIMARY KEY,
    project_path TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size INTEGER,
    message_count INTEGER,
    processed_at TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    classification TEXT,
    extraction_ids TEXT,
    cost_usd REAL
);

CREATE TABLE IF NOT EXISTS extractions (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    extraction_type TEXT NOT NULL,
    topic TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    confidence REAL,
    context TEXT,
    clustered_into TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS wiki_pages (
    page_path TEXT PRIMARY KEY,
    page_type TEXT NOT NULL,
    title TEXT NOT NULL,
    extraction_ids TEXT NOT NULL,
    last_generated TEXT NOT NULL,
    word_count INTEGER,
    quality_score REAL,
    needs_review INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    sessions_processed INTEGER,
    extractions_created INTEGER,
    pages_created INTEGER,
    pages_updated INTEGER,
    total_cost_usd REAL,
    config TEXT
);
"""


class StateStore:
    """Thin wrapper around a local SQLite database for pipeline state."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- extractions --------------------------------------------------------

    def save_extraction(self, ext: Extraction) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO extractions
               (id, source_id, source_type, extraction_type, topic, title,
                content, confidence, context, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ext.id, ext.source_id, ext.source_type, ext.extraction_type,
                ext.topic, ext.title, ext.content, ext.confidence,
                ext.context, ext.created_at,
            ),
        )
        self._conn.commit()

    def save_extractions(self, extractions: list[Extraction]) -> None:
        for ext in extractions:
            self.save_extraction(ext)

    def get_extractions(self) -> list[Extraction]:
        rows = self._conn.execute(
            "SELECT id, source_id, source_type, extraction_type, topic, "
            "title, content, confidence, context, created_at "
            "FROM extractions"
        ).fetchall()
        return [
            Extraction(
                id=r[0], source_id=r[1], source_type=r[2],
                extraction_type=r[3], topic=r[4], title=r[5],
                content=r[6], confidence=r[7], context=r[8],
                created_at=r[9],
            )
            for r in rows
        ]

    def get_extraction_ids_for_source(self, source_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT id FROM extractions WHERE source_id = ?", (source_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def is_source_processed(self, source_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM extractions WHERE source_id = ? LIMIT 1",
            (source_id,),
        ).fetchone()
        return row is not None

    # -- wiki pages ---------------------------------------------------------

    def save_wiki_page(self, page: WikiPageRecord) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO wiki_pages
               (page_path, page_type, title, extraction_ids,
                last_generated, word_count, needs_review)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                page.page_path, page.page_type, page.title,
                json.dumps(page.extraction_ids), page.last_generated,
                page.word_count, int(page.needs_review),
            ),
        )
        self._conn.commit()

    def get_wiki_pages(self) -> list[WikiPageRecord]:
        rows = self._conn.execute(
            "SELECT page_path, page_type, title, extraction_ids, "
            "last_generated, word_count, needs_review FROM wiki_pages"
        ).fetchall()
        return [
            WikiPageRecord(
                page_path=r[0], page_type=r[1], title=r[2],
                extraction_ids=json.loads(r[3]), last_generated=r[4],
                word_count=r[5], needs_review=bool(r[6]),
            )
            for r in rows
        ]

    # -- pipeline runs ------------------------------------------------------

    def start_run(self, run: PipelineRun) -> None:
        self._conn.execute(
            """INSERT INTO pipeline_runs
               (run_id, started_at, config)
               VALUES (?, ?, ?)""",
            (run.run_id, run.started_at, json.dumps(run.config)),
        )
        self._conn.commit()

    def finish_run(self, run: PipelineRun) -> None:
        self._conn.execute(
            """UPDATE pipeline_runs SET
               completed_at = ?, sessions_processed = ?,
               extractions_created = ?, pages_created = ?,
               pages_updated = ?, total_cost_usd = ?
               WHERE run_id = ?""",
            (
                run.completed_at, run.sessions_processed,
                run.extractions_created, run.pages_created,
                run.pages_updated, 0.0, run.run_id,
            ),
        )
        self._conn.commit()

    # -- processed sessions ------------------------------------------------

    def save_processed_session(
        self,
        session_id: str,
        project_path: str,
        file_path: str,
        file_size: int,
        message_count: int,
        pipeline_version: str,
        classification: object | None = None,
        extraction_ids: list[str] | None = None,
    ) -> None:
        """Record a session as processed for incremental run support."""
        classification_json = ""
        if classification is not None:
            import dataclasses
            if dataclasses.is_dataclass(classification):
                classification_json = json.dumps(dataclasses.asdict(classification))

        self._conn.execute(
            """INSERT OR REPLACE INTO processed_sessions
               (session_id, project_path, file_path, file_size,
                message_count, processed_at, pipeline_version,
                classification, extraction_ids)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, project_path, file_path, file_size,
                message_count, datetime.now(timezone.utc).isoformat(),
                pipeline_version, classification_json,
                json.dumps(extraction_ids or []),
            ),
        )
        self._conn.commit()

    def is_session_processed(self, session_id: str, pipeline_version: str) -> bool:
        """Check if a session has been processed with the given pipeline version."""
        row = self._conn.execute(
            "SELECT 1 FROM processed_sessions "
            "WHERE session_id = ? AND pipeline_version = ? LIMIT 1",
            (session_id, pipeline_version),
        ).fetchone()
        return row is not None

    def get_processed_sessions(self) -> list[dict]:
        """Return all processed session records."""
        rows = self._conn.execute(
            "SELECT session_id, project_path, file_path, file_size, "
            "message_count, processed_at, pipeline_version "
            "FROM processed_sessions"
        ).fetchall()
        return [
            {
                "session_id": r[0], "project_path": r[1], "file_path": r[2],
                "file_size": r[3], "message_count": r[4],
                "processed_at": r[5], "pipeline_version": r[6],
            }
            for r in rows
        ]

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        self._conn.close()
