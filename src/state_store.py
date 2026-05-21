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
    # v1.1.0: monotonic ordinal for "N runs back" comparisons (rot pruning).
    # Assigned in start_run via COALESCE(MAX,0)+1; 0 only for unsaved instances.
    run_id_int: int = 0


@dataclass
class PruneReport:
    """Outcome of a prune_stale() call. Dry-run returns this without deleting."""

    extractions_deleted: int = 0
    extractions_inspected: int = 0
    affected_pages: list[str] = field(default_factory=list)
    cutoff_run_id_int: int = 0


class StateMigrationError(RuntimeError):
    """Raised when the v1.1.0 schema migration cannot complete.

    The migration is non-destructive (ALTERs only). When this is raised, the
    pre-existing v1.0.0 data is intact. Recover by freeing disk space and
    re-running, or by `mv state.db state.db.v1.0.bak` to start fresh — the
    DB is a CACHE; extractions are reconstructible from memory sources.
    """


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
    created_at TEXT NOT NULL,
    last_confirmed_run_id INTEGER DEFAULT 0
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
    config TEXT,
    run_id_int INTEGER
);
"""


class StateStore:
    """Thin wrapper around a local SQLite database for pipeline state."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        try:
            self._migrate_v1_1_0()
        except sqlite3.OperationalError as exc:
            raise StateMigrationError(
                f"Failed to migrate state.db to v1.1.0 schema: {exc}. "
                "This is non-destructive — your v1.0.0 data is intact. "
                "Free disk space and re-run, or `mv state.db state.db.v1.0.bak` "
                "to start fresh."
            ) from exc
        self._conn.commit()

    # -- v1.1.0 migration ---------------------------------------------------

    def _column_exists(self, table: str, column: str) -> bool:
        """True iff the table has a column with the given name.

        PRAGMA table_info returns (cid, name, type, notnull, dflt, pk).
        """
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(r[1] == column for r in rows)

    def _migrate_v1_1_0(self) -> None:
        """Add last_confirmed_run_id and run_id_int columns if missing.

        Idempotent: each ALTER is gated by PRAGMA table_info, and the
        ``duplicate column name`` race is swallowed (another process may
        have won the DDL race in WAL mode). On a FRESH DB, ``_SCHEMA``
        already creates both columns and both guards short-circuit.
        """
        migrations = [
            (
                "extractions",
                "last_confirmed_run_id",
                "ALTER TABLE extractions "
                "ADD COLUMN last_confirmed_run_id INTEGER DEFAULT 0",
            ),
            (
                "pipeline_runs",
                "run_id_int",
                "ALTER TABLE pipeline_runs ADD COLUMN run_id_int INTEGER",
            ),
        ]
        for table, column, ddl in migrations:
            if self._column_exists(table, column):
                continue
            try:
                self._conn.execute(ddl)
            except sqlite3.OperationalError as exc:
                if "duplicate column name" in str(exc).lower():
                    # Concurrent ALTER won the race; column now exists.
                    continue
                raise
        self._backfill_run_id_int_if_needed()

    def _backfill_run_id_int_if_needed(self) -> None:
        """Backfill NULL run_id_int values on existing pipeline_runs rows.

        Ordered by started_at so legacy runs receive a deterministic ordinal.
        Idempotent: only touches rows where run_id_int IS NULL.
        """
        rows = self._conn.execute(
            "SELECT run_id FROM pipeline_runs "
            "WHERE run_id_int IS NULL ORDER BY started_at"
        ).fetchall()
        if not rows:
            return
        # Resume the sequence above any already-backfilled rows.
        max_row = self._conn.execute(
            "SELECT COALESCE(MAX(run_id_int), 0) FROM pipeline_runs"
        ).fetchone()
        next_int = (max_row[0] or 0) + 1
        for (run_id,) in rows:
            self._conn.execute(
                "UPDATE pipeline_runs SET run_id_int = ? WHERE run_id = ?",
                (next_int, run_id),
            )
            next_int += 1

    # -- extractions --------------------------------------------------------

    def save_extraction(
        self, ext: Extraction, run_id_int: int = 0
    ) -> None:
        """Insert-or-replace an extraction.

        ``run_id_int`` is stamped into ``last_confirmed_run_id`` so a
        freshly-saved extraction starts life confirmed by the current run.
        Default 0 keeps the v1.0.0 callsite signature working (existing
        callers that don't pass the argument get the sentinel; a later
        touch_extractions call from the pipeline will update it).
        """
        self._conn.execute(
            """INSERT OR REPLACE INTO extractions
               (id, source_id, source_type, extraction_type, topic, title,
                content, confidence, context, created_at,
                last_confirmed_run_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ext.id, ext.source_id, ext.source_type, ext.extraction_type,
                ext.topic, ext.title, ext.content, ext.confidence,
                ext.context, ext.created_at, run_id_int,
            ),
        )
        self._conn.commit()

    def save_extractions(
        self, extractions: list[Extraction], run_id_int: int = 0
    ) -> None:
        for ext in extractions:
            self.save_extraction(ext, run_id_int=run_id_int)

    def touch_extractions(
        self, extraction_ids: list[str], run_id_int: int
    ) -> int:
        """UPDATE last_confirmed_run_id for the listed extractions.

        Empty list is a no-op (no SQL executed). Returns the number of rows
        updated, which may be less than len(extraction_ids) if some ids do
        not exist in the DB (e.g. fresh extractions not yet saved).
        """
        if not extraction_ids:
            return 0
        # Chunk the IN clause to avoid SQLite's default 999-parameter cap.
        chunk = 500
        total = 0
        for i in range(0, len(extraction_ids), chunk):
            batch = extraction_ids[i : i + chunk]
            placeholders = ",".join("?" for _ in batch)
            cur = self._conn.execute(
                f"UPDATE extractions SET last_confirmed_run_id = ? "
                f"WHERE id IN ({placeholders})",
                (run_id_int, *batch),
            )
            total += cur.rowcount or 0
        self._conn.commit()
        return total

    def prune_stale(
        self, older_than_run_id_int: int, dry_run: bool = False
    ) -> PruneReport:
        """Identify (and optionally delete) extractions confirmed before cutoff.

        An extraction is stale when ``last_confirmed_run_id < cutoff``
        (strict ``<``; rows AT the cutoff survive).
        ``affected_pages`` is derived by joining the doomed extraction ids
        against ``wiki_pages.extraction_ids`` (stored as JSON arrays).
        With ``dry_run=True`` no DELETE runs; the report is identical to
        what an ``--apply`` run would produce.
        """
        if not self._column_exists("extractions", "last_confirmed_run_id"):
            raise StateMigrationError(
                "prune_stale called on a DB missing last_confirmed_run_id. "
                "Re-open the store to run the v1.1.0 migration."
            )

        inspected_row = self._conn.execute(
            "SELECT COUNT(*) FROM extractions"
        ).fetchone()
        inspected = int(inspected_row[0]) if inspected_row else 0

        stale_rows = self._conn.execute(
            "SELECT id FROM extractions "
            "WHERE last_confirmed_run_id < ?",
            (older_than_run_id_int,),
        ).fetchall()
        stale_ids = [r[0] for r in stale_rows]

        affected_pages = self._pages_referencing_extractions(stale_ids)

        report = PruneReport(
            extractions_deleted=0 if dry_run else len(stale_ids),
            extractions_inspected=inspected,
            affected_pages=affected_pages,
            cutoff_run_id_int=older_than_run_id_int,
        )

        if dry_run or not stale_ids:
            return report

        chunk = 500
        for i in range(0, len(stale_ids), chunk):
            batch = stale_ids[i : i + chunk]
            placeholders = ",".join("?" for _ in batch)
            self._conn.execute(
                f"DELETE FROM extractions WHERE id IN ({placeholders})",
                batch,
            )
        self._conn.commit()
        return report

    def _pages_referencing_extractions(
        self, extraction_ids: list[str]
    ) -> list[str]:
        """Return distinct wiki_pages.page_path values that reference any
        of the given extraction ids (extraction_ids column stores a JSON
        array). Returns [] if the input list is empty."""
        if not extraction_ids:
            return []
        wanted = set(extraction_ids)
        rows = self._conn.execute(
            "SELECT page_path, extraction_ids FROM wiki_pages"
        ).fetchall()
        affected: list[str] = []
        for page_path, ext_ids_json in rows:
            try:
                page_ext_ids = json.loads(ext_ids_json or "[]")
            except json.JSONDecodeError:
                continue
            if any(eid in wanted for eid in page_ext_ids):
                affected.append(page_path)
        return sorted(affected)

    def get_unfinished_run(self) -> str | None:
        """Return the run_id of an in-flight pipeline run, or None.

        Used by the prune subcommand to refuse-to-run while a pipeline is
        mid-flight (avoids racing the touch_extractions tail).
        """
        row = self._conn.execute(
            "SELECT run_id FROM pipeline_runs "
            "WHERE completed_at IS NULL LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def max_run_id_int(self) -> int:
        """Highest assigned run_id_int, or 0 if no runs exist yet."""
        row = self._conn.execute(
            "SELECT COALESCE(MAX(run_id_int), 0) FROM pipeline_runs"
        ).fetchone()
        return int(row[0]) if row else 0

    def get_extraction(self, extraction_id: str) -> Extraction | None:
        """Fetch a single extraction by id, or None."""
        row = self._conn.execute(
            "SELECT id, source_id, source_type, extraction_type, topic, "
            "title, content, confidence, context, created_at "
            "FROM extractions WHERE id = ?",
            (extraction_id,),
        ).fetchone()
        if row is None:
            return None
        return Extraction(
            id=row[0], source_id=row[1], source_type=row[2],
            extraction_type=row[3], topic=row[4], title=row[5],
            content=row[6], confidence=row[7], context=row[8],
            created_at=row[9],
        )

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
        """Persist a new pipeline run.

        Atomically assigns ``run_id_int = COALESCE(MAX,0)+1`` inside the
        INSERT to avoid a TOCTOU window between two concurrent runs, and
        mutates ``run.run_id_int`` to the assigned value so the caller
        can pass it to touch_extractions later.
        """
        self._conn.execute(
            """INSERT INTO pipeline_runs
               (run_id, started_at, config, run_id_int)
               VALUES (?, ?, ?,
                       (SELECT COALESCE(MAX(run_id_int), 0) + 1
                        FROM pipeline_runs))""",
            (run.run_id, run.started_at, json.dumps(run.config)),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT run_id_int FROM pipeline_runs WHERE run_id = ?",
            (run.run_id,),
        ).fetchone()
        if row is not None and row[0] is not None:
            run.run_id_int = int(row[0])

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
