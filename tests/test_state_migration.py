"""
tests/test_state_migration.py -- v1.0.0 -> v1.1.0 schema migration tests.

Covers the 7 cases from architecture-v1.1.0/state-migration.md section
"Migration test plan": fresh DB, v1.0.0 upgrade, idempotent re-open,
mid-migration self-healing, duplicate-column race, read-only filesystem,
and backfill ordering by started_at.
"""

from __future__ import annotations

import os
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.state_store import (
    Extraction,
    PipelineRun,
    StateMigrationError,
    StateStore,
)


# v1.0.0 schema captured as a fixture. Used to seed synthetic legacy DBs.
_V1_0_0_SCHEMA = """
CREATE TABLE processed_sessions (
    session_id TEXT PRIMARY KEY, project_path TEXT NOT NULL,
    file_path TEXT NOT NULL, file_size INTEGER, message_count INTEGER,
    processed_at TEXT NOT NULL, pipeline_version TEXT NOT NULL,
    classification TEXT, extraction_ids TEXT, cost_usd REAL
);
CREATE TABLE extractions (
    id TEXT PRIMARY KEY, source_id TEXT NOT NULL, source_type TEXT NOT NULL,
    extraction_type TEXT NOT NULL, topic TEXT NOT NULL, title TEXT NOT NULL,
    content TEXT NOT NULL, confidence REAL, context TEXT,
    clustered_into TEXT, created_at TEXT NOT NULL
);
CREATE TABLE wiki_pages (
    page_path TEXT PRIMARY KEY, page_type TEXT NOT NULL, title TEXT NOT NULL,
    extraction_ids TEXT NOT NULL, last_generated TEXT NOT NULL,
    word_count INTEGER, quality_score REAL, needs_review INTEGER DEFAULT 0
);
CREATE TABLE pipeline_runs (
    run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, completed_at TEXT,
    sessions_processed INTEGER, extractions_created INTEGER,
    pages_created INTEGER, pages_updated INTEGER, total_cost_usd REAL,
    config TEXT
);
"""


def _seed_v1_0_0(db_path: Path) -> None:
    """Create a synthetic v1.0.0 DB at the given path."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_V1_0_0_SCHEMA)
    conn.commit()
    conn.close()


class TestFreshDb(unittest.TestCase):
    """Case 1: fresh DB has both new columns at install."""

    def test_fresh_db_has_new_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(Path(tmp) / "fresh.db")
            try:
                ext_cols = [
                    r[1] for r in store._conn.execute(
                        "PRAGMA table_info(extractions)"
                    ).fetchall()
                ]
                run_cols = [
                    r[1] for r in store._conn.execute(
                        "PRAGMA table_info(pipeline_runs)"
                    ).fetchall()
                ]
                self.assertIn("last_confirmed_run_id", ext_cols)
                self.assertIn("run_id_int", run_cols)
            finally:
                store.close()


class TestV1_0_0Upgrade(unittest.TestCase):
    """Case 2: opening a v1.0.0 DB migrates it in place."""

    def test_upgrade_adds_columns_and_keeps_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "legacy.db"
            _seed_v1_0_0(db)
            # Seed one legacy extraction.
            conn = sqlite3.connect(str(db))
            conn.execute(
                "INSERT INTO extractions (id, source_id, source_type, "
                "extraction_type, topic, title, content, confidence, context, "
                "created_at) VALUES "
                "('e1', 's1', 'pact_memory', 'note', 'topicA', 'Title', "
                "'content', 1.0, '', '2024-01-01T00:00:00')"
            )
            conn.commit()
            conn.close()

            store = StateStore(db)
            try:
                ext_cols = [
                    r[1] for r in store._conn.execute(
                        "PRAGMA table_info(extractions)"
                    ).fetchall()
                ]
                run_cols = [
                    r[1] for r in store._conn.execute(
                        "PRAGMA table_info(pipeline_runs)"
                    ).fetchall()
                ]
                self.assertIn("last_confirmed_run_id", ext_cols)
                self.assertIn("run_id_int", run_cols)

                # Legacy row defaulted to 0 sentinel (not NULL).
                row = store._conn.execute(
                    "SELECT last_confirmed_run_id FROM extractions "
                    "WHERE id = ?", ("e1",)
                ).fetchone()
                self.assertEqual(row, (0,))
            finally:
                store.close()


class TestIdempotentReopen(unittest.TestCase):
    """Case 3: opening a migrated DB a second time is a no-op."""

    def test_second_open_does_not_alter(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "twice.db"
            _seed_v1_0_0(db)
            # First open performs the migration.
            store_first = StateStore(db)
            cols_after_first = sorted(
                r[1] for r in store_first._conn.execute(
                    "PRAGMA table_info(extractions)"
                ).fetchall()
            )
            store_first.close()

            # Second open: re-invoking _migrate_v1_1_0 must be a no-op. Both
            # _column_exists checks return True, no DDL fires, schema is
            # byte-identical to first-open state.
            store_second = StateStore(db)
            try:
                self.assertTrue(store_second._column_exists(
                    "extractions", "last_confirmed_run_id"
                ))
                self.assertTrue(store_second._column_exists(
                    "pipeline_runs", "run_id_int"
                ))
                # Calling the migration again must remain a no-op.
                store_second._migrate_v1_1_0()
                cols_after_second = sorted(
                    r[1] for r in store_second._conn.execute(
                        "PRAGMA table_info(extractions)"
                    ).fetchall()
                )
                self.assertEqual(cols_after_first, cols_after_second)
            finally:
                store_second.close()


class TestPartialMigrationSelfHealing(unittest.TestCase):
    """Case 4: synthetic mid-migration DB heals itself on next open."""

    def test_only_missing_column_is_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "partial.db"
            _seed_v1_0_0(db)
            # Manually add only the first column to simulate a crash
            # between the two ALTERs.
            conn = sqlite3.connect(str(db))
            conn.execute(
                "ALTER TABLE extractions "
                "ADD COLUMN last_confirmed_run_id INTEGER DEFAULT 0"
            )
            conn.commit()
            conn.close()

            store = StateStore(db)
            try:
                run_cols = [
                    r[1] for r in store._conn.execute(
                        "PRAGMA table_info(pipeline_runs)"
                    ).fetchall()
                ]
                self.assertIn("run_id_int", run_cols)
            finally:
                store.close()


class TestDuplicateColumnRace(unittest.TestCase):
    """Case 5: 'duplicate column name' from a concurrent ALTER is swallowed."""

    def test_duplicate_column_error_swallowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "race.db"
            _seed_v1_0_0(db)
            store = StateStore(db)
            try:
                # Force _column_exists to lie (returns False) so the migration
                # path tries to ALTER an already-present column, raising the
                # 'duplicate column name' OperationalError that the migration
                # is supposed to swallow.
                with mock.patch.object(
                    store, "_column_exists", return_value=False
                ):
                    # Should not raise.
                    store._migrate_v1_1_0()
            finally:
                store.close()


class TestReadOnlyFilesystem(unittest.TestCase):
    """Case 6: real failure path -- mock _column_exists to force a re-ALTER
    that produces an OperationalError unrelated to duplicate-column, and
    verify it gets wrapped in StateMigrationError.

    We can't reliably make the SQLite file truly unwritable in a temp dir
    on all CI platforms, so we exercise the wrapper logic via a mock.
    """

    def test_operational_error_wrapped_as_migration_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ro.db"
            # Use a fresh DB so __init__ is the relevant codepath.
            store = StateStore(db)
            store.close()

            # Reopen with a forced failure during migration.
            with mock.patch(
                "src.state_store.StateStore._migrate_v1_1_0",
                side_effect=sqlite3.OperationalError(
                    "database or disk is full"
                ),
            ):
                with self.assertRaises(StateMigrationError) as ctx:
                    StateStore(db)
            self.assertIn("v1.1.0", str(ctx.exception))


class TestBackfillOrdering(unittest.TestCase):
    """Case 7: legacy pipeline_runs rows are backfilled in started_at order."""

    def test_backfill_orders_by_started_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "backfill.db"
            _seed_v1_0_0(db)
            conn = sqlite3.connect(str(db))
            # Insert out-of-order to ensure ROWID does not accidentally match.
            conn.execute(
                "INSERT INTO pipeline_runs (run_id, started_at) "
                "VALUES ('rZ', '2024-03-01')"
            )
            conn.execute(
                "INSERT INTO pipeline_runs (run_id, started_at) "
                "VALUES ('rA', '2024-01-01')"
            )
            conn.execute(
                "INSERT INTO pipeline_runs (run_id, started_at) "
                "VALUES ('rM', '2024-02-01')"
            )
            conn.commit()
            conn.close()

            store = StateStore(db)
            try:
                rows = store._conn.execute(
                    "SELECT run_id, run_id_int FROM pipeline_runs "
                    "ORDER BY run_id_int"
                ).fetchall()
                # Backfilled 1,2,3 in started_at order rA, rM, rZ.
                self.assertEqual(
                    rows, [("rA", 1), ("rM", 2), ("rZ", 3)]
                )
            finally:
                store.close()


class TestStartRunAssignsOrdinal(unittest.TestCase):
    """start_run() assigns run_id_int atomically and writes back."""

    def test_start_run_assigns_monotonic_run_id_int(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(Path(tmp) / "ord.db")
            try:
                r1 = PipelineRun(config={})
                store.start_run(r1)
                self.assertEqual(r1.run_id_int, 1)

                r2 = PipelineRun(config={})
                store.start_run(r2)
                self.assertEqual(r2.run_id_int, 2)
            finally:
                store.close()


class TestSaveExtractionStampsRunId(unittest.TestCase):
    """save_extraction(ext, run_id_int=N) stamps last_confirmed_run_id."""

    def test_save_extraction_stamps_lcri(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(Path(tmp) / "stamp.db")
            try:
                ext = Extraction(
                    id="e1", source_id="s1", source_type="pact_memory",
                    extraction_type="note", topic="x", title="t",
                    content="c", confidence=1.0, context="",
                )
                store.save_extraction(ext, run_id_int=5)
                row = store._conn.execute(
                    "SELECT last_confirmed_run_id FROM extractions "
                    "WHERE id = ?", ("e1",)
                ).fetchone()
                self.assertEqual(row, (5,))
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
