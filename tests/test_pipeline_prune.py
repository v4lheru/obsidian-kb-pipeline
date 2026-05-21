"""
tests/test_pipeline_prune.py -- prune subcommand integration tests.

Exercises run_prune() at the orchestration layer (above StateStore but
below the argparse CLI): refuses on unfinished run, dry-run by default,
--apply deletes, --rebuild calls synthesizer for affected pages.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.config import PipelineConfig
from src.run_prune import run_prune
from src.state_store import (
    Extraction,
    PipelineRun,
    StateStore,
    WikiPageRecord,
)


def _make_ext(eid: str) -> Extraction:
    return Extraction(
        id=eid, source_id=f"src-{eid}", source_type="pact_memory",
        extraction_type="note", topic="x", title=f"t-{eid}",
        content=f"c-{eid}", confidence=1.0, context="",
    )


def _populate_runs(store: StateStore, n: int) -> int:
    """Create n completed pipeline_runs rows, return last run_id_int."""
    last = 0
    for _ in range(n):
        run = PipelineRun(config={})
        store.start_run(run)
        run.completed_at = "2024-01-01T00:00:00"
        store.finish_run(run)
        last = run.run_id_int
    return last


class TestRunPruneRefusesOnUnfinishedRun(unittest.TestCase):
    """Refuses to delete while a pipeline_runs row has completed_at IS NULL."""

    def test_refuses_when_in_flight(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "uf.db"
            store = StateStore(db)
            run = PipelineRun(config={})
            store.start_run(run)
            # Do NOT call finish_run -- simulates a crashed run.
            store.close()

            config = PipelineConfig()
            object.__setattr__(config.paths, "state_db", db)

            with self.assertRaises(RuntimeError) as ctx:
                run_prune(config, keep_runs=1, apply=True)
            self.assertIn("incomplete", str(ctx.exception).lower())


class TestRunPruneDryRun(unittest.TestCase):
    """Without --apply, no rows are deleted."""

    def test_dry_run_keeps_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "dry.db"
            store = StateStore(db)
            _populate_runs(store, 20)  # max_run_id_int = 20
            store.save_extraction(_make_ext("stale"), run_id_int=3)
            store.save_extraction(_make_ext("fresh"), run_id_int=18)
            store.close()

            config = PipelineConfig()
            object.__setattr__(config.paths, "state_db", db)

            # apply=False (default) -- report only.
            report = run_prune(config, keep_runs=12, apply=False)
            self.assertEqual(report.extractions_deleted, 0)
            # Cutoff = 20 - 12 + 1 = 9. 'stale' (3) is below; 'fresh' (18) is above.
            self.assertEqual(report.cutoff_run_id_int, 9)

            # Verify nothing was deleted.
            store2 = StateStore(db)
            try:
                rows = store2._conn.execute(
                    "SELECT id FROM extractions"
                ).fetchall()
                self.assertEqual({r[0] for r in rows}, {"stale", "fresh"})
            finally:
                store2.close()


class TestRunPruneApply(unittest.TestCase):
    """With --apply, stale rows are deleted; fresh rows survive."""

    def test_apply_deletes_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "apply.db"
            store = StateStore(db)
            _populate_runs(store, 20)
            store.save_extraction(_make_ext("stale"), run_id_int=3)
            store.save_extraction(_make_ext("fresh"), run_id_int=18)
            store.close()

            config = PipelineConfig()
            object.__setattr__(config.paths, "state_db", db)

            report = run_prune(config, keep_runs=12, apply=True)
            self.assertEqual(report.extractions_deleted, 1)

            store2 = StateStore(db)
            try:
                rows = store2._conn.execute(
                    "SELECT id FROM extractions"
                ).fetchall()
                self.assertEqual({r[0] for r in rows}, {"fresh"})
            finally:
                store2.close()


class TestRunPruneNoopBelowMinHistory(unittest.TestCase):
    """When history < keep_runs, prune is a no-op."""

    def test_no_op_when_fewer_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "few.db"
            store = StateStore(db)
            _populate_runs(store, 3)  # only 3 runs
            store.save_extraction(_make_ext("legacy"))  # DEFAULT 0
            store.close()

            config = PipelineConfig()
            object.__setattr__(config.paths, "state_db", db)

            report = run_prune(config, keep_runs=12, apply=True)
            self.assertEqual(report.extractions_deleted, 0)
            self.assertEqual(report.cutoff_run_id_int, 0)


class TestRunPruneRebuild(unittest.TestCase):
    """--rebuild invokes synthesizer for affected pages."""

    def test_rebuild_calls_synthesizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "rebuild.db"
            store = StateStore(db)
            _populate_runs(store, 20)
            # One stale, one fresh, both on same page.
            store.save_extraction(_make_ext("stale"), run_id_int=3)
            store.save_extraction(_make_ext("fresh"), run_id_int=18)
            store.save_wiki_page(WikiPageRecord(
                page_path="Patterns/mixed.md",
                page_type="auto",
                title="Mixed",
                extraction_ids=["stale", "fresh"],
                word_count=100,
            ))
            store.close()

            vault_root = Path(tmp) / "vault"
            (vault_root / "Coding-Notes").mkdir(parents=True)

            config = PipelineConfig()
            object.__setattr__(config.paths, "state_db", db)

            # VAULT_ROOT is required because _rebuild_affected_pages
            # instantiates a real WikiWriter pointed at the vault. We patch
            # both the synthesizer call and WikiWriter at the deferred import
            # sites used by run_prune._rebuild_affected_pages.
            with mock.patch.dict("os.environ", {"VAULT_ROOT": str(vault_root)}), \
                 mock.patch("src.synthesizer.synthesize_page") as mock_syn, \
                 mock.patch("src.wiki_writer.WikiWriter") as mock_writer_cls:
                mock_syn.return_value = mock.MagicMock(
                    path="Patterns/mixed.md", is_new=False
                )
                report = run_prune(
                    config, keep_runs=12, apply=True, rebuild=True
                )
                # Stale was pruned.
                self.assertEqual(report.extractions_deleted, 1)
                # synthesize_page was called once for the affected page.
                self.assertEqual(mock_syn.call_count, 1)
                # Writer was instantiated and write_pages called.
                self.assertTrue(mock_writer_cls.called)


class TestRunPruneInvalidKeepRuns(unittest.TestCase):
    """keep_runs < 1 raises ValueError."""

    def test_zero_keep_runs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "z.db"
            store = StateStore(db)
            store.close()
            config = PipelineConfig()
            object.__setattr__(config.paths, "state_db", db)
            with self.assertRaises(ValueError):
                run_prune(config, keep_runs=0, apply=True)


if __name__ == "__main__":
    unittest.main()
