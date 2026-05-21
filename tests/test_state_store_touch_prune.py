"""
tests/test_state_store_touch_prune.py -- touch_extractions + prune_stale tests.

Covers row counts, the empty-list no-op, dry-run isolation, strict < cutoff
semantics, the DEFAULT 0 legacy-row behavior, and affected_pages derivation
via the wiki_pages.extraction_ids JSON join.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.state_store import (
    Extraction,
    PipelineRun,
    StateStore,
    WikiPageRecord,
)


def _make_extraction(eid: str, topic: str = "x") -> Extraction:
    return Extraction(
        id=eid, source_id=f"src-{eid}", source_type="pact_memory",
        extraction_type="note", topic=topic, title=f"t-{eid}",
        content=f"c-{eid}", confidence=1.0, context="",
    )


class TestTouchExtractions(unittest.TestCase):
    """touch_extractions row counts + empty no-op."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = StateStore(Path(self._tmp.name) / "touch.db")

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_touches_only_listed_ids(self):
        for eid in ["a", "b", "c"]:
            self.store.save_extraction(_make_extraction(eid), run_id_int=0)
        n = self.store.touch_extractions(["a", "c"], 7)
        self.assertEqual(n, 2)
        rows = dict(self.store._conn.execute(
            "SELECT id, last_confirmed_run_id FROM extractions"
        ).fetchall())
        self.assertEqual(rows["a"], 7)
        self.assertEqual(rows["b"], 0)
        self.assertEqual(rows["c"], 7)

    def test_empty_list_no_op(self):
        # Should not raise, returns 0, no SQL executed visibly.
        self.assertEqual(self.store.touch_extractions([], 99), 0)

    def test_missing_ids_silently_skipped(self):
        self.store.save_extraction(_make_extraction("a"), run_id_int=0)
        # 'nope' doesn't exist; touch_extractions stamps only matching ids.
        n = self.store.touch_extractions(["a", "nope"], 3)
        self.assertEqual(n, 1)


class TestPruneStale(unittest.TestCase):
    """prune_stale strict < semantics + dry-run + DEFAULT 0 behavior."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = StateStore(Path(self._tmp.name) / "prune.db")

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def _save_with_lcri(self, eid: str, lcri: int) -> None:
        ext = _make_extraction(eid)
        self.store.save_extraction(ext, run_id_int=lcri)

    def test_strict_less_than_cutoff(self):
        # last_confirmed_run_id values: 3, 4, 5, 6; cutoff=5 -> delete 3,4
        self._save_with_lcri("e3", 3)
        self._save_with_lcri("e4", 4)
        self._save_with_lcri("e5", 5)
        self._save_with_lcri("e6", 6)

        report = self.store.prune_stale(5, dry_run=False)
        self.assertEqual(report.extractions_deleted, 2)
        survivors = {
            r[0] for r in self.store._conn.execute(
                "SELECT id FROM extractions"
            ).fetchall()
        }
        self.assertEqual(survivors, {"e5", "e6"})
        self.assertEqual(report.cutoff_run_id_int, 5)

    def test_dry_run_does_not_delete(self):
        self._save_with_lcri("e3", 3)
        self._save_with_lcri("e6", 6)

        report = self.store.prune_stale(5, dry_run=True)
        self.assertEqual(report.extractions_deleted, 0)
        # Row count unchanged.
        n = self.store._conn.execute(
            "SELECT COUNT(*) FROM extractions"
        ).fetchone()[0]
        self.assertEqual(n, 2)

    def test_default_zero_legacy_rows_pruned_above_cutoff(self):
        # Save without specifying run_id_int -> DEFAULT 0 sentinel.
        self.store.save_extraction(_make_extraction("legacy"))
        self.store.save_extraction(_make_extraction("fresh"), run_id_int=15)

        # cutoff=13 (e.g. max_run=24, keep_runs=12 -> cutoff=13)
        report = self.store.prune_stale(13, dry_run=False)
        self.assertEqual(report.extractions_deleted, 1)
        survivors = {
            r[0] for r in self.store._conn.execute(
                "SELECT id FROM extractions"
            ).fetchall()
        }
        self.assertEqual(survivors, {"fresh"})

    def test_default_zero_legacy_rows_survive_below_first_cutoff(self):
        # Before 13 runs accumulate, the cutoff is < 1 and legacy stays.
        self.store.save_extraction(_make_extraction("legacy"))
        # cutoff = 0: only rows with lcri < 0 would be deleted; legacy at 0
        # survives by strict <.
        report = self.store.prune_stale(0, dry_run=False)
        self.assertEqual(report.extractions_deleted, 0)

    def test_inspected_count_reflects_all_rows(self):
        for eid in ["a", "b", "c"]:
            self._save_with_lcri(eid, 1)
        report = self.store.prune_stale(2, dry_run=True)
        self.assertEqual(report.extractions_inspected, 3)


class TestAffectedPages(unittest.TestCase):
    """affected_pages comes from the wiki_pages.extraction_ids JSON join."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = StateStore(Path(self._tmp.name) / "pages.db")

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_page_with_mixed_pruned_and_survivor(self):
        # Pruned: e1 (lcri=1). Survivor: e2 (lcri=10).
        self.store.save_extraction(_make_extraction("e1"), run_id_int=1)
        self.store.save_extraction(_make_extraction("e2"), run_id_int=10)
        # Page references both ids.
        self.store.save_wiki_page(WikiPageRecord(
            page_path="Patterns/mixed.md",
            page_type="auto",
            title="Mixed",
            extraction_ids=["e1", "e2"],
            word_count=100,
        ))
        # Untouched page references only survivor.
        self.store.save_wiki_page(WikiPageRecord(
            page_path="Patterns/clean.md",
            page_type="auto",
            title="Clean",
            extraction_ids=["e2"],
            word_count=80,
        ))

        report = self.store.prune_stale(5, dry_run=True)
        self.assertEqual(report.affected_pages, ["Patterns/mixed.md"])
        self.assertEqual(report.extractions_deleted, 0)  # dry-run

    def test_no_affected_pages_when_no_extractions_stale(self):
        self.store.save_extraction(_make_extraction("e1"), run_id_int=10)
        self.store.save_wiki_page(WikiPageRecord(
            page_path="Patterns/x.md", page_type="auto", title="X",
            extraction_ids=["e1"], word_count=50,
        ))
        report = self.store.prune_stale(5, dry_run=True)
        self.assertEqual(report.affected_pages, [])


class TestUnfinishedRunDetection(unittest.TestCase):
    """get_unfinished_run() / max_run_id_int() helpers."""

    def test_unfinished_run_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(Path(tmp) / "uf.db")
            try:
                run = PipelineRun(config={})
                store.start_run(run)
                self.assertEqual(store.get_unfinished_run(), run.run_id)

                run.completed_at = "2024-01-01T00:00:00"
                store.finish_run(run)
                self.assertIsNone(store.get_unfinished_run())
            finally:
                store.close()

    def test_max_run_id_int_zero_on_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(Path(tmp) / "max.db")
            try:
                self.assertEqual(store.max_run_id_int(), 0)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
