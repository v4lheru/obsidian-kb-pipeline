"""
tests/test_synthesizer_v1_1.py -- v1.1.0 synthesizer integration tests.

Covers: drift annotation injection below H1, drift disabled => no annotation,
intra-batch dedup on new pages, page-size guard wins over drift annotation,
semantic dedup disabled passes candidates through.
"""

from __future__ import annotations

import unittest
from unittest import mock

from src import drift_check
from src.clusterer import PageCandidate
from src.state_store import Extraction
from src.synthesizer import synthesize_page


def _ext(content: str, ext_id: str = "x", etype: str = "lesson") -> Extraction:
    return Extraction(
        id=ext_id, source_id="s1", source_type="pact_memory",
        extraction_type=etype, topic="t",
        title=f"Title {ext_id}", content=content,
    )


def _update_candidate(existing: str, extractions: list[Extraction]) -> PageCandidate:
    return PageCandidate(
        action="update",
        target_path="Patterns/page.md",
        title="Page",
        topic="t",
        extractions=extractions,
        existing_content=existing,
        page_type="pattern",
    )


def _new_candidate(extractions: list[Extraction]) -> PageCandidate:
    return PageCandidate(
        action="create",
        target_path="Patterns/new-page.md",
        title="New Page",
        topic="t",
        extractions=extractions,
        existing_content="",
        page_type="pattern",
    )


_EXISTING_BODY = (
    "# Page\n\n"
    "## Patterns\n\n"
    "- Use Drizzle for type-safe queries.\n"
    "- Foreign keys should be NOT NULL when possible.\n\n"
    + "Filler paragraph. " * 20
)


class TestDriftAnnotationInsertion(unittest.TestCase):
    def test_finding_appended_as_html_comment(self):
        finding = drift_check.DriftFinding(
            existing_snippet="Use Drizzle",
            new_snippet="Replace with Prisma",
            note="Tool recommendation reversed",
        )
        fake_result = drift_check.DriftResult(findings=[finding], skipped=False)

        cand = _update_candidate(_EXISTING_BODY, [
            _ext("Replace Drizzle with Prisma in new services", "1"),
        ])
        with mock.patch("src.synthesizer.drift_check.check_drift", return_value=fake_result):
            page = synthesize_page(cand, drift_check_enabled=True,
                                   semantic_dedup_enabled=False)

        self.assertIn("<!-- DRIFT", page.body)
        self.assertIn("Tool recommendation reversed", page.body)

    def test_drift_disabled_no_annotation(self):
        cand = _update_candidate(_EXISTING_BODY, [
            _ext("A completely different statement about indexes", "1"),
        ])
        with mock.patch("src.synthesizer.drift_check.check_drift") as mocked:
            synthesize_page(cand, drift_check_enabled=False,
                            semantic_dedup_enabled=False)
        # When disabled at the synthesize_page boundary, drift_check is still
        # called but with enabled=False -- result is skipped, no annotation.
        # Verify by asserting the resulting body has no DRIFT marker.
        with mock.patch(
            "src.synthesizer.drift_check.check_drift",
            return_value=drift_check.DriftResult(skipped=True, skip_reason="disabled"),
        ):
            page = synthesize_page(cand, drift_check_enabled=False,
                                   semantic_dedup_enabled=False)
        self.assertNotIn("<!-- DRIFT", page.body)

    def test_drift_check_called_even_when_disabled_at_module_level(self):
        """drift_check.check_drift handles its own enabled flag; synthesizer
        forwards drift_check_enabled. Ensures the contract is enforced."""
        cand = _update_candidate(_EXISTING_BODY, [
            _ext("Yet another factual statement about Postgres tuning.", "1"),
        ])
        with mock.patch("src.synthesizer.drift_check.check_drift") as mocked:
            mocked.return_value = drift_check.DriftResult(
                skipped=True, skip_reason="disabled",
            )
            synthesize_page(cand, drift_check_enabled=False,
                            semantic_dedup_enabled=False)
            mocked.assert_called_once()
            _, kwargs = mocked.call_args
            self.assertIs(kwargs["enabled"], False)


class TestPageSizeGuardWithDrift(unittest.TestCase):
    def test_page_at_cap_skips_drift_and_extractions(self):
        # Build a body already at the cap so the early-return guard fires.
        huge_body = "# Page\n\n" + ("word " * 3500)
        cand = _update_candidate(huge_body, [_ext("new content", "1")])

        with mock.patch("src.synthesizer.drift_check.check_drift") as mocked:
            page = synthesize_page(cand, max_page_words=3000,
                                   drift_check_enabled=True,
                                   semantic_dedup_enabled=False)
            mocked.assert_not_called()
        self.assertNotIn("<!-- DRIFT", page.body)


class TestSemanticDedupDisabled(unittest.TestCase):
    def test_semantic_dedup_disabled_passes_candidates(self):
        """When semantic_dedup_enabled=False, intra_batch_dedup is not invoked.

        New-page path: 3 distinct extractions all survive to the page body.
        """
        cand = _new_candidate([
            _ext("Statement about indexing strategies for write-heavy tables.", "1", "lesson"),
            _ext("Decision on adopting Drizzle ORM across new services.", "2", "decision"),
            _ext("PaymentGateway component encapsulates Stripe and Adyen.", "3", "entity"),
        ])
        with mock.patch("src.synthesizer.embeddings.intra_batch_dedup") as mocked:
            page = synthesize_page(cand, semantic_dedup_enabled=False,
                                   drift_check_enabled=False)
            mocked.assert_not_called()

        self.assertEqual(len(page.extraction_ids), 3)


if __name__ == "__main__":
    unittest.main()
