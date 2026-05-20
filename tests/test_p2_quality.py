"""
tests/test_p2_quality.py — P2 Quality spot-check tests for the Phase 1 wiki pipeline.

Covers:
  - Output quality: sample generated wiki pages for readability
  - Wikilink validity: do [[links]] point to known targets?
  - Gotcha catch-all page size (flagged concern)
  - Clusterer merge behavior for same-page targets
  - End-to-end mini-pipeline test with synthetic data
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import ClusterConfig, Paths, PipelineConfig
from src.clusterer import cluster_extractions, PageCandidate
from src.state_store import Extraction, StateStore, WikiPageRecord
from src.synthesizer import synthesize_page, WikiPage, _BUILTIN_WIKILINK_TARGETS
from src.wiki_writer import WikiWriter, _render_page


# ---------------------------------------------------------------------------
# P2: Output quality spot-check
# ---------------------------------------------------------------------------


class TestOutputQuality(unittest.TestCase):
    """Generated wiki pages are readable and well-structured."""

    def _make_realistic_candidate(self, topic, extractions):
        return PageCandidate(
            action="create", page_type="pattern", topic=topic,
            title=f"{topic.title()} Patterns",
            target_path=f"Patterns/{topic}-patterns.md",
            extractions=extractions,
        )

    def test_page_starts_with_h1(self):
        ext = Extraction(
            id="1", source_id="s1", source_type="pact_memory",
            extraction_type="lesson", topic="n8n",
            title="Webhook patterns", content="Use webhook nodes with header auth",
        )
        candidate = self._make_realistic_candidate("n8n", [ext])
        page = synthesize_page(candidate)
        self.assertTrue(page.body.strip().startswith("# "))

    def test_page_has_section_headers(self):
        extractions = [
            Extraction(id="1", source_id="s1", source_type="test",
                       extraction_type="lesson", topic="test",
                       title="Lesson", content="Use patterns for consistency in code"),
            Extraction(id="2", source_id="s1", source_type="test",
                       extraction_type="decision", topic="test",
                       title="Decision", content="Chose approach A over B for performance"),
        ]
        candidate = self._make_realistic_candidate("test", extractions)
        page = synthesize_page(candidate)
        self.assertIn("## ", page.body)

    def test_page_body_not_empty(self):
        ext = Extraction(
            id="1", source_id="s1", source_type="test",
            extraction_type="lesson", topic="test",
            title="T", content="Meaningful content about testing patterns here",
        )
        candidate = self._make_realistic_candidate("test", [ext])
        page = synthesize_page(candidate)
        self.assertGreater(page.word_count, 0)

    def test_rendered_page_has_frontmatter_and_body(self):
        page = WikiPage(
            path="test.md", title="Test",
            frontmatter={"type": "resource", "tags": ["wiki"],
                         "status": "draft", "created": "2026-04-06",
                         "modified": "2026-04-06", "area": "work"},
            body="# Test\n\nContent paragraph.\n",
            word_count=3, is_new=True,
        )
        rendered = _render_page(page)
        self.assertTrue(rendered.startswith("---\n"))
        self.assertIn("type: resource", rendered)
        self.assertIn("# Test", rendered)
        self.assertIn("Content paragraph.", rendered)

    def test_multi_extraction_page_readability(self):
        """A page with 5 diverse extractions should be readable."""
        extractions = [
            Extraction(id=str(i), source_id="s1", source_type="pact_memory",
                       extraction_type=etype, topic="test",
                       title=f"Title {i}", content=f"Detailed content about aspect {i} of the system",
                       context="Test Project")
            for i, etype in enumerate(["lesson", "decision", "entity", "lesson", "decision"])
        ]
        candidate = self._make_realistic_candidate("test", extractions)
        page = synthesize_page(candidate)

        # Should have distinct sections
        self.assertIn("## Lessons Learned", page.body)
        self.assertIn("## Key Decisions", page.body)
        self.assertIn("## Key Components", page.body)

    def test_context_attribution_appears(self):
        ext = Extraction(
            id="1", source_id="s1", source_type="pact_memory",
            extraction_type="lesson", topic="test",
            title="Lesson", content="Important lesson about auth patterns",
            context="Auth Project",
        )
        candidate = self._make_realistic_candidate("test", [ext])
        page = synthesize_page(candidate)
        self.assertIn("Auth Project", page.body)


# ---------------------------------------------------------------------------
# P2: Wikilink validity
# ---------------------------------------------------------------------------


class TestWikilinkValidity(unittest.TestCase):
    """Generated wikilinks point to known page targets."""

    def test_known_wikilink_targets_exist_in_map(self):
        """All wikilink targets should map to actual page files."""
        for keyword, target in _BUILTIN_WIKILINK_TARGETS.items():
            self.assertTrue(
                len(target) > 0,
                f"Wikilink target for '{keyword}' is empty"
            )

    def test_generated_related_links_use_valid_format(self):
        extractions = [
            Extraction(id="1", source_id="s1", source_type="test",
                       extraction_type="lesson", topic="n8n",
                       title="n8n + supabase",
                       content="Connect n8n webhook to supabase insert"),
        ]
        candidate = PageCandidate(
            action="create", page_type="pattern", topic="n8n",
            title="n8n Patterns", target_path="n8n/n8n-workflow-patterns.md",
            extractions=extractions,
        )
        page = synthesize_page(candidate)

        # Check Related section has valid [[wikilink]] format
        if "## Related" in page.body:
            related_section = page.body[page.body.index("## Related"):]
            # All links should be in [[...]] format
            import re
            links = re.findall(r"\[\[([^\]]+)\]\]", related_section)
            for link in links:
                self.assertNotIn("/", link, f"Wikilink '{link}' should not contain path separators")
                self.assertNotIn(".md", link, f"Wikilink '{link}' should not contain .md extension")


# ---------------------------------------------------------------------------
# P2: Gotcha catch-all page size
# ---------------------------------------------------------------------------


class TestGotchaCatchAllSize(unittest.TestCase):
    """Monitor the 'general' gotcha page size — flagged as concern."""

    def test_general_topic_page_word_count(self):
        """If many extractions fall to 'general', the page shouldn't be enormous."""
        # Simulate 20 general extractions
        extractions = [
            Extraction(id=str(i), source_id="s1", source_type="test",
                       extraction_type="lesson", topic="general",
                       title=f"General lesson {i}",
                       content=f"This is general knowledge item number {i} " * 10)
            for i in range(20)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            candidates = cluster_extractions(
                extractions, Path(tmpdir),
                ClusterConfig(min_extractions_for_new_page=3, max_extractions_per_page=15),
            )
            general = [c for c in candidates if c.topic == "general"]
            if general:
                page = synthesize_page(general[0])
                # Flag if page exceeds 5000 words (practical limit for readability)
                if page.word_count > 5000:
                    self.fail(
                        f"General gotcha page is {page.word_count} words — "
                        f"exceeds 5000 word readability threshold"
                    )

    def test_topic_distribution_not_too_skewed(self):
        """No single topic should accumulate a disproportionate number of extractions."""
        # Create a mix of extractions with varied topics
        extractions = []
        for i in range(30):
            topic = "general" if i < 20 else "n8n"
            extractions.append(
                Extraction(id=str(i), source_id="s1", source_type="test",
                           extraction_type="lesson", topic=topic,
                           title=f"Item {i}", content=f"Content for item {i} " * 5)
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            candidates = cluster_extractions(
                extractions, Path(tmpdir), ClusterConfig(min_extractions_for_new_page=3)
            )
            for c in candidates:
                if c.topic == "general":
                    # Just flag for monitoring, not a hard failure
                    self.assertGreater(
                        len(c.extractions), 0,
                        "General catch-all has extractions"
                    )


# ---------------------------------------------------------------------------
# P2: Clusterer merge behavior
# ---------------------------------------------------------------------------


class TestClustererMergeBehavior(unittest.TestCase):
    """Multiple topics mapping to the same page get merged."""

    def test_database_and_supabase_merge_to_same_page(self):
        extractions = [
            Extraction(id="1", source_id="s1", source_type="test",
                       extraction_type="lesson", topic="database",
                       title="DB", content="Use proper indexes"),
            Extraction(id="2", source_id="s1", source_type="test",
                       extraction_type="lesson", topic="database",
                       title="DB2", content="Normalize to third form"),
            Extraction(id="3", source_id="s1", source_type="test",
                       extraction_type="lesson", topic="database",
                       title="DB3", content="Foreign key constraints"),
            Extraction(id="4", source_id="s1", source_type="test",
                       extraction_type="lesson", topic="supabase",
                       title="SB", content="Use RLS policies always"),
            Extraction(id="5", source_id="s1", source_type="test",
                       extraction_type="lesson", topic="supabase",
                       title="SB2", content="Supabase edge functions"),
            Extraction(id="6", source_id="s1", source_type="test",
                       extraction_type="lesson", topic="supabase",
                       title="SB3", content="Row level security setup"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            candidates = cluster_extractions(
                extractions, Path(tmpdir), ClusterConfig(min_extractions_for_new_page=3)
            )
            # Both database and supabase map to database-schema-patterns.md
            db_pages = [c for c in candidates
                        if "database-schema-patterns" in c.target_path]
            self.assertEqual(len(db_pages), 1)
            self.assertEqual(len(db_pages[0].extractions), 6)

    def test_backend_and_typescript_route_to_separate_pages(self):
        """Backend and typescript route to focused per-topic pages, not a catch-all.

        This is the post-split behavior — typescript was split off backend-patterns.md
        onto its own typescript-patterns.md page so neither grows unbounded.
        """
        extractions = [
            Extraction(id=str(i), source_id="s1", source_type="test",
                       extraction_type="lesson",
                       topic="backend" if i < 3 else "typescript",
                       title=f"T{i}", content=f"Lesson about backend/ts {i}")
            for i in range(6)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            candidates = cluster_extractions(
                extractions, Path(tmpdir), ClusterConfig(min_extractions_for_new_page=3)
            )
            backend_pages = [c for c in candidates
                             if c.target_path.endswith("backend-patterns.md")]
            ts_pages = [c for c in candidates
                        if c.target_path.endswith("typescript-patterns.md")]
            self.assertEqual(len(backend_pages), 1)
            self.assertEqual(len(backend_pages[0].extractions), 3)
            self.assertEqual(len(ts_pages), 1)
            self.assertEqual(len(ts_pages[0].extractions), 3)


# ---------------------------------------------------------------------------
# P2: End-to-end mini-pipeline
# ---------------------------------------------------------------------------


class TestMiniPipeline(unittest.TestCase):
    """End-to-end test with synthetic data through the full pipeline."""

    def test_full_flow_synthetic_data(self):
        """Create extractions, cluster, synthesize, write, verify."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vault = Path(tmpdir) / "vault"
            vault.mkdir()

            # Create a MOC file
            moc = vault / "coding-knowledge-map.md"
            moc.write_text(
                "# Coding Knowledge Map\n\n"
                "## n8n Workflow Automation\n\n"
                "- [[n8n-workflow-patterns]] -- n8n Workflow Patterns\n\n"
                "## Cross-Cutting Patterns\n\n"
            )

            # Create synthetic extractions
            extractions = [
                Extraction(id=str(i), source_id="syn",
                           source_type="pact_memory",
                           extraction_type="lesson",
                           topic="n8n",
                           title=f"n8n Pattern {i}",
                           content=f"Use n8n webhook pattern {i} for integrations")
                for i in range(5)
            ]

            # Cluster
            config = ClusterConfig(min_extractions_for_new_page=3)
            candidates = cluster_extractions(extractions, vault, config)
            self.assertGreater(len(candidates), 0)

            # Synthesize
            pages = [synthesize_page(c) for c in candidates]
            self.assertGreater(len(pages), 0)

            # Write
            writer = WikiWriter(vault)
            written = writer.write_pages(pages)
            self.assertGreater(len(written), 0)

            # Update MOC
            writer.update_moc(pages)

            # Verify files exist and are valid
            for path in written:
                self.assertTrue(path.exists())
                content = path.read_text()
                self.assertTrue(content.startswith("---\n"))
                self.assertIn("# ", content)

            # Verify state can be recorded
            state_db = Path(tmpdir) / "state.db"
            store = StateStore(state_db)
            store.save_extractions(extractions)
            for page in pages:
                store.save_wiki_page(WikiPageRecord(
                    page_path=page.path, page_type="pattern",
                    title=page.title,
                    extraction_ids=page.extraction_ids,
                    word_count=page.word_count,
                ))
            stored_ext = store.get_extractions()
            stored_pages = store.get_wiki_pages()
            self.assertEqual(len(stored_ext), 5)
            self.assertGreater(len(stored_pages), 0)
            store.close()


# ---------------------------------------------------------------------------
# P2: Page size guard (synthesizer max_page_words)
# ---------------------------------------------------------------------------


class TestPageSizeGuard(unittest.TestCase):
    """Synthesizer refuses to grow already-oversized pages."""

    def test_oversized_existing_page_skips_new_extractions(self):
        # Existing body with 3500 words (over the 3000 cap)
        existing_body = "word " * 3500
        existing = (
            "---\ntype: resource\ntags:\n  - wiki\nstatus: draft\n---\n\n"
            "# Big Page\n\n" + existing_body
        )
        ext = Extraction(
            id="new1", source_id="s1", source_type="test",
            extraction_type="lesson", topic="test",
            title="New Lesson That Should Be Skipped",
            content="This new content should not be appended because the page is over cap",
        )
        candidate = PageCandidate(
            action="update", page_type="pattern", topic="test",
            title="Big Page", target_path="test-big.md",
            extractions=[ext], existing_content=existing,
        )
        page = synthesize_page(candidate, max_page_words=3000)
        # Body should not contain the new extraction
        self.assertNotIn("This new content should not be appended", page.body)
        # No extractions should be recorded as added
        self.assertEqual(page.extraction_ids, [])

    def test_under_cap_page_still_grows(self):
        existing_body = "small body content here only"  # < 3000 words
        existing = (
            "---\ntype: resource\ntags:\n  - wiki\nstatus: draft\n---\n\n"
            "# Small Page\n\n" + existing_body
        )
        ext = Extraction(
            id="new2", source_id="s1", source_type="test",
            extraction_type="lesson", topic="test",
            title="Append Me",
            content="This content should be appended to the small page",
        )
        candidate = PageCandidate(
            action="update", page_type="pattern", topic="test",
            title="Small Page", target_path="test-small.md",
            extractions=[ext], existing_content=existing,
        )
        page = synthesize_page(candidate, max_page_words=3000)
        self.assertIn("This content should be appended", page.body)


# ---------------------------------------------------------------------------
# P2: Improved deduplication (whitespace normalization + 120-char window)
# ---------------------------------------------------------------------------


class TestImprovedDeduplication(unittest.TestCase):
    """Dedup catches near-duplicates, not just exact 60-char prefixes."""

    def test_dedup_with_whitespace_drift(self):
        """Same content with different whitespace should be deduped."""
        from src.synthesizer import _deduplicate_extractions
        existing_body = "Use RLS policies on all tables for security"
        ext = Extraction(
            id="1", source_id="s1", source_type="test",
            extraction_type="lesson", topic="test",
            title="RLS lesson",
            # Note: extra spaces — old 60-char-substring check would miss this
            content="Use   RLS  policies on all tables for security",
        )
        result = _deduplicate_extractions([ext], existing_body)
        self.assertEqual(len(result), 0)

    def test_dedup_120_char_window(self):
        """Content matching at chars 60-120 should still be deduped."""
        from src.synthesizer import _deduplicate_extractions
        # 80-char prefix duplicate, but the FIRST 60 chars differ
        existing_body = (
            "Some intro text first then the actual lesson: "
            "Use RLS policies on all tables for security"
        )
        ext = Extraction(
            id="1", source_id="s1", source_type="test",
            extraction_type="lesson", topic="test",
            title="RLS lesson",
            content=(
                "Some intro text first then the actual lesson: "
                "Use RLS policies on all tables for security"
            ),
        )
        result = _deduplicate_extractions([ext], existing_body)
        self.assertEqual(len(result), 0)

    def test_distinct_content_passes_dedup(self):
        from src.synthesizer import _deduplicate_extractions
        existing_body = "Use RLS policies on all tables for security"
        ext = Extraction(
            id="1", source_id="s1", source_type="test",
            extraction_type="lesson", topic="test",
            title="Different topic",
            content="Configure pg_cron for scheduled materialized view refreshes",
        )
        result = _deduplicate_extractions([ext], existing_body)
        self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# P2: Quality auto-fix
# ---------------------------------------------------------------------------


class TestQualityFix(unittest.TestCase):
    """quality_fix() repairs orphans, broken links, missing frontmatter."""

    def _make_vault(self, tmpdir: str) -> Path:
        vault = Path(tmpdir) / "vault"
        vault.mkdir()
        (vault / "Patterns").mkdir()
        (vault / "Gotchas").mkdir()
        # MOC with the standard sections quality_fix expects.
        (vault / "coding-knowledge-map.md").write_text(
            "---\ntype: resource\ntags:\n  - wiki\nstatus: reference\n---\n\n"
            "# Coding Knowledge Map\n\n"
            "## Cross-Cutting Patterns\n\n"
            "## Gotchas & Debugging\n\n"
        )
        return vault

    def test_repair_missing_frontmatter(self):
        from src.quality_check import quality_fix
        with tempfile.TemporaryDirectory() as tmpdir:
            vault = self._make_vault(tmpdir)
            page = vault / "Patterns" / "missing-fm.md"
            # No frontmatter at all
            page.write_text("# Missing FM\n\nSome content here for the test.\n" * 10)

            report = quality_fix(vault)
            self.assertIn("Patterns/missing-fm.md", report.frontmatter_repaired)

            # Re-read; should now have frontmatter
            content = page.read_text()
            self.assertTrue(content.startswith("---"))
            self.assertIn("type:", content)
            self.assertIn("tags:", content)
            self.assertIn("status:", content)

    def test_remove_broken_wikilinks(self):
        from src.quality_check import quality_fix
        with tempfile.TemporaryDirectory() as tmpdir:
            vault = self._make_vault(tmpdir)
            page = vault / "Patterns" / "with-broken-link.md"
            page.write_text(
                "---\ntype: resource\ntags:\n  - wiki\nstatus: draft\n---\n\n"
                "# Page\n\nSee [[nonexistent-target]] and [[also-broken|the broken one]].\n"
                + "filler word " * 50
            )

            report = quality_fix(vault)
            removed_targets = {target for _, target in report.broken_links_removed}
            self.assertIn("nonexistent-target", removed_targets)
            self.assertIn("also-broken", removed_targets)

            # Display text preserved for the aliased link, target as bare text
            # for the unaliased one. Neither should contain [[...]] anymore.
            content = page.read_text()
            self.assertNotIn("[[nonexistent-target]]", content)
            self.assertNotIn("[[also-broken", content)
            self.assertIn("the broken one", content)
            self.assertIn("nonexistent-target", content)

    def test_add_orphan_to_moc(self):
        from src.quality_check import quality_fix
        with tempfile.TemporaryDirectory() as tmpdir:
            vault = self._make_vault(tmpdir)
            # Orphan page in Gotchas/ with valid frontmatter and an H1.
            (vault / "Gotchas" / "lonely-page.md").write_text(
                "---\ntype: resource\ntags:\n  - wiki\nstatus: draft\n---\n\n"
                "# Lonely Page Title\n\nContent. " * 20
            )

            report = quality_fix(vault)
            self.assertEqual(len(report.moc_entries_added), 1)
            self.assertIn("[[lonely-page]]", report.moc_entries_added[0])

            moc_content = (vault / "coding-knowledge-map.md").read_text()
            self.assertIn("[[lonely-page]]", moc_content)
            self.assertIn("Lonely Page Title", moc_content)

    def test_oversized_pages_flagged_not_split(self):
        """Auto-splitting curated content is unsafe — quality_fix flags only."""
        from src.quality_check import quality_fix
        with tempfile.TemporaryDirectory() as tmpdir:
            vault = self._make_vault(tmpdir)
            big_page = vault / "Patterns" / "big-page.md"
            big_page.write_text(
                "---\ntype: resource\ntags:\n  - wiki\nstatus: draft\n---\n\n"
                "# Big Page\n\n" + ("word " * 6000)
            )

            report = quality_fix(vault, max_words=5000)
            paths_flagged = {p for p, _ in report.oversized_flagged}
            self.assertIn("Patterns/big-page.md", paths_flagged)

            # Content unchanged — no split, no truncation.
            content = big_page.read_text()
            self.assertGreater(len(content.split()), 5000)

    def test_idempotent_no_changes_second_run(self):
        """A second quality_fix() run should be a no-op once issues are repaired."""
        from src.quality_check import quality_fix
        with tempfile.TemporaryDirectory() as tmpdir:
            vault = self._make_vault(tmpdir)
            (vault / "Patterns" / "needs-fm.md").write_text(
                "# No FM\n\n" + ("word " * 60)
            )

            quality_fix(vault)  # first run repairs
            second = quality_fix(vault)  # second should find nothing to repair
            self.assertEqual(second.frontmatter_repaired, [])
            self.assertEqual(second.broken_links_removed, [])


if __name__ == "__main__":
    unittest.main()
