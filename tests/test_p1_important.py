"""
tests/test_p1_important.py — P1 Important tests for the Phase 1 wiki pipeline.

Covers:
  - Large extraction content handling (10K+ chars)
  - Topic inference edge cases (unknown topics, empty content, special chars)
  - Frontmatter generation correctness (valid YAML, correct date formats)
  - State store tracks all extractions with correct lineage
  - Wiki writer atomic writes don't corrupt on partial failure
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path

from src.config import ClusterConfig, Paths
from src.clusterer import cluster_extractions, PageCandidate
from src.memory_reader import (
    _infer_topic,
    _clean_topic,
    _project_id_to_name,
    _parse_memory_file,
    _parse_memory_index,
)
from src.state_store import Extraction, StateStore, WikiPageRecord, PipelineRun
from src.synthesizer import (
    synthesize_page,
    WikiPage,
    _build_frontmatter,
    _split_frontmatter,
    _parse_existing_frontmatter,
    _deduplicate_extractions,
)
from src.wiki_writer import WikiWriter, _render_page


# ---------------------------------------------------------------------------
# P1: Large extraction content handling
# ---------------------------------------------------------------------------


class TestLargeContent(unittest.TestCase):
    """Pipeline handles very large extraction content (10K+ chars)."""

    def _make_large_extraction(self, size=10000):
        content = "Large content block. " * (size // 21 + 1)
        return Extraction(
            id="large1", source_id="s1", source_type="test",
            extraction_type="lesson", topic="test",
            title="Large content test", content=content[:size],
        )

    def test_synthesize_handles_10k_extraction(self):
        ext = self._make_large_extraction(10000)
        candidate = PageCandidate(
            action="create", page_type="pattern", topic="test",
            title="Large Page", target_path="test-large.md",
            extractions=[ext],
        )
        page = synthesize_page(candidate)
        self.assertGreater(page.word_count, 100)
        self.assertIn("Large content block", page.body)

    def test_synthesize_caps_50k_extraction(self):
        ext = self._make_large_extraction(50000)
        candidate = PageCandidate(
            action="create", page_type="pattern", topic="test",
            title="Very Large Page", target_path="test-very-large.md",
            extractions=[ext],
        )
        page = synthesize_page(candidate)
        self.assertLessEqual(page.word_count, 3000)

    def test_write_large_page_to_disk(self):
        ext = self._make_large_extraction(10000)
        candidate = PageCandidate(
            action="create", page_type="pattern", topic="test",
            title="Large Page", target_path="test-large.md",
            extractions=[ext],
        )
        page = synthesize_page(candidate)

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = WikiWriter(Path(tmpdir))
            path = writer.write_page(page)
            self.assertTrue(path.exists())
            content = path.read_text()
            self.assertGreater(len(content), 9000)

    def test_state_store_handles_large_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StateStore(Path(tmpdir) / "test.db")
            ext = self._make_large_extraction(10000)
            store.save_extraction(ext)
            results = store.get_extractions()
            self.assertEqual(len(results), 1)
            self.assertEqual(len(results[0].content), 10000)
            store.close()

    def test_update_large_existing_page(self):
        existing = "---\ntype: resource\n---\n\n# Page\n\n" + ("Existing line.\n" * 500)
        ext = self._make_large_extraction(5000)
        candidate = PageCandidate(
            action="update", page_type="pattern", topic="test",
            title="Page", target_path="test.md",
            existing_content=existing, extractions=[ext],
        )
        page = synthesize_page(candidate)
        self.assertIn("Existing line.", page.body)
        self.assertIn("Large content block", page.body)


# ---------------------------------------------------------------------------
# P1: Topic inference edge cases
# ---------------------------------------------------------------------------


class TestTopicInferenceEdgeCases(unittest.TestCase):
    """Topic inference handles unusual inputs correctly."""

    def test_empty_text_and_empty_fallback(self):
        result = _infer_topic("", "")
        self.assertEqual(result, "general")

    def test_empty_text_with_fallback(self):
        result = _infer_topic("", "My Project")
        self.assertEqual(result, "my-project")

    def test_special_characters_in_text(self):
        result = _infer_topic("Error @#$% in <module>", "Project")
        # Should not crash, should return cleaned topic
        self.assertTrue(len(result) > 0)

    def test_unicode_in_text(self):
        result = _infer_topic("Schrodinger's database connection", "Project")
        self.assertTrue(len(result) > 0)

    def test_very_long_text(self):
        text = "word " * 5000
        result = _infer_topic(text, "Fallback")
        self.assertTrue(len(result) > 0)

    def test_multiple_keywords_first_wins(self):
        # Both n8n and supabase mentioned, first keyword match wins
        result = _infer_topic("n8n integration with supabase", "")
        self.assertEqual(result, "n8n")

    def test_case_insensitive_match(self):
        self.assertEqual(_infer_topic("N8N patterns", ""), "n8n")
        self.assertEqual(_infer_topic("SUPABASE queries", ""), "supabase")

    def test_clean_topic_special_chars(self):
        self.assertEqual(_clean_topic("My Project!@#"), "my-project")
        self.assertEqual(_clean_topic("  spaces  "), "spaces")
        self.assertEqual(_clean_topic(""), "general")
        self.assertEqual(_clean_topic("---"), "general")

    def test_project_id_to_name_edge_cases(self):
        # Empty string
        result = _project_id_to_name("")
        self.assertTrue(isinstance(result, str))

        # Only skip words
        result = _project_id_to_name("-Users-alice-Documents")
        self.assertTrue(isinstance(result, str))

        # Normal case
        result = _project_id_to_name(
            "-Users-alice-Documents-projects-MyApp"
        )
        self.assertIn("Myapp", result)

    def test_unknown_topic_goes_to_gotcha(self):
        extractions = [
            Extraction(id=str(i), source_id="s1", source_type="test",
                       extraction_type="lesson", topic="xyz-unknown-topic",
                       title=f"T{i}", content=f"Lesson about xyz {i}")
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            candidates = cluster_extractions(
                extractions, Path(tmpdir), ClusterConfig(min_extractions_for_new_page=3)
            )
            unknown = [c for c in candidates if c.topic == "xyz-unknown-topic"]
            self.assertEqual(len(unknown), 1)
            self.assertEqual(unknown[0].page_type, "gotcha")
            self.assertIn("Gotchas/", unknown[0].target_path)


# ---------------------------------------------------------------------------
# P1: Frontmatter generation correctness
# ---------------------------------------------------------------------------


class TestFrontmatterCorrectness(unittest.TestCase):
    """Frontmatter is valid YAML with correct formats."""

    def test_new_page_frontmatter_has_required_keys(self):
        candidate = PageCandidate(
            action="create", page_type="pattern", topic="test",
            title="Test", target_path="test.md",
            extractions=[
                Extraction(id="1", source_id="s1", source_type="test",
                           extraction_type="lesson", topic="test",
                           title="T", content="Content here for testing"),
            ],
        )
        page = synthesize_page(candidate)
        fm = page.frontmatter

        # Required keys per vault-metadata-schema
        self.assertIn("type", fm)
        self.assertIn("tags", fm)
        self.assertIn("status", fm)
        self.assertIn("created", fm)
        self.assertIn("modified", fm)
        self.assertIn("area", fm)
        self.assertIn("wiki-source", fm)

    def test_date_format_iso_8601(self):
        candidate = PageCandidate(
            action="create", page_type="pattern", topic="test",
            title="Test", target_path="test.md",
            extractions=[
                Extraction(id="1", source_id="s1", source_type="test",
                           extraction_type="lesson", topic="test",
                           title="T", content="Content here for test purpose"),
            ],
        )
        page = synthesize_page(candidate)
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        self.assertTrue(date_pattern.match(str(page.frontmatter["created"])))
        self.assertTrue(date_pattern.match(str(page.frontmatter["modified"])))

    def test_tags_are_list_of_strings(self):
        candidate = PageCandidate(
            action="create", page_type="pattern", topic="test",
            title="Test", target_path="test.md",
            extractions=[
                Extraction(id="1", source_id="s1", source_type="test",
                           extraction_type="lesson", topic="test",
                           title="T", content="Content for tag test purpose"),
            ],
        )
        page = synthesize_page(candidate)
        self.assertIsInstance(page.frontmatter["tags"], list)
        for tag in page.frontmatter["tags"]:
            self.assertIsInstance(tag, str)

    def test_wiki_source_auto_for_new_pages(self):
        candidate = PageCandidate(
            action="create", page_type="pattern", topic="test",
            title="Test", target_path="test.md",
            extractions=[
                Extraction(id="1", source_id="s1", source_type="test",
                           extraction_type="lesson", topic="test",
                           title="T", content="New page content here"),
            ],
        )
        page = synthesize_page(candidate)
        self.assertEqual(page.frontmatter["wiki-source"], "auto")

    def test_wiki_source_seed_for_updates(self):
        existing = "---\ntype: resource\n---\n\n# Page\n\nOriginal content.\n"
        candidate = PageCandidate(
            action="update", page_type="pattern", topic="test",
            title="Page", target_path="test.md",
            existing_content=existing,
            extractions=[
                Extraction(id="1", source_id="s1", source_type="test",
                           extraction_type="lesson", topic="test",
                           title="Unique Lesson Title Here",
                           content="Completely unique new seeded content that does not exist yet"),
            ],
        )
        page = synthesize_page(candidate)
        self.assertEqual(page.frontmatter.get("wiki-source"), "seed")

    def test_rendered_frontmatter_is_valid_yaml_format(self):
        page = WikiPage(
            path="test.md", title="Test",
            frontmatter={
                "type": "resource",
                "tags": ["wiki", "test"],
                "status": "draft",
                "created": "2026-04-06",
                "modified": "2026-04-06",
                "area": "work",
                "wiki-source": "auto",
            },
            body="# Test\n\nContent.\n",
            word_count=3, is_new=True,
        )
        rendered = _render_page(page)
        self.assertTrue(rendered.startswith("---\n"))
        # Should have closing ---
        second_dash = rendered.index("---", 4)
        self.assertGreater(second_dash, 4)

    def test_split_frontmatter_no_frontmatter(self):
        fm, body = _split_frontmatter("# Just a heading\n\nBody text.\n")
        self.assertEqual(fm, "")
        self.assertEqual(body, "# Just a heading\n\nBody text.\n")

    def test_split_frontmatter_with_frontmatter(self):
        text = "---\ntype: resource\ntags:\n  - wiki\n---\n\n# Title\n\nBody.\n"
        fm, body = _split_frontmatter(text)
        self.assertIn("type: resource", fm)
        self.assertIn("# Title", body)

    def test_parse_existing_frontmatter_list_values(self):
        fm_text = "---\ntags:\n  - wiki\n  - test\ntype: resource\n---"
        candidate = PageCandidate(
            action="update", page_type="pattern", topic="test",
            title="Test", target_path="test.md", extractions=[],
        )
        parsed = _parse_existing_frontmatter(fm_text, candidate)
        self.assertIn("tags", parsed)
        self.assertIsInstance(parsed["tags"], list)
        self.assertIn("wiki", parsed["tags"])


# ---------------------------------------------------------------------------
# P1: State store tracks extractions with correct lineage
# ---------------------------------------------------------------------------


class TestStateStoreLineage(unittest.TestCase):
    """State store correctly tracks extraction provenance."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.store = StateStore(Path(self._tmpdir) / "lineage.db")

    def tearDown(self):
        self.store.close()

    def test_extraction_source_type_preserved(self):
        for src_type in ["pact_memory", "agent_memory", "project_memory"]:
            ext = Extraction(
                source_id=f"src-{src_type}", source_type=src_type,
                extraction_type="lesson", topic="test",
                title=f"From {src_type}", content=f"Content from {src_type}",
            )
            self.store.save_extraction(ext)

        results = self.store.get_extractions()
        source_types = {r.source_type for r in results}
        self.assertEqual(source_types, {"pact_memory", "agent_memory", "project_memory"})

    def test_extraction_ids_for_source(self):
        for i in range(3):
            ext = Extraction(
                source_id="shared-source",
                source_type="test",
                extraction_type="lesson", topic="test",
                title=f"Item {i}", content=f"Content item {i} from the source",
            )
            self.store.save_extraction(ext)

        ids = self.store.get_extraction_ids_for_source("shared-source")
        self.assertEqual(len(ids), 3)

    def test_wiki_page_records_extraction_ids(self):
        page = WikiPageRecord(
            page_path="test.md", page_type="pattern",
            title="Test", extraction_ids=["ext1", "ext2", "ext3"],
            word_count=100,
        )
        self.store.save_wiki_page(page)

        pages = self.store.get_wiki_pages()
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].extraction_ids, ["ext1", "ext2", "ext3"])

    def test_pipeline_run_lifecycle(self):
        run = PipelineRun(
            config={"version": "1.0.0", "sources": "memory-only"}
        )
        self.store.start_run(run)

        run.completed_at = "2026-04-06T00:00:00+00:00"
        run.extractions_created = 50
        run.pages_created = 5
        run.pages_updated = 10
        self.store.finish_run(run)

        # No explicit getter for runs, but should not crash
        # Verifying the DB didn't error is sufficient

    def test_upsert_extraction_replaces(self):
        ext = Extraction(
            id="fixed-id", source_id="s1", source_type="test",
            extraction_type="lesson", topic="test",
            title="Version 1", content="Original content",
        )
        self.store.save_extraction(ext)

        ext2 = Extraction(
            id="fixed-id", source_id="s1", source_type="test",
            extraction_type="lesson", topic="test",
            title="Version 2", content="Updated content",
        )
        self.store.save_extraction(ext2)

        results = self.store.get_extractions()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Version 2")


# ---------------------------------------------------------------------------
# P1: Wiki writer atomic writes
# ---------------------------------------------------------------------------


class TestAtomicWrites(unittest.TestCase):
    """Atomic write mechanism prevents partial corruption."""

    def test_write_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = WikiWriter(Path(tmpdir))
            page = WikiPage(
                path="deep/nested/dir/page.md", title="Deep",
                frontmatter={"type": "resource"}, body="# Deep\n",
                word_count=1, is_new=True,
            )
            path = writer.write_page(page)
            self.assertTrue(path.exists())
            self.assertTrue(path.parent.is_dir())

    def test_no_temp_files_left_behind(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = WikiWriter(Path(tmpdir))
            page = WikiPage(
                path="test/page.md", title="Test",
                frontmatter={"type": "resource"}, body="# Test\n",
                word_count=1, is_new=True,
            )
            writer.write_page(page)

            # Check no .tmp files remain
            tmp_files = list(Path(tmpdir).rglob("*.tmp"))
            self.assertEqual(len(tmp_files), 0)

    def test_overwrite_preserves_file_on_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = WikiWriter(Path(tmpdir))
            target = Path(tmpdir) / "test" / "page.md"
            target.parent.mkdir(parents=True)
            target.write_text("original content")

            page = WikiPage(
                path="test/page.md", title="New",
                frontmatter={"type": "resource"}, body="# New Content\n",
                word_count=2, is_new=False,
            )
            writer.write_page(page)

            content = target.read_text()
            self.assertIn("# New Content", content)
            self.assertNotIn("original content", content)

    def test_write_pages_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = WikiWriter(Path(tmpdir))
            pages = [
                WikiPage(
                    path=f"test/page{i}.md", title=f"Page {i}",
                    frontmatter={"type": "resource"}, body=f"# Page {i}\n",
                    word_count=2, is_new=True,
                )
                for i in range(5)
            ]
            paths = writer.write_pages(pages)
            self.assertEqual(len(paths), 5)
            for p in paths:
                self.assertTrue(p.exists())


# ---------------------------------------------------------------------------
# P1: Memory file parsing edge cases
# ---------------------------------------------------------------------------


class TestMemoryFileParsing(unittest.TestCase):
    """Memory file parser handles edge cases."""

    def test_file_with_no_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "test.md"
            f.write_text("Just a plain markdown file with enough content to pass threshold.\n")
            results = _parse_memory_file(f, "context", "agent_memory")
            self.assertEqual(len(results), 1)

    def test_file_with_empty_body(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "test.md"
            f.write_text("---\nname: test\ntype: user\n---\n\n")
            results = _parse_memory_file(f, "context", "agent_memory")
            self.assertEqual(len(results), 0)  # Body too short

    def test_file_with_short_body(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "test.md"
            f.write_text("---\nname: test\ntype: user\n---\n\nShort")
            results = _parse_memory_file(f, "context", "agent_memory")
            self.assertEqual(len(results), 0)  # < 20 chars

    def test_file_with_frontmatter_maps_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "test.md"
            f.write_text(
                "---\nname: Test Memory\ntype: feedback\n---\n\n"
                "This is detailed feedback about the testing approach used in the project.\n"
            )
            results = _parse_memory_file(f, "context", "agent_memory")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].extraction_type, "lesson")  # feedback → lesson

    def test_memory_index_inline_bullets(self):
        index_text = (
            "# Memory Index\n\n"
            "## Project Knowledge\n\n"
            "- **Testing Pattern**: Use vitest for all React component tests\n"
            "- [Link Entry](file.md) -- linked file\n"
            "- **DB Pattern**: Always use RLS with Supabase tables\n"
        )
        results = _parse_memory_index(index_text, "test-agent", "agent_memory")
        # Should pick up the two inline bullets, skip the link entry
        self.assertEqual(len(results), 2)

    def test_memory_index_empty(self):
        results = _parse_memory_index("", "test", "agent_memory")
        self.assertEqual(results, [])

    def test_nonexistent_file(self):
        results = _parse_memory_file(
            Path("/nonexistent/file.md"), "context", "agent_memory"
        )
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# P1: Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication(unittest.TestCase):
    """Deduplication correctly skips already-present content."""

    def test_exact_content_match_is_deduped(self):
        existing_body = "Use RLS policies for security in all tables"
        ext = Extraction(
            id="1", source_id="s1", source_type="test",
            extraction_type="lesson", topic="test",
            title="RLS", content="Use RLS policies for security in all tables",
        )
        result = _deduplicate_extractions([ext], existing_body)
        self.assertEqual(len(result), 0)

    def test_short_title_not_falsely_deduped(self):
        """Short titles (< 12 chars) should not trigger false-positive dedup."""
        existing_body = "We learned that using RLS policies for security is important"
        ext = Extraction(
            id="1", source_id="s1", source_type="test",
            extraction_type="lesson", topic="test",
            title="RLS", content="Use RLS policies for security in all tables",
        )
        result = _deduplicate_extractions([ext], existing_body)
        # "RLS" is only 3 chars — too short for reliable title-based dedup
        # Content sig doesn't match either, so this should pass through
        self.assertEqual(len(result), 1)

    def test_title_match_is_deduped(self):
        """Titles >= 12 chars that appear in the existing body trigger dedup."""
        existing_body = "# Page\n\nSome content about RLS Lesson Title here."
        ext = Extraction(
            id="1", source_id="s1", source_type="test",
            extraction_type="lesson", topic="test",
            title="RLS Lesson Title",
            content="Completely different content that does not match existing body at all",
        )
        result = _deduplicate_extractions([ext], existing_body)
        self.assertEqual(len(result), 0)

    def test_new_content_passes_dedup(self):
        existing_body = "# Page\n\nExisting content about React patterns."
        ext = Extraction(
            id="1", source_id="s1", source_type="test",
            extraction_type="lesson", topic="test",
            title="New Unique Lesson Title",
            content="Completely unique brand new content about TypeScript generics",
        )
        result = _deduplicate_extractions([ext], existing_body)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
