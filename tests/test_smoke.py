"""
tests/test_smoke.py — Smoke tests for the Phase 1 wiki pipeline.

Verifies: imports work, modules load, core functions execute without
crashing on real data, and output is valid Obsidian markdown.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import PipelineConfig, Paths
from src.scrubber import scrub_text
from src.state_store import Extraction, StateStore, WikiPageRecord
from src.memory_reader import (
    read_pact_memory,
    read_agent_memory,
    read_project_memory,
    _infer_topic,
    _project_id_to_name,
)
from src.clusterer import cluster_extractions, PageCandidate
from src.synthesizer import synthesize_page, WikiPage
from src.wiki_writer import WikiWriter


class TestScrubber(unittest.TestCase):
    """Credential scrubbing catches known patterns."""

    def test_scrubs_aws_key(self):
        text = "Use key AKIAIOSFODNN7EXAMPLE to connect"
        result = scrub_text(text)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", result)
        self.assertIn("[REDACTED_AWS_KEY]", result)

    def test_scrubs_anthropic_key(self):
        text = "export ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz"
        result = scrub_text(text)
        self.assertNotIn("sk-ant-", result)

    def test_scrubs_github_token(self):
        text = "token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"
        result = scrub_text(text)
        self.assertNotIn("ghp_", result)

    def test_preserves_normal_text(self):
        text = "Use supabase RLS policies for row-level security"
        result = scrub_text(text)
        self.assertEqual(text, result)


class TestStateStore(unittest.TestCase):
    """State store CRUD operations work."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.store = StateStore(Path(self._tmpdir) / "test.db")

    def tearDown(self):
        self.store.close()

    def test_save_and_retrieve_extraction(self):
        ext = Extraction(
            id="test-1", source_id="mem-1", source_type="pact_memory",
            extraction_type="lesson", topic="supabase",
            title="Test lesson", content="Use RLS policies",
            confidence=0.9, context="Test Project",
        )
        self.store.save_extraction(ext)
        results = self.store.get_extractions()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].topic, "supabase")

    def test_is_source_processed(self):
        ext = Extraction(
            id="test-2", source_id="mem-2", source_type="agent_memory",
            extraction_type="entity", topic="n8n",
            title="Test entity", content="n8n webhook node",
        )
        self.store.save_extraction(ext)
        self.assertTrue(self.store.is_source_processed("mem-2"))
        self.assertFalse(self.store.is_source_processed("mem-999"))

    def test_save_wiki_page(self):
        page = WikiPageRecord(
            page_path="Patterns/test.md", page_type="pattern",
            title="Test Page", extraction_ids=["e1", "e2"],
            word_count=100,
        )
        self.store.save_wiki_page(page)
        pages = self.store.get_wiki_pages()
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].extraction_ids, ["e1", "e2"])


class TestTopicInference(unittest.TestCase):
    """Topic inference from text works."""

    def test_detects_n8n(self):
        self.assertEqual(_infer_topic("n8n webhook patterns", ""), "n8n")

    def test_detects_supabase(self):
        self.assertEqual(_infer_topic("Supabase RLS policy", ""), "supabase")

    def test_falls_back_to_context(self):
        self.assertEqual(_infer_topic("some generic text", "My Project"), "my-project")


class TestProjectNameParsing(unittest.TestCase):
    """Project ID to human name conversion."""

    def test_parses_project_dashboard(self):
        name = _project_id_to_name(
            "-Users-alice-Documents-projects-myteam---dashboard"
        )
        self.assertIn("Myteam", name)
        self.assertIn("Dashboard", name)


class TestMemoryReaders(unittest.TestCase):
    """Memory readers produce extractions from real data."""

    def test_pact_memory_reads(self):
        paths = Paths()
        if not paths.pact_memory_db.exists():
            self.skipTest("pact-memory DB not found")
        extractions = read_pact_memory(paths)
        self.assertGreater(len(extractions), 0)

    def test_agent_memory_reads(self):
        paths = Paths()
        if not paths.agent_memory_root.exists():
            self.skipTest("agent-memory root not found")
        extractions = read_agent_memory(paths)
        self.assertGreater(len(extractions), 0)

    def test_project_memory_reads(self):
        paths = Paths()
        if not paths.project_memory_root.exists():
            self.skipTest("project-memory root not found")
        extractions = read_project_memory(paths)
        self.assertGreater(len(extractions), 0)


class TestClusterer(unittest.TestCase):
    """Clusterer groups extractions and produces page candidates."""

    def test_groups_by_topic(self):
        extractions = [
            Extraction(id="1", source_id="s1", source_type="test",
                       extraction_type="lesson", topic="n8n",
                       title="T1", content="n8n pattern 1"),
            Extraction(id="2", source_id="s1", source_type="test",
                       extraction_type="lesson", topic="n8n",
                       title="T2", content="n8n pattern 2"),
            Extraction(id="3", source_id="s1", source_type="test",
                       extraction_type="lesson", topic="n8n",
                       title="T3", content="n8n pattern 3"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.config import ClusterConfig
            candidates = cluster_extractions(
                extractions, Path(tmpdir), ClusterConfig()
            )
            # All 3 should cluster into one n8n page candidate
            n8n_candidates = [c for c in candidates if c.topic == "n8n"]
            self.assertEqual(len(n8n_candidates), 1)
            self.assertEqual(len(n8n_candidates[0].extractions), 3)

    def test_threshold_blocks_small_groups(self):
        extractions = [
            Extraction(id="1", source_id="s1", source_type="test",
                       extraction_type="lesson", topic="obscure-topic",
                       title="T1", content="only one extraction"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.config import ClusterConfig
            candidates = cluster_extractions(
                extractions, Path(tmpdir), ClusterConfig(min_extractions_for_new_page=3)
            )
            obscure = [c for c in candidates if c.topic == "obscure-topic"]
            self.assertEqual(len(obscure), 0)


class TestSynthesizer(unittest.TestCase):
    """Synthesizer produces valid wiki markdown."""

    def test_new_page_has_frontmatter_markers(self):
        candidate = PageCandidate(
            action="create", page_type="gotcha", topic="test",
            title="Test Gotchas", target_path="Gotchas/gotcha-test.md",
            extractions=[
                Extraction(id="1", source_id="s1", source_type="test",
                           extraction_type="lesson", topic="test",
                           title="Test lesson", content="Always check nulls"),
            ],
        )
        page = synthesize_page(candidate)
        self.assertIn("wiki-source", str(page.frontmatter))
        self.assertEqual(page.frontmatter["wiki-source"], "auto")
        self.assertTrue(page.body.startswith("# "))

    def test_update_preserves_existing(self):
        existing = "---\ntype: resource\n---\n\n# Existing Page\n\nOriginal content here.\n"
        candidate = PageCandidate(
            action="update", page_type="pattern", topic="test",
            title="Existing Page", target_path="test.md",
            existing_content=existing,
            extractions=[
                Extraction(id="1", source_id="s1", source_type="test",
                           extraction_type="lesson", topic="test",
                           title="New lesson", content="Brand new knowledge added here"),
            ],
        )
        page = synthesize_page(candidate)
        self.assertIn("Original content here", page.body)
        self.assertIn("Brand new knowledge", page.body)


class TestWikiWriter(unittest.TestCase):
    """Wiki writer produces valid files."""

    def test_writes_valid_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = WikiWriter(Path(tmpdir))
            page = WikiPage(
                path="test/page.md", title="Test Page",
                frontmatter={"type": "resource", "tags": ["wiki", "test"],
                             "status": "draft", "created": "2026-04-06",
                             "modified": "2026-04-06", "area": "work"},
                body="# Test Page\n\nSome content.\n",
                word_count=4, is_new=True,
            )
            path = writer.write_page(page)
            self.assertTrue(path.exists())
            content = path.read_text()
            self.assertTrue(content.startswith("---\n"))
            self.assertIn("type: resource", content)
            self.assertIn("# Test Page", content)


if __name__ == "__main__":
    unittest.main()
