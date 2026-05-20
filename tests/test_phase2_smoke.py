"""
tests/test_phase2_smoke.py -- Smoke tests for Phase 2 JSONL session processing.

Verifies: new module imports, JSONL parsing on real data, filtering,
classification, extraction, and pipeline integration.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config import PipelineConfig, Paths, SessionConfig
from src.jsonl_parser import (
    ContentBlock,
    SessionFile,
    SessionMessage,
    discover_sessions,
    parse_session,
    _parse_content,
)
from src.prefilter import (
    FilteredMessage,
    FilteredSession,
    filter_session,
    _filter_message,
)
from src.classifier import (
    SessionClassification,
    classify_session,
    _infer_domain,
    _detect_topics,
    _assess_engagement,
)
from src.session_extractor import (
    extract_session,
    _chunk_session,
    _canonicalize_topic,
    _select_topic,
    _extract_title,
    _project_dir_to_name,
    _stable_id,
)
from src.state_store import StateStore


# ---------------------------------------------------------------------------
# JSONL Parser
# ---------------------------------------------------------------------------

class TestContentParsing(unittest.TestCase):
    """Content block parsing handles all formats."""

    def test_parses_string_content(self):
        blocks = _parse_content("Hello, this is a user prompt")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].block_type, "text")
        self.assertIn("Hello", blocks[0].text)

    def test_parses_list_content_with_text(self):
        raw = [{"type": "text", "text": "Some response text"}]
        blocks = _parse_content(raw)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].block_type, "text")

    def test_parses_tool_use_block(self):
        raw = [{"type": "tool_use", "name": "Read", "id": "tu_123", "input": {}}]
        blocks = _parse_content(raw)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].block_type, "tool_use")
        self.assertEqual(blocks[0].tool_name, "Read")

    def test_parses_tool_result_block(self):
        raw = [{"type": "tool_result", "tool_use_id": "tu_123", "content": "file data"}]
        blocks = _parse_content(raw)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].block_type, "tool_result")

    def test_parses_thinking_block(self):
        raw = [{"type": "thinking", "thinking": "Let me reason..."}]
        blocks = _parse_content(raw)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].block_type, "thinking")

    def test_parses_image_block(self):
        raw = [{"type": "image", "source": {"type": "base64", "data": "..."}}]
        blocks = _parse_content(raw)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].block_type, "image")

    def test_parses_mixed_blocks(self):
        raw = [
            {"type": "text", "text": "Here's what I found"},
            {"type": "tool_use", "name": "Grep", "id": "tu_1", "input": {}},
            {"type": "thinking", "thinking": "reasoning..."},
        ]
        blocks = _parse_content(raw)
        self.assertEqual(len(blocks), 3)
        types = [b.block_type for b in blocks]
        self.assertEqual(types, ["text", "tool_use", "thinking"])

    def test_skips_empty_text(self):
        raw = [{"type": "text", "text": "  \n  "}]
        blocks = _parse_content(raw)
        self.assertEqual(len(blocks), 0)

    def test_handles_empty_content(self):
        self.assertEqual(_parse_content(""), [])
        self.assertEqual(_parse_content([]), [])


class TestSessionDiscovery(unittest.TestCase):
    """Session discovery finds real JSONL files."""

    def test_discovers_sessions(self):
        paths = Paths()
        sessions = discover_sessions(paths.project_memory_root)
        self.assertGreater(len(sessions), 0)
        # All should be .jsonl files
        for s in sessions:
            self.assertTrue(s.file_path.name.endswith(".jsonl"))
            self.assertFalse(s.is_subagent)

    def test_returns_empty_for_nonexistent_dir(self):
        sessions = discover_sessions(Path("/nonexistent"))
        self.assertEqual(len(sessions), 0)


class TestSessionParsing(unittest.TestCase):
    """Session parsing streams records from real files."""

    def test_parses_real_session(self):
        paths = Paths()
        sessions = discover_sessions(paths.project_memory_root)
        if not sessions:
            self.skipTest("No sessions found")

        # Pick a session large enough to have actual conversation (>50KB)
        candidates = [s for s in sessions if s.file_size > 50_000]
        if not candidates:
            self.skipTest("No sessions large enough to test")

        target = min(candidates, key=lambda s: s.file_size)
        messages = list(parse_session(target))
        self.assertGreater(len(messages), 0)

        # Should have at least some user/assistant messages
        types = set(m.record_type for m in messages)
        self.assertTrue(types.intersection({"user", "assistant"}))


# ---------------------------------------------------------------------------
# Prefilter
# ---------------------------------------------------------------------------

class TestPrefilter(unittest.TestCase):
    """Prefilter strips noise and credentials."""

    def test_filters_real_session(self):
        paths = Paths()
        sessions = discover_sessions(paths.project_memory_root)
        if not sessions:
            self.skipTest("No sessions found")

        # Pick one that's large enough to have content
        candidates = [s for s in sessions if s.file_size > 10_000]
        if not candidates:
            self.skipTest("No sessions large enough")

        result = filter_session(candidates[0])
        # Result may be None (< 5 messages), but shouldn't crash
        if result:
            self.assertGreater(len(result.messages), 0)
            self.assertGreater(result.total_text_chars, 0)

    def test_filter_message_strips_tool_result(self):
        msg = SessionMessage(
            record_type="user",
            role="user",
            content_blocks=[
                ContentBlock(block_type="tool_result", tool_id="tu_1"),
                ContentBlock(block_type="text", text="Please fix this bug"),
            ],
        )
        filtered = _filter_message(msg)
        self.assertIsNotNone(filtered)
        self.assertIn("fix this bug", filtered.text)
        self.assertEqual(len(filtered.tool_names), 0)

    def test_filter_message_strips_thinking(self):
        msg = SessionMessage(
            record_type="assistant",
            role="assistant",
            content_blocks=[
                ContentBlock(block_type="thinking"),
                ContentBlock(block_type="text", text="Here is the answer"),
            ],
        )
        filtered = _filter_message(msg)
        self.assertIsNotNone(filtered)
        self.assertIn("answer", filtered.text)

    def test_filter_message_keeps_tool_names(self):
        msg = SessionMessage(
            record_type="assistant",
            role="assistant",
            content_blocks=[
                ContentBlock(block_type="text", text="Let me read the file"),
                ContentBlock(block_type="tool_use", tool_name="Read"),
            ],
        )
        filtered = _filter_message(msg)
        self.assertIsNotNone(filtered)
        self.assertIn("Read", filtered.tool_names)

    def test_filter_message_returns_none_for_empty(self):
        msg = SessionMessage(
            record_type="user",
            role="user",
            content_blocks=[
                ContentBlock(block_type="tool_result", tool_id="tu_1"),
            ],
        )
        filtered = _filter_message(msg)
        self.assertIsNone(filtered)

    def test_scrubs_credentials(self):
        msg = SessionMessage(
            record_type="user",
            role="user",
            content_blocks=[
                ContentBlock(
                    block_type="text",
                    text="Use key AKIAIOSFODNN7EXAMPLE please",
                ),
            ],
        )
        filtered = _filter_message(msg)
        self.assertIsNotNone(filtered)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", filtered.text)
        self.assertIn("[REDACTED_AWS_KEY]", filtered.text)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class TestDomainInference(unittest.TestCase):
    """Domain inference from project paths."""

    def test_detects_supabase(self):
        self.assertEqual(
            _infer_domain("-Users-alice-Documents-projects-supabase-dashboard"),
            "supabase",
        )

    def test_detects_devops_from_docker(self):
        self.assertEqual(
            _infer_domain("-Users-alice-Documents-projects-docker-stack"),
            "devops",
        )

    def test_detects_n8n(self):
        self.assertEqual(
            _infer_domain("-Users-alice-Documents-projects-n8n-workflows"),
            "n8n",
        )

    def test_falls_back_to_general(self):
        self.assertEqual(
            _infer_domain("-Users-alice-Documents-SomeUnknownProject"),
            "general",
        )


class TestEngagement(unittest.TestCase):
    """Engagement level assessment."""

    def test_low_engagement(self):
        session = FilteredSession(
            session_id="test", project_dir="test", file_path="test",
            file_size=1000,
            messages=[FilteredMessage(role="user", text="hi")] * 3,
            total_text_chars=100,
        )
        self.assertEqual(_assess_engagement(session), "low")

    def test_high_engagement(self):
        session = FilteredSession(
            session_id="test", project_dir="test", file_path="test",
            file_size=100000,
            messages=[FilteredMessage(role="user", text="x" * 1000)] * 25,
            total_text_chars=60_000,
        )
        self.assertEqual(_assess_engagement(session), "high")


class TestClassification(unittest.TestCase):
    """Full session classification."""

    def test_scores_minimal_session(self):
        session = FilteredSession(
            session_id="test", project_dir="-Users-test-SomeProject",
            file_path="test", file_size=1000,
            messages=[FilteredMessage(role="user", text="hello")] * 6,
            total_text_chars=30,
        )
        c = classify_session(session)
        self.assertGreaterEqual(c.value_score, 1)
        self.assertLessEqual(c.value_score, 5)

    def test_scores_rich_session_higher(self):
        rich_text = (
            "We decided to use a microservices architecture because "
            "the application needs to scale independently. The trade-off "
            "is complexity. We designed the component interface to be "
            "loosely coupled with clear data flow patterns. "
        ) * 50
        session = FilteredSession(
            session_id="test", project_dir="-Users-alice-Documents-projects-bigapp",
            file_path="test", file_size=500000,
            messages=[FilteredMessage(role="user", text=rich_text)] * 30,
            total_text_chars=len(rich_text) * 30,
        )
        c = classify_session(session)
        self.assertGreaterEqual(c.value_score, 4)
        self.assertTrue(c.has_decisions)
        self.assertTrue(c.has_architecture)


# ---------------------------------------------------------------------------
# Session Extractor
# ---------------------------------------------------------------------------

class TestTopicSelection(unittest.TestCase):
    """Topic selection and canonicalization."""

    def test_domain_preferred_for_specific(self):
        c = SessionClassification(
            session_id="test", project_dir="test",
            value_score=5, topics=["api-integration", "backend"],
            domain="n8n",
        )
        topic = _select_topic(c)
        self.assertEqual(topic, "n8n")

    def test_topic_used_for_general_domain(self):
        c = SessionClassification(
            session_id="test", project_dir="test",
            value_score=5, topics=["n8n", "backend"],
            domain="general",
        )
        topic = _select_topic(c)
        self.assertEqual(topic, "n8n")

    def test_canonicalize_known_topics(self):
        self.assertEqual(_canonicalize_topic("api-integration"), "backend")
        self.assertEqual(_canonicalize_topic("n8n"), "n8n")
        self.assertEqual(_canonicalize_topic("supabase"), "supabase")


class TestChunking(unittest.TestCase):
    """Session text chunking at conversation boundaries."""

    def test_chunks_small_session(self):
        messages = [
            FilteredMessage(role="user", text="How do I set up n8n?"),
            FilteredMessage(role="assistant", text="Here's how to set up n8n: " + "x" * 500),
            FilteredMessage(role="user", text="What about webhooks?"),
            FilteredMessage(role="assistant", text="Webhooks work like this: " + "x" * 500),
        ]
        session = FilteredSession(
            session_id="test", project_dir="test", file_path="test",
            file_size=1000, messages=messages, total_text_chars=1200,
        )
        chunks = _chunk_session(session)
        # With small content, it may be 1 or 2 chunks
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 50_001)  # allow small overshoot

    def test_drops_tiny_chunks(self):
        messages = [
            FilteredMessage(role="user", text="hi"),
            FilteredMessage(role="assistant", text="hello"),
        ]
        session = FilteredSession(
            session_id="test", project_dir="test", file_path="test",
            file_size=100, messages=messages, total_text_chars=10,
        )
        chunks = _chunk_session(session)
        # "hi" + "hello" < MIN_CHUNK_CHARS (200), so should be dropped
        self.assertEqual(len(chunks), 0)


class TestExtraction(unittest.TestCase):
    """Full extraction from session to Extraction objects."""

    def test_extracts_from_session(self):
        messages = [
            FilteredMessage(role="user", text="How do I implement OAuth with Okta?"),
            FilteredMessage(role="assistant", text="Here's how to implement OAuth: " + "a" * 400),
            FilteredMessage(role="user", text="What about the callback URL?"),
            FilteredMessage(role="assistant", text="The callback URL should be: " + "b" * 400),
        ]
        session = FilteredSession(
            session_id="test-session-123", project_dir="-Users-alice-Documents-myapp",
            file_path="test", file_size=5000,
            messages=messages, total_text_chars=1000,
        )
        classification = SessionClassification(
            session_id="test-session-123", project_dir="-Users-alice-Documents-myapp",
            value_score=4, topics=["auth", "backend"],
            domain="backend",
        )
        extractions = extract_session(session, classification)
        self.assertGreater(len(extractions), 0)
        for ext in extractions:
            self.assertEqual(ext.source_type, "session")
            self.assertEqual(ext.source_id, "test-session-123")
            self.assertGreater(len(ext.content), 0)

    def test_stable_ids_are_deterministic(self):
        id1 = _stable_id("session-abc", 0)
        id2 = _stable_id("session-abc", 0)
        id3 = _stable_id("session-abc", 1)
        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id3)


class TestProjectNameParsing(unittest.TestCase):
    """Project directory to human name conversion."""

    def test_parses_project_path(self):
        name = _project_dir_to_name(
            "-Users-alice-Documents-projects-myteam-Dashboard"
        )
        self.assertIn("myteam", name)
        self.assertIn("Dashboard", name)

    def test_parses_arbitrary_path(self):
        name = _project_dir_to_name(
            "-Users-alice-Documents-WorkRoot-projectx"
        )
        self.assertIn("WorkRoot", name)


class TestTitleExtraction(unittest.TestCase):
    """Title extraction from chunk text."""

    def test_extracts_from_user_line(self):
        text = "User: How do I fix the authentication bug?\n\nAssistant: Here's the fix..."
        title = _extract_title(text, "abc123", 0)
        self.assertIn("authentication", title)

    def test_truncates_long_titles(self):
        text = "User: " + "x" * 200 + "\n\nAssistant: response"
        title = _extract_title(text, "abc123", 0)
        self.assertLessEqual(len(title), 103)  # 97 + "..."

    def test_fallback_title(self):
        text = "No user prefix here"
        title = _extract_title(text, "abc123", 0)
        self.assertIn("abc123", title)


# ---------------------------------------------------------------------------
# State Store (Phase 2 additions)
# ---------------------------------------------------------------------------

class TestSessionStateStore(unittest.TestCase):
    """State store session tracking."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.store = StateStore(Path(self._tmpdir) / "test.db")

    def tearDown(self):
        self.store.close()

    def test_save_and_check_processed_session(self):
        self.store.save_processed_session(
            session_id="sess-1",
            project_path="test-project",
            file_path="/tmp/test.jsonl",
            file_size=1000,
            message_count=10,
            pipeline_version="2.0.0",
            extraction_ids=["e1", "e2"],
        )
        self.assertTrue(self.store.is_session_processed("sess-1", "2.0.0"))
        self.assertFalse(self.store.is_session_processed("sess-1", "1.0.0"))
        self.assertFalse(self.store.is_session_processed("sess-999", "2.0.0"))

    def test_get_processed_sessions(self):
        self.store.save_processed_session(
            session_id="sess-1",
            project_path="test-project",
            file_path="/tmp/test.jsonl",
            file_size=1000,
            message_count=10,
            pipeline_version="2.0.0",
        )
        sessions = self.store.get_processed_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_id"], "sess-1")


if __name__ == "__main__":
    unittest.main()
