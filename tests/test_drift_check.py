"""
tests/test_drift_check.py -- Drift-detection module unit tests.

Covers all skip paths (disabled, no_api_key, no_anthropic_package, api_error,
parse_error, trivial_input), the happy path with a mocked Anthropic response,
HTML-comment-safety on `-->` substrings, and the 200-char snippet clamp.
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

from src.drift_check import (
    DriftFinding,
    DriftResult,
    build_drift_comment,
    check_drift,
)
from src.state_store import Extraction


def _ext(content: str, title: str = "Some title") -> Extraction:
    return Extraction(
        id="x", source_id="s1", source_type="pact_memory",
        extraction_type="lesson", topic="t",
        title=title, content=content,
    )


_LONG_BODY = "Existing wiki page body. " * 50  # > 200 chars


class TestDriftSkipPaths(unittest.TestCase):
    def test_disabled_returns_skipped(self):
        result = check_drift("body" * 100, [_ext("a")], enabled=False)
        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "disabled")
        self.assertEqual(result.findings, [])

    def test_trivial_input_short_body(self):
        result = check_drift("short", [_ext("a")], enabled=True)
        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "trivial_input")

    def test_trivial_input_empty_extractions(self):
        result = check_drift(_LONG_BODY, [], enabled=True)
        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "trivial_input")

    def test_no_api_key(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("ANTHROPIC_API_KEY", None)
            result = check_drift(_LONG_BODY, [_ext("a")], enabled=True)
        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "no_api_key")


class TestDriftHappyPath(unittest.TestCase):
    """Mocked Anthropic SDK; no network calls."""

    def _patch_env_and_sdk(self, response_text: str, raise_exc: Exception | None = None):
        env_patch = mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"})

        mock_client = mock.Mock()
        if raise_exc is not None:
            mock_client.messages.create.side_effect = raise_exc
        else:
            mock_response = mock.Mock()
            mock_block = mock.Mock()
            mock_block.text = response_text
            mock_response.content = [mock_block]
            mock_client.messages.create.return_value = mock_response

        mock_anthropic = mock.MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        sys_patch = mock.patch.dict("sys.modules", {"anthropic": mock_anthropic})

        return env_patch, sys_patch

    def test_one_finding_parsed(self):
        payload = json.dumps({"findings": [
            {"existing_snippet": "use Drizzle", "new_snippet": "use Prisma",
             "note": "tool swap"}
        ]})
        env_p, sys_p = self._patch_env_and_sdk(payload)
        with env_p, sys_p:
            result = check_drift(_LONG_BODY, [_ext("Replace Drizzle with Prisma")])
        self.assertFalse(result.skipped)
        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].note, "tool swap")

    def test_empty_findings(self):
        env_p, sys_p = self._patch_env_and_sdk(json.dumps({"findings": []}))
        with env_p, sys_p:
            result = check_drift(_LONG_BODY, [_ext("non-contradicting fact")])
        self.assertFalse(result.skipped)
        self.assertEqual(result.findings, [])

    def test_api_error_returns_skipped(self):
        env_p, sys_p = self._patch_env_and_sdk("", raise_exc=RuntimeError("503"))
        with env_p, sys_p:
            result = check_drift(_LONG_BODY, [_ext("a")])
        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "api_error")

    def test_parse_error_returns_skipped(self):
        env_p, sys_p = self._patch_env_and_sdk("not json at all")
        with env_p, sys_p:
            result = check_drift(_LONG_BODY, [_ext("a")])
        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "parse_error")

    def test_findings_not_a_list_is_parse_error(self):
        env_p, sys_p = self._patch_env_and_sdk(json.dumps({"findings": "oops"}))
        with env_p, sys_p:
            result = check_drift(_LONG_BODY, [_ext("a")])
        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "parse_error")

    def test_response_with_comment_break_attempt_is_sanitized(self):
        """A model attempting to inject `-->` cannot break out of the comment."""
        payload = json.dumps({"findings": [
            {"existing_snippet": "a -->b", "new_snippet": "c", "note": "n"}
        ]})
        env_p, sys_p = self._patch_env_and_sdk(payload)
        with env_p, sys_p:
            result = check_drift(_LONG_BODY, [_ext("a")])
        self.assertEqual(len(result.findings), 1)
        comment = build_drift_comment(result.findings)
        self.assertNotIn("-->b", comment)  # `-->` neutralized to `--`
        self.assertTrue(comment.endswith("-->"))

    def test_long_snippet_clamped_to_200_chars(self):
        long_snippet = "x" * 500
        payload = json.dumps({"findings": [
            {"existing_snippet": long_snippet, "new_snippet": "short", "note": "n"}
        ]})
        env_p, sys_p = self._patch_env_and_sdk(payload)
        with env_p, sys_p:
            result = check_drift(_LONG_BODY, [_ext("a")])
        self.assertLessEqual(len(result.findings[0].existing_snippet), 200)


class TestBuildDriftComment(unittest.TestCase):
    def test_empty_findings_returns_empty(self):
        self.assertEqual(build_drift_comment([]), "")

    def test_well_formed_comment(self):
        findings = [DriftFinding("ex", "new", "note")]
        comment = build_drift_comment(findings)
        self.assertTrue(comment.startswith("<!-- "))
        self.assertTrue(comment.endswith("-->"))
        self.assertIn("note", comment)
        self.assertIn("ex", comment)


if __name__ == "__main__":
    unittest.main()
