"""
tests/test_domains.py -- Tests for src/domains.py loader + merge semantics.

Covers the five behaviors enumerated in the refactor-map's domains.py section:
load-with-no-file, explicit-missing-raises, user-override merge, malformed-yaml
raises, and the wiki_writer safe-default for unset ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.domains import (
    BUILTIN_DOMAIN_CONFIG,
    ConfigError,
    DomainConfig,
    DomainPattern,
    load_domain_config,
    set_domain_config,
)


class TestLoadDomainConfig(unittest.TestCase):
    """Behavior of load_domain_config under different config-file states."""

    def setUp(self) -> None:
        # Reset the module-level singleton between tests so prior state
        # never leaks. set_domain_config(None) triggers auto-load fallback.
        set_domain_config(None)

    def tearDown(self) -> None:
        set_domain_config(None)

    def test_load_returns_builtin_when_no_file(self) -> None:
        """Default path missing -> silent fallback to BUILTIN_DOMAIN_CONFIG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = load_domain_config(Path(tmpdir) / "nonexistent.yaml")
        # The fallback is the built-in constant itself (no merge happens when
        # there is no user file).
        self.assertEqual(cfg.project_domain_patterns,
                         BUILTIN_DOMAIN_CONFIG.project_domain_patterns)
        self.assertEqual(cfg.path_skip_prefixes,
                         BUILTIN_DOMAIN_CONFIG.path_skip_prefixes)

    def test_load_explicit_env_override_missing_raises(self) -> None:
        """KB_BRAIN_DOMAINS_FILE set to a missing path -> ConfigError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "absent.yaml"
            with patch.dict(os.environ, {"KB_BRAIN_DOMAINS_FILE": str(missing)},
                            clear=False):
                with self.assertRaises(ConfigError):
                    load_domain_config()

    def test_load_user_entries_override(self) -> None:
        """User entries from the fixture YAML merge over built-in defaults."""
        fixture = Path(__file__).parent / "fixtures" / "domains.yaml"
        self.assertTrue(fixture.exists(),
                        f"Fixture file must exist at {fixture}")

        cfg = load_domain_config(fixture)

        # User entry is first in the tuple (so iteration finds it before built-ins).
        self.assertEqual(cfg.project_domain_patterns[0],
                         DomainPattern("fixture-app", "fixture-domain"))

        # Built-in n8n pattern is still present (further in the tuple).
        domains_in_tuple = {p.keyword for p in cfg.project_domain_patterns}
        self.assertIn("n8n", domains_in_tuple)

        # Skip prefixes union: user 'alice' + built-in 'users'.
        self.assertIn("alice", cfg.path_skip_prefixes)
        self.assertIn("users", cfg.path_skip_prefixes)

        # User topic_canonicalization entry merged in.
        self.assertEqual(cfg.topic_canonicalization.get("fixture-domain"), "backend")
        # Built-in canonicalization entry still present.
        self.assertEqual(cfg.topic_canonicalization.get("n8n"), "n8n")

    def test_load_malformed_yaml_raises(self) -> None:
        """Malformed YAML -> ConfigError with parse detail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad = Path(tmpdir) / "bad.yaml"
            bad.write_text("x: : :\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_domain_config(bad)

    def test_anthropic_key_safe_default(self) -> None:
        """wiki_writer._generate_moc_description returns '' when API key unset."""
        from src.wiki_writer import _llm_summarize_page_for_moc
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_llm_summarize_page_for_moc("any body", "any title"), "")


if __name__ == "__main__":
    unittest.main()
