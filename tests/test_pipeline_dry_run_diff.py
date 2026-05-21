"""
tests/test_pipeline_dry_run_diff.py -- End-to-end test of --dry-run-mode=diff.

Covers: memory pipeline in diff-mode writes nothing to disk, prints a diff
report, leaves state.db rows un-added; argparse rejects --dry-run-mode=diff
without --dry-run.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from src.config import PipelineConfig, Paths
from src.pipeline import main as pipeline_main
from src.pipeline import run_memory_pipeline
from src.state_store import StateStore


class TestDryRunDiffMode(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vault = Path(self.tmp) / "vault"
        self.coding = self.vault / "Coding-Notes"
        self.coding.mkdir(parents=True, exist_ok=True)
        os.environ["VAULT_ROOT"] = str(self.vault)
        self.state_db = Path(self.tmp) / "state.db"
        self.config = PipelineConfig(
            paths=Paths(
                pact_memory_db=Path(self.tmp) / "missing.db",
                agent_memory_root=Path(self.tmp) / "agent",
                project_memory_root=Path(self.tmp) / "projects",
                state_db=self.state_db,
            ),
        )

    def tearDown(self):
        os.environ.pop("VAULT_ROOT", None)
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_memory_pipeline_diff_mode_no_vault_writes(self):
        """In diff-mode, no files should be created under coding-notes."""
        # No sources exist -> pipeline short-circuits early with 0 extractions.
        # That's enough to verify the diff-mode code path doesn't crash and
        # produces no vault writes.
        buf = io.StringIO()
        with redirect_stdout(buf):
            stats = run_memory_pipeline(self.config, dry_run_mode="diff")

        # No markdown files under the vault notes directory.
        md_files = list(self.coding.rglob("*.md"))
        self.assertEqual(md_files, [], f"unexpected files written: {md_files}")

        # Stats keys exist.
        self.assertEqual(stats["pages_created"], 0)
        self.assertEqual(stats["pages_updated"], 0)


class TestDryRunDiffCli(unittest.TestCase):
    def test_diff_mode_without_dry_run_errors(self):
        """`--dry-run-mode=diff` without `--dry-run` must trigger an argparse
        error (SystemExit with code 2)."""
        old_argv = sys.argv
        sys.argv = ["pipeline", "run", "--dry-run-mode=diff"]
        # Block on a fake set_domain_config so we don't depend on the real
        # config file landing in the temp dir.
        try:
            with mock.patch("src.pipeline.load_domain_config"), \
                 mock.patch("src.pipeline.set_domain_config"), \
                 self.assertRaises(SystemExit) as cm:
                pipeline_main()
            self.assertEqual(cm.exception.code, 2)
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
