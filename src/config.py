"""
src/config.py -- Pipeline configuration and path constants.

Centralizes configurable paths, thresholds, and pipeline settings. The vault
path is env-driven and resolved lazily via VaultPaths (cached_property) so
tests that never touch the vault don't trip on a missing VAULT_ROOT.

Used by every other module to locate data sources and output targets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from .domains import ConfigError


def _home() -> Path:
    return Path(os.path.expanduser("~"))


@dataclass(frozen=True)
class Paths:
    """Filesystem paths for data sources and non-vault output targets.

    The vault path lives on VaultPaths (lazy) so env-var reads can be
    deferred until the vault is actually accessed.
    """

    pact_memory_db: Path = field(
        default_factory=lambda: _home() / ".claude" / "pact-memory" / "memory.db"
    )
    agent_memory_root: Path = field(
        default_factory=lambda: _home() / ".claude" / "agent-memory"
    )
    project_memory_root: Path = field(
        default_factory=lambda: _home() / ".claude" / "projects"
    )
    state_db: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "state.db"
    )
    domains_config: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "config" / "domains.yaml"
    )


class VaultPaths:
    """Lazy vault paths. Env-var read deferred until first access of `coding_notes`."""

    @cached_property
    def root(self) -> Path:
        v = os.environ.get("VAULT_ROOT")
        if not v:
            raise ConfigError(
                "VAULT_ROOT not set. Add it to .env or export it before running the pipeline. "
                "See .env.example."
            )
        p = Path(v).expanduser()
        if not p.is_absolute():
            raise ConfigError(f"VAULT_ROOT must be an absolute path: {v!r}")
        return p

    @cached_property
    def coding_notes(self) -> Path:
        subpath = os.environ.get("VAULT_NOTES_SUBPATH", "Coding-Notes")
        if subpath.startswith("/"):
            raise ConfigError(f"VAULT_NOTES_SUBPATH must be relative: {subpath!r}")
        return self.root / subpath


@dataclass(frozen=True)
class ClusterConfig:
    """Thresholds for the topic clusterer."""

    min_extractions_for_new_page: int = 3
    max_extractions_per_page: int = 15


@dataclass(frozen=True)
class SynthesisConfig:
    """Thresholds for the synthesizer (page-build stage)."""

    # Refuse to grow a page past this word count. Existing curated content
    # is left alone; new extractions that would push past the cap are skipped
    # and logged so they surface in the next quality-check report.
    max_page_words: int = 3000

    # v1.1.0: drift detection master switch. When True, page-merge runs an
    # LLM drift check that annotates contradictions as HTML comments. Falls
    # back to additive-only merge whenever the LLM call is unavailable.
    drift_check_enabled: bool = True

    # v1.1.0: semantic-dedup master switch. When True, a second-pass embedding
    # comparison runs after the existing exact/prefix dedup. Falls back to
    # v1.0.0 dedup when model2vec is not installed.
    semantic_dedup_enabled: bool = True

    # v1.1.0: cosine cutoff above which a new extraction is treated as a
    # paraphrase of an existing chunk. Strict `>` -- equality keeps the
    # candidate. Range [0, 1].
    semantic_dedup_threshold: float = 0.86


@dataclass(frozen=True)
class SessionConfig:
    """Thresholds for JSONL session processing (Phase 2)."""

    min_value_score: int = 3          # minimum classifier score to extract
    min_messages: int = 5             # minimum messages after filtering
    max_chunk_chars: int = 50_000     # max chars per text chunk


@dataclass(frozen=True)
class PruneConfig:
    """Defaults for the v1.1.0 rot-pruning subcommand."""

    # Number of recent runs that protect an extraction from pruning. ~3
    # months at a weekly schedule. Override per-invocation with --keep-runs.
    default_keep_runs: int = 12


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration."""

    version: str = "2.0.0"
    paths: Paths = field(default_factory=Paths)
    vault: VaultPaths = field(default_factory=VaultPaths)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    prune: PruneConfig = field(default_factory=PruneConfig)
    # Phase 2 uses template-based synthesis only (no LLM API calls).
