"""
src/run_prune.py -- Rot-pruning subcommand implementation (v1.1.0).

Drops extractions whose ``last_confirmed_run_id`` is older than N runs
back. Dry-run by default; ``--apply`` actually deletes. ``--rebuild``
resynthesizes affected wiki pages from the surviving extractions.

Called from src/pipeline.py's ``prune`` argparse subcommand. The state
schema migration (see state_store._migrate_v1_1_0) must have run before
this module's run_prune() is invoked; StateStore.__init__ handles that.
"""

from __future__ import annotations

import logging

from .config import PipelineConfig
from .state_store import PruneReport, StateStore

logger = logging.getLogger(__name__)


def run_prune(
    config: PipelineConfig,
    keep_runs: int | None = None,
    apply: bool = False,
    rebuild: bool = False,
) -> PruneReport:
    """Execute the prune subcommand.

    Refuses to run when any pipeline_runs row has ``completed_at IS NULL``
    (an in-flight run could still call touch_extractions on rows we're
    about to prune). Without ``apply``, no DELETE runs and the report
    reflects what an ``--apply`` invocation would do. With ``rebuild``,
    every wiki page that referenced a pruned extraction is resynthesized
    from the surviving extractions.

    ``keep_runs`` defaults to ``config.prune.default_keep_runs``. The
    cutoff is ``max_run_id_int - keep_runs + 1``; an extraction is stale
    when its ``last_confirmed_run_id`` is strictly less than the cutoff,
    so a row stamped exactly ``keep_runs`` runs back survives.
    """
    store = StateStore(config.paths.state_db)
    try:
        unfinished = store.get_unfinished_run()
        if unfinished is not None:
            raise RuntimeError(
                f"An incomplete pipeline run exists (run_id={unfinished}). "
                "Wait for it to finish or delete the row from state.db "
                "before pruning."
            )

        effective_keep = (
            keep_runs if keep_runs is not None
            else config.prune.default_keep_runs
        )
        if effective_keep < 1:
            raise ValueError("keep_runs must be >= 1")

        max_run = store.max_run_id_int()
        if max_run < effective_keep:
            logger.info(
                "Only %d runs in history; need >= %d to prune. No-op.",
                max_run, effective_keep,
            )
            return PruneReport(
                extractions_deleted=0,
                extractions_inspected=0,
                affected_pages=[],
                cutoff_run_id_int=0,
            )

        cutoff = max_run - effective_keep + 1
        logger.info(
            "Pruning extractions with last_confirmed_run_id < %d "
            "(keep_runs=%d, max_run_id_int=%d, mode=%s)",
            cutoff, effective_keep, max_run,
            "apply" if apply else "dry-run",
        )

        report = store.prune_stale(cutoff, dry_run=not apply)

        if apply and rebuild and report.affected_pages:
            _rebuild_affected_pages(store, config, report.affected_pages)

        return report
    finally:
        store.close()


def _rebuild_affected_pages(
    store: StateStore,
    config: PipelineConfig,
    affected_pages: list[str],
) -> None:
    """Resynthesize pages whose extractions were just pruned.

    Imports are deferred so the prune subcommand stays usable without
    VAULT_ROOT in environments where only state inspection is needed.
    """
    from .clusterer import PageCandidate
    from .synthesizer import synthesize_page
    from .wiki_writer import WikiWriter

    surviving_by_page = _group_surviving_extractions(store, affected_pages)
    if not surviving_by_page:
        logger.info("No surviving extractions found for affected pages; "
                    "leaving page markdown intact.")
        return

    writer = WikiWriter(config.vault.coding_notes)
    vault_root = config.vault.coding_notes
    pages = []
    for page_path, extractions in surviving_by_page.items():
        if not extractions:
            continue
        topic = extractions[0].topic
        title = _title_from_path(page_path)
        existing_content = _read_existing_page(vault_root, page_path)
        candidate = PageCandidate(
            action="update",
            page_type="pattern",
            topic=topic,
            title=title,
            target_path=page_path,
            extractions=extractions,
            existing_content=existing_content,
        )
        pages.append(synthesize_page(
            candidate, max_page_words=config.synthesis.max_page_words
        ))

    if pages:
        writer.write_pages(pages)
        logger.info("Rebuilt %d page(s) from surviving extractions.",
                    len(pages))


def _read_existing_page(vault_root, page_path: str) -> str:
    """Read the existing page body from disk; empty string if absent."""
    full = vault_root / page_path
    try:
        return full.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _group_surviving_extractions(
    store: StateStore, affected_pages: list[str]
) -> dict[str, list]:
    """For each affected page path, collect surviving extractions whose
    ids appear in the page's stored extraction_ids JSON. Pruned ids are
    dropped silently — the join works because DELETE has already run."""
    import json
    page_rows = store._conn.execute(
        "SELECT page_path, extraction_ids FROM wiki_pages"
    ).fetchall()
    affected_set = set(affected_pages)
    out: dict[str, list] = {}
    for page_path, ext_ids_json in page_rows:
        if page_path not in affected_set:
            continue
        try:
            page_ext_ids = json.loads(ext_ids_json or "[]")
        except json.JSONDecodeError:
            continue
        survivors = []
        for eid in page_ext_ids:
            extraction = store.get_extraction(eid)
            if extraction is not None:
                survivors.append(extraction)
        out[page_path] = survivors
    return out


def _title_from_path(page_path: str) -> str:
    """Derive a human-friendly title from a page path's basename."""
    from pathlib import PurePosixPath
    stem = PurePosixPath(page_path).stem
    return stem.replace("-", " ").replace("_", " ").strip().title() or stem


def format_report(report: PruneReport, apply: bool) -> str:
    """Render a PruneReport for terminal output."""
    verb = "Deleted" if apply else "Would delete"
    lines = [
        "=" * 60,
        f"Prune report ({'apply' if apply else 'dry-run'})",
        "=" * 60,
        f"  Cutoff run_id_int : {report.cutoff_run_id_int}",
        f"  Extractions inspected: {report.extractions_inspected}",
        f"  {verb}: {report.extractions_deleted} extraction(s)",
    ]
    if report.affected_pages:
        lines.append(f"  Affected pages ({len(report.affected_pages)}):")
        for p in report.affected_pages:
            lines.append(f"    - {p}")
    else:
        lines.append("  Affected pages: none")
    lines.append("=" * 60)
    return "\n".join(lines)
