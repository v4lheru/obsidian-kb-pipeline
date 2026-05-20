"""
src/pipeline.py -- Pipeline orchestrator and CLI entry point.

Wires together the memory readers, JSONL session processors, clusterer,
synthesizer, wiki writer, and state store.  Supports three source modes:
  - memory-only: Phase 1 memory sources
  - sessions:    Phase 2 JSONL session processing
  - all:         Both memory + sessions

Usage:
  python -m src.pipeline run --sources memory-only
  python -m src.pipeline run --sources sessions --project "MyProject"
  python -m src.pipeline run --sources sessions --dry-run
  python -m src.pipeline run --sources all
  python -m src.pipeline status
  python -m src.pipeline quality-check
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from .classifier import SessionClassification, classify_session
from .clusterer import cluster_extractions
from .config import PipelineConfig
from .domains import load_domain_config, set_domain_config
from .jsonl_parser import discover_sessions
from .memory_reader import read_agent_memory, read_pact_memory, read_project_memory
from .prefilter import filter_session
from .quality_check import (
    print_fix_report,
    print_report,
    quality_fix,
    run_quality_check,
)
from .session_extractor import extract_session
from .state_store import Extraction, PipelineRun, StateStore, WikiPageRecord
from .synthesizer import WikiPage, synthesize_page
from .wiki_writer import WikiWriter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1: Memory pipeline
# ---------------------------------------------------------------------------

def run_memory_pipeline(config: PipelineConfig) -> dict[str, int]:
    """Execute the Phase 1 memory-only pipeline.

    Steps:
      1. Read all memory sources (pact-memory, agent, project)
      2. Cluster extractions by topic
      3. Synthesize wiki pages from clusters
      4. Write pages to vault
      5. Update MOC
      6. Record state

    Returns summary stats dict.
    """
    store = StateStore(config.paths.state_db)

    # Start run record
    run = PipelineRun(config={"version": config.version, "sources": "memory-only"})
    store.start_run(run)

    try:
        # Step 1: Read memory sources
        logger.info("=== Step 1: Reading memory sources ===")
        all_extractions: list[Extraction] = []

        pact_extractions = read_pact_memory(config.paths)
        all_extractions.extend(pact_extractions)

        agent_extractions = read_agent_memory(config.paths)
        all_extractions.extend(agent_extractions)

        project_extractions = read_project_memory(config.paths)
        all_extractions.extend(project_extractions)

        logger.info("Total extractions read: %d (pact=%d, agent=%d, project=%d)",
                     len(all_extractions), len(pact_extractions),
                     len(agent_extractions), len(project_extractions))

        if not all_extractions:
            logger.warning("No extractions found — nothing to do")
            run.completed_at = datetime.now(timezone.utc).isoformat()
            store.finish_run(run)
            store.close()
            return {"extractions": 0, "pages_created": 0, "pages_updated": 0}

        # Step 2: Cluster by topic
        logger.info("=== Step 2: Clustering extractions ===")
        candidates = cluster_extractions(
            all_extractions,
            config.vault.coding_notes,
            config.cluster,
        )

        # Step 3: Synthesize wiki pages
        logger.info("=== Step 3: Synthesizing wiki pages ===")
        pages: list[WikiPage] = []
        for candidate in candidates:
            page = synthesize_page(
                candidate, max_page_words=config.synthesis.max_page_words
            )
            pages.append(page)

        # Step 4: Write pages to vault
        logger.info("=== Step 4: Writing wiki pages ===")
        writer = WikiWriter(config.vault.coding_notes)
        writer.write_pages(pages)

        # Step 5: Auto-fix safe quality issues (orphans, broken links,
        # missing frontmatter) and flag oversized pages for manual split.
        logger.info("=== Step 5: Quality auto-fix ===")
        fix_report = quality_fix(config.vault.coding_notes)
        _log_fix_summary(fix_report)

        # Step 6: Update MOC
        logger.info("=== Step 6: Updating MOC ===")
        writer.update_moc(pages)

        # Step 7: Record state
        logger.info("=== Step 7: Recording state ===")
        store.save_extractions(all_extractions)
        for page in pages:
            store.save_wiki_page(WikiPageRecord(
                page_path=page.path,
                page_type=page.frontmatter.get("wiki-source", "auto"),
                title=page.title,
                extraction_ids=page.extraction_ids,
                word_count=page.word_count,
            ))

        pages_created = sum(1 for p in pages if p.is_new)
        pages_updated = sum(1 for p in pages if not p.is_new)

        run.completed_at = datetime.now(timezone.utc).isoformat()
        run.extractions_created = len(all_extractions)
        run.pages_created = pages_created
        run.pages_updated = pages_updated
        store.finish_run(run)

        logger.info(
            "Pipeline complete: %d extractions, %d pages created, %d pages updated",
            len(all_extractions), pages_created, pages_updated,
        )

        stats = {
            "extractions": len(all_extractions),
            "pages_created": pages_created,
            "pages_updated": pages_updated,
        }

    finally:
        store.close()

    return stats


# ---------------------------------------------------------------------------
# Phase 2: Session pipeline
# ---------------------------------------------------------------------------

def run_session_pipeline(
    config: PipelineConfig,
    project_filter: str | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Execute the Phase 2 JSONL session processing pipeline.

    Steps:
      1. Discover JSONL session files
      2. Filter and classify each session
      3. (dry-run stops here — reports classification stats)
      4. Extract text from high-value sessions
      5. Cluster extractions by topic
      6. Synthesize wiki pages
      7. Write pages to vault
      8. Update MOC
      9. Record state

    Returns summary stats dict.
    """
    store = StateStore(config.paths.state_db)
    run = PipelineRun(config={
        "version": config.version,
        "sources": "sessions",
        "project_filter": project_filter,
        "dry_run": dry_run,
    })
    store.start_run(run)

    try:
        # Step 1: Discover sessions
        logger.info("=== Step 1: Discovering JSONL sessions ===")
        sessions = discover_sessions(config.paths.project_memory_root)

        if project_filter:
            filter_lower = project_filter.lower()
            sessions = [s for s in sessions if filter_lower in s.project_dir.lower()]
            logger.info("Filtered to %d sessions matching '%s'",
                        len(sessions), project_filter)

        if not sessions:
            logger.warning("No sessions found — nothing to do")
            return _finish_session_run(store, run, len(sessions))

        # Step 2: Filter and classify
        logger.info("=== Step 2: Filtering and classifying %d sessions ===",
                     len(sessions))
        classifications: list[tuple] = []  # (FilteredSession, SessionClassification)

        for session_file in sessions:
            # Skip already-processed sessions (incremental runs)
            if store.is_session_processed(session_file.session_id, config.version):
                logger.debug("Skipping already-processed session %s",
                             session_file.session_id)
                continue

            filtered = filter_session(session_file)
            if filtered is None:
                continue

            classification = classify_session(filtered)
            classifications.append((filtered, classification))

        logger.info("Classified %d sessions: %s",
                     len(classifications),
                     _classification_summary(classifications))

        # Step 3: Dry run — report and exit
        if dry_run:
            _print_dry_run_report(classifications)
            run.sessions_processed = len(classifications)
            return _finish_session_run(store, run, len(sessions),
                                       sessions_processed=len(classifications))

        # Step 4: Extract text from high-value sessions
        logger.info("=== Step 4: Extracting text from high-value sessions ===")
        all_extractions: list[Extraction] = []
        sessions_processed = 0

        for filtered_session, classification in classifications:
            if classification.value_score < config.session.min_value_score:
                continue

            extractions = extract_session(filtered_session, classification)
            all_extractions.extend(extractions)
            sessions_processed += 1

            # Record session as processed
            store.save_processed_session(
                session_id=filtered_session.session_id,
                project_path=filtered_session.project_dir,
                file_path=filtered_session.file_path,
                file_size=filtered_session.file_size,
                message_count=len(filtered_session.messages),
                pipeline_version=config.version,
                classification=classification,
                extraction_ids=[e.id for e in extractions],
            )

        logger.info("Extracted %d chunks from %d sessions",
                     len(all_extractions), sessions_processed)

        if not all_extractions:
            logger.warning("No extractions from sessions — nothing to write")
            run.sessions_processed = sessions_processed
            return _finish_session_run(store, run, len(sessions),
                                       sessions_processed=sessions_processed)

        # Step 5: Cluster by topic
        logger.info("=== Step 5: Clustering %d extractions ===", len(all_extractions))
        candidates = cluster_extractions(
            all_extractions,
            config.vault.coding_notes,
            config.cluster,
        )

        # Step 6: Synthesize wiki pages
        logger.info("=== Step 6: Synthesizing wiki pages ===")
        pages: list[WikiPage] = []
        for candidate in candidates:
            page = synthesize_page(
                candidate, max_page_words=config.synthesis.max_page_words
            )
            pages.append(page)

        # Step 7: Write pages to vault
        logger.info("=== Step 7: Writing wiki pages ===")
        writer = WikiWriter(config.vault.coding_notes)
        writer.write_pages(pages)

        # Step 8: Auto-fix safe quality issues; flag oversized for manual split.
        logger.info("=== Step 8: Quality auto-fix ===")
        fix_report = quality_fix(config.vault.coding_notes)
        _log_fix_summary(fix_report)

        # Step 9: Update MOC
        logger.info("=== Step 9: Updating MOC ===")
        writer.update_moc(pages)

        # Step 10: Record state
        logger.info("=== Step 10: Recording state ===")
        store.save_extractions(all_extractions)
        for page in pages:
            store.save_wiki_page(WikiPageRecord(
                page_path=page.path,
                page_type=page.frontmatter.get("wiki-source", "auto"),
                title=page.title,
                extraction_ids=page.extraction_ids,
                word_count=page.word_count,
            ))

        pages_created = sum(1 for p in pages if p.is_new)
        pages_updated = sum(1 for p in pages if not p.is_new)

        run.completed_at = datetime.now(timezone.utc).isoformat()
        run.sessions_processed = sessions_processed
        run.extractions_created = len(all_extractions)
        run.pages_created = pages_created
        run.pages_updated = pages_updated
        store.finish_run(run)

        logger.info(
            "Session pipeline complete: %d sessions → %d extractions → "
            "%d pages created, %d pages updated",
            sessions_processed, len(all_extractions), pages_created, pages_updated,
        )

        return {
            "sessions_found": len(sessions),
            "sessions_processed": sessions_processed,
            "extractions": len(all_extractions),
            "pages_created": pages_created,
            "pages_updated": pages_updated,
        }

    finally:
        store.close()


def _log_fix_summary(fix_report) -> None:
    """One-line per-category log summary of a QualityFixReport."""
    logger.info(
        "Quality auto-fix: %d frontmatter repairs, %d broken links removed, "
        "%d orphans added to MOC, %d oversized pages flagged",
        len(fix_report.frontmatter_repaired),
        len(fix_report.broken_links_removed),
        len(fix_report.moc_entries_added),
        len(fix_report.oversized_flagged),
    )


def _finish_session_run(
    store: StateStore,
    run: PipelineRun,
    sessions_found: int,
    sessions_processed: int = 0,
    extractions: int = 0,
    pages_created: int = 0,
    pages_updated: int = 0,
) -> dict[str, int]:
    """Finalize a session run and return stats."""
    run.completed_at = datetime.now(timezone.utc).isoformat()
    run.extractions_created = extractions
    run.pages_created = pages_created
    run.pages_updated = pages_updated
    store.finish_run(run)
    store.close()
    return {
        "sessions_found": sessions_found,
        "sessions_processed": sessions_processed,
        "extractions": extractions,
        "pages_created": pages_created,
        "pages_updated": pages_updated,
    }


def _classification_summary(
    classifications: list[tuple],
) -> str:
    """Build a one-line summary of classification score distribution."""
    by_score: dict[int, int] = {}
    for _, c in classifications:
        by_score[c.value_score] = by_score.get(c.value_score, 0) + 1
    parts = [f"score {s}: {n}" for s, n in sorted(by_score.items())]
    return ", ".join(parts)


def _print_dry_run_report(classifications: list[tuple]) -> None:
    """Print a dry-run classification report to stdout."""
    by_score: dict[int, list] = {}
    for filtered, classification in classifications:
        by_score.setdefault(classification.value_score, []).append(
            (filtered, classification)
        )

    print(f"\n{'=' * 60}")
    print(f"DRY RUN — {len(classifications)} sessions classified")
    print(f"{'=' * 60}")

    for score in sorted(by_score.keys()):
        items = by_score[score]
        total_mb = sum(f.file_size for f, _ in items) / 1024 / 1024
        print(f"  Score {score}: {len(items):3d} sessions ({total_mb:.1f} MB)")

    high_value = [(f, c) for f, c in classifications if c.value_score >= 3]
    print(f"\nHigh-value (>= 3): {len(high_value)}")

    if high_value:
        by_domain: dict[str, int] = {}
        for _, c in high_value:
            by_domain[c.domain] = by_domain.get(c.domain, 0) + 1
        for domain, count in sorted(by_domain.items(), key=lambda x: -x[1]):
            print(f"  {domain}: {count}")

        est_chunks = sum(max(1, f.total_text_chars // 50_000) for f, c in high_value)
        print(f"\nEstimated chunks: ~{est_chunks}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def show_status(config: PipelineConfig) -> None:
    """Print pipeline status report."""
    store = StateStore(config.paths.state_db)

    extractions = store.get_extractions()
    pages = store.get_wiki_pages()

    print(f"obsidian-kb-pipeline Status (v{config.version})")
    print(f"{'=' * 50}")
    print(f"State DB: {config.paths.state_db}")
    print(f"Vault:    {config.vault.coding_notes}")
    print()
    print(f"Extractions: {len(extractions)}")
    print(f"Wiki pages:  {len(pages)}")

    if extractions:
        # Count by source type
        by_source: dict[str, int] = {}
        for ext in extractions:
            by_source[ext.source_type] = by_source.get(ext.source_type, 0) + 1
        print("\nExtractions by source:")
        for source, count in sorted(by_source.items()):
            print(f"  {source}: {count}")

    if pages:
        print("\nWiki pages:")
        for page in sorted(pages, key=lambda p: p.page_path):
            print(f"  {page.page_path} ({page.word_count} words)")

    # Phase 2: Show processed sessions
    processed = store.get_processed_sessions()
    if processed:
        print(f"\nProcessed sessions: {len(processed)}")

    store.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="obsidian-kb-pipeline",
        description="LLM Wiki Knowledge pipeline for Obsidian",
    )
    sub = parser.add_subparsers(dest="command")

    # run command
    run_parser = sub.add_parser("run", help="Run the pipeline")
    run_parser.add_argument(
        "--sources", default="memory-only",
        choices=["memory-only", "sessions", "all"],
        help="Which sources to process (default: memory-only)",
    )
    run_parser.add_argument(
        "--project", default=None,
        help="Filter sessions to those matching this project keyword",
    )
    run_parser.add_argument(
        "--dry-run", action="store_true",
        help="Classify sessions only — report scores without processing",
    )

    # status command
    sub.add_parser("status", help="Show pipeline status")

    # quality-check command
    qc_parser = sub.add_parser(
        "quality-check",
        help="Scan wiki pages for quality issues",
    )
    qc_parser.add_argument(
        "--max-words", type=int, default=5000,
        help="Flag pages with more than this many words (default: 5000)",
    )
    qc_parser.add_argument(
        "--min-words", type=int, default=100,
        help="Flag pages with fewer than this many words (default: 100)",
    )

    # quality-fix command
    qf_parser = sub.add_parser(
        "quality-fix",
        help="Auto-repair safe quality issues (orphans, broken links, "
             "missing frontmatter); flag oversized pages",
    )
    qf_parser.add_argument(
        "--max-words", type=int, default=5000,
        help="Flag pages with more than this many words (default: 5000)",
    )
    qf_parser.add_argument(
        "--min-words", type=int, default=100,
        help="Flag pages with fewer than this many words (default: 100)",
    )

    args = parser.parse_args()
    config = PipelineConfig()

    # Load + install the active DomainConfig before any pipeline stage runs.
    # All downstream modules read taxonomy through get_domain_config().
    set_domain_config(load_domain_config(config.paths.domains_config))

    if args.command == "run":
        if args.sources == "memory-only":
            stats = run_memory_pipeline(config)
            print(f"\nDone: {stats['extractions']} extractions → "
                  f"{stats['pages_created']} new pages, "
                  f"{stats['pages_updated']} updated pages")

        elif args.sources == "sessions":
            stats = run_session_pipeline(
                config,
                project_filter=args.project,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                print(f"\nDone: {stats['sessions_processed']} sessions → "
                      f"{stats['extractions']} extractions → "
                      f"{stats['pages_created']} new pages, "
                      f"{stats['pages_updated']} updated pages")

        elif args.sources == "all":
            # Run both pipelines sequentially
            print("=== Memory sources ===")
            mem_stats = run_memory_pipeline(config)
            print(f"Memory: {mem_stats['extractions']} extractions → "
                  f"{mem_stats['pages_created']} new, "
                  f"{mem_stats['pages_updated']} updated")

            print("\n=== Session sources ===")
            sess_stats = run_session_pipeline(
                config,
                project_filter=args.project,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                print(f"Sessions: {sess_stats['sessions_processed']} sessions → "
                      f"{sess_stats['extractions']} extractions → "
                      f"{sess_stats['pages_created']} new, "
                      f"{sess_stats['pages_updated']} updated")

    elif args.command == "status":
        show_status(config)

    elif args.command == "quality-check":
        report = run_quality_check(
            config.vault.coding_notes,
            max_words=args.max_words,
            min_words=args.min_words,
        )
        print_report(report)

    elif args.command == "quality-fix":
        fix_report = quality_fix(
            config.vault.coding_notes,
            max_words=args.max_words,
            min_words=args.min_words,
        )
        print_fix_report(fix_report)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
