# LLM Wiki System Architecture

## Executive Summary

This document defines the architecture for an LLM-powered knowledge distillation system that transforms ~1.9GB of Claude Code session data (3,755 JSONL files across 55 projects) into a curated Obsidian knowledge wiki. The system follows a three-layer model: **Raw Sources** (Claude Code sessions, pact-memory, agent-memory) feed into a **Distillation Pipeline** that produces **Wiki Pages** (Obsidian markdown with wikilinks). A **Schema Document** governs how LLMs interact with and extend the wiki over time.

The design prioritizes incremental processing, quality over quantity, and integration with the existing Obsidian vault structure.

---

## 1. System Context (C4 Level 1)

```
                        +-------------------+
                        |   User (Uros)     |
                        |  Obsidian client  |
                        +--------+----------+
                                 |
                                 | reads/browses
                                 v
+------------------+     +-------+--------+     +-------------------+
| Claude Code      |     |                |     | PACT Sessions     |
| Sessions (JSONL) |---->|  LLM Wiki      |<----| (ongoing work)    |
| 55 projects      |     |  System        |     | via PACT hooks    |
| ~1.9GB           |     |                |     +-------------------+
+------------------+     +-------+--------+
                                 |
| pact-memory DB   |---->|      |      |<----| Agent Memory      |
| 109 entries      |     |      |      |     | 20 specialists    |
+------------------+     +------+------+     +-------------------+
                                 |
                                 | writes
                                 v
                        +--------+----------+
                        |  Obsidian Vault   |
                        |  05-Education/    |
                        |  Coding-Notes/    |
                        +-------------------+
```

### External Dependencies

| Dependency | Role | Interface |
|------------|------|-----------|
| Claude Code sessions | Primary raw data source | JSONL files at `~/.claude/projects/` |
| pact-memory SQLite | Structured knowledge (decisions, lessons, entities) | SQLite at `~/.claude/pact-memory/memory.db` |
| Agent persistent memory | Domain-specific specialist knowledge | Markdown at `~/.claude/agent-memory/*/` |
| Per-project memory | Project-scoped learnings | Markdown at `~/.claude/projects/*/memory/` |
| Obsidian vault | Target output location | Markdown + YAML frontmatter at vault path |
| LLM (Claude API) | Distillation engine for classification, extraction, synthesis | Anthropic API (Claude Sonnet for bulk, Opus for synthesis) |

---

## 2. Container Architecture (C4 Level 2)

```
+===========================================================================+
|                          LLM Wiki System                                   |
|                                                                            |
|  +-----------------+    +------------------+    +--------------------+     |
|  |  Source Reader  |    |  Pipeline        |    |  Wiki Writer       |     |
|  |                 |--->|  Orchestrator    |--->|                    |     |
|  | - JSONL parser  |    |                  |    | - Page renderer    |     |
|  | - Memory reader |    | - Session queue  |    | - Frontmatter gen  |     |
|  | - Dedup filter  |    | - Batch manager  |    | - Wikilink injector|     |
|  +-----------------+    | - Cost tracker   |    | - MOC updater      |     |
|                         +--------+---------+    +--------------------+     |
|                                  |                                         |
|                         +--------v---------+                               |
|                         |  Distillation    |                               |
|                         |  Engine          |                               |
|                         |                  |                               |
|                         | - Pre-filter     |                               |
|                         | - Classifier     |                               |
|                         | - Extractor      |                               |
|                         | - Clusterer      |                               |
|                         | - Synthesizer    |                               |
|                         +------------------+                               |
|                                                                            |
|  +-----------------+    +------------------+                               |
|  |  State Store    |    |  Schema          |                               |
|  |                 |    |  Document        |                               |
|  | - Run log       |    |                  |                               |
|  | - Session index |    | - Page types     |                               |
|  | - Quality scores|    | - Conventions    |                               |
|  +-----------------+    | - Taxonomy       |                               |
|                         +------------------+                               |
+===========================================================================+
```

### Container Responsibilities

| Container | Responsibility | Technology |
|-----------|---------------|------------|
| **Source Reader** | Parse JSONL sessions, read memory DBs, deduplicate | Python (json, sqlite3) |
| **Pipeline Orchestrator** | Queue sessions, manage batches, track costs, resume on failure | Python (asyncio) |
| **Distillation Engine** | Five-stage LLM processing (filter → classify → extract → cluster → synthesize) | Claude API (Sonnet 4 for bulk, Opus for synthesis) |
| **Wiki Writer** | Render markdown pages, generate frontmatter, inject wikilinks, update MOC | Python (string templates) |
| **State Store** | Track processed sessions, run history, quality scores | SQLite (local) |
| **Schema Document** | Define wiki conventions, page types, taxonomy rules | Markdown (in vault) |

---

## 3. Data Architecture

### 3.1 Source Data Analysis

| Source | Format | Size | Count | Value Density |
|--------|--------|------|-------|---------------|
| Claude Code sessions | JSONL | ~1.9GB | 3,755 files | Low-to-medium (lots of tool output noise) |
| pact-memory DB | SQLite | ~500KB | 109 rows | High (pre-distilled decisions, lessons) |
| Agent persistent memory | Markdown | ~2MB | ~80 files across 20 dirs | High (domain expertise) |
| Per-project memory | Markdown | ~200KB | ~40 files across 10 projects | High (project-specific context) |

### 3.2 Session JSONL Structure

Each JSONL file contains messages with these types:

| Type | Content | Extraction Value |
|------|---------|-----------------|
| `user` | User prompts, requirements, questions | HIGH — reveals intent, requirements |
| `assistant` | LLM responses, code, explanations | HIGH — contains solutions, decisions |
| `system` | System prompts, tool configurations | LOW — mostly boilerplate |
| `file-history-snapshot` | File state tracking | LOW — metadata only |
| `last-prompt` | Session resume pointer | NONE — operational only |

**Key fields in user/assistant messages:**
- `message.content` — the actual text (string or content blocks array)
- `message.role` — user or assistant
- `timestamp` — for temporal ordering
- `cwd` — working directory (maps to project)
- `sessionId` — groups messages into conversations

**Subagent sessions** exist at `{session_id}/subagents/agent-{id}.jsonl` — these contain specialist agent work. Valuable for detailed implementation context but high volume.

### 3.3 Wiki Page Schema

#### Page Types

| Type | Purpose | Source | Example |
|------|---------|--------|---------|
| **entity** | A specific tool, API, service, or framework | Sessions + memory | `api-reference.md` |
| **architecture** | How a system was designed and why | Sessions + pact-memory decisions | `arch-speaker-background-check.md` |
| **pattern** | Reusable technique across projects | Cross-session clustering | `backend-patterns.md` |
| **gotcha** | Hard-won debugging knowledge, quirks | Session failures + lessons_learned | `n8n-apify-integration.md` |
| **workflow** | Step-by-step process for a recurring task | Session sequences | `n8n-project-workflows.md` |
| **comparison** | Side-by-side evaluation of alternatives | Decision points in sessions | (new type) |
| **synthesis** | Cross-cutting insight from multiple sources | Multi-session analysis | (new type) |

#### Frontmatter Schema (extends vault-metadata-schema.md)

```yaml
---
type: resource                    # Always 'resource' for wiki pages
tags:
  - wiki                          # Required tag for all wiki pages
  - {domain-tags}                 # e.g., n8n, supabase, gcp
  - {page-type}                   # e.g., pattern, architecture, gotcha
status: reference                 # 'reference' for stable, 'draft' for auto-generated
created: YYYY-MM-DD
modified: YYYY-MM-DD
area: work                        # or 'side-project', 'education'
tool: {primary-tool}              # Optional: primary tool/service covered
wiki-source: auto | manual | seed # Provenance: auto-extracted, manually written, or seed content
wiki-confidence: high | medium | low  # Extraction confidence
wiki-sources:                     # Session IDs that contributed
  - {session-id-1}
  - {session-id-2}
---
```

The `wiki-source`, `wiki-confidence`, and `wiki-sources` fields are wiki-specific extensions. They enable quality tracking and provenance without breaking existing vault conventions.

### 3.4 State Store Schema

```sql
-- Track which sessions have been processed
CREATE TABLE processed_sessions (
  session_id TEXT PRIMARY KEY,
  project_path TEXT NOT NULL,
  file_path TEXT NOT NULL,
  file_size INTEGER,
  message_count INTEGER,
  processed_at TEXT NOT NULL,       -- ISO 8601
  pipeline_version TEXT NOT NULL,   -- Semantic version for reprocessing
  classification TEXT,              -- JSON: {value_score, topics, skip_reason}
  extraction_ids TEXT,              -- JSON array of extraction IDs
  cost_usd REAL                     -- API cost for this session
);

-- Track individual knowledge extractions
CREATE TABLE extractions (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  extraction_type TEXT NOT NULL,    -- entity, pattern, gotcha, decision, etc.
  topic TEXT NOT NULL,
  content TEXT NOT NULL,            -- Raw extracted content
  confidence REAL,                  -- 0.0-1.0
  clustered_into TEXT,              -- ID of wiki page this was merged into
  created_at TEXT NOT NULL,
  FOREIGN KEY (session_id) REFERENCES processed_sessions(session_id)
);

-- Track wiki pages and their lineage
CREATE TABLE wiki_pages (
  page_path TEXT PRIMARY KEY,       -- Relative path in vault
  page_type TEXT NOT NULL,          -- entity, architecture, pattern, etc.
  title TEXT NOT NULL,
  extraction_ids TEXT NOT NULL,     -- JSON array of source extraction IDs
  last_generated TEXT NOT NULL,     -- ISO 8601
  word_count INTEGER,
  quality_score REAL,               -- 0.0-1.0 from quality check
  needs_review BOOLEAN DEFAULT 0
);

-- Run history for auditability
CREATE TABLE pipeline_runs (
  run_id TEXT PRIMARY KEY,
  started_at TEXT NOT NULL,
  completed_at TEXT,
  sessions_processed INTEGER,
  extractions_created INTEGER,
  pages_created INTEGER,
  pages_updated INTEGER,
  total_cost_usd REAL,
  config TEXT                       -- JSON of run configuration
);
```

---

## 4. Distillation Pipeline (Component Architecture)

### 4.1 Five-Stage Pipeline

```
Raw Sessions                                          Wiki Pages
    |                                                     ^
    v                                                     |
+--------+    +-----------+    +-----------+    +--------+--------+
| Stage 1|    | Stage 2   |    | Stage 3   |    | Stage 4| Stage 5|
| Pre-   |--->| Classify  |--->| Extract   |--->|Cluster |--->|Synth- |
| Filter |    |           |    |           |    |        |    |esize  |
+--------+    +-----------+    +-----------+    +--------+---------+
  Local         LLM call        LLM call         Local      LLM call
  (no API)     (Sonnet)        (Sonnet)         (no API)    (Opus)
```

### Stage 1: Pre-Filter (Local, No API)

**Purpose**: Eliminate sessions with no extractable value before spending API tokens.

**Filters**:
1. **Size filter**: Skip sessions < 5 messages (too short for meaningful content)
2. **Content filter**: Skip sessions that are purely tool output with no user/assistant dialogue
3. **Subagent filter**: Flag subagent sessions for deferred processing (process parent first)
4. **Already-processed filter**: Check state store for session_id + pipeline_version match
5. **Project grouping**: Group sessions by project for context-aware processing

**Output**: Ordered queue of session groups, sorted by expected value (larger sessions from active projects first).

**Estimated reduction**: ~40-50% of sessions filtered out (empty projects, tiny sessions, pure subagent files).

### Stage 2: Classify (LLM — Sonnet)

**Purpose**: Score each session's knowledge value and identify topics.

**Input**: First 4,000 tokens + last 2,000 tokens of each session (head+tail sampling captures intent and outcome).

**Prompt pattern**:
```
Given this Claude Code session excerpt, classify it:

1. VALUE SCORE (1-5):
   1 = No extractable knowledge (casual chat, failed attempts with no learning)
   2 = Minor operational knowledge (simple config, routine changes)
   3 = Useful technical knowledge (API usage, patterns, debugging)
   4 = Significant knowledge (architecture decisions, novel solutions, hard-won debugging)
   5 = Critical knowledge (security findings, production gotchas, cross-cutting patterns)

2. TOPICS: List 1-5 topic tags (e.g., "n8n", "supabase-rls", "okta-saml", "api-pagination")

3. PAGE TYPES: Which wiki page types could this feed into?
   (entity | architecture | pattern | gotcha | workflow | comparison)

4. SKIP REASON: If score <= 2, explain why in one line.
```

**Decision gate**: Sessions scoring <= 2 are skipped. Sessions scoring 3+ proceed to extraction.

**Estimated pass rate**: ~30-40% of sessions proceed (based on typical session value distribution).

### Stage 3: Extract (LLM — Sonnet)

**Purpose**: Pull discrete knowledge units from high-value sessions.

**Input**: Full session content (for sessions < 100K tokens) or chunked with overlap (for larger sessions).

**Extraction types**:
- **Decisions**: What was decided and why (with alternatives considered)
- **Gotchas**: Bugs found, undocumented behaviors, workarounds
- **Patterns**: Reusable code patterns, architectural approaches
- **API quirks**: Undocumented API behavior, rate limits, parameter gotchas
- **Entity facts**: Tool capabilities, version-specific behaviors
- **Process steps**: Ordered workflows that were developed

**Prompt pattern**:
```
Extract knowledge units from this session. For each unit:

- TYPE: decision | gotcha | pattern | api_quirk | entity_fact | process_step
- TOPIC: Primary topic tag
- TITLE: One-line summary
- CONTENT: The knowledge itself (2-10 sentences, specific and actionable)
- CONFIDENCE: high | medium | low
- CONTEXT: What project/tool this relates to

Return ONLY knowledge that would be valuable to a developer encountering the same tool/problem in the future. Skip routine operations, boilerplate, and tool output.
```

### Stage 4: Cluster (Local, No API)

**Purpose**: Group related extractions across sessions into wiki page candidates.

**Algorithm**:
1. **Topic matching**: Group extractions by primary topic tag
2. **Merge with existing**: Check if an existing wiki page already covers this topic
3. **Threshold check**: A new page needs >= 3 extractions to be worth creating
4. **Split check**: If a topic cluster exceeds 15 extractions, consider splitting by sub-topic
5. **Cross-reference detection**: Identify extractions that reference each other's topics

**Output**: Page candidates — groups of extractions destined for the same wiki page, with merge/create/update action tags.

### Stage 5: Synthesize (LLM — Opus)

**Purpose**: Transform extraction clusters into polished wiki pages.

**Why Opus**: Synthesis requires judgment — deciding what matters most, how to organize, what to emphasize. This is the quality-critical step.

**Input per page**: All extractions for that topic cluster + existing page content (if updating).

**Prompt pattern**:
```
Write an Obsidian wiki page from these knowledge extractions.

PAGE TYPE: {type}
TOPIC: {topic}
EXISTING CONTENT: {existing_page or "none — new page"}

Rules:
- Use ## headings, bullet lists, code blocks
- Include [[wikilinks]] to related pages where natural
- End with a ## Related section listing linked pages
- If updating an existing page: preserve manually-written content, integrate new knowledge
- Be specific and actionable — avoid generic advice
- Include code examples when they illustrate a non-obvious point
- Mark uncertain claims with "(unverified)" suffix

Output the full page body (no frontmatter — that's generated separately).
```

### 4.2 Pipeline Configuration

```python
# pipeline_config.py (conceptual)
PIPELINE_CONFIG = {
    "version": "1.0.0",

    # Pre-filter
    "min_session_messages": 5,
    "skip_subagent_sessions": True,  # Phase 1; include in Phase 2

    # Classify
    "classify_model": "claude-sonnet-4-6",
    "classify_max_tokens": 500,
    "classify_head_tokens": 4000,
    "classify_tail_tokens": 2000,
    "min_value_score": 3,

    # Extract
    "extract_model": "claude-sonnet-4-6",
    "extract_max_tokens": 4000,
    "extract_chunk_size": 80000,  # tokens per chunk for large sessions
    "extract_chunk_overlap": 2000,

    # Cluster
    "min_extractions_for_new_page": 3,
    "max_extractions_per_page": 15,

    # Synthesize
    "synthesize_model": "claude-opus-4-6",
    "synthesize_max_tokens": 8000,

    # Cost management
    "max_cost_per_run_usd": 50.0,
    "cost_warning_threshold_usd": 30.0,

    # Batch control
    "batch_size": 20,  # Sessions per batch
    "pause_between_batches_sec": 5,
}
```

### 4.3 Memory Source Integration

The pipeline doesn't only process JSONL sessions. It also reads pre-distilled knowledge:

| Source | Integration Point | Strategy |
|--------|-------------------|----------|
| pact-memory DB (109 entries) | Injected at Stage 4 (Cluster) | Each memory entry is treated as a pre-classified extraction with HIGH confidence. Clustered alongside session extractions. |
| Agent persistent memory (~80 files) | Injected at Stage 4 (Cluster) | Parsed as markdown extractions. Agent name maps to topic domain. |
| Per-project memory (~40 files) | Injected at Stage 4 (Cluster) | Parsed as markdown extractions. Project path maps to project tag. |
| Existing wiki pages (18 seed files) | Loaded at Stage 5 (Synthesize) | Existing pages are enriched, not replaced. Manual content is preserved. |

This means Stages 1-3 are JSONL-only. Stage 4 merges all sources. Stage 5 outputs the final pages.

---

## 5. Wiki Taxonomy and Vault Structure

### 5.1 Target Structure

The wiki lives within the existing vault at `05-Education/Coding-Notes/`. No restructuring of the top-level vault folders is needed — the existing 00-08 structure is preserved.

```
05-Education/Coding-Notes/
  coding-knowledge-map.md          # MOC (Map of Content) — auto-updated
  git-workflow-best-practices.md   # Existing seed
  
  APIs/                            # Entity pages for external services
    api-reference.md               # Example seed
    impact-com-api-reference.md    # Existing seed
    apify-platform-reference.md    # Existing seed
    third-party-api-misc.md        # Existing seed
    {new-api-pages}.md             # Auto-generated
  
  Architecture/                    # Architecture decision records
    arch-*.md                      # Existing seeds (6 files)
    {new-arch-pages}.md            # Auto-generated
  
  Patterns/                        # Cross-cutting coding patterns
    backend-patterns.md            # Existing seed
    frontend-patterns.md           # Existing seed
    database-schema-patterns.md    # Existing seed
    monorepo-and-devops-patterns.md # Existing seed
    {new-pattern-pages}.md         # Auto-generated
  
  n8n/                             # n8n-specific knowledge
    n8n-workflow-patterns.md       # Existing seed
    n8n-apify-integration.md       # Existing seed
    n8n-project-workflows.md       # Existing seed
    {new-n8n-pages}.md             # Auto-generated
  
  Gotchas/                         # NEW: Hard-won debugging knowledge
    {gotcha-pages}.md              # Auto-generated
  
  Workflows/                       # NEW: Step-by-step process guides
    {workflow-pages}.md            # Auto-generated
  
  Comparisons/                     # NEW: Technology evaluation records
    {comparison-pages}.md          # Auto-generated
```

### 5.2 Naming Conventions

| Page Type | Prefix/Pattern | Example |
|-----------|---------------|---------|
| Entity (API) | `{service}-{topic}.md` | `supabase-rls-patterns.md` |
| Architecture | `arch-{project-name}.md` | `arch-reddit-scheduler.md` |
| Pattern | `{domain}-patterns.md` | `auth-patterns.md` |
| Gotcha | `gotcha-{tool}-{topic}.md` | `gotcha-n8n-webhook-timing.md` |
| Workflow | `workflow-{process}.md` | `workflow-gke-deployment.md` |
| Comparison | `compare-{option-a}-vs-{option-b}.md` | `compare-bullmq-vs-pg-boss.md` |

### 5.3 MOC (Map of Content) Update Strategy

The `coding-knowledge-map.md` file serves as the wiki's index. It is auto-updated after each pipeline run:

1. Parse existing MOC sections (## headings)
2. For each new wiki page, determine which section it belongs to
3. Add entry as `- [[page-name]] -- one-line description`
4. If a page doesn't fit existing sections, create a new ## section
5. Preserve any manually-added entries (detected by absence from wiki_pages table)

---

## 6. PACT Integration Architecture

### 6.1 Continuous Update Flow

The wiki system hooks into PACT session completion to capture knowledge incrementally:

```
PACT Session Completes
        |
        v
+------------------+
| Session HANDOFF  |    (already happens — agents produce HANDOFFs)
| files written    |
+--------+---------+
         |
         v
+------------------+
| PACT Secretary   |    (already harvests HANDOFFs at phase boundaries)
| processes HANDOFF|
+--------+---------+
         |
         | (NEW) writes extraction candidates
         v
+------------------+
| Wiki Extraction  |
| Queue            |    SQLite table: pending_extractions
+--------+---------+
         |
         | (batch processed periodically)
         v
+------------------+
| Wiki Pipeline    |
| Stages 4-5 only  |    (skip Stages 1-3 — PACT HANDOFFs are pre-classified)
+------------------+
         |
         v
+------------------+
| Wiki Pages       |
| created/updated  |
+------------------+
```

### 6.2 Integration Points

| PACT Event | Wiki Action | Mechanism |
|------------|-------------|-----------|
| Session completion | JSONL file available for batch processing | Pipeline scheduled run picks it up |
| HANDOFF harvested by secretary | High-value extraction queued directly | Secretary writes to extraction queue |
| pact-memory updated | Re-cluster affected topics | Pipeline checks pact-memory timestamps |
| Agent memory updated | Include in next clustering pass | Pipeline scans agent-memory modified dates |
| Manual `/PACT:pin-memory` | May trigger wiki update if CLAUDE.md has new patterns | Pipeline checks CLAUDE.md modified date |

### 6.3 Schema Document Role

The schema document (`05-Education/Coding-Notes/wiki-schema.md`) governs LLM behavior when reading or writing wiki content. It is a markdown file in the vault that any LLM agent can reference.

**Schema document contents**:
- Page type definitions with examples
- Frontmatter field requirements
- Naming conventions
- Quality criteria (what makes a good wiki page)
- Anti-patterns (what NOT to write: generic advice, stale info, opinions without evidence)
- Wikilink conventions
- How to handle conflicts between new extractions and existing content

This document is read by the Synthesis stage (Stage 5) and by any PACT agent writing to the vault via the existing Obsidian Knowledge Base instructions in CLAUDE.md.

---

## 7. API Specifications

### 7.1 Pipeline CLI Interface

```bash
# Full batch run (process all unprocessed sessions)
python wiki_pipeline.py run --config pipeline_config.yaml

# Process specific project
python -m src.pipeline run --project "my-project"

# Dry run (classify only, report what would be extracted)
python wiki_pipeline.py dry-run --config pipeline_config.yaml

# Reprocess sessions (ignore processed_sessions state)
python wiki_pipeline.py reprocess --min-value 4

# Status report
python wiki_pipeline.py status

# Cost estimate before running
python wiki_pipeline.py estimate
```

### 7.2 Pipeline Module Interface

```python
# source_reader.py
class SourceReader:
    def read_sessions(self, project_filter: str | None = None) -> Iterator[Session]
    def read_pact_memory(self) -> list[Extraction]
    def read_agent_memory(self) -> list[Extraction]
    def read_project_memory(self) -> list[Extraction]

# classifier.py
class SessionClassifier:
    def classify(self, session: Session) -> Classification
    # Classification: { value_score, topics, page_types, skip_reason }

# extractor.py
class KnowledgeExtractor:
    def extract(self, session: Session, classification: Classification) -> list[Extraction]
    # Extraction: { type, topic, title, content, confidence, context }

# clusterer.py
class ExtractionClusterer:
    def cluster(self, extractions: list[Extraction], existing_pages: list[WikiPage]) -> list[PageCandidate]
    # PageCandidate: { action (create|update|merge), page_type, topic, extractions, target_path }

# synthesizer.py
class WikiSynthesizer:
    def synthesize(self, candidate: PageCandidate, schema: SchemaDocument) -> WikiPage
    # WikiPage: { path, frontmatter, body, wikilinks }

# wiki_writer.py
class WikiWriter:
    def write_page(self, page: WikiPage) -> None
    def update_moc(self, new_pages: list[WikiPage]) -> None

# state_store.py
class StateStore:
    def is_processed(self, session_id: str, pipeline_version: str) -> bool
    def record_processing(self, session_id: str, classification: Classification, extractions: list[Extraction]) -> None
    def record_page(self, page: WikiPage, extraction_ids: list[str]) -> None
    def record_run(self, run: PipelineRun) -> None
```

---

## 8. Technology Decisions

### ADR-001: Python for Pipeline Implementation

**Decision**: Use Python for the extraction pipeline.

**Rationale**:
- Native JSON/JSONL parsing (no dependencies for core data handling)
- sqlite3 in standard library (state store requires no external DB)
- Anthropic Python SDK is first-class
- asyncio for concurrent API calls
- User's existing environment has Python 3 installed

**Alternatives considered**:
- Node.js/TypeScript: Better match for n8n integration but weaker for data processing
- Shell scripts: Too brittle for multi-stage pipeline with state management

### ADR-002: SQLite for State Store (Not Supabase)

**Decision**: Use local SQLite for pipeline state tracking.

**Rationale**:
- Pipeline runs locally — no need for remote database
- Zero configuration, zero cost
- State is operational (tracking processed sessions), not shared knowledge
- Portable — state file moves with the project

### ADR-003: Claude Sonnet for Bulk, Opus for Synthesis

**Decision**: Use Sonnet for classification and extraction (Stages 2-3), Opus for synthesis (Stage 5).

**Rationale**:
- Classification and extraction are structured tasks with clear instructions — Sonnet handles these well at lower cost
- Synthesis requires editorial judgment, quality writing, and nuanced merging — Opus produces noticeably better wiki pages
- Cost optimization: ~80% of API calls are Sonnet (cheap), ~20% are Opus (quality-critical)

**Estimated cost for full initial run**:
- ~1,500 sessions past pre-filter
- ~500 sessions past classification (value >= 3)
- Classification: 1,500 calls x ~$0.005 = ~$7.50
- Extraction: 500 calls x ~$0.02 = ~$10.00
- Synthesis: ~100 page generations x ~$0.15 = ~$15.00
- **Total estimate: ~$32-40 for full initial distillation**

### ADR-004: Additive Wiki Updates (No Destructive Overwrites)

**Decision**: Wiki pages are updated additively. New extractions enrich existing content; nothing is deleted by the pipeline.

**Rationale**:
- User may have manually edited wiki pages — those edits must survive
- Prevents regression if extraction quality varies across runs
- Deletion requires human judgment (was the information wrong, or just restated?)

**Mechanism**: Synthesizer receives existing page content and is instructed to integrate, not replace.

### ADR-005: Wiki Lives in Existing Vault Structure

**Decision**: Wiki pages go in `05-Education/Coding-Notes/` using existing subfolder conventions. No top-level vault restructuring.

**Rationale**:
- 18 seed pages already exist there with established conventions
- Existing MOC (`coding-knowledge-map.md`) already indexes this location
- Adding new subfolders (Gotchas/, Workflows/, Comparisons/) extends naturally
- Avoids disrupting the user's established vault navigation habits

---

## 9. Security Considerations

| Concern | Mitigation |
|---------|------------|
| **API keys in sessions** | Pre-filter scans for credential patterns (`AKIA`, `sk-`, `ghp_`, etc.) and redacts before sending to LLM |
| **PII in sessions** | Sessions are processed locally; only extracted knowledge (not raw sessions) goes to API |
| **Cost runaway** | Hard cap per run (`max_cost_per_run_usd`), per-batch cost tracking, dry-run mode |
| **Vault corruption** | Wiki Writer uses atomic writes (write to temp, rename). Git provides rollback. |
| **Stale/wrong knowledge** | `wiki-confidence` frontmatter field; `needs_review` flag in state store; `(unverified)` markers in text |

---

## 10. Implementation Roadmap

### Phase 1: Foundation (MVP)

**Goal**: Process existing memory sources (pact-memory, agent-memory, project-memory) into wiki pages. These are already distilled and high-value — no JSONL processing yet.

**Deliverables**:
1. State store schema (SQLite)
2. Source reader for memory databases and markdown files
3. Clusterer for grouping memory entries by topic
4. Synthesizer for generating wiki pages from clusters
5. Wiki writer with frontmatter generation and MOC updating
6. Schema document in vault
7. CLI: `wiki_pipeline.py run --sources memory-only`

**Why start here**: Memory sources are small (< 5MB total), pre-classified, and high confidence. This validates the Cluster → Synthesize → Write chain without API cost risk.

**Expected output**: ~20-30 wiki pages enriching the existing 18 seed pages.

### Phase 2: Session Distillation

**Goal**: Process Claude Code JSONL sessions through the full five-stage pipeline.

**Deliverables**:
1. JSONL parser and session reader
2. Pre-filter implementation
3. Classifier (Stage 2) with LLM integration
4. Extractor (Stage 3) with chunking for large sessions
5. Cost estimator and dry-run mode
6. CLI: `wiki_pipeline.py run` (full pipeline)

**Processing strategy**: Start with the 10 highest-value projects (by size) to validate quality, then expand.

**Expected output**: ~50-80 additional wiki pages.

### Phase 3: PACT Integration

**Goal**: Connect the wiki pipeline to PACT session lifecycle for continuous updates.

**Deliverables**:
1. Extraction queue table in state store
2. Secretary hook: write extraction candidates after HANDOFF harvest
3. Incremental pipeline mode: process queue → cluster → synthesize
4. CLAUDE.md update: add wiki schema reference to Obsidian Knowledge Base section
5. CLI: `wiki_pipeline.py incremental`

**Expected behavior**: After each PACT session, new knowledge flows into the wiki within the next pipeline run.

### Phase 4: Quality and Maintenance

**Goal**: Quality assurance, staleness detection, and wiki health monitoring.

**Deliverables**:
1. Quality scoring for wiki pages (automated review pass)
2. Staleness detector: flag pages whose source sessions are old and whose content may be outdated
3. Duplicate detection: find wiki pages covering the same topic
4. Usage tracking: which pages are linked most/least (identify orphans)
5. Dashboard: `wiki_pipeline.py status` with page counts, quality distribution, staleness report

---

## 11. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **LLM extraction quality varies** | HIGH | MEDIUM | Confidence scoring, human review flags, Opus for synthesis |
| **Cost overrun on full JSONL processing** | MEDIUM | MEDIUM | Dry-run mode, per-run caps, start with memory-only (Phase 1) |
| **Wiki bloat (too many low-value pages)** | MEDIUM | LOW | min_extractions_for_new_page threshold, value score gate |
| **Existing seed pages overwritten** | LOW | HIGH | Additive-only updates, atomic writes, git rollback |
| **Session format changes** | LOW | MEDIUM | Parser version tracking, graceful handling of unknown message types |
| **PACT integration creates coupling** | LOW | MEDIUM | Wiki system is standalone; PACT hook is optional addon |
| **Obsidian plugin conflicts** | LOW | LOW | Standard markdown + YAML frontmatter; no special plugin dependencies |

---

## 12. Open Questions for Discussion

1. **Subagent session processing**: Should Phase 2 include subagent JSONL files, or defer to a later phase? They add ~2x volume but contain detailed implementation context.

2. **Existing seed page authorship**: The 18 seed pages were auto-generated today (2026-04-05). Should the pipeline treat them as "existing manual content" (preserve) or "initial extractions" (may revise)?

3. **Cross-project synthesis pages**: Should the pipeline create synthesis pages that span multiple projects (e.g., "Authentication Patterns Across Backend Services") or keep pages project-scoped?

4. **Wiki search**: Is grep-based search sufficient for the expected ~100-150 pages, or should we plan for an index/embedding layer?

5. **Scheduling**: Should the pipeline run on-demand only, or on a schedule (e.g., weekly batch + post-session incremental)?
