# Install Guide

Reference for the full setup, both manual and scheduled. The audience already knows what `launchctl`, `pyenv`, and Python virtualenvs are — this is procedure, not tutorial.

## Prerequisites

- macOS 13+ (for the built-in launchd scheduling path). Linux users skip Cadence B and use `systemd-timer` or `cron` instead.
- Python 3.11 or newer.
- An Obsidian vault — any directory works; Obsidian is not actually required to run the pipeline, only to read the output.
- Optional: an Anthropic API key for LLM-generated MOC summaries.

## Quick setup

```bash
git clone git@github.com:v4lheru/obsidian-kb-pipeline.git
cd obsidian-kb-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -e .                          # core
# or:
pip install -e ".[llm]"                   # with Anthropic SDK for MOC summaries
cp .env.example .env
cp config/domains.example.yaml config/domains.yaml
```

## Configure

Edit `.env` and set `VAULT_ROOT` to the absolute path of your Obsidian vault root. The other env vars are optional — defaults are documented inline in `.env.example`.

Edit `config/domains.yaml` if you want custom project domain routing. For a first run, the default contents (a copy of `config/domains.example.yaml`) are fine.

The pipeline writes into `${VAULT_ROOT}/${VAULT_NOTES_SUBPATH}` (default subpath: `Coding-Notes`). It creates the subpath if it does not exist.

## Cadence A — manual / on demand

Two wrappers, both self-locating (no hardcoded paths):

```bash
# Single Claude Code project, fast iteration. ~every 4 hours during active development.
bash scripts/run-incremental.sh

# Full batch across all sources (sessions + memory). ~weekly.
bash scripts/run-weekly.sh
```

Direct invocation works too, from the repo root with the virtualenv active. After `pip install -e .` an `obsidian-kb-pipeline` console script is on `PATH`; both invocations are equivalent:

```bash
obsidian-kb-pipeline run --sources all
python -m src.pipeline run --sources all
```

`--sources` is one of `memory-only` (Phase 1), `sessions` (Phase 2), or `all`. Other useful flags: `--project NAME` (scope a sessions run to one Claude Code project directory), `--dry-run` (classify only, no writes). `obsidian-kb-pipeline status` inspects the state.db cache.

## Cadence B — scheduled via launchd (macOS)

The launchd job runs the weekly batch every Sunday at 9am local time. Install it:

```bash
bash scripts/install-launchd.sh
```

The installer reads `com.kbbrain.weekly-pipeline.plist.template` from the repo, substitutes `${REPO}` with the absolute path of your checkout, writes the result to `~/Library/LaunchAgents/com.kbbrain.weekly-pipeline.plist`, then loads it via `launchctl`.

Verify it loaded:

```bash
launchctl list | grep kbbrain
```

If you also want the 4-hour incremental cadence, add a second launchd job pointing at `scripts/run-incremental.sh`. The repo does not ship a second plist by default — the weekly cadence is the recommended baseline.

## Logs

| Path | What it contains |
|---|---|
| `logs/pipeline-YYYY-MM-DD.log` | Per-run pipeline log (created by the wrapper scripts). |
| `logs/launchd-stdout.log` | Stdout captured by launchd. |
| `logs/launchd-stderr.log` | Stderr captured by launchd. |

Tail the latest:

```bash
tail -f "logs/pipeline-$(date +%Y-%m-%d).log"
tail -f logs/launchd-stderr.log
```

The `logs/` directory itself is tracked (via a `.gitkeep`); its contents are gitignored.

## Verify it works

Run the manual incremental wrapper once and inspect the vault:

```bash
bash scripts/run-incremental.sh
ls -la "$VAULT_ROOT/Coding-Notes/"
```

If `state.db` was missing before this run, the pipeline processes every applicable session and memory entry on first invocation. Subsequent runs are incremental — only sessions and memory entries with timestamps newer than the cached watermark are processed.

## Uninstall scheduling

```bash
launchctl unload ~/Library/LaunchAgents/com.kbbrain.weekly-pipeline.plist
rm ~/Library/LaunchAgents/com.kbbrain.weekly-pipeline.plist
```

The repo itself stays put; only the LaunchAgent is removed.

## Troubleshooting

### "VAULT_ROOT not set"

The pipeline cannot find your Obsidian vault. Set `VAULT_ROOT` in `.env` to the absolute path of your vault root, then re-run. The wrapper scripts source `.env` automatically; if you are invoking `python -m src.pipeline run` directly, export the variable yourself.

### Launchd job is loaded but never runs

Check `logs/launchd-stderr.log` first. Common causes: Python 3.11+ not on launchd's `PATH`, the wrapper script lost its executable bit (`chmod +x scripts/*.sh`), or `.env` is missing from the repo root (the wrappers source it).

### "No module named 'src'"

You are not running from the repo root. The pipeline expects to be invoked as `python -m src.pipeline run` from a checkout that has `src/` and `config/` as siblings. The wrapper scripts cd to the right directory; if running Python directly, cd there yourself.

### "No config/domains.yaml found" (info-level log)

This is informational, not an error. The pipeline runs fine with built-in defaults. Copy `config/domains.example.yaml` to `config/domains.yaml` to silence the message and add your own taxonomy.

### Obsidian shows no new pages

The pipeline writes atomically. If you opened Obsidian before the writer finished, refresh the file explorer (`Cmd-R` in the Obsidian window). Also confirm the writer wrote where you expect: `ls "$VAULT_ROOT/Coding-Notes/"`.
