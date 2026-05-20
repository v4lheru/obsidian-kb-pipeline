# Project Memory

This file is the project-level CLAUDE.md for `obsidian-kb-pipeline`. Claude Code
agents read it before working on this repo. Add project-specific conventions
below.

## Conventions

- Tests use plain `unittest`; run via `python -m unittest discover`.
- Source taxonomy lives in `config/domains.yaml` (gitignored). Edit it, don't
  edit the constants in `src/`.
- The vault path is set via `.env` (`VAULT_ROOT`, `VAULT_NOTES_SUBPATH`).
