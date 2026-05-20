#!/bin/bash
# Incremental pipeline runner — memory-only, intended for a 4-hour cadence.
#
# Invoked by launchd (com.kbbrain.incremental-pipeline.plist) every 4 hours.
# Can also be run manually:
#     bash scripts/run-incremental.sh
#
# Exits 0 even on python failure so launchd does not mark the job as crashed.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_DIR}/logs"
LOG_FILE="${LOG_DIR}/incremental-$(date +%Y-%m-%d).log"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

log() {
  echo "[${TIMESTAMP}] $1" >> "${LOG_FILE}"
}

mkdir -p "${LOG_DIR}"

# Add common PATH locations (launchd starts with a minimal PATH).
for dir in /opt/homebrew/bin /usr/local/bin; do
  if [ -d "${dir}" ] && [[ ":$PATH:" != *":${dir}:"* ]]; then
    PATH="${dir}:$PATH"
  fi
done
export PATH

# Source .env if present.
if [ -f "${REPO_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "${REPO_DIR}/.env"
  set +a
fi

if ! command -v python3 >/dev/null 2>&1; then
  log "ERROR: python3 not found in PATH (${PATH})"
  exit 0
fi

if [ ! -d "${REPO_DIR}" ]; then
  log "ERROR: Repo directory not found: ${REPO_DIR}"
  exit 0
fi

cd "${REPO_DIR}" || { log "ERROR: Failed to cd to ${REPO_DIR}"; exit 0; }

log "Starting incremental pipeline run..."

if python3 -m src.pipeline run --sources memory-only >> "${LOG_FILE}" 2>&1; then
  DONE_TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
  echo "[${DONE_TIMESTAMP}] Pipeline run completed successfully" >> "${LOG_FILE}"
else
  EXIT_CODE=$?
  DONE_TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
  echo "[${DONE_TIMESTAMP}] Pipeline run failed with exit code ${EXIT_CODE}" >> "${LOG_FILE}"
  exit 0
fi
