#!/bin/bash
# Weekly pipeline runner — full batch across all sources.
#
# Invoked by launchd (com.kbbrain.weekly-pipeline.plist) once per week.
# Can also be run manually:
#     bash scripts/run-weekly.sh

set -u
set -o pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_DIR}/logs"
LOG_FILE="${LOG_DIR}/pipeline-$(date +%Y-%m-%d).log"

# launchd starts with a minimal PATH; add common locations so python3 resolves.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

mkdir -p "${LOG_DIR}"

cd "${REPO_DIR}" || {
  echo "[$(date -Iseconds)] FATAL: cannot cd into ${REPO_DIR}" >> "${LOG_FILE}"
  exit 1
}

# Source .env if present (sets VAULT_ROOT and any optional vars).
if [ -f "${REPO_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "${REPO_DIR}/.env"
  set +a
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] FATAL: python3 not found in PATH (${PATH})" >> "${LOG_FILE}"
  exit 1
fi

{
  echo "==========================================="
  echo "[$(date -Iseconds)] Weekly pipeline run starting"
  echo "PWD: $(pwd)"
  echo "Python: $(python3 --version 2>&1)"
  echo "==========================================="
} >> "${LOG_FILE}"

python3 -m src.pipeline run --sources all >> "${LOG_FILE}" 2>&1
exit_code=$?

{
  echo "==========================================="
  echo "[$(date -Iseconds)] Weekly pipeline run finished (exit=${exit_code})"
  echo "==========================================="
} >> "${LOG_FILE}"

exit "${exit_code}"
