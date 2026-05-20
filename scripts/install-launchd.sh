#!/usr/bin/env bash
# Install or uninstall the launchd jobs for the obsidian-kb-pipeline.
#
# Usage:
#   bash scripts/install-launchd.sh                    # install both (weekly + incremental)
#   bash scripts/install-launchd.sh --weekly-only      # install weekly only
#   bash scripts/install-launchd.sh --incremental-only # install incremental only
#   bash scripts/install-launchd.sh --uninstall        # unload + delete both
#
# Substitutes ${REPO} in the *.plist.template files with the absolute path of
# this repo, writes the result to ~/Library/LaunchAgents/, and loads each job.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENTS_DIR="${HOME}/Library/LaunchAgents"

WEEKLY_LABEL="com.kbbrain.weekly-pipeline"
INC_LABEL="com.kbbrain.incremental-pipeline"

WEEKLY_TPL="${REPO_DIR}/${WEEKLY_LABEL}.plist.template"
INC_TPL="${REPO_DIR}/${INC_LABEL}.plist.template"
WEEKLY_TGT="${AGENTS_DIR}/${WEEKLY_LABEL}.plist"
INC_TGT="${AGENTS_DIR}/${INC_LABEL}.plist"

mode="both"
for arg in "$@"; do
  case "${arg}" in
    --weekly-only)      mode="weekly" ;;
    --incremental-only) mode="incremental" ;;
    --uninstall)        mode="uninstall" ;;
    -h|--help)          sed -n '2,11p' "$0"; exit 0 ;;
    *)                  echo "Unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

install_one() {
  local tpl="$1" tgt="$2" label="$3"
  if [ ! -f "${tpl}" ]; then
    echo "Template not found: ${tpl}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${tgt}")"
  sed "s|\${REPO}|${REPO_DIR}|g" "${tpl}" > "${tgt}"
  launchctl unload "${tgt}" 2>/dev/null || true
  launchctl load "${tgt}"
  echo "Loaded: ${label}"
  echo "  Plist: ${tgt}"
}

uninstall_one() {
  local tgt="$1" label="$2"
  if [ -f "${tgt}" ]; then
    launchctl unload "${tgt}" 2>/dev/null || true
    rm -f "${tgt}"
    echo "Uninstalled: ${label}"
  else
    echo "Not installed: ${label} (skip)"
  fi
}

case "${mode}" in
  both)
    install_one "${WEEKLY_TPL}" "${WEEKLY_TGT}" "${WEEKLY_LABEL}"
    install_one "${INC_TPL}"    "${INC_TGT}"    "${INC_LABEL}"
    ;;
  weekly)
    install_one "${WEEKLY_TPL}" "${WEEKLY_TGT}" "${WEEKLY_LABEL}"
    ;;
  incremental)
    install_one "${INC_TPL}" "${INC_TGT}" "${INC_LABEL}"
    ;;
  uninstall)
    uninstall_one "${WEEKLY_TGT}" "${WEEKLY_LABEL}"
    uninstall_one "${INC_TGT}"    "${INC_LABEL}"
    ;;
esac

echo "Repo: ${REPO_DIR}"
