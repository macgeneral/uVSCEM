#!/usr/bin/env bash
#
# Mode 1 — skip decision:
#   check-build-needed.sh <sha> <repo> <force> <workflow> <pattern> [pattern...]
#   Writes skip=true|false to $GITHUB_OUTPUT.
#   Skips if: no relevant files changed, or a successful <workflow> run already
#   exists for the SHA. Set <workflow> to "none" or "" to skip the API check.
#   force=true bypasses both checks.
#
# Mode 2 — find run ID:
#   check-build-needed.sh --find-run-id <repo> <workflow> <pattern> [pattern...]
#   Walks git log backwards from HEAD, finds the most recent commit that touched
#   a pattern-matching file and has a successful <workflow> run, then writes
#   run-id=<id> to $GITHUB_OUTPUT.
#
set -euo pipefail

# ── Helpers ──────────────────────────────────────────────────────────────────

# Returns 0 if the given SHA touched any file matching the supplied patterns.
commit_touches_patterns() {
  local sha="$1"; shift
  local patterns=("$@")
  local file p
  while IFS= read -r file; do
    for p in "${patterns[@]}"; do
      # shellcheck disable=SC2254
      case "${file}" in ${p}) return 0 ;; esac
    done
  done <<< "$(git show --name-only --pretty="" "${sha}")"
  return 1
}

# Prints the run ID of the latest successful <workflow> run for <sha>, or "".
query_run_id() {
  local sha="$1" repo="$2" workflow="$3"
  gh api "/repos/${repo}/actions/runs?head_sha=${sha}&status=success" \
    --jq '[.workflow_runs[] | select(.name == "'"${workflow}"'")] | .[0].id // empty'
}

# ── Mode 2: --find-run-id ────────────────────────────────────────────────────

cmd_find_run_id() {
  local repo="$1" workflow="$2"; shift 2
  local patterns=("$@")
  local sha run_id

  git fetch --all --quiet
  while IFS= read -r sha; do
    if commit_touches_patterns "${sha}" "${patterns[@]}"; then
      run_id=$(query_run_id "${sha}" "${repo}" "${workflow}")
      if [[ -n "${run_id}" ]]; then
        echo "Found successful ${workflow} run ${run_id} at ${sha}."
        echo "run-id=${run_id}" >> "$GITHUB_OUTPUT"
        return 0
      fi
      echo "Commit ${sha} touched relevant files but has no successful ${workflow} run yet."
    fi
  done <<< "$(git log --format=%H)"

  echo "No successful ${workflow} run found in history for the given patterns." >&2
  return 1
}

# ── Mode 1: skip decision ────────────────────────────────────────────────────

output_skip()  { echo "skip=true"  >> "$GITHUB_OUTPUT"; echo "$1"; }
output_build() { echo "skip=false" >> "$GITHUB_OUTPUT"; echo "$1"; }

cmd_decide_skip() {
  local sha="$1" repo="$2" force="$3" workflow="$4"; shift 4
  local patterns=("$@")
  local run_id

  if [[ "${force}" == 'true' ]]; then
    output_build "Force rebuild requested."
    return 0
  fi

  git fetch --all --quiet
  if ! commit_touches_patterns "${sha}" "${patterns[@]}"; then
    output_skip "No relevant file changes, skipping build."
    return 0
  fi

  if [[ -n "${workflow}" && "${workflow}" != 'none' ]]; then
    run_id=$(query_run_id "${sha}" "${repo}" "${workflow}")
    if [[ -n "${run_id}" ]]; then
      output_skip "Existing successful ${workflow} run ${run_id} found for SHA ${sha}, skipping."
      return 0
    fi
  fi

  output_build "Relevant changes detected and no prior successful build found, proceeding."
}

# ── Dispatch ─────────────────────────────────────────────────────────────────

if [[ "${1:-}" == '--find-run-id' ]]; then
  shift
  cmd_find_run_id "$@"
else
  cmd_decide_skip "$@"
fi
