#!/usr/bin/env bash
# Usage: wait-for-workflow.sh <workflow> <sha> <repo> [attempts=50] [interval=30] [min-created-at=""]
# If min-created-at is provided (ISO-8601 UTC, e.g. 2026-02-26T12:34:56Z),
# only workflow runs created at or after that timestamp are considered.
set -euo pipefail

WORKFLOW="${1:?Usage: wait-for-workflow.sh <workflow> <sha> <repo> [attempts] [interval] [min-created-at]}"
SHA="${2:?Missing SHA}"
REPO="${3:?Missing repo}"
ATTEMPTS="${4:-50}"
INTERVAL="${5:-30}"
MIN_CREATED_AT="${6:-}"

echo "Waiting for '${WORKFLOW}' workflow to complete for ${SHA}..."
if [[ -n "${MIN_CREATED_AT}" ]]; then
  echo "Considering only runs created at or after ${MIN_CREATED_AT}."
fi

for attempt in $(seq 1 "${ATTEMPTS}"); do
  STATUS=$(gh api "/repos/${REPO}/actions/runs?head_sha=${SHA}" \
    --jq "[.workflow_runs[]
      | select(
          .name == \"${WORKFLOW}\"
          and (\"${MIN_CREATED_AT}\" == \"\" or .created_at >= \"${MIN_CREATED_AT}\")
        )]
      | if length == 0 then \"pending\"
        else (sort_by(.created_at) | .[-1]) as \$latest
        | if \$latest.status != \"completed\" then \"pending\"
          elif \$latest.conclusion == \"success\" then \"success\"
          else \"failure\"
          end
        end")

  if [[ "${STATUS}" == "success" ]]; then
    echo "'${WORKFLOW}' workflow succeeded."
    exit 0
  elif [[ "${STATUS}" == "failure" ]]; then
    echo "'${WORKFLOW}' workflow failed for commit ${SHA}."
    exit 1
  fi

  echo "Attempt ${attempt}/${ATTEMPTS}: '${WORKFLOW}' not yet complete, retrying in ${INTERVAL}s..."
  sleep "${INTERVAL}"
done

echo "Timed out waiting for '${WORKFLOW}' workflow to complete."
exit 1
