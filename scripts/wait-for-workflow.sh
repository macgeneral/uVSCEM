#!/usr/bin/env bash
# Usage: wait-for-workflow.sh <workflow> <sha> <repo> [attempts=50] [interval=30]
set -euo pipefail

WORKFLOW="${1:?Usage: wait-for-workflow.sh <workflow> <sha> <repo> [attempts] [interval]}"
SHA="${2:?Missing SHA}"
REPO="${3:?Missing repo}"
ATTEMPTS="${4:-50}"
INTERVAL="${5:-30}"

echo "Waiting for '${WORKFLOW}' workflow to complete for ${SHA}..."

for attempt in $(seq 1 "${ATTEMPTS}"); do
  STATUS=$(gh api "/repos/${REPO}/actions/runs?head_sha=${SHA}" \
    --jq "[.workflow_runs[] | select(.name == \"${WORKFLOW}\")]
      | if map(select(.conclusion == \"success\")) | length > 0 then \"success\"
        elif map(select(.status == \"completed\" and .conclusion != \"success\")) | length > 0 then \"failure\"
        else \"pending\"
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
