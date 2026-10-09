#!/usr/bin/env bash
# Prints the digest of <image>:sha-<commit> — the tested image master.yml attached to a
# master commit — waiting (bounded) while a master.yml run for that commit is still
# queued or in progress. Prints nothing when there is no such image and no run that could
# still make one, so the caller decides: release.yml fails closed, master.yml's
# docs-only inheritance skips.
#
# Usage: wait-image.sh <image> <commit>
# Env: GH_TOKEN, GITHUB_REPOSITORY; WAIT_TIMEOUT seconds (default 1200).
set -euo pipefail
shopt -s inherit_errexit

image=${1:?usage: wait-image.sh <image> <commit>}
commit=${2:?usage: wait-image.sh <image> <commit>}
here=$(dirname "$0")
deadline=$((SECONDS + ${WAIT_TIMEOUT:-1200}))

master_run_active() {
  local n
  n=$(gh run list --repo "$GITHUB_REPOSITORY" --workflow master.yml --commit "$commit" \
    --json status --jq '[.[] | select(.status != "completed")] | length')
  [[ "$n" != "0" ]]
}

while :; do
  digest=$("${here}/image-digest.sh" "${image}:sha-${commit}")
  if [[ -n "$digest" ]]; then
    echo "$digest"
    exit 0
  fi
  if ! master_run_active; then
    echo "no ${image}:sha-${commit}, and no master.yml run for ${commit} is still running." >&2
    exit 0
  fi
  if ((SECONDS >= deadline)); then
    echo "::error::timed out waiting for master.yml to finish on ${commit}." >&2
    exit 1
  fi
  echo "waiting for master.yml on ${commit} to attach :sha-${commit}" >&2
  sleep 20
done
