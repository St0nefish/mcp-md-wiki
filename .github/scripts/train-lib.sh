#!/usr/bin/env bash
# Shared helpers for train.yml's jobs, sourced (`. .github/scripts/train-lib.sh`), so the
# `select` and `finish` jobs agree on what "armed" and "untested" mean.
#
# Needs: gh (authenticated through GH_TOKEN), jq; env GITHUB_REPOSITORY, CTX_FAST,
# CTX_SLOW, and RUN_URL for post_status.

# Open PRs into master with auto-merge armed, oldest first: "<number> <head sha>" lines.
# Armed is how a PR joins the queue (auto-merge.yml arms the owner's PRs on open; the
# owner arming a contributor's PR is the approval), so an unarmed PR is never tested.
armed_prs() {
  gh api --paginate \
    "repos/${GITHUB_REPOSITORY}/pulls?state=open&base=master&sort=created&direction=asc&per_page=100" \
    --jq '.[] | select(.auto_merge != null) | "\(.number) \(.head.sha)"'
}

# The latest status of <context> on <sha>: "<state> <description>", or nothing when the
# commit has none. The combined-status endpoint already reduces to the newest per context.
ctx_status() {
  local sha=$1 ctx=$2
  gh api "repos/${GITHUB_REPOSITORY}/commits/${sha}/status?per_page=100" |
    jq -r --arg ctx "$ctx" '.statuses[] | select(.context == $ctx) | "\(.state) \(.description // "")"'
}

# True when <sha> carries neither train context: the PR's current head has not been
# through the train. A failure counts as tested — it is not retried until a push (a new
# head) or a manual `pr` dispatch.
untested() {
  local sha=$1
  [[ -z "$(ctx_status "$sha" "$CTX_FAST")" && -z "$(ctx_status "$sha" "$CTX_SLOW")" ]]
}

# post_status <sha> <context> <state> <description>. Every description the train posts
# starts with "train:" — the sweep in `select` tells train statuses apart by it. GitHub
# caps a description at 140 characters.
post_status() {
  local sha=$1 ctx=$2 state=$3 desc=$4
  gh api --silent -X POST "repos/${GITHUB_REPOSITORY}/statuses/${sha}" \
    -f state="$state" -f context="$ctx" -f description="${desc:0:140}" -f target_url="$RUN_URL"
  echo "${ctx} on ${sha:0:7}: ${state} (${desc})"
}

comment() {
  local pr=$1 body=$2
  gh api --silent -X POST "repos/${GITHUB_REPOSITORY}/issues/${pr}/comments" -f body="$body"
}
