#!/usr/bin/env bash
# Commit-status helpers shared by slow.yml, master.yml and release.yml; sourced
# (`. .github/scripts/status-lib.sh`). Needs gh (GH_TOKEN), jq, GITHUB_REPOSITORY.
#
# Every status this pipeline relies on is posted with the job's GITHUB_TOKEN, so its
# creator is `github-actions[bot]` (the ruleset pins the required checks to GitHub
# Actions). Readers ignore a status from any other creator: anyone with write access can
# post a status under any context.

STATUS_CREATOR="github-actions[bot]"

# post_status <sha> <context> <state> <description>   (GitHub caps descriptions at 140)
post_status() {
  local sha=$1 ctx=$2 state=$3 desc=$4
  gh api --silent -X POST "repos/${GITHUB_REPOSITORY}/statuses/${sha}" \
    -f state="$state" -f context="$ctx" -f description="${desc:0:140}" \
    -f target_url="${RUN_URL:-${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}}"
  echo "${ctx} on ${sha:0:7}: ${state} (${desc})"
}

# The newest <context> status on <sha> posted by STATUS_CREATOR, as
# "<state> <description>"; nothing when there is none. The list endpoint is newest first.
latest_status() {
  local sha=$1 ctx=$2
  gh api --paginate "repos/${GITHUB_REPOSITORY}/commits/${sha}/statuses?per_page=100" |
    jq -rs --arg ctx "$ctx" --arg who "$STATUS_CREATOR" \
      '[.[][] | select(.context == $ctx and .creator.login == $who)] | first // empty
       | "\(.state) \(.description // "")"'
}

# True when <head> carries a `ci-slow` success that names <tree>: that exact tree passed
# the slow tier. slow.yml writes "... tree <full tree hash>" into the description.
tested_tree() {
  local head=$1 tree=$2 st
  st=$(latest_status "$head" ci-slow)
  [[ "$st" == success* && "$st" == *"tree ${tree}"* ]]
}

comment() {
  local pr=$1 body=$2
  gh api --silent -X POST "repos/${GITHUB_REPOSITORY}/issues/${pr}/comments" -f body="$body"
}
