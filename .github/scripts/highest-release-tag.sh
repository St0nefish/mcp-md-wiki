#!/usr/bin/env bash
# Prints the highest stable release tag, v<x.y.z> with no suffix, compared as versions.
# With `--below <tag>`, prints the highest stable release tag strictly lower than <tag>
# instead (the previous release).
#
# Input (stdin): tag names or refs, one per line — `refs/tags/` prefixes, `git ls-remote`
#   columns and `^{}` peel lines are accepted. Every line is read; nothing is paginated.
# Output: the tag; nothing (and exit 1) when there is none, so a caller fails closed or
#   handles "no previous release" explicitly.
#
# Usage:
#   git ls-remote --tags --refs origin 'refs/tags/v*' | highest-release-tag.sh
#   git for-each-ref --format='%(refname)' 'refs/tags/v*' | highest-release-tag.sh --below v1.2.3
set -euo pipefail
shopt -s inherit_errexit

below=""
if (($# > 0)); then
  if [[ "$1" != "--below" || $# -ne 2 ]]; then
    echo "usage: highest-release-tag.sh [--below v<x.y.z>]" >&2
    exit 2
  fi
  below=$2
  if [[ ! "$below" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "::error::highest-release-tag.sh: '${below}' is not v<major>.<minor>.<patch>" >&2
    exit 2
  fi
fi

# Stable tags, version-sorted ascending, deduplicated. With --below, <tag> itself joins the
# list so the answer is simply the entry just before it.
sorted=$({
  sed -E 's/^[0-9a-f]+[[:space:]]+//; s#^refs/tags/##'
  if [[ -n "$below" ]]; then printf '%s\n' "$below"; fi
} | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' |
  sort -u -t. -k1.2,1n -k2,2n -k3,3n || true)

if [[ -n "$below" ]]; then
  result=$(awk -v b="$below" '$0 == b { print prev; exit } { prev = $0 }' <<<"$sorted")
else
  result=$(tail -n 1 <<<"$sorted")
fi
if [[ -z "$result" ]]; then
  exit 1
fi
printf '%s\n' "$result"
