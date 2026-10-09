#!/usr/bin/env bash
# Merges a PR head onto a pinned master commit, deterministically, in the current
# checkout: the tree `ci-slow` tests. GitHub's merge commit of the same head onto the
# same master has the same tree, so master.yml finds the tested build by the merge
# commit's tree (`:tree-<tree>`) and the `ci-slow` status that names it.
#
# Usage: merge-tree.sh <base-sha> <head-sha> [expected-tree]
#   base-sha       master commit to merge onto (full sha)
#   head-sha       the PR head (full sha); fetched by sha, or from refs/pull/*/head
#   expected-tree  optional: fail unless the merge's tree equals it
#
# Leaves HEAD detached at the merge commit and prints its tree hash on stdout;
# diagnostics go to stderr. Exit 2: the head does not merge cleanly (a conflict). Any
# other non-zero exit is an error. DESTRUCTIVE to the working copy (hard reset, clean):
# run it in a CI checkout, never in a working copy you care about.
set -euo pipefail
shopt -s inherit_errexit

# All in main(), called on the last line: the merge can rewrite this very file when the
# PR edits it, and bash reads a script as it runs it.
main() {
  local base=${1:-} head=${2:-} expected=${3:-}
  local sha_re='^[0-9a-f]{40}$'
  if [[ ! "$base" =~ $sha_re || ! "$head" =~ $sha_re ]]; then
    echo "::error::usage: merge-tree.sh <base-sha> <head-sha> [expected-tree] (full shas; got '${base}' '${head}')" >&2
    exit 1
  fi

  local sha
  for sha in "$base" "$head"; do
    if ! git cat-file -e "${sha}^{commit}" 2>/dev/null; then
      git fetch --quiet --no-tags origin "$sha" >&2 ||
        git fetch --quiet --no-tags origin '+refs/pull/*/head:refs/remotes/origin/pull/*' >&2
      git cat-file -e "${sha}^{commit}" || {
        echo "::error::commit ${sha} is not available from origin" >&2
        exit 1
      }
    fi
  done

  git merge --abort >/dev/null 2>&1 || true
  git reset --quiet --hard
  git clean -ffdxq
  git checkout --quiet --force --detach "$base"

  # Pinned identity and dates; no signing, hooks, line-ending conversion or rerere (which
  # would silently replay a recorded resolution instead of reporting the conflict).
  local -a git_det=(-c commit.gpgsign=false -c core.hooksPath=/dev/null -c core.autocrlf=false
    -c rerere.enabled=false)
  export GIT_AUTHOR_NAME="ci-slow" GIT_AUTHOR_EMAIL="ci-slow@invalid"
  export GIT_COMMITTER_NAME="ci-slow" GIT_COMMITTER_EMAIL="ci-slow@invalid"
  export GIT_AUTHOR_DATE="@0 +0000" GIT_COMMITTER_DATE="@0 +0000"

  if ! git "${git_det[@]}" merge --no-ff --no-edit --quiet -m "ci-slow: merge ${head} onto ${base}" "$head" >&2; then
    echo "::error::${head} does not merge cleanly onto ${base} (conflict)." >&2
    git diff --name-only --diff-filter=U >&2 || true
    git merge --abort >/dev/null 2>&1 || true
    git reset --quiet --hard
    exit 2
  fi

  # Drop the token actions/checkout persisted: the code under test must not read it.
  git config --local --unset-all 'http.https://github.com/.extraheader' >/dev/null 2>&1 || true

  local tree
  tree=$(git rev-parse 'HEAD^{tree}')
  echo "merge of ${head} onto ${base}: commit $(git rev-parse HEAD), tree ${tree}" >&2
  if [[ -n "$expected" && "$tree" != "$expected" ]]; then
    echo "::error::merge tree ${tree} differs from the pinned ${expected}; refusing to test a different tree." >&2
    exit 1
  fi
  echo "$tree"
}

main "$@"
