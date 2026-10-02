#!/usr/bin/env bash
# Squashes a PR head onto a pinned master commit, deterministically, in the current
# checkout. The merge train (train.yml) tests exactly this tree, and every job that needs
# it — on whichever runner — rebuilds it with this script rather than receiving it as an
# artifact. Fixed author, committer, dates and message make the commit itself
# reproducible, not just the tree: two runs on the same <base> and <head> produce the
# same commit SHA, so jobs on different runners provably test the same thing.
#
# What lands on master is GitHub's squash merge of the PR, whose TREE equals the tree of
# this local `git merge --squash` onto the same master (verified live; see the knowledge
# base's dev/tools/merge-train-pattern.md). The commit SHA differs (GitHub's author,
# dates and message), which is why images are named after the tree, never the commit.
#
# Usage: train-squash.sh <base-sha> <head-sha> [expected-tree]
#   base-sha       the master commit to squash onto (full 40-hex sha)
#   head-sha       the PR head commit (full 40-hex sha); fetched by sha, or from
#                  refs/pull/*/head, if this clone does not have it
#   expected-tree  optional: fail unless the squash's tree equals it (every test job
#                  passes the tree the train's `select` job pinned)
#
# Leaves HEAD detached at the squash commit and prints its tree hash on stdout; all
# diagnostics go to stderr. Exit 2: the head does not squash cleanly onto base (a merge
# conflict). Any other non-zero exit is an error.
#
# DESTRUCTIVE to the working copy: it aborts any in-progress merge/rebase, hard-resets
# and runs `git clean -ffdx`, because a persistent self-hosted runner's workspace may hold
# a previous run's half-finished state. Run it in a CI checkout or a scratch clone, never
# in a working copy you care about.
set -euo pipefail
shopt -s inherit_errexit

# Everything lives in main(), called on the last line: bash reads a script as it runs it,
# and the checkout/merge below can rewrite this very file when the PR edits it. Parsing
# the whole body first means the run finishes as the version it started as.
main() {
  local base=${1:-} head=${2:-} expected=${3:-}
  local sha_re='^[0-9a-f]{40}$'
  if [[ ! "$base" =~ $sha_re || ! "$head" =~ $sha_re ]]; then
    echo "::error::usage: train-squash.sh <base-sha> <head-sha> [expected-tree] (full 40-hex shas; got '${base}' '${head}')" >&2
    exit 1
  fi

  # Fetch only what is missing. A head from a fork is on no branch of this repo; GitHub
  # serves any commit by sha (actions/checkout fetches that way too), and refs/pull/*/head
  # is the fallback that always holds every PR's head.
  local sha
  for sha in "$base" "$head"; do
    if ! git cat-file -e "${sha}^{commit}" 2>/dev/null; then
      echo "fetching ${sha}" >&2
      git fetch --quiet --no-tags origin "$sha" >&2 ||
        git fetch --quiet --no-tags origin '+refs/pull/*/head:refs/remotes/origin/pull/*' >&2
      git cat-file -e "${sha}^{commit}" || {
        echo "::error::commit ${sha} is not available from origin" >&2
        exit 1
      }
    fi
  done

  # Persistent-runner hygiene: a previous run may have died mid-merge or mid-rebase.
  git merge --abort >/dev/null 2>&1 || true
  git rebase --abort >/dev/null 2>&1 || true
  rm -rf .git/rebase-merge .git/rebase-apply
  git reset --quiet --hard
  git clean -ffdxq
  git checkout --quiet --force --detach "$base"

  # Pin every identity, date and message input the commit object depends on, and override
  # the runner settings that could change the commit or the tree (signing, hooks, line
  # ending conversion). The global config is otherwise left alone: a containerized
  # runner may need its safe.directory entry to use this checkout at all.
  local -a git_det=(-c commit.gpgsign=false -c core.hooksPath=/dev/null -c core.autocrlf=false)
  export GIT_AUTHOR_NAME="merge train" GIT_AUTHOR_EMAIL="train@invalid"
  export GIT_COMMITTER_NAME="merge train" GIT_COMMITTER_EMAIL="train@invalid"
  export GIT_AUTHOR_DATE="@0 +0000" GIT_COMMITTER_DATE="@0 +0000"

  if ! git "${git_det[@]}" merge --squash --quiet "$head" >&2; then
    echo "::error::${head} does not squash cleanly onto ${base} (merge conflict)." >&2
    git diff --name-only --diff-filter=U >&2 || true
    git reset --quiet --hard
    exit 2
  fi
  # --allow-empty: a head whose changes are already on master squashes to nothing. That
  # is still a well-defined tree (base's), and the train lands it like any other.
  git "${git_det[@]}" commit --quiet --allow-empty --no-verify \
    -m "train: squash ${head} onto ${base}" >&2

  local tree
  tree=$(git rev-parse 'HEAD^{tree}')
  echo "squash of ${head} onto ${base}: commit $(git rev-parse HEAD), tree ${tree}" >&2
  if [[ -n "$expected" && "$tree" != "$expected" ]]; then
    echo "::error::squash tree ${tree} differs from the tree the train pinned (${expected}); refusing to test a different tree." >&2
    exit 1
  fi
  echo "$tree"
}

main "$@"
