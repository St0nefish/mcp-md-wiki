#!/usr/bin/env bash
# Computes the dev-image version and tags for a commit on master.
#
#   version  <x.y.z>-dev.<n>: x.y.z is the latest release tag reachable from the commit
#            with its patch number bumped, n the number of commits since that tag. With no
#            release tag yet, x.y.z is Cargo.toml's [package] version at the commit and n
#            counts every commit. Semver-valid, and it sorts after the release it follows.
#   sha_tag  sha-<full commit sha>: the tag release.yml resolves to find this build.
#
# Needs full history and tags (actions/checkout with fetch-depth: 0). Read-only.
# Inputs (env): COMMIT (default HEAD).
# Outputs (stdout, and $GITHUB_OUTPUT when set): commit, version, sha_tag.
set -euo pipefail
shopt -s inherit_errexit

commit=$(git rev-parse --verify "${COMMIT:-HEAD}^{commit}")

# Release tags only: v<x.y.z>, no pre-release suffix.
if tag=$(git describe --tags --abbrev=0 --match 'v[0-9]*' --exclude '*-*' "$commit" 2>/dev/null); then
  if [[ ! "${tag#v}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    echo "::error::release tag ${tag} is not v<major>.<minor>.<patch>"
    exit 1
  fi
  base="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.$((BASH_REMATCH[3] + 1))"
  count=$(git rev-list --count "${tag}..${commit}")
else
  # The `version` key of the `[package]` table, not the first `version =` line in the
  # file (a dependency table written inline could otherwise match first).
  base=$(git show "${commit}:Cargo.toml" | awk '
    /^\[/ { in_package = ($0 == "[package]") }
    in_package && /^version *=/ { sub(/^version *= *"/, ""); sub(/".*$/, ""); print; exit }
  ')
  if [[ ! "$base" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "::error::no release tag, and Cargo.toml's [package] version '${base}' is not <major>.<minor>.<patch>"
    exit 1
  fi
  count=$(git rev-list --count "$commit")
fi

{
  echo "commit=${commit}"
  echo "version=${base}-dev.${count}"
  echo "sha_tag=sha-${commit}"
} | tee -a "${GITHUB_OUTPUT:-/dev/null}"
