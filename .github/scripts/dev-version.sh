#!/usr/bin/env bash
# Computes the dev-image version and tags for a commit on master.
#
#   version  <x.y.z>-dev.<n>: x.y.z is Cargo.toml's [package] version at the commit —
#            which on master is always the NEXT release, since the post-release roll PR
#            (release.yml's roll) bumps it — and n is the number of commits since the latest
#            release tag reachable from the commit (every commit, before the first
#            release). Semver-valid, and it sorts after the release it follows.
#   sha_tag  sha-<full commit sha>: the tag release.yml promotes.
#
# A commit whose Cargo.toml still names an already-released version (one merged after a
# release but before its roll PR) would get <released>-dev.<n>, which sorts BEFORE that
# release. For that commit x.y.z falls back to the release tag's patch + 1, with a warning.
#
# Needs full history and tags (actions/checkout with fetch-depth: 0). Read-only.
# Inputs (env): COMMIT (default HEAD).
# Outputs (stdout, and $GITHUB_OUTPUT when set): commit, version, sha_tag.
set -euo pipefail
shopt -s inherit_errexit

commit=$(git rev-parse --verify "${COMMIT:-HEAD}^{commit}")
semver='^([0-9]+)\.([0-9]+)\.([0-9]+)$'

# The `version` key of the `[package]` table, not the first `version =` line in the file
# (a dependency table written inline could otherwise match first).
base=$(git show "${commit}:Cargo.toml" | awk '
  /^\[/ { in_package = ($0 == "[package]") }
  in_package && /^version *=/ { sub(/^version *= *"/, ""); sub(/".*$/, ""); print; exit }
')
if [[ ! "$base" =~ $semver ]]; then
  echo "::error::Cargo.toml's [package] version '${base}' at ${commit} is not <major>.<minor>.<patch>"
  exit 1
fi

# Release tags only: v<x.y.z>, no pre-release suffix.
if tag=$(git describe --tags --abbrev=0 --match 'v[0-9]*' --exclude '*-*' "$commit" 2>/dev/null); then
  if [[ ! "${tag#v}" =~ $semver ]]; then
    echo "::error::release tag ${tag} is not v<major>.<minor>.<patch>"
    exit 1
  fi
  next="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.$((BASH_REMATCH[3] + 1))"
  if [[ "$(printf '%s\n' "$base" "${tag#v}" | sort -V | tail -n 1)" != "$base" || "$base" == "${tag#v}" ]]; then
    echo "::warning::Cargo.toml at ${commit} names ${base}, not later than the latest release ${tag}; the version roll has not landed yet, so this build is labelled ${next}."
    base=$next
  fi
  count=$(git rev-list --first-parent --count "${tag}..${commit}")
else
  count=$(git rev-list --first-parent --count "$commit")
fi

{
  echo "commit=${commit}"
  echo "version=${base}-dev.${count}"
  echo "sha_tag=sha-${commit}"
} | tee -a "${GITHUB_OUTPUT:-/dev/null}"
