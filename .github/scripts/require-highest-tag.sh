#!/usr/bin/env bash
# Fails unless TAG is the highest stable v<x.y.z> tag on origin, so publishing an older
# (hotfix) release can never move :latest backwards. `git ls-remote` lists every tag in
# one answer (no API pagination to hide the true highest); an empty or failed listing
# fails closed.
# Usage: require-highest-tag.sh <tag>   Needs: a checkout whose `origin` is readable.
set -euo pipefail
shopt -s inherit_errexit

tag=${1:?usage: require-highest-tag.sh <tag>}
here=$(dirname "$0")

refs=$(git ls-remote --tags --refs origin 'refs/tags/v*')
highest=$("${here}/highest-release-tag.sh" <<<"$refs" || true)
if [[ -z "$highest" || "$tag" != "$highest" ]]; then
  echo "::error::${tag} is not the highest stable release tag (${highest:-none found}); refusing to ship it."
  exit 1
fi
echo "${tag} is the highest stable release tag."
