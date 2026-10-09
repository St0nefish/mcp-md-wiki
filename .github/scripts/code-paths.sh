#!/usr/bin/env bash
# Prints `true` when the change from <from> to <to> touches a code path, `false` when it
# does not. The ONE definition of "code path": ci-fast.yml (whether to run the cargo
# checks), slow.yml (whether to build and test) and master.yml (whether a merge needs an
# artifact of its own) all call this, so they cannot disagree.
#
# Code paths: anything compiled into, tested against, or building the binary or image —
#   src/**  assets/**  migrations/**  Cargo.toml  Cargo.lock  rust-toolchain.toml
#   Dockerfile  docker-compose.yml (the Qdrant pin CI and the smoke test run)
#   deploy/**  (deploy/config.example.yaml is read by a test through include_str!)
#   .github/workflows/**  .github/scripts/**
#
# One exception: a change to Cargo.toml / Cargo.lock that ONLY moves this crate's own
# `version` (the [package] version and Cargo.lock's own entry, which is what the
# post-release roll PR and a hand-made minor/major bump do) is not a code change. Such a
# PR lands with no build and its master commit gets no image, so releasing it fails
# closed (nothing new to release), and the next code change builds with the new version
# compiled in.
#
# Unlike the template, Markdown is NOT excluded: assets/mcp/*.md are compiled in
# (include_str!).
#
# Usage: code-paths.sh <from> <to>   (any commit-ish; needs both in the local clone)
set -euo pipefail
shopt -s inherit_errexit

from=${1:?usage: code-paths.sh <from> <to>}
to=${2:?usage: code-paths.sh <from> <to>}

code_re='^(src/|assets/|migrations/|deploy/|\.github/workflows/|\.github/scripts/)|^(Cargo\.toml|Cargo\.lock|rust-toolchain\.toml|Dockerfile|docker-compose\.yml)$'

# Cargo.toml with the [package] table's `version` value blanked.
normalize_manifest() {
  awk '
    /^\[/ { in_package = ($0 == "[package]") }
    in_package && /^version *=/ { print "version = \"<own version>\""; next }
    { print }
  '
}

# Cargo.lock with this crate's own `version` value blanked (the line after its
# `name = "mcp-md-wiki"` line), every other package untouched.
normalize_lock() {
  awk '
    own && /^version = / { print "version = \"<own version>\""; own = 0; next }
    { own = ($0 == "name = \"mcp-md-wiki\""); print }
  '
}

# Content of <path> at <rev>, or nothing when it does not exist there.
show() { git show "${1}:${2}" 2>/dev/null || true; }

version_only() {
  local path=$1 norm=$2
  [[ "$(show "$from" "$path" | "$norm")" == "$(show "$to" "$path" | "$norm")" ]]
}

changed=$(git diff --no-renames --name-only "$from" "$to")
code=false
while IFS= read -r path; do
  [[ -n "$path" && "$path" =~ $code_re ]] || continue
  case "$path" in
    Cargo.toml) version_only Cargo.toml normalize_manifest && continue ;;
    Cargo.lock) version_only Cargo.lock normalize_lock && continue ;;
  esac
  echo "code path changed: ${path}" >&2
  code=true
  break
done <<<"$changed"

echo "$code"
