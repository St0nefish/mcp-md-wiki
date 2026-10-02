#!/usr/bin/env bash
# The post-release version roll, applied to the working copy (release.yml's `roll` job
# commits the result and opens it as a PR). On master Cargo.toml always names the NEXT
# release, so once <released> has shipped this moves it to <next>:
#
#   Cargo.toml    [package] version  <released> -> <next>
#   Cargo.lock    this crate's own entry, likewise (no other package is touched, so no
#                 registry access is needed)
#   CHANGELOG.md  `## [Unreleased]` becomes `## [<released>] - <date>` under a fresh,
#                 empty `## [Unreleased]`; the `[Unreleased]:` link reference moves to
#                 compare from v<released>, and a `[<released>]:` reference comparing it
#                 with the previous release is added below it.
#
# The released section's text is the `[Unreleased]` section AT THE RELEASED COMMIT
# (<notes-file>), which is what the release notes were made from. When master's own
# `[Unreleased]` section has moved on since (a PR that landed between the released commit
# and the release added entries), the two cannot be told apart mechanically: the released
# section gets <notes-file> verbatim, master's whole `[Unreleased]` text is KEPT under the
# fresh heading, and the script exits 3 so the caller leaves the PR for the owner to
# reconcile instead of letting it land unattended.
#
# Usage: roll-version.sh <released> <next> <date> <notes-file>
# Exit 0: rolled. Exit 3: rolled, but CHANGELOG.md needs a human (see above). Else: error,
# with the files possibly partly edited (the caller works on a throwaway checkout).
# shellcheck disable=SC2016 # the single-quoted programs are awk's; $ is awk syntax
set -euo pipefail
shopt -s inherit_errexit

released=${1:?usage: roll-version.sh <released> <next> <date> <notes-file>}
next=${2:?usage: roll-version.sh <released> <next> <date> <notes-file>}
date=${3:?usage: roll-version.sh <released> <next> <date> <notes-file>}
notes=${4:?usage: roll-version.sh <released> <next> <date> <notes-file>}
here=$(dirname "$0")

for v in "$released" "$next"; do
  [[ "$v" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
    echo "::error::'${v}' is not <major>.<minor>.<patch>" >&2
    exit 1
  }
done
if grep -qF "## [${released}]" CHANGELOG.md; then
  echo "::error::CHANGELOG.md already has a '## [${released}]' section; nothing to roll." >&2
  exit 1
fi

# Rewrites <file> through an awk program, in place (same inode, so mode and owner stay).
rewrite() {
  local file=$1
  shift
  local out
  out=$(awk "$@" "$file")
  printf '%s\n' "$out" >"$file"
}

# Cargo.toml: exactly one [package] version line, and it must name <released>.
rewrite Cargo.toml -v from="$released" -v to="$next" '
  /^\[/ { in_package = ($0 == "[package]") }
  in_package && /^version *=/ {
    if ($0 !~ "^version *= *\"" from "\"") { print "unexpected [package] " $0 > "/dev/stderr"; exit 1 }
    print "version = \"" to "\""; n++; next
  }
  { print }
  END { if (n != 1) { print "found " n " [package] version lines" > "/dev/stderr"; exit 1 } }
'

# Cargo.lock: the version line directly after this crate's own name line.
rewrite Cargo.lock -v from="$released" -v to="$next" '
  own && /^version = / {
    if ($0 != "version = \"" from "\"") { print "unexpected Cargo.lock entry " $0 > "/dev/stderr"; exit 1 }
    print "version = \"" to "\""; own = 0; n++; next
  }
  { own = ($0 == "name = \"mcp-md-wiki\""); print }
  END { if (n != 1) { print "found " n " mcp-md-wiki entries in Cargo.lock" > "/dev/stderr"; exit 1 } }
'

# CHANGELOG.md. Does master's [Unreleased] still read exactly as the released one did?
current=$("$here/changelog-section.sh" Unreleased CHANGELOG.md 2>/dev/null || true)
diverged=0
if [[ "$current" != "$(cat "$notes")" ]]; then
  diverged=1
  echo "::warning::CHANGELOG.md's [Unreleased] section on master differs from the released one; the roll keeps master's text under [Unreleased] for the owner to reconcile." >&2
fi

rewrite CHANGELOG.md -v rel="$released" -v date="$date" -v diverged="$diverged" -v notes="$notes" '
  function released_section(   line) {
    print "## [" rel "] - " date
    print ""
    while ((getline line < notes) > 0) print line
    print ""
  }
  # The heading itself, then (normally) the dated heading straight under it, so every
  # entry below now belongs to the release.
  $0 == "## [Unreleased]" {
    print; print ""
    if (!diverged) { print "## [" rel "] - " date } else { in_unreleased = 1; skip_blank = 1 }
    headings++
    next
  }
  # Diverged: keep the text on master, then insert the released section before the next one.
  in_unreleased && skip_blank && /^[[:space:]]*$/ { next }
  in_unreleased { skip_blank = 0 }
  in_unreleased && (/^## \[/ || /^\[[^]]+\]: /) { released_section(); in_unreleased = 0 }
  # [Unreleased]: <repo>/compare/v<prev>...HEAD
  /^\[Unreleased\]: .*\/compare\/v[0-9.]+\.\.\.HEAD$/ {
    url = $0; sub(/^\[Unreleased\]: /, "", url); sub(/\/compare\/.*$/, "", url)
    prev = $0; sub(/^.*\/compare\/v/, "", prev); sub(/\.\.\.HEAD$/, "", prev)
    print "[Unreleased]: " url "/compare/v" rel "...HEAD"
    print "[" rel "]: " url "/compare/v" prev "...v" rel
    links++
    next
  }
  { print }
  END {
    if (in_unreleased) { print ""; released_section() }
    if (headings != 1) { print "found " headings " \"## [Unreleased]\" headings" > "/dev/stderr"; exit 1 }
    if (links != 1) { print "no \"[Unreleased]: <repo>/compare/v<x.y.z>...HEAD\" link reference" > "/dev/stderr"; exit 1 }
  }
'

exit $((diverged ? 3 : 0))
