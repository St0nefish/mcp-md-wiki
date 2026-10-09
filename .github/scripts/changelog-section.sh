#!/usr/bin/env bash
# Prints the body of one CHANGELOG.md section: the lines after `## [<name>]` up to the next
# `## [` heading or the link-reference block at the end of the file. release.yml reads the
# `[Unreleased]` section with it twice — `check` requires it to be non-empty and
# release-notes.sh uses it for an empty release body — and the version roll
# (roll-version.sh) compares against it, so all three read the same lines.
#
# Usage: changelog-section.sh <name> [file]   e.g. changelog-section.sh Unreleased
# Exit 1 when the section is missing or holds nothing but whitespace.
set -euo pipefail
shopt -s inherit_errexit

name=${1:?usage: changelog-section.sh <name> [file]}
file=${2:-CHANGELOG.md}

body=$(awk -v heading="## [${name}]" '
  index($0, heading) == 1 { in_section = 1; next }
  in_section && (/^## \[/ || /^\[[^]]+\]: /) { exit }
  in_section { print }
' "$file")

if ! grep -q '[^[:space:]]' <<<"$body"; then
  echo "::error::${file}'s '## [${name}]' section is missing or empty." >&2
  exit 1
fi
# Without the blank lines that separate the section from its neighbours.
sed -e '/[^[:space:]]/,$!d' <<<"$body" | awk '{ lines[NR] = $0 } /[^[:space:]]/ { last = NR } END { for (i = 1; i <= last; i++) print lines[i] }'
