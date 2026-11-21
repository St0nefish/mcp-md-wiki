#!/usr/bin/env bash
# Runs the client plugin tests (stdlib unittest only: no pytest, no pip) and checks
# that the HTTP code the two clients share has not diverged.
#
# Usage: .github/scripts/clients-test.sh   (from anywhere)
set -euo pipefail
shopt -s inherit_errexit

root=$(git rev-parse --show-toplevel)

shared() {
  sed -n '/^# --- shared:begin/,/^# --- shared:end/p' "$1" | tail -n +2
}

hermes=$root/clients/hermes/mcp_md_wiki_memory/__init__.py
webui=$root/clients/open-webui/mcp_md_wiki_memory.py

if ! diff <(shared "$hermes") <(shared "$webui") >/dev/null; then
  echo "clients: the shared block differs between hermes and open-webui" >&2
  diff <(shared "$hermes") <(shared "$webui") >&2 || true
  exit 1
fi
if [[ -z $(shared "$hermes") ]]; then
  echo "clients: shared block markers not found" >&2
  exit 1
fi

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root/clients/hermes" \
  python3 -m unittest discover -s "$root/clients/hermes/tests" -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root/clients/open-webui" \
  python3 -m unittest discover -s "$root/clients/open-webui/tests" -v
