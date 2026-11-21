#!/usr/bin/env bash
# Installs or updates the memory Filter in an Open WebUI instance through its admin API.
#
# Environment:
#   OPENWEBUI_URL    base URL, e.g. https://chat.example.com (required)
#   OPENWEBUI_TOKEN  an admin API key (required; never printed)
#
# It reads the function's current state first and uses only create / update: it
# never calls the bulk sync endpoint, which would replace every function on the
# instance. A new function is left disabled until you enable it in the admin UI,
# and an existing function keeps its enabled and global switches.
#
# The endpoints below are from Open WebUI's documented API and have NOT been run
# against a live instance from this repository; see docs/memory.md.
set -euo pipefail
shopt -s inherit_errexit

: "${OPENWEBUI_URL:?set OPENWEBUI_URL}"
: "${OPENWEBUI_TOKEN:?set OPENWEBUI_TOKEN}"

here=$(dirname "$(realpath "$0")")
id=mcp_md_wiki_memory
base=${OPENWEBUI_URL%/}/api/v1/functions

api() { # method path [body-file]
  local args=(-sS -X "$1" -H "Authorization: Bearer $OPENWEBUI_TOKEN" -H 'Content-Type: application/json')
  [[ -n ${3:-} ]] && args+=(--data-binary "@$3")
  curl "${args[@]}" -w '\n%{http_code}' "$base$2"
}

body=$(mktemp)
trap 'rm -f "$body"' EXIT
jq -n --arg id "$id" --rawfile content "$here/mcp_md_wiki_memory.py" \
  '{id: $id, name: "mcp-md-wiki memory", content: $content,
    meta: {description: "Adds confirmed memories from mcp-md-wiki to each chat turn.", manifest: {}}}' >"$body"

existing=$(api GET "/id/$id")
code=${existing##*$'\n'}
case $code in
  200) path=/id/$id/update; verb=updated ;;
  401 | 403) echo "Open WebUI refused the token (HTTP $code)" >&2; exit 1 ;;
  *) path=/create; verb=created ;;
esac

result=$(api POST "$path" "$body")
code=${result##*$'\n'}
if [[ $code != 200 ]]; then
  echo "Open WebUI answered HTTP $code" >&2
  exit 1
fi
echo "memory filter $verb (id: $id). Enable it, and set MCP_MD_WIKI_TOKEN and the url Valve, in the admin UI."
