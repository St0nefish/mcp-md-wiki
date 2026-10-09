#!/usr/bin/env bash
# Smoke-tests an mcp-md-wiki image. build-image.yml runs it on every build (ci-slow's,
# and master.yml's fallback) before tagging it; release.yml runs it again on the exact
# image it is about to promote.
#
# `mcp-md-wiki serve` needs (src/config.rs resolve_inner, src/server.rs run_server):
#   - EMBEDDING_BASE_URL, EMBEDDING_MODEL, QDRANT_URL: required env, no defaults.
#   - a reachable Qdrant (gRPC :6334): run_server calls ensure_collection BEFORE it
#     binds the listener, so a missing Qdrant means the process exits, not a 503.
#   - MCP_BEARER_TOKEN: the static bearer token (mcp.bearer_token_env default).
#   The server listens on MCP_PORT, default 8001.
# It does NOT need config.yaml (all defaults), a git remote (GIT_URL unset skips the
# clone), or a working embeddings endpoint: indexing is async and the KB is empty, so
# EMBEDDING_BASE_URL points at a port that refuses connections. /health then reports
# 503 with embeddings "unavailable", which is expected here.
#
# Asserts:
#   - GET /health answers, with qdrant "ok" (200, or 503 with only embeddings down).
#   - POST /mcp with no token, and with a wrong one, is 401.
#   - POST /mcp `initialize` with the correct token is 200.
#
# Networking: the script may itself run inside a runner container that drives the host
# Docker daemon through /var/run/docker.sock. There, `-p 127.0.0.1:...` publishes on
# the HOST's loopback, which the script's own network namespace cannot reach. So
# nothing is published: Qdrant, the server and the curl probes all live on a private
# per-run docker network and address each other by container name. That works the
# same on a plain workstation.
#
# Usage: smoke-test-image.sh <image ref>   (pulled by `docker run` if not present)
# Needs: docker, jq. Everything is named per run and removed by a trap; several runner
# containers share one daemon.
set -euo pipefail
shopt -s inherit_errexit

image=${1:?usage: smoke-test-image.sh <image ref>}
run_id="${GITHUB_RUN_ID:-local}-$$"
net="mdw-smoke-${run_id}"
qdrant_name="mdw-smoke-qdrant-${run_id}"
app_name="mdw-smoke-app-${run_id}"
token=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')
body=$(mktemp)
code=000

# Qdrant image is single-sourced from docker-compose.yml, as checks.yml does. Override with
# SMOKE_QDRANT_IMAGE (e.g. when running from outside the repo root).
qdrant_image=${SMOKE_QDRANT_IMAGE:-$(docker compose -f docker-compose.yml config --images qdrant 2>/dev/null | head -n1 || true)}
curl_image=${SMOKE_CURL_IMAGE:-curlimages/curl:8.11.1}

cleanup() {
  docker rm -f "$app_name" "$qdrant_name" >/dev/null 2>&1 || true
  docker network rm "$net" >/dev/null 2>&1 || true
  rm -f "$body"
}
trap cleanup EXIT

fail() {
  echo "::error::smoke test of ${image}: $*"
  for c in "$app_name" "$qdrant_name"; do
    echo "--- ${c} logs (last 100 lines) ---"
    docker logs "$c" 2>&1 | tail -n 100 || true
  done
  exit 1
}

[[ -n "$qdrant_image" ]] || fail "could not determine the Qdrant image (set SMOKE_QDRANT_IMAGE)"

docker network create "$net" >/dev/null || fail "could not create docker network"
docker run -d --name "$qdrant_name" --network "$net" "$qdrant_image" >/dev/null ||
  fail "could not start Qdrant (${qdrant_image})"

# Runs curl on the private network. Prints the response body, then the status code on
# its own last line ("000" when curl could not connect).
probe() {
  docker run --rm --network "$net" "$curl_image" -s --max-time 10 -w '\n%{http_code}' "$@" || true
}
# Runs a probe; leaves the body in $body (file) and the status in $code.
request() {
  local out
  out=$(probe "$@")
  code=${out##*$'\n'}
  printf '%s' "${out%$'\n'*}" >"$body"
}

# Qdrant must be up before the app starts: the app exits if it cannot reach it.
for _ in $(seq 1 60); do
  request "http://${qdrant_name}:6333/readyz"
  [[ "$code" == "200" ]] && break
  sleep 1
done
[[ "$code" == "200" ]] || fail "Qdrant did not become ready within 60s (last status ${code})"

# Port 9 (discard) refuses connections, so embeddings are unavailable but harmless.
docker run -d --name "$app_name" --network "$net" \
  -e MCP_BEARER_TOKEN="$token" \
  -e QDRANT_URL="http://${qdrant_name}:6334" \
  -e EMBEDDING_BASE_URL="http://127.0.0.1:9/v1" \
  -e EMBEDDING_MODEL="smoke-test" \
  "$image" serve >/dev/null || fail "docker run failed"
base="http://${app_name}:8001"

answered=false
for _ in $(seq 1 60); do
  if [[ "$(docker inspect -f '{{.State.Running}}' "$app_name")" != "true" ]]; then
    fail "container exited before serving /health"
  fi
  request "${base}/health"
  if [[ "$code" == "200" || "$code" == "503" ]]; then
    answered=true
    break
  fi
  sleep 1
done
$answered || fail "/health did not answer within 60s (last status ${code})"
jq -e '.qdrant.status == "ok"' "$body" >/dev/null ||
  fail "/health ${code} does not report qdrant ok: $(cat "$body")"
echo "ok: /health ${code} $(cat "$body")"

init='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"smoke-test","version":"0"}}}'
post_mcp() {
  request "${base}/mcp" -X POST \
    -H 'Content-Type: application/json' \
    -H 'Accept: application/json, text/event-stream' \
    "$@" --data "$init"
}

post_mcp
[[ "$code" == "401" ]] || fail "POST /mcp without a token returned ${code}, expected 401"
echo "ok: POST /mcp without a token -> 401"

post_mcp -H "Authorization: Bearer not-the-token-${token}"
[[ "$code" == "401" ]] || fail "POST /mcp with a wrong token returned ${code}, expected 401"
echo "ok: POST /mcp with a wrong token -> 401"

post_mcp -H "Authorization: Bearer ${token}"
[[ "$code" == "200" ]] || fail "POST /mcp initialize with the token returned ${code}, expected 200: $(head -c 500 "$body")"
echo "ok: POST /mcp initialize with the token -> 200"

echo "smoke test passed: ${image}"
