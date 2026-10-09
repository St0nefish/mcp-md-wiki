#!/usr/bin/env bash
# Prints the manifest digest a registry reference names, or nothing when the tag does
# not exist. Any other registry error fails, so an unreadable registry is never taken
# for "no image".
# Usage: image-digest.sh <image>:<tag>
set -euo pipefail

ref=${1:?usage: image-digest.sh <image>:<tag>}
if out=$(docker buildx imagetools inspect "$ref" --format '{{json .Manifest}}' 2>&1); then
  digest=$(jq -r '.digest // empty' <<<"$out")
  if [[ ! "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "::error::${ref} resolved to no usable digest: ${out}" >&2
    exit 1
  fi
  echo "$digest"
elif grep -qiE 'not found|manifest unknown' <<<"$out"; then
  exit 0
else
  echo "::error::cannot inspect ${ref}: ${out}" >&2
  exit 1
fi
