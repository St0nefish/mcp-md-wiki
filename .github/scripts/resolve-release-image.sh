#!/usr/bin/env bash
# Finds the image dev.yml built for the commit a release targets.
#
# Every commit pushed to master is built once, and tagged sha-<full commit sha> only after
# its smoke test passes (dev.yml; build-<sha> is a staging tag nothing reads). A release
# promotes that build and never builds one itself, so this waits (bounded, backing off) for
# the dev build of RELEASE_SHA to appear and then fails closed. It stops early when every
# dev.yml run for the commit has finished and none succeeded. Read-only: safe to run
# locally against any commit after `docker login ghcr.io` with a token that has
# read:packages.
#
# Inputs (env): GITHUB_REPOSITORY, RELEASE_SHA (full sha); optional IMAGE (default
#   ghcr.io/<repository, lowercased>), RESOLVE_TIMEOUT (seconds, default 1200), GH_TOKEN
#   (or an authenticated gh; lets it read dev.yml's runs for the early stop, else that
#   stop is skipped and only the timeout ends the wait).
# Outputs (stdout, and $GITHUB_OUTPUT when set): image, sha_tag, digest, image_ref.
set -euo pipefail
shopt -s inherit_errexit

: "${GITHUB_REPOSITORY:?}" "${RELEASE_SHA:?}"
if [[ ! "$RELEASE_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "::error::RELEASE_SHA must be a full 40-character commit sha, got '${RELEASE_SHA}'."
  exit 1
fi
image="${IMAGE:-ghcr.io/${GITHUB_REPOSITORY,,}}"
sha_tag="sha-${RELEASE_SHA}"
timeout="${RESOLVE_TIMEOUT:-1200}"

# Prints "failed" when dev.yml has run for the commit, every run has completed and none
# succeeded; prints nothing otherwise, including when the runs cannot be read (no gh, no
# token), which only means the wait runs to its timeout.
dev_build_failed() {
  gh run list --repo "$GITHUB_REPOSITORY" --workflow dev.yml --commit "$RELEASE_SHA" \
    --json status,conclusion \
    --jq 'if length > 0 and all(.[]; .status == "completed" and .conclusion != "success")
          then "failed" else empty end' 2>/dev/null || true
}

deadline=$((SECONDS + timeout))
delay=15
while :; do
  if out=$(docker buildx imagetools inspect "${image}:${sha_tag}" --format '{{json .Manifest}}' 2>&1); then
    digest=$(jq -r '.digest // empty' <<<"$out")
    break
  fi
  if ! grep -qiE 'not found|manifest unknown' <<<"$out"; then
    echo "::error::Cannot inspect ${image}:${sha_tag}: ${out}"
    exit 1
  fi
  if [[ "$(dev_build_failed)" == "failed" ]]; then
    echo "::error::The dev build of ${RELEASE_SHA} failed, so there is no ${sha_tag} image to promote. Fix master, or re-run that dev.yml run, then re-run this release."
    exit 1
  fi
  if ((SECONDS + delay > deadline)); then
    echo "::error::No ${image}:${sha_tag} after ${timeout}s. A release promotes the image dev.yml built for the tagged commit and never builds one; check that dev.yml ran for ${RELEASE_SHA}. 'gh workflow run dev.yml --ref <tag>' builds it if not (a commit older than dev.yml cannot be built that way; release a newer commit)."
    exit 1
  fi
  echo "Waiting for ${image}:${sha_tag} (dev build not pushed yet); next check in ${delay}s."
  sleep "$delay"
  delay=$((delay * 2 > 120 ? 120 : delay * 2))
done

if [[ ! "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "::error::${image}:${sha_tag} resolved to no usable digest: ${out}"
  exit 1
fi

{
  echo "image=${image}"
  echo "sha_tag=${sha_tag}"
  echo "digest=${digest}"
  echo "image_ref=${image}@${digest}"
} | tee -a "${GITHUB_OUTPUT:-/dev/null}"
