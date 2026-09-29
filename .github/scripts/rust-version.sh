#!/usr/bin/env bash
# Prints the Rust toolchain pinned in rust-toolchain.toml.
#
# rust-toolchain.toml's `channel` is the single source of truth (#235).
# dtolnay/rust-toolchain does not parse that file, and `@stable` would resolve a
# toolchain unrelated to the pin, so every workflow reads it here and feeds the same
# value to the toolchain action and to the image build's RUST_VERSION build arg. The
# cargo checks and the image cannot diverge.
#
# Outputs (stdout, and $GITHUB_OUTPUT when set): version.
set -euo pipefail

version=$(grep -m1 '^channel' rust-toolchain.toml | sed -E 's/channel *= *"([^"]+)"/\1/')
if [[ -z "$version" ]]; then
  echo "::error::could not parse channel from rust-toolchain.toml"
  exit 1
fi
echo "version=$version" | tee -a "${GITHUB_OUTPUT:-/dev/null}"
