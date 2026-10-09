# Single source of truth for the pinned Rust version is rust-toolchain.toml
# at the repo root (`[toolchain] channel = "..."`, #235). CI reads that
# file's channel and passes it here via `--build-arg RUST_VERSION=...`, so
# this pin can no longer silently drift from what CI's own cargo steps (and
# `dtolnay/rust-toolchain`) actually use. The default below only matters for
# a plain local `docker build .` run with no build args — same convention as
# the VERSION/REVISION ARGs further down.
#
# MSRV 1.89: `ingest::acquire_reindex_lock` uses `std::fs::File::lock` /
# `lock_shared`, stabilized in 1.89.0. Do not lower this pin (here or in
# rust-toolchain.toml) without replacing that call. Before #235, this pin and
# CI's `dtolnay/rust-toolchain@stable` had no relationship at all: a too-old
# pin here compiled clean through every cargo step CI ran and failed only in
# this Docker build, at the end of a long job.
ARG RUST_VERSION=1.89
# Dependency compilation goes through cargo-chef so it is an ordinary image layer
# (keyed on Cargo.toml/Cargo.lock only, via recipe.json) that a buildx registry
# cache can store and reuse. A `RUN --mount=type=cache` target dir cannot be
# exported to a registry cache, so on a fresh builder (CI) it starts empty and
# every dependency would recompile on every build.
#
# Pinned so the layer is reproducible; `--locked` uses cargo-chef's own lockfile.
# Installed with `cargo install` rather than the lukemathwalker/cargo-chef image so
# it works for whatever RUST_VERSION is passed, not only versions that image tags.
FROM rust:${RUST_VERSION}-alpine AS chef
RUN apk add --no-cache musl-dev openssl-dev openssl-libs-static perl
RUN cargo install cargo-chef --locked --version 0.1.78
WORKDIR /build

# Reduce the manifests + sources to recipe.json. Only its content flows into the
# cook stage, so editing a source file leaves the cook layer's cache key unchanged
# unless dependencies changed.
FROM chef AS planner
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
RUN cargo chef prepare --recipe-path recipe.json

# Compile dependencies only. Must not depend on the registry cache mount for
# correctness; the mount just saves re-downloads on local rebuilds.
FROM chef AS builder
COPY --from=planner /build/recipe.json recipe.json
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cargo chef cook --release --recipe-path recipe.json

# Build the real binary. src/, assets/ (include_str!) and migrations/ are the
# compile-time inputs; deploy/ is only read by a #[cfg(test)] include_str!.
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY migrations/ migrations/
COPY assets/ assets/
# After the cook layer so a new revision does not bust the dependency cache (#264).
# Read by option_env!("REVISION") in src/server.rs.
ARG REVISION=unknown
ENV REVISION=$REVISION
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cargo build --release && \
    cp target/release/mcp-md-wiki /usr/local/bin/mcp-md-wiki

# Runtime image
FROM alpine:3.21

# Populated by CI (`docker buildx build --build-arg VERSION=... --build-arg
# REVISION=...`); default to "unknown" so a plain local `docker build .` with no
# build args still produces a valid, if uninformative, label instead of an empty
# one. REVISION (org.opencontainers.image.revision) records the git TREE hash the
# image was built from, not a commit: ci-slow builds the image before the merge
# commit exists, from a local merge with the same tree (.github/workflows/slow.yml).
# `git log --format='%H %T' master` maps it back to the commit(s) with that tree.
ARG VERSION=unknown
ARG REVISION=unknown

LABEL org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.source="https://github.com/St0nefish/mcp-md-wiki"

RUN apk add --no-cache ca-certificates git

COPY --from=builder /usr/local/bin/mcp-md-wiki /usr/local/bin/mcp-md-wiki

RUN addgroup -g 65532 -S nonroot && adduser -u 65532 -S nonroot -G nonroot

WORKDIR /app

# The app's actual default data_path is /data (source.data_path, config.rs), which is
# where every compose file and deploy template mounts the named volume. Pre-creating
# and chowning it here means Docker propagates that ownership when it initializes a
# fresh named volume, instead of the mountpoint coming up root-owned and unwritable
# by the non-root user below.
RUN mkdir -p /data && chown nonroot:nonroot /data

USER nonroot

HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=10s \
  CMD ["mcp-md-wiki", "health"]

ENTRYPOINT ["mcp-md-wiki"]
CMD ["serve"]
