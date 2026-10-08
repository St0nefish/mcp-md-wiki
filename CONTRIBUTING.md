# Contributing

mcp-md-wiki is hosted on GitHub (this repo, its issues, and its PRs). The knowledge
bases it indexes live on separate git hosts entirely — don't confuse the two when
you're reading webhook or `deploy/` config; that side of the config always refers to
the *indexed* knowledge base's git host, never this repo's.

## Workflow

`master` is branch-protected: direct pushes are disabled, and a passing status check
is required before a PR can merge. Concretely:

1. **Branch.** Work on a feature branch, not directly on `master` (you can't push to
   it anyway). There's no enforced naming convention beyond "descriptive" —
   `fix/reranking-timeout`, `docs/backup-recovery`, that kind of thing.
2. **Open a PR against `master`.** `pr-fast` (`.github/workflows/pr-fast.yml`) runs
   `cargo fmt -- --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`
   and `cargo audit` on your PR, on GitHub-hosted runners, when your diff touches a
   code path (`.github/scripts/code-paths.sh` has the list; a docs-only PR skips it).
   It is feedback, not the merge gate.
3. **Merge.** Once the maintainer has reviewed the PR, they arm GitHub's auto-merge
   (merge commit) on it, which queues it for the merge train (`.github/workflows/train.yml`):
   the train squashes the PR onto current `master`, runs the full suite and image
   build on exactly that tree, and posts the `ci-fast` and `ci-slow` statuses that
   let GitHub merge it. If the train reports a conflict or a failure on the PR, push
   a fix — the new head re-queues automatically.
4. **Closing issues.** Include `fix #N` (or `Fixes #N`, `Closes #N` — GitHub's usual
   set of magic words) in the PR body to auto-close the
   corresponding issue when the PR merges. The commits land on `master` as they are, so
   a keyword in one of their messages closes the issue too.
5. **Cleanup.** Branches auto-delete on merge — no need to clean up your own feature
   branch afterward.

Bugs, features, and enhancements are all tracked as GitHub issues; this repo doesn't
keep in-repo TODO files.

## Local setup

After cloning, run the one-time setup script:

```bash
./scripts/setup-dev.sh
```

This points `git config core.hooksPath` at `.githooks/`, which activates a
pre-commit hook that:

- Runs `cargo fmt` on your staged `.rs` files and re-stages the result.
- Runs `cargo clippy --all-targets -- -D warnings` (the `--all-targets` matters —
  without it, lint failures confined to `#[cfg(test)]` code slip past both the hook
  and, historically, CI too) and **blocks the commit** if it fails.
- If your change touches `deploy/config.example.yaml`, also runs
  `cargo test -q -- config::tests::example_config_deserializes` — a single test that
  parses the example config against the live `Config` struct and spot-checks several
  values, specifically to catch the file drifting out of sync with `src/config.rs`.
  A key that no longer exists, or a value that no longer matches its default, fails
  the commit rather than merging silently.

You can bypass the hook with `git commit --no-verify` when you genuinely need to, but
anything it would have caught still has to pass in CI before the PR can merge.

## Rust toolchain / MSRV

The pinned toolchain — currently **1.89** — lives in one place, `rust-toolchain.toml`
at the repo root, and everything else derives from it (#235):

- **Local dev** needs nothing extra: rustup reads `rust-toolchain.toml` automatically
  for every `cargo`/`rustc` invocation anywhere under this directory tree.
- **CI** (`.github/workflows/checks.yml`, via `.github/scripts/rust-version.sh`) extracts the file's `channel` value in a "Read
  pinned Rust toolchain version" step and feeds it explicitly to
  `dtolnay/rust-toolchain@master` (that action doesn't read the file itself) — both the
  `test` and `qdrant-integration` jobs do this, so neither can end up compiling against
  a different toolchain than the other.
- **The Docker build** takes the same value as a `--build-arg RUST_VERSION=...`, passed
  by that same CI step, so the release image is built with the exact toolchain CI just
  tested against — not `rust:X.Y-alpine` pinned by hand in the `Dockerfile` and left to
  drift out of sync with whatever `stable` CI happened to resolve to that day.
- **`rust-version` in `Cargo.toml`** carries the same number, so a plain `cargo build`
  under a too-old local toolchain fails fast with a clear "package requires rustc X but
  Y is installed" message instead of a confusing type error partway through a build.

This exists because the split bit twice in one session before #235: CI resolving
`dtolnay/rust-toolchain@stable` while the `Dockerfile` pinned an explicit, unrelated
version meant a feature stabilized after the Dockerfile's pin (`std::fs::File::lock`,
1.89.0) compiled clean through every `cargo` step CI ran and failed only in the Docker
build, at the end of a long job. Bumping the MSRV means updating `channel` in
`rust-toolchain.toml`, `rust-version` in `Cargo.toml`, and the `ARG RUST_VERSION`
default in `Dockerfile` together — CI and the Docker build then just follow.

## Testing

There is no `tests/` directory — every test lives inline in the module it tests,
inside a `#[cfg(test)] mod tests { ... }` block. Add tests next to the code they
cover rather than in a separate top-level tree.

```bash
cargo test                          # full suite (unit + most integration-style tests)
cargo fmt -- --check                # what the pre-commit hook and CI both enforce
cargo clippy --all-targets -- -D warnings
```

A small number of tests in `src/qdrant.rs` are `#[ignore]`d because they need a real
Qdrant server, not the mocked `VectorStore` the rest of the suite uses — plain
`cargo test` never runs them. Exercise them against this project's own Qdrant
service:

```bash
docker compose up -d qdrant
cargo test -- --ignored
```

CI runs this as the `qdrant-integration` job in `checks.yml`; the merge train runs it
as part of its slow tier, so a failure there blocks the merge (the `ci-slow` status).

## Style

Skim any of `src/schema.rs`, `src/ingest.rs`, or `src/server.rs` before writing new
code here — the house style leans heavily on dense explanatory comments that state
*why* a piece of logic exists (the failure mode it prevents, the issue number it
traces to, the tradeoff it made and the alternative it rejected), not just what the
code does line by line. A one-line "what" comment on a non-obvious function is
usually a sign more context belongs there. This applies to config field doc comments
too — `src/config.rs` and `deploy/config.example.yaml` are meant to explain the
reasoning behind a default, not just name it.

Formatting and linting are enforced mechanically (`cargo fmt`, `cargo clippy -D
warnings`) rather than by convention, so there's nothing further to memorize there.
