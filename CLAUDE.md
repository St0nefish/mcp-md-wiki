# mcp-md-wiki

Rust binary with subcommands: `serve`, `index`, `validate`, `status`.

## Hosting context

This project is hosted on **GitHub** (issues, PRs, CI) — use the `gh` CLI for this repo's remote operations. The knowledge bases it indexes live on separate Git hosts (typically Gitea). Do not conflate the two — webhook provider config, signature headers, and the `deploy/ci-examples/gitea-reindex.yml` workflow all refer to the *indexed knowledge base's* Git host, not this repo's host.

## Architecture

Single binary (`mcp-md-wiki`) that combines MCP server, webhook handler, and CLI indexer. Docker Compose runs 3 services: qdrant, embeddings, mcp-md-wiki.

In `serve` mode, indexing is asynchronous: MCP write tools and the webhook handler never call the indexer directly — they mark repo-relative paths (or a full reconcile) dirty on a `reindex::ReindexQueue` and return immediately. That queue is an injected dependency, not a global: `server::run_server` builds exactly one `Arc<ReindexQueue>` and clones it into every producer (`KbSearchServer`, `UiState`, `WebhookState`, `AdminState`) and into the worker, so every producer and the worker are provably talking to the same queue rather than relying on convention — `write::WriteDeps::queue` is how the write pipeline receives its handle. A single background worker (`reindex::run_worker`) drains that queue and is the only thing that calls `ingest::index_paths`, which is itself the only function that mutates Qdrant or the state DB. `ingest::scan_for_dirty` is the read-only detector behind a full reconcile — it walks the corpus and stat/hash-compares against `indexed_files`, producing a worklist for `index_paths` rather than indexing anything itself. The `index` CLI subcommand has no worker: it runs `ingest::scan_and_index` synchronously in-process.

Every git invocation against the KB clone is serialized by `git::GIT_LOCK`. A working copy cannot take concurrent mutation — `add`/`commit`/`merge`/`rebase` all contend for `.git/index.lock` — and the write tools and the webhook handler reach the same clone routinely, because each write pushes and the push webhooks straight back. Functions in `git.rs` therefore take a `&GitLock` as their first argument, so the type checker, not review, answers "is the lock held". Acquire once per logical sequence and hold it across the whole thing: a failed write and its rollback must share one acquisition, or the rollback races the very writers it is protecting against. The guard is never re-acquired internally, which is what keeps the non-reentrant mutex from deadlocking a call chain against itself.

That rule covers every git call that *mutates* the clone. Read-only history calls (`git::recent_commits`, `document_history`, `document_commit_diff`) deliberately take no `&GitLock`, and the deviation is argued in their doc comment: a read cannot corrupt an immutable object store, while taking the lock would queue history reads behind an in-flight write for up to `GIT_TIMEOUT` — the inverse of the contention #236 was filed for. Verified empirically against a repo held mid-rebase-conflict: `git log` exits 0 with a transiently reverted view (the in-flight commit hidden while HEAD is detached), never an error or invalid output, and a stray `.git/index.lock` does not affect reads at all. A new *mutating* git function still takes the guard.

Authentication on the protected routes (`/mcp`, `/status`, `/metrics`, `/admin/reload`) is **dual-mode**, not either/or: `server::bearer_auth` admits a request if the static bearer token matches (constant-time, tried first — it is a string compare and it is what Claude Code sends) **or** a presented JWT validates through `oauth::OAuthValidator`. Adding OAuth did not deprecate, gate or reshape the static path; a deployment can run either, both, or neither (`mcp.allow_unauthenticated`). What OAuth changes about refusals is the `WWW-Authenticate` header, which goes on *every* 401/403 once OAuth is configured — including a failed static-token request, because the server cannot tell which credential the caller meant to present, and claude.ai will not start the authorization flow at all without the `resource_metadata` parameter (Claude Code tolerates its absence, which is why a missing header is easy to ship and hard to notice). The two `/.well-known/oauth-protected-resource*` routes are registered outside the auth layer for the same reason `/health` is: discovery has to work for a caller who has no credential yet, and gating it behind the authentication it bootstraps makes the flow unstartable.

**Per-tool write-scope enforcement is not implemented.** `bearer_auth` sits in front of the whole `/mcp` endpoint and cannot see which MCP tool a request invokes — that is in the JSON-RPC body, which only rmcp parses — so any valid token currently grants full access, writes included. The validated scopes are parked in request extensions as an `oauth::AuthorizedToken` so a later change can enforce them at tool dispatch.

## Key conventions

- All async code uses tokio
- Config loaded from `config.yaml` (deserialized in `src/config.rs`)
- State tracked in SQLite via sqlx (default `/data/state.db`, i.e. `<source.data_path>/state.db`)
- Point IDs are UUID5 from `file_path::chunk_index`
- Qdrant accessed via gRPC (port 6334)
- Embeddings via OpenAI-compatible API (async-openai)
- MCP via rmcp with Streamable HTTP transport

## Keeping docs in sync

Docs are part of the change, not a follow-up. Any change to behavior, a tool parameter or response field, a config key or default, or a reload effect updates every place that describes it **in the same change** — and again each time review fixes reshape it:

- `assets/mcp/**/*.md` and `descriptions.rs` (what the model is told — the highest-stakes copy)
- `README.md` (tool parameter tables, configuration, reindex behavior)
- `deploy/config.example.yaml` (every config key, with its real default)
- `CHANGELOG.md` `[Unreleased]` (including upgrade/reindex consequences)
- this file's architecture notes and module table, and `ARCHITECTURE.md`

Check each claim against the code rather than the plan that preceded it. Code comments explain the code as it is: cite issue numbers (`#286`), never local plan/review labels ("Step 3", "Batch B") or narration of how the change evolved.

## Module layout

| File | Purpose |
|---|---|
| `main.rs` | CLI entrypoint (clap subcommands) |
| `config.rs` | Config deserialization. `Granularity` is the `search` granularity enum shared with `mcp.rs`; `ResolvedConfig::effective_granularities` is the one definition of the enabled set. The "section listed but gated off" load warning fires only when `config.yaml` sets `search.granularities` itself (`yaml_sets_search_granularities`), never for the default |
| `validate.rs` | Frontmatter validation against the resolved `.kb-schema.yaml` cascade for each file's path (not the global config directly — `frontmatter` in `config.yaml` is only the implicit root schema) |
| `ingest.rs` | Indexing: `index_paths` (the only function that mutates Qdrant/state — chunk/embed/upsert changed paths, purge missing ones, refresh stale metadata) and `scan_for_dirty` (read-only detector: stat/hash-compares the corpus against `indexed_files` and produces a dirty-path worklist). `scan_and_index` composes the two for callers with no worker (CLI, startup bootstrap). `heading::body_line_offset`-shifted chunk and section line numbers (counted from the top of the raw file, matching `get_document` — #286) and, when `chunking.heading_metadata` is on, the `heading_path`/`heading_level`/`heading_prefixes`/`section_key`/`section_line_start`/`section_line_end` payload fields (`qdrant.rs`'s `HEADING_*`/`SECTION_*` keys) get written alongside the existing ones — off by default (#286). `heading_prefixes` holds one `heading::heading_prefix_key` per contiguous run of the chunk's `heading_path` (`derive_heading_prefixes`), so `heading_prefix` matches a run starting at any level. Each `indexed_files` row stores `chunking_fingerprint` (every `chunking.*` field, except `target_chunk_size` while `heading_metadata` is on, + `CHUNKER_VERSION` — bump it when chunk text or payload logic changes); `scan_for_dirty` and `process_file`'s skip check treat a mismatch as dirty, so chunking changes re-chunk automatically on the next reconcile |
| `reindex.rs` | `ReindexQueue` (`mark_paths`/`mark_full`), an injected dependency (not a global — see the Architecture note above) cloned as an `Arc` into every producer and into its single background worker (`run_worker`), which drains the queue into `ingest::index_paths` with coalesce-don't-drop semantics and transient-vs-permanent retry/backoff |
| `heading.rs` | The heading model — single source of truth for section boundaries, heading text, levels, ancestry and line numbers, consumed by both `chunk.rs` and `retrieval::outline` so they cannot disagree. `HeadingTree::parse` runs pulldown-cmark offset iteration (top-level block headings only: not inside blockquotes/lists/footnotes; setext counts; text is rendered plain text with layout-only invisible characters (soft hyphen, ZWSP, BOM) removed but rendering ones (ZWJ/ZWNJ, direction marks, variation selectors) kept, whitespace-collapsed, capped at `MAX_HEADING_TEXT_CHARS` (200) matching characters; `normalize_heading_text` drops both kinds, case-folds and re-caps after folding so both sides of a comparison are cut identically; empty headings ignored; only the first heading per `\n`-line) and precomputes each `Heading`'s `parent`/`section_end`/`subtree_end`; headings are identified by index, never by text. `SectionId` (`Root`/`Preamble`/`Heading(i)`) is a chunk's attributed section. `LineIndex` gives O(log n) byte→line. `body_line_offset` derives the frontmatter line count from the raw file using gray_matter's own delimiter rules (#286) |
| `chunk.rs` | Markdown chunking over `heading::HeadingTree`. With `chunking.heading_metadata` on, no chunk crosses a heading boundary (no accumulate-to-target merge, fragment merges only within one section), so every chunk is attributed to `Heading(i)` or `Preamble`, never `Root`. With it off, the merge loop narrows each chunk's breadcrumb and attributed section to the common ancestor (by heading identity) of what each merged piece would have had alone; `Chunk::heading_path`/`heading_level`/`section_line_start`/`section_line_end` are always computed but only reach the Qdrant payload when `chunking.heading_metadata` is on (#286) |
| `embed.rs` | Embedding API client |
| `qdrant.rs` | Qdrant operations. `ensure_collection` takes an `IndexFeatures` (`from_config`) naming which optional payload indexes to create |
| `state.rs` | SQLite state DB: file bookkeeping (`indexed_files`, including `mtime`/`size` for the reconcile scan's stat pre-filter and `chunking_fingerprint` for its chunking-change check) plus the document metadata index (`documents`, `document_fields`) backing `list_documents` |
| `document_fields.rs` | Projects frontmatter JSON into filterable `document_fields` rows (dot-path flattening, array/range support) |
| `schema.rs` | Directory-cascading `.kb-schema.yaml` support: parse, cascade merge, `SchemaCache` tree resolution, type/value checking, schema fingerprinting |
| `retrieval.rs` | Shared retrieval core (`search` + `get_document`) used by MCP and (future) CLI, plus the line-range slicer behind `get_document`'s `start_line`/`end_line` (`LineRange`/`slice_lines`: 1-based inclusive, `end` clamped, byte-exact substrings). `get_document`'s section/outline core, shared by `mcp.rs` and `web.rs`: `parse_document_view_request` (mode validation) → `resolve_document_view` (size-capped section, scoped outline, or range) → `document_view_json` (the one JSON shape both adapters emit) (#286). `search.section_max_bytes` caps every read the caller did not bound itself (#290): a whole-document read over it degrades to the document's outline (`OutlineView::document_oversized` + `intro`, the range above the first heading) when the document has headings, and to a `truncate_to_line_budget` slice (`LineSlice::truncated`) when it has none — as does an oversized section with no sub-headings (`SectionView::truncated`/`end_line`). An explicit `start_line`/`end_line` range is exempt. The #290 keys are emitted only when they apply, so pre-#290 response shapes are byte-identical. `search_sections` backs `search`'s `section` granularity, sharing `fetch_grouped` with `search_grouped` |
| `sparse.rs` | Pure-Rust BM25-style sparse-vector tokenizer (FNV-1a term hashing) feeding the `sparse` named vector for hybrid retrieval — no model, no network |
| `rerank.rs` | Cross-encoder reranking client; truncates each candidate to a byte budget derived from `chunking.max_chunk_size` before sending, with exponential-backoff retry |
| `mcp.rs` | The six MCP tools (rmcp): `search` (query-mode chunk/document/**section** retrieval and, with no `query`, the exhaustive enumeration formerly served by `list_documents`), `get_document`, `get_schema`, `update_schema` — thin handlers delegating to `retrieval`/`state`/`schema` — and the write tools `write_document`/`delete_document` (the former `create_document`/`edit_document`/`move_directory` unified into one upsert-and/or-relocate tool) — thin adapters over `write::write_document`/`write::delete_document` that map `WriteSuccess`/`WriteError` back onto the exact `CallToolResult`/`McpError` text and `data` payloads; they never index inline. `search`'s `section` granularity (`search_sections`, rejects `explain`/`fields`, requires `chunking.heading_metadata`) returns path-only section rows, no text; `heading_prefix` filters query results to a contiguous run of headings anywhere in the chunk's heading path (normalized via `heading::heading_prefix_key` on both the payload and query side), same flag gate, rejected without a `query`. `overlay_input_schema` rewrites the `search` schema per call from the effective granularity set: the `granularity` `enum`/description, plus `descriptions::search_property_descriptions` (rewords or removes properties that only apply to a disabled granularity or to no-query search). `get_document`'s `line`/`heading_path` (exclusive with each other), `levels_up` and `outline` (combinable with a selector to outline one section) resolve through `retrieval::parse_document_view_request`/`resolve_document_view`; `structured_content` is `retrieval::document_view_json` plus the envelope (#286). Tool and server descriptions come from `descriptions.rs`'s compiled+config-derived overlay, not literal strings in this file. `list_tools`/`get_tool` apply `overlay_description`, then `overlay_input_schema`, then `tool_schema::self_contained` — the last step, in every case |
| `tool_schema.rs` | Pure JSON post-processor, `make_self_contained`, rewriting a `Tool::input_schema` into one with no `$ref`, no `$defs`/`definitions`, and no boolean `true`/`false` subschema in a schema position (data keywords — `enum`/`const`/`default`/`examples` — and a boolean `additionalProperties`/`unevaluatedProperties` are left alone) — the shape llama.cpp's grammar converter requires, since llama-server fails the whole `tools/list` request with HTTP 400 if one tool's schema doesn't convert (#288). Refs are inlined recursively, siblings (e.g. `description`) merged over the inlined copy with the sibling winning; a ref back to a name already being expanded (`schema::RawFieldDef.fields`'s self-reference) is cut to `{"type": ...}` at the repeat rather than expanding forever, so a nested `update_schema` `fields` entry advertises as a plain object below one level — the server still deserializes and validates it exactly as before. An unresolvable ref degrades to `{}` and fires a `debug_assert!` |
| `descriptions.rs` | Assembles MCP tool/server descriptions from three layers: compiled-in `assets/mcp/*.md` (mechanics true of every deployment), config-derived sentences (the hybrid/phrase retrieval-mode sentence; for `search`, a granularity sentence naming only the `search.granularities` effective set, the no-query listing sentence only when an enabled granularity supports it, and a `heading_prefix` mention when `chunking.heading_metadata` is on), and per-KB policy loaded at runtime from `<mcp.extensions_path>/` in the served knowledge base (append-only, editable via `write_document`, no restart) |
| `write.rs` | Transport-agnostic write pipeline extracted from `mcp.rs`'s write tools: `write_document`/`delete_document` taking a `WriteDeps` bundle, returning `WriteSuccess`/`WriteError`. Owns schema-frozen check, frontmatter validation, dedup gate, commit-message validation, the pre-commit rollback (remove/unstage on create, restore-from-HEAD on edit), `git::commit_and_sync`, and marking paths dirty on `WriteDeps::queue` (a `&ReindexQueue`, unlike `WriteDeps::state` no `None` mode — every write must mark its path). Used by both `mcp.rs` (rmcp) and `web.rs` (HTTP) so the two transports share one behavior |
| `oauth.rs` | OAuth 2.1 **resource server** (RFC 9728 + MCP authorization spec): `OAuthValidator` verifies RS256 access tokens against a lazily-fetched, in-memory JWKS (unknown `kid` refetches at most once per 60s, since `kid` is attacker-controlled; every fetch failure fails closed), checks `iss`/`aud`/`exp` inside one `jsonwebtoken::decode` so the signature can never be reordered after the claim checks, then the required scope. Also owns the RFC 9728 metadata document and both `WWW-Authenticate` challenge strings. **`aud` is the OAuth client_id, not the resource URL** — Authentik does not implement RFC 8707 resource indicators; this is deliberate, argued in the code, and must not be "fixed". This is strictly additive to the static bearer token, never a replacement — see `server.rs` |
| `status.rs` | Process-global indexing run state (`INDEX_STATUS`) backing `/status` and `/metrics`: in-flight phase/progress, last-run outcome and counters, payload-index health. `IndexStatus::is_bulk_indexing` (a full run, or one over more than one file, is in flight — or the sticky `stale_chunking` latch is set) gates the "results may be incomplete" note on `section`/`heading_prefix` results. The latch is set by `ReconcileScan::mark_rechunk` the moment a reconcile scan finds a non-frozen file with a stale chunking fingerprint, and — unlike the per-scan `ReconcileScan` guard itself — does NOT clear when that guard drops: only `IndexStatus::clear_stale_chunking` clears it, which `ingest::scan_and_index` calls both when a scan finds nothing to re-chunk and when the run consuming its worklist finishes successfully, so the note stays lit through retry backoff, a permanent give-up, and the wait for the next reconcile (#286 round-6 L1). Frozen scopes never set the latch — `scan_for_dirty` skips them before the check — since they are never re-chunked until the schema is fixed |
| `webhook.rs` | Webhook handler: fetches + ff-only merges the push, diffs the range (`git::git_diff_name_status`), and marks the changed paths dirty on `WebhookState::reindex_queue` — does not index inline |
| `reload.rs` | `POST /admin/reload`: re-reads and re-validates `config.yaml`, swaps it into the live `SharedConfig`, and classifies each changed setting `applied` (read fresh on next use) / `restart_required` (baked into a startup-built value) / `reindex_scheduled` (`chunking.*` — re-chunked automatically via the per-file chunking fingerprint) / `reindex_required` (`ui.semantic_edges.*`) |
| `git.rs` | Git operations for the KB clone (clone, fetch with timeout, `commit_and_sync`: add→commit→fetch→rebase→push, returning the changed paths the rebase pulled in alongside the commit SHA). Owns `GIT_LOCK` and the `GitLock` guard — see the git serialization rule below |
| `server.rs` | Axum server (MCP + webhook + `/health`, `/status`, `/metrics` routes, plus the unauthenticated web UI routes from `web.rs` — `/`, `/assets/*`, `/api/graph`, `/api/search`, `/api/doc/{*path}`, `/api/schema/{*path}` — and the unauthenticated OAuth discovery routes `/.well-known/oauth-protected-resource[/mcp]`, all merged in before the rate-limit `GovernorLayer` wrap; owns the Prometheus encoder); spawns the reindex worker and the periodic reconcile-sweep timer (`indexing.reconcile_interval_secs`). `bearer_auth` is dual-mode — see below |
| `web.rs` | The knowledge-base web UI, served straight from the binary (`assets/ui/`, embedded via `include_str!`, no filesystem reads at request time). Docs-first shell: hash-routed home/browse/doc views with a sidebar document tree and a semantic-search results panel; the Cytoscape.js graph is secondary, reachable as a per-doc neighborhood view or a full-KB view (fcose layout, hover/zoom-gated labels). `UiState` mirrors `StatusState`/`KbSearchServer::deps()`, reusing the same `Arc<OnceCell<StateDb>>` as `StatusState` rather than opening a second pool. Handlers: static shell/asset routes, `/api/graph` (nodes from `StateDb::all_document_summaries`, edges from `StateDb::all_links` filtered to the current node set, server-computed type palette), `/api/search` (thin wrapper over `retrieval::search`), `/api/doc/{*path}` GET/POST/DELETE (GET via `retrieval::get_document`, with the MCP tool's range/section/outline query params resolved through the same `retrieval::parse_document_view_request`/`resolve_document_view`/`document_view_json` — same contract, including #290's `section_max_bytes` cap on an unranged read, which is why the UI's viewer and editor fetch `?start_line=1`: a caller-named range is exempt, and the editor posts the whole body back on save; `content_hash` always over the whole file; POST/DELETE thin adapters over `write::write_document`/`write::delete_document`), `/api/schema/{*path}`. Deliberately unauthenticated — same open-route posture as `/health` — because the deployment sits behind Authentik via Traefik |

## Workflow

**Pattern A (CI-gated)**, per the knowledge base:
`dev/tools/repo-workflow-patterns.md`. `master` takes no direct pushes and is
protected by a repo **ruleset** (not classic branch protection) — direct push
disabled, status checks required, exactly as before, but the mechanism is a
ruleset and the details below are what changed:

- Work on a branch, open a PR against `master`. `ci.yml` runs `test` and
  `qdrant-integration`, fanning into a single `ci-pass` job — that is the
  **only** required status check the ruleset enforces (never list individual
  job names as required checks).
- Merges are **squash only** (`allow_merge_commit: false`,
  `allow_rebase_merge: false`) via `auto-merge.yml` (`gh pr merge --auto
  --squash`), titled `<PR title> (#<number>)`. Auto-merge fires as soon as
  `ci-pass` is green. The `auto-merge` job only runs for PRs authored by the
  owner (`St0nefish`); other contributors' PRs still run full CI but require
  a manual review and merge — they never land unattended.
- The ruleset's required-status-checks rule has
  `strict_required_status_checks_policy: false` — PR branches do **not** need
  to be up to date with `master` before merging. This is deliberate and
  load-bearing: it's what lets several PRs opened from the same base commit
  land in any order without serializing on each other. **Do not routinely
  rebase a PR branch just because `master` moved, and do not add a bot that
  force-pushes/rebases open PRs when `master` advances** — with the strict
  flag off, that would be pure wasted CI for zero benefit. Refresh a branch
  only to resolve a real conflict, or because the workflow file itself
  changed (rare here, since GitHub reads workflow files from the PR's base
  ref, not this Gitea gotcha).
- `post-merge.yml` re-runs the cheap lint/test checks on `master` after every
  push, to catch the rare semantic conflict that non-strict merging permits
  (two PRs each green against an older base, broken once combined). It is
  **not** a required check — it runs after the merge, not before.
- `fix #N` in the merge commit auto-closes GitHub issues.
- Branches auto-delete after merge.
- Pre-commit hook enforces `cargo fmt` + `cargo clippy` (activate with
  `./scripts/setup-dev.sh` after cloning).
- **Batch related changes into one PR.** Every merge to `master` runs
  `release.yml`, which builds a new image and pokes Watchtower to restart the
  live service — so N small PRs cost N builds and N restarts, and they
  serialize on the single self-hosted runner. Splitting a fix from the doc
  correction that belongs with it, or opening a second PR for something that
  could have been another commit on a branch already in flight, is pure
  churn. If several must land separately anyway, set the `DEPLOY_HOLD` repo
  variable to `true`, let them accumulate, then ship the batch with one
  `workflow_dispatch` run (`force=true`).

## Issue tracking

Bugs, features, and enhancements are tracked as GitHub issues (not in-repo TODO files).

## Build & run

```bash
cargo build
cargo run -- serve          # Start server (MCP + webhook)
cargo run -- index --full   # Full reindex
cargo run -- validate       # Validate frontmatter
cargo run -- status         # Collection stats
cargo run -- reproject-fields # Rebuild document_fields from stored frontmatter (no re-embed)
```
