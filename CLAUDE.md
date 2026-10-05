# mcp-md-wiki

Rust binary with subcommands: `serve`, `index`, `validate`, `status`.

## Hosting context

This project is hosted on **GitHub** (issues, PRs, CI) — use the `gh` CLI for this repo's remote operations. The knowledge bases it indexes live on separate Git hosts (typically Gitea). Do not conflate the two — webhook provider config, signature headers, and the `deploy/ci-examples/gitea-reindex.yml` workflow all refer to the *indexed knowledge base's* Git host, not this repo's host.

## Architecture

Single binary (`mcp-md-wiki`) that combines MCP server, webhook handler, and CLI indexer. Docker Compose runs 3 services: qdrant, embeddings, mcp-md-wiki.

In `serve` mode, indexing is asynchronous: MCP write tools and the webhook handler never call the indexer directly — they mark repo-relative paths (or a full reconcile) dirty on a `reindex::ReindexQueue` and return immediately. That queue is an injected dependency, not a global: `server::run_server` builds exactly one `Arc<ReindexQueue>` and clones it into every producer (`KbSearchServer`, `UiState`, `WebhookState`, `AdminState`) and into the worker, so every producer and the worker are provably talking to the same queue rather than relying on convention — `write::WriteDeps::queue` is how the write pipeline receives its handle. A single background worker (`reindex::run_worker`) drains that queue and is the only thing that calls `ingest::index_paths`, which is itself the only function that mutates Qdrant or the state DB. `ingest::scan_for_dirty` is the read-only detector behind a full reconcile — it walks the corpus and stat/hash-compares against `indexed_files`, producing a worklist for `index_paths` rather than indexing anything itself. The `index` CLI subcommand has no worker: it runs `ingest::scan_and_index` synchronously in-process.

Every git invocation against the KB clone is serialized by `git::GIT_LOCK`. A working copy cannot take concurrent mutation — `add`/`commit`/`merge`/`rebase` all contend for `.git/index.lock` — and the write tools and the webhook handler reach the same clone routinely, because each write pushes and the push webhooks straight back. Functions in `git.rs` therefore take a `&GitLock` as their first argument, so the type checker, not review, answers "is the lock held". Acquire once per logical sequence and hold it across the whole thing: a failed write and its rollback must share one acquisition, or the rollback races the very writers it is protecting against. The guard is never re-acquired internally, which is what keeps the non-reentrant mutex from deadlocking a call chain against itself.

That rule covers every git call that *mutates* the clone. Read-only history calls (`git::recent_commits`, `document_history`, `document_commit_diff`) deliberately take no `&GitLock`, and the deviation is argued in their doc comment: a read cannot corrupt an immutable object store, while taking the lock would queue history reads behind an in-flight write for up to `GIT_TIMEOUT` — the inverse of the contention #236 was filed for. Verified empirically against a repo held mid-rebase-conflict: `git log` exits 0 with a transiently reverted view (the in-flight commit hidden while HEAD is detached), never an error or invalid output, and a stray `.git/index.lock` does not affect reads at all. A new *mutating* git function still takes the guard.

Authentication on the protected routes (`/mcp`, `/status`, `/metrics`, `/admin/reload`) is **dual-mode**, not either/or: the [`oauth-resource-server`](https://github.com/St0nefish/oauth-resource-server) crate's `require_auth` middleware (mcp-md-wiki#308), layered by `server::assemble_router`, admits a request if the static bearer token matches (constant-time, tried first — it is a string compare) **or** a presented JWT validates through the crate's `OAuthValidator`. OAuth is the recommended path (the docs lead with it, and a static-token-only startup logs a warning), but the static path is not deprecated or reshaped; a deployment can run either, both, or neither (`mcp.allow_unauthenticated`), and `mcp.oauth.accept_static_bearer: false` turns the static path off while OAuth is on. The validator, the credential layer (`Credential`/`authenticate`/`static_token_policy`) and the RFC 9728 metadata router all live in the crate now — this repo owns only the `mcp.oauth` config shape (`config::resolve_mcp_oauth`, which layers this application's own defaults, `required_scope` = `mcp:read` and `scopes_supported` = `[mcp:read, mcp:write]`, on top of the crate's), route selection (which paths sit behind `require_auth`, which — `/health`, the well-known routes, the web UI — deliberately don't), and `server::static_bearer_token`, a thin wrapper around the crate's `static_token_policy` that supplies this server's own log lines and message text; it is still the one place that decides which static token, if any, the auth layer holds, and it returns the crate's `StaticTokenDecision`, which `server::auth_layer` turns into the layer through `AuthLayerBuilder::build_with_decision` (so the crate, not this repo, maps a decision to a pass-through or an enforcing layer, and checks it agrees with the validator). `auth_layer` also sets `static_challenge(None)`: a static-token-only 401 has never carried `WWW-Authenticate` here, and opting out of the crate's default `Bearer error="invalid_token"` keeps it that way. The validator is provider-agnostic: every per-provider difference (audience value, scope claim name/shape, algorithm, `typ`, username claim) is an `mcp.oauth` key, never a code path — a new provider's shape gets a documented-shape fixture test in the crate, not a provider branch here. What OAuth changes about refusals is the `WWW-Authenticate` header, which goes on *every* 401/403 once OAuth is configured — including a failed static-token request, because the server cannot tell which credential the caller meant to present, and claude.ai will not start the authorization flow at all without the `resource_metadata` parameter (Claude Code tolerates its absence, which is why a missing header is easy to ship and hard to notice). The `/.well-known/oauth-protected-resource*` routes (served by the crate's `metadata_router`) are registered outside the auth layer for the same reason `/health` is: discovery has to work for a caller who has no credential yet, and gating it behind the authentication it bootstraps makes the flow unstartable.

**Per-tool write-scope enforcement is not implemented.** `require_auth` sits in front of the whole `/mcp` endpoint and cannot see which MCP tool a request invokes — that is in the JSON-RPC body, which only rmcp parses — so any valid token currently grants full access, writes included. The validated scopes are parked in request extensions as an `oauth_resource_server::AuthorizedToken` so a later change can enforce them at tool dispatch.

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
- `docs/oauth.md` for anything touching the `mcp.oauth` key table or this server's own setup. Validator behavior, security rationale and provider recipes live in the [`oauth-resource-server`](https://github.com/St0nefish/oauth-resource-server) crate's own `README.md`/`docs/providers.md` (edited in that repo, not this one) — recipes there still only ever describe what was actually tested
- this file's architecture notes and module table, and `docs/ARCHITECTURE.md` (note the `docs/` prefix — it is not at the repo root)

Check each claim against the code rather than the plan that preceded it. Code comments explain the code as it is: cite issue numbers (`#286`), never local plan/review labels ("Step 3", "Batch B") or narration of how the change evolved.

## Module layout

| File | Purpose |
|---|---|
| `main.rs` | CLI entrypoint (clap subcommands) |
| `config.rs` | Config deserialization. `Granularity` is the `search` granularity enum shared with `mcp.rs`; `ResolvedConfig::effective_granularities` is the one definition of the enabled set. The "section listed but gated off" load warning fires only when `config.yaml` sets `search.granularities` itself (`yaml_sets_search_granularities`), never for the default. `mcp.disabled_tools` (`Vec<String>`, default empty) validates against `descriptions::TOOL_NAMES` — the one edge from this module into `descriptions.rs` — bailing with every unknown name at once, on any duplicate, and if the list would disable all six tools. `mcp.enabled_tools` (`Option<Vec<String>>`, default `None`, distinct from `Some(vec![])`) is the allowlist alternative — same unknown/duplicate validation (factored into `validate_tool_name_list`, shared with `disabled_tools`), plus its own "would disable everything" case: `Some([])` is rejected, and setting it alongside a non-empty `disabled_tools` is rejected too (exactly one of the two may be set). `ResolvedMcpConfig::disabled_tools` is the EFFECTIVE disabled set, computed once in `Config::resolve_inner`: `TOOL_NAMES` minus `enabled_tools` when that's `Some`, otherwise `disabled_tools` unchanged — every downstream reader (`enabled_tool_router`, `descriptions::tool_enabled`, `server::build_instructions`) reads this one field and has no `enabled_tools` awareness at all. `mcp.oauth`'s YAML schema, per-key defaults and validation (`OAuthConfig`/`ResolvedOAuthConfig`) are re-exported from `oauth_resource_server` (mcp-md-wiki#308); `resolve_mcp_oauth` is what's still this module's own — it layers this application's defaults (`required_scope` = `mcp:read` when neither scope key is set, `scopes_supported` = `[mcp:read, mcp:write]` when omitted, both via `apply_mcp_oauth_defaults`) and its own `resource_name` and validation-message wording on top of the crate's `OAuthConfig::resolve` |
| `validate.rs` | Frontmatter validation against the resolved `.kb-schema.yaml` cascade for each file's path (not the global config directly — `frontmatter` in `config.yaml` is only the implicit root schema) |
| `ingest.rs` | Indexing: `index_paths` (the only function that mutates Qdrant/state — chunk/embed/upsert changed paths, purge missing ones, refresh stale metadata) and `scan_for_dirty` (read-only detector: stat/hash-compares the corpus against `indexed_files` and produces a dirty-path worklist). `PathFilter` (`from_config`/`is_indexable`) is the one `indexing.include`/`exclude`/`exclude_files` predicate, and `partition_indexable` (build the filter, fail open + log loudly on a glob-build error, then split into indexable/filtered-out) is the one entry point every dirty-marking producer that reaches `ReindexQueue::mark_paths` from outside a reconcile calls before marking — `discover_files`'s full-corpus walk, `webhook.rs`'s push diff, and `write.rs`'s own write target(s), rewritten referencing documents, and rebase-pulled-in paths alike — so none of them can mark a non-indexable path dirty that a reconcile would have ignored (#278; `check_include_pattern` alone does not guarantee this, since it only checks `include`, never `exclude`/`exclude_files`). `scan_and_index` composes the two for callers with no worker (CLI, startup bootstrap). `heading::body_line_offset`-shifted chunk and section line numbers (counted from the top of the raw file, matching `get_document` — #286) and, when `chunking.heading_metadata` is on, the `heading_path`/`heading_level`/`heading_prefixes`/`section_key`/`section_line_start`/`section_line_end` payload fields (`qdrant.rs`'s `HEADING_*`/`SECTION_*` keys) get written alongside the existing ones — off by default (#286). `heading_prefixes` holds one `heading::heading_prefix_key` per contiguous run of the chunk's `heading_path` (`derive_heading_prefixes`), so `heading_prefix` matches a run starting at any level. Each `indexed_files` row stores `chunking_fingerprint` (every `chunking.*` field, except `target_chunk_size` while `heading_metadata` is on, + `CHUNKER_VERSION` — bump it when chunk text or payload logic changes); `scan_for_dirty` and `process_file`'s skip check treat a mismatch as dirty, so chunking changes re-chunk automatically on the next reconcile |
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
| `rerank.rs` | Cross-encoder reranking client; truncates each candidate to a byte budget from `reranking.max_document_bytes` before sending, with exponential-backoff retry |
| `mcp.rs` | The six MCP tools (rmcp): `search` (query-mode chunk/document/**section** retrieval and, with no `query`, the exhaustive enumeration formerly served by `list_documents`), `get_document`, `get_schema`, `update_schema` — thin handlers delegating to `retrieval`/`state`/`schema` — and the write tools `write_document`/`delete_document` (the former `create_document`/`edit_document`/`move_directory` unified into one upsert-and/or-relocate tool) — thin adapters over `write::write_document`/`write::delete_document` that map `WriteSuccess`/`WriteError` back onto the exact `CallToolResult`/`McpError` text and `data` payloads; they never index inline. `search`'s `section` granularity (`search_sections`, rejects `explain`/`fields`, requires `chunking.heading_metadata`) returns path-only section rows, no text; `heading_prefix` filters query results to a contiguous run of headings anywhere in the chunk's heading path (normalized via `heading::heading_prefix_key` on both the payload and query side), same flag gate, rejected without a `query`. `overlay_input_schema` rewrites the `search` schema per call from the effective granularity set: the `granularity` `enum`/description, plus `descriptions::search_property_descriptions` (rewords or removes properties that only apply to a disabled granularity or to no-query search). `get_document`'s `line`/`heading_path` (exclusive with each other), `levels_up` and `outline` (combinable with a selector to outline one section) resolve through `retrieval::parse_document_view_request`/`resolve_document_view`; `structured_content` is `retrieval::document_view_json` plus the envelope (#286). Tool and server descriptions come from `descriptions.rs`'s compiled+config-derived overlay, not literal strings in this file. `list_tools`/`get_tool` apply `overlay_description`, then `overlay_input_schema`, then `tool_schema::self_contained` — the last step, in every case. `KbSearchServer::enabled_tool_router` builds a fresh `ToolRouter` per call with every `mcp.disabled_tools` name disabled (rmcp's `disable_route`) and is what `list_tools`, `get_tool`, and a hand-written `call_tool` (defined in the same `#[tool_handler] impl ServerHandler` block, so the macro does not also generate one) all build their router from, so a disabled tool is hidden from listing/lookup and `tools/call` refuses it with the same `invalid_params("tool not found")` an unknown name gets. `mcp.disabled_tools` here is already the resolved, effective set — an `mcp.enabled_tools` allowlist configured instead is folded into it by `config.rs`'s `Config::resolve_inner` before this code ever runs, so `enabled_tool_router` has no allowlist-specific branch |
| `tool_schema.rs` | Pure JSON post-processor, `make_self_contained`, rewriting a `Tool::input_schema` into one with no `$ref`, no `$defs`/`definitions`, and no boolean `true`/`false` subschema in a schema position (data keywords — `enum`/`const`/`default`/`examples` — and a boolean `additionalProperties`/`unevaluatedProperties` are left alone) — the shape llama.cpp's grammar converter requires, since llama-server fails the whole `tools/list` request with HTTP 400 if one tool's schema doesn't convert (#288). Refs are inlined recursively, siblings (e.g. `description`) merged over the inlined copy with the sibling winning; a ref back to a name already being expanded (`schema::RawFieldDef.fields`'s self-reference) is cut to `{"type": ...}` at the repeat rather than expanding forever, so a nested `update_schema` `fields` entry advertises as a plain object below one level — the server still deserializes and validates it exactly as before. An unresolvable ref degrades to `{}` and fires a `debug_assert!` |
| `descriptions.rs` | Assembles MCP tool/server descriptions from three layers: compiled-in `assets/mcp/*.md` (mechanics true of every deployment), config-derived sentences (the hybrid/phrase retrieval-mode sentence; for `search`, a one-sentence `granularity_summary` naming only the `search.granularities` effective set — the full per-value text, `granularity_description`, is served only on the `granularity` property —, the no-query listing sentence only when an enabled granularity supports it, and a `heading_prefix` mention when `chunking.heading_metadata` is on), and per-KB policy loaded at runtime from `<mcp.extensions_path>/` in the served knowledge base (append-only, editable via `write_document`, no restart). Claude Code truncates a tool description or the server instructions at 2048 chars (`CLIENT_DESCRIPTION_CAP`; property descriptions are not truncated), so tests hold the compiled + config-derived text to `COMPILED_DESCRIPTION_BUDGET` (1500) — the server instructions against a realistic corpus in `server.rs` — and `warn_if_over_client_cap` logs a composed description over 2048. Detailed per-parameter rules belong on the parameter's doc comment (the served schema description), not in `assets/mcp/tools/*.md`; keep Rust paths and issue numbers out of those doc comments (`//` for maintainer notes) |
| `write.rs` | Transport-agnostic write pipeline extracted from `mcp.rs`'s write tools: `write_document`/`delete_document` taking a `WriteDeps` bundle, returning `WriteSuccess`/`WriteError`. Owns schema-frozen check, frontmatter validation, dedup gate, commit-message validation, the pre-commit rollback (remove/unstage on create, restore-from-HEAD on edit), `git::commit_and_sync`, and marking paths dirty on `WriteDeps::queue` (a `&ReindexQueue`, unlike `WriteDeps::state` no `None` mode — every write must mark its path). Every path a write is about to mark dirty — its own target path(s), any rewritten referencing documents, and `commit_outcome.rebased_paths` (paths pulled in from OTHER commits by this write's own rebase) — is filtered through `WriteDeps::indexing` (`ingest::PathFilter`/`partition_indexable`, same predicate as a reconcile, fail-open) before reaching the mark call, so a write's own target matching `include` but also `exclude`/`exclude_files` (e.g. the default-excluded `README.md`) is committed but never marked dirty, the same way a reconcile would never have indexed it (#278). Used by both `mcp.rs` (rmcp) and `web.rs` (HTTP) so the two transports share one behavior |
| `status.rs` | Process-global indexing run state (`INDEX_STATUS`) backing `/status` and `/metrics`: in-flight phase/progress, last-run outcome and counters, payload-index health. `IndexStatus::is_bulk_indexing` (a full run, or one over more than one file, is in flight — or the sticky `stale_chunking` latch is set) gates the "results may be incomplete" note on `section`/`heading_prefix` results. The latch is set by `ReconcileScan::mark_rechunk` the moment a reconcile scan finds a non-frozen file with a stale chunking fingerprint, and — unlike the per-scan `ReconcileScan` guard itself — does NOT clear when that guard drops: only `IndexStatus::clear_stale_chunking` clears it, which `ingest::scan_and_index` calls both when a scan finds nothing to re-chunk and when the run consuming its worklist finishes successfully, so the note stays lit through retry backoff, a permanent give-up, and the wait for the next reconcile (#286 round-6 L1). Frozen scopes never set the latch — `scan_for_dirty` skips them before the check — since they are never re-chunked until the schema is fixed |
| `webhook.rs` | Webhook handler: fetches + ff-only merges the push, diffs the range (`git::git_diff_name_status`), filters the diffed paths through `ingest::PathFilter` (the same `indexing.include`/`exclude`/`exclude_files` predicate a full reconcile applies, failing open on a glob-build error — #278), and marks only the indexable ones dirty on `WebhookState::reindex_queue` — does not index inline |
| `reload.rs` | `POST /admin/reload`: re-reads and re-validates `config.yaml`, swaps it into the live `SharedConfig`, and classifies each changed setting `applied` (read fresh on next use) / `restart_required` (baked into a startup-built value) / `reindex_scheduled` (`chunking.*` — re-chunked automatically via the per-file chunking fingerprint) / `reindex_required` (`ui.semantic_edges.*`). `mcp.disabled_tools` / `mcp.enabled_tools` is `applied`, reported under one shared `DiffField` (keyed on `mcp.disabled_tools`, the resolved effective field both settings feed): `KbSearchServer::enabled_tool_router` reads it fresh on every `tools/list`/`tools/get`/`tools/call`, no restart or metadata-refresh-tick wait — though an already-connected client caching an earlier `tools/list` is not sent `notifications/tools/list_changed`, since this server never advertises that capability. `mcp.enabled_tools` itself has no `ResolvedConfig` field of its own, so it lives in the test-only `RELOAD_DIFF_EXCLUDED` list rather than a second `DiffField` — the bidirectional drift test (#226) would otherwise demand one. Every `mcp.oauth.*` key is `restart_required`: the auth middleware's state (`oauth_resource_server::axum::AuthLayer`, including the `OAuthValidator` and its JWKS cache) is built once in `server.rs run_server` and never rebuilt (mcp-md-wiki#308). `mcp.oauth.required_scope` and `mcp.oauth.required_scopes` feed one resolved value, `ResolvedOAuthConfig::required_scopes`, so one `DiffField` (keyed on `required_scope`) reports a change to either — the same shared-field shape as `mcp.disabled_tools`/`mcp.enabled_tools` above — and `mcp.oauth.required_scopes` is in `RELOAD_DIFF_EXCLUDED` for the same reason `mcp.enabled_tools` is |
| `git.rs` | Git operations for the KB clone (clone, fetch with timeout, `commit_and_sync`: add→commit→fetch→rebase→push, returning the changed paths the rebase pulled in alongside the commit SHA). Owns `GIT_LOCK` and the `GitLock` guard — see the git serialization rule below |
| `secrets.rs` | The one place the five secret env vars (`source.git_token_env`, `webhook.secret_env`, `mcp.bearer_token_env`, and — via `config.rs`'s `resolve_inner` — `embedding.api_key_env`/`reranking.api_key_env`) are read. `resolve_secret` is `oauth_resource_server::env::config_value_from_env` (crate 0.4, `env` feature), not `secret_from_env`: a set plain variable comes back exactly as set (untrimmed, empty as `Some("")`, as `std::env::var(..).ok()` returned it) and only a `<name>_FILE`'s contents are trimmed, so adopting `_FILE` changed no existing value (#332). Both forms set (a blank plain variable yields to the file) or an empty/unreadable file is an error; `resolve_nonempty_secret` additionally reads empty as unset, for the git token and webhook secret. `git_token` is the `ResolvedConfig` shortcut used by every git-credential site (`mcp.rs`, `web.rs`, `ingest.rs`), re-read per use so a rotated file needs no restart. `StartupSecrets::resolve` (git token, bearer token, webhook secret) is what `run_server` calls once up front with `?`, so a bad combination refuses to start; `resolve_with` takes the lookup and file reader injected, which is how its abort behavior is tested without touching the environment |
| `server.rs` | Axum server (MCP + webhook + `/health`, `/status`, `/metrics` routes, plus the unauthenticated web UI routes from `web.rs` — `/`, `/assets/*`, `/api/graph`, `/api/search`, `/api/doc/{*path}`, `/api/schema/{*path}` — and the unauthenticated OAuth discovery routes `/.well-known/oauth-protected-resource[/mcp]` (the suffix derived from `mcp.oauth.resource`'s path, RFC 9728 §3.1 — `/mcp` for the usual resource) served by `oauth_resource_server::axum::metadata_router`, all merged in before the rate-limit `GovernorLayer` wrap; owns the Prometheus encoder); spawns the reindex worker and the periodic reconcile-sweep timer (`indexing.reconcile_interval_secs`). `require_auth`, from `oauth_resource_server::axum` (mcp-md-wiki#308), is dual-mode — see below. `auth_layer` builds the crate's `AuthLayer` from `static_bearer_token`'s `StaticTokenDecision` and the `OAuthValidator` (`build_with_decision`, `static_challenge(None)`), and is fail-closed by construction: an `AuthLayer::allow_unauthenticated()` pass-through only comes from a `StaticTokenDecision::Unauthenticated`, which the policy returns only for the explicit `mcp.allow_unauthenticated` with neither credential configured |
| `web.rs` | The knowledge-base web UI, served straight from the binary (`assets/ui/`, embedded via `include_str!`, no filesystem reads at request time). Docs-first shell: hash-routed home/browse/doc views with a sidebar document tree and a semantic-search results panel; the Cytoscape.js graph is secondary, reachable as a per-doc neighborhood view or a full-KB view (fcose layout, hover/zoom-gated labels). `UiState` mirrors `StatusState`/`KbSearchServer::deps()`, reusing the same `Arc<OnceCell<StateDb>>` as `StatusState` rather than opening a second pool. Handlers: static shell/asset routes, `/api/graph` (nodes from `StateDb::all_document_summaries`, edges from `StateDb::all_links` filtered to the current node set, server-computed type palette), `/api/search` (thin wrapper over `retrieval::search`), `/api/doc/{*path}` GET/POST/DELETE (GET via `retrieval::get_document`, with the MCP tool's range/section/outline query params resolved through the same `retrieval::parse_document_view_request`/`resolve_document_view`/`document_view_json` — same contract, including #290's `section_max_bytes` cap on an unranged read, which is why the UI's viewer and editor fetch `?start_line=1`: a caller-named range is exempt, and the editor posts the whole body back on save; `content_hash` always over the whole file; POST/DELETE thin adapters over `write::write_document`/`write::delete_document`), `/api/schema/{*path}`. Deliberately unauthenticated — same open-route posture as `/health` — because the deployment sits behind Authentik via Traefik |

## Workflow

**Merge train (serialized, test-once)**, per the knowledge base:
`dev/tools/merge-train-pattern.md` (it replaces the PR gate and post-merge
re-test of Pattern C in `dev/tools/repo-workflow-patterns.md`). `master` takes
no direct pushes and is protected by a repo **ruleset** (not classic branch
protection): direct push disabled, status checks required.

- Work on a branch and open a PR against `master`. Opening an owner PR arms
  squash auto-merge (`auto-merge.yml`, as the GitHub App, owner-authored PRs
  only), and **an armed PR is a queued PR**. Arming fires `train-trigger.yml`
  (`pull_request_target`, which only runs `gh workflow run train.yml --ref
  master`), and so does a push to an armed PR. A contributor's PR is never
  armed automatically: the owner arming it is the approval and the merge
  trigger in one act.
- `train.yml` runs one PR at a time (concurrency group `train`, never
  cancelled, always master's copy of the workflow): it takes the oldest armed
  PR whose current head has no train status, pins that head and current
  `master`, squashes the head onto `master` locally
  (`.github/scripts/train-squash.sh`, fixed identity and dates, so every job
  rebuilds the identical commit and asserts its tree), and tests exactly that
  tree. The fast tier (`checks.yml` `lint` + `test`) posts `ci-fast`; then the
  slow tier runs in parallel (`checks.yml` `qdrant-integration`, and
  `build-image.yml`, which builds the `linux/amd64` image, smoke-tests it and tags
  it `:tree-<tree hash>`) and `ci-slow` is posted. GitHub's auto-merge then
  squash-merges the PR, and the train waits for that before dispatching the
  next run. Every status the train posts carries the run as its target URL and
  a `train:` description. A conflict with `master`, a failure, a cancellation
  or a merge that does not happen within 10 minutes turns both statuses
  `failure` and comments on the PR. A failed head is not retried until it
  changes: push a fix (the new head re-queues), or re-test the same head after
  a flaky failure with `gh workflow run train.yml --ref master -f pr=<n>`. A
  15-minute schedule restarts the queue after a run that died; each run first
  resets any train `success`/`pending` left on an armed head by such a run.
- A PR that touches no code path (`.github/scripts/code-paths.sh`, the one
  list the train, `master.yml` and `pr-fast.yml` share; a change that only
  moves this crate's own version in `Cargo.toml`/`Cargo.lock` does not count)
  still queues, so it lands in order, but the train posts both statuses green
  without running anything.
- **Required checks.** The ruleset requires exactly `ci-fast` + `ci-slow`,
  statuses only `train.yml` posts. There is no per-PR gate workflow: a PR is
  tested once, by the train, on its squash onto `master`. Never list individual
  job names as required checks.
- `pr-fast.yml` gives PRs the owner did not open (and that no bot opened) the
  fast tier on GitHub-hosted runners, as the non-required `pr-fast` check — an
  unreviewed PR's code never runs on the self-hosted runners, and it cannot
  start the slow tier.
- Merges are **squash only** (`allow_merge_commit: false`,
  `allow_rebase_merge: false`), titled `<PR title> (#<number>)`.
- **Auto-merge is the workflow, not an escalation.** An owner PR landing on
  green CI without a human reading the diff first is the intended, configured
  behavior — CI is the gate. Do not disable auto-merge on a PR, do not open
  one as a draft to dodge it, and do not ask whether it should be left armed:
  the answer is always yes. Review happens before the PR is opened (the
  author, agent or human, verifies its own diff) and after it lands
  (`/code-review` on demand). If a change is genuinely too risky for that, the
  fix is to say so and not open the PR yet — not to open one and hold it.
- The ruleset's required-status-checks rule has
  `strict_required_status_checks_policy: false` — PR branches do **not** need
  to be up to date with `master` before merging, and must not: the train tests
  the squash onto current `master`, not the head, so the head stays behind
  `master` by design and the strict flag would block every merge.
  Serialization comes from the train instead. **Do not routinely rebase a PR
  branch just because `master` moved, and do not add a bot that
  force-pushes/rebases open PRs when `master` advances.** Rebase only to
  resolve a real conflict (the train reports it on the PR).
- On every push to `master`, `master.yml` attaches the train's image to the
  merge commit — no build, no tests: it looks up `:tree-<the commit's tree>`
  and adds `:sha-<commit>` and `:<x.y.z>-dev.<n>` to that digest, and moves
  `:dev` if the commit is still master's tip. A code-path merge with no
  `:tree-` image (one that bypassed the train, or a server-side squash that
  differs from git's) gets a warning and the fallback: full `checks.yml`, then
  `build-image.yml` on the commit. A docs-only or version-roll merge gets no
  image, and `:dev` stays where it is. **A merge deploys nothing:** it never
  touches `:latest` or Watchtower.
- **Watching a PR** covers the train run that tests it (`select` → `fast` →
  `slow-integration`/`slow-image` → `land` → `finish`), the merge, and
  `master.yml` on the merge commit, each reported by its own outcome, not just
  "merged". A green `master.yml` run that tagged an image is the signal the
  commit is releasable (it has its `:sha-<commit>` image). **Watching a
  release** is a separate watch: `release.yml`'s `resolve` → `check` →
  `verify-smoke` (/ `verify-full`) → `release` → `roll` + `arm64`, and it is not
  done until Watchtower reports `failed: 0`, `:latest` actually moved to the new
  digest, the roll PR (if any) is armed, and the `arm64-image.yml` run it
  dispatched has tagged `:latest-arm64`.
- `fix #N` in the merge commit auto-closes GitHub issues.
- Branches auto-delete after merge.
- Pre-commit hook enforces `cargo fmt` + `cargo clippy` (activate with
  `./scripts/setup-dev.sh` after cloning).
- **No need to batch PRs to save builds or restarts.** Merging does not
  deploy, so several small PRs cost only their train runs (queued PRs wait
  for each other's full pipeline — the accepted cost of testing once); only a
  release restarts the live service.

### Releasing

`Cargo.toml`'s version on `master` is always the **next** release; the tag is
derived from it, never typed. A change that warrants a minor or major bump
raises the version in its own PR before it merges; patch bumps are automatic.
The release itself is one owner-only step, and nothing else moves `:latest`:

```bash
gh workflow run release.yml --ref master                  # release what :dev names
gh workflow run release.yml --ref master -f sha=<commit>  # or one master commit
# add -f full_tests=true to re-run the whole checks.yml first
```

`release.yml` (trigger: `workflow_dispatch` only; concurrency `release`, never
cancelled) never builds; it promotes the train's tested image. `resolve` pins
the commit and digest (`:dev`'s digest and the newest master commit whose tree
is that image's `revision` label, or the given `sha` and its `:sha-<commit>`
digest; `:sha-<commit>` must name the pinned digest either way). `check` fails
unless the owner started the run (`github.triggering_actor`), the commit is on
`master`, `v<Cargo.toml version>` is either absent and would be the highest
stable `v*` tag or already points at this commit (a resumed release), and the
commit's `CHANGELOG.md` `[Unreleased]` section is non-empty. `verify-smoke`
smoke-tests the pinned digest (`verify-full` runs `checks.yml` too, with
`full_tests`). `release` (environment `release`, self-hosted) creates the tag
and the release at the commit as the GitHub App (the `[Unreleased]` section is
the notes), uploads `mcp-md-wiki-linux-amd64` copied out of the
image, re-checks that the tag is the highest stable one, retags the digest
`:vX.Y.Z` and `:latest` (no rebuild) and triggers Watchtower on atlas. `roll`
then opens `release: roll version to <next>` as the App (patch bump in
`Cargo.toml`/`Cargo.lock`; `[Unreleased]` becomes `## [X.Y.Z] - <date>` under
a fresh `[Unreleased]`, link references updated), arms it, and dispatches the
train; it skips when `master` already moved past the released version, and
leaves the PR unarmed for the owner when `master`'s `[Unreleased]` changed
after the released commit. `arm64` dispatches `arm64-image.yml` for the
released commit as a separate run (the release does not wait for it, and an
arm64 failure does not fail the release). No `:sha-<commit>` image means no
release.

Recovery: every step is idempotent — a transient failure is
`gh run rerun <id> --failed`, or dispatch again with the same `sha`. If the
released commit is genuinely broken, merge the fix and release that.

Image tags (`ghcr.io/st0nefish/mcp-md-wiki`, `linux/amd64` only, all one digest
per tested tree):

| Tag | Set by | Meaning |
|---|---|---|
| `:tree-<tree hash>` | `train.yml` (`build-image.yml`) | The build of a tested squash, after its smoke test; labels: `revision` = the tree hash, `version` = `Cargo.toml`'s version |
| `:sha-<full commit sha>` | `master.yml` | Immutable; the merge commit's image; what `release.yml` resolves |
| `:<x.y.z>-dev.<n>` | `master.yml` | Immutable; x.y.z = `Cargo.toml`'s version (the next release), n = commits since the latest release tag |
| `:dev` | `master.yml` | Newest master commit that has an image (only master's tip moves it) |
| `:latest` | `release.yml` | Most recent release; the only tag Watchtower deploys |
| `:vX.Y.Z` | `release.yml` | Pins one release |
| `:sha-<commit>-arm64`, `:vX.Y.Z-arm64`, `:latest-arm64` | `arm64-image.yml` (dispatched by every release, or by hand) | Single-platform `linux/arm64` builds of a master commit (default: the latest release), smoke-tested but not train-tested; never added to the amd64 tags above, so Watchtower never sees them |

`:build-<tag>` (staging) and `:buildcache-<arch>` (cargo-chef layer cache) also
exist but are not images to run.

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
