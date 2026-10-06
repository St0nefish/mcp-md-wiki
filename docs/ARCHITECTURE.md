# Architecture

<!-- verify-merge-b: inert marker for concurrent-PR auto-merge test, safe to delete -->

Current-state reference for mcp-md-wiki. For setup instructions see [`deploy/USAGE.md`](../deploy/USAGE.md); for config options see [`deploy/config.example.yaml`](../deploy/config.example.yaml).

## Overview

`mcp-md-wiki` is a single Rust binary that combines four concerns:

- **Indexing pipeline** — walks a markdown knowledge base, chunks and embeds files, and stores vectors + state
- **MCP server** — exposes exactly six tools over Streamable HTTP (port 8001): `search`, `get_document`, `write_document`, `delete_document`, `get_schema`, and `update_schema`
- **Webhook handler** — receives push events from a Git forge, pulls changes, and marks the changed paths dirty for the reindex worker (see [Webhook Flow](#webhook-flow))
- **Web UI** — a docs-first browser and semantic-search UI, served at `/` on the same port, with a Cytoscape graph view and a full create/edit/move/delete document editor; deliberately unauthenticated by design (see [Web UI](#web-ui))

All four share the same binary and config. The `serve` subcommand runs the server (MCP + webhook + web UI); the remaining subcommands are standalone CLI operations.

### Subcommands

| Subcommand | Purpose |
|---|---|
| `serve` | Start the MCP server, webhook handler, and web UI |
| `index` | Incremental reindex (changed files only) |
| `index --full` | Full reindex — drops Qdrant collection and re-embeds everything |
| `validate` | Check frontmatter on all files without indexing |
| `status` | Print collection stats and state DB info, including the document-metadata count (warns if it lags behind indexed files) |
| `health` | Query the running server's `/health` endpoint |
| `search` | Search the knowledge base from the CLI, through the same `retrieval::search` core the MCP `search` tool (query mode) uses — ranked results only, no enumeration mode |
| `get` | Retrieve one document (or a line range of it) by path from the CLI, resolved the same way as the MCP `get_document` tool |
| `reproject-fields` | Rebuild `document_fields` from stored frontmatter JSON (state DB only, no re-embed) |

## Docker Topology

Three services, all defined in `docker-compose.yml` (or a hardware-specific template from `deploy/templates/`):

```text
┌─────────────────────────────────────────┐
│                 mcp-md-wiki                  │
│  (MCP :8001, webhook /hooks/reindex)    │
│                                         │
│  ┌──────────────┐  ┌───────────────┐   │
│  │  ingest.rs   │  │   mcp.rs      │   │
│  │  (indexer)   │  │   server.rs   │   │
│  │  webhook.rs  │  │   retrieval.rs│   │
│  └──────┬───────┘  └──────┬────────┘   │
└─────────┼─────────────────┼────────────┘
          │ gRPC :6334       │ gRPC :6334
          ▼                  ▼
┌─────────────────┐   (same Qdrant instance)
│    qdrant       │
│  gRPC :6334     │
│  REST  :6333    │
└─────────────────┘

          ▲ HTTP :8080
┌─────────────────┐
│   embeddings    │
│  (llama.cpp)    │
│  OpenAI-compat  │
└─────────────────┘
```

| Service | Image | Port(s) |
|---|---|---|
| `qdrant` | `qdrant/qdrant` | 6334 (gRPC, used by mcp-md-wiki), 6333 (REST, debugging) |
| `embeddings` | `ghcr.io/ggml-org/llama.cpp:server[-cuda12/-rocm/-vulkan]` | 8080 (internal only) |
| `mcp-md-wiki` | `ghcr.io/st0nefish/mcp-md-wiki` | 8001 (MCP + webhook, exposed to host) |

The mcp-md-wiki service waits for both `qdrant` and `embeddings` to pass their healthchecks before starting.

The MCP port (8001) is the only externally exposed port. This service is designed for intranet/tailnet deployment — it is not hardened for direct public internet exposure.

## Module Layout

| File | Purpose |
|---|---|
| `main.rs` | CLI entrypoint (clap subcommands), startup wiring |
| `config.rs` | Config deserialization (`config.yaml` + env-var overrides) |
| `ingest.rs` | Indexing pipeline: discover → hash → validate → chunk → embed → upsert |
| `heading.rs` | The heading model shared by `chunk.rs` and `retrieval.rs`'s outline/section modes, so chunk attribution and `get_document` cannot disagree: `HeadingTree::parse` detects headings with pulldown-cmark (top-level ATX and setext headings; not inside code, HTML blocks, blockquotes, lists or footnotes), stores each heading's rendered plain text (invisible characters that never affect rendering removed, joiners/direction marks/variation selectors kept, whitespace collapsed, capped at 200 characters), level, parent and section/subtree line ranges; `normalize_heading_text`/`heading_prefix_key` are the one normalization (every invisible format character dropped, plus Unicode case folding, re-capped after folding) behind `get_document`'s `heading_path` and `search`'s `heading_prefix`; `body_line_offset` maps body lines to raw-file lines past the frontmatter |
| `chunk.rs` | Section-aware markdown chunker over `heading::HeadingTree`. With `chunking.heading_metadata` off, small consecutive sections merge up to `target_chunk_size`; with it on, no chunk crosses a heading boundary, so every chunk is attributed to exactly one heading's section or the preamble |
| `embed.rs` | Embedding API client (async-openai, batched, exponential backoff) |
| `qdrant.rs` | Qdrant gRPC operations: collection and payload-index setup, upsert, delete, dense/hybrid/grouped search and recommend-by-point-id |
| `state.rs` | SQLite state DB (sqlx): tracks relative path → content hash + chunk count + schema fingerprint + chunking fingerprint, plus the `documents`/`document_fields` metadata index and the `document_links` graph-edge table (see [State Model](#state-model)) |
| `document_fields.rs` | Projects frontmatter JSON into filterable `document_fields` rows (dot-path flattening, array expansion, numeric coercion) |
| `schema.rs` | `.schema.yaml` cascade: parsing, cascade merge, `SchemaCache` tree walk + resolution (fail-fast: `SchemaCache::build` returns a `SchemaBuildError` listing every invalid file, and `apply_rebuild` keeps the last good shared cache on a refused runtime rebuild — see [Schema Cascade](#schema-cascade)), type/value checking, schema fingerprinting,, the per-directory `dedup:` override of `write.dedup_*` (#272; not part of the fingerprint), and `SchemaCache::generation`: every build is stamped from a global counter when it starts, and `store_shared` discards a cache older than the one already shared, so two rebuilds finishing out of order (two concurrent `update_schema` calls) never leave the older rules in place |
| `retrieval.rs` | Shared retrieval core: `search`, `get_document`, and `list_documents`, consumed by `mcp.rs`, `web.rs`, and the `search`/`get` CLI subcommands alike. `history_json` is `/api/history`'s commit-listing envelope; `document_changes_json` is the git-free projection behind `get_document`'s `history` parameter (#257: `{date, author, subject, operation?}` per change, no revision id or email). `document_view_json`/`outline_entry_json` serialize a `get_document` view for both transports, omitting every false/null/default key |
| `sparse.rs` | Pure-Rust BM25-style sparse-vector tokenizer (FNV-1a term hashing) feeding the `sparse` named vector for hybrid retrieval — no model, no network |
| `rerank.rs` | Cross-encoder reranking client; truncates each candidate to a byte budget from `reranking.max_document_bytes` before sending, with exponential-backoff retry |
| `mcp.rs` | The six MCP tool handlers (`search` — covering both ranked query results and, with no `query`, the exhaustive enumeration formerly served by `list_documents` — `get_document`, `write_document`, `delete_document`, `get_schema`, `update_schema`): input validation, result formatting; delegates to `retrieval.rs` / `write.rs` / `state.rs` / `schema.rs`. `KbSearchServer::enabled_tool_router` builds a fresh `ToolRouter` per request with every name in `mcp.disabled_tools` disabled (rmcp's `disable_route`), so `list_tools`, `get_tool`, and the hand-written `call_tool` all agree on which tools currently exist — a disabled tool is hidden from listing/lookup and `tools/call` refuses it exactly as it would an unknown name. `mcp.disabled_tools` here is always the *resolved, effective* set: `mcp.enabled_tools` (an allowlist) is a config-side alternative that `Config::resolve_inner` (config.rs) folds into this same field — its complement against `descriptions::TOOL_NAMES` — before `mcp.rs` ever runs, so this code has no separate allowlist path to know about. Nothing git-related reaches an MCP tool caller: the write-result mappers (`create_edit_success_to_result`, `delete_success_to_result`, `move_directory_success_to_result`, `batch_write_success_to_result`, and `update_schema`'s own result) drop `WriteSuccess::outcome`/`sha`/`sync_failure_cause` (a committed-but-unpushed write is a plain success), report only `version` and `merged_with_other_changes` (when the write was merged with someone else's change to the same document), and map a pre-commit failure to `not_saved_error`/`not_saved_unverified_error`, logging the cause rather than relaying it; `get_document`'s opt-in `history` (#257) is git-free too. Every result is one JSON object built with rmcp's `CallToolResult::structured` — `structured_content`, plus a single text block that is its compact serialization — so no fact exists only in prose (Claude Code shows the model `structured_content` alone); keys that would be false/null/empty/default are omitted. Write results carry `path` and `action`, and a diff only for an edit or move that changed the document; `update_schema` carries `warning` when the change is saved but not in effect yet, and resolves the inherited parent definition from disk under the write lock (`SchemaCache::resolve_from_disk`). `search` filters teach on error in both modes (`check_filter_vocabulary`): an unknown field (no schema declares it, not built in, no document carries it) or a value outside the closed set of the scopes `path_prefix` names (`SchemaCache::filter_closed_values`, given the needle `search` itself matches on: a trailing `/` is ignored, a leading one is not) that no document uses is refused with the options listed (capped at 30). A value filter is refused only when no document could match it: an `all_of` as soon as one of its values is refused, a scalar or array (any-of) only when every value is. It fails open when the metadata index is unavailable. `get_schema` never lists a derived field (`domain`), even one a scope declares; its `values_in_use` adds, from the metadata index (`StateDb::top_values`/`fields_in_use`), the most-used values of open fields (`in_use`) and the fields documents under the path carry that no scope declares (`other_fields_in_use`: `SchemaCache::declared_field_paths` and the derived fields are left out) |
| `descriptions.rs` | Assembles each tool's and the server's MCP description from compiled-in `assets/mcp/*.md` (mechanics true of every deployment), config- and corpus-derived sentences (`search`'s quoted-phrase and enumeration/granularity lines; for the server instructions, `top_level_areas_sentence` and the one `SCHEMA_POINTER_SENTENCE` to `get_schema` — no vocabularies, scoped folders or authoring rules), and a per-KB policy extension loaded at runtime from `mcp.extensions_path` in the served knowledge base (append-only, editable via `write_document`). One home per rule: a description carries what a tool does and the cross-parameter workflow, per-parameter rules live on schema property descriptions (not truncated), responses describe themselves, and the instructions repeat no tool description. Compiled + config-derived text is held to `COMPILED_DESCRIPTION_BUDGET` (600 chars) by tests, leaving room for extensions under Claude Code's 2048-char truncation (`CLIENT_DESCRIPTION_CAP`); `warn_if_over_client_cap` logs a composed description or server instructions over that cap |
| `write.rs` | Transport-agnostic write pipeline (`write_document`/`delete_document`) shared by the MCP write tools and `web.rs`'s `POST`/`DELETE /api/doc/{*path}`: the `.schema.yaml` path guard (`WriteError::SchemaFile` — schema files are edited only through `update_schema`), frontmatter validation, dedup gate (`effective_dedup`: the governing schema's `dedup:` override, else `write.*` — #272), pre-commit rollback, `git::commit_and_sync`, and marking the written path dirty on the `ReindexQueue`. Every path a write is about to mark dirty — its own target path(s), rewritten referencing documents, and paths pulled in from OTHER commits by the rebase step (`commit_outcome.rebased_paths`) — is filtered through `ingest::PathFilter` (the same `indexing.include`/`exclude`/`exclude_files` predicate a reconcile applies) before joining that mark call, so a write's own target can be committed without being marked dirty if it matches `exclude`/`exclude_files` (#278). A `.schema.yaml` among those paths (a directory move, a rebase) instead queues a full reconcile (`mark_dirty` → `ReindexQueue::mark_schema_changes`). **Concurrency:** every write starts with `lock_and_sync` — take `GIT_LOCK`, then `sync_clone` (`git::sync_to_remote`: fetch and fast-forward, or rebase leftover unpushed commits, skipped without a remote; paths it pulls in are marked dirty through the same filter, since the webhook that follows will find nothing new) — and holds that one guard across read, edit, validation (lint included), write, commit and any rollback; only a create, whose content is fixed, is validated and dedup-gated before the lock. Validation uses the schema cache as it stands: a schema file the sync pulls in only queues a full reconcile, whose worker rebuilds the cache later, so a write racing a schema push is validated against the rules from before it. A `DocChange` says what the caller asked for: `Create`, `Relative` (a closure applied to the content read under the lock — surgical, `frontmatter_patch`, `append`), or `Replace`, an absolute change that, like a single-document move and a delete, requires `expected_version` (the git blob id, `document_version`; `WriteError::VersionRequired` without it). A stale `Replace` is three-way merged (`git::merge_text` over `git::cat_blob` of the caller's version as base): clean → saved with `WriteSuccess::merged`, conflict or unknown base → `WriteError::EditedElsewhere` (in a repository whose checkout applies eol/clean filters, or that uses SHA-256 objects, the version names no stored blob, so every stale `Replace` has an unknown base); a stale move or delete is refused. A `Relative` edit still lands over a stale `expected_version` and is flagged `merged`, a create carrying an `expected_version` is refused as `WriteError::NotFound` (the document it was read from is gone), and a `Relative` edit may not grow a document past `MAX_CONTENT_LEN` (512 KiB, the cap `mcp.rs` and `web.rs` apply to created and replaced content). An edit that leaves the document exactly as it is succeeds without a commit (`diff` empty, `sha` HEAD as it stands; a batch of nothing but such edits likewise). A single-document create's or edit's filesystem write that fails is undone before the failure is reported (`write_or_undo`: the previous content back, or the partly written new file removed), since an uncommitted change in a tracked file fails every later pre-write rebase. `move_directory` normalizes its two prefixes to `a/b` form (`normalize_dir_prefix`: `./notes/`, `/notes` and `notes//.` all read as `notes`), refuses a source that is the KB root, and re-walks its subtree under the lock, refusing (`DirectoryMoveError::EditedElsewhere`) if it differs from the pre-lock snapshot. A `CommitSyncError::Conflict` (rebase conflict or non-fast-forward push; `git.rs` has already dropped this write's own commit, and nothing older) re-syncs and re-applies, up to `MAX_WRITE_ATTEMPTS` (3); a batch's rollback restores the snapshots taken under the lock, for every document it had started to write (a create that collided with an existing file is never removed), and its error names any it could not restore |
| `reindex.rs` | `ReindexQueue` and its single background worker (`run_worker`): every producer (write tools, webhook handler, startup, the periodic reconcile sweep) marks paths — or a full reconcile — dirty and returns immediately; the worker drains the queue into `ingest::index_paths` with coalesce-don't-drop semantics and transient-vs-permanent retry/backoff (see [Webhook Flow](#webhook-flow)) |
| `web.rs` | The knowledge-base web UI — docs browser, semantic search, Cytoscape graph view, and a create/edit/move/delete editor — served straight from the binary and deliberately unauthenticated (see [Web UI](#web-ui)) |
| `server.rs` | Axum server: MCP route, webhook route, `/health`/`/status`/`/metrics`/`POST /admin/reload`, and the unauthenticated web UI routes from `web.rs`, all merged in before the rate-limit `GovernorLayer` wrap; spawns the reindex worker and the periodic reconcile-sweep timer (`indexing.reconcile_interval_secs`). OAuth 2.1 resource-server auth (JWT access-token validation, JWKS discovery/caching/rotation, the RFC 9728 protected-resource metadata) lives in the [`oauth-resource-server`](https://github.com/St0nefish/oauth-resource-server) crate (mcp-md-wiki#308); this file builds the crate's `AuthLayer`/`require_auth` middleware and `metadata_router` from the resolved `mcp.oauth` config. See [oauth.md](oauth.md) |
| `webhook.rs` | Webhook handler: provider signature verification, branch filter, `git fetch` + `git merge --ff-only`, then diffs the pulled range, filters it through `ingest::PathFilter` (same `indexing.include`/`exclude`/`exclude_files` predicate a reconcile applies, failing open on a glob-build error — #278), and marks exactly the indexable paths dirty on the `ReindexQueue`, plus a full reconcile (which rebuilds the schema first) when the push changed a `.schema.yaml` — it never indexes inline (see [Webhook Flow](#webhook-flow)) |
| `reload.rs` | `POST /admin/reload`: re-reads and re-validates `config.yaml`, swaps it into the live `SharedConfig`, and classifies each changed setting as `applied` (read fresh on next use), `restart_required` (baked into a startup-built value), `reindex_scheduled` (`chunking.*`, re-chunked automatically via the per-file chunking fingerprint), or `reindex_required` (`ui.semantic_edges.*`) |
| `status.rs` | Process-global indexing run state (`INDEX_STATUS`) backing `/status` and `/metrics`: in-flight phase/progress, last-run outcome and counters, payload-index health, and the refused-schema-rebuild record (`schema_error`, `kb_schema_invalid`) |
| `validate.rs` | Frontmatter validation against the resolved `.schema.yaml` cascade for each file's path (falls back to the `frontmatter` config as the implicit root schema); a derived field (`domain`) is never required of an author nor filled from a schema `default` |
| `git.rs` | Git subprocess helpers: token injection, URL redaction, fetch/merge with timeout, serialized by `GIT_LOCK`. `commit_and_sync` distinguishes a conflict from a transport failure: a rebase conflict or a non-fast-forward push rejection drops the call's own commit (`drop_own_commit`: the branch is reset to HEAD as it was before that commit, so the clone is never left diverged and an older commit a remote outage left unpushed stays) and returns `CommitSyncError::Conflict`; a fetch, push or non-conflict rebase failure keeps the local commit (`PostCommit`, reported to callers as success, pushed by a later write). An outage never blocks a write, but a hung remote holds `GIT_LOCK` for up to `GIT_TIMEOUT` (120 s) per fetch, and a write makes two (`sync_to_remote`'s and `commit_and_sync`'s own). `sync_to_remote` (before every write) fast-forwards or rebases such leftover commits onto the remote; only when that rebase genuinely conflicts does it park them (HEAD saved under `refs/mcp-md-wiki/unsynced/<sha>`, the clone reset to the fetched tip), and a rebase that fails for another reason (a dirty tracked file, say) is aborted and logged, the clone left as it is. `git_ok` redacts the whole failure message, argv included, since a fetch or push argv carries the token-bearing URL. `cat_blob`, `merge_text` (`git merge-file -p`) and `blob_id` back `write.rs`'s concurrency handling. A test-only task-local hook (`TEST_HOOK`, `HookPoint::{BeforeLock, BeforeSync, BeforePush}`) lets tests inject a concurrent writer at an exact point; `test_hook` compiles to nothing outside tests |

## Indexing Pipeline

```text
discover files (relative paths)
        │
        ▼
 read file content
        │
        ▼
 compute SHA256 hash
        │
   unchanged? ──── yes ──► skip (increment skipped counter)
   (same hash, schema fingerprint and chunking fingerprint)
        │ no
        ▼
 validate frontmatter
        │
  invalid? ─── yes ──► warn + skip (increment invalid counter)
        │ no
        ▼
 chunk markdown (section-aware)
        │
  empty? ──── yes ──► warn + skip (increment empty counter)
        │ no
        ▼
 embed chunks in batches
 (async-openai, configurable batch_size, exponential backoff)
        │
        ▼
 upsert points in-place by deterministic UUID5 id
 (id = UUID5(namespace, "relative_path::chunk_index"))
        │
 file shrank? ─── yes ──► delete tail points (old_count - new_count)
        │
        ▼
 update SQLite state (relative_path → hash + chunk_count
                      + schema and chunking fingerprints)
        │
        ▼ (after all files)
 orphan removal: delete Qdrant points + state rows
 for paths in state DB not found on disk
 or now excluded by indexing.include/exclude/exclude_files
        │
        ▼
 log structured summary:
 discovered / indexed / skipped / invalid / empty /
 read_errors / orphans_removed / elapsed_secs
```

### Key design decisions

**Relative paths as canonical keys.** File paths are stored relative to `source.data_path` everywhere: as the Qdrant `file_path` payload field, as the SQLite state key, and as the UUID5 input for point IDs. This makes the index portable across mount points. Upgrading from an older version that stored absolute paths requires a `--full` reindex.

**Upsert-in-place, no pre-delete window.** Points are upserted by their deterministic ID. A file that grows adds new tail points; a file that shrinks has its tail trimmed after the upsert. There is no window where a file's points are absent from the index.

**`mtime` payload field.** Each point carries an integer Unix-timestamp `mtime` field (indexed as a Qdrant integer index). `search`'s `modified_after`/`modified_before` parameters filter on it directly (see [Retrieval](#retrieval) below); documents indexed before `mtime` tracking was introduced may have `mtime = 0` and can be silently excluded by a recency filter.

## State Model

### SQLite (`state.db`)

Written to `<source.data_path>/state.db` — by default `/data/state.db`, which is inside the knowledge-base volume. No separate mount is required.

| Column | Type | Notes |
|---|---|---|
| `file_path` | TEXT PK | Relative path from `data_path` |
| `content_hash` | TEXT | SHA256 hex digest of file content |
| `chunk_count` | INTEGER | Number of chunks produced on last index |
| `schema_hash` | TEXT | Fingerprint of the `.schema.yaml` cascade the file was last validated against (see [Schema Cascade](#schema-cascade)); added via a guarded `ALTER TABLE ... ADD COLUMN`, since there is no migration runner |
| `chunking_fingerprint` | TEXT | Fingerprint of every `chunking.*` setting (except `target_chunk_size` while `heading_metadata` is on, where it has no effect) plus the code-level `CHUNKER_VERSION` the file was last chunked under (`ingest::chunking_fingerprint`). The reconcile scan and the indexer's skip check treat a mismatch as dirty, so a chunking change re-chunks automatically. Added the same guarded way, defaulting to `''`, which never matches — so rows from before the column existed are re-chunked once (#286). |

### Document metadata index (`documents`, `document_fields`)

Alongside `indexed_files`, `state.db` holds a document metadata index that backs `list_documents`.

**`documents`** — one row per file:

| Column | Type | Notes |
|---|---|---|
| `file_path` | TEXT PK | Relative path from `data_path` |
| `title` | TEXT | From frontmatter |
| `description` | TEXT | From frontmatter |
| `frontmatter` | TEXT (JSON) | Full frontmatter, stored faithfully |
| `mtime` | INTEGER | File modification time (Unix timestamp) |
| `content_hash` | TEXT | SHA256 hex digest of file content |
| `chunk_count` | INTEGER | Number of chunks produced on last index |
| `indexed_at` | INTEGER | When this row was last written |

**`document_fields`** — inverted index over frontmatter, one row per (file, field, value):

| Column | Type | Notes |
|---|---|---|
| `file_path` | TEXT | Joins to `documents.file_path` |
| `field` | TEXT | Dot-path (nested frontmatter flattens, e.g. `planning.prep_minutes`) |
| `value_text` | TEXT | String form of the value; booleans store as `"true"`/`"false"` |
| `value_num` | REAL | Numeric form, when applicable (enables `gte`/`lte`/`gt`/`lt` range queries); booleans store as `1.0`/`0.0` |

Arrays produce one row per element. This table is projected by `document_fields.rs` from the `documents.frontmatter` JSON and is what `list_documents`'s `filters` and `order_by` query against — with one exception: `title` and `description` are deliberately excluded from this projection and filtered against their dedicated `documents.title`/`documents.description` columns instead (`PROMOTED_FIELDS` in `document_fields.rs`), so a `filters` entry for either field still works, it just doesn't go through `document_fields`. Because those columns each hold one scalar, `all_of` with more than one value on `title` or `description` is unsatisfiable — the query resolves to no matches rather than erroring.

**Backfill:** existing deployments self-heal — on the next `index` run, files unchanged by content hash but missing metadata get their frontmatter parsed and stored into `documents`/`document_fields`, with no re-embedding and no Qdrant writes. No operator migration step is needed. After a field-projection rule change, `mcp-md-wiki reproject-fields` rebuilds `document_fields` from the stored frontmatter JSON alone (no markdown re-read, no re-embed).

`reproject-fields` is safe to run against a live server: `StateDb::reproject_all_fields` re-reads each document's stored frontmatter *inside* the same transaction that rewrites its projection (rather than snapshotting paths and frontmatter up front), and retries on `SQLITE_BUSY`/`SQLITE_LOCKED` — so a concurrent index run or write-tool commit can never be reverted by a reprojection that raced it. A document whose stored frontmatter JSON is unparseable is skipped (logged as a warning) rather than aborting the whole run, and the command prints the count of documents successfully reprojected.

A full reindex (`index --full`) also clears the `documents`/`document_fields` metadata index via `StateDb::clear`, alongside `indexed_files` — so a file removed from disk since the last full run cannot leave a phantom `list_documents` entry behind.

### Link graph

A third table, `document_links`, holds the edges behind the web UI's graph view and the move-time link rewriter — one row per (`source_path`, `target_path`, `kind`):

| Column | Type | Notes |
|---|---|---|
| `source_path` | TEXT | Document the link is written in |
| `target_path` | TEXT | Document the link points at |
| `kind` | TEXT | `markdown` (an inline link extracted from the source's body at ingest time) or `semantic` (a precomputed kNN neighbor, carrying a similarity `score`; opt-in via `ui.semantic_edges.enabled`, off by default) |
| `score` | REAL, nullable | Cosine similarity for `semantic` edges; unset for `markdown` ones |

Unlike `document_fields`, this table carries no foreign key to `documents` and no `ON DELETE CASCADE`: a link's target need not exist as an indexed document at all — it may point at a file that hasn't been indexed yet, or one that was since renamed or deleted out from under it. `GET /api/graph` (`web.rs`) drops dangling edges at read time instead of relying on the schema to keep them consistent. Rows for one `(source_path, kind)` are replaced wholesale (delete-then-insert in one transaction) whenever that file's outgoing links are recomputed, so a reader never observes a partially-replaced edge set, and a file's rows are removed outright on `delete_document`.

The reverse lookup — "what points at this target" — is what `write_document`'s directory/document move uses to rewrite links: when a path moves, `StateDb::links_targeting` (scoped to `kind = "markdown"`, so precomputed semantic neighbors are never treated as literal link text to rewrite) names every source document whose body needs its link text updated to the new path, and the move updates each of them in the same commit.

### Qdrant payload schema

Each indexed chunk is stored as a Qdrant point with this payload:

| Field | Type | Indexed | Notes |
|---|---|---|---|
| `file_path` | keyword | yes | Relative path from `data_path` |
| `chunk_index` | integer | no | 0-based chunk position within the file (internal; not returned by `search`) |
| `text` | text | no | Chunk content as embedded: heading breadcrumb + description + body |
| `text_body_offset` | integer | no | Byte offset in `text` where the chunk's own body starts; `qdrant::chunk_body_text` starts search snippets there (`CHUNKER_VERSION` 2) |
| `line_start` | integer | no | First line of the chunk in the source file, counted from the top of the raw file (frontmatter included), matching `get_document` |
| `line_end` | integer | no | Last line of the chunk in the source file, same numbering |
| `heading_path` | keyword array | no | Only with `chunking.heading_metadata` (#286). Heading texts from the top-level heading down to the chunk's attributed heading; `[]` before the first heading |
| `heading_level` | integer | no | Only with `chunking.heading_metadata`. Level (1-6) of the attributed heading, `0` when `heading_path` is empty |
| `heading_prefixes` | keyword array | yes (when the flag is on) | Only with `chunking.heading_metadata`. One key per run of consecutive headings in `heading_path` (each segment normalized by `heading::normalize_heading_text`, joined with U+001F) — what `search`'s `heading_prefix` filter matches exactly |
| `section_key` | keyword | yes (when the flag is on) | Only with `chunking.heading_metadata`. `file#start-end:path` identity of the attributed section; `search`'s `section` granularity groups by it |
| `section_line_start` / `section_line_end` | integer | no | Only with `chunking.heading_metadata`. Line range of the attributed section: the heading's whole subtree, or the preamble |
| `mtime` | integer | yes | File modification time as Unix timestamp |
| frontmatter fields | keyword / array | yes | Fields listed in `frontmatter.indexed_fields` (e.g. `type`, `domain`, `tags`, `title`) |

Keyword and array fields listed in `frontmatter.indexed_fields` get Qdrant keyword indexes, enabling exact-match and match-any filtering in the `search` tool.

`domain` is not read from frontmatter — `ingest.rs`'s `with_derived_domain` (built on `derive_domain`) computes it from the document's top-level folder and inserts it into the frontmatter map before it's written to the Qdrant payload, so from the payload's perspective it's an ordinary keyword field. The same derived map is what's persisted to `documents`/`document_fields` in the state DB (see [State Model](#state-model)), so `domain` behaves identically as a `search` query-mode payload filter and as a `search` enumeration-mode (`list_documents`) filter — only its origin (derived vs. author-written) changed. A `domain:` key in a file's own frontmatter is discarded; `with_derived_domain` logs a warning when it disagreed with the folder-derived value. Documents at the knowledge-base root (no top-level folder) get no `domain` at all. The write path refuses an authored one outright: `write::validate_document` adds a `rule: "derived"` field error from `ingest::authored_derived_field`, keyed on `ingest::DERIVED_FIELDS` — the same list `search`'s filter vocabulary treats as built in — while indexing and the `validate` CLI keep accepting existing documents that carry one. The check judges the frontmatter as written, before schema defaults, and a `required` or `default` a schema (or the legacy `frontmatter` config) declares for a derived field is ignored — `validate::apply_defaults` and the required check skip `ingest::DERIVED_FIELDS` — so such a rule can neither fail every write nor demand an authored value. `get_schema` never lists a derived field either, even one a scope declares (a deployment may name `domain` in `frontmatter.indexed_fields` to filter on it).

## Retrieval

`retrieval.rs` provides three shared functions consumed by `mcp.rs`: `search` and `get_document` as before, plus `list_documents` — the exhaustive, no-embedding-call enumeration that used to be its own MCP tool. The MCP `search` tool now covers both: its handler dispatches to `list_documents` when the caller omits `query`, and to `search` (ranked, top-k) otherwise; `granularity` (`chunk` or `document`, defaulting per `query`'s presence, or `section` — see `search_sections` — when `chunking.heading_metadata` is on) then picks the result shape. `get_document` remains its own separate MCP tool.

**`search`** — embeds the query, builds a Qdrant filter map from the `filters` map (any frontmatter field with a payload index; an unindexed field is rejected by name, not silently unfiltered), runs the retrieval (see below), applies an optional `min_score` floor, and returns raw results. Timing (embed + search ms) is logged at `debug`.

When `search.hybrid` is enabled (the default), retrieval is **hybrid**: the query is embedded into a dense vector *and* tokenized into a BM25-style sparse vector (`sparse.rs`, pure-Rust; Qdrant applies IDF weighting server-side via the `sparse` named vector's `Modifier::Idf`). Both arms run as Query-API prefetches — each fetching `search.rrf_candidates` candidates with the same payload filters — and are fused server-side with Reciprocal Rank Fusion (RRF). This sharply improves recall for exact tokens (hostnames, error codes, CLI flags, config keys) that pure dense search ranks poorly. With `search.hybrid: false`, only the dense (`dense`) named vector is queried.

**Vector schema & migration** — collections are created with two named vectors: `dense` (cosine, `embedding.vector_size`) and `sparse` (`Modifier::Idf`). Both are always written at index time, so toggling `search.hybrid` never requires a reindex. Upgrading a knowledge base indexed by a pre-hybrid version (single unnamed vector) *does* require a one-time full reindex (`index --full`) to migrate to the named-vector schema.

**Phrase search** (`search.phrase`, default `true`) — a double-quoted span in `query` adds a third prefetch arm alongside dense/sparse: an exact-phrase condition on the `text` payload field (a Qdrant `phrase_matching` text index), fused into the same server-side RRF. `ensure_collection` degrades gracefully against a Qdrant server too old to support `phrase_matching` — it logs and disables the phrase arm for the process (`status::IndexStatus::phrase_matching_available`) rather than failing startup. Grouped (document-granularity) `search` uses the same dense/sparse/phrase fusion arms as chunk-granularity search.

**`get_document`** — resolves a user-supplied path to a file on disk:

1. **Literal resolution** — joins the path against `data_path`, canonicalizes, and checks two security conditions: the resolved path must be under `data_path` (path-traversal guard) and must match `indexing.include` patterns (file-type guard). Before this, `retrieval::kb_root_relative` strips a leading `/` so `/food/chili.md` and `food/chili.md` resolve identically — a caller has no way to know where the KB actually lives inside the container, so a leading `/` is read as "the KB root," not a filesystem path. (For backwards compatibility, an absolute path that exists literally on disk is still tried first; only when that lookup misses does the KB-root-relative reading apply.)
2. **Fuzzy basename fallback** — if the literal path is not found, reads every indexed path from the state DB's `documents` table (no Qdrant call, no cap) and looks for an exact basename match. A unique match auto-resolves; multiple matches return an `Ambiguous` error with the candidates listed. Zero matches return a `NotFound` error with up to 3 Levenshtein-ranked suggestions.
3. **Line-range slicing** — the optional `start_line`/`end_line` parameters (1-based, inclusive) trim the resolved content to a slice. All three read surfaces share this one implementation: the MCP tool's parameters, `GET /api/doc/{*path}`'s query string, and the CLI's `get --start-line/--end-line`. Validation splits across two points on purpose: `retrieval::LineRange::new` checks the bounds against each other (no line 0, no inverted range) *before* the path is resolved, since a malformed range is wrong regardless of which document it names, while `retrieval::slice_lines` checks them against the content, which it must read first. An `end_line` past the last line is clamped rather than rejected, and a `start_line` past it is an error — except `start_line: 1`, which is always valid, because reading from the beginning of a document is meaningful even when there is nothing there: an empty document served that way is an empty slice, not a `StartPastEnd` (#298). The response reports the slice as served (`start_line`, `end_line`, `total_lines`, `partial`), so a caller can page without guessing. Slicing is done on `split_inclusive('\n')`, so the result is a byte-exact substring — CRLF and an unterminated last line survive, which is what lets a slice be fed straight back as a `write_document` `old_string`. `version` is deliberately **not** sliced: it is the blob id of the whole file, because its job is the `expected_version` of `write_document` and `delete_document` (and the web editor's, over the same route), which guards the document on disk.
4. **Read size cap** — `search.section_max_bytes` (default 16000) bounds every `get_document` read the caller did not bound itself, resolved in `retrieval::resolve_document_view` and serialized by `retrieval::document_view_json`, so the MCP tool and `GET /api/doc/{*path}` degrade identically. A selected section over the cap comes back as the outline of its sub-headings (`outline_only: true`) plus an `intro` range for its own text above the first sub-heading (#286); a whole-document read over the cap comes back as the *document's* outline the same way, with `intro` covering line 1 (frontmatter included) through the line before the first heading (#290). When there is no structure to fall back on — a document or a section with no headings beneath it — the text is cut on a whole-line boundary by `truncate_to_line_budget` and flagged (`truncated: true`, plus the `end_line` actually served) rather than returned unbounded; at least one line is always kept, even one larger than the budget, the same posture `cap_outline` takes with entries. An explicit `start_line`/`end_line` range is exempt: the caller named the bounds, so the size is its decision. Every flag is emitted only when it applies (`truncated`, `outline_only`, `oversized`, `partial`, `intro`), so a read that fits carries none of them.

`mcp.rs` wraps these functions with MCP-specific input validation (path and query length limits, filter, patch-operation and batch-size caps) and returns each result as one JSON object (see the `mcp.rs` row in the [Module Layout](#module-layout) table).

This same `kb_root_relative` reading of a leading `/` is shared by every path-taking tool, not just `get_document`. The write tools (`write_document`/`delete_document`, via `resolve_safe_write_path` in `write.rs`) and the schema tools (`get_schema`/`update_schema`, via `normalize_scope_path`) all strip a leading `/` the same way before applying their own traversal checks — `/../x` is rejected exactly like `../x` in every one of them. `normalize_scope_path` also rebuilds a schema tool's path from its normal segments, so `food/./recipes` reads as `food/recipes` and a lone `.` is the root. `write_document` and `delete_document` resolve an existing document's `path` with `retrieval::resolve_within_data`, the literal step `get_document` tries first; there is no basename fallback (only `get_document` has one), so a bare basename is "does not exist". `write_document` first checks, with `resolve_safe_write_path` and `is_dir`, whether `path` names an existing directory on disk; if so, the whole call is routed to a directory move instead (`path` and `new_path` are the source and destination prefixes — `resolve_within_data` cannot make this test, since a directory never satisfies its include-pattern check). `get_schema` and `update_schema` resolve a partial *directory* reference against the scopes `SchemaCache` knows about (`SchemaCache::match_scope_dirs`), matching on trailing path segments — a unique match resolves, several matches are refused with the candidates listed (never a silent guess), and for `update_schema` specifically, zero matches falls back to the literal path rather than erroring, since declaring a schema for a directory that has none yet is the normal way to introduce one.

**`list_documents`** — lives in `retrieval.rs` alongside `search` and `get_document`, but is architecturally a separate path from `search`'s ranked retrieval: it queries the `documents`/`document_fields` tables in `state.db` directly (via `state.rs`/`document_fields.rs`), with no embedding call and no relevance ranking. It lists documents by frontmatter — `filters` (equality, any-of, all-of, or numeric range per dot-path field), `path_prefix`, `order_by` (`path`/`title`/`mtime`/`indexed_at`), `descending`, `limit`/`offset` for paging, and an optional `fields` projection. It's what the `search` MCP tool calls when the caller omits `query`, since ranked search returns *chunks* (or, at document granularity, a top-k page) and cannot reliably enumerate a complete set of documents. The MCP response carries `total`, `returned`, `documents[]` and — when another page exists — `has_more: true`, so truncation is never silent.

`parse_field_filter` (`mcp.rs`) rejects combining set matching (`any_of`/`all_of`) with a numeric range (`gte`/`lte`/`gt`/`lt`) on the same field, and rejects `any_of` together with `all_of` on the same field — each is a validation error, not a silent pick-one.

## Schema Cascade

`schema.rs` lets a `.schema.yaml` file govern its directory and everything beneath it, cascading like `CLAUDE.md`. It replaces the single global `frontmatter` config block as the source of field rules — though that block is still honored, as the implicit root schema, when no `.schema.yaml` files exist.

**File names.** `SCHEMA_FILE_NAME` is `.schema.yaml`; the legacy `.kb-schema.yaml` (`LEGACY_SCHEMA_FILE_NAME`) is read everywhere the canonical name is (`SCHEMA_FILE_NAMES`, and `is_schema_file_path` — the one predicate every guard uses: the write tools' `check_not_schema_file`, `SchemaWalkFilter::governs`, `reindex::mark_schema_changes`/`unit_touches_schema`, `move_directory`'s schema carry). A directory holding both is an `InvalidSchemaFile` (`BOTH_NAMES_REASON`) for `SchemaCache::build`, `SchemaCache::raw_file_at` and a directory move alike — never resolved by picking one. `update_schema` always writes `.schema.yaml`; when `raw_file_at` reports the directory is on the legacy name, `write_raw_file` removes the legacy file under the same `GitLock` acquisition and names it in the same `commit_and_sync` call (a tracked path's removal is staged by `git add`), restoring it on a pre-commit rollback, so a directory is never left with both names or with neither.

**Model-facing names.** `ResolvedSchema::origin` holds the scope directory as `schema::scope_label` writes it (`food/recipes/`, the root as `/`), not a file path, so `get_schema`'s `declared_in`, the web UI's `/api/schema` `declared_in`, and validation errors' `[declared in …]` never name a file. Every other model-facing string — tool descriptions, `update_schema`'s result `path`, its errors, the move-refusal data keys `invalid_schema_dir`/`moved_schema_dirs`, and `SchemaBuildError::model_facing` for a refused rebuild — names the directory too. Operator-facing surfaces (logs, `/status`'s `schema_error`, the CLI, `SchemaBuildError`'s `Display`) keep real file paths; the `kb_schema_invalid*` metric names are unchanged.

**Resolution.** `SchemaCache::build` walks the indexed tree of `data_path` once, parsing every `.schema.yaml` it finds and merging each against its nearest ancestor. This produces one `ResolvedSchema` per directory that declares a schema of its own. The walk skips (and does not descend into) a hidden directory or one `indexing.exclude` rules out entirely — `schema::SchemaWalkFilter`, which asks `ingest::PathFilter::excludes_dir` whether `exclude` matches two synthetic documents in the directory, one directly in it and one a level below, so `templates/**` drops `templates/` while a pattern that leaves some document under a directory indexed keeps it. `include` and `exclude_files` never remove a directory: an `include` of `docs/**/*.md` matches nothing in the root, yet the root's schema governs `docs/`. A filter that cannot be built fails open (every non-hidden directory read, logged at `error`), as `partition_indexable` does (#272). After that single walk, resolving a document's effective rules is an in-memory longest-prefix lookup (`SchemaCache::resolve_for`) — no filesystem access, and not repeated per file during indexing.

**Merge semantics.** The set of fields unions across cascade levels; a field redefined at a deeper level replaces its inherited definition wholesale. `extend: true` is the sole exception, unioning only `values` with the inherited set while everything else on the child definition still wins. `ResolvedSchema` tracks, per field, which scope (`origin`, a `scope_label` directory) contributed its current definition — this backs `get_schema`'s provenance output.

**Dedup override.** A schema file may also carry a `dedup:` block (`RawDedup`: optional `enabled`/`threshold`, threshold validated to 0.0–1.0 in `SchemaFile::validate_self`). It merges per key in `merged_with` (child over parent, nearest wins) into `ResolvedSchema::dedup_enabled`/`dedup_threshold`, where `None` means "use `write.*`" — the cache is built without `WriteConfig`, so global values are not baked in. `write::effective_dedup` resolves override-or-global at both create gates (`write_document` and the batch path), and `get_schema` reports the override when set (#272). It is deliberately excluded from `fingerprint()`: it affects neither validation nor indexing, so editing it must not revalidate the subtree.

**Schema fingerprint and incremental indexing.** `ResolvedSchema::fingerprint()` hashes a schema in a way that's stable regardless of map iteration order. `ingest.rs` compares both the file's content hash *and* this fingerprint against the stored `schema_hash` (see [State Model](#state-model)) before skipping a file as unchanged — matching content alone is not enough, since editing a schema doesn't touch any document's bytes. Rows written before this feature carry an empty `schema_hash`, which never matches a real fingerprint, so the first index run after upgrading revalidates every file once. Editing a root-level `.schema.yaml` changes the root fingerprint and so revalidates the whole KB on the next run. That same forced revalidation is also when `with_derived_domain` (re)writes every document's `domain` from its folder — see the note on the Qdrant payload schema above — so an upgrade can change the effective `domain` of documents whose old frontmatter value disagreed with their folder.

**Fail-fast validation.** A `.schema.yaml` in the schema tree (see Resolution above) that is present must be valid; a root one is optional. `SchemaCache::build` returns `Result<SchemaCache, SchemaBuildError>`: any file it cannot read, that is over `MAX_SCHEMA_FILE_BYTES` (256 KB — checked on `fs::metadata` size alone, before `read_to_string`/parsing runs), that fails to parse (`SchemaFile` and `RawDedup` are `deny_unknown_fields`), or that fails `SchemaFile::validate_self` makes the whole build an `Err`, and the error collects every such file (`InvalidSchemaFile`: path + reason), not just the first. There is no partial cache and no fallback to a parent's rules. What a caller does with the error is the policy:

- **Startup** (`server::build_startup_schema_cache`) and the CLI (`index` via `ingest::build_run_schemas`, `validate` in `main.rs`) fail with it — the server refuses to start, and `validate` exits non-zero regardless of `--strict`.
- **Runtime rebuilds** — the reindex worker's `schema_rebuild_runner` (before every full reconcile) and `update_schema`'s synchronous rebuild — go through `schema::apply_rebuild`: `Ok` is swapped into the `SharedSchemaCache` and clears `INDEX_STATUS`'s schema error; `Err` is refused, so the shared cache keeps the last good tree, and the refusal is logged at `error` and recorded (`IndexStatus::record_schema_error`) for `/status` (`schema_error`) and `/metrics` (`kb_schema_invalid`, `kb_schema_invalid_files`). Because the worker rebuilds before every full reconcile, the periodic sweep re-logs it while it persists. A refused worker rebuild skips its unit outright.
- **Indexing runs** build their own throwaway cache from disk (`ingest::build_run_schemas`, in both `scan_for_dirty` and `index_paths_generic`) and abort on `Err` before touching anything — including before a `--full` run drops the collection. The `SchemaBuildError` stays in the error chain; the worker's `run_with_retry` finds it there, drops the unit without retrying, logs every invalid file at `error`, and records it through `IndexStatus::record_schema_error` immediately (keeping an existing streak's `since`), so `/status` and `/metrics` show it without waiting for the next sweep's rebuild (#272). The dropped paths stay dirty on disk. The fix (a push touching the schema file) queues its own full reconcile, and a worker rebuild that succeeds while a schema error is recorded queues one too when its own unit is a `Paths` unit, so writes made while the schema was invalid are indexed promptly rather than at the next periodic sweep.
- **Directory moves** (`write::move_directory`) re-read every schema file in the source subtree that is part of the schema tree at its source or at its destination (one coming out of a hidden or wholly excluded directory was never read by any cache), and refuse the move (`DirectoryMoveError::InvalidSchemaInSource`) when one fails `schema::parse_schema_text` — the shared cache may still hold its last good rules, which would otherwise validate the move and carry the broken file along (#272).

A changed schema file reaches the worker as a full reconcile: producers that learn about changed paths from git — `webhook.rs`'s push diff, `write.rs`'s `mark_dirty` over a write's own paths and its `rebased_paths`, and a `move_directory` carrying schema files — call `ReindexQueue::mark_schema_changes` before the include filter drops the (never indexable) schema path. A schema file outside the schema tree (hidden or wholly excluded directory, per `SchemaWalkFilter::governs`) queues nothing, and the worker's `unit_touches_schema` ignores one in a `Paths` unit for the same reason. The document tools refuse a `.schema.yaml` path outright (`WriteError::SchemaFile`, checked ahead of the include filter), so schema files only change through `update_schema`, git, or a directory move.

**Field shape validation.** `validate_raw` (`schema.rs`) rejects a field definition that declares both a scalar `type` and nested `fields:` — a field is either a value or a container, never both (`type: object` is exempt, since `object` inherently means "has nested fields"). This applies uniformly whether the schema was hand-edited or produced by `update_schema`'s `set_field`. `SchemaFile::validate_self` also rejects a path declared twice in one file (nested `fields:` plus a flat dot-path key), since `flattened()` maps both spellings to one key. `SchemaFile::fields` and `RawFieldDef::fields` are `BTreeMap`s so flattening is deterministic. `SchemaFile::apply` resolves an edit's dot-path through nested `fields:` and flat dot-path keys alike (`find_field_mut`/`field_entry_mut`/`remove_field_path`), creating missing parents as `type: object` containers (#268).

**Typed Qdrant indexes.** `SchemaCache::all_indexed_fields()` collects every dot-path declared `indexed: true` anywhere in the tree, along with the payload index kind its declared type needs (`ResolvedSchema::index_kind`): integer/number/boolean fields get Integer/Float/Bool payload indexes instead of a blanket Keyword index, enabling range and comparison filters. Payload indexes are created once for the whole collection, so a field declared only in a deep scope is still registered up front. If two scopes declare the same path with different types, the first one encountered wins and the server logs a warning — one collection can't hold two index kinds for one path; the operator must delete the stale payload index in Qdrant and reindex to pick up a type change.

Separately, the Qdrant call that actually creates a payload index (any schema-declared field, plus the built-in `mtime` index) can itself fail — typically because the field is already indexed in Qdrant under a different kind. `QdrantStore::ensure_collection` logs that at `error` level and continues rather than aborting the run or server startup: a filter on the affected field still returns correct results, just without the index to speed it up. As above, the fix is to delete the stale payload index in Qdrant and reindex.

**`get_schema` / `update_schema`.** Both tools read the server's shared `SharedSchemaCache` (`schema::load_shared`) rather than walking the tree per call; it is kept current by the reindex worker's rebuild before every full reconcile and by `update_schema`'s own synchronous rebuild after a write. `update_schema` takes `GIT_LOCK` and syncs with the remote (`write::lock_and_sync`) before reading the target directory's raw file from disk (`SchemaCache::raw_file_at`, which reads a candidate the way the build walk finds one: a symlink or other non-regular entry is absent, never followed or read, and a file over `MAX_SCHEMA_FILE_BYTES` is refused before it is read), and holds that guard through the edit, validation and install, so its edit applies to the file as it is now and two concurrent operations on one schema both land. When the file changed between a pre-lock preview and the locked read in a way that makes the operation no longer apply, it is refused with a re-read instruction. The guard is dropped before the cache rebuild, whose generation stamp (`SchemaCache::generation`) keeps an out-of-order finish from installing older rules. Both resolve a possibly-partial `path` against known scopes first (see [Retrieval](#retrieval) above for the shared partial-match/ambiguity contract); `get_schema` then resolves the merged rules for that path (a document resolves via its parent directory) and returns them with per-field provenance, a derived field never among them (`values_in_use` adds what documents use — see the `mcp.rs` row above). `update_schema` applies a constrained `SchemaEdit` (`add_values` / `remove_values` / `set_field` / `remove_field`) to the target directory's raw schema file via `SchemaFile::apply_inheriting` — passing the definition the scope's ancestors alone resolve the field to, so `add_values` in a child scope with no local `values` writes `[$values, ...new]` (extending the inherited set, `$values` kept first, the inherited `type` kept — `enum` when no ancestor gives a type or values) rather than narrowing it — renders it back to YAML, and — critically — re-parses and size-checks (`MAX_SCHEMA_FILE_BYTES`) that YAML before writing, so a change the next rebuild would refuse is refused instead of committed. Before writing, it calls `SchemaCache::resolve_with_candidate` per document to see what that document's own effective schema *would* be under the proposed edit — rebuilding its full ancestor chain with the candidate substituted in, since merge is per-field and a deeper scope that redeclares one field still inherits the rest. Each document already indexed under that scope is then validated against its own resolved candidate (via the `documents`/`document_fields` metadata index, not a markdown re-read), with schema defaults applied first so a required field that carries a default is not reported as breaking documents that omit it. Documents that would fail block the write unless `force` is set; `dry_run` reports the same check without writing. A successful write goes through the same commit-and-push path as the document write tools and triggers an incremental reindex.

## Webhook Flow

```text
POST /hooks/reindex
        │
        ▼
 verify HMAC signature
 (GitHub: x-hub-signature-256 / Gitea: x-gitea-signature / GitLab: x-gitlab-token)
        │
  fail? ──► 401 + warn log
        │ ok
        ▼
 check branch filter (source.branch)
        │
  mismatch? ──► 200 (ignored, no reindex)
        │ match
        ▼
 git_url configured?
        │
   no ──┴──────────────────────────────► mark_full() on the ReindexQueue
        │ yes                                          │
        ▼                                               │
 capture HEAD, then git fetch + git merge --ff-only     │
 (git_token injected transiently, never written to disk) │
 (120-second timeout on each subprocess)                 │
        │                                                │
        ▼                                                │
 diff old HEAD..new HEAD, mark_paths(changed)             │
 (+ mark_full() if a .schema.yaml changed)            │
        │                                                │
        └──────────────────────────────┬─────────────────┘
                                        ▼
                    200 "Changes queued for indexing" — handler returns
                                        │
                     (asynchronously, off the request path)
                                        ▼
                     reindex::run_worker drains the queue
                     and calls ingest::index_paths / scan_and_index
```

There is no lock, no single-flight, and no coalesce-or-skip in the handler itself. `handle_webhook` (`src/webhook.rs`) never indexes anything — it fetches, merges, diffs the pulled range with `git::git_diff_name_status`, filters the diffed paths through `ingest::PathFilter` (the same `indexing.include`/`exclude`/`exclude_files` predicate a full reconcile applies via `discover_files`, so a push touching a non-indexable path like the default-excluded `README.md` is never marked dirty — #278; a glob-build error fails open, marking everything unfiltered rather than 500ing an otherwise-successful pull), and marks exactly the surviving paths dirty via `ReindexQueue::mark_paths`. A changed `.schema.yaml` is never indexable, so it is checked first, against the unfiltered diff: `ReindexQueue::mark_schema_changes` queues a full reconcile, before which the worker rebuilds the shared schema cache (refusing it, and keeping the previous one, if the pushed file is invalid — see [Schema Cascade](#schema-cascade)). When `source.git_url` is unset and there is nothing to fetch or diff, it falls back to `ReindexQueue::mark_full`. `mark_paths`/`mark_full` never block and never fail, so the handler returns `200 "Changes queued for indexing"` as soon as the git operations finish — it does not wait for the paths it just marked to actually be reindexed.

The queue itself is what used to be a single-flight `REINDEX_LOCK` (`Arc<tokio::Mutex<()>>`, `try_lock_owned()`), and that design had a real bug: a webhook arriving while a previous one's inline reindex was still running lost its race for the lock and was **dropped** — skipped, not queued or replayed. `reindex::ReindexQueue` replaces that with coalesce-don't-drop semantics instead: every delivery's `mark_paths` call lands in the same dirty-path set (a `HashSet`, so marking an already-pending path is a no-op, not data loss), and the single background worker (`reindex::run_worker`) drains that set, indexes it, and immediately drains again — if new work landed while it was running, it loops and runs again before going back to sleep, rather than a losing webhook being dropped outright. A path is lost only if it is never marked at all, never because of *when* it was marked. The worker retries a transient failure (embeddings/Qdrant unreachable, git I/O) with exponential backoff up to a bounded attempt count before deferring to the next periodic reconcile sweep, and drops a permanent one (an invalid `.schema.yaml`, which no retry can fix) immediately; the fix queues its own full reconcile. See `reindex.rs`'s module doc for the full design rationale, and [Webhook](../README.md#webhook) in the README for the operator-facing summary.

## Web UI

`web.rs` serves a docs-first web UI on the *same port* as MCP (8001) — it is a second router merged into the one Axum app `server::run_server` builds, not a separate service. Its shell (`assets/ui/`) and every script it loads — Cytoscape.js plus its `layout-base`/`cose-base`/`fcose` layout plugins, `marked`/`dompurify` for rendering markdown safely, `edit.js` for the editor — are embedded into the binary via `include_str!` and served from `/assets/*`, so there are no filesystem reads or external fetches at request time.

**Routes:** `/` (the shell), `/assets/*`, `/api/graph` (nodes from `StateDb::all_document_summaries`, edges from `StateDb::all_links` — see [Link graph](#link-graph) above — filtered to the current node set), `/api/search` (a thin wrapper over `retrieval::search`), `/api/schema/{*path}`, and `/api/doc/{*path}`, which carries all three HTTP methods on one route: `GET` (via `retrieval::get_document`, with the same `?start_line=&end_line=` slicing contract as the MCP tool and the CLI's `get`, plus the MCP tool's `line`/`heading_path`/`levels_up`/`outline` section and outline modes, parsed, resolved and serialized by the same `retrieval.rs` functions so the two transports return identical JSON — including the read size cap described under [Retrieval](#retrieval), which is why the UI's document view and editor request `?start_line=1`: a caller-named range is exempt, and the editor posts the whole body back when it saves), `POST` (create, full-replace edit, or move — `write::write_document`), and `DELETE` (`write::delete_document`).

The UI itself is docs-first: a sidebar document tree and a semantic-search results panel are the primary views, hash-routed (home/browse/doc); the Cytoscape graph — full-KB or a per-document neighborhood, `fcose` layout, hover/zoom-gated labels — is reachable but secondary. A full create/edit/move/delete editor rides on top of the same `/api/doc/{*path}` route the read view uses.

**`POST`/`DELETE /api/doc/{*path}` are thin adapters over the exact same `write::write_document`/`write::delete_document` pipeline the MCP `write_document`/`delete_document` tools use** — the same frontmatter validation, the same dedup gate on create, the same pre-commit rollback, the same `git::commit_and_sync` (commit and push to the knowledge base's real git remote), and the same marking of the written path dirty on the `ReindexQueue`. A web UI edit is indistinguishable, downstream, from an MCP write. Each call runs on its own task (`detached` in `web.rs`), so a client that disconnects mid-request cannot drop a write between its filesystem change and its commit. The editor sends the version it loaded as `expected_version` on save and move, and the document view the version it rendered on delete: a stale save is three-way merged when it cleanly can be (the editor says so), and a stale save that conflicts, a stale move and a stale delete are refused with a 409 whose body carries `edited_elsewhere: true`, so a document changed since it was opened is never overwritten or deleted unseen.

**Deliberately unauthenticated, independent of `MCP_BEARER_TOKEN`.** Unlike `mcp_router`/`status_router`/`admin_router`, the web UI's router is merged into the app with no bearer-auth middleware layer at all (`server.rs`) — not "unauthenticated because `mcp.allow_unauthenticated` is set", but unauthenticated unconditionally, regardless of that setting. This applies equally to the read routes and to `POST`/`DELETE /api/doc/{*path}`: there is no separate, stricter gate on the write/delete routes. `web.rs`'s own module doc states the reasoning directly: the deployment this was built for sits behind an identity-aware reverse proxy (Authentik via Traefik), and the binary does not gate these routes itself — the same open-route posture `/health` already has. The startup log names the UI as unauthenticated, but does not currently spell out that this is independent of the bearer-token setting; see [README.md's Web UI section](../README.md#web-ui) for the operator-facing statement of what exposing this port directly, with no reverse-proxy auth in front, actually grants.

## Security Model

This service is designed for intranet/tailnet deployment — the threat model assumes network-level access control at the perimeter. That assumption carries the most weight for the web UI (see above), which has no authentication story of its own at all.

| Control | Mechanism |
|---|---|
| MCP authentication | OAuth 2.1 JWT access tokens (`mcp.oauth`, recommended — provider-agnostic, validated by the [`oauth-resource-server`](https://github.com/St0nefish/oauth-resource-server) crate, see [oauth.md](oauth.md)) and/or a static bearer token (`mcp.bearer_token_env`); rejections logged at WARN with the reason, never the token |
| Webhook authentication | HMAC-SHA256 (`webhook.secret_env`); failures logged at WARN with provider |
| Web UI authentication | **None — by design, and independent of `mcp.bearer_token_env`/`mcp.allow_unauthenticated`.** Every route in `web.rs`, including `POST`/`DELETE /api/doc/{*path}`, is mounted with no bearer-auth layer; see [Web UI](#web-ui) above |
| Path traversal | Every path-taking tool (`get_document`, `write_document`, `delete_document`, `get_schema`, `update_schema`) and the equivalent web UI routes resolve and canonicalize paths, then check `starts_with(data_path)`; a leading `/` is treated as the KB root, not a filesystem escape hatch, and `..` components are rejected either way (see [Retrieval](#retrieval)) |
| File-type restriction | `get_document`, the write tools, and the web UI's write route check the resolved path against `indexing.include` glob patterns |
| Reflected-value sanitization | Strings the knowledge base controls (top-level folder names in the MCP server instructions; in `get_schema`, field names, permitted, default and in-use values and `declared_in`; the options a refused `search` filter lists; the declaring scope in a validation error) come from folder names, synced schema files and **indexed document frontmatter**. Each passes through `server::sanitize_facet_value` (control characters replaced with spaces, length-capped at 64 characters) before it reaches the model, to mitigate prompt-injection against connected AI clients |
| Rate limiting | Configurable token-bucket rate limiter (`rate_limit.requests_per_second`, `rate_limit.burst_size`), keyed per client by `rate_limit.client_ip_source` (`peer` by default; the leftmost `X-Forwarded-For` entry is never trusted, since callers write it), applied to the whole app — MCP, webhook, and the unauthenticated web UI routes alike, since the `GovernorLayer` wraps the fully-merged router |

## Configuration and Environment Variables

See [`deploy/config.example.yaml`](../deploy/config.example.yaml) for all options with defaults and [`README.md`](../README.md#configuration) for the env-var table.
