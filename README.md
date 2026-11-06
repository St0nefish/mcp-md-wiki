# mcp-md-wiki

A Docker-first RAG server that indexes markdown knowledge bases with YAML frontmatter into Qdrant and exposes them over MCP (Streamable HTTP) — semantic search, document retrieval, and agent-driven writes (create/edit/delete) that commit straight back to the knowledge base's git repo.

Built as a single Rust binary for type safety, small Docker images, and simple deployment.

## Documentation

- [`deploy/USAGE.md`](deploy/USAGE.md) — Setup guide, configuration, frontmatter, chunking
- [`deploy/TROUBLESHOOTING.md`](deploy/TROUBLESHOOTING.md) — Common issues and fixes
- [`docs/oauth.md`](docs/oauth.md) — OAuth setup (recommended auth): the `mcp.oauth` keys and defaults, and this server's setup; provider recipes marked by what was actually tested live in the [`oauth-resource-server`](https://github.com/St0nefish/oauth-resource-server) crate's `docs/providers.md`
- [`deploy/config.example.yaml`](deploy/config.example.yaml) — Full annotated config reference
- [`deploy/ci-examples/`](deploy/ci-examples/) — Sample CI workflows for webhook-triggered reindex

## Quick Start

```bash
# Clone and configure
git clone https://github.com/St0nefish/mcp-md-wiki.git
cd mcp-md-wiki
cp deploy/.env.example .env
# Edit .env: set MODEL_PATH/MODEL_FILE, and MCP_BEARER_TOKEN unless you use
# OAuth only (recommended: configure mcp.oauth — see docs/oauth.md)
# (GIT_PULL_TOKEN is optional — only needed to clone/fetch a private knowledge-base repo)

# Download the embedding model (see "Embedding Models" below)

# Set source.git_url in config.yaml to point at your knowledge base repo
cp deploy/config.example.yaml config.yaml
# Edit config.yaml — at minimum, set source.git_url and uncomment the mount:
#   - ./config.yaml:/app/config.yaml:ro

# Start the stack (CPU mode by default)
# With git_url set, the server auto-clones the repo and runs a full index on first start
# Port 8001 is published on 127.0.0.1 only: set MCP_BIND=0.0.0.0 in .env for
# clients on other hosts (the web UI's write API is unauthenticated by default —
# see "Web UI" below)
docker compose up -d

# Add MCP to Claude Code
claude mcp add --transport http kb-search \
  https://your-host:8001/mcp \
  --header "Authorization: Bearer $TOKEN"
```

### Authentication: OAuth (recommended)

Enable `mcp.oauth` in `config.yaml` and point it at your identity provider. Clients
then log in through that provider and present short-lived, per-user access tokens
instead of sharing one static secret. claude.ai, Claude Desktop and the mobile apps
can only connect this way. With OAuth on, register a client with your provider and
add the server to Claude Code with that client id:

```bash
claude mcp add --transport http --client-id <client-id> --callback-port <port> \
  kb-search https://your-host:8001/mcp
```

[docs/oauth.md](docs/oauth.md) covers the `mcp.oauth` keys and this server's setup;
the validator, JWKS handling and provider recipes themselves live in the
[`oauth-resource-server`](https://github.com/St0nefish/oauth-resource-server)
crate this server's OAuth support is built on (mcp-md-wiki#308). The
static bearer token keeps working alongside OAuth unless you turn it off
(`mcp.oauth.accept_static_bearer: false`). With OAuth on, the write tools
(`write_document`, `delete_document`, `update_schema`) additionally require the
`mcp:write` scope (a token with only `mcp:read` gets `403 insufficient_scope`);
static bearer tokens keep full access.

### Claude Desktop

Claude Desktop has no native remote-MCP transport for bearer-token servers, so it
connects through the [`mcp-remote`](https://www.npmjs.com/package/mcp-remote) stdio
bridge. Add this to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "kb-search": {
      "command": "npx",
      "args": [
        "-y", "mcp-remote", "https://your-host:8001/mcp",
        "--header", "Authorization:${KB_AUTH}"
      ],
      "env": { "KB_AUTH": "Bearer YOUR_TOKEN" }
    }
  }
}
```

Note the `Authorization:${KB_AUTH}` form — no space after the colon. Claude Desktop
mangles arguments containing spaces, so the token is passed via the `env` block and
substituted by `mcp-remote`.

The server runs its Streamable HTTP transport in stateless mode, which matters most
for this path: there is no session for a dropped connection to invalidate, so the
bridge recovers on its own after a laptop sleep or a server restart.

The recommended setup uses a **named Docker volume** for the knowledge base. The container clones the repo on first start and pulls updates via webhook — no host-side git operations needed. See [deploy/USAGE.md](deploy/USAGE.md#knowledge-base-storage) for details on this approach vs. bind-mounting.

See [deploy/config.example.yaml](deploy/config.example.yaml) for all available options and their defaults.

## Architecture

Three Docker services:

| Service | Purpose |
|---|---|
| `qdrant` | Vector database (gRPC + REST) |
| `embeddings` | Local embedding server (llama.cpp, OpenAI-compatible API) |
| `mcp-md-wiki` | Indexer, MCP server, and webhook handler (single Rust binary) |

## CLI Commands

```bash
mcp-md-wiki serve              # Start server (MCP + webhook endpoints + web UI)
mcp-md-wiki index              # Incremental index (only changed files)
mcp-md-wiki index --full       # Full re-index (clear state, re-embed everything)
mcp-md-wiki validate           # Validate all markdown files without indexing
mcp-md-wiki search "query"     # Search the knowledge base from the CLI (same pipeline as MCP search)
mcp-md-wiki search "query" --limit 10 --explain      # Score-breakdown per result
mcp-md-wiki search "query" --domain infra --type guide --tags docker,node:ares
mcp-md-wiki search "query" --modified-after 2026-01-01 --json
mcp-md-wiki get PATH           # Print one document (path resolves like the MCP tool)
mcp-md-wiki get PATH --start-line 40 --end-line 60   # ...or just those lines, 1-based inclusive
mcp-md-wiki status             # Aggregate counts + metadata breakdown
mcp-md-wiki status --json      # Same data as the server's /status endpoint
mcp-md-wiki status --files     # List every indexed file instead
mcp-md-wiki health             # Check if server is healthy
mcp-md-wiki reproject-fields   # Rebuild document_fields from stored frontmatter under the schema (no re-embed; also automatic after an upgrade)
mcp-md-wiki git-pull           # Fetch + fast-forward the clone from its remote, then index what changed (no server needed)
mcp-md-wiki lint               # Corpus-wide dead-link report from the last indexed state (read-only; exit 0 unless there is no state DB)
mcp-md-wiki lint --check dead-links --json --limit 500   # Same report as JSON (default limit 200, max 5000)
```

`git-pull` is the same fetch, fast-forward, diff and filter sequence the push webhook runs (`sync.rs`), for instances that have no webhook wired up; against a running server, `POST /admin/sync` does the same and queues the changed paths instead of indexing in-process.

## Configuration

Every setting has **exactly one source**. Settings needed at startup — connection wiring, model identity, and secrets — come from environment variables only. Runtime and tuning settings come from `config.yaml` (or the path passed via `--config`) only. Nothing is readable from both, so there is no precedence to reason about.

**Environment only** — startup bindings, not settable in `config.yaml`:

| Env Var | Purpose | Default |
|---|---|---|
| `EMBEDDING_BASE_URL` | embeddings endpoint | *(required)* |
| `EMBEDDING_MODEL` | embedding model | *(required)* |
| `EMBEDDING_VECTOR_SIZE` | vector dimension | `768` |
| `QDRANT_URL` | Qdrant gRPC endpoint | *(required)* |
| `QDRANT_COLLECTION` | collection name | `knowledge-base` |
| `RERANKING_BASE_URL` | reranker endpoint | *(required when reranking is on)* |
| `RERANKING_MODEL` | reranker model | *(required when reranking is on)* |
| `GIT_URL` | knowledge-base repo to clone/pull | *(unset — no git integration)* |
| `GIT_BRANCH` | branch to track | `master` |
| `DATA_PATH` | knowledge base + state DB location | `/data` |
| `MCP_PORT` | listen port | `8001` |

Missing required env vars are named together in a single startup error rather than surfacing one per restart. An env var that is set but no longer honored logs a warning instead of being silently ignored.

**Indirected secret env vars** — the config field names *which* env var holds the secret, never the value itself:

| Config Field | Default env var name |
|---|---|
| `webhook.secret_env` | `WEBHOOK_SECRET` |
| `source.git_token_env` | `GIT_PULL_TOKEN` |
| `mcp.bearer_token_env` | `MCP_BEARER_TOKEN` |
| `embedding.api_key_env` | `EMBEDDING_API_KEY` |
| `reranking.api_key_env` | `RERANKING_API_KEY` |

For example, `webhook.secret_env: "WEBHOOK_SECRET"` tells the server to read the HMAC secret from the env var named `WEBHOOK_SECRET`. Change it in config if you want a different env var name.

**Docker secrets (`_FILE`)** — each of the five secret variables above (`webhook.secret_env`, `source.git_token_env`, `mcp.bearer_token_env`, `embedding.api_key_env`, `reranking.api_key_env`) also accepts `<configured name>_FILE`, naming a file to read the value from (by default `WEBHOOK_SECRET_FILE`, `GIT_PULL_TOKEN_FILE`, `MCP_BEARER_TOKEN_FILE`, `EMBEDDING_API_KEY_FILE`, `RERANKING_API_KEY_FILE`) — e.g. `MCP_BEARER_TOKEN_FILE=/run/secrets/mcp_bearer_token` with a Docker Compose `secrets:` mount. The file's surrounding whitespace (its trailing newline) is trimmed; a value given in the plain variable is used exactly as set, as before. Setting both is a startup error rather than a silent preference (a blank plain variable, as Compose's `X=${X:-}` produces, yields to the file); an empty or unreadable `_FILE` target is an error too, and neither set behaves as before. The git token is re-read on each MCP/web write and on each index run, so a rotated secret file takes effect there without a restart; the webhook pull and the startup clone keep the value read at startup, and every other secret is read at startup (or at a `/admin/reload`, which fails if a `_FILE` is bad).

Everything else — `search`, `reranking`, `chunking`, `indexing`, `validation`, `write`, `rate_limit`, `frontmatter`, `mcp.instructions` *(deprecated)*, `mcp.extensions_path`, `mcp.disabled_tools`, `mcp.enabled_tools` — lives in `config.yaml` only, and can be changed without a restart via [Config reload](#config-reload). Startup logs and `GET /status` report where every setting came from (env, yaml, or default).

See [deploy/config.example.yaml](deploy/config.example.yaml) for all options:

- **source** — Git URL (auto-cloned on first start) or bind-mount path for your knowledge base
- **indexing** — Include/exclude glob patterns
- **frontmatter** *(deprecated)* — Required fields, indexed fields, defaults, and `allowed` closed-set enums (enforced by `validate` and the write tools); prefer a root `.schema.yaml` instead — see [`.schema.yaml` Directory Schemas](#schemayaml-directory-schemas)
- **chunking** — Markdown-aware splitting with configurable chunk size. `heading_metadata` (default `false`; changing it re-indexes affected documents automatically) stores structured heading data (`heading_path`, `heading_level`, and a few derived fields) on each chunk's Qdrant payload, powering `search`'s `heading_prefix` filter and `section` granularity. It also changes chunk boundaries: with it on, small consecutive sections are never merged into one chunk, so every chunk lies within a single heading's own section (or the text before the first heading) — an oversized section still splits into several chunks, and a heading with no text of its own before its first sub-heading becomes a small chunk of just its heading line. With it off, small consecutive sections are merged into one chunk up to `target_chunk_size`. Headings are detected as CommonMark does (via pulldown-cmark): ATX `#` and setext (`Title` over `===`/`---`) headings count; headings inside code blocks, HTML blocks, blockquotes or list items do not, nor do `#tag` lines. Heading text is its rendered plain text with invisible characters that never affect rendering (soft hyphens, zero-width spaces, word joiners, BOM) removed, capped at 200 characters; zero-width joiners, direction marks and variation selectors, which do affect rendering, are kept in the stored text (so emoji sequences and Persian or Indic joining display intact), don't count toward the cap, and are ignored when matching
- **embedding** — OpenAI-compatible endpoint (works with llama.cpp, vLLM, etc.). `query_prefix` / `document_prefix` (default empty; YAML-only; restart required) are prepended to search queries and to chunk text (and the write tools' near-duplicate / related-documents query) for asymmetric models such as nomic-embed (`search_query:` / `search_document: `) — use the prefixes the model was trained with. `document_prefix` is part of each file's index fingerprint, so setting or changing it re-embeds the corpus on the next reconcile; changing only `query_prefix` needs no reindex.
- **validation** — Strict/lenient mode, optional lint command
- **webhook** — HMAC verification for Gitea/GitHub/GitLab (disabled if `WEBHOOK_SECRET` is unset)
- **mcp** — Server port, authentication (OAuth resource server — recommended, see [docs/oauth.md](docs/oauth.md) — and/or a static bearer token), `extensions_path` (where per-KB tool/server description policy lives in the served knowledge base; `instructions` is a deprecated narrative override — see [MCP Tools](#mcp-tools) below), and `disabled_tools` (default `[]`) — a list of MCP tool names to hide from `tools/list`/`tools/get` and refuse from `tools/call` (same error as an unknown tool name). Every name must be one of the seven MCP tools, with no duplicates, and the list may not name all seven. `enabled_tools` (default: unset) is the allowlist alternative — when set, every tool NOT named is disabled, exactly as if it had been listed in `disabled_tools`; set one or the other, never both (config load fails if both are set), and an empty `enabled_tools: []` is rejected the same way an all-seven `disabled_tools` is, since it would disable every tool. An allowlist also disables any tool added to this binary in a future release until the allowlist itself names it — `disabled_tools` has no such trap, since a new tool starts enabled by default. The server instructions' one pointer to `get_schema` ("Field values and per-folder rules: get_schema with a path.") is dropped while `get_schema` is disabled (from the next `mcp.metadata_refresh_secs` refresh on, since the instructions are recomposed on that timer), rather than pointing at a tool that no longer answers; the top-level-areas line names no tool and stays whichever tools are enabled. The web UI's `/api/doc/*` write routes go through `write::write_document`/`write::delete_document` directly, not through MCP dispatch, so neither setting has any effect on them.
- **write** — Behaviour of the write tools: near-duplicate detection (`dedup_enabled`, `dedup_threshold`), the related-documents listing a create returns (`related_min_score`, default `0.50`; `related_limit`, default `5`, `0` turns it off, max `20`), the archive flow (`archive_prefix`, default `_archive`, which must not be excluded by `indexing.exclude`; `archive_links`, default `repoint`) and the git commit identity. All are applied on reload.
- **search** — Retrieval behaviour: hybrid sparse+dense search with RRF fusion (`hybrid`, default `true`) and per-arm candidate count (`rrf_candidates`). Set `hybrid: false` for legacy dense-only search. See the migration note below. Also `phrase` (default `true`) — exact-phrase matching for double-quoted spans in a `search` query, fused as a third RRF arm. `granularities` (default: all of `chunk`, `document`, `section`) restricts which `search` granularity values this instance exposes — the effective set additionally drops `section` whenever `chunking.heading_metadata` is off and must not end up empty (config load fails; explicitly listing `section` while the flag is off logs a warning — the default, which never names `section` in `config.yaml`, does not), and disabled values are removed from the MCP tool schema/description, not just rejected at call time. The same goes for everything that only applies to a disabled granularity or, when `document` is disabled, to searching without a query: `explain`/`fields`/`order_by`/`descending` drop out of the schema, the no-query sentences drop out of the tool and server descriptions, and rejection errors never suggest a disabled granularity. `section_max_bytes` (default `16000`) is the size cap on every `get_document` read the caller did not bound itself: a resolved section falls back to an outline of its children past it, a whole-document read falls back to the document's own outline (or truncated text when the document has no headings), and outlines are capped to roughly that many bytes of entries. An explicit `start_line`/`end_line` range is exempt.
- **reranking** — Optional cross-encoder reranking pass over the top hybrid candidates (`enabled`, default `false`; `candidate_limit`, default `50`; `max_document_bytes`, default `6144` — per-document byte budget for the rerank request, sized to an 8192-token reranker served with `--ubatch-size` equal to `--ctx-size`; restart required). Requires `RERANKING_BASE_URL`/`RERANKING_MODEL` when enabled — see [Configuration](#configuration) above and `deploy/config.example.yaml`
- **ui** — Web UI graph-view tuning: `semantic_edges` adds precomputed kNN neighbor edges alongside markdown-link edges (off by default — each enabled run costs one Qdrant `recommend` query per indexed document). `api_auth` (default `false`; restart required) puts the same authentication as `/mcp` in front of the UI's `/api/*` routes. See [Web UI](#web-ui) below.

> **Hybrid search migration:** collections now use named `dense` + `sparse` vectors. Upgrading a knowledge base that was indexed by a pre-hybrid version requires a one-time full reindex (`mcp-md-wiki index --full`) — the old single-unnamed-vector schema is incompatible. After that, toggling `search.hybrid` needs no reindex (both vectors are always stored).

## Embedding Models

The default config is tuned for **nomic-embed-text-v2-moe** (768 dimensions, GGUF via llama.cpp).

### Download

```bash
# Download from Hugging Face (requires huggingface-cli: pip install huggingface_hub)
huggingface-cli download nomic-ai/nomic-embed-text-v2-moe-GGUF \
  nomic-embed-text-v2-moe-Q8_0.gguf --local-dir ./data/models

# Or download directly from:
# https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF
```

Then set in `.env`:

```env
MODEL_PATH=./data/models
MODEL_FILE=nomic-embed-text-v2-moe-Q8_0.gguf
```

### Alternative Models

To use a different model, override in `.env`:

```env
EMBEDDING_MODEL=bge-large-en-v1.5
EMBEDDING_VECTOR_SIZE=1024
MODEL_FILE=bge-large-en-v1.5-q8_0.gguf
```

These are environment-only — `embedding.model` and `embedding.vector_size` are not settable in `config.yaml`. Changing either invalidates every vector already in the collection, so it requires a full reindex, not just a restart.

| Model | `vector_size` | Notes |
|---|---|---|
| nomic-embed-text-v2-moe (default) | 768 | Recommended. MoE, strong quality/speed. |
| nomic-embed-text-v1.5 | 768 | Older nomic, same dimensions. |
| all-MiniLM-L6-v2 | 384 | Lightweight, lower quality. |
| bge-large-en-v1.5 | 1024 | Strong quality, larger vectors. |
| mxbai-embed-large-v1 | 1024 | Good alternative to bge. |

**Note:** Changing `vector_size` requires a full reindex (`index --full`) which drops and recreates the Qdrant collection.

## Embedding Backends

The dev `docker-compose.yml` defaults to **CPU mode** which works on any hardware. For production deployment, pick a hardware-specific template from `deploy/templates/`.

### Context Window Override

nomic-embed-text-v2-moe natively supports 8192-token context windows, but the GGUF file metadata incorrectly reports a 512-token limit. The `docker-compose.yml` command includes `--override-kv nomic-bert-moe.context_length=int:8192` to correct this, along with matching `--ctx-size`, `--batch-size`, and `--ubatch-size` flags. This allows embedding larger markdown chunks in a single pass. If you switch to a different model, adjust or remove these overrides accordingly.

### CPU (default)

Works everywhere with no special drivers. Good for small knowledge bases or initial testing. The compose file uses `ghcr.io/ggml-org/llama.cpp:server`.

### NVIDIA CUDA

Most common GPU backend. Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed on the host. In `docker-compose.yml`, uncomment the `## --- NVIDIA CUDA ---` block (which uses `server-cuda12` with `deploy.resources.reservations.devices` for GPU access), or use the `deploy/templates/compose-nvidia.yml` template.

### AMD ROCm

Best performance on AMD GPUs. Requires ROCm userspace drivers on the host.

> **RDNA3/RDNA4 (gfx11xx/gfx12xx):** check `rocm-smi` at idle after starting the
> stack. An AMD MES firmware bug ([ROCm/ROCm#5706](https://github.com/ROCm/ROCm/issues/5706))
> pins the GPU at 100% and ~90W continuously whenever an encoder model is offloaded
> through HIP — which is what the embedding and reranking servers do. Either update
> GPU firmware to MES `0x8b` (amdgpu 31.20.0+), or use the Vulkan backend below.
> See [GPU Backends](deploy/TROUBLESHOOTING.md#gpu-backends) for diagnosis and the
> workarounds that do and do not work. In `docker-compose.yml`, uncomment the `## --- AMD ROCm ---` block (which uses `server-rocm` with `/dev/kfd` and `/dev/dri` device access), or use the `deploy/templates/compose-rocm.yml` template.

For fine-grained control (e.g. targeting a specific GPU render node or setting `HSA_OVERRIDE_GFX_VERSION`), use a `docker-compose.override.yml`:

```yaml
services:
  embeddings:
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri/cardN:/dev/dri/cardN       # replace N with your GPU card number
      - /dev/dri/renderDN:/dev/dri/renderDN  # replace N with your GPU render node
    group_add:
      - "video"
      - "render"
    security_opt:
      - seccomp=unconfined
    environment:
      - HSA_OVERRIDE_GFX_VERSION=12.0.1  # 11.0.0 for RDNA 3 (RX 7000), 12.0.1 for RDNA 4 (RX 9000)
      - HIP_VISIBLE_DEVICES=0
```

Find your device nodes with `ls /dev/dri/` and match card/render numbers to your target GPU. Check `getent group video render` for the correct GIDs on your system.

### AMD Vulkan

Simpler driver setup than ROCm — works with standard Mesa Vulkan drivers. In `docker-compose.yml`, uncomment the `## --- AMD Vulkan ---` block (which uses `server-vulkan` with `/dev/dri` device access), or use the `deploy/templates/compose-vulkan.yml` template. Supports multi-GPU setups.

### Apple Silicon (Metal)

Metal GPU acceleration is **not available in Docker** (Docker on macOS runs a Linux VM). Options:

1. **Run llama-server natively** — `brew install llama.cpp`, then start it with your model and point `EMBEDDING_BASE_URL` at it (`http://host.docker.internal:8080/v1` if mcp-md-wiki runs in Docker).
2. **Use the CPU Docker image** — works but is slower than native Metal.

### External API

Skip the bundled embedding service entirely. Point `EMBEDDING_BASE_URL` at any OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, TEI) and remove the `embeddings` service from compose.

## MCP Tools

The server exposes exactly seven MCP tools. Read tools (`search`, `get_document`, `get_schema`, `changes_since`) are always safe; write tools (`write_document`, `delete_document`, `update_schema`) mutate the knowledge base and commit to git.

**Every tool result is one JSON object.** It is the MCP `structured_content`, and the result's single text block is that same object serialized compactly (the MCP spec's recommended fallback for text-only clients), so a client that reads only one of the two misses nothing. Keys whose value would be `false`, `null`, empty or a default are left out — an absent flag means false, an absent list means none. Errors are plain MCP error messages.

**Path handling is unified across every tool that takes a `path`.** A leading `/` means the knowledge-base root, not a filesystem path — callers have no way to know where the KB actually lives inside the container, so `/food/recipes/chili.md` and `food/recipes/chili.md` address the same document. Path-escape protections still apply on top of that: `/../x` is rejected the same as `../x`, and `get_document`, `write_document` and `delete_document` refuse a path with a `.git` component or a symlink anywhere along it, whatever `indexing.include` says. Partial paths resolve to a best match only where a tool says so — `get_document` accepts a bare basename when it's unique across the index, and `get_schema`/`update_schema` accept a partial directory when it matches on trailing segments (there `.` and `./` name the root, and `.` segments and doubled `/` are ignored). A single match resolves silently; multiple matches are refused with the candidates listed rather than guessed at. A partial directory that matches no existing scope is used literally rather than refused: `get_schema` then reports the rules of whichever ancestor scope governs it, and `update_schema` treats it as the directory to give a new schema, which is the normal way to introduce one. `write_document` and `delete_document` resolve nothing: the path is taken as spelled, relative to the KB root (a bare basename is a file at the root).

### Read

**`search`** — find documents. `query` is optional: with one, results are ranked (semantic, hybrid, or phrase — see below); without one, every match is returned in a stable order with an exact `total`/`has_more` for a *complete* enumeration. This one tool folds in the old `search` and `list_documents` tools; omitting `query` reproduces `list_documents`'s behavior exactly.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | string | no | Natural-language search query. Omit to enumerate instead of rank |
| `granularity` | string | no | `chunk`, `document`, or `section`. Defaults to `chunk` when `query` is set, `document` when it isn't — `section` is never a default while `chunk` or `document` is enabled. The available set is restricted per instance by `search.granularities` (see Configuration below); a disabled value is rejected and dropped from the tool schema/description entirely, and if the usual default is disabled the first enabled one that can serve the call is used (with a query: `chunk`, then `document`, then `section`; without one only `document` can) |
| `heading_prefix` | string[] | no | Restrict query results (any granularity) to everything under a run of consecutive headings, which may start at any level of the heading path: for a chunk under `# Chapter 10: Game Mastering` > `## Conditions` > `### Blinded`, each of `["Conditions"]`, `["Chapter 10: Game Mastering", "Conditions"]`, `["Conditions", "Blinded"]` and `["Chapter 10: Game Mastering", "Conditions", "Blinded"]` matches, while `["Chapter 10: Game Mastering", "Blinded"]` (not consecutive) does not. Each segment is a complete heading name, so `["Chapter 10", "Conditions"]` matches nothing there. Unlike `get_document`'s `heading_path`, whose segments (contiguous or not) must still appear in the path's own order, the run here must be contiguous but can sit anywhere in it. Matched ignoring case, invisible characters and whitespace differences, but not Unicode normalization form (NFC vs. NFD — a precomposed accented character does not match a decomposed spelling of the same text). Requires a `query` (rejected in enumeration mode, which has no heading data) and `chunking.heading_metadata`; an empty list or a blank segment (including one of only invisible characters) is rejected rather than silently matching everything |
| `filters` | object | no | Keyed by frontmatter field (dot-paths for nested fields, e.g. `planning.prep_minutes`). A scalar means equality (`{"type": "guide"}`); an array means any-of (`{"tags": ["recipe","dinner"]}`); an object means all-of or a numeric range (`{"tags": {"all_of": ["recipe","dinner"]}}`, `{"planning.prep_minutes": {"lt": 30}}`). Range operators: `gte`, `lte`, `gt`, `lt`; a bound may also be a date or RFC 3339 timestamp (`{"published": {"gte": "2024-01-01"}}`), in both modes. A range on a field declared with a type that stores no number (`text`, `enum`, …) is refused, naming the type. Also `any_of`, `all_of`. A field of a list's objects is `list[].field` (`{"npcs[].role": "guard"}`, any element), and a key ending in `[]` is a correlated list filter whose conditions must all hold in one element (`{"npcs[]": {"role": "guard", "level": {"gte": 3}}}`), in both modes — see below |
| `path_prefix` | string | no | Restrict by location, e.g. `lifestyle/kitchen/recipes/`. A case-insensitive substring of the path — see below |
| `modified_after` / `modified_before` | string | no | Exclude documents outside this mtime range; works in both modes |
| `limit` / `offset` | integer | no | Paging. Query mode: default 10, max 50 (`search.default_limit`/`search.max_limit`). Enumeration mode: default 100, max 1000 |
| `order_by` / `descending` | string / boolean | no | Enumeration only: sort by `path` (default), `title`, `mtime`, `indexed_at` |
| `min_score` | number | no | Query mode only: relevance floor |
| `explain` | boolean | no | Query mode only: add per-arm scores to each result (`dense_score`, `sparse_score`, `pre_rerank_score`, `phrase_matched`) and the ranking `mode` to the response |
| `fields` | string[] | no | Frontmatter dot-paths to include per result; omit for all |

Per field, `filters` operators are **mutually exclusive**: set matching (`any_of`/`all_of`) *or* a numeric range (`gte`/`lte`/`gt`/`lt`), never both on the same field, and never `any_of` together with `all_of` — mixing them is a validation error, not silently-honor-one-of-them behavior. `title` and `description` are filterable too, matched against dedicated columns rather than the generic dot-path projection every other field goes through; since each holds a single scalar, `all_of` with more than one value on either can never match anything (zero results, not an error).

**How frontmatter becomes filterable (enumeration mode).** Each document's frontmatter is projected into the SQLite `document_fields` table: nested objects as dot-paths, a list of scalars as one row per value under the list's name, and the objects of a list as `list[].field` rows keyed by their element (`npcs: [{role: guard, level: 3}]` gives `npcs[].role` and `npcs[].level`), down to 16 path steps (each name and each `[]` counts as one — the same limit `update_schema` allows a schema to declare). Range filters compare a numeric value whose meaning comes from the governing schema's declared type: the number for `integer`/`number`, 1/0 for `boolean`, seconds since the Unix epoch (UTC) for `date` (its midnight UTC) and `timestamp` (normalized from its offset), and **nothing** for `text` and `enum` — a `text` field holding `42` matches equality but no range. A field no schema types gets a value from its JSON type (numbers and booleans). Range bounds given as date strings are converted the same way, so `{"published": {"gte": "2024-01-01"}}` orders dates and timestamps together.

**How frontmatter becomes filterable (query mode).** Each chunk's Qdrant payload carries the document's whole frontmatter under one `fm` key, structure kept (`fm.planning.prep_minutes`, `fm.npcs[].role`), next to the built-in keys (`file_path`, `mtime`, `text`, line and heading keys). A value of a declared type is written as that type: `integer` as an integer (`3.0` becomes `3`), `number` as the number written, `text`/`enum` as a string, `boolean` as is, `date`/`timestamp` as RFC 3339 UTC (`2024-01-02` → `2024-01-02T00:00:00Z`, offsets normalized to `Z`) under a datetime index; undeclared fields are written as they are. Dates are read by the same parser as the enumeration mode's epoch seconds, so a date range means the same instants in both modes: a date-only bound is midnight UTC, so `{"lte": "2024-01-02"}` excludes a timestamp later that day in both. Equality on a date or timestamp field matches the instant (a timestamp written with another offset for the same instant matches in query mode; enumeration mode compares the text as written). Datetime bounds compare at microsecond precision. Float equality is refused in query mode (use a range), and a range needs a numeric or date field.

**Correlated list filters.** `{"npcs[].role": "guard", "npcs[].level": {"gte": 3}}` matches a document where *some* element is a guard and *some* (possibly other) element has level 3 or more. To require both of one element, key the filter by the list with `[]` and give its sub-fields as conditions: `{"npcs[]": {"role": "guard", "level": {"gte": 3}}}`. Each condition is any ordinary filter form (scalar, array, `any_of`, `all_of`, range); a sub-field may be a dot-path (`"stats.level"`) or a field of a nested list (`"gear[].kind"`, matched anywhere inside that element). One level only — the list may not itself sit inside a list, and a condition may not be another correlated filter — with at most 20 row tests per filter (each `all_of` value is one). Unknown sub-fields and values outside a sub-field's closed set are refused with the options listed, like any filter. With a `query` it runs as a Qdrant nested filter over the same element semantics; each sub-field must then be indexed (`indexed: true` on, say, `npcs[].role`), like any query-mode filter field.

**Query-mode `filters` can only name an indexed field.** Query mode runs the filter against Qdrant, which can only do that efficiently on a field it has a payload index for — naming an unindexed field is rejected by name with an actionable error (mark it `indexed: true` in the governing `.schema.yaml`) rather than silently filtering more slowly than expected. Enumeration mode has no such limit: it runs against the SQLite `document_fields` projection, not Qdrant.

**A filter that could only match nothing is refused, with the options listed** (both modes). A field no schema scope declares, that is not built in (`title`, `description`, the root schema's [`derived`](#root-only-blocks-derived-and-root_edit) fields, `frontmatter.indexed_fields`) and that no document carries is rejected as `unknown filter field '…'; filterable fields: …` (`data.filterable_fields`). A value outside a field's closed value set is rejected as `filter '…': '…' is not an allowed value; allowed: …` (`data.field`, `data.allowed`; several refused values read `'a', 'b' are not allowed values`). An any-of filter (a scalar or an array) is refused only when every one of its values is, since it still matches through the others; `all_of` is refused as soon as one is, since no document can carry it. The closed set is the union across the scopes `path_prefix` names (every scope whose directory contains the needle or is contained in it; all scopes when it names none or is absent), with the needle read as search matches it: a trailing `/` is ignored, a leading one is not, so `/food` names no top-level scope. A value some document actually uses is always accepted, so a refusal never hides a match. Lists are capped at 30. The server instructions do not enumerate field values: `get_schema` (with `values_in_use` for open fields) is where a caller lists them.

**Phrase search:** a double-quoted span inside `query` matches as an exact phrase, fused as a third RRF arm alongside the dense and sparse ones (`search.phrase`, default `true` — see [Configuration](#configuration) below). This degrades gracefully — logged, not fatal — against a Qdrant server too old for phrase-matching text indexes.

**`section` granularity** (requires a `query` and `chunking.heading_metadata` — enabling the flag re-indexes affected documents automatically; see Configuration below) returns one row per matching section — `file_path`, `heading_path`, `line_start`, `line_end`, `hit_line_start`, `hit_line_end`, `score`, and `scope` when it is not `section` — with **no chunk text**. `line_start`/`line_end` are the section's whole range; `hit_line_start`/`hit_line_end` are the best-scoring matched chunk's own lines inside it, which `get_document`'s `start_line`/`end_line` read on their own. Multiple chunk hits inside the same section collapse into one row, ranked by its best-scoring chunk. `explain` and `fields` are rejected the same way they are for granularities that don't support them. `scope` says what the row is and how to fetch it with `get_document` (described below): `section` — a heading's subtree, fetched by `heading_path` or `line`; `preamble` — the text before the first heading (no `heading_path`), fetched by `line`; `whole_document` — a document with no headings to select by (no `heading_path`, whole-body range; also reported when the file changed since it was indexed), read in full. Because `chunking.heading_metadata` stops chunks from spanning headings, a `section` row's `heading_path` is always the deepest heading containing the match, and its range is that heading's whole subtree. So a row for a heading that has sub-headings matched in that heading's own text before its first sub-heading (each sub-heading gets its own rows): read `hit_line_start`–`hit_line_end` for just that text (`get_document` reports the same text as `intro` when the section is too large to return whole), or the section for everything under the heading. `chunk`-granularity rows carry `file_path`, `title`, `score` (four significant digits, like every score), `text` — up to 800 characters of the chunk's own body, ending in `…` when cut; the heading breadcrumb and description the chunker prepends for the embedding are not repeated in it (the web UI's `/api/search` strips them too) — plus `line_start`/`line_end` and `type`/`tags` when set. With `chunking.heading_metadata` on they also always carry `heading_path` (`[]` for text before the first heading), and without it the key is omitted. `heading_prefix` is usable as a filter on query results only with the flag on. While an indexing run spanning more than one file is in progress (e.g. the re-index right after enabling the flag, from the moment the reconcile scan first finds a document chunked under different settings — a routine single-file reindex doesn't count), `section` and `heading_prefix` results set `indexing_in_progress: true`, since documents not yet reached lack heading metadata. That flag stays on through a failed run's retry backoff (or a permanent give-up) and the wait for the next reconcile, not just while a run is actively in flight — it clears only once a scan confirms no such file remains, or the run consuming one finishes successfully.

**`path_prefix` is a case-insensitive substring match on the document's path, and means the same thing in both modes.** It matches anywhere in the path, so it doubles as filename discovery: `stir_fr` finds `kitchen/recipes/stir_fry.md` without your knowing the folder, and `recipes/` finds everything under any `recipes` folder. A trailing slash is optional. Because it is a substring rather than a prefix, a short needle is broad — `sys` matches `sysadmin/` and `archive/old-sys/` alike — so prefer the longest fragment you're sure of.

Both modes resolve the needle through the same SQLite matcher (`LIKE '%needle%'` against the `documents` table), which is what keeps them from disagreeing: enumeration applies it inline, and query mode resolves it to a concrete path set and pushes that down to Qdrant as an exact filter. Case-insensitivity is SQLite's, so it folds ASCII only — a non-ASCII filename still matches case-sensitively.

Two things can make a filtered response incomplete, and either sets `path_prefix_truncated: true` rather than silently under-returning: the needle matched more documents than the server will filter on at once (use a longer needle), or — in query mode only — a document indexed before the `path_ancestors` payload field existed crowded out genuine matches within the over-fetch window. The second cause disappears per document as each is reindexed. Enumeration mode is exhaustive by construction and only ever reports the first.

Query mode returns `returned` and `results[]` (chunk/section) or `documents[]` (document), plus `path_prefix_truncated`/`offset_truncated` when set; enumeration mode returns `total`, `returned`, `documents[]`, and `has_more: true` when another page exists. Truncation is never silent either way. A document row is `file_path`, `title`, `description`, `score` in query mode, and `frontmatter` — without the keys already on the row (`title`, `description`) or derived from the path (the root schema's `derived` fields, such as `domain`, unless named in `fields`), and omitted when nothing is left; enumeration rows add `mtime` (Unix seconds) only when ordered by it.

**Derived fields, such as `domain`, are set from the path, not authored — when the root schema says so.** A root `.schema.yaml` with `derived: {domain: {from: path_segment, index: 0}}` (see [Root-only blocks](#root-only-blocks-derived-and-root_edit)) computes `domain` from each document's top-level folder name (a file at `infrastructure/docker-compose.md` gets `domain: infrastructure`; a file at the KB root has none) and writes it into both the Qdrant payload and the SQLite metadata index — it is not read from a `domain:` frontmatter key. A `domain:` key an author writes anyway is overwritten on the next index run, and the server logs a warning when the two disagree. Filtering is unchanged: `search(filters={"domain": ...})` — in either mode — and the CLI's `--domain` flag work as for any field. A write whose frontmatter carries a derived field is refused by validation (`rule: "derived"`: "`domain` is set from the document's folder; remove it from the frontmatter"); indexing and the CLI `validate` still accept existing documents that have one. A schema rule that makes a derived field `required`, or gives it a `default`, is ignored: it is never demanded of an author and never filled in. Without a `derived` block nothing is derived, and `domain` is an ordinary frontmatter field.

**`get_document`** — fetch the raw markdown (including frontmatter) for one document: in full, by line range, by resolved section, or as a heading outline.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | Path relative to the KB root (as returned by `search`), or a unique basename. A leading `/` also means the KB root — see path handling above |
| `start_line` | integer | no | First line to return, 1-based and inclusive. Omit to start at line 1 |
| `end_line` | integer | no | Last line to return, 1-based and inclusive. Omit to read to the end |
| `line` | integer | no | Select a section by any 1-based line in it: the deepest heading whose range contains the line (a line on a section's heading or in its text before the first sub-heading is that section). Not combinable with `start_line`/`end_line`/`heading_path` |
| `heading_path` | string[] | no | Select a section by heading path, e.g. `["Spells", "Fireball"]`, resolved in tiers, each tried only when the previous named no unique section: the exact full path; a trailing *suffix* (`["Fireball"]` alone is often enough); then an ordered match that may also skip a *middle* segment (`["Feats", "Dual Wielding"]` matches `Feats > Combat > Dual Wielding`) — segments must still appear in the path's own order. Each segment is a complete heading name, compared ignoring case (Unicode case folding), whitespace differences and invisible characters, but not Unicode normalization form (NFC vs. NFD); a segment that is empty once those are removed is rejected. When no tier resolves, the error may list candidate sections matched with a looser, substring-per-segment comparison — a suggestion, never a resolved read. Not combinable with `start_line`/`end_line`/`line` |
| `levels_up` | integer | no | Climb this many parent headings above the section selected by `line`/`heading_path` — a heading's parent is the nearest heading above it with a lower level, so from a `###` directly under a `#`, `levels_up: 1` reaches the `#` — clamped at the top-most heading rather than erroring. Requires `line` or `heading_path` |
| `outline` | boolean | no | Return a heading outline (with line ranges) instead of content: the whole document's, or — combined with `line`/`heading_path` (and optionally `levels_up`) — the selected section's sub-headings. Not combinable with `start_line`/`end_line` |
| `history` | integer | no | Also return this document's last N changes, newest first, clamped to 1–100. Combinable with every other parameter. Omitted: no history is read and the response is unchanged |
| `links` | boolean | no | Link lists ride on a whole-document read by default; `true` adds them to any read, `false` drops them |

Returns `path`, `content`, `version` and `total_lines`; a read of less than the whole file adds `start_line`, `end_line` and `partial: true`.

A whole-document read (or any read with `links: true`) also carries the link graph as plain paths: `links_out` (link targets that exist), `broken_links` (link targets that do not), `links_in` (documents linking here), and `similar` (`{path, score}` inferred neighbors in either direction, only when `ui.semantic_edges.enabled` is on). Each list is omitted when empty. A direction is one page of at most 20 edges (markdown links before semantic ones) shared by its lists; when it was cut, `links_out_total`/`links_in_total` give that direction's full edge count, which covers every edge (links to existing documents, broken links and semantic neighbors), not only the entries of one list.

With `history`, `history` is `{changes, truncated?}`: each change is `{date, author, subject, operation?}` — `date` in UTC (`2026-10-03T14:22Z`), `author` the author's name, and `operation` the tool that made it when this server did (parsed from the `Operation:` trailer it writes on every commit). `truncated: true` means older changes exist. A knowledge base that is not a git repository (no `GIT_URL`) returns `{available: false}` rather than an error. No revision id, email or trailer name reaches the caller. History reads never take the git write lock, so they don't wait behind an in-flight write. The web UI's `GET /api/history?path=…` serves a fuller listing (`{available, path, commits, limit, returned, truncated}`, each commit with `sha`, `author_name`, `author_email`, `timestamp`, `subject`, `tool`, `operation`, `tool_authored`), plus repository-wide recent activity and per-commit diffs.

**Whole-document reads are capped** at `search.section_max_bytes` (default 16000 bytes), the same dial the section and outline modes use. A document over the cap comes back as the whole document's outline — `outline_only: true`, the outline fields below, and `intro`: the range from line 1 (frontmatter included) through the line before the first heading, present only when there is anything there — so the caller narrows with `heading_path`/`line` instead of receiving the file. A document over the cap with *no* headings has nothing to narrow into, so its text is cut on a line boundary and flagged `truncated: true`, with `end_line` reporting the last line served: read on from `end_line + 1`. Both capped forms are still whole-document reads, so they carry the link lists. An explicit `start_line`/`end_line` range is never capped — the caller named the bounds. The web UI's document view and editor read `?start_line=1` for exactly that reason: a save posts the whole body back, so what it loads must be the complete file.

**Section and outline modes** (`line`/`heading_path`/`outline` — a long, heading-structured document): a section is a heading line through the end of everything nested under it, and `line` and `heading_path` are two ways to select the same section. Pass either, optionally with `levels_up` to climb to an ancestor, and the response carries `section: {heading_path, level, line_start, line_end}` and `content` in place of a raw range. `version` still covers the whole file, unchanged. A selected section larger than `search.section_max_bytes` (default 16000 bytes — see Configuration below) comes back as `outline_only: true` with no text: instead it carries the outline of its sub-headings (exactly what `outline: true` with the same selector returns) and `intro`: `{…, line_start, line_end}` when the section has non-blank text before its first sub-heading — read that with `start_line`/`end_line` — and no `intro` key otherwise. A section with no sub-headings has nothing smaller to offer, so its text is returned flagged `oversized: true`, cut on a line boundary when it exceeds the cap — `truncated: true` and `end_line` (the last line of the returned `content`) then say where it stopped. `outline: true` returns the whole document's headings, or with a selector the selected section's sub-headings (plus `section`), with no `content` at all. Every outline entry is `{heading_path, level, line_start, line_end}` (the heading is the last element of `heading_path`; the text before the first heading is an entry with `level: 0` and an empty path). An outline is capped to roughly `section_max_bytes` of entries, keeping the shallowest levels first; a capped one adds `truncated: true`, `total_entries` and a `hint` on reaching the rest (outline a listed section to go deeper). Section and outline responses also report `total_lines`, and `partial: true` unless the section is the whole file. The same modes are available on the web UI's `GET /api/doc/{*path}` via query params, including repeated `heading_path=` occurrences for a multi-segment path; both transports build these fields from one shared function, so the JSON is identical apart from MCP's link lists and `history`.

Line ranges are inclusive on both ends, and an `end_line` past the last line is clamped rather than rejected — `end_line` in the response reports what was actually served. A `start_line` past the last line *is* an error, and says how many lines the document has — except `start_line: 1`, which is always valid, even against an empty (0-line) document: it reads from the beginning, which is meaningful even when there is nothing there, and comes back as `content: ""`, `total_lines: 0` rather than an error. Content is sliced byte-exactly: line endings and an unterminated final line survive, so a slice can be handed straight back to `write_document` as an `old_string`.

**`version` always covers the whole document, never the slice.** Its purpose is the `expected_version` that `write_document` and `delete_document` take, which guards the file on disk — so reading lines 40–60 of a document and then replacing it works exactly as it does after a full read. The flip side is that it is not a checksum of the bytes you were handed. It is opaque to callers; pass it back unchanged.

**`get_schema`** — show the fully merged frontmatter rules governing a path, with per-field provenance (`declared_in`: the directory whose schema declared each field, e.g. `food/recipes/`, the root as `/`). See [`.schema.yaml` Directory Schemas](#schemayaml-directory-schemas) below for how the cascade works.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | no | Directory or document path to resolve rules for; a partial directory resolves if it matches one scope uniquely (multiple matches are refused with the candidates listed); omit for the root |
| `fields` | string[] | no | Only report these dot-paths; omit for all |
| `values_only` | boolean | no | Only report fields that declare a closed value set |
| `values_in_use` | boolean | no | Also report what documents under the path actually use: `in_use` (the 20 most-used values with document counts) on each open field, and `other_fields_in_use` (up to 30 fields documents here carry that no schema declares, derived fields excluded) |
| `compact` | boolean | no | Name each declaring directory once: fields carry no `declared_in`, and the result adds a top-level `declared_in` map of directory → the field names it declares (e.g. `{"/": ["tags", "title"], "food/": ["prep"]}`). Off by default, so existing callers keep the per-field shape |

Returns `path`, `fields` — an object keyed by field name, each definition carrying only what applies: `type` and `values` when set, `required`/`indexed` when true, `default` when there is one, `open: false` on an object that refuses undeclared keys, and `declared_in` (unless `compact`) — plus `omitted_fields` when the field cap cut the list and `dedup` when the scope overrides the near-duplicate check. A field a scope removes (`removed: true`) is not listed there or below, and `values` is the set after any `exclude_values`. A derived field (the root schema's `derived` block, e.g. `domain`) is never listed, even when a scope declares it (to index it, say). With `values_in_use`, open fields (not `text`/`timestamp`/`object`) add `in_use` and the result adds `other_fields_in_use`, each only when non-empty. The root is always `/`, including when its rules come from `config.yaml`'s `frontmatter` block.

**`changes_since`** — list the documents that changed across the whole knowledge base since a point in time or a previously seen head, for catching up on a shared corpus or processing only what is new. It is derived from the knowledge base's git history (so it needs the data directory to be a git working copy; otherwise the result is `{available: false}`) and, like `get_document`'s `history`, is a deliberate exception to "nothing git-related reaches a tool caller" — though the result carries no per-change revision id: each change has `path`, `status` (with `old_path` for a rename), `date`, `author`, `subject` and `operation`, and the only commit-shaped value is the opaque `head` token. `since`, `path_prefix` and `cursor` are capped at 4096 characters. `GET /api/changes` takes the same parameters and additionally returns each change's `commit`.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `since` | string | conditional | A timestamp (RFC 3339, `2026-03-01`, or `2026-03-01T12:30Z`, UTC) or the `head` an earlier call returned. Required unless `cursor` is given |
| `path_prefix` | string | no | Only documents under this folder |
| `limit` | integer | no | Changes per page, default 50, max 200, ordered by path |
| `cursor` | string | no | The `next_cursor` of a truncated page; continues that same range |

Returns `available`, `head` (pass it as the next `since`) and `changes[]`: `path`, `status` (`added`, `modified`, `deleted` or `renamed` — net of every commit in between), `old_path`? (a rename), and the latest change's `commit`, `date`, `author`, `subject` and `operation`?. A page cut at `limit` carries `truncated: true` and `next_cursor`. Only paths that are indexed documents (`indexing.include`/`exclude`/`exclude_files`) are listed. The web UI's `GET /api/changes` is the same listing over REST.

### Write

Both write tools share the same pipeline: **path-safety guard** (no `..`, no `.git` component, no symlink anywhere along the path (the document itself included), never a `.schema.yaml` — schema files are edited with `update_schema` — and the path must match `indexing.include`; a create, edit or move destination must also not be excluded by `exclude`/`exclude_files`, so a document the indexer would ignore, such as the default-excluded `README.md`, is refused with an "excluded from indexing" error rather than committed. A move source and a delete need only match `include`, so documents already in an excluded directory can still be moved out or deleted) → **frontmatter validation** (against the *destination* directory's schema, for a move) → **filesystem write** → **git commit with provenance trailers** (`Tool: mcp-md-wiki`, `Operation: <tool>`) → **push to the remote** → **queued incremental reindex** (the call returns once the push is done; the single background worker that also serves the webhook indexes the path shortly after). Commits are authored under the `write.commit_author_*` identity so tool edits are easy to spot in `git log`.

**Nothing about git reaches the tool caller.** Each write returns what happened — `path`, `action` — and the tool-specific fields below; an edit or a move adds its unified `diff` (capped at 8 KiB, with `diff_truncated: true` and `diff_total_bytes` when cut), while a create or a delete echoes no diff of content the caller already has; an edit that leaves the document exactly as it was commits nothing and succeeds as `updated` with no `diff`. Never a commit SHA, a sync state or the paths a rebase pulled in. A change committed locally whose push failed is reported as a plain success (logged at `warn`; a later write syncs it, as described below). The one concurrency signal on a success is `merged_with_other_changes: true` (present only when it applies): another writer's change to a document this call wrote was merged with this one, so the caller should re-read it before editing again. A failure says only whether the change was saved: "could not be saved. Nothing was changed; try again" after a clean rollback, or — when the rollback itself failed — not to retry and to report it to the operator. The underlying cause (git's own output included) is logged, never returned, and failures carry no `data.outcome`. The web UI's `/api/doc/...` is a separate contract and still returns `outcome`/`sha`.

| Tool / mode | Result fields |
|---|---|
| `write_document` (create / edit / single-document move) | `path`, `action` (`created`/`updated`/`moved`), `from`? (a move's source), `version`, `diff`? (edit/move), `diff_truncated`?, `diff_total_bytes`?, `rewritten_paths`?, `merged_with_other_changes`?, `related`? (create only: `[{path, score}]`, nearest first) |
| `write_document` (`archive: true`) | `path` (under the archive prefix), `from`, `action: "archived"`, `version`, `diff`, `rewritten_paths`?, `status: "archived"`? or `status_note`? |
| `write_document` (batch, `documents`) | `documents[]` (`path`, `action` (`created`/`updated`), `version`, `diff`? (update), `diff_truncated`?, `diff_total_bytes`?, `merged_with_other_changes`?) |
| `write_document` (directory move) | `path`, `action: "moved"`, `from`, `moved[]` (`from`, `to`), `moved_schema_dirs[]`?, `rewritten_paths`?, `merged_with_other_changes`? |
| `delete_document` | `path`, `action: "deleted"`, `referencing_paths`?, `rewritten_paths`? (with `unlink_inbound`, in which case `referencing_paths` is empty), `merged_with_other_changes`? |
| `update_schema` | `path`, `summary` (each op's summary, joined with `; `), `invalidated`?, `casualties_total`?/`casualties_truncated`? (only when the list was capped), `warning`? (saved but not in effect yet: another directory's schema is invalid), `warnings`? (lint findings for the schema file as written, cascade checks included — see [Lint warnings](#syntax); capped at 20, never a refusal). A `dry_run` preview instead returns `dry_run: true`, `path`, `summary`, `would_invalidate`? and `warnings`?, plus — for a single op — `field` and `definition` (the edited field as the scope would resolve it, in `get_schema`'s shape), or — for a batch of several — `ops`: `[{operation, field, summary, definition?}]`, each field as the whole batch leaves it |

`?` marks a field present only when it applies.

**Concurrent writers.** Every write brings the server's copy of the knowledge base up to date with the remote first, then reads, edits, validates and saves under one lock, so two writers never interleave. What happens when the document changed since the caller read it depends on the kind of change:

- **Relative edits** — `old_string`/`new_string`, `frontmatter_patch`, `append`, and every `update_schema` operation — apply to the current document, so a concurrent change to another part of it is kept. If `old_string` no longer matches because someone else changed that text, the write is refused with "the document was edited by someone else; re-read it and try again". An `expected_version` is optional for them: when one is given and stale, the edit still applies and the result is flagged `merged_with_other_changes`, since it carries the changes made since.
- **Absolute changes to an existing document** — a full `content` replace, a single-document move, a delete, and a batch entry that replaces an existing document — require `expected_version`, the `version` from `get_document`; without it they are refused and told to pass it. A full replace built on an older version is three-way merged with the changes made since: a clean merge is saved (re-validated) and reported as `merged_with_other_changes`; overlapping changes are refused. A stale move or delete is refused. A directory move re-checks its whole subtree and is refused if anything in it changed. A create (a path with no document) given an `expected_version` is refused as not found — the document it was read from is gone; omit `expected_version` to create one anew.
- A save whose push loses a race with another writer, or whose rebase conflicts, has its own commit dropped, the server's copy re-synced with the remote, and the change re-applied to the fresh document, up to three attempts in all; a change that no longer fits the fresh document, or one that loses the race every time, is refused as edited by someone else. A failure of any other kind (the remote unreachable, a push a hook refuses) keeps the local commit and reports success. The next write rebases that commit onto the remote and pushes it; if it genuinely conflicts with what the remote gained meanwhile, it is saved under `refs/mcp-md-wiki/unsynced/<sha>` in the clone (logged at `error`) and the clone is reset to the remote tip.

**`write_document`** — create, edit, and/or move a document. It's an upsert: `content` creates `path` when it's new and replaces it when it already exists. This one tool replaces the old `create_document`, `edit_document`, and `move_directory` tools.

- **Full-replace** — `content`: the whole file, including frontmatter. Creates on a new path, replaces on an existing one.
- **Surgical** — `old_string` + `new_string`: replaces a single unique occurrence (Claude Code-style) instead of resending the whole file. Errors if `old_string` is missing or appears more than once. Mutually exclusive with `content`.
- **Move** — `new_path`: relocates the document. Combines with either edit mode above (edit-then-move, one commit), or stands alone for a pure move (the server reads the current body itself). Moving a single document requires its `expected_version`. Links pointing at the document are rewritten. If `path` is a directory, its *whole subtree* moves — this is what replaces `move_directory`.

On create, a **near-duplicate check** runs first: it embeds the content and, if an existing document scores at or above `write.dedup_threshold`, refuses the write and names the match (pass `force_new: true` to override). `write.dedup_enabled`/`dedup_threshold` are the global defaults; a directory's `.schema.yaml` can override either via a [`dedup:` block](#per-directory-near-duplicate-override). That score is always a dense cosine similarity — the check is pinned to dense-only retrieval with reranking detached, so it is unaffected by `search.hybrid` and `reranking.enabled`.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | conditional | Document or directory to write to, relative to the KB root; a leading `/` also means the KB root. Required unless `documents` is set |
| `content` | string | conditional | Full-replace mode: whole file, including frontmatter |
| `old_string` | string | conditional | Surgical mode: exact text to find (must be unique) |
| `new_string` | string | conditional | Surgical mode: replacement text |
| `frontmatter_patch` | object[] | conditional | Structured frontmatter edits, body untouched, fields not changed keep their exact formatting (a dot-path re-renders its whole top-level field): `{operation, field, value}` / `{operation, field, values}` applied in order, `operation` one of `set_field`, `remove_field`, `add_values`, `remove_values`, `field` a dot-path |
| `append` | string | conditional | Text added to the end of the body, separated by exactly one newline; never lands inside the frontmatter |
| `new_path` | string | no | Relocate here; combines with any edit mode, or stands alone for a pure move (or directory move) |
| `expected_version` | string | conditional | The `version` from a prior `get_document`. Required to replace (`content`) or move an existing document; optional for the relative edit modes, which apply to the current document; a create given one is refused as not found |
| `message` | string | no | Commit subject (default depends on what changed) |
| `force_new` | boolean | no | Skip the duplicate-detection gate on create (default: `false`) |
| `archive` | boolean | no | Retire the document: move it to `<write.archive_prefix>/<path>`, set `status: archived` and fix inbound links, in one commit (see below). Needs `expected_version` |
| `archive_links` | string | no | With `archive`: `repoint` or `unlink`, overriding `write.archive_links` for this call |
| `documents` | object[] | no | Batch write: up to 25 entries (`path` plus the same edit modes, `expected_version` (required for an entry replacing an existing document), `force_new`; no `new_path`) written as one atomic commit. Excludes `path` and every other top-level field except `message` |

**Related documents on create.** The duplicate check's query also yields near neighbours below the duplicate threshold: a create returns up to `write.related_limit` of them (default 5, max 20) that score at least `write.related_min_score` (default 0.50) as `related: [{path, score}]`, nearest first, so the writer can link to them and check them for conflicts. There is no extra embedding call. The key is omitted when empty, when dedup is disabled or bypassed (`force_new`), for edits, moves and batch writes, and when the limit is `0`. A `.schema.yaml` [`dedup:` block](#per-directory-near-duplicate-override) can override either value per directory.

**Archive.** `archive: true` (with `expected_version`) retires a document in one commit: it moves to `<write.archive_prefix>/<its path>` (default `_archive/…`; the prefix is an ordinary indexed path, so archived documents stay searchable and can be filtered on `status`), `status: archived` is set when the destination's schema declares a `status` field that allows it (otherwise the result carries a `status_note` and the status is left alone), and links from other documents are handled per `archive_links`: `repoint` (default) rewrites them to the archived location, `unlink` replaces each with its plain label (wiki links and autolinks included; a reference-style definition is dropped and its use sites left). It refuses a document already under the prefix, an existing destination, and a state where the document index is unavailable. Like a plain move, `repoint` replaces only a link's path: a `#fragment`, `"title"` or wiki `|alias` is kept as written.

Exactly one edit mode is required unless the call is a pure move (`new_path` alone). `content` and `old_string`/`new_string` are whole-document edits, mutually exclusive with each other and with `frontmatter_patch`/`append`; those two combine (patch first, then append). `new_path` is orthogonal to every edit mode. A document is capped at 512 KiB: `content` over it is refused, and so is a relative edit that would grow a document past it (one that shrinks a larger document is fine).

**`delete_document`** — remove a file. Commits the deletion (with provenance trailers) and pushes, then marks the path for the reindex worker, which purges the document's vectors from Qdrant and its row from the state DB. Links to it from other documents are left as they are; those documents come back as `referencing_paths`.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | Path relative to the KB root; a leading `/` also means the KB root. A bare basename is not resolved |
| `expected_version` | string | yes | The `version` from a prior `get_document`; a document changed since then is not deleted |
| `unlink_inbound` | boolean | no | Replace links that point at the document with their plain label, in the same commit, instead of leaving them dangling (the rewritten documents come back as `rewritten_paths`) |
| `message` | string | no | Commit subject (default: `docs: delete <path>`) |

### Schema

**`update_schema`** — edit a directory's `.schema.yaml` through constrained operations instead of free-form text. It always writes `.schema.yaml`, removing a legacy `.kb-schema.yaml` in the same commit.

The operations go in `ops`, a list applied in order to the one schema file, **atomically**: every op is applied in memory, then the result is validated once (self-consistency, the casualty check below, the round trip) and written as **one commit** — if any op fails, nothing is written and the error names it (`ops[2] (remove_field 'x'): …`). At most 50 ops per call. Each op is `{operation, field, values?, definition?}`:

| `operation` | Takes | Effect |
|---|---|---|
| `add_values` | `values` | Permits the values. In a subdirectory whose schema has no `values` of its own for the field it writes `[$values, ...new]` — extending the inherited set, keeping the inherited `type` (`enum` when no ancestor gives a type or values) — rather than narrowing it to just the new values; a value the scope already inherits is not written again, and one the scope excludes has its exclusion dropped |
| `remove_values` | `values` | Stops permitting the values. A value in this directory's own list is deleted from it; an **inherited** value is added to the field's `exclude_values` here instead (the parent keeps it) |
| `set_field` | `definition` | Declares or replaces this directory's definition of the field outright (the same keys as a `.schema.yaml` entry) |
| `patch_field` | `definition` and/or `reinherit` | Changes only the keys given (a merge patch): nested `fields` merge per child (and `items` likewise), any other key replaces, and `null` drops this directory's own setting so the attribute is inherited again (`{"required": null}`; `{"fields": {"x": null}}` drops a child's declaration). `reinherit` names keys to inherit again without a `null` (`"reinherit": ["required"]`), for clients that enforce the advertised schema, which has no `null` arm; a key may not be both set in `definition` and listed there, and `reinherit` on any other operation is refused. The result is **minimized** against the inherited definition: an attribute equal to what the field inherits is not written, and a declaration left saying nothing is removed. A patch that would change nothing in a directory that does not declare the field is refused rather than writing an empty file |
| `remove_field` | — | Drops this directory's declaration of the field (to hide an inherited field here and below, `set_field` it to `{"removed": true}`) |

Example — retire a value, add one and make a field required, in one commit:

```json
{"path": "food/recipes", "ops": [
  {"operation": "remove_values", "field": "tags", "values": ["draft"]},
  {"operation": "add_values", "field": "tags", "values": ["weeknight"]},
  {"operation": "patch_field", "field": "servings", "definition": {"required": true}}
]}
```

The flat `operation`/`field`/`values`/`definition` parameters of earlier releases are **deprecated** and accepted for one more release as a one-op batch (same result, same response shape) — except `patch_field`, which is only available through `ops`, since the flat `definition` cannot carry a patch's `null`s; passing them together with `ops` is refused. `add_values`, `remove_values` and `patch_field` on a field the directory itself removes (`removed: true`) are refused: re-declare it with `set_field` first. The result's `path` is the scope directory (`notes/`, the root as `/`). Before writing, the change is validated against every document already under that scope, using the frontmatter stored in the metadata index (no markdown re-read). If any document would fail the new rules, the change is refused and they're listed; `force` applies anyway, `dry_run` reports what would happen without writing. The rendered YAML is re-parsed and size-checked (256 KB cap) before writing, so an invalid schema can never be committed; an existing file that has gone invalid on disk can't be edited through this tool and has to be fixed in the git repository. The file is written temp-then-rename and, like the document write tools, committed and pushed; a full reconcile is then queued, which re-validates the documents the schema governs.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | no | Directory whose schema to edit; a partial directory resolves if it uniquely matches an existing scope (multiple matches are refused with the candidates listed, no match falls back to the literal path so a new scope can be introduced); omit for the root |
| `ops` | object[] | yes (or the deprecated flat parameters) | The operations, in order, all or nothing (table above). Each op's `field` is a dot-path for nested fields (`npcs[]` for a list's elements, `npcs[].name` for a field of each); `definition` accepts the same keys as a `.schema.yaml` entry (`type`, `required`, `indexed`, `values`, `default`, `open`, `fields`, `items`, `exclude_values`, `removed`) — also as a JSON string containing that object; `values` replaces the inherited set unless it includes `$values`. The deprecated `extend` is still accepted but not advertised in the tool schema |
| `operation`, `field`, `values`, `definition` | — | no | **Deprecated** (one release): one op, as top-level parameters; mutually exclusive with `ops` |
| `dry_run` | boolean | no | Report the effect without writing (default: `false`) |
| `force` | boolean | no | Apply even if existing documents would fail the new rules (default: `false`) |
| `acknowledge_root_change` | boolean | no | When the root `.schema.yaml` sets [`root_edit: {require_ack: true}`](#root-only-blocks-derived-and-root_edit), must be `true` to change the root schema with any op but `remove_values` — in a batch, when any one op needs it; `remove_values` and `dry_run` are exempt. Ignored when the root schema sets no guard (default: `false`) |

When validation fails, write tools return a structured error whose `data.field_errors` array names each offending `field` (a dot-path, with list elements indexed: `npcs[2].name`), the `rule` it broke (`required` / `allowed_value` / `lint` / `type_mismatch` / `closed_object` / `derived`; past 100 errors, one final `too_many_errors` entry counts the rest), and — for closed-set fields — the value it `got` and the values it `expected`, so an agent can self-correct.

Server and tool descriptions are assembled from three layers, in order: mechanics true of every deployment (compiled into the binary), short sentences that depend on this deployment's config or corpus (e.g. the quoted-phrase sentence on `search` when `search.phrase` is on, the enabled granularities, and — in the server instructions — the top-level areas plus one pointer to `get_schema` for field values and per-folder rules; no field vocabularies, scoped folders or authoring rules are enumerated up front), and this knowledge base's own policy — what belongs here, tagging conventions, writing style — loaded at runtime from `<mcp.extensions_path>/` in the served repo (default `meta/mcp`) and refreshed on `mcp.metadata_refresh_secs`. The per-KB layer is append-only: editing it is a normal `write_document` call, no restart required. `mcp.instructions` is a deprecated narrative override that still works but is superseded by `<extensions_path>/server.md` when both are set.

**Description length budget.** Claude Code truncates each tool description and the server instructions at 2048 characters (its observed behavior, not an MCP spec limit); input-schema *parameter* descriptions are not truncated. So each rule has one home: tool descriptions carry what a tool does, when to use it, and the cross-parameter workflow a model would get wrong without being told; per-parameter rules live on the parameters; responses describe themselves; and the server instructions repeat nothing a tool description says. The compiled and config-derived text is held to 600 characters per description by tests (server instructions measured against a realistic corpus), which leaves roughly 1400 characters for each per-KB extension file — keep extensions under that, or Claude Code cuts the end off. Input schemas are compacted before they are served (no `$schema`, no `null` arm on optional properties, no `"default": null`, no non-standard `format`), and `operation`, `frontmatter_patch[].operation` and `order_by` advertise their values as an `enum` (the server still accepts the same strings and aliases as before). The server logs a warning when a fully composed description or the server instructions exceed 2048 characters, naming the surface and its length. See the [`mcp` section of the config reference](deploy/config.example.yaml) and [deploy/USAGE.md](deploy/USAGE.md#agent-write-tools) for details.

## Web UI

The server also ships a browser UI, served at `/` on the same port as MCP (8001) — no separate service, no separate port. Point a browser at your reverse proxy's hostname, or at `http://localhost:8001/` on the Docker host — the shipped compose files publish the port on `127.0.0.1` only (`MCP_BIND`, see `deploy/.env.example`).

It's docs-first: a sidebar document tree and a semantic-search results panel (backed by the same `search` retrieval core as the MCP tool) are the primary views. A Cytoscape.js graph view is available alongside them — either a whole-knowledge-base view or, from any open document, its link neighborhood — and a full document editor lets you create, edit, move, and delete documents directly from the browser.

The editor's writes are not a separate, lesser code path: `POST`/`DELETE /api/doc/{*path}` run through the exact same `write_document`/`delete_document` pipeline the MCP write tools use — the same frontmatter validation against the `.schema.yaml` cascade, the same near-duplicate check on create, the same commit-and-push to the knowledge base's git remote, and the same incremental reindex trigger. Editing a document in the browser and editing it through an MCP client produce indistinguishable commits.

Besides the shell and `/api/doc/{*path}`, the UI's JSON routes are `GET /api/graph`, `/api/search` (`query`, `granularity` = `chunk`/`section`/`document`, `filters`, `path_prefix`, `limit`, repeatable `heading_prefix`; chunk rows carry `text_truncated`, always `false` here), `/api/enumerate` (search without a query: `filters` as JSON, `path_prefix`, `limit` default 100 and clamped to 1–1000, `offset`; returns `{total, has_more, documents: [{file_path, title, description, frontmatter}]}`, 400 on a bad filter or vocabulary), `/api/schema/{*path}`, `/api/history` and `/api/changes` (the `changes_since` listing).

**By default the web UI has no authentication of its own, on any of its routes, including the write and delete ones.** This is deliberate, not an oversight: `MCP_BEARER_TOKEN` (and `mcp.allow_unauthenticated`) gate `/mcp`, `/status`, `/metrics`, `POST /admin/reload` and `POST /admin/sync` — they do **not** gate `/`, `/api/graph`, `/api/search`, `/api/enumerate`, `/api/schema/{*path}`, `/api/changes` or `/api/doc/{*path}` (GET, POST, *or* DELETE). Setting a bearer token does not add a credential requirement to the web UI in any way; it remains exactly as open as `/health`. The opt-in exception is `ui.api_auth: true` (restart required), which puts the same authentication layer as `/mcp` (the static token and/or an OAuth access token) in front of every `/api/*` route; `/` and `/assets/*` stay open so the page can load, and a browser then has to present the credential itself, so it suits API clients and proxies that inject a header better than a plain browser session. The intended deployment shape puts an identity-aware reverse proxy — Authentik via Traefik, in the reference deployment this project targets — in front of the whole port, and relies on that proxy for the web UI's access control entirely.

**Practical consequence:** if you publish port 8001 directly (a bare `docker run -p 8001:8001`, a cloud load balancer with no auth in front, a home-lab port-forward) without a reverse-proxy auth layer ahead of it, anyone who can reach that port can read, create, edit, and delete every document in the knowledge base through the web UI — regardless of whether `MCP_BEARER_TOKEN` is set. Put an authenticating reverse proxy in front of port 8001 for any deployment reachable beyond a fully trusted network; the MCP bearer token alone does not cover this surface. The server logs a startup warning whenever `ui.api_auth` is off. The shipped compose files bind the published port to `127.0.0.1` by default (`MCP_BIND`); a Traefik on the same Docker network reaches the container port directly and is unaffected, while clients on other hosts connecting straight to `:8001` need `MCP_BIND=0.0.0.0`. `deploy/templates/traefik-labels.yml` carries a commented-out forward-auth (Authentik outpost) example that protects the UI while leaving `/mcp` and the other self-authenticating routes on their own router.

**Browser-side guards.** Every `/api/*` route checks where a request came from, independent of `ui.api_auth`: with `mcp.allowed_hosts` set, a request whose `Host` is not in the list is refused with `403` (the same matching `/mcp` uses — an entry without a port matches any port, case-insensitive), which defeats DNS rebinding. With `mcp.allowed_hosts` unset (the default) there is no `Host` check at all, on `/mcp` or on `/api/*`: a DNS-rebinding page in a browser that can reach the port gets through, and with `ui.api_auth` off that includes the write routes — the server warns about exactly that combination at startup. Set it to the hostname(s) clients use; and a `POST`/`PUT`/`PATCH`/`DELETE` that carries an `Origin` header must come from the request's own host (compared host-only, since a TLS-terminating proxy changes the scheme and port) or from a host in `mcp.allowed_hosts`, else `403`. A request with no `Origin` (curl, scripts) passes. Every UI response — the shell, `/assets/*` and `/api/*` — carries `Content-Security-Policy` (scripts from this server only, no framing, external images allowed for document markdown), `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` and `X-Frame-Options: DENY`.

## `.schema.yaml` Directory Schemas

A file named `.schema.yaml` governs its directory and everything beneath it, cascading like `CLAUDE.md` — including at the knowledge-base root. `frontmatter` in `config.yaml` used to be the *only* way to declare root-level rules; it's now a deprecated fallback (see [Backward compatibility](#backward-compatibility) below), and a root `.schema.yaml` — authored with the exact same syntax as any other directory — is the preferred way to declare them, since it's part of the knowledge base's own git repo rather than deployment config on the container host.

Top-level folder names are the KB's areas — what a `domain` derived with `index: 0` holds, when the root schema declares one (see [Root-only blocks](#root-only-blocks-derived-and-root_edit)). The MCP server's dynamic instructions list them from a directory read (`Top-level areas: …`). Filtering on a derived field with a query needs it indexed, whether via the deprecated `frontmatter.indexed_fields` or an `indexed: true` entry for it in a root `.schema.yaml`.

### Syntax

```yaml
fields:
  planning:
    type: object
    open: false          # reject undeclared keys under planning.* (default: true)
    fields:
      prep_minutes: { type: integer, indexed: true }
      effort:       { type: enum, values: [low, medium, high], indexed: true }
  tags:
    type: list
    values: [$values, dinner, quick]   # the inherited values, then these (without $values the list replaces them)
```

Nested authoring (as above) and flat dot-paths (`planning.prep_minutes:`) are equivalent — both name the same field in the resolved schema tree.

**List elements (`items`).** A `list` field can describe its elements with `items:` — a scalar definition, or an `object` with its own `fields`. Declaring `items` on a field with no `type` makes it a `list`, the way `fields` makes it an `object`; `items` on any other type, or alongside `fields`, makes the file invalid:

```yaml
fields:
  npcs:
    items:
      type: object
      fields:
        name: { type: text, required: true }
        role: { type: enum, values: [ally, foe] }
```

**Paths.** Everywhere a schema names a field — a key under `fields:`, `update_schema`'s `field` — it uses one grammar: `a.b` is a child field, `a[]` the elements of list `a`, and `a[].b` a field of each element (`npcs[].role:` as a key is the same field as the nested form above). `items` definitions are merged down the cascade like any other field, shown by `get_schema` under the list field's `items`, part of the schema fingerprint, and enforced on every element (see below). Their fields are filterable (`npcs[].role`, or a [correlated list filter](#read)), and their types decide each element field's numeric value and payload type; one marked `indexed: true` gets a Qdrant payload index (`fm.npcs[].role`), which query-mode filters on it need.

**Types:** `text`, `integer`, `number`, `boolean`, `enum`, `list`, `date` (`YYYY-MM-DD`), `timestamp` (RFC 3339), `object`. Declared types are strictly enforced with no coercion — `prep_minutes: "45"` fails against `type: integer` — except that an `integer` accepts a whole number written as a decimal (`3.0`, never `3.5`). Undeclared fields are not type-checked and remain legal.

**How nested rules apply.** Validation walks the schema tree:

- `required` on a nested field applies only when its parent is present: with `planning.effort` required, a document without `planning` (or with `planning: null`) passes, and one with `planning: {}` does not. A top-level required field is always checked. A nested `default` creates its parent object, so once a scope defaults one child of `planning`, the other required children of `planning` apply to every document.
- `open: false` is enforced on every object it is set on, at any depth, inside list elements included.
- A list's elements are checked against its `items` — type, `values`, and for object elements their own `required`/`open`/child rules — and an error names the element by index: `npcs[2].name`. Defaults are never applied inside list elements.

**Lint warnings.** Some definitions load fine but are probably mistakes: a `default` that fails the field's own `type` or `values`, `values` on an `object` or `boolean` field, a `default` under a list's `items` (never applied to elements), and a top-level field named like a key the search index writes itself (`file_path`, `mtime`, `chunk_index`, `text`, `text_body_offset`, `line_start`, `line_end`, `path_ancestors`, the `heading_*`/`section_*` keys, and `fm`), and a root-only block (`derived`, `root_edit`) in a schema file below the root, where it has no effect. Cascade checks compare a file with what its directory inherits: a `type` changed to a different kind of value (see [Cascade and merge rules](#cascade-and-merge-rules)), `removed: true` on a field no parent declares, `exclude_values` naming a value the field does not have there (or on a field with no value set at all), and `values`/`exclude_values` that no longer permit the default the field inherits (every document omitting it would get an invalid value). These are warnings, never errors: the server logs them (naming the scope) once at startup and in `mcp-md-wiki validate`, and afterwards only those a schema rebuild newly introduces; `update_schema` returns them in `warnings`.

A field can't declare both a scalar `type` and nested `fields:` — it's either a value or a container, not both (`type: object` is the exception, since `object` means "has nested fields"). `update_schema` enforces this the same way a hand-edited `.schema.yaml` does.

Declaring one path twice in a file (nested under `fields:`/`items:` and as a flat path key) makes the file invalid (see [Invalid schema files](#invalid-schema-files)). `update_schema` takes a nested field as a path (`planning.method`, `npcs[].role`) and edits it in place; `add_values`/`set_field`/`patch_field` create any missing parents as `type: object` fields (`type: list` before a `[]`).

### Cascade and merge rules

- The **set** of fields unions across cascade levels. Merging is per attribute: a field redefined at a deeper level overrides only the attributes it writes (`type`, `required`, `indexed`, `default`, `open`, `values`), and every attribute it leaves unwritten inherits from the nearest ancestor that declared the field. The same holds for nested fields and `items` at any depth, and redefining a container never drops the children it inherits.
- `values` is the one attribute with an in-band merge: a list you write replaces the inherited set outright unless it contains the `$values` placeholder, which splices the inherited set in at that position (`[$values, dinner, quick]`: inherited first, then these, duplicates dropped). Any other `$`-prefixed token in the list makes the file invalid. `extend: true` is the deprecated spelling of a leading `$values`; it still works and logs a warning. See [deploy/USAGE.md](deploy/USAGE.md#cascade-and-merge-rules).
- `exclude_values: [...]` takes values out of the set this directory ends up with: it applies **after** the declaration's own `values` (and its `$values` splice), so `values: [$values, x]` with `exclude_values: [draft]` is the inherited set plus `x`, minus `draft` — a value in both lists is excluded. On a field with no value set it does nothing, and on a field nothing above declares it does not even declare the field. `$`-prefixed entries make the file invalid. `update_schema`'s `remove_values` writes it for an inherited value.
- `removed: true` deletes an inherited field — with everything nested under it — from this directory and every directory below it; a deeper directory may declare the field again, from scratch. It works at any depth (`planning.effort: {removed: true}`, `npcs[]: {removed: true}` drops a list's element schema). A removing declaration carries no other key (that makes the file invalid).
- **Only same-kind type changes are meant to cascade.** A deeper directory may retype an inherited field within one kind of value — `text`/`enum`/`date`/`timestamp` (a string), `integer`/`number` — but changing it to a different kind (a scalar to a `list` or `object`, a list to an object, anything to or from `boolean`) makes documents valid in the parent invalid below it. For now that is a [lint warning](#syntax), not an error; it is meant to become one.
- `removed` and `exclude_values` change the schema fingerprint only through what they do to the resolved rules, so a schema that uses neither hashes exactly as before and nothing is reprocessed on upgrade.
- **Upgrade order:** an older server rejects these keys (and `items`) as unknown and refuses the whole schema. Deploy the server that understands them before any schema file uses them; rolling back afterwards needs those keys taken out first.
- Resolution is a single tree walk at startup/reindex time, with in-memory longest-prefix lookup per document afterward — not a walk per file.

### Per-directory near-duplicate override

A `.schema.yaml` may carry a `dedup:` block that overrides `write.dedup_enabled` / `write.dedup_threshold` (and the create result's `write.related_limit` / `write.related_min_score`) for its directory and everything beneath it — useful for structurally templated folders (meal plans, recipes) whose documents are legitimately near-identical:

```yaml
dedup:
  enabled: false     # skip the near-duplicate check on create here, or
  threshold: 0.95    # keep it but only refuse at >= 0.95 (range 0.0–1.0)
  related_limit: 3         # related documents a create returns here (0 = off, max 20)
  related_min_score: 0.6   # their minimum score (range 0.0–1.0)
```

All keys are optional and cascade independently, nearest scope winning; a key no schema sets falls back to the global `write.*` value. A threshold outside 0.0–1.0, a wrong-typed value or an unknown key under `dedup:` makes the file invalid like any other schema error. Hand-edit the file — `update_schema` has no operation for it but preserves an existing block. `get_schema` reports the effective override (only when set). Changing it does not trigger revalidation or reindexing.

### Root-only blocks: `derived` and `root_edit`

Two blocks configure corpus-wide policy. They take effect only in the **root** `.schema.yaml`, apply to every directory beneath it, and are absent by default — absent means nothing is derived and root edits are not guarded. In any other schema file they are ignored, and a lint warning says so. Like `dedup:`, they are hand-edited: `update_schema` has no operation for them but preserves both when it rewrites the root file.

```yaml
derived:
  domain: { from: path_segment, index: 0 }    # the top-level folder
  campaign: { from: path_segment, index: 1 }  # the folder below it
root_edit:
  require_ack: true
  policy_doc: meta/schema-tag-policy.md        # optional; named in the refusal
```

**`derived`** names frontmatter fields that are set from a document's location rather than read from its frontmatter. `from: path_segment` with `index: n` takes the document's `n`-th directory (0 is the top-level folder; the file name never counts): `food/recipes/chili.md` gets `domain: food` and `campaign: recipes`. A document with no such directory — one at the knowledge-base root, or one not deep enough — gets no value for that field. The derived value is written into both the Qdrant payload and the metadata index, overriding any value an author wrote (the server logs a warning when the two disagree). A write whose frontmatter carries a derived field is refused (`rule: "derived"`: "`domain` is set from the document's folder; remove it from the frontmatter"), while indexing and the CLI `validate` still accept existing documents that carry one. A schema `required` or `default` on a derived field is ignored, `get_schema` never lists one, search result rows leave it out of `frontmatter` unless it is named in `fields`, and a `search` filter on one is always accepted. `from` must be `path_segment`; `index` is at most 32; a field name is 1–64 ASCII letters, digits, `_` or `-`, never a name the index stores itself (`file_path`, `text`, … `fm`, plus `title` and `description`); at most 16 fields; a field named twice is an error. Any of these, or an unknown key, makes the file invalid. `derived` is part of the schema fingerprint (only when non-empty, so a schema without it hashes as before), so adding or changing it reprocesses the documents it governs — with the root schema, all of them — by refreshing their metadata and Qdrant payload in place, not by re-embedding them (see [Schema-change detection](#schema-change-detection)).

**`root_edit`**: with `require_ack: true`, `update_schema` refuses an `add_values`, `set_field`, `patch_field` or `remove_field` on the root scope — a batch holding any one of them — unless the call passes `acknowledge_root_change: true` (`remove_values` and `dry_run` are exempt). The refusal names `policy_doc` when set (at most 256 bytes, no control characters). `root_edit` is not part of the fingerprint.

### Invalid schema files

Every `.schema.yaml` in the indexed tree that is present must be valid; a root one is optional. The schema tree skips hidden directories and any directory whose every document `indexing.exclude` rules out (`templates/**`, say): a schema file there is never read, so it cannot break anything, and changing it triggers nothing (a directory move that would bring one into the schema tree checks it first, below). A file is invalid when it can't be read, is over 256 KB (refused on its size alone, never parsed), doesn't parse (an unknown key, a wrong type, a bad `dedup:` block), or contradicts itself. A symlink or other non-regular entry named `.schema.yaml` is not a schema file: neither the schema walk nor `update_schema` follows it. An invalid file is never loaded, and never falls back to the parent's rules:

- **At startup it is fatal.** `serve` refuses to start, and the error lists every invalid file with its reason. `mcp-md-wiki index` (incremental or `--full`) aborts the same way before touching anything, and `mcp-md-wiki validate` prints them in a `SCHEMA ERRORS` section and exits non-zero whether or not strict mode is on.
- **At runtime it is refused.** When a push or a reconcile brings in an invalid file, the schema rebuild is refused: the server keeps enforcing the last schema that loaded cleanly, indexes nothing until the file is fixed, and logs the refusal at error level on every reconcile sweep. An indexing run that hits the file in the meantime is dropped (logged at error level, every invalid file named) and sets the status straight away. `/status` reports it as `schema_error` and `/metrics` as `kb_schema_invalid`/`kb_schema_invalid_files` (see [Observability](#observability)); `/health` stays healthy. Fixing the file (a push to the knowledge base's git host) queues a full reconcile that catches up on everything written meanwhile, and so does the first rebuild that succeeds after a refusal.

Schema files are edited through `update_schema`, never through the document tools: `write_document`, `delete_document` and the web UI refuse any path named `.schema.yaml` (or the legacy `.kb-schema.yaml`). `update_schema` validates the file it writes (parse, self-consistency, round-trip, the 256 KB cap), so it cannot introduce an invalid one. A directory move carries the schema files under it along — and is refused, with nothing moved, while one of them is invalid on disk: a file the schema tree reads at the move's source, or would read at its destination (one coming out of a hidden or excluded directory included, which no rebuild has ever vouched for); fix or revert it in git first.

A changed `.schema.yaml` — in a webhook push, in a write's rebase, or carried by a directory move — queues a full reconcile, which rebuilds the schema before re-validating the documents it governs.

**Legacy file name.** Schema files used to be named `.kb-schema.yaml`. That name is still read everywhere `.schema.yaml` is, so an existing knowledge base keeps working with no change. The next `update_schema` in a directory still on the old name writes `.schema.yaml` and removes `.kb-schema.yaml` in the same commit, so knowledge bases migrate one directory at a time as they are edited (or all at once, by renaming the files in git yourself). A directory holding **both** names is an invalid schema file — the server never picks one — until they are merged into `.schema.yaml`. A directory move carries whichever name it finds.

Model-facing text — tool descriptions, results and errors — names a schema by its directory (`food/recipes/`, the root as `/`), never by file name. Operator-facing surfaces (logs, `/status`'s `schema_error`, the `validate` CLI) keep real file paths, and the `kb_schema_invalid*` metric names are unchanged.

### Backward compatibility

`config.yaml`'s `frontmatter` block is a deprecated fallback for root-level rules, used only when no root `.schema.yaml` exists anywhere — in that case, existing deployments keep working unchanged, though every index run logs a warning naming the fallback.

The moment a root `.schema.yaml` is added, it **replaces** `config.yaml`'s `frontmatter` block outright rather than merging with it: `required`/`indexed_fields`/`defaults`/`allowed` there stop applying unless the same field is also declared in the root `.schema.yaml` (every index run also logs a warning about this, so the switch is never silent). This is deliberate — a schema describes the knowledge base's own content rules, and a KB that carries its own root `.schema.yaml` must validate the same way regardless of which host's `config.yaml` happens to be serving it. `get_schema` (omit `path` for the root) always shows exactly what's in effect and where each field came from.

### Schema-change detection

A `schema_hash` fingerprint is stored per file alongside its content hash. The incremental indexer skips a file only when **both** match, so editing a schema revalidates every document under it even though their bytes didn't change. Two consequences: the first index run after upgrading to this feature revalidates every file once (backfilling the fingerprint), and editing a root-level schema revalidates the whole KB.

A schema edit does **not** re-chunk or re-embed anything: chunk text does not depend on the schema, so a document whose bytes and chunking settings are unchanged is revalidated, its metadata index row and links are rewritten, and — only when the edit changed what goes into its Qdrant payload (a field `default`, the root `derived` block, or a `type` change that writes the value differently, such as `text` → `integer`) — its payload is updated in place. A second per-file hash, `payload_hash`, tracks that payload-shaping part separately. The one exception: while `chunking.prepend_description` is on (the default), a document's `description` is prepended to every chunk's text, so a schema edit that changes the description a document gets from a `description` default (adding, changing or removing the default for a document without its own) still re-embeds that document; a document with its own `description` is not affected. A document the edited schema rejects is handled as before (dropped from the index in non-strict mode, kept at its last good version in strict mode).

The root schema's [`derived`](#root-only-blocks-derived-and-root_edit) block is part of that fingerprint, so adding or changing it is when every document's derived fields get (re)computed from its folder and written to Qdrant (in place, without re-embedding) and the state DB. If a document's old, hand-authored value (`domain:`, say) disagreed with its folder, the effective value changes at that point — update any saved filters that assumed the old value, and remove the now-redundant key from frontmatter (it's overridden either way).

### Qdrant payload indexes

Declared fields get typed Qdrant payload indexes on their `fm.` payload key (`fm.tags`, `fm.planning.prep_minutes`, `fm.npcs[].role`) — integer/number/boolean fields get Integer/Float/Bool indexes and date/timestamp fields a Datetime index (enabling range filters) instead of a blanket Keyword index; a `list` with typed `items` is indexed as the item type. The same applies to the built-in `mtime` index used by `search`'s recency filters. `/status` and `/metrics` name each index by its payload key (`fm.tags`; `file_path`, `mtime`, `path_ancestors`, `text` and the heading keys stay unprefixed). Every startup and indexing run reconciles the indexes against the collection: a missing one is created, one whose kind no longer matches the declared type is deleted and recreated, and an unprefixed index left on a declared field's name by a version before the `fm` namespace is dropped — so a type change needs no manual step. Index-creation failures are logged as errors but never abort startup or indexing; a filter on an unindexed field still returns correct results, just more slowly.

## Observability

These HTTP endpoints have different audiences and different auth:

| Endpoint | Auth | Purpose |
|---|---|---|
| `/health` | open | Liveness/readiness. Reports Qdrant and embedding-service reachability only. Returns 503 when either is down. |
| `/status` | bearer / OAuth | Full runtime state as JSON, including the running `version` and build `revision` (the image's tree hash, `unknown` outside a Docker build). |
| `/metrics` | bearer / OAuth | The same data in Prometheus text exposition format; `kb_build_info{version,revision}` carries the build. |
| `/status/links` | bearer / OAuth | The corpus-wide dead-link report (`GET /status/links?limit=N`, default 200, max 5000): `mcp-md-wiki lint`'s output as JSON. 503 `{"error": "no index yet"}` before the first index. It can list slightly fewer links than the per-document `broken_links`, since it also skips targets that exist on disk but are excluded from indexing. |
| `POST /admin/reload` | bearer / OAuth | Re-read and re-validate `config.yaml` and swap it in, without a restart. See [Config reload](#config-reload). |
| `POST /admin/sync` | bearer / OAuth | Fetch the configured branch, fast-forward the clone and queue what changed — the webhook's sequence, on demand. Returns `{status, changed, old_head, new_head, paths_queued, paths_filtered_out, full_reconcile_queued}`; 400 without `source.git_url` or when the data path is not a checkout, 502 when git fails. |

`/status`, `/status/links`, `/metrics`, `/admin/reload` and `/admin/sync` require the same credential as `/mcp` — the static bearer token (`MCP_BEARER_TOKEN`) or, with `mcp.oauth` enabled, a valid OAuth access token — because unlike `/health` they enumerate tag vocabularies, area names and document counts (`/status`/`/metrics`) or can change how the write tools authenticate content and which webhook provider is trusted (`/admin/reload`) — none of that gets a weaker gate than `/mcp` itself. Scrape `/status`/`/metrics` with an `authorization` stanza:

```yaml
scrape_configs:
  - job_name: mcp-md-wiki
    metrics_path: /metrics
    authorization:
      credentials: ${MCP_BEARER_TOKEN}
    static_configs:
      - targets: ["mcp-md-wiki:8001"]
```

Both report:

- **Indexing state** — whether a run is in flight right now, its phase (`discovering` → `scanning` → `embedding` → `backfilling` → `removing_orphans`), how far through it is, what triggered it (`cli`, `startup`, `webhook`, `write_tool`), and how long it has been going.
- **Last run** — outcome, duration, error message on failure, and the full per-outcome tallies (`discovered`, `indexed`, `skipped`, `invalid`, `empty`, `read_errors`, `metadata_backfilled`, `orphans_removed` — files deleted from disk or newly excluded by `indexing.include`/`exclude`/`exclude_files`, which a reconcile purges from the index — `strict_rejected`, and `refreshed`: unchanged files revalidated under an edited schema and refreshed without re-embedding).
- **Store counts** — `indexed_files` (state DB), `documents` (metadata index), and `qdrant_points`. `documents_missing_metadata` is the divergence between the first two; non-zero means the metadata index is behind and the next run will backfill it.
- **Metadata breakdown** — document counts per value for each indexed field, widest document coverage first, plus a synthetic `area` field grouping by top-level directory. Fields are ordered by how many documents carry them rather than how many values they take, so a scoped schema's twenty recipe fields can't crowd out `type` and `status`. Broad vocabularies like `tags` report their most common values with `truncated: true`. The root schema's [`derived`](#root-only-blocks-derived-and-root_edit) fields are omitted (a path-derived field restates the path; a `domain` derived from the top-level folder duplicates `area`), as are date and timestamp fields and the fields of list elements (`npcs[].role`), which describe records inside a document rather than the document.
- **Payload index health** — which Qdrant payload indexes are in place and which failed. Failures are non-fatal by design, so this is the only lasting signal that a filter may be slow or incomplete.
- **Schema errors** — `schema_error` (`since_unix`, `since_at`, `files[]` of `path` and `reason`) while a runtime schema rebuild is being refused and the previous schema is still in effect; absent otherwise. `kb_schema_invalid` is 1 for as long as that lasts and `kb_schema_invalid_files` counts the files. Alert on `kb_schema_invalid`: nothing is indexed while it is 1.

`kb_index_last_success_timestamp_seconds` is the metric worth alerting on: its age answers "is the index actually keeping up", which neither `/health` nor a bare error count can. It is absent until a run succeeds, so an alert on timestamp age will not fire spuriously against a freshly started process.

If a backing store is unreachable, `/status` still answers — the failure is reported in `store.errors` (and counted by `kb_status_errors`) rather than failing the request, so "is it indexing?" stays answerable while Qdrant is down. Those error strings are scrubbed of credentials first: the Qdrant client renders its full connection URL on a transport failure, and since there is no separate `qdrant.api_key` setting, an authenticated Qdrant can only be reached by embedding the credential in `QDRANT_URL`.

Responses are cached for 5 seconds and collected single-flight. One request costs roughly two dozen SQLite queries plus a Qdrant round trip, and nothing about the answer changes meaningfully within that window — a scrape burst collapses into one refresh instead of a query storm.

`documents_missing_metadata` can legitimately read negative, and is reported that way rather than clamped. It means the metadata index holds more documents than the state DB tracks, which orphan removal cannot produce on its own — the likely cause is a CLI `index` run interleaving with the server's, since the reindex lock only serializes within one process.

### Config reload

`POST /admin/reload` re-reads `config.yaml` from disk, re-validates it exactly the way startup does (same checks, same errors), and swaps it into the running server — no restart, no dropped connections. Every setting has exactly one legal source (see [Configuration](#configuration) above): ENV vars are for startup/secrets and are never re-read by a reload, so a reload always means the same thing — re-run the YAML side of config resolution and swap the result in.

```bash
curl -X POST -H "Authorization: Bearer $MCP_BEARER_TOKEN" http://localhost:8001/admin/reload
```

The response reports exactly what happened, bucketed by whether the change actually took effect:

- **`applied`** — read fresh by the code that uses it (an MCP tool call, a webhook request, the reindex worker's next drain, a periodic timer's next tick), so the very next read observes the new value. `search.*`, `write.*`, `webhook.provider`, `indexing.include`/`exclude`, `frontmatter.*`, `validation.*`, `mcp.instructions`, `mcp.extensions_path`, `mcp.disabled_tools` / `mcp.enabled_tools`, `embedding.batch_size` / `embedding.batch_concurrency` (only indexing uses them, and each indexing run builds its embedding client from the live config), and more fall here. A change to either `mcp.disabled_tools` or `mcp.enabled_tools` takes effect starting with the very next `tools/list`/`tools/get`/`tools/call` — no metadata-refresh-tick or restart wait — but an already-connected client that cached an earlier `tools/list` is not proactively notified (this server does not advertise the `listChanged` tool capability), so it may keep offering a now-disabled tool in its own UI until it calls `tools/list` again, though the call itself is refused either way.
- **`restart_required`** — baked into a value or service built once at server startup (the search/write-side embedding client's `reqwest::Client` timeout — `embedding.request_timeout_secs` is reported twice, `applied` for indexing and `restart_required` for that client — the MCP path-filter `GlobSet`, the rate limiter, anything security-critical like the bearer token, `allow_unauthenticated` or any `mcp.oauth` key, `ui.api_auth`, and `embedding.query_prefix` / `embedding.document_prefix`, whose running values are kept). The swap updates what everything else sees, but this particular consumer keeps behaving exactly as it did before the reload.
- **`reindex_scheduled`** — `chunking.*`. Every indexed file records a fingerprint of the chunking config (plus the chunker's code version) it was chunked under, so the full reconcile the reload queues re-chunks and re-embeds every document built under the old value automatically — no `index --full` needed (`target_chunk_size` is the exception while `heading_metadata` is on: it has no effect then and is not fingerprinted). Expect one re-embed of the affected corpus; search may serve a mix of old and new chunks until that reconcile finishes. The same happens on restart after editing `chunking.*`.
- **`reindex_required`** — `ui.semantic_edges.*`. The indexer reads the new value on its next run, but only for documents that run re-embeds; existing semantic edges are unchanged otherwise. Run `mcp-md-wiki index --full` for a consistent graph.

A malformed or invalid `config.yaml` — bad YAML, a value that fails validation, a missing required env var — is rejected with a 400 and that error message, and the running config is left **completely untouched**, the same guarantee a failed restart on that file would give you. A successful reload also queues an immediate full reconcile, so indexing-observing changes reach the corpus on the reindex worker's very next wake rather than waiting for the periodic sweep.

### Logging

`RUST_LOG` defaults to `info,rmcp=warn`. The `rmcp` demotion matters on an actively used server: the MCP transport logs three INFO lines per request, which otherwise buries the indexing pipeline's output entirely. Its warnings and errors, including tool-call failures, still come through.

Long phases report progress on a 10-second cadence rather than staying silent — a full re-embed is a single API sequence that can run for many minutes, and a run that simply stops logging is indistinguishable from one that is still working. Every run ends with an explicit terminal line on both the success and failure paths.

## Webhook

POST to `/hooks/reindex` triggers:

1. HMAC signature verification (Gitea/GitHub/GitLab)
2. Branch matching against `source.branch`
3. `git fetch` + `git merge --ff-only` (if `source.git_url` is configured). When the clone cannot fast-forward because it holds a commit a failed push left unpushed, the pull rebases that commit onto the remote instead (the next write pushes it); one that conflicts with the remote is saved under `refs/mcp-md-wiki/unsynced/<sha>` and the clone reset to the remote tip, as a write's own pre-write sync does
4. The changed paths from that pull are filtered through `indexing.include`/`exclude`/`exclude_files` — the same predicate a full reconcile applies, so a push touching a non-indexable path (e.g. the default-excluded `README.md`) is never marked dirty — and the survivors are marked dirty as the handler returns immediately; it does not reindex inline. A single background worker drains that queue and does the actual indexing asynchronously, off the request path (see [Architecture](docs/ARCHITECTURE.md#webhook-flow) for the full design). If `source.git_url` isn't configured, there's nothing to fetch or diff, so the webhook instead marks the whole corpus for a full reconcile.

The webhook endpoint is only available if `WEBHOOK_SECRET` is set to a non-empty value. Without a webhook, `POST /admin/sync` (running server) or `mcp-md-wiki git-pull` (CLI, indexes in-process) run steps 3 and 4 on demand — see [CLI Commands](#cli-commands) and [Observability](#observability).

**Setup options:**

- **Native forge webhook** (recommended) — configure directly in your Git forge's webhook settings (or via `tea`/`gh` CLI). No CI runner needed.
- **CI workflow** — trigger from a pipeline step. See [`deploy/ci-examples/`](deploy/ci-examples/) for sample Gitea and GitHub workflows.

**Git pull on webhook:** Set `source.git_url` in your config and `GIT_PULL_TOKEN` in `.env` (for private HTTPS repos) to have the container pull changes automatically when the webhook fires. See [`deploy/USAGE.md`](deploy/USAGE.md#7-set-up-incremental-reindexing-optional) for detailed setup instructions.

## Incremental Indexing

Files are tracked by SHA256 content hash in a SQLite state database. On each run:

- **New files** — validate, chunk, embed, upsert
- **Changed files** — delete old vectors, re-process
- **Deleted files** — remove vectors and state entry
- **Unchanged files** — skip

Point IDs are deterministic UUIDs (v5) derived from `file_path::chunk_index`.

## Deployment

All deployment artifacts live in [`deploy/`](deploy/):

- **Compose templates** — `deploy/templates/` has self-contained compose files for each hardware backend (CPU, NVIDIA, ROCm, Vulkan, Apple Silicon)
- **Config examples** — `deploy/.env.example` and `deploy/config.example.yaml`
- **CI examples** — `deploy/ci-examples/` has sample webhook workflows for Gitea and GitHub
- **Deploy script** — `deploy/deploy.sh` pulls and restarts via Docker context (configure with `deploy/deploy.env`)

**Claude Code users:** Run `/deploy-md-rag` for an interactive guided setup that walks through hardware selection, model download, configuration, and MCP client connection.

**Manual setup:** Copy the matching template from `deploy/templates/` to your target as `docker-compose.yml`, configure `.env` from the example, and follow [`deploy/USAGE.md`](deploy/USAGE.md).

## Development

```bash
# Set up git hooks (fmt + clippy on commit)
./scripts/setup-dev.sh

# Start only the dependencies
docker compose up qdrant embeddings -d

# Run the server locally (requires env vars for connection settings)
export EMBEDDING_BASE_URL=http://localhost:8080/v1
export EMBEDDING_MODEL=nomic-embed-text-v2-moe
export QDRANT_URL=http://localhost:6334
export MCP_BEARER_TOKEN=dev-token
cargo run -- serve
```

Typical workflow: develop locally, push to a feature branch, CI builds and tests, merge via PR. See [deploy/USAGE.md](deploy/USAGE.md) for full setup walkthrough.

## License

AGPL-3.0-or-later. See [`LICENSE`](LICENSE).
