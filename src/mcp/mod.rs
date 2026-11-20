use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::RwLock;

use anyhow::Context as _;
use globset::{Glob, GlobSet, GlobSetBuilder};
use rmcp::{
    ErrorData as McpError, RoleServer, ServerHandler,
    handler::server::{router::tool::ToolRouter, tool::ToolCallContext},
    model::*,
    service::RequestContext,
    tool_handler,
};
use tracing::{error, warn};

mod get_document;
mod schema_tools;
mod search;
mod search_ctx;
mod write_document;

/// Test builds only: `state::filter_parity` is the sole consumer.
#[cfg(test)]
pub(crate) use self::search::parse_field_filter;
pub(crate) use self::search::{
    SearchFiltersInput, SearchParams, resolve_search_granularity, validate_path_prefix,
    validate_search_params,
};
pub(crate) use self::search_ctx::SearchCtx;
pub(crate) use self::write_document::related_json;

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;
use crate::{
    config::ResolvedConfig,
    embed::EmbedClient,
    qdrant::QdrantStore,
    rerank::RerankClient,
    retrieval::RetrievalDeps,
    state::StateDb,
    tool_schema,
    write::{self},
};

const MAX_QUERY_LEN: usize = 4096;
const MAX_PATH_LEN: usize = 4096;
const MAX_FILTER_STR_LEN: usize = 256;
/// Deepest dot-path `update_schema` will nest; see `build_schema_edit`. The same
/// limit the metadata projection flattens to (#349), so every declarable field is
/// filterable.
const MAX_SCHEMA_PATH_SEGMENTS: usize = crate::schema::MAX_FIELD_PATH_DEPTH;
const MAX_CONTENT_LEN: usize = write::MAX_CONTENT_LEN;
/// Cap on the number of operations a single `write_document` `frontmatter_patch`
/// call may carry — a document write should rarely need more than a handful of
/// field edits at once; this bounds the work `write::apply_frontmatter_patch`
/// does per call, mirroring `MAX_SCHEMA_VALUES`'s reasoning for `update_schema`.
const MAX_FRONTMATTER_PATCH_OPS: usize = 20;
/// Cap on the `values` list of a single `add_values`/`remove_values`
/// `frontmatter_patch` operation.
const MAX_FRONTMATTER_PATCH_VALUES: usize = 200;
/// Cap on the serialized size of a single `frontmatter_patch` value —
/// mirrors `MAX_SCHEMA_DEFINITION_LEN`'s identical reasoning for `update_schema`'s
/// `set_field` definitions: a document's frontmatter is committed and re-parsed
/// on every read, so an oversized value is a durable cost, not a transient one.
const MAX_FRONTMATTER_VALUE_LEN: usize = 4 * 1024;

/// Resolve a caller's requested `limit` against the configured default and ceiling.
///
/// An over-large request is clamped rather than rejected — a caller asking for more
/// than the maximum gets the maximum, which is friendlier than an error and matches
/// the historical behaviour when both values were hardcoded.
fn resolve_limit(requested: Option<u64>, default_limit: u64, max_limit: u64) -> u64 {
    requested.unwrap_or(default_limit).min(max_limit)
}

/// Parse an ISO 8601 date/datetime string to a Unix timestamp (seconds).
///
/// Accepts RFC 3339 datetimes (e.g. `2024-01-15T12:00:00Z`) and date-only
/// strings (e.g. `2024-01-15`, interpreted as midnight UTC).
pub(crate) fn parse_date_to_timestamp(s: &str) -> Result<i64, String> {
    // RFC 3339 / ISO 8601 datetime first, then date-only YYYY-MM-DD (start of day UTC).
    crate::document_fields::parse_instant(s)
        .map(crate::datetime::floor_seconds)
        .ok_or_else(|| {
            format!(
                "invalid date '{}': expected RFC 3339 (e.g. 2024-01-15T00:00:00Z) \
                 or date-only (e.g. 2024-01-15)",
                s
            )
        })
}

/// How many invalidated documents to name before summarizing the rest.
const MAX_REPORTED_CASUALTIES: usize = 20;
/// Cap on permitted values a single `update_schema` call may add to a field.
const MAX_SCHEMA_VALUES: usize = 500;
/// Cap on the serialized size of a `set_field` definition.
const MAX_SCHEMA_DEFINITION_LEN: usize = 8 * 1024;
/// Cap on permitted values echoed back per field by `get_schema`.
const MAX_REPORTED_VALUES: usize = 200;
/// Cap on fields echoed back per scope by `get_schema`.
const MAX_REPORTED_FIELDS: usize = 500;
/// Most-used values listed per open field by `get_schema`'s `values_in_use`.
const MAX_VALUES_IN_USE: usize = 20;
/// Cap on the unified diff an edit's result carries (fix #129). Mirrors
/// `MAX_SCHEMA_DEFINITION_LEN`'s convention for bounding a single text blob:
/// `WriteSuccess::diff` is unbounded by design (a full replace of a large
/// document produces a large diff), but embedding it verbatim would let one
/// write emit a multi-megabyte tool result. See
/// [`capped_diff`] for the same unbounded-source/bounded-payload split
/// [`capped_casualties`] already applies to `update_schema`'s casualty list.
const MAX_STRUCTURED_DIFF_BYTES: usize = 8 * 1024;

/// Normalize a caller-supplied scope path into a safe KB-relative directory.
///
/// Rejects absolute paths and any `..` component — a schema written outside the KB
/// would govern nothing and could clobber unrelated files.
///
/// The result is rebuilt from the path's normal segments alone, so a `.` segment or a
/// doubled separator (`food/./recipes`, `food//recipes`) never reaches a scope label, a
/// commit message or the `LIKE` pattern `get_schema` counts values with: each is plain
/// `food/recipes`. A lone `.` (or `./`) is the KB root, like an empty path.
fn normalize_scope_path(raw: &str) -> Result<std::path::PathBuf, McpError> {
    use std::path::{Component, PathBuf};

    let trimmed = raw.trim().trim_matches('/');
    if trimmed.is_empty() {
        return Ok(PathBuf::new());
    }
    if trimmed.len() > MAX_PATH_LEN {
        return Err(McpError::invalid_params(
            format!(
                "path too long: {} chars (max {})",
                trimmed.len(),
                MAX_PATH_LEN
            ),
            None,
        ));
    }

    let candidate = PathBuf::from(trimmed);
    if candidate.is_absolute() {
        return Err(McpError::invalid_params(
            format!("path must be relative to the knowledge-base root, got '{raw}'"),
            None,
        ));
    }
    let mut normalized = PathBuf::new();
    for component in candidate.components() {
        match component {
            Component::Normal(segment) => normalized.push(segment),
            // Only a leading `.` survives `components()`; it names no directory.
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(McpError::invalid_params(
                    format!("path must not contain '..' or absolute segments, got '{raw}'"),
                    None,
                ));
            }
        }
    }

    Ok(normalized)
}

/// Default page size for `list_documents` — well above `search`'s cap, since
/// enumeration is the point.
const DEFAULT_LIST_LIMIT: u64 = 100;
/// Hard cap on a single `list_documents` page.
const MAX_LIST_LIMIT: u64 = 1000;
/// Cap on how many filter fields one call may specify.
const MAX_LIST_FILTERS: usize = 20;
/// Cap on values within a single field's filter, so one call cannot generate an
/// unbounded number of bound SQL parameters.
const MAX_FILTER_VALUES: usize = 500;

#[derive(Clone)]
pub struct KbSearchServer {
    embed_client: Arc<EmbedClient>,
    qdrant: Arc<QdrantStore>,
    collection: String,
    canonical_data_path: PathBuf,
    /// Glob patterns (from `indexing.include`) used to restrict `get_document` to permitted file types.
    include_patterns: Arc<GlobSet>,
    /// Dynamic MCP server instructions, refreshed periodically with discovered metadata.
    instructions: Arc<RwLock<String>>,
    /// Per-tool description overlay, keyed by tool name, applied over the
    /// router's own `Tool` entries in `list_tools`/`get_tool` (see this
    /// module's hand-written `ServerHandler` impl). `#[tool_handler]`
    /// regenerates the router from `Self::tool_router()` on every call, so a
    /// description baked into a `#[tool(...)]` attribute can never be swapped
    /// at runtime — this overlay is what makes `descriptions::compose_all`
    /// (recomputed by the same periodic refresh that updates `instructions`
    /// above) actually reach `tools/list` and `tools/get` without a restart.
    /// `std::sync::RwLock`, not tokio's: `get_tool` is generated as a
    /// non-async fn by the trait, so it cannot `.await` a tokio lock.
    description_overlay: Arc<RwLock<HashMap<String, String>>>,
    /// Live config handle. Every tool call fetches its own fresh snapshot via
    /// [`Self::config`] rather than caching one at construction, so `POST
    /// /admin/reload` is observed by the very next call — see that method's doc
    /// comment for exactly which settings this makes dynamic.
    config: crate::config::SharedConfig,
    /// The shared, cached schema tree — built once at server startup and kept
    /// current by the reindex worker (which rebuilds it before indexing any
    /// dirty schema file) and by `update_schema`'s own synchronous rebuild.
    /// `get_schema`, `update_schema`, and the write path all read this instead of
    /// re-walking the knowledge base on every call; see `schema::SharedSchemaCache`.
    schema_cache: crate::schema::SharedSchemaCache,
    rerank_client: Option<Arc<RerankClient>>,
    /// Handle to the document metadata index, opened on first use and held for the
    /// process lifetime. Under WAL this reader coexists with the short-lived writer
    /// pools that the reindex worker and CLI open.
    ///
    /// Lazy rather than constructor-injected so building a server stays synchronous
    /// and infallible with respect to SQLite availability.
    state_db: Arc<tokio::sync::OnceCell<StateDb>>,
    /// The dirty-path queue every write tool marks paths on (see
    /// `write::WriteDeps::queue`) and `write_raw_file`/`update_schema` call
    /// `mark_full` on directly. `server::run_server` constructs exactly one
    /// `ReindexQueue`, clones this `Arc` into `KbSearchServer`, `UiState`,
    /// `WebhookState`, and `reindex::run_worker` — all four MUST share the same
    /// instance, since the worker only ever drains the one it was handed. There
    /// is no ambient/global fallback (see `reindex::ReindexQueue`'s doc
    /// comment); a server built without one has no way to get indexing to
    /// happen at all, which is deliberate — it makes "which queue does this
    /// producer feed" a constructor argument instead of a runtime assumption.
    reindex_queue: Arc<crate::reindex::ReindexQueue>,
}

/// Build an include `GlobSet` for MCP path filtering, with a `**/*.md` fallback
/// when no patterns are valid. Per-pattern parsing uses [`crate::ingest::parse_globs`]
/// so both sites share the same skip-and-warn policy; this function keeps its own
/// failure policy (fall back to `**/*.md` on empty or builder error) distinct from
/// `ingest::build_globset`, which propagates errors to the caller.
fn build_include_globset(patterns: &[String]) -> GlobSet {
    let (mut builder, valid_count) = crate::ingest::parse_globs(patterns);
    if valid_count == 0 {
        warn!("No valid include patterns configured — falling back to **/*.md");
        builder.add(Glob::new("**/*.md").unwrap());
    }
    builder.build().unwrap_or_else(|e| {
        error!(
            "Failed to build include globset: {} — falling back to **/*.md",
            e
        );
        let mut fallback = GlobSetBuilder::new();
        fallback.add(Glob::new("**/*.md").unwrap());
        fallback
            .build()
            .expect("hardcoded fallback glob '**/*.md' must compile")
    })
}

/// A relevance score as reported to a caller: four significant digits. The
/// full `f32`/`f64` precision is noise to a reader and costs bytes on every row
/// (and a widened `f32` such as 0.93 would print as 0.9300000071525574). Also
/// how a dedup refusal reports its `similarity` and `threshold`, here and in
/// `web.rs`.
pub(crate) fn round_score(score: f64) -> serde_json::Value {
    if !score.is_finite() || score == 0.0 {
        return serde_json::json!(score);
    }
    let magnitude = score.abs().log10().floor() as i32;
    let decimals = (3 - magnitude).max(0) as usize;
    let rounded = format!("{score:.decimals$}")
        .parse::<f64>()
        .unwrap_or(score);
    serde_json::json!(rounded)
}

/// The longest search snippet, in characters, before it is cut with `…`.
const SNIPPET_CHARS: usize = 800;

/// Builds a minimal `ResolvedConfig` for tests. Shared with `server.rs`'s test
/// module, which needs the same handler-construction pattern for the MCP
/// service without pulling in a real config file.
#[cfg(test)]
pub(crate) fn make_test_resolved_config(data_path: &std::path::Path) -> Arc<ResolvedConfig> {
    Arc::new(ResolvedConfig {
        source: crate::config::ResolvedSourceConfig {
            git_url: None,
            branch: "master".into(),
            data_path: Some(data_path.to_string_lossy().into_owned()),
            git_token_env: "GIT_PULL_TOKEN".into(),
        },
        indexing: crate::config::IndexingConfig::default(),
        frontmatter: crate::config::FrontmatterConfig::default(),
        chunking: crate::config::ChunkingConfig::default(),
        embedding: crate::config::ResolvedEmbeddingConfig {
            base_url: "http://localhost:8080/v1".into(),
            model: "test".into(),
            api_key: None,
            vector_size: 768,
            batch_size: 32,
            request_timeout_secs: 60,
            batch_concurrency: 4,
            query_prefix: String::new(),
            document_prefix: String::new(),
        },
        qdrant: crate::config::ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: "test".into(),
        },
        validation: crate::config::ValidationConfig::default(),
        webhook: crate::config::WebhookConfig::default(),
        mcp: crate::config::ResolvedMcpConfig::default(),
        rate_limit: crate::config::RateLimitConfig::default(),
        write: crate::config::WriteConfig::default(),
        search: crate::config::SearchConfig::default(),
        reranking: None,
        ui: crate::config::UiConfig::default(),
        lint: crate::config::LintConfig::default(),
        formatting: crate::config::FormattingConfig::off(),
        contradictions: crate::config::ContradictionsConfig::default(),
        contradictions_judge: None,
        provenance: Default::default(),
    })
}

/// An empty `SharedSchemaCache`, for tests that exercise a `KbSearchServer` but do
/// not care about schema content (e.g. instructions plumbing, path validation).
/// Tests that DO care build a real one from a temp dir's schema files —
/// see `make_write_test_server` below.
#[cfg(test)]
pub(crate) fn empty_test_schema_cache() -> crate::schema::SharedSchemaCache {
    Arc::new(RwLock::new(Arc::new(crate::schema::SchemaCache::default())))
}

/// An empty description overlay, for tests that exercise a `KbSearchServer` but
/// do not care about tool descriptions — `list_tools`/`get_tool` fall back to
/// whatever the router itself produced (`None`, since no `#[tool(...)]`
/// attribute carries one) when a tool has no overlay entry.
#[cfg(test)]
pub(crate) fn empty_test_description_overlay() -> Arc<RwLock<HashMap<String, String>>> {
    Arc::new(RwLock::new(HashMap::new()))
}

impl KbSearchServer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        embed_client: Arc<EmbedClient>,
        qdrant: Arc<QdrantStore>,
        collection: String,
        data_path: PathBuf,
        include_patterns: &[String],
        instructions: Arc<RwLock<String>>,
        config: crate::config::SharedConfig,
        schema_cache: crate::schema::SharedSchemaCache,
        rerank_client: Option<Arc<RerankClient>>,
        reindex_queue: Arc<crate::reindex::ReindexQueue>,
        description_overlay: Arc<RwLock<HashMap<String, String>>>,
    ) -> anyhow::Result<Self> {
        let canonical_data_path = data_path.canonicalize().with_context(|| {
            format!("Failed to canonicalize data path: {}", data_path.display())
        })?;
        Ok(Self {
            embed_client,
            qdrant,
            collection,
            canonical_data_path,
            // Compiled once from the config snapshot at construction time — a
            // `POST /admin/reload` that changes `indexing.include` does NOT change
            // what get_document accepts until a restart. See
            // `reload.rs`'s "indexing.include (MCP get_document path filter)" entry.
            include_patterns: Arc::new(build_include_globset(include_patterns)),
            instructions,
            config,
            schema_cache,
            rerank_client,
            state_db: Arc::new(tokio::sync::OnceCell::new()),
            reindex_queue,
            description_overlay,
        })
    }

    /// A fresh snapshot of the live config — a lock acquisition plus an `Arc`
    /// clone, mirroring `schema::load_shared`. Every tool call fetches its own
    /// snapshot here rather than reading a value captured at construction, so a
    /// `POST /admin/reload` swap is observed starting with the very next call.
    fn config(&self) -> Arc<ResolvedConfig> {
        crate::config::load_shared_config(&self.config)
    }

    /// A fresh `tool_router()` with every name in the live `mcp.disabled_tools`
    /// disabled (`ToolRouter::disable_route`) — the one place `list_tools`,
    /// `get_tool`, and `call_tool` all build their router from, so the three
    /// can never disagree about which tools are currently enabled. Reads
    /// `self.config()` fresh on every call, same live-reload contract as every
    /// other config consumer here: a `POST /admin/reload` that changes
    /// `mcp.disabled_tools` is observed starting with the very next request,
    /// no restart or metadata-refresh-tick wait required (`config::validate`
    /// has already rejected an unknown or all-disabling set, so every name
    /// reaching `disable_route` here is one of `descriptions::TOOL_NAMES`).
    fn enabled_tool_router(&self) -> ToolRouter<Self> {
        let mut router = Self::tool_router();
        for name in &self.config().mcp.disabled_tools {
            router.disable_route(name.clone());
        }
        router
    }

    /// Apply the live description overlay to one router-provided `Tool`,
    /// recovering from a poisoned lock the same way `get_info` does for
    /// `instructions`. A tool name absent from the overlay (should not
    /// happen — every tool in `tool_router()` is one of `descriptions::TOOL_NAMES`)
    /// is left with whatever the router itself produced, which is `None`
    /// since every `#[tool(...)]` attribute below carries no `description`.
    fn overlay_description(&self, mut tool: Tool) -> Tool {
        let overlay = self.description_overlay.read().unwrap_or_else(|poisoned| {
            warn!("Description overlay RwLock poisoned on read; using last value");
            poisoned.into_inner()
        });
        if let Some(desc) = overlay.get(tool.name.as_ref()) {
            tool.description = Some(std::borrow::Cow::Owned(desc.clone()));
        }
        tool
    }

    /// Apply the live per-instance `search` restrictions to one
    /// router-provided `Tool`'s `input_schema` (#286) — the schema
    /// counterpart to `overlay_description` above, called from the same two
    /// sites (`list_tools`/`get_tool`) so a disabled granularity or
    /// `heading_prefix` disappears from what a caller/model can even see,
    /// not just from prose it might skip: the `granularity` property's
    /// `enum` becomes the effective set and its `description` becomes
    /// `descriptions::granularity_description` for that set (the full
    /// per-value text; the tool description carries only
    /// `descriptions::granularity_summary`), and `heading_prefix` is removed while
    /// `chunking.heading_metadata` is off. Reads `self.config()` live, same
    /// fresh-snapshot-per-call contract as every other config consumer
    /// here. A no-op for every tool but `search`.
    ///
    /// `Tool::input_schema` is `Arc<JsonObject>`, shared with every OTHER
    /// concurrent caller of `list_tools`/`get_tool` until the next one — this
    /// clones the object before editing rather than mutating through the
    /// `Arc`, so one request's overlay can never leak into another's, or
    /// into the router's own cached `Tool` (see `ResolvedConfig::
    /// effective_granularities`'s doc comment for why all three consumers of
    /// that helper, this one included, must read a fresh snapshot rather
    /// than share mutable state).
    fn overlay_input_schema(&self, mut tool: Tool) -> Tool {
        if tool.name.as_ref() != "search" {
            return tool;
        }
        let config = self.config();
        let effective = config.effective_granularities();

        let mut schema: JsonObject = (*tool.input_schema).clone();
        if let Some(properties) = schema.get_mut("properties").and_then(|v| v.as_object_mut()) {
            if let Some(granularity) = properties
                .get_mut("granularity")
                .and_then(|v| v.as_object_mut())
            {
                // Strings only: an omitted granularity is expressed by leaving
                // the optional property out (`tool_schema::compact` drops its
                // `null` type arm too); the server still accepts an explicit
                // `null`, since the field is an `Option<String>`.
                let values: Vec<serde_json::Value> = effective
                    .iter()
                    .map(|g| serde_json::Value::String(g.as_str().to_string()))
                    .collect();
                granularity.insert("enum".to_string(), serde_json::Value::Array(values));
                granularity.insert(
                    "description".to_string(),
                    serde_json::Value::String(crate::descriptions::granularity_description(
                        &effective,
                    )),
                );
            }
            if !config.chunking.heading_metadata {
                properties.remove("heading_prefix");
            }
            // Properties whose static doc comments assume every granularity
            // and mode is on: rewritten, or removed when nothing enabled can
            // use them.
            for (name, description) in
                crate::descriptions::search_property_descriptions(&effective, config.search.hybrid)
            {
                match description {
                    Some(text) => {
                        if let Some(property) =
                            properties.get_mut(name).and_then(|v| v.as_object_mut())
                        {
                            property
                                .insert("description".to_string(), serde_json::Value::String(text));
                        }
                    }
                    None => {
                        properties.remove(name);
                    }
                }
            }
        }
        tool.input_schema = Arc::new(schema);
        tool
    }

    /// The document metadata index, opened on first use.
    async fn state_db(&self) -> anyhow::Result<&StateDb> {
        self.state_db
            .get_or_try_init(|| async {
                let path = self.config().state_db_path();
                StateDb::new(std::path::Path::new(&path))
                    .await
                    .with_context(|| format!("Failed to open state DB at {}", path))
            })
            .await
    }

    /// Build a `RetrievalDeps` bundle from this server's fields.
    fn deps(&self) -> RetrievalDeps<'_, EmbedClient, QdrantStore> {
        RetrievalDeps {
            embed_client: &self.embed_client,
            qdrant: &self.qdrant,
            collection: &self.collection,
            data_path: &self.canonical_data_path,
            include_patterns: &self.include_patterns,
            reranker: self
                .rerank_client
                .as_ref()
                .map(|c| c.as_ref() as &(dyn crate::rerank::Reranker + Send + Sync)),
        }
    }

    /// The `search` call's view of this server's state.
    fn search_ctx(&self) -> SearchCtx<'_> {
        SearchCtx {
            deps: self.deps(),
            config: &self.config,
            schema_cache: &self.schema_cache,
            state_db: &self.state_db,
            extra_conditions: Vec::new(),
        }
    }

    /// See [`SearchCtx::resolve_path_filter`].
    #[cfg(test)]
    async fn resolve_path_filter(
        &self,
        path_prefix: Option<&str>,
    ) -> Result<Option<crate::retrieval::PathFilter>, McpError> {
        self.search_ctx().resolve_path_filter(path_prefix).await
    }

    /// The router every `tools/list`, `tools/get` and `tools/call` is served
    /// from: the sum of the per-family routers each family file's
    /// `#[tool_router]` block generates. `ToolRouter` implements `Add`, so this
    /// is exactly the set of routes the single `#[tool_router]` impl block held
    /// before this file was split by tool family, and `ToolRouter::list_all`
    /// sorts by name, so `tools/list` reports them in the same order.
    pub(crate) fn tool_router() -> ToolRouter<Self> {
        Self::search_router()
            + Self::get_document_router()
            + Self::write_router()
            + Self::schema_router()
    }
}

#[tool_handler]
impl ServerHandler for KbSearchServer {
    fn get_info(&self) -> ServerConfig {
        let instructions = self
            .instructions
            .read()
            .unwrap_or_else(|poisoned| {
                warn!("Instructions RwLock poisoned on read; using last value");
                poisoned.into_inner()
            })
            .clone();
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION"),
            ))
            .with_instructions(instructions)
    }

    // Hand-written rather than left for `#[tool_handler]` to generate: the
    // macro regenerates the tool list from `Self::tool_router()` (and its
    // compile-time `#[tool(...)]` attributes) on every call, so a description
    // that needs to change at runtime — per `descriptions.rs`'s whole point —
    // cannot be baked into that attribute. All three methods below build their
    // router from `self.enabled_tool_router()` rather than the bare
    // `Self::tool_router()`, so a name in the live `mcp.disabled_tools` is
    // hidden from `list_tools`, absent from `get_tool`, and refused by
    // `call_tool` — see that method's doc comment for why `#[tool_handler]`
    // does not generate a conflicting one. `list_tools`/`get_tool` additionally
    // apply this server's live `description_overlay`, then the live
    // per-instance schema restrictions (`overlay_input_schema`, #286), then the
    // stateless `tool_schema::self_contained` rewrite that strips `$ref`/
    // `$defs` and boolean subschemas for llama.cpp-backed clients (#288), on
    // top of the router's own `Tool` entries.
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let tools = self
            .enabled_tool_router()
            .list_all()
            .into_iter()
            .map(|tool| self.overlay_description(tool))
            .map(|tool| self.overlay_input_schema(tool))
            .map(tool_schema::self_contained)
            .collect();
        Ok(ListToolsResult::with_all_items(tools))
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        self.enabled_tool_router()
            .get(name)
            .cloned()
            .map(|tool| self.overlay_description(tool))
            .map(|tool| self.overlay_input_schema(tool))
            .map(tool_schema::self_contained)
    }

    /// Hand-written, rather than left for `#[tool_handler]` to generate
    /// (`rmcp_macros::tool_handler` only fills in a `call_tool` when the impl
    /// block does not already define one, so this is not a duplicate): the
    /// generated version dispatches through the bare `Self::tool_router()`,
    /// which knows nothing about `mcp.disabled_tools`. This does exactly what
    /// the macro would — `ToolCallContext::new` plus `.call(tcc).await` — but
    /// through `self.enabled_tool_router()`, so a disabled tool's `call`
    /// reaches `ToolRouter::call`'s own disabled-route check and comes back as
    /// `invalid_params("tool not found")`, identical to an unknown tool name.
    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let router = self.enabled_tool_router();
        let tcc = ToolCallContext::new(self, request, context);
        router.call(tcc).await
    }
}
