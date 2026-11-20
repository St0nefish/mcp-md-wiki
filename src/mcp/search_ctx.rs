use super::search::*;
use super::*;
use std::sync::Arc;

use rmcp::ErrorData as McpError;
use tracing::{debug, error};

use crate::{
    config::{Granularity, ResolvedConfig},
    embed::EmbedClient,
    qdrant::QdrantStore,
    retrieval::{self, DocumentIndexDeps, RetrievalDeps, SearchFilters, SearchOptions},
    schema::SchemaCache,
    state::{DocumentIndex, FieldFilter, StateDb},
};

/// The pieces of server state a `search` call reads, borrowed for one call.
///
/// `KbSearchServer` (the MCP tool) and `web.rs`'s `UiState` (the REST routes)
/// each build one, so both transports run the same validation, filter
/// lowering, vocabulary check and retrieval calls rather than two copies of
/// them. The methods return `McpError` because that is the shape the MCP tool
/// emits; the web adapter maps `invalid_params` to a 4xx and everything else
/// to a 500.
pub(crate) struct SearchCtx<'a> {
    pub(crate) deps: RetrievalDeps<'a, EmbedClient, QdrantStore>,
    pub(crate) config: &'a crate::config::SharedConfig,
    pub(crate) schema_cache: &'a crate::schema::SharedSchemaCache,
    pub(crate) state_db: &'a tokio::sync::OnceCell<StateDb>,
    /// Qdrant conditions ANDed onto the lowered `filters` in query mode. The
    /// MCP tool passes none; the REST `domain`/`type`/`tags` sugar lands here.
    pub(crate) extra_conditions: Vec<qdrant_client::qdrant::Condition>,
}

impl<'a> SearchCtx<'a> {
    fn config(&self) -> Arc<ResolvedConfig> {
        crate::config::load_shared_config(self.config)
    }

    fn deps(&self) -> &RetrievalDeps<'a, EmbedClient, QdrantStore> {
        &self.deps
    }

    /// The fields ingest derives (the root schema's `derived:` block, #345), which
    /// a document row leaves out of its `frontmatter` unless asked for by name.
    fn derived_fields(&self) -> Vec<String> {
        crate::schema::load_shared(self.schema_cache)
            .root()
            .derived_field_names()
            .map(str::to_string)
            .collect()
    }

    /// The document metadata index, opened on first use.
    pub(crate) async fn state_db(&self) -> anyhow::Result<&StateDb> {
        self.state_db
            .get_or_try_init(|| async {
                let path = self.config().state_db_path();
                StateDb::new(std::path::Path::new(&path))
                    .await
                    .with_context(|| format!("Failed to open state DB at {}", path))
            })
            .await
    }

    /// Query-mode Qdrant conditions: the lowered `filters` and `heading_prefix`
    /// ([`build_query_conditions`]) plus [`Self::extra_conditions`].
    pub(crate) fn query_conditions(
        &self,
        params: &SearchParams,
        config: &ResolvedConfig,
        schemas: &SchemaCache,
    ) -> Result<Vec<qdrant_client::qdrant::Condition>, McpError> {
        let mut conditions = build_query_conditions(params, config, schemas)?;
        conditions.extend(self.extra_conditions.iter().cloned());
        Ok(conditions)
    }

    /// Refuse a `search` filter that can only ever match nothing, listing what
    /// would match instead — the teach-on-error replacement for enumerating
    /// every vocabulary in the server instructions. Applies to every
    /// granularity, so the metadata-index (no query) and Qdrant (query)
    /// backends share it.
    ///
    /// - A field is unknown when no schema scope declares it
    ///   (`SchemaCache::declared_field_paths`), it is not a built-in filter key
    ///   ([`builtin_filter_field`]), and no document carries it.
    /// - A value is refused when the field has a closed value set across the
    ///   governing scopes (`SchemaCache::filter_closed_values`, narrowed by
    ///   `path_prefix` as [`retrieval::normalize_path_needle`] reads it, the
    ///   needle `search` itself matches on), the value is outside it, and no
    ///   document uses it either — so a refusal never hides a match, whichever
    ///   scope the matching document sits in.
    /// - A filter is refused only when no document could match it: an `all_of`
    ///   as soon as one of its values is refused (no document can carry it),
    ///   but a scalar or array filter is an any-of and still matches through
    ///   its other values, so it is refused only when every value is.
    ///
    /// Fails open: when the metadata index cannot be opened or queried, the
    /// filter runs as given.
    pub(crate) async fn check_filter_vocabulary(
        &self,
        params: &SearchParams,
    ) -> Result<(), McpError> {
        if params.filters.as_ref().is_none_or(|f| f.0.is_empty()) {
            return Ok(());
        }
        let filters = parse_filters(&params.filters)?;
        let Ok(index) = self.state_db().await else {
            return Ok(());
        };
        let schemas = crate::schema::load_shared(self.schema_cache);
        let config = self.config();
        let declared = schemas.declared_field_paths();
        let path_needle = retrieval::normalize_path_needle(params.path_prefix.as_deref());

        // A correlated list filter (#349) is checked condition by condition: each
        // sub-field (`npcs[].role`) is a field of its own, refused the same ways.
        let mut checks: Vec<(&String, &FieldFilter, Option<&String>)> = Vec::new();
        for (field, filter) in &filters {
            match filter {
                FieldFilter::Elem(conditions) => checks.extend(
                    conditions
                        .iter()
                        .map(|(sub, condition)| (sub, condition, Some(field))),
                ),
                _ => checks.push((field, filter, None)),
            }
        }

        for (field, filter, list) in checks {
            let known = declared.contains(field)
                || (list.is_none() && builtin_filter_field(field, &config, &schemas))
                || index.field_in_use(field).await.unwrap_or(true);
            if !known {
                let mut names: std::collections::BTreeSet<String> = declared.clone();
                if list.is_none() {
                    names.extend(
                        config
                            .effective_indexed_fields()
                            .into_iter()
                            .chain(schemas.root().derived_field_names().map(str::to_string)),
                    );
                }
                if let Ok(in_use) = index.fields_in_use(None, 500).await {
                    names.extend(in_use.into_iter().map(|(name, _)| name));
                }
                names.remove("file_path");
                if let Some(list) = list {
                    let prefix = format!("{list}.");
                    names.retain(|name| name.starts_with(&prefix));
                }
                let names: Vec<String> = names
                    .iter()
                    .map(|n| crate::server::sanitize_facet_value(n))
                    .collect();
                let message = match list {
                    Some(list) => format!(
                        "filter '{}': unknown sub-field '{}'; fields of {}: {}",
                        crate::server::sanitize_facet_value(list),
                        crate::server::sanitize_facet_value(field),
                        crate::server::sanitize_facet_value(list),
                        if names.is_empty() {
                            "none".to_string()
                        } else {
                            capped_list(&names)
                        }
                    ),
                    None => format!(
                        "unknown filter field '{}'; filterable fields: {}",
                        crate::server::sanitize_facet_value(field),
                        capped_list(&names)
                    ),
                };
                return Err(McpError::invalid_params(
                    message,
                    Some(serde_json::json!({ "filterable_fields": capped_vec(&names) })),
                ));
            }

            let (values, needs_every_value) = match filter {
                FieldFilter::AnyOf(values) => (values, false),
                FieldFilter::AllOf(values) => (values, true),
                FieldFilter::Range { .. } => {
                    // A range compares the stored number, which a declared type with
                    // no numeric mapping (`text`, `enum`, …) never stores (#349).
                    // Refused only when every scope that types the field gives it
                    // such a type; an undeclared field passes as before.
                    let types = schemas.declared_types(field);
                    let numeric = |ty: &crate::schema::FieldType| {
                        ty.kind().numeric != crate::schema::NumericValue::None
                    };
                    if !types.is_empty() && !types.iter().any(numeric) {
                        let names: Vec<&str> = types.iter().map(|ty| ty.name()).collect();
                        return Err(McpError::invalid_params(
                            format!(
                                "filter '{}': a range needs a numeric field, but it is declared \
                                 `{}`, which stores no number; match it by value, or declare \
                                 it `integer` or `number` (update_schema) to range over it",
                                crate::server::sanitize_facet_value(field),
                                names.join("`/`")
                            ),
                            Some(serde_json::json!({ "field": field, "declared_type": names })),
                        ));
                    }
                    continue;
                }
                FieldFilter::Elem(_) => continue,
            };
            let Some(allowed) = schemas.filter_closed_values(field, path_needle) else {
                continue;
            };
            let outside: Vec<String> = values
                .iter()
                .filter(|v| !allowed.contains(*v))
                .cloned()
                .collect();
            if outside.is_empty() {
                continue;
            }
            let Ok(present) = index.values_present(field, &outside).await else {
                continue;
            };
            let refused: Vec<&String> = outside.iter().filter(|v| !present.contains(*v)).collect();
            // `all_of` needs a document to carry every value, so one that nothing can
            // match sinks the filter; an any-of still matches through the rest of its
            // values and is empty only when none of them could match.
            let matches_nothing = if needs_every_value {
                !refused.is_empty()
            } else {
                !refused.is_empty() && refused.len() == values.len()
            };
            if matches_nothing {
                let mut seen = std::collections::HashSet::new();
                let refused: Vec<String> = refused
                    .iter()
                    .map(|v| format!("'{}'", crate::server::sanitize_facet_value(v)))
                    .filter(|v| seen.insert(v.clone()))
                    .collect();
                let allowed: Vec<String> = allowed
                    .iter()
                    .map(|v| crate::server::sanitize_facet_value(v))
                    .collect();
                let verb = if refused.len() == 1 {
                    "is not an allowed value"
                } else {
                    "are not allowed values"
                };
                return Err(McpError::invalid_params(
                    format!(
                        "filter '{}': {} {verb}; allowed: {}",
                        crate::server::sanitize_facet_value(field),
                        capped_list(&refused),
                        capped_list(&allowed)
                    ),
                    Some(serde_json::json!({ "field": field, "allowed": capped_vec(&allowed) })),
                ));
            }
        }
        Ok(())
    }

    /// Resolve a `path_prefix` needle to the documents it actually matches (#182).
    ///
    /// The metadata index is the single authority for that question, so both query
    /// modes route through here and enumeration applies the identical `LIKE` inline
    /// (`state::StateDb::push_where`) — one needle can only ever mean one thing.
    /// `None` in gives `None` out: no filter, matching everything.
    pub(crate) async fn resolve_path_filter(
        &self,
        path_prefix: Option<&str>,
    ) -> Result<Option<retrieval::PathFilter>, McpError> {
        let Some(needle) = retrieval::normalize_path_needle(path_prefix) else {
            return Ok(None);
        };
        let index = self.state_db().await.map_err(|e| {
            error!("search could not open the metadata index: {:#}", e);
            McpError::internal_error(format!("Document index unavailable: {}", e), None)
        })?;
        let matches = index
            .paths_matching(needle, retrieval::PATH_FILTER_MAX_PATHS)
            .await
            .map_err(|e| {
                error!("search could not resolve path_prefix '{}': {:#}", needle, e);
                McpError::internal_error(format!("path_prefix lookup failed: {}", e), None)
            })?;
        Ok(Some(matches.into()))
    }

    /// query+chunk: the original `search` tool's behavior, unchanged output shape.
    pub(crate) async fn search_chunks(
        &self,
        params: &SearchParams,
    ) -> Result<CallToolResult, McpError> {
        let query = params.query.as_deref().unwrap_or_default();

        validate_path_prefix(&params.path_prefix)?;
        // #132 (audit follow-up): `fields` selects per-result frontmatter fields
        // from the document metadata index — a chunk result is hydrated straight
        // from the Qdrant chunk payload and never joins that index, so `fields`
        // has nothing to draw from here. `search_grouped`/`build_document_query`
        // (document granularity, query or enumeration) are its only real
        // consumers; reject explicitly here rather than silently drop it (the
        // mirror image of `explain` at document granularity).
        if params.fields.is_some() {
            return Err(fields_rejection(
                Granularity::Chunk,
                &self.config().effective_granularities(),
            ));
        }
        let schemas = crate::schema::load_shared(self.schema_cache);
        // Fetched once per call so every field below — including
        // reranking.candidate_limit — reflects the same live snapshot, rather than
        // racing a concurrent `POST /admin/reload` mid-request.
        let config = self.config();
        let conditions = self.query_conditions(params, &config, &schemas)?;

        let limit = resolve_limit(
            params.limit,
            config.search.default_limit,
            config.search.max_limit,
        );

        let filters = SearchFilters { conditions };

        let modified_after = params
            .modified_after
            .as_deref()
            .map(parse_date_to_timestamp)
            .transpose()
            .map_err(|e| McpError::invalid_params(e, None))?;
        let modified_before = params
            .modified_before
            .as_deref()
            .map(parse_date_to_timestamp)
            .transpose()
            .map_err(|e| McpError::invalid_params(e, None))?;
        let explain = params.explain.unwrap_or(false);
        let path_filter = self
            .resolve_path_filter(params.path_prefix.as_deref())
            .await?;
        let opts = SearchOptions {
            limit,
            min_score: params.min_score.or(config.search.min_score),
            hybrid: config.search.hybrid,
            rrf_candidates: config.search.rrf_candidates as u64,
            // Config-enabled AND confirmed available on this server right now —
            // see `status::IndexStatus::phrase_matching_available`'s doc comment
            // for why an unconfirmed index must not attempt a phrase arm.
            phrase: config.search.phrase && crate::status::INDEX_STATUS.phrase_matching_available(),
            explain,
            modified_after,
            modified_before,
            path_filter,
            rerank_candidate_limit: config.reranking.as_ref().map(|r| r.candidate_limit as u64),
            diversity_max_per_document: config.search.diversity_max_per_document,
        };

        // The `explain` mode label must describe the query that actually ran, not
        // the config that permitted it. An enabled phrase arm only fires when the
        // query carries a quoted span, and with `hybrid` off a phrase query is
        // still a two-arm RRF fusion — reporting that as "dense cosine" would be a
        // lie in exactly the situation someone turned `explain` on to diagnose.
        let phrase_arm_ran = opts.phrase && !retrieval::extract_phrases(query).1.is_empty();

        // #224: `offset` pages over the already fused/reranked/diversity-capped
        // ranking — see `retrieval::search_paged`'s doc comment for the funnel
        // placement and the depth bound `offset_truncated` reports against.
        let offset = params.offset.unwrap_or(0);
        let outcome = retrieval::search_paged(self.deps(), query, &filters, &opts, offset)
            .await
            .map_err(|e| match e {
                retrieval::SearchError::Embed(err) => {
                    error!("Embedding query failed: {:#}", err);
                    McpError::internal_error("Failed to generate query embedding".to_string(), None)
                }
                retrieval::SearchError::Search(err) => {
                    error!("Qdrant search failed: {:#}", err);
                    McpError::internal_error("Search query failed".to_string(), None)
                }
                retrieval::SearchError::Document(err) => {
                    error!("Document metadata lookup failed: {:#}", err);
                    McpError::internal_error("Document metadata lookup failed".to_string(), None)
                }
            })?;
        let path_prefix_truncated = outcome.path_prefix_truncated;
        let offset_truncated = outcome.offset_truncated;
        let results = outcome.results;

        debug!(
            result_count = results.len(),
            path_prefix_truncated, offset_truncated, "search returned results"
        );

        let mode = match (config.search.hybrid, phrase_arm_ran) {
            (true, true) => "hybrid RRF + phrase",
            (true, false) => "hybrid RRF",
            (false, true) => "dense + phrase RRF",
            (false, false) => "dense cosine",
        };

        let mut response = build_chunk_search_payload(
            &results,
            self.deps.data_path,
            explain,
            mode,
            path_prefix_truncated,
            offset_truncated,
        );
        if params.heading_prefix.is_some() {
            annotate_heading_results(&mut response, heading_results_indexing_note());
        }
        Ok(CallToolResult::structured(response))
    }

    /// query+document: Qdrant grouped by `file_path`, collapsed to each document's
    /// best-scoring chunk, hydrated with document-level metadata from `StateDb`. New
    /// combination — there is no output shape to preserve, only the two the design
    /// specifies: the document shape `search_enumerate` returns, plus a per-document
    /// `score`, and no `total`/`has_more` (grouped vector search cannot back either).
    pub(crate) async fn search_grouped(
        &self,
        params: &SearchParams,
    ) -> Result<CallToolResult, McpError> {
        let query = params.query.as_deref().unwrap_or_default();

        // #132: `explain` produces a per-result score breakdown (dense/sparse/
        // phrase/pre-rerank) from the per-arm scores `search_chunks` collects on
        // each chunk `SearchResult`. This path collapses every document to its
        // best-scoring chunk via Qdrant's server-side grouping (see this
        // function's doc comment) with no explain-mode fusion query behind it —
        // there is no per-arm breakdown to attach, so `GroupedDocument` carries
        // none. Implementing the real thing would mean threading an `explain`
        // flag through `RetrievalStore::search_grouped` and adding a
        // client-side-fused, per-arm-scored grouped-query path in `qdrant.rs`
        // mirroring `hybrid_search_explain` — real work, out of scope for this
        // fix. Reject explicitly rather than ignore the flag silently:
        // a caller that turned `explain` on to diagnose a confusing document
        // ranking deserves an error telling them so, not a response that quietly
        // drops the one thing they asked for.
        if params.explain == Some(true) {
            return Err(explain_rejection(
                Granularity::Document,
                &self.config().effective_granularities(),
            ));
        }

        validate_path_prefix(&params.path_prefix)?;
        validate_fields_count(&params.fields)?;
        let schemas = crate::schema::load_shared(self.schema_cache);
        let config = self.config();
        let conditions = self.query_conditions(params, &config, &schemas)?;

        let limit = resolve_limit(
            params.limit,
            config.search.default_limit,
            config.search.max_limit,
        );

        let filters = SearchFilters { conditions };

        let modified_after = params
            .modified_after
            .as_deref()
            .map(parse_date_to_timestamp)
            .transpose()
            .map_err(|e| McpError::invalid_params(e, None))?;
        let modified_before = params
            .modified_before
            .as_deref()
            .map(parse_date_to_timestamp)
            .transpose()
            .map_err(|e| McpError::invalid_params(e, None))?;

        let path_filter = self
            .resolve_path_filter(params.path_prefix.as_deref())
            .await?;
        let opts = SearchOptions {
            limit,
            min_score: params.min_score.or(config.search.min_score),
            // Grouped queries share `search_chunks`'s dense/sparse/phrase arms
            // (see `retrieval::search_grouped`'s doc comment) — a query only the
            // sparse or phrase arm can find must be just as retrievable at
            // document granularity as at chunk granularity. Reranking and
            // diversity still don't apply here: there is one row per document
            // already, so per-document diversity is a no-op and there is no
            // per-chunk candidate pool for a cross-encoder to rerank.
            hybrid: config.search.hybrid,
            rrf_candidates: config.search.rrf_candidates as u64,
            phrase: config.search.phrase && crate::status::INDEX_STATUS.phrase_matching_available(),
            // Always false: `explain: true` is rejected above (#132) before this
            // point is reached, and `retrieval::search_grouped` never reads this
            // field regardless — grouped results carry no per-arm score to
            // explain in the first place. `false` here (rather than
            // `params.explain.unwrap_or(false)`) says so plainly instead of
            // leaving a dead-looking reference to a value that can only be
            // `None`/`Some(false)` by now.
            explain: false,
            modified_after,
            modified_before,
            path_filter,
            rerank_candidate_limit: None,
            diversity_max_per_document: None,
        };

        let index = self.state_db().await.map_err(|e| {
            error!(
                "search (query+document) could not open the metadata index: {:#}",
                e
            );
            McpError::internal_error(format!("Document index unavailable: {}", e), None)
        })?;

        // #224: same paging entry point and depth-bound reasoning as
        // `search_chunks` — see `retrieval::search_grouped`'s doc comment.
        let offset = params.offset.unwrap_or(0);
        let outcome = retrieval::search_grouped(
            self.deps(),
            index,
            query,
            &filters,
            &opts,
            params.fields.as_deref(),
            offset,
        )
        .await
        .map_err(|e| match e {
            retrieval::SearchError::Embed(err) => {
                error!("Embedding query failed: {:#}", err);
                McpError::internal_error("Failed to generate query embedding".to_string(), None)
            }
            retrieval::SearchError::Search(err) => {
                error!("Qdrant grouped search failed: {:#}", err);
                McpError::internal_error("Search query failed".to_string(), None)
            }
            retrieval::SearchError::Document(err) => {
                error!("Document metadata lookup failed: {:#}", err);
                McpError::internal_error("Document metadata lookup failed".to_string(), None)
            }
        })?;
        let path_prefix_truncated = outcome.path_prefix_truncated;
        let offset_truncated = outcome.offset_truncated;
        let documents = outcome.documents;

        let mut response = build_grouped_search_payload(
            &documents,
            params.fields.as_deref(),
            &self.derived_fields(),
            path_prefix_truncated,
            offset_truncated,
        );
        if params.heading_prefix.is_some() {
            annotate_heading_results(&mut response, heading_results_indexing_note());
        }
        Ok(CallToolResult::structured(response))
    }

    /// query+section: Qdrant grouped by `section_key` (#286), collapsed to
    /// each section's best-scoring chunk, returned as a path with no text — the
    /// `search` tool's `section` granularity. Requires
    /// `chunking.heading_metadata` (the flag that gates writing `section_key` at
    /// all — see `ingest.rs`) and, like `search_grouped`, rejects `explain`
    /// (no per-arm breakdown to report) and `fields` (no document-metadata join,
    /// same reasoning `search_chunks` gives for the identical rejection).
    pub(crate) async fn search_sections(
        &self,
        params: &SearchParams,
    ) -> Result<CallToolResult, McpError> {
        let query = params.query.as_deref().unwrap_or_default();
        let config = self.config();

        if !config.chunking.heading_metadata {
            return Err(McpError::invalid_params(
                HEADING_METADATA_OFF_SECTION.to_string(),
                None,
            ));
        }
        if params.explain == Some(true) {
            return Err(explain_rejection(
                Granularity::Section,
                &config.effective_granularities(),
            ));
        }
        if params.fields.is_some() {
            return Err(fields_rejection(
                Granularity::Section,
                &config.effective_granularities(),
            ));
        }

        validate_path_prefix(&params.path_prefix)?;
        let schemas = crate::schema::load_shared(self.schema_cache);
        let conditions = self.query_conditions(params, &config, &schemas)?;

        let limit = resolve_limit(
            params.limit,
            config.search.default_limit,
            config.search.max_limit,
        );

        let filters = SearchFilters { conditions };

        let modified_after = params
            .modified_after
            .as_deref()
            .map(parse_date_to_timestamp)
            .transpose()
            .map_err(|e| McpError::invalid_params(e, None))?;
        let modified_before = params
            .modified_before
            .as_deref()
            .map(parse_date_to_timestamp)
            .transpose()
            .map_err(|e| McpError::invalid_params(e, None))?;

        let path_filter = self
            .resolve_path_filter(params.path_prefix.as_deref())
            .await?;
        let opts = SearchOptions {
            limit,
            min_score: params.min_score.or(config.search.min_score),
            hybrid: config.search.hybrid,
            rrf_candidates: config.search.rrf_candidates as u64,
            phrase: config.search.phrase && crate::status::INDEX_STATUS.phrase_matching_available(),
            explain: false,
            modified_after,
            modified_before,
            path_filter,
            rerank_candidate_limit: None,
            diversity_max_per_document: None,
        };

        let offset = params.offset.unwrap_or(0);
        let outcome = retrieval::search_sections(self.deps(), query, &filters, &opts, offset)
            .await
            .map_err(|e| match e {
                retrieval::SearchError::Embed(err) => {
                    error!("Embedding query failed: {:#}", err);
                    McpError::internal_error("Failed to generate query embedding".to_string(), None)
                }
                retrieval::SearchError::Search(err) => {
                    error!("Qdrant grouped search failed: {:#}", err);
                    McpError::internal_error("Search query failed".to_string(), None)
                }
                retrieval::SearchError::Document(err) => {
                    error!("Document metadata lookup failed: {:#}", err);
                    McpError::internal_error("Document metadata lookup failed".to_string(), None)
                }
            })?;
        let path_prefix_truncated = outcome.path_prefix_truncated;
        let offset_truncated = outcome.offset_truncated;
        let sections = outcome.sections;

        let mut response =
            build_section_search_payload(&sections, path_prefix_truncated, offset_truncated);
        annotate_heading_results(&mut response, heading_results_indexing_note());
        Ok(CallToolResult::structured(response))
    }

    /// document, no query: the former `list_documents` tool's behavior, unchanged
    /// output shape (`{total, returned, offset, has_more, documents}`).
    pub(crate) async fn search_enumerate(
        &self,
        params: &SearchParams,
    ) -> Result<CallToolResult, McpError> {
        // Enumeration reads the state DB, which holds no heading data, so
        // `heading_prefix` cannot be honored here. Validate it the same way
        // query mode does first (so an empty list / the flag being off get
        // their own message), then reject rather than silently ignore it —
        // ignoring it would return every document as if the filter matched.
        if params.heading_prefix.is_some() {
            heading_prefix_condition(&params.heading_prefix, &self.config())?;
            return Err(McpError::invalid_params(
                "heading_prefix requires a query — a search without one lists whole \
                 documents, which cannot be filtered by heading"
                    .to_string(),
                None,
            ));
        }
        let query = build_document_query(params)?;

        let index = self.state_db().await.map_err(|e| {
            error!(
                "search (enumerate) could not open the metadata index: {:#}",
                e
            );
            McpError::internal_error(format!("Document index unavailable: {}", e), None)
        })?;

        let result = retrieval::list_documents(&DocumentIndexDeps { index }, &query)
            .await
            .map_err(|e| {
                error!("search (enumerate) failed: {:#}", e);
                McpError::internal_error(format!("Failed to list documents: {}", e), None)
            })?;

        // `total` always; `has_more` only when another page exists. A row
        // carries `mtime` only when the listing is ordered by it.
        let fields = params.fields.as_deref();
        let derived = self.derived_fields();
        let with_mtime = query.order_by == crate::state::OrderBy::Mtime;
        let mut response = serde_json::json!({
            "total": result.total,
            "returned": result.documents.len(),
            "documents": result
                .documents
                .iter()
                .map(|d| {
                    let mut row = document_row(d, fields, &derived);
                    if with_mtime {
                        row.insert("mtime".into(), serde_json::json!(d.mtime));
                    }
                    serde_json::Value::Object(row)
                })
                .collect::<Vec<_>>(),
        });
        if result.has_more(query.offset) {
            response["has_more"] = serde_json::json!(true);
        }
        Ok(CallToolResult::structured(response))
    }
}
