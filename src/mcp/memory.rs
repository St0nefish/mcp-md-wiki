//! The memory tool group (`memory.enabled`): `memory_create`, `memory_search`,
//! `memory_get`, `memory_update`, `memory_forget` and `memory_recall`. Thin adapters
//! over the existing search and write pipelines: a memory is an ordinary markdown
//! document under `memory.root`, validated against the schema the binary supplies
//! for that area, so every access rule, index and history feature applies to it
//! unchanged. The rules the tools add (ids, limits, statuses, the recall budget) are
//! in `crate::memory::logic`.

use std::collections::HashMap;

use rmcp::{
    ErrorData as McpError, handler::server::wrapper::Parameters, model::CallToolResult, schemars,
    tool, tool_router,
};
use serde_json::{Value, json};

use super::search::{SearchFiltersInput, SearchParams};
use super::write_document::{
    batch_write_error_to_mcp_error, create_edit_error_to_mcp_error, credential_error,
};
use super::*;
use crate::memory::{self, logic};
use crate::policy::{Op, Reach};
use crate::retrieval;
use crate::state::{DocumentIndex, DocumentQuery, FieldFilter};
use crate::write::{self, BatchWriteRequest, DocChange, WriteRequest};

/// How many near neighbours `memory_create` looks at before dropping the dead ones.
const DUPLICATE_CANDIDATES: usize = 8;
const DEFAULT_SEARCH_LIMIT: u64 = 10;
const MAX_SEARCH_LIMIT: u64 = 50;
const DEFAULT_RECALL_ITEMS: usize = 8;
const MAX_RECALL_ITEMS: usize = 20;
const DEFAULT_RECALL_CHARS: usize = 1500;
const MIN_RECALL_CHARS: usize = 200;
const MAX_RECALL_CHARS: usize = 8000;
/// How many supersede steps `memory_get` follows in either direction.
const MAX_CHAIN: usize = 10;

/// Parameters for `memory_create`.
#[derive(Debug, Default, serde::Deserialize, schemars::JsonSchema)]
pub struct MemoryCreateParams {
    /// Short one-line headline.
    pub title: String,
    /// The memory: one self-contained statement, up to 4000 bytes. Write what will
    /// still make sense on its own later.
    pub body: String,
    /// What kind of memory this is.
    #[serde(rename = "type")]
    #[schemars(extend("enum" = ["preference", "fact", "decision", "finding", "procedure"]))]
    pub ty: String,
    /// Labels, up to 16.
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    /// Folder that groups memories; the server default when omitted.
    #[serde(default)]
    pub area: Option<String>,
    /// Where this came from, when it came from a page.
    #[serde(default)]
    pub source_url: Option<String>,
    /// `id` of a live memory this one replaces. It is marked `superseded` in the
    /// same change.
    #[serde(default)]
    pub supersedes: Option<String>,
    /// What to do when a live memory says nearly the same thing: `reject`
    /// (default) saves nothing and returns the matches, `supersede` replaces the
    /// closest, `keep_both` saves anyway.
    #[serde(default)]
    #[schemars(extend("enum" = ["reject", "supersede", "keep_both"]))]
    pub on_duplicate: Option<String>,
}

/// Parameters for `memory_search`.
#[derive(Debug, Default, serde::Deserialize, schemars::JsonSchema)]
pub struct MemorySearchParams {
    /// What to look for, in plain words.
    pub query: String,
    /// Only these types.
    #[serde(default, rename = "type")]
    pub ty: Option<Vec<String>>,
    /// Only memories carrying every one of these tags.
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    /// Only these statuses. Default: `proposed` and `confirmed`.
    #[serde(default)]
    pub status: Option<Vec<String>>,
    /// Only this area.
    #[serde(default)]
    pub area: Option<String>,
    /// How many results (default 10, at most 50).
    #[serde(default)]
    pub limit: Option<u64>,
}

/// Parameters for `memory_get`.
#[derive(Debug, Default, serde::Deserialize, schemars::JsonSchema)]
pub struct MemoryGetParams {
    /// The memory's `id`.
    pub id: String,
}

/// Parameters for `memory_update`.
#[derive(Debug, Default, serde::Deserialize, schemars::JsonSchema)]
pub struct MemoryUpdateParams {
    /// The memory's `id`.
    pub id: String,
    /// New headline.
    #[serde(default)]
    pub title: Option<String>,
    /// New text, replacing the old.
    #[serde(default)]
    pub body: Option<String>,
    /// New tags, replacing the old; an empty list clears them.
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    /// New source; an empty string clears it.
    #[serde(default)]
    pub source_url: Option<String>,
}

/// Parameters for `memory_forget`.
#[derive(Debug, Default, serde::Deserialize, schemars::JsonSchema)]
pub struct MemoryForgetParams {
    /// The memory's `id`.
    pub id: String,
}

/// Parameters for `memory_recall`.
#[derive(Debug, Default, serde::Deserialize, schemars::JsonSchema)]
pub struct MemoryRecallParams {
    /// What the conversation is about, in plain words.
    pub query: String,
    /// Most memories in the block (default 8, at most 20).
    #[serde(default)]
    pub max_items: Option<usize>,
    /// Most characters in the block (default 1500, from 200 to 8000).
    #[serde(default)]
    pub max_chars: Option<usize>,
    /// Only this area.
    #[serde(default)]
    pub area: Option<String>,
    /// Also use `proposed` memories, which no one has confirmed yet.
    #[serde(default)]
    pub include_proposed: Option<bool>,
}

fn invalid(msg: impl Into<String>) -> McpError {
    McpError::invalid_params(msg.into(), None)
}

fn no_such_memory() -> McpError {
    invalid("no memory with that id")
}

/// What `memory_search` and `memory_recall` ask the index.
struct MemoryQuery<'a> {
    query: &'a str,
    area: Option<&'a str>,
    types: Option<&'a [String]>,
    tags: Option<&'a [String]>,
    statuses: &'a [String],
    limit: u64,
}

/// A stored memory located by id.
pub(super) struct Found {
    path: String,
    fm: HashMap<String, Value>,
}

impl Found {
    pub(super) fn str(&self, key: &str) -> Option<&str> {
        self.fm.get(key).and_then(Value::as_str)
    }

    pub(super) fn id(&self) -> &str {
        self.str("id").unwrap_or_default()
    }

    fn is_live(&self) -> bool {
        self.str("status")
            .is_some_and(|s| logic::LIVE_STATUSES.contains(&s))
    }
}

/// The fields of a memory a caller is shown, omitting the ones it does not have.
fn record(path: &str, fm: &HashMap<String, Value>) -> serde_json::Map<String, Value> {
    let mut out = serde_json::Map::new();
    out.insert("path".into(), json!(path));
    for key in [
        "id",
        "title",
        "type",
        "status",
        "tags",
        "created",
        "updated",
        "source_client",
        "source_url",
        "supersedes",
    ] {
        if let Some(v) = fm.get(key).filter(|v| !v.is_null()) {
            out.insert(key.into(), v.clone());
        }
    }
    out
}

/// The memories in a document-granularity `search` response, in the order it ranked
/// them. A row without a path or frontmatter is not a memory and is skipped.
pub(super) fn memory_rows(response: &Value) -> Vec<(f64, Found)> {
    let mut out = Vec::new();
    for row in response["documents"].as_array().into_iter().flatten() {
        let (Some(path), Some(fm)) = (
            row["file_path"].as_str(),
            row["frontmatter"].as_object().cloned(),
        ) else {
            continue;
        };
        let mut fm: HashMap<String, Value> = fm.into_iter().collect();
        if let Some(title) = row["title"].as_str() {
            fm.entry("title".to_string()).or_insert(json!(title));
        }
        if let Some(d) = row["description"].as_str() {
            fm.insert("description".to_string(), json!(d));
        }
        out.push((
            row["score"].as_f64().unwrap_or(0.0),
            Found {
                path: path.to_string(),
                fm,
            },
        ));
    }
    out
}

fn status_filter(requested: Option<&[String]>, default: &[&str]) -> Result<Vec<String>, McpError> {
    let list: Vec<String> = match requested {
        Some(list) if !list.is_empty() => list.to_vec(),
        _ => default.iter().map(|s| s.to_string()).collect(),
    };
    for s in &list {
        if !logic::STATUSES.contains(&s.as_str()) {
            return Err(invalid(format!(
                "status must be among: {}",
                logic::STATUSES.join(", ")
            )));
        }
    }
    Ok(list)
}

#[tool_router(router = memory_router, vis = "pub(crate)")]
impl KbSearchServer {
    /// The config snapshot and memory root for one call. The tools are hidden while
    /// the group is off, so this only refuses a call that bypassed that.
    fn memory_settings(&self) -> Result<(std::sync::Arc<ResolvedConfig>, String), McpError> {
        if !self.memory_enabled {
            return Err(McpError::invalid_params("tool not found".to_string(), None));
        }
        let config = self.config();
        let root = memory::root_dir(&self.memory_cfg);
        Ok((config, root))
    }

    /// What this caller may read, inside the memory area (and `area`, when named).
    fn memory_reach(&self, root: &str, area: Option<&str>) -> Result<Reach, McpError> {
        let dir = match area.map(str::trim).filter(|a| !a.is_empty()) {
            Some(a) if crate::config::valid_memory_area(a) => format!("{root}/{a}"),
            Some(_) => return Err(invalid("area is not a valid folder name")),
            None => root.to_string(),
        };
        Ok(self.principal().reach(Op::Read).within(&dir))
    }

    async fn memory_read_text(&self, rel: &str) -> Result<String, McpError> {
        let canonical =
            retrieval::resolve_within_data(rel, &self.canonical_data_path, &self.include_patterns)
                .map_err(|_| no_such_memory())?;
        tokio::fs::read_to_string(canonical)
            .await
            .map_err(|_| no_such_memory())
    }

    /// The memory `id`, if it exists and the caller may read it. The index answers
    /// first; a memory written moments ago is not indexed yet, so a miss falls back
    /// to the folders on disk.
    async fn memory_find(&self, root: &str, id: &str) -> Result<Option<Found>, McpError> {
        if !logic::is_valid_id(id) {
            return Ok(None);
        }
        let reach = self.memory_reach(root, None)?;
        if let Ok(db) = self.state_db().await {
            let query = DocumentQuery {
                filters: vec![("id".to_string(), FieldFilter::AnyOf(vec![id.to_string()]))],
                scope: reach.dirs(),
                limit: 2,
                ..Default::default()
            };
            if let Ok(found) = db.query_documents(&query).await {
                for doc in found.documents {
                    if let Ok(Some(fm)) = db.document_frontmatter(&doc.file_path).await {
                        let fm: HashMap<String, Value> = fm.into_iter().collect();
                        if fm.get("id").and_then(Value::as_str) == Some(id) {
                            return Ok(Some(Found {
                                path: doc.file_path,
                                fm,
                            }));
                        }
                    }
                }
            }
        }
        self.memory_find_on_disk(root, id, &reach).await
    }

    async fn memory_find_on_disk(
        &self,
        root: &str,
        id: &str,
        reach: &Reach,
    ) -> Result<Option<Found>, McpError> {
        let Ok(mut areas) = tokio::fs::read_dir(self.canonical_data_path.join(root)).await else {
            return Ok(None);
        };
        while let Ok(Some(area)) = areas.next_entry().await {
            let area_name = area.file_name().to_string_lossy().into_owned();
            if !area.file_type().await.is_ok_and(|t| t.is_dir())
                || !crate::config::valid_memory_area(&area_name)
            {
                continue;
            }
            let Ok(mut files) = tokio::fs::read_dir(area.path()).await else {
                continue;
            };
            while let Ok(Some(file)) = files.next_entry().await {
                let rel = format!("{root}/{area_name}/{}", file.file_name().to_string_lossy());
                if !logic::path_is_memory(&rel, id) || !reach.covers(&rel) {
                    continue;
                }
                let Ok(text) = self.memory_read_text(&rel).await else {
                    continue;
                };
                let (fm, _) = crate::validate::parse_frontmatter_raw(&text);
                if fm.get("id").and_then(Value::as_str) == Some(id) {
                    return Ok(Some(Found { path: rel, fm }));
                }
            }
        }
        Ok(None)
    }

    /// Live memories the caller can read whose text is close to `body`: at or above
    /// `memory.duplicate_threshold`, closest first. An unavailable index or
    /// embedder is an error, never "no duplicates".
    async fn memory_duplicates(
        &self,
        config: &ResolvedConfig,
        root: &str,
        body: &str,
    ) -> Result<Vec<(Found, f32)>, McpError> {
        let query = crate::similarity::build_dedup_query(
            body,
            Some(&logic::describe(body)),
            config.chunking.prepend_description,
        );
        let scope = self.memory_reach(root, None)?;
        #[cfg(test)]
        let seeded = self.test_duplicate_hits.lock().unwrap().take();
        #[cfg(not(test))]
        let seeded: Option<Vec<crate::similarity::Neighbour>> = None;
        let hits = match seeded {
            Some(hits) => Ok(hits),
            None => {
                crate::similarity::find_near_duplicates(
                    &self.deps(),
                    &self.canonical_data_path,
                    &query,
                    &scope,
                    DUPLICATE_CANDIDATES,
                )
                .await
            }
        }
        .map_err(|e| {
            tracing::warn!("memory duplicate check failed: {e:?}");
            McpError::internal_error(
                "the duplicate check is unavailable right now; try again, or pass \
                 on_duplicate: keep_both to save without it"
                    .to_string(),
                None,
            )
        })?;
        let db = self.state_db().await.ok();
        let mut out = Vec::new();
        for hit in hits
            .into_iter()
            .filter(|h| h.score >= self.memory_cfg.duplicate_threshold)
        {
            let fm = match db {
                Some(db) => db.document_frontmatter(&hit.path).await.ok().flatten(),
                None => None,
            };
            let Some(fm) = fm else { continue };
            let found = Found {
                path: hit.path,
                fm: fm.into_iter().collect(),
            };
            if found.is_live() {
                out.push((found, hit.score));
            }
        }
        Ok(out)
    }

    fn memory_now() -> Result<String, McpError> {
        crate::datetime::payload_text(jiff::Timestamp::now())
            .ok_or_else(|| McpError::internal_error("the clock is unavailable".to_string(), None))
    }

    #[tool(annotations(destructive_hint = true, idempotent_hint = false))]
    pub(super) async fn memory_create(
        &self,
        Parameters(p): Parameters<MemoryCreateParams>,
    ) -> Result<CallToolResult, McpError> {
        let (config, root) = self.memory_settings()?;
        let ty = p.ty.trim().to_ascii_lowercase();
        if !logic::TYPES.contains(&ty.as_str()) {
            return Err(invalid(format!(
                "type must be one of: {}",
                logic::TYPES.join(", ")
            )));
        }
        logic::validate_fields(
            Some(&p.title),
            Some(&p.body),
            p.tags.as_deref(),
            p.source_url.as_deref(),
        )
        .map_err(invalid)?;
        let area = p
            .area
            .as_deref()
            .map(str::trim)
            .filter(|a| !a.is_empty())
            .unwrap_or(&self.memory_cfg.default_area)
            .to_string();
        if !crate::config::valid_memory_area(&area) {
            return Err(invalid("area is not a valid folder name"));
        }
        let on_duplicate = p.on_duplicate.as_deref().unwrap_or("reject");
        if !["reject", "supersede", "keep_both"].contains(&on_duplicate) {
            return Err(invalid(
                "on_duplicate must be one of: reject, supersede, keep_both",
            ));
        }

        let id = logic::new_id().map_err(|e| McpError::internal_error(e, None))?;
        let path = logic::memory_path(&root, &area, &p.title, &id);
        // First, before any lookup: a caller with no write grant here learns nothing
        // about what exists.
        self.authorize_write(&path)?;

        // What this memory replaces: the one named, or (on_duplicate: supersede) the
        // closest live duplicate.
        let mut replaces: Option<Found> = None;
        if let Some(old_id) = p.supersedes.as_deref() {
            let old = self
                .memory_find(&root, old_id.trim())
                .await?
                .ok_or_else(no_such_memory)?;
            if !old.is_live() {
                return Err(invalid(
                    "that memory is already superseded or forgotten; save a new one without supersedes",
                ));
            }
            replaces = Some(old);
        } else if on_duplicate != "keep_both" {
            let mut duplicates = self.memory_duplicates(&config, &root, &p.body).await?;
            if !duplicates.is_empty() {
                if on_duplicate == "reject" {
                    let rows: Vec<Value> = duplicates
                        .iter()
                        .map(|(f, score)| {
                            let mut row = record(&f.path, &f.fm);
                            row.insert("score".into(), round_score(f64::from(*score)));
                            Value::Object(row)
                        })
                        .collect();
                    return Ok(CallToolResult::structured(json!({
                        "created": false,
                        "reason": "duplicate",
                        "duplicates": rows,
                        "hint": "Nothing was saved. Use memory_update on one of these, or call \
                                 memory_create again with on_duplicate: supersede (replace the \
                                 closest) or keep_both.",
                    })));
                }
                replaces = Some(duplicates.remove(0).0);
            }
        }
        if let Some(old) = &replaces {
            self.authorize_write(&old.path)?;
        }

        let now = Self::memory_now()?;
        let tags = p.tags.clone().unwrap_or_default();
        let principal = self.principal();
        let content = logic::render_new(&logic::NewMemory {
            id: &id,
            title: &p.title,
            body: &p.body,
            ty: &ty,
            tags: &tags,
            source_client: principal.id(),
            source_url: p.source_url.as_deref(),
            status: self.memory_cfg.default_status.as_str(),
            supersedes: replaces.as_ref().map(Found::id),
            now: &now,
        })
        .map_err(|e| McpError::internal_error(e, None))?;

        let token = crate::secrets::git_token(&config).map_err(credential_error)?;
        let deps = self.write_deps(&config, token.as_deref(), None);
        let message = format!("memory: add {id}");
        match &replaces {
            None => {
                write::write_document(
                    &deps,
                    WriteRequest {
                        rel_path: &path,
                        change: DocChange::Create(&content),
                        message: Some(&message),
                        default_verb: "add",
                        force_new: Some(true),
                        operation: "memory_create",
                        expected_version: None,
                        dest_path: None,
                        archive: None,
                    },
                )
                .await
                .map_err(|e| {
                    create_edit_error_to_mcp_error(e, &path, true, &self.canonical_data_path, None)
                })?;
            }
            Some(old) => {
                // One commit: the new revision and the old one's new status.
                let supersede = |text: &str| -> Result<String, String> {
                    match logic::status_of(text).as_deref() {
                        Some(s) if logic::LIVE_STATUSES.contains(&s) => {
                            logic::with_status(text, "superseded", &now)
                        }
                        other => Err(format!(
                            "the memory being replaced is already {}",
                            other.unwrap_or("unknown")
                        )),
                    }
                };
                let requests = [
                    BatchWriteRequest {
                        rel_path: &path,
                        change: DocChange::Create(&content),
                        force_new: Some(true),
                        expected_version: None,
                    },
                    BatchWriteRequest {
                        rel_path: &old.path,
                        change: DocChange::Relative(&supersede),
                        force_new: None,
                        expected_version: None,
                    },
                ];
                write::write_documents_batch(&deps, &requests, Some(&message))
                    .await
                    .map_err(batch_write_error_to_mcp_error)?;
            }
        }

        let mut result = json!({
            "created": true,
            "id": id,
            "path": path,
            "status": self.memory_cfg.default_status.as_str(),
        });
        if let Some(old) = &replaces {
            result["superseded"] = json!(old.id());
        }
        Ok(CallToolResult::structured(result))
    }

    /// `search` over the memory area in document mode, as memory rows.
    async fn memory_query(
        &self,
        root: &str,
        q: MemoryQuery<'_>,
    ) -> Result<Vec<(f64, Found)>, McpError> {
        let MemoryQuery {
            query,
            area,
            types,
            tags,
            statuses,
            limit,
        } = q;
        // Fails on a bad area before the search runs.
        self.memory_reach(root, area)?;
        let mut filters = serde_json::Map::new();
        filters.insert("status".into(), json!(statuses));
        if let Some(types) = types.filter(|t| !t.is_empty()) {
            for t in types {
                if !logic::TYPES.contains(&t.as_str()) {
                    return Err(invalid(format!(
                        "type must be among: {}",
                        logic::TYPES.join(", ")
                    )));
                }
            }
            filters.insert("type".into(), json!(types));
        }
        if let Some(tags) = tags.filter(|t| !t.is_empty()) {
            filters.insert("tags".into(), json!({ "all_of": tags }));
        }
        let prefix = match area.map(str::trim).filter(|a| !a.is_empty()) {
            Some(a) => format!("{root}/{a}"),
            None => root.to_string(),
        };
        let response = self
            .search(Parameters(SearchParams {
                query: Some(query.to_string()),
                filters: Some(SearchFiltersInput(filters)),
                path_prefix: Some(prefix.clone()),
                granularity: Some("document".to_string()),
                limit: Some(limit),
                ..Default::default()
            }))
            .await?;
        // `path_prefix` is a substring match (`memory` also finds
        // `projects/memory-leak/x.md`), so keep only what is really under the prefix:
        // anything else with a `status` field would otherwise be returned, and
        // recalled into a model's context, as a memory.
        let mut rows = memory_rows(&response.structured_content.unwrap_or(Value::Null));
        rows.retain(|(_, f)| crate::policy::prefix_covers(&prefix, &f.path));
        Ok(rows)
    }

    #[tool(annotations(read_only_hint = true))]
    pub(super) async fn memory_search(
        &self,
        Parameters(p): Parameters<MemorySearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let (_config, root) = self.memory_settings()?;
        if p.query.trim().is_empty() {
            return Err(invalid("query must not be empty"));
        }
        let statuses = status_filter(p.status.as_deref(), &logic::LIVE_STATUSES)?;
        let limit = p
            .limit
            .unwrap_or(DEFAULT_SEARCH_LIMIT)
            .clamp(1, MAX_SEARCH_LIMIT);
        let found = self
            .memory_query(
                &root,
                MemoryQuery {
                    query: &p.query,
                    area: p.area.as_deref(),
                    types: p.ty.as_deref(),
                    tags: p.tags.as_deref(),
                    statuses: &statuses,
                    limit,
                },
            )
            .await?;
        let memories: Vec<Value> = found
            .iter()
            .map(|(score, f)| {
                let mut row = record(&f.path, &f.fm);
                if let Some(d) = f.str("description") {
                    row.insert("snippet".into(), json!(d));
                }
                row.insert("score".into(), json!(score));
                Value::Object(row)
            })
            .collect();
        Ok(CallToolResult::structured(json!({
            "returned": memories.len(),
            "memories": memories,
        })))
    }

    #[tool(annotations(read_only_hint = true))]
    pub(super) async fn memory_get(
        &self,
        Parameters(p): Parameters<MemoryGetParams>,
    ) -> Result<CallToolResult, McpError> {
        let (_config, root) = self.memory_settings()?;
        let found = self
            .memory_find(&root, p.id.trim())
            .await?
            .ok_or_else(no_such_memory)?;
        let text = self.memory_read_text(&found.path).await?;
        // The file is the truth: the index can lag a write that just landed.
        let (stored, body) = crate::validate::parse_frontmatter_raw(&text);
        let mut result = record(&found.path, &stored);
        result.insert("body".into(), json!(body.trim()));
        result.insert(
            "version".into(),
            json!(write::document_version(text.as_bytes())),
        );

        // What this one replaced, oldest last.
        let mut replaced: Vec<Value> = Vec::new();
        let mut next = found.str("supersedes").map(str::to_string);
        while let Some(old_id) = next.take() {
            if replaced.len() >= MAX_CHAIN {
                break;
            }
            let Some(old) = self.memory_find(&root, &old_id).await? else {
                break;
            };
            next = old.str("supersedes").map(str::to_string);
            replaced.push(Value::Object(record(&old.path, &old.fm)));
        }
        if !replaced.is_empty() {
            result.insert("supersedes_chain".into(), json!(replaced));
        }
        // What replaced this one, from the index.
        if let Ok(db) = self.state_db().await {
            let query = DocumentQuery {
                filters: vec![(
                    "supersedes".to_string(),
                    FieldFilter::AnyOf(vec![found.id().to_string()]),
                )],
                scope: self.memory_reach(&root, None)?.dirs(),
                limit: 1,
                ..Default::default()
            };
            if let Ok(res) = db.query_documents(&query).await
                && let Some(doc) = res.documents.first()
                && let Ok(Some(fm)) = db.document_frontmatter(&doc.file_path).await
            {
                let fm: HashMap<String, Value> = fm.into_iter().collect();
                result.insert(
                    "superseded_by".into(),
                    Value::Object(record(&doc.file_path, &fm)),
                );
            }
        }
        Ok(CallToolResult::structured(Value::Object(result)))
    }

    #[tool(annotations(destructive_hint = true, idempotent_hint = true))]
    pub(super) async fn memory_update(
        &self,
        Parameters(p): Parameters<MemoryUpdateParams>,
    ) -> Result<CallToolResult, McpError> {
        let (config, root) = self.memory_settings()?;
        let changes = logic::Changes {
            title: p.title,
            body: p.body,
            tags: p.tags,
            source_url: p.source_url,
        };
        if changes.is_empty() {
            return Err(invalid(
                "nothing to change: give title, body, tags or source_url",
            ));
        }
        logic::validate_fields(
            changes.title.as_deref(),
            changes.body.as_deref(),
            changes.tags.as_deref(),
            changes.source_url.as_deref(),
        )
        .map_err(invalid)?;
        let found = self
            .memory_find(&root, p.id.trim())
            .await?
            .ok_or_else(no_such_memory)?;
        self.authorize_write(&found.path)?;
        if !found.is_live() {
            return Err(invalid(
                "that memory is superseded or forgotten and cannot be updated; save a new one",
            ));
        }

        let now = Self::memory_now()?;
        let after = self.memory_cfg.default_status.as_str();
        let edit = |text: &str| -> Result<String, String> {
            match logic::status_of(text).as_deref() {
                Some(s) if logic::LIVE_STATUSES.contains(&s) => {
                    logic::apply_changes(text, &changes, after, &now)
                }
                other => Err(format!(
                    "the memory is {} and cannot be updated",
                    other.unwrap_or("unknown")
                )),
            }
        };
        let token = crate::secrets::git_token(&config).map_err(credential_error)?;
        let deps = self.write_deps(&config, token.as_deref(), None);
        let message = format!("memory: update {}", found.id());
        let outcome = write::write_document(
            &deps,
            WriteRequest {
                rel_path: &found.path,
                change: DocChange::Relative(&edit),
                message: Some(&message),
                default_verb: "update",
                force_new: None,
                operation: "memory_update",
                expected_version: None,
                dest_path: None,
                archive: None,
            },
        )
        .await
        .map_err(|e| {
            create_edit_error_to_mcp_error(e, &found.path, false, &self.canonical_data_path, None)
        })?;
        let status = self
            .memory_read_text(&found.path)
            .await
            .ok()
            .and_then(|t| logic::status_of(&t));
        let mut result = json!({
            "id": found.id(),
            "path": found.path,
            "changed": !outcome.diff.is_empty(),
        });
        if let Some(status) = status {
            result["status"] = json!(status);
        }
        Ok(CallToolResult::structured(result))
    }

    #[tool(annotations(destructive_hint = true, idempotent_hint = true))]
    pub(super) async fn memory_forget(
        &self,
        Parameters(p): Parameters<MemoryForgetParams>,
    ) -> Result<CallToolResult, McpError> {
        let (config, root) = self.memory_settings()?;
        let found = self
            .memory_find(&root, p.id.trim())
            .await?
            .ok_or_else(no_such_memory)?;
        self.authorize_write(&found.path)?;
        let now = Self::memory_now()?;
        let edit = |text: &str| logic::with_status(text, "forgotten", &now);
        let token = crate::secrets::git_token(&config).map_err(credential_error)?;
        let deps = self.write_deps(&config, token.as_deref(), None);
        let message = format!("memory: forget {}", found.id());
        let outcome = write::write_document(
            &deps,
            WriteRequest {
                rel_path: &found.path,
                change: DocChange::Relative(&edit),
                message: Some(&message),
                default_verb: "update",
                force_new: None,
                operation: "memory_forget",
                expected_version: None,
                dest_path: None,
                archive: None,
            },
        )
        .await
        .map_err(|e| {
            create_edit_error_to_mcp_error(e, &found.path, false, &self.canonical_data_path, None)
        })?;
        Ok(CallToolResult::structured(json!({
            "id": found.id(),
            "status": "forgotten",
            "changed": !outcome.diff.is_empty(),
        })))
    }

    #[tool(annotations(read_only_hint = true))]
    pub(super) async fn memory_recall(
        &self,
        Parameters(p): Parameters<MemoryRecallParams>,
    ) -> Result<CallToolResult, McpError> {
        let (_config, root) = self.memory_settings()?;
        if p.query.trim().is_empty() {
            return Err(invalid("query must not be empty"));
        }
        let max_items = p
            .max_items
            .unwrap_or(DEFAULT_RECALL_ITEMS)
            .clamp(1, MAX_RECALL_ITEMS);
        let max_chars = p
            .max_chars
            .unwrap_or(DEFAULT_RECALL_CHARS)
            .clamp(MIN_RECALL_CHARS, MAX_RECALL_CHARS);
        let statuses: Vec<String> = if p.include_proposed.unwrap_or(false) {
            vec!["confirmed".into(), "proposed".into()]
        } else {
            vec!["confirmed".into()]
        };
        // Over-fetch: reading each memory can drop some (gone since indexing).
        let found = self
            .memory_query(
                &root,
                MemoryQuery {
                    query: &p.query,
                    area: p.area.as_deref(),
                    types: None,
                    tags: None,
                    statuses: &statuses,
                    limit: (max_items as u64 * 2).min(MAX_SEARCH_LIMIT),
                },
            )
            .await?;
        let mut items = Vec::new();
        for (score, f) in found {
            let Ok(text) = self.memory_read_text(&f.path).await else {
                continue;
            };
            let (_, body) = crate::validate::parse_frontmatter_raw(&text);
            items.push(logic::RecallItem {
                id: f.id().to_string(),
                ty: f.str("type").unwrap_or("fact").to_string(),
                title: f.str("title").unwrap_or("(untitled)").to_string(),
                text: body,
                score,
                path: f.path,
            });
        }
        logic::recall_order(&mut items);
        let pack = logic::recall_pack(&items, max_items, max_chars);
        let used: Vec<Value> = pack
            .included
            .iter()
            .map(|&i| {
                json!({
                    "id": items[i].id,
                    "type": items[i].ty,
                    "title": items[i].title,
                })
            })
            .collect();
        Ok(CallToolResult::structured(json!({
            "context": pack.context,
            "items": used,
            "truncated": pack.truncated,
        })))
    }
}
