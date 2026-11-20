use super::*;

use rmcp::{
    ErrorData as McpError, handler::server::wrapper::Parameters, schemars, tool, tool_router,
};
use tracing::{debug, error, warn};

use crate::{
    retrieval::{self, GetDocumentError},
    write::{self},
};

/// Parameters for `get_document`.
///
/// Modes — see `retrieval::parse_document_view_request` for the exact
/// validation rules (#286):
///   - Nothing but `path`: the whole document (unchanged).
///   - `start_line`/`end_line`: a raw line range (unchanged); excludes every
///     other mode parameter.
///   - `line` or `heading_path` (optionally with `levels_up`): one resolved
///     section, by line or by heading path.
///   - `outline: true`: the heading outline, no content — of the whole
///     document, or of the selected section when combined with a selector.
#[derive(Debug, Default, serde::Deserialize, schemars::JsonSchema)]
pub struct GetDocumentParams {
    /// Relative path, unique basename, or absolute path.
    pub path: String,
    // Per-mode rules beyond these one-liners (ambiguous or unresolved
    // selectors, oversized sections, capped outlines) are taught by the error
    // or the response itself — `SectionError::data()`, `OutlineView::hint`.
    /// First line (1-based, inclusive). An explicit range is never size-capped.
    #[serde(default)]
    pub start_line: Option<usize>,
    /// Last line (1-based, inclusive).
    #[serde(default)]
    pub end_line: Option<usize>,
    /// Select the section containing this 1-based line (its deepest enclosing
    /// heading). Not combinable with `heading_path` or `start_line`/`end_line`.
    #[serde(default)]
    pub line: Option<usize>,
    /// Select a section by heading path, e.g. `["Spells", "Fireball"]`:
    /// complete heading names in order. A trailing part (`["Fireball"]`) or an
    /// in-order subset that skips middle levels also resolves when it names
    /// exactly one section. Ignores case and whitespace. Not combinable with
    /// `line` or `start_line`/`end_line`.
    #[serde(default)]
    pub heading_path: Option<Vec<String>>,
    /// With `line`/`heading_path`: widen to the Nth parent heading (clamped at
    /// the top).
    #[serde(default)]
    pub levels_up: Option<usize>,
    /// Return the heading outline with line ranges instead of text — of the
    /// document, or of the section chosen by `line`/`heading_path`.
    #[serde(default)]
    pub outline: Option<bool>,
    /// Also return the last N changes to this document (1-100), newest first.
    #[serde(default)]
    pub history: Option<usize>,
    /// Link lists come with a whole-document read; `true` adds them to any
    /// read, `false` drops them.
    #[serde(default)]
    pub links: Option<bool>,
}

/// `get_document`'s link graph, in the compact shape a model reads: `links_out`
/// (paths this document links to that exist), `broken_links` (link targets
/// that do not), `links_in` (paths linking here), and `similar` (`{path,
/// score}` inferred neighbors, either direction, only when semantic edges are
/// on). Each key appears only when non-empty; `links_out_total`/`links_in_total`
/// only when that direction was capped at `retrieval::MAX_LINKS_PER_DIRECTION`.
pub(super) fn insert_link_lists(
    structured: &mut serde_json::Map<String, serde_json::Value>,
    links_out: &crate::state::LinkPage<crate::state::OutboundLink>,
    links_in: &crate::state::LinkPage<crate::state::InboundLink>,
) {
    let mut out: Vec<&str> = Vec::new();
    let mut broken: Vec<&str> = Vec::new();
    let mut incoming: Vec<&str> = Vec::new();
    let mut similar: Vec<serde_json::Value> = Vec::new();
    let mut seen_similar: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for l in &links_out.links {
        if l.kind == "semantic" {
            if seen_similar.insert(&l.target_path) {
                similar.push(serde_json::json!({
                    "path": l.target_path,
                    "score": l.score.map_or(serde_json::Value::Null, round_score),
                }));
            }
        } else if l.exists {
            out.push(&l.target_path);
        } else {
            broken.push(&l.target_path);
        }
    }
    for l in &links_in.links {
        if l.kind == "semantic" {
            if seen_similar.insert(&l.source_path) {
                similar.push(serde_json::json!({
                    "path": l.source_path,
                    "score": l.score.map_or(serde_json::Value::Null, round_score),
                }));
            }
        } else {
            incoming.push(&l.source_path);
        }
    }
    for (key, list) in [
        ("links_out", out),
        ("broken_links", broken),
        ("links_in", incoming),
    ] {
        if !list.is_empty() {
            structured.insert(key.to_string(), serde_json::json!(list));
        }
    }
    if !similar.is_empty() {
        structured.insert("similar".to_string(), serde_json::Value::Array(similar));
    }
    if links_out.has_more() {
        structured.insert(
            "links_out_total".to_string(),
            serde_json::json!(links_out.total),
        );
    }
    if links_in.has_more() {
        structured.insert(
            "links_in_total".to_string(),
            serde_json::json!(links_in.total),
        );
    }
}

/// Map `retrieval::resolve_document_view`'s error into the `McpError` this
/// tool surface returns. The `Range` arm keeps `get_document`'s existing
/// range-error wording; `Section` follows the same "invalid_params + a `data`
/// payload the caller can act on" shape `GetDocumentError::NotFound`'s
/// suggestions already establish, except here the payload is structured
/// (`hint`/`candidates`) rather than folded into the message text alone,
/// since `SectionError::data()` already builds exactly that (#286).
pub(super) fn document_view_error_to_mcp(err: retrieval::DocumentViewError) -> McpError {
    match err {
        retrieval::DocumentViewError::Range(e) => McpError::invalid_params(e.to_string(), None),
        retrieval::DocumentViewError::Section(e) => {
            McpError::invalid_params(e.message(), Some(e.data()))
        }
    }
}

#[tool_router(router = get_document_router, vis = "pub(crate)")]
impl KbSearchServer {
    #[tool(annotations(read_only_hint = true))]
    pub(super) async fn get_document(
        &self,
        Parameters(params): Parameters<GetDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        let raw = params.path.trim();
        if raw.is_empty() {
            return Err(McpError::invalid_params(
                "Path parameter is empty".to_string(),
                None,
            ));
        }
        if raw.len() > MAX_PATH_LEN {
            return Err(McpError::invalid_params(
                format!("path exceeds maximum length of {MAX_PATH_LEN} characters"),
                None,
            ));
        }

        // Checked before the path is resolved: a malformed combination of
        // parameters is wrong no matter which document it was aimed at, so it
        // should not cost a metadata-index open and a file read to say so
        // (#286 — mode resolution itself lives in retrieval.rs so this
        // tool and `/api/doc` can't drift apart on it).
        let request = retrieval::parse_document_view_request(
            params.start_line,
            params.end_line,
            params.line,
            params.heading_path,
            params.levels_up,
            params.outline.unwrap_or(false),
        )
        .map_err(|e| McpError::invalid_params(e, None))?;

        debug!(path = %raw, "get_document called");

        // The fuzzy-basename fallback resolves against the SQLite metadata index
        // rather than a Qdrant facet fetch, so the index has to be opened here.
        // Only the fallback needs it — an exact path hit is served from disk and
        // never touches this — but the resolve happens inside `get_document`, so
        // it is passed unconditionally.
        let index = self.state_db().await.map_err(|e| {
            error!("get_document could not open the metadata index: {:#}", e);
            McpError::internal_error(format!("Document index unavailable: {}", e), None)
        })?;

        match retrieval::get_document(&self.deps(), index, raw).await {
            Ok(doc) => {
                debug!(path = %raw, "get_document served");
                // The caller round-trips this into write_document/delete_document as
                // `expected_version`. Computed before any slicing and always over the
                // whole file: it names the document on disk, so a slice's version
                // could never match.
                let version = write::document_version(doc.content.as_bytes());
                let rel_path = retrieval::relative_to_data(
                    &doc.path.to_string_lossy(),
                    &self.canonical_data_path,
                );
                let section_max_bytes = self.config().search.section_max_bytes;
                let view =
                    retrieval::resolve_document_view(&doc.content, &request, section_max_bytes)
                        .map_err(document_view_error_to_mcp)?;

                // The view-specific fields come from `retrieval::document_view_json`,
                // shared with `/api/doc`; only the envelope is added here.
                let mut structured = retrieval::document_view_json(&view);
                structured.insert("path".to_string(), serde_json::json!(rel_path));
                structured.insert("version".to_string(), serde_json::json!(version));
                // The link graph rides on a whole-document read (the one a caller
                // makes to understand a document) unless asked for or declined
                // explicitly: on a targeted read it would outweigh the text itself.
                // An oversized document read whole is still that read when it comes
                // back cut short (`truncated`, #290), as the outline an oversized
                // document with headings degrades to is; only a range the caller
                // named is targeted.
                let whole_read = match &view {
                    retrieval::DocumentView::Range(slice) => !slice.partial() || slice.truncated,
                    retrieval::DocumentView::Outline(outline) => outline.document_oversized,
                    retrieval::DocumentView::Section(_) => false,
                };
                if params.links.unwrap_or(whole_read) {
                    insert_link_lists(&mut structured, &doc.links_out, &doc.links_in);
                }
                // Opt-in (#257): a caller that did not ask pays no subprocess.
                if let Some(limit) = params.history {
                    let data_path = self.canonical_data_path.to_string_lossy().into_owned();
                    let history = retrieval::document_changes_json(&data_path, &rel_path, limit)
                        .await
                        .map_err(|e| {
                            error!("get_document history failed for '{rel_path}': {e:#}");
                            McpError::internal_error(
                                "this document's change history is unavailable".to_string(),
                                None,
                            )
                        })?;
                    structured.insert("history".to_string(), history);
                }
                Ok(CallToolResult::structured(serde_json::Value::Object(
                    structured,
                )))
            }
            Err(GetDocumentError::Outside) => {
                warn!(path = %raw, "get_document: path outside data directory");
                Err(McpError::invalid_params(
                    "File path is outside the data directory".to_string(),
                    None,
                ))
            }
            Err(GetDocumentError::NotPermitted) => {
                warn!(path = %raw, "get_document: file type not permitted");
                Err(McpError::invalid_params(
                    "File type not permitted".to_string(),
                    None,
                ))
            }
            Err(GetDocumentError::NotFound { suggestions }) => {
                let mut msg = format!("File not found: '{}'.", raw);
                if suggestions.is_empty() {
                    msg.push_str(" Use the `search` tool to find paths.");
                } else {
                    msg.push_str(&format!(
                        " Closest indexed files: {}. Or use the `search` tool.",
                        suggestions.join(", ")
                    ));
                }
                Err(McpError::invalid_params(msg, None))
            }
            Err(GetDocumentError::Ambiguous { matches }) => {
                let basename = std::path::Path::new(raw)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(raw);
                Err(McpError::invalid_params(
                    format!(
                        "Multiple files match basename '{}': {}. Use a more specific path.",
                        basename,
                        matches.join(", ")
                    ),
                    None,
                ))
            }
            Err(GetDocumentError::Io(msg)) => Err(McpError::invalid_params(msg, None)),
        }
    }
}
