//! Transport-agnostic core of the write tools: validation, the create-path dedup
//! gate, filesystem write/removal, git commit+sync, and the pre-commit /
//! post-commit rollback state machine.
//!
//! Extracted out of `mcp.rs` so both the MCP tool surface (`create_document`,
//! `edit_document`, `delete_document`) and the HTTP UI (`web.rs`, a later chunk)
//! drive the exact same commit/rollback logic instead of maintaining two copies of
//! it. `mcp.rs`'s tool methods are thin adapters: they do transport-specific path
//! resolution (fuzzy basename matching, turning tool parameters into a
//! [`DocChange`]) and map [`WriteSuccess`]/[`WriteError`] back onto the exact
//! `CallToolResult`/`McpError` shapes their existing tests pin down.
//!
//! A note on `WriteError::PostCommitPending`, which does not exist here: the plan
//! this module was built from listed it as a `WriteError` variant, but a
//! "committed locally, push still pending" write is not a failure — every existing
//! caller (and the fixed HTTP contract this module was built against) reports it
//! as a 200/`Ok` result, just like a fully synced write, so it is modeled as
//! `WriteSuccess { outcome: WriteOutcome::CommittedPendingSync, .. }` instead. The
//! push-failure cause is logged where the sync failed and not carried on the
//! result: no caller relays it.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use futures::stream::{self, StreamExt};
use tracing::{error, warn};

use crate::config::ValidationConfig;
use crate::embed::QueryEmbedder;
use crate::git;
use crate::qdrant::RetrievalStore;
use crate::retrieval::{RetrievalDeps, SearchFilters};
use crate::schema::{self, SharedSchemaCache};
use crate::state::StateDb;
use crate::validate::{self, ValidationResult};

// ---------------------------------------------------------------------------
// Dedup gate (create-path near-duplicate refusal)
// ---------------------------------------------------------------------------

/// Maximum number of characters from the new document's content used to build
/// the dedup query text. Keeps the embedding request within typical token limits
/// (most embedding models cap at ~512–8192 tokens; 2000 chars ≈ 400–500 tokens).
const DEDUP_QUERY_CHAR_LIMIT: usize = 2000;

/// A near-duplicate found during dedup gate evaluation.
#[derive(Debug, Clone)]
pub struct DuplicateHit {
    pub file_path: String,
    pub score: f32,
}

/// Pure decision function: given the closest match from Qdrant (if any) and a
/// threshold, decide whether to refuse the write. Returns `Some(DuplicateHit)`
/// when the write should be blocked, `None` when it is safe to proceed.
///
/// This is factored out so it can be unit-tested without a live Qdrant/embedder.
pub fn dedup_verdict(top: Option<(String, f32)>, threshold: f32) -> Option<DuplicateHit> {
    match top {
        Some((path, score)) if score >= threshold => Some(DuplicateHit {
            file_path: path,
            score,
        }),
        _ => None,
    }
}

/// Build the dedup query text on the same textual basis the indexer uses, then
/// truncate it to `DEDUP_QUERY_CHAR_LIMIT`.
///
/// This must stay aligned with `chunk::chunk_markdown`: every indexed chunk is
/// prefixed with its document's own `description` when
/// `chunking.prepend_description` is set, so a dedup query built without that
/// prefix would be compared against a different textual basis than the
/// candidates it is scored against.
///
/// It deliberately does NOT reproduce `chunking.prepend_heading_path`'s
/// breadcrumb, and that is safe for a non-obvious structural reason worth
/// stating: a chunk's breadcrumb is seeded from the section that starts it and
/// merging further pieces in can only narrow it (to a common ancestor, see
/// `chunk::chunk_markdown`). A document's FIRST chunk is always seeded from its
/// first section — the preamble, which has no breadcrumb, or the first heading,
/// which has no ancestors because no heading precedes it — so chunk 0 never
/// carries a breadcrumb, whatever the document's heading structure. Since a create-path dedup query is doc-start text scored
/// against the corpus, matching chunk 0's basis is what matters.
///
/// If a future chunking change breaks that invariant — anything that can give
/// a document's first chunk a prefix this function does not build — the dedup
/// gate silently starts comparing unlike text and lets near-duplicates through
/// with no error. `mcp::tests::build_dedup_query_matches_chunk_prepend_format`
/// is the pin; keep it honest rather than adjusting it to match a regression.
pub(crate) fn build_dedup_query(
    body: &str,
    description: Option<&str>,
    prepend_description: bool,
) -> String {
    let assembled = match (prepend_description, description) {
        (true, Some(desc)) => format!("{}\n\n{}", desc, body),
        _ => body.to_string(),
    };
    assembled.chars().take(DEDUP_QUERY_CHAR_LIMIT).collect()
}

/// The dedup gate's effective `(enabled, threshold)` for a document governed by
/// `schema`: the schema file `dedup:` cascade where it sets a key, else the
/// global `write.*` value in `deps` (#272). Resolved per key, so a scope that only
/// sets `threshold` still inherits the global `enabled`.
pub(crate) fn effective_dedup<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &WriteDeps<'_, E, Q>,
    schema: &schema::ResolvedSchema,
) -> (bool, f32) {
    (
        schema.dedup_enabled.unwrap_or(deps.dedup_enabled),
        schema.dedup_threshold.unwrap_or(deps.dedup_threshold),
    )
}

/// Search options for the dedup gate.
///
/// Deliberately pinned to dense-only rather than inheriting `search.hybrid`, so
/// the returned score is a cosine similarity comparable to
/// `write.dedup_threshold`. Hybrid RRF scores top out around 0.03 — against a
/// cosine threshold like the 0.80 default the gate could never fire — and a
/// cross-encoder relevance score is not a similarity at all, so reranking is
/// also kept out of this path (see the `reranker: None` at the call site).
pub(crate) fn dedup_search_opts() -> crate::retrieval::SearchOptions {
    crate::retrieval::SearchOptions {
        limit: 1,
        min_score: None,
        hybrid: false,
        // Unused in the dense-only path, which performs no RRF fusion.
        rrf_candidates: 0,
        // Same reasoning as `hybrid` above: phrase matching adds a third fused
        // arm, which this dense-only comparison has no use for.
        phrase: false,
        explain: false,
        modified_after: None,
        modified_before: None,
        path_filter: None,
        rerank_candidate_limit: None,
        // The dedup gate wants the single closest existing chunk, full stop — not
        // a diversified page of results (limit: 1 above makes a per-document cap
        // moot anyway, but this keeps intent explicit rather than accidental).
        diversity_max_per_document: None,
    }
}

// ---------------------------------------------------------------------------
// Commit message helpers
// ---------------------------------------------------------------------------

/// Above this length (or containing a newline) a caller-supplied commit message
/// would confuse `git log`/`git commit -m`, so it is rejected up front.
const MAX_COMMIT_MESSAGE_LEN: usize = 1000;

/// Reject a caller-supplied commit message that git or the log would mangle.
fn validate_commit_message(message: Option<&str>) -> Result<(), WriteError> {
    let Some(msg) = message else {
        return Ok(());
    };
    if msg.contains('\n') {
        return Err(WriteError::InvalidCommitMessage {
            reason: "commit message must not contain newlines".to_string(),
        });
    }
    if msg.len() > MAX_COMMIT_MESSAGE_LEN {
        return Err(WriteError::InvalidCommitMessage {
            reason: format!(
                "commit message too long ({} chars); maximum is {}",
                msg.len(),
                MAX_COMMIT_MESSAGE_LEN
            ),
        });
    }
    Ok(())
}

/// Build a commit message with git trailers identifying the tool and operation.
///
/// The resulting message has the form:
/// ```text
/// <subject line>
///
/// Tool: mcp-md-wiki
/// Operation: <operation>
/// ```
///
/// `user_subject` is the caller-supplied commit message (if any). When absent,
/// `default_subject` is used. The trailer block is always appended after a blank line.
pub fn build_commit_message(
    user_subject: Option<&str>,
    default_subject: &str,
    operation: &str,
) -> String {
    let subject = user_subject.unwrap_or(default_subject);
    format!("{}\n\nTool: mcp-md-wiki\nOperation: {}", subject, operation)
}

/// Render a unified diff between `old` and `new` content, labelled with
/// `a/<relpath>` and `b/<relpath>`. Returns an empty string if there is no
/// diff (shouldn't happen for a real change).
pub fn render_unified_diff(old: &str, new: &str, relpath: &str) -> String {
    use similar::TextDiff;
    let diff = TextDiff::from_lines(old, new);
    diff.unified_diff()
        .context_radius(3)
        .header(&format!("a/{relpath}"), &format!("b/{relpath}"))
        .to_string()
}

// ---------------------------------------------------------------------------
// Dependencies, request/response/error shapes
// ---------------------------------------------------------------------------

/// Dependencies needed to run the write pipeline, independent of transport.
///
/// Mirrors what `KbSearchServer` currently pulls from `self`/its config snapshot
/// for `write_document`/`delete_document`. `retrieval` is reused wholesale for the
/// create-path dedup gate — see `crate::retrieval::RetrievalDeps`.
pub struct WriteDeps<'a, E: QueryEmbedder, Q: RetrievalStore> {
    pub retrieval: RetrievalDeps<'a, E, Q>,
    /// Canonicalized knowledge-base root. Every filesystem action re-resolves
    /// against this immediately before it happens (see `resolve_safe_write_path`),
    /// closing the TOCTOU window a slow schema/dedup lookup would otherwise leave
    /// open.
    pub canonical_data_path: &'a Path,
    pub schema_cache: &'a SharedSchemaCache,
    pub validation: &'a ValidationConfig,
    /// The same `indexing.include`/`exclude`/`exclude_files` predicate a full
    /// reconcile applies (`ingest::discover_files`), used here to filter every
    /// path a write is about to mark dirty — this write's own target path(s),
    /// any rewritten referencing documents, and `commit_outcome.rebased_paths`
    /// (paths pulled in from OTHER commits during this write's own git
    /// fetch/rebase/push) — before they reach `queue.mark_paths` (#278).
    /// `check_include_pattern` only checks `indexing.include`, never
    /// `exclude`/`exclude_files`, so a write's own target can match `include`
    /// (e.g. the default-excluded `README.md` against the default `**/*.md`)
    /// and still not be reconcile-indexable; this filter is what keeps that
    /// path out of the reindex queue the same way a reconcile would.
    pub indexing: &'a crate::config::IndexingConfig,
    /// Mirrors `chunking.prepend_description` — the dedup query must be built on
    /// the same textual basis the indexer embeds.
    pub prepend_description: bool,
    pub dedup_enabled: bool,
    pub dedup_threshold: f32,
    pub git_url: Option<&'a str>,
    pub branch: &'a str,
    pub token: Option<&'a str>,
    pub commit_author_name: &'a str,
    pub commit_author_email: &'a str,
    /// The dirty-path queue every successful write marks paths on before
    /// returning — see `reindex::ReindexQueue`. Unlike `state` below, this has
    /// no `None`/optional mode: a write that lands but never marks its path
    /// dirty is a correctness bug (the document silently never gets indexed),
    /// not a degraded-but-acceptable path, so every call site must supply a
    /// real queue. Borrowed the same way `schema_cache`/`retrieval.qdrant`/
    /// `retrieval.embed_client` are — the owning transport (`KbSearchServer`,
    /// `UiState`) holds an `Arc<ReindexQueue>` field and lends a plain
    /// reference in for the duration of one write.
    pub queue: &'a crate::reindex::ReindexQueue,
    /// The document metadata index. Three consumers, all going through
    /// `StateDb::links_targeting`'s reverse lookup ("what points at this
    /// path"): `write_document_move` and `move_directory` find documents whose
    /// body links to the move's SOURCE path so their link text can be
    /// rewritten in the same commit as the move, and `delete_document` warns
    /// about documents that link to the file being removed.
    ///
    /// `None` disables all three — the move/delete itself still happens, the
    /// reverse-link query is just skipped. For a move that means it
    /// leaves every other document's links exactly as they were (they still
    /// self-heal on that document's own next reindex, since `document_links`
    /// is rebuilt from each document's current on-disk body, not trusted as an
    /// authoritative index). This exists for callers/tests that have no
    /// `StateDb` handle at all — it is NOT a normal operating mode. Both real
    /// callers (`mcp.rs`'s `KbSearchServer`, `web.rs`'s `UiState`) have a
    /// `StateDb` available via their own `Arc<OnceCell<StateDb>>` and MUST
    /// pass `Some` here so production writes always rewrite incoming links.
    pub state: Option<&'a StateDb>,
}

/// Mark every path a write touched (its own target(s), rewritten referencing
/// documents, and `commit_outcome.rebased_paths`) dirty on `queue`.
///
/// Paths are first filtered down to the ones `indexing`'s include/exclude/
/// exclude_files predicate ([`crate::ingest::PathFilter`]) would actually index
/// (#278). That filter fails open (marks `paths` unfiltered) on a glob-build
/// error, logging loudly, rather than dropping otherwise-legitimate paths
/// because of a config problem this write did not cause.
///
/// A schema file among `paths` — carried by a `move_directory`, or pulled
/// in by this write's own rebase — is never a document, so the filter drops it;
/// it instead queues a full reconcile, which rebuilds the shared schema cache
/// before scanning (see `reindex::ReindexQueue::mark_schema_changes`).
fn mark_dirty(
    queue: &crate::reindex::ReindexQueue,
    indexing: &crate::config::IndexingConfig,
    paths: Vec<PathBuf>,
) {
    queue.mark_schema_changes(indexing, &paths);
    queue.mark_paths(crate::ingest::partition_indexable(indexing, paths).0);
}

/// A create, edit or move request against the write pipeline.
pub struct WriteRequest<'a> {
    /// Repo-relative path, already resolved and validated by the caller.
    pub rel_path: &'a str,
    /// What to do to the document's content. The new content is computed by the
    /// pipeline from the document as read under the lock — see [`DocChange`].
    pub change: DocChange<'a>,
    pub message: Option<&'a str>,
    /// Verb for the default commit message, e.g. `"add"` or `"update"`.
    pub default_verb: &'a str,
    /// When `Some(true)`, bypasses the dedup gate on create paths.
    pub force_new: Option<bool>,
    /// Label for the `Operation:` git trailer, e.g. `"create_document"`.
    pub operation: &'a str,
    /// The document version (`document_version`) the caller based this change
    /// on. Required for an absolute change to an existing document (a full
    /// replace, and any move); optional for a relative one, which still applies
    /// when it is stale: it then sharpens the refusal when an anchor no longer
    /// matches, and marks a result that carries changes made since it as
    /// [`WriteSuccess::merged`]. A create given one is refused as
    /// [`WriteError::NotFound`]: the document it was read from is gone.
    pub expected_version: Option<&'a str>,
    /// When `Some`, turns this call into a document MOVE: `rel_path` is the move's
    /// SOURCE and this is its DESTINATION. Both paths get the schema-file,
    /// eligibility and path-safety checks; frontmatter is validated against the
    /// DESTINATION's schema; the create-only dedup gate never runs. `change` must
    /// not be a create (reported as `WriteError::Internal`).
    pub dest_path: Option<&'a str>,
}

/// The two outcomes a write can land on. Both are the tool's happy path from a
/// caller's point of view — `synced` fully so, `committed_pending_sync` with a
/// caveat — which is why both live under `WriteSuccess` rather than being split
/// across the `Ok`/`Err` boundary. See this module's doc comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteOutcome {
    /// Committed and pushed.
    Synced,
    /// Committed locally, but the remote push failed. NOT rolled back — the commit
    /// is real. Will sync on the next successful write, or on manual intervention.
    CommittedPendingSync,
}

/// A successful write or delete.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WriteSuccess {
    pub outcome: WriteOutcome,
    pub sha: String,
    pub rebased_paths: Vec<PathBuf>,
    /// Unified diff of the change. Empty when an edit left the document exactly
    /// as it was — nothing is written or committed then, and `sha` is HEAD as
    /// it stands.
    pub diff: String,
    /// Repo-relative paths of OTHER documents whose link text to the move's
    /// SOURCE was rewritten to point at its new location, and which rode along
    /// in the same commit. Always empty for a create/edit/delete, and for a
    /// move with `WriteDeps::state == None` or with no incoming links to
    /// rewrite. A move silently editing other documents without surfacing
    /// which ones is not acceptable — callers must report this list, not just
    /// the move's own source/destination.
    pub rewritten_paths: Vec<String>,
    /// (#229) Repo-relative paths of OTHER documents that still link to the
    /// document just DELETED, per `StateDb::links_targeting`'s reverse lookup.
    /// Always empty for a create/edit/move — only `delete_document` populates
    /// this, and only when `WriteDeps::state` is `Some` (see that field's doc
    /// comment) and at least one referencing document exists. `delete_document`
    /// does not refuse the delete or rewrite these documents — see its own
    /// comment for why a dangling link here is treated as self-healing, same
    /// as everywhere else in this pipeline — this field exists purely so a
    /// caller with no access to server logs (the `warn!` #181 added) can still
    /// learn what it warned about and decide whether follow-up work is needed.
    pub referencing_paths: Vec<String>,
    /// The written document had other concurrent changes that were merged with
    /// this one (a three-way merge of a stale full replace, a relative edit
    /// applied over changes made since its `expected_version`, or a clean rebase
    /// over someone else's edit to the same file). Callers say so, so the agent
    /// re-reads before editing further.
    pub merged: bool,
    /// The written document's new version (`document_version`), so a caller can
    /// chain a further absolute change without re-reading. `None` for a delete,
    /// and when a merge during sync means the on-disk bytes are not known here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// Every structured failure mode of the write pipeline. Callers map these onto
/// their own transport's error shape (`McpError` for MCP, an HTTP status + JSON
/// body for the web UI) — see this module's doc comment for why
/// `PostCommitPending` is not among them.
#[derive(Debug)]
pub enum WriteError {
    /// `rel_path` names a schema file. Schema files are not documents: a
    /// direct write, delete or single-document move of one is refused whatever
    /// `indexing.include` says, because it would bypass the parse/validate/
    /// round-trip/size checks `update_schema` applies — and a schema file that is
    /// present must be valid. `rel_path` is the caller-supplied path, safe to echo.
    SchemaFile { rel_path: String },
    /// Frontmatter validation failed. Carries the full structured result so a
    /// caller can report per-field errors, not just a flat message.
    Validation { result: ValidationResult },
    /// A near-duplicate document already exists (create-path dedup gate).
    DedupHit {
        duplicate_of: String,
        similarity: f32,
        threshold: f32,
    },
    /// The caller-supplied commit message would confuse git or the log.
    InvalidCommitMessage { reason: String },
    /// `rel_path` failed the path-safety check (traversal, absolute path escaping
    /// the root, or a symlinked ancestor pointing outside it). `msg` is always
    /// built from caller-supplied input only — safe to relay to an untrusted
    /// caller verbatim (see `WriteError::Internal` for the counterpart that is
    /// not).
    UnsafePath { msg: String },
    /// The path-safety check itself failed for a reason that has nothing to do
    /// with what the caller supplied — `resolve_safe_write_path`'s
    /// canonicalize-failure branches, which embed a server-side absolute
    /// filesystem path in their message (e.g. "cannot canonicalize data root
    /// '<abs>': <io error>"). Kept distinct from `UnsafePath` so a transport
    /// that must not leak container paths to an untrusted caller (`web.rs`) can
    /// map this to a generic message while `mcp.rs` — a trusted surface where
    /// this text is diagnostically useful and was already being returned
    /// before this variant existed — keeps emitting it verbatim.
    Internal { msg: String },
    /// Create was requested but `rel_path` already exists.
    AlreadyExists,
    /// Edit or delete was requested but `rel_path` does not exist, or a create
    /// carried an `expected_version`: the document it was read from is no longer
    /// at `rel_path` (deleted or moved).
    NotFound,
    /// The document changed since the caller read it and the change cannot be
    /// carried over: a relative edit whose anchor no longer matches, a full
    /// replace that conflicts with the other change (or that git could not
    /// merge at all), a delete or move of a version that is no longer current,
    /// or a remote that kept moving under every attempt. Caller-facing text:
    /// [`EDITED_ELSEWHERE`].
    EditedElsewhere,
    /// An absolute change to an existing document (full replace, delete, move)
    /// arrived without `expected_version`.
    VersionRequired,
    /// A relative edit does not apply to the document (an anchor that never
    /// matched, a frontmatter patch of a field that is not there), or would grow
    /// it past [`MAX_CONTENT_LEN`]. `msg` is built from the caller's own input and
    /// the document, safe to relay.
    InvalidEdit { msg: String },
    /// `git add`/`git commit` failed (HEAD never moved). `rolled_back = true`
    /// means the working tree and git index were successfully restored to their
    /// pre-call state — safe to retry. `rolled_back = false` means the rollback
    /// attempt ITSELF also failed, leaving filesystem and git state inconsistent
    /// with each other and with HEAD — this needs operator attention, not a blind
    /// retry. `msg` is the formatted cause (or causes, for the `false` case).
    PreCommitFailed { rolled_back: bool, msg: String },
    /// Any other I/O or internal error (parent-directory creation, the write
    /// itself, reading a file, or validation blowing up rather than just failing).
    Io { msg: String },
}

// ---------------------------------------------------------------------------
// Shared path-safety helper
// ---------------------------------------------------------------------------

/// The two failure shapes [`resolve_safe_write_path_detailed`] can produce,
/// split by whether the message is safe to relay to an untrusted caller.
///
/// `resolve_safe_write_path` (the public, `String`-returning function every
/// existing caller outside this module already uses) flattens both variants
/// back into a plain string, so its behavior and exact message text are
/// unchanged by this split — it exists purely so `safe_write_path` (below),
/// the wrapper `write_document`/`delete_document` use internally, can pick the
/// right `WriteError` variant without re-deriving the classification by
/// sniffing message text (fragile — see `WriteError::Internal`'s doc comment
/// for why that classification exists at all).
enum PathSafetyError {
    /// Built entirely from the caller-supplied relative path and fixed text —
    /// safe to show the caller as-is.
    Rejected(String),
    /// Embeds a server-side absolute filesystem path (a canonicalize
    /// failure) — must not reach an untrusted caller verbatim.
    Internal(String),
}

impl PathSafetyError {
    fn into_message(self) -> String {
        match self {
            PathSafetyError::Rejected(msg) | PathSafetyError::Internal(msg) => msg,
        }
    }
}

/// Validate that `rel_path` is safe to write inside `data_root`, returning the
/// absolute target path on success or a classified error on failure — see
/// [`PathSafetyError`].
///
/// Checks performed (in order):
/// 1. Reject absolute paths.
/// 2. Reject any `..` component.
/// 3. Lexical `starts_with` check on the joined abs path.
/// 4. Canonicalize the deepest *existing* ancestor of the target; verify it
///    still `starts_with` the canonical data_root. This catches a symlinked
///    ancestor directory that resolves to a location outside data_root.
fn resolve_safe_write_path_detailed(
    data_root: &Path,
    rel_path: &str,
) -> Result<PathBuf, PathSafetyError> {
    // A leading `/` means "the knowledge-base root", not a filesystem path — callers
    // have no way to know where the KB lives inside the container, so treating `/x.md`
    // and `x.md` as the same location is the only reading that makes sense here.
    let rel_path = crate::retrieval::kb_root_relative(rel_path);
    let requested = Path::new(rel_path);
    if requested.is_absolute() {
        return Err(PathSafetyError::Rejected(
            "path must be relative to the knowledge base root".to_string(),
        ));
    }

    // 2. Reject any `..` component.
    for component in requested.components() {
        if component == std::path::Component::ParentDir {
            return Err(PathSafetyError::Rejected(
                "path must not contain '..' components".to_string(),
            ));
        }
    }

    let abs_path = data_root.join(rel_path);

    // 3. Lexical starts_with check.
    if !abs_path.starts_with(data_root) {
        return Err(PathSafetyError::Rejected(
            "path escapes the knowledge base root".to_string(),
        ));
    }

    // 4. Canonical-ancestor check: canonicalize data_root, then walk up from
    //    abs_path to find the deepest ancestor that actually exists on disk,
    //    canonicalize it, and confirm it still sits under canonical data_root.
    let canonical_root = data_root.canonicalize().map_err(|e| {
        PathSafetyError::Internal(format!(
            "cannot canonicalize data root '{}': {}",
            data_root.display(),
            e
        ))
    })?;

    // Walk from abs_path upward until we find an existing ancestor.
    let mut candidate = abs_path.as_path();
    let existing_ancestor = loop {
        if candidate.exists() {
            break candidate;
        }
        match candidate.parent() {
            Some(p) => candidate = p,
            None => {
                // No ancestor exists at all (shouldn't happen since data_root must exist).
                break data_root;
            }
        }
    };

    let canonical_ancestor = existing_ancestor.canonicalize().map_err(|e| {
        PathSafetyError::Internal(format!(
            "cannot canonicalize ancestor '{}': {}",
            existing_ancestor.display(),
            e
        ))
    })?;

    if !canonical_ancestor.starts_with(&canonical_root) {
        return Err(PathSafetyError::Rejected(
            "path escapes the knowledge base root (symlink detected)".to_string(),
        ));
    }

    Ok(abs_path)
}

/// Validate that `rel_path` is safe to write inside `data_root`, returning the
/// absolute target path on success or an error string on failure.
///
/// Moved here (from `mcp.rs`) so this core write module does not depend back on
/// the transport layer it is meant to be independent of — `mcp.rs`'s tool
/// methods call this via `crate::write::resolve_safe_write_path`.
///
/// A thin `String`-returning wrapper over [`resolve_safe_write_path_detailed`]:
/// every caller outside this module predates the `PathSafetyError` split and
/// expects a flat string (several of them — `mcp.rs`'s `create_document` and
/// `write_raw_file` — relay it directly to a trusted MCP client), so this
/// function's behavior and exact message text are unchanged by that split.
pub fn resolve_safe_write_path(data_root: &Path, rel_path: &str) -> Result<PathBuf, String> {
    resolve_safe_write_path_detailed(data_root, rel_path).map_err(PathSafetyError::into_message)
}

/// Resolve `rel_path` against the KB root via
/// [`resolve_safe_write_path_detailed`], mapping a caller-facing rejection onto
/// `WriteError::UnsafePath` and a canonicalize failure onto
/// `WriteError::Internal` (see that variant's doc comment for why the two must
/// not be conflated).
fn safe_write_path<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &WriteDeps<'_, E, Q>,
    rel_path: &str,
) -> Result<PathBuf, WriteError> {
    resolve_safe_write_path_detailed(deps.canonical_data_path, rel_path).map_err(|e| match e {
        PathSafetyError::Rejected(msg) => WriteError::UnsafePath {
            msg: format!("Invalid path: {}", msg),
        },
        PathSafetyError::Internal(msg) => WriteError::Internal {
            msg: format!("Invalid path: {}", msg),
        },
    })
}

/// Reject `rel_path` if it does not match `include_patterns` — a path the
/// indexer would never pick up (e.g. anything outside `**/*.md`, including
/// files under `.git/`) must never be written or deleted through this
/// pipeline, no matter which caller reaches it.
///
/// Before this existed, every transport enforced this itself: MCP's
/// `create_document` checked `include_patterns.is_match` directly, and
/// `edit_document`/`delete_document` got it for free from
/// `retrieval::resolve_within_data`'s `NotPermitted` arm. That meant a new
/// caller (the HTTP UI's write routes, briefly) had to remember to re-derive
/// the same check itself, and a caller that forgot it entirely would let a
/// write reach `commit_and_sync` for a path the indexer would never clean up
/// or even see — including files under `.git/`, where a hostile write is a
/// well-known local code-execution primitive. Enforcing it once, here, means
/// no caller can drop it by omission.
///
/// Reuses `WriteError::UnsafePath` rather than adding a new variant so every
/// existing caller's (necessarily exhaustive) match over `WriteError` keeps
/// compiling unchanged — see `mcp.rs`'s and `web.rs`'s error-mapping functions.
///
/// `pub(crate)` (rather than folded entirely into `check_include_pattern`
/// below) so `mcp.rs`'s `create_document` adapter can run this exact check —
/// same message text — ahead of its own `exists()` pre-check, restoring the
/// pre-refactor error priority when a path both exists on disk and fails this
/// check. See that call site's comment.
pub(crate) fn check_include_pattern_against(
    include_patterns: &globset::GlobSet,
    rel_path: &str,
) -> Result<(), WriteError> {
    let normalized = crate::retrieval::kb_root_relative(rel_path);
    if include_patterns.is_match(normalized) {
        return Ok(());
    }
    Err(WriteError::UnsafePath {
        msg: format!(
            "path '{}' does not match any indexable include pattern \
             (e.g. must be a markdown file under an included path)",
            rel_path
        ),
    })
}

fn check_include_pattern<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &WriteDeps<'_, E, Q>,
    rel_path: &str,
) -> Result<(), WriteError> {
    check_include_pattern_against(deps.retrieval.include_patterns, rel_path)
}

// ---------------------------------------------------------------------------
// (#179) Content-mode helpers: frontmatter patch / append
//
// Both compute a full `new_content` string from the current content, the same
// way `mcp.rs`'s `apply_surgical` does for old_string/new_string — pure
// functions that `mcp.rs` wraps in a `DocChange::Relative` closure, which
// `write_document` runs on the content it reads under `GIT_LOCK`. That keeps
// the core pipeline (schema validation, the dedup gate, the commit, the
// pre-commit rollback) unaware of which kind of relative edit produced the new
// content, with no special-casing anywhere below this point.
// ---------------------------------------------------------------------------

/// A single structured edit to a document's OWN frontmatter, applied by
/// [`apply_frontmatter_patch`].
///
/// Mirrors `schema::SchemaEdit`'s vocabulary
/// (`set_field`/`remove_field`/`add_values`/`remove_values`) — this
/// codebase's established idiom for "a structured edit instead of a
/// free-text patch" (see `update_schema`) — applied here to a document's own
/// frontmatter VALUES rather than a schema's field DECLARATIONS. `field` is a
/// dot-path, same convention `schema::get_by_dotpath`/`set_by_dotpath` and
/// `GetSchemaParams::fields`/`SearchFiltersInput` already use throughout this
/// codebase.
#[derive(Debug, Clone, PartialEq)]
pub enum FrontmatterEdit {
    /// Set (or create) a field to an exact value, replacing whatever was
    /// there. Always succeeds — mirrors `SchemaEdit::SetField`'s unconditional
    /// insert-or-replace.
    SetField {
        field: String,
        value: serde_json::Value,
    },
    /// Remove a field declaration. Errors if the field is not currently set —
    /// mirrors `SchemaEdit::RemoveField`'s identical refusal, so a caller
    /// cannot mistake "nothing to remove" for a silent no-op.
    RemoveField { field: String },
    /// Append values to a list field, creating it (as a new list) if absent,
    /// de-duplicated against what is already there. Errors if the field
    /// exists and is not a list.
    AddValues {
        field: String,
        values: Vec<serde_json::Value>,
    },
    /// Remove values from a list field. Errors if the field is absent or is
    /// not a list — mirrors `SchemaEdit::RemoveValues`'s identical refusal.
    RemoveValues {
        field: String,
        values: Vec<serde_json::Value>,
    },
}

/// Split `content` into `(frontmatter_block, body)` at the byte level,
/// preserving BOTH halves EXACTLY as they appear in `content` — every byte of
/// `content` is accounted for in exactly one of the two halves, concatenating
/// them losslessly reconstructs `content`.
///
/// This is deliberately NOT `validate::parse_frontmatter_raw` (the
/// `gray_matter`-backed parser used everywhere else in this codebase): that
/// function trims trailing whitespace from the body it returns, so its output
/// is not always a byte-exact suffix of the original content — fine for
/// validation (which only cares about the parsed VALUES), fatal here, where
/// [`apply_append`] depends on exactness to guarantee it can never write into
/// the frontmatter block at all, structurally, rather than by carefully
/// avoiding it.
///
/// Delimiter convention: `content` must begin with a `---` line (`\n`- or
/// `\r\n`-terminated); the block ends at the next line that is exactly `---`
/// (optionally `\r`-terminated), found by scanning line-by-line so a literal
/// `---` inside a YAML value (e.g. a horizontal-rule in a `description`
/// string) can never be mistaken for the closing delimiter. Returns `None`
/// when `content` has no opening delimiter, or an opening delimiter with no
/// matching close — both read as "no frontmatter block", the same as
/// `gray_matter` treats them.
fn split_frontmatter_bytes(content: &str) -> Option<(&str, &str)> {
    let after_open = content
        .strip_prefix("---\r\n")
        .or_else(|| content.strip_prefix("---\n"))?;

    let mut offset = 0usize;
    for line in after_open.split_inclusive('\n') {
        let trimmed = line.strip_suffix('\n').unwrap_or(line);
        let trimmed = trimmed.strip_suffix('\r').unwrap_or(trimmed);
        if trimmed == "---" {
            let close_end = offset + line.len();
            let fm_len = (content.len() - after_open.len()) + close_end;
            return Some((&content[..fm_len], &content[fm_len..]));
        }
        offset += line.len();
    }
    None
}

/// Remove the value at a dot-path, mirroring `schema::get_by_dotpath`'s own
/// traversal. Not in `schema.rs` alongside its `get`/`set` siblings because
/// this module's remit is deliberately kept to `write.rs`/`mcp.rs`/`web.rs`
/// (see this crate's write-pipeline module boundaries) — the two sibling
/// functions are schema-declaration helpers `schema.rs` owns for its own
/// `apply_defaults`; this one is document-frontmatter-only. Returns whether
/// anything was actually removed.
fn remove_by_dotpath(frontmatter: &mut HashMap<String, serde_json::Value>, path: &str) -> bool {
    let segments: Vec<&str> = path.split('.').collect();
    if segments.len() == 1 {
        return frontmatter.remove(segments[0]).is_some();
    }
    let Some(mut cursor) = frontmatter
        .get_mut(segments[0])
        .and_then(|v| v.as_object_mut())
    else {
        return false;
    };
    for segment in &segments[1..segments.len() - 1] {
        let Some(next) = cursor.get_mut(*segment).and_then(|v| v.as_object_mut()) else {
            return false;
        };
        cursor = next;
    }
    cursor.remove(segments[segments.len() - 1]).is_some()
}

/// Render a frontmatter map back to a `---`-delimited YAML block (including
/// both delimiters and the trailing newline after the closing one).
///
/// This is the whole-block path: used when a document has no frontmatter yet,
/// and as the fallback when [`splice_frontmatter_block`] cannot safely edit
/// the author's own block in place (#269). Everything below describes what
/// THIS path costs; the splice path preserves order, comments and block
/// scalars for fields the patch did not touch.
///
/// Keys are sorted (`BTreeMap`) before serializing — the exact same reasoning
/// as `SchemaFile::to_yaml`'s identical `BTreeMap` conversion: `HashMap`
/// iteration order is unspecified (and randomized per-process), so
/// serializing straight from it would reorder every field on every patch, for
/// no reason connected to what actually changed. Sorting trades that for a
/// deterministic, minimal diff — at the cost of NOT preserving whatever key
/// order (or comments) the document's own frontmatter happened to have,
/// exactly the same trade-off this codebase already made for
/// schema file when `update_schema` rewrites one.
fn render_frontmatter_block(
    frontmatter: &HashMap<String, serde_json::Value>,
    newline: &str,
) -> Result<String, String> {
    let ordered: BTreeMap<&String, &serde_json::Value> = frontmatter.iter().collect();
    let yaml = serde_yaml_ng::to_string(&ordered)
        .map_err(|e| format!("failed to serialize frontmatter: {e}"))?;
    if newline == "\n" {
        return Ok(format!("---\n{yaml}---\n"));
    }
    // serde_yaml_ng always emits LF. Re-terminate every line with the
    // document's own ending so a patch does not silently convert a CRLF file
    // into a mixed-ending one — the bytes would all still be there, but every
    // subsequent diff of that document would show the whole frontmatter block
    // as changed.
    let converted: String = yaml
        .split_inclusive('\n')
        .map(|line| format!("{}{newline}", line.trim_end_matches('\n')))
        .collect();
    Ok(format!("---{newline}{converted}---{newline}"))
}

/// Render one top-level frontmatter entry (`key: value`, however many lines
/// the value needs), every line terminated with `newline`.
fn render_frontmatter_entry(
    key: &str,
    value: &serde_json::Value,
    newline: &str,
) -> Result<String, String> {
    let single: BTreeMap<&str, &serde_json::Value> = BTreeMap::from([(key, value)]);
    let yaml = serde_yaml_ng::to_string(&single)
        .map_err(|e| format!("failed to serialize frontmatter: {e}"))?;
    Ok(yaml
        .split_inclusive('\n')
        .map(|line| format!("{}{newline}", line.trim_end_matches('\n')))
        .collect())
}

/// The key a top-level frontmatter line introduces, or `None` when the line
/// is not a plain `key:` line this splicer understands.
///
/// Only block-style mappings are recognized: a line opening with a flow
/// (`{`/`[`), anchor, alias, tag, directive or block-scalar indicator is not
/// a key line, so the caller reads it as "unsure" and falls back.
fn top_level_key(line: &str) -> Option<String> {
    let first = line.chars().next()?;
    if first.is_whitespace() || "#-{[&*!?|>%@`,]}".contains(first) {
        return None;
    }
    if first == '"' || first == '\'' {
        let close = line[1..].find(first)? + 1;
        let rest = &line[close + 1..];
        if !(rest.starts_with(':') && (rest.len() == 1 || rest[1..].starts_with(' '))) {
            return None;
        }
        return serde_yaml_ng::from_str::<String>(&line[..=close]).ok();
    }
    let colon = line
        .char_indices()
        .find(|&(i, c)| c == ':' && (i + 1 == line.len() || line[i + 1..].starts_with(' ')))?
        .0;
    let key = line[..colon].trim_end();
    (!key.is_empty() && key != "<<").then(|| key.to_string())
}

/// Rewrite `old_block` (the original `---`…`---` frontmatter, byte-exact) so
/// that only the top-level fields that differ between `original` and
/// `updated` change, keeping every other byte — key order, comments, block
/// scalars, flow lists, blank lines — exactly as authored.
///
/// A top-level entry is its `key:` line plus every following line up to the
/// next `key:` line (so a block scalar's body, a nested map and a block list
/// all travel with their key). Trailing blank and `#` comment lines of an
/// entry are kept when the entry is re-rendered or removed, since they
/// usually describe what follows. A changed field is re-rendered in place with
/// `serde_yaml_ng`, so a nested edit (`planning.role`) re-renders its whole
/// top-level field; a new field is appended in sorted order; a removed one is
/// dropped.
///
/// Returns `None` whenever the block is not plainly segmentable (a line at
/// column 0 that is neither a key, comment nor list item; tab indentation;
/// duplicate keys; a key the parsed map disagrees about). The caller treats
/// that as "fall back to a full re-render", and also re-parses whatever this
/// returns before trusting it.
fn splice_frontmatter_block(
    old_block: &str,
    original: &HashMap<String, serde_json::Value>,
    updated: &HashMap<String, serde_json::Value>,
    newline: &str,
) -> Option<String> {
    let lines: Vec<&str> = old_block.split_inclusive('\n').collect();
    let (open, rest) = lines.split_first()?;
    let (close, inner) = rest.split_last()?;
    if !close.trim_end_matches(['\r', '\n']).eq("---") {
        return None;
    }

    // (key, lines) per entry; `preamble` is whatever precedes the first key.
    let mut preamble = String::new();
    let mut entries: Vec<(String, Vec<&str>)> = Vec::new();
    for line in inner {
        let bare = line.trim_end_matches(['\r', '\n']);
        if bare.starts_with('\t') || bare.starts_with(" \t") {
            return None;
        }
        if let Some(key) = top_level_key(bare) {
            if entries.iter().any(|(k, _)| *k == key) {
                return None;
            }
            entries.push((key, vec![line]));
            continue;
        }
        let first = bare.chars().next();
        let continuation = bare.trim().is_empty()
            || first.is_some_and(|c| c.is_whitespace() || c == '#' || c == '-');
        if !continuation {
            return None;
        }
        match entries.last_mut() {
            Some((_, body)) => body.push(line),
            None if bare.trim().is_empty() || first == Some('#') => preamble.push_str(line),
            None => return None,
        }
    }

    // The segmentation must agree with the parsed map about which keys exist.
    if entries.len() != original.len() || entries.iter().any(|(k, _)| !original.contains_key(k)) {
        return None;
    }

    let mut out = String::from(*open);
    out.push_str(&preamble);
    for (key, body) in &entries {
        let unchanged = original.get(key) == updated.get(key);
        if unchanged {
            body.iter().for_each(|l| out.push_str(l));
            continue;
        }
        let tail_start = body
            .iter()
            .rposition(|l| {
                let b = l.trim_end_matches(['\r', '\n']);
                !(b.trim().is_empty() || b.starts_with('#'))
            })
            .map_or(1, |i| i + 1);
        if let Some(value) = updated.get(key) {
            out.push_str(&render_frontmatter_entry(key, value, newline).ok()?);
        }
        body[tail_start..].iter().for_each(|l| out.push_str(l));
    }
    let mut added: Vec<&String> = updated
        .keys()
        .filter(|k| !original.contains_key(*k))
        .collect();
    added.sort();
    for key in added {
        out.push_str(&render_frontmatter_entry(key, &updated[key], newline).ok()?);
    }
    out.push_str(close);
    Some(out)
}

/// The line ending `content` uses, for round-tripping a rewrite through it.
///
/// Decided by the first ending actually present, not by a majority vote: a
/// document with mixed endings is already inconsistent, and picking its first
/// one at least keeps a rewrite from making the inconsistency worse. Content
/// with no newline at all gets LF, matching what this project writes by
/// default everywhere else.
fn detect_newline(content: &str) -> &'static str {
    match content.find('\n') {
        Some(i) if i > 0 && content.as_bytes()[i - 1] == b'\r' => "\r\n",
        _ => "\n",
    }
}

/// Apply a structured frontmatter patch to `old_content`, returning the full
/// new document content (frontmatter block + body, byte-identical body).
///
/// Parses the existing frontmatter via `validate::parse_frontmatter_raw` (the
/// same basis `write_document`'s own validation step re-parses immediately
/// afterward — see the doc comment on this module's content-mode-helpers
/// section for why that duplication is fine: this function's OUTPUT is just
/// ordinary `new_content`, re-validated from scratch like any other write),
/// applies each edit in order, then re-serializes and reattaches the ORIGINAL
/// body untouched — this function never reads, modifies, or even fully
/// re-parses the body, so it cannot corrupt it, structurally, not just by
/// convention.
///
/// Handles a document with no existing frontmatter block by creating one
/// (mirrors `SchemaEdit::AddValues`'s "creating the field if absent"): the
/// whole original `old_content` becomes the body, separated from the new
/// frontmatter block by a blank line (unless the body is empty, in which case
/// no trailing blank line is added either).
///
/// `write_document` calls this on the content it read under `GIT_LOCK`, after
/// syncing to the remote, so the body reattached here is always the current
/// one: a concurrent body edit is carried along rather than overwritten, and no
/// patch-specific stale-read handling is needed.
pub fn apply_frontmatter_patch(
    old_content: &str,
    edits: &[FrontmatterEdit],
) -> Result<String, String> {
    let (fm_block, body) = split_frontmatter_bytes(old_content).unwrap_or(("", old_content));
    let had_frontmatter = !fm_block.is_empty();

    let (mut frontmatter, _) = validate::parse_frontmatter_raw(old_content);
    let original = frontmatter.clone();

    for edit in edits {
        match edit {
            FrontmatterEdit::SetField { field, value } => {
                schema::set_by_dotpath(&mut frontmatter, field, value.clone());
            }
            FrontmatterEdit::RemoveField { field } => {
                if !remove_by_dotpath(&mut frontmatter, field) {
                    return Err(format!(
                        "field '{field}' is not set in this document's frontmatter"
                    ));
                }
            }
            FrontmatterEdit::AddValues { field, values } => {
                let mut existing: Vec<serde_json::Value> =
                    match schema::get_by_dotpath(&frontmatter, field) {
                        Some(serde_json::Value::Array(arr)) => arr.clone(),
                        Some(_) => {
                            return Err(format!(
                                "field '{field}' is not a list in this document's frontmatter"
                            ));
                        }
                        None => Vec::new(),
                    };
                for v in values {
                    if !existing.contains(v) {
                        existing.push(v.clone());
                    }
                }
                schema::set_by_dotpath(&mut frontmatter, field, serde_json::Value::Array(existing));
            }
            FrontmatterEdit::RemoveValues { field, values } => {
                let existing = match schema::get_by_dotpath(&frontmatter, field) {
                    Some(serde_json::Value::Array(arr)) => arr.clone(),
                    Some(_) => {
                        return Err(format!(
                            "field '{field}' is not a list in this document's frontmatter"
                        ));
                    }
                    None => {
                        return Err(format!(
                            "field '{field}' has no value list in this document's frontmatter"
                        ));
                    }
                };
                let filtered: Vec<serde_json::Value> = existing
                    .into_iter()
                    .filter(|v| !values.contains(v))
                    .collect();
                schema::set_by_dotpath(&mut frontmatter, field, serde_json::Value::Array(filtered));
            }
        }
    }

    // Splice the edit into the author's own block when that is provably
    // safe; the re-parse is what makes it provable, so a segmentation the
    // splicer got wrong degrades to the full re-render instead of corrupting.
    let newline = detect_newline(old_content);
    let spliced = if had_frontmatter {
        splice_frontmatter_block(fm_block, &original, &frontmatter, newline)
            .filter(|block| validate::parse_frontmatter_raw(block).0 == frontmatter)
    } else {
        None
    };
    let new_fm_block = match spliced {
        Some(block) => block,
        None => render_frontmatter_block(&frontmatter, newline)?,
    };

    if had_frontmatter {
        Ok(format!("{new_fm_block}{body}"))
    } else if body.is_empty() {
        Ok(new_fm_block)
    } else {
        Ok(format!("{new_fm_block}\n{body}"))
    }
}

/// Append `text` to the end of `old_content`'s BODY — never past the
/// frontmatter block, structurally guaranteed by reusing
/// [`split_frontmatter_bytes`] rather than a substring/offset computed by
/// hand: whatever that function calls the frontmatter block is copied through
/// completely untouched, and `text` only ever lands inside whatever it calls
/// the body.
///
/// Exactly one newline separates existing body content from `text` — this
/// function does not fabricate blank-line spacing beyond that (a caller that
/// wants a blank line before its entry includes the leading newline in
/// `text` itself; see `write_document.md`), except for the one case where
/// there is no separator to reuse at all: a frontmatter block with an empty
/// body gets a single blank line before `text`, so the appended content does
/// not land glued to the closing `---`.
///
/// Handles a document with no frontmatter block (appends to the whole
/// content), an empty file (the result is just `text`), and a body with no
/// trailing newline (one is inserted before appending) — see this function's
/// tests for each case.
pub fn apply_append(old_content: &str, text: &str) -> String {
    let (fm_block, body) = split_frontmatter_bytes(old_content).unwrap_or(("", old_content));
    // Match the document's own line ending rather than always emitting LF —
    // otherwise the first append to a CRLF document glues an LF-terminated
    // block onto it and leaves the file with mixed endings.
    let nl = detect_newline(old_content);

    let mut new_body = body.to_string();
    if !new_body.is_empty() && !new_body.ends_with('\n') {
        new_body.push_str(nl);
    }
    if !fm_block.is_empty() && new_body.is_empty() {
        new_body.push_str(nl);
    }
    // Normalize the caller's text to the document's ending too: an agent
    // composing an append has no idea what the file on disk uses.
    let appended: String = text
        .trim_end_matches('\n')
        .trim_end_matches('\r')
        .split_inclusive('\n')
        .map(|line| {
            let stripped = line.trim_end_matches('\n').trim_end_matches('\r');
            if line.ends_with('\n') {
                format!("{stripped}{nl}")
            } else {
                stripped.to_string()
            }
        })
        .collect();
    new_body.push_str(&appended);
    new_body.push_str(nl);

    format!("{fm_block}{new_body}")
}

// ---------------------------------------------------------------------------
// create_document / edit_document core
// ---------------------------------------------------------------------------

/// Refuse `rel_path` when it names a schema file — see
/// [`WriteError::SchemaFile`]. Checked first by every document write, delete and
/// single-document move, independently of `indexing.include` (a widened include
/// must not turn schema files into writable documents).
fn check_not_schema_file(rel_path: &str) -> Result<(), WriteError> {
    if crate::schema::is_schema_file_path(Path::new(rel_path)) {
        return Err(WriteError::SchemaFile {
            rel_path: rel_path.to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Concurrent edits: version token, lock-then-sync, re-apply, three-way merge
// ---------------------------------------------------------------------------
//
// Every read-modify-write runs under ONE `GitLock` acquisition that begins with a
// fetch + fast-forward to the remote (`git::sync_to_remote`), then reads the
// document, applies the change, validates, writes, commits and pushes. No
// document content a write computes from is read outside that acquisition, so a
// concurrent writer in this process (serialized by the lock) or on the remote
// (pulled in by the sync) is never silently overwritten. Validation is the
// exception: it uses the shared schema cache as it stands, and a schema file the
// sync pulls in only queues a full reconcile (`mark_dirty`) whose worker rebuilds
// the cache later — so a write racing someone else's schema push is validated
// against the rules from before that push.
//
// - A relative change (surgical replace, frontmatter patch, append) is applied to
//   the content read under the lock. If its anchor no longer matches, it is
//   refused as edited-by-someone-else; if it applies over changes made since the
//   caller's `expected_version`, it lands and is reported as merged.
// - An absolute change to an existing document (full replace, delete, move)
//   requires the caller's `expected_version`. A stale full replace is three-way
//   merged against the caller's base version (`git merge-file`); a clean merge is
//   written and reported, a conflict refused. A stale delete or move is refused.
//   A create carrying an `expected_version` is refused as not found.
// - An edit that leaves the document exactly as it is commits nothing and is
//   reported as it stands.
// - A push the remote rejects, or a rebase that conflicts, drops that attempt's
//   own commit (`git::CommitSyncError::Conflict`; a commit an earlier outage left
//   unpushed stays), syncs the clone with the remote again (`sync_clone`) and runs
//   the whole attempt again against fresh content, up to `MAX_WRITE_ATTEMPTS`
//   times; then the write is refused. A remote that is merely unreachable is not a
//   conflict: the local commit stands and syncs with a later write
//   (`CommitSyncError::PostCommit`), so an outage never blocks writes.

/// The refusal for a change that no longer fits the document as it now is.
pub const EDITED_ELSEWHERE: &str =
    "the document was edited by someone else; re-read it and try again";

/// How many times a write re-applies its change after the remote moved under it
/// (a rejected push or a rebase conflict) before refusing.
pub(crate) const MAX_WRITE_ATTEMPTS: usize = 3;

/// The largest document a write may produce. The transport adapters refuse
/// created or replaced content over it; [`apply_change`] refuses a relative edit
/// whose result would grow a document past it, while still letting an edit shrink
/// a document that is already larger.
pub const MAX_CONTENT_LEN: usize = 512 * 1024;

/// The caller-facing version of a document's bytes, as `get_document` reports it
/// and `expected_version` echoes it back. Opaque to callers (it is the git blob
/// id, which is also what lets a stale full replace find its merge base); distinct
/// from the indexer's `content_hash`, which change detection keeps using.
pub fn document_version(bytes: &[u8]) -> String {
    git::blob_id(bytes)
}

/// A relative change: computes the new content from the current content, or
/// explains why it cannot (an anchor that does not match).
pub type RelativeEdit<'a> = dyn Fn(&str) -> Result<String, String> + Send + Sync + 'a;

/// What a write does to a document's content. Relative changes are re-applied
/// to whatever the document holds under the lock; absolute ones replace it and so
/// need the caller's `expected_version` when the document exists.
#[derive(Clone, Copy)]
pub enum DocChange<'a> {
    /// Create a new document with this content.
    Create(&'a str),
    /// Replace an existing document's whole content.
    Replace(&'a str),
    /// Surgical replace, frontmatter patch, append — applied to current content.
    Relative(&'a RelativeEdit<'a>),
    /// Keep the content as it is (a pure move).
    Keep,
}

impl DocChange<'_> {
    pub fn is_create(&self) -> bool {
        matches!(self, DocChange::Create(_))
    }

    /// Whether this change overwrites an existing document wholesale and so
    /// requires `expected_version`.
    pub fn is_absolute(&self) -> bool {
        matches!(self, DocChange::Replace(_) | DocChange::Keep)
    }
}

/// New content computed under the lock, and whether it folds in someone else's
/// concurrent change.
struct Applied {
    content: String,
    merged: bool,
}

fn data_path_of<'a, E: QueryEmbedder, Q: RetrievalStore>(deps: &WriteDeps<'a, E, Q>) -> &'a str {
    deps.canonical_data_path.to_str().unwrap_or_default()
}

/// Sync the clone to the remote (see `git::sync_to_remote`) and mark every path
/// that changed since `since` (default: HEAD right now) dirty — the webhook that
/// will follow those remote commits finds the clone already up to date and would
/// mark nothing. Returns HEAD after the sync, the baseline for the next attempt.
///
/// A schema file among those paths queues a full reconcile; the shared schema
/// cache is rebuilt by the worker, not here, so the write in flight still
/// validates against the cache as it was.
pub(crate) async fn sync_clone<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &WriteDeps<'_, E, Q>,
    lock: &git::GitLock,
    since: Option<String>,
) -> Option<String> {
    let data_path = data_path_of(deps);
    let before = match since {
        Some(sha) => Some(sha),
        None => git::head_if_repo(lock, data_path).await,
    };
    git::sync_to_remote(
        lock,
        deps.git_url,
        deps.branch,
        data_path,
        deps.token,
        deps.commit_author_name,
        deps.commit_author_email,
    )
    .await;
    let after = git::head_if_repo(lock, data_path).await;
    if let (Some(b), Some(a)) = (&before, &after)
        && b != a
    {
        match git::git_diff_name_status(lock, data_path, b, a).await {
            Ok(paths) => mark_dirty(deps.queue, deps.indexing, paths),
            Err(e) => {
                warn!(
                    "Could not diff the paths a pre-write sync pulled in; queuing a full reconcile: {e:#}"
                );
                deps.queue.mark_full();
            }
        }
    }
    after
}

/// Take `GIT_LOCK` and bring the clone up to date with the remote: the start of
/// every read-modify-write. Returns the guard and HEAD after the sync.
pub(crate) async fn lock_and_sync<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &WriteDeps<'_, E, Q>,
) -> (git::GitLock, Option<String>) {
    git::test_hook(git::HookPoint::BeforeLock).await;
    let lock = git::lock_git().await;
    let head = sync_clone(deps, &lock, None).await;
    (lock, head)
}

fn versions_match(expected: &str, current: &str) -> bool {
    expected
        .trim()
        .eq_ignore_ascii_case(&document_version(current.as_bytes()))
}

/// Compute the new content for `change` from `current`, the document as read
/// under the lock. `pre_read` is the document as read before the lock, which tells
/// a stale anchor (it matched then, the document changed since) from one that
/// never matched.
async fn apply_change(
    lock: &git::GitLock,
    data_path: &str,
    change: DocChange<'_>,
    current: &str,
    expected_version: Option<&str>,
    pre_read: Option<&str>,
) -> Result<Applied, WriteError> {
    let applied = |content: String| Applied {
        content,
        merged: false,
    };
    match change {
        DocChange::Create(content) => Ok(applied(content.to_string())),
        DocChange::Keep => match expected_version {
            None => Err(WriteError::VersionRequired),
            Some(e) if versions_match(e, current) => Ok(applied(current.to_string())),
            Some(_) => Err(WriteError::EditedElsewhere),
        },
        DocChange::Replace(content) => match expected_version {
            None => Err(WriteError::VersionRequired),
            Some(e) if versions_match(e, current) => Ok(applied(content.to_string())),
            Some(e) => {
                // The cause (git stderr, which names the clone's own path) stays in
                // the log; the caller re-reads and resubmits, as after a conflict.
                let failed = |e: anyhow::Error| {
                    warn!("Three-way merge failed: {e:#}");
                    WriteError::EditedElsewhere
                };
                let base = git::cat_blob(lock, data_path, &e.trim().to_ascii_lowercase())
                    .await
                    .map_err(failed)?;
                // A version the object store has never seen cannot be a base.
                let Some(base) = base else {
                    return Err(WriteError::EditedElsewhere);
                };
                match git::merge_text(
                    lock,
                    data_path,
                    &base,
                    current.as_bytes(),
                    content.as_bytes(),
                )
                .await
                .map_err(failed)?
                {
                    Some(merged) => Ok(Applied {
                        content: merged,
                        merged: true,
                    }),
                    None => Err(WriteError::EditedElsewhere),
                }
            }
        },
        DocChange::Relative(edit) => match edit(current) {
            Ok(content) => {
                if content.len() > MAX_CONTENT_LEN && content.len() > current.len() {
                    return Err(WriteError::InvalidEdit {
                        msg: format!(
                            "the edited document would be too large ({} bytes); maximum is {} \
                             bytes",
                            content.len(),
                            MAX_CONTENT_LEN
                        ),
                    });
                }
                Ok(Applied {
                    content,
                    // Applied to the content under the lock, a relative edit lands
                    // even when `expected_version` is stale — and its result then
                    // carries the changes made since that version as well.
                    merged: expected_version.is_some_and(|e| !versions_match(e, current)),
                })
            }
            Err(msg) => {
                let stale = match expected_version {
                    Some(e) => !versions_match(e, current),
                    None => pre_read.is_some_and(|p| p != current && edit(p).is_ok()),
                };
                Err(if stale {
                    WriteError::EditedElsewhere
                } else {
                    WriteError::InvalidEdit { msg }
                })
            }
        },
    }
}

/// Validate `content` against the schema governing `rel_path`.
async fn validate_document<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &WriteDeps<'_, E, Q>,
    rel_path: &str,
    content: &str,
) -> Result<Option<validate::ValidatedFile>, WriteError> {
    // The shared cache only ever holds a tree in which every schema file was
    // valid: a runtime rebuild that hit an invalid one was refused and the last
    // good cache kept (`schema::apply_rebuild`). It is the cache as it stands: a
    // schema file this write's own pre-write sync pulled in only queues a full
    // reconcile, whose worker swaps the rebuilt cache in later.
    let schemas = crate::schema::load_shared(deps.schema_cache);
    let schema = schemas.resolve_for(Path::new(rel_path));
    let (mut result, validated) =
        validate::validate_content(Path::new(rel_path), content, schema, deps.validation)
            .await
            .map_err(|e| {
                error!("Validation error for '{}': {:#}", rel_path, e);
                WriteError::Io {
                    msg: format!("Failed to validate content: {}", e),
                }
            })?;
    // A field ingest derives (`ingest::DERIVED_FIELDS`) would be silently
    // overridden at index time, so authoring one is refused here rather than
    // announced up front in the server instructions. Write path only: ingest
    // and the CLI `validate` keep accepting existing documents that carry one.
    // Judged on the frontmatter as written, before schema defaults (which never
    // fill a derived field anyway), and whether or not other rules failed too.
    if deps.validation.enabled
        && let Some((field, message)) =
            crate::ingest::authored_derived_field(&validate::parse_frontmatter_raw(content).0)
    {
        result.valid = false;
        result.errors.push(message.clone());
        result.field_errors.push(validate::FieldError {
            field: field.to_string(),
            rule: "derived".to_string(),
            message,
            got: None,
            expected: None,
            schema_origin: None,
        });
    }
    if !result.valid {
        return Err(WriteError::Validation { result });
    }
    Ok(validated)
}

/// The create-path dedup gate: refuse when an existing document is a near
/// duplicate of `validated`. Runs before the lock — it embeds and queries
/// Qdrant, which can take a while — and only for a create, whose content is fixed.
async fn dedup_gate<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &WriteDeps<'_, E, Q>,
    rel_path: &str,
    validated: Option<&validate::ValidatedFile>,
    force_new: Option<bool>,
) -> Result<(), WriteError> {
    let schemas = crate::schema::load_shared(deps.schema_cache);
    let schema = schemas.resolve_for(Path::new(rel_path));
    let (dedup_enabled, dedup_threshold) = effective_dedup(deps, schema);
    if !dedup_enabled || matches!(force_new, Some(true)) {
        return Ok(());
    }
    // The body already parsed during validation keeps the dedup query on exactly
    // the frontmatter-stripped basis the indexer embeds.
    let query_text = validated
        .map(|v| {
            let description = v.frontmatter.get("description").and_then(|d| d.as_str());
            build_dedup_query(&v.body, description, deps.prepend_description)
        })
        .unwrap_or_default();
    if query_text.trim().is_empty() {
        warn!(
            "Dedup gate skipped for '{}': no body text to compare",
            rel_path
        );
        return Ok(());
    }
    // Detach the reranker: `dedup_threshold` is a cosine similarity, and a
    // cross-encoder relevance score is not comparable to it.
    let dedup_deps = RetrievalDeps {
        embed_client: deps.retrieval.embed_client,
        qdrant: deps.retrieval.qdrant,
        collection: deps.retrieval.collection,
        data_path: deps.retrieval.data_path,
        include_patterns: deps.retrieval.include_patterns,
        reranker: None,
    };
    match crate::retrieval::search(
        &dedup_deps,
        &query_text,
        &SearchFilters::default(),
        &dedup_search_opts(),
    )
    .await
    {
        Ok(results) => {
            let top = results.into_iter().next().map(|r| {
                let path = r
                    .payload
                    .get("file_path")
                    .and_then(|v| v.as_str())
                    .map(|p| crate::retrieval::relative_to_data(p, deps.canonical_data_path))
                    .unwrap_or_default();
                (path, r.score)
            });
            if let Some((path, score)) = top.as_ref() {
                tracing::debug!(
                    "Dedup gate for '{}': nearest '{}' at dense cosine {:.4} (threshold {:.2})",
                    rel_path,
                    path,
                    score,
                    dedup_threshold
                );
            }
            if let Some(hit) = dedup_verdict(top, dedup_threshold) {
                return Err(WriteError::DedupHit {
                    duplicate_of: hit.file_path,
                    similarity: hit.score,
                    threshold: dedup_threshold,
                });
            }
        }
        Err(e) => {
            warn!(
                "Dedup search failed for '{}' (proceeding with write): {:#?}",
                rel_path, e
            );
        }
    }
    Ok(())
}

/// Read a document, `None` when it does not exist.
async fn read_if_exists(abs_path: &Path) -> Result<Option<String>, WriteError> {
    match tokio::fs::read_to_string(abs_path).await {
        Ok(c) => Ok(Some(c)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => {
            error!("Failed to read '{}': {}", abs_path.display(), e);
            Err(WriteError::Io {
                msg: format!("Failed to read existing file: {}", e),
            })
        }
    }
}

/// Write all of `bytes` to a file just opened for writing, then flush it.
/// `tokio::fs::File` hands the bytes to a blocking task and reports that task's
/// failure only on the next write or flush, so a file dropped straight after
/// `write_all` loses its last write error, and may still be writing when the
/// caller moves on to `git add`.
async fn write_and_flush(file: &mut tokio::fs::File, bytes: &[u8]) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt as _;
    file.write_all(bytes).await?;
    file.flush().await
}

/// Run `write`, a document's one filesystem write, and when it fails undo what it
/// may have left behind before reporting the failure: put `previous` back over an
/// existing document (`fs::write` truncates before it writes, so an ENOSPC or EIO
/// part-way leaves it truncated), or remove a partly written new file (`previous`
/// is `None`). Either leftover is an uncommitted change in the clone, and a dirty
/// tracked file fails every later pre-write rebase. A failed undo is logged; the
/// caller gets the write's own error.
async fn write_or_undo(
    abs_path: &Path,
    previous: Option<&str>,
    write: impl std::future::Future<Output = std::io::Result<()>>,
) -> Result<(), WriteError> {
    let Err(e) = write.await else {
        return Ok(());
    };
    error!("Failed to write file '{}': {}", abs_path.display(), e);
    let undone = match previous {
        Some(previous) => tokio::fs::write(abs_path, previous.as_bytes()).await,
        None => tokio::fs::remove_file(abs_path).await,
    };
    if let Err(undo_err) = undone {
        error!(
            "Failed to undo the failed write to '{}': {}. It is left as that write left it, \
             uncommitted; this needs operator attention.",
            abs_path.display(),
            undo_err
        );
    }
    Err(WriteError::Io {
        msg: format!("Failed to write file: {}", e),
    })
}

/// Whether the rebase in `commit_and_sync` folded someone else's change to one
/// of this write's own paths into its commit (a clean, non-overlapping merge).
fn rebase_touched(rebased: &[PathBuf], own: &[&str]) -> bool {
    rebased
        .iter()
        .any(|p| own.iter().any(|o| Path::new(o) == p.as_path()))
}

/// Shared pipeline for a create or edit write.
///
/// Handles the schema-file guard, validation, the create-only dedup gate, the
/// lock-then-sync read-modify-write (see the section comment above), filesystem
/// write, git commit, reindex queuing, and diff output. Callers resolve
/// `req.rel_path` and describe the change; the new content is computed here,
/// from the document as it is under the lock.
///
/// The absolute path is re-resolved from `rel_path` immediately before each
/// filesystem action rather than computed once up front, so a path validated
/// earlier in this call cannot go stale across the awaits in between.
pub async fn write_document<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &WriteDeps<'_, E, Q>,
    req: WriteRequest<'_>,
) -> Result<WriteSuccess, WriteError> {
    // A move touches two paths at nearly every step — see `write_document_move`.
    if req.dest_path.is_some() {
        return write_document_move(deps, req).await;
    }

    let WriteRequest {
        rel_path,
        change,
        message,
        default_verb,
        force_new,
        operation,
        expected_version,
        dest_path: _,
    } = req;
    let is_create = change.is_create();

    // 0. Schema-file guard, then include-pattern eligibility guard: reject paths
    //    the indexer would not pick up, before anything else runs. See
    //    `check_include_pattern`'s doc comment for why this must live here rather
    //    than in each caller.
    check_not_schema_file(rel_path)?;
    check_include_pattern(deps, rel_path)?;

    // 0.5. Early path-safety check, before validation: when
    // `validation.lint_command` is configured it execs the lint program with
    // `rel_path` as an argument, and `GlobSet::is_match` (the include check
    // above) accepts `..` segments as ordinary path characters.
    safe_write_path(deps, rel_path)?;

    if matches!(change, DocChange::Keep) {
        return Err(WriteError::Internal {
            msg: "write_document called with no content change and no destination".to_string(),
        });
    }
    if change.is_absolute() && expected_version.is_none() {
        return Err(WriteError::VersionRequired);
    }
    // A create carrying `expected_version` was meant for a document the caller
    // read, which is no longer at `rel_path` (deleted or moved since, or never
    // there): refused rather than silently created anew.
    if is_create && expected_version.is_some() {
        return Err(WriteError::NotFound);
    }

    // 1. A create's content is fixed, so it is validated and dedup-gated before
    //    the lock: the gate's embedding call and Qdrant query are the one slow step
    //    deliberately kept outside it. Every other change is validated under the
    //    lock, against the content it actually computes there.
    if let DocChange::Create(content) = change {
        let validated = validate_document(deps, rel_path, content).await?;
        dedup_gate(deps, rel_path, validated.as_ref(), force_new).await?;
    }

    // 2. Validate the commit message BEFORE touching the filesystem.
    validate_commit_message(message)?;

    // 3. The document as it was before the lock — only to tell a stale anchor
    //    from a wrong one (`apply_change`); never what the change is applied to.
    let pre_read = if is_create {
        None
    } else {
        read_if_exists(&safe_write_path(deps, rel_path)?).await?
    };

    let (git_lock, mut head) = lock_and_sync(deps).await;
    let data_path_str = data_path_of(deps);

    for attempt in 1..=MAX_WRITE_ATTEMPTS {
        // Resolve fresh before creating directories: a concurrent git sync could
        // have swapped a component for a symlink since the early check.
        let abs_path = safe_write_path(deps, rel_path)?;
        if let Some(parent) = abs_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                error!(
                    "Failed to create parent directories for '{}': {}",
                    abs_path.display(),
                    e
                );
                WriteError::Io {
                    msg: format!("Failed to create parent directories: {}", e),
                }
            })?;
        }
        let abs_path = safe_write_path(deps, rel_path)?;

        let (old_content, applied) = if let DocChange::Create(content) = change {
            let mut file = tokio::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&abs_path)
                .await
                .map_err(|e| {
                    if e.kind() == std::io::ErrorKind::AlreadyExists {
                        WriteError::AlreadyExists
                    } else {
                        error!("Failed to create file '{}': {}", abs_path.display(), e);
                        WriteError::Io {
                            msg: format!("Failed to create file: {}", e),
                        }
                    }
                })?;
            write_or_undo(&abs_path, None, async move {
                write_and_flush(&mut file, content.as_bytes()).await
            })
            .await?;
            (
                String::new(),
                Applied {
                    content: content.to_string(),
                    merged: false,
                },
            )
        } else {
            let Some(current) = read_if_exists(&abs_path).await? else {
                return Err(WriteError::NotFound);
            };
            let applied = apply_change(
                &git_lock,
                data_path_str,
                change,
                &current,
                expected_version,
                pre_read.as_deref(),
            )
            .await?;
            validate_document(deps, rel_path, &applied.content).await?;
            // An edit that leaves the document exactly as it is has nothing to
            // commit (`git commit` refuses an empty change): the document already
            // says what the caller asked for, so report it as it stands.
            if applied.content == current {
                return Ok(WriteSuccess {
                    outcome: WriteOutcome::Synced,
                    sha: head.unwrap_or_default(),
                    rebased_paths: Vec::new(),
                    diff: String::new(),
                    rewritten_paths: Vec::new(),
                    referencing_paths: Vec::new(),
                    merged: applied.merged,
                    version: Some(document_version(current.as_bytes())),
                });
            }
            write_or_undo(
                &abs_path,
                Some(&current),
                tokio::fs::write(&abs_path, applied.content.as_bytes()),
            )
            .await?;
            (current, applied)
        };

        let commit_message = build_commit_message(
            message,
            &format!("docs: {} {}", default_verb, rel_path),
            operation,
        );

        // `commit_and_sync` distinguishes WHERE it failed — see
        // `git::CommitSyncError`. `PreCommit`: HEAD never moved, so the file write
        // above is rolled back under this same acquisition (releasing in between
        // would let another writer commit the half-staged entry). `PostCommit`:
        // the commit is durable, left alone and reported as pending sync.
        // `Conflict`: the remote moved underneath the commit, which is already
        // dropped (the branch is back at its pre-commit HEAD, taking the file
        // change with it); sync again, then run the attempt against the fresh
        // content.
        let commit_outcome = match git::commit_and_sync(
            &git_lock,
            deps.git_url,
            deps.branch,
            data_path_str,
            deps.token,
            &[rel_path],
            &commit_message,
            deps.commit_author_name,
            deps.commit_author_email,
        )
        .await
        {
            Ok(outcome) => outcome,

            Err(git::CommitSyncError::PreCommit(source)) => {
                error!(
                    "commit_and_sync pre-commit failure for '{}', rolling back: {:#}",
                    rel_path, source
                );
                // A create has no HEAD content to restore to — remove the file and
                // unstage it. An edit restores HEAD's content (un-staging too).
                let rollback = if is_create {
                    match tokio::fs::remove_file(&abs_path).await {
                        Ok(()) => git::unstage(&git_lock, data_path_str, rel_path).await,
                        Err(e) => Err(anyhow::Error::new(e)
                            .context("Failed to remove newly-written file during rollback")),
                    }
                } else {
                    git::restore_from_head(&git_lock, data_path_str, rel_path).await
                };
                return match rollback {
                    Ok(()) => Err(WriteError::PreCommitFailed {
                        rolled_back: true,
                        msg: format!("{:#}", source),
                    }),
                    Err(rollback_err) => {
                        error!(
                            "Rollback FAILED after a pre-commit git failure for '{}': {:#}. \
                             Original cause: {:#}. Filesystem and git state may now be \
                             inconsistent.",
                            rel_path, rollback_err, source
                        );
                        Err(WriteError::PreCommitFailed {
                            rolled_back: false,
                            msg: format!(
                                "Commit cause: {:#}. Rollback cause: {:#}",
                                source, rollback_err
                            ),
                        })
                    }
                };
            }

            Err(git::CommitSyncError::PostCommit { sha, source }) => {
                warn!(
                    "commit_and_sync post-commit (sync) failure for '{}', commit {} stands \
                     uncorrected: {:#}",
                    rel_path, sha, source
                );
                mark_dirty(deps.queue, deps.indexing, vec![PathBuf::from(rel_path)]);
                return Ok(WriteSuccess {
                    outcome: WriteOutcome::CommittedPendingSync,
                    sha,
                    rebased_paths: Vec::new(),
                    diff: render_unified_diff(&old_content, &applied.content, rel_path),
                    rewritten_paths: Vec::new(),
                    referencing_paths: Vec::new(),
                    merged: applied.merged,
                    version: Some(document_version(applied.content.as_bytes())),
                });
            }

            Err(git::CommitSyncError::Conflict { source }) => {
                warn!(
                    "Remote changed underneath the write to '{}' (attempt {}/{}), re-applying \
                     against fresh content: {:#}",
                    rel_path, attempt, MAX_WRITE_ATTEMPTS, source
                );
                head = sync_clone(deps, &git_lock, head).await;
                continue;
            }
        };

        // Mark this path — and anything the rebase pulled in — dirty and return
        // immediately; the reindex worker does the embedding out of band.
        mark_dirty(
            deps.queue,
            deps.indexing,
            std::iter::once(PathBuf::from(rel_path))
                .chain(commit_outcome.rebased_paths.iter().cloned())
                .collect(),
        );

        // A clean rebase that touched this very file merged someone else's
        // change into it, so the on-disk version differs from what was written.
        let rebase_merged = rebase_touched(&commit_outcome.rebased_paths, &[rel_path]);
        let version = if rebase_merged {
            read_if_exists(&abs_path)
                .await
                .ok()
                .flatten()
                .map(|c| document_version(c.as_bytes()))
        } else {
            Some(document_version(applied.content.as_bytes()))
        };
        return Ok(WriteSuccess {
            outcome: WriteOutcome::Synced,
            sha: commit_outcome.sha,
            diff: render_unified_diff(&old_content, &applied.content, rel_path),
            rebased_paths: commit_outcome.rebased_paths,
            rewritten_paths: Vec::new(),
            referencing_paths: Vec::new(),
            merged: applied.merged || rebase_merged,
            version,
        });
    }

    // The remote kept moving under every attempt. Each attempt's commit was
    // dropped, and the clone synced with the remote again after the last, so
    // nothing of this write is left behind.
    Err(WriteError::EditedElsewhere)
}

// ---------------------------------------------------------------------------
// write_document_move: the MOVE branch of write_document (WriteRequest::dest_path)
// ---------------------------------------------------------------------------

/// The MOVE branch of `write_document`, split out because a move touches TWO
/// paths at every stage the create/edit path touches one: schema-file guard,
/// eligibility, path-safety, the filesystem mutation, the commit, and the
/// rollback.
///
/// A move is an absolute change to its source: `expected_version` is required
/// and must match the source as read under the lock, or the move is refused. An
/// accompanying content change (`req.change`) is then applied to that content.
async fn write_document_move<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &WriteDeps<'_, E, Q>,
    req: WriteRequest<'_>,
) -> Result<WriteSuccess, WriteError> {
    let WriteRequest {
        rel_path: source_rel,
        change,
        message,
        default_verb: _,
        force_new: _,
        operation,
        expected_version,
        dest_path,
    } = req;
    let dest_rel = dest_path.expect("write_document_move called with req.dest_path == None");

    // 1. A create with a dest_path is a caller bug: a create has nothing to move.
    if change.is_create() {
        return Err(WriteError::Internal {
            msg: "write_document called with a create and dest_path set; a create cannot \
                  also be a move"
                .to_string(),
        });
    }

    // 2. Eligibility + path-safety for BOTH paths, before anything else (and in
    //    particular before a configured validation.lint_command exec).
    check_not_schema_file(source_rel)?;
    check_not_schema_file(dest_rel)?;
    check_include_pattern(deps, source_rel)?;
    check_include_pattern(deps, dest_rel)?;
    safe_write_path(deps, source_rel)?;
    safe_write_path(deps, dest_rel)?;

    // 3. A move overwrites what is at the source wholesale.
    let Some(expected_version) = expected_version else {
        return Err(WriteError::VersionRequired);
    };

    validate_commit_message(message)?;

    let (git_lock, mut head) = lock_and_sync(deps).await;
    let data_path_str = data_path_of(deps);

    for attempt in 1..=MAX_WRITE_ATTEMPTS {
        // 4. Source must exist and still be the version the caller read.
        let abs_source = safe_write_path(deps, source_rel)?;
        let Some(current) = read_if_exists(&abs_source).await? else {
            return Err(WriteError::NotFound);
        };
        if !versions_match(expected_version, &current) {
            return Err(WriteError::EditedElsewhere);
        }

        // 5. Destination must NOT already exist — a move never overwrites.
        let abs_dest = safe_write_path(deps, dest_rel)?;
        if abs_dest.exists() {
            return Err(WriteError::AlreadyExists);
        }

        let applied = apply_change(
            &git_lock,
            data_path_str,
            change,
            &current,
            Some(expected_version),
            None,
        )
        .await?;

        // 6. Outbound-link re-relativization: every relative link inside the
        //    moved document was authored against its OLD directory; each is
        //    re-relativized from `dest_rel` to its own target, a self-reference
        //    mapping to `dest_rel`.
        let content_to_write =
            rewrite_outbound_links(&applied.content, source_rel, dest_rel, |resolved| {
                (resolved == source_rel).then(|| dest_rel.to_string())
            });

        // 7. Validation against the DESTINATION's schema, not the source's.
        validate_document(deps, dest_rel, &content_to_write).await?;

        // 8. Filesystem: write the DESTINATION first (`create_new`), THEN remove
        //    the source — the only order in which every failure point still has a
        //    path back to "nothing lost".
        if let Some(parent) = abs_dest.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                error!(
                    "Failed to create parent directories for '{}': {}",
                    abs_dest.display(),
                    e
                );
                WriteError::Io {
                    msg: format!("Failed to create parent directories: {}", e),
                }
            })?;
        }
        let abs_dest = safe_write_path(deps, dest_rel)?;
        {
            let mut file = tokio::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&abs_dest)
                .await
                .map_err(|e| {
                    if e.kind() == std::io::ErrorKind::AlreadyExists {
                        WriteError::AlreadyExists
                    } else {
                        error!("Failed to create file '{}': {}", abs_dest.display(), e);
                        WriteError::Io {
                            msg: format!("Failed to create file: {}", e),
                        }
                    }
                })?;
            write_and_flush(&mut file, content_to_write.as_bytes())
                .await
                .map_err(|e| {
                    error!("Failed to write file '{}': {}", abs_dest.display(), e);
                    WriteError::Io {
                        msg: format!("Failed to write file: {}", e),
                    }
                })?;
        }

        let abs_source = safe_write_path(deps, source_rel)?;
        if let Err(e) = tokio::fs::remove_file(&abs_source).await {
            error!(
                "Failed to remove source '{}' while moving it to '{}'; deleting the \
                 destination copy so the move leaves nothing behind: {}",
                source_rel, dest_rel, e
            );
            if let Err(cleanup_err) = tokio::fs::remove_file(&abs_dest).await {
                error!(
                    "Failed to clean up destination '{}' after a failed source removal during \
                     a move: {}. The document now exists at BOTH '{}' and '{}' — this needs \
                     operator attention.",
                    dest_rel, cleanup_err, source_rel, dest_rel
                );
            }
            return Err(WriteError::Io {
                msg: format!("Failed to remove source file during move: {}", e),
            });
        }

        // 9. Rewrite OTHER documents whose body links to the SOURCE path, in the
        //    same commit, reading each one's current body under the lock.
        //    Best-effort against the reverse-link index: no `StateDb`, or a failed
        //    query, skips it; the links self-heal on each document's next reindex.
        let mut rewritten_paths: Vec<String> = Vec::new();
        if let Some(state) = deps.state {
            match state.links_targeting(source_rel, "markdown").await {
                Ok(referencing_paths) => {
                    for ref_path in referencing_paths {
                        // A self-reference was handled in `content_to_write`.
                        if ref_path == source_rel {
                            continue;
                        }
                        let abs_ref = match safe_write_path(deps, &ref_path) {
                            Ok(p) => p,
                            Err(_) => {
                                warn!(
                                    "Skipping link rewrite in '{}' while moving '{}' -> '{}': \
                                     the path no longer resolves safely (stale document_links \
                                     row?)",
                                    ref_path, source_rel, dest_rel
                                );
                                continue;
                            }
                        };
                        let body = match tokio::fs::read_to_string(&abs_ref).await {
                            Ok(b) => b,
                            Err(e) => {
                                warn!(
                                    "Skipping link rewrite in '{}' while moving '{}' -> '{}': \
                                     failed to read it, likely a stale document_links row: {}",
                                    ref_path, source_rel, dest_rel, e
                                );
                                continue;
                            }
                        };
                        let occurrences: Vec<_> =
                            crate::ingest::find_markdown_link_occurrences(&body, &ref_path)
                                .into_iter()
                                .filter(|o| o.resolved.as_str() == source_rel)
                                .collect();
                        if occurrences.is_empty() {
                            continue;
                        }
                        let replacement = crate::ingest::relativize_md_path(&ref_path, dest_rel);
                        let new_body = apply_link_replacements(&body, &occurrences, &replacement);
                        if let Err(e) = tokio::fs::write(&abs_ref, new_body.as_bytes()).await {
                            error!(
                                "Failed to rewrite links into '{}' while moving '{}' -> '{}': \
                                 {}. Undoing every filesystem change made for this move so far.",
                                ref_path, source_rel, dest_rel, e
                            );
                            // Nothing has touched git yet: restore the tracked paths
                            // from HEAD and delete the untracked destination.
                            for done in &rewritten_paths {
                                if let Err(e) =
                                    git::restore_from_head(&git_lock, data_path_str, done).await
                                {
                                    error!(
                                        "Rollback: failed to restore rewritten referencing \
                                         document '{}': {:#}. This needs operator attention.",
                                        done, e
                                    );
                                }
                            }
                            if let Err(e) =
                                git::restore_from_head(&git_lock, data_path_str, source_rel).await
                            {
                                error!(
                                    "Rollback: failed to restore source '{}': {:#}. This needs \
                                     operator attention.",
                                    source_rel, e
                                );
                            }
                            if let Err(e) = tokio::fs::remove_file(&abs_dest).await {
                                error!(
                                    "Rollback: failed to remove destination '{}': {}. This \
                                     needs operator attention.",
                                    dest_rel, e
                                );
                            }
                            return Err(WriteError::Io {
                                msg: format!("Failed to rewrite links in '{}': {}", ref_path, e),
                            });
                        }
                        rewritten_paths.push(ref_path);
                    }
                }
                Err(e) => {
                    warn!(
                        "Skipping incoming-link rewrite while moving '{}' -> '{}': the \
                         reverse-link query failed: {:#}",
                        source_rel, dest_rel, e
                    );
                }
            }
        }

        let commit_message = build_commit_message(
            message,
            &format!("docs: move {} to {}", source_rel, dest_rel),
            operation,
        );

        // 10. Commit the move AND every rewritten referencing document as ONE
        //     commit, under the same acquisition as everything above.
        let mut commit_paths: Vec<&str> = vec![source_rel, dest_rel];
        for p in &rewritten_paths {
            if !commit_paths.contains(&p.as_str()) {
                commit_paths.push(p.as_str());
            }
        }

        let commit_outcome = match git::commit_and_sync(
            &git_lock,
            deps.git_url,
            deps.branch,
            data_path_str,
            deps.token,
            &commit_paths,
            &commit_message,
            deps.commit_author_name,
            deps.commit_author_email,
        )
        .await
        {
            Ok(outcome) => outcome,

            Err(git::CommitSyncError::PreCommit(source_err)) => {
                error!(
                    "commit_and_sync pre-commit failure moving '{}' -> '{}', rolling back both \
                     halves and {} rewritten referencing document(s): {:#}",
                    source_rel,
                    dest_rel,
                    rewritten_paths.len(),
                    source_err
                );
                // Roll back EVERY part: the source (tracked at HEAD) restored, the
                // destination removed and unstaged, every rewritten referencing
                // document restored. All run unconditionally; `rolled_back` only
                // if all three groups succeed.
                let source_restore =
                    git::restore_from_head(&git_lock, data_path_str, source_rel).await;
                let dest_rollback = match tokio::fs::remove_file(&abs_dest).await {
                    Ok(()) => git::unstage(&git_lock, data_path_str, dest_rel).await,
                    Err(e) => Err(anyhow::Error::new(e)
                        .context("Failed to remove the new destination file during rollback")),
                };
                let mut rewrite_restore_failures: Vec<(String, anyhow::Error)> = Vec::new();
                for path in &rewritten_paths {
                    if let Err(e) = git::restore_from_head(&git_lock, data_path_str, path).await {
                        rewrite_restore_failures.push((path.clone(), e));
                    }
                }
                let rolled_back = source_restore.is_ok()
                    && dest_rollback.is_ok()
                    && rewrite_restore_failures.is_empty();
                if !rolled_back {
                    error!(
                        "Rollback FAILED after a pre-commit git failure moving '{}' -> '{}'. \
                         Source restore: {:?}. Destination rollback: {:?}. Rewritten-document \
                         restore failures: {:?}. Original cause: {:#}. Filesystem and git \
                         state may now be inconsistent.",
                        source_rel,
                        dest_rel,
                        source_restore,
                        dest_rollback,
                        rewrite_restore_failures,
                        source_err
                    );
                }
                let mut msg = format!("{:#}", source_err);
                if let Err(e) = &source_restore {
                    msg.push_str(&format!(". Source restore cause: {:#}", e));
                }
                if let Err(e) = &dest_rollback {
                    msg.push_str(&format!(". Destination rollback cause: {:#}", e));
                }
                for (path, e) in &rewrite_restore_failures {
                    msg.push_str(&format!(". Restore of '{}' cause: {:#}", path, e));
                }
                return Err(WriteError::PreCommitFailed { rolled_back, msg });
            }

            Err(git::CommitSyncError::PostCommit {
                sha,
                source: source_err,
            }) => {
                warn!(
                    "commit_and_sync post-commit (sync) failure moving '{}' -> '{}', commit {} \
                     stands uncorrected: {:#}",
                    source_rel, dest_rel, sha, source_err
                );
                mark_dirty(
                    deps.queue,
                    deps.indexing,
                    [PathBuf::from(source_rel), PathBuf::from(dest_rel)]
                        .into_iter()
                        .chain(rewritten_paths.iter().map(PathBuf::from))
                        .collect(),
                );
                return Ok(WriteSuccess {
                    outcome: WriteOutcome::CommittedPendingSync,
                    sha,
                    rebased_paths: Vec::new(),
                    diff: render_unified_diff(&current, &content_to_write, dest_rel),
                    rewritten_paths,
                    referencing_paths: Vec::new(),
                    merged: applied.merged,
                    version: Some(document_version(content_to_write.as_bytes())),
                });
            }

            Err(git::CommitSyncError::Conflict { source }) => {
                warn!(
                    "Remote changed underneath the move '{}' -> '{}' (attempt {}/{}), \
                     re-checking against fresh content: {:#}",
                    source_rel, dest_rel, attempt, MAX_WRITE_ATTEMPTS, source
                );
                head = sync_clone(deps, &git_lock, head).await;
                continue;
            }
        };

        // 11. Mark the source, the destination, every rewritten referencing
        //     document and anything the rebase pulled in dirty, in one call.
        mark_dirty(
            deps.queue,
            deps.indexing,
            [PathBuf::from(source_rel), PathBuf::from(dest_rel)]
                .into_iter()
                .chain(rewritten_paths.iter().map(PathBuf::from))
                .chain(commit_outcome.rebased_paths.iter().cloned())
                .collect(),
        );

        // `merged` covers every document this call wrote, a rewritten referencing
        // document included; the moved document's bytes differ from what was
        // written — so its version is unknown here — only when the rebase merged
        // into the destination itself.
        let rebase_merged = rebase_touched(&commit_outcome.rebased_paths, &commit_paths);
        let dest_merged = rebase_touched(&commit_outcome.rebased_paths, &[dest_rel]);
        return Ok(WriteSuccess {
            outcome: WriteOutcome::Synced,
            sha: commit_outcome.sha,
            diff: render_unified_diff(&current, &content_to_write, dest_rel),
            rebased_paths: commit_outcome.rebased_paths,
            rewritten_paths,
            referencing_paths: Vec::new(),
            merged: applied.merged || rebase_merged,
            version: (!dest_merged).then(|| document_version(content_to_write.as_bytes())),
        });
    }

    Err(WriteError::EditedElsewhere)
}

// ---------------------------------------------------------------------------
// Wiki pipe-alias link support (`[[target|Display Text]]`) — fix #131
// ---------------------------------------------------------------------------
//
// This module used to carry a duplicate scanner/resolver here (find_pipe_alias_
// link_occurrences / scan_pipe_alias_wiki_links / resolve_pipe_alias_wiki_target /
// resolve_relative_md_path_for_pipe_alias) because `ingest::resolve_link_target`
// rejected any wiki target containing a `|` outright, so `ingest::
// find_markdown_link_occurrences` never produced an occurrence for
// `[[old/path|Display Text]]` at all. Review of that duplicate surfaced a KNOWN
// DIVERGENCE from the original it mirrored (the original cuts a target at its
// first whitespace character; the duplicate did not) — a live risk of the
// extractor and this rewriter resolving the same pipe-alias target to two
// DIFFERENT paths, which is exactly the silent-broken-link failure mode #131
// exists to fix.
//
// Fixed at the root instead: `ingest::scan_line_constructs` (the one scanner
// `extract_markdown_links` and `find_markdown_link_occurrences` both share) now
// splits `[[target|alias]]` into target and alias itself, so
// `ingest::find_markdown_link_occurrences` — already `pub(crate)` and already the
// function every rewrite site below calls — returns a correctly target-only-
// spanned occurrence for a pipe-alias link with no extra code here at all. There
// is exactly one implementation of "what counts as a link and where it resolves"
// now, shared by `document_links` extraction and this module's rewriter, so the
// two can never diverge again by construction rather than by discipline.

/// Rewrite every outbound Markdown link occurrence in `content` — a document being
/// relocated from `old_rel` to `new_rel` — so each one keeps resolving to its
/// intended target once the document lives at its new location.
///
/// `find_markdown_link_occurrences(content, old_rel)` resolves every occurrence
/// against `old_rel` — the document's OLD location, which is the directory the link
/// text was actually authored against, regardless of what a given link points at.
/// For each occurrence, `translate(resolved)` decides that occurrence's TRUE target
/// after the move: `Some(new_target)` when the linked document is ITSELF moving in
/// lockstep with this one (its own new path), `None` when it is staying exactly
/// where it is (the target is unchanged; only the relative spelling needs to
/// change because the mover's own directory changed). Either way, the final
/// replacement text is `relativize_md_path(new_rel, true_target)`.
///
/// Shared by two callers with different `translate` closures:
/// - `write_document_move`, whose closure maps only a self-reference
///   (`resolved == old_rel`) to `new_rel` and nothing else — a single document has
///   no OTHER document moving alongside it.
/// - `move_directory`, whose closure is backed by the whole batch's old→new map, so
///   a link between two documents that are BOTH moving in the same directory move
///   keeps pointing at each other post-move (see that function's doc comment).
fn rewrite_outbound_links(
    content: &str,
    old_rel: &str,
    new_rel: &str,
    translate: impl Fn(&str) -> Option<String>,
) -> String {
    let occurrences = crate::ingest::find_markdown_link_occurrences(content, old_rel);
    if occurrences.is_empty() {
        return content.to_string();
    }
    let replacements: Vec<(crate::ingest::LinkOccurrence, String)> = occurrences
        .into_iter()
        .map(|o| {
            let true_target = translate(&o.resolved).unwrap_or_else(|| o.resolved.clone());
            let replacement = crate::ingest::relativize_md_path(new_rel, &true_target);
            (o, replacement)
        })
        .collect();
    apply_link_replacements_each(content, &replacements)
}

/// Apply a PER-OCCURRENCE text replacement at each occurrence's own span,
/// back-to-front by span start so an earlier edit's byte-length change can
/// never invalidate a later span still waiting to be applied. Every
/// occurrence must have come from scanning `body` itself (e.g.
/// `ingest::find_markdown_link_occurrences`) — a span from a different string
/// is undefined behavior for `String::replace_range` (it may panic on a
/// non-char-boundary, or silently replace the wrong bytes).
///
/// This is the ONE place that owns the back-to-front span-ordering rule —
/// [`apply_link_replacements`] is a thin single-replacement wrapper over this
/// function rather than a second copy of the sort, so the two rewrite sites
/// below can never drift apart on ordering.
///
/// Used by [`rewrite_outbound_links`] (each link in a moved document's content
/// can resolve to a DIFFERENT target, so each occurrence needs its own
/// re-relativized replacement text) and directly by `move_directory`'s
/// outside-referencing-document rewrite, where one document can reference
/// SEVERAL different moved documents, each needing its own replacement text —
/// unlike `apply_link_replacements` below, where every occurrence shares one.
fn apply_link_replacements_each(
    body: &str,
    replacements: &[(crate::ingest::LinkOccurrence, String)],
) -> String {
    let mut spans: Vec<(std::ops::Range<usize>, &str)> = replacements
        .iter()
        .map(|(o, r)| (o.span.clone(), r.as_str()))
        .collect();
    // Sort back-to-front (descending by start) so replacing an earlier-in-text
    // span never shifts the byte offsets a later-in-iteration-but-earlier-in-text
    // span still needs.
    spans.sort_by_key(|(span, _)| std::cmp::Reverse(span.start));

    let mut out = body.to_string();
    for (span, replacement) in spans {
        out.replace_range(span, replacement);
    }
    out
}

/// Single-replacement convenience wrapper over
/// [`apply_link_replacements_each`], for the common case where every
/// occurrence gets the SAME replacement text.
///
/// Used by `write_document_move`'s referencing-document rewrite (step
/// 10.5): every occurrence found there resolves to the same moved source
/// path, so they all become the same relativized destination text.
fn apply_link_replacements(
    body: &str,
    occurrences: &[crate::ingest::LinkOccurrence],
    replacement: &str,
) -> String {
    let paired: Vec<(crate::ingest::LinkOccurrence, String)> = occurrences
        .iter()
        .cloned()
        .map(|o| (o, replacement.to_string()))
        .collect();
    apply_link_replacements_each(body, &paired)
}

// ---------------------------------------------------------------------------
// write_documents_batch: multiple documents, one commit (#180)
// ---------------------------------------------------------------------------

/// Hard cap on how many documents a single [`write_documents_batch`] call may
/// carry. An unbounded batch is a denial-of-service on the write path — one
/// call would stage an arbitrary amount of validation, dedup-embedding, and
/// git-add/commit work under a single [`GitLock`](git::GitLock) acquisition,
/// starving every other writer and the webhook handler for however long that
/// takes — and an unbounded MCP request payload besides. 25 is generous for
/// the batch's actual motivating case (an agent restructuring a handful of
/// related pages, or a bulk status change across a small set of documents,
/// as one logical change) while keeping a single call's worst-case work
/// bounded. The MCP adapter reports this value verbatim in its error so a
/// caller that hits it learns the exact ceiling rather than guessing.
pub const MAX_BATCH_DOCUMENTS: usize = 25;

/// One document's request within [`write_documents_batch`]. A deliberately
/// narrower cousin of [`WriteRequest`]: no `dest_path` — a batch entry can
/// create or change a document, but cannot MOVE one (a move's commit scope spans
/// the source, the destination and every document whose links get rewritten,
/// and several moves in one batch could name the same document twice with no
/// well-defined ordering). There is also no per-document
/// `message`/`default_verb`/`operation` — the whole batch lands as ONE commit
/// with one message.
pub struct BatchWriteRequest<'a> {
    /// Repo-relative path, already resolved and validated by the caller —
    /// same contract as `WriteRequest::rel_path`.
    pub rel_path: &'a str,
    /// Same contract as `WriteRequest::change`; never `DocChange::Keep`.
    pub change: DocChange<'a>,
    /// When `Some(true)`, bypasses the dedup gate for THIS entry if it is a
    /// create — dedup stays a per-document decision even inside a batch.
    pub force_new: Option<bool>,
    /// Same contract as `WriteRequest::expected_version`, for this document.
    pub expected_version: Option<&'a str>,
}

/// One document's outcome within a successful [`write_documents_batch`] call.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BatchDocumentResult {
    pub rel_path: String,
    pub is_create: bool,
    /// Unified diff of this document's own change — same rendering
    /// (`render_unified_diff`) `WriteSuccess::diff` uses for a single write.
    pub diff: String,
    /// Same contract as `WriteSuccess::merged`, for this document.
    pub merged: bool,
    /// Same contract as `WriteSuccess::version`, for this document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// A successful batch write: every document landed in ONE commit, pushed or —
/// when the push failed — committed locally with its push still pending (logged
/// where it failed, not reported here: no caller relays it). There is no
/// partial-success shape — a batch commits every document or none of them.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BatchWriteSuccess {
    /// Every document in the batch, in the order the caller supplied them.
    pub documents: Vec<BatchDocumentResult>,
}

/// Every structured failure mode of [`write_documents_batch`]. Distinct from
/// [`WriteError`] because most of what can go wrong at this level is
/// BATCH-shaped (size, a duplicate path, the one shared commit) rather than
/// document-shaped — but a per-document problem is exactly a [`WriteError`]
/// already, so [`Documents`](Self::Documents) reuses that vocabulary.
#[derive(Debug)]
pub enum BatchWriteError {
    /// The batch was empty — nothing to write.
    Empty,
    /// More documents than [`MAX_BATCH_DOCUMENTS`] were supplied.
    TooMany { count: usize, max: usize },
    /// The same `rel_path` appeared more than once. Two entries targeting the
    /// same file has no well-defined "which one wins" answer, so this is
    /// rejected before anything is touched.
    DuplicatePath { rel_path: String },
    /// One or more documents failed a check: path safety, a schema-file path,
    /// frontmatter validation, the dedup gate, a missing or stale
    /// `expected_version`, an anchor that no longer matches, create-on-existing
    /// or edit-on-missing. Carries EVERY failing document of the phase that
    /// found them, so a caller fixes the whole batch in one round trip. Every
    /// such failure leaves the batch with nothing written: checks that need the
    /// current content run under `GIT_LOCK` before the first write. A write
    /// that fails is reported here too, alone, after every document written so
    /// far is rolled back; its error names any the rollback could not restore.
    Documents { failures: Vec<(String, WriteError)> },
    /// The caller-supplied commit message would confuse git or the log.
    InvalidCommitMessage { reason: String },
    /// `git add`/`git commit` failed for the one shared commit. Mirrors
    /// `WriteError::PreCommitFailed`: `rolled_back` reports whether EVERY
    /// document this batch had written to disk was successfully restored to
    /// its pre-call state. The cause is logged here and deliberately not
    /// carried: no caller relays it.
    PreCommitFailed { rolled_back: bool },
    /// The remote kept moving under every attempt (see `MAX_WRITE_ATTEMPTS`);
    /// nothing was written: each attempt's commit was dropped, and the clone
    /// synced with the remote again after the last.
    EditedElsewhere,
}

/// Roll back every document in `written` to the content it had under the lock,
/// before this batch wrote it, using plain filesystem operations — no git calls,
/// because this runs only before `git::commit_and_sync` (and therefore `git
/// add`) has run for any of them: a create is undone by deleting the file, an
/// edit by writing its snapshot back. Snapshots are taken under the same lock
/// acquisition as the writes, so a rollback can never revert another writer's
/// change. `written` holds only files this batch created or started to
/// overwrite (see the write loop in [`write_documents_batch`]), so a rollback
/// never deletes a file it did not create. Returns the paths whose rollback
/// itself failed.
async fn rollback_batch_filesystem_writes(
    deps_data_path: &Path,
    written: &[(String, bool)],
    snapshots: &HashMap<&str, String>,
) -> Vec<String> {
    let mut failed = Vec::new();
    for (rel_path, is_create) in written {
        let abs = deps_data_path.join(rel_path);
        let result = if *is_create {
            tokio::fs::remove_file(&abs).await
        } else {
            let original = snapshots
                .get(rel_path.as_str())
                .map(String::as_str)
                .unwrap_or("");
            tokio::fs::write(&abs, original.as_bytes()).await
        };
        if let Err(e) = result {
            error!(
                "write_documents_batch: failed to roll back '{}' (pre-git-add phase): {}",
                rel_path, e
            );
            failed.push(rel_path.clone());
        }
    }
    failed
}

/// `err`, the error a failed batch write reports, with one sentence appended
/// naming every path its rollback ([`rollback_batch_filesystem_writes`]) could
/// not restore: those may hold partial content, so the failure must not read as
/// "nothing written". Names KB-relative paths only; the rollback logged each OS
/// error where it happened.
fn note_unrestored(err: WriteError, unrestored: &[String]) -> WriteError {
    if unrestored.is_empty() {
        return err;
    }
    let paths = unrestored
        .iter()
        .map(|p| format!("'{p}'"))
        .collect::<Vec<_>>()
        .join(", ");
    let note = format!("Undoing the batch failed for {paths}, which may hold partial content");
    match err {
        WriteError::Io { msg } => WriteError::Io {
            msg: format!("{msg}. {note}"),
        },
        WriteError::UnsafePath { msg } => WriteError::UnsafePath {
            msg: format!("{msg}. {note}"),
        },
        WriteError::Internal { msg } => WriteError::Internal {
            msg: format!("{msg}. {note}"),
        },
        // The write loop's one error without a message: a create whose path
        // was taken after the batch planned it.
        WriteError::AlreadyExists => WriteError::Io {
            msg: format!("document already exists. {note}"),
        },
        // The write loop returns none of the rest.
        other => other,
    }
}

/// Write every document in `requests` and land them in ONE git commit, under
/// ONE [`git::GitLock`] acquisition — the batched counterpart to
/// [`write_document`] (#180).
///
/// ## Phases
///
/// 1. **Pre-flight, no lock, no filesystem mutation.** Every document is checked
///    — schema-file guard, include-pattern eligibility, path safety, a missing
///    `expected_version` on a full replace (or one on a create), and, for a
///    create (whose content is fixed), frontmatter validation and the dedup gate
///    — and EVERY failure across the batch is collected before anything else
///    happens.
/// 2. **Under `GIT_LOCK`, after syncing with the remote.** Every document's
///    current content is read and its change applied and validated (see
///    `apply_change`) — again collecting every failure, with nothing written
///    yet. An edit that leaves its document exactly as it is is neither written
///    nor committed (when every one does, the batch returns with no commit).
///    Only then is each document written, snapshotting what it held first;
///    a write failure restores every document written so far from those
///    snapshots, the failing one included, and names any it could not restore.
/// 3. **One `git::commit_and_sync` call, all paths at once.** `PreCommit`:
///    every document is rolled back via `restore_from_head`
///    (edits)/`unstage`+`remove_file` (creates). `PostCommit`: the commit is
///    durable, so the batch succeeds and the failed sync is logged. `Conflict`:
///    the remote moved underneath the commit, which is already dropped (the
///    branch is back at its pre-commit HEAD); the clone syncs again and phase 2
///    runs again.
///
/// `message` is the whole batch's own commit subject (`None` gets a generated
/// default naming the document count).
pub async fn write_documents_batch<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &WriteDeps<'_, E, Q>,
    requests: &[BatchWriteRequest<'_>],
    message: Option<&str>,
) -> Result<BatchWriteSuccess, BatchWriteError> {
    if requests.is_empty() {
        return Err(BatchWriteError::Empty);
    }
    if requests.len() > MAX_BATCH_DOCUMENTS {
        return Err(BatchWriteError::TooMany {
            count: requests.len(),
            max: MAX_BATCH_DOCUMENTS,
        });
    }

    {
        let mut seen: HashSet<&str> = HashSet::with_capacity(requests.len());
        for req in requests {
            if !seen.insert(req.rel_path) {
                return Err(BatchWriteError::DuplicatePath {
                    rel_path: req.rel_path.to_string(),
                });
            }
        }
    }

    validate_commit_message(message).map_err(|e| match e {
        WriteError::InvalidCommitMessage { reason } => {
            BatchWriteError::InvalidCommitMessage { reason }
        }
        other => unreachable!(
            "validate_commit_message only ever returns InvalidCommitMessage, got {:?}",
            other
        ),
    })?;

    // --- Phase 1: pre-flight checks for every document, no lock, nothing written.
    let mut failures: Vec<(String, WriteError)> = Vec::new();
    for req in requests {
        let checked = async {
            check_not_schema_file(req.rel_path)?;
            check_include_pattern(deps, req.rel_path)?;
            safe_write_path(deps, req.rel_path)?;
            if matches!(req.change, DocChange::Keep) {
                return Err(WriteError::Internal {
                    msg: "a batch entry must change the document".to_string(),
                });
            }
            if req.change.is_absolute() && req.expected_version.is_none() {
                return Err(WriteError::VersionRequired);
            }
            // Same as `write_document`: a create carrying `expected_version` was
            // meant for a document that is no longer at this path.
            if req.change.is_create() && req.expected_version.is_some() {
                return Err(WriteError::NotFound);
            }
            if let DocChange::Create(content) = req.change {
                let validated = validate_document(deps, req.rel_path, content).await?;
                dedup_gate(deps, req.rel_path, validated.as_ref(), req.force_new).await?;
            }
            Ok(())
        }
        .await;
        if let Err(e) = checked {
            failures.push((req.rel_path.to_string(), e));
        }
    }
    if !failures.is_empty() {
        return Err(BatchWriteError::Documents { failures });
    }

    // Each edited document as it was before the lock, only to tell a stale
    // anchor from a wrong one (see `apply_change`).
    let mut pre_reads: HashMap<&str, String> = HashMap::new();
    for req in requests {
        if !req.change.is_create()
            && let Ok(abs) = safe_write_path(deps, req.rel_path)
            && let Ok(Some(content)) = read_if_exists(&abs).await
        {
            pre_reads.insert(req.rel_path, content);
        }
    }

    // --- Phase 2 and 3, under one acquisition.
    let (git_lock, mut head) = lock_and_sync(deps).await;
    let data_path_str = data_path_of(deps);

    for attempt in 1..=MAX_WRITE_ATTEMPTS {
        // 2a. Read, apply and validate every document; nothing written yet.
        let mut snapshots: HashMap<&str, String> = HashMap::new();
        let mut planned: Vec<(PathBuf, Applied)> = Vec::with_capacity(requests.len());
        let mut failures: Vec<(String, WriteError)> = Vec::new();
        // Edits that leave their document exactly as it is: nothing to write or
        // commit for them (and `git commit` refuses a commit with no change).
        let mut unchanged: HashSet<&str> = HashSet::new();
        for req in requests {
            let outcome = async {
                let abs_path = safe_write_path(deps, req.rel_path)?;
                let current = read_if_exists(&abs_path).await?;
                let applied = match (req.change, current) {
                    (DocChange::Create(_), Some(_)) => return Err(WriteError::AlreadyExists),
                    (DocChange::Create(content), None) => Applied {
                        content: content.to_string(),
                        merged: false,
                    },
                    (_, None) => return Err(WriteError::NotFound),
                    (change, Some(current)) => {
                        let applied = apply_change(
                            &git_lock,
                            data_path_str,
                            change,
                            &current,
                            req.expected_version,
                            pre_reads.get(req.rel_path).map(String::as_str),
                        )
                        .await?;
                        validate_document(deps, req.rel_path, &applied.content).await?;
                        if applied.content == current {
                            unchanged.insert(req.rel_path);
                        }
                        snapshots.insert(req.rel_path, current);
                        applied
                    }
                };
                Ok((abs_path, applied))
            }
            .await;
            match outcome {
                Ok(p) => planned.push(p),
                Err(e) => failures.push((req.rel_path.to_string(), e)),
            }
        }
        if !failures.is_empty() {
            return Err(BatchWriteError::Documents { failures });
        }
        // Every document already says what the batch asked for: report each as
        // it stands, with no commit.
        if unchanged.len() == requests.len() {
            return Ok(BatchWriteSuccess {
                documents: requests
                    .iter()
                    .zip(&planned)
                    .map(|(req, (_, applied))| BatchDocumentResult {
                        rel_path: req.rel_path.to_string(),
                        is_create: false,
                        diff: String::new(),
                        merged: applied.merged,
                        version: Some(document_version(applied.content.as_bytes())),
                    })
                    .collect(),
            });
        }

        // 2b. Write every document; a failure rolls back every one written so
        // far, the failing one included.
        let mut written: Vec<(String, bool)> = Vec::with_capacity(requests.len());
        for (req, (abs_path, applied)) in requests.iter().zip(&planned) {
            // Neither written nor committed; still reported, with an empty diff.
            if unchanged.contains(req.rel_path) {
                continue;
            }
            let is_create = req.change.is_create();
            // Set once this document's file may no longer hold what it held
            // under the lock — a create once `create_new` has made it (never on
            // `AlreadyExists`: that file is not this batch's to remove), an edit
            // once its truncating write starts. Only such a file is rolled back.
            let mut touched = false;
            let result: Result<(), WriteError> = async {
                if let Some(parent) = abs_path.parent() {
                    tokio::fs::create_dir_all(parent).await.map_err(|e| {
                        error!(
                            "Failed to create parent directories for '{}' (batch): {}",
                            abs_path.display(),
                            e
                        );
                        WriteError::Io {
                            msg: format!("Failed to create parent directories: {}", e),
                        }
                    })?;
                }
                let abs_path = safe_write_path(deps, req.rel_path)?;
                if is_create {
                    let mut file = tokio::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&abs_path)
                        .await
                        .map_err(|e| {
                            if e.kind() == std::io::ErrorKind::AlreadyExists {
                                WriteError::AlreadyExists
                            } else {
                                error!(
                                    "Failed to create file '{}' (batch): {}",
                                    abs_path.display(),
                                    e
                                );
                                WriteError::Io {
                                    msg: format!("Failed to create file: {}", e),
                                }
                            }
                        })?;
                    touched = true;
                    write_and_flush(&mut file, applied.content.as_bytes())
                        .await
                        .map_err(|e| WriteError::Io {
                            msg: format!("Failed to write file: {}", e),
                        })?;
                } else {
                    touched = true;
                    tokio::fs::write(&abs_path, applied.content.as_bytes())
                        .await
                        .map_err(|e| {
                            error!(
                                "Failed to write file '{}' (batch): {}",
                                abs_path.display(),
                                e
                            );
                            WriteError::Io {
                                msg: format!("Failed to write file: {}", e),
                            }
                        })?;
                }
                Ok(())
            }
            .await;
            if let Err(e) = result {
                if touched {
                    written.push((req.rel_path.to_string(), is_create));
                }
                let unrestored = rollback_batch_filesystem_writes(
                    deps.canonical_data_path,
                    &written,
                    &snapshots,
                )
                .await;
                return Err(BatchWriteError::Documents {
                    failures: vec![(req.rel_path.to_string(), note_unrestored(e, &unrestored))],
                });
            }
            written.push((req.rel_path.to_string(), is_create));
        }

        // --- Phase 3: one commit for the whole batch.
        let default_subject = format!("docs: batch update {} documents", requests.len());
        let commit_message =
            build_commit_message(message, &default_subject, "write_document_batch");
        let commit_paths: Vec<&str> = written.iter().map(|(p, _)| p.as_str()).collect();

        let documents = |rebased: &[PathBuf]| -> Vec<BatchDocumentResult> {
            requests
                .iter()
                .zip(&planned)
                .map(|(req, (_, applied))| {
                    let rebase_merged = rebase_touched(rebased, &[req.rel_path]);
                    let old = snapshots
                        .get(req.rel_path)
                        .map(String::as_str)
                        .unwrap_or("");
                    BatchDocumentResult {
                        rel_path: req.rel_path.to_string(),
                        is_create: req.change.is_create(),
                        diff: render_unified_diff(old, &applied.content, req.rel_path),
                        merged: applied.merged || rebase_merged,
                        version: (!rebase_merged)
                            .then(|| document_version(applied.content.as_bytes())),
                    }
                })
                .collect()
        };

        let commit_outcome = match git::commit_and_sync(
            &git_lock,
            deps.git_url,
            deps.branch,
            data_path_str,
            deps.token,
            &commit_paths,
            &commit_message,
            deps.commit_author_name,
            deps.commit_author_email,
        )
        .await
        {
            Ok(outcome) => outcome,

            Err(git::CommitSyncError::PreCommit(source)) => {
                error!(
                    "write_documents_batch: commit_and_sync pre-commit failure, rolling back {} \
                     document(s): {:#}",
                    written.len(),
                    source
                );
                let mut rolled_back = true;
                for (rel_path, is_create) in &written {
                    let result = if *is_create {
                        match tokio::fs::remove_file(deps.canonical_data_path.join(rel_path)).await
                        {
                            Ok(()) => git::unstage(&git_lock, data_path_str, rel_path).await,
                            Err(e) => Err(anyhow::Error::new(e)
                                .context("Failed to remove newly-written file during rollback")),
                        }
                    } else {
                        git::restore_from_head(&git_lock, data_path_str, rel_path).await
                    };
                    if let Err(e) = result {
                        rolled_back = false;
                        error!(
                            "write_documents_batch rollback: failed to restore '{}': {:#}. \
                             Filesystem and git state may now be inconsistent for this path.",
                            rel_path, e
                        );
                    }
                }
                return Err(BatchWriteError::PreCommitFailed { rolled_back });
            }

            Err(git::CommitSyncError::PostCommit { sha, source }) => {
                warn!(
                    "write_documents_batch: commit_and_sync post-commit (sync) failure, commit {} \
                     stands uncorrected for {} document(s): {:#}",
                    sha,
                    written.len(),
                    source
                );
                mark_dirty(
                    deps.queue,
                    deps.indexing,
                    written.iter().map(|(p, _)| PathBuf::from(p)).collect(),
                );
                return Ok(BatchWriteSuccess {
                    documents: documents(&[]),
                });
            }

            Err(git::CommitSyncError::Conflict { source }) => {
                warn!(
                    "Remote changed underneath a batch write (attempt {}/{}), re-applying \
                     against fresh content: {:#}",
                    attempt, MAX_WRITE_ATTEMPTS, source
                );
                head = sync_clone(deps, &git_lock, head).await;
                continue;
            }
        };

        mark_dirty(
            deps.queue,
            deps.indexing,
            written
                .iter()
                .map(|(p, _)| PathBuf::from(p))
                .chain(commit_outcome.rebased_paths.iter().cloned())
                .collect(),
        );

        return Ok(BatchWriteSuccess {
            documents: documents(&commit_outcome.rebased_paths),
        });
    }

    Err(BatchWriteError::EditedElsewhere)
}

// ---------------------------------------------------------------------------
// move_directory: atomic relocation of every document under a source prefix
// ---------------------------------------------------------------------------

/// A successful [`move_directory`] call.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DirectoryMoveSuccess {
    /// `(old_rel, new_rel)` for every document AND schema file moved (a
    /// schema file found under the source subtree moves along with the
    /// documents it governs, and the move queues a full reconcile so the worker
    /// rebuilds the shared schema cache), sorted by `old_rel`.
    pub moved: Vec<(String, String)>,
    /// Documents OUTSIDE the moved subtree whose inline links were rewritten to
    /// point at a moved document's new location, and which rode along in the same
    /// commit. Never includes a document that was itself moved — those are
    /// reported in `moved` instead. Empty when `WriteDeps::state` is `None` or
    /// nothing outside the subtree referenced it.
    pub rewritten_paths: Vec<String>,
    /// A clean rebase folded someone else's concurrent change to a moved or
    /// rewritten document into this commit — same contract as
    /// `WriteSuccess::merged`.
    pub merged: bool,
}

/// Every structured failure mode of [`move_directory`]. Mirrors
/// [`WriteError`]'s split of caller-facing vs. operator-facing variants — see
/// that enum's doc comments for the reasoning behind each shape reused here.
#[derive(Debug)]
pub enum DirectoryMoveError {
    /// `source_dir` does not exist, is not a directory, or contains no document
    /// matching the configured include patterns — there is nothing to move.
    SourceEmpty {
        msg: String,
    },
    /// At least one file already lives under `dest_dir` — a directory move never
    /// merges into, or overwrites, an existing prefix.
    AlreadyExists,
    /// Frontmatter validation against the DESTINATION's schema cascade failed
    /// for one or more documents. Carries every failure, not just the first —
    /// the whole move is all-or-nothing, so a caller needs to know everything
    /// that would need fixing, not just whichever document happened to be
    /// checked first.
    Validation {
        failures: Vec<(String, ValidationResult)>,
        /// `(old_rel, new_rel)` for every schema file this move is
        /// relocating — empty when the source subtree carries no schema file
        /// of its own. Non-empty means the destination cascade these failures
        /// were checked against is not just "whatever already governed the
        /// destination" but a genuinely NEW cascade, re-parented by this very
        /// move — see `SchemaCache::with_remapped_scopes`. Lets the caller-
        /// facing error name that explicitly instead of leaving a document
        /// that was valid moments ago looking like an unexplained failure.
        moved_schema_files: Vec<(String, String)>,
    },
    /// A schema file under the source subtree is invalid on disk (#272).
    /// The shared cache keeps the last good schema while a file on disk is
    /// invalid, so it may still hold that directory's previous rules — the move
    /// would validate against those, then carry the broken file to the
    /// destination. A file in a hidden or excluded directory, which no cache
    /// ever read, is refused the same way when the move would bring it into the
    /// schema tree. Refused until the file is fixed or reverted in git. `path` is
    /// the first such file (by path), KB-relative; `reason` is why it is invalid.
    InvalidSchemaInSource {
        path: String,
        reason: String,
    },
    UnsafePath {
        msg: String,
    },
    Internal {
        msg: String,
    },
    InvalidCommitMessage {
        reason: String,
    },
    /// `git add`/`git commit` failed. See `WriteError::PreCommitFailed`'s doc
    /// comment for the `rolled_back` contract — identical here, just scaled to
    /// every path this move touched: `true` only if every document's source
    /// restore, every document's destination removal, and every rewritten
    /// referencing document's restore all succeeded. The cause is logged here
    /// and deliberately not carried: no caller relays it.
    PreCommitFailed {
        rolled_back: bool,
    },
    Io {
        msg: String,
    },
    /// Something under the source (or the destination prefix) changed between
    /// the pre-lock scan and the re-check under the lock — a document edited,
    /// added or removed — or the remote kept moving under every attempt. Nothing
    /// was moved; the caller re-reads and tries again.
    EditedElsewhere,
}

/// Maps [`safe_write_path`]/[`check_include_pattern`]/[`validate_commit_message`]
/// failures onto [`DirectoryMoveError`], so [`move_directory`] can reuse those
/// helpers with `?` instead of duplicating their logic. In practice only the
/// first four arms are ever produced by those three call sites — the fallback
/// exists purely to keep this conversion exhaustive against `WriteError`'s full
/// variant set, which those helpers' return types do not restrict.
impl From<WriteError> for DirectoryMoveError {
    fn from(err: WriteError) -> Self {
        match err {
            WriteError::UnsafePath { msg } => DirectoryMoveError::UnsafePath { msg },
            WriteError::Internal { msg } => DirectoryMoveError::Internal { msg },
            WriteError::InvalidCommitMessage { reason } => {
                DirectoryMoveError::InvalidCommitMessage { reason }
            }
            WriteError::Io { msg } => DirectoryMoveError::Io { msg },
            other => DirectoryMoveError::Internal {
                msg: format!("unexpected error surfaced in move_directory: {:?}", other),
            },
        }
    }
}

/// Recursively collect the KB-root-relative path of every regular file under
/// `abs_dir` (which must itself already exist as a directory), at any depth.
/// Symlinks are skipped — delegates the actual recursive walk to
/// `ingest::walk_dir_unfiltered`, the same walker `discover_files` runs (just in
/// its unfiltered mode), so a future fix to symlink-loop or entry-error
/// handling in one reaches both instead of only whichever one it landed in.
///
/// Unfiltered: returns every file, not just indexable documents. `move_directory`
/// uses this both for the source-subtree scan (filtered to indexable documents,
/// and scanned for schema files to carry along, by the caller) and the
/// destination-prefix collision check (deliberately left UNFILTERED there, since
/// ANY file under the destination — indexable or not — means the prefix is not
/// free).
///
/// Synchronous (a plain recursive `std::fs` walk) — callers on the async path
/// must run this via [`walk_subtree_files_async`] instead of calling it
/// directly, so a large subtree scan runs off the tokio worker thread rather
/// than blocking every other task scheduled on it.
fn walk_subtree_files(canonical_data_path: &Path, abs_dir: &Path) -> std::io::Result<Vec<String>> {
    let files = crate::ingest::walk_dir_unfiltered(canonical_data_path, abs_dir)
        .map_err(|e| std::io::Error::other(format!("{e:#}")))?;
    let mut out: Vec<String> = files
        .into_iter()
        .map(|path| {
            path.strip_prefix(canonical_data_path)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/")
        })
        .collect();
    out.sort();
    Ok(out)
}

/// Off-thread wrapper around [`walk_subtree_files`] for callers inside async
/// `move_directory`: a large recursive subtree scan is exactly the kind of
/// blocking filesystem work `ingest.rs`'s own `discover_relative` already runs
/// via `spawn_blocking` (see its doc comment) rather than directly on a tokio
/// worker thread — done inline here it would stall every unrelated MCP/webhook
/// task scheduled on that same worker for the whole walk. Takes owned buffers
/// so the spawned closure needs nothing borrowed from the caller's stack
/// (`Path` args are cloned into `PathBuf`s before crossing into the blocking
/// closure) — in particular, this never captures the caller's held `GitLock`
/// guard, which must stay on the calling task.
async fn walk_subtree_files_async(
    canonical_data_path: &Path,
    abs_dir: &Path,
) -> std::io::Result<Vec<String>> {
    let canonical_data_path = canonical_data_path.to_path_buf();
    let abs_dir = abs_dir.to_path_buf();
    match tokio::task::spawn_blocking(move || walk_subtree_files(&canonical_data_path, &abs_dir))
        .await
    {
        Ok(result) => result,
        Err(e) => Err(std::io::Error::other(format!(
            "walk_subtree_files task panicked: {e}"
        ))),
    }
}

/// Best-effort, recursive, deepest-first removal of every now-empty directory
/// under (and including) `dir`. `std::fs::remove_dir` only ever succeeds on an
/// actually-empty directory, so this silently leaves anything non-empty (a
/// stray non-indexable file the include patterns never touched, e.g.) exactly
/// where it is — this is tidying up after a move, not a second guarantee
/// layered on top of `move_directory`'s own guards. A missing `dir` (already
/// gone, or never existed) is likewise a silent no-op.
///
/// Git does not track empty directories at all, so this has no bearing on
/// what gets committed — it exists purely so that after every document under
/// `source_dir` has been moved out, `source_dir` itself does not linger as an
/// empty husk on disk (and, symmetrically, so a rolled-back move's
/// now-empty destination directory does not linger either).
///
/// Synchronous, same reason as [`walk_subtree_files`] — async callers must go
/// through [`remove_empty_dirs_best_effort_async`].
fn remove_empty_dirs_best_effort(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.file_type().is_ok_and(|t| t.is_dir()) {
            remove_empty_dirs_best_effort(&entry.path());
        }
    }
    let _ = std::fs::remove_dir(dir);
}

/// Off-thread wrapper around [`remove_empty_dirs_best_effort`], same rationale
/// as [`walk_subtree_files_async`]: a deep, mostly-empty directory tree left
/// behind by a large move is still a synchronous recursive `std::fs` walk, and
/// every call site here runs while `move_directory` holds `GitLock` — this
/// takes an owned `PathBuf` rather than borrowing the caller's `&Path`, so
/// nothing from the caller's stack (least of all the lock guard, which callers
/// never pass in) crosses into the blocking closure. Best-effort by
/// construction already (the sync version swallows every error), so a panicked
/// blocking task is just logged, not propagated.
async fn remove_empty_dirs_best_effort_async(dir: &Path) {
    let dir = dir.to_path_buf();
    if let Err(e) = tokio::task::spawn_blocking(move || remove_empty_dirs_best_effort(&dir)).await {
        error!("remove_empty_dirs_best_effort task panicked: {e}");
    }
}

/// Undo every filesystem change [`move_directory`] has made SO FAR — before it
/// has touched git at all (no `commit_and_sync` call has run yet) — on a failure
/// partway through writing destinations, removing sources, or rewriting outside
/// referencing documents.
///
/// Safe to call unconditionally over the FULL `moves` list regardless of how far
/// phase 1/2 actually got: `git::restore_from_head` on a source whose worktree
/// content already matches HEAD (i.e. not yet removed) is a harmless no-op, so
/// passing every source rather than tracking exactly which ones were already
/// removed keeps this one function correct for every call site instead of
/// several slightly-different partial-rollback paths. `written_dest` and
/// `rewritten_refs`, by contrast, are exactly the paths actually mutated so far
/// — there is no such no-op equivalent for creating/removing a file that may or
/// may not exist, so those two stay precise per call site.
///
/// Takes the caller's already-held [`git::GitLock`] rather than acquiring its
/// own: `move_directory` now holds ONE guard across its entire mutating
/// sequence — every destination write, every source removal, every outside
/// referencing-document rewrite, the commit, and any rollback — for the same
/// data-loss reason `write_document_move` holds one across its equivalent
/// sequence (an unlocked read-modify-write of another document can be raced
/// and silently clobbered by a concurrent writer). This function runs from
/// failure paths INSIDE that sequence, so it must reuse the same guard rather
/// than call `git::lock_git()` itself — `GIT_LOCK` is a non-reentrant mutex,
/// and a second acquisition here while the first is still held by the caller
/// would deadlock the whole call chain against itself. (Previously this
/// acquired its own lock, on the reasoning that nothing had been staged or
/// committed yet at any call site; that remains true of the git plumbing, but
/// not of the filesystem mutations this function itself performs, which is
/// exactly the gap the data-loss finding this change fixes was about.)
/// Every individual restore/removal failure is logged, not propagated — the
/// caller has already committed to failing the whole move and just needs
/// everything recoverable put back, best effort.
async fn rollback_directory_move_filesystem(
    lock: &git::GitLock,
    data_path_str: &str,
    moves: &[(String, String)],
    abs_dest_dir: &Path,
    written_dest: &[PathBuf],
    rewritten_refs: &[String],
) {
    for (old_rel, _new_rel) in moves {
        if let Err(e) = git::restore_from_head(lock, data_path_str, old_rel).await {
            error!(
                "move_directory rollback: failed to restore source '{}': {:#}. This needs \
                 operator attention.",
                old_rel, e
            );
        }
    }
    for dest in written_dest {
        if let Err(e) = tokio::fs::remove_file(dest).await {
            error!(
                "move_directory rollback: failed to remove destination '{}': {}. This needs \
                 operator attention.",
                dest.display(),
                e
            );
        }
    }
    // Best-effort: tidy up any destination directory left empty by the removals
    // above, so a rolled-back move does not leave an empty destination prefix
    // behind — see `remove_empty_dirs_best_effort`'s doc comment.
    remove_empty_dirs_best_effort_async(abs_dest_dir).await;
    for ref_path in rewritten_refs {
        if let Err(e) = git::restore_from_head(lock, data_path_str, ref_path).await {
            error!(
                "move_directory rollback: failed to restore referencing document '{}': {:#}. \
                 This needs operator attention.",
                ref_path, e
            );
        }
    }
}

/// `raw`, a [`move_directory`] prefix, in canonical `a/b` form: a leading `/`
/// (the knowledge-base root, as in every path a tool takes) and every `.`
/// component or empty segment are dropped, so `./notes/`, `/./notes` and
/// `notes//.` all read as `notes`, and the root itself, however spelled, comes
/// back empty. A `..` component is kept for [`safe_write_path`] to refuse. Every
/// path a move derives from its prefixes — each moved file's old and new path,
/// the commit's paths, the paths it marks dirty — is built on this form, never
/// on the caller's spelling.
fn normalize_dir_prefix(raw: &str) -> String {
    use std::path::Component;
    Path::new(raw)
        .components()
        .filter_map(|c| match c {
            Component::CurDir | Component::RootDir => None,
            other => Some(other.as_os_str().to_string_lossy()),
        })
        .collect::<Vec<_>>()
        .join("/")
}

/// Relocate every document under `source_dir` to the same relative path under
/// `dest_dir`, as ONE atomic commit — the directory-move counterpart to
/// [`write_document_move`]'s single-document move, sharing its path-safety,
/// eligibility, schema-resolution, link-rewriting, and commit/rollback helpers
/// rather than duplicating them.
///
/// # Guards (all before any mutation)
/// 1. `source_dir` must exist and contain at least one document matching the
///    configured include patterns, or this is refused as
///    [`DirectoryMoveError::SourceEmpty`].
/// 2. No file may already live anywhere under `dest_dir`
///    ([`DirectoryMoveError::AlreadyExists`]) — a directory move never merges.
/// 3. A schema file under the source subtree is not a blocker: it moves
///    WITH the documents it governs (as a raw copy — schema files are never
///    frontmatter-validated or link-rewritten), and every moved document is
///    validated against a cascade rebuilt with that schema file's governing
///    directory re-parented onto the destination
///    (`SchemaCache::with_remapped_scopes`), not against the live cache, which
///    still reflects the OLD parentage until the post-commit full reconcile
///    rebuilds it (`reindex::ReindexQueue::mark_schema_changes`). The shared
///    cache only ever holds schema files that were valid when it was built, so
///    the remapped cascade is built from validated content — but it may be the
///    last good content of a file that is invalid on disk now, so every schema
///    file under the source that is part of the schema tree at its source or at
///    its destination is re-read, parsed and self-validated first, and an
///    invalid one refuses the move
///    ([`DirectoryMoveError::InvalidSchemaInSource`], #272). Relocating a schema
///    file is a genuine semantic change — a document valid under the source's
///    cascade can fail under the destination's — and that is exactly what guard
///    4 below (`DirectoryMoveError::Validation`) exists to catch before anything
///    moves.
/// 4. Every moved document's frontmatter must validate against the (possibly
///    re-parented, per guard 3) destination cascade.
/// 5. Every source and destination document path passes the same path-safety
///    ([`safe_write_path`]) and include-pattern eligibility
///    ([`check_include_pattern`]) checks a single-document write applies.
///
/// # Link rewriting
/// Every relative link inside a moved document was authored against wherever it
/// used to live. For each moved document, every link occurrence is resolved
/// against its OLD path ([`rewrite_outbound_links`]), then re-targeted:
/// - A link whose resolved target is ALSO moving (i.e. is another document under
///   `source_dir`) is translated to that target's own post-move path, then
///   re-relativized from the mover's NEW path — preserving the link's original
///   target now that both ends moved in lockstep. For a target moving in
///   lockstep with the source this usually reproduces identical text; no rewrite
///   is emitted when it does (`rewrite_outbound_links` returns the original
///   content unchanged whenever there are no occurrences, and otherwise only
///   ever replaces the exact occurrence spans it found).
/// - A link whose resolved target is NOT moving keeps that exact target; only
///   its relative spelling is recomputed from the mover's new path.
/// - A link targeting the document itself maps to its own new path (the
///   self-reference case, same as `write_document_move`).
///
/// Separately, every document OUTSIDE the moved subtree that links INTO it
/// (`StateDb::links_targeting_many`, one batched query over every moved path,
/// filtered to sources outside `source_dir` — sources INSIDE it are the
/// outbound pass above, and are never processed twice) has its link text
/// rewritten to the moved document's new location,
/// riding along in the same commit. Same best-effort semantics as
/// `write_document_move`: with no `StateDb` (`WriteDeps::state == None`), this
/// step is skipped entirely and the move still proceeds.
///
/// # Commit / rollback / reindex
/// One `git::commit_and_sync` call over every old path, every new path, and
/// every rewritten outside-referencing document (deduped). On a pre-commit
/// failure, EVERY part of the move is rolled back under the SAME held
/// `GitLock`: every source restored from HEAD, every destination removed and
/// unstaged, every rewritten referencing document restored from HEAD — all
/// steps run unconditionally, and `rolled_back` is `true` only if every single
/// one of them succeeded. On success, every path is marked dirty in one
/// `reindex::mark_paths` call, and a full reconcile is queued when a schema file
/// moved (see [`mark_dirty`]).
///
/// How many documents' `validate::validate_content` calls run at once (see the
/// body below): each may exec an external `lint_command` subprocess, so this
/// is a *process* concurrency bound, not just an async-task one. 8 is
/// deliberately conservative — in the same range as a typical machine's core
/// count — chosen to get most of the win over a fully serial loop (N times
/// fewer round trips through subprocess spawn/exit latency) without letting a
/// large subtree move fork hundreds of lint processes at once and risk
/// exhausting file descriptors or thrashing the host.
const DIRECTORY_MOVE_VALIDATION_CONCURRENCY: usize = 8;

pub async fn move_directory<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &WriteDeps<'_, E, Q>,
    source_dir: &str,
    dest_dir: &str,
    message: Option<&str>,
) -> Result<DirectoryMoveSuccess, DirectoryMoveError> {
    let source_dir = normalize_dir_prefix(source_dir);
    let dest_dir = normalize_dir_prefix(dest_dir);
    if source_dir.is_empty() {
        return Err(DirectoryMoveError::UnsafePath {
            msg: "Invalid path: a directory move needs a directory under the knowledge base \
                  root, not the root itself"
                .to_string(),
        });
    }
    let (source_dir, dest_dir) = (source_dir.as_str(), dest_dir.as_str());

    // Guard 5 (path safety) against the two prefixes themselves, ahead of
    // walking either one.
    let abs_source_dir = safe_write_path(deps, source_dir)?;
    let abs_dest_dir = safe_write_path(deps, dest_dir)?;

    if !abs_source_dir.is_dir() {
        return Err(DirectoryMoveError::SourceEmpty {
            msg: format!("source directory '{}' does not exist", source_dir),
        });
    }

    // Guard 1: walk the whole source subtree once, collecting both the indexable
    // documents (guard 1) and any schema file living anywhere underneath —
    // one filesystem walk answers both. A schema file no longer blocks the move
    // (guard 3, checked further below, once `deps.schema_cache` is loaded) — it
    // travels WITH the subtree instead, see this function's doc comment.
    let source_files = walk_subtree_files_async(deps.canonical_data_path, &abs_source_dir)
        .await
        .map_err(|e| DirectoryMoveError::Io {
            msg: format!("Failed to scan source directory '{}': {}", source_dir, e),
        })?;

    // Everything under the source as this pre-lock scan saw it; re-checked under
    // the lock before anything moves.
    let source_snapshot = source_files.clone();

    let schema_files_in_source: Vec<String> = source_files
        .iter()
        .filter(|p| crate::schema::is_schema_file_path(Path::new(p)))
        .cloned()
        .collect();

    let documents: Vec<String> = source_files
        .into_iter()
        .filter(|p| {
            // A schema file is never a document, even under a widened `include`
            // (`**/*`): it travels via `schema_moves` below, and listing it here too
            // would validate it as markdown and rename it twice.
            !crate::schema::is_schema_file_path(Path::new(p.as_str()))
                && deps.retrieval.include_patterns.is_match(p.as_str())
        })
        .collect();
    if documents.is_empty() {
        return Err(DirectoryMoveError::SourceEmpty {
            msg: format!(
                "source directory '{}' contains no indexable document",
                source_dir
            ),
        });
    }

    // Guard 2: the destination prefix must be completely free — ANY file
    // underneath it, indexable or not, is a collision.
    if abs_dest_dir.is_file() {
        return Err(DirectoryMoveError::AlreadyExists);
    }
    if abs_dest_dir.is_dir() {
        let dest_files = walk_subtree_files_async(deps.canonical_data_path, &abs_dest_dir)
            .await
            .map_err(|e| DirectoryMoveError::Io {
                msg: format!("Failed to scan destination directory '{}': {}", dest_dir, e),
            })?;
        if !dest_files.is_empty() {
            return Err(DirectoryMoveError::AlreadyExists);
        }
    }

    // Every document's new path, preserving its position under the source
    // subtree. `documents` is already sorted (`walk_subtree_files` sorts), so
    // `moves` is too.
    let source_prefix = format!("{}/", source_dir);
    // `source_dir` is normalized, so every file the walk found carries this
    // prefix; one that somehow does not is an error, never a panic.
    let relocate = |old_rel: &String| -> Result<(String, String), DirectoryMoveError> {
        let suffix =
            old_rel
                .strip_prefix(&source_prefix)
                .ok_or_else(|| DirectoryMoveError::Internal {
                    msg: format!(
                        "'{old_rel}' is not under the directory being moved, '{source_dir}'"
                    ),
                })?;
        Ok((old_rel.clone(), format!("{}/{}", dest_dir, suffix)))
    };
    let moves: Vec<(String, String)> = documents.iter().map(&relocate).collect::<Result<_, _>>()?;
    let moving: HashMap<&str, &str> = moves
        .iter()
        .map(|(old, new)| (old.as_str(), new.as_str()))
        .collect();

    // Every schema file's new path, same prefix substitution as `moves` above.
    // These move as raw copies alongside the documents they govern: never
    // through frontmatter validation (a schema file has no frontmatter of
    // its own) and never through link rewriting (nothing in one is a markdown
    // link). Deliberately excluded from `moving`/`moves` above — those drive the
    // markdown link-rewrite passes, which a schema file relocation has nothing
    // to do with.
    let schema_moves: Vec<(String, String)> = schema_files_in_source
        .iter()
        .map(&relocate)
        .collect::<Result<_, _>>()?;
    // Every path this move touches, documents and schema files alike — used for
    // commit staging, dirty-marking, rollback, and the success report. `moves`/
    // `moving` above stay document-only: a schema file is never a markdown link
    // target and never runs through `validate::validate_content`.
    let mut all_moves: Vec<(String, String)> = moves
        .iter()
        .cloned()
        .chain(schema_moves.iter().cloned())
        .collect();
    // `moves` and `schema_moves` are each individually sorted by `old_rel`
    // (`walk_subtree_files` sorts), but the chained concatenation is not —
    // re-sort so `DirectoryMoveSuccess::moved`'s documented ordering holds.
    all_moves.sort_by(|a, b| a.0.cmp(&b.0));

    let schemas = crate::schema::load_shared(deps.schema_cache);

    // Guard 3: a NEW, detached cache with every schema file under `source_dir` re-parented
    // onto `dest_dir` (see `SchemaCache::with_remapped_scopes`). Moved documents
    // are validated against THIS cache below, not the live one: if the subtree
    // carries its own schema file(s), the live cache still reflects the OLD
    // parentage until the post-commit reindex rebuilds it
    // (`reindex::unit_touches_schema`), so validating against it here would
    // silently ignore the exact re-parenting this move is about to cause. When
    // the subtree has no schema file of its own, `remap` never matches anything
    // actually present in `schemas.raw`... other than possibly one of its own
    // ancestors, which is intentional: an ancestor's schema is not itself under
    // `source_dir`, so it is never relocated, and `remapped_schemas` resolves
    // identically to `schemas` for every moved document in that case.
    let source_dir_path = Path::new(source_dir);
    let dest_dir_path = Path::new(dest_dir);
    let remapped_schemas = schemas.with_remapped_scopes(|dir| {
        if dir.starts_with(source_dir_path) {
            let suffix = dir.strip_prefix(source_dir_path).unwrap_or(Path::new(""));
            Some(dest_dir_path.join(suffix))
        } else {
            None
        }
    });

    // Guard 5 (eligibility/safety, per document) + the outbound link rewrite.
    // ALL before any mutation: every document is only ever READ here, and every
    // failure path below returns before touching the filesystem. These checks
    // are all cheap, in-memory or single-file-read work, so they stay a plain
    // serial loop — same first-failure-wins behavior as before, e.g.
    // `AlreadyExists` on whichever document trips it first. Only `validate::validate_content`
    // below (which may exec a `lint_command` subprocess) is expensive enough,
    // and independent enough per document, to run concurrently.
    // (old_rel, new_rel, content_to_write)
    let mut contents: Vec<(String, String, String)> = Vec::new();
    // (old_rel, content as read) — the snapshot the under-lock check compares.
    let mut originals: Vec<(String, String)> = Vec::new();

    for (old_rel, new_rel) in &moves {
        check_include_pattern(deps, old_rel)?;
        check_include_pattern(deps, new_rel)?;
        let abs_source_doc = safe_write_path(deps, old_rel)?;
        let abs_dest_doc = safe_write_path(deps, new_rel)?;

        if abs_dest_doc.exists() {
            // Guard 2 already checked the whole prefix; this is a defensive
            // re-check against a TOCTOU race between that walk and here.
            return Err(DirectoryMoveError::AlreadyExists);
        }

        let old_content = tokio::fs::read_to_string(&abs_source_doc)
            .await
            .map_err(|e| DirectoryMoveError::Io {
                msg: format!("Failed to read '{}': {}", old_rel, e),
            })?;

        let content_to_write = rewrite_outbound_links(&old_content, old_rel, new_rel, |resolved| {
            moving.get(resolved).map(|new| new.to_string())
        });

        contents.push((old_rel.clone(), new_rel.clone(), content_to_write));
        originals.push((old_rel.clone(), old_content));
    }

    // Read every schema file's raw content too — same path-safety (guard 5) as
    // any other moved file, but deliberately NOT `check_include_pattern` (a
    // schema file never matches the markdown include patterns, so that
    // check would always reject it) and no `rewrite_outbound_links` (schema
    // files hold no markdown links). These ride along in the same physical
    // write/remove phases as `contents` below, chained rather than merged into
    // it, so they never enter `validate::validate_content`.
    let mut schema_contents: Vec<(String, String, String)> = Vec::new();
    for (old_rel, new_rel) in &schema_moves {
        let abs_source_schema = safe_write_path(deps, old_rel)?;
        let _abs_dest_schema = safe_write_path(deps, new_rel)?;
        let raw = tokio::fs::read_to_string(&abs_source_schema)
            .await
            .map_err(|e| DirectoryMoveError::Io {
                msg: format!("Failed to read '{}': {}", old_rel, e),
            })?;
        schema_contents.push((old_rel.clone(), new_rel.clone(), raw));
    }

    // Guard 3a (#272): every schema file being carried along must be valid ON
    // DISK. A runtime rebuild that finds one invalid keeps the last good cache,
    // so `schemas` (and `remapped_schemas`) may still hold that directory's old
    // rules: validating against them and then relocating the broken file would
    // move it out from under the very check that should have stopped it. A file
    // is checked when it is in the schema tree (`schema::SchemaWalkFilter`) at
    // its source or at its destination — the rebuild this move queues reads it
    // there, and one coming out of a hidden or wholly excluded directory was
    // never read, so no cache vouches for it. Only a file outside the schema
    // tree at both ends moves unchecked: no rebuild reads it.
    let walk = crate::schema::SchemaWalkFilter::from_config(deps.indexing);
    let mut governed_dirs: HashSet<&Path> = HashSet::new();
    for (old_rel, new_rel, raw) in &schema_contents {
        if !walk.governs(Path::new(new_rel)) && !walk.governs(Path::new(old_rel)) {
            continue;
        }
        // Both schema file names in one directory is invalid for a rebuild too
        // (`schema::BOTH_NAMES_REASON`), whichever name each one carries.
        if !governed_dirs.insert(Path::new(new_rel).parent().unwrap_or(Path::new(""))) {
            return Err(DirectoryMoveError::InvalidSchemaInSource {
                path: old_rel.clone(),
                reason: crate::schema::BOTH_NAMES_REASON.to_string(),
            });
        }
        if let Err(reason) = crate::schema::parse_schema_text(raw) {
            return Err(DirectoryMoveError::InvalidSchemaInSource {
                path: old_rel.clone(),
                reason,
            });
        }
    }

    // Destination-schema validation, run concurrently across every document
    // rather than one `validate::validate_content` await at a time — each call
    // may exec an external `lint_command` subprocess, so a 200-document
    // subtree previously paid 200x that subprocess's spawn/exit latency
    // serially. Bounded via `buffer_unordered`, not spawned one task per
    // document unbounded: an unbounded fan-out on a large subtree would fork
    // hundreds of lint subprocesses at once and risks exhausting file
    // descriptors or thrashing the machine. `DIRECTORY_MOVE_VALIDATION_CONCURRENCY`
    // documents `N`'s reasoning.
    //
    // Semantics are preserved exactly: this collects EVERY document's outcome
    // before deciding anything (`.collect::<Vec<_>>().await` drains the whole
    // bounded stream), so a document that fails validation can never be
    // reported as "the only failure" just because it finished first under
    // concurrency — same as the old serial loop reporting every failure
    // encountered before returning `Validation`. A genuine `validate_content`
    // error (as opposed to an ordinary "invalid" result) still aborts the move
    // exactly like the old loop's `?` did — the first one found after the
    // whole bounded batch settles, rather than mid-loop, since concurrent
    // tasks already in flight cannot be un-started once launched.
    // Built via a plain loop (not `Iterator::map`) so each future's captures are
    // inferred independently rather than through one `FnMut` closure signature
    // that has to hold for every item uniformly — the latter runs into a known
    // rustc limitation ("implementation of `FnOnce` is not general enough")
    // once the closure's return type borrows from both the loop item and an
    // outer variable (`remapped_schemas`) at once.
    //
    // Deliberately `remapped_schemas`, not the live `schemas` snapshot: every
    // moved document's frontmatter is checked against the cascade it will
    // ACTUALLY resolve to post-move, with any schema file in this subtree
    // already re-parented onto the destination — see `remapped_schemas`'s doc
    // comment above and `SchemaCache::with_remapped_scopes`.
    let mut validation_futures = Vec::with_capacity(contents.len());
    for (_old_rel, new_rel, content_to_write) in &contents {
        let schema = remapped_schemas.resolve_for(Path::new(new_rel.as_str()));
        validation_futures.push(async move {
            let outcome = validate::validate_content(
                Path::new(new_rel.as_str()),
                content_to_write,
                schema,
                deps.validation,
            )
            .await
            .map(|(validation_result, _validated)| validation_result);
            (new_rel.clone(), outcome)
        });
    }
    let validation_outcomes: Vec<(String, anyhow::Result<ValidationResult>)> =
        stream::iter(validation_futures)
            .buffer_unordered(DIRECTORY_MOVE_VALIDATION_CONCURRENCY)
            .collect()
            .await;

    let mut validation_failures: Vec<(String, ValidationResult)> = Vec::new();
    for (new_rel, outcome) in validation_outcomes {
        match outcome {
            Ok(validation_result) => {
                if !validation_result.valid {
                    validation_failures.push((new_rel, validation_result));
                }
            }
            Err(e) => {
                error!(
                    "Validation error moving into '{}' (source '{}' -> '{}'): {:#}",
                    new_rel, source_dir, dest_dir, e
                );
                return Err(DirectoryMoveError::Io {
                    msg: format!("Failed to validate content: {}", e),
                });
            }
        }
    }

    if !validation_failures.is_empty() {
        // Sort so the reported order is deterministic regardless of which
        // validation happened to finish first under `buffer_unordered` — the
        // old serial loop always reported failures in `moves` order (which is
        // sorted, per `walk_subtree_files`), and callers' error text
        // (`mcp::move_directory_error_to_mcp_error`) reads more like a stable
        // report when it stays that way.
        validation_failures.sort_by(|a, b| a.0.cmp(&b.0));
        return Err(DirectoryMoveError::Validation {
            failures: validation_failures,
            // Empty unless this subtree carries its own schema file(s) — lets
            // the caller-facing error explain WHY a document that was valid at
            // the source can fail here: the cascade it is being checked against
            // just re-parented, not merely relocated.
            moved_schema_files: schema_moves.clone(),
        });
    }

    validate_commit_message(message)?;

    let data_path_str = deps.canonical_data_path.to_str().unwrap_or_default();

    // Take GIT_LOCK, sync with the remote, and hold ONE guard across every
    // remaining step — the re-check below, phase 1 (destination writes), phase 2
    // (source removals), phase 3 (outside referencing-document rewrites), the
    // commit, and any rollback. Every helper reachable from here takes this same
    // guard by reference, which keeps the non-reentrant mutex from deadlocking.
    let (git_lock, mut head) = lock_and_sync(deps).await;

    for attempt in 1..=MAX_WRITE_ATTEMPTS {
        // Everything above read and validated the subtree before the lock. Under it,
        // re-walk the source and compare every file with that snapshot — a document
        // edited, added or removed in between (locally, or pulled in by the sync)
        // refuses the move rather than moving content nobody validated or leaving a
        // new file behind. The destination prefix must still be free, too; a
        // destination that filled up meanwhile is the same `AlreadyExists` the
        // pre-check reports.
        let source_unchanged = async {
            let now = walk_subtree_files_async(deps.canonical_data_path, &abs_source_dir)
                .await
                .ok()?;
            if now != source_snapshot {
                return Some(false);
            }
            let schema_originals = schema_contents
                .iter()
                .map(|(old_rel, _, raw)| (old_rel, raw));
            for (old_rel, before) in originals
                .iter()
                .map(|(o, c)| (o, c))
                .chain(schema_originals)
            {
                let abs = safe_write_path(deps, old_rel).ok()?;
                if tokio::fs::read_to_string(&abs).await.ok()? != *before {
                    return Some(false);
                }
            }
            Some(true)
        }
        .await;
        if source_unchanged != Some(true) {
            return Err(DirectoryMoveError::EditedElsewhere);
        }
        let dest_occupied = abs_dest_dir.is_file()
            || (abs_dest_dir.is_dir()
                && walk_subtree_files_async(deps.canonical_data_path, &abs_dest_dir)
                    .await
                    .map_or(true, |files| !files.is_empty()));
        if dest_occupied {
            return Err(DirectoryMoveError::AlreadyExists);
        }

        // Filesystem mutation, phase 1: write every DESTINATION first (`create_new`,
        // same non-clobbering guarantee `write_document_move` relies on), before
        // touching a single source — the same write-then-remove ordering that
        // function uses, batched: if any destination write fails partway through,
        // no source has been touched at all, so recovery is just deleting whatever
        // destinations already landed. Chained with `schema_contents` so every
        // schema file under the subtree gets the same treatment as any other moved
        // file — `rollback_directory_move_filesystem` below is always handed
        // `all_moves` (documents AND schema files), never the document-only `moves`.
        let mut written_dest: Vec<PathBuf> = Vec::new();
        for (_old_rel, new_rel, content_to_write) in contents.iter().chain(schema_contents.iter()) {
            let abs_dest = match safe_write_path(deps, new_rel) {
                Ok(p) => p,
                Err(e) => {
                    rollback_directory_move_filesystem(
                        &git_lock,
                        data_path_str,
                        &all_moves,
                        &abs_dest_dir,
                        &written_dest,
                        &[],
                    )
                    .await;
                    return Err(e.into());
                }
            };
            if let Some(parent) = abs_dest.parent()
                && let Err(e) = tokio::fs::create_dir_all(parent).await
            {
                rollback_directory_move_filesystem(
                    &git_lock,
                    data_path_str,
                    &all_moves,
                    &abs_dest_dir,
                    &written_dest,
                    &[],
                )
                .await;
                return Err(DirectoryMoveError::Io {
                    msg: format!(
                        "Failed to create parent directories for '{}': {}",
                        new_rel, e
                    ),
                });
            }

            let write_outcome: std::io::Result<()> = async {
                let mut file = tokio::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&abs_dest)
                    .await?;
                write_and_flush(&mut file, content_to_write.as_bytes()).await
            }
            .await;

            match write_outcome {
                Ok(()) => written_dest.push(abs_dest),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    // TOCTOU collision: guard 2 above walked the destination prefix
                    // and found it clear, but something has landed at this exact
                    // computed destination since then. This is the same benign,
                    // retryable race `write_document_move`'s equivalent
                    // `create_new` open maps to `WriteError::AlreadyExists` — map
                    // it identically here rather than letting it fall into the
                    // generic `Io` arm below, which `move_directory_error_to_mcp_error`
                    // reports as an opaque internal error instead of a clear
                    // "already exists" the caller can act on.
                    error!(
                        "Destination '{}' already exists (TOCTOU collision) while moving directory \
                     '{}' -> '{}'. Undoing every filesystem change made so far.",
                        new_rel, source_dir, dest_dir
                    );
                    rollback_directory_move_filesystem(
                        &git_lock,
                        data_path_str,
                        &all_moves,
                        &abs_dest_dir,
                        &written_dest,
                        &[],
                    )
                    .await;
                    return Err(DirectoryMoveError::AlreadyExists);
                }
                Err(e) => {
                    error!(
                        "Failed to write destination '{}' while moving directory '{}' -> '{}': {}. \
                     Undoing every filesystem change made so far.",
                        new_rel, source_dir, dest_dir, e
                    );
                    rollback_directory_move_filesystem(
                        &git_lock,
                        data_path_str,
                        &all_moves,
                        &abs_dest_dir,
                        &written_dest,
                        &[],
                    )
                    .await;
                    return Err(DirectoryMoveError::Io {
                        msg: format!(
                            "Failed to write destination '{}' during directory move: {}",
                            new_rel, e
                        ),
                    });
                }
            }
        }

        // Filesystem mutation, phase 2: remove every SOURCE, now that every
        // destination is confirmed written. A failure partway through is recovered
        // by restoring every source from HEAD and deleting every destination
        // written in phase 1 — nothing has touched git yet, so this is a pure
        // filesystem undo (see `rollback_directory_move_filesystem`'s doc comment
        // for why restoring the FULL source list, not just the ones already
        // removed, is safe). Chained with `schema_contents`, same reasoning as
        // phase 1 above.
        for (old_rel, _new_rel, _content_to_write) in contents.iter().chain(schema_contents.iter())
        {
            let abs_source = match safe_write_path(deps, old_rel) {
                Ok(p) => p,
                Err(e) => {
                    rollback_directory_move_filesystem(
                        &git_lock,
                        data_path_str,
                        &all_moves,
                        &abs_dest_dir,
                        &written_dest,
                        &[],
                    )
                    .await;
                    return Err(e.into());
                }
            };
            if let Err(e) = tokio::fs::remove_file(&abs_source).await {
                error!(
                    "Failed to remove source '{}' while moving directory '{}' -> '{}': {}. \
                 Restoring every source and deleting every written destination.",
                    old_rel, source_dir, dest_dir, e
                );
                rollback_directory_move_filesystem(
                    &git_lock,
                    data_path_str,
                    &all_moves,
                    &abs_dest_dir,
                    &written_dest,
                    &[],
                )
                .await;
                return Err(DirectoryMoveError::Io {
                    msg: format!(
                        "Failed to remove source '{}' during directory move: {}",
                        old_rel, e
                    ),
                });
            }
        }

        // Every document under `source_dir` is gone; tidy up any subdirectory (and
        // `source_dir` itself) that removing them left empty, best-effort — see
        // `remove_empty_dirs_best_effort`'s doc comment. Git does not track empty
        // directories, so this has no bearing on the commit below; it just keeps
        // the old prefix from lingering as an empty husk on disk.
        remove_empty_dirs_best_effort_async(&abs_source_dir).await;

        // Phase 3: rewrite documents OUTSIDE the moved subtree that link INTO it, so
        // those links keep resolving after the move — riding along in the SAME
        // commit as the move itself. Sources INSIDE the subtree are handled by the
        // outbound pass above and must never be processed again here.
        //
        // One batched `links_targeting_many` call over every moved path rather than
        // a `links_targeting` call per document: for a large subtree that was
        // hundreds of independent SQLite round-trips in a plain for-loop. The
        // per-source aggregation below is unchanged — a referencing document that
        // links to several moved targets still gets exactly one `outside_refs`
        // entry, with every target it references collected onto it.
        let mut outside_refs: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
        if let Some(state) = deps.state {
            let target_paths: Vec<String> =
                moves.iter().map(|(old_rel, _)| old_rel.clone()).collect();
            match state.links_targeting_many(&target_paths, "markdown").await {
                Ok(by_target) => {
                    for (old_rel, new_rel) in &moves {
                        let Some(referencing_paths) = by_target.get(old_rel.as_str()) else {
                            continue;
                        };
                        for ref_path in referencing_paths {
                            if moving.contains_key(ref_path.as_str()) {
                                // Inside the subtree — handled by the outbound
                                // rewrite above; processing it again here would
                                // double-edit it.
                                continue;
                            }
                            outside_refs
                                .entry(ref_path.clone())
                                .or_default()
                                .push((old_rel.clone(), new_rel.clone()));
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        "Skipping incoming-link rewrite for every moved document while moving \
                     directory '{}' -> '{}': the batched reverse-link query failed: {:#}",
                        source_dir, dest_dir, e
                    );
                }
            }
        }

        let mut rewritten_paths: Vec<String> = Vec::new();
        for (ref_path, targets) in &outside_refs {
            let abs_ref = match safe_write_path(deps, ref_path) {
                Ok(p) => p,
                Err(_) => {
                    warn!(
                        "Skipping link rewrite in '{}' while moving directory '{}' -> '{}': the \
                     path no longer resolves safely (stale document_links row?)",
                        ref_path, source_dir, dest_dir
                    );
                    continue;
                }
            };
            let body = match tokio::fs::read_to_string(&abs_ref).await {
                Ok(b) => b,
                Err(e) => {
                    warn!(
                        "Skipping link rewrite in '{}' while moving directory '{}' -> '{}': failed \
                     to read it, likely a stale document_links row: {}",
                        ref_path, source_dir, dest_dir, e
                    );
                    continue;
                }
            };

            let mut replacements: Vec<(crate::ingest::LinkOccurrence, String)> = Vec::new();
            for (old_rel, new_rel) in targets {
                let replacement = crate::ingest::relativize_md_path(ref_path, new_rel);
                for occurrence in crate::ingest::find_markdown_link_occurrences(&body, ref_path)
                    .into_iter()
                    .filter(|o| &o.resolved == old_rel)
                {
                    replacements.push((occurrence, replacement.clone()));
                }
            }
            if replacements.is_empty() {
                // Stale document_links row(s): nothing in the CURRENT body actually
                // resolves to any moved target anymore.
                continue;
            }

            let new_body = apply_link_replacements_each(&body, &replacements);
            if let Err(e) = tokio::fs::write(&abs_ref, new_body.as_bytes()).await {
                error!(
                    "Failed to rewrite links into '{}' while moving directory '{}' -> '{}': {}. \
                 Undoing every filesystem change made for this move so far.",
                    ref_path, source_dir, dest_dir, e
                );
                rollback_directory_move_filesystem(
                    &git_lock,
                    data_path_str,
                    &all_moves,
                    &abs_dest_dir,
                    &written_dest,
                    &rewritten_paths,
                )
                .await;
                return Err(DirectoryMoveError::Io {
                    msg: format!("Failed to rewrite links in '{}': {}", ref_path, e),
                });
            }
            rewritten_paths.push(ref_path.clone());
        }

        // Commit the move AND every rewritten referencing document as ONE atomic
        // commit, under the SAME lock acquisition (above) already held across
        // phases 1-3 — releasing it in between any of those and the commit would
        // let another writer stage into (and, since it commits its own path,
        // commit) the very half-staged state this call is about to undo. See
        // `write_document_move`'s identical comment for the full reasoning.
        let commit_message = build_commit_message(
            message,
            &format!("docs: move {} to {}", source_dir, dest_dir),
            "move_directory",
        );

        let mut commit_paths: Vec<&str> = Vec::new();
        for (old_rel, new_rel) in &all_moves {
            commit_paths.push(old_rel.as_str());
            commit_paths.push(new_rel.as_str());
        }
        for ref_path in &rewritten_paths {
            if !commit_paths.contains(&ref_path.as_str()) {
                commit_paths.push(ref_path.as_str());
            }
        }

        let commit_outcome = match git::commit_and_sync(
            &git_lock,
            deps.git_url,
            deps.branch,
            data_path_str,
            deps.token,
            &commit_paths,
            &commit_message,
            deps.commit_author_name,
            deps.commit_author_email,
        )
        .await
        {
            Ok(outcome) => outcome,

            Err(git::CommitSyncError::PreCommit(source_err)) => {
                error!(
                    "commit_and_sync pre-commit failure moving directory '{}' -> '{}', rolling \
                 back {} document(s) and {} rewritten referencing document(s): {:#}",
                    source_dir,
                    dest_dir,
                    all_moves.len(),
                    rewritten_paths.len(),
                    source_err
                );

                // Roll back EVERY part of this move — every source, every
                // destination, plus every referencing document rewritten above.
                // Each group is independent of the others and ALL of them always
                // run unconditionally, so a failure in one never leaves a
                // recoverable part undone. `rolled_back` is true only if every
                // single one of these succeeds.
                let mut rolled_back = true;
                for (old_rel, _new_rel) in &all_moves {
                    if let Err(e) = git::restore_from_head(&git_lock, data_path_str, old_rel).await
                    {
                        rolled_back = false;
                        error!(
                            "move_directory rollback: failed to restore source '{}': {:#}. This \
                         needs operator attention.",
                            old_rel, e
                        );
                    }
                }
                for (_old_rel, new_rel) in &all_moves {
                    let result = match safe_write_path(deps, new_rel) {
                        Ok(abs) => match tokio::fs::remove_file(&abs).await {
                            Ok(()) => git::unstage(&git_lock, data_path_str, new_rel).await,
                            Err(e) => Err(anyhow::Error::new(e).context(
                                "Failed to remove the new destination file during rollback",
                            )),
                        },
                        Err(e) => Err(anyhow::anyhow!(
                            "destination '{}' no longer resolves safely during rollback: {:?}",
                            new_rel,
                            e
                        )),
                    };
                    if let Err(e) = result {
                        rolled_back = false;
                        error!(
                            "move_directory rollback: failed to remove destination '{}': {:#}. \
                         This needs operator attention.",
                            new_rel, e
                        );
                    }
                }
                // Best-effort: tidy up any destination directory left empty by the
                // removals above — see `remove_empty_dirs_best_effort`'s doc comment.
                remove_empty_dirs_best_effort_async(&abs_dest_dir).await;
                for ref_path in &rewritten_paths {
                    if let Err(e) = git::restore_from_head(&git_lock, data_path_str, ref_path).await
                    {
                        rolled_back = false;
                        error!(
                            "move_directory rollback: failed to restore referencing document \
                         '{}': {:#}. This needs operator attention.",
                            ref_path, e
                        );
                    }
                }

                if !rolled_back {
                    error!(
                        "Rollback FAILED after a pre-commit git failure moving directory '{}' -> \
                     '{}'. Filesystem and git state may now be inconsistent. Original cause: \
                     {:#}",
                        source_dir, dest_dir, source_err
                    );
                }

                return Err(DirectoryMoveError::PreCommitFailed { rolled_back });
            }

            Err(git::CommitSyncError::PostCommit {
                sha,
                source: source_err,
            }) => {
                warn!(
                    "commit_and_sync post-commit (sync) failure moving directory '{}' -> '{}', \
                 commit {} stands uncorrected: {:#}",
                    source_dir, dest_dir, sha, source_err
                );

                // Still filtered through `mark_dirty` (#278): a moved path can
                // match `include` and still be excluded, same as the success path
                // below.
                mark_dirty(
                    deps.queue,
                    deps.indexing,
                    all_moves
                        .iter()
                        .flat_map(|(o, n)| [PathBuf::from(o.clone()), PathBuf::from(n.clone())])
                        .chain(rewritten_paths.iter().map(PathBuf::from))
                        .collect(),
                );

                return Ok(DirectoryMoveSuccess {
                    moved: all_moves,
                    rewritten_paths,
                    merged: false,
                });
            }

            Err(git::CommitSyncError::Conflict { source }) => {
                warn!(
                    "Remote changed underneath the directory move '{}' -> '{}' (attempt {}/{}), \
                 re-checking against fresh content: {:#}",
                    source_dir, dest_dir, attempt, MAX_WRITE_ATTEMPTS, source
                );
                head = sync_clone(deps, &git_lock, head).await;
                continue;
            }
        };

        let merged = rebase_touched(&commit_outcome.rebased_paths, &commit_paths);

        // Mark the source, the destination, and every rewritten referencing
        // document dirty — plus anything the rebase pulled in — in the SAME
        // marking call, same reasoning as `write_document_move`'s identical final
        // step. `all_moves` includes any relocated schema file alongside every
        // document, which is exactly what makes `reindex::unit_touches_schema`
        // force the shared `SchemaCache` to rebuild before this unit is next
        // indexed — see that function's doc comment; nothing further is needed
        // here for the post-commit self-correction this move depends on.
        mark_dirty(
            deps.queue,
            deps.indexing,
            all_moves
                .iter()
                .flat_map(|(o, n)| [PathBuf::from(o.clone()), PathBuf::from(n.clone())])
                .chain(rewritten_paths.iter().map(PathBuf::from))
                .chain(commit_outcome.rebased_paths.iter().cloned())
                .collect(),
        );

        return Ok(DirectoryMoveSuccess {
            moved: all_moves,
            rewritten_paths,
            merged,
        });
    }

    Err(DirectoryMoveError::EditedElsewhere)
}

// ---------------------------------------------------------------------------
// delete_document core
// ---------------------------------------------------------------------------

/// Delete `rel_path`: under the lock, after syncing with the remote, check it is
/// still the version the caller read (`expected_version`, required), remove it,
/// commit the deletion, and queue reindex cleanup on success.
///
/// `rel_path` must already have been resolved by the caller against the KB root
/// (this function re-resolves and re-verifies it itself immediately before each
/// filesystem action — see `write_document`'s doc comment for why).
pub async fn delete_document<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &WriteDeps<'_, E, Q>,
    rel_path: &str,
    message: Option<&str>,
    expected_version: Option<&str>,
) -> Result<WriteSuccess, WriteError> {
    // Schema-file guard, then include-pattern eligibility guard, then the early
    // path-safety check, ahead of everything else — see `write_document`.
    check_not_schema_file(rel_path)?;
    check_include_pattern(deps, rel_path)?;
    safe_write_path(deps, rel_path)?;

    // Validate the commit message BEFORE deleting anything: rejecting it after
    // the removal would leave the file gone from disk but never committed.
    validate_commit_message(message)?;

    if !safe_write_path(deps, rel_path)?.exists() {
        return Err(WriteError::NotFound);
    }
    let Some(expected_version) = expected_version else {
        return Err(WriteError::VersionRequired);
    };

    // Best-effort: warn if anything else in the KB still links to the document
    // about to be deleted (#181), and (#229) carry the same paths through to
    // `WriteSuccess::referencing_paths` — a `warn!` reaches an operator, not the
    // caller deciding whether the delete was a good idea. Deliberately
    // WARN/report, not refuse: a dangling link self-heals to a dropped edge on
    // the referencing document's own next reindex, same as any stale
    // `document_links` row. A read of the index, so it runs before the lock.
    let mut referencing_paths: Vec<String> = Vec::new();
    if let Some(state) = deps.state {
        match state.links_targeting(rel_path, "markdown").await {
            Ok(inbound) if !inbound.is_empty() => {
                warn!(
                    "Deleting '{}', which is still linked from {} other document(s): {}. \
                     This delete does not rewrite or remove those links — they will dangle \
                     until each referencing document's own next reindex drops the now-stale \
                     edge.",
                    rel_path,
                    inbound.len(),
                    inbound.join(", ")
                );
                referencing_paths = inbound;
            }
            Ok(_) => {}
            Err(e) => {
                warn!(
                    "Skipping inbound-link check before deleting '{}': the reverse-link \
                     query failed: {:#}",
                    rel_path, e
                );
            }
        }
    }

    let commit_message = build_commit_message(
        message,
        &format!("docs: delete {}", rel_path),
        "delete_document",
    );

    let (git_lock, mut head) = lock_and_sync(deps).await;
    let data_path_str = data_path_of(deps);

    for attempt in 1..=MAX_WRITE_ATTEMPTS {
        let abs_path = safe_write_path(deps, rel_path)?;
        let Some(old_content) = read_if_exists(&abs_path).await? else {
            return Err(WriteError::NotFound);
        };
        if !versions_match(expected_version, &old_content) {
            return Err(WriteError::EditedElsewhere);
        }

        tokio::fs::remove_file(&abs_path).await.map_err(|e| {
            error!("Failed to remove '{}': {}", abs_path.display(), e);
            WriteError::Io {
                msg: format!("Failed to remove file: {}", e),
            }
        })?;

        // `PreCommit`: HEAD never recorded the deletion — restore the file from
        // HEAD so the caller sees "nothing changed". `PostCommit`: the deletion is
        // a real local commit; leave it and report the sync as pending.
        // `Conflict`: the deletion commit is already dropped (the branch is back at
        // its pre-commit HEAD, which restores the file); sync again, then check
        // the document against what the remote now holds.
        let commit_outcome = match git::commit_and_sync(
            &git_lock,
            deps.git_url,
            deps.branch,
            data_path_str,
            deps.token,
            &[rel_path],
            &commit_message,
            deps.commit_author_name,
            deps.commit_author_email,
        )
        .await
        {
            Ok(outcome) => outcome,

            Err(git::CommitSyncError::PreCommit(source)) => {
                error!(
                    "commit_and_sync pre-commit failure deleting '{}', restoring from HEAD: {:#}",
                    rel_path, source
                );
                return match git::restore_from_head(&git_lock, data_path_str, rel_path).await {
                    Ok(()) => Err(WriteError::PreCommitFailed {
                        rolled_back: true,
                        msg: format!("{:#}", source),
                    }),
                    Err(restore_err) => {
                        error!(
                            "Restore FAILED after a pre-commit git failure deleting '{}': {:#}. \
                             Original cause: {:#}. The file is gone from disk and NOT \
                             committed — filesystem and git are now inconsistent.",
                            rel_path, restore_err, source
                        );
                        Err(WriteError::PreCommitFailed {
                            rolled_back: false,
                            msg: format!(
                                "Commit cause: {:#}. Restore cause: {:#}",
                                source, restore_err
                            ),
                        })
                    }
                };
            }

            Err(git::CommitSyncError::PostCommit { sha, source }) => {
                warn!(
                    "commit_and_sync post-commit (sync) failure deleting '{}', deletion commit \
                     {} stands uncorrected: {:#}",
                    rel_path, sha, source
                );
                mark_dirty(deps.queue, deps.indexing, vec![PathBuf::from(rel_path)]);
                return Ok(WriteSuccess {
                    outcome: WriteOutcome::CommittedPendingSync,
                    sha,
                    rebased_paths: Vec::new(),
                    diff: render_unified_diff(&old_content, "", rel_path),
                    rewritten_paths: Vec::new(),
                    referencing_paths,
                    merged: false,
                    version: None,
                });
            }

            Err(git::CommitSyncError::Conflict { source }) => {
                warn!(
                    "Remote changed underneath the delete of '{}' (attempt {}/{}), re-checking \
                     against fresh content: {:#}",
                    rel_path, attempt, MAX_WRITE_ATTEMPTS, source
                );
                head = sync_clone(deps, &git_lock, head).await;
                continue;
            }
        };

        // Mark this path — and anything the rebase pulled in — dirty. The worker's
        // scoped indexer purges a path's points and state rows itself once it finds
        // the file gone.
        mark_dirty(
            deps.queue,
            deps.indexing,
            std::iter::once(PathBuf::from(rel_path))
                .chain(commit_outcome.rebased_paths.iter().cloned())
                .collect(),
        );

        return Ok(WriteSuccess {
            outcome: WriteOutcome::Synced,
            sha: commit_outcome.sha,
            diff: render_unified_diff(&old_content, "", rel_path),
            rebased_paths: commit_outcome.rebased_paths,
            rewritten_paths: Vec::new(),
            referencing_paths,
            merged: false,
            version: None,
        });
    }

    Err(WriteError::EditedElsewhere)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ResolvedConfig;
    use crate::embed::EmbedClient;
    use crate::qdrant::QdrantStore;
    use std::sync::Arc;

    // -----------------------------------------------------------------------
    // dedup_verdict / build_dedup_query / dedup_search_opts — pure unit tests
    // (ported from mcp.rs; these are now write.rs's own logic)
    // -----------------------------------------------------------------------

    #[test]
    fn dedup_verdict_score_above_threshold_returns_hit() {
        let result = dedup_verdict(Some(("docs/existing.md".into(), 0.92)), 0.85);
        assert!(result.is_some());
        let hit = result.unwrap();
        assert_eq!(hit.file_path, "docs/existing.md");
        assert!((hit.score - 0.92).abs() < 1e-6);
    }

    #[test]
    fn dedup_verdict_no_results_allows() {
        assert!(dedup_verdict(None, 0.85).is_none());
    }

    #[test]
    fn build_dedup_query_prepends_description() {
        let q = build_dedup_query("Body text here.", Some("A short summary."), true);
        assert_eq!(q, "A short summary.\n\nBody text here.");
    }

    #[test]
    fn build_dedup_query_truncates_to_limit() {
        let long_body = "x".repeat(DEDUP_QUERY_CHAR_LIMIT * 2);
        let q = build_dedup_query(&long_body, None, false);
        assert_eq!(q.chars().count(), DEDUP_QUERY_CHAR_LIMIT);
    }

    #[test]
    fn dedup_search_opts_is_dense_only() {
        let opts = dedup_search_opts();
        assert!(!opts.hybrid);
        assert_eq!(opts.limit, 1);
        assert!(opts.min_score.is_none());
    }

    // -----------------------------------------------------------------------
    // resolve_safe_write_path unit tests (ported from mcp.rs; the function
    // itself moved here too — see this module's `resolve_safe_write_path`)
    // -----------------------------------------------------------------------

    #[test]
    fn safe_write_path_treats_a_leading_slash_as_the_kb_root() {
        // Callers cannot know where the KB lives inside the container, so `/x.md` and
        // `x.md` must address the same document. The escape checks still apply — this
        // resolves under the data root, it does not reach the real /etc.
        let tmp = tempfile::tempdir().unwrap();

        let rooted = resolve_safe_write_path(tmp.path(), "/notes/a.md").unwrap();
        let relative = resolve_safe_write_path(tmp.path(), "notes/a.md").unwrap();
        assert_eq!(rooted, relative);
        assert!(rooted.starts_with(tmp.path()));

        // And traversal is still rejected however it is spelled.
        assert!(resolve_safe_write_path(tmp.path(), "/../escape.md").is_err());
        assert!(resolve_safe_write_path(tmp.path(), "../escape.md").is_err());
    }

    #[test]
    fn safe_write_path_rejects_parent_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let result = resolve_safe_write_path(tmp.path(), "../../etc/shadow");
        assert!(result.is_err(), "parent-dir component must be rejected");
        let msg = result.unwrap_err();
        assert!(msg.contains(".."), "error should mention '..', got: {msg}");
    }

    #[test]
    fn safe_write_path_accepts_normal_nested_path() {
        let tmp = tempfile::tempdir().unwrap();
        // The file doesn't need to exist; the ancestor (tmp itself) does.
        let result = resolve_safe_write_path(tmp.path(), "subdir/docs/guide.md");
        assert!(
            result.is_ok(),
            "normal nested path should be accepted, got: {:?}",
            result
        );
        let abs = result.unwrap();
        assert!(
            abs.starts_with(tmp.path()),
            "returned path should be under data_root"
        );
    }

    #[test]
    fn safe_write_path_rejects_symlinked_ancestor_outside_root() {
        // Create two separate temp directories.
        let inside_tmp = tempfile::tempdir().unwrap();
        let outside_tmp = tempfile::tempdir().unwrap();

        // Create a subdirectory inside inside_tmp that is actually a symlink
        // pointing to outside_tmp.
        let escaped_dir = inside_tmp.path().join("escaped");
        std::os::unix::fs::symlink(outside_tmp.path(), &escaped_dir)
            .expect("failed to create symlink");

        // A path through the symlink: "escaped/secret.md"
        // Lexically this is under inside_tmp, but canonically it resolves outside.
        let result = resolve_safe_write_path(inside_tmp.path(), "escaped/secret.md");

        assert!(
            result.is_err(),
            "path through symlinked ancestor pointing outside root must be rejected"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("symlink") || msg.contains("escapes"),
            "error should mention symlink or escape, got: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // build_commit_message / render_unified_diff — pure unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn commit_message_has_trailers() {
        let msg = build_commit_message(None, "docs: add notes/guide.md", "create_document");
        assert!(msg.contains("Tool: mcp-md-wiki"));
        assert!(msg.contains("Operation: create_document"));
        assert!(msg.starts_with("docs: add notes/guide.md"));
    }

    #[test]
    fn commit_message_user_subject_overrides_default() {
        let msg = build_commit_message(Some("chore: x"), "docs: add y", "create_document");
        assert!(msg.starts_with("chore: x"));
    }

    #[test]
    fn diff_create_shows_all_additions() {
        let diff = render_unified_diff("", "line1\n", "docs/new.md");
        assert!(diff.contains("+line1"));
    }

    #[test]
    fn diff_delete_shows_all_removals() {
        let diff = render_unified_diff("line1\n", "", "docs/gone.md");
        assert!(diff.contains("-line1"));
    }

    // -----------------------------------------------------------------------
    // (#179) split_frontmatter_bytes / apply_frontmatter_patch / apply_append —
    // pure unit tests. These are the exact edge cases #179 calls out by name:
    // no frontmatter, an empty file, and no trailing newline.
    // -----------------------------------------------------------------------

    #[test]
    fn split_frontmatter_bytes_normal_document() {
        let content = "---\ntitle: X\n---\n\n# Body\ntext\n";
        let (fm, body) = split_frontmatter_bytes(content).unwrap();
        assert_eq!(fm, "---\ntitle: X\n---\n");
        assert_eq!(body, "\n# Body\ntext\n");
        assert_eq!(format!("{fm}{body}"), content, "split must be lossless");
    }

    #[test]
    fn split_frontmatter_bytes_no_trailing_newline_on_body() {
        let content = "---\ntitle: X\n---\n\n# Body\ntext";
        let (fm, body) = split_frontmatter_bytes(content).unwrap();
        assert_eq!(fm, "---\ntitle: X\n---\n");
        assert_eq!(body, "\n# Body\ntext");
    }

    #[test]
    fn split_frontmatter_bytes_no_frontmatter_returns_none() {
        assert!(split_frontmatter_bytes("# Just a doc\nbody text\n").is_none());
    }

    #[test]
    fn split_frontmatter_bytes_empty_content_returns_none() {
        assert!(split_frontmatter_bytes("").is_none());
    }

    #[test]
    fn split_frontmatter_bytes_unterminated_delimiter_returns_none() {
        // Opens with `---` but never closes — must not be mistaken for a
        // (frontmatter, "") split.
        assert!(split_frontmatter_bytes("---\ntitle: X\nno closing delimiter\n").is_none());
    }

    #[test]
    fn split_frontmatter_bytes_dashes_inside_a_value_are_not_the_closing_delimiter() {
        let content = "---\ndescription: a---b\n---\n\nBody\n";
        let (fm, body) = split_frontmatter_bytes(content).unwrap();
        assert_eq!(fm, "---\ndescription: a---b\n---\n");
        assert_eq!(body, "\nBody\n");
    }

    #[test]
    fn apply_frontmatter_patch_set_field_on_existing_document() {
        let old = "---\ntitle: X\nstatus: draft\n---\n\n# Body\n";
        let new = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::SetField {
                field: "status".into(),
                value: serde_json::json!("active"),
            }],
        )
        .unwrap();
        let (fm, body) = validate::parse_frontmatter_raw(&new);
        assert_eq!(fm.get("status").unwrap(), "active");
        assert_eq!(fm.get("title").unwrap(), "X");
        assert_eq!(body.trim(), "# Body");
    }

    #[test]
    fn apply_frontmatter_patch_never_touches_the_body() {
        let old = "---\ntitle: X\n---\n\n# Body\nwith --- a dash-line\nand text\n";
        let new = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::SetField {
                field: "title".into(),
                value: serde_json::json!("Y"),
            }],
        )
        .unwrap();
        assert!(new.ends_with("# Body\nwith --- a dash-line\nand text\n"));
    }

    #[test]
    fn apply_frontmatter_patch_creates_frontmatter_when_absent() {
        let old = "# Just a doc\nno frontmatter here\n";
        let new = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::SetField {
                field: "title".into(),
                value: serde_json::json!("New Title"),
            }],
        )
        .unwrap();
        let (fm, body) = validate::parse_frontmatter_raw(&new);
        assert_eq!(fm.get("title").unwrap(), "New Title");
        assert_eq!(body.trim(), "# Just a doc\nno frontmatter here");
    }

    #[test]
    fn apply_frontmatter_patch_on_empty_file_creates_frontmatter_only() {
        let new = apply_frontmatter_patch(
            "",
            &[FrontmatterEdit::SetField {
                field: "title".into(),
                value: serde_json::json!("T"),
            }],
        )
        .unwrap();
        assert!(new.starts_with("---\n"));
        assert!(new.trim_end().ends_with("---"));
    }

    #[test]
    fn apply_frontmatter_patch_remove_field_errors_when_absent() {
        let old = "---\ntitle: X\n---\n\nBody\n";
        let err = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::RemoveField {
                field: "nonexistent".into(),
            }],
        )
        .unwrap_err();
        assert!(err.contains("nonexistent"), "got: {err}");
    }

    #[test]
    fn apply_frontmatter_patch_remove_field_removes_when_present() {
        let old = "---\ntitle: X\nstatus: draft\n---\n\nBody\n";
        let new = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::RemoveField {
                field: "status".into(),
            }],
        )
        .unwrap();
        let (fm, _) = validate::parse_frontmatter_raw(&new);
        assert!(!fm.contains_key("status"));
        assert!(fm.contains_key("title"));
    }

    #[test]
    fn apply_frontmatter_patch_add_values_creates_and_dedupes() {
        let old = "---\ntitle: X\ntags: [a]\n---\n\nBody\n";
        let new = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::AddValues {
                field: "tags".into(),
                values: vec![serde_json::json!("a"), serde_json::json!("b")],
            }],
        )
        .unwrap();
        let (fm, _) = validate::parse_frontmatter_raw(&new);
        let tags: Vec<String> = fm
            .get("tags")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(tags, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn apply_frontmatter_patch_add_values_creates_field_when_absent() {
        let old = "---\ntitle: X\n---\n\nBody\n";
        let new = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::AddValues {
                field: "tags".into(),
                values: vec![serde_json::json!("new")],
            }],
        )
        .unwrap();
        let (fm, _) = validate::parse_frontmatter_raw(&new);
        assert_eq!(
            fm.get("tags").unwrap().as_array().unwrap(),
            &vec![serde_json::json!("new")]
        );
    }

    #[test]
    fn apply_frontmatter_patch_add_values_errors_on_non_list_field() {
        let old = "---\ntitle: X\n---\n\nBody\n";
        let err = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::AddValues {
                field: "title".into(),
                values: vec![serde_json::json!("x")],
            }],
        )
        .unwrap_err();
        assert!(err.contains("title"), "got: {err}");
    }

    #[test]
    fn apply_frontmatter_patch_remove_values_filters_the_list() {
        let old = "---\ntitle: X\ntags: [a, b, c]\n---\n\nBody\n";
        let new = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::RemoveValues {
                field: "tags".into(),
                values: vec![serde_json::json!("b")],
            }],
        )
        .unwrap();
        let (fm, _) = validate::parse_frontmatter_raw(&new);
        let tags: Vec<String> = fm
            .get("tags")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(tags, vec!["a".to_string(), "c".to_string()]);
    }

    #[test]
    fn apply_frontmatter_patch_remove_values_errors_when_field_absent() {
        let old = "---\ntitle: X\n---\n\nBody\n";
        let err = apply_frontmatter_patch(
            old,
            &[FrontmatterEdit::RemoveValues {
                field: "tags".into(),
                values: vec![serde_json::json!("a")],
            }],
        )
        .unwrap_err();
        assert!(err.contains("tags"), "got: {err}");
    }

    #[test]
    fn apply_frontmatter_patch_multiple_edits_apply_in_order() {
        let old = "---\ntitle: X\nstatus: draft\ntags: [a]\n---\n\nBody\n";
        let new = apply_frontmatter_patch(
            old,
            &[
                FrontmatterEdit::SetField {
                    field: "status".into(),
                    value: serde_json::json!("active"),
                },
                FrontmatterEdit::AddValues {
                    field: "tags".into(),
                    values: vec![serde_json::json!("b")],
                },
                FrontmatterEdit::RemoveField {
                    field: "title".into(),
                },
            ],
        )
        .unwrap();
        let (fm, _) = validate::parse_frontmatter_raw(&new);
        assert_eq!(fm.get("status").unwrap(), "active");
        assert!(!fm.contains_key("title"));
        assert_eq!(
            fm.get("tags").unwrap().as_array().unwrap().len(),
            2,
            "got: {:?}",
            fm.get("tags")
        );
    }

    #[test]
    fn apply_append_no_frontmatter() {
        let result = apply_append("existing line\n", "new entry");
        assert_eq!(result, "existing line\nnew entry\n");
    }

    /// A CRLF document must stay entirely CRLF after an append.
    ///
    /// Both halves matter: the separator this function inserts, and the
    /// caller's own text, which arrives LF-terminated because an agent
    /// composing an append has no idea what the file on disk uses. Getting
    /// either wrong leaves a file with mixed endings — no bytes lost, but
    /// every later diff of that document shows lines nobody edited.
    #[test]
    fn apply_append_preserves_crlf_line_endings() {
        let result = apply_append("# Body\r\nold line\r\n", "new entry");
        assert_eq!(result, "# Body\r\nold line\r\nnew entry\r\n");

        let multi = apply_append("# Body\r\n", "first\nsecond");
        assert_eq!(multi, "# Body\r\nfirst\r\nsecond\r\n");

        let no_trailing = apply_append("# Body\r\nold line", "new entry");
        assert_eq!(no_trailing, "# Body\r\nold line\r\nnew entry\r\n");
    }

    /// An LF document must not acquire CRLF from CRLF-terminated input text.
    #[test]
    fn apply_append_normalizes_crlf_input_into_an_lf_document() {
        let result = apply_append("# Body\nold line\n", "first\r\nsecond");
        assert_eq!(result, "# Body\nold line\nfirst\nsecond\n");
    }

    /// A frontmatter patch re-serializes the whole block, so it is the other
    /// place a CRLF document can silently become mixed — `serde_yaml_ng`
    /// always emits LF.
    #[test]
    fn apply_frontmatter_patch_preserves_crlf_line_endings() {
        let doc = "---\r\ntitle: X\r\nstatus: draft\r\n---\r\n\r\n# Body\r\ntext\r\n";
        let result = apply_frontmatter_patch(
            doc,
            &[FrontmatterEdit::SetField {
                field: "status".into(),
                value: serde_json::json!("active"),
            }],
        )
        .unwrap();
        // Deliberately not asserting the absence of "\n\r": two adjacent CRLF
        // endings contain that sequence at their boundary, so it says nothing.
        // The per-line check below is the real invariant.
        for line in result.split_inclusive('\n') {
            assert!(
                !line.ends_with('\n') || line.ends_with("\r\n"),
                "every terminated line must keep CRLF, got {line:?}"
            );
        }
        assert!(
            result.contains("# Body\r\ntext\r\n"),
            "body must survive byte-exact: {result:?}"
        );
    }

    fn set(field: &str, value: serde_json::Value) -> FrontmatterEdit {
        FrontmatterEdit::SetField {
            field: field.into(),
            value,
        }
    }

    const FOLDED_DOC: &str = concat!(
        "---\n",
        "title: Samosadillas\n",
        "description: >-\n",
        "    Curry-spiced filling folded into tortillas\n",
        "    and pan-fried until crisp.\n",
        "# the role drives meal planning\n",
        "type: recipe\n",
        "tags: [skillet, appetizer]\n",
        "\n",
        "planning:\n",
        "  servings: 16\n",
        "  scale_limit: >-\n",
        "    The batch already sits at the fridge ceiling\n",
        "    from one cook.\n",
        "---\n\n# Body\n"
    );

    /// #269: a patch must leave fields it did not touch byte-for-byte alone.
    #[test]
    fn frontmatter_patch_preserves_untouched_formatting() {
        let new =
            apply_frontmatter_patch(FOLDED_DOC, &[set("type", serde_json::json!("dish"))]).unwrap();
        assert_eq!(new, FOLDED_DOC.replace("type: recipe", "type: dish"));
    }

    #[test]
    fn frontmatter_patch_rerenders_only_the_changed_field_in_place() {
        let new =
            apply_frontmatter_patch(FOLDED_DOC, &[set("tags", serde_json::json!(["skillet"]))])
                .unwrap();
        // Key order, the `>-` blocks and the comment survive; the edited
        // list moves from flow to block style and the blank line stays.
        assert!(new.starts_with("---\ntitle: Samosadillas\ndescription: >-\n    Curry"));
        assert!(new.contains(
            "# the role drives meal planning\ntype: recipe\ntags:\n- skillet\n\nplanning:"
        ));
        assert!(new.contains("  scale_limit: >-\n    The batch already"));
        assert!(new.ends_with("---\n\n# Body\n"));
    }

    #[test]
    fn frontmatter_patch_nested_edit_rerenders_its_top_level_field_only() {
        let new = apply_frontmatter_patch(
            FOLDED_DOC,
            &[set("planning.servings", serde_json::json!(8))],
        )
        .unwrap();
        assert!(new.contains("description: >-\n    Curry-spiced"), "{new}");
        assert!(new.contains("tags: [skillet, appetizer]\n"), "{new}");
        let (fm, _) = validate::parse_frontmatter_raw(&new);
        assert_eq!(fm["planning"]["servings"], 8);
    }

    #[test]
    fn frontmatter_patch_remove_field_drops_the_whole_entry_and_appends_new_keys() {
        let new = apply_frontmatter_patch(
            FOLDED_DOC,
            &[
                FrontmatterEdit::RemoveField {
                    field: "description".into(),
                },
                set("status", serde_json::json!("active")),
            ],
        )
        .unwrap();
        assert!(!new.contains("Curry-spiced"), "{new}");
        assert!(new.contains("status: active\n---\n"), "{new}");
        assert!(
            new.starts_with("---\ntitle: Samosadillas\n# the role"),
            "{new}"
        );
    }

    #[test]
    fn frontmatter_patch_crlf_splice_keeps_every_line_crlf() {
        let doc =
            "---\r\ntitle: X\r\nnote: >-\r\n  a\r\n  b\r\nstatus: draft\r\n---\r\n\r\nBody\r\n";
        let new =
            apply_frontmatter_patch(doc, &[set("status", serde_json::json!("active"))]).unwrap();
        assert_eq!(new, doc.replace("draft", "active"));
    }

    #[test]
    fn frontmatter_patch_falls_back_for_flow_style_top_level() {
        let doc = "---\n{title: X, status: draft}\n---\n\nBody\n";
        let new =
            apply_frontmatter_patch(doc, &[set("status", serde_json::json!("active"))]).unwrap();
        let (fm, body) = validate::parse_frontmatter_raw(&new);
        assert_eq!(fm["status"], "active");
        assert_eq!(fm["title"], "X");
        assert_eq!(body.trim(), "Body");
    }

    #[test]
    fn frontmatter_patch_falls_back_for_anchors_and_merge_keys() {
        let doc = "---\nbase: &b\n  a: 1\nderived:\n  <<: *b\n  c: 2\n---\n\nBody\n";
        let new = apply_frontmatter_patch(doc, &[set("x", serde_json::json!(1))]).unwrap();
        let (before, _) = validate::parse_frontmatter_raw(doc);
        let (after, _) = validate::parse_frontmatter_raw(&new);
        assert_eq!(after["x"], 1);
        assert_eq!(after["base"], before["base"]);
        assert_eq!(after["derived"], before["derived"]);
    }

    #[test]
    fn splice_refuses_a_block_it_cannot_segment() {
        let original: HashMap<String, serde_json::Value> = HashMap::new();
        assert!(
            splice_frontmatter_block("---\nstray text\n---\n", &original, &original, "\n")
                .is_none()
        );
        assert!(
            splice_frontmatter_block("---\na: 1\na: 2\n---\n", &original, &original, "\n")
                .is_none()
        );
    }

    #[test]
    fn apply_append_no_trailing_newline_on_existing_content() {
        let result = apply_append("existing line", "new entry");
        assert_eq!(result, "existing line\nnew entry\n");
    }

    #[test]
    fn apply_append_empty_file() {
        let result = apply_append("", "first entry");
        assert_eq!(result, "first entry\n");
    }

    #[test]
    fn apply_append_with_frontmatter_and_body() {
        let old = "---\ntitle: X\n---\n\n# Log\n- entry one\n";
        let result = apply_append(old, "- entry two");
        assert_eq!(
            result,
            "---\ntitle: X\n---\n\n# Log\n- entry one\n- entry two\n"
        );
    }

    #[test]
    fn apply_append_with_frontmatter_and_no_body_inserts_a_separator() {
        let old = "---\ntitle: X\n---\n";
        let result = apply_append(old, "first body line");
        assert_eq!(result, "---\ntitle: X\n---\n\nfirst body line\n");
    }

    #[test]
    fn apply_append_never_touches_the_frontmatter_block() {
        let old = "---\ntitle: X\ndescription: a---b\n---\n\nBody\n";
        let result = apply_append(old, "more");
        assert!(result.starts_with("---\ntitle: X\ndescription: a---b\n---\n"));
    }

    #[test]
    fn apply_append_strips_a_trailing_newline_the_caller_included() {
        // Whether the caller's `text` ends in `\n` or not, the result always
        // has exactly one trailing newline — never two.
        let a = apply_append("body\n", "entry\n");
        let b = apply_append("body\n", "entry");
        assert_eq!(a, b);
        assert!(a.ends_with("entry\n") && !a.ends_with("entry\n\n"));
    }

    // -----------------------------------------------------------------------
    // Test harness: a WriteDeps backed by a temp dir and (unreachable) real
    // EmbedClient/QdrantStore — matching the pattern `mcp.rs`'s own write tests
    // use, since dedup is disabled by default in `make_test_resolved_config`.
    // -----------------------------------------------------------------------

    fn test_embed_and_qdrant() -> (Arc<EmbedClient>, Arc<QdrantStore>) {
        let qdrant_config = crate::config::ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: "test".into(),
        };
        let qdrant = Arc::new(QdrantStore::new(&qdrant_config).unwrap());
        let embed_config = crate::config::ResolvedEmbeddingConfig {
            base_url: "http://localhost:8080/v1".into(),
            model: "test".into(),
            api_key: None,
            vector_size: 768,
            batch_size: 32,
            request_timeout_secs: 60,
            batch_concurrency: 4,
        };
        let embed = Arc::new(EmbedClient::new(&embed_config));
        (embed, qdrant)
    }

    /// Bundle owning everything `WriteDeps<'_, EmbedClient, QdrantStore>` borrows,
    /// so a test can build the deps and hold this alive for the call.
    ///
    /// `state_db` is `None` by default — matching `WriteDeps::state`'s own
    /// "disabled unless explicitly wired up" semantics — so every existing test
    /// that never calls `with_state_db` keeps exercising the no-rewrite path.
    /// Tests that exercise link rewriting call `with_state_db` to open a real
    /// (temp-file-backed) `StateDb` and seed it via `StateDb::replace_links`.
    struct Harness {
        embed: Arc<EmbedClient>,
        qdrant: Arc<QdrantStore>,
        canonical_data_path: PathBuf,
        include_patterns: globset::GlobSet,
        schema_cache: SharedSchemaCache,
        config: Arc<ResolvedConfig>,
        token: Option<String>,
        state_db: Option<StateDb>,
        /// Fresh, private to this `Harness` instance — this is the whole point
        /// of `WriteDeps::queue` becoming an injected dependency: every test
        /// that builds its own `Harness` gets its own `ReindexQueue`, so a path
        /// literal used by another test's harness cannot collide with this
        /// one's, structurally rather than by convention. See
        /// `same_path_literal_in_two_independent_tests_a`/`_b` for the
        /// regression guard this makes possible.
        reindex_queue: crate::reindex::ReindexQueue,
        /// Keeps the state DB's backing temp directory alive for as long as the
        /// harness lives. Deliberately a SEPARATE temp dir from the KB root
        /// (`canonical_data_path`) — the state DB file must never sit inside the
        /// git working copy, or every git-backed test's `git status --porcelain
        /// == ""` assertion would start seeing it as an untracked file.
        _state_db_dir: Option<tempfile::TempDir>,
    }

    impl Harness {
        /// The current version of `rel_path` under this harness's root, as a
        /// caller would have read it; `None` when it does not exist.
        fn version_of(&self, rel_path: &str) -> Option<&'static str> {
            std::fs::read(self.canonical_data_path.join(rel_path))
                .ok()
                .map(|bytes| leak(document_version(&bytes)))
        }

        fn new(tmp: &tempfile::TempDir, config: Arc<ResolvedConfig>) -> Self {
            let (embed, qdrant) = test_embed_and_qdrant();
            let canonical_data_path = tmp.path().canonicalize().unwrap();
            let mut builder = globset::GlobSetBuilder::new();
            builder.add(globset::Glob::new("**/*.md").unwrap());
            let schema_cache: SharedSchemaCache = Arc::new(std::sync::RwLock::new(Arc::new(
                crate::schema::SchemaCache::build_for_test_with(
                    &canonical_data_path,
                    &config.frontmatter,
                    &config.indexing,
                ),
            )));
            Harness {
                embed,
                qdrant,
                canonical_data_path,
                include_patterns: builder.build().unwrap(),
                schema_cache,
                config,
                token: None,
                state_db: None,
                reindex_queue: crate::reindex::ReindexQueue::new(),
                _state_db_dir: None,
            }
        }

        /// Open a real, temp-file-backed `StateDb` (in its own directory, NOT
        /// the KB root — see `_state_db_dir`'s doc comment) and attach it, so
        /// `deps()` passes `Some` for `WriteDeps::state` and
        /// `write_document_move` performs link rewriting. Returns `self` so
        /// callers can chain it onto `Harness::new(..)`.
        async fn with_state_db(mut self) -> Self {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("state.db");
            self.state_db = Some(StateDb::new(&db_path).await.unwrap());
            self._state_db_dir = Some(dir);
            self
        }

        fn deps(&self) -> WriteDeps<'_, EmbedClient, QdrantStore> {
            WriteDeps {
                retrieval: RetrievalDeps {
                    embed_client: &self.embed,
                    qdrant: &self.qdrant,
                    collection: &self.config.qdrant.collection,
                    data_path: &self.canonical_data_path,
                    include_patterns: &self.include_patterns,
                    reranker: None,
                },
                canonical_data_path: &self.canonical_data_path,
                schema_cache: &self.schema_cache,
                validation: &self.config.validation,
                indexing: &self.config.indexing,
                prepend_description: self.config.chunking.prepend_description,
                dedup_enabled: self.config.write.dedup_enabled,
                dedup_threshold: self.config.write.dedup_threshold,
                git_url: self.config.source.git_url.as_deref(),
                branch: &self.config.source.branch,
                token: self.token.as_deref(),
                commit_author_name: &self.config.write.commit_author_name,
                commit_author_email: &self.config.write.commit_author_email,
                queue: &self.reindex_queue,
                state: self.state_db.as_ref(),
            }
        }
    }

    /// A whole-content change that ignores what is on disk: a relative edit whose
    /// result is always `new_content`, so it needs no `expected_version`. Stands in
    /// for the blind overwrites these tests were written against.
    fn overwrite<'a>(new_content: &'a str) -> DocChange<'a> {
        let edit: &'a RelativeEdit<'a> =
            Box::leak(Box::new(move |_: &str| Ok(new_content.to_string())));
        DocChange::Relative(edit)
    }

    fn leak(s: String) -> &'static str {
        Box::leak(s.into_boxed_str())
    }

    fn make_req<'a>(rel_path: &'a str, new_content: &'a str, is_create: bool) -> WriteRequest<'a> {
        WriteRequest {
            rel_path,
            change: if is_create {
                DocChange::Create(new_content)
            } else {
                overwrite(new_content)
            },
            message: None,
            default_verb: if is_create { "add" } else { "update" },
            force_new: Some(true),
            operation: "test",
            expected_version: None,
            dest_path: None,
        }
    }

    /// A move of a source whose current content is `old_content` (its version is
    /// passed as `expected_version`), writing `new_content` at the destination.
    fn make_move_req<'a>(
        source_rel: &'a str,
        dest_rel: &'a str,
        old_content: &'a str,
        new_content: &'a str,
    ) -> WriteRequest<'a> {
        WriteRequest {
            rel_path: source_rel,
            change: overwrite(new_content),
            message: None,
            default_verb: "update",
            force_new: Some(true),
            operation: "test",
            expected_version: Some(leak(document_version(old_content.as_bytes()))),
            dest_path: Some(dest_rel),
        }
    }

    /// A schema `dedup:` key overrides the global `write.*` value per key; an unset
    /// key falls back to it (#272).
    #[test]
    fn effective_dedup_prefers_schema_override_per_key() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = (*crate::mcp::make_test_resolved_config(tmp.path())).clone();
        config.write.dedup_enabled = true;
        config.write.dedup_threshold = 0.8;
        let harness = Harness::new(&tmp, Arc::new(config));
        let deps = harness.deps();

        let unset = schema::ResolvedSchema::default();
        assert_eq!(effective_dedup(&deps, &unset), (true, 0.8));

        let disabled = schema::ResolvedSchema {
            dedup_enabled: Some(false),
            ..Default::default()
        };
        assert_eq!(effective_dedup(&deps, &disabled), (false, 0.8));

        let stricter = schema::ResolvedSchema {
            dedup_threshold: Some(0.97),
            ..Default::default()
        };
        assert_eq!(effective_dedup(&deps, &stricter), (true, 0.97));

        let enabled_when_global_off = {
            let mut config = (*crate::mcp::make_test_resolved_config(tmp.path())).clone();
            config.write.dedup_enabled = false;
            Harness::new(&tmp, Arc::new(config))
        };
        let schema = schema::ResolvedSchema {
            dedup_enabled: Some(true),
            ..Default::default()
        };
        assert!(effective_dedup(&enabled_when_global_off.deps(), &schema).0);
    }

    // -----------------------------------------------------------------------
    // write_document core tests (ported from mcp.rs's write/delete suite,
    // exercised directly against `write::write_document`/`write::delete_document`
    // rather than through the MCP tool layer)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn full_replace_without_expected_version_is_refused_before_touching_the_file() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);
        std::fs::create_dir_all(tmp.path().join("docs")).unwrap();
        let old = "---\ntitle: Old\n---\n# Old";
        std::fs::write(tmp.path().join("docs/edit-me.md"), old).unwrap();

        let mut req = make_req("docs/edit-me.md", "---\ntitle: New\n---\n# New", false);
        req.change = DocChange::Replace("---\ntitle: New\n---\n# New");
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(matches!(err, WriteError::VersionRequired), "{err:?}");
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("docs/edit-me.md")).unwrap(),
            old
        );
    }

    #[tokio::test]
    async fn stale_version_with_no_merge_base_is_refused_and_mutates_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);
        std::fs::create_dir_all(tmp.path().join("docs")).unwrap();
        let old = "---\ntitle: Old\n---\n# Old";
        std::fs::write(tmp.path().join("docs/edit-me.md"), old).unwrap();

        // A version the object store has never seen cannot be a merge base.
        let stale = document_version(b"not the current content");
        let mut req = make_req("docs/edit-me.md", "---\ntitle: New\n---\n# New", false);
        req.change = DocChange::Replace("---\ntitle: New\n---\n# New");
        req.expected_version = Some(&stale);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(matches!(err, WriteError::EditedElsewhere), "{err:?}");
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("docs/edit-me.md")).unwrap(),
            old
        );
    }

    #[test]
    fn document_version_is_the_git_blob_id() {
        // `git hash-object` of "hello\n".
        assert_eq!(
            document_version(b"hello\n"),
            "ce013625030ba8dba906f756967f9e9ca394464a"
        );
    }

    #[tokio::test]
    async fn a_schema_file_path_is_refused_by_every_document_write_even_with_a_widened_include() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("notes");
        std::fs::create_dir_all(&sub).unwrap();
        let schema_text = "fields:\n  title:\n    required: true\n";
        std::fs::write(sub.join(crate::schema::SCHEMA_FILE_NAME), schema_text).unwrap();
        std::fs::write(sub.join("doc.md"), "---\ntitle: T\n---\n# Body").unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let mut harness = Harness::new(&tmp, config);
        // An include that would otherwise admit the schema file: the guard must
        // not depend on `indexing.include`.
        let mut builder = globset::GlobSetBuilder::new();
        builder.add(globset::Glob::new("**/*").unwrap());
        harness.include_patterns = builder.build().unwrap();
        let schema_rel = "notes/.kb-schema.yaml";
        let is_refused = |err: &WriteError| matches!(err, WriteError::SchemaFile { rel_path } if rel_path == schema_rel);

        let edit = make_req(schema_rel, "fields: {}\n", false);
        let err = write_document(&harness.deps(), edit).await.unwrap_err();
        assert!(is_refused(&err), "edit: {err:?}");

        let err = write_document(&harness.deps(), make_req(schema_rel, "fields: {}\n", true))
            .await
            .unwrap_err();
        assert!(is_refused(&err), "create: {err:?}");

        let err = delete_document(
            &harness.deps(),
            schema_rel,
            None,
            harness.version_of(schema_rel),
        )
        .await
        .unwrap_err();
        assert!(is_refused(&err), "delete: {err:?}");

        let err = write_document(
            &harness.deps(),
            make_move_req(schema_rel, "notes/moved.yaml", schema_text, schema_text),
        )
        .await
        .unwrap_err();
        assert!(is_refused(&err), "move from: {err:?}");

        let doc = "---\ntitle: T\n---\n# Body";
        let err = write_document(
            &harness.deps(),
            make_move_req("notes/doc.md", "other/.kb-schema.yaml", doc, doc),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(&err, WriteError::SchemaFile { rel_path } if rel_path == "other/.kb-schema.yaml"),
            "move onto: {err:?}"
        );

        // The canonical name is refused exactly like the legacy one.
        let canonical = "notes/.schema.yaml";
        let err = write_document(&harness.deps(), make_req(canonical, "fields: {}\n", false))
            .await
            .unwrap_err();
        assert!(
            matches!(&err, WriteError::SchemaFile { rel_path } if rel_path == canonical),
            "canonical edit: {err:?}"
        );

        let batch = [BatchWriteRequest {
            rel_path: schema_rel,
            change: overwrite("fields: {}\n"),
            force_new: Some(true),
            expected_version: None,
        }];
        match write_documents_batch(&harness.deps(), &batch, None).await {
            Err(BatchWriteError::Documents { failures }) => {
                assert_eq!(failures.len(), 1);
                assert!(is_refused(&failures[0].1), "batch: {:?}", failures[0].1);
            }
            other => panic!("expected a per-document batch failure, got {other:?}"),
        }

        assert_eq!(
            std::fs::read_to_string(sub.join(crate::schema::SCHEMA_FILE_NAME)).unwrap(),
            schema_text,
            "the schema file is untouched"
        );
    }

    /// A `.kb-schema.yaml` among a write's touched paths (here, as if pulled in by
    /// the write's own rebase) is not a document to index, but it must make the
    /// worker rebuild the shared schema cache — a full reconcile does.
    #[test]
    fn mark_dirty_turns_a_changed_schema_file_into_a_full_reconcile() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let queue = crate::reindex::ReindexQueue::new();

        mark_dirty(&queue, &config.indexing, vec![PathBuf::from("notes/a.md")]);
        assert!(!queue.snapshot().full_pending);

        mark_dirty(
            &queue,
            &config.indexing,
            vec![
                PathBuf::from("notes/b.md"),
                PathBuf::from("food/.kb-schema.yaml"),
            ],
        );
        assert!(queue.snapshot().full_pending);
        let pending = queue.snapshot_paths();
        assert!(pending.contains(&PathBuf::from("notes/b.md")));
        assert!(!pending.contains(&PathBuf::from("food/.kb-schema.yaml")));
    }

    #[tokio::test]
    async fn validation_failure_carries_the_structured_result() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = crate::mcp::make_test_resolved_config(tmp.path());
        Arc::get_mut(&mut config).unwrap().frontmatter.required = vec!["title".into()];
        let harness = Harness::new(&tmp, config);

        let req = make_req(
            "guide/missing-title.md",
            "---\ntype: guide\n---\n# No title",
            true,
        );
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::Validation { result } => {
                assert!(!result.valid);
                assert!(result.field_errors.iter().any(|e| e.field == "title"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn create_on_existing_file_reports_already_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("existing.md"), "# Already here").unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let req = make_req("docs/existing.md", "---\ntitle: T\n---\n# New", true);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(matches!(err, WriteError::AlreadyExists), "got {err:?}");
    }

    // -----------------------------------------------------------------------
    // check_include_pattern: the eligibility guard must fire for create, edit,
    // and delete alike, regardless of which caller reaches this pipeline.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn create_rejects_path_outside_include_patterns() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        // Harness's include globset is `**/*.md` only (see `Harness::new`).
        let harness = Harness::new(&tmp, config);

        let req = make_req("notes.txt", "Some plain text", true);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::UnsafePath { msg } => {
                assert!(msg.contains("indexable include pattern"), "got: {msg}");
            }
            other => panic!("expected UnsafePath, got {other:?}"),
        }
        assert!(
            !tmp.path().join("notes.txt").exists(),
            "nothing should be written when the include-pattern guard rejects the create"
        );
    }

    #[tokio::test]
    async fn edit_rejects_path_outside_include_patterns() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("notes.txt"), "old text").unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let req = make_req("notes.txt", "new text", false);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(matches!(err, WriteError::UnsafePath { .. }), "got {err:?}");
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("notes.txt")).unwrap(),
            "old text",
            "the existing file must be untouched when the include-pattern guard rejects the edit"
        );
    }

    #[tokio::test]
    async fn delete_rejects_path_outside_include_patterns() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("notes.txt"), "content").unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let err = delete_document(
            &harness.deps(),
            "notes.txt",
            None,
            harness.version_of("notes.txt"),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, WriteError::UnsafePath { .. }), "got {err:?}");
        assert!(
            tmp.path().join("notes.txt").exists(),
            "file must be untouched when the include-pattern guard rejects the delete"
        );
    }

    #[tokio::test]
    async fn edit_of_missing_file_reports_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let req = make_req("docs/nonexistent.md", "---\ntitle: T\n---\n# Body", false);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(matches!(err, WriteError::NotFound), "got {err:?}");
    }

    #[tokio::test]
    async fn delete_of_missing_file_reports_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let err = delete_document(
            &harness.deps(),
            "docs/nonexistent.md",
            None,
            harness.version_of("docs/nonexistent.md"),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, WriteError::NotFound), "got {err:?}");
    }

    #[tokio::test]
    async fn invalid_commit_message_is_rejected_before_any_filesystem_change() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("del-me.md"), "# Content").unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let err = delete_document(
            &harness.deps(),
            "docs/del-me.md",
            Some("bad\nmessage"),
            harness.version_of("docs/del-me.md"),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(err, WriteError::InvalidCommitMessage { .. }),
            "got {err:?}"
        );
        assert!(
            sub.join("del-me.md").exists(),
            "file must be untouched when the commit message is rejected up front"
        );
    }

    // -----------------------------------------------------------------------
    // Git-backed pre-commit / post-commit rollback tests (ported from mcp.rs)
    // -----------------------------------------------------------------------

    fn head_sha(work: &tempfile::TempDir) -> String {
        let out = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(work.path())
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn git_status(work: &tempfile::TempDir) -> String {
        let out = std::process::Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(work.path())
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn git_commit_all(work: &tempfile::TempDir, rel_path: &str, message: &str) {
        std::process::Command::new("git")
            .args(["add", "--", rel_path])
            .current_dir(work.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args([
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "commit",
                "-m",
                message,
            ])
            .current_dir(work.path())
            .output()
            .unwrap();
    }

    /// Like [`git_commit_all`], but stages and commits several paths in one
    /// commit — used by the `move_directory` tests below to seed a source
    /// subtree with more than one document without a separate commit per file.
    fn git_commit_paths(work: &tempfile::TempDir, rel_paths: &[&str], message: &str) {
        for rel_path in rel_paths {
            std::process::Command::new("git")
                .args(["add", "--", rel_path])
                .current_dir(work.path())
                .output()
                .unwrap();
        }
        std::process::Command::new("git")
            .args([
                "-c",
                "user.email=test@test.com",
                "-c",
                "user.name=Test",
                "commit",
                "-m",
                message,
            ])
            .current_dir(work.path())
            .output()
            .unwrap();
    }

    /// See `mcp.rs`'s identical helper for why this is repo-local git CONFIG
    /// rather than a `.git/hooks/pre-commit` script.
    fn force_git_commit_to_fail(work: &tempfile::TempDir) {
        for args in [
            ["config", "commit.gpgsign", "true"],
            ["config", "user.signingkey", "nonexistent-bogus-key-id"],
        ] {
            std::process::Command::new("git")
                .args(args)
                .current_dir(work.path())
                .output()
                .unwrap();
        }
    }

    fn git_backed_harness(work: &tempfile::TempDir) -> Harness {
        let mut config = crate::mcp::make_test_resolved_config(work.path());
        // Bypass the dedup gate: it would otherwise call out to a (nonexistent)
        // embedding service before we ever reach the commit.
        Arc::get_mut(&mut config).unwrap().write.dedup_enabled = false;
        Harness::new(work, config)
    }

    /// Push a new file directly to `bare_path`'s `branch` from a throwaway
    /// clone, simulating a concurrent commit landing in the remote after
    /// `work` (this test's own `Harness` clone) was already checked out —
    /// exactly the situation `git::commit_and_sync`'s fetch+rebase step
    /// exists for. Mirrors `webhook.rs`'s identical helper.
    fn push_file_from_a_fresh_clone(
        bare_path: &std::path::Path,
        branch: &str,
        rel_path: &str,
        contents: &str,
    ) {
        let clone = crate::git::tests::clone_bare_repo(bare_path, branch);
        let file_path = clone.path().join(rel_path);
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&file_path, contents).unwrap();
        git_commit_all(&clone, rel_path, &format!("add {rel_path}"));
        std::process::Command::new("git")
            .args(["push", "origin", branch])
            .current_dir(clone.path())
            .output()
            .unwrap();
    }

    /// #278 regression: `commit_outcome.rebased_paths` — paths pulled in from a
    /// CONCURRENT commit by this write's own fetch+rebase, not this write's own
    /// target — must be filtered through the same `indexing.include`/`exclude`/
    /// `exclude_files` predicate a full reconcile applies before reaching
    /// `queue.mark_paths`. Mirrors `git.rs`'s
    /// `commit_and_sync_reports_paths_pulled_in_by_the_rebase` for how the
    /// concurrent commit is simulated, but asserts on what actually reaches the
    /// dirty queue rather than on `commit_and_sync`'s own return value.
    #[tokio::test]
    async fn write_document_filters_non_indexable_rebased_paths_out_of_the_queue_mark() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");

        // Lands in the remote after this write's pre-write sync but before its
        // commit_and_sync, which must therefore fetch + rebase to pull it in.
        // `README.md` is in the default `indexing.exclude_files`.
        let bare_path = bare.path().to_path_buf();
        let pushed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let hook: crate::git::TestHook = Arc::new(move |point| {
            let bare_path = bare_path.clone();
            let pushed = Arc::clone(&pushed);
            Box::pin(async move {
                if point == crate::git::HookPoint::BeforeSync
                    && !pushed.swap(true, std::sync::atomic::Ordering::SeqCst)
                {
                    push_file_from_a_fresh_clone(&bare_path, "master", "README.md", "readme");
                }
            })
        });

        let mut config = crate::mcp::make_test_resolved_config(work.path());
        {
            let c = Arc::get_mut(&mut config).unwrap();
            c.write.dedup_enabled = false;
            c.source.git_url = Some(format!("file://{}", bare.path().to_str().unwrap()));
        }
        let harness = Harness::new(&work, config);
        let req = make_req(
            "docs/new.md",
            "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n",
            true,
        );
        let deps = harness.deps();
        let success = crate::git::TEST_HOOK
            .scope(hook, write_document(&deps, req))
            .await
            .unwrap();
        assert_eq!(success.outcome, WriteOutcome::Synced);

        // The rebase really did pull README.md in — `rebased_paths` still
        // reports it in full, unfiltered, for the caller's own reporting.
        assert_eq!(
            success.rebased_paths,
            vec![std::path::PathBuf::from("README.md")],
            "the rebase should have pulled in the concurrent README.md commit"
        );

        // ...but only this write's own target path reached the queue: README.md
        // is excluded and must not have been marked dirty, even though the
        // rebase pulled it in.
        crate::reindex::test_support::assert_marked_dirty(&harness.reindex_queue, &["docs/new.md"]);
        let pending = harness.reindex_queue.snapshot_paths();
        assert!(
            !pending.contains(&std::path::PathBuf::from("README.md")),
            "README.md is in the default exclude_files list and must not be marked \
             dirty, even though the rebase pulled it in"
        );
    }

    /// The sync at the start of a write pulls in remote commits the webhook will
    /// then find nothing new in, so the write itself marks those paths dirty —
    /// through the same include/exclude filter (#278).
    #[tokio::test]
    async fn write_document_marks_paths_its_pre_write_sync_pulled_in() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        push_file_from_a_fresh_clone(bare.path(), "master", "README.md", "readme");
        push_file_from_a_fresh_clone(
            bare.path(),
            "master",
            "docs/remote.md",
            "---\ntitle: R\n---\n# R\n",
        );

        let mut config = crate::mcp::make_test_resolved_config(work.path());
        {
            let c = Arc::get_mut(&mut config).unwrap();
            c.write.dedup_enabled = false;
            c.source.git_url = Some(format!("file://{}", bare.path().to_str().unwrap()));
        }
        let harness = Harness::new(&work, config);
        let req = make_req(
            "docs/new.md",
            "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n",
            true,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();
        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(work.path().join("docs/remote.md").exists());

        crate::reindex::test_support::assert_marked_dirty(
            &harness.reindex_queue,
            &["docs/new.md", "docs/remote.md"],
        );
        assert!(
            !harness
                .reindex_queue
                .snapshot_paths()
                .contains(&std::path::PathBuf::from("README.md"))
        );
    }

    /// #278 (own-target gap): a write's OWN target path — not just
    /// `commit_outcome.rebased_paths` — must also be filtered through
    /// `indexing.include`/`exclude`/`exclude_files` before it reaches
    /// `queue.mark_paths`. `check_include_pattern` only checks `include`, so a
    /// path that matches the default `**/*.md` include glob but is ALSO in the
    /// default `exclude_files` (here, `README.md`) still passes eligibility
    /// and commits successfully — but a full reconcile would never index it,
    /// so it must never be marked dirty either.
    #[tokio::test]
    async fn write_document_does_not_mark_its_own_excluded_target_path_dirty() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let harness = git_backed_harness(&work);

        // `create_bare_repo` already seeds a `README.md` at the repo root, so
        // this is an edit, not a create.
        let req = make_req("README.md", "# Test repo\n\nUpdated.\n", false);
        let success = write_document(&harness.deps(), req).await.unwrap();
        assert_eq!(success.outcome, WriteOutcome::Synced);

        let pending = harness.reindex_queue.snapshot_paths();
        assert!(
            pending.is_empty(),
            "README.md is in the default exclude_files list and must not be marked \
             dirty, even though it is its own write's target: {pending:?}"
        );
    }

    #[tokio::test]
    async fn create_precommit_failure_removes_the_new_file_and_rolls_back_cleanly() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let harness = git_backed_harness(&work);

        let req = make_req(
            "docs/new.md",
            "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n",
            true,
        );
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(rolled_back),
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }
        assert!(!work.path().join("docs/new.md").exists());
        assert_eq!(head_before, head_sha(&work));
        assert_eq!(git_status(&work), "");
    }

    #[tokio::test]
    async fn edit_precommit_failure_restores_previous_content() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original =
            "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
        std::fs::write(work.path().join("edit-me.md"), original).unwrap();
        git_commit_all(&work, "edit-me.md", "add edit-me.md");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let harness = git_backed_harness(&work);

        let req = make_req(
            "edit-me.md",
            "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New body\n",
            false,
        );
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(rolled_back),
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }
        assert_eq!(
            std::fs::read_to_string(work.path().join("edit-me.md")).unwrap(),
            original
        );
        assert_eq!(head_before, head_sha(&work));
        assert_eq!(git_status(&work), "");
    }

    // -----------------------------------------------------------------------
    // (#179) frontmatter_patch / append end-to-end through `write_document`.
    // These prove the design decision in this module's content-mode-helpers
    // section: `apply_frontmatter_patch`/`apply_append` only ever COMPUTE
    // `new_content` — by the time it reaches `write_document`, a patch/append
    // write is indistinguishable from an ordinary full-replace write of that
    // same content, so schema validation and the pre-commit rollback apply to
    // it with no special-casing.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn frontmatter_patch_synced_write_updates_only_frontmatter_end_to_end() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original = "---\ntitle: Log\nstatus: draft\n---\n\n# Body\nunchanged\n";
        std::fs::write(work.path().join("log.md"), original).unwrap();
        git_commit_all(&work, "log.md", "add log.md");

        let harness = git_backed_harness(&work);
        let new_content = apply_frontmatter_patch(
            original,
            &[FrontmatterEdit::SetField {
                field: "status".into(),
                value: serde_json::json!("active"),
            }],
        )
        .unwrap();

        let req = make_req("log.md", &new_content, false);
        let success = write_document(&harness.deps(), req).await.unwrap();
        assert_eq!(success.outcome, WriteOutcome::Synced);

        let on_disk = std::fs::read_to_string(work.path().join("log.md")).unwrap();
        let (fm, body) = validate::parse_frontmatter_raw(&on_disk);
        assert_eq!(fm.get("status").unwrap(), "active");
        assert_eq!(fm.get("title").unwrap(), "Log");
        assert_eq!(body.trim(), "# Body\nunchanged");

        crate::reindex::test_support::assert_marked_dirty(&harness.reindex_queue, &["log.md"]);
    }

    #[tokio::test]
    async fn frontmatter_patch_validation_failure_leaves_the_file_untouched() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original = "---\ntitle: Log\nstatus: draft\n---\n\n# Body\n";
        std::fs::write(work.path().join("log.md"), original).unwrap();
        git_commit_all(&work, "log.md", "add log.md");

        let mut config = crate::mcp::make_test_resolved_config(work.path());
        Arc::get_mut(&mut config).unwrap().frontmatter.required = vec!["title".into()];
        let harness = Harness::new(&work, config);

        // Patch removes the very field the schema requires — this must fail
        // schema validation exactly like any other write, not silently commit.
        let new_content = apply_frontmatter_patch(
            original,
            &[FrontmatterEdit::RemoveField {
                field: "title".into(),
            }],
        )
        .unwrap();

        let req = make_req("log.md", &new_content, false);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::Validation { result } => {
                assert!(result.field_errors.iter().any(|e| e.field == "title"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
        assert_eq!(
            std::fs::read_to_string(work.path().join("log.md")).unwrap(),
            original,
            "a rejected patch must never touch the file on disk"
        );
    }

    #[tokio::test]
    async fn frontmatter_patch_precommit_failure_rolls_back_to_the_original_content() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original = "---\ntitle: Log\nstatus: draft\n---\n\n# Body\n";
        std::fs::write(work.path().join("log.md"), original).unwrap();
        git_commit_all(&work, "log.md", "add log.md");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let harness = git_backed_harness(&work);

        let new_content = apply_frontmatter_patch(
            original,
            &[FrontmatterEdit::SetField {
                field: "status".into(),
                value: serde_json::json!("active"),
            }],
        )
        .unwrap();

        let req = make_req("log.md", &new_content, false);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(rolled_back),
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }
        assert_eq!(
            std::fs::read_to_string(work.path().join("log.md")).unwrap(),
            original,
            "a patch write's rollback must participate exactly like any other edit's"
        );
        assert_eq!(head_before, head_sha(&work));
        assert_eq!(git_status(&work), "");
    }

    #[tokio::test]
    async fn append_synced_write_adds_to_the_end_of_the_body_end_to_end() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original = "---\ntitle: Log\n---\n\n# Log\n- entry one\n";
        std::fs::write(work.path().join("log.md"), original).unwrap();
        git_commit_all(&work, "log.md", "add log.md");

        let harness = git_backed_harness(&work);
        let new_content = apply_append(original, "- entry two");

        let req = make_req("log.md", &new_content, false);
        let success = write_document(&harness.deps(), req).await.unwrap();
        assert_eq!(success.outcome, WriteOutcome::Synced);

        assert_eq!(
            std::fs::read_to_string(work.path().join("log.md")).unwrap(),
            "---\ntitle: Log\n---\n\n# Log\n- entry one\n- entry two\n"
        );
    }

    #[tokio::test]
    async fn append_precommit_failure_rolls_back_to_the_original_content() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original = "---\ntitle: Log\n---\n\n# Log\n- entry one\n";
        std::fs::write(work.path().join("log.md"), original).unwrap();
        git_commit_all(&work, "log.md", "add log.md");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let harness = git_backed_harness(&work);
        let new_content = apply_append(original, "- entry two");

        let req = make_req("log.md", &new_content, false);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(rolled_back),
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }
        assert_eq!(
            std::fs::read_to_string(work.path().join("log.md")).unwrap(),
            original,
            "an append write's rollback must participate exactly like any other edit's"
        );
        assert_eq!(head_before, head_sha(&work));
        assert_eq!(git_status(&work), "");
    }

    /// #147: the "rollback ITSELF also failed" branch of `write_document`'s
    /// CREATE path (`rolled_back: false`) — previously exercised only by
    /// `delete_document`'s equivalent test. Mirrors that test's technique:
    /// point the harness at a plain temp directory with no `.git` at all, so
    /// `git add` fails at `commit_and_sync`'s very first step (a `PreCommit`
    /// failure, same as any other pre-commit failure), and the create
    /// rollback's SECOND step — `git::unstage`, which runs after
    /// `tokio::fs::remove_file` already succeeded — fails too, because there
    /// is no repository to run `git reset` against.
    #[tokio::test]
    async fn create_rollback_failure_with_no_git_repo_reports_rolled_back_false() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let req = make_req(
            "docs/new-no-repo.md",
            "---\ntitle: New\n---\n\n# Body\n",
            true,
        );
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(!rolled_back),
            other => panic!("expected PreCommitFailed{{rolled_back: false}}, got {other:?}"),
        }
        // `remove_file` (the first of the two rollback steps) succeeded — the
        // filesystem write really is undone. Only `git::unstage` (the second
        // step) failed, which is exactly what makes this the "rollback
        // itself also failed" case rather than a clean rollback.
        assert!(!tmp.path().join("docs/new-no-repo.md").exists());
    }

    /// #147: the same "rollback itself also failed" branch, but for
    /// `write_document`'s EDIT path, which rolls back via a single call to
    /// `git::restore_from_head` instead of create's two-step remove+unstage —
    /// a genuinely different call to fail, so the create test above does not
    /// cover it.
    #[tokio::test]
    async fn edit_rollback_failure_with_no_git_repo_reports_rolled_back_false() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(
            sub.join("edit-me-no-repo.md"),
            "---\ntitle: Old\n---\n# Old",
        )
        .unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let req = make_req(
            "docs/edit-me-no-repo.md",
            "---\ntitle: New\n---\n# New",
            false,
        );
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(!rolled_back),
            other => panic!("expected PreCommitFailed{{rolled_back: false}}, got {other:?}"),
        }
        // The overwrite already landed on disk (an edit writes in place, with
        // no separate "create the new file" step to undo) and `git restore`
        // cannot put the old content back with no HEAD to restore from — the
        // file is left holding the new, uncommitted content, which IS the
        // inconsistency `rolled_back: false` is reporting.
        assert_eq!(
            std::fs::read_to_string(sub.join("edit-me-no-repo.md")).unwrap(),
            "---\ntitle: New\n---\n# New"
        );
    }

    /// #142, with versions: a full replace based on `original` must not clobber a
    /// concurrent change that landed after the caller read it. Here the two
    /// changes overlap, so the three-way merge conflicts and the write is refused,
    /// leaving the concurrent change untouched.
    #[tokio::test]
    async fn stale_full_replace_overlapping_a_concurrent_change_is_refused() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original =
            "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
        std::fs::write(work.path().join("edit-me.md"), original).unwrap();
        git_commit_all(&work, "edit-me.md", "add edit-me.md");

        let harness = git_backed_harness(&work);
        let expected_version = document_version(original.as_bytes());

        let concurrent = "---\ntitle: Concurrent\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Concurrent body\n";
        std::fs::write(work.path().join("edit-me.md"), concurrent).unwrap();
        git_commit_all(&work, "edit-me.md", "concurrent change");
        let head_before = head_sha(&work);

        let mut req = make_req("edit-me.md", "", false);
        req.change = DocChange::Replace(
            "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New body\n",
        );
        req.expected_version = Some(&expected_version);

        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(matches!(err, WriteError::EditedElsewhere), "{err:?}");
        assert_eq!(
            std::fs::read_to_string(work.path().join("edit-me.md")).unwrap(),
            concurrent
        );
        assert_eq!(head_before, head_sha(&work), "no commit must be made");
    }

    /// A stale full replace git cannot merge at all (content with a NUL byte,
    /// which `git merge-file` refuses as binary) is refused like a conflict, and
    /// none of git's output — which names the clone's own path — reaches the
    /// error.
    #[tokio::test]
    async fn a_stale_full_replace_git_cannot_merge_is_refused_without_git_output() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original =
            "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
        std::fs::write(work.path().join("edit-me.md"), original).unwrap();
        git_commit_all(&work, "edit-me.md", "add edit-me.md");
        let expected_version = document_version(original.as_bytes());
        let concurrent = original.replace("# Old body", "# Concurrent body");
        std::fs::write(work.path().join("edit-me.md"), &concurrent).unwrap();
        git_commit_all(&work, "edit-me.md", "concurrent change");
        let head_before = head_sha(&work);
        let harness = git_backed_harness(&work);

        let content = original.replace("title: Old", "title: New\u{0}");
        let mut req = make_req("edit-me.md", "", false);
        req.change = DocChange::Replace(&content);
        req.expected_version = Some(&expected_version);
        let err = write_document(&harness.deps(), req).await.unwrap_err();

        assert!(matches!(err, WriteError::EditedElsewhere), "{err:?}");
        let text = format!("{err:?}");
        for leaked in [
            ".git",
            "merge-file",
            harness.canonical_data_path.to_str().unwrap(),
        ] {
            assert!(!text.contains(leaked), "'{leaked}' leaked: {text}");
        }
        assert_eq!(
            std::fs::read_to_string(work.path().join("edit-me.md")).unwrap(),
            concurrent
        );
        assert_eq!(head_before, head_sha(&work), "no commit must be made");
    }

    /// A relative edit applies to the document as it is under the lock even
    /// when its `expected_version` is stale — and then reports that its result
    /// carries the changes made since, so the caller does not chain a full
    /// replace from its stale copy with the returned version.
    #[tokio::test]
    async fn a_relative_edit_over_a_stale_expected_version_is_reported_as_merged() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original = "---\ntitle: Log\n---\n\n# Log\n- one\n";
        std::fs::write(work.path().join("log.md"), original).unwrap();
        git_commit_all(&work, "log.md", "add log.md");
        let harness = git_backed_harness(&work);
        let append = |entry: &'static str| -> DocChange<'static> {
            let edit: &'static RelativeEdit<'static> = Box::leak(Box::new(move |current: &str| {
                Ok(apply_append(current, entry))
            }));
            DocChange::Relative(edit)
        };

        // The version of the document before someone else added "- one".
        let stale = document_version(b"---\ntitle: Log\n---\n\n# Log\n");
        let mut req = make_req("log.md", "", false);
        req.change = append("- two");
        req.expected_version = Some(&stale);
        let success = write_document(&harness.deps(), req).await.unwrap();
        assert!(
            success.merged,
            "applied over changes made since its version"
        );
        let on_disk = std::fs::read(work.path().join("log.md")).unwrap();
        assert_eq!(
            success.version.as_deref(),
            Some(document_version(&on_disk).as_str())
        );

        let mut req = make_req("log.md", "", false);
        req.change = append("- three");
        req.expected_version = harness.version_of("log.md");
        let success = write_document(&harness.deps(), req).await.unwrap();
        assert!(!success.merged, "a current version folds nothing else in");

        let mut req = make_req("log.md", "", false);
        req.change = append("- four");
        let success = write_document(&harness.deps(), req).await.unwrap();
        assert!(!success.merged, "no version, nothing to compare against");
        assert_eq!(
            std::fs::read_to_string(work.path().join("log.md")).unwrap(),
            "---\ntitle: Log\n---\n\n# Log\n- one\n- two\n- three\n- four\n"
        );
    }

    /// A relative edit may not grow a document past `MAX_CONTENT_LEN` — the cap
    /// content sent whole already has — but may still shrink one that is over it.
    #[tokio::test]
    async fn a_relative_edit_may_not_grow_a_document_past_the_size_cap() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let near_cap = format!(
            "---\ntitle: Big\n---\n\n{}\n",
            "x".repeat(MAX_CONTENT_LEN - 100)
        );
        std::fs::write(work.path().join("big.md"), &near_cap).unwrap();
        git_commit_all(&work, "big.md", "add big.md");
        let head_before = head_sha(&work);
        let harness = git_backed_harness(&work);

        let grown = apply_append(&near_cap, &"y".repeat(200));
        let err = write_document(&harness.deps(), make_req("big.md", &grown, false))
            .await
            .unwrap_err();
        match &err {
            WriteError::InvalidEdit { msg } => assert!(msg.contains("too large"), "{msg}"),
            other => panic!("expected InvalidEdit, got {other:?}"),
        }
        assert_eq!(
            std::fs::read_to_string(work.path().join("big.md")).unwrap(),
            near_cap
        );
        assert_eq!(head_sha(&work), head_before, "no commit must be made");

        // Grown past the cap some other way, it can still be edited down.
        let over_cap = format!("{near_cap}{}\n", "z".repeat(1000));
        std::fs::write(work.path().join("big.md"), &over_cap).unwrap();
        git_commit_all(&work, "big.md", "grow big.md outside the tools");
        let shrunk = over_cap.replacen(&"z".repeat(500), "", 1);
        assert!(shrunk.len() > MAX_CONTENT_LEN);
        let success = write_document(&harness.deps(), make_req("big.md", &shrunk, false))
            .await
            .unwrap();
        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert_eq!(
            std::fs::read_to_string(work.path().join("big.md")).unwrap(),
            shrunk
        );
    }

    /// An edit whose result is exactly the current document has nothing to
    /// commit: it succeeds without one, reporting the document as it stands —
    /// whether it was a stale full replace that merges to the current text, an
    /// identical full replace, or a relative edit that changes nothing.
    #[tokio::test]
    async fn an_edit_that_changes_nothing_succeeds_without_a_commit() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original =
            "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
        std::fs::write(work.path().join("edit-me.md"), original).unwrap();
        git_commit_all(&work, "edit-me.md", "add edit-me.md");
        let base = document_version(original.as_bytes());
        let concurrent = original.replace("# Old body", "# Concurrent body");
        std::fs::write(work.path().join("edit-me.md"), &concurrent).unwrap();
        git_commit_all(&work, "edit-me.md", "concurrent change");
        let current_version = document_version(concurrent.as_bytes());
        let head_before = head_sha(&work);
        let harness = git_backed_harness(&work);

        let assert_unchanged = |success: &WriteSuccess, merged: bool| {
            assert_eq!(success.outcome, WriteOutcome::Synced);
            assert_eq!(success.sha, head_before);
            assert!(success.diff.is_empty(), "{}", success.diff);
            assert!(success.rebased_paths.is_empty());
            assert_eq!(success.merged, merged);
            assert_eq!(success.version.as_deref(), Some(current_version.as_str()));
        };

        // A stale full replace making the very change someone else already made.
        let mut req = make_req("edit-me.md", "", false);
        req.change = DocChange::Replace(&concurrent);
        req.expected_version = Some(&base);
        let success = write_document(&harness.deps(), req).await.unwrap();
        assert_unchanged(&success, true);

        // An identical full replace of the current version.
        let mut req = make_req("edit-me.md", "", false);
        req.change = DocChange::Replace(&concurrent);
        req.expected_version = Some(&current_version);
        let success = write_document(&harness.deps(), req).await.unwrap();
        assert_unchanged(&success, false);

        // A relative edit that changes nothing.
        let success = write_document(&harness.deps(), make_req("edit-me.md", &concurrent, false))
            .await
            .unwrap();
        assert_unchanged(&success, false);

        assert_eq!(head_sha(&work), head_before, "no commit must be made");
        assert_eq!(git_status(&work), "");
        assert_eq!(
            std::fs::read_to_string(work.path().join("edit-me.md")).unwrap(),
            concurrent
        );
        assert!(harness.reindex_queue.snapshot_paths().is_empty());
    }

    /// A create carrying `expected_version` was meant for a document that is no
    /// longer there (deleted, moved, or never at this path): refused as not
    /// found — single and batched alike — never created anew.
    #[tokio::test]
    async fn a_create_with_expected_version_is_refused_as_not_found() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let head_before = head_sha(&work);
        let harness = git_backed_harness(&work);
        let content = "---\ntitle: Gone\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        let version = document_version(content.as_bytes());

        let mut req = make_req("docs/gone.md", content, true);
        req.expected_version = Some(&version);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(matches!(err, WriteError::NotFound), "{err:?}");
        assert!(!work.path().join("docs/gone.md").exists());

        let requests = [BatchWriteRequest {
            rel_path: "docs/gone.md",
            change: DocChange::Create(content),
            force_new: Some(true),
            expected_version: Some(&version),
        }];
        match write_documents_batch(&harness.deps(), &requests, None).await {
            Err(BatchWriteError::Documents { failures }) => {
                assert_eq!(failures.len(), 1);
                assert!(
                    matches!(failures[0].1, WriteError::NotFound),
                    "{:?}",
                    failures[0].1
                );
            }
            other => panic!("expected a per-document NotFound, got {other:?}"),
        }
        assert!(!work.path().join("docs/gone.md").exists());
        assert_eq!(head_sha(&work), head_before, "no commit must be made");
        assert_eq!(git_status(&work), "");

        // Without `expected_version`, the same create goes through.
        write_document(&harness.deps(), make_req("docs/gone.md", content, true))
            .await
            .unwrap();
        assert!(work.path().join("docs/gone.md").exists());
    }

    /// `domain` is derived from the folder: a schema `default:` for it is not the
    /// author writing it, and `required` on it is never asked of the author. Only
    /// a `domain` the author actually wrote is refused — also when other rules
    /// fail alongside it.
    #[tokio::test]
    async fn a_default_or_required_derived_field_does_not_block_writes() {
        let tmp = tempfile::tempdir().unwrap();
        let with_default = {
            let mut config = crate::mcp::make_test_resolved_config(tmp.path());
            Arc::get_mut(&mut config)
                .unwrap()
                .frontmatter
                .defaults
                .insert("domain".into(), "x".into());
            Harness::new(&tmp, config)
        };
        let with_required = {
            let mut config = crate::mcp::make_test_resolved_config(tmp.path());
            Arc::get_mut(&mut config).unwrap().frontmatter.required =
                vec!["title".into(), "domain".into()];
            Harness::new(&tmp, config)
        };

        for harness in [&with_default, &with_required] {
            let deps = harness.deps();
            validate_document(&deps, "docs/a.md", "---\ntitle: A\n---\n# A\n")
                .await
                .expect("no domain written: accepted");
        }

        let err = validate_document(
            &with_required.deps(),
            "docs/a.md",
            "---\ndomain: docs\n---\n# A\n",
        )
        .await
        .unwrap_err();
        match err {
            WriteError::Validation { result } => {
                let rules: Vec<(&str, &str)> = result
                    .field_errors
                    .iter()
                    .map(|e| (e.field.as_str(), e.rule.as_str()))
                    .collect();
                assert!(rules.contains(&("domain", "derived")), "{rules:?}");
                assert!(rules.contains(&("title", "required")), "{rules:?}");
                assert!(!rules.contains(&("domain", "required")), "{rules:?}");
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    /// A document write that fails part-way is undone before the error is
    /// returned: an overwrite gets its previous content back, a new file is
    /// removed — never a truncated or partial file left uncommitted in the clone.
    #[tokio::test]
    async fn a_failed_document_write_is_undone() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("doc.md");
        std::fs::write(&path, "previous\n").unwrap();

        // What an ENOSPC part-way through `fs::write` leaves: truncated, then
        // partly written.
        let err = write_or_undo(&path, Some("previous\n"), async {
            std::fs::write(&path, "new co")?;
            Err(std::io::Error::other("No space left on device"))
        })
        .await
        .unwrap_err();
        assert!(matches!(err, WriteError::Io { .. }), "{err:?}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "previous\n");

        let fresh = tmp.path().join("fresh.md");
        let err = write_or_undo(&fresh, None, async {
            std::fs::write(&fresh, "par")?;
            Err(std::io::Error::other("No space left on device"))
        })
        .await
        .unwrap_err();
        assert!(matches!(err, WriteError::Io { .. }), "{err:?}");
        assert!(!fresh.exists(), "a partly written new file is removed");

        // A write that succeeds is left as it is.
        write_or_undo(&path, Some("previous\n"), async {
            std::fs::write(&path, "new content\n")
        })
        .await
        .unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new content\n");
    }

    /// A new document's bytes are flushed before `git add` can see the file, and
    /// a write error that only the flush reports is not lost: `/dev/full` accepts
    /// the buffered write and fails it in the background (ENOSPC).
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn write_and_flush_reports_a_failure_only_the_flush_sees() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("doc.md");
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .await
            .unwrap();
        write_and_flush(&mut file, b"content\n").await.unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "content\n");

        let mut full = tokio::fs::OpenOptions::new()
            .write(true)
            .open("/dev/full")
            .await
            .unwrap();
        let err = write_and_flush(&mut full, b"content\n").await.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::StorageFull, "{err:?}");
    }

    /// A batch skips every edit that leaves its document exactly as it is: none
    /// of them is written, committed or marked dirty, and a batch of nothing
    /// but such edits succeeds with no commit at all.
    #[tokio::test]
    async fn write_documents_batch_skips_documents_it_leaves_unchanged() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let a = "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# A\n";
        let b = "---\ntitle: B\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# B\n";
        std::fs::write(work.path().join("a.md"), a).unwrap();
        std::fs::write(work.path().join("b.md"), b).unwrap();
        git_commit_paths(&work, &["a.md", "b.md"], "add a.md and b.md");
        let head_before = head_sha(&work);
        let harness = git_backed_harness(&work);

        let requests = vec![
            make_batch_req("a.md", "", a, false),
            make_batch_req("b.md", "", b, false),
        ];
        let success = write_documents_batch(&harness.deps(), &requests, None)
            .await
            .expect("a batch that changes nothing succeeds");
        assert_eq!(success.documents.len(), 2);
        for (doc, content) in success.documents.iter().zip([a, b]) {
            assert!(doc.diff.is_empty() && !doc.is_create && !doc.merged);
            assert_eq!(
                doc.version.as_deref(),
                Some(document_version(content.as_bytes()).as_str())
            );
        }
        assert_eq!(head_sha(&work), head_before, "no commit must be made");

        let a_new = a.replace("# A", "# A, edited");
        let requests = vec![
            make_batch_req("a.md", "", &a_new, false),
            make_batch_req("b.md", "", b, false),
        ];
        let success = write_documents_batch(&harness.deps(), &requests, None)
            .await
            .unwrap();
        assert!(success.documents[0].diff.contains("+# A, edited"));
        assert!(success.documents[1].diff.is_empty());
        let show = std::process::Command::new("git")
            .args(["show", "--name-only", "--format=", "HEAD"])
            .current_dir(work.path())
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&show.stdout).trim(), "a.md");
        let pending = harness.reindex_queue.snapshot_paths();
        assert!(pending.contains(&PathBuf::from("a.md")));
        assert!(!pending.contains(&PathBuf::from("b.md")), "{pending:?}");
    }

    /// A move is an absolute change to its source: a source that changed since
    /// the caller read it refuses the move before anything is written.
    #[tokio::test]
    async fn move_of_a_stale_source_version_is_refused() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let harness = git_backed_harness(&work);

        let concurrent =
            "---\ntitle: Concurrent\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Concurrent\n";
        std::fs::write(work.path().join("old/loc.md"), concurrent).unwrap();
        git_commit_all(&work, "old/loc.md", "concurrent change to old/loc.md");
        let head_before = head_sha(&work);

        // `make_move_req` passes the version of `original`, the stale read.
        let req = make_move_req("old/loc.md", "new/loc.md", original, original);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(matches!(err, WriteError::EditedElsewhere), "{err:?}");

        assert_eq!(
            std::fs::read_to_string(work.path().join("old/loc.md")).unwrap(),
            concurrent,
            "source must be left exactly as the concurrent writer left it"
        );
        assert!(!work.path().join("new/loc.md").exists());
        assert_eq!(head_before, head_sha(&work), "no commit must be made");
        assert_eq!(git_status(&work), "");
    }

    #[tokio::test]
    async fn create_synced_write_marks_the_path_dirty_and_returns_a_diff() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let harness = git_backed_harness(&work);

        // `harness.reindex_queue` is private to this `Harness` instance — no
        // other test's writes can land on it, so the path literal below needs
        // no cross-test uniqueness of its own (see `Harness::reindex_queue`'s
        // doc comment and the `same_path_literal_*` regression guard near the
        // bottom of this module).
        let req = make_req(
            "docs/queued-write-core-test.md",
            "---\ntitle: Queued\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n",
            true,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();
        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(!success.sha.is_empty());
        assert!(success.diff.contains("+title: Queued"));

        crate::reindex::test_support::assert_marked_dirty(
            &harness.reindex_queue,
            &["docs/queued-write-core-test.md"],
        );
    }

    // Regression guard for the original bug: before the queue became an
    // injected dependency, both tests below would have collided on
    // `crate::reindex::REINDEX_QUEUE` — a single process-wide `HashSet<PathBuf>`
    // shared by the whole test binary. Marking an already-pending path is a
    // no-op on a `HashSet`'s cardinality, so whichever of these two ran second
    // would silently fail to observe its own write having marked its path
    // dirty. That failure showed up only under `cargo test --
    // --test-threads=1`, where libtest's alphabetical run order made the
    // collision deterministic (a plain `cargo test` run could get lucky and
    // interleave them apart). Each test here now builds its own `Harness` —
    // and therefore its own private `ReindexQueue` (see
    // `Harness::reindex_queue`'s doc comment) — so using the IDENTICAL path
    // literal in both is not just safe, it is the point: this is the case that
    // used to fail and now provably does not.

    #[tokio::test]
    async fn same_path_literal_in_two_independent_tests_a() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let harness = git_backed_harness(&work);

        let req = make_req(
            "docs/same-literal-regression-guard.md",
            "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n",
            true,
        );
        write_document(&harness.deps(), req).await.unwrap();

        crate::reindex::test_support::assert_marked_dirty(
            &harness.reindex_queue,
            &["docs/same-literal-regression-guard.md"],
        );
    }

    #[tokio::test]
    async fn same_path_literal_in_two_independent_tests_b() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let harness = git_backed_harness(&work);

        let req = make_req(
            "docs/same-literal-regression-guard.md",
            "---\ntitle: B\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n",
            true,
        );
        write_document(&harness.deps(), req).await.unwrap();

        crate::reindex::test_support::assert_marked_dirty(
            &harness.reindex_queue,
            &["docs/same-literal-regression-guard.md"],
        );
    }

    #[tokio::test]
    async fn delete_precommit_failure_restores_the_file() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original = "---\ntitle: D\n---\n\n# Body\n";
        std::fs::write(work.path().join("doomed.md"), original).unwrap();
        git_commit_all(&work, "doomed.md", "add doomed.md");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let harness = git_backed_harness(&work);

        let err = delete_document(
            &harness.deps(),
            "doomed.md",
            None,
            harness.version_of("doomed.md"),
        )
        .await
        .unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(rolled_back),
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }
        assert!(work.path().join("doomed.md").exists());
        assert_eq!(
            std::fs::read_to_string(work.path().join("doomed.md")).unwrap(),
            original
        );
        assert_eq!(head_before, head_sha(&work));
        assert_eq!(git_status(&work), "");
    }

    #[tokio::test]
    async fn delete_with_no_git_repo_reports_unrecoverable_precommit_failure() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("docs");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(
            sub.join("delete-me.md"),
            "---\ntitle: Delete Me\n---\n# Body",
        )
        .unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let err = delete_document(
            &harness.deps(),
            "docs/delete-me.md",
            None,
            harness.version_of("docs/delete-me.md"),
        )
        .await
        .unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(!rolled_back),
            other => panic!("expected PreCommitFailed{{rolled_back: false}}, got {other:?}"),
        }
        // The restore could not put it back (there is no repo to restore from), so
        // the file really is gone — that IS the inconsistent state being reported.
        assert!(!sub.join("delete-me.md").exists());
    }

    #[tokio::test]
    async fn delete_postcommit_failure_leaves_the_commit_and_reports_pending_sync() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::write(
            work.path().join("doomed.md"),
            "---\ntitle: D\n---\n\n# Body\n",
        )
        .unwrap();
        git_commit_all(&work, "doomed.md", "add doomed.md");

        let mut config = crate::mcp::make_test_resolved_config(work.path());
        {
            let c = Arc::get_mut(&mut config).unwrap();
            c.write.dedup_enabled = false;
            c.source.git_url = Some("/nonexistent/path/to/repo.git".to_string());
        }
        let harness = Harness::new(&work, config);

        let (captured, guard) = capture_warnings();
        let success = delete_document(
            &harness.deps(),
            "doomed.md",
            None,
            harness.version_of("doomed.md"),
        )
        .await
        .unwrap();
        drop(guard);
        assert_eq!(success.outcome, WriteOutcome::CommittedPendingSync);
        assert!(!work.path().join("doomed.md").exists());
        // The sync failure's cause is not on the result: it is logged next to the
        // commit that stands.
        let log = captured.text();
        let marker = format!("deletion commit {} stands uncorrected: ", head_sha(&work));
        assert!(
            log.split_once(&marker)
                .is_some_and(|(_, cause)| !cause.trim().is_empty()),
            "expected the sync failure and its cause in the log, got: {log:?}"
        );

        // #150: this is the ONLY trigger for the reindex worker to purge the
        // deleted document's Qdrant points and state rows (see
        // `ingest::index_paths`'s missing-file branch) — a regression that
        // dropped this call would leave the document searchable forever,
        // with every other assertion above still green.
        crate::reindex::test_support::assert_marked_dirty(&harness.reindex_queue, &["doomed.md"]);
    }

    /// #150: `delete_document`'s OTHER success path — commit AND push both
    /// succeed — was, per the issue, never exercised by any test at all
    /// (only the `CommittedPendingSync` branch above was reached, and even
    /// that omitted this assertion). Mirrors `create_synced_write_marks_the_path_dirty_and_returns_a_diff`'s
    /// pattern for the create path.
    #[tokio::test]
    async fn delete_synced_write_marks_the_path_dirty() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::write(
            work.path().join("doomed-synced.md"),
            "---\ntitle: D\n---\n\n# Body\n",
        )
        .unwrap();
        git_commit_all(&work, "doomed-synced.md", "add doomed-synced.md");

        let harness = git_backed_harness(&work);

        let success = delete_document(
            &harness.deps(),
            "doomed-synced.md",
            None,
            harness.version_of("doomed-synced.md"),
        )
        .await
        .unwrap();
        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(!work.path().join("doomed-synced.md").exists());

        crate::reindex::test_support::assert_marked_dirty(
            &harness.reindex_queue,
            &["doomed-synced.md"],
        );
    }

    // -----------------------------------------------------------------------
    // #181 / #229 — delete_document warns about inbound links instead of
    // silently orphaning them, AND (#229) surfaces the same referencing paths
    // on `WriteSuccess::referencing_paths` so a caller with no access to
    // server logs can see them too. `CapturedLogs` below is a minimal
    // `tracing_subscriber::fmt::MakeWriter` that redirects exactly the calls
    // made while its guard is alive into an in-memory buffer a test can
    // assert on — `tracing::subscriber::set_default`'s guard scopes it to
    // this one call, so it cannot leak into (or be polluted by) any other
    // test's logging.
    // -----------------------------------------------------------------------

    #[derive(Clone, Default)]
    struct CapturedLogs(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for CapturedLogs {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLogs {
        type Writer = Self;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl CapturedLogs {
        fn text(&self) -> String {
            String::from_utf8_lossy(&self.0.lock().unwrap()).into_owned()
        }
    }

    /// Scoped to `WARN` and above so that unrelated `info!`/`debug!` output
    /// elsewhere in the write pipeline (or in `git.rs`) can never show up in
    /// `CapturedLogs::text()` and be mistaken for the inbound-link warning
    /// this module's tests care about.
    fn capture_warnings() -> (CapturedLogs, tracing::subscriber::DefaultGuard) {
        let captured = CapturedLogs::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(captured.clone())
            .with_max_level(tracing::Level::WARN)
            .without_time()
            .with_target(false)
            .finish();
        let guard = tracing::subscriber::set_default(subscriber);
        (captured, guard)
    }

    #[tokio::test]
    async fn delete_warns_about_inbound_links_but_does_not_refuse() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::write(
            work.path().join("linked.md"),
            "---\ntitle: Linked\n---\n\n# Body\n",
        )
        .unwrap();
        git_commit_all(&work, "linked.md", "add linked.md");

        let harness = git_backed_harness_with_state_db(&work).await;
        // `referencer.md` need not exist on disk — the check queries the
        // reverse-link INDEX, not the filesystem, same as
        // `write_document_move`'s step 10.5.
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencer.md",
                "markdown",
                &[("linked.md".to_string(), None)],
            )
            .await
            .unwrap();

        let (captured, guard) = capture_warnings();
        let success = delete_document(
            &harness.deps(),
            "linked.md",
            None,
            harness.version_of("linked.md"),
        )
        .await
        .expect("an inbound link must WARN, not refuse the delete — see #181's PR notes");
        drop(guard);

        assert_eq!(
            success.outcome,
            WriteOutcome::Synced,
            "the delete itself must still succeed"
        );
        let log_text = captured.text();
        assert!(
            log_text.contains("referencer.md"),
            "expected a warning naming the referencing document, got log: {log_text:?}"
        );
        assert!(
            log_text.contains("linked.md"),
            "expected the warning to name the document being deleted too, got log: {log_text:?}"
        );
        assert_eq!(
            success.referencing_paths,
            vec!["referencer.md".to_string()],
            "#229: the same referencing path must also reach the caller via \
             WriteSuccess, not just the server log"
        );
    }

    #[tokio::test]
    async fn delete_with_no_inbound_links_logs_no_warning() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::write(
            work.path().join("unlinked.md"),
            "---\ntitle: Unlinked\n---\n\n# Body\n",
        )
        .unwrap();
        git_commit_all(&work, "unlinked.md", "add unlinked.md");

        // With a state DB wired up but no `document_links` row targeting this
        // document, the query must come back empty and stay silent.
        let harness = git_backed_harness_with_state_db(&work).await;

        let (captured, guard) = capture_warnings();
        let success = delete_document(
            &harness.deps(),
            "unlinked.md",
            None,
            harness.version_of("unlinked.md"),
        )
        .await
        .unwrap();
        drop(guard);

        assert!(
            captured.text().is_empty(),
            "no inbound links means no warning, got log: {:?}",
            captured.text()
        );
        assert!(
            success.referencing_paths.is_empty(),
            "#229: no inbound links means an empty referencing_paths too"
        );
    }

    #[tokio::test]
    async fn delete_with_no_state_db_skips_the_inbound_link_check_silently() {
        // `WriteDeps::state == None` (no `with_state_db`) must not be treated
        // as "querying failed" — it is a normal, documented degraded mode
        // (see that field's doc comment), so the delete must proceed exactly
        // as it always has, with no warning and no error.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::write(
            work.path().join("no-state-db.md"),
            "---\ntitle: No State DB\n---\n\n# Body\n",
        )
        .unwrap();
        git_commit_all(&work, "no-state-db.md", "add no-state-db.md");

        let harness = git_backed_harness(&work);

        let (captured, guard) = capture_warnings();
        let success = delete_document(
            &harness.deps(),
            "no-state-db.md",
            None,
            harness.version_of("no-state-db.md"),
        )
        .await
        .unwrap();
        drop(guard);

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(captured.text().is_empty());
        assert!(
            success.referencing_paths.is_empty(),
            "#229: with no state DB wired up, referencing_paths must stay empty, \
             same as the warning it mirrors"
        );
    }

    // -----------------------------------------------------------------------
    // G1 — the traversal check must run before validation (and, in particular,
    // before an exec of a configured `validation.lint_command` against the raw
    // caller-supplied path). See `write_document`'s "0.5" step comment.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn create_traversal_is_rejected_before_the_lint_command_ever_runs() {
        let tmp = tempfile::tempdir().unwrap();
        let marker = tmp.path().join("lint-invoked-marker");
        let mut config = crate::mcp::make_test_resolved_config(tmp.path());
        // A fake lint command that leaves undeniable evidence it ran. If the
        // traversal check did not run until after `validate::validate_content`
        // (the pre-fix ordering), this file would exist afterwards.
        Arc::get_mut(&mut config).unwrap().validation.lint_command = Some(vec![
            "sh".into(),
            "-c".into(),
            format!("touch '{}'", marker.display()),
        ]);
        let harness = Harness::new(&tmp, config);

        // Both spellings from the plan: `GlobSet::is_match` (the include-pattern
        // check that runs just before this one) accepts `..` segments as plain
        // characters, so `**/*.md` matches both of these — the traversal check
        // is the only thing standing between them and the lint command.
        for traversal_path in ["../escape.md", "a/../../escape.md"] {
            let req = make_req(traversal_path, "---\ntitle: T\n---\n# Body", true);
            let err = write_document(&harness.deps(), req).await.unwrap_err();
            assert!(
                matches!(err, WriteError::UnsafePath { .. }),
                "expected UnsafePath (rejected before validation) for '{traversal_path}', \
                 got {err:?}"
            );
            assert!(
                !marker.exists(),
                "lint command must not run for a traversal path '{traversal_path}' \
                 rejected before validation"
            );
        }
    }

    #[tokio::test]
    async fn delete_traversal_is_rejected_before_commit_message_validation() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        // An invalid commit message (a newline) would normally be reported as
        // `InvalidCommitMessage` — but a traversal path must be rejected ahead
        // of that check, mirroring `write_document`'s ordering.
        let err = delete_document(
            &harness.deps(),
            "../escape.md",
            Some("bad\nmessage"),
            harness.version_of("../escape.md"),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(err, WriteError::UnsafePath { .. }),
            "expected UnsafePath ahead of commit-message validation, got {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // G2 — a canonicalize failure inside the path-safety check must surface as
    // `WriteError::Internal`, never `WriteError::UnsafePath`, so a caller-safe
    // transport (`web.rs`) can tell the two apart. See `WriteError::Internal`'s
    // doc comment.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn missing_data_root_reports_internal_not_unsafe_path() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        // Remove the data root out from under the already-built deps: this
        // forces `resolve_safe_write_path`'s `data_root.canonicalize()` call to
        // fail. The resulting message is allowed to embed this absolute path
        // (this is `write.rs`'s own classification test, not a transport-facing
        // one) — the point is that it must be `Internal`, not `UnsafePath`.
        std::fs::remove_dir_all(tmp.path()).unwrap();

        let req = make_req("docs/new.md", "---\ntitle: T\n---\n# Body", true);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::Internal { msg } => {
                assert!(msg.contains("cannot canonicalize"), "got: {msg}");
            }
            other => panic!("expected Internal, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // WriteRequest::dest_path (document MOVE) — write_document_move
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn successful_move_relocates_the_file_in_one_commit_and_marks_both_paths_dirty() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc-move-core-test-src.md"), original).unwrap();
        git_commit_all(
            &work,
            "old/loc-move-core-test-src.md",
            "add old/loc-move-core-test-src.md",
        );
        let head_before = head_sha(&work);

        let harness = git_backed_harness(&work);

        let req = make_move_req(
            "old/loc-move-core-test-src.md",
            "new/loc-moved-write-core-test.md",
            original,
            original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(!success.sha.is_empty());
        assert_ne!(
            success.sha, head_before,
            "the move must produce a new commit"
        );
        assert!(
            !work.path().join("old/loc-move-core-test-src.md").exists(),
            "source must be gone after a successful move"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("new/loc-moved-write-core-test.md")).unwrap(),
            original
        );
        // The removal and the addition both landed in the single commit — nothing
        // left staged or dangling in the working tree afterward.
        assert_eq!(git_status(&work), "");

        crate::reindex::test_support::assert_marked_dirty(
            &harness.reindex_queue,
            &[
                "old/loc-move-core-test-src.md",
                "new/loc-moved-write-core-test.md",
            ],
        );
    }

    #[tokio::test]
    async fn move_with_a_stale_expected_version_is_rejected_and_mutates_nothing() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");
        let head_before = head_sha(&work);

        let harness = git_backed_harness(&work);

        let stale = document_version(b"not the current content");
        let mut req = make_move_req("old/loc.md", "new/loc.md", original, original);
        req.expected_version = Some(&stale);

        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(matches!(err, WriteError::EditedElsewhere), "{err:?}");
        assert!(
            work.path().join("old/loc.md").exists(),
            "source must be untouched when the expected_version is stale"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("old/loc.md")).unwrap(),
            original
        );
        assert!(
            !work.path().join("new/loc.md").exists(),
            "destination must never be created when the expected_version is stale"
        );
        assert_eq!(head_before, head_sha(&work), "no commit must be made");
        assert_eq!(git_status(&work), "");
    }

    #[tokio::test]
    async fn move_with_a_matching_expected_version_proceeds_normally() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let harness = git_backed_harness(&work);

        // `make_move_req` passes `original`'s version, the current one.
        let req = make_move_req(
            "old/loc.md",
            "new/loc-matching-hash-test.md",
            original,
            original,
        );

        let success = write_document(&harness.deps(), req).await.unwrap();
        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(!work.path().join("old/loc.md").exists());
        assert_eq!(
            std::fs::read_to_string(work.path().join("new/loc-matching-hash-test.md")).unwrap(),
            original
        );
    }

    #[tokio::test]
    async fn move_to_an_existing_destination_reports_already_exists_and_mutates_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let source_dir = tmp.path().join("docs");
        std::fs::create_dir_all(&source_dir).unwrap();
        let original = "---\ntitle: T\n---\n# Body";
        std::fs::write(source_dir.join("source.md"), original).unwrap();
        let dest_dir = tmp.path().join("other");
        std::fs::create_dir_all(&dest_dir).unwrap();
        std::fs::write(dest_dir.join("dest.md"), "# Already here").unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let req = make_move_req("docs/source.md", "other/dest.md", original, original);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(matches!(err, WriteError::AlreadyExists), "got {err:?}");
        assert_eq!(
            std::fs::read_to_string(source_dir.join("source.md")).unwrap(),
            original,
            "source must be untouched when the destination already exists"
        );
        assert_eq!(
            std::fs::read_to_string(dest_dir.join("dest.md")).unwrap(),
            "# Already here",
            "the pre-existing destination file must be untouched"
        );
    }

    #[tokio::test]
    async fn move_validates_against_the_destination_schema_not_the_sources() {
        let tmp = tempfile::tempdir().unwrap();
        let source_dir = tmp.path().join("loose");
        std::fs::create_dir_all(&source_dir).unwrap();
        // Valid under the (schema-less) source directory, but missing a field the
        // destination directory's schema requires.
        let content = "---\ntitle: T\n---\n# Body";
        std::fs::write(source_dir.join("source.md"), content).unwrap();

        let dest_dir = tmp.path().join("strict");
        std::fs::create_dir_all(&dest_dir).unwrap();
        std::fs::write(
            dest_dir.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  strict_field:\n    required: true\n",
        )
        .unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let req = make_move_req("loose/source.md", "strict/dest.md", content, content);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::Validation { result } => {
                assert!(!result.valid);
                assert!(
                    result
                        .field_errors
                        .iter()
                        .any(|e| e.field == "strict_field"),
                    "expected a strict_field error, got {:?}",
                    result.field_errors
                );
            }
            other => panic!("expected Validation, got {other:?}"),
        }
        assert!(
            source_dir.join("source.md").exists(),
            "source must be untouched when destination validation fails"
        );
        assert!(
            !dest_dir.join("dest.md").exists(),
            "nothing should be written to the destination when validation fails"
        );
    }

    #[tokio::test]
    async fn move_precommit_failure_rolls_back_both_halves() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let harness = git_backed_harness(&work);

        let req = make_move_req("old/loc.md", "new/loc.md", original, original);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(rolled_back),
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }
        assert!(
            work.path().join("old/loc.md").exists(),
            "source must be restored after a rolled-back move"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("old/loc.md")).unwrap(),
            original
        );
        assert!(
            !work.path().join("new/loc.md").exists(),
            "destination must be gone after a rolled-back move"
        );
        assert_eq!(head_before, head_sha(&work));
        assert_eq!(git_status(&work), "");
    }

    /// #147: `write_document_move`'s own "rollback itself also failed" branch
    /// (`rolled_back: false`) — same no-git-repo technique as the create/edit
    /// tests above and `delete_document`'s existing test. `rolled_back` here
    /// is `source_restore.is_ok() && dest_rollback.is_ok() &&
    /// rewrite_restore_failures.is_empty()`: with no `.git` at all, `git add`
    /// fails first (`PreCommit`), and then BOTH `source_restore`
    /// (`git::restore_from_head`) and the git half of `dest_rollback`
    /// (`git::unstage`, reached after its own `tokio::fs::remove_file` step
    /// already succeeded) fail too, since neither has a repository to run
    /// against — so `rolled_back` ends up `false` on two independent counts
    /// at once, not just one.
    #[tokio::test]
    async fn move_rollback_failure_with_no_git_repo_reports_rolled_back_false() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("old");
        std::fs::create_dir_all(&sub).unwrap();
        let source_original = "---\ntitle: Move Me\n---\n\n# Body\n";
        std::fs::write(sub.join("loc.md"), source_original).unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let req = make_move_req("old/loc.md", "new/loc.md", source_original, source_original);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(!rolled_back),
            other => panic!("expected PreCommitFailed{{rolled_back: false}}, got {other:?}"),
        }
        // The destination copy was written, then successfully removed during
        // rollback (`remove_file` needs no git repo) — but `restore_from_head`
        // on the source can't run with no HEAD to restore from, so the
        // source is left gone too. Both paths now missing IS the
        // inconsistency `rolled_back: false` reports.
        assert!(!tmp.path().join("new/loc.md").exists());
        assert!(!sub.join("loc.md").exists());
    }

    #[tokio::test]
    async fn move_with_a_content_change_writes_the_new_content_to_the_destination() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let original =
            "---\ntitle: Old Title\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
        std::fs::write(work.path().join("old/loc.md"), original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let harness = git_backed_harness(&work);

        let new_content =
            "---\ntitle: New Title\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New body\n";
        let req = make_move_req("old/loc.md", "new/loc.md", original, new_content);
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(!work.path().join("old/loc.md").exists());
        assert_eq!(
            std::fs::read_to_string(work.path().join("new/loc.md")).unwrap(),
            new_content,
            "the destination must hold the NEW content, not a copy of the source's old content"
        );
        assert!(success.diff.contains("+title: New Title"));
        assert!(success.diff.contains("-title: Old Title"));
    }

    // -----------------------------------------------------------------------
    // GIT_LOCK hoist regression (data-loss finding fix): `write_document_move`
    // must acquire `GIT_LOCK` before its destination write / source removal /
    // referencing-document rewrite, not just before the commit, and must hold
    // that ONE guard across all of it. Proven at runtime rather than merely
    // structurally: another holder of `GIT_LOCK` must observably block the
    // call, and releasing that holder must let it proceed to completion
    // without hanging (a hang would mean something reachable from this call
    // tried to reacquire the already-held, non-reentrant mutex).
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn write_document_move_blocks_while_git_lock_is_externally_held_then_completes() {
        let tmp = tempfile::tempdir().unwrap();
        let original = "---\ntitle: A\n---\n# A";
        std::fs::write(tmp.path().join("source-lockcheck.md"), original).unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        // Hold GIT_LOCK exactly as a concurrent writer, the webhook handler, or
        // the reindex worker would.
        let held = git::lock_git().await;

        let req = make_move_req(
            "source-lockcheck.md",
            "dest-lockcheck.md",
            original,
            original,
        );
        let deps = harness.deps();
        let move_fut = write_document(&deps, req);
        tokio::pin!(move_fut);
        let still_blocked =
            tokio::time::timeout(std::time::Duration::from_millis(200), &mut move_fut).await;
        assert!(
            still_blocked.is_err(),
            "write_document_move must block on GIT_LOCK (acquired ahead of the destination \
             write, per the finding's fix) while another holder has it"
        );

        drop(held);
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), move_fut)
            .await
            .expect(
                "write_document_move must proceed to completion once GIT_LOCK is released, not \
                 hang against its own held guard -- a hang here would mean this non-reentrant \
                 mutex is being acquired a second time somewhere in the call chain",
            );
        // Not git-backed, so the commit itself fails fast once the lock is free --
        // the point of this test is that nothing deadlocks, not the outcome.
        match result.unwrap_err() {
            WriteError::PreCommitFailed { .. } => {}
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Move + incoming-link rewrite (WriteDeps::state, `document_links` reverse
    // lookup, `ingest::find_markdown_link_occurrences`/`relativize_md_path`)
    // -----------------------------------------------------------------------

    /// A git-backed harness whose `WriteDeps::state` is wired up to a real,
    /// temp-file-backed `StateDb` (see `Harness::with_state_db`'s doc comment
    /// for why it lives outside the git working copy), so `write_document_move`
    /// actually performs link rewriting.
    async fn git_backed_harness_with_state_db(work: &tempfile::TempDir) -> Harness {
        git_backed_harness(work).with_state_db().await
    }

    #[tokio::test]
    async fn move_rewrites_a_referencing_documents_link_in_the_same_commit() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let referencing_original = "---\ntitle: Referencer\ndescription: d\ntype: guide\n\
             tags: [t]\n---\n\nSee [the moved doc](old/loc.md) for more.\n";
        std::fs::write(work.path().join("referencing.md"), referencing_original).unwrap();
        git_commit_all(&work, "referencing.md", "add referencing.md");
        let head_before = head_sha(&work);

        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencing.md",
                "markdown",
                &[("old/loc.md".to_string(), None)],
            )
            .await
            .unwrap();

        let req = make_move_req(
            "old/loc.md",
            "new/loc-rewrite-test1.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert_eq!(success.rewritten_paths, vec!["referencing.md".to_string()]);

        let referencing_after =
            std::fs::read_to_string(work.path().join("referencing.md")).unwrap();
        assert!(
            referencing_after.contains("[the moved doc](new/loc-rewrite-test1.md)"),
            "referencing document's link must point at the new location, got: {referencing_after}"
        );
        assert!(!referencing_after.contains("old/loc.md"));

        // Both halves of the move AND the referencing-document rewrite landed in
        // exactly ONE commit: the working tree is clean and HEAD moved exactly
        // once (`success.sha` is the only new commit, matching this test's own
        // assertions on the file contents above having already landed).
        assert_eq!(git_status(&work), "");
        assert_ne!(head_before, head_sha(&work));
        assert_eq!(success.sha, head_sha(&work));
    }

    /// A rebase that merged someone else's change into a rewritten referencing
    /// document — not into the moved one — reports the move as merged, but keeps
    /// the moved document's version: its bytes are exactly what was written.
    #[tokio::test]
    async fn a_move_whose_rebase_merged_only_a_referencing_document_keeps_its_version() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source = "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source).unwrap();
        let referencing = "---\ntitle: Referencer\ndescription: d\ntype: guide\ntags: [t]\n---\n\n\
             See [the moved doc](old/loc.md) for more.\n\none\ntwo\nthree\nlast line\n";
        std::fs::write(work.path().join("referencing.md"), referencing).unwrap();
        git_commit_paths(&work, &["old/loc.md", "referencing.md"], "seed");
        std::process::Command::new("git")
            .args(["push", "origin", "master"])
            .current_dir(work.path())
            .output()
            .unwrap();

        // Lands after the move's own commit and before its fetch + rebase: a
        // change to the referencing document well away from the link the move
        // rewrites, so the rebase merges it cleanly.
        let bare_path = bare.path().to_path_buf();
        let peer = referencing.replace("last line", "last line, edited by a peer");
        let pushed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let hook: crate::git::TestHook = Arc::new(move |point| {
            let bare_path = bare_path.clone();
            let peer = peer.clone();
            let pushed = Arc::clone(&pushed);
            Box::pin(async move {
                if point == crate::git::HookPoint::BeforeSync
                    && !pushed.swap(true, std::sync::atomic::Ordering::SeqCst)
                {
                    push_file_from_a_fresh_clone(&bare_path, "master", "referencing.md", &peer);
                }
            })
        });

        let mut config = crate::mcp::make_test_resolved_config(work.path());
        {
            let c = Arc::get_mut(&mut config).unwrap();
            c.write.dedup_enabled = false;
            c.source.git_url = Some(format!("file://{}", bare.path().to_str().unwrap()));
        }
        let harness = Harness::new(&work, config).with_state_db().await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencing.md",
                "markdown",
                &[("old/loc.md".to_string(), None)],
            )
            .await
            .unwrap();

        let deps = harness.deps();
        let req = make_move_req("old/loc.md", "new/loc.md", source, source);
        let success = crate::git::TEST_HOOK
            .scope(hook, write_document(&deps, req))
            .await
            .unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert_eq!(success.rewritten_paths, vec!["referencing.md".to_string()]);
        assert_eq!(
            success.rebased_paths,
            vec![PathBuf::from("referencing.md")],
            "the rebase merged the peer's change into the referencing document"
        );
        assert!(success.merged, "a document this move wrote was merged");
        let moved = std::fs::read(work.path().join("new/loc.md")).unwrap();
        assert_eq!(
            success.version.as_deref(),
            Some(document_version(&moved).as_str()),
            "the moved document is exactly what was written"
        );
        let referencing_after =
            std::fs::read_to_string(work.path().join("referencing.md")).unwrap();
        assert!(
            referencing_after.contains("(new/loc.md)")
                && referencing_after.contains("edited by a peer"),
            "{referencing_after}"
        );
    }

    #[tokio::test]
    async fn move_rewrite_relativizes_correctly_for_a_referencing_document_elsewhere() {
        // The case a naive string substitution gets wrong: the referencing
        // document lives in a THIRD directory, unrelated to both the source's
        // and the destination's, so the correct replacement text is neither the
        // raw destination path nor a copy of the old relative text — it must be
        // freshly computed from the referencing document's own location,
        // climbing up ("../../") before descending back down.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("docs/sub")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("docs/sub/loc.md"), source_original).unwrap();
        git_commit_all(&work, "docs/sub/loc.md", "add docs/sub/loc.md");

        std::fs::create_dir_all(work.path().join("other/deep")).unwrap();
        let referencing_original = "---\ntitle: Referencer\ndescription: d\ntype: guide\n\
             tags: [t]\n---\n\nSee [it](../../docs/sub/loc.md) for more.\n";
        std::fs::write(work.path().join("other/deep/ref.md"), referencing_original).unwrap();
        git_commit_all(&work, "other/deep/ref.md", "add other/deep/ref.md");

        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "other/deep/ref.md",
                "markdown",
                &[("docs/sub/loc.md".to_string(), None)],
            )
            .await
            .unwrap();

        let req = make_move_req(
            "docs/sub/loc.md",
            "archive/2024/loc.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert_eq!(
            success.rewritten_paths,
            vec!["other/deep/ref.md".to_string()]
        );

        let referencing_after =
            std::fs::read_to_string(work.path().join("other/deep/ref.md")).unwrap();
        assert!(
            referencing_after.contains("[it](../../archive/2024/loc.md)"),
            "expected a correctly relativized (climbing) link, got: {referencing_after}"
        );
    }

    #[tokio::test]
    async fn move_rewrite_skips_links_inside_fences_and_code_spans() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let referencing_original = "---\ntitle: Referencer\ndescription: d\ntype: guide\n\
             tags: [t]\n---\n\nSee [Real Link](old/loc.md) for docs.\n\n\
             ```md\n[Fenced](old/loc.md)\n```\n\n\
             Use `[Code](old/loc.md)` literally.\n";
        std::fs::write(work.path().join("referencing.md"), referencing_original).unwrap();
        git_commit_all(&work, "referencing.md", "add referencing.md");

        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencing.md",
                "markdown",
                &[("old/loc.md".to_string(), None)],
            )
            .await
            .unwrap();

        let req = make_move_req(
            "old/loc.md",
            "new/loc-rewrite-test3.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.rewritten_paths, vec!["referencing.md".to_string()]);
        let referencing_after =
            std::fs::read_to_string(work.path().join("referencing.md")).unwrap();
        assert!(
            referencing_after.contains("[Real Link](new/loc-rewrite-test3.md)"),
            "the real inline link must be rewritten, got: {referencing_after}"
        );
        assert!(
            referencing_after.contains("[Fenced](old/loc.md)"),
            "a link inside a fenced code block must NOT be rewritten, got: {referencing_after}"
        );
        assert!(
            referencing_after.contains("`[Code](old/loc.md)`"),
            "a link inside an inline code span must NOT be rewritten, got: {referencing_after}"
        );
    }

    // -----------------------------------------------------------------------
    // Wiki pipe-alias links `[[target|Display]]` — fix #131
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn move_rewrites_a_referencing_documents_pipe_alias_link_with_alias_preserved() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let referencing_original = "---\ntitle: Referencer\ndescription: d\ntype: guide\n\
             tags: [t]\n---\n\nSee [[old/loc.md|Display Text]] for more.\n";
        std::fs::write(work.path().join("referencing.md"), referencing_original).unwrap();
        git_commit_all(&work, "referencing.md", "add referencing.md");

        let harness = git_backed_harness_with_state_db(&work).await;
        // Seeded directly, matching every other incoming-link-rewrite test in this
        // file, to isolate what THIS module is responsible for: once a
        // referencing document is known/visited (`document_links` has an edge to
        // it, from wherever), its pipe-alias occurrences must be found and
        // correctly rewritten with the alias intact. `document_links` itself is
        // now populated for pipe-alias links too by `ingest::extract_markdown_links`
        // (fix #131) — see `move_rewrites_a_referencing_documents_pipe_alias_link_\
        // found_purely_through_the_real_extractor` below for the test that proves
        // THAT half end-to-end, with no manual seeding.
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencing.md",
                "markdown",
                &[("old/loc.md".to_string(), None)],
            )
            .await
            .unwrap();

        let req = make_move_req(
            "old/loc.md",
            "new/loc-alias-test1.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert_eq!(success.rewritten_paths, vec!["referencing.md".to_string()]);

        let referencing_after =
            std::fs::read_to_string(work.path().join("referencing.md")).unwrap();
        assert!(
            referencing_after.contains("[[new/loc-alias-test1.md|Display Text]]"),
            "the pipe-alias link's target must be rewritten to the new location while its \
             alias survives byte-identical, got: {referencing_after}"
        );
        assert!(!referencing_after.contains("old/loc.md"));
    }

    #[tokio::test]
    async fn move_rewrites_pipe_alias_link_whose_alias_contains_path_like_characters() {
        // The alias itself may contain `/` and look like a path — the rewriter
        // must split on the FIRST `|` only and never mistake alias text for
        // more of the target.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let referencing_original = "---\ntitle: Referencer\ndescription: d\ntype: guide\n\
             tags: [t]\n---\n\nSee [[old/loc.md|old/style/looking/alias]] for more.\n";
        std::fs::write(work.path().join("referencing.md"), referencing_original).unwrap();
        git_commit_all(&work, "referencing.md", "add referencing.md");

        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencing.md",
                "markdown",
                &[("old/loc.md".to_string(), None)],
            )
            .await
            .unwrap();

        let req = make_move_req(
            "old/loc.md",
            "new/loc-alias-test2.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.rewritten_paths, vec!["referencing.md".to_string()]);
        let referencing_after =
            std::fs::read_to_string(work.path().join("referencing.md")).unwrap();
        assert!(
            referencing_after.contains("[[new/loc-alias-test2.md|old/style/looking/alias]]"),
            "the path-like alias must survive byte-identical while only the target is \
             rewritten, got: {referencing_after}"
        );
        assert!(!referencing_after.contains("old/loc.md"));
    }

    #[tokio::test]
    async fn move_rewrite_skips_pipe_alias_links_inside_fences_and_code_spans() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let referencing_original = "---\ntitle: Referencer\ndescription: d\ntype: guide\n\
             tags: [t]\n---\n\nSee [[old/loc.md|Real Alias]] for docs.\n\n\
             ```md\n[[old/loc.md|Fenced Alias]]\n```\n\n\
             Use `[[old/loc.md|Code Alias]]` literally.\n";
        std::fs::write(work.path().join("referencing.md"), referencing_original).unwrap();
        git_commit_all(&work, "referencing.md", "add referencing.md");

        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencing.md",
                "markdown",
                &[("old/loc.md".to_string(), None)],
            )
            .await
            .unwrap();

        let req = make_move_req(
            "old/loc.md",
            "new/loc-alias-test3.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.rewritten_paths, vec!["referencing.md".to_string()]);
        let referencing_after =
            std::fs::read_to_string(work.path().join("referencing.md")).unwrap();
        assert!(
            referencing_after.contains("[[new/loc-alias-test3.md|Real Alias]]"),
            "the real pipe-alias link must be rewritten, got: {referencing_after}"
        );
        assert!(
            referencing_after.contains("[[old/loc.md|Fenced Alias]]"),
            "a pipe-alias link inside a fenced code block must NOT be rewritten, got: \
             {referencing_after}"
        );
        assert!(
            referencing_after.contains("`[[old/loc.md|Code Alias]]`"),
            "a pipe-alias link inside an inline code span must NOT be rewritten, got: \
             {referencing_after}"
        );
    }

    #[tokio::test]
    async fn move_rewrites_a_pipe_alias_self_reference_with_alias_preserved() {
        // The moved document's own outbound pipe-alias link to itself — exercised
        // through `rewrite_outbound_links`, which scans `new_content` directly and
        // has no dependency on the `document_links` reverse-lookup index at all,
        // unlike the incoming-link-rewrite tests above.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original = "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [[loc.md|Myself]] too.\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let harness = git_backed_harness(&work);

        let req = make_move_req(
            "old/loc.md",
            "new/loc-self-alias.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);

        let dest_content =
            std::fs::read_to_string(work.path().join("new/loc-self-alias.md")).unwrap();
        assert!(
            dest_content.contains("[[loc-self-alias.md|Myself]]"),
            "the self-referencing pipe-alias link must be rewritten relative to the \
             destination's own directory, with the alias preserved, got: {dest_content}"
        );
        assert!(!dest_content.contains("old/loc.md"));
    }

    #[tokio::test]
    async fn move_directory_rewrites_an_outside_referencing_documents_pipe_alias_link() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old6")).unwrap();
        let a = "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# A\n";
        std::fs::write(work.path().join("old6/a.md"), a).unwrap();
        git_commit_paths(&work, &["old6/a.md"], "add old6/a.md");

        let referencing = "---\ntitle: Referencer\ndescription: d\ntype: guide\ntags: [t]\n---\n\n\
                            See [[old6/a.md|A Doc]] for more.\n";
        std::fs::write(work.path().join("referencing6.md"), referencing).unwrap();
        git_commit_paths(&work, &["referencing6.md"], "add referencing6.md");

        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencing6.md",
                "markdown",
                &[("old6/a.md".to_string(), None)],
            )
            .await
            .unwrap();

        let success = move_directory(&harness.deps(), "old6", "new6", None)
            .await
            .unwrap();
        assert_eq!(success.moved.len(), 1);
        assert_eq!(success.rewritten_paths, vec!["referencing6.md".to_string()]);

        let ref_after = std::fs::read_to_string(work.path().join("referencing6.md")).unwrap();
        assert!(
            ref_after.contains("[[new6/a.md|A Doc]]"),
            "the outside document's pipe-alias link must resolve to the moved document's new \
             location with its alias preserved, got: {ref_after}"
        );
        assert!(!ref_after.contains("old6/a.md"));
    }

    #[tokio::test]
    async fn move_rewrites_a_referencing_documents_pipe_alias_link_found_purely_through_the_real_extractor()
     {
        // The decisive test for fix #131's actual bug: a referencing document
        // whose ONLY link to the moved target is a pipe-alias link. Unlike every
        // other pipe-alias test above, `document_links` is NOT seeded with a
        // hand-written literal here — it is populated by calling
        // `ingest::extract_markdown_links` on the referencing body, exactly as
        // production indexing does. Before fix #131, `extract_markdown_links`
        // rejected any wiki target containing `|` and returned no edge at all for
        // this document, so `links_targeting` (which this rewrite is built on)
        // would never surface it, `rewritten_paths` would stay empty, and the
        // file on disk would still contain "old/loc.md" after the move — the
        // exact silent-broken-link failure #131 is about. After the fix,
        // `extract_markdown_links` records the edge like any other wiki link, so
        // this document is visited and rewritten like any other.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        // This document's ONLY outbound link is the pipe-alias one — no bare
        // `[[old/loc.md]]` or `[text](old/loc.md)` alongside it that would let a
        // stale/partial extractor find it by a different route.
        let referencing_original = "---\ntitle: Referencer\ndescription: d\ntype: guide\n\
             tags: [t]\n---\n\nSee [[old/loc.md|Display Text]] for more.\n";
        std::fs::write(work.path().join("referencing.md"), referencing_original).unwrap();
        git_commit_all(&work, "referencing.md", "add referencing.md");

        let harness = git_backed_harness_with_state_db(&work).await;
        let extracted =
            crate::ingest::extract_markdown_links(referencing_original, "referencing.md");
        let link_targets: Vec<(String, Option<f64>)> =
            extracted.into_iter().map(|t| (t, None)).collect();
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links("referencing.md", "markdown", &link_targets)
            .await
            .unwrap();

        let req = make_move_req(
            "old/loc.md",
            "new/loc-real-extractor-test.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert_eq!(
            success.rewritten_paths,
            vec!["referencing.md".to_string()],
            "the real extractor must have recorded the pipe-alias edge in \
             document_links for links_targeting to find this document at all"
        );

        let referencing_after =
            std::fs::read_to_string(work.path().join("referencing.md")).unwrap();
        assert!(
            referencing_after.contains("[[new/loc-real-extractor-test.md|Display Text]]"),
            "the pipe-alias link's target must be rewritten to the new location while its \
             alias survives byte-identical, got: {referencing_after}"
        );
        assert!(!referencing_after.contains("old/loc.md"));
    }

    #[tokio::test]
    async fn move_rewrites_a_self_reference_relative_to_the_destination() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original = "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [self](loc.md) too.\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        // No `with_state_db` — the self-reference rewrite is computed purely from
        // `new_content` and does not depend on the reverse-link index at all, so
        // this must work identically whether or not `WriteDeps::state` is set.
        let harness = git_backed_harness(&work);

        let req = make_move_req(
            "old/loc.md",
            "new/loc-self.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(
            success.rewritten_paths.is_empty(),
            "the moved document itself is not a separate 'referencing document'"
        );

        let dest_content = std::fs::read_to_string(work.path().join("new/loc-self.md")).unwrap();
        assert!(
            dest_content.contains("[self](loc-self.md)"),
            "the self-link must be rewritten relative to the destination's own directory, got: \
             {dest_content}"
        );
        assert!(!dest_content.contains("old/loc.md"));
    }

    /// Guards the self-reference filter (`o.resolved.as_str() == source_rel`,
    /// step 6.5 above) against being "fixed" to also match on raw link text
    /// (`|| o.raw == source_rel`).
    ///
    /// Markdown link targets in this codebase are ALWAYS resolved relative to
    /// the containing document's own directory — there is no root-relative
    /// form, not even via a leading `/` (`ingest::resolve_relative_md_path`
    /// treats it as an empty, no-op component). That means a document's raw
    /// link text can be textually identical to that same document's own
    /// repo-relative path while resolving somewhere else entirely. Here,
    /// `old/loc.md` contains a link literally written as `old/loc.md`; from
    /// inside `old/`, that resolves to `old/old/loc.md` — a different
    /// document — NOT to `old/loc.md` itself.
    ///
    /// The move DOES legitimately rewrite this link's spelling — every
    /// outbound link in the moved document gets re-relativized, because a
    /// relative link's meaning depends on the containing document's
    /// directory, and that directory just changed. "Unchanged text" was
    /// never the invariant. What must be preserved is the link's *resolved
    /// target*: comparing the self-reference filter on `o.resolved` (the
    /// actually-resolved target) correctly re-relativizes this link while
    /// keeping it pointed at `old/old/loc.md`. Broadening that comparison to
    /// `o.resolved.as_str() == source_rel || o.raw == source_rel` would
    /// corrupt it: the raw text matches `source_rel` by coincidence, so the
    /// rewrite would mistake it for a self-reference and repoint it at the
    /// moved document — a file it never referenced — while the real target,
    /// `old/old/loc.md`, is silently dropped. That broadened check only
    /// "works" by accident for documents living at the KB root, where raw
    /// text and resolved path happen to coincide; it is wrong everywhere
    /// else.
    #[tokio::test]
    async fn move_preserves_the_target_of_a_link_whose_raw_text_matches_the_source_path() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original = "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [not self](old/loc.md) too.\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        // No `with_state_db` — the self-reference rewrite is computed purely from
        // `new_content` and does not depend on the reverse-link index at all, so
        // this must work identically whether or not `WriteDeps::state` is set.
        let harness = git_backed_harness(&work);

        let req = make_move_req("old/loc.md", "new/loc.md", source_original, source_original);
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(
            success.rewritten_paths.is_empty(),
            "the moved document itself is not a separate 'referencing document'"
        );

        let dest_content = std::fs::read_to_string(work.path().join("new/loc.md")).unwrap();
        let occurrences =
            crate::ingest::find_markdown_link_occurrences(&dest_content, "new/loc.md");
        let not_self = occurrences
            .iter()
            .find(|o| o.raw.contains("loc.md"))
            .unwrap_or_else(|| panic!("expected a link to loc.md in: {dest_content}"));
        assert_eq!(
            not_self.resolved, "old/old/loc.md",
            "the link's raw text happens to equal the source path but must keep resolving \
             to old/old/loc.md, a different document, after the move — got raw text {:?} \
             in: {dest_content}",
            not_self.raw
        );
        assert_ne!(
            not_self.resolved, "new/loc.md",
            "the link must not be mistaken for a self-reference (matching on raw text \
             instead of resolved target) and hijacked onto the moved document — got: \
             {dest_content}"
        );
    }

    #[tokio::test]
    async fn move_rewrites_a_non_self_link_across_a_depth_change() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        std::fs::create_dir_all(work.path().join("shared")).unwrap();
        std::fs::write(
            work.path().join("shared/doc.md"),
            "---\ntitle: Shared\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Shared\n",
        )
        .unwrap();
        git_commit_all(&work, "shared/doc.md", "add shared/doc.md");

        let source_original = "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [shared](../shared/doc.md) too.\n";
        std::fs::write(work.path().join("old/a.md"), source_original).unwrap();
        git_commit_all(&work, "old/a.md", "add old/a.md");

        let harness = git_backed_harness(&work);

        let req = make_move_req(
            "old/a.md",
            "new/deep/a.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);

        let dest_content = std::fs::read_to_string(work.path().join("new/deep/a.md")).unwrap();
        // Before the fix, this link's raw text (`../shared/doc.md`) would be left
        // unchanged by the move and silently resolve to `new/shared/doc.md` — a
        // different, likely nonexistent, file — once read from the destination's
        // deeper directory. Assert the actual resolved target, not just a string
        // match, so this states the real invariant.
        let occurrences =
            crate::ingest::find_markdown_link_occurrences(&dest_content, "new/deep/a.md");
        assert_eq!(
            occurrences.len(),
            1,
            "expected exactly one outbound link, got: {dest_content}"
        );
        assert_eq!(
            occurrences[0].resolved, "shared/doc.md",
            "the link must still resolve to shared/doc.md after the move, got raw text {:?} \
             in: {dest_content}",
            occurrences[0].raw
        );
        assert!(
            dest_content.contains("[shared](../../shared/doc.md)"),
            "expected a correctly re-relativized (deeper-climbing) link, got: {dest_content}"
        );
    }

    #[tokio::test]
    async fn move_rewrites_a_wiki_link_and_a_reference_definition_across_a_depth_change() {
        // Companion to `move_rewrites_a_non_self_link_across_a_depth_change`, but for
        // the two syntaxes that function predates: a wiki-style `[[target]]` link and
        // a reference-style definition, both pointing at a document that stays put
        // while the mover's own depth changes. Both must have their relative
        // spelling recomputed exactly like an inline link does — this is the "new
        // syntax exercised across a move with a depth change" case called for
        // alongside the inline-only depth-change test above.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        std::fs::create_dir_all(work.path().join("shared")).unwrap();
        std::fs::write(
            work.path().join("shared/doc.md"),
            "---\ntitle: Shared\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Shared\n",
        )
        .unwrap();
        git_commit_all(&work, "shared/doc.md", "add shared/doc.md");

        let source_original = "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [[../shared/doc]] and [Shared][ref] too.\n\n[ref]: ../shared/doc.md\n";
        std::fs::write(work.path().join("old/a.md"), source_original).unwrap();
        git_commit_all(&work, "old/a.md", "add old/a.md");

        let harness = git_backed_harness(&work);

        let req = make_move_req(
            "old/a.md",
            "new/deep/a.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);

        let dest_content = std::fs::read_to_string(work.path().join("new/deep/a.md")).unwrap();
        let occurrences =
            crate::ingest::find_markdown_link_occurrences(&dest_content, "new/deep/a.md");
        assert_eq!(
            occurrences.len(),
            2,
            "expected exactly one wiki-link occurrence and one reference-definition \
             occurrence, got: {dest_content}"
        );
        assert!(
            occurrences.iter().all(|o| o.resolved == "shared/doc.md"),
            "both must still resolve to shared/doc.md after the move, got: {occurrences:?}"
        );

        // The wiki link's bracketed target is rewritten the same way an inline
        // link's parenthesized target is: `relativize_md_path` always emits the
        // full `.md`-suffixed path, so the extension the author omitted is now
        // explicit — the link still resolves identically either way.
        assert!(
            dest_content.contains("[[../../shared/doc.md]]"),
            "expected the wiki link's climb to deepen from ../ to ../../, got: {dest_content}"
        );
        // The reference DEFINITION is rewritten once...
        assert!(
            dest_content.contains("[ref]: ../../shared/doc.md"),
            "expected the reference definition's climb to deepen from ../ to ../../, got: \
             {dest_content}"
        );
        // ...and the use site is left completely untouched.
        assert!(
            dest_content.contains("[Shared][ref]"),
            "the reference use site's text must be untouched, got: {dest_content}"
        );
    }

    #[tokio::test]
    async fn move_rewrites_a_non_self_link_across_a_sibling_directory_change() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("teams/alpha")).unwrap();
        std::fs::create_dir_all(work.path().join("shared/inner")).unwrap();
        std::fs::write(
            work.path().join("shared/inner/target.md"),
            "---\ntitle: Target\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Target\n",
        )
        .unwrap();
        git_commit_all(
            &work,
            "shared/inner/target.md",
            "add shared/inner/target.md",
        );

        let source_original = "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [target](../../shared/inner/target.md) too.\n";
        std::fs::write(work.path().join("teams/alpha/doc.md"), source_original).unwrap();
        git_commit_all(&work, "teams/alpha/doc.md", "add teams/alpha/doc.md");

        let harness = git_backed_harness(&work);

        let req = make_move_req(
            "teams/alpha/doc.md",
            "shared/beta/doc.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);

        let dest_content = std::fs::read_to_string(work.path().join("shared/beta/doc.md")).unwrap();
        // Same directory DEPTH (2 components) on both sides, so this is testing
        // something `move_rewrites_a_non_self_link_across_a_depth_change` does not:
        // the destination now shares a top-level ancestor with the target it never
        // shared before, so the correct climb SHRINKS from 2 segments to 1, not
        // grows.
        let occurrences =
            crate::ingest::find_markdown_link_occurrences(&dest_content, "shared/beta/doc.md");
        assert_eq!(occurrences.len(), 1, "got: {dest_content}");
        assert_eq!(occurrences[0].resolved, "shared/inner/target.md");
        assert!(
            dest_content.contains("[target](../inner/target.md)"),
            "expected the climb to shorten from ../../ to ../, got: {dest_content}"
        );
    }

    #[tokio::test]
    async fn move_pure_rename_in_same_directory_leaves_non_self_links_byte_identical() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("docs")).unwrap();
        std::fs::create_dir_all(work.path().join("other")).unwrap();
        std::fs::write(
            work.path().join("other/x.md"),
            "---\ntitle: Other\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Other\n",
        )
        .unwrap();
        git_commit_all(&work, "other/x.md", "add other/x.md");

        let source_original = "---\ntitle: Rename Me\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [other](../other/x.md) too.\n";
        std::fs::write(work.path().join("docs/a.md"), source_original).unwrap();
        git_commit_all(&work, "docs/a.md", "add docs/a.md");

        let harness = git_backed_harness(&work);

        let req = make_move_req("docs/a.md", "docs/b.md", source_original, source_original);
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);

        let dest_content = std::fs::read_to_string(work.path().join("docs/b.md")).unwrap();
        assert_eq!(
            dest_content, source_original,
            "a pure rename within one directory must leave non-self outbound links \
             byte-identical — no spurious rewrite, no needless diff"
        );
    }

    #[tokio::test]
    async fn move_outbound_rewrite_skips_links_inside_fences_and_code_spans() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        std::fs::create_dir_all(work.path().join("shared")).unwrap();
        std::fs::write(
            work.path().join("shared/doc.md"),
            "---\ntitle: Shared\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Shared\n",
        )
        .unwrap();
        git_commit_all(&work, "shared/doc.md", "add shared/doc.md");

        let source_original = "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [Real Link](../shared/doc.md) for docs.\n\n\
             ```md\n[Fenced](../shared/doc.md)\n```\n\n\
             Use `[Code](../shared/doc.md)` literally.\n";
        std::fs::write(work.path().join("old/a.md"), source_original).unwrap();
        git_commit_all(&work, "old/a.md", "add old/a.md");

        let harness = git_backed_harness(&work);

        let req = make_move_req(
            "old/a.md",
            "new/deep/a.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);

        let dest_content = std::fs::read_to_string(work.path().join("new/deep/a.md")).unwrap();
        assert!(
            dest_content.contains("[Real Link](../../shared/doc.md)"),
            "the real inline link must be re-relativized, got: {dest_content}"
        );
        assert!(
            dest_content.contains("[Fenced](../shared/doc.md)"),
            "a link inside a fenced code block must NOT be rewritten, got: {dest_content}"
        );
        assert!(
            dest_content.contains("`[Code](../shared/doc.md)`"),
            "a link inside an inline code span must NOT be rewritten, got: {dest_content}"
        );
    }

    #[tokio::test]
    async fn move_with_a_stale_document_links_row_for_a_deleted_referencing_file_does_not_fail() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let harness = git_backed_harness_with_state_db(&work).await;
        // `ghost.md` was never written to disk — a stale row, as if the
        // referencing document had since been deleted without the index
        // catching up yet.
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links("ghost.md", "markdown", &[("old/loc.md".to_string(), None)])
            .await
            .unwrap();

        let req = make_move_req(
            "old/loc.md",
            "new/loc-rewrite-test5.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req)
            .await
            .expect("a stale document_links row must not fail the move");

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(
            success.rewritten_paths.is_empty(),
            "nothing was actually rewritten — the referencing file doesn't exist"
        );
        assert!(!work.path().join("old/loc.md").exists());
        assert!(work.path().join("new/loc-rewrite-test5.md").exists());
    }

    #[tokio::test]
    async fn move_precommit_failure_with_rewrites_restores_the_referencing_document_too() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let referencing_original = "---\ntitle: Referencer\ndescription: d\ntype: guide\n\
             tags: [t]\n---\n\nSee [the moved doc](old/loc.md) for more.\n";
        std::fs::write(work.path().join("referencing.md"), referencing_original).unwrap();
        git_commit_all(&work, "referencing.md", "add referencing.md");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencing.md",
                "markdown",
                &[("old/loc.md".to_string(), None)],
            )
            .await
            .unwrap();

        let req = make_move_req(
            "old/loc.md",
            "new/loc-rewrite-test6.md",
            source_original,
            source_original,
        );
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        match err {
            WriteError::PreCommitFailed { rolled_back, .. } => assert!(rolled_back),
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }

        assert!(
            work.path().join("old/loc.md").exists(),
            "source must be restored after a rolled-back move"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("old/loc.md")).unwrap(),
            source_original
        );
        assert!(!work.path().join("new/loc-rewrite-test6.md").exists());

        let referencing_after =
            std::fs::read_to_string(work.path().join("referencing.md")).unwrap();
        assert_eq!(
            referencing_after, referencing_original,
            "the referencing document's link rewrite must be rolled back too, not just the move"
        );

        assert_eq!(head_before, head_sha(&work));
        assert_eq!(git_status(&work), "");
    }

    /// Covers write.rs's step-10.5 self-contained rollback (fires when a
    /// `tokio::fs::write` into a referencing document fails DURING the
    /// rewrite loop, before git is touched at all) — a distinct code path
    /// from `move_precommit_failure_with_rewrites_restores_the_referencing_document_too`
    /// above, which exercises the LATER rollback triggered by `git commit`
    /// itself failing (#145).
    ///
    /// Two referencing documents point at the source. `StateDb::links_targeting`
    /// returns sources `ORDER BY source_path`, so `ref-a.md` is rewritten
    /// FIRST and lands on disk successfully, then `ref-b.md`'s write is forced
    /// to fail (its permission bits stripped to read-only) — standing in for
    /// a permission race or a full disk hitting the SECOND of several
    /// referencing documents mid-loop, which is exactly the scenario the
    /// issue's "duplicated at both old and new locations, or another
    /// document's content corrupted" failure mode depends on. The rollback
    /// must undo THREE things, not just the failing write: the
    /// already-rewritten `ref-a.md`, the source removal, and the destination
    /// write — proving the loop's own by-hand unwind (not the later
    /// git-commit-triggered one) is correct.
    #[tokio::test]
    async fn move_link_rewrite_write_failure_rolls_back_the_move_and_every_already_rewritten_document()
     {
        use std::os::unix::fs::PermissionsExt;

        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let ref_a_original = "---\ntitle: Ref A\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [the moved doc](old/loc.md) for more.\n";
        let ref_b_original = "---\ntitle: Ref B\ndescription: d\ntype: guide\ntags: [t]\n\
             ---\n\nSee [the moved doc](old/loc.md) too.\n";
        std::fs::write(work.path().join("ref-a.md"), ref_a_original).unwrap();
        std::fs::write(work.path().join("ref-b.md"), ref_b_original).unwrap();
        git_commit_paths(&work, &["ref-a.md", "ref-b.md"], "add referencing docs");
        let head_before = head_sha(&work);

        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links("ref-a.md", "markdown", &[("old/loc.md".to_string(), None)])
            .await
            .unwrap();
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links("ref-b.md", "markdown", &[("old/loc.md".to_string(), None)])
            .await
            .unwrap();

        // Read-only: `read_to_string` (earlier in the same loop iteration)
        // still succeeds, so the code genuinely reaches — and fails at — the
        // write this test targets, rather than taking the earlier "stale
        // row, skip" branch on a read failure. That read-succeeds/write-fails
        // asymmetry is why this uses a permission bit rather than, say,
        // replacing the file with a directory: EISDIR would fail the *read*
        // and route around the branch under test entirely.
        //
        // The catch is that DAC permission bits do not stop a privileged
        // writer. Root — or anything holding CAP_DAC_OVERRIDE — writes through
        // 0o444 unimpeded, the move then succeeds, and `unwrap_err()` below
        // panics on an `Ok`. This repo has already been bitten by exactly that:
        // see `git::tests::reject_pushes`, whose doc comment records a
        // read-only-remote test that passed locally and failed on the
        // self-hosted CI runner for this reason.
        //
        // There is no privilege-independent way to make one regular file
        // readable but not writable, so rather than assume the injection
        // worked, probe it: if the permission bit does not actually block a
        // write here, skip loudly instead of reporting a failure that says
        // nothing about the code under test.
        let ref_b_path = work.path().join("ref-b.md");
        let mut perms = std::fs::metadata(&ref_b_path).unwrap().permissions();
        perms.set_mode(0o444);
        std::fs::set_permissions(&ref_b_path, perms).unwrap();

        if std::fs::OpenOptions::new()
            .write(true)
            .open(&ref_b_path)
            .is_ok()
        {
            let mut perms = std::fs::metadata(&ref_b_path).unwrap().permissions();
            perms.set_mode(0o644);
            std::fs::set_permissions(&ref_b_path, perms).unwrap();
            eprintln!(
                "SKIP move_link_rewrite_write_failure_rolls_back_the_move_and_every_already_rewritten_document: \
                 this process can write through a 0o444 file (running as root or with \
                 CAP_DAC_OVERRIDE), so the write failure this test injects cannot be produced. \
                 Run as an unprivileged user to exercise the link-rewrite rollback."
            );
            return;
        }

        let req = make_move_req("old/loc.md", "new/loc.md", source_original, source_original);
        let err = write_document(&harness.deps(), req).await.unwrap_err();
        assert!(
            matches!(err, WriteError::Io { .. }),
            "expected Io (the write-failure branch, not a git failure), got {err:?}"
        );

        // Restore permissions before the assertions below (and the tempdir's
        // own cleanup) touch the file again.
        let mut perms = std::fs::metadata(&ref_b_path).unwrap().permissions();
        perms.set_mode(0o644);
        std::fs::set_permissions(&ref_b_path, perms).unwrap();

        assert!(
            !work.path().join("new/loc.md").exists(),
            "destination must be removed by the rollback"
        );
        assert!(
            work.path().join("old/loc.md").exists(),
            "source must be restored by the rollback"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("old/loc.md")).unwrap(),
            source_original
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("ref-a.md")).unwrap(),
            ref_a_original,
            "the first (already-rewritten) referencing document must be restored too, not \
             just the move itself"
        );

        // This rollback runs entirely before git is touched — no `git add`
        // ever ran, so there is nothing to unstage and HEAD never moved.
        assert_eq!(head_before, head_sha(&work));
        assert_eq!(git_status(&work), "");
    }

    #[tokio::test]
    async fn move_with_no_state_db_does_not_rewrite_a_genuinely_referencing_document() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old")).unwrap();
        let source_original =
            "---\ntitle: Move Me\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
        std::fs::write(work.path().join("old/loc.md"), source_original).unwrap();
        git_commit_all(&work, "old/loc.md", "add old/loc.md");

        let referencing_original = "---\ntitle: Referencer\ndescription: d\ntype: guide\n\
             tags: [t]\n---\n\nSee [the moved doc](old/loc.md) for more.\n";
        std::fs::write(work.path().join("referencing.md"), referencing_original).unwrap();
        git_commit_all(&work, "referencing.md", "add referencing.md");

        // Plain `git_backed_harness`, with no `with_state_db` — `WriteDeps::state`
        // is `None`, so this exercises the "existing non-move / no-DB tests still
        // pass" contract even though a real referencing document (and, unlike
        // `move_precommit_failure_with_rewrites_restores_the_referencing_document_too`,
        // no `document_links` row to find it by) exists on disk.
        let harness = git_backed_harness(&work);

        let req = make_move_req(
            "old/loc.md",
            "new/loc-rewrite-test7.md",
            source_original,
            source_original,
        );
        let success = write_document(&harness.deps(), req).await.unwrap();

        assert_eq!(success.outcome, WriteOutcome::Synced);
        assert!(success.rewritten_paths.is_empty());
        let referencing_after =
            std::fs::read_to_string(work.path().join("referencing.md")).unwrap();
        assert_eq!(
            referencing_after, referencing_original,
            "with no state DB wired up, the referencing document must be left untouched"
        );
    }

    // -----------------------------------------------------------------------
    // write_documents_batch: multiple documents, one commit (#180)
    // -----------------------------------------------------------------------

    fn make_batch_req<'a>(
        rel_path: &'a str,
        _old_content: &'a str,
        new_content: &'a str,
        is_create: bool,
    ) -> BatchWriteRequest<'a> {
        BatchWriteRequest {
            rel_path,
            change: if is_create {
                DocChange::Create(new_content)
            } else {
                overwrite(new_content)
            },
            force_new: Some(true),
            expected_version: None,
        }
    }

    #[tokio::test]
    async fn write_documents_batch_lands_every_document_in_one_commit_and_marks_paths_dirty() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let head_before = head_sha(&work);
        let harness = git_backed_harness(&work);

        let requests = vec![
            make_batch_req(
                "docs/batch-one.md",
                "",
                "---\ntitle: One\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# One\n",
                true,
            ),
            make_batch_req(
                "docs/batch-two.md",
                "",
                "---\ntitle: Two\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Two\n",
                true,
            ),
        ];

        let success = write_documents_batch(&harness.deps(), &requests, None)
            .await
            .expect("batch write should succeed");

        assert_eq!(success.documents.len(), 2);
        assert_eq!(success.documents[0].rel_path, "docs/batch-one.md");
        assert!(success.documents[0].is_create);
        assert!(success.documents[0].diff.contains("+title: One"));
        assert_eq!(success.documents[1].rel_path, "docs/batch-two.md");
        assert!(success.documents[1].diff.contains("+title: Two"));

        assert!(work.path().join("docs/batch-one.md").exists());
        assert!(work.path().join("docs/batch-two.md").exists());

        // Exactly ONE commit landed for both documents — the whole point of
        // batching (#180) — not two.
        let head_after = head_sha(&work);
        let count_out = std::process::Command::new("git")
            .args([
                "rev-list",
                "--count",
                &format!("{head_before}..{head_after}"),
            ])
            .current_dir(work.path())
            .output()
            .unwrap();
        let count = String::from_utf8_lossy(&count_out.stdout)
            .trim()
            .to_string();
        assert_eq!(count, "1", "both documents must land in exactly one commit");

        let show_out = std::process::Command::new("git")
            .args(["show", "--name-only", "--format=", "HEAD"])
            .current_dir(work.path())
            .output()
            .unwrap();
        let show_str = String::from_utf8_lossy(&show_out.stdout);
        assert!(show_str.contains("docs/batch-one.md"));
        assert!(show_str.contains("docs/batch-two.md"));

        crate::reindex::test_support::assert_marked_dirty(
            &harness.reindex_queue,
            &["docs/batch-one.md", "docs/batch-two.md"],
        );
    }

    #[tokio::test]
    async fn write_documents_batch_rejects_more_than_the_size_cap() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let owned: Vec<(String, String)> = (0..(MAX_BATCH_DOCUMENTS + 1))
            .map(|i| (format!("docs/over-cap-{i}.md"), format!("# doc {i}")))
            .collect();
        let requests: Vec<BatchWriteRequest<'_>> = owned
            .iter()
            .map(|(path, content)| make_batch_req(path, "", content, true))
            .collect();

        let err = write_documents_batch(&harness.deps(), &requests, None)
            .await
            .expect_err("over-cap batch must be rejected");
        match err {
            BatchWriteError::TooMany { count, max } => {
                assert_eq!(count, MAX_BATCH_DOCUMENTS + 1);
                assert_eq!(max, MAX_BATCH_DOCUMENTS);
            }
            other => panic!("expected TooMany, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn write_documents_batch_rejects_a_duplicate_path() {
        let tmp = tempfile::tempdir().unwrap();
        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let requests = vec![
            make_batch_req(
                "docs/dup.md",
                "",
                "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# A\n",
                true,
            ),
            make_batch_req(
                "docs/dup.md",
                "",
                "---\ntitle: B\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# B\n",
                true,
            ),
        ];

        let err = write_documents_batch(&harness.deps(), &requests, None)
            .await
            .expect_err("a duplicate path within one batch must be rejected");
        match err {
            BatchWriteError::DuplicatePath { rel_path } => {
                assert_eq!(rel_path, "docs/dup.md");
            }
            other => panic!("expected DuplicatePath, got {other:?}"),
        }
    }

    /// Validation failure (bad frontmatter) on ONE document in an otherwise
    /// valid batch must fail the WHOLE batch before anything is written —
    /// mirrors `move_directory`'s own "validate every document before
    /// touching any of them" contract, extended to a batch of otherwise
    /// unrelated documents. Verified to fail before this feature existed
    /// (there was no `write_documents_batch` to call at all) and to pass
    /// after.
    #[tokio::test]
    async fn write_documents_batch_validation_failure_touches_nothing() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let head_before = head_sha(&work);

        // `git_backed_harness`'s default test config has no required
        // frontmatter fields configured — mirrors
        // `validation_failure_carries_the_structured_result`'s own need to
        // set this explicitly to get a real validation failure to test
        // against.
        let mut config = crate::mcp::make_test_resolved_config(work.path());
        {
            let cfg = Arc::get_mut(&mut config).unwrap();
            cfg.write.dedup_enabled = false;
            cfg.frontmatter.required = vec!["title".into()];
        }
        let harness = Harness::new(&work, config);

        let requests = vec![
            make_batch_req(
                "docs/valid.md",
                "",
                "---\ntitle: Valid\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Valid\n",
                true,
            ),
            // Missing the required `title` field.
            make_batch_req(
                "docs/invalid.md",
                "",
                "---\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Invalid\n",
                true,
            ),
        ];

        let err = write_documents_batch(&harness.deps(), &requests, None)
            .await
            .expect_err("a validation failure on one document must fail the whole batch");
        match err {
            BatchWriteError::Documents { failures } => {
                assert_eq!(
                    failures.len(),
                    1,
                    "only the invalid document should be reported"
                );
                assert_eq!(failures[0].0, "docs/invalid.md");
                assert!(matches!(failures[0].1, WriteError::Validation { .. }));
            }
            other => panic!("expected Documents, got {other:?}"),
        }

        // Nothing written for EITHER document — not even the valid one.
        assert!(!work.path().join("docs/valid.md").exists());
        assert!(!work.path().join("docs/invalid.md").exists());
        assert_eq!(
            git_status(&work),
            "",
            "no filesystem change should be staged or untracked after a pre-flight failure"
        );
        assert_eq!(
            head_sha(&work),
            head_before,
            "no commit should have been made"
        );
    }

    /// A rolled-back batch restores what it read under `GIT_LOCK`, so a change
    /// another writer landed before the lock (pulled in by the pre-write sync)
    /// survives the rollback rather than being reverted to an older snapshot.
    #[tokio::test]
    async fn write_documents_batch_rollback_keeps_another_writers_change() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let original = "---\ntitle: X\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# X\n";
        std::fs::create_dir_all(work.path().join("docs")).unwrap();
        std::fs::write(work.path().join("docs/x.md"), original).unwrap();
        git_commit_all(&work, "docs/x.md", "add docs/x.md");
        std::process::Command::new("git")
            .args(["push", "origin", "master"])
            .current_dir(work.path())
            .output()
            .unwrap();

        let others =
            "---\ntitle: X\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# X by someone else\n";
        let bare_path = bare.path().to_path_buf();
        let pushed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let hook: crate::git::TestHook = Arc::new(move |point| {
            let bare_path = bare_path.clone();
            let pushed = Arc::clone(&pushed);
            Box::pin(async move {
                if point == crate::git::HookPoint::BeforeLock
                    && !pushed.swap(true, std::sync::atomic::Ordering::SeqCst)
                {
                    push_file_from_a_fresh_clone(&bare_path, "master", "docs/x.md", others);
                }
            })
        });

        force_git_commit_to_fail(&work);
        let mut config = crate::mcp::make_test_resolved_config(work.path());
        {
            let c = Arc::get_mut(&mut config).unwrap();
            c.write.dedup_enabled = false;
            c.source.git_url = Some(format!("file://{}", bare.path().to_str().unwrap()));
        }
        let harness = Harness::new(&work, config);

        let append: &'static RelativeEdit<'static> =
            Box::leak(Box::new(|current: &str| Ok(format!("{current}\nmine\n"))));
        let requests = vec![
            BatchWriteRequest {
                rel_path: "docs/x.md",
                change: DocChange::Relative(append),
                force_new: Some(true),
                expected_version: None,
            },
            make_batch_req(
                "docs/new.md",
                "",
                "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New\n",
                true,
            ),
        ];

        let deps = harness.deps();
        let err = crate::git::TEST_HOOK
            .scope(hook, write_documents_batch(&deps, &requests, None))
            .await
            .expect_err("a forced commit failure must be reported as an error");
        assert!(
            matches!(
                err,
                BatchWriteError::PreCommitFailed {
                    rolled_back: true,
                    ..
                }
            ),
            "{err:?}"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("docs/x.md")).unwrap(),
            others,
            "the rollback must restore the other writer's version, not the older one"
        );
        assert!(!work.path().join("docs/new.md").exists());
        assert_eq!(git_status(&work), "");
    }

    /// A `git commit` failure for the batch's single shared commit must roll
    /// back EVERY document this call had written to disk — mirrors
    /// `move_directory_precommit_failure_rolls_back_every_document_and_referencing_document`'s
    /// all-or-nothing rollback contract, for a batch of otherwise unrelated
    /// documents instead of a move's paired source/destination. Verified to
    /// fail before this feature existed and to pass after.
    #[tokio::test]
    async fn write_documents_batch_precommit_failure_rolls_back_every_document() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let harness = git_backed_harness(&work);

        let requests = vec![
            make_batch_req(
                "docs/rollback-one.md",
                "",
                "---\ntitle: One\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# One\n",
                true,
            ),
            make_batch_req(
                "docs/rollback-two.md",
                "",
                "---\ntitle: Two\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Two\n",
                true,
            ),
        ];

        let err = write_documents_batch(&harness.deps(), &requests, None)
            .await
            .expect_err("a forced commit failure must be reported as an error");
        match err {
            BatchWriteError::PreCommitFailed { rolled_back, .. } => {
                assert!(
                    rolled_back,
                    "rollback of every document should succeed cleanly"
                );
            }
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }

        assert!(
            !work.path().join("docs/rollback-one.md").exists(),
            "the first document's file must be removed by the rollback"
        );
        assert!(
            !work.path().join("docs/rollback-two.md").exists(),
            "the second document's file must be removed by the rollback"
        );
        assert_eq!(
            git_status(&work),
            "",
            "the git index must be back to exactly its pre-call state"
        );
        assert_eq!(
            head_sha(&work),
            head_before,
            "HEAD must be untouched — the commit never landed"
        );
    }

    /// A create whose target is taken by the time the batch writes it — here a
    /// dangling symlink, which reads as missing but makes `create_new` fail —
    /// fails the batch, and the rollback leaves that entry alone: only files
    /// this batch created are removed.
    #[tokio::test]
    async fn write_documents_batch_create_collision_does_not_remove_the_existing_entry() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("docs")).unwrap();
        let link = tmp.path().join("docs/link.md");
        std::os::unix::fs::symlink("nowhere.md", &link).unwrap();
        let mut config = crate::mcp::make_test_resolved_config(tmp.path());
        Arc::get_mut(&mut config).unwrap().write.dedup_enabled = false;
        let harness = Harness::new(&tmp, config);

        let requests = vec![
            make_batch_req(
                "docs/first.md",
                "",
                "---\ntitle: First\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# First\n",
                true,
            ),
            make_batch_req(
                "docs/link.md",
                "",
                "---\ntitle: Link\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Link\n",
                true,
            ),
        ];
        let err = write_documents_batch(&harness.deps(), &requests, None)
            .await
            .expect_err("a create onto an existing entry must fail the batch");
        match err {
            BatchWriteError::Documents { failures } => {
                assert_eq!(failures.len(), 1, "{failures:?}");
                assert_eq!(failures[0].0, "docs/link.md");
                assert!(
                    matches!(failures[0].1, WriteError::AlreadyExists),
                    "{:?}",
                    failures[0].1
                );
            }
            other => panic!("expected Documents, got {other:?}"),
        }
        assert!(
            std::fs::symlink_metadata(&link).is_ok_and(|m| m.file_type().is_symlink()),
            "the rollback must not remove an entry this batch did not create"
        );
        assert!(
            !tmp.path().join("docs/first.md").exists(),
            "the earlier create is rolled back"
        );
    }

    /// A failed write is rolled back with every document before it, and a
    /// document the rollback cannot restore is named in the error. The failing
    /// write is a read-only file, so its own restore fails too — which shows the
    /// failing document is rolled back, not just the earlier ones.
    #[tokio::test]
    async fn write_documents_batch_write_failure_names_a_document_it_could_not_restore() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("docs")).unwrap();
        let first_original =
            "---\ntitle: First\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# First\n";
        let locked_original =
            "---\ntitle: Locked\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Locked\n";
        std::fs::write(tmp.path().join("docs/first.md"), first_original).unwrap();
        let locked = tmp.path().join("docs/locked.md");
        std::fs::write(&locked, locked_original).unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o444)).unwrap();
        // Same probe as
        // `move_link_rewrite_write_failure_rolls_back_the_move_and_every_already_rewritten_document`:
        // a privileged process writes through 0o444, so the failure cannot be
        // injected. `rollback_batch_filesystem_writes_restores_a_failed_edit_and_names_the_rest`
        // covers the rollback itself either way.
        if std::fs::OpenOptions::new()
            .write(true)
            .open(&locked)
            .is_ok()
        {
            eprintln!(
                "SKIP write_documents_batch_write_failure_names_a_document_it_could_not_restore: \
                 this process can write through a 0o444 file (running as root or with \
                 CAP_DAC_OVERRIDE), so the write failure this test injects cannot be produced."
            );
            return;
        }
        let mut config = crate::mcp::make_test_resolved_config(tmp.path());
        Arc::get_mut(&mut config).unwrap().write.dedup_enabled = false;
        let harness = Harness::new(&tmp, config);

        let requests = vec![
            make_batch_req(
                "docs/first.md",
                "",
                "---\ntitle: First\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Edited\n",
                false,
            ),
            make_batch_req(
                "docs/locked.md",
                "",
                "---\ntitle: Locked\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Edited\n",
                false,
            ),
        ];
        let err = write_documents_batch(&harness.deps(), &requests, None)
            .await
            .expect_err("a failed write must fail the batch");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o644)).unwrap();

        match err {
            BatchWriteError::Documents { failures } => {
                assert_eq!(failures.len(), 1, "{failures:?}");
                assert_eq!(failures[0].0, "docs/locked.md");
                match &failures[0].1 {
                    WriteError::Io { msg } => assert!(
                        msg.contains("'docs/locked.md'") && msg.contains("partial content"),
                        "the error must name the document the rollback could not restore: {msg}"
                    ),
                    other => panic!("expected Io, got {other:?}"),
                }
            }
            other => panic!("expected Documents, got {other:?}"),
        }
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("docs/first.md")).unwrap(),
            first_original,
            "the earlier edit is rolled back"
        );
        assert_eq!(std::fs::read_to_string(&locked).unwrap(), locked_original);
    }

    /// The failing edit's own entry is restored from its snapshot (here, a file
    /// its write left truncated), a path whose restore fails is returned, and
    /// `note_unrestored` names exactly those paths in the batch's error.
    #[tokio::test]
    async fn rollback_batch_filesystem_writes_restores_a_failed_edit_and_names_the_rest() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::write(root.join("cut.md"), "the first half of the new con").unwrap();
        // A regular file where the rollback needs a directory: its write fails
        // whatever the process's privileges.
        std::fs::write(root.join("blocked"), "not a directory").unwrap();
        let snapshots: HashMap<&str, String> = HashMap::from([
            ("cut.md", "the original\n".to_string()),
            ("blocked/doc.md", "unreachable\n".to_string()),
        ]);
        let written = vec![
            ("cut.md".to_string(), false),
            ("blocked/doc.md".to_string(), false),
        ];

        let unrestored = rollback_batch_filesystem_writes(root, &written, &snapshots).await;
        assert_eq!(
            std::fs::read_to_string(root.join("cut.md")).unwrap(),
            "the original\n"
        );
        assert_eq!(unrestored, vec!["blocked/doc.md".to_string()]);

        let err = WriteError::Io {
            msg: "Failed to write file: disk full".to_string(),
        };
        match note_unrestored(err, &unrestored) {
            WriteError::Io { msg } => assert_eq!(
                msg,
                "Failed to write file: disk full. Undoing the batch failed for \
                 'blocked/doc.md', which may hold partial content"
            ),
            other => panic!("expected Io, got {other:?}"),
        }
        assert!(matches!(
            note_unrestored(WriteError::AlreadyExists, &[]),
            WriteError::AlreadyExists
        ));
    }

    // -----------------------------------------------------------------------
    // move_directory: atomic directory move
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn move_directory_relocates_every_document_in_one_commit() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old-dir/sub")).unwrap();
        let a = "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# A\n";
        let b = "---\ntitle: B\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# B\n";
        let c = "---\ntitle: C\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# C\n";
        std::fs::write(work.path().join("old-dir/a.md"), a).unwrap();
        std::fs::write(work.path().join("old-dir/b.md"), b).unwrap();
        std::fs::write(work.path().join("old-dir/sub/c.md"), c).unwrap();
        git_commit_paths(
            &work,
            &["old-dir/a.md", "old-dir/b.md", "old-dir/sub/c.md"],
            "add old-dir",
        );
        let head_before = head_sha(&work);

        let harness = git_backed_harness(&work);
        let success = move_directory(&harness.deps(), "old-dir", "new-dir-test1", None)
            .await
            .unwrap();

        assert_eq!(success.moved.len(), 3);
        assert_ne!(
            head_sha(&work),
            head_before,
            "the move must produce a new commit"
        );
        assert!(
            !work.path().join("old-dir").exists(),
            "the old prefix must be gone entirely"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("new-dir-test1/a.md")).unwrap(),
            a
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("new-dir-test1/b.md")).unwrap(),
            b
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("new-dir-test1/sub/c.md")).unwrap(),
            c
        );
        // Every document plus both prefixes' worth of filesystem changes landed
        // in the single commit — nothing left staged or dangling.
        assert_eq!(git_status(&work), "");
    }

    #[tokio::test]
    async fn move_directory_preserves_a_link_between_two_documents_inside_the_moved_subtree() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old2/sub1")).unwrap();
        std::fs::create_dir_all(work.path().join("old2/sub2")).unwrap();
        let a = "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n\
                 See [b](../sub2/b.md) for more.\n";
        let b = "---\ntitle: B\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# B\n";
        std::fs::write(work.path().join("old2/sub1/a.md"), a).unwrap();
        std::fs::write(work.path().join("old2/sub2/b.md"), b).unwrap();
        git_commit_paths(&work, &["old2/sub1/a.md", "old2/sub2/b.md"], "add old2");

        let harness = git_backed_harness(&work);
        let success = move_directory(&harness.deps(), "old2", "moved2/deep/target", None)
            .await
            .unwrap();
        assert_eq!(success.moved.len(), 2);

        // Assert on the RESOLVED target, not the link text: a correct
        // lockstep-move rewrite usually reproduces identical text (both
        // documents moved by the same prefix change), so the meaningful check
        // is that the link still resolves to the MOVED copy of `b.md`, not a
        // stale path under the old, now-nonexistent `old2/`.
        let a_after =
            std::fs::read_to_string(work.path().join("moved2/deep/target/sub1/a.md")).unwrap();
        let occurrences =
            crate::ingest::find_markdown_link_occurrences(&a_after, "moved2/deep/target/sub1/a.md");
        assert_eq!(occurrences.len(), 1, "expected exactly one link occurrence");
        assert_eq!(occurrences[0].resolved, "moved2/deep/target/sub2/b.md");
    }

    #[tokio::test]
    async fn move_directory_preserves_a_link_from_inside_the_subtree_to_a_document_outside_it() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old3")).unwrap();
        std::fs::create_dir_all(work.path().join("shared")).unwrap();
        let a = "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n\
                 See [shared](../shared/target.md) for more.\n";
        let target =
            "---\ntitle: Target\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Target\n";
        std::fs::write(work.path().join("old3/a.md"), a).unwrap();
        std::fs::write(work.path().join("shared/target.md"), target).unwrap();
        git_commit_paths(
            &work,
            &["old3/a.md", "shared/target.md"],
            "add old3 and shared",
        );

        let harness = git_backed_harness(&work);
        // Move to a destination several levels DEEPER than the source — a naive
        // "leave outbound links untouched" implementation would break this link,
        // since reaching the untouched `shared/target.md` from the new, deeper
        // location requires more `../` climbs than the original text has.
        let success = move_directory(&harness.deps(), "old3", "moved3/deeper/still/here", None)
            .await
            .unwrap();
        assert_eq!(success.moved.len(), 1);

        let a_after =
            std::fs::read_to_string(work.path().join("moved3/deeper/still/here/a.md")).unwrap();
        let occurrences = crate::ingest::find_markdown_link_occurrences(
            &a_after,
            "moved3/deeper/still/here/a.md",
        );
        assert_eq!(occurrences.len(), 1, "expected exactly one link occurrence");
        assert_eq!(
            occurrences[0].resolved, "shared/target.md",
            "the link must still resolve to the same, unmoved outside document"
        );
        assert!(
            work.path().join("shared/target.md").exists(),
            "the outside document must never have moved"
        );
    }

    #[tokio::test]
    async fn move_directory_rewrites_an_outside_referencing_documents_link() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old4")).unwrap();
        let a = "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# A\n";
        std::fs::write(work.path().join("old4/a.md"), a).unwrap();
        git_commit_paths(&work, &["old4/a.md"], "add old4/a.md");

        let referencing = "---\ntitle: Referencer\ndescription: d\ntype: guide\ntags: [t]\n---\n\n\
                            See [a](old4/a.md) for more.\n";
        std::fs::write(work.path().join("referencing4.md"), referencing).unwrap();
        git_commit_paths(&work, &["referencing4.md"], "add referencing4.md");

        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencing4.md",
                "markdown",
                &[("old4/a.md".to_string(), None)],
            )
            .await
            .unwrap();

        let success = move_directory(&harness.deps(), "old4", "new4", None)
            .await
            .unwrap();
        assert_eq!(success.moved.len(), 1);
        assert_eq!(success.rewritten_paths, vec!["referencing4.md".to_string()]);

        let ref_after = std::fs::read_to_string(work.path().join("referencing4.md")).unwrap();
        let occurrences =
            crate::ingest::find_markdown_link_occurrences(&ref_after, "referencing4.md");
        assert_eq!(occurrences.len(), 1, "expected exactly one link occurrence");
        assert_eq!(
            occurrences[0].resolved, "new4/a.md",
            "the outside document's link must resolve to the moved document's new location"
        );
    }

    #[tokio::test]
    async fn move_directory_validation_failure_for_one_document_aborts_the_whole_move() {
        let tmp = tempfile::tempdir().unwrap();
        let source_dir = tmp.path().join("loose5");
        std::fs::create_dir_all(&source_dir).unwrap();
        // Valid under the (schema-less) source directory, but missing a field the
        // destination's schema requires.
        let a = "---\ntitle: A\n---\n# A";
        let b = "---\ntitle: B\n---\n# B";
        std::fs::write(source_dir.join("a.md"), a).unwrap();
        std::fs::write(source_dir.join("b.md"), b).unwrap();

        // The schema lives on the DESTINATION'S PARENT, not literally inside the
        // (as-yet-nonexistent, and so guard-2-empty) destination prefix itself —
        // `strict5/target` inherits it via the normal cascade.
        let dest_parent = tmp.path().join("strict5");
        std::fs::create_dir_all(&dest_parent).unwrap();
        std::fs::write(
            dest_parent.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  strict_field:\n    required: true\n",
        )
        .unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let err = move_directory(&harness.deps(), "loose5", "strict5/target", None)
            .await
            .unwrap_err();
        match err {
            DirectoryMoveError::Validation {
                failures,
                moved_schema_files,
            } => {
                assert_eq!(
                    failures.len(),
                    2,
                    "both documents are missing strict_field, expected {:?}",
                    failures
                );
                assert!(
                    moved_schema_files.is_empty(),
                    "no schema file is moving in this scenario"
                );
            }
            other => panic!("expected Validation, got {other:?}"),
        }
        assert!(
            source_dir.join("a.md").exists(),
            "nothing must be mutated when even one document fails validation"
        );
        assert!(source_dir.join("b.md").exists());
        assert!(!tmp.path().join("strict5/target").exists());
    }

    // -- move_directory: schema files travel with the subtree ---------------
    //
    // These replace the old, stricter behavior (a source subtree containing its
    // own `.kb-schema.yaml` was rejected outright — see git history for
    // `move_directory_with_a_schema_file_in_the_source_subtree_is_rejected`).
    // Lifting that restriction is the whole point of this suite: a schema file
    // now moves along with the documents it governs, re-parenting its cascade
    // onto the destination — see `SchemaCache::with_remapped_scopes`.

    #[tokio::test]
    async fn move_directory_relocates_a_subtree_including_its_own_schema_file() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let source_dir = work.path().join("old6/sub");
        std::fs::create_dir_all(&source_dir).unwrap();
        let content = "---\ntitle: A\n---\n# A";
        std::fs::write(source_dir.join("a.md"), content).unwrap();
        std::fs::write(
            source_dir.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  title:\n    required: true\n",
        )
        .unwrap();
        git_commit_paths(
            &work,
            &[
                "old6/sub/a.md",
                &format!("old6/sub/{}", crate::schema::SCHEMA_FILE_NAME),
            ],
            "add old6",
        );

        let harness = git_backed_harness(&work);

        let success = move_directory(&harness.deps(), "old6", "new6", None)
            .await
            .unwrap();

        assert!(
            !work.path().join("old6").exists(),
            "the whole old prefix, schema file included, must be gone"
        );
        assert!(work.path().join("new6/sub/a.md").exists());
        let moved_schema = work
            .path()
            .join("new6/sub")
            .join(crate::schema::SCHEMA_FILE_NAME);
        assert!(
            moved_schema.exists(),
            "the schema file must have moved along with the document it governs"
        );
        assert!(
            success
                .moved
                .contains(&("old6/sub/a.md".to_string(), "new6/sub/a.md".to_string())),
        );
        assert!(
            success.moved.contains(&(
                format!("old6/sub/{}", crate::schema::SCHEMA_FILE_NAME),
                format!("new6/sub/{}", crate::schema::SCHEMA_FILE_NAME),
            )),
            "the schema file's own relocation must be reported in `moved` too: {:?}",
            success.moved
        );
        assert_eq!(git_status(&work), "");

        // Post-move: rebuilding a real cache off disk must agree with what the
        // move validated against — proving the prediction was right, not merely
        // self-consistent.
        let rebuilt = crate::schema::SchemaCache::build_for_test(
            &work.path().canonicalize().unwrap(),
            &crate::config::FrontmatterConfig::default(),
        );
        assert!(
            rebuilt
                .resolve_for(Path::new("new6/sub/a.md"))
                .fields
                .get("title")
                .is_some_and(|f| f.required),
            "the rebuilt cache must show the relocated schema's rule in effect"
        );
    }

    #[tokio::test]
    async fn move_directory_schema_file_travels_into_a_stricter_destination_is_rejected() {
        // The crux case the old guard existed to prevent: the subtree's OWN
        // schema file travels with it, but the destination's ancestor declares
        // an ADDITIONAL required field the source's ancestor never did. Nothing
        // in the moved subtree satisfies it, so the move must fail — and fail
        // for exactly this reason, not some other validation quirk.
        let tmp = tempfile::tempdir().unwrap();
        let source_dir = tmp.path().join("src7/sub");
        std::fs::create_dir_all(&source_dir).unwrap();
        let content = "---\ntitle: A\n---\n# A";
        std::fs::write(source_dir.join("a.md"), content).unwrap();
        std::fs::write(
            source_dir.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  title:\n    required: true\n",
        )
        .unwrap();

        // The destination's PARENT declares a field the source's parent never
        // required.
        let dest_parent = tmp.path().join("dest7");
        std::fs::create_dir_all(&dest_parent).unwrap();
        std::fs::write(
            dest_parent.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  extra_required:\n    required: true\n",
        )
        .unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let err = move_directory(&harness.deps(), "src7", "dest7/target", None)
            .await
            .unwrap_err();
        match err {
            DirectoryMoveError::Validation {
                failures,
                moved_schema_files,
            } => {
                assert_eq!(failures.len(), 1);
                assert_eq!(failures[0].0, "dest7/target/sub/a.md");
                assert!(
                    failures[0]
                        .1
                        .errors
                        .iter()
                        .any(|e| e.contains("extra_required")),
                    "must name the field the destination newly requires: {:?}",
                    failures[0].1.errors
                );
                assert_eq!(
                    moved_schema_files,
                    vec![(
                        format!("src7/sub/{}", crate::schema::SCHEMA_FILE_NAME),
                        format!("dest7/target/sub/{}", crate::schema::SCHEMA_FILE_NAME),
                    )],
                    "the error must name the schema file that is relocating"
                );
            }
            other => panic!("expected Validation, got {other:?}"),
        }

        assert!(
            source_dir.join("a.md").exists(),
            "nothing must be mutated on a rejected move"
        );
        assert!(source_dir.join(crate::schema::SCHEMA_FILE_NAME).exists());
        assert!(!tmp.path().join("dest7/target").exists());
    }

    #[tokio::test]
    async fn move_directory_schema_file_travels_into_a_more_permissive_destination_succeeds() {
        // The mirror of the crux case: the destination's ancestor is more
        // permissive than the source's was, so documents that were valid stay
        // valid.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let source_parent = work.path().join("src8");
        std::fs::create_dir_all(&source_parent).unwrap();
        std::fs::write(
            source_parent.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  status:\n    required: true\n",
        )
        .unwrap();
        let source_dir = source_parent.join("sub");
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::write(
            source_dir.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  status:\n    type: enum\n    values: [draft, active]\n",
        )
        .unwrap();
        std::fs::write(source_dir.join("a.md"), "---\nstatus: draft\n---\n# A").unwrap();
        git_commit_paths(
            &work,
            &[
                &format!("src8/{}", crate::schema::SCHEMA_FILE_NAME),
                &format!("src8/sub/{}", crate::schema::SCHEMA_FILE_NAME),
                "src8/sub/a.md",
            ],
            "add src8",
        );

        // Destination has NO ancestor schema at all — strictly more permissive
        // than the source's parent, which required `status`.
        let harness = git_backed_harness(&work);

        let success = move_directory(&harness.deps(), "src8/sub", "dest8/sub", None)
            .await
            .unwrap();
        assert_eq!(success.moved.len(), 2, "the document and its schema file");
        assert!(work.path().join("dest8/sub/a.md").exists());
        assert_eq!(git_status(&work), "");
    }

    /// A legacy-named schema moves under its own name; a source directory holding
    /// both names is refused like any other invalid schema file.
    #[tokio::test]
    async fn move_directory_carries_a_legacy_schema_and_refuses_both_names() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let source = work.path().join("src12");
        std::fs::create_dir_all(&source).unwrap();
        let legacy_rel = format!("src12/{}", crate::schema::LEGACY_SCHEMA_FILE_NAME);
        let canonical_rel = format!("src12/{}", crate::schema::SCHEMA_FILE_NAME);
        let schema = "fields:\n  status:\n    required: true\n";
        std::fs::write(work.path().join(&legacy_rel), schema).unwrap();
        std::fs::write(source.join("a.md"), "---\nstatus: draft\n---\n# A").unwrap();
        git_commit_paths(&work, &[&legacy_rel, "src12/a.md"], "add src12");

        let mut config = crate::mcp::make_test_resolved_config(work.path());
        Arc::get_mut(&mut config).unwrap().write.dedup_enabled = false;
        let harness = Harness::new(&work, config);

        std::fs::write(work.path().join(&canonical_rel), schema).unwrap();
        let err = move_directory(&harness.deps(), "src12", "dest12", None)
            .await
            .expect_err("both names in one directory must block the move");
        match err {
            DirectoryMoveError::InvalidSchemaInSource { reason, .. } => {
                assert_eq!(reason, crate::schema::BOTH_NAMES_REASON);
            }
            other => panic!("expected InvalidSchemaInSource, got {other:?}"),
        }
        assert!(!work.path().join("dest12").exists(), "nothing moved");

        std::fs::remove_file(work.path().join(&canonical_rel)).unwrap();
        move_directory(&harness.deps(), "src12", "dest12", None)
            .await
            .expect("a legacy-only schema moves with its directory");
        assert!(
            work.path()
                .join("dest12")
                .join(crate::schema::LEGACY_SCHEMA_FILE_NAME)
                .exists()
        );
    }

    /// #272: the shared cache keeps the last good schema while a file on disk is
    /// invalid, so the move re-reads every schema file it would carry and refuses
    /// on an invalid one — nothing moves. Once the file is valid on disk again the
    /// same move goes through. A broken schema in a directory `indexing.exclude`
    /// rules out entirely is not part of the schema tree and does not block.
    #[tokio::test]
    async fn move_directory_refuses_a_source_schema_file_that_is_invalid_on_disk() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let source = work.path().join("src10");
        std::fs::create_dir_all(source.join("templates")).unwrap();
        let schema_rel = format!("src10/{}", crate::schema::SCHEMA_FILE_NAME);
        let valid = "fields:\n  status:\n    required: true\n";
        std::fs::write(work.path().join(&schema_rel), valid).unwrap();
        let ignored_rel = format!("src10/templates/{}", crate::schema::SCHEMA_FILE_NAME);
        std::fs::write(work.path().join(&ignored_rel), "fields: [broken\n").unwrap();
        std::fs::write(source.join("a.md"), "---\nstatus: draft\n---\n# A").unwrap();
        git_commit_paths(
            &work,
            &[&schema_rel, &ignored_rel, "src10/a.md"],
            "add src10",
        );

        let mut config = crate::mcp::make_test_resolved_config(work.path());
        {
            let cfg = Arc::get_mut(&mut config).unwrap();
            cfg.write.dedup_enabled = false;
            cfg.indexing.exclude.push("**/templates/**".into());
        }
        // Built while the schema is valid — the last good cache from here on.
        let harness = Harness::new(&work, config);

        std::fs::write(
            work.path().join(&schema_rel),
            "fields: [this is not a map\n",
        )
        .unwrap();
        let err = move_directory(&harness.deps(), "src10", "dest10", None)
            .await
            .expect_err("an invalid schema file on disk must block the move");
        match err {
            DirectoryMoveError::InvalidSchemaInSource { path, reason } => {
                assert_eq!(path, schema_rel);
                assert!(!reason.is_empty());
            }
            other => panic!("expected InvalidSchemaInSource, got {other:?}"),
        }
        assert!(source.join("a.md").exists());
        assert!(!work.path().join("dest10").exists(), "nothing moved");

        std::fs::write(work.path().join(&schema_rel), valid).unwrap();
        let success = move_directory(&harness.deps(), "src10", "dest10", None)
            .await
            .unwrap();
        assert_eq!(success.moved.len(), 3, "{:?}", success.moved);
        assert!(work.path().join("dest10/a.md").exists());
        assert_eq!(git_status(&work), "");
    }

    /// Commits `source/a.md` with an invalid schema file beside it, `source`
    /// being outside the schema tree (`exclude` added to `indexing.exclude`, or
    /// a hidden directory), and moves it to `dest`, inside the schema tree. No
    /// cache ever read that schema file, so the move checks it where it would
    /// land: refused with nothing moved while it is invalid or sits beside the
    /// legacy name, moved once it is valid.
    async fn assert_a_schema_file_entering_the_schema_tree_is_checked(
        source: &str,
        dest: &str,
        exclude: Option<&str>,
    ) {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join(source)).unwrap();
        let doc_rel = format!("{source}/a.md");
        let schema_rel = format!("{source}/{}", crate::schema::SCHEMA_FILE_NAME);
        std::fs::write(work.path().join(&doc_rel), "---\ntitle: A\n---\n# A").unwrap();
        std::fs::write(work.path().join(&schema_rel), "fields: [broken\n").unwrap();
        git_commit_paths(&work, &[&doc_rel, &schema_rel], "add the source");

        let mut config = crate::mcp::make_test_resolved_config(work.path());
        {
            let cfg = Arc::get_mut(&mut config).unwrap();
            cfg.write.dedup_enabled = false;
            cfg.indexing.exclude.extend(exclude.map(str::to_string));
        }
        let harness = Harness::new(&work, config);

        let err = move_directory(&harness.deps(), source, dest, None)
            .await
            .expect_err("an invalid schema file entering the schema tree must block the move");
        match err {
            DirectoryMoveError::InvalidSchemaInSource { path, reason } => {
                assert_eq!(path, schema_rel);
                assert!(!reason.is_empty());
            }
            other => panic!("expected InvalidSchemaInSource, got {other:?}"),
        }
        assert!(work.path().join(&doc_rel).exists());
        assert!(!work.path().join(dest).exists(), "nothing moved");

        let valid = "fields:\n  title:\n    required: true\n";
        let legacy_rel = format!("{source}/{}", crate::schema::LEGACY_SCHEMA_FILE_NAME);
        std::fs::write(work.path().join(&schema_rel), valid).unwrap();
        std::fs::write(work.path().join(&legacy_rel), valid).unwrap();
        let err = move_directory(&harness.deps(), source, dest, None)
            .await
            .expect_err("both schema file names in one directory must block the move");
        match err {
            DirectoryMoveError::InvalidSchemaInSource { reason, .. } => {
                assert_eq!(reason, crate::schema::BOTH_NAMES_REASON);
            }
            other => panic!("expected InvalidSchemaInSource, got {other:?}"),
        }
        assert!(!work.path().join(dest).exists(), "nothing moved");

        std::fs::remove_file(work.path().join(&legacy_rel)).unwrap();
        let success = move_directory(&harness.deps(), source, dest, None)
            .await
            .expect("a valid schema file moves with its directory");
        assert_eq!(success.moved.len(), 2, "{:?}", success.moved);
        assert!(
            work.path()
                .join(dest)
                .join(crate::schema::SCHEMA_FILE_NAME)
                .exists()
        );
        assert!(work.path().join(dest).join("a.md").exists());
        assert_eq!(git_status(&work), "");
    }

    #[tokio::test]
    async fn move_directory_checks_a_schema_file_moving_out_of_a_hidden_directory() {
        assert_a_schema_file_entering_the_schema_tree_is_checked(
            ".templates/proj",
            "notes/proj",
            None,
        )
        .await;
    }

    #[tokio::test]
    async fn move_directory_checks_a_schema_file_moving_out_of_an_excluded_directory() {
        assert_a_schema_file_entering_the_schema_tree_is_checked(
            "drafts/proj",
            "notes/proj",
            Some("drafts/**"),
        )
        .await;
    }

    /// A `./` or `/` spelling of a directory moves it as if written plainly, and
    /// a `./` destination reads the same way, so no moved path or path marked
    /// dirty carries `./`. The root itself, however spelled, is refused with an
    /// error rather than walked.
    #[tokio::test]
    async fn move_directory_normalizes_dot_and_slash_spellings_and_refuses_the_root() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("notes")).unwrap();
        std::fs::write(work.path().join("notes/a.md"), "---\ntitle: A\n---\n# A").unwrap();
        git_commit_paths(&work, &["notes/a.md"], "add notes");
        let harness = git_backed_harness(&work);

        for root in [".", "./", "/", "//", "/."] {
            let err = move_directory(&harness.deps(), root, "elsewhere", None)
                .await
                .expect_err(root);
            assert!(
                matches!(err, DirectoryMoveError::UnsafePath { .. }),
                "{root}: {err:?}"
            );
        }
        assert!(work.path().join("notes/a.md").exists());
        assert!(!work.path().join("elsewhere").exists());

        for (source, dest, from, to) in [
            ("./notes", "./x1", "notes/a.md", "x1/a.md"),
            ("./x1/", "x2", "x1/a.md", "x2/a.md"),
            ("/./x2", "/./x3/", "x2/a.md", "x3/a.md"),
        ] {
            let success = move_directory(&harness.deps(), source, dest, None)
                .await
                .unwrap_or_else(|e| panic!("{source} -> {dest}: {e:?}"));
            assert_eq!(success.moved, vec![(from.to_string(), to.to_string())]);
        }
        assert!(work.path().join("x3/a.md").exists());
        assert_eq!(git_status(&work), "");
        let marked = harness.reindex_queue.snapshot_paths();
        assert!(
            marked.iter().all(|p| !p.to_string_lossy().contains("./")),
            "{marked:?}"
        );
        assert!(marked.contains(Path::new("x3/a.md")), "{marked:?}");
    }

    /// A widened `indexing.include` (`**/*`) admits `.kb-schema.yaml` as a path, but
    /// the move carries it through `schema_moves` only: it is neither validated as a
    /// document nor listed a second time.
    #[tokio::test]
    async fn move_directory_carries_a_schema_file_once_with_a_widened_include() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let source = work.path().join("src11");
        std::fs::create_dir_all(&source).unwrap();
        let schema_rel = format!("src11/{}", crate::schema::SCHEMA_FILE_NAME);
        std::fs::write(
            work.path().join(&schema_rel),
            "fields:\n  status:\n    required: true\n",
        )
        .unwrap();
        std::fs::write(source.join("a.md"), "---\nstatus: draft\n---\n# A").unwrap();
        git_commit_paths(&work, &[&schema_rel, "src11/a.md"], "add src11");

        let mut config = crate::mcp::make_test_resolved_config(work.path());
        Arc::get_mut(&mut config).unwrap().write.dedup_enabled = false;
        let mut harness = Harness::new(&work, config);
        let mut builder = globset::GlobSetBuilder::new();
        builder.add(globset::Glob::new("**/*").unwrap());
        harness.include_patterns = builder.build().unwrap();

        let success = move_directory(&harness.deps(), "src11", "dest11", None)
            .await
            .unwrap();
        let dest_schema = format!("dest11/{}", crate::schema::SCHEMA_FILE_NAME);
        let schema_arrivals = success
            .moved
            .iter()
            .filter(|m| format!("{m:?}").contains(crate::schema::SCHEMA_FILE_NAME))
            .count();
        assert_eq!(schema_arrivals, 1, "{:?}", success.moved);
        assert_eq!(success.moved.len(), 2, "{:?}", success.moved);
        assert!(work.path().join(&dest_schema).exists());
        assert!(work.path().join("dest11/a.md").exists());
        assert!(!source.exists(), "source is gone");
        assert_eq!(git_status(&work), "");
    }

    #[tokio::test]
    async fn move_directory_relocates_multiple_schema_files_at_different_depths() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let root = work.path().join("src9");
        std::fs::create_dir_all(root.join("mid/deep")).unwrap();
        std::fs::write(
            root.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  top:\n    required: true\n",
        )
        .unwrap();
        std::fs::write(
            root.join("mid").join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  mid_field:\n    required: true\n",
        )
        .unwrap();
        std::fs::write(
            root.join("mid/deep").join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  deep_field:\n    required: true\n",
        )
        .unwrap();
        std::fs::write(
            root.join("mid/deep/doc.md"),
            "---\ntop: t\nmid_field: m\ndeep_field: d\n---\n# D",
        )
        .unwrap();
        git_commit_paths(
            &work,
            &[
                &format!("src9/{}", crate::schema::SCHEMA_FILE_NAME),
                &format!("src9/mid/{}", crate::schema::SCHEMA_FILE_NAME),
                &format!("src9/mid/deep/{}", crate::schema::SCHEMA_FILE_NAME),
                "src9/mid/deep/doc.md",
            ],
            "add src9",
        );

        let harness = git_backed_harness(&work);

        let success = move_directory(&harness.deps(), "src9", "dest9", None)
            .await
            .unwrap();
        // 1 document + 3 schema files, at 3 different depths.
        assert_eq!(success.moved.len(), 4, "{:?}", success.moved);
        for suffix in ["".to_string(), "mid/".to_string(), "mid/deep/".to_string()] {
            assert!(
                work.path()
                    .join(format!(
                        "dest9/{}{}",
                        suffix,
                        crate::schema::SCHEMA_FILE_NAME
                    ))
                    .exists(),
                "schema file at depth '{}' must have relocated",
                suffix
            );
        }
        assert!(work.path().join("dest9/mid/deep/doc.md").exists());
        assert_eq!(git_status(&work), "");

        let rebuilt = crate::schema::SchemaCache::build_for_test(
            &work.path().canonicalize().unwrap(),
            &crate::config::FrontmatterConfig::default(),
        );
        let resolved = rebuilt.resolve_for(Path::new("dest9/mid/deep/doc.md"));
        assert!(resolved.fields["top"].required);
        assert!(resolved.fields["mid_field"].required);
        assert!(resolved.fields["deep_field"].required);
    }

    #[tokio::test]
    async fn move_directory_values_splicing_field_resolves_differently_under_destination_parent() {
        // A `$values`-splicing field must resolve against the DESTINATION
        // parent's set, not the source's — proving the remapped cache re-runs
        // the splice rather than reusing whatever the live cache had cached.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let source_parent = work.path().join("src10");
        std::fs::create_dir_all(&source_parent).unwrap();
        std::fs::write(
            source_parent.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  tags:\n    values: [source_tag]\n",
        )
        .unwrap();
        let source_dir = source_parent.join("sub");
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::write(
            source_dir.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  tags:\n    values: [$values, own_tag]\n",
        )
        .unwrap();
        std::fs::write(source_dir.join("a.md"), "---\ntags: [own_tag]\n---\n# A").unwrap();

        let dest_parent = work.path().join("dest10");
        std::fs::create_dir_all(&dest_parent).unwrap();
        std::fs::write(
            dest_parent.join(crate::schema::SCHEMA_FILE_NAME),
            "fields:\n  tags:\n    values: [dest_tag]\n",
        )
        .unwrap();
        git_commit_paths(
            &work,
            &[
                &format!("src10/{}", crate::schema::SCHEMA_FILE_NAME),
                &format!("src10/sub/{}", crate::schema::SCHEMA_FILE_NAME),
                "src10/sub/a.md",
                &format!("dest10/{}", crate::schema::SCHEMA_FILE_NAME),
            ],
            "add src10 and dest10",
        );

        let harness = git_backed_harness(&work);

        // `source_tag` is no longer permitted post-move (the source's parent
        // set is gone), but the document only ever used `own_tag`, which the
        // moved schema's own splice still contributes — so it stays valid.
        let success = move_directory(&harness.deps(), "src10/sub", "dest10/sub", None)
            .await
            .unwrap();
        assert_eq!(success.moved.len(), 2);
        assert_eq!(git_status(&work), "");
        assert!(
            harness.reindex_queue.snapshot().full_pending,
            "a moved schema file queues a full reconcile, which rebuilds the shared cache"
        );

        let rebuilt = crate::schema::SchemaCache::build_for_test(
            &work.path().canonicalize().unwrap(),
            &crate::config::FrontmatterConfig::default(),
        );
        let resolved = rebuilt.resolve_for(Path::new("dest10/sub/a.md"));
        assert_eq!(
            resolved.fields["tags"].values,
            Some(vec!["dest_tag".to_string(), "own_tag".to_string()]),
            "the splice must resolve against the DESTINATION parent's set"
        );
    }

    #[tokio::test]
    async fn move_directory_to_an_occupied_destination_prefix_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let source_dir = tmp.path().join("old7");
        std::fs::create_dir_all(&source_dir).unwrap();
        let content = "---\ntitle: A\n---\n# A";
        std::fs::write(source_dir.join("a.md"), content).unwrap();

        let dest_dir = tmp.path().join("occupied7");
        std::fs::create_dir_all(&dest_dir).unwrap();
        std::fs::write(dest_dir.join("already-here.md"), "# Already here").unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let err = move_directory(&harness.deps(), "old7", "occupied7", None)
            .await
            .unwrap_err();
        assert!(
            matches!(err, DirectoryMoveError::AlreadyExists),
            "got {err:?}"
        );
        assert!(
            source_dir.join("a.md").exists(),
            "source must be untouched when the destination prefix is occupied"
        );
        assert_eq!(
            std::fs::read_to_string(dest_dir.join("already-here.md")).unwrap(),
            "# Already here",
            "the pre-existing destination content must be untouched"
        );
    }

    #[tokio::test]
    async fn move_directory_toctou_destination_collision_reports_already_exists_not_io() {
        // A collision that appears AFTER the batch pre-check (guard 2) and the
        // per-document defensive re-check — caught by the re-check under
        // `GIT_LOCK`, or failing that by phase 1's `create_new` — must surface as
        // `AlreadyExists` (a
        // benign, retryable race), not the generic `Io` arm. `Io` maps to
        // `McpError::internal_error`, which would misreport a completely
        // ordinary race as a server fault.
        //
        // Reproduced deterministically rather than via a hopeful thread race:
        // a configured `lint_command` makes `validate::validate_content`
        // `.await` a real subprocess (`sleep 0.3`) for the one document in
        // this move, which happens AFTER that document's own per-document
        // `exists()` check but BEFORE phase 1 ever runs. `tokio::join!` runs
        // that subprocess wait concurrently (same task, cooperative
        // scheduling) with a second future that creates the destination file
        // partway through the sleep -- landing squarely in the TOCTOU window
        // this fix closes. This also exercises `rollback_directory_move_filesystem`
        // for real with the lock-hoist fix in place: if that helper still
        // tried to acquire its own `GitLock` (the pre-fix behavior) instead of
        // reusing the one `move_directory` already holds by this point, this
        // test would hang until the outer `timeout` below fails it.
        let tmp = tempfile::tempdir().unwrap();
        let source_dir = tmp.path().join("old-toctou");
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::write(source_dir.join("a.md"), "---\ntitle: A\n---\n# A").unwrap();

        let mut config = crate::mcp::make_test_resolved_config(tmp.path());
        Arc::get_mut(&mut config).unwrap().validation.lint_command =
            Some(vec!["sh".into(), "-c".into(), "sleep 0.3".into()]);
        let harness = Harness::new(&tmp, config);

        let dest_dir = tmp.path().join("new-toctou");
        let dest_file = dest_dir.join("a.md");
        let collision_content = "collision, landed after the pre-check";

        let deps = harness.deps();
        let (move_result, _) = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            tokio::join!(
                move_directory(&deps, "old-toctou", "new-toctou", None),
                async {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    std::fs::create_dir_all(&dest_dir).unwrap();
                    std::fs::write(&dest_file, collision_content).unwrap();
                },
            )
        })
        .await
        .expect("move_directory must not hang");

        let err = move_result.unwrap_err();
        assert!(
            matches!(err, DirectoryMoveError::AlreadyExists),
            "a destination collision appearing after the pre-check must map to \
             AlreadyExists, not a generic Io error; got {err:?}"
        );
        assert!(
            source_dir.join("a.md").exists(),
            "the source must be untouched -- phase 1 never got far enough to remove it"
        );
        assert_eq!(
            std::fs::read_to_string(&dest_file).unwrap(),
            collision_content,
            "move_directory must never have touched the colliding file (create_new can't \
             overwrite it, and rollback only removes what IT wrote)"
        );
    }

    #[tokio::test]
    async fn move_directory_blocks_while_git_lock_is_externally_held_then_completes() {
        // GIT_LOCK hoist regression (data-loss finding fix): `move_directory`
        // must acquire `GIT_LOCK` before phase 1 (destination writes), not
        // just before the commit, and hold that ONE guard across phases 1-3,
        // the commit, and any rollback. Proven at runtime: another holder of
        // `GIT_LOCK` must observably block the call, and releasing that
        // holder must let it proceed to completion without hanging -- a hang
        // would mean something reachable from `move_directory` (e.g.
        // `rollback_directory_move_filesystem`) is trying to reacquire the
        // already-held, non-reentrant mutex instead of reusing it.
        let tmp = tempfile::tempdir().unwrap();
        let source_dir = tmp.path().join("old-lockcheck");
        std::fs::create_dir_all(&source_dir).unwrap();
        std::fs::write(source_dir.join("a.md"), "---\ntitle: A\n---\n# A").unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let held = git::lock_git().await;

        let deps = harness.deps();
        let move_fut = move_directory(&deps, "old-lockcheck", "new-lockcheck", None);
        tokio::pin!(move_fut);
        let still_blocked =
            tokio::time::timeout(std::time::Duration::from_millis(200), &mut move_fut).await;
        assert!(
            still_blocked.is_err(),
            "move_directory must block on GIT_LOCK (acquired ahead of phase 1, per the \
             finding's fix) while another holder has it"
        );

        drop(held);
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), move_fut)
            .await
            .expect(
                "move_directory must proceed to completion once GIT_LOCK is released, not hang \
                 against its own held guard -- a hang here would mean this non-reentrant mutex \
                 is being acquired a second time somewhere in the call chain",
            );
        // Not git-backed, so the commit itself fails fast once the lock is free --
        // the point of this test is that nothing deadlocks, not the outcome.
        match result.unwrap_err() {
            DirectoryMoveError::PreCommitFailed { .. } => {}
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn move_directory_precommit_failure_rolls_back_every_document_and_referencing_document() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("old8")).unwrap();
        let a = "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# A\n";
        let b = "---\ntitle: B\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# B\n";
        std::fs::write(work.path().join("old8/a.md"), a).unwrap();
        std::fs::write(work.path().join("old8/b.md"), b).unwrap();
        git_commit_paths(&work, &["old8/a.md", "old8/b.md"], "add old8");

        let referencing = "---\ntitle: Referencer\ndescription: d\ntype: guide\ntags: [t]\n---\n\n\
                            See [a](old8/a.md) for more.\n";
        std::fs::write(work.path().join("referencing8.md"), referencing).unwrap();
        git_commit_paths(&work, &["referencing8.md"], "add referencing8.md");
        let head_before = head_sha(&work);

        force_git_commit_to_fail(&work);
        let harness = git_backed_harness_with_state_db(&work).await;
        harness
            .state_db
            .as_ref()
            .unwrap()
            .replace_links(
                "referencing8.md",
                "markdown",
                &[("old8/a.md".to_string(), None)],
            )
            .await
            .unwrap();

        let err = move_directory(&harness.deps(), "old8", "new8", None)
            .await
            .unwrap_err();
        match err {
            DirectoryMoveError::PreCommitFailed { rolled_back, .. } => assert!(rolled_back),
            other => panic!("expected PreCommitFailed, got {other:?}"),
        }

        assert_eq!(
            std::fs::read_to_string(work.path().join("old8/a.md")).unwrap(),
            a,
            "source a.md must be restored after a rolled-back directory move"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("old8/b.md")).unwrap(),
            b,
            "source b.md must be restored after a rolled-back directory move"
        );
        assert!(
            !work.path().join("new8").exists(),
            "destination prefix must be gone after a rolled-back directory move"
        );
        assert_eq!(
            std::fs::read_to_string(work.path().join("referencing8.md")).unwrap(),
            referencing,
            "the referencing document's link rewrite must be rolled back too, not just the move"
        );
        assert_eq!(head_before, head_sha(&work));
        assert_eq!(git_status(&work), "");
    }

    /// #147: `move_directory`'s own "rollback itself also failed" branch
    /// (`rolled_back: false`) — the fourth and last of the four
    /// structurally-identical sites the issue names, and like the other
    /// three, exercised only by `delete_document`'s test before this. Same
    /// no-git-repo technique: `git add` fails first (`PreCommit`), and then
    /// `git::restore_from_head` on the moved source ALSO fails (no
    /// repository to restore from), which alone is enough to flip this
    /// move's `rolled_back` to `false` — `move_directory` ORs a failure in
    /// per-source, per-destination, and per-rewritten-document, and any
    /// single one failing is enough.
    #[tokio::test]
    async fn move_directory_rollback_failure_with_no_git_repo_reports_rolled_back_false() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("old-dir-no-repo");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("a.md"), "---\ntitle: A\n---\n\n# A\n").unwrap();

        let config = crate::mcp::make_test_resolved_config(tmp.path());
        let harness = Harness::new(&tmp, config);

        let err = move_directory(&harness.deps(), "old-dir-no-repo", "new-dir-no-repo", None)
            .await
            .unwrap_err();
        match err {
            DirectoryMoveError::PreCommitFailed { rolled_back, .. } => assert!(!rolled_back),
            other => panic!("expected PreCommitFailed{{rolled_back: false}}, got {other:?}"),
        }
        // The destination copy was written, then successfully removed during
        // rollback — but the source restore can't run with no HEAD to
        // restore from, so the source is left gone too, same inconsistency
        // as the single-document move's equivalent test above.
        assert!(!tmp.path().join("new-dir-no-repo/a.md").exists());
        assert!(!sub.join("a.md").exists());
    }
}
