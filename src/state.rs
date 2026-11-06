use crate::document_fields;
use anyhow::{Context, Result};
use serde_json::Value;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
use sqlx::{QueryBuilder, Sqlite, SqlitePool};
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use tracing::{debug, warn};

/// Conservative cap on values bound into a single `WHERE file_path IN (...)` query.
/// SQLite's default limit is 999 bound parameters per statement; staying well under it
/// leaves room for drivers/wrappers that reserve a few, and keeps a single query from
/// dominating the connection while a large batch is chunked through it.
const SQLITE_MAX_PARAMS_PER_QUERY: usize = 500;

/// How a single field is matched when listing documents.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldFilter {
    /// Document has the field with any of these values. Multi-valued fields (a tag
    /// list) match if any element matches.
    AnyOf(Vec<String>),
    /// Document has the field with every one of these values — only meaningful for
    /// multi-valued fields.
    AllOf(Vec<String>),
    /// Numeric bounds. Only values that projected to a number participate.
    Range {
        gte: Option<f64>,
        lte: Option<f64>,
        gt: Option<f64>,
        lt: Option<f64>,
    },
    /// A correlated list filter (#349), the parsed form of
    /// `{"npcs[]": {"role": "guard", "level": {"gte": 3}}}`: every condition must be
    /// satisfied by the rows of ONE element of the list (one `document_fields.elem`),
    /// not merely somewhere in the document. Each entry is the projected field path
    /// of a sub-field (`npcs[].role`) and its condition, any of the other forms —
    /// never another `Elem`. Query mode lowers it to one Qdrant nested condition on
    /// the list (`qdrant::lower_field_filters`, #350).
    Elem(Vec<(String, FieldFilter)>),
}

/// The version of the rules that turn stored frontmatter into `document_fields`
/// rows (`document_fields::flatten_frontmatter`) and of that table's shape. Kept in
/// SQLite's `PRAGMA user_version`; when the stored value differs,
/// [`StateDb::ensure_projection_current`] reprojects every document from
/// `documents.frontmatter` — no markdown read, no embedding, no Qdrant write — and
/// only then records this value. Bump it whenever the projection rules change.
///
/// `1`: schema-driven projection with list-of-object rows keyed by `elem` (#349).
/// Every database written before has `user_version` 0.
pub const PROJECTION_VERSION: i64 = 1;

/// What [`StateDb::reproject_all_fields`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReprojectReport {
    /// Documents whose projection was rebuilt.
    pub reprojected: u64,
    /// Documents whose stored frontmatter is not a JSON object, left without rows.
    /// Reprojecting again cannot change that; reindexing the file does.
    pub unparseable: usize,
    /// Documents a database error kept from being reprojected. Non-zero means the
    /// projection is incomplete, so [`StateDb::ensure_projection_current`] does not
    /// record [`PROJECTION_VERSION`].
    pub failed: usize,
}

/// Ordering for a document listing. An allowlist rather than a free string, because
/// `ORDER BY` cannot be a bound parameter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OrderBy {
    #[default]
    Path,
    Title,
    Mtime,
    IndexedAt,
}

impl OrderBy {
    fn column(self) -> &'static str {
        match self {
            OrderBy::Path => "d.file_path",
            OrderBy::Title => "d.title",
            OrderBy::Mtime => "d.mtime",
            OrderBy::IndexedAt => "d.indexed_at",
        }
    }

    /// Parse a caller-supplied ordering, rejecting anything unrecognized.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "path" | "file_path" => Ok(OrderBy::Path),
            "title" => Ok(OrderBy::Title),
            "mtime" | "modified" => Ok(OrderBy::Mtime),
            "indexed_at" | "indexed" => Ok(OrderBy::IndexedAt),
            other => Err(format!(
                "unknown order_by '{}': expected one of path, title, mtime, indexed_at",
                other
            )),
        }
    }
}

/// A document listing request. Every field is optional at the tool boundary; defaults
/// are resolved before reaching here.
#[derive(Debug, Clone)]
pub struct DocumentQuery {
    /// Field path to filter, ANDed across entries.
    pub filters: Vec<(String, FieldFilter)>,
    pub path_prefix: Option<String>,
    pub order_by: OrderBy,
    pub order_desc: bool,
    pub limit: u64,
    pub offset: u64,
    /// Dot-paths to include in each result's frontmatter. `None` returns all of it.
    pub fields: Option<Vec<String>>,
    /// Exclude documents whose `mtime` is before this Unix timestamp.
    pub mtime_after: Option<i64>,
    /// Exclude documents whose `mtime` is after this Unix timestamp.
    pub mtime_before: Option<i64>,
}

impl Default for DocumentQuery {
    fn default() -> Self {
        Self {
            filters: Vec::new(),
            path_prefix: None,
            order_by: OrderBy::default(),
            order_desc: false,
            limit: 100,
            offset: 0,
            fields: None,
            mtime_after: None,
            mtime_before: None,
        }
    }
}

/// One document in a listing.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentSummary {
    pub file_path: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub mtime: i64,
    pub indexed_at: String,
    pub frontmatter: Value,
}

/// A page of documents plus the total matching the same filters, so a caller can
/// always tell whether the listing was truncated.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentQueryResult {
    pub documents: Vec<DocumentSummary>,
    pub total: u64,
}

impl DocumentQueryResult {
    pub fn has_more(&self, offset: u64) -> bool {
        offset + (self.documents.len() as u64) < self.total
    }
}

/// The paths matching a `path_prefix` needle, plus how many matched in total.
///
/// #182: `paths` is capped by the caller so a very broad needle cannot build an
/// unbounded Qdrant filter; `total` is the uncapped count, so a caller can always
/// tell a complete answer from a truncated one — the same `total`-alongside-the-page
/// convention as [`DocumentQueryResult`].
#[derive(Debug, Clone, PartialEq)]
pub struct PathMatches {
    pub paths: Vec<String>,
    pub total: u64,
}

impl PathMatches {
    /// True when the cap hid matches that really exist.
    pub fn truncated(&self) -> bool {
        (self.paths.len() as u64) < self.total
    }
}

/// Read side of the document metadata index. Separate from [`StateDb`]'s inherent API
/// so retrieval can be tested against a mock without SQLite.
pub trait DocumentIndex: Send + Sync {
    async fn query_documents(&self, query: &DocumentQuery) -> Result<DocumentQueryResult>;

    /// Every indexed path containing `needle`, capped at `cap`.
    ///
    /// #182: this is the single authority for what a `path_prefix` matches. Both
    /// search modes resolve through it — enumeration applies the same `LIKE` inline
    /// via `push_where`, query mode pushes the paths this returns down to Qdrant —
    /// so one matcher decides, rather than SQL and the vector store's `path_ancestors`
    /// index each deciding differently for the same input (issue #259).
    ///
    /// Matching is case-insensitive for ASCII, inherited from SQLite's `LIKE`.
    async fn paths_matching(&self, needle: &str, cap: u64) -> Result<PathMatches>;

    /// Every indexed document's file path, unfiltered and unordered.
    ///
    /// Backs `retrieval::get_document`'s fuzzy fallback: basename matching and
    /// "did you mean?" suggestions need the full path list, and `documents` already
    /// holds exactly that — one row per successfully indexed file — so this is a
    /// plain local `SELECT`, no cap and no round trip to the vector store required.
    async fn all_paths(&self) -> Result<Vec<String>>;

    /// Document summaries for exactly these paths, in unspecified order.
    ///
    /// Backs `retrieval::search`'s query+document (grouped) mode: Qdrant's
    /// `query_groups` collapses each document to its single best-scoring chunk, whose
    /// payload carries only that chunk's fields — not the whole-document metadata
    /// (`title`/`description`/`mtime`/full `frontmatter`) a document-shaped result
    /// needs. The caller re-associates each returned summary with its Qdrant score by
    /// `file_path` and restores the score-descending order itself; this method makes
    /// no ordering promise and may return fewer rows than `paths` if the metadata
    /// index is transiently behind Qdrant (same consistency caveat as
    /// `get_document`'s fuzzy fallback).
    async fn get_summaries_by_paths(
        &self,
        paths: &[String],
        fields: Option<&[String]>,
    ) -> Result<Vec<DocumentSummary>>;

    /// Outbound `document_links` edges for `source_path`, both `kind`s together
    /// (author-written `markdown` and machine-inferred `semantic`), capped at
    /// `limit` with the true total attached so a caller can tell a page from the
    /// whole set.
    ///
    /// Backs `get_document`'s `links_out` (#157). Unlike `links_targeting`
    /// (the move-time link rewriter's helper, which filters to a single `kind`
    /// and returns bare source paths because it only ever needs "who must be
    /// rewritten"), this returns every kind at once with its `score` and an
    /// `exists` flag per edge — `get_document`'s caller cannot assume `markdown`
    /// like the rewriter can, and unlike `/api/graph` (which drops dangling edges
    /// at read time) a broken outbound link is exactly the kind of thing an agent
    /// traversing the KB wants surfaced, not hidden — see [`OutboundLink::exists`].
    async fn links_out(&self, source_path: &str, limit: u64) -> Result<LinkPage<OutboundLink>>;

    /// Inbound `document_links` edges targeting `target_path`, both kinds
    /// together, capped at `limit` with the true total. The reverse-direction
    /// sibling of [`Self::links_out`], using `idx_document_links_target` like
    /// [`StateDb::links_targeting`]. Backs `get_document`'s `links_in` (#157).
    async fn links_in(&self, target_path: &str, limit: u64) -> Result<LinkPage<InboundLink>>;
}

/// One outbound edge from a document, as exposed by `get_document`'s `links_out`.
///
/// `kind` is `"markdown"` for a link extracted from the document's own body at
/// ingest time, or `"semantic"` for a machine-inferred kNN neighbor computed only
/// when `ui.semantic_edges.enabled` is on (off by default) — see the
/// `document_links` table's doc comment. Both kinds are returned undifferentiated
/// except by this field: mixing them silently would let a reader mistake a
/// similarity guess for something the author actually wrote, so the field is
/// mandatory on every edge rather than assumed from context.
#[derive(Debug, Clone, PartialEq)]
pub struct OutboundLink {
    pub target_path: String,
    pub kind: String,
    /// Similarity score for a `semantic` edge; always `None` for `markdown`
    /// (`document_links.score` is only ever written for semantic neighbors — see
    /// `ingest::update_semantic_edges`).
    pub score: Option<f64>,
    /// Whether `target_path` currently exists as an indexed `documents` row.
    /// `document_links` targets are not foreign-keyed to `documents` (see that
    /// table's doc comment) — a link may point at a file that was never indexed,
    /// or one that was renamed out from under it — so this lets a caller
    /// distinguish a live link from a broken one without a second round trip.
    /// `/api/graph` instead drops dangling edges at read time; `get_document`
    /// surfaces them, since "this link is broken" is information worth having,
    /// not noise to hide (the premise of #158's future `validate --links`
    /// report, which this field is deliberately shaped to support without
    /// implementing it).
    pub exists: bool,
}

/// One inbound edge targeting a document, as exposed by `get_document`'s
/// `links_in`. Same `kind`/`score` contract as [`OutboundLink`], but no `exists`
/// flag: a source only ever appears here because its own `document_links` row is
/// still live, and `delete_document` purges a deleted file's outgoing rows
/// (`delete_links_for`) as part of removing it, so a source surviving in this
/// list is — modulo the same one-indexing-run consistency window `get_document`'s
/// fuzzy fallback already documents — a currently indexed document. The asymmetry
/// with `OutboundLink::exists` is deliberate, not an oversight.
#[derive(Debug, Clone, PartialEq)]
pub struct InboundLink {
    pub source_path: String,
    pub kind: String,
    pub score: Option<f64>,
}

/// A capped page of link edges plus the true total, so a caller can always tell
/// whether the list was truncated — the same `total`/`has_more`-style contract as
/// [`DocumentQueryResult`], parameterized over which edge direction produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkPage<T> {
    pub links: Vec<T>,
    pub total: u64,
}

impl<T> LinkPage<T> {
    pub fn has_more(&self) -> bool {
        (self.links.len() as u64) < self.total
    }
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct IndexedFile {
    pub file_path: String,
    pub content_hash: String,
    pub chunk_count: i64,
    pub indexed_at: String,
    /// Fingerprint of the schema this file was last validated and projected against
    /// (`ResolvedSchema::fingerprint`). Empty for rows written before schema tracking
    /// existed, which forces one revalidation pass. A mismatch alone never re-embeds
    /// (#351): it revalidates and refreshes the metadata, see `payload_hash`.
    pub schema_hash: String,
    /// Fingerprint of the schema inputs that shape this file's Qdrant payload
    /// frontmatter (`ingest::payload_hash_for`, over `ResolvedSchema::payload_fingerprint`)
    /// as of the last index or refresh (#351). `''` means unknown — every row written
    /// before this column existed — and is treated as current while `schema_hash`
    /// matches, so upgrading writes nothing to Qdrant; once `schema_hash` moves, an
    /// unknown value counts as changed and the payload is refreshed in place.
    pub payload_hash: String,
    /// Fingerprint of the chunking configuration (`ingest::chunking_fingerprint`: every
    /// `chunking.*` setting that shapes chunk text or payload, plus
    /// `ingest::CHUNKER_VERSION`) this file was last chunked and embedded under. Empty
    /// for rows written before this column existed, which never equals a real
    /// fingerprint and so forces one re-chunk/re-embed per file after upgrading.
    pub chunking_fingerprint: String,
    /// File modification time (Unix seconds) as of the last successful index of this
    /// file, or the last stat-only refresh of an unchanged file (`process_file`'s skip
    /// path via `StateDb::update_stat`, #139). The reconcile scan
    /// (`ingest::scan_for_dirty`) reads these two columns directly via [`ScanRow`]
    /// rather than through this struct; `index_paths`'s scoped scan reads them via
    /// `get_many` to decide whether a skipped file's stat baseline needs refreshing.
    pub mtime: i64,
    /// File size in bytes as of the last successful index or stat-only refresh, same
    /// purpose and caveat as `mtime`. `-1` for rows written before this column existed,
    /// which never matches a real size and so forces one stat-mismatch (and therefore a
    /// content-hash check) per file on the first reconcile scan after upgrading.
    pub size: i64,
}

/// Every column of one `indexed_files` row that [`StateDb::upsert_record`] writes —
/// a struct rather than more positional arguments on [`StateDb::upsert`].
#[derive(Debug, Clone, Copy)]
pub struct IndexedFileRecord<'a> {
    pub file_path: &'a str,
    pub content_hash: &'a str,
    pub chunk_count: i64,
    pub schema_hash: &'a str,
    pub payload_hash: &'a str,
    pub chunking_fingerprint: &'a str,
    pub mtime: i64,
    pub size: i64,
}

/// One row from [`StateDb::scan_indexed_files`]: an `indexed_files` row plus the
/// `documents.content_hash` for the same path, if a metadata row exists at all.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct ScanRow {
    pub file_path: String,
    pub content_hash: String,
    pub schema_hash: String,
    /// `indexed_files.payload_hash` (#351); see [`IndexedFile::payload_hash`].
    pub payload_hash: String,
    pub chunking_fingerprint: String,
    pub mtime: i64,
    pub size: i64,
    /// `indexed_files.chunk_count`. Zero marks a file recorded as present but not
    /// indexable (a document with no chunks — see `ingest::index_paths_generic`'s
    /// handling of `FileOutcome::Empty`), which has no `documents` row on purpose.
    pub chunk_count: i64,
    /// `documents.content_hash` for this path, or `None` if no metadata row exists yet.
    pub doc_hash: Option<String>,
}

/// What happened to one document during reprojection.
enum ReprojectOutcome {
    Done,
    /// The row disappeared mid-run; nothing to do.
    Gone,
    /// Stored frontmatter is not a JSON object; left as-is, with a logged reason.
    Unparseable,
    /// A database error; left as-is, with a logged reason.
    Failed,
    /// Placeholder before the first attempt; counted as failed if it survives.
    Retry,
}

/// Replace `file_path`'s `document_fields` rows with `frontmatter`'s projection
/// under `schema`, inside the caller's transaction.
async fn write_projection(
    tx: &mut sqlx::Transaction<'_, Sqlite>,
    file_path: &str,
    frontmatter: &HashMap<String, Value>,
    schema: &crate::schema::ResolvedSchema,
) -> Result<(), sqlx::Error> {
    sqlx::query("DELETE FROM document_fields WHERE file_path = ?")
        .bind(file_path)
        .execute(&mut **tx)
        .await?;

    for row in document_fields::flatten_frontmatter(frontmatter, schema) {
        // INSERT OR IGNORE: a document may legitimately repeat a value within one
        // array (duplicate tags), which collides on the primary key.
        sqlx::query(
            "INSERT OR IGNORE INTO document_fields (file_path, field, elem, value_text, value_num)
             VALUES (?, ?, ?, ?, ?)",
        )
        .bind(file_path)
        .bind(&row.field)
        .bind(row.elem)
        .bind(&row.value_text)
        .bind(row.value_num)
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

/// Whether an error is SQLite reporting the database as busy or locked.
///
/// A deferred transaction that opens with a read establishes a snapshot; if another
/// connection commits before it writes, SQLite rejects the write outright rather than
/// serializing it, and `busy_timeout` does not help because retrying the same
/// transaction can never succeed. Only a fresh transaction can.
fn is_busy_or_locked(e: &sqlx::Error) -> bool {
    match e {
        sqlx::Error::Database(db) => {
            let msg = db.message().to_ascii_lowercase();
            msg.contains("database is locked") || msg.contains("database table is locked")
        }
        _ => false,
    }
}

/// Whether an error is SQLite's "duplicate column name", i.e. the column already exists.
fn is_duplicate_column(e: &sqlx::Error) -> bool {
    // Match on the database error specifically rather than any stringified error, so an
    // unrelated failure that happens to mention the phrase cannot be swallowed.
    match e {
        sqlx::Error::Database(db) => db
            .message()
            .to_ascii_lowercase()
            .contains("duplicate column"),
        _ => false,
    }
}

/// Add a column to an existing table when it is not already present.
///
/// SQLite has no `ADD COLUMN IF NOT EXISTS`, and this project has no migration runner,
/// so the presence check goes through `PRAGMA table_info`.
async fn add_column_if_missing(
    pool: &SqlitePool,
    table: &str,
    column: &str,
    definition: &str,
) -> Result<()> {
    let existing: Vec<(i64, String)> = sqlx::query_as(&format!("PRAGMA table_info({table})"))
        .fetch_all(pool)
        .await
        .with_context(|| format!("Failed to inspect columns of '{table}'"))?;

    if existing.iter().any(|(_, name)| name == column) {
        return Ok(());
    }

    match sqlx::query(&format!(
        "ALTER TABLE {table} ADD COLUMN {column} {definition}"
    ))
    .execute(pool)
    .await
    {
        Ok(_) => Ok(()),
        // The check and the ALTER are not atomic, so two processes upgrading at the
        // same instant can both see the column as missing. Losing that race means the
        // column exists, which is precisely the desired end state.
        Err(e) if is_duplicate_column(&e) => {
            debug!("Column '{column}' on '{table}' was added concurrently");
            Ok(())
        }
        Err(e) => {
            Err(anyhow::Error::new(e)
                .context(format!("Failed to add column '{column}' to '{table}'")))
        }
    }
}

/// Whether an existing `document_fields` table has the current shape (`elem` in its
/// primary key, #349); `None` when there is no such table yet.
async fn document_fields_current_shape(
    conn: &mut sqlx::SqliteConnection,
) -> Result<Option<bool>, sqlx::Error> {
    let columns: Vec<(String, i64)> =
        sqlx::query_as("SELECT name, pk FROM pragma_table_info('document_fields')")
            .fetch_all(&mut *conn)
            .await?;
    if columns.is_empty() {
        return Ok(None);
    }
    Ok(Some(
        columns.iter().any(|(name, pk)| name == "elem" && *pk > 0),
    ))
}

/// Drop a `document_fields` table of an older shape, and reset `PRAGMA
/// user_version` to 0 in the same transaction so the next
/// [`StateDb::ensure_projection_current`] reprojects into the recreated table
/// whatever version was recorded. Returns whether it dropped one.
///
/// Checked once outside a transaction (the common, current case costs one pragma
/// read) and again under `BEGIN IMMEDIATE`, so two processes opening an old
/// database at once rebuild it once and neither fails.
async fn rebuild_document_fields_if_old_shape(pool: &SqlitePool) -> Result<bool> {
    let mut conn = pool.acquire().await?;
    if document_fields_current_shape(&mut conn).await? != Some(false) {
        return Ok(false);
    }
    sqlx::query("BEGIN IMMEDIATE").execute(&mut *conn).await?;
    let rebuilt = async {
        if document_fields_current_shape(&mut conn).await? != Some(false) {
            return Ok::<bool, sqlx::Error>(false);
        }
        sqlx::query("DROP TABLE document_fields")
            .execute(&mut *conn)
            .await?;
        sqlx::query("PRAGMA user_version = 0")
            .execute(&mut *conn)
            .await?;
        Ok(true)
    }
    .await;
    match rebuilt {
        Ok(rebuilt) => {
            sqlx::query("COMMIT").execute(&mut *conn).await?;
            if rebuilt {
                tracing::info!(
                    "document_fields had an older shape; dropped it to be rebuilt from \
                     stored frontmatter (no re-embedding)"
                );
            }
            Ok(rebuilt)
        }
        Err(e) => {
            let _ = sqlx::query("ROLLBACK").execute(&mut *conn).await;
            Err(e).context("Failed to rebuild the document_fields table")
        }
    }
}

pub struct StateDb {
    pool: SqlitePool,
    /// As opened; keys [`REPROJECT_FAILURES`].
    db_path: std::path::PathBuf,
}

/// How many automatic reprojects (`ensure_projection_current` without `force`)
/// have failed in this process, per database path. After
/// [`MAX_AUTOMATIC_REPROJECT_ATTEMPTS`] failures the automatic check stops
/// retrying until the process restarts, so a persistent failure is not re-run (and
/// re-logged) at the start of every index run; `reproject-fields` still forces one.
static REPROJECT_FAILURES: std::sync::LazyLock<
    std::sync::Mutex<HashMap<std::path::PathBuf, usize>>,
> = std::sync::LazyLock::new(Default::default);

/// The first attempt plus one retry.
const MAX_AUTOMATIC_REPROJECT_ATTEMPTS: usize = 2;

/// How many times [`StateDb::new`] tries to open and set up the database when it
/// keeps finding it busy — with a backoff of 10 ms per attempt (capped at 200 ms),
/// a few seconds in all.
const MAX_OPEN_ATTEMPTS: u32 = 25;

/// Whether SQLite reported the database busy or locked anywhere in `e`'s chain.
fn is_busy_in_chain(e: &anyhow::Error) -> bool {
    e.chain().any(|cause| {
        cause
            .downcast_ref::<sqlx::Error>()
            .is_some_and(is_busy_or_locked)
    })
}

impl StateDb {
    pub async fn new(db_path: &Path) -> Result<Self> {
        let db_str = db_path.to_str().ok_or_else(|| {
            anyhow::anyhow!("State DB path is not valid UTF-8: {}", db_path.display())
        })?;
        let options = SqliteConnectOptions::from_str(&format!("sqlite:{}?mode=rwc", db_str))?
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(std::time::Duration::from_secs(5))
            // SQLite disables foreign keys per connection by default. Without this the
            // ON DELETE CASCADE from documents to document_fields silently never fires,
            // leaving orphaned projection rows behind on every delete.
            .foreign_keys(true);

        // Opening is retried as a whole on SQLITE_BUSY. `busy_timeout` makes SQLite
        // wait for a lock, but not in every case: switching a rollback-journal file
        // to WAL (the `journal_mode` pragma each new connection runs) needs an
        // exclusive lock and fails at once while another process holds any lock on
        // the file, and an autocommit DDL statement (`CREATE TABLE IF NOT EXISTS`,
        // the shape rebuild's `DROP`) that read the schema before another
        // connection changed it cannot upgrade its snapshot — retrying the same
        // statement can never succeed, a fresh attempt does. Several processes
        // opening one database at once (serve plus a CLI command, typically on the
        // first start after an upgrade, when the shape rebuild also runs) hit both.
        // Every setup step is idempotent, so a fresh pool and a rerun from the top
        // is always safe.
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            match Self::open_and_migrate(options.clone(), db_path).await {
                Ok(db) => return Ok(db),
                Err(e) if attempt < MAX_OPEN_ATTEMPTS && is_busy_in_chain(&e) => {
                    debug!(attempt, "state DB busy while opening; retrying: {e:#}");
                    let backoff = (10 * u64::from(attempt)).min(200);
                    tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// One attempt at [`Self::new`]: connect, then create or upgrade every table.
    async fn open_and_migrate(options: SqliteConnectOptions, db_path: &Path) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS indexed_files (
                file_path    TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                chunk_count  INTEGER NOT NULL,
                indexed_at   TEXT NOT NULL DEFAULT (datetime('now'))
            )",
        )
        .execute(&pool)
        .await?;

        // `CREATE TABLE IF NOT EXISTS` is a no-op against a table that already exists,
        // so a column added after a deployment shipped needs an explicit ALTER. There
        // is no migration runner in this project; this guarded add is the mechanism.
        //
        // Existing rows get '', which never equals a real fingerprint, so the first
        // index run after upgrading revalidates every file exactly once. That is the
        // correct behavior, but it is worth expecting rather than being surprised by.
        add_column_if_missing(
            &pool,
            "indexed_files",
            "schema_hash",
            "TEXT NOT NULL DEFAULT ''",
        )
        .await?;

        // Payload-shape fingerprint (#351, `ingest::payload_hash_for`). Unlike the
        // columns above, '' here does NOT force anything: it means "unknown", which
        // ingest treats as current while `schema_hash` still matches. Upgrading is
        // therefore a no-op reindex — no re-embed, no Qdrant write — and a later
        // schema edit refreshes the payload in place rather than re-embedding.
        add_column_if_missing(
            &pool,
            "indexed_files",
            "payload_hash",
            "TEXT NOT NULL DEFAULT ''",
        )
        .await?;

        // Chunking-config fingerprint (`ingest::chunking_fingerprint`). Same upgrade
        // semantics as `schema_hash` above: pre-existing rows get '', which never equals
        // a real fingerprint, so both the reconcile scan and `process_file`'s skip check
        // treat every such file as dirty exactly once — a one-time re-chunk/re-embed of
        // the whole corpus after upgrading. That is the point: without it a changed
        // `chunking.*` setting (or chunker code) would leave the index silently mixed.
        add_column_if_missing(
            &pool,
            "indexed_files",
            "chunking_fingerprint",
            "TEXT NOT NULL DEFAULT ''",
        )
        .await?;

        // Stat-based pre-filter columns for the reconcile scan (`ingest::scan_for_dirty`):
        // at corpus sizes in the thousands-to-tens-of-thousands range, content-hashing
        // every file on every sweep is too expensive, so the scan compares mtime/size
        // first and only asks the scoped indexer to open (and therefore hash) a file
        // when one of them changed. `size` defaults to -1, which never matches a real
        // file size, so every row written before this column existed gets one forced
        // stat-mismatch (and therefore a real hash check) on the first post-upgrade scan.
        add_column_if_missing(
            &pool,
            "indexed_files",
            "mtime",
            "INTEGER NOT NULL DEFAULT 0",
        )
        .await?;
        add_column_if_missing(
            &pool,
            "indexed_files",
            "size",
            "INTEGER NOT NULL DEFAULT -1",
        )
        .await?;

        // Document-level metadata index. `frontmatter` is the faithful JSON record and
        // the source of truth; document_fields below is a rebuildable projection of it.
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS documents (
                file_path    TEXT PRIMARY KEY,
                title        TEXT,
                description  TEXT,
                frontmatter  TEXT NOT NULL,
                mtime        INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                chunk_count  INTEGER NOT NULL,
                indexed_at   TEXT NOT NULL DEFAULT (datetime('now'))
            )",
        )
        .execute(&pool)
        .await?;

        // `query_documents` supports ordering by mtime/title/indexed_at, each with a
        // mandatory `d.file_path ASC` tiebreaker (ties are common: a batch reindex
        // gives many rows the same mtime/indexed_at). Without an index shaped to match,
        // SQLite must materialize and sort the whole filtered set before LIMIT/OFFSET
        // can trim it — `CREATE INDEX IF NOT EXISTS` is already idempotent against an
        // existing production database, same as the document_fields indexes below.
        //
        // A single ASC-ordered composite index does not cover DESC queries: SQLite can
        // only satisfy an ORDER BY straight from an index when the whole key (including
        // the tiebreaker) matches the index's declared direction, or is its exact
        // reverse. `(col ASC, file_path ASC)` scanned backward yields `(col DESC,
        // file_path DESC)` — not what these queries ask for, since the tiebreaker is
        // always ASC. So each orderable column needs one index per direction; verified
        // with `EXPLAIN QUERY PLAN` (see the `query_documents_order_by_uses_index`
        // tests) that omitting either half still falls back to `USE TEMP B-TREE FOR
        // ORDER BY`. `file_path` itself needs no such pair: it is the primary key, and
        // a single-column index can satisfy both directions by scanning in reverse.
        for (name, column) in [
            ("idx_documents_mtime_asc", "mtime ASC"),
            ("idx_documents_mtime_desc", "mtime DESC"),
            ("idx_documents_title_asc", "title ASC"),
            ("idx_documents_title_desc", "title DESC"),
            ("idx_documents_indexed_at_asc", "indexed_at ASC"),
            ("idx_documents_indexed_at_desc", "indexed_at DESC"),
        ] {
            sqlx::query(&format!(
                "CREATE INDEX IF NOT EXISTS {name} ON documents({column}, file_path ASC)"
            ))
            .execute(&pool)
            .await?;
        }

        // `elem` (#349) is the element of the outermost list of objects a row came
        // from (`document_fields::FieldRow::elem`), part of the key so two elements
        // carrying the same value both keep a row. A table from before it existed
        // cannot be ALTERed into this key, so `rebuild_document_fields_if_old_shape`
        // drops it first; the projection is derived data, rebuilt from
        // `documents.frontmatter` by `ensure_projection_current`.
        rebuild_document_fields_if_old_shape(&pool).await?;
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS document_fields (
                file_path  TEXT NOT NULL REFERENCES documents(file_path) ON DELETE CASCADE,
                field      TEXT NOT NULL,
                elem       INTEGER NOT NULL DEFAULT 0,
                value_text TEXT NOT NULL,
                value_num  REAL,
                PRIMARY KEY (file_path, field, elem, value_text)
            )",
        )
        .execute(&pool)
        .await?;

        // The composite primary key is file_path-first, which does not serve the
        // "which documents have field = value" lookup that filtering actually performs.
        // These secondary indexes are load-bearing, not an optimization. A correlated
        // list filter's per-element lookup (`file_path`, `field`, `elem`) is a prefix
        // of the primary key and needs no index of its own.
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_document_fields_field_text
             ON document_fields(field, value_text)",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_document_fields_field_num
             ON document_fields(field, value_num)",
        )
        .execute(&pool)
        .await?;

        // Edges for the KB web UI's graph view: markdown links extracted at ingest
        // (`kind = 'markdown'`) and semantic kNN neighbors precomputed at index time
        // (`kind = 'semantic'`, carrying a similarity `score`). No `REFERENCES ...
        // documents(file_path)` / `ON DELETE CASCADE` here — unlike document_fields,
        // a link's target need not exist yet (or ever) as a `documents` row: a link may
        // point at a file that hasn't been indexed yet, or one that was renamed out from
        // under it; the graph handler drops dangling edges at read time instead. Rows
        // are removed explicitly via `delete_links_for` (called from `delete_document`)
        // and replaced wholesale per (source_path, kind) via `replace_links`.
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS document_links (
                source_path TEXT NOT NULL,
                target_path TEXT NOT NULL,
                kind        TEXT NOT NULL DEFAULT 'markdown',
                score       REAL,
                PRIMARY KEY (source_path, target_path, kind)
            )",
        )
        .execute(&pool)
        .await?;

        // The reverse lookup — "what points at this target" — which the primary key
        // (source-first) does not serve. `delete_links_for` does NOT need this: it
        // purges a renamed/deleted file's OWN outgoing rows, filtered by
        // `source_path`, which the primary key already serves directly. The live user
        // of this index is `links_targeting`, which answers "which documents must be
        // updated when this path moves" for the document-move link rewriter.
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_document_links_target
             ON document_links(target_path)",
        )
        .execute(&pool)
        .await?;

        Ok(Self {
            pool,
            db_path: db_path.to_path_buf(),
        })
    }

    /// Raw pool access so tests can assert on rows this API does not expose.
    #[cfg(test)]
    pub(crate) fn pool_for_test(&self) -> &SqlitePool {
        &self.pool
    }

    #[cfg(test)]
    pub async fn get(&self, file_path: &str) -> Result<Option<IndexedFile>> {
        let row = sqlx::query_as::<_, IndexedFile>(
            "SELECT file_path, content_hash, chunk_count, indexed_at, schema_hash, payload_hash, chunking_fingerprint, mtime, size
             FROM indexed_files WHERE file_path = ?",
        )
        .bind(file_path)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row)
    }

    /// Write a whole `indexed_files` row with an unknown (`''`) `payload_hash` — see
    /// [`IndexedFile::payload_hash`]. Test-only shorthand: ingest writes rows through
    /// [`Self::upsert_record`], which records the payload hash too.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub async fn upsert(
        &self,
        file_path: &str,
        content_hash: &str,
        chunk_count: i64,
        schema_hash: &str,
        chunking_fingerprint: &str,
        mtime: i64,
        size: i64,
    ) -> Result<()> {
        self.upsert_record(&IndexedFileRecord {
            file_path,
            content_hash,
            chunk_count,
            schema_hash,
            payload_hash: "",
            chunking_fingerprint,
            mtime,
            size,
        })
        .await
    }

    /// Write a whole `indexed_files` row (`INSERT OR REPLACE`, `indexed_at` now).
    pub async fn upsert_record(&self, record: &IndexedFileRecord<'_>) -> Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO indexed_files
                (file_path, content_hash, chunk_count, indexed_at, schema_hash,
                 payload_hash, chunking_fingerprint, mtime, size)
             VALUES (?, ?, ?, datetime('now'), ?, ?, ?, ?, ?)",
        )
        .bind(record.file_path)
        .bind(record.content_hash)
        .bind(record.chunk_count)
        .bind(record.schema_hash)
        .bind(record.payload_hash)
        .bind(record.chunking_fingerprint)
        .bind(record.mtime)
        .bind(record.size)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Record that an already-indexed file was revalidated and its metadata (and, when
    /// its payload hash moved, its Qdrant payload) refreshed under a new schema without
    /// re-chunking or re-embedding it (#351): only `schema_hash`, `payload_hash` and the
    /// stat baseline change. `content_hash`, `chunk_count`, `chunking_fingerprint` and
    /// `indexed_at` describe the chunks still in Qdrant, which this does not touch.
    pub async fn update_schema_hashes(
        &self,
        file_path: &str,
        schema_hash: &str,
        payload_hash: &str,
        mtime: i64,
        size: i64,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE indexed_files SET schema_hash = ?, payload_hash = ?, mtime = ?, size = ?
             WHERE file_path = ?",
        )
        .bind(schema_hash)
        .bind(payload_hash)
        .bind(mtime)
        .bind(size)
        .bind(file_path)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Refresh only the stat pre-filter baseline (`mtime`/`size`) for an already-indexed
    /// row, leaving `content_hash`/`chunk_count`/`schema_hash`/`indexed_at` untouched.
    ///
    /// Exists for the "content unchanged, mtime moved" case (#139): `process_file`'s
    /// skip path reads a fresh mtime/size but has no changed content to justify a full
    /// [`Self::upsert`] (which would also bump `indexed_at`, misrepresenting the file as
    /// just-reindexed). A no-op `UPDATE` still costs a write, so the caller is expected
    /// to compare against the stored value first and only call this when it differs —
    /// otherwise every skip on every scan would rewrite the row it exists to avoid
    /// rewriting.
    pub async fn update_stat(&self, file_path: &str, mtime: i64, size: i64) -> Result<()> {
        sqlx::query("UPDATE indexed_files SET mtime = ?, size = ? WHERE file_path = ?")
            .bind(mtime)
            .bind(size)
            .bind(file_path)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    pub async fn delete(&self, file_path: &str) -> Result<()> {
        sqlx::query("DELETE FROM indexed_files WHERE file_path = ?")
            .bind(file_path)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    pub async fn list_all(&self) -> Result<Vec<IndexedFile>> {
        let rows = sqlx::query_as::<_, IndexedFile>(
            "SELECT file_path, content_hash, chunk_count, indexed_at, schema_hash, payload_hash, chunking_fingerprint, mtime, size
             FROM indexed_files ORDER BY file_path",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// One page of `indexed_files` joined with `documents`, ordered by path.
    ///
    /// Used only by the reconcile scan (`ingest::scan_for_dirty`), which needs to
    /// interleave a `tokio::fs::metadata` stat call (async) between rows — a plain
    /// callback-based "visit each row" API cannot do that without either boxing
    /// futures or blocking the executor, so the scan owns its own paging loop and
    /// calls this once per page instead. At corpus sizes in the thousands-to-tens-of-
    /// thousands range, a single `list_all()`-style `Vec<IndexedFile>` covering the
    /// whole table would be a multi-megabyte allocation held for the whole scan; paging
    /// keeps peak memory bounded by `limit` regardless of corpus size. `LIMIT ...
    /// OFFSET` is good enough here — a row inserted or deleted by a concurrent write
    /// mid-scan can shift a page and be seen twice or missed once, but the scan is a
    /// best-effort detector with a self-healing follow-up (the next scheduled
    /// reconcile), not the system of record.
    ///
    /// `documents.content_hash` is carried along as `doc_hash` so the caller can detect
    /// metadata staleness — a file whose content is unchanged but whose `documents` row
    /// is missing or stale relative to `indexed_files.content_hash` — without a second
    /// query per row.
    pub async fn fetch_indexed_files_page(&self, limit: i64, offset: i64) -> Result<Vec<ScanRow>> {
        let rows: Vec<ScanRow> = sqlx::query_as(
            "SELECT i.file_path, i.content_hash, i.schema_hash, i.payload_hash, i.chunking_fingerprint, i.mtime, i.size, \
                    i.chunk_count, d.content_hash AS doc_hash \
             FROM indexed_files i LEFT JOIN documents d ON d.file_path = i.file_path \
             ORDER BY i.file_path LIMIT ? OFFSET ?",
        )
        .bind(limit.max(1))
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// Batch equivalent of [`Self::get`], for a caller that already knows exactly which
    /// paths it cares about (the scoped indexer's dirty-path list) and wants one or a
    /// few round trips instead of one per path. Chunks the `IN (...)` list so a large
    /// batch cannot exceed SQLite's bound-parameter limit (default 999).
    pub async fn get_many(&self, file_paths: &[String]) -> Result<HashMap<String, IndexedFile>> {
        let mut out = HashMap::with_capacity(file_paths.len());
        for chunk in file_paths.chunks(SQLITE_MAX_PARAMS_PER_QUERY) {
            if chunk.is_empty() {
                continue;
            }
            let mut builder = QueryBuilder::<Sqlite>::new(
                "SELECT file_path, content_hash, chunk_count, indexed_at, schema_hash, payload_hash, chunking_fingerprint, mtime, size \
                 FROM indexed_files WHERE file_path IN (",
            );
            let mut separated = builder.separated(", ");
            for path in chunk {
                separated.push_bind(path);
            }
            builder.push(")");
            let rows: Vec<IndexedFile> = builder.build_query_as().fetch_all(&self.pool).await?;
            for row in rows {
                out.insert(row.file_path.clone(), row);
            }
        }
        Ok(out)
    }

    /// Batch equivalent of [`Self::get_document_hash`], for the same reason as
    /// [`Self::get_many`].
    pub async fn get_document_hashes_many(
        &self,
        file_paths: &[String],
    ) -> Result<HashMap<String, String>> {
        let mut out = HashMap::with_capacity(file_paths.len());
        for chunk in file_paths.chunks(SQLITE_MAX_PARAMS_PER_QUERY) {
            if chunk.is_empty() {
                continue;
            }
            let mut builder = QueryBuilder::<Sqlite>::new(
                "SELECT file_path, content_hash FROM documents WHERE file_path IN (",
            );
            let mut separated = builder.separated(", ");
            for path in chunk {
                separated.push_bind(path);
            }
            builder.push(")");
            let rows: Vec<(String, String)> =
                builder.build_query_as().fetch_all(&self.pool).await?;
            out.extend(rows);
        }
        Ok(out)
    }

    /// Empty every index table: `indexed_files`, the metadata index (`documents`,
    /// `document_fields`) and the link graph (`document_links`), in one transaction.
    ///
    /// A full reindex clears `indexed_files` before orphan detection reads it, so
    /// orphans are never computed on a full run — anything left here for a file that is
    /// gone is a phantom row no future run can detect, because detection is driven off
    /// `indexed_files`. That holds for `document_links` as much as for `documents`:
    /// it has no foreign key to `documents` (a link target need not be indexed), so
    /// nothing cascades into it, and a removed document's outgoing rows would keep
    /// surfacing as `links_in` sources forever. Every surviving file's links are
    /// rebuilt by the run that follows. `document_fields` would cascade from
    /// `documents`, but is cleared explicitly so the result does not depend on the
    /// connection's `foreign_keys` pragma.
    pub async fn clear(&self) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        for table in [
            "indexed_files",
            "document_fields",
            "documents",
            "document_links",
        ] {
            sqlx::query(&format!("DELETE FROM {table}"))
                .execute(&mut *tx)
                .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    pub async fn count(&self) -> Result<i64> {
        let row: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM indexed_files")
            .fetch_one(&self.pool)
            .await?;

        Ok(row.0)
    }

    /// Sum of `indexed_files.chunk_count` across every tracked file — the number of
    /// Qdrant points state.db believes are live. `index_paths` writes exactly
    /// `chunks.len()` points per file and records that same count in the same
    /// bookkeeping step (see `ingest::upsert_pending`), so in a healthy steady state
    /// this equals Qdrant's own point count exactly. `/status` (`collect_status` in
    /// `server.rs`) compares the two as a passive check for a Qdrant data wipe that
    /// left state.db intact (#155) — a deficit (Qdrant points < this sum) is the
    /// signal; a surplus is not, since a couple of documented conditions leave stale
    /// Qdrant points behind on purpose. See the comment at the comparison site for why
    /// the check is one-sided.
    pub async fn total_chunk_count(&self) -> Result<i64> {
        let row: (i64,) = sqlx::query_as("SELECT COALESCE(SUM(chunk_count), 0) FROM indexed_files")
            .fetch_one(&self.pool)
            .await?;

        Ok(row.0)
    }

    /// [`Self::upsert_document_metadata_with`] under an empty schema — every field
    /// undeclared, so `value_num` follows the JSON type. For tests that are not about
    /// the schema-driven projection.
    #[cfg(test)]
    pub async fn upsert_document_metadata(
        &self,
        file_path: &str,
        frontmatter: &HashMap<String, Value>,
        mtime: i64,
        content_hash: &str,
        chunk_count: i64,
    ) -> Result<()> {
        self.upsert_document_metadata_with(
            file_path,
            frontmatter,
            &crate::schema::ResolvedSchema::default(),
            mtime,
            content_hash,
            chunk_count,
        )
        .await
    }

    /// Replace a document's metadata row and its full field projection, the latter
    /// read under `schema` — the one governing `file_path` (#349).
    ///
    /// Runs as a single transaction: the projection is deleted and rebuilt rather than
    /// merged, so a field removed from frontmatter does not leave a stale row behind.
    pub async fn upsert_document_metadata_with(
        &self,
        file_path: &str,
        frontmatter: &HashMap<String, Value>,
        schema: &crate::schema::ResolvedSchema,
        mtime: i64,
        content_hash: &str,
        chunk_count: i64,
    ) -> Result<()> {
        let title = frontmatter.get("title").and_then(|v| v.as_str());
        let description = frontmatter.get("description").and_then(|v| v.as_str());
        let frontmatter_json = serde_json::to_string(frontmatter)
            .with_context(|| format!("Failed to serialize frontmatter for '{}'", file_path))?;

        let mut tx = self.pool.begin().await?;

        sqlx::query(
            "INSERT INTO documents
                (file_path, title, description, frontmatter, mtime, content_hash, chunk_count, indexed_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
             ON CONFLICT(file_path) DO UPDATE SET
                title        = excluded.title,
                description  = excluded.description,
                frontmatter  = excluded.frontmatter,
                mtime        = excluded.mtime,
                content_hash = excluded.content_hash,
                chunk_count  = excluded.chunk_count,
                indexed_at   = excluded.indexed_at",
        )
        .bind(file_path)
        .bind(title)
        .bind(description)
        .bind(&frontmatter_json)
        .bind(mtime)
        .bind(content_hash)
        .bind(chunk_count)
        .execute(&mut *tx)
        .await?;

        write_projection(&mut tx, file_path, frontmatter, schema).await?;

        tx.commit().await?;
        Ok(())
    }

    /// The projection version recorded in `PRAGMA user_version`.
    pub async fn projection_version(&self) -> Result<i64> {
        let (version,): (i64,) = sqlx::query_as("PRAGMA user_version")
            .fetch_one(&self.pool)
            .await?;
        Ok(version)
    }

    /// Bring `document_fields` up to [`PROJECTION_VERSION`]: when the recorded
    /// version differs (an upgrade, or a table [`StateDb::new`] just rebuilt from an
    /// older shape, which resets it to 0) — or `force` is set, as `reproject-fields`
    /// does — reproject every document from its stored frontmatter under `schemas`.
    /// No markdown is read, nothing is embedded, Qdrant is untouched.
    ///
    /// The version is written only after a reproject in which no document failed,
    /// so a crash or error part-way leaves it as it was and the next call starts
    /// over — though an automatic check retries a failed reproject only once per
    /// process ([`MAX_AUTOMATIC_REPROJECT_ATTEMPTS`]) and then returns `None`
    /// without trying until a restart or a forced run, so a persistent failure is
    /// not repeated on every index run. Returns `None` when the projection was
    /// already current (one pragma
    /// read), else what the reproject did. Callers run it right after opening the
    /// database and before serving or indexing, so an upgraded database is never
    /// queried with an empty projection.
    pub async fn ensure_projection_current(
        &self,
        schemas: &crate::schema::SchemaCache,
        force: bool,
    ) -> Result<Option<ReprojectReport>> {
        let stored = self.projection_version().await?;
        if stored == PROJECTION_VERSION && !force {
            return Ok(None);
        }
        let failures = || {
            REPROJECT_FAILURES
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        };
        if !force
            && failures().get(&self.db_path).copied().unwrap_or(0)
                >= MAX_AUTOMATIC_REPROJECT_ATTEMPTS
        {
            debug!("automatic reprojection already failed in this process; not retrying");
            return Ok(None);
        }
        tracing::info!(
            stored,
            current = PROJECTION_VERSION,
            "reprojecting document fields from stored frontmatter (no re-embedding)"
        );
        let outcome = async {
            let report = self.reproject_all_fields(schemas).await?;
            if report.failed > 0 {
                anyhow::bail!(
                    "{} document(s) could not be reprojected; the projection version stays at \
                     {stored}, so a later start (or `reproject-fields`) retries",
                    report.failed
                );
            }
            sqlx::query(&format!("PRAGMA user_version = {PROJECTION_VERSION}"))
                .execute(&self.pool)
                .await?;
            Ok(report)
        }
        .await;
        match outcome {
            Ok(report) => {
                failures().remove(&self.db_path);
                Ok(Some(report))
            }
            Err(e) => {
                *failures().entry(self.db_path.clone()).or_default() += 1;
                Err(e)
            }
        }
    }

    /// A document's stored `documents.frontmatter`, or `None` when it has no metadata
    /// row (or the stored JSON is not an object).
    ///
    /// Ingest writes the same frontmatter (derived fields included) to this row and to
    /// every chunk's Qdrant payload, so its `description` is what chunking prepended.
    /// A payload refresh (#351) reads it before rewriting the row, to know whether the
    /// prepended description changed; with no row it takes the full path.
    pub async fn document_frontmatter(
        &self,
        file_path: &str,
    ) -> Result<Option<serde_json::Map<String, Value>>> {
        let row: Option<(String,)> =
            sqlx::query_as("SELECT frontmatter FROM documents WHERE file_path = ?")
                .bind(file_path)
                .fetch_optional(&self.pool)
                .await?;
        let Some((json,)) = row else {
            return Ok(None);
        };
        Ok(match serde_json::from_str::<Value>(&json) {
            Ok(Value::Object(map)) => Some(map),
            _ => None,
        })
    }

    /// Remove a document's metadata. Projection rows cascade via the foreign key;
    /// `document_links` does not (its targets may not be `documents` rows at all), so
    /// its outgoing edges are cleared explicitly here.
    pub async fn delete_document(&self, file_path: &str) -> Result<()> {
        sqlx::query("DELETE FROM documents WHERE file_path = ?")
            .bind(file_path)
            .execute(&self.pool)
            .await?;

        self.delete_links_for(file_path).await?;

        Ok(())
    }

    /// Replace every `document_links` row for `(source_path, kind)` with `targets`.
    ///
    /// Scoped to `kind` rather than the whole `source_path`, so refreshing one edge
    /// kind (e.g. recomputed semantic neighbors) never touches the other (markdown
    /// links extracted from the same file's body). Runs as delete-then-insert inside a
    /// transaction so a reader never observes a partially-replaced edge set.
    pub async fn replace_links(
        &self,
        source_path: &str,
        kind: &str,
        targets: &[(String, Option<f64>)],
    ) -> Result<()> {
        let mut tx = self.pool.begin().await?;

        sqlx::query("DELETE FROM document_links WHERE source_path = ? AND kind = ?")
            .bind(source_path)
            .bind(kind)
            .execute(&mut *tx)
            .await?;

        for (target_path, score) in targets {
            sqlx::query(
                "INSERT OR IGNORE INTO document_links (source_path, target_path, kind, score)
                 VALUES (?, ?, ?, ?)",
            )
            .bind(source_path)
            .bind(target_path)
            .bind(kind)
            .bind(score)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    /// Every edge in the graph, unfiltered: (source_path, target_path, kind, score).
    ///
    /// Backs `GET /api/graph`, which filters dangling edges (targets not among the
    /// current node set) itself — this is a plain unlimited read, same rationale as
    /// [`DocumentIndex::all_paths`].
    pub async fn all_links(&self) -> Result<Vec<(String, String, String, Option<f64>)>> {
        let rows: Vec<(String, String, String, Option<f64>)> =
            sqlx::query_as("SELECT source_path, target_path, kind, score FROM document_links")
                .fetch_all(&self.pool)
                .await?;

        Ok(rows)
    }

    /// Distinct source paths whose `document_links` rows target `target_path`, scoped
    /// to `kind` (callers pass `"markdown"` to find only real inline-link edges, not
    /// precomputed `"semantic"` neighbors).
    ///
    /// Backs `write::write_document_move`'s "which documents must be updated"
    /// question: when `target_path` moves, every source this returns has a body that
    /// needs `find_markdown_link_occurrences` re-run and its link text rewritten. Uses
    /// `idx_document_links_target`, unlike every other query on this table, which is
    /// source-first and served by the primary key.
    pub async fn links_targeting(&self, target_path: &str, kind: &str) -> Result<Vec<String>> {
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT DISTINCT source_path FROM document_links
             WHERE target_path = ? AND kind = ?
             ORDER BY source_path",
        )
        .bind(target_path)
        .bind(kind)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|(path,)| path).collect())
    }

    /// Batch equivalent of [`Self::links_targeting`], for a caller that already knows
    /// every target it cares about up front (`write::move_directory`'s reverse-link
    /// scan, run once for every moved document) and wants one or a few round trips
    /// instead of one `links_targeting` call per target.
    ///
    /// Returns every `(target_path, source_path)` pair among `target_paths`, scoped to
    /// `kind` exactly like `links_targeting`, grouped by target: a target with no
    /// referencing sources simply has no key in the returned map (mirroring
    /// `links_targeting`'s empty-`Vec` result for the same case), and each target's
    /// sources come back distinct and sorted, matching `links_targeting`'s own
    /// `SELECT DISTINCT ... ORDER BY source_path`.
    ///
    /// Chunks `target_paths` at [`SQLITE_MAX_PARAMS_PER_QUERY`] (500) bound
    /// parameters per statement, same as [`Self::get_many`] — SQLite's default limit is
    /// 999 total bound parameters per statement, and each chunk here also binds `kind`,
    /// so 500 leaves comfortable headroom under that ceiling for a single query.
    pub async fn links_targeting_many(
        &self,
        target_paths: &[String],
        kind: &str,
    ) -> Result<HashMap<String, Vec<String>>> {
        let mut out: HashMap<String, Vec<String>> = HashMap::new();
        for chunk in target_paths.chunks(SQLITE_MAX_PARAMS_PER_QUERY) {
            if chunk.is_empty() {
                continue;
            }
            let mut builder = QueryBuilder::<Sqlite>::new(
                "SELECT DISTINCT target_path, source_path FROM document_links WHERE target_path IN (",
            );
            let mut separated = builder.separated(", ");
            for path in chunk {
                separated.push_bind(path);
            }
            builder.push(") AND kind = ");
            builder.push_bind(kind);
            builder.push(" ORDER BY target_path, source_path");
            let rows: Vec<(String, String)> =
                builder.build_query_as().fetch_all(&self.pool).await?;
            for (target_path, source_path) in rows {
                out.entry(target_path).or_default().push(source_path);
            }
        }
        Ok(out)
    }

    /// Distinct `(source_path, target_path)` pairs among `document_links` rows whose
    /// `kind` is `"markdown"` and whose `target_path` has no matching `documents` row
    /// — the one-statement query #158 sketches, promoted to a named method rather
    /// than embedding raw SQL in `main.rs`.
    ///
    /// Scoped to `kind = 'markdown'` deliberately: `"semantic"` edges are
    /// machine-inferred kNN neighbors (`ingest::update_semantic_edges`) that can
    /// legitimately point at any document in the collection, not something an
    /// author wrote and could have mistyped — including them here would report a
    /// modeling artifact as a broken link. `"markdown"` edges are author-written
    /// and therefore both meaningful to report and actually fixable.
    ///
    /// Deliberately NOT on [`DocumentIndex`]: unlike [`Self::links_out`] /
    /// [`Self::links_in`] (#157), which both `mcp::get_document` and any future
    /// trait-object caller need, this exists purely to feed `validate`'s CLI
    /// report and has exactly one caller — matching [`Self::list_all`] and
    /// [`Self::reproject_all_fields`]'s precedent of staying an inherent
    /// `StateDb` method rather than growing the trait for a single consumer.
    ///
    /// This reads a snapshot of the state DB, not the filesystem `validate`
    /// otherwise walks when checking frontmatter — the result reflects the *last
    /// successful index run*, not necessarily the files on disk right now. A
    /// target removed from the KB since, or a target file added since but not yet
    /// indexed, only shows up here (or stops showing up here) after the next
    /// reindex. The caller is responsible for saying so in whatever it renders —
    /// see `validate::broken_links_report`'s doc comment.
    ///
    /// Also does NOT filter out a target that exists on disk but was excluded
    /// from indexing (`indexing.exclude`/`exclude_files`) — this query has no
    /// filesystem access and cannot tell "never existed" from "exists but
    /// excluded" apart; both simply have no `documents` row. `validate` is
    /// responsible for that distinction (the false-positive trap #158 calls
    /// out), since only it has the KB root path to stat against.
    pub async fn broken_markdown_links(&self) -> Result<Vec<(String, String)>> {
        let rows: Vec<(String, String)> = sqlx::query_as(
            "SELECT DISTINCT source_path, target_path FROM document_links
             WHERE kind = 'markdown'
               AND target_path NOT IN (SELECT file_path FROM documents)
             ORDER BY source_path, target_path",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// Remove every edge originating from `path`, in either direction of kind.
    ///
    /// Called when a document is deleted/purged (from [`Self::delete_document`]) so a
    /// removed file's outgoing links do not linger. Deliberately does not also delete
    /// edges where `path` is the *target* — those become dangling edges that
    /// `GET /api/graph` drops at read time, which is the same self-healing behavior a
    /// renamed target gets, rather than a special case here.
    pub async fn delete_links_for(&self, path: &str) -> Result<()> {
        sqlx::query("DELETE FROM document_links WHERE source_path = ?")
            .bind(path)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// Every document's summary metadata, unfiltered, unpaged, ordered by path.
    ///
    /// Backs `GET /api/graph`'s node listing, which needs every document in one shot —
    /// unlike [`DocumentIndex::query_documents`], which is built around a caller-facing
    /// `limit`/`offset` page. Same row shape and frontmatter-parsing behavior (a
    /// malformed frontmatter JSON blob degrades to an empty object rather than failing
    /// the whole listing) as `query_documents`'s mapping.
    pub async fn all_document_summaries(&self) -> Result<Vec<DocumentSummary>> {
        let rows: Vec<DocumentRow> = sqlx::query_as(
            "SELECT file_path, title, description, mtime, indexed_at, frontmatter
             FROM documents ORDER BY file_path",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(
                |(file_path, title, description, mtime, indexed_at, frontmatter_json)| {
                    let frontmatter: Value = serde_json::from_str(&frontmatter_json)
                        .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
                    DocumentSummary {
                        file_path,
                        title,
                        description,
                        mtime,
                        indexed_at,
                        frontmatter,
                    }
                },
            )
            .collect())
    }

    /// Content hash of every document that has metadata, keyed by path.
    ///
    /// The scoped indexer (`ingest::index_paths`) now uses [`Self::get_document_hashes_many`]
    /// instead, scoped to just the paths it is processing — loading every document's
    /// hash at once does not scale to a large corpus. This whole-table equivalent is
    /// kept as a general-purpose `StateDb` accessor (parallel to [`Self::list_all`])
    /// rather than removed with its test coverage; `#[allow(dead_code)]` because
    /// nothing in production calls it now.
    #[allow(dead_code)]
    pub async fn list_document_hashes(&self) -> Result<HashMap<String, String>> {
        let rows: Vec<(String, String)> =
            sqlx::query_as("SELECT file_path, content_hash FROM documents")
                .fetch_all(&self.pool)
                .await?;

        Ok(rows.into_iter().collect())
    }

    /// Rebuild `document_fields` for every document from the stored frontmatter JSON.
    ///
    /// The escape hatch for when the projection rules change: no markdown is re-read,
    /// nothing is re-embedded, and Qdrant is untouched. Each document is read under
    /// the schema `schemas` resolves for its path (#349). Run automatically by
    /// [`Self::ensure_projection_current`] when [`PROJECTION_VERSION`] moves.
    pub async fn reproject_all_fields(
        &self,
        schemas: &crate::schema::SchemaCache,
    ) -> Result<ReprojectReport> {
        // Only the PATHS are snapshotted. Each document's frontmatter is re-read inside
        // the same transaction that rewrites its projection, so a concurrent reindex
        // can never be reverted: either it lands first and we reproject its new value,
        // or it lands after and overwrites ours. Snapshotting the frontmatter up front
        // would silently roll back any document updated mid-run — and an in-process
        // mutex cannot help here, since this runs as a separate process from `serve`.
        let paths: Vec<(String,)> =
            sqlx::query_as("SELECT file_path FROM documents ORDER BY file_path")
                .fetch_all(&self.pool)
                .await?;

        let mut report = ReprojectReport::default();

        const MAX_ATTEMPTS: usize = 5;

        for (file_path,) in paths {
            let mut outcome = ReprojectOutcome::Retry;
            let schema = schemas.resolve_for(Path::new(&file_path));

            for attempt in 1..=MAX_ATTEMPTS {
                match self.reproject_one(&file_path, schema).await {
                    Ok(result) => {
                        outcome = result;
                        break;
                    }
                    // A concurrent commit invalidated this transaction's snapshot.
                    // Retrying the SAME transaction can never succeed, but a fresh one
                    // re-reads and will — so one colliding write from the server must
                    // not abandon the rest of the run.
                    Err(e) if is_busy_or_locked(&e) && attempt < MAX_ATTEMPTS => {
                        debug!(file = %file_path, attempt, "reprojection lost a race; retrying");
                        continue;
                    }
                    Err(e) => {
                        warn!(file = %file_path, "skipping reprojection: {e:#}");
                        outcome = ReprojectOutcome::Failed;
                        break;
                    }
                }
            }

            match outcome {
                ReprojectOutcome::Done => report.reprojected += 1,
                ReprojectOutcome::Gone => {}
                ReprojectOutcome::Unparseable => report.unparseable += 1,
                ReprojectOutcome::Failed | ReprojectOutcome::Retry => report.failed += 1,
            }
        }

        let skipped = report.unparseable + report.failed;
        if skipped > 0 {
            warn!("{skipped} document(s) skipped during reprojection; see logs above");
        }

        Ok(report)
    }

    /// Reproject one document, re-reading its frontmatter inside the write transaction.
    async fn reproject_one(
        &self,
        file_path: &str,
        schema: &crate::schema::ResolvedSchema,
    ) -> Result<ReprojectOutcome, sqlx::Error> {
        let mut tx = self.pool.begin().await?;

        let row: Option<(String,)> =
            sqlx::query_as("SELECT frontmatter FROM documents WHERE file_path = ?")
                .bind(file_path)
                .fetch_optional(&mut *tx)
                .await?;
        let Some((frontmatter_json,)) = row else {
            // Deleted while we were working; nothing to reproject.
            return Ok(ReprojectOutcome::Gone);
        };

        // One unparseable row must not abandon every document after it — this is the
        // designated repair tool for exactly that class of problem.
        let Ok(frontmatter) = serde_json::from_str::<HashMap<String, Value>>(&frontmatter_json)
        else {
            warn!(file = %file_path, "stored frontmatter is unparseable; leaving it alone");
            return Ok(ReprojectOutcome::Unparseable);
        };

        write_projection(&mut tx, file_path, &frontmatter, schema).await?;

        tx.commit().await?;
        Ok(ReprojectOutcome::Done)
    }

    /// Append the shared `WHERE` clause for a document query.
    ///
    /// Field paths and filter values are always bound parameters — under this schema a
    /// field name is row data, never a SQL identifier, so caller-supplied field names
    /// cannot reach the statement text.
    fn push_where(builder: &mut QueryBuilder<'_, Sqlite>, query: &DocumentQuery) {
        builder.push(" WHERE 1 = 1");

        if let Some(needle) = &query.path_prefix {
            builder.push(" AND d.file_path LIKE ");
            builder.push_bind(escape_like_substring(needle));
            builder.push(" ESCAPE '\\'");
        }

        if let Some(after) = query.mtime_after {
            builder.push(" AND d.mtime >= ");
            builder.push_bind(after);
        }
        if let Some(before) = query.mtime_before {
            builder.push(" AND d.mtime <= ");
            builder.push_bind(before);
        }

        for (field, filter) in &query.filters {
            // `title` and `description` live in dedicated columns and are deliberately
            // excluded from the projection, so filter them directly rather than letting
            // them silently match nothing.
            if let Some(column) = promoted_column(field) {
                match filter {
                    FieldFilter::AllOf(values) if values.len() != 1 => {
                        // These columns hold one scalar, so requiring two distinct
                        // values is unsatisfiable. Falling back to IN would silently
                        // widen the query to any-of, which is the opposite of asked.
                        builder.push(" AND 0 = 1");
                    }
                    FieldFilter::AnyOf(values) | FieldFilter::AllOf(values) => {
                        if values.is_empty() {
                            builder.push(" AND 0 = 1");
                            continue;
                        }
                        builder.push(" AND ");
                        builder.push(column);
                        builder.push(" IN (");
                        let mut separated = builder.separated(", ");
                        for value in values {
                            separated.push_bind(value.clone());
                        }
                        builder.push(")");
                    }
                    FieldFilter::Range { .. } | FieldFilter::Elem(_) => {
                        // Prose columns have no numeric ordering and no elements; an
                        // unsatisfiable clause is more honest than ignoring the filter.
                        builder.push(" AND 0 = 1");
                    }
                }
                continue;
            }

            match filter {
                FieldFilter::AnyOf(values) => {
                    if values.is_empty() {
                        // An empty set matches nothing; say so explicitly rather than
                        // silently dropping the filter.
                        builder.push(" AND 0 = 1");
                        continue;
                    }
                    builder.push(
                        " AND EXISTS (SELECT 1 FROM document_fields f \
                         WHERE f.file_path = d.file_path AND f.field = ",
                    );
                    builder.push_bind(field.clone());
                    builder.push(" AND f.value_text IN (");
                    let mut separated = builder.separated(", ");
                    for value in values {
                        separated.push_bind(value.clone());
                    }
                    builder.push("))");
                }
                FieldFilter::AllOf(values) => {
                    if values.is_empty() {
                        // Mirror the AnyOf convention: an empty required set is an
                        // unsatisfiable filter, not an absent one. Emitting no clause
                        // would silently widen the query to match everything.
                        builder.push(" AND 0 = 1");
                        continue;
                    }
                    for value in values {
                        builder.push(
                            " AND EXISTS (SELECT 1 FROM document_fields f \
                             WHERE f.file_path = d.file_path AND f.field = ",
                        );
                        builder.push_bind(field.clone());
                        builder.push(" AND f.value_text = ");
                        builder.push_bind(value.clone());
                        builder.push(")");
                    }
                }
                FieldFilter::Range { gte, lte, gt, lt } => {
                    builder.push(
                        " AND EXISTS (SELECT 1 FROM document_fields f \
                         WHERE f.file_path = d.file_path AND f.field = ",
                    );
                    builder.push_bind(field.clone());
                    builder.push(" AND f.value_num IS NOT NULL");
                    for (op, bound) in [
                        (" AND f.value_num >= ", gte),
                        (" AND f.value_num <= ", lte),
                        (" AND f.value_num > ", gt),
                        (" AND f.value_num < ", lt),
                    ] {
                        if let Some(value) = bound {
                            builder.push(op);
                            builder.push_bind(*value);
                        }
                    }
                    builder.push(")");
                }
                FieldFilter::Elem(conditions) => push_elem_filter(builder, conditions),
            }
        }
    }

    /// Number of documents carrying metadata. Diverging from [`Self::count`] means the
    /// metadata index needs a backfill pass.
    pub async fn document_count(&self) -> Result<i64> {
        let row: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM documents")
            .fetch_one(&self.pool)
            .await?;

        Ok(row.0)
    }

    /// Fields worth breaking down, with how many distinct values each takes,
    /// widest document coverage first.
    ///
    /// Two decisions matter here, and both are in SQL because a caller filtering
    /// afterwards would get the wrong rows:
    ///
    /// - `max_distinct` bounds *value* cardinality. A field with three values
    ///   (`status`) is a useful histogram; one with several hundred is noise.
    /// - Ordering is by how many documents carry the field, not by how many values it
    ///   takes. A scoped schema can declare twenty fields that apply to one folder;
    ///   ordering by value count floats all of them above `type` and `status`, and the
    ///   fields that describe the whole knowledge base fall off the end of `limit`.
    ///
    /// Fields of a list's elements (`npcs[].role`, #349) are left out: a per-value
    /// document count of one sub-field says little about a list of records, and
    /// would crowd out the document-level fields `/status` is for.
    pub async fn breakdown_fields(
        &self,
        max_distinct: i64,
        limit: i64,
    ) -> Result<Vec<(String, i64)>> {
        // SQLite reads a negative LIMIT as "no limit".
        let limit = limit.max(0);
        let rows: Vec<(String, i64, i64)> = sqlx::query_as(
            "SELECT field, COUNT(DISTINCT value_text) AS n, COUNT(DISTINCT file_path) AS docs \
             FROM document_fields WHERE instr(field, '[]') = 0 \
             GROUP BY field HAVING n <= ? \
             ORDER BY docs DESC, n DESC, field ASC LIMIT ?",
        )
        .bind(max_distinct)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|(f, n, _docs)| (f, n)).collect())
    }

    /// Document counts per value of `field`, most common first, over every
    /// document — [`Self::top_values`] with no directory.
    ///
    /// Counts distinct documents rather than rows: multi-valued fields such as `tags`
    /// store one row per value, so a `COUNT(*)` here would be counting projections
    /// rather than documents the moment a field ever repeats within one document.
    pub async fn count_by_field(&self, field: &str, limit: i64) -> Result<Vec<(String, i64)>> {
        self.top_values(field, None, limit).await
    }

    /// Whether any document carries `field` — `search`'s unknown-filter-field
    /// check, run only for a field no schema declares.
    pub async fn field_in_use(&self, field: &str) -> Result<bool> {
        let row: Option<(i64,)> =
            sqlx::query_as("SELECT 1 FROM document_fields WHERE field = ? LIMIT 1")
                .bind(field)
                .fetch_optional(&self.pool)
                .await?;
        Ok(row.is_some())
    }

    /// Which of `values` some document actually uses for `field` — `search`'s
    /// closed-value check accepts these even when no governing schema lists
    /// them, so it can never reject a filter that would have matched.
    pub async fn values_present(
        &self,
        field: &str,
        values: &[String],
    ) -> Result<std::collections::HashSet<String>> {
        if values.is_empty() {
            return Ok(Default::default());
        }
        let mut builder = sqlx::QueryBuilder::<sqlx::Sqlite>::new(
            "SELECT DISTINCT value_text FROM document_fields WHERE field = ",
        );
        builder.push_bind(field);
        builder.push(" AND value_text IN (");
        let mut separated = builder.separated(", ");
        for value in values {
            separated.push_bind(value.clone());
        }
        builder.push(")");
        let rows: Vec<(String,)> = builder.build_query_as().fetch_all(&self.pool).await?;
        Ok(rows.into_iter().map(|(v,)| v).collect())
    }

    /// Distinct field names documents use, most widely used first, optionally
    /// only among documents under `dir_prefix` (a KB-relative directory, no
    /// trailing slash).
    pub async fn fields_in_use(
        &self,
        dir_prefix: Option<&str>,
        limit: i64,
    ) -> Result<Vec<(String, i64)>> {
        let pattern = dir_prefix.map(escape_like_dir_prefix);
        let rows: Vec<(String, i64)> = sqlx::query_as(
            "SELECT field, COUNT(DISTINCT file_path) AS n FROM document_fields \
             WHERE (?1 IS NULL OR file_path LIKE ?1 ESCAPE '\\') \
             GROUP BY field ORDER BY n DESC, field ASC LIMIT ?2",
        )
        .bind(pattern)
        .bind(limit.max(0))
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// The most-used values of `field`, with document counts, optionally only
    /// among documents under `dir_prefix` — `get_schema`'s `values_in_use`.
    pub async fn top_values(
        &self,
        field: &str,
        dir_prefix: Option<&str>,
        limit: i64,
    ) -> Result<Vec<(String, i64)>> {
        let pattern = dir_prefix.map(escape_like_dir_prefix);
        // SQLite reads a negative LIMIT as "no limit", so a caller passing one through
        // from arithmetic would silently get the whole vocabulary instead of a page.
        let limit = limit.max(0);
        let rows: Vec<(String, i64)> = sqlx::query_as(
            "SELECT value_text, COUNT(DISTINCT file_path) AS n FROM document_fields \
             WHERE field = ?1 AND (?2 IS NULL OR file_path LIKE ?2 ESCAPE '\\') \
             GROUP BY value_text ORDER BY n DESC, value_text ASC LIMIT ?3",
        )
        .bind(field)
        .bind(pattern)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// Document counts per top-level directory, derived from stored paths.
    ///
    /// Deliberately sourced from the metadata index rather than a directory walk: this
    /// answers "what has actually been indexed", which is the question a status report
    /// is for. A directory present on disk but absent here is precisely the discrepancy
    /// worth seeing.
    pub async fn area_counts(&self) -> Result<Vec<(String, i64)>> {
        let rows: Vec<(String, i64)> = sqlx::query_as(
            "SELECT CASE WHEN instr(file_path, '/') > 0 \
                         THEN substr(file_path, 1, instr(file_path, '/') - 1) \
                         ELSE '' END AS area, \
                    COUNT(*) AS n \
             FROM documents GROUP BY area ORDER BY n DESC, area ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }
}

/// One row-level test inside a correlated list filter: a single `document_fields`
/// row of the element must satisfy it.
enum ElemAtom<'a> {
    In(&'a [String]),
    Eq(&'a String),
    Range([(&'static str, Option<f64>); 4]),
}

/// Append a [`FieldFilter::Elem`] clause (#349). Each sub-condition becomes one or
/// more row tests (an `all_of` needs one row per value, as it does outside a list);
/// the first test is an `EXISTS` over the document's rows, and every further test an
/// `EXISTS` correlated to that row's `elem`, so all of them hold within ONE element:
///
/// ```sql
/// EXISTS (SELECT 1 FROM document_fields e0
///         WHERE e0.file_path = d.file_path AND e0.field = 'npcs[].role' AND ...
///           AND EXISTS (SELECT 1 FROM document_fields e1
///                       WHERE e1.file_path = e0.file_path AND e1.elem = e0.elem
///                         AND e1.field = 'npcs[].level' AND ...))
/// ```
///
/// An empty condition list, an empty set, or a nested `Elem` (none of which the
/// tool parser lets through) is unsatisfiable rather than dropped.
fn push_elem_filter(builder: &mut QueryBuilder<'_, Sqlite>, conditions: &[(String, FieldFilter)]) {
    let mut atoms: Vec<(&String, ElemAtom<'_>)> = Vec::new();
    for (field, filter) in conditions {
        match filter {
            FieldFilter::AnyOf(values) if !values.is_empty() => {
                atoms.push((field, ElemAtom::In(values)));
            }
            FieldFilter::AllOf(values) if !values.is_empty() => {
                atoms.extend(values.iter().map(|v| (field, ElemAtom::Eq(v))));
            }
            FieldFilter::Range { gte, lte, gt, lt } => atoms.push((
                field,
                ElemAtom::Range([(">=", *gte), ("<=", *lte), (">", *gt), ("<", *lt)]),
            )),
            FieldFilter::AnyOf(_) | FieldFilter::AllOf(_) | FieldFilter::Elem(_) => {
                builder.push(" AND 0 = 1");
                return;
            }
        }
    }
    let Some(((first_field, first), rest)) = atoms.split_first() else {
        builder.push(" AND 0 = 1");
        return;
    };

    fn push_atom(builder: &mut QueryBuilder<'_, Sqlite>, alias: &str, atom: &ElemAtom<'_>) {
        match atom {
            ElemAtom::In(values) => {
                builder.push(format!(" AND {alias}.value_text IN ("));
                let mut separated = builder.separated(", ");
                for value in values.iter() {
                    separated.push_bind(value.clone());
                }
                builder.push(")");
            }
            ElemAtom::Eq(value) => {
                builder.push(format!(" AND {alias}.value_text = "));
                builder.push_bind((*value).clone());
            }
            ElemAtom::Range(bounds) => {
                builder.push(format!(" AND {alias}.value_num IS NOT NULL"));
                for (op, bound) in bounds {
                    if let Some(value) = bound {
                        builder.push(format!(" AND {alias}.value_num {op} "));
                        builder.push_bind(*value);
                    }
                }
            }
        }
    }

    builder.push(
        " AND EXISTS (SELECT 1 FROM document_fields e0 \
         WHERE e0.file_path = d.file_path AND e0.field = ",
    );
    builder.push_bind((*first_field).clone());
    push_atom(builder, "e0", first);
    for (i, (field, atom)) in rest.iter().enumerate() {
        let alias = format!("e{}", i + 1);
        builder.push(format!(
            " AND EXISTS (SELECT 1 FROM document_fields {alias} \
             WHERE {alias}.file_path = e0.file_path AND {alias}.elem = e0.elem \
             AND {alias}.field = "
        ));
        builder.push_bind((*field).clone());
        push_atom(builder, &alias, atom);
        builder.push(")");
    }
    builder.push(")");
}

/// The `documents` column backing a promoted field, if it is one.
fn promoted_column(field: &str) -> Option<&'static str> {
    match field {
        "title" => Some("d.title"),
        "description" => Some("d.description"),
        _ => None,
    }
}

/// Escape a caller-supplied path fragment's LIKE metacharacters.
///
/// Real KB paths routinely contain `_`, which is a single-character LIKE wildcard —
/// left unescaped, `lifestyle/my_notes/` would also match `lifestyle/myXnotes/`.
///
/// Returns only the escaped needle; the caller adds its own wildcards, which is what
/// keeps every wildcard in the final pattern non-caller-controlled.
fn escape_like_metachars(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '%' => out.push_str("\\%"),
            '_' => out.push_str("\\_"),
            c => out.push(c),
        }
    }
    out
}

/// Escape a KB-relative directory (no trailing slash) into the `LIKE ... ESCAPE '\'`
/// pattern for every path beneath it, as `fields_in_use` and `top_values` filter on.
///
/// The `/%` is added after escaping, so it is the only wildcard and is never
/// caller-controlled — the same rule [`escape_like_substring`] follows.
fn escape_like_dir_prefix(dir: &str) -> String {
    format!("{}/%", escape_like_metachars(dir))
}

/// Escape a caller-supplied path fragment for a *substring* `LIKE ... ESCAPE '\'`.
///
/// #182: `path_prefix` matches anywhere in the KB-relative path, not just at its
/// start, so an agent can find a document from a half-remembered fragment
/// (`stir_fr`) or a mid-path one. SQLite's `LIKE` is ASCII-case-insensitive, which
/// is where this match's case-insensitivity comes from — there is no explicit
/// folding here, and non-ASCII characters therefore stay case-sensitive.
fn escape_like_substring(raw: &str) -> String {
    // Our wildcards, added after escaping so they are never caller-controlled.
    format!("%{}%", escape_like_metachars(raw))
}

/// Look up a dot-path inside parsed frontmatter.
fn value_at_dotpath<'a>(root: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = root;
    for segment in path.split('.') {
        current = current.as_object()?.get(segment)?;
    }
    Some(current)
}

/// Reduce frontmatter to the requested dot-paths, preserving nesting so the shape a
/// caller sees matches the shape in the document.
fn project_fields(frontmatter: &Value, fields: &[String]) -> Value {
    let mut out = serde_json::Map::new();
    for field in fields {
        let Some(value) = value_at_dotpath(frontmatter, field) else {
            continue;
        };
        let mut cursor = &mut out;
        let segments: Vec<&str> = field.split('.').collect();
        for segment in &segments[..segments.len() - 1] {
            cursor = cursor
                .entry(*segment)
                .or_insert_with(|| Value::Object(serde_json::Map::new()))
                .as_object_mut()
                .expect("intermediate segments are always objects");
        }
        cursor.insert(segments[segments.len() - 1].to_string(), value.clone());
    }
    Value::Object(out)
}

/// A `documents` row as selected by the listing query, in SELECT order:
/// file_path, title, description, mtime, indexed_at, frontmatter.
type DocumentRow = (String, Option<String>, Option<String>, i64, String, String);

impl DocumentIndex for StateDb {
    async fn query_documents(&self, query: &DocumentQuery) -> Result<DocumentQueryResult> {
        // Count and page run on one connection inside a transaction so `total` and the
        // returned rows reflect the same committed snapshot. Without this a concurrent
        // reindex could make has_more disagree with the page.
        let mut tx = self.pool.begin().await?;

        let mut count_builder: QueryBuilder<Sqlite> =
            QueryBuilder::new("SELECT COUNT(*) FROM documents d");
        Self::push_where(&mut count_builder, query);
        let total: i64 = count_builder
            .build_query_scalar()
            .fetch_one(&mut *tx)
            .await?;

        let mut page_builder: QueryBuilder<Sqlite> = QueryBuilder::new(
            "SELECT d.file_path, d.title, d.description, d.mtime, d.indexed_at, d.frontmatter \
             FROM documents d",
        );
        Self::push_where(&mut page_builder, query);
        page_builder.push(" ORDER BY ");
        page_builder.push(query.order_by.column());
        page_builder.push(if query.order_desc { " DESC" } else { " ASC" });
        // Mandatory tiebreaker: ties on mtime are common (many docs share a reindex
        // run) and would otherwise make LIMIT/OFFSET paging non-deterministic.
        if query.order_by != OrderBy::Path {
            page_builder.push(", d.file_path ASC");
        }
        page_builder.push(" LIMIT ");
        page_builder.push_bind(query.limit as i64);
        page_builder.push(" OFFSET ");
        page_builder.push_bind(query.offset as i64);

        let rows: Vec<DocumentRow> = page_builder.build_query_as().fetch_all(&mut *tx).await?;

        tx.commit().await?;

        let documents = rows
            .into_iter()
            .map(
                |(file_path, title, description, mtime, indexed_at, frontmatter_json)| {
                    let frontmatter: Value = serde_json::from_str(&frontmatter_json)
                        .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
                    let frontmatter = match &query.fields {
                        Some(fields) => project_fields(&frontmatter, fields),
                        None => frontmatter,
                    };
                    DocumentSummary {
                        file_path,
                        title,
                        description,
                        mtime,
                        indexed_at,
                        frontmatter,
                    }
                },
            )
            .collect();

        Ok(DocumentQueryResult {
            documents,
            total: total.max(0) as u64,
        })
    }

    async fn all_paths(&self) -> Result<Vec<String>> {
        let rows: Vec<(String,)> = sqlx::query_as("SELECT file_path FROM documents")
            .fetch_all(&self.pool)
            .await?;
        Ok(rows.into_iter().map(|(file_path,)| file_path).collect())
    }

    async fn paths_matching(&self, needle: &str, cap: u64) -> Result<PathMatches> {
        let pattern = escape_like_substring(needle);

        // Count and page share one transaction so `total` and `paths` reflect the same
        // committed snapshot — same reasoning as `query_documents`.
        let mut tx = self.pool.begin().await?;

        let total: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM documents WHERE file_path LIKE ? ESCAPE '\\'")
                .bind(&pattern)
                .fetch_one(&mut *tx)
                .await?;

        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT file_path FROM documents WHERE file_path LIKE ? ESCAPE '\\' \
             ORDER BY file_path ASC LIMIT ?",
        )
        .bind(&pattern)
        .bind(cap as i64)
        .fetch_all(&mut *tx)
        .await?;

        tx.commit().await?;

        Ok(PathMatches {
            paths: rows.into_iter().map(|(file_path,)| file_path).collect(),
            total: total.max(0) as u64,
        })
    }

    async fn get_summaries_by_paths(
        &self,
        paths: &[String],
        fields: Option<&[String]>,
    ) -> Result<Vec<DocumentSummary>> {
        let mut out = Vec::with_capacity(paths.len());
        for chunk in paths.chunks(SQLITE_MAX_PARAMS_PER_QUERY) {
            if chunk.is_empty() {
                continue;
            }
            let mut builder: QueryBuilder<Sqlite> = QueryBuilder::new(
                "SELECT file_path, title, description, mtime, indexed_at, frontmatter \
                 FROM documents WHERE file_path IN (",
            );
            let mut separated = builder.separated(", ");
            for path in chunk {
                separated.push_bind(path);
            }
            builder.push(")");

            let rows: Vec<DocumentRow> = builder.build_query_as().fetch_all(&self.pool).await?;
            out.extend(rows.into_iter().map(
                |(file_path, title, description, mtime, indexed_at, frontmatter_json)| {
                    let frontmatter: Value = serde_json::from_str(&frontmatter_json)
                        .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
                    let frontmatter = match fields {
                        Some(fields) => project_fields(&frontmatter, fields),
                        None => frontmatter,
                    };
                    DocumentSummary {
                        file_path,
                        title,
                        description,
                        mtime,
                        indexed_at,
                        frontmatter,
                    }
                },
            ));
        }
        Ok(out)
    }

    async fn links_out(&self, source_path: &str, limit: u64) -> Result<LinkPage<OutboundLink>> {
        // Count and page share one transaction, same rationale as `query_documents`:
        // without it, a concurrent reindex landing between the two statements could
        // make `total` disagree with the page it is supposed to describe.
        let mut tx = self.pool.begin().await?;

        let total: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM document_links WHERE source_path = ?")
                .bind(source_path)
                .fetch_one(&mut *tx)
                .await?;

        // LEFT JOIN documents rather than a second query per row: `exists` is
        // just "did the join find a match", computed as part of the same page
        // fetch instead of an all_paths()-style full-table load per call.
        let rows: Vec<(String, String, Option<f64>, Option<String>)> = sqlx::query_as(
            "SELECT l.target_path, l.kind, l.score, d.file_path
             FROM document_links l
             LEFT JOIN documents d ON d.file_path = l.target_path
             WHERE l.source_path = ?
             ORDER BY l.kind, l.target_path
             LIMIT ?",
        )
        .bind(source_path)
        .bind(limit as i64)
        .fetch_all(&mut *tx)
        .await?;

        tx.commit().await?;

        let links = rows
            .into_iter()
            .map(|(target_path, kind, score, matched)| OutboundLink {
                target_path,
                kind,
                score,
                exists: matched.is_some(),
            })
            .collect();

        Ok(LinkPage {
            links,
            total: total.max(0) as u64,
        })
    }

    async fn links_in(&self, target_path: &str, limit: u64) -> Result<LinkPage<InboundLink>> {
        let mut tx = self.pool.begin().await?;

        let total: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM document_links WHERE target_path = ?")
                .bind(target_path)
                .fetch_one(&mut *tx)
                .await?;

        // Uses `idx_document_links_target`, same as `links_targeting` — the
        // reverse lookup is not served by the (source_path, target_path, kind)
        // primary key.
        let rows: Vec<(String, String, Option<f64>)> = sqlx::query_as(
            "SELECT source_path, kind, score FROM document_links
             WHERE target_path = ?
             ORDER BY kind, source_path
             LIMIT ?",
        )
        .bind(target_path)
        .bind(limit as i64)
        .fetch_all(&mut *tx)
        .await?;

        tx.commit().await?;

        let links = rows
            .into_iter()
            .map(|(source_path, kind, score)| InboundLink {
                source_path,
                kind,
                score,
            })
            .collect();

        Ok(LinkPage {
            links,
            total: total.max(0) as u64,
        })
    }
}

/// The shared fixture of the SQLite/Qdrant filter parity checks (#350): one typed
/// schema, documents exercising every declared kind (an integral float under
/// `integer`, timestamps at several offsets and with fractional seconds, a list of
/// objects whose elements split the conditions of a correlated filter), and a
/// matrix of `search` filters with the documents each must find. The offline half
/// (`state::tests`, evaluating the lowered conditions against the production
/// payload) runs in the fast tier; the live half
/// (`qdrant::tests::sqlite_and_qdrant_filters_agree_on_the_typed_fixture`) seeds a
/// real Qdrant through the same builders in the `qdrant-integration` tier.
#[cfg(test)]
pub(crate) mod filter_parity {
    use super::*;
    use crate::schema::SchemaCache;
    use std::path::Path;

    /// The root schema: every field `indexed`, so query mode accepts each filter.
    pub const SCHEMA_YAML: &str = "fields:
  type:
    type: enum
    values: [guide, recipe, reference]
    indexed: true
  tags:
    type: list
    indexed: true
    items:
      type: text
  count:
    type: integer
    indexed: true
  rating:
    type: number
    indexed: true
  tested:
    type: boolean
    indexed: true
  published:
    type: date
    indexed: true
  seen:
    type: timestamp
    indexed: true
  npcs:
    type: list
    items:
      type: object
      fields:
        role:
          type: enum
          values: [guard, cook]
          indexed: true
        level:
          type: integer
          indexed: true
";

    /// The documents, by path.
    pub fn docs() -> Vec<(&'static str, HashMap<String, Value>)> {
        let doc = |v: Value| -> HashMap<String, Value> {
            v.as_object().unwrap().clone().into_iter().collect()
        };
        vec![
            (
                "a.md",
                doc(serde_json::json!({
                    "title": "A", "type": "guide", "tags": ["x", "y"], "count": 3,
                    "rating": 4.5, "tested": true, "published": "2024-01-02",
                    "seen": "2024-01-02T10:00:00+02:00",
                    "npcs": [{"role": "guard", "level": 1}, {"role": "cook", "level": 5}]
                })),
            ),
            (
                "b.md",
                doc(serde_json::json!({
                    "title": "B", "type": "recipe", "tags": ["y"], "count": 3.0,
                    "rating": 2.0, "tested": false, "published": "2024-01-03",
                    "seen": "2024-01-01T23:30:00-01:00",
                    "npcs": [{"role": "cook", "level": 2}, {"role": "guard", "level": 5}]
                })),
            ),
            (
                "c.md",
                doc(serde_json::json!({
                    "title": "C", "type": "reference", "tags": ["z"], "count": 7,
                    "rating": 3.25, "tested": true, "published": "2023-12-31",
                    "seen": "2024-01-02T00:00:00.5Z", "npcs": []
                })),
            ),
            (
                "d.md",
                doc(serde_json::json!({ "title": "D", "type": "guide", "count": 12 })),
            ),
        ]
    }

    /// `(filter key, filter JSON, documents it must find)`, in `search`'s `filters`
    /// syntax — parsed by the production parser (`mcp::parse_field_filter`).
    pub fn cases() -> Vec<(&'static str, Value, Vec<&'static str>)> {
        use serde_json::json;
        vec![
            ("type", json!("guide"), vec!["a.md", "d.md"]),
            ("type", json!(["recipe", "reference"]), vec!["b.md", "c.md"]),
            ("tags", json!({"all_of": ["x", "y"]}), vec!["a.md"]),
            ("tags", json!(["y", "z"]), vec!["a.md", "b.md", "c.md"]),
            // `3.0` is written as the integer 3 in the payload.
            ("count", json!(3), vec!["a.md", "b.md"]),
            ("count", json!({"gte": 5}), vec!["c.md", "d.md"]),
            ("rating", json!({"gte": 3}), vec!["a.md", "c.md"]),
            ("rating", json!({"lt": 2.5}), vec!["b.md"]),
            ("tested", json!(true), vec!["a.md", "c.md"]),
            ("tested", json!(false), vec!["b.md"]),
            ("published", json!("2024-01-02"), vec!["a.md"]),
            (
                "published",
                json!({"gte": "2024-01-02"}),
                vec!["a.md", "b.md"],
            ),
            (
                "published",
                json!({"lte": "2024-01-02"}),
                vec!["a.md", "c.md"],
            ),
            // a: 08:00Z, b: 00:30Z, c: 00:00:00.5Z — all on 2024-01-02 in UTC.
            (
                "seen",
                json!({"gte": "2024-01-02"}),
                vec!["a.md", "b.md", "c.md"],
            ),
            (
                "seen",
                json!({"lt": "2024-01-02T08:00:00Z"}),
                vec!["b.md", "c.md"],
            ),
            (
                "seen",
                json!({"lte": "2024-01-02T10:00:00+02:00"}),
                vec!["a.md", "b.md", "c.md"],
            ),
            (
                "seen",
                json!({"gt": "2024-01-02T00:00:00.5Z"}),
                vec!["a.md", "b.md"],
            ),
            (
                "seen",
                json!({"gte": "2024-01-02T00:00:00.5Z"}),
                vec!["a.md", "b.md", "c.md"],
            ),
            // Correlated: both conditions within one element. a's guard is level 1.
            (
                "npcs[]",
                json!({"role": "guard", "level": {"gte": 3}}),
                vec!["b.md"],
            ),
            (
                "npcs[]",
                json!({"role": "guard", "level": {"lte": 1}}),
                vec!["a.md"],
            ),
            (
                "npcs[]",
                json!({"role": "cook", "level": {"lte": 2}}),
                vec!["b.md"],
            ),
            // Uncorrelated: any element.
            ("npcs[].role", json!("guard"), vec!["a.md", "b.md"]),
            ("npcs[].level", json!({"gte": 5}), vec!["a.md", "b.md"]),
        ]
    }

    /// The schema cache built from [`SCHEMA_YAML`] at the root of `root`.
    pub fn schemas(root: &Path) -> SchemaCache {
        std::fs::write(root.join(crate::schema::SCHEMA_FILE_NAME), SCHEMA_YAML).unwrap();
        SchemaCache::build_for_test(root, &crate::config::FrontmatterConfig::default())
    }

    /// Every indexed field's kind, as `mcp::build_query_conditions` passes it.
    pub fn indexed(schemas: &SchemaCache) -> HashMap<String, crate::qdrant::IndexKind> {
        schemas
            .all_indexed_fields()
            .into_iter()
            .map(|f| (f.name, f.kind))
            .collect()
    }

    /// A state DB at `db_path` holding every document's production projection.
    pub async fn seed_db(db_path: &Path, schemas: &SchemaCache) -> StateDb {
        let db = StateDb::new(db_path).await.unwrap();
        for (path, fm) in docs() {
            db.upsert_document_metadata_with(
                path,
                &fm,
                schemas.resolve_for(Path::new(path)),
                100,
                "h",
                1,
            )
            .await
            .unwrap();
        }
        db
    }

    /// The frontmatter part of `path`'s production Qdrant payload.
    pub fn payload(
        path: &str,
        fm: &HashMap<String, Value>,
        schemas: &SchemaCache,
    ) -> HashMap<String, Value> {
        crate::ingest::frontmatter_payload(
            fm,
            path,
            &Default::default(),
            schemas.resolve_for(Path::new(path)),
        )
    }

    /// `(field, filter)` parsed as `search` parses it.
    pub fn parse(field: &str, raw: &Value) -> Vec<(String, FieldFilter)> {
        vec![(
            field.to_string(),
            crate::mcp::parse_field_filter(field, raw).unwrap(),
        )]
    }

    /// The paths the metadata index (enumeration mode) finds for `filters`, sorted.
    pub async fn sqlite_paths(db: &StateDb, filters: Vec<(String, FieldFilter)>) -> Vec<String> {
        let mut paths: Vec<String> = db
            .query_documents(&DocumentQuery {
                filters,
                ..Default::default()
            })
            .await
            .unwrap()
            .documents
            .into_iter()
            .map(|d| d.file_path)
            .collect();
        paths.sort();
        paths
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qdrant;
    use tempfile::TempDir;

    async fn test_db() -> (StateDb, TempDir) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.db");
        let db = StateDb::new(&path).await.unwrap();
        (db, dir)
    }

    #[tokio::test]
    async fn upsert_and_get() {
        let (db, _dir) = test_db().await;
        db.upsert("test.md", "abc123", 3, "", "", 0, 0)
            .await
            .unwrap();
        let entry = db.get("test.md").await.unwrap().unwrap();
        assert_eq!(entry.file_path, "test.md");
        assert_eq!(entry.content_hash, "abc123");
        assert_eq!(entry.chunk_count, 3);
    }

    #[tokio::test]
    async fn upsert_replaces() {
        let (db, _dir) = test_db().await;
        db.upsert("test.md", "hash1", 2, "", "", 0, 0)
            .await
            .unwrap();
        db.upsert("test.md", "hash2", 5, "", "", 0, 0)
            .await
            .unwrap();
        let entry = db.get("test.md").await.unwrap().unwrap();
        assert_eq!(entry.content_hash, "hash2");
        assert_eq!(entry.chunk_count, 5);
    }

    #[tokio::test]
    async fn delete_removes() {
        let (db, _dir) = test_db().await;
        db.upsert("test.md", "hash", 1, "", "", 0, 0).await.unwrap();
        db.delete("test.md").await.unwrap();
        assert!(db.get("test.md").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn list_and_count() {
        let (db, _dir) = test_db().await;
        db.upsert("a.md", "h1", 1, "", "", 0, 0).await.unwrap();
        db.upsert("b.md", "h2", 2, "", "", 0, 0).await.unwrap();
        assert_eq!(db.count().await.unwrap(), 2);
        let all = db.list_all().await.unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].file_path, "a.md"); // sorted
    }

    #[tokio::test]
    async fn total_chunk_count_sums_across_files() {
        let (db, _dir) = test_db().await;
        // Empty table: COALESCE must turn SQL NULL into 0, not surface as an error or
        // as a bare NULL that panics the row-mapping — this is the case that motivates
        // the COALESCE in the query at all.
        assert_eq!(db.total_chunk_count().await.unwrap(), 0);

        db.upsert("a.md", "h1", 3, "", "", 0, 0).await.unwrap();
        db.upsert("b.md", "h2", 5, "", "", 0, 0).await.unwrap();
        assert_eq!(db.total_chunk_count().await.unwrap(), 8);

        // A re-upsert (simulating a shrinking file) replaces, not adds to, the prior
        // chunk_count — the sum must track the replacement, not accumulate it.
        db.upsert("a.md", "h1v2", 1, "", "", 0, 0).await.unwrap();
        assert_eq!(db.total_chunk_count().await.unwrap(), 6);
    }

    #[tokio::test]
    async fn clear_removes_all() {
        let (db, _dir) = test_db().await;
        db.upsert("a.md", "h1", 1, "", "", 0, 0).await.unwrap();
        db.upsert("b.md", "h2", 2, "", "", 0, 0).await.unwrap();
        db.clear().await.unwrap();
        assert_eq!(db.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn get_missing_returns_none() {
        let (db, _dir) = test_db().await;
        assert!(db.get("nonexistent.md").await.unwrap().is_none());
    }

    /// Regression: WAL journal mode must be enabled (#9)
    #[tokio::test]
    async fn wal_mode_enabled() {
        let (db, _dir) = test_db().await;
        let row: (String,) = sqlx::query_as("PRAGMA journal_mode")
            .fetch_one(&db.pool)
            .await
            .unwrap();
        assert_eq!(row.0, "wal");
    }

    /// Regression: concurrent writes must not fail with SQLITE_BUSY (#9)
    #[tokio::test]
    async fn concurrent_writes_succeed() {
        let (db, _dir) = test_db().await;
        let mut handles = Vec::new();
        // Share the pool across tasks via Arc
        let pool = db.pool.clone();
        for i in 0..10 {
            let pool = pool.clone();
            handles.push(tokio::spawn(async move {
                let path = format!("file_{}.md", i);
                sqlx::query(
                    "INSERT OR REPLACE INTO indexed_files (file_path, content_hash, chunk_count, indexed_at) VALUES (?, ?, ?, datetime('now'))",
                )
                .bind(&path)
                .bind("hash")
                .bind(1i64)
                .execute(&pool)
                .await
                .unwrap();
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(db.count().await.unwrap(), 10);
    }

    /// Regression: deleting state entry after upsert failure ensures re-processing (#4)
    #[tokio::test]
    async fn delete_after_failure_allows_reprocessing() {
        let (db, _dir) = test_db().await;
        // Simulate: file was indexed with hash1
        db.upsert("doc.md", "hash1", 3, "", "", 0, 0).await.unwrap();

        // Simulate: upsert to Qdrant fails, so we delete the state entry
        // (this is what ingest.rs now does on failure)
        db.delete("doc.md").await.unwrap();

        // On next run, the file should appear as new (not in state DB)
        assert!(db.get("doc.md").await.unwrap().is_none());
    }

    // -- schema migration ----------------------------------------------------

    /// Build a state DB with the pre-`schema_hash` `indexed_files` schema and rows in
    /// it, exactly as a deployment running the previous release would have.
    async fn legacy_db(dir: &TempDir) -> std::path::PathBuf {
        let path = dir.path().join("legacy.db");
        let options =
            SqliteConnectOptions::from_str(&format!("sqlite:{}?mode=rwc", path.to_str().unwrap()))
                .unwrap()
                .journal_mode(SqliteJournalMode::Wal);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await
            .unwrap();

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS indexed_files (
                file_path    TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                chunk_count  INTEGER NOT NULL,
                indexed_at   TEXT NOT NULL DEFAULT (datetime('now'))
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        for (path_str, hash) in [("a.md", "hash-a"), ("b.md", "hash-b")] {
            sqlx::query(
                "INSERT INTO indexed_files (file_path, content_hash, chunk_count)
                 VALUES (?, ?, 2)",
            )
            .bind(path_str)
            .bind(hash)
            .execute(&pool)
            .await
            .unwrap();
        }

        pool.close().await;
        path
    }

    #[tokio::test]
    async fn upgrading_a_populated_legacy_db_adds_the_column_and_keeps_rows() {
        // The actual upgrade path a second deployment takes. Everything else in this
        // module starts from a fresh file, where the ALTER never has rows to preserve.
        let dir = TempDir::new().unwrap();
        let path = legacy_db(&dir).await;

        let db = StateDb::new(&path).await.expect("upgrade must not fail");

        assert_eq!(db.count().await.unwrap(), 2, "existing rows must survive");
        let entry = db.get("a.md").await.unwrap().unwrap();
        assert_eq!(entry.content_hash, "hash-a");
        assert_eq!(
            entry.schema_hash, "",
            "pre-migration rows carry an empty fingerprint, which forces exactly one \
             revalidation pass"
        );
        assert_eq!(
            entry.chunking_fingerprint, "",
            "pre-migration rows carry an empty chunking fingerprint, which never matches \
             ingest::chunking_fingerprint and so forces exactly one re-chunk/re-embed"
        );
    }

    #[tokio::test]
    async fn upgrading_a_pre_chunking_fingerprint_db_adds_only_that_column() {
        // The shape the release immediately before chunking fingerprints shipped with:
        // schema_hash/mtime/size already present, chunking_fingerprint absent. Its rows
        // carry real values in every existing column, which must survive untouched.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("pre-fingerprint.db");
        let options =
            SqliteConnectOptions::from_str(&format!("sqlite:{}?mode=rwc", path.to_str().unwrap()))
                .unwrap()
                .journal_mode(SqliteJournalMode::Wal);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await
            .unwrap();
        sqlx::query(
            "CREATE TABLE indexed_files (
                file_path    TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                chunk_count  INTEGER NOT NULL,
                indexed_at   TEXT NOT NULL DEFAULT (datetime('now')),
                schema_hash  TEXT NOT NULL DEFAULT '',
                mtime        INTEGER NOT NULL DEFAULT 0,
                size         INTEGER NOT NULL DEFAULT -1
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO indexed_files (file_path, content_hash, chunk_count, schema_hash, mtime, size)
             VALUES ('a.md', 'hash-a', 4, 'schema-fp', 123, 45)",
        )
        .execute(&pool)
        .await
        .unwrap();
        pool.close().await;

        let db = StateDb::new(&path).await.expect("upgrade must not fail");
        let row = db.get("a.md").await.unwrap().unwrap();
        assert_eq!(row.content_hash, "hash-a");
        assert_eq!(row.chunk_count, 4);
        assert_eq!(row.schema_hash, "schema-fp");
        assert_eq!((row.mtime, row.size), (123, 45));
        assert_eq!(
            row.chunking_fingerprint, "",
            "a pre-existing row must read back an empty (never-matching) chunking fingerprint"
        );

        let page = db.fetch_indexed_files_page(10, 0).await.unwrap();
        assert_eq!(page.len(), 1);
        assert_eq!(page[0].chunking_fingerprint, "");
    }

    #[tokio::test]
    async fn upgrading_a_pre_payload_hash_db_adds_only_that_column() {
        // #351: the shape of the release before `payload_hash` — every other column
        // present with real values. They must survive untouched, and the new column
        // must read back '' ("unknown"), which ingest treats as current while
        // `schema_hash` matches, so the upgrade re-embeds and rewrites nothing.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("pre-payload-hash.db");
        let options =
            SqliteConnectOptions::from_str(&format!("sqlite:{}?mode=rwc", path.to_str().unwrap()))
                .unwrap()
                .journal_mode(SqliteJournalMode::Wal);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await
            .unwrap();
        sqlx::query(
            "CREATE TABLE indexed_files (
                file_path            TEXT PRIMARY KEY,
                content_hash         TEXT NOT NULL,
                chunk_count          INTEGER NOT NULL,
                indexed_at           TEXT NOT NULL DEFAULT (datetime('now')),
                schema_hash          TEXT NOT NULL DEFAULT '',
                chunking_fingerprint TEXT NOT NULL DEFAULT '',
                mtime                INTEGER NOT NULL DEFAULT 0,
                size                 INTEGER NOT NULL DEFAULT -1
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO indexed_files
                (file_path, content_hash, chunk_count, indexed_at, schema_hash,
                 chunking_fingerprint, mtime, size)
             VALUES ('a.md', 'hash-a', 4, 'then', 'schema-fp', 'chunk-fp', 123, 45)",
        )
        .execute(&pool)
        .await
        .unwrap();
        pool.close().await;

        let db = StateDb::new(&path).await.expect("upgrade must not fail");
        let row = db.get("a.md").await.unwrap().unwrap();
        assert_eq!(row.content_hash, "hash-a");
        assert_eq!(row.chunk_count, 4);
        assert_eq!(row.indexed_at, "then");
        assert_eq!(row.schema_hash, "schema-fp");
        assert_eq!(row.chunking_fingerprint, "chunk-fp");
        assert_eq!((row.mtime, row.size), (123, 45));
        assert_eq!(
            row.payload_hash, "",
            "a pre-existing row's payload hash is unknown"
        );

        let page = db.fetch_indexed_files_page(10, 0).await.unwrap();
        assert_eq!(page.len(), 1);
        assert_eq!(page[0].payload_hash, "");
        assert_eq!(page[0].schema_hash, "schema-fp");
        let many = db.get_many(&["a.md".to_string()]).await.unwrap();
        assert_eq!(many["a.md"].payload_hash, "");
    }

    #[tokio::test]
    async fn upsert_record_and_update_schema_hashes_round_trip() {
        let dir = TempDir::new().unwrap();
        let db = StateDb::new(&dir.path().join("s.db")).await.unwrap();
        db.upsert_record(&IndexedFileRecord {
            file_path: "a.md",
            content_hash: "h",
            chunk_count: 3,
            schema_hash: "s1",
            payload_hash: "p1",
            chunking_fingerprint: "c",
            mtime: 1,
            size: 2,
        })
        .await
        .unwrap();
        let row = db.get("a.md").await.unwrap().unwrap();
        assert_eq!(
            (row.schema_hash.as_str(), row.payload_hash.as_str()),
            ("s1", "p1")
        );
        let indexed_at = row.indexed_at.clone();

        db.update_schema_hashes("a.md", "s2", "p2", 5, 6)
            .await
            .unwrap();
        let row = db.get("a.md").await.unwrap().unwrap();
        assert_eq!(row.schema_hash, "s2");
        assert_eq!(row.payload_hash, "p2");
        assert_eq!((row.mtime, row.size), (5, 6));
        // What describes the chunks still in Qdrant is untouched.
        assert_eq!(row.content_hash, "h");
        assert_eq!(row.chunk_count, 3);
        assert_eq!(row.chunking_fingerprint, "c");
        assert_eq!(row.indexed_at, indexed_at);

        // The positional `upsert` writes an unknown payload hash.
        db.upsert("a.md", "h", 3, "s3", "c", 1, 2).await.unwrap();
        assert_eq!(db.get("a.md").await.unwrap().unwrap().payload_hash, "");
    }

    #[tokio::test]
    async fn document_frontmatter_reads_the_stored_object() {
        let dir = TempDir::new().unwrap();
        let db = StateDb::new(&dir.path().join("s.db")).await.unwrap();
        assert_eq!(db.document_frontmatter("a.md").await.unwrap(), None);
        let mut fm = HashMap::new();
        fm.insert("title".to_string(), serde_json::json!("A"));
        fm.insert("planning".to_string(), serde_json::json!({"effort": "low"}));
        db.upsert_document_metadata("a.md", &fm, 0, "h", 1)
            .await
            .unwrap();
        let stored = db.document_frontmatter("a.md").await.unwrap().unwrap();
        assert_eq!(
            Value::Object(stored),
            serde_json::json!({"title": "A", "planning": {"effort": "low"}})
        );
    }

    #[tokio::test]
    async fn upgrading_is_idempotent() {
        let dir = TempDir::new().unwrap();
        let path = legacy_db(&dir).await;

        let first = StateDb::new(&path).await.unwrap();
        drop(first);
        let second = StateDb::new(&path).await.expect("second open must succeed");

        assert_eq!(second.count().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn upserting_a_legacy_row_after_upgrade_works() {
        let dir = TempDir::new().unwrap();
        let path = legacy_db(&dir).await;
        let db = StateDb::new(&path).await.unwrap();

        db.upsert("a.md", "hash-a2", 3, "fingerprint", "chunk-fp", 0, 0)
            .await
            .expect("writing to a migrated row must work");

        let entry = db.get("a.md").await.unwrap().unwrap();
        assert_eq!(entry.content_hash, "hash-a2");
        assert_eq!(entry.schema_hash, "fingerprint");
        assert_eq!(entry.chunking_fingerprint, "chunk-fp");
    }

    #[tokio::test]
    async fn concurrent_first_open_does_not_fail_either_caller() {
        // Two independent pools racing the check-then-ALTER on first upgrade. The loser
        // sees "duplicate column name", which is the desired end state, not an error.
        let dir = TempDir::new().unwrap();
        let path = legacy_db(&dir).await;

        let (a, b) = tokio::join!(StateDb::new(&path), StateDb::new(&path));

        assert!(a.is_ok(), "first opener failed: {:?}", a.err());
        assert!(b.is_ok(), "second opener failed: {:?}", b.err());
    }

    #[tokio::test]
    async fn duplicate_column_is_recognized_but_other_errors_are_not() {
        // The migration swallows exactly one error. Anything broader would hide a real
        // failure behind a successful-looking open.
        let dir = TempDir::new().unwrap();
        let path = legacy_db(&dir).await;
        let db = StateDb::new(&path).await.unwrap();

        let duplicate = sqlx::query("ALTER TABLE indexed_files ADD COLUMN schema_hash TEXT")
            .execute(db.pool_for_test())
            .await
            .expect_err("the column already exists");
        assert!(is_duplicate_column(&duplicate));

        let unrelated = sqlx::query("SELECT * FROM no_such_table")
            .execute(db.pool_for_test())
            .await
            .expect_err("table does not exist");
        assert!(
            !is_duplicate_column(&unrelated),
            "an unrelated database error must not be treated as a benign race"
        );
    }

    #[tokio::test]
    async fn a_fresh_db_already_has_the_column() {
        let (db, _dir) = test_db().await;
        db.upsert("new.md", "h", 1, "fp", "cfp", 0, 0)
            .await
            .unwrap();
        let row = db.get("new.md").await.unwrap().unwrap();
        assert_eq!(row.schema_hash, "fp");
        assert_eq!(row.chunking_fingerprint, "cfp");
    }

    // -- document metadata index --------------------------------------------

    fn recipe_frontmatter() -> HashMap<String, Value> {
        let json = serde_json::json!({
            "title": "Stir Fry",
            "description": "Weekly improvised stir fry.",
            "type": "reference",
            "tags": ["recipe", "stir-fry", "wok"],
            "planning": { "prep_minutes": 45, "needs_recipe": false, "effort": "medium" }
        });
        match json {
            Value::Object(map) => map.into_iter().collect(),
            _ => unreachable!(),
        }
    }

    async fn field_rows(db: &StateDb, file_path: &str) -> Vec<(String, String, Option<f64>)> {
        sqlx::query_as(
            "SELECT field, value_text, value_num FROM document_fields
             WHERE file_path = ? ORDER BY field, value_text",
        )
        .bind(file_path)
        .fetch_all(&db.pool)
        .await
        .unwrap()
    }

    /// Frontmatter with a chosen type, area-agnostic, plus two tags.
    fn doc_frontmatter(doc_type: &str, tags: &[&str]) -> HashMap<String, Value> {
        let json = serde_json::json!({
            "title": "T",
            "type": doc_type,
            "tags": tags,
        });
        match json {
            Value::Object(map) => map.into_iter().collect(),
            _ => unreachable!(),
        }
    }

    async fn seed_breakdown_corpus(db: &StateDb) {
        for (path, ty, tags) in [
            ("food/recipes/a.md", "recipe", vec!["dinner", "quick"]),
            ("food/recipes/b.md", "recipe", vec!["dinner"]),
            ("food/plans/c.md", "project", vec!["quick"]),
            ("sysadmin/d.md", "guide", vec!["docker"]),
            ("top-level.md", "guide", vec![]),
        ] {
            db.upsert_document_metadata(path, &doc_frontmatter(ty, &tags), 1700, "h", 1)
                .await
                .unwrap();
        }
    }

    #[tokio::test]
    async fn count_by_field_counts_documents_per_value() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;

        let by_type = db.count_by_field("type", 50).await.unwrap();
        assert_eq!(
            by_type,
            vec![
                ("guide".to_string(), 2),
                ("recipe".to_string(), 2),
                ("project".to_string(), 1),
            ],
            "ordered by count desc, then value asc for a stable report"
        );

        // `tags` stores one row per value, so this is the case where counting rows
        // instead of distinct documents would go wrong.
        let by_tag = db.count_by_field("tags", 50).await.unwrap();
        assert_eq!(by_tag.iter().find(|(v, _)| v == "dinner").unwrap().1, 2);
        assert_eq!(by_tag.iter().find(|(v, _)| v == "quick").unwrap().1, 2);
        assert_eq!(by_tag.iter().find(|(v, _)| v == "docker").unwrap().1, 1);
    }

    #[tokio::test]
    async fn count_by_field_respects_the_limit() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;

        let limited = db.count_by_field("type", 2).await.unwrap();
        assert_eq!(limited.len(), 2);
        // The most common values survive truncation.
        assert_eq!(limited[0].1, 2);
    }

    #[tokio::test]
    async fn count_by_field_clamps_a_negative_limit() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;
        // SQLite reads a negative LIMIT as "unbounded", so an unclamped negative would
        // silently return the whole vocabulary instead of nothing.
        assert!(db.count_by_field("type", -1).await.unwrap().is_empty());
        assert!(db.count_by_field("type", 0).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn count_by_field_is_empty_for_an_unknown_field() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;
        assert!(db.count_by_field("nope", 50).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn breakdown_fields_reports_distinct_value_counts() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;

        let cards: HashMap<String, i64> = db
            .breakdown_fields(50, 50)
            .await
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(cards.get("type"), Some(&3));
        // dinner, quick, docker
        assert_eq!(cards.get("tags"), Some(&3));
        // `title`/`description` are promoted to columns on `documents` and never
        // projected into `document_fields`, so the highest-cardinality free-text fields
        // cannot reach a breakdown at all.
        assert_eq!(cards.get("title"), None);
        assert_eq!(cards.get("description"), None);
    }

    #[tokio::test]
    async fn breakdown_fields_excludes_high_cardinality_fields_in_sql() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;

        // `type` and `tags` each take 3 distinct values. With the ceiling at 2, both
        // drop — the filter is applied in SQL, not by the caller.
        let fields: Vec<String> = db
            .breakdown_fields(2, 50)
            .await
            .unwrap()
            .into_iter()
            .map(|(f, _)| f)
            .collect();
        assert!(!fields.contains(&"type".to_string()), "{fields:?}");
        assert!(!fields.contains(&"tags".to_string()), "{fields:?}");
    }

    #[tokio::test]
    async fn breakdown_fields_limit_keeps_usable_fields_not_the_widest() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;
        // A field far too wide to be a useful histogram.
        for i in 0..40 {
            let fm: HashMap<String, Value> = match serde_json::json!({
                "title": "T", "type": "guide", "serial": format!("sn-{i}")
            }) {
                Value::Object(m) => m.into_iter().collect(),
                _ => unreachable!(),
            };
            db.upsert_document_metadata(&format!("wide/{i}.md"), &fm, 1700, "h", 1)
                .await
                .unwrap();
        }

        // Rows come back richest-first, so an unfiltered `ORDER BY n DESC LIMIT 1`
        // would return `serial` — and every reportable field would be lost to the cap.
        let fields: Vec<String> = db
            .breakdown_fields(10, 1)
            .await
            .unwrap()
            .into_iter()
            .map(|(f, _)| f)
            .collect();
        assert_eq!(fields.len(), 1);
        assert!(
            !fields.contains(&"serial".to_string()),
            "the 40-value field must not consume the only slot: {fields:?}"
        );
        // `tags` and `type` both take 3 values and tie; either is a usable histogram.
        assert!(
            fields[0] == "tags" || fields[0] == "type",
            "expected a low-cardinality field, got {fields:?}"
        );
    }

    #[tokio::test]
    async fn breakdown_fields_orders_by_document_coverage_not_value_count() {
        let (db, _dir) = test_db().await;
        // `type` covers every document with few values; a scoped field covers a handful
        // of documents with many. Ordering by value count would float the scoped field
        // above the one that describes the whole corpus.
        for i in 0..30 {
            let mut fm: HashMap<String, Value> = match serde_json::json!({"title": "T", "type": "guide"})
            {
                Value::Object(m) => m.into_iter().collect(),
                _ => unreachable!(),
            };
            if i < 5 {
                fm.insert("scoped".into(), Value::String(format!("v{i}")));
            }
            db.upsert_document_metadata(&format!("d{i}.md"), &fm, 1700, "h", 1)
                .await
                .unwrap();
        }

        let fields: Vec<String> = db
            .breakdown_fields(500, 10)
            .await
            .unwrap()
            .into_iter()
            .map(|(f, _)| f)
            .collect();
        assert_eq!(
            fields.first().map(String::as_str),
            Some("type"),
            "the field covering every document must lead: {fields:?}"
        );
        assert!(fields.contains(&"scoped".to_string()), "{fields:?}");
    }

    #[tokio::test]
    async fn breakdown_fields_clamps_a_negative_limit() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;
        assert!(db.breakdown_fields(50, -1).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn area_counts_group_by_top_level_directory() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;

        let areas = db.area_counts().await.unwrap();
        assert_eq!(
            areas,
            vec![
                ("food".to_string(), 3),
                ("".to_string(), 1),
                ("sysadmin".to_string(), 1),
            ],
            "a document at the KB root has no area and must not be dropped"
        );
    }

    #[tokio::test]
    async fn aggregates_are_empty_on_a_fresh_database() {
        let (db, _dir) = test_db().await;
        assert!(db.breakdown_fields(50, 50).await.unwrap().is_empty());
        assert!(db.area_counts().await.unwrap().is_empty());
        assert!(db.count_by_field("type", 50).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn aggregates_follow_document_deletion() {
        let (db, _dir) = test_db().await;
        seed_breakdown_corpus(&db).await;
        db.delete_document("food/recipes/a.md").await.unwrap();

        let by_type: HashMap<String, i64> = db
            .count_by_field("type", 50)
            .await
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(by_type.get("recipe"), Some(&1));

        let areas: HashMap<String, i64> = db.area_counts().await.unwrap().into_iter().collect();
        assert_eq!(areas.get("food"), Some(&2));
    }

    #[tokio::test]
    async fn upsert_document_metadata_stores_promoted_columns() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("r.md", &recipe_frontmatter(), 1700, "h1", 2)
            .await
            .unwrap();

        let row: (String, Option<String>, Option<String>, i64, i64) = sqlx::query_as(
            "SELECT file_path, title, description, mtime, chunk_count FROM documents WHERE file_path = ?",
        )
        .bind("r.md")
        .fetch_one(&db.pool)
        .await
        .unwrap();

        assert_eq!(row.0, "r.md");
        assert_eq!(row.1.as_deref(), Some("Stir Fry"));
        assert_eq!(row.2.as_deref(), Some("Weekly improvised stir fry."));
        assert_eq!(row.3, 1700);
        assert_eq!(row.4, 2);
    }

    #[tokio::test]
    async fn upsert_document_metadata_projects_nested_and_list_fields() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("r.md", &recipe_frontmatter(), 1700, "h1", 2)
            .await
            .unwrap();

        let rows = field_rows(&db, "r.md").await;
        assert!(rows.contains(&("planning.prep_minutes".into(), "45".into(), Some(45.0))));
        assert!(rows.contains(&("planning.needs_recipe".into(), "false".into(), Some(0.0))));
        assert_eq!(
            rows.iter().filter(|r| r.0 == "tags").count(),
            3,
            "one row per tag"
        );
    }

    #[tokio::test]
    async fn frontmatter_json_is_stored_faithfully() {
        let (db, _dir) = test_db().await;
        let fm = recipe_frontmatter();
        db.upsert_document_metadata("r.md", &fm, 1700, "h1", 2)
            .await
            .unwrap();

        let (json,): (String,) =
            sqlx::query_as("SELECT frontmatter FROM documents WHERE file_path = ?")
                .bind("r.md")
                .fetch_one(&db.pool)
                .await
                .unwrap();
        let round_tripped: HashMap<String, Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(round_tripped, fm, "stored JSON must be lossless");
    }

    #[tokio::test]
    async fn reupsert_removes_fields_dropped_from_frontmatter() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("r.md", &recipe_frontmatter(), 1700, "h1", 2)
            .await
            .unwrap();

        // Same document, planning block removed entirely.
        let mut trimmed = recipe_frontmatter();
        trimmed.remove("planning");
        db.upsert_document_metadata("r.md", &trimmed, 1800, "h2", 2)
            .await
            .unwrap();

        let rows = field_rows(&db, "r.md").await;
        assert!(
            rows.iter().all(|r| !r.0.starts_with("planning.")),
            "stale projection rows must not survive a re-upsert"
        );
        assert_eq!(rows.iter().filter(|r| r.0 == "tags").count(), 3);
    }

    #[tokio::test]
    async fn delete_document_cascades_to_fields() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("r.md", &recipe_frontmatter(), 1700, "h1", 2)
            .await
            .unwrap();
        assert!(!field_rows(&db, "r.md").await.is_empty());

        db.delete_document("r.md").await.unwrap();

        assert_eq!(db.document_count().await.unwrap(), 0);
        assert!(
            field_rows(&db, "r.md").await.is_empty(),
            "ON DELETE CASCADE requires foreign_keys(true) to be set on the connection"
        );
    }

    #[tokio::test]
    async fn list_document_hashes_maps_paths_to_hashes() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("a.md", &recipe_frontmatter(), 1, "hash-a", 1)
            .await
            .unwrap();
        db.upsert_document_metadata("b.md", &recipe_frontmatter(), 1, "hash-b", 1)
            .await
            .unwrap();

        let hashes = db.list_document_hashes().await.unwrap();
        assert_eq!(hashes.get("a.md").map(String::as_str), Some("hash-a"));
        assert_eq!(hashes.get("b.md").map(String::as_str), Some("hash-b"));
        assert_eq!(hashes.len(), 2);
    }

    #[tokio::test]
    async fn reproject_rebuilds_fields_from_stored_json() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("r.md", &recipe_frontmatter(), 1700, "h1", 2)
            .await
            .unwrap();

        // Simulate a projection corrupted or emptied by a rules change.
        sqlx::query("DELETE FROM document_fields")
            .execute(&db.pool)
            .await
            .unwrap();
        assert!(field_rows(&db, "r.md").await.is_empty());

        let report = db.reproject_all_fields(&no_schemas()).await.unwrap();

        assert_eq!(report.reprojected, 1);
        let rows = field_rows(&db, "r.md").await;
        assert!(rows.contains(&("planning.prep_minutes".into(), "45".into(), Some(45.0))));
    }

    #[tokio::test]
    async fn duplicate_array_values_do_not_break_upsert() {
        let (db, _dir) = test_db().await;
        let fm: HashMap<String, Value> = match serde_json::json!({
            "title": "Dupes",
            "tags": ["recipe", "recipe", "wok"]
        }) {
            Value::Object(map) => map.into_iter().collect(),
            _ => unreachable!(),
        };

        db.upsert_document_metadata("d.md", &fm, 1, "h", 1)
            .await
            .unwrap();

        // Collides on the (file_path, field, elem, value_text) primary key; must not
        // error.
        assert_eq!(field_rows(&db, "d.md").await.len(), 2);
    }

    // -- schema-driven projection, correlated filters, versioning (#349) -----

    fn no_schemas() -> crate::schema::SchemaCache {
        crate::schema::SchemaCache::from_config_only(&Default::default())
    }

    /// A schema cache whose root `.schema.yaml` is `yaml`.
    fn schemas_from(yaml: &str) -> (crate::schema::SchemaCache, TempDir) {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(".schema.yaml"), yaml).unwrap();
        let cache = crate::schema::SchemaCache::build_for_test(dir.path(), &Default::default());
        (cache, dir)
    }

    fn fm_of(value: Value) -> HashMap<String, Value> {
        match value {
            Value::Object(map) => map.into_iter().collect(),
            _ => panic!("frontmatter fixture must be an object"),
        }
    }

    fn query_with(filters: Vec<(&str, FieldFilter)>) -> DocumentQuery {
        DocumentQuery {
            filters: filters
                .into_iter()
                .map(|(f, filter)| (f.to_string(), filter))
                .collect(),
            ..Default::default()
        }
    }

    fn any(value: &str) -> FieldFilter {
        FieldFilter::AnyOf(vec![value.to_string()])
    }

    fn at_least(n: f64) -> FieldFilter {
        FieldFilter::Range {
            gte: Some(n),
            lte: None,
            gt: None,
            lt: None,
        }
    }

    /// `split.md`: role guard on element 0, level 5 on element 1 — the two conditions
    /// hold in the document, never in one element. `same.md`: both on element 1.
    async fn npc_db() -> (StateDb, TempDir) {
        let (db, dir) = test_db().await;
        let split = fm_of(serde_json::json!({
            "npcs": [{ "role": "guard", "level": 1 }, { "role": "cook", "level": 5 }]
        }));
        let same = fm_of(serde_json::json!({
            "npcs": [{ "role": "cook", "level": 2 }, { "role": "guard", "level": 5 }]
        }));
        db.upsert_document_metadata("split.md", &split, 1, "h", 1)
            .await
            .unwrap();
        db.upsert_document_metadata("same.md", &same, 1, "h", 1)
            .await
            .unwrap();
        (db, dir)
    }

    #[tokio::test]
    async fn a_correlated_filter_needs_every_condition_in_one_element() {
        let (db, _dir) = npc_db().await;
        let correlated = query_with(vec![(
            "npcs[]",
            FieldFilter::Elem(vec![
                ("npcs[].level".into(), at_least(3.0)),
                ("npcs[].role".into(), any("guard")),
            ]),
        )]);
        let result = db.query_documents(&correlated).await.unwrap();
        assert_eq!(paths(&result), vec!["same.md"]);

        // The same two conditions uncorrelated match both documents.
        let uncorrelated = query_with(vec![
            ("npcs[].level", at_least(3.0)),
            ("npcs[].role", any("guard")),
        ]);
        let result = db.query_documents(&uncorrelated).await.unwrap();
        assert_eq!(paths(&result), vec!["same.md", "split.md"]);

        // A single correlated condition is just a filter on the sub-field.
        let one = query_with(vec![(
            "npcs[]",
            FieldFilter::Elem(vec![("npcs[].role".into(), any("cook"))]),
        )]);
        assert_eq!(db.query_documents(&one).await.unwrap().total, 2);

        // all_of inside an element: both values on one element's list.
        let tagged = fm_of(serde_json::json!({
            "npcs": [{ "tags": ["a"] }, { "tags": ["b"] }]
        }));
        db.upsert_document_metadata("tags.md", &tagged, 1, "h", 1)
            .await
            .unwrap();
        let all_of = query_with(vec![(
            "npcs[]",
            FieldFilter::Elem(vec![(
                "npcs[].tags".into(),
                FieldFilter::AllOf(vec!["a".into(), "b".into()]),
            )]),
        )]);
        assert_eq!(db.query_documents(&all_of).await.unwrap().total, 0);
        let plain_all_of = query_with(vec![(
            "npcs[].tags",
            FieldFilter::AllOf(vec!["a".into(), "b".into()]),
        )]);
        assert_eq!(
            paths(&db.query_documents(&plain_all_of).await.unwrap()),
            vec!["tags.md"]
        );

        // Unsatisfiable shapes stay unsatisfiable.
        for bad in [
            FieldFilter::Elem(vec![]),
            FieldFilter::Elem(vec![("npcs[].role".into(), FieldFilter::AnyOf(vec![]))]),
        ] {
            let q = query_with(vec![("npcs[]", bad)]);
            assert_eq!(db.query_documents(&q).await.unwrap().total, 0);
        }
    }

    #[tokio::test]
    async fn date_range_filters_order_dates_and_timestamps_on_one_axis() {
        let (schemas, _schema_dir) =
            schemas_from("fields:\n  on:\n    type: date\n  at:\n    type: timestamp\n");
        let (db, _dir) = test_db().await;
        for (path, on, at) in [
            ("a.md", "2024-01-01", "2024-01-01T12:00:00Z"),
            ("b.md", "2024-02-01", "2024-01-02T01:00:00+02:00"),
            ("c.md", "2023-12-31", "2024-01-02T00:30:00Z"),
        ] {
            db.upsert_document_metadata_with(
                path,
                &fm_of(serde_json::json!({ "on": on, "at": at })),
                schemas.resolve_for(Path::new(path)),
                1,
                "h",
                1,
            )
            .await
            .unwrap();
        }
        let jan_2 = crate::document_fields::epoch_seconds("2024-01-02").unwrap();
        let on_or_after = query_with(vec![("on", at_least(jan_2))]);
        assert_eq!(
            paths(&db.query_documents(&on_or_after).await.unwrap()),
            vec!["b.md"]
        );
        // b's 01:00+02:00 is 23:00 UTC on Jan 1, before Jan 2's midnight; c's is after.
        let at_or_after = query_with(vec![("at", at_least(jan_2))]);
        assert_eq!(
            paths(&db.query_documents(&at_or_after).await.unwrap()),
            vec!["c.md"]
        );
    }

    async fn columns_of_document_fields(db: &StateDb) -> Vec<(String, i64)> {
        sqlx::query_as("SELECT name, pk FROM pragma_table_info('document_fields') ORDER BY cid")
            .fetch_all(&db.pool)
            .await
            .unwrap()
    }

    /// A database as written before #349: `document_fields` without `elem`.
    async fn old_shape_db(dir: &TempDir) -> std::path::PathBuf {
        let path = dir.path().join("old.db");
        let pool = SqlitePoolOptions::new()
            .connect_with(
                SqliteConnectOptions::from_str(&format!("sqlite:{}?mode=rwc", path.display()))
                    .unwrap(),
            )
            .await
            .unwrap();
        // Left in the default rollback journal mode, as a database some other
        // tool created: the first `StateDb::new` converts it to WAL, which is the
        // hardest case for concurrent openers.
        for statement in [
            "CREATE TABLE documents (
                file_path TEXT PRIMARY KEY, title TEXT, description TEXT,
                frontmatter TEXT NOT NULL, mtime INTEGER NOT NULL,
                content_hash TEXT NOT NULL, chunk_count INTEGER NOT NULL,
                indexed_at TEXT NOT NULL DEFAULT (datetime('now')))",
            "CREATE TABLE document_fields (
                file_path TEXT NOT NULL REFERENCES documents(file_path) ON DELETE CASCADE,
                field TEXT NOT NULL, value_text TEXT NOT NULL, value_num REAL,
                PRIMARY KEY (file_path, field, value_text))",
            "CREATE INDEX idx_document_fields_field_text ON document_fields(field, value_text)",
            "INSERT INTO documents (file_path, frontmatter, mtime, content_hash, chunk_count)
             VALUES ('n.md', '{\"npcs\": [{\"role\": \"guard\"}], \"type\": \"guide\"}', 1, 'h', 1)",
            "INSERT INTO document_fields VALUES ('n.md', 'type', 'guide', NULL)",
        ] {
            sqlx::query(statement).execute(&pool).await.unwrap();
        }
        pool.close().await;
        path
    }

    #[tokio::test]
    async fn an_old_shaped_field_table_is_rebuilt_and_reprojected() {
        let dir = TempDir::new().unwrap();
        let path = old_shape_db(&dir).await;

        let db = StateDb::new(&path).await.unwrap();
        let columns = columns_of_document_fields(&db).await;
        assert!(
            columns.iter().any(|(name, pk)| name == "elem" && *pk > 0),
            "elem is part of the key: {columns:?}"
        );
        assert!(field_rows(&db, "n.md").await.is_empty(), "old rows dropped");
        assert_eq!(
            db.projection_version().await.unwrap(),
            0,
            "reproject needed"
        );

        let report = db
            .ensure_projection_current(&no_schemas(), false)
            .await
            .unwrap()
            .expect("a reproject ran");
        assert_eq!(report.reprojected, 1);
        assert_eq!(db.projection_version().await.unwrap(), PROJECTION_VERSION);
        let rows = field_rows(&db, "n.md").await;
        assert!(
            rows.contains(&("npcs[].role".into(), "guard".into(), None)),
            "{rows:?}"
        );
        assert!(rows.contains(&("type".into(), "guide".into(), None)));

        // Current now: reopening rebuilds nothing, and the check is a no-op.
        drop(db);
        let db = StateDb::new(&path).await.unwrap();
        assert!(!field_rows(&db, "n.md").await.is_empty());
        assert_eq!(
            db.ensure_projection_current(&no_schemas(), false)
                .await
                .unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn concurrent_opens_of_an_old_shaped_db_all_succeed() {
        // Each open is what a separate process (serve, a CLI command) does against
        // the same file; repeated with fresh databases because the race is timing
        // dependent.
        const OPENERS: usize = 8;
        const ROUNDS: usize = 20;
        for round in 0..ROUNDS {
            let dir = TempDir::new().unwrap();
            let path = old_shape_db(&dir).await;
            let opens = (0..OPENERS).map(|_| {
                let path = path.clone();
                tokio::spawn(async move { StateDb::new(&path).await })
            });
            for (i, open) in futures::future::join_all(opens)
                .await
                .into_iter()
                .enumerate()
            {
                let db = open
                    .unwrap()
                    .unwrap_or_else(|e| panic!("round {round}, opener {i}: {e:#}"));
                let columns = columns_of_document_fields(&db).await;
                assert!(
                    columns.iter().any(|(name, pk)| name == "elem" && *pk > 0),
                    "round {round}, opener {i}: {columns:?}"
                );
                assert_eq!(db.projection_version().await.unwrap(), 0);
            }
        }
    }

    #[tokio::test]
    async fn a_projection_version_change_reprojects_without_touching_anything_else() {
        let (db, _dir) = test_db().await;
        // A fresh database records the version on its first check.
        assert!(
            db.ensure_projection_current(&no_schemas(), false)
                .await
                .unwrap()
                .is_some()
        );
        db.upsert_document_metadata("r.md", &recipe_frontmatter(), 1, "h", 1)
            .await
            .unwrap();
        db.upsert("r.md", "h", 1, "schema", "chunking", 0, 10)
            .await
            .unwrap();

        // An older version on disk, and a projection that no longer matches.
        sqlx::query("PRAGMA user_version = 0")
            .execute(&db.pool)
            .await
            .unwrap();
        sqlx::query("DELETE FROM document_fields")
            .execute(&db.pool)
            .await
            .unwrap();

        let report = db
            .ensure_projection_current(&no_schemas(), false)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(report.reprojected, 1);
        assert!(!field_rows(&db, "r.md").await.is_empty());
        assert_eq!(db.projection_version().await.unwrap(), PROJECTION_VERSION);
        // Reads only the state DB: the indexed_files bookkeeping is untouched.
        let row = db.get("r.md").await.unwrap().unwrap();
        assert_eq!(row.content_hash, "h");
        assert_eq!(row.schema_hash, "schema");
    }

    #[tokio::test]
    async fn a_failed_reproject_does_not_advance_the_projection_version() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("r.md", &recipe_frontmatter(), 1, "h", 1)
            .await
            .unwrap();
        // Every projection write fails, as a full disk or a crash mid-run would.
        sqlx::query(
            "CREATE TRIGGER refuse BEFORE INSERT ON document_fields \
             BEGIN SELECT RAISE(ABORT, 'refused'); END",
        )
        .execute(&db.pool)
        .await
        .unwrap();

        assert!(
            db.ensure_projection_current(&no_schemas(), false)
                .await
                .is_err()
        );
        assert_eq!(db.projection_version().await.unwrap(), 0);

        sqlx::query("DROP TRIGGER refuse")
            .execute(&db.pool)
            .await
            .unwrap();
        db.ensure_projection_current(&no_schemas(), false)
            .await
            .unwrap()
            .expect("the retry reprojects");
        assert_eq!(db.projection_version().await.unwrap(), PROJECTION_VERSION);
    }

    #[tokio::test]
    async fn a_persistently_failing_reproject_is_retried_once_per_process() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("r.md", &recipe_frontmatter(), 1, "h", 1)
            .await
            .unwrap();
        sqlx::query(
            "CREATE TRIGGER refuse BEFORE INSERT ON document_fields \
             BEGIN SELECT RAISE(ABORT, 'refused'); END",
        )
        .execute(&db.pool)
        .await
        .unwrap();

        let schemas = no_schemas();
        assert!(db.ensure_projection_current(&schemas, false).await.is_err());
        assert!(
            db.ensure_projection_current(&schemas, false).await.is_err(),
            "one retry"
        );
        assert_eq!(
            db.ensure_projection_current(&schemas, false).await.unwrap(),
            None,
            "then no more attempts in this process"
        );
        assert_eq!(db.projection_version().await.unwrap(), 0);
        // A forced run still tries, and a success clears the count.
        assert!(db.ensure_projection_current(&schemas, true).await.is_err());
        sqlx::query("DROP TRIGGER refuse")
            .execute(&db.pool)
            .await
            .unwrap();
        db.ensure_projection_current(&schemas, true).await.unwrap();
        assert_eq!(db.projection_version().await.unwrap(), PROJECTION_VERSION);
    }

    #[tokio::test]
    async fn reproject_reads_each_document_under_its_schema() {
        let (schemas, _schema_dir) = schemas_from("fields:\n  code:\n    type: text\n");
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("c.md", &fm_of(serde_json::json!({ "code": 42 })), 1, "h", 1)
            .await
            .unwrap();
        assert_eq!(
            field_rows(&db, "c.md").await,
            vec![("code".into(), "42".into(), Some(42.0))],
            "undeclared: guessed"
        );
        db.ensure_projection_current(&schemas, true).await.unwrap();
        assert_eq!(
            field_rows(&db, "c.md").await,
            vec![("code".into(), "42".into(), None)],
            "declared text: no value_num"
        );
    }

    #[tokio::test]
    async fn breakdown_fields_leave_out_list_element_fields() {
        let (db, _dir) = npc_db().await;
        let fields = db.breakdown_fields(50, 50).await.unwrap();
        assert!(
            fields.iter().all(|(name, _)| !name.contains("[]")),
            "{fields:?}"
        );
        // Still filterable and still in use for the vocabulary check.
        assert!(db.field_in_use("npcs[].role").await.unwrap());
    }

    // -- listing / query builder --------------------------------------------

    fn doc(title: &str, tags: &[&str], prep: i64, tested: bool) -> HashMap<String, Value> {
        let json = serde_json::json!({
            "title": title,
            "description": format!("{} description", title),
            "type": "reference",
            "tags": tags,
            "planning": { "prep_minutes": prep, "tested": tested }
        });
        match json {
            Value::Object(map) => map.into_iter().collect(),
            _ => unreachable!(),
        }
    }

    async fn seeded_db() -> (StateDb, TempDir) {
        let (db, dir) = test_db().await;
        for (path, title, tags, prep, tested) in [
            (
                "kitchen/recipes/chili.md",
                "Chili",
                &["recipe", "dinner"][..],
                20,
                true,
            ),
            (
                "kitchen/recipes/congee.md",
                "Congee",
                &["recipe", "breakfast"][..],
                45,
                true,
            ),
            (
                "kitchen/recipes/stir_fry.md",
                "Stir Fry",
                &["recipe", "dinner"][..],
                45,
                false,
            ),
            ("sysadmin/zfs.md", "ZFS", &["zfs"][..], 0, true),
        ] {
            db.upsert_document_metadata(path, &doc(title, tags, prep, tested), 100, "h", 1)
                .await
                .unwrap();
        }
        (db, dir)
    }

    fn paths(result: &DocumentQueryResult) -> Vec<&str> {
        result
            .documents
            .iter()
            .map(|d| d.file_path.as_str())
            .collect()
    }

    #[tokio::test]
    async fn listing_with_no_filters_returns_everything_ordered_by_path() {
        let (db, _dir) = seeded_db().await;
        let result = db.query_documents(&DocumentQuery::default()).await.unwrap();

        assert_eq!(result.total, 4);
        assert_eq!(
            paths(&result),
            vec![
                "kitchen/recipes/chili.md",
                "kitchen/recipes/congee.md",
                "kitchen/recipes/stir_fry.md",
                "sysadmin/zfs.md",
            ]
        );
    }

    #[tokio::test]
    async fn any_of_matches_documents_with_either_tag() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![(
                "tags".into(),
                FieldFilter::AnyOf(vec!["breakfast".into(), "zfs".into()]),
            )],
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        assert_eq!(result.total, 2);
        assert_eq!(
            paths(&result),
            vec!["kitchen/recipes/congee.md", "sysadmin/zfs.md"]
        );
    }

    #[tokio::test]
    async fn all_of_requires_every_value() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![(
                "tags".into(),
                FieldFilter::AllOf(vec!["recipe".into(), "dinner".into()]),
            )],
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        assert_eq!(result.total, 2, "congee is a recipe but not dinner");
        assert_eq!(
            paths(&result),
            vec!["kitchen/recipes/chili.md", "kitchen/recipes/stir_fry.md"]
        );
    }

    #[tokio::test]
    async fn filters_are_anded_across_fields() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![
                ("tags".into(), FieldFilter::AnyOf(vec!["recipe".into()])),
                (
                    "planning.tested".into(),
                    FieldFilter::AnyOf(vec!["false".into()]),
                ),
            ],
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        assert_eq!(paths(&result), vec!["kitchen/recipes/stir_fry.md"]);
    }

    #[tokio::test]
    async fn numeric_range_filters_on_nested_field() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![(
                "planning.prep_minutes".into(),
                FieldFilter::Range {
                    gte: None,
                    lte: None,
                    gt: None,
                    lt: Some(30.0),
                },
            )],
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        // chili is 20; zfs is 0. String comparison would have ordered "45" before "5".
        assert_eq!(
            paths(&result),
            vec!["kitchen/recipes/chili.md", "sysadmin/zfs.md"]
        );
    }

    #[tokio::test]
    async fn range_bounds_combine_as_between() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![(
                "planning.prep_minutes".into(),
                FieldFilter::Range {
                    gte: Some(10.0),
                    lte: Some(30.0),
                    gt: None,
                    lt: None,
                },
            )],
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        assert_eq!(paths(&result), vec!["kitchen/recipes/chili.md"]);
    }

    // ------------------------------------------------------------------
    // query mode / enumeration mode: filter semantics must agree
    // ------------------------------------------------------------------
    //
    // `search`'s query-mode filters lower to Qdrant conditions via
    // `qdrant::lower_field_filters`; its enumeration-mode filters lower to SQL
    // via `push_where`, exercised here through the real `query_documents` path.
    // No live Qdrant is available in this environment to run the lowered
    // conditions for real (see `qdrant::tests::qdrant_filters_agree_with_the_offline_prediction`
    // for the `#[ignore]`d live counterpart), so this section evaluates the
    // lowered `Condition` tree directly against the same frontmatter this
    // SQLite fixture was seeded from, using a small interpreter that only needs
    // to understand the shapes `lower_field_filters` can actually produce
    // (`Match`/`Range` field conditions, and a `Filter::should` OR for
    // multi-value boolean equality) — anything else means this test's
    // evaluator is out of date with production, so it panics rather than
    // silently passing.

    fn equivalence_frontmatter(
        type_: &str,
        tags: &[&str],
        prep_minutes: i64,
        tested: bool,
        rating: f64,
    ) -> HashMap<String, Value> {
        let json = serde_json::json!({
            "title": type_,
            "type": type_,
            "tags": tags,
            "rating": rating,
            "planning": { "prep_minutes": prep_minutes, "tested": tested }
        });
        match json {
            Value::Object(map) => map.into_iter().collect(),
            _ => unreachable!(),
        }
    }

    fn equivalence_docs() -> Vec<(&'static str, HashMap<String, Value>)> {
        vec![
            (
                "a.md",
                equivalence_frontmatter("guide", &["recipe", "dinner"], 20, true, 4.5),
            ),
            (
                "b.md",
                equivalence_frontmatter("recipe", &["recipe", "breakfast"], 45, true, 3.0),
            ),
            (
                "c.md",
                equivalence_frontmatter("reference", &["recipe", "dinner", "zfs"], 45, false, 4.5),
            ),
            (
                "d.md",
                equivalence_frontmatter("guide", &["zfs"], 15, true, 2.0),
            ),
        ]
    }

    async fn equivalence_db() -> (StateDb, TempDir) {
        let (db, dir) = test_db().await;
        for (path, fm) in equivalence_docs() {
            db.upsert_document_metadata(path, &fm, 100, "h", 1)
                .await
                .unwrap();
        }
        (db, dir)
    }

    /// The values at payload key `key` — a dot-path where a `name[]` segment steps
    /// into every element of that array (`fm.npcs[].role`) — the way Qdrant
    /// addresses a nested payload field natively. A key that reaches nothing gives
    /// no values.
    fn payload_at<'a>(payload: &'a Value, key: &str) -> Vec<&'a Value> {
        let mut current: Vec<&Value> = vec![payload];
        for part in key.split('.') {
            let (name, elements) = match part.strip_suffix("[]") {
                Some(name) => (name, true),
                None => (part, false),
            };
            current = current
                .into_iter()
                .filter_map(|v| v.as_object()?.get(name))
                .flat_map(|v| match (elements, v) {
                    (true, Value::Array(items)) => items.iter().collect::<Vec<_>>(),
                    (true, _) => Vec::new(),
                    (false, v) => vec![v],
                })
                .collect();
        }
        current
    }

    /// Whether a datetime payload value is inside a `DatetimeRange`, at the
    /// microsecond precision the bounds are rounded to (`qdrant::epoch_to_timestamp`).
    fn qdrant_datetime_range_matches(
        val: &Value,
        range: &qdrant_client::qdrant::DatetimeRange,
    ) -> bool {
        let Some(epoch) = val.as_str().and_then(crate::document_fields::epoch_seconds) else {
            return false;
        };
        let at = qdrant::epoch_to_timestamp(epoch);
        let at = (at.seconds, at.nanos);
        let bound = |t: &Option<qdrant_client::qdrant::Timestamp>| t.map(|t| (t.seconds, t.nanos));
        bound(&range.gte).is_none_or(|b| at >= b)
            && bound(&range.lte).is_none_or(|b| at <= b)
            && bound(&range.gt).is_none_or(|b| at > b)
            && bound(&range.lt).is_none_or(|b| at < b)
    }

    fn qdrant_range_matches(val: &Value, range: &qdrant_client::qdrant::Range) -> bool {
        let n = match val {
            Value::Number(n) => n.as_f64(),
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        };
        let Some(n) = n else { return false };
        if let Some(gte) = range.gte
            && n < gte
        {
            return false;
        }
        if let Some(lte) = range.lte
            && n > lte
        {
            return false;
        }
        if let Some(gt) = range.gt
            && n <= gt
        {
            return false;
        }
        if let Some(lt) = range.lt
            && n >= lt
        {
            return false;
        }
        true
    }

    fn qdrant_match_matches(val: &Value, m: &qdrant_client::qdrant::Match) -> bool {
        use qdrant_client::qdrant::r#match::MatchValue;
        let Some(mv) = &m.match_value else {
            return false;
        };
        // Qdrant matches an array-valued payload field by checking whether any
        // element matches, not by comparing the whole array as one value.
        let candidates: Vec<&Value> = match val {
            Value::Array(items) => items.iter().collect(),
            other => vec![other],
        };
        candidates.into_iter().any(|c| match mv {
            MatchValue::Keyword(s) => c.as_str() == Some(s.as_str()),
            MatchValue::Integer(i) => c.as_i64() == Some(*i),
            MatchValue::Boolean(b) => c.as_bool() == Some(*b),
            MatchValue::Keywords(ks) => ks.strings.iter().any(|s| c.as_str() == Some(s.as_str())),
            MatchValue::Integers(is) => is.integers.iter().any(|i| c.as_i64() == Some(*i)),
            other => panic!(
                "equivalence test's evaluator does not model match variant {other:?} — \
                 lower_field_filters should not be producing this shape"
            ),
        })
    }

    fn qdrant_condition_matches(cond: &qdrant_client::qdrant::Condition, payload: &Value) -> bool {
        use qdrant_client::qdrant::condition::ConditionOneOf;
        match cond
            .condition_one_of
            .as_ref()
            .expect("every lowered condition sets one variant")
        {
            ConditionOneOf::Field(fc) => {
                // Every value at the key is a candidate: Qdrant matches an array (or
                // a `[]` projection) when any element does.
                let values = payload_at(payload, &fc.key);
                let candidates: Vec<&Value> = values
                    .iter()
                    .flat_map(|v| match v {
                        Value::Array(items) => items.iter().collect::<Vec<_>>(),
                        other => vec![*other],
                    })
                    .collect();
                candidates.into_iter().any(|val| {
                    if let Some(range) = &fc.range {
                        qdrant_range_matches(val, range)
                    } else if let Some(range) = &fc.datetime_range {
                        qdrant_datetime_range_matches(val, range)
                    } else if let Some(m) = &fc.r#match {
                        qdrant_match_matches(val, m)
                    } else {
                        panic!(
                            "equivalence test's evaluator only models match/range/datetime \
                             field conditions: {fc:?}"
                        )
                    }
                })
            }
            ConditionOneOf::IsNull(n) => payload_at(payload, &n.key)
                .first()
                .is_none_or(|v| v.is_null()),
            ConditionOneOf::Nested(n) => {
                // Every condition of the nested filter must hold within ONE element of
                // the array at the key (Qdrant's nested object filter).
                let filter = n.filter.as_ref().expect("a nested condition has a filter");
                assert!(
                    filter.should.is_empty()
                        && filter.must_not.is_empty()
                        && filter.min_should.is_none(),
                    "unexpected nested Filter shape from lower_field_filters: {filter:?}"
                );
                payload_at(payload, &n.key)
                    .into_iter()
                    .filter_map(|v| v.as_array())
                    .flatten()
                    .any(|element| {
                        filter
                            .must
                            .iter()
                            .all(|c| qdrant_condition_matches(c, element))
                    })
            }
            ConditionOneOf::Filter(f) => {
                // `lower_field_filters` produces a bare `should` OR (multi-value
                // boolean or datetime equality) and `unsatisfiable`'s must + must_not
                // pair; anything else means this evaluator is out of date.
                assert!(f.min_should.is_none(), "unexpected Filter shape: {f:?}");
                (f.should.is_empty()
                    || f.should
                        .iter()
                        .any(|c| qdrant_condition_matches(c, payload)))
                    && f.must.iter().all(|c| qdrant_condition_matches(c, payload))
                    && !f
                        .must_not
                        .iter()
                        .any(|c| qdrant_condition_matches(c, payload))
            }
            other => panic!(
                "equivalence test's evaluator does not model condition variant {other:?} — \
                 lower_field_filters should not be producing this shape"
            ),
        }
    }

    /// The frontmatter part of the Qdrant payload production writes for `fm` under
    /// `schema` (`ingest::frontmatter_payload`, #350), as one JSON object.
    fn production_payload(
        path: &str,
        fm: &HashMap<String, Value>,
        schema: &crate::schema::ResolvedSchema,
    ) -> Value {
        Value::Object(
            crate::ingest::frontmatter_payload(fm, path, &Default::default(), schema)
                .into_iter()
                .collect(),
        )
    }

    /// Seed [`equivalence_db`], run `filters` through both the real
    /// enumeration-mode SQL path (`StateDb::query_documents`) and
    /// `qdrant::lower_field_filters` (evaluated offline against the same
    /// fixture's frontmatter — see this section's header comment), then assert
    /// both name exactly the same document set.
    async fn assert_query_and_enumeration_modes_agree(
        filters: Vec<(String, FieldFilter)>,
        indexed: HashMap<String, qdrant::IndexKind>,
        expected: &[&str],
    ) {
        let (db, _dir) = equivalence_db().await;

        let sql_result = db
            .query_documents(&DocumentQuery {
                filters: filters.clone(),
                ..Default::default()
            })
            .await
            .unwrap();
        let mut sql_paths: Vec<&str> = sql_result
            .documents
            .iter()
            .map(|d| d.file_path.as_str())
            .collect();
        sql_paths.sort();

        let conditions = qdrant::lower_field_filters(&filters, &indexed)
            .expect("query mode should accept this filter");
        let mut qdrant_paths: Vec<&str> = equivalence_docs()
            .into_iter()
            .filter(|(path, fm)| {
                let payload =
                    production_payload(path, fm, &crate::schema::ResolvedSchema::default());
                conditions
                    .iter()
                    .all(|c| qdrant_condition_matches(c, &payload))
            })
            .map(|(path, _)| path)
            .collect();
        qdrant_paths.sort();

        let mut expected_sorted: Vec<&str> = expected.to_vec();
        expected_sorted.sort();

        assert_eq!(
            sql_paths, expected_sorted,
            "enumeration mode (SQL) produced a different document set than expected"
        );
        assert_eq!(
            qdrant_paths, expected_sorted,
            "query mode (Qdrant conditions) produced a different document set than expected"
        );
    }

    /// #350: every case of the shared typed fixture (`filter_parity`) finds the same
    /// documents through the metadata index (enumeration mode) and through the
    /// lowered Qdrant conditions evaluated offline against the payload production
    /// writes — `fm.` keys, kind coercion, datetime ranges and nested conditions
    /// included. `qdrant::tests::sqlite_and_qdrant_filters_agree_on_the_typed_fixture`
    /// runs the same matrix against a live Qdrant.
    #[tokio::test]
    async fn typed_fixture_filters_agree_offline_across_both_modes() {
        let dir = TempDir::new().unwrap();
        let schemas = filter_parity::schemas(dir.path());
        let db = filter_parity::seed_db(&dir.path().join("state.db"), &schemas).await;
        let indexed = filter_parity::indexed(&schemas);

        for (field, raw, expected) in filter_parity::cases() {
            let filters = filter_parity::parse(field, &raw);
            let sql = filter_parity::sqlite_paths(&db, filters.clone()).await;
            assert_eq!(sql, expected, "enumeration mode: {field} {raw}");

            let conditions = qdrant::lower_field_filters(&filters, &indexed)
                .unwrap_or_else(|e| panic!("query mode refused {field} {raw}: {e}"));
            let mut offline: Vec<String> = filter_parity::docs()
                .into_iter()
                .filter(|(path, fm)| {
                    let payload = Value::Object(
                        filter_parity::payload(path, fm, &schemas)
                            .into_iter()
                            .collect(),
                    );
                    conditions
                        .iter()
                        .all(|c| qdrant_condition_matches(c, &payload))
                })
                .map(|(path, _)| path.to_string())
                .collect();
            offline.sort();
            assert_eq!(offline, expected, "query mode (offline): {field} {raw}");
        }
    }

    /// The one documented divergence of the typed fixture (#350): equality on a
    /// timestamp compares the instant in query mode (the payload holds RFC 3339 UTC
    /// text) but the text as written in enumeration mode (`value_text`), so the same
    /// instant spelled with another offset matches only with a query. Pinned so a
    /// change to either side is deliberate.
    #[tokio::test]
    async fn timestamp_equality_matches_the_instant_only_in_query_mode() {
        let dir = TempDir::new().unwrap();
        let schemas = filter_parity::schemas(dir.path());
        let db = filter_parity::seed_db(&dir.path().join("state.db"), &schemas).await;
        let indexed = filter_parity::indexed(&schemas);
        // a.md's `seen` is written `2024-01-02T10:00:00+02:00`.
        let filters = filter_parity::parse("seen", &serde_json::json!("2024-01-02T08:00:00Z"));

        assert!(
            filter_parity::sqlite_paths(&db, filters.clone())
                .await
                .is_empty()
        );
        let conditions = qdrant::lower_field_filters(&filters, &indexed).unwrap();
        let matched: Vec<&str> = filter_parity::docs()
            .into_iter()
            .filter(|(path, fm)| {
                let payload = Value::Object(
                    filter_parity::payload(path, fm, &schemas)
                        .into_iter()
                        .collect(),
                );
                conditions
                    .iter()
                    .all(|c| qdrant_condition_matches(c, &payload))
            })
            .map(|(path, _)| path)
            .collect();
        assert_eq!(matched, vec!["a.md"]);
        // Spelled as written, both modes agree.
        let as_written =
            filter_parity::parse("seen", &serde_json::json!("2024-01-02T10:00:00+02:00"));
        assert_eq!(
            filter_parity::sqlite_paths(&db, as_written).await,
            vec!["a.md".to_string()]
        );
    }

    #[tokio::test]
    async fn query_and_enumeration_agree_on_scalar_equality() {
        assert_query_and_enumeration_modes_agree(
            vec![("type".into(), FieldFilter::AnyOf(vec!["guide".into()]))],
            HashMap::from([("type".to_string(), qdrant::IndexKind::Keyword)]),
            &["a.md", "d.md"],
        )
        .await;
    }

    #[tokio::test]
    async fn query_and_enumeration_agree_on_any_of() {
        assert_query_and_enumeration_modes_agree(
            vec![(
                "tags".into(),
                FieldFilter::AnyOf(vec!["breakfast".into(), "zfs".into()]),
            )],
            HashMap::from([("tags".to_string(), qdrant::IndexKind::Keyword)]),
            &["b.md", "c.md", "d.md"],
        )
        .await;
    }

    #[tokio::test]
    async fn query_and_enumeration_agree_on_all_of() {
        assert_query_and_enumeration_modes_agree(
            vec![(
                "tags".into(),
                FieldFilter::AllOf(vec!["recipe".into(), "dinner".into()]),
            )],
            HashMap::from([("tags".to_string(), qdrant::IndexKind::Keyword)]),
            &["a.md", "c.md"],
        )
        .await;
    }

    #[tokio::test]
    async fn query_and_enumeration_agree_on_numeric_range() {
        assert_query_and_enumeration_modes_agree(
            vec![(
                "planning.prep_minutes".into(),
                FieldFilter::Range {
                    gte: Some(10.0),
                    lte: None,
                    gt: None,
                    lt: Some(45.0),
                },
            )],
            HashMap::from([(
                "planning.prep_minutes".to_string(),
                qdrant::IndexKind::Integer,
            )]),
            &["a.md", "d.md"],
        )
        .await;
    }

    #[tokio::test]
    async fn query_and_enumeration_agree_on_a_nested_dot_path_field() {
        assert_query_and_enumeration_modes_agree(
            vec![(
                "planning.tested".into(),
                FieldFilter::AnyOf(vec!["true".into()]),
            )],
            HashMap::from([("planning.tested".to_string(), qdrant::IndexKind::Bool)]),
            &["a.md", "b.md", "d.md"],
        )
        .await;
    }

    #[tokio::test]
    async fn query_mode_rejects_float_equality_that_enumeration_mode_accepts() {
        // Deliberate divergence, not a bug: exact float equality is unreliable,
        // so query mode refuses it outright (`lower_field_filters` -> `Err`)
        // while enumeration mode's SQL just does a text comparison and accepts
        // it. Pin both halves so a future change can't silently narrow or widen
        // either side without this test noticing.
        let (db, _dir) = equivalence_db().await;
        let filters = vec![("rating".to_string(), FieldFilter::AnyOf(vec!["4.5".into()]))];

        let sql_result = db
            .query_documents(&DocumentQuery {
                filters: filters.clone(),
                ..Default::default()
            })
            .await
            .unwrap();
        let mut sql_paths: Vec<&str> = sql_result
            .documents
            .iter()
            .map(|d| d.file_path.as_str())
            .collect();
        sql_paths.sort();
        assert_eq!(
            sql_paths,
            vec!["a.md", "c.md"],
            "enumeration mode must still allow float equality"
        );

        let indexed = HashMap::from([("rating".to_string(), qdrant::IndexKind::Float)]);
        let err = qdrant::lower_field_filters(&filters, &indexed).unwrap_err();
        assert!(
            err.contains("float"),
            "query mode must reject float equality; got: {err}"
        );
    }

    #[tokio::test]
    async fn path_prefix_narrows_to_a_folder() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            path_prefix: Some("kitchen/recipes/".into()),
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        assert_eq!(result.total, 3);
        assert!(paths(&result).iter().all(|p| p.starts_with("kitchen/")));
    }

    #[tokio::test]
    async fn path_prefix_escapes_like_wildcards() {
        // `_` is a single-character LIKE wildcard. Unescaped, this prefix would also
        // match a hypothetical `kitchen/recipesX/`.
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            path_prefix: Some("kitchen/recipes/stir_fry".into()),
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        assert_eq!(paths(&result), vec!["kitchen/recipes/stir_fry.md"]);

        // And a prefix whose underscore does not literally match finds nothing.
        let query = DocumentQuery {
            path_prefix: Some("kitchen/recipes/stir_fr".into()),
            ..Default::default()
        };
        assert_eq!(db.query_documents(&query).await.unwrap().total, 1);

        let query = DocumentQuery {
            path_prefix: Some("kitchen/recipes/stirXfry".into()),
            ..Default::default()
        };
        assert_eq!(
            db.query_documents(&query).await.unwrap().total,
            0,
            "underscore must not behave as a wildcard"
        );
    }

    #[tokio::test]
    async fn path_prefix_matches_a_mid_path_fragment() {
        // #182: the discovery case the old prefix-only match could not serve — a
        // fragment the caller does not know the start of.
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            path_prefix: Some("recipes/stir".into()),
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        assert_eq!(paths(&result), vec!["kitchen/recipes/stir_fry.md"]);
    }

    #[tokio::test]
    async fn path_prefix_matches_a_bare_filename_fragment() {
        // The half-remembered filename from the issue, with no path context at all.
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            path_prefix: Some("stir_fr".into()),
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        assert_eq!(paths(&result), vec!["kitchen/recipes/stir_fry.md"]);
    }

    #[tokio::test]
    async fn path_prefix_is_case_insensitive() {
        // Inherited from SQLite's `LIKE`, which folds ASCII. Documented as such
        // because it is behavior we rely on rather than implement.
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            path_prefix: Some("STIR_FRY".into()),
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        assert_eq!(paths(&result), vec!["kitchen/recipes/stir_fry.md"]);
    }

    #[tokio::test]
    async fn paths_matching_returns_every_match_under_the_cap() {
        let (db, _dir) = seeded_db().await;
        let matches = db.paths_matching("recipes", 100).await.unwrap();

        assert_eq!(matches.total, 3);
        assert_eq!(
            matches.paths,
            vec![
                "kitchen/recipes/chili.md",
                "kitchen/recipes/congee.md",
                "kitchen/recipes/stir_fry.md",
            ],
            "ordered by path, so a capped resolution is at least deterministic"
        );
        assert!(!matches.truncated());
    }

    #[tokio::test]
    async fn paths_matching_reports_a_total_beyond_the_cap() {
        // The cap must never look like exhaustion — `total` is what tells a caller
        // its filter is an incomplete picture of what matches.
        let (db, _dir) = seeded_db().await;
        let matches = db.paths_matching("recipes", 2).await.unwrap();

        assert_eq!(matches.paths.len(), 2);
        assert_eq!(matches.total, 3);
        assert!(matches.truncated());
    }

    #[tokio::test]
    async fn paths_matching_escapes_like_wildcards() {
        // Same wildcard hazard as `path_prefix_escapes_like_wildcards`, on the
        // resolution path this time: `_` must be a literal underscore.
        let (db, _dir) = seeded_db().await;

        let hit = db.paths_matching("stir_fry", 100).await.unwrap();
        assert_eq!(hit.paths, vec!["kitchen/recipes/stir_fry.md"]);

        let miss = db.paths_matching("stirXfry", 100).await.unwrap();
        assert_eq!(
            miss.total, 0,
            "underscore must not behave as a wildcard during resolution either"
        );

        // `%` likewise — a needle of pure wildcards must match nothing, not
        // everything.
        let percent = db.paths_matching("%", 100).await.unwrap();
        assert_eq!(percent.total, 0);
    }

    #[tokio::test]
    async fn paths_matching_and_query_documents_agree_on_the_same_needle() {
        // #182/#259: the two search modes resolve paths through these two calls.
        // If they ever disagree, one needle means two things again.
        let (db, _dir) = seeded_db().await;
        for needle in ["recipes", "stir_fr", "STIR_FRY", "sysadmin", "zfs.md", "/"] {
            let resolved = db.paths_matching(needle, 100).await.unwrap();
            let query = DocumentQuery {
                path_prefix: Some(needle.into()),
                ..Default::default()
            };
            let enumerated = db.query_documents(&query).await.unwrap();

            assert_eq!(
                resolved.total, enumerated.total,
                "needle '{needle}' matched differently in the two modes"
            );
            let mut a = resolved.paths.clone();
            let mut b = paths(&enumerated);
            a.sort();
            b.sort();
            assert_eq!(a, b, "needle '{needle}' resolved to a different set");
        }
    }

    #[tokio::test]
    async fn total_reflects_all_matches_not_just_the_page() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            limit: 2,
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        assert_eq!(result.documents.len(), 2);
        assert_eq!(result.total, 4, "truncation must never be silent");
        assert!(result.has_more(0));
    }

    #[tokio::test]
    async fn offset_pages_deterministically() {
        let (db, _dir) = seeded_db().await;
        let page = |offset| DocumentQuery {
            limit: 2,
            offset,
            ..Default::default()
        };

        let first = db.query_documents(&page(0)).await.unwrap();
        let second = db.query_documents(&page(2)).await.unwrap();

        assert_eq!(
            paths(&first),
            vec!["kitchen/recipes/chili.md", "kitchen/recipes/congee.md"]
        );
        assert_eq!(
            paths(&second),
            vec!["kitchen/recipes/stir_fry.md", "sysadmin/zfs.md"]
        );
        assert!(!second.has_more(2));
    }

    #[tokio::test]
    async fn ordering_by_mtime_still_paginates_deterministically() {
        // Every seeded document shares mtime 100, so only the file_path tiebreaker
        // keeps paging stable.
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            order_by: OrderBy::Mtime,
            order_desc: true,
            limit: 2,
            ..Default::default()
        };

        let first = db.query_documents(&query).await.unwrap();
        for _ in 0..5 {
            assert_eq!(
                paths(&db.query_documents(&query).await.unwrap()),
                paths(&first)
            );
        }
    }

    /// Rebuild the exact page query `query_documents` runs for a given `DocumentQuery`
    /// — same `push_where`, same `ORDER BY` construction — prefixed with `EXPLAIN QUERY
    /// PLAN`, and return the concatenated `detail` column of the plan.
    ///
    /// `query_documents` doesn't expose its SQL, so this mirrors it exactly rather than
    /// re-deriving a plan from first principles: a mismatch here would mean the test
    /// stopped testing what production runs, not that the index is wrong.
    async fn explain_page_query(db: &StateDb, query: &DocumentQuery) -> String {
        let mut builder: QueryBuilder<Sqlite> = QueryBuilder::new(
            "EXPLAIN QUERY PLAN SELECT d.file_path, d.title, d.description, d.mtime, \
             d.indexed_at, d.frontmatter FROM documents d",
        );
        StateDb::push_where(&mut builder, query);
        builder.push(" ORDER BY ");
        builder.push(query.order_by.column());
        builder.push(if query.order_desc { " DESC" } else { " ASC" });
        if query.order_by != OrderBy::Path {
            builder.push(", d.file_path ASC");
        }
        builder.push(" LIMIT ");
        builder.push_bind(query.limit as i64);
        builder.push(" OFFSET ");
        builder.push_bind(query.offset as i64);

        // EXPLAIN QUERY PLAN rows are (id, parent, notused, detail).
        let rows: Vec<(i64, i64, i64, String)> = builder
            .build_query_as()
            .fetch_all(db.pool_for_test())
            .await
            .unwrap();
        rows.into_iter()
            .map(|(_, _, _, detail)| detail)
            .collect::<Vec<_>>()
            .join(" | ")
    }

    #[tokio::test]
    async fn query_documents_order_by_uses_an_index_not_a_temp_sort() {
        // Regression for #90: every supported (order_by, order_desc) combination must
        // be servable straight from an index. A plan containing "USE TEMP B-TREE" means
        // SQLite is sorting the whole filtered set in memory before LIMIT/OFFSET can
        // trim it — exactly the bug the new indexes exist to fix. Asserting this on the
        // real planner output (rather than just asserting rows come back correct) is
        // the point: a wrong-shaped or missing index still returns correct rows, just
        // slowly, and a correctness-only test would never catch that regressing.
        let (db, _dir) = seeded_db().await;

        for order_by in [
            OrderBy::Path,
            OrderBy::Title,
            OrderBy::Mtime,
            OrderBy::IndexedAt,
        ] {
            for order_desc in [false, true] {
                let query = DocumentQuery {
                    order_by,
                    order_desc,
                    ..Default::default()
                };
                let plan = explain_page_query(&db, &query).await;
                assert!(
                    !plan.to_ascii_uppercase().contains("TEMP B-TREE"),
                    "order_by={order_by:?} desc={order_desc} should be served by an \
                     index without a temp-b-tree sort; plan: {plan}"
                );
                assert!(
                    plan.to_ascii_uppercase().contains("INDEX"),
                    "order_by={order_by:?} desc={order_desc} should scan via an index; \
                     plan: {plan}"
                );
            }
        }
    }

    #[tokio::test]
    async fn query_documents_order_by_with_path_prefix_still_uses_an_index() {
        // A filter alongside the ordering (the common real-world shape: "recent docs
        // under sysadmin/") must not knock the planner off the ordering index either.
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            path_prefix: Some("kitchen/".into()),
            order_by: OrderBy::Mtime,
            order_desc: true,
            ..Default::default()
        };
        let plan = explain_page_query(&db, &query).await;
        assert!(
            !plan.to_ascii_uppercase().contains("TEMP B-TREE"),
            "filtered + ordered query should still avoid a temp-b-tree sort; plan: {plan}"
        );
    }

    #[tokio::test]
    async fn empty_any_of_matches_nothing_rather_than_everything() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![("tags".into(), FieldFilter::AnyOf(vec![]))],
            ..Default::default()
        };

        assert_eq!(db.query_documents(&query).await.unwrap().total, 0);
    }

    #[tokio::test]
    async fn fields_projection_preserves_nesting() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            path_prefix: Some("kitchen/recipes/chili.md".into()),
            fields: Some(vec!["planning.prep_minutes".into(), "type".into()]),
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        let fm = &result.documents[0].frontmatter;
        assert_eq!(fm["planning"]["prep_minutes"], serde_json::json!(20));
        assert_eq!(fm["type"], serde_json::json!("reference"));
        assert!(fm.get("tags").is_none(), "unrequested fields are omitted");
        assert!(
            fm["planning"].get("tested").is_none(),
            "sibling nested fields are omitted"
        );
    }

    #[tokio::test]
    async fn full_frontmatter_is_returned_when_no_fields_requested() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            path_prefix: Some("kitchen/recipes/chili.md".into()),
            ..Default::default()
        };
        let result = db.query_documents(&query).await.unwrap();

        let fm = &result.documents[0].frontmatter;
        assert_eq!(fm["planning"]["tested"], serde_json::json!(true));
        assert_eq!(fm["tags"], serde_json::json!(["recipe", "dinner"]));
    }

    #[tokio::test]
    async fn order_by_parse_rejects_unknown_values() {
        assert_eq!(OrderBy::parse("path"), Ok(OrderBy::Path));
        assert_eq!(OrderBy::parse("MTIME"), Ok(OrderBy::Mtime));
        assert!(OrderBy::parse("d.file_path; DROP TABLE documents").is_err());
    }

    #[tokio::test]
    async fn offset_at_or_past_the_end_returns_an_empty_page_not_an_error() {
        let (db, _dir) = seeded_db().await;

        let at_end = db
            .query_documents(&DocumentQuery {
                offset: 4,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(at_end.documents.is_empty());
        assert_eq!(at_end.total, 4);
        assert!(!at_end.has_more(4));

        let past_end = db
            .query_documents(&DocumentQuery {
                offset: 999,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(past_end.documents.is_empty());
        assert_eq!(
            past_end.total, 4,
            "total still reports the full match count"
        );
        assert!(!past_end.has_more(999));
    }

    #[tokio::test]
    async fn ordering_by_a_nullable_column_paginates_without_gaps_or_repeats() {
        // `title` is nullable, and SQLite groups NULLs together — only the file_path
        // tiebreaker keeps paging stable across the NULL group.
        let (db, _dir) = test_db().await;
        for (path, title) in [
            ("a.md", Some("Zebra")),
            ("b.md", None),
            ("c.md", Some("Apple")),
            ("d.md", None),
        ] {
            let mut fm: HashMap<String, Value> = HashMap::new();
            if let Some(t) = title {
                fm.insert("title".into(), Value::String(t.into()));
            }
            fm.insert("type".into(), Value::String("reference".into()));
            db.upsert_document_metadata(path, &fm, 100, "h", 1)
                .await
                .unwrap();
        }

        let page = |offset| DocumentQuery {
            order_by: OrderBy::Title,
            limit: 2,
            offset,
            ..Default::default()
        };
        let first = db.query_documents(&page(0)).await.unwrap();
        let second = db.query_documents(&page(2)).await.unwrap();

        let mut seen: Vec<&str> = paths(&first);
        seen.extend(paths(&second));
        seen.sort();
        assert_eq!(
            seen,
            vec!["a.md", "b.md", "c.md", "d.md"],
            "every document appears exactly once across the two pages"
        );
    }

    #[tokio::test]
    async fn promoted_fields_are_filterable_through_their_columns() {
        // title/description are excluded from the projection, so without the dedicated
        // column path a filter on them would silently match nothing.
        let (db, _dir) = seeded_db().await;

        let result = db
            .query_documents(&DocumentQuery {
                filters: vec![("title".into(), FieldFilter::AnyOf(vec!["Chili".into()]))],
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(paths(&result), vec!["kitchen/recipes/chili.md"]);
    }

    #[tokio::test]
    async fn all_of_on_a_scalar_column_is_unsatisfiable_not_widened() {
        // title holds one value, so requiring two is impossible. Reusing the any-of
        // IN(...) form would silently return documents matching either.
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![(
                "title".into(),
                FieldFilter::AllOf(vec!["Chili".into(), "Congee".into()]),
            )],
            ..Default::default()
        };

        assert_eq!(db.query_documents(&query).await.unwrap().total, 0);
    }

    #[tokio::test]
    async fn all_of_with_one_value_still_matches_a_scalar_column() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![("title".into(), FieldFilter::AllOf(vec!["Chili".into()]))],
            ..Default::default()
        };

        assert_eq!(
            paths(&db.query_documents(&query).await.unwrap()),
            vec!["kitchen/recipes/chili.md"]
        );
    }

    #[tokio::test]
    async fn description_is_filterable_too() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![(
                "description".into(),
                FieldFilter::AnyOf(vec!["Chili description".into()]),
            )],
            ..Default::default()
        };

        assert_eq!(
            paths(&db.query_documents(&query).await.unwrap()),
            vec!["kitchen/recipes/chili.md"]
        );
    }

    #[tokio::test]
    async fn empty_all_of_matches_nothing_like_empty_any_of() {
        let (db, _dir) = seeded_db().await;
        let query = DocumentQuery {
            filters: vec![("tags".into(), FieldFilter::AllOf(vec![]))],
            ..Default::default()
        };

        assert_eq!(
            db.query_documents(&query).await.unwrap().total,
            0,
            "an empty required set is unsatisfiable, not absent"
        );
    }

    #[tokio::test]
    async fn deleted_documents_leave_the_listing() {
        let (db, _dir) = seeded_db().await;
        db.delete_document("sysadmin/zfs.md").await.unwrap();

        let result = db.query_documents(&DocumentQuery::default()).await.unwrap();
        assert_eq!(result.total, 3);
        assert!(!paths(&result).contains(&"sysadmin/zfs.md"));
    }

    #[tokio::test]
    async fn all_paths_returns_every_indexed_document_path() {
        // Backs `retrieval::get_document`'s fuzzy fallback (#87): it needs the
        // complete, current path list, unfiltered and with no cap.
        let (db, _dir) = seeded_db().await;
        let mut result = db.all_paths().await.unwrap();
        result.sort();
        assert_eq!(
            result,
            vec![
                "kitchen/recipes/chili.md",
                "kitchen/recipes/congee.md",
                "kitchen/recipes/stir_fry.md",
                "sysadmin/zfs.md",
            ]
        );
    }

    #[tokio::test]
    async fn all_paths_reflects_deletions() {
        let (db, _dir) = seeded_db().await;
        db.delete_document("sysadmin/zfs.md").await.unwrap();

        let result = db.all_paths().await.unwrap();
        assert_eq!(result.len(), 3);
        assert!(!result.contains(&"sysadmin/zfs.md".to_string()));
    }

    #[tokio::test]
    async fn all_paths_on_an_empty_database_is_empty() {
        let (db, _dir) = test_db().await;
        assert!(db.all_paths().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn clear_also_wipes_document_metadata() {
        // A full reindex clears state BEFORE orphan detection reads it, so orphans are
        // never computed on a full run. If clear() left `documents` behind, a file
        // deleted from disk would remain listed forever with no run able to detect it.
        let (db, _dir) = test_db().await;
        db.upsert("gone.md", "h", 1, "", "", 0, 0).await.unwrap();
        db.upsert_document_metadata("gone.md", &recipe_frontmatter(), 1, "h", 1)
            .await
            .unwrap();

        db.clear().await.unwrap();

        assert_eq!(db.count().await.unwrap(), 0);
        assert_eq!(
            db.document_count().await.unwrap(),
            0,
            "metadata must not survive a full-reindex clear"
        );
        assert!(field_rows(&db, "gone.md").await.is_empty());
    }

    #[tokio::test]
    async fn clear_also_wipes_document_links_so_no_phantom_links_in_survives() {
        // `document_links` has no foreign key to `documents`, so nothing cascades into
        // it. A file deleted while no reconcile ran, then `index --full`, must not keep
        // showing up as a `links_in` source of the document it linked to.
        let (db, _dir) = test_db().await;
        db.upsert("gone.md", "h", 1, "", "", 0, 0).await.unwrap();
        db.replace_links("gone.md", "markdown", &[("kept.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("gone.md", "semantic", &[("kept.md".to_string(), Some(0.9))])
            .await
            .unwrap();

        db.clear().await.unwrap();

        assert!(db.all_links().await.unwrap().is_empty());
        assert!(
            db.links_targeting("kept.md", "markdown")
                .await
                .unwrap()
                .is_empty(),
            "a cleared index must not report a phantom inbound link"
        );
    }

    #[tokio::test]
    async fn reprojection_reads_current_data_not_a_snapshot() {
        // The whole point of re-reading inside the write transaction: a value updated
        // after the run began must be reprojected as its NEW value, never reverted.
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("r.md", &recipe_frontmatter(), 1, "h", 1)
            .await
            .unwrap();

        let mut updated = recipe_frontmatter();
        updated.insert(
            "planning".into(),
            serde_json::json!({ "prep_minutes": 99, "needs_recipe": true }),
        );
        db.upsert_document_metadata("r.md", &updated, 2, "h2", 1)
            .await
            .unwrap();

        db.reproject_all_fields(&no_schemas()).await.unwrap();

        let rows = field_rows(&db, "r.md").await;
        assert!(
            rows.contains(&("planning.prep_minutes".into(), "99".into(), Some(99.0))),
            "must reproject the current value, got: {rows:?}"
        );
    }

    #[tokio::test]
    async fn reprojection_skips_a_corrupt_row_and_finishes_the_rest() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("good.md", &recipe_frontmatter(), 1, "h", 1)
            .await
            .unwrap();
        db.upsert_document_metadata("zz-later.md", &recipe_frontmatter(), 1, "h", 1)
            .await
            .unwrap();

        // Corrupt one row's stored JSON; it sorts before the other.
        sqlx::query("UPDATE documents SET frontmatter = ? WHERE file_path = ?")
            .bind("{not json")
            .bind("good.md")
            .execute(db.pool_for_test())
            .await
            .unwrap();

        let report = db
            .reproject_all_fields(&no_schemas())
            .await
            .expect("one bad row must not fail the whole repair run");

        assert_eq!(
            report.reprojected, 1,
            "the healthy document was still reprojected"
        );
        assert_eq!(report.unparseable, 1);
        assert_eq!(report.failed, 0);
        assert!(!field_rows(&db, "zz-later.md").await.is_empty());
    }

    #[tokio::test]
    async fn document_metadata_is_independent_of_indexed_files() {
        // The two tables track different things; an existing deployment has one
        // populated and the other empty until a backfill runs.
        let (db, _dir) = test_db().await;
        db.upsert("r.md", "h1", 2, "", "", 0, 0).await.unwrap();

        assert_eq!(db.count().await.unwrap(), 1);
        assert_eq!(db.document_count().await.unwrap(), 0);
    }

    // -- pagination / chunking boundaries -------------------------------------
    //
    // `fetch_indexed_files_page`, `get_many`, and `get_document_hashes_many` are
    // exercised end-to-end by `ingest.rs`'s `scan_for_dirty` tests, but those
    // fixtures hold only a handful of files — never enough to reach a second page or
    // a second 500-param chunk. That leaves exactly the boundary these methods exist
    // for untested: an off-by-one here would silently skip files during a reconcile
    // sweep, and nothing would ever error — search would just quietly go stale. The
    // tests below insert through hand-rolled multi-row `INSERT`s rather than
    // `upsert`/`upsert_document_metadata` (one round trip per row) specifically to
    // keep the 500-1000+ row cases fast.

    /// Insert `entries` (file_path, content_hash) into `indexed_files` via a handful
    /// of multi-row `INSERT`s rather than one round trip per row, so the 500+ row
    /// boundary tests below stay fast. Each statement stays well under SQLite's
    /// 999-bound-parameter ceiling on its own — this helper's job is to seed data
    /// quickly, not to probe that limit itself.
    async fn bulk_insert_indexed_files(db: &StateDb, entries: &[(String, String)]) {
        const ROWS_PER_STATEMENT: usize = 100;
        for chunk in entries.chunks(ROWS_PER_STATEMENT) {
            let mut builder = QueryBuilder::<Sqlite>::new(
                "INSERT INTO indexed_files (file_path, content_hash, chunk_count, schema_hash, mtime, size) ",
            );
            builder.push_values(chunk, |mut b, (path, hash)| {
                b.push_bind(path)
                    .push_bind(hash)
                    .push_bind(1i64)
                    .push_bind("")
                    .push_bind(0i64)
                    .push_bind(0i64);
            });
            builder.build().execute(db.pool_for_test()).await.unwrap();
        }
    }

    /// Insert `entries` (file_path, content_hash) directly into `documents`,
    /// bypassing `upsert_document_metadata`'s per-row transaction and field
    /// projection — this helper only needs to produce rows `get_document_hashes_many`
    /// can read, not realistic frontmatter, and it needs to do it fast.
    async fn bulk_insert_documents(db: &StateDb, entries: &[(String, String)]) {
        const ROWS_PER_STATEMENT: usize = 100;
        for chunk in entries.chunks(ROWS_PER_STATEMENT) {
            let mut builder = QueryBuilder::<Sqlite>::new(
                "INSERT INTO documents (file_path, frontmatter, mtime, content_hash, chunk_count) ",
            );
            builder.push_values(chunk, |mut b, (path, hash)| {
                b.push_bind(path)
                    .push_bind("{}")
                    .push_bind(0i64)
                    .push_bind(hash)
                    .push_bind(1i64);
            });
            builder.build().execute(db.pool_for_test()).await.unwrap();
        }
    }

    // -- fetch_indexed_files_page ---------------------------------------------

    /// Walk `fetch_indexed_files_page` to exhaustion at `page_size`, reassembling
    /// every returned `file_path` (in page order) and counting how many pages it
    /// took. Guards against a regression that makes the loop never terminate — the
    /// exact failure an off-by-one in the `LIMIT`/`OFFSET` boundary would cause.
    async fn drain_all_pages(db: &StateDb, page_size: i64) -> (Vec<String>, usize) {
        let mut all = Vec::new();
        let mut offset = 0i64;
        let mut pages = 0usize;
        loop {
            let page = db
                .fetch_indexed_files_page(page_size, offset)
                .await
                .unwrap();
            if page.is_empty() {
                break;
            }
            pages += 1;
            all.extend(page.into_iter().map(|r| r.file_path));
            offset += page_size;
            assert!(
                pages < 10_000,
                "pagination did not terminate for page_size={page_size} — likely an \
                 off-by-one that never advances past a non-empty page"
            );
        }
        (all, pages)
    }

    #[tokio::test]
    async fn fetch_indexed_files_page_on_an_empty_table_returns_no_pages() {
        let (db, _dir) = test_db().await;
        let (all, pages) = drain_all_pages(&db, 5).await;
        assert_eq!(pages, 0);
        assert!(all.is_empty());
    }

    #[tokio::test]
    async fn pagination_covers_every_row_exactly_once_at_an_exact_page_multiple() {
        let (db, _dir) = test_db().await;
        let entries: Vec<(String, String)> = (0..10)
            .map(|i| (format!("f-{i:03}.md"), "h".to_string()))
            .collect();
        bulk_insert_indexed_files(&db, &entries).await;

        let (all, pages) = drain_all_pages(&db, 5).await;

        assert_eq!(
            pages, 2,
            "10 rows at page size 5 must take exactly 2 pages, not a trailing empty \
             third page"
        );
        let mut expected: Vec<String> = entries.into_iter().map(|(p, _)| p).collect();
        expected.sort();
        let mut got = all;
        got.sort();
        assert_eq!(
            got, expected,
            "every row must be reassembled exactly once — no gaps, no duplicates"
        );
    }

    #[tokio::test]
    async fn pagination_handles_one_less_than_a_page_multiple() {
        let (db, _dir) = test_db().await;
        let entries: Vec<(String, String)> = (0..9)
            .map(|i| (format!("f-{i:03}.md"), "h".to_string()))
            .collect();
        bulk_insert_indexed_files(&db, &entries).await;

        let (all, pages) = drain_all_pages(&db, 5).await;

        assert_eq!(
            pages, 2,
            "9 rows at page size 5: one full page of 5, one partial page of 4"
        );
        assert_eq!(all.len(), 9);
    }

    #[tokio::test]
    async fn pagination_handles_one_more_than_a_page_multiple() {
        let (db, _dir) = test_db().await;
        let entries: Vec<(String, String)> = (0..11)
            .map(|i| (format!("f-{i:03}.md"), "h".to_string()))
            .collect();
        bulk_insert_indexed_files(&db, &entries).await;

        let (all, pages) = drain_all_pages(&db, 5).await;

        assert_eq!(
            pages, 3,
            "11 rows at page size 5: two full pages plus one row spilling into a \
             third — an off-by-one here would either drop that 11th row or spin \
             forever re-fetching an empty page"
        );
        assert_eq!(all.len(), 11);
    }

    #[tokio::test]
    async fn fetch_indexed_files_page_limit_is_clamped_to_at_least_one() {
        // The implementation binds `limit.max(1)`. SQLite reads `LIMIT 0` as "zero
        // rows" — unlike the negative-limit-means-unbounded convention this file uses
        // elsewhere (`count_by_field`, `breakdown_fields`), so this method cannot
        // reuse that clamp. An unclamped 0 (or a negative value slipping through)
        // here would make `ingest::scan_for_dirty`'s paging loop spin forever, since
        // `offset` would never advance past a permanently empty page.
        let (db, _dir) = test_db().await;
        db.upsert("only.md", "h", 1, "", "", 0, 0).await.unwrap();

        let zero = db.fetch_indexed_files_page(0, 0).await.unwrap();
        assert_eq!(
            zero.len(),
            1,
            "a limit of 0 must be clamped to at least 1 row, not returned empty"
        );

        let negative = db.fetch_indexed_files_page(-5, 0).await.unwrap();
        assert_eq!(negative.len(), 1, "a negative limit must also clamp to 1");
    }

    // -- get_many ---------------------------------------------------------------

    /// Insert `n` distinct rows and assert `get_many` returns every single one with
    /// its correct content hash. `n` is chosen by each call site to sit at or
    /// straddle the 500-bind-parameter chunk boundary, so this checks both that
    /// nothing is dropped at a chunk seam and that no row's data gets attached to the
    /// wrong key while chunks are reassembled into one map.
    async fn assert_get_many_returns_every_requested_key(n: usize) {
        let (db, _dir) = test_db().await;
        let entries: Vec<(String, String)> = (0..n)
            .map(|i| (format!("file-{i:05}.md"), format!("hash-{i}")))
            .collect();
        bulk_insert_indexed_files(&db, &entries).await;

        let paths: Vec<String> = entries.iter().map(|(p, _)| p.clone()).collect();
        let result = db.get_many(&paths).await.unwrap();

        assert_eq!(result.len(), n, "every requested key must come back, n={n}");
        for (path, hash) in &entries {
            let row = result.get(path).unwrap_or_else(|| {
                panic!("'{path}' missing from get_many's result — a chunk-seam drop at n={n}")
            });
            assert_eq!(
                &row.content_hash, hash,
                "wrong row attached to '{path}' at n={n}"
            );
        }
    }

    #[tokio::test]
    async fn get_many_at_499_below_the_chunk_boundary() {
        assert_get_many_returns_every_requested_key(499).await;
    }

    #[tokio::test]
    async fn get_many_at_exactly_500_the_chunk_boundary() {
        assert_get_many_returns_every_requested_key(500).await;
    }

    #[tokio::test]
    async fn get_many_at_501_one_past_the_chunk_boundary() {
        assert_get_many_returns_every_requested_key(501).await;
    }

    #[tokio::test]
    async fn get_many_at_1001_spanning_three_chunks() {
        assert_get_many_returns_every_requested_key(1001).await;
    }

    #[tokio::test]
    async fn get_many_distinguishes_absent_keys_from_present_ones() {
        let (db, _dir) = test_db().await;
        let entries: Vec<(String, String)> = (0..5)
            .map(|i| (format!("present-{i}.md"), format!("hash-{i}")))
            .collect();
        bulk_insert_indexed_files(&db, &entries).await;

        let mut requested: Vec<String> = entries.iter().map(|(p, _)| p.clone()).collect();
        requested.push("missing-1.md".into());
        requested.push("missing-2.md".into());

        let result = db.get_many(&requested).await.unwrap();

        assert_eq!(
            result.len(),
            5,
            "only rows that actually exist come back; a missing key must not be \
             confused with a chunking bug"
        );
        assert!(!result.contains_key("missing-1.md"));
        assert!(!result.contains_key("missing-2.md"));
        for (path, _) in &entries {
            assert!(result.contains_key(path));
        }
    }

    #[tokio::test]
    async fn get_many_empty_input_returns_empty_map() {
        let (db, _dir) = test_db().await;
        assert!(db.get_many(&[]).await.unwrap().is_empty());
    }

    // -- get_document_hashes_many ------------------------------------------------

    /// Same shape as [`assert_get_many_returns_every_requested_key`], for the
    /// `documents`-table equivalent.
    async fn assert_get_document_hashes_many_returns_every_requested_key(n: usize) {
        let (db, _dir) = test_db().await;
        let entries: Vec<(String, String)> = (0..n)
            .map(|i| (format!("doc-{i:05}.md"), format!("hash-{i}")))
            .collect();
        bulk_insert_documents(&db, &entries).await;

        let paths: Vec<String> = entries.iter().map(|(p, _)| p.clone()).collect();
        let result = db.get_document_hashes_many(&paths).await.unwrap();

        assert_eq!(result.len(), n, "every requested key must come back, n={n}");
        for (path, hash) in &entries {
            assert_eq!(
                result.get(path).map(String::as_str),
                Some(hash.as_str()),
                "wrong hash attached to '{path}' at n={n}"
            );
        }
    }

    #[tokio::test]
    async fn get_document_hashes_many_at_499_below_the_chunk_boundary() {
        assert_get_document_hashes_many_returns_every_requested_key(499).await;
    }

    #[tokio::test]
    async fn get_document_hashes_many_at_exactly_500_the_chunk_boundary() {
        assert_get_document_hashes_many_returns_every_requested_key(500).await;
    }

    #[tokio::test]
    async fn get_document_hashes_many_at_501_one_past_the_chunk_boundary() {
        assert_get_document_hashes_many_returns_every_requested_key(501).await;
    }

    #[tokio::test]
    async fn get_document_hashes_many_at_1001_spanning_three_chunks() {
        assert_get_document_hashes_many_returns_every_requested_key(1001).await;
    }

    #[tokio::test]
    async fn get_document_hashes_many_distinguishes_absent_keys_from_present_ones() {
        let (db, _dir) = test_db().await;
        let entries: Vec<(String, String)> = (0..5)
            .map(|i| (format!("present-{i}.md"), format!("hash-{i}")))
            .collect();
        bulk_insert_documents(&db, &entries).await;

        let mut requested: Vec<String> = entries.iter().map(|(p, _)| p.clone()).collect();
        requested.push("missing-1.md".into());

        let result = db.get_document_hashes_many(&requested).await.unwrap();

        assert_eq!(result.len(), 5);
        assert!(!result.contains_key("missing-1.md"));
    }

    #[tokio::test]
    async fn get_document_hashes_many_empty_input_returns_empty_map() {
        let (db, _dir) = test_db().await;
        assert!(db.get_document_hashes_many(&[]).await.unwrap().is_empty());
    }

    // -- document_links --------------------------------------------------------

    #[tokio::test]
    async fn replace_links_round_trip() {
        let (db, _dir) = test_db().await;
        db.replace_links(
            "a.md",
            "markdown",
            &[("b.md".to_string(), None), ("c.md".to_string(), None)],
        )
        .await
        .unwrap();

        let mut links = db.all_links().await.unwrap();
        links.sort_by(|a, b| (&a.0, &a.1, &a.2).cmp(&(&b.0, &b.1, &b.2)));
        assert_eq!(
            links,
            vec![
                (
                    "a.md".to_string(),
                    "b.md".to_string(),
                    "markdown".to_string(),
                    None
                ),
                (
                    "a.md".to_string(),
                    "c.md".to_string(),
                    "markdown".to_string(),
                    None
                ),
            ]
        );
    }

    #[tokio::test]
    async fn replace_links_carries_score() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "semantic", &[("b.md".to_string(), Some(0.87))])
            .await
            .unwrap();

        let links = db.all_links().await.unwrap();
        assert_eq!(
            links,
            vec![(
                "a.md".to_string(),
                "b.md".to_string(),
                "semantic".to_string(),
                Some(0.87)
            )]
        );
    }

    #[tokio::test]
    async fn replace_links_overwrites_prior_call_for_same_source_and_kind() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("old.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("a.md", "markdown", &[("new.md".to_string(), None)])
            .await
            .unwrap();

        let links = db.all_links().await.unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].1, "new.md");
    }

    #[tokio::test]
    async fn replace_links_is_scoped_to_kind() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("md-target.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links(
            "a.md",
            "semantic",
            &[("sem-target.md".to_string(), Some(0.9))],
        )
        .await
        .unwrap();

        // Replacing the semantic edges must not disturb the markdown edges for the
        // same source.
        db.replace_links(
            "a.md",
            "semantic",
            &[("sem-target-2.md".to_string(), Some(0.5))],
        )
        .await
        .unwrap();

        let mut links = db.all_links().await.unwrap();
        links.sort_by(|a, b| (&a.0, &a.1, &a.2).cmp(&(&b.0, &b.1, &b.2)));
        assert_eq!(
            links,
            vec![
                (
                    "a.md".to_string(),
                    "md-target.md".to_string(),
                    "markdown".to_string(),
                    None
                ),
                (
                    "a.md".to_string(),
                    "sem-target-2.md".to_string(),
                    "semantic".to_string(),
                    Some(0.5)
                ),
            ]
        );
    }

    #[tokio::test]
    async fn replace_links_empty_targets_clears_existing() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("b.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("a.md", "markdown", &[]).await.unwrap();

        assert!(db.all_links().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn links_targeting_returns_distinct_sources_ordered() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("target.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("c.md", "markdown", &[("target.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("b.md", "markdown", &[("target.md".to_string(), None)])
            .await
            .unwrap();
        // A source unrelated to `target.md` must not show up.
        db.replace_links("d.md", "markdown", &[("other.md".to_string(), None)])
            .await
            .unwrap();

        let sources = db.links_targeting("target.md", "markdown").await.unwrap();
        assert_eq!(sources, vec!["a.md", "b.md", "c.md"]);
    }

    #[tokio::test]
    async fn links_targeting_filters_by_kind() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("target.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("b.md", "semantic", &[("target.md".to_string(), Some(0.9))])
            .await
            .unwrap();

        let markdown_sources = db.links_targeting("target.md", "markdown").await.unwrap();
        assert_eq!(
            markdown_sources,
            vec!["a.md"],
            "a semantic-kind row must not come back when asking for markdown"
        );

        let semantic_sources = db.links_targeting("target.md", "semantic").await.unwrap();
        assert_eq!(semantic_sources, vec!["b.md"]);
    }

    #[tokio::test]
    async fn links_targeting_no_results_returns_empty() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("other.md".to_string(), None)])
            .await
            .unwrap();

        let sources = db
            .links_targeting("nonexistent.md", "markdown")
            .await
            .unwrap();
        assert!(sources.is_empty());
    }

    // -- links_targeting_many ----------------------------------------------------

    /// Insert `(source_path, target_path, kind)` rows into `document_links` via a
    /// handful of multi-row `INSERT`s, so the chunk-boundary test below (500+ distinct
    /// targets) stays fast — same rationale as `bulk_insert_indexed_files`/
    /// `bulk_insert_documents` above. `score` is always bound `NULL`: none of these
    /// tests care about it.
    async fn bulk_insert_document_links(db: &StateDb, rows: &[(String, String, String)]) {
        const ROWS_PER_STATEMENT: usize = 100;
        for chunk in rows.chunks(ROWS_PER_STATEMENT) {
            let mut builder = QueryBuilder::<Sqlite>::new(
                "INSERT INTO document_links (source_path, target_path, kind, score) ",
            );
            builder.push_values(chunk, |mut b, (source_path, target_path, kind)| {
                b.push_bind(source_path)
                    .push_bind(target_path)
                    .push_bind(kind)
                    .push_bind(Option::<f64>::None);
            });
            builder.build().execute(db.pool_for_test()).await.unwrap();
        }
    }

    #[tokio::test]
    async fn links_targeting_many_returns_matches_grouped_by_target() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("target1.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("c.md", "markdown", &[("target1.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("b.md", "markdown", &[("target1.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("d.md", "markdown", &[("target2.md".to_string(), None)])
            .await
            .unwrap();
        // A source unrelated to either queried target must not show up.
        db.replace_links("e.md", "markdown", &[("other.md".to_string(), None)])
            .await
            .unwrap();

        let by_target = db
            .links_targeting_many(
                &["target1.md".to_string(), "target2.md".to_string()],
                "markdown",
            )
            .await
            .unwrap();

        assert_eq!(
            by_target.get("target1.md"),
            Some(&vec![
                "a.md".to_string(),
                "b.md".to_string(),
                "c.md".to_string()
            ]),
            "multiple sources targeting the same path must all come back, sorted, \
             matching links_targeting's own ordering"
        );
        assert_eq!(by_target.get("target2.md"), Some(&vec!["d.md".to_string()]));
        assert_eq!(
            by_target.len(),
            2,
            "a target with no referencing sources must have no key at all"
        );
    }

    #[tokio::test]
    async fn links_targeting_many_filters_by_kind() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("target.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("b.md", "semantic", &[("target.md".to_string(), Some(0.9))])
            .await
            .unwrap();

        let markdown = db
            .links_targeting_many(&["target.md".to_string()], "markdown")
            .await
            .unwrap();
        assert_eq!(
            markdown.get("target.md"),
            Some(&vec!["a.md".to_string()]),
            "a semantic-kind row must not come back when asking for markdown"
        );

        let semantic = db
            .links_targeting_many(&["target.md".to_string()], "semantic")
            .await
            .unwrap();
        assert_eq!(semantic.get("target.md"), Some(&vec!["b.md".to_string()]));
    }

    #[tokio::test]
    async fn links_targeting_many_empty_input_returns_empty_map() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("target.md".to_string(), None)])
            .await
            .unwrap();

        let result = db.links_targeting_many(&[], "markdown").await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn links_targeting_many_spans_a_500_param_chunk_boundary() {
        // 501 distinct target paths: one full 500-param chunk plus one row spilling
        // into a second chunk. Each has exactly one source pointing at it, so an
        // off-by-one at the chunk seam would show up as a missing target rather than
        // silently merging two targets' sources together.
        let (db, _dir) = test_db().await;
        let targets: Vec<String> = (0..501).map(|i| format!("target-{i:05}.md")).collect();
        let rows: Vec<(String, String, String)> = targets
            .iter()
            .map(|target| {
                (
                    format!("source-for-{target}"),
                    target.clone(),
                    "markdown".to_string(),
                )
            })
            .collect();
        bulk_insert_document_links(&db, &rows).await;

        let by_target = db.links_targeting_many(&targets, "markdown").await.unwrap();

        assert_eq!(
            by_target.len(),
            501,
            "every target across the chunk boundary must come back"
        );
        for target in &targets {
            assert_eq!(
                by_target.get(target),
                Some(&vec![format!("source-for-{target}")]),
                "wrong (or missing) source attached to '{target}' at the chunk boundary"
            );
        }
    }

    #[tokio::test]
    async fn delete_links_for_removes_outgoing_edges_only() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("b.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("c.md", "markdown", &[("a.md".to_string(), None)])
            .await
            .unwrap();

        db.delete_links_for("a.md").await.unwrap();

        let links = db.all_links().await.unwrap();
        assert_eq!(
            links,
            vec![(
                "c.md".to_string(),
                "a.md".to_string(),
                "markdown".to_string(),
                None
            )],
            "incoming edges (a.md as target) must survive; only a.md's outgoing edges are removed"
        );
    }

    #[tokio::test]
    async fn delete_document_also_clears_outgoing_links() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("a.md", &recipe_frontmatter(), 1700, "h1", 1)
            .await
            .unwrap();
        db.replace_links("a.md", "markdown", &[("b.md".to_string(), None)])
            .await
            .unwrap();

        db.delete_document("a.md").await.unwrap();

        assert!(db.all_links().await.unwrap().is_empty());
    }

    // -- links_out / links_in (#157) ---------------------------------------------

    #[tokio::test]
    async fn links_out_returns_both_kinds_and_flags_a_dangling_target() {
        let (db, _dir) = test_db().await;
        // "b.md" is a real indexed document; "missing.md" is a link target that
        // was never indexed — exactly the dangling case `document_links`'s doc
        // comment describes.
        db.upsert_document_metadata("b.md", &recipe_frontmatter(), 1700, "h1", 1)
            .await
            .unwrap();
        db.replace_links("a.md", "markdown", &[("b.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links(
            "a.md",
            "semantic",
            &[("missing.md".to_string(), Some(0.75))],
        )
        .await
        .unwrap();

        let page = db.links_out("a.md", 10).await.unwrap();

        assert_eq!(page.total, 2);
        assert!(!page.has_more());
        let mut links = page.links;
        links.sort_by(|a, b| a.target_path.cmp(&b.target_path));
        assert_eq!(
            links,
            vec![
                OutboundLink {
                    target_path: "b.md".to_string(),
                    kind: "markdown".to_string(),
                    score: None,
                    exists: true,
                },
                OutboundLink {
                    target_path: "missing.md".to_string(),
                    kind: "semantic".to_string(),
                    score: Some(0.75),
                    exists: false,
                },
            ],
            "a dangling target must still be returned, just flagged exists: false"
        );
    }

    #[tokio::test]
    async fn links_out_on_a_source_with_no_edges_is_empty() {
        let (db, _dir) = test_db().await;
        let page = db.links_out("lonely.md", 10).await.unwrap();
        assert_eq!(page.total, 0);
        assert!(page.links.is_empty());
        assert!(!page.has_more());
    }

    #[tokio::test]
    async fn links_out_caps_at_limit_and_reports_the_true_total() {
        let (db, _dir) = test_db().await;
        let targets: Vec<(String, Option<f64>)> =
            (0..5).map(|i| (format!("target-{i}.md"), None)).collect();
        db.replace_links("hub.md", "markdown", &targets)
            .await
            .unwrap();

        let page = db.links_out("hub.md", 2).await.unwrap();

        assert_eq!(page.links.len(), 2, "page must be capped at `limit`");
        assert_eq!(
            page.total, 5,
            "total must report the true edge count, not the page size"
        );
        assert!(page.has_more());
    }

    #[tokio::test]
    async fn links_in_returns_sources_across_kinds_ordered() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("target.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("b.md", "semantic", &[("target.md".to_string(), Some(0.6))])
            .await
            .unwrap();

        let page = db.links_in("target.md", 10).await.unwrap();

        assert_eq!(page.total, 2);
        assert!(!page.has_more());
        assert_eq!(
            page.links,
            vec![
                InboundLink {
                    source_path: "a.md".to_string(),
                    kind: "markdown".to_string(),
                    score: None,
                },
                InboundLink {
                    source_path: "b.md".to_string(),
                    kind: "semantic".to_string(),
                    score: Some(0.6),
                },
            ],
            "ORDER BY kind, source_path: markdown before semantic"
        );
    }

    #[tokio::test]
    async fn links_in_caps_at_limit_and_reports_the_true_total() {
        let (db, _dir) = test_db().await;
        for i in 0..5 {
            db.replace_links(
                &format!("source-{i}.md"),
                "markdown",
                &[("hub.md".to_string(), None)],
            )
            .await
            .unwrap();
        }

        let page = db.links_in("hub.md", 2).await.unwrap();

        assert_eq!(page.links.len(), 2, "page must be capped at `limit`");
        assert_eq!(
            page.total, 5,
            "total must report the true edge count, not the page size"
        );
        assert!(page.has_more());
    }

    #[tokio::test]
    async fn links_in_on_a_target_with_no_edges_is_empty() {
        let (db, _dir) = test_db().await;
        let page = db.links_in("nobody-links-here.md", 10).await.unwrap();
        assert_eq!(page.total, 0);
        assert!(page.links.is_empty());
        assert!(!page.has_more());
    }

    // -- broken_markdown_links (#158) --------------------------------------------

    #[tokio::test]
    async fn broken_markdown_links_reports_a_dangling_markdown_target() {
        let (db, _dir) = test_db().await;
        db.replace_links("a.md", "markdown", &[("missing.md".to_string(), None)])
            .await
            .unwrap();

        let dangling = db.broken_markdown_links().await.unwrap();

        assert_eq!(
            dangling,
            vec![("a.md".to_string(), "missing.md".to_string())]
        );
    }

    #[tokio::test]
    async fn broken_markdown_links_excludes_targets_that_are_indexed() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("b.md", &recipe_frontmatter(), 1700, "h1", 1)
            .await
            .unwrap();
        db.replace_links("a.md", "markdown", &[("b.md".to_string(), None)])
            .await
            .unwrap();

        let dangling = db.broken_markdown_links().await.unwrap();

        assert!(
            dangling.is_empty(),
            "a target with a documents row is not broken: {dangling:?}"
        );
    }

    #[tokio::test]
    async fn broken_markdown_links_ignores_dangling_semantic_edges() {
        let (db, _dir) = test_db().await;
        // A semantic (kNN) edge to a target with no documents row must never be
        // reported — it is a machine-inferred neighbor, not an author-written link
        // that could be "fixed", and #158 scopes the report to `kind = 'markdown'`
        // for exactly this reason.
        db.replace_links("a.md", "semantic", &[("missing.md".to_string(), Some(0.9))])
            .await
            .unwrap();

        let dangling = db.broken_markdown_links().await.unwrap();

        assert!(
            dangling.is_empty(),
            "a dangling semantic edge must not be reported: {dangling:?}"
        );
    }

    #[tokio::test]
    async fn broken_markdown_links_on_a_clean_kb_is_empty() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("b.md", &recipe_frontmatter(), 1700, "h1", 1)
            .await
            .unwrap();
        db.replace_links("a.md", "markdown", &[("b.md".to_string(), None)])
            .await
            .unwrap();
        db.replace_links("a.md", "semantic", &[("b.md".to_string(), Some(0.5))])
            .await
            .unwrap();

        let dangling = db.broken_markdown_links().await.unwrap();

        assert!(dangling.is_empty());
    }

    #[tokio::test]
    async fn all_document_summaries_returns_every_row_unpaged() {
        let (db, _dir) = test_db().await;
        let entries: Vec<(String, String)> = (0..150)
            .map(|i| (format!("doc-{i:03}.md"), format!("hash-{i}")))
            .collect();
        bulk_insert_documents(&db, &entries).await;

        let summaries = db.all_document_summaries().await.unwrap();

        assert_eq!(
            summaries.len(),
            150,
            "all_document_summaries must not apply a hidden limit"
        );
        // ORDER BY file_path, ascending.
        assert_eq!(summaries[0].file_path, "doc-000.md");
        assert_eq!(summaries[149].file_path, "doc-149.md");
    }

    #[tokio::test]
    async fn all_document_summaries_degrades_malformed_frontmatter() {
        let (db, _dir) = test_db().await;
        sqlx::query(
            "INSERT INTO documents
                (file_path, title, description, frontmatter, mtime, content_hash, chunk_count, indexed_at)
             VALUES ('bad.md', NULL, NULL, 'not valid json{{', 0, 'h', 0, datetime('now'))",
        )
        .execute(db.pool_for_test())
        .await
        .unwrap();

        let summaries = db.all_document_summaries().await.unwrap();

        assert_eq!(summaries.len(), 1);
        assert_eq!(
            summaries[0].frontmatter,
            Value::Object(serde_json::Map::new())
        );
    }

    #[tokio::test]
    async fn all_document_summaries_maps_fields_from_upsert() {
        let (db, _dir) = test_db().await;
        db.upsert_document_metadata("r.md", &recipe_frontmatter(), 1700, "h1", 2)
            .await
            .unwrap();

        let summaries = db.all_document_summaries().await.unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].file_path, "r.md");
        assert_eq!(summaries[0].mtime, 1700);
        assert!(summaries[0].frontmatter.is_object());
    }
}
