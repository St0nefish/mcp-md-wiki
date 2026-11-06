//! Corpus-wide health checks (#354): dead links, orphans, stale documents, oversize
//! documents and a tag audit, over the last indexed state.
//!
//! Every check is read-only. It reads the state DB (`documents`, `indexed_files`,
//! `document_links`), the resolved schema tree (tag enums) and, for the oversize
//! check only, the size-qualified files themselves; it never touches Qdrant or the
//! embedding API, and never writes. Each check is a pure function over pre-loaded
//! inputs (with `now` injected), so its output is deterministic and unit-testable;
//! [`run_health_checks`] is the one place that loads inputs and assembles a
//! [`HealthReport`].
//!
//! # Dead links
//!
//! A dead link here is exactly what `get_document`'s per-document `broken_links`
//! lists — an author-written (`kind = 'markdown'`) edge whose target has no
//! `documents` row — corpus-wide instead of per document, with the same
//! false-positive filter `validate` applies (`validate::broken_links_report_limited`):
//! a target that exists on disk but is excluded from indexing is not reported. The
//! one deliberate difference from the per-document list is that filter, which only
//! ever removes entries. Like both of those, it reads the state DB as of the last
//! successful index run, and only indexed sources have link rows, so excluded
//! source files are never reported.
//!
//! Link line numbers are not stored in `document_links`, so [`DeadLink::line`] is
//! always omitted today; the field exists so adding it later is not a wire change.
//! The standalone [`DeadLinkReport`] is what `GET /status/links` serves; the health
//! report carries the same findings flattened per link.
//!
//! # Severities
//!
//! Fixed, not configurable: dead links and tags outside a declared enum are `error`;
//! orphans, stale and oversize documents are `warning`; singleton tags are `info`.
//! The tag audit therefore yields two reports, `tags-outside-enum` and
//! `tags-singleton`, both selected by the `tags` check.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use crate::config::{LintConfig, ResolvedConfig};
use crate::document_fields::epoch_seconds;
use crate::ingest::{KbEntry, inspect_kb_path, parse_globs};
use crate::schema::SchemaCache;
use crate::state::{DocumentSummary, StateDb};
use crate::validate;

/// Default cap on the number of findings a report lists per check.
pub const DEFAULT_DEAD_LINK_LIMIT: usize = validate::MAX_BROKEN_LINKS_SHOWN;

/// Largest cap a caller (the `/status/links` and `/status/lint` query strings, the
/// `lint --limit` flag) may ask for.
pub const MAX_DEAD_LINK_LIMIT: usize = 5000;

const SECONDS_PER_DAY: i64 = 86_400;

// ---------------------------------------------------------------------------
// Dead links (also the standalone `/status/links` report)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct DeadLink {
    pub target: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct DeadLinkDocument {
    pub path: String,
    pub broken: Vec<DeadLink>,
}

/// Every broken internal link, grouped by source document.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct DeadLinkReport {
    /// Total broken links in the corpus, before the cap.
    pub count: usize,
    pub documents: Vec<DeadLinkDocument>,
    /// True when `documents` lists fewer links than `count`.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

/// Build the report from the state DB, listing at most `limit` links.
pub async fn dead_links(state: &StateDb, data_path: &Path, limit: usize) -> Result<DeadLinkReport> {
    let pairs = state.broken_markdown_links().await?;
    Ok(from_pairs(pairs, data_path, limit))
}

fn from_pairs(pairs: Vec<(String, String)>, data_path: &Path, limit: usize) -> DeadLinkReport {
    let r = validate::broken_links_report_limited(pairs, data_path, limit);
    DeadLinkReport {
        count: r.total,
        truncated: r.truncated,
        documents: r
            .by_source
            .into_iter()
            .map(|g| DeadLinkDocument {
                path: g.source_path,
                broken: g
                    .broken_targets
                    .into_iter()
                    .map(|target| DeadLink { target, line: None })
                    .collect(),
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// The health report envelope
// ---------------------------------------------------------------------------

/// One kind of check. The clap value names (`dead-links`, `orphans`, ...) are also
/// the names `GET /status/lint?check=` accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, clap::ValueEnum)]
pub enum Check {
    /// Broken internal markdown links across the corpus
    DeadLinks,
    /// Documents no other document links to
    Orphans,
    /// Documents not updated within the configured thresholds
    Stale,
    /// Documents past the configured line or byte limit
    Oversize,
    /// Singleton tags and tags outside a declared enum
    Tags,
}

impl Check {
    /// Every check, in the order a report lists them.
    pub const ALL: [Check; 5] = [
        Check::DeadLinks,
        Check::Orphans,
        Check::Stale,
        Check::Oversize,
        Check::Tags,
    ];

    /// The kebab-case name used on the command line and in `?check=`.
    pub fn name(self) -> &'static str {
        match self {
            Check::DeadLinks => "dead-links",
            Check::Orphans => "orphans",
            Check::Stale => "stale",
            Check::Oversize => "oversize",
            Check::Tags => "tags",
        }
    }

    pub fn parse(name: &str) -> Option<Check> {
        Check::ALL.into_iter().find(|c| c.name() == name)
    }

    /// Whether `lint.<check>` enables this check.
    pub fn enabled_in(self, lint: &LintConfig) -> bool {
        match self {
            Check::DeadLinks => lint.dead_links,
            Check::Orphans => lint.orphans,
            Check::Stale => lint.stale,
            Check::Oversize => lint.oversize,
            Check::Tags => lint.tags,
        }
    }
}

/// The checks `lint.*` enables, in report order.
pub fn enabled_checks(lint: &LintConfig) -> Vec<Check> {
    Check::ALL
        .into_iter()
        .filter(|c| c.enabled_in(lint))
        .collect()
}

/// Parse a `?check=` query value: names separated by commas, the parameter possibly
/// repeated. Every unknown name is reported at once.
pub fn parse_check_names<'a>(
    names: impl IntoIterator<Item = &'a str>,
) -> Result<Vec<Check>, String> {
    let mut checks = BTreeSet::new();
    let mut unknown = Vec::new();
    for raw in names {
        for name in raw.split(',').map(str::trim).filter(|n| !n.is_empty()) {
            match Check::parse(name) {
                Some(c) => {
                    checks.insert(c);
                }
                None => unknown.push(name.to_string()),
            }
        }
    }
    if !unknown.is_empty() {
        return Err(format!(
            "unknown check(s): {} — valid checks are {}",
            unknown.join(", "),
            Check::ALL.map(Check::name).join(", ")
        ));
    }
    Ok(checks.into_iter().collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
    Info,
}

impl Severity {
    fn label(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Info => "info",
        }
    }
}

/// One thing a check found.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
pub struct Finding {
    /// The KB-relative path of the document the finding is about.
    pub path: String,
    pub detail: String,
}

impl Finding {
    fn new(path: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            detail: detail.into(),
        }
    }
}

/// The result of one check.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CheckReport {
    /// `dead-links`, `orphans`, `stale`, `oversize`, `tags-outside-enum` or
    /// `tags-singleton`.
    pub check: &'static str,
    pub severity: Severity,
    /// Total findings, before the cap.
    pub count: usize,
    /// At most `limit` findings, sorted by (path, detail). Omitted when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub findings: Vec<Finding>,
    /// True when `findings` lists fewer than `count`.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
    /// Why the check was skipped or is partial, when it was.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl CheckReport {
    /// Sort `findings` by (path, detail), drop exact duplicates, and cap them.
    fn from_findings(
        check: &'static str,
        severity: Severity,
        mut findings: Vec<Finding>,
        limit: usize,
    ) -> Self {
        findings.sort();
        findings.dedup();
        let count = findings.len();
        findings.truncate(limit);
        Self {
            check,
            severity,
            count,
            truncated: count > findings.len(),
            findings,
            note: None,
        }
    }

    fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    fn skipped(check: &'static str, severity: Severity, note: impl Into<String>) -> Self {
        Self::from_findings(check, severity, Vec::new(), 0).with_note(note)
    }
}

/// What `lint` and `GET /status/lint` return.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct HealthReport {
    pub checks: Vec<CheckReport>,
}

/// The dead-link report as a [`CheckReport`]: one finding per broken link.
fn dead_link_check(r: &DeadLinkReport) -> CheckReport {
    let findings = r
        .documents
        .iter()
        .flat_map(|d| {
            d.broken.iter().map(|l| {
                Finding::new(
                    d.path.clone(),
                    match l.line {
                        Some(n) => format!("broken link to {} (line {n})", l.target),
                        None => format!("broken link to {}", l.target),
                    },
                )
            })
        })
        .collect();
    let mut report =
        CheckReport::from_findings("dead-links", Severity::Error, findings, usize::MAX);
    // The cap was applied upstream, before grouping; the true total and the
    // truncation flag are the report's own.
    report.count = r.count;
    report.truncated = r.truncated;
    report
}

// ---------------------------------------------------------------------------
// Orphans
// ---------------------------------------------------------------------------

/// Documents (of `paths`) that no other document links to through an author-written
/// link (`links` holds `(source, target)` pairs of `kind = 'markdown'`). A link from
/// a document to itself does not count; a link from any other document does, even
/// one that is an orphan itself. `entry_points` are paths or globs exempt from the
/// check.
pub fn orphan_findings(
    paths: &[String],
    links: &[(String, String)],
    entry_points: &[String],
) -> Vec<Finding> {
    let linked: BTreeSet<&str> = links
        .iter()
        .filter(|(source, target)| source != target)
        .map(|(_, target)| target.as_str())
        .collect();
    let (builder, valid) = parse_globs(entry_points);
    let exempt = if valid == 0 {
        None
    } else {
        builder.build().ok()
    };
    paths
        .iter()
        .filter(|p| !linked.contains(p.as_str()))
        .filter(|p| exempt.as_ref().is_none_or(|set| !set.is_match(p.as_str())))
        .map(|p| Finding::new(p.clone(), "no other document links to it"))
        .collect()
}

// ---------------------------------------------------------------------------
// Stale
// ---------------------------------------------------------------------------

/// The value at a dot-separated `path` in a frontmatter object.
fn frontmatter_value<'a>(frontmatter: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.')
        .try_fold(frontmatter, |value, key| value.get(key))
}

/// Documents not updated within the configured threshold as of `now` (epoch
/// seconds). A document's last update is its `lint.stale_field` frontmatter value
/// when that parses as a date or timestamp, else `documents.mtime` (git commit time
/// on a git-backed KB, file mtime otherwise; 0 means unknown and the document is
/// skipped). A `status: draft` document uses the shorter draft threshold. A date in
/// the future is never stale. Each finding is paired with whether the document was
/// judged by its modification time rather than the frontmatter field.
fn stale_entries(docs: &[DocumentSummary], now: i64, lint: &LintConfig) -> Vec<(Finding, bool)> {
    docs.iter()
        .filter_map(|doc| {
            let from_field = frontmatter_value(&doc.frontmatter, &lint.stale_field)
                .and_then(Value::as_str)
                .and_then(epoch_seconds)
                .map(|secs| secs.floor() as i64);
            let (updated, source) = match from_field {
                Some(secs) => (secs, format!("frontmatter '{}'", lint.stale_field)),
                None if doc.mtime > 0 => (doc.mtime, "file modification time".to_string()),
                None => return None,
            };
            let draft = doc
                .frontmatter
                .get("status")
                .and_then(Value::as_str)
                .is_some_and(|s| s.eq_ignore_ascii_case("draft"));
            let limit_days = if draft {
                lint.draft_stale_after_days
            } else {
                lint.stale_after_days
            };
            let age_days = (now - updated).div_euclid(SECONDS_PER_DAY);
            if age_days < 0 || (age_days as u64) < limit_days {
                return None;
            }
            let date = chrono::DateTime::from_timestamp(updated, 0)
                .map(|t| t.date_naive().to_string())
                .unwrap_or_else(|| updated.to_string());
            let finding = Finding::new(
                doc.file_path.clone(),
                format!(
                    "not updated for {age_days} days (last update {date}, from {source}); \
                     limit {limit_days} days{}",
                    if draft { " for a draft" } else { "" }
                ),
            );
            Some((finding, from_field.is_none()))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Oversize
// ---------------------------------------------------------------------------

/// One document's size. `lines` is `None` when it was not counted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SizedDoc {
    pub path: String,
    pub bytes: u64,
    pub lines: Option<u64>,
}

/// Whether a file of `bytes` can exceed `max_lines` lines. A file of N lines has at
/// least N bytes (each line is a newline or a final character), so a file no larger
/// than `max_lines` bytes never needs reading.
pub fn may_exceed_max_lines(bytes: u64, max_lines: u64) -> bool {
    bytes > max_lines
}

/// Documents with more than `max_lines` lines or more than `max_bytes` bytes.
pub fn oversize_findings(docs: &[SizedDoc], max_lines: u64, max_bytes: u64) -> Vec<Finding> {
    docs.iter()
        .filter_map(|d| {
            let mut reasons = Vec::new();
            if let Some(lines) = d.lines.filter(|l| *l > max_lines) {
                reasons.push(format!("{lines} lines (limit {max_lines})"));
            }
            if d.bytes > max_bytes {
                reasons.push(format!("{} bytes (limit {max_bytes})", d.bytes));
            }
            (!reasons.is_empty()).then(|| Finding::new(d.path.clone(), reasons.join("; ")))
        })
        .collect()
}

/// Count the lines of the regular file at `rel` under `root`: newline-terminated
/// lines plus a final unterminated one. A symlink, a `.git` path, a directory or an
/// unreadable file yields `None` — never followed, never an error.
fn count_lines(root: &Path, rel: &str) -> Option<u64> {
    use std::io::Read;
    if !matches!(inspect_kb_path(root, Path::new(rel)), Ok(KbEntry::File)) {
        return None;
    }
    let mut file = std::fs::File::open(root.join(rel)).ok()?;
    let mut buf = [0u8; 64 * 1024];
    let mut lines = 0u64;
    let mut last = b'\n';
    loop {
        let n = file.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        lines += buf[..n].iter().filter(|b| **b == b'\n').count() as u64;
        last = buf[n - 1];
    }
    Some(if last == b'\n' { lines } else { lines + 1 })
}

// ---------------------------------------------------------------------------
// Tags
// ---------------------------------------------------------------------------

/// The distinct string tags a document carries in `field` (a list of strings, or a
/// single string). Other value types are not tags.
fn document_tags<'a>(frontmatter: &'a Value, field: &str) -> BTreeSet<&'a str> {
    match frontmatter_value(frontmatter, field) {
        Some(Value::Array(items)) => items.iter().filter_map(Value::as_str).collect(),
        Some(Value::String(s)) => BTreeSet::from([s.as_str()]),
        _ => BTreeSet::new(),
    }
}

/// Tags used by exactly one document: a likely typo or a one-off worth folding into
/// an existing tag. The finding names the one document that carries it.
pub fn singleton_tag_findings(docs: &[DocumentSummary], field: &str) -> Vec<Finding> {
    let mut users: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for doc in docs {
        for tag in document_tags(&doc.frontmatter, field) {
            users.entry(tag).or_default().push(doc.file_path.as_str());
        }
    }
    users
        .into_iter()
        .filter_map(|(tag, paths)| match paths.as_slice() {
            [only] => Some(Finding::new(
                *only,
                format!("tag '{tag}' is used by no other document"),
            )),
            _ => None,
        })
        .collect()
}

/// Tags outside the enum the governing schema declares for `field`. `allowed` maps a
/// document path to its permitted values, or `None` when the schema declares no
/// closed set there (nothing to audit).
pub fn out_of_enum_tag_findings(
    docs: &[DocumentSummary],
    field: &str,
    allowed: &dyn Fn(&str) -> Option<Vec<String>>,
) -> Vec<Finding> {
    docs.iter()
        .flat_map(|doc| {
            let permitted = allowed(&doc.file_path);
            document_tags(&doc.frontmatter, field)
                .into_iter()
                .filter(move |tag| {
                    permitted
                        .as_ref()
                        .is_some_and(|set| !set.iter().any(|v| v == tag))
                })
                .map(|tag| {
                    Finding::new(
                        doc.file_path.clone(),
                        format!("tag '{tag}' is not one of the permitted values"),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The permitted values the schema declares for `field` at `rel_path`: a list's
/// `items` values, else the field's own `values`.
fn permitted_values(schemas: &SchemaCache, rel_path: &str, field: &str) -> Option<Vec<String>> {
    let schema = schemas.resolve_for(Path::new(rel_path));
    let def = schema.fields.lookup(field)?;
    def.items
        .as_ref()
        .and_then(|items| items.values.clone())
        .or_else(|| def.values.clone())
}

// ---------------------------------------------------------------------------
// The orchestrator
// ---------------------------------------------------------------------------

/// Everything [`run_health_checks`] reads besides the state DB. Borrowed from a
/// config snapshot so each run sees the live `lint.*` values.
pub struct HealthContext<'a> {
    pub state: &'a StateDb,
    pub data_path: &'a Path,
    pub lint: &'a LintConfig,
    /// `search.section_max_bytes`, the fallback for an unset `lint.max_bytes`.
    pub section_max_bytes: u64,
    /// The schema tree, for the tag enum check. `None` skips that check, with a note.
    pub schemas: Option<&'a SchemaCache>,
    /// Epoch seconds; injected so a report is reproducible.
    pub now: i64,
}

impl<'a> HealthContext<'a> {
    pub fn new(
        state: &'a StateDb,
        cfg: &'a ResolvedConfig,
        schemas: Option<&'a SchemaCache>,
        now: i64,
    ) -> Self {
        Self {
            state,
            data_path: Path::new(cfg.data_path()),
            lint: &cfg.lint,
            section_max_bytes: cfg.search.section_max_bytes as u64,
            schemas,
            now,
        }
    }
}

/// Seconds since the Unix epoch now.
pub fn now_epoch_seconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Run `checks` (each at most once, reported in [`Check::ALL`] order), listing at
/// most `limit` findings per report.
pub async fn run_health_checks(
    ctx: &HealthContext<'_>,
    checks: &[Check],
    limit: usize,
) -> Result<HealthReport> {
    let wanted = |c: Check| checks.contains(&c);
    let limit = limit.clamp(1, MAX_DEAD_LINK_LIMIT);

    // Loaded once for every check that reads document metadata.
    let docs = if wanted(Check::Orphans) || wanted(Check::Stale) || wanted(Check::Tags) {
        ctx.state.all_document_summaries().await?
    } else {
        Vec::new()
    };

    let mut reports = Vec::new();
    for check in Check::ALL.into_iter().filter(|c| wanted(*c)) {
        match check {
            Check::DeadLinks => {
                let report = dead_links(ctx.state, ctx.data_path, limit).await?;
                reports.push(dead_link_check(&report));
            }
            Check::Orphans => {
                let paths: Vec<String> = docs.iter().map(|d| d.file_path.clone()).collect();
                let links = ctx.state.markdown_link_pairs().await?;
                let findings = orphan_findings(&paths, &links, &ctx.lint.orphan_entry_points);
                reports.push(CheckReport::from_findings(
                    "orphans",
                    Severity::Warning,
                    findings,
                    limit,
                ));
            }
            Check::Stale => {
                let entries = stale_entries(&docs, ctx.now, ctx.lint);
                let by_mtime = entries.iter().any(|(_, by_mtime)| *by_mtime);
                let findings = entries.into_iter().map(|(f, _)| f).collect();
                let report =
                    CheckReport::from_findings("stale", Severity::Warning, findings, limit);
                reports.push(if by_mtime {
                    report.with_note(format!(
                        "some documents have no date in frontmatter '{}' and were judged by \
                         modification time (git commit time on a git-backed KB, which is more \
                         reliable than a file mtime that resets on clone)",
                        ctx.lint.stale_field
                    ))
                } else {
                    report
                });
            }
            Check::Oversize => {
                let max_lines = ctx.lint.max_lines;
                let max_bytes = ctx.lint.max_bytes.unwrap_or(ctx.section_max_bytes);
                let sizes = ctx.state.indexed_document_sizes().await?;
                let root = ctx.data_path.to_path_buf();
                let sized = tokio::task::spawn_blocking(move || {
                    sizes
                        .into_iter()
                        .map(|(path, size)| {
                            let bytes = size.max(0) as u64;
                            let lines = may_exceed_max_lines(bytes, max_lines)
                                .then(|| count_lines(&root, &path))
                                .flatten();
                            SizedDoc { path, bytes, lines }
                        })
                        .collect::<Vec<_>>()
                })
                .await?;
                reports.push(CheckReport::from_findings(
                    "oversize",
                    Severity::Warning,
                    oversize_findings(&sized, max_lines, max_bytes),
                    limit,
                ));
            }
            Check::Tags => {
                let field = ctx.lint.tags_field.as_str();
                reports.push(match ctx.schemas {
                    Some(schemas) => CheckReport::from_findings(
                        "tags-outside-enum",
                        Severity::Error,
                        out_of_enum_tag_findings(&docs, field, &|path| {
                            permitted_values(schemas, path, field)
                        }),
                        limit,
                    ),
                    None => CheckReport::skipped(
                        "tags-outside-enum",
                        Severity::Error,
                        "skipped: the schema tree is invalid or unavailable, so the permitted \
                         tags are unknown (run `validate` for details)",
                    ),
                });
                reports.push(CheckReport::from_findings(
                    "tags-singleton",
                    Severity::Info,
                    singleton_tag_findings(&docs, field),
                    limit,
                ));
            }
        }
    }
    Ok(HealthReport { checks: reports })
}

/// Human-readable rendering for the `lint` subcommand: one block per report.
pub fn write_health(w: &mut impl std::io::Write, r: &HealthReport) -> std::io::Result<()> {
    for c in &r.checks {
        if c.count == 0 {
            writeln!(w, "{} [{}]: none", c.check, c.severity.label())?;
        } else {
            writeln!(
                w,
                "{} [{}]: {} finding(s) (as of the last index run)",
                c.check,
                c.severity.label(),
                c.count
            )?;
        }
        for f in &c.findings {
            writeln!(w, "  {}: {}", f.path, f.detail)?;
        }
        if c.truncated {
            writeln!(w, "  ... and {} more not shown", c.count - c.findings.len())?;
        }
        if let Some(note) = &c.note {
            writeln!(w, "  note: {note}")?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pairs(v: &[(&str, &str)]) -> Vec<(String, String)> {
        v.iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect()
    }

    fn strings(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn doc(path: &str, frontmatter: Value, mtime: i64) -> DocumentSummary {
        DocumentSummary {
            file_path: path.to_string(),
            title: None,
            description: None,
            mtime,
            indexed_at: String::new(),
            frontmatter,
        }
    }

    /// 2024-01-01T00:00:00Z.
    const NOW: i64 = 1_704_067_200;

    fn days_ago(days: i64) -> i64 {
        NOW - days * SECONDS_PER_DAY
    }

    fn stale_findings(docs: &[DocumentSummary], now: i64, lint: &LintConfig) -> Vec<Finding> {
        stale_entries(docs, now, lint)
            .into_iter()
            .map(|(finding, _)| finding)
            .collect()
    }

    #[test]
    fn dead_link_is_detected_and_present_target_is_not() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("excluded.md"), "x").unwrap();
        let r = from_pairs(
            pairs(&[("a.md", "excluded.md"), ("a.md", "gone.md")]),
            dir.path(),
            10,
        );
        assert_eq!(r.count, 1);
        assert_eq!(r.documents[0].path, "a.md");
        assert_eq!(r.documents[0].broken[0].target, "gone.md");
        assert!(!r.truncated);
    }

    #[test]
    fn groups_by_source() {
        let dir = tempfile::tempdir().unwrap();
        let r = from_pairs(
            pairs(&[("a.md", "x.md"), ("a.md", "y.md"), ("b.md", "x.md")]),
            dir.path(),
            10,
        );
        assert_eq!(r.count, 3);
        assert_eq!(r.documents.len(), 2);
        assert_eq!(r.documents[0].broken.len(), 2);
    }

    #[test]
    fn limit_truncates_but_count_stays_true() {
        let dir = tempfile::tempdir().unwrap();
        let r = from_pairs(
            pairs(&[("a.md", "x.md"), ("a.md", "y.md"), ("b.md", "x.md")]),
            dir.path(),
            2,
        );
        assert_eq!(r.count, 3);
        assert!(r.truncated);
        assert_eq!(r.documents.iter().map(|d| d.broken.len()).sum::<usize>(), 2);
        let check = dead_link_check(&r);
        assert_eq!(check.check, "dead-links");
        assert_eq!(check.severity, Severity::Error);
        assert_eq!(check.count, 3);
        assert_eq!(check.findings.len(), 2);
        assert!(check.truncated);
        let mut buf = Vec::new();
        write_health(
            &mut buf,
            &HealthReport {
                checks: vec![check],
            },
        )
        .unwrap();
        assert!(String::from_utf8(buf).unwrap().contains("1 more not shown"));
    }

    #[test]
    fn orphans_skip_linked_documents_entry_points_and_self_links() {
        let paths = strings(&["a.md", "b.md", "c.md", "index.md", "guides/home.md", "d.md"]);
        let links = pairs(&[
            ("a.md", "b.md"),
            // A self-link does not make c.md reachable.
            ("c.md", "c.md"),
            // An orphan can still be a source of inbound links.
            ("d.md", "a.md"),
        ]);
        let found = orphan_findings(&paths, &links, &strings(&["index.md", "guides/*.md"]));
        let orphans: Vec<&str> = found.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(orphans, ["c.md", "d.md"]);
        // No entry points: the hubs are reported too.
        let all = orphan_findings(&paths, &links, &[]);
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn stale_uses_the_frontmatter_date_before_mtime() {
        let lint = LintConfig::default();
        let old = (days_ago(400) / SECONDS_PER_DAY) * SECONDS_PER_DAY;
        let old_date = chrono::DateTime::from_timestamp(old, 0)
            .unwrap()
            .date_naive()
            .to_string();
        let docs = vec![
            // Recent mtime, but the authored date says it is old: the field wins.
            doc("old-field.md", json!({"updated": old_date}), days_ago(1)),
            // Old mtime, but the authored date is recent.
            doc(
                "fresh-field.md",
                json!({"updated": "2023-12-25"}),
                days_ago(900),
            ),
            // No field: falls back to the old mtime.
            doc("old-mtime.md", json!({}), days_ago(500)),
            // Unparseable field: falls back too.
            doc("bad-field.md", json!({"updated": "soon"}), days_ago(500)),
            // Unknown mtime and no field: nothing to judge by.
            doc("unknown.md", json!({}), 0),
            // A date in the future is not stale.
            doc("future.md", json!({"updated": "2030-01-01"}), days_ago(900)),
        ];
        let found = stale_findings(&docs, NOW, &lint);
        let paths: Vec<&str> = found.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, ["old-field.md", "old-mtime.md", "bad-field.md"]);
        assert!(found[0].detail.contains("frontmatter 'updated'"));
        assert!(found[0].detail.contains("400 days"));
        assert!(found[1].detail.contains("file modification time"));
    }

    #[test]
    fn stale_threshold_is_inclusive_and_draft_is_shorter() {
        let lint = LintConfig::default();
        let docs = vec![
            doc("exact.md", json!({}), days_ago(365)),
            doc("just-under.md", json!({}), days_ago(364)),
            doc("draft-old.md", json!({"status": "Draft"}), days_ago(90)),
            doc("draft-ok.md", json!({"status": "draft"}), days_ago(89)),
            doc("done-90.md", json!({"status": "done"}), days_ago(90)),
        ];
        let found = stale_findings(&docs, NOW, &lint);
        let paths: Vec<&str> = found.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, ["exact.md", "draft-old.md"]);
        assert!(found[1].detail.contains("limit 90 days for a draft"));
    }

    #[test]
    fn stale_field_may_be_a_dotted_path() {
        let lint = LintConfig {
            stale_field: "meta.reviewed".into(),
            ..LintConfig::default()
        };
        let docs = vec![doc(
            "a.md",
            json!({"meta": {"reviewed": "2020-01-01T00:00:00Z"}}),
            days_ago(1),
        )];
        assert_eq!(stale_findings(&docs, NOW, &lint).len(), 1);
    }

    #[test]
    fn oversize_prefilter_and_limits() {
        // A file no larger than max_lines bytes cannot have more than max_lines lines.
        assert!(!may_exceed_max_lines(2000, 2000));
        assert!(may_exceed_max_lines(2001, 2000));

        let docs = vec![
            SizedDoc {
                path: "long.md".into(),
                bytes: 10_000,
                lines: Some(2500),
            },
            SizedDoc {
                path: "fat.md".into(),
                bytes: 30_000,
                lines: Some(300),
            },
            SizedDoc {
                path: "both.md".into(),
                bytes: 30_000,
                lines: Some(3000),
            },
            SizedDoc {
                path: "ok.md".into(),
                bytes: 5_000,
                lines: None,
            },
            // Exactly at the limits is not over.
            SizedDoc {
                path: "edge.md".into(),
                bytes: 16_000,
                lines: Some(2000),
            },
        ];
        let found = oversize_findings(&docs, 2000, 16_000);
        let by_path: BTreeMap<&str, &str> = found
            .iter()
            .map(|f| (f.path.as_str(), f.detail.as_str()))
            .collect();
        assert_eq!(by_path.len(), 3);
        assert_eq!(by_path["long.md"], "2500 lines (limit 2000)");
        assert_eq!(by_path["fat.md"], "30000 bytes (limit 16000)");
        assert_eq!(
            by_path["both.md"],
            "3000 lines (limit 2000); 30000 bytes (limit 16000)"
        );
    }

    #[test]
    fn count_lines_counts_a_final_unterminated_line_and_refuses_symlinks() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), "one\ntwo\nthree").unwrap();
        std::fs::write(dir.path().join("b.md"), "one\ntwo\n").unwrap();
        std::fs::write(dir.path().join("empty.md"), "").unwrap();
        std::os::unix::fs::symlink("a.md", dir.path().join("link.md")).unwrap();
        assert_eq!(count_lines(dir.path(), "a.md"), Some(3));
        assert_eq!(count_lines(dir.path(), "b.md"), Some(2));
        assert_eq!(count_lines(dir.path(), "empty.md"), Some(0));
        assert_eq!(count_lines(dir.path(), "link.md"), None);
        assert_eq!(count_lines(dir.path(), "missing.md"), None);
    }

    #[test]
    fn singleton_tags_name_the_one_document() {
        let docs = vec![
            doc("a.md", json!({"tags": ["shared", "only-a", "shared"]}), 1),
            doc("b.md", json!({"tags": ["shared"]}), 1),
            doc("c.md", json!({"tags": "lonely"}), 1),
            doc("d.md", json!({"tags": [1, null]}), 1),
            doc("e.md", json!({}), 1),
        ];
        let mut found = singleton_tag_findings(&docs, "tags");
        found.sort();
        let got: Vec<(&str, &str)> = found
            .iter()
            .map(|f| (f.path.as_str(), f.detail.as_str()))
            .collect();
        assert_eq!(
            got,
            [
                ("a.md", "tag 'only-a' is used by no other document"),
                ("c.md", "tag 'lonely' is used by no other document"),
            ]
        );
    }

    #[test]
    fn out_of_enum_tags_follow_the_per_document_schema() {
        let docs = vec![
            doc("food/a.md", json!({"tags": ["soup", "bogus"]}), 1),
            doc("other/b.md", json!({"tags": ["anything"]}), 1),
        ];
        let allowed = |path: &str| {
            path.starts_with("food/")
                .then(|| strings(&["soup", "stew"]))
        };
        let found = out_of_enum_tag_findings(&docs, "tags", &allowed);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].path, "food/a.md");
        assert!(found[0].detail.contains("'bogus'"));
    }

    #[test]
    fn findings_are_sorted_deduplicated_and_capped() {
        let findings = vec![
            Finding::new("b.md", "x"),
            Finding::new("a.md", "z"),
            Finding::new("a.md", "y"),
            Finding::new("a.md", "y"),
        ];
        let r = CheckReport::from_findings("orphans", Severity::Warning, findings.clone(), 10);
        let got: Vec<(&str, &str)> = r
            .findings
            .iter()
            .map(|f| (f.path.as_str(), f.detail.as_str()))
            .collect();
        assert_eq!(got, [("a.md", "y"), ("a.md", "z"), ("b.md", "x")]);
        assert_eq!(r.count, 3);
        assert!(!r.truncated);

        let capped = CheckReport::from_findings("orphans", Severity::Warning, findings, 2);
        assert_eq!(capped.count, 3);
        assert_eq!(capped.findings.len(), 2);
        assert!(capped.truncated);
        // Input order does not change the output.
        let reversed = CheckReport::from_findings(
            "orphans",
            Severity::Warning,
            vec![
                Finding::new("a.md", "y"),
                Finding::new("a.md", "z"),
                Finding::new("b.md", "x"),
            ],
            2,
        );
        assert_eq!(reversed, capped);
    }

    #[test]
    fn report_json_omits_default_keys() {
        let clean = CheckReport::from_findings("orphans", Severity::Warning, Vec::new(), 10);
        let json = serde_json::to_value(&clean).unwrap();
        assert_eq!(
            json,
            json!({"check": "orphans", "severity": "warning", "count": 0})
        );
        let noisy = CheckReport::from_findings(
            "tags-singleton",
            Severity::Info,
            vec![Finding::new("a.md", "d")],
            1,
        );
        let json = serde_json::to_value(&noisy).unwrap();
        assert_eq!(json["severity"], "info");
        assert_eq!(json["findings"][0]["path"], "a.md");
        assert!(json.get("truncated").is_none() && json.get("note").is_none());
    }

    #[test]
    fn check_names_parse_and_the_config_picks_the_default_set() {
        assert_eq!(
            parse_check_names(["orphans,stale", "orphans"]).unwrap(),
            [Check::Orphans, Check::Stale]
        );
        let err = parse_check_names(["orphans", "nope,worse"]).unwrap_err();
        assert!(err.contains("nope, worse"), "{err}");

        let lint = LintConfig {
            stale: false,
            tags: false,
            ..LintConfig::default()
        };
        assert_eq!(
            enabled_checks(&lint),
            [Check::DeadLinks, Check::Orphans, Check::Oversize]
        );
    }

    async fn seed_doc(db: &StateDb, path: &str, fm: Value, mtime: i64, size: i64) {
        db.upsert(path, "h", 1, "sh", "", mtime, size)
            .await
            .unwrap();
        let map: std::collections::HashMap<String, Value> = serde_json::from_value(fm).unwrap();
        db.upsert_document_metadata(path, &map, mtime, "h", 1)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn dead_links_agrees_with_the_state_db() {
        let dir = tempfile::tempdir().unwrap();
        let db = StateDb::new(&dir.path().join("state.db")).await.unwrap();
        db.upsert("b.md", "h", 1, "sh", "", 0, 0).await.unwrap();
        db.upsert_document_metadata(
            "b.md",
            &std::collections::HashMap::from([("title".to_string(), serde_json::json!("B"))]),
            1,
            "h",
            1,
        )
        .await
        .unwrap();
        db.replace_links(
            "a.md",
            "markdown",
            &[("b.md".to_string(), None), ("missing.md".to_string(), None)],
        )
        .await
        .unwrap();
        let r = dead_links(&db, dir.path(), 10).await.unwrap();
        assert_eq!(r.count, 1);
        assert_eq!(r.documents[0].broken[0].target, "missing.md");
        let json = serde_json::to_value(&r).unwrap();
        assert!(json.get("truncated").is_none());
    }

    #[tokio::test]
    async fn health_checks_agree_with_the_state_db() {
        let dir = tempfile::tempdir().unwrap();
        // Two files on disk: one past the line limit, one small.
        std::fs::write(dir.path().join("long.md"), "line\n".repeat(30)).unwrap();
        std::fs::write(dir.path().join("hub.md"), "hub\n").unwrap();
        let db = StateDb::new(&dir.path().join("state.db")).await.unwrap();
        seed_doc(
            &db,
            "hub.md",
            json!({"tags": ["shared", "typo"], "updated": "2020-01-01"}),
            days_ago(1),
            4,
        )
        .await;
        seed_doc(
            &db,
            "long.md",
            json!({"tags": ["shared"]}),
            days_ago(1),
            150,
        )
        .await;
        // A third document nobody links to, linking out to the hub and to a ghost.
        seed_doc(&db, "loner.md", json!({}), days_ago(1), 10).await;
        db.replace_links(
            "loner.md",
            "markdown",
            &[
                ("hub.md".to_string(), None),
                ("long.md".to_string(), None),
                ("ghost.md".to_string(), None),
            ],
        )
        .await
        .unwrap();
        // A semantic edge must not make loner.md reachable.
        db.replace_links("hub.md", "semantic", &[("loner.md".to_string(), Some(0.9))])
            .await
            .unwrap();

        let lint = LintConfig {
            max_lines: 20,
            orphan_entry_points: Vec::new(),
            ..LintConfig::default()
        };
        let ctx = HealthContext {
            state: &db,
            data_path: dir.path(),
            lint: &lint,
            section_max_bytes: 16_000,
            schemas: None,
            now: NOW,
        };
        let report = run_health_checks(&ctx, &Check::ALL, 100).await.unwrap();
        let names: Vec<&str> = report.checks.iter().map(|c| c.check).collect();
        assert_eq!(
            names,
            [
                "dead-links",
                "orphans",
                "stale",
                "oversize",
                "tags-outside-enum",
                "tags-singleton"
            ]
        );
        let get = |n: &str| report.checks.iter().find(|c| c.check == n).unwrap();
        assert_eq!(
            get("dead-links").findings[0].detail,
            "broken link to ghost.md"
        );
        let orphans: Vec<&str> = get("orphans")
            .findings
            .iter()
            .map(|f| f.path.as_str())
            .collect();
        assert_eq!(orphans, ["loner.md"]);
        let stale: Vec<&str> = get("stale")
            .findings
            .iter()
            .map(|f| f.path.as_str())
            .collect();
        assert_eq!(stale, ["hub.md"]);
        // hub.md was judged by its frontmatter date, so no mtime caveat applies.
        assert!(get("stale").note.is_none());
        let oversize = get("oversize");
        assert_eq!(oversize.count, 1);
        assert_eq!(oversize.findings[0].path, "long.md");
        assert_eq!(oversize.findings[0].detail, "30 lines (limit 20)");
        let outside = get("tags-outside-enum");
        assert_eq!(outside.count, 0);
        assert!(outside.note.as_deref().unwrap().contains("skipped"));
        assert_eq!(get("tags-singleton").findings[0].path, "hub.md");

        // A single named check runs alone.
        let only = run_health_checks(&ctx, &[Check::Orphans], 100)
            .await
            .unwrap();
        assert_eq!(only.checks.len(), 1);

        // A cap applies per report and keeps the true count.
        let capped = run_health_checks(&ctx, &[Check::Tags], 1).await.unwrap();
        let singleton = capped
            .checks
            .iter()
            .find(|c| c.check == "tags-singleton")
            .unwrap();
        assert_eq!(singleton.count, 1);
        assert!(!singleton.truncated);
    }

    #[tokio::test]
    async fn out_of_enum_tags_use_the_schema_tree() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("food")).unwrap();
        std::fs::write(
            dir.path().join("food/.schema.yaml"),
            "fields:\n  tags:\n    type: list\n    items:\n      type: enum\n      values: [soup, stew]\n",
        )
        .unwrap();
        let db = StateDb::new(&dir.path().join("state.db")).await.unwrap();
        seed_doc(
            &db,
            "food/a.md",
            json!({"tags": ["soup", "bogus"]}),
            days_ago(1),
            10,
        )
        .await;
        seed_doc(
            &db,
            "other/b.md",
            json!({"tags": ["bogus"]}),
            days_ago(1),
            10,
        )
        .await;
        let schemas = SchemaCache::build(
            dir.path(),
            &crate::config::FrontmatterConfig::default(),
            &crate::config::IndexingConfig::default(),
        )
        .unwrap();
        let lint = LintConfig::default();
        let ctx = HealthContext {
            state: &db,
            data_path: dir.path(),
            lint: &lint,
            section_max_bytes: 16_000,
            schemas: Some(&schemas),
            now: NOW,
        };
        let report = run_health_checks(&ctx, &[Check::Tags], 100).await.unwrap();
        let outside = &report.checks[0];
        assert_eq!(outside.check, "tags-outside-enum");
        assert_eq!(outside.severity, Severity::Error);
        assert_eq!(outside.count, 1);
        assert_eq!(outside.findings[0].path, "food/a.md");
        assert!(outside.findings[0].detail.contains("'bogus'"));
        assert!(outside.note.is_none());
    }
}
