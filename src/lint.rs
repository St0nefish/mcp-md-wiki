//! Corpus-wide lint checks (#354). One check today: the dead-link report.
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

use std::path::Path;

use anyhow::Result;

use crate::state::StateDb;
use crate::validate;

/// Default cap on the number of broken links a report lists.
pub const DEFAULT_DEAD_LINK_LIMIT: usize = validate::MAX_BROKEN_LINKS_SHOWN;

/// Largest cap a caller (the `/status/links` query string) may ask for.
pub const MAX_DEAD_LINK_LIMIT: usize = 5000;

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

/// Human-readable rendering for the `lint` subcommand.
pub fn write_dead_links(w: &mut impl std::io::Write, r: &DeadLinkReport) -> std::io::Result<()> {
    if r.count == 0 {
        return writeln!(w, "dead-links: none");
    }
    writeln!(
        w,
        "dead-links: {} broken link(s) in {} document(s) (as of the last index run)",
        r.count,
        r.documents.len()
    )?;
    for d in &r.documents {
        writeln!(w, "  {}", d.path)?;
        for l in &d.broken {
            match l.line {
                Some(n) => writeln!(w, "    -> {} (line {n})", l.target)?,
                None => writeln!(w, "    -> {}", l.target)?,
            }
        }
    }
    if r.truncated {
        let shown: usize = r.documents.iter().map(|d| d.broken.len()).sum();
        writeln!(w, "  ... and {} more not shown", r.count - shown)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(v: &[(&str, &str)]) -> Vec<(String, String)> {
        v.iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
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
        let mut buf = Vec::new();
        write_dead_links(&mut buf, &r).unwrap();
        assert!(String::from_utf8(buf).unwrap().contains("1 more not shown"));
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
}
