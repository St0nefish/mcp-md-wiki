//! `changes_since` (#358) against real throwaway repositories (no remote).

use super::*;
use crate::config::IndexingConfig;

/// `GIT_CEILING_DIRECTORIES` keeps git from climbing into an enclosing
/// checkout (#218).
struct Repo {
    dir: tempfile::TempDir,
}

impl Repo {
    fn new() -> Self {
        let r = Self {
            dir: tempfile::TempDir::new().unwrap(),
        };
        r.git(&["init", "-q"], None);
        r
    }

    fn path(&self) -> &str {
        self.dir.path().to_str().unwrap()
    }

    fn git(&self, args: &[&str], date: Option<&str>) -> String {
        let mut cmd = std::process::Command::new("git");
        cmd.current_dir(self.dir.path())
            .env("GIT_CEILING_DIRECTORIES", self.dir.path().parent().unwrap())
            .args([
                "-c",
                "user.email=t@example.com",
                "-c",
                "user.name=Tester",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args);
        if let Some(d) = date {
            cmd.env("GIT_COMMITTER_DATE", d).env("GIT_AUTHOR_DATE", d);
        }
        let out = cmd.output().unwrap();
        assert!(out.status.success(), "git {args:?}: {out:?}");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn write(&self, rel: &str, content: &str) {
        let p = self.dir.path().join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, content).unwrap();
    }

    fn commit(&self, msg: &str) -> String {
        self.commit_at(msg, "2026-01-01T00:00:00Z")
    }

    fn commit_at(&self, msg: &str, date: &str) -> String {
        self.git(&["add", "-A"], None);
        self.git(&["commit", "-q", "-m", msg], Some(date));
        self.git(&["rev-parse", "HEAD"], None)
    }
}

const BODY: &str = "# Title\n\nSome reasonably long body text so that rename detection\nhas enough content to match the two files on similarity.\nLine three.\nLine four.\nLine five.\n";

async fn run(
    repo: &Repo,
    since: Option<&str>,
    prefix: Option<&str>,
    limit: Option<usize>,
    cursor: Option<&str>,
) -> Result<serde_json::Value, ChangesSinceError> {
    changes_since_json(
        repo.path(),
        &IndexingConfig::default(),
        since,
        prefix,
        limit,
        cursor,
    )
    .await
}

fn statuses(v: &serde_json::Value) -> Vec<(String, String)> {
    v["changes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            (
                c["path"].as_str().unwrap().to_string(),
                c["status"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

fn pair(p: &str, s: &str) -> (String, String) {
    (p.to_string(), s.to_string())
}

#[tokio::test]
async fn net_status_across_a_commit_range() {
    let r = Repo::new();
    r.write("a.md", "# A\n\none\n");
    r.write("b.md", "# B\n\ngone soon\n");
    r.write("old/e.md", BODY);
    r.write("README.md", "readme v1\n");
    let base = r.commit("base");

    r.write("a.md", "# A\n\ntwo\n");
    r.commit("modify a");
    r.write("c.md", "# C\n\nnew\n");
    r.commit("add c");
    r.write("c.md", "# C\n\nnew, edited\n");
    r.commit("modify c");
    r.write("d.md", "# D\n");
    r.commit("add d");
    r.git(&["rm", "-q", "d.md"], None);
    r.commit("delete d");
    r.git(&["rm", "-q", "b.md"], None);
    r.commit("delete b");
    r.write("new/placeholder.md", "# P\n");
    r.git(&["mv", "old/e.md", "new/f.md"], None);
    r.commit("rename e");
    r.write("README.md", "readme v2\n");
    r.write("notes.txt", "not a document\n");
    let head = r.commit("noise");

    let v = run(&r, Some(&base), None, None, None).await.unwrap();
    assert_eq!(v["head"], head);
    assert_eq!(
        statuses(&v),
        vec![
            pair("a.md", "modified"),
            pair("b.md", "deleted"),
            pair("c.md", "added"),
            pair("new/f.md", "renamed"),
            pair("new/placeholder.md", "added"),
        ]
    );
    let renamed = &v["changes"][3];
    assert_eq!(renamed["old_path"], "old/e.md");
    assert_eq!(renamed["subject"], "rename e");
    assert_eq!(renamed["author"], "Tester");
    assert!(renamed["commit"].as_str().unwrap().len() >= 40);
    assert_eq!(v["changes"][2]["subject"], "modify c");
    assert!(v.get("truncated").is_none());
    assert!(!v.to_string().contains("t@example.com"));
    // From the head itself: nothing changed.
    let none = run(&r, Some(&head), None, None, None).await.unwrap();
    assert_eq!(none["changes"], serde_json::json!([]));
    assert_eq!(none["head"], head);
}

#[tokio::test]
async fn provenance_operation_is_reported() {
    let r = Repo::new();
    r.write("a.md", "# A\n");
    let base = r.commit("base");
    r.write("a.md", "# A\n\nmore\n");
    r.commit("Edit a\n\nTool: mcp-md-wiki\nOperation: edit");
    let v = run(&r, Some(&base), None, None, None).await.unwrap();
    assert_eq!(v["changes"][0]["subject"], "Edit a");
    assert_eq!(v["changes"][0]["operation"], "edit");
}

#[tokio::test]
async fn timestamp_since_matches_commits_after_it() {
    let r = Repo::new();
    r.write("a.md", "# A\n");
    r.commit_at("first", "2026-01-01T00:00:00Z");
    r.write("b.md", "# B\n");
    r.commit_at("second", "2026-02-01T00:00:00Z");
    r.write("a.md", "# A\n\nedit\n");
    r.commit_at("third", "2026-03-01T00:00:00Z");

    let v = run(&r, Some("2026-01-15"), None, None, None).await.unwrap();
    assert_eq!(
        statuses(&v),
        vec![pair("a.md", "modified"), pair("b.md", "added")]
    );
    let v = run(&r, Some("2026-02-15T00:00:00Z"), None, None, None)
        .await
        .unwrap();
    assert_eq!(statuses(&v), vec![pair("a.md", "modified")]);
    // Before the first commit: everything is added.
    let v = run(&r, Some("2025-06-01"), None, None, None).await.unwrap();
    assert_eq!(
        statuses(&v),
        vec![pair("a.md", "added"), pair("b.md", "added")]
    );
    // After the last: nothing.
    let v = run(&r, Some("2027-01-01"), None, None, None).await.unwrap();
    assert!(v["changes"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn bad_input_is_invalid_not_a_git_failure() {
    let r = Repo::new();
    r.write("a.md", "# A\n");
    r.commit("base");
    for since in ["deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "nope", "--all"] {
        assert!(
            matches!(
                run(&r, Some(since), None, None, None).await,
                Err(ChangesSinceError::Invalid(_))
            ),
            "{since}"
        );
    }
    assert!(matches!(
        run(&r, None, None, None, None).await,
        Err(ChangesSinceError::Invalid(_))
    ));
    assert!(matches!(
        run(&r, None, None, None, Some("garbage")).await,
        Err(ChangesSinceError::Invalid(_))
    ));
    let bad = format!("{}.{}.{}", "1".repeat(40), "2".repeat(40), "00");
    assert!(matches!(
        run(&r, None, None, None, Some(&bad)).await,
        Err(ChangesSinceError::Invalid(_))
    ));
}

#[tokio::test]
async fn paginates_with_a_cursor_pinned_to_the_original_range() {
    let r = Repo::new();
    r.write("seed.md", "# S\n");
    let base = r.commit("base");
    for n in 0..5 {
        r.write(&format!("d/p{n}.md"), "# P\n");
    }
    let head = r.commit("add five");

    let p1 = run(&r, Some(&base), None, Some(2), None).await.unwrap();
    assert_eq!(p1["truncated"], true);
    assert_eq!(p1["head"], head);
    assert_eq!(statuses(&p1).len(), 2);
    let c1 = p1["next_cursor"].as_str().unwrap().to_string();

    // The corpus moves on between pages; the cursor still reads the same range.
    r.write("d/p9.md", "# P\n");
    r.commit("late arrival");

    let p2 = run(&r, None, None, Some(2), Some(&c1)).await.unwrap();
    assert_eq!(p2["head"], head);
    assert_eq!(
        statuses(&p2),
        vec![pair("d/p2.md", "added"), pair("d/p3.md", "added")]
    );
    let c2 = p2["next_cursor"].as_str().unwrap().to_string();
    let p3 = run(&r, None, None, Some(2), Some(&c2)).await.unwrap();
    assert_eq!(statuses(&p3), vec![pair("d/p4.md", "added")]);
    assert!(p3.get("next_cursor").is_none());
    assert!(p3.get("truncated").is_none());
}

#[tokio::test]
async fn path_prefix_scopes_and_moves_across_the_boundary_read_as_add_or_delete() {
    let r = Repo::new();
    r.write("in/x.md", BODY);
    r.write("out/y.md", &format!("{BODY}unique tail\n"));
    r.write("in/keep.md", "# K\n");
    let base = r.commit("base");
    r.git(&["mv", "in/x.md", "out/x.md"], None);
    r.git(&["mv", "out/y.md", "in/y.md"], None);
    r.write("in/keep.md", "# K\n\nedited\n");
    r.write("elsewhere/z.md", "# Z\n");
    r.commit("shuffle");

    let v = run(&r, Some(&base), Some("/in/"), None, None)
        .await
        .unwrap();
    assert_eq!(
        statuses(&v),
        vec![
            pair("in/keep.md", "modified"),
            pair("in/x.md", "deleted"),
            pair("in/y.md", "added"),
        ]
    );
    // A prefix is a folder, not a substring.
    let v = run(&r, Some(&base), Some("in"), None, None).await.unwrap();
    assert_eq!(statuses(&v).len(), 3);
    let v = run(&r, Some(&base), Some("i"), None, None).await.unwrap();
    assert!(v["changes"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn non_git_directory_is_unavailable_and_an_empty_repo_has_no_changes() {
    let dir = tempfile::TempDir::new().unwrap();
    let v = changes_since_json(
        dir.path().to_str().unwrap(),
        &IndexingConfig::default(),
        Some("2026-01-01"),
        None,
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(v, serde_json::json!({"available": false}));

    let r = Repo::new();
    let v = run(&r, Some("2026-01-01"), None, None, None).await.unwrap();
    assert_eq!(v["changes"], serde_json::json!([]));
    assert!(matches!(
        run(&r, Some("deadbeef"), None, None, None).await,
        Err(ChangesSinceError::Invalid(_))
    ));
}

#[test]
fn since_timestamps_parse_in_the_documented_forms() {
    for s in [
        "2026-03-01",
        "2026-03-01T12:30Z",
        "2026-03-01T12:30:00Z",
        "2026-03-01T12:30:00+02:00",
        "2026-03-01 12:30",
    ] {
        assert!(parse_since_timestamp(s).is_some(), "{s}");
    }
    for s in ["", "abc1234", "20260301", "main"] {
        assert!(parse_since_timestamp(s).is_none(), "{s}");
    }
}
