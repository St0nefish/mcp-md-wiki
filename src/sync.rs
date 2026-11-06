//! On-demand sync of the knowledge-base clone from its remote: `git fetch` + ff-only
//! merge (falling back to rebasing unpushed local commits when the clone has
//! diverged) + diff of exactly what the pull brought in, then the include/exclude
//! filter and marking the survivors dirty on the reindex queue.
//!
//! One implementation behind three entry points — the push webhook
//! (`webhook::handle_webhook`), `POST /admin/sync` (`server.rs`) and the `git-pull` CLI
//! subcommand (`main.rs`) — so they cannot drift apart. The git phase ([`pull_clone`])
//! holds [`crate::git::GIT_LOCK`] across the whole rev-parse -> fetch -> merge ->
//! rev-parse -> diff sequence: per-command locking would let a write land between the
//! fetch and the merge, or between the merge and the diff and attribute that write's
//! files to this pull. The fetch and merge run inline here rather than in `git.rs`;
//! they are covered by that guard being alive, not by a signature. The diverged-clone
//! fallback (`git::integrate_fetch_head`) takes the same guard by reference.

use std::path::PathBuf;

use tracing::{debug, error, info, warn};

use crate::config::ResolvedConfig;
use crate::git::{
    GIT_TIMEOUT, Integrated, authenticated_url, git_command, integrate_fetch_head, redact_url,
    run_bounded,
};
use crate::reindex::ReindexQueue;

/// Why a sync failed. `message` is a short fixed string safe to return to a webhook
/// caller; `detail` carries the (URL-redacted) operator-facing specifics and is shown
/// only on authenticated surfaces (`/admin/sync`, the CLI).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncError {
    pub kind: SyncErrorKind,
    pub message: &'static str,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncErrorKind {
    /// `source.git_url` is not set, so there is no remote to pull from.
    NoRemote,
    /// The data path is not a git checkout.
    NotACheckout,
    /// A git step failed or timed out.
    Git,
}

impl std::fmt::Display for SyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.detail {
            Some(d) => write!(f, "{}: {}", self.message, d),
            None => f.write_str(self.message),
        }
    }
}

impl std::error::Error for SyncError {}

fn git_err(message: &'static str, detail: impl Into<Option<String>>) -> SyncError {
    SyncError {
        kind: SyncErrorKind::Git,
        message,
        detail: detail.into(),
    }
}

/// What the git phase of a sync found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PullOutcome {
    pub old_head: String,
    pub new_head: String,
    /// Every path the pull changed, before any filtering.
    pub changed: Vec<PathBuf>,
}

/// What a full sync (pull + filter + mark) did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncOutcome {
    pub old_head: String,
    pub new_head: String,
    /// Paths marked dirty (indexable ones).
    pub marked: usize,
    /// Changed paths dropped by the include/exclude filter.
    pub filtered_out: usize,
    /// A schema file changed, so a full reconcile was queued.
    pub schema_changed: bool,
}

/// Fetch the configured branch and fast-forward the clone to it, returning the paths
/// the pull changed. Does not touch the reindex queue.
///
/// When the clone cannot fast-forward because it holds local commits the remote
/// lacks (a push that failed after its commit landed), this falls back to
/// `git::integrate_fetch_head`, the same rebase (or, on a genuine conflict,
/// park-and-reset) a write's pre-write sync applies, under the one guard this
/// function already holds. The local commits are not pushed here; the next write
/// pushes them. `changed` is still the diff between HEAD before the fetch and HEAD
/// after, so it holds what the remote brought in (plus, after a conflict reset,
/// the parked commits' paths, which changed on disk too).
///
/// `git_token` is injected into the fetch URL per call (never persisted in the clone's
/// remote config), so a clone made before a token existed still pulls.
pub async fn pull_clone(
    config: &ResolvedConfig,
    git_token: Option<&str>,
) -> Result<PullOutcome, SyncError> {
    let Some(git_url) = config.source.git_url.as_deref() else {
        return Err(SyncError {
            kind: SyncErrorKind::NoRemote,
            message: "No git remote configured",
            detail: Some(
                "source.git_url is not set, so there is no remote to pull from".to_string(),
            ),
        });
    };
    let data_path = config.data_path();
    let branch = &config.source.branch;

    if !std::path::Path::new(data_path).join(".git").exists() {
        return Err(SyncError {
            kind: SyncErrorKind::NotACheckout,
            message: "Data path is not a git checkout",
            detail: Some(format!("{data_path} has no .git directory")),
        });
    }

    let fetch_url = authenticated_url(git_url, git_token);
    if git_token.is_none_or(str::is_empty) {
        debug!("Sync running without a git token; a private remote will refuse the fetch");
    }

    let git_lock = crate::git::lock_git().await;

    // Captured before the fetch so the range diffed below covers exactly what this
    // pull brought in.
    let old_head = crate::git::rev_parse_head(&git_lock, data_path)
        .await
        .map_err(|e| {
            error!("Failed to read HEAD before fetch: {:#}", e);
            git_err("Failed to read local HEAD", format!("{e:#}"))
        })?;

    info!(
        "Running git fetch in {} from {}",
        data_path,
        redact_url(&fetch_url)
    );

    let fetch = run_bounded(
        git_command()
            .args([
                "-c",
                &format!("safe.directory={}", data_path),
                "fetch",
                "--no-tags",
                &fetch_url,
                branch,
            ])
            .current_dir(data_path),
    )
    .await;
    match fetch {
        Err(_) => {
            error!("git fetch timed out after {:?}", GIT_TIMEOUT);
            return Err(git_err("Git fetch timed out", None));
        }
        Ok(Err(e)) => {
            error!("Failed to run git fetch: {}", e);
            return Err(git_err("Failed to run git", format!("{e}")));
        }
        Ok(Ok(o)) if o.status.success() => info!("Git fetch succeeded"),
        Ok(Ok(o)) => {
            let stderr = redact_url(&String::from_utf8_lossy(&o.stderr));
            error!("Git fetch failed: {}", stderr);
            let hint = if git_token.is_none_or(str::is_empty) {
                " (no git token is set; a private remote needs one, see source.git_token_env)"
            } else {
                ""
            };
            return Err(git_err(
                "Git fetch failed",
                format!("{}{hint}", stderr.trim()),
            ));
        }
    }

    let merge = run_bounded(
        git_command()
            .args([
                "-c",
                &format!("safe.directory={}", data_path),
                "merge",
                "--ff-only",
                "FETCH_HEAD",
            ])
            .current_dir(data_path),
    )
    .await;
    match merge {
        Err(_) => {
            error!("git merge timed out after {:?}", GIT_TIMEOUT);
            return Err(git_err("Git merge timed out", None));
        }
        Ok(Err(e)) => {
            error!("Failed to run git merge: {}", e);
            return Err(git_err("Failed to run git", format!("{e}")));
        }
        Ok(Ok(o)) if o.status.success() => info!("Git merge (ff-only) succeeded"),
        Ok(Ok(o)) => {
            // No fast-forward: most often the clone holds a commit a failed push
            // left unpushed while the remote moved on. Reconcile it the way the
            // next write would (`git::sync_to_remote`), under this same guard,
            // rather than leaving every pull failing until a write happens.
            let stderr = redact_url(&String::from_utf8_lossy(&o.stderr));
            info!(
                "Git merge (ff-only) failed; rebasing local commits onto the remote: {}",
                stderr.trim()
            );
            match integrate_fetch_head(
                &git_lock,
                data_path,
                &config.write.commit_author_name,
                &config.write.commit_author_email,
            )
            .await
            {
                Integrated::FastForwarded => info!("Git merge (ff-only) succeeded on retry"),
                Integrated::Rebased => info!(
                    "Rebased unpushed local commits onto the remote; they are pushed with \
                     the next write"
                ),
                Integrated::ResetToRemote => warn!(
                    "Unpushed local commits conflicted with the remote; the clone was reset \
                     to the remote tip with them saved under a recovery ref"
                ),
                Integrated::RebaseFailed(e) => {
                    let detail = redact_url(&format!("{e:#}"));
                    error!("Git merge failed and the rebase fallback failed too: {detail}");
                    return Err(git_err(
                        "Git merge failed",
                        format!("{}; rebase fallback: {detail}", stderr.trim()),
                    ));
                }
                Integrated::ResetFailed(e) => {
                    let detail = redact_url(&format!("{e:#}"));
                    error!("Git merge failed and resetting to the remote tip failed: {detail}");
                    return Err(git_err(
                        "Git merge failed",
                        format!("{}; reset to the remote tip: {detail}", stderr.trim()),
                    ));
                }
            }
        }
    }

    let new_head = crate::git::rev_parse_head(&git_lock, data_path)
        .await
        .map_err(|e| {
            error!("Failed to read HEAD after merge: {:#}", e);
            git_err("Failed to read local HEAD", format!("{e:#}"))
        })?;

    let changed = crate::git::git_diff_name_status(&git_lock, data_path, &old_head, &new_head)
        .await
        .map_err(|e| {
            error!("Failed to diff pulled range: {:#}", e);
            git_err("Failed to diff pulled changes", format!("{e:#}"))
        })?;

    Ok(PullOutcome {
        old_head,
        new_head,
        changed,
    })
}

/// Filter `changed` through the include/exclude predicate a full reconcile applies
/// (#278) and mark the survivors dirty. A changed schema file is never indexable, so
/// it is checked first: it queues a full reconcile (which rebuilds the schema cache).
/// Returns `(marked, filtered_out, schema_changed)`.
pub fn mark_changed(
    config: &ResolvedConfig,
    queue: &ReindexQueue,
    changed: Vec<PathBuf>,
) -> (usize, usize, bool) {
    let schema_changed = queue.mark_schema_changes(&config.indexing, &changed);
    if schema_changed {
        info!("Pull changed a schema file; queued a full reconcile");
    }
    let (indexable, filtered_out) = crate::ingest::partition_indexable(&config.indexing, changed);
    if !filtered_out.is_empty() {
        debug!(
            filtered = filtered_out.len(),
            paths = ?filtered_out,
            "Pull diff included non-indexable paths; skipping"
        );
    }
    let marked = indexable.len();
    queue.mark_paths(indexable);
    (marked, filtered_out.len(), schema_changed)
}

/// The whole sequence: [`pull_clone`] then [`mark_changed`]. Marking happens after the
/// git lock is released (it touches no git state).
pub async fn pull_and_mark(
    config: &ResolvedConfig,
    git_token: Option<&str>,
    queue: &ReindexQueue,
) -> Result<SyncOutcome, SyncError> {
    let pulled = pull_clone(config, git_token).await?;
    let (marked, filtered_out, schema_changed) = mark_changed(config, queue, pulled.changed);
    info!(
        old = %pulled.old_head,
        new = %pulled.new_head,
        changed = marked,
        "Pull applied; marked changed paths dirty"
    );
    Ok(SyncOutcome {
        old_head: pulled.old_head,
        new_head: pulled.new_head,
        marked,
        filtered_out,
        schema_changed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::process::Command as StdCommand;

    fn git(dir: &Path, args: &[&str]) {
        let out = StdCommand::new("git")
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@example.com",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .current_dir(dir)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    fn base_config() -> ResolvedConfig {
        ResolvedConfig {
            source: crate::config::ResolvedSourceConfig {
                git_url: None,
                branch: "master".into(),
                data_path: Some("/tmp".into()),
                git_token_env: "GIT_PULL_TOKEN".into(),
            },
            indexing: Default::default(),
            frontmatter: Default::default(),
            chunking: Default::default(),
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
                collection: "knowledge-base".into(),
            },
            validation: Default::default(),
            webhook: Default::default(),
            mcp: Default::default(),
            rate_limit: Default::default(),
            write: Default::default(),
            search: Default::default(),
            reranking: None,
            ui: Default::default(),
            lint: Default::default(),
            provenance: Default::default(),
        }
    }

    /// A remote with one commit, a clone of it, and a config pointing at both.
    fn fixture() -> (tempfile::TempDir, tempfile::TempDir, ResolvedConfig) {
        let remote = tempfile::tempdir().unwrap();
        git(remote.path(), &["init", "-q", "-b", "master"]);
        std::fs::write(remote.path().join("a.md"), "# A\n").unwrap();
        git(remote.path(), &["add", "-A"]);
        git(remote.path(), &["commit", "-q", "-m", "init"]);

        let clone = tempfile::tempdir().unwrap();
        let out = StdCommand::new("git")
            .args(["clone", "-q"])
            .arg(remote.path())
            .arg(clone.path())
            .output()
            .unwrap();
        assert!(out.status.success());

        let mut config = base_config();
        config.source.git_url = Some(remote.path().to_string_lossy().into_owned());
        config.source.branch = "master".into();
        config.source.data_path = Some(clone.path().to_string_lossy().into_owned());
        (remote, clone, config)
    }

    #[tokio::test]
    async fn pull_and_mark_marks_only_indexable_changed_paths() {
        let (remote, _clone, config) = fixture();
        std::fs::write(remote.path().join("b.md"), "# B\n").unwrap();
        std::fs::write(remote.path().join("notes.txt"), "not markdown\n").unwrap();
        git(remote.path(), &["add", "-A"]);
        git(remote.path(), &["commit", "-q", "-m", "more"]);

        let queue = ReindexQueue::new();
        let out = pull_and_mark(&config, None, &queue).await.unwrap();
        assert_ne!(out.old_head, out.new_head);
        assert_eq!(out.marked, 1);
        assert_eq!(out.filtered_out, 1);
        assert!(!out.schema_changed);
        assert!(!queue.snapshot().full_pending);
        assert_eq!(
            queue.snapshot_paths(),
            std::collections::HashSet::from([PathBuf::from("b.md")])
        );
    }

    /// A clone holding a commit a failed push left unpushed, while the remote
    /// moved on, cannot fast-forward. The pull rebases the local commit onto the
    /// remote instead of failing, and reports only what the remote brought in.
    #[tokio::test]
    async fn a_diverged_clone_is_rebased_onto_the_remote() {
        let (remote, clone, config) = fixture();
        std::fs::write(clone.path().join("local.md"), "# Local\n").unwrap();
        git(clone.path(), &["add", "-A"]);
        git(clone.path(), &["commit", "-q", "-m", "unpushed"]);
        std::fs::write(remote.path().join("remote.md"), "# Remote\n").unwrap();
        git(remote.path(), &["add", "-A"]);
        git(remote.path(), &["commit", "-q", "-m", "remote"]);

        let out = pull_clone(&config, None).await.unwrap();
        assert_ne!(out.old_head, out.new_head);
        assert_eq!(out.changed, vec![PathBuf::from("remote.md")]);
        assert!(clone.path().join("local.md").is_file());
        assert!(clone.path().join("remote.md").is_file());

        // The local commit now sits on top of the remote tip, so the next pull
        // fast-forwards (here: has nothing to do) and the next write pushes it.
        let log = StdCommand::new("git")
            .args(["log", "--format=%s"])
            .current_dir(clone.path())
            .output()
            .unwrap();
        let subjects: Vec<String> = String::from_utf8_lossy(&log.stdout)
            .lines()
            .map(str::to_string)
            .collect();
        assert_eq!(subjects, ["unpushed", "remote", "init"]);
        assert!(!clone.path().join(".git/rebase-merge").exists());

        let again = pull_clone(&config, None).await.unwrap();
        assert_eq!(again.old_head, again.new_head);
        assert!(again.changed.is_empty());
    }

    /// A local commit that conflicts with the remote is parked under a recovery
    /// ref and the clone reset to the remote tip, as a write's pre-write sync does.
    #[tokio::test]
    async fn a_diverged_clone_that_conflicts_is_reset_to_the_remote() {
        let (remote, clone, config) = fixture();
        std::fs::write(clone.path().join("a.md"), "# A local\n").unwrap();
        git(clone.path(), &["commit", "-q", "-am", "unpushed"]);
        std::fs::write(remote.path().join("a.md"), "# A remote\n").unwrap();
        git(remote.path(), &["commit", "-q", "-am", "remote"]);

        let out = pull_clone(&config, None).await.unwrap();
        assert_eq!(out.changed, vec![PathBuf::from("a.md")]);
        assert_eq!(
            std::fs::read_to_string(clone.path().join("a.md")).unwrap(),
            "# A remote\n"
        );
        let refs = StdCommand::new("git")
            .args([
                "for-each-ref",
                "--format=%(refname)",
                "refs/mcp-md-wiki/unsynced",
            ])
            .current_dir(clone.path())
            .output()
            .unwrap();
        assert!(!String::from_utf8_lossy(&refs.stdout).trim().is_empty());
    }

    #[tokio::test]
    async fn a_schema_change_queues_a_full_reconcile() {
        let (remote, _clone, config) = fixture();
        std::fs::write(remote.path().join(".schema.yaml"), "fields: {}\n").unwrap();
        git(remote.path(), &["add", "-A"]);
        git(remote.path(), &["commit", "-q", "-m", "schema"]);

        let queue = ReindexQueue::new();
        let out = pull_and_mark(&config, None, &queue).await.unwrap();
        assert!(out.schema_changed);
        assert!(queue.snapshot().full_pending);
    }

    #[tokio::test]
    async fn an_up_to_date_clone_marks_nothing() {
        let (_remote, _clone, config) = fixture();
        let queue = ReindexQueue::new();
        let out = pull_and_mark(&config, None, &queue).await.unwrap();
        assert_eq!(out.old_head, out.new_head);
        assert_eq!(out.marked, 0);
        assert_eq!(queue.snapshot().pending_paths, 0);
    }

    #[tokio::test]
    async fn no_remote_configured_is_a_clear_error() {
        let (_remote, _clone, mut config) = fixture();
        config.source.git_url = None;
        let err = pull_clone(&config, None).await.unwrap_err();
        assert_eq!(err.kind, SyncErrorKind::NoRemote);
        assert!(err.to_string().contains("source.git_url"));
    }

    #[tokio::test]
    async fn a_data_path_that_is_not_a_checkout_is_a_clear_error() {
        let (_remote, _clone, mut config) = fixture();
        let plain = tempfile::tempdir().unwrap();
        config.source.data_path = Some(plain.path().to_string_lossy().into_owned());
        let err = pull_clone(&config, None).await.unwrap_err();
        assert_eq!(err.kind, SyncErrorKind::NotACheckout);
    }

    #[tokio::test]
    async fn a_failed_fetch_without_a_token_says_so() {
        let (_remote, _clone, mut config) = fixture();
        config.source.git_url = Some("/nonexistent/remote/repo.git".into());
        let err = pull_clone(&config, None).await.unwrap_err();
        assert_eq!(err.message, "Git fetch failed");
        assert!(err.detail.unwrap().contains("no git token"));
    }
}
