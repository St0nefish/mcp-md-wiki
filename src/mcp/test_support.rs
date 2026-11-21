use rmcp::ErrorData as McpError;

use crate::config::Granularity;
use crate::mcp::*;
/// Caller-facing rejections name no config keys, payload fields or indexing
/// mechanics — none of which a caller can act on.
pub(super) fn assert_no_internals_in_caller_error(msg: &str) {
    for internal in [
        "chunking.",
        "heading_metadata",
        "section_key",
        "heading_prefixes",
        "payload",
        "re-index",
        "index --full",
    ] {
        assert!(!msg.contains(internal), "{internal:?} leaked into: {msg}");
    }
}

/// A server built from `make_test_resolved_config`'s defaults but with
/// `chunking.heading_metadata` forced on — needed to reach `search_sections`'
/// explain/fields rejections, which sit BEHIND the heading_metadata check in
/// the handler and would otherwise always report the wrong error.
pub(super) fn schema_tool_server_with_heading_metadata(tmp: &tempfile::TempDir) -> KbSearchServer {
    let mut config = make_test_resolved_config(tmp.path());
    Arc::make_mut(&mut config).chunking.heading_metadata = true;
    make_write_test_server(tmp, &["**/*.md".to_string()], config)
}

/// A server built from `make_test_resolved_config`'s defaults but with
/// `search.granularities` restricted to exactly `granularities` (#286) —
/// needed to reach `resolve_search_granularity`'s
/// disabled-granularity rejection and default-fallback behavior through
/// the real `search` tool entry point, rather than only unit-testing the
/// pure resolver directly (see the tests above this section).
pub(super) fn schema_tool_server_with_granularities(
    tmp: &tempfile::TempDir,
    granularities: &[Granularity],
) -> KbSearchServer {
    let mut config = make_test_resolved_config(tmp.path());
    Arc::make_mut(&mut config).search.granularities = granularities.to_vec();
    make_write_test_server(tmp, &["**/*.md".to_string()], config)
}

/// A server whose state DB lives under the temp KB, so metadata-backed tools work.
pub(super) fn schema_tool_server(tmp: &tempfile::TempDir) -> KbSearchServer {
    let config = make_test_resolved_config(tmp.path());
    make_write_test_server(tmp, &["**/*.md".to_string()], config)
}

pub(super) async fn seed_document(
    server: &KbSearchServer,
    rel_path: &str,
    frontmatter: serde_json::Value,
) {
    let map = match frontmatter {
        serde_json::Value::Object(m) => m.into_iter().collect::<std::collections::HashMap<_, _>>(),
        _ => panic!("frontmatter fixture must be an object"),
    };
    server
        .state_db()
        .await
        .unwrap()
        .upsert_document_metadata(rel_path, &map, 100, "hash", 1)
        .await
        .unwrap();
}

pub(super) fn write_schema_file(tmp: &tempfile::TempDir, dir: &str, yaml: &str) {
    let target = tmp.path().join(dir);
    std::fs::create_dir_all(&target).unwrap();
    std::fs::write(target.join(crate::schema::SCHEMA_FILE_NAME), yaml).unwrap();
}

/// Build a bare-bones `KbSearchServer` for overlay tests — same
/// construction pattern as `get_document_rejects_overlong_path` below,
/// parameterized by the description overlay under test.
pub(super) fn make_overlay_test_server(overlay: HashMap<String, String>) -> KbSearchServer {
    let tmp = tempfile::tempdir().unwrap();
    make_overlay_test_server_with_config(overlay, make_test_resolved_config(tmp.path()))
}

/// Same as [`make_overlay_test_server`], but with a caller-supplied
/// config rather than `make_test_resolved_config`'s defaults — needed by
/// the schema-overlay tests below (#286), which exercise
/// `overlay_input_schema`/`get_tool`/`list_tools`'s equivalent against a
/// restricted `search.granularities` or `chunking.heading_metadata`.
pub(super) fn make_overlay_test_server_with_config(
    overlay: HashMap<String, String>,
    config: Arc<ResolvedConfig>,
) -> KbSearchServer {
    let tmp = tempfile::tempdir().unwrap();
    let instructions = Arc::new(RwLock::new("test".to_string()));
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
        query_prefix: String::new(),
        document_prefix: String::new(),
    };
    let embed = Arc::new(EmbedClient::new(&embed_config));
    // `KbSearchServer::new` canonicalizes `tmp`'s path during construction
    // but never touches the filesystem again afterward (neither does
    // anything these overlay tests exercise — `get_tool`/`list_tools` are
    // pure lock reads), so `tmp` can safely drop once construction
    // returns.
    let server = KbSearchServer::new(
        embed,
        qdrant,
        "test".into(),
        tmp.path().to_path_buf(),
        &["**/*.md".to_string()],
        instructions,
        crate::config::shared_config(config),
        empty_test_schema_cache(),
        None,
        Arc::new(crate::reindex::ReindexQueue::new()),
        Arc::new(RwLock::new(overlay)),
    )
    .unwrap();
    drop(tmp);
    server
}

/// A server whose live config disables a tool: `mcp.disabled_tools`.
pub(super) fn make_test_server_with_disabled_tools(disabled_tools: Vec<String>) -> KbSearchServer {
    let mut config = make_test_resolved_config(&std::env::temp_dir());
    Arc::make_mut(&mut config).mcp.disabled_tools = disabled_tools;
    make_overlay_test_server_with_config(HashMap::new(), config)
}

pub(super) fn overlay_test_config(
    granularities: &[Granularity],
    heading_metadata: bool,
) -> Arc<ResolvedConfig> {
    let mut config = make_test_resolved_config(&std::env::temp_dir());
    {
        let resolved = Arc::make_mut(&mut config);
        resolved.search.granularities = granularities.to_vec();
        resolved.chunking.heading_metadata = heading_metadata;
    }
    config
}

/// Run `git <args>` in `dir`, panicking on failure.
pub(super) fn run_git(dir: &std::path::Path, args: &[&str]) {
    let out = std::process::Command::new("git")
        .current_dir(dir)
        .args([
            "-c",
            "user.email=t@localhost",
            "-c",
            "user.name=Test",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .output()
        .unwrap();
    assert!(out.status.success(), "git {args:?}: {out:?}");
}

/// Build a KbSearchServer suitable for write_document unit tests.
/// Uses a temp directory as the data path. No real Qdrant/git required
/// for tests that fail before those steps.
pub(super) fn make_write_test_server(
    tmp: &tempfile::TempDir,
    include_patterns: &[String],
    config: Arc<ResolvedConfig>,
) -> KbSearchServer {
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
        query_prefix: String::new(),
        document_prefix: String::new(),
    };
    let embed = Arc::new(EmbedClient::new(&embed_config));
    let instructions = Arc::new(RwLock::new(String::new()));
    // Built from whatever `.kb-schema.yaml` files already exist under `tmp` at
    // this point — callers that need a test to see a schema written must write it
    // before calling this, exactly as they already do for `write_schema_file`.
    let canonical = tmp.path().canonicalize().unwrap();
    let schema_cache: crate::schema::SharedSchemaCache = Arc::new(RwLock::new(Arc::new(
        crate::schema::SchemaCache::build_for_test(&canonical, &config.frontmatter),
    )));
    KbSearchServer::new(
        embed,
        qdrant,
        "test".into(),
        tmp.path().to_path_buf(),
        include_patterns,
        instructions,
        crate::config::shared_config(config),
        schema_cache,
        None,
        Arc::new(crate::reindex::ReindexQueue::new()),
        empty_test_description_overlay(),
    )
    .unwrap()
}

/// Build a `KbSearchServer` backed by a real git working clone, so write tools get
/// past `commit_and_sync` and reach the point where they mark paths dirty.
///
/// `make_write_test_server` gives this server its own private `ReindexQueue`
/// (see that function), so the tests below read it back via
/// `server.reindex_queue` — no other test in the binary can have marked
/// anything on it.
pub(super) fn make_git_backed_server(
    work: &tempfile::TempDir,
) -> (KbSearchServer, Arc<crate::config::ResolvedConfig>) {
    let mut config = make_test_resolved_config(work.path());
    // Bypass the dedup gate: it would otherwise call out to a (nonexistent)
    // embedding service before we ever reach the commit.
    Arc::get_mut(&mut config).unwrap().write.dedup_enabled = false;
    let server = make_write_test_server(work, &["**/*.md".to_string()], Arc::clone(&config));
    (server, config)
}

/// Run git in `dir` as a peer writer, never escaping into an enclosing repo.
pub(super) fn peer_git(dir: &std::path::Path, args: &[&str]) {
    let out = std::process::Command::new("git")
        .args(["-c", "user.email=peer@test.com", "-c", "user.name=Peer"])
        .args(args)
        .current_dir(dir)
        .env("GIT_CEILING_DIRECTORIES", dir.parent().unwrap_or(dir))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A knowledge base clone the server writes through (`work`, syncing to
/// `bare`), seeded with `docs/target.md` (`target`), plus a second clone
/// (`peer`) of the same state for a concurrent writer.
pub(super) struct ConcurrentHarness {
    pub(super) _bare: tempfile::TempDir,
    pub(super) work: tempfile::TempDir,
    pub(super) peer: tempfile::TempDir,
    pub(super) server: KbSearchServer,
}

pub(super) const CONCURRENT_TARGET: &str = "---\ntitle: Old\n---\n\n# Body\n\none\ntwo\nthree\nfour\n\
                                 five\nsix\nseven\neight\nlast line\n";

pub(super) fn concurrent_harness() -> ConcurrentHarness {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    std::fs::create_dir_all(work.path().join("docs")).unwrap();
    std::fs::write(work.path().join("docs/target.md"), CONCURRENT_TARGET).unwrap();
    peer_git(work.path(), &["add", "docs/target.md"]);
    peer_git(work.path(), &["commit", "-q", "-m", "seed"]);
    peer_git(work.path(), &["push", "-q", "origin", "HEAD:master"]);
    let peer = crate::git::tests::clone_bare_repo(bare.path(), "master");

    let mut config = make_test_resolved_config(work.path());
    {
        let c = Arc::get_mut(&mut config).unwrap();
        c.write.dedup_enabled = false;
        c.source.git_url = Some(format!("file://{}", bare.path().display()));
    }
    let server = make_write_test_server(&work, &["**/*.md".to_string()], config);
    ConcurrentHarness {
        _bare: bare,
        work,
        peer,
        server,
    }
}

/// Run `fut` with a git test hook that runs `action` the first time the write
/// reaches `point` (a nested write inside `action` reaches it too, and is left
/// alone, since the action is already taken).
pub(super) async fn with_hook_once<A, AF, T>(
    point: crate::git::HookPoint,
    action: A,
    fut: impl std::future::Future<Output = T>,
) -> T
where
    A: FnOnce() -> AF + Send + 'static,
    AF: std::future::Future<Output = ()> + Send + 'static,
{
    let action = Arc::new(std::sync::Mutex::new(Some(action)));
    let hook: crate::git::TestHook = Arc::new(move |p| {
        let action = Arc::clone(&action);
        Box::pin(async move {
            if p != point {
                return;
            }
            let taken = action.lock().unwrap().take();
            if let Some(action) = taken {
                action().await;
            }
        })
    });
    crate::git::TEST_HOOK.scope(hook, fut).await
}

/// Like [`with_hook_once`], but runs `action(n)` every time (`n` counts from 1).
pub(super) async fn with_hook_each<A, T>(
    point: crate::git::HookPoint,
    action: A,
    fut: impl std::future::Future<Output = T>,
) -> T
where
    A: Fn(usize) + Send + Sync + 'static,
{
    let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let action = Arc::new(action);
    let hook: crate::git::TestHook = Arc::new(move |p| {
        let count = Arc::clone(&count);
        let action = Arc::clone(&action);
        Box::pin(async move {
            if p == point {
                let n = count.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                action(n);
            }
        })
    });
    crate::git::TEST_HOOK.scope(hook, fut).await
}

/// Commit `content` to `rel` in the peer clone and push it.
pub(super) fn peer_push(peer: &std::path::Path, rel: &str, content: &str) {
    if let Some(parent) = std::path::Path::new(rel).parent() {
        std::fs::create_dir_all(peer.join(parent)).unwrap();
    }
    peer_git(peer, &["pull", "-q", "--rebase", "origin", "master"]);
    std::fs::write(peer.join(rel), content).unwrap();
    peer_git(peer, &["add", rel]);
    peer_git(peer, &["commit", "-q", "-m", &format!("peer writes {rel}")]);
    peer_git(peer, &["push", "-q", "origin", "HEAD:master"]);
}

pub(super) fn rev(dir: &std::path::Path, rev: &str) -> String {
    let out = std::process::Command::new("git")
        .args(["rev-parse", rev])
        .current_dir(dir)
        .env("GIT_CEILING_DIRECTORIES", dir.parent().unwrap_or(dir))
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

pub(super) fn target_version(h: &ConcurrentHarness) -> String {
    write::document_version(&std::fs::read(h.work.path().join("docs/target.md")).unwrap())
}

/// `HEAD` of `work`, as a trimmed hex string.
/// The current version of `rel` under `dir`, as `get_document` would report
/// it; `None` when it does not exist.
pub(super) fn version_at(dir: &tempfile::TempDir, rel: &str) -> Option<String> {
    std::fs::read(dir.path().join(rel))
        .ok()
        .map(|bytes| write::document_version(&bytes))
}

pub(super) fn head_sha(work: &tempfile::TempDir) -> String {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(work.path())
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// `git status --porcelain` of `work`, as a trimmed string ("" means clean).
pub(super) fn git_status(work: &tempfile::TempDir) -> String {
    let out = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(work.path())
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

pub(super) fn git_commit_all(work: &tempfile::TempDir, rel_path: &str, message: &str) {
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

/// Force `git commit` to fail in `work` while leaving the repo otherwise
/// completely healthy: `git add` still succeeds, HEAD is still valid, and a
/// restore from HEAD should still succeed. This is `CommitSyncError::PreCommit`
/// with the failure specifically at the `commit` step rather than `add`.
///
/// Deliberately NOT a `.git/hooks/pre-commit` script: a machine (or CI image)
/// with a global `core.hooksPath` override — common for commit-signing or
/// linting setups — would silently ignore a repo-local hook, making that
/// approach environment-dependent. Repo-local git CONFIG, by contrast, always
/// applies: enabling commit signing and pointing at a signing key that does not
/// exist makes `git commit` fail deterministically on any machine, with or
/// without gpg installed (no gpg binary fails it too, just with a different
/// message), and without touching global state.
pub(super) fn force_git_commit_to_fail(work: &tempfile::TempDir) {
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

/// Words that name how the server versions the knowledge base. None of them
/// may reach an MCP tool caller — results, errors or descriptions. Matched
/// against whole lowercase alphanumeric tokens (`commit` also as a prefix, for
/// `commits`/`committed`), so `github` or a word like `pushes` in a document's
/// own diff body is not a false positive unless it really is the token.
pub(super) fn git_terms_in(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| {
            matches!(
                *t,
                "git" | "sha" | "push" | "pushed" | "rebase" | "rebased" | "remote"
            ) || t.starts_with("commit")
        })
        .map(str::to_string)
        .collect()
}

pub(super) fn assert_git_free(what: &str, text: &str) {
    let hits = git_terms_in(text);
    assert!(hits.is_empty(), "{what} leaks {hits:?}: {text}");
}

/// A plain success: none of the removed versioning fields, and git-free text.
pub(super) fn assert_plain_success(result: &CallToolResult) {
    let structured = result
        .structured_content
        .as_ref()
        .expect("a write result carries structured_content");
    for key in ["outcome", "sha", "rebased_paths", "sync_failure_cause"] {
        assert!(structured.get(key).is_none(), "{key} leaked: {structured}");
    }
    assert_git_free("structured_content", &structured.to_string());
    assert_git_free("result text", &format!("{:?}", result.content));
}

/// A rolled-back write: "not saved, nothing changed, try again", no data
/// payload, nothing about the cause.
pub(super) fn assert_not_saved(err: &McpError) {
    assert!(
        err.message.contains("could not be saved") && err.message.contains("try again"),
        "{}",
        err.message
    );
    assert!(err.data.is_none(), "{err:?}");
    assert_git_free("error", &err.message);
}

/// A write whose rollback also failed: told not to trust the state.
pub(super) fn assert_not_saved_unverified(err: &McpError) {
    assert!(
        err.message.contains("could not be saved")
            && err.message.contains("could not confirm")
            && err.message.contains("operator"),
        "{}",
        err.message
    );
    assert!(err.data.is_none(), "{err:?}");
    assert_git_free("error", &err.message);
}

/// `git_status`, filtered to drop the SQLite state-DB files (`state.db` plus its
/// `-shm`/`-wal` siblings). `update_schema`'s casualty check
/// (`documents_broken_by`) opens the metadata index on first use, which lazily
/// creates those files under `work` as an ordinary, expected side effect of
/// running the tool at all — unrelated to whether a `commit_and_sync` rollback
/// left the WRITE ITSELF clean. `delete_document` does the same (#229: it always
/// opens the state DB for the inbound-link check, unlike a MOVE's conditional
/// open), so its tests need this filtered helper too. `create_document`/plain
/// `edit_document` (no `new_path`) never touch the metadata index at all, so
/// their equivalent tests can keep the plain, exact `git_status` assertion.
pub(super) fn git_status_ignoring_state_db(work: &tempfile::TempDir) -> String {
    git_status(work)
        .lines()
        .filter(|line| {
            let path = line.split_whitespace().last().unwrap_or("");
            !path.starts_with("state.db")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn commit_count(work: &tempfile::TempDir) -> usize {
    let out = std::process::Command::new("git")
        .args(["rev-list", "--count", "HEAD"])
        .current_dir(work.path())
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .unwrap_or(0)
}

pub(super) fn head_subject(work: &tempfile::TempDir) -> String {
    let out = std::process::Command::new("git")
        .args(["log", "-1", "--format=%s"])
        .current_dir(work.path())
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}
/// A git-backed server with the memory tool group on (`memory.enabled`), its
/// built-in schema laid into the schema cache the way startup does. The duplicate
/// check is bypassed by the tests themselves (`on_duplicate: keep_both`), since no
/// embedding service runs here.
pub(super) struct MemoryHarness {
    pub(super) _bare: tempfile::TempDir,
    pub(super) work: tempfile::TempDir,
    pub(super) server: KbSearchServer,
}

pub(super) fn memory_harness() -> MemoryHarness {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let mut config = make_test_resolved_config(work.path());
    {
        let c = Arc::get_mut(&mut config).unwrap();
        c.write.dedup_enabled = false;
        c.source.git_url = Some(format!("file://{}", bare.path().display()));
        c.memory.enabled = true;
    }
    let canonical = work.path().canonicalize().unwrap();
    let schema_cache: crate::schema::SharedSchemaCache = Arc::new(RwLock::new(Arc::new(
        crate::schema::SchemaCache::build_with(
            &canonical,
            &config.frontmatter,
            &config.indexing,
            &crate::memory::builtin_scopes(&config.memory),
        )
        .unwrap(),
    )));
    let qdrant = Arc::new(
        QdrantStore::new(&crate::config::ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: "test".into(),
        })
        .unwrap(),
    );
    let embed = Arc::new(EmbedClient::new(&crate::config::ResolvedEmbeddingConfig {
        base_url: "http://localhost:8080/v1".into(),
        model: "test".into(),
        api_key: None,
        vector_size: 768,
        batch_size: 32,
        request_timeout_secs: 60,
        batch_concurrency: 4,
        query_prefix: String::new(),
        document_prefix: String::new(),
    }));
    let server = KbSearchServer::new(
        embed,
        qdrant,
        "test".into(),
        work.path().to_path_buf(),
        &["**/*.md".to_string()],
        Arc::new(RwLock::new(String::new())),
        crate::config::shared_config(config),
        schema_cache,
        None,
        Arc::new(crate::reindex::ReindexQueue::new()),
        empty_test_description_overlay(),
    )
    .unwrap();
    MemoryHarness {
        _bare: bare,
        work,
        server,
    }
}
