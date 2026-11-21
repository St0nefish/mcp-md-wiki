use rmcp::handler::server::wrapper::Parameters;

use super::get_document::single_representation;
use super::schema::add_values_params;
use crate::mcp::get_document::*;
use crate::mcp::schema_tools::*;
use crate::mcp::test_support::*;
use crate::mcp::write_document::*;
use crate::mcp::*;
use crate::similarity::{DEDUP_QUERY_CHAR_LIMIT, build_dedup_query, dedup_search_opts};
use crate::write::FrontmatterEdit;
use crate::write::{
    self, DirectoryMoveError, WriteError, WriteSuccess, build_commit_message, dedup_verdict,
    render_unified_diff,
};
use std::path::Path;

// --- list_documents filter parsing ---

#[test]
fn commit_message_create_document_trailer() {
    let msg = build_commit_message(None, "docs: add notes/guide.md", "create_document");
    assert!(
        msg.contains("Tool: mcp-md-wiki"),
        "should contain Tool trailer: {msg}"
    );
    assert!(
        msg.contains("Operation: create_document"),
        "should contain Operation trailer: {msg}"
    );
    assert!(
        msg.starts_with("docs: add notes/guide.md"),
        "should start with default subject: {msg}"
    );
}

#[test]
fn commit_message_edit_surgical_trailer() {
    let msg = build_commit_message(
        None,
        "docs: update notes/guide.md",
        "edit_document (surgical replace)",
    );
    assert!(
        msg.contains("Operation: edit_document (surgical replace)"),
        "should contain surgical operation label: {msg}"
    );
}

#[test]
fn commit_message_edit_full_replace_trailer() {
    let msg = build_commit_message(
        None,
        "docs: update notes/guide.md",
        "edit_document (full replace)",
    );
    assert!(
        msg.contains("Operation: edit_document (full replace)"),
        "should contain full replace operation label: {msg}"
    );
}

#[test]
fn commit_message_user_subject_overrides_default() {
    let msg = build_commit_message(
        Some("fix: correct typo in introduction"),
        "docs: update notes/guide.md",
        "edit_document (surgical replace)",
    );
    assert!(
        msg.starts_with("fix: correct typo in introduction"),
        "user subject should take precedence: {msg}"
    );
    assert!(
        msg.contains("Operation: edit_document (surgical replace)"),
        "trailer should still be appended: {msg}"
    );
}

#[test]
fn commit_message_trailer_separated_by_blank_line() {
    let msg = build_commit_message(None, "docs: add test.md", "create_document");
    // Git requires a blank line between subject and trailer block
    assert!(
        msg.contains("\n\nTool: mcp-md-wiki"),
        "blank line must precede trailer block: {msg}"
    );
}

#[tokio::test]
async fn write_document_on_nonexistent_path_upserts_as_a_create() {
    // The merged tool UPSERTS: a `content` write against a path that does not
    // exist creates it (as `create_document` used to), rather than the old
    // standalone `edit_document`'s "does not exist" refusal.
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let (server, _config) = make_git_backed_server(&work);

    let params = WriteDocumentParams {
        path: Some("docs/nonexistent.md".to_string()),
        old_string: None,
        new_string: None,
        content: Some("---\ntitle: Test\n---\n# Body".to_string()),
        message: None,
        expected_version: None,
        new_path: None,
        force_new: Some(true),
        frontmatter_patch: None,
        append: None,
        documents: None,
        archive: None,
        archive_links: None,
    };
    let result = server.write_document(Parameters(params)).await;

    let result = result.expect("a write against a nonexistent path must create it");
    let structured = single_representation(&result);
    assert_eq!(structured["action"], "created", "{structured}");
    assert_eq!(structured["path"], "docs/nonexistent.md");
}

#[tokio::test]
async fn write_document_on_existing_path_upserts_as_an_edit() {
    // The merged write_document tool UPSERTS: calling it with `content` against
    // a path that already exists no longer refuses (as `create_document` used
    // to) — it replaces the file, the same as an explicit edit would.
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    std::fs::create_dir_all(work.path().join("docs")).unwrap();
    std::fs::write(
        work.path().join("docs/existing.md"),
        "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n",
    )
    .unwrap();
    git_commit_all(&work, "docs/existing.md", "add docs/existing.md");
    let (server, _config) = make_git_backed_server(&work);

    let params = WriteDocumentParams {
        path: Some("docs/existing.md".to_string()),
        content: Some(
            "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New body\n"
                .to_string(),
        ),
        old_string: None,
        new_string: None,
        new_path: None,
        message: None,
        expected_version: version_at(&work, "docs/existing.md"),
        force_new: None,
        frontmatter_patch: None,
        append: None,
        documents: None,
        archive: None,
        archive_links: None,
    };
    let result = server.write_document(Parameters(params)).await;

    let result = result.expect("write_document must upsert rather than refuse an existing path");
    let structured = single_representation(&result);
    assert_eq!(
        structured["action"], "updated",
        "an upsert onto an existing path is an EDIT, not a create: {structured}"
    );
    assert_eq!(structured["path"], "docs/existing.md");
    assert_eq!(
        std::fs::read_to_string(work.path().join("docs/existing.md")).unwrap(),
        "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New body\n",
    );
}

#[tokio::test]
async fn write_document_frontmatter_patch_end_to_end() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    std::fs::create_dir_all(work.path().join("docs")).unwrap();
    std::fs::write(
        work.path().join("docs/log.md"),
        "---\ntitle: Log\nstatus: draft\n---\n\n# Body\n",
    )
    .unwrap();
    git_commit_all(&work, "docs/log.md", "add docs/log.md");
    let (server, _config) = make_git_backed_server(&work);

    let params = WriteDocumentParams {
        path: Some("docs/log.md".to_string()),
        content: None,
        old_string: None,
        new_string: None,
        new_path: None,
        message: None,
        expected_version: None,
        force_new: None,
        frontmatter_patch: Some(vec![FrontmatterPatchOp {
            operation: "set_field".to_string(),
            field: "status".to_string(),
            value: Some(serde_json::json!("active")),
            values: None,
        }]),
        append: None,
        documents: None,
        archive: None,
        archive_links: None,
    };
    let result = server.write_document(Parameters(params)).await;
    let result = result.expect("frontmatter_patch must succeed against an existing document");
    let structured = single_representation(&result);
    assert_eq!(structured["action"], "updated", "{structured}");
    assert!(
        structured["diff"]
            .as_str()
            .is_some_and(|d| d.contains("+status: active")),
        "an edit carries its diff: {structured}"
    );

    let on_disk = std::fs::read_to_string(work.path().join("docs/log.md")).unwrap();
    assert!(on_disk.contains("status: active"), "got: {on_disk}");
    assert!(on_disk.contains("title: Log"), "title must be preserved");
    assert!(
        on_disk.ends_with("# Body\n"),
        "body must be untouched: {on_disk}"
    );
}

#[tokio::test]
async fn write_document_append_end_to_end() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    std::fs::create_dir_all(work.path().join("docs")).unwrap();
    std::fs::write(
        work.path().join("docs/log.md"),
        "---\ntitle: Log\n---\n\n# Log\n- entry one\n",
    )
    .unwrap();
    git_commit_all(&work, "docs/log.md", "add docs/log.md");
    let (server, _config) = make_git_backed_server(&work);

    let params = WriteDocumentParams {
        path: Some("docs/log.md".to_string()),
        content: None,
        old_string: None,
        new_string: None,
        new_path: None,
        message: None,
        expected_version: None,
        force_new: None,
        frontmatter_patch: None,
        append: Some("- entry two".to_string()),
        documents: None,
        archive: None,
        archive_links: None,
    };
    let result = server.write_document(Parameters(params)).await;
    result.expect("append must succeed against an existing document");

    assert_eq!(
        std::fs::read_to_string(work.path().join("docs/log.md")).unwrap(),
        "---\ntitle: Log\n---\n\n# Log\n- entry one\n- entry two\n"
    );
}

#[tokio::test]
async fn write_document_frontmatter_patch_and_append_combine_end_to_end() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    std::fs::create_dir_all(work.path().join("docs")).unwrap();
    std::fs::write(
        work.path().join("docs/log.md"),
        "---\ntitle: Log\nstatus: draft\n---\n\n# Log\n- entry one\n",
    )
    .unwrap();
    git_commit_all(&work, "docs/log.md", "add docs/log.md");
    let (server, _config) = make_git_backed_server(&work);

    let params = WriteDocumentParams {
        path: Some("docs/log.md".to_string()),
        content: None,
        old_string: None,
        new_string: None,
        new_path: None,
        message: None,
        expected_version: None,
        force_new: None,
        frontmatter_patch: Some(vec![FrontmatterPatchOp {
            operation: "set_field".to_string(),
            field: "status".to_string(),
            value: Some(serde_json::json!("active")),
            values: None,
        }]),
        append: Some("- entry two".to_string()),
        documents: None,
        archive: None,
        archive_links: None,
    };
    let result = server.write_document(Parameters(params)).await;
    result.expect("frontmatter_patch + append must succeed together");

    let on_disk = std::fs::read_to_string(work.path().join("docs/log.md")).unwrap();
    assert!(on_disk.contains("status: active"), "got: {on_disk}");
    assert!(
        on_disk.ends_with("- entry one\n- entry two\n"),
        "got: {on_disk}"
    );
}

#[tokio::test]
async fn write_document_frontmatter_patch_validation_failure_reports_error_and_writes_nothing() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    std::fs::write(
        work.path().join("notes.md"),
        "---\ntitle: T\n---\n\n# Body\n",
    )
    .unwrap();
    git_commit_all(&work, "notes.md", "add notes.md");

    let mut config = make_test_resolved_config(work.path());
    Arc::get_mut(&mut config).unwrap().write.dedup_enabled = false;
    Arc::get_mut(&mut config).unwrap().frontmatter.required = vec!["title".into()];
    let server = make_write_test_server(&work, &["**/*.md".to_string()], config);

    // The patch removes the very field the schema requires.
    let params = WriteDocumentParams {
        path: Some("notes.md".to_string()),
        content: None,
        old_string: None,
        new_string: None,
        new_path: None,
        message: None,
        expected_version: None,
        force_new: None,
        frontmatter_patch: Some(vec![FrontmatterPatchOp {
            operation: "remove_field".to_string(),
            field: "title".to_string(),
            value: None,
            values: None,
        }]),
        append: None,
        documents: None,
        archive: None,
        archive_links: None,
    };
    let result = server.write_document(Parameters(params)).await;
    let err = result.expect_err("a schema-violating patch must fail validation");
    assert!(err.message.contains("validation"), "got: {}", err.message);
    assert_eq!(
        std::fs::read_to_string(work.path().join("notes.md")).unwrap(),
        "---\ntitle: T\n---\n\n# Body\n",
        "a rejected patch must never touch the file on disk"
    );
}

#[tokio::test]
async fn write_document_frontmatter_patch_on_a_nonexistent_document_is_rejected() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let (server, _config) = make_git_backed_server(&work);

    let params = WriteDocumentParams {
        path: Some("does-not-exist.md".to_string()),
        content: None,
        old_string: None,
        new_string: None,
        new_path: None,
        message: None,
        expected_version: None,
        force_new: None,
        frontmatter_patch: Some(vec![FrontmatterPatchOp {
            operation: "set_field".to_string(),
            field: "status".to_string(),
            value: Some(serde_json::json!("active")),
            values: None,
        }]),
        append: None,
        documents: None,
        archive: None,
        archive_links: None,
    };
    let err = server
        .write_document(Parameters(params))
        .await
        .expect_err("cannot patch frontmatter on a document that does not exist");
    assert!(
        err.message.contains("does not exist"),
        "got: {}",
        err.message
    );
}

#[tokio::test]
async fn write_document_append_on_a_nonexistent_document_is_rejected() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let (server, _config) = make_git_backed_server(&work);

    let params = WriteDocumentParams {
        path: Some("does-not-exist.md".to_string()),
        content: None,
        old_string: None,
        new_string: None,
        new_path: None,
        message: None,
        expected_version: None,
        force_new: None,
        frontmatter_patch: None,
        append: Some("text".to_string()),
        documents: None,
        archive: None,
        archive_links: None,
    };
    let err = server
        .write_document(Parameters(params))
        .await
        .expect_err("cannot append to a document that does not exist");
    assert!(
        err.message.contains("does not exist"),
        "got: {}",
        err.message
    );
}

#[tokio::test]
async fn write_document_on_existing_but_not_permitted_file_returns_include_pattern_error() {
    // G3 regression, preserved across the merge: a path that BOTH exists on
    // disk AND fails the include-pattern check must be reported with the
    // include-pattern message, not "already exists" — the latter would be
    // misleading circular guidance, since a retry would reject the same path
    // as not permitted right back.
    let tmp = tempfile::tempdir().unwrap();
    let sub = tmp.path().join("docs");
    std::fs::create_dir_all(&sub).unwrap();
    // Exists on disk as a `.md` file, but the server below only permits
    // `.txt` files — so it both exists AND fails the include-pattern check.
    std::fs::write(sub.join("existing.md"), "# Already here").unwrap();

    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.txt".to_string()], config);

    let params = WriteDocumentParams {
        path: Some("docs/existing.md".to_string()),
        content: Some("---\ntitle: Test\n---\n# New content".to_string()),
        old_string: None,
        new_string: None,
        new_path: None,
        message: None,
        expected_version: None,
        force_new: None,
        frontmatter_patch: None,
        append: None,
        documents: None,
        archive: None,
        archive_links: None,
    };
    let result = server.write_document(Parameters(params)).await;

    assert!(result.is_err(), "write should be rejected");
    let err = result.unwrap_err();
    assert!(
        err.message.contains("indexable include pattern"),
        "error should report the include-pattern rejection, got: {}",
        err.message
    );
    assert!(
        !err.message.contains("already exists"),
        "error should not fall back to the misleading 'already exists' message, got: {}",
        err.message
    );
}

#[test]
fn already_exists_race_error_reports_the_absolute_path() {
    // `write_document`'s own create-path pre-check (`abs_path.exists()`)
    // catches the ordinary "already exists" case before ever reaching
    // `write.rs` — see `write_document_on_existing_path_upserts_as_an_edit`
    // above (which never reaches this arm at all, since resolving to an
    // existing file routes to the edit path instead of the create path). The
    // `WriteError::AlreadyExists` arm this test drives is only reachable via a
    // genuine TOCTOU race (the file appears between that pre-check and
    // `write::write_document`'s `create_new` open), which pre-`write.rs`-
    // extraction code reported with the absolute filesystem path rather than
    // the repo-relative one — restore that wording (N2).
    let tmp = tempfile::tempdir().unwrap();
    let canonical_data_path = tmp.path().canonicalize().unwrap();
    let rel_path = "docs/existing.md";

    let err = create_edit_error_to_mcp_error(
        WriteError::AlreadyExists,
        rel_path,
        true,
        &canonical_data_path,
        None,
    );

    let expected_abs = canonical_data_path.join(rel_path);
    assert!(
        err.message.contains(&expected_abs.display().to_string()),
        "expected the absolute path '{}' in the message, got: {}",
        expected_abs.display(),
        err.message
    );
    assert!(
        err.message.contains("write_document"),
        "error should still mention write_document, got: {}",
        err.message
    );
}

#[test]
fn internal_write_error_reaches_mcp_callers_with_the_same_text_as_before() {
    // G2: `WriteError::Internal` exists so `web.rs` can hide a canonicalize
    // failure's embedded absolute path from an untrusted caller. MCP is a
    // trusted surface that was already relaying this exact text via
    // `WriteError::UnsafePath` before that split — both adapters must keep
    // doing so, verbatim, for `Internal` too.
    let msg = "Invalid path: cannot canonicalize data root '/data/kb': \
                No such file or directory (os error 2)"
        .to_string();
    let tmp = tempfile::tempdir().unwrap();
    let canonical_data_path = tmp.path().canonicalize().unwrap();

    let create_err = create_edit_error_to_mcp_error(
        WriteError::Internal { msg: msg.clone() },
        "docs/x.md",
        true,
        &canonical_data_path,
        None,
    );
    assert_eq!(create_err.message, msg);

    let delete_err =
        delete_error_to_mcp_error(WriteError::Internal { msg: msg.clone() }, "docs/x.md");
    assert_eq!(delete_err.message, msg);
}

#[tokio::test]
async fn include_pattern_guard_rejects_non_matching_path() {
    let tmp = tempfile::tempdir().unwrap();
    let config = make_test_resolved_config(tmp.path());
    // Only markdown files are indexed
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    // Try to write a .txt file (not matched by **/*.md)
    let params = WriteDocumentParams {
        path: Some("notes.txt".to_string()),
        content: Some("Some plain text".to_string()),
        old_string: None,
        new_string: None,
        new_path: None,
        message: None,
        expected_version: None,
        force_new: None,
        frontmatter_patch: None,
        append: None,
        documents: None,
        archive: None,
        archive_links: None,
    };
    let result = server.write_document(Parameters(params)).await;

    assert!(
        result.is_err(),
        "non-matching path should be rejected by include-pattern guard"
    );
    let err = result.unwrap_err();
    assert!(
        err.message.contains("indexable include pattern"),
        "error should mention include pattern, got: {}",
        err.message
    );
}

#[tokio::test]
async fn include_pattern_guard_rejects_absolute_path() {
    let tmp = tempfile::tempdir().unwrap();
    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    // Absolute path should be caught by a guard before any write happens.
    let params = WriteDocumentParams {
        path: Some("/etc/passwd".to_string()),
        content: Some("# Evil".to_string()),
        old_string: None,
        new_string: None,
        new_path: None,
        message: None,
        expected_version: None,
        force_new: None,
        frontmatter_patch: None,
        append: None,
        documents: None,
        archive: None,
        archive_links: None,
    };
    let result = server.write_document(Parameters(params)).await;

    // A leading `/` always means the KB root, so `/etc/passwd` names
    // `<kb>/etc/passwd` — the host file is never probed — and that path is
    // refused by create's include-pattern check (not a markdown file).
    assert!(
        result.is_err(),
        "a non-markdown absolute path must be rejected"
    );
    let err = result.unwrap_err();
    assert!(
        err.message.contains("indexable include pattern"),
        "error should cite the include-pattern guard, got: {}",
        err.message
    );
}

#[tokio::test]
async fn write_document_on_absolute_host_path_never_touches_the_host_file() {
    // A leading `/` always means the KB root, for reads and writes alike, so an
    // absolute path naming a real host file outside the KB addresses
    // `<kb>/<that path>` — the host file is never read, probed or written.
    let tmp = tempfile::tempdir().unwrap();
    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    // A real, existing .md file OUTSIDE the KB data root.
    let outside = tempfile::tempdir().unwrap();
    let outside_file = outside.path().join("real.md");
    std::fs::write(&outside_file, "# Not part of the KB").unwrap();

    let params = WriteDocumentParams {
        path: Some(outside_file.to_str().unwrap().to_string()),
        content: Some("# Evil overwrite".to_string()),
        old_string: None,
        new_string: None,
        new_path: None,
        message: None,
        expected_version: None,
        force_new: None,
        frontmatter_patch: None,
        append: None,
        documents: None,
        archive: None,
        archive_links: None,
    };
    // Whatever the KB-relative create does (this test KB has no git remote,
    // so its commit fails and is rolled back), it must not reach the host.
    let _ = server.write_document(Parameters(params)).await;

    assert_eq!(
        std::fs::read_to_string(&outside_file).unwrap(),
        "# Not part of the KB",
        "the real file outside the KB must not be overwritten"
    );
}

#[tokio::test]
async fn validation_failure_carries_field_errors_in_data() {
    let tmp = tempfile::tempdir().unwrap();

    // Config with validation enabled requiring "title" field
    let config = Arc::new(ResolvedConfig {
        source: crate::config::ResolvedSourceConfig {
            git_url: None,
            branch: "master".into(),
            data_path: Some(tmp.path().to_string_lossy().into_owned()),
            git_token_env: "GIT_PULL_TOKEN".into(),
        },
        indexing: crate::config::IndexingConfig::default(),
        frontmatter: crate::config::FrontmatterConfig {
            required: vec!["title".into()],
            ..Default::default()
        },
        chunking: crate::config::ChunkingConfig::default(),
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
            collection: "test".into(),
        },
        validation: crate::config::ValidationConfig {
            enabled: true,
            strict: false,
            lint_command: None,
            ..Default::default()
        },
        webhook: crate::config::WebhookConfig::default(),
        mcp: crate::config::ResolvedMcpConfig::default(),
        rate_limit: crate::config::RateLimitConfig::default(),
        write: crate::config::WriteConfig::default(),
        search: crate::config::SearchConfig::default(),
        reranking: None,
        ui: crate::config::UiConfig::default(),
        lint: crate::config::LintConfig::default(),
        formatting: crate::config::FormattingConfig::off(),
        contradictions: crate::config::ContradictionsConfig::default(),
        contradictions_judge: None,
        provenance: Default::default(),
    });

    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    // Content intentionally missing the "title" frontmatter field
    let params = WriteDocumentParams {
        path: Some("guide/missing-title.md".to_string()),
        content: Some("---\ntype: guide\n---\n# No title in frontmatter".to_string()),
        old_string: None,
        new_string: None,
        new_path: None,
        message: None,
        expected_version: None,
        force_new: None,
        frontmatter_patch: None,
        append: None,
        documents: None,
        archive: None,
        archive_links: None,
    };
    let result = server.write_document(Parameters(params)).await;

    assert!(result.is_err(), "validation failure should return Err");
    let err = result.unwrap_err();

    // Message should be human-readable
    assert!(
        err.message.contains("frontmatter validation failed"),
        "error message should describe validation failure, got: {}",
        err.message
    );

    // Data field must contain structured field_errors
    let data = err.data.expect("error should carry structured data");
    let field_errors = data
        .get("field_errors")
        .expect("data must have field_errors key");
    assert!(
        field_errors.is_array(),
        "field_errors must be a JSON array, got: {}",
        field_errors
    );
    let arr = field_errors.as_array().unwrap();
    assert!(
        !arr.is_empty(),
        "field_errors array must be non-empty for a validation failure"
    );

    // At least one entry should mention "title" as the failed field
    let mentions_title = arr.iter().any(|fe| {
        fe.get("field")
            .and_then(|f| f.as_str())
            .map(|f| f == "title")
            .unwrap_or(false)
    });
    assert!(
        mentions_title,
        "field_errors should contain an entry for 'title', got: {}",
        serde_json::to_string_pretty(&data).unwrap()
    );
}

// -----------------------------------------------------------------------
// dedup_verdict unit tests — no live Qdrant/embedder required
// -----------------------------------------------------------------------

#[test]
fn dedup_verdict_score_above_threshold_returns_hit() {
    let result = dedup_verdict(Some(("docs/existing.md".into(), 0.92)), 0.85);
    assert!(
        result.is_some(),
        "score 0.92 >= threshold 0.85 should refuse"
    );
    let hit = result.unwrap();
    assert_eq!(hit.file_path, "docs/existing.md");
    assert!((hit.score - 0.92).abs() < 1e-6);
}

#[test]
fn dedup_verdict_score_at_threshold_returns_hit() {
    // Boundary: score exactly equal to threshold is also a duplicate.
    let result = dedup_verdict(Some(("docs/boundary.md".into(), 0.85)), 0.85);
    assert!(
        result.is_some(),
        "score == threshold should be treated as duplicate"
    );
}

#[test]
fn dedup_verdict_score_below_threshold_allows() {
    let result = dedup_verdict(Some(("docs/different.md".into(), 0.70)), 0.85);
    assert!(
        result.is_none(),
        "score 0.70 < threshold 0.85 should allow the write"
    );
}

#[test]
fn dedup_verdict_no_results_allows() {
    // Empty collection or no results → no duplicate → allow.
    let result = dedup_verdict(None, 0.85);
    assert!(
        result.is_none(),
        "no results should allow the write (empty collection case)"
    );
}

#[test]
fn dedup_verdict_hit_carries_correct_fields() {
    let hit = dedup_verdict(Some(("sysadmin/networking/dns.md".into(), 0.95)), 0.85)
        .expect("should be a hit");
    assert_eq!(hit.file_path, "sysadmin/networking/dns.md");
    assert!((hit.score - 0.95).abs() < 1e-6, "score should be preserved");
}

// -----------------------------------------------------------------------
// dedup query construction + search options — pure, no live services
// -----------------------------------------------------------------------

#[test]
fn build_dedup_query_prepends_description() {
    let q = build_dedup_query("Body text here.", Some("A short summary."), true);
    assert_eq!(q, "A short summary.\n\nBody text here.");
}

#[test]
fn build_dedup_query_omits_description_when_disabled() {
    let q = build_dedup_query("Body text here.", Some("A short summary."), false);
    assert_eq!(
        q, "Body text here.",
        "prepend_description=false must match the indexer, which also omits it"
    );
}

#[test]
fn build_dedup_query_omits_description_when_absent() {
    let q = build_dedup_query("Body text here.", None, true);
    assert_eq!(q, "Body text here.");
}

#[test]
fn build_dedup_query_truncates_to_limit() {
    let long_body = "x".repeat(DEDUP_QUERY_CHAR_LIMIT * 2);
    let q = build_dedup_query(&long_body, None, false);
    assert_eq!(q.chars().count(), DEDUP_QUERY_CHAR_LIMIT);
}

#[test]
fn build_dedup_query_truncation_counts_chars_not_bytes() {
    // Multi-byte input must not panic or split a character.
    let long_body = "é".repeat(DEDUP_QUERY_CHAR_LIMIT * 2);
    let q = build_dedup_query(&long_body, None, false);
    assert_eq!(q.chars().count(), DEDUP_QUERY_CHAR_LIMIT);
}

/// The dedup query must be built on the same textual basis the indexer
/// embeds, otherwise the gate scores a query against candidates that were
/// assembled differently. Pin the two together so they cannot drift.
#[test]
fn build_dedup_query_matches_chunk_prepend_format() {
    let body = "## Heading\n\nSome body content.";
    let description = "A short summary.";
    let chunking = crate::config::ChunkingConfig::default();
    assert!(
        chunking.prepend_description,
        "this test assumes the indexer default prepends description"
    );

    let chunks = crate::chunk::chunk_markdown(body, Some(description), &chunking);
    let first_chunk = &chunks.first().expect("body should produce a chunk").text;
    let query = build_dedup_query(body, Some(description), chunking.prepend_description);

    assert!(
        first_chunk.starts_with(&format!("{}\n\n", description)),
        "indexed chunk should carry the description prefix, got: {:?}",
        first_chunk
    );
    assert_eq!(
        query, *first_chunk,
        "dedup query and indexed chunk text must share one textual basis"
    );
}

/// Regression guard for issue #67: `write.dedup_threshold` is a cosine
/// similarity, so the dedup search must never inherit `search.hybrid`
/// (RRF scores top out near 0.03 and would make the gate unable to fire).
#[test]
fn dedup_search_opts_is_dense_only() {
    let opts = dedup_search_opts();
    assert!(
        !opts.hybrid,
        "dedup must be dense-only so its score is a cosine similarity"
    );
    assert!(
        crate::config::SearchConfig::default().hybrid,
        "search.hybrid defaults to true — this is exactly what dedup must not inherit"
    );
    assert_eq!(opts.limit, 1, "dedup only needs the nearest neighbour");
    assert!(
        opts.min_score.is_none(),
        "thresholding is dedup_verdict's job, not the search floor's"
    );
}

/// A cross-encoder relevance score is not a cosine similarity, so the gate
/// must not request rerank candidate expansion either.
#[test]
fn dedup_search_opts_requests_no_rerank_expansion() {
    assert!(dedup_search_opts().rerank_candidate_limit.is_none());
}

/// Test that the gating booleans (`dedup_enabled`, `must_already_exist`, `force_new`)
/// correctly bypass the dedup gate.  We use a server with dedup_enabled=false / true
/// and call write_document up to the point where the gate would fire — since the
/// embed client isn't reachable the gate's embed call fails-open (logs a warning and
/// continues), but with dedup_enabled=false the gate is never entered at all, so we
/// reach a different error (validation or file existence) and NOT a dedup refusal.
#[tokio::test]
async fn dedup_gate_disabled_via_config_does_not_embed() {
    let tmp = tempfile::tempdir().unwrap();

    // Config with dedup disabled.
    let config = Arc::new(ResolvedConfig {
        source: crate::config::ResolvedSourceConfig {
            git_url: None,
            branch: "master".into(),
            data_path: Some(tmp.path().to_string_lossy().into_owned()),
            git_token_env: "GIT_PULL_TOKEN".into(),
        },
        indexing: crate::config::IndexingConfig::default(),
        frontmatter: crate::config::FrontmatterConfig::default(),
        chunking: crate::config::ChunkingConfig::default(),
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
            collection: "test".into(),
        },
        validation: crate::config::ValidationConfig::default(),
        webhook: crate::config::WebhookConfig::default(),
        mcp: crate::config::ResolvedMcpConfig::default(),
        rate_limit: crate::config::RateLimitConfig::default(),
        write: crate::config::WriteConfig {
            dedup_enabled: false,
            dedup_threshold: 0.85,
            related_min_score: 0.5,
            related_limit: 5,
            archive_prefix: "_archive".to_string(),
            archive_links: crate::config::ArchiveLinkPolicy::Repoint,
            commit_author_name: "mcp-md-wiki".to_string(),
            commit_author_email: "mcp-md-wiki@localhost".to_string(),
        },
        search: crate::config::SearchConfig::default(),
        reranking: None,
        ui: crate::config::UiConfig::default(),
        lint: crate::config::LintConfig::default(),
        formatting: crate::config::FormattingConfig::off(),
        contradictions: crate::config::ContradictionsConfig::default(),
        contradictions_judge: None,
        provenance: Default::default(),
    });

    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    // When dedup is disabled we should NOT get a dedup refusal.
    // The write will fail at git/reindex (no live services) — that's fine.
    let params = WriteDocumentParams {
        path: Some("docs/new.md".to_string()),
        content: Some("---\ntitle: Test Doc\n---\n# Content".to_string()),
        old_string: None,
        new_string: None,
        new_path: None,
        message: None,
        expected_version: None,
        force_new: None,
        frontmatter_patch: None,
        append: None,
        documents: None,
        archive: None,
        archive_links: None,
    };
    let result = server.write_document(Parameters(params)).await;

    // We expect an error (git/reindex will fail in unit test), but it must
    // NOT be a dedup refusal — i.e. it should not mention "similar document".
    if let Err(e) = result {
        assert!(
            !e.message.contains("similar document"),
            "dedup disabled: error must not be a dedup refusal, got: {}",
            e.message
        );
    }
    // (Ok is also fine — means we somehow reached the write step, which is
    // unexpected in a unit test but not a test failure for this assertion.)
}

#[tokio::test]
async fn dedup_gate_bypassed_by_force_new() {
    let tmp = tempfile::tempdir().unwrap();

    // Config with dedup ENABLED (default).
    let config = make_test_resolved_config(tmp.path());

    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    // With force_new=Some(true), the gate must be skipped even when dedup is
    // enabled.  The embed/qdrant will fail-open (no live services), so we will
    // reach git/reindex and fail there — but NOT with a dedup message.
    let params = WriteDocumentParams {
        path: Some("docs/forced.md".to_string()),
        content: Some("---\ntitle: Forced Doc\n---\n# Content".to_string()),
        old_string: None,
        new_string: None,
        new_path: None,
        message: None,
        expected_version: None,
        force_new: Some(true),
        frontmatter_patch: None,
        append: None,
        documents: None,
        archive: None,
        archive_links: None,
    };
    let result = server.write_document(Parameters(params)).await;

    if let Err(e) = result {
        assert!(
            !e.message.contains("similar document"),
            "force_new=true must bypass dedup gate, got: {}",
            e.message
        );
    }
}

#[tokio::test]
async fn dedup_gate_skipped_for_edit_path() {
    let tmp = tempfile::tempdir().unwrap();
    // Create the file so the edit path can proceed past existence check.
    let sub = tmp.path().join("docs");
    std::fs::create_dir_all(&sub).unwrap();
    std::fs::write(
        sub.join("edit-me.md"),
        "---\ntitle: Old Doc\n---\n# old content",
    )
    .unwrap();

    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    // Edit path should never trigger the dedup gate.
    // It will fail at git/reindex — but NOT with a dedup message.
    let params = WriteDocumentParams {
        path: Some("docs/edit-me.md".to_string()),
        old_string: None,
        new_string: None,
        content: Some("---\ntitle: Edited Doc\n---\n# New content".to_string()),
        message: None,
        expected_version: None,
        new_path: None,
        force_new: None,
        frontmatter_patch: None,
        append: None,
        documents: None,
        archive: None,
        archive_links: None,
    };
    let result = server.write_document(Parameters(params)).await;

    if let Err(e) = result {
        assert!(
            !e.message.contains("similar document"),
            "edit path must never trigger dedup gate, got: {}",
            e.message
        );
    }
}

// -----------------------------------------------------------------------
// delete_document unit tests
// -----------------------------------------------------------------------

#[tokio::test]
async fn delete_document_nonexistent_returns_error() {
    let tmp = tempfile::tempdir().unwrap();
    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    let params = DeleteDocumentParams {
        path: "docs/nonexistent.md".to_string(),
        message: None,
        expected_version: None,
        unlink_inbound: None,
    };
    let result = server.delete_document(Parameters(params)).await;

    assert!(
        result.is_err(),
        "delete of non-existent file should return Err"
    );
    let err = result.unwrap_err();
    assert!(
        err.message.contains("does not exist"),
        "error should mention 'does not exist', got: {}",
        err.message
    );
}

#[test]
fn delete_document_relpath_derivation() {
    // Verify that the relpath strip logic produces the expected relative path.
    let canonical_data = std::path::PathBuf::from("/data/kb");
    let canonical_file = std::path::PathBuf::from("/data/kb/sysadmin/networking/dns.md");
    let rel = canonical_file
        .strip_prefix(&canonical_data)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert_eq!(rel, "sysadmin/networking/dns.md");
}

#[test]
fn delete_document_diff_shows_all_as_removals() {
    // When deleting a file, render_unified_diff(old, "", path) should show all
    // lines as removals (every non-header line starts with '-').
    let old = "---\ntitle: My Doc\n---\n# Content\nSome text.\n";
    let diff = render_unified_diff(old, "", "docs/my-doc.md");
    assert!(!diff.is_empty(), "delete diff should be non-empty");
    for line in diff.lines() {
        if !line.starts_with("---")
            && !line.starts_with("+++")
            && !line.starts_with("@@")
            && !line.is_empty()
        {
            assert!(
                line.starts_with('-'),
                "all content lines in a delete diff should be removals, got: {line}"
            );
        }
    }
}

#[test]
fn delete_document_commit_message_has_correct_trailers() {
    let msg = build_commit_message(None, "docs: delete notes/guide.md", "delete_document");
    assert!(
        msg.contains("Tool: mcp-md-wiki"),
        "should contain Tool trailer: {msg}"
    );
    assert!(
        msg.contains("Operation: delete_document"),
        "should contain Operation: delete_document trailer: {msg}"
    );
    assert!(
        msg.starts_with("docs: delete notes/guide.md"),
        "should start with delete subject: {msg}"
    );
}

#[test]
fn delete_document_user_message_overrides_default() {
    let msg = build_commit_message(
        Some("chore: remove obsolete guide"),
        "docs: delete notes/guide.md",
        "delete_document",
    );
    assert!(
        msg.starts_with("chore: remove obsolete guide"),
        "user subject should override default: {msg}"
    );
    assert!(
        msg.contains("Operation: delete_document"),
        "trailer still present: {msg}"
    );
}

#[tokio::test]
async fn delete_document_empty_path_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    let params = DeleteDocumentParams {
        path: "   ".to_string(), // whitespace-only
        message: None,
        expected_version: None,
        unlink_inbound: None,
    };
    let result = server.delete_document(Parameters(params)).await;

    assert!(result.is_err(), "empty path should return Err");
    let err = result.unwrap_err();
    assert!(
        err.message.contains("empty"),
        "error should mention empty path, got: {}",
        err.message
    );
}

#[tokio::test]
async fn delete_document_overlong_path_rejected() {
    // Mirrors `get_document`'s overlong-path test (see that test's comment on
    // MAX_PATH_LEN): `delete_document` must reject the same class of input
    // before the resolver ever runs, not fall through to a resolver-level
    // "not found" error.
    let tmp = tempfile::tempdir().unwrap();
    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    let params = DeleteDocumentParams {
        path: "a".repeat(MAX_PATH_LEN + 1),
        message: None,
        expected_version: None,
        unlink_inbound: None,
    };
    let err = server
        .delete_document(Parameters(params))
        .await
        .expect_err("overlong path should return Err");
    assert!(
        err.message.contains("exceeds maximum length"),
        "error should name the length problem, got: {}",
        err.message
    );
}

#[tokio::test]
async fn delete_document_path_is_literal_and_its_description_does_not_promise_a_basename() {
    // Only `get_document` falls back to a unique basename; `delete_document`
    // resolves `path` relative to the KB root, so the served description must
    // not promise more than that.
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("docs")).unwrap();
    std::fs::write(tmp.path().join("docs/guide.md"), "# Guide\n").unwrap();
    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    let err = server
        .delete_document(Parameters(DeleteDocumentParams {
            path: "guide.md".to_string(),
            message: None,
            expected_version: None,
            unlink_inbound: None,
        }))
        .await
        .expect_err("a bare basename is not a path relative to the KB root");
    assert!(
        err.message.contains("document does not exist"),
        "{}",
        err.message
    );
    assert!(tmp.path().join("docs/guide.md").exists());

    let tool = server
        .get_tool("delete_document")
        .expect("delete_document tool should be registered");
    let description = tool.input_schema["properties"]["path"]["description"]
        .as_str()
        .expect("the path property is described");
    assert!(
        !description.to_lowercase().contains("basename"),
        "{description}"
    );
    assert!(
        description.contains("relative to the KB root"),
        "{description}"
    );
}

// `delete_document_existing_file_proceeds_to_git_step` used to live here: create
// a file in a plain tempdir with no git repo, delete it, and check the failure
// wasn't a path-resolution error. `delete_document_with_no_git_repo_reports_
// inconsistent_state` (below, in the pre-commit/post-commit test group) drives
// the exact same scenario with assertions that actually pin down the new
// behavior — the FailedInconsistentState outcome and message — so it replaces
// this test rather than sitting alongside a strictly weaker duplicate.

// `resolve_safe_write_path` unit tests moved to `write.rs` alongside the
// function itself — see that module's test suite.

// -----------------------------------------------------------------------
// parse_edit_mode unit tests
// -----------------------------------------------------------------------

fn make_edit_params(
    content: Option<&str>,
    old_string: Option<&str>,
    new_string: Option<&str>,
    new_path: Option<&str>,
) -> WriteDocumentParams {
    WriteDocumentParams {
        path: Some("docs/test.md".to_string()),
        content: content.map(|s| s.to_string()),
        old_string: old_string.map(|s| s.to_string()),
        new_string: new_string.map(|s| s.to_string()),
        message: None,
        expected_version: None,
        new_path: new_path.map(|s| s.to_string()),
        force_new: None,
        frontmatter_patch: None,
        append: None,
        documents: None,
        archive: None,
        archive_links: None,
    }
}

#[test]
fn parse_edit_mode_full_replace_is_recognized() {
    let params = make_edit_params(Some("new content"), None, None, None);
    let mode = parse_edit_mode(&params).unwrap();
    assert_eq!(
        mode,
        Some(EditMode::Full {
            content: "new content".to_string()
        })
    );
}

#[test]
fn parse_edit_mode_surgical_is_recognized() {
    let params = make_edit_params(None, Some("old text"), Some("new text"), None);
    let mode = parse_edit_mode(&params).unwrap();
    assert_eq!(
        mode,
        Some(EditMode::Surgical {
            old: "old text".to_string(),
            new: "new text".to_string()
        })
    );
}

#[test]
fn parse_edit_mode_both_modes_rejected() {
    let params = make_edit_params(Some("full content"), Some("old"), Some("new"), None);
    let err = parse_edit_mode(&params).unwrap_err();
    assert!(
        err.contains("mutually exclusive"),
        "expected 'mutually exclusive' in error, got: {err}"
    );
}

#[test]
fn parse_edit_mode_neither_mode_rejected() {
    let params = make_edit_params(None, None, None, None);
    let err = parse_edit_mode(&params).unwrap_err();
    assert!(
        err.contains("must provide"),
        "expected 'must provide' in error, got: {err}"
    );
    assert!(
        err.contains("new_path"),
        "the error must name new_path as a third option now that moves exist, got: {err}"
    );
}

#[test]
fn parse_edit_mode_only_old_string_rejected() {
    let params = make_edit_params(None, Some("old"), None, None);
    let err = parse_edit_mode(&params).unwrap_err();
    assert!(
        err.contains("new_string"),
        "expected mention of new_string in error, got: {err}"
    );
}

#[test]
fn parse_edit_mode_only_new_string_rejected() {
    let params = make_edit_params(None, None, Some("new"), None);
    let err = parse_edit_mode(&params).unwrap_err();
    assert!(
        err.contains("old_string"),
        "expected mention of old_string in error, got: {err}"
    );
}

#[test]
fn parse_edit_mode_identical_old_new_rejected() {
    let params = make_edit_params(None, Some("same text"), Some("same text"), None);
    let err = parse_edit_mode(&params).unwrap_err();
    assert!(
        err.contains("identical"),
        "expected 'identical' in error, got: {err}"
    );
}

// -----------------------------------------------------------------------
// (#179) parse_edit_mode: frontmatter_patch / append arms
// -----------------------------------------------------------------------

fn set_status_op(value: &str) -> FrontmatterPatchOp {
    FrontmatterPatchOp {
        operation: "set_field".to_string(),
        field: "status".to_string(),
        value: Some(serde_json::json!(value)),
        values: None,
    }
}

#[test]
fn parse_edit_mode_patch_alone_is_recognized() {
    let mut params = make_edit_params(None, None, None, None);
    params.frontmatter_patch = Some(vec![set_status_op("active")]);
    let mode = parse_edit_mode(&params).unwrap();
    match mode {
        Some(EditMode::Patch { edits }) => {
            assert_eq!(edits.len(), 1);
            assert_eq!(
                edits[0],
                FrontmatterEdit::SetField {
                    field: "status".to_string(),
                    value: serde_json::json!("active"),
                }
            );
        }
        other => panic!("expected Patch, got {other:?}"),
    }
}

#[test]
fn parse_edit_mode_append_alone_is_recognized() {
    let mut params = make_edit_params(None, None, None, None);
    params.append = Some("- new entry".to_string());
    let mode = parse_edit_mode(&params).unwrap();
    assert_eq!(
        mode,
        Some(EditMode::Append {
            text: "- new entry".to_string()
        })
    );
}

#[test]
fn parse_edit_mode_patch_and_append_combine() {
    let mut params = make_edit_params(None, None, None, None);
    params.frontmatter_patch = Some(vec![set_status_op("active")]);
    params.append = Some("- new entry".to_string());
    let mode = parse_edit_mode(&params).unwrap();
    match mode {
        Some(EditMode::PatchAppend { edits, text }) => {
            assert_eq!(edits.len(), 1);
            assert_eq!(text, "- new entry");
        }
        other => panic!("expected PatchAppend, got {other:?}"),
    }
}

#[test]
fn parse_edit_mode_patch_combined_with_move_is_recognized() {
    let mut params = make_edit_params(None, None, None, Some("docs/new-home.md"));
    params.frontmatter_patch = Some(vec![set_status_op("active")]);
    let mode = parse_edit_mode(&params).unwrap();
    assert!(matches!(mode, Some(EditMode::Patch { .. })));
}

#[test]
fn parse_edit_mode_patch_with_content_is_rejected() {
    let mut params = make_edit_params(Some("full content"), None, None, None);
    params.frontmatter_patch = Some(vec![set_status_op("active")]);
    let err = parse_edit_mode(&params).unwrap_err();
    assert!(
        err.contains("mutually exclusive"),
        "expected 'mutually exclusive' in error, got: {err}"
    );
}

#[test]
fn parse_edit_mode_append_with_surgical_is_rejected() {
    let mut params = make_edit_params(None, Some("old"), Some("new"), None);
    params.append = Some("more text".to_string());
    let err = parse_edit_mode(&params).unwrap_err();
    assert!(
        err.contains("mutually exclusive"),
        "expected 'mutually exclusive' in error, got: {err}"
    );
}

#[test]
fn parse_edit_mode_empty_frontmatter_patch_list_is_rejected() {
    let mut params = make_edit_params(None, None, None, None);
    params.frontmatter_patch = Some(vec![]);
    let err = parse_edit_mode(&params).unwrap_err();
    assert!(err.contains("at least one operation"), "got: {err}");
}

#[test]
fn parse_edit_mode_blank_append_is_rejected() {
    let mut params = make_edit_params(None, None, None, None);
    params.append = Some("   ".to_string());
    let err = parse_edit_mode(&params).unwrap_err();
    assert!(err.contains("empty"), "got: {err}");
}

// -----------------------------------------------------------------------
// (#179) build_frontmatter_edit / parse_frontmatter_patch_ops
// -----------------------------------------------------------------------

#[test]
fn build_frontmatter_edit_set_field_requires_a_value() {
    let op = FrontmatterPatchOp {
        operation: "set_field".to_string(),
        field: "status".to_string(),
        value: None,
        values: None,
    };
    let err = build_frontmatter_edit(&op).unwrap_err();
    assert!(err.contains("set_field"), "got: {err}");
}

#[test]
fn build_frontmatter_edit_add_values_requires_non_empty_values() {
    let op = FrontmatterPatchOp {
        operation: "add_values".to_string(),
        field: "tags".to_string(),
        value: None,
        values: Some(vec![]),
    };
    let err = build_frontmatter_edit(&op).unwrap_err();
    assert!(err.contains("non-empty"), "got: {err}");
}

#[test]
fn build_frontmatter_edit_rejects_unknown_operation() {
    let op = FrontmatterPatchOp {
        operation: "delete_everything".to_string(),
        field: "status".to_string(),
        value: None,
        values: None,
    };
    let err = build_frontmatter_edit(&op).unwrap_err();
    assert!(err.contains("unknown operation"), "got: {err}");
}

#[test]
fn build_frontmatter_edit_rejects_an_empty_field() {
    let op = FrontmatterPatchOp {
        operation: "remove_field".to_string(),
        field: "  ".to_string(),
        value: None,
        values: None,
    };
    let err = build_frontmatter_edit(&op).unwrap_err();
    assert!(err.contains("field"), "got: {err}");
}

#[test]
fn build_frontmatter_edit_remove_values_parses() {
    let op = FrontmatterPatchOp {
        operation: "remove_values".to_string(),
        field: "tags".to_string(),
        value: None,
        values: Some(vec![serde_json::json!("a")]),
    };
    let edit = build_frontmatter_edit(&op).unwrap();
    assert_eq!(
        edit,
        FrontmatterEdit::RemoveValues {
            field: "tags".to_string(),
            values: vec![serde_json::json!("a")],
        }
    );
}

#[test]
fn parse_frontmatter_patch_ops_rejects_too_many_operations() {
    let ops: Vec<FrontmatterPatchOp> = (0..(MAX_FRONTMATTER_PATCH_OPS + 1))
        .map(|i| set_status_op(&i.to_string()))
        .collect();
    let err = parse_frontmatter_patch_ops(&ops).unwrap_err();
    assert!(err.contains("too many operations"), "got: {err}");
}

// -----------------------------------------------------------------------
// parse_edit_mode: new_path (move) arms
// -----------------------------------------------------------------------

#[test]
fn parse_edit_mode_move_alone_is_a_pure_move() {
    // Neither content mode, but new_path set: Ok(None) — a pure move, content
    // unchanged.
    let params = make_edit_params(None, None, None, Some("docs/new-home.md"));
    let mode = parse_edit_mode(&params).unwrap();
    assert_eq!(
        mode, None,
        "move-only should parse to Ok(None), got: {mode:?}"
    );
}

#[test]
fn parse_edit_mode_surgical_combined_with_move_is_recognized() {
    let params = make_edit_params(
        None,
        Some("old text"),
        Some("new text"),
        Some("docs/new-home.md"),
    );
    let mode = parse_edit_mode(&params).unwrap();
    assert_eq!(
        mode,
        Some(EditMode::Surgical {
            old: "old text".to_string(),
            new: "new text".to_string()
        }),
        "surgical + new_path must still parse as Ok(Some(Surgical))"
    );
}

#[test]
fn parse_edit_mode_full_replace_combined_with_move_is_recognized() {
    let params = make_edit_params(Some("new content"), None, None, Some("docs/new-home.md"));
    let mode = parse_edit_mode(&params).unwrap();
    assert_eq!(
        mode,
        Some(EditMode::Full {
            content: "new content".to_string()
        }),
        "full-replace + new_path must still parse as Ok(Some(Full))"
    );
}

#[test]
fn parse_edit_mode_both_modes_still_rejected_even_with_move() {
    // surgical and full-replace remain mutually exclusive WITH EACH OTHER
    // regardless of whether new_path is also present.
    let params = make_edit_params(
        Some("full content"),
        Some("old"),
        Some("new"),
        Some("docs/new-home.md"),
    );
    let err = parse_edit_mode(&params).unwrap_err();
    assert!(
        err.contains("mutually exclusive"),
        "expected 'mutually exclusive' in error even with new_path set, got: {err}"
    );
}

// -----------------------------------------------------------------------
// apply_surgical unit tests
// -----------------------------------------------------------------------

#[test]
fn apply_surgical_single_occurrence_replaced() {
    let old = "Hello world!\nGoodbye earth!";
    let result = apply_surgical(old, "world", "Rust", "d.md").unwrap();
    assert_eq!(result, "Hello Rust!\nGoodbye earth!");
}

#[test]
fn apply_surgical_not_found_names_the_document_path() {
    // Regression coverage for issue #88: the error must name the actual document,
    // not the word "document" — and, now that the message is built directly
    // rather than via a blind `.replace("document", ...)`, an occurrence of the
    // word "document" elsewhere in the message (e.g. in "get_document") must
    // survive untouched.
    let old = "Hello world! Nothing here resembles the needle at all, at all.";
    let err = apply_surgical(
        old,
        "missing text entirely unrelated to this content, over forty chars long",
        "replacement",
        "food/plans/2026-07-30.md",
    )
    .unwrap_err();
    assert!(
        err.contains("not found in 'food/plans/2026-07-30.md'"),
        "expected the document path in the error, got: {err}"
    );
    assert!(
        err.contains("get_document"),
        "the word 'document' inside 'get_document' must survive intact, got: {err}"
    );
}

#[test]
fn apply_surgical_not_found_but_whitespace_normalized_match_exists() {
    // The most common real cause per issue #88: same text, different indentation /
    // line endings / trailing whitespace. Must be called out explicitly rather than
    // left for the caller to guess.
    let old = "line one\n    line two   \nline three";
    let old_string = "line one\nline two\nline three"; // no indentation, no trailing spaces
    let err = apply_surgical(old, old_string, "replacement", "notes.md").unwrap_err();
    assert!(
        err.contains("whitespace"),
        "expected a whitespace near-match callout, got: {err}"
    );
    assert!(
        err.contains("notes.md"),
        "expected the document path in the error, got: {err}"
    );
}

#[test]
fn apply_surgical_not_found_but_anchor_matches_shows_context() {
    // old_string's first 40 chars appear verbatim in the document, but the text
    // diverges after that point — the caller gets to see exactly where and how.
    let anchor = "0123456789012345678901234567890123456789"; // exactly 40 chars
    let old_string = format!("{anchor}XYZ_EXPECTED_TAIL");
    let old = format!("prefix text before it {anchor}ABC_ACTUAL_TAIL and trailing text after");
    let err = apply_surgical(&old, &old_string, "replacement", "d.md").unwrap_err();
    assert!(
        err.contains(anchor),
        "expected the matched anchor text in the error, got: {err}"
    );
    assert!(
        err.contains("ABC_ACTUAL_TAIL"),
        "expected surrounding document context in the error, got: {err}"
    );
}

#[test]
fn apply_surgical_not_found_and_nothing_resembles_it() {
    let old = "A short paragraph about something else entirely.";
    let old_string = "Completely unrelated text that will never appear anywhere in the source, over forty chars.";
    let err = apply_surgical(old, old_string, "replacement", "d.md").unwrap_err();
    assert!(
        err.contains("wrong document") || err.contains("changed substantially"),
        "expected guidance that nothing resembles old_string, got: {err}"
    );
}

#[test]
fn apply_surgical_not_found_diagnostics_are_skipped_above_the_size_cap() {
    // Bounds the diagnostic work: past NOT_FOUND_DIAGNOSTIC_MAX_BYTES, fall back to
    // the plain message rather than running whitespace-normalization or an anchor
    // search over an arbitrarily large document.
    let old = "x".repeat(NOT_FOUND_DIAGNOSTIC_MAX_BYTES + 1);
    let err = apply_surgical(
        &old,
        "missing text over forty characters long, easily",
        "r",
        "big.md",
    )
    .unwrap_err();
    assert_eq!(
        err, "old_string not found in 'big.md'",
        "past the size cap the message must be the plain fallback, got: {err}"
    );
}

#[test]
fn apply_surgical_multiple_occurrences_returns_error_with_count() {
    let old = "foo bar foo baz foo";
    let err = apply_surgical(old, "foo", "qux", "d.md").unwrap_err();
    assert!(
        err.contains("3"),
        "error should mention occurrence count (3), got: {err}"
    );
    assert!(
        err.contains("not unique in 'd.md'"),
        "error should name the document and say 'not unique', got: {err}"
    );
}

#[test]
fn apply_surgical_exact_single_unique_string() {
    let old = "---\ntitle: My Doc\n---\n# Content\nSome text here.";
    let result = apply_surgical(old, "Some text here.", "Updated text.", "d.md").unwrap();
    assert_eq!(result, "---\ntitle: My Doc\n---\n# Content\nUpdated text.");
}

// -----------------------------------------------------------------------
// render_unified_diff unit tests
// -----------------------------------------------------------------------

#[test]
fn render_unified_diff_shows_added_lines() {
    let old = "line1\nline2\n";
    let new = "line1\nline2\nline3\n";
    let diff = render_unified_diff(old, new, "docs/test.md");
    assert!(
        !diff.is_empty(),
        "diff should be non-empty for a changed doc"
    );
    assert!(
        diff.contains("+line3"),
        "diff should show added line, got:\n{diff}"
    );
    assert!(
        diff.contains("a/docs/test.md"),
        "diff header should name the file, got:\n{diff}"
    );
}

#[test]
fn render_unified_diff_shows_removed_lines() {
    let old = "line1\nline2\nline3\n";
    let new = "line1\nline3\n";
    let diff = render_unified_diff(old, new, "docs/test.md");
    assert!(
        diff.contains("-line2"),
        "diff should show removed line, got:\n{diff}"
    );
}

#[test]
fn render_unified_diff_identical_content_is_empty() {
    let content = "line1\nline2\n";
    let diff = render_unified_diff(content, content, "docs/test.md");
    assert!(
        diff.is_empty(),
        "identical content should produce empty diff, got:\n{diff}"
    );
}

#[test]
fn render_unified_diff_create_shows_all_as_additions() {
    let old = "";
    let new = "---\ntitle: New Doc\n---\n# Hello\n";
    let diff = render_unified_diff(old, new, "docs/new.md");
    assert!(!diff.is_empty(), "new file diff should be non-empty");
    // Every non-header line should be an addition.
    for line in diff.lines() {
        if !line.starts_with("---")
            && !line.starts_with("+++")
            && !line.starts_with("@@")
            && !line.is_empty()
        {
            assert!(
                line.starts_with('+'),
                "all content lines in a create diff should be additions, got: {line}"
            );
        }
    }
}

// ------------------------------------------------------------------
// parse_date_to_timestamp tests
// ------------------------------------------------------------------

#[test]
fn parse_date_rfc3339_returns_unix_timestamp() {
    // 2024-01-15T00:00:00Z is a known timestamp
    let ts = parse_date_to_timestamp("2024-01-15T00:00:00Z").unwrap();
    assert_eq!(
        ts, 1_705_276_800,
        "RFC 3339 midnight UTC should parse correctly"
    );
}

#[test]
fn parse_date_date_only_treated_as_midnight_utc() {
    let ts = parse_date_to_timestamp("2024-01-15").unwrap();
    assert_eq!(
        ts, 1_705_276_800,
        "date-only should be treated as midnight UTC"
    );
}

#[test]
fn parse_date_invalid_string_returns_err() {
    let result = parse_date_to_timestamp("not-a-date");
    assert!(
        result.is_err(),
        "invalid date string should return an error"
    );
}

#[test]
fn parse_date_rfc3339_with_offset_returns_utc_equivalent() {
    // 2024-01-15T01:00:00+01:00 == 2024-01-15T00:00:00Z
    let ts = parse_date_to_timestamp("2024-01-15T01:00:00+01:00").unwrap();
    assert_eq!(ts, 1_705_276_800, "offset datetime should convert to UTC");
}

// --- write tools queue instead of reindexing inline ---
//
// Before the async reindex worker, these tools awaited `ingest::run_index` inline
// and used `REINDEX_LOCK` to keep that from racing the webhook. Now they just mark
// paths dirty on their server's `ReindexQueue` and return — which also means these
// tests no longer need a live Qdrant/embeddings service to reach that point, since
// nothing here calls into the indexer at all.

#[tokio::test]
async fn create_document_reports_queued_indexing_without_touching_the_indexer() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let (server, _config) = make_git_backed_server(&work);

    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("docs/queued.md".to_string()),
            content: Some(
                "---\ntitle: Queued\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n"
                    .to_string(),
            ),
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_version: None,
            force_new: Some(true),
            frontmatter_patch: None,
            append: None,
            documents: None,
            archive: None,
            archive_links: None,
        }))
        .await;

    let result = result.expect("write must succeed even though nothing indexes it inline");
    let structured = single_representation(&result);
    assert_eq!(
        structured,
        serde_json::json!({
            "action": "created",
            "path": "docs/queued.md",
            "version": structured["version"],
        }),
        "a create reports where and what, and echoes no diff of what was sent"
    );

    crate::reindex::test_support::assert_marked_dirty(&server.reindex_queue, &["docs/queued.md"]);
}

/// The correctness case the synchronous rebuild in `update_schema` exists for:
/// an agent that widens a rule and then immediately writes against it must be
/// validated against the NEW schema, not whatever was cached when the server
/// started. If `update_schema` only marked a full reconcile (`mark_full`) and
/// left the refresh to the reindex worker, this would fail here, because
/// nothing in this test spawns that worker — exactly the regression this test
/// is meant to catch.
#[tokio::test]
async fn create_document_immediately_after_update_schema_validates_against_the_new_rules() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    // The schema baked into the server's cache at construction time below: only
    // "active" is permitted yet.
    write_schema_file(
        &work,
        "notes",
        "fields:\n  status:\n    type: enum\n    values: [active]\n",
    );
    let (server, _config) = make_git_backed_server(&work);

    // Sanity check that the OLD schema really does reject "beta" — otherwise the
    // second half of this test would not be proving anything.
    let rejected = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("notes/before.md".to_string()),
            content: Some("---\ntitle: Before\nstatus: beta\n---\n\n# Body\n".to_string()),
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_version: None,
            force_new: Some(true),
            frontmatter_patch: None,
            append: None,
            documents: None,
            archive: None,
            archive_links: None,
        }))
        .await;
    assert!(
        rejected.is_err(),
        "sanity check failed: 'beta' must not be permitted before the schema \
         change — got {rejected:?}"
    );

    // Widen the rule through the tool an agent would actually use.
    server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("notes".into()),
            operation: "add_values".into(),
            field: "status".into(),
            values: Some(vec!["beta".into()]),
            definition: None,
            dry_run: None,
            force: None,
            acknowledge_root_change: None,
        }))
        .await
        .expect("update_schema must succeed against this git-backed harness");

    // Same server, same cached schema handle, next call: must see the new rule.
    let accepted = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("notes/after.md".to_string()),
            content: Some("---\ntitle: After\nstatus: beta\n---\n\n# Body\n".to_string()),
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_version: None,
            force_new: Some(true),
            frontmatter_patch: None,
            append: None,
            documents: None,
            archive: None,
            archive_links: None,
        }))
        .await;
    assert!(
        accepted.is_ok(),
        "the write immediately after update_schema must validate against the \
         NEW schema, not a stale cached copy: {:?}",
        accepted.err()
    );
}

#[tokio::test]
async fn delete_document_reports_queued_cleanup_without_touching_the_indexer() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    std::fs::write(
        work.path().join("doomed-queued-cleanup-test.md"),
        "---\ntitle: D\n---\n\n# Body\n",
    )
    .unwrap();
    // delete_document git-adds the removed path, so the file must already be tracked.
    std::process::Command::new("git")
        .args(["add", "doomed-queued-cleanup-test.md"])
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
            "add doomed-queued-cleanup-test.md",
        ])
        .current_dir(work.path())
        .output()
        .unwrap();
    let (server, _config) = make_git_backed_server(&work);

    let result = server
        .delete_document(Parameters(DeleteDocumentParams {
            path: "doomed-queued-cleanup-test.md".to_string(),
            message: None,
            expected_version: version_at(&work, "doomed-queued-cleanup-test.md"),
            unlink_inbound: None,
        }))
        .await;

    let result = result.expect("delete must succeed even though nothing purges it inline");
    let structured = single_representation(&result);
    assert_eq!(
        structured,
        serde_json::json!({"action": "deleted", "path": "doomed-queued-cleanup-test.md"}),
        "a delete echoes no diff of the removed document"
    );

    crate::reindex::test_support::assert_marked_dirty(
        &server.reindex_queue,
        &["doomed-queued-cleanup-test.md"],
    );
}

#[tokio::test]
async fn delete_document_surfaces_referencing_paths_end_to_end() {
    // (#229) The reverse-link check #181 added only ever reached a server
    // log — this proves it now also reaches the caller, through both the
    // human-readable text and `structured_content`.
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    std::fs::write(
        work.path().join("linked.md"),
        "---\ntitle: Linked\n---\n\n# Body\n",
    )
    .unwrap();
    git_commit_all(&work, "linked.md", "add linked.md");
    let (server, _config) = make_git_backed_server(&work);

    // Seed the reverse-link index directly (same shortcut other link-graph
    // tests in this module use) rather than depending on a real reindex.
    let db = server.state_db().await.unwrap();
    db.replace_links(
        "referencer.md",
        "markdown",
        &[("linked.md".to_string(), None)],
    )
    .await
    .unwrap();

    let result = server
        .delete_document(Parameters(DeleteDocumentParams {
            path: "linked.md".to_string(),
            message: None,
            expected_version: version_at(&work, "linked.md"),
            unlink_inbound: None,
        }))
        .await
        .expect("an inbound link must not block the delete");

    let structured = single_representation(&result);
    assert_eq!(
        structured["referencing_paths"],
        serde_json::json!(["referencer.md"])
    );
}

#[tokio::test]
async fn delete_document_reports_empty_referencing_paths_when_nothing_links_to_it() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    std::fs::write(
        work.path().join("unlinked.md"),
        "---\ntitle: Unlinked\n---\n\n# Body\n",
    )
    .unwrap();
    git_commit_all(&work, "unlinked.md", "add unlinked.md");
    let (server, _config) = make_git_backed_server(&work);

    let result = server
        .delete_document(Parameters(DeleteDocumentParams {
            path: "unlinked.md".to_string(),
            message: None,
            expected_version: version_at(&work, "unlinked.md"),
            unlink_inbound: None,
        }))
        .await
        .unwrap();

    let structured = result.structured_content.unwrap();
    assert!(
        structured.get("referencing_paths").is_none(),
        "an empty list is omitted: {structured}"
    );
}

// -----------------------------------------------------------------------
// An edit's diff in the result — fix #129 — and
// no versioning internals in either
// -----------------------------------------------------------------------

#[tokio::test]
async fn write_document_edit_carries_the_diff_and_no_versioning_detail() {
    // A client reading only structured content (Claude Code does) must see the
    // diff of an edit (#129) — and nothing may say how the change was versioned.
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    std::fs::create_dir_all(work.path().join("docs")).unwrap();
    std::fs::write(
        work.path().join("docs/new.md"),
        "---\ntitle: Test\n---\n# Body\n",
    )
    .unwrap();
    git_commit_all(&work, "docs/new.md", "add docs/new.md");
    let (server, _config) = make_git_backed_server(&work);

    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("docs/new.md".to_string()),
            old_string: Some("# Body".to_string()),
            new_string: Some("# Better body".to_string()),
            ..Default::default()
        }))
        .await
        .unwrap();

    let structured = single_representation(&result);
    assert!(
        !structured.to_string().contains(&head_sha(&work)[..8]),
        "the result must not name the commit: {structured}"
    );
    let diff = structured["diff"]
        .as_str()
        .expect("an edit carries the diff as a string");
    assert!(diff.contains("+# Better body"), "{diff}");
    assert!(
        structured.get("diff_truncated").is_none() && structured.get("diff_total_bytes").is_none(),
        "an uncut diff carries no cap keys: {structured}"
    );
    assert!(structured.get("merged_with_other_changes").is_none());
    assert!(structured.get("rewritten_paths").is_none());
    assert_plain_success(&result);
}

#[tokio::test]
async fn write_document_structured_diff_is_capped_for_a_large_write() {
    // The diff of a full replace of a big document can be large — never
    // truncated silently: capped WITH a flag and the true size.
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    std::fs::create_dir_all(work.path().join("docs")).unwrap();
    std::fs::write(
        work.path().join("docs/big.md"),
        "---\ntitle: Big\n---\n# Body\n",
    )
    .unwrap();
    git_commit_all(&work, "docs/big.md", "add docs/big.md");
    let (server, _config) = make_git_backed_server(&work);

    // Comfortably over MAX_STRUCTURED_DIFF_BYTES (8 KiB) so the added-lines
    // diff itself, not just the raw content, exceeds the cap.
    let big_body: String = (0..2000)
        .map(|i| format!("line {i} of filler body text\n"))
        .collect();
    let content = format!("---\ntitle: Big\n---\n# Body\n{big_body}");

    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("docs/big.md".to_string()),
            content: Some(content),
            expected_version: version_at(&work, "docs/big.md"),
            ..Default::default()
        }))
        .await
        .unwrap();

    let structured = single_representation(&result);
    let diff = structured["diff"].as_str().unwrap();
    assert!(
        diff.len() <= MAX_STRUCTURED_DIFF_BYTES,
        "capped diff must not exceed the byte cap, got {} bytes",
        diff.len()
    );
    assert_eq!(structured["diff_truncated"], serde_json::json!(true));
    let diff_total_bytes = structured["diff_total_bytes"].as_u64().unwrap() as usize;
    assert!(
        diff_total_bytes > MAX_STRUCTURED_DIFF_BYTES,
        "diff_total_bytes must report the TRUE, uncapped length"
    );
}

#[tokio::test]
async fn write_document_refuses_an_authored_derived_field() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    write_schema_file(
        &work,
        "",
        "derived:\n  domain: {from: path_segment, index: 0}\n",
    );
    git_commit_all(&work, crate::schema::SCHEMA_FILE_NAME, "add root schema");
    let (server, _config) = make_git_backed_server(&work);

    let err = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("docs/fresh.md".to_string()),
            content: Some("---\ntitle: Fresh\ndomain: docs\n---\n# Body\n".to_string()),
            force_new: Some(true),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    let text = format!("{err:?}");
    assert!(
        text.contains("`domain` is set from the document's folder"),
        "{text}"
    );
    assert!(!work.path().join("docs/fresh.md").exists());

    // The same document without it is accepted.
    server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("docs/fresh.md".to_string()),
            content: Some("---\ntitle: Fresh\n---\n# Body\n".to_string()),
            force_new: Some(true),
            ..Default::default()
        }))
        .await
        .unwrap();
}

#[tokio::test]
async fn write_document_create_and_delete_echo_no_diff() {
    // The caller just sent the content (create) or knows what it removed
    // (delete); echoing it back as a diff only costs context.
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    std::fs::write(
        work.path().join("gone.md"),
        "---\ntitle: Gone\n---\n\n# Body\n",
    )
    .unwrap();
    git_commit_all(&work, "gone.md", "add gone.md");
    let (server, _config) = make_git_backed_server(&work);

    let created = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("docs/fresh.md".to_string()),
            content: Some("---\ntitle: Fresh\n---\n# Body\n".to_string()),
            force_new: Some(true),
            ..Default::default()
        }))
        .await
        .unwrap();
    let created = single_representation(&created);
    assert!(created.get("diff").is_none(), "{created}");
    assert_eq!(created["action"], "created");

    let result = server
        .delete_document(Parameters(DeleteDocumentParams {
            path: "gone.md".to_string(),
            message: None,
            expected_version: version_at(&work, "gone.md"),
            unlink_inbound: None,
        }))
        .await
        .unwrap();
    let structured = single_representation(&result);
    assert!(structured.get("diff").is_none(), "{structured}");
    assert_eq!(structured["action"], "deleted");
    assert_eq!(structured["path"], "gone.md");
    assert_plain_success(&result);
}

#[tokio::test]
async fn write_document_directory_move_result_carries_no_versioning_detail() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    std::fs::create_dir_all(work.path().join("old-project7")).unwrap();
    let doc_content = "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
    std::fs::write(work.path().join("old-project7/a.md"), doc_content).unwrap();
    git_commit_all(&work, "old-project7/a.md", "add old-project7/a.md");
    let (server, _config) = make_git_backed_server(&work);

    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("old-project7".to_string()),
            content: None,
            old_string: None,
            new_string: None,
            new_path: Some("archive/new-project7".to_string()),
            message: None,
            expected_version: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
            documents: None,
            archive: None,
            archive_links: None,
        }))
        .await
        .unwrap();

    assert_eq!(
        result.structured_content.as_ref().unwrap()["moved"][0]["to"],
        "archive/new-project7/a.md"
    );
    assert_plain_success(&result);
}

// -----------------------------------------------------------------------
// Concurrent changes: re-applied under the lock, merged, or refused
// -----------------------------------------------------------------------

fn retitle_target() -> WriteDocumentParams {
    WriteDocumentParams {
        path: Some("docs/target.md".to_string()),
        old_string: Some("title: Old".to_string()),
        new_string: Some("title: New".to_string()),
        content: None,
        message: None,
        expected_version: None,
        new_path: None,
        force_new: None,
        frontmatter_patch: None,
        append: None,
        documents: None,
        archive: None,
        archive_links: None,
    }
}

#[tokio::test]
async fn a_concurrent_change_to_another_document_is_not_reported() {
    let h = concurrent_harness();
    std::fs::write(h.peer.path().join("elsewhere.md"), "# Elsewhere\n").unwrap();
    peer_git(h.peer.path(), &["add", "elsewhere.md"]);
    peer_git(
        h.peer.path(),
        &["commit", "-q", "-m", "peer adds elsewhere"],
    );
    peer_git(h.peer.path(), &["push", "-q", "origin", "HEAD:master"]);

    let result = h
        .server
        .write_document(Parameters(retitle_target()))
        .await
        .expect("the edit must succeed");

    // The server did pull the peer's document in — and marks it for indexing…
    assert!(h.work.path().join("elsewhere.md").exists());
    crate::reindex::test_support::assert_marked_dirty(
        &h.server.reindex_queue,
        &["docs/target.md", "elsewhere.md"],
    );
    // …but the caller hears nothing about it.
    assert_plain_success(&result);
    let structured = result.structured_content.as_ref().unwrap();
    assert!(
        structured.get("merged_with_other_changes").is_none(),
        "{structured}"
    );
    let text = format!("{:?}", result.content);
    assert!(!text.contains("elsewhere"), "{text}");
    assert!(!text.contains("someone else"), "{text}");
}

#[tokio::test]
async fn a_concurrent_relative_edit_and_this_one_both_land() {
    let h = concurrent_harness();
    let other = h.server.clone();
    let result = with_hook_once(
        crate::git::HookPoint::BeforeLock,
        move || async move {
            other
                .write_document(Parameters(WriteDocumentParams {
                    old_string: Some("last line".to_string()),
                    new_string: Some("last line, edited concurrently".to_string()),
                    ..retitle_target()
                }))
                .await
                .expect("the concurrent edit lands");
        },
        h.server.write_document(Parameters(retitle_target())),
    )
    .await
    .expect("the edit applies to the content as it is under the lock");

    let on_disk = std::fs::read_to_string(h.work.path().join("docs/target.md")).unwrap();
    assert!(
        on_disk.contains("title: New") && on_disk.contains("edited concurrently"),
        "{on_disk}"
    );
    // Applied to current content, not merged after the fact: nothing to report.
    assert_plain_success(&result);
    let structured = result.structured_content.as_ref().unwrap();
    assert!(structured.get("merged_with_other_changes").is_none());
    assert_eq!(
        structured["version"],
        serde_json::json!(write::document_version(on_disk.as_bytes()))
    );
}

#[tokio::test]
async fn a_relative_edit_based_on_a_stale_version_says_it_merged() {
    // The anchor still applies, so the edit lands on the peer's content —
    // and the result must say so, or a caller chaining a full replace from
    // its stale copy with the returned version would revert the peer's edit.
    let h = concurrent_harness();
    let base = target_version(&h);
    peer_push(
        h.peer.path(),
        "docs/target.md",
        &CONCURRENT_TARGET.replace("last line", "last line, edited by the peer"),
    );

    let result = h
        .server
        .write_document(Parameters(WriteDocumentParams {
            expected_version: Some(base),
            ..retitle_target()
        }))
        .await
        .expect("the anchor still applies");

    let on_disk = std::fs::read_to_string(h.work.path().join("docs/target.md")).unwrap();
    assert!(
        on_disk.contains("title: New") && on_disk.contains("edited by the peer"),
        "{on_disk}"
    );
    let structured = result.structured_content.as_ref().unwrap();
    assert_eq!(
        structured["merged_with_other_changes"],
        serde_json::json!(true)
    );
    assert_eq!(
        structured["version"],
        serde_json::json!(write::document_version(on_disk.as_bytes()))
    );
}

#[tokio::test]
async fn a_create_carrying_expected_version_says_the_document_is_gone() {
    // `path` resolves nowhere, so this is a create — but `expected_version`
    // says the caller read a document there, which has since been deleted or
    // moved: re-creating it silently would resurrect it.
    let h = concurrent_harness();
    let base = target_version(&h);
    let head_before = rev(h.work.path(), "HEAD");

    let err = h
        .server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("docs/moved-away.md".to_string()),
            old_string: None,
            new_string: None,
            content: Some(CONCURRENT_TARGET.to_string()),
            expected_version: Some(base),
            ..retitle_target()
        }))
        .await
        .expect_err("nothing is at that path any more");
    for needle in [
        "does not exist",
        "search",
        "get_document",
        "expected_version",
    ] {
        assert!(err.message.contains(needle), "{}", err.message);
    }
    assert!(!h.work.path().join("docs/moved-away.md").exists());
    assert_eq!(rev(h.work.path(), "HEAD"), head_before);
}

#[tokio::test]
async fn a_relative_edit_whose_anchor_a_concurrent_edit_removed_is_refused() {
    let h = concurrent_harness();
    let peer = h.peer.path().to_path_buf();
    let err = with_hook_once(
        crate::git::HookPoint::BeforeLock,
        move || async move {
            peer_push(
                &peer,
                "docs/target.md",
                &CONCURRENT_TARGET.replace("title: Old", "title: Peer"),
            );
        },
        h.server.write_document(Parameters(retitle_target())),
    )
    .await
    .expect_err("the anchor is gone");
    assert!(
        err.message.contains(write::EDITED_ELSEWHERE),
        "{}",
        err.message
    );
    let on_disk = std::fs::read_to_string(h.work.path().join("docs/target.md")).unwrap();
    assert!(on_disk.contains("title: Peer"), "{on_disk}");
}

#[tokio::test]
async fn a_stale_full_replace_merges_a_non_overlapping_concurrent_change() {
    let h = concurrent_harness();
    let base = target_version(&h);
    peer_push(
        h.peer.path(),
        "docs/target.md",
        &CONCURRENT_TARGET.replace("last line", "last line, edited by the peer"),
    );

    let result = h
        .server
        .write_document(Parameters(WriteDocumentParams {
            old_string: None,
            new_string: None,
            content: Some(CONCURRENT_TARGET.replace("title: Old", "title: Mine")),
            expected_version: Some(base),
            ..retitle_target()
        }))
        .await
        .expect("non-overlapping changes merge");

    let on_disk = std::fs::read_to_string(h.work.path().join("docs/target.md")).unwrap();
    assert!(
        on_disk.contains("title: Mine") && on_disk.contains("edited by the peer"),
        "{on_disk}"
    );
    let structured = result.structured_content.as_ref().unwrap();
    assert_eq!(
        structured["merged_with_other_changes"],
        serde_json::json!(true)
    );
    let text = single_representation(&result).to_string();
    assert_git_free("merged write result", &text);
}

#[tokio::test]
async fn a_stale_full_replace_overlapping_a_concurrent_change_is_refused() {
    let h = concurrent_harness();
    let base = target_version(&h);
    let peer_content = CONCURRENT_TARGET.replace("title: Old", "title: Peer");
    peer_push(h.peer.path(), "docs/target.md", &peer_content);

    let err = h
        .server
        .write_document(Parameters(WriteDocumentParams {
            old_string: None,
            new_string: None,
            content: Some(CONCURRENT_TARGET.replace("title: Old", "title: Mine")),
            expected_version: Some(base),
            ..retitle_target()
        }))
        .await
        .expect_err("overlapping changes conflict");
    assert!(
        err.message.contains(write::EDITED_ELSEWHERE),
        "{}",
        err.message
    );
    assert_eq!(
        std::fs::read_to_string(h.work.path().join("docs/target.md")).unwrap(),
        peer_content
    );
}

#[tokio::test]
async fn absolute_changes_without_expected_version_are_refused() {
    let h = concurrent_harness();
    let replace = h
        .server
        .write_document(Parameters(WriteDocumentParams {
            old_string: None,
            new_string: None,
            content: Some(CONCURRENT_TARGET.replace("Old", "New")),
            ..retitle_target()
        }))
        .await
        .expect_err("a full replace needs expected_version");
    assert!(
        replace.message.contains("expected_version"),
        "{}",
        replace.message
    );

    let moved = h
        .server
        .write_document(Parameters(WriteDocumentParams {
            old_string: None,
            new_string: None,
            new_path: Some("docs/moved.md".to_string()),
            ..retitle_target()
        }))
        .await
        .expect_err("a move needs expected_version");
    assert!(
        moved.message.contains("expected_version"),
        "{}",
        moved.message
    );

    let deleted = h
        .server
        .delete_document(Parameters(DeleteDocumentParams {
            path: "docs/target.md".to_string(),
            message: None,
            expected_version: None,
            unlink_inbound: None,
        }))
        .await
        .expect_err("a delete needs expected_version");
    assert!(
        deleted.message.contains("expected_version"),
        "{}",
        deleted.message
    );

    assert_eq!(
        std::fs::read_to_string(h.work.path().join("docs/target.md")).unwrap(),
        CONCURRENT_TARGET
    );
}

#[tokio::test]
async fn delete_and_move_of_a_stale_version_are_refused() {
    let h = concurrent_harness();
    let base = target_version(&h);
    let peer_content = CONCURRENT_TARGET.replace("last line", "peer line");
    peer_push(h.peer.path(), "docs/target.md", &peer_content);

    let deleted = h
        .server
        .delete_document(Parameters(DeleteDocumentParams {
            path: "docs/target.md".to_string(),
            message: None,
            expected_version: Some(base.clone()),
            unlink_inbound: None,
        }))
        .await
        .expect_err("a stale delete is refused");
    assert!(
        deleted.message.contains(write::EDITED_ELSEWHERE),
        "{}",
        deleted.message
    );

    let moved = h
        .server
        .write_document(Parameters(WriteDocumentParams {
            old_string: None,
            new_string: None,
            new_path: Some("docs/moved.md".to_string()),
            expected_version: Some(base),
            ..retitle_target()
        }))
        .await
        .expect_err("a stale move is refused");
    assert!(
        moved.message.contains(write::EDITED_ELSEWHERE),
        "{}",
        moved.message
    );

    assert_eq!(
        std::fs::read_to_string(h.work.path().join("docs/target.md")).unwrap(),
        peer_content
    );
    assert!(!h.work.path().join("docs/moved.md").exists());
}

#[tokio::test]
async fn a_push_rejected_by_a_concurrent_push_resets_and_retries() {
    let h = concurrent_harness();
    let peer = h.peer.path().to_path_buf();
    let result = with_hook_once(
        crate::git::HookPoint::BeforePush,
        move || async move { peer_push(&peer, "elsewhere.md", "# Elsewhere\n") },
        h.server.write_document(Parameters(retitle_target())),
    )
    .await
    .expect("a push race with another file is retried, not refused");
    assert_plain_success(&result);

    // Both changes are on the remote, and the clone is exactly the remote tip.
    let bare = h._bare.path();
    assert_eq!(rev(h.work.path(), "HEAD"), rev(bare, "master"));
    assert!(h.work.path().join("elsewhere.md").exists());
    let on_disk = std::fs::read_to_string(h.work.path().join("docs/target.md")).unwrap();
    assert!(on_disk.contains("title: New"), "{on_disk}");
    crate::reindex::test_support::assert_marked_dirty(
        &h.server.reindex_queue,
        &["docs/target.md", "elsewhere.md"],
    );
}

#[tokio::test]
async fn a_conflicting_remote_change_refuses_the_write_and_leaves_no_divergence() {
    let h = concurrent_harness();
    let peer = h.peer.path().to_path_buf();
    let err = with_hook_each(
        crate::git::HookPoint::BeforePush,
        move |n| {
            peer_push(
                &peer,
                "docs/target.md",
                &CONCURRENT_TARGET.replace("title: Old", &format!("title: Peer {n}")),
            )
        },
        h.server.write_document(Parameters(retitle_target())),
    )
    .await
    .expect_err("the peer's change removed the anchor");
    assert!(
        err.message.contains(write::EDITED_ELSEWHERE),
        "{}",
        err.message
    );

    // No diverged clone: HEAD is the remote tip and nothing is left behind, so
    // the webhook's fetch + ff-only merge keeps working.
    let bare = h._bare.path();
    assert_eq!(rev(h.work.path(), "HEAD"), rev(bare, "master"));
    assert_eq!(git_status(&h.work), "");
    peer_push(h.peer.path(), "after.md", "# After\n");
    peer_git(h.work.path(), &["fetch", "-q", "origin", "master"]);
    peer_git(h.work.path(), &["merge", "--ff-only", "FETCH_HEAD"]);
    assert!(h.work.path().join("after.md").exists());
}

#[tokio::test]
async fn concurrent_update_schema_operations_both_land() {
    let h = concurrent_harness();
    let schema_rel = format!("docs/{}", crate::schema::SCHEMA_FILE_NAME);
    peer_push(
        h.peer.path(),
        &schema_rel,
        "fields:\n  status:\n    type: enum\n    values: [active]\n",
    );
    let other = h.server.clone();
    with_hook_once(
        crate::git::HookPoint::BeforeLock,
        move || async move {
            other
                .update_schema(Parameters(add_values_params("docs", "status", &["gamma"])))
                .await
                .expect("the concurrent schema edit lands");
        },
        h.server
            .update_schema(Parameters(add_values_params("docs", "status", &["beta"]))),
    )
    .await
    .expect("the schema edit applies to the schema as it is under the lock");

    let written = std::fs::read_to_string(h.work.path().join(&schema_rel)).unwrap();
    assert!(
        written.contains("active") && written.contains("beta") && written.contains("gamma"),
        "{written}"
    );
    assert_eq!(rev(h.work.path(), "HEAD"), rev(h._bare.path(), "master"));
}

#[tokio::test]
async fn update_schema_refused_after_its_last_conflict_still_marks_what_reached_the_remote() {
    let h = concurrent_harness();
    let schema_rel = format!("docs/{}", crate::schema::SCHEMA_FILE_NAME);
    peer_push(
        h.peer.path(),
        &schema_rel,
        "fields:\n  status:\n    type: enum\n    values: [active]\n",
    );

    // Every attempt loses its push race, so the edit is refused after the last
    // one. In each, a peer commit lands before the fetch and another before the
    // push, which makes the push lose; the sync after the dropped commit is what
    // brings both into the clone.
    let peer = h.peer.path().to_path_buf();
    let attempt = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let hook: crate::git::TestHook = Arc::new(move |point| {
        let peer = peer.clone();
        let attempt = Arc::clone(&attempt);
        Box::pin(async move {
            match point {
                crate::git::HookPoint::BeforeSync => {
                    let n = attempt.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                    peer_push(&peer, &format!("elsewhere-{n}.md"), "# Elsewhere\n");
                }
                crate::git::HookPoint::BeforePush => {
                    let n = attempt.load(std::sync::atomic::Ordering::SeqCst);
                    peer_push(&peer, &format!("late-{n}.md"), "# Late\n");
                }
                crate::git::HookPoint::BeforeLock => {}
            }
        })
    });
    let err = crate::git::TEST_HOOK
        .scope(
            hook,
            h.server
                .update_schema(Parameters(add_values_params("docs", "status", &["beta"]))),
        )
        .await
        .expect_err("every attempt lost its push, so the edit is refused");
    assert!(
        err.message.contains("edited by someone else"),
        "{}",
        err.message
    );

    // The last attempt's commits are marked like the earlier attempts' were,
    // and the clone is left on the remote tip the webhook's ff-only merge needs.
    let expected: Vec<String> = (1..=write::MAX_WRITE_ATTEMPTS)
        .flat_map(|n| [format!("elsewhere-{n}.md"), format!("late-{n}.md")])
        .collect();
    let expected: Vec<&str> = expected.iter().map(String::as_str).collect();
    crate::reindex::test_support::assert_marked_dirty(&h.server.reindex_queue, &expected);
    assert_eq!(rev(h.work.path(), "HEAD"), rev(h._bare.path(), "master"));
}

#[test]
fn write_and_delete_are_annotated_destructive_and_reads_read_only() {
    let router = KbSearchServer::tool_router();
    let annotations = |name: &str| {
        router
            .get(name)
            .unwrap_or_else(|| panic!("{name} is registered"))
            .annotations
            .clone()
            .unwrap_or_else(|| panic!("{name} has annotations"))
    };
    for name in ["write_document", "delete_document"] {
        let a = annotations(name);
        assert_eq!(a.destructive_hint, Some(true), "{name}");
        assert_eq!(a.idempotent_hint, Some(false), "{name}");
    }
    for name in ["search", "get_document", "get_schema", "changes_since"] {
        assert_eq!(annotations(name).read_only_hint, Some(true), "{name}");
    }
}

#[test]
fn write_failures_relay_no_cause() {
    let raw = "git commit failed: error: gpg failed to sign the data; HEAD is abc123";
    let create = create_edit_error_to_mcp_error(
        WriteError::PreCommitFailed {
            rolled_back: true,
            msg: raw.to_string(),
        },
        "a.md",
        true,
        Path::new("/kb"),
        None,
    );
    assert_not_saved(&create);
    assert!(!create.message.contains("gpg"), "{}", create.message);
    let delete = delete_error_to_mcp_error(
        WriteError::PreCommitFailed {
            rolled_back: false,
            msg: raw.to_string(),
        },
        "a.md",
    );
    assert_not_saved_unverified(&delete);
    assert_not_saved(&batch_write_error_to_mcp_error(
        write::BatchWriteError::PreCommitFailed { rolled_back: true },
    ));
    assert_not_saved_unverified(&move_directory_error_to_mcp_error(
        DirectoryMoveError::PreCommitFailed { rolled_back: false },
        "src",
        "dest",
    ));
    let message = create_edit_error_to_mcp_error(
        WriteError::InvalidCommitMessage {
            reason: "commit message must not contain newlines".to_string(),
        },
        "a.md",
        false,
        Path::new("/kb"),
        None,
    );
    assert_git_free("message error", &message.message);
    let credential = credential_error("GIT_PULL_TOKEN_FILE is unreadable");
    assert_git_free("credential error", &credential.message);
    assert!(
        !credential.message.contains("TOKEN"),
        "{}",
        credential.message
    );
}

#[test]
fn related_json_rounds_scores_and_is_omitted_when_empty() {
    assert!(related_json(&[]).is_none());
    let json = related_json(&[crate::write::RelatedDocument {
        path: "a.md".into(),
        score: 0.723_456_8,
    }])
    .unwrap();
    assert_eq!(json[0]["path"], "a.md");
    assert_eq!(json[0]["score"].as_f64(), Some(0.7235));
}

#[test]
fn create_result_carries_related_only_when_present() {
    let mk = |related| WriteSuccess {
        outcome: crate::write::WriteOutcome::Synced,
        sha: "abc".into(),
        rebased_paths: Vec::new(),
        diff: String::new(),
        rewritten_paths: Vec::new(),
        referencing_paths: Vec::new(),
        status_set: false,
        formatting: Default::default(),
        merged: false,
        version: Some("v".into()),
        related,
    };
    let with = create_edit_success_to_result(
        mk(vec![crate::write::RelatedDocument {
            path: "x.md".into(),
            score: 0.6,
        }]),
        "new.md",
        true,
        None,
        false,
    );
    let structured = with.structured_content.unwrap();
    assert_eq!(structured["related"][0]["path"], "x.md");
    let without = create_edit_success_to_result(mk(Vec::new()), "new.md", true, None, false);
    assert!(without.structured_content.unwrap().get("related").is_none());
}

#[test]
fn dedup_refusal_data_reports_scores_without_f32_widening_noise() {
    let err = create_edit_error_to_mcp_error(
        WriteError::DedupHit {
            duplicate_of: "docs/existing.md".to_string(),
            similarity: 0.93,
            threshold: 0.95,
        },
        "docs/new.md",
        true,
        Path::new("/kb"),
        None,
    );
    let data = err.data.expect("a dedup refusal carries structured data");
    assert_eq!(data["duplicate_of"], "docs/existing.md");
    // `json!(0.93_f32)` would widen to 0.9300000071525574.
    assert_eq!(data["similarity"], serde_json::json!(0.93));
    assert_eq!(data["threshold"], serde_json::json!(0.95));
}

#[test]
fn no_tool_description_or_input_schema_mentions_versioning() {
    let mechanics = crate::descriptions::compose_server_mechanics();
    assert_git_free("server instructions", &mechanics);
    for phrase in [true, false] {
        let overlay = crate::descriptions::compose_tool_descriptions(
            None,
            phrase,
            &crate::config::Granularity::ALL,
        );
        let server = make_overlay_test_server(overlay);
        for tool in KbSearchServer::tool_router().list_all() {
            let tool = tool_schema::self_contained(
                server.overlay_input_schema(server.overlay_description(tool)),
            );
            let description = tool.description.as_deref().unwrap_or_default();
            assert_git_free(&format!("{} description", tool.name), description);
            let schema = serde_json::Value::Object((*tool.input_schema).clone());
            assert_git_free(&format!("{} input schema", tool.name), &schema.to_string());
        }
    }
}

// -----------------------------------------------------------------------
// Pre-commit vs. post-commit failure handling (git::CommitSyncError) —
// create_document / edit_document / delete_document
// -----------------------------------------------------------------------

#[tokio::test]
async fn delete_document_precommit_failure_restores_the_file_and_reports_no_change() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let original = "---\ntitle: D\n---\n\n# Body\n";
    std::fs::write(work.path().join("doomed.md"), original).unwrap();
    git_commit_all(&work, "doomed.md", "add doomed.md");
    let head_before = head_sha(&work);

    force_git_commit_to_fail(&work);
    let (server, _config) = make_git_backed_server(&work);

    let result = server
        .delete_document(Parameters(DeleteDocumentParams {
            path: "doomed.md".to_string(),
            message: None,
            expected_version: version_at(&work, "doomed.md"),
            unlink_inbound: None,
        }))
        .await;

    let err = result.expect_err("a rejected pre-commit hook must fail the delete");
    assert_not_saved(&err);

    assert!(
        work.path().join("doomed.md").exists(),
        "the file must be restored to disk after a rolled-back pre-commit failure"
    );
    assert_eq!(
        std::fs::read_to_string(work.path().join("doomed.md")).unwrap(),
        original,
        "restored content must match what was at HEAD"
    );
    assert_eq!(
        head_before,
        head_sha(&work),
        "HEAD must not move on a rolled-back pre-commit failure"
    );
    // (#229) `delete_document` now unconditionally opens the state DB for
    // the inbound-link check, which lazily materializes `state.db`/`-shm`/
    // `-wal` under `work` — see `git_status_ignoring_state_db`'s doc
    // comment for why that is an expected, unrelated side effect rather
    // than evidence the rollback itself left something dirty.
    assert_eq!(
        git_status_ignoring_state_db(&work),
        "",
        "working tree must be clean after rollback"
    );
}

#[tokio::test]
async fn delete_document_postcommit_failure_leaves_commit_and_reports_pending_sync() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    std::fs::write(
        work.path().join("doomed.md"),
        "---\ntitle: D\n---\n\n# Body\n",
    )
    .unwrap();
    git_commit_all(&work, "doomed.md", "add doomed.md");

    let mut config = make_test_resolved_config(work.path());
    {
        let c = Arc::get_mut(&mut config).unwrap();
        c.write.dedup_enabled = false;
        // No such path — `git fetch` fails immediately, no network required, but
        // only AFTER the deletion's `git add`/`git commit` have already
        // succeeded locally.
        c.source.git_url = Some("/nonexistent/path/to/repo.git".to_string());
    }
    let server = make_write_test_server(&work, &["**/*.md".to_string()], config);

    let result = server
        .delete_document(Parameters(DeleteDocumentParams {
            path: "doomed.md".to_string(),
            message: None,
            expected_version: version_at(&work, "doomed.md"),
            unlink_inbound: None,
        }))
        .await;

    let result = result.expect("a post-commit sync failure must still report as success");
    // A saved-but-unsynced delete is, to the caller, simply a delete.
    assert_plain_success(&result);
    assert_eq!(single_representation(&result)["action"], "deleted");

    // The deletion IS a real local commit — the file must remain gone, and HEAD
    // must record the deletion. None of this is rolled back.
    assert!(
        !work.path().join("doomed.md").exists(),
        "a post-commit failure must NOT resurrect the file"
    );
    let show = std::process::Command::new("git")
        .args(["show", "--name-only", "--format=", "HEAD"])
        .current_dir(work.path())
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&show.stdout).contains("doomed.md"),
        "the deletion commit must be present in local HEAD"
    );
}

/// When there is no git repository at all, `git add` fails (`PreCommit`) and the
/// rollback attempt (`git restore`) ALSO fails, since there is nothing to restore
/// from. This is the third, worse outcome: the file is gone from disk with no
/// corresponding commit anywhere. It must be reported distinctly rather than
/// masquerading as either a clean delete or a clean no-op.
#[tokio::test]
async fn delete_document_with_no_git_repo_reports_inconsistent_state() {
    let tmp = tempfile::tempdir().unwrap();
    let sub = tmp.path().join("docs");
    std::fs::create_dir_all(&sub).unwrap();
    std::fs::write(
        sub.join("delete-me.md"),
        "---\ntitle: Delete Me\n---\n# Body",
    )
    .unwrap();

    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    let params = DeleteDocumentParams {
        path: "docs/delete-me.md".to_string(),
        message: None,
        expected_version: version_at(&tmp, "docs/delete-me.md"),
        unlink_inbound: None,
    };
    let result = server.delete_document(Parameters(params)).await;

    let err = result.expect_err("deleting with no git repo must fail");
    assert_not_saved_unverified(&err);

    // The restore could not put it back (there is no repo to restore from), so
    // the file really is gone — that IS the inconsistent state being reported.
    assert!(!sub.join("delete-me.md").exists());
}

#[tokio::test]
async fn create_document_precommit_failure_removes_the_new_file_and_reports_no_change() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let head_before = head_sha(&work);

    force_git_commit_to_fail(&work);
    let (server, _config) = make_git_backed_server(&work);

    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("docs/new.md".to_string()),
            content: Some(
                "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n"
                    .to_string(),
            ),
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_version: None,
            force_new: Some(true),
            frontmatter_patch: None,
            append: None,
            documents: None,
            archive: None,
            archive_links: None,
        }))
        .await;

    let err = result.expect_err("a rejected pre-commit hook must fail the create");
    assert_not_saved(&err);

    assert!(
        !work.path().join("docs/new.md").exists(),
        "the newly-written file must be removed on rollback — there is no HEAD \
         content for a brand-new create to fall back to"
    );
    assert_eq!(
        head_before,
        head_sha(&work),
        "HEAD must not move on a rolled-back pre-commit failure"
    );
    assert_eq!(
        git_status(&work),
        "",
        "the aborted `git add` must be unstaged too — no leftover addition that \
         could ride along on a later, unrelated commit"
    );
}

#[tokio::test]
async fn edit_document_precommit_failure_restores_previous_content_and_reports_no_change() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let original = "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
    std::fs::write(work.path().join("edit-me.md"), original).unwrap();
    git_commit_all(&work, "edit-me.md", "add edit-me.md");
    let head_before = head_sha(&work);

    force_git_commit_to_fail(&work);
    let (server, _config) = make_git_backed_server(&work);

    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("edit-me.md".to_string()),
            old_string: None,
            new_string: None,
            content: Some(
                "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New body\n"
                    .to_string(),
            ),
            message: None,
            expected_version: version_at(&work, "edit-me.md"),
            new_path: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
            documents: None,
            archive: None,
            archive_links: None,
        }))
        .await;

    let err = result.expect_err("a rejected pre-commit hook must fail the edit");
    assert_not_saved(&err);

    assert_eq!(
        std::fs::read_to_string(work.path().join("edit-me.md")).unwrap(),
        original,
        "the edit must be rolled back to the previous HEAD content"
    );
    assert_eq!(
        head_before,
        head_sha(&work),
        "HEAD must not move on a rolled-back pre-commit failure"
    );
    assert_eq!(
        git_status(&work),
        "",
        "working tree must be clean after rollback"
    );
}

#[tokio::test]
async fn edit_document_postcommit_failure_leaves_commit_and_reports_pending_sync() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let original = "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
    std::fs::write(work.path().join("edit-me.md"), original).unwrap();
    git_commit_all(&work, "edit-me.md", "add edit-me.md");

    let mut config = make_test_resolved_config(work.path());
    {
        let c = Arc::get_mut(&mut config).unwrap();
        c.write.dedup_enabled = false;
        c.source.git_url = Some("/nonexistent/path/to/repo.git".to_string());
    }
    let server = make_write_test_server(&work, &["**/*.md".to_string()], config);

    let new_content =
        "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New body\n";
    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("edit-me.md".to_string()),
            old_string: None,
            new_string: None,
            content: Some(new_content.to_string()),
            message: None,
            expected_version: version_at(&work, "edit-me.md"),
            new_path: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
            documents: None,
            archive: None,
            archive_links: None,
        }))
        .await;

    let result = result.expect("a post-commit sync failure must still report as success");
    // A saved-but-unsynced edit is, to the caller, simply an edit.
    assert_plain_success(&result);

    // The edit IS a real local commit — the new content stays on disk, and HEAD
    // records it. None of this is rolled back just because the push failed.
    assert_eq!(
        std::fs::read_to_string(work.path().join("edit-me.md")).unwrap(),
        new_content,
        "a post-commit failure must NOT revert the edit"
    );
    let show = std::process::Command::new("git")
        .args(["show", "HEAD:edit-me.md"])
        .current_dir(work.path())
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&show.stdout), new_content);
}

// -----------------------------------------------------------------------
// expected_version — the document version token
// -----------------------------------------------------------------------

#[tokio::test]
async fn get_document_reports_the_document_version() {
    let tmp = tempfile::tempdir().unwrap();
    let sub = tmp.path().join("docs");
    std::fs::create_dir_all(&sub).unwrap();
    let content = "---\ntitle: T\ntype: guide\n---\n# Body\n";
    std::fs::write(sub.join("guide.md"), content).unwrap();

    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    let result = server
        .get_document(Parameters(GetDocumentParams {
            path: "docs/guide.md".to_string(),
            ..Default::default()
        }))
        .await
        .unwrap();

    let structured = result
        .structured_content
        .expect("get_document must report a version in structured_content");
    let hash = structured["version"]
        .as_str()
        .expect("version must be a string");
    assert_eq!(
        hash,
        crate::write::document_version(content.as_bytes()),
        "a caller round-trips the version into write_document's expected_version"
    );
    assert_eq!(
        structured["content"].as_str(),
        Some(content),
        "structured_content must carry the full document: clients that prefer \
         structuredContent render only it, so a hash-only payload hides the \
         document entirely"
    );
    assert_eq!(structured["path"].as_str(), Some("docs/guide.md"));
}

#[tokio::test]
async fn full_replace_with_an_unknown_expected_version_is_refused_before_touching_the_file() {
    let tmp = tempfile::tempdir().unwrap();
    let sub = tmp.path().join("docs");
    std::fs::create_dir_all(&sub).unwrap();
    let original = "---\ntitle: Old\ntype: guide\n---\n# Old body\n";
    std::fs::write(sub.join("edit-me.md"), original).unwrap();

    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    // A hash of some OTHER content — as if the caller read the document at an
    // earlier revision.
    let stale_hash = crate::write::document_version(b"not the current content");

    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("docs/edit-me.md".to_string()),
            old_string: None,
            new_string: None,
            content: Some("---\ntitle: New\ntype: guide\n---\n# New body\n".to_string()),
            message: None,
            expected_version: Some(stale_hash),
            new_path: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
            documents: None,
            archive: None,
            archive_links: None,
        }))
        .await;

    let err = result.expect_err("a stale expected_version must be rejected");
    assert!(
        err.message.contains("edited by someone else"),
        "expected an explicit stale-read message, got: {}",
        err.message
    );
    assert!(
        err.message.contains("re-read it"),
        "expected guidance to re-read via get_document, got: {}",
        err.message
    );

    // Rejected before any write: the file on disk must be untouched.
    assert_eq!(
        std::fs::read_to_string(sub.join("edit-me.md")).unwrap(),
        original,
        "a stale expected_version must fail before the file is touched"
    );
}

#[tokio::test]
async fn full_replace_with_a_matching_expected_version_proceeds_to_a_synced_write() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let original = "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
    std::fs::write(work.path().join("edit-me.md"), original).unwrap();
    git_commit_all(&work, "edit-me.md", "add edit-me.md");
    let (server, _config) = make_git_backed_server(&work);

    let correct_hash = crate::write::document_version(original.as_bytes());
    let new_content =
        "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New body\n";

    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("edit-me.md".to_string()),
            old_string: None,
            new_string: None,
            content: Some(new_content.to_string()),
            message: None,
            expected_version: Some(correct_hash),
            new_path: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
            documents: None,
            archive: None,
            archive_links: None,
        }))
        .await;

    let result = result.expect("a correct expected_version must not block the edit");
    assert_plain_success(&result);
    assert_eq!(
        std::fs::read_to_string(work.path().join("edit-me.md")).unwrap(),
        new_content
    );
}

// -----------------------------------------------------------------------
// edit_document: new_path (move), end-to-end through the tool
// -----------------------------------------------------------------------

#[tokio::test]
async fn edit_document_move_alone_relocates_content_unchanged() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let original = "---\ntitle: Old Home\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
    std::fs::create_dir_all(work.path().join("docs")).unwrap();
    std::fs::write(work.path().join("docs/old-home.md"), original).unwrap();
    git_commit_all(&work, "docs/old-home.md", "add docs/old-home.md");
    let (server, _config) = make_git_backed_server(&work);

    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("docs/old-home.md".to_string()),
            old_string: None,
            new_string: None,
            content: None,
            message: None,
            expected_version: version_at(&work, "docs/old-home.md"),
            new_path: Some("docs/new-home.md".to_string()),
            force_new: None,
            frontmatter_patch: None,
            append: None,
            documents: None,
            archive: None,
            archive_links: None,
        }))
        .await;

    let result = result.expect("a pure move (new_path alone) must succeed");
    assert_plain_success(&result);

    assert_eq!(
        std::fs::read_to_string(work.path().join("docs/new-home.md")).unwrap(),
        original,
        "the destination must have the source's exact original content, unchanged"
    );
    assert!(
        !work.path().join("docs/old-home.md").exists(),
        "the source must be gone after a move"
    );
}

#[tokio::test]
async fn edit_document_move_combined_with_edit_relocates_transformed_content() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let original = "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
    std::fs::write(work.path().join("edit-me.md"), original).unwrap();
    git_commit_all(&work, "edit-me.md", "add edit-me.md");
    let (server, _config) = make_git_backed_server(&work);

    let new_content =
        "---\ntitle: New\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# New body\n";

    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("edit-me.md".to_string()),
            old_string: None,
            new_string: None,
            content: Some(new_content.to_string()),
            message: None,
            expected_version: version_at(&work, "edit-me.md"),
            new_path: Some("archive/edit-me.md".to_string()),
            force_new: None,
            frontmatter_patch: None,
            append: None,
            documents: None,
            archive: None,
            archive_links: None,
        }))
        .await;

    let result = result.expect("a combined move+edit must succeed");
    assert_plain_success(&result);

    assert_eq!(
        std::fs::read_to_string(work.path().join("archive/edit-me.md")).unwrap(),
        new_content,
        "the destination must hold the TRANSFORMED content, not the pre-move original"
    );
    assert!(
        !work.path().join("edit-me.md").exists(),
        "the source must be gone after a move"
    );
}

#[tokio::test]
async fn edit_document_move_onto_existing_destination_reports_the_destination_as_the_collision() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let source_content =
        "---\ntitle: Source\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Source body\n";
    let dest_content =
        "---\ntitle: Dest\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Dest body\n";
    std::fs::write(work.path().join("source.md"), source_content).unwrap();
    std::fs::write(work.path().join("dest.md"), dest_content).unwrap();
    git_commit_all(&work, "source.md", "add source.md");
    git_commit_all(&work, "dest.md", "add dest.md");
    let (server, _config) = make_git_backed_server(&work);

    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("source.md".to_string()),
            old_string: None,
            new_string: None,
            content: None,
            message: None,
            expected_version: version_at(&work, "source.md"),
            new_path: Some("dest.md".to_string()),
            force_new: None,
            frontmatter_patch: None,
            append: None,
            documents: None,
            archive: None,
            archive_links: None,
        }))
        .await;

    let err = result.expect_err("moving onto an existing destination must be rejected");
    assert!(
        err.message.contains("dest.md"),
        "error should name the destination path, got: {}",
        err.message
    );
    assert!(
        err.message.contains("destination"),
        "error should make clear it is the DESTINATION that collided, got: {}",
        err.message
    );
    assert!(
        !err.message.contains("'source.md' already exists"),
        "error must not misattribute the collision to the source, got: {}",
        err.message
    );

    // Rejected before any filesystem mutation (write_document_move checks the
    // destination's existence before writing anything) — both files, source and
    // pre-existing destination, must be untouched.
    assert_eq!(
        std::fs::read_to_string(work.path().join("source.md")).unwrap(),
        source_content
    );
    assert_eq!(
        std::fs::read_to_string(work.path().join("dest.md")).unwrap(),
        dest_content
    );
}

// -----------------------------------------------------------------------
// write_document: directory-move dispatch (path resolves to a directory)
// -----------------------------------------------------------------------

#[tokio::test]
async fn write_document_dispatches_to_directory_move_when_path_is_a_directory() {
    // `path` resolving to an existing directory dispatches write_document to a
    // whole-subtree move via `write::move_directory`, mirroring the old
    // standalone `move_directory` tool.
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    std::fs::create_dir_all(work.path().join("old-project")).unwrap();
    let doc_content = "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
    std::fs::write(work.path().join("old-project/a.md"), doc_content).unwrap();
    git_commit_all(&work, "old-project/a.md", "add old-project/a.md");
    let (server, _config) = make_git_backed_server(&work);

    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("old-project".to_string()),
            content: None,
            old_string: None,
            new_string: None,
            new_path: Some("archive/new-project".to_string()),
            message: None,
            expected_version: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
            documents: None,
            archive: None,
            archive_links: None,
        }))
        .await;

    let result = result.expect("a directory path must dispatch to a directory move");
    let structured = single_representation(&result);
    assert_eq!(
        structured["moved"],
        serde_json::json!([{"from": "old-project/a.md", "to": "archive/new-project/a.md"}]),
        "must report the directory move, not a single-document write: {structured}"
    );
    assert!(
        !work.path().join("old-project/a.md").exists(),
        "the source directory's document must be gone after the move"
    );
    assert_eq!(
        std::fs::read_to_string(work.path().join("archive/new-project/a.md")).unwrap(),
        doc_content,
    );
}

#[tokio::test]
async fn write_document_directory_move_requires_new_path() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("some-dir")).unwrap();
    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("some-dir".to_string()),
            content: None,
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_version: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
            documents: None,
            archive: None,
            archive_links: None,
        }))
        .await;

    let err = result.expect_err("a directory move without new_path must be rejected");
    assert!(
        err.message.contains("new_path"),
        "error should name new_path as required, got: {}",
        err.message
    );
}

#[tokio::test]
async fn write_document_directory_move_rejects_single_document_fields() {
    // Every field that only makes sense for a single document must be
    // rejected, by name, when `path` resolves to a directory.
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("some-dir")).unwrap();
    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    let base = || WriteDocumentParams {
        path: Some("some-dir".to_string()),
        content: None,
        old_string: None,
        new_string: None,
        new_path: Some("other-dir".to_string()),
        message: None,
        expected_version: None,
        force_new: None,
        frontmatter_patch: None,
        append: None,
        documents: None,
        archive: None,
        archive_links: None,
    };

    let cases: Vec<(&str, WriteDocumentParams)> = vec![
        (
            "content",
            WriteDocumentParams {
                content: Some("x".to_string()),
                ..base()
            },
        ),
        (
            "old_string",
            WriteDocumentParams {
                old_string: Some("x".to_string()),
                ..base()
            },
        ),
        (
            "new_string",
            WriteDocumentParams {
                new_string: Some("x".to_string()),
                ..base()
            },
        ),
        (
            "expected_version",
            WriteDocumentParams {
                expected_version: Some("deadbeef".to_string()),
                ..base()
            },
        ),
        (
            "force_new",
            WriteDocumentParams {
                force_new: Some(true),
                ..base()
            },
        ),
        (
            "frontmatter_patch",
            WriteDocumentParams {
                frontmatter_patch: Some(vec![FrontmatterPatchOp {
                    operation: "set_field".to_string(),
                    field: "status".to_string(),
                    value: Some(serde_json::json!("active")),
                    values: None,
                }]),
                ..base()
            },
        ),
        (
            "append",
            WriteDocumentParams {
                append: Some("more text".to_string()),
                ..base()
            },
        ),
    ];

    for (field, params) in cases {
        let result = server.write_document(Parameters(params)).await;
        let err = result.expect_err(&format!(
            "a directory move with {field} set must be rejected"
        ));
        assert!(
            err.message.contains(field),
            "error should name '{field}' as the rejected field, got: {}",
            err.message
        );
    }
}

// -----------------------------------------------------------------------
// write_document: batch dispatch (#180)
// -----------------------------------------------------------------------

#[tokio::test]
async fn write_document_batch_rejects_single_document_fields_alongside_documents() {
    let tmp = tempfile::tempdir().unwrap();
    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    let one_doc = || {
        Some(vec![BatchDocumentInput {
            path: "docs/x.md".to_string(),
            content: Some("---\ntitle: X\n---\n\n# X\n".to_string()),
            old_string: None,
            new_string: None,
            frontmatter_patch: None,
            append: None,
            expected_version: None,
            force_new: None,
        }])
    };
    let base = || WriteDocumentParams {
        path: None,
        content: None,
        old_string: None,
        new_string: None,
        new_path: None,
        message: None,
        expected_version: None,
        force_new: None,
        frontmatter_patch: None,
        append: None,
        documents: one_doc(),
        archive: None,
        archive_links: None,
    };

    let cases: Vec<(&str, WriteDocumentParams)> = vec![
        (
            "path",
            WriteDocumentParams {
                path: Some("docs/y.md".to_string()),
                ..base()
            },
        ),
        (
            "content",
            WriteDocumentParams {
                content: Some("x".to_string()),
                ..base()
            },
        ),
        (
            "new_path",
            WriteDocumentParams {
                new_path: Some("docs/z.md".to_string()),
                ..base()
            },
        ),
    ];

    for (field, params) in cases {
        let result = server.write_document(Parameters(params)).await;
        let err = result.expect_err(&format!("{field} set alongside documents must be rejected"));
        assert!(
            err.message.contains(field),
            "error should name '{field}' as the rejected field, got: {}",
            err.message
        );
    }
}

#[tokio::test]
async fn write_document_batch_rejects_more_than_the_cap() {
    let tmp = tempfile::tempdir().unwrap();
    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    let documents: Vec<BatchDocumentInput> = (0..(write::MAX_BATCH_DOCUMENTS + 1))
        .map(|i| BatchDocumentInput {
            path: format!("docs/over-cap-{i}.md"),
            content: Some(format!("---\ntitle: {i}\n---\n\n# {i}\n")),
            old_string: None,
            new_string: None,
            frontmatter_patch: None,
            append: None,
            expected_version: None,
            force_new: None,
        })
        .collect();

    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: None,
            content: None,
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_version: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
            documents: Some(documents),
            archive: None,
            archive_links: None,
        }))
        .await;
    let err = result.expect_err("a batch over the size cap must be rejected");
    assert!(
        err.message
            .contains(&write::MAX_BATCH_DOCUMENTS.to_string()),
        "error should report the cap: {}",
        err.message
    );
}

#[tokio::test]
async fn write_document_batch_happy_path_lands_every_document_in_one_commit() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let (server, _config) = make_git_backed_server(&work);

    let documents = vec![
        BatchDocumentInput {
            path: "docs/batch-a.md".to_string(),
            content: Some(
                "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# A\n".to_string(),
            ),
            old_string: None,
            new_string: None,
            frontmatter_patch: None,
            append: None,
            expected_version: None,
            force_new: Some(true),
        },
        BatchDocumentInput {
            path: "docs/batch-b.md".to_string(),
            content: Some(
                "---\ntitle: B\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# B\n".to_string(),
            ),
            old_string: None,
            new_string: None,
            frontmatter_patch: None,
            append: None,
            expected_version: None,
            force_new: Some(true),
        },
    ];

    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: None,
            content: None,
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_version: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
            documents: Some(documents),
            archive: None,
            archive_links: None,
        }))
        .await
        .expect("batch write should succeed");

    assert_plain_success(&result);
    let structured = single_representation(&result);
    let docs = structured["documents"].as_array().unwrap();
    assert_eq!(docs.len(), 2);
    assert_eq!(docs[0]["path"], "docs/batch-a.md");
    assert_eq!(docs[0]["action"], "created");
    assert!(
        docs[0].get("diff").is_none(),
        "a create echoes no diff: {structured}"
    );
    assert_eq!(docs[1]["path"], "docs/batch-b.md");

    crate::reindex::test_support::assert_marked_dirty(
        &server.reindex_queue,
        &["docs/batch-a.md", "docs/batch-b.md"],
    );
}

#[tokio::test]
async fn write_document_batch_reports_every_failing_document_at_once() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let (server, _config) = make_git_backed_server(&work);

    let documents = vec![
        BatchDocumentInput {
            path: "docs/ok.md".to_string(),
            content: Some(
                "---\ntitle: Ok\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Ok\n".to_string(),
            ),
            old_string: None,
            new_string: None,
            frontmatter_patch: None,
            append: None,
            expected_version: None,
            force_new: Some(true),
        },
        // Neither content, old_string/new_string, frontmatter_patch, nor
        // append — parse_edit_mode must reject this the same way it
        // rejects a single-document call with no edit mode at all.
        BatchDocumentInput {
            path: "docs/no-edit-mode.md".to_string(),
            content: None,
            old_string: None,
            new_string: None,
            frontmatter_patch: None,
            append: None,
            expected_version: None,
            force_new: None,
        },
    ];

    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: None,
            content: None,
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_version: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
            documents: Some(documents),
            archive: None,
            archive_links: None,
        }))
        .await;
    let err = result.expect_err("a batch entry with no edit mode must be rejected");
    assert!(
        err.message.contains("docs/no-edit-mode.md"),
        "error should name the offending entry: {}",
        err.message
    );

    // Nothing should have been written for the OTHER (valid) document either
    // — this is a pre-flight parse failure in mcp/, before
    // `write::write_documents_batch` is ever called.
    assert!(!work.path().join("docs/ok.md").exists());
    assert!(!work.path().join("docs/no-edit-mode.md").exists());
}

// -----------------------------------------------------------------------
// write_document: create-path guards (surgical/content on a nonexistent path)
// -----------------------------------------------------------------------

#[tokio::test]
async fn write_document_surgical_edit_on_nonexistent_path_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("docs/nonexistent.md".to_string()),
            content: None,
            old_string: Some("old".to_string()),
            new_string: Some("new".to_string()),
            new_path: None,
            message: None,
            expected_version: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
            documents: None,
            archive: None,
            archive_links: None,
        }))
        .await;

    let err = result.expect_err("a surgical edit against a nonexistent path must be rejected");
    assert!(
        err.message
            .contains("cannot surgically edit a document that does not exist"),
        "got: {}",
        err.message
    );
}

#[tokio::test]
async fn write_document_overlong_path_rejected() {
    // Mirrors `get_document`'s overlong-path test: `write_document` must reject
    // the same class of input before `resolve_safe_write_path`, the
    // include-pattern check, or git staging ever see it — see #153.
    let tmp = tempfile::tempdir().unwrap();
    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    let err = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("a".repeat(MAX_PATH_LEN + 1)),
            content: Some("---\ntitle: Test\n---\n# Body".to_string()),
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_version: None,
            force_new: Some(true),
            frontmatter_patch: None,
            append: None,
            documents: None,
            archive: None,
            archive_links: None,
        }))
        .await
        .expect_err("overlong path should return Err");
    assert!(
        err.message.contains("exceeds maximum length"),
        "error should name the length problem, got: {}",
        err.message
    );
}

#[tokio::test]
async fn write_document_overlong_new_path_rejected() {
    // Same guard, applied to `new_path` (used for moves) rather than `path` —
    // see #153.
    let tmp = tempfile::tempdir().unwrap();
    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);
    std::fs::write(
        tmp.path().join("existing.md"),
        "---\ntitle: Old\n---\n# Old\n",
    )
    .unwrap();

    let err = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("existing.md".to_string()),
            content: None,
            old_string: None,
            new_string: None,
            new_path: Some("a".repeat(MAX_PATH_LEN + 1)),
            message: None,
            expected_version: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
            documents: None,
            archive: None,
            archive_links: None,
        }))
        .await
        .expect_err("overlong new_path should return Err");
    assert!(
        err.message.contains("exceeds maximum length"),
        "error should name the length problem, got: {}",
        err.message
    );
}

#[tokio::test]
async fn write_document_create_without_content_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("docs/nonexistent.md".to_string()),
            content: None,
            old_string: None,
            new_string: None,
            new_path: None,
            message: None,
            expected_version: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
            documents: None,
            archive: None,
            archive_links: None,
        }))
        .await;

    let err = result.expect_err("a create with no content must be rejected");
    assert!(
        err.message
            .contains("content is required to create a new document"),
        "got: {}",
        err.message
    );
}

#[tokio::test]
async fn write_document_new_path_on_a_create_is_rejected() {
    // Not explicitly specced, but forced by `write::WriteRequest::dest_path`'s
    // own contract: `is_create` must be `false` whenever `dest_path` is
    // `Some`, since a create has no source to move from. write_document must
    // reject this combination outright rather than silently drop new_path or
    // pass an invalid request down to the write pipeline.
    let tmp = tempfile::tempdir().unwrap();
    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    let result = server
        .write_document(Parameters(WriteDocumentParams {
            path: Some("docs/nonexistent.md".to_string()),
            content: Some("---\ntitle: Test\n---\n# Body".to_string()),
            old_string: None,
            new_string: None,
            new_path: Some("docs/elsewhere.md".to_string()),
            message: None,
            expected_version: None,
            force_new: None,
            frontmatter_patch: None,
            append: None,
            documents: None,
            archive: None,
            archive_links: None,
        }))
        .await;

    let err = result.expect_err("new_path on a create must be rejected");
    assert!(err.message.contains("new_path"), "got: {}", err.message);
}

// -----------------------------------------------------------------------
// update_schema / write_raw_file — the same PreCommit/PostCommit rollback
// treatment as create_document/edit_document/delete_document above, applied to
// the one write path (`write_raw_file`, used only by `update_schema`) PR #93
// deferred. See `write_raw_file`'s doc comment for the rollback rules and
// `update_schema`'s comment above its call to it for why a rolled-back write
// must skip both the schema-cache rebuild and `reindex::mark_full`.
// -----------------------------------------------------------------------

#[test]
fn invalid_schema_in_source_names_the_directory_reason_and_fix() {
    let err = move_directory_error_to_mcp_error(
        DirectoryMoveError::InvalidSchemaInSource {
            path: "src/.kb-schema.yaml".to_string(),
            reason: "fields: expected a map".to_string(),
        },
        "src",
        "dest",
    );
    assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
    for needle in ["'src/'", "fields: expected a map", "operator must repair"] {
        assert!(err.message.contains(needle), "{needle}: {}", err.message);
    }
    assert!(!err.message.contains("git"), "{}", err.message);
    assert!(!err.message.contains("schema.yaml"), "{}", err.message);
    let data = err.data.expect("structured data");
    assert_eq!(data["invalid_schema_dir"], "src/");
    assert!(data.get("invalid_schema_file").is_none());
}

#[test]
fn validation_error_names_the_destination_when_a_schema_file_relocated() {
    // The crux requirement: when the source subtree's own schema file is
    // moving too, the MCP-facing error must make clear the DESTINATION's
    // (re-parented) cascade is why documents that were valid at the source
    // now fail, not leave the caller staring at a bare validation failure.
    let err = move_directory_error_to_mcp_error(
        DirectoryMoveError::Validation {
            failures: vec![(
                "dest/target/sub/a.md".to_string(),
                crate::validate::ValidationResult {
                    file_path: "dest/target/sub/a.md".to_string(),
                    valid: false,
                    errors: vec!["missing required field 'extra_required'".to_string()],
                    field_errors: vec![],
                },
            )],
            moved_schema_files: vec![(
                format!("src/sub/{}", crate::schema::SCHEMA_FILE_NAME),
                format!("dest/target/sub/{}", crate::schema::SCHEMA_FILE_NAME),
            )],
        },
        "src",
        "dest/target",
    );

    assert!(
        err.message.contains("DESTINATION"),
        "must name the destination cascade as the reason: {}",
        err.message
    );
    assert!(
        err.message.contains("dest/target"),
        "must name the destination path: {}",
        err.message
    );
    assert!(
        err.message.contains("re-parent"),
        "must explain the schema file re-parented onto the destination: {}",
        err.message
    );
    assert!(
        err.message.contains("src/sub/ -> dest/target/sub/"),
        "must name which directory schema relocated: {}",
        err.message
    );
    assert!(!err.message.contains("schema.yaml"), "{}", err.message);
    let data = err.data.expect("structured data");
    assert_eq!(data["moved_schema_dirs"][0]["from"], "src/sub/");
    assert_eq!(data["moved_schema_dirs"][0]["to"], "dest/target/sub/");
    assert!(data.get("moved_schema_files").is_none());
}

#[test]
fn validation_error_without_a_relocated_schema_file_omits_the_reparenting_note() {
    let err = move_directory_error_to_mcp_error(
        DirectoryMoveError::Validation {
            failures: vec![(
                "dest/a.md".to_string(),
                crate::validate::ValidationResult {
                    file_path: "dest/a.md".to_string(),
                    valid: false,
                    errors: vec!["missing required field 'x'".to_string()],
                    field_errors: vec![],
                },
            )],
            moved_schema_files: vec![],
        },
        "src",
        "dest",
    );

    assert!(
        !err.message.contains("re-parent"),
        "no schema file moved, so there is nothing to explain re-parenting for: {}",
        err.message
    );
}

// --- build_chunk_search_payload / build_grouped_search_payload ---
//
// These drive the response-assembly seam directly with hand-built
// `SearchResult`/`GroupedDocument` values — no network, no mocked
// `KbSearchServer`, none of `EmbedClient`'s retry/backoff to defeat.
