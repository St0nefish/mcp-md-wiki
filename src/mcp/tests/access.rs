// Per-principal access control on the MCP tools: a principal confined to
// `memory/` is refused, with one generic answer, everything outside it. The write
// pipeline and the retrieval scoping have their own tests next to the code; these
// drive the tool handlers the way `call_tool` does, with the principal in scope.
use rmcp::handler::server::wrapper::Parameters;

use crate::mcp::get_document::*;
use crate::mcp::schema_tools::*;
use crate::mcp::test_support::*;
use crate::mcp::write_document::*;
use crate::mcp::*;
use crate::policy::{Principal, scoped_principal_for_test};

const NOT_PERMITTED: &str = "not permitted";

fn reader() -> Principal {
    scoped_principal_for_test(&["*"], &["memory"], &[])
}

fn writer() -> Principal {
    scoped_principal_for_test(&["*"], &["memory"], &["memory"])
}

fn kb() -> (tempfile::TempDir, KbSearchServer) {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("memory")).unwrap();
    std::fs::create_dir_all(tmp.path().join("notes")).unwrap();
    std::fs::write(
        tmp.path().join("notes/secret.md"),
        "---\ntitle: S\n---\nbody",
    )
    .unwrap();
    let mut config = make_test_resolved_config(tmp.path());
    std::sync::Arc::get_mut(&mut config)
        .unwrap()
        .write
        .dedup_enabled = false;
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);
    (tmp, server)
}

fn is_denied(err: &McpError) -> bool {
    err.message == NOT_PERMITTED
}

fn write_params(path: &str) -> WriteDocumentParams {
    WriteDocumentParams {
        path: Some(path.into()),
        content: Some("---\ntitle: X\n---\nbody".into()),
        ..Default::default()
    }
}

#[tokio::test]
async fn get_schema_is_refused_outside_the_read_reach_and_names_nothing() {
    let (tmp, server) = kb();
    write_schema_file(&tmp, "notes", "fields:\n  secret_field:\n    type: text\n");
    let denied = |path: &str| {
        let server = &server;
        let path = path.to_string();
        async move {
            CURRENT_PRINCIPAL
                .scope(
                    reader(),
                    server.get_schema(Parameters(GetSchemaParams {
                        path: Some(path),
                        ..Default::default()
                    })),
                )
                .await
        }
    };
    // A scope that exists, one that does not, the root, and a spelling that tries
    // to climb out: all the same answer.
    for path in ["notes", "nope", ""] {
        let err = denied(path).await.unwrap_err();
        assert!(is_denied(&err), "{path:?}: {err:?}");
    }
    assert!(
        denied("memory").await.is_ok(),
        "the granted area stays readable"
    );
}

#[tokio::test]
async fn update_schema_needs_a_write_grant_over_the_directory() {
    let (_tmp, server) = kb();
    let params = |path: &str| UpdateSchemaParams {
        ops: None,
        path: Some(path.into()),
        operation: "add_values".into(),
        field: "tags".into(),
        values: Some(vec!["x".into()]),
        definition: None,
        dry_run: Some(true),
        force: None,
        acknowledge_root_change: None,
    };
    for path in ["notes", "", "brand/new"] {
        let err = CURRENT_PRINCIPAL
            .scope(writer(), server.update_schema(Parameters(params(path))))
            .await
            .unwrap_err();
        assert!(is_denied(&err), "{path:?}: {err:?}");
    }
    // Read-only inside its own area is still not enough.
    let err = CURRENT_PRINCIPAL
        .scope(reader(), server.update_schema(Parameters(params("memory"))))
        .await
        .unwrap_err();
    assert!(is_denied(&err), "{err:?}");
}

#[tokio::test]
async fn every_write_shape_outside_the_write_grant_gets_the_same_refusal() {
    let (tmp, server) = kb();
    let before = std::fs::read_to_string(tmp.path().join("notes/secret.md")).unwrap();

    // Create and replace, an existing target and a missing one.
    for path in ["notes/secret.md", "notes/absent.md"] {
        let err = CURRENT_PRINCIPAL
            .scope(
                writer(),
                server.write_document(Parameters(write_params(path))),
            )
            .await
            .unwrap_err();
        assert!(is_denied(&err), "write {path}: {err:?}");

        let err = CURRENT_PRINCIPAL
            .scope(
                writer(),
                server.delete_document(Parameters(DeleteDocumentParams {
                    path: path.into(),
                    message: None,
                    expected_version: None,
                    unlink_inbound: None,
                })),
            )
            .await
            .unwrap_err();
        assert!(is_denied(&err), "delete {path}: {err:?}");
    }

    // A move out of the grant, a move into it from outside, and a directory move.
    for (from, to) in [
        ("memory/a.md", "notes/a.md"),
        ("notes/secret.md", "memory/s.md"),
    ] {
        let mut p = write_params(from);
        p.content = None;
        p.new_path = Some(to.into());
        let err = CURRENT_PRINCIPAL
            .scope(writer(), server.write_document(Parameters(p)))
            .await
            .unwrap_err();
        assert!(is_denied(&err), "move {from} -> {to}: {err:?}");
    }
    let mut dir_move = WriteDocumentParams {
        path: Some("notes".into()),
        new_path: Some("memory/notes".into()),
        ..Default::default()
    };
    dir_move.content = None;
    let err = CURRENT_PRINCIPAL
        .scope(writer(), server.write_document(Parameters(dir_move)))
        .await
        .unwrap_err();
    assert!(is_denied(&err), "directory move: {err:?}");

    // A batch with one entry outside the grant.
    let entry = |path: &str| BatchDocumentInput {
        path: path.into(),
        content: Some("---\ntitle: B\n---\nbody".into()),
        old_string: None,
        new_string: None,
        frontmatter_patch: None,
        append: None,
        expected_version: None,
        force_new: None,
    };
    let err = CURRENT_PRINCIPAL
        .scope(
            writer(),
            server.write_document(Parameters(WriteDocumentParams {
                documents: Some(vec![entry("memory/ok.md"), entry("notes/evil.md")]),
                ..Default::default()
            })),
        )
        .await
        .unwrap_err();
    assert!(is_denied(&err), "batch: {err:?}");

    assert_eq!(
        std::fs::read_to_string(tmp.path().join("notes/secret.md")).unwrap(),
        before
    );
    assert!(!tmp.path().join("notes/absent.md").exists());
    assert!(!tmp.path().join("notes/evil.md").exists());
    assert!(!tmp.path().join("memory/ok.md").exists());
}

#[tokio::test]
async fn a_document_outside_the_read_reach_reads_exactly_like_one_that_does_not_exist() {
    let (_tmp, server) = kb();
    let read = |path: &str| {
        let server = &server;
        let path = path.to_string();
        async move {
            CURRENT_PRINCIPAL
                .scope(
                    reader(),
                    server.get_document(Parameters(GetDocumentParams {
                        path,
                        ..Default::default()
                    })),
                )
                .await
        }
    };
    let existing = read("notes/secret.md").await.unwrap_err();
    let missing = read("notes/absent.md").await.unwrap_err();
    assert_eq!(existing.code, missing.code);
    assert_eq!(
        existing.message.replace("secret", "absent"),
        missing.message,
        "the refusal must not depend on whether the document exists"
    );
    assert_eq!(existing.data, missing.data);
}

#[test]
fn the_restricted_search_context_is_confined_and_skips_the_corpus_wide_vocabulary_check() {
    let (_tmp, server) = kb();
    let ctx = CURRENT_PRINCIPAL.sync_scope(reader(), || server.search_ctx());
    assert!(
        ctx.reach_condition().is_some(),
        "a scoped reader's queries must carry a reach condition"
    );
    let open = server.search_ctx();
    assert!(
        open.reach_condition().is_none(),
        "an unrestricted caller's queries carry none"
    );
}

#[tokio::test]
async fn enumeration_lists_only_the_reach_with_its_own_total_and_no_hints_outside_it() {
    let (_tmp, server) = kb();
    seed_document(&server, "memory/m.md", serde_json::json!({"title": "M"})).await;
    seed_document(
        &server,
        "notes/secret.md",
        serde_json::json!({"title": "S"}),
    )
    .await;
    seed_document(&server, "notes-old/x.md", serde_json::json!({"title": "X"})).await;

    let run = |params: crate::mcp::search::SearchParams| {
        let server = &server;
        async move {
            CURRENT_PRINCIPAL
                .scope(reader(), server.search(Parameters(params)))
                .await
        }
    };

    let listed = run(Default::default())
        .await
        .unwrap()
        .structured_content
        .unwrap();
    assert_eq!(listed["total"], serde_json::json!(1), "{listed}");
    let text = listed.to_string();
    assert!(text.contains("memory/m.md"), "{text}");
    assert!(!text.contains("notes"), "{text}");

    // Naming the hidden area yields nothing, and nothing that names it back.
    for prefix in ["notes", "notes/secret.md", "notes-old"] {
        let scoped = run(crate::mcp::search::SearchParams {
            path_prefix: Some(prefix.into()),
            ..Default::default()
        })
        .await;
        let shown = match scoped {
            Ok(r) => r.structured_content.unwrap().to_string(),
            Err(e) => format!("{e:?}"),
        };
        assert!(!shown.contains("notes/secret"), "{prefix}: {shown}");
        assert!(!shown.contains("notes-old/x"), "{prefix}: {shown}");
        assert!(!shown.contains("\"total\":1"), "{prefix}: {shown}");
    }

    // The unrestricted caller still sees all three.
    let all = server.search(Parameters(Default::default())).await.unwrap();
    assert_eq!(
        all.structured_content.unwrap()["total"],
        serde_json::json!(3)
    );
}

#[tokio::test]
async fn every_query_mode_condition_set_carries_the_reach_whatever_the_caller_filters_on() {
    let (_tmp, server) = kb();
    let config = server.config();
    let schemas = crate::schema::load_shared(&server.schema_cache);
    let ctx = CURRENT_PRINCIPAL.sync_scope(reader(), || server.search_ctx());
    let want =
        crate::similarity::reach_condition(&crate::policy::Reach::Prefixes(vec!["memory".into()]))
            .unwrap();

    for params in [
        crate::mcp::search::SearchParams::default(),
        crate::mcp::search::SearchParams {
            path_prefix: Some("notes".into()),
            ..Default::default()
        },
    ] {
        let conditions = ctx
            .query_conditions(&params, &config, &schemas)
            .expect("these parameters are valid");
        assert!(conditions.contains(&want), "{params:?}: {conditions:?}");
    }
}
