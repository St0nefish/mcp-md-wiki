use rmcp::{ErrorData as McpError, handler::server::wrapper::Parameters};

use crate::mcp::get_document::*;
use crate::mcp::search::*;
use crate::mcp::test_support::*;
use crate::mcp::*;
use crate::retrieval;
#[tokio::test]
async fn get_document_rejects_overlong_path() {
    let tmp = tempfile::tempdir().unwrap();
    let instructions = Arc::new(RwLock::new("test".to_string()));
    let config = crate::config::ResolvedQdrantConfig {
        url: "http://localhost:6334".into(),
        collection: "test".into(),
    };
    let qdrant = Arc::new(QdrantStore::new(&config).unwrap());
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
    let server = KbSearchServer::new(
        embed,
        qdrant,
        "test".into(),
        tmp.path().to_path_buf(),
        &["**/*.md".to_string()],
        instructions,
        crate::config::shared_config(make_test_resolved_config(tmp.path())),
        empty_test_schema_cache(),
        None,
        Arc::new(crate::reindex::ReindexQueue::new()),
        empty_test_description_overlay(),
    )
    .unwrap();

    let overlong_path = "a".repeat(MAX_PATH_LEN + 1);
    let params = GetDocumentParams {
        path: overlong_path,
        ..Default::default()
    };
    let result = server.get_document(Parameters(params)).await;
    assert!(result.is_err(), "overlong path should return an error");
}

// --- get_document line ranges -------------------------------------------

/// Numbered lines so a failed assertion names the line it actually got.
const RANGE_DOC: &str = "l1\nl2\nl3\nl4\nl5\n";

/// The one representation every tool result carries: `structured_content`,
/// with a single text block that is exactly its compact JSON serialization
/// (so a text-only client reads the same facts). Returns the structured value.
pub(super) fn single_representation(result: &CallToolResult) -> serde_json::Value {
    let structured = result
        .structured_content
        .clone()
        .expect("every tool result carries structured_content");
    assert_eq!(result.content.len(), 1, "one content block: {result:?}");
    let text = match result.content[0].as_text() {
        Some(t) => t.text.clone(),
        other => panic!("expected a text content block, got {other:?}"),
    };
    assert_eq!(
        text,
        structured.to_string(),
        "text must be the structured JSON"
    );
    assert!(
        !text.contains("\n  "),
        "compact, not pretty-printed: {text}"
    );
    structured
}

/// Read `range_doc.md` through the real handler and return
/// (`content`, structured_content).
async fn get_range(
    server: &KbSearchServer,
    start_line: Option<usize>,
    end_line: Option<usize>,
) -> Result<(String, serde_json::Value), McpError> {
    let result = server
        .get_document(Parameters(GetDocumentParams {
            path: "range_doc.md".into(),
            start_line,
            end_line,
            ..Default::default()
        }))
        .await?;
    let structured = single_representation(&result);
    let content = structured["content"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    Ok((content, structured))
}

fn range_test_server(tmp: &tempfile::TempDir) -> KbSearchServer {
    std::fs::write(tmp.path().join("range_doc.md"), RANGE_DOC).unwrap();
    schema_tool_server(tmp)
}

async fn get_with_history(
    server: &KbSearchServer,
    history: Option<usize>,
) -> (String, serde_json::Value) {
    let result = server
        .get_document(Parameters(GetDocumentParams {
            path: "range_doc.md".into(),
            history,
            ..Default::default()
        }))
        .await
        .unwrap();
    let structured = single_representation(&result);
    (structured.to_string(), structured)
}

#[tokio::test]
async fn get_document_history_reports_provenance_for_this_document_only() {
    let tmp = tempfile::tempdir().unwrap();
    let server = range_test_server(&tmp);
    run_git(tmp.path(), &["init", "-q"]);
    run_git(tmp.path(), &["add", "range_doc.md"]);
    run_git(
        tmp.path(),
        &[
            "commit",
            "-q",
            "-m",
            "docs: add range_doc.md\n\nTool: mcp-md-wiki\nOperation: write_document",
        ],
    );
    std::fs::write(tmp.path().join("other.md"), "x").unwrap();
    run_git(tmp.path(), &["add", "other.md"]);
    run_git(tmp.path(), &["commit", "-q", "-m", "hand edit of other"]);

    let (text, structured) = get_with_history(&server, Some(10)).await;

    let history = &structured["history"];
    let changes = history["changes"].as_array().unwrap();
    assert_eq!(changes.len(), 1, "other.md's change must not appear");
    let change = changes[0].as_object().unwrap();
    assert_eq!(change["operation"], "write_document");
    assert_eq!(change["subject"], "docs: add range_doc.md");
    assert_eq!(change["author"], "Test");
    assert!(
        change["date"].as_str().unwrap().ends_with('Z'),
        "{change:?}"
    );
    // Only what a caller can use: no revision id, email or provenance trailer.
    let mut keys: Vec<&str> = change.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, vec!["author", "date", "operation", "subject"]);
    assert!(history.get("truncated").is_none(), "{history}");
    assert!(history.get("available").is_none(), "{history}");
    assert!(!text.contains("t@localhost"), "{text}");
    assert_git_free("get_document history", &text);
}

#[tokio::test]
async fn changes_since_entries_carry_no_revision_id() {
    let tmp = tempfile::tempdir().unwrap();
    let server = range_test_server(&tmp);
    run_git(tmp.path(), &["init", "-q"]);
    run_git(tmp.path(), &["add", "range_doc.md"]);
    run_git(
        tmp.path(),
        &[
            "commit",
            "-q",
            "-m",
            "docs: add range_doc.md\n\nTool: mcp-md-wiki\nOperation: write_document",
        ],
    );

    let result = server
        .changes_since(Parameters(ChangesSinceParams {
            since: Some("2000-01-01".into()),
            ..Default::default()
        }))
        .await
        .unwrap();
    let structured = single_representation(&result);

    let changes = structured["changes"].as_array().unwrap();
    assert_eq!(changes.len(), 1, "{structured}");
    let mut keys: Vec<&str> = changes[0]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec!["author", "date", "operation", "path", "status", "subject"]
    );
    assert!(structured["head"].as_str().is_some(), "{structured}");
}

#[tokio::test]
async fn changes_since_rejects_over_long_parameters() {
    let tmp = tempfile::tempdir().unwrap();
    let server = range_test_server(&tmp);
    let long = "a".repeat(MAX_PATH_LEN + 1);
    for params in [
        ChangesSinceParams {
            since: Some(long.clone()),
            ..Default::default()
        },
        ChangesSinceParams {
            cursor: Some(long.clone()),
            ..Default::default()
        },
        ChangesSinceParams {
            since: Some("2000-01-01".into()),
            path_prefix: Some(long.clone()),
            ..Default::default()
        },
    ] {
        let err = server.changes_since(Parameters(params)).await.unwrap_err();
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        assert!(err.message.contains("maximum length"), "{}", err.message);
    }
}

#[tokio::test]
async fn get_document_history_is_clamped_and_reports_truncation() {
    let tmp = tempfile::tempdir().unwrap();
    let server = range_test_server(&tmp);
    run_git(tmp.path(), &["init", "-q"]);
    for n in 0..3 {
        std::fs::write(tmp.path().join("range_doc.md"), format!("rev {n}\n")).unwrap();
        run_git(tmp.path(), &["add", "range_doc.md"]);
        run_git(tmp.path(), &["commit", "-q", "-m", &format!("rev {n}")]);
    }

    // 0 clamps up to 1, which leaves older changes unreported.
    let (text, structured) = get_with_history(&server, Some(0)).await;
    assert_eq!(
        structured["history"]["changes"].as_array().unwrap().len(),
        1
    );
    assert_eq!(structured["history"]["truncated"], true);
    assert!(
        structured["history"]["changes"][0]
            .get("operation")
            .is_none()
    );
    assert_git_free("get_document history", &text);
}

#[tokio::test]
async fn get_document_history_degrades_without_a_git_repository() {
    let tmp = tempfile::tempdir().unwrap();
    let server = range_test_server(&tmp);

    let (text, structured) = get_with_history(&server, Some(5)).await;

    assert_eq!(
        structured["history"],
        serde_json::json!({ "available": false })
    );
    assert_git_free("get_document history", &text);
}

#[tokio::test]
async fn get_document_without_history_adds_no_history_field() {
    let tmp = tempfile::tempdir().unwrap();
    let server = range_test_server(&tmp);

    let (_, structured) = get_with_history(&server, None).await;

    assert_eq!(structured["content"], RANGE_DOC);
    assert!(structured.get("history").is_none());
}

#[tokio::test]
async fn get_document_without_a_range_serves_the_whole_document() {
    let tmp = tempfile::tempdir().unwrap();
    let server = range_test_server(&tmp);

    let (text, structured) = get_range(&server, None, None).await.unwrap();

    assert_eq!(text, RANGE_DOC);
    assert_eq!(structured["total_lines"], 5);
    for key in ["start_line", "end_line", "partial"] {
        assert!(
            structured.get(key).is_none(),
            "a full read carries no range or partial flag: {structured}"
        );
    }
}

#[tokio::test]
async fn get_document_serves_an_inclusive_line_range() {
    let tmp = tempfile::tempdir().unwrap();
    let server = range_test_server(&tmp);

    let (text, structured) = get_range(&server, Some(2), Some(4)).await.unwrap();

    assert_eq!(text, "l2\nl3\nl4\n");
    assert_eq!(structured["start_line"], 2);
    assert_eq!(structured["end_line"], 4);
    assert_eq!(structured["total_lines"], 5);
    assert_eq!(structured["partial"], true);
}

#[tokio::test]
async fn get_document_reads_to_eof_with_only_a_start_line() {
    let tmp = tempfile::tempdir().unwrap();
    let server = range_test_server(&tmp);

    let (text, structured) = get_range(&server, Some(4), None).await.unwrap();

    assert_eq!(text, "l4\nl5\n");
    assert_eq!(structured["end_line"], 5);
}

#[tokio::test]
async fn get_document_reads_from_the_top_with_only_an_end_line() {
    let tmp = tempfile::tempdir().unwrap();
    let server = range_test_server(&tmp);

    let (text, structured) = get_range(&server, None, Some(2)).await.unwrap();

    assert_eq!(text, "l1\nl2\n");
    assert_eq!(structured["start_line"], 1);
}

#[tokio::test]
async fn get_document_clamps_an_end_line_past_the_document() {
    let tmp = tempfile::tempdir().unwrap();
    let server = range_test_server(&tmp);

    let (text, structured) = get_range(&server, Some(4), Some(900)).await.unwrap();

    assert_eq!(text, "l4\nl5\n");
    assert_eq!(structured["end_line"], 5);
    assert_eq!(
        structured["partial"], true,
        "a clamped tail still omits the head of the document"
    );
}

#[tokio::test]
async fn get_document_covering_every_line_is_not_reported_as_partial() {
    let tmp = tempfile::tempdir().unwrap();
    let server = range_test_server(&tmp);

    let (_, structured) = get_range(&server, Some(1), Some(5)).await.unwrap();

    assert!(structured.get("partial").is_none(), "{structured}");
}

#[tokio::test]
async fn get_document_hashes_the_whole_document_even_for_a_partial_read() {
    let tmp = tempfile::tempdir().unwrap();
    let server = range_test_server(&tmp);

    let (_, full) = get_range(&server, None, None).await.unwrap();
    let (_, partial) = get_range(&server, Some(2), Some(3)).await.unwrap();

    assert_eq!(
        partial["version"], full["version"],
        "version is write_document's expected_version: it must describe the file \
         on disk, not the slice served"
    );
    assert_eq!(
        partial["version"],
        crate::write::document_version(RANGE_DOC.as_bytes())
    );
}

#[tokio::test]
async fn get_document_rejects_a_malformed_range() {
    let tmp = tempfile::tempdir().unwrap();
    let server = range_test_server(&tmp);

    assert!(
        get_range(&server, Some(0), None).await.is_err(),
        "line 0 does not exist; lines are 1-based"
    );
    assert!(
        get_range(&server, Some(4), Some(2)).await.is_err(),
        "an inverted range should be refused, not silently swapped"
    );
}

#[tokio::test]
async fn get_document_reports_links_out_and_links_in() {
    // End-to-end through the real adapter: seed `document_links` directly via
    // the state DB (bypassing ingest, same shortcut `seed_document` takes for
    // `documents`), then confirm the JSON shape `get_document` hands back —
    // key names, the `exists` flag on a dangling outbound target, and both
    // edge kinds riding the same list distinguished only by `kind`.
    let tmp = tempfile::tempdir().unwrap();
    let server = range_test_server(&tmp);
    let db = server.state_db().await.unwrap();

    // "range_doc.md" is itself indexed as a document so the markdown edge
    // below resolves to an existing target; "missing.md" never is, so the
    // semantic edge to it must come back with exists: false.
    db.upsert_document_metadata(
        "range_doc.md",
        &std::collections::HashMap::new(),
        100,
        "hash",
        1,
    )
    .await
    .unwrap();
    db.replace_links(
        "range_doc.md",
        "markdown",
        &[("missing.md".to_string(), None)],
    )
    .await
    .unwrap();
    db.replace_links(
        "referrer.md",
        "semantic",
        &[("range_doc.md".to_string(), Some(0.42))],
    )
    .await
    .unwrap();

    let (_, structured) = get_range(&server, None, None).await.unwrap();

    // missing.md was never indexed, so the outbound edge is a broken link;
    // the semantic edge is an inferred neighbour, not an author's link.
    assert_eq!(
        structured["broken_links"],
        serde_json::json!(["missing.md"])
    );
    assert_eq!(
        structured["similar"],
        serde_json::json!([{"path": "referrer.md", "score": 0.42}])
    );
    for absent in ["links_out", "links_in", "links_out_total", "links_in_total"] {
        assert!(structured.get(absent).is_none(), "{absent}: {structured}");
    }

    // A targeted read leaves the link graph out unless asked for.
    let (_, ranged) = get_range(&server, Some(2), Some(3)).await.unwrap();
    assert!(ranged.get("broken_links").is_none(), "{ranged}");
    let asked = server
        .get_document(Parameters(GetDocumentParams {
            path: "range_doc.md".into(),
            start_line: Some(2),
            end_line: Some(3),
            links: Some(true),
            ..Default::default()
        }))
        .await
        .unwrap();
    assert_eq!(
        single_representation(&asked)["broken_links"],
        serde_json::json!(["missing.md"])
    );
    let declined = server
        .get_document(Parameters(GetDocumentParams {
            path: "range_doc.md".into(),
            links: Some(false),
            ..Default::default()
        }))
        .await
        .unwrap();
    assert!(single_representation(&declined).get("similar").is_none());
}

#[tokio::test]
async fn get_document_caps_each_link_direction_and_reports_the_total() {
    let tmp = tempfile::tempdir().unwrap();
    let server = range_test_server(&tmp);
    let db = server.state_db().await.unwrap();
    for i in 0..25 {
        let source = format!("src{i:02}.md");
        db.replace_links(&source, "markdown", &[("range_doc.md".to_string(), None)])
            .await
            .unwrap();
    }
    let (_, structured) = get_range(&server, None, None).await.unwrap();
    assert_eq!(
        structured["links_in"].as_array().unwrap().len(),
        retrieval::MAX_LINKS_PER_DIRECTION as usize
    );
    assert_eq!(structured["links_in_total"], 25);
    assert!(structured["links_in"][0].is_string(), "{structured}");
}

#[tokio::test]
async fn get_document_rejects_a_start_line_past_the_document() {
    let tmp = tempfile::tempdir().unwrap();
    let server = range_test_server(&tmp);

    let err = get_range(&server, Some(99), None).await.unwrap_err();
    assert!(
        err.to_string().contains('5'),
        "the error should say how many lines the document actually has, got: {err}"
    );
}

#[tokio::test]
async fn get_document_accepts_start_line_1_against_an_empty_document() {
    // #298, regression from #290: start_line=1 must not 404/error against a
    // 0-byte document — the same request the web UI sends for every doc.
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("empty.md"), "").unwrap();
    let server = schema_tool_server(&tmp);

    let result = server
        .get_document(Parameters(GetDocumentParams {
            path: "empty.md".into(),
            start_line: Some(1),
            ..Default::default()
        }))
        .await
        .unwrap();
    let structured = single_representation(&result);

    assert_eq!(structured["content"], "");
    assert_eq!(structured["total_lines"], 0);
    assert!(structured.get("partial").is_none(), "{structured}");
}

#[tokio::test]
async fn get_document_validates_the_range_before_resolving_the_path() {
    let tmp = tempfile::tempdir().unwrap();
    let server = range_test_server(&tmp);

    let err = server
        .get_document(Parameters(GetDocumentParams {
            path: "no/such/document.md".into(),
            start_line: Some(9),
            end_line: Some(2),
            ..Default::default()
        }))
        .await
        .unwrap_err()
        .to_string();

    assert!(
        !err.contains("not found"),
        "a bad range should be reported as such, not masked by the path lookup: {err}"
    );
}

// -----------------------------------------------------------------------
// get_document section/outline modes (#286)
// -----------------------------------------------------------------------

const SECTION_DOC: &str = "# Guide\n\n## Alpha\n\nAlpha body.\n\n### Alpha Sub\n\nAlpha sub body.\n\n## Beta\n\nBeta body.\n";

/// A server whose `search.section_max_bytes` is `section_max_bytes`, with
/// `SECTION_DOC` written at `section_doc.md`.
fn section_test_server(tmp: &tempfile::TempDir, section_max_bytes: usize) -> KbSearchServer {
    std::fs::write(tmp.path().join("section_doc.md"), SECTION_DOC).unwrap();
    let mut config = (*make_test_resolved_config(tmp.path())).clone();
    config.search.section_max_bytes = section_max_bytes;
    make_write_test_server(tmp, &["**/*.md".to_string()], Arc::new(config))
}

async fn get_document_result(
    server: &KbSearchServer,
    params: GetDocumentParams,
) -> Result<(String, serde_json::Value), McpError> {
    let result = server.get_document(Parameters(params)).await?;
    let structured = single_representation(&result);
    // The text a reader gets: `content` when the view has one, else the JSON.
    let text = structured["content"]
        .as_str()
        .map_or_else(|| structured.to_string(), str::to_string);
    Ok((text, structured))
}

fn section_doc_params(overrides: GetDocumentParams) -> GetDocumentParams {
    GetDocumentParams {
        path: "section_doc.md".into(),
        ..overrides
    }
}

// --- param exclusivity ---------------------------------------------------

#[tokio::test]
async fn get_document_rejects_range_combined_with_line() {
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 16000);
    let err = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            start_line: Some(1),
            line: Some(3),
            ..Default::default()
        }),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("mutually exclusive"),
        "expected 'mutually exclusive' in error, got: {err}"
    );
}

#[tokio::test]
async fn get_document_rejects_range_combined_with_outline() {
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 16000);
    let err = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            end_line: Some(2),
            outline: Some(true),
            ..Default::default()
        }),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("mutually exclusive"),
        "expected 'mutually exclusive' in error, got: {err}"
    );
}

#[tokio::test]
async fn get_document_structured_content_is_the_shared_view_json_plus_the_envelope() {
    // Parity with `/api/doc`, structurally: both adapters must emit
    // exactly `retrieval::document_view_json` plus their envelope (web.rs
    // has the mirror test), so neither can grow a field the other lacks.
    let hash = crate::write::document_version(SECTION_DOC.as_bytes());
    let cases: Vec<(usize, GetDocumentParams)> = vec![
        (16000, GetDocumentParams::default()),
        // A whole-document read over the cap, which degrades to the
        // document's outline (#290).
        (10, GetDocumentParams::default()),
        (
            16000,
            GetDocumentParams {
                start_line: Some(2),
                end_line: Some(4),
                ..Default::default()
            },
        ),
        (
            16000,
            GetDocumentParams {
                line: Some(7),
                ..Default::default()
            },
        ),
        (
            10,
            GetDocumentParams {
                heading_path: Some(vec!["Alpha".into()]),
                ..Default::default()
            },
        ),
        (
            10,
            GetDocumentParams {
                heading_path: Some(vec!["Beta".into()]),
                ..Default::default()
            },
        ),
        (
            1,
            GetDocumentParams {
                outline: Some(true),
                ..Default::default()
            },
        ),
        (
            16000,
            GetDocumentParams {
                line: Some(3),
                levels_up: Some(1),
                outline: Some(true),
                ..Default::default()
            },
        ),
    ];
    for (cap, params) in cases {
        let request = retrieval::parse_document_view_request(
            params.start_line,
            params.end_line,
            params.line,
            params.heading_path.clone(),
            params.levels_up,
            params.outline.unwrap_or(false),
        )
        .unwrap();
        let label = format!("{params:?}");
        let tmp = tempfile::tempdir().unwrap();
        let server = section_test_server(&tmp, cap);
        let (_, mut structured) = get_document_result(&server, section_doc_params(params))
            .await
            .unwrap();
        // No links are seeded, so no link list rides on any of these.
        let _ = structured.as_object_mut().unwrap();
        let view = retrieval::resolve_document_view(SECTION_DOC, &request, cap).unwrap();
        let mut expected = retrieval::document_view_json(&view);
        expected.insert("path".into(), serde_json::json!("section_doc.md"));
        expected.insert("version".into(), serde_json::json!(hash));
        assert_eq!(structured, serde_json::Value::Object(expected), "{label}");
    }
}

#[tokio::test]
async fn get_document_outline_with_a_selector_outlines_that_section() {
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 16000);
    let (text, structured) = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            line: Some(3),
            outline: Some(true),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(
        structured["section"]["heading_path"],
        serde_json::json!(["Guide", "Alpha"])
    );
    assert_eq!(
        structured["outline"],
        serde_json::json!([{
            "heading_path": ["Guide", "Alpha", "Alpha Sub"],
            "level": 3,
            "line_start": 7,
            "line_end": 10,
        }])
    );
    assert!(structured.get("truncated").is_none(), "{structured}");
    assert!(structured.get("content").is_none());
    assert!(
        !text.contains("\"heading\""),
        "no duplicate heading key: {text}"
    );
}

#[tokio::test]
async fn get_document_rejects_line_combined_with_heading_path() {
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 16000);
    let err = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            line: Some(3),
            heading_path: Some(vec!["Alpha".to_string()]),
            ..Default::default()
        }),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("mutually exclusive"),
        "expected 'mutually exclusive' in error, got: {err}"
    );
}

#[tokio::test]
async fn get_document_rejects_levels_up_without_a_selector() {
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 16000);
    let err = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            levels_up: Some(1),
            ..Default::default()
        }),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("levels_up"),
        "expected an error naming levels_up, got: {err}"
    );
}

#[tokio::test]
async fn get_document_rejects_an_empty_heading_path() {
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 16000);
    let err = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            heading_path: Some(vec![]),
            ..Default::default()
        }),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("heading_path"),
        "expected an error naming heading_path, got: {err}"
    );
}

// --- section mode ---------------------------------------------------------

#[tokio::test]
async fn get_document_section_by_line_returns_the_resolved_section() {
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 16000);

    // Line 7 ("Alpha sub body.") sits inside "### Alpha Sub".
    let (text, structured) = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            line: Some(7),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert!(text.contains("Alpha sub body."));
    assert_eq!(
        structured["section"]["heading_path"],
        serde_json::json!(["Guide", "Alpha", "Alpha Sub"])
    );
    assert_eq!(structured["section"]["level"], 3);
    assert!(structured.get("outline_only").is_none(), "{structured}");
    assert_eq!(structured["partial"], true);
    assert!(
        structured.get("outline").is_none(),
        "a full section response must not also carry an outline"
    );
    // version is always over the whole file, matching range mode.
    assert_eq!(
        structured["version"],
        crate::write::document_version(SECTION_DOC.as_bytes()).to_string()
    );
}

#[tokio::test]
async fn get_document_section_by_heading_path_suffix() {
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 16000);

    let (text, structured) = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            heading_path: Some(vec!["Beta".to_string()]),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert!(text.contains("Beta body."));
    assert_eq!(
        structured["section"]["heading_path"],
        serde_json::json!(["Guide", "Beta"])
    );
}

#[tokio::test]
async fn get_document_section_levels_up_climbs_to_the_ancestor() {
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 16000);

    let (text, structured) = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            heading_path: Some(vec!["Alpha Sub".to_string()]),
            levels_up: Some(1),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(
        structured["section"]["heading_path"],
        serde_json::json!(["Guide", "Alpha"])
    );
    assert!(
        text.contains("Alpha Sub"),
        "climbing must include the child subtree"
    );
}

#[tokio::test]
async fn get_document_section_not_found_reports_a_hint() {
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 16000);

    let err = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            heading_path: Some(vec!["Nonexistent".to_string()]),
            ..Default::default()
        }),
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("Guide"));
    assert_eq!(
        err.data,
        Some(serde_json::json!({ "hint": ["Guide"], "candidates": [] })),
        "the McpError data payload should carry the structured hint, and an empty \
         candidates list when nothing plausibly matches"
    );
}

#[tokio::test]
async fn get_document_section_subsequence_matches_a_skipped_middle_segment() {
    // "Guide" > "Alpha" > "Alpha Sub" — asking for ["Guide", "Alpha Sub"]
    // skips "Alpha" in the middle, a shape the suffix tier alone can't
    // resolve (fix #291).
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 16000);

    let (_, structured) = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            heading_path: Some(vec!["Guide".to_string(), "Alpha Sub".to_string()]),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(
        structured["section"]["heading_path"],
        serde_json::json!(["Guide", "Alpha", "Alpha Sub"])
    );
}

#[tokio::test]
async fn get_document_section_not_found_candidates_use_substring_but_never_resolve() {
    // "Alpha S" is a substring of "Alpha Sub", not an exact segment
    // match at any tier — it must appear as a suggestion, never resolve
    // directly (fix #291).
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 16000);

    let err = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            heading_path: Some(vec!["Alpha S".to_string()]),
            ..Default::default()
        }),
    )
    .await
    .unwrap_err();
    let data = err.data.expect("NotFound must carry structured data");
    let candidates = data["candidates"].as_array().unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(
        candidates[0]["heading_path"],
        serde_json::json!(["Guide", "Alpha", "Alpha Sub"])
    );
}

#[tokio::test]
async fn get_document_section_exceeding_the_cap_falls_back_to_children() {
    let tmp = tempfile::tempdir().unwrap();
    // Trivially small — "## Alpha"'s own subtree text always exceeds it.
    let server = section_test_server(&tmp, 10);

    let (text, structured) = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            heading_path: Some(vec!["Alpha".to_string()]),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(structured["outline_only"], true);
    assert!(structured.get("content").is_none());
    let outline = structured["outline"].as_array().unwrap();
    assert_eq!(outline.len(), 1);
    assert_eq!(
        outline[0]["heading_path"],
        serde_json::json!(["Guide", "Alpha", "Alpha Sub"])
    );
    assert!(text.contains("Alpha Sub"), "{text}");
    // The intro is surfaced as a plain start_line/end_line range.
    assert_eq!(
        structured["intro"]["heading_path"],
        serde_json::json!(["Guide", "Alpha"])
    );
    let intro_start = structured["intro"]["line_start"].as_u64().unwrap() as usize;
    let intro_end = structured["intro"]["line_end"].as_u64().unwrap() as usize;
    let (intro_text, _) = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            start_line: Some(intro_start),
            end_line: Some(intro_end),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(intro_text, "## Alpha\n\nAlpha body.\n\n");

    // Fetching by a line inside the intro resolves to the same section,
    // so it gets the same outline_only response — never a silent slice.
    let (_, by_line) = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            line: Some(intro_start + 2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(by_line, structured);
}

#[tokio::test]
async fn get_document_outline_only_text_does_not_repeat_the_json_or_claim_a_blank_intro() {
    // Heading-only intro: no intro, and no "has its own text" claim.
    let tmp = tempfile::tempdir().unwrap();
    let big = "x".repeat(300);
    std::fs::write(
        tmp.path().join("intro.md"),
        format!("# A\n## B\n{big}\n## C\n{big}\n"),
    )
    .unwrap();
    let mut config = (*make_test_resolved_config(tmp.path())).clone();
    config.search.section_max_bytes = 400;
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], Arc::new(config));
    let (text, structured) = get_document_result(
        &server,
        GetDocumentParams {
            path: "intro.md".into(),
            line: Some(1),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(structured["outline_only"], true);
    assert!(
        structured.get("intro").is_none(),
        "a heading-only section has no intro to report: {text}"
    );
    assert!(!text.contains("\"heading\""), "{text}");
}

#[tokio::test]
async fn get_document_section_exceeding_the_cap_with_no_children_is_truncated() {
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 10);

    let (text, structured) = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            heading_path: Some(vec!["Beta".to_string()]),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert!(structured.get("outline_only").is_none(), "{text}");
    // Nothing smaller to narrow into, so text comes back — but bounded by
    // the cap, flagged, and reporting where it stops so the caller can
    // page on with start_line (#290).
    assert_eq!(structured["oversized"], true);
    assert_eq!(structured["truncated"], true);
    assert_eq!(structured["content"], "## Beta\n\n");
    assert_eq!(structured["end_line"], 12);
    assert_eq!(structured["section"]["line_end"], 13);
    assert_eq!(structured["partial"], true);
}

#[tokio::test]
async fn get_document_whole_document_over_the_cap_degrades_to_an_outline() {
    // The default call shape on a document too big to serve as text:
    // its outline, marked as a degraded read, never silent full text (#290).
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 10);

    let (text, structured) = get_document_result(&server, section_doc_params(Default::default()))
        .await
        .unwrap();

    assert_eq!(structured["outline_only"], true);
    assert!(structured.get("content").is_none(), "{structured}");
    assert_eq!(
        structured["outline"][0]["heading_path"],
        serde_json::json!(["Guide"])
    );
    // SECTION_DOC opens on its first heading, so there is no intro.
    assert!(structured.get("intro").is_none(), "{text}");
}

#[tokio::test]
async fn get_document_whole_document_over_the_cap_reports_its_preamble_as_intro() {
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 10);
    std::fs::write(
        tmp.path().join("preamble.md"),
        "Preamble text.\n\n# Guide\n\nbody\n",
    )
    .unwrap();

    let (text, structured) = get_document_result(
        &server,
        GetDocumentParams {
            path: "preamble.md".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    assert_eq!(structured["outline_only"], true);
    assert_eq!(structured["intro"]["line_start"], 1);
    assert_eq!(structured["intro"]["line_end"], 2);
    assert!(!text.contains("\"heading\""), "{text}");
}

#[tokio::test]
async fn get_document_whole_document_over_the_cap_without_headings_is_truncated() {
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 10);
    std::fs::write(
        tmp.path().join("flat.md"),
        "one line\ntwo line\nthree line\n",
    )
    .unwrap();

    let (text, structured) = get_document_result(
        &server,
        GetDocumentParams {
            path: "flat.md".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    // No headings to navigate by, so the text is cut on a line boundary
    // and the response says where it stopped (#290).
    assert_eq!(structured["truncated"], true);
    assert_eq!(structured["content"], "one line\n");
    assert_eq!(structured["end_line"], 1);
    assert_eq!(structured["total_lines"], 3);
    assert_eq!(structured["partial"], true);
    assert_eq!(text, "one line\n");
}

#[tokio::test]
async fn get_document_truncated_whole_read_without_headings_keeps_the_link_lists() {
    // An oversized document with headings degrades to an outline and keeps its
    // link lists; the heading-less one degrades to a truncated slice and must
    // too, or a missing list reads as "no links".
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 10);
    std::fs::write(
        tmp.path().join("flat.md"),
        "one line\ntwo line\nthree line\n",
    )
    .unwrap();
    let db = server.state_db().await.unwrap();
    db.replace_links("flat.md", "markdown", &[("missing.md".to_string(), None)])
        .await
        .unwrap();
    db.replace_links("referrer.md", "markdown", &[("flat.md".to_string(), None)])
        .await
        .unwrap();

    let (_, whole) = get_document_result(
        &server,
        GetDocumentParams {
            path: "flat.md".into(),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(whole["truncated"], true, "{whole}");
    assert_eq!(
        whole["broken_links"],
        serde_json::json!(["missing.md"]),
        "{whole}"
    );
    assert_eq!(whole["links_in"], serde_json::json!(["referrer.md"]));

    // A range the caller named is a targeted read: no link lists.
    let (_, ranged) = get_document_result(
        &server,
        GetDocumentParams {
            path: "flat.md".into(),
            start_line: Some(1),
            end_line: Some(2),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(ranged["partial"], true, "{ranged}");
    for absent in ["links_out", "broken_links", "links_in", "similar"] {
        assert!(ranged.get(absent).is_none(), "{absent}: {ranged}");
    }
}

#[tokio::test]
async fn get_document_whole_document_within_the_cap_is_unchanged() {
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 16000);

    let (text, structured) = get_document_result(&server, section_doc_params(Default::default()))
        .await
        .unwrap();

    assert_eq!(text, SECTION_DOC);
    assert_eq!(structured["content"], SECTION_DOC);
    assert!(structured.get("partial").is_none(), "{structured}");
    assert!(structured.get("truncated").is_none(), "{structured}");
    assert!(structured.get("outline_only").is_none(), "{structured}");
}

#[tokio::test]
async fn get_document_explicit_range_over_the_cap_is_served_whole() {
    // The caller named the bounds, so the cap does not apply (#290).
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 10);

    let (text, structured) = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            start_line: Some(1),
            end_line: Some(13),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(text, SECTION_DOC);
    assert_eq!(structured["content"], SECTION_DOC);
    assert!(structured.get("truncated").is_none(), "{structured}");
}

#[tokio::test]
async fn get_document_section_duplicate_exact_heading_path_is_ambiguous() {
    // Two sections sharing the exact same full
    // heading_path must not silently resolve to the first one.
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 16000);
    std::fs::write(
        tmp.path().join("dup.md"),
        "# S\n## Fireball\na\n## Fireball\nb\n",
    )
    .unwrap();

    let err = get_document_result(
        &server,
        GetDocumentParams {
            path: "dup.md".into(),
            heading_path: Some(vec!["Fireball".to_string()]),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();
    let data = err.data.expect("Ambiguous must carry structured data");
    let candidates = data["candidates"].as_array().unwrap();
    assert_eq!(
        candidates.len(),
        2,
        "both duplicate sections must be listed"
    );
}

// --- outline mode -----------------------------------------------------

#[tokio::test]
async fn get_document_outline_mode_returns_no_content() {
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 16000);

    let (text, structured) = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            outline: Some(true),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert!(
        structured.get("content").is_none(),
        "outline mode must not carry a content field"
    );
    let outline = structured["outline"].as_array().unwrap();
    assert_eq!(outline.len(), 4, "Guide, Alpha, Alpha Sub, Beta");
    assert_eq!(outline[0]["heading_path"], serde_json::json!(["Guide"]));
    assert_eq!(
        outline[1]["heading_path"],
        serde_json::json!(["Guide", "Alpha"])
    );
    for heading in ["Guide", "Alpha", "Alpha Sub", "Beta"] {
        assert!(
            text.contains(heading),
            "missing heading {heading:?}: {text}"
        );
    }
    assert_eq!(
        structured["version"],
        crate::write::document_version(SECTION_DOC.as_bytes()).to_string()
    );
    // An uncapped outline carries no truncation keys; it always reports
    // the whole document's line count.
    assert!(structured.get("total_entries").is_none(), "{structured}");
    assert!(structured.get("truncated").is_none(), "{structured}");
    assert_eq!(
        structured["total_lines"].as_u64().unwrap(),
        crate::retrieval::count_lines(SECTION_DOC) as u64
    );
}

#[tokio::test]
async fn get_document_outline_mode_truncates_past_the_section_max_bytes_budget() {
    // An outline with many headings must not produce
    // an unbounded response.
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 200);
    let mut many_headings = String::new();
    for i in 0..200 {
        many_headings.push_str(&format!("# Heading number {i}\n\nbody\n\n"));
    }
    std::fs::write(tmp.path().join("many.md"), &many_headings).unwrap();

    let (_, structured) = get_document_result(
        &server,
        GetDocumentParams {
            path: "many.md".into(),
            outline: Some(true),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    assert_eq!(structured["total_entries"], 200);
    assert_eq!(structured["truncated"], true);
    let outline = structured["outline"].as_array().unwrap();
    assert!(!outline.is_empty());
    assert!((outline.len() as u64) < 200);
}

// --- selector validation ------------------------------------------------

#[tokio::test]
async fn get_document_rejects_line_zero() {
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 16000);
    let err = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            line: Some(0),
            ..Default::default()
        }),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("1-based") || err.contains("1 or greater"),
        "expected a 1-based line error, got: {err}"
    );
}

#[tokio::test]
async fn get_document_rejects_a_blank_heading_path_segment() {
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 16000);
    let err = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            heading_path: Some(vec!["Alpha".to_string(), "  ".to_string()]),
            ..Default::default()
        }),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("heading_path"),
        "expected an error naming heading_path, got: {err}"
    );
}

#[tokio::test]
async fn get_document_section_by_heading_path_ignores_internal_whitespace_and_case() {
    // Matching must collapse internal whitespace, not just trim/lowercase.
    let tmp = tempfile::tempdir().unwrap();
    let server = section_test_server(&tmp, 16000);
    let (_, structured) = get_document_result(
        &server,
        section_doc_params(GetDocumentParams {
            heading_path: Some(vec!["  alpha   sub  ".to_string()]),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(
        structured["section"]["heading_path"],
        serde_json::json!(["Guide", "Alpha", "Alpha Sub"])
    );
}

// -----------------------------------------------------------------------
// write_document helper tests
// -----------------------------------------------------------------------
