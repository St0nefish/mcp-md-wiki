use crate::mcp::search::*;
use crate::mcp::write_document::*;
use crate::mcp::*;
use crate::retrieval;
use std::path::Path;
fn payload_search_result(file_path: &str, title: &str, score: f32) -> crate::qdrant::SearchResult {
    let mut payload = HashMap::new();
    payload.insert("file_path".to_string(), serde_json::json!(file_path));
    // Frontmatter under `fm`, as `ingest::upsert_pending` writes it (#350).
    payload.insert("fm".to_string(), serde_json::json!({ "title": title }));
    payload.insert("text".to_string(), serde_json::json!("some body text"));
    crate::qdrant::SearchResult {
        score,
        pre_rerank_score: None,
        dense_score: None,
        sparse_score: None,
        phrase_score: None,
        payload,
    }
}

fn chunk_payload(
    results: &[crate::qdrant::SearchResult],
    explain: bool,
    path_prefix_truncated: bool,
    offset_truncated: bool,
) -> serde_json::Value {
    build_chunk_search_payload(
        results,
        Path::new("/data"),
        explain,
        "dense cosine",
        path_prefix_truncated,
        offset_truncated,
    )
}

#[test]
fn build_chunk_search_payload_structured_results_present_and_populated() {
    let results = vec![
        payload_search_result("/data/notes/a.md", "A", 0.9),
        payload_search_result("/data/notes/b.md", "B", 0.5),
    ];
    let structured = chunk_payload(&results, false, false, false);
    let arr = structured["results"]
        .as_array()
        .expect("results must be an array, not missing/null");
    assert_eq!(arr.len(), results.len());
    for entry in arr {
        assert!(entry["file_path"].is_string());
        assert!(entry["title"].is_string());
        assert!(entry["score"].is_number());
        assert!(entry["text"].is_string());
    }
    assert_eq!(structured["returned"], serde_json::json!(2));
    assert_eq!(
        arr.iter()
            .map(|e| e["file_path"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["notes/a.md", "notes/b.md"]
    );
    assert_eq!(
        arr.iter()
            .map(|e| e["title"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["A", "B"],
        "the title is read from the payload's `fm` object (#350)"
    );
}

#[test]
fn build_chunk_search_payload_reads_type_and_tags_from_fm() {
    let mut result = payload_search_result("/data/notes/a.md", "A", 0.9);
    result.payload.insert(
        "fm".to_string(),
        serde_json::json!({ "title": "A", "type": "guide", "tags": ["x"] }),
    );
    // A top-level key of the shape before #350 is not frontmatter any more.
    result
        .payload
        .insert("type".to_string(), serde_json::json!("stale"));
    let structured = chunk_payload(&[result], false, false, false);
    let row = &structured["results"][0];
    assert_eq!(row["type"], "guide");
    assert_eq!(row["tags"], serde_json::json!(["x"]));
}

#[test]
fn build_chunk_search_payload_heading_path_is_omitted_when_the_payload_has_none() {
    let bare = payload_search_result("/data/notes/a.md", "A", 0.9);
    let mut with_path = payload_search_result("/data/notes/b.md", "B", 0.5);
    with_path.payload.insert(
        crate::qdrant::HEADING_PATH_KEY.to_string(),
        serde_json::json!(["Guide", "Setup"]),
    );
    let structured = chunk_payload(&[bare, with_path], false, false, false);
    let rows = structured["results"].as_array().unwrap();
    assert!(
        rows[0].get("heading_path").is_none(),
        "no heading_path key (not even null) without heading metadata: {}",
        rows[0]
    );
    assert_eq!(
        rows[1]["heading_path"],
        serde_json::json!(["Guide", "Setup"])
    );
}

#[test]
fn build_chunk_search_payload_empty_results_report_zero_not_missing_key() {
    let structured = chunk_payload(&[], false, false, false);
    assert_eq!(
        structured,
        serde_json::json!({"returned": 0, "results": []})
    );
}

/// A row carries nothing that is null, false, empty, internal or derived:
/// no `chunk_index`, no `domain`, no unset per-arm scores, no
/// `phrase_matched: false`, no `text_truncated`, and no false envelope flags.
#[test]
fn build_chunk_search_payload_omits_null_false_and_internal_fields() {
    let mut result = payload_search_result("/data/notes/a.md", "A", 0.9);
    result
        .payload
        .insert("chunk_index".to_string(), serde_json::json!(3));
    result.payload.insert(
        "fm".to_string(),
        serde_json::json!({ "title": "A", "domain": "notes", "tags": [], "type": "" }),
    );
    for explain in [false, true] {
        let structured = chunk_payload(std::slice::from_ref(&result), explain, false, false);
        let text = structured.to_string();
        for absent in [
            "chunk_index",
            "domain",
            "dense_score",
            "sparse_score",
            "pre_rerank_score",
            "phrase_matched",
            "text_truncated",
            "path_prefix_truncated",
            "offset_truncated",
            "\"tags\"",
            "\"type\"",
            "null",
            "false",
        ] {
            assert!(!text.contains(absent), "{absent} present: {text}");
        }
    }
}

#[test]
fn build_chunk_search_payload_rounds_scores_to_four_significant_digits() {
    let mut result = payload_search_result("/data/notes/a.md", "A", 3.342_690_5);
    result.dense_score = Some(0.012_345_678);
    let structured = chunk_payload(&[result], true, false, false);
    let row = &structured["results"][0];
    assert_eq!(row["score"], serde_json::json!(3.343));
    assert_eq!(row["dense_score"], serde_json::json!(0.01235));
}

/// The breadcrumb/description prefix the chunker adds for the embedding is
/// not part of the snippet: a stored body offset skips it, and a payload
/// written before the offset existed still drops a leading description.
#[test]
fn build_chunk_search_payload_snippet_starts_at_the_chunk_body() {
    let mut with_offset = payload_search_result("/data/notes/a.md", "A", 0.9);
    let prefix = "Guide > Setup\n\nA long document description.\n\n";
    with_offset.payload.insert(
        "text".to_string(),
        serde_json::json!(format!("{prefix}## Setup\n\nBody text.")),
    );
    with_offset.payload.insert(
        crate::qdrant::CHUNK_BODY_OFFSET_KEY.to_string(),
        serde_json::json!(prefix.len()),
    );
    let mut legacy = payload_search_result("/data/notes/b.md", "B", 0.5);
    legacy.payload.insert(
        "description".to_string(),
        serde_json::json!("A long document description."),
    );
    legacy.payload.insert(
        "text".to_string(),
        serde_json::json!("A long document description.\n\nLegacy body."),
    );
    let structured = chunk_payload(&[with_offset, legacy], false, false, false);
    let rows = structured["results"].as_array().unwrap();
    assert_eq!(rows[0]["text"], "## Setup\n\nBody text.");
    assert_eq!(rows[1]["text"], "Legacy body.");
}

#[test]
fn build_chunk_search_payload_cuts_a_long_snippet_with_an_ellipsis() {
    let mut result = payload_search_result("/data/notes/a.md", "A", 0.9);
    result
        .payload
        .insert("text".to_string(), serde_json::json!("x".repeat(900)));
    let structured = chunk_payload(&[result], false, false, false);
    let text = structured["results"][0]["text"].as_str().unwrap();
    assert_eq!(text.chars().count(), SNIPPET_CHARS + 1);
    assert!(text.ends_with('…'));
}

#[test]
fn build_chunk_search_payload_truncation_flags_appear_only_when_set() {
    let results = [payload_search_result("/data/notes/a.md", "A", 0.9)];
    for rows in [&results[..], &[]] {
        let structured = chunk_payload(rows, false, true, true);
        assert_eq!(structured["path_prefix_truncated"], serde_json::json!(true));
        assert_eq!(structured["offset_truncated"], serde_json::json!(true));
    }
}

#[test]
fn build_chunk_search_payload_explain_adds_mode_and_per_arm_scores() {
    let mut matched = payload_search_result("/data/notes/a.md", "A", 0.9);
    matched.phrase_score = Some(0.7);
    matched.dense_score = Some(0.8);
    let unmatched = payload_search_result("/data/notes/b.md", "B", 0.5);

    let off = chunk_payload(&[matched.clone(), unmatched.clone()], false, false, false);
    assert!(off.get("mode").is_none(), "{off}");
    assert!(off["results"][0].get("dense_score").is_none(), "{off}");

    for mode in [
        "hybrid RRF + phrase",
        "hybrid RRF",
        "dense + phrase RRF",
        "dense cosine",
    ] {
        let on = build_chunk_search_payload(
            &[matched.clone(), unmatched.clone()],
            Path::new("/data"),
            true,
            mode,
            false,
            false,
        );
        assert_eq!(on["mode"], serde_json::json!(mode));
        assert_eq!(on["results"][0]["phrase_matched"], serde_json::json!(true));
        assert_eq!(on["results"][0]["dense_score"], serde_json::json!(0.8));
        assert!(on["results"][1].get("phrase_matched").is_none(), "{on}");
    }
}

fn payload_grouped_document(
    file_path: &str,
    title: &str,
    score: f32,
) -> retrieval::GroupedDocument {
    retrieval::GroupedDocument {
        score,
        summary: crate::state::DocumentSummary {
            file_path: file_path.to_string(),
            title: Some(title.to_string()),
            description: None,
            mtime: 0,
            indexed_at: "2026-01-01T00:00:00Z".to_string(),
            frontmatter: serde_json::json!({}),
        },
    }
}

#[test]
fn build_grouped_search_payload_omits_total_and_has_more_but_keeps_score() {
    let documents = vec![
        payload_grouped_document("notes/a.md", "A", 0.9),
        payload_grouped_document("notes/b.md", "B", 0.5),
    ];
    let structured = build_grouped_search_payload(&documents, None, &[], false, false);
    let obj = structured.as_object().unwrap();
    assert!(
        !obj.contains_key("total"),
        "grouped search cannot back `total` and must not claim one: {structured}"
    );
    assert!(
        !obj.contains_key("has_more"),
        "grouped search cannot back `has_more` and must not claim one: {structured}"
    );
    let docs = structured["documents"].as_array().unwrap();
    assert_eq!(docs.len(), 2);
    for doc in docs {
        assert!(
            doc["score"].is_number(),
            "each document must carry a score: {doc}"
        );
    }
    assert_eq!(structured["returned"], serde_json::json!(2));
}

#[test]
fn build_grouped_search_payload_empty_reports_zero() {
    let structured = build_grouped_search_payload(&[], None, &[], false, false);
    assert_eq!(
        structured,
        serde_json::json!({"returned": 0, "documents": []})
    );
}

#[test]
fn build_grouped_search_payload_truncation_flags_appear_only_when_set() {
    let documents = vec![payload_grouped_document("notes/a.md", "A", 0.9)];
    let structured = build_grouped_search_payload(&documents, None, &[], true, true);
    assert_eq!(structured["path_prefix_truncated"], serde_json::json!(true));
    assert_eq!(structured["offset_truncated"], serde_json::json!(true));
    let plain = build_grouped_search_payload(&documents, None, &[], false, false);
    assert!(plain.get("path_prefix_truncated").is_none(), "{plain}");
    assert!(plain.get("offset_truncated").is_none(), "{plain}");
}

/// A document row's `frontmatter` drops what the row already promotes
/// (`title`, `description`) and what the path derives (the root schema's
/// `derived:` fields, #345), keeping a derived field only when the caller asked
/// for it by name; `mtime` is not on a grouped row at all. A field nothing
/// derives stays.
#[test]
fn document_rows_drop_promoted_and_derived_frontmatter_keys() {
    let mut doc = payload_grouped_document("notes/a.md", "A", 0.9);
    doc.summary.description = Some("About A".to_string());
    doc.summary.frontmatter = serde_json::json!({
        "title": "A", "description": "About A", "domain": "notes", "tags": ["x"],
    });
    let derived = vec!["domain".to_string()];
    let structured =
        build_grouped_search_payload(std::slice::from_ref(&doc), None, &derived, false, false);
    let row = &structured["documents"][0];
    assert_eq!(row["frontmatter"], serde_json::json!({"tags": ["x"]}));
    assert_eq!(row["description"], "About A");
    assert!(row.get("mtime").is_none(), "{row}");

    let not_derived =
        build_grouped_search_payload(std::slice::from_ref(&doc), None, &[], false, false);
    assert_eq!(
        not_derived["documents"][0]["frontmatter"]["domain"], "notes",
        "derivation off: domain is an authored field"
    );

    let fields = vec!["domain".to_string()];
    let asked = build_grouped_search_payload(&[doc], Some(&fields), &derived, false, false);
    assert_eq!(asked["documents"][0]["frontmatter"]["domain"], "notes");

    let mut bare = payload_grouped_document("notes/b.md", "B", 0.5);
    bare.summary.frontmatter = serde_json::json!({"title": "B"});
    let structured = build_grouped_search_payload(&[bare], None, &[], false, false);
    assert!(
        structured["documents"][0].get("frontmatter").is_none(),
        "an empty projection is omitted: {structured}"
    );
}
/// `WriteDocumentParams::documents`'s doc comment states the batch cap as a
/// literal (doc comments cannot interpolate a const); this keeps that
/// served number in step with `write::MAX_BATCH_DOCUMENTS`.
#[test]
fn batch_documents_description_states_the_real_cap() {
    let schema = schemars::schema_for!(WriteDocumentParams);
    let description = schema.as_value()["properties"]["documents"]["description"]
        .as_str()
        .expect("documents must carry a description");
    let expected = format!("at most {}", write::MAX_BATCH_DOCUMENTS);
    assert!(
        description
            .replace('\n', " ")
            .contains(&expected.replace('\n', " ")),
        "documents description must state the real cap ({}): {description}",
        write::MAX_BATCH_DOCUMENTS
    );
}
