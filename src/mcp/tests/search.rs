use rmcp::handler::server::wrapper::Parameters;

use super::get_document::single_representation;
use crate::config::Granularity;
use crate::mcp::search::*;
use crate::mcp::test_support::*;
use crate::mcp::*;
use crate::retrieval;
use crate::schema::SchemaCache;
use crate::state::{FieldFilter, OrderBy};
fn filters_from(json: serde_json::Value) -> SearchParams {
    SearchParams {
        filters: Some(SearchFiltersInput(json.as_object().unwrap().clone())),
        ..Default::default()
    }
}

fn parsed_filters(json: serde_json::Value) -> Vec<(String, FieldFilter)> {
    build_document_query(&filters_from(json)).unwrap().filters
}

#[test]
fn scalar_filter_becomes_equality() {
    let filters = parsed_filters(serde_json::json!({ "type": "guide" }));
    assert_eq!(
        filters,
        vec![("type".to_string(), FieldFilter::AnyOf(vec!["guide".into()]))]
    );
}

#[test]
fn array_filter_becomes_any_of() {
    let filters = parsed_filters(serde_json::json!({ "tags": ["recipe", "dinner"] }));
    assert_eq!(
        filters,
        vec![(
            "tags".to_string(),
            FieldFilter::AnyOf(vec!["recipe".into(), "dinner".into()])
        )]
    );
}

#[test]
fn all_of_object_becomes_all_of() {
    let filters = parsed_filters(serde_json::json!({ "tags": { "all_of": ["recipe", "dinner"] } }));
    assert_eq!(
        filters,
        vec![(
            "tags".to_string(),
            FieldFilter::AllOf(vec!["recipe".into(), "dinner".into()])
        )]
    );
}

#[test]
fn numeric_operators_become_a_range() {
    let filters =
        parsed_filters(serde_json::json!({ "planning.prep_minutes": { "gte": 10, "lt": 30 } }));
    assert_eq!(
        filters,
        vec![(
            "planning.prep_minutes".to_string(),
            FieldFilter::Range {
                gte: Some(10.0),
                lte: None,
                gt: None,
                lt: Some(30.0),
            }
        )]
    );
}

#[test]
fn booleans_and_numbers_canonicalize_like_the_write_path() {
    // The property that makes {"planning.needs_recipe": false} match stored rows.
    let filters =
        parsed_filters(serde_json::json!({ "planning.needs_recipe": false, "planning.rating": 5 }));
    assert!(filters.contains(&(
        "planning.needs_recipe".to_string(),
        FieldFilter::AnyOf(vec!["false".into()])
    )));
    assert!(filters.contains(&(
        "planning.rating".to_string(),
        FieldFilter::AnyOf(vec!["5".into()])
    )));
}

#[test]
fn unknown_filter_operator_is_rejected_with_guidance() {
    let err = build_document_query(&filters_from(
        serde_json::json!({ "planning.prep_minutes": { "lte_": 30 } }),
    ))
    .unwrap_err();
    let msg = format!("{:?}", err);
    assert!(msg.contains("unknown operator"), "got: {msg}");
    assert!(msg.contains("all_of"), "error should list valid operators");
}

#[test]
fn empty_object_filter_is_rejected() {
    assert!(
        build_document_query(&filters_from(serde_json::json!({ "tags": {} }))).is_err(),
        "an operator-less object would otherwise match everything"
    );
}

#[test]
fn null_filter_is_rejected() {
    assert!(build_document_query(&filters_from(serde_json::json!({ "tags": null }))).is_err());
}

#[test]
fn non_numeric_range_bound_is_rejected() {
    let err = build_document_query(&filters_from(
        serde_json::json!({ "planning.prep_minutes": { "lt": "thirty" } }),
    ))
    .unwrap_err();
    assert!(format!("{:?}", err).contains("must be a number"));
}

#[test]
fn nested_object_filter_value_is_rejected() {
    assert!(
        build_document_query(&filters_from(
            serde_json::json!({ "tags": [{ "nested": true }] })
        ))
        .is_err()
    );
}

#[test]
fn list_defaults_are_applied_when_nothing_is_supplied() {
    let query = build_document_query(&SearchParams::default()).unwrap();
    assert_eq!(query.limit, DEFAULT_LIST_LIMIT);
    assert_eq!(query.offset, 0);
    assert_eq!(query.order_by, OrderBy::Path);
    assert!(!query.order_desc);
    assert!(query.filters.is_empty());
    assert!(query.path_prefix.is_none());
    assert!(query.fields.is_none());
}

#[test]
fn list_limit_is_clamped_to_the_cap() {
    let query = build_document_query(&SearchParams {
        limit: Some(999_999),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(query.limit, MAX_LIST_LIMIT);

    let query = build_document_query(&SearchParams {
        limit: Some(0),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(query.limit, 1, "a zero-size page would return nothing");
}

#[test]
fn invalid_order_by_is_rejected() {
    let err = build_document_query(&SearchParams {
        order_by: Some("file_path; DROP TABLE documents".into()),
        ..Default::default()
    })
    .unwrap_err();
    assert!(format!("{:?}", err).contains("unknown order_by"));
}

#[test]
fn filters_are_sorted_for_stable_sql() {
    let query = build_document_query(&filters_from(
        serde_json::json!({ "zeta": "a", "alpha": "b", "mid": "c" }),
    ))
    .unwrap();
    let names: Vec<&str> = query.filters.iter().map(|(f, _)| f.as_str()).collect();
    assert_eq!(names, vec!["alpha", "mid", "zeta"]);
}

#[test]
fn too_many_filters_is_rejected() {
    let mut map = serde_json::Map::new();
    for i in 0..(MAX_LIST_FILTERS + 1) {
        map.insert(format!("field{i}"), serde_json::json!("x"));
    }
    let err = build_document_query(&SearchParams {
        filters: Some(SearchFiltersInput(map)),
        ..Default::default()
    })
    .unwrap_err();
    assert!(format!("{:?}", err).contains("too many filters"));
}

#[test]
fn search_filters_advertises_as_a_typed_object() {
    // Regression test for #151: `filters` used to be typed
    // `Option<serde_json::Map<String, serde_json::Value>>`, which schemars turns
    // into an unconstrained `{"type": "object"}` with no `additionalProperties`
    // constraint at all — no signal to a calling client about what a field's
    // condition may look like, and (per `SearchFiltersInput`'s doc comment) the
    // same under-specification that drove at least one real client to send a
    // nested-object parameter JSON-encoded as a string instead of an object.
    let schema = schemars::schema_for!(SearchParams);
    let root = schema.as_value();

    let filters_schema = &root["properties"]["filters"];
    // `Option<SearchFiltersInput>` becomes `anyOf: [<real schema>, {"type":
    // "null"}]`; find the non-null branch.
    let object_schema = filters_schema["anyOf"]
        .as_array()
        .expect("filters must offer a typed alternative, not a bare {}")
        .iter()
        .find(|branch| branch["type"] != serde_json::json!("null"))
        .expect("filters must have a non-null branch");

    // schemars refs a named type's schema into `$defs` rather than inlining it;
    // resolve it so the assertions below see the real shape (mirrors
    // `update_schema_definition_advertises_as_a_typed_object`'s resolution).
    let resolved = match object_schema["$ref"].as_str() {
        Some(reference) => &root["$defs"][reference.rsplit('/').next().unwrap()],
        None => object_schema,
    };

    assert_eq!(
        resolved["type"],
        serde_json::json!("object"),
        "filters must advertise as an object, got: {resolved}"
    );
    let condition_schema = &resolved["additionalProperties"];
    assert_ne!(
        *condition_schema,
        serde_json::json!(true),
        "a bare `additionalProperties: true` (schemars' rendering of an \
         unconstrained serde_json::Map) tells a client nothing about a \
         condition's shape — this is the exact bug being fixed, got: {resolved}"
    );

    let branches = condition_schema["anyOf"]
        .as_array()
        .expect("a filter condition must advertise its scalar/array/object forms");

    // Scalar branch: equality against a string, number, or boolean.
    assert!(
        branches
            .iter()
            .any(|b| b["type"].as_array().is_some_and(|types| {
                let types: Vec<&str> = types.iter().filter_map(|t| t.as_str()).collect();
                types.contains(&"string") && types.contains(&"number") && types.contains(&"boolean")
            })),
        "missing the scalar-equality branch, got: {condition_schema}"
    );

    // Array branch: any-of against a list of scalars.
    assert!(
        branches
            .iter()
            .any(|b| b["type"] == serde_json::json!("array") && !b["items"].is_null()),
        "missing the any-of-array branch, got: {condition_schema}"
    );

    // Object branch: named any_of/all_of/gte/lte/gt/lt properties.
    let object_branch = branches
        .iter()
        .find(|b| b["type"] == serde_json::json!("object"))
        .expect("missing the any_of/all_of/range object branch");
    for key in ["any_of", "all_of", "gte", "lte", "gt", "lt"] {
        assert!(
            !object_branch["properties"][key].is_null(),
            "condition object schema is missing documented key '{key}': \
             {object_branch}"
        );
    }
}

#[test]
fn search_filters_accepts_a_json_encoded_string_as_a_fallback() {
    // At least one real MCP client sends nested-object tool arguments as a
    // JSON-encoded string rather than an object, regardless of what the tool
    // schema advertises (same failure mode `FieldDefinitionInput` exists to
    // cover — see #151). `SearchFiltersInput` must tolerate that as a fallback,
    // and the parsed result must behave identically to the equivalent object.
    let params: SearchParams = serde_json::from_value(serde_json::json!({
        "filters": r#"{"type":"guide"}"#,
    }))
    .expect("a JSON-encoded filters string must deserialize");
    let query = build_document_query(&params).unwrap();
    assert_eq!(
        query.filters,
        vec![("type".to_string(), FieldFilter::AnyOf(vec!["guide".into()]))]
    );
}

#[test]
fn search_filters_rejects_a_string_that_is_not_valid_json() {
    let err = serde_json::from_value::<SearchFiltersInput>(serde_json::Value::String(
        "not json at all".to_string(),
    ))
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("not valid JSON"),
        "expected a message explaining the string wasn't parseable JSON, got: {msg}"
    );
}

#[test]
fn search_filters_rejects_a_json_array_naming_the_expected_shape() {
    let err =
        serde_json::from_value::<SearchFiltersInput>(serde_json::json!(["type", "x"])).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("JSON object"),
        "expected the error to name the expected shape, got: {msg}"
    );
    assert!(
        !msg.contains("SearchFiltersInput"),
        "a Rust type name is meaningless to an MCP client, got: {msg}"
    );
}

#[test]
fn overlong_path_prefix_is_rejected() {
    let err = build_document_query(&SearchParams {
        path_prefix: Some("a".repeat(MAX_FILTER_STR_LEN + 1)),
        ..Default::default()
    })
    .unwrap_err();
    assert!(format!("{:?}", err).contains("path_prefix too long"));
}

// --- search: granularity resolution ---

#[test]
fn granularity_defaults_to_chunk_when_a_query_is_present() {
    assert_eq!(default_granularity(true), Granularity::Chunk);
}

#[test]
fn granularity_defaults_to_document_when_no_query_is_present() {
    assert_eq!(default_granularity(false), Granularity::Document);
}

#[test]
fn explicit_granularity_overrides_the_default_either_direction() {
    assert_eq!(
        resolve_search_granularity(true, Some("document"), &Granularity::ALL, false).unwrap(),
        Granularity::Document
    );
    assert_eq!(
        resolve_search_granularity(false, Some("chunk"), &Granularity::ALL, false).unwrap(),
        Granularity::Chunk
    );
}

#[test]
fn unknown_granularity_is_rejected() {
    let err =
        resolve_search_granularity(true, Some("paragraph"), &Granularity::ALL, false).unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("unknown granularity"), "{msg}");
    assert!(
        msg.contains("expected one of 'chunk', 'document', 'section'"),
        "{msg}"
    );
}

#[test]
fn unknown_granularity_lists_only_the_effective_set() {
    // Default config: `section` configured but gated off by heading_metadata.
    let err = resolve_search_granularity(
        true,
        Some("paragraph"),
        &[Granularity::Chunk, Granularity::Document],
        true,
    )
    .unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("expected one of 'chunk', 'document'"), "{msg}");
    assert!(
        !msg.contains("section"),
        "a disabled granularity leaked: {msg}"
    );
}

#[test]
fn blank_query_is_treated_as_absent_for_granularity_purposes() {
    assert!(!query_is_present(&Some("   ".to_string())));
    assert!(!query_is_present(&None));
    assert!(query_is_present(&Some("pasta".to_string())));
}

// --- search: per-instance granularity restriction (#286) --------------
// Pure unit tests over `resolve_search_granularity` directly — no server,
// no config, matching `default_granularity`'s own tests above. The
// integration-level fallback/rejection behavior through the real `search`
// tool entry point is covered further below, once `schema_tool_server_*`
// helpers are in scope.

#[test]
fn explicit_request_for_an_enabled_granularity_is_accepted() {
    assert_eq!(
        resolve_search_granularity(
            true,
            Some("chunk"),
            &[Granularity::Chunk, Granularity::Document],
            false
        )
        .unwrap(),
        Granularity::Chunk
    );
}

#[test]
fn explicit_request_for_a_disabled_granularity_is_rejected_naming_the_enabled_set() {
    let err = resolve_search_granularity(true, Some("chunk"), &[Granularity::Document], false)
        .unwrap_err();
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("granularity 'chunk' is not enabled"),
        "got: {msg}"
    );
    assert!(msg.contains("document"), "must list the enabled set: {msg}");
    assert!(
        !msg.contains("heading metadata"),
        "the heading-metadata clause is section-specific and must not appear for \
         chunk: {msg}"
    );
}

#[test]
fn explicit_request_for_disabled_section_names_the_heading_metadata_gate() {
    // `section` configured but dropped because heading_metadata is off:
    // the message carries the same explanation `search_sections` gives.
    let err = resolve_search_granularity(
        true,
        Some("section"),
        &[Granularity::Chunk, Granularity::Document],
        true,
    )
    .unwrap_err();
    let msg = format!("{:?}", err);
    assert!(msg.contains("heading metadata enabled"), "got: {msg}");
    assert_no_internals_in_caller_error(&msg);
    assert!(
        !msg.contains("section granularity is unavailable"),
        "says 'not enabled' once, not twice: {msg}"
    );
    assert!(
        msg.contains("enabled: chunk, document"),
        "must label the effective set as what is enabled: {msg}"
    );
}

#[test]
fn explicit_request_for_administratively_disabled_section_does_not_blame_heading_metadata() {
    // `section` simply left out of search.granularities (flag state
    // irrelevant): heading_metadata is not the reason and must not be named.
    let err = resolve_search_granularity(
        true,
        Some("section"),
        &[Granularity::Chunk, Granularity::Document],
        false,
    )
    .unwrap_err();
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("granularity 'section' is not enabled"),
        "got: {msg}"
    );
    assert!(!msg.contains("heading metadata"), "got: {msg}");
}

/// For every non-empty effective set, the prose
/// `descriptions::granularity_description` gives callers (tool description
/// AND the schema's `granularity` property) must state exactly the default
/// `resolve_search_granularity` actually picks, with and without a query.
#[test]
fn granularity_description_matches_resolution_for_every_effective_set() {
    use Granularity::{Chunk, Document, Section};
    let sets: [&[Granularity]; 7] = [
        &[Chunk, Document, Section],
        &[Chunk, Document],
        &[Chunk, Section],
        &[Document, Section],
        &[Chunk],
        &[Document],
        &[Section],
    ];
    for set in sets {
        let desc = crate::descriptions::granularity_description(set);
        let with_query = resolve_search_granularity(true, None, set, false)
            .expect("every non-empty set can serve a query");
        let without_query = resolve_search_granularity(false, None, set, false).ok();

        // Never mentions a value outside the set.
        for g in Granularity::ALL {
            assert_eq!(
                desc.contains(&format!("`{}`", g.as_str())),
                set.contains(&g),
                "{set:?}: mention of `{g}` must match membership: {desc}"
            );
        }

        match without_query {
            Some(nq) if nq == with_query => {
                let q = with_query.as_str();
                assert!(
                    desc.contains(&format!("Defaults to `{q}`"))
                        || desc.contains(&format!("fixed to `{q}`")),
                    "{set:?}: default is `{q}` either way: {desc}"
                );
                assert!(!desc.contains(" without"), "{set:?}: {desc}");
            }
            Some(nq) => assert!(
                desc.contains(&format!(
                    "Defaults to `{}` with a query, `{}` without",
                    with_query.as_str(),
                    nq.as_str()
                )),
                "{set:?}: {desc}"
            ),
            None => {
                let q = with_query.as_str();
                assert!(
                    desc.contains(&format!("Defaults to `{q}`"))
                        || desc.contains(&format!("fixed to `{q}`")),
                    "{set:?}: default with a query is `{q}`: {desc}"
                );
                assert!(
                    desc.contains("every search needs a query"),
                    "{set:?}: no-query search fails here, so the description must say a \
                     query is always needed: {desc}"
                );
            }
        }
        if without_query.is_some() {
            assert!(
                !desc.contains("every search needs a query"),
                "{set:?}: enumeration works here: {desc}"
            );
        }
    }
}

#[test]
fn omitted_granularity_falls_back_to_the_first_enabled_one_supporting_the_query_mode() {
    // Ordinary default (chunk, since a query is present) is disabled —
    // falls back to document, which is enabled and supports either mode.
    assert_eq!(
        resolve_search_granularity(true, None, &[Granularity::Document], false).unwrap(),
        Granularity::Document
    );
    // Same, but section is the only other one enabled alongside document —
    // chunk (the ordinary default) is still skipped over.
    assert_eq!(
        resolve_search_granularity(
            true,
            None,
            &[Granularity::Document, Granularity::Section],
            false
        )
        .unwrap(),
        Granularity::Document
    );
}

#[test]
fn omitted_granularity_with_no_query_and_document_disabled_has_no_fallback() {
    // Enumeration's only granularity is `document`; `chunk`/`section` both
    // require a query, so if `document` is disabled there is nothing left
    // to fall back to — this is a deployment configuration problem, not a
    // caller one, and must be reported as such rather than silently
    // picking something that can't work.
    let err = resolve_search_granularity(
        false,
        None,
        &[Granularity::Chunk, Granularity::Section],
        false,
    )
    .unwrap_err();
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("no enabled search granularity supports"),
        "got: {msg}"
    );
    assert!(msg.contains("enumeration"), "got: {msg}");
}

#[test]
fn granularity_supports_mode_matches_search_query_requirements() {
    // document works either way; chunk/section both need a query — the
    // same requirement `search`'s own (false, Chunk)/(false, Section)
    // match arms enforce.
    assert!(granularity_supports_mode(Granularity::Document, true));
    assert!(granularity_supports_mode(Granularity::Document, false));
    assert!(granularity_supports_mode(Granularity::Chunk, true));
    assert!(!granularity_supports_mode(Granularity::Chunk, false));
    assert!(granularity_supports_mode(Granularity::Section, true));
    assert!(!granularity_supports_mode(Granularity::Section, false));
}

#[tokio::test]
async fn chunk_granularity_without_a_query_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);

    let err = server
        .search(Parameters(SearchParams {
            granularity: Some("chunk".to_string()),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert!(
        format!("{:?}", err).contains("requires a query"),
        "got: {err:?}"
    );
}

// --- search: search_grouped adapter, through the real tool entry point ---
//
// A live embeddings endpoint and a live Qdrant are both unavailable in this
// environment (and `EmbedClient`'s retry/backoff classifies a refused local
// connection as transient, retrying for up to two minutes — see
// `embed::embed_backoff` — so even a deliberate connection failure is too
// slow to use as a fast test signal). These tests therefore cover only what
// is reachable BEFORE `retrieval::search_grouped` ever calls the embedder:
// routing, and that `modified_after`/`fields` are parsed by this specific
// branch rather than silently dropped. The response-shape assembly and
// `min_score` forwarding (which has no offline-observable effect) need a
// live Qdrant + embeddings stack to verify; see this module's Report for
// that gap.

#[tokio::test]
async fn search_grouped_routes_a_document_query_to_the_grouped_path_not_enumeration() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);
    seed_document(
        &server,
        "notes/a.md",
        serde_json::json!({ "title": "A", "random_field": "x" }),
    )
    .await;

    let mut filters = serde_json::Map::new();
    filters.insert("random_field".into(), serde_json::json!("x"));

    // Control: enumeration mode (no query) accepts this filter fine, since
    // it never requires a Qdrant payload index — so the divergence below is
    // attributable to routing, not to the filter being universally invalid.
    let enumerate_result = server
        .search(Parameters(SearchParams {
            filters: Some(SearchFiltersInput(filters.clone())),
            ..Default::default()
        }))
        .await
        .expect("enumeration mode does not require a Qdrant payload index");
    assert_eq!(
        enumerate_result.structured_content.unwrap()["total"],
        serde_json::json!(1)
    );

    // The same filter, with a query AND an explicit `document` granularity,
    // must route to the grouped (query+document) path — which DOES require
    // every filter field to carry a Qdrant payload index — not silently
    // fall back to enumeration.
    let err = server
        .search(Parameters(SearchParams {
            query: Some("test".to_string()),
            granularity: Some("document".to_string()),
            filters: Some(SearchFiltersInput(filters)),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert!(
        format!("{:?}", err).contains("not indexed for Qdrant queries"),
        "a query+document search must route to the grouped path, which rejects an \
         unindexed filter field before ever reaching Qdrant; got: {err:?}"
    );
}

#[tokio::test]
async fn search_grouped_parses_modified_after_before_reaching_qdrant() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);

    let err = server
        .search(Parameters(SearchParams {
            query: Some("test".to_string()),
            granularity: Some("document".to_string()),
            modified_after: Some("not-a-date".to_string()),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert!(
        format!("{:?}", err).contains("invalid date"),
        "the grouped adapter must parse modified_after itself rather than dropping \
         it before retrieval::search_grouped ever runs; got: {err:?}"
    );
}

#[tokio::test]
async fn search_grouped_validates_fields_count_before_reaching_qdrant() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);

    let err = server
        .search(Parameters(SearchParams {
            query: Some("test".to_string()),
            granularity: Some("document".to_string()),
            fields: Some(
                (0..(MAX_LIST_FILTERS + 1))
                    .map(|i| format!("f{i}"))
                    .collect(),
            ),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert!(
        format!("{:?}", err).contains("too many fields requested"),
        "the grouped adapter must validate fields itself, proving the param actually \
         reaches this branch rather than being silently dropped; got: {err:?}"
    );
}

#[tokio::test]
async fn search_grouped_rejects_explain_true() {
    // #132: `explain: true` used to be a silent no-op at document
    // granularity — accepted, never producing a score breakdown, with no
    // signal to the caller that it did nothing. Must now be rejected
    // outright, and rejected before ever reaching Qdrant (no live Qdrant is
    // configured in this test harness, so a Qdrant-side error here would
    // prove the rejection did NOT happen early).
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);

    let err = server
        .search(Parameters(SearchParams {
            query: Some("test".to_string()),
            granularity: Some("document".to_string()),
            explain: Some(true),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert!(
        format!("{:?}", err).contains("chunk-granularity only"),
        "explain: true at document granularity must be rejected with an \
         explicit error, not silently ignored; got: {err:?}"
    );
}

#[tokio::test]
async fn search_chunks_rejects_fields() {
    // #132 audit follow-up: `fields` used to be a silent no-op at chunk
    // granularity — accepted, never read by `search_chunks` or
    // `build_chunk_search_payload`, the mirror image of `explain` silently
    // no-opping at document granularity. Must now be rejected outright.
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);

    let err = server
        .search(Parameters(SearchParams {
            query: Some("test".to_string()),
            granularity: Some("chunk".to_string()),
            fields: Some(vec!["status".to_string()]),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert!(
        format!("{:?}", err).contains("fields only applies to document-granularity results"),
        "fields at chunk granularity must be rejected with an explicit error, \
         not silently ignored; got: {err:?}"
    );
}

// --- search: section granularity (#286) ---

#[tokio::test]
async fn search_rejects_an_explicitly_disabled_granularity_through_the_real_tool() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server_with_granularities(&tmp, &[Granularity::Document]);

    let err = server
        .search(Parameters(SearchParams {
            query: Some("test".to_string()),
            granularity: Some("chunk".to_string()),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("granularity 'chunk' is not enabled"),
        "got: {msg}"
    );
    assert!(msg.contains("document"), "got: {msg}");
}

#[tokio::test]
async fn search_falls_back_past_a_disabled_default_granularity_through_the_real_tool() {
    // `chunk` is the ordinary default for a query, but it's disabled
    // here — the call must not error out; it should fall back to
    // `document` (still enabled) and actually route to the grouped
    // (query+document) path. Proven the same way
    // `search_grouped_routes_a_document_query_to_the_grouped_path_not_enumeration`
    // proves routing: an unindexed filter field is rejected only on
    // that path, before ever reaching Qdrant.
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server_with_granularities(&tmp, &[Granularity::Document]);
    // In use, so the filter vocabulary check lets it through to routing.
    seed_document(
        &server,
        "notes/a.md",
        serde_json::json!({ "title": "A", "random_field": "x" }),
    )
    .await;

    let mut filters = serde_json::Map::new();
    filters.insert("random_field".into(), serde_json::json!("x"));

    let err = server
        .search(Parameters(SearchParams {
            query: Some("test".to_string()),
            filters: Some(SearchFiltersInput(filters)),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert!(
        format!("{:?}", err).contains("not indexed for Qdrant queries"),
        "an omitted granularity with chunk disabled must fall back to the document \
         (grouped) path rather than erroring outright; got: {err:?}"
    );
}

#[tokio::test]
async fn search_reports_no_fallback_as_a_configuration_problem_through_the_real_tool() {
    // Only `chunk` enabled, which requires a query — enumeration (no
    // query, no explicit granularity) has no enabled granularity that
    // can serve it.
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server_with_granularities(&tmp, &[Granularity::Chunk]);

    let err = server
        .search(Parameters(SearchParams::default()))
        .await
        .unwrap_err();
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("no enabled search granularity supports"),
        "got: {msg}"
    );
    assert!(msg.contains("enumeration"), "got: {msg}");
}

#[tokio::test]
async fn section_granularity_without_a_query_is_rejected() {
    // Uses `schema_tool_server_with_heading_metadata`, not plain
    // `schema_tool_server` — with heading_metadata off, `section` is
    // already excluded from the effective granularity set (#286), so `resolve_search_granularity`'s disabled-granularity check
    // would fire first and this test would observe THAT message
    // instead of the one under test here. With the flag on, `section`
    // is enabled and this exercises the `(query_present=false,
    // Granularity::Section)` arm specifically.
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server_with_heading_metadata(&tmp);

    let err = server
        .search(Parameters(SearchParams {
            granularity: Some("section".to_string()),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert!(
        format!("{:?}", err).contains("requires a query"),
        "got: {err:?}"
    );
}

#[tokio::test]
async fn section_granularity_without_heading_metadata_is_rejected() {
    // `make_test_resolved_config`'s default has `chunking.heading_metadata`
    // off (matching `ChunkingConfig::default()`), so `schema_tool_server`
    // (unlike `schema_tool_server_with_heading_metadata` above) is exactly
    // the fixture this test needs.
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);

    let err = server
        .search(Parameters(SearchParams {
            query: Some("test".to_string()),
            granularity: Some("section".to_string()),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("heading metadata enabled"), "got: {msg}");
    assert_no_internals_in_caller_error(&msg);
}

#[tokio::test]
async fn search_sections_rejects_explain_true() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server_with_heading_metadata(&tmp);

    let err = server
        .search(Parameters(SearchParams {
            query: Some("test".to_string()),
            granularity: Some("section".to_string()),
            explain: Some(true),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert!(
        format!("{:?}", err).contains("chunk-granularity only"),
        "got: {err:?}"
    );
}

#[tokio::test]
async fn search_sections_rejects_fields() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server_with_heading_metadata(&tmp);

    let err = server
        .search(Parameters(SearchParams {
            query: Some("test".to_string()),
            granularity: Some("section".to_string()),
            fields: Some(vec!["status".to_string()]),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert!(
        format!("{:?}", err).contains("fields only applies to document-granularity results"),
        "got: {err:?}"
    );
}

#[tokio::test]
async fn heading_prefix_is_rejected_when_the_flag_is_off() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);

    let err = server
        .search(Parameters(SearchParams {
            query: Some("test".to_string()),
            heading_prefix: Some(vec!["Conditions".to_string()]),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert!(
        format!("{:?}", err).contains("heading metadata enabled"),
        "got: {err:?}"
    );
}

#[tokio::test]
async fn heading_prefix_empty_list_is_rejected_even_with_the_flag_on() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server_with_heading_metadata(&tmp);

    let err = server
        .search(Parameters(SearchParams {
            query: Some("test".to_string()),
            heading_prefix: Some(vec![]),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert!(
        format!("{:?}", err).contains("must not be empty"),
        "got: {err:?}"
    );
}

#[tokio::test]
async fn heading_prefix_off_error_names_no_internals() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);
    let err = server
        .search(Parameters(SearchParams {
            query: Some("test".to_string()),
            heading_prefix: Some(vec!["Conditions".to_string()]),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("Search without heading_prefix"), "got: {msg}");
    assert_no_internals_in_caller_error(&msg);
}

#[tokio::test]
async fn heading_prefix_blank_segment_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server_with_heading_metadata(&tmp);
    // Whitespace, or only invisible characters (blank once normalized, so
    // it would otherwise lower to a key that silently matches nothing).
    for blank in ["  ", "\u{200b}", " \u{200d}\u{feff} "] {
        let err = server
            .search(Parameters(SearchParams {
                query: Some("test".to_string()),
                heading_prefix: Some(vec!["Conditions".to_string(), blank.to_string()]),
                ..Default::default()
            }))
            .await
            .unwrap_err();
        assert!(
            format!("{err:?}").contains("empty or blank segments"),
            "{blank:?} got: {err:?}"
        );
    }
}

#[tokio::test]
async fn heading_prefix_without_a_query_is_rejected_not_ignored() {
    // Enumeration reads the state DB, which has no heading data. Before
    // this was rejected, the filter was silently dropped and every
    // document came back with an exact `total`, as if it had matched.
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server_with_heading_metadata(&tmp);
    seed_document(&server, "notes/a.md", serde_json::json!({ "title": "A" })).await;

    let err = server
        .search(Parameters(SearchParams {
            heading_prefix: Some(vec!["Conditions".to_string()]),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert!(
        format!("{err:?}").contains("heading_prefix requires a query"),
        "got: {err:?}"
    );
}

#[tokio::test]
async fn heading_prefix_without_a_query_still_reports_the_flag_or_empty_list_first() {
    // With the flag off, the more fundamental error wins over "requires a
    // query", matching query mode's validation order.
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);
    let err = server
        .search(Parameters(SearchParams {
            heading_prefix: Some(vec!["Conditions".to_string()]),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert!(
        format!("{err:?}").contains("heading metadata enabled"),
        "got: {err:?}"
    );

    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server_with_heading_metadata(&tmp);
    let err = server
        .search(Parameters(SearchParams {
            heading_prefix: Some(vec![]),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert!(
        format!("{err:?}").contains("must not be empty"),
        "got: {err:?}"
    );
}

#[test]
fn heading_prefix_key_normalizes_case_and_whitespace() {
    assert_eq!(
        crate::heading::heading_prefix_key(&["conditions"]),
        "conditions"
    );
    assert_eq!(
        crate::heading::heading_prefix_key(&[" Conditions ", "Blinded  Condition"]),
        ["conditions", "blinded condition"].join(crate::heading::HEADING_KEY_SEPARATOR)
    );
    // Same key whichever spelling the caller used.
    assert_eq!(
        crate::heading::heading_prefix_key(&["CONDITIONS"]),
        crate::heading::heading_prefix_key(&["Conditions"])
    );
}

#[test]
fn heading_prefix_condition_matches_the_normalized_key() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = make_test_resolved_config(tmp.path());
    Arc::make_mut(&mut config).chunking.heading_metadata = true;
    let condition = heading_prefix_condition(
        &Some(vec!["  Conditions".to_string(), "BLINDED".to_string()]),
        &config,
    )
    .unwrap()
    .expect("a non-empty prefix lowers to a condition");
    let expected = qdrant_client::qdrant::Condition::matches(
        crate::qdrant::HEADING_PREFIXES_KEY,
        ["conditions", "blinded"].join(crate::heading::HEADING_KEY_SEPARATOR),
    );
    assert_eq!(condition, expected);
}

#[test]
fn section_search_payload_distinguishes_section_preamble_and_document_rows() {
    let hit = |heading_path: &[&str], scope| retrieval::SectionHit {
        file_path: "a.md".to_string(),
        heading_path: heading_path.iter().map(|s| s.to_string()).collect(),
        line_start: 4,
        line_end: 7,
        hit_line_start: 5,
        hit_line_end: 6,
        score: 0.5,
        scope,
    };
    let structured = build_section_search_payload(
        &[
            hit(&["A", "B"], retrieval::SectionScope::Section),
            hit(&[], retrieval::SectionScope::Preamble),
            hit(&[], retrieval::SectionScope::WholeDocument),
        ],
        false,
        false,
    );
    let rows = structured["results"].as_array().unwrap();
    // An ordinary section is the default and carries no `scope`; the two
    // heading-less kinds say what they are and carry no empty heading_path.
    let scopes: Vec<Option<&str>> = rows.iter().map(|r| r["scope"].as_str()).collect();
    assert_eq!(scopes, vec![None, Some("preamble"), Some("whole_document")]);
    assert_eq!(rows[0]["heading_path"], serde_json::json!(["A", "B"]));
    assert!(rows[1].get("heading_path").is_none(), "{}", rows[1]);
    assert_eq!(
        (
            rows[0]["hit_line_start"].as_u64(),
            rows[0]["hit_line_end"].as_u64()
        ),
        (Some(5), Some(6))
    );
    assert!(structured.get("path_prefix_truncated").is_none());
}

#[test]
fn annotate_heading_results_marks_the_response_only_when_a_note_applies() {
    let mut response = serde_json::json!({ "returned": 1 });
    annotate_heading_results(&mut response, None);
    assert_eq!(
        response,
        serde_json::json!({ "returned": 1 }),
        "no note → untouched"
    );

    annotate_heading_results(&mut response, Some("Note: indexing."));
    assert_eq!(response["indexing_in_progress"], serde_json::json!(true));
}
// --- search: query-mode filter lowering ---

/// A `ResolvedConfig` (with `frontmatter.indexed_fields` set to `fields`) paired
/// with the `SchemaCache` built from that same config with no `.kb-schema.yaml`
/// on disk, whose root schema falls back to `config.indexed_fields` (see
/// `SchemaCache::build`'s doc comment) — enough to exercise
/// `build_query_conditions`'s indexed-field check (now the
/// `qdrant::all_indexed_fields` union) without a real KB tree. Callers that
/// want to exercise the *config-only* (legacy) half of that union build a
/// `SchemaCache` from an empty field list directly instead.
fn schema_cache_with_indexed(fields: &[&str]) -> (Arc<ResolvedConfig>, SchemaCache) {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = make_test_resolved_config(tmp.path());
    Arc::make_mut(&mut config).frontmatter = crate::config::FrontmatterConfig {
        indexed_fields: fields.iter().map(|f| f.to_string()).collect(),
        ..Default::default()
    };
    let schemas = SchemaCache::build_for_test(tmp.path(), &config.frontmatter);
    (config, schemas)
}

fn filters_param(json: serde_json::Value) -> SearchParams {
    SearchParams {
        filters: Some(SearchFiltersInput(json.as_object().unwrap().clone())),
        ..Default::default()
    }
}

#[test]
fn query_mode_rejects_a_filter_on_a_field_with_no_payload_index() {
    let (config, schemas) = schema_cache_with_indexed(&["type"]);
    let err = build_query_conditions(
        &filters_param(serde_json::json!({ "untracked_field": "x" })),
        &config,
        &schemas,
    )
    .unwrap_err();
    let msg = format!("{:?}", err);
    assert!(msg.contains("untracked_field"), "got: {msg}");
    assert!(msg.contains("indexed: true"), "got: {msg}");
}

#[test]
fn query_mode_accepts_a_filter_on_an_indexed_field() {
    let (config, schemas) = schema_cache_with_indexed(&["type"]);
    let conditions = build_query_conditions(
        &filters_param(serde_json::json!({ "type": "guide" })),
        &config,
        &schemas,
    )
    .unwrap();
    assert_eq!(conditions.len(), 1);
}

/// A field indexed only via the legacy `frontmatter.indexed_fields` config
/// list — never declared `indexed: true` in any `.kb-schema.yaml` — is
/// genuinely filterable in Qdrant (`qdrant::all_indexed_fields` unions both
/// sources when creating payload indexes), so `build_query_conditions` must
/// accept it too rather than rejecting it by name for only checking the
/// schema half of that union.
#[test]
fn query_mode_accepts_a_legacy_config_only_indexed_field() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = make_test_resolved_config(tmp.path());
    Arc::make_mut(&mut config).frontmatter = crate::config::FrontmatterConfig {
        indexed_fields: vec!["legacy_only".to_string()],
        ..Default::default()
    };
    // Built from an empty field list: nothing is `indexed: true` in any schema,
    // so the schema half of the union contributes nothing for this field.
    let schemas =
        SchemaCache::build_for_test(tmp.path(), &crate::config::FrontmatterConfig::default());

    let conditions = build_query_conditions(
        &filters_param(serde_json::json!({ "legacy_only": "x" })),
        &config,
        &schemas,
    )
    .unwrap();
    assert_eq!(conditions.len(), 1);
}

/// When the SAME field name is declared indexed by both a `.kb-schema.yaml`
/// (with an explicit type) and the legacy `frontmatter.indexed_fields` list,
/// `qdrant::all_indexed_fields` must let the schema's kind win — not fall
/// back to the legacy union's implicit keyword default — since that is
/// exactly what determines whether a numeric range filter on the field is
/// accepted here or rejected as "declared as Keyword".
#[test]
fn query_mode_accepts_a_range_filter_when_schema_kind_beats_the_legacy_default() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(
        tmp.path().join(crate::schema::SCHEMA_FILE_NAME),
        "fields:\n  prep_minutes:\n    type: integer\n    indexed: true\n",
    )
    .unwrap();
    let mut config = make_test_resolved_config(tmp.path());
    Arc::make_mut(&mut config).frontmatter = crate::config::FrontmatterConfig {
        indexed_fields: vec!["prep_minutes".to_string()],
        ..Default::default()
    };
    let schemas = SchemaCache::build_for_test(tmp.path(), &config.frontmatter);

    let conditions = build_query_conditions(
        &filters_param(serde_json::json!({ "prep_minutes": { "gte": 10 } })),
        &config,
        &schemas,
    )
    .expect(
        "a numeric range must be accepted once the schema's integer kind wins over \
         the legacy keyword default",
    );
    assert_eq!(conditions.len(), 1);
}

#[test]
fn query_mode_filters_reproduce_the_old_domain_type_tags_conditions() {
    // `filters` is the one and only path to narrowing a query-mode search now;
    // this pins that {"domain": ..., "type": ..., "tags": [...]} through it
    // produces the exact same Qdrant conditions the deleted domain/type/tags
    // params used to build directly.
    let (config, schemas) = schema_cache_with_indexed(&["domain", "type", "tags"]);
    let conditions = build_query_conditions(
        &filters_param(serde_json::json!({
            "domain": "sysadmin",
            "type": "guide",
            "tags": ["rust", "rag"],
        })),
        &config,
        &schemas,
    )
    .unwrap();

    assert_eq!(conditions.len(), 3);
    // Every keyword-kind `AnyOf` — regardless of value count — lowers through the
    // same `Condition::matches(_, Vec<String>)` (match-any) shape the old
    // `tags: Vec<String>` array filter always used; `domain`/`type` (formerly
    // single-value scalar matches) now go through that same shape too, since
    // `filters` no longer distinguishes "one value" from "any of these values".
    // Keys are under the payload's `fm` namespace (#350).
    assert!(
        conditions.contains(&qdrant_client::qdrant::Condition::matches(
            "fm.domain",
            vec!["sysadmin".to_string()]
        ))
    );
    assert!(
        conditions.contains(&qdrant_client::qdrant::Condition::matches(
            "fm.type",
            vec!["guide".to_string()]
        ))
    );
    assert!(
        conditions.contains(&qdrant_client::qdrant::Condition::matches(
            "fm.tags",
            vec!["rust".to_string(), "rag".to_string()]
        ))
    );
}

// --- schema tool helpers ---

#[tokio::test]
async fn search_filter_on_an_unknown_field_lists_the_filterable_fields() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(
        &tmp,
        "",
        "derived:\n  domain: {from: path_segment, index: 0}\n\
         fields:\n  status:\n    type: enum\n    values: [active]\n",
    );
    let server = schema_tool_server(&tmp);
    seed_document(&server, "notes/a.md", serde_json::json!({ "owner": "me" })).await;

    for backend_query in [None, Some("q".to_string())] {
        let mut filters = serde_json::Map::new();
        filters.insert("staus".into(), serde_json::json!("active"));
        let err = server
            .search(Parameters(SearchParams {
                query: backend_query.clone(),
                filters: Some(SearchFiltersInput(filters)),
                ..Default::default()
            }))
            .await
            .unwrap_err();
        assert!(
            err.message.contains("unknown filter field 'staus'"),
            "{}",
            err.message
        );
        for known in ["status", "owner", "domain"] {
            assert!(err.message.contains(known), "{}", err.message);
        }
        assert!(!err.message.contains("file_path"), "{}", err.message);
    }

    // A field documents use but no schema declares is known.
    let mut filters = serde_json::Map::new();
    filters.insert("owner".into(), serde_json::json!("me"));
    let ok = server
        .search(Parameters(SearchParams {
            filters: Some(SearchFiltersInput(filters)),
            ..Default::default()
        }))
        .await
        .unwrap();
    assert_eq!(
        ok.structured_content.unwrap()["total"],
        serde_json::json!(1)
    );
}

#[tokio::test]
async fn search_correlated_list_filter_matches_within_one_element() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(
        &tmp,
        "",
        "fields:\n  npcs:\n    type: list\n    items:\n      type: object\n      fields:\n        \
         role:\n          type: enum\n          values: [guard, cook]\n        level:\n          \
         type: integer\n",
    );
    let server = schema_tool_server(&tmp);
    seed_document(
        &server,
        "split.md",
        serde_json::json!({ "npcs": [{ "role": "guard", "level": 1 }, { "role": "cook", "level": 5 }] }),
    )
    .await;
    seed_document(
        &server,
        "same.md",
        serde_json::json!({ "npcs": [{ "role": "cook", "level": 2 }, { "role": "guard", "level": 5 }] }),
    )
    .await;

    let search = |filter: serde_json::Value, query: Option<&str>| {
        let mut filters = serde_json::Map::new();
        filters.insert("npcs[]".into(), filter);
        server.search(Parameters(SearchParams {
            query: query.map(str::to_string),
            filters: Some(SearchFiltersInput(filters)),
            ..Default::default()
        }))
    };

    let found = search(
        serde_json::json!({ "role": "guard", "level": { "gte": 3 } }),
        None,
    )
    .await
    .unwrap()
    .structured_content
    .unwrap();
    assert_eq!(found["total"], serde_json::json!(1), "{found}");
    assert_eq!(found["documents"][0]["file_path"], "same.md", "{found}");

    // An unknown sub-field lists the list's fields, not every field.
    let err = search(serde_json::json!({ "rank": "x" }), None)
        .await
        .unwrap_err();
    assert!(
        err.message.contains(
            "filter 'npcs[]': unknown sub-field 'npcs[].rank'; fields of npcs[]: \
             npcs[].level, npcs[].role"
        ),
        "{}",
        err.message
    );
    // A value outside the sub-field's closed set.
    let err = search(serde_json::json!({ "role": "king" }), None)
        .await
        .unwrap_err();
    assert!(
        err.message
            .contains("filter 'npcs[].role': 'king' is not an allowed value; allowed: cook, guard"),
        "{}",
        err.message
    );
    // Query mode lowers one to a nested condition (#350) — but, like any field,
    // only over sub-fields with a payload index; these are not `indexed`.
    let mut filters = serde_json::Map::new();
    filters.insert("npcs[]".into(), serde_json::json!({ "role": "guard" }));
    let params = SearchParams {
        query: Some("q".into()),
        filters: Some(SearchFiltersInput(filters)),
        ..Default::default()
    };
    let schemas = crate::schema::load_shared(&server.schema_cache);
    let err = build_query_conditions(&params, &server.config(), &schemas).unwrap_err();
    assert!(
        err.message
            .contains("filter 'npcs[].role' is not indexed for Qdrant queries"),
        "{}",
        err.message
    );
}

#[tokio::test]
async fn query_mode_lowers_correlated_and_date_filters_over_indexed_fields() {
    use qdrant_client::qdrant::condition::ConditionOneOf;
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(
        &tmp,
        "",
        "fields:\n  published:\n    type: date\n    indexed: true\n  npcs:\n    type: list\n    \
         items:\n      type: object\n      fields:\n        role:\n          type: enum\n          \
         values: [guard, cook]\n          indexed: true\n        level:\n          type: \
         integer\n          indexed: true\n",
    );
    let server = schema_tool_server(&tmp);
    let schemas = crate::schema::load_shared(&server.schema_cache);
    let lower = |field: &str, filter: serde_json::Value| {
        let mut filters = serde_json::Map::new();
        filters.insert(field.into(), filter);
        let params = SearchParams {
            query: Some("q".into()),
            filters: Some(SearchFiltersInput(filters)),
            ..Default::default()
        };
        build_query_conditions(&params, &server.config(), &schemas)
    };

    let nested = lower(
        "npcs[]",
        serde_json::json!({ "role": "guard", "level": { "gte": 3 } }),
    )
    .unwrap();
    let Some(ConditionOneOf::Nested(n)) = &nested[0].condition_one_of else {
        panic!("expected a nested condition: {nested:?}");
    };
    assert_eq!(n.key, "fm.npcs");
    let keys: Vec<String> = n
        .filter
        .as_ref()
        .unwrap()
        .must
        .iter()
        .map(|c| match &c.condition_one_of {
            Some(ConditionOneOf::Field(f)) => f.key.clone(),
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(keys, vec!["level", "role"]);

    let dated = lower("published", serde_json::json!({ "gte": "2024-01-01" })).unwrap();
    let Some(ConditionOneOf::Field(f)) = &dated[0].condition_one_of else {
        panic!("{dated:?}");
    };
    assert_eq!(f.key, "fm.published");
    assert_eq!(
        f.datetime_range.unwrap().gte.unwrap().seconds,
        1_704_067_200,
        "midnight UTC, as the metadata index's epoch seconds"
    );
}

#[tokio::test]
async fn search_refuses_ranges_that_could_never_compare_a_number() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(
        &tmp,
        "",
        "fields:\n  published:\n    type: date\n    indexed: true\n  count:\n    \
         type: integer\n    indexed: true\n  code:\n    type: text\n  npcs:\n    \
         type: list\n    items:\n      type: object\n      fields:\n        name:\n          \
         type: text\n",
    );
    let server = schema_tool_server(&tmp);
    seed_document(
        &server,
        "a.md",
        serde_json::json!({ "code": "7", "loose": 3, "published": "2024-02-01",
                            "npcs": [{ "name": "Ada" }] }),
    )
    .await;
    let search = |field: &str, filter: serde_json::Value, query: Option<&str>| {
        let mut filters = serde_json::Map::new();
        filters.insert(field.into(), filter);
        server.search(Parameters(SearchParams {
            query: query.map(str::to_string),
            filters: Some(SearchFiltersInput(filters)),
            ..Default::default()
        }))
    };

    // Declared text: no stored number, so a range is refused, not empty.
    for (field, filter) in [
        ("code", serde_json::json!({ "gte": 3 })),
        ("npcs[]", serde_json::json!({ "name": { "gte": 1 } })),
    ] {
        let err = search(field, filter, None).await.unwrap_err();
        assert!(
            err.message.contains("a range needs a numeric field")
                && err.message.contains("declared `text`")
                && err.message.contains("`integer` or `number`"),
            "{field}: {}",
            err.message
        );
    }
    // Undeclared and date fields range as before when listing.
    search("loose", serde_json::json!({ "gte": 1 }), None)
        .await
        .unwrap();
    search(
        "published",
        serde_json::json!({ "gte": "2024-01-01" }),
        None,
    )
    .await
    .unwrap();

    // With a query (#350) a date range lowers too, on the epoch-seconds axis the
    // metadata index uses: a date field takes a date or a numeric (epoch) bound,
    // and a date bound on a numeric field compares as its epoch seconds, exactly
    // as the listing compares `value_num`.
    let schemas = crate::schema::load_shared(&server.schema_cache);
    for (field, filter) in [
        ("published", serde_json::json!({ "gte": "2024-01-01" })),
        ("published", serde_json::json!({ "gte": 1 })),
        ("count", serde_json::json!({ "gte": "2024-01-01" })),
    ] {
        let mut filters = serde_json::Map::new();
        filters.insert(field.into(), filter.clone());
        let params = SearchParams {
            query: Some("q".into()),
            filters: Some(SearchFiltersInput(filters)),
            ..Default::default()
        };
        let conditions = build_query_conditions(&params, &server.config(), &schemas)
            .unwrap_or_else(|e| panic!("{field} {filter}: {}", e.message));
        assert_eq!(conditions.len(), 1, "{field} {filter}");
    }
}

#[test]
fn correlated_filters_parse_with_caps_and_one_level_only() {
    let parsed = parse_field_filter(
        "npcs[]",
        &serde_json::json!({ "role": "guard", "level": { "gte": 3 } }),
    )
    .unwrap();
    assert_eq!(
        parsed,
        FieldFilter::Elem(vec![
            (
                "npcs[].level".into(),
                FieldFilter::Range {
                    gte: Some(3.0),
                    lte: None,
                    gt: None,
                    lt: None
                }
            ),
            (
                "npcs[].role".into(),
                FieldFilter::AnyOf(vec!["guard".into()])
            ),
        ])
    );
    let refused =
        |field: &str, raw: serde_json::Value| parse_field_filter(field, &raw).unwrap_err();
    assert!(refused("npcs[]", serde_json::json!("guard")).contains("takes an object"));
    assert!(refused("npcs[]", serde_json::json!({})).contains("at least one"));
    assert!(refused("a[].b[]", serde_json::json!({ "c": 1 })).contains("outside any other list"));
    assert!(refused("[]", serde_json::json!({ "c": 1 })).contains("outside any other list"));
    let nested = refused("npcs[]", serde_json::json!({ "gear[]": { "x": 1 } }));
    assert!(nested.contains("do not nest"), "{nested}");
    assert!(
        nested.contains("e.g. 'gear[].name'"),
        "the hint names a path the vocabulary check accepts: {nested}"
    );
    assert!(refused("npcs[]", serde_json::json!({ "any_of": ["x"] })).contains("is an operator"));
    let many: serde_json::Map<String, serde_json::Value> = (0..=MAX_LIST_FILTERS)
        .map(|i| (format!("f{i}"), serde_json::json!(1)))
        .collect();
    assert!(refused("npcs[]", serde_json::Value::Object(many)).contains("too many conditions"));
    let values: Vec<String> = (0..=MAX_LIST_FILTERS).map(|i| format!("v{i}")).collect();
    assert!(
        refused(
            "npcs[]",
            serde_json::json!({ "tags": { "all_of": values } })
        )
        .contains("too many conditions"),
        "each all_of value is a row test"
    );
    // A sub-condition is any ordinary filter, its own caps included.
    let long = "x".repeat(MAX_FILTER_STR_LEN + 1);
    assert!(refused("npcs[]", serde_json::json!({ "role": long })).contains("too long"));
}

#[test]
fn range_bounds_accept_dates_and_timestamps_as_epoch_seconds() {
    let parsed = parse_field_filter(
        "on",
        &serde_json::json!({ "gte": "2024-01-02", "lt": "2024-01-03T00:00:00+01:00" }),
    )
    .unwrap();
    assert_eq!(
        parsed,
        FieldFilter::Range {
            gte: crate::document_fields::epoch_seconds("2024-01-02"),
            lte: None,
            gt: None,
            lt: Some(1_704_236_400.0),
        }
    );
    let err = parse_field_filter("on", &serde_json::json!({ "gte": "soon" })).unwrap_err();
    assert!(err.contains("must be a number, or a date"), "{err}");
}

#[tokio::test]
async fn search_filter_value_outside_a_closed_set_lists_the_allowed_values() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(
        &tmp,
        "",
        "fields:\n  status:\n    type: enum\n    values: [active, draft]\n",
    );
    write_schema_file(
        &tmp,
        "food",
        "fields:\n  status:\n    type: enum\n    values: [active, cooking]\n",
    );
    let server = schema_tool_server(&tmp);
    // A legacy value no schema lists any more, still used by a document.
    seed_document(
        &server,
        "notes/a.md",
        serde_json::json!({ "status": "legacy" }),
    )
    .await;

    let search = |status: &str, path_prefix: Option<&str>| {
        let mut filters = serde_json::Map::new();
        filters.insert("status".into(), serde_json::json!(status));
        server.search(Parameters(SearchParams {
            filters: Some(SearchFiltersInput(filters)),
            path_prefix: path_prefix.map(str::to_string),
            ..Default::default()
        }))
    };

    let err = search("actve", None).await.unwrap_err();
    assert!(
        err.message.contains(
            "filter 'status': 'actve' is not an allowed value; allowed: active, cooking, draft"
        ),
        "the union across scopes without a path: {}",
        err.message
    );
    let err = search("actve", Some("food/")).await.unwrap_err();
    assert!(
        err.message.contains("allowed: active, cooking"),
        "path_prefix narrows to the scopes it names: {}",
        err.message
    );
    // Permitted in some scope, or used by a document: never refused.
    search("cooking", None).await.unwrap();
    search("legacy", Some("food/")).await.unwrap();
}

#[tokio::test]
async fn search_filter_any_of_is_refused_only_when_no_value_could_match() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(
        &tmp,
        "",
        "fields:\n  status:\n    type: enum\n    values: [active, draft]\n",
    );
    let server = schema_tool_server(&tmp);
    // A legacy value no schema lists any more, still used by a document.
    seed_document(
        &server,
        "notes/a.md",
        serde_json::json!({ "status": "legacy" }),
    )
    .await;

    let search = |filter: serde_json::Value| {
        let mut filters = serde_json::Map::new();
        filters.insert("status".into(), filter);
        server.search(Parameters(SearchParams {
            filters: Some(SearchFiltersInput(filters)),
            ..Default::default()
        }))
    };

    // An array and `any_of` are the same any-of: one value that can match keeps
    // it alive, whether the schema permits it or a document merely uses it.
    search(serde_json::json!(["active", "actve"]))
        .await
        .unwrap();
    search(serde_json::json!({ "any_of": ["actve", "draft"] }))
        .await
        .unwrap();
    search(serde_json::json!(["actve", "legacy"]))
        .await
        .unwrap();

    // When none of its values could match it is refused, every one named.
    for filter in [
        serde_json::json!(["actve", "draf"]),
        serde_json::json!({ "any_of": ["actve", "draf"] }),
    ] {
        let err = search(filter).await.unwrap_err();
        assert!(
            err.message.contains(
                "filter 'status': 'actve', 'draf' are not allowed values; \
                 allowed: active, draft"
            ),
            "{}",
            err.message
        );
    }
    let err = search(serde_json::json!(["actve", "actve"]))
        .await
        .unwrap_err();
    assert!(
        err.message
            .contains("filter 'status': 'actve' is not an allowed value; allowed: active, draft"),
        "a repeated value is listed once: {}",
        err.message
    );

    // `all_of` needs one document to carry every value, so a single value that
    // nothing can match sinks it even beside a permitted one.
    let err = search(serde_json::json!({ "all_of": ["active", "actve"] }))
        .await
        .unwrap_err();
    assert!(
        err.message
            .contains("filter 'status': 'actve' is not an allowed value; allowed: active, draft"),
        "{}",
        err.message
    );
    search(serde_json::json!({ "all_of": ["active", "draft"] }))
        .await
        .unwrap();

    // A scalar is a one-value any-of.
    let err = search(serde_json::json!("actve")).await.unwrap_err();
    assert!(
        err.message
            .contains("filter 'status': 'actve' is not an allowed value; allowed: active, draft"),
        "{}",
        err.message
    );
}

#[tokio::test]
async fn search_filter_vocabulary_narrows_by_the_needle_search_matches_on() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(
        &tmp,
        "",
        "fields:\n  status:\n    type: enum\n    values: [active, draft]\n",
    );
    write_schema_file(
        &tmp,
        "food",
        "fields:\n  status:\n    type: enum\n    values: [cooking]\n",
    );
    let server = schema_tool_server(&tmp);
    seed_document(
        &server,
        "food/a.md",
        serde_json::json!({ "status": "cooking" }),
    )
    .await;
    seed_document(
        &server,
        "lifestyle/food/b.md",
        serde_json::json!({ "status": "active" }),
    )
    .await;

    let search = |path_prefix: &str| {
        let mut filters = serde_json::Map::new();
        filters.insert("status".into(), serde_json::json!("draft"));
        server.search(Parameters(SearchParams {
            filters: Some(SearchFiltersInput(filters)),
            path_prefix: Some(path_prefix.to_string()),
            ..Default::default()
        }))
    };

    // `food`, `food/` and `FOOD` are one needle: they name the top-level `food/`
    // scope, whose vocabulary does not permit `draft`.
    for needle in ["food", "food/", "FOOD"] {
        let err = search(needle).await.unwrap_err();
        assert!(
            err.message.contains("allowed: cooking"),
            "{needle:?}: {}",
            err.message
        );
    }
    // `/food` is a fragment of `lifestyle/food/b.md` only — no prefix of the
    // top-level `food/a.md` — so it names no scope: every scope governs, and the
    // root permits `draft`.
    let ok = search("/food").await.unwrap();
    assert_eq!(
        ok.structured_content.unwrap()["total"],
        serde_json::json!(0)
    );
}

#[tokio::test]
async fn search_enumerate_reports_total_and_truncation() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);
    for i in 0..5 {
        seed_document(
            &server,
            &format!("notes/{i}.md"),
            serde_json::json!({ "title": format!("Doc {i}"), "tags": ["note"] }),
        )
        .await;
    }

    let result = server
        .search(Parameters(SearchParams {
            limit: Some(2),
            ..Default::default()
        }))
        .await
        .unwrap();

    let structured = result.structured_content.unwrap();
    assert_eq!(structured["total"], serde_json::json!(5));
    assert_eq!(structured["returned"], serde_json::json!(2));
    assert_eq!(
        structured["has_more"],
        serde_json::json!(true),
        "truncation must never be silent"
    );
}

/// An enumeration row carries only what is not already on it or derivable:
/// no `mtime` unless the listing is ordered by it, no `offset` echo, no
/// `has_more: false`, and no promoted or derived keys in `frontmatter`.
#[tokio::test]
async fn search_enumerate_rows_omit_promoted_derived_and_default_fields() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(
        &tmp,
        "",
        "derived:\n  domain: {from: path_segment, index: 0}\n",
    );
    let server = schema_tool_server(&tmp);
    seed_document(
        &server,
        "notes/a.md",
        serde_json::json!({
            "title": "A", "description": "About A", "domain": "notes", "tags": ["x"],
        }),
    )
    .await;

    let result = server
        .search(Parameters(SearchParams::default()))
        .await
        .unwrap();
    let structured = single_representation(&result);
    assert_eq!(
        structured,
        serde_json::json!({
            "total": 1,
            "returned": 1,
            "documents": [{
                "file_path": "notes/a.md",
                "title": "A",
                "description": "About A",
                "frontmatter": {"tags": ["x"]},
            }],
        })
    );

    let by_mtime = server
        .search(Parameters(SearchParams {
            order_by: Some("mtime".into()),
            ..Default::default()
        }))
        .await
        .unwrap();
    assert_eq!(
        single_representation(&by_mtime)["documents"][0]["mtime"],
        serde_json::json!(100)
    );
}

#[tokio::test]
async fn search_enumerate_filters_through_the_tool_surface() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);
    seed_document(
        &server,
        "food/chili.md",
        serde_json::json!({ "title": "Chili", "tags": ["recipe"], "prep": 20 }),
    )
    .await;
    seed_document(
        &server,
        "food/stew.md",
        serde_json::json!({ "title": "Stew", "tags": ["recipe"], "prep": 90 }),
    )
    .await;

    let mut filters = serde_json::Map::new();
    filters.insert("prep".into(), serde_json::json!({ "lt": 30 }));
    let result = server
        .search(Parameters(SearchParams {
            filters: Some(SearchFiltersInput(filters)),
            ..Default::default()
        }))
        .await
        .unwrap();

    let structured = result.structured_content.unwrap();
    assert_eq!(structured["total"], serde_json::json!(1));
    assert_eq!(
        structured["documents"][0]["file_path"],
        serde_json::json!("food/chili.md")
    );
}

/// #182: the discovery case the issue was filed for, through the tool surface —
/// a fragment the caller does not know the start of.
#[tokio::test]
async fn search_enumerate_path_prefix_matches_a_mid_path_fragment() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);
    seed_document(
        &server,
        "kitchen/recipes/stir_fry.md",
        serde_json::json!({ "title": "Stir Fry" }),
    )
    .await;
    seed_document(
        &server,
        "sysadmin/zfs.md",
        serde_json::json!({ "title": "ZFS" }),
    )
    .await;

    // Neither the leading path component nor the whole basename — exactly what
    // a prefix match could not find.
    let result = server
        .search(Parameters(SearchParams {
            path_prefix: Some("stir_fr".to_string()),
            ..Default::default()
        }))
        .await
        .unwrap();

    let structured = result.structured_content.unwrap();
    assert_eq!(structured["total"], serde_json::json!(1));
    assert_eq!(
        structured["documents"][0]["file_path"],
        serde_json::json!("kitchen/recipes/stir_fry.md")
    );
}

/// Query mode cannot be driven end-to-end here (it needs a live Qdrant), so
/// this covers the part that is #182's actual change to it: the needle is
/// resolved against the metadata index — the same matcher enumeration uses —
/// before any vector search happens.
#[tokio::test]
async fn resolve_path_filter_resolves_a_fragment_to_its_documents() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);
    seed_document(
        &server,
        "kitchen/recipes/stir_fry.md",
        serde_json::json!({ "title": "Stir Fry" }),
    )
    .await;
    seed_document(
        &server,
        "sysadmin/zfs.md",
        serde_json::json!({ "title": "ZFS" }),
    )
    .await;

    let filter = server
        .resolve_path_filter(Some("stir_fr"))
        .await
        .unwrap()
        .expect("a needle resolves to a filter");

    assert_eq!(filter.paths, vec!["kitchen/recipes/stir_fry.md"]);
    assert!(!filter.truncated);
}

#[tokio::test]
async fn resolve_path_filter_is_none_without_a_needle() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);

    assert!(server.resolve_path_filter(None).await.unwrap().is_none());
    // Empty once normalized — "match everything" is the absence of a filter,
    // not a filter on the empty string (which as a substring matches all).
    assert!(
        server
            .resolve_path_filter(Some(""))
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        server
            .resolve_path_filter(Some("/"))
            .await
            .unwrap()
            .is_none()
    );
}

/// A needle matching nothing must resolve to an empty filter, not to `None` —
/// `None` means "no filter", which would return the whole corpus.
#[tokio::test]
async fn resolve_path_filter_empty_for_a_needle_that_matches_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);
    seed_document(
        &server,
        "sysadmin/zfs.md",
        serde_json::json!({ "title": "ZFS" }),
    )
    .await;

    let filter = server
        .resolve_path_filter(Some("nothing-matches-this"))
        .await
        .unwrap()
        .expect("a needle that matches nothing is still a filter");

    assert!(filter.paths.is_empty());
    assert!(!filter.truncated);
}

#[tokio::test]
async fn search_enumerate_reports_an_empty_result_clearly() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);

    let result = server
        .search(Parameters(SearchParams::default()))
        .await
        .unwrap();

    assert_eq!(
        result.structured_content.unwrap()["total"],
        serde_json::json!(0)
    );
}

/// Enumeration mode's `path_prefix` compiles to an exact SQL `LIKE '%needle%'`
/// (see `state::query_documents`) — never the query-mode post-fetch retain
/// the over-fetch/`path_prefix_truncated` fix exists for. A selective needle
/// must still report an exact `total`/`returned`/`has_more`, and the response
/// must carry no `path_prefix_truncated` key at all: that concept only exists
/// where a fetch can come up short of what actually matches.
#[tokio::test]
async fn search_enumerate_path_prefix_is_unaffected_by_query_mode_overfetch() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);
    for i in 0..5 {
        seed_document(
            &server,
            &format!("keep/{i}.md"),
            serde_json::json!({ "title": format!("Doc {i}") }),
        )
        .await;
    }
    for i in 0..20 {
        seed_document(
            &server,
            &format!("skip/{i}.md"),
            serde_json::json!({ "title": format!("Other {i}") }),
        )
        .await;
    }

    let result = server
        .search(Parameters(SearchParams {
            path_prefix: Some("keep/".to_string()),
            limit: Some(2),
            ..Default::default()
        }))
        .await
        .unwrap();

    let structured = result.structured_content.unwrap();
    assert_eq!(
        structured["total"],
        serde_json::json!(5),
        "an exact SQL LIKE prefix match, unaffected by the query-mode over-fetch \
         even though far more non-matching documents exist elsewhere"
    );
    assert_eq!(structured["returned"], serde_json::json!(2));
    assert_eq!(structured["has_more"], serde_json::json!(true));
    assert!(
        structured.get("path_prefix_truncated").is_none(),
        "path_prefix_truncated is a query-mode-only concept"
    );
}

#[test]
fn compound_filters_are_rejected_rather_than_silently_narrowed() {
    // Honoring only one half of a mixed filter returns a broader set than asked for.
    let err = build_document_query(&filters_from(
        serde_json::json!({ "tags": { "all_of": ["a"], "gte": 5 } }),
    ))
    .unwrap_err();
    assert!(format!("{:?}", err).contains("cannot combine"));

    let err = build_document_query(&filters_from(
        serde_json::json!({ "tags": { "any_of": ["a"], "all_of": ["b"] } }),
    ))
    .unwrap_err();
    assert!(format!("{:?}", err).contains("not both"));
}

#[test]
fn oversized_filter_value_lists_are_rejected() {
    let many: Vec<serde_json::Value> = (0..MAX_FILTER_VALUES + 1)
        .map(|i| serde_json::json!(format!("v{i}")))
        .collect();
    let err = build_document_query(&filters_from(
        serde_json::json!({ "tags": serde_json::Value::Array(many) }),
    ))
    .unwrap_err();
    assert!(format!("{:?}", err).contains("too many values"));
}

#[test]
fn oversized_filter_scalar_value_is_rejected() {
    // Master enforced MAX_FILTER_STR_LEN on each scalar filter value
    // (domain_too_long_is_rejected et al., since folded into the generic
    // `filters` map). `parse_filters`/`parse_field_filter` cap the field NAME
    // length and the values COUNT, but a single value's length must be capped
    // too, or one call can smuggle an arbitrarily large string into a Qdrant
    // query / SQLite bound parameter.
    let long = "x".repeat(MAX_FILTER_STR_LEN + 1);
    let err = build_document_query(&filters_from(serde_json::json!({ "tags": long.clone() })))
        .unwrap_err();
    assert!(
        format!("{:?}", err).contains("too long"),
        "scalar-equality value over MAX_FILTER_STR_LEN must be rejected: {:?}",
        err
    );
}

#[test]
fn oversized_filter_any_of_value_is_rejected() {
    let long = "x".repeat(MAX_FILTER_STR_LEN + 1);
    let err = build_document_query(&filters_from(
        serde_json::json!({ "tags": { "any_of": [long] } }),
    ))
    .unwrap_err();
    assert!(
        format!("{:?}", err).contains("too long"),
        "an any_of value over MAX_FILTER_STR_LEN must be rejected: {:?}",
        err
    );
}

#[test]
fn oversized_filter_all_of_value_is_rejected() {
    let long = "x".repeat(MAX_FILTER_STR_LEN + 1);
    let err = build_document_query(&filters_from(
        serde_json::json!({ "tags": { "all_of": [long] } }),
    ))
    .unwrap_err();
    assert!(
        format!("{:?}", err).contains("too long"),
        "an all_of value over MAX_FILTER_STR_LEN must be rejected: {:?}",
        err
    );
}

fn make_params(query: &str) -> SearchParams {
    SearchParams {
        query: Some(query.to_string()),
        ..Default::default()
    }
}

#[test]
fn valid_params_accepted() {
    let params = make_params("find documents about authentication");
    assert!(validate_search_params(&params).is_ok());
}

#[test]
fn query_at_limit_is_accepted() {
    let params = make_params(&"a".repeat(MAX_QUERY_LEN));
    assert!(validate_search_params(&params).is_ok());
}

#[test]
fn query_too_long_is_rejected() {
    let params = make_params(&"a".repeat(MAX_QUERY_LEN + 1));
    assert!(validate_search_params(&params).is_err());
}

#[test]
fn no_query_is_accepted() {
    // Enumeration mode: `query` is entirely optional now.
    let params = SearchParams::default();
    assert!(validate_search_params(&params).is_ok());
}

#[test]
fn canonicalize_nonexistent_file_produces_not_found_message() {
    let bad_path = std::path::PathBuf::from("/tmp/nonexistent-kb-test-dir/missing.md");
    let err = bad_path
        .canonicalize()
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => format!("File not found: {}", bad_path.display()),
            std::io::ErrorKind::PermissionDenied => {
                format!("Permission denied: {}", bad_path.display())
            }
            _ => format!("Cannot access file '{}': {}", bad_path.display(), e),
        })
        .unwrap_err();
    assert!(
        err.contains("File not found"),
        "expected 'File not found', got: {err}"
    );
}
