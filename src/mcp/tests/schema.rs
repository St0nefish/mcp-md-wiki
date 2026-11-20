use rmcp::handler::server::wrapper::Parameters;

use super::get_document::single_representation;
use crate::config::Granularity;
use crate::mcp::schema_tools::*;
use crate::mcp::test_support::*;
use crate::mcp::write_document::*;
use crate::mcp::*;
use crate::write::{DirectoryMoveError, DirectoryMoveSuccess, WriteError};
#[test]
fn scope_paths_normalize_to_relative_dirs() {
    assert_eq!(normalize_scope_path("").unwrap(), std::path::PathBuf::new());
    assert_eq!(
        normalize_scope_path("/food/recipes/").unwrap(),
        std::path::PathBuf::from("food/recipes")
    );
    assert_eq!(
        normalize_scope_path("./food/recipes").unwrap(),
        std::path::PathBuf::from("food/recipes")
    );
}

#[test]
fn scope_paths_reject_traversal() {
    // A schema written outside the KB governs nothing and could clobber other files.
    assert!(normalize_scope_path("../../etc").is_err());
    assert!(normalize_scope_path("food/../../etc").is_err());
}

#[test]
fn scope_paths_reject_overlong_input() {
    assert!(normalize_scope_path(&"a".repeat(MAX_PATH_LEN + 1)).is_err());
}

#[test]
fn scope_paths_drop_dot_segments_and_doubled_separators() {
    // `PathBuf` equality compares components, so it would hide a leftover `.`;
    // the text is what reaches a scope label, a commit message and a `LIKE`.
    for raw in [
        "food/./recipes",
        "food//recipes",
        "./food/./recipes/.",
        "/./food//recipes//",
        " food/recipes ",
    ] {
        let normalized = normalize_scope_path(raw).unwrap();
        assert_eq!(normalized.to_str(), Some("food/recipes"), "{raw:?}");
    }
}

#[test]
fn a_lone_dot_scope_path_is_the_root() {
    for raw in [".", "./", "/.", " . ", "././"] {
        let normalized = normalize_scope_path(raw)
            .unwrap_or_else(|e| panic!("{raw:?} must name the root, got: {}", e.message));
        assert_eq!(normalized.to_str(), Some(""), "{raw:?}");
    }
    // `..` is still refused wherever it sits, with the traversal message.
    for raw in ["..", "./..", "food/./../x"] {
        let err = normalize_scope_path(raw).unwrap_err();
        assert!(err.message.contains("'..'"), "{raw:?}: {}", err.message);
    }
}

fn update_params(operation: &str, field: &str) -> UpdateSchemaParams {
    UpdateSchemaParams {
        ops: None,
        path: None,
        operation: operation.into(),
        field: field.into(),
        values: None,
        definition: None,
        dry_run: None,
        force: None,
        acknowledge_root_change: None,
    }
}

/// Build a `set_field` definition the way a real MCP client's JSON arrives: through
/// `FieldDefinitionInput`'s own `Deserialize` impl, not by constructing
/// `RawFieldDef` directly. Fixtures here are expected to be valid; use
/// `serde_json::from_value::<FieldDefinitionInput>` directly in tests that assert on
/// a parse failure.
fn definition(json: serde_json::Value) -> FieldDefinitionInput {
    serde_json::from_value(json).expect("test fixture must be a valid definition")
}

#[test]
fn add_values_requires_a_non_empty_list() {
    let mut params = update_params("add_values", "tags");
    assert!(build_schema_edit(&params).is_err());

    params.values = Some(vec![]);
    assert!(build_schema_edit(&params).is_err());

    params.values = Some(vec!["recipe".into()]);
    assert!(build_schema_edit(&params).is_ok());
}

#[test]
fn unknown_operation_lists_the_valid_ones() {
    let err = build_schema_edit(&update_params("delete_everything", "tags")).unwrap_err();
    let msg = format!("{:?}", err);
    assert!(msg.contains("unknown operation"));
    assert!(msg.contains("add_values"));
}

#[test]
fn update_schema_definition_advertises_as_a_typed_object() {
    // This is the regression test for the actual bug: `definition` used to be typed
    // `serde_json::Value`, which schemars turns into an unconstrained `{}` schema —
    // no `type` keyword, no listed properties, nothing telling a client this must be
    // an object. A client with no other signal is then free to encode the value
    // however it likes, including as a JSON-encoded string, which is exactly what
    // happened in practice (see `FieldDefinitionInput`'s doc comment).
    let schema = schemars::schema_for!(UpdateSchemaParams);
    let root = schema.as_value();

    let definition_schema = &root["properties"]["definition"];
    // `Option<FieldDefinitionInput>` becomes `anyOf: [<real schema>, {"type": "null"}]`;
    // find the non-null branch.
    let object_schema = definition_schema["anyOf"]
        .as_array()
        .expect("definition must offer a typed alternative, not a bare {}")
        .iter()
        .find(|branch| branch["type"] != serde_json::json!("null"))
        .expect("definition must have a non-null branch");

    // schemars refs the RawFieldDef schema into `$defs` rather than inlining it;
    // resolve it so the assertions below see the real shape.
    let resolved = match object_schema["$ref"].as_str() {
        Some(reference) => &root["$defs"][reference.rsplit('/').next().unwrap()],
        None => object_schema,
    };

    assert_eq!(
        resolved["type"],
        serde_json::json!("object"),
        "definition must advertise as an object, got: {resolved}"
    );
    assert_eq!(
        resolved["additionalProperties"],
        serde_json::json!(false),
        "an unknown key must be rejected by a conforming client's own schema \
         validation too, not just our runtime check, got: {resolved}"
    );
    for key in ["type", "required", "indexed", "values", "default", "open"] {
        assert!(
            !resolved["properties"][key].is_null(),
            "definition schema is missing documented key '{key}': {resolved}"
        );
    }
}

#[test]
fn update_schema_definition_keeps_all_named_properties_after_slimming_descriptions() {
    // Regression test for the `#[schemars(description = "...")]` overrides added to
    // `RawFieldDef` (see that struct in schema.rs): they replace the huge doc-comment
    // text schemars would otherwise copy into every field's `description`, but must
    // not touch the shape of the schema itself. This asserts the full set of named,
    // typed properties the delegation exists to guarantee — including the recursive
    // `fields` map, which is what would silently degrade to an empty object if the
    // override attributes were misapplied (e.g. put on the wrong item, or swallowing
    // the field itself via a typo) rather than just shortening a description.
    let schema = schemars::schema_for!(UpdateSchemaParams);
    let root = schema.as_value();

    let definition_schema = &root["properties"]["definition"];
    let object_schema = definition_schema["anyOf"]
        .as_array()
        .expect("definition must offer a typed alternative, not a bare {}")
        .iter()
        .find(|branch| branch["type"] != serde_json::json!("null"))
        .expect("definition must have a non-null branch");
    let resolved = match object_schema["$ref"].as_str() {
        Some(reference) => &root["$defs"][reference.rsplit('/').next().unwrap()],
        None => object_schema,
    };

    assert_eq!(resolved["type"], serde_json::json!("object"));
    for key in [
        "type",
        "required",
        "indexed",
        "values",
        "default",
        "open",
        "fields",
        "items",
        "exclude_values",
        "removed",
    ] {
        let prop = &resolved["properties"][key];
        assert!(
            !prop.is_null(),
            "definition schema is missing documented key '{key}': {resolved}"
        );
        let desc = prop["description"]
            .as_str()
            .unwrap_or_else(|| panic!("property '{key}' lost its description: {prop}"));
        assert!(
            desc.len() <= 120,
            "property '{key}' description should be a short override, not the full \
             doc comment ({} chars): {desc:?}",
            desc.len()
        );
    }

    // `fields` is a map keyed by field name, whose values recurse back into the same
    // definition — confirm that recursion still resolves to the real object schema
    // (via `$ref`) rather than being erased or inlined without bound.
    let fields_prop = &resolved["properties"]["fields"];
    let nested_ref = fields_prop["additionalProperties"]["$ref"]
        .as_str()
        .expect("recursive 'fields' map must $ref back into the definition schema");
    let nested = &root["$defs"][nested_ref.rsplit('/').next().unwrap()];
    assert_eq!(
        nested["type"],
        serde_json::json!("object"),
        "recursive fields entry must resolve to the real object schema: {nested}"
    );
    assert!(
        !nested["properties"]["values"].is_null(),
        "recursive fields entry must keep its own named properties: {nested}"
    );
}

#[test]
fn set_field_parses_a_definition() {
    let mut params = update_params("set_field", "planning.prep_minutes");
    params.definition = Some(definition(
        serde_json::json!({ "type": "integer", "indexed": true }),
    ));

    match build_schema_edit(&params).unwrap() {
        crate::schema::SchemaEdit::SetField { field, definition } => {
            assert_eq!(field, "planning.prep_minutes");
            assert_eq!(definition.ty, Some(crate::schema::FieldType::Integer));
            assert_eq!(definition.indexed, Some(true));
        }
        other => panic!("expected SetField, got {other:?}"),
    }
}

#[test]
fn set_field_accepts_a_json_encoded_string_as_a_fallback() {
    // At least one real MCP client sends nested-object tool arguments as a
    // JSON-encoded string rather than an object, regardless of what the tool
    // schema advertises. `FieldDefinitionInput` tolerates that as a fallback.
    let mut params = update_params("set_field", "planning.prep_minutes");
    params.definition = Some(definition(serde_json::Value::String(
        r#"{"type":"integer","indexed":true}"#.to_string(),
    )));

    match build_schema_edit(&params).unwrap() {
        crate::schema::SchemaEdit::SetField { field, definition } => {
            assert_eq!(field, "planning.prep_minutes");
            assert_eq!(definition.ty, Some(crate::schema::FieldType::Integer));
            assert_eq!(definition.indexed, Some(true));
        }
        other => panic!("expected SetField, got {other:?}"),
    }
}

#[test]
fn update_schema_rejects_a_field_path_that_nests_too_deeply() {
    let deep = vec!["a"; MAX_SCHEMA_PATH_SEGMENTS + 1].join(".");
    let err = build_schema_edit(&update_params("remove_field", &deep)).unwrap_err();
    assert!(err.message.contains("too deep"), "got: {}", err.message);
    let ok = vec!["a"; MAX_SCHEMA_PATH_SEGMENTS].join(".");
    assert!(build_schema_edit(&update_params("remove_field", &ok)).is_ok());
}

#[test]
fn update_schema_rejects_a_field_path_with_an_empty_segment() {
    for bad in ["", ".a", "a.", "a..b", "a. .b", "a. b", "a .b"] {
        let mut params = update_params("remove_field", bad);
        params.values = Some(vec!["x".into()]);
        let err = build_schema_edit(&params).unwrap_err();
        assert!(
            err.message.contains("non-empty"),
            "{bad:?} should be rejected, got: {}",
            err.message
        );
    }
    assert!(build_schema_edit(&update_params("remove_field", "planning.method")).is_ok());
}

#[test]
fn update_schema_field_paths_follow_the_list_item_grammar() {
    // `[]` steps count toward the depth cap like names do (#346).
    for ok in ["npcs[]", "npcs[].name", "a.b[][].c"] {
        assert!(
            build_schema_edit(&update_params("remove_field", ok)).is_ok(),
            "{ok}"
        );
    }
    for bad in ["[]", "a.[]", "a[]. b", "a[].", "a[]..b"] {
        let err = build_schema_edit(&update_params("remove_field", bad)).unwrap_err();
        assert!(err.message.contains("non-empty"), "{bad}: {}", err.message);
    }
    let deep = format!("{}[]", vec!["a"; MAX_SCHEMA_PATH_SEGMENTS].join("."));
    let err = build_schema_edit(&update_params("remove_field", &deep)).unwrap_err();
    assert!(err.message.contains("too deep"), "got: {}", err.message);
}

#[test]
fn field_def_json_reports_items_only_when_declared() {
    let resolved = crate::schema::ResolvedSchema::default().merged_with_for_test(
        &crate::yaml::from_str(
            "fields:\n  npcs:\n    required: true\n    items:\n      type: object\n      \
             fields:\n        name:\n          type: text\n          required: true\n  \
             title:\n    type: text\n",
        )
        .unwrap(),
        "/",
    );
    assert_eq!(
        field_def_json(&resolved.fields["npcs"], Some("/")),
        serde_json::json!({
            "type": "list",
            "required": true,
            "items": {
                "type": "object",
                "fields": {"name": {"type": "text", "required": true}},
            },
            "declared_in": "/",
        })
    );
    assert_eq!(
        field_def_json(&resolved.fields["title"], Some("/")),
        serde_json::json!({"type": "text", "declared_in": "/"})
    );
    // Every type is named as `FieldType::name` spells it.
    for ty in crate::schema::FieldType::ALL {
        let def = crate::schema::FieldDef {
            ty: Some(ty),
            ..Default::default()
        };
        assert_eq!(
            field_def_json(&def, None)["type"],
            serde_json::json!(ty.name())
        );
    }
}

#[test]
fn set_field_rejects_a_string_that_is_not_valid_json() {
    let err = serde_json::from_value::<FieldDefinitionInput>(serde_json::Value::String(
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
fn set_field_rejects_a_json_array_naming_the_expected_shape() {
    let err =
        serde_json::from_value::<FieldDefinitionInput>(serde_json::json!(["type", "integer"]))
            .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("JSON object"),
        "expected the error to name the expected shape, got: {msg}"
    );
    assert!(
        !msg.contains("RawFieldDef"),
        "a Rust type name is meaningless to an MCP client, got: {msg}"
    );
}

#[test]
fn set_field_shape_errors_name_exactly_the_advertised_definition_keys() {
    // The keys a malformed definition's error lists are the properties the
    // `definition` schema advertises: `fields` is one, the deprecated `extend`
    // (accepted, never advertised) is not.
    let schema = schemars::schema_for!(crate::schema::RawFieldDef);
    let root = schema.as_value();
    let resolved = match root["$ref"].as_str() {
        Some(reference) => &root["$defs"][reference.rsplit('/').next().unwrap()],
        None => root,
    };
    let mut advertised: Vec<&str> = resolved["properties"]
        .as_object()
        .expect("the definition schema advertises named properties")
        .keys()
        .map(String::as_str)
        .collect();
    advertised.sort_unstable();
    let mut listed: Vec<&str> = FIELD_DEFINITION_KEYS
        .split('`')
        .skip(1)
        .step_by(2)
        .collect();
    listed.sort_unstable();
    assert_eq!(listed, advertised);

    for bad in [serde_json::json!(["type"]), serde_json::json!("[1]")] {
        let msg = serde_json::from_value::<FieldDefinitionInput>(bad)
            .unwrap_err()
            .to_string();
        assert!(msg.contains(FIELD_DEFINITION_KEYS), "{msg}");
    }
}

#[test]
fn set_field_rejects_an_unknown_key() {
    let err =
        serde_json::from_value::<FieldDefinitionInput>(serde_json::json!({ "typ": "integer" }))
            .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("typ"),
        "a typo'd key must be named in the error, not just silently dropped: {msg}"
    );
    assert!(
        !msg.contains("RawFieldDef"),
        "a Rust type name is meaningless to an MCP client, got: {msg}"
    );
}

#[test]
fn update_schema_params_reject_a_misspelled_definition_key() {
    // The same check, but through the exact path a real tool call takes: the whole
    // `UpdateSchemaParams` deserialized from one JSON blob, the way rmcp's
    // `Parameters<T>` extractor does it.
    let err = serde_json::from_value::<UpdateSchemaParams>(serde_json::json!({
        "operation": "set_field",
        "field": "tags",
        "definition": { "typ": "integer" },
    }))
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("typ"), "got: {msg}");
    assert!(!msg.contains("RawFieldDef"), "got: {msg}");
}

#[test]
fn schema_edits_round_trip_through_yaml() {
    // The property that matters: whatever update_schema writes must parse back,
    // because an unparseable schema fails the whole schema build.
    let mut file = crate::schema::SchemaFile::default();
    file.apply(&crate::schema::SchemaEdit::AddValues {
        field: "tags".into(),
        values: vec!["recipe".into(), "dinner".into()],
    })
    .unwrap();
    file.apply(&crate::schema::SchemaEdit::SetField {
        field: "planning.prep_minutes".into(),
        definition: Box::new(
            serde_json::from_value(serde_json::json!({ "type": "integer", "indexed": true }))
                .unwrap(),
        ),
    })
    .unwrap();

    let yaml = file.to_yaml().unwrap();
    let reparsed: crate::schema::SchemaFile = crate::yaml::from_str(&yaml).unwrap();

    assert_eq!(
        reparsed.fields["tags"].values,
        Some(vec!["dinner".to_string(), "recipe".to_string()]),
        "values are sorted for a stable diff"
    );
    assert_eq!(
        reparsed.fields["planning"].fields.as_ref().unwrap()["prep_minutes"].ty,
        Some(crate::schema::FieldType::Integer),
        "a dotted path nests under a created container, never a dotted top-level key"
    );
}

#[test]
fn adding_an_existing_value_is_reported_as_a_no_op() {
    let mut file = crate::schema::SchemaFile::default();
    let edit = crate::schema::SchemaEdit::AddValues {
        field: "tags".into(),
        values: vec!["recipe".into()],
    };
    file.apply(&edit).unwrap();
    let summary = file.apply(&edit).unwrap();

    assert!(summary.contains("already permitted"), "got: {summary}");
    assert_eq!(file.fields["tags"].values, Some(vec!["recipe".to_string()]));
}

#[test]
fn removing_an_undeclared_field_is_an_error() {
    let mut file = crate::schema::SchemaFile::default();
    assert!(
        file.apply(&crate::schema::SchemaEdit::RemoveField {
            field: "nope".into()
        })
        .is_err()
    );
}

#[test]
fn casualty_rendering_summarizes_beyond_the_cap() {
    let many: Vec<serde_json::Value> = (0..MAX_REPORTED_CASUALTIES + 5)
        .map(|i| serde_json::json!({ "path": format!("d{i}.md"), "reason": "missing" }))
        .collect();
    let rendered = render_casualties(&many);

    assert!(rendered.contains("d0.md"));
    assert!(rendered.contains("and 5 more"));
}

// --- schema tool handlers (no Qdrant, no embeddings, no git required) ---

#[tokio::test]
async fn get_schema_reports_merged_fields_with_provenance() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(&tmp, "", "fields:\n  title:\n    required: true\n");
    write_schema_file(
        &tmp,
        "food/recipes",
        "fields:\n  prep:\n    type: integer\n    indexed: true\n",
    );
    let server = schema_tool_server(&tmp);

    let result = server
        .get_schema(Parameters(GetSchemaParams {
            path: Some("food/recipes".into()),
            ..Default::default()
        }))
        .await
        .unwrap();

    let structured = single_representation(&result);
    let fields = structured["fields"].as_object().unwrap();

    // Keyed by field name; only what constrains a document is listed.
    assert_eq!(
        fields["title"],
        serde_json::json!({"required": true, "declared_in": "/"}),
        "inherited from the root scope, with nothing false, null or default"
    );
    assert_eq!(
        fields["prep"],
        serde_json::json!({"type": "integer", "indexed": true, "declared_in": "food/recipes/"}),
        "declared in this scope"
    );
    assert!(
        structured.get("frozen").is_none() && structured.get("frozen_reason").is_none(),
        "there is no frozen state: an invalid schema never loads"
    );
    assert!(structured.get("omitted_fields").is_none(), "{structured}");
}

#[tokio::test]
async fn get_schema_reports_the_config_derived_root_as_slash() {
    // No root schema file: the root rules come from the server's config,
    // which is deployment detail — the caller sees the root scope, `/`.
    let tmp = tempfile::tempdir().unwrap();
    let mut config = (*make_test_resolved_config(tmp.path())).clone();
    config.frontmatter.required = vec!["legacy_field".into()];
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], Arc::new(config));
    let result = server
        .get_schema(Parameters(GetSchemaParams::default()))
        .await
        .unwrap();
    let structured = single_representation(&result);
    assert_eq!(structured["path"], "/");
    assert_eq!(structured["fields"]["legacy_field"]["declared_in"], "/");
    assert!(!structured.to_string().contains("config"), "{structured}");
}

#[tokio::test]
async fn get_schema_reports_dedup_override_only_when_set() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(&tmp, "", "fields:\n  title:\n    required: true\n");
    write_schema_file(
        &tmp,
        "food/plans",
        "dedup:\n  enabled: false\n  threshold: 0.95\n",
    );
    let server = schema_tool_server(&tmp);

    let plans = server
        .get_schema(Parameters(GetSchemaParams {
            path: Some("food/plans".into()),
            ..Default::default()
        }))
        .await
        .unwrap();
    let structured = single_representation(&plans);
    assert_eq!(structured["dedup"]["enabled"], serde_json::json!(false));
    assert_eq!(structured["dedup"]["threshold"].as_f64(), Some(0.95));

    let root = server
        .get_schema(Parameters(GetSchemaParams::default()))
        .await
        .unwrap();
    assert!(
        root.structured_content.unwrap().get("dedup").is_none(),
        "no override set, no dedup key"
    );
}

#[tokio::test]
async fn get_schema_root_ignores_config_frontmatter_once_a_root_file_exists() {
    // A root .kb-schema.yaml declares `title`. config.yaml ALSO declares a
    // required field (`legacy_field`) that the root file never mentions. Per issue
    // #91's policy, a root schema file is authoritative for the KB root: config's
    // `frontmatter` block must not still be contributing fields through
    // get_schema, and every field it does report must be attributed to the root
    // schema file, not "config.yaml".
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(&tmp, "", "fields:\n  title:\n    required: true\n");

    let mut config = (*make_test_resolved_config(tmp.path())).clone();
    config.frontmatter.required = vec!["legacy_field".into()];
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], Arc::new(config));

    let result = server
        .get_schema(Parameters(GetSchemaParams::default()))
        .await
        .unwrap();

    let structured = result.structured_content.unwrap();
    let fields = structured["fields"].as_object().unwrap();

    assert!(
        fields.contains_key("title"),
        "the root file's own field is reported"
    );
    assert!(
        !fields.contains_key("legacy_field"),
        "a config-only field must not leak into the root once a root schema file exists"
    );
    for field in fields.values() {
        assert_eq!(
            field["declared_in"],
            serde_json::json!("/"),
            "every reported root field must be attributed to the root schema, \
             not to config.yaml, once one exists"
        );
    }
}

#[tokio::test]
async fn schema_tools_accept_a_leading_slash_as_the_kb_root() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(
        &tmp,
        "food/recipes",
        "fields:\n  prep:\n    type: integer\n",
    );
    let server = schema_tool_server(&tmp);

    let with_slash = server
        .get_schema(Parameters(GetSchemaParams {
            path: Some("/food/recipes".into()),
            ..Default::default()
        }))
        .await
        .unwrap();
    let without = server
        .get_schema(Parameters(GetSchemaParams {
            path: Some("food/recipes".into()),
            ..Default::default()
        }))
        .await
        .unwrap();

    assert_eq!(
        with_slash.structured_content, without.structured_content,
        "callers cannot know where the KB lives, so `/x` and `x` must agree"
    );
}

#[tokio::test]
async fn get_schema_resolves_a_partial_directory() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(
        &tmp,
        "food/recipes",
        "fields:\n  prep:\n    type: integer\n",
    );
    let server = schema_tool_server(&tmp);

    let result = server
        .get_schema(Parameters(GetSchemaParams {
            path: Some("recipes".into()),
            ..Default::default()
        }))
        .await
        .expect("a unique trailing match resolves");

    let fields = result.structured_content.unwrap()["fields"]
        .as_object()
        .unwrap()
        .clone();
    assert!(
        fields.contains_key("prep"),
        "should have resolved to food/recipes, got: {fields:?}"
    );
}

#[tokio::test]
async fn an_ambiguous_partial_scope_reports_the_candidates() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(&tmp, "food/recipes", "fields:\n  a:\n    type: text\n");
    write_schema_file(&tmp, "archive/recipes", "fields:\n  b:\n    type: text\n");
    let server = schema_tool_server(&tmp);

    let err = server
        .get_schema(Parameters(GetSchemaParams {
            path: Some("recipes".into()),
            ..Default::default()
        }))
        .await
        .expect_err("two scopes end in recipes; guessing would be wrong");

    let msg = format!("{:?}", err);
    assert!(msg.contains("matches 2 scopes"), "got: {msg}");
    assert!(msg.contains("food/recipes"));
    assert!(msg.contains("archive/recipes"));
}

#[tokio::test]
async fn update_schema_can_still_create_a_scope_that_does_not_exist_yet() {
    // Needs a real git-backed harness (not `schema_tool_server`'s bare tempdir):
    // `write_raw_file` now rolls a failed `commit_and_sync` back (see
    // `write_raw_file`'s doc comment), so a harness where the git step can never
    // succeed would have this call fail and its rollback remove the very file
    // this test is checking for.
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let (server, _config) = make_git_backed_server(&work);

    server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("/brand/new".into()),
            operation: "add_values".into(),
            field: "tags".into(),
            values: Some(vec!["x".into()]),
            definition: None,
            dry_run: None,
            force: None,
            acknowledge_root_change: None,
        }))
        .await
        .expect("update_schema must succeed against this git-backed harness");

    assert!(
        work.path()
            .join("brand/new")
            .join(crate::schema::SCHEMA_FILE_NAME)
            .exists(),
        "an unmatched path must be taken literally so new scopes can be created"
    );
}

#[tokio::test]
async fn update_schema_refuses_to_write_a_file_over_the_size_cap() {
    // Flow style on disk (~12 bytes a value, under the cap) re-serializes in block
    // style (~17 bytes a value, over it), so one added value crosses the limit.
    let tmp = tempfile::tempdir().unwrap();
    let values: Vec<String> = (0..20_000).map(|i| format!("value{i:05}")).collect();
    let on_disk = format!("fields:\n  tags:\n    values: [{}]\n", values.join(", "));
    assert!((on_disk.len() as u64) < crate::schema::MAX_SCHEMA_FILE_BYTES);
    write_schema_file(&tmp, "big", &on_disk);
    let server = schema_tool_server(&tmp);

    let err = server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("big".into()),
            operation: "add_values".into(),
            field: "tags".into(),
            values: Some(vec!["new".into()]),
            definition: None,
            dry_run: None,
            force: None,
            acknowledge_root_change: None,
        }))
        .await
        .unwrap_err();

    assert!(err.message.contains("limit"), "got: {}", err.message);
    assert_no_schema_file_name(&err.message);
    assert_eq!(
        std::fs::read_to_string(tmp.path().join("big").join(crate::schema::SCHEMA_FILE_NAME))
            .unwrap(),
        on_disk,
        "nothing written"
    );
}

#[tokio::test]
async fn update_schema_on_a_file_made_invalid_on_disk_says_how_to_recover() {
    // The server built its cache while the file was valid; it then went bad on
    // disk (as a git push would leave it) and the runtime rebuild was refused.
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(&tmp, "notes", "fields:\n  tags:\n    values: [a]\n");
    let server = schema_tool_server(&tmp);
    write_schema_file(&tmp, "notes", "fields: [not a mapping\n");

    let err = server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("notes".into()),
            operation: "add_values".into(),
            field: "tags".into(),
            values: Some(vec!["b".into()]),
            definition: None,
            dry_run: None,
            force: None,
            acknowledge_root_change: None,
        }))
        .await
        .unwrap_err();

    assert!(err.message.contains("is invalid"), "got: {}", err.message);
    assert!(
        err.message.contains("last valid schema"),
        "got: {}",
        err.message
    );
    assert!(err.message.contains("'notes/'"), "got: {}", err.message);
    assert_no_schema_file_name(&err.message);
}

#[tokio::test]
async fn update_schema_refuses_a_directory_holding_both_schema_names() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(&tmp, "notes", "fields:\n  tags:\n    values: [a]\n");
    let server = schema_tool_server(&tmp);
    std::fs::write(
        tmp.path()
            .join("notes")
            .join(crate::schema::LEGACY_SCHEMA_FILE_NAME),
        "fields:\n  tags:\n    values: [z]\n",
    )
    .unwrap();

    let err = server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("notes".into()),
            operation: "add_values".into(),
            field: "tags".into(),
            values: Some(vec!["b".into()]),
            definition: None,
            dry_run: None,
            force: None,
            acknowledge_root_change: None,
        }))
        .await
        .unwrap_err();

    assert!(
        err.message.contains("two schema files"),
        "got: {}",
        err.message
    );
    assert_no_schema_file_name(&err.message);
    assert!(
        tmp.path()
            .join("notes")
            .join(crate::schema::LEGACY_SCHEMA_FILE_NAME)
            .exists(),
        "nothing is resolved by deleting either file"
    );
}

/// Model-facing text names a schema by its directory, never by file name.
fn assert_no_schema_file_name(text: &str) {
    for name in ["kb-schema", ".schema.yaml"] {
        assert!(!text.contains(name), "'{name}' leaked into: {text}");
    }
}

#[test]
fn no_model_facing_surface_names_the_schema_file() {
    // Every tool description and input schema, as `tools/list` serves them.
    let config = overlay_test_config(
        &[
            Granularity::Chunk,
            Granularity::Document,
            Granularity::Section,
        ],
        true,
    );
    let server = make_overlay_test_server_with_config(HashMap::new(), config);
    for tool in KbSearchServer::tool_router().list_all() {
        let tool = server.get_tool(tool.name.as_ref()).unwrap();
        assert_no_schema_file_name(tool.description.as_deref().unwrap_or(""));
        assert_no_schema_file_name(
            &serde_json::Value::Object((*tool.input_schema).clone()).to_string(),
        );
    }
    assert_no_schema_file_name(&crate::descriptions::compose_server_mechanics());

    // Refusals that echo a caller-supplied path add nothing beyond it.
    for path in ["notes/.schema.yaml", "notes/.kb-schema.yaml"] {
        let err = schema_file_path_error(path);
        assert_no_schema_file_name(&err.message.replace(path, ""));
        let text = batch_write_document_error_text(&WriteError::SchemaFile {
            rel_path: path.to_string(),
        });
        assert_no_schema_file_name(&text.replace(path, ""));
    }

    // A refused rebuild, as `update_schema` reports it.
    let refusal = crate::schema::SchemaBuildError {
        invalid: vec![
            crate::schema::InvalidSchemaFile {
                path: "a/.kb-schema.yaml".into(),
                reason: crate::schema::BOTH_NAMES_REASON.to_string(),
            },
            crate::schema::InvalidSchemaFile {
                path: ".schema.yaml".into(),
                reason: "fields: expected a map".to_string(),
            },
        ],
    };
    let shown = refusal.model_facing();
    assert!(shown.contains("a/: two schema files"), "{shown}");
    assert!(shown.contains("/: fields: expected a map"), "{shown}");
    assert_no_schema_file_name(&shown);
    // The operator-facing Display keeps the real paths.
    assert!(refusal.to_string().contains("a/.kb-schema.yaml"));

    // A move refusal over both names in one source directory.
    let err = move_directory_error_to_mcp_error(
        DirectoryMoveError::InvalidSchemaInSource {
            path: "src/.schema.yaml".to_string(),
            reason: crate::schema::BOTH_NAMES_REASON.to_string(),
        },
        "src",
        "dest",
    );
    assert_no_schema_file_name(&err.message);
    assert_no_schema_file_name(&err.data.unwrap().to_string());
}

#[tokio::test]
async fn get_schema_values_only_filters_to_vocabularies() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(
        &tmp,
        "",
        "fields:\n  title:\n    required: true\n  status:\n    type: enum\n    values: [active]\n",
    );
    let server = schema_tool_server(&tmp);

    let result = server
        .get_schema(Parameters(GetSchemaParams {
            values_only: Some(true),
            ..Default::default()
        }))
        .await
        .unwrap();

    let fields = result.structured_content.unwrap()["fields"]
        .as_object()
        .unwrap()
        .clone();
    let names: Vec<&str> = fields.keys().map(String::as_str).collect();
    assert_eq!(names, vec!["status"], "only closed-set fields are reported");
}

#[tokio::test]
async fn get_schema_values_in_use_lists_open_field_values_and_other_fields() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(
        &tmp,
        "",
        "derived:\n  domain: {from: path_segment, index: 0}\n\
         fields:\n  tags:\n    type: list\n  status:\n    type: enum\n    values: [active]\n",
    );
    let server = schema_tool_server(&tmp);
    seed_document(
        &server,
        "notes/a.md",
        serde_json::json!({ "tags": ["docker", "x"], "status": "active", "owner": "me" }),
    )
    .await;
    seed_document(
        &server,
        "notes/b.md",
        serde_json::json!({ "tags": ["docker"], "domain": "notes" }),
    )
    .await;
    seed_document(
        &server,
        "other/c.md",
        serde_json::json!({ "tags": ["elsewhere"] }),
    )
    .await;

    let plain = server
        .get_schema(Parameters(GetSchemaParams {
            path: Some("notes".into()),
            ..Default::default()
        }))
        .await
        .unwrap()
        .structured_content
        .unwrap();
    assert!(plain["fields"]["tags"].get("in_use").is_none(), "{plain}");
    assert!(plain.get("other_fields_in_use").is_none(), "{plain}");

    let with_values = server
        .get_schema(Parameters(GetSchemaParams {
            path: Some("notes".into()),
            values_in_use: Some(true),
            ..Default::default()
        }))
        .await
        .unwrap()
        .structured_content
        .unwrap();
    assert_eq!(
        with_values["fields"]["tags"]["in_use"],
        serde_json::json!({ "docker": 2, "x": 1 }),
        "counts are scoped to the path: {with_values}"
    );
    assert!(
        with_values["fields"]["status"].get("in_use").is_none(),
        "a closed field already lists its values: {with_values}"
    );
    assert_eq!(
        with_values["other_fields_in_use"],
        serde_json::json!(["owner"]),
        "undeclared fields in use, never a derived one: {with_values}"
    );
}

#[tokio::test]
async fn get_schema_other_fields_in_use_skips_fields_a_deeper_scope_declares() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(&tmp, "", "fields:\n  tags:\n    type: list\n");
    // Declared only below `food/`: a scope governs it, so from `food/` or the
    // root it is not an undeclared field.
    write_schema_file(
        &tmp,
        "food/recipes",
        "fields:\n  prep:\n    type: integer\n",
    );
    let server = schema_tool_server(&tmp);
    seed_document(
        &server,
        "food/recipes/a.md",
        serde_json::json!({ "prep": 5, "tags": ["x"] }),
    )
    .await;
    seed_document(&server, "food/b.md", serde_json::json!({ "owner": "me" })).await;

    for path in [None, Some("food")] {
        let result = server
            .get_schema(Parameters(GetSchemaParams {
                path: path.map(str::to_string),
                values_in_use: Some(true),
                ..Default::default()
            }))
            .await
            .unwrap()
            .structured_content
            .unwrap();
        assert_eq!(
            result["other_fields_in_use"],
            serde_json::json!(["owner"]),
            "{path:?}: {result}"
        );
    }
}

#[tokio::test]
async fn get_schema_never_lists_a_derived_field() {
    // A deployment that filters on a derived `domain` declares it `indexed`.
    // Ingest still derives it from the folder and a write that authors it is
    // refused, so it must not be offered as a field to fill in — anywhere below
    // the root that derives it (#345).
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(
        &tmp,
        "",
        "derived:\n  domain: {from: path_segment, index: 0}\n\
         fields:\n  domain:\n    indexed: true\n  status:\n    indexed: true\n",
    );
    write_schema_file(&tmp, "notes", "fields:\n  tags:\n    type: list\n");
    let server = schema_tool_server(&tmp);

    for path in [None, Some("notes".to_string())] {
        let all = server
            .get_schema(Parameters(GetSchemaParams {
                path: path.clone(),
                ..Default::default()
            }))
            .await
            .unwrap()
            .structured_content
            .unwrap();
        let names: Vec<&str> = all["fields"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert!(!names.contains(&"domain"), "{path:?}: {all}");
        assert!(names.contains(&"status"), "{path:?}: {all}");
        assert!(all.get("omitted_fields").is_none(), "{all}");
    }

    let named = server
        .get_schema(Parameters(GetSchemaParams {
            fields: Some(vec!["domain".into()]),
            ..Default::default()
        }))
        .await
        .unwrap()
        .structured_content
        .unwrap();
    assert!(named["fields"].as_object().unwrap().is_empty(), "{named}");
}

/// Without a `derived:` block (#345) `domain` is an ordinary field: listed
/// like any other declared one.
#[tokio::test]
async fn get_schema_lists_domain_when_nothing_derives_it() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = (*make_test_resolved_config(tmp.path())).clone();
    config.frontmatter.indexed_fields = vec!["domain".into(), "status".into()];
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], Arc::new(config));

    let all = server
        .get_schema(Parameters(GetSchemaParams::default()))
        .await
        .unwrap()
        .structured_content
        .unwrap();
    let names: Vec<&str> = all["fields"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(names, vec!["domain", "status"], "{all}");
}

#[tokio::test]
async fn get_schema_counts_values_under_a_directory_however_its_path_is_spelled() {
    let tmp = tempfile::tempdir().unwrap();
    // `food/recipes` has no schema file of its own, so its requested path is used
    // as given when counting what documents there use.
    write_schema_file(&tmp, "", "fields:\n  tags:\n    type: list\n");
    let server = schema_tool_server(&tmp);
    seed_document(
        &server,
        "food/recipes/a.md",
        serde_json::json!({ "tags": ["dinner"] }),
    )
    .await;

    let get = |path: &str| {
        server.get_schema(Parameters(GetSchemaParams {
            path: Some(path.to_string()),
            values_in_use: Some(true),
            ..Default::default()
        }))
    };
    let plain = get("food/recipes")
        .await
        .unwrap()
        .structured_content
        .unwrap();
    assert_eq!(
        plain["fields"]["tags"]["in_use"],
        serde_json::json!({ "dinner": 1 }),
        "{plain}"
    );
    for spelling in ["food/./recipes", "food//recipes", "./food/recipes/."] {
        let got = get(spelling).await.unwrap().structured_content.unwrap();
        assert_eq!(got, plain, "{spelling:?}");
    }

    let root = get(".").await.unwrap().structured_content.unwrap();
    assert_eq!(root["path"], "/", "a lone `.` is the root: {root}");
}

#[test]
fn schema_path_depth_and_projection_depth_are_one_limit() {
    assert_eq!(
        MAX_SCHEMA_PATH_SEGMENTS,
        crate::schema::MAX_FIELD_PATH_DEPTH
    );
    assert_eq!(MAX_SCHEMA_PATH_SEGMENTS, 16);
}

#[tokio::test]
async fn update_schema_dry_run_writes_nothing_and_reports_casualties() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);
    seed_document(&server, "notes/a.md", serde_json::json!({ "title": "A" })).await;

    let result = server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("notes".into()),
            operation: "set_field".into(),
            field: "status".into(),
            values: None,
            definition: Some(definition(serde_json::json!({ "required": true }))),
            dry_run: Some(true),
            force: None,
            acknowledge_root_change: None,
        }))
        .await
        .unwrap();

    let structured = single_representation(&result);
    assert_no_schema_file_name(&structured.to_string());
    assert_eq!(structured["dry_run"], serde_json::json!(true));
    assert_eq!(
        structured["would_invalidate"].as_array().unwrap().len(),
        1,
        "the seeded document has no status and would fail the new rule"
    );
    // The edited field as the scope would resolve it — not the file.
    assert!(structured.get("yaml").is_none(), "{structured}");
    assert_eq!(structured["field"], "status");
    assert_eq!(
        structured["definition"],
        serde_json::json!({"required": true, "declared_in": "notes/"})
    );
    assert!(
        !tmp.path()
            .join("notes")
            .join(crate::schema::SCHEMA_FILE_NAME)
            .exists(),
        "a dry run must not touch the filesystem"
    );
}

#[tokio::test]
async fn update_schema_names_a_scope_without_dot_segments_or_doubled_separators() {
    // The scope label (and so the commit message built from it) is the normalized
    // path, never the raw text.
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);
    let mut params = update_params("set_field", "status");
    params.path = Some("notes/./sub//deep".into());
    params.definition = Some(definition(serde_json::json!({ "required": true })));
    params.dry_run = Some(true);

    let result = server.update_schema(Parameters(params)).await.unwrap();

    let structured = single_representation(&result);
    assert_eq!(structured["path"], "notes/sub/deep/", "{structured}");
    assert_eq!(
        structured["definition"]["declared_in"], "notes/sub/deep/",
        "{structured}"
    );
}

#[tokio::test]
async fn update_schema_caps_structured_casualties_like_it_caps_the_text() {
    // #148: `documents_broken_by` deliberately returns every casualty — the
    // force/refuse decision needs completeness — but before this fix that full
    // list went straight into `structured_content` while the text half was
    // already capped at MAX_REPORTED_CASUALTIES via `render_casualties`. Seed
    // enough documents to blow past the cap and assert the structured half is
    // bounded too, with a total/truncated flag so a client reading only
    // `structured_content` can still tell it was cut off.
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);
    let seeded = MAX_REPORTED_CASUALTIES + 5;
    for i in 0..seeded {
        seed_document(
            &server,
            &format!("notes/doc{i}.md"),
            serde_json::json!({ "title": format!("Doc {i}") }),
        )
        .await;
    }

    let result = server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("notes".into()),
            operation: "set_field".into(),
            field: "status".into(),
            values: None,
            definition: Some(definition(serde_json::json!({ "required": true }))),
            dry_run: Some(true),
            force: None,
            acknowledge_root_change: None,
        }))
        .await
        .unwrap();

    let structured = result.structured_content.unwrap();
    assert_eq!(
        structured["would_invalidate"].as_array().unwrap().len(),
        MAX_REPORTED_CASUALTIES,
        "structured_content must cap the casualty list the same way the text \
         rendering does, not embed all {seeded} verbatim"
    );
    assert_eq!(
        structured["casualties_total"],
        serde_json::json!(seeded),
        "the true count must still be reported even though the list is capped"
    );
    assert_eq!(
        structured["casualties_truncated"],
        serde_json::json!(true),
        "truncation must never be silent"
    );
}

#[tokio::test]
async fn update_schema_reports_untruncated_casualties_below_the_cap() {
    // Companion to the truncation test above: when the casualty count is at or
    // under the cap, `casualties_truncated` must read false and
    // `casualties_total` must match the (uncapped) list length exactly, so a
    // client cannot mistake "small" for "truncated."
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);
    seed_document(&server, "notes/a.md", serde_json::json!({ "title": "A" })).await;

    let result = server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("notes".into()),
            operation: "set_field".into(),
            field: "status".into(),
            values: None,
            definition: Some(definition(serde_json::json!({ "required": true }))),
            dry_run: Some(true),
            force: None,
            acknowledge_root_change: None,
        }))
        .await
        .unwrap();

    // Below the cap the list is complete, so neither cap key is sent.
    let structured = result.structured_content.unwrap();
    assert_eq!(structured["would_invalidate"].as_array().unwrap().len(), 1);
    assert!(structured.get("casualties_total").is_none(), "{structured}");
    assert!(
        structured.get("casualties_truncated").is_none(),
        "{structured}"
    );
}

#[tokio::test]
async fn update_schema_refuses_a_breaking_change_without_force() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);
    seed_document(&server, "notes/a.md", serde_json::json!({ "title": "A" })).await;

    let err = server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("notes".into()),
            operation: "set_field".into(),
            field: "status".into(),
            values: None,
            definition: Some(definition(serde_json::json!({ "required": true }))),
            dry_run: None,
            force: None,
            acknowledge_root_change: None,
        }))
        .await
        .expect_err("must refuse rather than silently invalidate documents");

    let msg = format!("{:?}", err);
    assert!(msg.contains("Refusing to apply"), "got: {msg}");
    assert!(
        !tmp.path()
            .join("notes")
            .join(crate::schema::SCHEMA_FILE_NAME)
            .exists(),
        "a refused change must leave the filesystem untouched"
    );
}

#[tokio::test]
async fn update_schema_accepts_a_change_that_breaks_nothing() {
    // Git-backed harness — see the comment on
    // `update_schema_can_still_create_a_scope_that_does_not_exist_yet` for why
    // `schema_tool_server`'s bare tempdir no longer works for a write that must
    // actually land: `write_raw_file` now rolls back a failed `commit_and_sync`
    // instead of leaving the file behind.
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let (server, _config) = make_git_backed_server(&work);
    seed_document(
        &server,
        "notes/a.md",
        serde_json::json!({ "title": "A", "status": "active" }),
    )
    .await;

    server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("notes".into()),
            operation: "add_values".into(),
            field: "status".into(),
            values: Some(vec!["active".into(), "draft".into()]),
            definition: None,
            dry_run: None,
            force: None,
            acknowledge_root_change: None,
        }))
        .await
        .expect("a non-breaking change must succeed against this git-backed harness");

    let written = work
        .path()
        .join("notes")
        .join(crate::schema::SCHEMA_FILE_NAME);
    assert!(
        written.exists(),
        "a non-breaking change must be written and committed"
    );
    let yaml = std::fs::read_to_string(&written).unwrap();
    let reparsed: crate::schema::SchemaFile = crate::yaml::from_str(&yaml).unwrap();
    assert_eq!(
        reparsed.fields["status"].values,
        Some(vec!["active".to_string(), "draft".to_string()])
    );
}

#[tokio::test]
async fn update_schema_reports_lint_warnings_without_refusing() {
    // A default that fails its own type and a reserved payload name are lint
    // warnings (#347): reported in `warnings`, dry run and real write alike, and
    // the write still lands.
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let (server, _config) = make_git_backed_server(&work);
    let set = |field: &str, def: serde_json::Value, dry_run: bool| UpdateSchemaParams {
        ops: None,
        path: Some("notes".into()),
        definition: Some(definition(def)),
        dry_run: Some(dry_run),
        ..update_params("set_field", field)
    };

    let dry = server
        .update_schema(Parameters(set(
            "servings",
            serde_json::json!({ "type": "integer", "default": "four" }),
            true,
        )))
        .await
        .unwrap();
    let dry = single_representation(&dry);
    assert_eq!(
        dry["warnings"],
        serde_json::json!([
            "field 'servings': the default fails the field's own type (expected an \
             integer, got a string)"
        ])
    );

    let real = server
        .update_schema(Parameters(set(
            "servings",
            serde_json::json!({ "type": "integer", "default": "four" }),
            false,
        )))
        .await
        .expect("a lint warning never refuses a write");
    let real = single_representation(&real);
    assert_eq!(real["warnings"], dry["warnings"]);
    assert!(
        work.path()
            .join("notes")
            .join(crate::schema::SCHEMA_FILE_NAME)
            .exists()
    );

    // Warnings cover the whole file as written, not just the edited field.
    let next = server
        .update_schema(Parameters(set(
            "text",
            serde_json::json!({ "type": "text" }),
            false,
        )))
        .await
        .expect("a reserved name is a warning, not a refusal");
    let next = single_representation(&next);
    let warnings = next["warnings"].as_array().unwrap();
    assert_eq!(warnings.len(), 2, "{warnings:?}");
    assert!(
        warnings[0]
            .as_str()
            .unwrap()
            .starts_with("field 'servings'")
    );
    assert!(warnings[1].as_str().unwrap().contains("reserved"));
}

#[tokio::test]
async fn update_schema_omits_warnings_for_a_clean_schema() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);
    let result = server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("notes".into()),
            definition: Some(definition(
                serde_json::json!({ "type": "integer", "default": 4 }),
            )),
            dry_run: Some(true),
            ..update_params("set_field", "servings")
        }))
        .await
        .unwrap();
    assert!(single_representation(&result).get("warnings").is_none());
}

#[tokio::test]
async fn update_schema_casualty_check_validates_list_elements() {
    // The dry run's casualty pass shares the recursive validator (#347), so an
    // element rule names the element that would break.
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);
    seed_document(
        &server,
        "notes/a.md",
        serde_json::json!({ "title": "A", "npcs": [{ "name": "Bob" }, { "role": "foe" }] }),
    )
    .await;
    seed_document(
        &server,
        "notes/b.md",
        serde_json::json!({ "title": "B", "npcs": [{ "name": "Eve" }] }),
    )
    .await;

    let result = server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("notes".into()),
            definition: Some(definition(serde_json::json!({
                "type": "list",
                "items": { "fields": { "name": { "type": "text", "required": true } } },
            }))),
            dry_run: Some(true),
            ..update_params("set_field", "npcs")
        }))
        .await
        .unwrap();
    let structured = single_representation(&result);
    let casualties = structured["would_invalidate"].as_array().unwrap();
    assert_eq!(casualties.len(), 1, "{casualties:?}");
    assert!(
        casualties[0].to_string().contains("npcs[1].name"),
        "{casualties:?}"
    );
}

#[tokio::test]
async fn update_schema_dry_run_ignores_documents_a_deeper_scope_governs() {
    // A descendant scope that redefines the edited field shadows the parent, so its
    // documents are unaffected and must not be reported as casualties.
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(
        &tmp,
        "notes/archive",
        "fields:\n  status:\n    type: enum\n    values: [archived]\n",
    );
    let server = schema_tool_server(&tmp);
    seed_document(
        &server,
        "notes/archive/old.md",
        serde_json::json!({ "title": "Old", "status": "archived" }),
    )
    .await;

    let result = server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("notes".into()),
            operation: "set_field".into(),
            field: "status".into(),
            values: Some(vec!["active".into()]),
            definition: Some(definition(
                serde_json::json!({ "type": "enum", "values": ["active"], "required": true }),
            )),
            dry_run: Some(true),
            force: None,
            acknowledge_root_change: None,
        }))
        .await
        .unwrap();

    let structured = result.structured_content.unwrap();
    assert!(
        structured.get("would_invalidate").is_none(),
        "notes/archive/ has its own status rule and is unaffected, got: {structured}"
    );
}

#[tokio::test]
async fn update_schema_force_applies_despite_casualties() {
    // Git-backed harness — see the comment on
    // `update_schema_can_still_create_a_scope_that_does_not_exist_yet`: this test
    // asserts the write survives, so the git step must actually succeed rather
    // than trigger `write_raw_file`'s rollback.
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let (server, _config) = make_git_backed_server(&work);
    seed_document(&server, "notes/a.md", serde_json::json!({ "title": "A" })).await;

    // The point of this test: force must not be blocked by the casualty check.
    server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("notes".into()),
            operation: "set_field".into(),
            field: "status".into(),
            values: None,
            definition: Some(definition(serde_json::json!({ "required": true }))),
            dry_run: None,
            force: Some(true),
            acknowledge_root_change: None,
        }))
        .await
        .expect("force must succeed against this git-backed harness");

    assert!(
        work.path()
            .join("notes")
            .join(crate::schema::SCHEMA_FILE_NAME)
            .exists(),
        "force must write the schema even though a document would fail it"
    );
}

#[tokio::test]
async fn update_schema_rejects_a_self_contradictory_definition() {
    // Parses fine, but declaring a scalar type alongside nested children fails
    // the next schema rebuild (and startup) — long after this call reported
    // success. It must be caught here instead.
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);

    let err = server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("notes".into()),
            operation: "set_field".into(),
            field: "planning".into(),
            values: None,
            definition: Some(definition(serde_json::json!({
                "type": "integer",
                "fields": { "prep": { "type": "integer" } }
            }))),
            dry_run: None,
            force: None,
            acknowledge_root_change: None,
        }))
        .await
        .expect_err("a field cannot be both a value and a container");

    assert!(format!("{:?}", err).contains("not both"));
    assert!(
        !tmp.path()
            .join("notes")
            .join(crate::schema::SCHEMA_FILE_NAME)
            .exists()
    );
}

#[tokio::test]
async fn update_schema_rejects_a_path_escaping_the_knowledge_base() {
    let tmp = tempfile::tempdir().unwrap();
    let server = schema_tool_server(&tmp);

    let err = server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("../escape".into()),
            operation: "add_values".into(),
            field: "tags".into(),
            values: Some(vec!["x".into()]),
            definition: None,
            dry_run: None,
            force: None,
            acknowledge_root_change: None,
        }))
        .await
        .expect_err("traversal must be rejected");

    assert!(format!("{:?}", err).contains(".."));
}

/// A root schema guarding root edits (#345), naming the owner KB's policy
/// document as its example.
const GUARDED_ROOT: &str =
    "root_edit:\n  require_ack: true\n  policy_doc: meta/schema-tag-policy.md\n";

fn root_add_values(acknowledge: Option<bool>) -> UpdateSchemaParams {
    UpdateSchemaParams {
        ops: None,
        path: None,
        operation: "add_values".into(),
        field: "tags".into(),
        values: Some(vec!["x".into()]),
        definition: None,
        dry_run: None,
        force: None,
        acknowledge_root_change: acknowledge,
    }
}

#[tokio::test]
async fn update_schema_root_add_values_refused_without_acknowledgment() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(&tmp, "", GUARDED_ROOT);
    let server = schema_tool_server(&tmp);

    let err = server
        .update_schema(Parameters(root_add_values(None)))
        .await
        .expect_err("root add_values must be refused without acknowledge_root_change");

    let msg = format!("{:?}", err);
    assert!(
        msg.contains("root schema changes are guarded"),
        "got: {msg}"
    );
    assert!(msg.contains("meta/schema-tag-policy.md"), "got: {msg}");
    assert!(msg.contains("acknowledge_root_change"), "got: {msg}");
    assert_eq!(
        std::fs::read_to_string(tmp.path().join(crate::schema::SCHEMA_FILE_NAME)).unwrap(),
        GUARDED_ROOT,
        "a refused change must leave the filesystem untouched"
    );
}

/// `require_ack` with no `policy_doc`: still refused, and the refusal names no
/// document at all.
#[tokio::test]
async fn update_schema_root_guard_without_a_policy_doc_names_none() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(&tmp, "", "root_edit:\n  require_ack: true\n");
    let server = schema_tool_server(&tmp);

    let msg = format!(
        "{:?}",
        server
            .update_schema(Parameters(root_add_values(None)))
            .await
            .expect_err("require_ack alone still guards")
    );
    assert!(
        msg.contains("root schema changes are guarded"),
        "got: {msg}"
    );
    assert!(msg.contains("acknowledge_root_change"), "got: {msg}");
    assert!(!msg.contains("policy in"), "got: {msg}");
    assert!(!msg.contains(".md"), "got: {msg}");
}

/// No `root_edit` block (or `require_ack: false`): root edits need no
/// acknowledgement (#345).
#[tokio::test]
async fn update_schema_root_edits_are_unguarded_without_root_edit() {
    for root in [None, Some("root_edit:\n  require_ack: false\n")] {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        if let Some(yaml) = root {
            write_schema_file(&work, "", yaml);
            git_commit_all(&work, crate::schema::SCHEMA_FILE_NAME, "add root schema");
        }
        let (server, _config) = make_git_backed_server(&work);

        server
            .update_schema(Parameters(root_add_values(None)))
            .await
            .expect("an unguarded root edit needs no acknowledgement");
        let yaml =
            std::fs::read_to_string(work.path().join(crate::schema::SCHEMA_FILE_NAME)).unwrap();
        let reparsed: crate::schema::SchemaFile = crate::yaml::from_str(&yaml).unwrap();
        assert_eq!(reparsed.fields["tags"].values, Some(vec!["x".to_string()]));
    }
}

/// A `root_edit` block that is on disk but not yet in the shared cache (it arrived
/// through git, and the worker has not rebuilt the cache) still guards: the root
/// file read under the lock is checked too (#345).
#[tokio::test]
async fn update_schema_root_guard_applies_before_the_cache_sees_it() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let (server, _config) = make_git_backed_server(&work);
    assert!(
        crate::schema::load_shared(&server.schema_cache)
            .root()
            .root_edit
            .is_none()
    );
    write_schema_file(&work, "", GUARDED_ROOT);
    git_commit_all(&work, crate::schema::SCHEMA_FILE_NAME, "guard root edits");

    let msg = format!(
        "{:?}",
        server
            .update_schema(Parameters(root_add_values(None)))
            .await
            .expect_err("the on-disk guard must refuse an unacknowledged root edit")
    );
    assert!(
        msg.contains("root schema changes are guarded"),
        "got: {msg}"
    );
    assert_eq!(
        std::fs::read_to_string(work.path().join(crate::schema::SCHEMA_FILE_NAME)).unwrap(),
        GUARDED_ROOT
    );
    server
        .update_schema(Parameters(root_add_values(Some(true))))
        .await
        .expect("acknowledged, the same edit lands");
}

#[tokio::test]
async fn update_schema_root_set_field_and_remove_field_are_refused_without_acknowledgment() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(
        &tmp,
        "",
        &format!("{GUARDED_ROOT}fields:\n  tags:\n    type: enum\n    values: [x]\n"),
    );
    let server = schema_tool_server(&tmp);

    let err = server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: None,
            operation: "set_field".into(),
            field: "status".into(),
            values: None,
            definition: Some(definition(serde_json::json!({ "type": "text" }))),
            dry_run: None,
            force: None,
            acknowledge_root_change: None,
        }))
        .await
        .expect_err("root set_field must be refused without acknowledge_root_change");
    assert!(format!("{:?}", err).contains("root schema changes are guarded"));

    let err = server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: None,
            operation: "remove_field".into(),
            field: "tags".into(),
            values: None,
            definition: None,
            dry_run: None,
            force: None,
            acknowledge_root_change: None,
        }))
        .await
        .expect_err("root remove_field must be refused without acknowledge_root_change");
    assert!(format!("{:?}", err).contains("root schema changes are guarded"));
}

/// Also pins that an `update_schema` rewrite of the root file keeps its
/// hand-edited root-only blocks (#345).
#[tokio::test]
async fn update_schema_root_add_values_allowed_with_acknowledgment() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    write_schema_file(
        &work,
        "",
        &format!("{GUARDED_ROOT}derived:\n  domain: {{from: path_segment, index: 0}}\n"),
    );
    git_commit_all(&work, crate::schema::SCHEMA_FILE_NAME, "add root schema");
    let (server, _config) = make_git_backed_server(&work);

    server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: None,
            operation: "add_values".into(),
            field: "tags".into(),
            values: Some(vec!["identity".into()]),
            definition: None,
            dry_run: None,
            force: None,
            acknowledge_root_change: Some(true),
        }))
        .await
        .expect("root add_values must succeed once acknowledged");

    let yaml = std::fs::read_to_string(work.path().join(crate::schema::SCHEMA_FILE_NAME)).unwrap();
    let reparsed: crate::schema::SchemaFile = crate::yaml::from_str(&yaml).unwrap();
    assert_eq!(
        reparsed.fields["tags"].values,
        Some(vec!["identity".to_string()]),
        "the acknowledged change must actually be written"
    );
    assert_eq!(
        reparsed.root_edit,
        Some(crate::schema::RootEditPolicy {
            require_ack: true,
            policy_doc: Some("meta/schema-tag-policy.md".into()),
        }),
        "{yaml}"
    );
    assert_eq!(
        reparsed.derived,
        Some(std::collections::BTreeMap::from([(
            "domain".to_string(),
            crate::schema::DerivedField::path_segment(0),
        )])),
        "{yaml}"
    );
}

#[tokio::test]
async fn update_schema_root_add_values_allowed_with_dry_run() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(&tmp, "", GUARDED_ROOT);
    let server = schema_tool_server(&tmp);

    let result = server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: None,
            operation: "add_values".into(),
            field: "tags".into(),
            values: Some(vec!["identity".into()]),
            definition: None,
            dry_run: Some(true),
            force: None,
            acknowledge_root_change: None,
        }))
        .await
        .expect("a root dry run must not be gated");

    let structured = result.structured_content.unwrap();
    assert_eq!(structured["dry_run"], serde_json::json!(true));
    assert_eq!(
        structured["summary"],
        serde_json::json!("added to 'tags': identity"),
        "the dry-run result must be exactly what the same edit against a non-root \
         scope would report — the root guard must not alter dry-run behavior"
    );
    assert_eq!(
        std::fs::read_to_string(tmp.path().join(crate::schema::SCHEMA_FILE_NAME)).unwrap(),
        GUARDED_ROOT,
        "a dry run must not touch the filesystem"
    );
}

#[tokio::test]
async fn update_schema_root_remove_values_allowed_without_acknowledgment() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    write_schema_file(
        &work,
        "",
        &format!("{GUARDED_ROOT}fields:\n  tags:\n    type: enum\n    values: [x, y]\n"),
    );
    git_commit_all(&work, crate::schema::SCHEMA_FILE_NAME, "add root schema");
    let (server, _config) = make_git_backed_server(&work);

    server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: None,
            operation: "remove_values".into(),
            field: "tags".into(),
            values: Some(vec!["y".into()]),
            definition: None,
            dry_run: None,
            force: None,
            acknowledge_root_change: None,
        }))
        .await
        .expect("root remove_values must not require acknowledge_root_change");

    let written = work.path().join(crate::schema::SCHEMA_FILE_NAME);
    let yaml = std::fs::read_to_string(&written).unwrap();
    let reparsed: crate::schema::SchemaFile = crate::yaml::from_str(&yaml).unwrap();
    assert_eq!(reparsed.fields["tags"].values, Some(vec!["x".to_string()]));
}

#[tokio::test]
async fn update_schema_non_root_add_values_allowed_without_acknowledgment() {
    // Already exercised incidentally by other update_schema tests (e.g.
    // `update_schema_accepts_a_change_that_breaks_nothing`), but this test names
    // the property the root guard must not regress: the gate is root-only.
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    write_schema_file(&work, "", GUARDED_ROOT);
    git_commit_all(&work, crate::schema::SCHEMA_FILE_NAME, "add root schema");
    let (server, _config) = make_git_backed_server(&work);

    server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("notes".into()),
            operation: "add_values".into(),
            field: "tags".into(),
            values: Some(vec!["x".into()]),
            definition: None,
            dry_run: None,
            force: None,
            acknowledge_root_change: None,
        }))
        .await
        .expect("non-root scopes must not require acknowledge_root_change");

    assert!(
        work.path()
            .join("notes")
            .join(crate::schema::SCHEMA_FILE_NAME)
            .exists()
    );
}

#[test]
fn oversized_schema_edits_are_rejected() {
    let mut params = update_params("add_values", "tags");
    params.values = Some(
        (0..MAX_SCHEMA_VALUES + 1)
            .map(|i| format!("v{i}"))
            .collect(),
    );
    assert!(build_schema_edit(&params).is_err());

    let mut params = update_params("add_values", "tags");
    params.values = Some(vec!["x".repeat(MAX_FILTER_STR_LEN + 1)]);
    assert!(build_schema_edit(&params).is_err());
}

// --- build_commit_message tests ---

#[tokio::test]
async fn update_schema_precommit_failure_on_new_schema_rolls_it_back() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let head_before = head_sha(&work);

    // No `.kb-schema.yaml` exists anywhere under `notes/` yet, so this write
    // creates one from scratch — the `is_new` branch of `write_raw_file`'s
    // rollback.
    force_git_commit_to_fail(&work);
    let (server, _config) = make_git_backed_server(&work);

    let result = server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("notes".into()),
            operation: "add_values".into(),
            field: "tags".into(),
            values: Some(vec!["x".into()]),
            definition: None,
            dry_run: None,
            force: None,
            acknowledge_root_change: None,
        }))
        .await;

    let err = result.expect_err("a rejected pre-commit hook must fail the schema write");
    assert_not_saved(&err);

    assert!(
        !work
            .path()
            .join("notes")
            .join(crate::schema::SCHEMA_FILE_NAME)
            .exists(),
        "the newly-written schema file must be removed on rollback — there is no \
         HEAD content for a brand-new scope to fall back to"
    );
    assert_eq!(
        head_before,
        head_sha(&work),
        "HEAD must not move on a rolled-back pre-commit failure"
    );
    assert_eq!(
        git_status_ignoring_state_db(&work),
        "",
        "the aborted `git add` must be unstaged too — no leftover addition that \
         could ride along on a later, unrelated commit"
    );
}

#[tokio::test]
async fn update_schema_precommit_failure_on_existing_schema_restores_previous_content() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");

    // Commit an existing schema for `notes/` BEFORE forcing commits to fail, so
    // there is real HEAD content for the rollback to restore.
    let original = "fields:\n  status:\n    type: enum\n    values: [active]\n";
    write_schema_file(&work, "notes", original);
    git_commit_all(
        &work,
        &format!("notes/{}", crate::schema::SCHEMA_FILE_NAME),
        "add notes schema",
    );
    let head_before = head_sha(&work);

    force_git_commit_to_fail(&work);
    // Built AFTER the real commit above, so the cache actually knows about the
    // existing `notes/` scope (needed for `update_schema` to resolve it and for
    // `write_raw_file` to see the write as an overwrite, not a create).
    let (server, _config) = make_git_backed_server(&work);

    let result = server
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
        .await;

    let err = result.expect_err("a rejected pre-commit hook must fail the schema write");
    assert_not_saved(&err);

    let written = work
        .path()
        .join("notes")
        .join(crate::schema::SCHEMA_FILE_NAME);
    assert_eq!(
        std::fs::read_to_string(&written).unwrap(),
        original,
        "the overwrite must be rolled back to the previous HEAD content, not left \
         holding the new (uncommitted) schema"
    );
    assert_eq!(
        head_before,
        head_sha(&work),
        "HEAD must not move on a rolled-back pre-commit failure"
    );
    assert_eq!(
        git_status_ignoring_state_db(&work),
        "",
        "working tree must be clean after rollback"
    );
}

pub(super) fn add_values_params(path: &str, field: &str, values: &[&str]) -> UpdateSchemaParams {
    UpdateSchemaParams {
        ops: None,
        path: Some(path.into()),
        operation: "add_values".into(),
        field: field.into(),
        values: Some(values.iter().map(|v| v.to_string()).collect()),
        definition: None,
        dry_run: None,
        force: None,
        acknowledge_root_change: None,
    }
}

fn write_legacy_schema_file(work: &tempfile::TempDir, dir: &str, yaml: &str) -> String {
    let target = work.path().join(dir);
    std::fs::create_dir_all(&target).unwrap();
    std::fs::write(target.join(crate::schema::LEGACY_SCHEMA_FILE_NAME), yaml).unwrap();
    format!("{dir}/{}", crate::schema::LEGACY_SCHEMA_FILE_NAME)
}

#[tokio::test]
async fn update_schema_migrates_a_legacy_schema_file_in_the_same_commit() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let legacy = write_legacy_schema_file(
        &work,
        "notes",
        "fields:\n  status:\n    type: enum\n    values: [active]\n",
    );
    git_commit_all(&work, &legacy, "add legacy notes schema");
    let head_before = head_sha(&work);
    let (server, _config) = make_git_backed_server(&work);

    let result = server
        .update_schema(Parameters(add_values_params("notes", "status", &["beta"])))
        .await
        .expect("update_schema on a legacy-named schema must succeed");

    let structured = single_representation(&result);
    assert_eq!(
        structured["path"], "notes/",
        "result path is the scope directory"
    );
    assert!(
        !structured.to_string().contains("schema.yaml"),
        "{structured}"
    );

    let canonical = work
        .path()
        .join("notes")
        .join(crate::schema::SCHEMA_FILE_NAME);
    let written = std::fs::read_to_string(&canonical).unwrap();
    assert!(
        written.contains("active") && written.contains("beta"),
        "{written}"
    );
    assert!(
        !work.path().join(&legacy).exists(),
        "the legacy file is removed"
    );

    // Exactly one commit, carrying both the addition and the removal.
    let parent = std::process::Command::new("git")
        .args(["rev-parse", "HEAD~1"])
        .current_dir(work.path())
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&parent.stdout).trim(), head_before);
    let show = std::process::Command::new("git")
        .args(["show", "--name-status", "--format=", "HEAD"])
        .current_dir(work.path())
        .output()
        .unwrap();
    let show = String::from_utf8_lossy(&show.stdout);
    assert!(show.contains("D\tnotes/.kb-schema.yaml"), "{show}");
    assert!(show.contains("A\tnotes/.schema.yaml"), "{show}");
    assert_eq!(git_status_ignoring_state_db(&work), "");

    let got = server
        .get_schema(Parameters(GetSchemaParams {
            path: Some("notes".into()),
            fields: None,
            values_only: None,
            values_in_use: None,
            compact: None,
        }))
        .await
        .unwrap();
    let fields = got.structured_content.unwrap()["fields"].clone();
    assert_eq!(fields["status"]["declared_in"], "notes/", "{fields}");
}

#[tokio::test]
async fn update_schema_precommit_failure_restores_the_legacy_schema_file() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let original = "fields:\n  status:\n    type: enum\n    values: [active]\n";
    let legacy = write_legacy_schema_file(&work, "notes", original);
    git_commit_all(&work, &legacy, "add legacy notes schema");
    let head_before = head_sha(&work);
    force_git_commit_to_fail(&work);
    let (server, _config) = make_git_backed_server(&work);

    let err = server
        .update_schema(Parameters(add_values_params("notes", "status", &["beta"])))
        .await
        .expect_err("a rejected commit must fail the schema write");
    assert_not_saved(&err);
    assert!(!err.message.contains("schema.yaml"), "{}", err.message);

    assert_eq!(
        std::fs::read_to_string(work.path().join(&legacy)).unwrap(),
        original,
        "the legacy file is restored"
    );
    assert!(
        !work
            .path()
            .join("notes")
            .join(crate::schema::SCHEMA_FILE_NAME)
            .exists()
    );
    assert_eq!(head_before, head_sha(&work));
    assert_eq!(git_status_ignoring_state_db(&work), "");
}

#[tokio::test]
async fn update_schema_migrating_an_untracked_legacy_file_leaves_it_out_of_the_commit() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    // Never committed: git does not know the legacy file, so the commit cannot
    // name its removal (`git add` of a missing untracked path fails).
    let legacy = write_legacy_schema_file(
        &work,
        "notes",
        "fields:\n  status:\n    type: enum\n    values: [active]\n",
    );
    let (server, _config) = make_git_backed_server(&work);

    server
        .update_schema(Parameters(add_values_params("notes", "status", &["beta"])))
        .await
        .expect("an untracked legacy schema file must still migrate");

    assert!(
        !work.path().join(&legacy).exists(),
        "the legacy file is removed"
    );
    let show = std::process::Command::new("git")
        .args(["show", "--name-status", "--format=", "HEAD"])
        .current_dir(work.path())
        .output()
        .unwrap();
    let show = String::from_utf8_lossy(&show.stdout);
    assert!(show.contains("A\tnotes/.schema.yaml"), "{show}");
    assert!(!show.contains("kb-schema"), "{show}");
    assert_eq!(git_status_ignoring_state_db(&work), "");
}

#[tokio::test]
async fn update_schema_precommit_failure_rewrites_an_untracked_legacy_file() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let original = "fields:\n  status:\n    type: enum\n    values: [active]\n";
    let legacy = write_legacy_schema_file(&work, "notes", original);
    let head_before = head_sha(&work);
    force_git_commit_to_fail(&work);
    let (server, _config) = make_git_backed_server(&work);

    let err = server
        .update_schema(Parameters(add_values_params("notes", "status", &["beta"])))
        .await
        .expect_err("a rejected commit must fail the schema write");
    assert_not_saved(&err);

    assert_eq!(
        std::fs::read_to_string(work.path().join(&legacy)).unwrap(),
        original,
        "an untracked legacy file has no HEAD copy, so it is rewritten from memory"
    );
    assert!(
        !work
            .path()
            .join("notes")
            .join(crate::schema::SCHEMA_FILE_NAME)
            .exists()
    );
    assert_eq!(head_before, head_sha(&work));
}

#[tokio::test]
async fn update_schema_add_values_of_an_inherited_value_changes_nothing() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    write_schema_file(
        &work,
        "",
        "fields:\n  tags:\n    type: list\n    values: [a, b]\n",
    );
    git_commit_all(&work, crate::schema::SCHEMA_FILE_NAME, "add root schema");
    let head_before = head_sha(&work);
    let (server, _config) = make_git_backed_server(&work);

    // The scope already permits `a` through the root, so there is nothing to
    // write: no child schema, no commit, and no claim that `a` was added.
    let err = server
        .update_schema(Parameters(add_values_params("notes", "tags", &["a"])))
        .await
        .expect_err("an inherited value leaves nothing to add");
    assert!(err.message.contains("already permits"), "{}", err.message);
    assert_eq!(head_before, head_sha(&work));
    assert!(
        !work
            .path()
            .join("notes")
            .join(crate::schema::SCHEMA_FILE_NAME)
            .exists()
    );
}

#[tokio::test]
async fn update_schema_add_values_in_a_child_scope_extends_the_inherited_set() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    write_schema_file(
        &work,
        "",
        "fields:\n  tags:\n    type: list\n    values: [a, b]\n",
    );
    git_commit_all(&work, crate::schema::SCHEMA_FILE_NAME, "add root schema");
    let (server, _config) = make_git_backed_server(&work);

    server
        .update_schema(Parameters(add_values_params("notes", "tags", &["c"])))
        .await
        .expect("add_values in a child scope");

    let written = std::fs::read_to_string(
        work.path()
            .join("notes")
            .join(crate::schema::SCHEMA_FILE_NAME),
    )
    .unwrap();
    let parsed: crate::schema::SchemaFile = crate::yaml::from_str(&written).unwrap();
    let tags = &parsed.fields["tags"];
    assert_eq!(
        tags.values.as_deref(),
        Some(&["$values".to_string(), "c".to_string()][..])
    );
    assert_eq!(
        tags.ty, None,
        "the inherited type is kept, not forced to enum"
    );

    let got = server
        .get_schema(Parameters(GetSchemaParams {
            path: Some("notes".into()),
            fields: Some(vec!["tags".into()]),
            values_only: None,
            values_in_use: None,
            compact: None,
        }))
        .await
        .unwrap();
    let tags = got.structured_content.unwrap()["fields"]["tags"].clone();
    assert_eq!(tags["values"], serde_json::json!(["a", "b", "c"]));
    assert_eq!(tags["type"], "list");
}

/// A schema change saved while ANOTHER directory's schema is invalid is not
/// in effect yet. Clients that read only `structured_content` (Claude Code)
/// must be told, or they write against rules that are not enforced.
#[tokio::test]
async fn update_schema_reports_a_saved_but_not_in_effect_change_in_structured_content() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    let (server, _config) = make_git_backed_server(&work);
    // Arrives outside update_schema (as a push would) after startup.
    std::fs::create_dir_all(work.path().join("broken")).unwrap();
    std::fs::write(
        work.path()
            .join("broken")
            .join(crate::schema::SCHEMA_FILE_NAME),
        "not_a_schema_key: true\n",
    )
    .unwrap();

    let result = server
        .update_schema(Parameters(add_values_params(
            "notes",
            "status",
            &["active"],
        )))
        .await
        .expect("the change itself is valid and saved");
    let structured = single_representation(&result);
    let warning = structured["warning"].as_str().expect("warning present");
    assert!(warning.contains("NOT in effect"), "{warning}");
    assert!(warning.contains("broken/"), "{warning}");
    assert_git_free("update_schema warning", warning);
}

#[tokio::test]
async fn update_schema_postcommit_failure_leaves_commit_and_reports_pending_sync() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");

    let mut config = make_test_resolved_config(work.path());
    {
        let c = Arc::get_mut(&mut config).unwrap();
        c.write.dedup_enabled = false;
        // No such path — `git fetch` fails immediately, no network required, but
        // only AFTER the schema write's `git add`/`git commit` have already
        // succeeded locally.
        c.source.git_url = Some("/nonexistent/path/to/repo.git".to_string());
    }
    let server = make_write_test_server(&work, &["**/*.md".to_string()], config);

    let result = server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("notes".into()),
            operation: "add_values".into(),
            field: "status".into(),
            values: Some(vec!["active".into()]),
            definition: None,
            dry_run: None,
            force: None,
            acknowledge_root_change: None,
        }))
        .await;

    let result = result.expect("a post-commit sync failure must still report as success");
    // A saved-but-unsynced schema change is, to the caller, simply saved.
    assert_plain_success(&result);
    let structured = single_representation(&result);
    assert_eq!(structured["path"], "notes/");
    assert!(structured.get("warning").is_none(), "{structured}");

    // The schema change IS a real local commit — the file stays written, and HEAD
    // records it. None of this is rolled back just because the push failed.
    let written = work
        .path()
        .join("notes")
        .join(crate::schema::SCHEMA_FILE_NAME);
    assert!(written.exists());
    let show = std::process::Command::new("git")
        .args(["show", "--name-only", "--format=", "HEAD"])
        .current_dir(work.path())
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&show.stdout).contains(crate::schema::SCHEMA_FILE_NAME),
        "the schema commit must be present in local HEAD"
    );

    // The reasoning behind NOT rolling back a PostCommit failure only pays off if
    // the shared schema cache was actually rebuilt from the new (committed)
    // content despite the push failure — mirrors
    // `create_document_immediately_after_update_schema_validates_against_the_new_rules`,
    // but for the pending-sync outcome instead of a clean sync. If `update_schema`
    // skipped the cache rebuild whenever the write wasn't a clean `Synced`, this
    // next call would wrongly reject "beta" against the stale pre-change schema.
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
        accepted.is_err(),
        "sanity check: this call's OWN schema edit only added 'active', not \
         'beta' — this must still be rejected, otherwise this test would not be \
         distinguishing 'cache rebuilt' from 'no schema at all'"
    );
    let err_msg = format!("{:?}", accepted.err());
    assert!(
        err_msg.contains("beta"),
        "must be rejected specifically for the 'beta' value, not some unrelated \
         failure: {err_msg}"
    );
}

/// When there is no git repository at all, `git add` fails (`PreCommit`) and the
/// rollback attempt (`git reset` via `unstage`) ALSO fails, since there is no
/// repo to reset anything in. This is the third, worse outcome: the schema file
/// is gone from disk with no corresponding commit anywhere. It must be reported
/// distinctly rather than masquerading as either a clean write or a clean no-op —
/// mirrors `delete_document_with_no_git_repo_reports_inconsistent_state`.
#[tokio::test]
async fn update_schema_rollback_failure_reports_inconsistent_state() {
    let tmp = tempfile::tempdir().unwrap();
    let config = make_test_resolved_config(tmp.path());
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);

    let result = server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("notes".into()),
            operation: "add_values".into(),
            field: "tags".into(),
            values: Some(vec!["x".into()]),
            definition: None,
            dry_run: None,
            force: None,
            acknowledge_root_change: None,
        }))
        .await;

    let err = result.expect_err("writing a schema with no git repo must fail");
    assert_not_saved_unverified(&err);

    // The remove succeeded (there is no repo to fail that part), but the
    // subsequent `unstage` could not run against a nonexistent repo — that
    // mismatch (file gone, no git awareness of it ever having existed) IS the
    // inconsistent state being reported.
    assert!(
        !tmp.path()
            .join("notes")
            .join(crate::schema::SCHEMA_FILE_NAME)
            .exists()
    );
}

#[test]
fn move_directory_success_reports_a_carried_schema_by_directory() {
    let success = DirectoryMoveSuccess {
        moved: vec![
            ("src/a.md".to_string(), "dest/a.md".to_string()),
            (
                "src/sub/.kb-schema.yaml".to_string(),
                "dest/sub/.kb-schema.yaml".to_string(),
            ),
        ],
        rewritten_paths: Vec::new(),
        merged: false,
    };

    let result = move_directory_success_to_result(success, "src", "dest");

    let structured = single_representation(&result);
    assert_eq!(structured["action"], "moved");
    assert_eq!(structured["from"], "src");
    assert_eq!(structured["path"], "dest");
    assert_eq!(
        structured["moved"],
        serde_json::json!([{"from": "src/a.md", "to": "dest/a.md"}])
    );
    assert!(structured.get("rewritten_paths").is_none(), "{structured}");
    assert_eq!(structured["moved_schema_dirs"][0]["from"], "src/sub/");
    assert_eq!(structured["moved_schema_dirs"][0]["to"], "dest/sub/");
    assert_no_schema_file_name(&structured.to_string());
}

// -- move_directory_error_to_mcp_error: destination-cascade wording -----

/// `update_schema` arguments as a client sends them, through the real
/// `Deserialize` (so `ops` entries are parsed exactly as on the wire).
fn schema_args(args: serde_json::Value) -> UpdateSchemaParams {
    serde_json::from_value(args).expect("test fixture must be valid update_schema args")
}

const BATCH_ROOT: &str = "fields:
  tags:
    type: list
    values: [a, b, c]
  status:
    type: enum
    values: [draft, active]
";

/// A git-backed server over a work tree whose root schema is `BATCH_ROOT`,
/// committed.
fn batch_server() -> (
    tempfile::TempDir,
    tempfile::TempDir,
    KbSearchServer,
    Arc<crate::config::ResolvedConfig>,
) {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    write_schema_file(&work, "", BATCH_ROOT);
    git_commit_all(&work, crate::schema::SCHEMA_FILE_NAME, "add root schema");
    let (server, config) = make_git_backed_server(&work);
    (bare, work, server, config)
}

fn notes_schema(work: &tempfile::TempDir) -> Option<crate::schema::SchemaFile> {
    let path = work
        .path()
        .join("notes")
        .join(crate::schema::SCHEMA_FILE_NAME);
    std::fs::read_to_string(path)
        .ok()
        .map(|yaml| crate::yaml::from_str(&yaml).unwrap())
}

#[test]
fn ops_and_the_flat_parameters_are_mutually_exclusive_and_bounded() {
    let msg = |args: serde_json::Value| build_schema_edits(&schema_args(args)).unwrap_err().message;
    assert!(msg(serde_json::json!({})).contains("needs `ops`"));
    assert!(msg(serde_json::json!({"ops": []})).contains("at least one"));
    let both = msg(serde_json::json!({
        "operation": "add_values",
        "field": "tags",
        "values": ["x"],
        "ops": [{"operation": "remove_field", "field": "tags"}]
    }));
    assert!(both.contains("not both"), "{both}");
    // The flat `definition` cannot carry a patch's nulls, so the shim refuses it.
    let legacy_patch = msg(serde_json::json!({
        "operation": "patch_field",
        "field": "tags",
        "definition": {"required": true}
    }));
    assert!(
        legacy_patch.contains("only available through `ops`"),
        "{legacy_patch}"
    );
    let many: Vec<serde_json::Value> = (0..=MAX_SCHEMA_OPS)
        .map(|i| serde_json::json!({"operation": "remove_field", "field": format!("f{i}")}))
        .collect();
    assert!(msg(serde_json::json!({"ops": many})).contains("too many operations"));
    // Per-op bounds still apply, and name the offending op in a batch.
    let bad = msg(serde_json::json!({"ops": [
        {"operation": "remove_field", "field": "ok"},
        {"operation": "add_values", "field": "tags", "values": []}
    ]}));
    assert!(bad.starts_with("ops[1]: "), "{bad}");
    assert!(bad.contains("non-empty values list"), "{bad}");
    let deep = vec!["a"; MAX_SCHEMA_PATH_SEGMENTS + 1].join(".");
    assert!(
        msg(serde_json::json!({"ops": [{"operation": "remove_field", "field": deep}]}))
            .contains("too deep")
    );
    // `operation` is a closed enum, and an op takes no unknown keys.
    for bad in [
        serde_json::json!({"ops": [{"operation": "drop_table", "field": "x"}]}),
        serde_json::json!({"ops": [{"operation": "remove_field", "field": "x", "valuez": []}]}),
    ] {
        assert!(serde_json::from_value::<UpdateSchemaParams>(bad).is_err());
    }
    // patch_field needs keys to change.
    assert!(
        msg(serde_json::json!({"ops": [{"operation": "patch_field", "field": "x", "definition": {}}]}))
            .contains("requires a definition")
    );
}

#[test]
fn a_patch_definition_keeps_its_nulls_and_also_arrives_as_a_string() {
    let edits = build_schema_edits(&schema_args(serde_json::json!({"ops": [
        {"operation": "patch_field", "field": "x", "definition": {"required": null, "indexed": true}},
        {"operation": "patch_field", "field": "y", "definition": "{\"open\": null}"}
    ]})))
    .unwrap();
    match &edits[0] {
        crate::schema::SchemaEdit::PatchField { patch, .. } => {
            assert_eq!(patch["required"], serde_json::Value::Null);
            assert_eq!(patch["indexed"], serde_json::json!(true));
        }
        other => panic!("{other:?}"),
    }
    assert!(matches!(
        &edits[1],
        crate::schema::SchemaEdit::PatchField { patch, .. } if patch.contains_key("open")
    ));
    // set_field in an op still refuses an unknown key, naming it.
    let err = build_schema_edits(&schema_args(serde_json::json!({"ops": [
        {"operation": "set_field", "field": "x", "definition": {"typ": "text"}}
    ]})))
    .unwrap_err();
    assert!(err.message.contains("typ"), "{}", err.message);
}

#[test]
fn reinherit_is_the_null_free_way_to_inherit_a_key_again() {
    let edits = |args: serde_json::Value| build_schema_edits(&schema_args(args));
    // `reinherit` lowers to exactly the patch a literal `null` gives, alone or
    // next to keys being set.
    let via_list = edits(serde_json::json!({"ops": [
        {"operation": "patch_field", "field": "x", "reinherit": ["required"]},
        {"operation": "patch_field", "field": "y", "definition": {"indexed": true},
         "reinherit": ["values", "open"]}
    ]}))
    .unwrap();
    let via_null = edits(serde_json::json!({"ops": [
        {"operation": "patch_field", "field": "x", "definition": {"required": null}},
        {"operation": "patch_field", "field": "y",
         "definition": {"indexed": true, "values": null, "open": null}}
    ]}))
    .unwrap();
    assert_eq!(format!("{via_list:?}"), format!("{via_null:?}"));

    let msg = |args: serde_json::Value| edits(args).unwrap_err().message;
    let not_patch = msg(serde_json::json!({"ops": [
        {"operation": "set_field", "field": "x", "definition": {"type": "text"},
         "reinherit": ["required"]}
    ]}));
    assert!(not_patch.contains("only to patch_field"), "{not_patch}");
    let unknown = msg(serde_json::json!({"ops": [
        {"operation": "patch_field", "field": "x", "reinherit": ["requird"]}
    ]}));
    assert!(
        unknown.contains("requird") && unknown.contains(FIELD_DEFINITION_KEYS),
        "{unknown}"
    );
    let both = msg(serde_json::json!({"ops": [
        {"operation": "patch_field", "field": "x", "definition": {"required": true},
         "reinherit": ["required"]}
    ]}));
    assert!(both.contains("both set"), "{both}");
    // An empty list with no definition still has nothing to change.
    assert!(
        msg(serde_json::json!({"ops": [
            {"operation": "patch_field", "field": "x", "reinherit": []}
        ]}))
        .contains("requires a definition")
    );
}

#[test]
fn field_definition_key_names_match_the_listed_keys() {
    let listed: Vec<&str> = FIELD_DEFINITION_KEYS
        .split(',')
        .map(|k| k.trim().trim_matches('`'))
        .collect();
    assert_eq!(listed, FIELD_DEFINITION_KEY_NAMES);
}

#[tokio::test]
async fn reinherit_through_the_tool_drops_the_local_key() {
    let (_bare, work, server, _config) = batch_server();
    write_schema_file(
        &work,
        "notes",
        "fields:\n  status:\n    required: true\n    indexed: true\n",
    );
    git_commit_all(
        &work,
        &format!("notes/{}", crate::schema::SCHEMA_FILE_NAME),
        "notes schema",
    );
    server
        .update_schema(Parameters(schema_args(serde_json::json!({
            "path": "notes",
            "ops": [{"operation": "patch_field", "field": "status",
                     "reinherit": ["required"]}]
        }))))
        .await
        .unwrap();
    assert_eq!(
        notes_schema(&work).unwrap().fields["status"],
        crate::schema::RawFieldDef {
            indexed: Some(true),
            ..Default::default()
        }
    );
}

#[tokio::test]
async fn update_schema_ops_land_as_one_commit() {
    let (_bare, work, server, _config) = batch_server();
    let before = commit_count(&work);

    let result = server
        .update_schema(Parameters(schema_args(serde_json::json!({
            "path": "notes",
            "ops": [
                {"operation": "add_values", "field": "tags", "values": ["d"]},
                {"operation": "remove_values", "field": "status", "values": ["draft"]},
                {"operation": "set_field", "field": "rating", "definition": {"type": "integer"}},
                {"operation": "patch_field", "field": "rating", "definition": {"indexed": true}}
            ]
        }))))
        .await
        .expect("the batch lands");
    let structured = single_representation(&result);
    assert_eq!(structured["path"], "notes/");
    let summary = structured["summary"].as_str().unwrap();
    assert!(summary.contains("added to 'tags': d"), "{summary}");
    assert!(summary.contains("patched 'rating'"), "{summary}");

    assert_eq!(
        commit_count(&work),
        before + 1,
        "one commit for the whole batch"
    );
    let subject = head_subject(&work);
    assert!(
        subject.starts_with("schema: 4 changes in notes/: added to 'tags'"),
        "{subject}"
    );
    let file = notes_schema(&work).unwrap();
    assert_eq!(
        file.fields["tags"].values,
        Some(vec!["$values".into(), "d".into()])
    );
    assert_eq!(
        file.fields["status"].exclude_values,
        Some(vec!["draft".into()])
    );
    assert_eq!(
        file.fields["rating"].ty,
        Some(crate::schema::FieldType::Integer)
    );
    assert_eq!(file.fields["rating"].indexed, Some(true));
}

#[tokio::test]
async fn update_schema_ops_are_all_or_nothing() {
    let (_bare, work, server, _config) = batch_server();
    let head_before = head_sha(&work);

    let err = server
        .update_schema(Parameters(schema_args(serde_json::json!({
            "path": "notes",
            "ops": [
                {"operation": "add_values", "field": "tags", "values": ["d"]},
                {"operation": "set_field", "field": "rating", "definition": {"type": "integer"}},
                {"operation": "remove_field", "field": "never_declared"}
            ]
        }))))
        .await
        .expect_err("the last op fails, so the batch does");
    assert!(
        err.message
            .starts_with("ops[2] (remove_field 'never_declared'): "),
        "{}",
        err.message
    );
    assert_eq!(head_sha(&work), head_before, "nothing committed");
    assert!(notes_schema(&work).is_none(), "nothing written");
    assert_eq!(git_status(&work), "", "the work tree is clean");

    // A batch that validates op by op but leaves a self-contradictory file is
    // refused as a whole too: `validate_self` runs once, on the result.
    let err = server
        .update_schema(Parameters(schema_args(serde_json::json!({
            "path": "notes",
            "ops": [
                {"operation": "set_field", "field": "planning", "definition": {"type": "text"}},
                {"operation": "patch_field", "field": "planning", "definition": {"fields": {"x": {"type": "text"}}}}
            ]
        }))))
        .await
        .expect_err("a value with nested fields is refused");
    assert!(err.message.contains("nested fields"), "{}", err.message);
    assert_eq!(head_sha(&work), head_before);
    assert!(notes_schema(&work).is_none());
}

#[tokio::test]
async fn the_flat_parameters_act_exactly_like_a_one_op_batch() {
    for dry_run in [true, false] {
        let mut results = Vec::new();
        let mut files = Vec::new();
        for args in [
            serde_json::json!({
                "path": "notes",
                "operation": "set_field",
                "field": "status",
                "definition": {"type": "enum", "values": ["x"], "required": true},
                "dry_run": dry_run,
            }),
            serde_json::json!({
                "path": "notes",
                "ops": [{
                    "operation": "set_field",
                    "field": "status",
                    "definition": {"type": "enum", "values": ["x"], "required": true}
                }],
                "dry_run": dry_run,
            }),
        ] {
            let (_bare, work, server, _config) = batch_server();
            let result = server
                .update_schema(Parameters(schema_args(args)))
                .await
                .unwrap();
            results.push(single_representation(&result));
            files.push((
                notes_schema(&work).map(|f| f.to_yaml().unwrap()),
                head_subject(&work),
            ));
        }
        assert_eq!(results[0], results[1], "dry_run={dry_run}");
        assert_eq!(files[0], files[1], "dry_run={dry_run}");
        if dry_run {
            assert_eq!(results[0]["field"], "status");
            assert!(results[0].get("ops").is_none());
        }
    }
}

#[tokio::test]
async fn a_batch_dry_run_reports_each_op_and_the_combined_casualties() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(&tmp, "", BATCH_ROOT);
    let server = schema_tool_server(&tmp);
    seed_document(
        &server,
        "notes/a.md",
        serde_json::json!({"title": "A", "status": "draft"}),
    )
    .await;

    let result = server
        .update_schema(Parameters(schema_args(serde_json::json!({
            "path": "notes",
            "dry_run": true,
            "ops": [
                {"operation": "remove_values", "field": "status", "values": ["draft"]},
                {"operation": "set_field", "field": "rating", "definition": {"required": true}},
                {"operation": "add_values", "field": "status", "values": ["paused"]}
            ]
        }))))
        .await
        .unwrap();
    let structured = single_representation(&result);
    assert_eq!(structured["dry_run"], true);
    assert!(structured.get("field").is_none(), "{structured}");
    let ops = structured["ops"].as_array().unwrap();
    assert_eq!(ops.len(), 3);
    assert_eq!(ops[0]["operation"], "remove_values");
    assert!(
        ops[0]["summary"]
            .as_str()
            .unwrap()
            .contains("now excluded here")
    );
    // Each field as the whole batch leaves it.
    assert_eq!(
        ops[0]["definition"]["values"],
        serde_json::json!(["active", "paused"])
    );
    assert_eq!(ops[1]["definition"]["required"], true);
    // One casualty pass over the batch's result: the document fails both the
    // excluded value and the new required field, and is reported once.
    let casualties = structured["would_invalidate"].as_array().unwrap();
    assert_eq!(casualties.len(), 1, "{casualties:?}");
    assert_eq!(casualties[0]["error_count"], 2, "{casualties:?}");
    assert!(
        !tmp.path()
            .join("notes")
            .join(crate::schema::SCHEMA_FILE_NAME)
            .exists()
    );
}

#[tokio::test]
async fn the_root_guard_is_judged_per_op_in_a_batch() {
    let bare = crate::git::tests::create_bare_repo("master");
    let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
    write_schema_file(
        &work,
        "",
        &format!("{GUARDED_ROOT}fields:\n  tags:\n    values: [a, b, c]\n"),
    );
    git_commit_all(&work, crate::schema::SCHEMA_FILE_NAME, "add root schema");
    let (server, _config) = make_git_backed_server(&work);
    let root_values = || {
        let yaml =
            std::fs::read_to_string(work.path().join(crate::schema::SCHEMA_FILE_NAME)).unwrap();
        crate::yaml::from_str::<crate::schema::SchemaFile>(&yaml)
            .unwrap()
            .fields["tags"]
            .values
            .clone()
    };
    let ops = serde_json::json!([
        {"operation": "remove_values", "field": "tags", "values": ["a"]},
        {"operation": "add_values", "field": "tags", "values": ["z"]}
    ]);

    let err = server
        .update_schema(Parameters(schema_args(serde_json::json!({"ops": ops}))))
        .await
        .expect_err("one gated op guards the whole batch");
    assert!(
        err.message.contains("root schema changes are guarded"),
        "{}",
        err.message
    );
    assert_eq!(
        root_values(),
        Some(vec!["a".into(), "b".into(), "c".into()])
    );

    // A dry run of it is not guarded.
    server
        .update_schema(Parameters(schema_args(
            serde_json::json!({"ops": ops, "dry_run": true}),
        )))
        .await
        .expect("a dry run is exempt");

    // A batch of only exempt ops needs no acknowledgement.
    server
        .update_schema(Parameters(schema_args(serde_json::json!({"ops": [
            {"operation": "remove_values", "field": "tags", "values": ["c"]},
            {"operation": "remove_values", "field": "tags", "values": ["b"]}
        ]}))))
        .await
        .expect("remove_values is exempt");
    assert_eq!(root_values(), Some(vec!["a".into()]));

    server
        .update_schema(Parameters(schema_args(serde_json::json!({
            "ops": ops,
            "acknowledge_root_change": true
        }))))
        .await
        .expect("acknowledged");
    assert_eq!(root_values(), Some(vec!["z".into()]));
}

#[tokio::test]
async fn remove_values_on_an_inherited_value_excludes_it_and_add_values_undoes_it() {
    let (_bare, work, server, _config) = batch_server();

    server
        .update_schema(Parameters(UpdateSchemaParams {
            operation: "remove_values".into(),
            ..add_values_params("notes", "tags", &["a"])
        }))
        .await
        .expect("an inherited value is excluded rather than refused");
    let file = notes_schema(&work).unwrap();
    assert_eq!(file.fields["tags"].exclude_values, Some(vec!["a".into()]));
    assert_eq!(file.fields["tags"].values, None);
    let got = server
        .get_schema(Parameters(GetSchemaParams {
            path: Some("notes".into()),
            fields: Some(vec!["tags".into()]),
            ..Default::default()
        }))
        .await
        .unwrap();
    assert_eq!(
        got.structured_content.unwrap()["fields"]["tags"]["values"],
        serde_json::json!(["b", "c"])
    );

    let result = server
        .update_schema(Parameters(add_values_params("notes", "tags", &["a"])))
        .await
        .expect("adding the excluded value back");
    assert_eq!(
        single_representation(&result)["summary"],
        "added to 'tags': a"
    );
    let file = notes_schema(&work).unwrap();
    assert!(!file.fields.contains_key("tags"), "{:?}", file.fields);
}

#[tokio::test]
async fn patch_field_through_the_tool_minimizes_and_reports_cross_kind_changes() {
    let (_bare, work, server, _config) = batch_server();
    write_schema_file(
        &work,
        "notes",
        "fields:\n  status:\n    required: true\n    indexed: true\n",
    );
    git_commit_all(
        &work,
        &format!("notes/{}", crate::schema::SCHEMA_FILE_NAME),
        "notes schema",
    );

    // `null` re-inherits `required`; `type: enum` is what the root says already.
    let result = server
        .update_schema(Parameters(schema_args(serde_json::json!({
            "path": "notes",
            "ops": [{"operation": "patch_field", "field": "status",
                     "definition": {"required": null, "type": "enum"}}]
        }))))
        .await
        .unwrap();
    assert_eq!(
        single_representation(&result)["summary"],
        "patched 'status'"
    );
    let file = notes_schema(&work).unwrap();
    assert_eq!(
        file.fields["status"],
        crate::schema::RawFieldDef {
            indexed: Some(true),
            ..Default::default()
        }
    );

    // A same-kind change warns about nothing; a cross-kind one lands with a
    // warning.
    let same = server
        .update_schema(Parameters(schema_args(serde_json::json!({
            "path": "notes",
            "ops": [{"operation": "patch_field", "field": "status",
                     "definition": {"type": "text"}}]
        }))))
        .await
        .unwrap();
    assert!(single_representation(&same).get("warnings").is_none());
    let cross = server
        .update_schema(Parameters(schema_args(serde_json::json!({
            "path": "notes",
            "ops": [{"operation": "patch_field", "field": "tags",
                     "definition": {"type": "text"}}]
        }))))
        .await
        .expect("a cross-kind change is a warning, not a refusal");
    let warnings = single_representation(&cross)["warnings"].clone();
    assert!(
        warnings[0]
            .as_str()
            .is_some_and(|w| w.contains("'list'") && w.contains("'text'")),
        "{warnings}"
    );
    assert_eq!(
        notes_schema(&work).unwrap().fields["tags"].ty,
        Some(crate::schema::FieldType::Text)
    );
}

#[tokio::test]
async fn a_removed_field_disappears_from_get_schema_below_the_scope() {
    let (_bare, work, server, _config) = batch_server();
    server
        .update_schema(Parameters(schema_args(serde_json::json!({
            "path": "notes",
            "ops": [{"operation": "set_field", "field": "status",
                     "definition": {"removed": true}}]
        }))))
        .await
        .unwrap();
    assert!(notes_schema(&work).unwrap().fields["status"].removed);
    for path in ["notes", "notes/deeper"] {
        let got = server
            .get_schema(Parameters(GetSchemaParams {
                path: Some(path.into()),
                ..Default::default()
            }))
            .await
            .unwrap();
        let fields = got.structured_content.unwrap()["fields"].clone();
        assert!(fields.get("status").is_none(), "{path}: {fields}");
        assert!(fields.get("tags").is_some(), "{path}: {fields}");
    }
}

#[tokio::test]
async fn concurrent_batches_both_land() {
    let h = concurrent_harness();
    let schema_rel = format!("docs/{}", crate::schema::SCHEMA_FILE_NAME);
    peer_push(
        h.peer.path(),
        &schema_rel,
        "fields:\n  status:\n    type: enum\n    values: [active]\n",
    );
    let batch = |value: &str, field: &str| {
        schema_args(serde_json::json!({
            "path": "docs",
            "ops": [
                {"operation": "add_values", "field": "status", "values": [value]},
                {"operation": "set_field", "field": field, "definition": {"type": "text"}}
            ]
        }))
    };
    let other = h.server.clone();
    let theirs = batch("gamma", "theirs");
    with_hook_once(
        crate::git::HookPoint::BeforeLock,
        move || async move {
            other
                .update_schema(Parameters(theirs))
                .await
                .expect("the concurrent batch lands");
        },
        h.server.update_schema(Parameters(batch("beta", "ours"))),
    )
    .await
    .expect("the batch applies to the schema as it is under the lock");

    let written = std::fs::read_to_string(h.work.path().join(&schema_rel)).unwrap();
    for expected in ["active", "beta", "gamma", "theirs", "ours"] {
        assert!(written.contains(expected), "{expected}: {written}");
    }
    assert_eq!(rev(h.work.path(), "HEAD"), rev(h._bare.path(), "master"));
}

#[tokio::test]
async fn get_schema_compact_names_each_declaring_directory_once() {
    let tmp = tempfile::tempdir().unwrap();
    write_schema_file(
        &tmp,
        "",
        "fields:\n  title:\n    required: true\n  a:\n    type: text\n",
    );
    write_schema_file(&tmp, "food", "fields:\n  prep:\n    type: integer\n");
    let server = schema_tool_server(&tmp);
    let get = |compact: Option<bool>| {
        server.get_schema(Parameters(GetSchemaParams {
            path: Some("food".into()),
            compact,
            ..Default::default()
        }))
    };

    let full = get(None).await.unwrap().structured_content.unwrap();
    assert_eq!(full["fields"]["prep"]["declared_in"], "food/");
    assert!(full.get("declared_in").is_none());

    let compact = get(Some(true)).await.unwrap().structured_content.unwrap();
    assert_eq!(
        compact["declared_in"],
        serde_json::json!({"/": ["a", "title"], "food/": ["prep"]})
    );
    assert_eq!(
        compact["fields"],
        serde_json::json!({
            "a": {"type": "text"},
            "prep": {"type": "integer"},
            "title": {"required": true}
        })
    );
}
