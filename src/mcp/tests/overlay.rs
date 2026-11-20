use rmcp::handler::server::wrapper::Parameters;

use crate::config::Granularity;
use crate::mcp::search::*;
use crate::mcp::test_support::*;
use crate::mcp::*;
#[test]
fn get_info_returns_dynamic_instructions() {
    use rmcp::ServerHandler;

    let tmp = tempfile::tempdir().unwrap();
    let custom_text = "Custom KB instructions.\nAvailable domain: infra, networking";
    let instructions = Arc::new(RwLock::new(custom_text.to_string()));

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

    let info = server.get_info();
    let returned = info.instructions.unwrap();
    assert_eq!(returned, custom_text);
}

#[test]
fn get_info_reports_correct_server_name_and_version() {
    // Verify that serverInfo.name and .version come from CARGO_PKG_NAME
    // and CARGO_PKG_VERSION, not from rmcp's build env (#277).
    use rmcp::ServerHandler;

    let tmp = tempfile::tempdir().unwrap();
    let instructions = Arc::new(RwLock::new("Test instructions".to_string()));

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

    let info = server.get_info();
    assert_eq!(info.server_info.name, env!("CARGO_PKG_NAME"));
    assert_eq!(info.server_info.version, env!("CARGO_PKG_VERSION"));
}

#[test]
fn get_info_reflects_updated_instructions() {
    use rmcp::ServerHandler;

    let tmp = tempfile::tempdir().unwrap();
    let instructions = Arc::new(RwLock::new("Initial instructions".to_string()));

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

    let server = KbSearchServer::new(
        embed,
        qdrant,
        "test".into(),
        tmp.path().to_path_buf(),
        &["**/*.md".to_string()],
        Arc::clone(&instructions),
        crate::config::shared_config(make_test_resolved_config(tmp.path())),
        empty_test_schema_cache(),
        None,
        Arc::new(crate::reindex::ReindexQueue::new()),
        empty_test_description_overlay(),
    )
    .unwrap();

    *instructions.write().unwrap() = "Updated with metadata".to_string();

    let info = server.get_info();
    assert_eq!(info.instructions.unwrap(), "Updated with metadata");
}

#[test]
fn test_get_info_recovers_from_poisoned_lock() {
    use std::panic;

    let lock = Arc::new(RwLock::new("valid instructions".to_string()));
    let lock_clone = Arc::clone(&lock);

    let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let _guard = lock_clone.write().unwrap();
        panic!("intentional panic to poison the lock");
    }));

    assert!(lock.read().is_err(), "lock should be poisoned");

    let recovered = lock
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();

    assert_eq!(recovered, "valid instructions");
}

#[test]
fn write_document_tool_description_covers_every_mode() {
    // The COMPILED description owns the API contract and nothing else: the
    // three content modes and how they combine. What belongs in this KB —
    // durable reference material, a shared scratchpad, anything else — is a
    // property of the deployment, not of the binary, so it lives in the
    // per-tool extension (`<kb>/meta/mcp/tools/write_document.md`) which is
    // appended at runtime. One binary serves knowledge bases whose scope
    // policies contradict each other; baking either one in here would ship a
    // description that lies to half of them.
    //
    // The compiled description no longer lives on the `#[tool(...)]`
    // attribute (see `descriptions.rs`) — every tool method carries a bare
    // `#[tool]`, and `KbSearchServer::tool_router().list_all()` therefore
    // returns `description: None` for all six. This asserts against the
    // actual runtime source of the description instead.
    let name = "write_document";
    let description =
        crate::descriptions::compose_tool_description(name, false, &Granularity::ALL, None)
            .unwrap_or_else(|| panic!("no compiled description for tool '{name}'"));

    for mode in [
        "content",
        "old_string",
        "new_path",
        "frontmatter_patch",
        "append",
    ] {
        assert!(
            description.contains(mode),
            "'{name}' description should document the '{mode}' mode: {description}"
        );
    }
}

#[test]
fn tool_router_tools_carry_no_compiled_description() {
    // Every `#[tool(...)]` attribute deliberately carries no `description`
    // (and no doc comment that would leak in as one via the macro's
    // fallback) — the description overlay in `list_tools`/`get_tool` is the
    // only source of a tool's description. This guards against a future
    // edit accidentally reintroducing one, which would silently bypass the
    // overlay for that tool (`list_tools` always applies the overlay
    // unconditionally, but a stray compiled description would still betray
    // the "compiled layer can't state per-KB policy" rule the moment
    // someone writes one back in).
    for tool in KbSearchServer::tool_router().list_all() {
        assert!(
            tool.description.is_none(),
            "tool '{}' unexpectedly carries a compiled description: {:?}",
            tool.name,
            tool.description
        );
    }
}

// --- description overlay: list_tools / get_tool ---------------------

#[test]
fn list_tools_returns_the_overlay_composed_descriptions() {
    // `list_tools`'s hand-written body (see the `ServerHandler` impl) is
    // exactly `tool_router().list_all().map(overlay_description)` behind a
    // thin async/RequestContext wrapper the framework provides — `Peer`'s
    // constructor needed to build that context is `pub(crate)` inside
    // `rmcp` and unreachable from here, so this exercises the same overlay
    // logic directly. The wrapper itself (real dispatch through the MCP
    // transport) is covered end-to-end by
    // `server::tests::tools_list_without_session_header_succeeds`
    // and its sibling assertions on `tools/list` descriptions.
    let default_config = make_test_resolved_config(&std::env::temp_dir());
    let effective_granularities = default_config.effective_granularities();
    let overlay =
        crate::descriptions::compose_tool_descriptions(None, false, &effective_granularities);
    let server = make_overlay_test_server(overlay.clone());

    let tools: Vec<Tool> = KbSearchServer::tool_router()
        .list_all()
        .into_iter()
        .map(|tool| server.overlay_description(tool))
        .collect();

    assert_eq!(tools.len(), crate::descriptions::TOOL_NAMES.len());
    for tool in &tools {
        let expected = overlay
            .get(tool.name.as_ref())
            .unwrap_or_else(|| panic!("no overlay entry for tool '{}'", tool.name));
        assert_eq!(
            tool.description.as_deref(),
            Some(expected.as_str()),
            "tool '{}' description mismatch",
            tool.name
        );
    }
}

#[test]
fn get_tool_returns_the_same_text_as_list_tools_would_for_that_tool() {
    let mut overlay = HashMap::new();
    overlay.insert(
        "search".to_string(),
        "Overlay search description.".to_string(),
    );
    let server = make_overlay_test_server(overlay);

    let tool = server
        .get_tool("search")
        .expect("search tool should be registered");
    assert_eq!(
        tool.description.as_deref(),
        Some("Overlay search description.")
    );

    // Same value the router itself produces, run through the same
    // `overlay_description` path `list_tools` uses.
    let router_tool = KbSearchServer::tool_router()
        .get("search")
        .cloned()
        .unwrap();
    let via_list_tools_path = server.overlay_description(router_tool);
    assert_eq!(tool.description, via_list_tools_path.description);
}

#[test]
fn get_tool_returns_none_for_an_unknown_name() {
    let server = make_overlay_test_server(HashMap::new());
    assert!(server.get_tool("not_a_real_tool").is_none());
}

#[test]
fn get_tool_returns_none_for_a_disabled_tool() {
    // Same contract as an unknown name (`get_tool_returns_none_for_an_unknown_name`
    // above) — `enabled_tool_router` hides a disabled name from `ToolRouter::get`
    // exactly the way a genuinely nonexistent one is hidden.
    let server = make_test_server_with_disabled_tools(vec!["write_document".to_string()]);
    assert!(
        server.get_tool("write_document").is_none(),
        "a disabled tool must not be returned by get_tool"
    );
    assert!(
        server.get_tool("search").is_some(),
        "an unrelated, still-enabled tool must be unaffected"
    );
}

#[test]
fn enabled_tool_router_omits_every_disabled_name_from_list_all() {
    let server = make_test_server_with_disabled_tools(vec![
        "write_document".to_string(),
        "delete_document".to_string(),
    ]);
    let names: Vec<String> = server
        .enabled_tool_router()
        .list_all()
        .into_iter()
        .map(|t| t.name.to_string())
        .collect();

    assert!(!names.contains(&"write_document".to_string()));
    assert!(!names.contains(&"delete_document".to_string()));
    assert_eq!(names.len(), crate::descriptions::TOOL_NAMES.len() - 2);
}

#[test]
fn enabled_tool_router_with_no_disabled_tools_matches_the_bare_tool_router() {
    let server = make_test_server_with_disabled_tools(Vec::new());
    let bare: Vec<String> = KbSearchServer::tool_router()
        .list_all()
        .into_iter()
        .map(|t| t.name.to_string())
        .collect();
    let enabled: Vec<String> = server
        .enabled_tool_router()
        .list_all()
        .into_iter()
        .map(|t| t.name.to_string())
        .collect();
    assert_eq!(bare, enabled);
}

// --- input-schema overlay: granularity enum / heading_prefix (#286) ---

#[test]
fn overlay_input_schema_sets_granularity_enum_to_the_effective_set() {
    let server = make_overlay_test_server_with_config(
        HashMap::new(),
        overlay_test_config(&[Granularity::Chunk, Granularity::Document], false),
    );
    let tool = KbSearchServer::tool_router()
        .get("search")
        .cloned()
        .unwrap();

    let overlaid = server.overlay_input_schema(tool);
    let properties = overlaid.input_schema["properties"].as_object().unwrap();
    let granularity = properties["granularity"].as_object().unwrap();
    let enum_values = granularity["enum"].as_array().unwrap();
    let names: Vec<&str> = enum_values.iter().filter_map(|v| v.as_str()).collect();

    assert_eq!(names, vec!["chunk", "document"]);
    assert!(
        !enum_values.iter().any(serde_json::Value::is_null),
        "an omitted granularity is expressed by leaving the optional property \
         out, not by a null enum member: {enum_values:?}"
    );
    // The property description is the effective-set text — never the
    // static doc comment, and never mentioning the disabled `section`.
    let description = granularity["description"].as_str().unwrap();
    assert_eq!(
        description,
        crate::descriptions::granularity_description(&[Granularity::Chunk, Granularity::Document])
    );
    assert!(!description.contains("section"), "got: {description}");
}

#[test]
fn search_schema_property_descriptions_carry_no_internal_notes() {
    // The router's raw schema (before any overlay) comes straight from
    // `SearchParams`' doc comments — they must stay caller-facing.
    let tool = KbSearchServer::tool_router()
        .get("search")
        .cloned()
        .unwrap();
    let schema = serde_json::Value::Object((*tool.input_schema).clone()).to_string();
    for leak in [
        "#286",
        "Step ",
        "default_granularity",
        "heading_prefixes",
        "chunking.",
    ] {
        assert!(
            !schema.contains(leak),
            "search's input schema leaks internal detail {leak:?}: {schema}"
        );
    }
}

#[test]
fn overlay_input_schema_removes_heading_prefix_when_heading_metadata_is_off() {
    let server = make_overlay_test_server_with_config(
        HashMap::new(),
        overlay_test_config(&Granularity::ALL, false),
    );
    let tool = KbSearchServer::tool_router()
        .get("search")
        .cloned()
        .unwrap();

    let overlaid = server.overlay_input_schema(tool);
    let properties = overlaid.input_schema["properties"].as_object().unwrap();
    assert!(
        !properties.contains_key("heading_prefix"),
        "heading_prefix must be removed from the schema when heading_metadata \
         is off: {properties:?}"
    );
}

#[test]
fn overlay_input_schema_keeps_heading_prefix_when_heading_metadata_is_on() {
    let server = make_overlay_test_server_with_config(
        HashMap::new(),
        overlay_test_config(&Granularity::ALL, true),
    );
    let tool = KbSearchServer::tool_router()
        .get("search")
        .cloned()
        .unwrap();

    let overlaid = server.overlay_input_schema(tool);
    let properties = overlaid.input_schema["properties"].as_object().unwrap();
    assert!(properties.contains_key("heading_prefix"));
}

#[test]
fn overlay_input_schema_query_description_follows_search_hybrid() {
    for (hybrid, expect_literal) in [(true, true), (false, false)] {
        let mut config = overlay_test_config(&Granularity::ALL, false);
        Arc::make_mut(&mut config).search.hybrid = hybrid;
        let server = make_overlay_test_server_with_config(HashMap::new(), config);
        let tool = KbSearchServer::tool_router()
            .get("search")
            .cloned()
            .unwrap();

        let overlaid = server.overlay_input_schema(tool);
        let description = overlaid.input_schema["properties"]["query"]["description"]
            .as_str()
            .unwrap();
        assert_eq!(description.contains("literal terms"), expect_literal);
        assert_eq!(description.starts_with("Semantic query."), !expect_literal);
    }
}

/// Every way a text can mention granularity `g` as a choice.
fn granularity_mentions(g: Granularity) -> Vec<String> {
    let g = g.as_str();
    vec![
        format!("`{g}`"),
        format!("'{g}'"),
        format!("\"{g}\""),
        format!("{g} granularity"),
        format!("{g}-granularity"),
    ]
}

/// Phrases that only make sense when a search without a query is possible.
const ENUMERATION_MENTIONS: &[&str] = &[
    "enumerat",
    "no query",
    "without a query",
    "without a `query`",
    "without `query`",
    "omit to list",
    "exhaustive listing",
    "listing without",
    "order_by",
    "descending",
];

#[tokio::test]
async fn search_tool_and_server_description_never_mention_unavailable_choices() {
    // Table test over all 7 non-empty granularity subsets x the
    // heading_metadata flag: the full serialized `search` tool definition
    // (description + input schema, as `get_tool`/`list_tools` serve it)
    // and the composed server description must not mention a disabled
    // granularity, nor no-query listing when no enabled granularity can
    // serve one (#286).
    let subsets: [&[Granularity]; 7] = [
        &[
            Granularity::Chunk,
            Granularity::Document,
            Granularity::Section,
        ],
        &[Granularity::Chunk, Granularity::Document],
        &[Granularity::Chunk, Granularity::Section],
        &[Granularity::Document, Granularity::Section],
        &[Granularity::Chunk],
        &[Granularity::Document],
        &[Granularity::Section],
    ];
    let data = tempfile::tempdir().unwrap();
    std::fs::create_dir(data.path().join("area")).unwrap();
    for subset in subsets {
        for heading_metadata in [false, true] {
            let config = overlay_test_config(subset, heading_metadata);
            let effective = config.effective_granularities();
            if effective.is_empty() {
                continue; // Rejected at config load.
            }
            let overlay = crate::descriptions::compose_tool_descriptions(None, true, &effective);
            let server = make_overlay_test_server_with_config(overlay, Arc::clone(&config));
            let tool = server.get_tool("search").unwrap();
            let tool_json = serde_json::to_string(&tool).unwrap();
            let instructions =
                crate::server::compose_server_instructions(&config, data.path()).await;
            assert!(
                instructions.contains("Top-level areas"),
                "the areas sentence must be exercised: {instructions}"
            );

            let label = format!("{subset:?} heading_metadata={heading_metadata}");
            for (surface, text) in [("tool", &tool_json), ("server", &instructions)] {
                // JSON-escaped quotes in the tool definition read as `\"`.
                let text = text.replace("\\\"", "\"").to_lowercase();
                if surface == "tool" && effective == Granularity::ALL {
                    // Guard against needles that can never match: with
                    // everything enabled, they must all be findable.
                    for g in Granularity::ALL {
                        assert!(text.contains(&format!("\"{}\"", g.as_str())), "{text}");
                        assert!(text.contains(&format!("`{}`", g.as_str())), "{text}");
                    }
                    assert!(text.contains("without `query`"), "{text}");
                    assert!(text.contains("order_by"), "{text}");
                }
                for g in Granularity::ALL
                    .into_iter()
                    .filter(|g| !effective.contains(g))
                {
                    for needle in granularity_mentions(g) {
                        assert!(
                            !text.contains(&needle),
                            "{label}: {surface} mentions disabled {needle}: {text}"
                        );
                    }
                }
                if !crate::descriptions::enumeration_available(&effective) {
                    for needle in ENUMERATION_MENTIONS {
                        assert!(
                            !text.contains(needle),
                            "{label}: {surface} mentions unavailable no-query listing \
                             ({needle}): {text}"
                        );
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn fields_and_explain_errors_only_suggest_enabled_granularities() {
    let tmp = tempfile::tempdir().unwrap();
    let chunk_only = schema_tool_server_with_granularities(&tmp, &[Granularity::Chunk]);
    let err = chunk_only
        .search(Parameters(SearchParams {
            query: Some("q".into()),
            fields: Some(vec!["title".into()]),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert!(!err.message.contains("'document'"), "{}", err.message);
    assert!(err.message.contains("fields"), "{}", err.message);

    let tmp = tempfile::tempdir().unwrap();
    let mut config = make_test_resolved_config(tmp.path());
    Arc::make_mut(&mut config).search.granularities =
        vec![Granularity::Document, Granularity::Section];
    Arc::make_mut(&mut config).chunking.heading_metadata = true;
    let server = make_write_test_server(&tmp, &["**/*.md".to_string()], config);
    for granularity in ["document", "section"] {
        let err = server
            .search(Parameters(SearchParams {
                query: Some("q".into()),
                granularity: Some(granularity.into()),
                explain: Some(true),
                ..Default::default()
            }))
            .await
            .unwrap_err();
        assert!(!err.message.contains("'chunk'"), "{}", err.message);
        assert!(err.message.contains("explain"), "{}", err.message);
    }
}

#[test]
fn overlay_input_schema_is_a_noop_for_non_search_tools() {
    let server = make_overlay_test_server_with_config(
        HashMap::new(),
        overlay_test_config(&[Granularity::Document], false),
    );
    let tool = KbSearchServer::tool_router()
        .get("get_document")
        .cloned()
        .unwrap();
    let original_schema = Arc::clone(&tool.input_schema);

    let overlaid = server.overlay_input_schema(tool);
    assert!(
        Arc::ptr_eq(&original_schema, &overlaid.input_schema),
        "a non-search tool's input_schema must be returned unchanged, not even \
         cloned"
    );
}

#[test]
fn list_tools_schema_enum_and_heading_prefix_removal_follow_config() {
    // Same end-to-end intent as `list_tools_returns_the_overlay_composed_descriptions`
    // above: `list_tools`'s hand-written body cannot be exercised directly
    // here (see that test's comment on `Peer`'s constructor being
    // unreachable), so this drives the exact two-step overlay
    // (`overlay_description` then `overlay_input_schema`) `list_tools`
    // applies to every router `Tool`, for a restricted config.
    let config = overlay_test_config(&[Granularity::Document], false);
    let server = make_overlay_test_server_with_config(HashMap::new(), Arc::clone(&config));

    let tools: Vec<Tool> = KbSearchServer::tool_router()
        .list_all()
        .into_iter()
        .map(|tool| server.overlay_description(tool))
        .map(|tool| server.overlay_input_schema(tool))
        .collect();

    let search_tool = tools
        .iter()
        .find(|t| t.name.as_ref() == "search")
        .expect("search tool should be registered");
    let properties = search_tool.input_schema["properties"].as_object().unwrap();
    let enum_values = properties["granularity"]["enum"].as_array().unwrap();
    let names: Vec<&str> = enum_values.iter().filter_map(|v| v.as_str()).collect();
    assert_eq!(names, vec!["document"]);
    assert!(!properties.contains_key("heading_prefix"));

    // Every OTHER tool's schema must be untouched by the overlay.
    for tool in &tools {
        if tool.name.as_ref() != "search" {
            let router_schema = KbSearchServer::tool_router()
                .get(tool.name.as_ref())
                .unwrap()
                .input_schema
                .clone();
            assert_eq!(
                tool.input_schema, router_schema,
                "'{}' schema must be unaffected by the search-only overlay",
                tool.name
            );
        }
    }
}

// --- tool_schema::self_contained: llama.cpp-backed clients (#288) ------

/// Recursively asserts that `schema` (any JSON Schema position — the root,
/// or something reached by walking into it) is free of `$ref`, `$defs`/
/// `definitions`, and a boolean value standing in for a subschema.
/// Reimplements the walk independently of `tool_schema`'s own keyword
/// list rather than calling it, so this is a real check on the shape
/// `list_tools`/`get_tool` actually produce, not a tautology against the
/// code under test.
fn assert_schema_is_self_contained(schema: &serde_json::Value, tool_name: &str) {
    match schema {
        serde_json::Value::Bool(_) => {
            panic!(
                "tool '{tool_name}' has a bare boolean subschema after \
                 tool_schema::self_contained: {schema}"
            );
        }
        serde_json::Value::Object(obj) => {
            for key in ["$ref", "$defs", "definitions"] {
                assert!(
                    !obj.contains_key(key),
                    "tool '{tool_name}' still has '{key}' after \
                     tool_schema::self_contained: {schema}"
                );
            }
            for key in [
                "items",
                "unevaluatedItems",
                "not",
                "if",
                "then",
                "else",
                "contains",
                "propertyNames",
            ] {
                if let Some(v) = obj.get(key) {
                    assert_schema_is_self_contained(v, tool_name);
                }
            }
            for key in ["properties", "patternProperties", "dependentSchemas"] {
                if let Some(serde_json::Value::Object(map)) = obj.get(key) {
                    for v in map.values() {
                        assert_schema_is_self_contained(v, tool_name);
                    }
                }
            }
            for key in ["prefixItems", "anyOf", "oneOf", "allOf"] {
                if let Some(serde_json::Value::Array(items)) = obj.get(key) {
                    for v in items {
                        assert_schema_is_self_contained(v, tool_name);
                    }
                }
            }
            // Left untouched deliberately when boolean (the conventional
            // `deny_unknown_fields` form) — only recurse when it is
            // itself a nested schema object.
            for key in ["additionalProperties", "unevaluatedProperties"] {
                if let Some(v @ serde_json::Value::Object(_)) = obj.get(key) {
                    assert_schema_is_self_contained(v, tool_name);
                }
            }
        }
        _ => {}
    }
}

/// Keyword-agnostic companion to `assert_schema_is_self_contained`: finds
/// `key` as an object key at any depth, schema position or not, so a ref
/// under a keyword neither walker lists still fails the test. Safe because
/// no tool schema carries a data value (`default`/`examples`) with such a key.
fn has_key_anywhere(value: &serde_json::Value, key: &str) -> bool {
    match value {
        serde_json::Value::Object(obj) => {
            obj.contains_key(key) || obj.values().any(|v| has_key_anywhere(v, key))
        }
        serde_json::Value::Array(items) => items.iter().any(|v| has_key_anywhere(v, key)),
        _ => false,
    }
}

#[test]
fn list_tools_schemas_are_self_contained_for_llama_cpp_backed_clients() {
    // #288: llama-server turns every tool's `inputSchema` into a GBNF
    // grammar at `tools/list` time and fails the WHOLE request with HTTP
    // 400 if any one tool's schema doesn't convert ($ref/$defs, or a
    // boolean subschema) — this broke Crush and OpenCode against
    // `search`, `write_document` and `update_schema`. Walks every tool
    // the real `list_tools` chain produces (`overlay_description` then
    // `overlay_input_schema` then `tool_schema::self_contained`), with
    // `chunking.heading_metadata` both on and off, so a new tool or a
    // newly-recursive parameter type can't regress this silently.
    for heading_metadata in [false, true] {
        let config = overlay_test_config(
            &[
                Granularity::Chunk,
                Granularity::Document,
                Granularity::Section,
            ],
            heading_metadata,
        );
        let server = make_overlay_test_server_with_config(HashMap::new(), config);

        // Through the production `get_tool`, not a hand-rebuilt chain, so
        // dropping the `self_contained` step there fails this test;
        // `get_tool_input_schema_matches_the_list_tools_overlay_path`
        // pins `list_tools` to the same chain.
        let tools: Vec<Tool> = KbSearchServer::tool_router()
            .list_all()
            .into_iter()
            .map(|tool| {
                server
                    .get_tool(tool.name.as_ref())
                    .expect("every router tool resolves through get_tool")
            })
            .collect();

        assert_eq!(tools.len(), crate::descriptions::TOOL_NAMES.len());
        for tool in &tools {
            let schema = serde_json::Value::Object((*tool.input_schema).clone());
            assert_schema_is_self_contained(&schema, tool.name.as_ref());
            for key in ["$ref", "$defs", "definitions"] {
                assert!(
                    !has_key_anywhere(&schema, key),
                    "tool '{}' has '{key}' somewhere in its schema: {schema}",
                    tool.name
                );
            }
        }
    }
}

#[test]
fn self_contained_write_document_frontmatter_patch_keeps_its_shape() {
    // Shape-preservation regression for #288's cycle-cut/boolean-replacement:
    // `write_document`'s `frontmatter_patch` items must still document
    // `operation`/`field`/`value`/`values`, and `values`'s element schema
    // must degrade from the bare `true` schemars emits for
    // `Vec<serde_json::Value>` to the equivalent `{}` rather than
    // disappearing.
    let config = overlay_test_config(&[Granularity::Chunk], false);
    let server = make_overlay_test_server_with_config(HashMap::new(), config);
    let tool = server
        .get_tool("write_document")
        .expect("write_document tool should be registered");

    let op_schema = &tool.input_schema["properties"]["frontmatter_patch"]["items"];
    let op_properties = &op_schema["properties"];
    for key in ["operation", "field", "value", "values"] {
        assert!(
            !op_properties[key].is_null(),
            "frontmatter_patch op is missing '{key}': {op_schema}"
        );
    }
    assert_eq!(
        op_properties["values"]["items"],
        serde_json::json!({}),
        "a bare `true` items schema must become `{{}}`, not disappear: {op_schema}"
    );
}

#[test]
fn batch_entries_advertise_the_same_frontmatter_patch_ops_as_the_top_level() {
    // Both places use `FrontmatterPatchOp`, so what a batch entry's ops
    // advertise cannot drift from the single-document ones.
    let config = overlay_test_config(&[Granularity::Chunk], false);
    let server = make_overlay_test_server_with_config(HashMap::new(), config);
    let tool = server
        .get_tool("write_document")
        .expect("write_document tool should be registered");

    let properties = &tool.input_schema["properties"];
    let top = &properties["frontmatter_patch"]["items"];
    let batch = &properties["documents"]["items"]["properties"]["frontmatter_patch"]["items"];
    assert!(
        top["properties"]["field"].is_object(),
        "the top-level op schema is missing 'field': {top}"
    );
    assert_eq!(batch, top);
}

#[test]
fn self_contained_update_schema_definition_keeps_its_named_properties() {
    // Shape-preservation regression: `update_schema`'s `definition` must
    // still advertise every `RawFieldDef` property (including the
    // recursive `fields` map) at the top level, with
    // `additionalProperties: false` preserved, even though the def is no
    // longer reached through `$ref`.
    let config = overlay_test_config(&[Granularity::Chunk], false);
    let server = make_overlay_test_server_with_config(HashMap::new(), config);
    let tool = server
        .get_tool("update_schema")
        .expect("update_schema tool should be registered");

    // `tool_schema::compact` unwraps the optional property's null arm, so
    // the object schema is the property itself.
    let object_branch = &tool.input_schema["properties"]["definition"];
    assert!(object_branch.get("anyOf").is_none(), "{object_branch}");

    assert_eq!(object_branch["type"], serde_json::json!("object"));
    assert!(
        object_branch["properties"].get("extend").is_none(),
        "the deprecated `extend` must not be advertised: {object_branch}"
    );
    assert!(
        object_branch["properties"]["values"]["description"]
            .as_str()
            .is_some_and(|d| d.contains("$values")),
        "{object_branch}"
    );
    assert!(
        object_branch["properties"]["type"]["enum"].is_array(),
        "the field type is a flat enum: {object_branch}"
    );
    assert_eq!(
        object_branch["additionalProperties"],
        serde_json::json!(false)
    );
    let keys = [
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
    ];
    for key in keys {
        assert!(
            !object_branch["properties"][key].is_null(),
            "definition schema is missing documented key '{key}': {object_branch}"
        );
    }

    // `ops` (#348): an array of op objects whose `operation` is a flat string
    // enum (not a `oneOf` of described consts) and whose `definition` is the same
    // inlined, closed `RawFieldDef` object.
    let ops = &tool.input_schema["properties"]["ops"];
    assert_eq!(ops["type"], serde_json::json!("array"), "{ops}");
    let op = &ops["items"];
    assert_eq!(op["type"], serde_json::json!("object"), "{op}");
    assert_eq!(op["additionalProperties"], serde_json::json!(false), "{op}");
    assert_eq!(
        op["required"],
        serde_json::json!(["operation", "field"]),
        "{op}"
    );
    assert_eq!(
        op["properties"]["operation"]["enum"],
        serde_json::json!([
            "add_values",
            "remove_values",
            "set_field",
            "remove_field",
            "patch_field"
        ]),
        "{op}"
    );
    assert_eq!(op["properties"]["operation"]["type"], "string", "{op}");
    assert!(op["properties"]["operation"].get("oneOf").is_none(), "{op}");
    let op_definition = &op["properties"]["definition"];
    assert_eq!(
        op_definition["additionalProperties"],
        serde_json::json!(false)
    );
    for key in keys {
        assert!(
            !op_definition["properties"][key].is_null(),
            "an op's definition is missing '{key}': {op_definition}"
        );
    }
    // `reinherit` is how a schema-enforcing client re-inherits a key: compaction
    // strips the `null` arm of every definition property, so a literal `null` is
    // not something such a client can send. It must survive as a string array.
    let reinherit = &op["properties"]["reinherit"];
    assert_eq!(reinherit["type"], serde_json::json!("array"), "{op}");
    assert_eq!(
        reinherit["items"]["type"],
        serde_json::json!("string"),
        "{op}"
    );
    assert!(reinherit["description"].is_string(), "{op}");
    // The flat parameters are still advertised, marked deprecated.
    for key in ["operation", "field", "values", "definition"] {
        assert!(
            tool.input_schema["properties"][key]["description"]
                .as_str()
                .is_some_and(|d| d.starts_with("Deprecated")),
            "{key}: {}",
            tool.input_schema["properties"][key]
        );
    }
    assert!(
        tool.input_schema
            .get("required")
            .and_then(|r| r.as_array())
            .is_none_or(|r| r.is_empty()),
        "no parameter is required any more: {}",
        serde_json::Value::Object((*tool.input_schema).clone())
    );
}

#[test]
fn self_contained_search_filters_keeps_its_typed_properties() {
    // Shape-preservation regression: `search`'s `filters` must still
    // advertise as a typed object with a real (non-`true`)
    // `additionalProperties` condition schema once `SearchFilters` is
    // inlined instead of `$ref`ed.
    let config = overlay_test_config(&[Granularity::Chunk], false);
    let server = make_overlay_test_server_with_config(HashMap::new(), config);
    let tool = server
        .get_tool("search")
        .expect("search tool should be registered");

    let object_branch = &tool.input_schema["properties"]["filters"];
    assert!(object_branch.get("anyOf").is_none(), "{object_branch}");

    assert_eq!(object_branch["type"], serde_json::json!("object"));
    assert_ne!(
        object_branch["additionalProperties"],
        serde_json::json!(true),
        "a bare `additionalProperties: true` tells a client nothing about a \
         condition's shape: {object_branch}"
    );
}

#[test]
fn get_tool_input_schema_matches_the_list_tools_overlay_path() {
    // `list_tools`'s hand-written body applies `overlay_description`,
    // then `overlay_input_schema` (#286), then `tool_schema::self_contained`
    // (#288) to each router `Tool`; `get_tool` must apply the exact same
    // three steps for a single tool, or a client calling `tools/get`
    // could see a schema `tools/list` would never have produced — the
    // schema counterpart to
    // `get_tool_returns_the_same_text_as_list_tools_would_for_that_tool`
    // above.
    let config = overlay_test_config(&[Granularity::Chunk, Granularity::Document], false);
    let server = make_overlay_test_server_with_config(HashMap::new(), config);

    let tool = server
        .get_tool("search")
        .expect("search tool should be registered");

    let router_tool = KbSearchServer::tool_router()
        .get("search")
        .cloned()
        .unwrap();
    let via_list_tools_path = tool_schema::self_contained(
        server.overlay_input_schema(server.overlay_description(router_tool)),
    );

    assert_eq!(tool.input_schema, via_list_tools_path.input_schema);
}

#[test]
fn description_overlay_recovers_from_a_poisoned_lock() {
    use std::panic;

    let mut overlay = HashMap::new();
    overlay.insert("search".to_string(), "Before poison.".to_string());
    let server = make_overlay_test_server(overlay);

    let overlay_lock = Arc::clone(&server.description_overlay);
    let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let _guard = overlay_lock.write().unwrap();
        panic!("intentional panic to poison the description overlay lock");
    }));
    assert!(
        overlay_lock.read().is_err(),
        "overlay lock should be poisoned"
    );

    // get_tool must not panic, and must recover the last-good value.
    let tool = server
        .get_tool("search")
        .expect("search tool should still be registered after recovery");
    assert_eq!(tool.description.as_deref(), Some("Before poison."));
}

/// Doc comments on schemars-derived parameter types become the served
/// input schema; none may leak Rust paths, issue numbers or notes aimed at
/// this codebase's maintainers.
#[test]
fn served_tool_schemas_carry_no_internal_references() {
    let server = make_overlay_test_server_with_config(
        HashMap::new(),
        overlay_test_config(&Granularity::ALL, true),
    );
    let leaks = |d: &str| {
        ["::", "doc comment", "codebase"]
            .iter()
            .any(|n| d.contains(n))
            || d.as_bytes()
                .windows(2)
                .any(|w| w[0] == b'#' && w[1].is_ascii_digit())
    };
    for name in crate::descriptions::TOOL_NAMES {
        let tool = server.get_tool(name).unwrap();
        let schema = serde_json::Value::Object((*tool.input_schema).clone());
        let mut stack = vec![&schema];
        while let Some(value) = stack.pop() {
            match value {
                serde_json::Value::Object(map) => {
                    if let Some(serde_json::Value::String(d)) = map.get("description") {
                        assert!(
                            !leaks(d),
                            "`{name}` schema description leaks an internal reference: {d}"
                        );
                    }
                    stack.extend(map.values());
                }
                serde_json::Value::Array(items) => stack.extend(items),
                _ => {}
            }
        }
    }
}

// -- batched update_schema ops and explicit cascade semantics (#348) --------
