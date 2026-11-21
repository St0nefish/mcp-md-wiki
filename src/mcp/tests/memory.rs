// The memory tool group through the real handlers: a git-backed server with the
// group on, no Qdrant or embedder (the duplicate check is exercised by its pure
// parts and by `on_duplicate: keep_both` here).
use rmcp::handler::server::wrapper::Parameters;

use crate::mcp::memory::*;
use crate::mcp::schema_tools::*;
use crate::mcp::test_support::*;
use crate::mcp::*;
use crate::policy::scoped_principal_for_test;

fn create(title: &str, body: &str) -> MemoryCreateParams {
    MemoryCreateParams {
        title: title.into(),
        body: body.into(),
        ty: "fact".into(),
        on_duplicate: Some("keep_both".into()),
        ..Default::default()
    }
}

async fn created(server: &KbSearchServer, p: MemoryCreateParams) -> serde_json::Value {
    let result = server
        .memory_create(Parameters(p))
        .await
        .expect("memory_create");
    result.structured_content.expect("structured result")
}

fn git_log(work: &tempfile::TempDir) -> Vec<String> {
    let out = std::process::Command::new("git")
        .args(["log", "--format=%s"])
        .current_dir(work.path())
        .env("GIT_CEILING_DIRECTORIES", work.path().parent().unwrap())
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_string)
        .collect()
}

#[test]
fn the_memory_tools_exist_only_while_the_group_is_on() {
    let off = make_test_server_with_disabled_tools(Vec::new());
    let listed: Vec<String> = off
        .enabled_tool_router()
        .list_all()
        .into_iter()
        .map(|t| t.name.to_string())
        .collect();
    assert_eq!(
        listed.len(),
        crate::descriptions::TOOL_NAMES.len(),
        "{listed:?}"
    );
    assert!(
        listed.iter().all(|n| !n.starts_with("memory_")),
        "{listed:?}"
    );

    let on = memory_harness();
    let listed: Vec<String> = on
        .server
        .enabled_tool_router()
        .list_all()
        .into_iter()
        .map(|t| t.name.to_string())
        .collect();
    assert_eq!(listed.len(), crate::descriptions::all_tool_names().count());
    for tool in crate::descriptions::MEMORY_TOOL_NAMES {
        assert!(
            listed.iter().any(|n| n == tool),
            "{tool} missing: {listed:?}"
        );
    }
}

#[tokio::test]
async fn a_memory_call_is_refused_while_the_group_is_off() {
    let off = make_test_server_with_disabled_tools(Vec::new());
    let err = off
        .memory_create(Parameters(create("t", "b")))
        .await
        .unwrap_err();
    assert_eq!(err.message, "tool not found");
}

#[tokio::test]
async fn a_created_memory_is_a_valid_document_with_provenance_and_is_readable_by_id() {
    let h = memory_harness();
    let result = created(
        &h.server,
        MemoryCreateParams {
            tags: Some(vec!["editor".into()]),
            source_url: Some("https://example.com/post".into()),
            ..create("Prefers tabs", "The user prefers tabs over spaces.")
        },
    )
    .await;
    assert_eq!(result["created"], true, "{result}");
    assert_eq!(result["status"], "proposed");
    let id = result["id"].as_str().unwrap().to_string();
    let path = result["path"].as_str().unwrap().to_string();
    assert!(path.starts_with("memory/general/prefers-tabs-"), "{path}");
    assert!(h.work.path().join(&path).exists());

    // Read back before any reindex: the id resolves from disk.
    let got = h
        .server
        .memory_get(Parameters(MemoryGetParams { id: id.clone() }))
        .await
        .unwrap()
        .structured_content
        .unwrap();
    assert_eq!(got["id"], serde_json::json!(id));
    assert_eq!(got["title"], "Prefers tabs");
    assert_eq!(got["type"], "fact");
    assert_eq!(got["status"], "proposed");
    assert_eq!(got["tags"], serde_json::json!(["editor"]));
    assert_eq!(got["source_url"], "https://example.com/post");
    assert!(got["body"].as_str().unwrap().contains("tabs over spaces"));
    assert!(got["version"].as_str().is_some_and(|v| !v.is_empty()));
    assert!(got["source_client"].as_str().is_some_and(|s| !s.is_empty()));

    assert!(
        git_log(&h.work)
            .iter()
            .any(|s| s == &format!("memory: add {id}")),
        "{:?}",
        git_log(&h.work)
    );
}

#[tokio::test]
async fn bad_input_is_refused_before_anything_is_written() {
    let h = memory_harness();
    for (p, needle) in [
        (
            MemoryCreateParams {
                ty: "rumour".into(),
                ..create("t", "b")
            },
            "type must be one of",
        ),
        (create("", "b"), "title"),
        (create("t", &"x".repeat(5000)), "body is over"),
        (
            MemoryCreateParams {
                area: Some("../etc".into()),
                ..create("t", "b")
            },
            "area",
        ),
        (
            MemoryCreateParams {
                on_duplicate: Some("shrug".into()),
                ..create("t", "b")
            },
            "on_duplicate",
        ),
        (
            MemoryCreateParams {
                supersedes: Some("mem-000000000000".into()),
                ..create("t", "b")
            },
            "no memory with that id",
        ),
    ] {
        let err = h.server.memory_create(Parameters(p)).await.unwrap_err();
        assert!(err.message.contains(needle), "{needle}: {}", err.message);
    }
    assert!(
        !h.work.path().join("memory").exists(),
        "nothing was written"
    );
}

#[tokio::test]
async fn an_update_is_a_new_revision_and_a_noop_update_commits_nothing() {
    let h = memory_harness();
    let made = created(&h.server, create("Editor", "Uses vim.")).await;
    let id = made["id"].as_str().unwrap().to_string();

    let updated = h
        .server
        .memory_update(Parameters(MemoryUpdateParams {
            id: id.clone(),
            body: Some("Uses neovim.".into()),
            ..Default::default()
        }))
        .await
        .unwrap()
        .structured_content
        .unwrap();
    assert_eq!(updated["changed"], true, "{updated}");
    let again = h
        .server
        .memory_update(Parameters(MemoryUpdateParams {
            id: id.clone(),
            body: Some("Uses neovim.".into()),
            ..Default::default()
        }))
        .await
        .unwrap()
        .structured_content
        .unwrap();
    assert_eq!(again["changed"], false, "{again}");

    let log = git_log(&h.work);
    let updates = log
        .iter()
        .filter(|s| *s == &format!("memory: update {id}"))
        .count();
    assert_eq!(updates, 1, "{log:?}");

    let got = h
        .server
        .memory_get(Parameters(MemoryGetParams { id }))
        .await
        .unwrap()
        .structured_content
        .unwrap();
    assert!(got["body"].as_str().unwrap().contains("neovim"));

    let nothing = h
        .server
        .memory_update(Parameters(MemoryUpdateParams {
            id: got["id"].as_str().unwrap().into(),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert!(nothing.message.contains("nothing to change"));
}

#[tokio::test]
async fn forgetting_keeps_the_memory_readable_but_closes_it_to_updates() {
    let h = memory_harness();
    let id = created(&h.server, create("Old", "Outdated fact.")).await["id"]
        .as_str()
        .unwrap()
        .to_string();
    let forgotten = h
        .server
        .memory_forget(Parameters(MemoryForgetParams { id: id.clone() }))
        .await
        .unwrap()
        .structured_content
        .unwrap();
    assert_eq!(forgotten["status"], "forgotten");
    assert_eq!(forgotten["changed"], true);

    // Idempotent.
    let again = h
        .server
        .memory_forget(Parameters(MemoryForgetParams { id: id.clone() }))
        .await
        .unwrap()
        .structured_content
        .unwrap();
    assert_eq!(again["changed"], false);

    let got = h
        .server
        .memory_get(Parameters(MemoryGetParams { id: id.clone() }))
        .await
        .unwrap()
        .structured_content
        .unwrap();
    assert_eq!(got["status"], "forgotten");

    let err = h
        .server
        .memory_update(Parameters(MemoryUpdateParams {
            id,
            body: Some("Revived.".into()),
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert!(err.message.contains("cannot be updated"), "{}", err.message);
}

#[tokio::test]
async fn superseding_replaces_the_old_memory_in_one_commit_and_links_the_chain() {
    let h = memory_harness();
    let old_id = created(&h.server, create("Region", "Deploys to us-east.")).await["id"]
        .as_str()
        .unwrap()
        .to_string();
    let before = git_log(&h.work).len();

    let new = created(
        &h.server,
        MemoryCreateParams {
            supersedes: Some(old_id.clone()),
            on_duplicate: None,
            ..create("Region", "Deploys to eu-west now.")
        },
    )
    .await;
    assert_eq!(new["created"], true, "{new}");
    assert_eq!(new["superseded"], serde_json::json!(old_id));
    let new_id = new["id"].as_str().unwrap().to_string();
    assert_eq!(
        git_log(&h.work).len(),
        before + 1,
        "the new revision and the old one's status are one commit"
    );

    let old = h
        .server
        .memory_get(Parameters(MemoryGetParams { id: old_id.clone() }))
        .await
        .unwrap()
        .structured_content
        .unwrap();
    assert_eq!(old["status"], "superseded");
    let newer = h
        .server
        .memory_get(Parameters(MemoryGetParams { id: new_id.clone() }))
        .await
        .unwrap()
        .structured_content
        .unwrap();
    assert_eq!(newer["supersedes"], serde_json::json!(old_id));
    assert_eq!(
        newer["supersedes_chain"][0]["id"],
        serde_json::json!(old_id)
    );

    // A superseded memory cannot be superseded again.
    let err = h
        .server
        .memory_create(Parameters(MemoryCreateParams {
            supersedes: Some(old_id),
            on_duplicate: None,
            ..create("Region", "Again.")
        }))
        .await
        .unwrap_err();
    assert!(
        err.message.contains("already superseded"),
        "{}",
        err.message
    );
}

#[tokio::test]
async fn ids_that_are_not_memory_ids_find_nothing() {
    let h = memory_harness();
    created(&h.server, create("T", "B")).await;
    for id in [
        "",
        "../etc/passwd",
        "mem-",
        "mem-ZZZZZZZZZZZZ",
        "memory/general/x.md",
    ] {
        let err = h
            .server
            .memory_get(Parameters(MemoryGetParams { id: id.into() }))
            .await
            .unwrap_err();
        assert_eq!(err.message, "no memory with that id", "{id:?}");
    }
}

#[tokio::test]
async fn a_scoped_principal_writes_only_where_its_policy_says() {
    let h = memory_harness();
    let tools = ["*"];
    let writer = scoped_principal_for_test(&tools, &["memory"], &["memory/general"]);
    let reader = scoped_principal_for_test(&tools, &["memory"], &[]);

    // In its area: allowed, and the author is recorded as that principal.
    let ok = CURRENT_PRINCIPAL
        .scope(
            writer.clone(),
            h.server.memory_create(Parameters(create("Mine", "Fine."))),
        )
        .await
        .unwrap()
        .structured_content
        .unwrap();
    assert_eq!(ok["created"], true, "{ok}");
    let got = h
        .server
        .memory_get(Parameters(MemoryGetParams {
            id: ok["id"].as_str().unwrap().into(),
        }))
        .await
        .unwrap()
        .structured_content
        .unwrap();
    assert_eq!(got["source_client"], "scoped");

    // Another area, and a reader with no write grant at all: the same refusal.
    for (who, area) in [(writer, Some("other")), (reader, None)] {
        let err = CURRENT_PRINCIPAL
            .scope(
                who,
                h.server.memory_create(Parameters(MemoryCreateParams {
                    area: area.map(str::to_string),
                    ..create("Nope", "Not here.")
                })),
            )
            .await
            .unwrap_err();
        assert_eq!(err.message, "not permitted", "{area:?}");
    }
    assert!(!h.work.path().join("memory/other").exists());

    // A reader sees its area's memories but cannot change one.
    let id = ok["id"].as_str().unwrap().to_string();
    let reader = scoped_principal_for_test(&tools, &["memory"], &[]);
    for outcome in [
        CURRENT_PRINCIPAL
            .scope(
                reader.clone(),
                h.server
                    .memory_forget(Parameters(MemoryForgetParams { id: id.clone() })),
            )
            .await
            .unwrap_err(),
        CURRENT_PRINCIPAL
            .scope(
                reader.clone(),
                h.server.memory_update(Parameters(MemoryUpdateParams {
                    id: id.clone(),
                    body: Some("Changed.".into()),
                    ..Default::default()
                })),
            )
            .await
            .unwrap_err(),
    ] {
        assert_eq!(outcome.message, "not permitted");
    }
    assert!(
        CURRENT_PRINCIPAL
            .scope(
                reader,
                h.server.memory_get(Parameters(MemoryGetParams { id }))
            )
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn a_principal_that_cannot_read_the_memory_area_finds_no_memory_by_id() {
    let h = memory_harness();
    let id = created(&h.server, create("Secret", "Hidden.")).await["id"]
        .as_str()
        .unwrap()
        .to_string();
    let outsider = scoped_principal_for_test(&["*"], &["notes"], &[]);
    let err = CURRENT_PRINCIPAL
        .scope(
            outsider,
            h.server.memory_get(Parameters(MemoryGetParams { id })),
        )
        .await
        .unwrap_err();
    assert_eq!(err.message, "no memory with that id");
}

#[tokio::test]
async fn update_schema_refuses_the_memory_area_and_get_schema_reports_the_contract() {
    let h = memory_harness();
    let err = h
        .server
        .update_schema(Parameters(UpdateSchemaParams {
            ops: None,
            path: Some("memory/general".into()),
            operation: "add_values".into(),
            field: "tags".into(),
            values: Some(vec!["x".into()]),
            definition: None,
            dry_run: Some(true),
            force: None,
            acknowledge_root_change: None,
        }))
        .await
        .unwrap_err();
    assert!(
        err.message.contains("built into the server"),
        "{}",
        err.message
    );

    let schema = h
        .server
        .get_schema(Parameters(GetSchemaParams {
            path: Some("memory/general".into()),
            ..Default::default()
        }))
        .await
        .unwrap()
        .structured_content
        .unwrap();
    assert_eq!(
        schema["memory_contract_version"],
        serde_json::json!(crate::memory::logic::CONTRACT_VERSION)
    );
    assert!(schema["fields"]["status"]["values"].is_array(), "{schema}");
}

fn neighbour(path: &str, score: f32) -> crate::similarity::Neighbour {
    crate::similarity::Neighbour {
        path: path.to_string(),
        score,
    }
}

/// Index `path` as the search backend would have: the metadata the duplicate check
/// reads back.
async fn index(server: &KbSearchServer, path: &str, id: &str, status: &str) {
    seed_document(
        server,
        path,
        serde_json::json!({"id": id, "title": "Existing", "type": "fact", "status": status}),
    )
    .await;
}

#[tokio::test]
async fn a_near_duplicate_is_refused_by_default_with_the_matches_listed() {
    let h = memory_harness();
    let first = created(&h.server, create("Region", "Deploys to us-east.")).await;
    let (old_id, old_path) = (
        first["id"].as_str().unwrap().to_string(),
        first["path"].as_str().unwrap().to_string(),
    );
    index(&h.server, &old_path, &old_id, "confirmed").await;
    let before = git_log(&h.work).len();

    *h.server.test_duplicate_hits.lock().unwrap() = Some(vec![neighbour(&old_path, 0.93)]);
    let refused = created(
        &h.server,
        MemoryCreateParams {
            on_duplicate: None,
            ..create("Region", "We deploy to us-east.")
        },
    )
    .await;
    assert_eq!(refused["created"], false, "{refused}");
    assert_eq!(refused["reason"], "duplicate");
    assert_eq!(refused["duplicates"][0]["id"], serde_json::json!(old_id));
    assert_eq!(
        refused["duplicates"][0]["path"],
        serde_json::json!(old_path)
    );
    assert!(refused["duplicates"][0]["score"].as_f64().unwrap() > 0.9);
    assert_eq!(git_log(&h.work).len(), before, "nothing was written");
}

#[tokio::test]
async fn on_duplicate_supersede_replaces_the_closest_live_memory() {
    let h = memory_harness();
    let first = created(&h.server, create("Region", "Deploys to us-east.")).await;
    let (old_id, old_path) = (
        first["id"].as_str().unwrap().to_string(),
        first["path"].as_str().unwrap().to_string(),
    );
    index(&h.server, &old_path, &old_id, "proposed").await;

    *h.server.test_duplicate_hits.lock().unwrap() = Some(vec![neighbour(&old_path, 0.9)]);
    let replaced = created(
        &h.server,
        MemoryCreateParams {
            on_duplicate: Some("supersede".into()),
            ..create("Region", "We deploy to us-east.")
        },
    )
    .await;
    assert_eq!(replaced["created"], true, "{replaced}");
    assert_eq!(replaced["superseded"], serde_json::json!(old_id));
    let old = h
        .server
        .memory_get(Parameters(MemoryGetParams { id: old_id }))
        .await
        .unwrap()
        .structured_content
        .unwrap();
    assert_eq!(old["status"], "superseded");
}

#[tokio::test]
async fn neighbours_below_the_threshold_or_no_longer_live_are_not_duplicates() {
    let h = memory_harness();
    let first = created(&h.server, create("Old", "Outdated.")).await;
    let (id, path) = (
        first["id"].as_str().unwrap().to_string(),
        first["path"].as_str().unwrap().to_string(),
    );
    // Forgotten in the index: a neighbour, but history.
    index(&h.server, &path, &id, "forgotten").await;
    *h.server.test_duplicate_hits.lock().unwrap() = Some(vec![neighbour(&path, 0.99)]);
    let r = created(
        &h.server,
        MemoryCreateParams {
            on_duplicate: None,
            ..create("New", "Fresh.")
        },
    )
    .await;
    assert_eq!(
        r["created"], true,
        "a forgotten memory is not a duplicate: {r}"
    );

    // Live but under the 0.85 default threshold.
    let second = created(&h.server, create("Live", "Live one.")).await;
    index(
        &h.server,
        second["path"].as_str().unwrap(),
        second["id"].as_str().unwrap(),
        "confirmed",
    )
    .await;
    *h.server.test_duplicate_hits.lock().unwrap() =
        Some(vec![neighbour(second["path"].as_str().unwrap(), 0.6)]);
    let r = created(
        &h.server,
        MemoryCreateParams {
            on_duplicate: None,
            ..create("Other", "Something else entirely.")
        },
    )
    .await;
    assert_eq!(r["created"], true, "{r}");
}

#[test]
fn search_rows_become_memories_in_ranked_order_and_skip_what_is_not_one() {
    let response = serde_json::json!({
        "documents": [
            {"file_path": "memory/general/a.md", "title": "A", "description": "first line",
             "frontmatter": {"id": "mem-aaaaaaaaaaaa", "status": "confirmed"}, "score": 0.9},
            {"file_path": "memory/general/no-frontmatter.md", "title": "B", "score": 0.8},
            {"file_path": "memory/general/c.md",
             "frontmatter": {"id": "mem-cccccccccccc"}, "score": 0.7},
        ]
    });
    let rows = memory_rows(&response);
    let ids: Vec<&str> = rows.iter().map(|(_, f)| f.id()).collect();
    assert_eq!(ids, ["mem-aaaaaaaaaaaa", "mem-cccccccccccc"]);
    assert_eq!(rows[0].0, 0.9);
    assert_eq!(rows[0].1.str("description"), Some("first line"));
    assert_eq!(rows[0].1.str("title"), Some("A"));
    assert!(memory_rows(&serde_json::Value::Null).is_empty());
}
