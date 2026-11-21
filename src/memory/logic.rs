//! The memory tools' pure rules: ids, paths, the frontmatter a memory carries, the
//! edits the tools make to a stored one, and the budgeted block `memory_recall`
//! returns. Nothing here touches the filesystem, the index or the network, so each
//! rule is tested on its own.

use std::collections::{BTreeSet, HashMap};

use serde_json::{Value, json};

/// The contract version clients check against (`docs/memory-contract.md`). Bump it
/// when a tool's parameters or results change in a way a client must notice.
pub const CONTRACT_VERSION: u32 = 1;

pub const TYPES: [&str; 5] = ["preference", "fact", "decision", "finding", "procedure"];
pub const STATUSES: [&str; 4] = ["proposed", "confirmed", "superseded", "forgotten"];

/// Statuses a memory is searched, recalled and updated in. A superseded or forgotten
/// one is history.
pub const LIVE_STATUSES: [&str; 2] = ["proposed", "confirmed"];

pub const MAX_TITLE_CHARS: usize = 200;
/// A memory is a short statement, not a document: it is also what gets pasted into a
/// model's context.
pub const MAX_BODY_BYTES: usize = 4000;
pub const MAX_TAGS: usize = 16;
pub const MAX_TAG_CHARS: usize = 64;
pub const MAX_SOURCE_URL_CHARS: usize = 2000;

const ID_PREFIX: &str = "mem-";
const ID_HEX_LEN: usize = 12;
const DESCRIPTION_CHARS: usize = 160;
const SLUG_CHARS: usize = 40;

/// A fresh memory id: `mem-` and 12 random hex digits.
pub fn new_id() -> Result<String, String> {
    let mut bytes = [0u8; ID_HEX_LEN / 2];
    getrandom::fill(&mut bytes).map_err(|e| format!("no randomness available: {e}"))?;
    Ok(format!("{ID_PREFIX}{}", hex::encode(bytes)))
}

/// Whether `id` has the shape [`new_id`] makes. Checked before an id is used in a
/// path, a file-name match or a filter, so nothing else can be smuggled through one.
pub fn is_valid_id(id: &str) -> bool {
    id.strip_prefix(ID_PREFIX).is_some_and(|hex| {
        hex.len() == ID_HEX_LEN && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    })
}

/// A file-name-safe rendering of `title`: lowercase ASCII letters and digits joined
/// by single dashes, at most [`SLUG_CHARS`] long, `memory` when nothing is left.
pub fn slugify(title: &str) -> String {
    let mut slug = String::new();
    let mut pending_dash = false;
    for c in title.chars() {
        if c.is_ascii_alphanumeric() {
            if pending_dash && !slug.is_empty() {
                slug.push('-');
            }
            pending_dash = false;
            slug.push(c.to_ascii_lowercase());
        } else {
            pending_dash = true;
        }
        if slug.len() >= SLUG_CHARS {
            break;
        }
    }
    slug.truncate(SLUG_CHARS);
    let slug = slug.trim_end_matches('-');
    if slug.is_empty() {
        "memory".to_string()
    } else {
        slug.to_string()
    }
}

/// Where a memory lives: `<root>/<area>/<slug>-<id>.md`.
pub fn memory_path(root: &str, area: &str, title: &str, id: &str) -> String {
    format!("{root}/{area}/{}-{id}.md", slugify(title))
}

/// Whether the file at `path` is the memory `id`: its name ends `-<id>.md` (or is
/// `<id>.md`).
pub fn path_is_memory(path: &str, id: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    name == format!("{id}.md") || name.ends_with(&format!("-{id}.md"))
}

/// The one-line `description` kept in frontmatter: the first non-empty line of the
/// text, cut at [`DESCRIPTION_CHARS`].
pub fn describe(body: &str) -> String {
    let first = body
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    let mut out: String = first.chars().take(DESCRIPTION_CHARS).collect();
    if first.chars().count() > DESCRIPTION_CHARS {
        out.push('…');
    }
    out
}

/// Check the caller-supplied parts of a new or edited memory.
pub fn validate_fields(
    title: Option<&str>,
    body: Option<&str>,
    tags: Option<&[String]>,
    source_url: Option<&str>,
) -> Result<(), String> {
    if let Some(title) = title {
        if title.trim().is_empty() {
            return Err("title must not be empty".into());
        }
        if title.chars().count() > MAX_TITLE_CHARS {
            return Err(format!("title is over {MAX_TITLE_CHARS} characters"));
        }
        if title.contains('\n') {
            return Err("title must be one line".into());
        }
    }
    if let Some(body) = body {
        if body.trim().is_empty() {
            return Err("body must not be empty".into());
        }
        if body.len() > MAX_BODY_BYTES {
            return Err(format!(
                "body is over {MAX_BODY_BYTES} bytes: a memory is a short statement; save the \
                 point, not the source"
            ));
        }
    }
    if let Some(tags) = tags {
        if tags.len() > MAX_TAGS {
            return Err(format!("at most {MAX_TAGS} tags"));
        }
        for tag in tags {
            if tag.trim().is_empty() || tag.chars().count() > MAX_TAG_CHARS || tag.contains('\n') {
                return Err(format!(
                    "a tag must be 1 to {MAX_TAG_CHARS} characters on one line"
                ));
            }
        }
    }
    if let Some(url) = source_url
        && (url.chars().count() > MAX_SOURCE_URL_CHARS || url.contains('\n'))
    {
        return Err("source_url is too long or spans lines".into());
    }
    Ok(())
}

/// Tags trimmed, de-duplicated, in the order given.
pub fn clean_tags(tags: &[String]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    tags.iter()
        .map(|t| t.trim().to_string())
        .filter(|t| seen.insert(t.clone()))
        .collect()
}

/// Everything a new memory's file is made from.
pub struct NewMemory<'a> {
    pub id: &'a str,
    pub title: &'a str,
    pub body: &'a str,
    pub ty: &'a str,
    pub tags: &'a [String],
    pub source_client: &'a str,
    pub source_url: Option<&'a str>,
    pub status: &'a str,
    pub supersedes: Option<&'a str>,
    pub now: &'a str,
}

fn body_text(body: &str) -> String {
    format!("{}\n", body.trim_end())
}

/// The full file for a new memory.
pub fn render_new(m: &NewMemory<'_>) -> Result<String, String> {
    let mut fm: HashMap<String, Value> = HashMap::new();
    fm.insert("id".into(), json!(m.id));
    fm.insert("title".into(), json!(m.title.trim()));
    fm.insert("description".into(), json!(describe(m.body)));
    fm.insert("type".into(), json!(m.ty));
    fm.insert("status".into(), json!(m.status));
    fm.insert("source_client".into(), json!(m.source_client));
    fm.insert("created".into(), json!(m.now));
    fm.insert("updated".into(), json!(m.now));
    if !m.tags.is_empty() {
        fm.insert("tags".into(), json!(clean_tags(m.tags)));
    }
    if let Some(url) = m.source_url.filter(|u| !u.trim().is_empty()) {
        fm.insert("source_url".into(), json!(url.trim()));
    }
    if let Some(old) = m.supersedes {
        fm.insert("supersedes".into(), json!(old));
    }
    let block = crate::write::render_frontmatter_block(&fm, "\n")?;
    Ok(format!("{block}\n{}", body_text(m.body)))
}

/// What `memory_update` may change.
#[derive(Debug, Default, Clone)]
pub struct Changes {
    pub title: Option<String>,
    pub body: Option<String>,
    pub tags: Option<Vec<String>>,
    pub source_url: Option<String>,
}

impl Changes {
    pub fn is_empty(&self) -> bool {
        self.title.is_none()
            && self.body.is_none()
            && self.tags.is_none()
            && self.source_url.is_none()
    }
}

fn split(old: &str) -> (HashMap<String, Value>, String) {
    crate::validate::parse_frontmatter_raw(old)
}

fn string_field<'a>(fm: &'a HashMap<String, Value>, key: &str) -> Option<&'a str> {
    fm.get(key).and_then(Value::as_str)
}

/// The status recorded in a stored memory's text.
pub fn status_of(content: &str) -> Option<String> {
    string_field(&split(content).0, "status").map(str::to_string)
}

/// Apply `changes` to a stored memory. A changed title or text puts the status back
/// to `status_after_content_change` (so an edit is reviewed again); `updated` moves
/// only when something really changed. Returns the old text untouched when nothing
/// does.
pub fn apply_changes(
    old: &str,
    changes: &Changes,
    status_after_content_change: &str,
    now: &str,
) -> Result<String, String> {
    let (mut fm, mut body) = split(old);
    let before = (fm.clone(), body.clone());
    let mut content_changed = false;

    if let Some(title) = &changes.title
        && string_field(&fm, "title") != Some(title.trim())
    {
        fm.insert("title".into(), json!(title.trim()));
        content_changed = true;
    }
    if let Some(new_body) = &changes.body
        && body.trim() != new_body.trim()
    {
        body = body_text(new_body);
        fm.insert("description".into(), json!(describe(new_body)));
        content_changed = true;
    }
    if let Some(tags) = &changes.tags {
        let tags = clean_tags(tags);
        if tags.is_empty() {
            fm.remove("tags");
        } else {
            fm.insert("tags".into(), json!(tags));
        }
    }
    if let Some(url) = &changes.source_url {
        if url.trim().is_empty() {
            fm.remove("source_url");
        } else {
            fm.insert("source_url".into(), json!(url.trim()));
        }
    }
    if content_changed {
        fm.insert("status".into(), json!(status_after_content_change));
    }
    if (fm.clone(), body.clone()) == before {
        return Ok(old.to_string());
    }
    fm.insert("updated".into(), json!(now));
    let block = crate::write::render_frontmatter_block(&fm, "\n")?;
    Ok(format!("{block}\n{}", body.trim_start_matches('\n')))
}

/// The stored memory with `status` set (and `updated` moved), whatever it was.
pub fn with_status(old: &str, status: &str, now: &str) -> Result<String, String> {
    let (mut fm, body) = split(old);
    if string_field(&fm, "status") == Some(status) {
        return Ok(old.to_string());
    }
    fm.insert("status".into(), json!(status));
    fm.insert("updated".into(), json!(now));
    let block = crate::write::render_frontmatter_block(&fm, "\n")?;
    Ok(format!("{block}\n{}", body.trim_start_matches('\n')))
}

/// One memory offered to [`recall_pack`].
#[derive(Debug, Clone, PartialEq)]
pub struct RecallItem {
    pub id: String,
    pub ty: String,
    pub title: String,
    pub text: String,
    pub score: f64,
    pub path: String,
}

/// The block `memory_recall` returns.
#[derive(Debug, PartialEq)]
pub struct Pack {
    pub context: String,
    /// Indices into the offered items, in the order they were written.
    pub included: Vec<usize>,
    /// Something that matched did not fit.
    pub truncated: bool,
}

const PACK_HEADER: &str = "Relevant memories:";
/// The least room worth spending on a cut-off line.
const MIN_PARTIAL_CHARS: usize = 60;

fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Order `items` for recall: best score first, then path, so equal inputs always give
/// equal output.
pub fn recall_order(items: &mut [RecallItem]) {
    items.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
    });
}

/// A compact block of at most `max_items` memories within `max_chars` characters,
/// taken in the order given. A line that does not fit whole is cut when enough room
/// is left for it to say something, and the block ends there.
pub fn recall_pack(items: &[RecallItem], max_items: usize, max_chars: usize) -> Pack {
    let mut context = String::new();
    let mut included = Vec::new();
    let mut truncated = false;
    if items.is_empty() {
        return Pack {
            context,
            included,
            truncated,
        };
    }
    let mut used = PACK_HEADER.chars().count();
    if used > max_chars {
        return Pack {
            context,
            included,
            truncated: true,
        };
    }
    context.push_str(PACK_HEADER);
    for (i, item) in items.iter().enumerate() {
        if included.len() >= max_items {
            truncated = true;
            break;
        }
        let head = format!("\n- [{}] {}: ", item.ty, collapse(&item.title));
        let text = collapse(&item.text);
        let whole = head.chars().count() + text.chars().count();
        if used + whole <= max_chars {
            context.push_str(&head);
            context.push_str(&text);
            used += whole;
            included.push(i);
            continue;
        }
        truncated = true;
        let room = max_chars.saturating_sub(used + head.chars().count() + 1);
        if room >= MIN_PARTIAL_CHARS {
            let cut: String = text.chars().take(room).collect();
            context.push_str(&head);
            context.push_str(cut.trim_end());
            context.push('…');
            included.push(i);
        }
        break;
    }
    Pack {
        context,
        included,
        truncated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: &str = "2026-11-21T10:00:00Z";

    fn new_memory<'a>(tags: &'a [String]) -> NewMemory<'a> {
        NewMemory {
            id: "mem-0123456789ab",
            title: "Prefers tabs",
            body: "The user prefers tabs over spaces.\n\nAlways.",
            ty: "preference",
            tags,
            source_client: "hermes",
            source_url: None,
            status: "proposed",
            supersedes: None,
            now: NOW,
        }
    }

    #[test]
    fn ids_have_one_shape_and_are_validated_before_use() {
        let id = new_id().unwrap();
        assert!(is_valid_id(&id), "{id}");
        assert_ne!(id, new_id().unwrap());
        for bad in [
            "",
            "mem-",
            "mem-0123456789a",
            "mem-0123456789abc",
            "mem-0123456789AB",
            "../etc/passwd",
            "mem-0123456789a/",
            "x-0123456789ab",
        ] {
            assert!(!is_valid_id(bad), "{bad:?}");
        }
    }

    #[test]
    fn slugs_are_short_ascii_and_never_empty() {
        assert_eq!(
            slugify("Prefers   TABS, not spaces!"),
            "prefers-tabs-not-spaces"
        );
        assert_eq!(slugify("日本語"), "memory");
        assert_eq!(slugify("  --  "), "memory");
        assert!(slugify(&"word ".repeat(50)).len() <= SLUG_CHARS);
        assert!(!slugify(&"a".repeat(100)).contains('/'));
        assert_eq!(
            memory_path("memory", "general", "Hello World", "mem-0123456789ab"),
            "memory/general/hello-world-mem-0123456789ab.md"
        );
        assert!(path_is_memory(
            "memory/general/hello-world-mem-0123456789ab.md",
            "mem-0123456789ab"
        ));
        assert!(!path_is_memory(
            "memory/general/hello-world-mem-0123456789ac.md",
            "mem-0123456789ab"
        ));
        assert!(!path_is_memory(
            "memory/general/xmem-0123456789ab.md",
            "mem-0123456789ab"
        ));
    }

    #[test]
    fn input_limits_are_enforced_with_a_message_that_says_what_to_change() {
        assert!(validate_fields(Some("ok"), Some("ok"), Some(&[]), None).is_ok());
        for (title, body) in [
            (Some(""), None),
            (Some("two\nlines"), None),
            (Some(&"t".repeat(MAX_TITLE_CHARS + 1)[..]), None),
            (None, Some("  ")),
            (None, Some(&"b".repeat(MAX_BODY_BYTES + 1)[..])),
        ] {
            assert!(
                validate_fields(title, body, None, None).is_err(),
                "{title:?}"
            );
        }
        let many: Vec<String> = (0..=MAX_TAGS).map(|i| i.to_string()).collect();
        assert!(validate_fields(None, None, Some(&many), None).is_err());
        assert!(validate_fields(None, None, Some(&["".to_string()]), None).is_err());
        assert!(validate_fields(None, None, None, Some("a\nb")).is_err());
    }

    #[test]
    fn a_new_memory_renders_the_frontmatter_the_schema_requires() {
        let tags = vec![
            "editor".to_string(),
            "editor".to_string(),
            " style ".to_string(),
        ];
        let text = render_new(&new_memory(&tags)).unwrap();
        let (fm, body) = crate::validate::parse_frontmatter_raw(&text);
        for key in [
            "id",
            "title",
            "description",
            "type",
            "status",
            "created",
            "updated",
        ] {
            assert!(fm.contains_key(key), "{key} missing from {text}");
        }
        assert_eq!(fm["status"], json!("proposed"));
        assert_eq!(fm["source_client"], json!("hermes"));
        assert_eq!(
            fm["tags"],
            json!(["editor", "style"]),
            "trimmed and de-duplicated"
        );
        assert_eq!(
            fm["description"],
            json!("The user prefers tabs over spaces.")
        );
        assert!(!fm.contains_key("supersedes") && !fm.contains_key("source_url"));
        assert!(body.contains("Always."), "{body}");
    }

    #[test]
    fn a_content_edit_resets_the_status_but_a_tag_edit_does_not() {
        let tags = Vec::new();
        let mut m = new_memory(&tags);
        m.status = "confirmed";
        let stored = render_new(&m).unwrap();

        let tagged = apply_changes(
            &stored,
            &Changes {
                tags: Some(vec!["x".into()]),
                ..Default::default()
            },
            "proposed",
            "2026-12-01T00:00:00Z",
        )
        .unwrap();
        assert_eq!(status_of(&tagged).as_deref(), Some("confirmed"));
        let (fm, _) = crate::validate::parse_frontmatter_raw(&tagged);
        assert_eq!(fm["updated"], json!("2026-12-01T00:00:00Z"));
        assert_eq!(fm["created"], json!(NOW));

        let edited = apply_changes(
            &tagged,
            &Changes {
                body: Some("Spaces, actually.".into()),
                ..Default::default()
            },
            "proposed",
            "2026-12-02T00:00:00Z",
        )
        .unwrap();
        assert_eq!(status_of(&edited).as_deref(), Some("proposed"));
        let (fm, body) = crate::validate::parse_frontmatter_raw(&edited);
        assert_eq!(fm["description"], json!("Spaces, actually."));
        assert_eq!(body.trim(), "Spaces, actually.");
    }

    #[test]
    fn an_edit_that_changes_nothing_returns_the_stored_text_untouched() {
        let tags = Vec::new();
        let stored = render_new(&new_memory(&tags)).unwrap();
        let same = apply_changes(
            &stored,
            &Changes {
                title: Some("Prefers tabs".into()),
                body: Some("The user prefers tabs over spaces.\n\nAlways.\n".into()),
                ..Default::default()
            },
            "proposed",
            "2027-01-01T00:00:00Z",
        )
        .unwrap();
        assert_eq!(same, stored, "no commit, no `updated` bump");
    }

    #[test]
    fn with_status_moves_only_the_status_and_the_update_time() {
        let tags = Vec::new();
        let stored = render_new(&new_memory(&tags)).unwrap();
        let forgotten = with_status(&stored, "forgotten", "2027-01-01T00:00:00Z").unwrap();
        assert_eq!(status_of(&forgotten).as_deref(), Some("forgotten"));
        assert_eq!(
            with_status(&forgotten, "forgotten", "2028-01-01T00:00:00Z").unwrap(),
            forgotten,
            "already there: unchanged"
        );
        let (a, body_a) = crate::validate::parse_frontmatter_raw(&stored);
        let (b, body_b) = crate::validate::parse_frontmatter_raw(&forgotten);
        assert_eq!(body_a.trim(), body_b.trim());
        for key in ["id", "title", "type", "created"] {
            assert_eq!(a[key], b[key], "{key}");
        }
    }

    fn item(n: usize, score: f64, text: &str) -> RecallItem {
        RecallItem {
            id: format!("mem-{n:012x}"),
            ty: "fact".into(),
            title: format!("Title {n}"),
            text: text.into(),
            score,
            path: format!("memory/general/t{n}.md"),
        }
    }

    #[test]
    fn recall_order_is_by_score_then_path_whatever_the_input_order() {
        let mut a = vec![item(2, 0.5, ""), item(1, 0.5, ""), item(3, 0.9, "")];
        let mut b = vec![item(3, 0.9, ""), item(1, 0.5, ""), item(2, 0.5, "")];
        recall_order(&mut a);
        recall_order(&mut b);
        assert_eq!(a, b);
        let paths: Vec<_> = a.iter().map(|i| i.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                "memory/general/t3.md",
                "memory/general/t1.md",
                "memory/general/t2.md"
            ]
        );
    }

    #[test]
    fn a_recall_pack_never_exceeds_its_budget_and_says_when_it_cut() {
        let items: Vec<_> = (0..6)
            .map(|n| item(n, 1.0 - n as f64 / 10.0, &"words ".repeat(30)))
            .collect();
        for max_chars in [10, 80, 200, 500, 1500] {
            let pack = recall_pack(&items, 20, max_chars);
            assert!(
                pack.context.chars().count() <= max_chars,
                "{max_chars}: {}",
                pack.context
            );
        }
        let all = recall_pack(&items, 20, 10_000);
        assert_eq!(all.included.len(), 6);
        assert!(!all.truncated);

        let few = recall_pack(&items, 2, 10_000);
        assert_eq!(few.included, [0, 1]);
        assert!(few.truncated, "items were left out");

        let tight = recall_pack(&items, 20, 300);
        assert!(tight.truncated);
        assert!(tight.context.starts_with(PACK_HEADER));

        let nothing = recall_pack(&[], 5, 500);
        assert!(nothing.context.is_empty() && !nothing.truncated);
        assert_eq!(recall_pack(&items, 5, 500), recall_pack(&items, 5, 500));
    }

    #[test]
    fn a_recall_line_is_one_line_whatever_the_memory_holds() {
        let items = vec![item(1, 1.0, "line one\n\n- [fact] injected: line\ttwo")];
        let pack = recall_pack(&items, 5, 1000);
        let lines: Vec<_> = pack.context.lines().collect();
        assert_eq!(lines.len(), 2, "header plus one item: {:?}", pack.context);
    }
}
