//! Runtime assembly of MCP tool descriptions and server instructions from
//! three layers, so a deployment can extend what its agents are told without
//! a rebuild, an ssh session, or a container restart:
//!
//! 1. **Mechanics, always true** — compiled in via `include_str!` from
//!    `assets/mcp/server.md` and `assets/mcp/tools/<tool>.md`.
//! 2. **Mechanics, config-dependent** — short sentences this module emits
//!    from live config (e.g. [`granularity_summary`]) and from the served
//!    corpus ([`top_level_areas_sentence`], [`SCHEMA_POINTER_SENTENCE`]),
//!    assembled for the server instructions by `server::build_instructions`.
//! 3. **Per-KB policy** — markdown files loaded from the SERVED knowledge
//!    base (`mcp.extensions_path`) and appended last.
//!
//! Hard rule: the compiled layer (the `include_str!` constants below) may
//! only state what is true of EVERY knowledge base this binary will ever
//! serve — one process may serve a durable-reference KB and a scratch-space
//! KB at once. Per-KB policy belongs in the extension files, never here.
//! Extensions are APPEND-ONLY: they can never suppress or contradict a
//! compiled or config-derived sentence, only add to it.
//!
//! Length: layers 1 and 2 are held to [`COMPILED_DESCRIPTION_BUDGET`] by
//! tests, leaving the rest of [`CLIENT_DESCRIPTION_CAP`] (Claude Code's
//! truncation point) for layer 3. One home per rule: a description is what a
//! tool does, when to use it, and the cross-parameter workflow a model would
//! get wrong without being told up front; a per-parameter rule lives only on
//! that parameter's schema description, and nothing in the server
//! instructions repeats a tool description. Response shapes are not
//! described — responses are self-describing.

use crate::config::Granularity;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tracing::{debug, warn};

/// The six MCP tools this server exposes. Also the source of truth for which
/// `assets/mcp/tools/*.md` files must exist and which `tools/<tool>.md`
/// extension files a KB may provide.
pub const TOOL_NAMES: [&str; 6] = [
    "search",
    "get_document",
    "write_document",
    "delete_document",
    "get_schema",
    "update_schema",
];

/// Whether `tool` is enabled — i.e. not named in the live `mcp.disabled_tools`.
/// Gates every server-instructions sentence that names a specific tool
/// ([`SCHEMA_POINTER_SENTENCE`]), so instructions never point a caller at a
/// tool `tools/list`/`tools/call` will refuse.
pub fn tool_enabled(tool: &str, disabled_tools: &[String]) -> bool {
    !disabled_tools.iter().any(|t| t == tool)
}

/// Compiled-in server narrative — mechanics true of every KB this binary
/// could ever serve. See this module's doc comment for the hard rule that
/// keeps per-KB policy out of this file.
const SERVER_BASE: &str = include_str!("../assets/mcp/server.md");

/// Compiled-in per-tool base description, indexed by tool name. `None` for
/// any name outside [`TOOL_NAMES`].
fn compiled_tool_base(tool: &str) -> Option<&'static str> {
    match tool {
        "search" => Some(include_str!("../assets/mcp/tools/search.md")),
        "get_document" => Some(include_str!("../assets/mcp/tools/get_document.md")),
        "write_document" => Some(include_str!("../assets/mcp/tools/write_document.md")),
        "delete_document" => Some(include_str!("../assets/mcp/tools/delete_document.md")),
        "get_schema" => Some(include_str!("../assets/mcp/tools/get_schema.md")),
        "update_schema" => Some(include_str!("../assets/mcp/tools/update_schema.md")),
        _ => None,
    }
}

/// Cap on the body text taken from one KB extension file, in bytes. Keeps a
/// misbehaving or oversized document from ballooning a description that is
/// paid on every `tools/list` response. Truncated on a char boundary; a
/// truncation is logged via [`cap_extension_body`].
const MAX_EXTENSION_BODY_BYTES: usize = 8 * 1024;

/// Length, in characters, past which Claude Code truncates a tool
/// `description` or the server `instructions` (appending "… [truncated]").
/// This is that client's observed behavior, not an MCP specification limit;
/// input-schema property descriptions are not truncated, which is why
/// detailed per-parameter rules live on the properties.
pub const CLIENT_DESCRIPTION_CAP: usize = 2048;

/// Ceiling, in characters, on every compiled + config-derived tool
/// description and on the server instructions (with a realistic corpus),
/// under the configuration that makes each longest. The gap up to
/// [`CLIENT_DESCRIPTION_CAP`] is left for the knowledge base's own extension
/// files. Enforced by tests, not at runtime.
pub const COMPILED_DESCRIPTION_BUDGET: usize = 600;

/// Last over-cap length warned about, per surface (`"server instructions"` or
/// a tool name), so [`warn_if_over_client_cap`] logs once per distinct length
/// instead of on every metadata-refresh tick.
static OVER_CAP_WARNED: std::sync::LazyLock<std::sync::Mutex<HashMap<String, usize>>> =
    std::sync::LazyLock::new(Default::default);

/// Warn when a fully composed description (extension included) is longer
/// than [`CLIENT_DESCRIPTION_CAP`] — Claude Code will cut it off. The
/// compiled + config-derived part is held under
/// [`COMPILED_DESCRIPTION_BUDGET`] by tests, so the extension file is the
/// likely culprit. Logged once per surface per distinct length; dropping back
/// under the cap re-arms it.
pub fn warn_if_over_client_cap(surface: &str, composed: &str) {
    // UTF-16 code units, not chars: Claude Code is a JavaScript client, and
    // counting the larger of the two can only make this fire earlier.
    let len = composed.encode_utf16().count();
    let mut warned = OVER_CAP_WARNED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if len <= CLIENT_DESCRIPTION_CAP {
        warned.remove(surface);
        return;
    }
    if warned.get(surface) == Some(&len) {
        return;
    }
    warned.insert(surface.to_string(), len);
    warn!(
        surface,
        chars = len,
        cap = CLIENT_DESCRIPTION_CAP,
        "MCP {surface} is {len} characters, over the {CLIENT_DESCRIPTION_CAP}-character \
         length Claude Code truncates at; its knowledge-base extension file is the likely \
         culprit — keep extensions under ~{} characters",
        CLIENT_DESCRIPTION_CAP - COMPILED_DESCRIPTION_BUDGET
    );
}

/// The sentence appended to `search`'s description when quoted-phrase
/// matching is both configured (`search.phrase`) and actually available at
/// runtime (`status::INDEX_STATUS.phrase_matching_available()`) — callers
/// compute that combined `phrase_effective` flag the same way the search
/// handler does, then pass it to [`compose_tool_description`].
const PHRASE_SYNTAX_SENTENCE: &str =
    "Wrap a span in double quotes for an exact phrase match, e.g. `\"node:ares\"`.";

/// The config-derived text documenting `search`'s `granularity` parameter
/// (#286): which values this server accepts, what each returns, and what an
/// omitted `granularity` defaults to. Describes ONLY the currently-enabled
/// granularities — a disabled one is never even mentioned, since
/// `KbSearchServer::overlay_input_schema` has already removed it from the
/// tool's schema `enum`. `effective` is
/// `ResolvedConfig::effective_granularities`'s output.
///
/// Served as the `granularity` property's description in the tool's input
/// schema (`overlay_input_schema`) — property descriptions are not subject to
/// the client's description cap (see [`CLIENT_DESCRIPTION_CAP`]), so the full
/// per-value detail lives here. The tool description carries only
/// [`granularity_summary`], a one-sentence pointer naming the same values.
///
/// Each default clause must match `mcp::resolve_search_granularity`: with a
/// query, the first enabled of chunk → document → section; without one,
/// `document` or an error. `mcp`'s
/// `granularity_description_matches_resolution_for_every_effective_set` test
/// checks every non-empty subset against that function. The seven cases are
/// enumerated as deliberately-worded prose rather than templated.
pub fn granularity_description(effective: &[Granularity]) -> String {
    const SECTION: &str = "the heading path and line range of each matching section, no \
                           text — requires a query; read one with `get_document`'s \
                           `start_line`/`end_line` (a `scope: whole_document` row's range \
                           may be stale: read that document whole)";
    let chunk = effective.contains(&Granularity::Chunk);
    let document = effective.contains(&Granularity::Document);
    let section = effective.contains(&Granularity::Section);

    match (chunk, document, section) {
        (true, true, true) => format!(
            "`granularity` decides what a result is: `document` (one row each), `chunk` \
             (scored snippets, several per document), or `section` ({SECTION}). Defaults to \
             `chunk` with a query, `document` without; `section` is only ever explicit."
        ),
        (true, true, false) => "`granularity` decides what a result is: `document` (one row \
             each) or `chunk` (scored snippets, several per document). Defaults to `chunk` \
             with a query, `document` without."
            .to_string(),
        (true, false, true) => format!(
            "`granularity` decides what a result is: `chunk` (scored snippets, several per \
             document) or `section` ({SECTION}). Defaults to `chunk`; every search needs a \
             query on this server."
        ),
        (false, true, true) => format!(
            "`granularity` decides what a result is: `document` (one row each) or `section` \
             ({SECTION}). Defaults to `document`; `section` is only ever explicit."
        ),
        (true, false, false) => "`granularity` is fixed to `chunk` (scored snippets, several \
             per document) on this server; every search needs a query."
            .to_string(),
        (false, true, false) => {
            "`granularity` is fixed to `document` (one row each) on this server.".to_string()
        }
        (false, false, true) => format!(
            "`granularity` is fixed to `section` ({SECTION}) on this server; every search \
             needs a query."
        ),
        // Rejected at config load (`search.granularities leaves no granularity
        // enabled`), so unreachable from a validated config; still worded rather
        // than empty in case a test or future caller builds one by hand.
        (false, false, false) => "No `granularity` is currently enabled on this server — \
             every `search` call will fail. This is a deployment configuration problem \
             (search.granularities / chunking.heading_metadata), not something a caller can \
             work around."
            .to_string(),
    }
}

/// The one-sentence `granularity` pointer in `search`'s composed tool
/// description: names only the enabled values (never a disabled one — the
/// same rule [`granularity_description`] follows), leaving what each value
/// returns and the default rules to the `granularity` property's description.
/// The empty set is unreachable from a validated config and falls back to
/// [`granularity_description`]'s configuration-problem wording.
pub fn granularity_summary(effective: &[Granularity]) -> String {
    let names: Vec<String> = effective
        .iter()
        .map(|g| format!("`{}`", g.as_str()))
        .collect();
    match names.as_slice() {
        [] => granularity_description(effective),
        [only] => format!("`granularity` is fixed to {only} on this server."),
        [init @ .., last] => format!(
            "`granularity` chooses what each result is: {} or {last}.",
            init.join(", ")
        ),
    }
}

/// Whether any enabled granularity can serve a `search` without a query
/// (enumeration). When none can, every sentence and schema property that
/// only makes sense for enumeration is dropped or rewritten, so a caller is
/// never told about a mode it cannot use (#286).
pub fn enumeration_available(effective: &[Granularity]) -> bool {
    effective.iter().any(|g| g.supports_no_query())
}

/// `search`'s enumeration paragraph, included only when
/// [`enumeration_available`]. Kept out of the compiled `search.md` for that
/// reason.
pub const ENUMERATION_SENTENCE: &str = "Without `query`, every match is listed in a stable order \
     with an exact total — use this for a complete set, not the best few.";

/// Effective-set-aware descriptions for `search`'s schema properties whose
/// static doc comments assume every granularity and mode is available
/// (#286): `(property, Some(description))` to replace a description,
/// `(property, None)` to remove a property that no enabled granularity can
/// use. Applied by `KbSearchServer::overlay_input_schema` on every
/// `list_tools`/`get_tool`, alongside the `granularity` property's own text.
/// Never names a granularity that isn't enabled. `hybrid` is `search.hybrid`:
/// the `query` text says the query is matched literally as well as
/// semantically only when the sparse arm actually runs (#309).
pub fn search_property_descriptions(
    effective: &[Granularity],
    hybrid: bool,
) -> Vec<(&'static str, Option<String>)> {
    let enumeration = enumeration_available(effective);
    let text = |s: &str| Some(s.to_string());
    let query_lead = if hybrid {
        "Query, matched by literal terms and by semantic similarity."
    } else {
        "Semantic query."
    };
    let mut out = Vec::new();
    if enumeration {
        out.push((
            "query",
            Some(format!("{query_lead} Omit to list every match.")),
        ));
        out.push((
            "offset",
            text(
                "Results to skip. Without a query it pages the full listing; with one, only \
                 as deep as the ranked results go.",
            ),
        ));
        out.push(("order_by", text("Sort key when listing without a query.")));
        out.push((
            "descending",
            text("Sort descending (when listing without a query)."),
        ));
        out.push(("min_score", text("Relevance floor (with a query).")));
    } else {
        out.push((
            "query",
            Some(format!("{query_lead} Required on this server.")),
        ));
        out.push((
            "offset",
            text("Results to skip; pages only as deep as the ranked results go."),
        ));
        out.push(("order_by", None));
        out.push(("descending", None));
        out.push(("min_score", text("Relevance floor.")));
    }
    if effective.contains(&Granularity::Chunk) {
        out.push((
            "explain",
            text("Add a score breakdown per result (`chunk` granularity only)."),
        ));
    } else {
        out.push(("explain", None));
    }
    if effective.contains(&Granularity::Document) {
        out.push((
            "fields",
            text("Frontmatter fields to include per result, as dot-paths (`document` only)."),
        ));
    } else {
        out.push(("fields", None));
    }
    out
}

/// The server-instructions line naming the knowledge base's top-level areas —
/// useful both for targeting `path_prefix` and for placing a new document, so
/// it names no tool and is emitted whichever tools are enabled. `None` for a
/// corpus with no top-level directories.
pub fn top_level_areas_sentence(areas: &[String]) -> Option<String> {
    (!areas.is_empty()).then(|| format!("Top-level areas: {}.", areas.join(", ")))
}

/// The one server-instructions pointer to per-field and per-folder rules —
/// emitted only while `get_schema` is enabled. Field vocabularies and scoped
/// folders are never enumerated up front: `get_schema` serves them on demand
/// (with `values_in_use` for open fields), and a search filter or write that
/// gets one wrong is refused with the options listed.
pub const SCHEMA_POINTER_SENTENCE: &str =
    "Field values and per-folder rules: get_schema with a path.";

/// The config-derived sentence(s) for `search`'s tool description:
/// [`ENUMERATION_SENTENCE`] when [`enumeration_available`], then
/// [`granularity_summary`]. A pure function of live config bits, no I/O.
/// `heading_prefix` is not mentioned: its rules live on the property, which
/// `overlay_input_schema` removes while `chunking.heading_metadata` is off.
pub fn granularity_sentence(effective: &[Granularity]) -> String {
    let mut sentence = String::new();
    if enumeration_available(effective) {
        sentence.push_str(ENUMERATION_SENTENCE);
        sentence.push('\n');
    }
    sentence.push_str(&granularity_summary(effective));
    sentence
}

/// Join non-empty, trimmed sections with a blank line between them, then
/// trim the result's trailing whitespace. Shared by every composition
/// function so the append rule (blank line between sections, no trailing
/// whitespace) is enforced in exactly one place.
fn join_sections<'a>(sections: impl IntoIterator<Item = &'a str>) -> String {
    sections
        .into_iter()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// The compiled server base — the only part of the server instructions that
/// depends on neither config nor corpus. `server::build_instructions` takes
/// this as its `base` and appends the corpus-dependent lines
/// ([`top_level_areas_sentence`], [`SCHEMA_POINTER_SENTENCE`]); the KB
/// extension (category 3) is appended after THAT, via [`append_extension`].
/// How `search` matches text is not repeated here: the `search` description
/// and its `query`/`path_prefix` properties are its one home.
pub fn compose_server_mechanics() -> String {
    join_sections([SERVER_BASE])
}

/// Compose one tool's final description: its compiled base, then for
/// `search` the phrase-syntax sentence when applicable and the
/// enumeration/granularity sentence ([`granularity_sentence`], #286) — one
/// per line, not as paragraphs — then the KB's extension for that tool.
/// Returns `None` for a name outside [`TOOL_NAMES`].
pub fn compose_tool_description(
    tool: &str,
    phrase_effective: bool,
    effective_granularities: &[Granularity],
    extension: Option<&str>,
) -> Option<String> {
    let base = compiled_tool_base(tool)?;
    let compiled = if tool == "search" {
        let mut lines = vec![base.trim().to_string()];
        if phrase_effective {
            lines.push(PHRASE_SYNTAX_SENTENCE.to_string());
        }
        lines.push(granularity_sentence(effective_granularities));
        lines.join("\n")
    } else {
        base.to_string()
    };
    Some(join_sections(
        std::iter::once(compiled.as_str()).chain(extension),
    ))
}

/// Append a KB extension (category 3) onto an already-composed base (server
/// instructions after `build_instructions`, or a tool's compiled+config-derived
/// description). A no-op when `extension` is `None` or blank.
pub fn append_extension(base: &str, extension: Option<&str>) -> String {
    join_sections(std::iter::once(base).chain(extension))
}

/// Every tool's composed description, keyed by tool name — the exact map
/// installed into `KbSearchServer`'s description overlay. `extensions_dir`
/// should already be resolved (see [`resolve_extensions_dir`]) so a caller
/// that also needs the server's `server.md` extension resolves the
/// directory only once per call. Each composed description goes through
/// [`warn_if_over_client_cap`].
pub fn compose_tool_descriptions(
    extensions_dir: Option<&Path>,
    phrase_effective: bool,
    effective_granularities: &[Granularity],
) -> HashMap<String, String> {
    TOOL_NAMES
        .iter()
        .filter_map(|&tool| {
            let extension = load_tool_extension(extensions_dir, tool);
            compose_tool_description(
                tool,
                phrase_effective,
                effective_granularities,
                extension.as_deref(),
            )
            .map(|desc| {
                warn_if_over_client_cap(&format!("{tool} tool description"), &desc);
                (tool.to_string(), desc)
            })
        })
        .collect()
}

/// Which of a KB's `server.md` extension file and the deprecated
/// `mcp.instructions` setting wins when composing server instructions. The
/// file always wins when both are present — see [`log_instructions_deprecation`]
/// for the one-time startup log that reports which case applied.
pub fn effective_server_extension(
    file_extension: Option<String>,
    legacy_instructions: Option<&str>,
) -> Option<String> {
    file_extension.or_else(|| legacy_instructions.map(str::to_string))
}

/// Log, once, whether the deprecated `mcp.instructions` setting is doing
/// anything. Intended to be called once at server startup with the
/// startup-resolved extension-file presence; a later `POST /admin/reload`
/// does not re-trigger this (same "once at startup" contract as
/// [`log_absolute_extensions_path_advisory`]) — the periodic refresh
/// recomposes instructions on every tick without re-logging this.
pub fn log_instructions_deprecation(
    legacy_instructions: Option<&str>,
    file_present: bool,
    extensions_path: &str,
) {
    if legacy_instructions.is_none() {
        return;
    }
    if file_present {
        warn!(
            "mcp.instructions is deprecated and ignored: {extensions_path}/server.md is \
             present in the knowledge base and wins."
        );
    } else {
        warn!(
            "mcp.instructions is deprecated; move its text to {extensions_path}/server.md \
             so the knowledge base can edit its own policy narrative without a restart."
        );
    }
}

/// Log, once, that an absolute `mcp.extensions_path` is accepted but not
/// reachable through `write_document` — only paths under the knowledge base
/// root are. Intended to be called once at server startup; a later reload
/// that changes this value does not re-trigger it.
pub fn log_absolute_extensions_path_advisory(extensions_path: &str) {
    if !extensions_path.is_empty() && Path::new(extensions_path).is_absolute() {
        warn!(
            extensions_path,
            "mcp.extensions_path is absolute; its MCP description extension files live \
             outside the knowledge base and are not editable through write_document."
        );
    }
}

/// Resolve `mcp.extensions_path` to an absolute directory, or `None` when
/// extension loading is disabled (empty string) or a relative value escapes
/// the data root (rejected and logged).
///
/// A relative value resolves against `data_root` via the same
/// [`crate::write::resolve_safe_write_path`] the write tools use — which is
/// what makes the extension files editable through `write_document` in the
/// first place. An absolute value is accepted as-is; see
/// [`log_absolute_extensions_path_advisory`] for the one-time advisory that
/// goes with it.
pub fn resolve_extensions_dir(data_root: &Path, extensions_path: &str) -> Option<PathBuf> {
    if extensions_path.is_empty() {
        return None;
    }
    let candidate = Path::new(extensions_path);
    if candidate.is_absolute() {
        return Some(candidate.to_path_buf());
    }
    match crate::write::resolve_safe_write_path(data_root, extensions_path) {
        Ok(p) => Some(p),
        Err(e) => {
            warn!(
                extensions_path,
                error = %e,
                "mcp.extensions_path escapes the knowledge base root; MCP description \
                 extensions disabled until this is fixed"
            );
            None
        }
    }
}

/// Read one extension file's body, degrading to `None` (compiled base
/// alone) on any missing/unreadable/non-UTF-8 file. This loader must never
/// fail startup or a refresh tick, whatever is (or isn't) in the KB.
fn read_extension_body(path: &Path) -> Option<String> {
    // The knowledge base is a synced git repo, and git materializes tracked
    // symlinks on checkout, so a hostile upstream commit could place a
    // symlink here pointing at any process-readable file (state.db, a
    // credential, /proc/self/environ) and have its contents folded straight
    // into the MCP instructions sent to every connected client. Refuse to
    // read through a symlink, the same policy `ingest.rs`'s directory walk
    // already enforces for indexed content.
    if is_symlink(path) {
        warn!(
            path = %path.display(),
            "Skipping symlink: MCP description extension path is a symlink; using compiled base only"
        );
        return None;
    }
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            debug!(path = %path.display(), "no MCP description extension at this path");
            return None;
        }
        Err(e) => {
            warn!(
                path = %path.display(),
                error = %e,
                "failed to read MCP description extension; using compiled base only"
            );
            return None;
        }
    };
    let content = match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => {
            warn!(
                path = %path.display(),
                error = %e,
                "MCP description extension is not valid UTF-8; using compiled base only"
            );
            return None;
        }
    };

    // These files are indexed KB documents and carry frontmatter to satisfy
    // the KB's own schema; only the body may ever reach a tool description.
    let (_frontmatter, body) = crate::validate::parse_frontmatter_raw(&content);
    Some(cap_extension_body(body.trim(), path))
}

/// Enforce [`MAX_EXTENSION_BODY_BYTES`], truncating on a char boundary and
/// warning with both sizes when it actually cuts anything.
fn cap_extension_body(body: &str, path: &Path) -> String {
    if body.len() <= MAX_EXTENSION_BODY_BYTES {
        return body.to_string();
    }
    let mut end = MAX_EXTENSION_BODY_BYTES;
    while end > 0 && !body.is_char_boundary(end) {
        end -= 1;
    }
    warn!(
        path = %path.display(),
        original_bytes = body.len(),
        truncated_bytes = end,
        "MCP description extension exceeds {MAX_EXTENSION_BODY_BYTES} bytes; truncating"
    );
    body[..end].to_string()
}

/// Load the `server.md` extension body from `extensions_dir`, if any.
pub fn load_server_extension(extensions_dir: Option<&Path>) -> Option<String> {
    read_extension_body(&extensions_dir?.join("server.md"))
}

/// Load a tool's extension body (`tools/<tool>.md`) from `extensions_dir`, if any.
pub fn load_tool_extension(extensions_dir: Option<&Path>, tool: &str) -> Option<String> {
    let tools_dir = extensions_dir?.join("tools");
    // Guard the intermediate `tools/` component too, not just the leaf file:
    // a symlinked directory would let every `tools/<tool>.md` read escape the
    // extensions root just as effectively as a symlinked leaf file would.
    if is_symlink(&tools_dir) {
        warn!(
            path = %tools_dir.display(),
            "Skipping symlink: MCP description extension path is a symlink; using compiled base only"
        );
        return None;
    }
    read_extension_body(&tools_dir.join(format!("{tool}.md")))
}

/// True if `path`'s leaf component is itself a symlink (not following it).
/// A nonexistent or otherwise unstattable path is not a symlink for this
/// check's purposes — `read_extension_body`'s subsequent `std::fs::read`
/// handles the missing/unreadable cases and degrades the same way it always
/// has.
fn is_symlink(path: &Path) -> bool {
    std::fs::symlink_metadata(path)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    const ALL: [Granularity; 3] = Granularity::ALL;

    fn query_description(effective: &[Granularity], hybrid: bool) -> String {
        search_property_descriptions(effective, hybrid)
            .into_iter()
            .find(|(name, _)| *name == "query")
            .and_then(|(_, text)| text)
            .expect("query description is always set")
    }

    #[test]
    fn query_description_names_literal_matching_when_hybrid() {
        for effective in [&ALL[..], &[Granularity::Chunk][..]] {
            let text = query_description(effective, true);
            assert!(text.contains("literal terms"), "{text}");
            assert!(!text.contains("Semantic query"), "{text}");
        }
    }

    #[test]
    fn query_description_stays_semantic_when_not_hybrid() {
        for effective in [&ALL[..], &[Granularity::Chunk][..]] {
            let text = query_description(effective, false);
            assert!(text.starts_with("Semantic query."), "{text}");
            assert!(!text.contains("literal"), "{text}");
        }
    }

    #[test]
    fn query_description_keeps_the_enumeration_tail() {
        assert!(query_description(&ALL, true).ends_with("Omit to list every match."));
        assert!(
            query_description(&[Granularity::Chunk], true).ends_with("Required on this server.")
        );
    }

    // --- compiled bases ------------------------------------------------

    #[test]
    fn compiled_base_loads_for_every_tool() {
        for tool in TOOL_NAMES {
            let base = compiled_tool_base(tool);
            assert!(base.is_some(), "no compiled base for tool '{tool}'");
            assert!(
                !base.unwrap().trim().is_empty(),
                "compiled base for '{tool}' is blank"
            );
        }
    }

    #[test]
    fn compiled_base_is_none_for_unknown_tool() {
        assert!(compiled_tool_base("not_a_real_tool").is_none());
    }

    #[test]
    fn server_base_is_nonempty_and_does_not_assert_hybrid_fusion() {
        assert!(!SERVER_BASE.trim().is_empty());
        // The hybrid-fusion claim belongs on `search`'s `query` property, not
        // the compiled file, since it's false when `search.hybrid` is off.
        assert!(
            !SERVER_BASE.to_lowercase().contains("fuses"),
            "server.md must not assert hybrid fusion — that's config-dependent now"
        );
    }

    // --- composition order / append rules ------------------------------

    #[test]
    fn extension_appends_after_base_with_blank_line() {
        let desc =
            compose_tool_description("get_schema", false, &ALL, Some("Extra KB policy.")).unwrap();
        let base = compiled_tool_base("get_schema").unwrap().trim();
        assert!(desc.starts_with(base));
        assert!(desc.ends_with("Extra KB policy."));
        assert!(desc.contains("\n\n"));
    }

    #[test]
    fn missing_extension_yields_base_alone() {
        let desc = compose_tool_description("delete_document", false, &ALL, None).unwrap();
        assert_eq!(desc, compiled_tool_base("delete_document").unwrap().trim());
    }

    #[test]
    fn compose_tool_description_unknown_tool_is_none() {
        assert!(compose_tool_description("bogus", false, &ALL, None).is_none());
    }

    #[test]
    fn append_extension_is_noop_for_none_or_blank() {
        let base = "Base text.";
        assert_eq!(append_extension(base, None), base);
        assert_eq!(append_extension(base, Some("   ")), base);
        assert_eq!(append_extension(base, Some("More.")), "Base text.\n\nMore.");
    }

    #[test]
    fn join_sections_trims_trailing_whitespace() {
        let joined = join_sections(["a", "b  \n\n"]);
        assert_eq!(joined, "a\n\nb");
    }

    // --- search's phrase-syntax sentence, gated --------------------------

    #[test]
    fn search_description_includes_phrase_syntax_only_when_effective() {
        let with_phrase = compose_tool_description("search", true, &ALL, None).unwrap();
        let without_phrase = compose_tool_description("search", false, &ALL, None).unwrap();
        assert!(with_phrase.contains("double quotes"));
        assert!(!without_phrase.contains("double quotes"));
    }

    #[test]
    fn non_search_tools_never_get_the_phrase_sentence() {
        let desc = compose_tool_description("get_document", true, &ALL, None).unwrap();
        assert!(!desc.contains("double quotes"));
    }

    /// The server mechanics repeat nothing a tool description or property
    /// already says: how `search` matches text lives on `search` alone.
    #[test]
    fn compose_server_mechanics_is_the_base_alone() {
        let mechanics = compose_server_mechanics();
        assert_eq!(mechanics, SERVER_BASE.trim());
        for search_only in ["path_prefix", "phrase", "semantic", "regex"] {
            assert!(!mechanics.contains(search_only), "{mechanics}");
        }
    }

    // --- KB extension loading: frontmatter stripping, degradation -------

    #[test]
    fn frontmatter_is_stripped_and_never_appears_in_output() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("server.md"),
            "---\nstatus: active\ntype: reference\nsecret_tag: should-not-leak\n---\n\
             This is the real policy body.\n",
        )
        .unwrap();

        let body = load_server_extension(Some(tmp.path())).unwrap();
        assert!(body.contains("This is the real policy body."));
        assert!(!body.contains("secret_tag"));
        assert!(!body.contains("---"));
    }

    #[test]
    fn missing_extension_file_yields_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(load_server_extension(Some(tmp.path())).is_none());
        assert!(load_tool_extension(Some(tmp.path()), "search").is_none());
    }

    #[test]
    fn no_extensions_dir_yields_none() {
        assert!(load_server_extension(None).is_none());
    }

    #[test]
    fn frontmatter_only_extension_degrades_to_an_empty_body() {
        // A file that's all frontmatter and no body must not surface as `None`
        // (that would be indistinguishable from a missing/unreadable file in
        // logs) — it degrades to an empty string, which `append_extension`/
        // `join_sections` already treat as a no-op (see
        // `append_extension_is_noop_for_none_or_blank`).
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("server.md"),
            "---\nstatus: active\ntype: reference\n---\n",
        )
        .unwrap();

        assert_eq!(load_server_extension(Some(tmp.path())).as_deref(), Some(""));
    }

    #[test]
    fn pure_whitespace_extension_degrades_to_an_empty_body() {
        // A file with no frontmatter delimiters at all, and nothing but
        // whitespace, must degrade the same way: `Some("")`, not `None` and not
        // a body full of blank lines.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("server.md"), "   \n\n\t \n").unwrap();

        assert_eq!(load_server_extension(Some(tmp.path())).as_deref(), Some(""));
    }

    #[test]
    fn unreadable_or_unparseable_extension_yields_none() {
        let tmp = tempfile::tempdir().unwrap();
        // Invalid UTF-8 bytes: `read_to_string`/`String::from_utf8` reject this,
        // which is this loader's stand-in for "unparseable" — the frontmatter
        // parser itself is lenient by design and does not error on malformed
        // YAML, so a byte-level decode failure is the realistic failure mode.
        std::fs::write(tmp.path().join("server.md"), [0xFF, 0xFE, 0xFD]).unwrap();
        assert!(load_server_extension(Some(tmp.path())).is_none());
    }

    #[test]
    fn oversized_extension_is_truncated() {
        let tmp = tempfile::tempdir().unwrap();
        let huge = "x".repeat(MAX_EXTENSION_BODY_BYTES + 500);
        std::fs::write(tmp.path().join("server.md"), &huge).unwrap();

        let body = load_server_extension(Some(tmp.path())).unwrap();
        assert!(body.len() <= MAX_EXTENSION_BODY_BYTES);
        assert!(body.len() < huge.len());
    }

    #[test]
    fn tool_extension_path_is_tools_subdir() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("tools")).unwrap();
        std::fs::write(tmp.path().join("tools/search.md"), "Search policy body.").unwrap();

        assert_eq!(
            load_tool_extension(Some(tmp.path()), "search").as_deref(),
            Some("Search policy body.")
        );
    }

    // --- symlink refusal on the read path --------------------------------

    #[test]
    fn symlinked_server_extension_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("real_target.md");
        std::fs::write(&target, "Secret content that must not leak.").unwrap();
        std::os::unix::fs::symlink(&target, tmp.path().join("server.md")).unwrap();

        assert!(
            load_server_extension(Some(tmp.path())).is_none(),
            "a symlinked server.md must be refused, not read through"
        );
    }

    #[test]
    fn symlinked_tool_extension_leaf_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("tools")).unwrap();
        let target = tmp.path().join("real_target.md");
        std::fs::write(&target, "Secret content that must not leak.").unwrap();
        std::os::unix::fs::symlink(&target, tmp.path().join("tools/search.md")).unwrap();

        assert!(
            load_tool_extension(Some(tmp.path()), "search").is_none(),
            "a symlinked tools/<tool>.md must be refused, not read through"
        );
    }

    #[test]
    fn symlinked_tools_directory_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let real_tools_dir = tmp.path().join("real_tools");
        std::fs::create_dir_all(&real_tools_dir).unwrap();
        std::fs::write(real_tools_dir.join("search.md"), "Secret content.").unwrap();
        std::os::unix::fs::symlink(&real_tools_dir, tmp.path().join("tools")).unwrap();

        assert!(
            load_tool_extension(Some(tmp.path()), "search").is_none(),
            "a symlinked tools/ intermediate directory must be refused, not walked through"
        );
    }

    // --- extensions_path resolution -------------------------------------

    #[test]
    fn empty_extensions_path_disables_loading() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(resolve_extensions_dir(tmp.path(), "").is_none());
    }

    #[test]
    fn relative_extensions_path_resolves_under_data_root() {
        let tmp = tempfile::tempdir().unwrap();
        let canonical = tmp.path().canonicalize().unwrap();
        let resolved = resolve_extensions_dir(&canonical, "meta/mcp").unwrap();
        assert_eq!(resolved, canonical.join("meta/mcp"));
    }

    #[test]
    fn absolute_extensions_path_is_accepted_as_is() {
        let tmp = tempfile::tempdir().unwrap();
        let canonical = tmp.path().canonicalize().unwrap();
        let abs = canonical.join("elsewhere/mcp");
        let resolved = resolve_extensions_dir(&canonical, abs.to_str().unwrap()).unwrap();
        assert_eq!(resolved, abs);
    }

    #[test]
    fn relative_extensions_path_escaping_root_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let canonical = tmp.path().canonicalize().unwrap();
        assert!(resolve_extensions_dir(&canonical, "../../etc").is_none());
    }

    // --- mcp.instructions deprecation fallback ---------------------------

    #[test]
    fn file_wins_over_legacy_instructions_when_both_present() {
        let effective = effective_server_extension(
            Some("From the file.".to_string()),
            Some("From mcp.instructions."),
        );
        assert_eq!(effective.as_deref(), Some("From the file."));
    }

    #[test]
    fn legacy_instructions_used_when_no_file_present() {
        let effective = effective_server_extension(None, Some("From mcp.instructions."));
        assert_eq!(effective.as_deref(), Some("From mcp.instructions."));
    }

    #[test]
    fn neither_present_yields_none() {
        assert!(effective_server_extension(None, None).is_none());
    }

    // --- compose_tool_descriptions covers every tool --------------------

    #[test]
    fn compose_tool_descriptions_covers_every_tool() {
        let overlay = compose_tool_descriptions(None, false, &ALL);
        assert_eq!(overlay.len(), TOOL_NAMES.len());
        for tool in TOOL_NAMES {
            assert!(
                overlay.contains_key(tool),
                "missing overlay entry for '{tool}'"
            );
        }
    }

    #[test]
    fn compose_tool_descriptions_appends_each_tools_own_extension() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("tools")).unwrap();
        std::fs::write(tmp.path().join("tools/search.md"), "Search-only note.").unwrap();
        std::fs::write(
            tmp.path().join("tools/get_document.md"),
            "get_document-only note.",
        )
        .unwrap();

        let overlay = compose_tool_descriptions(Some(tmp.path()), false, &ALL);
        assert!(overlay["search"].ends_with("Search-only note."));
        assert!(overlay["get_document"].ends_with("get_document-only note."));
        assert!(!overlay["search"].contains("get_document-only note."));
        assert!(!overlay["delete_document"].contains("note."));
    }

    // --- granularity_sentence (#286) --------------------------------------

    #[test]
    fn granularity_sentence_all_three_mentions_every_value_and_the_default_rule() {
        let s = granularity_sentence(&ALL);
        assert!(s.contains("`chunk`"));
        assert!(s.contains("`document`"));
        assert!(s.contains("`section`"));
        // The default rule lives on the `granularity` property, not the
        // tool description.
        assert!(granularity_description(&ALL).contains("Defaults to `chunk` with a query"));
        assert!(
            !s.contains("heading_prefix"),
            "heading_prefix is described on its own property only: {s}"
        );
    }

    #[test]
    fn granularity_sentence_omits_disabled_values() {
        // `section` disabled: must not appear at all, not even to say it's
        // unavailable — a caller/model should never be told about a choice
        // it cannot make.
        let s = granularity_sentence(&[Granularity::Chunk, Granularity::Document]);
        assert!(s.contains("`chunk`"));
        assert!(s.contains("`document`"));
        assert!(
            !s.contains("`section`"),
            "a disabled granularity must not be mentioned: {s}"
        );
    }

    #[test]
    fn granularity_sentence_single_value_says_fixed() {
        let s = granularity_sentence(&[Granularity::Document]);
        assert!(s.contains("fixed to `document`"));
        assert!(!s.contains("`chunk`"));
        assert!(!s.contains("`section`"));
    }

    #[test]
    fn granularity_sentence_empty_effective_names_the_configuration_problem() {
        // Unreachable from a validated config (config load rejects an empty
        // effective set), but the sentence must still say something rather
        // than silently describing nothing.
        let s = granularity_sentence(&[]);
        assert!(s.contains("No `granularity` is currently enabled"));
    }

    // --- tool_enabled / top_level_areas_sentence (mcp.disabled_tools) ------

    #[test]
    fn tool_enabled_is_true_when_not_named_in_disabled_tools() {
        assert!(tool_enabled("search", &[]));
        assert!(tool_enabled(
            "search",
            &["get_schema".to_string(), "update_schema".to_string()]
        ));
    }

    #[test]
    fn tool_enabled_is_false_when_named_in_disabled_tools() {
        assert!(!tool_enabled("search", &["search".to_string()]));
    }

    #[test]
    fn top_level_areas_sentence_lists_the_areas_and_names_no_tool() {
        let areas = vec!["dev".to_string(), "food".to_string()];
        assert_eq!(
            top_level_areas_sentence(&areas).as_deref(),
            Some("Top-level areas: dev, food.")
        );
        assert_eq!(top_level_areas_sentence(&[]), None);
    }

    // --- composed `search` description mentions only enabled values ------

    #[test]
    fn composed_search_description_mentions_only_enabled_granularities() {
        let full = compose_tool_description("search", false, &ALL, None).unwrap();
        let restricted = compose_tool_description(
            "search",
            false,
            &[Granularity::Chunk, Granularity::Document],
            None,
        )
        .unwrap();

        assert!(full.contains("`section`"));
        assert!(
            !restricted.contains("`section`"),
            "search's composed description must not mention a disabled granularity: \
             {restricted}"
        );
        assert!(restricted.contains("`chunk`"));
        assert!(restricted.contains("`document`"));
    }

    #[test]
    fn composed_non_search_descriptions_never_carry_the_granularity_sentence() {
        let desc = compose_tool_description("get_document", false, &ALL, None).unwrap();
        assert!(!desc.contains("`granularity`"));
        assert!(!desc.contains("heading_prefix"));
    }

    // --- description length budget ----------------------------------------

    /// Every non-empty granularity subset, so the budget is checked against
    /// whichever combination of config-derived sentences is longest rather
    /// than one assumed to be.
    const GRANULARITY_SUBSETS: [&[Granularity]; 7] = [
        &ALL,
        &[Granularity::Chunk, Granularity::Document],
        &[Granularity::Chunk, Granularity::Section],
        &[Granularity::Document, Granularity::Section],
        &[Granularity::Chunk],
        &[Granularity::Document],
        &[Granularity::Section],
    ];

    #[test]
    fn every_compiled_tool_description_fits_the_budget_under_every_config() {
        for tool in TOOL_NAMES {
            for phrase in [false, true] {
                for effective in GRANULARITY_SUBSETS {
                    let desc = compose_tool_description(tool, phrase, effective, None).unwrap();
                    let len = desc.chars().count();
                    assert!(
                        len <= COMPILED_DESCRIPTION_BUDGET,
                        "`{tool}` description is {len} chars (phrase={phrase}, \
                         granularities={effective:?}), over the \
                         {COMPILED_DESCRIPTION_BUDGET}-char budget. Move detail onto the \
                         parameter it governs:\n{desc}"
                    );
                }
            }
        }
    }

    #[test]
    fn compiled_server_mechanics_fit_the_budget() {
        // The corpus-dependent remainder is budgeted together with this in
        // `server.rs`'s realistic-corpus test; this pins the static part.
        let len = compose_server_mechanics().chars().count();
        assert!(
            len <= COMPILED_DESCRIPTION_BUDGET,
            "server mechanics are {len} chars"
        );
    }

    #[test]
    fn warn_if_over_client_cap_remembers_the_last_warned_length_per_surface() {
        let surface = "test-only surface";
        let over = "x".repeat(CLIENT_DESCRIPTION_CAP + 1);
        warn_if_over_client_cap(surface, &over);
        assert_eq!(
            OVER_CAP_WARNED.lock().unwrap().get(surface),
            Some(&(CLIENT_DESCRIPTION_CAP + 1))
        );
        // Back under the cap re-arms the warning for this surface.
        warn_if_over_client_cap(surface, "short");
        assert!(OVER_CAP_WARNED.lock().unwrap().get(surface).is_none());
    }

    #[test]
    fn granularity_summary_names_only_enabled_values() {
        assert_eq!(
            granularity_summary(&ALL),
            "`granularity` chooses what each result is: `chunk`, `document` or `section`."
        );
        assert_eq!(
            granularity_summary(&[Granularity::Chunk, Granularity::Document]),
            "`granularity` chooses what each result is: `chunk` or `document`."
        );
        assert_eq!(
            granularity_summary(&[Granularity::Section]),
            "`granularity` is fixed to `section` on this server."
        );
    }

    #[test]
    fn granularity_description_tells_the_model_to_read_a_whole_document_row_whole() {
        // A `scope: whole_document` row carries the range indexed for it, which a
        // document edited since indexing no longer matches, so a `start_line`/
        // `end_line` read of it can be partial. Only a server that offers `section`
        // has such rows to explain.
        for effective in GRANULARITY_SUBSETS {
            let text = granularity_description(effective);
            if effective.contains(&Granularity::Section) {
                for part in [
                    "`scope: whole_document`",
                    "stale",
                    "read that document whole",
                ] {
                    assert!(text.contains(part), "{effective:?} lacks {part:?}: {text}");
                }
            } else {
                assert!(
                    !text.contains("whole_document"),
                    "{effective:?} must not mention section rows: {text}"
                );
            }
        }
    }
}
