use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::RwLock;

use chrono::{DateTime, NaiveDate};

use anyhow::Context as _;
use globset::{Glob, GlobSet, GlobSetBuilder};
use rmcp::{
    ErrorData as McpError, RoleServer, ServerHandler,
    handler::server::{router::tool::ToolRouter, tool::ToolCallContext, wrapper::Parameters},
    model::*,
    schemars,
    service::RequestContext,
    tool, tool_handler, tool_router,
};
use tracing::{debug, error, warn};

use crate::{
    config::{Granularity, ResolvedConfig},
    document_fields,
    embed::EmbedClient,
    git,
    qdrant::{IndexKind, QdrantStore},
    rerank::RerankClient,
    retrieval::{
        self, DocumentIndexDeps, GetDocumentError, RetrievalDeps, SearchFilters, SearchOptions,
    },
    schema::SchemaCache,
    state::{DocumentIndex, DocumentQuery, FieldFilter, OrderBy, StateDb},
    tool_schema, validate,
    write::{
        self, DirectoryMoveError, DirectoryMoveSuccess, FrontmatterEdit, WriteDeps, WriteError,
        WriteRequest, WriteSuccess,
    },
};

const MAX_QUERY_LEN: usize = 4096;
const MAX_PATH_LEN: usize = 4096;
const MAX_FILTER_STR_LEN: usize = 256;
/// Deepest dot-path `update_schema` will nest; see `build_schema_edit`.
const MAX_SCHEMA_PATH_SEGMENTS: usize = 16;
const MAX_CONTENT_LEN: usize = write::MAX_CONTENT_LEN;
/// Cap on the number of operations a single `write_document` `frontmatter_patch`
/// call may carry — a document write should rarely need more than a handful of
/// field edits at once; this bounds the work `write::apply_frontmatter_patch`
/// does per call, mirroring `MAX_SCHEMA_VALUES`'s reasoning for `update_schema`.
const MAX_FRONTMATTER_PATCH_OPS: usize = 20;
/// Cap on the `values` list of a single `add_values`/`remove_values`
/// `frontmatter_patch` operation.
const MAX_FRONTMATTER_PATCH_VALUES: usize = 200;
/// Cap on the serialized size of a single `frontmatter_patch` value —
/// mirrors `MAX_SCHEMA_DEFINITION_LEN`'s identical reasoning for `update_schema`'s
/// `set_field` definitions: a document's frontmatter is committed and re-parsed
/// on every read, so an oversized value is a durable cost, not a transient one.
const MAX_FRONTMATTER_VALUE_LEN: usize = 4 * 1024;

/// Resolve a caller's requested `limit` against the configured default and ceiling.
///
/// An over-large request is clamped rather than rejected — a caller asking for more
/// than the maximum gets the maximum, which is friendlier than an error and matches
/// the historical behaviour when both values were hardcoded.
fn resolve_limit(requested: Option<u64>, default_limit: u64, max_limit: u64) -> u64 {
    requested.unwrap_or(default_limit).min(max_limit)
}

/// Parse an ISO 8601 date/datetime string to a Unix timestamp (seconds).
///
/// Accepts RFC 3339 datetimes (e.g. `2024-01-15T12:00:00Z`) and date-only
/// strings (e.g. `2024-01-15`, interpreted as midnight UTC).
pub(crate) fn parse_date_to_timestamp(s: &str) -> Result<i64, String> {
    // Try RFC 3339 / ISO 8601 datetime first
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Ok(dt.timestamp());
    }
    // Fall back to date-only YYYY-MM-DD (treated as start of day UTC)
    if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date
            .and_hms_opt(0, 0, 0)
            .expect("midnight is always valid")
            .and_utc();
        return Ok(dt.timestamp());
    }
    Err(format!(
        "invalid date '{}': expected RFC 3339 (e.g. 2024-01-15T00:00:00Z) \
         or date-only (e.g. 2024-01-15)",
        s
    ))
}

/// How many invalidated documents to name before summarizing the rest.
const MAX_REPORTED_CASUALTIES: usize = 20;
/// Cap on permitted values a single `update_schema` call may add to a field.
const MAX_SCHEMA_VALUES: usize = 500;
/// Cap on the serialized size of a `set_field` definition.
const MAX_SCHEMA_DEFINITION_LEN: usize = 8 * 1024;
/// Cap on permitted values echoed back per field by `get_schema`.
const MAX_REPORTED_VALUES: usize = 200;
/// Cap on fields echoed back per scope by `get_schema`.
const MAX_REPORTED_FIELDS: usize = 500;
/// Most-used values listed per open field by `get_schema`'s `values_in_use`.
const MAX_VALUES_IN_USE: usize = 20;
/// Cap on the unified diff an edit's result carries (fix #129). Mirrors
/// `MAX_SCHEMA_DEFINITION_LEN`'s convention for bounding a single text blob:
/// `WriteSuccess::diff` is unbounded by design (a full replace of a large
/// document produces a large diff), but embedding it verbatim would let one
/// write emit a multi-megabyte tool result. See
/// [`capped_diff`] for the same unbounded-source/bounded-payload split
/// [`capped_casualties`] already applies to `update_schema`'s casualty list.
const MAX_STRUCTURED_DIFF_BYTES: usize = 8 * 1024;

/// Normalize a caller-supplied scope path into a safe KB-relative directory.
///
/// Rejects absolute paths and any `..` component — a schema written outside the KB
/// would govern nothing and could clobber unrelated files.
///
/// The result is rebuilt from the path's normal segments alone, so a `.` segment or a
/// doubled separator (`food/./recipes`, `food//recipes`) never reaches a scope label, a
/// commit message or the `LIKE` pattern `get_schema` counts values with: each is plain
/// `food/recipes`. A lone `.` (or `./`) is the KB root, like an empty path.
fn normalize_scope_path(raw: &str) -> Result<std::path::PathBuf, McpError> {
    use std::path::{Component, PathBuf};

    let trimmed = raw.trim().trim_matches('/');
    if trimmed.is_empty() {
        return Ok(PathBuf::new());
    }
    if trimmed.len() > MAX_PATH_LEN {
        return Err(McpError::invalid_params(
            format!(
                "path too long: {} chars (max {})",
                trimmed.len(),
                MAX_PATH_LEN
            ),
            None,
        ));
    }

    let candidate = PathBuf::from(trimmed);
    if candidate.is_absolute() {
        return Err(McpError::invalid_params(
            format!("path must be relative to the knowledge-base root, got '{raw}'"),
            None,
        ));
    }
    let mut normalized = PathBuf::new();
    for component in candidate.components() {
        match component {
            Component::Normal(segment) => normalized.push(segment),
            // Only a leading `.` survives `components()`; it names no directory.
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(McpError::invalid_params(
                    format!("path must not contain '..' or absolute segments, got '{raw}'"),
                    None,
                ));
            }
        }
    }

    Ok(normalized)
}

/// Keys a `set_field` definition accepts, as the errors that reject a malformed one name
/// them: `schema::RawFieldDef`'s advertised properties, kept in one place so those errors
/// all say the same thing. The deprecated `extend` is still accepted but is not
/// advertised, so it is not listed here either.
const FIELD_DEFINITION_KEYS: &str =
    "`type`, `required`, `indexed`, `values`, `default`, `open`, `fields`";

/// A `set_field` definition as delivered by an MCP client.
///
/// The `update_schema` tool schema advertises this parameter as the plain JSON object
/// described by [`crate::schema::RawFieldDef`] — the same shape a schema file's
/// entry uses. That's a deliberate fix: `serde_json::Value` (the old type here) produces
/// no `type` constraint at all in the advertised schema, and at least one real MCP
/// client responded to that ambiguity by sending the definition as a JSON-encoded
/// *string* instead of an object, which the old handler rejected with an error naming a
/// Rust struct the caller has no way to act on.
///
/// This type's [`Deserialize`](serde::Deserialize) impl still tolerates that string
/// form as a runtime fallback — some clients stringify nested-object arguments
/// regardless of what the schema says — but the *advertised* schema is not widened to
/// document it: a `oneOf: [object, string]` schema would just reopen the same
/// ambiguity for clients that DO read it. A conforming client only ever needs to send
/// the object.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldDefinitionInput(pub crate::schema::RawFieldDef);

impl<'de> serde::Deserialize<'de> for FieldDefinitionInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        parse_field_definition(value)
            .map(FieldDefinitionInput)
            .map_err(serde::de::Error::custom)
    }
}

// Delegate schema generation to `RawFieldDef`'s own derived `JsonSchema` impl rather
// than hand-duplicating its fields here, so the advertised shape and the accepted shape
// can never drift apart. This is what actually fixes the bug: the tool schema now
// advertises a real object with named, typed properties instead of `{}`.
impl schemars::JsonSchema for FieldDefinitionInput {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        crate::schema::RawFieldDef::schema_name()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        crate::schema::RawFieldDef::schema_id()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        crate::schema::RawFieldDef::json_schema(generator)
    }
}

/// Parse a `set_field` definition from JSON: a JSON object directly, or (see
/// [`FieldDefinitionInput`]) a string containing one. Every error names the expected
/// shape in caller-facing terms — never a bare Rust type name, which means nothing to
/// an MCP client on the other end of the wire.
fn parse_field_definition(value: serde_json::Value) -> Result<crate::schema::RawFieldDef, String> {
    use serde_json::Value;

    let object = match value {
        Value::Object(_) => value,
        Value::String(s) => match serde_json::from_str::<Value>(&s) {
            Ok(parsed @ Value::Object(_)) => parsed,
            Ok(other) => return Err(definition_shape_error(&other)),
            Err(e) => {
                return Err(format!(
                    "field definition must be a JSON object with keys \
                     {FIELD_DEFINITION_KEYS}. A JSON string containing that object is \
                     also accepted, but this string is not valid JSON: {e}"
                ));
            }
        },
        other => return Err(definition_shape_error(&other)),
    };

    serde_json::from_value(object).map_err(|e| format!("invalid field definition: {e}"))
}

/// Build the "wrong shape entirely" error for [`parse_field_definition`], naming what
/// was actually received without echoing its (possibly large) content.
fn definition_shape_error(value: &serde_json::Value) -> String {
    format!(
        "field definition must be a JSON object with keys {FIELD_DEFINITION_KEYS}, got {}",
        json_value_kind(value)
    )
}

/// Describe a JSON value's kind in a few words, for error messages.
fn json_value_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// `search`'s `filters` parameter as delivered by an MCP client.
///
/// Same fix as [`FieldDefinitionInput`], applied to the same shape of bug: the old
/// type here, `Option<serde_json::Map<String, serde_json::Value>>`, advertises no
/// schema constraint at all (schemars emits `{}` for a bare `serde_json::Map`), so a
/// calling model has no way to learn from the tool schema that a scalar means
/// equality, an array means any-of, or that an object accepts
/// `any_of`/`all_of`/`gte`/`lte`/`gt`/`lt`. It has to learn that from prose alone —
/// and, per the same failure mode `FieldDefinitionInput` exists to cover, at least
/// one client class responds to an under-specified object parameter by sending it
/// JSON-encoded as a string instead, which the old `Option<Map<...>>` field rejected
/// with a raw deserialize error rather than a caller-actionable one.
///
/// Unlike `FieldDefinitionInput`, there is no existing typed Rust struct to delegate
/// to for the advertised schema — `filters` is a map keyed by arbitrary caller-chosen
/// field names, each valued by one of several shapes — so [`json_schema`] below
/// builds that schema by hand instead of deriving it. Actual per-condition parsing
/// still happens in `parse_field_filter`, unchanged: this type only fixes what the
/// tool schema advertises and adds the same string-tolerance fallback, it does not
/// duplicate that function's shape checking or its field-named error messages.
///
/// [`json_schema`]: schemars::JsonSchema::json_schema
#[derive(Debug, Clone, PartialEq)]
pub struct SearchFiltersInput(pub serde_json::Map<String, serde_json::Value>);

impl<'de> serde::Deserialize<'de> for SearchFiltersInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        parse_search_filters_input(value)
            .map(SearchFiltersInput)
            .map_err(serde::de::Error::custom)
    }
}

/// Parse `search`'s `filters` argument from JSON: an object directly, or (see
/// [`SearchFiltersInput`]) a string containing one. Mirrors
/// [`parse_field_definition`]'s two-shape acceptance and error style.
fn parse_search_filters_input(
    value: serde_json::Value,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    use serde_json::Value;

    match value {
        Value::Object(map) => Ok(map),
        Value::String(s) => match serde_json::from_str::<Value>(&s) {
            Ok(Value::Object(map)) => Ok(map),
            Ok(other) => Err(filters_shape_error(&other)),
            Err(e) => Err(format!(
                "filters must be a JSON object mapping frontmatter field names to \
                 conditions. A JSON string containing that object is also accepted, \
                 but this string is not valid JSON: {e}"
            )),
        },
        other => Err(filters_shape_error(&other)),
    }
}

/// Build the "wrong shape entirely" error for [`parse_search_filters_input`], naming
/// what was actually received without echoing its (possibly large) content.
fn filters_shape_error(value: &serde_json::Value) -> String {
    format!(
        "filters must be a JSON object mapping frontmatter field names to conditions, \
         got {}",
        json_value_kind(value)
    )
}

impl schemars::JsonSchema for SearchFiltersInput {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "SearchFilters".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::SearchFiltersInput"))
    }

    // Hand-built rather than derived (see this type's doc comment): describes an
    // object whose values ("filter conditions") are, per field, a scalar
    // (equality), an array of scalars (any-of), or an object carrying
    // `any_of`/`all_of`/`gte`/`lte`/`gt`/`lt` — exactly the shapes
    // `parse_field_filter` accepts. Kept tight and caller-facing (see #126): no
    // implementation rationale leaks into the emitted schema, only the shape a
    // caller needs to construct a valid `filters` value. The prose lives once,
    // on the `filters` property (`SearchParams::filters`); the structure here
    // carries none of its own.
    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "object",
            "additionalProperties": {
                "anyOf": [
                    { "type": ["string", "number", "boolean"] },
                    {
                        "type": "array",
                        "items": { "type": ["string", "number", "boolean"] }
                    },
                    {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "any_of": {
                                "type": "array",
                                "items": { "type": ["string", "number", "boolean"] }
                            },
                            "all_of": {
                                "type": "array",
                                "items": { "type": ["string", "number", "boolean"] }
                            },
                            "gte": { "type": "number" },
                            "lte": { "type": "number" },
                            "gt": { "type": "number" },
                            "lt": { "type": "number" }
                        }
                    }
                ]
            }
        })
    }
}

/// Turn tool parameters into a typed schema edit.
fn build_schema_edit(params: &UpdateSchemaParams) -> Result<crate::schema::SchemaEdit, McpError> {
    use crate::schema::SchemaEdit;
    let invalid = |msg: String| McpError::invalid_params(msg, None);

    // A schema file is committed, pushed, and re-parsed on every cache build, so an
    // oversized one is a durable cost rather than a transient one. Bound the inputs
    // here, mirroring the content cap the document write tools enforce.
    if params.field.len() > MAX_FILTER_STR_LEN {
        return Err(invalid(format!(
            "field name too long: {} chars (max {})",
            params.field.len(),
            MAX_FILTER_STR_LEN
        )));
    }
    // `field` is a dot-path split into nested containers. Each segment becomes two
    // levels of YAML nesting, so cap the depth well under the parser's recursion limit
    // rather than let the round-trip check fail with a 500.
    if params.field.split('.').count() > MAX_SCHEMA_PATH_SEGMENTS {
        return Err(invalid(format!(
            "field path too deep: more than {MAX_SCHEMA_PATH_SEGMENTS} dot-separated segments"
        )));
    }
    // An empty or whitespace-padded segment ("a..b", "a.", "a. b") would create a key no
    // frontmatter field can match.
    if params
        .field
        .split('.')
        .any(|seg| seg.is_empty() || seg != seg.trim())
    {
        return Err(invalid(format!(
            "invalid field path '{}': every dot-separated segment must be non-empty with no surrounding whitespace",
            params.field
        )));
    }
    if let Some(values) = &params.values {
        if values.len() > MAX_SCHEMA_VALUES {
            return Err(invalid(format!(
                "too many values: {} (max {})",
                values.len(),
                MAX_SCHEMA_VALUES
            )));
        }
        if let Some(long) = values.iter().find(|v| v.len() > MAX_FILTER_STR_LEN) {
            return Err(invalid(format!(
                "value too long: {} chars (max {})",
                long.len(),
                MAX_FILTER_STR_LEN
            )));
        }
    }
    if let Some(definition) = &params.definition {
        // Measured on the parsed-and-reserialized form rather than whatever bytes the
        // client happened to send: that's what actually gets committed to the schema
        // file (via `SchemaFile::to_yaml`), and it makes the cap apply identically
        // whether the definition arrived as an object or as the string fallback.
        let size = serde_json::to_string(&definition.0)
            .map(|s| s.len())
            .unwrap_or(usize::MAX);
        if size > MAX_SCHEMA_DEFINITION_LEN {
            return Err(invalid(format!(
                "field definition too large (max {} bytes)",
                MAX_SCHEMA_DEFINITION_LEN
            )));
        }
    }

    let values = || -> Result<Vec<String>, McpError> {
        params
            .values
            .clone()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| {
                invalid(format!(
                    "'{}' requires a non-empty values list",
                    params.operation
                ))
            })
    };

    match params.operation.trim().to_ascii_lowercase().as_str() {
        "add_values" => Ok(SchemaEdit::AddValues {
            field: params.field.clone(),
            values: values()?,
        }),
        "remove_values" => Ok(SchemaEdit::RemoveValues {
            field: params.field.clone(),
            values: values()?,
        }),
        "set_field" => {
            // Parsing already happened when the tool call's arguments were
            // deserialized into `UpdateSchemaParams` (see `FieldDefinitionInput`), so
            // there's nothing left to do here but unwrap it.
            let definition = params
                .definition
                .clone()
                .ok_or_else(|| invalid("'set_field' requires a definition".into()))?;
            Ok(SchemaEdit::SetField {
                field: params.field.clone(),
                definition: Box::new(definition.0),
            })
        }
        "remove_field" => Ok(SchemaEdit::RemoveField {
            field: params.field.clone(),
        }),
        other => Err(invalid(format!(
            "unknown operation '{other}': expected add_values, remove_values, set_field, \
             or remove_field"
        ))),
    }
}

/// Resolve a possibly-partial scope reference to exactly one directory.
///
/// Mirrors `get_document`'s contract: an exact match wins, several matches are an
/// explicit ambiguity error rather than a guess, and none is a not-found error naming
/// what does exist.
fn resolve_scope_reference(
    schemas: &SchemaCache,
    requested: &std::path::Path,
) -> Result<std::path::PathBuf, McpError> {
    let matches = schemas.match_scope_dirs(requested);
    match matches.len() {
        // No scope declares its own schema here; the path still resolves through the
        // cascade to whatever ancestor governs it.
        0 => Ok(requested.to_path_buf()),
        1 => Ok(matches.into_iter().next().expect("length checked")),
        _ => Err(McpError::invalid_params(
            format!(
                "'{}' matches {} scopes: {}. Use a more specific path.",
                requested.display(),
                matches.len(),
                matches
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            None,
        )),
    }
}

/// Strip control characters and cap length anywhere inside a value echoed back to an
/// agent. Defaults come from schema files in a synced repo, so an array or object
/// default is just as attacker-controlled as a string one.
fn sanitize_reflected_value(value: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match value {
        Value::String(s) => Value::String(crate::server::sanitize_facet_value(s)),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .take(MAX_REPORTED_VALUES)
                .map(sanitize_reflected_value)
                .collect(),
        ),
        Value::Object(map) => Value::Object(
            map.iter()
                .take(MAX_REPORTED_VALUES)
                .map(|(k, v)| {
                    (
                        crate::server::sanitize_facet_value(k),
                        sanitize_reflected_value(v),
                    )
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Render a casualty list for a human-readable message.
/// Add an `update_schema` casualty list under `key` (`invalidated` after a
/// forced change, `would_invalidate` on a dry run) — only when non-empty — with
/// `casualties_total`/`casualties_truncated` only when the list was capped.
fn insert_casualties(
    response: &mut serde_json::Value,
    key: &str,
    capped: Vec<serde_json::Value>,
    total: usize,
    truncated: bool,
) {
    if capped.is_empty() {
        return;
    }
    response[key] = serde_json::Value::Array(capped);
    if truncated {
        response["casualties_total"] = serde_json::json!(total);
        response["casualties_truncated"] = serde_json::json!(true);
    }
}

/// One resolved field definition as `get_schema` and an `update_schema` dry run
/// report it. Only what constrains a document is sent: `type` and `values`
/// when set, `required`/`indexed` when true, `default` when there is one,
/// `open: false` on an object that refuses undeclared keys (`true` is the
/// default), and `declared_in` — the scope directory that last set it (`/` for
/// the root).
fn field_def_json(def: &crate::schema::FieldDef, origin: Option<&str>) -> serde_json::Value {
    let mut entry = serde_json::Map::new();
    if let Some(ty) = def.ty {
        entry.insert(
            "type".into(),
            serde_json::json!(format!("{ty:?}").to_lowercase()),
        );
    }
    if def.required {
        entry.insert("required".into(), serde_json::json!(true));
    }
    if def.indexed {
        entry.insert("indexed".into(), serde_json::json!(true));
    }
    if let Some(values) = &def.values {
        entry.insert(
            "values".into(),
            values
                .iter()
                .take(MAX_REPORTED_VALUES)
                .map(|v| serde_json::json!(crate::server::sanitize_facet_value(v)))
                .collect(),
        );
    }
    if let Some(default) = &def.default {
        entry.insert("default".into(), sanitize_reflected_value(default));
    }
    if def.ty == Some(crate::schema::FieldType::Object) && !def.open {
        entry.insert("open".into(), serde_json::json!(false));
    }
    if let Some(origin) = origin {
        entry.insert(
            "declared_in".into(),
            serde_json::json!(crate::server::sanitize_facet_value(origin)),
        );
    }
    serde_json::Value::Object(entry)
}

fn render_casualties(casualties: &[serde_json::Value]) -> String {
    let mut out = String::new();
    for entry in casualties.iter().take(MAX_REPORTED_CASUALTIES) {
        out.push_str(&format!(
            "  - {}: {}\n",
            entry["path"].as_str().unwrap_or("?"),
            entry["reason"].as_str().unwrap_or("?")
        ));
    }
    if casualties.len() > MAX_REPORTED_CASUALTIES {
        out.push_str(&format!(
            "  … and {} more\n",
            casualties.len() - MAX_REPORTED_CASUALTIES
        ));
    }
    out
}

/// Cap a casualty list for a tool result, mirroring the cap
/// [`render_casualties`] applies to a refusal's message.
///
/// `documents_broken_by` deliberately returns the *complete* casualty list — the
/// force/refuse decision needs completeness, so that query stays unbounded — but
/// embedding the full `Vec` verbatim let a schema tightening that broke thousands
/// of documents emit a multi-megabyte tool result (#148). Returns the capped list alongside the true total and whether it was truncated,
/// the same `total`/`has_more` shape `search` uses elsewhere in this file, so a
/// client that only reads `structured_content` can still tell "empty" from
/// "truncated" rather than only ever seeing the first page.
fn capped_casualties(casualties: &[serde_json::Value]) -> (Vec<serde_json::Value>, usize, bool) {
    let total = casualties.len();
    let capped = casualties
        .iter()
        .take(MAX_REPORTED_CASUALTIES)
        .cloned()
        .collect();
    (capped, total, total > MAX_REPORTED_CASUALTIES)
}

/// Cap `diff` (a `WriteSuccess`/`DirectoryMoveSuccess` unified diff) for
/// `structured_content` — see [`MAX_STRUCTURED_DIFF_BYTES`]'s doc comment.
/// Returns `(capped_diff, diff_truncated, diff_total_bytes)`, the same
/// capped-value/truncated-flag/true-total shape [`capped_casualties`] returns,
/// so a client reading only `structured_content` can tell "the whole diff" from
/// "cut off, and by how much" rather than only ever seeing the first slice.
///
/// Cuts at a `char_boundary` at or before the byte cap — `diff` is arbitrary
/// document text, so a naive byte-index cut could otherwise land inside a
/// multi-byte UTF-8 character and produce an invalid `&str` slice.
fn capped_diff(diff: &str) -> (String, bool, usize) {
    let total = diff.len();
    if total <= MAX_STRUCTURED_DIFF_BYTES {
        return (diff.to_string(), false, total);
    }
    let mut end = MAX_STRUCTURED_DIFF_BYTES;
    while end > 0 && !diff.is_char_boundary(end) {
        end -= 1;
    }
    (diff[..end].to_string(), true, total)
}

/// Default page size for `list_documents` — well above `search`'s cap, since
/// enumeration is the point.
const DEFAULT_LIST_LIMIT: u64 = 100;
/// Hard cap on a single `list_documents` page.
const MAX_LIST_LIMIT: u64 = 1000;
/// Cap on how many filter fields one call may specify.
const MAX_LIST_FILTERS: usize = 20;
/// Cap on values within a single field's filter, so one call cannot generate an
/// unbounded number of bound SQL parameters.
const MAX_FILTER_VALUES: usize = 500;

/// Translate one JSON filter value into a typed [`FieldFilter`].
///
/// Accepts a scalar for equality, an array for any-of, or an object carrying
/// `any_of` / `all_of` / `gte` / `lte` / `gt` / `lt`.
fn parse_field_filter(field: &str, raw: &serde_json::Value) -> Result<FieldFilter, String> {
    use serde_json::Value;

    // Scalar values go through the same canonicalization as the write path, so a JSON
    // `false` matches a stored boolean and `45` matches a stored integer. Also caps
    // each value's length — master enforced this per-value (domain_too_long_is_rejected
    // et al.); folding those scalar params into this generic `filters` map must not
    // lose it, since an unbounded value is still a durable cost against Qdrant/SQLite
    // regardless of which filter form carried it in.
    let canonical = |value: &Value| -> Result<String, String> {
        let text = document_fields::canonical_text(value).ok_or_else(|| {
            format!(
                "filter '{}': expected a string, number, or boolean, got {}",
                field, value
            )
        })?;
        if text.len() > MAX_FILTER_STR_LEN {
            return Err(format!(
                "filter '{}': value too long ({} chars, max {})",
                field,
                text.len(),
                MAX_FILTER_STR_LEN
            ));
        }
        Ok(text)
    };

    // One place enforces the value cap, so no filter form can slip past it. `all_of`
    // in particular compiles to one correlated subquery per value, so an uncapped list
    // is a query-complexity attack, not merely a large response.
    let values_of = |items: &[Value]| -> Result<Vec<String>, String> {
        if items.len() > MAX_FILTER_VALUES {
            return Err(format!(
                "filter '{}': too many values ({}, max {})",
                field,
                items.len(),
                MAX_FILTER_VALUES
            ));
        }
        items.iter().map(&canonical).collect()
    };

    match raw {
        Value::String(_) | Value::Number(_) | Value::Bool(_) => {
            Ok(FieldFilter::AnyOf(vec![canonical(raw)?]))
        }
        Value::Array(items) => Ok(FieldFilter::AnyOf(values_of(items)?)),
        Value::Object(map) => {
            let number = |key: &str| -> Result<Option<f64>, String> {
                match map.get(key) {
                    None | Some(Value::Null) => Ok(None),
                    Some(Value::Number(n)) => Ok(n.as_f64()),
                    Some(other) => Err(format!(
                        "filter '{}': '{}' must be a number, got {}",
                        field, key, other
                    )),
                }
            };

            let known = ["any_of", "all_of", "gte", "lte", "gt", "lt"];
            if let Some(unknown) = map.keys().find(|k| !known.contains(&k.as_str())) {
                return Err(format!(
                    "filter '{}': unknown operator '{}'; expected one of {}",
                    field,
                    unknown,
                    known.join(", ")
                ));
            }

            // Set matching and range matching are separate modes. Accepting a mix and
            // honoring only one silently returns a broader result set than the caller
            // asked for, which is exactly the class of silent-wrong-answer this tool
            // exists to eliminate — so reject it rather than pick a winner.
            let has_set = map.contains_key("any_of") || map.contains_key("all_of");
            let has_range = ["gte", "lte", "gt", "lt"]
                .iter()
                .any(|k| map.contains_key(*k));
            if has_set && has_range {
                return Err(format!(
                    "filter '{}': cannot combine set matching (any_of/all_of) with a \
                     numeric range (gte/lte/gt/lt); use one or the other",
                    field
                ));
            }
            if map.contains_key("any_of") && map.contains_key("all_of") {
                return Err(format!(
                    "filter '{}': specify either any_of or all_of, not both",
                    field
                ));
            }

            if let Some(values) = map.get("all_of") {
                let items = values
                    .as_array()
                    .ok_or_else(|| format!("filter '{}': 'all_of' must be an array", field))?;
                if items.is_empty() {
                    return Err(format!("filter '{}': 'all_of' must not be empty", field));
                }
                return Ok(FieldFilter::AllOf(values_of(items)?));
            }

            if let Some(values) = map.get("any_of") {
                let items = values
                    .as_array()
                    .ok_or_else(|| format!("filter '{}': 'any_of' must be an array", field))?;
                return Ok(FieldFilter::AnyOf(values_of(items)?));
            }

            let (gte, lte, gt, lt) = (number("gte")?, number("lte")?, number("gt")?, number("lt")?);
            if gte.is_none() && lte.is_none() && gt.is_none() && lt.is_none() {
                return Err(format!(
                    "filter '{}': object filters need at least one of {}",
                    field,
                    known.join(", ")
                ));
            }
            Ok(FieldFilter::Range { gte, lte, gt, lt })
        }
        Value::Null => Err(format!(
            "filter '{}': null is not a filter; omit the field instead",
            field
        )),
    }
}

/// Cap on the fields or values a refused `search` filter lists
/// (`KbSearchServer::check_filter_vocabulary`).
const MAX_FILTER_OPTIONS_LISTED: usize = 30;

/// The first [`MAX_FILTER_OPTIONS_LISTED`] of `items`.
fn capped_vec(items: &[String]) -> Vec<&String> {
    items.iter().take(MAX_FILTER_OPTIONS_LISTED).collect()
}

/// `items` joined for an error message, capped at
/// [`MAX_FILTER_OPTIONS_LISTED`] with a `(+N more)` tail.
fn capped_list(items: &[String]) -> String {
    let mut joined = items
        .iter()
        .take(MAX_FILTER_OPTIONS_LISTED)
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(", ");
    let overflow = items.len().saturating_sub(MAX_FILTER_OPTIONS_LISTED);
    if overflow > 0 {
        joined.push_str(&format!(" (+{overflow} more)"));
    }
    joined
}

/// Filter keys that need no schema declaration: the promoted `title` and
/// `description`, the fields ingest derives (`ingest::DERIVED_FIELDS`), and
/// the configured indexed fields.
fn builtin_filter_field(field: &str, config: &ResolvedConfig) -> bool {
    matches!(field, "title" | "description")
        || crate::ingest::DERIVED_FIELDS.contains(&field)
        || config.effective_indexed_fields().iter().any(|f| f == field)
}

/// Parse a `search` call's raw `filters` map into the typed representation shared by
/// both backends: SQLite (`state::StateDb::push_where`, enumeration mode) and Qdrant
/// (`qdrant::lower_field_filters`, query mode).
fn parse_filters(
    raw_filters: &Option<SearchFiltersInput>,
) -> Result<Vec<(String, FieldFilter)>, McpError> {
    let invalid = |msg: String| McpError::invalid_params(msg, None);

    let mut filters = Vec::new();
    if let Some(raw_filters) = raw_filters {
        let raw_filters = &raw_filters.0;
        if raw_filters.len() > MAX_LIST_FILTERS {
            return Err(invalid(format!(
                "too many filters: {} (max {})",
                raw_filters.len(),
                MAX_LIST_FILTERS
            )));
        }
        for (field, raw) in raw_filters {
            if field.len() > MAX_FILTER_STR_LEN {
                return Err(invalid(format!(
                    "filter field name too long: {} chars (max {})",
                    field.len(),
                    MAX_FILTER_STR_LEN
                )));
            }
            filters.push((
                field.clone(),
                parse_field_filter(field, raw).map_err(invalid)?,
            ));
        }
        // Deterministic order keeps generated SQL stable across calls.
        filters.sort_by(|a, b| a.0.cmp(&b.0));
    }
    Ok(filters)
}

/// Shared length check for `path_prefix`, used by both enumeration mode
/// ([`build_document_query`]) and query mode ([`KbSearchServer::search`]).
fn validate_path_prefix(path_prefix: &Option<String>) -> Result<(), McpError> {
    if let Some(prefix) = path_prefix
        && prefix.len() > MAX_FILTER_STR_LEN
    {
        return Err(McpError::invalid_params(
            format!(
                "path_prefix too long: {} chars (max {})",
                prefix.len(),
                MAX_FILTER_STR_LEN
            ),
            None,
        ));
    }
    Ok(())
}

/// Shared count check for `fields`, used by both enumeration mode
/// ([`build_document_query`]) and query+document mode ([`KbSearchServer::search`]).
fn validate_fields_count(fields: &Option<Vec<String>>) -> Result<(), McpError> {
    if let Some(fields) = fields
        && fields.len() > MAX_LIST_FILTERS
    {
        return Err(McpError::invalid_params(
            format!(
                "too many fields requested: {} (max {})",
                fields.len(),
                MAX_LIST_FILTERS
            ),
            None,
        ));
    }
    Ok(())
}

/// Build a validated [`DocumentQuery`] from tool parameters — the `search` tool's
/// document-granularity, no-query (enumeration) combination.
fn build_document_query(params: &SearchParams) -> Result<DocumentQuery, McpError> {
    let invalid = |msg: String| McpError::invalid_params(msg, None);

    let filters = parse_filters(&params.filters)?;

    let order_by = match &params.order_by {
        Some(raw) => OrderBy::parse(raw).map_err(invalid)?,
        None => OrderBy::default(),
    };

    validate_fields_count(&params.fields)?;
    validate_path_prefix(&params.path_prefix)?;

    let mtime_after = params
        .modified_after
        .as_deref()
        .map(parse_date_to_timestamp)
        .transpose()
        .map_err(invalid)?;
    let mtime_before = params
        .modified_before
        .as_deref()
        .map(parse_date_to_timestamp)
        .transpose()
        .map_err(invalid)?;

    Ok(DocumentQuery {
        filters,
        // #182: normalized through the same function the query-mode resolution uses,
        // so one needle means one thing in both modes. `push_where` applies the
        // substring `LIKE` that `DocumentIndex::paths_matching` applies for query
        // mode, rather than resolving to a path list enumeration would only re-query.
        path_prefix: retrieval::normalize_path_needle(params.path_prefix.as_deref())
            .map(str::to_string),
        order_by,
        order_desc: params.descending.unwrap_or(false),
        limit: params
            .limit
            .unwrap_or(DEFAULT_LIST_LIMIT)
            .clamp(1, MAX_LIST_LIMIT),
        offset: params.offset.unwrap_or(0),
        fields: params.fields.clone(),
        mtime_after,
        mtime_before,
    })
}

/// Parse `search`'s `filters` for query mode (chunk or grouped-document
/// granularity, both against Qdrant) and lower them to Qdrant conditions.
///
/// Every named field must carry a payload index — checked here against
/// [`crate::qdrant::all_indexed_fields`] (the union of schema-declared and legacy
/// config-declared fields, i.e. the real Qdrant index set) and rejected by name
/// otherwise: Qdrant can filter an unindexed field, just by a full scan rather than
/// an index, and silently doing that would return more than what was asked for.
/// This is the one piece of validation [`crate::qdrant::lower_field_filters`]
/// deliberately leaves to its caller — see that function's doc comment.
fn build_query_conditions(
    params: &SearchParams,
    config: &ResolvedConfig,
    schemas: &SchemaCache,
) -> Result<Vec<qdrant_client::qdrant::Condition>, McpError> {
    let filters = parse_filters(&params.filters)?;

    let indexed: std::collections::HashMap<String, IndexKind> =
        crate::qdrant::all_indexed_fields(config, schemas)
            .into_iter()
            .map(|f| (f.name, f.kind))
            .collect();

    if let Some((field, _)) = filters.iter().find(|(f, _)| !indexed.contains_key(f)) {
        return Err(McpError::invalid_params(
            format!(
                "filter '{field}' is not indexed for Qdrant queries; mark it `indexed: true` \
                 in the governing directory's schema (update_schema) to filter on it \
                 with a search query"
            ),
            None,
        ));
    }

    let mut conditions = crate::qdrant::lower_field_filters(&filters, &indexed)
        .map_err(|e| McpError::invalid_params(e, None))?;

    if let Some(condition) = heading_prefix_condition(&params.heading_prefix, config)? {
        conditions.push(condition);
    }

    Ok(conditions)
}

/// Lower `search`'s `heading_prefix` parameter to a Qdrant condition (#286):
/// `Condition::matches(HEADING_PREFIXES_KEY, joined)`, where `joined` is
/// `heading::heading_prefix_key` — matching against every chunk's own precomputed
/// `heading_prefixes` array (`ingest::derive_heading_prefixes`, normalized the
/// same way), which is what makes this a PREFIX restriction rather than an
/// exact-depth match. `None` in gives `None` out (nothing to filter on). An
/// empty list, a blank segment (empty once normalized, so one made only of
/// whitespace or invisible characters), or any list while `chunking.heading_metadata`
/// is off (the payload field this filters on is never written in that case) is
/// a caller error — not a silent no-op that would otherwise look like "matches
/// everything" or "matches nothing".
fn heading_prefix_condition(
    heading_prefix: &Option<Vec<String>>,
    config: &ResolvedConfig,
) -> Result<Option<qdrant_client::qdrant::Condition>, McpError> {
    let Some(heading_prefix) = heading_prefix else {
        return Ok(None);
    };
    if heading_prefix.is_empty() {
        return Err(McpError::invalid_params(
            "heading_prefix must not be empty — omit it entirely to search without a \
             heading restriction"
                .to_string(),
            None,
        ));
    }
    if heading_prefix
        .iter()
        .any(|s| crate::heading::normalize_heading_text(s).is_empty())
    {
        return Err(McpError::invalid_params(
            "heading_prefix must not contain empty or blank segments (a segment of only \
             whitespace or invisible characters is blank)"
                .to_string(),
            None,
        ));
    }
    if !config.chunking.heading_metadata {
        return Err(McpError::invalid_params(
            HEADING_METADATA_OFF_HEADING_PREFIX.to_string(),
            None,
        ));
    }
    Ok(Some(qdrant_client::qdrant::Condition::matches(
        crate::qdrant::HEADING_PREFIXES_KEY,
        crate::heading::heading_prefix_key(heading_prefix),
    )))
}

/// Rejection for `heading_prefix` while `chunking.heading_metadata` is off.
/// Caller-facing, so it names neither the config key nor payload fields: it
/// says what is missing and what the caller (or the server's operator) can do
/// about it. The operator gets the config-key detail from the docs.
const HEADING_METADATA_OFF_HEADING_PREFIX: &str = "heading_prefix is unavailable: this server does not have heading metadata enabled. \
     Search without heading_prefix, or ask the server's operator to enable heading metadata";

/// Rejection for `section` granularity while `chunking.heading_metadata` is
/// off. Shared by `granularity_disabled_error` and `search_sections` so a
/// caller sees one explanation for one cause. Caller-facing in the same way as
/// [`HEADING_METADATA_OFF_HEADING_PREFIX`].
const HEADING_METADATA_OFF_SECTION: &str = "section granularity is unavailable: this server does not have heading metadata \
     enabled. Use another granularity, or ask the server's operator to enable heading metadata";

/// The tail [`granularity_disabled_error`] appends when `section` is missing
/// only because heading metadata is off — [`HEADING_METADATA_OFF_SECTION`]'s
/// explanation, worded to follow "granularity 'section' is not enabled on
/// this server" without repeating it.
const HEADING_METADATA_OFF_SECTION_REASON: &str = ", because this server does not have heading \
     metadata enabled. Use another granularity, or ask the server's operator to enable heading \
     metadata";

/// Appended to `section`-granularity and `heading_prefix`-filtered results
/// while a multi-file indexing run is in flight: heading payload fields are
/// written per document as the run reaches it, so e.g. right after
/// `chunking.heading_metadata` is enabled the corpus is only partly covered.
/// `None` otherwise — including during a routine single-file reindex, which
/// can't leave other documents' metadata incomplete (see
/// `status::IndexStatus::is_bulk_indexing`). Cheap — one read lock on
/// `status::INDEX_STATUS`.
fn heading_results_indexing_note() -> Option<&'static str> {
    crate::status::INDEX_STATUS.is_bulk_indexing().then_some(
        "Note: an indexing run is in progress; documents it has not reached yet may lack \
         heading metadata, so these results may be incomplete.",
    )
}

/// Mark a finished `search` response `indexing_in_progress: true` when
/// [`heading_results_indexing_note`] applies.
fn annotate_heading_results(response: &mut serde_json::Value, note: Option<&str>) {
    if note.is_some()
        && let serde_json::Value::Object(map) = response
    {
        map.insert(
            "indexing_in_progress".to_string(),
            serde_json::Value::Bool(true),
        );
    }
}

/// Parse a caller-supplied `granularity`, rejecting anything unrecognized. The
/// rejection lists only `effective` — the granularities this deployment
/// enables — so a disabled one is never advertised to a caller (#286).
fn parse_granularity(raw: &str, effective: &[Granularity]) -> Result<Granularity, McpError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "chunk" => Ok(Granularity::Chunk),
        "document" => Ok(Granularity::Document),
        "section" => Ok(Granularity::Section),
        other => Err(McpError::invalid_params(
            format!(
                "unknown granularity '{other}': expected one of {}",
                effective
                    .iter()
                    .map(|g| format!("'{g}'"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            None,
        )),
    }
}

/// The granularity a `search` call gets when it names none: `chunk` when a query
/// is present and `document` when it is not — reproducing the old `search` and
/// `list_documents` tools' behaviour exactly. Pure and I/O-free so the
/// default-in-both-directions property is unit-testable without a live index.
fn default_granularity(query_present: bool) -> Granularity {
    if query_present {
        Granularity::Chunk
    } else {
        Granularity::Document
    }
}

/// Whether a granularity can serve a `search` call in the given query mode.
/// `chunk` and `section` both require a query — `search_chunks`/
/// `search_sections` reject its absence outright (see the `(false, ...)`
/// arms in `search`'s match) — while `document` serves either mode
/// (enumeration without a query, grouped results with one). Shared by
/// `resolve_search_granularity`'s fallback walk below so "supports the
/// mode" has exactly one definition.
fn granularity_supports_mode(g: Granularity, query_present: bool) -> bool {
    query_present || g.supports_no_query()
}

/// `effective`, as a comma-separated list for error messages.
fn join_granularities(effective: &[Granularity]) -> String {
    effective
        .iter()
        .map(|g| g.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Rejection for `fields` at a granularity whose results carry no frontmatter
/// (`chunk`, `section`: they never join the document metadata index `fields`
/// draws from). Suggests switching to `document` only when that granularity
/// is enabled here (#286). Caller-facing, so it says what a result carries, not
/// where the data lives.
fn fields_rejection(granularity: Granularity, effective: &[Granularity]) -> McpError {
    let msg = if effective.contains(&Granularity::Document) {
        format!(
            "fields only applies to document-granularity results (a {granularity} result \
             carries no frontmatter fields) — omit it, or set granularity to 'document'"
        )
    } else {
        "fields is not supported on this server (only document results carry frontmatter \
         fields, and document granularity is not enabled) — omit it"
            .to_string()
    };
    McpError::invalid_params(msg, None)
}

/// Rejection for `explain` at a grouped granularity (`document`, `section`),
/// which has no per-arm score breakdown to report. Suggests switching to
/// `chunk` only when that granularity is enabled here (#286).
fn explain_rejection(granularity: Granularity, effective: &[Granularity]) -> McpError {
    let reason = format!(
        "{granularity}-granularity results collapse to one row per {granularity} with no \
         per-arm score breakdown available to report"
    );
    let msg = if effective.contains(&Granularity::Chunk) {
        format!(
            "explain is chunk-granularity only; {reason} — omit it, or set granularity to 'chunk'"
        )
    } else {
        format!("explain is not supported on this server; {reason} — omit it")
    };
    McpError::invalid_params(msg, None)
}

/// Build the caller-facing error for an explicitly-requested granularity
/// that this deployment does not currently allow (#286). Lists exactly the
/// granularities enabled here (the EFFECTIVE set), so a caller knows what to
/// fall back to. The `chunking.heading_metadata` explanation is added only
/// when that is actually why `section` is missing (`section_gated_off`:
/// configured in `search.granularities`, dropped because the flag is off) —
/// not when an operator simply left `section` out of the configured set.
fn granularity_disabled_error(
    requested: Granularity,
    effective: &[Granularity],
    section_gated_off: bool,
) -> McpError {
    let mut msg = format!(
        "granularity '{requested}' is not enabled on this server (enabled: {})",
        join_granularities(effective)
    );
    if requested == Granularity::Section && section_gated_off {
        msg.push_str(HEADING_METADATA_OFF_SECTION_REASON);
    }
    McpError::invalid_params(msg, None)
}

/// Resolve a `search` call's granularity against this deployment's enabled
/// set (#286), on top of `default_granularity`'s ordinary query-presence
/// default:
///
/// - An explicit request is parsed, then checked against
///   `effective` (`ResolvedConfig::effective_granularities`) and rejected
///   by name — via [`granularity_disabled_error`] — if it isn't a member.
/// - An omitted one first tries `default_granularity`'s usual default
///   (`chunk` with a query, `document` without). If THAT default is itself
///   disabled here, the first enabled granularity — in canonical
///   chunk/document/section order — that also supports the current query
///   mode ([`granularity_supports_mode`]) is used instead. If none does,
///   that's a configuration problem, not a caller one, and is reported as
///   such rather than silently doing nothing.
///
/// `descriptions::granularity_description` states these defaults in prose;
/// `granularity_description_matches_resolution_for_every_effective_set`
/// keeps the two in step.
fn resolve_search_granularity(
    query_present: bool,
    requested: Option<&str>,
    effective: &[Granularity],
    section_gated_off: bool,
) -> Result<Granularity, McpError> {
    if let Some(raw) = requested {
        let g = parse_granularity(raw, effective)?;
        return if effective.contains(&g) {
            Ok(g)
        } else {
            Err(granularity_disabled_error(g, effective, section_gated_off))
        };
    }

    let default = default_granularity(query_present);
    if effective.contains(&default) {
        return Ok(default);
    }

    Granularity::ALL
        .into_iter()
        .find(|&g| effective.contains(&g) && granularity_supports_mode(g, query_present))
        .ok_or_else(|| {
            McpError::invalid_params(
                format!(
                    "no enabled search granularity supports {} (enabled: {})",
                    if query_present {
                        "a query"
                    } else {
                        "enumeration (search without a query — only document does); provide \
                         a query"
                    },
                    join_granularities(effective)
                ),
                None,
            )
        })
}

/// Whether `search`'s `query` should be treated as present. A blank/whitespace-only
/// string is treated the same as an absent query — there is nothing to embed —
/// rather than sent to the embedder as a no-op vector search.
fn query_is_present(query: &Option<String>) -> bool {
    query.as_deref().is_some_and(|q| !q.trim().is_empty())
}

fn validate_search_params(params: &SearchParams) -> Result<(), McpError> {
    if let Some(query) = &params.query
        && query.len() > MAX_QUERY_LEN
    {
        return Err(McpError::invalid_params(
            format!("query exceeds maximum length of {MAX_QUERY_LEN} characters"),
            None,
        ));
    }
    Ok(())
}

/// Parameters for `get_document`.
///
/// Modes — see `retrieval::parse_document_view_request` for the exact
/// validation rules (#286):
///   - Nothing but `path`: the whole document (unchanged).
///   - `start_line`/`end_line`: a raw line range (unchanged); excludes every
///     other mode parameter.
///   - `line` or `heading_path` (optionally with `levels_up`): one resolved
///     section, by line or by heading path.
///   - `outline: true`: the heading outline, no content — of the whole
///     document, or of the selected section when combined with a selector.
#[derive(Debug, Default, serde::Deserialize, schemars::JsonSchema)]
pub struct GetDocumentParams {
    /// Relative path, unique basename, or absolute path.
    pub path: String,
    // Per-mode rules beyond these one-liners (ambiguous or unresolved
    // selectors, oversized sections, capped outlines) are taught by the error
    // or the response itself — `SectionError::data()`, `OutlineView::hint`.
    /// First line (1-based, inclusive). An explicit range is never size-capped.
    #[serde(default)]
    pub start_line: Option<usize>,
    /// Last line (1-based, inclusive).
    #[serde(default)]
    pub end_line: Option<usize>,
    /// Select the section containing this 1-based line (its deepest enclosing
    /// heading). Not combinable with `heading_path` or `start_line`/`end_line`.
    #[serde(default)]
    pub line: Option<usize>,
    /// Select a section by heading path, e.g. `["Spells", "Fireball"]`:
    /// complete heading names in order. A trailing part (`["Fireball"]`) or an
    /// in-order subset that skips middle levels also resolves when it names
    /// exactly one section. Ignores case and whitespace. Not combinable with
    /// `line` or `start_line`/`end_line`.
    #[serde(default)]
    pub heading_path: Option<Vec<String>>,
    /// With `line`/`heading_path`: widen to the Nth parent heading (clamped at
    /// the top).
    #[serde(default)]
    pub levels_up: Option<usize>,
    /// Return the heading outline with line ranges instead of text — of the
    /// document, or of the section chosen by `line`/`heading_path`.
    #[serde(default)]
    pub outline: Option<bool>,
    /// Also return the last N changes to this document (1-100), newest first.
    #[serde(default)]
    pub history: Option<usize>,
    /// Link lists come with a whole-document read; `true` adds them to any
    /// read, `false` drops them.
    #[serde(default)]
    pub links: Option<bool>,
}

/// `get_document`'s link graph, in the compact shape a model reads: `links_out`
/// (paths this document links to that exist), `broken_links` (link targets
/// that do not), `links_in` (paths linking here), and `similar` (`{path,
/// score}` inferred neighbors, either direction, only when semantic edges are
/// on). Each key appears only when non-empty; `links_out_total`/`links_in_total`
/// only when that direction was capped at `retrieval::MAX_LINKS_PER_DIRECTION`.
fn insert_link_lists(
    structured: &mut serde_json::Map<String, serde_json::Value>,
    links_out: &crate::state::LinkPage<crate::state::OutboundLink>,
    links_in: &crate::state::LinkPage<crate::state::InboundLink>,
) {
    let mut out: Vec<&str> = Vec::new();
    let mut broken: Vec<&str> = Vec::new();
    let mut incoming: Vec<&str> = Vec::new();
    let mut similar: Vec<serde_json::Value> = Vec::new();
    let mut seen_similar: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for l in &links_out.links {
        if l.kind == "semantic" {
            if seen_similar.insert(&l.target_path) {
                similar.push(serde_json::json!({
                    "path": l.target_path,
                    "score": l.score.map_or(serde_json::Value::Null, round_score),
                }));
            }
        } else if l.exists {
            out.push(&l.target_path);
        } else {
            broken.push(&l.target_path);
        }
    }
    for l in &links_in.links {
        if l.kind == "semantic" {
            if seen_similar.insert(&l.source_path) {
                similar.push(serde_json::json!({
                    "path": l.source_path,
                    "score": l.score.map_or(serde_json::Value::Null, round_score),
                }));
            }
        } else {
            incoming.push(&l.source_path);
        }
    }
    for (key, list) in [
        ("links_out", out),
        ("broken_links", broken),
        ("links_in", incoming),
    ] {
        if !list.is_empty() {
            structured.insert(key.to_string(), serde_json::json!(list));
        }
    }
    if !similar.is_empty() {
        structured.insert("similar".to_string(), serde_json::Value::Array(similar));
    }
    if links_out.has_more() {
        structured.insert(
            "links_out_total".to_string(),
            serde_json::json!(links_out.total),
        );
    }
    if links_in.has_more() {
        structured.insert(
            "links_in_total".to_string(),
            serde_json::json!(links_in.total),
        );
    }
}

/// Map `retrieval::resolve_document_view`'s error into the `McpError` this
/// tool surface returns. The `Range` arm keeps `get_document`'s existing
/// range-error wording; `Section` follows the same "invalid_params + a `data`
/// payload the caller can act on" shape `GetDocumentError::NotFound`'s
/// suggestions already establish, except here the payload is structured
/// (`hint`/`candidates`) rather than folded into the message text alone,
/// since `SectionError::data()` already builds exactly that (#286).
fn document_view_error_to_mcp(err: retrieval::DocumentViewError) -> McpError {
    match err {
        retrieval::DocumentViewError::Range(e) => McpError::invalid_params(e.to_string(), None),
        retrieval::DocumentViewError::Section(e) => {
            McpError::invalid_params(e.message(), Some(e.data()))
        }
    }
}

/// Parameters for `get_schema`.
#[derive(Debug, Default, serde::Deserialize, schemars::JsonSchema)]
pub struct GetSchemaParams {
    /// Directory or document path; omit for the root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,

    /// Only report these fields (dot-paths).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fields: Option<Vec<String>>,

    /// Only fields with a closed value set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values_only: Option<bool>,

    /// Also list the most-used values of each open field, with document
    /// counts, and other fields documents here use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values_in_use: Option<bool>,
}

/// Parameters for `update_schema`.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct UpdateSchemaParams {
    /// Directory to edit; omit for the KB root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,

    // A string with an advertised `enum`, not a Rust enum, so parsing stays
    // as tolerant as `build_schema_edit` already is.
    #[schemars(extend("enum" = ["add_values", "remove_values", "set_field", "remove_field"]))]
    pub operation: String,

    /// Field to change, as a dot-path (`planning.method`); `add_values` and
    /// `set_field` create missing parents.
    pub field: String,

    /// Values, for add_values/remove_values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<String>>,

    /// Field definition for set_field; a JSON string also works.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub definition: Option<FieldDefinitionInput>,

    /// Preview the change without writing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_run: Option<bool>,

    /// Apply even if existing documents would fail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force: Option<bool>,

    /// Must be true to change the root schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acknowledge_root_change: Option<bool>,
}

/// Parameters for `search` (also covers enumeration).
#[derive(Debug, Default, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchParams {
    // Doc comments on these fields become the tool schema's property
    // descriptions (schemars). Several are replaced at list time with
    // effective-set-aware text, or removed when no enabled granularity can
    // use them (`descriptions::search_property_descriptions`, applied in
    // `KbSearchServer::overlay_input_schema`).
    /// Query. Omit to list every match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,

    /// Frontmatter criteria keyed by field (dot-paths for nested fields): a
    /// scalar means equals (`{"type": "guide"}`), an array any-of, an object
    /// all-of or a numeric range (`{"planning.prep_minutes": {"lt": 30}}`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filters: Option<SearchFiltersInput>,

    /// Case-insensitive substring of the path, not just a prefix: `recipes/`
    /// matches any recipes folder, `stir_fr` finds `stir_fry.md`. Prefer the
    /// longest fragment you are sure of.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_prefix: Option<String>,

    // Doc comments on these fields become the tool schema's property
    // descriptions (schemars), so they stay caller-facing. `granularity`'s is
    // replaced at list time with the effective-set text
    // (`KbSearchServer::overlay_input_schema`); `heading_prefix` is removed
    // from the schema entirely when `chunking.heading_metadata` is off.
    /// What each result is. Omit for the default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub granularity: Option<String>,

    /// Restrict query results to text under this run of complete heading
    /// names, starting at any level: `["Conditions", "Blinded"]` matches under
    /// `Ch. 10 > Conditions > Blinded`. Ignores case and whitespace. Needs a
    /// query.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heading_prefix: Option<Vec<String>>,

    /// Maximum results to return.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u64>,

    // Replaced at list time with effective-set-aware text
    // (`descriptions::search_property_descriptions`).
    /// Results to skip, for paging.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<u64>,

    /// Sort key, enumeration only.
    // A string with an advertised `enum`: `OrderBy::parse` also accepts the
    // aliases `file_path`, `modified` and `indexed`, case-insensitively.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("enum" = ["path", "title", "mtime", "indexed_at"]))]
    pub order_by: Option<String>,

    /// Sort descending (enumeration only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub descending: Option<bool>,

    /// Relevance floor (query only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_score: Option<f32>,

    /// Add a score breakdown per result (`chunk` granularity only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explain: Option<bool>,

    /// Frontmatter fields to include per result, as dot-paths (`document`
    /// only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fields: Option<Vec<String>>,

    /// Only documents modified after this date (YYYY-MM-DD).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified_after: Option<String>,

    /// Only documents modified before this date (YYYY-MM-DD).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified_before: Option<String>,
}

/// Parameters for `write_document`.
#[derive(Debug, Default, serde::Deserialize, schemars::JsonSchema)]
pub struct WriteDocumentParams {
    // Each edit mode states what it combines with once, here; a bad
    // combination is refused by `parse_edit_mode` with the rule spelled out.
    /// Document or directory path, relative to the KB root. Not with
    /// `documents`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// The whole file, including YAML frontmatter: creates `path`, or
    /// replaces it. Not combinable with another edit mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Exact text to replace with `new_string`; must occur exactly once. Not
    /// combinable with another edit mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_string: Option<String>,
    /// Replacement for `old_string`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_string: Option<String>,
    /// Frontmatter edits applied in order, body untouched: `set_field`
    /// (`value`), `remove_field`, `add_values`/`remove_values` on a list
    /// (`values`; add creates the list). Unchanged fields keep their
    /// formatting. Combines with `append` (patch first).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frontmatter_patch: Option<Vec<FrontmatterPatchOp>>,
    /// Text to add to the end of the body, after one newline (start it with a
    /// blank line for a paragraph break).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub append: Option<String>,
    /// Move to this path, with any edit mode or alone; links to it are
    /// rewritten. A directory `path` moves its whole subtree.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_path: Option<String>,
    /// Optional one-line summary of the change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// The `version` from your last `get_document` read; required to replace
    /// an existing document with `content` or to move one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_version: Option<String>,
    /// Skip the near-duplicate check when creating.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force_new: Option<bool>,
    // The cap below is `write::MAX_BATCH_DOCUMENTS`; doc comments cannot
    // interpolate it, so `batch_documents_description_states_the_real_cap`
    // pins the number. Moves are excluded per `write::BatchWriteRequest`.
    /// Batch: several documents as one atomic change — all are saved or none.
    /// Pass ONLY this (plus optional `message`). Each entry takes `path` and
    /// the single-document edit modes with the same rules, but no
    /// `new_path`. Paths unique; at most 25.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub documents: Option<Vec<BatchDocumentInput>>,
}

// One document within a `write_document` batch call
// (`WriteDocumentParams::documents`): the single-document content-edit
// vocabulary minus `new_path` (no per-entry moves) and `message` (the batch
// has one commit message, supplied once at the top level). Every property
// means what its top-level twin does, which `documents` says once, so none
// carries a description of its own — the item schema is pure structure apart
// from the one field description `FrontmatterPatchOp` brings along.
#[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
pub struct BatchDocumentInput {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_string: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_string: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frontmatter_patch: Option<Vec<FrontmatterPatchOp>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub append: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force_new: Option<bool>,
}

// Mirrors `update_schema`'s `operation`/`field`/`values`/`definition` shape
// (`UpdateSchemaParams`, `build_schema_edit`), applied to a document's own
// frontmatter values; parsed into `write::FrontmatterEdit`. `operation` stays
// a string with an advertised `enum`, so `build_frontmatter_edit` keeps
// parsing it case-insensitively. The operations themselves are described
// once, on `WriteDocumentParams::frontmatter_patch`.
#[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
pub struct FrontmatterPatchOp {
    #[schemars(extend("enum" = ["set_field", "remove_field", "add_values", "remove_values"]))]
    pub operation: String,
    /// Dot-path.
    pub field: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<serde_json::Value>>,
}

/// Parameters for `delete_document`.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct DeleteDocumentParams {
    /// Path relative to the KB root.
    pub path: String,
    /// Optional one-line summary of the change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// The `version` from your last `get_document` read; required.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_version: Option<String>,
}

/// Validated edit mode, produced by `parse_edit_mode`.
#[derive(Debug, PartialEq)]
pub enum EditMode {
    /// Replace `old_string` with `new_string` (must appear exactly once).
    Surgical { old: String, new: String },
    /// Replace the entire document content.
    Full { content: String },
    /// Structured frontmatter edits only — see `write::FrontmatterEdit`.
    Patch { edits: Vec<FrontmatterEdit> },
    /// Append only.
    Append { text: String },
    /// Structured frontmatter edits AND an append, in the same call — applied
    /// patch-then-append (see `write_document_edit`).
    PatchAppend {
        edits: Vec<FrontmatterEdit>,
        text: String,
    },
}

/// An owned `write::DocChange`: what a parsed edit mode does to a document,
/// handed to the write pipeline, which applies it to the document as read under
/// the lock (so a relative edit always lands on current content).
pub(crate) enum OwnedChange {
    Create(String),
    Replace(String),
    Relative(Box<write::RelativeEdit<'static>>),
}

impl OwnedChange {
    /// `display_path` names the document in an anchor-not-found message.
    pub(crate) fn from_mode(mode: EditMode, display_path: String) -> Self {
        match mode {
            EditMode::Full { content } => OwnedChange::Replace(content),
            EditMode::Surgical { old, new } => OwnedChange::Relative(Box::new(move |c: &str| {
                apply_surgical(c, &old, &new, &display_path)
            })),
            EditMode::Patch { edits } => OwnedChange::Relative(Box::new(move |c: &str| {
                write::apply_frontmatter_patch(c, &edits)
            })),
            EditMode::Append { text } => {
                OwnedChange::Relative(Box::new(move |c: &str| Ok(write::apply_append(c, &text))))
            }
            // Patch first, then append — the patch only ever touches the
            // frontmatter block, so the two compose with no ordering ambiguity.
            EditMode::PatchAppend { edits, text } => {
                OwnedChange::Relative(Box::new(move |c: &str| {
                    write::apply_frontmatter_patch(c, &edits)
                        .map(|patched| write::apply_append(&patched, &text))
                }))
            }
        }
    }

    pub(crate) fn as_change(&self) -> write::DocChange<'_> {
        match self {
            OwnedChange::Create(c) => write::DocChange::Create(c),
            OwnedChange::Replace(c) => write::DocChange::Replace(c),
            OwnedChange::Relative(edit) => write::DocChange::Relative(edit.as_ref()),
        }
    }
}

/// The `Operation:` trailer label for an edit (`None` = a pure move).
fn edit_operation_label(mode: Option<&EditMode>, is_move: bool) -> &'static str {
    match (mode, is_move) {
        (None, _) => "write_document (move)",
        (Some(EditMode::Full { .. }), false) => "write_document (full replace)",
        (Some(EditMode::Full { .. }), true) => "write_document (full replace + move)",
        (Some(EditMode::Surgical { .. }), false) => "write_document (surgical replace)",
        (Some(EditMode::Surgical { .. }), true) => "write_document (surgical replace + move)",
        (Some(EditMode::Patch { .. }), false) => "write_document (frontmatter patch)",
        (Some(EditMode::Patch { .. }), true) => "write_document (frontmatter patch + move)",
        (Some(EditMode::Append { .. }), false) => "write_document (append)",
        (Some(EditMode::Append { .. }), true) => "write_document (append + move)",
        (Some(EditMode::PatchAppend { .. }), false) => {
            "write_document (frontmatter patch + append)"
        }
        (Some(EditMode::PatchAppend { .. }), true) => {
            "write_document (frontmatter patch + append + move)"
        }
    }
}

/// Build a single `write::FrontmatterEdit` from the wire shape a caller sent,
/// mirroring `build_schema_edit`'s identical role for `update_schema`.
fn build_frontmatter_edit(op: &FrontmatterPatchOp) -> Result<FrontmatterEdit, String> {
    if op.field.trim().is_empty() {
        return Err("frontmatter_patch: field must not be empty".to_string());
    }
    if op.field.len() > MAX_FILTER_STR_LEN {
        return Err(format!(
            "frontmatter_patch: field name too long: {} chars (max {})",
            op.field.len(),
            MAX_FILTER_STR_LEN
        ));
    }
    let check_value_size = |v: &serde_json::Value| -> Result<(), String> {
        let size = serde_json::to_string(v)
            .map(|s| s.len())
            .unwrap_or(usize::MAX);
        if size > MAX_FRONTMATTER_VALUE_LEN {
            return Err(format!(
                "frontmatter_patch: value for '{}' too large (max {} bytes)",
                op.field, MAX_FRONTMATTER_VALUE_LEN
            ));
        }
        Ok(())
    };

    match op.operation.trim().to_ascii_lowercase().as_str() {
        "set_field" => {
            let value = op
                .value
                .clone()
                .ok_or_else(|| "frontmatter_patch: 'set_field' requires a value".to_string())?;
            check_value_size(&value)?;
            Ok(FrontmatterEdit::SetField {
                field: op.field.clone(),
                value,
            })
        }
        "remove_field" => Ok(FrontmatterEdit::RemoveField {
            field: op.field.clone(),
        }),
        op_name @ ("add_values" | "remove_values") => {
            let values = op.values.clone().filter(|v| !v.is_empty()).ok_or_else(|| {
                format!("frontmatter_patch: '{op_name}' requires a non-empty values list")
            })?;
            if values.len() > MAX_FRONTMATTER_PATCH_VALUES {
                return Err(format!(
                    "frontmatter_patch: too many values for '{}': {} (max {})",
                    op.field,
                    values.len(),
                    MAX_FRONTMATTER_PATCH_VALUES
                ));
            }
            for v in &values {
                check_value_size(v)?;
            }
            if op_name == "add_values" {
                Ok(FrontmatterEdit::AddValues {
                    field: op.field.clone(),
                    values,
                })
            } else {
                Ok(FrontmatterEdit::RemoveValues {
                    field: op.field.clone(),
                    values,
                })
            }
        }
        other => Err(format!(
            "frontmatter_patch: unknown operation '{other}': expected set_field, remove_field, \
             add_values, or remove_values"
        )),
    }
}

/// Parse every op in a `frontmatter_patch` list, in order, applying the same
/// per-call size cap `MAX_FRONTMATTER_PATCH_OPS` bounds.
fn parse_frontmatter_patch_ops(ops: &[FrontmatterPatchOp]) -> Result<Vec<FrontmatterEdit>, String> {
    if ops.len() > MAX_FRONTMATTER_PATCH_OPS {
        return Err(format!(
            "frontmatter_patch: too many operations: {} (max {})",
            ops.len(),
            MAX_FRONTMATTER_PATCH_OPS
        ));
    }
    ops.iter().map(build_frontmatter_edit).collect()
}

/// Parse and validate the content-edit fields of `WriteDocumentParams`,
/// returning a typed `Option<EditMode>` or a human-readable error string.
///
/// Rules:
/// - SURGICAL = `old_string` AND `new_string` both `Some`, `content` is `None`.
/// - FULL = `content` is `Some`, both `old_string` and `new_string` are `None`.
/// - PATCH/APPEND/PATCH_APPEND = `frontmatter_patch` and/or `append` is `Some`.
///   These two combine freely with EACH OTHER (patch is applied first, then
///   append — see `write_document_edit`), but neither may combine with FULL
///   or SURGICAL: those are whole-document edits, these are structured edits
///   to part of the document, and there is no well-defined way to apply both
///   to the same call.
/// - Neither FULL, SURGICAL, nor PATCH/APPEND, but `new_path` is `Some`:
///   `Ok(None)` — a pure move with content left unchanged. The caller
///   (`write_document`'s edit path) reads the document's current content
///   itself and passes it through unchanged.
/// - None of the above, and `new_path` is also `None`: rejected — at least
///   one edit mode (or a move) must be requested.
/// - FULL and SURGICAL together are always rejected, regardless of `new_path`
///   — the two whole-document edit modes remain mutually exclusive WITH EACH
///   OTHER; `new_path` is an orthogonal, independent axis that may combine
///   with any one edit mode (or neither, for a pure move).
/// - Surgical with `old_string == new_string` is rejected (no-op), same as an
///   `append` that is empty or whitespace-only.
pub fn parse_edit_mode(params: &WriteDocumentParams) -> Result<Option<EditMode>, String> {
    let has_content = params.content.is_some();
    let has_move = params.new_path.is_some();

    // old_string/new_string must arrive as a pair, independent of every other
    // mode — checked first so a caller who supplied only one gets that
    // specific error rather than a generic "mutually exclusive" one.
    match (params.old_string.is_some(), params.new_string.is_some()) {
        (true, false) => {
            return Err(
                "old_string requires new_string; provide both for a surgical edit. (If you only \
                 meant to move the document, omit old_string entirely and pass new_path alone.)"
                    .to_string(),
            );
        }
        (false, true) => {
            return Err(
                "new_string requires old_string; provide both for a surgical edit. (If you only \
                 meant to move the document, omit new_string entirely and pass new_path alone.)"
                    .to_string(),
            );
        }
        _ => {}
    }
    let has_surgical = params.old_string.is_some();
    let has_patch = params.frontmatter_patch.is_some();
    let has_append = params.append.is_some();

    if (has_content || has_surgical) && (has_patch || has_append) {
        return Err(
            "content and old_string/new_string are whole-document edits, mutually exclusive \
             with frontmatter_patch/append (structured edits to part of the document). \
             Provide one or the other — frontmatter_patch and append may combine with each \
             other, just not with content or old_string/new_string."
                .to_string(),
        );
    }
    if has_content && has_surgical {
        return Err("content is mutually exclusive with old_string/new_string; \
             provide either content (full replace) or old_string+new_string (surgical edit) \
             — not both. new_path may be combined with either one (or with neither, for a \
             pure move), but it does not resolve a conflict between the two edit modes \
             themselves."
            .to_string());
    }

    if has_content {
        return Ok(Some(EditMode::Full {
            content: params.content.clone().unwrap(),
        }));
    }
    if has_surgical {
        let old = params.old_string.clone().unwrap();
        let new = params.new_string.clone().unwrap();
        if old == new {
            return Err(
                "old_string and new_string are identical — no change would be made".to_string(),
            );
        }
        return Ok(Some(EditMode::Surgical { old, new }));
    }
    if has_patch || has_append {
        let edits = match &params.frontmatter_patch {
            Some(ops) if !ops.is_empty() => Some(parse_frontmatter_patch_ops(ops)?),
            Some(_) => {
                return Err("frontmatter_patch must contain at least one operation".to_string());
            }
            None => None,
        };
        let text = match params.append.as_deref() {
            Some(t) if !t.trim().is_empty() => Some(t.to_string()),
            Some(_) => return Err("append must not be empty".to_string()),
            None => None,
        };
        return Ok(Some(match (edits, text) {
            (Some(edits), Some(text)) => EditMode::PatchAppend { edits, text },
            (Some(edits), None) => EditMode::Patch { edits },
            (None, Some(text)) => EditMode::Append { text },
            (None, None) => unreachable!(
                "has_patch || has_append guarantees at least one of edits/text is Some"
            ),
        }));
    }

    // No edit mode at all: a pure move if new_path was given, otherwise an error.
    if has_move {
        Ok(None)
    } else {
        Err(
            "must provide content (full replace), old_string+new_string (surgical edit), \
             frontmatter_patch (structured frontmatter edit), append (add to the end of the \
             body), or new_path (move) — at least one is required"
                .to_string(),
        )
    }
}

/// Adapt one `BatchDocumentInput` into the shape [`parse_edit_mode`] expects,
/// so a batch entry's content-edit mode is parsed by the EXACT SAME function
/// (and therefore the exact same rules/error text) a single-document
/// `write_document` call uses — no second copy of that logic to drift out of
/// sync with the first. `path`/`new_path`/`message` are irrelevant to
/// `parse_edit_mode` (it only reads the content-edit fields plus
/// `new_path.is_some()`, and a batch entry never has one), so they are
/// filled with harmless placeholders rather than threaded through.
fn batch_input_as_write_params(input: &BatchDocumentInput) -> WriteDocumentParams {
    WriteDocumentParams {
        path: Some(input.path.clone()),
        content: input.content.clone(),
        old_string: input.old_string.clone(),
        new_string: input.new_string.clone(),
        frontmatter_patch: input.frontmatter_patch.clone(),
        append: input.append.clone(),
        new_path: None,
        message: None,
        expected_version: input.expected_version.clone(),
        force_new: input.force_new,
        documents: None,
    }
}

/// Number of leading characters of `old_string` used as a search anchor when no exact
/// or whitespace-normalized match exists at all. Long enough to be a specific location
/// in most real documents, short enough that a single `str::find` over the document
/// stays cheap.
const NOT_FOUND_ANCHOR_CHARS: usize = 40;

/// How many characters of surrounding document text to show on each side of an anchor
/// match, when reporting a near-match diagnostic. Kept small deliberately: this text
/// goes into an error a caller (and possibly its logs) will see, so it is an
/// orientation snippet, not a document excerpt.
const NOT_FOUND_CONTEXT_CHARS: usize = 80;

/// Above this size, skip the near-match/anchor diagnostics below and fall back to the
/// plain not-found message. `old_content` is the on-disk document, which the write
/// path's `MAX_CONTENT_LEN` does not bound: it could predate that cap, have been
/// written outside these tools entirely, or be an over-cap document an edit is
/// shrinking (an edit is refused only for growing a document past the cap) — so
/// this is a second, independent bound rather than an assumption that the
/// write-path cap already covers it.
const NOT_FOUND_DIAGNOSTIC_MAX_BYTES: usize = MAX_CONTENT_LEN;

/// Collapse every run of whitespace (including newlines) to a single space and trim
/// the ends, for a whitespace-insensitive comparison.
///
/// This treats differing indentation, trailing spaces, and CRLF-vs-LF line endings as
/// equal — by far the most common reason a caller's `old_string` fails to match
/// verbatim (e.g. it was retyped or reformatted rather than copied from
/// `get_document`'s output). One linear pass, no allocation beyond the output string,
/// so it stays cheap up to `NOT_FOUND_DIAGNOSTIC_MAX_BYTES`.
fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A small window of `content` centered on the char-boundary byte offset `pos`,
/// clamped to `radius` characters on each side. Used to show just enough surrounding
/// text to orient a caller, never a large slice of the document.
fn context_window(content: &str, pos: usize, radius: usize) -> &str {
    let start = content[..pos]
        .char_indices()
        .rev()
        .nth(radius.saturating_sub(1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let end = content[pos..]
        .char_indices()
        .nth(radius)
        .map(|(i, _)| pos + i)
        .unwrap_or(content.len());
    &content[start..end]
}

/// Apply a surgical edit: replace the single occurrence of `old_string` with
/// `new_string` in `old_content`.
///
/// Returns the new content string on success, or a descriptive error string.
/// `display_path` is the document's KB-relative path, used only to make the error
/// message name the file directly rather than the generic word "document".
///
/// When `old_string` occurs zero times, the error is built to help the caller decide
/// what to do next rather than just restating the failure (issue #88):
/// 1. If a whitespace-insensitive match exists, say so — that is almost always the
///    actual cause, and the fix is "copy it verbatim" rather than "re-read the file".
/// 2. Otherwise, anchor on the start of `old_string` and, if that much appears in the
///    document, show the surrounding text so the caller can see exactly how reality
///    diverges (a stale read, a typo partway through, wrong section, etc).
/// 3. Otherwise, nothing in the document resembles `old_string` at all — most likely
///    the wrong document or content that changed substantially.
///
/// Cost is bounded: every diagnostic pass here is a single linear scan (`split_whitespace`,
/// `contains`, or `str::find`) over `old_content`, and the whole diagnostic block is
/// skipped above `NOT_FOUND_DIAGNOSTIC_MAX_BYTES`, so a failed edit is never
/// accidentally quadratic in document size.
pub fn apply_surgical(
    old_content: &str,
    old_string: &str,
    new_string: &str,
    display_path: &str,
) -> Result<String, String> {
    let count = old_content.matches(old_string).count();
    match count {
        0 => {
            if old_content.len() > NOT_FOUND_DIAGNOSTIC_MAX_BYTES {
                return Err(format!("old_string not found in '{display_path}'"));
            }

            if !old_string.trim().is_empty()
                && normalize_whitespace(old_content).contains(&normalize_whitespace(old_string))
            {
                return Err(format!(
                    "old_string not found in '{display_path}', but a near-match exists \
                     differing only in whitespace — indentation, trailing spaces, or \
                     line endings. Copy old_string exactly as returned by get_document \
                     rather than retyping or reformatting it."
                ));
            }

            let anchor_end = old_string
                .char_indices()
                .nth(NOT_FOUND_ANCHOR_CHARS)
                .map(|(i, _)| i)
                .unwrap_or(old_string.len());
            let anchor = old_string[..anchor_end].trim();

            if !anchor.is_empty()
                && let Some(pos) = old_content.find(anchor)
            {
                let window = context_window(old_content, pos, NOT_FOUND_CONTEXT_CHARS);
                return Err(format!(
                    "old_string not found in '{display_path}'. Its start ('{anchor}') \
                     does appear, but the text after that point differs from \
                     old_string — nearby document content: \"…{window}…\". The file may \
                     have changed since you read it, or old_string may have a typo past \
                     that point."
                ));
            }

            Err(format!(
                "old_string not found in '{display_path}', and nothing resembling it \
                 was located either. This may be the wrong document, or its content has \
                 changed substantially since it was last read — re-read it with \
                 get_document before editing."
            ))
        }
        1 => Ok(old_content.replacen(old_string, new_string, 1)),
        n => Err(format!(
            "old_string is not unique in '{display_path}' (found {n} occurrences); \
             include more surrounding context to disambiguate"
        )),
    }
}

/// Add `merged_with_other_changes: true` to a `structured_content` map — only
/// when it applies, so the usual result carries no field about it at all.
fn insert_merged(fields: &mut serde_json::Map<String, serde_json::Value>, merged: bool) {
    if merged {
        fields.insert(
            "merged_with_other_changes".to_string(),
            serde_json::json!(true),
        );
    }
}

/// The refusal for a change that no longer fits the document
/// (`WriteError::EditedElsewhere`).
fn edited_elsewhere_error(rel_path: &str) -> McpError {
    McpError::invalid_params(format!("'{rel_path}': {}", write::EDITED_ELSEWHERE), None)
}

/// The instruction every refusal for an absolute change sent without
/// `expected_version` (`WriteError::VersionRequired`) gives, `action` being the
/// operation-specific rest of the sentence.
fn pass_expected_version(action: &str) -> String {
    format!("pass expected_version (the version from get_document) to {action}")
}

/// The refusal for an absolute change sent without `expected_version`
/// (`WriteError::VersionRequired`).
fn version_required_error(rel_path: &str) -> McpError {
    McpError::invalid_params(
        format!(
            "'{rel_path}' already exists: {}",
            pass_expected_version("replace, move or delete it")
        ),
        None,
    )
}

/// How `KbSearchServer::write_raw_file` ended when it did not fail outright.
enum RawWrite {
    /// The commit landed (synced, or pending sync).
    Committed,
    /// The remote changed underneath it. Its commit was dropped (the branch is
    /// back at its pre-commit HEAD) and nothing was written; the caller syncs
    /// again, then re-applies its edit or refuses.
    Conflict,
}

/// The message a write tool returns when its change could not be saved and has
/// been fully undone. The underlying cause is a server-side detail — it was
/// already logged where it happened — so it is deliberately not relayed: the
/// caller cannot act on it, and the only useful instruction is to retry.
fn not_saved_error(what: &str) -> McpError {
    McpError::internal_error(
        format!("{what} could not be saved. Nothing was changed; try again."),
        None,
    )
}

/// The message a write tool returns when its change could not be saved AND
/// undoing the partial write failed too. That is an operator problem (logged at
/// `error!` where it happened), not something the caller can fix or retry its way
/// out of, so the caller is only told not to trust the current state.
fn not_saved_unverified_error(what: &str) -> McpError {
    McpError::internal_error(
        format!(
            "{what} could not be saved, and the server could not confirm it was left \
             unchanged. Do not retry; re-read it with get_document, and report the \
             problem to the operator."
        ),
        None,
    )
}

/// A failure to resolve the server's own sync credential. Configuration, not a
/// caller error: logged, and reported as a plain save failure.
fn credential_error(e: impl std::fmt::Display) -> McpError {
    error!("write refused: could not resolve the sync credential: {e:#}");
    McpError::internal_error(
        "The change could not be saved: the server is misconfigured. Nothing was \
         changed; report the problem to the operator."
            .to_string(),
        None,
    )
}

/// `write.rs`'s message-length/newline rejections name the field by its internal
/// purpose; the tool parameter is just `message`.
fn message_param_reason(reason: &str) -> String {
    reason.replace("commit message", "message")
}

/// A `write::validate_commit_message` rejection, worded for the `message` parameter.
fn invalid_message_error(reason: String) -> McpError {
    McpError::invalid_params(message_param_reason(&reason), None)
}

/// The response every successful single-document write or delete shares:
/// `path` (where the document is now), `action` (`created`, `updated`,
/// `moved` — with `from` — or `deleted`), `version` when there is a document to
/// version, and `merged_with_other_changes: true` when that applies. `diff`
/// rides only on an edit: a create would echo the content the caller just
/// sent, and a delete the document it just removed. It is capped (see
/// [`capped_diff`]), with `diff_truncated: true`/`diff_total_bytes` only when
/// cut.
fn write_response(
    path: &str,
    action: &str,
    diff: &str,
    version: Option<&str>,
    merged: bool,
) -> serde_json::Map<String, serde_json::Value> {
    let mut fields = serde_json::Map::new();
    fields.insert("path".to_string(), serde_json::json!(path));
    fields.insert("action".to_string(), serde_json::json!(action));
    if matches!(action, "updated" | "moved") && !diff.is_empty() {
        let (diff, diff_truncated, diff_total_bytes) = capped_diff(diff);
        fields.insert("diff".to_string(), serde_json::json!(diff));
        if diff_truncated {
            fields.insert("diff_truncated".to_string(), serde_json::json!(true));
            fields.insert(
                "diff_total_bytes".to_string(),
                serde_json::json!(diff_total_bytes),
            );
        }
    }
    if let Some(version) = version {
        fields.insert("version".to_string(), serde_json::json!(version));
    }
    insert_merged(&mut fields, merged);
    fields
}

/// Insert `key: paths` only when `paths` is non-empty.
fn insert_paths(
    fields: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    paths: &[String],
) {
    if !paths.is_empty() {
        fields.insert(key.to_string(), serde_json::json!(paths));
    }
}

/// Map a successful `write::write_documents_batch` result (#180) onto this
/// tool surface's `CallToolResult`: a `documents` array of per-document
/// [`write_response`]s (`created` or `updated`), saved as one change.
fn batch_write_success_to_result(success: write::BatchWriteSuccess) -> CallToolResult {
    let documents: Vec<serde_json::Value> = success
        .documents
        .iter()
        .map(|d| {
            serde_json::Value::Object(write_response(
                &d.rel_path,
                if d.is_create { "created" } else { "updated" },
                &d.diff,
                d.version.as_deref(),
                d.merged,
            ))
        })
        .collect();
    CallToolResult::structured(serde_json::json!({ "documents": documents }))
}

/// Short, human-readable text for one document's `WriteError` inside a batch
/// failure report (#180) — a compact cousin of `create_edit_error_to_mcp_error`'s
/// per-variant messages (full sentences addressed to a single-document
/// caller); this one is designed to read well joined with several siblings
/// in one list. Exhaustive over `WriteError` deliberately, same as
/// `create_edit_error_to_mcp_error`/`write_error_response` — a new
/// `WriteError` variant must be given a line here too, not silently folded
/// into a generic fallback.
fn batch_write_document_error_text(err: &WriteError) -> String {
    match err {
        WriteError::SchemaFile { rel_path } => {
            format!("'{rel_path}' is a schema file; change schemas with update_schema")
        }
        WriteError::Validation { result } => result.errors.join("; "),
        WriteError::DedupHit {
            duplicate_of,
            similarity,
            threshold,
        } => format!(
            "a similar document already exists: '{duplicate_of}' (similarity {similarity:.2} \
             >= threshold {threshold:.2})"
        ),
        WriteError::InvalidCommitMessage { reason } => message_param_reason(reason),
        WriteError::UnsafePath { msg } => msg.clone(),
        WriteError::Internal { msg } => msg.clone(),
        WriteError::AlreadyExists => "document already exists".to_string(),
        WriteError::NotFound => "document does not exist (deleted or moved?): find it with search \
                                 or get_document, or give content and no expected_version to \
                                 create it"
            .to_string(),
        WriteError::EditedElsewhere => write::EDITED_ELSEWHERE.to_string(),
        WriteError::VersionRequired => {
            format!("already exists: {}", pass_expected_version("replace it"))
        }
        WriteError::InvalidEdit { msg } => msg.clone(),
        // The cause is logged where it happened; see `not_saved_error`.
        WriteError::PreCommitFailed { .. } => "could not be saved".to_string(),
        WriteError::Io { msg } => msg.clone(),
    }
}

/// Map a `write::write_documents_batch` failure (#180) onto this tool
/// surface's `McpError`.
fn batch_write_error_to_mcp_error(err: write::BatchWriteError) -> McpError {
    match err {
        write::BatchWriteError::Empty => {
            McpError::invalid_params("documents must not be empty".to_string(), None)
        }
        write::BatchWriteError::TooMany { count, max } => McpError::invalid_params(
            format!("documents has {count} entries; maximum is {max}"),
            None,
        ),
        write::BatchWriteError::DuplicatePath { rel_path } => McpError::invalid_params(
            format!("documents contains '{rel_path}' more than once"),
            None,
        ),
        write::BatchWriteError::InvalidCommitMessage { reason } => invalid_message_error(reason),
        write::BatchWriteError::Documents { failures } => {
            let detail: Vec<serde_json::Value> = failures
                .iter()
                .map(|(path, err)| {
                    serde_json::json!({
                        "path": path,
                        "error": batch_write_document_error_text(err),
                    })
                })
                .collect();
            let summary = failures
                .iter()
                .map(|(path, err)| format!("'{path}': {}", batch_write_document_error_text(err)))
                .collect::<Vec<_>>()
                .join("; ");
            McpError::invalid_params(
                format!(
                    "batch write failed: {} document(s) had a problem — {summary}",
                    failures.len(),
                ),
                Some(serde_json::json!({ "failures": detail })),
            )
        }
        // `msg` was logged by `write::write_documents_batch` and stays server-side.
        write::BatchWriteError::PreCommitFailed {
            rolled_back: true, ..
        } => not_saved_error("The batch"),
        write::BatchWriteError::PreCommitFailed {
            rolled_back: false, ..
        } => not_saved_unverified_error("The batch"),
        write::BatchWriteError::EditedElsewhere => McpError::invalid_params(
            format!("batch write failed: {}", write::EDITED_ELSEWHERE),
            None,
        ),
    }
}

/// Map a successful `write::write_document` result (create, edit or move) onto
/// this tool surface's `CallToolResult`: [`write_response`] plus
/// `rewritten_paths`, the OTHER documents a move rewrote incoming links in —
/// a move that silently edits other documents must say which ones.
fn create_edit_success_to_result(
    success: WriteSuccess,
    rel_path: &str,
    is_create: bool,
    dest_path: Option<&str>,
) -> CallToolResult {
    let (path, action) = match dest_path {
        Some(dest) => (dest, "moved"),
        None if is_create => (rel_path, "created"),
        None => (rel_path, "updated"),
    };
    let mut fields = write_response(
        path,
        action,
        &success.diff,
        success.version.as_deref(),
        success.merged,
    );
    if dest_path.is_some() {
        fields.insert("from".to_string(), serde_json::json!(rel_path));
    }
    insert_paths(&mut fields, "rewritten_paths", &success.rewritten_paths);
    CallToolResult::structured(serde_json::Value::Object(fields))
}

/// The scope directory ([`crate::schema::scope_label`]) of a KB-relative schema file
/// path — how model-facing text names a schema file the server itself found.
fn schema_file_scope(schema_file: &str) -> String {
    crate::schema::scope_label(
        std::path::Path::new(schema_file)
            .parent()
            .unwrap_or(std::path::Path::new("")),
    )
}

/// The refusal for a document write, delete or move that names a schema file
/// (`WriteError::SchemaFile`), pointing at the one tool that edits schemas. Echoes
/// only the caller's own path, never the schema file name convention.
fn schema_file_path_error(schema_path: &str) -> McpError {
    McpError::invalid_params(
        format!(
            "'{schema_path}' is a schema file; change schemas with update_schema. The \
             document tools cannot write, move or delete it."
        ),
        None,
    )
}

/// Map a `write::write_document` failure (create or edit) onto this tool
/// surface's `McpError`, preserving the exact text/data shapes the existing
/// create/edit tests pin down.
///
/// `canonical_data_path` is used only to reconstruct the absolute path for the
/// `AlreadyExists` race (see that arm's comment) — every other arm reports
/// against `rel_path`, matching what the tool surface's other errors already do.
///
/// `dest_path` is `Some` only when this call is (or was attempting to be) a
/// document MOVE — i.e. `write_document` was called with `new_path` set. It
/// disambiguates the `AlreadyExists` arm, which for a move reports a collision
/// at the DESTINATION, not at `rel_path` (the source, which — for a move — is
/// expected to already exist). The create path's own TOCTOU race, and every
/// non-move edit call, pass `None` here, preserving the original
/// `AlreadyExists` wording keyed on `rel_path`.
fn create_edit_error_to_mcp_error(
    err: WriteError,
    rel_path: &str,
    is_create: bool,
    canonical_data_path: &Path,
    dest_path: Option<&str>,
) -> McpError {
    match err {
        WriteError::SchemaFile {
            rel_path: schema_path,
        } => schema_file_path_error(&schema_path),
        WriteError::Validation { result } => McpError::invalid_params(
            format!(
                "frontmatter validation failed for '{}': {}",
                rel_path,
                result.errors.join("; ")
            ),
            Some(serde_json::json!({ "field_errors": result.field_errors })),
        ),
        WriteError::DedupHit {
            duplicate_of,
            similarity,
            threshold,
        } => McpError::invalid_params(
            format!(
                "A similar document already exists: '{}' \
                 (similarity {:.2} ≥ threshold {:.2}). \
                 Edit it with write_document, or pass \
                 force_new=true to create a new document anyway.",
                duplicate_of, similarity, threshold
            ),
            Some(serde_json::json!({
                "duplicate_of": duplicate_of,
                "similarity": round_score(f64::from(similarity)),
                "threshold": round_score(f64::from(threshold)),
            })),
        ),
        WriteError::InvalidCommitMessage { reason } => invalid_message_error(reason),
        WriteError::UnsafePath { msg } => McpError::invalid_params(msg, None),
        // Same text MCP callers already saw for this failure before
        // `WriteError::Internal` existed to split it out of `UnsafePath` — see
        // that variant's doc comment. MCP is a trusted surface, so unlike
        // `web.rs` (which maps this to a generic message) this stays verbatim.
        WriteError::Internal { msg } => McpError::invalid_params(msg, None),
        // Two distinct races share this variant:
        //
        // 1. `write_document`'s own create-path pre-check (`abs_path.exists()`)
        //    already passed, so the file was created between that check and
        //    `write::write_document`'s `create_new` open. Restored to the
        //    pre-`write.rs`-extraction wording, which reported the absolute
        //    filesystem path rather than the repo-relative one — reconstructed
        //    here via `resolve_safe_write_path` since `WriteError::AlreadyExists`
        //    itself carries no path (kept a unit variant so `web.rs`'s exhaustive
        //    match needs no change to accommodate this). `dest_path` is `None`
        //    here, so this is the arm that runs.
        //
        // 2. `write_document` was called with `new_path` set (a MOVE, possibly
        //    combined with a content edit) and the DESTINATION already exists.
        //    `rel_path` here is the move's SOURCE (which legitimately exists —
        //    that's what made this an edit rather than a create), so reporting
        //    the collision against `rel_path` would misdirect the caller at the
        //    wrong file. `dest_path` disambiguates: when it's `Some`, this arm
        //    names the DESTINATION as what collided, not the source.
        WriteError::AlreadyExists => {
            if let Some(dest) = dest_path {
                let abs_dest = crate::write::resolve_safe_write_path(canonical_data_path, dest)
                    .unwrap_or_else(|_| {
                        canonical_data_path.join(crate::retrieval::kb_root_relative(dest))
                    });
                McpError::invalid_params(
                    format!(
                        "Cannot move '{}' to '{}': the destination '{}' already exists. \
                         A move never overwrites — choose a different new_path, or delete \
                         or move the document already at the destination first.",
                        rel_path,
                        dest,
                        abs_dest.display()
                    ),
                    None,
                )
            } else {
                let abs_path = crate::write::resolve_safe_write_path(canonical_data_path, rel_path)
                    .unwrap_or_else(|_| {
                        canonical_data_path.join(crate::retrieval::kb_root_relative(rel_path))
                    });
                McpError::invalid_params(
                    format!(
                        "File '{}' already exists; use write_document to modify it",
                        abs_path.display()
                    ),
                    None,
                )
            }
        }
        // A create only ends here when it carried `expected_version`: the caller
        // meant a document it read, which is no longer at this path.
        WriteError::NotFound if is_create => McpError::invalid_params(
            format!(
                "File '{rel_path}' does not exist: the document you read was deleted or moved. \
                 Find it with search or get_document, or omit expected_version to create a \
                 new document."
            ),
            None,
        ),
        WriteError::NotFound => {
            McpError::invalid_params(format!("File '{}' does not exist", rel_path), None)
        }
        WriteError::EditedElsewhere => edited_elsewhere_error(rel_path),
        WriteError::VersionRequired => version_required_error(rel_path),
        WriteError::InvalidEdit { msg } => McpError::invalid_params(msg, None),
        // `msg` was logged by `write::write_document` and stays server-side.
        WriteError::PreCommitFailed { rolled_back, .. } => {
            let what = if is_create {
                format!("New document '{rel_path}'")
            } else {
                format!("The change to '{rel_path}'")
            };
            if rolled_back {
                not_saved_error(&what)
            } else {
                not_saved_unverified_error(&what)
            }
        }
        WriteError::Io { msg } => McpError::internal_error(msg, None),
    }
}

/// Map a successful `write::delete_document` result onto this tool surface's
/// `CallToolResult`: [`write_response`] plus (#229) `referencing_paths`, the
/// OTHER documents that still link to the deleted one — the caller-visible
/// counterpart of the reverse-link check #181 logs server-side.
fn delete_success_to_result(success: WriteSuccess, rel_path: &str) -> CallToolResult {
    let mut fields = write_response(rel_path, "deleted", "", None, success.merged);
    insert_paths(&mut fields, "referencing_paths", &success.referencing_paths);
    CallToolResult::structured(serde_json::Value::Object(fields))
}

/// Map a `write::delete_document` failure onto this tool surface's `McpError`,
/// preserving the exact text/data shapes the existing delete tests pin down.
fn delete_error_to_mcp_error(err: WriteError, rel_path: &str) -> McpError {
    match err {
        WriteError::InvalidCommitMessage { reason } => invalid_message_error(reason),
        WriteError::UnsafePath { msg } => McpError::invalid_params(msg, None),
        // See `create_edit_error_to_mcp_error`'s identical arm: same text MCP
        // callers already saw before `WriteError::Internal` existed.
        WriteError::Internal { msg } => McpError::invalid_params(msg, None),
        WriteError::NotFound => {
            McpError::invalid_params(format!("document does not exist: '{}'", rel_path), None)
        }
        // `msg` was logged by `write::delete_document` and stays server-side.
        WriteError::PreCommitFailed {
            rolled_back: true, ..
        } => not_saved_error(&format!("Deleting '{rel_path}'")),
        WriteError::PreCommitFailed {
            rolled_back: false, ..
        } => not_saved_unverified_error(&format!("Deleting '{rel_path}'")),
        WriteError::Io { msg } => McpError::internal_error(msg, None),
        WriteError::SchemaFile {
            rel_path: schema_path,
        } => schema_file_path_error(&schema_path),
        WriteError::EditedElsewhere => edited_elsewhere_error(rel_path),
        WriteError::VersionRequired => {
            McpError::invalid_params(pass_expected_version(&format!("delete '{rel_path}'")), None)
        }
        // `write::delete_document` never produces these — they are create/edit-only
        // failure modes (frontmatter validation, the dedup gate, and
        // create-vs-exists) that the delete pipeline doesn't run.
        other => McpError::internal_error(format!("unexpected write error: {:?}", other), None),
    }
}

/// Map a successful `write::move_directory` result onto this tool surface's
/// `CallToolResult`: `action: "moved"`, `from`/`path` (the directories),
/// `moved` (every document's `{from, to}`), `moved_schema_dirs` when the
/// subtree carried directory schemas (named by directory, never by file — see
/// `schema_file_scope`), and `rewritten_paths` when documents outside the
/// subtree had links rewritten.
fn move_directory_success_to_result(
    success: DirectoryMoveSuccess,
    source_dir: &str,
    dest_dir: &str,
) -> CallToolResult {
    let (schema_moves, doc_moves): (Vec<_>, Vec<_>) = success
        .moved
        .iter()
        .partition(|(old, _)| crate::schema::is_schema_file_path(std::path::Path::new(old)));
    let mut structured = serde_json::Map::new();
    structured.insert("path".to_string(), serde_json::json!(dest_dir));
    structured.insert("action".to_string(), serde_json::json!("moved"));
    structured.insert("from".to_string(), serde_json::json!(source_dir));
    structured.insert(
        "moved".to_string(),
        doc_moves
            .iter()
            .map(|(old, new)| serde_json::json!({ "from": old, "to": new }))
            .collect(),
    );
    if !schema_moves.is_empty() {
        structured.insert(
            "moved_schema_dirs".to_string(),
            schema_moves
                .iter()
                .map(|(old, new)| {
                    serde_json::json!({
                        "from": schema_file_scope(old),
                        "to": schema_file_scope(new),
                    })
                })
                .collect(),
        );
    }
    insert_paths(&mut structured, "rewritten_paths", &success.rewritten_paths);
    insert_merged(&mut structured, success.merged);
    CallToolResult::structured(serde_json::Value::Object(structured))
}

/// Map a `write::move_directory` failure onto this tool surface's `McpError`.
/// Mirrors `create_edit_error_to_mcp_error`'s shape, scaled to
/// `DirectoryMoveError`'s directory-move-specific variants.
fn move_directory_error_to_mcp_error(
    err: DirectoryMoveError,
    source_dir: &str,
    dest_dir: &str,
) -> McpError {
    match err {
        DirectoryMoveError::SourceEmpty { msg } => {
            McpError::invalid_params(format!("Cannot move '{}': {}", source_dir, msg), None)
        }
        DirectoryMoveError::AlreadyExists => McpError::invalid_params(
            format!(
                "Cannot move '{}' to '{}': the destination already has at least one file \
                 living under it. A directory move never merges into or overwrites an \
                 existing prefix — choose a different destination, or clear it first.",
                source_dir, dest_dir
            ),
            None,
        ),
        DirectoryMoveError::Validation {
            failures,
            moved_schema_files,
        } => {
            let summary = failures
                .iter()
                .map(|(path, result)| format!("{}: {}", path, result.errors.join("; ")))
                .collect::<Vec<_>>()
                .join(" | ");
            let schema_note = if moved_schema_files.is_empty() {
                String::new()
            } else {
                let relocated = moved_schema_files
                    .iter()
                    .map(|(old, new)| {
                        format!("{} -> {}", schema_file_scope(old), schema_file_scope(new))
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "\n\nThis subtree carries its own directory schema(s), which relocate \
                     along with it ({}). That means these documents are being checked against a \
                     GENUINELY DIFFERENT schema cascade than the one that governed them at \
                     the source — a relocated schema file re-parents onto the destination's \
                     ancestors, not the source's, so a document that was valid moments ago \
                     can legitimately stop being valid. Either adjust the destination's \
                     cascade to still admit these documents (update_schema), or fix the \
                     documents themselves.",
                    relocated
                )
            };
            McpError::invalid_params(
                format!(
                    "Cannot move '{}' to '{}': frontmatter validation against the \
                     DESTINATION's schema cascade failed for {} document(s): {}{}",
                    source_dir,
                    dest_dir,
                    failures.len(),
                    summary,
                    schema_note
                ),
                Some(serde_json::json!({
                    "failures": failures.iter().map(|(path, result)| serde_json::json!({
                        "path": path,
                        "field_errors": result.field_errors,
                    })).collect::<Vec<_>>(),
                    "moved_schema_dirs": moved_schema_files.iter().map(|(old, new)| serde_json::json!({
                        "from": schema_file_scope(old),
                        "to": schema_file_scope(new),
                    })).collect::<Vec<_>>(),
                })),
            )
        }
        DirectoryMoveError::InvalidSchemaInSource { path, reason } => {
            let dir = schema_file_scope(&path);
            let reason = crate::schema::model_facing_reason(&reason);
            McpError::invalid_params(
                format!(
                    "Cannot move '{source_dir}' to '{dest_dir}': the schema for '{dir}' inside \
                     it is invalid: {reason}. The operator must repair that schema before \
                     the directory can move — nothing was moved."
                ),
                Some(serde_json::json!({ "invalid_schema_dir": dir, "reason": reason })),
            )
        }
        DirectoryMoveError::UnsafePath { msg } => McpError::invalid_params(msg, None),
        DirectoryMoveError::Internal { msg } => McpError::invalid_params(msg, None),
        DirectoryMoveError::InvalidCommitMessage { reason } => invalid_message_error(reason),
        // `msg` was logged by `write::move_directory` and stays server-side.
        DirectoryMoveError::PreCommitFailed {
            rolled_back: true, ..
        } => not_saved_error(&format!("Moving '{source_dir}' to '{dest_dir}'")),
        DirectoryMoveError::PreCommitFailed {
            rolled_back: false, ..
        } => not_saved_unverified_error(&format!("Moving '{source_dir}' to '{dest_dir}'")),
        DirectoryMoveError::Io { msg } => McpError::internal_error(msg, None),
        DirectoryMoveError::EditedElsewhere => McpError::invalid_params(
            format!(
                "Cannot move '{source_dir}' to '{dest_dir}': a document in it was edited by \
                 someone else while the move was prepared; nothing was moved. Try again."
            ),
            None,
        ),
    }
}

#[derive(Clone)]
pub struct KbSearchServer {
    embed_client: Arc<EmbedClient>,
    qdrant: Arc<QdrantStore>,
    collection: String,
    canonical_data_path: PathBuf,
    /// Glob patterns (from `indexing.include`) used to restrict `get_document` to permitted file types.
    include_patterns: Arc<GlobSet>,
    /// Dynamic MCP server instructions, refreshed periodically with discovered metadata.
    instructions: Arc<RwLock<String>>,
    /// Per-tool description overlay, keyed by tool name, applied over the
    /// router's own `Tool` entries in `list_tools`/`get_tool` (see this
    /// module's hand-written `ServerHandler` impl). `#[tool_handler]`
    /// regenerates the router from `Self::tool_router()` on every call, so a
    /// description baked into a `#[tool(...)]` attribute can never be swapped
    /// at runtime — this overlay is what makes `descriptions::compose_all`
    /// (recomputed by the same periodic refresh that updates `instructions`
    /// above) actually reach `tools/list` and `tools/get` without a restart.
    /// `std::sync::RwLock`, not tokio's: `get_tool` is generated as a
    /// non-async fn by the trait, so it cannot `.await` a tokio lock.
    description_overlay: Arc<RwLock<HashMap<String, String>>>,
    /// Live config handle. Every tool call fetches its own fresh snapshot via
    /// [`Self::config`] rather than caching one at construction, so `POST
    /// /admin/reload` is observed by the very next call — see that method's doc
    /// comment for exactly which settings this makes dynamic.
    config: crate::config::SharedConfig,
    /// The shared, cached schema tree — built once at server startup and kept
    /// current by the reindex worker (which rebuilds it before indexing any
    /// dirty schema file) and by `update_schema`'s own synchronous rebuild.
    /// `get_schema`, `update_schema`, and the write path all read this instead of
    /// re-walking the knowledge base on every call; see `schema::SharedSchemaCache`.
    schema_cache: crate::schema::SharedSchemaCache,
    rerank_client: Option<Arc<RerankClient>>,
    /// Handle to the document metadata index, opened on first use and held for the
    /// process lifetime. Under WAL this reader coexists with the short-lived writer
    /// pools that the reindex worker and CLI open.
    ///
    /// Lazy rather than constructor-injected so building a server stays synchronous
    /// and infallible with respect to SQLite availability.
    state_db: Arc<tokio::sync::OnceCell<StateDb>>,
    /// The dirty-path queue every write tool marks paths on (see
    /// `write::WriteDeps::queue`) and `write_raw_file`/`update_schema` call
    /// `mark_full` on directly. `server::run_server` constructs exactly one
    /// `ReindexQueue`, clones this `Arc` into `KbSearchServer`, `UiState`,
    /// `WebhookState`, and `reindex::run_worker` — all four MUST share the same
    /// instance, since the worker only ever drains the one it was handed. There
    /// is no ambient/global fallback (see `reindex::ReindexQueue`'s doc
    /// comment); a server built without one has no way to get indexing to
    /// happen at all, which is deliberate — it makes "which queue does this
    /// producer feed" a constructor argument instead of a runtime assumption.
    reindex_queue: Arc<crate::reindex::ReindexQueue>,
}

/// Build an include `GlobSet` for MCP path filtering, with a `**/*.md` fallback
/// when no patterns are valid. Per-pattern parsing uses [`crate::ingest::parse_globs`]
/// so both sites share the same skip-and-warn policy; this function keeps its own
/// failure policy (fall back to `**/*.md` on empty or builder error) distinct from
/// `ingest::build_globset`, which propagates errors to the caller.
fn build_include_globset(patterns: &[String]) -> GlobSet {
    let (mut builder, valid_count) = crate::ingest::parse_globs(patterns);
    if valid_count == 0 {
        warn!("No valid include patterns configured — falling back to **/*.md");
        builder.add(Glob::new("**/*.md").unwrap());
    }
    builder.build().unwrap_or_else(|e| {
        error!(
            "Failed to build include globset: {} — falling back to **/*.md",
            e
        );
        let mut fallback = GlobSetBuilder::new();
        fallback.add(Glob::new("**/*.md").unwrap());
        fallback
            .build()
            .expect("hardcoded fallback glob '**/*.md' must compile")
    })
}

#[tool_router]
impl KbSearchServer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        embed_client: Arc<EmbedClient>,
        qdrant: Arc<QdrantStore>,
        collection: String,
        data_path: PathBuf,
        include_patterns: &[String],
        instructions: Arc<RwLock<String>>,
        config: crate::config::SharedConfig,
        schema_cache: crate::schema::SharedSchemaCache,
        rerank_client: Option<Arc<RerankClient>>,
        reindex_queue: Arc<crate::reindex::ReindexQueue>,
        description_overlay: Arc<RwLock<HashMap<String, String>>>,
    ) -> anyhow::Result<Self> {
        let canonical_data_path = data_path.canonicalize().with_context(|| {
            format!("Failed to canonicalize data path: {}", data_path.display())
        })?;
        Ok(Self {
            embed_client,
            qdrant,
            collection,
            canonical_data_path,
            // Compiled once from the config snapshot at construction time — a
            // `POST /admin/reload` that changes `indexing.include` does NOT change
            // what get_document accepts until a restart. See
            // `reload.rs`'s "indexing.include (MCP get_document path filter)" entry.
            include_patterns: Arc::new(build_include_globset(include_patterns)),
            instructions,
            config,
            schema_cache,
            rerank_client,
            state_db: Arc::new(tokio::sync::OnceCell::new()),
            reindex_queue,
            description_overlay,
        })
    }

    /// A fresh snapshot of the live config — a lock acquisition plus an `Arc`
    /// clone, mirroring `schema::load_shared`. Every tool call fetches its own
    /// snapshot here rather than reading a value captured at construction, so a
    /// `POST /admin/reload` swap is observed starting with the very next call.
    fn config(&self) -> Arc<ResolvedConfig> {
        crate::config::load_shared_config(&self.config)
    }

    /// A fresh `tool_router()` with every name in the live `mcp.disabled_tools`
    /// disabled (`ToolRouter::disable_route`) — the one place `list_tools`,
    /// `get_tool`, and `call_tool` all build their router from, so the three
    /// can never disagree about which tools are currently enabled. Reads
    /// `self.config()` fresh on every call, same live-reload contract as every
    /// other config consumer here: a `POST /admin/reload` that changes
    /// `mcp.disabled_tools` is observed starting with the very next request,
    /// no restart or metadata-refresh-tick wait required (`config::validate`
    /// has already rejected an unknown or all-disabling set, so every name
    /// reaching `disable_route` here is one of `descriptions::TOOL_NAMES`).
    fn enabled_tool_router(&self) -> ToolRouter<Self> {
        let mut router = Self::tool_router();
        for name in &self.config().mcp.disabled_tools {
            router.disable_route(name.clone());
        }
        router
    }

    /// Apply the live description overlay to one router-provided `Tool`,
    /// recovering from a poisoned lock the same way `get_info` does for
    /// `instructions`. A tool name absent from the overlay (should not
    /// happen — every tool in `tool_router()` is one of `descriptions::TOOL_NAMES`)
    /// is left with whatever the router itself produced, which is `None`
    /// since every `#[tool(...)]` attribute below carries no `description`.
    fn overlay_description(&self, mut tool: Tool) -> Tool {
        let overlay = self.description_overlay.read().unwrap_or_else(|poisoned| {
            warn!("Description overlay RwLock poisoned on read; using last value");
            poisoned.into_inner()
        });
        if let Some(desc) = overlay.get(tool.name.as_ref()) {
            tool.description = Some(std::borrow::Cow::Owned(desc.clone()));
        }
        tool
    }

    /// Apply the live per-instance `search` restrictions to one
    /// router-provided `Tool`'s `input_schema` (#286) — the schema
    /// counterpart to `overlay_description` above, called from the same two
    /// sites (`list_tools`/`get_tool`) so a disabled granularity or
    /// `heading_prefix` disappears from what a caller/model can even see,
    /// not just from prose it might skip: the `granularity` property's
    /// `enum` becomes the effective set and its `description` becomes
    /// `descriptions::granularity_description` for that set (the full
    /// per-value text; the tool description carries only
    /// `descriptions::granularity_summary`), and `heading_prefix` is removed while
    /// `chunking.heading_metadata` is off. Reads `self.config()` live, same
    /// fresh-snapshot-per-call contract as every other config consumer
    /// here. A no-op for every tool but `search`.
    ///
    /// `Tool::input_schema` is `Arc<JsonObject>`, shared with every OTHER
    /// concurrent caller of `list_tools`/`get_tool` until the next one — this
    /// clones the object before editing rather than mutating through the
    /// `Arc`, so one request's overlay can never leak into another's, or
    /// into the router's own cached `Tool` (see `ResolvedConfig::
    /// effective_granularities`'s doc comment for why all three consumers of
    /// that helper, this one included, must read a fresh snapshot rather
    /// than share mutable state).
    fn overlay_input_schema(&self, mut tool: Tool) -> Tool {
        if tool.name.as_ref() != "search" {
            return tool;
        }
        let config = self.config();
        let effective = config.effective_granularities();

        let mut schema: JsonObject = (*tool.input_schema).clone();
        if let Some(properties) = schema.get_mut("properties").and_then(|v| v.as_object_mut()) {
            if let Some(granularity) = properties
                .get_mut("granularity")
                .and_then(|v| v.as_object_mut())
            {
                // Strings only: an omitted granularity is expressed by leaving
                // the optional property out (`tool_schema::compact` drops its
                // `null` type arm too); the server still accepts an explicit
                // `null`, since the field is an `Option<String>`.
                let values: Vec<serde_json::Value> = effective
                    .iter()
                    .map(|g| serde_json::Value::String(g.as_str().to_string()))
                    .collect();
                granularity.insert("enum".to_string(), serde_json::Value::Array(values));
                granularity.insert(
                    "description".to_string(),
                    serde_json::Value::String(crate::descriptions::granularity_description(
                        &effective,
                    )),
                );
            }
            if !config.chunking.heading_metadata {
                properties.remove("heading_prefix");
            }
            // Properties whose static doc comments assume every granularity
            // and mode is on: rewritten, or removed when nothing enabled can
            // use them.
            for (name, description) in
                crate::descriptions::search_property_descriptions(&effective, config.search.hybrid)
            {
                match description {
                    Some(text) => {
                        if let Some(property) =
                            properties.get_mut(name).and_then(|v| v.as_object_mut())
                        {
                            property
                                .insert("description".to_string(), serde_json::Value::String(text));
                        }
                    }
                    None => {
                        properties.remove(name);
                    }
                }
            }
        }
        tool.input_schema = Arc::new(schema);
        tool
    }

    /// Write a non-document file into the KB, commit it, and queue a full reconcile.
    ///
    /// Used for a directory's schema file, which is versioned and synced like a
    /// document but is not itself indexed. The write goes to a temp file and is renamed into place, so a
    /// failure part-way through the *filesystem* write cannot leave a half-written
    /// schema that the next rebuild would refuse.
    ///
    /// `commit_and_sync` below has its own two-phase failure mode — see
    /// `git::CommitSyncError` — and is rolled back exactly like `write_document`'s: a
    /// `PreCommit` failure undoes the filesystem write (remove + `unstage` for a
    /// brand-new schema file that has no HEAD content to fall back to;
    /// `restore_from_head` for an overwrite of an existing, already-tracked one) and
    /// reports [`not_saved_error`], or [`not_saved_unverified_error`] if that rollback
    /// itself fails. A `PostCommit` failure leaves the local commit in place — it is
    /// real — is logged, and is reported to the caller as a plain success: the
    /// pending sync is a server-side concern, never part of the tool result.
    ///
    /// `reindex::mark_full` fires only once the commit has actually landed locally
    /// (synced or not), never on a rolled-back write: queuing a
    /// full reconcile against a schema change that was never actually committed (or,
    /// worse, against a filesystem/git state a failed rollback left inconsistent)
    /// would be pointless at best and actively misleading at worst — the reconcile
    /// would revalidate every document under the scope against content that is not,
    /// in fact, what's in git history.
    ///
    /// `replaces`, when set, is a sibling file this write supersedes — the legacy
    /// `.kb-schema.yaml` that `update_schema` migrates to `.schema.yaml`. It is
    /// removed from disk after `rel_path` is installed and staged in the SAME commit
    /// (`git add` of a deleted tracked path records the removal), so the directory is
    /// never left with both names or with neither. A pre-commit failure restores it
    /// alongside `rel_path`'s own rollback. An untracked `replaces` (one git never
    /// knew about) is removed from disk without being named in the commit, and
    /// rewritten from memory on rollback.
    ///
    /// Every message names the scope directory (`schema::scope_label`), never the
    /// file: this text reaches the model.
    async fn write_raw_file(
        &self,
        git_lock: &git::GitLock,
        rel_path: &str,
        content: &str,
        commit_message: &str,
        replaces: Option<&str>,
    ) -> Result<RawWrite, McpError> {
        let config = self.config();
        let scope = schema_file_scope(rel_path);

        // Resolved before anything touches the filesystem: a bad `<NAME>_FILE`
        // (both forms set, empty or unreadable file) must fail the call with nothing
        // written, since an early return after the rename below would leave an
        // uncommitted schema file in the working tree with no rollback.
        let token = crate::secrets::git_token(&config).map_err(credential_error)?;

        // Same resolver the document write tools use. Joining the data root with a
        // caller-supplied path is NOT sufficient on its own: the knowledge base is a
        // synced git repo, and git materializes tracked symlinks on checkout, so a
        // hostile upstream commit could otherwise redirect this write outside the KB.
        let abs_path = crate::write::resolve_safe_write_path(&self.canonical_data_path, rel_path)
            .map_err(|e| {
            McpError::invalid_params(format!("Invalid schema path: {}", e), None)
        })?;
        let abs_replaced = replaces
            .map(|p| {
                crate::write::resolve_safe_write_path(&self.canonical_data_path, p)
                    .map(|abs| (p, abs))
                    .map_err(|e| {
                        McpError::invalid_params(format!("Invalid schema path: {}", e), None)
                    })
            })
            .transpose()?;

        if let Some(parent) = abs_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                error!("Failed to create directory '{}': {}", parent.display(), e);
                McpError::internal_error(format!("Failed to create directory: {}", e), None)
            })?;
        }

        // Re-check after creating the directory: `resolve_safe_write_path` can only
        // canonicalize ancestors that existed at the time, so a newly created path
        // component is verified here.
        crate::write::resolve_safe_write_path(&self.canonical_data_path, rel_path)
            .map_err(|e| McpError::invalid_params(format!("Invalid schema path: {}", e), None))?;

        // Whether this call is creating `rel_path` for the first time or overwriting
        // an existing, already-tracked one — determines which rollback primitive
        // applies if `commit_and_sync` fails before landing (see the match below).
        // Checked as late as possible, immediately before the write, to keep the
        // TOCTOU window against a concurrent writer as small as the temp+rename
        // strategy below allows.
        let is_new = !abs_path.exists();

        // Unique per call, not merely per process: two concurrent requests inside one
        // server would otherwise share a temp path and silently clobber each other,
        // with the loser still reporting success.
        static WRITE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = WRITE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let temp_path = abs_path.with_extension(format!("tmp-{}-{}", std::process::id(), seq));
        tokio::fs::write(&temp_path, content.as_bytes())
            .await
            .map_err(|e| {
                error!("Failed to write '{}': {}", temp_path.display(), e);
                McpError::internal_error(format!("Failed to write file: {}", e), None)
            })?;
        tokio::fs::rename(&temp_path, &abs_path)
            .await
            .map_err(|e| {
                error!("Failed to install '{}': {}", abs_path.display(), e);
                McpError::internal_error(format!("Failed to write file: {}", e), None)
            })?;

        // `commit_and_sync` distinguishes WHERE it failed — see `git::CommitSyncError`
        // — and the two phases demand opposite handling, exactly as in
        // `write_document`. A `PreCommit` failure means HEAD never moved, so the
        // filesystem write above is rolled back and reported as "nothing changed". A
        // `PostCommit` failure means the commit is a real, durable part of local
        // history — rolling it back here would silently undo a schema change that
        // genuinely happened, so it is left alone, logged, and reported to the caller
        // as an ordinary success.
        let data_path_str = self.canonical_data_path.to_str().unwrap_or_default();

        // The superseded file goes under the same lock as the commit that records its
        // removal. Its content is kept for the rollback of an untracked one, which
        // has no HEAD copy to restore from.
        let mut commit_paths: Vec<&str> = vec![rel_path];
        let mut removed: Option<(&str, std::path::PathBuf, String, bool)> = None;
        if let Some((replaced_rel, abs_replaced)) = abs_replaced {
            // Read, classify and remove as one step, so every failure unwinds the same
            // way: nothing is committed yet, and leaving the install of the new file
            // in place would leave the directory holding both names.
            let superseded: anyhow::Result<Option<(String, bool)>> = async {
                let previous = match tokio::fs::read_to_string(&abs_replaced).await {
                    Ok(previous) => previous,
                    // Already gone: nothing left to supersede.
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                    Err(e) => {
                        return Err(
                            anyhow::Error::new(e).context("could not read the superseded file")
                        );
                    }
                };
                let tracked = git::is_tracked(git_lock, data_path_str, replaced_rel).await?;
                tokio::fs::remove_file(&abs_replaced)
                    .await
                    .context("could not remove the superseded file")?;
                Ok(Some((previous, tracked)))
            }
            .await;
            match superseded {
                Ok(Some((previous, tracked))) => {
                    if tracked {
                        commit_paths.push(replaced_rel);
                    }
                    removed = Some((replaced_rel, abs_replaced, previous, tracked));
                }
                Ok(None) => {}
                Err(e) => {
                    error!("Failed to retire the superseded schema file for {scope}: {e:#}");
                    let undo = if is_new {
                        tokio::fs::remove_file(&abs_path)
                            .await
                            .map_err(anyhow::Error::new)
                    } else {
                        git::restore_from_head(git_lock, data_path_str, rel_path).await
                    };
                    return Err(match undo {
                        Ok(()) => not_saved_error(&format!("The schema change for '{scope}'")),
                        Err(undo) => {
                            error!("Failed to undo schema write for {scope}: {undo:#}");
                            not_saved_unverified_error(&format!("The schema change for '{scope}'"))
                        }
                    });
                }
            }
        }

        match git::commit_and_sync(
            git_lock,
            config.source.git_url.as_deref(),
            &config.source.branch,
            data_path_str,
            token.as_deref(),
            &commit_paths,
            commit_message,
            &config.write.commit_author_name,
            &config.write.commit_author_email,
        )
        .await
        {
            // `rebased_paths` needs no handling: `mark_full` below covers every path.
            Ok(_) => {}

            Err(git::CommitSyncError::PreCommit(source)) => {
                error!(
                    "commit_and_sync pre-commit failure writing schema '{}', rolling back: {:#}",
                    rel_path, source
                );

                // For a brand-new schema file, there is no HEAD content to restore
                // to — remove it from disk directly and unstage whatever `git add`
                // staged. For an overwrite of an existing schema, HEAD already has
                // the previous content, so restore it (this also un-stages any
                // partial `git add`, in one step).
                let mut rollback = if is_new {
                    match tokio::fs::remove_file(&abs_path).await {
                        Ok(()) => git::unstage(git_lock, data_path_str, rel_path).await,
                        Err(e) => Err(anyhow::Error::new(e)
                            .context("Failed to remove newly-written schema file during rollback")),
                    }
                } else {
                    git::restore_from_head(git_lock, data_path_str, rel_path).await
                };
                // Put the superseded file back too: from HEAD when git tracks it
                // (which also unstages its removal), from memory otherwise.
                if let Some((replaced_rel, abs_replaced, previous, tracked)) = &removed {
                    let restored = if *tracked {
                        git::restore_from_head(git_lock, data_path_str, replaced_rel).await
                    } else {
                        tokio::fs::write(abs_replaced, previous.as_bytes())
                            .await
                            .map_err(|e| {
                                anyhow::Error::new(e)
                                    .context("Failed to rewrite superseded schema file")
                            })
                    };
                    if let Err(e) = restored {
                        rollback = Err(match rollback {
                            Ok(()) => e,
                            Err(first) => first.context(format!("{e:#}")),
                        });
                    }
                }

                // The cause was logged above and stays server-side.
                return match rollback {
                    Ok(()) => Err(not_saved_error(&format!("The schema change for '{scope}'"))),
                    // The rollback ITSELF failed — a third, worse state than either of
                    // the above. The schema file may now be gone/changed on disk with
                    // no corresponding commit, or the index may not match HEAD.
                    // Report it distinctly and loudly rather than letting it
                    // masquerade as a clean no-op.
                    Err(rollback_err) => {
                        error!(
                            "Rollback FAILED after a pre-commit git failure writing schema \
                             '{}': {:#}. Original cause: {:#}. Filesystem and git state may \
                             now be inconsistent.",
                            rel_path, rollback_err, source
                        );
                        Err(not_saved_unverified_error(&format!(
                            "The schema change for '{scope}'"
                        )))
                    }
                };
            }

            Err(git::CommitSyncError::PostCommit { sha, source }) => {
                warn!(
                    "commit_and_sync post-commit (sync) failure writing schema '{}', commit {} \
                     stands uncorrected: {:#}",
                    rel_path, sha, source
                );

                // The commit landed locally regardless of push status, so the schema
                // change is real and durable as far as this clone's git history is
                // concerned — queue the same full reconcile a clean success would.
                // See this method's doc comment for why that reconcile must NOT run
                // on the rolled-back (PreCommit) branch above but must here.
                self.reindex_queue.mark_full();

                return Ok(RawWrite::Committed);
            }

            // This commit is dropped: the branch is back at its pre-commit HEAD, and
            // that reset restored every tracked file. An untracked superseded file
            // never was in git, so it is rewritten from memory. The caller syncs
            // again and re-applies.
            Err(git::CommitSyncError::Conflict { source }) => {
                warn!("Remote changed underneath the schema write for {scope}: {source:#}");
                if let Some((_, abs_replaced, previous, false)) = &removed
                    && let Err(e) = tokio::fs::write(abs_replaced, previous.as_bytes()).await
                {
                    error!("Failed to rewrite superseded schema file for {scope}: {e}");
                }
                return Ok(RawWrite::Conflict);
            }
        };

        // A schema change revalidates its whole subtree via the schema fingerprint —
        // any document under this scope can flip from valid to invalid or vice versa —
        // and there is no cheap way to enumerate exactly which paths that touches
        // without a walk. Rather than approximate it, mark a full reconcile: the
        // worker will scan, and `index_paths`' existing schema-fingerprint check
        // (unrelated to this reconcile's OWN full-walk vs scoped distinction) is what
        // actually catches the affected documents once it re-reads them.
        self.reindex_queue.mark_full();

        Ok(RawWrite::Committed)
    }

    /// Documents already under `rel_dir` that a candidate schema would reject.
    ///
    /// Answered from the metadata index rather than by re-reading markdown: every
    /// document's frontmatter is stored as JSON, so this is a query.
    async fn documents_broken_by(
        &self,
        rel_dir: &std::path::Path,
        schemas: &SchemaCache,
        candidate_file: &crate::schema::SchemaFile,
    ) -> Result<Vec<serde_json::Value>, McpError> {
        let index = self.state_db().await.map_err(|e| {
            error!("Schema dry-run could not open the metadata index: {:#}", e);
            McpError::internal_error(
                format!("Cannot check existing documents: {e}. Index unavailable."),
                None,
            )
        })?;

        let prefix = if rel_dir.as_os_str().is_empty() {
            None
        } else {
            Some(format!("{}/", rel_dir.to_string_lossy()))
        };

        let query = DocumentQuery {
            path_prefix: prefix,
            // The whole point is completeness; a truncated check would report a clean
            // dry-run for a change that breaks documents beyond the page.
            limit: u32::MAX as u64,
            ..Default::default()
        };

        let listing = index.query_documents(&query).await.map_err(|e| {
            error!("Schema dry-run query failed: {:#}", e);
            McpError::internal_error(format!("Cannot check existing documents: {e}"), None)
        })?;

        let mut casualties = Vec::new();
        for doc in &listing.documents {
            let Some(map) = doc.frontmatter.as_object() else {
                continue;
            };

            // Resolve each document against ITS OWN effective schema under the proposed
            // edit, not against the edited directory's. A descendant scope that
            // redefines the field being changed is unaffected by this edit, and
            // validating it against the parent's new rule would report a casualty that
            // does not exist — blocking a legitimate change.
            let doc_path = std::path::Path::new(&doc.file_path);
            let effective = schemas.resolve_with_candidate(doc_path, rel_dir, candidate_file);
            if effective.is_none() {
                continue;
            }
            let effective = effective.expect("checked above");

            let mut frontmatter: std::collections::HashMap<String, serde_json::Value> =
                map.clone().into_iter().collect();
            // The real indexing path fills in schema defaults before validating, so
            // skipping that here reports a required field WITH a default as breaking
            // every document that omits it — blocking a genuinely safe change and
            // pushing the operator toward `force`, which also bypasses the real checks.
            validate::apply_defaults(&mut frontmatter, &effective);
            let errors = validate::validate_frontmatter(&frontmatter, &effective);
            if let Some(first) = errors.first() {
                casualties.push(serde_json::json!({
                    "path": doc.file_path,
                    "reason": first.message,
                    "error_count": errors.len(),
                }));
            }
        }

        Ok(casualties)
    }

    /// The document metadata index, opened on first use.
    async fn state_db(&self) -> anyhow::Result<&StateDb> {
        self.state_db
            .get_or_try_init(|| async {
                let path = self.config().state_db_path();
                StateDb::new(std::path::Path::new(&path))
                    .await
                    .with_context(|| format!("Failed to open state DB at {}", path))
            })
            .await
    }

    /// Refuse a `search` filter that can only ever match nothing, listing what
    /// would match instead — the teach-on-error replacement for enumerating
    /// every vocabulary in the server instructions. Applies to every
    /// granularity, so the metadata-index (no query) and Qdrant (query)
    /// backends share it.
    ///
    /// - A field is unknown when no schema scope declares it
    ///   (`SchemaCache::declared_field_paths`), it is not a built-in filter key
    ///   ([`builtin_filter_field`]), and no document carries it.
    /// - A value is refused when the field has a closed value set across the
    ///   governing scopes (`SchemaCache::filter_closed_values`, narrowed by
    ///   `path_prefix` as [`retrieval::normalize_path_needle`] reads it, the
    ///   needle `search` itself matches on), the value is outside it, and no
    ///   document uses it either — so a refusal never hides a match, whichever
    ///   scope the matching document sits in.
    /// - A filter is refused only when no document could match it: an `all_of`
    ///   as soon as one of its values is refused (no document can carry it),
    ///   but a scalar or array filter is an any-of and still matches through
    ///   its other values, so it is refused only when every value is.
    ///
    /// Fails open: when the metadata index cannot be opened or queried, the
    /// filter runs as given.
    async fn check_filter_vocabulary(&self, params: &SearchParams) -> Result<(), McpError> {
        if params.filters.as_ref().is_none_or(|f| f.0.is_empty()) {
            return Ok(());
        }
        let filters = parse_filters(&params.filters)?;
        let Ok(index) = self.state_db().await else {
            return Ok(());
        };
        let schemas = crate::schema::load_shared(&self.schema_cache);
        let config = self.config();
        let declared = schemas.declared_field_paths();
        let path_needle = retrieval::normalize_path_needle(params.path_prefix.as_deref());

        for (field, filter) in &filters {
            let known = declared.contains(field)
                || builtin_filter_field(field, &config)
                || index.field_in_use(field).await.unwrap_or(true);
            if !known {
                let mut names: std::collections::BTreeSet<String> = declared.clone();
                names.extend(
                    config
                        .effective_indexed_fields()
                        .into_iter()
                        .chain(crate::ingest::DERIVED_FIELDS.iter().map(|f| f.to_string())),
                );
                if let Ok(in_use) = index.fields_in_use(None, 500).await {
                    names.extend(in_use.into_iter().map(|(name, _)| name));
                }
                names.remove("file_path");
                let names: Vec<String> = names
                    .iter()
                    .map(|n| crate::server::sanitize_facet_value(n))
                    .collect();
                return Err(McpError::invalid_params(
                    format!(
                        "unknown filter field '{}'; filterable fields: {}",
                        crate::server::sanitize_facet_value(field),
                        capped_list(&names)
                    ),
                    Some(serde_json::json!({ "filterable_fields": capped_vec(&names) })),
                ));
            }

            let (values, needs_every_value) = match filter {
                FieldFilter::AnyOf(values) => (values, false),
                FieldFilter::AllOf(values) => (values, true),
                FieldFilter::Range { .. } => continue,
            };
            let Some(allowed) = schemas.filter_closed_values(field, path_needle) else {
                continue;
            };
            let outside: Vec<String> = values
                .iter()
                .filter(|v| !allowed.contains(*v))
                .cloned()
                .collect();
            if outside.is_empty() {
                continue;
            }
            let Ok(present) = index.values_present(field, &outside).await else {
                continue;
            };
            let refused: Vec<&String> = outside.iter().filter(|v| !present.contains(*v)).collect();
            // `all_of` needs a document to carry every value, so one that nothing can
            // match sinks the filter; an any-of still matches through the rest of its
            // values and is empty only when none of them could match.
            let matches_nothing = if needs_every_value {
                !refused.is_empty()
            } else {
                !refused.is_empty() && refused.len() == values.len()
            };
            if matches_nothing {
                let mut seen = std::collections::HashSet::new();
                let refused: Vec<String> = refused
                    .iter()
                    .map(|v| format!("'{}'", crate::server::sanitize_facet_value(v)))
                    .filter(|v| seen.insert(v.clone()))
                    .collect();
                let allowed: Vec<String> = allowed
                    .iter()
                    .map(|v| crate::server::sanitize_facet_value(v))
                    .collect();
                let verb = if refused.len() == 1 {
                    "is not an allowed value"
                } else {
                    "are not allowed values"
                };
                return Err(McpError::invalid_params(
                    format!(
                        "filter '{}': {} {verb}; allowed: {}",
                        crate::server::sanitize_facet_value(field),
                        capped_list(&refused),
                        capped_list(&allowed)
                    ),
                    Some(serde_json::json!({ "field": field, "allowed": capped_vec(&allowed) })),
                ));
            }
        }
        Ok(())
    }

    /// Resolve a `path_prefix` needle to the documents it actually matches (#182).
    ///
    /// The metadata index is the single authority for that question, so both query
    /// modes route through here and enumeration applies the identical `LIKE` inline
    /// (`state::StateDb::push_where`) — one needle can only ever mean one thing.
    /// `None` in gives `None` out: no filter, matching everything.
    async fn resolve_path_filter(
        &self,
        path_prefix: Option<&str>,
    ) -> Result<Option<retrieval::PathFilter>, McpError> {
        let Some(needle) = retrieval::normalize_path_needle(path_prefix) else {
            return Ok(None);
        };
        let index = self.state_db().await.map_err(|e| {
            error!("search could not open the metadata index: {:#}", e);
            McpError::internal_error(format!("Document index unavailable: {}", e), None)
        })?;
        let matches = index
            .paths_matching(needle, retrieval::PATH_FILTER_MAX_PATHS)
            .await
            .map_err(|e| {
                error!("search could not resolve path_prefix '{}': {:#}", needle, e);
                McpError::internal_error(format!("path_prefix lookup failed: {}", e), None)
            })?;
        Ok(Some(matches.into()))
    }

    /// Build a `RetrievalDeps` bundle from this server's fields.
    fn deps(&self) -> RetrievalDeps<'_, EmbedClient, QdrantStore> {
        RetrievalDeps {
            embed_client: &self.embed_client,
            qdrant: &self.qdrant,
            collection: &self.collection,
            data_path: &self.canonical_data_path,
            include_patterns: &self.include_patterns,
            reranker: self
                .rerank_client
                .as_ref()
                .map(|c| c.as_ref() as &(dyn crate::rerank::Reranker + Send + Sync)),
        }
    }

    #[tool(annotations(read_only_hint = true))]
    async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        validate_search_params(&params)?;
        self.check_filter_vocabulary(&params).await?;

        let query_present = query_is_present(&params.query);
        // This gate reads its own config snapshot; each downstream handler
        // (search_chunks/search_grouped/search_sections/search_enumerate) then
        // takes a fresh one via self.config(), so a `POST /admin/reload`
        // landing between the two is visible to the handler. That is
        // harmless: every handler re-checks what it depends on
        // (`search_sections` the heading_metadata flag, `heading_prefix_condition`
        // likewise) against its own snapshot rather than trusting this gate.
        let config = self.config();
        let effective_granularities = config.effective_granularities();
        let section_gated_off = config.section_gated_off();
        let granularity = resolve_search_granularity(
            query_present,
            params.granularity.as_deref(),
            &effective_granularities,
            section_gated_off,
        )?;

        debug!(
            query_present,
            granularity = ?granularity,
            has_filters = params.filters.is_some(),
            "search called"
        );

        match (query_present, granularity) {
            (false, Granularity::Chunk) => Err(McpError::invalid_params(
                if effective_granularities.contains(&Granularity::Document) {
                    "chunk granularity requires a query; omit granularity (or set it to \
                     'document') to enumerate without one"
                } else {
                    "chunk granularity requires a query"
                }
                .to_string(),
                None,
            )),
            (false, Granularity::Section) => Err(McpError::invalid_params(
                "section granularity requires a query — there is nothing to rank \
                 sections by without one"
                    .to_string(),
                None,
            )),
            (false, Granularity::Document) => self.search_enumerate(&params).await,
            (true, Granularity::Chunk) => self.search_chunks(&params).await,
            (true, Granularity::Document) => self.search_grouped(&params).await,
            (true, Granularity::Section) => self.search_sections(&params).await,
        }
    }

    /// query+chunk: the original `search` tool's behavior, unchanged output shape.
    async fn search_chunks(&self, params: &SearchParams) -> Result<CallToolResult, McpError> {
        let query = params.query.as_deref().unwrap_or_default();

        validate_path_prefix(&params.path_prefix)?;
        // #132 (audit follow-up): `fields` selects per-result frontmatter fields
        // from the document metadata index — a chunk result is hydrated straight
        // from the Qdrant chunk payload and never joins that index, so `fields`
        // has nothing to draw from here. `search_grouped`/`build_document_query`
        // (document granularity, query or enumeration) are its only real
        // consumers; reject explicitly here rather than silently drop it the way
        // it used to (this was the SAME silent-no-op bug `explain` at document
        // granularity is, just for the mirror-image parameter/granularity pair).
        if params.fields.is_some() {
            return Err(fields_rejection(
                Granularity::Chunk,
                &self.config().effective_granularities(),
            ));
        }
        let schemas = crate::schema::load_shared(&self.schema_cache);
        // Fetched once per call so every field below — including
        // reranking.candidate_limit — reflects the same live snapshot, rather than
        // racing a concurrent `POST /admin/reload` mid-request.
        let config = self.config();
        let conditions = build_query_conditions(params, &config, &schemas)?;

        let limit = resolve_limit(
            params.limit,
            config.search.default_limit,
            config.search.max_limit,
        );

        let filters = SearchFilters { conditions };

        let modified_after = params
            .modified_after
            .as_deref()
            .map(parse_date_to_timestamp)
            .transpose()
            .map_err(|e| McpError::invalid_params(e, None))?;
        let modified_before = params
            .modified_before
            .as_deref()
            .map(parse_date_to_timestamp)
            .transpose()
            .map_err(|e| McpError::invalid_params(e, None))?;
        let explain = params.explain.unwrap_or(false);
        let path_filter = self
            .resolve_path_filter(params.path_prefix.as_deref())
            .await?;
        let opts = SearchOptions {
            limit,
            min_score: params.min_score.or(config.search.min_score),
            hybrid: config.search.hybrid,
            rrf_candidates: config.search.rrf_candidates as u64,
            // Config-enabled AND confirmed available on this server right now —
            // see `status::IndexStatus::phrase_matching_available`'s doc comment
            // for why an unconfirmed index must not attempt a phrase arm.
            phrase: config.search.phrase && crate::status::INDEX_STATUS.phrase_matching_available(),
            explain,
            modified_after,
            modified_before,
            path_filter,
            rerank_candidate_limit: config.reranking.as_ref().map(|r| r.candidate_limit as u64),
            diversity_max_per_document: config.search.diversity_max_per_document,
        };

        // The `explain` mode label must describe the query that actually ran, not
        // the config that permitted it. An enabled phrase arm only fires when the
        // query carries a quoted span, and with `hybrid` off a phrase query is
        // still a two-arm RRF fusion — reporting that as "dense cosine" would be a
        // lie in exactly the situation someone turned `explain` on to diagnose.
        let phrase_arm_ran = opts.phrase && !retrieval::extract_phrases(query).1.is_empty();

        // #224: `offset` pages over the already fused/reranked/diversity-capped
        // ranking — see `retrieval::search_paged`'s doc comment for the funnel
        // placement and the depth bound `offset_truncated` reports against.
        let offset = params.offset.unwrap_or(0);
        let outcome = retrieval::search_paged(&self.deps(), query, &filters, &opts, offset)
            .await
            .map_err(|e| match e {
                retrieval::SearchError::Embed(err) => {
                    error!("Embedding query failed: {:#}", err);
                    McpError::internal_error("Failed to generate query embedding".to_string(), None)
                }
                retrieval::SearchError::Search(err) => {
                    error!("Qdrant search failed: {:#}", err);
                    McpError::internal_error("Search query failed".to_string(), None)
                }
                retrieval::SearchError::Document(err) => {
                    error!("Document metadata lookup failed: {:#}", err);
                    McpError::internal_error("Document metadata lookup failed".to_string(), None)
                }
            })?;
        let path_prefix_truncated = outcome.path_prefix_truncated;
        let offset_truncated = outcome.offset_truncated;
        let results = outcome.results;

        debug!(
            result_count = results.len(),
            path_prefix_truncated, offset_truncated, "search returned results"
        );

        let mode = match (config.search.hybrid, phrase_arm_ran) {
            (true, true) => "hybrid RRF + phrase",
            (true, false) => "hybrid RRF",
            (false, true) => "dense + phrase RRF",
            (false, false) => "dense cosine",
        };

        let mut response = build_chunk_search_payload(
            &results,
            &self.canonical_data_path,
            explain,
            mode,
            path_prefix_truncated,
            offset_truncated,
        );
        if params.heading_prefix.is_some() {
            annotate_heading_results(&mut response, heading_results_indexing_note());
        }
        Ok(CallToolResult::structured(response))
    }

    /// query+document: Qdrant grouped by `file_path`, collapsed to each document's
    /// best-scoring chunk, hydrated with document-level metadata from `StateDb`. New
    /// combination — there is no output shape to preserve, only the two the design
    /// specifies: the document shape `search_enumerate` returns, plus a per-document
    /// `score`, and no `total`/`has_more` (grouped vector search cannot back either).
    async fn search_grouped(&self, params: &SearchParams) -> Result<CallToolResult, McpError> {
        let query = params.query.as_deref().unwrap_or_default();

        // #132: `explain` produces a per-result score breakdown (dense/sparse/
        // phrase/pre-rerank) from the per-arm scores `search_chunks` collects on
        // each chunk `SearchResult`. This path collapses every document to its
        // best-scoring chunk via Qdrant's server-side grouping (see this
        // function's doc comment) with no explain-mode fusion query behind it —
        // there is no per-arm breakdown to attach, so `GroupedDocument` carries
        // none. Implementing the real thing would mean threading an `explain`
        // flag through `RetrievalStore::search_grouped` and adding a
        // client-side-fused, per-arm-scored grouped-query path in `qdrant.rs`
        // mirroring `hybrid_search_explain` — real work, out of scope for this
        // fix. Reject explicitly instead of the silent no-op this used to be:
        // a caller that turned `explain` on to diagnose a confusing document
        // ranking deserves an error telling them so, not a response that quietly
        // drops the one thing they asked for.
        if params.explain == Some(true) {
            return Err(explain_rejection(
                Granularity::Document,
                &self.config().effective_granularities(),
            ));
        }

        validate_path_prefix(&params.path_prefix)?;
        validate_fields_count(&params.fields)?;
        let schemas = crate::schema::load_shared(&self.schema_cache);
        let config = self.config();
        let conditions = build_query_conditions(params, &config, &schemas)?;

        let limit = resolve_limit(
            params.limit,
            config.search.default_limit,
            config.search.max_limit,
        );

        let filters = SearchFilters { conditions };

        let modified_after = params
            .modified_after
            .as_deref()
            .map(parse_date_to_timestamp)
            .transpose()
            .map_err(|e| McpError::invalid_params(e, None))?;
        let modified_before = params
            .modified_before
            .as_deref()
            .map(parse_date_to_timestamp)
            .transpose()
            .map_err(|e| McpError::invalid_params(e, None))?;

        let path_filter = self
            .resolve_path_filter(params.path_prefix.as_deref())
            .await?;
        let opts = SearchOptions {
            limit,
            min_score: params.min_score.or(config.search.min_score),
            // Grouped queries now share `search_chunks`'s dense/sparse/phrase arms
            // (see `retrieval::search_grouped`'s doc comment) — a query only the
            // sparse or phrase arm can find must be just as retrievable at
            // document granularity as at chunk granularity. Reranking and
            // diversity still don't apply here: there is one row per document
            // already, so per-document diversity is a no-op and there is no
            // per-chunk candidate pool for a cross-encoder to rerank.
            hybrid: config.search.hybrid,
            rrf_candidates: config.search.rrf_candidates as u64,
            phrase: config.search.phrase && crate::status::INDEX_STATUS.phrase_matching_available(),
            // Always false: `explain: true` is rejected above (#132) before this
            // point is reached, and `retrieval::search_grouped` never reads this
            // field regardless — grouped results carry no per-arm score to
            // explain in the first place. `false` here (rather than
            // `params.explain.unwrap_or(false)`) says so plainly instead of
            // leaving a dead-looking reference to a value that can only be
            // `None`/`Some(false)` by now.
            explain: false,
            modified_after,
            modified_before,
            path_filter,
            rerank_candidate_limit: None,
            diversity_max_per_document: None,
        };

        let index = self.state_db().await.map_err(|e| {
            error!(
                "search (query+document) could not open the metadata index: {:#}",
                e
            );
            McpError::internal_error(format!("Document index unavailable: {}", e), None)
        })?;

        // #224: same paging entry point and depth-bound reasoning as
        // `search_chunks` — see `retrieval::search_grouped`'s doc comment.
        let offset = params.offset.unwrap_or(0);
        let outcome = retrieval::search_grouped(
            &self.deps(),
            index,
            query,
            &filters,
            &opts,
            params.fields.as_deref(),
            offset,
        )
        .await
        .map_err(|e| match e {
            retrieval::SearchError::Embed(err) => {
                error!("Embedding query failed: {:#}", err);
                McpError::internal_error("Failed to generate query embedding".to_string(), None)
            }
            retrieval::SearchError::Search(err) => {
                error!("Qdrant grouped search failed: {:#}", err);
                McpError::internal_error("Search query failed".to_string(), None)
            }
            retrieval::SearchError::Document(err) => {
                error!("Document metadata lookup failed: {:#}", err);
                McpError::internal_error("Document metadata lookup failed".to_string(), None)
            }
        })?;
        let path_prefix_truncated = outcome.path_prefix_truncated;
        let offset_truncated = outcome.offset_truncated;
        let documents = outcome.documents;

        let mut response = build_grouped_search_payload(
            &documents,
            params.fields.as_deref(),
            path_prefix_truncated,
            offset_truncated,
        );
        if params.heading_prefix.is_some() {
            annotate_heading_results(&mut response, heading_results_indexing_note());
        }
        Ok(CallToolResult::structured(response))
    }

    /// query+section: Qdrant grouped by `section_key` (#286), collapsed to
    /// each section's best-scoring chunk, returned as a path with no text — the
    /// `search` tool's `section` granularity. Requires
    /// `chunking.heading_metadata` (the flag that gates writing `section_key` at
    /// all — see `ingest.rs`) and, like `search_grouped`, rejects `explain`
    /// (no per-arm breakdown to report) and `fields` (no document-metadata join,
    /// same reasoning `search_chunks` gives for the identical rejection).
    async fn search_sections(&self, params: &SearchParams) -> Result<CallToolResult, McpError> {
        let query = params.query.as_deref().unwrap_or_default();
        let config = self.config();

        if !config.chunking.heading_metadata {
            return Err(McpError::invalid_params(
                HEADING_METADATA_OFF_SECTION.to_string(),
                None,
            ));
        }
        if params.explain == Some(true) {
            return Err(explain_rejection(
                Granularity::Section,
                &config.effective_granularities(),
            ));
        }
        if params.fields.is_some() {
            return Err(fields_rejection(
                Granularity::Section,
                &config.effective_granularities(),
            ));
        }

        validate_path_prefix(&params.path_prefix)?;
        let schemas = crate::schema::load_shared(&self.schema_cache);
        let conditions = build_query_conditions(params, &config, &schemas)?;

        let limit = resolve_limit(
            params.limit,
            config.search.default_limit,
            config.search.max_limit,
        );

        let filters = SearchFilters { conditions };

        let modified_after = params
            .modified_after
            .as_deref()
            .map(parse_date_to_timestamp)
            .transpose()
            .map_err(|e| McpError::invalid_params(e, None))?;
        let modified_before = params
            .modified_before
            .as_deref()
            .map(parse_date_to_timestamp)
            .transpose()
            .map_err(|e| McpError::invalid_params(e, None))?;

        let path_filter = self
            .resolve_path_filter(params.path_prefix.as_deref())
            .await?;
        let opts = SearchOptions {
            limit,
            min_score: params.min_score.or(config.search.min_score),
            hybrid: config.search.hybrid,
            rrf_candidates: config.search.rrf_candidates as u64,
            phrase: config.search.phrase && crate::status::INDEX_STATUS.phrase_matching_available(),
            explain: false,
            modified_after,
            modified_before,
            path_filter,
            rerank_candidate_limit: None,
            diversity_max_per_document: None,
        };

        let offset = params.offset.unwrap_or(0);
        let outcome = retrieval::search_sections(&self.deps(), query, &filters, &opts, offset)
            .await
            .map_err(|e| match e {
                retrieval::SearchError::Embed(err) => {
                    error!("Embedding query failed: {:#}", err);
                    McpError::internal_error("Failed to generate query embedding".to_string(), None)
                }
                retrieval::SearchError::Search(err) => {
                    error!("Qdrant grouped search failed: {:#}", err);
                    McpError::internal_error("Search query failed".to_string(), None)
                }
                retrieval::SearchError::Document(err) => {
                    error!("Document metadata lookup failed: {:#}", err);
                    McpError::internal_error("Document metadata lookup failed".to_string(), None)
                }
            })?;
        let path_prefix_truncated = outcome.path_prefix_truncated;
        let offset_truncated = outcome.offset_truncated;
        let sections = outcome.sections;

        let mut response =
            build_section_search_payload(&sections, path_prefix_truncated, offset_truncated);
        annotate_heading_results(&mut response, heading_results_indexing_note());
        Ok(CallToolResult::structured(response))
    }

    /// document, no query: the former `list_documents` tool's behavior, unchanged
    /// output shape (`{total, returned, offset, has_more, documents}`).
    async fn search_enumerate(&self, params: &SearchParams) -> Result<CallToolResult, McpError> {
        // Enumeration reads the state DB, which holds no heading data, so
        // `heading_prefix` cannot be honored here. Validate it the same way
        // query mode does first (so an empty list / the flag being off get
        // their own message), then reject rather than silently ignore it —
        // ignoring it would return every document as if the filter matched.
        if params.heading_prefix.is_some() {
            heading_prefix_condition(&params.heading_prefix, &self.config())?;
            return Err(McpError::invalid_params(
                "heading_prefix requires a query — a search without one lists whole \
                 documents, which cannot be filtered by heading"
                    .to_string(),
                None,
            ));
        }
        let query = build_document_query(params)?;

        let index = self.state_db().await.map_err(|e| {
            error!(
                "search (enumerate) could not open the metadata index: {:#}",
                e
            );
            McpError::internal_error(format!("Document index unavailable: {}", e), None)
        })?;

        let result = retrieval::list_documents(&DocumentIndexDeps { index }, &query)
            .await
            .map_err(|e| {
                error!("search (enumerate) failed: {:#}", e);
                McpError::internal_error(format!("Failed to list documents: {}", e), None)
            })?;

        // `total` always; `has_more` only when another page exists. A row
        // carries `mtime` only when the listing is ordered by it.
        let fields = params.fields.as_deref();
        let with_mtime = query.order_by == crate::state::OrderBy::Mtime;
        let mut response = serde_json::json!({
            "total": result.total,
            "returned": result.documents.len(),
            "documents": result
                .documents
                .iter()
                .map(|d| {
                    let mut row = document_row(d, fields);
                    if with_mtime {
                        row.insert("mtime".into(), serde_json::json!(d.mtime));
                    }
                    serde_json::Value::Object(row)
                })
                .collect::<Vec<_>>(),
        });
        if result.has_more(query.offset) {
            response["has_more"] = serde_json::json!(true);
        }
        Ok(CallToolResult::structured(response))
    }

    #[tool(annotations(read_only_hint = true))]
    async fn get_schema(
        &self,
        Parameters(params): Parameters<GetSchemaParams>,
    ) -> Result<CallToolResult, McpError> {
        let raw = params.path.clone().unwrap_or_default();
        let rel = normalize_scope_path(&raw)?;
        // Reads the shared, server-owned cache rather than walking the tree on every
        // call — this tool is the one the server's own instructions tell agents to
        // call before every write, so it needs to be cheap. See
        // `KbSearchServer::schema_cache`'s doc comment for how it stays current.
        let schemas = crate::schema::load_shared(&self.schema_cache);

        // A document path resolves via its parent; a directory resolves to itself. A
        // partial directory reference resolves like a partial document path does —
        // exact match wins, otherwise report the candidates rather than guessing.
        let rel = if raw.ends_with(".md") || rel.as_os_str().is_empty() {
            rel
        } else {
            resolve_scope_reference(&schemas, &rel)?
        };
        let lookup = if raw.ends_with(".md") {
            rel.clone()
        } else {
            rel.join("_")
        };
        let schema = schemas.resolve_for(&lookup);

        let values_only = params.values_only.unwrap_or(false);
        let values_in_use = params.values_in_use.unwrap_or(false);
        // (reported key, field path) of each open field `values_in_use` fills in.
        let mut open_fields: Vec<(String, String)> = Vec::new();
        let mut reported = serde_json::Map::new();
        let mut omitted = 0usize;
        for (field, def) in &schema.fields {
            // A derived field (`domain`) can be declared — a deployment lists it in
            // `indexed_fields` to filter on it — but ingest sets it from the folder and
            // a write that authors it is refused, so it is never offered as one to write.
            if crate::ingest::DERIVED_FIELDS.contains(&field.as_str()) {
                continue;
            }
            if let Some(wanted) = &params.fields
                && !wanted.contains(field)
            {
                continue;
            }
            if values_only && def.values.is_none() {
                continue;
            }
            // Schema files arrive via git sync, so field count is attacker-controlled
            // and the per-field value cap alone does not bound the response. Counted
            // after the filters so `omitted` reflects fields the caller actually asked
            // for, not every remaining field in the schema.
            if reported.len() >= MAX_REPORTED_FIELDS {
                omitted += 1;
                continue;
            }
            // Field names, permitted values, and provenance directories all originate
            // in schema files from a synced repo. The instructions actively steer
            // agents to call this tool before every write, so it is a reliably-triggered
            // reflection point — strip control characters and cap length on everything
            // that came from the knowledge base.
            let key = crate::server::sanitize_facet_value(field);
            // Free text, timestamps and object containers have no vocabulary
            // worth listing.
            if values_in_use
                && def.values.is_none()
                && !matches!(
                    def.ty,
                    Some(
                        crate::schema::FieldType::Text
                            | crate::schema::FieldType::Timestamp
                            | crate::schema::FieldType::Object
                    )
                )
            {
                open_fields.push((key.clone(), field.clone()));
            }
            reported.insert(
                key,
                field_def_json(def, schema.origin.get(field).map(String::as_str)),
            );
        }

        // `values_in_use`: what documents under this path actually use, so a
        // caller can list filter fields and values on demand. Best effort — an
        // unavailable metadata index just leaves these out.
        let mut other_fields_in_use: Vec<String> = Vec::new();
        if values_in_use && let Ok(index) = self.state_db().await {
            let dir = if raw.ends_with(".md") {
                rel.parent().map(Path::to_path_buf).unwrap_or_default()
            } else {
                rel.clone()
            };
            let dir = dir.to_string_lossy().trim_matches('/').to_string();
            let dir = (!dir.is_empty()).then_some(dir.as_str());
            for (key, field) in &open_fields {
                let Ok(top) = index.top_values(field, dir, MAX_VALUES_IN_USE as i64).await else {
                    continue;
                };
                if top.is_empty() {
                    continue;
                }
                let in_use: serde_json::Map<String, serde_json::Value> = top
                    .into_iter()
                    .map(|(value, n)| (crate::server::sanitize_facet_value(&value), n.into()))
                    .collect();
                if let Some(entry) = reported.get_mut(key).and_then(|v| v.as_object_mut()) {
                    entry.insert("in_use".into(), serde_json::Value::Object(in_use));
                }
            }
            // "Undeclared" means declared by no scope — the set `search`'s filter check
            // uses — not just absent from this one: a field a deeper scope declares
            // would otherwise be offered here as one nothing governs.
            let declared = schemas.declared_field_paths();
            let excluded = declared.len() + crate::ingest::DERIVED_FIELDS.len();
            if let Ok(fields) = index
                .fields_in_use(dir, (excluded + MAX_FILTER_OPTIONS_LISTED) as i64)
                .await
            {
                other_fields_in_use = fields
                    .into_iter()
                    .map(|(name, _)| name)
                    .filter(|name| {
                        !declared.contains(name)
                            && !crate::ingest::DERIVED_FIELDS.contains(&name.as_str())
                    })
                    .take(MAX_FILTER_OPTIONS_LISTED)
                    .map(|name| crate::server::sanitize_facet_value(&name))
                    .collect();
            }
        }

        let mut structured = serde_json::json!({
            "path": if rel.as_os_str().is_empty() {
                "/".to_string()
            } else {
                rel.to_string_lossy().into_owned()
            },
            "fields": reported,
        });
        if omitted > 0 {
            structured["omitted_fields"] = serde_json::json!(omitted);
        }
        if !other_fields_in_use.is_empty() {
            structured["other_fields_in_use"] = serde_json::json!(other_fields_in_use);
        }
        // The `dedup:` override cascade (#272), only when this scope sets a key —
        // unset keys fall back to the server's global `write.dedup_*`, which a
        // schema read has no business reporting.
        let mut dedup = serde_json::Map::new();
        if let Some(enabled) = schema.dedup_enabled {
            dedup.insert("enabled".into(), serde_json::json!(enabled));
        }
        if let Some(threshold) = schema.dedup_threshold {
            // `json!(f32)` widens to f64 and would emit 0.949999988079071 for a
            // file that says 0.95; round-trip through the shortest f32 text instead.
            let shown = threshold
                .to_string()
                .parse::<f64>()
                .map_or(serde_json::json!(threshold), |v| serde_json::json!(v));
            dedup.insert("threshold".into(), shown);
        }
        if !dedup.is_empty() {
            structured["dedup"] = serde_json::Value::Object(dedup);
        }

        Ok(CallToolResult::structured(structured))
    }

    #[tool(annotations(destructive_hint = true, idempotent_hint = false))]
    async fn update_schema(
        &self,
        Parameters(params): Parameters<UpdateSchemaParams>,
    ) -> Result<CallToolResult, McpError> {
        let invalid = |msg: String| McpError::invalid_params(msg, None);

        let requested = normalize_scope_path(params.path.as_deref().unwrap_or(""))?;
        let edit = build_schema_edit(&params)?;

        // Read the shared cache for the pre-edit view (casualty check, raw file
        // lookup below). This does NOT need to be maximally fresh — worst case a
        // concurrent change this call doesn't see yet is caught by the casualty
        // check failing safe or by a subsequent call — but see the synchronous
        // rebuild after the write below, which is NOT optional.
        let schemas = crate::schema::load_shared(&self.schema_cache);

        // Resolve a partial reference against existing scopes, but fall back to the
        // literal path: creating a schema for a directory that has none yet is the
        // normal way to introduce one, so "no match" is not an error here.
        let matches = schemas.match_scope_dirs(&requested);
        let rel_dir = match matches.len() {
            0 => requested,
            1 => matches.into_iter().next().expect("length checked"),
            _ => {
                return Err(McpError::invalid_params(
                    format!(
                        "'{}' matches {} scopes: {}. Use a more specific path.",
                        requested.display(),
                        matches.len(),
                        matches
                            .iter()
                            .map(|p| p.display().to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    None,
                ));
            }
        };

        // Root-scope mutations need an explicit opt-in (see `acknowledge_root_change`'s
        // doc comment). `remove_values` is exempt — shrinking the root vocabulary is the
        // policy-aligned direction, and the casualty check below already guards it — and
        // `dry_run` is exempt everywhere, since it writes nothing. This must run before
        // `file.apply()` below: the point is to refuse before any edit is computed or
        // written, not merely before the commit.
        let is_root = rel_dir.as_os_str().is_empty();
        let is_gated_op = !matches!(edit, crate::schema::SchemaEdit::RemoveValues { .. });
        let dry_run_requested = params.dry_run.unwrap_or(false);
        let acknowledged = params.acknowledge_root_change.unwrap_or(false);
        if is_root && is_gated_op && !dry_run_requested && !acknowledged {
            return Err(McpError::invalid_params(
                "root schema changes are guarded: the root tag vocabulary is \
                 identity-only by policy (see meta/schema-tag-policy.md in this \
                 knowledge base). Pass acknowledge_root_change=true only if this change \
                 is a deliberate design decision consistent with that policy."
                    .to_string(),
                None,
            ));
        }

        // What this scope's ancestors alone resolve the field to, so `add_values` in a
        // child scope extends the inherited set instead of replacing it. Read from the
        // ancestors' files on disk, the same way the edit itself reads its own file:
        // under the lock below for the real edit, so a concurrent change to a parent
        // scope cannot leave the edit extending a stale inherited set. The root has no
        // ancestors: a root schema file replaces the config-derived root outright.
        let scope = crate::schema::scope_label(&rel_dir);
        let inherited_field = |schemas: &SchemaCache| -> Option<crate::schema::FieldDef> {
            if rel_dir.as_os_str().is_empty() {
                return None;
            }
            match schemas.resolve_from_disk(&rel_dir, None) {
                Ok(resolved) => resolved.fields.get(edit.field()).cloned(),
                // An ancestor's file no longer parses (it arrived through git): the
                // server still enforces its last valid rules, so extend those rather
                // than nothing, which would narrow the inherited set.
                Err(_) => {
                    let parent = rel_dir.parent().unwrap_or(std::path::Path::new(""));
                    schemas
                        .resolve_for(&parent.join("_"))
                        .fields
                        .get(edit.field())
                        .cloned()
                }
            }
        };

        // Whether the edit applies to the file as it is before the lock — only to
        // tell an edit that a concurrent change made inapplicable from one that never
        // applied, should it fail under the lock.
        let applied_before_lock = schemas.raw_file_at(&rel_dir).is_ok_and(|(mut file, _)| {
            file.apply_inheriting(&edit, inherited_field(&schemas).as_ref())
                .is_ok()
        });

        let dry_run = params.dry_run.unwrap_or(false);
        let force = params.force.unwrap_or(false);

        let config = self.config();
        let token = crate::secrets::git_token(&config).map_err(credential_error)?;
        let deps = self.write_deps(&config, token.as_deref(), None);

        // The whole read / edit / install runs under one `GIT_LOCK` acquisition that
        // starts by syncing with the remote, so the edit applies to the schema as it
        // is now — two concurrent edits both land, neither overwriting the other.
        let (git_lock, mut head) = write::lock_and_sync(&deps).await;
        let mut attempt = 0;
        let (summary, casualties) = loop {
            attempt += 1;

            // Read from disk, not the shared cache: the edit must apply to the file as
            // it is now. A file that no longer parses got there outside this tool (a
            // push to the knowledge base's git host); a running server refuses it and
            // keeps serving the last valid schema, and only a fix to the file itself
            // clears that.
            let (mut file, on_disk_name) = schemas.raw_file_at(&rel_dir).map_err(|e| {
                let e = crate::schema::model_facing_reason(&e);
                invalid(format!(
                    "The schema for '{scope}' on disk is invalid: {e}. The server is still \
                     enforcing the last valid schema (and would refuse to start on this one). \
                     The operator must repair it; update_schema can only edit a schema that \
                     parses."
                ))
            })?;
            let inherited = inherited_field(&schemas);
            let summary = match file.apply_inheriting(&edit, inherited.as_ref()) {
                Ok(summary) => summary,
                Err(_) if applied_before_lock || attempt > 1 => {
                    return Err(invalid(format!(
                        "The schema for '{scope}' was edited by someone else; re-read it with \
                         get_schema and try again."
                    )));
                }
                Err(e) => return Err(invalid(e)),
            };

            // A self-contradictory definition parses fine but would be refused by the
            // next schema rebuild (and stop the server from starting) — after this call
            // has already reported success. Catch it here, where the caller can act.
            file.validate_self().map_err(invalid)?;

            let yaml = file.to_yaml().map_err(invalid)?;

            // The same size cap `SchemaCache::build` enforces before parsing: a file
            // over it would be refused at the next rebuild, so never write one.
            if yaml.len() as u64 > crate::schema::MAX_SCHEMA_FILE_BYTES {
                return Err(invalid(format!(
                    "Refusing to write a schema for '{scope}' of {} bytes: the limit is {} \
                     bytes. Split the rules across subdirectory schemas instead.",
                    yaml.len(),
                    crate::schema::MAX_SCHEMA_FILE_BYTES
                )));
            }

            // Re-parse what we are about to write. A schema that does not round-trip
            // would be refused by the next schema rebuild.
            serde_yaml_ng::from_str::<crate::schema::SchemaFile>(&yaml).map_err(|e| {
                McpError::internal_error(
                    format!("Refusing to write a schema that does not parse: {e}"),
                    None,
                )
            })?;

            // Dry-run the change against documents that already exist under this scope.
            let casualties = self.documents_broken_by(&rel_dir, &schemas, &file).await?;

            if !casualties.is_empty() && !force && !dry_run {
                let (would_invalidate, casualties_total, casualties_truncated) =
                    capped_casualties(&casualties);
                return Err(McpError::invalid_params(
                    format!(
                        "Refusing to apply: {} existing document(s) would fail the new rules. \
                         Fix them first, or pass force to apply anyway.\n{}",
                        casualties.len(),
                        render_casualties(&casualties)
                    ),
                    Some(serde_json::json!({
                        "would_invalidate": would_invalidate,
                        "casualties_total": casualties_total,
                        "casualties_truncated": casualties_truncated,
                    })),
                ));
            }

            if dry_run {
                // The edited field as the scope would resolve it, not the file: the
                // file format is the server's business, and `get_schema` serves the
                // rest of the scope.
                let resolved = schemas.resolve_from_disk(&rel_dir, Some(&file)).ok();
                let field = edit.field();
                let definition = resolved.as_ref().and_then(|r| {
                    r.fields
                        .get(field)
                        .map(|def| field_def_json(def, r.origin.get(field).map(String::as_str)))
                });
                let (would_invalidate, casualties_total, casualties_truncated) =
                    capped_casualties(&casualties);
                let mut response = serde_json::json!({
                    "dry_run": true,
                    "path": scope,
                    "summary": summary,
                    "field": field,
                });
                if let Some(definition) = definition {
                    response["definition"] = definition;
                }
                insert_casualties(
                    &mut response,
                    "would_invalidate",
                    would_invalidate,
                    casualties_total,
                    casualties_truncated,
                );
                return Ok(CallToolResult::structured(response));
            }

            // Always the canonical name. A directory still on the legacy name is
            // migrated by this write: the legacy file is removed in the same commit.
            let rel_file = rel_dir.join(crate::schema::SCHEMA_FILE_NAME);
            let rel_file_str = rel_file.to_string_lossy().to_string();
            let replaced = on_disk_name
                .filter(|name| *name != crate::schema::SCHEMA_FILE_NAME)
                .map(|name| rel_dir.join(name).to_string_lossy().to_string());
            let commit_message = format!("schema: {summary} in {scope}");

            // `write_raw_file` rolls itself back on a pre-commit failure and returns
            // `Err` (see its doc comment) — this `?` propagates that WITHOUT reaching
            // the cache rebuild below: a rolled-back write leaves the schema on disk
            // unchanged. A conflict with the remote drops this write's commit, so
            // nothing is left written; the clone syncs again and the edit is applied
            // once more to the fresh file.
            match self
                .write_raw_file(
                    &git_lock,
                    &rel_file_str,
                    &yaml,
                    &commit_message,
                    replaced.as_deref(),
                )
                .await?
            {
                RawWrite::Committed => break (summary, casualties),
                // Every conflict syncs, the last one included: the reset that dropped
                // this commit went back to `head`, so this sync is what brings the
                // clone onto whatever reached the remote since and marks those paths
                // dirty — the webhook those commits trigger would then find the clone
                // already up to date and mark nothing.
                RawWrite::Conflict => {
                    head = write::sync_clone(&deps, &git_lock, head).await;
                    if attempt >= write::MAX_WRITE_ATTEMPTS {
                        return Err(invalid(format!(
                            "The schema for '{scope}' was edited by someone else; re-read it \
                             with get_schema and try again."
                        )));
                    }
                }
            }
        };
        drop(git_lock);

        // Rebuild the shared schema cache and swap it in SYNCHRONOUSLY — before this
        // call returns, not merely "soon". `write_raw_file` already called
        // `reindex::mark_full`, so the reindex worker will ALSO rebuild it, but that
        // happens out of band on the worker's own schedule and cannot be relied on to
        // win the race against whatever the calling agent does next. The scenario this
        // guards against: an agent calls `update_schema` to permit a new value, then
        // immediately calls `write_document` relying on that new rule
        // — if the write path read a stale cache, it would validate against the schema
        // this very call just replaced and wrongly reject a now-valid document. Wrapped
        // in `spawn_blocking` because the walk itself is blocking filesystem work, not
        // because anything here needs to run off-thread for its own sake — `.await`ing
        // it still makes this call return only once the rebuild has completed.
        //
        // This runs whether or not the commit has synced yet: an unsynced write is
        // still a real local commit — the new schema is genuinely in effect for this
        // clone regardless of whether the push to the remote landed — so the cache
        // must reflect it just the same.
        //
        // The file this call wrote was validated above, so a refused rebuild means a
        // DIFFERENT schema file in the tree is invalid (it arrived through git).
        // `apply_rebuild` keeps the last good cache, logs and records it; the call
        // still succeeds — the write is committed — but says loudly that the change
        // is not in effect yet.
        let rebuild_data_path = self.canonical_data_path.clone();
        let rebuild_frontmatter = self.config().frontmatter.clone();
        let rebuild_indexing = self.config().indexing.clone();
        let mut refused_rebuild: Option<crate::schema::SchemaBuildError> = None;
        match tokio::task::spawn_blocking(move || {
            SchemaCache::build(&rebuild_data_path, &rebuild_frontmatter, &rebuild_indexing)
        })
        .await
        {
            Ok(built) => {
                let refusal = built.as_ref().err().cloned();
                if !crate::schema::apply_rebuild(
                    &self.schema_cache,
                    built,
                    &crate::status::INDEX_STATUS,
                    "update_schema",
                ) {
                    refused_rebuild = refusal;
                }
            }
            Err(e) => {
                // A panic in the walk itself. Leave the previous cache in place rather
                // than fail the whole call: the write already succeeded and is
                // committed: an agent's NEXT read/write may briefly see the pre-edit
                // schema, which is the same staleness window this call exists to
                // close, not a new one — failing here would not close it either, just
                // add a spurious error on top of a successful write.
                error!("Schema rebuild panicked after update_schema write: {e}");
            }
        }

        let (invalidated, casualties_total, casualties_truncated) = capped_casualties(&casualties);
        let mut response = serde_json::json!({
            "path": scope,
            "summary": summary,
        });
        insert_casualties(
            &mut response,
            "invalidated",
            invalidated,
            casualties_total,
            casualties_truncated,
        );
        if let Some(refusal) = &refused_rebuild {
            response["warning"] = serde_json::json!(format!(
                "Saved but NOT in effect yet: another directory's schema is invalid, so \
                 the previous rules stay enforced until it is fixed. {}",
                refusal.model_facing()
            ));
        }
        Ok(CallToolResult::structured(response))
    }

    #[tool(annotations(read_only_hint = true))]
    async fn get_document(
        &self,
        Parameters(params): Parameters<GetDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        let raw = params.path.trim();
        if raw.is_empty() {
            return Err(McpError::invalid_params(
                "Path parameter is empty".to_string(),
                None,
            ));
        }
        if raw.len() > MAX_PATH_LEN {
            return Err(McpError::invalid_params(
                format!("path exceeds maximum length of {MAX_PATH_LEN} characters"),
                None,
            ));
        }

        // Checked before the path is resolved: a malformed combination of
        // parameters is wrong no matter which document it was aimed at, so it
        // should not cost a metadata-index open and a file read to say so
        // (#286 — mode resolution itself lives in retrieval.rs so this
        // tool and `/api/doc` can't drift apart on it).
        let request = retrieval::parse_document_view_request(
            params.start_line,
            params.end_line,
            params.line,
            params.heading_path,
            params.levels_up,
            params.outline.unwrap_or(false),
        )
        .map_err(|e| McpError::invalid_params(e, None))?;

        debug!(path = %raw, "get_document called");

        // The fuzzy-basename fallback resolves against the SQLite metadata index
        // rather than a Qdrant facet fetch, so the index has to be opened here.
        // Only the fallback needs it — an exact path hit is served from disk and
        // never touches this — but the resolve happens inside `get_document`, so
        // it is passed unconditionally.
        let index = self.state_db().await.map_err(|e| {
            error!("get_document could not open the metadata index: {:#}", e);
            McpError::internal_error(format!("Document index unavailable: {}", e), None)
        })?;

        match retrieval::get_document(&self.deps(), index, raw).await {
            Ok(doc) => {
                debug!(path = %raw, "get_document served");
                // The caller round-trips this into write_document/delete_document as
                // `expected_version`. Computed before any slicing and always over the
                // whole file: it names the document on disk, so a slice's version
                // could never match.
                let version = write::document_version(doc.content.as_bytes());
                let rel_path = retrieval::relative_to_data(
                    &doc.path.to_string_lossy(),
                    &self.canonical_data_path,
                );
                let section_max_bytes = self.config().search.section_max_bytes;
                let view =
                    retrieval::resolve_document_view(&doc.content, &request, section_max_bytes)
                        .map_err(document_view_error_to_mcp)?;

                // The view-specific fields come from `retrieval::document_view_json`,
                // shared with `/api/doc`; only the envelope is added here.
                let mut structured = retrieval::document_view_json(&view);
                structured.insert("path".to_string(), serde_json::json!(rel_path));
                structured.insert("version".to_string(), serde_json::json!(version));
                // The link graph rides on a whole-document read (the one a caller
                // makes to understand a document) unless asked for or declined
                // explicitly: on a targeted read it would outweigh the text itself.
                // An oversized document read whole is still that read when it comes
                // back cut short (`truncated`, #290), as the outline an oversized
                // document with headings degrades to is; only a range the caller
                // named is targeted.
                let whole_read = match &view {
                    retrieval::DocumentView::Range(slice) => !slice.partial() || slice.truncated,
                    retrieval::DocumentView::Outline(outline) => outline.document_oversized,
                    retrieval::DocumentView::Section(_) => false,
                };
                if params.links.unwrap_or(whole_read) {
                    insert_link_lists(&mut structured, &doc.links_out, &doc.links_in);
                }
                // Opt-in (#257): a caller that did not ask pays no subprocess.
                if let Some(limit) = params.history {
                    let data_path = self.canonical_data_path.to_string_lossy().into_owned();
                    let history = retrieval::document_changes_json(&data_path, &rel_path, limit)
                        .await
                        .map_err(|e| {
                            error!("get_document history failed for '{rel_path}': {e:#}");
                            McpError::internal_error(
                                "this document's change history is unavailable".to_string(),
                                None,
                            )
                        })?;
                    structured.insert("history".to_string(), history);
                }
                Ok(CallToolResult::structured(serde_json::Value::Object(
                    structured,
                )))
            }
            Err(GetDocumentError::Outside) => {
                warn!(path = %raw, "get_document: path outside data directory");
                Err(McpError::invalid_params(
                    "File path is outside the data directory".to_string(),
                    None,
                ))
            }
            Err(GetDocumentError::NotPermitted) => {
                warn!(path = %raw, "get_document: file type not permitted");
                Err(McpError::invalid_params(
                    "File type not permitted".to_string(),
                    None,
                ))
            }
            Err(GetDocumentError::NotFound { suggestions }) => {
                let mut msg = format!("File not found: '{}'.", raw);
                if suggestions.is_empty() {
                    msg.push_str(" Use the `search` tool to find paths.");
                } else {
                    msg.push_str(&format!(
                        " Closest indexed files: {}. Or use the `search` tool.",
                        suggestions.join(", ")
                    ));
                }
                Err(McpError::invalid_params(msg, None))
            }
            Err(GetDocumentError::Ambiguous { matches }) => {
                let basename = std::path::Path::new(raw)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(raw);
                Err(McpError::invalid_params(
                    format!(
                        "Multiple files match basename '{}': {}. Use a more specific path.",
                        basename,
                        matches.join(", ")
                    ),
                    None,
                ))
            }
            Err(GetDocumentError::Io(msg)) => Err(McpError::invalid_params(msg, None)),
        }
    }

    /// The `WriteDeps` every write tool hands the shared pipeline, from this
    /// server's fields and one config snapshot.
    fn write_deps<'a>(
        &'a self,
        config: &'a ResolvedConfig,
        token: Option<&'a str>,
        state: Option<&'a crate::state::StateDb>,
    ) -> WriteDeps<'a, EmbedClient, QdrantStore> {
        WriteDeps {
            retrieval: self.deps(),
            canonical_data_path: &self.canonical_data_path,
            schema_cache: &self.schema_cache,
            validation: &config.validation,
            indexing: &config.indexing,
            prepend_description: config.chunking.prepend_description,
            dedup_enabled: config.write.dedup_enabled,
            dedup_threshold: config.write.dedup_threshold,
            git_url: config.source.git_url.as_deref(),
            branch: &config.source.branch,
            token,
            commit_author_name: &config.write.commit_author_name,
            commit_author_email: &config.write.commit_author_email,
            queue: &self.reindex_queue,
            state,
        }
    }

    /// Shared pipeline for `write_document`'s create, edit and single-document
    /// move paths: a thin adapter over `write::write_document` that maps the
    /// structured `WriteSuccess`/`WriteError` back onto this tool surface's
    /// `CallToolResult`/`McpError` shapes (see `create_edit_success_to_result` /
    /// `create_edit_error_to_mcp_error`).
    ///
    /// `change` describes the edit; the new content is computed by the pipeline
    /// from the document as it is under the lock — see `write::DocChange`.
    /// `expected_version` is the caller's `version` from a prior read: required
    /// for a full replace or a move of an existing document.
    #[allow(clippy::too_many_arguments)]
    async fn run_document_write(
        &self,
        change: write::DocChange<'_>,
        rel_path: &str,
        message: Option<&str>,
        default_verb: &str,
        force_new: Option<bool>,
        operation: &str,
        dest_path: Option<&str>,
        expected_version: Option<&str>,
    ) -> Result<CallToolResult, McpError> {
        // One snapshot for the whole call, so a concurrent `POST /admin/reload`
        // cannot mix old and new values across this method's several config reads.
        let config = self.config();

        // Content sent whole is capped here; a relative edit's result is capped by
        // the pipeline that computes it (`write::apply_change`).
        if let write::DocChange::Create(content) | write::DocChange::Replace(content) = change
            && content.len() > MAX_CONTENT_LEN
        {
            return Err(McpError::invalid_params(
                format!(
                    "content is too large ({} bytes); maximum is {} bytes",
                    content.len(),
                    MAX_CONTENT_LEN
                ),
                None,
            ));
        }

        let token = crate::secrets::git_token(&config).map_err(credential_error)?;

        // Only opened for a MOVE — nothing else reads `deps.state` — so a plain
        // create/edit never materializes `state.db`. Best-effort: a state DB that
        // fails to open degrades a move to "without link rewriting" (see
        // `WriteDeps::state`'s doc comment) rather than failing the write.
        let state_db = if dest_path.is_some() {
            self.state_db().await.ok()
        } else {
            None
        };
        let deps = self.write_deps(&config, token.as_deref(), state_db);

        let is_create = change.is_create();
        let req = WriteRequest {
            rel_path,
            change,
            message,
            default_verb,
            force_new,
            operation,
            expected_version,
            dest_path,
        };

        match crate::write::write_document(&deps, req).await {
            Ok(success) => Ok(create_edit_success_to_result(
                success, rel_path, is_create, dest_path,
            )),
            Err(err) => Err(create_edit_error_to_mcp_error(
                err,
                rel_path,
                is_create,
                &self.canonical_data_path,
                dest_path,
            )),
        }
    }

    /// `write_document`'s directory-move dispatch: `source_dir` resolved to an
    /// existing directory, so this relocates the whole subtree via
    /// `write::move_directory` instead of writing a single document. Mirrors the
    /// old standalone `move_directory` tool.
    ///
    /// Every field that only makes sense for a single document (`content`,
    /// `old_string`/`new_string`, `expected_version`, `force_new`) is rejected
    /// here — a directory move has no body to replace and no dedup gate to
    /// bypass; it re-checks every document under the lock itself instead.
    /// `new_path` is required: it is the destination prefix.
    async fn write_document_move_dir(
        &self,
        params: &WriteDocumentParams,
        source_dir: &str,
    ) -> Result<CallToolResult, McpError> {
        let dest_dir = match params.new_path.as_deref().map(str::trim) {
            Some(p) if !p.is_empty() => p,
            _ => {
                return Err(McpError::invalid_params(
                    "new_path is required to move the directory at path".to_string(),
                    None,
                ));
            }
        };

        for (field, is_set) in [
            ("content", params.content.is_some()),
            ("old_string", params.old_string.is_some()),
            ("new_string", params.new_string.is_some()),
            ("frontmatter_patch", params.frontmatter_patch.is_some()),
            ("append", params.append.is_some()),
            ("expected_version", params.expected_version.is_some()),
            ("force_new", params.force_new.is_some()),
        ] {
            if is_set {
                return Err(McpError::invalid_params(
                    format!(
                        "{} is not valid when path is a directory: a directory move \
                         only takes new_path (and optionally message)",
                        field
                    ),
                    None,
                ));
            }
        }

        let config = self.config();
        let token = crate::secrets::git_token(&config).map_err(credential_error)?;

        // Best-effort, same as `run_document_write`'s own lazy state-DB open for a
        // single-document MOVE — see `WriteDeps::state`'s doc comment.
        let state_db = self.state_db().await.ok();
        let deps = self.write_deps(&config, token.as_deref(), state_db);

        match crate::write::move_directory(&deps, source_dir, dest_dir, params.message.as_deref())
            .await
        {
            Ok(success) => Ok(move_directory_success_to_result(
                success, source_dir, dest_dir,
            )),
            Err(err) => Err(move_directory_error_to_mcp_error(err, source_dir, dest_dir)),
        }
    }

    /// `write_document`'s create path: `path` did not resolve to an existing,
    /// permitted file. Mirrors the old standalone `create_document` tool.
    async fn write_document_create(
        &self,
        params: WriteDocumentParams,
        raw: &str,
    ) -> Result<CallToolResult, McpError> {
        if params.old_string.is_some() || params.new_string.is_some() {
            return Err(McpError::invalid_params(
                "cannot surgically edit a document that does not exist".to_string(),
                None,
            ));
        }
        if params.frontmatter_patch.is_some() {
            return Err(McpError::invalid_params(
                "cannot patch frontmatter on a document that does not exist — use content to \
                 create it"
                    .to_string(),
                None,
            ));
        }
        if params.append.is_some() {
            return Err(McpError::invalid_params(
                "cannot append to a document that does not exist — use content to create it"
                    .to_string(),
                None,
            ));
        }
        if params.new_path.is_some() {
            return Err(McpError::invalid_params(
                "new_path is not valid when creating a new document — there is \
                 nothing at path yet to move; create it directly at the final path"
                    .to_string(),
                None,
            ));
        }
        let Some(content) = params.content.as_deref() else {
            return Err(McpError::invalid_params(
                "content is required to create a new document".to_string(),
                None,
            ));
        };

        // Resolve path: must be relative, no traversal, not already existing.
        let data_root = self.canonical_data_path.clone();
        let abs_path = crate::write::resolve_safe_write_path(&data_root, raw)
            .map_err(|e| McpError::invalid_params(e, None))?;

        // The include-pattern guard runs inside `write::write_document` too; it
        // runs here as well, ahead of the `exists()` pre-check below, so a path
        // that both exists and fails it gets the include-pattern message rather
        // than a misleading "already exists".
        crate::write::check_include_pattern_against(&self.include_patterns, raw).map_err(|e| {
            create_edit_error_to_mcp_error(e, raw, true, &self.canonical_data_path, None)
        })?;

        // File must not already exist for create.
        if abs_path.exists() {
            return Err(McpError::invalid_params(
                format!(
                    "File '{}' already exists. Call write_document again without \
                     old_string/new_string to edit it in place.",
                    raw
                ),
                None,
            ));
        }

        self.run_document_write(
            write::DocChange::Create(content),
            raw,
            params.message.as_deref(),
            "add",
            params.force_new,
            "write_document",
            None, // a create never moves a document
            params.expected_version.as_deref(),
        )
        .await
    }

    /// `write_document`'s edit path: `path` resolved to an existing, permitted
    /// file. Mirrors the old standalone `edit_document` tool. Nothing is read
    /// here: the pipeline applies the edit to the document as it is under the
    /// lock.
    async fn write_document_edit(
        &self,
        params: WriteDocumentParams,
        canonical: PathBuf,
    ) -> Result<CallToolResult, McpError> {
        // Parse and validate the content-edit mode (surgical vs full-replace vs
        // neither). `new_path` is an orthogonal axis handled below, independent of
        // this — see `parse_edit_mode`'s doc comment.
        let mode = parse_edit_mode(&params).map_err(|e| McpError::invalid_params(e, None))?;
        let dest_path = params.new_path.as_deref();

        // Derive the repo-relative path from the canonical absolute path.
        let rel_path = canonical
            .strip_prefix(&self.canonical_data_path)
            .unwrap_or(&canonical)
            .to_string_lossy()
            .into_owned();

        let operation = edit_operation_label(mode.as_ref(), dest_path.is_some());
        let owned = mode.map(|m| OwnedChange::from_mode(m, rel_path.clone()));

        self.run_document_write(
            owned
                .as_ref()
                .map_or(write::DocChange::Keep, OwnedChange::as_change),
            &rel_path,
            params.message.as_deref(),
            "update",
            None, // no dedup gate for edit
            operation,
            dest_path,
            params.expected_version.as_deref(),
        )
        .await
    }

    /// `write_document`'s batch dispatch (#180): `documents` was supplied.
    ///
    /// Rejects every single-document field (`path`, `content`,
    /// `old_string`/`new_string`, `frontmatter_patch`, `append`, `new_path`,
    /// `expected_version`, `force_new`) if set alongside `documents` at the top
    /// level — a batch entry carries its own copies of the content-edit fields
    /// inside `documents` instead.
    ///
    /// Resolves each entry's create-vs-edit status and parses its edit mode up
    /// front (reusing `parse_edit_mode` via `batch_input_as_write_params`, so a
    /// batch entry's content-edit rules are byte-for-byte the same as a
    /// single-document call's), collecting EVERY entry's problem rather than
    /// stopping at the first one. `write::write_documents_batch` then owns the
    /// atomic, single-commit part, including applying each change to the
    /// document as it is under the lock.
    async fn write_document_batch(
        &self,
        params: WriteDocumentParams,
    ) -> Result<CallToolResult, McpError> {
        for (field, is_set) in [
            ("path", params.path.is_some()),
            ("content", params.content.is_some()),
            ("old_string", params.old_string.is_some()),
            ("new_string", params.new_string.is_some()),
            ("frontmatter_patch", params.frontmatter_patch.is_some()),
            ("append", params.append.is_some()),
            ("new_path", params.new_path.is_some()),
            ("expected_version", params.expected_version.is_some()),
            ("force_new", params.force_new.is_some()),
        ] {
            if is_set {
                return Err(McpError::invalid_params(
                    format!(
                        "{field} is not valid alongside documents: a batch write supplies \
                         ONLY documents (and, optionally, one message describing the whole \
                         batch) — each document's own path/content/edit fields \
                         belong inside its entry in documents"
                    ),
                    None,
                ));
            }
        }

        let documents = params
            .documents
            .expect("write_document_batch is only called when params.documents.is_some()");

        if documents.is_empty() {
            return Err(McpError::invalid_params(
                "documents must not be empty".to_string(),
                None,
            ));
        }
        if documents.len() > write::MAX_BATCH_DOCUMENTS {
            return Err(McpError::invalid_params(
                format!(
                    "documents has {} entries; maximum is {}",
                    documents.len(),
                    write::MAX_BATCH_DOCUMENTS
                ),
                None,
            ));
        }

        let mut parse_errors: Vec<String> = Vec::new();
        // (rel_path, change, index into `documents`)
        let mut entries: Vec<(String, OwnedChange, usize)> = Vec::with_capacity(documents.len());

        for (index, entry) in documents.iter().enumerate() {
            let raw = entry.path.trim().to_string();
            if raw.is_empty() {
                parse_errors.push("documents: an entry's path is empty".to_string());
                continue;
            }
            if raw.len() > MAX_PATH_LEN {
                parse_errors.push(format!(
                    "documents: '{raw}' exceeds the maximum path length of {MAX_PATH_LEN} \
                     characters"
                ));
                continue;
            }
            if let Some(content) = entry.content.as_deref()
                && content.len() > MAX_CONTENT_LEN
            {
                parse_errors.push(format!(
                    "documents: '{}' content is too large ({} bytes); maximum is {} bytes",
                    raw,
                    content.len(),
                    MAX_CONTENT_LEN
                ));
                continue;
            }

            match retrieval::resolve_within_data(
                &raw,
                &self.canonical_data_path,
                &self.include_patterns,
            ) {
                Ok(canonical) => {
                    let fake_params = batch_input_as_write_params(entry);
                    let mode = match parse_edit_mode(&fake_params) {
                        Ok(Some(m)) => m,
                        Ok(None) => {
                            // `parse_edit_mode` only returns `None` for a pure move,
                            // and a batch entry never has `new_path` — failed soft
                            // rather than `unreachable!()`.
                            parse_errors.push(format!(
                                "documents: '{raw}' provides no edit mode (content, \
                                 old_string+new_string, frontmatter_patch, or append)"
                            ));
                            continue;
                        }
                        Err(e) => {
                            parse_errors.push(format!("documents: '{raw}': {e}"));
                            continue;
                        }
                    };
                    let rel_path = canonical
                        .strip_prefix(&self.canonical_data_path)
                        .unwrap_or(&canonical)
                        .to_string_lossy()
                        .into_owned();
                    let change = OwnedChange::from_mode(mode, rel_path.clone());
                    entries.push((rel_path, change, index));
                }
                Err(retrieval::ResolveErr::NotFound) | Err(retrieval::ResolveErr::NotPermitted) => {
                    // Create path — mirrors `write_document_create`'s own field
                    // guards: none of the edit-only fields make sense against a
                    // document that does not exist yet.
                    if entry.old_string.is_some() || entry.new_string.is_some() {
                        parse_errors.push(format!(
                            "documents: '{raw}' cannot be surgically edited — it does not \
                             exist yet"
                        ));
                        continue;
                    }
                    if entry.frontmatter_patch.is_some() {
                        parse_errors.push(format!(
                            "documents: '{raw}' cannot be frontmatter-patched — it does not \
                             exist yet; use content to create it"
                        ));
                        continue;
                    }
                    if entry.append.is_some() {
                        parse_errors.push(format!(
                            "documents: '{raw}' cannot be appended to — it does not exist \
                             yet; use content to create it"
                        ));
                        continue;
                    }
                    let Some(content) = entry.content.clone() else {
                        parse_errors.push(format!(
                            "documents: '{raw}' does not exist yet — content is required to \
                             create it"
                        ));
                        continue;
                    };
                    if let Err(e) =
                        write::check_include_pattern_against(&self.include_patterns, &raw)
                    {
                        parse_errors.push(format!(
                            "documents: '{raw}': {}",
                            batch_write_document_error_text(&e)
                        ));
                        continue;
                    }
                    entries.push((raw, OwnedChange::Create(content), index));
                }
                Err(retrieval::ResolveErr::Outside) => {
                    parse_errors.push(format!("documents: '{raw}' is outside the data directory"));
                }
                Err(retrieval::ResolveErr::Other(msg)) => {
                    parse_errors.push(format!("documents: '{raw}': {msg}"));
                }
            }
        }

        if !parse_errors.is_empty() {
            return Err(McpError::invalid_params(parse_errors.join("; "), None));
        }

        let config = self.config();
        let token = crate::secrets::git_token(&config).map_err(credential_error)?;
        // Batch writes never move or delete a document, so nothing in
        // `write::write_documents_batch` reads `WriteDeps::state`.
        let deps = self.write_deps(&config, token.as_deref(), None);

        let requests: Vec<write::BatchWriteRequest<'_>> = entries
            .iter()
            .map(|(rel_path, change, index)| write::BatchWriteRequest {
                rel_path,
                change: change.as_change(),
                force_new: documents[*index].force_new,
                expected_version: documents[*index].expected_version.as_deref(),
            })
            .collect();

        match write::write_documents_batch(&deps, &requests, params.message.as_deref()).await {
            Ok(success) => Ok(batch_write_success_to_result(success)),
            Err(err) => Err(batch_write_error_to_mcp_error(err)),
        }
    }

    #[tool(annotations(destructive_hint = true, idempotent_hint = false))]
    async fn write_document(
        &self,
        Parameters(params): Parameters<WriteDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        // Branch 0: a batch write. Checked first and exclusively — every
        // single-document field is rejected if set alongside `documents`
        // (enforced inside `write_document_batch`), so there is no ambiguity
        // about which of the two shapes a call with both would mean.
        if params.documents.is_some() {
            return self.write_document_batch(params).await;
        }

        let raw = params.path.as_deref().unwrap_or("").trim().to_string();
        if raw.is_empty() {
            return Err(McpError::invalid_params(
                "path parameter is empty (or, for a batch write, documents is required)"
                    .to_string(),
                None,
            ));
        }
        // Mirrors `get_document`'s length guard (see its comment): rejecting an
        // oversized path here avoids paying for `resolve_safe_write_path`, an
        // include-pattern check, and git staging on an input that was never going
        // to resolve to anything.
        if raw.len() > MAX_PATH_LEN {
            return Err(McpError::invalid_params(
                format!("path exceeds maximum length of {MAX_PATH_LEN} characters"),
                None,
            ));
        }
        if let Some(new_path) = params.new_path.as_deref()
            && new_path.len() > MAX_PATH_LEN
        {
            return Err(McpError::invalid_params(
                format!("new_path exceeds maximum length of {MAX_PATH_LEN} characters"),
                None,
            ));
        }

        // Branch 1: a directory move. Detected via the literal (non-fuzzy)
        // resolver — a directory can never satisfy `resolve_within_data`'s
        // include-pattern check, so that resolver cannot be used for this test.
        if let Ok(abs) = crate::write::resolve_safe_write_path(&self.canonical_data_path, &raw)
            && abs.is_dir()
        {
            return self.write_document_move_dir(&params, &raw).await;
        }

        // Branch 2: a document create or edit. `resolve_within_data` (the same
        // resolver `get_document` uses to try a literal path before its own
        // basename fallback, which lives only there, not here) tells them apart:
        // a path that resolves to an existing, permitted file is an edit;
        // `NotFound` (nothing there yet) or `NotPermitted` (something exists
        // there, but of a non-indexable type) falls through to create, which
        // re-validates the path itself via `resolve_safe_write_path` and
        // `check_include_pattern_against` — independent, literal, KB-root-relative
        // checks that do not share `resolve_within_data`'s "try the raw absolute
        // path against the real filesystem first" ambiguity. That re-validation is
        // what makes it safe to fall through for those two cases rather than treat
        // every non-success as a hard failure, and it is also what restores the
        // pre-merge priority: a path that both exists on disk and fails the
        // include-pattern check is reported with that specific message, not a
        // generic "not found" one.
        //
        // `Outside` and `Other` are different: `resolve_within_data` only returns
        // `Outside` when the literal absolute path actually EXISTS outside the
        // data root (see its doc comment) — falling through to create would strip
        // the leading `/` and join it KB-relative, silently writing a *new* file
        // inside the KB at a path the caller never asked for while reporting
        // success. `Other` (a real I/O error resolving the path) has no safe
        // fallback interpretation either. Both are hard errors here, same as
        // `delete_document`'s handling of the same resolver just above.
        match retrieval::resolve_within_data(
            &raw,
            &self.canonical_data_path,
            &self.include_patterns,
        ) {
            Ok(canonical) => self.write_document_edit(params, canonical).await,
            Err(retrieval::ResolveErr::NotFound) | Err(retrieval::ResolveErr::NotPermitted) => {
                self.write_document_create(params, &raw).await
            }
            Err(retrieval::ResolveErr::Outside) => Err(McpError::invalid_params(
                "File path is outside the data directory".to_string(),
                None,
            )),
            Err(retrieval::ResolveErr::Other(msg)) => Err(McpError::invalid_params(msg, None)),
        }
    }

    #[tool(annotations(destructive_hint = true, idempotent_hint = false))]
    async fn delete_document(
        &self,
        Parameters(params): Parameters<DeleteDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        let config = self.config();

        let raw = params.path.trim();
        if raw.is_empty() {
            return Err(McpError::invalid_params(
                "path parameter is empty".to_string(),
                None,
            ));
        }
        // Mirrors `get_document`'s length guard (see its comment) and
        // `write_document`'s identical check: reject before the resolver, the
        // metadata index, or git ever see an input that was never going to resolve.
        if raw.len() > MAX_PATH_LEN {
            return Err(McpError::invalid_params(
                format!("path exceeds maximum length of {MAX_PATH_LEN} characters"),
                None,
            ));
        }

        // Resolve the path (must already exist on disk) with `resolve_within_data`,
        // the literal resolver `write_document` also tries first: relative to the KB
        // root (a leading `/` means the root), and bounded by the traversal and
        // `indexing.include` checks. Unlike `get_document`'s resolution it has no
        // basename fallback, so a bare basename is "does not exist". It produces
        // this tool's richer NotFound text, and stays here rather than in
        // `write::delete_document`, which does its own plain existence check as a
        // defense-in-depth fallback for callers (like the HTTP UI) that address a
        // document by exact path instead.
        let canonical = match retrieval::resolve_within_data(
            raw,
            &self.canonical_data_path,
            &self.include_patterns,
        ) {
            Ok(c) => c,
            Err(retrieval::ResolveErr::NotFound) => {
                return Err(McpError::invalid_params(
                    format!("document does not exist: '{}'", raw),
                    None,
                ));
            }
            Err(retrieval::ResolveErr::Outside) => {
                return Err(McpError::invalid_params(
                    "File path is outside the data directory".to_string(),
                    None,
                ));
            }
            Err(retrieval::ResolveErr::NotPermitted) => {
                return Err(McpError::invalid_params(
                    "File type not permitted".to_string(),
                    None,
                ));
            }
            Err(retrieval::ResolveErr::Other(msg)) => {
                return Err(McpError::invalid_params(msg, None));
            }
        };

        // Derive repo-relative path (used for git staging, commit messages, index purge).
        let rel_path = canonical
            .strip_prefix(&self.canonical_data_path)
            .unwrap_or(&canonical)
            .to_string_lossy()
            .into_owned();

        let token = crate::secrets::git_token(&config).map_err(credential_error)?;

        // (#229) Best-effort, same as `run_document_write`'s and
        // `write_document_move_dir`'s own lazy state-DB opens: a state DB that
        // fails to open degrades `delete_document`'s inbound-link check to
        // "skip it" (see `WriteDeps::state`'s doc comment), not a failed
        // delete. Without this, `write::delete_document`'s reverse-link query
        // never runs at all — `WriteDeps::state == None` — and
        // `referencing_paths` would always come back empty regardless of what
        // actually links to the document being deleted.
        let state_db = self.state_db().await.ok();
        let deps = self.write_deps(&config, token.as_deref(), state_db);

        match crate::write::delete_document(
            &deps,
            &rel_path,
            params.message.as_deref(),
            params.expected_version.as_deref(),
        )
        .await
        {
            Ok(success) => Ok(delete_success_to_result(success, &rel_path)),
            Err(err) => Err(delete_error_to_mcp_error(err, &rel_path)),
        }
    }
}

/// A relevance score as reported to a caller: four significant digits. The
/// full `f32`/`f64` precision is noise to a reader and costs bytes on every row
/// (and a widened `f32` such as 0.93 would print as 0.9300000071525574). Also
/// how a dedup refusal reports its `similarity` and `threshold`, here and in
/// `web.rs`.
pub(crate) fn round_score(score: f64) -> serde_json::Value {
    if !score.is_finite() || score == 0.0 {
        return serde_json::json!(score);
    }
    let magnitude = score.abs().log10().floor() as i32;
    let decimals = (3 - magnitude).max(0) as usize;
    let rounded = format!("{score:.decimals$}")
        .parse::<f64>()
        .unwrap_or(score);
    serde_json::json!(rounded)
}

/// The longest search snippet, in characters, before it is cut with `…`.
const SNIPPET_CHARS: usize = 800;

/// A row's frontmatter as `search` reports it: without the keys already promoted
/// to the row itself (`title`, `description`) or derived from the path
/// (`domain`, unless the caller named it in `fields`). `None` when nothing is
/// left.
fn row_frontmatter(
    frontmatter: &serde_json::Value,
    fields: Option<&[String]>,
) -> Option<serde_json::Value> {
    let mut map = frontmatter.as_object()?.clone();
    map.remove("title");
    map.remove("description");
    if !fields.is_some_and(|f| f.iter().any(|k| k == "domain")) {
        map.remove("domain");
    }
    (!map.is_empty()).then_some(serde_json::Value::Object(map))
}

/// The `search_grouped`/`search_enumerate` row for one document: `file_path`,
/// plus `title`, `description` and the remaining `frontmatter` when present.
fn document_row(
    summary: &crate::state::DocumentSummary,
    fields: Option<&[String]>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut row = serde_json::Map::new();
    row.insert("file_path".into(), serde_json::json!(summary.file_path));
    if let Some(title) = &summary.title {
        row.insert("title".into(), serde_json::json!(title));
    }
    if let Some(description) = &summary.description {
        row.insert("description".into(), serde_json::json!(description));
    }
    if let Some(frontmatter) = row_frontmatter(&summary.frontmatter, fields) {
        row.insert("frontmatter".into(), frontmatter);
    }
    row
}

/// The envelope keys every query-mode `search` response shares: `returned`, and
/// `path_prefix_truncated`/`offset_truncated` only when set.
fn search_envelope(
    returned: usize,
    path_prefix_truncated: bool,
    offset_truncated: bool,
) -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    map.insert("returned".into(), serde_json::json!(returned));
    if path_prefix_truncated {
        map.insert("path_prefix_truncated".into(), serde_json::json!(true));
    }
    if offset_truncated {
        map.insert("offset_truncated".into(), serde_json::json!(true));
    }
    map
}

/// Builds `search_chunks`' response from already-fetched results. A pure
/// function so the "results -> response" seam is reachable by a plain unit
/// test: no network, no mocked `KbSearchServer`.
///
/// Each row is `file_path`, `title`, `score` (four significant digits), `text`
/// (the chunk body without its breadcrumb/description prefix — see
/// `qdrant::chunk_body_text` — cut at [`SNIPPET_CHARS`] with `…`), plus
/// `heading_path`, `line_start`/`line_end`, `type` and `tags` when the payload
/// has them. With `explain`, the response carries `mode` (constant across the
/// result set) and each row its per-arm scores and `phrase_matched: true` when
/// they apply.
fn build_chunk_search_payload(
    results: &[crate::qdrant::SearchResult],
    data_root: &Path,
    explain: bool,
    mode: &str,
    path_prefix_truncated: bool,
    offset_truncated: bool,
) -> serde_json::Value {
    let mut rows: Vec<serde_json::Value> = Vec::with_capacity(results.len());
    for result in results {
        let payload = &result.payload;
        let mut row = serde_json::Map::new();
        let file_path_raw = payload
            .get("file_path")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        row.insert(
            "file_path".into(),
            serde_json::json!(retrieval::relative_to_data(file_path_raw, data_root)),
        );
        row.insert(
            "title".into(),
            serde_json::json!(
                payload
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(untitled)")
            ),
        );
        // Only when `chunking.heading_metadata` was on at index time (#286).
        if let Some(heading_path) = payload.get(crate::qdrant::HEADING_PATH_KEY) {
            row.insert("heading_path".into(), heading_path.clone());
        }
        for key in ["line_start", "line_end"] {
            if let Some(v) = payload.get(key).filter(|v| !v.is_null()) {
                row.insert(key.into(), v.clone());
            }
        }
        row.insert("score".into(), round_score(f64::from(result.score)));
        let body = crate::qdrant::chunk_body_text(payload);
        let mut chars = body.chars();
        let mut snippet: String = chars.by_ref().take(SNIPPET_CHARS).collect();
        if chars.next().is_some() {
            snippet.push('…');
        }
        row.insert("text".into(), serde_json::json!(snippet));
        if let Some(t) = payload
            .get("type")
            .filter(|v| v.as_str().is_some_and(|s| !s.is_empty()))
        {
            row.insert("type".into(), t.clone());
        }
        if let Some(tags) = payload
            .get("tags")
            .filter(|v| v.as_array().is_some_and(|a| !a.is_empty()))
        {
            row.insert("tags".into(), tags.clone());
        }
        if explain {
            for (key, value) in [
                ("dense_score", result.dense_score),
                ("sparse_score", result.sparse_score),
                ("pre_rerank_score", result.pre_rerank_score),
            ] {
                if let Some(v) = value {
                    row.insert(key.into(), round_score(f64::from(v)));
                }
            }
            // Presence, not magnitude, is the signal: the phrase arm's "score" is
            // just the dense-ranked query re-run under a phrase filter.
            if result.phrase_score.is_some() {
                row.insert("phrase_matched".into(), serde_json::json!(true));
            }
        }
        rows.push(serde_json::Value::Object(row));
    }

    let mut out = search_envelope(rows.len(), path_prefix_truncated, offset_truncated);
    if explain {
        out.insert("mode".into(), serde_json::json!(mode));
    }
    out.insert("results".into(), serde_json::Value::Array(rows));
    serde_json::Value::Object(out)
}

/// Builds `search_grouped`'s response from already-fetched grouped documents:
/// one [`document_row`] plus a rounded `score` per document, and deliberately no
/// `total`/`has_more` (grouped vector search cannot back either).
fn build_grouped_search_payload(
    documents: &[retrieval::GroupedDocument],
    fields: Option<&[String]>,
    path_prefix_truncated: bool,
    offset_truncated: bool,
) -> serde_json::Value {
    let mut out = search_envelope(documents.len(), path_prefix_truncated, offset_truncated);
    out.insert(
        "documents".into(),
        documents
            .iter()
            .map(|d| {
                let mut row = document_row(&d.summary, fields);
                row.insert("score".into(), round_score(f64::from(d.score)));
                serde_json::Value::Object(row)
            })
            .collect(),
    );
    serde_json::Value::Object(out)
}

/// Builds `search`'s `section` granularity response from already-fetched
/// section hits (#286): `file_path`, `heading_path` (omitted for the two
/// heading-less kinds), the section's `line_start`/`line_end`, the matched
/// chunk's `hit_line_start`/`hit_line_end`, a rounded `score`, and `scope` only
/// when it is not an ordinary `section`. Deliberately **no** text: that
/// omission is the whole point of this granularity.
fn build_section_search_payload(
    sections: &[retrieval::SectionHit],
    path_prefix_truncated: bool,
    offset_truncated: bool,
) -> serde_json::Value {
    let mut out = search_envelope(sections.len(), path_prefix_truncated, offset_truncated);
    out.insert(
        "results".into(),
        sections
            .iter()
            .map(|s| {
                let mut row = serde_json::Map::new();
                row.insert("file_path".into(), serde_json::json!(s.file_path));
                if !s.heading_path.is_empty() {
                    row.insert("heading_path".into(), serde_json::json!(s.heading_path));
                }
                row.insert("line_start".into(), serde_json::json!(s.line_start));
                row.insert("line_end".into(), serde_json::json!(s.line_end));
                row.insert("hit_line_start".into(), serde_json::json!(s.hit_line_start));
                row.insert("hit_line_end".into(), serde_json::json!(s.hit_line_end));
                row.insert("score".into(), round_score(f64::from(s.score)));
                if !matches!(s.scope, retrieval::SectionScope::Section) {
                    row.insert("scope".into(), serde_json::json!(s.scope.as_str()));
                }
                serde_json::Value::Object(row)
            })
            .collect(),
    );
    serde_json::Value::Object(out)
}
#[tool_handler]
impl ServerHandler for KbSearchServer {
    fn get_info(&self) -> ServerInfo {
        let instructions = self
            .instructions
            .read()
            .unwrap_or_else(|poisoned| {
                warn!("Instructions RwLock poisoned on read; using last value");
                poisoned.into_inner()
            })
            .clone();
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION"),
            ))
            .with_instructions(instructions)
    }

    // Hand-written rather than left for `#[tool_handler]` to generate: the
    // macro regenerates the tool list from `Self::tool_router()` (and its
    // compile-time `#[tool(...)]` attributes) on every call, so a description
    // that needs to change at runtime — per `descriptions.rs`'s whole point —
    // cannot be baked into that attribute. All three methods below build their
    // router from `self.enabled_tool_router()` rather than the bare
    // `Self::tool_router()`, so a name in the live `mcp.disabled_tools` is
    // hidden from `list_tools`, absent from `get_tool`, and refused by
    // `call_tool` — see that method's doc comment for why `#[tool_handler]`
    // does not generate a conflicting one. `list_tools`/`get_tool` additionally
    // apply this server's live `description_overlay`, then the live
    // per-instance schema restrictions (`overlay_input_schema`, #286), then the
    // stateless `tool_schema::self_contained` rewrite that strips `$ref`/
    // `$defs` and boolean subschemas for llama.cpp-backed clients (#288), on
    // top of the router's own `Tool` entries.
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let tools = self
            .enabled_tool_router()
            .list_all()
            .into_iter()
            .map(|tool| self.overlay_description(tool))
            .map(|tool| self.overlay_input_schema(tool))
            .map(tool_schema::self_contained)
            .collect();
        Ok(ListToolsResult::with_all_items(tools))
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        self.enabled_tool_router()
            .get(name)
            .cloned()
            .map(|tool| self.overlay_description(tool))
            .map(|tool| self.overlay_input_schema(tool))
            .map(tool_schema::self_contained)
    }

    /// Hand-written, rather than left for `#[tool_handler]` to generate
    /// (`rmcp_macros::tool_handler` only fills in a `call_tool` when the impl
    /// block does not already define one, so this is not a duplicate): the
    /// generated version dispatches through the bare `Self::tool_router()`,
    /// which knows nothing about `mcp.disabled_tools`. This does exactly what
    /// the macro would — `ToolCallContext::new` plus `.call(tcc).await` — but
    /// through `self.enabled_tool_router()`, so a disabled tool's `call`
    /// reaches `ToolRouter::call`'s own disabled-route check and comes back as
    /// `invalid_params("tool not found")`, identical to an unknown tool name.
    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let router = self.enabled_tool_router();
        let tcc = ToolCallContext::new(self, request, context);
        router.call(tcc).await
    }
}

/// Builds a minimal `ResolvedConfig` for tests. Shared with `server.rs`'s test
/// module, which needs the same handler-construction pattern for the MCP
/// service without pulling in a real config file.
#[cfg(test)]
pub(crate) fn make_test_resolved_config(data_path: &std::path::Path) -> Arc<ResolvedConfig> {
    Arc::new(ResolvedConfig {
        source: crate::config::ResolvedSourceConfig {
            git_url: None,
            branch: "master".into(),
            data_path: Some(data_path.to_string_lossy().into_owned()),
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
        },
        qdrant: crate::config::ResolvedQdrantConfig {
            url: "http://localhost:6334".into(),
            collection: "test".into(),
        },
        validation: crate::config::ValidationConfig::default(),
        webhook: crate::config::WebhookConfig::default(),
        mcp: crate::config::ResolvedMcpConfig::default(),
        rate_limit: crate::config::RateLimitConfig::default(),
        write: crate::config::WriteConfig::default(),
        search: crate::config::SearchConfig::default(),
        reranking: None,
        ui: crate::config::UiConfig::default(),
        provenance: Default::default(),
    })
}

/// An empty `SharedSchemaCache`, for tests that exercise a `KbSearchServer` but do
/// not care about schema content (e.g. instructions plumbing, path validation).
/// Tests that DO care build a real one from a temp dir's schema files —
/// see `make_write_test_server` below.
#[cfg(test)]
pub(crate) fn empty_test_schema_cache() -> crate::schema::SharedSchemaCache {
    Arc::new(RwLock::new(Arc::new(crate::schema::SchemaCache::default())))
}

/// An empty description overlay, for tests that exercise a `KbSearchServer` but
/// do not care about tool descriptions — `list_tools`/`get_tool` fall back to
/// whatever the router itself produced (`None`, since no `#[tool(...)]`
/// attribute carries one) when a tool has no overlay entry.
#[cfg(test)]
pub(crate) fn empty_test_description_overlay() -> Arc<RwLock<HashMap<String, String>>> {
    Arc::new(RwLock::new(HashMap::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    // These now live in `write.rs` (the dedup gate and commit-message/diff
    // helpers moved there with the rest of the write pipeline); imported here
    // so the tests below — ported verbatim — keep compiling unchanged.
    use crate::write::{
        build_commit_message, build_dedup_query, dedup_search_opts, dedup_verdict,
        render_unified_diff,
    };

    /// Mirrors `write::DEDUP_QUERY_CHAR_LIMIT` (private there), so the ported
    /// `build_dedup_query_*` tests below keep compiling unchanged.
    const DEDUP_QUERY_CHAR_LIMIT: usize = 2000;

    // --- list_documents filter parsing ---

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
        let filters =
            parsed_filters(serde_json::json!({ "tags": { "all_of": ["recipe", "dinner"] } }));
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
        let filters = parsed_filters(
            serde_json::json!({ "planning.needs_recipe": false, "planning.rating": 5 }),
        );
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
                    types.contains(&"string")
                        && types.contains(&"number")
                        && types.contains(&"boolean")
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
        let err = serde_json::from_value::<SearchFiltersInput>(serde_json::json!(["type", "x"]))
            .unwrap_err();
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
        let err = resolve_search_granularity(true, Some("paragraph"), &Granularity::ALL, false)
            .unwrap_err();
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

    /// Caller-facing rejections name no config keys, payload fields or indexing
    /// mechanics — none of which a caller can act on.
    fn assert_no_internals_in_caller_error(msg: &str) {
        for internal in [
            "chunking.",
            "heading_metadata",
            "section_key",
            "heading_prefixes",
            "payload",
            "re-index",
            "index --full",
        ] {
            assert!(!msg.contains(internal), "{internal:?} leaked into: {msg}");
        }
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

    /// A server built from `make_test_resolved_config`'s defaults but with
    /// `chunking.heading_metadata` forced on — needed to reach `search_sections`'
    /// explain/fields rejections, which sit BEHIND the heading_metadata check in
    /// the handler and would otherwise always report the wrong error.
    fn schema_tool_server_with_heading_metadata(tmp: &tempfile::TempDir) -> KbSearchServer {
        let mut config = make_test_resolved_config(tmp.path());
        Arc::make_mut(&mut config).chunking.heading_metadata = true;
        make_write_test_server(tmp, &["**/*.md".to_string()], config)
    }

    /// A server built from `make_test_resolved_config`'s defaults but with
    /// `search.granularities` restricted to exactly `granularities` (#286) —
    /// needed to reach `resolve_search_granularity`'s
    /// disabled-granularity rejection and default-fallback behavior through
    /// the real `search` tool entry point, rather than only unit-testing the
    /// pure resolver directly (see the tests above this section).
    fn schema_tool_server_with_granularities(
        tmp: &tempfile::TempDir,
        granularities: &[Granularity],
    ) -> KbSearchServer {
        let mut config = make_test_resolved_config(tmp.path());
        Arc::make_mut(&mut config).search.granularities = granularities.to_vec();
        make_write_test_server(tmp, &["**/*.md".to_string()], config)
    }

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
        assert!(
            conditions.contains(&qdrant_client::qdrant::Condition::matches(
                "domain",
                vec!["sysadmin".to_string()]
            ))
        );
        assert!(
            conditions.contains(&qdrant_client::qdrant::Condition::matches(
                "type",
                vec!["guide".to_string()]
            ))
        );
        assert!(
            conditions.contains(&qdrant_client::qdrant::Condition::matches(
                "tags",
                vec!["rust".to_string(), "rag".to_string()]
            ))
        );
    }

    // --- schema tool helpers ---

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
            "type", "required", "indexed", "values", "default", "open", "fields",
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
        let reparsed: crate::schema::SchemaFile = serde_yaml_ng::from_str(&yaml).unwrap();

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

    /// A server whose state DB lives under the temp KB, so metadata-backed tools work.
    fn schema_tool_server(tmp: &tempfile::TempDir) -> KbSearchServer {
        let config = make_test_resolved_config(tmp.path());
        make_write_test_server(tmp, &["**/*.md".to_string()], config)
    }

    async fn seed_document(
        server: &KbSearchServer,
        rel_path: &str,
        frontmatter: serde_json::Value,
    ) {
        let map = match frontmatter {
            serde_json::Value::Object(m) => {
                m.into_iter().collect::<std::collections::HashMap<_, _>>()
            }
            _ => panic!("frontmatter fixture must be an object"),
        };
        server
            .state_db()
            .await
            .unwrap()
            .upsert_document_metadata(rel_path, &map, 100, "hash", 1)
            .await
            .unwrap();
    }

    fn write_schema_file(tmp: &tempfile::TempDir, dir: &str, yaml: &str) {
        let target = tmp.path().join(dir);
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join(crate::schema::SCHEMA_FILE_NAME), yaml).unwrap();
    }

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
            "fields:\n  tags:\n    type: list\n  status:\n    type: enum\n    values: [active]\n",
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
        // A deployment that filters on `domain` lists it in `indexed_fields`, which
        // declares it. Ingest still derives it from the folder and a write that
        // authors it is refused, so it must not be offered as a field to fill in.
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
        assert_eq!(names, vec!["status"], "{all}");
        assert!(all.get("omitted_fields").is_none(), "{all}");

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

    #[tokio::test]
    async fn search_filter_on_an_unknown_field_lists_the_filterable_fields() {
        let tmp = tempfile::tempdir().unwrap();
        write_schema_file(
            &tmp,
            "",
            "fields:\n  status:\n    type: enum\n    values: [active]\n",
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
            err.message.contains(
                "filter 'status': 'actve' is not an allowed value; allowed: active, draft"
            ),
            "a repeated value is listed once: {}",
            err.message
        );

        // `all_of` needs one document to carry every value, so a single value that
        // nothing can match sinks it even beside a permitted one.
        let err = search(serde_json::json!({ "all_of": ["active", "actve"] }))
            .await
            .unwrap_err();
        assert!(
            err.message.contains(
                "filter 'status': 'actve' is not an allowed value; allowed: active, draft"
            ),
            "{}",
            err.message
        );
        search(serde_json::json!({ "all_of": ["active", "draft"] }))
            .await
            .unwrap();

        // A scalar is a one-value any-of.
        let err = search(serde_json::json!("actve")).await.unwrap_err();
        assert!(
            err.message.contains(
                "filter 'status': 'actve' is not an allowed value; allowed: active, draft"
            ),
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
    async fn update_schema_dry_run_writes_nothing_and_reports_casualties() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);
        seed_document(&server, "notes/a.md", serde_json::json!({ "title": "A" })).await;

        let result = server
            .update_schema(Parameters(UpdateSchemaParams {
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
        let reparsed: crate::schema::SchemaFile = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(
            reparsed.fields["status"].values,
            Some(vec!["active".to_string(), "draft".to_string()])
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

    #[tokio::test]
    async fn update_schema_root_add_values_refused_without_acknowledgment() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);

        let err = server
            .update_schema(Parameters(UpdateSchemaParams {
                path: None,
                operation: "add_values".into(),
                field: "tags".into(),
                values: Some(vec!["x".into()]),
                definition: None,
                dry_run: None,
                force: None,
                acknowledge_root_change: None,
            }))
            .await
            .expect_err("root add_values must be refused without acknowledge_root_change");

        let msg = format!("{:?}", err);
        assert!(
            msg.contains("root schema changes are guarded"),
            "got: {msg}"
        );
        assert!(msg.contains("meta/schema-tag-policy.md"), "got: {msg}");
        assert!(msg.contains("acknowledge_root_change"), "got: {msg}");
        assert!(
            !tmp.path().join(crate::schema::SCHEMA_FILE_NAME).exists(),
            "a refused change must leave the filesystem untouched"
        );
    }

    #[tokio::test]
    async fn update_schema_root_set_field_and_remove_field_are_refused_without_acknowledgment() {
        let tmp = tempfile::tempdir().unwrap();
        write_schema_file(
            &tmp,
            "",
            "fields:\n  tags:\n    type: enum\n    values: [x]\n",
        );
        let server = schema_tool_server(&tmp);

        let err = server
            .update_schema(Parameters(UpdateSchemaParams {
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

    #[tokio::test]
    async fn update_schema_root_add_values_allowed_with_acknowledgment() {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let (server, _config) = make_git_backed_server(&work);

        server
            .update_schema(Parameters(UpdateSchemaParams {
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

        assert!(
            work.path().join(crate::schema::SCHEMA_FILE_NAME).exists(),
            "the acknowledged change must actually be written"
        );
    }

    #[tokio::test]
    async fn update_schema_root_add_values_allowed_with_dry_run() {
        let tmp = tempfile::tempdir().unwrap();
        let server = schema_tool_server(&tmp);

        let result = server
            .update_schema(Parameters(UpdateSchemaParams {
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
        assert!(
            !tmp.path().join(crate::schema::SCHEMA_FILE_NAME).exists(),
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
            "fields:\n  tags:\n    type: enum\n    values: [x, y]\n",
        );
        git_commit_all(&work, crate::schema::SCHEMA_FILE_NAME, "add root schema");
        let (server, _config) = make_git_backed_server(&work);

        server
            .update_schema(Parameters(UpdateSchemaParams {
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
        let reparsed: crate::schema::SchemaFile = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(reparsed.fields["tags"].values, Some(vec!["x".to_string()]));
    }

    #[tokio::test]
    async fn update_schema_non_root_add_values_allowed_without_acknowledgment() {
        // Already exercised incidentally by other update_schema tests (e.g.
        // `update_schema_accepts_a_change_that_breaks_nothing`), but this test names
        // the property the root guard must not regress: the gate is root-only.
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        let (server, _config) = make_git_backed_server(&work);

        server
            .update_schema(Parameters(UpdateSchemaParams {
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

    #[test]
    fn omitted_limit_uses_the_configured_default() {
        assert_eq!(resolve_limit(None, 10, 50), 10);
        // The default is configurable, not baked in — a deployment that raised it
        // must see its own value, not the historical 10.
        assert_eq!(resolve_limit(None, 25, 50), 25);
    }

    #[test]
    fn requested_limit_within_max_is_preserved() {
        assert_eq!(resolve_limit(Some(25), 10, 50), 25);
    }

    #[test]
    fn requested_limit_above_max_is_clamped_to_the_configured_max() {
        assert_eq!(resolve_limit(Some(1_000_000), 10, 50), 50);
        // Clamped to the CONFIGURED ceiling, not a hardcoded one: raising
        // max_limit must actually raise what a caller can ask for.
        assert_eq!(resolve_limit(Some(1_000_000), 10, 200), 200);
    }

    #[test]
    fn zero_limit_is_passed_through() {
        assert_eq!(resolve_limit(Some(0), 10, 50), 0);
    }

    #[test]
    fn ellipsis_uses_char_count_not_byte_len() {
        // 800 chars of a 2-byte character = 1600 bytes
        let text: String = "é".repeat(801);
        assert!(text.len() > 800, "byte len should exceed 800");
        assert!(text.chars().count() > 800, "char count should exceed 800");
        // If we used .len() on a 800-char string it would wrongly trigger ellipsis
        let short: String = "é".repeat(800);
        assert!(
            short.len() > 800,
            "byte len of 800 2-byte chars exceeds 800"
        );
        assert_eq!(short.chars().count(), 800, "char count is exactly 800");
    }

    #[test]
    fn include_globset_matches_markdown() {
        let patterns = vec!["**/*.md".to_string()];
        let gs = build_include_globset(&patterns);
        assert!(
            gs.is_match("docs/guide.md"),
            "**/*.md should match docs/guide.md"
        );
        assert!(
            gs.is_match("README.md"),
            "**/*.md should match top-level README.md"
        );
    }

    #[test]
    fn include_globset_rejects_non_markdown() {
        let patterns = vec!["**/*.md".to_string()];
        let gs = build_include_globset(&patterns);
        assert!(
            !gs.is_match("state.db"),
            "**/*.md should not match state.db"
        );
        assert!(
            !gs.is_match("scripts/run.sh"),
            "**/*.md should not match shell scripts"
        );
        assert!(
            !gs.is_match(".env"),
            "**/*.md should not match credential files"
        );
    }

    #[test]
    fn include_globset_respects_custom_patterns() {
        let patterns = vec!["**/*.md".to_string(), "**/*.txt".to_string()];
        let gs = build_include_globset(&patterns);
        assert!(gs.is_match("notes/todo.txt"), "should match *.txt");
        assert!(!gs.is_match("data.json"), "should not match *.json");
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

    /// Build a bare-bones `KbSearchServer` for overlay tests — same
    /// construction pattern as `get_document_rejects_overlong_path` below,
    /// parameterized by the description overlay under test.
    fn make_overlay_test_server(overlay: HashMap<String, String>) -> KbSearchServer {
        let tmp = tempfile::tempdir().unwrap();
        make_overlay_test_server_with_config(overlay, make_test_resolved_config(tmp.path()))
    }

    /// Same as [`make_overlay_test_server`], but with a caller-supplied
    /// config rather than `make_test_resolved_config`'s defaults — needed by
    /// the schema-overlay tests below (#286), which exercise
    /// `overlay_input_schema`/`get_tool`/`list_tools`'s equivalent against a
    /// restricted `search.granularities` or `chunking.heading_metadata`.
    fn make_overlay_test_server_with_config(
        overlay: HashMap<String, String>,
        config: Arc<ResolvedConfig>,
    ) -> KbSearchServer {
        let tmp = tempfile::tempdir().unwrap();
        let instructions = Arc::new(RwLock::new("test".to_string()));
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
        };
        let embed = Arc::new(EmbedClient::new(&embed_config));
        // `KbSearchServer::new` canonicalizes `tmp`'s path during construction
        // but never touches the filesystem again afterward (neither does
        // anything these overlay tests exercise — `get_tool`/`list_tools` are
        // pure lock reads), so `tmp` can safely drop once construction
        // returns.
        let server = KbSearchServer::new(
            embed,
            qdrant,
            "test".into(),
            tmp.path().to_path_buf(),
            &["**/*.md".to_string()],
            instructions,
            crate::config::shared_config(config),
            empty_test_schema_cache(),
            None,
            Arc::new(crate::reindex::ReindexQueue::new()),
            Arc::new(RwLock::new(overlay)),
        )
        .unwrap();
        drop(tmp);
        server
    }

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

    /// A server whose live config disables a tool: `mcp.disabled_tools`.
    fn make_test_server_with_disabled_tools(disabled_tools: Vec<String>) -> KbSearchServer {
        let mut config = make_test_resolved_config(&std::env::temp_dir());
        Arc::make_mut(&mut config).mcp.disabled_tools = disabled_tools;
        make_overlay_test_server_with_config(HashMap::new(), config)
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

    fn overlay_test_config(
        granularities: &[Granularity],
        heading_metadata: bool,
    ) -> Arc<ResolvedConfig> {
        let mut config = make_test_resolved_config(&std::env::temp_dir());
        {
            let resolved = Arc::make_mut(&mut config);
            resolved.search.granularities = granularities.to_vec();
            resolved.chunking.heading_metadata = heading_metadata;
        }
        config
    }

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
            crate::descriptions::granularity_description(&[
                Granularity::Chunk,
                Granularity::Document
            ])
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
                let overlay =
                    crate::descriptions::compose_tool_descriptions(None, true, &effective);
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
        for key in [
            "type", "required", "indexed", "values", "default", "open", "fields",
        ] {
            assert!(
                !object_branch["properties"][key].is_null(),
                "definition schema is missing documented key '{key}': {object_branch}"
            );
        }
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

    #[test]
    fn include_globset_empty_patterns_falls_back_to_markdown() {
        let gs = build_include_globset(&[]);
        assert!(
            gs.is_match("docs/guide.md"),
            "empty patterns should fall back to **/*.md"
        );
        assert!(
            gs.is_match("README.md"),
            "empty patterns should match top-level .md"
        );
        assert!(
            !gs.is_match("state.db"),
            "empty patterns fallback should not match non-markdown"
        );
    }

    #[test]
    fn include_globset_all_invalid_falls_back_to_markdown() {
        let gs = build_include_globset(&["[invalid".into()]);
        assert!(
            gs.is_match("docs/guide.md"),
            "all-invalid patterns should fall back to **/*.md"
        );
        assert!(
            !gs.is_match("data.json"),
            "all-invalid patterns fallback should not match non-markdown"
        );
    }

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
    fn single_representation(result: &CallToolResult) -> serde_json::Value {
        let structured = result
            .structured_content
            .clone()
            .expect("every tool result carries structured_content");
        assert_eq!(result.content.len(), 1, "one content block: {result:?}");
        let text = match &result.content[0].raw {
            rmcp::model::RawContent::Text(t) => t.text.clone(),
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

    /// Run `git <args>` in `dir`, panicking on failure.
    fn run_git(dir: &std::path::Path, args: &[&str]) {
        let out = std::process::Command::new("git")
            .current_dir(dir)
            .args([
                "-c",
                "user.email=t@localhost",
                "-c",
                "user.name=Test",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success(), "git {args:?}: {out:?}");
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

        let (text, structured) =
            get_document_result(&server, section_doc_params(Default::default()))
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

        let (text, structured) =
            get_document_result(&server, section_doc_params(Default::default()))
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

    /// Build a KbSearchServer suitable for write_document unit tests.
    /// Uses a temp directory as the data path. No real Qdrant/git required
    /// for tests that fail before those steps.
    fn make_write_test_server(
        tmp: &tempfile::TempDir,
        include_patterns: &[String],
        config: Arc<ResolvedConfig>,
    ) -> KbSearchServer {
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
        };
        let embed = Arc::new(EmbedClient::new(&embed_config));
        let instructions = Arc::new(RwLock::new(String::new()));
        // Built from whatever `.kb-schema.yaml` files already exist under `tmp` at
        // this point — callers that need a test to see a schema written must write it
        // before calling this, exactly as they already do for `write_schema_file`.
        let canonical = tmp.path().canonicalize().unwrap();
        let schema_cache: crate::schema::SharedSchemaCache = Arc::new(RwLock::new(Arc::new(
            crate::schema::SchemaCache::build_for_test(&canonical, &config.frontmatter),
        )));
        KbSearchServer::new(
            embed,
            qdrant,
            "test".into(),
            tmp.path().to_path_buf(),
            include_patterns,
            instructions,
            crate::config::shared_config(config),
            schema_cache,
            None,
            Arc::new(crate::reindex::ReindexQueue::new()),
            empty_test_description_overlay(),
        )
        .unwrap()
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
        };
        let result = server.write_document(Parameters(params)).await;

        let result =
            result.expect("write_document must upsert rather than refuse an existing path");
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
    async fn write_document_frontmatter_patch_validation_failure_reports_error_and_writes_nothing()
    {
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
        };
        let result = server.write_document(Parameters(params)).await;

        // `/etc/passwd` really exists on the filesystem, so `resolve_within_data`
        // resolves the literal absolute path first (see its doc comment) and
        // returns `Outside` — a hard error from the dispatch above, caught before
        // ever reaching create's include-pattern check. This differs from the
        // "leading `/` means the KB root" convenience: that convenience only
        // applies once the literal absolute path does NOT exist (see
        // `write_document_on_absolute_path_to_real_file_outside_kb_is_hard_error`
        // for that case, and for why the misrouted version of this dispatch used
        // to make this test pass for the wrong reason — the include-pattern guard
        // it originally meant to exercise never actually ran on this input).
        assert!(
            result.is_err(),
            "an absolute path resolving to a real file outside the KB must be rejected"
        );
        let err = result.unwrap_err();
        assert!(
            err.message.contains("outside the data directory"),
            "error should cite the outside-data-directory guard, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn write_document_on_absolute_path_to_real_file_outside_kb_is_hard_error() {
        // `resolve_within_data` tries an absolute path literally first (see its
        // doc comment). When that literal path EXISTS but is outside the data
        // root, it returns `ResolveErr::Outside` — which the dispatch above must
        // treat as a hard failure, not fall through to `write_document_create`.
        // Falling through would strip the leading `/` and join it KB-relative,
        // silently creating a *new* file inside the KB at a path the caller never
        // asked for, while reporting success.
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
        };
        let result = server.write_document(Parameters(params)).await;

        assert!(
            result.is_err(),
            "a real absolute path outside the KB must be a hard error, not silently \
             redirected into a create inside the KB"
        );
        let err = result.unwrap_err();
        assert!(
            err.message.contains("outside the data directory"),
            "error should say the path is outside the data directory, got: {}",
            err.message
        );

        // The file outside the KB must be untouched, and nothing must have been
        // created inside the KB at the stripped-leading-slash path.
        assert_eq!(
            std::fs::read_to_string(&outside_file).unwrap(),
            "# Not part of the KB",
            "the real file outside the KB must not be overwritten"
        );
        let bogus_created_path = tmp.path().join(crate::retrieval::kb_root_relative(
            outside_file.to_str().unwrap(),
        ));
        assert!(
            !bogus_created_path.exists(),
            "a bogus file must not have been created inside the KB at {}",
            bogus_created_path.display()
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
                commit_author_name: "mcp-md-wiki".to_string(),
                commit_author_email: "mcp-md-wiki@localhost".to_string(),
            },
            search: crate::config::SearchConfig::default(),
            reranking: None,
            ui: crate::config::UiConfig::default(),
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

    /// Build a `KbSearchServer` backed by a real git working clone, so write tools get
    /// past `commit_and_sync` and reach the point where they mark paths dirty.
    ///
    /// `make_write_test_server` gives this server its own private `ReindexQueue`
    /// (see that function), so the tests below read it back via
    /// `server.reindex_queue` — no other test in the binary can have marked
    /// anything on it.
    fn make_git_backed_server(
        work: &tempfile::TempDir,
    ) -> (KbSearchServer, Arc<crate::config::ResolvedConfig>) {
        let mut config = make_test_resolved_config(work.path());
        // Bypass the dedup gate: it would otherwise call out to a (nonexistent)
        // embedding service before we ever reach the commit.
        Arc::get_mut(&mut config).unwrap().write.dedup_enabled = false;
        let server = make_write_test_server(work, &["**/*.md".to_string()], Arc::clone(&config));
        (server, config)
    }

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

        crate::reindex::test_support::assert_marked_dirty(
            &server.reindex_queue,
            &["docs/queued.md"],
        );
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
            structured.get("diff_truncated").is_none()
                && structured.get("diff_total_bytes").is_none(),
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
            text.contains("`domain` is set by the document's top-level folder"),
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

    /// Run git in `dir` as a peer writer, never escaping into an enclosing repo.
    fn peer_git(dir: &std::path::Path, args: &[&str]) {
        let out = std::process::Command::new("git")
            .args(["-c", "user.email=peer@test.com", "-c", "user.name=Peer"])
            .args(args)
            .current_dir(dir)
            .env("GIT_CEILING_DIRECTORIES", dir.parent().unwrap_or(dir))
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// A knowledge base clone the server writes through (`work`, syncing to
    /// `bare`), seeded with `docs/target.md` (`target`), plus a second clone
    /// (`peer`) of the same state for a concurrent writer.
    struct ConcurrentHarness {
        _bare: tempfile::TempDir,
        work: tempfile::TempDir,
        peer: tempfile::TempDir,
        server: KbSearchServer,
    }

    const CONCURRENT_TARGET: &str = "---\ntitle: Old\n---\n\n# Body\n\none\ntwo\nthree\nfour\n\
                                     five\nsix\nseven\neight\nlast line\n";

    fn concurrent_harness() -> ConcurrentHarness {
        let bare = crate::git::tests::create_bare_repo("master");
        let work = crate::git::tests::clone_bare_repo(bare.path(), "master");
        std::fs::create_dir_all(work.path().join("docs")).unwrap();
        std::fs::write(work.path().join("docs/target.md"), CONCURRENT_TARGET).unwrap();
        peer_git(work.path(), &["add", "docs/target.md"]);
        peer_git(work.path(), &["commit", "-q", "-m", "seed"]);
        peer_git(work.path(), &["push", "-q", "origin", "HEAD:master"]);
        let peer = crate::git::tests::clone_bare_repo(bare.path(), "master");

        let mut config = make_test_resolved_config(work.path());
        {
            let c = Arc::get_mut(&mut config).unwrap();
            c.write.dedup_enabled = false;
            c.source.git_url = Some(format!("file://{}", bare.path().display()));
        }
        let server = make_write_test_server(&work, &["**/*.md".to_string()], config);
        ConcurrentHarness {
            _bare: bare,
            work,
            peer,
            server,
        }
    }

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

    /// Run `fut` with a git test hook that runs `action` the first time the write
    /// reaches `point` (a nested write inside `action` reaches it too, and is left
    /// alone, since the action is already taken).
    async fn with_hook_once<A, AF, T>(
        point: crate::git::HookPoint,
        action: A,
        fut: impl std::future::Future<Output = T>,
    ) -> T
    where
        A: FnOnce() -> AF + Send + 'static,
        AF: std::future::Future<Output = ()> + Send + 'static,
    {
        let action = Arc::new(std::sync::Mutex::new(Some(action)));
        let hook: crate::git::TestHook = Arc::new(move |p| {
            let action = Arc::clone(&action);
            Box::pin(async move {
                if p != point {
                    return;
                }
                let taken = action.lock().unwrap().take();
                if let Some(action) = taken {
                    action().await;
                }
            })
        });
        crate::git::TEST_HOOK.scope(hook, fut).await
    }

    /// Like [`with_hook_once`], but runs `action(n)` every time (`n` counts from 1).
    async fn with_hook_each<A, T>(
        point: crate::git::HookPoint,
        action: A,
        fut: impl std::future::Future<Output = T>,
    ) -> T
    where
        A: Fn(usize) + Send + Sync + 'static,
    {
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let action = Arc::new(action);
        let hook: crate::git::TestHook = Arc::new(move |p| {
            let count = Arc::clone(&count);
            let action = Arc::clone(&action);
            Box::pin(async move {
                if p == point {
                    let n = count.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                    action(n);
                }
            })
        });
        crate::git::TEST_HOOK.scope(hook, fut).await
    }

    /// Commit `content` to `rel` in the peer clone and push it.
    fn peer_push(peer: &std::path::Path, rel: &str, content: &str) {
        if let Some(parent) = std::path::Path::new(rel).parent() {
            std::fs::create_dir_all(peer.join(parent)).unwrap();
        }
        peer_git(peer, &["pull", "-q", "--rebase", "origin", "master"]);
        std::fs::write(peer.join(rel), content).unwrap();
        peer_git(peer, &["add", rel]);
        peer_git(peer, &["commit", "-q", "-m", &format!("peer writes {rel}")]);
        peer_git(peer, &["push", "-q", "origin", "HEAD:master"]);
    }

    fn rev(dir: &std::path::Path, rev: &str) -> String {
        let out = std::process::Command::new("git")
            .args(["rev-parse", rev])
            .current_dir(dir)
            .env("GIT_CEILING_DIRECTORIES", dir.parent().unwrap_or(dir))
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn target_version(h: &ConcurrentHarness) -> String {
        write::document_version(&std::fs::read(h.work.path().join("docs/target.md")).unwrap())
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
        for name in ["search", "get_document", "get_schema"] {
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

    /// `HEAD` of `work`, as a trimmed hex string.
    /// The current version of `rel` under `dir`, as `get_document` would report
    /// it; `None` when it does not exist.
    fn version_at(dir: &tempfile::TempDir, rel: &str) -> Option<String> {
        std::fs::read(dir.path().join(rel))
            .ok()
            .map(|bytes| write::document_version(&bytes))
    }

    fn head_sha(work: &tempfile::TempDir) -> String {
        let out = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(work.path())
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// `git status --porcelain` of `work`, as a trimmed string ("" means clean).
    fn git_status(work: &tempfile::TempDir) -> String {
        let out = std::process::Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(work.path())
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn git_commit_all(work: &tempfile::TempDir, rel_path: &str, message: &str) {
        std::process::Command::new("git")
            .args(["add", "--", rel_path])
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
                message,
            ])
            .current_dir(work.path())
            .output()
            .unwrap();
    }

    /// Force `git commit` to fail in `work` while leaving the repo otherwise
    /// completely healthy: `git add` still succeeds, HEAD is still valid, and a
    /// restore from HEAD should still succeed. This is `CommitSyncError::PreCommit`
    /// with the failure specifically at the `commit` step rather than `add`.
    ///
    /// Deliberately NOT a `.git/hooks/pre-commit` script: a machine (or CI image)
    /// with a global `core.hooksPath` override — common for commit-signing or
    /// linting setups — would silently ignore a repo-local hook, making that
    /// approach environment-dependent. Repo-local git CONFIG, by contrast, always
    /// applies: enabling commit signing and pointing at a signing key that does not
    /// exist makes `git commit` fail deterministically on any machine, with or
    /// without gpg installed (no gpg binary fails it too, just with a different
    /// message), and without touching global state.
    fn force_git_commit_to_fail(work: &tempfile::TempDir) {
        for args in [
            ["config", "commit.gpgsign", "true"],
            ["config", "user.signingkey", "nonexistent-bogus-key-id"],
        ] {
            std::process::Command::new("git")
                .args(args)
                .current_dir(work.path())
                .output()
                .unwrap();
        }
    }

    /// Words that name how the server versions the knowledge base. None of them
    /// may reach an MCP tool caller — results, errors or descriptions. Matched
    /// against whole lowercase alphanumeric tokens (`commit` also as a prefix, for
    /// `commits`/`committed`), so `github` or a word like `pushes` in a document's
    /// own diff body is not a false positive unless it really is the token.
    fn git_terms_in(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|t| {
                matches!(
                    *t,
                    "git" | "sha" | "push" | "pushed" | "rebase" | "rebased" | "remote"
                ) || t.starts_with("commit")
            })
            .map(str::to_string)
            .collect()
    }

    fn assert_git_free(what: &str, text: &str) {
        let hits = git_terms_in(text);
        assert!(hits.is_empty(), "{what} leaks {hits:?}: {text}");
    }

    /// A plain success: none of the removed versioning fields, and git-free text.
    fn assert_plain_success(result: &CallToolResult) {
        let structured = result
            .structured_content
            .as_ref()
            .expect("a write result carries structured_content");
        for key in ["outcome", "sha", "rebased_paths", "sync_failure_cause"] {
            assert!(structured.get(key).is_none(), "{key} leaked: {structured}");
        }
        assert_git_free("structured_content", &structured.to_string());
        assert_git_free("result text", &format!("{:?}", result.content));
    }

    /// A rolled-back write: "not saved, nothing changed, try again", no data
    /// payload, nothing about the cause.
    fn assert_not_saved(err: &McpError) {
        assert!(
            err.message.contains("could not be saved") && err.message.contains("try again"),
            "{}",
            err.message
        );
        assert!(err.data.is_none(), "{err:?}");
        assert_git_free("error", &err.message);
    }

    /// A write whose rollback also failed: told not to trust the state.
    fn assert_not_saved_unverified(err: &McpError) {
        assert!(
            err.message.contains("could not be saved")
                && err.message.contains("could not confirm")
                && err.message.contains("operator"),
            "{}",
            err.message
        );
        assert!(err.data.is_none(), "{err:?}");
        assert_git_free("error", &err.message);
    }

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
        let original =
            "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
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
        let original =
            "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
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
        let original =
            "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
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
        let original =
            "---\ntitle: Old Home\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Body\n";
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
        let original =
            "---\ntitle: Old\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Old body\n";
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
    async fn edit_document_move_onto_existing_destination_reports_the_destination_as_the_collision()
    {
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
            let err =
                result.expect_err(&format!("{field} set alongside documents must be rejected"));
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
                    "---\ntitle: A\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# A\n"
                        .to_string(),
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
                    "---\ntitle: B\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# B\n"
                        .to_string(),
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
                    "---\ntitle: Ok\ndescription: d\ntype: guide\ntags: [t]\n---\n\n# Ok\n"
                        .to_string(),
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
            }))
            .await;
        let err = result.expect_err("a batch entry with no edit mode must be rejected");
        assert!(
            err.message.contains("docs/no-edit-mode.md"),
            "error should name the offending entry: {}",
            err.message
        );

        // Nothing should have been written for the OTHER (valid) document either
        // — this is a pre-flight parse failure in mcp.rs, before
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

    /// `git_status`, filtered to drop the SQLite state-DB files (`state.db` plus its
    /// `-shm`/`-wal` siblings). `update_schema`'s casualty check
    /// (`documents_broken_by`) opens the metadata index on first use, which lazily
    /// creates those files under `work` as an ordinary, expected side effect of
    /// running the tool at all — unrelated to whether a `commit_and_sync` rollback
    /// left the WRITE ITSELF clean. `delete_document` does the same (#229: it always
    /// opens the state DB for the inbound-link check, unlike a MOVE's conditional
    /// open), so its tests need this filtered helper too. `create_document`/plain
    /// `edit_document` (no `new_path`) never touch the metadata index at all, so
    /// their equivalent tests can keep the plain, exact `git_status` assertion.
    fn git_status_ignoring_state_db(work: &tempfile::TempDir) -> String {
        git_status(work)
            .lines()
            .filter(|line| {
                let path = line.split_whitespace().last().unwrap_or("");
                !path.starts_with("state.db")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

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

    fn add_values_params(path: &str, field: &str, values: &[&str]) -> UpdateSchemaParams {
        UpdateSchemaParams {
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
        let parsed: crate::schema::SchemaFile = serde_yaml_ng::from_str(&written).unwrap();
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

    fn payload_search_result(
        file_path: &str,
        title: &str,
        score: f32,
    ) -> crate::qdrant::SearchResult {
        let mut payload = HashMap::new();
        payload.insert("file_path".to_string(), serde_json::json!(file_path));
        payload.insert("title".to_string(), serde_json::json!(title));
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
        result
            .payload
            .insert("domain".to_string(), serde_json::json!("notes"));
        result
            .payload
            .insert("tags".to_string(), serde_json::json!([]));
        result
            .payload
            .insert("type".to_string(), serde_json::json!(""));
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
        let structured = build_grouped_search_payload(&documents, None, false, false);
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
        let structured = build_grouped_search_payload(&[], None, false, false);
        assert_eq!(
            structured,
            serde_json::json!({"returned": 0, "documents": []})
        );
    }

    #[test]
    fn build_grouped_search_payload_truncation_flags_appear_only_when_set() {
        let documents = vec![payload_grouped_document("notes/a.md", "A", 0.9)];
        let structured = build_grouped_search_payload(&documents, None, true, true);
        assert_eq!(structured["path_prefix_truncated"], serde_json::json!(true));
        assert_eq!(structured["offset_truncated"], serde_json::json!(true));
        let plain = build_grouped_search_payload(&documents, None, false, false);
        assert!(plain.get("path_prefix_truncated").is_none(), "{plain}");
        assert!(plain.get("offset_truncated").is_none(), "{plain}");
    }

    /// A document row's `frontmatter` drops what the row already promotes
    /// (`title`, `description`) and what the path derives (`domain`), keeping
    /// `domain` only when the caller asked for it by name; `mtime` is not on a
    /// grouped row at all.
    #[test]
    fn document_rows_drop_promoted_and_derived_frontmatter_keys() {
        let mut doc = payload_grouped_document("notes/a.md", "A", 0.9);
        doc.summary.description = Some("About A".to_string());
        doc.summary.frontmatter = serde_json::json!({
            "title": "A", "description": "About A", "domain": "notes", "tags": ["x"],
        });
        let structured =
            build_grouped_search_payload(std::slice::from_ref(&doc), None, false, false);
        let row = &structured["documents"][0];
        assert_eq!(row["frontmatter"], serde_json::json!({"tags": ["x"]}));
        assert_eq!(row["description"], "About A");
        assert!(row.get("mtime").is_none(), "{row}");

        let fields = vec!["domain".to_string()];
        let asked = build_grouped_search_payload(&[doc], Some(&fields), false, false);
        assert_eq!(asked["documents"][0]["frontmatter"]["domain"], "notes");

        let mut bare = payload_grouped_document("notes/b.md", "B", 0.5);
        bare.summary.frontmatter = serde_json::json!({"title": "B"});
        let structured = build_grouped_search_payload(&[bare], None, false, false);
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
}
