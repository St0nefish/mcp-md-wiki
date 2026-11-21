use super::search::*;
use super::write_document::*;
use super::*;
use std::path::Path;

use rmcp::{
    ErrorData as McpError, handler::server::wrapper::Parameters, schemars, tool, tool_router,
};
use tracing::{error, warn};

use crate::mcp::write_document::RawWrite;
use crate::{
    git,
    schema::SchemaCache,
    state::{DocumentIndex, DocumentQuery},
    validate,
    write::{self},
};

/// Keys a `set_field` definition accepts, as the errors that reject a malformed one name
/// them: `schema::RawFieldDef`'s advertised properties, kept in one place so those errors
/// all say the same thing. The deprecated `extend` is still accepted but is not
/// advertised, so it is not listed here either.
pub(super) const FIELD_DEFINITION_KEYS: &str = "`type`, `required`, `indexed`, `values`, `default`, \
     `open`, `fields`, `items`, `exclude_values`, `removed`";

/// The keys of [`FIELD_DEFINITION_KEYS`] as names: what a `patch_field` op's
/// `reinherit` may list (#348). A test keeps the two in step.
pub(super) const FIELD_DEFINITION_KEY_NAMES: &[&str] = &[
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

/// Most operations one `update_schema` call's `ops` may carry (#348). Each op is
/// bounded on its own (`MAX_SCHEMA_VALUES`, `MAX_SCHEMA_DEFINITION_LEN`); this bounds
/// the batch, and with it the work done under the git lock.
pub(super) const MAX_SCHEMA_OPS: usize = 50;

// The operations `update_schema` performs (#348). A real enum, so `ops` advertises
// (and a client's own validation enforces) the closed set.
//
// `//` rather than `///` throughout: schemars would serve a doc comment as the
// type's description, and on the variants would turn them into a `oneOf` of described
// `const` branches; without them it emits a flat string `enum`, and the caller-facing
// sentence lives on `SchemaOpInput::operation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SchemaOperation {
    AddValues,
    RemoveValues,
    SetField,
    RemoveField,
    PatchField,
}

impl SchemaOperation {
    fn name(self) -> &'static str {
        match self {
            SchemaOperation::AddValues => "add_values",
            SchemaOperation::RemoveValues => "remove_values",
            SchemaOperation::SetField => "set_field",
            SchemaOperation::RemoveField => "remove_field",
            SchemaOperation::PatchField => "patch_field",
        }
    }
}

// One entry of `update_schema`'s `ops` (#348). `//`, not `///`: a doc comment here
// would be served as the entry's description.
#[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SchemaOpInput {
    /// add_values/remove_values: change the permitted values (removing an inherited
    /// value excludes it here); set_field: declare or replace a definition;
    /// patch_field: change only the keys given; remove_field: drop this
    /// directory's declaration.
    pub operation: SchemaOperation,

    /// Field path (`planning.method`; `npcs[]` is a list's element, `npcs[].name` a
    /// field of it); add_values, set_field and patch_field create missing parents.
    pub field: String,

    /// For add_values/remove_values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<String>>,

    /// For set_field: the whole definition. For patch_field: only the keys to change
    /// (`fields` merge per child); to make a key inherit again, name it in
    /// `reinherit` (a literal `null` for the key also works).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub definition: Option<FieldPatchInput>,

    /// For patch_field: definition keys to drop from this directory's declaration so
    /// they inherit again (`required`, `values`, ...). A key may not also be set in
    /// `definition`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reinherit: Option<Vec<String>>,
}

/// A field definition (or `patch_field` merge patch) as an `ops` entry delivers it:
/// kept as the JSON object it arrived as, since a patch's `null`s mean "inherit
/// again" and a [`crate::schema::RawFieldDef`] cannot hold them. Advertised as the
/// same `RawFieldDef` object [`FieldDefinitionInput`] advertises, and, like it,
/// also accepted as a JSON string containing that object.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldPatchInput(pub serde_json::Map<String, serde_json::Value>);

impl<'de> serde::Deserialize<'de> for FieldPatchInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde_json::Value;
        let value = Value::deserialize(deserializer)?;
        let object = match value {
            Value::Object(map) => map,
            Value::String(s) => match serde_json::from_str::<Value>(&s) {
                Ok(Value::Object(map)) => map,
                Ok(other) => return Err(serde::de::Error::custom(definition_shape_error(&other))),
                Err(e) => {
                    return Err(serde::de::Error::custom(format!(
                        "field definition must be a JSON object with keys \
                         {FIELD_DEFINITION_KEYS}. A JSON string containing that object \
                         is also accepted, but this string is not valid JSON: {e}"
                    )));
                }
            },
            other => return Err(serde::de::Error::custom(definition_shape_error(&other))),
        };
        Ok(FieldPatchInput(object))
    }
}

impl schemars::JsonSchema for FieldPatchInput {
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
pub(super) fn parse_field_definition(
    value: serde_json::Value,
) -> Result<crate::schema::RawFieldDef, String> {
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
pub(super) fn definition_shape_error(value: &serde_json::Value) -> String {
    format!(
        "field definition must be a JSON object with keys {FIELD_DEFINITION_KEYS}, got {}",
        json_value_kind(value)
    )
}

/// Describe a JSON value's kind in a few words, for error messages.
pub(super) fn json_value_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// The schema edits an `update_schema` call asks for, in order: its `ops`, or the
/// deprecated flat `operation`/`field`/`values`/`definition` as a one-element batch
/// (#348). The two forms are mutually exclusive.
pub(super) fn build_schema_edits(
    params: &UpdateSchemaParams,
) -> Result<Vec<crate::schema::SchemaEdit>, McpError> {
    let invalid = |msg: String| McpError::invalid_params(msg, None);
    let legacy = !params.operation.trim().is_empty()
        || !params.field.is_empty()
        || params.values.is_some()
        || params.definition.is_some();
    let Some(ops) = &params.ops else {
        if !legacy {
            return Err(invalid(
                "update_schema needs `ops`: a list of {operation, field, values?, \
                 definition?} objects"
                    .into(),
            ));
        }
        return build_schema_edit(params).map(|edit| vec![edit]);
    };
    if legacy {
        return Err(invalid(
            "pass either `ops` or the deprecated `operation`/`field`/`values`/`definition`, \
             not both"
                .into(),
        ));
    }
    if ops.is_empty() {
        return Err(invalid("`ops` must hold at least one operation".into()));
    }
    if ops.len() > MAX_SCHEMA_OPS {
        return Err(invalid(format!(
            "too many operations: {} (max {MAX_SCHEMA_OPS}); split the change across calls",
            ops.len()
        )));
    }
    ops.iter()
        .enumerate()
        .map(|(i, op)| {
            build_schema_op(
                op.operation.name(),
                &op.field,
                op.values.as_deref(),
                op.definition.as_ref().map(|d| &d.0),
                op.reinherit.as_deref(),
            )
            .map_err(|e| {
                if ops.len() > 1 {
                    invalid(format!("ops[{i}]: {}", e.message))
                } else {
                    e
                }
            })
        })
        .collect()
}

/// Turn the deprecated flat tool parameters into a typed schema edit.
pub(super) fn build_schema_edit(
    params: &UpdateSchemaParams,
) -> Result<crate::schema::SchemaEdit, McpError> {
    // The flat `definition` is parsed as a whole field definition, which cannot hold
    // a patch's `null`s, so `patch_field` exists only in `ops` (#348).
    if params.operation.trim().eq_ignore_ascii_case("patch_field") {
        return Err(McpError::invalid_params(
            "patch_field is only available through `ops`".to_string(),
            None,
        ));
    }
    let definition = params
        .definition
        .as_ref()
        .map(|d| match serde_json::to_value(&d.0) {
            Ok(serde_json::Value::Object(map)) => Ok(map),
            _ => Err(McpError::internal_error(
                "could not re-read the field definition".to_string(),
                None,
            )),
        })
        .transpose()?;
    build_schema_op(
        &params.operation,
        &params.field,
        params.values.as_deref(),
        definition.as_ref(),
        None,
    )
}

/// One operation's arguments, however they arrived, bounded and turned into a typed
/// schema edit.
pub(super) fn build_schema_op(
    operation: &str,
    field: &str,
    values: Option<&[String]>,
    definition: Option<&serde_json::Map<String, serde_json::Value>>,
    reinherit: Option<&[String]>,
) -> Result<crate::schema::SchemaEdit, McpError> {
    use crate::schema::SchemaEdit;
    let invalid = |msg: String| McpError::invalid_params(msg, None);
    let operation_name = operation.trim().to_ascii_lowercase();

    // `reinherit` is the way to send a patch's "inherit again" without a JSON `null`,
    // which the advertised (compacted, #288) definition schema cannot carry.
    let reinherit = reinherit.unwrap_or_default();
    if !reinherit.is_empty() {
        if operation_name != "patch_field" {
            return Err(invalid(format!(
                "`reinherit` applies only to patch_field, not '{operation_name}'"
            )));
        }
        if let Some(unknown) = reinherit
            .iter()
            .find(|k| !FIELD_DEFINITION_KEY_NAMES.contains(&k.as_str()))
        {
            return Err(invalid(format!(
                "`reinherit` names unknown key '{unknown}': expected one of \
                 {FIELD_DEFINITION_KEYS}"
            )));
        }
        if let Some(both) = reinherit.iter().find(|k| {
            definition
                .and_then(|d| d.get(k.as_str()))
                .is_some_and(|v| !v.is_null())
        }) {
            return Err(invalid(format!(
                "'{both}' is both set in `definition` and listed in `reinherit`; pick one"
            )));
        }
    }

    // A schema file is committed, pushed, and re-parsed on every cache build, so an
    // oversized one is a durable cost rather than a transient one. Bound the inputs
    // here, mirroring the content cap the document write tools enforce.
    if field.len() > MAX_FILTER_STR_LEN {
        return Err(invalid(format!(
            "field name too long: {} chars (max {})",
            field.len(),
            MAX_FILTER_STR_LEN
        )));
    }
    // `field` is a path (`a.b`, `a[]`, `a[].b`) into nested containers and list
    // elements. Each step becomes two levels of YAML nesting, so cap the depth well
    // under the parser's recursion limit rather than let the round-trip check fail
    // with a 500.
    let path = crate::schema::FieldPath::parse(field);
    if path.depth() > MAX_SCHEMA_PATH_SEGMENTS {
        return Err(invalid(format!(
            "field path too deep: more than {MAX_SCHEMA_PATH_SEGMENTS} steps (each \
             dot-separated name and each `[]` counts as one)"
        )));
    }
    // An empty or whitespace-padded name ("a..b", "a.", "a. b", "[]") would create a
    // key no frontmatter field can match.
    if path.steps().iter().any(|step| match step {
        crate::schema::PathStep::Field(name) => name.is_empty() || name != name.trim(),
        crate::schema::PathStep::Items => false,
    }) {
        return Err(invalid(format!(
            "invalid field path '{}': every dot-separated segment must be non-empty with no surrounding whitespace",
            field
        )));
    }
    if let Some(values) = values {
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
    if let Some(definition) = definition {
        // Measured on the parsed-and-reserialized form rather than whatever bytes the
        // client happened to send: that's what actually gets committed to the schema
        // file (via `SchemaFile::to_yaml`), and it makes the cap apply identically
        // whether the definition arrived as an object or as the string fallback.
        let size = serde_json::to_string(definition)
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
        values
            .map(<[String]>::to_vec)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid(format!("'{operation}' requires a non-empty values list")))
    };
    let field = field.to_string();

    match operation_name.as_str() {
        "add_values" => Ok(SchemaEdit::AddValues {
            field,
            values: values()?,
        }),
        "remove_values" => Ok(SchemaEdit::RemoveValues {
            field,
            values: values()?,
        }),
        "set_field" => {
            let definition =
                definition.ok_or_else(|| invalid("'set_field' requires a definition".into()))?;
            let definition = parse_field_definition(serde_json::Value::Object(definition.clone()))
                .map_err(invalid)?;
            Ok(SchemaEdit::SetField {
                field,
                definition: Box::new(definition),
            })
        }
        "patch_field" => {
            let mut patch = definition.cloned().unwrap_or_default();
            for key in reinherit {
                patch.insert(key.clone(), serde_json::Value::Null);
            }
            if patch.is_empty() {
                return Err(invalid(
                    "'patch_field' requires a definition holding the keys to change, or \
                     `reinherit` naming keys to inherit again"
                        .into(),
                ));
            }
            Ok(SchemaEdit::PatchField { field, patch })
        }
        "remove_field" => Ok(SchemaEdit::RemoveField { field }),
        other => Err(invalid(format!(
            "unknown operation '{other}': expected add_values, remove_values, set_field, \
             remove_field, or patch_field"
        ))),
    }
}

/// Resolve a possibly-partial scope reference to exactly one directory.
///
/// Mirrors `get_document`'s contract: an exact match wins, several matches are an
/// explicit ambiguity error rather than a guess, and none is a not-found error naming
/// what does exist.
pub(super) fn resolve_scope_reference(
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
pub(super) fn sanitize_reflected_value(value: &serde_json::Value) -> serde_json::Value {
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
pub(super) fn insert_casualties(
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
/// default), `items` — a list's element definition, in the same shape with its own
/// `fields` — when declared (#346), and `declared_in` — the scope directory that
/// last set it (`/` for the root). The field's own children are not included: each
/// is reported as a field of its own.
pub(super) fn field_def_json(
    def: &crate::schema::FieldDef,
    origin: Option<&str>,
) -> serde_json::Value {
    let mut entry = field_attributes_json(def);
    if let Some(origin) = origin {
        entry.insert(
            "declared_in".into(),
            serde_json::json!(crate::server::sanitize_facet_value(origin)),
        );
    }
    serde_json::Value::Object(entry)
}

/// An element definition under `items`, for [`field_def_json`]: its attributes,
/// its own `items`, and its child fields under `fields` (they have no entry of their
/// own to be reported as).
pub(super) fn item_def_json(def: &crate::schema::FieldDef) -> serde_json::Value {
    let mut entry = field_attributes_json(def);
    let children: serde_json::Map<String, serde_json::Value> = def
        .fields
        .iter()
        .take(MAX_REPORTED_FIELDS)
        .map(|(name, child)| {
            (
                crate::server::sanitize_facet_value(name),
                item_def_json(child),
            )
        })
        .collect();
    if !children.is_empty() {
        entry.insert("fields".into(), serde_json::Value::Object(children));
    }
    serde_json::Value::Object(entry)
}

/// The attributes [`field_def_json`] and [`item_def_json`] share.
pub(super) fn field_attributes_json(
    def: &crate::schema::FieldDef,
) -> serde_json::Map<String, serde_json::Value> {
    let mut entry = serde_json::Map::new();
    if let Some(ty) = def.ty {
        entry.insert("type".into(), serde_json::json!(ty.name()));
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
    if let Some(items) = &def.items {
        entry.insert("items".into(), item_def_json(items));
    }
    entry
}

pub(super) fn render_casualties(casualties: &[serde_json::Value]) -> String {
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
pub(super) fn capped_casualties(
    casualties: &[serde_json::Value],
) -> (Vec<serde_json::Value>, usize, bool) {
    let total = casualties.len();
    let capped = casualties
        .iter()
        .take(MAX_REPORTED_CASUALTIES)
        .cloned()
        .collect();
    (capped, total, total > MAX_REPORTED_CASUALTIES)
}

/// The one commit an `update_schema` call makes: `schema: <summary> in <scope>` for a
/// single op, as before batching, and `schema: <n> changes in <scope>: <summaries>`
/// for a batch (#348), the summaries cut short (on a char boundary) so a 50-op batch
/// still makes a one-line subject of bounded length.
pub(super) fn schema_commit_message(summaries: &[String], scope: &str) -> String {
    const MAX_SUMMARY_BYTES: usize = 400;
    match summaries {
        [one] => format!("schema: {one} in {scope}"),
        many => {
            let mut joined = many.join("; ");
            if joined.len() > MAX_SUMMARY_BYTES {
                let mut end = MAX_SUMMARY_BYTES;
                while !joined.is_char_boundary(end) {
                    end -= 1;
                }
                joined.truncate(end);
                joined.push('…');
            }
            format!("schema: {} changes in {scope}: {joined}", many.len())
        }
    }
}

/// `update_schema`'s `warnings`: `schema::lint_raw` over the schema file as it will be
/// written (#347), capped at [`MAX_REPORTED_CASUALTIES`] with a closing line counting
/// the rest, so a file full of findings cannot balloon the result. Empty when the file
/// is clean (the key is then omitted). Includes `schema::lint_cascade` against
/// `inherited`, what the scope's ancestors resolve to (#348).
pub(super) fn schema_lint_warnings(
    file: &crate::schema::SchemaFile,
    is_root: bool,
    inherited: &crate::schema::ResolvedSchema,
) -> Vec<String> {
    let mut warnings = crate::schema::lint_raw(file, is_root);
    warnings.extend(crate::schema::lint_cascade(inherited, file));
    if warnings.len() > MAX_REPORTED_CASUALTIES {
        let rest = warnings.len() - MAX_REPORTED_CASUALTIES;
        warnings.truncate(MAX_REPORTED_CASUALTIES);
        warnings.push(format!("…and {rest} more"));
    }
    warnings
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
pub(super) fn capped_diff(diff: &str) -> (String, bool, usize) {
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

    /// Name each declaring directory once, in a top-level `declared_in` map of
    /// directory to field names, instead of on every field.
    // Opt-in (#348) so existing callers keep the per-field `declared_in` shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compact: Option<bool>,
}

/// Parameters for `update_schema`.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct UpdateSchemaParams {
    /// Directory to edit; omit for the KB root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,

    /// The changes, applied in order, all or nothing: if any fails, none is made.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ops: Option<Vec<SchemaOpInput>>,

    /// Deprecated: use `ops`. One operation.
    // A string with an advertised `enum`, not a Rust enum, so parsing stays as
    // tolerant as `build_schema_edit` already is. Empty means absent: the flat
    // parameters are a one-release shim (#348) for a one-element `ops`, mutually
    // exclusive with it.
    #[serde(default)]
    #[schemars(extend("enum" = ["add_values", "remove_values", "set_field", "remove_field"]))]
    pub operation: String,

    /// Deprecated: use `ops`.
    #[serde(default)]
    pub field: String,

    /// Deprecated: use `ops`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<String>>,

    /// Deprecated: use `ops`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub definition: Option<FieldDefinitionInput>,

    /// Preview the change without writing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_run: Option<bool>,

    /// Apply even if existing documents would fail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force: Option<bool>,

    /// Confirms a deliberate root schema change, where the knowledge base guards
    /// root edits.
    // Enforced only when the root schema sets `root_edit: {require_ack: true}`
    // (#345); dry runs and `remove_values` never need it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acknowledge_root_change: Option<bool>,
}

/// `update_schema`'s refusal of an unacknowledged root edit under a root schema's
/// `root_edit: {require_ack: true}` (#345). Names `policy_doc` only when the
/// root schema sets one.
pub(super) fn root_edit_refusal(policy: &crate::schema::RootEditPolicy) -> String {
    let mut msg = String::from(
        "root schema changes are guarded: the root schema governs every document in \
         this knowledge base.",
    );
    msg.push_str(
        " Pass acknowledge_root_change=true only if this change is a deliberate design decision",
    );
    // `policy_doc` is bounded and free of control characters (`validate_self`).
    match &policy.policy_doc {
        Some(doc) => msg.push_str(&format!(
            " consistent with the policy in {doc} (a document in this knowledge base)."
        )),
        None => msg.push('.'),
    }
    msg
}

#[tool_router(router = schema_router, vis = "pub(crate)")]
impl KbSearchServer {
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

            // `rebased_paths` needs no handling: `mark_full` below covers every path.
            Err(git::CommitSyncError::PostCommit { sha, source, .. }) => {
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

    #[tool(annotations(read_only_hint = true))]
    pub(super) async fn get_schema(
        &self,
        Parameters(params): Parameters<GetSchemaParams>,
    ) -> Result<CallToolResult, McpError> {
        let raw = params.path.clone().unwrap_or_default();
        let rel = normalize_scope_path(&raw)?;
        // A restricted principal reads schema only inside its reach, under the exact
        // path it names: the partial-reference resolution below enumerates scope
        // names (in its candidate list), which is a view of the whole tree.
        let reach = self.principal().reach(Op::Read);
        let restricted = !matches!(reach, crate::policy::Reach::All);
        if restricted && !reach.covers(&rel.to_string_lossy()) {
            return Err(denied_error());
        }
        // Reads the shared, server-owned cache rather than walking the tree on every
        // call — this tool is the one the server's own instructions tell agents to
        // call before every write, so it needs to be cheap. See
        // `KbSearchServer::schema_cache`'s doc comment for how it stays current.
        let schemas = crate::schema::load_shared(&self.schema_cache);

        // A document path resolves via its parent; a directory resolves to itself. A
        // partial directory reference resolves like a partial document path does —
        // exact match wins, otherwise report the candidates rather than guessing.
        let rel = if restricted || raw.ends_with(".md") || rel.as_os_str().is_empty() {
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
        let compact = params.compact.unwrap_or(false);
        // `compact`: declaring directory -> the reported fields it declares.
        let mut declared_in: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::new();
        // (reported key, field path) of each open field `values_in_use` fills in.
        let mut open_fields: Vec<(String, String)> = Vec::new();
        let mut reported = serde_json::Map::new();
        let mut omitted = 0usize;
        for (field, def) in &schema.fields {
            // A derived field (the root schema's `derived:` block, #345) can be
            // declared — a deployment lists it in `indexed_fields` to filter on it —
            // but ingest sets it from the folder and a write that authors it is
            // refused, so it is never offered as one to write.
            if schema.is_derived(field) {
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
            if values_in_use && def.values.is_none() && def.ty.is_none_or(|ty| ty.kind().vocabulary)
            {
                open_fields.push((key.clone(), field.clone()));
            }
            let origin = schema.origin().get(field).map(String::as_str);
            if compact {
                if let Some(origin) = origin {
                    declared_in
                        .entry(crate::server::sanitize_facet_value(origin))
                        .or_default()
                        .push(key.clone());
                }
                reported.insert(key, field_def_json(def, None));
            } else {
                reported.insert(key, field_def_json(def, origin));
            }
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
            let excluded = declared.len() + schema.derived.len();
            if let Ok(fields) = index
                .fields_in_use(dir, (excluded + MAX_FILTER_OPTIONS_LISTED) as i64)
                .await
            {
                other_fields_in_use = fields
                    .into_iter()
                    .map(|(name, _)| name)
                    .filter(|name| !declared.contains(name) && !schema.is_derived(name))
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
        if !declared_in.is_empty() {
            structured["declared_in"] = serde_json::json!(declared_in);
        }
        if omitted > 0 {
            structured["omitted_fields"] = serde_json::json!(omitted);
        }
        if !other_fields_in_use.is_empty() {
            structured["other_fields_in_use"] = serde_json::json!(other_fields_in_use);
        }
        // So a client can check the memory tools' contract without a second call.
        if self.memory_enabled {
            structured["memory_contract_version"] =
                serde_json::json!(crate::memory::logic::CONTRACT_VERSION);
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
        if let Some(limit) = schema.related_limit {
            dedup.insert("related_limit".into(), serde_json::json!(limit));
        }
        if let Some(score) = schema.related_min_score {
            let shown = score
                .to_string()
                .parse::<f64>()
                .map_or(serde_json::json!(score), |v| serde_json::json!(v));
            dedup.insert("related_min_score".into(), shown);
        }
        if !dedup.is_empty() {
            structured["dedup"] = serde_json::Value::Object(dedup);
        }

        Ok(CallToolResult::structured(structured))
    }

    #[tool(annotations(destructive_hint = true, idempotent_hint = false))]
    pub(super) async fn update_schema(
        &self,
        Parameters(params): Parameters<UpdateSchemaParams>,
    ) -> Result<CallToolResult, McpError> {
        let invalid = |msg: String| McpError::invalid_params(msg, None);

        let requested = normalize_scope_path(params.path.as_deref().unwrap_or(""))?;
        // A schema governs everything under its directory, so changing one is a write
        // to that whole directory: it needs a write grant covering it. Checked before
        // the partial-reference resolution below (whose "matches N scopes" error lists
        // scope names) and again on what that resolution picked.
        self.authorize_write(&requested.to_string_lossy())?;
        let edits = build_schema_edits(&params)?;

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

        self.authorize_write(&rel_dir.to_string_lossy())?;

        // The memory area's schema is built into the binary; a file written there
        // would be refused at the next rebuild, so say so now.
        if self.memory_enabled {
            let root = crate::memory::root_dir(&self.memory_cfg);
            if crate::policy::prefix_covers(&root, &rel_dir.to_string_lossy()) {
                return Err(invalid(
                    "the memory area's schema is built into the server and cannot be changed"
                        .to_string(),
                ));
            }
        }

        // Root-scope mutations need an explicit opt-in when the root schema's
        // `root_edit: {require_ack: true}` asks for one (#345; see
        // `acknowledge_root_change`'s doc comment). `remove_values` is exempt —
        // shrinking the root vocabulary is never the change the guard is for, and the
        // casualty check below already guards it — and `dry_run` is exempt everywhere,
        // since it writes nothing. This must run before `file.apply()` below: the
        // point is to refuse before any edit is computed or written, not merely before
        // the commit. Judged per op (#348): a batch needs the acknowledgement when any
        // one of its ops would.
        let is_root = rel_dir.as_os_str().is_empty();
        let is_gated_op = edits
            .iter()
            .any(|edit| !matches!(edit, crate::schema::SchemaEdit::RemoveValues { .. }));
        let dry_run_requested = params.dry_run.unwrap_or(false);
        let acknowledged = params.acknowledge_root_change.unwrap_or(false);
        if is_root
            && is_gated_op
            && !dry_run_requested
            && !acknowledged
            && let Some(policy) = schemas.root().root_edit.as_ref().filter(|p| p.require_ack)
        {
            return Err(McpError::invalid_params(root_edit_refusal(policy), None));
        }

        // What this scope's ancestors alone resolve the field to, so `add_values` in a
        // child scope extends the inherited set instead of replacing it. Read from the
        // ancestors' files on disk, the same way the edit itself reads its own file:
        // under the lock below for the real edit, so a concurrent change to a parent
        // scope cannot leave the edit extending a stale inherited set. The root has no
        // ancestors: a root schema file replaces the config-derived root outright.
        //
        // The whole scope's inheritance, not one field's: a batch looks each op's
        // field up in it (#348). The ancestors are not what a batch edits, so one
        // resolution serves every op, and it also feeds the cascade lint.
        let scope = crate::schema::scope_label(&rel_dir);
        let inherited_schema = |schemas: &SchemaCache| -> crate::schema::ResolvedSchema {
            if rel_dir.as_os_str().is_empty() {
                return crate::schema::ResolvedSchema::default();
            }
            match schemas.resolve_from_disk(&rel_dir, None) {
                Ok(resolved) => resolved,
                // An ancestor's file no longer parses (it arrived through git): the
                // server still enforces its last valid rules, so extend those rather
                // than nothing, which would narrow the inherited set.
                Err(_) => {
                    let parent = rel_dir.parent().unwrap_or(std::path::Path::new(""));
                    schemas.resolve_for(&parent.join("_")).clone()
                }
            }
        };
        // Every op, in order, against one in-memory file: the first failure fails the
        // batch, and nothing has been written. Per-op summaries on success.
        let apply_all = |file: &mut crate::schema::SchemaFile,
                         inherited: &crate::schema::ResolvedSchema|
         -> Result<Vec<String>, String> {
            edits
                .iter()
                .enumerate()
                .map(|(i, edit)| {
                    file.apply_inheriting(edit, inherited.fields.lookup(edit.field()))
                        .map_err(|e| {
                            if edits.len() > 1 {
                                format!("ops[{i}] ({} '{}'): {e}", edit.operation(), edit.field())
                            } else {
                                e
                            }
                        })
                })
                .collect()
        };

        // Whether the edit applies to the file as it is before the lock — only to
        // tell an edit that a concurrent change made inapplicable from one that never
        // applied, should it fail under the lock.
        let applied_before_lock = schemas
            .raw_file_at(&rel_dir)
            .is_ok_and(|(mut file, _)| apply_all(&mut file, &inherited_schema(&schemas)).is_ok());

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
        let (summary, casualties, warnings) = loop {
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
            // The root-edit guard again, on the root file as it is under the lock: the
            // check above reads the shared cache, which a `root_edit:` block the sync
            // just pulled in reaches only after the worker's next rebuild (#345).
            if is_root
                && is_gated_op
                && !dry_run
                && !acknowledged
                && let Some(policy) = file.root_edit.as_ref().filter(|p| p.require_ack)
            {
                return Err(invalid(root_edit_refusal(policy)));
            }
            let inherited = inherited_schema(&schemas);
            let summaries = match apply_all(&mut file, &inherited) {
                Ok(summaries) => summaries,
                Err(_) if applied_before_lock || attempt > 1 => {
                    return Err(invalid(format!(
                        "The schema for '{scope}' was edited by someone else; re-read it with \
                         get_schema and try again."
                    )));
                }
                Err(e) => return Err(invalid(e)),
            };
            let summary = summaries.join("; ");

            // A self-contradictory definition parses fine but would be refused by the
            // next schema rebuild (and stop the server from starting) — after this call
            // has already reported success. Catch it here, where the caller can act.
            // Once per batch, on the file the whole batch produced.
            file.validate_self().map_err(invalid)?;
            // Lint warnings (#347, #348) for the file as it will be written: reported,
            // never a refusal.
            let warnings = schema_lint_warnings(&file, is_root, &inherited);

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
            crate::yaml::from_str::<crate::schema::SchemaFile>(&yaml).map_err(|e| {
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
                let definition_of = |field: &str| {
                    resolved.as_ref().and_then(|r| {
                        r.fields
                            .lookup(field)
                            .map(|def| field_def_json(def, def.declared_in.as_deref()))
                    })
                };
                let (would_invalidate, casualties_total, casualties_truncated) =
                    capped_casualties(&casualties);
                let mut response = serde_json::json!({
                    "dry_run": true,
                    "path": scope,
                    "summary": summary,
                });
                if let [edit] = edits.as_slice() {
                    // One op (the deprecated flat parameters always are one): its
                    // field and resolved definition at the top level, as before #348.
                    response["field"] = serde_json::json!(edit.field());
                    if let Some(definition) = definition_of(edit.field()) {
                        response["definition"] = definition;
                    }
                } else {
                    // A batch: each op's own summary, and each field as the whole
                    // batch leaves it (a field edited twice shows its final state).
                    let ops: Vec<serde_json::Value> = edits
                        .iter()
                        .zip(&summaries)
                        .map(|(edit, summary)| {
                            let mut op = serde_json::json!({
                                "operation": edit.operation(),
                                "field": edit.field(),
                                "summary": summary,
                            });
                            if let Some(definition) = definition_of(edit.field()) {
                                op["definition"] = definition;
                            }
                            op
                        })
                        .collect();
                    response["ops"] = serde_json::Value::Array(ops);
                }
                insert_casualties(
                    &mut response,
                    "would_invalidate",
                    would_invalidate,
                    casualties_total,
                    casualties_truncated,
                );
                if !warnings.is_empty() {
                    response["warnings"] = serde_json::json!(warnings);
                }
                return Ok(CallToolResult::structured(response));
            }

            // Always the canonical name. A directory still on the legacy name is
            // migrated by this write: the legacy file is removed in the same commit.
            let rel_file = rel_dir.join(crate::schema::SCHEMA_FILE_NAME);
            let rel_file_str = rel_file.to_string_lossy().to_string();
            let replaced = on_disk_name
                .filter(|name| *name != crate::schema::SCHEMA_FILE_NAME)
                .map(|name| rel_dir.join(name).to_string_lossy().to_string());
            let commit_message = schema_commit_message(&summaries, &scope);

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
                RawWrite::Committed => break (summary, casualties, warnings),
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
        let rebuild_config = self.config();
        let mut refused_rebuild: Option<crate::schema::SchemaBuildError> = None;
        match tokio::task::spawn_blocking(move || {
            SchemaCache::build_for_config(&rebuild_data_path, &rebuild_config)
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
        if !warnings.is_empty() {
            response["warnings"] = serde_json::json!(warnings);
        }
        Ok(CallToolResult::structured(response))
    }
}
