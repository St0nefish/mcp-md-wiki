use super::schema_tools::*;
use super::*;
use std::path::Path;

use rmcp::{
    ErrorData as McpError, handler::server::wrapper::Parameters, schemars, tool, tool_router,
};
use tracing::{debug, error};

use crate::{
    config::{Granularity, ResolvedConfig},
    document_fields,
    qdrant::IndexKind,
    retrieval::{self},
    schema::SchemaCache,
    state::{DocumentQuery, FieldFilter, OrderBy},
};

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
pub(super) fn parse_search_filters_input(
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
pub(super) fn filters_shape_error(value: &serde_json::Value) -> String {
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
    // carries none of its own. The last branch is a correlated list filter's
    // value (#349, a key ending in `[]`): sub-field names mapped to the same
    // condition shapes, inlined rather than referenced so the schema stays
    // `$ref`-free (`tool_schema`, #288). A range bound may be a date string.
    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let scalar = serde_json::json!({ "type": ["string", "number", "boolean"] });
        let bound = serde_json::json!({ "type": ["number", "string"] });
        let condition = serde_json::json!({
            "anyOf": [
                scalar,
                { "type": "array", "items": scalar },
                {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "any_of": { "type": "array", "items": scalar },
                        "all_of": { "type": "array", "items": scalar },
                        "gte": bound,
                        "lte": bound,
                        "gt": bound,
                        "lt": bound
                    }
                }
            ]
        });
        let mut branches = condition["anyOf"].as_array().cloned().unwrap_or_default();
        branches.push(serde_json::json!({
            "type": "object",
            "additionalProperties": condition
        }));
        schemars::json_schema!({
            "type": "object",
            "additionalProperties": { "anyOf": branches }
        })
    }
}

/// The filter operator names, which are never a sub-field of a correlated filter.
pub(super) const FILTER_OPERATORS: [&str; 6] = ["any_of", "all_of", "gte", "lte", "gt", "lt"];

/// Parse a correlated list filter (#349): `field` is `list[]` and `raw` an object
/// whose keys are sub-fields of the list's elements (`role`, `stats.level`) and
/// whose values are ordinary filter conditions, all of which must hold within one
/// element. One level only: the list may not itself sit inside a list, and a
/// condition may not be another correlated filter. At most [`MAX_LIST_FILTERS`]
/// row tests (each `all_of` value counts as one), so one call cannot build an
/// unbounded join.
pub(super) fn parse_elem_filter(
    field: &str,
    list: &str,
    raw: &serde_json::Value,
) -> Result<FieldFilter, String> {
    let example = format!("{{\"{field}\": {{\"role\": \"guard\", \"level\": {{\"gte\": 3}}}}}}");
    let Some(map) = raw.as_object() else {
        return Err(format!(
            "filter '{field}': a key ending in `[]` is a correlated list filter and takes an \
             object of conditions on the list's sub-fields, e.g. {example}"
        ));
    };
    let path = crate::schema::FieldPath::parse(list);
    let plain = path
        .steps()
        .iter()
        .all(|step| matches!(step, crate::schema::PathStep::Field(name) if !name.is_empty()));
    if list.is_empty() || !plain {
        return Err(format!(
            "filter '{field}': a correlated list filter names a list outside any other list, \
             e.g. `npcs[]` or `campaign.npcs[]`"
        ));
    }
    if map.is_empty() {
        return Err(format!(
            "filter '{field}': give at least one sub-field condition, e.g. {example}"
        ));
    }
    let mut conditions = Vec::with_capacity(map.len());
    let mut row_tests = 0usize;
    for (sub, value) in map {
        if sub.is_empty() || sub.len() > MAX_FILTER_STR_LEN {
            return Err(format!(
                "filter '{field}': sub-field names must be 1-{MAX_FILTER_STR_LEN} characters"
            ));
        }
        if FILTER_OPERATORS.contains(&sub.as_str()) {
            return Err(format!(
                "filter '{field}': '{sub}' is an operator, not a sub-field; a correlated list \
                 filter maps sub-fields to conditions, e.g. {example}"
            ));
        }
        if sub.ends_with("[]") {
            return Err(format!(
                "filter '{field}': correlated list filters do not nest ('{sub}'); name the inner \
                 list's sub-fields as conditions, e.g. '{sub}.name', which match anywhere in \
                 the same element"
            ));
        }
        let full = format!("{field}.{sub}");
        let condition = parse_field_filter(&full, value)?;
        row_tests += match &condition {
            FieldFilter::AllOf(values) => values.len(),
            _ => 1,
        };
        conditions.push((full, condition));
    }
    if row_tests > MAX_LIST_FILTERS {
        return Err(format!(
            "filter '{field}': too many conditions ({row_tests}, max {MAX_LIST_FILTERS}; each \
             all_of value counts as one)"
        ));
    }
    conditions.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(FieldFilter::Elem(conditions))
}

/// Translate one JSON filter value into a typed [`FieldFilter`].
///
/// Accepts a scalar for equality, an array for any-of, or an object carrying
/// `any_of` / `all_of` / `gte` / `lte` / `gt` / `lt`. A range bound may be a number
/// or a date/RFC 3339 timestamp string, compared as epoch seconds the way a date
/// field's `value_num` is stored (#349). A `field` ending in `[]` is a correlated
/// list filter ([`parse_elem_filter`]).
pub(crate) fn parse_field_filter(
    field: &str,
    raw: &serde_json::Value,
) -> Result<FieldFilter, String> {
    use serde_json::Value;

    if let Some(list) = field.strip_suffix("[]") {
        return parse_elem_filter(field, list, raw);
    }

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
                    Some(Value::String(s)) if document_fields::epoch_seconds(s).is_some() => {
                        Ok(document_fields::epoch_seconds(s))
                    }
                    Some(other) => Err(format!(
                        "filter '{}': '{}' must be a number, or a date (YYYY-MM-DD) or RFC 3339 \
                         timestamp, got {}",
                        field, key, other
                    )),
                }
            };

            let known = FILTER_OPERATORS;
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
pub(super) const MAX_FILTER_OPTIONS_LISTED: usize = 30;

/// The first [`MAX_FILTER_OPTIONS_LISTED`] of `items`.
pub(super) fn capped_vec(items: &[String]) -> Vec<&String> {
    items.iter().take(MAX_FILTER_OPTIONS_LISTED).collect()
}

/// `items` joined for an error message, capped at
/// [`MAX_FILTER_OPTIONS_LISTED`] with a `(+N more)` tail.
pub(super) fn capped_list(items: &[String]) -> String {
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
/// `description`, the fields ingest derives (the root schema's `derived:` block,
/// #345), and the configured indexed fields.
pub(super) fn builtin_filter_field(
    field: &str,
    config: &ResolvedConfig,
    schemas: &SchemaCache,
) -> bool {
    matches!(field, "title" | "description")
        || schemas.root().is_derived(field)
        || config.effective_indexed_fields().iter().any(|f| f == field)
}

/// Parse a `search` call's raw `filters` map into the typed representation shared by
/// both backends: SQLite (`state::StateDb::push_where`, enumeration mode) and Qdrant
/// (`qdrant::lower_field_filters`, query mode).
pub(super) fn parse_filters(
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
pub(crate) fn validate_path_prefix(path_prefix: &Option<String>) -> Result<(), McpError> {
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
pub(super) fn validate_fields_count(fields: &Option<Vec<String>>) -> Result<(), McpError> {
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
pub(super) fn build_document_query(params: &SearchParams) -> Result<DocumentQuery, McpError> {
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
pub(super) fn build_query_conditions(
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

    // A correlated list filter (#350: a nested condition) is checked sub-field by
    // sub-field: `npcs[].role` must be indexed like any other field.
    let mut named: Vec<&String> = Vec::with_capacity(filters.len());
    for (field, filter) in &filters {
        match filter {
            FieldFilter::Elem(subs) => named.extend(subs.iter().map(|(sub, _)| sub)),
            _ => named.push(field),
        }
    }
    if let Some(field) = named.into_iter().find(|f| !indexed.contains_key(*f)) {
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
pub(super) fn heading_prefix_condition(
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
pub(super) const HEADING_METADATA_OFF_HEADING_PREFIX: &str = "heading_prefix is unavailable: this server does not have heading metadata enabled. \
     Search without heading_prefix, or ask the server's operator to enable heading metadata";

/// Rejection for `section` granularity while `chunking.heading_metadata` is
/// off. Shared by `granularity_disabled_error` and `search_sections` so a
/// caller sees one explanation for one cause. Caller-facing in the same way as
/// [`HEADING_METADATA_OFF_HEADING_PREFIX`].
pub(super) const HEADING_METADATA_OFF_SECTION: &str = "section granularity is unavailable: this server does not have heading metadata \
     enabled. Use another granularity, or ask the server's operator to enable heading metadata";

/// The tail [`granularity_disabled_error`] appends when `section` is missing
/// only because heading metadata is off — [`HEADING_METADATA_OFF_SECTION`]'s
/// explanation, worded to follow "granularity 'section' is not enabled on
/// this server" without repeating it.
pub(super) const HEADING_METADATA_OFF_SECTION_REASON: &str = ", because this server does not have heading \
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
pub(super) fn heading_results_indexing_note() -> Option<&'static str> {
    crate::status::INDEX_STATUS.is_bulk_indexing().then_some(
        "Note: an indexing run is in progress; documents it has not reached yet may lack \
         heading metadata, so these results may be incomplete.",
    )
}

/// Mark a finished `search` response `indexing_in_progress: true` when
/// [`heading_results_indexing_note`] applies.
pub(super) fn annotate_heading_results(response: &mut serde_json::Value, note: Option<&str>) {
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
pub(super) fn parse_granularity(
    raw: &str,
    effective: &[Granularity],
) -> Result<Granularity, McpError> {
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
pub(super) fn default_granularity(query_present: bool) -> Granularity {
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
pub(super) fn granularity_supports_mode(g: Granularity, query_present: bool) -> bool {
    query_present || g.supports_no_query()
}

/// `effective`, as a comma-separated list for error messages.
pub(super) fn join_granularities(effective: &[Granularity]) -> String {
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
pub(super) fn fields_rejection(granularity: Granularity, effective: &[Granularity]) -> McpError {
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
pub(super) fn explain_rejection(granularity: Granularity, effective: &[Granularity]) -> McpError {
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
pub(super) fn granularity_disabled_error(
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
pub(crate) fn resolve_search_granularity(
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
pub(super) fn query_is_present(query: &Option<String>) -> bool {
    query.as_deref().is_some_and(|q| !q.trim().is_empty())
}

pub(crate) fn validate_search_params(params: &SearchParams) -> Result<(), McpError> {
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

/// Parameters for `changes_since`.
#[derive(Debug, Default, serde::Deserialize, schemars::JsonSchema)]
pub struct ChangesSinceParams {
    /// A timestamp (RFC 3339, `2026-03-01`, or `2026-03-01T12:30Z`; UTC) or the
    /// `head` an earlier call returned. Required unless `cursor` is given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub since: Option<String>,

    /// Only documents under this folder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_prefix: Option<String>,

    /// Changes per page (default 50, max 200), ordered by path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,

    /// The `next_cursor` of a truncated page; continues that same range.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
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
    /// all-of or a range (`{"planning.prep_minutes": {"lt": 30}}`; a date field's
    /// bound may be `"2024-01-01"`). A list's object fields are `npcs[].role`; a
    /// key ending in `[]` needs every condition in one element
    /// (`{"npcs[]": {"role": "guard", "level": {"gte": 3}}}`).
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

/// A row's frontmatter as `search` reports it: without the keys already promoted
/// to the row itself (`title`, `description`) or derived from the path (the root
/// schema's `derived:` fields, #345 — each kept only when the caller named it in
/// `fields`), since the row's `file_path` already says what they hold. `None` when
/// nothing is left.
pub(super) fn row_frontmatter(
    frontmatter: &serde_json::Value,
    fields: Option<&[String]>,
    derived: &[String],
) -> Option<serde_json::Value> {
    let mut map = frontmatter.as_object()?.clone();
    map.remove("title");
    map.remove("description");
    for name in derived {
        if !fields.is_some_and(|f| f.contains(name)) {
            map.remove(name);
        }
    }
    (!map.is_empty()).then_some(serde_json::Value::Object(map))
}

/// The `search_grouped`/`search_enumerate` row for one document: `file_path`,
/// plus `title`, `description` and the remaining `frontmatter` when present.
pub(super) fn document_row(
    summary: &crate::state::DocumentSummary,
    fields: Option<&[String]>,
    derived: &[String],
) -> serde_json::Map<String, serde_json::Value> {
    let mut row = serde_json::Map::new();
    row.insert("file_path".into(), serde_json::json!(summary.file_path));
    if let Some(title) = &summary.title {
        row.insert("title".into(), serde_json::json!(title));
    }
    if let Some(description) = &summary.description {
        row.insert("description".into(), serde_json::json!(description));
    }
    if let Some(frontmatter) = row_frontmatter(&summary.frontmatter, fields, derived) {
        row.insert("frontmatter".into(), frontmatter);
    }
    row
}

/// The envelope keys every query-mode `search` response shares: `returned`, and
/// `path_prefix_truncated`/`offset_truncated` only when set.
pub(super) fn search_envelope(
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
pub(super) fn build_chunk_search_payload(
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
                crate::qdrant::frontmatter_value(payload, "title")
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
        if let Some(t) = crate::qdrant::frontmatter_value(payload, "type")
            .filter(|v| v.as_str().is_some_and(|s| !s.is_empty()))
        {
            row.insert("type".into(), t.clone());
        }
        if let Some(tags) = crate::qdrant::frontmatter_value(payload, "tags")
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
pub(super) fn build_grouped_search_payload(
    documents: &[retrieval::GroupedDocument],
    fields: Option<&[String]>,
    derived: &[String],
    path_prefix_truncated: bool,
    offset_truncated: bool,
) -> serde_json::Value {
    let mut out = search_envelope(documents.len(), path_prefix_truncated, offset_truncated);
    out.insert(
        "documents".into(),
        documents
            .iter()
            .map(|d| {
                let mut row = document_row(&d.summary, fields, derived);
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
pub(super) fn build_section_search_payload(
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

#[tool_router(router = search_router, vis = "pub(crate)")]
impl KbSearchServer {
    #[tool(annotations(read_only_hint = true))]
    pub(super) async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let ctx = self.search_ctx();
        validate_search_params(&params)?;
        ctx.check_filter_vocabulary(&params).await?;

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
            (false, Granularity::Document) => ctx.search_enumerate(&params).await,
            (true, Granularity::Chunk) => ctx.search_chunks(&params).await,
            (true, Granularity::Document) => ctx.search_grouped(&params).await,
            (true, Granularity::Section) => ctx.search_sections(&params).await,
        }
    }

    #[tool(annotations(read_only_hint = true))]
    pub(super) async fn changes_since(
        &self,
        Parameters(params): Parameters<ChangesSinceParams>,
    ) -> Result<CallToolResult, McpError> {
        let data_path = self.canonical_data_path.to_string_lossy().into_owned();
        let indexing = self.config().indexing.clone();
        match retrieval::changes_since_json(
            false,
            &data_path,
            &indexing,
            params.since.as_deref(),
            params.path_prefix.as_deref(),
            params.limit,
            params.cursor.as_deref(),
        )
        .await
        {
            Ok(body) => Ok(CallToolResult::structured(body)),
            Err(retrieval::ChangesSinceError::Invalid(msg)) => {
                Err(McpError::invalid_params(msg, None))
            }
            Err(retrieval::ChangesSinceError::Git(e)) => {
                error!("changes_since failed: {e:#}");
                Err(McpError::internal_error(
                    "the change listing is unavailable".to_string(),
                    None,
                ))
            }
        }
    }
}
