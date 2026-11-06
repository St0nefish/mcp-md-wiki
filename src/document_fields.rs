//! Projection of document frontmatter into flat, filterable rows.
//!
//! `documents.frontmatter` stores the faithful JSON record; this module derives the
//! `document_fields` inverted index from it. Nested objects flatten to dot-paths
//! (`planning.prep_minutes`), and the objects of a list to `[]` paths
//! (`npcs[].role`) keyed by element, so a filter never needs to know a document's
//! shape (#349).
//!
//! The projection is schema-driven where the schema speaks: a declared field's
//! [`crate::schema::FieldKind::numeric`] decides its `value_num`, and an undeclared
//! one keeps the JSON-type guess ([`numeric_value`]). It is rebuildable — see
//! `StateDb::reproject_all_fields` and `StateDb::ensure_projection_current`. Changing
//! the rules here requires no re-read of markdown and no re-embedding; bump
//! `state::PROJECTION_VERSION` so existing databases reproject on next open.

use crate::schema::{FieldDef, FieldType, MAX_FIELD_PATH_DEPTH, NumericValue, ResolvedSchema};
use serde_json::Value;
use std::collections::HashMap;
use tracing::warn;

/// Upper bound on projection rows from a single document. Checked inside every
/// expansion, so it bounds the work and not just the output.
const MAX_PROJECTED_ROWS: usize = 2_000;

/// Fields with a dedicated column on `documents`. They are prose, not filter-shaped,
/// and projecting them would bloat the index with never-matched rows.
const PROMOTED_FIELDS: [&str; 2] = ["title", "description"];

/// One row of the `document_fields` projection.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldRow {
    /// Field path, e.g. `planning.prep_minutes`, or `npcs[].role` for a field of a
    /// list's object elements.
    pub field: String,
    /// Which element of the outermost list of objects on the path this row came
    /// from (its 0-based position), so a correlated filter can require every
    /// condition to hold within one element. `0` for a row outside any list of
    /// objects; the scalars of a plain list (`tags`) all share their context's
    /// element — they are told apart by `value_text`, which is also in the key.
    pub elem: i64,
    /// Canonical text form, used for equality and any-of/all-of matching.
    pub value_text: String,
    /// Numeric form when the value is coercible, enabling range queries.
    pub value_num: Option<f64>,
}

/// Canonical text form of a scalar, used identically when writing rows and when
/// comparing a filter value. Keeping both sides on this one function is what makes
/// `{"planning.needs_recipe": false}` match a stored `needs_recipe: false`.
///
/// Returns `None` for values that are not scalars.
pub fn canonical_text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Bool(b) => Some(b.to_string()),
        // `Number::to_string` preserves representation rather than value: a YAML
        // `45.0` and a JSON `45` are both the number forty-five but stringify as
        // "45.0" and "45". Since the write side and the filter side both come through
        // here, a mismatch would silently drop documents from equality matches — so
        // integral values get one canonical spelling regardless of how they arrived.
        Value::Number(n) => Some(canonical_number(n)),
        _ => None,
    }
}

/// One spelling per numeric value, so equality matching cannot depend on whether the
/// author wrote `45` or `45.0`.
fn canonical_number(n: &serde_json::Number) -> String {
    if let Some(i) = n.as_i64() {
        return i.to_string();
    }
    if let Some(u) = n.as_u64() {
        return u.to_string();
    }
    match n.as_f64() {
        Some(f) if f.fract() != 0.0 || !f.is_finite() => f.to_string(),
        // Integral floats render as integers so `45.0` and `45` agree. The u64 arm
        // matters because a value above i64::MAX arriving as an integer takes the
        // as_u64 path above; without this it would arrive as a float and render in
        // exponential form, giving one value two spellings.
        Some(f) if f >= 0.0 && f <= u64::MAX as f64 => (f as u64).to_string(),
        // Both integer casts must be bounded on BOTH sides: Rust saturates float→int
        // casts, so an unbounded arm would collapse every value beyond the range onto
        // the same integer — giving distinct values one spelling, which is the exact
        // inverse of the bug this function exists to prevent.
        Some(f) if f < 0.0 && f >= i64::MIN as f64 => (f as i64).to_string(),
        Some(f) => f.to_string(),
        None => n.to_string(),
    }
}

/// Numeric form of a scalar by its JSON type alone, or `None` when it cannot
/// participate in range queries — the guess for a field no schema types.
///
/// Booleans map to 1.0/0.0 so they are usable by both equality and range operators
/// without needing a third column.
pub fn numeric_value(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Seconds since the Unix epoch (UTC, fractional seconds kept) of a `date`
/// (`YYYY-MM-DD`, read as midnight UTC) or an RFC 3339 `timestamp` (normalized from
/// its offset), or `None` for anything else. The one parser behind a date field's
/// `value_num` and a date-string range bound in a filter, so a stored date and a
/// filter bound always land on the same axis (#349).
pub(crate) fn epoch_seconds(text: &str) -> Option<f64> {
    if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(text) {
        return Some(ts.timestamp() as f64 + f64::from(ts.timestamp_subsec_nanos()) / 1e9);
    }
    chrono::NaiveDate::parse_from_str(text, "%Y-%m-%d")
        .ok()
        .and_then(|date| date.and_hms_opt(0, 0, 0))
        .map(|midnight| midnight.and_utc().timestamp() as f64)
}

/// The `value_num` a scalar of a declared type stores, per that type's
/// [`NumericValue`]: a `text` or `enum` field holding a number stores none.
pub(crate) fn declared_numeric(numeric: NumericValue, value: &Value) -> Option<f64> {
    match numeric {
        NumericValue::None => None,
        NumericValue::Number => value.as_f64(),
        NumericValue::Boolean => value.as_bool().map(|b| if b { 1.0 } else { 0.0 }),
        NumericValue::Epoch => value.as_str().and_then(epoch_seconds),
    }
}

/// A scalar's `value_num` under its schema node: the declared type's mapping when
/// the node has a type, else the JSON-type guess.
fn scalar_numeric(node: Option<&FieldDef>, value: &Value) -> Option<f64> {
    match node.and_then(|n| n.ty) {
        Some(ty) => declared_numeric(ty.kind().numeric, value),
        None => numeric_value(value),
    }
}

/// The schema node a list's elements are read under: its `items`, or — for a
/// scalar type that also accepts a list of itself (`enum`, and any scalar type a
/// non-strict document gives a list) — the node itself. `None` (guess) for an
/// untyped node or a `list` without `items`.
fn element_node(node: Option<&FieldDef>) -> Option<&FieldDef> {
    let node = node?;
    if let Some(items) = node.items.as_deref() {
        return Some(items);
    }
    match node.ty {
        None | Some(FieldType::List) | Some(FieldType::Object) => None,
        Some(_) => Some(node),
    }
}

/// Flatten a document's frontmatter into `document_fields` rows, reading types from
/// `schema` (the one governing the document).
///
/// - Nested objects become dot-paths, recursively, down to [`MAX_FIELD_PATH_DEPTH`]
///   path steps (each name and each `[]` is one, the same limit a schema may
///   declare to). The container key itself never gets a row — whole objects are not
///   filterable.
/// - Arrays of scalars become one row per element sharing a field name, which is what
///   makes any-of and all-of matching on `tags` possible.
/// - The objects of a list project under `list[]`: `npcs: [{role: x}]` gives an
///   `npcs[].role` row, with [`FieldRow::elem`] its position in `npcs`. Lists nested
///   inside an element (`npcs[].tags`, `npcs[].items[].name`) keep the element of the
///   outermost list; a list's list elements project under `list[]` the same way.
/// - A declared field's type decides its `value_num` ([`declared_numeric`]):
///   numbers for `integer`/`number`, 1/0 for `boolean`, epoch seconds for
///   `date`/`timestamp` ([`epoch_seconds`]), none for `text`/`enum`. A field no
///   schema types keeps the JSON-type guess ([`numeric_value`]). The structure
///   always follows the value, so an undeclared list of objects projects too.
/// - `null` produces no row, so "absent" and "explicitly null" filter identically.
pub fn flatten_frontmatter(
    frontmatter: &HashMap<String, Value>,
    schema: &ResolvedSchema,
) -> Vec<FieldRow> {
    let mut rows = Vec::new();
    // Sorted for deterministic output — makes tests stable and diffs readable —
    // with every top-level list holding objects or lists after the rest: such a
    // list can expand into many rows, and the row cap must not let it starve the
    // scalar fields (`type`, `tags`) that projected before #349 projected lists.
    let mut keys: Vec<&String> = frontmatter.keys().collect();
    keys.sort_by_key(|key| (holds_containers(&frontmatter[*key]), *key));
    for key in keys {
        if PROMOTED_FIELDS.contains(&key.as_str()) {
            continue;
        }
        let node = schema.fields.tree().get(key.as_str());
        flatten_value(key, 1, &frontmatter[key], node, None, &mut rows);
        if rows.len() >= MAX_PROJECTED_ROWS {
            warn!(
                rows = rows.len(),
                "frontmatter projection hit the row cap; later fields are not filterable"
            );
            rows.truncate(MAX_PROJECTED_ROWS);
            break;
        }
    }
    rows
}

/// Whether `value` is a list with an object or list among its elements.
fn holds_containers(value: &Value) -> bool {
    value
        .as_array()
        .is_some_and(|items| items.iter().any(|v| v.is_array() || v.is_object()))
}

/// Project `value` at `path` (`depth` path steps deep), read under `node`. `elem`
/// is the element of the outermost list of objects already entered, if any.
fn flatten_value(
    path: &str,
    depth: usize,
    value: &Value,
    node: Option<&FieldDef>,
    elem: Option<i64>,
    out: &mut Vec<FieldRow>,
) {
    match value {
        Value::Null => {}
        Value::String(_) | Value::Number(_) | Value::Bool(_) => {
            if out.len() >= MAX_PROJECTED_ROWS {
                return;
            }
            push_leaf(path, elem, value, node, out);
        }
        Value::Array(items) => {
            let items_node = element_node(node);
            let mut too_deep = false;
            for (i, item) in items.iter().enumerate() {
                // The cap is checked here, inside the expansion, so the WORK is bounded
                // and not just the result. Documents arriving via git bypass the MCP
                // write tools' size cap, so a single huge array would otherwise be
                // fully materialized before any outer check saw it.
                if out.len() >= MAX_PROJECTED_ROWS {
                    return;
                }
                match item {
                    Value::Array(_) | Value::Object(_) => {
                        if depth >= MAX_FIELD_PATH_DEPTH {
                            too_deep = true;
                            continue;
                        }
                        let elem = elem.or(Some(i as i64));
                        flatten_value(&format!("{path}[]"), depth + 1, item, items_node, elem, out);
                    }
                    scalar => push_leaf(path, elem, scalar, items_node, out),
                }
            }
            if too_deep {
                warn!(
                    field = path,
                    depth, "max flatten depth reached; deeper fields are not filterable"
                );
            }
        }
        Value::Object(map) => {
            if out.len() >= MAX_PROJECTED_ROWS {
                return;
            }
            if depth >= MAX_FIELD_PATH_DEPTH {
                warn!(
                    field = path,
                    depth, "max flatten depth reached; deeper fields are not filterable"
                );
                return;
            }
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for key in keys {
                // Per key, not merely once before the loop: the depth limit bounds
                // nesting but not key count, so a single very wide object would
                // otherwise be fully materialized before any outer check saw it.
                if out.len() >= MAX_PROJECTED_ROWS {
                    return;
                }
                let child = node.and_then(|n| n.fields.get(key.as_str()));
                flatten_value(
                    &format!("{path}.{key}"),
                    depth + 1,
                    &map[key],
                    child,
                    elem,
                    out,
                );
            }
        }
    }
}

fn push_leaf(
    path: &str,
    elem: Option<i64>,
    value: &Value,
    node: Option<&FieldDef>,
    out: &mut Vec<FieldRow>,
) {
    if let Some(value_text) = canonical_text(value) {
        out.push(FieldRow {
            field: path.to_string(),
            elem: elem.unwrap_or(0),
            value_text,
            value_num: scalar_numeric(node, value),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fm(value: Value) -> HashMap<String, Value> {
        match value {
            Value::Object(map) => map.into_iter().collect(),
            _ => panic!("frontmatter fixture must be an object"),
        }
    }

    fn find<'a>(rows: &'a [FieldRow], field: &str) -> Vec<&'a FieldRow> {
        rows.iter().filter(|r| r.field == field).collect()
    }

    /// The projection with no schema: every field undeclared.
    fn flatten_undeclared(frontmatter: &HashMap<String, Value>) -> Vec<FieldRow> {
        flatten_frontmatter(frontmatter, &ResolvedSchema::default())
    }

    /// A schema from root `.schema.yaml` text.
    fn schema(yaml: &str) -> ResolvedSchema {
        let file = crate::schema::parse_schema_text(yaml).expect("valid schema");
        ResolvedSchema::default().merged_with_for_test(&file, "/")
    }

    /// The real shape from lifestyle/kitchen/recipes/stir-fry.md — the document that
    /// motivated nested-field support in the first place.
    fn stir_fry() -> HashMap<String, Value> {
        fm(json!({
            "title": "Stir Fry",
            "description": "Shopping-list-only entry for the weekly improvised stir fry.",
            "type": "reference",
            "domain": "lifestyle",
            "tags": ["recipe", "stir-fry", "stovetop", "wok"],
            "planning": {
                "servings": 4,
                "nights_covered": 3,
                "nights_confidence": "estimated",
                "tested": true,
                "rating": 5,
                "prep_minutes": 45,
                "cook_minutes": 15,
                "effort": "medium",
                "cuisine": "asian",
                "scalable": true,
                "needs_recipe": false
            }
        }))
    }

    #[test]
    fn promoted_fields_are_not_projected() {
        let rows = flatten_undeclared(&stir_fry());
        assert!(find(&rows, "title").is_empty(), "title has its own column");
        assert!(
            find(&rows, "description").is_empty(),
            "description has its own column"
        );
    }

    #[test]
    fn scalars_project_with_canonical_text() {
        let rows = flatten_undeclared(&stir_fry());
        let ty = find(&rows, "type");
        assert_eq!(ty.len(), 1);
        assert_eq!(ty[0].value_text, "reference");
        assert_eq!(ty[0].value_num, None);
    }

    #[test]
    fn arrays_produce_one_row_per_element() {
        let rows = flatten_undeclared(&stir_fry());
        let tags = find(&rows, "tags");
        assert_eq!(tags.len(), 4, "one row per tag enables any-of/all-of");
        let values: Vec<&str> = tags.iter().map(|r| r.value_text.as_str()).collect();
        assert!(values.contains(&"recipe"));
        assert!(values.contains(&"wok"));
    }

    #[test]
    fn nested_objects_flatten_to_dot_paths() {
        let rows = flatten_undeclared(&stir_fry());
        let prep = find(&rows, "planning.prep_minutes");
        assert_eq!(prep.len(), 1);
        assert_eq!(prep[0].value_text, "45");
        assert_eq!(prep[0].value_num, Some(45.0));
    }

    #[test]
    fn container_key_itself_gets_no_row() {
        let rows = flatten_undeclared(&stir_fry());
        assert!(
            find(&rows, "planning").is_empty(),
            "whole objects are not filterable"
        );
    }

    #[test]
    fn booleans_project_to_text_and_number() {
        let rows = flatten_undeclared(&stir_fry());
        let needs = find(&rows, "planning.needs_recipe");
        assert_eq!(needs.len(), 1);
        assert_eq!(needs[0].value_text, "false");
        assert_eq!(needs[0].value_num, Some(0.0));

        let tested = find(&rows, "planning.tested");
        assert_eq!(tested[0].value_text, "true");
        assert_eq!(tested[0].value_num, Some(1.0));
    }

    #[test]
    fn filter_value_canonicalizes_to_stored_text() {
        // The property that makes {"planning.needs_recipe": false} match the stored row.
        let rows = flatten_undeclared(&stir_fry());
        let stored = find(&rows, "planning.needs_recipe")[0].value_text.clone();
        assert_eq!(canonical_text(&json!(false)), Some(stored));

        let prep = find(&rows, "planning.prep_minutes")[0].value_text.clone();
        assert_eq!(canonical_text(&json!(45)), Some(prep));
    }

    #[test]
    fn nulls_produce_no_row() {
        let rows = flatten_undeclared(&fm(json!({ "status": null, "type": "guide" })));
        assert!(find(&rows, "status").is_empty());
        assert_eq!(find(&rows, "type").len(), 1);
    }

    #[test]
    fn arrays_of_objects_project_per_element_under_brackets() {
        // #349: an undeclared list of objects projects too — the structure follows
        // the value; only types come from the schema.
        let rows = flatten_undeclared(&fm(json!({
            "steps": [{ "n": 1 }, { "n": 2 }],
            "type": "guide"
        })));
        assert!(
            find(&rows, "steps").is_empty(),
            "the list itself has no row"
        );
        let n: Vec<(i64, &str)> = find(&rows, "steps[].n")
            .iter()
            .map(|r| (r.elem, r.value_text.as_str()))
            .collect();
        assert_eq!(n, vec![(0, "1"), (1, "2")]);
        assert_eq!(find(&rows, "type").len(), 1, "siblings still project");
        assert_eq!(find(&rows, "type")[0].elem, 0);
    }

    #[test]
    fn declared_list_of_objects_projects_with_elem_and_typed_value_num() {
        let schema = schema(
            "fields:\n  npcs:\n    type: list\n    items:\n      type: object\n      fields:\n        \
             name:\n          type: text\n        level:\n          type: integer\n        \
             code:\n          type: text\n        tags:\n          type: list\n",
        );
        let rows = flatten_frontmatter(
            &fm(json!({
                "npcs": [
                    { "name": "Ada", "level": 3, "code": 7, "tags": ["a", "b", "a"] },
                    { "name": "Bo", "level": 5, "code": 9, "tags": ["c"],
                      "gear": [{ "kind": "sword" }] }
                ]
            })),
            &schema,
        );
        let names: Vec<(i64, &str)> = find(&rows, "npcs[].name")
            .iter()
            .map(|r| (r.elem, r.value_text.as_str()))
            .collect();
        assert_eq!(names, vec![(0, "Ada"), (1, "Bo")]);
        let levels: Vec<(i64, Option<f64>)> = find(&rows, "npcs[].level")
            .iter()
            .map(|r| (r.elem, r.value_num))
            .collect();
        assert_eq!(levels, vec![(0, Some(3.0)), (1, Some(5.0))]);
        // Declared text holding a number: text, no value_num.
        assert!(
            find(&rows, "npcs[].code")
                .iter()
                .all(|r| r.value_num.is_none())
        );
        // A scalar list inside an element keeps the element's index.
        let tags: Vec<(i64, &str)> = find(&rows, "npcs[].tags")
            .iter()
            .map(|r| (r.elem, r.value_text.as_str()))
            .collect();
        assert_eq!(tags, vec![(0, "a"), (0, "b"), (0, "a"), (1, "c")]);
        // A nested list of objects keeps the OUTERMOST element.
        let gear = find(&rows, "npcs[].gear[].kind");
        assert_eq!(gear.len(), 1);
        assert_eq!(gear[0].elem, 1);
    }

    #[test]
    fn declared_text_holding_a_number_gets_no_value_num_but_undeclared_still_guesses() {
        let schema = schema(
            "fields:\n  code:\n    type: text\n  count:\n    type: integer\n  flag:\n    \
             type: boolean\n  kind:\n    type: enum\n    values: [1, 2]\n",
        );
        let rows = flatten_frontmatter(
            &fm(json!({ "code": 42, "count": 3, "flag": true, "kind": 1, "loose": 7 })),
            &schema,
        );
        assert_eq!(find(&rows, "code")[0].value_text, "42");
        assert_eq!(find(&rows, "code")[0].value_num, None, "declared text");
        assert_eq!(find(&rows, "kind")[0].value_num, None, "declared enum");
        assert_eq!(find(&rows, "count")[0].value_num, Some(3.0));
        assert_eq!(find(&rows, "flag")[0].value_num, Some(1.0));
        assert_eq!(
            find(&rows, "loose")[0].value_num,
            Some(7.0),
            "an undeclared number still guesses"
        );
    }

    #[test]
    fn dates_and_timestamps_store_epoch_seconds_on_one_axis() {
        let schema = schema("fields:\n  on:\n    type: date\n  at:\n    type: timestamp\n");
        let rows = flatten_frontmatter(
            &fm(json!({ "on": "2024-01-02", "at": "2024-01-02T01:00:00+02:00" })),
            &schema,
        );
        let on = find(&rows, "on")[0].value_num.unwrap();
        let at = find(&rows, "at")[0].value_num.unwrap();
        assert_eq!(on, 1_704_153_600.0, "midnight UTC");
        // 01:00 at +02:00 is 23:00 UTC the day before — before the date's midnight.
        assert_eq!(at, 1_704_150_000.0);
        assert!(at < on);
        assert_eq!(
            epoch_seconds("2024-01-01T00:00:00.5Z"),
            Some(1_704_067_200.5)
        );
        assert_eq!(epoch_seconds("not a date"), None);
        // An undeclared date-looking string stays text-only.
        let rows = flatten_undeclared(&fm(json!({ "on": "2024-01-02" })));
        assert_eq!(find(&rows, "on")[0].value_num, None);
    }

    #[test]
    fn a_big_list_of_objects_cannot_starve_scalar_fields_of_the_row_cap() {
        // `aaa` sorts first; its rows alone exceed the cap.
        let big: Vec<Value> = (0..MAX_PROJECTED_ROWS + 100)
            .map(|i| json!({ "n": i }))
            .collect();
        let rows = flatten_undeclared(&fm(json!({
            "aaa": big,
            "type": "guide",
            "tags": ["a", "b"],
            "planning": { "effort": "low" }
        })));
        assert!(rows.len() <= MAX_PROJECTED_ROWS);
        assert_eq!(find(&rows, "type").len(), 1);
        assert_eq!(find(&rows, "tags").len(), 2);
        assert_eq!(find(&rows, "planning.effort").len(), 1);
        assert!(
            !find(&rows, "aaa[].n").is_empty(),
            "the list gets what is left"
        );
    }

    #[test]
    fn a_depth_eight_object_now_projects() {
        // The pre-#349 cap was 4 steps.
        let rows = flatten_undeclared(&fm(json!({
            "a": { "b": { "c": { "d": { "e": { "f": { "g": { "h": "deep" } } } } } } }
        })));
        assert_eq!(find(&rows, "a.b.c.d.e.f.g.h").len(), 1);
    }

    #[test]
    fn a_huge_array_is_bounded_during_expansion_not_after() {
        let many: Vec<Value> = (0..50_000).map(|i| json!(format!("t{i}"))).collect();
        let rows = flatten_undeclared(&fm(json!({ "tags": many })));

        assert!(
            rows.len() <= MAX_PROJECTED_ROWS,
            "one field must not exceed the cap, got {}",
            rows.len()
        );
    }

    #[test]
    fn a_very_wide_object_is_bounded_during_expansion() {
        let mut wide = serde_json::Map::new();
        for i in 0..50_000 {
            wide.insert(format!("k{i}"), json!(i));
        }
        let rows = flatten_undeclared(&fm(json!({ "custom": Value::Object(wide) })));

        assert!(
            rows.len() <= MAX_PROJECTED_ROWS,
            "breadth must be bounded too, got {}",
            rows.len()
        );
    }

    #[test]
    fn values_beyond_u64_keep_distinct_spellings() {
        // Rust saturates float->int casts, so an unbounded integer arm would give
        // 2e19 and 5e25 the same canonical text and make them match each other.
        let a = canonical_text(&json!(2.0e19f64)).unwrap();
        let b = canonical_text(&json!(5.0e25f64)).unwrap();
        assert_ne!(a, b, "distinct values must not collapse onto one spelling");

        let very_negative = canonical_text(&json!(-5.0e25f64)).unwrap();
        assert_ne!(very_negative, canonical_text(&json!(-2.0e19f64)).unwrap());
    }

    /// `{a0: {a1: ... {a<steps-1>: "leaf"}}}`: a leaf `steps` path steps deep.
    fn nested(steps: usize) -> (HashMap<String, Value>, String) {
        let mut value = json!("leaf");
        for i in (1..steps).rev() {
            value = json!({ format!("a{i}"): value });
        }
        let path = (0..steps)
            .map(|i| format!("a{i}"))
            .collect::<Vec<_>>()
            .join(".");
        (fm(json!({ "a0": value })), path)
    }

    #[test]
    fn flattening_stops_at_the_shared_max_depth() {
        let (doc, path) = nested(MAX_FIELD_PATH_DEPTH);
        assert_eq!(find(&flatten_undeclared(&doc), &path).len(), 1);
        let (doc, path) = nested(MAX_FIELD_PATH_DEPTH + 1);
        let rows = flatten_undeclared(&doc);
        assert!(
            rows.iter().all(|r| r.field != path),
            "depth beyond the cap is not projected"
        );
        // `[]` counts as a step too: `l`, `[]`, then the chain below `a0`.
        let (doc, _) = nested(MAX_FIELD_PATH_DEPTH - 1);
        let rows = flatten_undeclared(&fm(json!({ "l": [doc["a0"].clone()] })));
        assert_eq!(
            rows.len(),
            1,
            "exactly {MAX_FIELD_PATH_DEPTH} steps: {rows:?}"
        );
        let (doc, _) = nested(MAX_FIELD_PATH_DEPTH);
        let rows = flatten_undeclared(&fm(json!({ "l": [doc["a0"].clone()] })));
        assert!(rows.is_empty(), "one step too deep: {rows:?}");
    }

    #[test]
    fn floats_keep_numeric_form() {
        let rows = flatten_undeclared(&fm(json!({ "score": 1.5 })));
        let score = find(&rows, "score");
        assert_eq!(score[0].value_num, Some(1.5));
        assert_eq!(score[0].value_text, "1.5");
    }

    #[test]
    fn integral_floats_and_integers_share_one_spelling() {
        // A document written `prep_minutes: 45.0` must be findable by a filter for 45.
        // Without normalization the stored text is "45.0", the filter canonicalizes to
        // "45", and the document silently vanishes from equality matches.
        assert_eq!(canonical_text(&json!(45.0)), canonical_text(&json!(45)));
        assert_eq!(canonical_text(&json!(45.0)).as_deref(), Some("45"));

        let stored = flatten_undeclared(&fm(json!({ "planning": { "prep_minutes": 45.0 } })));
        assert_eq!(
            find(&stored, "planning.prep_minutes")[0].value_text,
            canonical_text(&json!(45)).unwrap()
        );
    }

    #[test]
    fn values_above_i64_agree_whether_written_as_int_or_float() {
        // u64::MAX-scale values take the as_u64 path when written as an integer; the
        // float path must produce the same spelling or the two never match.
        let big = 9_300_000_000_000_000_000u64;
        let as_int = serde_json::Number::from(big);
        let as_float = serde_json::Number::from_f64(big as f64).unwrap();

        assert_eq!(
            canonical_text(&Value::Number(as_int)),
            canonical_text(&Value::Number(as_float)),
            "one value must have exactly one canonical spelling"
        );
    }

    #[test]
    fn non_finite_floats_do_not_panic() {
        // serde_json cannot construct NaN/Infinity, but the guard must hold regardless.
        assert!(canonical_text(&json!(1e308)).is_some());
    }

    #[test]
    fn negative_and_large_numbers_canonicalize() {
        assert_eq!(canonical_text(&json!(-7)).as_deref(), Some("-7"));
        assert_eq!(canonical_text(&json!(-7.0)).as_deref(), Some("-7"));
        assert_eq!(
            canonical_text(&json!(i64::MAX)).as_deref(),
            Some("9223372036854775807")
        );
        // Non-integral stays as written.
        assert_eq!(canonical_text(&json!(-7.25)).as_deref(), Some("-7.25"));
    }

    #[test]
    fn output_is_deterministic() {
        // HashMap iteration order must not leak into the projection, or every
        // reindex would look like a change.
        let first = flatten_undeclared(&stir_fry());
        for _ in 0..8 {
            assert_eq!(flatten_undeclared(&stir_fry()), first);
        }
    }
}
