//! Post-processes a schemars-derived JSON Schema (a `Tool::input_schema`) into a
//! self-contained schema with no `$ref`, no `$defs`/`definitions`, and no boolean
//! subschema in a schema position (#288).
//!
//! llama-server turns every tool's `inputSchema` into a GBNF grammar at
//! `tools/list` time and returns HTTP 400 for the *whole* request if any one
//! tool's schema doesn't convert. This server's `search`, `write_document` and
//! `update_schema` schemas carry `$defs`/`$ref` (rmcp's `schema_for_type` hardcodes
//! `SchemaSettings::draft2020_12()`, which has no hook to inline them) and a bare
//! `true` subschema (from `FrontmatterPatchOp::value`/`values` and
//! `RawFieldDef::default`, both typed `serde_json::Value`) — which is what broke
//! llama.cpp-backed clients (Crush, OpenCode) against this server. Even with
//! schemars' `inline_subschemas` setting, a recursive type such as
//! `schema::RawFieldDef` (`fields: Option<HashMap<String, RawFieldDef>>`) still
//! needs a `$ref` somewhere, so inlining alone can't reach zero refs — this module
//! cuts the cycle instead.
//!
//! Called from `mcp.rs`'s `list_tools`/`get_tool` as the step after
//! `overlay_input_schema` (#286), on the same `Tool::input_schema` — see that
//! `ServerHandler` impl.

use rmcp::model::{JsonObject, Tool};
use serde_json::Value;

/// Sentinel key under which the root schema (as it looked right after `$defs`
/// removal, before any rewriting) is stored in the working `defs` map, so a
/// literal `"$ref": "#"` — a whole-document self-reference — resolves exactly
/// like a named `$defs` entry instead of needing its own code path. No real
/// `$defs` name can collide with it: JSON Pointer names come from a
/// `#/$defs/<name>` path, which always has at least one `/`.
const ROOT_REF: &str = "#";

/// Single-schema keywords: the value is itself one schema (object or boolean).
const SINGLE_SUBSCHEMA_KEYS: &[&str] = &[
    "items",
    "unevaluatedItems",
    "not",
    "if",
    "then",
    "else",
    "contains",
    "propertyNames",
];
/// Map-of-schemas keywords: every value is a schema; keys are just names.
const SUBSCHEMA_MAP_KEYS: &[&str] = &["properties", "patternProperties", "dependentSchemas"];
/// Array-of-schemas keywords: every element is a schema.
const SUBSCHEMA_ARRAY_KEYS: &[&str] = &["prefixItems", "anyOf", "oneOf", "allOf"];

/// `Tool`-level wrapper for the two `mcp.rs` call sites: clones the schema out
/// of the `Arc`, rewrites it, and reassigns — the same clone-don't-mutate-through-
/// the-`Arc` posture `overlay_input_schema` (#286) uses on the same field, so one
/// request's rewrite can never leak into another's or into the router's own
/// cached `Tool`.
pub fn self_contained(mut tool: Tool) -> Tool {
    let mut schema: JsonObject = (*tool.input_schema).clone();
    make_self_contained(&mut schema);
    tool.input_schema = std::sync::Arc::new(schema);
    tool
}

/// Rewrites `schema` in place: every `{"$ref": "#/$defs/Name", ...siblings}` (or
/// the whole-document `"$ref": "#"`) becomes a recursively-processed clone of its
/// target, with sibling keywords such as `description` merged over it (sibling
/// wins); a ref back to a name already being expanded is cut to
/// `{"type": <that def's type, if any>}` plus the ref-site's own keywords, so a
/// recursive type terminates instead of expanding forever; and every boolean
/// subschema in a schema position (`items: true`, `anyOf: [false, ...]`, etc. —
/// never `enum`/`const`/`default`/`examples`, and never a boolean
/// `additionalProperties`/`unevaluatedProperties`, which is the conventional
/// `deny_unknown_fields` form and not what #288 reports) becomes `{}`/`{"not":{}}`.
/// The root `$defs` (and `definitions`, its legacy name) is dropped; the server
/// still deserializes and validates every accepted input exactly as before —
/// this only changes what a client's schema-to-grammar converter sees.
pub fn make_self_contained(schema: &mut JsonObject) {
    let mut defs = take_defs(schema);
    defs.insert(ROOT_REF.to_string(), Value::Object(schema.clone()));

    // The whole call is itself "inside" the root's own expansion from the
    // start — `schema` IS the root — so a `"$ref": "#"` found anywhere below
    // is always a repeat, never a first expansion, and is cut to a stub
    // rather than recursing into `schema` again.
    let mut stack = vec![ROOT_REF.to_string()];
    resolve_and_walk(schema, &defs, &mut stack);
}

/// Removes and returns the root `$defs` map, merging in `definitions` (the
/// legacy name) if that is present too. Only the root's own map is a rewrite
/// target: schemars puts every named type reachable from a tool's parameter
/// struct at the root, never nested inside another `$defs` entry.
fn take_defs(schema: &mut JsonObject) -> JsonObject {
    let mut defs = JsonObject::new();
    for key in ["$defs", "definitions"] {
        if let Some(Value::Object(map)) = schema.remove(key) {
            defs.extend(map);
        }
    }
    defs
}

/// The last JSON Pointer segment of a `$ref` string, JSON-Pointer-unescaped
/// (`~1` -> `/`, `~0` -> `~`, in that order — reversing the escaping order the
/// spec defines). `#/$defs/Name` and the bare `#` (no `/` at all, so the whole
/// string is the "segment") both fall out of the same `rsplit`.
fn ref_target_name(ref_str: &str) -> String {
    let raw = ref_str.rsplit('/').next().unwrap_or(ref_str);
    if raw.contains('~') {
        raw.replace("~1", "/").replace("~0", "~")
    } else {
        raw.to_string()
    }
}

/// Handles one value known to sit in a schema position (a keyword's value, an
/// `items`/`properties.*`/`anyOf[*]`/etc. slot) — either a boolean subschema, or
/// an object schema to resolve-and-walk in place.
fn process_schema_position(value: &mut Value, defs: &JsonObject, stack: &mut Vec<String>) {
    match value {
        Value::Bool(true) => *value = Value::Object(JsonObject::new()),
        Value::Bool(false) => {
            let mut not_obj = JsonObject::new();
            not_obj.insert("not".to_string(), Value::Object(JsonObject::new()));
            *value = Value::Object(not_obj);
        }
        Value::Object(obj) => resolve_and_walk(obj, defs, stack),
        // Not a valid schema position (malformed input); leave it alone rather
        // than guess.
        _ => {}
    }
}

/// If `obj` is a `$ref` site, replaces it in place with its resolved,
/// recursively-processed target (siblings merged over it); either way, then
/// walks `obj`'s own schema-valued keywords.
fn resolve_and_walk(obj: &mut JsonObject, defs: &JsonObject, stack: &mut Vec<String>) {
    if obj.contains_key("$ref") {
        let ref_str = match obj.remove("$ref") {
            Some(Value::String(s)) => s,
            other => {
                // Not a well-formed $ref; put back whatever was there (if
                // anything) and fall through to the ordinary keyword walk
                // rather than losing data.
                if let Some(v) = other {
                    obj.insert("$ref".to_string(), v);
                }
                walk_keywords(obj, defs, stack);
                return;
            }
        };
        let siblings = std::mem::take(obj);
        let name = ref_target_name(&ref_str);

        let mut resolved = if stack.contains(&name) {
            // Cycle: a ref back to a def already being expanded. Cut it to a
            // permissive stub carrying just the type, so nested property
            // lists don't expand forever — the server still deserializes and
            // validates the real (possibly nested) value exactly as before;
            // only what a client's schema view advertises below one level
            // changes.
            let mut stub = JsonObject::new();
            if let Some(ty) = defs
                .get(&name)
                .and_then(Value::as_object)
                .and_then(|d| d.get("type"))
            {
                stub.insert("type".to_string(), ty.clone());
            }
            stub
        } else {
            match defs.get(&name) {
                Some(Value::Object(def_obj)) => {
                    let mut cloned = def_obj.clone();
                    stack.push(name.clone());
                    walk_keywords(&mut cloned, defs, stack);
                    stack.pop();
                    cloned
                }
                _ => {
                    // A ref schemars didn't actually put in `$defs`/`definitions`
                    // (or a `$defs` entry that isn't itself an object schema).
                    // Production degrades to "any value" rather than emitting a
                    // dangling ref; tests catch the case via this assert.
                    debug_assert!(false, "tool_schema: unresolved $ref '{ref_str}'");
                    JsonObject::new()
                }
            }
        };

        for (k, v) in siblings {
            resolved.insert(k, v);
        }
        *obj = resolved;
    }
    // Sibling keywords are legal alongside `$ref` in draft 2020-12 and, for
    // this codebase, are always plain data (`description`) — but walking
    // again here is cheap on these small tool schemas and keeps this correct
    // even if that ever stops being true.
    walk_keywords(obj, defs, stack);
}

/// Recurses into every schema-valued keyword of `obj` — the subschema
/// positions a JSON Schema can hold one in. Deliberately does not touch
/// data-valued keywords (`enum`, `const`, `default`, `examples`, `required`,
/// `type`, ...): a `default` value that happens to contain a `"$ref"` key must
/// stay literal data, not be mistaken for an actual reference.
fn walk_keywords(obj: &mut JsonObject, defs: &JsonObject, stack: &mut Vec<String>) {
    for key in SINGLE_SUBSCHEMA_KEYS {
        if let Some(v) = obj.get_mut(*key) {
            process_schema_position(v, defs, stack);
        }
    }
    for key in SUBSCHEMA_MAP_KEYS {
        if let Some(Value::Object(map)) = obj.get_mut(*key) {
            for v in map.values_mut() {
                process_schema_position(v, defs, stack);
            }
        }
    }
    for key in SUBSCHEMA_ARRAY_KEYS {
        if let Some(Value::Array(items)) = obj.get_mut(*key) {
            for v in items {
                process_schema_position(v, defs, stack);
            }
        }
    }
    // Schema-valued only when written as an object; a boolean here is the
    // conventional `deny_unknown_fields` form (`additionalProperties: false`)
    // and is deliberately left untouched — see the module doc comment.
    for key in ["additionalProperties", "unevaluatedProperties"] {
        if matches!(obj.get(key), Some(Value::Object(_))) {
            process_schema_position(obj.get_mut(key).unwrap(), defs, stack);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn obj(v: Value) -> JsonObject {
        match v {
            Value::Object(m) => m,
            other => panic!("expected a JSON object, got {other}"),
        }
    }

    #[test]
    fn simple_ref_is_inlined_and_defs_dropped() {
        let mut schema = obj(json!({
            "type": "object",
            "properties": { "foo": { "$ref": "#/$defs/Foo" } },
            "$defs": { "Foo": { "type": "string", "minLength": 1 } }
        }));
        make_self_contained(&mut schema);
        assert!(!schema.contains_key("$defs"));
        assert_eq!(
            schema["properties"]["foo"],
            json!({ "type": "string", "minLength": 1 })
        );
    }

    #[test]
    fn ref_site_sibling_description_wins_over_the_def_description() {
        let mut schema = obj(json!({
            "type": "object",
            "properties": {
                "foo": { "$ref": "#/$defs/Foo", "description": "site" }
            },
            "$defs": { "Foo": { "type": "string", "description": "def" } }
        }));
        make_self_contained(&mut schema);
        let foo = &schema["properties"]["foo"];
        assert_eq!(foo["description"], json!("site"));
        assert_eq!(foo["type"], json!("string"));
    }

    #[test]
    fn nested_ref_through_another_def_is_fully_inlined() {
        let mut schema = obj(json!({
            "type": "object",
            "properties": { "foo": { "$ref": "#/$defs/Foo" } },
            "$defs": {
                "Foo": {
                    "type": "object",
                    "properties": { "bar": { "$ref": "#/$defs/Bar" } }
                },
                "Bar": { "type": "integer" }
            }
        }));
        make_self_contained(&mut schema);
        assert_eq!(
            schema["properties"]["foo"]["properties"]["bar"],
            json!({ "type": "integer" })
        );
        assert!(
            !serde_json::to_string(&schema).unwrap().contains("$ref"),
            "no $ref of any kind should survive: {schema:?}"
        );
    }

    #[test]
    fn root_self_ref_is_inlined_against_the_root_and_terminates() {
        // A whole-document self-reference: the type used at `child` is the
        // root type itself, rather than a named `$defs` entry.
        let mut schema = obj(json!({
            "type": "object",
            "properties": { "child": { "$ref": "#" } }
        }));
        make_self_contained(&mut schema);
        let child = &schema["properties"]["child"];
        // The root is already "on the stack" the whole call, so this one
        // occurrence is a repeat and is cut to a stub rather than expanding
        // forever.
        assert_eq!(child["type"], json!("object"));
        assert!(child.get("properties").is_none());
    }

    #[test]
    fn a_recursive_def_is_cut_to_a_type_stub_at_the_repeat() {
        let mut schema = obj(json!({
            "type": "object",
            "properties": { "root_field": { "$ref": "#/$defs/Node" } },
            "$defs": {
                "Node": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "children": {
                            "type": "array",
                            "items": { "$ref": "#/$defs/Node" }
                        }
                    }
                }
            }
        }));
        make_self_contained(&mut schema);
        let node = &schema["properties"]["root_field"];
        // The outer expansion keeps its full shape...
        assert_eq!(node["additionalProperties"], json!(false));
        assert!(node["properties"]["children"].is_object());
        // ...but the self-referencing occurrence one level in terminates.
        let repeat = &node["properties"]["children"]["items"];
        assert_eq!(repeat, &json!({ "type": "object" }));
        assert!(repeat.get("properties").is_none());
    }

    #[test]
    fn boolean_true_and_false_subschemas_are_replaced_in_schema_positions() {
        let mut schema = obj(json!({
            "type": "object",
            "properties": {
                "any_list": { "type": "array", "items": true },
                "excluded": { "anyOf": [false, { "type": "string" }] }
            }
        }));
        make_self_contained(&mut schema);
        assert_eq!(schema["properties"]["any_list"]["items"], json!({}));
        assert_eq!(
            schema["properties"]["excluded"]["anyOf"][0],
            json!({ "not": {} })
        );
    }

    #[test]
    fn data_keywords_and_boolean_additional_properties_are_left_untouched() {
        let mut schema = obj(json!({
            "type": "object",
            "properties": {
                "flag": { "enum": [true], "default": false, "const": true }
            },
            "additionalProperties": false
        }));
        make_self_contained(&mut schema);
        let flag = &schema["properties"]["flag"];
        assert_eq!(flag["enum"], json!([true]));
        assert_eq!(flag["default"], json!(false));
        assert_eq!(flag["const"], json!(true));
        assert_eq!(schema["additionalProperties"], json!(false));
    }

    #[test]
    fn a_ref_inside_a_default_value_is_left_as_literal_data() {
        let mut schema = obj(json!({
            "type": "object",
            "properties": {
                "foo": {
                    "type": "object",
                    "default": { "$ref": "not a schema ref, just a data field" }
                }
            }
        }));
        make_self_contained(&mut schema);
        assert_eq!(
            schema["properties"]["foo"]["default"],
            json!({ "$ref": "not a schema ref, just a data field" })
        );
    }

    #[test]
    fn every_schema_position_keyword_resolves_refs_and_replaces_booleans() {
        // Each keyword the walker claims to handle gets a `$ref` and a boolean
        // subschema in its own position shape (single / map / array), so a
        // typo'd or dropped entry in the keyword lists fails here.
        let single = SINGLE_SUBSCHEMA_KEYS
            .iter()
            .map(|k| (*k, json!({ "$ref": "#/$defs/S" }), json!(true)));
        let map = SUBSCHEMA_MAP_KEYS.iter().map(|k| {
            (
                *k,
                json!({ "a": { "$ref": "#/$defs/S" } }),
                json!({ "a": true }),
            )
        });
        let array = SUBSCHEMA_ARRAY_KEYS
            .iter()
            .map(|k| (*k, json!([{ "$ref": "#/$defs/S" }]), json!([true])));
        let object_only = ["additionalProperties", "unevaluatedProperties"]
            .into_iter()
            .map(|k| (k, json!({ "$ref": "#/$defs/S" }), json!({ "items": true })));

        for (key, ref_value, bool_value) in single.chain(map).chain(array).chain(object_only) {
            let mut schema = obj(json!({
                "type": "object",
                "properties": {
                    "with_ref": { key: ref_value },
                    "with_bool": { key: bool_value }
                },
                "$defs": { "S": { "type": "string" } }
            }));
            make_self_contained(&mut schema);
            let text = serde_json::to_string(&schema).unwrap();
            assert!(
                !text.contains("$ref") && !text.contains("$defs"),
                "'{key}': ref not resolved: {text}"
            );
            assert!(
                !text.contains("true"),
                "'{key}': boolean subschema not replaced: {text}"
            );
            assert!(
                text.contains(r#"{"type":"string"}"#),
                "'{key}': def not inlined: {text}"
            );
        }
    }

    #[test]
    fn legacy_definitions_are_resolved_and_dropped() {
        let mut schema = obj(json!({
            "type": "object",
            "properties": { "foo": { "$ref": "#/definitions/Foo" } },
            "definitions": { "Foo": { "type": "boolean" } }
        }));
        make_self_contained(&mut schema);
        assert!(!schema.contains_key("definitions"));
        assert_eq!(schema["properties"]["foo"], json!({ "type": "boolean" }));
    }

    #[test]
    fn a_non_string_ref_is_kept_and_its_siblings_still_walked() {
        let mut schema = obj(json!({
            "type": "object",
            "properties": {
                "foo": { "$ref": 7, "items": true }
            }
        }));
        make_self_contained(&mut schema);
        let foo = &schema["properties"]["foo"];
        assert_eq!(foo["$ref"], json!(7));
        assert_eq!(foo["items"], json!({}));
    }

    #[test]
    #[should_panic]
    fn an_unresolvable_ref_panics_in_debug() {
        let mut schema = obj(json!({
            "type": "object",
            "properties": { "foo": { "$ref": "#/$defs/Missing" } }
        }));
        make_self_contained(&mut schema);
    }
}
