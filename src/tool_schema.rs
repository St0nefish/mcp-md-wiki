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
//! `ServerHandler` impl. [`self_contained`] then runs [`compact`], which drops
//! what costs context without constraining anything (`$schema`, null arms of
//! optional properties, `"default": null`, non-standard `format`s).

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
/// Keywords that hold a schema only when written as an object. A boolean here is
/// the conventional `deny_unknown_fields` form (`additionalProperties: false`),
/// not a subschema, and is left untouched.
const OBJECT_ONLY_SUBSCHEMA_KEYS: &[&str] = &["additionalProperties", "unevaluatedProperties"];

/// `Tool`-level wrapper for the two `mcp.rs` call sites: clones the schema out
/// of the `Arc`, rewrites it, and reassigns — the same clone-don't-mutate-through-
/// the-`Arc` posture `overlay_input_schema` (#286) uses on the same field, so one
/// request's rewrite can never leak into another's or into the router's own
/// cached `Tool`.
pub fn self_contained(mut tool: Tool) -> Tool {
    let mut schema: JsonObject = (*tool.input_schema).clone();
    make_self_contained(&mut schema);
    compact(&mut schema);
    tool.input_schema = std::sync::Arc::new(schema);
    tool
}

/// `format` values defined by JSON Schema itself. schemars also emits Rust
/// numeric widths (`uint`, `uint64`, `float`, ...), which no client acts on —
/// [`compact`] drops those and keeps these.
const STANDARD_FORMATS: &[&str] = &[
    "date-time",
    "date",
    "time",
    "duration",
    "email",
    "idn-email",
    "hostname",
    "idn-hostname",
    "ipv4",
    "ipv6",
    "uri",
    "uri-reference",
    "iri",
    "iri-reference",
    "uuid",
    "uri-template",
    "json-pointer",
    "relative-json-pointer",
    "regex",
];

/// Shrinks an already self-contained schema (see [`make_self_contained`])
/// without changing what it accepts in practice — every tool schema is paid in
/// context on every `tools/list`: drops the root `$schema` (MCP input schemas
/// default to draft 2020-12), every `"default": null`, every non-standard
/// `format`, and the `null` arm of each optional property — `"type": [X,
/// "null"]` becomes `X`, `anyOf: [S, {"type": "null"}]` becomes `S` with the
/// property's own keywords (its `description`) merged over it, and `null` leaves
/// an `enum`. A property is optional when its parent's `required` does not
/// name it; the server still accepts an explicit `null` for one, since every
/// such field deserializes into an `Option`. Introduces no `$ref` and no
/// boolean subschema, so the #288 guarantees hold.
pub fn compact(schema: &mut JsonObject) {
    schema.remove("$schema");
    compact_node(schema);
}

/// [`compact`]'s recursive step over one object schema.
fn compact_node(obj: &mut JsonObject) {
    if matches!(obj.get("default"), Some(Value::Null)) {
        obj.remove("default");
    }
    if let Some(Value::String(format)) = obj.get("format")
        && !STANDARD_FORMATS.contains(&format.as_str())
    {
        obj.remove("format");
    }

    let required: Vec<String> = match obj.get("required") {
        Some(Value::Array(names)) => names
            .iter()
            .filter_map(|n| n.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    };
    if let Some(Value::Object(props)) = obj.get_mut("properties") {
        for (name, prop) in props.iter_mut() {
            if let Value::Object(prop) = prop
                && !required.contains(name)
            {
                drop_null_arm(prop);
            }
        }
    }

    for_each_subschema(obj, |sub| {
        if let Value::Object(sub) = sub {
            compact_node(sub);
        }
    });
}

/// Removes the `null` alternative from one optional property's schema — see
/// [`compact`].
fn drop_null_arm(prop: &mut JsonObject) {
    let is_null_type = |v: &Value| v.as_str() == Some("null");
    if let Some(Value::Array(types)) = prop.get_mut("type") {
        types.retain(|t| !is_null_type(t));
        if types.len() == 1 {
            let only = types.remove(0);
            prop.insert("type".to_string(), only);
        }
    }
    if let Some(Value::Array(values)) = prop.get_mut("enum") {
        values.retain(|v| !v.is_null());
    }
    for key in ["anyOf", "oneOf"] {
        let Some(Value::Array(arms)) = prop.get_mut(key) else {
            continue;
        };
        let is_null_arm = |arm: &Value| {
            arm.as_object()
                .is_some_and(|a| a.len() == 1 && a.get("type").is_some_and(is_null_type))
        };
        if !arms.iter().any(is_null_arm) {
            continue;
        }
        arms.retain(|arm| !is_null_arm(arm));
        if arms.len() == 1
            && let Some(Value::Object(only)) = arms.pop()
        {
            prop.remove(key);
            // The property's own keywords (its description) win over the arm's.
            let own = std::mem::take(prop);
            *prop = only;
            prop.extend(own);
        }
    }
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

/// Recurses into every schema-valued keyword of `obj` — see
/// [`for_each_subschema`] — resolving refs and replacing boolean subschemas.
fn walk_keywords(obj: &mut JsonObject, defs: &JsonObject, stack: &mut Vec<String>) {
    for_each_subschema(obj, |sub| process_schema_position(sub, defs, stack));
}

/// Calls `visit` on the value in every subschema position of `obj` — the places a
/// JSON Schema can hold one: a single-schema keyword's value (whatever its JSON
/// type, so a boolean subschema reaches `visit`), each value of a map keyword, each
/// element of an array keyword, and an [`OBJECT_ONLY_SUBSCHEMA_KEYS`] value that is
/// an object. The one traversal [`walk_keywords`] and [`compact_node`] share, so
/// the two cannot disagree about where a subschema sits. Deliberately does not touch
/// data-valued keywords (`enum`, `const`, `default`, `examples`, `required`,
/// `type`, ...): a `default` value that happens to contain a `"$ref"` key must
/// stay literal data, not be mistaken for an actual reference.
fn for_each_subschema(obj: &mut JsonObject, mut visit: impl FnMut(&mut Value)) {
    for key in SINGLE_SUBSCHEMA_KEYS {
        if let Some(v) = obj.get_mut(*key) {
            visit(v);
        }
    }
    for key in SUBSCHEMA_MAP_KEYS {
        if let Some(Value::Object(map)) = obj.get_mut(*key) {
            for v in map.values_mut() {
                visit(v);
            }
        }
    }
    for key in SUBSCHEMA_ARRAY_KEYS {
        if let Some(Value::Array(items)) = obj.get_mut(*key) {
            for v in items {
                visit(v);
            }
        }
    }
    for key in OBJECT_ONLY_SUBSCHEMA_KEYS {
        if let Some(v) = obj.get_mut(*key)
            && v.is_object()
        {
            visit(v);
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

    /// Every subschema keyword the traversal handles, with how a subschema is
    /// written in its position: alone (a single-schema keyword, or an object-only
    /// one), as a map's value, or as an array's element. Built from the key lists
    /// themselves, so a dropped or mistyped entry fails every test that walks it.
    type Wrap = fn(Value) -> Value;

    fn position_cases() -> Vec<(&'static str, Wrap)> {
        let alone: fn(Value) -> Value = |sub| sub;
        let in_map: fn(Value) -> Value = |sub| json!({ "a": sub });
        let in_array: fn(Value) -> Value = |sub| json!([sub]);
        let single = SINGLE_SUBSCHEMA_KEYS.iter().map(|k| (*k, alone));
        let map = SUBSCHEMA_MAP_KEYS.iter().map(|k| (*k, in_map));
        let array = SUBSCHEMA_ARRAY_KEYS.iter().map(|k| (*k, in_array));
        let object_only = OBJECT_ONLY_SUBSCHEMA_KEYS.iter().map(|k| (*k, alone));
        single.chain(map).chain(array).chain(object_only).collect()
    }

    #[test]
    fn every_schema_position_keyword_resolves_refs_and_replaces_booleans() {
        // Each keyword the walker claims to handle gets a `$ref` and a boolean
        // subschema in its own position shape (single / map / array), so a
        // typo'd or dropped entry in the keyword lists fails here.
        for (key, in_position) in position_cases() {
            let ref_value = in_position(json!({ "$ref": "#/$defs/S" }));
            // A boolean `additionalProperties` is the conventional form and is left
            // alone by design, so for those keywords the boolean sits a level down.
            let bool_value = if OBJECT_ONLY_SUBSCHEMA_KEYS.contains(&key) {
                json!({ "items": true })
            } else {
                in_position(json!(true))
            };
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
    fn compact_drops_schema_null_defaults_and_nonstandard_formats() {
        let mut schema = obj(json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "limit": { "type": ["integer", "null"], "format": "uint64", "minimum": 0,
                           "default": null },
                "when": { "type": "string", "format": "date-time" },
                "flag": { "type": "boolean", "default": false }
            }
        }));
        compact(&mut schema);
        assert!(!schema.contains_key("$schema"));
        assert_eq!(
            schema["properties"]["limit"],
            json!({ "type": "integer", "minimum": 0 })
        );
        assert_eq!(schema["properties"]["when"]["format"], json!("date-time"));
        assert_eq!(schema["properties"]["flag"]["default"], json!(false));
    }

    #[test]
    fn compact_unwraps_the_null_arm_of_optional_properties_only() {
        let mut schema = obj(json!({
            "type": "object",
            "required": ["must"],
            "properties": {
                "must": { "type": ["string", "null"] },
                "opt": { "type": ["string", "null"], "enum": ["a", "b", null] },
                "obj": {
                    "description": "outer",
                    "anyOf": [
                        { "type": "object", "description": "inner",
                          "properties": { "x": { "type": ["boolean", "null"] } } },
                        { "type": "null" }
                    ]
                },
                "multi": { "anyOf": [{ "type": "string" }, { "type": "integer" },
                                     { "type": "null" }] }
            }
        }));
        compact(&mut schema);
        let props = &schema["properties"];
        assert_eq!(props["must"]["type"], json!(["string", "null"]));
        assert_eq!(
            props["opt"],
            json!({ "type": "string", "enum": ["a", "b"] })
        );
        assert_eq!(props["obj"]["description"], json!("outer"));
        assert_eq!(props["obj"]["type"], json!("object"));
        assert!(props["obj"].get("anyOf").is_none());
        // Nested optional properties are compacted too.
        assert_eq!(props["obj"]["properties"]["x"]["type"], json!("boolean"));
        assert_eq!(
            props["multi"]["anyOf"],
            json!([{ "type": "string" }, { "type": "integer" }])
        );
    }

    #[test]
    fn compact_keeps_the_self_contained_guarantees() {
        let mut schema = obj(json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "foo": { "anyOf": [{ "$ref": "#/$defs/Foo" }, { "type": "null" }] },
                "any": { "type": "array", "items": true }
            },
            "$defs": { "Foo": { "type": "object", "additionalProperties": false,
                                "properties": { "v": true } } }
        }));
        make_self_contained(&mut schema);
        compact(&mut schema);
        let text = serde_json::to_string(&schema).unwrap();
        assert!(!text.contains("$ref") && !text.contains("$defs"), "{text}");
        assert!(!text.contains("null"), "{text}");
        assert!(!text.contains(":true"), "{text}");
        assert_eq!(
            schema["properties"]["foo"]["additionalProperties"],
            json!(false)
        );
    }

    #[test]
    fn compact_reaches_every_schema_position_keyword() {
        // A nullable optional property one level inside each keyword's position: it
        // is compacted only if `compact` descends through that keyword.
        for (key, in_position) in position_cases() {
            let mut schema = obj(json!({
                "type": "object",
                "properties": {
                    "holder": {
                        key: in_position(json!({
                            "properties": {
                                "inner": {
                                    "type": ["string", "null"],
                                    "format": "uint",
                                    "default": null
                                }
                            }
                        }))
                    }
                }
            }));
            let before = serde_json::to_string(&schema).unwrap();
            assert!(
                before.contains(r#""type":["string","null"]"#),
                "'{key}': {before}"
            );

            compact(&mut schema);
            let after = serde_json::to_string(&schema).unwrap();
            assert!(
                after.contains(r#""inner":{"type":"string"}"#),
                "'{key}': not compacted: {after}"
            );
        }
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
