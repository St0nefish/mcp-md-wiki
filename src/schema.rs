//! Directory-scoped frontmatter schemas.
//!
//! A `.schema.yaml` file governs every document at or below its directory, the way
//! `CLAUDE.md` cascades. Deeper files refine shallower ones, so a `recipes/` folder can
//! require `planning.cook_minutes` without that field meaning anything elsewhere.
//!
//! A resolved schema is a **tree** of [`FieldDef`] nodes (#346): an object's child
//! fields hang off its node, and a list's element schema is its `items` node. Paths
//! into it use one grammar, [`FieldPath`]: `a.b` (a child field), `a[]` (the elements
//! of list `a`), `a[].b` (a field of each element). Nested `fields:` and a flat
//! dot-path key are two spellings of the same node, so authoring style never changes
//! the result. Cascading merges node by node and **attribute by attribute**
//! ([`ResolvedSchema::merged_with`]): a deeper scope that redeclares a field only
//! overrides the attributes it writes, and redefining a container never drops its
//! inherited children. Each node records which scope set each attribute.
//!
//! Most consumers read the schema's **flat dot-path view** ([`ResolvedFields`]):
//! every declared field outside a list's `items`, keyed by dot-path
//! (`planning.prep_minutes`), containers included. Fields under `items` (`a[]…`) live
//! only in the tree for now — reachable through [`ResolvedFields::lookup`]. Validation
//! walks the tree, so list elements are checked against them (#347); they are not
//! indexed or offered as filter vocabulary.
//!
//! [`lint_raw`] reports definitions that load but are probably mistakes (a default
//! that fails its own type, say). Those are warnings, never load errors: schema load
//! is fail-fast, so any new strictness check ships as a lint warning first.
//!
//! Deployments with no schema files keep working: the global `frontmatter` block in
//! `config.yaml` becomes the implicit root schema via [`ResolvedSchema::from_config`].
//! This is a **deprecated fallback**, though — a schema describes the knowledge base's
//! own content rules, and `config.yaml` is deployment config that lives on the
//! container host, not in the KB's git repo. A root `.schema.yaml` is the
//! non-deprecated way to declare root rules, and once one exists it is authoritative:
//! it REPLACES the config-derived root outright rather than layering onto it, so a KB
//! carries its root rules with it wherever it is cloned or served, independent of
//! whatever `config.yaml` the deploying host happens to have. `config.yaml`'s
//! `frontmatter` block is consulted only when no root `.schema.yaml` exists at all
//! — see [`SchemaCache::build`].
//!
//! The legacy name `.kb-schema.yaml` ([`LEGACY_SCHEMA_FILE_NAME`]) is still read
//! wherever the canonical name is, so an existing knowledge base keeps working
//! unchanged; `update_schema` migrates one directory at a time by writing
//! `.schema.yaml` and removing the legacy file in the same commit. A directory
//! holding both names is an invalid schema file, never resolved silently. Only a
//! regular file at either name is a schema file: a symlink or any other entry there
//! is absent to the tree walk and to [`SchemaCache::raw_file_at`] alike, so a link
//! pushed to the knowledge base can neither redirect a read nor count toward the
//! both-names rule.
//!
//! Model-facing text (tool descriptions, results, errors) names a schema by its scope
//! directory ([`scope_label`]), never by file name; only operator-facing surfaces
//! (logs, `/status`, the CLI) carry real file paths.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use serde::Deserialize;
use serde_json::Value;
use tracing::{debug, warn};

use crate::config::FrontmatterConfig;
use crate::qdrant::{IndexKind, IndexedField};

/// Filename that declares a schema for its directory and everything beneath it.
/// `update_schema` always writes this name.
pub const SCHEMA_FILE_NAME: &str = ".schema.yaml";

/// The schema file name used before [`SCHEMA_FILE_NAME`]. Still read everywhere the
/// canonical name is; `update_schema` replaces it with the canonical name the next
/// time it edits that directory.
pub const LEGACY_SCHEMA_FILE_NAME: &str = ".kb-schema.yaml";

/// Every file name that declares a schema, canonical first.
pub const SCHEMA_FILE_NAMES: [&str; 2] = [SCHEMA_FILE_NAME, LEGACY_SCHEMA_FILE_NAME];

/// The reason [`SchemaCache::build`] gives for a directory holding both
/// [`SCHEMA_FILE_NAME`] and [`LEGACY_SCHEMA_FILE_NAME`]. Operator-facing; see
/// [`model_facing_reason`] for the wording a model sees instead.
pub(crate) const BOTH_NAMES_REASON: &str = "both .schema.yaml and legacy .kb-schema.yaml \
     are present in this directory; merge them into .schema.yaml and delete the legacy file";

/// `reason` (an [`InvalidSchemaFile::reason`], or [`SchemaCache::raw_file_at`]'s error)
/// as a model should read it. [`BOTH_NAMES_REASON`] names both files, so it is
/// replaced; every other reason (a parse error, the size limit) names none.
pub(crate) fn model_facing_reason(reason: &str) -> &str {
    if reason == BOTH_NAMES_REASON {
        "two schema files declare rules for this directory; an operator must merge them \
         into one"
    } else {
        reason
    }
}

/// How model-facing text names the schema governing KB-relative directory `rel_dir`:
/// the directory with a trailing `/`, and the root as `/`. Never a file name — the
/// file is an implementation detail a model has no use for.
pub fn scope_label(rel_dir: &Path) -> String {
    let dir = rel_dir.to_string_lossy();
    let dir = dir.trim_matches('/');
    if dir.is_empty() {
        "/".to_string()
    } else {
        format!("{dir}/")
    }
}

/// Largest schema file we will attempt to parse.
///
/// Schema files arrive through git sync and are therefore untrusted. Deeply nested
/// YAML costs superlinear time to parse — hundreds of kilobytes can burn seconds of
/// CPU before the parser's own recursion guard even rejects it — and this parse runs
/// on every write, every instructions refresh, and every index run. A real schema is
/// a few kilobytes; anything approaching this cap is not a schema. A file over it
/// is an invalid schema file like any other (see [`SchemaCache::build`]), and
/// `update_schema` refuses to write one.
pub(crate) const MAX_SCHEMA_FILE_BYTES: u64 = 256 * 1024;

/// The reason a schema file of `bytes` bytes is refused for being over
/// [`MAX_SCHEMA_FILE_BYTES`] — the one wording every size check uses
/// ([`SchemaCache::build`], [`parse_schema_text`], [`SchemaCache::raw_file_at`]).
fn over_size_limit_reason(bytes: u64) -> String {
    format!(
        "file is {bytes} bytes, over the {MAX_SCHEMA_FILE_BYTES} byte limit; a schema this \
         large is not parsed"
    )
}

// Declared type of a frontmatter field. Undeclared fields are not type-checked.
//
// `//` rather than `///` on the type and its variants: schemars would turn doc
// comments into a `oneOf` of described `const` branches in `update_schema`'s
// advertised schema; without them it emits a flat string `enum`, and the one
// caller-facing sentence lives on `RawFieldDef::ty`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Deserialize,
    serde::Serialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum FieldType {
    Text,
    Integer,
    Number,
    Boolean,
    // A scalar drawn from a closed set. Also accepts an array, checking each
    // element, as the legacy `config.yaml` `allowed` map did.
    Enum,
    // An array; elements are checked against `values` when present.
    List,
    // `YYYY-MM-DD`.
    Date,
    // RFC 3339 datetime.
    Timestamp,
    // A container for dot-path children rather than a value of its own.
    Object,
}

impl FieldType {
    /// Every type, in declaration order (the order of [`FIELD_KINDS`]).
    pub const ALL: [FieldType; 9] = [
        FieldType::Text,
        FieldType::Integer,
        FieldType::Number,
        FieldType::Boolean,
        FieldType::Enum,
        FieldType::List,
        FieldType::Date,
        FieldType::Timestamp,
        FieldType::Object,
    ];

    /// How this type behaves in every layer that interprets a declared type.
    pub fn kind(self) -> &'static FieldKind {
        &FIELD_KINDS[self as usize]
    }

    /// The name a schema file, `get_schema` and the web UI spell this type with —
    /// the same lowercase name it (de)serializes as.
    pub fn name(self) -> &'static str {
        self.kind().name
    }
}

/// The deepest field path, in [`FieldPath`] steps (each dot-separated name and each
/// `[]` counts as one), that a schema may declare (`update_schema`) and that the
/// metadata projection flattens (`document_fields::flatten_frontmatter`) — one limit,
/// so a field a schema can declare is always a field a filter can reach (#349).
pub const MAX_FIELD_PATH_DEPTH: usize = 16;

/// How a declared [`FieldType`]'s values carry a numeric `value_num` in the
/// metadata projection (`document_fields::flatten_frontmatter`, #349). A field no
/// schema declares keeps the JSON-type guess (`document_fields::numeric_value`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericValue {
    /// No numeric value, even when the JSON value is a number: a `text` field
    /// holding `42` is text.
    None,
    /// The JSON number itself.
    Number,
    /// `1.0` for true, `0.0` for false, as `document_fields::numeric_value` stores a
    /// boolean.
    Boolean,
    /// Seconds since the Unix epoch, UTC (`document_fields::epoch_seconds`): a date
    /// is its midnight UTC, a timestamp is normalized from its offset, so range
    /// filters order dates and timestamps on one axis.
    Epoch,
}

/// The kind of value a [`FieldType`] describes, for the cascade's same-kind rule
/// (#348): a scope may retype an inherited field only within its family, and a
/// change across families is a [`lint_cascade`] warning.
///
/// Conservative by design. `text`, `enum`, `date` and `timestamp` are all a single
/// string scalar (an `enum` is a text field with a closed set, a date a text field
/// with a format), and `integer`/`number` are one numeric kind; a list, an object and
/// a boolean are each their own kind, so scalar↔list, scalar↔object and
/// list↔object changes all warn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KindFamily {
    /// A string scalar: `text`, `enum`, `date`, `timestamp`.
    String,
    /// `integer`, `number`.
    Numeric,
    Boolean,
    List,
    Object,
}

/// Everything a declared [`FieldType`] means to the layers that interpret it (#346):
/// what a value must look like (validation), how its type is named and described,
/// which Qdrant payload index serves it (and so which range filters work on it:
/// [`IndexKind::supports_range`]), how its values reach the metadata projection's
/// `value_num` ([`Self::numeric`], #349), and whether `get_schema` lists the values
/// documents use.
/// One row per type in [`FIELD_KINDS`], read through [`FieldType::kind`], so these
/// mappings cannot drift apart.
#[derive(Debug, Clone, Copy)]
pub struct FieldKind {
    /// The lowercase name; matches the type's serde spelling.
    pub name: &'static str,
    /// Which types a scope may change this one to down the cascade (#348): a change
    /// within one family is a narrowing or widening of the same kind of value; one
    /// across families makes documents valid above invalid below.
    pub family: KindFamily,
    /// The expected value in a type-mismatch message ("expected an integer").
    pub describe: &'static str,
    /// Whether a value satisfies the type ([`check_type`]).
    pub accepts: fn(&Value) -> bool,
    /// The payload index kind an `indexed` field of this type gets.
    pub index: IndexKind,
    /// How a value of this type maps to the metadata projection's numeric
    /// `value_num` (#349); see [`NumericValue`].
    pub numeric: NumericValue,
    /// Whether `get_schema`'s `values_in_use` lists the values documents use for an
    /// open field of this type: free text, timestamps and containers have no
    /// vocabulary worth listing.
    pub vocabulary: bool,
}

/// One row per [`FieldType`], in declaration order (`FieldType as usize` indexes it).
pub static FIELD_KINDS: [FieldKind; 9] = [
    FieldKind {
        name: "text",
        family: KindFamily::String,
        describe: "text",
        accepts: Value::is_string,
        index: IndexKind::Keyword,
        numeric: NumericValue::None,
        vocabulary: false,
    },
    FieldKind {
        name: "integer",
        family: KindFamily::Numeric,
        describe: "an integer",
        accepts: accepts_integer,
        index: IndexKind::Integer,
        numeric: NumericValue::Number,
        vocabulary: true,
    },
    FieldKind {
        name: "number",
        family: KindFamily::Numeric,
        describe: "a number",
        accepts: Value::is_number,
        index: IndexKind::Float,
        numeric: NumericValue::Number,
        vocabulary: true,
    },
    FieldKind {
        name: "boolean",
        family: KindFamily::Boolean,
        describe: "a boolean",
        accepts: Value::is_boolean,
        index: IndexKind::Bool,
        numeric: NumericValue::Boolean,
        vocabulary: true,
    },
    FieldKind {
        name: "enum",
        family: KindFamily::String,
        describe: "a scalar value",
        accepts: accepts_enum,
        index: IndexKind::Keyword,
        numeric: NumericValue::None,
        vocabulary: true,
    },
    FieldKind {
        name: "list",
        family: KindFamily::List,
        describe: "a list",
        accepts: Value::is_array,
        index: IndexKind::Keyword,
        numeric: NumericValue::None,
        vocabulary: true,
    },
    FieldKind {
        name: "date",
        family: KindFamily::String,
        describe: "a date (YYYY-MM-DD)",
        accepts: accepts_date,
        index: IndexKind::Keyword,
        numeric: NumericValue::Epoch,
        vocabulary: true,
    },
    FieldKind {
        name: "timestamp",
        family: KindFamily::String,
        describe: "an RFC 3339 timestamp",
        accepts: accepts_timestamp,
        index: IndexKind::Keyword,
        numeric: NumericValue::Epoch,
        vocabulary: false,
    },
    FieldKind {
        name: "object",
        family: KindFamily::Object,
        describe: "an object",
        accepts: Value::is_object,
        index: IndexKind::Keyword,
        numeric: NumericValue::None,
        vocabulary: false,
    },
];

/// An integer, including one that arrives as an integral float (`3.0`, which YAML
/// parses as a float), but not a fractional one (`3.5`) (#347).
fn accepts_integer(value: &Value) -> bool {
    value.as_i64().is_some()
        || value.as_u64().is_some()
        || value
            .as_f64()
            .is_some_and(|f| f.is_finite() && f.fract() == 0.0)
}

/// Enum accepts a scalar, or an array of scalars — the latter preserves how the
/// pre-cascade `allowed` map treated tag lists.
fn accepts_enum(value: &Value) -> bool {
    match value {
        Value::Array(items) => items.iter().all(|v| !v.is_array() && !v.is_object()),
        other => !other.is_null(),
    }
}

fn accepts_date(value: &Value) -> bool {
    value.as_str().is_some_and(is_date)
}

fn accepts_timestamp(value: &Value) -> bool {
    value.as_str().is_some_and(is_timestamp)
}

/// One step of a [`FieldPath`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PathStep {
    /// A named child field (`.name`, or the leading name).
    Field(String),
    /// The elements of a list (`[]`).
    Items,
}

/// A path into a schema's field tree, in the one grammar every schema path uses
/// (#346): dot-separated names, each optionally followed by `[]` for the elements of
/// that list — `a.b`, `a[]`, `a[].b`, `a[][]`.
///
/// Parsing is total and lossless: a path always starts with a name (possibly
/// empty), only a trailing `[]` on a segment means "elements", and
/// [`FieldPath::render`] gives back exactly the text that was parsed. A name that
/// merely contains brackets elsewhere (`a[0]`, `a[]b`) is just a name. Callers that
/// need stricter names (`update_schema` refuses empty or whitespace-padded ones)
/// check the parsed steps.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FieldPath(Vec<PathStep>);

impl FieldPath {
    pub fn parse(path: &str) -> Self {
        let mut steps = Vec::new();
        for segment in path.split('.') {
            let mut name = segment;
            let mut items = 0usize;
            while let Some(rest) = name.strip_suffix("[]") {
                name = rest;
                items += 1;
            }
            steps.push(PathStep::Field(name.to_string()));
            steps.extend(std::iter::repeat_n(PathStep::Items, items));
        }
        Self(steps)
    }

    pub fn steps(&self) -> &[PathStep] {
        &self.0
    }

    /// Number of steps — each a level of nesting (a name or a `[]`).
    pub fn depth(&self) -> usize {
        self.0.len()
    }

    /// The path's text: names joined by `.`, `[]` appended for each `Items` step.
    pub fn render(&self) -> String {
        render_steps(&self.0)
    }
}

impl std::fmt::Display for FieldPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.render())
    }
}

fn render_steps(steps: &[PathStep]) -> String {
    let mut out = String::new();
    for (i, step) in steps.iter().enumerate() {
        match step {
            PathStep::Field(name) => {
                if i > 0 {
                    out.push('.');
                }
                out.push_str(name);
            }
            PathStep::Items => out.push_str("[]"),
        }
    }
    out
}

/// Placeholder inside a `values:` list that splices in the inherited value set at that
/// position — see [`RawFieldDef::values`] and [`ResolvedSchema::merged_with`] for the
/// full splicing/dedup rules. The `$` prefix is reserved: any other `$`-prefixed token
/// in a `values:` list is a hard error rather than a literal value (see `validate_raw`),
/// so a typo here can never silently degrade into "just another permitted tag."
pub const VALUES_SENTINEL: &str = "$values";

/// A field definition exactly as written in a schema file.
///
/// Also doubles as the shape the `update_schema` MCP tool advertises for `set_field`'s
/// `definition` parameter (see `mcp::FieldDefinitionInput`), via a derived
/// [`schemars::JsonSchema`] impl. `deny_unknown_fields` here becomes
/// `additionalProperties: false` in that advertised schema, so a client's own
/// validation — not just our runtime error — can catch a typo'd key.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, serde::Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RawFieldDef {
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    #[schemars(description = "Type: `enum` is one of `values`, `date` YYYY-MM-DD, \
                              `timestamp` RFC 3339, `object` a container for child fields.")]
    pub ty: Option<FieldType>,
    /// `None` means "not declared here" and inherits the parent scope's `required`
    /// (`false` if there is no parent definition either) — see
    /// [`ResolvedSchema::merged_with`]. This is why the field is `Option<bool>` rather
    /// than a plain `bool` defaulting to `false`: a plain bool cannot distinguish "the
    /// author wrote `required: false`" from "the author said nothing," and per-attribute
    /// inheritance needs that distinction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(description = "Whether this field must be present.")]
    pub required: Option<bool>,
    /// Same absent-means-inherit rule as `required`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(description = "Whether this field is indexed for filtering.")]
    pub indexed: Option<bool>,
    /// Closed set of permitted values, for `enum` and `list`.
    ///
    /// Enforcement strictness is keyed off `ty`, not off whether `values` is present,
    /// and the two regimes are not equivalent: a field with `type: enum` is checked by
    /// [`check_values`] (every scalar, of any JSON type, must canonicalize to a
    /// permitted string); a field that sets `values` but leaves `ty` unset — which is
    /// how every legacy `config.yaml` `allowed` entry arrives, via
    /// [`ResolvedSchema::from_config`], but also any hand-written schema file
    /// field that forgets `type: enum` — is checked by [`check_values_lenient`]
    /// instead, which exempts non-string, non-array values entirely. This is
    /// deliberate (see both functions' docs), not an oversight: it preserves
    /// pre-cascade validation outcomes for configs that never declared types. An
    /// author who wants strict enforcement must write `type: enum` explicitly.
    ///
    /// `None` here inherits the parent's `values` wholesale, same as every other
    /// attribute. `Some(list)` **replaces** the parent's set outright unless `list`
    /// contains the [`VALUES_SENTINEL`] placeholder (`$values`), which splices the
    /// inherited set in at that position — see [`ResolvedSchema::merged_with`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(
        description = "Permitted values (enum/list). Replaces the inherited set \
                              unless it includes `$values`, which keeps it."
    )]
    pub values: Option<Vec<String>>,
    /// **Deprecated** alias for a leading [`VALUES_SENTINEL`]: `extend: true` behaves
    /// exactly like writing `values: [$values, ...]` (see [`ResolvedSchema::merged_with`]
    /// for the exact expansion), and using it logs a warning naming the offending schema
    /// file. Kept only so schema files written before the sentinel existed keep parsing
    /// and cascading correctly; new schemas should write `$values` directly.
    /// `validate_raw` rejects declaring both on the same field — the two ways of saying
    /// "inherit" must not be able to disagree about where the inherited values land.
    /// Not advertised in `update_schema`'s schema (`schemars(skip)`): a caller
    /// should never author it, though it is still accepted.
    #[serde(default, skip_serializing_if = "is_false")]
    #[schemars(skip)]
    pub extend: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(description = "Default value when the field is absent.")]
    pub default: Option<Value>,
    /// For `object`: whether undeclared child keys are permitted. `None` inherits the
    /// parent's `open` (`true`, the same default a fresh top-level declaration gets,
    /// when there is no parent definition either).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(description = "For type object: whether extra child keys are allowed.")]
    pub open: Option<bool>,
    /// Nested child field definitions; a key may itself be a path (`a.b`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(description = "Nested field definitions (dot-path shorthand).")]
    pub fields: Option<BTreeMap<String, RawFieldDef>>,
    /// For `list`: the schema of each element, a scalar definition or an `object`
    /// with its own `fields`. Declaring `items` makes a typeless field a `list`, the
    /// way `fields` makes it an `object`. The same node is reachable as the path
    /// `name[]` (and its fields as `name[].child`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(description = "For type list: the definition of each element.")]
    pub items: Option<Box<RawFieldDef>>,
    /// Inherited values this scope no longer permits (#348): the resolved set is the
    /// inherited one (after this declaration's own `values`, `$values` splice
    /// included) minus these. Like a `values` entry, no item may start with `$`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(description = "Inherited values this directory no longer permits.")]
    pub exclude_values: Option<Vec<String>>,
    /// Removes an inherited field, with everything below it, from this scope and its
    /// descendants (#348); a deeper scope may declare it again. A removing
    /// declaration carries no other attribute (`validate_raw`).
    #[serde(default, skip_serializing_if = "is_false")]
    #[schemars(
        description = "Removes the inherited field here and below; nothing else \
                              may be set with it."
    )]
    pub removed: bool,
}

impl RawFieldDef {
    /// Whether this declaration says nothing at all — no attribute, no children.
    fn is_empty(&self) -> bool {
        *self == RawFieldDef::default()
    }
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// Which scope set each attribute of a resolved [`FieldDef`] (#346), as a
/// [`scope_label`]; `None` when no scope set it (the attribute is at its default).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AttrOrigins {
    pub ty: Option<String>,
    pub required: Option<String>,
    pub indexed: Option<String>,
    pub values: Option<String>,
    pub default: Option<String>,
    pub open: Option<String>,
}

/// A merged field definition: one node of a [`ResolvedSchema`]'s field tree.
///
/// In the flat view ([`ResolvedFields`]) each entry is keyed by its dot-path and
/// carries no `fields` (every child is an entry of its own); `items` stays attached,
/// since the element schema has no flat-view entry.
///
/// Equality compares what the node means — its attributes, `items`, children, and
/// whether it is declared — not provenance (`declared_in`'s label, `set_by`).
#[derive(Debug, Clone)]
pub struct FieldDef {
    pub ty: Option<FieldType>,
    pub required: bool,
    pub indexed: bool,
    /// See [`RawFieldDef::values`]: whether this is enforced strictly or leniently
    /// depends on `ty`, and that split is deliberate.
    pub values: Option<Vec<String>>,
    pub default: Option<Value>,
    pub open: bool,
    /// The element schema of a list.
    pub items: Option<Box<FieldDef>>,
    /// Child fields by name (one path step each).
    pub fields: BTreeMap<String, FieldDef>,
    /// The scope that last declared this field. `None` for an implicit node: one that
    /// exists only because a deeper path was declared (`a.b` alone leaves `a`
    /// implicit), which has default attributes and no flat-view entry.
    pub declared_in: Option<String>,
    /// Which scope set each attribute.
    pub set_by: AttrOrigins,
}

impl Default for FieldDef {
    /// An implicit node: every attribute at the value a field no scope declares gets.
    fn default() -> Self {
        Self {
            ty: None,
            required: false,
            indexed: false,
            values: None,
            default: None,
            open: true,
            items: None,
            fields: BTreeMap::new(),
            declared_in: None,
            set_by: AttrOrigins::default(),
        }
    }
}

impl PartialEq for FieldDef {
    fn eq(&self, other: &Self) -> bool {
        self.ty == other.ty
            && self.required == other.required
            && self.indexed == other.indexed
            && self.values == other.values
            && self.default == other.default
            && self.open == other.open
            && self.items == other.items
            && self.fields == other.fields
            && self.declared_in.is_some() == other.declared_in.is_some()
    }
}

impl FieldDef {
    /// Merge one scope's explicit declaration of this field (`raw`) onto the node as
    /// inherited, recursively for `raw`'s nested `fields` and `items`.
    ///
    /// Per-attribute inheritance: every attribute `raw` leaves unset (`None`) keeps
    /// the inherited value; an implicit node's defaults are exactly what a field no
    /// ancestor declares gets (`required`/`indexed` false, `open` true, nothing
    /// else). `values` is the one attribute with an in-band way to request a merge
    /// instead of a plain override — see [`Self::merge_values`]. A declaration with
    /// nested `fields` and no `type` sets `type: object` (with `items`, `type: list`),
    /// as its own attribute, at this scope. Children the inherited node has and
    /// `raw` does not mention are kept: redefining a container never drops them.
    ///
    /// `origin` is the scope label recorded as `declared_in` and in `set_by`; `path`
    /// only attributes a `warn!` for the deprecated `extend: true` or an
    /// unsatisfiable `$values` sentinel.
    fn merge_node(&mut self, raw: &RawFieldDef, origin: &str, path: &str) {
        let container_ty = if raw.fields.is_some() {
            Some(FieldType::Object)
        } else if raw.items.is_some() {
            Some(FieldType::List)
        } else {
            None
        };
        let label = || Some(origin.to_string());
        if let Some(ty) = raw.ty.or(container_ty) {
            self.ty = Some(ty);
            self.set_by.ty = label();
        }
        if let Some(required) = raw.required {
            self.required = required;
            self.set_by.required = label();
        }
        if let Some(indexed) = raw.indexed {
            self.indexed = indexed;
            self.set_by.indexed = label();
        }
        if raw.extend || raw.values.is_some() {
            self.values = Self::merge_values(raw, self.values.as_deref(), origin, path);
            self.set_by.values = label();
        }
        // After the local `values` (#348): an exclusion narrows whatever this scope
        // ends up with. An unconstrained field has nothing to exclude from
        // (`lint_cascade` says so).
        if let Some(excluded) = &raw.exclude_values
            && let Some(values) = self.values.as_mut()
        {
            values.retain(|v| !excluded.contains(v));
            self.set_by.values = label();
        }
        if let Some(default) = &raw.default {
            self.default = Some(default.clone());
            self.set_by.default = label();
        }
        if let Some(open) = raw.open {
            self.open = open;
            self.set_by.open = label();
        }
        self.declared_in = label();

        for (key, child) in raw.fields.iter().flatten() {
            let steps = FieldPath::parse(key);
            if child.removed {
                remove_node(&mut self.fields, steps.steps());
                continue;
            }
            if excludes_from_nothing(child, &self.fields, steps.steps()) {
                continue;
            }
            node_in_children(&mut self.fields, steps.steps(), origin).merge_node(
                child,
                origin,
                &format!("{path}.{key}"),
            );
        }
        if let Some(items) = &raw.items {
            let undeclared = self.items.as_ref().is_none_or(|n| n.declared_in.is_none());
            if items.removed {
                self.items = None;
            } else if is_exclusion_only(items) && undeclared {
                // Nothing inherited to exclude from: no node (see
                // `excludes_from_nothing`).
            } else {
                self.items.get_or_insert_with(Default::default).merge_node(
                    items,
                    origin,
                    &format!("{path}[]"),
                );
            }
        }
    }

    /// The node at `rest` below this one, if any.
    fn descendant(&self, rest: &[PathStep]) -> Option<&FieldDef> {
        let Some((first, rest)) = rest.split_first() else {
            return Some(self);
        };
        match first {
            PathStep::Field(name) => self.fields.get(name)?.descendant(rest),
            PathStep::Items => self.items.as_deref()?.descendant(rest),
        }
    }

    /// This node as a flat-view entry: without its children, which are entries of
    /// their own.
    fn flat_entry(&self) -> FieldDef {
        FieldDef {
            ty: self.ty,
            required: self.required,
            indexed: self.indexed,
            values: self.values.clone(),
            default: self.default.clone(),
            open: self.open,
            items: self.items.clone(),
            fields: BTreeMap::new(),
            declared_in: self.declared_in.clone(),
            set_by: self.set_by.clone(),
        }
    }

    /// Resolve `raw.values` against the inherited value set.
    ///
    /// - `raw.values` is `None` (the field's `values` is never mentioned at all): plain
    ///   per-attribute inheritance, same as every other attribute — take whatever the
    ///   parent had, verbatim.
    /// - `raw.values` is `Some(list)`: `list` **replaces** the inherited set outright
    ///   — this is the default, deliberately, the same way a shell assignment
    ///   `PATH=/only/this` replaces rather than extends — *unless* `list` contains
    ///   [`VALUES_SENTINEL`] (`$values`), which splices the inherited set in at that
    ///   exact position (`$PATH:/usr/local/bin` names where the inherited part goes).
    ///   The result is deduplicated, keeping the first occurrence of each value, so a
    ///   value listed both explicitly and inherited appears once.
    /// - `raw.extend` (deprecated) is a shorthand for a leading sentinel: `extend: true`
    ///   behaves exactly like `values: [$values, ...raw.values]`. `validate_raw` already
    ///   rejects combining `extend: true` with an explicit `$values` in the same list,
    ///   so at most one of these two paths ever contributes the sentinel.
    fn merge_values(
        raw: &RawFieldDef,
        inherited_values: Option<&[String]>,
        origin: &str,
        path: &str,
    ) -> Option<Vec<String>> {
        if raw.extend {
            warn!(
                scope = %origin,
                field = %path,
                "'extend: true' is deprecated; write 'values: [{VALUES_SENTINEL}, ...]' \
                 instead (see deploy/USAGE.md)"
            );
        }
        let splices = raw.extend || raw.values.iter().flatten().any(|v| v == VALUES_SENTINEL);
        // Loud, not silent (see module docs on the project's general stance): a
        // sentinel with nothing to splice most often means the author expected an
        // ancestor to declare values for this field and it doesn't (a typo'd path, a
        // missing intermediate schema, etc). Resolving to "no values contributed" here
        // — rather than treating the whole `values:` as absent — keeps the field
        // CLOSED (nothing permitted) instead of silently making it unconstrained, so a
        // document that sets this field fails validation immediately and visibly
        // instead of the check quietly stopping enforcement.
        if splices && inherited_values.is_none_or(<[String]>::is_empty) {
            warn!(
                scope = %origin,
                field = %path,
                "'{VALUES_SENTINEL}' has nothing to inherit here (no ancestor \
                 scope declares values for this field); it contributes no \
                 values, so any other literals in this list are the complete \
                 permitted set — declare values on an ancestor, or drop the \
                 sentinel if this list is meant to stand alone"
            );
        }
        splice_values(raw, inherited_values)
    }
}

/// [`FieldDef::merge_values`]'s result without its warnings: `raw`'s `values` (and the
/// deprecated `extend`) resolved against the inherited set, before `exclude_values`.
/// Pure, so the cascade lint and `update_schema`'s patch minimization can ask what a
/// declaration resolves to without logging.
fn splice_values(raw: &RawFieldDef, inherited_values: Option<&[String]>) -> Option<Vec<String>> {
    let tokens: Vec<String> = if raw.extend {
        std::iter::once(VALUES_SENTINEL.to_string())
            .chain(raw.values.iter().flatten().cloned())
            .collect()
    } else {
        match &raw.values {
            Some(list) => list.clone(),
            // Not mentioned at all: inherit the parent's set verbatim, no splicing
            // involved.
            None => return inherited_values.map(<[String]>::to_vec),
        }
    };

    let mut out: Vec<String> = Vec::with_capacity(tokens.len());
    let push_dedup = |out: &mut Vec<String>, v: &str| {
        if !out.iter().any(|existing| existing == v) {
            out.push(v.to_string());
        }
    };
    for token in &tokens {
        if token == VALUES_SENTINEL {
            for v in inherited_values.into_iter().flatten() {
                push_dedup(&mut out, v);
            }
        } else {
            push_dedup(&mut out, token);
        }
    }
    Some(out)
}

/// Whether `raw` says nothing but `exclude_values`.
fn is_exclusion_only(raw: &RawFieldDef) -> bool {
    raw.exclude_values.is_some()
        && RawFieldDef {
            exclude_values: None,
            ..raw.clone()
        }
        .is_empty()
}

/// Whether `raw` (at `steps` under `children`) only excludes values from a field
/// nothing above declares (#348). Such a declaration has nothing to act on, so it
/// creates no node — no declared field, no fingerprint change — and is only
/// [`lint_cascade`]-warned, like a `removed: true` of nothing.
fn excludes_from_nothing(
    raw: &RawFieldDef,
    children: &BTreeMap<String, FieldDef>,
    steps: &[PathStep],
) -> bool {
    if !is_exclusion_only(raw) {
        return false;
    }
    let declared = match steps.split_first() {
        Some((PathStep::Field(name), rest)) => children
            .get(name)
            .and_then(|node| node.descendant(rest))
            .is_some_and(|node| node.declared_in.is_some()),
        _ => false,
    };
    !declared
}

/// Drop the node at `steps` (a parsed [`FieldPath`]) under `children`, with its
/// subtree — what a `removed: true` declaration does (#348). A path that reaches
/// nothing removes nothing; no implicit node is created on the way.
fn remove_node(children: &mut BTreeMap<String, FieldDef>, steps: &[PathStep]) {
    let Some((PathStep::Field(name), rest)) = steps.split_first() else {
        return;
    };
    if rest.is_empty() {
        children.remove(name);
        return;
    }
    if let Some(node) = children.get_mut(name) {
        remove_below(node, rest);
    }
}

/// [`remove_node`] below `node`, for the non-empty `rest` of a path.
fn remove_below(node: &mut FieldDef, rest: &[PathStep]) {
    match rest {
        [] => {}
        [PathStep::Items] => node.items = None,
        [PathStep::Items, tail @ ..] => {
            if let Some(items) = node.items.as_deref_mut() {
                remove_below(items, tail);
            }
        }
        [PathStep::Field(_), ..] => remove_node(&mut node.fields, rest),
    }
}

/// The node at `steps` (a parsed [`FieldPath`], so it starts with a name) under
/// `children`, created — implicit, with default attributes — along with every
/// missing node on the way.
///
/// Except before a `[]` step: a path through a list's elements (`tags[]`,
/// `npcs[].name`) says the node it passes through is a list, so an implicit node
/// there is declared by `origin` as a `type: list` — otherwise an element schema
/// declared only by a flat `tags[]` key would hang off a field with no flat-view
/// entry, invisible to `get_schema` and the fingerprint. A node already declared
/// keeps its own attributes.
fn node_in_children<'a>(
    children: &'a mut BTreeMap<String, FieldDef>,
    steps: &[PathStep],
    origin: &str,
) -> &'a mut FieldDef {
    fn items_of<'n>(node: &'n mut FieldDef, origin: &str) -> &'n mut FieldDef {
        if node.declared_in.is_none() {
            node.declared_in = Some(origin.to_string());
            if node.ty.is_none() {
                node.ty = Some(FieldType::List);
                node.set_by.ty = Some(origin.to_string());
            }
        }
        node.items.get_or_insert_with(Default::default)
    }

    let (first, rest) = steps
        .split_first()
        .expect("a parsed field path has at least one step");
    let name = match first {
        PathStep::Field(name) => name.clone(),
        // A parsed path always starts with a name; an `Items` step here would only
        // come from a hand-built path, and the empty name is what parsing `[]` gives.
        PathStep::Items => String::new(),
    };
    let mut node = children.entry(name).or_default();
    if matches!(first, PathStep::Items) {
        node = items_of(node, origin);
    }
    for step in rest {
        node = match step {
            PathStep::Field(name) => node.fields.entry(name.clone()).or_default(),
            PathStep::Items => items_of(node, origin),
        };
    }
    node
}

/// One parsed schema file, before merging with ancestors.
#[derive(Debug, Clone, Default, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct SchemaFile {
    #[serde(default)]
    pub fields: BTreeMap<String, RawFieldDef>,
    /// Per-directory override of the global `write.dedup_*` near-duplicate gate
    /// (#272). Hand-edited; `update_schema` never writes it but preserves it. A
    /// malformed block (unknown key, wrong type, threshold outside `0.0..=1.0`) is a
    /// schema error like any other.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dedup: Option<RawDedup>,
    /// Frontmatter fields ingest sets from the document's location rather than reads
    /// (#345), by field name. Root schema file only — anywhere else it has no effect
    /// and [`lint_raw`] warns. Absent means nothing is derived. Hand-edited like
    /// `dedup`: `update_schema` never writes it but preserves it.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_derived"
    )]
    pub derived: Option<BTreeMap<String, DerivedField>>,
    /// Whether `update_schema` edits to the root scope need an explicit
    /// acknowledgement (#345). Root schema file only, like `derived`; absent means
    /// root edits are not guarded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_edit: Option<RootEditPolicy>,
}

/// A present `derived:` block, refusing a field named twice: the YAML parser would
/// otherwise keep the last entry and silently drop the other. A null value is
/// absent.
fn deserialize_derived<'de, D>(
    deserializer: D,
) -> Result<Option<BTreeMap<String, DerivedField>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct UniqueMap(BTreeMap<String, DerivedField>);
    impl<'de> Deserialize<'de> for UniqueMap {
        fn deserialize<D2: serde::Deserializer<'de>>(d: D2) -> Result<Self, D2::Error> {
            d.deserialize_map(Unique).map(UniqueMap)
        }
    }
    struct Unique;
    impl<'de> serde::de::Visitor<'de> for Unique {
        type Value = BTreeMap<String, DerivedField>;

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a map of derived field names to rules")
        }

        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> Result<Self::Value, A::Error> {
            let mut out = BTreeMap::new();
            while let Some((name, rule)) = map.next_entry::<String, DerivedField>()? {
                if out.contains_key(&name) {
                    return Err(serde::de::Error::custom(format!(
                        "derived field '{name}' is declared more than once (duplicate key)"
                    )));
                }
                out.insert(name, rule);
            }
            Ok(out)
        }
    }
    // A YAML null (`derived:` with nothing after it, or `derived: ~`) reads as
    // absent, as it does for every other optional block.
    Ok(Option::<UniqueMap>::deserialize(deserializer)?.map(|m| m.0))
}

/// Where a derived field's value comes from (#345).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivedFrom {
    /// The document's `index`-th directory segment, counted from the top-level
    /// folder (index 0). The file name is never a segment.
    PathSegment,
}

/// One entry of a root schema's `derived:` block (#345), e.g.
/// `domain: {from: path_segment, index: 0}`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct DerivedField {
    pub from: DerivedFrom,
    pub index: usize,
}

impl DerivedField {
    /// A `path_segment` rule at `index`.
    #[cfg(test)]
    pub fn path_segment(index: usize) -> Self {
        Self {
            from: DerivedFrom::PathSegment,
            index,
        }
    }

    /// The value this rule gives a document at KB-relative `rel_path`, or `None`
    /// when the path has no such directory segment (a document at the corpus root,
    /// or one not that deep).
    pub fn derive(&self, rel_path: &str) -> Option<String> {
        match self.from {
            DerivedFrom::PathSegment => {
                let components: Vec<_> = Path::new(rel_path).components().collect();
                // The last component is the file name, never a segment.
                let (_, dirs) = components.split_last()?;
                match dirs.get(self.index)? {
                    std::path::Component::Normal(name) => name.to_str().map(str::to_string),
                    _ => None,
                }
            }
        }
    }
}

/// A root schema's `root_edit:` block (#345).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct RootEditPolicy {
    /// Refuse a root-scope `update_schema` edit unless the call passes
    /// `acknowledge_root_change: true`.
    pub require_ack: bool,
    /// A document in the knowledge base that states the root schema's policy, named
    /// in the refusal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_doc: Option<String>,
}

/// Most entries a `derived:` block may hold.
pub const MAX_DERIVED_FIELDS: usize = 16;
/// Highest `index` a `path_segment` rule may name.
pub const MAX_DERIVED_SEGMENT_INDEX: usize = 32;
/// Longest derived field name, in bytes.
pub const MAX_DERIVED_FIELD_NAME_LEN: usize = 64;
/// Longest `root_edit.policy_doc`, in bytes.
pub const MAX_POLICY_DOC_LEN: usize = 256;

/// Check a `derived:` block: a bounded number of plain top-level field names (ASCII
/// letters, digits, `_`, `-`), none of them a name the index stores its own value
/// under or promotes (`title`, `description`), each with a bounded segment index.
fn validate_derived(derived: &BTreeMap<String, DerivedField>) -> Result<(), String> {
    if derived.len() > MAX_DERIVED_FIELDS {
        return Err(format!(
            "derived declares {} fields; at most {MAX_DERIVED_FIELDS} are allowed",
            derived.len()
        ));
    }
    for (name, rule) in derived {
        if name.is_empty()
            || name.len() > MAX_DERIVED_FIELD_NAME_LEN
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(format!(
                "derived field '{name}' must be a top-level field name of 1 to \
                 {MAX_DERIVED_FIELD_NAME_LEN} ASCII letters, digits, '_' or '-'"
            ));
        }
        if RESERVED_PAYLOAD_NAMES.contains(&name.as_str())
            || matches!(name.as_str(), "title" | "description")
        {
            return Err(format!(
                "derived field '{name}' is reserved — the index stores its own value \
                 under that name"
            ));
        }
        if rule.index > MAX_DERIVED_SEGMENT_INDEX {
            return Err(format!(
                "derived field '{name}': index must be at most \
                 {MAX_DERIVED_SEGMENT_INDEX}, got {}",
                rule.index
            ));
        }
    }
    Ok(())
}

/// Check a `root_edit:` block's `policy_doc`: non-empty, bounded, no control
/// characters (it is quoted in a refusal a model reads).
fn validate_root_edit(root_edit: &RootEditPolicy) -> Result<(), String> {
    if let Some(doc) = &root_edit.policy_doc
        && (doc.trim().is_empty()
            || doc.len() > MAX_POLICY_DOC_LEN
            || doc.chars().any(char::is_control))
    {
        return Err(format!(
            "root_edit.policy_doc must be a non-empty path of at most \
             {MAX_POLICY_DOC_LEN} bytes with no control characters"
        ));
    }
    Ok(())
}

/// The `dedup:` block of a schema file. Each key is independently optional;
/// an unset key inherits from the nearest ancestor that sets it, then from `write.*`
/// in `config.yaml` (#272).
#[derive(Debug, Clone, Default, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct RawDedup {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
    /// Overrides `write.related_min_score` (#356).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub related_min_score: Option<f32>,
    /// Overrides `write.related_limit` (#356).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub related_limit: Option<usize>,
}

impl SchemaFile {
    /// Check every definition for internal contradictions.
    pub fn validate_self(&self) -> Result<(), String> {
        if let Some(threshold) = self.dedup.as_ref().and_then(|d| d.threshold)
            && !(0.0..=1.0).contains(&threshold)
        {
            return Err(format!(
                "dedup.threshold must be between 0.0 and 1.0, got {threshold}"
            ));
        }
        if let Some(score) = self.dedup.as_ref().and_then(|d| d.related_min_score)
            && !(0.0..=1.0).contains(&score)
        {
            return Err(format!(
                "dedup.related_min_score must be between 0.0 and 1.0, got {score}"
            ));
        }
        if let Some(limit) = self.dedup.as_ref().and_then(|d| d.related_limit)
            && limit > crate::write::MAX_RELATED_LIMIT
        {
            return Err(format!(
                "dedup.related_limit must be at most {}, got {limit}",
                crate::write::MAX_RELATED_LIMIT
            ));
        }
        if let Some(derived) = &self.derived {
            validate_derived(derived)?;
        }
        if let Some(root_edit) = &self.root_edit {
            validate_root_edit(root_edit)?;
        }
        for (name, raw) in &self.fields {
            validate_raw(name, raw)?;
        }
        // Nested `fields:` (or `items:`) and a flat path key spell the same node, and
        // the merge would silently layer one declaration over the other.
        let mut seen = BTreeSet::new();
        for (name, raw) in &self.fields {
            check_unique_paths(name, raw, &mut seen)?;
        }
        Ok(())
    }
}

/// Reject definitions that contradict themselves, so a schema file carrying one never
/// loads.
fn validate_raw(path: &str, raw: &RawFieldDef) -> Result<(), String> {
    // `removed` and `exclude_values` are new keys (#348), so a malformed use is refused
    // outright rather than linted: no existing schema file can be carrying one.
    if raw.removed
        && *raw
            != (RawFieldDef {
                removed: true,
                ..RawFieldDef::default()
            })
    {
        return Err(format!(
            "field '{path}' sets 'removed: true' alongside other attributes; a removed \
             field is gone from this directory, so it carries nothing else"
        ));
    }
    if let Some(bad) = raw
        .exclude_values
        .iter()
        .flatten()
        .find(|v| v.starts_with('$'))
    {
        return Err(format!(
            "field '{path}' lists '{bad}' in exclude_values; '$'-prefixed tokens are \
             reserved, and excluding the inherited set as a whole is 'values: []'"
        ));
    }

    if raw.fields.is_some()
        && let Some(ty) = raw.ty
        && ty != FieldType::Object
    {
        // Otherwise the field resolves to BOTH a scalar leaf and a set of dot-path
        // children, and any document satisfying the leaf can never satisfy the
        // children — producing an error that blames the document for a broken schema.
        return Err(format!(
            "field '{path}' declares type '{}' but also nested fields; a field is \
             either a value or a container, not both",
            ty.name()
        ));
    }

    if raw.items.is_some() {
        if raw.fields.is_some() {
            return Err(format!(
                "field '{path}' declares both nested fields and items; a field is either \
                 an object or a list, not both"
            ));
        }
        if let Some(ty) = raw.ty
            && ty != FieldType::List
        {
            return Err(format!(
                "field '{path}' declares type '{}' but also items; only a list has items",
                ty.name()
            ));
        }
    }

    if let Some(values) = &raw.values {
        // The `$` prefix is reserved for placeholders. An unrecognized `$`-prefixed
        // token is always a mistake — usually a typo of `$values` — and must be a hard
        // error here, not a literal value: silently accepting it as "just another
        // permitted tag" is exactly the class of quiet failure this cascade otherwise
        // goes out of its way to avoid (see the reserved-token note on
        // [`VALUES_SENTINEL`]).
        if let Some(bad) = values
            .iter()
            .find(|v| v.starts_with('$') && v.as_str() != VALUES_SENTINEL)
        {
            return Err(format!(
                "field '{path}' has an unrecognized placeholder '{bad}' in its values \
                 list; the only recognized '$'-prefixed token is '{VALUES_SENTINEL}'"
            ));
        }

        let sentinel_count = values
            .iter()
            .filter(|v| v.as_str() == VALUES_SENTINEL)
            .count();
        if sentinel_count > 1 {
            return Err(format!(
                "field '{path}' lists '{VALUES_SENTINEL}' {sentinel_count} times; at \
                 most one placeholder is allowed per values list"
            ));
        }

        // `extend: true` is a deprecated alias for a leading `$values` (see
        // `RawFieldDef::extend`); declaring both on the same field is ambiguous about
        // where the inherited values land; refuse it and make the author pick one.
        if raw.extend && sentinel_count > 0 {
            return Err(format!(
                "field '{path}' sets both 'extend: true' and a '{VALUES_SENTINEL}' \
                 placeholder; 'extend' is a deprecated alias for a leading \
                 '{VALUES_SENTINEL}', so combining them is ambiguous — use one or the \
                 other, not both"
            ));
        }
    }

    for (name, child) in raw.fields.iter().flatten() {
        validate_raw(&format!("{path}.{name}"), child)?;
    }
    if let Some(items) = &raw.items {
        validate_raw(&format!("{path}[]"), items)?;
    }
    Ok(())
}

/// Error if two declarations in one file spell the same node — a nested `fields:`
/// (or `items:`) entry and a flat path key, in [`FieldPath`]'s grammar.
fn check_unique_paths(
    path: &str,
    raw: &RawFieldDef,
    seen: &mut BTreeSet<String>,
) -> Result<(), String> {
    if !seen.insert(path.to_string()) {
        return Err(format!(
            "field '{path}' is declared more than once in this file (nested `fields:` \
             and a flat dot-path key spell the same path); keep only one declaration"
        ));
    }
    for (name, child) in raw.fields.iter().flatten() {
        check_unique_paths(&format!("{path}.{name}"), child, seen)?;
    }
    if let Some(items) = &raw.items {
        check_unique_paths(&format!("{path}[]"), items, seen)?;
    }
    Ok(())
}

/// Top-level names the indexer writes into every chunk's Qdrant payload itself
/// (`ingest::upsert_pending`), plus `fm`, reserved for the frontmatter namespace
/// (#350). A frontmatter field of the same name is overwritten in the payload by the
/// built-in value, so a query-mode filter on it never sees what the document says.
pub const RESERVED_PAYLOAD_NAMES: &[&str] = &[
    "file_path",
    "mtime",
    "chunk_index",
    crate::qdrant::CHUNK_TEXT_KEY,
    crate::qdrant::CHUNK_BODY_OFFSET_KEY,
    "line_start",
    "line_end",
    crate::qdrant::PATH_ANCESTORS_KEY,
    crate::qdrant::HEADING_PATH_KEY,
    crate::qdrant::HEADING_LEVEL_KEY,
    crate::qdrant::HEADING_PREFIXES_KEY,
    crate::qdrant::SECTION_KEY,
    crate::qdrant::SECTION_LINE_START_KEY,
    crate::qdrant::SECTION_LINE_END_KEY,
    "fm",
];

/// Definitions in one schema file that load but are probably mistakes (#347).
///
/// Pure, and **warnings only**: nothing here ever stops a schema from loading or an
/// `update_schema` from landing. Schema load is fail-fast, so a new strictness check
/// is introduced as a warning here first. Each warning names the field path (in
/// [`FieldPath`]'s grammar) and the problem. Checks:
///
/// - a `default` that fails the field's own `type` or `values` (checked with the same
///   functions validation uses; a `values` list carrying `$values` or `extend` is
///   skipped, since the inherited part is not in this file, and so is a field whose
///   type is inherited rather than written here);
/// - `values` on an `object` field (no object value can ever match it) or a `boolean`
///   field (where it is at best a roundabout way to forbid one of the two values);
/// - a `default` anywhere under a list's `items` (a `[]` path), which is never
///   applied to list elements;
/// - a top-level field whose name is one of [`RESERVED_PAYLOAD_NAMES`];
/// - a root-only block (`derived`, `root_edit`) in a file that is not the knowledge
///   base's root schema (`is_root` false), where it has no effect (#345).
pub fn lint_raw(file: &SchemaFile, is_root: bool) -> Vec<String> {
    let mut warnings = Vec::new();
    if !is_root {
        for (block, present) in [
            ("derived", file.derived.is_some()),
            ("root_edit", file.root_edit.is_some()),
        ] {
            if present {
                warnings.push(format!(
                    "'{block}' only takes effect in the root schema; here it is ignored"
                ));
            }
        }
    }
    for (key, raw) in &file.fields {
        if let Some(PathStep::Field(name)) = FieldPath::parse(key).steps().first()
            && RESERVED_PAYLOAD_NAMES.contains(&name.as_str())
        {
            warnings.push(format!(
                "field '{key}': '{name}' is reserved — the search index stores its own \
                 value under that name, which hides a frontmatter field of the same \
                 name from search filters; rename the field"
            ));
        }
        lint_field(key, raw, &mut warnings);
    }
    warnings
}

/// Cascade-aware warnings for one schema file (#348): what its declarations do to
/// what the directory inherits, `inherited` being what its ancestors alone resolve to
/// (an empty schema for the root file, which replaces the config-derived root). Like
/// [`lint_raw`], warnings only — the series rule for new strictness checks. Checks:
///
/// - `removed: true` on a field no ancestor declares (it removes nothing);
/// - `exclude_values` on a field that has no value set to exclude from, or naming a
///   value the field does not have at that point (after its own `values`);
/// - `values`/`exclude_values` that no longer permit the default the field inherits
///   (every document omitting the field would be given an invalid value);
/// - a `type` that changes the inherited one to a different [`KindFamily`] — only
///   same-kind changes (`enum` → `text`, `integer` → `number`) are meant to cascade;
///   a scalar becoming a list or an object, or the reverse, makes documents valid
///   above the directory invalid in it.
pub fn lint_cascade(inherited: &ResolvedSchema, file: &SchemaFile) -> Vec<String> {
    let mut warnings = Vec::new();
    for (key, raw) in &file.fields {
        lint_cascade_field(inherited, key, raw, &mut warnings);
    }
    warnings
}

fn lint_cascade_field(
    inherited: &ResolvedSchema,
    path: &str,
    raw: &RawFieldDef,
    warnings: &mut Vec<String>,
) {
    let parent = inherited.fields.lookup(path);
    if raw.removed {
        if parent.is_none() {
            warnings.push(format!(
                "field '{path}': 'removed: true' removes nothing — no parent directory \
                 declares this field"
            ));
        }
        return;
    }

    let ty = raw.ty.or(if raw.fields.is_some() {
        Some(FieldType::Object)
    } else if raw.items.is_some() {
        Some(FieldType::List)
    } else {
        None
    });
    if let (Some(new), Some(old)) = (ty, parent.and_then(|p| p.ty))
        && new.kind().family != old.kind().family
    {
        let from = parent
            .and_then(|p| p.set_by.ty.as_deref())
            .map(|scope| format!(" (set in {scope})"))
            .unwrap_or_default();
        warnings.push(format!(
            "field '{path}': changes the inherited type '{}'{from} to '{}', a different \
             kind of value; only same-kind changes (text/enum/date/timestamp, \
             integer/number) are meant to cascade — documents valid in the parent \
             directory can fail here. This will become an error",
            old.name(),
            new.name()
        ));
    }

    if let Some(excluded) = &raw.exclude_values {
        let parent_values = parent.and_then(|p| p.values.as_deref());
        match splice_values(raw, parent_values) {
            None => warnings.push(format!(
                "field '{path}': exclude_values has no effect — the field has no value \
                 set to exclude from"
            )),
            Some(values) => {
                let missing: Vec<&str> = excluded
                    .iter()
                    .filter(|v| !values.contains(v))
                    .map(String::as_str)
                    .collect();
                if !missing.is_empty() {
                    warnings.push(format!(
                        "field '{path}': exclude_values names {} the field does not \
                         permit here: {}",
                        if missing.len() == 1 {
                            "a value"
                        } else {
                            "values"
                        },
                        missing.join(", ")
                    ));
                }
            }
        }
    }

    // An inherited default this scope's values no longer permit (#348): every
    // document that omits the field would get an invalid value filled in. A default
    // written here is `lint_raw`'s to check.
    if raw.default.is_none()
        && (raw.values.is_some() || raw.extend || raw.exclude_values.is_some())
        && let Some(default) = parent.and_then(|p| p.default.as_ref())
        && let Some(mut values) = splice_values(raw, parent.and_then(|p| p.values.as_deref()))
    {
        if let Some(excluded) = &raw.exclude_values {
            values.retain(|v| !excluded.contains(v));
        }
        let check = match ty.or(parent.and_then(|p| p.ty)) {
            None => check_values_lenient(default, Some(&values)),
            Some(_) => check_values(default, &values),
        };
        if check.is_err() {
            warnings.push(format!(
                "field '{path}': the inherited default {default} is no longer a permitted \
                 value here, so every document that omits the field gets an invalid one; \
                 set a default this directory permits"
            ));
        }
    }

    for (name, child) in raw.fields.iter().flatten() {
        lint_cascade_field(inherited, &format!("{path}.{name}"), child, warnings);
    }
    if let Some(items) = &raw.items {
        lint_cascade_field(inherited, &format!("{path}[]"), items, warnings);
    }
}

fn lint_field(path: &str, raw: &RawFieldDef, warnings: &mut Vec<String>) {
    // The type this declaration gives the field, as `FieldDef::merge_node` resolves
    // it: nested `fields` mean object, `items` mean list.
    let ty = raw.ty.or(if raw.fields.is_some() {
        Some(FieldType::Object)
    } else if raw.items.is_some() {
        Some(FieldType::List)
    } else {
        None
    });

    if let Some(ty @ (FieldType::Object | FieldType::Boolean)) = ty
        && raw.values.is_some()
    {
        warnings.push(format!(
            "field '{path}': 'values' does not belong on {} field (it constrains enum \
             and list fields)",
            ty.kind().describe
        ));
    }

    if raw.default.is_some() && FieldPath::parse(path).steps().contains(&PathStep::Items) {
        // `validate::apply_defaults` never fills in list elements, so a default here
        // does nothing — and next to `required`, it suggests a rule that is not there.
        warnings.push(format!(
            "field '{path}': a default inside a list's items is never applied to its \
             elements; every element must set this field itself"
        ));
    }

    if let (Some(values), Some(excluded)) = (&raw.values, &raw.exclude_values)
        && let Some(both) = values.iter().find(|v| excluded.contains(v))
    {
        warnings.push(format!(
            "field '{path}': '{both}' is listed in both values and exclude_values; the \
             exclusion wins"
        ));
    }

    if let Some(default) = &raw.default {
        if let Some(ty) = ty
            && let Err(reason) = check_type(ty, default)
        {
            warnings.push(format!(
                "field '{path}': the default fails the field's own type ({reason})"
            ));
        } else if let Some(values) = &raw.values
            && !raw.extend
            && !values.iter().any(|v| v == VALUES_SENTINEL)
        {
            let check = match ty {
                None => check_values_lenient(default, Some(values)),
                Some(_) => check_values(default, values),
            };
            if let Err(reason) = check {
                warnings.push(format!(
                    "field '{path}': the default fails the field's own values ({reason})"
                ));
            }
        }
    }

    for (name, child) in raw.fields.iter().flatten() {
        lint_field(&format!("{path}.{name}"), child, warnings);
    }
    if let Some(items) = &raw.items {
        lint_field(&format!("{path}[]"), items, warnings);
    }
}

// Raw-tree navigation for `SchemaFile::apply_inheriting`. A path (`a.b`, `a[]`,
// `a[].b`) is parsed once into [`PathStep`]s; a key in a `fields:` map may itself be
// a multi-step path (`planning.method`, `npcs[]`), so each key is parsed too and
// matched as a whole-step prefix of the target (`plan` is not a prefix of
// `planning.method`).

/// The steps of `steps` below map key `key`, when `key` is a whole-step prefix of
/// them (`Some(&[])` for an exact match).
fn below_key<'s>(steps: &'s [PathStep], key: &str) -> Option<&'s [PathStep]> {
    steps.strip_prefix(FieldPath::parse(key).steps())
}

/// Whether [`find_field_mut`] would find `steps` among `fields`, without needing a
/// mutable borrow.
fn has_field_path(fields: &BTreeMap<String, RawFieldDef>, steps: &[PathStep]) -> bool {
    fields.iter().any(|(key, raw)| match below_key(steps, key) {
        Some([]) => true,
        Some(rest) => has_path_below(raw, rest),
        None => false,
    })
}

/// [`has_field_path`] below a definition, for the non-empty `rest` of a path.
fn has_path_below(raw: &RawFieldDef, rest: &[PathStep]) -> bool {
    match rest.first() {
        None => true,
        Some(PathStep::Items) => raw
            .items
            .as_deref()
            .is_some_and(|items| has_path_below(items, &rest[1..])),
        Some(PathStep::Field(_)) => raw
            .fields
            .as_ref()
            .is_some_and(|children| has_field_path(children, rest)),
    }
}

/// Find the declaration of `steps` among `fields`, whether it is a flat path key or
/// reached through nested `fields:`/`items:` (or a mix, at any level). An exact key
/// wins over a nested declaration of the same path.
fn find_field_mut<'a>(
    fields: &'a mut BTreeMap<String, RawFieldDef>,
    steps: &[PathStep],
) -> Option<&'a mut RawFieldDef> {
    if let Some(exact) = fields
        .keys()
        .find(|key| below_key(steps, key).is_some_and(<[PathStep]>::is_empty))
        .cloned()
    {
        return fields.get_mut(&exact);
    }
    for (key, raw) in fields.iter_mut() {
        if let Some(rest) = below_key(steps, key)
            && let Some(found) = find_path_below_mut(raw, rest)
        {
            return Some(found);
        }
    }
    None
}

/// [`find_field_mut`] below a definition, for the non-empty `rest` of a path.
fn find_path_below_mut<'a>(
    raw: &'a mut RawFieldDef,
    rest: &[PathStep],
) -> Option<&'a mut RawFieldDef> {
    match rest.first()? {
        PathStep::Items => {
            let items = raw.items.as_deref_mut()?;
            if rest.len() == 1 {
                Some(items)
            } else {
                find_path_below_mut(items, &rest[1..])
            }
        }
        PathStep::Field(_) => find_field_mut(raw.fields.as_mut()?, rest),
    }
}

/// An empty `type: object` container, for a missing parent of a nested edit. Every
/// other attribute is left unset so it inherits from an ancestor scope.
fn empty_container() -> RawFieldDef {
    RawFieldDef {
        ty: Some(FieldType::Object),
        fields: Some(BTreeMap::new()),
        ..RawFieldDef::default()
    }
}

/// The missing parent a nested edit creates on the way to its target: an object
/// container when the next step is a name, a `type: list` when it is `[]`.
fn missing_parent(next: &PathStep) -> RawFieldDef {
    match next {
        PathStep::Field(_) => empty_container(),
        PathStep::Items => RawFieldDef {
            ty: Some(FieldType::List),
            ..RawFieldDef::default()
        },
    }
}

/// Like [`find_field_mut`], but declares `steps` (with `leaf()`) when absent,
/// creating any missing parents (see [`missing_parent`]). Errors when a parent on the
/// way cannot hold the next step: a non-object under a name, a non-list under `[]`.
/// `full` is the path the caller asked for, for error messages.
fn field_entry_mut<'a>(
    fields: &'a mut BTreeMap<String, RawFieldDef>,
    steps: &[PathStep],
    full: &str,
    leaf: impl FnOnce() -> RawFieldDef,
) -> Result<&'a mut RawFieldDef, String> {
    if has_field_path(fields, steps) {
        return find_field_mut(fields, steps).ok_or_else(|| unreachable_lookup(full));
    }
    // Longest existing key that is a strict prefix of `steps`; failing that, the
    // first step becomes a new parent. A single step is a plain field of this map.
    let parent = fields
        .keys()
        .filter(|k| below_key(steps, k).is_some_and(|rest| !rest.is_empty()))
        .max_by_key(|k| FieldPath::parse(k).depth())
        .cloned();
    let (parent, rest) = match parent {
        Some(parent) => {
            let depth = FieldPath::parse(&parent).depth();
            (parent, &steps[depth..])
        }
        None => {
            let (PathStep::Field(head), rest) = steps
                .split_first()
                .ok_or_else(|| unreachable_lookup(full))?
            else {
                return Err(unreachable_lookup(full));
            };
            let Some(next) = rest.first() else {
                return Ok(fields.entry(head.clone()).or_insert_with(leaf));
            };
            fields.insert(head.clone(), missing_parent(next));
            (head.clone(), rest)
        }
    };
    let container = fields
        .get_mut(&parent)
        .ok_or_else(|| unreachable_lookup(full))?;
    entry_below(container, &parent, rest, full, leaf)
}

/// [`field_entry_mut`] below definition `def` (spelled `name` in messages), for the
/// non-empty `rest` of a path.
fn entry_below<'a>(
    def: &'a mut RawFieldDef,
    name: &str,
    rest: &[PathStep],
    full: &str,
    leaf: impl FnOnce() -> RawFieldDef,
) -> Result<&'a mut RawFieldDef, String> {
    let Some(first) = rest.first() else {
        return Err(unreachable_lookup(full));
    };
    match first {
        PathStep::Field(_) => {
            if let Some(ty) = def.ty
                && ty != FieldType::Object
            {
                return Err(format!(
                    "'{name}' is a {} field, not a container, so '{full}' cannot be nested \
                     under it",
                    ty.name()
                ));
            }
            // A typeless parent that carries `values:` is a leniently-enforced scalar
            // (see the README); nesting under it would retype it as an object.
            if def.ty.is_none() && def.values.is_some() {
                return Err(format!(
                    "'{name}' is a scalar field with a values list, not a container, so \
                     '{full}' cannot be nested under it"
                ));
            }
            if def.ty.is_none() && def.items.is_some() {
                return Err(format!(
                    "'{name}' is a list field, not a container, so '{full}' cannot be \
                     nested under it"
                ));
            }
            let children = def.fields.get_or_insert_with(BTreeMap::new);
            field_entry_mut(children, rest, full, leaf)
        }
        PathStep::Items => {
            if let Some(ty) = def.ty
                && ty != FieldType::List
            {
                return Err(format!(
                    "'{name}' is a {} field, not a list, so '{full}' cannot describe its \
                     elements",
                    ty.name()
                ));
            }
            if def.ty.is_none() && def.fields.is_some() {
                return Err(format!(
                    "'{name}' is a container, not a list, so '{full}' cannot describe its \
                     elements"
                ));
            }
            // A typeless parent that carries `values:` is a leniently-enforced scalar;
            // giving it `items` would retype it as a list. A parent whose type is only
            // inherited from an ancestor scope (an `enum` declared above, say) is not
            // caught here: the edit sees this file alone, not the resolved
            // definitions of the path's parents.
            if def.ty.is_none() && def.values.is_some() {
                return Err(format!(
                    "'{name}' is a scalar field with a values list, not a list, so '{full}' \
                     cannot describe its elements"
                ));
            }
            let Some(next) = rest.get(1) else {
                return Ok(def.items.get_or_insert_with(|| Box::new(leaf())));
            };
            let items = def
                .items
                .get_or_insert_with(|| Box::new(missing_parent(next)));
            entry_below(items, &format!("{name}[]"), &rest[1..], full, leaf)
        }
    }
}

fn unreachable_lookup(full: &str) -> String {
    format!("internal error: lost track of field '{full}' while editing the schema")
}

/// Remove the declaration of `steps`, wherever it nests. Returns whether one was
/// found. An emptied container stays: it may still carry `open: false`.
fn remove_field_path(fields: &mut BTreeMap<String, RawFieldDef>, steps: &[PathStep]) -> bool {
    if let Some(exact) = fields
        .keys()
        .find(|key| below_key(steps, key).is_some_and(<[PathStep]>::is_empty))
        .cloned()
    {
        fields.remove(&exact);
        return true;
    }
    for (key, raw) in fields.iter_mut() {
        if let Some(rest) = below_key(steps, key)
            && remove_path_below(raw, rest)
        {
            return true;
        }
    }
    false
}

/// [`remove_field_path`] below a definition, for the non-empty `rest` of a path.
fn remove_path_below(raw: &mut RawFieldDef, rest: &[PathStep]) -> bool {
    match rest.first() {
        None => false,
        Some(PathStep::Items) if rest.len() == 1 => raw.items.take().is_some(),
        Some(PathStep::Items) => raw
            .items
            .as_deref_mut()
            .is_some_and(|items| remove_path_below(items, &rest[1..])),
        Some(PathStep::Field(_)) => raw
            .fields
            .as_mut()
            .is_some_and(|children| remove_field_path(children, rest)),
    }
}

/// A constrained edit to a schema file.
///
/// Deliberately not free-form text: an invalid schema file stops the server from
/// starting and is refused by a running one, so callers describe intent and the
/// server renders the YAML.
#[derive(Debug, Clone)]
pub enum SchemaEdit {
    /// Add values to a field's permitted set, creating the field if absent. A value
    /// this scope excludes (`exclude_values`) is permitted again by dropping the
    /// exclusion (#348).
    AddValues { field: String, values: Vec<String> },
    /// Remove values from a field's permitted set: a value in this scope's own list
    /// is dropped from it, an inherited one is excluded (`exclude_values`, #348).
    RemoveValues { field: String, values: Vec<String> },
    /// Declare or replace a field definition outright.
    SetField {
        field: String,
        definition: Box<RawFieldDef>,
    },
    /// Remove a field declaration from this scope.
    RemoveField { field: String },
    /// Merge-patch this scope's declaration of a field (#348): only the keys in
    /// `patch` change, nested `fields` (and `items`) patch recursively, and a `null`
    /// drops the local setting so the attribute is inherited again. The result is
    /// minimized against the inherited definition — see
    /// [`SchemaFile::apply_inheriting`].
    PatchField {
        field: String,
        patch: serde_json::Map<String, Value>,
    },
}

impl SchemaEdit {
    /// The dot-path field this edit targets.
    pub fn field(&self) -> &str {
        match self {
            SchemaEdit::AddValues { field, .. }
            | SchemaEdit::RemoveValues { field, .. }
            | SchemaEdit::SetField { field, .. }
            | SchemaEdit::PatchField { field, .. }
            | SchemaEdit::RemoveField { field } => field,
        }
    }

    /// The operation's name, as `update_schema` spells it.
    pub fn operation(&self) -> &'static str {
        match self {
            SchemaEdit::AddValues { .. } => "add_values",
            SchemaEdit::RemoveValues { .. } => "remove_values",
            SchemaEdit::SetField { .. } => "set_field",
            SchemaEdit::RemoveField { .. } => "remove_field",
            SchemaEdit::PatchField { .. } => "patch_field",
        }
    }
}

/// Apply a field-definition merge patch to `target` (a definition as JSON), for
/// [`SchemaEdit::PatchField`]: each key in `patch` replaces the same key, `null`
/// removes it, and `fields` (per child) and `items` patch recursively instead of
/// being replaced. Shapes are checked when the result is read back as a
/// [`RawFieldDef`].
fn patch_definition(
    target: &mut serde_json::Map<String, Value>,
    patch: &serde_json::Map<String, Value>,
) {
    for (key, value) in patch {
        match (key.as_str(), value) {
            (_, Value::Null) => {
                target.remove(key);
            }
            ("fields", Value::Object(children)) => {
                let entry = target
                    .entry(key.clone())
                    .or_insert_with(|| Value::Object(Default::default()));
                if !entry.is_object() {
                    *entry = Value::Object(Default::default());
                }
                let existing = entry.as_object_mut().expect("made an object above");
                for (name, child) in children {
                    match child {
                        Value::Null => {
                            existing.remove(name);
                        }
                        Value::Object(child_patch) => {
                            let slot = existing
                                .entry(name.clone())
                                .or_insert_with(|| Value::Object(Default::default()));
                            if !slot.is_object() {
                                *slot = Value::Object(Default::default());
                            }
                            patch_definition(
                                slot.as_object_mut().expect("made an object above"),
                                child_patch,
                            );
                        }
                        other => {
                            existing.insert(name.clone(), other.clone());
                        }
                    }
                }
            }
            ("items", Value::Object(items_patch)) => {
                let entry = target
                    .entry(key.clone())
                    .or_insert_with(|| Value::Object(Default::default()));
                if !entry.is_object() {
                    *entry = Value::Object(Default::default());
                }
                patch_definition(
                    entry.as_object_mut().expect("made an object above"),
                    items_patch,
                );
            }
            _ => {
                target.insert(key.clone(), value.clone());
            }
        }
    }
}

/// Drop from `def` every attribute that resolves to what `inherited` already has
/// (#348), recursively through `fields` and `items`, so a patched declaration states
/// only what this scope changes. `inherited` is the field as the scope's ancestors
/// resolve it (`None`: nothing inherited, compared against a field no scope declares).
/// The resolved definition is unchanged by construction: every dropped attribute is
/// inherited with the same value. A child that ends up saying nothing and is
/// inherited anyway is dropped too. `removed`, `extend` and `exclude_values` are left
/// as written (an empty `exclude_values` excepted, which excludes nothing).
fn minimize_against(def: &mut RawFieldDef, inherited: Option<&FieldDef>) {
    if def.removed {
        return;
    }
    let fallback = FieldDef::default();
    let inh = inherited.unwrap_or(&fallback);

    if def.ty.is_some() && def.ty == inh.ty {
        def.ty = None;
    }
    if def.required == Some(inh.required) {
        def.required = None;
    }
    if def.indexed == Some(inh.indexed) {
        def.indexed = None;
    }
    if def.open == Some(inh.open) {
        def.open = None;
    }
    if def.default.is_some() && def.default == inh.default {
        def.default = None;
    }
    if def.exclude_values.as_ref().is_some_and(Vec::is_empty) {
        def.exclude_values = None;
    }
    if def.values.is_some() && !def.extend && def.exclude_values.is_none() {
        let as_set = |v: Option<Vec<String>>| v.map(|v| v.into_iter().collect::<BTreeSet<_>>());
        if as_set(splice_values(def, inh.values.as_deref())) == as_set(inh.values.clone()) {
            def.values = None;
        }
    }

    if let Some(children) = def.fields.as_mut() {
        children.retain(|key, child| {
            let path = FieldPath::parse(key);
            let child_inh = inh
                .descendant(path.steps())
                .filter(|n| n.declared_in.is_some());
            minimize_against(child, child_inh);
            !(child.is_empty() && child_inh.is_some())
        });
        // An emptied `fields:` only still matters as what makes this an object.
        if children.is_empty() && (def.ty.is_some() || inh.ty == Some(FieldType::Object)) {
            def.fields = None;
        }
    }
    if let Some(items) = def.items.as_deref_mut() {
        let items_inh = inh.items.as_deref().filter(|n| n.declared_in.is_some());
        minimize_against(items, items_inh);
        if items.is_empty() && items_inh.is_some() {
            def.items = None;
        }
    }
}

impl SchemaFile {
    /// Apply an edit with no inherited definition in play — see
    /// [`Self::apply_inheriting`].
    #[cfg(test)]
    pub fn apply(&mut self, edit: &SchemaEdit) -> Result<String, String> {
        self.apply_inheriting(edit, None)
    }

    /// Apply an edit, returning a description of what changed.
    ///
    /// `inherited` is the definition the edited field resolves to from this scope's
    /// ancestors alone (`None` when no ancestor declares it). `AddValues` uses it so a
    /// child scope extends the inherited set rather than narrowing it: with no local
    /// `values`, the new list starts with [`VALUES_SENTINEL`], and a field created
    /// here keeps the inherited `type` instead of being forced to `enum` — unless no
    /// ancestor gives it a type or values, in which case it is `enum`, as a brand-new
    /// field is. `RemoveValues` uses it to tell a value this scope lists from one it
    /// inherits, which it excludes (`exclude_values`) instead (#348). `PatchField`
    /// minimizes the patched declaration against it: an attribute equal to the
    /// inherited one is dropped, and a declaration left saying nothing is removed.
    pub fn apply_inheriting(
        &mut self,
        edit: &SchemaEdit,
        inherited: Option<&FieldDef>,
    ) -> Result<String, String> {
        let path = FieldPath::parse(edit.field());
        let steps = path.steps();
        // A field this scope removes (`removed: true`, #348) inherits nothing here, so
        // a values edit or a patch has nothing to act on: treating the declaration as
        // an ordinary one would read its absent `values` as "inherits the set" (and a
        // patch would drop `removed` silently). Re-declaring it is a `set_field`.
        if matches!(
            edit,
            SchemaEdit::AddValues { .. }
                | SchemaEdit::RemoveValues { .. }
                | SchemaEdit::PatchField { .. }
        ) && find_field_mut(&mut self.fields, steps).is_some_and(|def| def.removed)
        {
            return Err(format!(
                "'{}' is removed in this scope; re-declare it with set_field first",
                edit.field()
            ));
        }
        match edit {
            SchemaEdit::AddValues { field, values } => {
                // What this scope inherits for the field. A local list only carries it
                // when it splices it in (a sentinel or the deprecated `extend`); one
                // without replaces the inherited set outright.
                let (local_exists, local_values, inherits, local_excluded) =
                    match find_field_mut(&mut self.fields, steps) {
                        Some(def) => (
                            true,
                            def.values.clone(),
                            def.extend
                                || def
                                    .values
                                    .as_ref()
                                    .is_none_or(|l| l.iter().any(|v| v == VALUES_SENTINEL)),
                            def.exclude_values.clone().unwrap_or_default(),
                        ),
                        None => (false, None, true, Vec::new()),
                    };
                let inherited_values: &[String] = inherited
                    .and_then(|d| d.values.as_deref())
                    .filter(|_| inherits)
                    .unwrap_or(&[]);
                // A value already permitted through inheritance needs no local entry:
                // writing one would only add a redundant declaration. With nothing
                // else to add, a scope that declares nothing yet is left alone (the
                // caller would otherwise commit an empty schema file). A value this
                // scope excludes is not permitted, whatever it inherits.
                if values.iter().all(|v| {
                    (local_values.as_ref().is_some_and(|l| l.contains(v))
                        || inherited_values.contains(v))
                        && !local_excluded.contains(v)
                }) {
                    return if local_exists {
                        Ok(format!(
                            "'{}' already permitted every requested value",
                            field
                        ))
                    } else {
                        Err(format!(
                            "'{}' already permits every requested value through an ancestor's \
                             schema; there is nothing to add here",
                            field
                        ))
                    };
                }
                let def = field_entry_mut(&mut self.fields, steps, field, || RawFieldDef {
                    // `enum` unless an ancestor already supplies a type or a values
                    // list, which this declaration keeps through per-attribute
                    // inheritance. A brand-new field qualifies, and so does one an
                    // ancestor only declares attributes of (`required: true`): left
                    // typeless, its new list would be enforced leniently and let
                    // non-string scalars (`status: 3`) through.
                    ty: inherited
                        .is_none_or(|d| d.ty.is_none() && d.values.is_none())
                        .then_some(FieldType::Enum),
                    // Left unset rather than `Some(false)`/`Some(true)`: this
                    // scope may not be the field's first declaration, and a brand
                    // new definition created just to add a value must not clobber
                    // whatever an ancestor scope already said about `required`,
                    // `indexed`, or `open` for this field — see per-attribute
                    // inheritance in `ResolvedSchema::merged_with`.
                    ..RawFieldDef::default()
                })?;
                // No local list: a fresh one would REPLACE the inherited set, so it
                // splices that set in first. `extend: true` already means a leading
                // sentinel, and `validate_raw` refuses the two together.
                let splice_inherited = !def.extend && !inherited_values.is_empty();
                let mut added = Vec::new();
                for value in values {
                    // Permitting a value this scope excludes drops the exclusion
                    // (#348); one it already lists or inherits needs nothing more.
                    let was_excluded = def.exclude_values.as_mut().is_some_and(|excluded| {
                        let before = excluded.len();
                        excluded.retain(|v| v != value);
                        excluded.len() != before
                    });
                    if def.values.as_ref().is_some_and(|l| l.contains(value))
                        || inherited_values.contains(value)
                    {
                        if was_excluded {
                            added.push(value.clone());
                        }
                        continue;
                    }
                    // No local list: a fresh one would REPLACE the inherited set, so
                    // it splices that set in first. `extend: true` already means a
                    // leading sentinel, and `validate_raw` refuses the two together.
                    def.values
                        .get_or_insert_with(|| {
                            if splice_inherited {
                                vec![VALUES_SENTINEL.to_string()]
                            } else {
                                Vec::new()
                            }
                        })
                        .push(value.clone());
                    added.push(value.clone());
                }
                if def.exclude_values.as_ref().is_some_and(Vec::is_empty) {
                    def.exclude_values = None;
                    // A declaration that only excluded values now says nothing.
                    if def.is_empty() {
                        remove_field_path(&mut self.fields, steps);
                        return Ok(format!("added to '{}': {}", field, added.join(", ")));
                    }
                }
                if let Some(existing) = def.values.as_mut() {
                    // Sort only what follows the sentinel, so the inherited values keep
                    // their position in the spliced result.
                    let start = existing
                        .iter()
                        .position(|v| v == VALUES_SENTINEL)
                        .map_or(0, |i| i + 1);
                    existing[start..].sort();
                }
                if added.is_empty() {
                    Ok(format!(
                        "'{}' already permitted every requested value",
                        field
                    ))
                } else {
                    Ok(format!("added to '{}': {}", field, added.join(", ")))
                }
            }
            SchemaEdit::RemoveValues { field, values } => {
                // What this scope inherits for the field: only when its own list (if
                // any) splices the inherited set in, as for `AddValues`.
                let local = find_field_mut(&mut self.fields, steps);
                let local_exists = local.is_some();
                let (inherits, local_has_list, already_excluded) = match &local {
                    Some(def) => (
                        def.extend
                            || def
                                .values
                                .as_ref()
                                .is_none_or(|l| l.iter().any(|v| v == VALUES_SENTINEL)),
                        def.values.is_some(),
                        def.exclude_values.clone().unwrap_or_default(),
                    ),
                    None => (true, false, Vec::new()),
                };
                let inherited_values: &[String] = inherited
                    .and_then(|d| d.values.as_deref())
                    .filter(|_| inherits)
                    .unwrap_or(&[]);
                if !local_exists && inherited_values.is_empty() {
                    return Err(format!("field '{}' is not declared in this scope", field));
                }
                if !local_has_list && inherited_values.is_empty() {
                    return Err(format!("field '{}' has no value list", field));
                }
                // An inherited value is not in this file to delete: excluding it is how
                // this scope stops permitting it (#348).
                let to_exclude: Vec<String> = values
                    .iter()
                    .filter(|v| inherited_values.contains(v) && !already_excluded.contains(v))
                    .cloned()
                    .collect();
                let def = match local {
                    Some(def) => def,
                    None if to_exclude.is_empty() => {
                        return Ok(format!("removed 0 value(s) from '{}'", field));
                    }
                    None => field_entry_mut(&mut self.fields, steps, field, RawFieldDef::default)?,
                };
                // Each requested value counts once, whether it left this scope's own
                // list, was excluded, or both (listed here and spliced in).
                let mut gone: BTreeSet<String> = BTreeSet::new();
                if let Some(existing) = def.values.as_mut() {
                    existing.retain(|v| {
                        let drop = v != VALUES_SENTINEL && values.contains(v);
                        if drop {
                            gone.insert(v.clone());
                        }
                        !drop
                    });
                }
                let newly_excluded = to_exclude.len();
                if !to_exclude.is_empty() {
                    gone.extend(to_exclude.iter().cloned());
                    let excluded = def.exclude_values.get_or_insert_with(Vec::new);
                    excluded.extend(to_exclude);
                    excluded.sort();
                    excluded.dedup();
                }
                if newly_excluded > 0 {
                    Ok(format!(
                        "removed {} value(s) from '{field}' ({newly_excluded} inherited, now \
                         excluded here)",
                        gone.len()
                    ))
                } else {
                    Ok(format!("removed {} value(s) from '{field}'", gone.len()))
                }
            }
            SchemaEdit::SetField { field, definition } => {
                // The placeholder is overwritten immediately; only the slot's location
                // (and its missing parents) matter here.
                let slot = field_entry_mut(&mut self.fields, steps, field, empty_container)?;
                *slot = definition.as_ref().clone();
                Ok(format!("declared '{}'", field))
            }
            SchemaEdit::PatchField { field, patch } => {
                let current = find_field_mut(&mut self.fields, steps).map(|def| def.clone());
                let mut merged = match &current {
                    Some(def) => match serde_json::to_value(def) {
                        Ok(Value::Object(map)) => map,
                        _ => return Err(unreachable_lookup(field)),
                    },
                    None => serde_json::Map::new(),
                };
                patch_definition(&mut merged, patch);
                let mut patched: RawFieldDef = serde_json::from_value(Value::Object(merged))
                    .map_err(|e| format!("invalid patch for '{field}': {e}"))?;
                minimize_against(&mut patched, inherited);

                if patched.is_empty() && inherited.is_some() {
                    // Everything this scope said is now what it inherits anyway.
                    return if current.is_some() {
                        remove_field_path(&mut self.fields, steps);
                        Ok(format!(
                            "'{field}' now inherits its whole definition (local declaration \
                             removed)"
                        ))
                    } else {
                        Err(format!(
                            "'{field}' already resolves to this through an ancestor's \
                             schema; there is nothing to change here"
                        ))
                    };
                }
                if current.as_ref() == Some(&patched) {
                    return Ok(format!("'{field}' unchanged"));
                }
                let slot = field_entry_mut(&mut self.fields, steps, field, empty_container)?;
                *slot = patched;
                Ok(format!("patched '{field}'"))
            }
            SchemaEdit::RemoveField { field } => {
                if !remove_field_path(&mut self.fields, steps) {
                    return Err(format!("field '{}' is not declared in this scope", field));
                }
                Ok(format!("removed declaration of '{}'", field))
            }
        }
    }

    /// Render back to YAML for writing to disk.
    pub fn to_yaml(&self) -> Result<String, String> {
        // `fields` is a BTreeMap at every level, so key order (and a rewrite's diff) is
        // deterministic.
        let doc = serde_yaml_ng::to_string(self)
            .map_err(|e| format!("could not serialize schema: {e}"))?;
        Ok(format!(
            "# Frontmatter schema for this directory and everything beneath it.\n\
             # Managed by the update_schema MCP tool; hand edits are fine but must stay\n\
             # valid — a malformed file stops the server from starting, and a running\n\
             # server refuses it and keeps the previous schema.\n{doc}"
        ))
    }
}

/// A resolved schema's fields: the field tree, and the flat dot-path view most
/// consumers read (#346).
///
/// The flat view holds every declared field outside a list's `items`, keyed by
/// dot-path (`planning.prep_minutes`) — containers included, as entries of their own
/// with no children attached ([`FieldDef`]'s flat-view shape). Implicit nodes (only
/// a deeper path was declared) and everything under `items` (`a[]`, `a[].b`) are
/// absent from it: those are reached through [`Self::lookup`] or [`Self::tree`]. The
/// map-like methods (`get`, `contains_key`, `keys`, iteration, indexing) all read the
/// flat view. Both views, and the per-field origin map, are built together from the
/// tree, so they cannot disagree.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolvedFields {
    tree: BTreeMap<String, FieldDef>,
    flat: BTreeMap<String, FieldDef>,
    origin: BTreeMap<String, String>,
}

impl ResolvedFields {
    fn from_tree(tree: BTreeMap<String, FieldDef>) -> Self {
        let mut flat = BTreeMap::new();
        let mut origin = BTreeMap::new();
        for (name, node) in &tree {
            collect_flat(name, node, &mut flat, &mut origin);
        }
        Self { tree, flat, origin }
    }

    /// The top-level fields of the tree, by name.
    pub fn tree(&self) -> &BTreeMap<String, FieldDef> {
        &self.tree
    }

    /// The declared field at `path` in [`FieldPath`]'s grammar — `a.b`, and also
    /// `a[]` and `a[].b`, which the flat view does not hold. Unlike the flat view's
    /// entries, the node comes with its children. `None` when nothing declares it.
    pub fn lookup(&self, path: &str) -> Option<&FieldDef> {
        let path = FieldPath::parse(path);
        let (PathStep::Field(name), rest) = path.steps().split_first()? else {
            return None;
        };
        self.tree
            .get(name)?
            .descendant(rest)
            .filter(|node| node.declared_in.is_some())
    }

    pub fn get<Q>(&self, path: &Q) -> Option<&FieldDef>
    where
        String: std::borrow::Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.flat.get(path)
    }

    pub fn contains_key<Q>(&self, path: &Q) -> bool
    where
        String: std::borrow::Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.flat.contains_key(path)
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.flat.keys()
    }

    pub fn values(&self) -> impl Iterator<Item = &FieldDef> {
        self.flat.values()
    }

    pub fn iter(&self) -> std::collections::btree_map::Iter<'_, String, FieldDef> {
        self.flat.iter()
    }

    pub fn len(&self) -> usize {
        self.flat.len()
    }

    pub fn is_empty(&self) -> bool {
        self.flat.is_empty()
    }
}

impl<'a> IntoIterator for &'a ResolvedFields {
    type Item = (&'a String, &'a FieldDef);
    type IntoIter = std::collections::btree_map::Iter<'a, String, FieldDef>;

    fn into_iter(self) -> Self::IntoIter {
        self.flat.iter()
    }
}

impl<Q> std::ops::Index<&Q> for ResolvedFields
where
    String: std::borrow::Borrow<Q>,
    Q: Ord + ?Sized,
{
    type Output = FieldDef;

    fn index(&self, path: &Q) -> &FieldDef {
        self.flat
            .get(path)
            .expect("no field at this path in the flat view")
    }
}

/// Add `node` (at dot-path `path`) and its declared descendants to the flat view.
/// `items` subtrees are left out: paths through a list's elements are not part of the
/// flat view (#346).
fn collect_flat(
    path: &str,
    node: &FieldDef,
    flat: &mut BTreeMap<String, FieldDef>,
    origin: &mut BTreeMap<String, String>,
) {
    if let Some(declared_in) = &node.declared_in {
        flat.insert(path.to_string(), node.flat_entry());
        origin.insert(path.to_string(), declared_in.clone());
    }
    for (name, child) in &node.fields {
        collect_flat(&format!("{path}.{name}"), child, flat, origin);
    }
}

/// A fully merged schema for one directory.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolvedSchema {
    /// The field tree and its flat dot-path view; see [`ResolvedFields`].
    pub fields: ResolvedFields,
    /// Near-duplicate gate override from the `dedup:` cascade (#272). `None` means
    /// "use `write.dedup_enabled`": the cache is built without `WriteConfig`, so the
    /// global value is resolved at the gate, not baked in here.
    pub dedup_enabled: Option<bool>,
    /// As `dedup_enabled`, for `write.dedup_threshold`.
    pub dedup_threshold: Option<f32>,
    /// As `dedup_enabled`, for `write.related_min_score` (#356).
    pub related_min_score: Option<f32>,
    /// As `dedup_enabled`, for `write.related_limit` (#356).
    pub related_limit: Option<usize>,
    /// The root schema file's `derived:` block (#345), inherited unchanged by every
    /// scope below it; empty when there is no root schema file or it declares none.
    pub derived: BTreeMap<String, DerivedField>,
    /// The root schema file's `root_edit:` block (#345), inherited like `derived`.
    pub root_edit: Option<RootEditPolicy>,
}

impl ResolvedSchema {
    /// The fields ingest derives from a document's location (#345) rather than reads.
    pub fn derived_field_names(&self) -> impl Iterator<Item = &str> {
        self.derived.keys().map(String::as_str)
    }

    /// Whether ingest derives `field` (#345).
    pub fn is_derived(&self, field: &str) -> bool {
        self.derived.contains_key(field)
    }

    /// Which scope last declared each flat-view field — the scope that contributed its
    /// current definition (the [`FieldDef::declared_in`] of each entry). Drives error
    /// messages and `get_schema` provenance.
    pub fn origin(&self) -> &BTreeMap<String, String> {
        &self.fields.origin
    }

    /// Adapt the global `frontmatter` config block into the implicit root schema.
    ///
    /// **Deprecated fallback**, used only when the knowledge base has no root
    /// schema file of its own — see [`SchemaCache::build`] and the module docs.
    /// Lossless with respect to the pre-cascade behavior: `allowed` becomes `enum`
    /// fields, which still accept either a scalar or an array of scalars.
    pub fn from_config(config: &FrontmatterConfig) -> Self {
        // The root scope's label, like a root schema file's: where the rules come
        // from on the host is deployment detail, not something a caller can locate
        // or edit.
        let root = scope_label(Path::new(""));
        let mut tree: BTreeMap<String, FieldDef> = BTreeMap::new();
        fn declare<'t>(
            tree: &'t mut BTreeMap<String, FieldDef>,
            name: &str,
            root: &str,
        ) -> &'t mut FieldDef {
            let node = node_in_children(tree, FieldPath::parse(name).steps(), root);
            node.declared_in = Some(root.to_string());
            node
        }

        for name in &config.required {
            let node = declare(&mut tree, name, &root);
            node.required = true;
            node.set_by.required = Some(root.clone());
        }
        for name in &config.indexed_fields {
            let node = declare(&mut tree, name, &root);
            node.indexed = true;
            node.set_by.indexed = Some(root.clone());
        }
        for (name, default) in &config.defaults {
            let node = declare(&mut tree, name, &root);
            node.default = Some(Value::String(default.clone()));
            node.set_by.default = Some(root.clone());
        }
        for (name, values) in &config.allowed {
            // Deliberately leaves `ty` unset. Declaring these `Enum` would subject
            // numbers, booleans, and non-string array elements to value checking that
            // the pre-cascade validator exempted, newly rejecting documents that used
            // to pass. Undeclared type keeps the lenient path.
            let node = declare(&mut tree, name, &root);
            node.values = Some(values.clone());
            node.set_by.values = Some(root.clone());
        }

        Self {
            fields: ResolvedFields::from_tree(tree),
            ..Self::default()
        }
    }

    /// Test-only accessor for the private merge, so validation tests can build a
    /// resolved schema without going through a filesystem cascade.
    #[cfg(test)]
    pub(crate) fn merged_with_for_test(&self, child: &SchemaFile, origin: &str) -> Self {
        self.merged_with(child, origin)
    }

    /// Merge a child schema file onto this one.
    ///
    /// The set of fields unions. Merging is **per attribute**, not per field: a child
    /// that redefines a field overrides only the attributes it explicitly writes
    /// (`type`, `required`, `indexed`, `default`, `open`, `values`) — every attribute it
    /// leaves unset still inherits from the parent's definition of that same field. A
    /// child that writes `required: true` and nothing else, say, does not reset the
    /// parent's `values` or `default` to nothing; it only tightens `required`. The
    /// same holds at every depth of the tree, `items` included.
    ///
    /// This is deliberately NOT the old rule (a redefinition replacing the whole
    /// definition wholesale, `extend: true` as the sole opt-in to union `values`): that
    /// rule silently discarded a parent's `required`/`indexed`/`default` the moment any
    /// child so much as narrowed `values`, which is exactly the shape of footgun that
    /// let a root-level `required: true` on `tags` go unenforced everywhere, since
    /// every domain redeclared `tags` for its own `values` list. Per-attribute
    /// inheritance means only fields whose redefinition genuinely intends to override a
    /// given attribute do — see [`FieldDef::merge_node`] for the exact per-attribute
    /// rule.
    ///
    /// `values` is the one attribute with an in-band way to request a merge instead of
    /// a plain override — see [`FieldDef::merge_values`] for the `$values` placeholder
    /// and the deprecated `extend: true` alias for it.
    fn merged_with(&self, child: &SchemaFile, origin: &str) -> Self {
        let mut tree = self.fields.tree.clone();
        for (key, raw) in &child.fields {
            let path = FieldPath::parse(key);
            if raw.removed {
                remove_node(&mut tree, path.steps());
                continue;
            }
            if excludes_from_nothing(raw, &tree, path.steps()) {
                continue;
            }
            node_in_children(&mut tree, path.steps(), origin).merge_node(raw, origin, key);
        }

        let dedup = child.dedup.as_ref();
        // Root-only blocks (#345): the root schema file sets them, every other scope
        // inherits its parent's, and one written anywhere else is ignored (and linted).
        let is_root = origin == scope_label(Path::new(""));
        let (derived, root_edit) = if is_root {
            (
                child.derived.clone().unwrap_or_default(),
                child.root_edit.clone(),
            )
        } else {
            (self.derived.clone(), self.root_edit.clone())
        };
        Self {
            derived,
            root_edit,
            fields: ResolvedFields::from_tree(tree),
            dedup_enabled: dedup.and_then(|d| d.enabled).or(self.dedup_enabled),
            dedup_threshold: dedup.and_then(|d| d.threshold).or(self.dedup_threshold),
            related_min_score: dedup
                .and_then(|d| d.related_min_score)
                .or(self.related_min_score),
            related_limit: dedup.and_then(|d| d.related_limit).or(self.related_limit),
        }
    }

    /// The payload index kind a declared type needs ([`FieldKind::index`]).
    ///
    /// Numeric and boolean fields need their own index kinds for range and comparison
    /// filters to work; everything else, including undeclared fields, is a keyword.
    pub fn index_kind(ty: Option<FieldType>) -> IndexKind {
        ty.map_or(IndexKind::Keyword, |ty| ty.kind().index)
    }

    /// A stable fingerprint of this schema.
    ///
    /// Used to detect that a document needs revalidating because the rules changed,
    /// even though its content did not. Must not depend on map iteration order, or
    /// every incremental run would look like a schema change.
    ///
    /// Hashes the flat view in dot-path order. A field's `items` element schema is
    /// hashed only when present, so a schema that uses none hashes exactly as it did
    /// before `items` existed (#346) — changing the bytes of a field without `items`
    /// would re-validate (and reproject) the whole corpus. Since #351 a moved
    /// fingerprint alone never re-embeds a document (see [`Self::payload_fingerprint`]
    /// for what reaches Qdrant; a `description` default that changes a document's
    /// prepended description is the exception ingest checks per document).
    ///
    /// `removed` and `exclude_values` (#348) are not hashed as attributes: they act
    /// on the resolved model (a removed field is absent from it, an excluded value
    /// from its `values`), which is what is hashed, so a schema that uses neither
    /// hashes exactly as before, and one that does changes only where its effect is.
    ///
    /// Deliberately excludes `dedup_enabled`/`dedup_threshold` (#272): the near-duplicate
    /// gate affects neither validation nor indexing, so changing it must not mark a
    /// subtree's documents for revalidation. Provenance (`origin`, `set_by`) is not
    /// hashed either, nor is `root_edit`, which governs only `update_schema`.
    ///
    /// `derived` (#345) is hashed after the fields, behind a `0xfc` marker, and only
    /// when non-empty, so a schema without one hashes as it did before the block
    /// existed. It changes the frontmatter ingest stores, so a change must reprocess
    /// the documents it governs.
    pub fn fingerprint(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        for (path, def) in &self.fields {
            hasher.update(path.as_bytes());
            hasher.update([0u8]);
            hash_attributes(&mut hasher, def);
            hasher.update([0xffu8]);
        }
        if !self.derived.is_empty() {
            hasher.update([0xfcu8]);
            for (name, rule) in &self.derived {
                hasher.update(name.as_bytes());
                hasher.update([0u8]);
                hasher.update(match rule.from {
                    DerivedFrom::PathSegment => b"path_segment".as_slice(),
                });
                hasher.update([0u8]);
                hasher.update((rule.index as u64).to_le_bytes());
                hasher.update([0xffu8]);
            }
        }
        hex::encode(hasher.finalize())
    }

    /// A stable fingerprint of exactly the schema inputs that shape the frontmatter
    /// ingest writes into every chunk's Qdrant payload (#351): the defaults
    /// `validate::apply_defaults` fills in (flat-view path and value; a default on a
    /// derived field is never applied, so it is left out) and the `derived:` rules
    /// `ingest::with_derived_fields` applies.
    ///
    /// Stored per file as `indexed_files.payload_hash`, next to [`Self::fingerprint`]
    /// (`schema_hash`), which stays the validation and metadata-projection hash. A
    /// schema edit that moves `fingerprint` but not this — tightening `required`,
    /// narrowing `values`, marking a field `indexed` — revalidates and reprojects the
    /// document without touching Qdrant; one that moves this too also refreshes the
    /// payload in place. Neither re-embeds. Every input here is also an input of
    /// `fingerprint`, so a change here always moves both.
    pub fn payload_fingerprint(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        for (path, def) in &self.fields {
            let Some(default) = &def.default else {
                continue;
            };
            if self.is_derived(path) {
                continue;
            }
            hasher.update(path.as_bytes());
            hasher.update([0u8]);
            hasher.update(default.to_string().as_bytes());
            hasher.update([0xffu8]);
        }
        hasher.update([0xfcu8]);
        for (name, rule) in &self.derived {
            hasher.update(name.as_bytes());
            hasher.update([0u8]);
            hasher.update(match rule.from {
                DerivedFrom::PathSegment => b"path_segment".as_slice(),
            });
            hasher.update([0u8]);
            hasher.update((rule.index as u64).to_le_bytes());
            hasher.update([0xffu8]);
        }
        hex::encode(hasher.finalize())
    }
}

/// A field's attributes as [`ResolvedSchema::fingerprint`] hashes them, then its
/// `items` (when present) behind a `0xfe` marker. Every other byte this writes is
/// either fixed-width or UTF-8 text, which never contains `0xfe` or `0xff`, so the
/// markers cannot collide with attribute bytes.
fn hash_attributes(hasher: &mut sha2::Sha256, def: &FieldDef) {
    use sha2::Digest;
    hasher.update(format!("{:?}", def.ty).as_bytes());
    hasher.update([def.required as u8, def.indexed as u8, def.open as u8]);
    // Presence discriminant: `Some(vec![])` ("nothing is permitted", per
    // `check_values`/`check_values_lenient`) and `None` ("unconstrained")
    // enforce opposite rules but would otherwise hash identically — the
    // loop below contributes zero bytes either way when the list is empty
    // or absent. Emitted unconditionally, before the loop, so the two
    // states always diverge regardless of list contents (#152).
    hasher.update([def.values.is_some() as u8]);
    if let Some(values) = &def.values {
        let mut sorted = values.clone();
        sorted.sort();
        for value in sorted {
            hasher.update(value.as_bytes());
            hasher.update([1u8]);
        }
    }
    if let Some(default) = &def.default {
        hasher.update(default.to_string().as_bytes());
    }
    if let Some(items) = &def.items {
        hasher.update([0xfeu8]);
        hash_subtree(hasher, items);
    }
}

/// An element schema and everything below it, for [`hash_attributes`]: the node's
/// attributes and declared-ness, then each child by name (sorted), each framed by
/// `0xfd` and `0xff`.
fn hash_subtree(hasher: &mut sha2::Sha256, node: &FieldDef) {
    use sha2::Digest;
    hasher.update([node.declared_in.is_some() as u8]);
    hash_attributes(hasher, node);
    for (name, child) in &node.fields {
        hasher.update([0xfdu8]);
        hasher.update(name.as_bytes());
        hasher.update([0u8]);
        hash_subtree(hasher, child);
        hasher.update([0xffu8]);
    }
}

/// The discovered schema tree, with per-directory merge results precomputed.
#[derive(Debug, Clone, Default)]
pub struct SchemaCache {
    /// Resolved schema per governing directory (KB-relative), longest path last.
    scopes: Vec<(PathBuf, ResolvedSchema)>,
    /// Raw, unmerged schema files by governing directory, so a proposed edit can be
    /// re-cascaded exactly rather than approximated.
    raw: BTreeMap<PathBuf, SchemaFile>,
    root: ResolvedSchema,
    /// KB root, so a scope's raw schema file can be read back for editing.
    root_path: PathBuf,
    /// When the build that produced this cache started, in the process-wide
    /// order of [`SchemaCache::build`] calls. [`store_shared`] never replaces a
    /// cache with one whose build started earlier: two rebuilds racing (the
    /// reindex worker's and `update_schema`'s) can finish in either order, and
    /// the one that read the tree first must not win.
    generation: u64,
}

/// One schema file that [`SchemaCache::build`] refused, and why.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, Deserialize)]
pub struct InvalidSchemaFile {
    /// KB-relative path of the schema file itself (not its governing directory).
    pub path: PathBuf,
    pub reason: String,
}

/// Every invalid schema file found by one [`SchemaCache::build`] walk, sorted
/// by path (`read_dir` order is unspecified, and this list is shown in startup
/// errors and `/status`). Never empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaBuildError {
    pub invalid: Vec<InvalidSchemaFile>,
}

impl std::fmt::Display for SchemaBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} invalid schema file(s); a schema file that is present must be valid:",
            self.invalid.len(),
        )?;
        for file in &self.invalid {
            write!(f, "\n  - {}: {}", file.path.display(), file.reason)?;
        }
        Ok(())
    }
}

impl std::error::Error for SchemaBuildError {}

impl SchemaBuildError {
    /// The refusal as a model should read it: each invalid schema named by its scope
    /// directory ([`scope_label`]) rather than its file path. The `Display` impl keeps
    /// real paths for logs, `/status` and the CLI.
    pub fn model_facing(&self) -> String {
        let mut out = format!("{} directory schema(s) are invalid:", self.invalid.len());
        for file in &self.invalid {
            let dir = file.path.parent().unwrap_or(Path::new(""));
            out.push_str(&format!(
                "\n  - {}: {}",
                scope_label(dir),
                model_facing_reason(&file.reason)
            ));
        }
        out
    }
}

/// True when `rel_path`'s file name is one of [`SCHEMA_FILE_NAMES`] — a schema file,
/// not a document, whichever directory it sits in.
pub fn is_schema_file_path(rel_path: &Path) -> bool {
    rel_path
        .file_name()
        .is_some_and(|n| SCHEMA_FILE_NAMES.iter().any(|name| n == *name))
}

/// Parse and self-validate one schema file's text — the per-file check
/// [`SchemaCache::build`] applies, and the reason it gives on failure. Also
/// enforces [`MAX_SCHEMA_FILE_BYTES`] on the text itself, for callers that already
/// hold the content (`write::move_directory`, checking the schema files it is about
/// to carry along).
pub(crate) fn parse_schema_text(text: &str) -> Result<SchemaFile, String> {
    if text.len() as u64 > MAX_SCHEMA_FILE_BYTES {
        return Err(over_size_limit_reason(text.len() as u64));
    }
    let file = serde_yaml_ng::from_str::<SchemaFile>(text).map_err(|e| e.to_string())?;
    file.validate_self()?;
    Ok(file)
}

/// Which directories' schema files are part of the schema tree (#272):
/// every directory except a hidden one (any path component starting with `.`) and
/// one whose every document `indexing.exclude` rules out
/// ([`crate::ingest::PathFilter::excludes_dir`]). A schema in such a directory
/// governs nothing that is indexed, so [`SchemaCache::build`] never reads it — it
/// cannot stop startup or a rebuild — and a changed one never queues a full
/// reconcile ([`Self::governs`]).
///
/// Fails OPEN: an `indexing` glob set that cannot be built (the same failure
/// `ingest::partition_indexable` fails open on) reads every non-hidden directory,
/// logged at error level, rather than silently dropping schema files that may
/// govern indexed documents.
pub(crate) struct SchemaWalkFilter {
    paths: Option<crate::ingest::PathFilter>,
}

impl SchemaWalkFilter {
    pub(crate) fn from_config(indexing: &crate::config::IndexingConfig) -> Self {
        match crate::ingest::PathFilter::from_config(indexing) {
            Ok(f) => Self { paths: Some(f) },
            Err(e) => {
                tracing::error!(
                    "Failed to build indexing path filter; reading schema files in \
                     every non-hidden directory, excluded ones included: {e:#}"
                );
                Self { paths: None }
            }
        }
    }

    /// Whether a schema file in KB-relative directory `rel_dir` is read.
    pub(crate) fn reads_dir(&self, rel_dir: &Path) -> bool {
        let hidden = rel_dir
            .components()
            .any(|c| c.as_os_str().to_string_lossy().starts_with('.'));
        if hidden {
            return false;
        }
        if rel_dir.as_os_str().is_empty() {
            return true;
        }
        !self
            .paths
            .as_ref()
            .is_some_and(|f| f.excludes_dir(&rel_dir.to_string_lossy()))
    }

    /// Whether KB-relative `rel_path` is a schema file that is part of the
    /// schema tree — the test a changed path must pass to queue a schema rebuild.
    pub(crate) fn governs(&self, rel_path: &Path) -> bool {
        is_schema_file_path(rel_path) && self.reads_dir(rel_path.parent().unwrap_or(Path::new("")))
    }
}

/// A `SchemaCache` shared across the server, kept current by a single owner rather
/// than rebuilt (a full recursive tree walk) by every caller that needs it.
///
/// `RwLock<Arc<SchemaCache>>` rather than `arc-swap`: a reader takes the lock,
/// clones the `Arc`, and drops the guard immediately (see [`load_shared`]) — a
/// handful of atomic operations — which is cheap enough for a read-mostly value
/// that pulling in a new dependency for lock-free swaps is not justified. The
/// outer `Arc` is what makes this cloneable across the MCP handler, the web UI,
/// and the reindex worker, all of which hold a handle to the SAME lock rather than
/// independent copies.
pub type SharedSchemaCache = Arc<RwLock<Arc<SchemaCache>>>;

/// Clone the current cache out of `shared`. Cheap: a lock acquisition plus an
/// `Arc` clone, with the guard dropped before returning — never held across an
/// `.await`.
///
/// A poisoned lock (a reader or writer panicked while holding it) is recovered
/// rather than propagated, the same policy `server.rs` already applies to the
/// instructions lock: a panic in one caller must not brick every subsequent
/// `get_schema`/write for the rest of the process's life.
pub fn load_shared(shared: &SharedSchemaCache) -> Arc<SchemaCache> {
    shared
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// Swap a freshly built `SchemaCache` into `shared`, replacing whatever was there.
///
/// Callers of [`load_shared`] that are already mid-read hold their own `Arc` clone
/// and are unaffected by a swap landing underneath them — they simply keep using
/// the snapshot they took, and the next `load_shared` call sees the new one.
///
/// A cache whose build started before the one already installed is discarded
/// (see [`SchemaCache::generation`]): it read an older tree.
pub fn store_shared(shared: &SharedSchemaCache, new: SchemaCache) {
    let mut guard = match shared.write() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if new.generation < guard.generation {
        tracing::debug!(
            "Discarding a schema rebuild that started before the installed one ({} < {})",
            new.generation,
            guard.generation
        );
        return;
    }
    *guard = Arc::new(new);
}

/// One [`lint_raw`] finding: the governing directory of the schema file, and the
/// warning.
pub type LintFinding = (PathBuf, String);

/// The findings in `new` that `previous` does not have, in `new`'s order.
pub fn new_lint_findings(previous: &SchemaCache, new: &SchemaCache) -> Vec<LintFinding> {
    let seen: BTreeSet<LintFinding> = previous.lint_findings().into_iter().collect();
    new.lint_findings()
        .into_iter()
        .filter(|finding| !seen.contains(finding))
        .collect()
}

/// Log lint findings at `warn`, naming each one's scope (#347). Called for the
/// startup cache (all findings), the `validate` CLI, and a shared-cache rebuild
/// (only the findings that rebuild introduced, [`new_lint_findings`]); never for the
/// throwaway caches indexing runs build, which would repeat them on every write.
pub fn log_lint_findings(findings: &[LintFinding]) {
    for (rel_dir, warning) in findings {
        warn!(scope = %scope_label(rel_dir), "schema lint: {warning}");
    }
}

/// The process-wide order of [`SchemaCache::build`] calls.
static REBUILD_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Apply the result of a runtime [`SchemaCache::build`] to `shared` — the one
/// policy every runtime rebuild (the reindex worker, `update_schema`) goes through.
///
/// `Ok`: swap it in and clear `status`'s schema error. `Err`: refuse it —
/// `shared` keeps the last cache that built cleanly, so no document is ever
/// validated or indexed under a schema known to be invalid, and the refusal is
/// logged at error level (every invalid file, path and reason) and recorded as
/// `status`'s schema error for `/status` and `/metrics`. `/health` is
/// deliberately unaffected: the server keeps serving correctly under the previous
/// schema. Startup has no previous cache and treats the same error as fatal
/// instead (`server::run_server`).
///
/// `status` is `crate::status::INDEX_STATUS` in production — a parameter so tests
/// can observe a private instance instead of the process-global one other tests
/// also write to.
///
/// Returns whether the new cache was applied.
pub fn apply_rebuild(
    shared: &SharedSchemaCache,
    built: Result<SchemaCache, SchemaBuildError>,
    status: &crate::status::IndexStatus,
    context: &str,
) -> bool {
    match built {
        Ok(schemas) => {
            // Only findings the installed cache did not already have: a rebuild
            // before every full reconcile must not repeat the same warnings (#347).
            let previous = load_shared(shared);
            log_lint_findings(&new_lint_findings(&previous, &schemas));
            store_shared(shared, schemas);
            if status.schema_error().is_some() {
                tracing::info!(
                    "{context}: every schema file is valid again; the rebuilt schema \
                     is now in effect"
                );
            }
            status.clear_schema_error();
            true
        }
        Err(e) => {
            tracing::error!(
                invalid_files = e.invalid.len(),
                "{context}: schema rebuild REFUSED, keeping the previous schema — {e}"
            );
            status.record_schema_error(&e);
            false
        }
    }
}

/// Merge a set of already-parsed schema files, keyed by governing directory, into
/// the per-directory resolved schema tree — the shallow-first sort, per-scope
/// `nearest_schema` lookup, and the root replace-vs-merge special case that used
/// to be inlined in [`SchemaCache::build`].
///
/// Pure: no filesystem access (`files` must already hold fully parsed content —
/// this never reads a schema file off disk), and no state beyond the `warn!` calls
/// [`ResolvedSchema::merged_with`] itself already makes. This is what makes it
/// reusable for two different callers that arrive at a `(governing_dir ->
/// SchemaFile)` map by different means: [`SchemaCache::build`] gets there by
/// walking disk and separately tracking parse failures (irreducibly a filesystem
/// concern, so that bookkeeping stays in `build` rather than here — see its doc
/// comment), and [`SchemaCache::with_remapped_scopes`] gets there by rekeying an
/// existing cache's already-parsed `raw` map, entirely in memory. Compare
/// [`SchemaCache::resolve_with_candidate`], which does similar in-memory
/// ancestor-chain rebuilding for a single substituted directory but — unlike this
/// — cannot relocate entries, only replace one directory's content in place.
///
/// A root entry (governing directory `""`) REPLACES the config-derived root
/// outright rather than merging onto it — see [`SchemaCache::build`]'s doc
/// comment for why; that policy is cascade-merge logic, not disk-walk logic, so
/// it lives here.
fn merge_cascade(
    files: &BTreeMap<PathBuf, SchemaFile>,
    config_root: &ResolvedSchema,
) -> Vec<(PathBuf, ResolvedSchema)> {
    // Shallowest first so each merge sees its parent already resolved, then by path
    // so two scopes at the same depth resolve in a stable order. Depth alone would
    // leave a `BTreeMap`'s lexicographic key order in charge of siblings, which
    // does not track depth — a field declared with conflicting types in two
    // sibling scopes could otherwise silently pick a different index kind
    // depending on how their paths happen to sort.
    let mut order: Vec<&PathBuf> = files.keys().collect();
    order.sort_by(|a, b| {
        a.components()
            .count()
            .cmp(&b.components().count())
            .then_with(|| a.cmp(b))
    });

    let mut scopes: Vec<(PathBuf, ResolvedSchema)> = Vec::new();
    let mut root_file_found = false;

    for rel_dir in order {
        let file = &files[rel_dir];
        let origin = scope_label(rel_dir);
        let is_root = rel_dir.as_os_str().is_empty();
        let merged = if is_root {
            root_file_found = true;
            if !config_root.fields.is_empty() {
                // Loud, not silent: config.yaml declares root rules that are about
                // to stop applying. This fires on every cascade build (like the
                // "conflicting types" and "malformed schema" warnings elsewhere in
                // this module), not once — a build runs on every write and every
                // reconcile sweep, so an operator tailing logs sees it consistently
                // rather than only at the one moment it first became true.
                warn!(
                    "a root schema file exists at the knowledge-base root; config.yaml's \
                     `frontmatter` block no longer applies there — its \
                     required/indexed_fields/defaults/allowed entries are ignored \
                     unless the same fields are also declared in the root {}. Move \
                     anything still needed into it.",
                    SCHEMA_FILE_NAME
                );
            }
            // Replaces, not merges: an empty base, not `config_root`.
            ResolvedSchema::default().merged_with(file, &origin)
        } else {
            let parent = nearest_schema(&scopes, rel_dir).unwrap_or(config_root);
            parent.merged_with(file, &origin)
        };
        scopes.push((rel_dir.clone(), merged));
    }

    if !root_file_found && !config_root.fields.is_empty() {
        // The deprecated fallback: no root schema file in this map, so
        // config.yaml's `frontmatter` block is standing in as the root schema.
        // Still fully supported (see module docs), but this is the direction we
        // want deployments to move away from — flag it every time this runs, same
        // as the "config overridden" warning above, so it stays visible for as
        // long as it is true rather than only at startup.
        warn!(
            "no root schema file found; falling back to the deprecated `frontmatter` \
             block in config.yaml for root-level rules. This still works, but a root {} \
             is the non-deprecated way to declare them — see deploy/USAGE.md.",
            SCHEMA_FILE_NAME
        );
    }

    // Longest paths last so prefix lookup can scan backwards for the deepest match,
    // with path as a stable tiebreaker among equal depths.
    scopes.sort_by(|(a, _), (b, _)| {
        a.components()
            .count()
            .cmp(&b.components().count())
            .then_with(|| a.cmp(b))
    });

    scopes
}

impl SchemaCache {
    /// Walk `data_path` for schema files and precompute every directory's merged schema.
    ///
    /// One pass over the tree, not one per document: resolution afterwards is an
    /// in-memory prefix lookup that touches no filesystem.
    ///
    /// A root schema file (governing directory `""`) is handled differently from
    /// every other scope: instead of merging onto its nearest ancestor — which, at the
    /// root, would mean merging onto the config-derived schema — it REPLACES the
    /// config-derived root outright. Config-derived root rules apply only when no root
    /// schema file exists at all. See the module docs for why: the config block
    /// is deployment config on the container host, and a KB that brings its own root
    /// schema file must not have that schema silently blended with whatever
    /// `frontmatter` block the current host's `config.yaml` happens to declare.
    ///
    /// A thin wrapper around [`merge_cascade`]: this function's own job is just the
    /// disk walk and per-file parse/validate, which is also the only place an invalid
    /// schema file can be discovered — a parse failure needs the underlying I/O
    /// error, which an in-memory cascade merge (see
    /// [`SchemaCache::with_remapped_scopes`], the merge algorithm's other caller) has
    /// no way to produce, since it starts from content that already parsed.
    ///
    /// Fail-fast: a schema file that is present must be valid. Any file that
    /// cannot be read, is over [`MAX_SCHEMA_FILE_BYTES`], fails to parse, or fails
    /// [`SchemaFile::validate_self`] makes the whole build an error — every such file
    /// is collected into the one [`SchemaBuildError`], not just the first, so one
    /// pass names everything to fix. There is no partial cache: documents are never
    /// validated or indexed under rules that are known to be wrong. Callers decide
    /// what that means — fatal at startup and in the CLI, "keep the last good cache"
    /// for a runtime rebuild.
    ///
    /// Only the indexed tree is walked: a hidden directory, or one `indexing.exclude`
    /// rules out entirely, is never read (see [`SchemaWalkFilter`]), so a broken
    /// schema under `templates/**` cannot stop anything (#272).
    ///
    /// A directory holding both [`SCHEMA_FILE_NAME`] and [`LEGACY_SCHEMA_FILE_NAME`]
    /// is an invalid schema file ([`BOTH_NAMES_REASON`]), never resolved by picking
    /// one: the two could disagree, and which one wins would be invisible.
    pub fn build(
        data_path: &Path,
        fallback: &FrontmatterConfig,
        indexing: &crate::config::IndexingConfig,
    ) -> Result<Self, SchemaBuildError> {
        // Stamped before the walk reads anything: see `SchemaCache::generation`.
        let generation = REBUILD_GENERATION.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        let config_root = ResolvedSchema::from_config(fallback);
        let walk = SchemaWalkFilter::from_config(indexing);
        let mut discovered: Vec<(PathBuf, PathBuf)> = Vec::new();
        collect_schema_files(data_path, data_path, &walk, &mut discovered);

        let mut raw: BTreeMap<PathBuf, SchemaFile> = BTreeMap::new();
        let mut invalid: Vec<InvalidSchemaFile> = Vec::new();

        let mut seen: BTreeSet<&Path> = BTreeSet::new();
        let doubled: BTreeSet<PathBuf> = discovered
            .iter()
            .filter(|(rel_dir, _)| !seen.insert(rel_dir.as_path()))
            .map(|(rel_dir, _)| rel_dir.clone())
            .collect();
        for dir in &doubled {
            invalid.push(InvalidSchemaFile {
                path: dir.join(LEGACY_SCHEMA_FILE_NAME),
                reason: BOTH_NAMES_REASON.to_string(),
            });
        }

        for (rel_dir, abs_file) in discovered {
            if doubled.contains(&rel_dir) {
                continue;
            }
            let file_name = abs_file
                .file_name()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(SCHEMA_FILE_NAME));
            let parsed = std::fs::metadata(&abs_file)
                .map_err(|e| format!("could not stat: {e}"))
                .and_then(|meta| {
                    if meta.len() > MAX_SCHEMA_FILE_BYTES {
                        Err(over_size_limit_reason(meta.len()))
                    } else {
                        Ok(())
                    }
                })
                .and_then(|()| {
                    std::fs::read_to_string(&abs_file).map_err(|e| format!("could not read: {e}"))
                })
                .and_then(|text| parse_schema_text(&text));

            match parsed {
                Ok(file) => {
                    debug!(scope = %rel_dir.display(), "loaded schema");
                    // Lint warnings (#347) are not logged here: throwaway builds
                    // (one per indexing run) would repeat them. The shared-cache
                    // paths log them — see `SchemaCache::log_lint_findings`.
                    raw.insert(rel_dir, file);
                }
                Err(reason) => invalid.push(InvalidSchemaFile {
                    path: rel_dir.join(file_name),
                    reason,
                }),
            }
        }

        if !invalid.is_empty() {
            invalid.sort_by(|a, b| a.path.cmp(&b.path));
            return Err(SchemaBuildError { invalid });
        }

        let scopes = merge_cascade(&raw, &config_root);

        Ok(Self {
            scopes,
            raw,
            root: config_root,
            root_path: data_path.to_path_buf(),
            generation,
        })
    }

    /// [`SchemaCache::build`] for tests whose fixtures are known-valid: panics with
    /// the full error when they are not. Walks with the default `indexing` filter;
    /// see [`Self::build_for_test_with`] to pass one.
    #[cfg(test)]
    pub fn build_for_test(data_path: &Path, fallback: &FrontmatterConfig) -> Self {
        Self::build_for_test_with(data_path, fallback, &Default::default())
    }

    /// [`Self::build_for_test`] with an explicit `indexing` filter.
    #[cfg(test)]
    pub fn build_for_test_with(
        data_path: &Path,
        fallback: &FrontmatterConfig,
        indexing: &crate::config::IndexingConfig,
    ) -> Self {
        Self::build(data_path, fallback, indexing).unwrap_or_else(|e| panic!("{e}"))
    }

    /// A NEW, detached cache with every schema file's governing directory passed
    /// through `remap`: `Some(new_dir)` relocates that file's declarations to
    /// `new_dir` (and, since the whole tree is rebuilt from scratch below, changes
    /// what every OTHER scope under either the old or new directory resolves to,
    /// not just the relocated one); `None` leaves it exactly where it is.
    ///
    /// Purely in-memory: `raw` already holds fully parsed, validated content, so this
    /// never touches the filesystem and cannot fail — a schema file that parsed fine
    /// is not going to stop parsing just because its key moved. The result is never
    /// registered into a [`SharedSchemaCache`]
    /// — it is a local value for exactly one hypothetical-resolution pass (see
    /// `write::move_directory`'s use of it), the same "in-memory, not-for-storage"
    /// role [`SchemaCache::resolve_with_candidate`] plays for a single substituted
    /// scope, generalized to relocating any number of scopes at once.
    ///
    /// Origin provenance ([`ResolvedSchema::origin`]) comes out correct for free:
    /// [`merge_cascade`] derives each field's origin from the governing directory
    /// key it is merging under, and `raw` is rekeyed to the NEW directory before
    /// the merge runs — so a relocated field's origin names its new home, not its
    /// old one.
    ///
    /// `remap` must not send two distinct entries to the same directory, and must
    /// not send an entry onto a directory some OTHER, non-relocated entry already
    /// occupies — either collision silently drops one schema file's declarations
    /// via `BTreeMap` insertion order. `write::move_directory`'s caller satisfies
    /// this: it only ever relocates directories under the moved source subtree by
    /// the same injective prefix substitution the move itself applies to every
    /// document, onto a destination prefix its own guards have already confirmed
    /// is completely empty.
    pub fn with_remapped_scopes(&self, remap: impl Fn(&Path) -> Option<PathBuf>) -> SchemaCache {
        let mut raw: BTreeMap<PathBuf, SchemaFile> = BTreeMap::new();
        for (dir, file) in &self.raw {
            let key = remap(dir).unwrap_or_else(|| dir.clone());
            raw.insert(key, file.clone());
        }

        let scopes = merge_cascade(&raw, &self.root);

        SchemaCache {
            scopes,
            raw,
            root: self.root.clone(),
            root_path: self.root_path.clone(),
            generation: self.generation,
        }
    }

    /// The raw, unmerged schema file governing `rel_dir`, read from disk, and the
    /// name it is stored under ([`SCHEMA_FILE_NAME`] or [`LEGACY_SCHEMA_FILE_NAME`]);
    /// an empty file and `None` when that directory has no schema of its own. Both
    /// names present is an error, as it is for [`Self::build`].
    ///
    /// A candidate is read the way [`Self::build`]'s walk finds one
    /// ([`read_schema_candidate`]): a symlink or any other non-regular entry is not a
    /// schema file, so it is absent here too — and so does not count toward the
    /// both-names error — and a regular file over [`MAX_SCHEMA_FILE_BYTES`] is an
    /// error before it is read.
    pub fn raw_file_at(
        &self,
        rel_dir: &Path,
    ) -> Result<(SchemaFile, Option<&'static str>), String> {
        let mut found: Option<(SchemaFile, &'static str)> = None;
        for name in SCHEMA_FILE_NAMES {
            let file = self.root_path.join(rel_dir).join(name);
            let Some(text) = read_schema_candidate(&file)? else {
                continue;
            };
            if found.is_some() {
                return Err(BOTH_NAMES_REASON.to_string());
            }
            let parsed = serde_yaml_ng::from_str(&text).map_err(|e| e.to_string())?;
            found = Some((parsed, name));
        }
        Ok(match found {
            Some((file, name)) => (file, Some(name)),
            None => (SchemaFile::default(), None),
        })
    }

    /// What `rel_dir` resolves to with its own schema file replaced by `own`
    /// (`None`: what its ancestors alone resolve it to), every ancestor's file
    /// read from disk through [`Self::raw_file_at`] rather than taken from this
    /// cache — `update_schema` calls it under the write lock, so it sees the
    /// same files the edit itself is applied to. The root follows [`Self::build`]'s
    /// policy: a root schema file on disk (or `own` at the root) replaces the
    /// config-derived root outright.
    pub fn resolve_from_disk(
        &self,
        rel_dir: &Path,
        own: Option<&SchemaFile>,
    ) -> Result<ResolvedSchema, String> {
        let mut ancestors: Vec<&Path> = rel_dir.ancestors().skip(1).collect();
        ancestors.reverse();
        let mut chain: Vec<(&Path, SchemaFile)> = Vec::new();
        let mut root_file = rel_dir.as_os_str().is_empty() && own.is_some();
        for dir in ancestors {
            let (file, name) = self.raw_file_at(dir)?;
            if name.is_some() {
                root_file |= dir.as_os_str().is_empty();
                chain.push((dir, file));
            }
        }
        let mut resolved = if root_file {
            ResolvedSchema::default()
        } else {
            self.root.clone()
        };
        for (dir, file) in &chain {
            resolved = resolved.merged_with(file, &scope_label(dir));
        }
        if let Some(own) = own {
            resolved = resolved.merged_with(own, &scope_label(rel_dir));
        }
        Ok(resolved)
    }

    /// The schema `doc_path` would resolve to if `edited_dir`'s schema file held
    /// `candidate`.
    ///
    /// Rebuilds the document's full ancestor chain with the candidate substituted in,
    /// rather than guessing whether a deeper scope shadows the edit. Merge semantics are
    /// per FIELD: a descendant that redeclares one field still inherits every other one,
    /// so "a deeper scope exists" tells you nothing about whether this edit reaches the
    /// document. Returns `None` only when the document lies outside the edited subtree.
    pub fn resolve_with_candidate(
        &self,
        doc_path: &Path,
        edited_dir: &Path,
        candidate: &SchemaFile,
    ) -> Option<ResolvedSchema> {
        let dir = doc_path.parent().unwrap_or(Path::new(""));
        if !path_covers(edited_dir, dir) {
            return None;
        }

        // Every scope governing this document, shallowest first, with the candidate
        // standing in for the edited directory's own file.
        let mut chain: Vec<(&Path, &SchemaFile)> = Vec::new();
        for (scope, file) in &self.raw {
            if path_covers(scope, dir) {
                let source = if scope == edited_dir { candidate } else { file };
                chain.push((scope.as_path(), source));
            }
        }
        if !self.raw.contains_key(edited_dir) {
            // The edited directory has no schema file yet; insert the candidate at its
            // correct depth so deeper scopes still layer on top of it.
            chain.push((edited_dir, candidate));
        }
        chain.sort_by_key(|(scope, _)| scope.components().count());

        // Mirrors `build`'s root policy: when a root schema file governs this
        // document — one already exists on disk, or this very edit is creating one —
        // `chain` already contains an entry for `""` (real or candidate) that fully
        // determines the root's fields, so starting from the config-derived root here
        // would let config re-contaminate fields the root file doesn't mention. Only
        // fall back to the config-derived root when no root schema file is in play at
        // all — the same "root file present, if any, wins outright" rule `build` uses.
        let root_governed =
            self.raw.contains_key(Path::new("")) || edited_dir.as_os_str().is_empty();
        let mut resolved = if root_governed {
            ResolvedSchema::default()
        } else {
            self.root.clone()
        };
        for (scope, file) in chain {
            let origin = scope_label(scope);
            resolved = resolved.merged_with(file, &origin);
        }
        Some(resolved)
    }

    /// Build a cache with no schema files, backed only by the global config.
    ///
    /// Used where a cascade is required by signature but the caller has no tree to walk.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn from_config_only(fallback: &FrontmatterConfig) -> Self {
        Self {
            scopes: Vec::new(),
            raw: BTreeMap::new(),
            root: ResolvedSchema::from_config(fallback),
            root_path: PathBuf::new(),
            generation: 0,
        }
    }

    /// The effective schema for a KB-relative document path.
    pub fn resolve_for(&self, rel_path: &Path) -> &ResolvedSchema {
        let dir = rel_path.parent().unwrap_or(Path::new(""));
        nearest_schema(&self.scopes, dir).unwrap_or(&self.root)
    }

    /// The root schema: from a root schema file when one exists, otherwise the
    /// config-derived fallback. See [`SchemaCache::build`] for why these don't merge.
    pub fn root(&self) -> &ResolvedSchema {
        self.scopes
            .iter()
            .find(|(dir, _)| dir.as_os_str().is_empty())
            .map(|(_, schema)| schema)
            .unwrap_or(&self.root)
    }

    /// Every dot-path declared `indexed` anywhere in the tree, with its index kind.
    ///
    /// Payload indexes are created once for the whole collection, so a field declared
    /// only in a deep scope still has to be registered up front — otherwise filtering on
    /// it silently fails until the collection is rebuilt.
    ///
    /// When two scopes declare the same path with different types, the first wins and a
    /// warning is emitted: one collection cannot hold two index kinds for one path.
    pub fn all_indexed_fields(&self) -> Vec<IndexedField> {
        let mut fields: Vec<IndexedField> = Vec::new();

        let mut add = |schema: &ResolvedSchema| {
            for (path, def) in &schema.fields {
                if !def.indexed {
                    continue;
                }
                let kind = ResolvedSchema::index_kind(def.ty);
                match fields.iter_mut().find(|f| &f.name == path) {
                    Some(existing) if existing.kind == kind => {}
                    // Keyword is the fallback for an undeclared type, so an explicit
                    // declaration anywhere in the tree beats it — otherwise a field
                    // also named in the legacy config's `indexed_fields` would get a
                    // keyword index and every range filter on it would quietly fail.
                    Some(existing) if existing.kind == IndexKind::Keyword => {
                        existing.kind = kind;
                    }
                    Some(existing) if kind == IndexKind::Keyword => {
                        // Keep the more specific declaration already recorded.
                        let _ = existing;
                    }
                    Some(existing) => {
                        warn!(
                            field = %path,
                            "declared with conflicting types across scopes; indexing as {:?}",
                            existing.kind
                        );
                    }
                    None => fields.push(IndexedField {
                        name: path.clone(),
                        kind,
                    }),
                }
            }
        };

        add(&self.root);
        for (_, schema) in &self.scopes {
            add(schema);
        }

        fields.sort_by(|a, b| a.name.cmp(&b.name));
        fields
    }

    /// Resolve a possibly-partial directory reference to concrete scope directories.
    ///
    /// An exact match wins outright. Otherwise every directory whose trailing segments
    /// match is returned, so the caller can report an ambiguity rather than guessing —
    /// the same contract `get_document` offers for files.
    pub fn match_scope_dirs(&self, needle: &Path) -> Vec<PathBuf> {
        if needle.as_os_str().is_empty() {
            return vec![PathBuf::new()];
        }

        let mut candidates: Vec<&PathBuf> = self.raw.keys().collect();
        candidates.sort();

        if let Some(exact) = candidates.iter().find(|dir| *dir == &needle) {
            return vec![(*exact).clone()];
        }

        candidates
            .into_iter()
            .filter(|dir| dir.ends_with(needle))
            .cloned()
            .collect()
    }

    /// [`lint_raw`] over every schema file this cache was built from, by governing
    /// directory (#347).
    /// Includes [`lint_cascade`]'s findings for each file against what its parent
    /// directory resolves to (#348).
    pub fn lint_findings(&self) -> Vec<LintFinding> {
        let empty = ResolvedSchema::default();
        self.raw
            .iter()
            .flat_map(|(rel_dir, file)| {
                let is_root = rel_dir.as_os_str().is_empty();
                // The root file replaces the config-derived root: it inherits nothing.
                let inherited = if is_root {
                    &empty
                } else {
                    let parent = rel_dir.parent().unwrap_or(Path::new(""));
                    nearest_schema(&self.scopes, parent).unwrap_or(&self.root)
                };
                lint_raw(file, is_root)
                    .into_iter()
                    .chain(lint_cascade(inherited, file))
                    .map(move |warning| (rel_dir.clone(), warning))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Scopes that declare their own schema, shallowest first.
    #[cfg(test)]
    pub fn scope_paths(&self) -> impl Iterator<Item = &PathBuf> {
        self.scopes.iter().map(|(dir, _)| dir)
    }

    /// Every field path declared anywhere in the tree — the root and every
    /// scope — for `search`'s unknown-filter-field check and for `get_schema`'s
    /// `other_fields_in_use`, so the two agree on what counts as undeclared.
    ///
    /// Includes the paths through a list's element schema (`npcs[]`, `npcs[].role`),
    /// which the flat view leaves out: the metadata projection writes `npcs[].role`
    /// rows, and a correlated `npcs[]` filter names those sub-fields (#349). The flat
    /// view itself — what Qdrant payload indexes and the fingerprint read — is
    /// unchanged.
    pub fn declared_field_paths(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for schema in std::iter::once(self.root()).chain(self.scopes.iter().map(|(_, s)| s)) {
            for (name, node) in schema.fields.tree() {
                collect_declared_paths(name, node, &mut out);
            }
        }
        out
    }

    /// Every type some scope (the root included) gives the values stored under
    /// `field`, through the flat view or, for a `[]` path, the tree — what `search`
    /// reads to tell a range filter on a field that stores no number from one that
    /// does, and a date field from others (#349). A `list` contributes its `items`
    /// type, which its rows are projected under, or nothing when it has none (its
    /// elements are guessed from their JSON type). Empty when no scope types it.
    pub fn declared_types(&self, field: &str) -> BTreeSet<FieldType> {
        std::iter::once(self.root())
            .chain(self.scopes.iter().map(|(_, s)| s))
            .filter_map(|schema| {
                let def = schema
                    .fields
                    .get(field)
                    .or_else(|| schema.fields.lookup(field))?;
                match def.ty? {
                    FieldType::List => def.items.as_deref().and_then(|items| items.ty),
                    ty => Some(ty),
                }
            })
            .collect()
    }

    /// The closed value set a `search` filter on `field` is checked against,
    /// or `None` when the field is open (no value check).
    ///
    /// Governing scopes: `path_needle` is `search`'s `path_prefix` exactly as
    /// `retrieval::normalize_path_needle` returns it — the one needle search
    /// and enumeration match on — and is only case-folded here, never trimmed
    /// again: `/food` is a fragment of a deeper path (`lifestyle/food/a.md`),
    /// not of a top-level `food/a.md`, so it does not name the `food/` scope
    /// the way `food` does. It is a case-insensitive substring of a document
    /// path, so it narrows to the scopes whose directory contains the needle
    /// or is contained in it (`food/` and `food/recipes/` for `food`;
    /// `food/recipes/` for `food/recipes/pasta`). A needle no scope directory
    /// matches (a file-name fragment), or no needle, governs by the union of
    /// every scope and the root. The field is closed only when at least one
    /// governing scope gives it `values` and none declares it without them;
    /// the result is the union of those sets. A scope that does not declare
    /// the field at all does not open it: the caller also accepts any value
    /// documents actually use, so a value this rejects matches nothing anyway.
    pub fn filter_closed_values(
        &self,
        field: &str,
        path_needle: Option<&str>,
    ) -> Option<BTreeSet<String>> {
        let all: Vec<(String, &ResolvedSchema)> =
            std::iter::once((scope_label(Path::new("")), self.root()))
                .chain(
                    self.scopes
                        .iter()
                        .filter(|(dir, _)| !dir.as_os_str().is_empty())
                        .map(|(dir, schema)| (scope_label(dir), schema)),
                )
                .collect();
        let needle = path_needle.map(str::to_lowercase).filter(|n| !n.is_empty());
        let narrowed: Vec<&ResolvedSchema> = match &needle {
            Some(needle) => all
                .iter()
                .filter(|(label, _)| label != "/")
                .filter(|(label, _)| {
                    let label = label.to_lowercase();
                    label.contains(needle.as_str())
                        || format!("{needle}/").starts_with(label.as_str())
                })
                .map(|(_, schema)| *schema)
                .collect(),
            None => Vec::new(),
        };
        let governing: Vec<&ResolvedSchema> = if narrowed.is_empty() {
            all.iter().map(|(_, schema)| *schema).collect()
        } else {
            narrowed
        };

        let mut closed: Option<BTreeSet<String>> = None;
        for schema in governing {
            // The flat view, then the tree for a path through a list's elements
            // (`npcs[].role`, #349), which the flat view does not hold.
            match schema
                .fields
                .get(field)
                .or_else(|| schema.fields.lookup(field))
            {
                None => {}
                Some(FieldDef { values: None, .. }) => return None,
                Some(FieldDef {
                    values: Some(values),
                    ..
                }) => closed
                    .get_or_insert_with(BTreeSet::new)
                    .extend(values.iter().cloned()),
            }
        }
        closed
    }
}

/// Every declared path at or below `node` (at `path`), through `fields` (`a.b`) and
/// `items` (`a[]`, `a[].b`) alike, for [`SchemaCache::declared_field_paths`].
fn collect_declared_paths(path: &str, node: &FieldDef, out: &mut BTreeSet<String>) {
    if node.declared_in.is_some() {
        out.insert(path.to_string());
    }
    for (name, child) in &node.fields {
        collect_declared_paths(&format!("{path}.{name}"), child, out);
    }
    if let Some(items) = &node.items {
        collect_declared_paths(&format!("{path}[]"), items, out);
    }
}

/// Whether `ancestor` is `dir` or one of its parents.
fn path_covers(ancestor: &Path, dir: &Path) -> bool {
    ancestor.as_os_str().is_empty() || dir == ancestor || dir.starts_with(ancestor)
}

/// Deepest scope covering `dir`.
fn nearest_schema<'a>(
    scopes: &'a [(PathBuf, ResolvedSchema)],
    dir: &Path,
) -> Option<&'a ResolvedSchema> {
    scopes
        .iter()
        .rev()
        .find(|(scope, _)| path_covers(scope, dir))
        .map(|(_, schema)| schema)
}

/// Recursively collect `(relative dir, absolute schema file)` pairs, skipping
/// (and not descending into) any directory `walk` does not read — hidden ones and
/// ones `indexing.exclude` rules out entirely (#272).
fn collect_schema_files(
    root: &Path,
    dir: &Path,
    walk: &SchemaWalkFilter,
    out: &mut Vec<(PathBuf, PathBuf)>,
) {
    let rel_dir = dir.strip_prefix(root).unwrap_or(Path::new(""));
    if !walk.reads_dir(rel_dir) {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };

        if file_type.is_symlink() {
            continue;
        }

        if file_type.is_dir() {
            collect_schema_files(root, &path, walk, out);
        } else if file_type.is_file()
            && SCHEMA_FILE_NAMES
                .iter()
                .any(|name| entry.file_name() == *name)
        {
            out.push((rel_dir.to_path_buf(), path));
        }
    }
}

/// The text of the schema file candidate at `path`, read the way
/// [`collect_schema_files`] finds one: a symlink, or anything else that is not a
/// regular file, is not a schema file at all (`Ok(None)`) and is neither followed nor
/// read. Schema files arrive through git, which can plant a link at a schema file's
/// name; following it would read outside the tree, and a parse error would echo what
/// it points at to the caller. A regular file over [`MAX_SCHEMA_FILE_BYTES`] is an
/// error on its metadata size alone, before it is read.
fn read_schema_candidate(path: &Path) -> Result<Option<String>, String> {
    let meta = match std::fs::symlink_metadata(path) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.to_string()),
    };
    if !meta.file_type().is_file() {
        return Ok(None);
    }
    if meta.len() > MAX_SCHEMA_FILE_BYTES {
        return Err(over_size_limit_reason(meta.len()));
    }
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        // Removed between the stat and the read.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Value lookup and type checking
// ---------------------------------------------------------------------------

/// Look up a dot-path inside parsed frontmatter.
pub fn get_by_dotpath<'a>(
    frontmatter: &'a HashMap<String, Value>,
    path: &str,
) -> Option<&'a Value> {
    let mut segments = path.split('.');
    let first = segments.next()?;
    let mut current = frontmatter.get(first)?;
    for segment in segments {
        current = current.as_object()?.get(segment)?;
    }
    Some(current)
}

/// Insert a value at a dot-path, creating intermediate objects.
pub fn set_by_dotpath(frontmatter: &mut HashMap<String, Value>, path: &str, value: Value) {
    let segments: Vec<&str> = path.split('.').collect();
    if segments.len() == 1 {
        frontmatter.insert(segments[0].to_string(), value);
        return;
    }

    let entry = frontmatter
        .entry(segments[0].to_string())
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    let Some(mut cursor) = entry.as_object_mut() else {
        warn!(
            path,
            "cannot apply schema default: '{}' is not an object in this document", segments[0]
        );
        return;
    };
    for segment in &segments[1..segments.len() - 1] {
        let next = cursor
            .entry(*segment)
            .or_insert_with(|| Value::Object(serde_json::Map::new()));
        match next.as_object_mut() {
            Some(map) => cursor = map,
            None => {
                warn!(
                    path,
                    "cannot apply schema default: '{}' is not an object in this document", segment
                );
                return;
            }
        }
    }
    cursor.insert(segments[segments.len() - 1].to_string(), value);
}

fn is_date(s: &str) -> bool {
    chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").is_ok()
}

fn is_timestamp(s: &str) -> bool {
    chrono::DateTime::parse_from_rfc3339(s).is_ok()
}

/// Describe a JSON value's kind for error messages.
fn kind_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(n) => {
            if n.is_f64() {
                "a decimal number"
            } else {
                "an integer"
            }
        }
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "an object",
    }
}

/// Check a value against a declared type. Returns a human-readable reason on failure.
pub fn check_type(ty: FieldType, value: &Value) -> Result<(), String> {
    let kind = ty.kind();
    if (kind.accepts)(value) {
        Ok(())
    } else {
        Err(format!(
            "expected {}, got {}",
            kind.describe,
            kind_of(value)
        ))
    }
}

/// Check a value against a closed set of permitted values, with pre-cascade semantics,
/// for fields whose type was never declared.
///
/// Arrays are checked element-wise, so a tag list satisfies the set when every tag does.
///
/// The legacy `allowed` map only ever enforced against strings and the string elements
/// of an array; numbers, booleans, and non-string elements were exempt. Preserving that
/// exactly is what keeps a config-only deployment's validation outcomes unchanged.
///
/// This is the *lenient* counterpart to [`check_values`] — same `values` concept, two
/// enforcement regimes. `validate::field_errors` is the dispatch point: it calls this
/// function when `def.ty` is `None` and `check_values` when `def.ty` is `Some(_)`,
/// regardless of which config surface (`config.yaml` `allowed` vs schema file
/// `values`) produced the field. The split is deliberate (see [`RawFieldDef::values`]
/// for the full rationale) — do not converge these two functions.
pub fn check_values_lenient(value: &Value, permitted: Option<&[String]>) -> Result<(), String> {
    let Some(permitted) = permitted else {
        return Ok(());
    };

    // Non-scalars are exempt, matching the pre-cascade validator, which logged when it
    // declined to enforce. Keep that signal.
    if !matches!(value, Value::String(_) | Value::Array(_)) {
        debug!(
            "skipping value enforcement on a non-string, non-array value (legacy \
             `allowed` semantics)"
        );
    }

    let check_string = |s: &String| -> Result<(), String> {
        if permitted.contains(s) {
            Ok(())
        } else {
            Err(format!(
                "'{}' is not permitted here (allowed: {})",
                s,
                permitted.join(", ")
            ))
        }
    };

    match value {
        Value::String(s) => check_string(s),
        Value::Array(items) => items.iter().try_for_each(|item| match item {
            Value::String(s) => check_string(s),
            _ => Ok(()),
        }),
        _ => Ok(()),
    }
}

/// Check a value against a closed set of permitted values, for a field with an
/// explicitly declared type (`type: enum` or `type: list`).
///
/// Every scalar is canonicalized to text and compared, regardless of JSON type — unlike
/// [`check_values_lenient`], nothing is exempt. See that function's doc for why the two
/// differ and where the split is made.
pub fn check_values(value: &Value, permitted: &[String]) -> Result<(), String> {
    let matches = |v: &Value| -> Result<(), String> {
        let as_text = crate::document_fields::canonical_text(v)
            .ok_or_else(|| format!("{} cannot be checked against a value list", kind_of(v)))?;
        if permitted.iter().any(|p| p == &as_text) {
            Ok(())
        } else {
            Err(format!(
                "'{}' is not permitted here (allowed: {})",
                as_text,
                permitted.join(", ")
            ))
        }
    };

    match value {
        Value::Array(items) => items.iter().try_for_each(matches),
        other => matches(other),
    }
}

/// The flat dot-path resolver the field tree replaced (#346), kept verbatim as a
/// reference: `resolver_equivalence` tests assert the tree's flat view, origins and
/// fingerprint equal what this computes, so the refactor provably changed nothing a
/// consumer of the flat view can observe.
#[cfg(test)]
mod flat_reference {
    use super::*;

    /// The flat model's merged definition.
    #[derive(Debug, Clone, PartialEq)]
    pub(super) struct RefDef {
        pub ty: Option<FieldType>,
        pub required: bool,
        pub indexed: bool,
        pub values: Option<Vec<String>>,
        pub default: Option<Value>,
        pub open: bool,
    }

    #[derive(Debug, Clone, Default, PartialEq)]
    pub(super) struct RefResolved {
        pub fields: BTreeMap<String, RefDef>,
        pub origin: BTreeMap<String, String>,
    }

    impl SchemaFile {
        /// Flatten nested authoring sugar into dot-path entries.
        pub(super) fn flattened(&self) -> BTreeMap<String, RawFieldDef> {
            let mut out = BTreeMap::new();
            for (name, raw) in &self.fields {
                flatten_raw(name, raw, &mut out);
            }
            out
        }
    }

    fn flatten_raw(path: &str, raw: &RawFieldDef, out: &mut BTreeMap<String, RawFieldDef>) {
        if let Some(children) = &raw.fields {
            // The container itself is still a definition (it may declare `open: false`),
            // but without its nested children, which become their own entries.
            let mut container = raw.clone();
            container.fields = None;
            if container.ty.is_none() {
                container.ty = Some(FieldType::Object);
            }
            out.insert(path.to_string(), container);
            for (name, child) in children {
                flatten_raw(&format!("{path}.{name}"), child, out);
            }
        } else {
            out.insert(path.to_string(), raw.clone());
        }
    }

    fn merged(raw: &RawFieldDef, inherited: Option<&RefDef>, origin: &str, path: &str) -> RefDef {
        RefDef {
            ty: raw.ty.or(inherited.and_then(|f| f.ty)),
            required: raw
                .required
                .unwrap_or(inherited.is_some_and(|f| f.required)),
            indexed: raw.indexed.unwrap_or(inherited.is_some_and(|f| f.indexed)),
            values: FieldDef::merge_values(
                raw,
                inherited.and_then(|f| f.values.as_deref()),
                origin,
                path,
            ),
            default: raw
                .default
                .clone()
                .or_else(|| inherited.and_then(|f| f.default.clone())),
            open: raw.open.unwrap_or(inherited.is_none_or(|f| f.open)),
        }
    }

    pub(super) fn from_config(config: &FrontmatterConfig) -> RefResolved {
        let blank = || RefDef {
            ty: None,
            required: false,
            indexed: false,
            values: None,
            default: None,
            open: true,
        };
        let mut fields: BTreeMap<String, RefDef> = BTreeMap::new();
        for name in &config.required {
            fields.entry(name.clone()).or_insert_with(blank).required = true;
        }
        for name in &config.indexed_fields {
            fields.entry(name.clone()).or_insert_with(blank).indexed = true;
        }
        for (name, default) in &config.defaults {
            fields.entry(name.clone()).or_insert_with(blank).default =
                Some(Value::String(default.clone()));
        }
        for (name, values) in &config.allowed {
            fields.entry(name.clone()).or_insert_with(blank).values = Some(values.clone());
        }
        let origin = fields
            .keys()
            .map(|k| (k.clone(), scope_label(Path::new(""))))
            .collect();
        RefResolved { fields, origin }
    }

    pub(super) fn merged_with(base: &RefResolved, child: &SchemaFile, origin: &str) -> RefResolved {
        let mut fields = base.fields.clone();
        let mut origins = base.origin.clone();
        for (path, raw) in child.flattened() {
            let def = merged(&raw, base.fields.get(&path), origin, &path);
            origins.insert(path.clone(), origin.to_string());
            fields.insert(path, def);
        }
        RefResolved {
            fields,
            origin: origins,
        }
    }

    /// `merge_cascade` over the flat model: shallowest first, a root file replacing
    /// the config root, every other scope merging onto its nearest ancestor.
    pub(super) fn merge_cascade(
        files: &BTreeMap<PathBuf, SchemaFile>,
        config_root: &RefResolved,
    ) -> Vec<(PathBuf, RefResolved)> {
        let mut order: Vec<&PathBuf> = files.keys().collect();
        order.sort_by(|a, b| {
            a.components()
                .count()
                .cmp(&b.components().count())
                .then_with(|| a.cmp(b))
        });
        let mut scopes: Vec<(PathBuf, RefResolved)> = Vec::new();
        for rel_dir in order {
            let file = &files[rel_dir];
            let origin = scope_label(rel_dir);
            let merged = if rel_dir.as_os_str().is_empty() {
                merged_with(&RefResolved::default(), file, &origin)
            } else {
                let parent = scopes
                    .iter()
                    .rev()
                    .find(|(scope, _)| path_covers(scope, rel_dir))
                    .map(|(_, schema)| schema)
                    .unwrap_or(config_root);
                merged_with(parent, file, &origin)
            };
            scopes.push((rel_dir.clone(), merged));
        }
        scopes
    }

    pub(super) fn fingerprint(schema: &RefResolved) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        for (path, def) in &schema.fields {
            hasher.update(path.as_bytes());
            hasher.update([0u8]);
            hasher.update(format!("{:?}", def.ty).as_bytes());
            hasher.update([def.required as u8, def.indexed as u8, def.open as u8]);
            hasher.update([def.values.is_some() as u8]);
            if let Some(values) = &def.values {
                let mut sorted = values.clone();
                sorted.sort();
                for value in sorted {
                    hasher.update(value.as_bytes());
                    hasher.update([1u8]);
                }
            }
            if let Some(default) = &def.default {
                hasher.update(default.to_string().as_bytes());
            }
            hasher.update([0xffu8]);
        }
        hex::encode(hasher.finalize())
    }

    /// Panic unless `new`'s flat view, origins and fingerprint equal `old`'s.
    pub(super) fn assert_equivalent(new: &ResolvedSchema, old: &RefResolved, context: &str) {
        let new_fields: BTreeMap<String, RefDef> = new
            .fields
            .iter()
            .map(|(path, def)| {
                (
                    path.clone(),
                    RefDef {
                        ty: def.ty,
                        required: def.required,
                        indexed: def.indexed,
                        values: def.values.clone(),
                        default: def.default.clone(),
                        open: def.open,
                    },
                )
            })
            .collect();
        assert_eq!(new_fields, old.fields, "{context}: flat fields differ");
        assert_eq!(new.origin(), &old.origin, "{context}: origins differ");
        assert_eq!(
            new.fingerprint(),
            fingerprint(old),
            "{context}: fingerprints differ"
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use tempfile::TempDir;

    fn write_schema(root: &Path, dir: &str, yaml: &str) {
        let target = if dir.is_empty() {
            root.to_path_buf()
        } else {
            root.join(dir)
        };
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join(SCHEMA_FILE_NAME), yaml).unwrap();
    }

    fn empty_config() -> FrontmatterConfig {
        FrontmatterConfig::default()
    }

    #[test]
    fn declared_paths_reach_list_elements_but_the_flat_view_does_not() {
        let tmp = TempDir::new().unwrap();
        write_schema(
            tmp.path(),
            "",
            "fields:\n  npcs:\n    type: list\n    items:\n      type: object\n      fields:\n        \
             role:\n          type: enum\n          values: [guard, cook]\n          indexed: true\n",
        );
        write_schema(tmp.path(), "food", "fields:\n  status:\n    type: text\n");
        let cache = SchemaCache::build_for_test(tmp.path(), &empty_config());

        let declared = cache.declared_field_paths();
        for path in ["npcs", "npcs[]", "npcs[].role", "status"] {
            assert!(declared.contains(path), "{path}: {declared:?}");
        }
        assert_eq!(
            cache.filter_closed_values("npcs[].role", None),
            Some(BTreeSet::from(["cook".to_string(), "guard".to_string()]))
        );
        // Qdrant payload indexes still never see a `[]` path (#350's job).
        assert!(
            cache
                .all_indexed_fields()
                .iter()
                .all(|f| !f.name.contains("[]"))
        );
        assert!(!cache.root().fields.contains_key("npcs[].role"));
    }

    // -- flattening ---------------------------------------------------------

    #[test]
    fn nested_and_flat_authoring_produce_identical_schemas() {
        let nested: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  planning:\n    type: object\n    fields:\n      prep_minutes:\n        type: integer\n        indexed: true\n",
        )
        .unwrap();
        let flat: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  planning:\n    type: object\n  planning.prep_minutes:\n    type: integer\n    indexed: true\n",
        )
        .unwrap();

        let base = ResolvedSchema::default();
        assert_eq!(
            base.merged_with(&nested, "a").fields,
            base.merged_with(&flat, "a").fields,
            "nested authoring is sugar for dot-paths"
        );
    }

    #[test]
    fn nested_container_defaults_to_object_type() {
        let file: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  planning:\n    fields:\n      rating:\n        type: integer\n",
        )
        .unwrap();
        let flattened = file.flattened();
        assert_eq!(flattened["planning"].ty, Some(FieldType::Object));
        assert_eq!(flattened["planning.rating"].ty, Some(FieldType::Integer));
    }

    // -- merge semantics ----------------------------------------------------

    #[test]
    fn deeper_scopes_inherit_shallower_fields() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  title:\n    required: true\n");
        write_schema(
            dir.path(),
            "kitchen/recipes",
            "fields:\n  cook_minutes:\n    type: integer\n",
        );

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("kitchen/recipes/chili.md"));

        assert!(schema.fields["title"].required, "root field is inherited");
        assert_eq!(schema.fields["cook_minutes"].ty, Some(FieldType::Integer));
    }

    #[test]
    fn redefinition_overrides_only_the_attributes_it_declares() {
        // This is the exact footgun the per-attribute rewrite exists to close: the
        // child only mentions `values` (and replaces it outright — no sentinel), so
        // `type` and `required` must still come from the root. Under the old
        // wholesale-replace rule this redefinition would silently drop `required`,
        // which is why the live KB's root `tags: { required: true }` never actually
        // applied anywhere — every domain redeclared `tags` for its own `values`.
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "",
            "fields:\n  status:\n    type: enum\n    required: true\n    values: [active, draft]\n",
        );
        write_schema(
            dir.path(),
            "scratch",
            "fields:\n  status:\n    values: [wip]\n",
        );

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("scratch/note.md"));

        assert_eq!(
            schema.fields["status"].values,
            Some(vec!["wip".into()]),
            "values has no sentinel, so it replaces outright"
        );
        assert_eq!(
            schema.fields["status"].ty,
            Some(FieldType::Enum),
            "type was never redeclared, so it still comes from the root"
        );
        assert!(
            schema.fields["status"].required,
            "required was never redeclared, so it still comes from the root — the \
             whole point of per-attribute inheritance"
        );
    }

    #[test]
    fn a_child_can_still_explicitly_override_an_inherited_attribute() {
        // The other half of the same rule: an attribute the child DOES mention still
        // wins, same as before. Per-attribute inheritance only changes what happens to
        // attributes the child stays silent on.
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  status:\n    required: true\n");
        write_schema(
            dir.path(),
            "scratch",
            "fields:\n  status:\n    required: false\n",
        );

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("scratch/note.md"));

        assert!(!schema.fields["status"].required);
    }

    #[test]
    fn omitting_values_entirely_inherits_the_parents_set_verbatim() {
        // A child that redeclares a DIFFERENT attribute and never mentions `values` at
        // all inherits the parent's values set unchanged — ordinary per-attribute
        // inheritance, no splicing involved (that's only for when `values` itself is
        // redeclared).
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "",
            "fields:\n  status:\n    type: enum\n    values: [active, draft]\n",
        );
        write_schema(
            dir.path(),
            "scratch",
            "fields:\n  status:\n    required: true\n",
        );

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("scratch/note.md"));

        assert_eq!(
            schema.fields["status"].values,
            Some(vec!["active".into(), "draft".into()])
        );
        assert!(schema.fields["status"].required);
    }

    #[test]
    fn per_attribute_inheritance_covers_every_attribute() {
        // One field, every attribute set at the root, a child that overrides exactly
        // one of them (`indexed`). Everything else — type, required, default, open —
        // must survive untouched.
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "",
            "fields:\n  note:\n    type: object\n    open: false\n    required: true\n    \
             indexed: false\n    default: {}\n",
        );
        write_schema(dir.path(), "child", "fields:\n  note:\n    indexed: true\n");

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("child/doc.md"));
        let note = &schema.fields["note"];

        assert_eq!(note.ty, Some(FieldType::Object), "type inherited");
        assert!(note.required, "required inherited");
        assert!(note.indexed, "indexed is the one attribute the child set");
        assert!(!note.open, "open inherited");
        assert!(note.default.is_some(), "default inherited");
    }

    #[test]
    fn three_level_cascade_with_each_level_setting_a_different_attribute() {
        // root -> domain -> subdirectory, each level touching only ONE attribute of
        // the same field. The document under the subdirectory must see all three.
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  x:\n    required: true\n");
        write_schema(dir.path(), "a", "fields:\n  x:\n    indexed: true\n");
        write_schema(
            dir.path(),
            "a/b",
            "fields:\n  x:\n    type: enum\n    values: [one, two]\n",
        );

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("a/b/doc.md"));
        let x = &schema.fields["x"];

        assert!(x.required, "from root");
        assert!(x.indexed, "from the domain level");
        assert_eq!(x.ty, Some(FieldType::Enum), "from the subdirectory");
        assert_eq!(x.values, Some(vec!["one".into(), "two".into()]));
    }

    // -- $values sentinel -----------------------------------------------------

    #[test]
    fn values_replace_by_default_with_no_sentinel() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  tags:\n    values: [a, b]\n");
        write_schema(dir.path(), "child", "fields:\n  tags:\n    values: [c]\n");

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("child/doc.md"));

        assert_eq!(
            schema.fields["tags"].values,
            Some(vec!["c".into()]),
            "no sentinel present, so the child's list replaces outright — the shell \
             `PATH=/only/this` case"
        );
    }

    #[test]
    fn leading_sentinel_splices_inherited_values_first() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  tags:\n    values: [a, b]\n");
        write_schema(
            dir.path(),
            "child",
            "fields:\n  tags:\n    values: [$values, c]\n",
        );

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("child/doc.md"));

        assert_eq!(
            schema.fields["tags"].values,
            Some(vec!["a".into(), "b".into(), "c".into()])
        );
    }

    #[test]
    fn trailing_sentinel_splices_inherited_values_last() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  tags:\n    values: [a, b]\n");
        write_schema(
            dir.path(),
            "child",
            "fields:\n  tags:\n    values: [c, $values]\n",
        );

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("child/doc.md"));

        assert_eq!(
            schema.fields["tags"].values,
            Some(vec!["c".into(), "a".into(), "b".into()]),
            "position is meaningful: the sentinel sits after 'c', so inherited values \
             land after it too"
        );
    }

    #[test]
    fn splicing_deduplicates_keeping_first_occurrence_order() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  tags:\n    values: [a, b]\n");
        write_schema(
            dir.path(),
            "child",
            "fields:\n  tags:\n    values: [b, $values, c]\n",
        );

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("child/doc.md"));

        assert_eq!(
            schema.fields["tags"].values,
            Some(vec!["b".into(), "a".into(), "c".into()]),
            "'b' keeps its first (literal, pre-sentinel) position and is not repeated \
             when the sentinel splices in the inherited set that also contains it"
        );
    }

    #[test]
    fn sentinel_with_nothing_inherited_resolves_to_only_the_literals() {
        // No ancestor declares any values for this field at all. The sentinel
        // contributes nothing (a loud warning is logged, but not asserted on here —
        // see `merge_values`'s doc for why this degrades rather than hard-erroring),
        // and any literal tokens still in the list are the complete permitted set.
        let dir = TempDir::new().unwrap();
        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let candidate: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  tags:\n    values: [$values, only]\n").unwrap();

        let effective = cache
            .resolve_with_candidate(Path::new("doc.md"), Path::new(""), &candidate)
            .unwrap();

        assert_eq!(effective.fields["tags"].values, Some(vec!["only".into()]));
    }

    #[test]
    fn sentinel_alone_with_nothing_inherited_closes_the_field_rather_than_leaving_it_unconstrained()
    {
        // The sharper edge of the same case: NOTHING resolves (no inherited values, no
        // other literals), so the field ends up `Some(vec![])` — permitting nothing —
        // rather than `None` — permitting anything. An empty closed set fails loudly
        // the moment a document sets the field; `None` would fail silently by not
        // checking at all. See `merge_values`'s doc for the full reasoning.
        let dir = TempDir::new().unwrap();
        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let candidate: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  tags:\n    values: [$values]\n").unwrap();

        let effective = cache
            .resolve_with_candidate(Path::new("doc.md"), Path::new(""), &candidate)
            .unwrap();

        assert_eq!(effective.fields["tags"].values, Some(Vec::new()));
    }

    #[test]
    fn multi_level_cascade_splices_against_the_immediately_inherited_set_not_the_root() {
        // root -> domain -> subdirectory, each splicing in turn. The subdirectory's
        // sentinel must resolve against the DOMAIN's already-merged set ([a, b]), not
        // the root's raw set ([a]) — otherwise 'b' would silently disappear for any
        // document two levels down.
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  tags:\n    values: [a]\n");
        write_schema(
            dir.path(),
            "domain",
            "fields:\n  tags:\n    values: [$values, b]\n",
        );
        write_schema(
            dir.path(),
            "domain/sub",
            "fields:\n  tags:\n    values: [$values, c]\n",
        );

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("domain/sub/doc.md"));

        assert_eq!(
            schema.fields["tags"].values,
            Some(vec!["a".into(), "b".into(), "c".into()])
        );
    }

    #[test]
    fn deprecated_extend_true_behaves_like_a_leading_sentinel() {
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "",
            "fields:\n  tags:\n    type: list\n    values: [reference, guide]\n",
        );
        write_schema(
            dir.path(),
            "kitchen",
            "fields:\n  tags:\n    type: list\n    required: true\n    extend: true\n    values: [recipe]\n",
        );

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("kitchen/chili.md"));

        assert_eq!(
            schema.fields["tags"].values,
            Some(vec!["reference".into(), "guide".into(), "recipe".into()]),
            "extend: true == a leading $values placeholder"
        );
        assert!(
            schema.fields["tags"].required,
            "extend only ever affected values; every other attribute follows the same \
             per-attribute rule as always, and here the child explicitly sets it"
        );
    }

    #[test]
    fn an_unrecognized_dollar_token_is_a_loud_parse_error() {
        let file: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  tags:\n    values: [$typo, a]\n").unwrap();
        let err = file.validate_self().unwrap_err();
        assert!(err.contains("$typo"), "names the offending token: {err}");
        assert!(err.contains("$values"), "names what IS recognized: {err}");
    }

    #[test]
    fn more_than_one_sentinel_in_a_values_list_is_a_parse_error() {
        let file: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  tags:\n    values: [$values, a, $values]\n")
                .unwrap();
        let err = file.validate_self().unwrap_err();
        assert!(err.contains("tags"), "got: {err}");
    }

    #[test]
    fn combining_extend_true_with_an_explicit_sentinel_is_a_parse_error() {
        let file: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  tags:\n    extend: true\n    values: [$values, a]\n",
        )
        .unwrap();
        let err = file.validate_self().unwrap_err();
        assert!(err.contains("extend"), "got: {err}");
        assert!(err.contains("$values"), "got: {err}");
    }

    const NESTED_SCHEMA: &str = "fields:\n  planning:\n    type: object\n    open: false\n    fields:\n      method:\n        type: enum\n        open: false\n        values: [braise, oven]\n";

    fn nested_file() -> SchemaFile {
        serde_yaml_ng::from_str(NESTED_SCHEMA).unwrap()
    }

    #[test]
    fn add_values_on_a_nested_path_edits_the_nested_declaration() {
        let mut file = nested_file();
        file.apply(&SchemaEdit::AddValues {
            field: "planning.method".into(),
            values: vec!["slow-cooker".into()],
        })
        .unwrap();

        assert_eq!(
            file.fields.len(),
            1,
            "no sibling dotted key: {:?}",
            file.fields
        );
        let method = &file.fields["planning"].fields.as_ref().unwrap()["method"];
        assert_eq!(
            method.values,
            Some(vec!["braise".into(), "oven".into(), "slow-cooker".into()])
        );
        assert_eq!(method.open, Some(false), "stays closed");
        file.validate_self().unwrap();
    }

    #[test]
    fn set_remove_values_and_remove_field_reach_nested_paths() {
        let mut file = nested_file();
        file.apply(&SchemaEdit::RemoveValues {
            field: "planning.method".into(),
            values: vec!["oven".into()],
        })
        .unwrap();
        let method = &file.fields["planning"].fields.as_ref().unwrap()["method"];
        assert_eq!(method.values, Some(vec!["braise".to_string()]));

        let def: RawFieldDef = serde_json::from_value(json!({"type": "text"})).unwrap();
        file.apply(&SchemaEdit::SetField {
            field: "planning.method".into(),
            definition: Box::new(def.clone()),
        })
        .unwrap();
        assert_eq!(file.fields.len(), 1);
        assert_eq!(
            file.fields["planning"].fields.as_ref().unwrap()["method"],
            def
        );

        file.apply(&SchemaEdit::RemoveField {
            field: "planning.method".into(),
        })
        .unwrap();
        let planning = &file.fields["planning"];
        assert!(planning.fields.as_ref().unwrap().is_empty());
        assert_eq!(planning.open, Some(false), "the emptied container is kept");
    }

    #[test]
    fn a_flat_dot_path_key_is_edited_in_place() {
        let mut file: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  planning.method:\n    type: enum\n    values: [a]\n",
        )
        .unwrap();
        file.apply(&SchemaEdit::AddValues {
            field: "planning.method".into(),
            values: vec!["b".into()],
        })
        .unwrap();
        assert_eq!(file.fields.len(), 1);
        assert_eq!(
            file.fields["planning.method"].values,
            Some(vec!["a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn a_missing_parent_is_created_as_an_object_container() {
        let mut file = SchemaFile::default();
        file.apply(&SchemaEdit::AddValues {
            field: "a.b.c".into(),
            values: vec!["x".into()],
        })
        .unwrap();
        let a = &file.fields["a"];
        assert_eq!(a.ty, Some(FieldType::Object));
        assert_eq!(a.open, None, "inherits open from an ancestor scope");
        let b = &a.fields.as_ref().unwrap()["b"];
        assert_eq!(b.ty, Some(FieldType::Object));
        assert_eq!(
            b.fields.as_ref().unwrap()["c"].values,
            Some(vec!["x".to_string()])
        );
        file.validate_self().unwrap();
        assert!(file.flattened().contains_key("a.b.c"));
    }

    #[test]
    fn a_flat_dotted_key_inside_a_container_is_found_and_edited() {
        // `planning.x.y` is spelled as container `planning` + child key `x.y`.
        let mut file: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  planning:\n    type: object\n    fields:\n      x.y:\n        type: enum\n        values: [a]\n",
        )
        .unwrap();
        file.apply(&SchemaEdit::AddValues {
            field: "planning.x.y".into(),
            values: vec!["b".into()],
        })
        .unwrap();
        let children = file.fields["planning"].fields.as_ref().unwrap();
        assert_eq!(children.len(), 1, "no duplicate created: {children:?}");
        assert_eq!(
            children["x.y"].values,
            Some(vec!["a".to_string(), "b".to_string()])
        );
        file.validate_self().unwrap();
    }

    #[test]
    fn set_field_over_an_existing_container_replaces_its_children() {
        let mut file = nested_file();
        let def: RawFieldDef = serde_json::from_value(
            json!({"type": "object", "fields": {"other": {"type": "text"}}}),
        )
        .unwrap();
        file.apply(&SchemaEdit::SetField {
            field: "planning".into(),
            definition: Box::new(def),
        })
        .unwrap();
        let children = file.fields["planning"].fields.as_ref().unwrap();
        assert_eq!(children.keys().collect::<Vec<_>>(), ["other"]);
        file.validate_self().unwrap();
    }

    #[test]
    fn remove_field_on_a_doubly_declared_path_removes_the_flat_key_first() {
        // A file damaged by the pre-#268 bug: both spellings of one path. The flat key
        // is the one `update_schema` removes, leaving the nested declaration valid.
        let mut file: SchemaFile = serde_yaml_ng::from_str(&format!(
            "{NESTED_SCHEMA}  planning.method:\n    type: enum\n    values: [slow-cooker]\n"
        ))
        .unwrap();
        assert!(file.validate_self().is_err());
        file.apply(&SchemaEdit::RemoveField {
            field: "planning.method".into(),
        })
        .unwrap();
        file.validate_self().unwrap();
        let method = &file.fields["planning"].fields.as_ref().unwrap()["method"];
        assert_eq!(method.values, Some(vec!["braise".into(), "oven".into()]));
    }

    #[test]
    fn nesting_under_a_scalar_field_is_rejected() {
        let mut file: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  planning:\n    type: text\n").unwrap();
        let err = file
            .apply(&SchemaEdit::AddValues {
                field: "planning.method".into(),
                values: vec!["x".into()],
            })
            .unwrap_err();
        assert!(err.contains("not a container"), "got: {err}");
    }

    #[test]
    fn nesting_under_a_typeless_field_with_values_is_rejected() {
        // `values:` with no `type:` is a leniently-enforced scalar; nesting under it
        // would let `flatten_raw` retype it as an object.
        let mut file: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  status:\n    values: [a, b]\n").unwrap();
        let err = file
            .apply(&SchemaEdit::AddValues {
                field: "status.sub".into(),
                values: vec!["x".into()],
            })
            .unwrap_err();
        assert!(err.contains("not a container"), "got: {err}");
    }

    #[test]
    fn nested_and_flat_declarations_of_one_path_are_a_validation_error() {
        let file: SchemaFile = serde_yaml_ng::from_str(&format!(
            "{NESTED_SCHEMA}  planning.method:\n    type: enum\n    values: [slow-cooker]\n"
        ))
        .unwrap();
        let err = file.validate_self().unwrap_err();
        assert!(err.contains("planning.method"), "got: {err}");
        assert!(err.contains("more than once"), "got: {err}");
    }

    #[test]
    fn a_schema_using_the_unrecognized_token_fails_the_build() {
        // The parse-time rejection above must actually reach the cascade build, not
        // just the standalone validator.
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "bad", "fields:\n  tags:\n    values: [$oops]\n");

        let err = SchemaCache::build(dir.path(), &empty_config(), &Default::default()).unwrap_err();

        assert_eq!(err.invalid.len(), 1);
        assert_eq!(err.invalid[0].path, Path::new("bad/.schema.yaml"));
        assert!(err.invalid[0].reason.contains("$oops"), "got: {err}");
    }

    #[test]
    fn three_level_cascade_takes_the_nearest_definition() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  scope:\n    values: [root]\n");
        write_schema(dir.path(), "a", "fields:\n  scope:\n    values: [mid]\n");
        fs::create_dir_all(dir.path().join("a/b")).unwrap();

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let schema = cache.resolve_for(Path::new("a/b/doc.md"));

        assert_eq!(
            schema.fields["scope"].values,
            Some(vec!["mid".into()]),
            "the nearest ancestor wins, not the root"
        );
    }

    #[test]
    fn sibling_scopes_do_not_leak() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "a", "fields:\n  only_a:\n    required: true\n");
        write_schema(dir.path(), "b", "fields:\n  only_b:\n    required: true\n");

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());

        assert!(
            !cache
                .resolve_for(Path::new("b/doc.md"))
                .fields
                .contains_key("only_a")
        );
        assert!(
            !cache
                .resolve_for(Path::new("a/doc.md"))
                .fields
                .contains_key("only_b")
        );
    }

    #[test]
    fn intermediate_directories_resolve_to_nearest_ancestor() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  level:\n    values: [root]\n");
        write_schema(dir.path(), "a", "fields:\n  level:\n    values: [a]\n");
        write_schema(
            dir.path(),
            "a/b/c",
            "fields:\n  level:\n    values: [abc]\n",
        );

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());

        // a/b/ has no schema of its own; it must fall to a/, not root and not a/b/c.
        assert_eq!(
            cache.resolve_for(Path::new("a/b/doc.md")).fields["level"].values,
            Some(vec!["a".into()])
        );
        assert_eq!(
            cache.resolve_for(Path::new("a/b/c/doc.md")).fields["level"].values,
            Some(vec!["abc".into()])
        );
    }

    // -- backward compatibility ---------------------------------------------

    #[test]
    fn config_block_becomes_the_implicit_root_schema() {
        let mut config = FrontmatterConfig {
            required: vec!["title".into(), "description".into()],
            indexed_fields: vec!["type".into(), "tags".into()],
            ..Default::default()
        };
        config
            .allowed
            .insert("status".into(), vec!["active".into(), "draft".into()]);
        config
            .defaults
            .insert("status".into(), "active".to_string());

        let schema = ResolvedSchema::from_config(&config);

        assert!(schema.fields["title"].required);
        assert!(schema.fields["tags"].indexed);
        assert_eq!(
            schema.fields["status"].values,
            Some(vec!["active".into(), "draft".into()])
        );
        assert_eq!(
            schema.fields["status"].ty, None,
            "the adapter must NOT declare a type: doing so would subject numbers and \
             booleans to value checks the pre-cascade validator exempted"
        );
        assert_eq!(
            schema.fields["status"].default,
            Some(json!("active")),
            "defaults stay strings, exactly as before"
        );
    }

    #[test]
    fn a_tree_with_no_schema_files_uses_the_config_root() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("doc.md"), "# hi").unwrap();
        let config = FrontmatterConfig {
            required: vec!["title".into()],
            ..Default::default()
        };

        let cache = SchemaCache::build_for_test(dir.path(), &config);

        assert!(
            cache.resolve_for(Path::new("doc.md")).fields["title"].required,
            "existing deployments keep working untouched"
        );
    }

    #[test]
    fn a_root_schema_file_replaces_the_config_root_instead_of_merging_with_it() {
        // config.yaml declares two rules: `title` required, `legacy_only` indexed. A
        // root `.kb-schema.yaml` exists and redeclares `title` but says nothing about
        // `legacy_only`. Under the old (pre-issue-#91) behavior these merged, so
        // `legacy_only` would still show up, config-sourced, in the resolved root. The
        // whole point of this change is that it must not: once a root schema file
        // exists, it is authoritative and config.yaml's block stops applying.
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "",
            "fields:\n  title:\n    type: text\n    required: true\n",
        );
        let config = FrontmatterConfig {
            required: vec!["title".into()],
            indexed_fields: vec!["legacy_only".into()],
            ..Default::default()
        };

        let cache = SchemaCache::build_for_test(dir.path(), &config);
        let root = cache.root();

        assert!(
            root.fields["title"].required,
            "the root file's own rule applies"
        );
        assert!(
            !root.fields.contains_key("legacy_only"),
            "a config-only field must NOT leak into the root once a root schema file \
             exists — that would mean the KB's validation rules depend on whichever \
             config.yaml the deploying host happens to have"
        );
    }

    #[test]
    fn a_root_schema_files_override_reaches_subdirectories_not_the_config_leftover() {
        // Subdirectories inherit from the root scope. That inherited root must be the
        // REPLACED (root-file-only) schema, not a config-merged one — otherwise a
        // config-only field would reach every document in the tree via inheritance
        // even though the root schema itself no longer reports it.
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  title:\n    required: true\n");
        fs::create_dir_all(dir.path().join("food")).unwrap();
        let config = FrontmatterConfig {
            indexed_fields: vec!["legacy_only".into()],
            ..Default::default()
        };

        let cache = SchemaCache::build_for_test(dir.path(), &config);
        let resolved = cache.resolve_for(Path::new("food/chili.md"));

        assert!(
            resolved.fields["title"].required,
            "root rule still cascades"
        );
        assert!(
            !resolved.fields.contains_key("legacy_only"),
            "the config leftover must not reach subdirectories through inheritance either"
        );
    }

    #[test]
    fn resolve_with_candidate_replaces_config_when_creating_a_root_schema_file() {
        // update_schema's dry-run path (`resolve_with_candidate`) must apply the same
        // override policy as `build`: proposing a brand-new root `.kb-schema.yaml`
        // must not silently keep enforcing whatever config.yaml declared but the
        // candidate omits.
        let dir = TempDir::new().unwrap();
        let config = FrontmatterConfig {
            required: vec!["legacy_required".into()],
            ..Default::default()
        };
        let cache = SchemaCache::build_for_test(dir.path(), &config);

        let candidate: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  title:\n    required: true\n").unwrap();
        let effective = cache
            .resolve_with_candidate(Path::new("doc.md"), Path::new(""), &candidate)
            .expect("root always governs a root-level document");

        assert!(effective.fields["title"].required);
        assert!(
            !effective.fields.contains_key("legacy_required"),
            "the candidate root file replaces config, so a field only config declares \
             must not appear as still-enforced in the dry run"
        );
    }

    #[test]
    fn resolve_with_candidate_still_falls_back_to_config_with_no_root_file_in_play() {
        // The other half of the same policy: editing a NON-root scope while no root
        // schema file exists anywhere must still fall back to the config-derived root
        // for fields the edit itself doesn't touch — the deprecated fallback keeps
        // working until a root file actually shows up.
        let dir = TempDir::new().unwrap();
        let config = FrontmatterConfig {
            required: vec!["title".into()],
            ..Default::default()
        };
        let cache = SchemaCache::build_for_test(dir.path(), &config);

        let candidate: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  cook_minutes:\n    type: integer\n").unwrap();
        let effective = cache
            .resolve_with_candidate(Path::new("food/chili.md"), Path::new("food"), &candidate)
            .expect("food/ covers this document");

        assert!(
            effective.fields["title"].required,
            "no root schema file exists, so the config fallback must still reach this document"
        );
        assert_eq!(
            effective.fields["cook_minutes"].ty,
            Some(FieldType::Integer)
        );
    }

    // -- invalid schemas ----------------------------------------------------

    #[test]
    fn a_malformed_schema_fails_the_whole_build() {
        // A valid sibling and root do not rescue the tree: there is no partial
        // cache, so no document is ever validated under rules known to be wrong.
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  title:\n    required: true\n");
        write_schema(dir.path(), "bad", "fields: [this is not a map\n");
        write_schema(dir.path(), "good", "fields:\n  ok:\n    required: true\n");

        let err = SchemaCache::build(dir.path(), &empty_config(), &Default::default()).unwrap_err();

        let paths: Vec<_> = err.invalid.iter().map(|f| f.path.clone()).collect();
        assert_eq!(paths, vec![PathBuf::from("bad/.schema.yaml")]);
    }

    /// #272: only the indexed tree is walked. A broken schema under a directory
    /// `indexing.exclude` rules out entirely, or under a hidden directory, is never
    /// read — it neither fails the build nor contributes rules — while the same
    /// file under an included directory still fails it.
    #[test]
    fn schemas_outside_the_indexed_tree_are_never_read() {
        let indexing = crate::config::IndexingConfig {
            exclude: vec!["templates/**".into(), "archive/*/*.md".into()],
            ..Default::default()
        };
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  title:\n    required: true\n");
        write_schema(dir.path(), "templates", "fields: [this is not a map\n");
        write_schema(dir.path(), "templates/deep", "fields: [nor this\n");
        write_schema(dir.path(), ".obsidian", "fields: [nor this\n");
        write_schema(
            dir.path(),
            "notes",
            "fields:\n  topic:\n    required: true\n",
        );

        let cache = SchemaCache::build_for_test_with(dir.path(), &empty_config(), &indexing);
        let scopes: Vec<_> = cache.scope_paths().cloned().collect();
        assert_eq!(scopes, vec![PathBuf::new(), PathBuf::from("notes")]);

        // `archive/*/*.md` still indexes `archive/x.md`, so archive/ is not
        // wholly excluded and its schema is read — and an invalid one fails.
        write_schema(dir.path(), "archive", "fields: [broken\n");
        let err = SchemaCache::build(dir.path(), &empty_config(), &indexing).unwrap_err();
        let paths: Vec<_> = err.invalid.iter().map(|f| f.path.clone()).collect();
        assert_eq!(paths, vec![PathBuf::from("archive/.schema.yaml")]);

        // With no exclude rule for it, templates/ is part of the tree again.
        let err = SchemaCache::build(dir.path(), &empty_config(), &Default::default()).unwrap_err();
        let paths: Vec<_> = err.invalid.iter().map(|f| f.path.clone()).collect();
        assert_eq!(
            paths,
            vec![
                PathBuf::from("archive/.schema.yaml"),
                PathBuf::from("templates/.schema.yaml"),
                PathBuf::from("templates/deep/.schema.yaml"),
            ]
        );
    }

    #[test]
    fn schema_walk_filter_reads_only_the_indexed_tree() {
        let walk = SchemaWalkFilter::from_config(&crate::config::IndexingConfig {
            exclude: vec!["templates/**".into(), "**/drafts/**".into()],
            ..Default::default()
        });
        for dir in ["", "notes", "notes/sub", "templatesque"] {
            assert!(walk.reads_dir(Path::new(dir)), "{dir:?}");
        }
        for dir in [
            "templates",
            "templates/a",
            "notes/drafts",
            ".git",
            "a/.hidden",
        ] {
            assert!(!walk.reads_dir(Path::new(dir)), "{dir:?}");
        }
        assert!(walk.governs(Path::new("notes/.kb-schema.yaml")));
        assert!(!walk.governs(Path::new("templates/.kb-schema.yaml")));
        assert!(!walk.governs(Path::new("notes/a.md")));
    }

    #[test]
    fn build_error_lists_every_invalid_file_not_just_the_first() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  title:\n    required: true\n");
        write_schema(dir.path(), "parse", "not: a: valid: mapping:\n");
        write_schema(dir.path(), "dedup", "dedup:\n  threshold: 2.0\n");
        write_schema(dir.path(), "unknown", "fieldz:\n  title: {}\n");
        let huge = format!("fields:\n{}", "  a: {}\n".repeat(60_000));
        assert!(huge.len() as u64 > super::MAX_SCHEMA_FILE_BYTES);
        write_schema(dir.path(), "big", &huge);
        write_schema(dir.path(), "good", "fields:\n  ok:\n    required: true\n");

        let started = std::time::Instant::now();
        let err = SchemaCache::build(dir.path(), &empty_config(), &Default::default()).unwrap_err();
        let elapsed = started.elapsed();

        let mut by_path: BTreeMap<String, String> = err
            .invalid
            .iter()
            .map(|f| (f.path.to_string_lossy().into_owned(), f.reason.clone()))
            .collect();
        assert_eq!(by_path.len(), 4, "got: {err}");
        assert!(by_path.remove("parse/.schema.yaml").is_some());
        assert!(
            by_path
                .remove("dedup/.schema.yaml")
                .unwrap()
                .contains("dedup.threshold")
        );
        assert!(
            by_path
                .remove("unknown/.schema.yaml")
                .unwrap()
                .contains("fieldz")
        );
        assert!(
            by_path
                .remove("big/.schema.yaml")
                .unwrap()
                .contains("byte limit")
        );

        let shown = err.to_string();
        for dir in ["parse", "dedup", "unknown", "big"] {
            assert!(shown.contains(&format!("{dir}/.schema.yaml")), "{shown}");
        }
        assert!(!shown.contains("good/"), "{shown}");
        // Schema files arrive via git sync and are untrusted; deeply nested YAML
        // costs superlinear parse time, so the size cap must short-circuit first.
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "the cap must short-circuit before parsing, took {elapsed:?}"
        );
    }

    #[test]
    fn build_error_lists_invalid_files_sorted_by_path() {
        let dir = TempDir::new().unwrap();
        for d in ["zeta", "alpha", "mid/deep", "beta"] {
            write_schema(dir.path(), d, "fields: [not a map\n");
        }

        let err = SchemaCache::build(dir.path(), &empty_config(), &Default::default()).unwrap_err();

        let paths: Vec<_> = err.invalid.iter().map(|f| f.path.clone()).collect();
        let mut sorted = paths.clone();
        sorted.sort();
        assert_eq!(paths, sorted);
        assert_eq!(paths.len(), 4);
    }

    #[test]
    fn a_normal_sized_schema_is_still_accepted() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "ok", "fields:\n  title:\n    required: true\n");
        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());

        assert!(cache.resolve_for(Path::new("ok/doc.md")).fields["title"].required);
    }

    #[test]
    fn apply_rebuild_keeps_the_last_good_cache_and_tracks_the_status_error() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  title:\n    required: true\n");
        let shared: SharedSchemaCache = Arc::new(RwLock::new(Arc::new(
            SchemaCache::build_for_test(dir.path(), &empty_config()),
        )));
        let good = load_shared(&shared);
        let status = crate::status::IndexStatus::new();

        write_schema(dir.path(), "bad", "fields: [not a map\n");
        let built = SchemaCache::build(dir.path(), &empty_config(), &Default::default());
        assert!(!apply_rebuild(&shared, built, &status, "test"));
        assert!(
            Arc::ptr_eq(&good, &load_shared(&shared)),
            "previous cache kept"
        );
        let recorded = status.schema_error().expect("refusal recorded");
        assert_eq!(recorded.files[0].path, Path::new("bad/.schema.yaml"));

        fs::remove_dir_all(dir.path().join("bad")).unwrap();
        write_schema(dir.path(), "", "fields:\n  title:\n    required: false\n");
        let built = SchemaCache::build(dir.path(), &empty_config(), &Default::default());
        assert!(apply_rebuild(&shared, built, &status, "test"));
        assert!(!load_shared(&shared).root().fields["title"].required);
        assert!(status.schema_error().is_none(), "cleared by a good rebuild");
    }

    #[test]
    fn is_schema_file_path_matches_the_file_name_only() {
        assert!(is_schema_file_path(Path::new(".kb-schema.yaml")));
        assert!(is_schema_file_path(Path::new("a/b/.kb-schema.yaml")));
        assert!(!is_schema_file_path(Path::new("a/.kb-schema.yaml.md")));
        assert!(!is_schema_file_path(Path::new(".kb-schema.yaml/doc.md")));
        assert!(is_schema_file_path(Path::new(".schema.yaml")));
        assert!(is_schema_file_path(Path::new("a/b/.schema.yaml")));
        assert!(!is_schema_file_path(Path::new("a/.schema.yaml.md")));
        assert!(!is_schema_file_path(Path::new("a/schema.yaml")));
    }

    #[test]
    fn scope_label_names_the_directory_with_root_as_slash() {
        assert_eq!(scope_label(Path::new("")), "/");
        assert_eq!(scope_label(Path::new("food/recipes")), "food/recipes/");
    }

    #[test]
    fn either_schema_file_name_is_discovered_on_its_own() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "new", "fields:\n  a:\n    required: true\n");
        fs::create_dir_all(dir.path().join("old")).unwrap();
        fs::write(
            dir.path().join("old").join(LEGACY_SCHEMA_FILE_NAME),
            "fields:\n  b:\n    required: true\n",
        )
        .unwrap();

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let new = cache.resolve_for(Path::new("new/doc.md"));
        assert!(new.fields["a"].required);
        assert_eq!(new.origin()["a"], "new/");
        let old = cache.resolve_for(Path::new("old/doc.md"));
        assert!(old.fields["b"].required);
        assert_eq!(old.origin()["b"], "old/");

        let (_, name) = cache.raw_file_at(Path::new("new")).unwrap();
        assert_eq!(name, Some(SCHEMA_FILE_NAME));
        let (file, name) = cache.raw_file_at(Path::new("old")).unwrap();
        assert_eq!(name, Some(LEGACY_SCHEMA_FILE_NAME));
        assert!(file.fields.contains_key("b"));
        let (file, name) = cache.raw_file_at(Path::new("none")).unwrap();
        assert_eq!(name, None);
        assert!(file.fields.is_empty());
    }

    #[test]
    fn both_schema_file_names_in_one_directory_fail_the_build() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "both", "fields:\n  a:\n    required: true\n");
        fs::write(
            dir.path().join("both").join(LEGACY_SCHEMA_FILE_NAME),
            "fields:\n  a:\n    required: false\n",
        )
        .unwrap();
        write_schema(dir.path(), "fine", "fields:\n  b:\n    required: true\n");

        let err = SchemaCache::build(dir.path(), &empty_config(), &Default::default())
            .expect_err("both names in one directory must not resolve silently");
        assert_eq!(err.invalid.len(), 1, "{err}");
        assert_eq!(
            err.invalid[0].path,
            Path::new("both").join(LEGACY_SCHEMA_FILE_NAME)
        );
        assert_eq!(err.invalid[0].reason, BOTH_NAMES_REASON);

        let cache = SchemaCache::from_config_only(&empty_config());
        let cache = SchemaCache {
            root_path: dir.path().to_path_buf(),
            ..cache
        };
        assert_eq!(
            cache.raw_file_at(Path::new("both")).unwrap_err(),
            BOTH_NAMES_REASON
        );
    }

    /// A cache rooted at `root` that has walked nothing: `raw_file_at` reads from
    /// disk whatever the cache holds, so a test can shape the tree around it.
    fn cache_rooted_at(root: &Path) -> SchemaCache {
        SchemaCache {
            root_path: root.to_path_buf(),
            ..SchemaCache::from_config_only(&empty_config())
        }
    }

    #[test]
    fn a_symlinked_schema_file_is_absent_to_raw_file_at_and_never_echoed() {
        let dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        // A one-line scalar outside the tree: what a schema parse error would quote.
        let secret = outside.path().join("secret");
        fs::write(&secret, "hunter2-not-a-schema\n").unwrap();
        write_schema(dir.path(), "", "fields:\n  title:\n    required: true\n");
        for (scope, name) in [
            ("notes", SCHEMA_FILE_NAME),
            ("old", LEGACY_SCHEMA_FILE_NAME),
        ] {
            fs::create_dir_all(dir.path().join(scope).join("sub")).unwrap();
            std::os::unix::fs::symlink(&secret, dir.path().join(scope).join(name)).unwrap();
        }

        // The walk never sees a link, so the build is clean...
        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        for scope in ["notes", "old"] {
            // ...and the on-disk read agrees: no schema file, so no error to leak from.
            let (file, name) = cache
                .raw_file_at(Path::new(scope))
                .unwrap_or_else(|e| panic!("{scope}: a symlinked schema file is absent: {e}"));
            assert_eq!(name, None, "{scope}");
            assert!(file.fields.is_empty(), "{scope}");

            let sub = Path::new(scope).join("sub");
            let from_disk = cache
                .resolve_from_disk(&sub, None)
                .unwrap_or_else(|e| panic!("{scope}: the resolver reads it as absent: {e}"));
            assert_eq!(
                &from_disk,
                cache.resolve_for(&sub.join("doc.md")),
                "{scope}"
            );
        }
    }

    #[test]
    fn a_symlinked_or_non_regular_schema_name_does_not_count_toward_both_names() {
        let dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let other = outside.path().join("other.yaml");
        fs::write(&other, "fields:\n  other:\n    required: true\n").unwrap();
        let regular = "fields:\n  a:\n    required: true\n";

        // A regular canonical file beside a link at the legacy name...
        write_schema(dir.path(), "canonical", regular);
        let link = dir.path().join("canonical").join(LEGACY_SCHEMA_FILE_NAME);
        std::os::unix::fs::symlink(&other, link).unwrap();
        // ...a regular legacy file beside a link at the canonical name...
        fs::create_dir_all(dir.path().join("legacy")).unwrap();
        fs::write(
            dir.path().join("legacy").join(LEGACY_SCHEMA_FILE_NAME),
            regular,
        )
        .unwrap();
        let link = dir.path().join("legacy").join(SCHEMA_FILE_NAME);
        std::os::unix::fs::symlink(&other, link).unwrap();
        // ...and a directory squatting on the canonical name.
        fs::create_dir_all(dir.path().join("squat").join(SCHEMA_FILE_NAME)).unwrap();

        // The walk finds exactly one file in the first two directories and none in
        // the third, so `raw_file_at` must too.
        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        for (scope, expected) in [
            ("canonical", Some(SCHEMA_FILE_NAME)),
            ("legacy", Some(LEGACY_SCHEMA_FILE_NAME)),
            ("squat", None),
        ] {
            let (file, name) = cache
                .raw_file_at(Path::new(scope))
                .unwrap_or_else(|e| panic!("{scope}: {e}"));
            assert_eq!(name, expected, "{scope}");
            assert_eq!(file.fields.contains_key("a"), expected.is_some(), "{scope}");
            assert!(
                !file.fields.contains_key("other"),
                "{scope}: link target read"
            );
        }
    }

    #[test]
    fn an_oversized_schema_file_is_refused_on_its_size_before_it_is_read() {
        let dir = TempDir::new().unwrap();
        let cache = cache_rooted_at(dir.path());
        let limit = MAX_SCHEMA_FILE_BYTES as usize;
        // Valid YAML padded with a comment to exactly the limit.
        let head = "fields:\n  a:\n    required: true\n";
        let at_limit = format!("{head}{}", "#".repeat(limit - head.len()));
        assert_eq!(at_limit.len(), limit);

        for (i, name) in SCHEMA_FILE_NAMES.into_iter().enumerate() {
            let scope = format!("big{i}");
            let target = dir.path().join(&scope);
            fs::create_dir_all(&target).unwrap();

            // Not UTF-8, so reading it first would fail with a different message:
            // getting the size reason proves the size was checked before the read.
            fs::write(target.join(name), vec![0xFF; limit + 1]).unwrap();
            let err = cache.raw_file_at(Path::new(&scope)).unwrap_err();
            assert_eq!(
                err,
                over_size_limit_reason(MAX_SCHEMA_FILE_BYTES + 1),
                "{name}"
            );
            assert!(err.contains("byte limit"), "{err}");

            // The limit itself is still readable.
            fs::write(target.join(name), &at_limit).unwrap();
            let (file, found) = cache.raw_file_at(Path::new(&scope)).unwrap();
            assert_eq!(found, Some(name));
            assert_eq!(file.fields["a"].required, Some(true));
        }
    }

    fn inherited_tags(values: &[&str]) -> FieldDef {
        FieldDef {
            ty: Some(FieldType::List),
            required: false,
            indexed: false,
            values: Some(values.iter().map(|v| v.to_string()).collect()),
            default: None,
            open: true,
            ..FieldDef::default()
        }
    }

    #[test]
    fn add_values_in_a_child_scope_extends_the_inherited_set() {
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "",
            "fields:\n  tags:\n    type: list\n    values: [a, b]\n",
        );
        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let inherited = cache.resolve_for(Path::new("_")).fields["tags"].clone();

        let mut child = SchemaFile::default();
        child
            .apply_inheriting(
                &SchemaEdit::AddValues {
                    field: "tags".into(),
                    values: vec!["c".into()],
                },
                Some(&inherited),
            )
            .unwrap();
        let tags = &child.fields["tags"];
        assert_eq!(
            tags.values.as_deref(),
            Some(&[VALUES_SENTINEL.to_string(), "c".to_string()][..])
        );
        assert_eq!(tags.ty, None, "the inherited type is kept");

        write_schema(dir.path(), "child", &child.to_yaml().unwrap());
        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let resolved = &cache.resolve_for(Path::new("child/doc.md")).fields["tags"];
        assert_eq!(
            resolved.values.as_deref(),
            Some(&["a".to_string(), "b".to_string(), "c".to_string()][..])
        );
        assert_eq!(resolved.ty, Some(FieldType::List));
    }

    #[test]
    fn add_values_keeps_the_sentinel_leading_and_a_local_list_without_it_as_is() {
        let inherited = inherited_tags(&["a"]);

        // A second add keeps `$values` first; only what follows it is sorted.
        let mut file = SchemaFile::default();
        for value in ["z", "m"] {
            file.apply_inheriting(
                &SchemaEdit::AddValues {
                    field: "tags".into(),
                    values: vec![value.into()],
                },
                Some(&inherited),
            )
            .unwrap();
        }
        assert_eq!(
            file.fields["tags"].values.as_deref(),
            Some(
                &[
                    VALUES_SENTINEL.to_string(),
                    "m".to_string(),
                    "z".to_string()
                ][..]
            )
        );

        // A local list that deliberately replaces the inherited set stays a
        // replacement: the new value is appended, no sentinel added.
        let mut file: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  tags:\n    values: [x]\n").unwrap();
        file.apply_inheriting(
            &SchemaEdit::AddValues {
                field: "tags".into(),
                values: vec!["y".into()],
            },
            Some(&inherited),
        )
        .unwrap();
        assert_eq!(
            file.fields["tags"].values.as_deref(),
            Some(&["x".to_string(), "y".to_string()][..])
        );

        // Nothing inherited: a fresh `enum` with just the new values.
        let mut file = SchemaFile::default();
        file.apply_inheriting(
            &SchemaEdit::AddValues {
                field: "tags".into(),
                values: vec!["y".into()],
            },
            None,
        )
        .unwrap();
        assert_eq!(file.fields["tags"].ty, Some(FieldType::Enum));
        assert_eq!(
            file.fields["tags"].values.as_deref(),
            Some(&["y".to_string()][..])
        );
    }

    #[test]
    fn add_values_skips_what_the_scope_already_inherits() {
        let inherited = inherited_tags(&["a", "b"]);
        let add = |values: &[&str]| SchemaEdit::AddValues {
            field: "tags".into(),
            values: values.iter().map(|v| v.to_string()).collect(),
        };

        // Nothing declared here and everything inherited: an error, and no
        // declaration is created for the caller to write out as an empty file.
        let mut file = SchemaFile::default();
        let err = file
            .apply_inheriting(&add(&["a", "b"]), Some(&inherited))
            .unwrap_err();
        assert!(err.contains("already permits"), "{err}");
        assert!(file.fields.is_empty());

        // A mix: only the genuinely new value is written, behind the sentinel.
        file.apply_inheriting(&add(&["a", "c"]), Some(&inherited))
            .unwrap();
        assert_eq!(
            file.fields["tags"].values.as_deref(),
            Some(&[VALUES_SENTINEL.to_string(), "c".to_string()][..])
        );

        // A scope that declares the field without a list inherits the set
        // verbatim, so an inherited value changes nothing and leaves it unlisted.
        let mut file: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  tags:\n    required: true\n").unwrap();
        let summary = file
            .apply_inheriting(&add(&["a"]), Some(&inherited))
            .unwrap();
        assert!(summary.contains("already permitted"), "{summary}");
        assert_eq!(file.fields["tags"].values, None);

        // A local list that replaces the inherited set (no sentinel) does not
        // inherit, so an inherited value is a real addition there.
        let mut file: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  tags:\n    values: [x]\n").unwrap();
        file.apply_inheriting(&add(&["a"]), Some(&inherited))
            .unwrap();
        assert_eq!(
            file.fields["tags"].values.as_deref(),
            Some(&["a".to_string(), "x".to_string()][..])
        );
    }

    #[test]
    fn add_values_declares_a_field_no_ancestor_typed_as_a_strict_enum() {
        let dir = TempDir::new().unwrap();
        // The ancestor declares the field but gives it neither a type nor values.
        write_schema(dir.path(), "", "fields:\n  status:\n    required: true\n");
        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let inherited = cache.resolve_for(Path::new("_")).fields["status"].clone();
        assert_eq!((inherited.ty, inherited.values.as_ref()), (None, None));

        let mut child = SchemaFile::default();
        child
            .apply_inheriting(
                &SchemaEdit::AddValues {
                    field: "status".into(),
                    values: vec!["a".into(), "b".into()],
                },
                Some(&inherited),
            )
            .unwrap();
        assert_eq!(child.fields["status"].ty, Some(FieldType::Enum));
        assert_eq!(
            child.fields["status"].values.as_deref(),
            Some(&["a".to_string(), "b".to_string()][..]),
            "nothing inherited to splice in"
        );

        write_schema(dir.path(), "child", &child.to_yaml().unwrap());
        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let resolved = cache.resolve_for(Path::new("child/doc.md"));
        assert_eq!(resolved.fields["status"].ty, Some(FieldType::Enum));
        assert!(
            resolved.fields["status"].required,
            "the ancestor's `required` is kept"
        );

        // Strict enforcement: a non-string scalar outside the set is refused, not
        // exempted the way a typeless values list would exempt it.
        let rules = |value: Value| {
            let frontmatter = HashMap::from([("status".to_string(), value)]);
            crate::validate::validate_frontmatter(&frontmatter, resolved)
                .into_iter()
                .map(|e| e.rule)
                .collect::<Vec<_>>()
        };
        assert!(rules(json!("a")).is_empty());
        for refused in [json!(3), json!(true), json!("c")] {
            assert_eq!(rules(refused.clone()), ["allowed_value"], "{refused}");
        }
    }

    #[test]
    fn add_values_keeps_a_typeless_values_list_lenient_when_an_ancestor_supplies_one() {
        // An ancestor's values list with no type (how a legacy `config.yaml` `allowed`
        // entry arrives) is a deliberate lenient regime; a child extending it keeps it
        // rather than tightening the field on its own.
        let inherited = FieldDef {
            ty: None,
            required: false,
            indexed: false,
            values: Some(vec!["x".into()]),
            default: None,
            open: true,
            ..FieldDef::default()
        };
        let mut file = SchemaFile::default();
        file.apply_inheriting(
            &SchemaEdit::AddValues {
                field: "status".into(),
                values: vec!["y".into()],
            },
            Some(&inherited),
        )
        .unwrap();
        assert_eq!(file.fields["status"].ty, None);
    }

    #[test]
    fn unknown_schema_keys_are_rejected() {
        let parsed: Result<SchemaFile, _> =
            serde_yaml_ng::from_str("fields:\n  title:\n    requried: true\n");
        assert!(parsed.is_err(), "a typo'd key must not be silently ignored");
    }

    // -- dedup override (#272) ----------------------------------------------

    #[test]
    fn dedup_cascades_per_key_nearest_wins_without_sibling_leak() {
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "food",
            "dedup:\n  enabled: false\n  threshold: 0.8\n",
        );
        write_schema(dir.path(), "food/recipes", "dedup:\n  threshold: 0.97\n");
        write_schema(
            dir.path(),
            "food/plans",
            "fields:\n  title:\n    required: true\n",
        );
        write_schema(dir.path(), "dev", "fields:\n  title:\n    required: true\n");

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());

        let food = cache.resolve_for(Path::new("food/x.md"));
        assert_eq!(food.dedup_enabled, Some(false));
        assert_eq!(food.dedup_threshold, Some(0.8));

        let recipes = cache.resolve_for(Path::new("food/recipes/x.md"));
        assert_eq!(recipes.dedup_enabled, Some(false), "inherited from parent");
        assert_eq!(
            recipes.dedup_threshold,
            Some(0.97),
            "child overrides one key"
        );

        let plans = cache.resolve_for(Path::new("food/plans/x.md"));
        assert_eq!(plans.dedup_enabled, Some(false), "no dedup block inherits");
        assert_eq!(plans.dedup_threshold, Some(0.8));

        let dev = cache.resolve_for(Path::new("dev/x.md"));
        assert_eq!(dev.dedup_enabled, None, "sibling scope does not leak");
        assert_eq!(dev.dedup_threshold, None);
    }

    #[test]
    fn root_schema_file_dedup_applies_everywhere() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "dedup:\n  threshold: 0.99\n");

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());

        let resolved = cache.resolve_for(Path::new("any/deep/x.md"));
        assert_eq!(resolved.dedup_threshold, Some(0.99));
        assert_eq!(resolved.dedup_enabled, None);
    }

    #[test]
    fn dedup_typos_and_out_of_range_thresholds_are_rejected() {
        assert!(serde_yaml_ng::from_str::<SchemaFile>("dedup:\n  treshold: 0.9\n").is_err());
        assert!(serde_yaml_ng::from_str::<SchemaFile>("dedup:\n  enabled: maybe\n").is_err());

        for bad in ["1.5", "-0.1"] {
            let file: SchemaFile =
                serde_yaml_ng::from_str(&format!("dedup:\n  threshold: {bad}\n")).unwrap();
            let err = file.validate_self().unwrap_err();
            assert!(err.contains("dedup.threshold"), "got: {err}");
        }
        let ok: SchemaFile = serde_yaml_ng::from_str("dedup:\n  threshold: 1.0\n").unwrap();
        ok.validate_self().unwrap();
    }

    #[test]
    fn related_keys_cascade_per_key_and_are_range_checked() {
        let dir = tempfile::tempdir().unwrap();
        write_schema(
            dir.path(),
            "food",
            "dedup:\n  related_limit: 3\n  related_min_score: 0.6\n",
        );
        write_schema(dir.path(), "food/recipes", "dedup:\n  related_limit: 0\n");
        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let food = cache.resolve_for(Path::new("food/x.md"));
        assert_eq!(food.related_limit, Some(3));
        assert_eq!(food.related_min_score, Some(0.6));
        let recipes = cache.resolve_for(Path::new("food/recipes/x.md"));
        assert_eq!(recipes.related_limit, Some(0));
        assert_eq!(recipes.related_min_score, Some(0.6), "inherited");
        let dev = cache.resolve_for(Path::new("dev/x.md"));
        assert_eq!(dev.related_limit, None);

        for (yaml, key) in [
            ("dedup:\n  related_min_score: 1.5\n", "related_min_score"),
            ("dedup:\n  related_limit: 21\n", "related_limit"),
        ] {
            let file: SchemaFile = serde_yaml_ng::from_str(yaml).unwrap();
            let err = file.validate_self().unwrap_err();
            assert!(err.contains(key), "got: {err}");
        }
        assert!(serde_yaml_ng::from_str::<SchemaFile>("dedup:\n  related_limit: -1\n").is_err());
    }

    #[test]
    fn a_bad_dedup_block_fails_the_build_like_any_schema_error() {
        for bad in [
            "dedup:\n  threshold: 2.0\n",
            "dedup:\n  treshold: 0.9\n",
            "dedup:\n  enabled: maybe\n",
            "dedup: [0.9]\n",
        ] {
            let dir = TempDir::new().unwrap();
            write_schema(dir.path(), "bad", bad);

            let err =
                SchemaCache::build(dir.path(), &empty_config(), &Default::default()).unwrap_err();

            assert_eq!(err.invalid.len(), 1, "{bad:?}: {err}");
            assert_eq!(err.invalid[0].path, Path::new("bad/.schema.yaml"));
        }
    }

    #[test]
    fn dedup_block_survives_an_edit_and_yaml_round_trip() {
        let mut file: SchemaFile = serde_yaml_ng::from_str(
            "dedup:\n  enabled: false\n  threshold: 0.9\nfields:\n  tags:\n    values: [a]\n",
        )
        .unwrap();
        file.apply(&SchemaEdit::AddValues {
            field: "tags".into(),
            values: vec!["b".into()],
        })
        .unwrap();

        let reparsed: SchemaFile = serde_yaml_ng::from_str(&file.to_yaml().unwrap()).unwrap();
        let dedup = reparsed.dedup.expect("dedup block preserved");
        assert_eq!(dedup.enabled, Some(false));
        assert_eq!(dedup.threshold, Some(0.9));

        let plain: SchemaFile = serde_yaml_ng::from_str("fields: {}\n").unwrap();
        assert!(
            !plain.to_yaml().unwrap().contains("dedup"),
            "an absent block is not serialized"
        );
    }

    #[test]
    fn fingerprint_ignores_dedup_keys() {
        let base = ResolvedSchema::default();
        let plain: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  t:\n    required: true\n").unwrap();
        let with_dedup: SchemaFile = serde_yaml_ng::from_str(
            "dedup:\n  enabled: false\n  threshold: 0.9\nfields:\n  t:\n    required: true\n",
        )
        .unwrap();

        let a = base.merged_with_for_test(&plain, "x");
        let b = base.merged_with_for_test(&with_dedup, "x");
        assert_ne!(a, b);
        assert_eq!(a.fingerprint(), b.fingerprint());
    }

    // -- corpus policy: derived fields and the root-edit guard (#345) ---------

    fn root_resolved(yaml: &str) -> ResolvedSchema {
        let file: SchemaFile = serde_yaml_ng::from_str(yaml).unwrap();
        file.validate_self().unwrap();
        ResolvedSchema::default().merged_with_for_test(&file, "/")
    }

    #[test]
    fn fingerprint_changes_with_derived_but_not_with_root_edit() {
        let fields = "fields:\n  t:\n    required: true\n";
        let plain = root_resolved(fields);
        let derived = root_resolved(&format!(
            "{fields}derived:\n  domain: {{from: path_segment, index: 0}}\n"
        ));
        let other_index = root_resolved(&format!(
            "{fields}derived:\n  domain: {{from: path_segment, index: 1}}\n"
        ));
        let other_name = root_resolved(&format!(
            "{fields}derived:\n  area: {{from: path_segment, index: 0}}\n"
        ));
        let guarded = root_resolved(&format!(
            "{fields}root_edit:\n  require_ack: true\n  policy_doc: meta/p.md\n"
        ));
        let empty_derived = root_resolved(&format!("{fields}derived: {{}}\n"));

        assert_ne!(plain.fingerprint(), derived.fingerprint());
        assert_ne!(derived.fingerprint(), other_index.fingerprint());
        assert_ne!(derived.fingerprint(), other_name.fingerprint());
        assert_ne!(plain, guarded);
        assert_eq!(plain.fingerprint(), guarded.fingerprint());
        assert_eq!(
            plain.fingerprint(),
            empty_derived.fingerprint(),
            "an empty derived block hashes like an absent one"
        );
    }

    #[test]
    fn root_only_blocks_are_inherited_by_every_scope() {
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "",
            "derived:\n  domain: {from: path_segment, index: 0}\n\
             root_edit:\n  require_ack: true\n",
        );
        write_schema(dir.path(), "food", "fields:\n  tags:\n    type: list\n");
        // A non-root block has no effect: food/recipes still sees the root's.
        write_schema(
            dir.path(),
            "food/recipes",
            "derived:\n  dish: {from: path_segment, index: 2}\nroot_edit:\n  require_ack: false\n",
        );
        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());

        for doc in ["a.md", "food/a.md", "food/recipes/a.md", "other/a.md"] {
            let schema = cache.resolve_for(Path::new(doc));
            assert_eq!(
                schema.derived_field_names().collect::<Vec<_>>(),
                vec!["domain"],
                "{doc}"
            );
            assert!(
                schema.root_edit.as_ref().is_some_and(|p| p.require_ack),
                "{doc}"
            );
        }
        assert_eq!(
            cache.resolve_for(Path::new("food/a.md")).fingerprint(),
            cache
                .resolve_for(Path::new("food/recipes/a.md"))
                .fingerprint(),
            "an ignored non-root derived block does not change the fingerprint"
        );

        let findings = cache.lint_findings();
        let in_recipes: Vec<&String> = findings
            .iter()
            .filter(|(dir, _)| dir == Path::new("food/recipes"))
            .map(|(_, w)| w)
            .collect();
        assert_eq!(in_recipes.len(), 2, "{findings:?}");
        assert!(in_recipes.iter().any(|w| w.contains("'derived'")));
        assert!(in_recipes.iter().any(|w| w.contains("'root_edit'")));
        assert!(
            findings.iter().all(|(dir, _)| dir != Path::new("")),
            "the root's own blocks are not linted: {findings:?}"
        );
    }

    #[test]
    fn no_root_schema_file_means_nothing_is_derived() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "food", "fields:\n  tags:\n    type: list\n");
        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        assert!(cache.root().derived.is_empty());
        assert!(cache.resolve_for(Path::new("food/a.md")).derived.is_empty());
        assert!(cache.root().root_edit.is_none());
    }

    #[test]
    fn resolve_from_disk_and_candidates_carry_the_root_blocks() {
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "",
            "derived:\n  domain: {from: path_segment, index: 0}\n",
        );
        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let from_disk = cache.resolve_from_disk(Path::new("food"), None).unwrap();
        assert!(from_disk.is_derived("domain"));
        let candidate: SchemaFile = serde_yaml_ng::from_str("fields: {}\n").unwrap();
        let resolved = cache
            .resolve_with_candidate(Path::new("food/a.md"), Path::new("food"), &candidate)
            .unwrap();
        assert!(resolved.is_derived("domain"));
    }

    #[test]
    fn root_only_blocks_round_trip_through_to_yaml() {
        let yaml = "derived:\n  domain: {from: path_segment, index: 0}\n  campaign: {from: path_segment, index: 1}\n\
                    root_edit:\n  require_ack: true\n  policy_doc: meta/p.md\n\
                    fields:\n  t:\n    required: true\n";
        let file = parse_schema_text(yaml).unwrap();
        let again = parse_schema_text(&file.to_yaml().unwrap()).unwrap();
        assert_eq!(again.derived, file.derived);
        assert_eq!(again.root_edit, file.root_edit);
        assert_eq!(
            again.derived.as_ref().unwrap()["campaign"],
            DerivedField::path_segment(1)
        );

        let none = parse_schema_text("fields: {}\n")
            .unwrap()
            .to_yaml()
            .unwrap();
        assert!(
            !none.contains("derived") && !none.contains("root_edit"),
            "{none}"
        );
    }

    #[test]
    fn a_null_root_only_block_reads_as_absent() {
        for yaml in [
            "derived:\nroot_edit:\n",
            "derived: ~\nroot_edit: ~\n",
            "derived: null\nroot_edit: null\nfields: {}\n",
        ] {
            let file = parse_schema_text(yaml).unwrap_or_else(|e| panic!("{yaml:?}: {e}"));
            assert!(file.derived.is_none(), "{yaml:?}");
            assert!(file.root_edit.is_none(), "{yaml:?}");
        }
    }

    #[test]
    fn malformed_root_only_blocks_are_schema_errors() {
        for (yaml, needle) in [
            (
                "derived:\n  domain: {from: file_name, index: 0}\n",
                "file_name",
            ),
            ("derived:\n  domain: {from: path_segment}\n", "index"),
            ("derived:\n  domain: {from: path_segment, index: -1}\n", ""),
            (
                "derived:\n  domain: {from: path_segment, index: 33}\n",
                "at most 32",
            ),
            (
                "derived:\n  domain: {from: path_segment, index: 0, x: 1}\n",
                "x",
            ),
            (
                "derived:\n  a.b: {from: path_segment, index: 0}\n",
                "top-level field name",
            ),
            (
                "derived:\n  '': {from: path_segment, index: 0}\n",
                "top-level field name",
            ),
            (
                "derived:\n  file_path: {from: path_segment, index: 0}\n",
                "reserved",
            ),
            (
                "derived:\n  title: {from: path_segment, index: 0}\n",
                "reserved",
            ),
            (
                "derived:\n  d: {from: path_segment, index: 0}\n  d: {from: path_segment, index: 1}\n",
                "duplicate",
            ),
            ("root_edit:\n  policy_doc: meta/p.md\n", "require_ack"),
            ("root_edit:\n  require_ack: true\n  extra: 1\n", "extra"),
            (
                "root_edit:\n  require_ack: true\n  policy_doc: ''\n",
                "policy_doc",
            ),
            (
                "root_edit:\n  require_ack: true\n  policy_doc: \"a\\nb\"\n",
                "policy_doc",
            ),
        ] {
            let err = parse_schema_text(yaml).expect_err(yaml);
            assert!(err.contains(needle), "{yaml}: {err}");
        }
        let long = format!(
            "derived:\n  {}: {{from: path_segment, index: 0}}\n",
            "f".repeat(MAX_DERIVED_FIELD_NAME_LEN + 1)
        );
        assert!(parse_schema_text(&long).is_err());
        let many: String = (0..=MAX_DERIVED_FIELDS)
            .map(|i| format!("  f{i}: {{from: path_segment, index: 0}}\n"))
            .collect();
        assert!(
            parse_schema_text(&format!("derived:\n{many}"))
                .unwrap_err()
                .contains("at most")
        );
    }

    // -- indexed field union ------------------------------------------------

    #[test]
    fn indexed_fields_union_across_every_scope() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "", "fields:\n  tags:\n    indexed: true\n");
        write_schema(
            dir.path(),
            "kitchen/recipes",
            "fields:\n  planning.prep_minutes:\n    type: integer\n    indexed: true\n",
        );

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let fields = cache.all_indexed_fields();
        let named = |name: &str| fields.iter().find(|f| f.name == name);

        assert!(named("tags").is_some());
        let nested = named("planning.prep_minutes")
            .expect("a field declared only in a deep scope still needs a payload index");
        assert_eq!(
            nested.kind,
            IndexKind::Integer,
            "an integer field needs an integer index for range filters to work"
        );
    }

    // -- candidate resolution for the schema dry-run -------------------------

    #[test]
    fn candidate_reaches_documents_a_deeper_scope_does_not_override() {
        // Merge is per FIELD: recipes/ declares only cook_minutes, so a new root field
        // still reaches documents under it. Treating "a deeper scope exists" as a
        // shadow would silently report this change as safe.
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "recipes",
            "fields:\n  cook_minutes:\n    type: integer\n",
        );
        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());

        let candidate: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  format:\n    required: true\n").unwrap();
        let effective = cache
            .resolve_with_candidate(Path::new("recipes/chili.md"), Path::new(""), &candidate)
            .expect("the edit reaches this document");

        assert!(
            effective.fields["format"].required,
            "a field the deeper scope never touches must still apply"
        );
        assert_eq!(
            effective.fields["cook_minutes"].ty,
            Some(FieldType::Integer),
            "the deeper scope's own declarations survive"
        );
    }

    #[test]
    fn a_deeper_scope_still_wins_for_the_attribute_it_redeclares_but_inherits_the_rest() {
        // `archive/` only ever redeclares `values` (no sentinel — a plain replace) and
        // never mentions `required`. Under per-attribute inheritance that means
        // `values` still wins locally, but `required` — set by this candidate edit at
        // the root — reaches `archive/` anyway, since nothing there overrides it. This
        // is the deliberate reversal from wholesale replacement: a descendant no
        // longer needs to repeat every attribute it doesn't want to lose.
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "archive",
            "fields:\n  status:\n    type: enum\n    values: [archived]\n",
        );
        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());

        let candidate: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  status:\n    type: enum\n    values: [active]\n    required: true\n",
        )
        .unwrap();
        let effective = cache
            .resolve_with_candidate(Path::new("archive/old.md"), Path::new(""), &candidate)
            .expect("still inside the edited subtree");

        assert_eq!(
            effective.fields["status"].values,
            Some(vec!["archived".into()]),
            "archive/'s own values (no sentinel) still replace the root's outright"
        );
        assert!(
            effective.fields["status"].required,
            "required was never redeclared by archive/, so the root edit's `required: \
             true` reaches it — the opposite of the old wholesale-replace behavior"
        );
    }

    #[test]
    fn documents_outside_the_edited_subtree_are_unaffected() {
        let dir = TempDir::new().unwrap();
        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let candidate: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  x:\n    required: true\n").unwrap();

        assert!(
            cache
                .resolve_with_candidate(
                    Path::new("sysadmin/note.md"),
                    Path::new("food"),
                    &candidate
                )
                .is_none()
        );
    }

    #[test]
    fn a_candidate_for_a_directory_with_no_schema_yet_still_applies() {
        let dir = TempDir::new().unwrap();
        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let candidate: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  x:\n    required: true\n").unwrap();

        let effective = cache
            .resolve_with_candidate(Path::new("food/a.md"), Path::new("food"), &candidate)
            .expect("creating a scope must still evaluate against its documents");
        assert!(effective.fields["x"].required);
    }

    #[test]
    fn index_kind_follows_the_declared_type() {
        assert_eq!(
            ResolvedSchema::index_kind(Some(FieldType::Integer)),
            IndexKind::Integer
        );
        assert_eq!(
            ResolvedSchema::index_kind(Some(FieldType::Number)),
            IndexKind::Float
        );
        assert_eq!(
            ResolvedSchema::index_kind(Some(FieldType::Boolean)),
            IndexKind::Bool
        );
        assert_eq!(
            ResolvedSchema::index_kind(Some(FieldType::Enum)),
            IndexKind::Keyword
        );
        assert_eq!(
            ResolvedSchema::index_kind(None),
            IndexKind::Keyword,
            "undeclared fields fall back to keyword"
        );
    }

    #[test]
    fn conflicting_declared_types_take_the_first_and_warn() {
        let dir = TempDir::new().unwrap();
        write_schema(
            dir.path(),
            "a",
            "fields:\n  size:\n    type: integer\n    indexed: true\n",
        );
        write_schema(
            dir.path(),
            "b",
            "fields:\n  size:\n    type: text\n    indexed: true\n",
        );

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let fields = cache.all_indexed_fields();

        // One collection cannot hold two index kinds for one payload path.
        assert_eq!(fields.iter().filter(|f| f.name == "size").count(), 1);
    }

    // -- fingerprint --------------------------------------------------------

    #[test]
    fn fingerprint_is_stable_and_order_independent() {
        let a: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  b:\n    values: [x, y]\n  a:\n    required: true\n",
        )
        .unwrap();
        let b: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  a:\n    required: true\n  b:\n    values: [y, x]\n",
        )
        .unwrap();

        let base = ResolvedSchema::default();
        let first = base.merged_with(&a, "s").fingerprint();
        for _ in 0..8 {
            assert_eq!(base.merged_with(&a, "s").fingerprint(), first);
        }
        assert_eq!(
            base.merged_with(&b, "s").fingerprint(),
            first,
            "declaration order must not change the fingerprint"
        );
    }

    #[test]
    fn fingerprint_changes_when_a_rule_tightens() {
        let loose: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  title:\n    required: false\n").unwrap();
        let tight: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  title:\n    required: true\n").unwrap();

        let base = ResolvedSchema::default();
        assert_ne!(
            base.merged_with(&loose, "s").fingerprint(),
            base.merged_with(&tight, "s").fingerprint()
        );
    }

    #[test]
    fn fingerprint_distinguishes_absent_values_from_empty_values() {
        // `values` absent entirely (None: unconstrained) vs. `values: []`
        // (Some(vec![]): nothing permitted) enforce opposite rules but, before
        // #152's fix, hashed identically because the values loop contributes
        // zero bytes when the list is empty OR absent.
        let unconstrained: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  status:\n    type: text\n").unwrap();
        let closed: SchemaFile =
            serde_yaml_ng::from_str("fields:\n  status:\n    type: text\n    values: []\n")
                .unwrap();

        let base = ResolvedSchema::default();
        let unconstrained_resolved = base.merged_with(&unconstrained, "s");
        let closed_resolved = base.merged_with(&closed, "s");

        // Sanity check that the YAML actually produced the two states under test.
        assert_eq!(
            unconstrained_resolved.fields.get("status").unwrap().values,
            None
        );
        assert_eq!(
            closed_resolved.fields.get("status").unwrap().values,
            Some(Vec::new())
        );

        assert_ne!(
            unconstrained_resolved.fingerprint(),
            closed_resolved.fingerprint(),
            "None and Some(vec![]) values must produce different fingerprints"
        );
    }

    /// #351: the payload fingerprint moves with exactly what reaches the Qdrant payload
    /// frontmatter — defaults and derived rules — and not with validation-only edits,
    /// all of which still move `fingerprint`.
    #[test]
    fn payload_fingerprint_tracks_defaults_and_derived_only() {
        let resolve = |yaml: &str| {
            let file: SchemaFile = serde_yaml_ng::from_str(yaml).unwrap();
            ResolvedSchema::default().merged_with(&file, &scope_label(Path::new("")))
        };
        let base = resolve("fields:\n  status:\n    values: [draft, done]\n    default: draft\n");

        for (what, yaml) in [
            (
                "required",
                "fields:\n  status:\n    values: [draft, done]\n    default: draft\n    required: true\n",
            ),
            (
                "values",
                "fields:\n  status:\n    values: [draft]\n    default: draft\n",
            ),
            (
                "indexed",
                "fields:\n  status:\n    values: [draft, done]\n    default: draft\n    indexed: true\n",
            ),
            (
                "a field with no default",
                "fields:\n  status:\n    values: [draft, done]\n    default: draft\n  owner:\n    type: text\n",
            ),
        ] {
            let edited = resolve(yaml);
            assert_ne!(edited.fingerprint(), base.fingerprint(), "{what}");
            assert_eq!(
                edited.payload_fingerprint(),
                base.payload_fingerprint(),
                "{what} must not move the payload fingerprint"
            );
        }

        for (what, yaml) in [
            (
                "a changed default",
                "fields:\n  status:\n    values: [draft, done]\n    default: done\n",
            ),
            (
                "a removed default",
                "fields:\n  status:\n    values: [draft, done]\n",
            ),
            (
                "a nested default",
                "fields:\n  status:\n    values: [draft, done]\n    default: draft\n  \
                 planning:\n    fields:\n      effort:\n        default: low\n",
            ),
            (
                "a derived rule",
                "fields:\n  status:\n    values: [draft, done]\n    default: draft\n\
                 derived:\n  domain: {from: path_segment, index: 0}\n",
            ),
        ] {
            let edited = resolve(yaml);
            assert_ne!(edited.fingerprint(), base.fingerprint(), "{what}");
            assert_ne!(
                edited.payload_fingerprint(),
                base.payload_fingerprint(),
                "{what} must move the payload fingerprint"
            );
        }

        // A default on a derived field is never applied, so it is not payload input.
        let derived = resolve("derived:\n  domain: {from: path_segment, index: 0}\n");
        let derived_with_default = resolve(
            "fields:\n  domain:\n    default: misc\n\
             derived:\n  domain: {from: path_segment, index: 0}\n",
        );
        assert_eq!(
            derived.payload_fingerprint(),
            derived_with_default.payload_fingerprint()
        );
    }

    #[test]
    fn fingerprint_matches_the_pinned_golden() {
        // Pins the exact fingerprint bytes of a fixture covering every input the hash
        // reads. It guards the schema model refactor (#346) against fingerprint drift:
        // a changed fingerprint makes every governed document re-validate (and the
        // corpus re-embed), so a deliberate change must update this golden consciously.
        //
        // Coverage: a flat scalar; a container with children (hashed as its own entry
        // with type `Some(Object)`) and `open: false`; `values: []` vs absent values;
        // a `default`; a typeless values-only field; a `$values` splice inherited from
        // a parent scope; and the siblings `a`, `a-b`, `a.b`, which sort differently
        // under a depth-first walk than in flat dot-path order ('-' < '.').
        let parent: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  \
               title:\n    type: text\n    required: true\n    indexed: true\n  \
               tags:\n    values: [b, a]\n  \
               a:\n    type: object\n    fields:\n      b:\n        type: text\n  \
               a-b:\n    type: text\n",
        )
        .unwrap();
        let child: SchemaFile = serde_yaml_ng::from_str(
            "fields:\n  \
               tags:\n    values: [$values, c]\n  \
               planning:\n    type: object\n    open: false\n    fields:\n      \
                 method:\n        type: enum\n        values: [braise, oven]\n      \
                 minutes:\n        type: integer\n        default: 45\n  \
               closed:\n    type: text\n    values: []\n  \
               open_text:\n    type: text\n  \
               status:\n    values: [draft, done]\n    default: draft\n",
        )
        .unwrap();

        let resolved = ResolvedSchema::default()
            .merged_with(&parent, "parent")
            .merged_with(&child, "child");
        assert_eq!(
            resolved.fields["tags"].values,
            Some(vec!["b".into(), "a".into(), "c".into()]),
            "the sentinel must have spliced the inherited values"
        );

        assert_eq!(
            resolved.fingerprint(),
            "40db66e6f16490c85dd9efaca0aa87bf821f7ad19a0d60bde3671c6c367e8f4e"
        );
    }

    // -- dot-path access ----------------------------------------------------

    #[test]
    fn dotpath_reads_nested_values() {
        let fm: HashMap<String, Value> = match json!({
            "planning": { "prep_minutes": 45, "nested": { "deep": true } },
            "title": "x"
        }) {
            Value::Object(map) => map.into_iter().collect(),
            _ => unreachable!(),
        };

        assert_eq!(
            get_by_dotpath(&fm, "planning.prep_minutes"),
            Some(&json!(45))
        );
        assert_eq!(
            get_by_dotpath(&fm, "planning.nested.deep"),
            Some(&json!(true))
        );
        assert_eq!(get_by_dotpath(&fm, "title"), Some(&json!("x")));
        assert_eq!(get_by_dotpath(&fm, "planning.missing"), None);
        assert_eq!(get_by_dotpath(&fm, "title.nope"), None);
    }

    #[test]
    fn dotpath_writes_create_intermediate_objects() {
        let mut fm: HashMap<String, Value> = HashMap::new();
        set_by_dotpath(&mut fm, "planning.effort", json!("medium"));
        set_by_dotpath(&mut fm, "status", json!("active"));

        assert_eq!(fm["planning"]["effort"], json!("medium"));
        assert_eq!(fm["status"], json!("active"));
    }

    // -- type checking ------------------------------------------------------

    #[test]
    fn declared_types_are_enforced_strictly() {
        assert!(check_type(FieldType::Integer, &json!(45)).is_ok());
        assert!(
            check_type(FieldType::Integer, &json!("45")).is_err(),
            "a quoted number is the exact mistake strict typing exists to catch"
        );
        assert!(check_type(FieldType::Boolean, &json!(true)).is_ok());
        assert!(check_type(FieldType::Boolean, &json!("true")).is_err());
        assert!(check_type(FieldType::Number, &json!(1.5)).is_ok());
        assert!(check_type(FieldType::List, &json!(["a"])).is_ok());
        assert!(check_type(FieldType::List, &json!("a")).is_err());
        assert!(check_type(FieldType::Object, &json!({"a": 1})).is_ok());
    }

    #[test]
    fn an_integral_float_is_an_integer_and_a_fractional_one_is_not() {
        // YAML reads `3.0` as a float; it is still a whole number (#347).
        assert!(check_type(FieldType::Integer, &json!(3.0)).is_ok());
        assert!(check_type(FieldType::Integer, &json!(-12.0)).is_ok());
        let err = check_type(FieldType::Integer, &json!(3.5)).unwrap_err();
        assert_eq!(err, "expected an integer, got a decimal number");
    }

    // -- lint warnings (#347) -------------------------------------------------

    fn lint(yaml: &str) -> Vec<String> {
        let file: SchemaFile = serde_yaml_ng::from_str(yaml).unwrap();
        file.validate_self()
            .expect("lint findings are never load errors");
        lint_raw(&file, false)
    }

    #[test]
    fn lint_is_silent_for_a_clean_schema() {
        assert!(
            lint(
                "fields:
  status:
    type: enum
    values: [active, archived]
    default: active
  servings:
    type: integer
    default: 4.0
  tags:
    type: list
    values: [$values, x]
    default: [anything]
  planning:
    fields:
      effort:
        values: [low, high]
        default: low
  npcs:
    items:
      fields:
        name:
          type: text
          required: true
"
            )
            .is_empty()
        );
    }

    #[test]
    fn lint_flags_a_default_inside_list_items() {
        let warnings = lint(
            "fields:\n  npcs:\n    items:\n      fields:\n        name:\n          \
             required: true\n          default: anon\n  tags[]:\n    default: x\n",
        );
        assert_eq!(
            warnings,
            vec![
                "field 'npcs[].name': a default inside a list's items is never applied to \
                 its elements; every element must set this field itself"
                    .to_string(),
                "field 'tags[]': a default inside a list's items is never applied to its \
                 elements; every element must set this field itself"
                    .to_string(),
            ]
        );
    }

    #[test]
    fn lint_flags_a_default_that_fails_its_own_type() {
        let warnings = lint(
            "fields:\n  servings:\n    type: integer\n    default: four\n  \
             planning:\n    fields:\n      level:\n        type: integer\n        \
             default: 1.5\n",
        );
        assert_eq!(
            warnings,
            vec![
                "field 'planning.level': the default fails the field's own type (expected \
                 an integer, got a decimal number)"
                    .to_string(),
                "field 'servings': the default fails the field's own type (expected an \
                 integer, got a string)"
                    .to_string(),
            ]
        );
    }

    #[test]
    fn lint_flags_a_default_that_fails_its_own_values() {
        let warnings = lint(
            "fields:\n  status:\n    type: enum\n    values: [active]\n    default: draft\n  \
             legacy:\n    values: [a]\n    default: b\n",
        );
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert!(
            warnings[0].starts_with("field 'legacy': the default fails the field's own values")
        );
        assert!(
            warnings[1].starts_with("field 'status': the default fails the field's own values")
        );
        assert!(
            warnings[1].contains("'draft' is not permitted"),
            "{warnings:?}"
        );
    }

    #[test]
    fn lint_flags_values_on_object_and_boolean_fields() {
        let warnings = lint(
            "fields:\n  flag:\n    type: boolean\n    values: [\"true\"]\n  \
             box:\n    values: [a]\n    fields:\n      inner:\n        type: text\n",
        );
        assert_eq!(
            warnings,
            vec![
                "field 'box': 'values' does not belong on an object field (it constrains \
                 enum and list fields)"
                    .to_string(),
                "field 'flag': 'values' does not belong on a boolean field (it constrains \
                 enum and list fields)"
                    .to_string(),
            ]
        );
    }

    #[test]
    fn lint_flags_top_level_names_reserved_by_the_payload() {
        let warnings = lint(
            "fields:\n  text:\n    type: text\n  fm:\n    type: object\n  \
             mtime.sub:\n    type: text\n  meta:\n    fields:\n      text:\n        type: text\n",
        );
        let flagged: Vec<&str> = warnings
            .iter()
            .map(|w| w.split('\'').nth(1).unwrap())
            .collect();
        // Only top-level names: `meta.text` is not a payload key.
        assert_eq!(flagged, vec!["fm", "mtime.sub", "text"]);
        assert!(warnings[0].contains("reserved"), "{warnings:?}");
    }

    #[test]
    fn a_schema_with_lint_findings_still_builds() {
        let tmp = tempfile::tempdir().unwrap();
        write_schema(
            tmp.path(),
            "",
            "fields:\n  text:\n    type: integer\n    default: x\n",
        );
        let cache = SchemaCache::build(tmp.path(), &empty_config(), &Default::default())
            .expect("lint warnings never fail a build");
        assert!(cache.root().fields.contains_key("text"));
        assert_eq!(
            cache.lint_findings().len(),
            2,
            "{:?}",
            cache.lint_findings()
        );
    }

    #[test]
    fn a_rebuild_reports_only_lint_findings_it_introduced() {
        // `apply_rebuild` logs `new_lint_findings`, so a rebuild before every full
        // reconcile does not repeat warnings the installed cache already had.
        let tmp = tempfile::tempdir().unwrap();
        write_schema(tmp.path(), "", "fields:\n  fm:\n    type: text\n");
        let first = SchemaCache::build_for_test(tmp.path(), &empty_config());
        assert_eq!(first.lint_findings().len(), 1);

        let same = SchemaCache::build_for_test(tmp.path(), &empty_config());
        assert!(new_lint_findings(&first, &same).is_empty());

        write_schema(
            tmp.path(),
            "notes",
            "fields:\n  n:\n    type: integer\n    default: x\n",
        );
        let grown = SchemaCache::build_for_test(tmp.path(), &empty_config());
        let added = new_lint_findings(&first, &grown);
        assert_eq!(added.len(), 1, "{added:?}");
        assert_eq!(added[0].0, PathBuf::from("notes"));
        assert!(added[0].1.starts_with("field 'n':"), "{added:?}");
    }

    #[test]
    fn type_errors_name_both_expected_and_actual() {
        let err = check_type(FieldType::Integer, &json!("five")).unwrap_err();
        assert!(err.contains("an integer"), "got: {err}");
        assert!(err.contains("a string"), "got: {err}");
    }

    #[test]
    fn dates_and_timestamps_validate_their_formats() {
        assert!(check_type(FieldType::Date, &json!("2026-07-31")).is_ok());
        assert!(check_type(FieldType::Date, &json!("31/07/2026")).is_err());
        assert!(check_type(FieldType::Timestamp, &json!("2026-07-31T12:00:00Z")).is_ok());
        assert!(
            check_type(FieldType::Timestamp, &json!("2026-07-31")).is_err(),
            "a bare date is not a timestamp"
        );
    }

    #[test]
    fn enum_accepts_scalars_and_arrays_alike() {
        // Backward compatibility: the old `allowed` map checked both shapes.
        assert!(check_type(FieldType::Enum, &json!("guide")).is_ok());
        assert!(check_type(FieldType::Enum, &json!(["a", "b"])).is_ok());
        assert!(check_type(FieldType::Enum, &json!([{"nested": 1}])).is_err());
    }

    #[test]
    fn value_sets_check_arrays_element_wise() {
        let permitted = vec!["a".to_string(), "b".to_string()];
        assert!(check_values(&json!("a"), &permitted).is_ok());
        assert!(check_values(&json!(["a", "b"]), &permitted).is_ok());
        assert!(check_values(&json!(["a", "z"]), &permitted).is_err());
        assert!(check_values(&json!("z"), &permitted).is_err());
    }

    #[test]
    fn value_set_errors_list_what_is_permitted() {
        let err = check_values(&json!("z"), &["a".to_string(), "b".to_string()]).unwrap_err();
        assert!(err.contains("'z'"));
        assert!(err.contains("a, b"), "error should name the allowed set");
    }

    #[test]
    fn value_sets_compare_booleans_and_numbers_by_canonical_text() {
        assert!(check_values(&json!(true), &["true".to_string()]).is_ok());
        assert!(check_values(&json!(5), &["5".to_string()]).is_ok());
    }

    /// Pins the deliberate divergence documented on `check_values_lenient` and
    /// `RawFieldDef::values` (issue #77): a declared `type: enum` is strict about what
    /// counts as a value at all, while an undeclared-type field with the same
    /// `permitted` set waves non-string, non-array values through untouched. If either
    /// function is ever changed to converge with the other, this fails.
    #[test]
    fn strict_and_lenient_checks_diverge_on_non_string_values_by_design() {
        let permitted = vec!["true".to_string()];

        assert!(
            check_values(&json!(5), &permitted).is_err(),
            "strict check rejects a number not in the permitted (text-compared) set"
        );
        assert!(
            check_values_lenient(&json!(5), Some(&permitted)).is_ok(),
            "lenient check exempts numbers entirely, matching pre-cascade `allowed` semantics"
        );
    }

    // -- with_remapped_scopes ------------------------------------------------

    #[test]
    fn with_remapped_scopes_relocates_a_schema_files_governing_directory() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "old", "fields:\n  x:\n    required: true\n");

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let remapped = cache.with_remapped_scopes(|d| {
            if d == Path::new("old") {
                Some(PathBuf::from("new"))
            } else {
                None
            }
        });

        assert!(
            remapped
                .resolve_for(Path::new("new/doc.md"))
                .fields
                .contains_key("x"),
            "the relocated directory now governs the field"
        );
        assert!(
            !remapped
                .resolve_for(Path::new("old/doc.md"))
                .fields
                .contains_key("x"),
            "the old directory no longer does"
        );
        // The live cache is untouched — `with_remapped_scopes` returns a detached copy.
        assert!(
            cache
                .resolve_for(Path::new("old/doc.md"))
                .fields
                .contains_key("x")
        );
    }

    #[test]
    fn with_remapped_scopes_recomputes_descendants_of_a_relocated_scope() {
        // A schema file two levels deep under the relocated directory must still
        // see the relocated parent's rules — relocating a scope has to re-cascade
        // its whole subtree, not just patch the one entry that moved.
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "old", "fields:\n  x:\n    required: true\n");
        write_schema(
            dir.path(),
            "old/child",
            "fields:\n  y:\n    indexed: true\n",
        );
        write_schema(
            dir.path(),
            "dest",
            "fields:\n  z:\n    type: text\n    default: hi\n",
        );

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let remapped = cache.with_remapped_scopes(|d| {
            if d == Path::new("old") || d.starts_with("old") {
                let suffix = d.strip_prefix("old").unwrap();
                Some(Path::new("dest/moved").join(suffix))
            } else {
                None
            }
        });

        let resolved = remapped.resolve_for(Path::new("dest/moved/child/doc.md"));
        assert!(
            resolved.fields["x"].required,
            "inherited from the moved root"
        );
        assert!(resolved.fields["y"].indexed, "the child scope's own field");
        assert_eq!(
            resolved.fields["z"].default,
            Some(json!("hi")),
            "the moved subtree also picks up whatever already governed its new parent"
        );
    }

    #[test]
    fn with_remapped_scopes_updates_origin_to_the_new_path() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "old", "fields:\n  x:\n    required: true\n");

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let remapped = cache.with_remapped_scopes(|d| {
            if d == Path::new("old") {
                Some(PathBuf::from("new"))
            } else {
                None
            }
        });

        let resolved = remapped.resolve_for(Path::new("new/doc.md"));
        assert_eq!(
            resolved.origin()["x"],
            "new/",
            "provenance must name the field's NEW governing directory, not the old one"
        );
    }

    #[test]
    fn with_remapped_scopes_leaves_unmatched_directories_in_place() {
        let dir = TempDir::new().unwrap();
        write_schema(dir.path(), "old", "fields:\n  x:\n    required: true\n");
        write_schema(
            dir.path(),
            "elsewhere",
            "fields:\n  w:\n    required: true\n",
        );

        let cache = SchemaCache::build_for_test(dir.path(), &empty_config());
        let remapped = cache.with_remapped_scopes(|d| {
            if d == Path::new("old") {
                Some(PathBuf::from("new"))
            } else {
                None
            }
        });

        assert!(
            remapped
                .resolve_for(Path::new("elsewhere/doc.md"))
                .fields
                .contains_key("w"),
            "a scope the remap function declines still resolves exactly as before"
        );
    }

    // -- tree model (#346) --------------------------------------------------

    fn file(yaml: &str) -> SchemaFile {
        let file: SchemaFile = serde_yaml_ng::from_str(yaml).unwrap();
        file.validate_self().unwrap();
        file
    }

    fn files(entries: &[(&str, &str)]) -> BTreeMap<PathBuf, SchemaFile> {
        entries
            .iter()
            .map(|(dir, yaml)| (PathBuf::from(dir), file(yaml)))
            .collect()
    }

    fn legacy_config() -> FrontmatterConfig {
        FrontmatterConfig {
            required: vec!["title".into(), "planning.method".into()],
            indexed_fields: vec!["type".into(), "tags".into(), "planning.method".into()],
            defaults: HashMap::from([("status".into(), "draft".into())]),
            allowed: HashMap::from([
                ("type".into(), vec!["note".into(), "recipe".into()]),
                ("tags".into(), vec!["a".into(), "b".into()]),
            ]),
        }
    }

    /// Fixtures covering every feature the README's schema-file section describes:
    /// nested vs flat authoring, `$values` splicing (leading, trailing, alone, with
    /// nothing to inherit), the deprecated `extend`, defaults, `open: false`,
    /// containers defaulting to `object`, typeless values-only fields, `values: []`,
    /// a multi-level per-attribute cascade, sibling scopes, a dotted key under an
    /// undeclared parent, the config-root fallback, and a root file replacing it.
    fn equivalence_fixtures() -> Vec<(
        &'static str,
        FrontmatterConfig,
        BTreeMap<PathBuf, SchemaFile>,
    )> {
        vec![
            ("config root only", legacy_config(), BTreeMap::new()),
            (
                "config root fallback under child scopes",
                legacy_config(),
                files(&[
                    (
                        "food",
                        "fields:\n  tags:\n    values: [$values, c]\n  planning:\n    \
                         fields:\n      method:\n        type: enum\n        values: [oven]\n",
                    ),
                    (
                        "food/recipes",
                        "fields:\n  planning.method:\n    values: [braise, $values]\n  \
                         title:\n    indexed: true\n",
                    ),
                ]),
            ),
            (
                "root file replaces config root",
                legacy_config(),
                files(&[
                    (
                        "",
                        "fields:\n  title:\n    type: text\n    required: true\n",
                    ),
                    ("notes", "fields:\n  tags:\n    values: [$values]\n"),
                ]),
            ),
            (
                "nested and flat authoring across scopes",
                empty_config(),
                files(&[
                    (
                        "",
                        "fields:\n  planning:\n    type: object\n    open: false\n    fields:\n      \
                         prep_minutes:\n        type: integer\n        indexed: true\n      \
                         method:\n        values: [oven]\n  a:\n    type: object\n    fields:\n      \
                         b:\n        type: text\n  a-b:\n    type: text\n  \
                         deep.er.path:\n    type: boolean\n",
                    ),
                    (
                        "kitchen",
                        "fields:\n  planning.prep_minutes:\n    required: true\n  \
                         planning:\n    open: true\n  a.c:\n    type: date\n",
                    ),
                    (
                        "kitchen/recipes",
                        "fields:\n  planning:\n    fields:\n      method:\n        \
                         values: [$values, braise]\n        default: oven\n  deep:\n    \
                         fields:\n      er:\n        required: true\n",
                    ),
                    ("garden", "fields:\n  planning:\n    required: true\n"),
                ]),
            ),
            (
                "values splicing and extend",
                empty_config(),
                files(&[
                    (
                        "",
                        "fields:\n  tags:\n    type: list\n    values: [b, a]\n  \
                         status:\n    values: [draft, done]\n    default: draft\n  \
                         closed:\n    type: text\n    values: []\n  open_text:\n    type: text\n",
                    ),
                    (
                        "x",
                        "fields:\n  tags:\n    values: [c, $values]\n  status:\n    \
                         extend: true\n    values: [archived]\n  lonely:\n    \
                         values: [$values]\n  closed:\n    values: [$values, now]\n",
                    ),
                    (
                        "x/y",
                        "fields:\n  tags:\n    values: [$values, a, d]\n  status:\n    \
                         values: [only]\n  open_text:\n    required: true\n",
                    ),
                ]),
            ),
            (
                "three levels each setting a different attribute",
                empty_config(),
                files(&[
                    ("", "fields:\n  rating:\n    type: integer\n"),
                    ("a", "fields:\n  rating:\n    required: true\n"),
                    (
                        "a/b",
                        "fields:\n  rating:\n    indexed: true\n    default: 3\n",
                    ),
                    (
                        "a/b/c",
                        "fields:\n  rating:\n    type: number\n    required: false\n",
                    ),
                ]),
            ),
        ]
    }

    #[test]
    fn tree_resolver_matches_the_flat_reference_resolver() {
        for (name, config, files) in equivalence_fixtures() {
            let new_root = ResolvedSchema::from_config(&config);
            let old_root = flat_reference::from_config(&config);
            flat_reference::assert_equivalent(&new_root, &old_root, &format!("{name}: config"));

            let new = merge_cascade(&files, &new_root);
            let old = flat_reference::merge_cascade(&files, &old_root);
            assert_eq!(new.len(), old.len(), "{name}");
            for ((new_dir, new_schema), (old_dir, old_schema)) in new.iter().zip(&old) {
                assert_eq!(new_dir, old_dir, "{name}");
                flat_reference::assert_equivalent(
                    new_schema,
                    old_schema,
                    &format!("{name}: {}", scope_label(new_dir)),
                );
            }
        }
    }

    #[test]
    fn the_flat_view_keeps_containers_and_omits_implicit_parents() {
        let resolved = ResolvedSchema::default().merged_with(
            &file(
                "fields:\n  planning:\n    fields:\n      method:\n        type: text\n  \
                 deep.er:\n    type: text\n",
            ),
            "s/",
        );
        assert_eq!(resolved.fields["planning"].ty, Some(FieldType::Object));
        assert!(
            resolved.fields["planning"].fields.is_empty(),
            "a flat-view entry carries no children; each is an entry of its own"
        );
        assert!(resolved.fields.contains_key("planning.method"));
        assert!(
            !resolved.fields.contains_key("deep"),
            "a parent only reached through a dotted key is not declared"
        );
        assert!(resolved.fields.contains_key("deep.er"));
        // The tree holds the implicit parent, undeclared.
        let deep = &resolved.fields.tree()["deep"];
        assert!(deep.declared_in.is_none());
        assert!(deep.fields["er"].declared_in.is_some());
        assert!(resolved.fields.lookup("deep").is_none());
        assert_eq!(
            resolved.fields.lookup("planning").unwrap().fields["method"].ty,
            Some(FieldType::Text),
            "lookup returns the tree node with its children"
        );
    }

    #[test]
    fn attribute_origins_name_the_scope_that_set_each_attribute() {
        let resolved = ResolvedSchema::default()
            .merged_with(&file("fields:\n  rating:\n    type: integer\n"), "/")
            .merged_with(&file("fields:\n  rating:\n    required: true\n"), "a/")
            .merged_with(&file("fields:\n  rating:\n    default: 3\n"), "a/b/");
        let rating = &resolved.fields["rating"];
        assert_eq!(rating.set_by.ty.as_deref(), Some("/"));
        assert_eq!(rating.set_by.required.as_deref(), Some("a/"));
        assert_eq!(rating.set_by.default.as_deref(), Some("a/b/"));
        assert_eq!(rating.set_by.indexed, None);
        assert_eq!(rating.declared_in.as_deref(), Some("a/b/"));
        assert_eq!(resolved.origin()["rating"], "a/b/");

        // A nested container's implied `type: object` is set by the scope that nests.
        let nested = ResolvedSchema::default().merged_with(
            &file("fields:\n  planning:\n    fields:\n      x:\n        type: text\n"),
            "n/",
        );
        assert_eq!(nested.fields["planning"].set_by.ty.as_deref(), Some("n/"));
    }

    #[test]
    fn items_resolve_into_the_tree_but_not_the_flat_view() {
        let resolved = ResolvedSchema::default().merged_with(
            &file(
                "fields:\n  npcs:\n    items:\n      type: object\n      fields:\n        \
                 name:\n          type: text\n          required: true\n  \
                 tags:\n    type: list\n    items:\n      type: enum\n      values: [a, b]\n",
            ),
            "/",
        );
        // `items` makes a typeless field a list, the way `fields` makes an object.
        assert_eq!(resolved.fields["npcs"].ty, Some(FieldType::List));
        assert_eq!(
            resolved.fields.keys().collect::<Vec<_>>(),
            vec!["npcs", "tags"],
            "nothing under items reaches the flat view"
        );
        assert!(resolved.origin().keys().all(|k| !k.contains("[]")));

        let element = resolved.fields.lookup("npcs[]").expect("items node");
        assert_eq!(element.ty, Some(FieldType::Object));
        let name = resolved
            .fields
            .lookup("npcs[].name")
            .expect("element field");
        assert_eq!(name.ty, Some(FieldType::Text));
        assert!(name.required);
        assert_eq!(name.declared_in.as_deref(), Some("/"));
        assert_eq!(
            resolved.fields.lookup("tags[]").unwrap().values,
            Some(vec!["a".into(), "b".into()])
        );
        // The flat entry keeps its element schema attached.
        assert_eq!(
            resolved.fields["tags"].items.as_ref().unwrap().ty,
            Some(FieldType::Enum)
        );
        assert!(resolved.fields.lookup("npcs[].missing").is_none());
    }

    #[test]
    fn items_inherit_per_attribute_down_the_cascade() {
        let resolved = ResolvedSchema::default()
            .merged_with(
                &file(
                    "fields:\n  npcs:\n    type: list\n    items:\n      type: object\n      \
                     fields:\n        name:\n          type: text\n        role:\n          \
                     type: enum\n          values: [ally, foe]\n",
                ),
                "/",
            )
            .merged_with(
                &file(
                    "fields:\n  npcs:\n    required: true\n    items:\n      fields:\n        \
                     name:\n          required: true\n  npcs[].role:\n    \
                     values: [$values, neutral]\n",
                ),
                "campaign/",
            );
        let npcs = &resolved.fields["npcs"];
        assert_eq!(npcs.ty, Some(FieldType::List), "the inherited type stays");
        assert!(npcs.required);

        let element = resolved.fields.lookup("npcs[]").unwrap();
        assert_eq!(element.ty, Some(FieldType::Object));
        assert_eq!(
            element.set_by.ty.as_deref(),
            Some("campaign/"),
            "nesting `fields` under items sets the element's object type at that scope"
        );

        let name = resolved.fields.lookup("npcs[].name").unwrap();
        assert_eq!(name.ty, Some(FieldType::Text), "type inherited");
        assert!(name.required, "required added by the child");
        assert_eq!(name.set_by.ty.as_deref(), Some("/"));
        assert_eq!(name.set_by.required.as_deref(), Some("campaign/"));

        let role = resolved.fields.lookup("npcs[].role").unwrap();
        assert_eq!(role.ty, Some(FieldType::Enum));
        assert_eq!(
            role.values,
            Some(vec!["ally".into(), "foe".into(), "neutral".into()]),
            "`$values` splices the inherited element values"
        );
    }

    #[test]
    fn items_count_toward_the_fingerprint_only_when_present() {
        let plain =
            ResolvedSchema::default().merged_with(&file("fields:\n  tags:\n    type: list\n"), "/");
        let with_items = ResolvedSchema::default().merged_with(
            &file("fields:\n  tags:\n    type: list\n    items:\n      type: text\n"),
            "/",
        );
        let with_other_items = ResolvedSchema::default().merged_with(
            &file("fields:\n  tags:\n    type: list\n    items:\n      type: integer\n"),
            "/",
        );
        let with_element_field = ResolvedSchema::default().merged_with(
            &file(
                "fields:\n  tags:\n    type: list\n    items:\n      type: object\n      \
                 fields:\n        x:\n          type: text\n",
            ),
            "/",
        );
        let prints = [
            plain.fingerprint(),
            with_items.fingerprint(),
            with_other_items.fingerprint(),
            with_element_field.fingerprint(),
        ];
        let distinct: BTreeSet<&String> = prints.iter().collect();
        assert_eq!(distinct.len(), prints.len(), "{prints:?}");
        assert_eq!(
            plain.fingerprint(),
            flat_reference::fingerprint(&flat_reference::merged_with(
                &Default::default(),
                &file("fields:\n  tags:\n    type: list\n"),
                "/"
            )),
            "a schema without items hashes as it did before items existed"
        );
    }

    #[test]
    fn field_paths_parse_and_round_trip() {
        use PathStep::{Field, Items};
        let f = |s: &str| Field(s.to_string());
        let cases: Vec<(&str, Vec<PathStep>)> = vec![
            ("a", vec![f("a")]),
            ("a.b", vec![f("a"), f("b")]),
            ("a[]", vec![f("a"), Items]),
            ("a[].b", vec![f("a"), Items, f("b")]),
            ("a[][]", vec![f("a"), Items, Items]),
            (
                "a.b[].c[].d",
                vec![f("a"), f("b"), Items, f("c"), Items, f("d")],
            ),
            // Brackets anywhere but a segment's end are part of the name.
            ("a[0]", vec![f("a[0]")]),
            ("a[]b", vec![f("a[]b")]),
            // Empty names parse (callers that need non-empty names check them).
            ("a..b", vec![f("a"), f(""), f("b")]),
            ("[]", vec![f(""), Items]),
            ("", vec![f("")]),
        ];
        for (text, steps) in cases {
            let path = FieldPath::parse(text);
            assert_eq!(path.steps(), steps.as_slice(), "parse {text:?}");
            assert_eq!(path.render(), text, "render {text:?}");
            assert_eq!(path.to_string(), text);
            assert_eq!(path.depth(), steps.len());
        }
    }

    #[test]
    fn field_kind_table_covers_every_type_in_every_consumer() {
        use crate::qdrant::IndexKind as K;
        // (type, name, accepted, rejected, index kind, numeric, vocabulary)
        let expected: [(FieldType, &str, Value, Value, K, NumericValue, bool); 9] = [
            (
                FieldType::Text,
                "text",
                json!("x"),
                json!(1),
                K::Keyword,
                NumericValue::None,
                false,
            ),
            (
                FieldType::Integer,
                "integer",
                json!(3),
                json!(3.5),
                K::Integer,
                NumericValue::Number,
                true,
            ),
            (
                FieldType::Number,
                "number",
                json!(3.5),
                json!("3"),
                K::Float,
                NumericValue::Number,
                true,
            ),
            (
                FieldType::Boolean,
                "boolean",
                json!(true),
                json!("true"),
                K::Bool,
                NumericValue::Boolean,
                true,
            ),
            (
                FieldType::Enum,
                "enum",
                json!(["a", 1]),
                json!([{"a": 1}]),
                K::Keyword,
                NumericValue::None,
                true,
            ),
            (
                FieldType::List,
                "list",
                json!([]),
                json!("a"),
                K::Keyword,
                NumericValue::None,
                true,
            ),
            (
                FieldType::Date,
                "date",
                json!("2024-01-31"),
                json!("2024-13-01"),
                K::Keyword,
                NumericValue::Epoch,
                true,
            ),
            (
                FieldType::Timestamp,
                "timestamp",
                json!("2024-01-31T10:00:00Z"),
                json!("2024-01-31"),
                K::Keyword,
                NumericValue::Epoch,
                false,
            ),
            (
                FieldType::Object,
                "object",
                json!({}),
                json!([]),
                K::Keyword,
                NumericValue::None,
                false,
            ),
        ];
        assert_eq!(FieldType::ALL.len(), expected.len());
        for (ty, (want_ty, name, good, bad, index, numeric, vocabulary)) in
            FieldType::ALL.into_iter().zip(expected)
        {
            assert_eq!(ty, want_ty);
            let kind = ty.kind();
            // Display name (also proves `FIELD_KINDS` is in declaration order): serde's spelling, the Debug-lowercase spelling it replaced,
            // and what `get_schema`/the web UI show.
            assert_eq!(kind.name, name);
            assert_eq!(ty.name(), name);
            assert_eq!(serde_json::to_value(ty).unwrap(), json!(name));
            assert_eq!(format!("{ty:?}").to_lowercase(), name);
            assert_eq!(
                serde_json::from_value::<FieldType>(json!(name)).unwrap(),
                ty
            );
            // Validation.
            assert!(check_type(ty, &good).is_ok(), "{name} accepts {good}");
            let err = check_type(ty, &bad).unwrap_err();
            assert!(err.contains(kind.describe), "{err}");
            // Qdrant index kind and the filter operations it supports.
            assert_eq!(ResolvedSchema::index_kind(Some(ty)), index, "{name}");
            assert_eq!(kind.index, index);
            assert_eq!(
                kind.index.supports_range(),
                matches!(ty, FieldType::Integer | FieldType::Number),
                "{name}: range filters"
            );
            // Equality filters, as the Qdrant lowering applies them to the index kind:
            // refused only on a float.
            let sample = match index {
                K::Integer => "3",
                K::Bool => "true",
                _ => "x",
            };
            let lowered = crate::qdrant::lower_field_filters(
                &[(
                    "f".to_string(),
                    crate::state::FieldFilter::AnyOf(vec![sample.to_string()]),
                )],
                &HashMap::from([("f".to_string(), index)]),
            );
            assert_eq!(
                lowered.is_err(),
                ty == FieldType::Number,
                "{name}: equality filters"
            );
            // The `value_num` the schema-driven projection (#349) stores for an
            // accepted value of this type.
            assert_eq!(kind.numeric, numeric, "{name}");
            let numeric_of_good = crate::document_fields::declared_numeric(numeric, &good);
            match numeric {
                NumericValue::Number => assert_eq!(numeric_of_good, good.as_f64()),
                NumericValue::Boolean => assert_eq!(numeric_of_good, Some(1.0)),
                NumericValue::Epoch => assert_eq!(
                    numeric_of_good,
                    crate::document_fields::epoch_seconds(good.as_str().unwrap()),
                    "{name}"
                ),
                NumericValue::None => assert_eq!(numeric_of_good, None, "{name}"),
            }
            if numeric == NumericValue::Epoch {
                assert!(numeric_of_good.is_some(), "{name}: {good} has an epoch");
            }
            assert_eq!(kind.vocabulary, vocabulary, "{name}");
        }
        assert_eq!(ResolvedSchema::index_kind(None), K::Keyword);
    }

    #[test]
    fn items_round_trip_through_yaml_and_validate_their_shape() {
        let original = file(
            "fields:\n  npcs:\n    type: list\n    items:\n      type: object\n      fields:\n        \
             name:\n          type: text\n",
        );
        let reparsed = file(&original.to_yaml().unwrap());
        assert_eq!(reparsed.fields, original.fields);

        for (yaml, needle) in [
            (
                "fields:\n  a:\n    type: text\n    items:\n      type: text\n",
                "only a list has items",
            ),
            (
                "fields:\n  a:\n    fields:\n      b:\n        type: text\n    items:\n      type: text\n",
                "both nested fields and items",
            ),
            (
                "fields:\n  a:\n    items:\n      values: [$valuez]\n",
                "a[]",
            ),
            (
                "fields:\n  a:\n    items:\n      type: text\n  a[]:\n    type: text\n",
                "declared more than once",
            ),
        ] {
            let parsed: SchemaFile = serde_yaml_ng::from_str(yaml).unwrap();
            let err = parsed.validate_self().unwrap_err();
            assert!(err.contains(needle), "{yaml}: {err}");
        }
    }

    #[test]
    fn edits_reach_list_items_through_the_path_grammar() {
        let mut schema = SchemaFile::default();
        schema
            .apply(&SchemaEdit::SetField {
                field: "npcs[].name".into(),
                definition: Box::new(RawFieldDef {
                    ty: Some(FieldType::Text),
                    ..RawFieldDef::default()
                }),
            })
            .unwrap();
        let npcs = &schema.fields["npcs"];
        assert_eq!(npcs.ty, Some(FieldType::List), "a missing list is created");
        let element = npcs.items.as_deref().unwrap();
        assert_eq!(element.ty, Some(FieldType::Object));
        assert_eq!(
            element.fields.as_ref().unwrap()["name"].ty,
            Some(FieldType::Text)
        );
        schema.validate_self().unwrap();

        schema
            .apply(&SchemaEdit::AddValues {
                field: "npcs[].role".into(),
                values: vec!["foe".into(), "ally".into()],
            })
            .unwrap();
        let resolved = ResolvedSchema::default().merged_with(&schema, "/");
        assert_eq!(
            resolved.fields.lookup("npcs[].role").unwrap().values,
            Some(vec!["ally".into(), "foe".into()])
        );
        schema
            .apply(&SchemaEdit::RemoveValues {
                field: "npcs[].role".into(),
                values: vec!["foe".into()],
            })
            .unwrap();

        // A flat `[]` key is found and edited in place.
        let mut flat = file("fields:\n  tags[]:\n    type: enum\n    values: [a]\n");
        flat.apply(&SchemaEdit::AddValues {
            field: "tags[]".into(),
            values: vec!["b".into()],
        })
        .unwrap();
        assert_eq!(
            flat.fields["tags[]"].values,
            Some(vec!["a".into(), "b".into()])
        );

        schema
            .apply(&SchemaEdit::RemoveField {
                field: "npcs[]".into(),
            })
            .unwrap();
        assert!(schema.fields["npcs"].items.is_none());
        assert!(
            schema
                .apply(&SchemaEdit::RemoveField {
                    field: "npcs[]".into()
                })
                .is_err()
        );

        // `[]` under a non-list, and a name under a list, are refused.
        let mut typed = file("fields:\n  title:\n    type: text\n  tags:\n    type: list\n");
        let err = typed
            .apply(&SchemaEdit::SetField {
                field: "title[]".into(),
                definition: Box::default(),
            })
            .unwrap_err();
        assert!(err.contains("not a list"), "{err}");
        let err = typed
            .apply(&SchemaEdit::SetField {
                field: "tags.x".into(),
                definition: Box::default(),
            })
            .unwrap_err();
        assert!(err.contains("not a container"), "{err}");
    }

    #[test]
    fn items_cannot_be_added_under_a_typeless_values_field() {
        // `values:` with no `type:` is a leniently-enforced scalar; `[]` under it would
        // retype it as a list.
        let mut schema = file("fields:\n  status:\n    values: [draft, done]\n");
        let err = schema
            .apply(&SchemaEdit::AddValues {
                field: "status[]".into(),
                values: vec!["x".into()],
            })
            .unwrap_err();
        assert!(
            err.contains("'status' is a scalar field with a values list, not a list"),
            "{err}"
        );
        assert!(schema.fields["status"].items.is_none());
    }

    #[test]
    fn a_flat_items_key_declares_its_implicit_parent_as_a_list() {
        let resolved = ResolvedSchema::default().merged_with(
            &file("fields:\n  tags[]:\n    type: enum\n    values: [a, b]\n"),
            "s/",
        );
        let tags = &resolved.fields["tags"];
        assert_eq!(tags.ty, Some(FieldType::List));
        assert_eq!(tags.set_by.ty.as_deref(), Some("s/"));
        assert_eq!(resolved.origin()["tags"], "s/");
        assert_eq!(
            tags.items.as_ref().unwrap().values,
            Some(vec!["a".into(), "b".into()])
        );
        // The element schema reaches the fingerprint through its list's entry.
        let without = ResolvedSchema::default()
            .merged_with(&file("fields:\n  tags:\n    type: list\n"), "s/");
        assert_ne!(resolved.fingerprint(), without.fingerprint());

        // A parent declared elsewhere keeps its own attributes.
        let declared = ResolvedSchema::default()
            .merged_with(&file("fields:\n  tags:\n    required: true\n"), "/")
            .merged_with(&file("fields:\n  tags[]:\n    type: text\n"), "s/");
        assert_eq!(declared.fields["tags"].ty, None);
        assert_eq!(declared.origin()["tags"], "/");
        assert!(declared.fields["tags"].required);
    }

    // -- explicit cascade semantics (#348) ------------------------------------

    const CASCADE_ROOT: &str = "fields:
  status:
    type: enum
    values: [draft, active, archived]
    required: true
  planning:
    fields:
      method:
        type: text
      effort:
        type: integer
  npcs:
    type: list
    items:
      fields:
        name:
          type: text
";

    #[test]
    fn removed_hides_an_inherited_field_in_the_scope_and_below_until_redeclared() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        write_schema(root, "", CASCADE_ROOT);
        write_schema(root, "games", "fields:\n  status:\n    removed: true\n");
        write_schema(root, "games/rpg", "fields:\n  notes:\n    type: text\n");
        write_schema(
            root,
            "games/rpg/campaign",
            "fields:\n  status:\n    type: text\n",
        );
        let cache = SchemaCache::build_for_test(root, &empty_config());

        let games = cache.resolve_for(Path::new("games/a.md"));
        assert!(!games.fields.contains_key("status"), "removed in the scope");
        let rpg = cache.resolve_for(Path::new("games/rpg/a.md"));
        assert!(!rpg.fields.contains_key("status"), "and in its descendants");
        assert!(rpg.fields.contains_key("notes"));
        // Sibling scopes and the root are untouched.
        assert!(cache.resolve_for(Path::new("a.md")).fields["status"].required);

        // A deeper scope declares it afresh: nothing of the removed definition comes
        // back with it.
        let campaign = cache.resolve_for(Path::new("games/rpg/campaign/a.md"));
        let status = &campaign.fields["status"];
        assert_eq!(status.ty, Some(FieldType::Text));
        assert!(!status.required);
        assert_eq!(status.values, None);
        assert_eq!(campaign.origin()["status"], "games/rpg/campaign/");
    }

    #[test]
    fn removed_reaches_nested_fields_and_list_items() {
        let resolved = ResolvedSchema::default()
            .merged_with(&file(CASCADE_ROOT), "/")
            .merged_with(
                &file(
                    "fields:\n  planning.effort:\n    removed: true\n  npcs[]:\n    removed: true\n",
                ),
                "x/",
            );
        assert!(resolved.fields.contains_key("planning.method"));
        assert!(!resolved.fields.contains_key("planning.effort"));
        assert!(resolved.fields["npcs"].items.is_none());
        assert!(resolved.fields.lookup("npcs[].name").is_none());
        // The nested spelling removes the same node.
        let nested = ResolvedSchema::default()
            .merged_with(&file(CASCADE_ROOT), "/")
            .merged_with(
                &file("fields:\n  planning:\n    fields:\n      effort:\n        removed: true\n"),
                "x/",
            );
        assert!(!nested.fields.contains_key("planning.effort"));
        assert!(nested.fields.contains_key("planning.method"));
    }

    #[test]
    fn a_removed_declaration_carries_nothing_else() {
        let err = serde_yaml_ng::from_str::<SchemaFile>(
            "fields:\n  status:\n    removed: true\n    required: true\n",
        )
        .unwrap()
        .validate_self()
        .unwrap_err();
        assert!(
            err.contains("'removed: true' alongside other attributes"),
            "{err}"
        );
        // `removed: false` is the same as not writing it.
        file("fields:\n  status:\n    removed: false\n    required: true\n");
    }

    #[test]
    fn exclude_values_narrow_the_inherited_set_after_the_local_values() {
        let parent = ResolvedSchema::default().merged_with(&file(CASCADE_ROOT), "/");
        let excluded = parent.merged_with(
            &file("fields:\n  status:\n    exclude_values: [archived]\n"),
            "x/",
        );
        assert_eq!(
            excluded.fields["status"].values,
            Some(vec!["draft".into(), "active".into()])
        );
        assert_eq!(
            excluded.fields["status"].set_by.values.as_deref(),
            Some("x/")
        );
        // Everything else is still inherited.
        assert!(excluded.fields["status"].required);
        assert_eq!(excluded.fields["status"].ty, Some(FieldType::Enum));

        // Applied after the splice: the local values extend, then the exclusion
        // narrows — a value in both lists is excluded.
        let both = parent.merged_with(
            &file(
                "fields:\n  status:\n    values: [$values, paused, gone]\n    \
                 exclude_values: [draft, gone]\n",
            ),
            "x/",
        );
        assert_eq!(
            both.fields["status"].values,
            Some(vec!["active".into(), "archived".into(), "paused".into()])
        );

        // An unconstrained field has nothing to exclude from.
        let open = parent.merged_with(
            &file("fields:\n  planning.method:\n    exclude_values: [x]\n"),
            "x/",
        );
        assert_eq!(open.fields["planning.method"].values, None);

        let err = serde_yaml_ng::from_str::<SchemaFile>(
            "fields:\n  status:\n    exclude_values: [$values]\n",
        )
        .unwrap()
        .validate_self()
        .unwrap_err();
        assert!(err.contains("exclude_values"), "{err}");
    }

    #[test]
    fn new_cascade_syntax_changes_the_fingerprint_only_where_it_has_an_effect() {
        let parent = ResolvedSchema::default().merged_with(&file(CASCADE_ROOT), "/");
        let plain = parent.merged_with(&file("fields:\n  notes:\n    type: text\n"), "x/");
        let excluded = parent.merged_with(
            &file("fields:\n  notes:\n    type: text\n  status:\n    exclude_values: [archived]\n"),
            "x/",
        );
        let removed = parent.merged_with(
            &file("fields:\n  notes:\n    type: text\n  status:\n    removed: true\n"),
            "x/",
        );
        assert_ne!(plain.fingerprint(), excluded.fingerprint());
        assert_ne!(plain.fingerprint(), removed.fingerprint());
        // Excluding nothing, or removing what is not there, changes nothing.
        let no_op = parent.merged_with(
            &file(
                "fields:\n  notes:\n    type: text\n  status:\n    exclude_values: []\n  \
                 ghost:\n    removed: true\n  phantom:\n    exclude_values: [x]\n  \
                 planning.ghost:\n    exclude_values: [x]\n  npcs[].ghost:\n    \
                 exclude_values: [x]\n",
            ),
            "x/",
        );
        assert_eq!(plain.fingerprint(), no_op.fingerprint());
        // Excluding from a field nothing above declares creates no field at all.
        // (`status`'s empty exclusion still moves its provenance to `x/`.)
        assert!(no_op.fields.keys().eq(plain.fields.keys()));
        assert!(!no_op.fields.contains_key("phantom"));
        assert!(!no_op.fields.contains_key("planning.ghost"));
        assert!(no_op.fields.lookup("npcs[].ghost").is_none());
    }

    #[test]
    fn values_edits_and_patches_refuse_a_field_this_scope_removes() {
        let parent = ResolvedSchema::default().merged_with(&file(CASCADE_ROOT), "/");
        let inherited = parent.fields.lookup("status").cloned();
        let removed = file("fields:\n  status:\n    removed: true\n");
        for edit in [
            SchemaEdit::AddValues {
                field: "status".into(),
                values: vec!["draft".into()],
            },
            SchemaEdit::RemoveValues {
                field: "status".into(),
                values: vec!["draft".into()],
            },
            patch(json!({"required": true})),
        ] {
            let mut child = removed.clone();
            let err = child
                .apply_inheriting(&edit, inherited.as_ref())
                .unwrap_err();
            assert!(
                err.contains("'status' is removed in this scope; re-declare it with set_field"),
                "{err}"
            );
            assert!(child.fields["status"].removed, "left as it was");
        }
        // The same within one batch: removing, then adding values, is refused.
        let mut child = SchemaFile::default();
        child
            .apply_inheriting(
                &SchemaEdit::SetField {
                    field: "status".into(),
                    definition: Box::new(RawFieldDef {
                        removed: true,
                        ..RawFieldDef::default()
                    }),
                },
                inherited.as_ref(),
            )
            .unwrap();
        assert!(
            child
                .apply_inheriting(
                    &SchemaEdit::AddValues {
                        field: "status".into(),
                        values: vec!["draft".into()],
                    },
                    inherited.as_ref(),
                )
                .is_err()
        );
    }

    #[test]
    fn excluding_or_replacing_the_inherited_default_warns() {
        let parent = "fields:\n  status:\n    type: enum\n    values: [draft, active]\n    \
                      default: draft\n";
        for child in [
            "fields:\n  status:\n    exclude_values: [draft]\n",
            "fields:\n  status:\n    values: [active, paused]\n",
        ] {
            let warnings = cascade_lint(parent, child);
            assert_eq!(warnings.len(), 1, "{child}: {warnings:?}");
            assert!(
                warnings[0].contains("inherited default \"draft\" is no longer a permitted"),
                "{warnings:?}"
            );
        }
        // Still permitted, or a default of the scope's own: no warning here.
        for child in [
            "fields:\n  status:\n    exclude_values: [active]\n",
            "fields:\n  status:\n    values: [$values, paused]\n",
            "fields:\n  status:\n    exclude_values: [draft]\n    default: active\n",
        ] {
            assert!(cascade_lint(parent, child).is_empty(), "{child}");
        }
    }

    #[test]
    fn kind_families_group_types_by_the_value_they_describe() {
        use KindFamily as F;
        let expected = [
            (FieldType::Text, F::String),
            (FieldType::Integer, F::Numeric),
            (FieldType::Number, F::Numeric),
            (FieldType::Boolean, F::Boolean),
            (FieldType::Enum, F::String),
            (FieldType::List, F::List),
            (FieldType::Date, F::String),
            (FieldType::Timestamp, F::String),
            (FieldType::Object, F::Object),
        ];
        assert_eq!(expected.len(), FieldType::ALL.len());
        for (ty, family) in expected {
            assert_eq!(ty.kind().family, family, "{}", ty.name());
        }
    }

    fn cascade_lint(parent_yaml: &str, child_yaml: &str) -> Vec<String> {
        let parent = ResolvedSchema::default().merged_with(&file(parent_yaml), "/");
        lint_cascade(&parent, &file(child_yaml))
    }

    #[test]
    fn a_cross_kind_type_change_warns_and_a_same_kind_one_does_not() {
        let warnings = cascade_lint(CASCADE_ROOT, "fields:\n  status:\n    type: list\n");
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].contains("'enum' (set in /) to 'list'"),
            "{warnings:?}"
        );
        // Nested, and implied by `fields:`.
        let warnings = cascade_lint(
            CASCADE_ROOT,
            "fields:\n  planning:\n    fields:\n      effort:\n        type: boolean\n  \
             npcs:\n    fields:\n      x:\n        type: text\n",
        );
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert!(warnings.iter().any(|w| w.contains("'planning.effort'")));
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("'list'") && w.contains("'object'"))
        );

        for same_kind in [
            "fields:\n  status:\n    type: text\n",
            "fields:\n  planning.effort:\n    type: number\n",
            "fields:\n  planning.method:\n    type: enum\n    values: [a]\n",
            "fields:\n  npcs[].name:\n    type: date\n",
        ] {
            assert!(
                cascade_lint(CASCADE_ROOT, same_kind).is_empty(),
                "{same_kind}"
            );
        }
        // A brand-new field, or one whose type is not inherited, is no change.
        assert!(cascade_lint(CASCADE_ROOT, "fields:\n  other:\n    type: list\n").is_empty());
    }

    #[test]
    fn removing_or_excluding_what_is_not_inherited_warns() {
        let warnings = cascade_lint(
            CASCADE_ROOT,
            "fields:
  ghost:
    removed: true
  status:
    exclude_values: [archived, nope]
  planning.method:
    exclude_values: [x]
",
        );
        assert_eq!(warnings.len(), 3, "{warnings:?}");
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("'ghost'") && w.contains("removes nothing"))
        );
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("'status'") && w.contains(": nope"))
        );
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("'planning.method'") && w.contains("no effect"))
        );
    }

    #[test]
    fn cascade_lint_findings_reach_the_cache_and_never_stop_a_build() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        write_schema(root, "", CASCADE_ROOT);
        write_schema(root, "a", "fields:\n  status:\n    type: list\n");
        write_schema(root, "a/b", "fields:\n  status:\n    type: object\n");
        let cache = SchemaCache::build_for_test(root, &empty_config());
        let findings = cache.lint_findings();
        // `a/` against the root, `a/b/` against `a/` (a list becoming an object).
        assert!(
            findings
                .iter()
                .any(|(dir, w)| dir == Path::new("a") && w.contains("'enum'")),
            "{findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|(dir, w)| dir == Path::new("a/b") && w.contains("'list'")),
            "{findings:?}"
        );
        // The root file inherits nothing, so it never gets a cascade finding.
        assert!(findings.iter().all(|(dir, _)| !dir.as_os_str().is_empty()));
    }

    #[test]
    fn remove_values_on_an_inherited_value_excludes_it_and_add_values_brings_it_back() {
        let parent = ResolvedSchema::default().merged_with(&file(CASCADE_ROOT), "/");
        let inherited = parent.fields.lookup("status").cloned();
        let mut child = SchemaFile::default();

        let summary = child
            .apply_inheriting(
                &SchemaEdit::RemoveValues {
                    field: "status".into(),
                    values: vec!["archived".into(), "nope".into()],
                },
                inherited.as_ref(),
            )
            .unwrap();
        assert!(summary.contains("removed 1 value(s)"), "{summary}");
        assert!(summary.contains("now excluded here"), "{summary}");
        let def = &child.fields["status"];
        assert_eq!(def.exclude_values, Some(vec!["archived".into()]));
        assert_eq!(def.values, None, "the inherited list is not copied");
        child.validate_self().unwrap();
        let resolved = parent.merged_with(&child, "x/");
        assert_eq!(
            resolved.fields["status"].values,
            Some(vec!["draft".into(), "active".into()])
        );

        // Round trip: the exclusion survives YAML, and adding the value back drops it.
        let mut child: SchemaFile = serde_yaml_ng::from_str(&child.to_yaml().unwrap()).unwrap();
        let summary = child
            .apply_inheriting(
                &SchemaEdit::AddValues {
                    field: "status".into(),
                    values: vec!["archived".into()],
                },
                inherited.as_ref(),
            )
            .unwrap();
        assert_eq!(summary, "added to 'status': archived");
        assert!(
            !child.fields.contains_key("status"),
            "a declaration that only excluded is gone"
        );
        assert_eq!(
            parent.merged_with(&child, "x/").fields["status"],
            parent.fields["status"]
        );

        // A value this scope lists itself is just deleted from its list.
        let mut own = file("fields:\n  status:\n    values: [$values, paused]\n");
        own.apply_inheriting(
            &SchemaEdit::RemoveValues {
                field: "status".into(),
                values: vec!["paused".into()],
            },
            inherited.as_ref(),
        )
        .unwrap();
        assert_eq!(own.fields["status"].values, Some(vec!["$values".into()]));
        assert_eq!(own.fields["status"].exclude_values, None);

        // With nothing inherited and nothing declared, it is still an error.
        let err = SchemaFile::default()
            .apply_inheriting(
                &SchemaEdit::RemoveValues {
                    field: "status".into(),
                    values: vec!["x".into()],
                },
                None,
            )
            .unwrap_err();
        assert!(err.contains("not declared in this scope"), "{err}");
    }

    fn patch(json: Value) -> SchemaEdit {
        SchemaEdit::PatchField {
            field: "status".into(),
            patch: json.as_object().unwrap().clone(),
        }
    }

    #[test]
    fn patch_field_changes_only_the_given_keys_and_minimizes_against_the_parent() {
        let parent = ResolvedSchema::default().merged_with(&file(CASCADE_ROOT), "/");
        let inherited = parent.fields.lookup("status").cloned();
        let mut child = file("fields:\n  status:\n    indexed: true\n    default: draft\n");

        // Every patched key repeats what the parent already says, so none is
        // written: only what differs from the inherited definition is.
        let summary = child
            .apply_inheriting(
                &patch(json!({"type": "enum", "required": true, "values": ["draft", "active", "archived"], "open": true})),
                inherited.as_ref(),
            )
            .unwrap();
        assert_eq!(summary, "'status' unchanged");
        assert_eq!(
            child.fields["status"],
            RawFieldDef {
                indexed: Some(true),
                default: Some(json!("draft")),
                ..RawFieldDef::default()
            }
        );

        child
            .apply_inheriting(&patch(json!({"required": false})), inherited.as_ref())
            .unwrap();
        let status = &child.fields["status"];
        assert_eq!(status.required, Some(false));
        assert_eq!(status.indexed, Some(true), "untouched keys stay");

        // `null` drops the local setting: the attribute is inherited again.
        child
            .apply_inheriting(
                &patch(json!({"required": null, "default": null})),
                inherited.as_ref(),
            )
            .unwrap();
        let status = &child.fields["status"];
        assert_eq!(status.required, None);
        assert_eq!(status.default, None);
        assert!(parent.merged_with(&child, "x/").fields["status"].required);

        // Patching the last local attribute back to the inherited value removes the
        // declaration altogether.
        let summary = child
            .apply_inheriting(&patch(json!({"indexed": false})), inherited.as_ref())
            .unwrap();
        assert!(
            summary.contains("now inherits its whole definition"),
            "{summary}"
        );
        assert!(!child.fields.contains_key("status"));

        // And on a scope that never declared it, a no-op patch is refused rather than
        // writing an empty file.
        let err = child
            .apply_inheriting(&patch(json!({"required": true})), inherited.as_ref())
            .unwrap_err();
        assert!(err.contains("nothing to change"), "{err}");
    }

    #[test]
    fn patch_field_merges_nested_fields_recursively() {
        let parent = ResolvedSchema::default().merged_with(&file(CASCADE_ROOT), "/");
        let inherited = parent.fields.lookup("planning").cloned();
        let mut child = file(
            "fields:\n  planning:\n    fields:\n      method:\n        required: true\n      \
             extra:\n        type: text\n",
        );
        let edit = SchemaEdit::PatchField {
            field: "planning".into(),
            patch: json!({
                "open": false,
                "fields": {
                    "method": {"indexed": true},
                    "effort": {"type": "integer"},
                    "extra": null
                }
            })
            .as_object()
            .unwrap()
            .clone(),
        };
        child.apply_inheriting(&edit, inherited.as_ref()).unwrap();
        child.validate_self().unwrap();
        let planning = &child.fields["planning"];
        assert_eq!(planning.open, Some(false));
        let children = planning.fields.as_ref().unwrap();
        assert_eq!(children["method"].required, Some(true), "kept");
        assert_eq!(children["method"].indexed, Some(true), "added");
        // `effort` says exactly what it inherits, so it is not written at all.
        assert!(!children.contains_key("effort"), "{children:?}");
        assert!(!children.contains_key("extra"), "null removed the child");

        // A shape error in the merged definition is a refusal, not a corrupt file.
        let err = child
            .apply_inheriting(
                &SchemaEdit::PatchField {
                    field: "planning".into(),
                    patch: json!({"typ": "text"}).as_object().unwrap().clone(),
                },
                inherited.as_ref(),
            )
            .unwrap_err();
        assert!(err.contains("invalid patch for 'planning'"), "{err}");
    }
}
