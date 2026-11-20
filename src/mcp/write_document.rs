use super::schema_tools::*;
use super::*;
use std::path::{Path, PathBuf};

use rmcp::{
    ErrorData as McpError, handler::server::wrapper::Parameters, schemars, tool, tool_router,
};
use tracing::error;

use crate::{
    config::ResolvedConfig,
    embed::EmbedClient,
    qdrant::QdrantStore,
    retrieval::{self},
    write::{
        self, DirectoryMoveError, DirectoryMoveSuccess, FrontmatterEdit, WriteDeps, WriteError,
        WriteRequest, WriteSuccess,
    },
};

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
    /// replaces it. Not combinable with another edit mode. Markdown is
    /// auto-formatted on save, so the stored text can differ from what you
    /// sent (`formatted` lists the changes): re-read before anchoring an edit.
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
    /// Retire `path` instead of editing it: as one change, move it under the
    /// server's archive directory, set `status: archived` when the governing
    /// schema allows that (the result says whether it did), and handle links
    /// pointing at it as `archive_links` says. Needs `expected_version`; takes
    /// no edit mode, `new_path` or `documents`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archive: Option<bool>,
    /// With `archive`: `repoint` rewrites links to the archived location,
    /// `unlink` replaces each with its plain text. Default: the server's setting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("enum" = ["repoint", "unlink"]))]
    pub archive_links: Option<String>,
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
    /// Also replace links to this document, in the documents that carry them,
    /// with their plain text, in the same change. Default false: links are left
    /// as they are.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unlink_inbound: Option<bool>,
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
pub(super) fn edit_operation_label(mode: Option<&EditMode>, is_move: bool) -> &'static str {
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
pub(super) fn build_frontmatter_edit(op: &FrontmatterPatchOp) -> Result<FrontmatterEdit, String> {
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
pub(super) fn parse_frontmatter_patch_ops(
    ops: &[FrontmatterPatchOp],
) -> Result<Vec<FrontmatterEdit>, String> {
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
pub(super) fn batch_input_as_write_params(input: &BatchDocumentInput) -> WriteDocumentParams {
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
        archive: None,
        archive_links: None,
        documents: None,
    }
}

/// Number of leading characters of `old_string` used as a search anchor when no exact
/// or whitespace-normalized match exists at all. Long enough to be a specific location
/// in most real documents, short enough that a single `str::find` over the document
/// stays cheap.
pub(super) const NOT_FOUND_ANCHOR_CHARS: usize = 40;

/// How many characters of surrounding document text to show on each side of an anchor
/// match, when reporting a near-match diagnostic. Kept small deliberately: this text
/// goes into an error a caller (and possibly its logs) will see, so it is an
/// orientation snippet, not a document excerpt.
pub(super) const NOT_FOUND_CONTEXT_CHARS: usize = 80;

/// Above this size, skip the near-match/anchor diagnostics below and fall back to the
/// plain not-found message. `old_content` is the on-disk document, which the write
/// path's `MAX_CONTENT_LEN` does not bound: it could predate that cap, have been
/// written outside these tools entirely, or be an over-cap document an edit is
/// shrinking (an edit is refused only for growing a document past the cap) — so
/// this is a second, independent bound rather than an assumption that the
/// write-path cap already covers it.
pub(super) const NOT_FOUND_DIAGNOSTIC_MAX_BYTES: usize = MAX_CONTENT_LEN;

/// Collapse every run of whitespace (including newlines) to a single space and trim
/// the ends, for a whitespace-insensitive comparison.
///
/// This treats differing indentation, trailing spaces, and CRLF-vs-LF line endings as
/// equal — by far the most common reason a caller's `old_string` fails to match
/// verbatim (e.g. it was retyped or reformatted rather than copied from
/// `get_document`'s output). One linear pass, no allocation beyond the output string,
/// so it stays cheap up to `NOT_FOUND_DIAGNOSTIC_MAX_BYTES`.
pub(super) fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A small window of `content` centered on the char-boundary byte offset `pos`,
/// clamped to `radius` characters on each side. Used to show just enough surrounding
/// text to orient a caller, never a large slice of the document.
pub(super) fn context_window(content: &str, pos: usize, radius: usize) -> &str {
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
pub(super) fn insert_merged(fields: &mut serde_json::Map<String, serde_json::Value>, merged: bool) {
    if merged {
        fields.insert(
            "merged_with_other_changes".to_string(),
            serde_json::json!(true),
        );
    }
}

/// The refusal for a change that no longer fits the document
/// (`WriteError::EditedElsewhere`).
pub(super) fn edited_elsewhere_error(rel_path: &str) -> McpError {
    McpError::invalid_params(format!("'{rel_path}': {}", write::EDITED_ELSEWHERE), None)
}

/// The instruction every refusal for an absolute change sent without
/// `expected_version` (`WriteError::VersionRequired`) gives, `action` being the
/// operation-specific rest of the sentence.
pub(super) fn pass_expected_version(action: &str) -> String {
    format!("pass expected_version (the version from get_document) to {action}")
}

/// The refusal for an absolute change sent without `expected_version`
/// (`WriteError::VersionRequired`).
pub(super) fn version_required_error(rel_path: &str) -> McpError {
    McpError::invalid_params(
        format!(
            "'{rel_path}' already exists: {}",
            pass_expected_version("replace, move or delete it")
        ),
        None,
    )
}

/// How `KbSearchServer::write_raw_file` ended when it did not fail outright.
pub(super) enum RawWrite {
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
pub(super) fn not_saved_error(what: &str) -> McpError {
    McpError::internal_error(
        format!("{what} could not be saved. Nothing was changed; try again."),
        None,
    )
}

/// The message a write tool returns when its change could not be saved AND
/// undoing the partial write failed too. That is an operator problem (logged at
/// `error!` where it happened), not something the caller can fix or retry its way
/// out of, so the caller is only told not to trust the current state.
pub(super) fn not_saved_unverified_error(what: &str) -> McpError {
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
pub(super) fn credential_error(e: impl std::fmt::Display) -> McpError {
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
pub(super) fn message_param_reason(reason: &str) -> String {
    reason.replace("commit message", "message")
}

/// A `write::validate_commit_message` rejection, worded for the `message` parameter.
pub(super) fn invalid_message_error(reason: String) -> McpError {
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
pub(super) fn write_response(
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
pub(super) fn insert_paths(
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
pub(super) fn batch_write_success_to_result(success: write::BatchWriteSuccess) -> CallToolResult {
    let documents: Vec<serde_json::Value> = success
        .documents
        .iter()
        .map(|d| {
            let mut fields = write_response(
                &d.rel_path,
                if d.is_create { "created" } else { "updated" },
                &d.diff,
                d.version.as_deref(),
                d.merged,
            );
            insert_formatting(&mut fields, &d.formatting);
            serde_json::Value::Object(fields)
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
pub(super) fn batch_write_document_error_text(err: &WriteError) -> String {
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
pub(super) fn batch_write_error_to_mcp_error(err: write::BatchWriteError) -> McpError {
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
pub(super) fn create_edit_success_to_result(
    success: WriteSuccess,
    rel_path: &str,
    is_create: bool,
    dest_path: Option<&str>,
    archive: bool,
) -> CallToolResult {
    let (path, action) = match dest_path {
        Some(dest) => (dest, "moved"),
        None if is_create => (rel_path, "created"),
        None => (rel_path, "updated"),
    };
    let action = if archive { "archived" } else { action };
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
    if archive {
        if success.status_set {
            fields.insert("status".to_string(), serde_json::json!("archived"));
        } else {
            fields.insert(
                "status_note".to_string(),
                serde_json::json!(
                    "the governing schema declares no status field that allows 'archived'; \
                     frontmatter left unchanged"
                ),
            );
        }
    }
    insert_paths(&mut fields, "rewritten_paths", &success.rewritten_paths);
    if let Some(related) = related_json(&success.related) {
        fields.insert("related".to_string(), related);
    }
    insert_formatting(&mut fields, &success.formatting);
    CallToolResult::structured(serde_json::Value::Object(fields))
}

/// `formatted` (the rumdl rules that changed the stored markdown) and
/// `formatting_skipped` (why formatting could not be applied) on a write result; both
/// omitted when formatting did nothing.
pub(super) fn insert_formatting(
    fields: &mut serde_json::Map<String, serde_json::Value>,
    formatting: &crate::write::FormatReport,
) {
    if !formatting.rules.is_empty() {
        fields.insert("formatted".to_string(), serde_json::json!(formatting.rules));
    }
    if let Some(reason) = &formatting.skipped {
        fields.insert("formatting_skipped".to_string(), serde_json::json!(reason));
    }
}

/// The `related` array of a create result (#356): `[{path, score}]` nearest first,
/// scores rounded like every other score; `None` when there are none, so the key
/// is omitted.
pub(crate) fn related_json(related: &[crate::write::RelatedDocument]) -> Option<serde_json::Value> {
    if related.is_empty() {
        return None;
    }
    Some(serde_json::Value::Array(
        related
            .iter()
            .map(|r| serde_json::json!({"path": r.path, "score": round_score(f64::from(r.score))}))
            .collect(),
    ))
}

/// The scope directory ([`crate::schema::scope_label`]) of a KB-relative schema file
/// path — how model-facing text names a schema file the server itself found.
pub(super) fn schema_file_scope(schema_file: &str) -> String {
    crate::schema::scope_label(
        std::path::Path::new(schema_file)
            .parent()
            .unwrap_or(std::path::Path::new("")),
    )
}

/// The refusal for a document write, delete or move that names a schema file
/// (`WriteError::SchemaFile`), pointing at the one tool that edits schemas. Echoes
/// only the caller's own path, never the schema file name convention.
pub(super) fn schema_file_path_error(schema_path: &str) -> McpError {
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
pub(super) fn create_edit_error_to_mcp_error(
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
pub(super) fn delete_success_to_result(success: WriteSuccess, rel_path: &str) -> CallToolResult {
    let mut fields = write_response(rel_path, "deleted", "", None, success.merged);
    insert_paths(&mut fields, "rewritten_paths", &success.rewritten_paths);
    insert_paths(&mut fields, "referencing_paths", &success.referencing_paths);
    CallToolResult::structured(serde_json::Value::Object(fields))
}

/// Map a `write::delete_document` failure onto this tool surface's `McpError`,
/// preserving the exact text/data shapes the existing delete tests pin down.
pub(super) fn delete_error_to_mcp_error(err: WriteError, rel_path: &str) -> McpError {
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
pub(super) fn move_directory_success_to_result(
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
pub(super) fn move_directory_error_to_mcp_error(
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

#[tool_router(router = write_router, vis = "pub(crate)")]
impl KbSearchServer {
    /// The `WriteDeps` every write tool hands the shared pipeline, from this
    /// server's fields and one config snapshot.
    pub(super) fn write_deps<'a>(
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
            markdown_format: &config.formatting.markdown,
            prepend_description: config.chunking.prepend_description,
            embedding_prefix_field: config.chunking.embedding_prefix_field.as_deref(),
            dedup_enabled: config.write.dedup_enabled,
            dedup_threshold: config.write.dedup_threshold,
            related_limit: config.write.related_limit,
            related_min_score: config.write.related_min_score,
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
        archive: Option<crate::config::ArchiveLinkPolicy>,
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
            archive,
        };

        match crate::write::write_document(&deps, req).await {
            Ok(success) => Ok(create_edit_success_to_result(
                success,
                rel_path,
                is_create,
                dest_path,
                archive.is_some(),
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
            ("archive", params.archive.is_some()),
            ("archive_links", params.archive_links.is_some()),
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
        if params.archive.is_some() || params.archive_links.is_some() {
            return Err(McpError::invalid_params(
                "archive is not valid when creating a new document — there is nothing at path \
                 to archive"
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
        crate::write::check_include_pattern_against(
            &self.include_patterns,
            &self.config().indexing,
            raw,
        )
        .map_err(|e| {
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
            None,
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
        if params.archive == Some(true) {
            return self.write_document_archive(&params, canonical).await;
        }
        if params.archive_links.is_some() {
            return Err(McpError::invalid_params(
                "archive_links needs archive: true".to_string(),
                None,
            ));
        }
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
            None,
        )
        .await
    }

    /// `write_document`'s archive path (#357): `path` resolved to an existing
    /// document, `archive: true`. A pure move to `<write.archive_prefix>/<path>`
    /// with `status: archived` and inbound-link handling, all in the pipeline's
    /// one commit (`write::WriteRequest::archive`).
    async fn write_document_archive(
        &self,
        params: &WriteDocumentParams,
        canonical: PathBuf,
    ) -> Result<CallToolResult, McpError> {
        for (field, is_set) in [
            ("content", params.content.is_some()),
            ("old_string", params.old_string.is_some()),
            ("new_string", params.new_string.is_some()),
            ("frontmatter_patch", params.frontmatter_patch.is_some()),
            ("append", params.append.is_some()),
            ("new_path", params.new_path.is_some()),
            ("force_new", params.force_new.is_some()),
        ] {
            if is_set {
                return Err(McpError::invalid_params(
                    format!(
                        "{field} is not valid with archive: an archive only takes path, \
                         expected_version, archive_links and message"
                    ),
                    None,
                ));
            }
        }
        let config = self.config();
        let policy = match params.archive_links.as_deref().map(str::trim) {
            None => config.write.archive_links,
            Some(p) if p.eq_ignore_ascii_case("repoint") => {
                crate::config::ArchiveLinkPolicy::Repoint
            }
            Some(p) if p.eq_ignore_ascii_case("unlink") => crate::config::ArchiveLinkPolicy::Unlink,
            Some(other) => {
                return Err(McpError::invalid_params(
                    format!("archive_links must be 'repoint' or 'unlink', not '{other}'"),
                    None,
                ));
            }
        };
        let rel_path = canonical
            .strip_prefix(&self.canonical_data_path)
            .unwrap_or(&canonical)
            .to_string_lossy()
            .into_owned();
        let dest = write::archive_destination(&config.write.archive_prefix, &rel_path)
            .map_err(|e| McpError::invalid_params(e, None))?;
        if self.canonical_data_path.join(&dest).exists() {
            return Err(McpError::invalid_params(
                format!(
                    "Cannot archive '{rel_path}': '{dest}' already exists. Move or delete the \
                     document already there first."
                ),
                None,
            ));
        }
        self.run_document_write(
            write::DocChange::Keep,
            &rel_path,
            params.message.as_deref(),
            "update",
            None,
            "write_document (archive)",
            Some(&dest),
            params.expected_version.as_deref(),
            Some(policy),
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
            ("archive", params.archive.is_some()),
            ("archive_links", params.archive_links.is_some()),
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
                    if let Err(e) = write::check_include_pattern_against(
                        &self.include_patterns,
                        &self.config().indexing,
                        &raw,
                    ) {
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
    pub(super) async fn write_document(
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
        // `check_include_pattern_against` — independent, KB-root-relative checks
        // that refuse a `.git` component and a symlink just as
        // `resolve_within_data` does. That re-validation is
        // what makes it safe to fall through for those two cases rather than treat
        // every non-success as a hard failure, and it is also what restores the
        // pre-merge priority: a path that both exists on disk and fails the
        // include-pattern check is reported with that specific message, not a
        // generic "not found" one.
        //
        // `Outside` and `Other` are different: `resolve_within_data` returns
        // `Outside` for a path that climbs above the root or passes through a
        // symlink (see its doc comment), and a create must not proceed at such a
        // path either. `Other` (a real I/O error resolving the path) has no safe
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
    pub(super) async fn delete_document(
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

        match crate::write::delete_document_with(
            &deps,
            &rel_path,
            params.message.as_deref(),
            params.expected_version.as_deref(),
            params.unlink_inbound.unwrap_or(false),
        )
        .await
        {
            Ok(success) => Ok(delete_success_to_result(success, &rel_path)),
            Err(err) => Err(delete_error_to_mcp_error(err, &rel_path)),
        }
    }
}
