Create, edit, and/or move a document. Commits and pushes; the change becomes
searchable shortly after this returns.

Edit modes: `content` (the whole file, frontmatter included — creates or
replaces), `old_string`/`new_string` (replace one exact, unique occurrence),
`frontmatter_patch` (structured frontmatter edits, body untouched), `append`
(add to the end of the body). `new_path` relocates, with an edit or alone.

`content` and `old_string`/`new_string` are whole-document edits, mutually
exclusive with each other AND with `frontmatter_patch`/`append` (which may
combine: patch first, then append). Exactly one edit mode is required unless
the call is a pure move (`new_path` alone).

Frontmatter — a `frontmatter_patch` result included — is validated against the
*destination* directory's schema. `expected_hash` rejects an edit built on a
stale read.

`documents` writes several documents as ONE atomic commit; a batch call passes
only `documents` (and optionally `message`).

`structured_content` mirrors the text summary: `outcome`, `sha`,
`rebased_paths` (paths a concurrent push pulled in during the pre-push rebase),
`rewritten_paths` (documents whose links into a moved one were rewritten), and
`diff`, capped for size (`diff_truncated`/`diff_total_bytes`; the text content
has it in full). `sync_failure_cause` appears only when `outcome` is
`committed_pending_sync`.
