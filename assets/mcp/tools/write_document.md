Create, edit, and/or move a document. The change is saved on return and becomes
searchable shortly after.

Edit with `content` (whole file), `old_string`/`new_string`,
`frontmatter_patch` and/or `append`; `new_path` moves (a directory moves its
subtree); `documents` writes several as one change.

Frontmatter is validated against the destination directory's rules. If the
result says `merged_with_other_changes`, a concurrent edit was merged in:
re-read with `get_document` before editing further.
