Read a document by path — repo-relative (`sysadmin/docker/foo.md`) or a unique
basename. Returns the complete markdown, frontmatter included, as long as it
fits this server's size limit. `content_hash` always covers the whole file.

A read you did not bound that exceeds the limit is not returned whole: a
document with headings comes back as its outline (`outline_only: true`, plus
`intro` — the range above the first heading, frontmatter included, or `null`);
one without headings is cut on a line boundary (`truncated: true`, `end_line`;
read on from `end_line + 1`). A `start_line`/`end_line` range is always served
in full, however large.

For a long, heading-structured document, read by section — a heading line
through everything nested under it: select one with `line` or `heading_path`
(optionally `levels_up`), or pass `outline: true` for headings with line
ranges. An oversized section degrades the same way an oversized document does.
Section and outline responses report `total_lines` (the whole document's) and
`partial`.

The response also carries the link graph: `links_out` (documents this one
links to) and `links_in` (documents linking here). Each entry's `kind` is
`markdown` (a link in the body) or `semantic` (an inferred similarity neighbor,
only when semantic edges are enabled); `exists: false` in `links_out` marks a
broken link. Both lists are capped per call — check `has_more`/`total`.
