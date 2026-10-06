Read a document by path; returns its markdown, frontmatter included.

A long document comes back as an outline instead of whole: read it by section
with `heading_path` or `line` (`outline: true` lists headings with line
ranges), or by `start_line`/`end_line`, which is always served in full. A long
document without headings is cut at a line instead (`truncated`, `end_line`):
read on from `end_line + 1` with `start_line`.
