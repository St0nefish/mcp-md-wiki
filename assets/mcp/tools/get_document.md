Read a document by path — repo-relative (`sysadmin/docker/foo.md`) or a unique
basename. Returns the complete markdown, frontmatter included.

Pass `start_line`/`end_line` to read part of a long document. The returned
`content_hash` always covers the whole file — hand it to `write_document` as
`expected_hash` to guard against editing content that moved under you.

For a long, heading-structured document, navigate by section. A section is a
heading line through the end of everything nested under it. Select one with
`line` (any line in it — a line on its heading or in the text before its first
sub-heading is the section itself; a line inside a sub-heading's range is that
sub-heading) or `heading_path` (e.g. `["Spells", "Fireball"]`). `heading_path`
resolves in tiers, each only when it names exactly one section: the full path;
then a trailing suffix (`["Fireball"]` alone is enough); then an ordered match
that may also skip a *middle* segment (`["Feats", "Dual Wielding"]` matches
`Feats > Combat > Dual Wielding`, dropping "Combat"). Each segment is a
complete heading name, and matching ignores case, invisible characters such as
soft hyphens, and differences in whitespace, but not Unicode normalization
form — a precomposed accented character (`é`) does not match a decomposed
spelling of the same text (`e` + combining acute accent). Segments must still
appear in order — `["Dual Wielding", "Feats"]` does not match the same
section. Optionally pass `levels_up` to climb that many parent headings (a
heading's parent is the nearest heading above it with a lower level, whatever
the level gap). Heading text longer than 200 characters is cut to its
first 200 in outlines and paths (zero-width joiners, direction marks and variation
selectors don't count toward the cap); passing the full text still matches. `line` and
`heading_path` are two ways to name the same section, never a different read.

`outline: true` returns headings with line ranges instead of content: the whole
document's on its own, or just the sub-headings of the section you select with
`line`/`heading_path`. `start_line`/`end_line` can't be combined with any of
`line`, `heading_path` or `outline`. `line` must be 1 or greater, and
`heading_path` may not contain an empty or blank segment.

If `heading_path` names more than one section — the exact path is duplicated
(common in rulebook conversions with repeated section names), or the given
segments match more than one section in order — the response is an error
listing every candidate's line range: pick one with `line` when the
candidates' full paths are identical (a longer `heading_path` can't
distinguish two sections that already have the same one), or a longer or
reordered `heading_path` otherwise.

If nothing resolves, the error may still suggest candidates: sections whose
path the given segments match in order using a looser, per-segment substring
comparison (so `["Dual Wield"]` surfaces a `Dual Wielding` section) — these
are suggestions only, never a resolved read. Adjust `heading_path` to one of
them exactly, or fetch it by its `line`.

A selected section larger than the configured size limit comes back as
`outline_only: true`: no text, just its sub-heading outline — the same one
`outline: true` with that selector returns — so you can fetch a smaller part.
When the section also has text of its own before its first sub-heading, that
range is reported as `intro`; read it with `start_line`/`end_line`. A `search`
`section` row for a heading with sub-headings matched in that heading's own
text before its first sub-heading (its heading line, and the `intro` range if
any); the row's `hit_line_start`/`hit_line_end` name the matched lines. A section
with no sub-headings is returned in full regardless of size, flagged
`oversized: true`, since there is nothing smaller to offer.

Section and outline responses report `total_lines` (the whole document's line
count) and `partial`, on the same contract as a range read. Outlines are capped
to roughly the size limit's worth of headings, keeping the shallowest levels
first; a capped outline has `truncated: true`, `total_entries` (the uncapped
count) and a `hint` saying how to reach the rest — usually by outlining one of
the listed sections.

The response also carries `links_out` (documents this one links to) and
`links_in` (documents that link to this one) — the link graph, not just the
prose. Each entry has a `kind`: `markdown` for a link written in the document's
own body, or `semantic` for a machine-inferred similarity neighbor (only present
when semantic edges are enabled for this knowledge base). An entry in
`links_out` also carries `exists: false` when its target isn't currently an
indexed document — a broken link, not a link this tool failed to resolve. Both
lists are capped per call; check `has_more`/`total` before assuming you have
seen every edge.
