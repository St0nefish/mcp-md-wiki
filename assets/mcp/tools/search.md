Find documents. With `query`, results are ranked by relevance.

Narrow with `filters`, keyed by frontmatter field (dot-paths for nested fields):
a scalar means equals, an array any-of, an object all-of or a numeric range
(`{"planning.prep_minutes": {"lt": 30}}`). `path_prefix` is a case-insensitive
**substring** of the document's path, not a prefix — `stir_fr` finds
`kitchen/recipes/stir_fry.md` — so it also finds a document from a fragment of
its name.

With a query, `offset` pages only as deep as this search's ranked candidate
pool, not the whole corpus: past it the response sets `offset_truncated: true`
— narrow the query or stop paging rather than read an empty page as "no more
results".
