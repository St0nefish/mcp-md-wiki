Find documents. With `query`, results are ranked by semantic relevance.

Narrow with `filters`, keyed by frontmatter field — a scalar means equals
(`{"type": "guide"}`), an array means any-of, an object means all-of or a numeric
range (`{"planning.prep_minutes": {"lt": 30}}`). Nested fields use dot-paths.
`path_prefix` restricts by location: a case-insensitive **substring** of the
document's path. It matches anywhere in that path, so it is also how you find a
document from a fragment of its name — `stir_fr` finds `kitchen/recipes/stir_fry.md`,
and `recipes/` finds everything under any `recipes` folder. `sysadmin/` and
`sysadmin` behave identically. Being a substring rather than a prefix, a short
needle is broad: `sys` matches `sysadmin/` and `archive/old-sys/` alike, so prefer
the longest fragment you are sure of. A needle matching more documents than the
server will filter on at once sets `path_prefix_truncated: true` on the response,
rather than silently handing back fewer matches than actually exist.

`offset` pages results. With a query, paging reaches only as far as this search's
own ranked candidate pool, not the whole corpus: past that depth the response
marks `offset_truncated: true` instead of quietly returning less than you asked
for — narrow the query or stop paging rather than trust an empty page as "no more
results".
