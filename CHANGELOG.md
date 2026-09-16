# Changelog

This project has not yet cut a tagged release. Every merge to `master` builds and
pushes a new container image tagged by commit sha (`.github/workflows/release.yml`),
with `:latest` following `master` directly — there is no version number to pin to
yet, and no historical version list to reconstruct here. Until a tagging scheme
exists, this file tracks notable, operator-relevant changes on `master` under a
running `[Unreleased]` heading, in roughly chronological order (most recent first).
`fix #N` references are GitHub issues; see the repo's closed-issues list for the
complete history — recent activity included an automated multi-agent documentation
and correctness audit that closed roughly forty issues across security hardening,
indexing correctness, and doc drift, of which the entries below are a representative
sample rather than an exhaustive list.

<!-- verify-merge-a: inert marker for concurrent-PR auto-merge test, safe to delete -->

## [Unreleased]

### Retrieval

- **`start_line: 1` is valid against an empty document** (fix #298), a
  regression from #290: `get_document`/`/api/doc` now serve `start_line: 1`
  against a 0-byte document as `content: ""`, `end_line: 0`, `partial: false`
  instead of a `start_line past the end of the document` error, since reading
  from the beginning is meaningful even when there is nothing there. This is
  what let the web UI's `?start_line=1` viewer/editor fetch (added by #290)
  404 on an empty document; that call site is unchanged, the server now
  answers it correctly. `start_line: 2` or higher against an empty document
  is still an error, as is any out-of-range `start_line` against a non-empty
  document.
- **Whole-document reads and heading-less sections are capped** (#290).
  `search.section_max_bytes` (default 16000 bytes) now governs every
  `get_document` read the caller did not bound itself, not just section and
  outline modes. A bare `get_document` on a document over the cap returns the
  document's own outline instead of its text — `outline_only: true` plus
  `intro`, the range from line 1 (frontmatter included) through the line
  before the first heading, or `null` when there is nothing there — so the
  caller narrows with `heading_path`/`line`. With no headings to narrow into,
  the text is cut on a line boundary and flagged `truncated: true` with
  `end_line` naming the last line served; the same now applies to an
  `oversized` section with no sub-headings, which previously returned
  unbounded text. An explicit `start_line`/`end_line` range is still served in
  full, however large. No new config key, no reindex. The new keys appear only
  on responses that were impossible before, so existing response shapes are
  unchanged; both MCP `structured_content` and `/api/doc` get them, and MCP's
  text block carries a trailing note when it was cut. The web UI reads
  `/api/doc/{path}?start_line=1` so its viewer and editor always hold the
  whole file (the editor refuses to save anything it did not load in full).
- **`get_document` `heading_path` accepts a skipped middle segment, and
  suggests candidates when nothing resolves** (fix #291), extending #286's
  exact-path and suffix tiers with a third: an ordered, not-necessarily-
  contiguous subsequence match, so `["Feats", "Dual Wielding"]` now resolves
  against `Feats > Combat > Dual Wielding` even though it skips "Combat".
  Segment comparison stays exact at all three tiers, and each still resolves
  only when it names exactly one section — several matches at the same tier
  is `Ambiguous`, same as before. When no tier matches, the `NotFound` error
  now also carries `candidates`: sections the query matches as an ordered
  subsequence under a looser, substring-per-segment comparison (so
  `["Dual Wield"]` still surfaces a `Dual Wielding` section, without ever
  resolving to it), capped at 10 and ordered by fewest skipped segments then
  document order. The existing `hint` (a few top-level headings) remains for
  when there is nothing to suggest.
- **Small-to-big section retrieval** (#286): `get_document` gains `line`/
  `heading_path` (select the section containing a line, or matching a
  suffix of a heading path — two ways to name the same section) with
  optional `levels_up`, and `outline` (headings with line ranges: the whole
  document's, or with a selector just that section's sub-headings), so a
  client can fetch exactly the section it needs instead of a whole document
  or a raw line range whose end it couldn't know. A selected section over
  `search.section_max_bytes` (default 16000 bytes) comes back as
  `outline_only`: its sub-heading outline instead of text, plus an `intro`
  range (`null` when there is none) to read with `start_line`/`end_line`.
  Outlines are capped to about `section_max_bytes`, shallowest levels first,
  with `truncated`/`total_entries` and a `hint` on going deeper. The web UI's
  `GET /api/doc/{*path}` accepts the same `line`/`heading_path`/`levels_up`/
  `outline` query params (`outline` takes `true`/`false`; `?outline=1` is a
  400) with identical JSON. These parameters were previously ignored on both
  transports (the whole document came back); a request that passes them now
  gets a section, an outline, or — for combinations such as `start_line` with
  `outline` — an invalid-params error.
- **`section` granularity and `heading_prefix` filter** (#286), behind a new
  opt-in `chunking.heading_metadata` flag: `search` gains a `section`
  granularity — one path-only row (`heading_path`, the section's line range,
  the matched chunk's own `hit_line_start`/`hit_line_end`, score, `scope`, no
  text) per matching section, where `scope` (`section`, `preamble` or
  `whole_document`) says how to fetch it; a row for a heading with
  sub-headings matched in that heading's own text before its first
  sub-heading, which the hit range reads on its own — and a
  `heading_prefix` filter (query mode only) that matches a run of
  consecutive headings anywhere in a chunk's heading path. Each segment is a
  complete heading name: `["Conditions"]`, `["Chapter 10: Game Mastering",
  "Conditions"]` and `["Conditions", "Blinded"]` all match a chunk under
  `Chapter 10: Game Mastering > Conditions > Blinded`, while `["Chapter 10",
  "Conditions"]` does not. (`get_document`'s `heading_path`, by contrast,
  must match the end of a section's path.) Both match headings ignoring case
  (full Unicode case folding), invisible characters such as soft hyphens,
  zero-width spaces and joiners, and whitespace differences; a segment of
  only invisible characters is rejected as blank. With the flag on, chunk results carry `heading_path` (`[]`
  before the first heading); with it off the key is omitted. Chunk results
  now also carry `chunk_index` (MCP `structured_content` and `/api/search`),
  regardless of the flag.
- **`chunking.heading_metadata` changes chunk boundaries** (#286). While it is
  on, chunking never merges content across a heading: every chunk lies
  within one heading's own section (its heading line up to the next heading
  of any level) or the text before the first heading, so a `section` row
  names the deepest heading containing the match rather than a parent
  covering many short siblings. `target_chunk_size` has no effect; an
  oversized section still splits into several chunks, and a heading with no
  text before its first sub-heading becomes a small chunk of its heading
  line. Expect more, smaller chunks on documents with many short sections.
  With the flag off (the default) sections still merge up to
  `target_chunk_size`.
- **Heading detection follows CommonMark, which moves section boundaries in
  some existing documents** (#286, takes effect on the automatic re-embed
  below). Headings are parsed by pulldown-cmark instead of "a line starting
  with `#` outside a backtick or tilde fence": setext headings (`Title` over `===`
  or `---`, including a paragraph directly above a `---` line) and ATX
  headings indented by 1–3 spaces (`   # Title`) are now section boundaries;
  `#tag` lines, `#` followed by a non-breaking space, runs of 7+ `#`, and `#`
  lines inside HTML blocks are not; and a shorter fence inside a longer one, or a `~~~`
  line inside a backtick fence, no longer ends the code block. Headings
  inside blockquotes and list items remain non-boundaries, and with lone-CR
  (classic Mac) line endings only the first heading on a line counts.
  Heading text in breadcrumbs, `heading_path` and outlines is the rendered
  plain text — emphasis markers, link syntax and a closing `#`
  sequence are dropped, invisible characters that never affect rendering
  (soft hyphens, zero-width spaces, BOM) removed, whitespace collapsed — and
  is capped at 200 characters (cut on a character boundary, no ellipsis).
  Zero-width joiners, direction marks and variation selectors stay in the
  text, so emoji sequences and Persian or Indic joining display intact; they
  don't count toward the cap.
- **Config and schema surface** (#286). Both new config knobs default to
  preserving current behavior (`heading_metadata: false`;
  `search.granularities` defaults to all three granularities, but the
  effective set drops `section` while the flag is off). Disabled
  granularities and filters are removed from the MCP tool schema and
  description, not just rejected at call time — as are parameters and
  sentences that only apply to a disabled granularity or (with `document`
  disabled) to searching without a query — and errors name only enabled
  granularities. A `search.granularities` whose effective set is empty fails
  config load; explicitly setting it with `section` while the flag is off
  logs a warning (the default does not). Caller-visible changes on a
  default config: the `search` schema's `granularity` gains an `enum` of the
  enabled lowercase values (the server still accepts `"Chunk"`, but a
  schema-validating client will not send it); `search` and `get_document`
  tool and property descriptions are reworded; `heading_prefix`, previously
  ignored, is an invalid-params error while the flag is off, as is
  `granularity: "section"` (now "not enabled on this server" rather than
  "unknown granularity"); and the `fields`-at-chunk rejection is reworded.
  Enabling `chunking.heading_metadata` re-indexes the corpus automatically
  (see below); while an indexing run spanning more than one file is in
  progress (from the moment the reconcile scan first finds a document
  chunked under different settings), `section` and `heading_prefix` results
  carry a note (and
  `indexing_in_progress: true`) that they may be incomplete. That note is a
  sticky latch, not just an "is a run active right now" check: it stays on
  through a failed run's retry backoff or permanent give-up and the wait for
  the next reconcile, clearing only once a scan confirms nothing is left
  stale or the run that processed a scan's findings succeeds. A frozen
  scope's stale files never set it, by design — they are never re-chunked
  until the schema is fixed.
- **Changing `chunking.*` now re-chunks automatically — no silent mixed
  index.** Each `indexed_files` row stores a fingerprint of every
  `chunking.*` setting plus a code-level chunker version; the reconcile
  scan and the indexer's skip-if-unchanged check both treat a mismatch as
  dirty, so a chunking change applied by restart or `POST /admin/reload`
  re-chunks and re-embeds affected documents on the next reconcile. The
  reload response reports these settings under a new `reindex_scheduled`
  bucket instead of `reindex_required` (which now lists only
  `ui.semantic_edges.*`). **Upgrade note: one-time full re-embed.** Rows
  written before this change have no fingerprint, so the first reconcile
  after upgrading (startup) re-chunks and re-embeds the whole corpus once —
  expect embedding load proportional to corpus size. While
  `chunking.heading_metadata` is on, `target_chunk_size` has no effect and
  is left out of the fingerprint, so changing it re-embeds nothing.
- **Bug fix, takes effect on reindex:** chunk `line_start`/`line_end` now
  count from the top of the raw file (including frontmatter), matching
  `get_document` — previously they counted from the top of the frontmatter-
  stripped body. Also, a merged chunk's heading breadcrumb is now the
  deepest heading its merged sections actually share, rather than
  defaulting to the first merged section's — content from later sections
  was previously attributed to the wrong heading. The one-time
  post-upgrade re-embed above rewrites existing chunks with the corrected
  numbering, breadcrumbs and heading detection; no `index --full` is needed.
  **Exception:** documents under a frozen schema scope (a `.kb-schema.yaml`
  that fails to parse) are skipped by both the scan and the indexer, so
  they keep their old chunks — old line numbering, no heading metadata, and
  therefore absent from `section` and `heading_prefix` results — until the
  schema is fixed, which re-indexes them. Documents that fail validation
  likewise keep whatever chunks they already had.

### Documentation

- Added a Backup and Recovery section to `deploy/TROUBLESHOOTING.md` covering what's
  actually authoritative (the git-hosted corpus), how to rebuild `state.db` and the
  Qdrant collection from it, the frozen-schema interaction that can block that
  rebuild, and the current (passive-only) detection for a Qdrant data wipe (fix #184).
- Added a `deploy/TROUBLESHOOTING.md` entry for the schema-freezing failure mode —
  symptoms, `md-kb-rag validate`'s `SCHEMA ERRORS`/`FROZEN` output, and the fix
  (fix #188).
- Added a worked, three-level `.kb-schema.yaml` cascade example to `deploy/USAGE.md`,
  showing the resolved schema at each level and a matching `get_schema` excerpt
  (fix #190).
- Documented the `ui`/`ui.semantic_edges` config block in `deploy/config.example.yaml`
  (previously undocumented despite being a real, deployable feature), and closed a
  smaller drift gap around `search.min_score` found during the same pass (fix #197).
- Added this file, `CONTRIBUTING.md`, and `.github/ISSUE_TEMPLATE`/
  `.github/pull_request_template.md` (fix #189).

### Server hardening

- **Self-contained tool input schemas** (#288): `tools/list` now returns
  schemas with no `$ref`, no `$defs`/`definitions`, and no boolean
  `true`/`false` subschema, fixing llama.cpp-backed clients (Crush, OpenCode)
  that turn a tool's schema into a grammar and fail the whole request if it
  doesn't convert. `$ref`s are inlined recursively; a ref back to a type
  already being expanded (`update_schema`'s recursive `RawFieldDef.fields`)
  is cut to a plain object one level in rather than expanding forever, so
  nested `update_schema` `fields` entries advertise as plain objects instead
  of their named property list — the server still deserializes and validates
  them exactly as before. No config change, no reindex.
- **Breaking (config key rename):** `rate_limit.per_second` is now
  `rate_limit.requests_per_second`, and it finally means what it says. The old key
  was passed straight to `tower_governor`'s `GovernorConfigBuilder::per_second`,
  which takes *the interval between replenished tokens, in seconds* — not a rate. A
  config reading `per_second: 25` therefore admitted **one request every 25 seconds**
  (0.04 req/s) rather than 25 req/s, and the shipped default of `20` was one request
  every 20 seconds. The server now inverts the configured rate into a period
  (`RateLimitConfig::replenish_period`) before handing it to the builder, so the
  default is a genuine 20 req/s per IP. The old key still parses as an alias for the
  new one, so an image rollout that lands ahead of its config edit keeps booting;
  note that on upgrade this makes an existing limit dramatically *more* permissive —
  which was always the documented intent — so re-check the value if you were relying
  on the accidental behaviour. Minimum is 1 req/s.
- `POST /admin/reload` now warns explicitly about the class of settings it cannot
  apply live; the reindex worker is supervised and restarted rather than silently
  dying; `/mcp` request bodies are size-bounded; passive detection was added for a
  Qdrant collection wiped out from under a surviving `state.db` (`kb_qdrant_points_deficit`
  in `/status`/`/metrics`) (fix #154, fix #163, fix #205).
- Config provenance drift, a reranking `candidate_limit` bound, `min_score`
  documentation, and a schema-fingerprint collision were fixed together (fix #137,
  fix #144, fix #149, fix #152).

### Indexing and retrieval correctness

- Strict-mode indexing no longer drops an entire batch when one rejection occurs, and
  embedding-dimension mismatches are now caught rather than silently corrupting the
  collection (fix #156, fix #159).
- Git-layer correctness fixes: an orphaned commit sha, a stale content-hash race, and
  mishandling of non-ASCII paths in diff output (fix #140, fix #142, fix #143).
- MCP `search` filter, casualty-reporting, and path-length surface cleanup (fix #148,
  fix #151, fix #153).
- Wiki-style `[[link]]` rendering fixed in the web UI graph, and `/api/graph` node
  count capped (fix #170, fix #174).
- `validation.lint_command` is now bounded by a configurable timeout, so a hanging
  external linter can no longer stall the whole indexing pipeline (fix #146).

### Accessibility and UI

- Document navigation links in the web UI now carry real `href`s, restoring keyboard
  and assistive-technology navigation (fix #165).

### Earlier structural changes worth knowing about

These predate the `[Unreleased]` window above but are still the kind of thing an
operator upgrading an old deployment needs to know happened at some point:

- **Hybrid sparse+dense retrieval with RRF fusion** (#59) — added a `sparse` named
  vector alongside `dense`; an existing pre-hybrid collection (single unnamed vector)
  needs a one-time `md-kb-rag index --full` to migrate to the named-vector schema.
- **Relative-path indexing** (#58, part of a broader shared-retrieval-core refactor)
  — internal state moved from absolute to KB-relative paths.
- **Directory-cascading `.kb-schema.yaml` support** — every indexed file gained a
  `schema_hash` fingerprint column (added automatically via a guarded
  `ALTER TABLE ... ADD COLUMN`, no manual migration step); the first index run after
  upgrading to a version with schema support revalidates every file once to backfill
  it, and is also the run where every document's `domain` is (re)computed from its
  folder rather than trusted from frontmatter. See [Backward compatibility and
  upgrade note](deploy/USAGE.md#backward-compatibility-and-upgrade-note) in
  `deploy/USAGE.md` for the full detail.
