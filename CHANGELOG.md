# Changelog

Releases are cut by the owner running `.github/workflows/release.yml`, which tags the
version in `Cargo.toml` and promotes the already-tested image.
The container image `ghcr.io/st0nefish/mcp-md-wiki` has three kinds of tag: `:latest`
is the most recent release, `:vX.Y.Z` pins one release, and `:dev` follows `master`.
This file tracks notable, operator-relevant changes: unreleased ones under
`[Unreleased]`, which a release moves under its own version heading. `fix #N`
references are GitHub issues; see the repo's closed-issues list for the complete
history. 0.1.0 is the first tagged release and collects everything before it; its
entries are a representative sample rather than an exhaustive list — that period
included an automated multi-agent documentation and correctness audit that closed
roughly forty issues across security hardening, indexing correctness, and doc drift.

## [Unreleased]

### Smaller up-front context; vocabularies on demand

- **Smaller descriptions and schemas.** Every tool description now says what the tool
  does and the cross-parameter workflow only — per-parameter rules live on the
  parameter, response shapes are left to the (self-describing) response, and the
  server instructions repeat nothing a tool says. Compiled + config-derived text per
  description is budgeted at 600 characters (was 1500), leaving ~1400 for a KB
  extension. Input schemas are compacted when served (no `$schema`, no `null` arm on
  optional properties, no `"default": null`, no non-standard `format`), batch entries
  no longer repeat their top-level twins' descriptions, `update_schema`'s field `type`
  is a flat enum, and `update_schema`'s `operation`, `frontmatter_patch[].operation`
  and `search`'s `order_by` advertise an `enum` — the server accepts the same strings
  and aliases as before. `definition.extend` (deprecated) is still accepted but no
  longer advertised; `definition.values` now says it replaces the inherited set unless
  it includes `$values`.
- **Server instructions no longer enumerate vocabularies.** The `Available
  status/tags/type` lines, the list of folders with their own rules, the retrieval-mode
  sentence, the text-matching paragraph and the "Do NOT write a `domain` field" rule
  are gone. What remains is the base line, `Top-level areas: …` (capped at 30) and one
  pointer: "Field values and per-folder rules: get_schema with a path." (dropped when
  `get_schema` is disabled). Server instructions no longer query Qdrant facets.
- **Filters teach on error.** `search` refuses, in both modes, a filter that could only
  match nothing: one on a field no schema declares and no document carries (`unknown
  filter field '…'; filterable fields: …`, `data.filterable_fields`), or on a value
  outside the field's closed value set (`… is not an allowed value; allowed: …`, or
  `… are not allowed values; …` for several, with `data.field`/`data.allowed`), lists
  capped at 30. A list of values is refused only when none of them could match, an
  `all_of` as soon as one could not. The closed set is the union over the scopes
  `path_prefix` names (all scopes otherwise), `path_prefix` being read as `search`
  matches it: a trailing `/` is ignored, a leading one is not, so `/food` names no
  top-level scope. Any value a document actually uses is accepted, so a refusal never
  hides a match. Previously these returned an empty result (an unknown field with a
  query got a "not indexed" error).
- **`get_schema` `values_in_use`.** New optional flag: open fields gain `in_use` (the 20
  most-used values under the path, with document counts) and the result gains
  `other_fields_in_use` (fields documents there carry that no schema declares, derived
  ones such as `domain` aside).
- **Writing `domain` is a validation error.** A write whose frontmatter sets `domain`
  (derived from the top-level folder at index time) is refused with `rule: "derived"`
  and "`domain` is set by the document's top-level folder; remove it from the
  frontmatter". Indexing and the `validate` CLI still accept existing documents that
  carry one, so nothing is reindexed — but an edit or a single-document move of such a
  document now has to remove the key (a `frontmatter_patch` `remove_field` in the same
  call works). A `required` or `default` a schema gives `domain` is ignored, so it
  cannot make every write fail, and `get_schema` never lists the field, even when a
  schema declares it (to filter on it).

### Schema files are now `.schema.yaml`; the legacy name still works

- **Canonical name `.schema.yaml`.** `.kb-schema.yaml` no longer fits a server that
  hosts more than knowledge bases. The legacy name is still read everywhere the new one
  is, so the rename needs nothing on upgrade and reindexes nothing. The next
  `update_schema` in a directory on the legacy name writes `.schema.yaml` and removes
  `.kb-schema.yaml` in the same commit, so a corpus migrates one directory at a time as
  it is edited (or all at once with a `git mv`). A directory holding **both** names is
  an invalid schema file — fatal at startup, refused at runtime — until they are merged
  into `.schema.yaml`. **Downgrade:** a server from before this change reads only
  `.kb-schema.yaml`, so it silently ignores every directory already migrated to
  `.schema.yaml`; rename those files back before rolling back.
- **Schemas are named by directory in everything the model reads.** Tool descriptions,
  `get_schema`'s `declared_in` (and the web UI's `/api/schema`), validation errors'
  `[declared in …]`, and `update_schema`'s results and errors name the scope directory
  (`food/recipes/`, the root as `/`) instead of a schema file path. `update_schema`'s
  result `path` is now that directory.
  Logs, `/status`'s `schema_error`, the `validate` CLI and the `kb_schema_invalid*`
  metric names are unchanged and keep real file paths.
- **A directory move names the schemas it carries by directory.** A validation
  refusal's `moved_schema_files` is now `moved_schema_dirs` (`from`/`to` directories),
  and a successful move reports the schemas it carried there too, no longer as entries
  in `moved` (which now lists documents only).
- **Fix: `add_values` in a subdirectory no longer narrows the inherited set.** When the
  subdirectory's schema had no `values` of its own for the field, `add_values` wrote a
  fresh list of only the new values, which replaced the inherited set (and forced
  `type: enum`). It now writes `[$values, ...new]` and keeps the inherited `type`
  (`enum`, as for a new field, when no ancestor gives a type or values), so
  root `[a, b]` plus a child `add_values [c]` resolves to `[a, b, c]`. A child list that
  already exists without `$values` is still appended to as-is, and a value the scope
  already inherits is not written again (a call with nothing else to add is refused as
  already permitted). Schema files written by the old behavior are not rewritten;
  re-add `$values` to them by hand if the narrowing was not intended.

### Breaking: an invalid schema file is never loaded

The "frozen scope" mechanism is gone. A schema file in the indexed tree that is
present must be valid; a root one is still optional (config.yaml's `frontmatter` block
still stands in when there is none). A file is invalid when it cannot be read, is over
256 KB, does not parse (unknown key, wrong type, bad `dedup:` block), or contradicts itself.

- **Only the indexed tree's schema files are read.** A schema file in a hidden
  directory, or in a directory whose every document `indexing.exclude` rules out (for
  example `templates/**`), is never loaded or validated, so it cannot stop startup or a
  rebuild, and a push that changes one queues no reconcile. Previously every non-hidden
  directory's schema file was read, excluded or not. `include` and `exclude_files` do not
  remove a directory from the schema tree.
- **Startup is fatal.** `serve` refuses to start while any schema file is invalid, and
  the error lists every invalid file with its reason, not just the first. Previously the
  server started and silently stopped indexing the bad file's whole subtree. **Upgrade
  note:** run `mcp-md-wiki validate` before upgrading; a schema file the old version had
  frozen will now stop the server from starting.
- **`index` and `validate` fail the same way.** `index` (incremental or `--full`) aborts
  before touching anything. `validate` prints a `SCHEMA ERRORS` list and exits non-zero
  whether or not `--strict` is set; it no longer prints a `FROZEN` list or validates
  documents against the rest of the tree.
- **At runtime a bad schema is refused, and the previous one kept.** When a push
  (webhook), a reconcile or a moved directory brings in an invalid schema file, the
  rebuild is refused: every write and read keeps using the last schema that loaded
  cleanly, nothing is indexed until the file is fixed, and the refusal is logged at error
  level on every reconcile sweep while it lasts. `/status` gains `schema_error` (`since_unix`,
  `since_at`, and `files[]` with `path` and `reason`), and `/metrics` gains the
  `kb_schema_invalid` (0/1) and `kb_schema_invalid_files` gauges. Both clear on the next
  rebuild that succeeds. `/health` stays healthy. An indexing run that hits the invalid
  file (a document written in the meantime) is dropped without retrying, logs every
  invalid file at error level, and sets `schema_error` at once rather than at the next
  sweep. The fix itself queues a full reconcile, and the first rebuild that succeeds after
  a refusal queues one too, which indexes everything written in the meantime.
- **A pushed schema change now rebuilds the schema.** The webhook used to drop a changed
  schema file with the other non-document paths, so a schema pushed straight to the
  knowledge base's git host was not picked up until the next periodic sweep. A changed
  schema file in a push, in a write's own rebase, or carried by a directory move now
  queues a full reconcile, which rebuilds the schema and re-validates the documents it
  governs.
- **Document tools refuse schema file paths.** `write_document` (create, edit and
  move), `delete_document`, batch writes and the web UI's `POST`/`DELETE /api/doc/...`
  reject a path whose file name is `.schema.yaml` or `.kb-schema.yaml`, whatever `indexing.include` says,
  and point at `update_schema` (an invalid params error over MCP, a 400 over HTTP).
  Moving a directory that contains schema files still carries them along, but is refused
  while one of them is invalid on disk (see below).
- **`update_schema` refuses a file over 256 KB** — the same cap a rebuild enforces — and,
  if another schema file is invalid when it rebuilds, says its own change is saved but
  not in effect yet.
- **Response shapes:** `get_schema`'s `structured_content` and the web UI's
  `/api/schema/...` drop `frozen` and `frozen_reason`, and `get_schema`'s text loses its
  "this scope is frozen" warning. The write error for a frozen scope is gone, and so is
  directory moves' "frozen" error. A directory move whose source carries a schema file
  that is invalid on disk (while the server is still using that directory's last good
  rules) is refused with an invalid params error naming the directory and the reason, and
  `data.invalid_schema_dir`/`data.reason`; fix or revert the file in git, then move.
  That includes a schema file the move would bring into the schema tree out of a hidden
  or excluded directory, which no rebuild ever read: a file is checked when it is in the
  tree at its source or at its destination. `/status`'s per-run counters drop
  `frozen_by_broken_schema` and `broken_schemas`, and so do the matching
  `kb_index_last_run_files{outcome=...}` series in `/metrics`.

### Breaking: tool results are one compact JSON object

**Breaking response-shape change** for every MCP tool result. Claude Code shows the
model only `structured_content`, so a fact carried only by the prose text block never
reached it — notably `update_schema`'s "saved but not in effect yet" warning. Each result
is now a single JSON object: `structured_content`, with the text block its compact
serialization (the MCP spec's fallback for text-only clients). The prose renderings are
gone. Keys that would be `false`, `null`, empty or a default are omitted throughout.
Errors are unchanged (plain messages). The web UI's HTTP API keeps its own shapes except
where noted.

- **Writes** return `path` and `action` (`created`/`updated`/`moved` with `from`, or
  `deleted`). `diff` comes only with an edit or move — never a create or delete — and
  `diff_truncated`/`diff_total_bytes` only when it was cut. Batch entries carry `action`
  instead of `is_create`. Empty `rewritten_paths`/`referencing_paths` are omitted.
- **`update_schema`**: a change saved but not in effect yet carries `warning`. `dry_run:
  false` is gone; `invalidated`/`would_invalidate` appear only when non-empty and
  `casualties_total`/`casualties_truncated` only when capped. A dry run returns `field`
  and `definition` (the field as the scope would resolve it) instead of the whole
  `yaml`. It now declares `destructiveHint: true`, `idempotentHint: false`, and reads the
  inherited definition from its parent scopes under the write lock rather than from the
  cache before it.
- **`get_schema`**: `fields` is an object keyed by field name; each definition carries
  only `type`/`values` when set, `required`/`indexed` when true, `default` when set,
  `open: false` on a closed object, and `declared_in`. The config-derived root is `/`,
  never `config.yaml` — in `declared_in`, the web UI's `/api/schema`, and validation
  errors' `[declared in …]`. `path` is `/` for the root; `omitted_fields` only when > 0.
- **`search`**: chunk snippets start at the chunk's own text — the heading breadcrumb and
  description prepended for the embedding no longer fill half of every snippet — and a
  cut one ends in `…` (the web UI's `/api/search` and the `search` CLI command start at
  the chunk's own text too). Chunk rows drop
  `chunk_index`, `domain`, `text_truncated`, and unset per-arm scores; `explain` adds
  `mode` to the response. Document rows drop `title`/`description`/`domain` from
  `frontmatter` (`domain` stays when named in `fields`) and `mtime` unless enumerating
  ordered by it. Enumeration drops `offset` and `has_more: false`; section rows drop
  `scope: "section"` and an empty `heading_path`. Every score is rounded to four
  significant digits.
- **`get_document`**: a whole read omits `start_line`/`end_line`/`partial`; `outline_only`,
  `oversized`, `truncated`, `partial` appear only when true, `total_entries` only on a
  capped outline, `intro` only when there is one, and outline/section entries drop
  `heading` (the last element of `heading_path`) — the same in `/api/doc` (the web UI
  reads only `content`, `truncated` and `version`, so it is unaffected). The link graph
  comes only with a whole-document read (including one the size cap cuts short or turns
  into an outline) or the new `links: true`, as path lists — `links_out`, `broken_links`,
  `links_in`, `similar` (`{path, score}`) — capped at 20 per direction (was 100), with
  `links_out_total`/`links_in_total` (every kind of link counted) when cut. `history` is
  now `{changes: [{date, author, subject, operation?}], truncated?}` or
  `{available: false}`: no revision id, email or provenance trailer (the web UI's
  `/api/history` is unchanged).

**Reindex:** chunk payloads gain `text_body_offset`, so the chunker version is bumped and
the next reconcile re-chunks and re-embeds every document once, automatically. Until a
document is reached, its snippets drop only a leading description.

### Tool results no longer expose git

**Breaking response-shape change** for MCP clients that read the write tools' results.
How the server versions the knowledge base is an internal detail, so nothing about it
reaches an MCP tool caller any more — not in results, errors or descriptions. Logs,
`/status`, `/metrics` and the reindex worker are unchanged, and so is the web UI's HTTP
API (`/api/doc/...` still returns `outcome`/`sha`; its version fields are renamed — see
below).

- **Success results** of `write_document` (single, batch and directory move),
  `delete_document` and `update_schema` drop `outcome`, `sha`, `rebased_paths` and
  `sync_failure_cause` from `structured_content`, and the text drops `(commit <sha>)` and
  the "committed locally, but the push … failed" wording. A change saved locally whose
  push to the knowledge base's remote failed is reported as an ordinary success (it is
  still logged at `warn`); the next write pushes it along with its own. If it then
  conflicts with what reached the remote in the meantime, the server resets its copy to
  the remote and keeps the unpushed commits under `refs/mcp-md-wiki/unsynced/<sha>`,
  logging an error, for an operator to recover.
- Concurrent changes to *other* documents are no longer reported at all (they used to be
  listed in `rebased_paths`); they are still marked for reindexing. A concurrent change
  to the caller's own document is covered below.
- **Failures** carry no `data.outcome` and no git output. A change that could not be
  saved and was fully undone says so ("could not be saved. Nothing was changed; try
  again"); one whose undo also failed tells the caller not to retry and to report it to
  the operator. The cause, including git's own output, is logged where it happened. A
  bad server-side git credential is reported as a server misconfiguration, without the
  variable's name.
- **Wording:** the server instructions describe the corpus as "versioned" rather than
  "git-backed", tool descriptions no longer say a write "commits and pushes", and the
  batch `message` refusal and invalid-`message` errors no longer say "commit".
  `get_document`'s opt-in `history` (#257) is git-free too — see the previous section.

### Breaking: concurrent writes are safe, and replacing a document needs its version

Two writers editing the same knowledge base — two agents, an agent and the web UI, or a
write racing a push from elsewhere — could overwrite each other's changes or leave the
server's copy diverged from the remote. Every write now syncs with the remote first and
does its whole read-edit-validate-save under one lock, so writes never interleave.

- **`content_hash` is now `version`, and `expected_hash` is now `expected_version`** — in
  `get_document`'s result, `write_document`'s parameters (single and per batch entry),
  and the web UI's `GET`/`POST /api/doc/...`. `version` is still opaque and still covers
  the whole file, but its value changed: a hash a client stored from an older server
  will not match. Write results now also carry the document's new `version`.
- **`expected_version` is required** to replace an existing document with `content`, to
  move a single document, to delete one (`delete_document` and `DELETE /api/doc/...`
  take it now), and for a batch entry that replaces an existing document. Without it the
  call is refused and told to pass the `version` from `get_document`. Creating a new
  document and the relative edits (`old_string`/`new_string`, `frontmatter_patch`,
  `append`) do not need it. A create that does pass one is refused as not found — the
  document it was read from was deleted or moved: find it, or omit `expected_version`
  to create a new document.
- **Relative edits apply to the current document**, so a concurrent change elsewhere in it
  is kept; when the edit passed an `expected_version` that is no longer current, the
  result says so (`merged_with_other_changes: true`). An `old_string` someone else's
  change removed is refused with "the document was edited by someone else; re-read it
  and try again". `update_schema` operations work the same way: two concurrent
  operations on one schema both land.
- **A stale full replace is merged, not refused or clobbered**: when `expected_version`
  is older than the document, the changes made since are three-way merged with the
  caller's. A clean merge is re-validated and saved, and the result says so
  (`merged_with_other_changes: true`; the tool description says to re-read);
  overlapping changes are refused — as is every stale full replace in a repository that
  converts content on checkout (`.gitattributes` `eol` or filters, `core.autocrlf`) or
  stores SHA-256 object ids, where the old version cannot be found to merge against. A
  stale move or delete is refused. A directory move is refused if anything under it
  changed after the server checked it.
- A push that loses a race, or a rebase that conflicts with what reached the remote in
  the meantime, drops that write's own commit and runs the write again against the fresh
  remote state, up to three attempts; then it is refused as edited by someone else, with
  none of its own commits left behind to diverge from the remote. Changes an earlier
  outage left unpushed are kept and go out with the next write that gets through. A push
  that fails because the remote is unreachable still keeps the local save and reports
  success.
- `write_document` and `delete_document` now declare `destructiveHint: true`,
  `idempotentHint: false`; `search`, `get_document` and `get_schema` declare
  `readOnlyHint: true`.
- Web UI: the editor sends the version it loaded on save and move, and the document view
  sends the version it rendered on delete, so a document changed since it was opened is
  never deleted unseen (the view refreshes and says so). A stale save's or delete's 409
  body carries `edited_elsewhere: true` (replacing the old `expected_hash`/`actual_hash`
  pair).

No config change is needed. (The first reconcile after upgrading does re-chunk every
document once, for the reason given under "Breaking: tool results are one compact JSON
object".)

### MCP tools

- **Near-duplicate detection can be overridden per directory** (closes #272).
  `write.dedup_enabled` and `write.dedup_threshold` were global only, so structurally
  templated folders (meal plans, recipes) tripped the check on every create. A
  schema file may now carry `dedup: {enabled, threshold}`; each key cascades
  independently, nearest scope wins, and a key no schema sets falls back to the global
  `write.*` value. Hand-edited only (`update_schema` has no operation for it but preserves
  an existing block); `get_schema` reports the effective override when set. A threshold
  outside 0.0–1.0, a wrong-typed value or an unknown key makes the file invalid like any
  other schema error (see above). The block is not part of the schema fingerprint, so
  editing it revalidates and reindexes nothing. No config change needed.

### Fixed

Bugs in behavior that shipped in 0.1.3 or earlier.

- **`delete_document`'s `path` description no longer promises a basename.** It said a
  unique basename would resolve, as it does in `get_document`; it never did (a bare
  basename is "document does not exist"), so the description now reads "Path relative to
  the KB root."
- **A near-duplicate refusal reports clean numbers.** `similarity` and `threshold` in its
  error data carried float noise (`0.9300000071525574`); both are rounded to four
  significant digits, over MCP and the web UI alike.
- **An edit that changes nothing succeeds.** Saving unchanged content, or applying a
  `frontmatter_patch` that is already in effect, failed with "could not be saved … try
  again" over MCP (a 500 from the web UI) because there was nothing to commit. It now
  returns the document's current `version` with no `diff` and no commit; a batch whose
  entries all change nothing does the same.
- **A write that fails partway leaves no half-written document.** A `write_document`
  create or edit that hit a disk or I/O error left the file truncated and uncommitted in
  the server's copy; the previous content is restored (a partly written new file is
  removed). The same for the failing entry of a batch, which was left truncated while
  the entries before it were rolled back; its error names any document the undo could
  not restore.
- **The web UI's saves and deletes finish when the browser disconnects.** `POST` and
  `DELETE /api/doc/...` ran inside the request, so a client that went away mid-write
  could leave an uncommitted change behind; the write now runs to completion on its own.
- **A directory move no longer panics on `./notes`, `.` or `/`.** Such a source crashed
  the request instead of answering. A `.` segment and doubled slashes are dropped from
  both directories (`./notes/` is `notes`), including from the moved paths, and the
  knowledge-base root itself is refused as a source.
- **`get_schema` and `update_schema` accept `.` as the root path** (it was refused as an
  invalid path) and read `food/./recipes` and `food//recipes` as `food/recipes` (they
  were carried into scope names and commit messages as written).
- **`update_schema` no longer follows a symlinked schema file.** The schema walk already
  ignored one, but `update_schema` read through it: a link to another file on the server
  could surface that file's text in a parse error, and a link to a device read without
  bound. A symlink now counts as no schema file there too, and a schema file over 256 KB
  is refused before it is read.
- **A document already over the 512 KB write limit can be shrunk or moved over MCP.** An
  edit that shrinks it, and a move, were refused as too large; an edit is now refused
  only when it would grow the document past the limit.

## [0.1.3] - 2026-10-06

### MCP tools

- **`get_document` can return a document's git history** (closes #257). The new `history`
  parameter (commit count, clamped 1–100) adds `structured_content.history` and a text
  summary: each commit's `sha`, author, `timestamp`, `subject` and the `Tool:`/`Operation:`
  provenance trailers (`tool`, `operation`, `tool_authored`), so an agent can tell tool-made
  edits from hand-made ones. This was previously reachable only through the web UI's
  `/api/history`, which now shares the same response builder (its JSON is unchanged).
  Still six tools; no config or reindex impact.
- **`search`'s `query` description no longer calls retrieval semantic-only** (fix #309).
  The schema said "Semantic query." and the tool description "ranked by semantic
  relevance", although the default (`search.hybrid: true`) also matches literal terms. The
  `query` property now says so when `search.hybrid` is on, and still says "Semantic query."
  when it is off; the tool description just says "ranked by relevance". No reindex needed.
- **`write_document`'s `frontmatter_patch` keeps untouched formatting** (fix #269). A patch
  used to re-serialize the whole frontmatter block, alphabetizing keys, flattening `>-`
  block scalars to one long line, dropping comments and rewriting `[a, b]` lists. It now
  edits the document's own block in place: fields the patch does not change are left
  byte-for-byte as written, a changed field is re-rendered where it sits, and a new field
  is appended. A dot-path edit re-renders its whole top-level field, and comments on a
  changed field are dropped. A block that cannot be edited safely (top-level flow style,
  anchors, duplicate keys) still gets the old full re-render. No reindex is needed.
- **`update_schema` edits nested fields in place** (fix #268). `field` is a dot-path, but it
  was used as a literal top-level key, so editing `planning.method` created a sibling
  `planning.method` key beside the nested `planning` → `method` declaration. Both flattened
  to the same path and the winner varied per process, which could silently turn a closed
  enum into an open one. All four operations now resolve the dot-path through nested
  `fields:` and flat keys alike; `add_values`/`set_field` create missing parents as
  `type: object` fields. A `field` with an empty segment (`a..b`) or more than 16
  segments is refused.
- **Upgrade note:** a `.kb-schema.yaml` that declares one path twice (nested `fields:` plus
  a flat dot-path key) is now rejected as an invalid schema file, which stops the server
  from starting until the duplicate is removed. Previously one declaration was silently
  dropped. A schema already damaged by the old bug will show up this way.
- **The MCP handshake names this server, not `rmcp`** (fix #277). `initialize` reported
  `serverInfo` as `{"name":"rmcp","version":"1.8.0"}` because `Implementation::from_build_env()`
  reads rmcp's own build environment. It now reports `mcp-md-wiki` and this crate's version.
- **Server instructions no longer list `title`/`description` values** (fix #333). Both are
  free text, so their `Available title: ...` line was one value per document and could push
  the instructions past Claude Code's 2048-character cap by itself. Other fields, including
  declared `values:`, are unchanged. `title` and `description` stay filterable.

### Operations

- **`/status` and `/metrics` report the running build** (fix #264). `/status` gains
  `revision` and `kb_build_info` gains a `revision` label (`mcp-md-wiki status` prints it
  too), so "is the release I just published the one that's running" is one `curl`. The value
  is the image's `REVISION` build arg, i.e. the git **tree** hash the train tested, and equals
  the `org.opencontainers.image.revision` label (it maps to the `:tree-<hash>` image). A plain
  `cargo build` reports `unknown`. Dashboards keyed on the exact `kb_build_info{version=...}`
  label set will see the extra label.
- **`/api/history?commit=` returns 400 for an unknown revision** (fix #265). A nonexistent,
  malformed or non-commit revision used to be a 500 `failed to read commit diff` that looked
  like a server fault. It is now a 400 `unknown revision '<rev>'` logged at `warn`; a real git
  failure is still a 500 logged at `error`. The revision is also resolved with
  `--end-of-options`, so a leading-dash value can no longer be read as a git option.

### Indexing and retrieval correctness

- **Newly excluded files are purged from the index** (fix #266). A file indexed before it
  matched `indexing.exclude`/`exclude_files` (or stopped matching `include`) kept its
  `indexed_files`, `documents`, `document_fields` and link rows and its Qdrant chunks
  forever, so it stayed searchable and counted in `/api/graph` and enumeration totals. The
  next reconcile (a restart, `/admin/reload`, or the periodic sweep) now purges it like a
  deleted file; the file on disk is untouched. **Upgrade note:** an instance with such
  stale rows shrinks on its first reconcile after upgrading, and narrowing `include` purges
  everything it no longer matches. The purge fails closed: an unbuildable path filter, an
  `include` with no valid pattern, a walk that finds no indexable file at all, or one that
  would drop more than half of the indexed files while they are still on disk (an unmounted
  subtree, an `include` typo) purges nothing by exclusion and logs an error; files really
  gone from disk are still purged. To exclude a majority of the corpus on purpose, apply it in
  steps or run `index --full`. Excluded paths are reported under `orphans_removed`.

### Security

- **The rate limiter no longer trusts a caller-supplied `X-Forwarded-For`** (fix #275).
  It keyed on the *leftmost* forwarded address, which each proxy hop preserves, so a caller
  could rotate the header for a fresh bucket per request (no volumetric protection in front
  of the bearer compare, JWT verification and JWKS refetch) or forge a victim's address to
  drain theirs. The new `rate_limit.client_ip_source` picks the key: `peer` (the default,
  headers ignored), `cf_connecting_ip` (`CF-Connecting-IP`, for a Cloudflare-fronted
  deployment) or `x_forwarded_for_rightmost`; each falls back to the socket peer.
  **Upgrade note:** the default changed from forwarded-header keying to `peer`, so behind a
  reverse proxy all callers now share one bucket until you set `cf_connecting_ip` or
  `x_forwarded_for_rightmost`. `X-Real-IP` and `Forwarded` are no longer read. Restart
  required; no reindex.

## [0.1.2] - 2026-10-05

### Deployment

- **Docker secrets: `<NAME>_FILE` for the secret env vars** (fix #332).
  `GIT_PULL_TOKEN`, `WEBHOOK_SECRET`, `MCP_BEARER_TOKEN`, `EMBEDDING_API_KEY` and
  `RERANKING_API_KEY` (or whatever `source.git_token_env` / `webhook.secret_env` /
  `mcp.bearer_token_env` / `embedding.api_key_env` / `reranking.api_key_env` name) can now
  be supplied as `<NAME>_FILE=/run/secrets/...`; the file's contents are trimmed. Setting
  both forms (a blank plain variable yields to the file), or pointing `_FILE` at an empty
  or unreadable file, fails startup (and a `/admin/reload`). Upgrade: nothing to do —
  a value in the plain variable is used exactly as before. Built on
  `oauth-resource-server` 0.4 (`env::config_value_from_env`), which this release raises
  from 0.1.

- **Three new `mcp.oauth` keys, from `oauth-resource-server` 0.4:** `max_token_age_secs`
  (refuse a token issued longer ago, by `iat`), `allowed_client_ids` (accept only listed
  `client_id`/`azp` values — for a shared audience) and `required_claims` (claim name to
  required value). All default to off, are `restart_required` on `/admin/reload`, and are
  documented in `docs/oauth.md`. Also from the crate: a rejected `mcp.oauth.issuer` with a
  query string now echoes the query masked (`?***`) in the startup error.

- **The image is `linux/amd64` only.** The `linux/arm64` image and the
  `mcp-md-wiki-linux-arm64` release binary are no longer built. Upgrade: nothing to do
  on x86 hosts; an arm64 host pulling `:latest` gets no matching platform from the next
  release on — use `:latest-arm64` instead, which every release builds (as a separate
  run, after the amd64 image ships); `gh workflow run arm64-image.yml` rebuilds it on
  demand (`-f sha=<commit>` for any master commit).

## [0.1.1] - 2026-10-02

### Deployment

- **PRs merge through a merge train, and releases are started by hand.** Each armed
  PR is squashed onto `master` and tested once, on exactly the tree that lands; the
  image is built from that tree before the merge and tagged `:tree-<tree hash>`, and
  the merge commit then gets `:sha-<commit>`, `:<x.y.z>-dev.<n>` and `:dev` on the
  same digest without a rebuild. Docs-only merges no longer produce an image, so
  `:dev` stays on the last code change. A release is now `gh workflow run
  release.yml` rather than publishing a GitHub release: it tags `v<Cargo.toml
  version>`, attaches `mcp-md-wiki-linux-amd64`/`-arm64` binaries copied out of the
  image, retags `:latest`/`:vX.Y.Z`, and opens the next patch-version PR. Image
  labels change: `org.opencontainers.image.revision` is now the source **tree**
  hash (not a commit), and `org.opencontainers.image.version` is the plain
  `Cargo.toml` version.

## [0.1.0] - 2026-09-29

### Deployment

- **`:latest` now moves only on a GitHub release, not on every merge to `master`.**
  A merge builds one multi-arch image (`linux/amd64` + `linux/arm64`) and tags it
  `:sha-<commit>`, `:<x.y.z>-dev.<n>` and, if it is still master's tip, `:dev`. A
  release retags that same image as `:latest` and `:vX.Y.Z` (no rebuild) and
  triggers Watchtower. The arm64 image is now built natively per master commit
  into the same multi-arch tag; the separate nightly build and its single-platform
  `:latest-arm64` tag are gone. Upgrade: compose files that pull `:latest` keep
  working but now track releases, so they will not update until the first release
  is published; pull `:dev` instead to keep tracking `master`.

### MCP tools

- **`mcp.disabled_tools`**: a new config key (default `[]`) disables MCP tools
  at the server level. A disabled tool disappears from `tools/list`,
  `tools/get` returns nothing for it, and `tools/call` refuses it with the
  same `invalid_params("tool not found")` error an unknown tool name gets —
  `KbSearchServer::enabled_tool_router` builds a fresh `ToolRouter` per
  request with every configured name disabled (rmcp 1.8's
  `ToolRouter::disable_route`), so all three code paths agree. Every entry
  must be one of the six MCP tool names, with no duplicates, and the list may
  not disable all six — config load fails naming the problem otherwise.
  Server-instructions sentences that reference a disabled tool by name are
  dropped: the top-level-areas listing (which points at `search`) disappears
  when `search` is disabled, and the "call `get_schema` before writing there"
  prompt disappears when `get_schema` is disabled (the surrounding directory
  listing itself survives, since it isn't tool-specific). Live-reloadable:
  read fresh from the live config on every `tools/list`/`tools/get`/
  `tools/call`, so a `POST /admin/reload` change takes effect starting with
  the very next request — no restart, no metadata-refresh-tick wait — though
  an already-connected client that cached an earlier `tools/list` is not
  proactively notified, since this server does not advertise the
  `listChanged` tool capability. The web UI's `/api/doc/*` write routes call
  `write::write_document`/`write::delete_document` directly, not through MCP
  dispatch, so they are unaffected by this setting. No reindex.
  **`mcp.enabled_tools`**: an allowlist alternative — `Option<Vec<String>>`,
  absent by default — that resolves to the complement of `disabled_tools`:
  every tool NOT named is disabled, computed once at config-resolve time
  against the same six tool names. Setting both `enabled_tools` and a
  non-empty `disabled_tools` fails config load, as does an unknown name, a
  duplicate, or an empty `enabled_tools: []` (which would disable every
  tool). The resolved effective set is what every downstream reader
  (`enabled_tool_router`, server instructions, reload reporting) already
  reads, so an allowlist gets the exact same live-reload and instructions
  behavior described above with no separate code path. Unlike
  `disabled_tools`, a tool added to this binary in a future release stays
  disabled under an existing allowlist until the allowlist itself is updated
  to name it.
- **Tool descriptions and server instructions trimmed to fit Claude Code's
  2048-character cap.** Claude Code truncates each tool description and the
  server instructions at 2048 characters; `search` (3189), `get_document`
  (5726) and `write_document` (4977) were all being cut, and the server
  instructions went over once a KB's `server.md` extension was appended.
  Descriptions now carry what a tool does, when to use it, and the behavior a
  model would otherwise get wrong; the detailed rules moved onto the parameter
  they govern, whose schema descriptions are not truncated: `search`'s
  granularity details (now only on `granularity`), `path_prefix`'s
  substring/truncation rules and `filters`' shapes; `get_document`'s
  `heading_path` resolution tiers, ambiguity/suggestion errors and
  oversized-section behavior (on `heading_path`/`line`) and outline capping
  (on `outline`); `write_document`'s per-mode rules (on `content`,
  `old_string`, `frontmatter_patch`, `append`, `new_path`, `expected_hash`)
  and the whole batch contract (on `documents`). Tool-to-tool references that
  would dangle when a tool is disabled via `mcp.disabled_tools` were dropped,
  and with `search` disabled the server instructions also omit their
  text-matching/`path_prefix` narrative and retrieval-mode sentence.
  When every top-level area has its own `.kb-schema.yaml`, the server
  instructions' scoped-directory line now reads "Every top-level area has its
  own stricter frontmatter rules, as do: …" and lists only the deeper scopes,
  instead of repeating the whole areas list.
  Parameter descriptions no longer leak internal Rust paths or issue numbers.
  Every compiled + config-derived description, and the server instructions
  for a realistic corpus, is now held to 1500 characters by tests, leaving
  ~500 for each per-KB extension; **an extension that pushes a description
  past 2048 characters is still truncated by Claude Code**, and the server now
  logs a warning naming the tool (or the server instructions) and its length
  when that happens. No config change, no reindex.

### Authentication

- **OAuth support moved into a separate, general-purpose crate,
  [`oauth-resource-server`](https://github.com/St0nefish/oauth-resource-server)
  (mcp-md-wiki#308).** The validator, JWKS handling, RFC 9728 metadata,
  `WWW-Authenticate` challenges, and the axum auth middleware are now published
  code rather than living only in this binary, so the same reviewed
  implementation can protect other services. It is an internal move: every
  `mcp.oauth` key, default and validation message stays the same, and for a
  config that parsed before, the 401/403 statuses, headers and bodies a client
  sees are the same, except for the edge cases listed below. Three new keys:
  **`mcp.oauth.required_scopes`** (default `[]`) lists
  further required scopes, unioned with `required_scope` — a token must carry
  all of them. `required_scope` still defaults to `mcp:read` when neither key
  is set; setting `required_scopes` alone replaces that default with exactly
  the list given. **`mcp.oauth.allow_insecure_http`** (default `false`) is
  the opt-in for a plain-`http` URL described below.
  **`mcp.oauth.allow_unscoped_tokens`** (default `false`) exists because the
  crate has it and has no effect here, since this server always requires a
  scope. Auth log lines now carry the `oauth_resource_server` target
  rather than `mcp_md_wiki`. The WARN-level lines (rejection reasons, JWKS
  load/refresh failures) are visible under the default `RUST_LOG=info` as
  before; only the DEBUG-level accept/reject detail needs
  `oauth_resource_server=debug` added to `RUST_LOG`, e.g.
  `RUST_LOG=info,mcp_md_wiki=debug,oauth_resource_server=debug` — see
  `main.rs`'s `DEFAULT_LOG_FILTER` doc comment for the full reasoning.
  `oauth-resource-server` is an ordinary
  [crates.io](https://crates.io/crates/oauth-resource-server) dependency (docs at
  [docs.rs](https://docs.rs/oauth-resource-server)).
  **Upgrade action, only for a plain-`http` URL:** an `mcp.oauth.issuer`,
  `jwks_uri` or `resource` that is plain `http://` on a non-loopback host (an
  in-cluster `http://authentik-server:9000/...` `jwks_uri`, say) now fails
  startup — keys fetched over cleartext can be substituted by anyone on the
  path (RFC 8414 §2, RFC 9728 §1.2). Switch it to `https`, or set
  `mcp.oauth.allow_insecure_http: true` if that network is trusted; each such
  URL then logs a startup warning, as a plain-http `issuer` or `jwks_uri` did
  before.
  Tokens this server now refuses that it accepted before (none issued by a
  standard deployment of the tested providers):
  - a token whose header lists critical extensions (`crit`, RFC 7515
    §4.1.11 — none is supported);
  - a token whose `nbf` is present but not a non-negative number (it used to
    skip the not-before check entirely);
  - a sender-constrained token (a `cnf` claim: DPoP or mTLS-bound), which is
    no longer accepted as a plain bearer token (RFC 9449 §7.2, RFC 8705 §3);
  - a token signed by a JWKS key whose `key_ops` omits `verify`.
  Config values now refused at startup: a required or advertised scope that
  is not an RFC 6749 scope-token (a space, `"`, `\`, or a non-printable
  character), and an `issuer`, `resource` or `jwks_uri` containing a space,
  control or non-ASCII character (either could make every 401 go out without
  `WWW-Authenticate`).
  Other edge-case behavior changes, none affecting a default deployment:
  - With an explicit `mcp.oauth.scopes_supported: []`, the metadata document
    omits `scopes_supported` (it published `[]`; RFC 9728 §3.2 says to omit
    a parameter with no values), and the `invalid_token` challenge names the
    required scope (`scope="mcp:read"`) instead of sending `scope=""`. The
    startup warning that a required scope is missing from
    `scopes_supported` is no longer logged for that case, since clients are
    now told the required scope.
  - A `jwks_uri` discovered from a loopback `http://` issuer, or a redirect
    followed while fetching the metadata or keys, that lands on plain `http`
    on a non-loopback host is refused (a JWKS refresh failure) unless
    `mcp.oauth.allow_insecure_http` is set, which then logs each use as a
    warning. It used to be followed silently.
  - A `mcp.oauth.resource` whose path ends in a slash keeps it in the
    metadata URL (RFC 9728 §3.1): `https://host/mcp/` is described at
    `/.well-known/oauth-protected-resource/mcp/`, not `…/mcp`.
  - A JWKS refetch runs to completion even if the request that started it is
    dropped (a client disconnect used to spend the one-per-minute unknown-`kid`
    refetch without loading keys), and the background key refresh retries a
    failed pass after a minute, backing off to an hour, rather than waiting
    the full hour.
  - A JWKS key that declares no `alg` and could verify several algorithms is
    still used, and now logged at WARN (naming its `kid`) when it first
    appears.
  - The path-suffixed metadata route is derived from `mcp.oauth.resource`'s
    path (RFC 9728 §3.1) instead of being fixed at `/mcp`: a resource at
    `https://host/kb/mcp` is now described at
    `/.well-known/oauth-protected-resource/kb/mcp`, the URL its challenge
    already advertised, and no longer at `…/oauth-protected-resource/mcp`. The
    startup warning "mcp.oauth.resource derives a protected-resource metadata
    URL this server does not serve" is gone, since that can no longer happen.
    For the usual `https://…/mcp` resource nothing changes.
  - An explicit YAML null (`required_scope: ~`, `jwks_uri: ~`,
    `scopes_supported: ~`) now means unset, taking the default, instead of
    failing to parse.
  - With OAuth off, a non-GET request to `/.well-known/oauth-protected-resource/mcp`
    is answered 404 (it was 405); every method there is 404 when there is no
    document to serve.
  - `/admin/reload` reports a scope change under the setting
    `mcp.oauth.required_scope / mcp.oauth.required_scopes`, with the resolved
    scope list as its old/new value (e.g. `["mcp:read"]`), and renders
    `jwks_uri` values as `Some("…")`/`None` instead of a bare string.
  No reindex.
- **OAuth is now provider-agnostic, and the recommended way to authenticate.**
  Before this, the resource server read scopes only from the `scope` claim, so a
  perfectly valid Authelia access token (scopes in `scp`, a JSON array) got
  `403 insufficient_scope` with `present=[]`. Scopes are now read from every claim
  in `mcp.oauth.scope_claims` (default `["scope", "scp"]`), as either a
  space-delimited string or an array, and combined. Other additions, all under
  `mcp.oauth`:
  - `audiences`: extra accepted `aud` values, combined with `audience`. One of the
    two is required.
  - `jwks_uri` is now optional. When omitted it is discovered from the issuer
    (OIDC, then RFC 8414), and a document whose `issuer` differs is refused.
  - `algorithms`: an asymmetric allowlist, adding ES256/ES384/PS*/EdDSA (Kanidm
    signs ES256). Each key is limited to what its own type can produce, and
    `HS*`/`none` are refused at startup.
  - `leeway_secs` (default 60, max 300). `nbf` is now checked too.
  - `require_at_jwt` (default off): enforce RFC 9068's `typ`. Other explicit JWT
    types such as `dpop+jwt` are now always refused.
  - `principal_claims` (default `["preferred_username", "sub"]`) names the caller
    in logs.
  - `accept_static_bearer` (default true) runs OAuth-only when set to `false`,
    even with `MCP_BEARER_TOKEN` set.

  Signing keys load at startup and refresh hourly in the background, so an
  unreachable issuer shows up as one startup warning and a withdrawn key stops
  being trusted. Refreshes no longer hold the key lock across the network, so a
  slow IdP cannot stall requests whose key is cached. Metadata/JWKS bodies are
  capped, and https→http redirects are refused. The `Bearer` scheme is now matched
  case-insensitively. Running with only a static bearer token logs a startup
  warning recommending OAuth. The docs lead with OAuth; the new
  [`docs/oauth.md`](docs/oauth.md) covers the `mcp.oauth` keys and this
  server's setup — recipes for Authentik, Authelia and Kanidm, each marked
  with what was actually tested, plus a checklist for any other provider,
  now live in the [`oauth-resource-server`](https://github.com/St0nefish/oauth-resource-server)
  crate's `docs/providers.md` (mcp-md-wiki#308).

  **Upgrade:** no action needed. Existing `mcp.oauth` blocks and
  `MCP_BEARER_TOKEN` keep working unchanged; a regression test pins the
  production Authentik config and token shape. Config load is stricter only for
  values that could never have worked: `issuer`/`resource`/`jwks_uri` must be
  absolute http(s) URLs, and `required_scope` must be a single scope. No reindex.

### Retrieval

- **`reranking.max_document_bytes` decouples the reranker's per-document byte
  budget from `chunking.max_chunk_size`** (fix #307). Chunks get a description
  and heading breadcrumb prepended *after* chunking, so enriched chunk text
  routinely exceeded the old derived budget and was truncated before the
  cross-encoder ever saw it. The budget is now its own YAML-only key, default
  `6144` — sized to an 8192-token reranker (the documented
  `gte-reranker-modernbert-base` setup) served with `--ubatch-size` equal to
  `--ctx-size`, minus 2048 tokens reserved for the query and special tokens, at
  one token per byte worst case. It still guards #128 (the reranker rejects
  the *whole request* when one document exceeds its physical batch size), and
  is restart-required, same as before. **Upgrade note:** a reranker running a
  small `--ubatch-size` (such as llama.cpp's 512 default) previously got an
  implicit ~1500-byte budget from the default `max_chunk_size`; it must now
  set `reranking.max_document_bytes: 1500` explicitly, or it will hit #128's
  whole-request 500s. No reindex is needed.
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

- Webhook pushes, MCP/HTTP writes, and concurrent-write rebases now respect
  `indexing.include`/`exclude`/`exclude_files` the same way a full reconcile does
  (fix #278): a push, a write's own target path, or a rebase-pulled-in commit
  touching a non-indexable path (e.g. the default-excluded `README.md`) is no
  longer marked dirty, indexed, and counted as invalid — it is filtered out
  before it ever reaches the reindex queue, exactly as a reconcile would have
  ignored it. A glob-build error fails open (marks the paths unfiltered,
  logged loudly) rather than failing the webhook or the write.
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

[Unreleased]: https://github.com/St0nefish/mcp-md-wiki/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/St0nefish/mcp-md-wiki/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/St0nefish/mcp-md-wiki/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/St0nefish/mcp-md-wiki/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/St0nefish/mcp-md-wiki/releases/tag/v0.1.0
