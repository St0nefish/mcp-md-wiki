# Memory contract (version 1)

What a client may rely on when it uses the memory tools. The server reports the
version as `memory_contract_version` in `get_schema` results and under
`capabilities.experimental["mcp-md-wiki/memory"].contract_version` in the
`initialize` response; a client should check it equals the version it was written
for and stay out of the way (no recall, no writes) when it does not. The number
changes only when a parameter or result field a client uses changes meaning. The
plugins in `clients/` do not perform this check yet (they fail open on an unexpected
result instead), so bump the version only together with an update to them.

The tools exist only while `memory.enabled` is on. Every result is one JSON object
(the MCP `structuredContent`); errors are plain MCP errors with a short message.
Calls go to the ordinary `/mcp` endpoint with an ordinary bearer token; there is no
session to open first.

## A memory

| Field | Meaning |
|---|---|
| `id` | `mem-` plus twelve lowercase hex characters. Stable for the life of the memory. |
| `path` | Where the document lives (`<memory.root>/<area>/<slug>-<id>.md`). Informational: address a memory by `id`. |
| `title` | One-line headline, up to 200 characters. |
| `type` | `preference`, `fact`, `decision`, `finding` or `procedure`. |
| `status` | `proposed` (not yet confirmed by a person), `confirmed`, `superseded` (replaced by a newer memory) or `forgotten`. `proposed` and `confirmed` are *live*. |
| `tags` | Up to 16 labels of up to 64 characters. |
| `created`, `updated` | RFC 3339 UTC timestamps. |
| `source_client` | The caller that wrote it (the token or principal name). Set by the server, never by the caller. |
| `source_url` | Where it came from, when the caller said so. |
| `supersedes` | `id` of the memory this one replaced. |

The body is up to 4000 bytes of markdown. Fields the server sets (`id`, `status`,
`created`, `updated`, `source_client`) cannot be supplied by a caller.

## Tools

### `memory_create`

`title`, `body`, `type` (all required); `tags`, `area` (a folder name; default
`memory.default_area`), `source_url`, `supersedes` (`id` of a live memory this one
replaces), `on_duplicate` (`reject` default, `supersede`, `keep_both`).

The new memory starts in `memory.default_status`. Before saving, the server looks
for a live memory the caller can read whose text is within `memory.duplicate_threshold`
of the new one. With `on_duplicate: reject` nothing is saved and the result says so:

```json
{"created": false, "reason": "duplicate", "duplicates": [{"id": "mem-…", "path": "…", "title": "…", "type": "fact", "status": "confirmed", "score": 0.93}], "hint": "…"}
```

Otherwise `{"created": true, "id": "mem-…", "path": "…", "status": "proposed"}`, plus
`"superseded": "<old id>"` when it replaced one. A replacement is one commit: the new
memory is written and the old one's status becomes `superseded` together. A memory
that is already `superseded` or `forgotten` cannot be superseded again. If the
duplicate check cannot run (the index or embedder is down) the call fails with an
error rather than saving unchecked; `on_duplicate: keep_both` skips the check.

### `memory_search`

`query` (required); `type` (list), `tags` (all must match), `status` (list; default
`proposed` and `confirmed`), `area`, `limit` (default 10, at most 50). Returns
`{"returned": n, "memories": [...]}`, best first, each row the fields above (without
the body) plus `snippet` and `score`.

### `memory_get`

`id`. Returns the memory's fields, `body` and `version`, plus `supersedes_chain`
(older memories it replaced, newest first, at most 10) and `superseded_by` when a
newer one replaced it. An `id` that does not exist, is not a memory id, or lies where
the caller may not read is the same error: `no memory with that id`.

### `memory_update`

`id`, and at least one of `title`, `body`, `tags` (replaces; `[]` clears),
`source_url` (`""` clears). Returns `{"id", "path", "changed", "status"}`. Changing
the title or body puts the status back to `memory.default_status`, so an edit is
reviewed again; tags and source do not. A call that changes nothing writes nothing
(`changed: false`). A `superseded` or `forgotten` memory cannot be updated.

### `memory_forget`

`id`. Sets `status` to `forgotten` (the document stays in history and can be read by
`memory_get`; recall and the default search no longer return it). Idempotent:
`{"id", "status": "forgotten", "changed"}`.

### `memory_recall`

`query` (required); `max_items` (default 8, at most 20), `max_chars` (default 1500,
200 to 8000), `area`, `include_proposed` (default false). Returns

```json
{"context": "Relevant memories:\n- [preference] Prefers tabs: …", "items": [{"id": "…", "type": "…", "title": "…"}], "truncated": false}
```

`context` is ready to put in front of a model; it is empty when nothing matched.
Only `confirmed` memories are used unless `include_proposed` is set. The same query
over the same memories returns the same block (ordering is by score, then path), and
`context` never exceeds `max_chars`.

## Access

A caller needs a policy that lists the tool and reaches `memory.root`: `read` for
`memory_search`, `memory_get` and `memory_recall`; `write` as well for
`memory_create`, `memory_update` and `memory_forget`, which also need the `mcp:write`
scope under OAuth. A write outside the caller's `write` prefixes is refused with
`not permitted`, before anything about the target is looked up.

## Schema

The memory area's schema is part of the server. `update_schema` refuses the area, and a
`.schema.yaml` file under `memory.root` makes the knowledge base's schema invalid until
removed. A client never writes frontmatter itself.
