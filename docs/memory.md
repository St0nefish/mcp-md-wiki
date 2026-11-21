# Memory mode

An optional way to run this server as long-term memory for agents (Hermes, Open WebUI,
Claude Code and anything else that speaks MCP), next to the knowledge base it already
serves. It is three independent pieces, each usable alone:

1. **Per-caller access control** (`mcp.policy`, `mcp.tokens`, `mcp.principals`): which
   tools a caller may use and which directories it may read and write.
2. **The memory tool group** (`memory.enabled`): six tools that save, find, replace and
   recall short statements, stored as ordinary markdown documents.
3. **Client plugins** in `clients/`: a Hermes memory provider and an Open WebUI filter.

Nothing changes for a deployment that sets none of them: same seven tools, same
credentials, same behaviour.

## 1. Access control

Define policies, then map callers to them.

```yaml
mcp:
  policy:
    memory-agent:
      tools: [memory_search, memory_get, memory_recall, memory_create, memory_update, memory_forget]
      read: [memory]
      write: [memory]
    ops:
      tools: ["*"]
      read: [/]
      write: [/]
      admin: true
  tokens:
    - name: hermes
      secret_env: MCP_TOKEN_HERMES
      policy: memory-agent
```

```bash
mcp-md-wiki token mint --name hermes --policy memory-agent --out /run/secrets/hermes
```

`token mint` writes a random secret to a new file (mode 0600; it refuses to overwrite
and never prints the secret) and prints the `mcp.tokens` entry to add. Give the secret
to the server through `MCP_TOKEN_HERMES` or `MCP_TOKEN_HERMES_FILE`, and to the client as
its bearer token.

Rules worth knowing:

- **Default deny once any policy exists.** A credential no entry maps to can do nothing
  (an unmapped OAuth token gets `403`). The single `bearer_token_env` token (the
  principal `default`) stays full access and counts as admin, so an existing deployment
  keeps working while it migrates; remove it when it has.
- **Paths are directory prefixes, not globs.** `memory` covers `memory/…` and never
  `memory-old/…`; `/` is everything. A `write` prefix must lie inside a `read` prefix
  (config load refuses a policy that writes where it cannot read).
- **Out of reach looks like absent.** Search, enumeration, `get_document`, links,
  `changes_since`, history, `get_schema`, the web UI and the server instructions only show
  what the caller can read. Write tools check every path they touch, including a move's
  destination and archive targets. Fuzzy path resolution is refused for a caller that
  writes.
- **`/status*`, `/metrics` and `/admin/*`** need a policy with `admin: true`.
- Config load refuses a policy together with `ui.api_auth: false` or
  `allow_unauthenticated`, since the web routes would have no caller to scope.
- Live reload covers `mcp.policy` and `mcp.principals`; `mcp.tokens` needs a restart.
- OAuth callers are mapped with `mcp.principals` (first match wins, on `scope`,
  `client_id` or a `claim`). `mcp:write` is still required for the write tools.

Known leaks, accepted and recorded: operator-written extension text
(`mcp.extensions_path`) is shown to every caller, so keep it free of restricted
names; and a commit subject in `get_document` history can name a path the caller
cannot read when one commit touched several.

## 2. The memory tools

Turn it on:

```yaml
memory:
  enabled: true        # needs a restart; off by default
  root: memory
  default_area: general
  default_status: proposed
  duplicate_threshold: 0.85
```

A memory is a markdown document at `memory/<area>/<slug>-<id>.md`:

```markdown
---
id: mem-3fa9c2d17b04
title: Prefers tabs
type: preference
status: proposed
tags: [editor]
source_client: hermes
created: 2026-11-21T09:30:00Z
description: The user prefers tabs over spaces.
---

The user prefers tabs over spaces.
```

The schema for that area is compiled into the server, so nothing is configured in the
knowledge base, `update_schema` refuses the area, and a `.schema.yaml` under `memory/`
is rejected. Because a memory is an ordinary document it is indexed, searchable with
`search`, versioned in git, and covered by access control, lint and webhooks like any
other. The tool parameters and results are specified in
[memory-contract.md](memory-contract.md).

How it behaves:

- **New memories are `proposed`.** `memory_recall` uses only `confirmed` ones, so
  nothing an agent saves reaches a model until a person promotes it by setting
  `status: confirmed` in the document (an edit in the repository, or `write_document`
  with a policy that allows it). Set `default_status: confirmed` to skip review.
  There is no review screen in the web UI yet.
- **Duplicates are refused by default.** `memory_create` compares the new text with live
  memories the caller can read and, at or above `duplicate_threshold`, saves nothing and
  returns the matches; the agent then updates one, passes `on_duplicate: supersede`
  (replaces the closest in one commit) or `keep_both`. The 0.85 default is a starting
  point carried over from the document dedup gate, **not calibrated** against memory
  text; tune it on your own memories.
- **Editing re-opens review.** Changing a title or body puts the status back to
  `default_status`.
- **Forgetting is soft.** `memory_forget` sets `status: forgotten`; the document and its
  history stay.
- **Recall is deterministic and bounded** by `max_items` and `max_chars`.

The contradiction job (`contradictions.*`) works on any directory; point it at the
memory area to find memories that disagree.

## 3. Clients

Both talk to `/mcp` with a bearer token from the environment, use a short timeout and
fail open: when the server is down, slow or refuses, the agent runs without memory.

- [`clients/hermes/`](../clients/hermes) — a memory provider (`prefetch` for recall,
  the memory tools for the model). Writes are offered and allowed only for the primary
  agent context. Environment: `MCP_MD_WIKI_URL`, `MCP_MD_WIKI_TOKEN`,
  `MCP_MD_WIKI_TIMEOUT`.
- [`clients/open-webui/`](../clients/open-webui) — a Filter that prepends the recall block
  as a system message. The token comes from `MCP_MD_WIKI_TOKEN` in Open WebUI's
  environment, not from a Valve. `deploy.sh` creates or updates the function through
  the admin API (never the bulk sync endpoint) and leaves enabling it to you.

Run their tests with `.github/scripts/clients-test.sh` (Python standard library only).

**Not verified.** The tests use stubs, so these were never run against the real
products: Hermes' `MemoryProvider` interface and plugin install path, Open WebUI's
function loader, `inlet` behaviour and admin endpoints. Treat both clients as a
first draft to try, not a tested integration, and expect to adjust method names.
CI for `clients/` (a job in `ci-fast.yml`) is also not wired yet.
