# OAuth for mcp-md-wiki

mcp-md-wiki can act as an **OAuth 2.1 resource server**: MCP clients get an access
token from your identity provider (the *authorization server*, AS) and present it on
every request, and the server verifies it locally against the provider's published
signing keys. This is the recommended way to protect `/mcp`, `/status`, `/metrics`
and `POST /admin/reload`.

The validator, JWKS handling, RFC 9728 metadata, `WWW-Authenticate` challenges and
the auth middleware itself live in a separate, general-purpose crate,
[`oauth-resource-server`](https://github.com/St0nefish/oauth-resource-server)
(mcp-md-wiki#308) — extracted so the same tested implementation can protect other
services, not only this one. This page covers what's specific to mcp-md-wiki: the
`mcp.oauth` YAML keys and their defaults here, and how to set up this server with a
given client. For how the validator works internally, its security properties and
design rationale, and provider setup recipes, see the crate's
[README](https://github.com/St0nefish/oauth-resource-server#readme) and
[`docs/providers.md`](https://github.com/St0nefish/oauth-resource-server/blob/master/docs/providers.md).

Why use it:

- **Per user.** Every token names who it belongs to, so the logs do too.
- **Expiring and revocable.** A leaked token dies on its own, usually within an hour
  or less. A leaked static bearer token works until someone rotates it by hand.
- **Works with hosted clients.** claude.ai, Claude Desktop and the mobile apps can
  only connect to a remote MCP server through an OAuth authorization-code flow.

The static bearer token (`MCP_BEARER_TOKEN`) is still supported, and by default it
keeps working alongside OAuth. If you run with only the static token, the server
logs a startup warning recommending OAuth. Nothing about an existing deployment
changes until you edit its config.

> **Scope enforcement is coarse.** A token must carry every scope in
> `mcp.oauth.required_scope`/`required_scopes` (default: just `mcp:read`) to get in
> at all. There is no per-tool check yet, so any accepted token can also call the
> write tools (`write_document`, `delete_document`, `update_schema`). Only grant the
> scope to people you would let edit the knowledge base. The match is exact, with
> no scope hierarchy: a token carrying only `mcp:write` does not satisfy
> `mcp:read`, so grant `mcp:read` to everyone who should get in, writers
> included. (The MCP spec asks servers to account for a broader scope implying
> narrower ones; this server does not yet.)

## How it works

1. A client calls `/mcp` without a token and gets `401` with
   `WWW-Authenticate: Bearer error="invalid_token", resource_metadata="…", scope="…"`.
2. It fetches the protected-resource metadata (RFC 9728) from
   `/.well-known/oauth-protected-resource/mcp` (also served at
   `/.well-known/oauth-protected-resource`). The suffixed route is derived from
   `resource`'s path (RFC 9728 §3.1) — `/mcp` for the usual
   `https://kb.example.com/mcp`, `/kb/mcp` for a server published at
   `https://host/kb/mcp`. Both routes are unauthenticated. The
   document names your `issuer` as the authorization server, plus
   `scopes_supported`.
3. The client runs the authorization-code flow with PKCE against your AS and comes
   back with an access token.
4. mcp-md-wiki checks the token against `mcp.oauth` below, on every request.

Steps 1, 2 and 4 are all `oauth-resource-server`'s: this server just supplies the
config. For exactly what step 4 checks and in what order (JWT shape, `alg`
allowlist, signature against the JWKS, `iss`/`aud`/`exp`/`nbf`, `typ`, scope), how
the JWKS is fetched or discovered and kept fresh, and what happens when the AS is
unreachable, see the crate's
[What is checked, in order](https://github.com/St0nefish/oauth-resource-server#what-is-checked-in-order),
[Security model](https://github.com/St0nefish/oauth-resource-server#security-model) and
[Design rationale](https://github.com/St0nefish/oauth-resource-server#design-rationale).

## Configuration

Everything lives under `mcp.oauth` in `config.yaml`. Every key is restart-only
(`POST /admin/reload` reports it as `restart_required`). The full annotated block is
in [`deploy/config.example.yaml`](../deploy/config.example.yaml).

| Key | Default | Notes |
|---|---|---|
| `enabled` | `false` | Master switch. |
| `issuer` | (required) | Copy it exactly from the AS's discovery document, including any trailing slash. Must be `https` unless its host is loopback or `allow_insecure_http` is set. |
| `resource` | (required) | The public URL of this server's MCP endpoint, e.g. `https://kb.example.com/mcp`. Same `https` rule as `issuer`. A trailing slash is part of the path: `https://kb.example.com/mcp/` is described at `/.well-known/oauth-protected-resource/mcp/`. |
| `audience` | (required, unless `audiences` is set) | A value the token's `aud` must contain. See [Choosing the audience](#choosing-the-audience). |
| `audiences` | `[]` | More accepted audiences. Combined with `audience`; a match on any one is enough. |
| `jwks_uri` | unset (discover) | Set it to skip discovery. Unset, `~` and `""` all mean discover. A plain `http://` URL on a non-loopback host (anyone on that path could substitute the signing keys) fails startup unless `allow_insecure_http` is set. |
| `required_scope` | `mcp:read` | A single scope (printable ASCII, no spaces, `"` or `\`), matched exactly and case-sensitively. mcp-md-wiki defaults this to `mcp:read` when neither it nor `required_scopes` is set — the crate itself has no default scope. An explicit `required_scope: ""` is still an error, never "no scope". |
| `required_scopes` | `[]` | Further required scopes — a token must carry ALL of them, unioned with `required_scope`. Setting this alone (with `required_scope` unset) replaces the implicit `mcp:read` default with exactly this list. |
| `scopes_supported` | `[mcp:read, mcp:write]` | Advertised to clients, in the metadata document and in every 401's `scope=`. Enforcement is `required_scope`/`required_scopes`, but clients request exactly what this lists, so it must include every required scope — change both together. An explicit `[]` advertises nothing: the metadata document omits `scopes_supported`, and every 401's `scope=` names the required scopes instead. |
| `scope_claims` | `[scope, scp]` | The claims scopes are read from. Each can be a space-delimited string or an array, and their contents are combined. |
| `principal_claims` | `[preferred_username, sub]` | Log attribution only: the first claim present names the caller. |
| `algorithms` | RS/PS 256/384/512, ES256, ES384, EdDSA | Accepted signing algorithms. `HS*` and `none` are refused when the config loads. |
| `leeway_secs` | `60` | Clock-skew allowance on `exp`/`nbf`. Maximum 300. |
| `require_at_jwt` | `false` | Require header `typ: at+jwt` (RFC 9068 §4 requires it; off is a deliberate leniency for servers that emit `JWT`). Turn it on if your AS emits it. |
| `allow_insecure_http` | `false` | Accept a plain `http://` `issuer`, `jwks_uri` or `resource` on a non-loopback host — for an in-cluster address such as `http://authentik-server:9000/...` on a network you trust. Each such URL is still logged as a startup warning. It also governs URLs reached at run time: a `jwks_uri` discovered from a loopback `http://` issuer, and any redirect followed while fetching keys, may land on plain `http` on a non-loopback host only with it set (each use is logged as a warning). Loopback hosts never need it. |
| `allow_unscoped_tokens` | `false` | Accept a block with no required scope and `require_at_jwt` off. Has no effect here: this server always requires a scope (`mcp:read` unless you name another). |
| `accept_static_bearer` | `true` | Set `false` to run OAuth-only even when `MCP_BEARER_TOKEN` is set. |

A misconfigured block stops the server at startup, and the error lists every
problem at once. A problem that only shows up when the server contacts the AS
(unreachable issuer, discovery mismatch, no usable keys) is logged as one warning
at startup instead. Startup itself does not depend on the AS being up.

### Choosing the audience

The `aud` value is the setting that most often goes wrong, and there is no safe
default, so you must set it:

- **Your AS stamps the resource URL** (it honours RFC 8707, or it lets you configure
  an access-token audience per client): set `audience` to the same value as
  `resource`. This is what the MCP authorization spec intends.
- **Your AS stamps the OAuth client_id** and ignores the `resource` parameter: set
  `audience` to the client_id. This departs from the MCP authorization spec
  ("Token Handling" and "Access Token Privilege Restriction": an MCP server MUST
  accept only tokens issued for it as the audience, per RFC 8707 §2) and from
  RFC 9068 §4 (`aud` must identify this resource server). Every token that
  client obtains from your AS, for any resource, carries the same `aud` and is
  accepted here, so it is sound only when that OAuth client is **dedicated to
  this one server** — never shared with another MCP server or API. Use a
  resource-URL audience wherever your AS supports one. The crate's
  [Audience](https://github.com/St0nefish/oauth-resource-server#audience-which-value-to-configure) section has the detail.

To find out which yours does, decode one real access token (the middle segment is
base64url JSON) and look at `aud`. To switch from one to the other without
downtime, list both in `audiences` while you migrate.

## Setting up a provider and an MCP client

Provider recipes (Authentik, Authelia, Kanidm) and a checklist for any other
provider — issuing JWT access tokens, signing asymmetrically, choosing the
audience, finding the scope claim, and registering a client — now live in the
crate's [`docs/providers.md`](https://github.com/St0nefish/oauth-resource-server/blob/master/docs/providers.md),
each recipe labeled by exactly how much it's been verified.

**Using a crate recipe here:** the crate's recipes are written as a top-level
YAML block for a generic resource server, and don't paste into mcp-md-wiki's
config unchanged:

- Nest every key under `mcp.oauth` in `config.yaml` (i.e. `mcp:` with an
  `oauth:` block under it — see the example block below) — `resource =
  "https://api.example.com"` in the recipe becomes `mcp.oauth.resource:
  https://api.example.com` here.
- Always set `resource` to **this server's own `/mcp` URL**, e.g.
  `https://kb.example.com/mcp`, not the generic `https://api.example.com` the
  recipe shows.
- Set `audience` per [Choosing the audience](#choosing-the-audience) above,
  **not** by copying the recipe's `audience` line unchanged: Authentik and
  Kanidm stamp the OAuth client id, so `audience` there is that client's id
  (not a URL), while Authelia stamps the resource, so `audience` there must
  equal `resource` — on both the Authelia server config and the Authelia
  client itself.
- Drop the recipe's `required_scope` and `scopes_supported` lines, or set
  them to `mcp:read` and `[mcp:read, mcp:write]` explicitly — mcp-md-wiki
  already defaults them to exactly that (see the table above), and pasting the
  recipe's own scopes (e.g. `api:read`) verbatim means tokens need a scope
  your IdP likely isn't issuing, and every caller gets
  `403 insufficient_scope`. If your IdP can't issue `mcp:read` itself
  (Kanidm's recipe warns it may refuse a `:` in a scope name), use a name it
  does accept (e.g. `mcp_read`) in all three places, and keep them matched:
  `mcp.oauth.required_scope` and `mcp.oauth.scopes_supported` here (e.g.
  `mcp_read` and `[mcp_read, mcp_write]`), and the IdP's scope map.
  `scopes_supported` is this server's own setting, not the IdP's: it is what
  the metadata document and every 401 challenge tell clients to request, so
  leaving it at the `mcp:read`/`mcp:write` default while requiring `mcp_read`
  gets every call a 403.
- Replace the recipes' own example scope names — `api:read`/`api:write` —
  with `mcp:read`/`mcp:write` (or your chosen scope from the bullet above)
  everywhere they appear on the IdP side, not just in the `required_scope`
  line: Authelia's client `scopes:` list, Authentik's scope mappings, and
  Kanidm's scope map. If you drop the recipe's `required_scope` line and rely
  on the `mcp:read` default here but leave the IdP client registered with
  `api:read`/`api:write` as shown in the recipe, the issued token never
  carries `mcp:read` and every caller gets `403 insufficient_scope`.

For reference, the production-verified `mcp.oauth` block for Authentik looks
like:

```yaml
mcp:
  oauth:
    enabled: true
    issuer: "https://auth.example.com/application/o/mcp-md-wiki/"
    jwks_uri: "https://auth.example.com/application/o/mcp-md-wiki/jwks/"
    resource: "https://kb.example.com/mcp"
    audience: "mcp-md-wiki-client-id"   # Authentik's OAuth client id, not a URL
    required_scope: "mcp:read"
```

Everything else about setting up a provider — creating the application,
signing asymmetrically, registering the client — is not mcp-md-wiki-specific
and is covered in full in the crate's `docs/providers.md`.

What's specific to this server:

- **Grant the required scope** (`mcp:read` by default) to the users who should get
  in. Remember that write access currently comes with it.
- **Register the MCP client.** Most self-hosted authorization servers have no
  dynamic client registration, so create a public client with PKCE and give its
  client id to the MCP client. For Claude Code, that is `claude mcp add --transport
  http --client-id <id> --callback-port <port> <name> <url>`. The redirect URI must
  be registered exactly.
- **Hosted clients** (claude.ai, Claude Desktop, the mobile apps) can only connect
  to a remote MCP server through the OAuth authorization-code flow — the static
  bearer token does not work for them. See [README.md](../README.md#authentication-oauth-recommended).
- **Check the startup log.** It should say `authorization server signing keys
  loaded`. Then decode a failing request's reason from the `OAuth bearer auth
  rejected` warning, logged under the `oauth_resource_server` target. That
  line (and the JWKS load/refresh failure warnings) is WARN, so it's already
  visible under the default `RUST_LOG=info` — no extra directive needed,
  *unless* your `RUST_LOG` names only `mcp_md_wiki` with no global level
  (e.g. bare `mcp_md_wiki=debug`), which hides every other target including
  this one. Only the DEBUG-level accept/reject detail lines (`OAuth bearer
  auth accepted`, `No bearer credential presented`) need you to add
  `oauth_resource_server=debug` to `RUST_LOG` to see them, e.g.
  `RUST_LOG=info,mcp_md_wiki=debug,oauth_resource_server=debug` (see
  `main.rs`'s `DEFAULT_LOG_FILTER` doc comment for the full reasoning). See
  [TROUBLESHOOTING](../deploy/TROUBLESHOOTING.md#oauth).

## Security properties and design rationale

The algorithm allowlist, key-type binding, JWKS discovery/refresh, network limits,
scope-claim handling, and the design decisions behind each of those (why there's no
audience default, why HMAC is refused outright, why discovery doesn't block
startup, and so on) are all `oauth-resource-server`'s and documented once, for every
consumer of the crate, in its [Security model](https://github.com/St0nefish/oauth-resource-server#security-model) and
[Design rationale](https://github.com/St0nefish/oauth-resource-server#design-rationale) sections.
