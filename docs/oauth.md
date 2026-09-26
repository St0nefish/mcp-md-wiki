# OAuth for mcp-md-wiki

mcp-md-wiki can act as an **OAuth 2.1 resource server**: MCP clients get an access
token from your identity provider (the *authorization server*, AS) and present it on
every request, and the server verifies it locally against the provider's published
signing keys. This is the recommended way to protect `/mcp`, `/status`, `/metrics`
and `POST /admin/reload`:

- **Per user.** Every token names who it belongs to, so the logs do too.
- **Expiring and revocable.** A leaked token dies on its own, usually within an hour
  or less. A leaked static bearer token works until someone rotates it by hand.
- **Works with hosted clients.** claude.ai, Claude Desktop and the mobile apps can
  only connect to a remote MCP server through an OAuth authorization-code flow.

The static bearer token (`MCP_BEARER_TOKEN`) is still supported, and by default it
keeps working alongside OAuth. If you run with only the static token, the server
logs a startup warning recommending OAuth. Nothing about an existing deployment
changes until you edit its config.

> **Scope enforcement is coarse.** A token must carry `mcp.oauth.required_scope`
> (default `mcp:read`) to get in at all. There is no per-tool check yet, so any
> accepted token can also call the write tools (`write_document`,
> `delete_document`, `update_schema`). Only grant the scope to people you would let
> edit the knowledge base.

## How it works

1. A client calls `/mcp` without a token and gets `401` with
   `WWW-Authenticate: Bearer error="invalid_token", resource_metadata="…", scope="…"`.
2. It fetches the protected-resource metadata (RFC 9728) from
   `/.well-known/oauth-protected-resource/mcp` (also served at
   `/.well-known/oauth-protected-resource`). Both routes are unauthenticated. The
   document names your `issuer` as the authorization server, plus
   `scopes_supported`.
3. The client runs the authorization-code flow with PKCE against your AS and comes
   back with an access token.
4. For each request, mcp-md-wiki checks the token:
   1. It must be a JWT. Opaque tokens are refused.
   2. The `alg` must be on the allowlist, and `typ` must be an access-token type.
   3. The signature must verify against the AS's JWKS.
   4. `iss` must equal `issuer` byte for byte.
   5. `aud` must contain one of your configured audiences.
   6. `exp` and `nbf` must hold, give or take `leeway_secs`.
   7. The required scope must be present, or the request gets `403
      insufficient_scope`.

The JWKS URI comes from `jwks_uri` when you set it. If you leave it empty, the
server discovers it from the issuer's own metadata (OpenID Connect Discovery first,
then RFC 8414). A discovered document whose `issuer` does not match yours byte for
byte is refused. Keys are loaded at startup and re-read every hour, so a key the AS
withdraws stops being trusted. A token signed with an unknown `kid` triggers at
most one refetch per minute. If the AS is unreachable, the server rejects tokens
(fails closed) but keeps the keys it already has.

## Configuration

Everything lives under `mcp.oauth` in `config.yaml`. Every key is restart-only
(`POST /admin/reload` reports it as `restart_required`). The full annotated block is
in [`deploy/config.example.yaml`](../deploy/config.example.yaml).

| Key | Default | Notes |
|---|---|---|
| `enabled` | `false` | Master switch. |
| `issuer` | (required) | Copy it exactly from the AS's discovery document, including any trailing slash. |
| `resource` | (required) | The public URL of this server's MCP endpoint, e.g. `https://kb.example.com/mcp`. |
| `audience` | (required, unless `audiences` is set) | A value the token's `aud` must contain. See [Choosing the audience](#choosing-the-audience). |
| `audiences` | `[]` | More accepted audiences. Combined with `audience`; a match on any one is enough. |
| `jwks_uri` | `""` (discover) | Set it to skip discovery. |
| `required_scope` | `mcp:read` | A single scope. Matching is exact and case-sensitive. |
| `scopes_supported` | `[mcp:read, mcp:write]` | Advertised to clients. |
| `scope_claims` | `[scope, scp]` | The claims scopes are read from. Each can be a space-delimited string or an array, and their contents are combined. |
| `principal_claims` | `[preferred_username, sub]` | Log attribution only: the first claim present names the caller. |
| `algorithms` | RS/PS 256/384/512, ES256, ES384, EdDSA | Accepted signing algorithms. `HS*` and `none` are refused when the config loads. |
| `leeway_secs` | `60` | Clock-skew allowance on `exp`/`nbf`. Maximum 300. |
| `require_at_jwt` | `false` | Require header `typ: at+jwt` (RFC 9068). Turn it on if your AS emits it. |
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
  `audience` to the client_id.

To find out which yours does, decode one real access token (the middle segment is
base64url JSON) and look at `aud`. To switch from one to the other without
downtime, list both in `audiences` while you migrate.

## Provider recipes

Only providers that have actually been tested are listed here. Each line is
marked:

- **[verified]**: observed in a real deployment or a sandbox test.
- **[docs]**: taken from the provider's documentation and not independently tested.

Placeholders: `auth.example.com` / `idm.example.com` for the AS,
`kb.example.com` for this server, `example-client` for the client id.

### Authentik

Status: **verified in production**, the deployment this feature was first built
for, running OAuth alongside the static bearer token.

- [verified] Use an OAuth2/OpenID provider. The per-application issuer has a trailing
  slash: `https://auth.example.com/application/o/<slug>/`.
- [verified] The JWKS is at `<issuer>jwks/`. Discovery also finds it.
- [verified] `aud` is the provider's **client_id** as a string. The `resource`
  parameter is ignored.
- [verified] `scope` is a space-delimited string, signed RS256, with header `typ:
  JWT` (so leave `require_at_jwt` off).
- [docs] The provider needs a **signing key**. Without one, Authentik signs with
  HS256 using the client secret, which no resource server can verify (and
  mcp-md-wiki refuses HS256 outright).
- [docs] Only scopes backed by a scope mapping on the provider end up in `scope`.
  Create mappings for `mcp:read` / `mcp:write`, or change `required_scope` to a
  scope the provider does issue.

```yaml
mcp:
  oauth:
    enabled: true
    issuer: "https://auth.example.com/application/o/wiki/"
    jwks_uri: "https://auth.example.com/application/o/wiki/jwks/"   # optional
    audience: "example-client-id"
    resource: "https://kb.example.com/mcp"
```

### Authelia (4.39)

Status: **verified in a sandbox**, end to end. Real Authelia 4.39.4 tokens were
accepted and rejected as expected by this server, including discovery, the `scp`
claim, a resource-URL audience, `require_at_jwt`, a 403 for a token missing
`mcp:read`, and a 401 when the audience is configured as the client_id instead.

- [verified] Access tokens are **opaque by default**. Set
  `access_token_signed_response_alg: 'RS256'` on the client to get JWTs. The header
  is then `{"alg":"RS256","typ":"at+jwt"}`.
- [verified] Scopes arrive as `scp` (a JSON array), with no `scope` claim. Authelia
  refuses to add one through a claims policy. The default `scope_claims` reads
  `scp`.
- [verified] `aud` never contains the client_id. It holds only the client's
  configured `audience` values, and the `resource` parameter is ignored. Configure
  the client with `audience: ['https://kb.example.com/mcp']` and
  `requested_audience_mode: 'implicit'`, and set the server's `audience` to the same
  URL.
- [verified] `iss` has no trailing slash. `sub` is an opaque UUID, and the access
  token carries no username, so logs show the `sub`.
- [verified] There is no dynamic client registration, so pre-register a client. A
  public client (`public: true`, `token_endpoint_auth_method: 'none'`, PKCE S256)
  works. The custom scopes `mcp:read`/`mcp:write` cause only a validation warning.
- [verified] For redirect URIs, `http://127.0.0.1/callback` matches 127.0.0.1 on
  any port. `localhost` matches only exactly registered ports.

Authelia client (excerpt):

```yaml
identity_providers:
  oidc:
    clients:
      - client_id: 'example-client'
        public: true
        token_endpoint_auth_method: 'none'
        authorization_policy: 'one_factor'      # or two_factor
        consent_mode: 'implicit'
        require_pkce: true
        pkce_challenge_method: 'S256'
        redirect_uris:
          - 'http://127.0.0.1/callback'         # matches 127.0.0.1 on any port
          - 'http://localhost:38765/callback'   # exact port, e.g. claude --callback-port 38765
        scopes: ['openid', 'offline_access', 'mcp:read', 'mcp:write']
        response_types: ['code']
        grant_types: ['authorization_code', 'refresh_token']
        access_token_signed_response_alg: 'RS256'   # without this, tokens are opaque
        audience: ['https://kb.example.com/mcp']
        requested_audience_mode: 'implicit'
```

mcp-md-wiki:

```yaml
mcp:
  oauth:
    enabled: true
    issuer: "https://auth.example.com"          # no trailing slash
    audience: "https://kb.example.com/mcp"      # the resource URL, not the client id
    resource: "https://kb.example.com/mcp"
    require_at_jwt: true
```

### Kanidm

Status: **token shape verified in a sandbox; not tested end to end with this
server.** A real Kanidm access token was verified with an independent JWT tool. A
unit test replays that exact shape through mcp-md-wiki's validator
(`observed_shape_kanidm_…` in `src/oauth.rs`), but no live Kanidm token has been
sent to a running mcp-md-wiki.

- [verified] Each client has its own issuer: `https://idm.example.com/oauth2/openid/<client>`.
  The trailing-slash variant does not match. Discovery works at
  `<issuer>/.well-known/openid-configuration`, and the JWKS is per client.
- [verified] Tokens are signed **ES256** with header `typ: at+jwt`. `aud` is the
  **client name** (a string). `scope` is a space-delimited string. `sub` is a UUID,
  with no username or groups in the access token. Tokens live 900 s.
- [verified] The `resource` parameter is accepted and ignored. There is no dynamic
  client registration. Public clients require PKCE S256.
- [docs] Grant scopes to users with a scope map on the client, for a group. Whether
  Kanidm accepts a scope name containing `:` was not tested. If it refuses
  `mcp:read`, use e.g. `mcp_read` and set `required_scope` and `scopes_supported`
  to match.

```yaml
mcp:
  oauth:
    enabled: true
    issuer: "https://idm.example.com/oauth2/openid/example-client"
    audience: "example-client"
    resource: "https://kb.example.com/mcp"
    require_at_jwt: true
```

### Any other provider: checklist

1. **Issue JWT access tokens.** mcp-md-wiki cannot verify opaque tokens: there is no
   RFC 7662 introspection. Many servers issue opaque tokens by default and have a
   per-client or global switch for JWTs. A token without two `.` characters is
   opaque, and the server logs `credential is not a JWT`.
2. **Sign asymmetrically.** Use RS256, PS256, ES256, ES384 or EdDSA. HS256 cannot
   be verified by a resource server.
3. **Copy the issuer exactly** from `<issuer>/.well-known/openid-configuration`,
   trailing slash included.
4. **Decode a real token** and read three claims:
   - `aud`: put it in `audience`. See [Choosing the audience](#choosing-the-audience).
   - Where the scopes are: `scope` or `scp`, as a string or an array, all work by
     default. Anything else goes in `scope_claims`.
   - `typ` in the header: if it is `at+jwt`, turn on `require_at_jwt`.
5. **Grant the required scope** (`mcp:read` by default) to the users who should get
   in. Remember that write access currently comes with it.
6. **Register the MCP client.** Most self-hosted servers have no dynamic client
   registration, so create a public client with PKCE and give its client id to the
   MCP client. For Claude Code, that is `claude mcp add --transport http
   --client-id <id> --callback-port <port> <name> <url>`. The redirect URI must be
   registered exactly.
7. **Check the startup log.** It should say `authorization server signing keys
   loaded`. Then decode a failing request's reason from the `OAuth bearer auth
   rejected` warning. See [TROUBLESHOOTING](../deploy/TROUBLESHOOTING.md#oauth).

## Security notes

- **Use `require_at_jwt` when your AS supports it.** With `aud` set to the
  client_id (Authentik, Kanidm), an **ID token** for the same client also has a
  matching `iss` and `aud`. It is normally stopped by the scope check, because ID
  tokens carry no `scope`. The `typ` check stops it explicitly, but only for servers
  that emit `at+jwt`.
- **Algorithms and keys.** Each key is limited to the algorithms its own type (and
  its `alg`, if it declares one) can produce. A token cannot steer an RSA key into
  ECDSA verification, or any key into HMAC. Symmetric (`oct`) keys and `use: enc`
  keys in a JWKS are ignored.
- **Network.** The server fetches only URLs derived from your configured `issuer`,
  or your `jwks_uri`. Response bodies are capped. Redirects are limited, and a
  redirect from https to http is refused. A plain-http issuer on a non-loopback
  host produces a startup warning.
- **Logs.** Tokens are never logged. Accepted requests are attributed at DEBUG
  level, and refused ones are logged at WARN with the reason (for example
  `InvalidAudience`). The HTTP response carries no reason, so a client cannot probe
  which check failed.

## Design notes

Decisions made while making OAuth provider-agnostic, with the main alternatives
that were rejected:

- **Scopes: read `scope` and `scp` by default, in every shape, and combine them.**
  A per-provider "scope format" switch was rejected because the default already
  covers every shape seen in practice, and a claim a token does not carry adds
  nothing.
- **Audience: explicit config with no default, accepting any of several values.**
  Defaulting to `resource` was rejected: it would silently reject every Authentik
  and Kanidm token, and the choice is too consequential to guess.
- **Algorithms: a wide asymmetric allowlist, with each key bound to its own type.**
  Staying RS256-only was rejected because Kanidm signs ES256 by default. HMAC is
  refused outright rather than made configurable.
- **Discovery: used only when `jwks_uri` is empty, with an exact issuer match, run
  in the background at startup.** Making startup wait for discovery was rejected,
  because an outage at the AS would then also take this server down.
- **`typ`: accept `at+jwt`, `JWT` or no `typ` by default, with an opt-in strict
  mode.** Making strict the default was rejected because Authentik, Keycloak and
  Entra ID emit `JWT`.
- **Opaque tokens: not supported.** RFC 7662 introspection would need a client
  secret, an AS round trip per request (or an introspection cache with its own
  revocation semantics), and a second validation path. It is a possible follow-up,
  not part of this change. Every server this project has tested can issue JWT
  access tokens.
- **401 on a request with no credential keeps `error="invalid_token"`.** RFC 6750
  §3.1 says to omit the error code there. That change was deferred because clients
  already in use start their flow from the current challenge. The
  `resource_metadata` parameter they depend on is present either way.
