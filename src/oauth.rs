//! OAuth 2.1 resource-server support: JWT access-token validation and RFC 9728
//! protected-resource metadata, written to work with any standards-leaning
//! authorization server rather than one product's token shape.
//!
//! This module makes the process a *resource server* only. It never issues,
//! refreshes, revokes or introspects tokens, and it never talks to the
//! authorization server except to fetch its metadata (only when `jwks_uri` is not
//! configured) and its public signing keys (JWKS). Everything it needs is either
//! config (`mcp.oauth`, see `config::OAuthConfig`) or derived from it once at
//! construction.
//!
//! ## Provider-agnostic by construction
//!
//! Authorization servers agree on the signature and on `iss`/`aud`/`exp`, and
//! disagree on nearly everything else an access token carries:
//!
//! - **Scopes** arrive as an RFC 9068 `scope` string (Authentik, Kanidm), an `scp`
//!   array (Authelia, Okta, Ory Hydra), or an `scp` string (Entra ID). The claims
//!   read are `mcp.oauth.scope_claims`, every string/array shape is accepted, and
//!   the results are unioned — see [`extract_scopes`].
//! - **Audience** is the client_id on some servers (Authentik, Kanidm) and the
//!   resource URL or an API identifier on others (Authelia with `audience`
//!   configured, RFC 8707 servers). Which one this server accepts is config
//!   (`audience` + `audiences`), never guessed.
//! - **Algorithms** are RS256 on most servers but ES256 on Kanidm and EdDSA on
//!   some others; the allowlist is config, and each key is bound to the algorithms
//!   its own type can produce, so a token cannot pick a verification algorithm the
//!   key was not made for.
//! - **Usernames** are frequently absent from access tokens (Authelia, Kanidm:
//!   only a UUID `sub`), so the name logged for a request is the first present
//!   claim of `mcp.oauth.principal_claims`.
//!
//! Opaque (non-JWT) access tokens are refused. RFC 7662 introspection would cover
//! them but needs a client credential and per-request AS round trips; it is a
//! deliberate non-goal for now, not an oversight.
//!
//! ## Dual-mode auth
//!
//! OAuth is the recommended credential, and the static bearer token is still
//! accepted alongside it unless `mcp.oauth.accept_static_bearer: false` —
//! `server::bearer_auth` admits a request if the constant-time static-token
//! comparison matches OR a JWT validates here. A deployment can run either, both,
//! or neither.
//!
//! ## What is deliberately NOT here
//!
//! Per-tool write-scope enforcement — requiring `mcp:write` for `write_document` /
//! `delete_document` / `update_schema` — is **not implemented**. The auth
//! middleware sits in front of the whole `/mcp` endpoint and cannot see which MCP
//! tool a request invokes; that lives in the JSON-RPC body, which only rmcp's
//! transport parses. **Any token that passes validation here currently grants full
//! access, writes included.** The validated scope list is put into request
//! extensions as an [`AuthorizedToken`] precisely so a later change can enforce
//! per-tool scopes at the tool-dispatch layer without re-deriving them.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use jsonwebtoken::jwk::{AlgorithmParameters, EllipticCurve, Jwk, KeyAlgorithm, PublicKeyUse};
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use serde_json::{Map, Value};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::config::ResolvedOAuthConfig;

/// Path suffix RFC 9728 §3 splices between a resource's authority and its path to
/// form the metadata URL. Also the literal route prefix `server::assemble_router`
/// registers — see `OAuthValidator::new`, which warns when a configured `resource`
/// would derive a URL those routes do not answer on.
pub const PROTECTED_RESOURCE_METADATA_PREFIX: &str = "/.well-known/oauth-protected-resource";

/// Default `mcp.oauth.algorithms`: every asymmetric JWS algorithm `jsonwebtoken`
/// 9 (on `ring`) can verify. Deliberately wide — which algorithm a token may use is
/// ALSO constrained by the key it names (see [`key_algorithms`]), so accepting
/// ES256 here cannot make an RSA key verify an ES256 signature. HS256/384/512 and
/// `none` are not merely absent: [`parse_algorithm`] refuses them, so no config can
/// turn them on. ES512 is absent because `ring` cannot verify P-521.
pub const DEFAULT_ALGORITHMS: &[&str] = &[
    "RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "EdDSA",
];

/// Default `mcp.oauth.scope_claims`. `scope` is RFC 9068 §2.2.3's space-delimited
/// string (Authentik, Kanidm, Keycloak); `scp` is what Authelia (array), Okta
/// (array), Ory Hydra (array or string) and Entra ID (string) emit instead.
/// Reading both by default is what makes an Authelia token pass without
/// per-provider config, and it cannot widen access for a token that carries only
/// `scope` (every Authentik token) because a claim the token does not have
/// contributes nothing.
pub const DEFAULT_SCOPE_CLAIMS: &[&str] = &["scope", "scp"];

/// Default `mcp.oauth.principal_claims`. `email` is left out on purpose so a
/// default deployment does not write addresses into its logs; an operator who
/// wants it adds it.
pub const DEFAULT_PRINCIPAL_CLAIMS: &[&str] = &["preferred_username", "sub"];

/// Default `mcp.oauth.leeway_secs`: the value the original code hardcoded.
pub const DEFAULT_LEEWAY_SECS: u64 = 60;

/// Ceiling on `mcp.oauth.leeway_secs`. Leeway is for clock drift; a value large
/// enough to matter against a 5–15 minute token lifetime (Kanidm issues 900 s
/// tokens) is a way of switching `exp` off, which config must not be able to do.
pub const MAX_LEEWAY_SECS: u64 = 300;

/// How long an unknown `kid` is allowed to trigger a JWKS refetch again.
///
/// An unknown `kid` is attacker-controllable — it is just a field in an unverified
/// token header — so without this a stream of junk tokens would turn this server
/// into an amplifier pointed at the identity provider. One refetch per minute is
/// far faster than any real key rotation needs (JWKS rollovers publish the new key
/// alongside the old one well before signing with it) and slow enough that the IdP
/// never notices us.
const JWKS_MIN_REFETCH_INTERVAL: Duration = Duration::from_secs(60);

/// Ceiling on a single metadata or JWKS fetch. Bounds how long the write lock
/// below is held, and therefore how long a stalled IdP can stall token validation.
const JWKS_FETCH_TIMEOUT: Duration = Duration::from_secs(10);

/// How often the background task re-reads the JWKS even when every `kid` is known.
///
/// The unknown-`kid` refetch picks up a NEW key; only a periodic re-read notices a
/// key the authorization server has WITHDRAWN (rotated out after a compromise, for
/// instance). Without it a retired key would stay trusted for the life of the
/// process. An hour bounds that window without being a load anyone would notice.
const JWKS_BACKGROUND_REFRESH_INTERVAL: Duration = Duration::from_secs(3600);

/// Cap on a metadata/JWKS response body. Real key sets are a few KiB; the cap is
/// there so a misbehaving (or impersonated) endpoint cannot make this process
/// buffer an unbounded body on the credential-checking path.
const MAX_FETCH_BYTES: usize = 256 * 1024;

/// Cap on keys taken from one JWK Set, for the same reason as [`MAX_FETCH_BYTES`]:
/// every key is parsed and scanned on lookup, and no real AS publishes dozens.
const MAX_JWKS_KEYS: usize = 64;

/// Cap on a presented credential. Real access tokens are well under 8 KiB even
/// with group claims; anything larger is refused before it is base64-decoded.
const MAX_TOKEN_BYTES: usize = 16 * 1024;

/// Cap on any token-derived string that reaches a log line (`kid`, `typ`,
/// principal). A signed claim is trustworthy but not necessarily short, and an
/// unverified header field is neither.
const MAX_LOGGED_CHARS: usize = 128;

/// Parse one `mcp.oauth.algorithms` entry, refusing everything that must never be
/// accepted with the reason spelled out (the error lands in a startup failure).
pub fn parse_algorithm(name: &str) -> std::result::Result<Algorithm, String> {
    let name = name.trim();
    if name.eq_ignore_ascii_case("none") {
        return Err(format!(
            "\"{name}\" — an unsigned token is never acceptable"
        ));
    }
    match name.parse::<Algorithm>() {
        Ok(Algorithm::HS256 | Algorithm::HS384 | Algorithm::HS512) => Err(format!(
            "\"{name}\" — HMAC algorithms verify with a shared secret, which a resource \
             server must never hold, and accepting one alongside a public key set is the \
             classic key-confusion attack (a token signed with the PUBLIC key as the HMAC \
             secret)"
        )),
        Ok(alg) => Ok(alg),
        Err(_) => Err(format!(
            "\"{name}\" — not a JWS algorithm this server can verify (supported: {})",
            DEFAULT_ALGORITHMS.join(", ")
        )),
    }
}

/// A successfully validated access token, inserted into request extensions by
/// `server::bearer_auth`.
///
/// Nothing reads `scopes` yet — see the module doc: write-scope enforcement is not
/// implemented, and every valid token currently grants full access. This type
/// exists so that when it is, the scopes come from the one place that actually
/// verified them rather than being re-parsed out of the header a second time.
#[derive(Debug, Clone)]
pub struct AuthorizedToken {
    /// The token's `sub`, when it carried one.
    pub subject: Option<String>,
    /// The first present, non-empty string claim of `mcp.oauth.principal_claims`
    /// — who the request is from, for logs. Never the token itself.
    pub principal: Option<String>,
    /// The union of every `mcp.oauth.scope_claims` claim, in first-seen order.
    pub scopes: Vec<String>,
}

impl AuthorizedToken {
    /// Whether the token carries `scope`. Unused today (nothing enforces per-tool
    /// scopes yet — see the module doc); kept as the single place that will answer
    /// that question so callers never hand-roll a `.iter().any()` over `scopes`.
    #[allow(dead_code)]
    pub fn has_scope(&self, scope: &str) -> bool {
        self.scopes.iter().any(|s| s == scope)
    }
}

/// Why a bearer credential was refused, and — crucially — with which HTTP status.
///
/// The split is the whole point: RFC 6750 distinguishes "this token is not good"
/// (401 `invalid_token`, go get a new one) from "this token is fine but not
/// sufficient" (403 `insufficient_scope`). A client that gets 401 for an
/// insufficient-scope token will loop through the authorization flow forever and
/// land back on the same refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenRejection {
    /// 401: the request carried no bearer credential at all. Separate from
    /// `Invalid` so the server can log it quietly — every OAuth client's first
    /// request looks like this — not because the response differs (it does not;
    /// see `server::AuthState::challenge`).
    Missing,
    /// 401 `invalid_token`: malformed, unsigned, wrong issuer/audience/type,
    /// expired, or signed by a key we could not obtain. The string is for logs
    /// only — it is never returned to the caller, since telling an unauthenticated
    /// client exactly which check failed is a free oracle.
    Invalid(String),
    /// 403 `insufficient_scope`: signature, issuer, audience and expiry all passed,
    /// but the token does not carry the configured required scope.
    InsufficientScope,
}

/// One usable verification key from the JWK Set, with the algorithms it may verify.
///
/// `algorithms` is the intersection of what the key's TYPE can produce, what its
/// own `alg` parameter declares (when present) and the configured allowlist. A
/// token's `alg` must be in it, which is what stops an attacker-chosen header from
/// steering an RSA key into an ECDSA verification, or any key into HMAC.
struct CachedKey {
    kid: Option<String>,
    key: DecodingKey,
    algorithms: Vec<Algorithm>,
}

/// The in-memory JWKS, the URI it came from, plus when we last *attempted* to
/// refresh it.
///
/// Attempt, not success, on purpose: a failing IdP must be backed off exactly like
/// a successful-but-stale one, or an outage turns every junk token into a retry
/// against a service that is already struggling.
#[derive(Default)]
struct JwksCache {
    /// `mcp.oauth.jwks_uri` when configured; otherwise `None` until discovery
    /// fills it in, after which it is fixed for the life of the process (a new
    /// value means a config change, and every `mcp.oauth` key is restart-only).
    jwks_uri: Option<String>,
    keys: Vec<CachedKey>,
    last_attempt: Option<Instant>,
}

pub struct OAuthValidator {
    /// Everything below is derived from this once; it is kept for the metadata
    /// document and for logging.
    config: ResolvedOAuthConfig,
    /// Pre-rendered so the 401/403 paths are a string clone, not a `format!` per
    /// rejected request.
    resource_metadata_url: String,
    /// `scopes_supported`, space-joined, for the 401 challenge's `scope` parameter.
    supported_scopes: String,
    /// The RFC 9728 document both well-known routes serve, rendered once.
    metadata: Value,
    /// Issuer/audience/expiry/not-before policy, built once. Each token gets a
    /// clone with `algorithms` narrowed to its own (already allowlisted and
    /// key-compatible) `alg`, because `jsonwebtoken` refuses a `Validation` whose
    /// algorithms span more than one key family. The claim checks all run inside
    /// `decode`, which is what keeps signature verification and claim validation
    /// from being two separately-forgettable steps.
    validation: Validation,
    http: reqwest::Client,
    /// Only ever held for in-memory reads and swaps — never across a network
    /// call. tokio's `RwLock` queues new readers behind a waiting writer, so a
    /// writer parked on a slow IdP would stall every request, including ones whose
    /// key is already cached.
    jwks: RwLock<JwksCache>,
    /// Serializes refreshes instead: a burst of unknown-`kid` requests, or
    /// the background refresher racing one, collapse into one fetch while
    /// cached-key lookups carry on untouched.
    refresh_lock: tokio::sync::Mutex<()>,
    /// Normally [`JWKS_MIN_REFETCH_INTERVAL`]; overridden only by tests, which
    /// would otherwise have to sleep a minute to observe a refetch.
    jwks_min_refetch_interval: Duration,
}

impl OAuthValidator {
    pub fn new(config: &ResolvedOAuthConfig) -> Result<Self> {
        Self::build(config, JWKS_MIN_REFETCH_INTERVAL)
    }

    fn build(config: &ResolvedOAuthConfig, jwks_min_refetch_interval: Duration) -> Result<Self> {
        // `Config::resolve` already refuses both of these; re-checked here because
        // an empty audience set or algorithm list is the one construction mistake
        // that would fail OPEN-adjacent (an empty `aud` set in jsonwebtoken means
        // "reject everything", but an empty allowlist is a panic-free foot-gun
        // nobody should have to reason about).
        let audiences = config.accepted_audiences();
        if audiences.is_empty() {
            bail!("mcp.oauth: no accepted audience configured");
        }
        let Some(&first_alg) = config.algorithms.first() else {
            bail!("mcp.oauth.algorithms is empty");
        };

        let mut validation = Validation::new(first_alg);
        // Byte-exact issuer match. Authentik's issuer ends in a slash and the
        // difference matters — `.../mcp-kb-rag/` and `.../mcp-kb-rag` are different
        // strings and only one of them is in the tokens.
        validation.set_issuer(&[&config.issuer]);
        // Membership, per RFC 7519 §4.1.3: `aud` may be a string or an array, and
        // the token is accepted if ANY element is one of the configured audiences.
        // What those audiences should be is provider-specific and deliberately
        // config, never guessed — the client_id on servers that ignore RFC 8707
        // (Authentik, Kanidm), the resource URL on servers configured to stamp it
        // (Authelia with a client `audience`). See `config::OAuthConfig::audience`.
        validation.set_audience(&audiences);
        // `jsonwebtoken` only validates `iss`/`aud` when the claim is *present*, so
        // requiring them here is what turns "wrong issuer" and "no issuer at all"
        // into the same refusal. Without this a token carrying neither claim would
        // sail through both checks.
        validation.set_required_spec_claims(&["exp", "iss", "aud"]);
        validation.leeway = config.leeway_secs;
        validation.validate_exp = true;
        // Off by default in jsonwebtoken. RFC 9068 tokens (Authelia, Kanidm) carry
        // `nbf`; a token presented before it is not yet valid.
        validation.validate_nbf = true;
        validation.validate_aud = true;

        let resource_metadata_url = resource_metadata_url(&config.resource);
        let supported_scopes = config.scopes_supported.join(" ");

        // Warn rather than fail: the two well-known routes the server registers are
        // fixed (`/.well-known/oauth-protected-resource` and `.../mcp`), because
        // that is where this deployment's MCP endpoint lives and where clients
        // probe. A `resource` with some other path still produces a valid metadata
        // document, but the URL advertised in `WWW-Authenticate` would point at a
        // path this process does not answer on — worth saying out loud once at
        // startup, not worth refusing to boot over.
        let advertised_path = resource_metadata_url
            .split_once("://")
            .and_then(|(_, rest)| rest.find('/').map(|i| rest[i..].to_string()))
            .unwrap_or_default();
        if advertised_path != format!("{PROTECTED_RESOURCE_METADATA_PREFIX}/mcp")
            && advertised_path != PROTECTED_RESOURCE_METADATA_PREFIX
        {
            warn!(
                resource = %config.resource,
                advertised = %resource_metadata_url,
                "mcp.oauth.resource derives a protected-resource metadata URL this server \
                 does not serve — it only answers on {PROTECTED_RESOURCE_METADATA_PREFIX} \
                 and {PROTECTED_RESOURCE_METADATA_PREFIX}/mcp. OAuth clients will 404 on \
                 discovery."
            );
        }
        // A required scope nobody is told to ask for is a guaranteed 403 for every
        // client that requests exactly `scopes_supported`. Not fatal — an operator
        // may be advertising a narrower menu on purpose — but never silent.
        if !config.scopes_supported.contains(&config.required_scope) {
            warn!(
                required_scope = %config.required_scope,
                scopes_supported = ?config.scopes_supported,
                "mcp.oauth.required_scope is not in mcp.oauth.scopes_supported — clients \
                 that request the advertised scopes will get 403 insufficient_scope"
            );
        }
        if config.issuer.starts_with("http://") && !is_loopback_url(&config.issuer) {
            warn!(
                issuer = %config.issuer,
                "mcp.oauth.issuer uses plain http on a non-loopback host — signing keys \
                 fetched over it can be substituted by anyone on the path. Use https."
            );
        }

        let metadata = serde_json::json!({
            "resource": config.resource,
            // Echoed byte-identically. A client compares this against the `iss` of
            // the tokens it receives and against the AS metadata's `issuer`, so
            // normalizing (adding or trimming a trailing slash, lowercasing) here
            // would break that comparison.
            "authorization_servers": [config.issuer],
            "scopes_supported": config.scopes_supported,
            "bearer_methods_supported": ["header"],
            "resource_name": "mcp-md-wiki knowledge base (MCP)",
        });

        let http = reqwest::Client::builder()
            .timeout(JWKS_FETCH_TIMEOUT)
            // Some servers redirect their JWKS path (a trailing-slash rewrite, say).
            // A handful of hops covers that; an unbounded chain only stretches a
            // refresh out. A hop from https to plain http is refused outright: it
            // would let anyone on the path substitute the signing keys.
            .redirect(reqwest::redirect::Policy::custom(|attempt| {
                let downgrade = attempt.url().scheme() != "https"
                    && attempt.previous().iter().any(|u| u.scheme() == "https");
                if downgrade {
                    attempt.error("redirect from https to a non-https URL refused")
                } else if attempt.previous().len() > 3 {
                    attempt.error("too many redirects")
                } else {
                    attempt.follow()
                }
            }))
            .build()
            .context("Failed to build the HTTP client for OAuth metadata/JWKS fetches")?;

        let jwks_uri = (!config.jwks_uri.trim().is_empty()).then(|| config.jwks_uri.clone());

        Ok(Self {
            config: config.clone(),
            resource_metadata_url,
            supported_scopes,
            metadata,
            validation,
            http,
            jwks: RwLock::new(JwksCache {
                jwks_uri,
                ..JwksCache::default()
            }),
            refresh_lock: tokio::sync::Mutex::new(()),
            jwks_min_refetch_interval,
        })
    }

    /// The RFC 9728 document, served by both well-known routes.
    pub fn metadata(&self) -> Value {
        self.metadata.clone()
    }

    /// The `WWW-Authenticate` value for every 401 — a refused credential and, by
    /// deliberate choice, a missing one too (see `server::AuthState::challenge`).
    ///
    /// Load-bearing, not cosmetic: claude.ai has been observed refusing to start the
    /// authorization flow at all when a 401 arrives without it, because
    /// `resource_metadata` is how the client finds the authorization server in the
    /// first place. Claude Code tolerates its absence, which is exactly why it is
    /// easy to drop and hard to notice. Emitted on EVERY 401 once OAuth is
    /// configured — including a failed static-bearer request, since the server
    /// cannot tell which credential the caller meant to present.
    pub fn invalid_token_challenge(&self) -> String {
        format!(
            "Bearer error=\"invalid_token\", resource_metadata=\"{}\", scope=\"{}\"",
            quoted(&self.resource_metadata_url),
            quoted(&self.supported_scopes)
        )
    }

    /// The `WWW-Authenticate` value for a 403: the token was genuinely valid, so
    /// `scope` names what it is *missing* rather than everything on offer. That is
    /// the difference that lets a client re-authorize for the right thing instead of
    /// replaying the same request.
    pub fn insufficient_scope_challenge(&self) -> String {
        format!(
            "Bearer error=\"insufficient_scope\", scope=\"{}\", resource_metadata=\"{}\"",
            quoted(&self.config.required_scope),
            quoted(&self.resource_metadata_url)
        )
    }

    /// Validate a bearer credential as a JWT access token.
    ///
    /// Order matters and is RFC 9068 §4's: everything that can be refused from the
    /// unverified header alone (size, shape, `alg` allowlist, `typ`) is refused
    /// before any key is fetched, so junk cannot schedule IdP traffic; then the
    /// signature; then issuer / audience / expiry / not-before — all inside
    /// `jsonwebtoken::decode`, so they cannot be reordered ahead of the signature by
    /// accident — then scope.
    pub async fn validate(&self, token: &str) -> Result<AuthorizedToken, TokenRejection> {
        if token.is_empty() {
            return Err(TokenRejection::Missing);
        }
        if token.len() > MAX_TOKEN_BYTES {
            return Err(TokenRejection::Invalid(format!(
                "credential is {} bytes, over the {MAX_TOKEN_BYTES}-byte cap",
                token.len()
            )));
        }
        if token.split('.').count() != 3 {
            // The single most useful hint in this file for a new deployment:
            // Authelia (by default), Ory Hydra (by default) and others issue OPAQUE
            // access tokens, which no amount of JWKS can verify.
            return Err(TokenRejection::Invalid(
                "credential is not a JWT (a mistyped static token, or an opaque access \
                 token — this server validates JWT access tokens only; configure the \
                 authorization server to issue JWT access tokens)"
                    .into(),
            ));
        }

        // The header is unverified data. It is read only to pick which key to
        // verify WITH; nothing from it is trusted afterwards, and `alg` is checked
        // against our allowlist (and later against the key) rather than obeyed.
        // `alg: none` does not even get this far: jsonwebtoken's `Algorithm` has no
        // `none` variant, so the header fails to parse.
        let header = decode_header(token)
            .map_err(|e| TokenRejection::Invalid(format!("malformed token header: {e}")))?;
        if !self.config.algorithms.contains(&header.alg) {
            return Err(TokenRejection::Invalid(format!(
                "token algorithm {:?} is not in mcp.oauth.algorithms",
                header.alg
            )));
        }
        check_typ(header.typ.as_deref(), self.config.require_at_jwt)?;

        let key = self.decoding_key(header.kid.as_deref(), header.alg).await?;

        let mut validation = self.validation.clone();
        validation.algorithms = vec![header.alg];
        let data = decode::<Map<String, Value>>(token, &key, &validation).map_err(|e| {
            // `jsonwebtoken`'s error kinds already distinguish bad signature from
            // bad issuer/audience/expiry; all of them are 401 `invalid_token` to the
            // caller, and only the log gets to know which.
            TokenRejection::Invalid(format!("token rejected: {e}"))
        })?;
        let claims = data.claims;

        // Belt and braces on `iss`: jsonwebtoken also accepts an `iss` ARRAY that
        // merely contains the configured issuer. RFC 7519 makes `iss` a single
        // StringOrURI, and "one of several issuers" is not a shape any real AS
        // emits, so anything but the exact string is refused.
        if claims.get("iss").and_then(Value::as_str) != Some(self.config.issuer.as_str()) {
            return Err(TokenRejection::Invalid(
                "token iss is not a single string equal to mcp.oauth.issuer".into(),
            ));
        }

        let scopes = extract_scopes(&claims, &self.config.scope_claims);
        let principal = extract_principal(&claims, &self.config.principal_claims);
        let subject = claims.get("sub").and_then(Value::as_str).map(for_log);

        if !scopes.iter().any(|s| s == &self.config.required_scope) {
            // Info, not debug: this is the refusal an operator wiring up a new
            // authorization server hits first (Authelia's `scp`-only tokens were
            // exactly this), and `present=[]` next to the claims that were
            // read is most of the diagnosis. Scopes are not secret.
            info!(
                principal = ?principal,
                required = %self.config.required_scope,
                present = ?scopes,
                scope_claims = ?self.config.scope_claims,
                "OAuth token is valid but lacks the required scope"
            );
            return Err(TokenRejection::InsufficientScope);
        }

        Ok(AuthorizedToken {
            subject,
            principal,
            scopes,
        })
    }

    /// Load (or reload) the key set now, discovering the JWKS URI first if needed.
    /// Returns how many usable keys it holds. On failure the previous keys are
    /// kept — a transient IdP outage must not invalidate keys that are still good.
    pub async fn refresh_now(&self) -> Result<usize> {
        let _refreshing = self.refresh_lock.lock().await;
        self.refresh().await
    }

    /// Warm the key cache at startup and keep it fresh.
    ///
    /// The first pass turns a misconfigured issuer, an unreachable JWKS or a
    /// discovery mismatch into one clear log line at boot instead of a wall of
    /// 401s on the first real request — without making startup itself depend on
    /// the authorization server being up (a restart during an IdP outage must not
    /// take this service down too). Later passes are what drop a key the AS has
    /// withdrawn; see [`JWKS_BACKGROUND_REFRESH_INTERVAL`].
    pub fn spawn_background_refresh(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let mut first = true;
            loop {
                match this.refresh_now().await {
                    Ok(count) if first => info!(
                        issuer = %this.config.issuer,
                        keys = count,
                        "OAuth: authorization server signing keys loaded"
                    ),
                    Ok(count) => debug!(keys = count, "OAuth: signing keys refreshed"),
                    Err(e) => warn!(
                        issuer = %this.config.issuer,
                        error = %format!("{e:#}"),
                        "OAuth: could not load the authorization server's signing keys — \
                         tokens signed by a key this server does not already hold will be \
                         rejected (401) until a later attempt succeeds. Check \
                         mcp.oauth.issuer / mcp.oauth.jwks_uri and that this host can \
                         reach them."
                    ),
                }
                first = false;
                tokio::time::sleep(JWKS_BACKGROUND_REFRESH_INTERVAL).await;
            }
        })
    }

    /// Resolve the verification key for `kid` and `alg`, fetching or refetching the
    /// JWKS as needed.
    ///
    /// Fails closed in every failure mode — an unreachable IdP, a malformed key set,
    /// an unknown `kid` during the refetch cooldown, a key whose type cannot produce
    /// `alg` — because the alternative shape ("could not check, so allow") is the
    /// one bug in this file that would be worth a CVE.
    async fn decoding_key(
        &self,
        kid: Option<&str>,
        alg: Algorithm,
    ) -> Result<DecodingKey, TokenRejection> {
        if let Some(key) = lookup(&self.jwks.read().await.keys, kid, alg) {
            return Ok(key);
        }

        // One refresher at a time. A thundering herd of concurrent unknown-`kid`
        // requests queues HERE, not on the key lock, so requests whose key is
        // already cached are never held up by a slow IdP; `JWKS_FETCH_TIMEOUT`
        // bounds how long the queued ones wait.
        let _refreshing = self.refresh_lock.lock().await;

        // Another task may have fetched while we waited.
        let last_attempt = {
            let cache = self.jwks.read().await;
            if let Some(key) = lookup(&cache.keys, kid, alg) {
                return Ok(key);
            }
            cache.last_attempt
        };

        if let Some(last) = last_attempt
            && last.elapsed() < self.jwks_min_refetch_interval
        {
            // See `JWKS_MIN_REFETCH_INTERVAL`: `kid` comes from an unverified token
            // header, so an unknown one must not be able to schedule IdP traffic.
            return Err(TokenRejection::Invalid(format!(
                "no {alg:?} key for kid {} and the JWKS was refetched less than {}s ago",
                describe_kid(kid),
                self.jwks_min_refetch_interval.as_secs()
            )));
        }

        if let Err(e) = self.refresh().await {
            warn!(
                issuer = %self.config.issuer,
                error = %format!("{e:#}"),
                "JWKS refresh failed — tokens signed by a key we do not already hold will \
                 be rejected until the next attempt"
            );
            return Err(TokenRejection::Invalid(format!(
                "JWKS refresh failed: {e:#}"
            )));
        }

        lookup(&self.jwks.read().await.keys, kid, alg).ok_or_else(|| {
            TokenRejection::Invalid(format!(
                "no {alg:?} key for kid {} in the fetched JWKS",
                describe_kid(kid)
            ))
        })
    }

    /// One refresh attempt. The caller holds `refresh_lock`; the key lock is taken
    /// only for the instant it takes to read or swap in-memory state, never across
    /// the network. Records the attempt time first, so a failure is backed off like
    /// a success, and leaves the old keys in place on any failure.
    async fn refresh(&self) -> Result<usize> {
        let known_uri = {
            let mut cache = self.jwks.write().await;
            cache.last_attempt = Some(Instant::now());
            cache.jwks_uri.clone()
        };
        let jwks_uri = match known_uri {
            Some(uri) => uri,
            None => {
                let uri = self.discover_jwks_uri().await?;
                info!(
                    issuer = %self.config.issuer,
                    jwks_uri = %uri,
                    "OAuth: discovered the JWKS URI from the issuer's metadata"
                );
                self.jwks.write().await.jwks_uri = Some(uri.clone());
                uri
            }
        };
        let keys = self
            .fetch_jwks(&jwks_uri)
            .await
            .with_context(|| format!("fetching the JWKS from {jwks_uri}"))?;
        let count = keys.len();
        debug!(count, jwks_uri = %jwks_uri, "Fetched JWKS");
        self.jwks.write().await.keys = keys;
        Ok(count)
    }

    /// Find the JWKS URI in the issuer's own metadata (used only when
    /// `mcp.oauth.jwks_uri` is not configured).
    ///
    /// Only URLs derived from the CONFIGURED issuer are ever fetched — nothing in a
    /// token influences where this goes, so it is not an SSRF surface. The
    /// document's `issuer` must equal the configured one byte-for-byte (RFC 8414
    /// §3.3, OIDC Discovery §4.3: a mismatching document MUST NOT be used), which is
    /// what stops a proxy or a misconfigured path from handing us some other
    /// server's keys.
    async fn discover_jwks_uri(&self) -> Result<String> {
        let mut errors = Vec::new();
        for url in discovery_urls(&self.config.issuer) {
            match self.fetch_json(&url).await {
                Ok(doc) => match jwks_uri_from_metadata(&doc, &self.config.issuer) {
                    Ok(uri) => return Ok(uri),
                    Err(e) => errors.push(format!("{url}: {e:#}")),
                },
                Err(e) => errors.push(format!("{url}: {e:#}")),
            }
        }
        bail!(
            "could not discover a jwks_uri for mcp.oauth.issuer {:?} — set mcp.oauth.jwks_uri \
             explicitly or fix the issuer. Tried: {}",
            self.config.issuer,
            errors.join("; ")
        )
    }

    async fn fetch_jwks(&self, uri: &str) -> Result<Vec<CachedKey>> {
        let doc = self.fetch_json(uri).await?;
        let entries = doc
            .get("keys")
            .and_then(Value::as_array)
            .context("response is not a JWK Set (no \"keys\" array)")?;
        if entries.len() > MAX_JWKS_KEYS {
            warn!(
                published = entries.len(),
                used = MAX_JWKS_KEYS,
                "JWK Set has more keys than this server will consider; the rest are ignored"
            );
        }

        let mut keys = Vec::new();
        for entry in entries.iter().take(MAX_JWKS_KEYS) {
            // Parsed one key at a time: `jsonwebtoken::jwk::JwkSet` refuses the
            // WHOLE set if any single key has a kty/crv/alg it does not model (an
            // X25519 encryption key, say), and one exotic key must not take the
            // usable ones down with it.
            let jwk: Jwk = match serde_json::from_value(entry.clone()) {
                Ok(jwk) => jwk,
                Err(e) => {
                    debug!(error = %e, "Skipping a JWKS entry this server cannot parse");
                    continue;
                }
            };
            if let Some(key) = cached_key(&jwk, &self.config.algorithms) {
                keys.push(key);
            }
        }
        if keys.is_empty() {
            bail!(
                "the JWK Set contained no usable signature keys for mcp.oauth.algorithms {:?}",
                self.config.algorithms
            );
        }
        Ok(keys)
    }

    /// GET a JSON document with the body capped at [`MAX_FETCH_BYTES`].
    async fn fetch_json(&self, url: &str) -> Result<Value> {
        let mut resp = self
            .http
            .get(url)
            .header(reqwest::header::ACCEPT, "application/json")
            .send()
            .await
            .context("request failed")?
            .error_for_status()
            .context("non-success status")?;
        if let Some(len) = resp.content_length()
            && len > MAX_FETCH_BYTES as u64
        {
            bail!("response is {len} bytes, over the {MAX_FETCH_BYTES}-byte cap");
        }
        let mut body = Vec::new();
        while let Some(chunk) = resp.chunk().await.context("reading the response body")? {
            if body.len() + chunk.len() > MAX_FETCH_BYTES {
                bail!("response exceeds the {MAX_FETCH_BYTES}-byte cap");
            }
            body.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&body).context("response was not JSON")
    }
}

/// The union of every configured scope claim, in first-seen order, deduplicated.
///
/// Every claim is read in every shape: a string is split on whitespace (RFC 9068
/// §2.2.3's `scope`, and Entra ID's / Hydra's string `scp`), an array contributes
/// each string element whole (Authelia's and Okta's `scp`). Anything else — a
/// number, an object, a claim the token does not have — contributes nothing rather
/// than failing the token, since the only consequence of "no scopes found" is the
/// 403 below, which is the correct answer anyway.
fn extract_scopes(claims: &Map<String, Value>, claim_names: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut push = |s: &str| {
        let s = s.trim();
        if !s.is_empty() && seen.insert(s.to_string()) {
            out.push(s.to_string());
        }
    };
    for name in claim_names {
        match claims.get(name) {
            Some(Value::String(s)) => s.split_whitespace().for_each(&mut push),
            Some(Value::Array(items)) => items.iter().filter_map(Value::as_str).for_each(&mut push),
            _ => {}
        }
    }
    out
}

/// The first present, non-empty string claim among `claim_names`, truncated for
/// logging.
fn extract_principal(claims: &Map<String, Value>, claim_names: &[String]) -> Option<String> {
    claim_names.iter().find_map(|name| match claims.get(name) {
        Some(Value::String(s)) if !s.trim().is_empty() => Some(for_log(s)),
        _ => None,
    })
}

/// RFC 9068 §2.1 / §4: the header `typ` of a JWT access token is `at+jwt`
/// (`application/at+jwt` is the same media type, RFC 7515 §4.1.9, compared
/// case-insensitively). Many servers still emit `JWT` or nothing (Authentik, Entra
/// ID, Okta, Keycloak by default), so those pass unless `require_at_jwt` is on —
/// which an operator whose AS does emit `at+jwt` (Authelia, Kanidm) should turn on,
/// since it is the one check that tells an access token from an ID token minted
/// for the same client. Any OTHER explicit type (`dpop+jwt`, `logout+jwt`,
/// `secevent+jwt`...) is a different kind of JWT and is always refused.
fn check_typ(typ: Option<&str>, require_at_jwt: bool) -> Result<(), TokenRejection> {
    let Some(raw) = typ else {
        return if require_at_jwt {
            Err(TokenRejection::Invalid(
                "token header has no typ and mcp.oauth.require_at_jwt is on".into(),
            ))
        } else {
            Ok(())
        };
    };
    let lower = raw.trim().to_ascii_lowercase();
    let media = lower.strip_prefix("application/").unwrap_or(&lower);
    match media {
        "at+jwt" => Ok(()),
        "jwt" if !require_at_jwt => Ok(()),
        _ => Err(TokenRejection::Invalid(format!(
            "token typ {:?} is not accepted as an access token{}",
            for_log(raw),
            if require_at_jwt {
                " (mcp.oauth.require_at_jwt is on)"
            } else {
                ""
            }
        ))),
    }
}

/// Build a [`CachedKey`] from one JWK, or `None` when the key must not be used.
fn cached_key(jwk: &Jwk, allowed: &[Algorithm]) -> Option<CachedKey> {
    // `use: enc` keys exist in real key sets (Keycloak publishes one). An
    // encryption key verifying a signature is a key-confusion bug in waiting.
    match &jwk.common.public_key_use {
        None | Some(PublicKeyUse::Signature) => {}
        Some(_) => return None,
    }
    let mut algorithms = key_algorithms(&jwk.algorithm)?;
    // When the key names its own algorithm, that is the ONLY one it verifies. A
    // declared algorithm the key type cannot produce (an RSA key labelled ES256)
    // means the entry is broken; skipping it is safer than guessing.
    if let Some(declared) = &jwk.common.key_algorithm {
        match signing_algorithm(declared) {
            Some(alg) if algorithms.contains(&alg) => algorithms = vec![alg],
            _ => return None,
        }
    }
    algorithms.retain(|alg| allowed.contains(alg));
    if algorithms.is_empty() {
        return None;
    }
    match DecodingKey::from_jwk(jwk) {
        Ok(key) => Some(CachedKey {
            kid: jwk.common.key_id.clone(),
            key,
            algorithms,
        }),
        Err(e) => {
            warn!(
                kid = ?jwk.common.key_id.as_deref().map(for_log),
                error = %e,
                "Skipping unusable JWKS entry"
            );
            None
        }
    }
}

/// The signature algorithms a key of this type can produce. `None` for a key that
/// can never verify an access token here: symmetric (`oct`) keys above all — a
/// shared secret has no business in a public key set, and honouring one would
/// re-open the HMAC confusion that [`parse_algorithm`] closes — plus curves `ring`
/// cannot verify (P-521) and non-signature curves (X25519).
fn key_algorithms(params: &AlgorithmParameters) -> Option<Vec<Algorithm>> {
    use Algorithm::*;
    match params {
        AlgorithmParameters::RSA(_) => Some(vec![RS256, RS384, RS512, PS256, PS384, PS512]),
        AlgorithmParameters::EllipticCurve(p) => match p.curve {
            EllipticCurve::P256 => Some(vec![ES256]),
            EllipticCurve::P384 => Some(vec![ES384]),
            _ => None,
        },
        AlgorithmParameters::OctetKeyPair(p) => match p.curve {
            EllipticCurve::Ed25519 => Some(vec![EdDSA]),
            _ => None,
        },
        AlgorithmParameters::OctetKey(_) => None,
    }
}

/// A JWK `alg` as a JWS signature algorithm; `None` for HMAC and for key-management
/// (encryption) algorithms, neither of which may verify an access token.
fn signing_algorithm(alg: &KeyAlgorithm) -> Option<Algorithm> {
    Some(match alg {
        KeyAlgorithm::RS256 => Algorithm::RS256,
        KeyAlgorithm::RS384 => Algorithm::RS384,
        KeyAlgorithm::RS512 => Algorithm::RS512,
        KeyAlgorithm::PS256 => Algorithm::PS256,
        KeyAlgorithm::PS384 => Algorithm::PS384,
        KeyAlgorithm::PS512 => Algorithm::PS512,
        KeyAlgorithm::ES256 => Algorithm::ES256,
        KeyAlgorithm::ES384 => Algorithm::ES384,
        KeyAlgorithm::EdDSA => Algorithm::EdDSA,
        _ => return None,
    })
}

/// Find the key for `kid` that can verify `alg`.
///
/// A token header with no `kid` falls back to the single key that can verify its
/// `alg`, when there is exactly one. That is not laxity: with one candidate key
/// there is exactly one key the signature could have been made with, so the
/// fallback picks the same key an explicit `kid` would have. With two or more it
/// refuses rather than trying each, which would turn key rotation into a
/// signature-verification oracle.
fn lookup(keys: &[CachedKey], kid: Option<&str>, alg: Algorithm) -> Option<DecodingKey> {
    let mut candidates = keys.iter().filter(|k| k.algorithms.contains(&alg));
    match kid {
        Some(kid) => candidates
            .find(|k| k.kid.as_deref() == Some(kid))
            .map(|k| k.key.clone()),
        None => {
            let only = candidates.next()?;
            candidates.next().is_none().then(|| only.key.clone())
        }
    }
}

/// The metadata URLs to try for `issuer`, in order: OpenID Connect Discovery
/// (§4: the issuer with any trailing slash removed, plus
/// `/.well-known/openid-configuration` — where every server this project has been
/// tested against publishes, including per-application issuers like Authentik's and
/// Kanidm's), then RFC 8414 §3.1's form (the well-known segment inserted between
/// host and path).
fn discovery_urls(issuer: &str) -> Vec<String> {
    let trimmed = issuer.trim_end_matches('/');
    let mut urls = vec![format!("{trimmed}/.well-known/openid-configuration")];
    if let Some((scheme, rest)) = trimmed.split_once("://") {
        let (authority, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, ""),
        };
        let rfc8414 =
            format!("{scheme}://{authority}/.well-known/oauth-authorization-server{path}");
        if !urls.contains(&rfc8414) {
            urls.push(rfc8414);
        }
    }
    urls
}

/// Pull `jwks_uri` out of an authorization-server metadata document, refusing a
/// document for a different issuer and a key URL that would downgrade transport.
fn jwks_uri_from_metadata(doc: &Value, issuer: &str) -> Result<String> {
    let found = doc.get("issuer").and_then(Value::as_str);
    if found != Some(issuer) {
        bail!(
            "metadata issuer {} does not match mcp.oauth.issuer {issuer:?} byte-for-byte \
             (RFC 8414 §3.3 / OIDC Discovery §4.3: such a document must not be used)",
            found.map_or_else(|| "(absent)".to_string(), |f| format!("{:?}", for_log(f)))
        );
    }
    let uri = doc
        .get("jwks_uri")
        .and_then(Value::as_str)
        .context("metadata has no jwks_uri")?;
    let parsed = reqwest::Url::parse(uri).context("jwks_uri is not an absolute URL")?;
    match parsed.scheme() {
        "https" => {}
        // Plain http only when the issuer itself is plain http (a loopback test
        // setup); an https issuer must never hand us keys over http.
        "http" if issuer.starts_with("http://") => {}
        other => bail!("jwks_uri scheme {other:?} is not allowed for issuer {issuer:?}"),
    }
    Ok(uri.to_string())
}

/// Whether `url`'s host is a loopback address or `localhost`.
fn is_loopback_url(url: &str) -> bool {
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return false;
    };
    let Some(host) = parsed.host_str() else {
        return false;
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    host == "localhost"
        || host.ends_with(".localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// Truncate a token-derived string for a log line. See [`MAX_LOGGED_CHARS`].
fn for_log(s: &str) -> String {
    let mut out: String = s.chars().take(MAX_LOGGED_CHARS).collect();
    if s.chars().count() > MAX_LOGGED_CHARS {
        out.push('…');
    }
    out
}

fn describe_kid(kid: Option<&str>) -> String {
    match kid {
        Some(kid) => format!("{:?}", for_log(kid)),
        None => "(none)".to_string(),
    }
}

/// Escape a value for an HTTP `quoted-string` (RFC 9110 §5.6.4).
///
/// Every value in a `WWW-Authenticate` auth-param here is config-derived, so this
/// is defence against a typo in `config.yaml` producing a header that a client
/// parses as something other than intended — not against an attacker. Cheaper than
/// validating URLs at load time and it keeps the header well-formed regardless.
fn quoted(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Derive a resource's protected-resource metadata URL, per RFC 9728 §3: the
/// well-known segment goes between the authority and the resource's path, NOT at
/// the end. For `https://kb.example.com/mcp` that is
/// `https://kb.example.com/.well-known/oauth-protected-resource/mcp` — which is
/// also why clients probe the path-suffixed form before the bare one, and why
/// `server::assemble_router` registers both.
fn resource_metadata_url(resource: &str) -> String {
    let trimmed = resource.trim();
    let Some((scheme, rest)) = trimmed.split_once("://") else {
        // Not a URL we can take apart. `Config::resolve` refuses anything that is
        // not an absolute http(s) URL, so this is unreachable from config; append
        // rather than panic, so a bad value surfaces as a discovery 404 with the
        // offending string visible in the metadata document, not as a crash.
        return format!(
            "{}{PROTECTED_RESOURCE_METADATA_PREFIX}",
            trimmed.trim_end_matches('/')
        );
    };
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], rest[i..].trim_end_matches('/')),
        None => (rest, ""),
    };
    format!("{scheme}://{authority}{PROTECTED_RESOURCE_METADATA_PREFIX}{path}")
}

#[cfg(test)]
pub(crate) mod testing {
    //! Shared JWT/JWKS fixtures. `pub(crate)` because `server.rs`'s middleware and
    //! router tests need to mint the same tokens this module's own tests do, and a
    //! second copy of a throwaway keypair in another file is a copy that drifts.

    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use crate::config::ResolvedOAuthConfig;

    /// Throwaway 2048-bit RSA keypair, generated for this test suite and used
    /// nowhere else. `KEY_B` exists only to produce a signature that `KEY_A`'s
    /// public half must reject.
    pub const KID_A: &str = "test-key-a";
    pub const N_A: &str = "zXtrd9E8iuVecx_7KN0nxRV0m0DgZayGgW5D4bPJMwUcFX6SIsyYpSCAGjT1Fia85xH-YrMxk9XSjuMpYB8GphQ5NitAaVx8CQeoVQw8WEi1YSG53OfuSftmkX79D48nVP6VxKq3JW_RIaTM8xsisVV2zzFeQVN_NsFNCAsClYoXLUj8Wfc9WsFz8DszbQep6I4gceD6WNCs72AQMXR5vIOfGxK5eP5JWOjK7FN95njVNbXY6p5QUQii_3HkFSDQv9drzpzeKXdDziFdSG5qZfMwGuqjfCMDNfwYKxC4AbAGbtSTCHFEWe0CuWX95xgqvyJCsVjkh8xMz-WpPoWLSQ";

    pub const KEY_A_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDNe2t30TyK5V5z
H/so3SfFFXSbQOBlrIaBbkPhs8kzBRwVfpIizJilIIAaNPUWJrznEf5iszGT1dKO
4ylgHwamFDk2K0BpXHwJB6hVDDxYSLVhIbnc5+5J+2aRfv0PjydU/pXEqrclb9Eh
pMzzGyKxVXbPMV5BU382wU0ICwKVihctSPxZ9z1awXPwOzNtB6nojiBx4PpY0Kzv
YBAxdHm8g58bErl4/klY6MrsU33meNU1tdjqnlBRCKL/ceQVINC/12vOnN4pd0PO
IV1Ibmpl8zAa6qN8IwM1/BgrELgBsAZu1JMIcURZ7QK5Zf3nGCq/IkKxWOSHzEzP
5ak+hYtJAgMBAAECggEAPpyQWxKZGZOZi4ffroxw3VdT0CjdF24SECdKrN/s+0xf
ydbm7Y6dJpe4IQQo+AZ2wgwUEPwcK7lYLuzeAymBC6MW6cAVIOWq789zBfM0Agyp
o/60VTEgxU9C6iuhLZgHupjWhvYj11byiQdf4eXPVOy/RpP67fnkxgjxkXVVZL4C
zJ5KQZRLi+DH9l5Vd5nKqyRVVFVaaD0ws5Lw7n2HBrraq/omV6FlcIkePB4Tx2gD
WudBhUPnrhukXaoEWEvBNXnVSExU+bZMeWvQdcGVL6OE1LG9IqsjYiumF2kb0n0L
ZTalPbtAHoNDEKIG2+rwCqsLBZQvFnLcFTlc1WAExwKBgQDr8IjWaZ8I0mQHbRhM
BsrLDjf2qBVclBMchYafmh6NoI5E2+928NTo/uxssGTEX4ce0v6CW6dHo8vNVfN/
cUxjQW479qvugi8EBp6rQ9ZOjSra078L145jTCaJLfxMuYDgjycvcv8wOqxJL/80
F1Qjkn+pGsQFvjCAEikjwcG0cwKBgQDe8/J8W2tPpnRiv40T2Hgu4Th8gRfm/79k
RBZqeiO/EiI9zwHmOj9s02fK3tBPyQgZMyQyQNMJjFuWzCjHE8gaBdLAqGrpILL2
jR7EvXBPrGRpWcbnRCyODURcaIY1dTVImT1g9rhwzDJNQFd7XPGR60LCMUxL8b2p
hlgFlw9OUwKBgFAqvpP792mL8ykCzIqolCdCgYlxuzBlr8i1JfT87Py6XRzQjiEf
23f/hl234cVHoCW9E3U/pysUYJ84YTAgUxA2nzoIqoqz+T2o8ijHN/4gwTrxT6y6
ZUsgCMf7tAptzXh/q5TXwhWlGf0ULeaJNrGPiYjv60L4SIp7oTbhEuw5AoGAdJg8
yn4Am7HgEbg87hD5oQKVSL82Ic7DZ4sX8e0X/pdcIti8FIuHmcDg+b4WUHNAcfVF
y6YM92RYjX8NIDcfIUTEV458ApjgHoHkglzTfEcaZ+HUXCNR7aPQiUb8UL6P8/x3
ldrQz+Rpte6dEV2k03ul+OpRDTJJznr8U0gRcBMCgYB/aUF16/RJvW5nLGWTbAS8
D4d9SgETq0P0zbuDUk60Fk6kQbQ+bwX+ffgsEP/P/CFTNJ+opoCo0/6uK8WlKs15
uVx1QLn2oATcEUusHESeflBUSSaYlhHXFL7ahvAgBs3vzgWZnUVnz2A3QDCiLu6H
EmjUKNFGC0zInLUM1Cbu9w==
-----END PRIVATE KEY-----
";

    pub const KEY_B_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQClweugXTF1SY0q
ar8Z68ong9eCzOI3kCSipuiCDhVPad8Gn4be0RM4B7t342iuG4UjyXnCQpCoWGiN
L4KN52hFBE7M8c/7JutAtmpJFm33cFKZ+yWfAcX5FFtC/BdOPfaPtije98QJRmlv
lJ6n7c8uMpXhtV1ZIqwm9g7chVWlUHKAgMFGaUeKWdksQ9tTZgDKeHO1vfRZZlYT
XdvDpNe7Dxz1o3eefTsrKsE1DDTXrDfJPUDPPpBTMmT+xrPRehuNqrNQRUJWEIAR
bJpV5ltnhNX4zs3YQ39/XTCcQjnbu4wRpDUgTIhPuomg18t8vqi1CbhaN8+Ww3oU
FRdXKE59AgMBAAECggEAMeZ2umDD3mTFmCLpo/KNeabhrrFiWsrMlKC9t0VpGe6r
4xEMZ7C2YfRF9hoibePACZ2CR76FUQDIfNR0L6ceB0T8OguECr5VLTadOaKEeWy5
mTx3v24nvMvpi3lbxMS3oNz8Yd9iB07It/wcZT6c0/ILmBbi4s4i2FnT8IQ9W9Ym
iuMwqujeyrEUG/O3HrUJLHNe6PwJj6s8mbAKxfmCqnDLCyWlQejR2FL24mriCSQD
G03gZ6VazAnDt19SjToPKH1e6XjB6FySUX3gDhA9yXSPphCdaa9Ov9w/4UktlEtz
RRobV9e+e5e2qUrv77CZu3PMlH/gZARM/ncSCKlYGwKBgQDqVHUYvBArQWLO2po6
9SvjcvB631+cUO94k+nV4vlbXbzjMznXPPShijf5cirzRhazSRqxp5Yt6959vVDI
Pe/vjP0lM0dLZwiOnsEaq+ArEoAnidD09bUn81qMnsH3eUpXtIthB5ltkNX3tXb2
Xs04OxDm4IrSoMg7/w2HXb4YbwKBgQC1FhMz8PvXDAb7uJ5ydjd4Gw1CEqw93YSf
S1koX2x86qhELfUglAHb+h5RHxE7zqi5fqzrsHl3Ow3392O3clcqrbME4u//fvmV
XCr7eHraeIByX/ZpnBiiuYjrvN6MKDUy00yBDdGMGGA0JT2+aH/06qg/G5xG8SiV
0ajx8wAF0wKBgQDnbTQckpfRcIk6TBFoSvzmbHzujS9rPU/UoRifAcRNpP1I0i28
0lm0NMLlXAjpLH584KU5cY7TmZCqVE+1A960kmTs2YD/CiocWNPUGI2TXHkvE2BI
nWYlp6T1HlHorGRszEWfNZck65c2RoTP+37omwUtT/Qq41n+Tv44g6+bhwKBgGyQ
EW8gWDsycLVUl1lT2ildPnOQMkbcmPfO+mKj4qx5GevWCZFAamTw7GAB2hka6jha
41xhblC2zMcOP2/pUqy5egvB6dQo0YRjvzkHn89+UrM/KMFj3bkgth9uGZW5PTt9
Re5Q1IHC01ovwXZ3u86fJ8K90NEPHx/ClCCJaEgVAoGAHlopDQ/w7JN5sCBYDZeE
eAfND1Q/hnbfjdUgg13/Qmhqwm86RYJ3E9mxjcCNKZ3hNX3Xcs5NW5oC9tj9Nb9G
B5bK2earcA3sKw66Uvzd5AtypET7/RPOSgpXOD34f1RN38fWqc+L0pdHZ41D5eif
13kE7LEf//HMi5ix93dRdZw=
-----END PRIVATE KEY-----
";

    /// Throwaway P-256 and Ed25519 keypairs (PKCS#8), generated for this test
    /// suite with `openssl genpkey` and used nowhere else, with the public halves
    /// as JWK coordinates. They exist so ES256 (Kanidm's default) and EdDSA tokens
    /// are exercised end to end, not just RS256.
    pub const KID_EC: &str = "test-key-ec";
    pub const EC_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgt+Eh+ZhHxw1rLcOh
VFTMghCKj2Vjq7F3zWwemIamL62hRANCAATCkjHQ5M6RrM1TPQ6wuvqltcwRa4AL
s/Jd92N5PXaKwn94PezTTY6vFt/ivjcfSSG5wWncUlc92lsipOXRqgLZ
-----END PRIVATE KEY-----
";
    pub const EC_X: &str = "wpIx0OTOkazNUz0OsLr6pbXMEWuAC7PyXfdjeT12isI";
    pub const EC_Y: &str = "f3g97NNNjq8W3-K-Nx9JIbnBadxSVz3aWyKk5dGqAtk";

    pub const KID_ED: &str = "test-key-ed";
    pub const ED_PEM: &str = "-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEICpZSYX0J1AafpNnoaSXF7Lm/Nmt73HqecXyoFLjchf8
-----END PRIVATE KEY-----
";
    pub const ED_X: &str = "C4HUUU7zy0ZEyY__PV16YbPgh4b4clhBg0oVMg0_EaQ";

    pub const ISSUER: &str = "https://authentik.example.test/application/o/mcp-kb-rag/";
    pub const AUDIENCE: &str = "test-client-id";
    pub const RESOURCE: &str = "https://kb.example.test/mcp";

    /// The JWK for `KEY_A`'s public half, as an AS would serve it.
    pub fn jwk_rsa_a() -> serde_json::Value {
        serde_json::json!({
            "kty": "RSA", "use": "sig", "alg": "RS256", "kid": KID_A, "n": N_A, "e": "AQAB",
        })
    }

    /// The same RSA key with no `alg`, so it may verify any RS*/PS* algorithm.
    pub fn jwk_rsa_a_any_alg(kid: &str) -> serde_json::Value {
        serde_json::json!({"kty": "RSA", "use": "sig", "kid": kid, "n": N_A, "e": "AQAB"})
    }

    pub fn jwk_ec() -> serde_json::Value {
        serde_json::json!({
            "kty": "EC", "crv": "P-256", "use": "sig", "alg": "ES256", "kid": KID_EC,
            "x": EC_X, "y": EC_Y,
        })
    }

    pub fn jwk_ed() -> serde_json::Value {
        serde_json::json!({
            "kty": "OKP", "crv": "Ed25519", "use": "sig", "alg": "EdDSA", "kid": KID_ED,
            "x": ED_X,
        })
    }

    pub fn jwks_of(keys: Vec<serde_json::Value>) -> String {
        serde_json::json!({ "keys": keys }).to_string()
    }

    /// A JWK Set carrying only `KEY_A`'s public half — the shape the original
    /// suite (and Authentik) uses.
    pub fn jwks_body() -> String {
        jwks_of(vec![jwk_rsa_a()])
    }

    /// Every test key: RSA A (RS256-labelled), a PS256-capable copy of it, the
    /// P-256 key and the Ed25519 key.
    pub fn jwks_body_all() -> String {
        jwks_of(vec![
            jwk_rsa_a(),
            jwk_rsa_a_any_alg("test-key-a-pss"),
            jwk_ec(),
            jwk_ed(),
        ])
    }

    /// The config the original suite used, plus every new key at its default.
    pub fn resolved_config(jwks_uri: &str) -> ResolvedOAuthConfig {
        ResolvedOAuthConfig {
            issuer: ISSUER.to_string(),
            jwks_uri: jwks_uri.to_string(),
            audience: AUDIENCE.to_string(),
            audiences: Vec::new(),
            resource: RESOURCE.to_string(),
            required_scope: "mcp:read".to_string(),
            scopes_supported: vec!["mcp:read".to_string(), "mcp:write".to_string()],
            scope_claims: super::DEFAULT_SCOPE_CLAIMS
                .iter()
                .map(|s| s.to_string())
                .collect(),
            principal_claims: super::DEFAULT_PRINCIPAL_CLAIMS
                .iter()
                .map(|s| s.to_string())
                .collect(),
            algorithms: super::DEFAULT_ALGORITHMS
                .iter()
                .map(|s| super::parse_algorithm(s).unwrap())
                .collect(),
            leeway_secs: super::DEFAULT_LEEWAY_SECS,
            require_at_jwt: false,
            accept_static_bearer: true,
        }
    }

    /// Seconds since the epoch, for `exp`.
    pub fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Mint a token with full control over every field a test might want wrong.
    /// `pem` is which key signs it; `kid` is what the header *claims* signed it —
    /// letting a test say "signed by B, labelled A" for the bad-signature case.
    pub fn mint(pem: &str, kid: &str, claims: serde_json::Value) -> String {
        let mut header = Header::new(Algorithm::RS256);
        header.kid = Some(kid.to_string());
        encode(
            &header,
            &claims,
            &EncodingKey::from_rsa_pem(pem.as_bytes()).unwrap(),
        )
        .unwrap()
    }

    /// Mint with an explicit algorithm, `kid` and `typ`. RS*/PS* sign with
    /// `KEY_A`, ES256 with the P-256 key, EdDSA with the Ed25519 key.
    pub fn mint_with(
        alg: Algorithm,
        kid: Option<&str>,
        typ: Option<&str>,
        claims: serde_json::Value,
    ) -> String {
        let mut header = Header::new(alg);
        header.kid = kid.map(str::to_string);
        header.typ = typ.map(str::to_string);
        let key = match alg {
            Algorithm::RS256
            | Algorithm::RS384
            | Algorithm::RS512
            | Algorithm::PS256
            | Algorithm::PS384
            | Algorithm::PS512 => EncodingKey::from_rsa_pem(KEY_A_PEM.as_bytes()).unwrap(),
            Algorithm::ES256 => EncodingKey::from_ec_pem(EC_PEM.as_bytes()).unwrap(),
            Algorithm::EdDSA => EncodingKey::from_ed_pem(ED_PEM.as_bytes()).unwrap(),
            other => panic!("no test key for {other:?}"),
        };
        encode(&header, &claims, &key).unwrap()
    }

    /// The happy-path token: right key, right issuer, right audience, valid for an
    /// hour, carrying `mcp:read mcp:write`.
    pub fn valid_token() -> String {
        mint(
            KEY_A_PEM,
            KID_A,
            serde_json::json!({
                "iss": ISSUER,
                "aud": AUDIENCE,
                "azp": AUDIENCE,
                "sub": "user-1",
                "exp": now() + 3600,
                "scope": "mcp:read mcp:write",
            }),
        )
    }

    /// A throwaway loopback HTTP server, counting hits. Hand-rolled for the same
    /// reason `rerank.rs`'s `FakeReranker` is: the repo has no HTTP-mock
    /// dev-dependency and this is cheaper than adding one.
    ///
    /// `routes` maps a request path to `(status line, body)` and can be changed
    /// while the server runs (key rotation tests); a path with no route gets
    /// `fallback`, or a 404 when there is none.
    pub struct FakeJwksServer {
        /// `http://127.0.0.1:<port>/jwks` — kept for the original call sites.
        pub url: String,
        /// `http://127.0.0.1:<port>`, for building issuer and discovery URLs.
        pub base: String,
        pub hits: Arc<AtomicUsize>,
        pub routes: Arc<Mutex<HashMap<String, (&'static str, String)>>>,
        /// Milliseconds to stall before answering — a slow IdP.
        pub delay_ms: Arc<AtomicU64>,
    }

    /// Answer every request, whatever its path, with one canned response.
    pub async fn spawn_jwks_server(status_line: &'static str, body: String) -> FakeJwksServer {
        spawn_http_server(HashMap::new(), Some((status_line, body))).await
    }

    pub async fn spawn_http_server(
        routes: HashMap<String, (&'static str, String)>,
        fallback: Option<(&'static str, String)>,
    ) -> FakeJwksServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&hits);
        let routes = Arc::new(Mutex::new(routes));
        let shared_routes = Arc::clone(&routes);
        let fallback = Arc::new(fallback);
        let delay_ms = Arc::new(AtomicU64::new(0));
        let shared_delay = Arc::clone(&delay_ms);

        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let counter = Arc::clone(&counter);
                let routes = Arc::clone(&shared_routes);
                let fallback = Arc::clone(&fallback);
                let delay = shared_delay.load(Ordering::SeqCst);
                tokio::spawn(async move {
                    if delay > 0 {
                        tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                    }
                    // A GET has no body, so end-of-headers is end-of-request.
                    let mut buf = Vec::new();
                    let mut tmp = [0u8; 4096];
                    loop {
                        match sock.read(&mut tmp).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => buf.extend_from_slice(&tmp[..n]),
                        }
                        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                            break;
                        }
                    }
                    counter.fetch_add(1, Ordering::SeqCst);
                    let request = String::from_utf8_lossy(&buf);
                    let path = request
                        .lines()
                        .next()
                        .and_then(|line| line.split_whitespace().nth(1))
                        .unwrap_or("/")
                        .to_string();
                    let (status_line, body) = routes
                        .lock()
                        .unwrap()
                        .get(&path)
                        .cloned()
                        .or_else(|| (*fallback).clone())
                        .unwrap_or(("404 Not Found", "{}".to_string()));
                    let resp = format!(
                        "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\n\
                         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                    let _ = sock.flush().await;
                });
            }
        });

        FakeJwksServer {
            url: format!("http://{addr}/jwks"),
            base: format!("http://{addr}"),
            hits,
            routes,
            delay_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::Ordering;

    fn oauth_config(jwks_uri: &str) -> ResolvedOAuthConfig {
        resolved_config(jwks_uri)
    }

    /// Zero cooldown: a test that wants to observe a refetch should not have to
    /// sleep out `JWKS_MIN_REFETCH_INTERVAL`.
    fn validator_no_cooldown(jwks_uri: &str) -> OAuthValidator {
        OAuthValidator::build(&oauth_config(jwks_uri), Duration::ZERO).unwrap()
    }

    fn validator(jwks_uri: &str) -> OAuthValidator {
        OAuthValidator::new(&oauth_config(jwks_uri)).unwrap()
    }

    fn validator_with(cfg: ResolvedOAuthConfig) -> OAuthValidator {
        OAuthValidator::new(&cfg).unwrap()
    }

    fn claims(extra: serde_json::Value) -> serde_json::Value {
        let mut base = serde_json::json!({
            "iss": ISSUER, "aud": AUDIENCE, "sub": "user-1", "exp": now() + 3600,
        });
        for (k, v) in extra.as_object().unwrap() {
            base[k] = v.clone();
        }
        base
    }

    fn is_invalid<T: std::fmt::Debug>(r: &Result<T, TokenRejection>) -> bool {
        matches!(r, Err(TokenRejection::Invalid(_)))
    }

    // ── metadata URL derivation (RFC 9728 §3) ────────────────────────────────

    #[test]
    fn metadata_url_splices_the_well_known_segment_before_the_path() {
        // The well-known segment goes between authority and path, NOT appended.
        assert_eq!(
            resource_metadata_url("https://kb.example.com/mcp"),
            "https://kb.example.com/.well-known/oauth-protected-resource/mcp"
        );
    }

    #[test]
    fn metadata_url_for_a_path_less_resource_is_the_bare_well_known() {
        assert_eq!(
            resource_metadata_url("https://kb.example.com"),
            "https://kb.example.com/.well-known/oauth-protected-resource"
        );
        assert_eq!(
            resource_metadata_url("https://kb.example.com/"),
            "https://kb.example.com/.well-known/oauth-protected-resource"
        );
    }

    #[test]
    fn metadata_url_keeps_a_port_and_drops_a_trailing_slash() {
        assert_eq!(
            resource_metadata_url("http://localhost:8001/mcp/"),
            "http://localhost:8001/.well-known/oauth-protected-resource/mcp"
        );
    }

    #[test]
    fn metadata_url_of_a_malformed_resource_does_not_panic() {
        // `Config::resolve` refuses a non-URL `resource`, so this is unreachable
        // from config — but the function itself must still degrade, not panic.
        assert_eq!(
            resource_metadata_url("kb.example.com/mcp"),
            "kb.example.com/mcp/.well-known/oauth-protected-resource"
        );
    }

    // ── the metadata document and the challenge headers ──────────────────────

    #[test]
    fn metadata_document_has_the_rfc_9728_shape() {
        let v = validator("http://127.0.0.1:1/jwks");
        let doc = v.metadata();
        assert_eq!(doc["resource"], RESOURCE);
        // Byte-identical, trailing slash and all — a client matches this against
        // the `iss` of the tokens it receives.
        assert_eq!(doc["authorization_servers"][0], ISSUER);
        assert_eq!(doc["scopes_supported"][0], "mcp:read");
        assert_eq!(doc["scopes_supported"][1], "mcp:write");
        assert_eq!(doc["bearer_methods_supported"][0], "header");
        assert!(doc["resource_name"].is_string());
    }

    #[test]
    fn invalid_token_challenge_is_well_formed() {
        let v = validator("http://127.0.0.1:1/jwks");
        assert_eq!(
            v.invalid_token_challenge(),
            "Bearer error=\"invalid_token\", \
             resource_metadata=\"https://kb.example.test/.well-known/oauth-protected-resource/mcp\", \
             scope=\"mcp:read mcp:write\""
        );
    }

    #[test]
    fn insufficient_scope_challenge_names_the_missing_scope_not_the_menu() {
        let v = validator("http://127.0.0.1:1/jwks");
        assert_eq!(
            v.insufficient_scope_challenge(),
            "Bearer error=\"insufficient_scope\", scope=\"mcp:read\", \
             resource_metadata=\"https://kb.example.test/.well-known/oauth-protected-resource/mcp\""
        );
    }

    #[test]
    fn challenge_values_are_escaped_not_pasted() {
        assert_eq!(quoted(r#"a"b\c"#), r#"a\"b\\c"#);
    }

    // ── algorithm allowlist parsing ──────────────────────────────────────────

    #[test]
    fn hmac_and_none_can_never_be_configured() {
        for bad in [
            "HS256", "HS384", "HS512", "none", "None", "ES512", "rs256", "",
        ] {
            assert!(parse_algorithm(bad).is_err(), "{bad:?} must be refused");
        }
        assert!(
            parse_algorithm("HS256")
                .unwrap_err()
                .contains("key-confusion")
        );
        for good in DEFAULT_ALGORITHMS {
            assert!(parse_algorithm(good).is_ok(), "{good} must parse");
        }
    }

    // ── token validation: the happy path and the original checks ─────────────

    #[tokio::test]
    async fn a_well_formed_token_is_accepted_and_yields_its_scopes() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        let token = v.validate(&valid_token()).await.unwrap();
        assert_eq!(token.subject.as_deref(), Some("user-1"));
        assert_eq!(token.scopes, vec!["mcp:read", "mcp:write"]);
        assert!(token.has_scope("mcp:write"));
    }

    #[tokio::test]
    async fn an_empty_credential_is_missing_not_invalid() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        assert_eq!(v.validate("").await.unwrap_err(), TokenRejection::Missing);
        assert_eq!(jwks.hits.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn a_wrong_issuer_is_rejected() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        // Same issuer minus the trailing slash: the near-miss that actually happens
        // in practice, not an obviously foreign string.
        let token = mint(
            KEY_A_PEM,
            KID_A,
            claims(serde_json::json!({
                "iss": ISSUER.trim_end_matches('/'), "scope": "mcp:read",
            })),
        );
        assert!(is_invalid(&v.validate(&token).await));
    }

    #[tokio::test]
    async fn an_issuer_array_containing_the_right_issuer_is_rejected() {
        // jsonwebtoken on its own accepts this; `iss` is a single StringOrURI.
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        let token = mint(
            KEY_A_PEM,
            KID_A,
            claims(serde_json::json!({
                "iss": ["https://evil.example.test/", ISSUER], "scope": "mcp:read",
            })),
        );
        assert!(is_invalid(&v.validate(&token).await));
    }

    #[tokio::test]
    async fn a_missing_issuer_or_audience_is_rejected() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        // jsonwebtoken only checks iss/aud when the claim is present, so omitting
        // them entirely is the way a token would sneak past a validator that had
        // not set `required_spec_claims`.
        for claims in [
            serde_json::json!({"aud": AUDIENCE, "exp": now() + 3600, "scope": "mcp:read"}),
            serde_json::json!({"iss": ISSUER, "exp": now() + 3600, "scope": "mcp:read"}),
        ] {
            let token = mint(KEY_A_PEM, KID_A, claims);
            assert!(is_invalid(&v.validate(&token).await));
        }
    }

    // ── audience ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn aud_is_accepted_as_a_string_and_as_an_array() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        for aud in [
            serde_json::json!(AUDIENCE),
            serde_json::json!(["some-other-client", AUDIENCE]),
        ] {
            let token = mint(
                KEY_A_PEM,
                KID_A,
                claims(serde_json::json!({"aud": aud, "scope": "mcp:read"})),
            );
            assert!(
                v.validate(&token).await.is_ok(),
                "aud must be accepted in both RFC 7519 §4.1.3 shapes"
            );
        }
    }

    #[tokio::test]
    async fn a_wrong_empty_or_malformed_audience_is_rejected() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        for aud in [
            serde_json::json!("some-other-client"),
            serde_json::json!([]),
            serde_json::json!(["some-other-client"]),
            serde_json::json!(42),
            serde_json::json!([AUDIENCE, 42]),
            serde_json::json!(""),
        ] {
            let token = mint(
                KEY_A_PEM,
                KID_A,
                claims(serde_json::json!({"aud": aud, "scope": "mcp:read"})),
            );
            assert!(
                is_invalid(&v.validate(&token).await),
                "aud {aud} must never be accepted"
            );
        }
    }

    #[tokio::test]
    async fn every_configured_audience_is_accepted_and_nothing_else() {
        // `audience` (legacy single key) + `audiences` (list) are unioned: the
        // migration from client_id to resource-URL audience can run with both.
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let mut cfg = oauth_config(&jwks.url);
        cfg.audiences = vec![RESOURCE.to_string()];
        let v = validator_with(cfg);
        for aud in [AUDIENCE, RESOURCE] {
            let token = mint(
                KEY_A_PEM,
                KID_A,
                claims(serde_json::json!({"aud": aud, "scope": "mcp:read"})),
            );
            assert!(v.validate(&token).await.is_ok(), "{aud} is configured");
        }
        let token = mint(
            KEY_A_PEM,
            KID_A,
            claims(
                serde_json::json!({"aud": "https://other.example.test/mcp", "scope": "mcp:read"}),
            ),
        );
        assert!(is_invalid(&v.validate(&token).await));
    }

    // ── expiry, not-before and clock skew ────────────────────────────────────

    #[tokio::test]
    async fn an_expired_token_is_rejected_beyond_the_leeway() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        let token = mint(
            KEY_A_PEM,
            KID_A,
            claims(serde_json::json!({
                "exp": now() - (DEFAULT_LEEWAY_SECS + 60), "scope": "mcp:read",
            })),
        );
        assert!(is_invalid(&v.validate(&token).await));
    }

    #[tokio::test]
    async fn skew_within_the_leeway_is_tolerated_and_zero_leeway_is_strict() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let just_expired = mint(
            KEY_A_PEM,
            KID_A,
            claims(serde_json::json!({"exp": now() - 10, "scope": "mcp:read"})),
        );
        let not_yet_valid = mint(
            KEY_A_PEM,
            KID_A,
            claims(serde_json::json!({"nbf": now() + 10, "scope": "mcp:read"})),
        );

        let lenient = validator(&jwks.url);
        assert!(lenient.validate(&just_expired).await.is_ok());
        assert!(lenient.validate(&not_yet_valid).await.is_ok());

        let mut cfg = oauth_config(&jwks.url);
        cfg.leeway_secs = 0;
        let strict = validator_with(cfg);
        assert!(is_invalid(&strict.validate(&just_expired).await));
        assert!(is_invalid(&strict.validate(&not_yet_valid).await));
    }

    #[tokio::test]
    async fn a_token_used_before_nbf_is_rejected_beyond_the_leeway() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        let token = mint(
            KEY_A_PEM,
            KID_A,
            claims(serde_json::json!({
                "nbf": now() + DEFAULT_LEEWAY_SECS + 120, "scope": "mcp:read",
            })),
        );
        assert!(is_invalid(&v.validate(&token).await));
    }

    #[tokio::test]
    async fn a_token_signed_by_the_wrong_key_is_rejected() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        // Signed by B but LABELLED as A, so the lookup succeeds and the failure is
        // genuinely a signature failure rather than an unknown-kid failure.
        let token = mint(
            KEY_B_PEM,
            KID_A,
            claims(serde_json::json!({"scope": "mcp:read"})),
        );
        assert!(is_invalid(&v.validate(&token).await));
    }

    // ── scope extraction: every shape ────────────────────────────────────────

    async fn scopes_of(extra: serde_json::Value) -> Result<AuthorizedToken, TokenRejection> {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        v.validate(&mint(KEY_A_PEM, KID_A, claims(extra))).await
    }

    #[tokio::test]
    async fn scope_as_a_space_delimited_string_is_read() {
        let t = scopes_of(serde_json::json!({"scope": "openid  mcp:read\tmcp:write"}))
            .await
            .unwrap();
        assert_eq!(t.scopes, ["openid", "mcp:read", "mcp:write"]);
    }

    #[tokio::test]
    async fn scp_as_an_array_is_read() {
        // Authelia's shape — the incompatibility the `scp` fallback fixes.
        let t = scopes_of(serde_json::json!({"scp": ["mcp:read", "mcp:write"]}))
            .await
            .unwrap();
        assert_eq!(t.scopes, ["mcp:read", "mcp:write"]);
    }

    #[tokio::test]
    async fn scp_as_a_space_delimited_string_is_read() {
        // Entra ID's (and Ory Hydra's `scope_claim: string`) shape.
        let t = scopes_of(serde_json::json!({"scp": "mcp:read mcp:write"}))
            .await
            .unwrap();
        assert_eq!(t.scopes, ["mcp:read", "mcp:write"]);
    }

    #[tokio::test]
    async fn scope_and_scp_together_are_unioned_without_duplicates() {
        let t = scopes_of(serde_json::json!({
            "scope": "openid mcp:read", "scp": ["mcp:read", "mcp:write"],
        }))
        .await
        .unwrap();
        assert_eq!(t.scopes, ["openid", "mcp:read", "mcp:write"]);
    }

    #[tokio::test]
    async fn the_required_scope_in_scp_alone_satisfies_the_check() {
        let t = scopes_of(serde_json::json!({"scope": "openid", "scp": ["mcp:read"]}))
            .await
            .unwrap();
        assert!(t.has_scope("mcp:read"));
    }

    #[tokio::test]
    async fn neither_claim_or_non_string_shapes_are_insufficient_not_invalid() {
        for extra in [
            serde_json::json!({}),
            serde_json::json!({"scope": ""}),
            serde_json::json!({"scope": "openid profile"}),
            serde_json::json!({"scp": []}),
            serde_json::json!({"scp": [1, {"mcp:read": true}]}),
            serde_json::json!({"scope": {"mcp:read": true}}),
            // Scope matching is exact and case-sensitive (RFC 6749 §3.3).
            serde_json::json!({"scope": "MCP:READ mcp:read:extra"}),
        ] {
            assert_eq!(
                scopes_of(extra.clone()).await.unwrap_err(),
                TokenRejection::InsufficientScope,
                "{extra} — the token itself is fine; conflating this with \
                 invalid_token sends the client round the authorization flow to the \
                 same refusal"
            );
        }
    }

    #[tokio::test]
    async fn only_the_configured_scope_claims_are_read() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let mut cfg = oauth_config(&jwks.url);
        cfg.scope_claims = vec!["scope".to_string()];
        let v = validator_with(cfg);
        let token = mint(
            KEY_A_PEM,
            KID_A,
            claims(serde_json::json!({"scp": ["mcp:read"]})),
        );
        assert_eq!(
            v.validate(&token).await.unwrap_err(),
            TokenRejection::InsufficientScope
        );
    }

    // ── principal ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn the_principal_is_the_first_present_claim_of_the_chain() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let mut cfg = oauth_config(&jwks.url);
        cfg.principal_claims = vec!["preferred_username".into(), "email".into(), "sub".into()];
        let v = validator_with(cfg);
        for (extra, expected) in [
            (
                serde_json::json!({"preferred_username": "alice", "email": "a@example.com"}),
                "alice",
            ),
            (
                serde_json::json!({"preferred_username": "", "email": "a@example.com"}),
                "a@example.com",
            ),
            (serde_json::json!({"preferred_username": 7}), "user-1"),
        ] {
            let mut c = claims(extra);
            c["scope"] = "mcp:read".into();
            let t = v.validate(&mint(KEY_A_PEM, KID_A, c)).await.unwrap();
            assert_eq!(t.principal.as_deref(), Some(expected));
        }
    }

    #[test]
    fn logged_values_are_truncated() {
        let long = "x".repeat(MAX_LOGGED_CHARS * 3);
        assert_eq!(for_log(&long).chars().count(), MAX_LOGGED_CHARS + 1);
        assert_eq!(for_log("short"), "short");
    }

    // ── algorithm and key confusion ──────────────────────────────────────────

    #[tokio::test]
    async fn alg_none_is_rejected_before_any_jwks_fetch() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        // Hand-assembled (no crate will sign `none`): base64url of
        // `{"alg":"none","typ":"JWT"}` / `{"alg":"None"}`, a payload with a
        // plausible claim set, and an empty signature.
        let payload = "eyJpc3MiOiJ4IiwiYXVkIjoidGVzdC1jbGllbnQtaWQiLCJzY29wZSI6Im1jcDpyZWFkIiwiZXhwIjo5OTk5OTk5OTk5fQ";
        for header in ["eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0", "eyJhbGciOiJOb25lIn0"] {
            let token = format!("{header}.{payload}.");
            assert!(is_invalid(&v.validate(&token).await), "{header}");
        }
        assert_eq!(jwks.hits.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn hs256_signed_with_the_public_key_is_rejected_before_any_jwks_fetch() {
        // The classic confusion: an attacker HMACs a token with the server's
        // PUBLIC key bytes and hopes the verifier treats them as the HMAC secret.
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        let published = jwks_body();
        for secret in [N_A.as_bytes(), published.as_bytes()] {
            let mut header = jsonwebtoken::Header::new(Algorithm::HS256);
            header.kid = Some(KID_A.to_string());
            let token = jsonwebtoken::encode(
                &header,
                &claims(serde_json::json!({"scope": "mcp:read"})),
                &jsonwebtoken::EncodingKey::from_secret(secret),
            )
            .unwrap();
            assert!(is_invalid(&v.validate(&token).await));
        }
        assert_eq!(
            jwks.hits.load(Ordering::SeqCst),
            0,
            "a junk algorithm must not be able to schedule IdP traffic"
        );
    }

    #[tokio::test]
    async fn a_symmetric_key_in_the_jwks_is_never_used() {
        // Even a key set that (wrongly) publishes an `oct` key cannot make HMAC
        // verification reachable: the key is dropped at load, and HS* is not
        // configurable anyway.
        let body = jwks_of(vec![
            serde_json::json!({"kty": "oct", "kid": KID_A, "k": "c2VjcmV0"}),
        ]);
        let jwks = spawn_jwks_server("200 OK", body).await;
        let v = validator(&jwks.url);
        assert!(is_invalid(&v.validate(&valid_token()).await));
    }

    #[tokio::test]
    async fn a_token_alg_the_named_key_cannot_produce_is_rejected() {
        // Header says ES256 but names the RSA key: the key's type pins it to
        // RS*/PS*, so there is no key to verify with. Also the reverse.
        let jwks = spawn_jwks_server("200 OK", jwks_body_all()).await;
        let v = validator(&jwks.url);
        let c = claims(serde_json::json!({"scope": "mcp:read"}));
        let es_labelled_rsa = mint_with(Algorithm::ES256, Some(KID_A), None, c.clone());
        assert!(is_invalid(&v.validate(&es_labelled_rsa).await));
        let rs_labelled_ec = mint_with(Algorithm::RS256, Some(KID_EC), None, c.clone());
        assert!(is_invalid(&v.validate(&rs_labelled_ec).await));
        // KID_A declares `alg: RS256`, so it must refuse PS256 even though an RSA
        // key could technically verify it.
        let ps_on_rs_only_key = mint_with(Algorithm::PS256, Some(KID_A), None, c);
        assert!(is_invalid(&v.validate(&ps_on_rs_only_key).await));
    }

    #[tokio::test]
    async fn es256_ps256_and_eddsa_tokens_are_accepted() {
        let jwks = spawn_jwks_server("200 OK", jwks_body_all()).await;
        let v = validator(&jwks.url);
        let c = claims(serde_json::json!({"scope": "mcp:read"}));
        for (alg, kid) in [
            (Algorithm::ES256, KID_EC),
            (Algorithm::PS256, "test-key-a-pss"),
            (Algorithm::RS384, "test-key-a-pss"),
            (Algorithm::EdDSA, KID_ED),
            (Algorithm::RS256, KID_A),
        ] {
            let token = mint_with(alg, Some(kid), Some("at+jwt"), c.clone());
            assert!(v.validate(&token).await.is_ok(), "{alg:?} must verify");
        }
    }

    #[tokio::test]
    async fn an_algorithm_outside_the_allowlist_is_rejected_before_any_jwks_fetch() {
        let jwks = spawn_jwks_server("200 OK", jwks_body_all()).await;
        let mut cfg = oauth_config(&jwks.url);
        cfg.algorithms = vec![Algorithm::RS256];
        let v = validator_with(cfg);
        let token = mint_with(
            Algorithm::ES256,
            Some(KID_EC),
            None,
            claims(serde_json::json!({"scope": "mcp:read"})),
        );
        assert!(is_invalid(&v.validate(&token).await));
        assert_eq!(jwks.hits.load(Ordering::SeqCst), 0);
    }

    // ── typ ──────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn typ_access_token_types_pass_and_other_jwt_types_fail() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        let c = claims(serde_json::json!({"scope": "mcp:read"}));
        for typ in [
            None,
            Some("JWT"),
            Some("jwt"),
            Some("at+jwt"),
            Some("AT+JWT"),
            Some("application/at+jwt"),
        ] {
            let token = mint_with(Algorithm::RS256, Some(KID_A), typ, c.clone());
            assert!(v.validate(&token).await.is_ok(), "typ {typ:?} must pass");
        }
        for typ in ["dpop+jwt", "logout+jwt", "secevent+jwt", "JOSE"] {
            let token = mint_with(Algorithm::RS256, Some(KID_A), Some(typ), c.clone());
            assert!(is_invalid(&v.validate(&token).await), "typ {typ} must fail");
        }
    }

    #[tokio::test]
    async fn require_at_jwt_refuses_plain_jwt_and_a_missing_typ() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let mut cfg = oauth_config(&jwks.url);
        cfg.require_at_jwt = true;
        let v = validator_with(cfg);
        let c = claims(serde_json::json!({"scope": "mcp:read"}));
        for typ in [None, Some("JWT")] {
            let token = mint_with(Algorithm::RS256, Some(KID_A), typ, c.clone());
            assert!(is_invalid(&v.validate(&token).await), "typ {typ:?}");
        }
        let token = mint_with(Algorithm::RS256, Some(KID_A), Some("at+jwt"), c);
        assert!(v.validate(&token).await.is_ok());
    }

    // ── credential shape ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn garbage_opaque_and_oversized_credentials_are_rejected_without_a_fetch() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        let oversized = format!("{}.{}.{}", "a".repeat(MAX_TOKEN_BYTES), "b", "c");
        for junk in [
            "not-a-jwt",
            "a.b.c",
            "a.b",
            // An Authelia-style opaque access token.
            "authelia_at_Xy9vQ3c2bG9uZ3JhbmRvbXN0cmluZw.abc",
            oversized.as_str(),
        ] {
            assert!(is_invalid(&v.validate(junk).await), "{junk:.40}");
        }
        assert_eq!(jwks.hits.load(Ordering::SeqCst), 0);
    }

    // ── JWKS fetching, rotation and rate limiting ────────────────────────────

    #[tokio::test]
    async fn the_jwks_is_fetched_once_and_cached() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        for _ in 0..3 {
            v.validate(&valid_token()).await.unwrap();
        }
        assert_eq!(
            jwks.hits.load(Ordering::SeqCst),
            1,
            "a cached key must not be re-fetched per request"
        );
    }

    #[tokio::test]
    async fn an_unknown_kid_does_not_refetch_during_the_cooldown() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url); // real 60s cooldown
        // First call populates the cache (one fetch); the unknown kid is then NOT
        // worth a second fetch, because we just fetched.
        let token = mint(
            KEY_A_PEM,
            "rotated-key",
            claims(serde_json::json!({"scope": "mcp:read"})),
        );
        for _ in 0..5 {
            assert!(is_invalid(&v.validate(&token).await));
        }
        assert_eq!(
            jwks.hits.load(Ordering::SeqCst),
            1,
            "kid is attacker-controlled — five junk tokens must not mean five IdP hits"
        );
    }

    #[tokio::test]
    async fn concurrent_unknown_kids_cost_one_fetch() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = Arc::new(validator(&jwks.url));
        let mut tasks = Vec::new();
        for i in 0..20 {
            let v = Arc::clone(&v);
            tasks.push(tokio::spawn(async move {
                let token = mint(
                    KEY_A_PEM,
                    &format!("junk-{i}"),
                    claims(serde_json::json!({"scope": "mcp:read"})),
                );
                v.validate(&token).await
            }));
        }
        for t in tasks {
            assert!(is_invalid(&t.await.unwrap()));
        }
        assert_eq!(jwks.hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn an_unknown_kid_refetches_once_the_cooldown_has_passed() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator_no_cooldown(&jwks.url);
        let token = mint(
            KEY_A_PEM,
            "rotated-key",
            claims(serde_json::json!({"scope": "mcp:read"})),
        );
        assert!(is_invalid(&v.validate(&token).await));
        assert!(is_invalid(&v.validate(&token).await));
        assert_eq!(
            jwks.hits.load(Ordering::SeqCst),
            2,
            "with the cooldown elapsed, an unknown kid must trigger a refresh — this \
             is how a rotated signing key is picked up without a restart"
        );
    }

    #[tokio::test]
    async fn a_rotated_key_is_picked_up_and_a_withdrawn_key_is_dropped() {
        let jwks = spawn_http_server(HashMap::new(), None).await;
        let set = |body: String| {
            jwks.routes
                .lock()
                .unwrap()
                .insert("/jwks".to_string(), ("200 OK", body));
        };
        set(jwks_body());
        let v = validator_no_cooldown(&jwks.url);
        let c = claims(serde_json::json!({"scope": "mcp:read"}));
        let old = mint(KEY_A_PEM, KID_A, c.clone());
        let new = mint_with(Algorithm::ES256, Some(KID_EC), None, c);

        assert!(v.validate(&old).await.is_ok());
        // The AS publishes the new key alongside the old one: the unknown kid
        // triggers a refetch and both verify.
        set(jwks_of(vec![jwk_rsa_a(), jwk_ec()]));
        assert!(v.validate(&new).await.is_ok());
        assert!(v.validate(&old).await.is_ok());
        // The AS withdraws the old key; the next refresh (the background task's
        // job) must stop trusting it.
        set(jwks_of(vec![jwk_ec()]));
        assert_eq!(v.refresh_now().await.unwrap(), 1);
        assert!(is_invalid(&v.validate(&old).await));
        assert!(v.validate(&new).await.is_ok());
    }

    #[tokio::test]
    async fn a_slow_refresh_does_not_stall_requests_whose_key_is_cached() {
        // The original code held the key lock across the fetch; with tokio's
        // writer-preferring RwLock that parked every request behind a slow IdP.
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = Arc::new(validator_no_cooldown(&jwks.url));
        v.validate(&valid_token()).await.unwrap();

        jwks.delay_ms.store(1500, Ordering::SeqCst);
        let background = Arc::clone(&v);
        let refresh = tokio::spawn(async move { background.refresh_now().await });
        // And an unknown-kid request that also wants a refresh, queued behind it.
        let junk = Arc::clone(&v);
        let queued = tokio::spawn(async move {
            junk.validate(&mint(
                KEY_A_PEM,
                "unknown",
                claims(serde_json::json!({"scope": "mcp:read"})),
            ))
            .await
        });
        tokio::time::sleep(Duration::from_millis(200)).await;

        let fast = tokio::time::timeout(Duration::from_millis(500), v.validate(&valid_token()))
            .await
            .expect("a cached-key validation must not wait for the in-flight refresh");
        assert!(fast.is_ok());
        assert!(refresh.await.unwrap().is_ok());
        assert!(is_invalid(&queued.await.unwrap()));
    }

    #[tokio::test]
    async fn a_failed_refresh_keeps_the_keys_already_held() {
        let jwks = spawn_http_server(HashMap::new(), None).await;
        jwks.routes
            .lock()
            .unwrap()
            .insert("/jwks".to_string(), ("200 OK", jwks_body()));
        let v = validator_no_cooldown(&jwks.url);
        assert!(v.validate(&valid_token()).await.is_ok());
        jwks.routes.lock().unwrap().insert(
            "/jwks".to_string(),
            ("503 Service Unavailable", "{}".into()),
        );
        assert!(v.refresh_now().await.is_err());
        assert!(
            v.validate(&valid_token()).await.is_ok(),
            "an IdP outage must not revoke keys that are still good"
        );
    }

    #[tokio::test]
    async fn an_unreachable_jwks_endpoint_fails_closed() {
        // Port 1 refuses instantly — the same unreachable-backend trick the status
        // and rerank tests use.
        let v = validator("http://127.0.0.1:1/jwks");
        assert!(
            is_invalid(&v.validate(&valid_token()).await),
            "an IdP we cannot reach must mean 'no', never 'sure'"
        );
    }

    #[tokio::test]
    async fn a_jwks_error_response_fails_closed() {
        let jwks = spawn_jwks_server("500 Internal Server Error", "{}".into()).await;
        let v = validator(&jwks.url);
        assert!(is_invalid(&v.validate(&valid_token()).await));
    }

    #[tokio::test]
    async fn an_oversized_jwks_response_fails_closed() {
        let padding = "x".repeat(MAX_FETCH_BYTES);
        let body = format!("{{\"keys\":[{}],\"padding\":\"{padding}\"}}", jwk_rsa_a());
        let jwks = spawn_jwks_server("200 OK", body).await;
        let v = validator(&jwks.url);
        assert!(is_invalid(&v.validate(&valid_token()).await));
    }

    #[tokio::test]
    async fn a_key_set_with_no_usable_keys_fails_closed() {
        let body = jwks_of(vec![
            // Encryption key, symmetric key, a P-521 key ring cannot verify, and
            // an RSA key whose declared alg contradicts its type: none may verify.
            serde_json::json!({"kty": "RSA", "use": "enc", "kid": KID_A, "n": N_A, "e": "AQAB"}),
            serde_json::json!({"kty": "oct", "kid": "hmac", "k": "c2VjcmV0"}),
            serde_json::json!({"kty": "EC", "crv": "P-521", "kid": "p521", "x": "AA", "y": "AA"}),
            serde_json::json!({"kty": "RSA", "alg": "ES256", "kid": KID_A, "n": N_A, "e": "AQAB"}),
        ]);
        let jwks = spawn_jwks_server("200 OK", body).await;
        let v = validator(&jwks.url);
        assert!(is_invalid(&v.validate(&valid_token()).await));
        assert!(
            v.refresh_now()
                .await
                .unwrap_err()
                .to_string()
                .contains("fetching the JWKS")
        );
    }

    #[tokio::test]
    async fn one_unparseable_key_does_not_take_the_usable_ones_down() {
        let body = jwks_of(vec![
            serde_json::json!({"kty": "OKP", "crv": "X25519", "kid": "x", "x": "AA"}),
            serde_json::json!({"kty": "weird", "kid": "w"}),
            jwk_rsa_a(),
        ]);
        let jwks = spawn_jwks_server("200 OK", body).await;
        let v = validator(&jwks.url);
        assert!(v.validate(&valid_token()).await.is_ok());
    }

    #[tokio::test]
    async fn a_kid_less_header_uses_the_single_compatible_key() {
        let jwks = spawn_jwks_server("200 OK", jwks_body()).await;
        let v = validator(&jwks.url);
        let c = claims(serde_json::json!({"scope": "mcp:read"}));
        let token = mint_with(Algorithm::RS256, None, None, c.clone());
        assert!(v.validate(&token).await.is_ok());

        // Two RSA keys that could both verify RS256: refuse rather than try each.
        let jwks = spawn_jwks_server(
            "200 OK",
            jwks_of(vec![jwk_rsa_a(), jwk_rsa_a_any_alg("second")]),
        )
        .await;
        let v = validator(&jwks.url);
        assert!(is_invalid(&v.validate(&token).await));
    }

    // ── discovery ────────────────────────────────────────────────────────────

    /// Serve OIDC discovery for `issuer_path` on a fake server whose document
    /// claims `doc_issuer`, plus the JWKS.
    async fn discovery_server(
        issuer_path: &str,
        doc_issuer: impl Fn(&str) -> String,
        via_rfc8414: bool,
    ) -> (FakeJwksServer, String) {
        let server = spawn_http_server(HashMap::new(), None).await;
        let issuer = format!("{}{issuer_path}", server.base);
        let doc = serde_json::json!({
            "issuer": doc_issuer(&issuer),
            "jwks_uri": format!("{}/keys", server.base),
        })
        .to_string();
        let well_known = if via_rfc8414 {
            format!(
                "/.well-known/oauth-authorization-server{}",
                issuer_path.trim_end_matches('/')
            )
        } else {
            format!(
                "{}/.well-known/openid-configuration",
                issuer_path.trim_end_matches('/')
            )
        };
        {
            let mut routes = server.routes.lock().unwrap();
            routes.insert(well_known, ("200 OK", doc));
            routes.insert("/keys".to_string(), ("200 OK", jwks_body()));
        }
        (server, issuer)
    }

    fn discovering_validator(issuer: &str) -> OAuthValidator {
        let mut cfg = oauth_config("");
        cfg.issuer = issuer.to_string();
        validator_with(cfg)
    }

    fn token_from(issuer: &str) -> String {
        mint(
            KEY_A_PEM,
            KID_A,
            claims(serde_json::json!({"iss": issuer, "scope": "mcp:read"})),
        )
    }

    #[tokio::test]
    async fn an_omitted_jwks_uri_is_discovered_once_from_oidc_metadata() {
        // Per-application issuer with a trailing slash — Authentik's shape.
        let (server, issuer) =
            discovery_server("/application/o/wiki/", |i| i.to_string(), false).await;
        let v = discovering_validator(&issuer);
        for _ in 0..3 {
            assert!(v.validate(&token_from(&issuer)).await.is_ok());
        }
        assert_eq!(
            server.hits.load(Ordering::SeqCst),
            2,
            "one discovery fetch and one JWKS fetch, then cached"
        );
    }

    #[tokio::test]
    async fn discovery_falls_back_to_rfc_8414_metadata() {
        let (_server, issuer) = discovery_server("/tenant", |i| i.to_string(), true).await;
        let v = discovering_validator(&issuer);
        assert!(v.validate(&token_from(&issuer)).await.is_ok());
    }

    #[tokio::test]
    async fn a_discovery_document_for_a_different_issuer_is_refused() {
        // The near miss again: the document drops the trailing slash.
        let (server, issuer) = discovery_server(
            "/application/o/wiki/",
            |i| i.trim_end_matches('/').to_string(),
            false,
        )
        .await;
        let v = discovering_validator(&issuer);
        assert!(is_invalid(&v.validate(&token_from(&issuer)).await));
        let err = format!("{:#}", v.refresh_now().await.unwrap_err());
        assert!(err.contains("does not match mcp.oauth.issuer"), "{err}");
        // Two candidate URLs per attempt, two attempts, and the mismatching
        // document's jwks_uri was never followed.
        assert_eq!(server.hits.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn discovery_urls_follow_oidc_then_rfc_8414() {
        assert_eq!(
            discovery_urls("https://auth.example.com/application/o/wiki/"),
            [
                "https://auth.example.com/application/o/wiki/.well-known/openid-configuration",
                "https://auth.example.com/.well-known/oauth-authorization-server/application/o/wiki",
            ]
        );
        assert_eq!(
            discovery_urls("https://auth.example.com"),
            [
                "https://auth.example.com/.well-known/openid-configuration",
                "https://auth.example.com/.well-known/oauth-authorization-server",
            ]
        );
    }

    #[test]
    fn discovered_jwks_uri_must_not_downgrade_transport() {
        let doc = serde_json::json!({
            "issuer": "https://auth.example.com",
            "jwks_uri": "http://auth.example.com/jwks",
        });
        assert!(jwks_uri_from_metadata(&doc, "https://auth.example.com").is_err());
        let doc = serde_json::json!({
            "issuer": "https://auth.example.com",
            "jwks_uri": "file:///etc/passwd",
        });
        assert!(jwks_uri_from_metadata(&doc, "https://auth.example.com").is_err());
        let doc = serde_json::json!({"issuer": "https://auth.example.com"});
        assert!(jwks_uri_from_metadata(&doc, "https://auth.example.com").is_err());
    }

    #[test]
    fn loopback_detection() {
        assert!(is_loopback_url("http://127.0.0.1:8080/x"));
        assert!(is_loopback_url("http://[::1]:8080/x"));
        assert!(is_loopback_url("http://localhost/x"));
        assert!(!is_loopback_url("http://auth.example.com/x"));
        assert!(!is_loopback_url("not a url"));
    }

    // ── regression: the production Authentik shape, unchanged ────────────────

    /// The exact config and token shape of the original production deployment
    /// (Authentik, per-application issuer with a trailing slash, JWKS at
    /// `<issuer>jwks/`, `aud` = the OAuth client_id as a string, `scope` a
    /// space-delimited string, RS256, header `typ: JWT`), parsed from YAML that
    /// uses ONLY the original keys. It must validate with every new key at its
    /// default. Hostnames are placeholders; the fake server stands in for the AS.
    #[tokio::test]
    async fn production_authentik_config_and_token_still_pass_unchanged() {
        let server = spawn_http_server(HashMap::new(), None).await;
        let issuer = format!("{}/application/o/mcp-kb-rag/", server.base);
        server.routes.lock().unwrap().insert(
            "/application/o/mcp-kb-rag/jwks/".to_string(),
            ("200 OK", jwks_body()),
        );
        let yaml = format!(
            "enabled: true\n\
             issuer: \"{issuer}\"\n\
             jwks_uri: \"{issuer}jwks/\"\n\
             audience: \"example-client-id\"\n\
             resource: \"https://kb.example.com/mcp\"\n\
             required_scope: \"mcp:read\"\n\
             scopes_supported: [\"mcp:read\", \"mcp:write\"]\n"
        );
        let parsed: crate::config::OAuthConfig = serde_yaml_ng::from_str(&yaml).unwrap();
        let cfg = parsed.resolve().unwrap().expect("enabled");
        assert!(
            cfg.accept_static_bearer,
            "dual mode must stay on by default"
        );
        let v = validator_with(cfg);

        let token = mint_with(
            Algorithm::RS256,
            Some(KID_A),
            Some("JWT"),
            serde_json::json!({
                "iss": issuer,
                "sub": "0000000000000000example",
                "aud": "example-client-id",
                "azp": "example-client-id",
                "exp": now() + 300,
                "iat": now(),
                "auth_time": now(),
                "acr": "goauthentik.io/providers/oauth2/default",
                "email": "user@example.com",
                "email_verified": true,
                "name": "Example User",
                "given_name": "Example User",
                "preferred_username": "example",
                "nickname": "example",
                "groups": ["wiki-users"],
                "scope": "openid email profile mcp:read mcp:write",
            }),
        );
        let t = v.validate(&token).await.unwrap();
        assert_eq!(t.principal.as_deref(), Some("example"));
        assert_eq!(
            t.scopes,
            ["openid", "email", "profile", "mcp:read", "mcp:write"]
        );
        // And the metadata and challenges are what they were before provider-agnostic validation.
        assert_eq!(v.metadata()["authorization_servers"][0], issuer.as_str());
        assert!(
            v.invalid_token_challenge()
                .starts_with("Bearer error=\"invalid_token\", resource_metadata=")
        );
    }

    // ── observed shapes: sandbox-tested authorization servers ────────────────
    //
    // These mirror token shapes captured from real Authelia 4.39.4 and Kanidm
    // sandboxes. Hostnames and ids are placeholders.

    async fn accepts(
        cfg_edit: impl FnOnce(&mut ResolvedOAuthConfig),
        alg: Algorithm,
        kid: &str,
        typ: Option<&str>,
        token_claims: serde_json::Value,
    ) -> AuthorizedToken {
        let jwks = spawn_jwks_server("200 OK", jwks_body_all()).await;
        let mut cfg = oauth_config(&jwks.url);
        cfg_edit(&mut cfg);
        let v = validator_with(cfg);
        v.validate(&mint_with(alg, Some(kid), typ, token_claims))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn observed_shape_authelia_4_39_scp_array_and_resource_url_audience() {
        let issuer = "https://auth.example.com";
        let resource = "https://kb.example.com/mcp";
        let t = accepts(
            |c| {
                c.issuer = issuer.into();
                c.audience = resource.into();
                c.require_at_jwt = true;
            },
            Algorithm::RS256,
            "test-key-a-pss",
            Some("at+jwt"),
            serde_json::json!({
                "iss": issuer, "aud": [resource], "client_id": "example-client",
                "sub": "44726d41-0000-4000-8000-000000000000",
                "exp": now() + 3600, "iat": now(), "nbf": now(),
                "jti": "x", "scp": ["mcp:read", "mcp:write"],
            }),
        )
        .await;
        assert_eq!(t.scopes, ["mcp:read", "mcp:write"]);
        // No username claim in Authelia access tokens: the chain lands on `sub`.
        assert_eq!(
            t.principal.as_deref(),
            Some("44726d41-0000-4000-8000-000000000000")
        );
    }

    #[tokio::test]
    async fn observed_shape_kanidm_es256_per_client_issuer_and_client_audience() {
        let issuer = "https://idm.example.com/oauth2/openid/example-client";
        let t = accepts(
            |c| {
                c.issuer = issuer.into();
                c.audience = "example-client".into();
                c.require_at_jwt = true;
            },
            Algorithm::ES256,
            KID_EC,
            Some("at+jwt"),
            serde_json::json!({
                "iss": issuer, "aud": "example-client", "client_id": "example-client",
                "sub": "00000000-0000-4000-8000-000000000001",
                "exp": now() + 900, "iat": now(), "nbf": now(), "jti": "x",
                "scope": "mcp:read openid profile",
            }),
        )
        .await;
        assert!(t.has_scope("mcp:read"));
    }

    // ── documented-shape fixtures, NOT live-tested ───────────────────────────
    //
    // Each models the access-token shape the named authorization server
    // documents (or, where noted, its source code shows), to prove the generic
    // validator covers it with config alone. None of these has been run against
    // the real product; they are "documented-shape fixture, not live-tested" and
    // must not be cited as compatibility claims.

    #[tokio::test]
    async fn documented_shape_fixture_not_live_tested_keycloak() {
        // Realm issuer, `typ` JWT (at+jwt is an opt-in client switch since 26.2),
        // `scope` string, `azp` = client, `preferred_username` present.
        let issuer = "https://sso.example.com/realms/home";
        let t = accepts(
            |c| {
                c.issuer = issuer.into();
                c.audience = "wiki".into();
            },
            Algorithm::RS256,
            KID_A,
            Some("JWT"),
            serde_json::json!({
                "iss": issuer, "aud": ["wiki", "account"], "azp": "wiki",
                "sub": "u", "exp": now() + 300, "typ": "Bearer",
                "preferred_username": "alice", "scope": "openid profile mcp:read",
            }),
        )
        .await;
        assert_eq!(t.principal.as_deref(), Some("alice"));
    }

    #[tokio::test]
    async fn documented_shape_fixture_not_live_tested_okta_custom_as() {
        // Custom authorization server: no `typ` header at all, `scp` array,
        // `aud` = the configured API audience, `cid` = client.
        let issuer = "https://example.okta.com/oauth2/default";
        let t = accepts(
            |c| {
                c.issuer = issuer.into();
                c.audience = "api://default".into();
            },
            Algorithm::RS256,
            KID_A,
            None,
            serde_json::json!({
                "iss": issuer, "aud": "api://default", "cid": "client", "sub": "a@example.com",
                "exp": now() + 3600, "scp": ["openid", "mcp:read"],
            }),
        )
        .await;
        assert!(t.has_scope("mcp:read"));
    }

    #[tokio::test]
    async fn documented_shape_fixture_not_live_tested_entra_id_v2() {
        // v2.0 tenant issuer, `typ` JWT, `scp` space-delimited string, `aud` = the
        // API's client id.
        let issuer = "https://login.microsoftonline.com/00000000-0000-0000-0000-000000000000/v2.0";
        let t = accepts(
            |c| {
                c.issuer = issuer.into();
                c.audience = "11111111-1111-1111-1111-111111111111".into();
            },
            Algorithm::RS256,
            KID_A,
            Some("JWT"),
            serde_json::json!({
                "iss": issuer, "aud": "11111111-1111-1111-1111-111111111111",
                "sub": "pairwise", "oid": "o", "exp": now() + 3600,
                "preferred_username": "alice@example.com", "scp": "mcp.read mcp:read",
            }),
        )
        .await;
        assert!(t.has_scope("mcp:read"));
    }

    #[tokio::test]
    async fn documented_shape_fixture_not_live_tested_auth0() {
        // Issuer with a trailing slash, `aud` array (API identifier + userinfo),
        // `scope` string, both the Auth0 (`typ` JWT) and RFC 9068 (`at+jwt`)
        // profiles.
        let issuer = "https://tenant.example.auth0.com/";
        for typ in ["JWT", "at+jwt"] {
            let t = accepts(
                |c| {
                    c.issuer = issuer.into();
                    c.audience = "https://kb.example.com/mcp".into();
                },
                Algorithm::RS256,
                KID_A,
                Some(typ),
                serde_json::json!({
                    "iss": issuer,
                    "aud": ["https://kb.example.com/mcp", "https://tenant.example.auth0.com/userinfo"],
                    "azp": "client", "sub": "auth0|1", "exp": now() + 3600,
                    "scope": "openid mcp:read",
                }),
            )
            .await;
            assert!(t.has_scope("mcp:read"));
        }
    }

    #[tokio::test]
    async fn documented_shape_fixture_not_live_tested_ory_hydra_jwt_strategy() {
        // Only with `strategies.access_token: jwt` (the default is opaque);
        // `scp` is a list by default, a string with `oauth2.jwt.scope_claim: string`.
        let issuer = "https://hydra.example.com/";
        for scp in [
            serde_json::json!(["mcp:read"]),
            serde_json::json!("offline mcp:read"),
        ] {
            let t = accepts(
                |c| {
                    c.issuer = issuer.into();
                    c.audience = "https://kb.example.com/mcp".into();
                },
                Algorithm::RS256,
                KID_A,
                Some("JWT"),
                serde_json::json!({
                    "iss": issuer, "aud": ["https://kb.example.com/mcp"], "sub": "u",
                    "client_id": "c", "exp": now() + 3600, "scp": scp, "ext": {},
                }),
            )
            .await;
            assert!(t.has_scope("mcp:read"));
        }
    }

    #[tokio::test]
    async fn documented_shape_fixture_not_live_tested_logto_resource_indicator() {
        // `aud` = the registered API resource indicator (RFC 8707), `scope`
        // string, ES256 among its allowed signing algorithms.
        let issuer = "https://logto.example.com/oidc";
        let t = accepts(
            |c| {
                c.issuer = issuer.into();
                c.audience = "https://kb.example.com/mcp".into();
            },
            Algorithm::ES256,
            KID_EC,
            None,
            serde_json::json!({
                "iss": issuer, "aud": "https://kb.example.com/mcp", "sub": "u",
                "client_id": "c", "exp": now() + 3600, "scope": "mcp:read",
            }),
        )
        .await;
        assert!(t.has_scope("mcp:read"));
    }

    #[tokio::test]
    async fn documented_shape_fixture_not_live_tested_casdoor_jwt_standard() {
        // Source-derived: no `typ` beyond jsonwebtoken's default, `aud` =
        // [client_id] (or [resource] when RFC 8707 is used), `scope` string,
        // `preferred_username` with the JWT-Standard token format.
        let issuer = "https://casdoor.example.com";
        let t = accepts(
            |c| {
                c.issuer = issuer.into();
                c.audience = "example-client-id".into();
            },
            Algorithm::RS256,
            KID_A,
            Some("JWT"),
            serde_json::json!({
                "iss": issuer, "aud": ["example-client-id"], "sub": "u",
                "exp": now() + 3600, "preferred_username": "alice",
                "scope": "openid mcp:read",
            }),
        )
        .await;
        assert_eq!(t.principal.as_deref(), Some("alice"));
    }

    #[tokio::test]
    async fn documented_shape_fixture_not_live_tested_rauthy_eddsa_at_jwt() {
        // Source-derived: `typ` at+jwt, `scope` string, EdDSA available per
        // client, no `preferred_username` (the principal chain falls to `sub`).
        let issuer = "https://rauthy.example.com/auth/v1";
        let t = accepts(
            |c| {
                c.issuer = issuer.into();
                c.audience = "example-client".into();
                c.require_at_jwt = true;
            },
            Algorithm::EdDSA,
            KID_ED,
            Some("at+jwt"),
            serde_json::json!({
                "iss": issuer, "aud": "example-client", "azp": "example-client",
                "sub": "user-id", "exp": now() + 1800, "scope": "openid mcp:read",
            }),
        )
        .await;
        assert_eq!(t.principal.as_deref(), Some("user-id"));
    }

    #[tokio::test]
    async fn documented_shape_fixture_not_live_tested_dex_needs_a_group_claim_as_scope() {
        // Source-derived: Dex's access token is an ID token (`aud` = client_id, no
        // `scope`/`scp` claim at all). The only generic way to gate it is to read
        // a group claim as the scope source — a compromise, see the design notes.
        let issuer = "https://dex.example.com";
        let t = accepts(
            |c| {
                c.issuer = issuer.into();
                c.audience = "example-client".into();
                c.scope_claims = vec!["groups".into()];
                c.required_scope = "wiki-users".into();
            },
            Algorithm::RS256,
            KID_A,
            None,
            serde_json::json!({
                "iss": issuer, "aud": "example-client", "sub": "u",
                "exp": now() + 3600, "email": "a@example.com",
                "groups": ["wiki-users", "admins"],
            }),
        )
        .await;
        assert!(t.has_scope("wiki-users"));
    }

    #[tokio::test]
    async fn documented_shape_fixture_not_live_tested_zitadel_jwt_mode() {
        // Only with the application's token type switched to JWT (opaque is the
        // alternative). `aud` holds the client ids and the project id.
        // Zitadel's scope claim shape is not documented where we looked; this
        // fixture exercises the aud-array/project-id part only.
        let issuer = "https://zitadel.example.com";
        let t = accepts(
            |c| {
                c.issuer = issuer.into();
                c.audience = "123456789012345678".into();
            },
            Algorithm::RS256,
            KID_A,
            None,
            serde_json::json!({
                "iss": issuer,
                "aud": ["234567890123456789@wiki", "123456789012345678"],
                "client_id": "234567890123456789@wiki", "sub": "u",
                "exp": now() + 3600, "scope": "openid mcp:read",
            }),
        )
        .await;
        assert!(t.has_scope("mcp:read"));
    }
}
