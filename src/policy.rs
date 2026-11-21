//! Per-principal access control: who a request is, and what that principal may do.
//!
//! The model is deliberately small. A **principal** is the party behind a request:
//! a named static token, an OAuth subject, or the legacy single `bearer_token_env`
//! token. A **policy** is a named bundle of grants: the tools it may call, the
//! directories it may read and write, and whether it may reach the operator routes
//! (`/status*`, `/metrics`, `/admin/*`). `mcp.tokens` maps a token name to a policy,
//! `mcp.principals` maps an OAuth scope, client id or claim to one, and
//! [`AccessControl::principal`] turns an authenticated credential into a
//! [`Principal`] once per request.
//!
//! Three properties everything else relies on:
//!
//! - **Off means unchanged.** With no `mcp.policy` configured ([`AccessControl::enforcing`]
//!   is false) every principal is [`Grant::Unrestricted`] and nothing here refuses
//!   anything, so an upgrade changes no behaviour.
//! - **Default deny once on.** With a policy configured, a principal that no token
//!   name or principal rule maps to gets [`Grant::Nothing`], and a request that
//!   reaches a handler with no principal at all is treated the same way
//!   ([`Principal::from_extension`]). The legacy single static token is the one
//!   unrestricted principal, because it has always meant full access.
//! - **One choke point.** Every check is [`Principal::authorize`] (or its tool-only
//!   and admin forms), so a new surface adds one call, not a new rule.
//!
//! Paths are directory prefixes relative to the knowledge-base root, not globs. A
//! prefix is exactly what the search backends can enforce — a `path_ancestors`
//! keyword match in Qdrant and a prefix clause over the SQLite metadata index — so a
//! grant cannot be wider or narrower in one backend than the other. `""` (written
//! `/` in YAML) is the whole knowledge base.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

/// The tool name that grants every tool, including ones added later.
pub const ALL_TOOLS: &str = "*";

/// The id of the principal the legacy `mcp.bearer_token_env` token maps to.
pub const LEGACY_PRINCIPAL_ID: &str = "default";

/// What an operation does to a document, for [`Principal::authorize`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Read,
    Write,
}

/// A named bundle of grants (`mcp.policy.<name>`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyConfig {
    /// Tools the principal may call: names from `descriptions::TOOL_NAMES` (plus the
    /// memory tools when configured), or `"*"` for all of them. Empty grants none.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Directories (relative to the KB root, `/` for all of it) whose documents the
    /// principal may read: search, get, list, history, schema and link traversal.
    #[serde(default)]
    pub read: Vec<String>,
    /// Directories whose documents the principal may create, edit, move and delete.
    /// Writing does not imply reading: list a directory under both to get both.
    #[serde(default)]
    pub write: Vec<String>,
    /// Whether the principal may reach `/status*`, `/metrics` and `/admin/*`.
    #[serde(default)]
    pub admin: bool,
}

/// A named static token (`mcp.tokens[]`). The secret is read from the environment
/// variable `secret_env` names (or `<secret_env>_FILE`), never from the YAML.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TokenConfig {
    /// A label for logs and attribution: 1 to 64 visible ASCII characters, unique.
    pub name: String,
    /// The environment variable holding the secret (or, with a `_FILE` suffix, a
    /// file holding it).
    pub secret_env: String,
    /// The `mcp.policy` entry this token acts under.
    pub policy: String,
}

/// Which OAuth principals a [`PrincipalRule`] applies to. Exactly one key is set.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrincipalMatch {
    /// The token carries this scope.
    #[serde(default)]
    pub scope: Option<String>,
    /// The token was issued to this OAuth client (`client_id`, else `azp`).
    #[serde(default)]
    pub client_id: Option<String>,
    /// The token carries a claim with this name holding this value (a list claim
    /// matches when any entry equals it).
    #[serde(default)]
    pub claim: Option<ClaimMatch>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimMatch {
    pub name: String,
    pub value: String,
}

/// An OAuth principal rule (`mcp.principals[]`); the first match wins.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrincipalRule {
    #[serde(rename = "match")]
    pub matcher: PrincipalMatch,
    pub policy: String,
}

/// A policy with its paths normalized, ready to answer questions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Policy {
    name: String,
    tools: BTreeSet<String>,
    all_tools: bool,
    read: Vec<String>,
    write: Vec<String>,
    admin: bool,
}

/// Normalize a configured directory to the form every comparison uses: no leading
/// or trailing `/`, `""` for the KB root. Refuses `..`, `.git` and empty
/// components, so a grant can never name something the path-safety rules forbid.
pub fn normalize_prefix(raw: &str) -> Result<String> {
    let trimmed = raw.trim().trim_matches('/');
    if trimmed.is_empty() {
        return Ok(String::new());
    }
    for part in trimmed.split('/') {
        if part.is_empty() || part == "." || part == ".." || part.eq_ignore_ascii_case(".git") {
            bail!("'{raw}' is not a plain directory path (no '..', '.', '.git' or empty parts)");
        }
        if part.contains('\\') || part.contains('*') || part.contains('?') {
            bail!("'{raw}' must be a directory path, not a glob");
        }
    }
    Ok(trimmed.to_string())
}

/// Whether `prefix` (normalized) covers `path`: the root covers everything, any
/// other prefix covers itself and what is under it, never a sibling that merely
/// shares leading characters (`notes` does not cover `notes-old/a.md`).
///
/// `path` is read lexically: a leading `/` and `.` or empty parts are ignored, and
/// a `..` part makes it uncovered (never resolved), so no spelling of a path reaches
/// a directory the grant does not name.
pub fn prefix_covers(prefix: &str, path: &str) -> bool {
    let mut parts = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => return false,
            p => parts.push(p),
        }
    }
    if prefix.is_empty() {
        return true;
    }
    let want: Vec<&str> = prefix.split('/').collect();
    parts.len() >= want.len() && parts[..want.len()] == want[..]
}

impl Policy {
    fn compile(name: &str, cfg: &PolicyConfig, known_tools: &[&str]) -> Result<Self> {
        let mut tools = BTreeSet::new();
        let mut all_tools = false;
        for tool in &cfg.tools {
            if tool == ALL_TOOLS {
                all_tools = true;
            } else if known_tools.contains(&tool.as_str()) {
                tools.insert(tool.clone());
            } else {
                bail!(
                    "mcp.policy.{name}.tools names unknown tool '{tool}' (known: {}, or \"*\")",
                    known_tools.join(", ")
                );
            }
        }
        let paths = |key: &str, list: &[String]| -> Result<Vec<String>> {
            let mut out = BTreeSet::new();
            for raw in list {
                match normalize_prefix(raw) {
                    Ok(p) => {
                        out.insert(p);
                    }
                    Err(e) => bail!("mcp.policy.{name}.{key}: {e}"),
                }
            }
            Ok(out.into_iter().collect())
        };
        let read = paths("read", &cfg.read)?;
        let write = paths("write", &cfg.write)?;
        // A write answers with the diff it made, a refused edit says whether its
        // anchor matched, and a move or schema change reports what it touched: a
        // write grant over a directory the principal cannot read would let it read
        // that directory through the write tools. So every write prefix must sit
        // inside a read prefix.
        for w in &write {
            if !read.iter().any(|r| prefix_covers(r, w)) {
                let shown = if w.is_empty() { "/" } else { w.as_str() };
                bail!(
                    "mcp.policy.{name}.write: '{shown}' is not inside any `read` prefix; \
                     a principal that may write a directory must be able to read it, since \
                     a write returns what it changed"
                );
            }
        }
        Ok(Self {
            name: name.to_string(),
            tools,
            all_tools,
            read,
            write,
            admin: cfg.admin,
        })
    }

    fn allows_tool(&self, tool: &str) -> bool {
        self.all_tools || self.tools.contains(tool)
    }

    fn prefixes(&self, op: Op) -> &[String] {
        match op {
            Op::Read => &self.read,
            Op::Write => &self.write,
        }
    }
}

/// What a principal is allowed, as resolved for one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Grant {
    /// Policies are not enforced, or this is the legacy full-access token.
    Unrestricted,
    /// Constrained by one policy.
    Policy(Arc<Policy>),
    /// Authenticated but mapped to no policy: nothing is allowed.
    Nothing,
}

/// How a credential was presented, for attribution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrincipalSource {
    /// A static token: its `mcp.tokens` name, or `None` for the legacy token.
    Static(Option<String>),
    Oauth,
    /// No credential was presented or recognised.
    Anonymous,
}

/// The party behind a request, resolved once per request and carried in the
/// request extensions (`/mcp` and `/api/*`) next to the auth layer's own types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Principal {
    id: String,
    source: PrincipalSource,
    grant: Grant,
}

/// Why [`Principal::authorize`] refused. Carries nothing about the target, so a
/// caller learns no more from a refusal than "not permitted".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Denied;

impl std::fmt::Display for Denied {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("not permitted")
    }
}

impl std::error::Error for Denied {}

/// The read or write reach of a principal, for building a search or listing
/// restriction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reach {
    /// Everything.
    All,
    /// Only documents under these normalized directories (possibly none).
    Prefixes(Vec<String>),
}

impl Reach {
    /// This reach confined to the directory `dir` (normalized, `""` for the root):
    /// what a caller may reach that is also inside `dir`.
    pub fn within(&self, dir: &str) -> Reach {
        match self {
            Reach::All => Reach::Prefixes(vec![dir.to_string()]),
            Reach::Prefixes(prefixes) => {
                let mut out: Vec<String> = Vec::new();
                for p in prefixes {
                    // Whichever is deeper is inside both.
                    let inside = if prefix_covers(p, dir) {
                        dir
                    } else if prefix_covers(dir, p) {
                        p
                    } else {
                        continue;
                    };
                    if !out.iter().any(|o| o == inside) {
                        out.push(inside.to_string());
                    }
                }
                Reach::Prefixes(out)
            }
        }
    }

    /// The directories of a restricted reach, `None` when it is everything.
    pub fn dirs(&self) -> Option<Vec<String>> {
        match self {
            Reach::All => None,
            Reach::Prefixes(dirs) => Some(dirs.clone()),
        }
    }

    /// Whether a document at `path` is within reach.
    pub fn covers(&self, path: &str) -> bool {
        match self {
            Reach::All => true,
            Reach::Prefixes(prefixes) => prefixes.iter().any(|p| prefix_covers(p, path)),
        }
    }
}

impl Principal {
    /// A principal no policy constrains: what every request is while no policy is
    /// configured, and what unit tests that predate policies use.
    pub fn unrestricted(id: impl Into<String>, source: PrincipalSource) -> Self {
        Self {
            id: id.into(),
            source,
            grant: Grant::Unrestricted,
        }
    }

    fn nothing(id: impl Into<String>, source: PrincipalSource) -> Self {
        Self {
            id: id.into(),
            source,
            grant: Grant::Nothing,
        }
    }

    /// The principal for a request that reached a handler: the one the middleware
    /// stored, or — when there is none — unrestricted while no policy is enforced
    /// and [`Grant::Nothing`] once one is, so a route that was left out of the
    /// middleware fails closed.
    pub fn from_extension(found: Option<&Principal>, enforcing: bool) -> Principal {
        match found {
            Some(p) => p.clone(),
            None if enforcing => Principal::nothing("anonymous", PrincipalSource::Anonymous),
            None => Principal::unrestricted("anonymous", PrincipalSource::Anonymous),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    #[cfg(test)]
    pub fn grant(&self) -> &Grant {
        &self.grant
    }

    /// Whether the principal may call `tool` at all.
    pub fn authorize_tool(&self, tool: &str) -> Result<(), Denied> {
        match &self.grant {
            Grant::Unrestricted => Ok(()),
            Grant::Policy(p) if p.allows_tool(tool) => Ok(()),
            _ => Err(Denied),
        }
    }

    /// Tool and path together, as a test convenience: may this principal call
    /// `tool` and, when the call touches a document, `op` it at `path`? Production
    /// code asks [`Self::authorize_tool`] in the dispatcher and
    /// [`Self::authorize_path`] at each path it resolves.
    #[cfg(test)]
    pub fn authorize(&self, tool: &str, path: Option<&str>, op: Op) -> Result<(), Denied> {
        self.authorize_tool(tool)?;
        match path {
            Some(path) => self.authorize_path(path, op),
            None => Ok(()),
        }
    }

    /// Whether the principal may `op` the document at `path`.
    pub fn authorize_path(&self, path: &str, op: Op) -> Result<(), Denied> {
        if self.reach(op).covers(path) {
            Ok(())
        } else {
            Err(Denied)
        }
    }

    /// The directories the principal may `op`, for restricting a query rather than
    /// checking one path.
    pub fn reach(&self, op: Op) -> Reach {
        match &self.grant {
            Grant::Unrestricted => Reach::All,
            Grant::Policy(p) => {
                let prefixes = p.prefixes(op);
                if prefixes.iter().any(String::is_empty) {
                    Reach::All
                } else {
                    Reach::Prefixes(prefixes.to_vec())
                }
            }
            Grant::Nothing => Reach::Prefixes(Vec::new()),
        }
    }

    /// Whether the principal may reach the operator routes.
    pub fn can_admin(&self) -> bool {
        match &self.grant {
            Grant::Unrestricted => true,
            Grant::Policy(p) => p.admin,
            Grant::Nothing => false,
        }
    }

    /// Whether the principal has any access at all; false only for [`Grant::Nothing`].
    pub fn has_access(&self) -> bool {
        !matches!(self.grant, Grant::Nothing)
    }

    /// Whether this principal is constrained (as opposed to unrestricted).
    pub fn is_restricted(&self) -> bool {
        !matches!(self.grant, Grant::Unrestricted)
    }
}

/// The facts about a validated OAuth token that principal rules match on.
pub struct OauthFacts<'a> {
    pub subject: Option<&'a str>,
    pub scopes: &'a [String],
    pub client_id: Option<&'a str>,
    /// Whether the token carries a claim `name` holding `value`.
    pub has_claim_value: &'a dyn Fn(&str, &str) -> bool,
}

/// How a request authenticated, as far as [`AccessControl::principal`] cares.
pub enum Credential<'a> {
    /// A static token: its `mcp.tokens` name, or `None` for the legacy token.
    Static(Option<&'a str>),
    Oauth(OauthFacts<'a>),
    /// Nothing authenticated the request (the explicit unauthenticated pass-through).
    None,
}

/// The compiled access-control configuration: every policy, the token-to-policy
/// map and the OAuth rules. Built by [`AccessControl::compile`] when the config
/// resolves; the default is "not enforcing".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AccessControl {
    policies: BTreeMap<String, Arc<Policy>>,
    tokens: BTreeMap<String, String>,
    rules: Vec<(PrincipalMatch, String)>,
}

fn valid_label(label: &str) -> bool {
    (1..=64).contains(&label.len()) && label.bytes().all(|b| (b'!'..=b'~').contains(&b))
}

impl AccessControl {
    /// Validate and compile the three config sections. `known_tools` is the tool
    /// name list a policy may name; `oauth_enabled` says whether `mcp.principals`
    /// can ever match.
    pub fn compile(
        policies: &BTreeMap<String, PolicyConfig>,
        tokens: &[TokenConfig],
        rules: &[PrincipalRule],
        known_tools: &[&str],
        oauth_enabled: bool,
    ) -> Result<Self> {
        let mut compiled = BTreeMap::new();
        for (name, cfg) in policies {
            if !valid_label(name) {
                bail!(
                    "mcp.policy: the name {name:?} must be 1 to 64 visible ASCII characters (no spaces)"
                );
            }
            compiled.insert(
                name.clone(),
                Arc::new(Policy::compile(name, cfg, known_tools)?),
            );
        }

        let mut token_policy = BTreeMap::new();
        for (i, t) in tokens.iter().enumerate() {
            if !valid_label(&t.name) {
                bail!(
                    "mcp.tokens[{i}].name {:?} must be 1 to 64 visible ASCII characters (no spaces)",
                    t.name
                );
            }
            if t.name == LEGACY_PRINCIPAL_ID {
                bail!(
                    "mcp.tokens[{i}].name '{LEGACY_PRINCIPAL_ID}' is reserved for the \
                     mcp.bearer_token_env token"
                );
            }
            if t.secret_env.trim().is_empty() {
                bail!(
                    "mcp.tokens[{i}] ('{}'): secret_env must name an environment variable",
                    t.name
                );
            }
            if !compiled.contains_key(&t.policy) {
                bail!(
                    "mcp.tokens[{i}] ('{}') uses policy '{}', which is not defined under mcp.policy",
                    t.name,
                    t.policy
                );
            }
            if token_policy
                .insert(t.name.clone(), t.policy.clone())
                .is_some()
            {
                bail!("mcp.tokens: the name '{}' is used twice", t.name);
            }
        }
        let mut seen_env = BTreeSet::new();
        for t in tokens {
            if !seen_env.insert(t.secret_env.as_str()) {
                bail!(
                    "mcp.tokens: the secret_env '{}' is used twice; every token needs its own secret",
                    t.secret_env
                );
            }
        }

        let mut compiled_rules = Vec::new();
        for (i, r) in rules.iter().enumerate() {
            let m = &r.matcher;
            let set = usize::from(m.scope.is_some())
                + usize::from(m.client_id.is_some())
                + usize::from(m.claim.is_some());
            if set != 1 {
                bail!("mcp.principals[{i}].match must set exactly one of scope, client_id, claim");
            }
            for (key, v) in [("scope", &m.scope), ("client_id", &m.client_id)] {
                if v.as_deref().is_some_and(|s| s.trim().is_empty()) {
                    bail!("mcp.principals[{i}].match.{key} is empty");
                }
            }
            if let Some(c) = &m.claim
                && (c.name.trim().is_empty() || c.value.is_empty())
            {
                bail!("mcp.principals[{i}].match.claim needs a non-empty name and value");
            }
            if !compiled.contains_key(&r.policy) {
                bail!(
                    "mcp.principals[{i}] uses policy '{}', which is not defined under mcp.policy",
                    r.policy
                );
            }
            compiled_rules.push((m.clone(), r.policy.clone()));
        }
        if !compiled_rules.is_empty() && !oauth_enabled {
            bail!("mcp.principals match OAuth tokens, but mcp.oauth is not enabled");
        }
        if compiled.is_empty() && (!tokens.is_empty() || !rules.is_empty()) {
            bail!("mcp.tokens / mcp.principals refer to policies, but mcp.policy is empty");
        }

        Ok(Self {
            policies: compiled,
            tokens: token_policy,
            rules: compiled_rules,
        })
    }

    /// Whether policies are enforced: true exactly when `mcp.policy` is non-empty.
    pub fn enforcing(&self) -> bool {
        !self.policies.is_empty()
    }

    /// Resolve an authenticated credential to its principal.
    pub fn principal(&self, credential: Credential<'_>) -> Principal {
        if !self.enforcing() {
            return match credential {
                Credential::Static(label) => Principal::unrestricted(
                    label.unwrap_or(LEGACY_PRINCIPAL_ID),
                    PrincipalSource::Static(label.map(str::to_string)),
                ),
                Credential::Oauth(f) => {
                    Principal::unrestricted(f.subject.unwrap_or("oauth"), PrincipalSource::Oauth)
                }
                Credential::None => {
                    Principal::unrestricted("anonymous", PrincipalSource::Anonymous)
                }
            };
        }
        match credential {
            // The legacy token has always been full access; it stays so.
            Credential::Static(None) => {
                Principal::unrestricted(LEGACY_PRINCIPAL_ID, PrincipalSource::Static(None))
            }
            Credential::Static(Some(label)) => {
                let source = PrincipalSource::Static(Some(label.to_string()));
                match self.tokens.get(label).and_then(|p| self.policies.get(p)) {
                    Some(policy) => Principal {
                        id: label.to_string(),
                        source,
                        grant: Grant::Policy(Arc::clone(policy)),
                    },
                    None => Principal::nothing(label, source),
                }
            }
            Credential::Oauth(facts) => {
                let id = facts.subject.unwrap_or("oauth");
                for (m, policy) in &self.rules {
                    if Self::rule_matches(m, &facts)
                        && let Some(p) = self.policies.get(policy)
                    {
                        return Principal {
                            id: id.to_string(),
                            source: PrincipalSource::Oauth,
                            grant: Grant::Policy(Arc::clone(p)),
                        };
                    }
                }
                Principal::nothing(id, PrincipalSource::Oauth)
            }
            Credential::None => Principal::nothing("anonymous", PrincipalSource::Anonymous),
        }
    }

    fn rule_matches(m: &PrincipalMatch, f: &OauthFacts<'_>) -> bool {
        if let Some(scope) = &m.scope {
            return f.scopes.iter().any(|s| s == scope);
        }
        if let Some(client) = &m.client_id {
            return f.client_id == Some(client.as_str());
        }
        if let Some(c) = &m.claim {
            return (f.has_claim_value)(&c.name, &c.value);
        }
        false
    }
}

/// A principal confined to the given tools, read prefixes and write prefixes, for
/// tests in other modules.
#[cfg(test)]
pub(crate) fn scoped_principal_for_test(
    tools: &[&str],
    read: &[&str],
    write: &[&str],
) -> Principal {
    let policies = BTreeMap::from([(
        "scoped".to_string(),
        PolicyConfig {
            tools: tools.iter().map(|s| s.to_string()).collect(),
            read: read.iter().map(|s| s.to_string()).collect(),
            write: write.iter().map(|s| s.to_string()).collect(),
            admin: false,
        },
    )]);
    let tokens = [TokenConfig {
        name: "scoped".into(),
        secret_env: "SCOPED_TOKEN".into(),
        policy: "scoped".into(),
    }];
    let known: Vec<&str> = tools.iter().copied().filter(|t| *t != ALL_TOOLS).collect();
    AccessControl::compile(&policies, &tokens, &[], &known, false)
        .unwrap()
        .principal(Credential::Static(Some("scoped")))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOOLS: &[&str] = &[
        "search",
        "get_document",
        "write_document",
        "delete_document",
    ];

    fn policies() -> BTreeMap<String, PolicyConfig> {
        BTreeMap::from([
            (
                "memory".to_string(),
                PolicyConfig {
                    tools: vec!["search".into(), "write_document".into()],
                    read: vec!["memory".into()],
                    write: vec!["memory/".into()],
                    admin: false,
                },
            ),
            (
                "ops".to_string(),
                PolicyConfig {
                    tools: vec![ALL_TOOLS.into()],
                    read: vec!["/".into()],
                    write: vec!["/".into()],
                    admin: true,
                },
            ),
        ])
    }

    fn token(name: &str, env: &str, policy: &str) -> TokenConfig {
        TokenConfig {
            name: name.into(),
            secret_env: env.into(),
            policy: policy.into(),
        }
    }

    fn compile(
        tokens: &[TokenConfig],
        rules: &[PrincipalRule],
        oauth: bool,
    ) -> Result<AccessControl> {
        AccessControl::compile(&policies(), tokens, rules, TOOLS, oauth)
    }

    #[test]
    fn no_policy_is_not_enforcing_and_everyone_is_unrestricted() {
        let ac = AccessControl::compile(&BTreeMap::new(), &[], &[], TOOLS, false).unwrap();
        assert!(!ac.enforcing());
        for cred in [Credential::Static(None), Credential::None] {
            let p = ac.principal(cred);
            assert_eq!(p.grant(), &Grant::Unrestricted);
            assert!(
                p.authorize("write_document", Some("any/where.md"), Op::Write)
                    .is_ok()
            );
            assert!(p.can_admin());
        }
    }

    #[test]
    fn a_named_token_gets_exactly_its_policy() {
        let ac = compile(&[token("hermes", "T_HERMES", "memory")], &[], false).unwrap();
        let p = ac.principal(Credential::Static(Some("hermes")));
        assert_eq!(p.id(), "hermes");
        assert!(p.authorize("search", Some("memory/a.md"), Op::Read).is_ok());
        assert!(
            p.authorize("write_document", Some("memory/x/a.md"), Op::Write)
                .is_ok()
        );
        // Outside its directories, outside its tools, beyond read, and the operator routes.
        assert_eq!(
            p.authorize("search", Some("notes/a.md"), Op::Read),
            Err(Denied)
        );
        assert_eq!(
            p.authorize("search", Some("memory-old/a.md"), Op::Read),
            Err(Denied)
        );
        assert_eq!(
            p.authorize("delete_document", Some("memory/a.md"), Op::Write),
            Err(Denied)
        );
        assert_eq!(p.authorize("get_document", None, Op::Read), Err(Denied));
        assert!(!p.can_admin());
    }

    #[test]
    fn the_legacy_token_stays_unrestricted_under_a_policy() {
        let ac = compile(&[token("hermes", "T_HERMES", "memory")], &[], false).unwrap();
        let p = ac.principal(Credential::Static(None));
        assert_eq!(p.id(), LEGACY_PRINCIPAL_ID);
        assert_eq!(p.grant(), &Grant::Unrestricted);
    }

    #[test]
    fn an_unmapped_credential_gets_nothing_once_a_policy_exists() {
        let ac = compile(&[], &[], false).unwrap();
        assert!(ac.enforcing());
        let unknown = ac.principal(Credential::Static(Some("ghost")));
        assert_eq!(unknown.grant(), &Grant::Nothing);
        assert_eq!(ac.principal(Credential::None).grant(), &Grant::Nothing);
        assert!(!unknown.has_access());
        assert_eq!(unknown.authorize_tool("search"), Err(Denied));
        assert_eq!(unknown.reach(Op::Read), Reach::Prefixes(vec![]));
        // A handler that finds no principal at all fails closed too.
        assert_eq!(
            Principal::from_extension(None, true).grant(),
            &Grant::Nothing
        );
        assert_eq!(
            Principal::from_extension(None, false).grant(),
            &Grant::Unrestricted
        );
    }

    #[test]
    fn oauth_rules_match_scope_client_and_claim_first_wins() {
        let rules = vec![
            PrincipalRule {
                matcher: PrincipalMatch {
                    scope: Some("mcp:ops".into()),
                    ..Default::default()
                },
                policy: "ops".into(),
            },
            PrincipalRule {
                matcher: PrincipalMatch {
                    client_id: Some("hermes-app".into()),
                    ..Default::default()
                },
                policy: "memory".into(),
            },
            PrincipalRule {
                matcher: PrincipalMatch {
                    claim: Some(ClaimMatch {
                        name: "groups".into(),
                        value: "agents".into(),
                    }),
                    ..Default::default()
                },
                policy: "memory".into(),
            },
        ];
        let ac = compile(&[], &rules, true).unwrap();
        let no_claims = |_: &str, _: &str| false;
        let scopes = vec!["mcp:read".to_string(), "mcp:ops".to_string()];
        let ops = ac.principal(Credential::Oauth(OauthFacts {
            subject: Some("u1"),
            scopes: &scopes,
            client_id: Some("hermes-app"),
            has_claim_value: &no_claims,
        }));
        assert!(ops.can_admin(), "the scope rule comes first");
        let read_only = vec!["mcp:read".to_string()];
        let by_client = ac.principal(Credential::Oauth(OauthFacts {
            subject: Some("u2"),
            scopes: &read_only,
            client_id: Some("hermes-app"),
            has_claim_value: &no_claims,
        }));
        assert_eq!(
            by_client.reach(Op::Read),
            Reach::Prefixes(vec!["memory".into()])
        );
        let in_group = |n: &str, v: &str| n == "groups" && v == "agents";
        let by_claim = ac.principal(Credential::Oauth(OauthFacts {
            subject: None,
            scopes: &read_only,
            client_id: None,
            has_claim_value: &in_group,
        }));
        assert_eq!(
            by_claim.reach(Op::Read),
            Reach::Prefixes(vec!["memory".into()])
        );
        let nobody = ac.principal(Credential::Oauth(OauthFacts {
            subject: Some("u3"),
            scopes: &read_only,
            client_id: Some("other"),
            has_claim_value: &no_claims,
        }));
        assert_eq!(nobody.grant(), &Grant::Nothing);
    }

    #[test]
    fn a_root_grant_is_all_and_prefixes_never_match_siblings() {
        let ac = compile(&[token("ops", "T_OPS", "ops")], &[], false).unwrap();
        assert_eq!(
            ac.principal(Credential::Static(Some("ops")))
                .reach(Op::Read),
            Reach::All
        );
        assert!(prefix_covers("notes", "notes"));
        assert!(prefix_covers("notes", "notes/a/b.md"));
        assert!(prefix_covers("notes", "/notes/a.md"));
        assert!(prefix_covers("notes", "./notes//a.md"));
        // Never resolved: any `..` is uncovered, whatever it would resolve to.
        assert!(!prefix_covers("notes", "notes/../secret/a.md"));
        assert!(!prefix_covers("", "../x.md"));
        assert!(!prefix_covers("notes", "notes-old/a.md"));
        assert!(!prefix_covers("notes", "other/notes/a.md"));
        assert!(prefix_covers("", "anything.md"));
    }

    #[test]
    fn normalize_prefix_refuses_traversal_git_and_globs() {
        assert_eq!(normalize_prefix("/memory//").unwrap(), "memory");
        assert_eq!(normalize_prefix("/memory/").unwrap(), "memory");
        assert_eq!(normalize_prefix(" / ").unwrap(), "");
        for bad in ["a/../b", "./a", ".git", "a/.GIT/b", "a/*", "a\\b", "a//b"] {
            assert!(normalize_prefix(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_write_grant_must_sit_inside_a_read_grant() {
        let policy = |read: &[&str], write: &[&str]| {
            let cfg = PolicyConfig {
                tools: vec![ALL_TOOLS.into()],
                read: read.iter().map(|s| s.to_string()).collect(),
                write: write.iter().map(|s| s.to_string()).collect(),
                admin: false,
            };
            AccessControl::compile(
                &BTreeMap::from([("p".to_string(), cfg)]),
                &[],
                &[],
                TOOLS,
                false,
            )
        };
        // A write returns a diff and answers "anchor matched or not", so writing
        // what you cannot read is reading it.
        for (read, write) in [
            (&[][..], &["memory"][..]),
            (&["memory"], &["notes"]),
            (&["memory/general"], &["memory"]),
            (&["memory"], &["/"]),
        ] {
            let e = policy(read, write).unwrap_err().to_string();
            assert!(
                e.contains("mcp.policy.p.write") && e.contains("read"),
                "{e}"
            );
        }
        assert!(policy(&["memory"], &["memory"]).is_ok());
        assert!(policy(&["memory"], &["memory/general"]).is_ok());
        assert!(policy(&["/"], &["notes", "memory"]).is_ok());
        assert!(policy(&["memory"], &[]).is_ok());
    }

    #[test]
    fn config_errors_name_the_setting() {
        let mut p = policies();
        p.get_mut("memory").unwrap().tools.push("teleport".into());
        let e = AccessControl::compile(&p, &[], &[], TOOLS, false)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("mcp.policy.memory.tools") && e.contains("teleport"),
            "{e}"
        );

        for (tokens, needle) in [
            (
                vec![token("a", "T", "nope")],
                "not defined under mcp.policy",
            ),
            (
                vec![token("a", "T1", "memory"), token("a", "T2", "memory")],
                "used twice",
            ),
            (
                vec![token("a", "T", "memory"), token("b", "T", "memory")],
                "secret_env",
            ),
            (vec![token("bad name", "T", "memory")], "visible ASCII"),
            (vec![token(LEGACY_PRINCIPAL_ID, "T", "memory")], "reserved"),
            (vec![token("a", " ", "memory")], "secret_env"),
        ] {
            let e = compile(&tokens, &[], false).unwrap_err().to_string();
            assert!(e.contains(needle), "{e} should contain {needle}");
        }

        let rule = |m: PrincipalMatch| PrincipalRule {
            matcher: m,
            policy: "memory".into(),
        };
        let none = rule(PrincipalMatch::default());
        let two = rule(PrincipalMatch {
            scope: Some("a".into()),
            client_id: Some("b".into()),
            claim: None,
        });
        for r in [none, two] {
            assert!(
                compile(&[], &[r], true)
                    .unwrap_err()
                    .to_string()
                    .contains("exactly one")
            );
        }
        let ok = rule(PrincipalMatch {
            scope: Some("a".into()),
            ..Default::default()
        });
        assert!(
            compile(&[], &[ok], false)
                .unwrap_err()
                .to_string()
                .contains("mcp.oauth")
        );
        assert!(
            AccessControl::compile(
                &BTreeMap::new(),
                &[token("a", "T", "memory")],
                &[],
                TOOLS,
                false
            )
            .is_err()
        );
    }
}
