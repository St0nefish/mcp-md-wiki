//! Env-var lookup for the secret-bearing settings, with Docker-secrets `_FILE`
//! support (#332).
//!
//! Each secret is named by config (`source.git_token_env`, `webhook.secret_env`,
//! `mcp.bearer_token_env`, `embedding.api_key_env`, `reranking.api_key_env`) and is
//! read as `<name>`, or from the file named by `<name>_FILE`. The lookup is
//! `oauth_resource_server::env::config_value_from_env`, not `secret_from_env`: a set
//! `<name>` comes back exactly as set — untrimmed, and an empty one as `Some("")` — so
//! every value this server read before `_FILE` existed is read identically now, and
//! only a file's contents are trimmed. Setting both forms (a blank `<name>` yields to
//! the file) or pointing `_FILE` at an empty or unreadable file is an error.

use std::io;

use oauth_resource_server::env::{
    EnvError, config_value_from_env, config_value_from_lookup, read_secret_file,
};

use crate::config::ResolvedConfig;

/// The secret under env var `var`, or `<var>_FILE`'s contents. `Ok(None)` when
/// neither is set. A set-but-empty `var` is `Some("")`, as `std::env::var(..).ok()`
/// returned it.
pub fn resolve_secret(var: &str) -> Result<Option<String>, EnvError> {
    config_value_from_env(var)
}

/// [`resolve_secret`] with an empty value read as unset — the lookup for settings
/// whose empty value has always meant "not configured" (the git token, the webhook
/// secret).
pub fn resolve_nonempty_secret(var: &str) -> Result<Option<String>, EnvError> {
    nonempty_with(var, |v| std::env::var(v).ok(), read_secret_file)
}

/// [`resolve_nonempty_secret`] with the lookup and file reader injected — the one
/// definition of "empty reads as unset", shared with [`StartupSecrets`].
fn nonempty_with(
    var: &str,
    lookup: impl Fn(&str) -> Option<String>,
    read_file: impl Fn(&str) -> io::Result<String>,
) -> Result<Option<String>, EnvError> {
    config_value_from_lookup(var, lookup, read_file).map(|v| v.filter(|s| !s.is_empty()))
}

/// The variable a resolved secret actually came from, for provenance: `<var>_FILE` when
/// that is set (a non-blank plain `<var>` alongside it is an error, so the file is the
/// source), else `<var>`.
pub fn source_var(var: &str) -> String {
    let file_var = format!("{var}_FILE");
    match std::env::var(&file_var) {
        Ok(path) if !path.trim().is_empty() => file_var,
        _ => var.to_string(),
    }
}

/// The git-pull token for the KB clone. Read on every call, so a rotated secret
/// file takes effect without a restart.
pub fn git_token(config: &ResolvedConfig) -> Result<Option<String>, EnvError> {
    resolve_nonempty_secret(&config.source.git_token_env)
}

/// The three secrets `run_server` resolves once at startup. Resolving them together,
/// before anything is built from them, is what makes a bad `<NAME>_FILE`
/// combination refuse to start rather than surface on the first request.
#[derive(Debug, PartialEq, Eq)]
pub struct StartupSecrets {
    pub git_token: Option<String>,
    pub bearer_token: Option<String>,
    pub webhook_secret: Option<String>,
}

/// The env var names [`StartupSecrets`] reads (the `*_env` config fields).
#[derive(Clone, Copy)]
pub struct SecretNames<'a> {
    pub git_token: &'a str,
    pub bearer_token: &'a str,
    pub webhook_secret: &'a str,
}

impl StartupSecrets {
    pub fn resolve(config: &ResolvedConfig) -> Result<Self, EnvError> {
        Self::resolve_with(
            SecretNames {
                git_token: &config.source.git_token_env,
                bearer_token: &config.mcp.bearer_token_env,
                webhook_secret: &config.webhook.secret_env,
            },
            |v| std::env::var(v).ok(),
            read_secret_file,
        )
    }

    /// [`Self::resolve`] with the variable lookup and file reader injected, so tests
    /// never touch the process environment or the filesystem.
    pub fn resolve_with(
        names: SecretNames<'_>,
        lookup: impl Fn(&str) -> Option<String>,
        read_file: impl Fn(&str) -> io::Result<String>,
    ) -> Result<Self, EnvError> {
        let nonempty = |var: &str| nonempty_with(var, &lookup, &read_file);
        Ok(Self {
            git_token: nonempty(names.git_token)?,
            // Not filtered: `static_token_policy` owns what a blank bearer token means.
            bearer_token: config_value_from_lookup(names.bearer_token, &lookup, &read_file)?,
            webhook_secret: nonempty(names.webhook_secret)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    /// The three configurable startup secrets, with their defaults.
    const TOKEN_VARS: [&str; 3] = ["GIT_PULL_TOKEN", "WEBHOOK_SECRET", "MCP_BEARER_TOKEN"];

    fn resolve(
        var: &str,
        env: &[(&str, &str)],
        files: &[(&str, &str)],
    ) -> Result<Option<String>, EnvError> {
        let env: HashMap<String, String> = env
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let files: HashMap<String, String> = files
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        config_value_from_lookup(
            var,
            |name| env.get(name).cloned(),
            |path| {
                files
                    .get(path)
                    .cloned()
                    .ok_or_else(|| io::Error::from(io::ErrorKind::NotFound))
            },
        )
    }

    #[test]
    fn env_only() {
        for var in TOKEN_VARS {
            assert_eq!(
                resolve(var, &[(var, "tok")], &[]).unwrap().as_deref(),
                Some("tok"),
                "{var}"
            );
        }
    }

    #[test]
    fn file_only_trims_trailing_newline() {
        for var in TOKEN_VARS {
            let file_var = format!("{var}_FILE");
            let got = resolve(
                var,
                &[(&file_var, "/run/secrets/t")],
                &[("/run/secrets/t", "tok\n")],
            )
            .unwrap();
            assert_eq!(got.as_deref(), Some("tok"), "{var}");
        }
    }

    #[test]
    fn both_set_is_an_error() {
        for var in TOKEN_VARS {
            let file_var = format!("{var}_FILE");
            let err = resolve(
                var,
                &[(var, "a"), (&file_var, "/run/secrets/t")],
                &[("/run/secrets/t", "b")],
            )
            .unwrap_err();
            assert!(matches!(err, EnvError::BothSet { .. }), "{var}: {err}");
            assert!(err.to_string().contains(var), "{err}");
        }
    }

    #[test]
    fn neither_set_is_none() {
        for var in TOKEN_VARS {
            assert_eq!(resolve(var, &[], &[]).unwrap(), None, "{var}");
        }
    }

    /// A plain variable is read exactly as `std::env::var` returned it: untrimmed,
    /// and empty is a value (callers that treat empty as unset filter it themselves).
    #[test]
    fn plain_variable_is_returned_as_set() {
        let padded = resolve("K", &[("K", "  tok \n")], &[]).unwrap();
        assert_eq!(padded.as_deref(), Some("  tok \n"));
        let empty = resolve("K", &[("K", "")], &[]).unwrap();
        assert_eq!(empty.as_deref(), Some(""));
    }

    /// `docker-compose.yml` always passes `X=${X:-}`, so a blank `X` next to a
    /// `X_FILE` must not read as "both set".
    #[test]
    fn blank_variable_yields_to_the_file() {
        let got = resolve(
            "K",
            &[("K", ""), ("K_FILE", "/run/secrets/k")],
            &[("/run/secrets/k", "tok\n")],
        )
        .unwrap();
        assert_eq!(got.as_deref(), Some("tok"));
    }

    #[test]
    fn empty_or_missing_file_is_an_error() {
        let env = [("GIT_PULL_TOKEN_FILE", "/run/secrets/t")];
        let empty = resolve("GIT_PULL_TOKEN", &env, &[("/run/secrets/t", "\n")]).unwrap_err();
        assert!(matches!(empty, EnvError::EmptyFile { .. }), "{empty}");
        let missing = resolve("GIT_PULL_TOKEN", &env, &[]).unwrap_err();
        assert!(matches!(missing, EnvError::ReadFailed { .. }), "{missing}");
    }

    fn startup(env: &[(&str, &str)], files: &[(&str, &str)]) -> Result<StartupSecrets, EnvError> {
        let env: HashMap<String, String> = env
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let files: HashMap<String, String> = files
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        StartupSecrets::resolve_with(
            SecretNames {
                git_token: "GIT_PULL_TOKEN",
                bearer_token: "MCP_BEARER_TOKEN",
                webhook_secret: "WEBHOOK_SECRET",
            },
            |name| env.get(name).cloned(),
            |path| {
                files
                    .get(path)
                    .cloned()
                    .ok_or_else(|| io::Error::from(io::ErrorKind::NotFound))
            },
        )
    }

    #[test]
    fn startup_resolves_each_secret_from_its_own_source() {
        let got = startup(
            &[
                ("GIT_PULL_TOKEN_FILE", "/run/secrets/git"),
                ("MCP_BEARER_TOKEN", "bearer"),
                ("WEBHOOK_SECRET", ""),
            ],
            &[("/run/secrets/git", "gittok\n")],
        )
        .unwrap();
        assert_eq!(
            got,
            StartupSecrets {
                git_token: Some("gittok".into()),
                bearer_token: Some("bearer".into()),
                // An empty webhook secret still means "webhook disabled".
                webhook_secret: None,
            }
        );
    }

    /// Any one of the three secrets in a bad state aborts the whole startup
    /// resolution, naming the variable — `run_server` propagates it with `?`.
    #[test]
    fn startup_aborts_when_either_form_conflicts_for_any_secret() {
        for var in TOKEN_VARS {
            let file_var = format!("{var}_FILE");
            let err = startup(
                &[(var, "a"), (&file_var, "/run/secrets/t")],
                &[("/run/secrets/t", "b")],
            )
            .unwrap_err();
            assert!(matches!(err, EnvError::BothSet { .. }), "{var}: {err}");
            assert!(err.to_string().contains(var), "{err}");

            let missing = startup(&[(&file_var, "/run/secrets/gone")], &[]).unwrap_err();
            assert!(
                matches!(missing, EnvError::ReadFailed { .. }),
                "{var}: {missing}"
            );
        }
    }

    /// `_FILE` is derived from the *configured* name, not the default one.
    #[test]
    fn custom_configured_name_gets_its_own_file_variant() {
        let got = resolve(
            "MY_KB_TOKEN",
            &[("MY_KB_TOKEN_FILE", "/run/secrets/kb")],
            &[("/run/secrets/kb", "tok")],
        )
        .unwrap();
        assert_eq!(got.as_deref(), Some("tok"));
        // The default name's `_FILE` is not consulted for a renamed variable.
        let none = resolve(
            "MY_KB_TOKEN",
            &[("GIT_PULL_TOKEN_FILE", "/run/secrets/kb")],
            &[("/run/secrets/kb", "tok")],
        )
        .unwrap();
        assert_eq!(none, None);
    }
}
