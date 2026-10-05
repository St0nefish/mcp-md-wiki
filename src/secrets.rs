//! Token lookup for the three secret-bearing env vars (#332).
//!
//! Each token is named by config (`source.git_token_env`, `webhook.secret_env`,
//! `mcp.bearer_token_env`) and read as `<name>`, or from the file named by
//! `<name>_FILE` (Docker secrets mount at `/run/secrets/<name>`). Setting both is an
//! error. The semantics — surrounding whitespace trimmed, a blank `<name>` read as
//! unset, an empty `<name>_FILE` an error — are `oauth_resource_server::env`'s; this
//! module only fixes which variables go through it.

use oauth_resource_server::env::{EnvError, secret_from_env};

use crate::config::ResolvedConfig;

/// Resolve the secret configured under the env var `var`, reading `<var>_FILE`
/// when set. `Ok(None)` when neither form is set.
pub fn resolve_secret(var: &str) -> Result<Option<String>, EnvError> {
    secret_from_env(var)
}

/// The git-pull token for the KB clone. Read fresh on every call, so a rotated
/// secret file takes effect without a restart.
pub fn git_token(config: &ResolvedConfig) -> Result<Option<String>, EnvError> {
    resolve_secret(&config.source.git_token_env)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io;

    use oauth_resource_server::env::secret_from_lookup;

    use super::*;

    /// The three configurable token env vars, with their defaults.
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
        secret_from_lookup(
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
            let msg = err.to_string();
            assert!(msg.contains(var), "{msg}");
        }
    }

    #[test]
    fn neither_set_is_none() {
        for var in TOKEN_VARS {
            assert_eq!(resolve(var, &[], &[]).unwrap(), None, "{var}");
        }
    }

    #[test]
    fn blank_env_is_unset() {
        assert_eq!(
            resolve("GIT_PULL_TOKEN", &[("GIT_PULL_TOKEN", "")], &[]).unwrap(),
            None
        );
    }

    #[test]
    fn empty_or_missing_file_is_an_error() {
        let env = [("GIT_PULL_TOKEN_FILE", "/run/secrets/t")];
        let empty = resolve("GIT_PULL_TOKEN", &env, &[("/run/secrets/t", "\n")]).unwrap_err();
        assert!(matches!(empty, EnvError::EmptyFile { .. }), "{empty}");
        let missing = resolve("GIT_PULL_TOKEN", &env, &[]).unwrap_err();
        assert!(matches!(missing, EnvError::ReadFailed { .. }), "{missing}");
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
