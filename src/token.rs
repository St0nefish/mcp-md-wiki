//! `token mint`: generate a secret for an `mcp.tokens` entry and write it to a file.
//!
//! The secret goes to the file the operator names and nowhere else: it is never
//! printed, logged or placed in a process argument, so it cannot end up in shell
//! history or a CI log. The command prints what it did (the token name, the policy
//! it acts under and the file) plus the `mcp.tokens` entry to add. The file is created
//! with mode `0600` and the command refuses to overwrite one that exists, so a
//! second run can never silently replace a secret a client is already using.

use std::io::Write as _;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::policy::{AccessControl, PolicyConfig};

/// Bytes of randomness in a minted secret (256 bits).
const SECRET_BYTES: usize = 32;

/// A prefix that makes a leaked token recognisable in a secret scanner.
const SECRET_PREFIX: &str = "mdwiki_";

/// A fresh secret: [`SECRET_PREFIX`] and 64 hex characters from the OS random
/// source.
fn generate_secret() -> Result<String> {
    let mut bytes = [0u8; SECRET_BYTES];
    getrandom::fill(&mut bytes)
        .map_err(|e| anyhow::anyhow!("the OS random source is unavailable: {e}"))?;
    Ok(format!("{SECRET_PREFIX}{}", hex::encode(bytes)))
}

/// The environment variable a token named `name` is conventionally read from.
fn suggested_env(name: &str) -> String {
    let mut env = String::from("MCP_TOKEN_");
    env.extend(name.chars().map(|c| {
        if c.is_ascii_alphanumeric() {
            c.to_ascii_uppercase()
        } else {
            '_'
        }
    }));
    env
}

/// Write `secret` to `out`, new, owner-read/write only, with a trailing newline.
fn write_secret_file(out: &Path, secret: &str) -> Result<()> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let mut file = options.open(out).with_context(|| {
        format!(
            "cannot create {} (it must not exist already: a minted secret is never overwritten)",
            out.display()
        )
    })?;
    file.write_all(secret.as_bytes())
        .and_then(|()| file.write_all(b"\n"))
        .and_then(|()| file.flush())
        .with_context(|| format!("cannot write {}", out.display()))
}

/// Mint a token for `name` under `policy` into `out`, returning the text to print.
/// `policies` is the configured `mcp.policy`, so a typo in the policy name is
/// refused here rather than at the next server start.
pub fn mint(
    name: &str,
    policy: &str,
    out: &Path,
    policies: &std::collections::BTreeMap<String, PolicyConfig>,
) -> Result<String> {
    if !policies.contains_key(policy) {
        let known: Vec<&str> = policies.keys().map(String::as_str).collect();
        bail!(
            "policy '{policy}' is not defined under mcp.policy (defined: {})",
            if known.is_empty() {
                "none".to_string()
            } else {
                known.join(", ")
            }
        );
    }
    // The same name rules the config applies, so the entry printed below loads.
    let env = suggested_env(name);
    AccessControl::compile(
        policies,
        &[crate::policy::TokenConfig {
            name: name.to_string(),
            secret_env: env.clone(),
            policy: policy.to_string(),
        }],
        &[],
        &crate::descriptions::TOOL_NAMES,
        false,
    )
    .context("not a usable token")?;
    let secret = generate_secret()?;
    write_secret_file(out, &secret)?;
    Ok(format!(
        "Minted token '{name}' for policy '{policy}' into {path} (mode 0600).\n\
         The secret is only in that file; it is not shown here.\n\n\
         Add to config.yaml:\n\
         \x20 mcp:\n\
         \x20   tokens:\n\
         \x20     - name: {name}\n\
         \x20       secret_env: {env}\n\
         \x20       policy: {policy}\n\n\
         Give the server the secret as {env}_FILE={path}, and give the client the same \
         secret as its bearer token. A new token takes effect on restart.",
        path = out.display(),
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn policies() -> BTreeMap<String, PolicyConfig> {
        BTreeMap::from([("memory".to_string(), PolicyConfig::default())])
    }

    #[test]
    fn a_minted_secret_is_random_prefixed_and_hex() {
        let a = generate_secret().unwrap();
        let b = generate_secret().unwrap();
        assert_ne!(a, b);
        let hex_part = a.strip_prefix(SECRET_PREFIX).unwrap();
        assert_eq!(hex_part.len(), SECRET_BYTES * 2);
        assert!(hex_part.bytes().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn mint_writes_a_private_file_and_never_prints_the_secret() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("hermes.token");
        let text = mint("hermes", "memory", &out, &policies()).unwrap();
        let secret = std::fs::read_to_string(&out).unwrap();
        let secret = secret.trim();
        assert!(secret.starts_with(SECRET_PREFIX) && secret.len() > 40);
        assert!(!text.contains(secret), "the secret must not be printed");
        assert!(text.contains("MCP_TOKEN_HERMES") && text.contains("policy: memory"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(&out).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
    }

    #[test]
    fn mint_never_overwrites_and_refuses_an_unknown_policy_or_bad_name() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("t");
        std::fs::write(&out, "keep me").unwrap();
        assert!(mint("a", "memory", &out, &policies()).is_err());
        assert_eq!(std::fs::read_to_string(&out).unwrap(), "keep me");

        let fresh = dir.path().join("fresh");
        let e = mint("a", "nope", &fresh, &policies())
            .unwrap_err()
            .to_string();
        assert!(e.contains("not defined") && e.contains("memory"), "{e}");
        assert!(!fresh.exists());
        assert!(mint("bad name", "memory", &fresh, &policies()).is_err());
        assert!(mint("default", "memory", &fresh, &policies()).is_err());
        assert!(!fresh.exists());
    }

    #[test]
    fn the_suggested_env_name_is_shell_safe() {
        assert_eq!(suggested_env("hermes-1.a"), "MCP_TOKEN_HERMES_1_A");
    }
}
