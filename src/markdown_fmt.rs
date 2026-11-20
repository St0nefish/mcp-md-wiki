//! Markdown auto-formatting applied to every `.md` document the write tools store.
//!
//! A document written through this server is normalised in memory by rumdl's
//! auto-fixes (blank lines around headings and lists, trailing whitespace, list
//! markers, fenced-code style, ...) before it is committed and indexed, so a
//! knowledge base managed by this tool stays consistently formatted whatever client
//! wrote to it, and without anyone running a formatter by hand.
//!
//! # What it does and does not touch
//!
//! - **Formatting only.** rumdl knows nothing about this server's frontmatter schema, so
//!   it can neither validate nor repair field values; that stays with
//!   [`crate::validate`]. After formatting, the frontmatter is parsed again and must
//!   mean exactly what it did before (rumdl's two frontmatter rules, MD071 blank line
//!   after frontmatter and MD072 key order, are whitespace and ordering respectively;
//!   MD072 is off by default and ordering is the writer's business). If it does not,
//!   the formatting is discarded for that document ([`FormatError::FrontmatterChanged`]).
//!   The check compares parsed fields, so a frontmatter block that is not valid YAML
//!   (and reads as no fields on both sides) is not covered; the write path's schema
//!   validation is what refuses such a block.
//! - **In memory.** No path is given to rumdl, so rules that look at the filesystem
//!   (relative links, navigation) stay inert and nothing is read or written. The only
//!   file read is the optional `formatting.markdown.config_file`.
//! - **Deterministic and idempotent:** `format(format(x)) == format(x)`. A fix loop that
//!   does not converge (two rules undoing each other) is an error, never a half-applied
//!   result.
//!
//! # Policy
//!
//! The built-in default is [`DEFAULT_DISABLED_RULES`] on rumdl's `standard` flavor,
//! everything else at rumdl's own defaults. `formatting.markdown.disable` and `flavor`
//! change that; `formatting.markdown.config_file` points at a full `.rumdl.toml` and
//! replaces both. The version of rumdl is pinned exactly in `Cargo.toml` because its fix
//! behaviour defines the bytes written into the knowledge base.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use rumdl_lib::config::{Config, MarkdownFlavor, SourcedConfig};
use rumdl_lib::fix_coordinator::FixCoordinator;
use rumdl_lib::rule::Rule;
use rumdl_lib::rules::{all_rules, filter_rules};

/// Rules off by default: `MD013` (line length) is noise in knowledge-base prose full of
/// URLs, tables and pasted configuration.
pub const DEFAULT_DISABLED_RULES: &[&str] = &["MD013"];

/// Upper bound on fix passes. rumdl re-lints between iterations and stops when the
/// content is stable; this only bounds two rules that keep undoing each other.
const MAX_FIX_ITERATIONS: usize = 10;

/// What to format with. Built once per write from the live `formatting.markdown` config
/// ([`crate::config::MarkdownFormatConfig`]).
#[derive(Debug, Clone)]
pub struct Settings {
    pub flavor: MarkdownFlavor,
    pub disable: Vec<String>,
    /// A `.rumdl.toml` replacing `flavor` and `disable`, already resolved against the
    /// knowledge-base root.
    pub config_file: Option<PathBuf>,
}

impl Settings {
    /// The settings a `formatting.markdown` section describes. `root` is the
    /// knowledge-base root a relative `config_file` is resolved against.
    pub fn from_config(config: &crate::config::MarkdownFormatConfig, root: &Path) -> Self {
        Self {
            // Validated at config load; a value that no longer parses (a hand-built
            // config in a test) falls back to the default flavor.
            flavor: MarkdownFlavor::from_str(&config.flavor).unwrap_or_default(),
            disable: config.disable.clone(),
            config_file: config
                .config_file
                .as_deref()
                .map(|file| root.join(file.trim_start_matches('/'))),
        }
    }
}

/// A successful format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Formatted {
    /// The normalised document.
    pub content: String,
    /// Ids of the rules that changed something, sorted; empty when the input was already
    /// clean.
    pub rules_applied: Vec<String>,
}

/// Why a document was left as it was.
#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error("the rumdl config file could not be loaded: {0}")]
    Config(String),
    #[error("markdown auto-fix failed: {0}")]
    Fix(String),
    #[error("markdown auto-fix did not converge ({0})")]
    NotConverged(String),
    #[error("formatting would change the frontmatter, so it was not applied")]
    FrontmatterChanged,
    #[error("the markdown formatter panicked")]
    Panicked,
}

impl FormatError {
    /// A reason safe to show a tool caller: no filesystem path, no library internals.
    /// The full error goes to the server log.
    pub fn public_reason(&self) -> String {
        match self {
            Self::Config(_) => "the configured rumdl config file could not be loaded".into(),
            Self::Fix(_) => "the markdown formatter failed".into(),
            Self::NotConverged(_) => "the markdown formatter did not settle on one result".into(),
            Self::FrontmatterChanged => {
                "formatting would have changed the frontmatter, so it was not applied".into()
            }
            Self::Panicked => "the markdown formatter failed".into(),
        }
    }
}

/// Check a `formatting.markdown` section's values: every `disable` entry looks like a
/// rule id, the flavor is one rumdl knows. Returns the first problem.
pub fn validate_config(config: &crate::config::MarkdownFormatConfig) -> Result<(), String> {
    MarkdownFlavor::from_str(&config.flavor).map_err(|e| {
        format!("formatting.markdown.flavor: {e} (for example standard, mkdocs, obsidian)")
    })?;
    for rule in &config.disable {
        // rumdl's own registry: a typo (`MD0033`, `MD999`) is refused rather than
        // silently disabling nothing. Aliases (`line-length`) are accepted; `all` is
        // not, since disabling every rule is `enabled: false`.
        if rule.eq_ignore_ascii_case("all") || !rumdl_lib::config::is_valid_rule_name(rule) {
            return Err(format!(
                "formatting.markdown.disable: '{rule}' is not a rumdl rule (for example MD013 or line-length)"
            ));
        }
    }
    if let Some(file) = &config.config_file {
        if file.trim().is_empty() {
            return Err("formatting.markdown.config_file must not be empty when set".to_string());
        }
        if Path::new(file)
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(
                "formatting.markdown.config_file must stay inside the knowledge base (no '..')"
                    .to_string(),
            );
        }
    }
    Ok(())
}

fn build_config(settings: &Settings, root_hint: &Path) -> Result<Config, FormatError> {
    if let Some(file) = &settings.config_file {
        let root = file.parent().unwrap_or(root_hint);
        return SourcedConfig::load_config_for_path(file, root)
            .map_err(|e| FormatError::Config(format!("{}: {e}", file.display())));
    }
    let mut config = Config::default();
    config.global.disable = settings.disable.iter().map(|r| r.to_uppercase()).collect();
    // The fix loop builds its analysis context from `config.global.flavor`, not from
    // an argument, so the flavor has to live in the config.
    config.global.flavor = settings.flavor;
    Ok(config)
}

/// Format `content`, a whole markdown document (frontmatter included).
pub fn format(content: &str, settings: &Settings) -> Result<Formatted, FormatError> {
    // A rumdl bug must not take the write task down with it.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        format_inner(content, settings)
    }));
    match outcome {
        Ok(result) => result,
        Err(_) => Err(FormatError::Panicked),
    }
}

fn format_inner(content: &str, settings: &Settings) -> Result<Formatted, FormatError> {
    let config = build_config(settings, Path::new("."))?;
    let rules: Vec<Box<dyn Rule>> = filter_rules(&all_rules(&config), &config.global);

    let mut out = content.to_string();
    // The coordinator re-checks every rule on each pass and ignores the warnings it
    // is handed (kept for API compatibility), so none are computed up front.
    let result = FixCoordinator::new()
        .apply_fixes_iterative(&rules, &[], &mut out, &config, MAX_FIX_ITERATIONS, None)
        .map_err(|e| FormatError::Fix(e.to_string()))?;

    if !result.converged {
        return Err(FormatError::NotConverged(
            if result.conflict_cycle.is_empty() {
                format!("hit the {MAX_FIX_ITERATIONS}-iteration cap without stabilising")
            } else {
                format!("rules oscillate: {}", result.conflict_cycle.join(" -> "))
            },
        ));
    }

    // Formatting must never change what the frontmatter says.
    if out != content
        && crate::validate::parse_frontmatter_raw(content).0
            != crate::validate::parse_frontmatter_raw(&out).0
    {
        return Err(FormatError::FrontmatterChanged);
    }

    let mut rules_applied: Vec<String> = result.fixed_rule_names.into_iter().collect();
    rules_applied.sort();
    Ok(Formatted {
        content: out,
        rules_applied,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> Settings {
        Settings {
            flavor: MarkdownFlavor::Standard,
            disable: DEFAULT_DISABLED_RULES
                .iter()
                .map(|r| (*r).to_string())
                .collect(),
            config_file: None,
        }
    }

    const MESSY: &str = "---\ntitle: Notes\ntags: [a, b]\n---\n# Notes\nSome text with trailing spaces   \n## Section\n* one\n* two\n\n\n\nMore text.\n";

    #[test]
    fn a_messy_document_is_normalised() {
        let formatted = format(MESSY, &defaults()).unwrap();
        assert_ne!(formatted.content, MESSY);
        assert!(!formatted.rules_applied.is_empty(), "{formatted:?}");
        assert!(
            !formatted.content.contains("spaces   \n"),
            "trailing spaces stay: {:?}",
            formatted.content
        );
        assert!(
            !formatted.content.contains("\n\n\n"),
            "runs of blank lines stay: {:?}",
            formatted.content
        );
        assert!(
            formatted.content.contains("# Notes\n\nSome text"),
            "no blank line after the heading: {:?}",
            formatted.content
        );
    }

    /// With a `title:` in the frontmatter the document already has its H1 (rumdl's
    /// MD025 reads the frontmatter title), so a body `# Heading` is demoted. This is
    /// the convention the knowledge base follows (title in frontmatter, sections from
    /// `##`); a corpus that wants a body H1 as well turns MD025 off.
    #[test]
    fn a_body_h1_under_a_frontmatter_title_is_demoted_unless_md025_is_disabled() {
        let doc = "---\ntitle: T\n---\n\n# T\n\nText.\n";
        let on = format(doc, &defaults()).unwrap();
        assert_eq!(on.content, "---\ntitle: T\n---\n\n## T\n\nText.\n");
        assert!(on.rules_applied.contains(&"MD025".to_string()), "{on:?}");

        let mut off = defaults();
        off.disable.push("MD025".into());
        assert_eq!(format(doc, &off).unwrap().content, doc);
    }

    /// The fix loop reads the flavor from the config, not from an argument: a
    /// non-standard flavor has to change what the fixes do, not just the first lint.
    /// An mkdocs admonition body is indented text, so a tab in it is a hard tab to fix
    /// (MD010); under the standard flavor the same lines are an indented code block,
    /// where the tab is kept.
    #[test]
    fn the_configured_flavor_reaches_the_fixes() {
        let doc = "---\ntitle: T\n---\n\n## S\n\n!!! note\n\n    text\twith tab\n";
        let standard = format(doc, &defaults()).unwrap();
        assert_eq!(
            standard.content, doc,
            "an indented code block keeps its tab"
        );
        let mut mk = defaults();
        mk.flavor = MarkdownFlavor::MkDocs;
        let mkdocs = format(doc, &mk).unwrap();
        assert_eq!(
            mkdocs.content,
            "---\ntitle: T\n---\n\n## S\n\n!!! note\n\n    text    with tab\n"
        );
        assert!(mkdocs.rules_applied.contains(&"MD010".to_string()));
    }

    /// An alias in `disable` has to switch the rule off, not just pass validation.
    #[test]
    fn a_rule_alias_in_the_disable_list_disables_the_rule() {
        let doc = "---\ntitle: T\n---\n\n## S\n\ntrailing   \n";
        assert_ne!(format(doc, &defaults()).unwrap().content, doc);
        for name in ["MD009", "md009", "no-trailing-spaces"] {
            let mut s = defaults();
            s.disable.push(name.into());
            assert_eq!(
                format(doc, &s).unwrap().content,
                doc,
                "{name} must disable MD009"
            );
        }
    }

    #[test]
    fn formatting_is_idempotent() {
        let once = format(MESSY, &defaults()).unwrap();
        let twice = format(&once.content, &defaults()).unwrap();
        assert_eq!(twice.content, once.content);
        assert!(twice.rules_applied.is_empty(), "{twice:?}");
    }

    #[test]
    fn a_clean_document_is_untouched() {
        let clean = "---\ntitle: T\n---\n\n## T\n\nA paragraph.\n";
        let formatted = format(clean, &defaults()).unwrap();
        assert_eq!(formatted.content, clean);
        assert!(formatted.rules_applied.is_empty());
    }

    #[test]
    fn frontmatter_means_the_same_after_formatting() {
        let before = crate::validate::parse_frontmatter_raw(MESSY).0;
        let formatted = format(MESSY, &defaults()).unwrap();
        assert_eq!(
            crate::validate::parse_frontmatter_raw(&formatted.content).0,
            before
        );
    }

    #[test]
    fn a_long_line_is_left_alone_by_default_and_wrapped_never_by_the_defaults() {
        let long = format!("# T\n\n{}\n", "word ".repeat(60).trim_end());
        let formatted = format(&long, &defaults()).unwrap();
        assert_eq!(formatted.content, long, "MD013 is off by default");
    }

    #[test]
    fn the_disable_list_switches_a_rule_off() {
        let doc = "# T\n\ntext   \n";
        let on = format(doc, &defaults()).unwrap();
        assert!(on.rules_applied.contains(&"MD009".to_string()), "{on:?}");

        let mut off = defaults();
        off.disable.push("MD009".into());
        let formatted = format(doc, &off).unwrap();
        assert_eq!(formatted.content, doc);
    }

    #[test]
    fn a_rumdl_config_file_replaces_the_inline_settings() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join(".rumdl.toml");
        std::fs::write(&file, "[global]\ndisable = [\"MD009\"]\n").unwrap();
        let mut settings = defaults();
        settings.config_file = Some(file);
        let doc = "# T\n\ntext   \n";
        assert_eq!(format(doc, &settings).unwrap().content, doc);
    }

    #[test]
    fn a_missing_or_broken_config_file_is_an_error_not_a_silent_default() {
        let mut settings = defaults();
        settings.config_file = Some(PathBuf::from("/nonexistent/.rumdl.toml"));
        assert!(matches!(
            format("# T\n", &settings),
            Err(FormatError::Config(_))
        ));
    }

    #[test]
    fn config_values_are_checked() {
        let mut config = crate::config::MarkdownFormatConfig::default();
        assert!(validate_config(&config).is_ok());
        config.flavor = "nonsense".into();
        assert!(validate_config(&config).unwrap_err().contains("flavor"));
        config.flavor = "mkdocs".into();
        config.disable = vec!["MD013".into(), "not-a-rule".into()];
        assert!(validate_config(&config).unwrap_err().contains("not-a-rule"));
        for typo in ["MD999", "MD0033", "all"] {
            config.disable = vec![typo.into()];
            assert!(validate_config(&config).is_err(), "{typo} must be refused");
        }
        for ok in ["md033", "MD033", "line-length"] {
            config.disable = vec![ok.into()];
            assert!(validate_config(&config).is_ok(), "{ok} must be accepted");
        }
        config.config_file = Some("../outside/.rumdl.toml".into());
        assert!(validate_config(&config).unwrap_err().contains("'..'"));
        config.config_file = None;
        config.config_file = Some("  ".into());
        assert!(
            validate_config(&config)
                .unwrap_err()
                .contains("config_file")
        );
    }
}
