//! The one module that names the YAML library.
//!
//! Every read and write of YAML text in this crate (`config.yaml`, `.schema.yaml`,
//! the evaluation file, the frontmatter this server writes back) goes through
//! [`from_str`] and [`to_string`], so the library's behaviour is chosen once, here:
//!
//! - **YAML 1.2 booleans.** Only `true`/`false` (any of the three cases) are booleans;
//!   `yes`, `no`, `on` and `off` are strings, as the previous `serde_yaml_ng` read
//!   them. A typed `bool` field given `yes` is an error, not `true`.
//! - **A duplicate mapping key is an error**, as before.
//! - **One-line errors.** No source snippet is rendered; a message ends
//!   `at line L, column C` when the position is known. Operators and MCP callers see
//!   these messages, so they stay one line.
//! - **Writing** is block style with two-space steps. A multi-line string is a literal
//!   block (`|`), a long single-line string stays on one line (the library would
//!   otherwise fold it at 80 columns), and a string that would read back as something
//!   else (`"123"`, `"yes"`, `"2026-01-02"`, one with `: ` or a leading `-`) is quoted.
//!
//! Compared with `serde_yaml_ng`, quoted strings are written with **double** quotes
//! (`"123"`, not `'123'`), a mapping key such as `n` or `yes` is quoted, and a null where
//! a `String` is expected is an error rather than the empty string. The library offers no
//! switch for the first; it is one-time churn on the first re-render of an entry that was
//! written with single quotes, and the written text reads back to the same value.
//!
//! The library is pinned exactly in `Cargo.toml`: its option defaults and scalar
//! resolution decide how every YAML file this server reads is understood, so an upgrade
//! is a reviewed commit (the golden tests below pin the written form).

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_saphyr::{DuplicateKeyPolicy, Options, SerializerOptions};

pub use serde_saphyr::{Error, SerializeError};

fn read_options() -> Options {
    serde_saphyr::options! {
        // One-line messages: no source snippet.
        with_snippet: false,
        duplicate_keys: DuplicateKeyPolicy::Error,
        // YAML 1.2 booleans only.
        strict_booleans: true,
    }
}

fn write_options() -> SerializerOptions {
    serde_saphyr::ser_options! {
        // The library folds a string past `folded_wrap_chars` (default 80) into a `>`
        // block. `serde_yaml_ng` never did; keep a long line a line.
        folded_wrap_chars: 100_000_000,
    }
}

/// Deserialize `text` as a `T`.
///
/// An empty document (empty text, or only comments) reads as `T`'s defaults when `T`
/// is a struct whose fields all have them, as before.
pub fn from_str<T: DeserializeOwned>(text: &str) -> Result<T, Error> {
    serde_saphyr::from_str_with_options(text, read_options())
}

/// Serialize `value` as a YAML document (block style, ends with a newline, no leading
/// `---`).
pub fn to_string<T: Serialize>(value: &T) -> Result<String, SerializeError> {
    serde_saphyr::to_string_with_options(value, write_options())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serde_json::{Value, json};
    use std::collections::BTreeMap;

    /// Render `{key: value}` the way `write.rs` renders one frontmatter entry.
    fn entry(key: &str, value: &Value) -> String {
        let single: BTreeMap<&str, &Value> = BTreeMap::from([(key, value)]);
        to_string(&single).unwrap()
    }

    #[test]
    fn plain_scalars_and_collections_are_written_as_before() {
        let cases: Vec<(Value, &str)> = vec![
            (json!("hello world"), "key: hello world\n"),
            (json!(42), "key: 42\n"),
            (json!(-7), "key: -7\n"),
            (json!(3.5), "key: 3.5\n"),
            (json!(true), "key: true\n"),
            (Value::Null, "key: null\n"),
            (json!("café ☕ 日本語"), "key: café ☕ 日本語\n"),
            (json!([]), "key: []\n"),
            (json!({}), "key: {}\n"),
            (json!(["a", "b", 1, true]), "key:\n- a\n- b\n- 1\n- true\n"),
            (
                json!([{"name": "a", "role": "x"}, {"name": "b"}]),
                "key:\n- name: a\n  role: x\n- name: b\n",
            ),
            (json!([["a", "b"], ["c"]]), "key:\n- - a\n  - b\n- - c\n"),
            (
                json!({"k": "v", "m": {"x": 1}}),
                "key:\n  k: v\n  m:\n    x: 1\n",
            ),
        ];
        for (value, expected) in cases {
            assert_eq!(entry("key", &value), expected, "for {value}");
        }
    }

    #[test]
    fn multiline_strings_are_literal_blocks() {
        assert_eq!(
            entry("key", &json!("line1\nline2\n")),
            "key: |\n  line1\n  line2\n"
        );
        assert_eq!(
            entry("key", &json!("line1\nline2")),
            "key: |-\n  line1\n  line2\n"
        );
        assert_eq!(entry("key", &json!("a\r\nb")), "key: \"a\\r\\nb\"\n");
    }

    #[test]
    fn a_long_single_line_string_is_not_folded() {
        let long = "word ".repeat(40);
        let text = entry("key", &json!(long.trim_end()));
        assert_eq!(text, format!("key: {}\n", long.trim_end()));
    }

    #[test]
    fn strings_that_would_read_back_as_something_else_are_quoted() {
        for text in [
            "yes",
            "no",
            "null",
            "~",
            "true",
            "123",
            "1e3",
            "1.5",
            "0x1F",
            "0o17",
            "2026-01-02",
            "a: b",
            "a #b",
            "- x",
            " x",
            "x ",
            "",
            "!foo",
            "&a",
            "*a",
            "@x",
            "%x",
            "{a}",
            "[a]",
        ] {
            let written = entry("key", &json!(text));
            assert!(
                written.starts_with("key: \"") || written.starts_with("key: '"),
                "{text:?} written unquoted: {written:?}"
            );
            let back: BTreeMap<String, Value> = from_str(&written).unwrap();
            assert_eq!(back["key"], json!(text), "{text:?} did not round trip");
        }
    }

    #[test]
    fn a_key_that_reads_as_a_non_string_is_quoted() {
        for key in ["yes", "123", "null", "true", "n", ""] {
            let written = entry(key, &json!(1));
            assert!(
                written.starts_with('"') || written.starts_with('\''),
                "key {key:?} written unquoted: {written:?}"
            );
            let back: BTreeMap<String, Value> = from_str(&written).unwrap();
            assert_eq!(back[key], json!(1));
        }
    }

    #[derive(Debug, Deserialize, Default)]
    #[serde(deny_unknown_fields, default)]
    struct Probe {
        enabled: bool,
        name: String,
        n: u32,
    }

    #[test]
    fn an_empty_document_reads_as_the_defaults() {
        assert!(from_str::<Probe>("").unwrap().name.is_empty());
        assert!(!from_str::<Probe>("# only a comment\n").unwrap().enabled);
    }

    #[test]
    fn yaml_1_1_booleans_are_not_booleans() {
        let err = from_str::<Probe>("enabled: yes\n").unwrap_err().to_string();
        assert!(err.contains("line 1"), "{err}");
        let as_string: BTreeMap<String, Value> = from_str("k: yes\n").unwrap();
        assert_eq!(as_string["k"], json!("yes"));
        assert!(from_str::<Probe>("enabled: True\n").unwrap().enabled);
    }

    #[test]
    fn errors_are_one_line_and_carry_a_position() {
        for (text, needle) in [
            ("nme: a\n", "unknown field `nme`"),
            ("name: a\nname: b\n", "duplicate"),
            ("n: abc\n", "invalid u32"),
            ("n: -1\n", "invalid u32"),
        ] {
            let err = from_str::<Probe>(text).unwrap_err().to_string();
            assert!(err.contains(needle), "{text:?}: {err}");
            assert!(err.contains("line "), "{text:?}: {err}");
            assert!(!err.contains('\n'), "{text:?}: {err}");
        }
    }

    #[test]
    fn a_null_where_a_string_is_expected_is_an_error() {
        assert!(from_str::<Probe>("name:\n").is_err());
    }
}
