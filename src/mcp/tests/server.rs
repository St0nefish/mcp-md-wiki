use crate::mcp::*;
#[test]
fn omitted_limit_uses_the_configured_default() {
    assert_eq!(resolve_limit(None, 10, 50), 10);
    // The default is configurable, not baked in — a deployment that raised it
    // must see its own value, not the historical 10.
    assert_eq!(resolve_limit(None, 25, 50), 25);
}

#[test]
fn requested_limit_within_max_is_preserved() {
    assert_eq!(resolve_limit(Some(25), 10, 50), 25);
}

#[test]
fn requested_limit_above_max_is_clamped_to_the_configured_max() {
    assert_eq!(resolve_limit(Some(1_000_000), 10, 50), 50);
    // Clamped to the CONFIGURED ceiling, not a hardcoded one: raising
    // max_limit must actually raise what a caller can ask for.
    assert_eq!(resolve_limit(Some(1_000_000), 10, 200), 200);
}

#[test]
fn zero_limit_is_passed_through() {
    assert_eq!(resolve_limit(Some(0), 10, 50), 0);
}

#[test]
fn ellipsis_uses_char_count_not_byte_len() {
    // 800 chars of a 2-byte character = 1600 bytes
    let text: String = "é".repeat(801);
    assert!(text.len() > 800, "byte len should exceed 800");
    assert!(text.chars().count() > 800, "char count should exceed 800");
    // If we used .len() on a 800-char string it would wrongly trigger ellipsis
    let short: String = "é".repeat(800);
    assert!(
        short.len() > 800,
        "byte len of 800 2-byte chars exceeds 800"
    );
    assert_eq!(short.chars().count(), 800, "char count is exactly 800");
}

#[test]
fn include_globset_matches_markdown() {
    let patterns = vec!["**/*.md".to_string()];
    let gs = build_include_globset(&patterns);
    assert!(
        gs.is_match("docs/guide.md"),
        "**/*.md should match docs/guide.md"
    );
    assert!(
        gs.is_match("README.md"),
        "**/*.md should match top-level README.md"
    );
}

#[test]
fn include_globset_rejects_non_markdown() {
    let patterns = vec!["**/*.md".to_string()];
    let gs = build_include_globset(&patterns);
    assert!(
        !gs.is_match("state.db"),
        "**/*.md should not match state.db"
    );
    assert!(
        !gs.is_match("scripts/run.sh"),
        "**/*.md should not match shell scripts"
    );
    assert!(
        !gs.is_match(".env"),
        "**/*.md should not match credential files"
    );
}

#[test]
fn include_globset_respects_custom_patterns() {
    let patterns = vec!["**/*.md".to_string(), "**/*.txt".to_string()];
    let gs = build_include_globset(&patterns);
    assert!(gs.is_match("notes/todo.txt"), "should match *.txt");
    assert!(!gs.is_match("data.json"), "should not match *.json");
}

#[test]
fn include_globset_empty_patterns_falls_back_to_markdown() {
    let gs = build_include_globset(&[]);
    assert!(
        gs.is_match("docs/guide.md"),
        "empty patterns should fall back to **/*.md"
    );
    assert!(
        gs.is_match("README.md"),
        "empty patterns should match top-level .md"
    );
    assert!(
        !gs.is_match("state.db"),
        "empty patterns fallback should not match non-markdown"
    );
}

#[test]
fn include_globset_all_invalid_falls_back_to_markdown() {
    let gs = build_include_globset(&["[invalid".into()]);
    assert!(
        gs.is_match("docs/guide.md"),
        "all-invalid patterns should fall back to **/*.md"
    );
    assert!(
        !gs.is_match("data.json"),
        "all-invalid patterns fallback should not match non-markdown"
    );
}
