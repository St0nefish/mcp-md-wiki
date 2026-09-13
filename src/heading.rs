//! The heading model: the single source of truth for a markdown body's
//! section boundaries, heading text, heading levels and line numbers (#286).
//!
//! Both consumers read from here, so they cannot disagree about where a
//! section starts or what it is called: `chunk::chunk_markdown` (section
//! splitting, breadcrumbs, chunk attribution) and `retrieval::outline`
//! (`get_document`'s outline/section modes).
//!
//! ## What counts as a heading
//!
//! Detection is delegated to `pulldown-cmark` 0.12 (CommonMark plus the GFM
//! extensions in [`parser_options`]) rather than hand-rolled line matching. So
//! fenced code blocks (backtick or tilde, including a longer fence wrapping a
//! shorter one), indented code blocks, HTML blocks, `#hashtag` lines,
//! backslash-escaped `\#` and 7+ `#` runs are all handled the way pulldown-cmark
//! parses them. Other renderers can disagree in HTML-block edge cases: a line
//! holding only an HTML tag directly after a list item or table row
//! (`- item\n</span>\n# H`) is a lazy continuation or table row to
//! pulldown-cmark, so `# H` is a heading here, while cmark and markdown-rs start
//! an HTML block there that swallows `# H`. The deliberate choices on top of
//! CommonMark:
//!
//! - **Top-level block headings only.** A heading inside a blockquote, a list
//!   item or a footnote definition is part of that container's content, not a
//!   section boundary: `> # Quoted` does not split the document.
//! - **Setext headings count.** `Title` over `=====` is level 1 and over
//!   `-----` is level 2, exactly as CommonMark renders them. The heading's
//!   `line` is its first text line, not the underline.
//! - **Heading text is the rendered plain text.** Code spans keep their
//!   content, emphasis/strong/strikethrough markers are dropped, a link or an
//!   image contributes its text, raw inline HTML is dropped, a closing `#`
//!   sequence is not part of the text, runs of whitespace and control
//!   characters (including a multi-line setext heading's line break) collapse
//!   to one space, and invisible characters that never change rendering (soft
//!   hyphen, zero-width space, word joiner, BOM — see [`is_layout_only_char`])
//!   are removed. Invisible characters that do change rendering (zero-width
//!   joiner/non-joiner, direction marks, variation selectors — see
//!   [`is_rendering_format_char`]) stay in the stored text;
//!   [`normalize_heading_text`] ignores both kinds when matching.
//! - **No Unicode normalization (NFC/NFD).** [`normalize_heading_text`] case-folds
//!   but never re-composes or re-decomposes combining sequences, so a
//!   precomposed accented character (`"Café"`, U+00E9) does not match a
//!   canonically-equivalent decomposed spelling of the same text
//!   (`"Cafe\u{301}"`, e/U+0301) — no normalization crate is a dependency of
//!   this module. In practice this shows up as a miss (`NotFound`, or the
//!   section simply not matching), never a wrong hit, and both markdown
//!   sources and LLM-generated `heading_path`/`heading_prefix` input are
//!   almost always already NFC.
//! - **Heading text is capped at [`MAX_HEADING_TEXT_CHARS`] characters** (the
//!   rendering format characters don't count). A paragraph directly above a
//!   `---` line is a setext heading whose text is the whole paragraph, and
//!   heading text is copied into every chunk payload under it. Longer text is
//!   cut on a character boundary (no ellipsis, so the stored text stays a plain
//!   prefix of the heading, and a caller passing the full text still matches
//!   through [`normalize_heading_text`], which caps both sides identically).
//!   The cap applies everywhere heading text flows: chunk `heading_path`,
//!   breadcrumbs, payload keys and `get_document`'s outline.
//! - **A heading with no text is not a heading here.** A bare `#` (or a heading
//!   whose only content is inline HTML) has nothing to put in a path, so it
//!   is left as ordinary content of the enclosing section rather than adding
//!   an empty path segment callers could never address.
//!
//! ## Lines
//!
//! Every line number is 1-based and relative to the body this module was
//! handed. Lines are the `'\n'`-separated segments of that body (see
//! [`LineIndex`]), which is exactly how gray_matter's `content` relates to the
//! raw file, so [`body_line_offset`] plus a body line is always the raw file
//! line.
//!
//! CommonMark also ends a line at a lone `'\r'` (classic Mac line endings), so
//! pulldown-cmark can report several headings on what is one `'\n'`-line here
//! (`"# A\r# B\nbody"`). Only the first heading on a `'\n'`-line is a section
//! boundary; later ones on the same line are content of that section. Every
//! heading therefore has a strictly greater line than the one before it, which
//! is what keeps every section and subtree range non-inverted (#286).

use std::ops::Range;

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

/// The frontmatter delimiter gray_matter's `Matter::<YAML>::new()` uses.
const FRONTMATTER_DELIMITER: &str = "---";

/// Number of raw-file lines that precede the body `validate::
/// parse_frontmatter_raw` returns for `content`. Add it to a body-relative line
/// to get the line in the raw file, which is how `get_document` counts.
///
/// Derived from `content` alone by applying gray_matter 0.2's own rules,
/// rather than by locating the returned body inside `content`. gray_matter
/// rebuilds the body from `str::lines()`, so it drops the file's final line
/// terminator and turns CRLF into LF. A suffix comparison against the raw file
/// therefore fails for almost every real file. The rules
/// (`gray_matter::Matter::parse`):
///
/// 1. Input no longer than the delimiter yields an empty body. The offset is
///    moot, so 0.
/// 2. If the first line, trailing whitespace trimmed, is `---`, the lines up to
///    and including the next line that trims to `---` are frontmatter. If no
///    closing line exists, only the opening line is consumed and everything
///    after it is body.
/// 3. Empty lines at the start of what remains are stripped. Whitespace-only
///    lines are not.
///
/// This is not `write::split_frontmatter_bytes`. That function is deliberately
/// stricter (an exact `---` with no trailing spaces, and an unclosed block is
/// "no frontmatter"), and it must stay byte-exact for writes. What matters here
/// is matching the parser that produced the body being chunked. The tests pin
/// the invariant `content.lines().skip(offset)` joined by `"\n"` == body across
/// every variant.
pub fn body_line_offset(content: &str) -> usize {
    if content.len() <= FRONTMATTER_DELIMITER.len() {
        return 0;
    }
    let frontmatter_lines = match content.split_once('\n') {
        Some((first, rest)) if first.trim_end() == FRONTMATTER_DELIMITER => {
            match rest
                .lines()
                .position(|l| l.trim_end() == FRONTMATTER_DELIMITER)
            {
                // Opening line + the lines before the close + the close itself.
                Some(close) => close + 2,
                // Unclosed: gray_matter keeps everything after the opening line.
                None => 1,
            }
        }
        _ => 0,
    };
    let stripped_blank = content
        .lines()
        .skip(frontmatter_lines)
        .take_while(|l| l.is_empty())
        .count();
    frontmatter_lines + stripped_blank
}

/// Byte offset of every line start in a text, for O(log n) byte-to-line lookup
/// and O(1) line-to-byte slicing. Built once per document, so whole-corpus
/// chunking never rescans a prefix to count newlines.
///
/// A line is a `'\n'`-separated segment, so text ending in `'\n'` has a final
/// empty line. That matches how a gray_matter body maps onto the raw file: the
/// body has lost exactly one final terminator, so when it still ends in `'\n'`
/// the raw file really does have that trailing blank line. Empty text has no
/// lines.
#[derive(Debug, Clone)]
pub(crate) struct LineIndex {
    starts: Vec<usize>,
    len: usize,
}

impl LineIndex {
    pub(crate) fn new(text: &str) -> Self {
        let mut starts = Vec::new();
        if !text.is_empty() {
            starts.push(0);
            starts.extend(
                text.bytes()
                    .enumerate()
                    .filter(|&(_, b)| b == b'\n')
                    .map(|(i, _)| i + 1),
            );
        }
        Self {
            starts,
            len: text.len(),
        }
    }

    pub(crate) fn line_count(&self) -> usize {
        self.starts.len()
    }

    /// 1-based line containing byte `byte`. A newline byte belongs to the
    /// line it terminates.
    pub(crate) fn line_of(&self, byte: usize) -> usize {
        self.starts.partition_point(|&s| s <= byte).max(1)
    }

    /// Byte range of lines `first..=last` (1-based, inclusive), excluding the
    /// terminator of `last`.
    ///
    /// Total, never panics: `first` is clamped into `1..=line_count()` and `last`
    /// into `first..=line_count()`, so an out-of-range or inverted request yields
    /// the nearest valid range (at worst the single line `first`, or `0..0` for
    /// empty text) rather than an index panic or a `start > end` slice (#286).
    /// Callers still pass valid ranges; the clamp is what keeps a model bug from
    /// turning into a panic that aborts an indexing run.
    pub(crate) fn byte_range(&self, first: usize, last: usize) -> Range<usize> {
        let count = self.starts.len();
        if count == 0 {
            return 0..0;
        }
        let first = first.clamp(1, count);
        let last = last.clamp(first, count);
        let start = self.starts[first - 1];
        // `starts[last]` is one past a '\n' byte, so it is at least 1, and it is
        // strictly greater than `starts[first - 1]` because `last >= first`.
        let end = if last < count {
            self.starts[last] - 1
        } else {
            self.len
        };
        start..end
    }
}

/// One top-level heading. Headings are identified by their index in
/// [`HeadingTree::headings`] (document order), never by their text: two
/// headings with the same text are different headings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Heading {
    /// CommonMark level, 1-6.
    pub(crate) level: u8,
    /// Rendered plain text, whitespace-collapsed, never empty. See the module
    /// docs.
    pub(crate) text: String,
    /// 1-based line of the heading's first line.
    pub(crate) line: usize,
    /// Index of the nearest preceding heading with a lower level, i.e. the
    /// heading this one is nested under. `None` for a top-of-hierarchy heading.
    pub(crate) parent: Option<usize>,
    /// Last line of this heading's own section: the line before the next
    /// heading of any level, or the body's last line.
    pub(crate) section_end: usize,
    /// Last line of this heading's whole subtree: the line before the next
    /// heading at the same level or shallower, or the body's last line.
    pub(crate) subtree_end: usize,
}

/// Every top-level heading of a body, in document order, with the ancestry and
/// line ranges derived from them. See the module docs for detection rules.
#[derive(Debug, Clone)]
pub(crate) struct HeadingTree {
    headings: Vec<Heading>,
    lines: LineIndex,
}

/// GFM's block extensions, so a table, footnote definition or task list parses
/// the way the document's author saw it rendered. Deliberately not
/// `Options::all()` (which `text_splitter` uses for size-splitting, where text
/// fidelity does not matter): smart punctuation would rewrite quotes in
/// heading text, heading attributes would strip `{#id}` text GFM displays,
/// math would mangle `$` in headings, and metadata blocks could swallow a body
/// that happens to begin with `---`.
fn parser_options() -> Options {
    Options::ENABLE_TABLES
        | Options::ENABLE_FOOTNOTES
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TASKLISTS
}

impl HeadingTree {
    pub(crate) fn parse(body: &str) -> Self {
        let lines = LineIndex::new(body);

        // (level, first line, accumulated raw text) of the heading being read.
        let mut open: Option<(u8, usize, String)> = None;
        // Depth of block containers (blockquote, list, footnote definition).
        // Only headings at depth 0 are section boundaries.
        let mut container_depth = 0usize;
        let mut found: Vec<(u8, usize, String)> = Vec::new();

        for (event, range) in Parser::new_ext(body, parser_options()).into_offset_iter() {
            match event {
                Event::Start(Tag::BlockQuote(_) | Tag::List(_) | Tag::FootnoteDefinition(_)) => {
                    container_depth += 1;
                }
                Event::End(
                    TagEnd::BlockQuote(_) | TagEnd::List(_) | TagEnd::FootnoteDefinition,
                ) => {
                    container_depth = container_depth.saturating_sub(1);
                }
                Event::Start(Tag::Heading { level, .. }) if container_depth == 0 => {
                    open = Some((level as u8, lines.line_of(range.start), String::new()));
                }
                Event::End(TagEnd::Heading(_)) => {
                    if let Some((level, line, raw)) = open.take() {
                        let text = clean_heading_text(&raw);
                        // A second heading on the same '\n'-line (a lone '\r'
                        // separated them) is not a boundary — see the module
                        // docs. `found` is in document order, so comparing with
                        // the last kept heading is enough.
                        let new_line = found.last().is_none_or(|&(_, prev, _)| line > prev);
                        if !text.is_empty() && new_line {
                            found.push((level, line, text));
                        }
                    }
                }
                Event::Text(t) | Event::Code(t) => {
                    if let Some((_, _, raw)) = open.as_mut() {
                        raw.push_str(&t);
                    }
                }
                Event::SoftBreak | Event::HardBreak => {
                    if let Some((_, _, raw)) = open.as_mut() {
                        raw.push(' ');
                    }
                }
                _ => {}
            }
        }

        let line_count = lines.line_count();
        let mut headings: Vec<Heading> = Vec::with_capacity(found.len());
        // Open ancestry, as indices into `headings`; levels strictly increase
        // from bottom to top.
        let mut stack: Vec<usize> = Vec::new();
        for (level, line, text) in found {
            // `line` is strictly greater than every earlier heading's line (see
            // the dedup above), so `line - 1 >= earlier.line`. The `max` keeps
            // the "end never precedes start" invariant true by construction
            // even so.
            if let Some(prev) = headings.last_mut() {
                prev.section_end = line.saturating_sub(1).max(prev.line);
            }
            // A heading closes every open heading at its level or deeper: that
            // is both where their subtrees end and what leaves its parent on
            // top of the stack.
            while let Some(&top) = stack.last() {
                if headings[top].level < level {
                    break;
                }
                headings[top].subtree_end = line.saturating_sub(1).max(headings[top].line);
                stack.pop();
            }
            let idx = headings.len();
            headings.push(Heading {
                level,
                text,
                line,
                parent: stack.last().copied(),
                section_end: line_count.max(line),
                subtree_end: line_count.max(line),
            });
            stack.push(idx);
        }

        Self { headings, lines }
    }

    pub(crate) fn headings(&self) -> &[Heading] {
        &self.headings
    }

    pub(crate) fn lines(&self) -> &LineIndex {
        &self.lines
    }

    pub(crate) fn line_count(&self) -> usize {
        self.lines.line_count()
    }

    /// Last line before the first heading (0 when the body opens with a
    /// heading), or the body's last line when there are no headings.
    pub(crate) fn preamble_end(&self) -> usize {
        self.headings
            .first()
            .map_or(self.line_count(), |h| h.line.saturating_sub(1))
    }

    /// Heading texts from the top of the hierarchy down to `node` inclusive.
    /// Empty for `None`.
    pub(crate) fn path(&self, node: Option<usize>) -> Vec<String> {
        let mut path = Vec::new();
        let mut cur = node;
        while let Some(i) = cur {
            path.push(self.headings[i].text.clone());
            cur = self.headings[i].parent;
        }
        path.reverse();
        path
    }

    /// The deepest heading that is `a` or an ancestor of `a` AND `b` or an
    /// ancestor of `b`, by identity. `None` (the empty path) when they share
    /// nothing. Relies on a parent always preceding its child in document
    /// order, so stepping the later of the two up converges on the answer.
    pub(crate) fn common_ancestor(&self, a: Option<usize>, b: Option<usize>) -> Option<usize> {
        let (mut a, mut b) = (a?, b?);
        while a != b {
            if a > b {
                a = self.headings[a].parent?;
            } else {
                b = self.headings[b].parent?;
            }
        }
        Some(a)
    }
}

/// Separator between heading segments in machine keys (#286): the
/// `heading_prefixes` payload values ([`heading_prefix_key`]) and the path
/// portion of `ingest::section_key`. U+001F (INFORMATION SEPARATOR ONE) is a
/// control character, and neither stored heading text ([`HeadingTree::parse`])
/// nor [`normalize_heading_text`] output can contain a control character, so a
/// single heading `"Input > Output"` can never produce the same key as the path
/// `["Input", "Output"]`. Human-facing renderings (breadcrumbs, messages) keep
/// joining with `" > "`; this is only for keys that must be unambiguous.
pub(crate) const HEADING_KEY_SEPARATOR: &str = "\u{1f}";

/// Maximum stored heading text length (#286), counted in characters that
/// take part in matching: every character except the
/// [`is_rendering_format_char`] joiners and direction marks, which stored
/// text keeps but matching ignores. See the module docs: longer text is
/// truncated by [`clean_heading_text`].
pub(crate) const MAX_HEADING_TEXT_CHARS: usize = 200;

/// Hard bound on stored heading text in characters of any kind, rendering
/// format characters included, so a heading padded with thousands of joiners
/// is still bounded. Twice [`MAX_HEADING_TEXT_CHARS`]: real text (an emoji ZWJ
/// sequence has fewer joiners than visible characters) reaches the matching
/// cap first.
const MAX_HEADING_TEXT_RAW_CHARS: usize = 2 * MAX_HEADING_TEXT_CHARS;

/// Invisible characters that affect only line breaking, or nothing at all, so
/// dropping them never changes how a heading renders: soft hyphen (a
/// hyphenation hint, common in PDF conversions), zero-width space, word
/// joiner, the invisible math operators (U+2061–U+2064) and the BOM /
/// zero-width no-break space. Removed from stored heading text and ignored by
/// matching.
fn is_layout_only_char(c: char) -> bool {
    matches!(
        c,
        '\u{00AD}' | '\u{200B}' | '\u{2060}'..='\u{2064}' | '\u{FEFF}'
    )
}

/// Invisible characters that DO change how text renders: zero-width
/// non-joiner and joiner (emoji ZWJ sequences such as family emoji, Persian
/// and Indic letter joining), left-to-right, right-to-left and Arabic letter
/// marks, bidi embedding/override/isolate controls, the Mongolian vowel
/// separator, the deprecated shaping controls (U+206A–U+206F) and variation
/// selectors (U+FE00–U+FE0F, as in `❤️` vs `❤`). Stored heading text keeps
/// them, so a returned `heading_path` or outline label renders like the
/// heading; [`normalize_heading_text`] drops them, so a caller who types the
/// heading without them still matches. A hand-kept list (a subset of Unicode
/// categories Cf and Mn) rather than a Unicode property lookup, so no
/// dependency is needed.
fn is_rendering_format_char(c: char) -> bool {
    matches!(
        c,
        '\u{061C}'
            | '\u{180E}'
            | '\u{200C}'..='\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2066}'..='\u{206F}'
            | '\u{FE00}'..='\u{FE0F}'
    )
}

/// Every character heading matching ignores: [`is_layout_only_char`] plus
/// [`is_rendering_format_char`].
fn is_ignored_in_matching(c: char) -> bool {
    is_layout_only_char(c) || is_rendering_format_char(c)
}

/// The heading text `HeadingTree::parse` stores:
///
/// 1. Split into words on every whitespace AND control character, dropping
///    every word made only of [`is_ignored_in_matching`] characters (a lone
///    joiner between two spaces), and rejoined with single spaces. Dropping
///    control characters is what guarantees [`HEADING_KEY_SEPARATOR`] never
///    occurs in heading text; dropping invisible-only words means that removing
///    the rendering format characters later never leaves an empty word or a
///    doubled space.
/// 2. [`is_layout_only_char`] characters removed (`Fire\u{ad}ball` is
///    `Fireball`); [`is_rendering_format_char`] characters stay.
/// 3. Cut once it holds [`MAX_HEADING_TEXT_CHARS`] matching characters (spaces
///    count; rendering format characters don't) or
///    [`MAX_HEADING_TEXT_RAW_CHARS`] characters in all, dropping a trailing
///    space or a trailing word the cut left without a matching character.
///
/// Idempotent: cleaning already-cleaned text returns it unchanged.
fn clean_heading_text(s: &str) -> String {
    let mut out = String::new();
    let (mut matching, mut raw) = (0usize, 0usize);
    'words: for word in s
        .split(|c: char| c.is_whitespace() || c.is_control())
        .filter(|word| word.chars().any(|c| !is_ignored_in_matching(c)))
    {
        let word_start = out.len();
        if !out.is_empty() {
            if matching == MAX_HEADING_TEXT_CHARS || raw == MAX_HEADING_TEXT_RAW_CHARS {
                break;
            }
            out.push(' ');
            matching += 1;
            raw += 1;
        }
        let mut word_matching = false;
        for c in word.chars().filter(|&c| !is_layout_only_char(c)) {
            let counts = !is_rendering_format_char(c);
            if (counts && matching == MAX_HEADING_TEXT_CHARS) || raw == MAX_HEADING_TEXT_RAW_CHARS {
                if !word_matching {
                    out.truncate(word_start);
                }
                break 'words;
            }
            out.push(c);
            raw += 1;
            matching += usize::from(counts);
            word_matching |= counts;
        }
    }
    out.truncate(out.trim_end().len());
    out
}

/// Normalize a heading name for case-, whitespace- and invisible-character-
/// insensitive comparison (#286). Shared by `retrieval::resolve_section`'s
/// `heading_path` suffix/exact matching and `search`'s `heading_prefix` filter
/// (via [`heading_prefix_key`]), so the two features can't drift on what "the
/// same heading" means. Stored heading text and a caller-supplied segment go
/// through the same steps:
///
/// 1. [`clean_heading_text`], so already-parsed heading text is unchanged and a
///    long heading's full text is cut where the stored copy was.
/// 2. Every [`is_ignored_in_matching`] character removed (joiners, direction
///    marks and variation selectors included).
/// 3. Full Unicode case folding ("Straße" and "STRASSE" compare equal).
/// 4. Cut again at [`MAX_HEADING_TEXT_CHARS`] characters, trailing space
///    trimmed.
///
/// Step 4 is what makes the cap agree across spellings. Folding can lengthen
/// text (`ß` → `ss`), and a caller may type a folded or joiner-free spelling,
/// so step 1 can cut two equivalent strings at different places. Steps 2 and 3
/// map one character at a time, though, so both results are prefixes of the
/// same folded full text, and each is either all of it or at least
/// [`MAX_HEADING_TEXT_CHARS`] characters long (one less when the cut dropped a
/// trailing space, which step 4 drops too); step 4 then cuts both to the same
/// prefix. The one exception is a heading with more format characters than
/// visible ones, cut by [`MAX_HEADING_TEXT_RAW_CHARS`]: its full text still
/// matches the stored copy, but a spelling without those format characters may
/// not — and neither, in the same near-[`MAX_HEADING_TEXT_RAW_CHARS`] case,
/// may a case-folded spelling (folding can itself lengthen a string, e.g.
/// `ß` → `ss`, enough to trip the raw cap on its own). The output never
/// contains a control character, so never [`HEADING_KEY_SEPARATOR`].
pub(crate) fn normalize_heading_text(s: &str) -> String {
    let visible: String = clean_heading_text(s)
        .chars()
        .filter(|&c| !is_ignored_in_matching(c))
        .collect();
    let mut folded = unicase::UniCase::new(visible).to_folded_case();
    if let Some((cut, _)) = folded.char_indices().nth(MAX_HEADING_TEXT_CHARS) {
        folded.truncate(cut);
        folded.truncate(folded.trim_end().len());
    }
    folded
}

/// The `heading_prefixes` payload keyword for a run of heading segments: each
/// segment through [`normalize_heading_text`], joined with
/// [`HEADING_KEY_SEPARATOR`]. Both sides of the `heading_prefix` filter use it —
/// `ingest::derive_heading_prefixes` when writing the payload and
/// `mcp::heading_prefix_condition` when lowering a caller's filter — so
/// `["conditions"]` matches a "Conditions" heading, and a caller's single
/// segment `"a > b"` matches only a heading literally named that.
pub(crate) fn heading_prefix_key<S: AsRef<str>>(segments: &[S]) -> String {
    segments
        .iter()
        .map(|s| normalize_heading_text(s.as_ref()))
        .collect::<Vec<_>>()
        .join(HEADING_KEY_SEPARATOR)
}

/// The section a chunk is attributed to. Carried on the chunk itself so
/// consumers never have to rediscover it from heading text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SectionId {
    /// The whole body: what a chunk spanning sections with no common heading
    /// belongs to (for example the preamble merged with the first heading's
    /// section, or two sibling top-level headings).
    Root,
    /// The content before the first heading (the whole body when there are no
    /// headings), on its own.
    Preamble,
    /// A heading's whole subtree.
    Heading(usize),
}

impl SectionId {
    /// The narrowest section containing both `self` and `other`.
    pub(crate) fn merge(self, other: SectionId, tree: &HeadingTree) -> SectionId {
        match (self, other) {
            (a, b) if a == b => a,
            (SectionId::Heading(a), SectionId::Heading(b)) => tree
                .common_ancestor(Some(a), Some(b))
                .map_or(SectionId::Root, SectionId::Heading),
            _ => SectionId::Root,
        }
    }

    /// Heading path of the section. Empty for `Root` and `Preamble`.
    pub(crate) fn path(self, tree: &HeadingTree) -> Vec<String> {
        match self {
            SectionId::Heading(i) => tree.path(Some(i)),
            SectionId::Root | SectionId::Preamble => Vec::new(),
        }
    }

    /// The attributed heading's level, or 0.
    pub(crate) fn level(self, tree: &HeadingTree) -> u8 {
        match self {
            SectionId::Heading(i) => tree.headings[i].level,
            SectionId::Root | SectionId::Preamble => 0,
        }
    }

    /// Body-relative, 1-based inclusive line range. The same ranges
    /// `retrieval::outline` reports for a heading entry and for the preamble
    /// entry.
    pub(crate) fn line_range(self, tree: &HeadingTree) -> (usize, usize) {
        match self {
            SectionId::Heading(i) => (tree.headings[i].line, tree.headings[i].subtree_end),
            SectionId::Preamble => (1, tree.preamble_end()),
            SectionId::Root => (1, tree.line_count()),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::config::ChunkingConfig;

    /// `(level, text, line, parent)` for every heading.
    fn summary(body: &str) -> Vec<(u8, String, usize, Option<usize>)> {
        HeadingTree::parse(body)
            .headings()
            .iter()
            .map(|h| (h.level, h.text.clone(), h.line, h.parent))
            .collect()
    }

    fn h(
        level: u8,
        text: &str,
        line: usize,
        parent: Option<usize>,
    ) -> (u8, String, usize, Option<usize>) {
        (level, text.to_string(), line, parent)
    }

    // ── heading detection edge cases ────────────────────────────────────────

    #[test]
    fn atx_headings_nest_by_level_with_skipped_levels() {
        let body = "# Root\n\n#### Deep\n\ntext\n\n## Shallow\n\n### Leaf\n";
        assert_eq!(
            summary(body),
            vec![
                h(1, "Root", 1, None),
                h(4, "Deep", 3, Some(0)),
                h(2, "Shallow", 7, Some(0)),
                h(3, "Leaf", 9, Some(2)),
            ]
        );
        let tree = HeadingTree::parse(body);
        let ends: Vec<(usize, usize)> = tree
            .headings()
            .iter()
            .map(|h| (h.section_end, h.subtree_end))
            .collect();
        // The body ends in '\n', so it has a 10th (empty) line.
        assert_eq!(ends, vec![(2, 10), (6, 6), (8, 10), (10, 10)]);
        assert_eq!(tree.path(Some(3)), vec!["Root", "Shallow", "Leaf"]);
    }

    #[test]
    fn backtick_and_tilde_fences_hide_headings() {
        let body =
            "# Real\n```\n# not one\n~~~\n# still not\n```\n~~~md\n# nor this\n~~~\n## After\n";
        assert_eq!(
            summary(body),
            vec![h(1, "Real", 1, None), h(2, "After", 10, Some(0))]
        );
    }

    #[test]
    fn longer_fence_wrapping_a_shorter_one_stays_open() {
        // Regression (#286): the inner ``` must not close the outer ````.
        let body = "# A\n````\n```\n# inside4\n```\n````\n## D\nbody\n";
        assert_eq!(
            summary(body),
            vec![h(1, "A", 1, None), h(2, "D", 7, Some(0))]
        );
    }

    #[test]
    fn indented_code_block_is_not_a_heading() {
        let body = "Intro.\n\n    # code, not a heading\n\n# Real\n";
        assert_eq!(summary(body), vec![h(1, "Real", 5, None)]);
    }

    #[test]
    fn up_to_three_spaces_of_indentation_is_still_a_heading() {
        let body = "   # Indented\nbody\n## Next\n";
        assert_eq!(
            summary(body),
            vec![h(1, "Indented", 1, None), h(2, "Next", 3, Some(0))]
        );
    }

    #[test]
    fn hashtag_seven_hashes_and_escaped_hash_are_not_headings() {
        // Regression (#286): `#hashtag` must not become a level-1 heading that
        // re-parents every later section.
        let body = "# A\n## B\n#hashtag line\nmore\n####### seven\n\\# escaped\n## C\n";
        assert_eq!(
            summary(body),
            vec![
                h(1, "A", 1, None),
                h(2, "B", 2, Some(0)),
                h(2, "C", 7, Some(0))
            ]
        );
    }

    #[test]
    fn closing_hash_sequence_is_not_part_of_the_text() {
        let body = "## Title ##\n# Foo #bar\n### Trailing ###   \n";
        assert_eq!(
            summary(body),
            vec![
                h(2, "Title", 1, None),
                h(1, "Foo #bar", 2, None),
                h(3, "Trailing", 3, Some(1)),
            ]
        );
    }

    #[test]
    fn setext_headings_are_headings_at_their_text_line() {
        let body = "Title\n=====\n\nIntro.\n\nSub part\nsecond line\n---\n\nbody\n";
        assert_eq!(
            summary(body),
            vec![
                h(1, "Title", 1, None),
                h(2, "Sub part second line", 6, Some(0))
            ]
        );
    }

    #[test]
    fn headings_inside_blockquotes_lists_and_html_blocks_are_not_boundaries() {
        let body = "# Top\n\n> # Quoted\n\n- # Listed\n\n1. ## Numbered\n\n<div>\n# in html\n</div>\n\n## Next\n";
        assert_eq!(
            summary(body),
            vec![h(1, "Top", 1, None), h(2, "Next", 13, Some(0))]
        );
    }

    #[test]
    fn leading_whitespace_only_line_does_not_hide_the_first_heading() {
        // Regression (#286).
        let body = "  \n# A\nbody\n## B\nx\n";
        assert_eq!(
            summary(body),
            vec![h(1, "A", 2, None), h(2, "B", 4, Some(0))]
        );
    }

    #[test]
    fn heading_text_is_rendered_plain_text_with_collapsed_whitespace() {
        let body = "## The `code()`  *emph* **strong** ~~gone~~ [link](x.md) ![alt](i.png) <b>html</b> &amp; \\*x\\*\n#   Spaced    out   \n";
        assert_eq!(
            summary(body),
            vec![
                h(
                    2,
                    "The code() emph strong gone link alt html & *x*",
                    1,
                    None
                ),
                h(1, "Spaced out", 2, None),
            ]
        );
    }

    #[test]
    fn empty_headings_are_ordinary_content() {
        let body = "# A\n#\n## B\n##   ##\ntext\n";
        assert_eq!(
            summary(body),
            vec![h(1, "A", 1, None), h(2, "B", 3, Some(0))]
        );
        assert_eq!(HeadingTree::parse(body).headings()[1].section_end, 6);
    }

    #[test]
    fn same_text_headings_have_distinct_identities() {
        let body = "# R\n## X\n# R\n### C\n## C\n";
        let tree = HeadingTree::parse(body);
        assert_eq!(
            summary(body),
            vec![
                h(1, "R", 1, None),
                h(2, "X", 2, Some(0)),
                h(1, "R", 3, None),
                h(3, "C", 4, Some(2)),
                h(2, "C", 5, Some(2)),
            ]
        );
        assert_eq!(tree.common_ancestor(Some(1), Some(2)), None);
        assert_eq!(tree.common_ancestor(Some(3), Some(4)), Some(2));
        assert_eq!(tree.common_ancestor(Some(3), Some(3)), Some(3));
        assert_eq!(tree.common_ancestor(Some(2), Some(4)), Some(2));
        assert_eq!(tree.common_ancestor(None, Some(4)), None);
    }

    #[test]
    fn normalize_heading_text_collapses_whitespace_and_folds_case() {
        assert_eq!(normalize_heading_text("Fire  Ball"), "fire ball");
        assert_eq!(normalize_heading_text("  Fireball  "), "fireball");
        assert_eq!(normalize_heading_text("FIREBALL"), "fireball");
        // Full Unicode case folding, not just `to_lowercase`.
        assert_eq!(
            normalize_heading_text("Straße"),
            normalize_heading_text("STRASSE")
        );
        // Already-collapsed heading text (as `HeadingTree::parse` stores it)
        // is a no-op other than lowercasing.
        let tree = HeadingTree::parse("## The `code()`  *emph*\n");
        assert_eq!(
            normalize_heading_text(&tree.headings()[0].text),
            "the code() emph"
        );
    }

    #[test]
    fn invisible_format_characters_are_removed_from_heading_text() {
        // #286: soft hyphens (PDF conversions), zero-width spaces, word joiners
        // and a BOM never change how a heading renders, so they are dropped
        // from its stored text and must not change what matches it.
        let tree =
            HeadingTree::parse("# Fire\u{ad}ball\n## Zero\u{200b}Width \u{feff}Joined\u{2060}\n");
        assert_eq!(tree.headings()[0].text, "Fireball");
        assert_eq!(tree.headings()[1].text, "ZeroWidth Joined");
        assert_eq!(
            heading_prefix_key(&[tree.headings()[0].text.as_str()]),
            heading_prefix_key(&["Fireball"])
        );
        assert_eq!(normalize_heading_text("Fire\u{ad}ball"), "fireball");
        for c in [
            '\u{ad}', '\u{200b}', '\u{200c}', '\u{200d}', '\u{2060}', '\u{feff}',
        ] {
            assert_eq!(
                normalize_heading_text(&format!("Fire{c}ball")),
                "fireball",
                "U+{:04X}",
                c as u32
            );
        }
        // An invisible character between words does not become a separator of
        // its own, and a heading of only invisible characters has no text.
        assert_eq!(normalize_heading_text("a \u{200b} b"), "a b");
        assert!(
            HeadingTree::parse("# \u{ad}\u{200b}\n")
                .headings()
                .is_empty()
        );

        // A BOM before frontmatter gray_matter did not recognize: the setext
        // heading it forms carries no BOM.
        let bom = HeadingTree::parse("\u{feff}---\ntitle: T\n---\n");
        assert_eq!(bom.headings()[0].text, "--- title: T");

        // The no-control-character guarantee still holds for anything a caller
        // passes.
        let messy = "\u{feff}a\u{1f}\u{ad}b\u{0}\u{200d}\tc\u{7f}";
        let normalized = normalize_heading_text(messy);
        assert_eq!(normalized, "a b c");
        assert!(!normalized.chars().any(|c| c.is_control()));
    }

    #[test]
    fn rendering_format_characters_stay_in_stored_text_but_not_in_matching() {
        // #286: joiners, direction marks and variation selectors change how a
        // heading renders, so its stored text keeps them; matching ignores them.
        let family = "Family 👨\u{200d}👩\u{200d}👧";
        let persian = "می\u{200c}خواهم";
        let body = format!(
            "# {family}\n## {persian}\n## Love ❤\u{fe0f}\n## a\u{200e}b\n# \u{200d}\u{fe0f}\n"
        );
        let tree = HeadingTree::parse(&body);
        let texts: Vec<&str> = tree.headings().iter().map(|h| h.text.as_str()).collect();
        assert_eq!(
            texts,
            vec![family, persian, "Love ❤\u{fe0f}", "a\u{200e}b"],
            "a heading of only format characters has no text"
        );
        assert_eq!(normalize_heading_text(family), "family 👨👩👧");
        assert_eq!(
            normalize_heading_text(persian),
            normalize_heading_text("میخواهم")
        );
        assert_eq!(
            heading_prefix_key(&["Love ❤\u{fe0f}"]),
            heading_prefix_key(&["love ❤"])
        );
        // A joiner-only word between spaces vanishes rather than leaving a
        // doubled space once matching removes it.
        assert_eq!(clean_heading_text("a \u{200d} b"), "a b");
        assert_eq!(normalize_heading_text("a \u{200d}\u{200e} b"), "a b");
        // The no-control-character guarantee holds with them kept.
        for text in texts {
            assert!(!text.chars().any(|c| c.is_control()), "{text:?}");
            assert_eq!(clean_heading_text(text), text, "idempotent");
        }
    }

    #[test]
    fn capped_headings_match_across_folding_and_joiner_spellings() {
        // Folding lengthens `ß` to `ss`: a 150-char heading (under the cap)
        // must match a caller typing the 300-char folded spelling.
        let eszett = "ß".repeat(150);
        assert_eq!(clean_heading_text(&eszett), eszett);
        assert_eq!(
            normalize_heading_text(&eszett),
            normalize_heading_text(&"ss".repeat(150))
        );
        assert_eq!(
            normalize_heading_text(&eszett),
            normalize_heading_text(&"SS".repeat(150))
        );
        // Over the cap too, and the stored (capped) copy matches both.
        let long = format!("{}X", "ß".repeat(MAX_HEADING_TEXT_CHARS));
        let stored = clean_heading_text(&long);
        assert_eq!(stored, "ß".repeat(MAX_HEADING_TEXT_CHARS));
        let key = normalize_heading_text(&stored);
        assert_eq!(key.chars().count(), MAX_HEADING_TEXT_CHARS);
        assert_eq!(normalize_heading_text(&long), key);
        assert_eq!(
            normalize_heading_text(&format!("{}x", "SS".repeat(MAX_HEADING_TEXT_CHARS))),
            key
        );

        // Joiners don't count toward the cap, so a long run of ZWJ emoji is
        // cut after the same visible text whether or not the caller's copy
        // carries the joiners.
        let emoji = "👨\u{200d}👩 ".repeat(70);
        let stored = clean_heading_text(&emoji);
        assert!(stored.chars().count() > MAX_HEADING_TEXT_CHARS, "{stored}");
        assert_eq!(
            stored.chars().filter(|&c| c != '\u{200d}').count(),
            MAX_HEADING_TEXT_CHARS,
        );
        let plain = emoji.replace('\u{200d}', "");
        assert_eq!(
            normalize_heading_text(&stored),
            normalize_heading_text(&emoji)
        );
        assert_eq!(
            normalize_heading_text(&plain),
            normalize_heading_text(&emoji)
        );
        assert_eq!(clean_heading_text(&stored), stored, "idempotent");

        // A cut landing on a word that would keep only a joiner drops that word.
        let edge = format!("{} \u{200d}Y", "x".repeat(MAX_HEADING_TEXT_CHARS - 1));
        assert_eq!(
            clean_heading_text(&edge),
            "x".repeat(MAX_HEADING_TEXT_CHARS - 1)
        );

        // Padding with thousands of joiners stays bounded.
        let padded = format!("a{}", "\u{200d}".repeat(5000));
        assert!(clean_heading_text(&padded).chars().count() <= MAX_HEADING_TEXT_RAW_CHARS);
    }

    #[test]
    fn heading_text_is_capped_on_a_char_boundary() {
        // #286: a paragraph directly above `---` is a setext heading whose text
        // is the whole paragraph. A 2 KB one is stored capped.
        // 10 chars per word, so the cut lands right after a space.
        let word = "Paragrapé ";
        let paragraph = word.repeat(2000 / word.len() + 1);
        let body = format!("# Chapter\n\n{paragraph}\n---\n\n### Sub 1\n\nbody\n");
        let tree = HeadingTree::parse(&body);
        assert_eq!(tree.headings().len(), 3);
        let long = &tree.headings()[1];
        assert_eq!(long.level, 2);
        assert_eq!(long.text.chars().count(), MAX_HEADING_TEXT_CHARS - 1);
        assert!(
            !long.text.ends_with(' '),
            "the cut's trailing space is trimmed"
        );
        let full = clean_heading_text_uncapped(&paragraph);
        assert!(full.chars().count() > 1000);
        assert!(full.starts_with(&long.text));
        // Cleaning is idempotent, and a caller passing the full text matches
        // the stored heading.
        assert_eq!(clean_heading_text(&long.text), long.text);
        assert_eq!(
            normalize_heading_text(&full),
            normalize_heading_text(&long.text)
        );
        assert_eq!(
            heading_prefix_key(&["Chapter", full.as_str()]),
            heading_prefix_key(&["Chapter", long.text.as_str()])
        );
        // The cap reaches the outline and the chunk heading paths too.
        for entry in crate::retrieval::outline(&body) {
            for segment in &entry.heading_path {
                assert!(segment.chars().count() <= MAX_HEADING_TEXT_CHARS);
            }
        }
        let cfg = ChunkingConfig {
            max_chunk_size: 1500,
            target_chunk_size: Some(1000),
            prepend_description: false,
            prepend_heading_path: true,
            heading_metadata: true,
        };
        for chunk in crate::chunk::chunk_markdown(&body, None, &cfg) {
            for segment in &chunk.heading_path {
                assert!(segment.chars().count() <= MAX_HEADING_TEXT_CHARS);
            }
        }

        // Exactly at the cap: untouched. Multi-byte text cuts on a char boundary.
        let exact = "x".repeat(MAX_HEADING_TEXT_CHARS);
        assert_eq!(clean_heading_text(&exact), exact);
        let wide = "ß".repeat(MAX_HEADING_TEXT_CHARS + 50);
        assert_eq!(
            clean_heading_text(&wide),
            "ß".repeat(MAX_HEADING_TEXT_CHARS)
        );
    }

    /// The whitespace collapse of [`clean_heading_text`] without the cap, to
    /// compare a capped heading against its full text.
    fn clean_heading_text_uncapped(s: &str) -> String {
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn heading_keys_use_a_separator_heading_text_cannot_contain() {
        // A single heading spelled like a two-level path must not share a key.
        assert_ne!(
            heading_prefix_key(&["Input > Output"]),
            heading_prefix_key(&["Input", "Output"])
        );
        assert_eq!(
            heading_prefix_key(&["Input", "Output"]),
            format!("input{HEADING_KEY_SEPARATOR}output")
        );
        // Control characters (the separator included) collapse like whitespace,
        // in caller-supplied segments and in parsed heading text alike.
        assert_eq!(normalize_heading_text("a\u{1f}b\u{0}\tc"), "a b c");
        assert_eq!(
            heading_prefix_key(&["input\u{1f}output"]),
            heading_prefix_key(&["input output"])
        );
        let tree = HeadingTree::parse("# Input\u{1f}Output\n");
        assert_eq!(tree.headings()[0].text, "Input Output");
        // A heading whose only text is control characters has no text.
        assert!(HeadingTree::parse("# \u{1}\u{1f}\n").headings().is_empty());
    }

    // ── lone '\r' line endings (#286) ───────────────────────────────────────

    #[test]
    fn a_second_heading_on_the_same_newline_line_is_content() {
        // CommonMark ends a line at a lone '\r', so pulldown-cmark sees `# B`
        // as a heading on line 1 too. Treating it as a boundary would end A's
        // section before A's own line (an inverted range) and panic in
        // chunking.
        let body = "# A\r# B\nbody";
        assert_eq!(summary(body), vec![h(1, "A", 1, None)]);
        let tree = HeadingTree::parse(body);
        assert_eq!(
            (
                tree.headings()[0].section_end,
                tree.headings()[0].subtree_end
            ),
            (2, 2)
        );

        // A classic-Mac file is a single '\n'-line: one heading, one section.
        let mac = "# Title\rintro\r\r## Part\rtext\r";
        assert_eq!(summary(mac), vec![h(1, "Title", 1, None)]);

        for body in [body, mac] {
            let chunks = crate::chunk::chunk_markdown(
                body,
                None,
                &ChunkingConfig {
                    max_chunk_size: 60,
                    target_chunk_size: Some(1),
                    prepend_description: false,
                    prepend_heading_path: true,
                    heading_metadata: true,
                },
            );
            assert!(!chunks.is_empty());
            assert_chunk_ranges_valid(body, &chunks);
            assert_outline_ranges_valid(body);
        }
    }

    #[test]
    fn line_index_byte_range_is_total() {
        let text = "ab\ncd\nef";
        let idx = LineIndex::new(text);
        // Inverted, zero and past-the-end requests clamp instead of panicking.
        assert_eq!(&text[idx.byte_range(2, 1)], "cd");
        assert_eq!(&text[idx.byte_range(0, 0)], "ab");
        assert_eq!(&text[idx.byte_range(3, 99)], "ef");
        assert_eq!(&text[idx.byte_range(99, 1)], "ef");
        assert_eq!(LineIndex::new("").byte_range(1, 1), 0..0);
    }

    fn assert_chunk_ranges_valid(body: &str, chunks: &[crate::chunk::Chunk]) {
        let line_count = LineIndex::new(body).line_count();
        for c in chunks {
            assert!(
                1 <= c.section_line_start
                    && c.section_line_start <= c.line_start
                    && c.line_start <= c.line_end
                    && c.line_end <= c.section_line_end
                    && c.section_line_end <= line_count,
                "chunk {} lines {}-{} section {}-{} of {line_count}; body={body:?}",
                c.index,
                c.line_start,
                c.line_end,
                c.section_line_start,
                c.section_line_end
            );
        }
    }

    /// The `chunking.heading_metadata` invariant (#286): no chunk crosses a
    /// heading boundary. No chunk's line range contains a heading line other
    /// than its own first line; every chunk lies inside one heading's own
    /// section (and is attributed to exactly that heading, with its subtree
    /// range) or inside the preamble; attribution is never `Root`.
    pub(crate) fn assert_chunks_stay_within_one_section(
        body: &str,
        chunks: &[crate::chunk::Chunk],
    ) {
        let tree = HeadingTree::parse(body);
        for c in chunks {
            let ctx = format!(
                "chunk {} lines {}-{} section {}-{} path {:?}; body={body:?}",
                c.index,
                c.line_start,
                c.line_end,
                c.section_line_start,
                c.section_line_end,
                c.heading_path
            );
            for h in tree.headings() {
                assert!(
                    !(c.line_start < h.line && h.line <= c.line_end),
                    "heading {:?} at line {} is inside a chunk; {ctx}",
                    h.text,
                    h.line
                );
            }
            if c.heading_level == 0 {
                assert!(c.heading_path.is_empty(), "{ctx}");
                assert_eq!(
                    (c.section_line_start, c.section_line_end),
                    (1, tree.preamble_end()),
                    "a heading-less chunk must be attributed to the preamble, not Root; {ctx}"
                );
                assert!(c.line_end <= tree.preamble_end(), "{ctx}");
            } else {
                let (i, h) = tree
                    .headings()
                    .iter()
                    .enumerate()
                    .find(|(_, h)| h.line == c.section_line_start)
                    .unwrap_or_else(|| panic!("no heading starts the chunk's section; {ctx}"));
                assert_eq!(h.level, c.heading_level, "{ctx}");
                assert_eq!(c.heading_path, tree.path(Some(i)), "{ctx}");
                assert_eq!(c.section_line_end, h.subtree_end, "{ctx}");
                assert!(
                    h.line <= c.line_start && c.line_end <= h.section_end,
                    "chunk must lie within its heading's own section {}-{}; {ctx}",
                    h.line,
                    h.section_end
                );
            }
        }
    }

    fn assert_outline_ranges_valid(content: &str) {
        for entry in crate::retrieval::outline(content) {
            assert!(
                entry.line_start <= entry.line_end,
                "outline entry {:?} has lines {}-{}; content={content:?}",
                entry.heading_path,
                entry.line_start,
                entry.line_end
            );
        }
    }

    /// Fuzz-style property test over random mixes of `\n`, `\r\n` and lone `\r`
    /// line endings between heading, setext, fence, container and text lines,
    /// with and without frontmatter: nothing panics, headings are strictly
    /// increasing by line, and every heading, chunk and outline range is valid.
    /// Deterministic (fixed-seed xorshift) so a failure reproduces.
    #[test]
    fn mixed_line_endings_never_panic_or_invert_ranges() {
        let pieces = [
            "# A",
            "## B",
            "### C",
            "#",
            "#tag",
            "text",
            "",
            "  ",
            "Setext",
            "===",
            "---",
            "```",
            "~~~",
            "> # quoted",
            "- # listed",
            "## B ##",
            "    # code",
        ];
        let endings = ["\n", "\r\n", "\r"];
        let frontmatters = ["", "---\ntitle: T\n---\n", "---\r\ntitle: T\r\n---\r\n"];
        let configs = [
            (7, None, true),
            (60, Some(1), true),
            (120, Some(40), false),
            (200, None, false),
            (1000, Some(300), true),
        ];

        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move |bound: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % bound as u64) as usize
        };

        for _ in 0..3000 {
            let mut content = frontmatters[next(frontmatters.len())].to_string();
            for _ in 0..next(14) {
                content.push_str(pieces[next(pieces.len())]);
                content.push_str(endings[next(endings.len())]);
            }
            if next(2) == 0 {
                content.push_str(pieces[next(pieces.len())]);
            }

            let (_, body) = crate::validate::parse_frontmatter_raw(&content);
            let tree = HeadingTree::parse(&body);
            let line_count = tree.line_count();
            for pair in tree.headings().windows(2) {
                assert!(pair[0].line < pair[1].line, "content={content:?}");
            }
            for heading in tree.headings() {
                assert!(
                    1 <= heading.line
                        && heading.line <= heading.section_end
                        && heading.section_end <= heading.subtree_end
                        && heading.subtree_end <= line_count,
                    "{heading:?} of {line_count}; content={content:?}"
                );
            }
            assert!(tree.preamble_end() <= line_count);

            for (max, target, prepend) in configs {
                for heading_metadata in [false, true] {
                    let cfg = ChunkingConfig {
                        max_chunk_size: max,
                        target_chunk_size: target,
                        prepend_description: true,
                        prepend_heading_path: prepend,
                        heading_metadata,
                    };
                    let chunks = crate::chunk::chunk_markdown(&body, Some("desc"), &cfg);
                    assert_chunk_ranges_valid(&body, &chunks);
                    for c in &chunks {
                        // An empty breadcrumb is never prepended, whatever the
                        // budget.
                        assert!(
                            c.text.starts_with("desc") || !c.text.starts_with('\n'),
                            "chunk {} starts with a blank breadcrumb: {:?}; content={content:?}",
                            c.index,
                            c.text
                        );
                    }
                    if heading_metadata {
                        assert_chunks_stay_within_one_section(&body, &chunks);
                    }
                }
            }
            assert_outline_ranges_valid(&content);
        }
    }

    #[test]
    fn no_headings_and_empty_body() {
        let tree = HeadingTree::parse("just text\nmore\n");
        assert!(tree.headings().is_empty());
        assert_eq!(tree.line_count(), 3);
        assert_eq!(tree.preamble_end(), 3);
        let empty = HeadingTree::parse("");
        assert_eq!(empty.line_count(), 0);
        assert_eq!(empty.preamble_end(), 0);
    }

    #[test]
    fn section_id_merge_ranges_and_levels() {
        let body = "Intro\n# A\n## B\n## C\n# D\n";
        let tree = HeadingTree::parse(body);
        let (a, b, c, d) = (
            SectionId::Heading(0),
            SectionId::Heading(1),
            SectionId::Heading(2),
            SectionId::Heading(3),
        );
        assert_eq!(b.merge(c, &tree), a);
        assert_eq!(b.merge(d, &tree), SectionId::Root);
        assert_eq!(SectionId::Preamble.merge(a, &tree), SectionId::Root);
        assert_eq!(
            SectionId::Preamble.merge(SectionId::Preamble, &tree),
            SectionId::Preamble
        );
        assert_eq!(a.line_range(&tree), (2, 4));
        assert_eq!(SectionId::Preamble.line_range(&tree), (1, 1));
        assert_eq!(SectionId::Root.line_range(&tree), (1, 6));
        assert_eq!(c.path(&tree), vec!["A", "C"]);
        assert_eq!(c.level(&tree), 2);
        assert_eq!(SectionId::Root.level(&tree), 0);
    }

    #[test]
    fn line_index_lookup_and_slicing() {
        let text = "ab\n\ncd\n";
        let idx = LineIndex::new(text);
        assert_eq!(idx.line_count(), 4);
        assert_eq!(idx.line_of(0), 1);
        assert_eq!(idx.line_of(2), 1, "a newline belongs to the line it ends");
        assert_eq!(idx.line_of(3), 2);
        assert_eq!(idx.line_of(4), 3);
        assert_eq!(&text[idx.byte_range(1, 1)], "ab");
        assert_eq!(&text[idx.byte_range(2, 3)], "\ncd");
        assert_eq!(&text[idx.byte_range(3, 4)], "cd\n");
    }

    // ── frontmatter line offset ─────────────────────────────────────────────

    #[test]
    fn body_line_offset_counts_frontmatter_and_stripped_blank_lines() {
        // Files ending in '\n' and CRLF files, where a suffix comparison
        // against the raw file would find no match.
        assert_eq!(body_line_offset("---\ntitle: T\n---\n# H\n\nBody\n"), 3);
        assert_eq!(body_line_offset("---\ntitle: x\n---\n\n# A\nbody\n"), 4);
        assert_eq!(body_line_offset("\n\n# A\nbody\n"), 2);
        assert_eq!(
            body_line_offset("---\r\ntitle: T\r\n---\r\n# H\r\n\r\nBody\r\n"),
            3
        );
        assert_eq!(body_line_offset("---\ntitle: T\n---\n# H\n\nBody"), 3);
        assert_eq!(body_line_offset("# Hello\nBody text\n"), 0);
        // Unclosed frontmatter: gray_matter treats the rest as body.
        assert_eq!(body_line_offset("---\ntitle: T\n# H\n"), 1);
        // Closing delimiter with trailing spaces, then blank lines.
        assert_eq!(body_line_offset("---  \ntitle: T\n---   \n\n\n# H\n"), 5);
        // Whitespace-only lines are not stripped by gray_matter.
        assert_eq!(body_line_offset("---\na: 1\n---\n  \n# H\n"), 3);
        assert_eq!(body_line_offset(""), 0);
        assert_eq!(body_line_offset("---\ntitle: T\n---\n"), 3);
    }

    /// Property test (#286). For every combination of frontmatter shape, leading
    /// blank lines, line ending and trailing newlines:
    /// - the offset maps gray_matter's body back onto the raw file exactly,
    /// - every heading's reported line is the raw line holding that heading,
    /// - `retrieval::outline` and `chunk::chunk_markdown` report file lines
    ///   that agree with the raw file and with each other, and every chunk's
    ///   section range covers the chunk.
    #[test]
    fn line_numbers_agree_with_the_raw_file_across_generated_variants() {
        let frontmatters = [
            "",
            "---\ntitle: T\n---\n",
            "---\ntitle: T\ntags: [a, b]\n---\n\n",
            "---   \ntitle: T\n---  \n\n\n",
            "---\ndescription: \"a --- b\"\n---\n  \n",
            // Unclosed: gray_matter makes everything after line 1 body.
            "---\ntitle: T\n",
        ];
        let leading = ["", "\n", "\n\n", "  \n"];
        let bodies = [
            "# Alpha\n\nIntro.\n\n## Beta\n\nBeta body.\n\n```\n# fenced\n```\n\nGamma Heading\n-------------\n\n### Delta `x`\n\nEnd.",
            "Preamble line.\n\n# Alpha\n\n> # quoted\n\n## Beta ##\n\ntext\n\n# Alpha\n\nagain",
            "No headings at all.\n\nJust text.",
        ];
        let trailing = ["", "\n", "\n\n", "\n\n\n"];
        let mut checked = 0;

        for fm in frontmatters {
            for lead in leading {
                for body_src in bodies {
                    for trail in trailing {
                        for crlf in [false, true] {
                            let lf = format!("{fm}{lead}{body_src}{trail}");
                            let content = if crlf { lf.replace('\n', "\r\n") } else { lf };
                            check_variant(&content);
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(checked, 6 * 4 * 3 * 4 * 2);
    }

    fn check_variant(content: &str) {
        let (_, body) = crate::validate::parse_frontmatter_raw(content);
        let offset = body_line_offset(content);
        let raw_lines: Vec<&str> = content.lines().collect();
        let ctx = format!("content={content:?}");

        assert_eq!(
            raw_lines[offset.min(raw_lines.len())..].join("\n"),
            body,
            "offset {offset} does not map the body onto the raw file; {ctx}"
        );
        let tree = HeadingTree::parse(&body);
        assert_eq!(
            offset + tree.line_count(),
            raw_lines.len(),
            "body lines + offset must equal the file's lines; {ctx}"
        );

        for heading in tree.headings() {
            let raw = raw_lines[offset + heading.line - 1];
            let first_word = heading.text.split(' ').next().unwrap();
            assert!(
                raw.contains(first_word),
                "heading {:?} reported at file line {} which is {raw:?}; {ctx}",
                heading.text,
                offset + heading.line
            );
        }

        let outline = crate::retrieval::outline(content);
        let heading_entries: Vec<_> = outline.iter().filter(|e| e.level > 0).collect();
        assert_eq!(heading_entries.len(), tree.headings().len(), "{ctx}");
        for (entry, heading) in heading_entries.iter().zip(tree.headings()) {
            assert_eq!(entry.line_start, offset + heading.line, "{ctx}");
            assert_eq!(entry.line_end, offset + heading.subtree_end, "{ctx}");
            assert!(
                raw_lines[entry.line_start - 1].contains(heading.text.split(' ').next().unwrap())
            );
        }
        if let Some(last) = outline.last() {
            assert_eq!(
                last.line_end,
                raw_lines.len(),
                "last entry must reach EOF; {ctx}"
            );
        }

        let cfg = ChunkingConfig {
            max_chunk_size: 120,
            target_chunk_size: Some(40),
            prepend_description: false,
            prepend_heading_path: false,
            heading_metadata: true,
        };
        let chunks = crate::chunk::chunk_markdown(&body, None, &cfg);
        assert_chunks_stay_within_one_section(&body, &chunks);
        for chunk in chunks {
            let (start, end) = (chunk.line_start + offset, chunk.line_end + offset);
            let (s_start, s_end) = (
                chunk.section_line_start + offset,
                chunk.section_line_end + offset,
            );
            assert!(
                s_start <= start && end <= s_end,
                "section must cover chunk; {ctx}"
            );
            assert!(end <= raw_lines.len(), "{ctx}");
            // Only the first line is compared: merged pieces are joined with a
            // blank line, so later chunk lines need not line up with the file.
            // A blank first line (a preamble opening with a whitespace-only
            // line) must sit on a blank file line — `contains("")` would accept
            // any line.
            let first = chunk.text.lines().next().unwrap().trim();
            let raw = raw_lines[start - 1];
            let matches = if first.is_empty() {
                raw.trim().is_empty()
            } else {
                raw.contains(first)
            };
            assert!(
                matches,
                "chunk starting {first:?} reported at file line {start} = {raw:?}; {ctx}"
            );
            if chunk.heading_level > 0 {
                let entry = outline
                    .iter()
                    .find(|e| e.line_start == s_start && e.level == chunk.heading_level)
                    .unwrap_or_else(|| panic!("no outline entry for chunk section; {ctx}"));
                assert_eq!(entry.heading_path, chunk.heading_path, "{ctx}");
                assert_eq!(entry.line_end, s_end, "{ctx}");
            }
        }
    }
}
