//! Markdown chunking: split a document body into `Chunk`s at heading
//! boundaries, sized against `chunking.max_chunk_size`/`target_chunk_size`,
//! with an optional `description` (frontmatter) prefix.
//!
//! Section boundaries, heading text, levels and ancestry all come from
//! [`crate::heading::HeadingTree`], the same model `retrieval::outline` reads,
//! so a chunk's `heading_path`/section range and `get_document`'s outline
//! cannot disagree.
//!
//! ## Merging across headings (`chunking.heading_metadata`, #286)
//!
//! With `heading_metadata` off, small consecutive sections are accumulated
//! into one chunk up to `target_chunk_size`, and the merged chunk is attributed
//! to the deepest heading all of its pieces share ([`SectionId::merge`]) — which
//! can be a distant ancestor, or the whole document ([`SectionId::Root`]).
//!
//! With `heading_metadata` on, chunking never merges content across a heading
//! boundary: every chunk lies within one heading's own section (its heading
//! line through the line before the next heading of any level) or within the
//! preamble. That holds for the accumulate-to-target merge and for the
//! small-fragment merges inside an oversized section alike; an oversized
//! section still splits into several chunks. So every chunk is attributed to
//! exactly `Heading(i)` or `Preamble`, never `Root`, and a `section` search row
//! names the heading whose text actually matched rather than a parent covering
//! dozens of short siblings. A heading with no body of its own (a chapter
//! heading directly followed by its first sub-heading) becomes its own small
//! chunk holding just the heading line: folding it into the first child would
//! put the child's heading line inside a chunk attributed elsewhere, and
//! dropping it would make the heading's text unsearchable except through the
//! breadcrumbs of its children.
//!
//! ## Heading-breadcrumb prefix (fix #166)
//!
//! `chunking.prepend_heading_path` (default on) additionally prepends each
//! chunk's ancestor-heading breadcrumb — e.g. `ares > Hardware` for a chunk
//! sourced from a `### GPU Backends` section nested under `# ares` >
//! `## Hardware` — so the sparse (BM25) retrieval arm has something to match
//! an identifier-style query against even when the section's own body text
//! never repeats it. See the issue for the full argument; the design choices
//! specific to this module:
//!
//! - **Ancestry, not a separate title or file path.** Both of those would
//!   require `chunk_markdown` to take a new parameter, rippling into every
//!   caller (`ingest.rs`, and the dedup-query alignment test in `mcp.rs`) that
//!   this change's scope does not touch. Heading ancestry needs nothing beyond
//!   the heading structure this module already parses — and for the common
//!   single-root-H1 document, the root heading naturally becomes the top of
//!   every subsection's breadcrumb, subsuming most of what a separate "title"
//!   would have added anyway.
//! - **Never restates a heading already visible in the chunk's own text.** A
//!   piece of text that begins with its section's own heading line gets only
//!   the *ancestors* of that heading; a continuation fragment of a split
//!   oversized section — which carries no heading line of its own — gets the
//!   full chain including that heading. When pieces are merged into one chunk,
//!   the chunk's breadcrumb is the deepest heading common to the breadcrumbs
//!   each piece would have had on its own (#286).
//! - **Budget-reserved, not appended on top.** The breadcrumb is capped
//!   ([`heading_path_budget`]) and that cap is reserved out of
//!   `max_chunk_size`/`target_chunk_size` *before* section-splitting decisions
//!   are made (`effective_max`/`effective_target` in `chunk_markdown`), so
//!   adding the prefix afterward cannot push a chunk past `max_chunk_size` in
//!   the common case. `prepend_description`'s own overhead is deliberately
//!   left as-is — a pre-existing, separate concern this change does not widen
//!   the scope to fix.

use text_splitter::MarkdownSplitter;

use crate::config::ChunkingConfig;
use crate::heading::{HeadingTree, SectionId};

pub struct Chunk {
    pub text: String,
    pub index: usize,
    /// 1-based line number where this chunk starts. Body-relative, i.e.
    /// counted from the top of the frontmatter-stripped body this module was
    /// handed. `ingest.rs` shifts it (and every other line field here) to
    /// file-relative with `heading::body_line_offset` (#286).
    pub line_start: usize,
    /// 1-based line number where this chunk ends (inclusive). Body-relative.
    pub line_end: usize,
    /// Heading path of the section this chunk is attributed to: the deepest
    /// heading whose subtree contains every piece merged into the chunk,
    /// including a leading heading's own text (unlike the rendered
    /// `prepend_heading_path` breadcrumb, which leaves out a heading the chunk
    /// already starts with). Always computed; `ingest.rs` decides whether it
    /// reaches the Qdrant payload. With `chunking.heading_metadata` on no chunk
    /// spans a heading boundary (see the module docs), so this is always the
    /// path of the one heading whose section holds the chunk. Empty when the
    /// chunk is attributed to the preamble or to the whole document.
    pub heading_path: Vec<String>,
    /// CommonMark level (1-6) of the attributed heading, or `0` when
    /// `heading_path` is empty. Taken from the heading itself rather than
    /// `heading_path.len()`: a document that jumps from `#` to `###` has a
    /// 2-element path whose last level is 3.
    pub heading_level: u8,
    /// Line range of the attributed section (#286): the attributed heading's
    /// whole subtree, the preamble when the chunk is only preamble content, or
    /// the whole body when the merged pieces share no heading. Always covers
    /// `line_start..=line_end`. Body-relative, and equal to the range
    /// `retrieval::outline` reports for the same section.
    pub section_line_start: usize,
    /// Inclusive end of `section_line_start`'s range. Body-relative.
    pub section_line_end: usize,
}

/// When the MarkdownSplitter breaks up an oversized section, merge any
/// fragment smaller than this into its neighbor to avoid orphaned headings
/// or code fence openers.
const MIN_MERGE_SIZE: usize = 200;

/// Hard cap, in characters, on the heading-breadcrumb prefix `chunking.
/// prepend_heading_path` adds to a chunk (fix #166). A document nested many
/// headings deep, or with unusually long heading text, would otherwise let the
/// breadcrumb balloon and crowd out the section content the chunk exists to
/// carry in the first place — exactly the failure mode the budget note in the
/// issue warns about. `heading_path_budget` below additionally scales this down
/// for a small `max_chunk_size`, so the reservation never dominates a tightly
/// configured chunk size either.
const MAX_HEADING_PATH_CHARS: usize = 200;

/// The character budget reserved for the heading-breadcrumb prefix (including
/// its trailing separator) against a given `max_chunk_size`. Capped at
/// [`MAX_HEADING_PATH_CHARS`] so a generously sized chunk never devotes more
/// than a small, fixed slice to breadcrumb text, and additionally capped at a
/// quarter of `max_chunk_size` so a small `max_chunk_size` (as several tests
/// below use) still leaves the large majority of the chunk for actual content.
fn heading_path_budget(max_chunk_size: usize) -> usize {
    MAX_HEADING_PATH_CHARS.min(max_chunk_size / 4)
}

/// A run of body lines between heading boundaries: either one heading's own
/// section (its heading line through the line before the next heading of any
/// level) or the non-blank preamble before the first heading.
struct Section<'a> {
    /// The section's lines, verbatim from the body.
    text: &'a str,
    /// Byte offset of `text` within the body.
    byte_start: usize,
    /// 1-based start line.
    line_start: usize,
    /// 1-based end line (inclusive).
    line_end: usize,
    /// Index of the section's heading in the tree; `None` for the preamble.
    heading: Option<usize>,
}

/// Split `body` into sections at `tree`'s heading boundaries. A preamble of
/// only blank lines is dropped (it carries nothing to embed), so the first
/// heading's section still starts with its heading line.
fn split_sections<'a>(body: &'a str, tree: &HeadingTree) -> Vec<Section<'a>> {
    let lines = tree.lines();
    let mut sections = Vec::with_capacity(tree.headings().len() + 1);
    let mut push = |first: usize, last: usize, heading: Option<usize>| {
        let range = lines.byte_range(first, last);
        sections.push(Section {
            text: &body[range.clone()],
            byte_start: range.start,
            line_start: first,
            line_end: last,
            heading,
        });
    };

    let preamble_end = tree.preamble_end();
    if preamble_end > 0 && !body[lines.byte_range(1, preamble_end)].trim().is_empty() {
        push(1, preamble_end, None);
    }
    for (i, heading) in tree.headings().iter().enumerate() {
        push(heading.line, heading.section_end, Some(i));
    }
    sections
}

/// Intermediate chunk with line tracking before final indexing.
struct RawChunk {
    text: String,
    line_start: usize,
    line_end: usize,
    /// The heading whose path is rendered as this chunk's
    /// `prepend_heading_path` breadcrumb (`None`: no breadcrumb). Seeded with
    /// the breadcrumb of the first piece and narrowed by every `append` — see
    /// the module docs.
    breadcrumb: Option<usize>,
    /// The section this chunk is attributed to, becoming `Chunk::heading_path`,
    /// `heading_level` and the section range.
    section: SectionId,
}

impl RawChunk {
    /// Append another piece of text. `breadcrumb` and `section` are the values
    /// that piece would have had as a chunk of its own; the merged chunk keeps
    /// what both have in common, compared by heading identity rather than
    /// text.
    fn append(
        &mut self,
        text: &str,
        line_end: usize,
        breadcrumb: Option<usize>,
        section: SectionId,
        tree: &HeadingTree,
    ) {
        self.text.push_str("\n\n");
        self.text.push_str(text);
        self.line_end = line_end;
        self.breadcrumb = tree.common_ancestor(self.breadcrumb, breadcrumb);
        self.section = self.section.merge(section, tree);
    }
}

/// Join a heading breadcrumb into the literal text prepended to a chunk,
/// truncated to `budget_chars`. Returns `None` for an empty path (nothing to
/// prepend), which is what keeps this a no-op for every chunk whose section
/// has no open ancestry — flat single-level documents (a document's own H1,
/// or sibling top-level headings, as in `sections_split_at_headings` below)
/// included. Also `None` for a zero budget (a `max_chunk_size` too small to
/// reserve any breadcrumb characters), so a chunk never starts with an empty
/// breadcrumb followed by the `"\n\n"` separator.
///
/// The truncation is a hard character-count cut, deliberately matching
/// `write::build_dedup_query`'s `DEDUP_QUERY_CHAR_LIMIT` truncation style
/// (`.chars().take(n).collect()`, no ellipsis) rather than word-wrapping —
/// this is a budget backstop for a pathological document, not something
/// expected to fire in normal use, so simplicity wins over a prettier cut.
fn format_heading_path(path: &[String], budget_chars: usize) -> Option<String> {
    if path.is_empty() || budget_chars == 0 {
        return None;
    }
    let joined = path.join(" > ");
    if joined.chars().count() <= budget_chars {
        Some(joined)
    } else {
        Some(joined.chars().take(budget_chars).collect())
    }
}

pub fn chunk_markdown(
    body: &str,
    description: Option<&str>,
    config: &ChunkingConfig,
) -> Vec<Chunk> {
    let target = config.target();
    let max = config.max_chunk_size;

    // Reserve budget for the heading-breadcrumb prefix *before* deciding where
    // sections get split/merged, so that adding the prefix afterward cannot
    // push a chunk past `max_chunk_size` in the common (non-oversized-section)
    // path — see the doc comment on `heading_path_budget` and the budget tests
    // below. `effective_max`/`effective_target` are what all of this
    // function's internal sizing decisions use in place of the raw config
    // values; `max` itself is kept around only to compute the reservation and
    // to size the final prefix truncation.
    let heading_reserve = if config.prepend_heading_path {
        heading_path_budget(max)
    } else {
        0
    };
    // `- 2` reserves room for the "\n\n" separator placed between the prefix
    // and what follows it, so the *combined* prefix+separator never exceeds
    // `heading_reserve` — see the final assembly step below.
    let heading_path_chars = heading_reserve.saturating_sub(2);
    let effective_max = max.saturating_sub(heading_reserve);
    let effective_target = target.saturating_sub(heading_reserve);
    // With `heading_metadata` on, no chunk crosses a heading boundary — see the
    // module docs.
    let isolate_sections = config.heading_metadata;

    let tree = HeadingTree::parse(body);
    let sections = split_sections(body, &tree);

    // Greedily accumulate sections into chunks up to target size.
    // If a single section exceeds max, use MarkdownSplitter to break it down.
    let mut chunks: Vec<RawChunk> = Vec::new();
    let mut current: Option<RawChunk> = None;

    for section in sections {
        // A piece that starts with the section's heading line gets the
        // heading's ancestors as its breadcrumb; a piece that does not gets
        // the heading itself. The preamble has neither.
        let leading_breadcrumb = section.heading.and_then(|i| tree.headings()[i].parent);
        let continuation_breadcrumb = section.heading;
        let section_id = section
            .heading
            .map_or(SectionId::Preamble, SectionId::Heading);

        if section.text.trim().len() > effective_max {
            // Flush current accumulator first
            if let Some(cur) = current.take() {
                chunks.push(cur);
            }
            // Split oversized section with MarkdownSplitter, but merge
            // small leading fragments (headings, code fence openers) forward
            // so they stay attached to the content they introduce.
            let splitter = MarkdownSplitter::new(effective_max);
            let mut pending: Option<RawChunk> = None;
            // `chunk_indices` yields each fragment's byte offset within
            // `section.text`. MarkdownSplitter trims leading/trailing
            // whitespace from fragments, so the offset points at the
            // fragment's first non-whitespace character and the body's line
            // index gives its exact first and last line.
            //
            // Only the first fragment contains the section's own heading line
            // (the "always merge a tiny pending fragment forward" rule below
            // keeps it attached to content), so only it takes the leading
            // breadcrumb.
            for (frag_idx, (byte_offset, part)) in splitter.chunk_indices(section.text).enumerate()
            {
                let abs_start = section.byte_start + byte_offset;
                let frag_line_start = tree.lines().line_of(abs_start);
                // Clamped to the section and to the fragment's own start so the
                // range can never invert, whatever the splitter reports (#286).
                let frag_line_end = tree
                    .lines()
                    .line_of(abs_start + part.len().saturating_sub(1))
                    .min(section.line_end)
                    .max(frag_line_start);
                let frag_breadcrumb = if frag_idx == 0 {
                    leading_breadcrumb
                } else {
                    continuation_breadcrumb
                };
                let new_chunk = || RawChunk {
                    text: part.to_string(),
                    line_start: frag_line_start,
                    line_end: frag_line_end,
                    breadcrumb: frag_breadcrumb,
                    section: section_id,
                };

                if let Some(mut prev) = pending.take() {
                    let prev_len = prev.text.trim().len();
                    let combined = prev_len + 2 + part.trim().len();
                    // Always merge a tiny pending fragment (e.g. a lone heading)
                    // forward regardless of size — a heading-only chunk is useless.
                    // Only reject the merge when prev is already a substantial chunk
                    // and combining would exceed max.
                    if combined <= effective_max || prev_len < MIN_MERGE_SIZE {
                        prev.append(part, frag_line_end, frag_breadcrumb, section_id, &tree);
                        if prev.text.trim().len() < MIN_MERGE_SIZE {
                            pending = Some(prev);
                        } else {
                            chunks.push(prev);
                        }
                    } else {
                        // Would overflow and prev is already substantial — push
                        // prev as-is, then handle part independently.
                        chunks.push(prev);
                        if part.trim().len() < MIN_MERGE_SIZE {
                            pending = Some(new_chunk());
                        } else {
                            chunks.push(new_chunk());
                        }
                    }
                } else if part.trim().len() < MIN_MERGE_SIZE {
                    pending = Some(new_chunk());
                } else {
                    chunks.push(new_chunk());
                }
            }
            // Trailing small fragment — append to last chunk if it fits.
            // `last` is normally an earlier fragment of this section, but can
            // be the previous section's chunk when no fragment of this one was
            // pushed (only possible with a tiny `max_chunk_size`); `append`
            // narrows correctly either way. With `isolate_sections` the tail
            // only joins a chunk of its own section: every section has a
            // distinct `SectionId`, so comparing ids is enough.
            if let Some(tail) = pending.take() {
                if let Some(last) = chunks.last_mut() {
                    let combined = last.text.trim().len() + 2 + tail.text.trim().len();
                    let same_section = last.section == tail.section;
                    if combined <= effective_max && (same_section || !isolate_sections) {
                        last.append(
                            &tail.text,
                            tail.line_end,
                            tail.breadcrumb,
                            tail.section,
                            &tree,
                        );
                    } else {
                        chunks.push(tail);
                    }
                } else {
                    chunks.push(tail);
                }
            }
            continue;
        }

        let combined_len = if let Some(ref cur) = current {
            cur.text.trim().len() + 2 + section.text.trim().len()
        } else {
            section.text.trim().len()
        };

        let seeded = || RawChunk {
            text: section.text.to_string(),
            line_start: section.line_start,
            line_end: section.line_end,
            breadcrumb: leading_breadcrumb,
            section: section_id,
        };
        if combined_len <= effective_target && !isolate_sections {
            // Fits within target — accumulate. `cur` and `section` can be
            // different sections under different parents, so this is where
            // narrowing usually does something.
            match current {
                Some(ref mut cur) => cur.append(
                    section.text,
                    section.line_end,
                    leading_breadcrumb,
                    section_id,
                    &tree,
                ),
                None => current = Some(seeded()),
            }
        } else {
            // Would exceed target, or sections are isolated — flush and start
            // a new chunk.
            if let Some(cur) = current.take() {
                chunks.push(cur);
            }
            current = Some(seeded());
        }
    }

    if let Some(cur) = current.take() {
        chunks.push(cur);
    }

    chunks
        .into_iter()
        .enumerate()
        .map(|(index, raw)| {
            // Order: heading breadcrumb, then description, then body — outermost
            // structural context first, then the document-level summary, then
            // the section content itself. Building `with_desc` first and only
            // then optionally prepending the heading path means that with
            // `prepend_heading_path` off (or an empty breadcrumb — the common
            // case for a flat, unnested document) this produces byte-for-byte
            // the same text `prepend_description` alone always has, which is
            // exactly what keeps this change from disturbing the existing
            // description-prepend behavior and its tests.
            let with_desc = match (config.prepend_description, description) {
                (true, Some(desc)) => format!("{}\n\n{}", desc, raw.text),
                _ => raw.text,
            };
            let text = if config.prepend_heading_path {
                match format_heading_path(&tree.path(raw.breadcrumb), heading_path_chars) {
                    Some(prefix) => format!("{}\n\n{}", prefix, with_desc),
                    None => with_desc,
                }
            } else {
                with_desc
            };
            let (section_line_start, section_line_end) = raw.section.line_range(&tree);
            Chunk {
                text,
                index,
                line_start: raw.line_start,
                line_end: raw.line_end,
                heading_path: raw.section.path(&tree),
                heading_level: raw.section.level(&tree),
                section_line_start,
                section_line_end,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(
        max: usize,
        target: Option<usize>,
        prepend_description: bool,
        prepend_heading_path: bool,
    ) -> ChunkingConfig {
        ChunkingConfig {
            max_chunk_size: max,
            target_chunk_size: target,
            prepend_description,
            prepend_heading_path,
            // Off: the merge-across-headings behavior most tests here pin.
            // `cfg_isolated` turns it on for the no-merge tests.
            heading_metadata: false,
        }
    }

    #[test]
    fn single_chunk_short_text() {
        let chunks = chunk_markdown("Hello world", None, &cfg(1000, None, false, false));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Hello world");
        assert_eq!(chunks[0].index, 0);
    }

    #[test]
    fn sections_split_at_headings() {
        let filler = "Word ".repeat(60); // ~300 chars per section
        let body = format!("# Section 1\n\n{filler}\n\n# Section 2\n\n{filler}");
        // target=400 means each ~315-char section gets its own chunk
        let chunks = chunk_markdown(&body, None, &cfg(1500, Some(400), false, false));
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.starts_with("# Section 1"));
        assert!(chunks[1].text.starts_with("# Section 2"));
    }

    #[test]
    fn small_sections_combined_to_target() {
        let body = "# A\n\nSmall.\n\n# B\n\nAlso small.\n\n# C\n\nTiny.";
        // Everything is well under target, should combine into one chunk
        let chunks = chunk_markdown(body, None, &cfg(1500, Some(1000), false, false));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("# A"));
        assert!(chunks[0].text.contains("# C"));
    }

    #[test]
    fn oversized_section_split_by_splitter() {
        let big = "Word ".repeat(400); // ~2000 chars
        let body = format!("## Big Section\n\n{big}");
        let chunks = chunk_markdown(&body, None, &cfg(1000, Some(800), false, false));
        assert!(chunks.len() >= 2, "Oversized section should be split");
        // Allow up to max + MIN_MERGE_SIZE to accommodate a tiny pending heading
        // (< MIN_MERGE_SIZE chars) that is always merged forward into the first
        // content chunk to keep it attached to its section.
        let limit = 1000 + MIN_MERGE_SIZE;
        for chunk in &chunks {
            assert!(
                chunk.text.trim().len() <= limit,
                "No chunk should wildly exceed max (got {} chars trimmed, limit {})",
                chunk.text.trim().len(),
                limit,
            );
        }
    }

    #[test]
    fn heading_stays_with_content() {
        let filler_a = "Content A. ".repeat(50); // ~550 chars
        let filler_b = "Content B. ".repeat(50);
        let body = format!("## Section A\n\n{filler_a}\n\n## Section B\n\n{filler_b}");
        // target=600 — each section fits on its own but not combined
        let chunks = chunk_markdown(&body, None, &cfg(1500, Some(600), false, false));
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.starts_with("## Section A"));
        assert!(chunks[0].text.contains("Content A"));
        assert!(chunks[1].text.starts_with("## Section B"));
        assert!(chunks[1].text.contains("Content B"));
    }

    #[test]
    fn prepend_description() {
        let chunks = chunk_markdown(
            "Body text",
            Some("A description"),
            &cfg(1000, None, true, false),
        );
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.starts_with("A description\n\n"));
        assert!(chunks[0].text.contains("Body text"));
    }

    #[test]
    fn no_prepend_when_disabled() {
        let chunks = chunk_markdown(
            "Body text",
            Some("A description"),
            &cfg(1000, None, false, false),
        );
        assert_eq!(chunks[0].text, "Body text");
    }

    #[test]
    fn prepend_description_all_chunks() {
        // Create two sections large enough to land in separate chunks (target=400)
        let filler = "Word ".repeat(60); // ~300 chars per section
        let body = format!("# Section 1\n\n{filler}\n\n# Section 2\n\n{filler}");
        let chunks = chunk_markdown(
            &body,
            Some("My description"),
            &cfg(1500, Some(400), true, false),
        );
        assert!(chunks.len() >= 2, "Expected multiple chunks for this test");
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.text.starts_with("My description\n\n"),
                "Chunk {i} does not start with the description"
            );
        }
    }

    #[test]
    fn empty_body() {
        let chunks = chunk_markdown("", None, &cfg(1000, None, false, false));
        assert!(chunks.is_empty());
    }

    #[test]
    fn target_defaults_to_max() {
        let c = cfg(1500, None, false, false);
        assert_eq!(c.target(), 1500);
    }

    #[test]
    fn oversized_section_heading_stays_with_code_block() {
        // Heading + large code block in one section — when split by
        // MarkdownSplitter, the heading must stay attached to content.
        let big_yaml = "  key: value\n".repeat(150); // ~1950 chars
        let body = format!("## Docker Compose\n\n```yaml\n{big_yaml}```");
        let chunks = chunk_markdown(&body, None, &cfg(1500, Some(1000), false, false));
        assert!(
            chunks[0].text.contains("## Docker Compose"),
            "First chunk must contain the heading"
        );
        assert!(
            chunks[0].text.contains("key: value"),
            "First chunk must contain code block content, not just the heading"
        );
    }

    #[test]
    fn split_sections_basic() {
        let body = "# A\n\nContent A\n\n## B\n\nContent B";
        let tree = HeadingTree::parse(body);
        let sections = split_sections(body, &tree);
        assert_eq!(sections.len(), 2);
        // # A        = line 1
        // (blank)    = line 2
        // Content A  = line 3
        // (blank)    = line 4  ← still part of section A
        // ## B       = line 5
        // (blank)    = line 6
        // Content B  = line 7
        assert!(sections[0].text.starts_with("# A"));
        assert_eq!(sections[0].line_start, 1);
        assert_eq!(sections[0].line_end, 4);
        assert!(sections[1].text.starts_with("## B"));
        assert_eq!(sections[1].line_start, 5);
        assert_eq!(sections[1].line_end, 7);
    }

    #[test]
    fn chunks_have_line_ranges() {
        let filler = "Word ".repeat(50); // ~250 chars
        let body = format!("# A\n\n{filler}\n\n## B\n\n{filler}");
        let chunks = chunk_markdown(&body, None, &cfg(1500, Some(300), false, false));
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].line_start, 1);
        assert!(chunks[0].line_end > 1);
        assert!(chunks[1].line_start > chunks[0].line_end);
        assert!(chunks[1].line_end >= chunks[1].line_start);
    }

    #[test]
    fn oversized_section_never_exceeds_max_chunk_size() {
        // Generate a very large section with many paragraphs
        let paragraphs: Vec<String> = (0..20)
            .map(|i| {
                format!(
                    "Paragraph {}. {}",
                    i,
                    "Lorem ipsum dolor sit amet. ".repeat(10)
                )
            })
            .collect();
        let body = format!("## Big\n\n{}", paragraphs.join("\n\n"));
        let max = 1000;
        let chunks = chunk_markdown(&body, None, &cfg(max, Some(800), false, false));
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.text.trim().len() <= max,
                "Chunk {} has {} chars (max {})",
                i,
                chunk.text.trim().len(),
                max,
            );
        }
    }

    #[test]
    fn trailing_fragment_overflow_creates_own_chunk() {
        // Build a section where the last splitter fragment is small but the
        // previous chunk is already near max — merging would overflow.
        let near_max = "X ".repeat(490); // ~980 chars
        let tail = "Tail content here."; // small
        let body = format!("## Title\n\n{}\n\n{}", near_max, tail);
        let max = 1000;
        let chunks = chunk_markdown(&body, None, &cfg(max, Some(800), false, false));
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.text.trim().len() <= max,
                "Chunk {} has {} chars (max {})",
                i,
                chunk.text.trim().len(),
                max,
            );
        }
    }

    #[test]
    fn two_consecutive_small_fragments_stay_within_max() {
        // Two small parts that individually are below MIN_MERGE_SIZE but
        // together with a near-max preceding chunk would overflow.
        let big_part = "Y ".repeat(480); // ~960 chars
        let small_a = "Alpha. ".repeat(5); // ~35 chars
        let small_b = "Beta. ".repeat(5); // ~30 chars
        let body = format!("## S\n\n{}\n\n{}\n\n{}", big_part, small_a, small_b);
        let max = 1000;
        let chunks = chunk_markdown(&body, None, &cfg(max, Some(800), false, false));
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.text.trim().len() <= max,
                "Chunk {} has {} chars (max {})",
                i,
                chunk.text.trim().len(),
                max,
            );
        }
    }

    #[test]
    fn oversized_section_sub_chunks_have_distinct_monotonic_line_starts() {
        // Build a section large enough to produce at least 2 sub-chunks.
        // Each paragraph is on its own line so splitter fragments land on
        // different lines and we can verify monotonic line_start values.
        let paragraphs: Vec<String> = (0..30)
            .map(|i| {
                format!(
                    "Paragraph {}. {}",
                    i,
                    "Lorem ipsum dolor sit amet. ".repeat(5)
                )
            })
            .collect();
        // section_start is line 1 (the heading), paragraphs start at line 3
        let body = format!("## Large Section\n\n{}", paragraphs.join("\n\n"));
        let max = 600;
        let chunks = chunk_markdown(&body, None, &cfg(max, Some(400), false, false));
        assert!(
            chunks.len() >= 2,
            "Expected >=2 sub-chunks from oversized section, got {}",
            chunks.len()
        );
        // Sub-chunk line_start values must be strictly increasing.
        for w in chunks.windows(2) {
            assert!(
                w[1].line_start > w[0].line_start,
                "line_start not monotonically increasing: chunk has line_start={} after chunk with line_start={}",
                w[1].line_start,
                w[0].line_start,
            );
        }
        // All sub-chunks must stay within the document's line range.
        let total_lines = body.lines().count();
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.line_start >= 1,
                "Chunk {} line_start {} is below 1",
                i,
                chunk.line_start,
            );
            assert!(
                chunk.line_end <= total_lines,
                "Chunk {} line_end {} exceeds document line count {}",
                i,
                chunk.line_end,
                total_lines,
            );
            assert!(
                chunk.line_end >= chunk.line_start,
                "Chunk {} has line_end {} < line_start {}",
                i,
                chunk.line_end,
                chunk.line_start,
            );
        }
    }

    #[test]
    fn single_chunk_section_has_correct_line_range() {
        // A section that fits in one chunk must still report accurate line ranges.
        // Line 1: "# Title"
        // Line 2: "" (blank)
        // Line 3: "Line two."
        // Line 4: "Line three."
        // Line 5: "Line four."
        let body = "# Title\n\nLine two.\nLine three.\nLine four.";
        let chunks = chunk_markdown(body, None, &cfg(1000, None, false, false));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].line_start, 1);
        assert_eq!(chunks[0].line_end, 5);
    }

    // ── chunking.prepend_heading_path (fix #166) ────────────────────────────

    #[test]
    fn heading_path_prepended_across_nesting_levels() {
        // target=Some(1) forces every section into its own chunk regardless of
        // size (effective_target saturates to 0), which is what isolates the
        // deeply nested "### GPU Backends" section from its ancestor headings
        // so there is something real to prepend.
        let body = "# ares\n\n## Hardware\n\n### GPU Backends\n\nROCm and Vulkan notes.";
        let chunks = chunk_markdown(body, None, &cfg(1500, Some(1), false, true));
        assert_eq!(chunks.len(), 3, "each heading should land in its own chunk");

        // The root section's own chunk needs no breadcrumb — it has no
        // ancestors, and it already carries "# ares" as its literal first line.
        assert!(chunks[0].text.starts_with("# ares"));
        assert!(
            !chunks[0].text.contains(" > "),
            "root section should carry no breadcrumb, got: {:?}",
            chunks[0].text
        );

        // One level of ancestry.
        assert!(
            chunks[1].text.starts_with("ares\n\n## Hardware"),
            "got: {:?}",
            chunks[1].text
        );

        // Two levels of ancestry, outermost first, own heading excluded (it's
        // already the first line of the section text that follows).
        assert!(
            chunks[2]
                .text
                .starts_with("ares > Hardware\n\n### GPU Backends"),
            "got: {:?}",
            chunks[2].text
        );
        assert!(chunks[2].text.contains("ROCm and Vulkan notes."));
    }

    #[test]
    fn heading_path_disabled_by_config() {
        // Same nested body as heading_path_prepended_across_nesting_levels, but
        // with the knob off — no chunk should gain a breadcrumb.
        let body = "# ares\n\n## Hardware\n\n### GPU Backends\n\nROCm and Vulkan notes.";
        let chunks = chunk_markdown(body, None, &cfg(1500, Some(1), false, false));
        assert_eq!(chunks.len(), 3);
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                !chunk.text.contains(" > "),
                "chunk {i} should carry no breadcrumb with the knob off, got: {:?}",
                chunk.text
            );
        }
    }

    #[test]
    fn flat_single_heading_document_gets_no_heading_path_prefix() {
        // Regression guard for a cross-file invariant this module cannot enforce
        // by compiling against it directly: `mcp.rs`'s
        // `build_dedup_query_matches_chunk_prepend_format` pins dedup-query /
        // chunk-text alignment using a body of exactly this flat, unnested shape
        // under `ChunkingConfig::default()` (both prepend knobs on). Since this
        // body's only heading has no ancestors — it IS the top of its own
        // (trivial) hierarchy (`heading::Heading::parent` is `None`) — the
        // heading-path prefix must stay empty here, or that alignment (and its
        // test, outside this module's scope) breaks.
        let body = "## Heading\n\nSome body content.";
        let description = "A short summary.";
        let config = ChunkingConfig::default();
        assert!(
            config.prepend_heading_path,
            "this test assumes the production default"
        );
        let chunks = chunk_markdown(body, Some(description), &config);
        assert_eq!(
            chunks[0].text,
            format!("{}\n\n{}", description, body),
            "a flat, unnested heading must not gain a heading-path prefix"
        );
    }

    #[test]
    fn heading_path_and_prepend_description_compose_in_order() {
        // With both knobs on, the ordering is breadcrumb, then description,
        // then body — see the comment on the final assembly step in
        // `chunk_markdown` for why (and for why this ordering is what keeps
        // `prepend_description`'s own tests undisturbed when there is no
        // breadcrumb to add).
        let body = "# Root\n\n## Nested\n\nBody content here.";
        let chunks = chunk_markdown(body, Some("A description"), &cfg(1500, Some(1), true, true));
        // chunks[1] is the "## Nested" section, one level of ancestry below Root.
        assert!(
            chunks[1]
                .text
                .starts_with("Root\n\nA description\n\n## Nested"),
            "got: {:?}",
            chunks[1].text
        );
    }

    #[test]
    fn heading_path_budget_respects_max_chunk_size() {
        // Build a long, deeply nested ancestor chain (far longer than the
        // reserved breadcrumb budget once joined) sitting above a section whose
        // own content is sized to use most of the remaining budget, and confirm
        // the *final*, prefixed chunk text still never exceeds max_chunk_size —
        // not "max_chunk_size plus the prefix", which is what a naive
        // unconditional prepend (like `prepend_description`'s, a pre-existing
        // and deliberately separate concern) would produce.
        //
        // Ancestors use CommonMark's real levels 1-5 under a level-6 section
        // (7+ `#` is paragraph text, not a heading, so it cannot nest deeper).
        let max = 1000;
        let mut body = String::new();
        for level in 1..=5 {
            body.push_str(&"#".repeat(level));
            body.push_str(&format!(
                " AncestorLevel{level:02}WithSomeExtraPaddingToBeLongEnough\n\n"
            ));
        }
        let filler = "Y ".repeat(375); // ~750 chars — large but not oversized
        body.push_str(&"#".repeat(6));
        body.push_str(" DeepSection\n\n");
        body.push_str(&filler);

        // target=Some(1) isolates the deep section from its (tiny) ancestor
        // heading-only sections, same technique as the nesting test above.
        let chunks = chunk_markdown(&body, None, &cfg(max, Some(1), false, true));
        let deep_chunk = chunks.last().expect("expected at least one chunk");
        assert!(
            deep_chunk.text.contains("DeepSection"),
            "expected the deep section's own chunk, got: {:?}",
            deep_chunk.text
        );

        assert!(
            deep_chunk.text.trim().len() <= max,
            "chunk length {} must respect max_chunk_size {} even with a long \
             ancestor chain prepended",
            deep_chunk.text.trim().len(),
            max,
        );

        // The breadcrumb itself must have been truncated to the reserved
        // budget rather than being left to run away with the whole chunk.
        let (prefix, rest) = deep_chunk
            .text
            .split_once("\n\n")
            .expect("chunk should have a breadcrumb separated from its body by a blank line");
        assert!(
            prefix.chars().count() <= heading_path_budget(max),
            "breadcrumb should be truncated to the reserved budget, got {} chars: {:?}",
            prefix.chars().count(),
            prefix,
        );
        assert!(
            prefix.starts_with("AncestorLevel01WithSomeExtraPaddingToBeLongEnough > "),
            "breadcrumb must be the ancestor chain, got: {prefix:?}"
        );
        let untruncated = (1..=5)
            .map(|l| format!("AncestorLevel{l:02}WithSomeExtraPaddingToBeLongEnough"))
            .collect::<Vec<_>>()
            .join(" > ");
        assert!(
            untruncated.chars().count() > heading_path_budget(max),
            "fixture must actually exercise truncation"
        );
        assert!(
            rest.starts_with("###### DeepSection"),
            "body must still start with the deep section's own heading line, got: {:?}",
            rest
        );
    }

    #[test]
    fn oversized_section_continuation_fragment_gets_full_heading_path() {
        // A section nested two levels deep, oversized enough that
        // MarkdownSplitter breaks it into multiple fragments. Only the first
        // fragment carries the section's own "### Big" heading line verbatim
        // (per the "always merge a tiny pending fragment forward" rule this
        // mirrors); every fragment after that is pure body text with no
        // heading of its own, so its breadcrumb is the section's full path
        // (restating "Big") rather than only the heading's ancestors (which
        // would leave that fragment with no indication at all of which
        // section it belongs to) — this is also
        // exactly the case the MIN_MERGE_SIZE small-fragment-merging logic
        // still has to behave correctly under, since it runs unmodified against
        // `effective_max` here.
        let filler = "Lorem ipsum dolor sit amet consectetur. ".repeat(80); // ~3400 chars
        let body = format!("# Root\n\n## Section\n\n### Big\n\n{filler}");
        let chunks = chunk_markdown(&body, None, &cfg(1000, Some(800), false, true));
        assert!(
            chunks.len() >= 3,
            "expected the heading-only Root/Section chunk plus 2+ split \
             fragments of Big, got {}",
            chunks.len()
        );

        // chunks[0]: "# Root" + "## Section" merged (both tiny, well under
        // target) and flushed when the oversized "### Big" section is hit.
        assert!(chunks[0].text.starts_with("# Root"));

        // chunks[1]: the first fragment of "### Big" — carries the heading line
        // itself, so the breadcrumb excludes "Big" (ancestors only).
        assert!(
            chunks[1].text.starts_with("Root > Section\n\n### Big"),
            "got: {:?}",
            chunks[1].text
        );

        // chunks[2]: a later fragment — no heading line in its own text, so
        // the breadcrumb must restate "Big" too (the full path).
        assert!(
            !chunks[2].text.contains("### Big"),
            "later fragment should be pure body text with no heading line, got: {:?}",
            chunks[2].text
        );
        assert!(
            chunks[2].text.starts_with("Root > Section > Big\n\n"),
            "got: {:?}",
            chunks[2].text
        );
    }

    // ── merged-chunk breadcrumb narrowing (#286) ────────────────────────────

    #[test]
    fn merged_chunk_breadcrumb_narrows_to_shared_ancestor() {
        // "### Blinded" (nested under "## Conditions", itself under "# Root")
        // is small enough to merge forward with the following "## Actions"
        // section — a sibling of "## Conditions", not "### Blinded" itself.
        // Before the #286 fix, the merged chunk kept only the ancestors of
        // "### Blinded" ("Root > Conditions"), misattributing the
        // "## Actions" content it now also carries to a heading it was never
        // under. The fix narrows the breadcrumb to the deepest heading both
        // merged sections actually share: "Root".
        let filler = "Word ".repeat(60); // ~300 chars, forces Root/Conditions to flush separately
        let body = format!(
            "# Root\n\n## Conditions\n\n{filler}\n\n### Blinded\n\nBlinded is bad.\n\n## Actions\n\nActions cost."
        );
        let chunks = chunk_markdown(&body, None, &cfg(1500, Some(300), false, true));
        assert_eq!(
            chunks.len(),
            3,
            "expected Root, Conditions, and a merged Blinded+Actions chunk, got: {:?}",
            chunks.iter().map(|c| &c.text).collect::<Vec<_>>()
        );
        let merged = &chunks[2];
        assert!(
            merged.text.starts_with("Root\n\n### Blinded"),
            "breadcrumb should narrow to the shared ancestor \"Root\", got: {:?}",
            merged.text
        );
        assert!(
            merged.text.contains("## Actions"),
            "merged chunk should still carry the Actions section, got: {:?}",
            merged.text
        );
    }

    #[test]
    fn sibling_merge_leaves_breadcrumb_unchanged() {
        // "## A" and "## B" are true siblings — both nested directly under
        // "# Root" with no heading of their own between them — so merging
        // them must leave the breadcrumb exactly as it was for "## A" alone:
        // narrowing against an identical set of ancestors is a no-op.
        let filler = "Word ".repeat(60); // ~300 chars, forces Root to flush before A/B
        let body = format!("# Root\n\n{filler}\n\n## A\n\nSmall A.\n\n## B\n\nSmall B.");
        let chunks = chunk_markdown(&body, None, &cfg(1500, Some(300), false, true));
        assert_eq!(
            chunks.len(),
            2,
            "expected the Root chunk and a merged A+B chunk, got: {:?}",
            chunks.iter().map(|c| &c.text).collect::<Vec<_>>()
        );
        let merged = &chunks[1];
        assert!(
            merged.text.starts_with("Root\n\n## A"),
            "sibling merge must not change the breadcrumb, got: {:?}",
            merged.text
        );
        assert!(merged.text.contains("## B"));
    }

    #[test]
    fn raw_chunk_append_narrows_by_heading_identity() {
        // White-box test of `RawChunk::append`. Root > Conditions > Blinded,
        // then Root > Actions: both the breadcrumb and the section narrow to
        // Root.
        let body = "# Root\n## Conditions\n### Blinded\nBlinded is bad.\n## Actions\nActions cost.";
        let tree = HeadingTree::parse(body);
        let mut chunk = RawChunk {
            text: "### Blinded\nBlinded is bad.".to_string(),
            line_start: 3,
            line_end: 4,
            breadcrumb: Some(1),
            section: SectionId::Heading(2),
        };
        chunk.append(
            "## Actions\nActions cost.",
            6,
            Some(0),
            SectionId::Heading(3),
            &tree,
        );
        assert_eq!(chunk.breadcrumb, Some(0));
        assert_eq!(chunk.section, SectionId::Heading(0));
        assert_eq!(chunk.line_end, 6);
    }

    #[test]
    fn level_seven_hashes_are_section_text_not_a_boundary() {
        // Regression (#286): seven `#` is paragraph text in CommonMark, so it
        // neither splits the section nor re-parents anything.
        let body = "## Real Heading\n\n####### Not a heading\n\nSome text.";
        let tree = HeadingTree::parse(body);
        let sections = split_sections(body, &tree);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].line_end, 5);
        let chunks = chunk_markdown(body, None, &cfg(1000, None, false, false));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading_path, vec!["Real Heading".to_string()]);
    }

    // ── attribution regressions (#286) ──────────────────────────────────────

    /// Asserts the invariant ingest relies on: the attributed section range
    /// always covers the chunk.
    fn assert_sections_cover_chunks(chunks: &[Chunk]) {
        for c in chunks {
            assert!(
                c.section_line_start <= c.line_start && c.line_end <= c.section_line_end,
                "chunk {} lines {}-{} not covered by its section {}-{}",
                c.index,
                c.line_start,
                c.line_end,
                c.section_line_start,
                c.section_line_end
            );
        }
    }

    #[test]
    fn continuation_fragments_merged_together_keep_the_sections_own_heading() {
        // #286: a small continuation paragraph merged with the next one keeps
        // the section's own heading in its breadcrumb ("R > Big", not just the
        // ancestors' "R"), as #166 requires for continuation fragments.
        let long_a = format!("Alpha {}", "alpha ".repeat(115)); // ~700 chars
        let short = format!("Short {}", "short ".repeat(15)); // ~100 chars
        let long_b = format!("Beta {}", "beta ".repeat(139)); // ~700 chars
        let body = format!("# R\n\n## Big\n\n{long_a}\n\n{short}\n\n{long_b}");
        let chunks = chunk_markdown(&body, None, &cfg(1000, Some(800), false, true));
        assert_sections_cover_chunks(&chunks);

        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        // The tiny "# R" section is flushed on its own when the oversized
        // "## Big" section is reached.
        assert_eq!(chunks.len(), 3, "got: {texts:?}");
        assert_eq!(chunks[0].text, "# R\n");
        assert!(
            chunks[1].text.starts_with("R\n\n## Big\n\nAlpha"),
            "got: {:?}",
            chunks[1].text
        );
        let merged = &chunks[2];
        assert!(
            merged.text.contains("Short ") && merged.text.contains("Beta "),
            "the short paragraph must merge forward into the next one; got: {texts:?}"
        );
        assert!(
            merged.text.starts_with("R > Big\n\nShort "),
            "got: {:?}",
            merged.text
        );
        for c in &chunks[1..] {
            assert_eq!(c.heading_path, vec!["R".to_string(), "Big".to_string()]);
        }
    }

    #[test]
    fn merge_narrowing_uses_identity_not_text_for_same_named_headings() {
        // #286: `### C` then a *different* `## C`. Narrowing is by heading
        // identity, so the merged chunk is attributed to their common ancestor
        // `A`, not to either `C`.
        let filler = "Word ".repeat(60);
        let body = format!("# A\n\n{filler}\n\n### C\n\nsmall\n\n## C\n\nsmall2");
        let chunks = chunk_markdown(&body, None, &cfg(1500, Some(300), false, false));
        assert_sections_cover_chunks(&chunks);
        let merged = chunks.last().unwrap();
        assert!(merged.text.starts_with("### C") && merged.text.contains("## C"));
        assert_eq!(merged.heading_path, vec!["A".to_string()]);
        assert_eq!(merged.heading_level, 1);
        assert_eq!(
            (merged.section_line_start, merged.section_line_end),
            (1, 11)
        );
        assert_eq!((merged.line_start, merged.line_end), (5, 11));

        // #286: `## X` merged with a second, unrelated `# R`.
        let body = format!("# R\n\n{filler}\n\n## X\n\nsmall x\n\n# R\n\nsmall r2");
        let chunks = chunk_markdown(&body, None, &cfg(1500, Some(300), false, true));
        assert_sections_cover_chunks(&chunks);
        let merged = chunks.last().unwrap();
        assert!(
            merged.text.starts_with("## X"),
            "no shared heading, so no breadcrumb; got: {:?}",
            merged.text
        );
        assert!(merged.heading_path.is_empty());
        assert_eq!(merged.heading_level, 0);
        assert_eq!(
            (merged.section_line_start, merged.section_line_end),
            (1, 11)
        );
    }

    #[test]
    fn preamble_merged_with_sections_is_attributed_to_the_whole_body() {
        // #286: a chunk holding the preamble and later sections is attributed
        // to the whole body, not to the preamble's own 1-2 range.
        let body = "Intro text.\n\n# A\n\nA body.\n\n## B\n\nB body.";
        let chunks = chunk_markdown(body, None, &cfg(1000, None, false, false));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].heading_path.is_empty());
        assert_eq!((chunks[0].line_start, chunks[0].line_end), (1, 9));
        assert_eq!(
            (chunks[0].section_line_start, chunks[0].section_line_end),
            (1, 9)
        );
    }

    #[test]
    fn preamble_alone_is_attributed_to_the_preamble() {
        let filler = "Word ".repeat(60);
        let body = format!("Intro {filler}\n\n# A\n\n{filler}");
        let chunks = chunk_markdown(&body, None, &cfg(1500, Some(300), false, false));
        assert_sections_cover_chunks(&chunks);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].heading_path.is_empty());
        assert_eq!(
            (chunks[0].section_line_start, chunks[0].section_line_end),
            (1, 2)
        );
        assert_eq!(chunks[1].heading_path, vec!["A".to_string()]);
        assert_eq!(
            (chunks[1].section_line_start, chunks[1].section_line_end),
            (3, 5)
        );
    }

    #[test]
    fn blank_preamble_does_not_hide_the_first_heading() {
        // Regression (#286): a whitespace-only first line must not swallow the
        // first heading into an unnamed section.
        let body = "  \n# A\nbody\n## B\nx";
        let chunks = chunk_markdown(body, None, &cfg(1000, Some(1), false, true));
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.starts_with("# A"));
        assert_eq!(chunks[0].line_start, 2);
        assert_eq!(chunks[0].heading_path, vec!["A".to_string()]);
        assert!(
            chunks[1].text.starts_with("A\n\n## B"),
            "got: {:?}",
            chunks[1].text
        );
        assert_eq!(
            chunks[1].heading_path,
            vec!["A".to_string(), "B".to_string()]
        );
    }

    #[test]
    fn fenced_and_hashtag_lines_do_not_split_chunks() {
        // Regressions (#286): `#tag` and a `~~~` line inside a ``` fence.
        let body = "# A\n\n#tag not heading\n\n```\n~~~\n# not\n~~~\n```\n\n## B\n\nb";
        let chunks = chunk_markdown(body, None, &cfg(1000, Some(1), false, false));
        let paths: Vec<_> = chunks.iter().map(|c| c.heading_path.clone()).collect();
        assert_eq!(
            paths,
            vec![
                vec!["A".to_string()],
                vec!["A".to_string(), "B".to_string()]
            ]
        );
        assert_eq!((chunks[1].line_start, chunks[1].line_end), (11, 13));
    }

    #[test]
    fn sections_cover_chunks_across_shapes_and_sizes() {
        let para = |n: usize| format!("Para{n} {}", "lorem ipsum ".repeat(n * 7));
        let body = format!(
            "Preamble {p1}\n\n# One\n\n{p2}\n\n## Two\n\n{p3}\n\n```\n# fenced\n```\n\n\
             ### Three\n\n{p9}\n\n## Two\n\n{p1}\n\nSetext\n------\n\n{p5}\n\n# One\n\n{p2}\n",
            p1 = para(1),
            p2 = para(2),
            p3 = para(3),
            p5 = para(5),
            p9 = para(9),
        );
        for max in [120, 300, 700, 2000] {
            for target in [Some(1), Some(max / 2), None] {
                for prepend in [false, true] {
                    let chunks = chunk_markdown(&body, None, &cfg(max, target, false, prepend));
                    assert!(!chunks.is_empty());
                    assert_sections_cover_chunks(&chunks);
                    for w in chunks.windows(2) {
                        assert!(w[0].line_start <= w[1].line_start);
                    }
                }
            }
        }
    }

    // ── Chunk::heading_path / heading_level (#286) ──────────────────────────

    #[test]
    fn chunk_with_no_heading_at_all_has_empty_heading_path_and_zero_level() {
        let chunks = chunk_markdown("Hello world", None, &cfg(1000, None, false, false));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].heading_path.is_empty());
        assert_eq!(chunks[0].heading_level, 0);
    }

    #[test]
    fn heading_path_and_level_on_a_single_flat_chunk() {
        // A lone "## Heading" section with no ancestors: heading_path is always
        // the FULL path including the section's own heading — unlike the
        // rendered breadcrumb (which excludes it here to avoid duplicating the
        // chunk's own first line), `Chunk::heading_path` is computed the same
        // way regardless of `prepend_heading_path`.
        let chunks = chunk_markdown(
            "## Heading\n\nSome body content.",
            None,
            &cfg(1000, None, false, false),
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading_path, vec!["Heading".to_string()]);
        assert_eq!(chunks[0].heading_level, 2);
    }

    #[test]
    fn heading_path_and_level_across_nesting_levels() {
        // Same body/config as `heading_path_prepended_across_nesting_levels`,
        // but pinning the structured `heading_path`/`heading_level` fields
        // rather than the rendered text breadcrumb.
        let body = "# ares\n\n## Hardware\n\n### GPU Backends\n\nROCm and Vulkan notes.";
        let chunks = chunk_markdown(body, None, &cfg(1500, Some(1), false, false));
        assert_eq!(chunks.len(), 3);

        assert_eq!(chunks[0].heading_path, vec!["ares".to_string()]);
        assert_eq!(chunks[0].heading_level, 1);

        assert_eq!(
            chunks[1].heading_path,
            vec!["ares".to_string(), "Hardware".to_string()]
        );
        assert_eq!(chunks[1].heading_level, 2);

        assert_eq!(
            chunks[2].heading_path,
            vec![
                "ares".to_string(),
                "Hardware".to_string(),
                "GPU Backends".to_string()
            ]
        );
        assert_eq!(chunks[2].heading_level, 3);
    }

    #[test]
    fn heading_path_and_level_on_oversized_section_fragments() {
        // Same scenario as
        // `oversized_section_continuation_fragment_gets_full_heading_path`: the
        // first fragment AND every later continuation fragment carry the full
        // attributed path (unlike the rendered breadcrumb, which differs
        // between them), since `Chunk::heading_path` is always the attributed
        // section's full path.
        let filler = "Lorem ipsum dolor sit amet consectetur. ".repeat(80);
        let body = format!("# Root\n\n## Section\n\n### Big\n\n{filler}");
        let chunks = chunk_markdown(&body, None, &cfg(1000, Some(800), false, false));
        assert!(chunks.len() >= 3);

        let expected_path = vec!["Root".to_string(), "Section".to_string(), "Big".to_string()];
        assert_eq!(chunks[1].heading_path, expected_path);
        assert_eq!(chunks[1].heading_level, 3);
        assert_eq!(chunks[2].heading_path, expected_path);
        assert_eq!(chunks[2].heading_level, 3);
    }

    #[test]
    fn heading_path_and_level_narrow_on_a_merged_chunk() {
        // Same scenario as `merged_chunk_breadcrumb_narrows_to_shared_ancestor`:
        // the merged chunk's `heading_path`/`heading_level` narrow to "Root"
        // (level 1), the deepest heading both merged sections actually share —
        // not "### Blinded" (level 3), which only the FIRST merged section sat
        // under.
        let filler = "Word ".repeat(60);
        let body = format!(
            "# Root\n\n## Conditions\n\n{filler}\n\n### Blinded\n\nBlinded is bad.\n\n## Actions\n\nActions cost."
        );
        let chunks = chunk_markdown(&body, None, &cfg(1500, Some(300), false, false));
        assert_eq!(chunks.len(), 3);
        let merged = &chunks[2];
        assert_eq!(merged.heading_path, vec!["Root".to_string()]);
        assert_eq!(merged.heading_level, 1);
    }

    // ── chunking.heading_metadata: no merging across headings (#286) ───────

    fn cfg_isolated(
        max: usize,
        target: Option<usize>,
        prepend_heading_path: bool,
    ) -> ChunkingConfig {
        ChunkingConfig {
            heading_metadata: true,
            ..cfg(max, target, false, prepend_heading_path)
        }
    }

    fn s(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|p| p.to_string()).collect()
    }

    /// `# Chapter` > `## Level 1` > 40 short `###` feats.
    fn sibling_feats_body() -> String {
        let mut body = String::from("# Chapter\n\nChapter intro.\n\n## Level 1\n\n");
        for i in 0..40 {
            body.push_str(&format!(
                "### Feat {i}\n\n**Traits** General\n\nYou do thing number {i}.\n\n"
            ));
        }
        body
    }

    #[test]
    fn heading_metadata_on_never_merges_sibling_sections() {
        let body = sibling_feats_body();
        let tree = HeadingTree::parse(&body);
        for (max, target) in [
            (1500, None),
            (1500, Some(1000)),
            (300, Some(200)),
            (60, Some(1)),
        ] {
            for prepend in [false, true] {
                let chunks = chunk_markdown(&body, None, &cfg_isolated(max, target, prepend));
                crate::heading::tests::assert_chunks_stay_within_one_section(&body, &chunks);
                for i in 0..40 {
                    let name = format!("Feat {i}");
                    let feat = tree
                        .headings()
                        .iter()
                        .find(|h| h.text == name)
                        .expect("feat heading");
                    let own: Vec<&Chunk> = chunks
                        .iter()
                        .filter(|c| c.heading_path.last() == Some(&name))
                        .collect();
                    assert!(!own.is_empty(), "{name} has no chunk of its own");
                    for c in own {
                        assert_eq!(c.heading_path, s(&["Chapter", "Level 1", &name]));
                        assert_eq!(c.heading_level, 3);
                        assert_eq!(
                            (c.section_line_start, c.section_line_end),
                            (feat.line, feat.subtree_end)
                        );
                    }
                }
            }
        }

        // At the default sizes every section fits one chunk: the chapter (with
        // its intro), the heading-only `## Level 1`, and one chunk per feat.
        let chunks = chunk_markdown(&body, None, &cfg_isolated(1500, None, true));
        assert_eq!(chunks.len(), 42);
        assert_eq!(chunks[0].heading_path, s(&["Chapter"]));
        assert_eq!(chunks[1].heading_path, s(&["Chapter", "Level 1"]));
        assert_eq!(
            chunks[2].text,
            "Chapter > Level 1\n\n### Feat 0\n\n**Traits** General\n\nYou do thing number 0.\n"
        );
    }

    #[test]
    fn heading_metadata_off_still_merges_sibling_sections() {
        let body = sibling_feats_body();
        let chunks = chunk_markdown(&body, None, &cfg(1500, None, false, true));
        assert!(chunks.len() < 10, "got {} chunks", chunks.len());
        assert_sections_cover_chunks(&chunks);
        assert!(
            chunks.iter().any(|c| c.heading_path.len() < 3),
            "merged feats are attributed to an ancestor"
        );
    }

    #[test]
    fn heading_only_sections_become_their_own_chunks_when_isolated() {
        // A heading directly followed by its first sub-heading has no body.
        // With heading_metadata on it is its own chunk holding the heading
        // line, not folded into the child (which would put the child's heading
        // inside a chunk attributed to the parent).
        let body = "# Chapter\n\n## Level 1\n\n### Feat\n\nFeat body.";
        let chunks = chunk_markdown(body, None, &cfg_isolated(1500, None, true));
        let summary: Vec<_> = chunks
            .iter()
            .map(|c| {
                (
                    c.text.as_str(),
                    c.heading_path.clone(),
                    (c.line_start, c.line_end),
                    (c.section_line_start, c.section_line_end),
                )
            })
            .collect();
        assert_eq!(
            summary,
            vec![
                ("# Chapter\n", s(&["Chapter"]), (1, 2), (1, 7)),
                (
                    "Chapter\n\n## Level 1\n",
                    s(&["Chapter", "Level 1"]),
                    (3, 4),
                    (3, 7)
                ),
                (
                    "Chapter > Level 1\n\n### Feat\n\nFeat body.",
                    s(&["Chapter", "Level 1", "Feat"]),
                    (5, 7),
                    (5, 7)
                ),
            ]
        );
        // Off: the three tiny sections merge into one chunk attributed to the
        // chapter.
        let merged = chunk_markdown(body, None, &cfg(1500, None, false, true));
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].heading_path, s(&["Chapter"]));
    }

    #[test]
    fn isolated_preamble_is_never_merged_into_the_first_section() {
        let body = "Intro text.\n\n# A\n\nA body.\n\n## B\n\nB body.";
        let chunks = chunk_markdown(body, None, &cfg_isolated(1000, None, false));
        crate::heading::tests::assert_chunks_stay_within_one_section(body, &chunks);
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].heading_path.is_empty());
        assert_eq!(
            (chunks[0].section_line_start, chunks[0].section_line_end),
            (1, 2)
        );
    }

    #[test]
    fn isolated_trailing_fragment_does_not_join_the_previous_section() {
        // A tiny `max_chunk_size` makes every section oversized, so a small
        // trailing fragment would otherwise be appended to whatever chunk came
        // last — the previous section's — when none of this section's own
        // fragments were pushed.
        for max in [1, 7, 11, 20, 40] {
            let body = "# A\n\nalpha beta gamma\n\n## B\n\nx\n\n## C\n\ndelta epsilon";
            for prepend in [false, true] {
                let chunks = chunk_markdown(body, None, &cfg_isolated(max, Some(1), prepend));
                crate::heading::tests::assert_chunks_stay_within_one_section(body, &chunks);
            }
        }
    }

    #[test]
    fn tiny_max_chunk_size_never_prepends_an_empty_breadcrumb() {
        // #286: below 12, the breadcrumb budget is 0 characters. A nested
        // continuation chunk must not start with an empty breadcrumb plus
        // "\n\n".
        let body = "# Root\n\n## Section\n\nsome body words here and more";
        for max in 1..12 {
            for heading_metadata in [false, true] {
                let config = ChunkingConfig {
                    heading_metadata,
                    ..cfg(max, None, false, true)
                };
                for c in chunk_markdown(body, None, &config) {
                    assert!(
                        !c.text.starts_with('\n'),
                        "max {max}: chunk {} is {:?}",
                        c.index,
                        c.text
                    );
                }
            }
        }
        assert_eq!(format_heading_path(&s(&["A", "B"]), 0), None);
        assert_eq!(
            format_heading_path(&s(&["A", "B"]), 3),
            Some("A >".to_string())
        );
    }
}
