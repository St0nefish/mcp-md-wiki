//! Document similarity: the dense-only near-duplicate query behind the write
//! path's dedup gate, and the one ranking rule every "nearest other documents"
//! consumer shares.
//!
//! Two pieces, deliberately separate:
//!
//! - [`find_near_duplicates`] embeds a candidate text and asks Qdrant for the
//!   nearest documents, optionally confined to a directory. It needs an embedder
//!   and a store, so it is only for callers that have no stored vector to start
//!   from (a document that does not exist yet).
//! - [`rank_neighbours`] is pure: one hit per document, best score, never the
//!   document itself, nothing under a floor, best first, at most `k`. The write
//!   path's `related` list, the contradiction job and the semantic-edge refresh
//!   all take their raw hits from different places (a text query, a stored point's
//!   neighbours) and rank them here, so the three cannot drift apart on ties,
//!   self-exclusion or the floor.

use std::path::Path;

use qdrant_client::qdrant::Condition;

use crate::embed::QueryEmbedder;
use crate::policy::Reach;
use crate::qdrant::{PATH_ANCESTORS_KEY, RetrievalStore};
use crate::retrieval::{RetrievalDeps, SearchFilters};

/// Maximum number of characters from the new document's content used to build
/// the dedup query text. Keeps the embedding request within typical token limits
/// (most embedding models cap at ~512–8192 tokens; 2000 chars ≈ 400–500 tokens).
pub(crate) const DEDUP_QUERY_CHAR_LIMIT: usize = 2000;

/// Build the dedup query text on the same textual basis the indexer uses, then
/// truncate it to `DEDUP_QUERY_CHAR_LIMIT`.
///
/// This must stay aligned with `chunk::chunk_markdown`: every indexed chunk is
/// prefixed with its document's own `description` when
/// `chunking.prepend_description` is set, so a dedup query built without that
/// prefix would be compared against a different textual basis than the
/// candidates it is scored against.
///
/// It deliberately does NOT reproduce `chunking.prepend_heading_path`'s
/// breadcrumb, and that is safe for a non-obvious structural reason worth
/// stating: a chunk's breadcrumb is seeded from the section that starts it and
/// merging further pieces in can only narrow it (to a common ancestor, see
/// `chunk::chunk_markdown`). A document's FIRST chunk is always seeded from its
/// first section — the preamble, which has no breadcrumb, or the first heading,
/// which has no ancestors because no heading precedes it — so chunk 0 never
/// carries a breadcrumb, whatever the document's heading structure. Since a create-path dedup query is doc-start text scored
/// against the corpus, matching chunk 0's basis is what matters.
///
/// If a future chunking change breaks that invariant — anything that can give
/// a document's first chunk a prefix this function does not build — the dedup
/// gate silently starts comparing unlike text and lets near-duplicates through
/// with no error. `mcp::tests::write::build_dedup_query_matches_chunk_prepend_format`
/// is the pin; keep it honest rather than adjusting it to match a regression.
pub(crate) fn build_dedup_query(
    body: &str,
    description: Option<&str>,
    prepend_description: bool,
) -> String {
    let assembled = match (prepend_description, description) {
        (true, Some(desc)) => format!("{}\n\n{}", desc, body),
        _ => body.to_string(),
    };
    assembled.chars().take(DEDUP_QUERY_CHAR_LIMIT).collect()
}

/// Adapter that routes `embed_query` to the wrapped embedder's `embed_document`,
/// so a text compared against indexed chunks is embedded like those chunks.
pub(crate) struct DocumentSideEmbedder<'a, E: QueryEmbedder>(pub(crate) &'a E);

impl<E: QueryEmbedder> QueryEmbedder for DocumentSideEmbedder<'_, E> {
    async fn embed_query(&self, q: &str) -> anyhow::Result<Vec<f32>> {
        self.0.embed_document(q).await
    }
}

/// Search options for [`find_near_duplicates`].
///
/// Deliberately pinned to dense-only rather than inheriting `search.hybrid`, so
/// the returned score is a cosine similarity comparable to
/// `write.dedup_threshold`. Hybrid RRF scores top out around 0.03 — against a
/// cosine threshold like the 0.80 default the gate could never fire — and a
/// cross-encoder relevance score is not a similarity at all, so reranking is
/// also kept out of this path (see the `reranker: None` in
/// [`find_near_duplicates`]).
#[cfg(test)]
pub(crate) fn dedup_search_opts() -> crate::retrieval::SearchOptions {
    dedup_search_opts_with_limit(1)
}

/// [`dedup_search_opts`] asking for the `limit` nearest chunks, one per document
/// when more than one is wanted, so the extras are distinct neighbours (#356).
/// The top hit is the same chunk `dedup_search_opts` returns.
pub(crate) fn dedup_search_opts_with_limit(limit: usize) -> crate::retrieval::SearchOptions {
    crate::retrieval::SearchOptions {
        limit: limit.max(1) as u64,
        min_score: None,
        hybrid: false,
        // Unused in the dense-only path, which performs no RRF fusion.
        rrf_candidates: 0,
        // Same reasoning as `hybrid` above: phrase matching adds a third fused
        // arm, which this dense-only comparison has no use for.
        phrase: false,
        explain: false,
        modified_after: None,
        modified_before: None,
        path_filter: None,
        rerank_candidate_limit: None,
        // The dedup gate wants the single closest existing chunk, full stop — not
        // a diversified page of results (limit: 1 above makes a per-document cap
        // moot anyway, but this keeps intent explicit rather than accidental).
        diversity_max_per_document: (limit > 1).then_some(1),
    }
}

/// An existing document close to a query, with its dense cosine similarity.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Neighbour {
    pub path: String,
    pub score: f32,
}

/// The Qdrant condition confining a search to what `reach` covers, `None` when it
/// covers everything.
///
/// A plain match-any on `path_ancestors`, which holds every ancestor directory and
/// the document's own path; for a reach with nothing in it, the unsatisfiable
/// condition (never an empty match-any, whose meaning is not worth relying on). It
/// is deliberately NOT `retrieval::path_filter_condition`: that one ORs
/// `is_empty(path_ancestors)` so a document indexed before the field existed still
/// passes, which makes it a relevance filter, not a boundary. A query that must
/// stay inside a directory can have no such escape.
pub(crate) fn reach_condition(reach: &Reach) -> Option<Condition> {
    match reach {
        Reach::All => None,
        Reach::Prefixes(dirs) if dirs.is_empty() => {
            Some(crate::qdrant::unsatisfiable(PATH_ANCESTORS_KEY))
        }
        Reach::Prefixes(dirs) => Some(Condition::matches(PATH_ANCESTORS_KEY, dirs.clone())),
    }
}

/// The nearest documents to `query_text`, nearest first, one per document, at
/// most `limit`, among the documents `scope` covers.
///
/// `query_text` is embedded on the document side (it is compared against indexed
/// chunks, so it takes `embedding.document_prefix`, not the query prefix), the
/// search is dense-only and the reranker is detached — see
/// [`dedup_search_opts`]. Scores are raw cosine similarities: applying a
/// threshold or a floor is the caller's job (the dedup gate and the `related`
/// list use different ones). Paths are KB-relative, taken through
/// `canonical_data_path`.
///
/// An `Err` means the search itself failed (embedding or store); callers on the write path treat that
/// as "proceed", never as "no duplicates found" they can rely on.
pub(crate) async fn find_near_duplicates<E: QueryEmbedder, Q: RetrievalStore>(
    deps: &RetrievalDeps<'_, E, Q>,
    canonical_data_path: &Path,
    query_text: &str,
    scope: &Reach,
    limit: usize,
) -> Result<Vec<Neighbour>, crate::retrieval::SearchError> {
    let document_side = DocumentSideEmbedder(deps.embed_client);
    let dedup_deps = RetrievalDeps {
        embed_client: &document_side,
        qdrant: deps.qdrant,
        collection: deps.collection,
        data_path: deps.data_path,
        include_patterns: deps.include_patterns,
        reranker: None,
    };
    let filters = SearchFilters {
        conditions: reach_condition(scope).into_iter().collect(),
    };
    let results = crate::retrieval::search(
        &dedup_deps,
        query_text,
        &filters,
        &dedup_search_opts_with_limit(limit),
    )
    .await?;
    let hits = results.into_iter().map(|r| {
        let path = r
            .payload
            .get("file_path")
            .and_then(|v| v.as_str())
            .map(|p| crate::retrieval::relative_to_data(p, canonical_data_path))
            .unwrap_or_default();
        (path, r.score)
    });
    Ok(rank_neighbours(hits, "", f32::NEG_INFINITY, limit))
}

/// Rank raw `(path, score)` hits into distinct neighbouring documents.
///
/// Each document once, at its best score; never `own_path` (compared with a
/// leading `/` ignored on both sides, since a caller may hold either spelling);
/// nothing scoring under `floor`; an empty path is dropped (a payload with no
/// `file_path`); best first, ties by path so the order is deterministic; at most
/// `k`. A `k` of `0` yields none.
pub(crate) fn rank_neighbours<I, S>(hits: I, own_path: &str, floor: f32, k: usize) -> Vec<Neighbour>
where
    I: IntoIterator<Item = (S, f32)>,
    S: AsRef<str>,
{
    if k == 0 {
        return Vec::new();
    }
    let own = crate::retrieval::kb_root_relative(own_path);
    let mut best: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    for (path, score) in hits {
        let path = path.as_ref();
        if path.is_empty() || score < floor || crate::retrieval::kb_root_relative(path) == own {
            continue;
        }
        best.entry(path.to_string())
            .and_modify(|s| *s = s.max(score))
            .or_insert(score);
    }
    let mut ranked: Vec<Neighbour> = best
        .into_iter()
        .map(|(path, score)| Neighbour { path, score })
        .collect();
    ranked.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.path.cmp(&b.path))
    });
    ranked.truncate(k);
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hits(items: &[(&str, f32)]) -> Vec<(String, f32)> {
        items.iter().map(|(p, s)| (p.to_string(), *s)).collect()
    }

    #[test]
    fn rank_neighbours_keeps_each_document_once_at_its_best_score() {
        let got = rank_neighbours(
            hits(&[("a.md", 0.5), ("b.md", 0.7), ("a.md", 0.9), ("b.md", 0.1)]),
            "z.md",
            0.0,
            10,
        );
        assert_eq!(
            got,
            vec![
                Neighbour {
                    path: "a.md".into(),
                    score: 0.9
                },
                Neighbour {
                    path: "b.md".into(),
                    score: 0.7
                },
            ]
        );
    }

    #[test]
    fn rank_neighbours_drops_self_under_either_spelling_and_the_floor() {
        let got = rank_neighbours(
            hits(&[
                ("docs/me.md", 0.99),
                ("/docs/me.md", 0.98),
                ("docs/low.md", 0.3),
                ("docs/ok.md", 0.6),
                ("", 0.9),
            ]),
            "docs/me.md",
            0.5,
            10,
        );
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].path, "docs/ok.md");
        // The own path given with a leading `/` excludes the bare spelling too.
        let got = rank_neighbours(hits(&[("docs/me.md", 0.9)]), "/docs/me.md", 0.0, 10);
        assert!(got.is_empty());
    }

    #[test]
    fn rank_neighbours_orders_ties_by_path_and_truncates() {
        let got = rank_neighbours(
            hits(&[("c.md", 0.5), ("a.md", 0.5), ("b.md", 0.5), ("d.md", 0.4)]),
            "",
            0.0,
            3,
        );
        let paths: Vec<&str> = got.iter().map(|n| n.path.as_str()).collect();
        assert_eq!(paths, vec!["a.md", "b.md", "c.md"]);
        assert!(rank_neighbours(hits(&[("a.md", 0.9)]), "", 0.0, 0).is_empty());
        assert!(rank_neighbours(Vec::<(String, f32)>::new(), "", 0.0, 5).is_empty());
    }

    #[test]
    fn reach_condition_is_a_plain_match_with_no_empty_escape() {
        assert_eq!(reach_condition(&Reach::All), None);
        assert_eq!(
            reach_condition(&Reach::Prefixes(vec![
                "memory/general".into(),
                "notes".into()
            ])),
            Some(Condition::matches(
                PATH_ANCESTORS_KEY,
                vec!["memory/general".to_string(), "notes".to_string()]
            ))
        );
        // Nothing to read: unsatisfiable, never a match on an empty list.
        assert_eq!(
            reach_condition(&Reach::Prefixes(Vec::new())),
            Some(crate::qdrant::unsatisfiable(PATH_ANCESTORS_KEY))
        );
    }
}
