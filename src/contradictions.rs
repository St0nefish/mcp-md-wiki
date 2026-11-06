//! Report-only contradiction detection (#355).
//!
//! A run finds documents whose content changed since the job last analysed them,
//! asks the vector store for each one's nearest neighbours using the vector already
//! stored for its first chunk (no embedding call), keeps the close ones as candidate
//! pairs, and — when an LLM [`Judge`] is configured — has it decide whether each pair
//! makes conflicting claims. Results land in the state DB (`contradiction_pairs`);
//! nothing is written to a document. [`pairs_report`] reads them back for the
//! `contradictions --list` subcommand and `GET /status/contradictions`.
//!
//! # Idempotence
//!
//! A pair row remembers the content hash of both documents when it was evaluated. A
//! later run skips a pair whose stored hashes still match both documents, so a second
//! run over an unchanged corpus evaluates nothing, and a changed document re-evaluates
//! only the pairs it is part of. A pair is also skipped when it already holds a final
//! verdict, whichever judge model gave it. The exceptions are `candidate` rows (close
//! pairs with no verdict) once a judge is configured: those are retried, whether the
//! row came from a run with no judge or from a judge call that failed.
//!
//! # Carry-over
//!
//! A run evaluates at most `max_pairs_per_run` pairs, in path order. A document is
//! recorded as analysed only when every pair it needs was settled, so the pairs left
//! over are found again next run (the ones already stored are skipped), and a run that
//! is cancelled between pairs loses nothing.
//!
//! # Limits
//!
//! Only the first chunk of each document is used as the query vector, the same
//! compromise the web UI's semantic edges make, so a conflict confined to a later
//! section can be missed. Documents are sent to the judge as stored (frontmatter
//! included), cut to a byte budget.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use qdrant_client::qdrant::{Condition, Filter};
use tracing::{debug, info, warn};

use crate::config::ContradictionsConfig;
use crate::ingest::{KbEntry, NeighborStore, inspect_kb_path, make_point_id};
use crate::judge::{
    Confidence, ConflictingStatements, Judge, PairInput, truncate_on_char_boundary,
};
use crate::state::{ContradictionPair, StateDb};

/// `verdict` of a pair the judge found conflicting.
pub const VERDICT_CONFLICT: &str = "conflict";
/// `verdict` of a pair the judge found compatible.
pub const VERDICT_NO_CONFLICT: &str = "no_conflict";
/// `verdict` of a close pair with no judge verdict: no judge is configured, or the
/// judge call failed (then the row carries an `error`).
pub const VERDICT_CANDIDATE: &str = "candidate";

/// Candidates are over-fetched by this factor before being collapsed to one hit per
/// document, because the neighbour query ranks chunks and several can belong to one
/// document.
const OVERFETCH_FACTOR: u64 = 3;

/// A run gives up after this many judge calls fail in a row: a provider that is down
/// would otherwise cost the full retry budget on every one of the run's pairs.
const MAX_CONSECUTIVE_JUDGE_FAILURES: usize = 5;

/// Longest judge error stored on a pair row.
const MAX_STORED_ERROR_BYTES: usize = 500;

/// Default number of pairs `contradictions --list` / `GET /status/contradictions` show.
pub const DEFAULT_LIST_LIMIT: usize = 200;
/// Largest number of pairs a caller may ask to list.
pub const MAX_LIST_LIMIT: usize = 5000;

/// One run's tuning, read from `contradictions:` fresh for every run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunParams {
    pub top_k: u64,
    pub min_similarity: f32,
    pub max_pairs_per_run: usize,
    /// Per-document byte budget for a judge request. Passed in rather than read from
    /// `cfg`: the judge is built once at startup, and its budget goes with it.
    pub max_document_bytes: usize,
}

impl RunParams {
    pub fn new(cfg: &ContradictionsConfig, max_document_bytes: usize) -> Self {
        Self {
            top_k: cfg.top_k,
            min_similarity: cfg.min_similarity,
            max_pairs_per_run: cfg.max_pairs_per_run,
            max_document_bytes,
        }
    }
}

/// Everything [`run_once`] reads and writes. Bundled so tests (and the CLI and the
/// scheduled job) build the same thing against a fake neighbour store and judge.
pub struct RunDeps<'a, N> {
    pub state: &'a StateDb,
    pub neighbors: &'a N,
    pub collection: &'a str,
    /// `None`: record close pairs as `candidate` and send nothing anywhere.
    pub judge: Option<&'a dyn Judge>,
    /// The knowledge-base root, for reading the documents of a pair to be judged.
    pub data_path: &'a Path,
    pub params: RunParams,
}

/// What one run did. `pairs_evaluated` counts rows written; of those, `conflicts` are
/// judged conflicting, `candidates` are recorded without a verdict (no judge), and
/// `errors` are judge or read failures (stored as `candidate` rows with an `error`,
/// retried next run).
#[derive(Debug, Default, Clone, PartialEq, serde::Serialize)]
pub struct RunSummary {
    /// Changed documents whose neighbours were looked up this run.
    pub documents_scanned: usize,
    pub pairs_evaluated: usize,
    pub conflicts: usize,
    pub candidates: usize,
    pub errors: usize,
    /// Pairs found but left for the next run: the per-run cap was reached or the run
    /// was cancelled.
    pub deferred_pairs: usize,
    /// The run stopped early because it was cancelled.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub cancelled: bool,
}

/// Serialises runs that share a process. Held by the scheduled job; a second caller
/// finds it busy and skips rather than queueing behind a long judge run.
pub type RunLock = Arc<tokio::sync::Mutex<()>>;

pub fn new_run_lock() -> RunLock {
    Arc::new(tokio::sync::Mutex::new(()))
}

/// [`run_once`] under `lock`: `Ok(None)` (and a log line) when another run holds it.
pub async fn run_once_exclusive<N: NeighborStore>(
    lock: &RunLock,
    deps: &RunDeps<'_, N>,
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<Option<RunSummary>> {
    let Ok(_guard) = lock.try_lock() else {
        info!("Contradiction run skipped: another run is in progress");
        return Ok(None);
    };
    run_once(deps, cancelled).await.map(Some)
}

/// A pair waiting to be evaluated, `a < b`.
struct Work {
    hash_a: String,
    hash_b: String,
    similarity: f64,
}

type PairKey = (String, String);

fn pair_key(x: &str, y: &str) -> PairKey {
    if x <= y {
        (x.to_string(), y.to_string())
    } else {
        (y.to_string(), x.to_string())
    }
}

/// Whether a stored row already settles the pair for the current document hashes.
fn pair_is_settled(
    stored: Option<&ContradictionPair>,
    hash_a: &str,
    hash_b: &str,
    judge: bool,
) -> bool {
    let Some(row) = stored else {
        return false;
    };
    if row.hash_a != hash_a || row.hash_b != hash_b {
        return false;
    }
    match row.verdict.as_str() {
        VERDICT_CANDIDATE => !judge,
        _ => true,
    }
}

/// Run one pass. `cancelled` is polled between documents and between pairs, so a
/// shutdown never waits on more than the judge call in flight.
pub async fn run_once<N: NeighborStore>(
    deps: &RunDeps<'_, N>,
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<RunSummary> {
    let mut summary = RunSummary::default();
    let params = deps.params;

    let pruned = deps.state.prune_contradiction_data().await?;
    if pruned > 0 {
        debug!(
            rows = pruned,
            "Pruned contradiction rows for removed documents"
        );
    }

    let hashes = deps.state.list_document_hashes().await?;
    let analyzed = deps.state.contradiction_analyzed_hashes().await?;
    let stored: HashMap<PairKey, ContradictionPair> = deps
        .state
        .all_contradiction_pairs()
        .await?
        .into_iter()
        .map(|p| ((p.path_a.clone(), p.path_b.clone()), p))
        .collect();

    let mut changed: Vec<&String> = hashes
        .iter()
        .filter(|(path, hash)| analyzed.get(*path) != Some(*hash))
        .map(|(path, _)| path)
        .collect();
    changed.sort();

    // Phase 1: pairs a changed document needs evaluated. `needs` lists, per scanned
    // document, the pairs that must all settle before it counts as analysed.
    let mut work: BTreeMap<PairKey, Work> = BTreeMap::new();
    let mut needs: Vec<(&String, Vec<PairKey>)> = Vec::new();
    let overfetch = params.top_k.saturating_mul(OVERFETCH_FACTOR);

    for path in changed {
        if cancelled() {
            summary.cancelled = true;
            break;
        }
        let point_id = make_point_id(path, 0);
        let exclude_self = Filter::must_not([Condition::matches("file_path", path.clone())]);
        let hits = match deps
            .neighbors
            .recommend_by_point_id(deps.collection, &point_id, overfetch, Some(exclude_self))
            .await
        {
            Ok(hits) => hits,
            Err(e) => {
                warn!(file = %path, "Neighbour lookup failed (will retry next run): {e:#}");
                summary.errors += 1;
                continue;
            }
        };
        summary.documents_scanned += 1;

        // One hit per neighbouring document, best score; drop the document itself,
        // anything under the floor and anything the index no longer knows.
        let mut best: HashMap<&str, f32> = HashMap::new();
        for hit in &hits {
            let Some(target) = hit.payload.get("file_path").and_then(|v| v.as_str()) else {
                continue;
            };
            if target == path.as_str()
                || hit.score < params.min_similarity
                || !hashes.contains_key(target)
            {
                continue;
            }
            best.entry(target)
                .and_modify(|s| *s = s.max(hit.score))
                .or_insert(hit.score);
        }
        let mut ranked: Vec<(&str, f32)> = best.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        ranked.truncate(params.top_k as usize);

        let mut required = Vec::new();
        for (neighbour, score) in ranked {
            let key = pair_key(path, neighbour);
            let hash_a = &hashes[&key.0];
            let hash_b = &hashes[&key.1];
            if pair_is_settled(stored.get(&key), hash_a, hash_b, deps.judge.is_some()) {
                continue;
            }
            let entry = work.entry(key.clone()).or_insert_with(|| Work {
                hash_a: hash_a.clone(),
                hash_b: hash_b.clone(),
                similarity: f64::from(score),
            });
            entry.similarity = entry.similarity.max(f64::from(score));
            required.push(key);
        }
        needs.push((path, required));
    }

    // Phase 2: with a judge now configured, pairs recorded without a verdict (no judge
    // then, or a failed call) are retried even though neither document changed.
    if deps.judge.is_some() && !summary.cancelled {
        for (key, row) in &stored {
            if row.verdict != VERDICT_CANDIDATE || work.contains_key(key) {
                continue;
            }
            let (Some(hash_a), Some(hash_b)) = (hashes.get(&key.0), hashes.get(&key.1)) else {
                continue;
            };
            if *hash_a != row.hash_a || *hash_b != row.hash_b {
                continue;
            }
            work.insert(
                key.clone(),
                Work {
                    hash_a: hash_a.clone(),
                    hash_b: hash_b.clone(),
                    similarity: row.similarity,
                },
            );
        }
    }

    // Phase 3: evaluate, in path order, up to the cap.
    let mut settled: HashSet<PairKey> = HashSet::new();
    let mut attempted = 0usize;
    let mut skipped = 0usize;
    let mut consecutive_failures = 0usize;
    for (key, item) in &work {
        if attempted >= params.max_pairs_per_run {
            break;
        }
        if cancelled() {
            summary.cancelled = true;
            break;
        }
        match evaluate_pair(deps, key, item, &mut summary).await? {
            outcome @ (Evaluated::Stored | Evaluated::JudgeFailed) => {
                attempted += 1;
                settled.insert(key.clone());
                if outcome == Evaluated::JudgeFailed {
                    consecutive_failures += 1;
                    if consecutive_failures >= MAX_CONSECUTIVE_JUDGE_FAILURES {
                        warn!(
                            "Stopping the contradiction run after {consecutive_failures} judge \
                             failures in a row; the rest waits for the next run"
                        );
                        break;
                    }
                } else {
                    consecutive_failures = 0;
                }
            }
            Evaluated::Skipped => {
                skipped += 1;
            }
        }
    }
    summary.pairs_evaluated = attempted;
    summary.deferred_pairs = work.len() - settled.len() - skipped;

    // Phase 4: a scanned document whose every needed pair settled is analysed.
    let done: Vec<(String, String)> = needs
        .iter()
        .filter(|(_, required)| required.iter().all(|k| settled.contains(k)))
        .map(|(path, _)| ((*path).clone(), hashes[*path].clone()))
        .collect();
    deps.state.mark_documents_analyzed(&done).await?;

    info!(
        scanned = summary.documents_scanned,
        evaluated = summary.pairs_evaluated,
        conflicts = summary.conflicts,
        candidates = summary.candidates,
        errors = summary.errors,
        deferred = summary.deferred_pairs,
        cancelled = summary.cancelled,
        "Contradiction run finished"
    );
    Ok(summary)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Evaluated {
    /// A row was written.
    Stored,
    /// A `candidate` row with an `error` was written: the judge call failed.
    JudgeFailed,
    /// A document could not be read; nothing was written.
    Skipped,
}

async fn evaluate_pair<N: NeighborStore>(
    deps: &RunDeps<'_, N>,
    key: &PairKey,
    item: &Work,
    summary: &mut RunSummary,
) -> Result<Evaluated> {
    let (path_a, path_b) = (&key.0, &key.1);
    let mut failed = false;
    let mut row = ContradictionPair {
        path_a: path_a.clone(),
        path_b: path_b.clone(),
        hash_a: item.hash_a.clone(),
        hash_b: item.hash_b.clone(),
        similarity: item.similarity,
        verdict: VERDICT_CANDIDATE.to_string(),
        confidence: None,
        statements: None,
        judge_model: None,
        error: None,
        judged_at: String::new(),
    };

    if let Some(judge) = deps.judge {
        row.judge_model = Some(judge.model().to_string());
        let budget = deps.params.max_document_bytes;
        let text_a = match read_document(deps.data_path, path_a).await {
            Ok(Some(t)) => t,
            Ok(None) => {
                warn!(file = %path_a, "Document vanished before it could be judged; skipping pair");
                return Ok(Evaluated::Skipped);
            }
            Err(e) => {
                warn!(file = %path_a, "Cannot read document for judging: {e:#}");
                summary.errors += 1;
                return Ok(Evaluated::Skipped);
            }
        };
        let text_b = match read_document(deps.data_path, path_b).await {
            Ok(Some(t)) => t,
            Ok(None) => {
                warn!(file = %path_b, "Document vanished before it could be judged; skipping pair");
                return Ok(Evaluated::Skipped);
            }
            Err(e) => {
                warn!(file = %path_b, "Cannot read document for judging: {e:#}");
                summary.errors += 1;
                return Ok(Evaluated::Skipped);
            }
        };
        let input = PairInput {
            path_a,
            text_a: truncate_on_char_boundary(&text_a, budget),
            path_b,
            text_b: truncate_on_char_boundary(&text_b, budget),
        };
        match judge.judge(&input).await {
            Ok(verdict) => {
                row.verdict = if verdict.conflict {
                    summary.conflicts += 1;
                    VERDICT_CONFLICT
                } else {
                    VERDICT_NO_CONFLICT
                }
                .to_string();
                row.confidence = Some(confidence_str(verdict.confidence).to_string());
                if !verdict.statements.is_empty() {
                    row.statements = Some(
                        serde_json::to_string(&verdict.statements)
                            .context("serialising conflicting statements")?,
                    );
                }
            }
            Err(e) => {
                warn!(a = %path_a, b = %path_b, "Judge call failed (will retry next run): {e:#}");
                row.error = Some(
                    truncate_on_char_boundary(&format!("{e:#}"), MAX_STORED_ERROR_BYTES)
                        .to_string(),
                );
                summary.errors += 1;
                failed = true;
            }
        }
    } else {
        summary.candidates += 1;
    }

    deps.state.upsert_contradiction_pair(&row).await?;
    Ok(if failed {
        Evaluated::JudgeFailed
    } else {
        Evaluated::Stored
    })
}

fn confidence_str(c: Confidence) -> &'static str {
    match c {
        Confidence::High => "high",
        Confidence::Medium => "medium",
        Confidence::Low => "low",
    }
}

/// The text of a KB document, read without following a symlink. `Ok(None)` when the
/// file is not there (or is not a regular file).
async fn read_document(root: &Path, rel: &str) -> Result<Option<String>> {
    match inspect_kb_path(root, Path::new(rel)) {
        Ok(KbEntry::File) => {}
        Ok(_) => return Ok(None),
        Err(e) => anyhow::bail!("refused to read {rel}: {e:?}"),
    }
    match tokio::fs::read_to_string(root.join(rel)).await {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {rel}")),
    }
}

// ---------------------------------------------------------------------------
// Reading results back
// ---------------------------------------------------------------------------

/// One stored pair as `contradictions --list` and `GET /status/contradictions`
/// report it. Keys that are empty are omitted.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct PairView {
    pub path_a: String,
    pub path_b: String,
    pub similarity: f64,
    pub verdict: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub statements: Vec<ConflictingStatements>,
    /// UTC, `YYYY-MM-DD HH:MM:SS`.
    pub judged_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl From<ContradictionPair> for PairView {
    fn from(p: ContradictionPair) -> Self {
        let statements = p
            .statements
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();
        Self {
            path_a: p.path_a,
            path_b: p.path_b,
            similarity: p.similarity,
            verdict: p.verdict,
            confidence: p.confidence,
            statements,
            judged_at: p.judged_at,
            judge_model: p.judge_model,
            error: p.error,
        }
    }
}

/// The stored pairs: conflicts first, then candidates, then pairs judged compatible.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct PairsReport {
    /// Pairs stored, before the cap.
    pub count: usize,
    pub pairs: Vec<PairView>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

/// Read the stored pairs, listing at most `limit` (clamped to `1..=MAX_LIST_LIMIT`).
pub async fn pairs_report(state: &StateDb, limit: usize) -> Result<PairsReport> {
    let limit = limit.clamp(1, MAX_LIST_LIMIT);
    let count = usize::try_from(state.count_contradiction_pairs().await?).unwrap_or(0);
    let pairs: Vec<PairView> = state
        .list_contradiction_pairs(limit as i64)
        .await?
        .into_iter()
        .map(PairView::from)
        .collect();
    Ok(PairsReport {
        count,
        truncated: pairs.len() < count,
        pairs,
    })
}

/// Human-readable rendering of a [`PairsReport`].
pub fn write_pairs(w: &mut impl std::io::Write, r: &PairsReport) -> std::io::Result<()> {
    if r.count == 0 {
        return writeln!(w, "No contradiction pairs stored.");
    }
    writeln!(w, "{} pair(s) stored (as of the last run):", r.count)?;
    for p in &r.pairs {
        let mut line = format!("{} [{:.3}]", p.verdict, p.similarity);
        if let Some(c) = &p.confidence {
            line.push_str(&format!(" confidence={c}"));
        }
        writeln!(w, "  {line}: {} <-> {}", p.path_a, p.path_b)?;
        for s in &p.statements {
            writeln!(w, "      A: {}", s.a)?;
            writeln!(w, "      B: {}", s.b)?;
        }
        if let Some(e) = &p.error {
            writeln!(w, "      error: {e}")?;
        }
    }
    if r.truncated {
        writeln!(w, "  ... and {} more not shown", r.count - r.pairs.len())?;
    }
    Ok(())
}

/// Human-readable rendering of a [`RunSummary`].
pub fn write_summary(w: &mut impl std::io::Write, s: &RunSummary) -> std::io::Result<()> {
    writeln!(
        w,
        "Scanned {} changed document(s); evaluated {} pair(s): {} conflict(s), {} candidate(s) \
         without a verdict, {} error(s).",
        s.documents_scanned, s.pairs_evaluated, s.conflicts, s.candidates, s.errors
    )?;
    if s.deferred_pairs > 0 {
        writeln!(
            w,
            "{} pair(s) left for the next run (per-run cap reached or run cancelled).",
            s.deferred_pairs
        )?;
    }
    if s.cancelled {
        writeln!(w, "The run was cancelled before it finished.")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::Verdict;
    use crate::qdrant::SearchResult;
    use std::pin::Pin;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tempfile::TempDir;

    // ── fakes ────────────────────────────────────────────────────────────────

    /// Neighbours keyed by the queried document's path; unknown documents have none.
    #[derive(Default)]
    struct FakeNeighbors {
        hits: HashMap<String, Vec<(String, f32)>>,
        failing: HashSet<String>,
        lookups: Mutex<Vec<String>>,
    }

    impl FakeNeighbors {
        fn with(mut self, path: &str, hits: &[(&str, f32)]) -> Self {
            self.hits.insert(
                make_point_id(path, 0),
                hits.iter().map(|(p, s)| (p.to_string(), *s)).collect(),
            );
            self
        }

        fn failing_for(mut self, path: &str) -> Self {
            self.failing.insert(make_point_id(path, 0));
            self
        }

        fn lookup_count(&self) -> usize {
            self.lookups.lock().unwrap().len()
        }
    }

    impl NeighborStore for FakeNeighbors {
        async fn recommend_by_point_id(
            &self,
            _collection: &str,
            point_id: &str,
            _limit: u64,
            _filter: Option<Filter>,
        ) -> Result<Vec<SearchResult>> {
            self.lookups.lock().unwrap().push(point_id.to_string());
            if self.failing.contains(point_id) {
                anyhow::bail!("qdrant is down");
            }
            Ok(self
                .hits
                .get(point_id)
                .into_iter()
                .flatten()
                .map(|(path, score)| {
                    let mut payload = HashMap::new();
                    payload.insert("file_path".to_string(), serde_json::json!(path));
                    SearchResult {
                        score: *score,
                        pre_rerank_score: None,
                        dense_score: Some(*score),
                        sparse_score: None,
                        phrase_score: None,
                        payload,
                    }
                })
                .collect())
        }
    }

    /// What a [`FakeJudge`] saw for one call.
    #[derive(Debug, Clone)]
    struct JudgeCall {
        len_a: usize,
        len_b: usize,
    }

    #[derive(Default)]
    struct FakeJudge {
        conflicts: HashSet<(String, String)>,
        fail_all: AtomicBool,
        calls: Mutex<Vec<JudgeCall>>,
        /// Set on every call, so a test can cancel the run from "inside" the judge.
        on_call: Option<Arc<AtomicBool>>,
    }

    impl FakeJudge {
        fn conflicting(pairs: &[(&str, &str)]) -> Self {
            Self {
                conflicts: pairs
                    .iter()
                    .map(|(a, b)| (a.to_string(), b.to_string()))
                    .collect(),
                ..Self::default()
            }
        }

        fn call_count(&self) -> usize {
            self.calls.lock().unwrap().len()
        }
    }

    impl Judge for FakeJudge {
        fn model(&self) -> &str {
            "fake-model"
        }

        fn judge<'a>(
            &'a self,
            pair: &'a PairInput<'a>,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<Verdict>> + Send + 'a>> {
            Box::pin(async move {
                self.calls.lock().unwrap().push(JudgeCall {
                    len_a: pair.text_a.len(),
                    len_b: pair.text_b.len(),
                });
                if let Some(flag) = &self.on_call {
                    flag.store(true, Ordering::SeqCst);
                }
                if self.fail_all.load(Ordering::SeqCst) {
                    anyhow::bail!("judge unavailable");
                }
                let conflict = self
                    .conflicts
                    .contains(&(pair.path_a.to_string(), pair.path_b.to_string()));
                Ok(Verdict {
                    conflict,
                    confidence: Confidence::High,
                    statements: if conflict {
                        vec![ConflictingStatements {
                            a: "A says 5".into(),
                            b: "B says 7".into(),
                        }]
                    } else {
                        Vec::new()
                    },
                })
            })
        }
    }

    // ── fixtures ─────────────────────────────────────────────────────────────

    struct Fixture {
        state: StateDb,
        dir: TempDir,
    }

    /// A state DB holding `docs` (hash `h-<path>`) and, under `kb/`, a file for each.
    async fn fixture(docs: &[&str]) -> Fixture {
        let dir = TempDir::new().unwrap();
        let state = StateDb::new(&dir.path().join("state.db")).await.unwrap();
        for p in docs {
            state
                .upsert_document_metadata(p, &HashMap::new(), 1, &format!("h-{p}"), 1)
                .await
                .unwrap();
            let file = dir.path().join("kb").join(p);
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(&file, format!("content of {p} {}", "x".repeat(200))).unwrap();
        }
        Fixture { state, dir }
    }

    impl Fixture {
        fn root(&self) -> std::path::PathBuf {
            self.dir.path().join("kb")
        }

        async fn change(&self, path: &str, hash: &str) {
            self.state
                .upsert_document_metadata(path, &HashMap::new(), 2, hash, 1)
                .await
                .unwrap();
        }

        async fn run(
            &self,
            neighbors: &FakeNeighbors,
            judge: Option<&dyn Judge>,
            params: RunParams,
        ) -> RunSummary {
            self.run_cancellable(neighbors, judge, params, &|| false)
                .await
        }

        async fn run_cancellable(
            &self,
            neighbors: &FakeNeighbors,
            judge: Option<&dyn Judge>,
            params: RunParams,
            cancelled: &(dyn Fn() -> bool + Sync),
        ) -> RunSummary {
            let root = self.root();
            let deps = RunDeps {
                state: &self.state,
                neighbors,
                collection: "kb",
                judge,
                data_path: &root,
                params,
            };
            run_once(&deps, cancelled).await.unwrap()
        }
    }

    fn params() -> RunParams {
        RunParams {
            top_k: 5,
            min_similarity: 0.80,
            max_pairs_per_run: 100,
            max_document_bytes: 6000,
        }
    }

    // ── candidate selection ──────────────────────────────────────────────────

    #[tokio::test]
    async fn neighbours_are_floored_deduped_self_excluded_and_capped_to_top_k() {
        let fx = fixture(&["a.md", "b.md", "c.md", "d.md", "e.md"]).await;
        let n = FakeNeighbors::default().with(
            "a.md",
            &[
                ("a.md", 0.99), // itself
                ("b.md", 0.90), // the best score for b wins,
                ("b.md", 0.95), // whichever order they arrive in
                ("c.md", 0.85),
                ("d.md", 0.50),     // under the floor
                ("ghost.md", 0.97), // not in the index
                ("e.md", 0.82),     // pushed out by top_k = 2
            ],
        );
        let p = RunParams {
            top_k: 2,
            ..params()
        };
        let s = fx.run(&n, None, p).await;

        let pairs = fx.state.all_contradiction_pairs().await.unwrap();
        let keys: Vec<(&str, &str)> = pairs
            .iter()
            .map(|p| (p.path_a.as_str(), p.path_b.as_str()))
            .collect();
        assert_eq!(keys, [("a.md", "b.md"), ("a.md", "c.md")]);
        assert!((pairs[0].similarity - 0.95).abs() < 1e-6);
        assert_eq!(s.pairs_evaluated, 2);
        assert_eq!(s.documents_scanned, 5);
        assert_eq!(s.deferred_pairs, 0);
    }

    #[tokio::test]
    async fn a_pair_found_from_both_sides_is_one_row_with_the_better_score() {
        let fx = fixture(&["a.md", "b.md"]).await;
        let n = FakeNeighbors::default()
            .with("a.md", &[("b.md", 0.88)])
            .with("b.md", &[("a.md", 0.93)]);
        let s = fx.run(&n, None, params()).await;
        assert_eq!(s.pairs_evaluated, 1);
        let pairs = fx.state.all_contradiction_pairs().await.unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(
            (pairs[0].path_a.as_str(), pairs[0].path_b.as_str()),
            ("a.md", "b.md")
        );
        assert!((pairs[0].similarity - 0.93).abs() < 1e-6);
        assert_eq!(pairs[0].hash_a, "h-a.md");
        assert_eq!(pairs[0].hash_b, "h-b.md");
    }

    // ── no judge ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn without_a_judge_pairs_are_recorded_as_candidates() {
        let fx = fixture(&["a.md", "b.md"]).await;
        // No file is read without a judge.
        std::fs::remove_file(fx.root().join("a.md")).unwrap();
        let n = FakeNeighbors::default().with("a.md", &[("b.md", 0.9)]);
        let s = fx.run(&n, None, params()).await;
        assert_eq!(
            (s.pairs_evaluated, s.candidates, s.conflicts, s.errors),
            (1, 1, 0, 0)
        );
        let row = fx
            .state
            .get_contradiction_pair("a.md", "b.md")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(row.verdict, VERDICT_CANDIDATE);
        assert!(row.error.is_none() && row.judge_model.is_none() && row.confidence.is_none());
    }

    #[tokio::test]
    async fn a_second_run_over_an_unchanged_corpus_evaluates_nothing() {
        let fx = fixture(&["a.md", "b.md", "c.md"]).await;
        let n = FakeNeighbors::default()
            .with("a.md", &[("b.md", 0.9), ("c.md", 0.85)])
            .with("b.md", &[("a.md", 0.9)]);
        let first = fx.run(&n, None, params()).await;
        assert_eq!(first.pairs_evaluated, 2);
        let lookups = n.lookup_count();

        let second = fx.run(&n, None, params()).await;
        assert_eq!(second, RunSummary::default());
        assert_eq!(
            n.lookup_count(),
            lookups,
            "no document is changed, so none is queried"
        );
    }

    #[tokio::test]
    async fn only_pairs_of_a_changed_document_are_re_evaluated() {
        let fx = fixture(&["a.md", "b.md", "c.md"]).await;
        let n = FakeNeighbors::default()
            .with("a.md", &[("b.md", 0.9), ("c.md", 0.85)])
            .with("c.md", &[("a.md", 0.86)]);
        assert_eq!(fx.run(&n, None, params()).await.pairs_evaluated, 2);
        let ab_before = fx
            .state
            .get_contradiction_pair("a.md", "b.md")
            .await
            .unwrap()
            .unwrap();

        fx.change("c.md", "h-c-v2").await;
        let s = fx.run(&n, None, params()).await;
        assert_eq!((s.documents_scanned, s.pairs_evaluated), (1, 1));
        let ac = fx
            .state
            .get_contradiction_pair("a.md", "c.md")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(ac.hash_b, "h-c-v2");
        assert_eq!(
            fx.state
                .get_contradiction_pair("a.md", "b.md")
                .await
                .unwrap()
                .unwrap(),
            ab_before,
            "the untouched pair is not rewritten"
        );
    }

    #[tokio::test]
    async fn a_changed_document_skips_pairs_already_stored_for_current_hashes() {
        let fx = fixture(&["a.md", "b.md"]).await;
        let n = FakeNeighbors::default().with("a.md", &[("b.md", 0.9)]);
        assert_eq!(fx.run(&n, None, params()).await.pairs_evaluated, 1);

        // Forget the analysis markers (as after a restore): both documents look
        // changed again, but the stored pair still matches their hashes.
        sqlx::query("DELETE FROM contradiction_docs")
            .execute(fx.state.pool_for_test())
            .await
            .unwrap();
        let s = fx.run(&n, None, params()).await;
        assert_eq!((s.documents_scanned, s.pairs_evaluated), (2, 0));
        assert_eq!(
            fx.state
                .contradiction_analyzed_hashes()
                .await
                .unwrap()
                .len(),
            2
        );
    }

    // ── judge ────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn the_judge_verdict_is_recorded_with_statements_and_truncated_input() {
        let fx = fixture(&["a.md", "b.md", "c.md"]).await;
        let n = FakeNeighbors::default().with("a.md", &[("b.md", 0.9), ("c.md", 0.85)]);
        let judge = FakeJudge::conflicting(&[("a.md", "b.md")]);
        let p = RunParams {
            max_document_bytes: 20,
            ..params()
        };
        let s = fx.run(&n, Some(&judge), p).await;
        assert_eq!(
            (s.pairs_evaluated, s.conflicts, s.candidates, s.errors),
            (2, 1, 0, 0)
        );

        let ab = fx
            .state
            .get_contradiction_pair("a.md", "b.md")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(ab.verdict, VERDICT_CONFLICT);
        assert_eq!(ab.confidence.as_deref(), Some("high"));
        assert_eq!(ab.judge_model.as_deref(), Some("fake-model"));
        let view = PairView::from(ab);
        assert_eq!(view.statements.len(), 1);
        assert_eq!(view.statements[0].a, "A says 5");
        let ac = fx
            .state
            .get_contradiction_pair("a.md", "c.md")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(ac.verdict, VERDICT_NO_CONFLICT);
        assert!(ac.statements.is_none());

        for call in judge.calls.lock().unwrap().iter() {
            assert!(call.len_a <= 20 && call.len_b <= 20, "{call:?}");
        }
    }

    #[tokio::test]
    async fn a_judge_failure_is_recorded_and_the_run_continues() {
        let fx = fixture(&["a.md", "b.md", "c.md"]).await;
        let n = FakeNeighbors::default().with("a.md", &[("b.md", 0.9), ("c.md", 0.85)]);
        let judge = FakeJudge::default();
        judge.fail_all.store(true, Ordering::SeqCst);
        let s = fx.run(&n, Some(&judge), params()).await;
        assert_eq!((s.pairs_evaluated, s.errors, s.conflicts), (2, 2, 0));
        assert_eq!(judge.call_count(), 2, "the second pair is still tried");
        let row = fx
            .state
            .get_contradiction_pair("a.md", "b.md")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(row.verdict, VERDICT_CANDIDATE);
        assert!(row.error.as_deref().unwrap().contains("judge unavailable"));
        assert_eq!(row.judge_model.as_deref(), Some("fake-model"));

        // The provider recovers: the error rows are retried although nothing changed.
        judge.fail_all.store(false, Ordering::SeqCst);
        let s = fx.run(&n, Some(&judge), params()).await;
        assert_eq!(
            (s.pairs_evaluated, s.errors, s.documents_scanned),
            (2, 0, 0)
        );
        let row = fx
            .state
            .get_contradiction_pair("a.md", "b.md")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(row.verdict, VERDICT_NO_CONFLICT);
        assert!(row.error.is_none());

        // And now everything is final.
        assert_eq!(
            fx.run(&n, Some(&judge), params()).await,
            RunSummary::default()
        );
    }

    #[tokio::test]
    async fn candidates_from_a_judge_less_run_are_judged_once_a_judge_is_configured() {
        let fx = fixture(&["a.md", "b.md"]).await;
        let n = FakeNeighbors::default().with("a.md", &[("b.md", 0.9)]);
        assert_eq!(fx.run(&n, None, params()).await.candidates, 1);
        // Still nothing to do while there is no judge.
        assert_eq!(fx.run(&n, None, params()).await, RunSummary::default());

        let judge = FakeJudge::conflicting(&[("a.md", "b.md")]);
        let s = fx.run(&n, Some(&judge), params()).await;
        assert_eq!((s.pairs_evaluated, s.conflicts), (1, 1));
        assert_eq!(
            fx.state
                .get_contradiction_pair("a.md", "b.md")
                .await
                .unwrap()
                .unwrap()
                .verdict,
            VERDICT_CONFLICT
        );
    }

    #[tokio::test]
    async fn a_run_stops_after_repeated_judge_failures() {
        let docs: Vec<String> = (0..10).map(|i| format!("d{i}.md")).collect();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let fx = fixture(&refs).await;
        let hits: Vec<(&str, f32)> = refs[1..].iter().map(|p| (*p, 0.9)).collect();
        let n = FakeNeighbors::default().with("d0.md", &hits);
        let judge = FakeJudge::default();
        judge.fail_all.store(true, Ordering::SeqCst);
        let p = RunParams {
            top_k: 9,
            ..params()
        };
        let s = fx.run(&n, Some(&judge), p).await;
        assert_eq!(judge.call_count(), MAX_CONSECUTIVE_JUDGE_FAILURES);
        assert_eq!(s.errors, MAX_CONSECUTIVE_JUDGE_FAILURES);
        assert_eq!(s.deferred_pairs, 9 - MAX_CONSECUTIVE_JUDGE_FAILURES);
    }

    #[tokio::test]
    async fn a_vanished_file_skips_its_pair_and_leaves_the_document_for_next_run() {
        let fx = fixture(&["a.md", "b.md"]).await;
        std::fs::remove_file(fx.root().join("b.md")).unwrap();
        let n = FakeNeighbors::default().with("a.md", &[("b.md", 0.9)]);
        let judge = FakeJudge::default();
        let s = fx.run(&n, Some(&judge), params()).await;
        assert_eq!((s.pairs_evaluated, s.errors, s.deferred_pairs), (0, 0, 0));
        assert_eq!(judge.call_count(), 0);
        assert!(
            fx.state
                .get_contradiction_pair("a.md", "b.md")
                .await
                .unwrap()
                .is_none()
        );
        let analysed = fx.state.contradiction_analyzed_hashes().await.unwrap();
        assert!(
            !analysed.contains_key("a.md"),
            "retried once the index catches up"
        );
        assert!(analysed.contains_key("b.md"), "b.md needed no pair");
    }

    // ── cap, cancellation, lookup failures ───────────────────────────────────

    #[tokio::test]
    async fn the_per_run_cap_carries_the_rest_over() {
        let fx = fixture(&["a.md", "b.md", "c.md"]).await;
        let n = FakeNeighbors::default().with("a.md", &[("b.md", 0.9), ("c.md", 0.85)]);
        let p = RunParams {
            max_pairs_per_run: 1,
            ..params()
        };

        let s = fx.run(&n, None, p).await;
        assert_eq!((s.pairs_evaluated, s.deferred_pairs), (1, 1));
        assert!(
            !fx.state
                .contradiction_analyzed_hashes()
                .await
                .unwrap()
                .contains_key("a.md"),
            "a.md has a pair left, so it is not analysed yet"
        );

        let s = fx.run(&n, None, p).await;
        assert_eq!((s.pairs_evaluated, s.deferred_pairs), (1, 0));
        assert_eq!(fx.state.count_contradiction_pairs().await.unwrap(), 2);
        assert!(
            fx.state
                .contradiction_analyzed_hashes()
                .await
                .unwrap()
                .contains_key("a.md")
        );

        assert_eq!(fx.run(&n, None, p).await, RunSummary::default());
    }

    #[tokio::test]
    async fn cancellation_is_checked_between_pairs() {
        let fx = fixture(&["a.md", "b.md", "c.md", "d.md"]).await;
        let n =
            FakeNeighbors::default().with("a.md", &[("b.md", 0.9), ("c.md", 0.88), ("d.md", 0.86)]);
        let flag = Arc::new(AtomicBool::new(false));
        let judge = FakeJudge {
            on_call: Some(Arc::clone(&flag)),
            ..FakeJudge::default()
        };
        let poll = Arc::clone(&flag);
        let s = fx
            .run_cancellable(&n, Some(&judge), params(), &move || {
                poll.load(Ordering::SeqCst)
            })
            .await;
        assert!(s.cancelled);
        assert_eq!((s.pairs_evaluated, s.deferred_pairs), (1, 2));
        assert_eq!(
            judge.call_count(),
            1,
            "no judge call starts after cancellation"
        );
        assert!(
            !fx.state
                .contradiction_analyzed_hashes()
                .await
                .unwrap()
                .contains_key("a.md")
        );
    }

    #[tokio::test]
    async fn a_run_cancelled_before_it_starts_does_nothing() {
        let fx = fixture(&["a.md", "b.md"]).await;
        let n = FakeNeighbors::default().with("a.md", &[("b.md", 0.9)]);
        let s = fx.run_cancellable(&n, None, params(), &|| true).await;
        assert!(s.cancelled);
        assert_eq!(n.lookup_count(), 0);
        assert_eq!(fx.state.count_contradiction_pairs().await.unwrap(), 0);
        assert!(
            fx.state
                .contradiction_analyzed_hashes()
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn a_failed_neighbour_lookup_leaves_the_document_unanalysed() {
        let fx = fixture(&["a.md", "b.md"]).await;
        let n = FakeNeighbors::default()
            .with("b.md", &[("a.md", 0.9)])
            .failing_for("a.md");
        let s = fx.run(&n, None, params()).await;
        assert_eq!(
            (s.documents_scanned, s.errors, s.pairs_evaluated),
            (1, 1, 1)
        );
        let analysed = fx.state.contradiction_analyzed_hashes().await.unwrap();
        assert!(!analysed.contains_key("a.md") && analysed.contains_key("b.md"));
    }

    #[tokio::test]
    async fn rows_for_removed_documents_are_pruned_at_the_start_of_a_run() {
        let fx = fixture(&["a.md", "b.md"]).await;
        let n = FakeNeighbors::default().with("a.md", &[("b.md", 0.9)]);
        fx.run(&n, None, params()).await;
        // Removed behind the contradiction tables' back (not via `delete_document`).
        sqlx::query("DELETE FROM documents WHERE file_path = 'b.md'")
            .execute(fx.state.pool_for_test())
            .await
            .unwrap();
        fx.run(&n, None, params()).await;
        assert_eq!(fx.state.count_contradiction_pairs().await.unwrap(), 0);
        assert!(
            !fx.state
                .contradiction_analyzed_hashes()
                .await
                .unwrap()
                .contains_key("b.md")
        );
    }

    // ── exclusive runs ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn a_busy_run_lock_skips_the_run() {
        let fx = fixture(&["a.md", "b.md"]).await;
        let n = FakeNeighbors::default().with("a.md", &[("b.md", 0.9)]);
        let root = fx.root();
        let deps = RunDeps {
            state: &fx.state,
            neighbors: &n,
            collection: "kb",
            judge: None,
            data_path: &root,
            params: params(),
        };
        let lock = new_run_lock();
        let held = lock.clone().lock_owned().await;
        assert!(
            run_once_exclusive(&lock, &deps, &|| false)
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(n.lookup_count(), 0);
        drop(held);
        let summary = run_once_exclusive(&lock, &deps, &|| false)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(summary.pairs_evaluated, 1);
    }

    // ── reporting ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn the_pairs_report_lists_conflicts_first_and_flags_truncation() {
        let fx = fixture(&["a.md", "b.md", "c.md"]).await;
        let n = FakeNeighbors::default().with("a.md", &[("b.md", 0.9), ("c.md", 0.85)]);
        let judge = FakeJudge::conflicting(&[("a.md", "c.md")]);
        fx.run(&n, Some(&judge), params()).await;

        let full = pairs_report(&fx.state, 10).await.unwrap();
        assert_eq!((full.count, full.truncated), (2, false));
        assert_eq!(full.pairs[0].verdict, VERDICT_CONFLICT);
        assert_eq!(full.pairs[1].verdict, VERDICT_NO_CONFLICT);

        let capped = pairs_report(&fx.state, 1).await.unwrap();
        assert_eq!(
            (capped.count, capped.pairs.len(), capped.truncated),
            (2, 1, true)
        );

        // Default-valued keys are left out of the JSON.
        let json = serde_json::to_value(&full).unwrap();
        assert_eq!(json["count"], 2);
        assert!(json.get("truncated").is_none());
        assert!(json["pairs"][1].get("statements").is_none());
        assert!(json["pairs"][1].get("error").is_none());
        assert_eq!(json["pairs"][0]["statements"][0]["b"], "B says 7");
        assert_eq!(json["pairs"][0]["judge_model"], "fake-model");

        let mut out = Vec::new();
        write_pairs(&mut out, &capped).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.contains("conflict") && text.contains("a.md <-> c.md"),
            "{text}"
        );
        assert!(text.contains("and 1 more not shown"), "{text}");
    }

    #[tokio::test]
    async fn an_empty_store_reports_no_pairs() {
        let fx = fixture(&[]).await;
        let report = pairs_report(&fx.state, DEFAULT_LIST_LIMIT).await.unwrap();
        assert_eq!((report.count, report.pairs.len()), (0, 0));
        let mut out = Vec::new();
        write_pairs(&mut out, &report).unwrap();
        assert!(
            String::from_utf8(out)
                .unwrap()
                .contains("No contradiction pairs")
        );
    }
}
