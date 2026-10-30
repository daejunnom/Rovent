//! Optional, bounded observations of the original V3 ticket input. No live
//! view, store pin, physical completion or model authority is retained.
use crate::pals_native::ticketed::{
    NativeTicketQuestion, TicketExecutionEvidence, TicketPreparation,
    VirtualReviewCollectionObserver, VirtualReviewPhysicalInvocationEvidence,
    VirtualReviewPreparedTicket,
};
use rz_contracts::pals::RoleOutput;
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_eval::pals_onnx::VirtualReviewCudaPlacementWitness;
use rz_search::pals::coordinator::{PhysicalRoleCompletion, RoleFailure};
use rz_search::pals::review_context::historical::{
    HistoricalReviewDescriptor, HistoricalReviewDigests, verify_historical_review_digests,
};
use rz_search::pals::review_context::{ReviewAnchor, ReviewView};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::io::Write;
use std::sync::{Arc, Mutex};

// The last rows rather than the first ones make later roots visible. Eviction
// and skipped empty views are explicit; they are not complete coverage.
const MAX_ROWS: usize = 16;
const MAX_SAMPLE_RECORDS: usize = 4;
const MAX_SAMPLE_MOVES: usize = 16;

#[derive(Default)]
struct State {
    rows: VecDeque<Value>,
    omitted: u64,
    empty: u64,
    unavailable: Option<&'static str>,
}

pub(super) struct Handle(Arc<Mutex<State>>);
struct Observer(Arc<Mutex<State>>);

pub(super) fn attach(
    enabled: bool,
    observer: Option<Box<dyn VirtualReviewCollectionObserver>>,
) -> (
    Option<Box<dyn VirtualReviewCollectionObserver>>,
    Option<Handle>,
) {
    if !enabled {
        return (observer, None);
    }
    if observer.is_some() {
        let state = State {
            unavailable: Some("collection_observer_in_use"),
            ..State::default()
        };
        return (observer, Some(Handle(Arc::new(Mutex::new(state)))));
    }
    let state = Arc::new(Mutex::new(State::default()));
    (
        Some(Box::new(Observer(Arc::clone(&state)))),
        Some(Handle(state)),
    )
}

fn project(view: &ReviewView) -> Value {
    let sources: Vec<_> = view
        .records_metadata()
        .iter()
        .zip(view.source_snapshots().iter())
        .map(|(meta, snapshot)| ReviewAnchor {
            state: meta.source_state,
            snapshot: snapshot.clone(),
        })
        .collect();
    let descriptor = HistoricalReviewDescriptor {
        sequence: view.sequence(),
        generation: view.generation(),
        root_state: view.root_state(),
        policy: view.selection_policy(),
        anchors: view.anchors(),
        source_anchors: &sources,
        records: view.records(),
        metadata: view.records_metadata(),
        query: view.query(),
        tokens: view.token_count(),
        bucket: view.length_bucket(),
    };
    let expected = HistoricalReviewDigests {
        input_digest: view.digest(),
        view_seal_digest: view.seal_digest(),
        record_scope_digest: view.record_scope_digest(),
        policy_digest: view.policy_digest(),
    };
    let verification = verify_historical_review_digests(descriptor, &expected);
    let samples: Vec<_> = view.records().iter().zip(view.records_metadata())
        .zip(view.source_snapshots().iter()).take(MAX_SAMPLE_RECORDS)
        .map(|((record, meta), source)| json!({
            "source_state": meta.source_state.0, "source_revision": meta.source_revision,
            "source_rules_history_sha256": rz_position::contracts::rules_snapshot_semantic_digest(source).0,
            "selected_anchor": meta.anchor, "source_line_offset": meta.source_line_offset,
            "original_first_move": meta.original_first_move.map(|mv| mv.to_string()),
                "selected_line": record.line.iter().take(MAX_SAMPLE_MOVES).map(ToString::to_string).collect::<Vec<_>>(),
                "omitted_line_moves": record.line.len().saturating_sub(MAX_SAMPLE_MOVES),
            "parent": meta.parent, "supersedes": meta.supersedes,
        })).collect();
    json!({
        "view_sequence": view.sequence(), "generation": view.generation(),
        "root_state": view.root_state().0, "anchors": view.anchors().len(),
        "anchor_rules_history_sha256": view.anchors().iter().map(|anchor|
            rz_position::contracts::rules_snapshot_semantic_digest(&anchor.snapshot).0
        ).collect::<Vec<_>>(),
        "records": view.records().len(), "sample_records": samples,
        "omitted_records": view.records().len().saturating_sub(MAX_SAMPLE_RECORDS),
        "historical_rules_and_digest": if verification.is_ok() { "verified" } else { "failed" },
        "historical_error": verification.err().map(|error| error.to_string()),
        "archive_origin": "unknown", "physical_completion": "unknown",
    })
}

impl VirtualReviewCollectionObserver for Observer {
    fn submitted(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<ReviewView, NativeTicketQuestion>,
        _: &VirtualReviewPreparedTicket,
    ) -> Result<(), RoleFailure> {
        // Never block a product request on diagnostic state or fabricate an
        // observation for a view which was not actually submitted.
        let Ok(mut state) = self.0.try_lock() else {
            return Ok(());
        };
        if question.view.records().is_empty() {
            state.empty = state.empty.saturating_add(1);
            return Ok(());
        }
        let authority = ticket.authority;
        let key = (authority.epoch.0, authority.game.0, authority.root.0);
        if state
            .rows
            .iter()
            .any(|row| row["authority"] == json!([key.0, key.1, key.2]))
        {
            return Ok(());
        }
        let mut row = project(&question.view);
        row["authority"] = json!([key.0, key.1, key.2]);
        if state.rows.len() == MAX_ROWS {
            state.rows.pop_front();
            state.omitted = state.omitted.saturating_add(1);
        }
        state.rows.push_back(row);
        Ok(())
    }
    fn completed(
        &mut self,
        _: &RoleTicketEnvelope,
        _: PhysicalRoleCompletion,
        _: Option<&RoleOutput>,
        _: Option<TicketExecutionEvidence>,
        _: Option<VirtualReviewPhysicalInvocationEvidence>,
    ) -> Result<(), RoleFailure> {
        Ok(())
    }
    fn retired(&mut self, _: &RoleTicketEnvelope, _: bool) -> Result<(), RoleFailure> {
        Ok(())
    }
    fn runtime_audited(
        &mut self,
        _: &VirtualReviewCudaPlacementWitness,
    ) -> Result<(), RoleFailure> {
        Ok(())
    }
}

impl Handle {
    pub(super) fn emit(self) {
        if let Ok(state) = self.0.try_lock() {
            // Emitted only after original runner cleanup. A failed diagnostic
            // write never changes the game result or physical fence.
            let _ = writeln!(
                std::io::stderr().lock(),
                "{}",
                json!({
                    "domain": "rz-selected-review-observation/1",
                "scope": "sampled_submitted_views_only", "archive_origin": "unknown",
                    "rows": state.rows, "omitted_roots": state.omitted,
                    "empty_submissions": state.empty, "unsubmitted_views": "unknown",
                    "unavailable": state.unavailable, "lock_contention_coverage": "unknown",
                })
            );
        } else {
            let _ = writeln!(
                std::io::stderr().lock(),
                "{}",
                json!({
                "domain": "rz-selected-review-observation/1", "scope": "sampled_submitted_views_only",
                    "unavailable": "observation_busy_or_poisoned", "archive_origin": "unknown",
                })
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_position::{BoardMove, Color, Position};
    use rz_search::pals::engine::{RecordKind, RoleQueryPurpose, RoleRecord};
    use rz_search::pals::review_context::{
        ReviewContextIndex, ReviewQuerySequences, ReviewSelectionPolicy,
    };
    use rz_search::pals::store::{PalsStores, StoreLimits};

    #[test]
    fn submitted_projection_checks_exact_rebased_history_without_authority() {
        let mut stores = PalsStores::new(StoreLimits::default());
        let start = Position::startpos();
        let old = stores.focus_actual_moves(start.snapshot()).unwrap();
        let old_state = stores.situations.get(old).unwrap().state;
        let first = BoardMove::from_uci("e2e4").unwrap();
        let reply = BoardMove::from_uci("e7e5").unwrap();
        let mut index = ReviewContextIndex::default();
        index
            .publish(
                &stores,
                &RoleRecord {
                    revision: 1,
                    parent_revision: None,
                    supersedes_revision: None,
                    origin_state: old_state,
                    kind: RecordKind::Proposal,
                    line: vec![first, reply],
                    value: None,
                    completed_depth: 0,
                    score_scope: None,
                    cpu_observation: None,
                    perspective: Color::White,
                    critical: true,
                },
                None,
            )
            .unwrap();
        let mut current = start;
        current.make_move(first).unwrap();
        let root = stores.focus_actual_moves(current.snapshot()).unwrap();
        let root_state = stores.situations.get(root).unwrap().state;
        let view = index
            .select(
                &stores,
                root_state,
                ReviewQuerySequences {
                    purpose: RoleQueryPurpose::ValueFresh,
                    prefix: &[],
                    proposal: &[],
                    counterexample: None,
                    legal: &[],
                    divergences: &[],
                },
                ReviewSelectionPolicy::default(),
            )
            .unwrap();
        let row = project(&view);
        assert_eq!(row["historical_rules_and_digest"], "verified");
        assert_eq!(row["sample_records"][0]["source_line_offset"], 1);
        assert_eq!(row["sample_records"][0]["selected_line"], json!(["e7e5"]));
        assert_eq!(row["archive_origin"], "unknown");
        assert_eq!(row["physical_completion"], "unknown");
    }
}
