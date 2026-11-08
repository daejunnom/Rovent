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
    #[cfg(feature = "archive-origin-diagnostics")]
    archive_origins: ArchiveOriginObservation,
}

// This is a bounded diagnostic, not a record of NN execution or acceptance.
// It stores scalars only and does no file I/O or JSON serialization at submit.
#[cfg(feature = "archive-origin-diagnostics")]
#[derive(serde::Serialize)]
pub(super) struct ArchiveOriginEvent {
    pub(super) authority: [u64; 3],
    pub(super) authority_implementation: [u8; 32],
    pub(super) request_epoch: u64,
    pub(super) request_sequence: u64,
    pub(super) role: &'static str,
    pub(super) question: [u8; 32],
    pub(super) view_input_digest: [u8; 32],
    pub(super) view_seal: [u8; 32],
    pub(super) view_sequence: u64,
    pub(super) record_index: usize,
    pub(super) source_state: usize,
    pub(super) source_revision: u64,
    pub(super) source_line_offset: u16,
    pub(super) selected_anchor: u8,
    pub(super) archive_generation: u64,
    pub(super) archive_checksum: [u8; 32],
    pub(super) original_record_digest: [u8; 32],
}

#[cfg(feature = "archive-origin-diagnostics")]
#[derive(Default)]
pub(super) struct ArchiveOriginObservation {
    pub(super) events: VecDeque<ArchiveOriginEvent>,
    pub(super) omitted_events: u64,
    pub(super) unknown_records: u64,
    pub(super) binding_mismatches: u64,
    pub(super) unavailable: Option<&'static str>,
}

/// Taken by the actual native owner after it has stopped dispatching requests.
/// A failed nonblocking take, unsampled tickets and physical execution remain
/// unknown. This object owns no live view, Store pin or archive capability.
#[cfg(feature = "archive-origin-diagnostics")]
#[allow(dead_code)] // Read by the future managed collector after dispatch stops.
pub(super) struct ArchiveOriginSnapshot {
    pub(super) observation: ArchiveOriginObservation,
    pub(super) observer_unavailable: Option<&'static str>,
}

#[cfg(feature = "archive-origin-diagnostics")]
impl ArchiveOriginObservation {
    fn prepare(&mut self) {
        if self.events.try_reserve_exact(MAX_ROWS).is_err() {
            self.unavailable = Some("reservation_failed");
        }
    }

    fn push(&mut self, event: ArchiveOriginEvent) {
        if self.events.len() == MAX_ROWS {
            self.events.pop_front();
            self.omitted_events = self.omitted_events.saturating_add(1);
        }
        self.events.push_back(event);
    }

    fn observe(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<ReviewView, NativeTicketQuestion>,
    ) {
        if self.unavailable.is_some() {
            return;
        }
        let view = &question.view;
        if ticket.view != question.seal
            || ticket.question != question.question
            || ticket.role != question.role
            || ticket.view.authority != ticket.authority
            || ticket.id.epoch != ticket.authority.epoch
            || ticket.view.view_sequence != view.sequence()
            || ticket.view.view_digest.0 != view.seal_digest()
            || usize::from(ticket.view.selected_records) != view.records().len()
            || view.records_metadata().len() != view.records().len()
        {
            self.binding_mismatches = self.binding_mismatches.saturating_add(1);
            return;
        }
        for (index, meta) in view.records_metadata().iter().enumerate() {
            let Some(origin) = view.selected_archive_origin(index) else {
                self.unknown_records = self.unknown_records.saturating_add(1);
                continue;
            };
            let authority = ticket.authority;
            self.push(ArchiveOriginEvent {
                authority: [authority.epoch.0, authority.game.0, authority.root.0],
                authority_implementation: authority.implementation.0,
                request_epoch: ticket.id.epoch.0,
                request_sequence: ticket.id.sequence,
                role: match ticket.role {
                    rz_contracts::pals::Role::Proposer => "proposer",
                    rz_contracts::pals::Role::Critic => "critic",
                    rz_contracts::pals::Role::Validator => "validator",
                },
                question: ticket.question.0,
                view_input_digest: view.digest(),
                view_seal: view.seal_digest(),
                view_sequence: view.sequence(),
                record_index: index,
                source_state: meta.source_state.0,
                source_revision: meta.source_revision,
                source_line_offset: meta.source_line_offset,
                selected_anchor: meta.anchor,
                archive_generation: origin.archive_generation(),
                archive_checksum: origin.archive_checksum(),
                original_record_digest: origin.record_digest(),
            });
        }
    }
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
    let state = State::default();
    #[cfg(feature = "archive-origin-diagnostics")]
    let state = {
        let mut state = state;
        state.archive_origins.prepare();
        state
    };
    let state = Arc::new(Mutex::new(state));
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
        #[cfg(feature = "archive-origin-diagnostics")]
        state.archive_origins.observe(ticket, question);
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
    /// Transfer bounded scalar evidence to a future managed collector. This
    /// does not write to stderr or any file. Call only after owner dispatch has
    /// stopped; absence due to lock contention or poisoning is unknown.
    #[cfg(feature = "archive-origin-diagnostics")]
    pub(super) fn take_archive_origins(&self) -> Option<ArchiveOriginSnapshot> {
        let mut state = self.0.try_lock().ok()?;
        let observation = std::mem::take(&mut state.archive_origins);
        state.archive_origins.unavailable = Some("owner_already_took_observation");
        Some(ArchiveOriginSnapshot {
            observation,
            observer_unavailable: state.unavailable,
        })
    }

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
        // This is emitted after the original runner's cleanup attempt. The
        // bounded events identify views observed at the original submission
        // callback; neither callback nor output proves physical execution or
        // completion. A busy/poisoned observer stays unknown.
        #[cfg(feature = "archive-origin-diagnostics")]
        {
            let report = archive_origin_report(self.take_archive_origins());
            let _ = writeln!(std::io::stderr().lock(), "{report}");
        }
    }
}

#[cfg(feature = "archive-origin-diagnostics")]
fn archive_origin_report(snapshot: Option<ArchiveOriginSnapshot>) -> Value {
    match snapshot {
        Some(snapshot) => json!({
            "domain": "rz-selected-archive-origin-observation/1",
            "scope": "bounded_original_submission_callbacks_only",
            "owner_match_scope": "positive_events_only_verified_at_view_selection",
            "events": snapshot.observation.events,
            "omitted_events": snapshot.observation.omitted_events,
            "unknown_records": snapshot.observation.unknown_records,
            "binding_mismatches": snapshot.observation.binding_mismatches,
            "unavailable": snapshot.observation.unavailable,
            "observer_unavailable": snapshot.observer_unavailable,
            "lock_contention_coverage": "unknown",
            "physical_completion": "unknown",
        }),
        None => json!({
            "domain": "rz-selected-archive-origin-observation/1",
            "scope": "bounded_original_submission_callbacks_only",
            "unavailable": "observation_busy_or_poisoned",
            "physical_completion": "unknown",
        }),
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

    #[cfg(feature = "archive-origin-diagnostics")]
    #[test]
    fn archive_origin_observation_is_bounded_and_owner_take_is_one_shot() {
        let (observer, handle) = attach(true, None);
        let _observer = observer.expect("enabled observer");
        let handle = handle.expect("owner handle");
        {
            let mut state = handle.0.lock().unwrap();
            assert!(state.archive_origins.unavailable.is_none());
            for sequence in 0..(MAX_ROWS as u64 + 1) {
                state.archive_origins.push(ArchiveOriginEvent {
                    authority: [0; 3],
                    authority_implementation: [0; 32],
                    request_epoch: 0,
                    request_sequence: sequence,
                    role: "test",
                    question: [0; 32],
                    view_input_digest: [0; 32],
                    view_seal: [0; 32],
                    view_sequence: 0,
                    record_index: 0,
                    source_state: 0,
                    source_revision: 0,
                    source_line_offset: 0,
                    selected_anchor: 0,
                    archive_generation: 0,
                    archive_checksum: [0; 32],
                    original_record_digest: [0; 32],
                });
            }
        }
        let taken = handle.take_archive_origins().expect("uncontended take");
        assert_eq!(taken.observation.events.len(), MAX_ROWS);
        assert_eq!(taken.observation.omitted_events, 1);
        assert_eq!(
            taken.observation.events.front().unwrap().request_sequence,
            1
        );
        let report = archive_origin_report(Some(taken));
        assert_eq!(report["events"].as_array().unwrap().len(), MAX_ROWS);
        assert_eq!(report["omitted_events"], 1);
        assert_eq!(report["physical_completion"], "unknown");
        assert_eq!(report["lock_contention_coverage"], "unknown");
        let unavailable = archive_origin_report(None);
        assert_eq!(unavailable["unavailable"], "observation_busy_or_poisoned");
        assert_eq!(unavailable["physical_completion"], "unknown");
        let second = handle
            .take_archive_origins()
            .expect("uncontended second take");
        assert!(second.observation.events.is_empty());
        assert_eq!(
            second.observation.unavailable,
            Some("owner_already_took_observation")
        );
    }

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
