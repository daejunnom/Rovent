use super::*;
use crate::pals::engine::{RecordKind, RoleQueryPurpose};
use crate::pals::review_context::{ReviewContextIndex, ReviewQuerySequences, ReviewView};
use crate::pals::store::{PalsStores, StoreLimits};
use rz_position::{BoardMove, Color};
use std::sync::atomic::Ordering;

fn line(text: &str) -> Vec<BoardMove> {
    text.split_whitespace()
        .map(|movement| BoardMove::from_uci(movement).unwrap())
        .collect()
}

fn record(state: StateId, revision: u64, parent: Option<u64>) -> RoleRecord {
    RoleRecord {
        revision,
        parent_revision: parent,
        supersedes_revision: None,
        origin_state: state,
        kind: if parent.is_some() {
            RecordKind::Counterexample
        } else {
            RecordKind::Proposal
        },
        line: line("e2e4 e7e5 g1f3"),
        value: None,
        completed_depth: 0,
        score_scope: None,
        cpu_observation: None,
        perspective: Color::White,
        critical: true,
    }
}

fn view(policy: ReviewSelectionPolicy) -> (PalsStores, ReviewContextIndex, ReviewView) {
    let mut stores = PalsStores::new(StoreLimits::default());
    let situation = stores
        .focus_actual_moves(Position::startpos().snapshot())
        .unwrap();
    let state = stores.situations.get(situation).unwrap().state;
    let mut index = ReviewContextIndex::default();
    index
        .publish(&stores, &record(state, 1, None), None)
        .unwrap();
    index
        .publish(&stores, &record(state, 2, Some(1)), None)
        .unwrap();
    let prefix = line("e2e4 e7e5");
    let proposal = line("e2e4 e7e5 g1f3");
    let counter = line("e2e4 c7c5 g1f3");
    let candidates = line("g1f3 d2d4");
    let view = index
        .select(
            &stores,
            state,
            ReviewQuerySequences {
                purpose: RoleQueryPurpose::RepairPolicy,
                prefix: &prefix,
                proposal: &proposal,
                counterexample: Some(&counter),
                legal: &candidates,
                divergences: &[0, 2],
            },
            policy,
        )
        .unwrap();
    (stores, index, view)
}

fn sources(stores: &PalsStores, view: &ReviewView) -> Vec<ReviewAnchor> {
    view.records_metadata()
        .iter()
        .zip(view.source_snapshots().iter())
        .map(|(metadata, snapshot)| {
            assert!(
                stores
                    .states
                    .get(metadata.source_state)
                    .unwrap()
                    .weak_position_identity()
                    .matches(snapshot)
            );
            ReviewAnchor {
                state: metadata.source_state,
                snapshot: snapshot.clone(),
            }
        })
        .collect()
}

fn descriptor<'a>(
    view: &'a ReviewView,
    source_anchors: &'a [ReviewAnchor],
) -> HistoricalReviewDescriptor<'a> {
    HistoricalReviewDescriptor {
        sequence: view.sequence(),
        generation: view.generation(),
        root_state: view.root_state(),
        policy: view.policy,
        anchors: view.anchors(),
        source_anchors,
        records: view.records(),
        metadata: view.records_metadata(),
        query: view.query(),
        tokens: view.token_count(),
        bucket: view.length_bucket(),
    }
}

fn expected(view: &ReviewView) -> HistoricalReviewDigests {
    HistoricalReviewDigests {
        input_digest: view.digest(),
        view_seal_digest: view.seal_digest(),
        record_scope_digest: view.record_scope_digest(),
        policy_digest: view.policy_digest(),
    }
}

#[test]
fn historical_digests_match_live_view_without_pin_or_consumption_effects() {
    for mode in [
        ReviewAnchorMode::CurrentRootOnly,
        ReviewAnchorMode::AnchoredCompact,
    ] {
        let (stores, index, view) = view(ReviewSelectionPolicy {
            mode,
            max_anchors: if mode == ReviewAnchorMode::CurrentRootOnly {
                1
            } else {
                4
            },
            ..ReviewSelectionPolicy::default()
        });
        let source = sources(&stores, &view);
        let pins = stores.review_pin_count().unwrap();
        let bundle = index.bundle_for_revision(2).unwrap();
        let protected = index.bundle(bundle).unwrap().protected_consumptions();
        let actual = calculate_historical_review_digests(descriptor(&view, &source)).unwrap();
        assert_eq!(actual, expected(&view));
        verify_historical_review_digests(descriptor(&view, &source), &expected(&view)).unwrap();
        assert_eq!(stores.review_pin_count().unwrap(), pins);
        assert_eq!(
            index.bundle(bundle).unwrap().protected_consumptions(),
            protected
        );
        assert!(!view.consumed.load(Ordering::Acquire));
    }
}

fn rebased_view() -> (PalsStores, ReviewView) {
    let mut stores = PalsStores::new(StoreLimits::default());
    let start = Position::startpos();
    let situation = stores.focus_actual_moves(start.snapshot()).unwrap();
    let origin = stores.situations.get(situation).unwrap().state;
    let mut index = ReviewContextIndex::default();
    index
        .publish(&stores, &record(origin, 1, None), None)
        .unwrap();
    let mut actual = start;
    actual.make_uci("e2e4").unwrap();
    actual.make_uci("e7e5").unwrap();
    let situation = stores.focus_actual_moves(actual.snapshot()).unwrap();
    let current = stores.situations.get(situation).unwrap().state;
    let view = index
        .select(
            &stores,
            current,
            ReviewQuerySequences {
                purpose: RoleQueryPurpose::ProposePolicy,
                prefix: &[],
                proposal: &[],
                counterexample: None,
                legal: &[],
                divergences: &[],
            },
            ReviewSelectionPolicy::default(),
        )
        .unwrap();
    (stores, view)
}

#[test]
fn source_rebase_uses_exact_history_and_does_not_add_actual_moves_to_tokens() {
    let (stores, view) = rebased_view();
    let source = sources(&stores, &view);
    assert_eq!(view.token_count(), 1);
    assert_eq!(view.records_metadata()[0].source_line_offset, 2);
    assert_eq!(
        calculate_historical_review_digests(descriptor(&view, &source)).unwrap(),
        expected(&view)
    );
    let mut changed = view.records_metadata().to_vec();
    changed[0].source_line_offset = 1;
    let mut input = descriptor(&view, &source);
    input.metadata = &changed;
    assert!(calculate_historical_review_digests(input).is_err());
}

#[test]
fn fen_only_source_cannot_replace_original_history() {
    let (stores, view) = rebased_view();
    let mut source = sources(&stores, &view);
    source[0].snapshot = Position::from_fen(&source[0].snapshot.to_fen())
        .unwrap()
        .snapshot();
    assert!(calculate_historical_review_digests(descriptor(&view, &source)).is_err());
}

#[test]
fn exact_token_total_and_minimal_bucket_are_required() {
    let (stores, _, view) = view(ReviewSelectionPolicy::default());
    let source = sources(&stores, &view);
    let input = descriptor(&view, &source);
    let mut changed = input;
    changed.tokens -= 1;
    assert!(matches!(
        calculate_historical_review_digests(changed),
        Err(HistoricalReviewError::InvalidDescriptor(
            "actual repeated query and record token total"
        ))
    ));
    let mut changed = input;
    changed.bucket = 16;
    assert!(matches!(
        calculate_historical_review_digests(changed),
        Err(HistoricalReviewError::InvalidDescriptor(
            "non-minimal or unsupported physical length bucket"
        ))
    ));
    let mut changed = input;
    changed.metadata = &input.metadata[..1];
    assert!(calculate_historical_review_digests(changed).is_err());
    let mut changed = input;
    changed.source_anchors = &source[..1];
    assert!(calculate_historical_review_digests(changed).is_err());
}

#[test]
fn query_order_changes_full_input_but_not_public_record_scope() {
    let (stores, _, view) = view(ReviewSelectionPolicy::default());
    let source = sources(&stores, &view);
    let original = calculate_historical_review_digests(descriptor(&view, &source)).unwrap();
    let mut query = view.query().clone();
    query.legal.reverse();
    let mut input = descriptor(&view, &source);
    input.query = &query;
    let reordered = calculate_historical_review_digests(input).unwrap();
    assert_ne!(original.input_digest, reordered.input_digest);
    assert_eq!(original.record_scope_digest, reordered.record_scope_digest);
    assert!(matches!(
        verify_historical_review_digests(input, &original),
        Err(HistoricalReviewError::DigestMismatch("input"))
    ));
    query.legal = line("e7e5");
    let mut input = descriptor(&view, &source);
    input.query = &query;
    assert!(calculate_historical_review_digests(input).is_err());
    query.legal = view.query().legal.clone();
    query.divergences = vec![0, 0];
    let mut input = descriptor(&view, &source);
    input.query = &query;
    assert!(calculate_historical_review_digests(input).is_err());
}

#[test]
fn local_relations_cannot_skip_or_change_original_source_bundle() {
    let (stores, _, view) = view(ReviewSelectionPolicy::default());
    let source = sources(&stores, &view);
    let mut metadata = view.records_metadata().to_vec();
    metadata[1].parent = Some(1);
    let mut input = descriptor(&view, &source);
    input.metadata = &metadata;
    assert!(calculate_historical_review_digests(input).is_err());
    metadata[1].parent = Some(0);
    metadata[1].source_revision = 3;
    let mut input = descriptor(&view, &source);
    input.metadata = &metadata;
    assert!(calculate_historical_review_digests(input).is_err());
    metadata[1].source_revision = view.records_metadata()[1].source_revision;
    metadata[1].original_first_move = Some(BoardMove::from_uci("d2d4").unwrap());
    let mut input = descriptor(&view, &source);
    input.metadata = &metadata;
    assert!(calculate_historical_review_digests(input).is_err());
}

#[test]
fn compact_keeps_raw_score_metadata_out_of_model_digest_but_legacy_preserves_it() {
    for mode in [
        ReviewAnchorMode::CurrentRootOnly,
        ReviewAnchorMode::AnchoredCompact,
    ] {
        let (stores, _, view) = view(ReviewSelectionPolicy {
            mode,
            max_anchors: if mode == ReviewAnchorMode::CurrentRootOnly {
                1
            } else {
                4
            },
            ..ReviewSelectionPolicy::default()
        });
        let source = sources(&stores, &view);
        let original = calculate_historical_review_digests(descriptor(&view, &source)).unwrap();
        let mut records = view.records().to_vec();
        records[0].value = Some(-17);
        records[0].completed_depth = 7;
        records[0].critical = !records[0].critical;
        let mut input = descriptor(&view, &source);
        input.records = &records;
        let changed = calculate_historical_review_digests(input).unwrap();
        assert_eq!(
            original.input_digest == changed.input_digest,
            mode == ReviewAnchorMode::AnchoredCompact
        );
        assert_eq!(
            original.record_scope_digest == changed.record_scope_digest,
            mode == ReviewAnchorMode::AnchoredCompact
        );
    }
}

#[test]
fn source_sequence_and_generation_affect_only_original_view_seal() {
    let (stores, _, view) = view(ReviewSelectionPolicy::default());
    let source = sources(&stores, &view);
    let original = calculate_historical_review_digests(descriptor(&view, &source)).unwrap();
    for (sequence, generation) in [
        (view.sequence() + 1, view.generation()),
        (view.sequence(), view.generation() + 1),
    ] {
        let mut input = descriptor(&view, &source);
        input.sequence = sequence;
        input.generation = generation;
        let changed = calculate_historical_review_digests(input).unwrap();
        assert_eq!(original.input_digest, changed.input_digest);
        assert_eq!(original.record_scope_digest, changed.record_scope_digest);
        assert_ne!(original.view_seal_digest, changed.view_seal_digest);
    }
    let mut input = descriptor(&view, &source);
    input.sequence = 0;
    assert!(calculate_historical_review_digests(input).is_err());
}

#[test]
fn old_compact_anchor_is_valid_only_with_its_exact_source_and_local_reference() {
    let mut actual = Position::startpos();
    for movement in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        actual.make_uci(movement).unwrap();
    }
    let imported = Position::from_fen(&actual.to_fen()).unwrap();
    let mut stores = PalsStores::new(StoreLimits::default());
    let situation = stores.focus_actual_moves(actual.snapshot()).unwrap();
    let origin = stores.situations.get(situation).unwrap().state;
    let current = stores.states.insert(imported.snapshot()).unwrap();
    let mut index = ReviewContextIndex::default();
    let mut original = record(origin, 1, None);
    original.line = line("e2e4");
    index.publish(&stores, &original, None).unwrap();
    let view = index
        .select(
            &stores,
            current,
            ReviewQuerySequences {
                purpose: RoleQueryPurpose::ProposePolicy,
                prefix: &[],
                proposal: &[],
                counterexample: None,
                legal: &[],
                divergences: &[],
            },
            ReviewSelectionPolicy {
                mode: ReviewAnchorMode::AnchoredCompact,
                max_anchors: 4,
                ..ReviewSelectionPolicy::default()
            },
        )
        .unwrap();
    let source = sources(&stores, &view);
    assert_eq!(view.anchors().len(), 2);
    assert_eq!(
        calculate_historical_review_digests(descriptor(&view, &source)).unwrap(),
        expected(&view)
    );
    let mut anchors = view.anchors().to_vec();
    anchors.swap(0, 1);
    let mut input = descriptor(&view, &source);
    input.anchors = &anchors;
    assert!(calculate_historical_review_digests(input).is_err());
    let mut input = descriptor(&view, &source);
    input.policy = ReviewSelectionPolicy::default();
    assert!(calculate_historical_review_digests(input).is_err());
}
