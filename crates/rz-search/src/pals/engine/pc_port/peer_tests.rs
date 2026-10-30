//! Authored native receipts verify actual peer provenance without running NN.
//! This file is registered under the existing goal fixture module.
use super::*;
use crate::pals::coordinator::AcceptedRoleOriginKind;
use rz_contracts::rovent::SearchRecipient;

#[test]
fn pc_peer_exchange_preserves_actual_parent_question_view_and_original_budget() {
    let roots = [BoardMove::from_uci("e2e4").unwrap()];
    let mut port = goal_port(
        Position::startpos(),
        SearchGoal {
            depth_plies: NonZeroU16::new(5),
            mate_moves: None,
        },
        Some(&roots),
    );
    let root = port.next_role().unwrap().unwrap();
    assert!(root.question.reciprocal_question().is_none());
    let (accepted_root, output_root) = accepted(&mut port, &root, 1);
    port.consume_accepted(
        &root,
        &accepted_root,
        &output_root,
        Some(evidence(&port, &root, 1)),
    )
    .unwrap();
    let ready_after_root = port.ready.len();
    port.consume_accepted(
        &root,
        &accepted_root,
        &output_root,
        Some(evidence(&port, &root, 1)),
    )
    .unwrap();
    assert_eq!(port.ready.len(), ready_after_root);
    let counter = port.next_role().unwrap().unwrap();
    let first = counter.question.reciprocal_question().unwrap();
    assert_eq!(first.sender, SearchRecipient::Proposer);
    assert_eq!(first.recipient, SearchRecipient::Critic);
    assert!(first.parent.is_none());
    assert_eq!(first.view, counter.view_seal);
    assert_eq!(first.question_digest, counter.question_digest);
    assert_eq!(first.anchor_root_ply.0, 1);
    assert_eq!(Some(first.root_budget), port.original_root_budget());
    assert_eq!(first.original_deadline, port.deadline);
    let (accepted_counter, output_counter) = accepted(&mut port, &counter, 2);
    let origin = accepted_counter
        .origin(counter.question.frontier_frame(), Some(first))
        .unwrap();
    assert_eq!(origin.kind(), AcceptedRoleOriginKind::PeerQuestion(first));
    port.consume_accepted(
        &counter,
        &accepted_counter,
        &output_counter,
        Some(evidence(&port, &counter, 2)),
    )
    .unwrap();
    let repair = port.next_role().unwrap().unwrap();
    let second = repair.question.reciprocal_question().unwrap();
    assert_eq!(second.sender, SearchRecipient::Critic);
    assert_eq!(second.recipient, SearchRecipient::Proposer);
    assert_eq!(second.parent, Some(first.id));
    assert_eq!(second.view, repair.view_seal);
    assert_ne!(second.view, first.view);
    assert_eq!(second.original_first_move, first.original_first_move);
    assert_eq!(second.root_budget, first.root_budget);
    assert_eq!(second.original_deadline, first.original_deadline);
    assert_eq!(second.anchor_root_ply.0, 2);
    assert!(second.id.sequence > first.id.sequence);
    let before = port.counters;
    assert!(
        port.consume_accepted(
            &repair,
            &accepted_counter,
            &output_counter,
            Some(evidence(&port, &counter, 2)),
        )
        .is_err()
    );
    assert_eq!(port.counters, before);
    assert!(port.pending.contains_key(&repair.question_digest));
    assert_eq!(std::mem::size_of_val(port.researcher()), 0);
    assert!(
        port.owner
            .semantic
            .nodes
            .iter()
            .all(|node| node.resume.is_none())
    );
}
