use super::*;
use crate::pals::coordinator::test_support::{
    authority, confirm_mock_native_acceptance, deadline, key,
};
use crate::pals::coordinator::{RoleSubmission, TicketPhase};
use crate::pals::review_context::{
    ReviewContextIndex, ReviewQuerySequences, ReviewSelectionPolicy,
};
use crate::pals::store::{PalsStores, StoreLimits};
use rz_contracts::OwnerId;
use rz_contracts::pals_adaptive::ReviewViewSeal;
use rz_position::Position;
use rz_position::contracts::ContractPosition;
use rz_runtime::adaptive::CoordinatorPolicy;

fn submitted(
    purpose: RoleQueryPurpose,
) -> (
    RoleTicketRegistry<ReviewView, RoleOutput>,
    RoleTicketEnvelope,
    RoleOutput,
) {
    let position = Position::startpos();
    let rules = ContractPosition::new(OwnerId(17), position.clone())
        .export()
        .unwrap();
    let mut stores = PalsStores::new(StoreLimits::default());
    let situation = stores.focus_actual_moves(position.snapshot()).unwrap();
    let state = stores.situations.get(situation).unwrap().state;
    let view = Arc::new(
        ReviewContextIndex::default()
            .select(
                &stores,
                state,
                ReviewQuerySequences {
                    purpose,
                    prefix: &[],
                    proposal: &[],
                    counterexample: None,
                    legal: &position.legal_moves(),
                    divergences: &[],
                },
                ReviewSelectionPolicy::default(),
            )
            .unwrap(),
    );
    let seal = ReviewViewSeal {
        authority: authority(),
        view_sequence: view.sequence(),
        context_policy: Digest(view.policy_digest()),
        view_digest: Digest(view.seal_digest()),
        root_state: rules.snapshot().identity(),
        actual_tokens: view.token_count(),
        selected_records: view.records().len() as u16,
        anchors: view.anchors().len() as u8,
        length_bucket: view.length_bucket(),
    };
    let mut registry =
        RoleTicketRegistry::new(authority(), 512, CoordinatorPolicy::Fixed, 1).unwrap();
    let ticket = registry
        .admit(
            RoleSubmission {
                question: Digest(view.digest()),
                view: seal,
                representation: key(Role::Validator),
                role: Role::Validator,
                original_deadline: deadline(),
            },
            view,
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    registry.take_dispatch().unwrap();
    let output = RoleOutput {
        id: ticket.id,
        authority: ticket.authority,
        key: ticket.representation,
        payload: RolePayload::TaskRanking {
            task_logits: vec![0., 1., 2., 3., 4., 5., 6.],
        },
        private_latent: vec![0.; 16 * 384],
    };
    (registry, ticket, output)
}

#[test]
fn original_head_survives_raw_delivery_and_accepts_once() {
    let (mut registry, ticket, output) = submitted(RoleQueryPurpose::TaskPolicy);
    registry
        .complete_validator_task(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(output),
        })
        .unwrap();
    let mut ready = registry
        .take_ready(deadline().clock, MonotonicTick(0))
        .unwrap();
    if let RolePayload::TaskRanking { task_logits } = &mut ready.result.as_mut().unwrap().payload {
        task_logits.fill(999.);
    }
    confirm_mock_native_acceptance(&mut registry, &ticket);
    let proof = registry
        .accept_validator_task(&ticket, deadline().clock, MonotonicTick(0))
        .unwrap();
    assert_eq!(proof.task_logits(), &[0., 1., 2., 3., 4., 5., 6.]);
    assert_eq!(proof.ticket(), &ticket);
    assert_eq!(
        proof.source().view_input_digest,
        Digest(proof.view().digest())
    );
    assert!(
        registry
            .accept_validator_task(&ticket, deadline().clock, MonotonicTick(0))
            .is_err()
    );
    assert_eq!(registry.occupancy(), 0);
}

#[test]
fn generic_completion_and_wrong_query_cannot_publish_validator_authority() {
    let (mut registry, ticket, output) = submitted(RoleQueryPurpose::TaskPolicy);
    registry
        .complete(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(output),
        })
        .unwrap();
    registry
        .take_ready(deadline().clock, MonotonicTick(0))
        .unwrap();
    assert_eq!(
        registry
            .accept_validator_task(&ticket, deadline().clock, MonotonicTick(0))
            .unwrap_err(),
        CoordinatorError::MissingEvidence
    );
    let (mut registry, ticket, output) = submitted(RoleQueryPurpose::ProposePolicy);
    registry
        .complete_validator_task(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(output),
        })
        .unwrap();
    registry
        .take_ready(deadline().clock, MonotonicTick(0))
        .unwrap();
    confirm_mock_native_acceptance(&mut registry, &ticket);
    assert!(
        registry
            .accept_validator_task(&ticket, deadline().clock, MonotonicTick(0))
            .is_err()
    );
}

#[test]
fn malformed_physical_unknown_still_retains_data_and_closes_admission() {
    let (mut registry, ticket, mut output) = submitted(RoleQueryPurpose::TaskPolicy);
    output.id.sequence += 1;
    output.payload = RolePayload::TaskRanking {
        task_logits: vec![f32::NAN],
    };
    registry
        .complete_validator_task(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Unknown,
            result: Ok(output),
        })
        .unwrap();
    assert_eq!(registry.phase(ticket.id), Some(TicketPhase::Quarantined));
    assert_eq!(registry.physical_outstanding(), 1);
    assert!(!registry.admission_open());
    assert!(
        registry
            .accept_validator_task(&ticket, deadline().clock, MonotonicTick(0))
            .is_err()
    );
}

#[test]
fn wrong_identity_or_nonfinite_head_is_rejected_without_acceptance() {
    let (mut registry, ticket, mut output) = submitted(RoleQueryPurpose::TaskPolicy);
    output.key.frozen_epoch += 1;
    let rejected = registry
        .complete_validator_task(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(output),
        })
        .unwrap_err();
    assert_eq!(registry.phase(ticket.id), Some(TicketPhase::Submitted));
    let mut output = rejected.completion.result.unwrap();
    output.key = ticket.representation;
    output.payload = RolePayload::TaskRanking {
        task_logits: vec![f32::INFINITY; 7],
    };
    assert!(
        registry
            .complete_validator_task(RoleCompletion {
                ticket,
                physical: PhysicalRoleCompletion::Confirmed,
                result: Ok(output)
            })
            .is_err()
    );
}

#[test]
fn publication_requires_actual_registered_owner_receipt_not_just_a_permit() {
    let (mut registry, ticket, output) = submitted(RoleQueryPurpose::TaskPolicy);
    registry
        .complete_validator_task(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(output),
        })
        .unwrap();
    registry
        .take_ready(deadline().clock, MonotonicTick(0))
        .unwrap();
    assert_eq!(
        registry
            .accept_validator_task(&ticket, deadline().clock, MonotonicTick(0))
            .unwrap_err(),
        CoordinatorError::MissingEvidence
    );
    let (issuer, verifier) = super::super::NativeAcceptanceIssuer::new_pair();
    registry
        .register_native_acceptance_verifier(verifier.clone())
        .unwrap();
    assert_eq!(
        registry
            .register_native_acceptance_verifier(verifier)
            .unwrap_err(),
        CoordinatorError::Duplicate
    );
    let permit = registry
        .issue_native_acceptance(&ticket, deadline().clock, MonotonicTick(0))
        .unwrap();
    assert!(
        registry
            .issue_native_acceptance(&ticket, deadline().clock, MonotonicTick(0))
            .is_err()
    );
    assert_eq!(
        registry
            .accept_validator_task(&ticket, deadline().clock, MonotonicTick(0))
            .unwrap_err(),
        CoordinatorError::MissingEvidence
    );
    registry
        .confirm_native_acceptance(
            issuer.issue_success(permit),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap();
    assert!(
        registry
            .accept_validator_task(&ticket, deadline().clock, MonotonicTick(0))
            .is_ok()
    );
}

#[test]
fn receipt_from_another_owner_is_rejected_with_owned_data_retained() {
    let (mut registry, ticket, output) = submitted(RoleQueryPurpose::TaskPolicy);
    registry
        .complete_validator_task(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(output),
        })
        .unwrap();
    registry
        .take_ready(deadline().clock, MonotonicTick(0))
        .unwrap();
    let (_, verifier) = super::super::NativeAcceptanceIssuer::new_pair();
    let (wrong, _) = super::super::NativeAcceptanceIssuer::new_pair();
    registry
        .register_native_acceptance_verifier(verifier)
        .unwrap();
    let permit = registry
        .issue_native_acceptance(&ticket, deadline().clock, MonotonicTick(0))
        .unwrap();
    let rejected = registry
        .confirm_native_acceptance(
            wrong.issue_success(permit),
            deadline().clock,
            MonotonicTick(0),
        )
        .unwrap_err();
    assert!(matches!(
        rejected.error,
        CoordinatorError::IdentityMismatch(_)
    ));
    assert_eq!(rejected.receipt.ticket(), &ticket);
    assert_eq!(registry.phase(ticket.id), Some(TicketPhase::Delivered));
    assert!(
        registry
            .accept_validator_task(&ticket, deadline().clock, MonotonicTick(0))
            .is_err()
    );
}

#[test]
fn revoked_validator_head_permit_is_custody_only_and_keeps_its_exact_source() {
    let (mut registry, ticket, output) = submitted(RoleQueryPurpose::TaskPolicy);
    registry
        .complete_validator_task(RoleCompletion {
            ticket,
            physical: PhysicalRoleCompletion::Confirmed,
            result: Ok(output),
        })
        .unwrap();
    registry
        .take_ready(deadline().clock, MonotonicTick(0))
        .unwrap();
    let (_, verifier) = super::super::NativeAcceptanceIssuer::new_pair();
    registry
        .register_native_acceptance_verifier(verifier)
        .unwrap();
    let permit = registry
        .issue_native_acceptance(&ticket, deadline().clock, MonotonicTick(0))
        .unwrap();
    assert!(
        registry
            .validate_revoked_published_head_native_permit(&ticket, &permit)
            .is_err()
    );
    registry.revoke_all();
    let before = registry.cumulative_observation();
    registry
        .validate_revoked_published_head_native_permit(&ticket, &permit)
        .unwrap();
    assert!(
        registry
            .validate_revoked_reciprocal_native_permit(&ticket, &permit)
            .is_err()
    );
    let mut foreign = ticket;
    foreign.authority.root.0 += 1;
    assert!(
        registry
            .validate_revoked_published_head_native_permit(&foreign, &permit)
            .is_err()
    );
    let mut changed = ticket;
    changed.question = Digest([99; 32]);
    assert!(
        registry
            .validate_revoked_published_head_native_permit(&changed, &permit)
            .is_err()
    );
    assert_eq!(registry.cumulative_observation(), before);
    assert_eq!(registry.occupancy(), 1);
    assert_eq!(registry.phase(ticket.id), Some(TicketPhase::Delivered));
    assert!(
        registry
            .accept_validator_task(&ticket, deadline().clock, MonotonicTick(0))
            .is_err()
    );
}
