//! Actual route utility supervision. This separate source does not change the
//! numerical utility shadow, the V task actions or the legacy R readout wire.
use super::{
    AdaptiveManagerCounters, AdaptiveManagerError, AdaptiveShutdownReceipt,
    UtilityRouteClosureProof,
};
use rz_contracts::Digest;
use rz_search::pals::engine::adaptive_port::utility_route::{
    UtilityRouteActionId, UtilityRouteRecipient, UtilityRouteSemanticOutcome,
};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const UTILITY_ROUTE_ACTION_DOMAIN: &str = "rz-rovent-utility-actual-route-action/1";
pub const UTILITY_ROUTE_PAIR_DOMAIN: &str = "rz-rovent-utility-actual-route-pair/1";
pub const UTILITY_ROUTE_COST_SCOPE: &str =
    "exclusive_submitted_global_target_route_and_descendants_with_required_drain";

/// Only the actual manager gate and its physical fence can construct this.
/// An ordinary root wall interval is deliberately insufficient.
pub struct UtilityRouteWholeCost {
    sequence: u64,
    resource_profile: Digest,
    cost_epoch: Digest,
    started: Instant,
    finished: Instant,
    wall_cost: Duration,
    baseline: AdaptiveManagerCounters,
    final_counters: AdaptiveManagerCounters,
    native_baseline: (u64, u64),
    native_final: (u64, u64),
}
impl UtilityRouteWholeCost {
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn resource_profile(&self) -> Digest {
        self.resource_profile
    }
    pub fn cost_epoch(&self) -> Digest {
        self.cost_epoch
    }
    pub fn started(&self) -> Instant {
        self.started
    }
    pub fn finished(&self) -> Instant {
        self.finished
    }
    pub fn wall_cost(&self) -> Duration {
        self.wall_cost
    }
    pub fn baseline_counters(&self) -> AdaptiveManagerCounters {
        self.baseline
    }
    pub fn final_counters(&self) -> AdaptiveManagerCounters {
        self.final_counters
    }
    pub fn native_credit_baseline(&self) -> (u64, u64) {
        self.native_baseline
    }
    pub fn native_credit_final(&self) -> (u64, u64) {
        self.native_final
    }
    pub fn cost_scope(&self) -> &'static str {
        UTILITY_ROUTE_COST_SCOPE
    }
    /// No unobserved CPU thread time or energy is inferred from wall time.
    pub fn objective(&self) -> &'static str {
        "exclusive_wall_nanoseconds"
    }
}

/// Move-only original source/semantic/cost closure for Arena publication.
pub struct CompletedUtilityRouteAction {
    semantic: UtilityRouteSemanticOutcome,
    cost: UtilityRouteWholeCost,
    group: Digest,
    digest: Digest,
}
pub struct UtilityRouteClosureRejection {
    pub semantic: UtilityRouteSemanticOutcome,
    pub proof: UtilityRouteClosureProof,
    pub error: AdaptiveManagerError,
}
impl CompletedUtilityRouteAction {
    // Rejection returns the original semantic outcome and whole-cost proof;
    // preserving those owners must not require a fresh failure allocation.
    #[allow(clippy::result_large_err)]
    pub(super) fn after_manager_shutdown(
        semantic: UtilityRouteSemanticOutcome,
        proof: UtilityRouteClosureProof,
        receipt: &AdaptiveShutdownReceipt,
    ) -> Result<Self, UtilityRouteClosureRejection> {
        let checked = (|| -> Result<Duration, AdaptiveManagerError> {
            let baseline = proof.baseline_counters();
            let final_counters = proof.final_counters();
            if !receipt.permits_semantic_closure()
                || receipt.authority != semantic.id().authority()
                || proof.action() != semantic.id()
                || proof.authority() != receipt.authority
                || proof.final_counters() != receipt.counters
                || proof.native_credit_baseline().0 > proof.native_credit_final().0
                || proof.native_credit_baseline().1 != 0
                || proof.native_credit_final() != (receipt.native_cpu_spent, 0)
                || proof.root_budget() != semantic.root_budget()
                || proof.sequence() == 0
                || proof.started() > semantic.started()
                || !semantic.semantic_consumption_complete()
                || final_counters.nn_input_count_unknown != baseline.nn_input_count_unknown
                || final_counters.accepted_nn_input_count_unknown
                    != baseline.accepted_nn_input_count_unknown
                || final_counters.consumed_nn_input_count_unknown
                    != baseline.consumed_nn_input_count_unknown
                || final_counters.native_preparations < baseline.native_preparations
                || final_counters.submitted_role_tickets < baseline.submitted_role_tickets
                || final_counters.physical_role_completions < baseline.physical_role_completions
                || final_counters.completed_nn_inputs < baseline.completed_nn_inputs
                || final_counters.accepted_nn_inputs < baseline.accepted_nn_inputs
                || final_counters.consumed_nn_inputs < baseline.consumed_nn_inputs
                || final_counters.cpu_slice_receipts < baseline.cpu_slice_receipts
                || final_counters.rejected_events != baseline.rejected_events
            {
                return Err(AdaptiveManagerError::Semantic(
                    "utility route original semantic/exclusive physical closure mismatch".into(),
                ));
            }
            proof
                .finished()
                .checked_duration_since(proof.started())
                .ok_or_else(|| {
                    AdaptiveManagerError::Semantic(
                        "utility route reversed whole-cost interval".into(),
                    )
                })
        })();
        let wall_cost = match checked {
            Ok(cost) => cost,
            Err(error) => {
                return Err(UtilityRouteClosureRejection {
                    semantic,
                    proof,
                    error,
                });
            }
        };
        let cost = UtilityRouteWholeCost {
            sequence: proof.sequence(),
            resource_profile: proof.resource_profile(),
            cost_epoch: proof.cost_epoch(),
            started: proof.started(),
            finished: proof.finished(),
            wall_cost,
            baseline: proof.baseline_counters(),
            final_counters: proof.final_counters(),
            native_baseline: proof.native_credit_baseline(),
            native_final: proof.native_credit_final(),
        };
        let group = comparison_group(&semantic, &cost);
        let digest = action_digest(&semantic, &cost, group);
        Ok(Self {
            semantic,
            cost,
            group,
            digest,
        })
    }
    pub fn id(&self) -> &UtilityRouteActionId {
        self.semantic.id()
    }
    pub fn semantic(&self) -> &UtilityRouteSemanticOutcome {
        &self.semantic
    }
    pub fn cost(&self) -> &UtilityRouteWholeCost {
        &self.cost
    }
    pub fn comparison_group(&self) -> Digest {
        self.group
    }
    pub fn digest(&self) -> Digest {
        self.digest
    }
    /// Only the actually executed, accepted and physically closed route is
    /// supported here. Mode capability does not claim three available offers.
    pub fn support_mask(&self) -> [bool; 7] {
        let mut support = [false; 7];
        support[self.id().recipient().output_index()] = true;
        support
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UtilityRouteUnknownReason {
    DifferentFrozenGroup,
    DuplicateAction,
    SameRecipient,
    NoNewValidWitness,
    EqualObservation,
    ParetoIncomparable,
}
impl UtilityRouteUnknownReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DifferentFrozenGroup => "different_frozen_group",
            Self::DuplicateAction => "duplicate_actual_action",
            Self::SameRecipient => "same_recipient",
            Self::NoNewValidWitness => "no_new_valid_witness",
            Self::EqualObservation => "equal_observation",
            Self::ParetoIncomparable => "pareto_incomparable",
        }
    }
}
/// A pair owns only Arc references to already closed actual actions. No raw DTO
/// can be converted into either source capability by this API.
pub struct UtilityRouteComparison {
    left: Arc<CompletedUtilityRouteAction>,
    right: Arc<CompletedUtilityRouteAction>,
    preferred: Option<UtilityRouteRecipient>,
    unknown: Option<UtilityRouteUnknownReason>,
    digest: Digest,
}
impl UtilityRouteComparison {
    pub fn compare(
        left: Arc<CompletedUtilityRouteAction>,
        right: Arc<CompletedUtilityRouteAction>,
    ) -> Self {
        let left_witnesses: BTreeSet<_> = left
            .semantic
            .witnesses()
            .iter()
            .map(|w| w.identity().0)
            .collect();
        let right_witnesses: BTreeSet<_> = right
            .semantic
            .witnesses()
            .iter()
            .map(|w| w.identity().0)
            .collect();
        let (preferred, unknown) = if left.id() == right.id() || left.digest == right.digest {
            (None, Some(UtilityRouteUnknownReason::DuplicateAction))
        } else if left.group != right.group {
            (None, Some(UtilityRouteUnknownReason::DifferentFrozenGroup))
        } else if left.id().recipient() == right.id().recipient() {
            (None, Some(UtilityRouteUnknownReason::SameRecipient))
        } else if left_witnesses.is_empty() && right_witnesses.is_empty() {
            (None, Some(UtilityRouteUnknownReason::NoNewValidWitness))
        } else {
            match witness_cost_order(
                &left_witnesses,
                left.cost.wall_cost,
                &right_witnesses,
                right.cost.wall_cost,
            ) {
                Some(std::cmp::Ordering::Less) => (Some(left.id().recipient()), None),
                Some(std::cmp::Ordering::Greater) => (Some(right.id().recipient()), None),
                Some(std::cmp::Ordering::Equal) => {
                    (None, Some(UtilityRouteUnknownReason::EqualObservation))
                }
                None => (None, Some(UtilityRouteUnknownReason::ParetoIncomparable)),
            }
        };
        let mut hash = Sha256::new();
        hash.update(UTILITY_ROUTE_PAIR_DOMAIN.as_bytes());
        hash.update([0]);
        hash.update(left.digest.0);
        hash.update(right.digest.0);
        hash.update([preferred.map_or(255, |p| p.output_index() as u8)]);
        hash.update([unknown.map_or(255, unknown_code)]);
        Self {
            left,
            right,
            preferred,
            unknown,
            digest: Digest(hash.finalize().into()),
        }
    }
    pub fn left(&self) -> &Arc<CompletedUtilityRouteAction> {
        &self.left
    }
    pub fn right(&self) -> &Arc<CompletedUtilityRouteAction> {
        &self.right
    }
    pub fn preferred(&self) -> Option<UtilityRouteRecipient> {
        self.preferred
    }
    pub fn unknown_reason(&self) -> Option<UtilityRouteUnknownReason> {
        self.unknown
    }
    pub fn digest(&self) -> Digest {
        self.digest
    }
    pub fn support_mask(&self) -> [bool; 7] {
        let mut support = [false; 7];
        support[self.left.id().recipient().output_index()] = true;
        support[self.right.id().recipient().output_index()] = true;
        support
    }
    pub fn target_known_mask(&self) -> [bool; 7] {
        if self.preferred.is_some() {
            self.support_mask()
        } else {
            [false; 7]
        }
    }
}
fn unknown_code(reason: UtilityRouteUnknownReason) -> u8 {
    match reason {
        UtilityRouteUnknownReason::DifferentFrozenGroup => 0,
        UtilityRouteUnknownReason::DuplicateAction => 1,
        UtilityRouteUnknownReason::SameRecipient => 2,
        UtilityRouteUnknownReason::NoNewValidWitness => 3,
        UtilityRouteUnknownReason::EqualObservation => 4,
        UtilityRouteUnknownReason::ParetoIncomparable => 5,
    }
}
fn comparison_group(
    semantic: &UtilityRouteSemanticOutcome,
    cost: &UtilityRouteWholeCost,
) -> Digest {
    let mut hash = Sha256::new();
    hash.update(b"rz-rovent-utility-route-comparison-group/1\0");
    hash.update(semantic.frozen_context_digest().0);
    hash.update(cost.resource_profile.0);
    hash.update(cost.cost_epoch.0);
    hash.update(UTILITY_ROUTE_COST_SCOPE.as_bytes());
    hash.update([0]);
    hash.update(cost.objective().as_bytes());
    Digest(hash.finalize().into())
}
fn action_digest(
    semantic: &UtilityRouteSemanticOutcome,
    cost: &UtilityRouteWholeCost,
    group: Digest,
) -> Digest {
    let mut hash = Sha256::new();
    hash.update(UTILITY_ROUTE_ACTION_DOMAIN.as_bytes());
    hash.update([0]);
    let id = semantic.id();
    let authority = id.authority();
    hash.update(authority.epoch.0.to_le_bytes());
    hash.update(authority.game.0.to_le_bytes());
    hash.update(authority.root.0.to_le_bytes());
    hash.update(authority.implementation.0);
    hash.update(id.frame().sequence.to_le_bytes());
    hash.update(id.question().0);
    hash.update([id.recipient().output_index() as u8]);
    hash.update(cost.sequence.to_le_bytes());
    hash.update(semantic.root_budget().0);
    hash.update(group.0);
    hash.update(semantic.input().submission_target_sha256().as_bytes());
    hash.update(cost.wall_cost.as_nanos().to_le_bytes());
    hash.update((semantic.witnesses().len() as u64).to_le_bytes());
    for witness in semantic.witnesses() {
        hash.update(witness.identity().0);
        hash.update(witness.raw_evidence().0);
    }
    hash.update((semantic.pc_consumption().len() as u64).to_le_bytes());
    for source in semantic.pc_consumption() {
        hash.update(source.origin().seal_digest().0);
    }
    hash.update((semantic.r_pc_consumption().len() as u64).to_le_bytes());
    for source in semantic.r_pc_consumption() {
        hash.update(source.native_execution().epoch.0.to_le_bytes());
        hash.update(source.native_execution().sequence.to_le_bytes());
        if let Some(ticket) = source.selected_pc_ticket() {
            hash.update(ticket.id.sequence.to_le_bytes());
        }
    }
    Digest(hash.finalize().into())
}

// A conservative partial order. Distinct witness sets are never assigned an
// invented scalar quality, and a cheaper empty action cannot erase evidence.
fn witness_cost_order(
    left: &BTreeSet<[u8; 32]>,
    left_cost: Duration,
    right: &BTreeSet<[u8; 32]>,
    right_cost: Duration,
) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;
    if left.is_empty() && right.is_empty() {
        return None;
    }
    if left == right && left_cost == right_cost {
        return Some(Ordering::Equal);
    }
    if right.is_subset(left) && left_cost <= right_cost {
        return Some(Ordering::Less);
    }
    if left.is_subset(right) && right_cost <= left_cost {
        return Some(Ordering::Greater);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    fn witnesses(ids: &[u8]) -> BTreeSet<[u8; 32]> {
        ids.iter().map(|id| [*id; 32]).collect()
    }
    #[test]
    fn new_witness_and_lower_whole_cost_define_the_same_order_in_both_directions() {
        let one = witnesses(&[1]);
        let two = witnesses(&[1, 2]);
        let fast = Duration::from_nanos(10);
        let slow = Duration::from_nanos(20);
        assert_eq!(
            witness_cost_order(&two, fast, &one, slow),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            witness_cost_order(&one, slow, &two, fast),
            Some(std::cmp::Ordering::Greater)
        );
        assert_eq!(witness_cost_order(&two, slow, &one, fast), None);
    }
    #[test]
    fn different_valid_witnesses_are_incomparable_even_when_counts_or_time_match() {
        let left = witnesses(&[1, 2]);
        let right = witnesses(&[2, 3]);
        assert_eq!(
            witness_cost_order(&left, Duration::ZERO, &right, Duration::ZERO),
            None
        );
        assert_eq!(
            witness_cost_order(
                &left,
                Duration::from_nanos(1),
                &right,
                Duration::from_nanos(99)
            ),
            None
        );
    }
    #[test]
    fn duplication_never_increases_witness_value_and_equal_observation_is_not_preference() {
        let left = witnesses(&[1, 1, 2]);
        let right = witnesses(&[1, 2]);
        assert_eq!(left, right);
        assert_eq!(
            witness_cost_order(
                &left,
                Duration::from_nanos(8),
                &right,
                Duration::from_nanos(8)
            ),
            Some(std::cmp::Ordering::Equal)
        );
    }
    #[test]
    fn cheaper_empty_actions_do_not_create_a_valid_utility_target() {
        let empty = witnesses(&[]);
        assert_eq!(
            witness_cost_order(
                &empty,
                Duration::from_nanos(1),
                &empty,
                Duration::from_nanos(99)
            ),
            None
        );
        assert_eq!(
            witness_cost_order(
                &empty,
                Duration::ZERO,
                &witnesses(&[1]),
                Duration::from_nanos(99)
            ),
            None
        );
    }
    #[test]
    fn route_axes_keep_retention_and_response_risk_out_of_action_support() {
        for recipient in [
            UtilityRouteRecipient::Proposer,
            UtilityRouteRecipient::Critic,
            UtilityRouteRecipient::Researcher,
        ] {
            assert!((4..=6).contains(&recipient.output_index()));
        }
    }
    #[test]
    fn witness_cost_dominance_is_transitive_for_actual_comparison_metrics() {
        let first = witnesses(&[1, 2, 3]);
        let second = witnesses(&[1, 2]);
        let third = witnesses(&[1]);
        assert_eq!(
            witness_cost_order(
                &first,
                Duration::from_nanos(1),
                &second,
                Duration::from_nanos(2)
            ),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            witness_cost_order(
                &second,
                Duration::from_nanos(2),
                &third,
                Duration::from_nanos(3)
            ),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            witness_cost_order(
                &first,
                Duration::from_nanos(1),
                &third,
                Duration::from_nanos(3)
            ),
            Some(std::cmp::Ordering::Less)
        );
    }
}
