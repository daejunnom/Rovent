//! Opt-in scalar readback over the SAME private foreign entries and opaque
//! role journal. No decoder, work/Store/view owner or acceptance constructor.
use super::*;
use crate::pals::coordinator::{FreshPairSeal, StepEvidence, StepExecution};
use rz_contracts::pals_adaptive::ReviewViewSeal;
use rz_contracts::{Deadline, RequestId};

const MAX_CHAIN_SCAN: usize = 4096;

#[derive(Clone, Copy, Debug)]
pub(in crate::pals::engine::adaptive_port) struct ProfiledFeedbackRoleEdge {
    pub evidence: StepEvidence,
    pub request: RequestId,
    pub role: Role,
    pub ticket_question: ContractDigest,
    pub ticket_view: ReviewViewSeal,
    pub namespace: Option<ContractDigest>,
    pub original_deadline: Deadline,
    pub prefix_before_plies: usize,
    pub counter_source: Option<ObservationId>,
    pub published_repair_line: Option<AdaptivePublishedRepairLine>,
    pub repair_input: Option<AdaptiveRepairLineProvenance>,
    pub fresh_pair: Option<FreshPairSeal>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfiledFeedbackObservationUnavailable {
    NotOptedIn,
    JournalIncomplete,
    OriginalSourceUnavailable,
    NoNonzeroRepair,
    MissingCriticOrFresh,
    ScanCapacity,
}

/// Copied facts only. Construction stays inside the same actual Port; a
/// caller cannot submit this value to any worker/machine as a permission.
#[derive(Clone, Copy, Debug)]
pub struct ProfiledFeedbackConsumedChain {
    steps: [StepEvidence; 5],
    foreign_observation: ObservationId,
    repair: AdaptiveRepairLineProvenance,
    prefix_before_plies: usize,
    namespace: ContractDigest,
    fresh_pair: FreshPairSeal,
}
impl ProfiledFeedbackConsumedChain {
    /// Source R, nonzero P Repair, C recheck, repaired Fresh, counter Fresh.
    pub fn steps(&self) -> &[StepEvidence; 5] {
        &self.steps
    }
    pub fn foreign_observation(&self) -> ObservationId {
        self.foreign_observation
    }
    pub fn repair(&self) -> AdaptiveRepairLineProvenance {
        self.repair
    }
    pub fn prefix_before_plies(&self) -> usize {
        self.prefix_before_plies
    }
    pub fn namespace(&self) -> ContractDigest {
        self.namespace
    }
    pub fn fresh_pair(&self) -> FreshPairSeal {
        self.fresh_pair
    }
}
#[derive(Clone, Copy, Debug)]
pub struct ProfiledFeedbackConsumptionObservation {
    authority: SearchAuthority,
    chain: Option<ProfiledFeedbackConsumedChain>,
    unavailable: Option<ProfiledFeedbackObservationUnavailable>,
}
impl ProfiledFeedbackConsumptionObservation {
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn chain(&self) -> Option<&ProfiledFeedbackConsumedChain> {
        self.chain.as_ref()
    }
    pub fn unavailable(&self) -> Option<ProfiledFeedbackObservationUnavailable> {
        self.unavailable
    }
    fn missing(authority: SearchAuthority, why: ProfiledFeedbackObservationUnavailable) -> Self {
        Self {
            authority,
            chain: None,
            unavailable: Some(why),
        }
    }
}

fn exact_role_consumption(
    authority: SearchAuthority,
    journal: &AdaptiveRoleConsumptionObservation,
    edge: &ProfiledFeedbackRoleEdge,
) -> bool {
    if edge.evidence.frame.authority != authority
        || edge.ticket_view.authority != authority
        || edge.ticket_view != edge.evidence.submitted_view
        || edge.ticket_question != edge.evidence.submitted_question
        || edge.evidence.execution != StepExecution::Role(edge.request)
        || edge.namespace.is_none()
    {
        return false;
    }
    let common = |e: &&AdaptiveRoleConsumptionEvent| {
        e.authority == authority
            && e.request == edge.request
            && e.role == edge.role
            && e.question == edge.ticket_question
            && e.submitted_question == Some(edge.evidence.submitted_question)
            && e.ticket_view == edge.ticket_view.view_digest
            && e.submitted_view == edge.evidence.submitted_view.view_digest
    };
    let semantic = journal
        .events()
        .iter()
        .filter(common)
        .filter(|e| {
            e.stage == AdaptiveRoleConsumptionStage::SemanticCommit
                && e.outcome == AdaptiveRoleConsumptionOutcome::SemanticResult
                && e.frame == Some(edge.evidence.frame)
                && e.completed_evidence == Some(edge.evidence.completed_evidence)
                && e.prior_semantic_attempt == Some(false)
        })
        .count();
    let consumed = journal
        .events()
        .iter()
        .filter(common)
        .filter(|e| {
            e.stage == AdaptiveRoleConsumptionStage::ViewConsumption
                && e.outcome == AdaptiveRoleConsumptionOutcome::ConsumedView
        })
        .count();
    semantic == 1 && consumed == 1
}
fn lineage(edge: &ProfiledFeedbackRoleEdge, source: AdaptiveRepairLineProvenance) -> bool {
    edge.repair_input == Some(source)
}
fn same_root_input(edge: &ProfiledFeedbackRoleEdge, original: ReviewViewSeal) -> bool {
    edge.evidence.submitted_view.authority == original.authority
        && edge.evidence.submitted_view.root_state == original.root_state
        && edge.evidence.submitted_view.context_policy == original.context_policy
}
fn nonzero_repair(edge: &ProfiledFeedbackRoleEdge) -> Option<AdaptivePublishedRepairLine> {
    let line = edge.published_repair_line?;
    (edge.role == Role::Proposer
        && edge.evidence.stage == RefinementStage::Repair
        && line.committed_plies.checked_sub(edge.prefix_before_plies) == Some(1))
    .then_some(line)
}
fn spend(left: &mut usize) -> bool {
    if *left == 0 {
        return false;
    }
    *left -= 1;
    true
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Read once before the Port/opaque journal is returned or retired. This
    /// does not assert a coordinator acceptance, root fence or durable commit;
    /// the original UCI diagnostic must separately join those actual facts.
    pub fn profiled_feedback_consumption_observation(
        &self,
    ) -> ProfiledFeedbackConsumptionObservation {
        use ProfiledFeedbackObservationUnavailable as U;
        let missing = |why| ProfiledFeedbackConsumptionObservation::missing(self.authority, why);
        let Some(journal) = self.role_consumption_observation.as_ref() else {
            return missing(U::NotOptedIn);
        };
        let Some(edges) = journal.feedback_roles() else {
            return missing(U::JournalIncomplete);
        };
        if journal.authority() != self.authority || edges.len() > MAX_ROLE_CONSUMPTION_EVENTS {
            return missing(U::JournalIncomplete);
        }
        if self.profiled_external.entries.is_empty()
            || self
                .profiled_external
                .protected_pins(&self.engine.semantic.stores)
                .is_err()
        {
            return missing(U::OriginalSourceUnavailable);
        }
        let Some(namespace) = self
            .model_namespace()
            .ok()
            .map(|v| v.utility_cohort_digest())
        else {
            return missing(U::OriginalSourceUnavailable);
        };
        let mut eligible = [false; MAX_ROLE_CONSUMPTION_EVENTS];
        for (i, edge) in edges.iter().enumerate() {
            eligible[i] = edge.namespace == Some(namespace)
                && exact_role_consumption(self.authority, journal, edge);
        }
        let mut left = MAX_CHAIN_SCAN;
        let mut nonzero_repair_seen = false;
        for (p_index, p) in edges.iter().enumerate() {
            if !eligible[p_index]
                || p.role != Role::Proposer
                || p.evidence.stage != RefinementStage::Repair
            {
                continue;
            }
            let Some(published) = nonzero_repair(p) else {
                continue;
            };
            nonzero_repair_seen = true;
            let Some(observation) = p.counter_source else {
                continue;
            };
            let Some(admission) = self.profiled_external.entries.values().find(|a| {
                a.consumed == Some(observation)
                    && a.published == Some(observation)
                    && a.feedback_published
                    && a.terminal
                    && a.accounted
                    && a.publication_error.is_none()
                    && a.diagnostic_step_evidence.is_some()
                    && a.source.authority() == self.authority
                    && a.work.frame.authority == self.authority
                    && a.work.input.view_seal.authority == self.authority
                    && published
                        .first_move
                        .and_then(|m| rz_contracts::Move::try_from(m).ok())
                        .map(rz_contracts::pals::Move16::encode)
                        == Some(a.work.original_first_move)
            }) else {
                continue;
            };
            let original_root_view = admission.work.input.view_seal;
            if !same_root_input(p, original_root_view) {
                continue;
            }
            let r = StepEvidence {
                frame: admission.work.frame,
                stage: admission.work.input.stage,
                execution: StepExecution::ProfiledExternal {
                    question: admission.job.question.id,
                    execution: admission.job.execution,
                },
                submitted_question: admission.work.input.question_digest,
                submitted_view: admission.work.input.view_seal,
                completed_evidence: admission
                    .diagnostic_step_evidence
                    .expect("checked original result"),
            };
            let provenance = AdaptiveRepairLineProvenance {
                producer_frame: p.evidence.frame,
                producer_request: p.request,
                submitted_question: p.evidence.submitted_question,
                submitted_view: p.evidence.submitted_view.view_digest,
                completed_evidence: p.evidence.completed_evidence,
                original_deadline: p.original_deadline,
                published_line: published,
            };
            for (c_index, c) in edges.iter().enumerate().skip(p_index + 1) {
                if !spend(&mut left) {
                    return missing(U::ScanCapacity);
                }
                if !eligible[c_index]
                    || c.role != Role::Critic
                    || c.evidence.stage != RefinementStage::CriticRecheck
                    || !lineage(c, provenance)
                    || !same_root_input(c, original_root_view)
                {
                    continue;
                }
                for (first_index, first) in edges.iter().enumerate().skip(c_index + 1) {
                    if !spend(&mut left) {
                        return missing(U::ScanCapacity);
                    }
                    if !eligible[first_index]
                        || first.evidence.stage != RefinementStage::FreshRepairedEndpoint
                        || !lineage(first, provenance)
                        || !same_root_input(first, original_root_view)
                    {
                        continue;
                    }
                    let Some(pair) = first.fresh_pair else {
                        continue;
                    };
                    if pair.authority != self.authority
                        || !pair.validates_public_view(
                            &first.evidence.submitted_view,
                            pair.public_context,
                        )
                    {
                        continue;
                    }
                    for (second_index, second) in edges.iter().enumerate().skip(first_index + 1) {
                        if !spend(&mut left) {
                            return missing(U::ScanCapacity);
                        }
                        if !eligible[second_index]
                            || second.evidence.stage != RefinementStage::FreshCounterEndpoint
                            || !lineage(second, provenance)
                            || !same_root_input(second, original_root_view)
                            || second.fresh_pair != Some(pair)
                            || !pair.validates_public_view(
                                &second.evidence.submitted_view,
                                pair.public_context,
                            )
                        {
                            continue;
                        }
                        return ProfiledFeedbackConsumptionObservation {
                            authority: self.authority,
                            unavailable: None,
                            chain: Some(ProfiledFeedbackConsumedChain {
                                steps: [r, p.evidence, c.evidence, first.evidence, second.evidence],
                                foreign_observation: observation,
                                repair: provenance,
                                prefix_before_plies: p.prefix_before_plies,
                                namespace,
                                fresh_pair: pair,
                            }),
                        };
                    }
                }
            }
        }
        missing(if nonzero_repair_seen {
            U::MissingCriticOrFresh
        } else {
            U::NoNonzeroRepair
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::{
        ClockDomain, Digest, GameGeneration, MonotonicTick, OwnerId, ProcessEpoch, RootGeneration,
        StateIdentity, StateRevision,
    };
    fn authority() -> SearchAuthority {
        SearchAuthority {
            epoch: ProcessEpoch(11),
            game: GameGeneration(2),
            root: RootGeneration(3),
            implementation: Digest([1; 32]),
        }
    }
    fn edge() -> ProfiledFeedbackRoleEdge {
        let authority = authority();
        let view = ReviewViewSeal {
            authority,
            view_sequence: 7,
            context_policy: Digest([2; 32]),
            view_digest: Digest([3; 32]),
            root_state: StateIdentity {
                owner: OwnerId(1),
                revision: StateRevision(1),
                semantic: Digest([4; 32]),
            },
            actual_tokens: 128,
            selected_records: 1,
            anchors: 1,
            length_bucket: 32,
        };
        let request = RequestId::new(authority.epoch, 9);
        ProfiledFeedbackRoleEdge {
            evidence: StepEvidence {
                frame: crate::pals::coordinator::FrameId {
                    authority,
                    sequence: 5,
                },
                stage: RefinementStage::Repair,
                execution: StepExecution::Role(request),
                submitted_question: Digest([5; 32]),
                submitted_view: view,
                completed_evidence: Digest([6; 32]),
            },
            request,
            role: Role::Proposer,
            ticket_question: Digest([5; 32]),
            ticket_view: view,
            namespace: Some(Digest([7; 32])),
            original_deadline: Deadline {
                clock: ClockDomain(authority.epoch),
                at: MonotonicTick(100000),
            },
            prefix_before_plies: 3,
            counter_source: Some(ObservationId(2)),
            published_repair_line: Some(AdaptivePublishedRepairLine {
                line: LineId(1),
                observation: ObservationId(3),
                record_revision: 4,
                parent_revision: Some(2),
                supersedes_revision: None,
                origin_state: StateId(1),
                committed_plies: 4,
                first_move: None,
            }),
            repair_input: None,
            fresh_pair: None,
        }
    }
    fn journal(edge: ProfiledFeedbackRoleEdge) -> AdaptiveRoleConsumptionObservation {
        let mut journal = AdaptiveRoleConsumptionObservation::new(authority());
        let mut event = AdaptiveRoleConsumptionEvent {
            authority: authority(),
            request: edge.request,
            role: edge.role,
            question: edge.ticket_question,
            submitted_question: Some(edge.evidence.submitted_question),
            ticket_view: edge.ticket_view.view_digest,
            submitted_view: edge.evidence.submitted_view.view_digest,
            frame: Some(edge.evidence.frame),
            stage: AdaptiveRoleConsumptionStage::SemanticCommit,
            outcome: AdaptiveRoleConsumptionOutcome::SemanticResult,
            completed_evidence: Some(edge.evidence.completed_evidence),
            prior_semantic_attempt: Some(false),
            published_repair_line: edge.published_repair_line,
            repair_line_input: None,
        };
        journal.record(event);
        event.stage = AdaptiveRoleConsumptionStage::ViewConsumption;
        event.outcome = AdaptiveRoleConsumptionOutcome::ConsumedView;
        event.frame = None;
        event.completed_evidence = None;
        event.prior_semantic_attempt = None;
        journal.record(event);
        journal
    }
    #[test]
    fn original_role_join_rejects_old_go_and_stale_view() {
        let original = edge();
        let journal = journal(original);
        assert!(exact_role_consumption(authority(), &journal, &original));
        let mut old_go = original;
        old_go.evidence.frame.authority.root = RootGeneration(2);
        assert!(!exact_role_consumption(authority(), &journal, &old_go));
        let mut stale = original;
        stale.evidence.submitted_view.view_sequence += 1;
        stale.evidence.submitted_view.view_digest = Digest([99; 32]);
        assert!(!exact_role_consumption(authority(), &journal, &stale));
    }
    #[test]
    fn zero_step_or_missing_repair_is_not_nonzero_p_work() {
        let original = edge();
        assert!(nonzero_repair(&original).is_some());
        let mut zero = original;
        zero.published_repair_line.as_mut().unwrap().committed_plies = zero.prefix_before_plies;
        assert!(nonzero_repair(&zero).is_none());
        zero.published_repair_line = None;
        assert!(nonzero_repair(&zero).is_none());
    }
    #[test]
    fn duplicate_semantic_commit_does_not_make_an_extra_accepted_chain() {
        let original = edge();
        let mut journal = journal(original);
        journal.record(journal.events()[0]);
        assert!(!exact_role_consumption(authority(), &journal, &original));
    }
}
