//! Actual manager/owner integration, with authored neural outputs only.
//! The fixture never certifies a completed PALS refinement: its finite
//! semantic scope ends as explicit Unknown after checked worker evidence.
#[path = "fixed_trace_tests.rs"]
mod fixed_trace;
#[path = "nnuec_owner_tests.rs"]
#[cfg(all(feature = "pals-adaptive", feature = "onnx-cpu"))]
mod nnuec_owner;
#[path = "reciprocal_tests.rs"]
mod reciprocal;
#[path = "reserved_replay_tests.rs"]
mod reserved_replay;
#[path = "role_only_tests.rs"]
mod role_only;
#[path = "role_only_goal_tests.rs"]
mod role_only_goal;

use super::*;
use crate::pals_adaptive::{
    AdaptiveCpuSliceDiagnosticOptions, AdaptiveCpuSliceResultObservation, AdaptivePortDriver,
    CPU_SLICE_DIAGNOSTIC_RECORDS_MAX,
};
use crate::pals_native::ticketed::{
    TicketExecutionEvidence, TicketNativeBackend, TicketPhysicalResult, TicketPreparation,
    TicketServiceLimits, TicketedNativeService,
};
use rz_contracts::pals::{Move16, RepresentationKey, RoleOutput, RolePayload, SearchAuthority};
use rz_contracts::pals_adaptive::{CpuWorkerId, ReviewViewSeal, RoleTicketEnvelope};
use rz_contracts::{
    Digest, ExecutionId, GameGeneration, OwnerId, PrecisionProfile, ProcessEpoch, RootGeneration,
    Wdl,
};
use rz_position::contracts::{ContractPosition, ContractState};
use rz_position::{BoardMove, Position};
use rz_runtime::adaptive::{ControllerConfig, CoordinatorPolicy};
use rz_runtime::contracts::ContractSystemClock;
use rz_search::cpu::{
    CpuCompletion, CpuConfig, CpuEngine, CpuLimits, CpuOrderingPolicy, CpuProfile, CpuReport,
    CpuResumePolicy,
};
use rz_search::cpu_value::BootstrapCpuValue;
use rz_search::cpu_worker::{CpuSliceReport, CpuWorker, CpuWorkerProfile};
use rz_search::external_cpu::worker::ExternalWorkerReport;
use rz_search::external_cpu::{CpuEndpointSelection, ExternalCpuConfig};
use rz_search::pals::coordinator::{
    ClosedFrame, CoordinatorWork, FrameBudget, FrameInput, FrameTransition, PhysicalRoleCompletion,
    ProposerRankingPublished, RefinementStage, RoleFailure,
};
use rz_search::pals::engine::RoleQueryPurpose;
use rz_search::pals::review_context::{
    ReviewContextIndex, ReviewQuerySequences, ReviewSelectionPolicy, ReviewView,
};
use rz_search::pals::store::{PalsStores, StateId, StoreLimits};
use sha2::Sha256;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const WINDOW: Duration = Duration::from_secs(5);
const TOTAL_NODES: u64 = 100_000;

#[derive(Default)]
struct NativeFacts {
    started: Mutex<Vec<RoleTicketEnvelope>>,
    accepted: Mutex<Vec<RoleTicketEnvelope>>,
    child_gate: AtomicBool,
    /// Fixture-only scheduling barrier for a queued first acceptance ACK:
    /// let its run finish, then hold the second run on the native owner.
    block_after_first_run: AtomicBool,
    release_first_run: AtomicBool,
    /// The queued-Accept race must observe real cancellation of run two,
    /// rather than a fixture timeout that happened to free the native owner.
    hold_second_until_cancel: AtomicBool,
    second_run_observed_cancel: AtomicBool,
    reject_acceptance: AtomicBool,
    prepared: std::sync::atomic::AtomicU64,
    quiesce_called: AtomicBool,
    finish_called: AtomicBool,
    factories: std::sync::atomic::AtomicU64,
}
struct NativeFixture {
    clock: ContractSystemClock,
    facts: Arc<NativeFacts>,
    finish_ok: bool,
}
impl<Q: Clone + Send + Sync + 'static> TicketNativeBackend<ReviewView, Q> for NativeFixture {
    type Prepared = RepresentationKey;
    type Output = RoleOutput;
    fn clock(&self) -> ContractSystemClock {
        self.clock.clone()
    }
    fn prepare(
        &mut self,
        question: &TicketPreparation<ReviewView, Q>,
        _: &AtomicBool,
    ) -> Result<(Self::Prepared, RepresentationKey), RoleFailure> {
        let mut candidates = Sha256::new();
        for movement in &question.view.query().legal {
            candidates.update(
                Move16::encode(rz_contracts::Move::try_from(*movement).unwrap())
                    .bits()
                    .to_le_bytes(),
            );
        }
        let key = RepresentationKey {
            input: Digest(question.view.digest()),
            history: question.seal.root_state.semantic,
            candidates: Digest(candidates.finalize().into()),
            mask_positions: Digest([4; 32]),
            model: Digest([5; 32]),
            encoding: Digest([6; 32]),
            precision: PrecisionProfile::Fp32,
            frozen_epoch: 1,
            record_revision: 0,
            role: question.role,
        };
        self.facts.prepared.fetch_add(1, Ordering::Release);
        Ok((key, key))
    }
    fn prepared_host_bytes(&self, _: &Self::Prepared) -> u64 {
        std::mem::size_of::<RepresentationKey>() as u64
    }
    fn run(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<ReviewView, Q>,
        _: &Self::Prepared,
        canceled: &AtomicBool,
    ) -> TicketPhysicalResult<Self::Output> {
        let started_number = {
            let mut started = self.facts.started.lock().unwrap();
            started.push(*ticket);
            started.len()
        };
        let hold_second_until_cancel =
            started_number > 1 && self.facts.hold_second_until_cancel.load(Ordering::Acquire);
        let until = Instant::now()
            + if hold_second_until_cancel {
                WINDOW
            } else {
                Duration::from_secs(2)
            };
        let block_after_first = self.facts.block_after_first_run.load(Ordering::Acquire);
        let blocked_run =
            !question.view.query().prefix.is_empty() && (!block_after_first || started_number > 1);
        while (block_after_first
            && started_number == 1
            && !self.facts.release_first_run.load(Ordering::Acquire))
            || (blocked_run && !self.facts.child_gate.load(Ordering::Acquire))
        {
            let actual_cancel = canceled.load(Ordering::Acquire);
            if actual_cancel || Instant::now() >= until {
                if hold_second_until_cancel && actual_cancel {
                    self.facts
                        .second_run_observed_cancel
                        .store(true, Ordering::Release);
                }
                return TicketPhysicalResult {
                    physical: PhysicalRoleCompletion::Confirmed,
                    output: Err(if hold_second_until_cancel && !actual_cancel {
                        RoleFailure::Expired
                    } else {
                        RoleFailure::Canceled
                    }),
                };
            }
            thread::sleep(Duration::from_millis(1));
        }
        TicketPhysicalResult {
            physical: PhysicalRoleCompletion::Confirmed,
            output: Ok(RoleOutput {
                id: ticket.id,
                authority: ticket.authority,
                key: ticket.representation,
                payload: match ticket.role {
                    rz_contracts::pals::Role::Proposer => RolePayload::Proposal {
                        candidate_logits: (0..question.view.query().legal.len())
                            .map(|i| i as f32)
                            .collect(),
                        wdl: Wdl::try_new(0.25, 0.5, 0.25, 1e-4).unwrap(),
                    },
                    rz_contracts::pals::Role::Critic => RolePayload::Counterexample {
                        candidate_logits: (0..question.view.query().legal.len())
                            .map(|i| i as f32)
                            .collect(),
                        divergence_logits: vec![0.0; question.view.query().divergences.len()],
                        wdl: Wdl::try_new(0.25, 0.5, 0.25, 1e-4).unwrap(),
                    },
                    rz_contracts::pals::Role::Validator => RolePayload::TaskRanking {
                        task_logits: vec![0.0; 7],
                    },
                },
                private_latent: vec![0.; 16 * 384],
            }),
        }
    }
    fn output_host_bytes(&self, output: &Self::Output) -> u64 {
        let payload_values = match &output.payload {
            RolePayload::Proposal {
                candidate_logits, ..
            } => candidate_logits.len(),
            RolePayload::Counterexample {
                candidate_logits,
                divergence_logits,
                ..
            } => candidate_logits.len() + divergence_logits.len(),
            RolePayload::TaskRanking { task_logits } => task_logits.len(),
        };
        ((payload_values + output.private_latent.len()) * 4) as u64
    }
    fn acceptance_output_digest(&self, output: &Self::Output) -> Option<Digest> {
        rz_search::pals::coordinator::native_acceptance_output_digest(output).ok()
    }
    fn execution_evidence(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Option<TicketExecutionEvidence> {
        Some(TicketExecutionEvidence {
            execution: Some(ExecutionId {
                epoch: ticket.id.epoch,
                sequence: ticket.id.sequence,
            }),
            // This is an authored CPU neural-service fixture, not GPU evidence.
            completed_nn_inputs: Some(1),
            measurement_complete: true,
            physical_fresh: Some(true),
            accounting: None,
        })
    }
    fn accept(
        &mut self,
        ticket: &RoleTicketEnvelope,
        question: &TicketPreparation<ReviewView, Q>,
        _: &Self::Prepared,
        output: &Self::Output,
    ) -> Result<(), RoleFailure> {
        if output.id != ticket.id
            || output.authority != ticket.authority
            || output.key != ticket.representation
            || ticket.view.view_digest.0 != question.view.seal_digest()
            || self.facts.reject_acceptance.load(Ordering::Acquire)
        {
            return Err(RoleFailure::InvalidOutput);
        }
        self.facts.accepted.lock().unwrap().push(*ticket);
        Ok(())
    }
    fn discard(&mut self, _: &RoleTicketEnvelope) -> Result<(), RoleFailure> {
        Ok(())
    }
    fn finish(&mut self, _: Instant) -> bool {
        self.facts.finish_called.store(true, Ordering::Release);
        self.finish_ok
    }
    fn quiesce(&mut self, _: Instant) -> bool {
        // This authored CPU backend has no pending physical buffers after
        // service.run returns. It is not GPU completion evidence.
        self.facts.quiesce_called.store(true, Ordering::Release);
        self.finish_ok
    }
}

#[derive(Clone, Copy)]
enum FixtureScope {
    TwoIndependentRoles,
    OwnCpuResume,
    OwnCpuReciprocal,
    PeerAllDirections,
}
struct DriverFixture {
    authority: SearchAuthority,
    role_call_limit: Option<u64>,
    rules: Arc<ContractState>,
    stores: PalsStores,
    index: ReviewContextIndex,
    state: StateId,
    scope: FixtureScope,
    consumed: Vec<RoleTicketEnvelope>,
    closed: Vec<ClosedFrame>,
    expected_cpu: BTreeMap<u16, CpuReport>,
    actual_cpu: Vec<CpuSliceReport>,
    fail_raw_once: bool,
    semantic_owner_busy: bool,
    fail_role_once: bool,
    fail_after_role_consumption_once: bool,
    fail_role_consumption_readback: bool,
    #[cfg(feature = "pals-adaptive-diagnostics")]
    fail_publish_closed_once: bool,
    raw_cpu_receipts: Vec<ExecutionId>,
    reciprocal_questions: BTreeMap<u64, rz_contracts::rovent::QuestionEnvelope>,
    reciprocal_waits: Vec<ExecutionId>,
    reciprocal_replies: usize,
    peer_sources: Vec<(Role, rz_contracts::rovent::SearchRecipient)>,
    peer_opinions: usize,
    peer_cpu_answers: usize,
    peer_repeat_question: bool,
    replay_frontier: Option<rz_search::pals::rovent_frontier::RoventFrontier>,
    block_replay_while_reserved: bool,
    blocked_replays: usize,
}
impl DriverFixture {
    fn input(
        &mut self,
        stage: RefinementStage,
        first: Option<BoardMove>,
    ) -> FrameInput<ReviewView, u64> {
        let mut endpoint = Position::from_snapshot(
            self.rules.rules().snapshot(),
            rz_position::PositionLimits::default(),
        )
        .unwrap();
        let divergence = matches!(self.scope, FixtureScope::PeerAllDirections)
            && stage == RefinementStage::CriticDivergence;
        let prefix = if divergence {
            Vec::new()
        } else {
            first.into_iter().collect::<Vec<_>>()
        };
        let proposal = if divergence {
            let first = first.expect("actual first move for divergence fixture");
            let mut branch = endpoint.clone();
            branch.make_move(first).unwrap();
            let mut proposal = vec![first];
            let additional = if self.peer_repeat_question { 1 } else { 3 };
            for _ in 0..additional {
                let movement = branch.legal_moves()[0];
                branch.make_move(movement).unwrap();
                proposal.push(movement);
            }
            proposal
        } else {
            Vec::new()
        };
        let divergences: Vec<usize> = if divergence {
            vec![if self.peer_repeat_question { 1 } else { 3 }]
        } else {
            Vec::new()
        };
        for movement in &prefix {
            endpoint.make_move(*movement).unwrap();
        }
        let legal = if divergence {
            Vec::new()
        } else {
            endpoint.legal_moves()
        };
        let view = Arc::new(
            self.index
                .select(
                    &self.stores,
                    self.state,
                    ReviewQuerySequences {
                        purpose: if divergence {
                            RoleQueryPurpose::DivergencePolicy
                        } else {
                            RoleQueryPurpose::ProposePolicy
                        },
                        prefix: &prefix,
                        proposal: &proposal,
                        counterexample: None,
                        legal: &legal,
                        divergences: &divergences,
                    },
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let view_seal = ReviewViewSeal {
            authority: self.authority,
            view_sequence: view.sequence(),
            context_policy: Digest(view.policy_digest()),
            view_digest: Digest(view.seal_digest()),
            root_state: self.rules.snapshot().identity(),
            actual_tokens: view.token_count(),
            selected_records: view.records().len() as u16,
            anchors: view.anchors().len() as u8,
            length_bucket: view.length_bucket(),
        };
        let query = first.map_or(0, |mv| {
            u64::from(Move16::encode(rz_contracts::Move::try_from(mv).unwrap()).bits())
        });
        FrameInput {
            stage,
            question_digest: Digest(view.digest()),
            question: Arc::new(query),
            public_context: Digest(view.record_scope_digest()),
            view_seal,
            view,
        }
    }
}
impl AdaptiveSearchDriver<u64> for DriverFixture {
    fn semantic_owner_available(&self) -> bool {
        !self.semantic_owner_busy
    }
    fn frontier_native_capacity(
        &mut self,
        requested: u64,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<Option<u64>, AdaptiveManagerError> {
        self.replay_frontier
            .as_mut()
            .map(|frontier| frontier.native_capacity(requested, clock, now))
            .transpose()
            .map_err(AdaptiveManagerError::Frontier)
    }
    fn frontier_native_reserve(
        &mut self,
        job: &CpuJobEnvelope,
        grant: SliceGrant,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<Option<NativeReservation>, AdaptiveManagerError> {
        self.replay_frontier
            .as_mut()
            .map(|frontier| frontier.reserve_native(job, grant, clock, now))
            .transpose()
            .map_err(AdaptiveManagerError::Frontier)
    }
    fn frontier_native_settled(
        &mut self,
        reservation: &NativeReservation,
        report: &CpuSliceReport,
        accepted: &SettledCpuSlice,
    ) -> Result<NativeCloseReason, AdaptiveManagerError> {
        self.replay_frontier
            .as_mut()
            .ok_or(AdaptiveManagerError::Invalid(
                "fixture native frontier absent",
            ))?
            .settle_native(reservation, report, accepted)
            .map_err(AdaptiveManagerError::Frontier)
    }
    fn frontier_native_released(
        &mut self,
        reservation: &NativeReservation,
        accepted: &ReleasedCpuReservation,
    ) -> Result<NativeCloseReason, AdaptiveManagerError> {
        self.replay_frontier
            .as_mut()
            .ok_or(AdaptiveManagerError::Invalid(
                "fixture native frontier absent",
            ))?
            .release_unqueued(reservation, accepted)
            .map_err(AdaptiveManagerError::Frontier)
    }
    fn frontier_native_unknown(
        &mut self,
        reservation: &NativeReservation,
    ) -> Result<(), AdaptiveManagerError> {
        self.replay_frontier
            .as_mut()
            .ok_or(AdaptiveManagerError::Invalid(
                "fixture native frontier absent",
            ))?
            .native_usage_unknown(reservation)
            .map_err(AdaptiveManagerError::Frontier)
    }
    fn accept_accepted_role_source(
        &mut self,
        work: &CoordinatorWork<ReviewView, u64>,
        accepted: &rz_search::pals::coordinator::AcceptedRoleStep,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        assert!(accepted.validates_output(output));
        self.validate_role(work, accepted.ticket(), output)
    }
    fn discard_accepted_role_reciprocal(
        &mut self,
        id: rz_contracts::rovent::ReciprocalQuestionId,
    ) -> Result<(), AdaptiveManagerError> {
        self.reciprocal_questions.remove(&id.sequence);
        Ok(())
    }
    fn prepare_accepted_role_reciprocal(
        &mut self,
        work: &CoordinatorWork<ReviewView, u64>,
        accepted: &rz_search::pals::coordinator::AcceptedRoleStep,
        output: &RoleOutput,
        id: rz_contracts::rovent::ReciprocalQuestionId,
        mode: rz_contracts::rovent::ExecutionMode,
        budget: Digest,
        recipient: rz_contracts::rovent::SearchRecipient,
    ) -> Result<Option<AdaptivePeerQuestion<u64>>, AdaptiveManagerError> {
        self.fixture_prepare_peer(work, accepted, output, id, mode, budget, recipient)
            .map(Some)
    }
    fn consume_accepted_role_reciprocal(
        &mut self,
        work: &CoordinatorWork<ReviewView, u64>,
        accepted: &rz_search::pals::coordinator::AcceptedRoleStep,
        output: &RoleOutput,
        _: Option<TicketExecutionEvidence>,
    ) -> Result<Digest, AdaptiveManagerError> {
        self.validate_role(work, accepted.ticket(), output)?;
        assert!(accepted.validates_output(output));
        self.peer_opinions += 1;
        Ok(accepted.output_digest())
    }
    fn consume_reciprocal_cpu(
        &mut self,
        _: &CoordinatorWork<ReviewView, u64>,
        report: &CpuSliceReport,
    ) -> Result<Digest, AdaptiveManagerError> {
        assert!(report.terminal && !report.continuation_retained);
        assert_eq!(
            report.result.as_ref().unwrap().report.completion,
            CpuCompletion::DepthLimit
        );
        self.peer_cpu_answers += 1;
        Ok(Digest([87; 32]))
    }
    fn discard_cpu_reciprocal(
        &mut self,
        id: rz_contracts::rovent::ReciprocalQuestionId,
    ) -> Result<(), AdaptiveManagerError> {
        self.reciprocal_questions.remove(&id.sequence);
        Ok(())
    }
    fn reciprocal_parent(
        &mut self,
        work: &CoordinatorWork<ReviewView, u64>,
        job: &rz_contracts::pals_adaptive::CpuJobEnvelope,
        id: rz_contracts::rovent::ReciprocalQuestionId,
        mode: rz_contracts::rovent::ExecutionMode,
        root_budget: Digest,
    ) -> Result<rz_contracts::rovent::QuestionEnvelope, AdaptiveManagerError> {
        self.fixture_reciprocal_parent(work, job, id, mode, root_budget)
    }
    fn prepare_cpu_reciprocal(
        &mut self,
        work: &CoordinatorWork<ReviewView, u64>,
        request: &rz_search::cpu::CpuReciprocalRequest,
        id: rz_contracts::rovent::ReciprocalQuestionId,
        mode: rz_contracts::rovent::ExecutionMode,
        root_budget: Digest,
    ) -> Result<AdaptiveReciprocalQuestion<u64>, AdaptiveManagerError> {
        self.fixture_prepare_reciprocal(work, request, id, mode, root_budget)
    }
    fn cpu_reciprocal_wait(
        &mut self,
        _: &CoordinatorWork<ReviewView, u64>,
        report: &CpuSliceReport,
    ) -> Result<(), AdaptiveManagerError> {
        assert_eq!(
            report.result.as_ref().unwrap().report.completion,
            CpuCompletion::AwaitReciprocal
        );
        assert!(!self.reciprocal_waits.contains(&report.grant.execution));
        self.reciprocal_waits.push(report.grant.execution);
        Ok(())
    }
    fn consume_cpu_reciprocal(
        &mut self,
        work: &CoordinatorWork<ReviewView, u64>,
        accepted: &rz_search::pals::coordinator::AcceptedRoleStep,
        output: &RoleOutput,
        _: Option<TicketExecutionEvidence>,
        requests: &[(
            rz_search::cpu::CpuReciprocalRequest,
            rz_contracts::rovent::ReciprocalQuestionId,
        )],
    ) -> Result<Vec<rz_search::cpu::CpuReciprocalReply>, AdaptiveManagerError> {
        self.fixture_consume_reciprocal(work, accepted, output, requests)
    }
    fn root_rules(&self) -> Arc<ContractState> {
        Arc::clone(&self.rules)
    }
    fn root_role_call_limit(&self) -> Result<Option<u64>, AdaptiveManagerError> {
        Ok(self.role_call_limit)
    }
    fn ranking_published(
        &mut self,
        ranking: &ProposerRankingPublished,
    ) -> Result<Vec<CandidateFrame<u64>>, AdaptiveManagerError> {
        assert_eq!(ranking.source().ticket.authority, self.authority);
        assert!(
            ranking
                .rules()
                .rules()
                .snapshot()
                .same_state(self.rules.rules().snapshot())
        );
        let width = match self.scope {
            FixtureScope::TwoIndependentRoles => 2,
            FixtureScope::OwnCpuResume
            | FixtureScope::OwnCpuReciprocal
            | FixtureScope::PeerAllDirections => 1,
        };
        ranking
            .ranked_candidates()
            .iter()
            .take(width)
            .map(|movement| {
                let first = BoardMove::try_from(movement.decode().unwrap()).unwrap();
                Ok(CandidateFrame {
                    first_move: *movement,
                    input: self.input(RefinementStage::ProposalContinuation, Some(first)),
                    budget: FrameBudget {
                        role_submissions_remaining: if matches!(
                            self.scope,
                            FixtureScope::PeerAllDirections
                        ) {
                            12
                        } else {
                            4
                        },
                        original_cpu_credit: TOTAL_NODES,
                        original_store_byte_limit: 1 << 20,
                    },
                    deferred: None,
                    initialized: None,
                })
            })
            .collect()
    }
    fn validate_role(
        &self,
        work: &CoordinatorWork<ReviewView, u64>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
    ) -> Result<(), AdaptiveManagerError> {
        if output.id != ticket.id
            || output.authority != self.authority
            || output.key != ticket.representation
            || ticket.question != work.input.question_digest
            || ticket.view != work.input.view_seal
            || !work.input.view.validates(&self.stores)
        {
            return Err(AdaptiveManagerError::Invalid(
                "fixture original question/view",
            ));
        }
        let candidate_logits = match &output.payload {
            RolePayload::Proposal {
                candidate_logits, ..
            } if ticket.role == Role::Proposer => candidate_logits,
            RolePayload::Counterexample {
                candidate_logits, ..
            } if ticket.role == Role::Critic => candidate_logits,
            _ => return Err(AdaptiveManagerError::Invalid("fixture role head")),
        };
        if candidate_logits.len() != work.input.view.query().legal.len()
            || candidate_logits.iter().any(|v| !v.is_finite())
        {
            return Err(AdaptiveManagerError::Invalid("fixture legal head"));
        }
        Ok(())
    }
    fn commit_role(
        &mut self,
        work: &CoordinatorWork<ReviewView, u64>,
        ticket: &RoleTicketEnvelope,
        output: &RoleOutput,
        _: Option<TicketExecutionEvidence>,
    ) -> Result<CheckedStep<u64>, AdaptiveManagerError> {
        self.validate_role(work, ticket, output)?;
        if self.block_replay_while_reserved
            && self
                .replay_frontier
                .as_ref()
                .is_some_and(|frontier| frontier.snapshot().work.reserved_native_nodes != 0)
        {
            self.blocked_replays += 1;
            return Err(AdaptiveManagerError::Frontier(FrontierError::NodesReserved));
        }
        if self.fail_role_once {
            self.fail_role_once = false;
            return Err(AdaptiveManagerError::Semantic(
                "authored role hot-commit failure".into(),
            ));
        }
        if self.fail_after_role_consumption_once {
            self.fail_after_role_consumption_once = false;
            self.role_consumed(ticket, &work.input.view)?;
            return Err(AdaptiveManagerError::Semantic(
                "authored next-view materialization failure after actual consumption".into(),
            ));
        }
        let transition = match self.scope {
            FixtureScope::TwoIndependentRoles => FrameTransition::Unknown {
                reason: "authored integration fixture ends after actual role acceptance",
            },
            FixtureScope::OwnCpuResume | FixtureScope::OwnCpuReciprocal => {
                let mut input = work.input.clone();
                input.stage = RefinementStage::CpuValidation;
                FrameTransition::Continue(input)
            }
            FixtureScope::PeerAllDirections => {
                if ticket.role == Role::Proposer {
                    let mut input = work.input.clone();
                    input.stage = RefinementStage::CpuValidation;
                    FrameTransition::Continue(input)
                } else {
                    FrameTransition::Unknown {
                        reason: "authored peer fixture ends after original C source commits",
                    }
                }
            }
        };
        Ok(CheckedStep {
            evidence: Digest(ticket.representation.input.0),
            transition,
            fresh_pair: None,
            rules_pair: None,
            queued_recheck: None,
        })
    }
    fn cpu_question(
        &mut self,
        work: &CoordinatorWork<ReviewView, u64>,
        _: &[CpuWorkerProfile],
        _: Option<&ExternalCpuConfig>,
        _: CpuEndpoint,
        physical_budget_exhausted: bool,
    ) -> Result<CpuQuestion, AdaptiveManagerError> {
        if physical_budget_exhausted {
            return Ok(CpuQuestion::OwnBudgetExhausted {
                semantic_limit: TOTAL_NODES,
                semantic_spent: TOTAL_NODES,
            });
        }
        let mut endpoint = Position::from_snapshot(
            self.rules.rules().snapshot(),
            rz_position::PositionLimits::default(),
        )
        .unwrap();
        for movement in &work.input.view.query().prefix {
            endpoint.make_move(*movement).unwrap();
        }
        let restriction = endpoint
            .legal_moves()
            .first()
            .copied()
            .into_iter()
            .collect();
        Ok(CpuQuestion::Own(OwnCpuQuestion {
            position: ContractPosition::new(OwnerId(71), endpoint),
            root_moves: Some(restriction),
            target_depth: 2,
            total_native_node_credit: TOTAL_NODES,
        }))
    }
    fn commit_cpu(
        &mut self,
        work: &CoordinatorWork<ReviewView, u64>,
        report: &CpuSliceReport,
    ) -> Result<CheckedStep<u64>, AdaptiveManagerError> {
        assert!(report.terminal && !report.continuation_retained);
        let actual = &report.result.as_ref().unwrap().report;
        assert_eq!(actual.completion, CpuCompletion::DepthLimit);
        if matches!(
            self.scope,
            FixtureScope::OwnCpuReciprocal | FixtureScope::PeerAllDirections
        ) {
            assert_eq!(actual.completed_depth, 2);
            assert!(report.cumulative_nodes > 0);
            assert!(!self.reciprocal_waits.is_empty());
            assert!(self.reciprocal_replies > 0);
        } else {
            let expected = &self.expected_cpu[&work.original_first_move.bits()];
            assert_eq!(actual.completed_depth, expected.completed_depth);
            assert_eq!(actual.score, expected.score);
            assert_eq!(actual.pv, expected.pv);
            assert_eq!(report.cumulative_nodes, expected.nodes);
        }
        self.actual_cpu.push(report.clone());
        let transition = if matches!(self.scope, FixtureScope::PeerAllDirections) {
            let first = BoardMove::try_from(work.original_first_move.decode().unwrap()).unwrap();
            FrameTransition::Continue(self.input(RefinementStage::CriticDivergence, Some(first)))
        } else {
            FrameTransition::Unknown {
                reason: "authored integration fixture ends after actual private CPU completion",
            }
        };
        Ok(CheckedStep {
            evidence: Digest([42; 32]),
            transition,
            fresh_pair: None,
            rules_pair: None,
            queued_recheck: None,
        })
    }
    fn cpu_slice_observed(
        &mut self,
        _: &CoordinatorWork<ReviewView, u64>,
        report: &CpuSliceReport,
    ) -> Result<(), AdaptiveManagerError> {
        if self.fail_raw_once {
            self.fail_raw_once = false;
            return Err(AdaptiveManagerError::Semantic(
                "authored hot-commit failure".into(),
            ));
        }
        assert!(!self.raw_cpu_receipts.contains(&report.grant.execution));
        self.raw_cpu_receipts.push(report.grant.execution);
        Ok(())
    }
    fn commit_external(
        &mut self,
        _: &CoordinatorWork<ReviewView, u64>,
        _: &ExternalWorkerReport,
    ) -> Result<CheckedStep<u64>, AdaptiveManagerError> {
        Err(AdaptiveManagerError::Invalid(
            "fixture forbids external execution",
        ))
    }
    fn publish_closed(&mut self, closed: &ClosedFrame) -> Result<(), AdaptiveManagerError> {
        assert!(closed.unknown.is_some());
        assert!(closed.completion_scope.is_none());
        #[cfg(feature = "pals-adaptive-diagnostics")]
        if std::mem::take(&mut self.fail_publish_closed_once) {
            return Err(AdaptiveManagerError::Semantic(
                "authored closed-frame publication failure".into(),
            ));
        }
        self.closed.push(closed.clone());
        Ok(())
    }
    fn role_consumed(
        &mut self,
        ticket: &RoleTicketEnvelope,
        view: &Arc<ReviewView>,
    ) -> Result<(), AdaptiveManagerError> {
        assert_eq!(ticket.view.view_digest.0, view.seal_digest());
        if let Some(original) = self.consumed.iter().find(|old| old.id == ticket.id) {
            assert_eq!(original, ticket);
            return Ok(());
        }
        self.index.consume(view, &self.stores).unwrap();
        self.consumed.push(*ticket);
        Ok(())
    }
    fn role_consumption_recorded(
        &self,
        ticket: &RoleTicketEnvelope,
        view: &ReviewView,
    ) -> Result<bool, AdaptiveManagerError> {
        if self.fail_role_consumption_readback {
            return Err(AdaptiveManagerError::Semantic(
                "authored role consumption readback failure".into(),
            ));
        }
        assert_eq!(ticket.view.view_digest.0, view.seal_digest());
        Ok(self.consumed.iter().any(|old| old == ticket))
    }
}

type Manager = AdaptiveExecutionManager<u64, NativeFixture, DriverFixture>;

fn cpu_engine() -> CpuEngine {
    CpuEngine::with_evaluator_ordering_and_resume_policy(
        CpuConfig {
            profile: CpuProfile::PlanAssisted,
            tt_entries: 128,
            max_depth: 2,
            quiescence_ply: 4,
        },
        Arc::new(BootstrapCpuValue::default()),
        CpuOrderingPolicy::LegacyMvvLvaV1,
        CpuResumePolicy::PausedStack,
    )
    .unwrap()
}
fn fixture(
    scope: FixtureScope,
    finish_ok: bool,
    credit: u64,
) -> (
    Manager,
    Arc<NativeFacts>,
    FrameInput<ReviewView, u64>,
    Instant,
) {
    fixture_with_resource_limits(scope, finish_ok, credit, None)
}
fn fixture_with_resource_limits(
    scope: FixtureScope,
    finish_ok: bool,
    credit: u64,
    limits: Option<AdaptiveExecutionLimits>,
) -> (
    Manager,
    Arc<NativeFacts>,
    FrameInput<ReviewView, u64>,
    Instant,
) {
    fixture_with_registered_role_limit(scope, finish_ok, credit, limits, None)
}
fn fixture_with_registered_role_limit(
    scope: FixtureScope,
    finish_ok: bool,
    credit: u64,
    limits: Option<AdaptiveExecutionLimits>,
    role_call_limit: Option<u64>,
) -> (
    Manager,
    Arc<NativeFacts>,
    FrameInput<ReviewView, u64>,
    Instant,
) {
    fixture_with_registered_role_limit_window(
        scope,
        finish_ok,
        credit,
        limits,
        role_call_limit,
        WINDOW,
    )
}
fn fixture_with_registered_role_limit_window(
    scope: FixtureScope,
    finish_ok: bool,
    credit: u64,
    limits: Option<AdaptiveExecutionLimits>,
    role_call_limit: Option<u64>,
    fixture_window: Duration,
) -> (
    Manager,
    Arc<NativeFacts>,
    FrameInput<ReviewView, u64>,
    Instant,
) {
    let position = Position::from_fen("3rk3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
    let mut expected_cpu = BTreeMap::new();
    if matches!(scope, FixtureScope::OwnCpuResume) {
        // The independent direct search runs before the manager exists. It is
        // not a synchronous CPU callback from an adaptive manager turn.
        for first in position.legal_moves() {
            let mut endpoint = position.clone();
            endpoint.make_move(first).unwrap();
            let restriction = vec![endpoint.legal_moves()[0]];
            let report = cpu_engine()
                .analyze_root_moves(
                    &endpoint,
                    &restriction,
                    CpuLimits {
                        max_depth: 2,
                        max_nodes: TOTAL_NODES,
                        deadline: None,
                    },
                    &AtomicBool::new(false),
                )
                .unwrap();
            expected_cpu.insert(
                Move16::encode(rz_contracts::Move::try_from(first).unwrap()).bits(),
                report,
            );
        }
    }
    let clock = ContractSystemClock::new(ProcessEpoch(117));
    let until = clock.origin() + fixture_window;
    let authority = SearchAuthority {
        epoch: ProcessEpoch(117),
        game: GameGeneration(1),
        root: RootGeneration(1),
        implementation: Digest([19; 32]),
    };
    let rules = Arc::new(
        ContractPosition::new(OwnerId(17), position.clone())
            .export()
            .unwrap(),
    );
    let mut stores = PalsStores::new(StoreLimits::default());
    let root = stores.focus_actual_moves(position.snapshot()).unwrap();
    let state = stores.situations.get(root).unwrap().state;
    let mut driver = DriverFixture {
        authority,
        role_call_limit,
        rules,
        stores,
        index: ReviewContextIndex::default(),
        state,
        scope,
        consumed: Vec::new(),
        closed: Vec::new(),
        expected_cpu,
        actual_cpu: Vec::new(),
        fail_raw_once: false,
        semantic_owner_busy: false,
        fail_role_once: false,
        fail_after_role_consumption_once: false,
        fail_role_consumption_readback: false,
        #[cfg(feature = "pals-adaptive-diagnostics")]
        fail_publish_closed_once: false,
        raw_cpu_receipts: Vec::new(),
        reciprocal_questions: BTreeMap::new(),
        reciprocal_waits: Vec::new(),
        reciprocal_replies: 0,
        peer_sources: Vec::new(),
        peer_opinions: 0,
        peer_cpu_answers: 0,
        peer_repeat_question: false,
        replay_frontier: None,
        block_replay_while_reserved: false,
        blocked_replays: 0,
    };
    let root_input = driver.input(RefinementStage::ProposerRanking, None);
    let facts = Arc::new(NativeFacts::default());
    let thread_facts = Arc::clone(&facts);
    let native_clock = clock.clone();
    let factory = move || {
        thread_facts.factories.fetch_add(1, Ordering::Relaxed);
        Ok(NativeFixture {
            clock: native_clock,
            facts: thread_facts,
            finish_ok,
        })
    };
    let native = match limits {
        Some(limits) => TicketedNativeService::spawn_with_capacity(
            factory,
            TicketServiceLimits::default(),
            limits.role_tickets_max,
            limits.worker_commands_max,
        ),
        None => TicketedNativeService::spawn(factory, TicketServiceLimits::default()),
    }
    .unwrap();
    let mut cpu = cpu_engine();
    let worker = match limits {
        Some(limits) => {
            cpu.set_paused_stack_max_bytes(limits.paused_bytes_per_worker_max)
                .unwrap();
            CpuWorker::spawn_notified_retaining_with_capacity(
                CpuWorkerId(0),
                cpu,
                native.signal(),
                limits.worker_commands_max,
            )
            .unwrap_or_else(|(_, e)| panic!("fixture owner start failed: {e:?}"))
        }
        None => CpuWorker::spawn_notified(CpuWorkerId(0), cpu, native.signal()).unwrap(),
    };
    let registration = AdaptiveManagerRegistration {
        authority,
        hard_deadline: until,
        token_budget: 512,
        policy: CoordinatorPolicy::Fixed,
        first_request_sequence: 1,
        native_cpu_node_limit: TOTAL_NODES,
        endpoints: CpuEndpointSelection::deployment(),
        controller: ControllerConfig {
            fixed_node_credit: credit,
            ..ControllerConfig::default()
        },
        device_digest: Digest([71; 32]),
        cpu_device_digest: Digest([72; 32]),
    };
    let manager = match limits {
        Some(limits) => AdaptiveExecutionManager::new_retaining_with_limits(
            registration,
            clock,
            limits,
            AdaptiveManagerSources {
                native,
                own_workers: vec![worker],
                external: None,
                driver,
            },
        )
        .unwrap_or_else(|e| panic!("actual resource admission failed: {}", e.error)),
        None => {
            AdaptiveExecutionManager::new(registration, clock, native, vec![worker], None, driver)
                .unwrap()
        }
    };
    (manager, facts, root_input, until)
}

#[test]
fn explicit_lower_caps_bound_actual_admission_and_complete_both_independent_questions() {
    let limits = AdaptiveExecutionLimits {
        waiting_jobs_max: 1,
        worker_commands_max: 1,
        role_tickets_max: 1,
        paused_bytes_per_worker_max: 1 << 20,
    };
    let (mut manager, facts, root, until) =
        fixture_with_resource_limits(FixtureScope::TwoIndependentRoles, true, 13, Some(limits));
    assert_eq!(manager.limits, limits);
    assert_eq!(manager.native.command_capacity(), 1);
    assert_eq!(manager.native.pending_capacity(), 1);
    assert_eq!(
        manager.own[0].worker.as_ref().unwrap().command_capacity(),
        1
    );
    assert_eq!(
        manager.own[0].worker.as_ref().unwrap().paused_bytes_limit(),
        1 << 20
    );
    manager.prepare_root_ranking(root.clone()).unwrap();
    assert!(matches!(
        manager.prepare_root_ranking(root),
        Err(AdaptiveManagerError::Invalid(
            "root ranking admission order"
        ))
    ));
    facts.child_gate.store(true, Ordering::Release);
    pump(&mut manager, until, |manager| {
        assert!(manager.roles.len() <= limits.role_tickets_max);
        assert!(manager.ready.len() + manager.validator_waiting.len() <= limits.waiting_jobs_max);
        manager.driver().closed.len() == 2 && !manager.has_pending_work()
    });
    assert_eq!(manager.driver().consumed.len(), 3);
    assert_eq!(manager.counters().consumed_role_outputs, 3);
    assert!(manager.failures().next().is_none());
    assert!(shutdown(&mut manager, until).permits_archive_close());
}

#[test]
fn root_role_budget_reserves_both_pending_questions_before_native_preparation() {
    for limit in [2, 3] {
        let (mut manager, facts, root, until) = fixture_with_registered_role_limit(
            FixtureScope::TwoIndependentRoles,
            true,
            4096,
            None,
            Some(limit),
        );
        assert_eq!(manager.role_call_limit, Some(limit));
        manager.prepare_root_ranking(root).unwrap();
        assert_eq!(manager.role_call_budget_usage(), (0, 1));
        pump(&mut manager, until, |manager| {
            manager.counters().consumed_role_outputs == 1 && manager.roles.len() as u64 == limit - 1
        });
        // Both candidates carry four frame calls. Their copied ceilings do
        // not mint eight root calls: root ranking + pending reservations wins.
        assert_eq!(manager.role_call_budget_usage(), (1, limit - 1));
        assert!(!manager.role_call_limit_reached());
        let binding = &manager.roles[0];
        let work = binding.work.clone();
        let input = binding.input.clone();
        assert!(matches!(
            manager.prepare_role(work, input, RolePurpose::Refinement),
            Err(AdaptiveManagerError::RoleCallLimit)
        ));
        pump(&mut manager, until, |manager| {
            manager.role_call_budget_usage() == (limit, 0)
        });
        assert_eq!(manager.phase(), ManagerPhase::Running);
        assert_eq!(manager.counters().consumed_role_outputs, 1);
        assert_eq!(manager.counters().unused_role_results, 0);
        assert!(!manager.roles.is_empty());
        // Spend exhaustion cannot revoke the admitted physical/semantic
        // obligations. Open the original worker gate and accept each once.
        facts.child_gate.store(true, Ordering::Release);
        let receipt = manager
            .run_root_until(until, &AtomicBool::new(false))
            .unwrap();
        assert!(receipt.permits_root_reuse());
        assert!(manager.failures().next().is_none());
        assert!(manager.role_call_limit_reached());
        assert_eq!(manager.role_call_budget_usage(), (limit, 0));
        assert_eq!(manager.counters().submitted_role_tickets, limit);
        assert_eq!(manager.counters().physical_role_completions, limit);
        assert_eq!(manager.counters().consumed_role_outputs, limit);
        assert_eq!(manager.counters().unused_role_results, 0);
        assert_eq!(manager.driver().consumed.len() as u64, limit);
        assert_eq!(facts.accepted.lock().unwrap().len() as u64, limit);
        assert_eq!(manager.native_cpu_usage().unwrap(), (0, 0));
    }
}

#[test]
fn exhausted_root_role_budget_preserves_actual_commit_failure() {
    let (mut manager, facts, root, until) = fixture_with_registered_role_limit(
        FixtureScope::TwoIndependentRoles,
        true,
        4096,
        None,
        Some(2),
    );
    manager.driver_mut().fail_role_once = true;
    facts.child_gate.store(true, Ordering::Release);
    manager.prepare_root_ranking(root).unwrap();
    let receipt = manager.run_until(until, &AtomicBool::new(false)).unwrap();
    assert!(receipt.permits_archive_close());
    assert!(manager.role_call_limit_reached());
    assert_eq!(manager.counters().submitted_role_tickets, 2);
    assert_eq!(manager.counters().consumed_role_outputs, 1);
    assert_eq!(manager.counters().unused_role_results, 1);
    assert!(manager.failures().any(|error| matches!(error,
        AdaptiveManagerError::Semantic(message) if message == "authored role hot-commit failure")));
    assert!(manager.roles.is_empty());
    assert_eq!(manager.native_cpu_usage().unwrap(), (0, 0));
}

#[test]
fn failed_frame_binding_keeps_one_registry_charge_without_double_reserving() {
    let (mut manager, _, root, until) = fixture_with_registered_role_limit(
        FixtureScope::TwoIndependentRoles,
        true,
        4096,
        None,
        Some(2),
    );
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.counters().consumed_role_outputs == 1 && manager.roles.len() == 1
    });
    assert_eq!(manager.role_call_budget_usage(), (1, 1));
    // The original immutable preparation remains real, but its frame binding
    // is deliberately invalid. Registry admission must stay charged once even
    // though the native executable ticket is never installed on this binding.
    manager.roles[0].work.as_mut().unwrap().frame.sequence += 1_000_000;
    pump(&mut manager, until, |manager| {
        manager.failures().next().is_some()
    });
    assert_eq!(manager.role_call_budget_usage(), (2, 0));
    assert!(
        manager
            .failures()
            .any(|error| matches!(error, AdaptiveManagerError::Coordinator(_)))
    );
    let receipt = shutdown(&mut manager, until);
    assert!(receipt.permits_archive_close());
    assert_eq!(receipt.pending_tickets, 0);
    assert_eq!(manager.role_call_budget_usage(), (2, 0));
    assert_eq!(manager.counters().consumed_role_outputs, 1);
    assert_eq!(manager.counters().physical_role_completions, 1);
}

#[test]
fn explicit_resource_limits_reject_each_out_of_range_axis() {
    let defaults = AdaptiveExecutionLimits::default();
    assert!(defaults.validate().is_ok());
    for limits in [
        AdaptiveExecutionLimits {
            waiting_jobs_max: 0,
            ..defaults
        },
        AdaptiveExecutionLimits {
            waiting_jobs_max: 65,
            ..defaults
        },
        AdaptiveExecutionLimits {
            worker_commands_max: 0,
            ..defaults
        },
        AdaptiveExecutionLimits {
            worker_commands_max: 9,
            ..defaults
        },
        AdaptiveExecutionLimits {
            role_tickets_max: 0,
            ..defaults
        },
        AdaptiveExecutionLimits {
            role_tickets_max: 17,
            ..defaults
        },
        AdaptiveExecutionLimits {
            paused_bytes_per_worker_max: 0,
            ..defaults
        },
        AdaptiveExecutionLimits {
            paused_bytes_per_worker_max: (8 << 20) + 1,
            ..defaults
        },
    ] {
        assert!(limits.validate().is_err());
    }
}

#[test]
fn explicit_one_command_owner_repeated_resume_preserves_credit_and_completion() {
    let limits = AdaptiveExecutionLimits {
        waiting_jobs_max: 1,
        worker_commands_max: 1,
        role_tickets_max: 1,
        paused_bytes_per_worker_max: 1 << 20,
    };
    let (mut manager, facts, root, until) =
        fixture_with_resource_limits(FixtureScope::OwnCpuResume, true, 1, Some(limits));
    facts.child_gate.store(true, Ordering::Release);
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        assert!(manager.waiting_jobs_occupancy() <= limits.waiting_jobs_max);
        manager.driver().closed.len() == 1 && !manager.has_pending_work()
    });
    assert!(manager.counters().cpu_yields > 0);
    assert_eq!(manager.driver().actual_cpu.len(), 1);
    let nodes = manager.driver().actual_cpu[0].cumulative_nodes;
    assert_eq!(manager.native_cpu_usage().unwrap(), (nodes, 0));
    assert!(manager.failures().next().is_none());
    let receipt = shutdown(&mut manager, until);
    assert!(receipt.permits_archive_close());
    assert_eq!(
        (receipt.native_cpu_spent, receipt.native_cpu_reserved),
        (nodes, 0)
    );
}
fn pump(manager: &mut Manager, until: Instant, predicate: impl Fn(&Manager) -> bool) {
    while !predicate(manager) {
        assert!(
            Instant::now() < until,
            "manager fixture timeout: {:?}",
            manager.failures().collect::<Vec<_>>()
        );
        manager.tick().unwrap();
        thread::sleep(Duration::from_millis(1));
    }
}
fn shutdown(manager: &mut Manager, until: Instant) -> AdaptiveShutdownReceipt {
    manager.begin_shutdown(until).unwrap();
    pump(manager, until, |manager| {
        let receipt = manager.shutdown_receipt().unwrap();
        receipt.native_owner_joined && receipt.own_cpu_joined == 1
    });
    manager.shutdown_receipt().unwrap()
}

/// Wait for an actual owner event without manufacturing a Retired packet or
/// copying its linear permit. All earlier actual events are processed once.
fn take_original_native_retirement(
    manager: &mut Manager,
    until: Instant,
) -> TicketNativeEvent<RoleOutput> {
    loop {
        assert!(
            Instant::now() < until,
            "original native retirement fixture timeout"
        );
        if let Some(event) = manager.native.poll() {
            if matches!(&event, TicketNativeEvent::Retired { .. }) {
                return event;
            }
            manager.native_event(event).unwrap();
        } else {
            thread::sleep(Duration::from_millis(1));
        }
    }
}

#[test]
fn copied_publication_capacity_error_without_original_controller_remains_failure() {
    for cause in [
        FrontierError::PublicRecordCapacity,
        FrontierError::PublicReviewIndexCapacity,
        FrontierError::Capacity("another original bounded resource"),
    ] {
        let (mut manager, _, _, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
        manager
            .frontier_boundary_or_failure(AdaptiveManagerError::Frontier(cause))
            .unwrap();
        assert_eq!(manager.failures().count(), 1);
        assert_eq!(manager.phase(), ManagerPhase::Running);
        assert!(manager.shutdown_until.is_none());
        let receipt = shutdown(&mut manager, until);
        // This helper joins the process owner; it does not quiesce an owner
        // for another root. Physical closure must not erase logical failure.
        assert!(receipt.permits_archive_close());
        assert!(!receipt.permits_root_reuse());
        assert_eq!(manager.failures().count(), 1);
        assert_eq!(receipt.counters.accepted_nn_inputs, 0);
        assert_eq!(receipt.counters.consumed_role_outputs, 0);
    }
}

#[test]
fn pcr_keeps_actual_canceled_published_head_accept_until_original_fence() {
    for process_shutdown in [false, true] {
        let (mut manager, facts, root, until) =
            fixture(FixtureScope::TwoIndependentRoles, true, 4096);
        facts.block_after_first_run.store(true, Ordering::Release);
        facts
            .hold_second_until_cancel
            .store(true, Ordering::Release);
        // Two actual native questions let the second run occupy the owner
        // while the manager queues the first original P-ranking Accept.
        // They are transport/custody fixtures, not invented search evidence.
        let first = root.view.query().legal[0];
        manager
            .prepare_role(None, root, RolePurpose::RootRanking)
            .unwrap();
        let child = manager
            .driver_mut()
            .input(RefinementStage::ProposalContinuation, Some(first));
        manager
            .prepare_role(None, child, RolePurpose::RoventRefinement)
            .unwrap();
        pump(&mut manager, until, |manager| {
            manager.shutdown_receipt().unwrap().pending_tickets == 2
        });
        facts.release_first_run.store(true, Ordering::Release);
        while facts.started.lock().unwrap().len() < 2 {
            assert!(
                Instant::now() < until,
                "second actual native run did not start"
            );
            thread::sleep(Duration::from_millis(1));
        }
        manager.tick().unwrap();
        let ticket = manager
            .roles
            .iter()
            .find(|binding| binding.purpose == RolePurpose::RootRanking)
            .and_then(|binding| binding.ticket)
            .expect("original P-ranking ticket");
        let binding = manager
            .roles
            .iter()
            .find(|binding| binding.ticket == Some(ticket))
            .unwrap();
        assert!(binding.ack_dispatched && binding.ack_accepted == Some(true));
        assert!(binding.service_completed && !binding.native_acceptance_recorded);
        assert!(manager.driver().consumed.is_empty());
        assert!(facts.accepted.lock().unwrap().is_empty());
        let cleanup_until = Instant::now() + WINDOW;
        if process_shutdown {
            manager.begin_shutdown(cleanup_until).unwrap();
        } else {
            manager.begin_root_quiesce(cleanup_until).unwrap();
        }
        let original = take_original_native_retirement(&mut manager, cleanup_until);
        assert!(facts.second_run_observed_cancel.load(Ordering::Acquire));
        assert!(matches!(&original, TicketNativeEvent::Retired {
            ticket: actual, accepted: false, result: Err(RoleFailure::Canceled),
            rejected_permit: Some(_), native_receipt: None, observation_failures,
        } if *actual == ticket && observation_failures.is_empty()));
        manager.native_event(original).unwrap();
        let binding = manager
            .roles
            .iter()
            .find(|binding| binding.ticket == Some(ticket))
            .unwrap();
        assert_eq!(binding.revoked_acceptance, Some(RoleFailure::Canceled));
        assert!(binding.delivered.is_some());
        assert!(!binding.native_acceptance_recorded && !binding.consumption_recorded);
        assert_eq!(manager.retained_native_permits.len(), 1);
        assert_eq!(manager.registry.occupancy(), 2);
        assert!(manager.native_root_fence.is_none());
        pump(&mut manager, cleanup_until, |manager| {
            manager.phase() == ManagerPhase::Finished
        });
        let receipt = manager.shutdown_receipt().unwrap();
        assert!(receipt.permits_semantic_closure(), "{receipt:?}");
        assert!(
            manager.failures().next().is_none(),
            "{:?}",
            manager.failures().collect::<Vec<_>>()
        );
        assert_eq!(receipt.counters.accepted_nn_inputs, 0);
        assert_eq!(receipt.counters.consumed_role_outputs, 0);
        assert_eq!(receipt.pending_tickets, 0);
        assert!(manager.driver().consumed.is_empty());
        assert!(facts.accepted.lock().unwrap().is_empty());
        assert!(manager.roles.is_empty() && manager.retained_native_permits.is_empty());
        assert_eq!(manager.driver().stores.review_pin_count().unwrap(), 0);
        if !receipt.native_owner_joined {
            manager.native.begin_shutdown(cleanup_until);
            while !manager.native.try_join().unwrap() {
                assert!(Instant::now() < cleanup_until);
                manager.native.poll();
                thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

#[test]
fn pcr_drain_does_not_hide_actual_non_cancel_published_head_rejection() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
    facts.reject_acceptance.store(true, Ordering::Release);
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.roles.iter().any(|binding| binding.ack_dispatched)
    });
    // Capture the actual failed owner ACK before close, then consume that
    // same event once during Draining. All other cancellation/permit guards
    // hold; InvalidOutput must still remain a failure rather than cleanup.
    let original = take_original_native_retirement(&mut manager, until);
    assert!(matches!(
        &original,
        TicketNativeEvent::Retired {
            accepted: false,
            result: Err(RoleFailure::InvalidOutput),
            rejected_permit: Some(_),
            native_receipt: None,
            ..
        }
    ));
    let cleanup_until = Instant::now() + WINDOW;
    manager.begin_root_quiesce(cleanup_until).unwrap();
    let error = manager.native_event(original).unwrap_err();
    assert!(matches!(&error, AdaptiveManagerError::Semantic(message)
        if message == "native retirement: InvalidOutput"));
    assert!(
        manager
            .roles
            .iter()
            .all(|binding| binding.revoked_acceptance.is_none())
    );
    assert_eq!(manager.retained_native_permits.len(), 1);
    manager.record_failure(error);
    pump(&mut manager, cleanup_until, |manager| {
        manager.phase() == ManagerPhase::Finished
    });
    let receipt = manager.shutdown_receipt().unwrap();
    assert!(receipt.permits_root_reuse(), "{receipt:?}");
    assert!(manager.failures().any(|error| matches!(error,
        AdaptiveManagerError::Semantic(message) if message == "native retirement: InvalidOutput")));
    assert_eq!(receipt.counters.accepted_nn_inputs, 0);
    assert_eq!(receipt.counters.consumed_role_outputs, 0);
    assert!(manager.driver().consumed.is_empty());
    assert!(facts.accepted.lock().unwrap().is_empty());
    assert!(manager.roles.is_empty() && manager.retained_native_permits.is_empty());
    assert_eq!(manager.driver().stores.review_pin_count().unwrap(), 0);
    manager.native.begin_shutdown(cleanup_until);
    while !manager.native.try_join().unwrap() {
        assert!(Instant::now() < cleanup_until);
        manager.native.poll();
        thread::sleep(Duration::from_millis(1));
    }
}

type ActualPortManager = AdaptiveExecutionManager<
    rz_search::pals::engine::adaptive_port::AdaptivePortQuestion,
    NativeFixture,
    AdaptivePortDriver<rz_search::pals::engine::adaptive_port::AdaptivePortModel>,
>;
fn actual_search_port_fixture(
    config: rz_search::pals::engine::PalsConfig,
    model_wdl: bool,
) -> (
    ActualPortManager,
    Arc<NativeFacts>,
    FrameInput<ReviewView, rz_search::pals::engine::adaptive_port::AdaptivePortQuestion>,
    Instant,
    Position,
) {
    use rz_runtime::contracts::ContractClock as _;
    use rz_search::pals::engine::adaptive_port::{
        AdaptiveEnginePort, AdaptiveModelNamespace, AdaptivePortModel, AdaptivePortQuestion,
    };
    use rz_search::pals::engine::{
        PalsEngine, PalsLimits, PostRepairRecheckPolicy, ResolverPolicy,
    };
    use rz_search::pals::review_context::ReviewIndexLimits;
    use rz_search::pals::value::{MODEL_WDL_VALUE_SEMANTICS, ModelValueIdentity};
    use std::sync::atomic::AtomicU64;

    let position = if model_wdl {
        Position::startpos()
    } else {
        Position::from_fen("3rk3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap()
    };
    let node_limit = if model_wdl { 1024 } else { TOTAL_NODES };
    let clock = ContractSystemClock::new(ProcessEpoch(221));
    let until = clock.origin() + WINDOW;
    let authority = SearchAuthority {
        epoch: ProcessEpoch(221),
        game: GameGeneration(1),
        root: RootGeneration(1),
        implementation: Digest([89; 32]),
    };
    // This model has metadata only: every physical method is Unsupported.
    // Therefore an accidental synchronous model call cannot satisfy the test.
    let value = ModelValueIdentity {
        semantics: MODEL_WDL_VALUE_SEMANTICS.into(),
        model: "authored-native-fixture".into(),
        encoding: "authored-fixture-encoding".into(),
        precision: "fp32".into(),
        model_epoch: [5; 32],
    };
    let namespace = AdaptiveModelNamespace::new(
        value.clone(),
        Digest([5; 32]),
        Digest([6; 32]),
        PrecisionProfile::Fp32,
        1,
    )
    .unwrap();
    let model =
        AdaptivePortModel::new_with_namespace(value.model.clone(), Some(value), namespace).unwrap();
    let mut engine = if model_wdl {
        PalsEngine::new_with_boxed_checker_and_policies(
            config,
            model,
            Box::new(rz_search::cpu_checker::OwnedCpuChecker::new(Box::new(cpu_engine())).unwrap()),
            ResolverPolicy::ModelWdlRestricted,
            PostRepairRecheckPolicy::IterativeFrozenModelWdlV2,
        )
    } else {
        PalsEngine::new(config, model, cpu_engine())
    }
    .unwrap()
    .with_selected_review_context(
        ReviewSelectionPolicy::default(),
        ReviewIndexLimits::default(),
    )
    .unwrap();
    engine.new_game();
    let issuer = Arc::new(AtomicU64::new(100));
    let port = AdaptiveEnginePort::new(
        engine,
        position.clone(),
        authority,
        rz_contracts::Deadline {
            clock: clock.domain(),
            at: MonotonicTick(WINDOW.as_nanos() as u64),
        },
        PalsLimits {
            deadline: until,
            max_rounds: if model_wdl { 4 } else { 1 },
            max_cpu_nodes: node_limit,
            cpu_depth: if model_wdl { 2 } else { 1 },
        },
        Arc::new(move || Ok(OwnerId(issuer.fetch_add(1, Ordering::Relaxed)))),
    )
    .unwrap();
    let facts = Arc::new(NativeFacts::default());
    facts.child_gate.store(true, Ordering::Release);
    let owner_facts = Arc::clone(&facts);
    let owner_clock = clock.clone();
    let native = TicketedNativeService::<ReviewView, AdaptivePortQuestion, NativeFixture>::spawn(
        move || {
            Ok(NativeFixture {
                clock: owner_clock,
                facts: owner_facts,
                finish_ok: true,
            })
        },
        TicketServiceLimits::default(),
    )
    .unwrap();
    let worker = CpuWorker::spawn_notified(CpuWorkerId(0), cpu_engine(), native.signal()).unwrap();
    let mut driver = AdaptivePortDriver::new(
        port,
        until,
        Arc::new(native.signal()),
        rz_search::external_cpu::ExternalSlicePolicy::default(),
    )
    .unwrap();
    let root = driver.root_input().unwrap();
    let manager = AdaptiveExecutionManager::new(
        AdaptiveManagerRegistration {
            authority,
            hard_deadline: until,
            token_budget: 512,
            policy: CoordinatorPolicy::Fixed,
            first_request_sequence: 1,
            native_cpu_node_limit: node_limit,
            endpoints: CpuEndpointSelection::deployment(),
            controller: ControllerConfig::default(),
            device_digest: Digest([90; 32]),
            cpu_device_digest: Digest([91; 32]),
        },
        clock,
        native,
        vec![worker],
        None,
        driver,
    )
    .unwrap();
    (manager, facts, root, until, position)
}

#[test]
fn actual_search_port_commits_native_and_private_cpu_without_inline_model_calls() {
    use rz_search::pals::engine::{PalsCompletion, PalsConfig};
    let (mut manager, _, root, until, position) = actual_search_port_fixture(
        PalsConfig {
            beam_width: 1,
            line_plies: 1,
            ..PalsConfig::default()
        },
        false,
    );
    manager.prepare_root_ranking(root).unwrap();
    while manager.has_pending_work() {
        assert!(
            Instant::now() < until,
            "actual port timeout: {:?}",
            manager.failures().collect::<Vec<_>>()
        );
        manager.tick().unwrap();
        thread::sleep(Duration::from_millis(1));
    }
    assert!(manager.failures().next().is_none());
    let result = manager.driver().result(PalsCompletion::RoundLimit).unwrap();
    assert!(position.legal_moves().contains(&result.best_move.unwrap()));
    assert_eq!(result.counters.rounds, 1);
    assert!(result.counters.completed_cpu_tasks > 0);
    assert!(manager.counters().consumed_role_outputs > 0);
    assert!(manager.native_cpu_usage().unwrap().0 > 0);
    manager.begin_shutdown(until).unwrap();
    while manager.phase() != ManagerPhase::Finished {
        assert!(Instant::now() < until);
        manager.tick().unwrap();
        thread::sleep(Duration::from_millis(1));
    }
    assert!(manager.shutdown_receipt().unwrap().permits_archive_close());
}

#[test]
fn actual_model_wdl_fixed_root_role_budget_stops_before_a_seventeenth_call() {
    use rz_search::pals::engine::{PalsCompletion, PalsConfig};
    for limit in [1, 2, 16] {
        // Match the bounded CPU source profile's search shape and budgets.
        // Outputs remain authored fixtures, never actual neural/GPU evidence.
        let (mut manager, facts, root, until, position) = actual_search_port_fixture(
            PalsConfig {
                beam_width: 1,
                line_plies: 4,
                max_nodes: 4096,
                max_records: 64,
                max_role_calls: limit,
                cpu_nodes_per_task: 128,
            },
            true,
        );
        assert_eq!(manager.role_call_limit, Some(limit));
        manager.prepare_root_ranking(root).unwrap();
        let receipt = manager
            .run_root_until_observed(until, &AtomicBool::new(false), &mut |manager| {
                let (spent, reserved) = manager.role_call_budget_usage();
                assert!(spent + reserved <= limit);
                Ok(())
            })
            .unwrap();
        assert!(receipt.permits_root_reuse());
        assert!(
            manager.failures().next().is_none(),
            "{:?}",
            manager.failures().collect::<Vec<_>>()
        );
        assert!(manager.role_call_limit_reached());
        assert_eq!(manager.role_call_budget_usage(), (limit, 0));
        assert_eq!(manager.counters().submitted_role_tickets, limit);
        assert_eq!(manager.counters().physical_role_completions, limit);
        assert_eq!(manager.counters().consumed_role_outputs, limit);
        assert_eq!(manager.counters().unused_role_results, 0);
        assert_eq!(facts.accepted.lock().unwrap().len() as u64, limit);
        let result = manager
            .driver()
            .result(PalsCompletion::RoleCallLimit)
            .unwrap();
        assert_eq!(result.completion, PalsCompletion::RoleCallLimit);
        assert_eq!(result.counters.role_calls, limit);
        assert_eq!(result.counters.consumed_role_outputs, limit);
        assert!(position.legal_moves().contains(&result.best_move.unwrap()));
        let (cpu_spent, cpu_reserved) = manager.native_cpu_usage().unwrap();
        assert!(cpu_spent <= 1024);
        assert_eq!(cpu_reserved, 0);
    }
}

#[test]
fn root_ranking_fans_out_two_original_views_without_staling_or_double_consumption() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.counters().submitted_role_tickets == 3
    });
    assert_eq!(manager.counters().consumed_role_outputs, 1);
    assert_eq!(manager.driver().consumed.len(), 1);
    facts.child_gate.store(true, Ordering::Release);
    pump(&mut manager, until, |manager| {
        manager.driver().closed.len() == 2 && !manager.has_pending_work()
    });
    let consumed = &manager.driver().consumed;
    assert_eq!(consumed.len(), 3);
    assert_eq!(consumed[1].authority, consumed[2].authority);
    assert_eq!(consumed[1].authority.root, RootGeneration(1));
    assert_ne!(consumed[1].id, consumed[2].id);
    assert_ne!(consumed[1].question, consumed[2].question);
    assert_ne!(consumed[1].view.view_digest, consumed[2].view.view_digest);
    assert_eq!(facts.accepted.lock().unwrap().len(), 3);
    assert_eq!(manager.counters().consumed_role_outputs, 3);
    assert_eq!(manager.counters().completed_nn_inputs, 3);
    assert_eq!(manager.native_cpu_usage().unwrap(), (0, 0));
    assert!(manager.failures().next().is_none());
    let receipt = shutdown(&mut manager, until);
    assert!(receipt.permits_archive_close());
    assert_eq!(receipt.pending_tickets, 0);
}

#[cfg(feature = "search-work-receipts")]
#[test]
fn pending_snapshot_reads_two_original_questions_without_staling_or_consuming_them() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.roles.len() == 2 && manager.roles.iter().all(|binding| binding.ticket.is_some())
    });
    let before = manager.pending_question_snapshot().unwrap();
    assert_eq!(before.questions.len(), 2);
    let first = before.questions[0].ticket.unwrap();
    let second = before.questions[1].ticket.unwrap();
    assert_eq!(first.authority, second.authority);
    assert_ne!(first.id, second.id);
    assert_ne!(first.question, second.question);
    assert_ne!(first.view.view_digest, second.view.view_digest);
    for (row, binding) in before.questions.iter().zip(&manager.roles) {
        assert!(!row.consumption_recorded);
        assert_eq!(row.question, binding.input.question_digest);
        assert_eq!(row.view, binding.input.view_seal);
        assert_eq!(
            row.retained_view_digest,
            Digest(binding.input.view.digest())
        );
        assert_eq!(row.retained_view_seal_digest, row.view.view_digest);
        assert_eq!(
            row.selected_record_revisions,
            binding
                .input
                .view
                .records()
                .iter()
                .map(|record| record.revision)
                .collect::<Vec<_>>()
        );
    }
    // Reading twice changes no registry/manager transition. Native work can
    // progress independently while its last processed manager state is stable.
    assert_eq!(
        manager.pending_question_snapshot().unwrap().questions,
        before.questions
    );
    assert_eq!(manager.counters().consumed_role_outputs, 1);
    facts.child_gate.store(true, Ordering::Release);
    pump(&mut manager, until, |manager| {
        manager.driver().closed.len() == 2 && !manager.has_pending_work()
    });
    assert!(manager.driver().consumed.contains(&first));
    assert!(manager.driver().consumed.contains(&second));
    assert!(
        manager
            .pending_question_snapshot()
            .unwrap()
            .questions
            .is_empty()
    );
    assert!(manager.failures().next().is_none());
    assert!(shutdown(&mut manager, until).permits_archive_close());
}

#[cfg(feature = "search-work-receipts")]
#[test]
fn pending_snapshot_capacity_loss_is_observation_only() {
    let (mut manager, _, root, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
    manager.prepare_root_ranking(root).unwrap();
    let before = manager.pending_question_snapshot().unwrap();
    assert_eq!(before.questions.len(), 1);
    assert!(before.questions[0].ticket.is_none());
    let limit = manager.limits.role_tickets_max;
    manager.limits.role_tickets_max = 0; // Authored corrupt-observation boundary.
    assert_eq!(
        manager.pending_question_snapshot(),
        Err(AdaptivePendingSnapshotUnavailable::Capacity)
    );
    manager.limits.role_tickets_max = limit;
    assert_eq!(manager.phase(), ManagerPhase::Running);
    assert!(manager.failures().next().is_none());
    assert_eq!(
        manager.pending_question_snapshot().unwrap().questions,
        before.questions
    );
    assert!(shutdown(&mut manager, until).permits_archive_close());
}

#[cfg(feature = "search-work-receipts")]
#[test]
fn pending_snapshot_never_hides_a_registry_charge_without_a_bound_ticket() {
    let (mut manager, _, root, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
    manager.prepare_root_ranking(root).unwrap();
    let before = manager.pending_question_snapshot().unwrap();
    manager.roles[0].admission_charged = true; // Model a retained failed-bind gap.
    assert_eq!(
        manager.pending_question_snapshot(),
        Err(AdaptivePendingSnapshotUnavailable::RegistryBindingMismatch)
    );
    manager.roles[0].admission_charged = false;
    assert!(manager.failures().next().is_none());
    assert_eq!(
        manager.pending_question_snapshot().unwrap().questions,
        before.questions
    );
    assert!(shutdown(&mut manager, until).permits_archive_close());
}

#[test]
fn boxed_owner_transfer_retains_same_manager_on_rejection_and_closes_after_join() {
    let (manager, _, _, until) = fixture(FixtureScope::OwnCpuResume, true, 13);
    let manager = Box::new(manager);
    let original = std::ptr::from_ref(manager.as_ref());
    let mut rejected = match manager.into_persistent_parts_boxed() {
        Ok(_) => panic!("an active root cannot transfer its owner"),
        Err(rejected) => rejected,
    };
    assert_eq!(std::ptr::from_ref(rejected.manager.as_ref()), original);
    assert!(shutdown(&mut rejected.manager, until).permits_archive_close());
    let parts = match rejected.manager.into_process_parts_boxed() {
        Ok(parts) => parts,
        Err(rejected) => panic!("joined owner transfer: {}", rejected.error),
    };
    assert!(parts.receipt().permits_archive_close());
}

#[test]
fn two_actual_root_fences_reuse_one_native_factory_and_preserve_cpu_engine_work() {
    // This case observes a completed attempt across root reuse. Yield and its
    // final cancellation attempt are exercised separately below.
    let (mut manager, facts, root, until) = fixture(FixtureScope::OwnCpuResume, true, 13);
    facts.child_gate.store(true, Ordering::Release);
    let clock = manager.clock.clone();
    let mut registration = manager.registration;
    manager.prepare_root_ranking(root).unwrap();
    let first = manager
        .run_root_until(until, &AtomicBool::new(false))
        .unwrap();
    assert!(first.permits_root_reuse());
    assert!(!first.native_owner_joined);
    assert!(!facts.finish_called.load(Ordering::Acquire));
    let parts = match manager.into_persistent_parts() {
        Ok(parts) => parts,
        Err(rejected) => panic!("persistent transfer: {}", rejected.error),
    };
    let (native, mut cpu, external, mut driver, continuation, first) = parts.into_components();
    assert!(external.is_none());
    assert_eq!(cpu.len(), 1);
    let owner = cpu.pop().unwrap();
    let actual_cpu_work = owner
        .engine
        .last_attempt_work()
        .expect("actual CPU searched first root");
    assert!(actual_cpu_work.nodes > 0);
    registration.authority.root = RootGeneration(registration.authority.root.0 + 1);
    registration.first_request_sequence = continuation.first_request_sequence();
    let old_high_water = first.request_sequence_high_water;
    driver.authority = registration.authority;
    driver.scope = FixtureScope::TwoIndependentRoles;
    let focused = driver
        .stores
        .focus_actual_moves(driver.rules.rules().snapshot().clone())
        .unwrap();
    driver.state = driver.stores.situations.get(focused).unwrap().state;
    driver.closed.clear();
    driver.consumed.clear();
    let input = driver.input(RefinementStage::ProposerRanking, None);
    let worker =
        CpuWorker::spawn_notified(owner.profile.owner, owner.engine, native.signal()).unwrap();
    let mut next = AdaptiveExecutionManager::new_continued(
        registration,
        clock,
        native,
        vec![worker],
        None,
        driver,
        continuation,
    )
    .unwrap();
    next.prepare_root_ranking(input).unwrap();
    let second = next.run_root_until(until, &AtomicBool::new(false)).unwrap();
    assert!(second.permits_root_reuse());
    assert!(second.request_sequence_high_water > old_high_water);
    assert!(
        next.driver()
            .consumed
            .iter()
            .all(|ticket| ticket.id.sequence > old_high_water
                && ticket.authority.root == RootGeneration(2))
    );
    assert_eq!(facts.factories.load(Ordering::Acquire), 1);
    let parts = match next.into_persistent_parts() {
        Ok(parts) => parts,
        Err(rejected) => panic!("second persistent transfer: {}", rejected.error),
    };
    let (mut native, cpu, external, _, _, _) = parts.into_components();
    assert!(external.is_none());
    assert_eq!(cpu[0].engine.last_attempt_work(), Some(actual_cpu_work));
    // Process teardown is separate from both root quiescence receipts.
    native.begin_shutdown(until);
    while !native.try_join().unwrap() {
        assert!(Instant::now() < until);
        while native.poll().is_some() {}
        thread::sleep(Duration::from_millis(1));
    }
    assert!(facts.finish_called.load(Ordering::Acquire));
}

#[test]
fn actual_private_cpu_yield_resume_settles_exact_work_and_leaves_no_reservation() {
    // One native node forces a real checkpoint before this fixed job completes.
    let (mut manager, facts, root, until) = fixture(FixtureScope::OwnCpuResume, true, 1);
    assert!(manager.cpu_slice_diagnostic.get().is_none());
    let diagnostic =
        AdaptiveCpuSliceDiagnostic::new(AdaptiveCpuSliceDiagnosticOptions { records_max: 1 })
            .unwrap();
    assert!(diagnostic.snapshot().unwrap().authority.is_none());
    manager
        .install_cpu_slice_diagnostic(diagnostic.clone())
        .unwrap();
    let unused =
        AdaptiveCpuSliceDiagnostic::new(AdaptiveCpuSliceDiagnosticOptions { records_max: 1 })
            .unwrap();
    assert_eq!(
        manager.install_cpu_slice_diagnostic(unused),
        Err(AdaptiveCpuSliceDiagnosticError::AlreadyInstalled)
    );
    facts.child_gate.store(true, Ordering::Release);
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.driver().closed.len() == 1 && !manager.has_pending_work()
    });
    assert_eq!(manager.driver().actual_cpu.len(), 1);
    let actual = &manager.driver().actual_cpu[0];
    assert!(
        manager.counters().cpu_yields > 0,
        "actual traversal must cross slice boundaries"
    );
    assert_eq!(
        manager.counters().cpu_slice_receipts,
        manager.counters().cpu_yields + 1
    );
    assert_eq!(
        manager.native_cpu_usage().unwrap(),
        (actual.cumulative_nodes, 0)
    );
    assert_eq!(manager.counters().consumed_role_outputs, 2);
    assert!(manager.failures().next().is_none());
    // At least one Yield and its terminal receipt exceed the explicit cap1.
    // Keep the first actual settled sample; overflow must not affect search,
    // credit, continuation custody, semantic consume or the physical fence.
    let observed = diagnostic.snapshot().unwrap();
    assert_eq!(observed.authority, Some(actual.grant.job.authority));
    assert_eq!(
        observed.unavailable,
        Some(AdaptiveCpuSliceDiagnosticUnavailable::Overflow)
    );
    assert_eq!(observed.records.len(), 1);
    let first = &observed.records[0];
    assert_eq!(first.sequence, 1);
    assert_eq!(first.grant.job, actual.grant.job);
    assert_eq!(first.grant.owner, actual.grant.owner);
    assert_eq!(first.scope.job.id, actual.grant.job);
    assert_eq!(first.scope.view.authority, actual.grant.job.authority);
    assert!(first.continuation_retained && !first.terminal);
    assert!(matches!(
        first.result,
        AdaptiveCpuSliceResultObservation::Report {
            completion: CpuCompletion::Yielded,
            yield_ack_sequence: None,
            ..
        }
    ));
    // Fixed scheduling requested no Yield. A node/lease checkpoint alone
    // must not become a claimed request acknowledgement.
    assert!(
        first
            .controls
            .is_some_and(|controls| { controls.yield_sequence == 0 && controls.yield_ack == 0 })
    );
    let accounting = first.accounting_after_settlement.unwrap();
    assert_eq!(accounting.global_spent, first.actual_work.unwrap().nodes);
    assert_eq!(accounting.job_spent, first.cumulative_nodes);
    assert_eq!(accounting.global_reserved, 0);
    assert_eq!(accounting.job_reserved, 0);
    assert!(!accounting.job_quarantined);
    assert_eq!(
        manager.install_cpu_slice_diagnostic(diagnostic.clone()),
        Err(AdaptiveCpuSliceDiagnosticError::AdmissionAlreadyStarted)
    );
    let spent = actual.cumulative_nodes;
    let receipt = shutdown(&mut manager, until);
    assert_eq!(receipt.native_cpu_spent, spent);
    assert_eq!(receipt.native_cpu_reserved, 0);
    assert!(receipt.permits_archive_close());
}

#[test]
fn shutdown_without_physical_native_fence_quarantines_and_refuses_archive_close() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::TwoIndependentRoles, false, 4096);
    drop(root);
    let receipt = shutdown(&mut manager, until);
    assert!(facts.finish_called.load(Ordering::Acquire));
    assert!(receipt.native_owner_joined);
    assert_eq!(receipt.own_cpu_joined, 1);
    assert!(!receipt.native_physical_shutdown_confirmed);
    assert!(receipt.quarantined);
    assert!(!receipt.permits_archive_close());
    assert_eq!(manager.phase(), ManagerPhase::Quarantined);
    let rejected = manager
        .driver_mut()
        .input(RefinementStage::ProposerRanking, None);
    assert!(manager.prepare_root_ranking(rejected).is_err());
    assert_eq!(receipt.counters.consumed_role_outputs, 0);
}

#[test]
fn failed_role_hot_commit_retains_original_ticket_until_actual_owner_drain() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
    facts.child_gate.store(true, Ordering::Release);
    manager.driver_mut().fail_role_once = true;
    manager.prepare_root_ranking(root).unwrap();
    let receipt = manager.run_until(until, &AtomicBool::new(false)).unwrap();
    assert!(receipt.permits_archive_close());
    assert_eq!(receipt.pending_tickets, 0);
    assert_eq!(receipt.native_cpu_reserved, 0);
    assert_eq!(receipt.counters.consumed_role_outputs, 1);
    assert!(manager.failures().any(|error| matches!(error,
        AdaptiveManagerError::Semantic(message) if message == "authored role hot-commit failure")));
    assert!(!manager.failures().any(|error| matches!(
        error,
        AdaptiveManagerError::Invalid("unknown semantic retirement")
    )));
    assert!(facts.finish_called.load(Ordering::Acquire));
    assert!(manager.roles.is_empty());
}

#[test]
fn invalid_continued_registration_returns_live_native_and_same_cpu_owner() {
    let (mut manager, facts, _, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
    let mut registration = manager.registration;
    let clock = manager.clock.clone();
    manager.begin_root_quiesce(until).unwrap();
    pump(&mut manager, until, |manager| {
        manager.phase() == ManagerPhase::Finished
    });
    let parts = match manager.into_persistent_parts() {
        Ok(parts) => parts,
        Err(rejected) => panic!("initial root fence: {}", rejected.error),
    };
    let (native, mut cpu, external, driver, continuation, receipt) = parts.into_components();
    assert!(receipt.permits_root_reuse());
    let cpu = cpu.remove(0);
    let owner = cpu.profile.owner;
    let worker = CpuWorker::spawn_notified(owner, cpu.engine, native.signal()).unwrap();
    registration.authority.root = RootGeneration(2);
    registration.first_request_sequence = continuation.first_request_sequence();
    registration.native_cpu_node_limit = 0;
    let rejected = match AdaptiveExecutionManager::new_continued_retaining(
        registration,
        clock,
        AdaptiveManagerSources {
            native,
            own_workers: vec![worker],
            external,
            driver,
        },
        continuation,
    ) {
        Ok(_) => panic!("invalid node registration admitted"),
        Err(rejected) => rejected,
    };
    assert!(rejected.continuation.is_some());
    assert_eq!(rejected.sources.own_workers[0].profile().owner, owner);
    assert!(
        rejected
            .sources
            .native
            .root_fence()
            .unwrap()
            .permits_reuse()
    );
    assert!(!facts.finish_called.load(Ordering::Acquire));
    assert_eq!(facts.factories.load(Ordering::Acquire), 1);
    let mut sources = rejected.sources;
    sources.native.begin_shutdown(until);
    sources.own_workers[0].request_shutdown();
    while !sources.native.try_join().unwrap() {
        while sources.native.poll().is_some() {}
        assert!(Instant::now() < until);
        thread::sleep(Duration::from_millis(1));
    }
    loop {
        if let Some(stopped) = sources.own_workers[0].try_join().unwrap() {
            assert!(stopped.thread_joined && stopped.engine_recoverable);
            break;
        }
        assert!(Instant::now() < until);
        thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn next_view_failure_preserves_actual_original_role_consumption() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
    manager.driver_mut().fail_after_role_consumption_once = true;
    facts.child_gate.store(true, Ordering::Release);
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.failures().next().is_some()
    });
    let receipt = shutdown(&mut manager, until);
    assert!(receipt.permits_archive_close());
    assert_eq!(manager.counters().consumed_role_outputs, 2);
    assert_eq!(manager.driver().consumed.len(), 2);
    assert_eq!(manager.driver().closed.len(), 0);
    assert!(manager.roles.is_empty());
    assert!(matches!(
        manager.failures().next(),
        Some(AdaptiveManagerError::Semantic(_))
    ));
    assert!(facts.finish_called.load(Ordering::Acquire));
}

#[test]
fn failed_consumption_readback_retains_binding_after_real_owner_fence() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
    manager.driver_mut().fail_after_role_consumption_once = true;
    manager.driver_mut().fail_role_consumption_readback = true;
    facts.child_gate.store(true, Ordering::Release);
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.failures().next().is_some()
    });

    let receipt = shutdown(&mut manager, until);
    assert!(receipt.native_physical_shutdown_confirmed);
    assert!(receipt.quarantined);
    assert!(!receipt.permits_archive_close());
    let original = manager
        .roles
        .iter()
        .find(|binding| binding.consumption_unknown)
        .expect("uncertain original binding must remain in custody");
    assert!(!original.consumption_recorded);
    assert!(manager.driver().consumed.contains(&original.ticket.unwrap()));
    assert!(manager.failures().any(|error| matches!(
        error,
        AdaptiveManagerError::Semantic(detail)
            if detail == "authored role consumption readback failure"
    )));
    assert!(facts.finish_called.load(Ordering::Acquire));
}

#[test]
fn original_prepared_echo_after_cancel_retirement_never_recancels_or_admits() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
    let preparation = manager.prepare_root_ranking(root).unwrap();
    // Receive the actual owner event without letting the manager admit a
    // RequestId. This is an original queued Prepared, not a fabricated seal.
    let prepared = loop {
        let version = manager.native.signal().version();
        if let Some(event) = manager.native.poll() {
            match event {
                TicketNativeEvent::Prepared(prepared) => break prepared,
                other => manager.native_event(other).unwrap(),
            }
        } else {
            assert!(Instant::now() < until, "actual Prepared did not arrive");
            manager
                .native
                .signal()
                .wait_changed(version, Duration::from_millis(10));
        }
    };
    assert_eq!(prepared.preparation, preparation);
    assert_eq!(facts.prepared.load(Ordering::Acquire), 1);
    assert_eq!(manager.native.occupancy(), 1);
    assert_eq!(manager.roles.len(), 1);
    assert!(manager.roles[0].ticket.is_none());
    let original_view = Arc::downgrade(&manager.roles[0].input.view);
    // Hold the actual owner after a scan that saw this still-open Prepared.
    // Closing now fixes the race where cancellation and quiesce arrive in the
    // same iteration after the ordinary unsubmitted-retirement scan.
    assert!(matches!(
        manager.native.pause_after_cancellation_scan_for_test(
            PreparationId(preparation.0 + 1_000_000),
            until
        ),
        Err(TicketServiceError::UnknownPreparation)
    ));
    let (scan_entered, release_scan) = manager
        .native
        .pause_after_cancellation_scan_for_test(preparation, until)
        .unwrap();
    scan_entered
        .recv_timeout(until.saturating_duration_since(Instant::now()))
        .expect("original owner did not reach the one-shot cancellation-scan gate");
    assert!(matches!(
        manager
            .native
            .pause_after_cancellation_scan_for_test(preparation, until),
        Err(TicketServiceError::Duplicate)
    ));
    assert_eq!(manager.native.occupancy(), 1);
    assert!(!facts.quiesce_called.load(Ordering::Acquire));
    manager.begin_root_quiesce(until).unwrap();
    assert!(manager.roles[0].cancellation_requested);
    release_scan.send(()).unwrap();
    // Do not pump manager events: let its real successful cancel and the
    // original worker retire the unticketed preparation first. The queued
    // PreparationFailed/RootQuiesced remain for their one original delivery.
    while !facts.quiesce_called.load(Ordering::Acquire) || manager.native.occupancy() != 0 {
        let version = manager.native.signal().version();
        assert!(
            Instant::now() < until,
            "original cancel/quiesce did not finish"
        );
        manager
            .native
            .signal()
            .wait_changed(version, Duration::from_millis(10));
    }
    assert!(original_view.upgrade().is_some());
    assert_eq!(manager.counters().canceled_preparations, 0);
    // A proved cancel for this original binding must not permit another ID
    // or an altered immutable echo to take the same normal closing path.
    let mut foreign = prepared;
    foreign.preparation = PreparationId(preparation.0 + 1_000_000);
    assert!(matches!(
        manager.native_event(TicketNativeEvent::Prepared(foreign)),
        Err(AdaptiveManagerError::Invalid(_))
    ));
    let mut altered = prepared;
    altered.question = Digest([77; 32]);
    assert!(matches!(
        manager.native_event(TicketNativeEvent::Prepared(altered)),
        Err(AdaptiveManagerError::Invalid(_))
    ));
    assert_eq!(manager.counters().canceled_preparations, 0);
    manager
        .native_event(TicketNativeEvent::Prepared(prepared))
        .unwrap();
    assert_eq!(manager.counters().canceled_preparations, 1);
    // An exact duplicate is still a protocol error, not another normal cancel
    // or permission to charge the same prepared input twice.
    assert!(matches!(
        manager.native_event(TicketNativeEvent::Prepared(prepared)),
        Err(AdaptiveManagerError::Invalid(_))
    ));
    assert_eq!(manager.counters().canceled_preparations, 1);
    let mut failed = false;
    let mut fenced = false;
    while !failed || !fenced {
        let version = manager.native.signal().version();
        if let Some(event) = manager.native.poll() {
            match &event {
                TicketNativeEvent::PreparationFailed {
                    preparation: actual,
                    failure,
                } => {
                    assert_eq!(*actual, preparation);
                    assert_eq!(*failure, RoleFailure::Canceled);
                    assert!(!failed, "original preparation retired more than once");
                    failed = true;
                }
                TicketNativeEvent::RootQuiesced(fence) => {
                    assert!(
                        failed,
                        "original preparation terminal must precede root fence"
                    );
                    assert!(!fenced, "original root fenced more than once");
                    assert!(facts.quiesce_called.load(Ordering::Acquire));
                    assert_eq!(fence.authority(), manager.registration.authority);
                    assert!(fence.permits_reuse());
                    fenced = true;
                }
                _ => panic!("unsubmitted original preparation produced an unexpected event"),
            }
            manager.native_event(event).unwrap();
        } else {
            assert!(
                Instant::now() < until,
                "original terminal/fence did not arrive: failed={failed}; fenced={fenced}; quiesce_called={}; occupancy={}; owner_fence={:?}",
                facts.quiesce_called.load(Ordering::Acquire),
                manager.native.occupancy(),
                manager.native.root_fence()
            );
            manager
                .native
                .signal()
                .wait_changed(version, Duration::from_millis(10));
        }
    }
    pump(&mut manager, until, |manager| {
        manager.phase() == ManagerPhase::Finished
    });
    assert!(manager.failures().next().is_none());
    assert!(manager.roles.is_empty());
    assert!(original_view.upgrade().is_none());
    assert!(
        manager.native.poll().is_none(),
        "original terminal delivered once"
    );
    assert_eq!(manager.role_call_budget_usage(), (0, 0));
    assert_eq!(manager.counters().submitted_role_tickets, 0);
    assert_eq!(manager.counters().physical_role_completions, 0);
    assert_eq!(manager.counters().consumed_role_outputs, 0);
    assert_eq!(manager.native_cpu_usage().unwrap(), (0, 0));
    assert!(facts.started.lock().unwrap().is_empty());
    assert!(facts.accepted.lock().unwrap().is_empty());
    assert!(manager.shutdown_receipt().unwrap().permits_root_reuse());
    let parts = match manager.into_persistent_parts() {
        Ok(parts) => parts,
        Err(rejected) => panic!("original canceled root owner recovery: {}", rejected.error),
    };
    let (mut native, _, _, _, _, _) = parts.into_components();
    native.begin_shutdown(until);
    while !native.try_join().unwrap() {
        while native.poll().is_some() {}
        assert!(
            Instant::now() < until,
            "original native process owner did not join"
        );
        thread::sleep(Duration::from_millis(1));
    }
    assert!(facts.finish_called.load(Ordering::Acquire));
}

#[test]
fn changed_tick_observer_can_close_root_before_late_child_consumption() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
    assert!(manager.root_observation_snapshot().is_err());
    manager.prepare_root_ranking(root).unwrap();
    let mut saw_original_ranking = false;
    let receipt = manager
        .run_root_until_observed(until, &AtomicBool::new(false), &mut |manager| {
            if !saw_original_ranking && manager.counters().consumed_role_outputs == 1 {
                saw_original_ranking = true;
                assert_eq!(manager.phase(), ManagerPhase::Running);
                manager.begin_root_quiesce(until)?;
            }
            Ok(())
        })
        .unwrap();
    assert!(saw_original_ranking);
    assert!(receipt.permits_root_reuse());
    assert_eq!(manager.counters().consumed_role_outputs, 1);
    assert!(!facts.finish_called.load(Ordering::Acquire));
    let observed = manager.root_observation_snapshot().unwrap();
    assert_eq!(observed.authority(), receipt.authority);
    assert_eq!(observed.close_mode(), AdaptiveCloseMode::RootQuiesce);
    assert_eq!(observed.manager_counters(), manager.counters());
    assert_eq!(observed.cpu_outstanding_reserved(), 0);
    assert!(observed.role_ticket_accounting().admitted >= 1);
    assert_eq!(observed.cache_evaluations_accepted(), None);
    let parts = match manager.into_persistent_parts() {
        Ok(parts) => parts,
        Err(rejected) => panic!("observed root owner recovery: {}", rejected.error),
    };
    let (mut native, _, _, _, _, _) = parts.into_components();
    native.begin_shutdown(until);
    while !native.try_join().unwrap() {
        while native.poll().is_some() {}
        assert!(Instant::now() < until);
        thread::sleep(Duration::from_millis(1));
    }
    assert!(facts.finish_called.load(Ordering::Acquire));
}

#[test]
fn cooperative_collection_failure_drains_active_native_before_retained_semantic_error() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
    // Hold the actual first authored physical run until its same-owner cancel.
    // Merely having a submitted ticket is not evidence of an active run.
    facts.block_after_first_run.store(true, Ordering::Release);
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |_| {
        facts.started.lock().unwrap().len() == 1
    });
    assert!(facts.accepted.lock().unwrap().is_empty());
    let original = AdaptiveCollectionError::new(
        AdaptiveCollectionStage::Settled,
        "FixtureCollectionCaller",
        "original cooperative failure while physical role is active",
    );
    manager
        .record_collection_caller_failure(original.clone(), until)
        .unwrap();
    assert_eq!(manager.phase(), ManagerPhase::Draining);
    assert!(!manager.shutdown_receipt().unwrap().permits_root_reuse());
    // Uses the existing original deadline and real cancel/Retired/root fence,
    // rather than clearing the active run or inventing a completed output.
    let receipt = manager.run_until(until, &AtomicBool::new(false)).unwrap();
    assert!(receipt.permits_root_reuse());
    assert_eq!(manager.phase(), ManagerPhase::Finished);
    assert_eq!(receipt.native_cpu_reserved, 0);
    assert_eq!(receipt.pending_tickets, 0);
    assert_eq!(manager.counters().consumed_role_outputs, 0);
    assert!(facts.accepted.lock().unwrap().is_empty());
    assert!(facts.quiesce_called.load(Ordering::Acquire));
    assert!(!facts.finish_called.load(Ordering::Acquire));
    assert!(matches!(
        manager.failures().next(),
        Some(AdaptiveManagerError::Collection(error)) if error == &original
    ));
    // A physical reuse receipt never erases the retained semantic failure.
    // Finish the original process owner explicitly; no native thread is left
    // active merely because this fixture's root fence was successful.
    let parts = match manager.into_persistent_parts() {
        Ok(parts) => parts,
        Err(rejected) => panic!("failed Collection root owner recovery: {}", rejected.error),
    };
    let (mut native, _, _, _, _, _) = parts.into_components();
    native.begin_shutdown(until);
    while !native.try_join().unwrap() {
        while native.poll().is_some() {}
        assert!(
            Instant::now() < until,
            "original physical process finish timed out"
        );
        thread::sleep(Duration::from_millis(1));
    }
    assert!(facts.finish_called.load(Ordering::Acquire));
}

#[test]
fn cancel_during_actual_cpu_slice_preserves_final_receipts_before_archive_close() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::OwnCpuResume, true, 13);
    facts.child_gate.store(true, Ordering::Release);
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.own[0].binding.is_some()
    });
    manager.begin_shutdown(until).unwrap();
    assert!(!manager.shutdown_receipt().unwrap().permits_archive_close());
    let receipt = shutdown(&mut manager, until);
    assert_eq!(receipt.native_cpu_reserved, 0);
    assert_eq!(receipt.own_cpu_joined, receipt.expected_own_cpu_owners);
    assert!(receipt.permits_archive_close());
    assert!(
        manager.failures().next().is_none(),
        "{:?}",
        manager.failures().collect::<Vec<_>>()
    );
    // Revoke cannot accept the late CPU value as another semantic step.
    assert!(manager.driver().actual_cpu.is_empty());
}

#[test]
fn failed_hot_yield_commit_retains_actual_receipt_until_cancel_settlement() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::OwnCpuResume, true, 1);
    let diagnostic = AdaptiveCpuSliceDiagnostic::new(AdaptiveCpuSliceDiagnosticOptions {
        records_max: CPU_SLICE_DIAGNOSTIC_RECORDS_MAX,
    })
    .unwrap();
    manager
        .install_cpu_slice_diagnostic(diagnostic.clone())
        .unwrap();
    manager.driver_mut().fail_raw_once = true;
    facts.child_gate.store(true, Ordering::Release);
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.failures().next().is_some()
    });
    assert!(matches!(
        manager.failures().next(),
        Some(AdaptiveManagerError::Semantic(_))
    ));
    // A settled observation precedes semantic handoff. It cannot replace the
    // original failed handoff with success or claim its receipt was consumed.
    assert!(
        diagnostic
            .snapshot()
            .unwrap()
            .records
            .iter()
            .any(|record| { record.continuation_retained && !record.terminal })
    );
    let receipt = shutdown(&mut manager, until);
    assert_eq!(receipt.native_cpu_reserved, 0);
    assert!(receipt.permits_archive_close());
    assert!(!manager.driver().raw_cpu_receipts.is_empty());
    assert!(manager.driver().actual_cpu.is_empty());
}

#[cfg(test)]
mod adaptive_partial_feedback {
    use super::*;
    use rz_search::cpu::{CpuIterationProgress, CpuScoreScope};
    use rz_search::cpu_worker::{CpuJobTermination, CpuWorkerEvent, CpuWorkerProgress};

    // Only the continuation is authored. Every tested slice and retained token
    // comes from the same real PausedStack owner used by the existing fixture.
    struct PartialDriver {
        inner: DriverFixture,
        admissions: Vec<(CoordinatorWork<ReviewView, u64>, CpuJobEnvelope, SliceGrant)>,
        partial_commits: Vec<(CoordinatorWork<ReviewView, u64>, CpuSliceReport)>,
        terminal_commits: Vec<CpuSliceReport>,
        detached: Vec<(CoordinatorWork<ReviewView, u64>, CpuSliceReport)>,
        discarded: Vec<(CoordinatorWork<ReviewView, u64>, CpuJobTermination)>,
    }

    impl AdaptiveSearchDriver<u64> for PartialDriver {
        fn root_rules(&self) -> Arc<ContractState> {
            self.inner.root_rules()
        }
        fn ranking_published(
            &mut self,
            ranking: &ProposerRankingPublished,
        ) -> Result<Vec<CandidateFrame<u64>>, AdaptiveManagerError> {
            self.inner.ranking_published(ranking)
        }
        fn validate_role(
            &self,
            work: &CoordinatorWork<ReviewView, u64>,
            ticket: &RoleTicketEnvelope,
            output: &RoleOutput,
        ) -> Result<(), AdaptiveManagerError> {
            self.inner.validate_role(work, ticket, output)
        }
        fn commit_role(
            &mut self,
            work: &CoordinatorWork<ReviewView, u64>,
            ticket: &RoleTicketEnvelope,
            output: &RoleOutput,
            physical: Option<TicketExecutionEvidence>,
        ) -> Result<CheckedStep<u64>, AdaptiveManagerError> {
            self.inner.commit_role(work, ticket, output, physical)
        }
        fn cpu_question(
            &mut self,
            work: &CoordinatorWork<ReviewView, u64>,
            profiles: &[CpuWorkerProfile],
            external: Option<&ExternalCpuConfig>,
            endpoint: CpuEndpoint,
            physical_budget_exhausted: bool,
        ) -> Result<CpuQuestion, AdaptiveManagerError> {
            self.inner.cpu_question(
                work,
                profiles,
                external,
                endpoint,
                physical_budget_exhausted,
            )
        }
        fn cpu_admitted(
            &mut self,
            work: &CoordinatorWork<ReviewView, u64>,
            job: &CpuJobEnvelope,
            grant: &SliceGrant,
        ) -> Result<(), AdaptiveManagerError> {
            if let Some((original, original_job, original_grant)) = self.admissions.first() {
                assert_eq!(work.frame, original.frame);
                assert_eq!(work.input.stage, original.input.stage);
                assert_eq!(work.input.question_digest, original.input.question_digest);
                assert_eq!(work.input.view_seal, original.input.view_seal);
                assert_eq!(job, original_job);
                assert_eq!(grant.owner, original_grant.owner);
            }
            self.admissions.push((work.clone(), *job, *grant));
            Ok(())
        }
        fn cpu_partial_continuation_eligible(
            &self,
            work: &CoordinatorWork<ReviewView, u64>,
            report: &CpuSliceReport,
        ) -> bool {
            work.input.stage == RefinementStage::CpuValidation
                && !report.terminal
                && !report.admission_rejected
                && report.continuation_retained
                && report.actual_work.is_some()
                && report.result.as_ref().is_ok_and(|actual| {
                    actual.report.completion == CpuCompletion::Yielded
                        && actual.report.score_scope == CpuScoreScope::CompletedIteration
                        && actual.report.completed_depth > 0
                })
        }
        fn commit_cpu(
            &mut self,
            work: &CoordinatorWork<ReviewView, u64>,
            report: &CpuSliceReport,
        ) -> Result<CheckedStep<u64>, AdaptiveManagerError> {
            let transition = if report.terminal {
                assert!(!report.continuation_retained);
                self.terminal_commits.push(report.clone());
                FrameTransition::Unknown {
                    reason: "authored terminal-only control ends after the actual CPU receipt",
                }
            } else {
                assert!(self.cpu_partial_continuation_eligible(work, report));
                assert!(self.partial_commits.is_empty());
                assert!(
                    !self
                        .inner
                        .raw_cpu_receipts
                        .contains(&report.grant.execution)
                );
                self.partial_commits.push((work.clone(), report.clone()));
                let first =
                    BoardMove::try_from(work.original_first_move.decode().unwrap()).unwrap();
                let mut input = self
                    .inner
                    .input(RefinementStage::CriticDivergence, Some(first));
                input.question_digest =
                    receipt_digest(b"authored-partial-next-question", report.grant.execution);
                FrameTransition::Continue(input)
            };
            Ok(CheckedStep {
                evidence: receipt_digest(b"authored-partial-feedback", report.grant.execution),
                transition,
                fresh_pair: None,
                rules_pair: None,
                queued_recheck: None,
            })
        }
        fn cpu_slice_observed(
            &mut self,
            work: &CoordinatorWork<ReviewView, u64>,
            report: &CpuSliceReport,
        ) -> Result<(), AdaptiveManagerError> {
            self.inner.cpu_slice_observed(work, report)
        }
        fn cpu_detached_slice_observed(
            &mut self,
            work: &CoordinatorWork<ReviewView, u64>,
            report: &CpuSliceReport,
        ) -> Result<(), AdaptiveManagerError> {
            self.inner.cpu_slice_observed(work, report)?;
            self.detached.push((work.clone(), report.clone()));
            Ok(())
        }
        fn cpu_token_discarded(
            &mut self,
            work: &CoordinatorWork<ReviewView, u64>,
            termination: &CpuJobTermination,
        ) -> Result<(), AdaptiveManagerError> {
            self.discarded.push((work.clone(), termination.clone()));
            Ok(())
        }
        fn commit_external(
            &mut self,
            work: &CoordinatorWork<ReviewView, u64>,
            report: &ExternalWorkerReport,
        ) -> Result<CheckedStep<u64>, AdaptiveManagerError> {
            self.inner.commit_external(work, report)
        }
        fn publish_closed(&mut self, closed: &ClosedFrame) -> Result<(), AdaptiveManagerError> {
            self.inner.publish_closed(closed)
        }
        fn role_consumed(
            &mut self,
            ticket: &RoleTicketEnvelope,
            view: &Arc<ReviewView>,
        ) -> Result<(), AdaptiveManagerError> {
            self.inner.role_consumed(ticket, view)
        }
        fn role_consumption_recorded(
            &self,
            ticket: &RoleTicketEnvelope,
            view: &ReviewView,
        ) -> Result<bool, AdaptiveManagerError> {
            self.inner.role_consumption_recorded(ticket, view)
        }
    }

    type PartialManager = AdaptiveExecutionManager<u64, NativeFixture, PartialDriver>;

    fn partial_fixture(
        policy: CoordinatorPolicy,
        root_generation: RootGeneration,
    ) -> (
        PartialManager,
        Arc<NativeFacts>,
        FrameInput<ReviewView, u64>,
        Instant,
    ) {
        let position = Position::from_fen("3rk3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
        let clock = ContractSystemClock::new(ProcessEpoch(117));
        let until = clock.origin() + WINDOW;
        let authority = SearchAuthority {
            epoch: ProcessEpoch(117),
            game: GameGeneration(1),
            root: root_generation,
            implementation: Digest([19; 32]),
        };
        let rules = Arc::new(
            ContractPosition::new(OwnerId(17), position.clone())
                .export()
                .unwrap(),
        );
        let mut stores = PalsStores::new(StoreLimits::default());
        let root_registration = stores
            .prepare_hot_root_registration(position.snapshot(), root_generation.0)
            .unwrap();
        let root = stores.hot_focus_registered_root(root_registration).unwrap();
        assert_eq!(stores.generation(), authority.root.0);
        let state = stores.situations.get(root).unwrap().state;
        let mut driver = PartialDriver {
            inner: DriverFixture {
                authority,
                role_call_limit: None,
                rules,
                stores,
                index: ReviewContextIndex::default(),
                state,
                scope: FixtureScope::OwnCpuResume,
                consumed: Vec::new(),
                closed: Vec::new(),
                expected_cpu: BTreeMap::new(),
                actual_cpu: Vec::new(),
                fail_raw_once: false,
                semantic_owner_busy: false,
                fail_role_once: false,
                fail_after_role_consumption_once: false,
                fail_role_consumption_readback: false,
                #[cfg(feature = "pals-adaptive-diagnostics")]
                fail_publish_closed_once: false,
                raw_cpu_receipts: Vec::new(),
                reciprocal_questions: BTreeMap::new(),
                reciprocal_waits: Vec::new(),
                reciprocal_replies: 0,
                peer_sources: Vec::new(),
                peer_opinions: 0,
                peer_cpu_answers: 0,
                peer_repeat_question: false,
                replay_frontier: None,
                block_replay_while_reserved: false,
                blocked_replays: 0,
            },
            admissions: Vec::new(),
            partial_commits: Vec::new(),
            terminal_commits: Vec::new(),
            detached: Vec::new(),
            discarded: Vec::new(),
        };
        let input = driver.inner.input(RefinementStage::ProposerRanking, None);
        let facts = Arc::new(NativeFacts::default());
        let thread_facts = Arc::clone(&facts);
        let native_clock = clock.clone();
        let native = TicketedNativeService::spawn(
            move || {
                thread_facts.factories.fetch_add(1, Ordering::Relaxed);
                Ok(NativeFixture {
                    clock: native_clock,
                    facts: thread_facts,
                    finish_ok: true,
                })
            },
            TicketServiceLimits::default(),
        )
        .unwrap();
        let worker =
            CpuWorker::spawn_notified(CpuWorkerId(0), cpu_engine(), native.signal()).unwrap();
        let registration = AdaptiveManagerRegistration {
            authority,
            hard_deadline: until,
            token_budget: 512,
            policy,
            first_request_sequence: 1,
            native_cpu_node_limit: TOTAL_NODES,
            endpoints: CpuEndpointSelection::deployment(),
            controller: ControllerConfig {
                policy,
                fixed_node_credit: 1,
                ..ControllerConfig::default()
            },
            device_digest: Digest([71; 32]),
            cpu_device_digest: Digest([72; 32]),
        };
        let manager =
            AdaptiveExecutionManager::new(registration, clock, native, vec![worker], None, driver)
                .unwrap();
        (manager, facts, input, until)
    }

    fn pump_partial(
        manager: &mut PartialManager,
        until: Instant,
        predicate: impl Fn(&PartialManager) -> bool,
    ) {
        while !predicate(manager) {
            assert!(Instant::now() < until, "partial fixture timeout");
            manager.tick().unwrap();
            assert!(manager.failures().next().is_none());
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn start_cpu(
        manager: &mut PartialManager,
        facts: &NativeFacts,
        root: FrameInput<ReviewView, u64>,
        until: Instant,
    ) {
        facts.child_gate.store(true, Ordering::Release);
        manager.prepare_root_ranking(root).unwrap();
        pump_partial(manager, until, |manager| manager.own[0].binding.is_some());
    }

    fn next_slice(manager: &mut PartialManager, until: Instant) -> CpuWorkerEventObservation {
        loop {
            assert!(Instant::now() < until, "actual CPU receipt timeout");
            if let Some(event) = manager.own[0]
                .worker
                .as_ref()
                .unwrap()
                .try_event_observed()
                .unwrap()
            {
                return event;
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn forward_first(manager: &mut PartialManager, until: Instant) -> CpuSliceReport {
        loop {
            let event = next_slice(manager, until);
            let (event, _) = event.into_parts();
            let CpuWorkerEvent::Slice(report) = event else {
                panic!("live original owner terminated before its first completed iteration");
            };
            assert!(
                !report.terminal,
                "one-node slices must retain the depth-two traversal"
            );
            manager
                .own_event(0, CpuWorkerEvent::Slice(report.clone()))
                .unwrap();
            if manager.own[0].binding.as_ref().unwrap().feedback_forwarded {
                return report;
            }
            manager.resume_cpu(0, 1, Duration::from_millis(8)).unwrap();
        }
    }

    fn frame_fingerprint(
        manager: &PartialManager,
        frame: FrameId,
    ) -> Option<(FramePhase, RefinementStage, Digest, ReviewViewSeal, Digest)> {
        manager.machine.frame(frame).map(|current| {
            (
                current.phase,
                current.input.stage,
                current.input.question_digest,
                current.input.view_seal,
                current.input.public_context,
            )
        })
    }

    fn shutdown_partial(manager: &mut PartialManager, until: Instant) -> AdaptiveShutdownReceipt {
        manager.begin_shutdown(until).unwrap();
        while {
            let receipt = manager.shutdown_receipt().unwrap();
            !receipt.native_owner_joined || receipt.own_cpu_joined != 1
        } {
            assert!(Instant::now() < until, "partial owner shutdown timeout");
            manager.tick().unwrap();
            thread::sleep(Duration::from_millis(1));
        }
        manager.shutdown_receipt().unwrap()
    }

    #[test]
    fn real_completed_iteration_handoff_resumes_only_original_custody() {
        let (mut manager, facts, root, until) =
            partial_fixture(CoordinatorPolicy::Adaptive, RootGeneration(1));
        let diagnostic = AdaptiveCpuSliceDiagnostic::new(AdaptiveCpuSliceDiagnosticOptions {
            records_max: CPU_SLICE_DIAGNOSTIC_RECORDS_MAX,
        })
        .unwrap();
        manager
            .install_cpu_slice_diagnostic(diagnostic.clone())
            .unwrap();
        start_cpu(&mut manager, &facts, root, until);
        let first = forward_first(&mut manager, until);
        let original = manager.own[0].binding.as_ref().unwrap().work.clone();
        let original_job = manager.own[0].binding.as_ref().unwrap().envelope;
        let fingerprint = frame_fingerprint(&manager, original.frame);
        assert_eq!(fingerprint.as_ref().unwrap().0, FramePhase::Ready);
        assert_eq!(
            fingerprint.as_ref().unwrap().1,
            RefinementStage::CriticDivergence
        );
        assert_ne!(
            fingerprint.as_ref().unwrap().2,
            original.input.question_digest
        );
        assert_eq!(manager.driver.partial_commits.len(), 1);
        assert!(
            !manager
                .driver
                .inner
                .raw_cpu_receipts
                .contains(&first.grant.execution)
        );

        // A repeated publication of this already-settled real iteration cannot
        // bind its execution to the newly materialized logical question.
        let actual = &first.result.as_ref().unwrap().report;
        let progress = CpuWorkerProgress {
            job: first.grant.job,
            execution: first.grant.execution,
            owner: first.grant.owner,
            sequence: first.progress_sequence,
            iteration: CpuIterationProgress {
                best_move: actual.best_move,
                score: actual.score,
                completed_depth: actual.completed_depth,
                nodes: actual.nodes,
            },
        };
        manager.own_progress(0, progress.clone()).unwrap();
        manager.own_progress(0, progress).unwrap();
        assert_eq!(frame_fingerprint(&manager, original.frame), fingerprint);

        let terminal = loop {
            manager.resume_cpu(0, 1, Duration::from_millis(8)).unwrap();
            let binding = manager.own[0].binding.as_ref().unwrap();
            assert!(binding.feedback_forwarded);
            assert_eq!(binding.envelope, original_job);
            assert_eq!(
                binding.work.input.question_digest,
                original.input.question_digest
            );
            assert_eq!(binding.work.input.view_seal, original.input.view_seal);
            assert_eq!(frame_fingerprint(&manager, original.frame), fingerprint);
            let (event, _) = next_slice(&mut manager, until).into_parts();
            let CpuWorkerEvent::Slice(report) = event else {
                panic!("resumed original traversal terminated without its slice receipt");
            };
            manager
                .own_event(0, CpuWorkerEvent::Slice(report.clone()))
                .unwrap();
            assert_eq!(frame_fingerprint(&manager, original.frame), fingerprint);
            assert_eq!(manager.driver.partial_commits.len(), 1);
            assert!(manager.driver.terminal_commits.is_empty());
            assert!(manager.driver.inner.closed.is_empty());
            if report.terminal {
                break report;
            }
        };
        assert_eq!(
            terminal.result.as_ref().unwrap().report.completion,
            CpuCompletion::DepthLimit
        );
        assert!(manager.own[0].binding.is_none());
        assert!(manager.driver.detached.iter().all(|(work, report)| {
            work.frame == original.frame
                && work.input.question_digest == original.input.question_digest
                && work.input.view_seal == original.input.view_seal
                && report.grant.job == original_job.id
                && report.grant.owner == first.grant.owner
        }));
        assert_eq!(
            manager.native_cpu_usage().unwrap(),
            (terminal.cumulative_nodes, 0)
        );
        assert_eq!(
            manager.counters.cpu_slice_receipts,
            manager.counters.cpu_yields + 1
        );

        // A duplicate terminal is rejected before another semantic consumption
        // or diagnostic record. Its original ledger error remains unchanged.
        let observed_before = diagnostic.snapshot().unwrap().records.len();
        assert!(matches!(
            manager.own_event(0, CpuWorkerEvent::Slice(terminal)),
            Err(AdaptiveManagerError::Cpu(CpuWorkerError::Duplicate))
        ));
        let observed_after = diagnostic.snapshot().unwrap();
        assert_eq!(observed_after.records.len(), observed_before);
        assert!(observed_after.unavailable.is_some());
        assert_eq!(frame_fingerprint(&manager, original.frame), fingerprint);
        assert_eq!(manager.driver.partial_commits.len(), 1);
        assert!(manager.driver.terminal_commits.is_empty());
        let receipt = shutdown_partial(&mut manager, until);
        assert_eq!(receipt.native_cpu_reserved, 0);
        assert!(!receipt.permits_archive_close());
    }

    #[test]
    fn forwarded_paused_token_discard_after_cancel_never_retires_new_question() {
        let (mut manager, facts, root, until) =
            partial_fixture(CoordinatorPolicy::Adaptive, RootGeneration(1));
        start_cpu(&mut manager, &facts, root, until);
        let first = forward_first(&mut manager, until);
        let original = manager.own[0].binding.as_ref().unwrap().work.clone();
        manager.begin_shutdown(until).unwrap();
        let revoked = frame_fingerprint(&manager, original.frame);
        let closed = manager.driver.inner.closed.len();
        // Check the old token discard before the separate, intentional root
        // retirement at the all-owner join fence clears every frame.
        let (event, timing) = next_slice(&mut manager, until).into_parts();
        assert!(matches!(&event, CpuWorkerEvent::JobTerminated(_)));
        assert!(timing.is_none());
        manager.own_event(0, event).unwrap();
        manager.flush_deferred().unwrap();
        assert_eq!(frame_fingerprint(&manager, original.frame), revoked);
        assert_eq!(manager.driver.inner.closed.len(), closed);
        assert_eq!(manager.driver.discarded.len(), 1);
        let receipt = shutdown_partial(&mut manager, until);
        assert!(frame_fingerprint(&manager, original.frame).is_none());
        assert_eq!(manager.driver.inner.closed.len(), closed);
        assert_eq!(manager.driver.partial_commits.len(), 1);
        assert!(manager.driver.terminal_commits.is_empty());
        assert_eq!(manager.driver.discarded.len(), 1);
        let (work, termination) = &manager.driver.discarded[0];
        assert_eq!(work.input.question_digest, original.input.question_digest);
        assert_eq!(work.input.view_seal, original.input.view_seal);
        assert_eq!(termination.job, first.grant.job);
        assert_eq!(termination.owner, first.grant.owner);
        assert_eq!(termination.last_execution, first.grant.execution);
        assert_eq!(receipt.native_cpu_reserved, 0);
        assert!(receipt.permits_archive_close());
        assert!(manager.failures().next().is_none());
    }

    #[test]
    fn old_root_yield_receipt_cannot_change_a_new_root_live_frame() {
        let (mut old, facts, root, until) =
            partial_fixture(CoordinatorPolicy::Adaptive, RootGeneration(1));
        start_cpu(&mut old, &facts, root, until);
        let old_report = forward_first(&mut old, until);
        assert!(shutdown_partial(&mut old, until).permits_archive_close());

        let (mut next, facts, root, until) =
            partial_fixture(CoordinatorPolicy::Adaptive, RootGeneration(2));
        start_cpu(&mut next, &facts, root, until);
        let original = next.own[0].binding.as_ref().unwrap().work.clone();
        let fingerprint = frame_fingerprint(&next, original.frame);
        assert_ne!(old_report.grant.job.authority, original.frame.authority);
        assert!(
            next.own_event(0, CpuWorkerEvent::Slice(old_report))
                .is_err()
        );
        assert_eq!(frame_fingerprint(&next, original.frame), fingerprint);
        assert_eq!(
            next.own[0]
                .binding
                .as_ref()
                .unwrap()
                .work
                .input
                .question_digest,
            original.input.question_digest
        );
        assert!(next.driver.partial_commits.is_empty());
        assert!(next.driver.terminal_commits.is_empty());
        assert!(next.driver.detached.is_empty());
        let receipt = shutdown_partial(&mut next, until);
        assert_eq!(receipt.native_cpu_reserved, 0);
        assert!(!receipt.permits_archive_close());
    }

    #[test]
    fn fixed_and_synchronous_keep_partial_receipts_on_the_original_cpu_step() {
        for policy in [CoordinatorPolicy::Fixed, CoordinatorPolicy::Synchronous] {
            let (mut manager, facts, root, until) = partial_fixture(policy, RootGeneration(1));
            start_cpu(&mut manager, &facts, root, until);
            let original = manager.own[0].binding.as_ref().unwrap().work.clone();
            let terminal = loop {
                let (event, _) = next_slice(&mut manager, until).into_parts();
                let CpuWorkerEvent::Slice(report) = event else {
                    panic!("control owner terminated without its slice receipt");
                };
                manager
                    .own_event(0, CpuWorkerEvent::Slice(report.clone()))
                    .unwrap();
                assert!(manager.driver.partial_commits.is_empty());
                assert!(manager.driver.detached.is_empty());
                if report.terminal {
                    break report;
                }
                let binding = manager.own[0].binding.as_ref().unwrap();
                assert!(!binding.feedback_forwarded);
                assert_eq!(
                    binding.work.input.question_digest,
                    original.input.question_digest
                );
                assert_eq!(
                    manager.machine.frame(original.frame).unwrap().input.stage,
                    RefinementStage::CpuValidation
                );
                manager.resume_cpu(0, 1, Duration::from_millis(8)).unwrap();
            };
            assert_eq!(manager.driver.terminal_commits.len(), 1);
            assert_eq!(manager.driver.inner.closed.len(), 1);
            assert_eq!(
                manager.native_cpu_usage().unwrap(),
                (terminal.cumulative_nodes, 0)
            );
            assert_eq!(
                manager.counters.cpu_slice_receipts,
                manager.counters.cpu_yields + 1
            );
            assert!(shutdown_partial(&mut manager, until).permits_archive_close());
            assert!(manager.failures().next().is_none());
        }
    }

    #[test]
    fn scheduling_yield_ignores_closed_owners_and_preserves_live_control_errors() {
        use rz_runtime::CompletionSignal;
        use rz_runtime::adaptive::ControlMailbox;

        let terminal = ControlMailbox::new((), CompletionSignal::default());
        let terminal_before = terminal.seal_terminal();
        assert!(!scheduling_yield(&terminal, 1).unwrap());
        assert_eq!(terminal.snapshot(), terminal_before);
        assert_eq!(terminal.snapshot().yield_sequence, 0);
        assert_eq!(terminal.snapshot().yield_ack, 0);

        let canceled = ControlMailbox::new((), CompletionSignal::default());
        canceled.request_cancel(1).unwrap();
        let canceled_before = canceled.snapshot();
        assert!(!scheduling_yield(&canceled, 2).unwrap());
        assert_eq!(canceled.snapshot(), canceled_before);
        assert_eq!(canceled.snapshot().yield_sequence, 0);
        assert_eq!(canceled.snapshot().yield_ack, 0);

        let live = ControlMailbox::new((), CompletionSignal::default());
        assert!(scheduling_yield(&live, 1).unwrap());
        assert_eq!(live.snapshot().yield_sequence, 1);
        assert_eq!(live.snapshot().yield_ack, 0);
        let requested = live.snapshot();
        assert!(matches!(
            scheduling_yield(&live, 2),
            Err(AdaptiveManagerError::Invalid("adaptive Yield control"))
        ));
        assert!(matches!(
            scheduling_yield(&live, 1),
            Err(AdaptiveManagerError::Invalid("adaptive Yield control"))
        ));
        assert_eq!(live.snapshot(), requested);
    }
}

// Diagnostic regressions reuse the actual manager/native service/CPU fixture.
// Authored neural outputs remain fixtures, never actual model/Y1/T1/R1 admission.
#[cfg(feature = "pals-adaptive-diagnostics")]
#[test]
fn diagnostic_machine_acceptance_survives_original_closed_publication_failure() {
    use crate::pals_adaptive::{
        ContinuationDiagnostic, ContinuationDiagnosticOptions, ContinuationResolutionSource,
    };
    for observe in [false, true] {
        let (mut manager, facts, root, until) =
            fixture(FixtureScope::TwoIndependentRoles, true, 4096);
        let diagnostic = observe.then(|| {
            ContinuationDiagnostic::new(ContinuationDiagnosticOptions {
                pending_snapshots_max: 0,
                resolution_records_max: 64,
                role_consumption: false,
            })
            .unwrap()
        });
        if let Some(diagnostic) = &diagnostic {
            manager.install_continuation_diagnostic(diagnostic.clone());
        }
        manager.driver_mut().fail_publish_closed_once = true;
        facts.child_gate.store(true, Ordering::Release);
        manager.prepare_root_ranking(root).unwrap();
        let receipt = manager.run_until(until, &AtomicBool::new(false)).unwrap();
        // run_until returns the actual drained receipt even when it retained a
        // semantic failure. The original error must not become a normal result.
        assert!(manager.failures().any(|error| matches!(error,
            AdaptiveManagerError::Semantic(message)
                if message == "authored closed-frame publication failure")));
        assert!(receipt.permits_archive_close());
        assert_eq!(receipt.pending_tickets, 0);
        assert_eq!(receipt.native_cpu_reserved, 0);
        assert!(manager.roles.is_empty());
        assert_eq!(facts.factories.load(Ordering::Acquire), 1);
        assert!(facts.finish_called.load(Ordering::Acquire));
        if let Some(diagnostic) = diagnostic {
            let observed = diagnostic.snapshot().unwrap();
            assert_eq!(observed.unavailable_bits, 0);
            let pair = observed
                .resolutions
                .windows(2)
                .find(|pair| {
                    matches!(
                        pair[0].source,
                        ContinuationResolutionSource::MachineTransition {
                            checked_rules_pair: false,
                            accepted: true
                        }
                    ) && matches!(
                        pair[1].source,
                        ContinuationResolutionSource::ClosedFramePublication { published: false }
                    ) && pair[0].evidence == pair[1].evidence
                })
                .expect(
                    "actual machine acceptance and later failed publication must remain distinct",
                );
            assert_eq!(pair[0].authority, pair[1].authority);
            assert_eq!(pair[0].closed_frame, Some(pair[0].evidence.frame));
            assert_eq!(pair[1].closed_frame, pair[0].closed_frame);
            assert!(pair[0].closed_unknown.is_some());
            assert!(pair[1].closed_unknown.is_some());
            assert!(
                manager
                    .driver()
                    .closed
                    .iter()
                    .all(|frame| Some(frame.id) != pair[1].closed_frame)
            );
        }
    }
}

#[cfg(feature = "pals-adaptive-diagnostics")]
fn assert_diagnostic_question_identity(
    before: &AdaptivePendingQuestionRow,
    after: &AdaptivePendingQuestionRow,
) {
    assert_eq!(before.preparation, after.preparation);
    assert_eq!(before.frame, after.frame);
    assert_eq!(before.stage, after.stage);
    assert_eq!(before.question, after.question);
    assert_eq!(before.ticket, after.ticket);
    assert_eq!(before.view, after.view);
    assert_eq!(before.retained_view_digest, after.retained_view_digest);
    assert_eq!(
        before.retained_view_seal_digest,
        after.retained_view_seal_digest
    );
    assert_eq!(
        before.selected_record_revisions,
        after.selected_record_revisions
    );
}

#[cfg(feature = "pals-adaptive-diagnostics")]
#[test]
fn diagnostic_actual_native_queue_admissions_keep_other_pending_question_identity() {
    use crate::pals_adaptive::{
        ContinuationDiagnostic, ContinuationDiagnosticOptions, ContinuationPendingSource,
    };
    use rz_search::pals::coordinator::TicketPhase;
    let (mut manager, facts, root, until) = fixture(FixtureScope::TwoIndependentRoles, true, 4096);
    let diagnostic = ContinuationDiagnostic::new(ContinuationDiagnosticOptions {
        pending_snapshots_max: 16,
        resolution_records_max: 0,
        role_consumption: false,
    })
    .unwrap();
    manager.install_continuation_diagnostic(diagnostic.clone());
    manager.prepare_root_ranking(root).unwrap();
    // Reuse the existing finite fixture gate, never add a production barrier.
    // Two bound tickets alone are insufficient: require both actual queue clocks.
    pump(&mut manager, until, |manager| {
        manager.roles.len() == 2
            && manager.roles.iter().all(|binding| {
                binding.work.is_some()
                    && binding.ticket.is_some()
                    && binding.queue_admitted_at.is_some()
                    && !binding.service_completed
                    && !binding.consumption_recorded
            })
    });
    let before = diagnostic.snapshot().unwrap();
    assert_eq!(before.unavailable_bits, 0);
    let queued: Vec<_> = before
        .pending
        .iter()
        .filter(|event| {
            event.source == ContinuationPendingSource::QueueAdmitted
                && event.snapshot.as_ref().is_ok_and(|snapshot| {
                    snapshot.questions.iter().any(|row| {
                        row.frame.is_some()
                            && row.ticket.is_some_and(|ticket| ticket.id == event.trigger)
                    })
                })
        })
        .collect();
    assert_eq!(queued.len(), 2);
    assert_ne!(queued[0].trigger, queued[1].trigger);
    assert!(queued[0].queue_admitted_at.is_some());
    assert!(queued[1].queue_admitted_at.is_some());
    assert!(queued[0].queue_admitted_at <= queued[1].queue_admitted_at);
    let both = queued[1].snapshot.as_ref().unwrap();
    assert_eq!(both.questions.len(), 2);
    for row in &both.questions {
        assert!(row.frame.is_some());
        assert!(
            row.ticket
                .is_some_and(|ticket| queued.iter().any(|event| event.trigger == ticket.id))
        );
        assert!(matches!(
            row.registry_observed_phase,
            Some(TicketPhase::Submitted | TicketPhase::PhysicallyRunning)
        ));
        assert!(!row.completion_received && !row.consumption_recorded);
        let binding = manager
            .roles
            .iter()
            .find(|binding| binding.ticket == row.ticket)
            .unwrap();
        assert!(binding.queue_admitted_at.is_some());
        assert_eq!(row.question, binding.input.question_digest);
        assert_eq!(row.view, binding.input.view_seal);
    }
    let first = queued[0]
        .snapshot
        .as_ref()
        .unwrap()
        .questions
        .iter()
        .find(|row| {
            row.ticket
                .is_some_and(|ticket| ticket.id == queued[0].trigger)
        })
        .unwrap();
    let same_first = both
        .questions
        .iter()
        .find(|row| row.ticket == first.ticket)
        .unwrap();
    assert_diagnostic_question_identity(first, same_first);
    assert_eq!(manager.counters().consumed_role_outputs, 1);
    // Allow the same original service to complete its two authored outputs.
    facts.child_gate.store(true, Ordering::Release);
    pump(&mut manager, until, |manager| {
        manager.driver().closed.len() == 2 && !manager.has_pending_work()
    });
    let after = diagnostic.snapshot().unwrap();
    assert_eq!(after.unavailable_bits, 0);
    let (event, pending) = after
        .pending
        .iter()
        .filter(|event| event.source == ContinuationPendingSource::CompletionProcessed)
        .find_map(|event| {
            let snapshot = event.snapshot.as_ref().ok()?;
            let completed = snapshot.questions.iter().find(|row| {
                row.frame.is_some()
                    && row.ticket.is_some_and(|ticket| ticket.id == event.trigger)
                    && row.completion_received
            })?;
            let pending = snapshot.questions.iter().find(|row| {
                row.frame.is_some()
                    && row.ticket.is_some_and(|ticket| ticket.id != event.trigger)
                    && !row.completion_received
                    && !row.consumption_recorded
            })?;
            assert!(!completed.consumption_recorded);
            Some((event, pending))
        })
        .expect("first original completion source must preserve the other queued question");
    let snapshot = event.snapshot.as_ref().unwrap();
    assert_eq!(snapshot.authority, both.authority);
    assert_eq!(snapshot.original_deadline, both.original_deadline);
    let original = both
        .questions
        .iter()
        .find(|row| row.ticket == pending.ticket)
        .unwrap();
    assert_diagnostic_question_identity(original, pending);
    assert_eq!(manager.counters().consumed_role_outputs, 3);
    assert_eq!(facts.accepted.lock().unwrap().len(), 3);
    assert_eq!(facts.factories.load(Ordering::Acquire), 1);
    assert!(manager.failures().next().is_none());
    let receipt = shutdown(&mut manager, until);
    assert!(receipt.permits_archive_close());
    assert_eq!(receipt.pending_tickets, 0);
    assert_eq!(receipt.native_cpu_reserved, 0);
}

#[test]
fn begin_shutdown_preserves_existing_paused_owner_cancel_sequence() {
    let (mut manager, facts, root, until) = fixture_with_registered_role_limit_window(
        FixtureScope::OwnCpuResume,
        true,
        1,
        None,
        None,
        Duration::from_secs(30),
    );
    facts.child_gate.store(true, Ordering::Release);
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.own[0].binding.is_some()
    });
    let event = loop {
        if let Some(event) = manager.own[0].worker.as_ref().unwrap().try_event().unwrap() {
            break event;
        }
        assert!(
            Instant::now() < until,
            "original CPU receipt did not arrive"
        );
        thread::sleep(Duration::from_millis(1));
    };
    let CpuWorkerEvent::Slice(first) = event else {
        panic!("original CPU invocation must return its slice");
    };
    assert!(first.continuation_retained && !first.terminal);
    manager.own_event(0, CpuWorkerEvent::Slice(first)).unwrap();
    let controls = Arc::clone(&manager.own[0].binding.as_ref().unwrap().controls);
    let original_cancel = manager.issue_control().unwrap();
    controls.request_cancel(original_cancel).unwrap();
    manager.begin_shutdown(until).unwrap();
    assert_eq!(controls.snapshot().cancel_sequence, original_cancel);
    let receipt = shutdown(&mut manager, until);
    let terminal = controls.snapshot();
    assert!(terminal.terminal);
    assert_eq!(terminal.cancel_sequence, original_cancel);
    assert_eq!(terminal.cancel_ack, original_cancel);
    assert_eq!(receipt.native_cpu_reserved, 0);
    assert_eq!(receipt.own_cpu_joined, receipt.expected_own_cpu_owners);
    assert!(receipt.permits_archive_close());
    assert!(manager.failures().next().is_none());
}

#[test]
fn close_does_not_replace_cancel_observed_by_owning_worker() {
    let controls = Arc::new(rz_runtime::adaptive::ControlMailbox::new(
        "job",
        rz_runtime::CompletionSignal::default(),
    ));
    assert!(close_cancel_needed(&controls));
    controls.request_cancel(1).unwrap();
    let (observed_tx, observed_rx) = std::sync::mpsc::sync_channel(1);
    let (resume_tx, resume_rx) = std::sync::mpsc::sync_channel(1);
    let owner = Arc::clone(&controls);
    let handle = thread::spawn(move || {
        let sequence = owner.snapshot().cancel_sequence;
        observed_tx.send(sequence).ok()?;
        resume_rx.recv_timeout(Duration::from_secs(30)).ok()?;
        owner.acknowledge_cancel(sequence).ok()?;
        Some(owner.seal_terminal())
    });
    assert_eq!(
        observed_rx.recv_timeout(Duration::from_secs(30)).unwrap(),
        1
    );
    // Same decision used by begin_close, while the owner still holds seq 1.
    let duplicate = close_cancel_needed(&controls).then(|| controls.request_cancel(2));
    let retained_sequence = controls.snapshot().cancel_sequence;
    resume_tx.send(()).unwrap();
    let owner_result = handle.join();
    assert!(duplicate.is_none());
    assert_eq!(retained_sequence, 1);
    let terminal = owner_result
        .unwrap()
        .expect("original cancel must be acknowledged");
    assert!(terminal.terminal);
    assert_eq!((terminal.cancel_sequence, terminal.cancel_ack), (1, 1));
    assert!(!close_cancel_needed(&controls));
    let fresh = rz_runtime::adaptive::ControlMailbox::new(
        "new-job",
        rz_runtime::CompletionSignal::default(),
    );
    assert!(close_cancel_needed(&fresh));
}

#[test]
fn canceled_paused_owner_retains_queued_resume_until_its_actual_no_work_settlement() {
    let (mut manager, facts, root, until) = fixture(FixtureScope::OwnCpuResume, true, 1);
    facts.child_gate.store(true, Ordering::Release);
    manager.prepare_root_ranking(root).unwrap();
    pump(&mut manager, until, |manager| {
        manager.own[0].binding.is_some()
    });
    // Read one original reliable receipt without allowing the manager to
    // immediately schedule its next slice in the same tick.
    let take = |manager: &Manager| loop {
        if let Some(event) = manager.own[0].worker.as_ref().unwrap().try_event().unwrap() {
            break event;
        }
        assert!(
            Instant::now() < until,
            "original CPU receipt did not arrive"
        );
        thread::sleep(Duration::from_millis(1));
    };
    let CpuWorkerEvent::Slice(first) = take(&manager) else {
        panic!("first original CPU invocation must return its slice");
    };
    assert!(first.continuation_retained && !first.terminal);
    let spent = first.cumulative_nodes;
    manager.own_event(0, CpuWorkerEvent::Slice(first)).unwrap();
    assert!(manager.own[0].binding.as_ref().unwrap().resume_pending);
    let sequence = manager.issue_control().unwrap();
    manager.own[0]
        .binding
        .as_ref()
        .unwrap()
        .controls
        .request_cancel(sequence)
        .unwrap();
    let CpuWorkerEvent::JobTerminated(termination) = take(&manager) else {
        panic!("same paused owner must return its actual token discard");
    };
    // Keep that real event unread by the manager while admitting the newer
    // same-owner command. This deterministically exercises the two-message
    // order also possible when cancel wins just before the owner dequeues it.
    manager.resume_cpu(0, 1, Duration::from_millis(8)).unwrap();
    let current = manager.own[0].binding.as_ref().unwrap().grant;
    assert!(current.execution.sequence > termination.last_execution.sequence);
    assert_eq!(
        manager.native_cpu_usage().unwrap(),
        (spent, current.node_credit)
    );
    let CpuWorkerEvent::Slice(rejected) = take(&manager) else {
        panic!("queued Resume must return its separate original receipt");
    };
    assert_eq!(rejected.grant, current);
    assert!(rejected.admission_rejected && !rejected.terminal && !rejected.continuation_retained);
    assert_eq!(
        rejected.actual_work,
        Some(rz_search::cpu::CpuWork::default())
    );
    manager.begin_shutdown(until).unwrap();
    manager
        .own_event(0, CpuWorkerEvent::JobTerminated(termination))
        .unwrap();
    assert!(
        manager.own[0]
            .binding
            .as_ref()
            .unwrap()
            .queued_resume_termination
            .is_some()
    );
    assert_eq!(
        manager.native_cpu_usage().unwrap(),
        (spent, current.node_credit)
    );
    assert!(!manager.shutdown_receipt().unwrap().permits_archive_close());
    manager.driver_mut().semantic_owner_busy = true;
    manager
        .own_event(0, CpuWorkerEvent::Slice(rejected))
        .unwrap();
    // The same original pair survives normal archive/semantic-owner waiting.
    // Neither a generic raw flush nor a new manager may consume only one side.
    manager.flush_deferred().unwrap();
    let held = manager.own[0].binding.as_ref().unwrap();
    assert!(held.queued_resume_termination.is_some() && held.pending_slice.is_some());
    assert_eq!(manager.native_cpu_usage().unwrap(), (spent, 0));
    assert!(!manager.shutdown_receipt().unwrap().permits_archive_close());
    let proof = held.semantic_settlement.as_ref().unwrap();
    assert_eq!(
        proof.terminated_owner_receipt(),
        held.queued_resume_termination.as_ref()
    );
    let mut stale = held.queued_resume_termination.as_ref().unwrap().clone();
    stale.last_execution.sequence = stale.last_execution.sequence.saturating_sub(1);
    assert!(
        manager
            .machine
            .retire_rejected_cpu_resume(
                held.work.frame,
                &stale,
                proof,
                Digest([88; 32]),
                "stale terminal"
            )
            .is_err()
    );
    assert!(manager.machine.frame(held.work.frame).is_some());
    manager.driver_mut().semantic_owner_busy = false;
    manager.flush_deferred().unwrap();
    assert!(manager.own[0].binding.is_none());
    assert_eq!(manager.native_cpu_usage().unwrap(), (spent, 0));
    let receipt = shutdown(&mut manager, until);
    assert!(receipt.permits_archive_close());
    assert_eq!(receipt.native_cpu_reserved, 0);
    assert!(manager.failures().next().is_none());
    assert!(
        manager.driver().actual_cpu.is_empty(),
        "cancel cannot publish a late score"
    );
    assert!(
        manager
            .driver()
            .raw_cpu_receipts
            .contains(&current.execution)
    );
}
