//! Projection of a Search-owned immutable question on the native owner.
//!
//! The projection is prepared once and retained beside the actual backend
//! preparation. Dispatch and acknowledgement reuse that original projection;
//! neither reads a newer Search question nor regenerates an input on the manager.

use crate::pals_native::ticketed::{
    NativeTicketQuestion, TicketExecutionEvidence, TicketNativeBackend, TicketPhysicalResult,
    TicketPreparation, TicketOwnerDispatchContext,
};
use rz_contracts::pals::RepresentationKey;
use rz_contracts::pals_adaptive::RoleTicketEnvelope;
use rz_runtime::Clock;
use rz_runtime::contracts::{ContractClock, ContractSystemClock};
use rz_search::pals::coordinator::{PhysicalRoleCompletion, RoleFailure};
use rz_search::pals::engine::adaptive_port::AdaptivePortQuestion;
use rz_search::pals::review_context::ReviewView;
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// This conversion copies only the bounded native question fields. It must not
/// encode an input, execute a model, or change the submitted Rules/history/view.
pub trait NativeQuestionProjection: Send + 'static {
    fn project_native_question(&self) -> Result<NativeTicketQuestion, RoleFailure>;
    /// Inline and owned backing of the original Q, including any private lines.
    /// Unknown/overflow is rejected rather than reported as zero host memory.
    fn owned_host_bytes(&self) -> Option<u64>;
}

impl NativeQuestionProjection for AdaptivePortQuestion {
    fn project_native_question(&self) -> Result<NativeTicketQuestion, RoleFailure> {
        Ok(NativeTicketQuestion {
            position: self.position.clone(),
            context: self.context.clone(),
            deadline: self.deadline,
            legacy_full_records: self.legacy_full_records.clone(),
        })
    }
    fn owned_host_bytes(&self) -> Option<u64> {
        AdaptivePortQuestion::owned_host_bytes(self)
    }
}

impl NativeQuestionProjection for NativeTicketQuestion {
    fn project_native_question(&self) -> Result<NativeTicketQuestion, RoleFailure> {
        Ok(self.clone())
    }
    fn owned_host_bytes(&self) -> Option<u64> {
        native_question_bytes(self)
    }
}

pub struct ProjectedNativeBackend<Q, B> {
    backend: B,
    last_executed: Option<RoleTicketEnvelope>,
    question: PhantomData<fn(Q) -> Q>,
}

/// Its question, view and prepared buffers stay owned by the ticket service
/// through physical completion and the eventual accept/discard acknowledgement.
pub struct ProjectedNativePrepared<P> {
    question: TicketPreparation<ReviewView, NativeTicketQuestion>,
    representation: RepresentationKey,
    prepared: P,
}

impl<Q, B> ProjectedNativeBackend<Q, B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            last_executed: None,
            question: PhantomData,
        }
    }
}

impl<Q, B> TicketNativeBackend<ReviewView, Q> for ProjectedNativeBackend<Q, B>
where
    Q: NativeQuestionProjection,
    B: TicketNativeBackend<ReviewView, NativeTicketQuestion>,
{
    type Prepared = ProjectedNativePrepared<B::Prepared>;
    type Output = B::Output;

    fn clock(&self) -> ContractSystemClock {
        self.backend.clock()
    }

    fn prepare(
        &mut self,
        source: &TicketPreparation<ReviewView, Q>,
        canceled: &AtomicBool,
    ) -> Result<(Self::Prepared, RepresentationKey), RoleFailure> {
        if canceled.load(Ordering::Acquire) {
            return Err(RoleFailure::Canceled);
        }
        source
            .query
            .owned_host_bytes()
            .ok_or(RoleFailure::InvalidOutput)?;
        let query = source.query.project_native_question()?;
        let clock = self.backend.clock();
        if source.deadline.clock != clock.domain()
            || source.deadline.clock.0 != source.seal.authority.epoch
            || clock
                .origin()
                .checked_add(Duration::from_nanos(source.deadline.at.0))
                != Some(query.deadline)
            || query.context.game_generation != source.seal.authority.game.0
            || query.context.search_generation != source.seal.authority.root.0
            || query.context.purpose != source.view.query().purpose
            || query.context.prefix != source.view.query().prefix
        {
            return Err(RoleFailure::InvalidOutput);
        }
        source
            .deadline
            .accepts(clock.domain(), clock.now())
            .map_err(|_| RoleFailure::Expired)?;
        let projected = TicketPreparation {
            question: source.question,
            seal: source.seal,
            role: source.role,
            deadline: source.deadline,
            view: Arc::clone(&source.view),
            query,
        };
        let (prepared, representation) = self.backend.prepare(&projected, canceled)?;
        Ok((
            ProjectedNativePrepared {
                question: projected,
                representation,
                prepared,
            },
            representation,
        ))
    }

    fn prepared_host_bytes(&self, prepared: &Self::Prepared) -> u64 {
        // The original service question counts the shared view once. Count the
        // retained projection separately, including its own position/prefix.
        let bytes = || -> Option<u64> {
            let view = prepared.question.view.owned_host_bytes()?;
            let native = native_question_bytes(&prepared.question.query)?;
            let declared = self.backend.question_host_bytes(&prepared.question);
            let native_backing = declared
                .max(native.checked_add(view)?)
                .checked_sub(view)?
                .checked_sub(size_of::<NativeTicketQuestion>() as u64)?;
            let wrapper = size_of::<Self::Prepared>().checked_sub(size_of::<B::Prepared>())? as u64;
            self.backend
                .prepared_host_bytes(&prepared.prepared)
                .checked_add(wrapper)?
                .checked_add(native_backing)
        };
        bytes().unwrap_or(u64::MAX)
    }

    fn question_host_bytes(&self, source: &TicketPreparation<ReviewView, Q>) -> u64 {
        let bytes = || -> Option<u64> {
            let outer =
                size_of::<TicketPreparation<ReviewView, Q>>().checked_sub(size_of::<Q>())? as u64;
            source
                .query
                .owned_host_bytes()?
                .checked_add(outer)?
                .checked_add(source.view.owned_host_bytes()?)
        };
        bytes().unwrap_or(u64::MAX)
    }

    fn run(
        &mut self,
        ticket: &RoleTicketEnvelope,
        source: &TicketPreparation<ReviewView, Q>,
        prepared: &Self::Prepared,
        canceled: &AtomicBool,
    ) -> TicketPhysicalResult<Self::Output> {
        self.last_executed = None;
        if !matches_original(ticket, source, prepared) {
            return TicketPhysicalResult {
                physical: PhysicalRoleCompletion::Confirmed,
                output: Err(RoleFailure::InvalidOutput),
            };
        }
        self.last_executed = Some(*ticket);
        self.backend
            .run(ticket, &prepared.question, &prepared.prepared, canceled)
    }

    fn run_owned(
        &mut self,
        context: TicketOwnerDispatchContext<'_, ReviewView, Q, Self::Prepared>,
        canceled: &AtomicBool,
    ) -> TicketPhysicalResult<Self::Output> {
        self.last_executed = None;
        let ticket = context.ticket();
        let source = context.question();
        let prepared = context.prepared();
        if !matches_original(ticket, source, prepared) {
            return TicketPhysicalResult {
                physical: PhysicalRoleCompletion::Confirmed,
                output: Err(RoleFailure::InvalidOutput),
            };
        }
        let projected = match context.project(&prepared.question, &prepared.prepared) {
            Ok(projected) => projected,
            Err(error) => return TicketPhysicalResult {
                physical: PhysicalRoleCompletion::Confirmed,
                output: Err(error),
            },
        };
        self.last_executed = Some(*ticket);
        self.backend.run_owned(projected, canceled)
    }

    fn output_host_bytes(&self, output: &Self::Output) -> u64 {
        self.backend.output_host_bytes(output)
    }

    fn acceptance_output_digest(&self, output: &Self::Output) -> Option<rz_contracts::Digest> {
        self.backend.acceptance_output_digest(output)
    }

    fn execution_evidence(
        &mut self,
        ticket: &RoleTicketEnvelope,
    ) -> Option<TicketExecutionEvidence> {
        if self.last_executed.take() == Some(*ticket) {
            self.backend.execution_evidence(ticket)
        } else {
            None
        }
    }

    fn accept(
        &mut self,
        ticket: &RoleTicketEnvelope,
        source: &TicketPreparation<ReviewView, Q>,
        prepared: &Self::Prepared,
        output: &Self::Output,
    ) -> Result<(), RoleFailure> {
        if !matches_original(ticket, source, prepared) {
            return Err(RoleFailure::InvalidOutput);
        }
        self.backend
            .accept(ticket, &prepared.question, &prepared.prepared, output)
    }

    fn discard(&mut self, ticket: &RoleTicketEnvelope) -> Result<(), RoleFailure> {
        self.backend.discard(ticket)
    }

    fn quiesce(&mut self, until: Instant) -> bool {
        self.backend.quiesce(until)
    }
    fn finish(&mut self, until: Instant) -> bool {
        self.backend.finish(until)
    }
}

fn matches_original<Q, P>(
    ticket: &RoleTicketEnvelope,
    source: &TicketPreparation<ReviewView, Q>,
    prepared: &ProjectedNativePrepared<P>,
) -> bool {
    let original = &prepared.question;
    source.question == original.question
        && source.seal == original.seal
        && source.role == original.role
        && source.deadline == original.deadline
        && Arc::ptr_eq(&source.view, &original.view)
        && ticket.question == original.question
        && ticket.view == original.seal
        && ticket.authority == original.seal.authority
        && ticket.role == original.role
        && ticket.original_deadline == original.deadline
        && ticket.representation == prepared.representation
}

fn native_question_bytes(query: &NativeTicketQuestion) -> Option<u64> {
    let bytes = size_of::<NativeTicketQuestion>()
        .checked_add(query.position.snapshot().retained_history_bytes()?)?
        .checked_add(
            query
                .context
                .prefix
                .capacity()
                .checked_mul(size_of::<rz_position::BoardMove>())?,
        )?;
    let bytes = u64::try_from(bytes).ok()?;
    match &query.legacy_full_records {
        Some(records) => bytes.checked_add(records.owned_host_bytes()?),
        None => Some(bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_contracts::pals::{Role, SearchAuthority};
    use rz_contracts::pals_adaptive::ReviewViewSeal;
    use rz_contracts::{
        Deadline, Digest, GameGeneration, MonotonicTick, OwnerId, PrecisionProfile, ProcessEpoch,
        RequestId, RootGeneration,
    };
    use rz_position::Position;
    use rz_position::contracts::ContractPosition;
    use rz_search::pals::engine::{RoleLogicalContext, RoleQueryPurpose};
    use rz_search::pals::review_context::{
        ReviewContextIndex, ReviewQuerySequences, ReviewSelectionPolicy,
    };
    use rz_search::pals::store::{PalsStores, StoreLimits};
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    struct CountingQuestion {
        native: NativeTicketQuestion,
        projections: Arc<AtomicUsize>,
    }
    impl NativeQuestionProjection for CountingQuestion {
        fn project_native_question(&self) -> Result<NativeTicketQuestion, RoleFailure> {
            self.projections.fetch_add(1, Ordering::Relaxed);
            Ok(self.native.clone())
        }
        fn owned_host_bytes(&self) -> Option<u64> {
            native_question_bytes(&self.native)?
                .checked_sub(size_of::<NativeTicketQuestion>() as u64)?
                .checked_add(size_of::<Self>() as u64)
        }
    }

    struct Fake {
        clock: ContractSystemClock,
        calls: Arc<Mutex<Vec<&'static str>>>,
        unknown: bool,
        finish_confirmed: bool,
        acceptance_digest: Option<Digest>,
    }
    impl TicketNativeBackend<ReviewView, NativeTicketQuestion> for Fake {
        type Prepared = u64;
        type Output = u64;
        fn clock(&self) -> ContractSystemClock {
            self.clock.clone()
        }
        fn prepare(
            &mut self,
            input: &TicketPreparation<ReviewView, NativeTicketQuestion>,
            _: &AtomicBool,
        ) -> Result<(u64, RepresentationKey), RoleFailure> {
            self.calls.lock().unwrap().push("prepare");
            assert_eq!(input.question.0, input.view.digest());
            Ok((7, key(input.role)))
        }
        fn prepared_host_bytes(&self, _: &u64) -> u64 {
            8
        }
        fn run(
            &mut self,
            _: &RoleTicketEnvelope,
            input: &TicketPreparation<ReviewView, NativeTicketQuestion>,
            prepared: &u64,
            _: &AtomicBool,
        ) -> TicketPhysicalResult<u64> {
            self.calls.lock().unwrap().push("run");
            assert_eq!(*prepared, 7);
            assert_eq!(input.question.0, input.view.digest());
            TicketPhysicalResult {
                physical: if self.unknown {
                    PhysicalRoleCompletion::Unknown
                } else {
                    PhysicalRoleCompletion::Confirmed
                },
                output: if self.unknown {
                    Err(RoleFailure::PhysicalUnknown)
                } else {
                    Ok(11)
                },
            }
        }
        fn output_host_bytes(&self, _: &u64) -> u64 {
            8
        }
        fn acceptance_output_digest(&self, output: &u64) -> Option<Digest> {
            self.calls.lock().unwrap().push("acceptance_digest");
            assert_eq!(*output, 11);
            self.acceptance_digest
        }
        fn execution_evidence(
            &mut self,
            _: &RoleTicketEnvelope,
        ) -> Option<TicketExecutionEvidence> {
            self.calls.lock().unwrap().push("evidence");
            Some(TicketExecutionEvidence {
                execution: None,
                completed_nn_inputs: Some(1),
                measurement_complete: true,
                physical_fresh: Some(true),
                accounting: None,
            })
        }
        fn accept(
            &mut self,
            _: &RoleTicketEnvelope,
            input: &TicketPreparation<ReviewView, NativeTicketQuestion>,
            _: &u64,
            _: &u64,
        ) -> Result<(), RoleFailure> {
            self.calls.lock().unwrap().push("accept");
            assert_eq!(input.question.0, input.view.digest());
            Ok(())
        }
        fn discard(&mut self, _: &RoleTicketEnvelope) -> Result<(), RoleFailure> {
            self.calls.lock().unwrap().push("discard");
            Ok(())
        }
        fn finish(&mut self, _: Instant) -> bool {
            self.calls.lock().unwrap().push("finish");
            self.finish_confirmed
        }
    }

    fn key(role: Role) -> RepresentationKey {
        RepresentationKey {
            input: Digest([1; 32]),
            history: Digest([2; 32]),
            candidates: Digest([3; 32]),
            mask_positions: Digest([4; 32]),
            model: Digest([5; 32]),
            encoding: Digest([6; 32]),
            precision: PrecisionProfile::Fp32,
            frozen_epoch: 1,
            record_revision: 1,
            role,
        }
    }
    fn fixture(clock: &ContractSystemClock) -> TicketPreparation<ReviewView, CountingQuestion> {
        let position = Position::startpos();
        let rules = ContractPosition::new(OwnerId(17), position.clone())
            .export()
            .unwrap();
        let mut stores = PalsStores::new(StoreLimits::default());
        let situation = stores.focus_actual_moves(position.snapshot()).unwrap();
        let state = stores.situations.get(situation).unwrap().state;
        let focus = stores.situations.get(situation).unwrap().focus;
        let legal = position.legal_moves();
        let view = Arc::new(
            ReviewContextIndex::default()
                .select(
                    &stores,
                    state,
                    ReviewQuerySequences {
                        purpose: RoleQueryPurpose::ProposePolicy,
                        prefix: &[],
                        proposal: &[],
                        counterexample: None,
                        legal: &legal,
                        divergences: &[],
                    },
                    ReviewSelectionPolicy::default(),
                )
                .unwrap(),
        );
        let authority = SearchAuthority {
            epoch: clock.domain().0,
            game: GameGeneration(2),
            root: RootGeneration(view.generation()),
            implementation: Digest([9; 32]),
        };
        let seal = ReviewViewSeal {
            authority,
            view_sequence: view.sequence(),
            context_policy: Digest(view.policy_digest()),
            view_digest: Digest(view.seal_digest()),
            root_state: rules.snapshot().identity(),
            actual_tokens: view.token_count(),
            selected_records: view.records().len() as u16,
            anchors: view.anchors().len() as u8,
            length_bucket: view.length_bucket(),
        };
        let deadline = Deadline {
            clock: clock.domain(),
            at: MonotonicTick(10_000_000_000),
        };
        TicketPreparation {
            question: Digest(view.digest()),
            seal,
            role: Role::Proposer,
            deadline,
            view,
            query: CountingQuestion {
                native: NativeTicketQuestion {
                    position,
                    context: RoleLogicalContext {
                        game_generation: authority.game.0,
                        search_generation: authority.root.0,
                        situation,
                        state,
                        focus,
                        purpose: RoleQueryPurpose::ProposePolicy,
                        prefix: Vec::new(),
                        focus_sha256: [0; 32],
                        prefix_sha256: [0; 32],
                        proposal_sha256: [0; 32],
                        refutation_sha256: None,
                        divergence_sha256: [0; 32],
                        public_revision: 1,
                        situation_revision: 1,
                    },
                    deadline: clock.origin() + Duration::from_nanos(deadline.at.0),
                    legacy_full_records: None,
                },
                projections: Arc::new(AtomicUsize::new(0)),
            },
        }
    }
    fn ticket(source: &TicketPreparation<ReviewView, CountingQuestion>) -> RoleTicketEnvelope {
        RoleTicketEnvelope {
            id: RequestId::new(source.seal.authority.epoch, 1),
            authority: source.seal.authority,
            question: source.question,
            view: source.seal,
            representation: key(source.role),
            role: source.role,
            original_deadline: source.deadline,
        }
    }
    fn backend(
        clock: &ContractSystemClock,
        unknown: bool,
    ) -> (
        ProjectedNativeBackend<CountingQuestion, Fake>,
        Arc<Mutex<Vec<&'static str>>>,
    ) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        (
            ProjectedNativeBackend::new(Fake {
                clock: clock.clone(),
                calls: Arc::clone(&calls),
                unknown,
                finish_confirmed: !unknown,
                acceptance_digest: None,
            }),
            calls,
        )
    }

    #[test]
    fn acceptance_digest_preserves_underlying_some_and_none_with_original_output() {
        let clock = ContractSystemClock::new(ProcessEpoch(23));
        for expected in [Some(Digest([29; 32])), None] {
            let (mut backend, calls) = backend(&clock, false);
            backend.backend.acceptance_digest = expected;
            let output = 11;

            assert_eq!(backend.acceptance_output_digest(&output), expected);
            assert_eq!(output, 11);
            assert_eq!(*calls.lock().unwrap(), ["acceptance_digest"]);
        }
    }

    #[test]
    fn original_projection_is_used_once_through_run_and_accept() {
        let clock = ContractSystemClock::new(ProcessEpoch(23));
        let source = fixture(&clock);
        let (mut backend, calls) = backend(&clock, false);
        let canceled = AtomicBool::new(false);
        let (prepared, _) = backend.prepare(&source, &canceled).unwrap();
        assert!(backend.prepared_host_bytes(&prepared) > 8);
        assert!(backend.question_host_bytes(&source) >= source.view.owned_host_bytes().unwrap());
        let ticket = ticket(&source);
        let result = backend.run(&ticket, &source, &prepared, &canceled);
        assert_eq!(result.physical, PhysicalRoleCompletion::Confirmed);
        let output = result.output.unwrap();
        assert_eq!(
            backend
                .execution_evidence(&ticket)
                .unwrap()
                .completed_nn_inputs,
            Some(1)
        );
        assert!(backend.execution_evidence(&ticket).is_none());
        backend
            .accept(&ticket, &source, &prepared, &output)
            .unwrap();
        assert!(backend.finish(source.query.native.deadline));
        assert_eq!(source.query.projections.load(Ordering::Relaxed), 1);
        assert_eq!(
            *calls.lock().unwrap(),
            ["prepare", "run", "evidence", "accept", "finish"]
        );
    }

    #[test]
    fn deadline_and_changed_envelope_are_rejected_before_native_execution() {
        let clock = ContractSystemClock::new(ProcessEpoch(23));
        let mut source = fixture(&clock);
        let (mut backend, calls) = backend(&clock, false);
        let canceled = AtomicBool::new(false);
        source.query.native.deadline += Duration::from_nanos(1);
        assert!(matches!(
            backend.prepare(&source, &canceled),
            Err(RoleFailure::InvalidOutput)
        ));
        assert!(calls.lock().unwrap().is_empty());
        source.query.native.deadline -= Duration::from_nanos(1);
        let (prepared, _) = backend.prepare(&source, &canceled).unwrap();
        let ticket = ticket(&source);
        source.seal.view_sequence += 1;
        let result = backend.run(&ticket, &source, &prepared, &canceled);
        assert_eq!(result.output, Err(RoleFailure::InvalidOutput));
        assert!(backend.execution_evidence(&ticket).is_none());
        assert_eq!(
            backend.accept(&ticket, &source, &prepared, &11),
            Err(RoleFailure::InvalidOutput)
        );
        assert_eq!(*calls.lock().unwrap(), ["prepare"]);
    }

    #[test]
    fn physical_unknown_and_failed_finish_remain_unknown() {
        let clock = ContractSystemClock::new(ProcessEpoch(23));
        let source = fixture(&clock);
        let (mut backend, calls) = backend(&clock, true);
        let canceled = AtomicBool::new(false);
        let (prepared, _) = backend.prepare(&source, &canceled).unwrap();
        let result = backend.run(&ticket(&source), &source, &prepared, &canceled);
        assert_eq!(result.physical, PhysicalRoleCompletion::Unknown);
        assert_eq!(result.output, Err(RoleFailure::PhysicalUnknown));
        assert!(!backend.finish(source.query.native.deadline));
        assert_eq!(*calls.lock().unwrap(), ["prepare", "run", "finish"]);
        assert_eq!(source.query.projections.load(Ordering::Relaxed), 1);
    }
}
