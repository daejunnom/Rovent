//! Explicitly profiled external R. Single limits and foreign reports never
//! acquire an own CPU Job, native node credit, completed depth or Rules proof.
use super::*;
use crate::external_cpu::worker::external_root_restriction_digest;
use crate::external_cpu::{
    ExternalCpuConfig, ExternalEnginePurpose, ExternalGoConstraint, ExternalGoIssue,
    ProfiledExternalCompletion, ProfiledExternalCurrentRootFence, ProfiledExternalDispatch,
    ProfiledExternalJob, ProfiledExternalProcessCloseProof, ProfiledExternalRootSource,
    ProfiledExternalScope,
};
use crate::pals::coordinator::CheckedProfiledExternalBinding;
use crate::pals::rovent_frontier::{FrontierExternalReservation, FrontierExternalWork};
use crate::pals::store::{
    ProfiledExternalConstraint, ProfiledForeignAppendPermit, ProfiledForeignGoIssuedProof,
    StorePins, StoreReadOwner,
};
use rz_contracts::rovent::{
    ChessPly, QuestionEnvelope, QuestionPurpose, ReciprocalQuestionId, SearchRecipient,
};
use rz_contracts::{ClockDomain, MonotonicTick};
use rz_runtime::{CompletionSignal, adaptive::ControlMailbox};
pub(super) mod feedback_observation;
pub use feedback_observation::{
    ProfiledFeedbackConsumedChain, ProfiledFeedbackConsumptionObservation,
    ProfiledFeedbackObservationUnavailable,
};

const PROFILED_QUESTION_CAPACITY: usize = 64;
fn reserve_profiled_question_slot<T>(
    entries: &mut HashMap<ContractDigest, T>,
) -> Result<(), PalsError> {
    if entries.len() >= PROFILED_QUESTION_CAPACITY {
        return Err(PalsError::Capacity);
    }
    entries.try_reserve(1).map_err(|_| PalsError::Capacity)
}

fn profiled_question_hex(digest: ContractDigest) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in digest.0 {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 15) as usize] as char);
    }
    output
}
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfiledExternalUnsupported {
    Searchmoves,
    CombinedConstraint,
    ReciprocalResume,
}

// Transfer the exact admitted Job without a new allocation in the admission
// path. Any future indirection must retain the same failure/owner contract.
#[allow(clippy::large_enum_variant)]
pub enum AdaptiveProfiledExternalPreparation {
    Ready(ProfiledExternalJob),
    Unsupported(ProfiledExternalUnsupported),
}

pub struct AdaptiveProfiledExternalAdmission {
    pub binding: CheckedProfiledExternalBinding,
    pub reservation: FrontierExternalReservation,
}

pub struct AdaptiveProfiledExternalCommit {
    pub observation: ObservationId,
    pub consumed: bool,
}

/// Only this actual Store consumer can mint a peer R answer. A public raw
/// report, completion flag or caller-supplied digest cannot construct it.
pub struct CheckedProfiledExternalResponse {
    frame: FrameId,
    stage: RefinementStage,
    first: rz_contracts::pals::Move16,
    deadline: Deadline,
    question: QuestionEnvelope,
    execution: rz_contracts::ExecutionId,
    input_digest: ContractDigest,
    public_context: ContractDigest,
    evidence: ContractDigest,
}
impl CheckedProfiledExternalResponse {
    pub fn evidence(&self) -> ContractDigest {
        self.evidence
    }
    pub(crate) fn validates<Q>(
        &self,
        evidence: &crate::pals::coordinator::StepEvidence,
        frame: FrameId,
        first: rz_contracts::pals::Move16,
        deadline: Deadline,
        input: &FrameInput<ReviewView, Q>,
    ) -> bool {
        self.frame == frame
            && self.first == first
            && self.deadline == deadline
            && self.stage == input.stage
            && self.question.question_digest == input.question_digest
            && self.question.view == input.view_seal
            && self.input_digest == ContractDigest(input.view.digest())
            && self.public_context == input.public_context
            && self.evidence == evidence.completed_evidence
            && evidence.frame == frame
            && evidence.stage == input.stage
            && evidence.submitted_question == input.question_digest
            && evidence.submitted_view == input.view_seal
            && evidence.execution
                == crate::pals::coordinator::StepExecution::ProfiledExternal {
                    question: self.question.id,
                    execution: self.execution,
                }
    }
}

pub struct AdaptiveProfiledForeignPublication {
    pub observation: Observation,
    pub issued: ProfiledForeignGoIssuedProof,
}

#[derive(Default)]
pub(super) struct ProfiledExternalPortState {
    entries: HashMap<ContractDigest, ProfiledAdmission>,
    retirement_issued: bool,
    original_cleanup_deadline: Option<Instant>,
}
impl ProfiledExternalPortState {
    pub(super) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    fn protected_pins(&self, stores: &PalsStores) -> Result<StorePins, PalsError> {
        let mut pins = StorePins::default();
        for admission in self.entries.values() {
            if stores.tasks.get(admission.execution)?.key != admission.task_key
                || !stores
                    .states
                    .get(admission.task_key.state)?
                    .same_state(&admission.job.position.snapshot())
                || !admission.work.input.view.validates(stores)
            {
                return Err(StoreError::StaleConsumer.into());
            }
            let context = &admission.work.input.question.context;
            let query = &admission.work.input.question;
            for selected in [
                Some(&admission.work.input.view),
                query.frozen_counter.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                merge_profiled_retirement_pins(
                    &mut pins,
                    &selected.retirement_protected_pins(stores)?,
                );
            }
            if let Some(legacy) = &query.legacy_full_records {
                merge_profiled_retirement_pins(
                    &mut pins,
                    &legacy.retirement_protected_pins(stores)?,
                );
            }
            pins.states.insert(admission.task_key.state);
            pins.situations.insert(context.situation);
            pins.lines.insert(context.focus);
            pins.executions.insert(admission.execution);
            pins.observations.extend(admission.published);
            pins.observations.extend(query.counter_source);
            pins.observations.extend(query.paired_repaired);
            if let Some(observation) = &admission.prepared_observation {
                pins.lines.extend(observation.cpu_pv);
            }
        }
        Ok(pins)
    }

    fn owned_host_bytes(&self) -> Option<u64> {
        // Physical worker packet/config allocations include private nested
        // backing not exposed by those owners. Nonempty is explicitly unknown,
        // never zero or a serialized-size surrogate for actual host backing.
        self.entries
            .is_empty()
            .then_some(std::mem::size_of::<Self>() as u64)
    }
}
fn merge_profiled_retirement_pins(into: &mut StorePins, source: &StorePins) {
    into.states.extend(&source.states);
    into.lines.extend(&source.lines);
    into.observations.extend(&source.observations);
    into.situations.extend(&source.situations);
    into.executions.extend(&source.executions);
}

/// Move-only historical source recovery after actual same-root drain. It
/// carries no new admission/continuation authority and retains every original
/// packet, failed append, Work/view pin and known Store handle for its consumer.
pub struct ProfiledExternalRootRetirement {
    identity: ProfiledExternalRetirementIdentity,
    boundary: ProfiledRetirementBoundary,
    authority: SearchAuthority,
    source: ProfiledExternalRootSource,
    owner: StoreReadOwner,
    generation: u64,
    pins: Result<StorePins, PalsError>,
    state: ProfiledExternalPortState,
}
enum ProfiledRetirementBoundary {
    Root(ProfiledExternalCurrentRootFence),
    Process(ProfiledExternalProcessCloseProof),
}
#[derive(Clone, Debug)]
pub struct ProfiledExternalRetirementIdentity(Arc<()>);
impl ProfiledExternalRetirementIdentity {
    pub fn same_retirement(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl ProfiledExternalRootRetirement {
    pub(crate) fn root_fence(&self) -> Option<&ProfiledExternalCurrentRootFence> {
        match &self.boundary {
            ProfiledRetirementBoundary::Root(fence) => Some(fence),
            _ => None,
        }
    }
    pub(crate) fn process_close_proof(&self) -> Option<&ProfiledExternalProcessCloseProof> {
        match &self.boundary {
            ProfiledRetirementBoundary::Process(proof) => Some(proof),
            _ => None,
        }
    }
    pub fn identity(&self) -> ProfiledExternalRetirementIdentity {
        self.identity.clone()
    }
    pub fn matches_identity(&self, identity: &ProfiledExternalRetirementIdentity) -> bool {
        self.identity.same_retirement(identity)
    }
    pub fn authority(&self) -> SearchAuthority {
        self.authority
    }
    pub fn source(&self) -> &ProfiledExternalRootSource {
        &self.source
    }
    pub fn retained_questions(&self) -> usize {
        self.state.entries.len()
    }
    /// The caller's original root closure ceiling, frozen before the first
    /// semantic question. A missing legacy registration grants no I/O window.
    pub fn original_cleanup_deadline(&self) -> Option<Instant> {
        self.state.original_cleanup_deadline
    }
    pub fn owned_host_bytes(&self) -> Option<u64> {
        self.state
            .owned_host_bytes()?
            .checked_add(std::mem::size_of::<Self>() as u64)
    }
    pub fn packets(&self) -> impl Iterator<Item = ProfiledExternalRetiredPacket<'_>> {
        self.state
            .entries
            .values()
            .map(|admission| ProfiledExternalRetiredPacket { admission })
    }
    pub fn validate_owner(&self, stores: &PalsStores) -> Result<(), PalsError> {
        if self.owner.matches(stores) && self.generation == stores.generation() {
            Ok(())
        } else {
            Err(StoreError::StaleConsumer.into())
        }
    }
    pub fn protected_pins(&self) -> Result<&StorePins, PalsError> {
        self.pins.as_ref().map_err(Clone::clone)
    }
}
pub struct ProfiledExternalRetiredPacket<'a> {
    admission: &'a ProfiledAdmission,
}
impl ProfiledExternalRetiredPacket<'_> {
    pub(crate) fn job(&self) -> &ProfiledExternalJob {
        &self.admission.job
    }
    pub(crate) fn config(&self) -> &ExternalCpuConfig {
        &self.admission.config
    }
    pub(crate) fn task_key(&self) -> &TaskKey {
        &self.admission.task_key
    }
    pub(crate) fn execution(&self) -> ExecutionId {
        self.admission.execution
    }
    pub(crate) fn consumer(&self) -> TaskConsumer {
        self.admission.consumer
    }
    pub(crate) fn prepared_observation(&self) -> Option<&Observation> {
        self.admission.prepared_observation.as_ref()
    }
    pub(crate) fn actual_consumed_observation(&self) -> Option<ObservationId> {
        self.admission.consumed
    }
    pub(crate) fn feedback_published(&self) -> bool {
        self.admission.feedback_published
    }
    pub(crate) fn terminal(&self) -> bool {
        self.admission.terminal
    }
    pub(crate) fn accounted(&self) -> bool {
        self.admission.accounted
    }
    pub fn work(&self) -> &CoordinatorWork<ReviewView, AdaptivePortQuestion> {
        &self.admission.work
    }
    pub fn dispatch(&self) -> Option<&ProfiledExternalDispatch> {
        self.admission.dispatch.as_ref()
    }
    pub fn completion(&self) -> Option<&Arc<ProfiledExternalCompletion>> {
        self.admission.completion.as_ref()
    }
    pub fn observation(&self) -> Option<ObservationId> {
        self.admission.published
    }
    pub fn search_consumed(&self) -> bool {
        self.admission.consumed.is_some()
    }
    pub fn publication_error(&self) -> Option<&PalsError> {
        self.admission.publication_error.as_ref()
    }
}

/// Interrupted facade conversion must retain the original, possibly live or
/// unpublishable sources with its Store. This is not a drain/reuse receipt.
pub(in crate::pals::engine) struct ProfiledExternalRetiredCustody {
    state: ProfiledExternalPortState,
    pins: Result<StorePins, PalsError>,
    owner: StoreReadOwner,
    generation: u64,
    authority: SearchAuthority,
}
impl ProfiledExternalRetiredCustody {
    pub(super) fn preserve(
        state: ProfiledExternalPortState,
        pins: Result<StorePins, PalsError>,
        owner: StoreReadOwner,
        generation: u64,
        authority: SearchAuthority,
    ) -> Option<Box<Self>> {
        (!state.is_empty()).then(|| {
            Box::new(Self {
                state,
                pins,
                owner,
                generation,
                authority,
            })
        })
    }
    pub(in crate::pals::engine) fn validate_owner(
        &self,
        stores: &PalsStores,
    ) -> Result<(), PalsError> {
        if self.owner.matches(stores)
            && self.generation == stores.generation()
            && self.generation == self.authority.root.0
        {
            Ok(())
        } else {
            Err(StoreError::StaleConsumer.into())
        }
    }
    pub(in crate::pals::engine) fn protected_pins(&self) -> Result<&StorePins, PalsError> {
        self.pins.as_ref().map_err(Clone::clone)
    }
    pub(in crate::pals::engine) fn owned_host_bytes(&self) -> Option<u64> {
        self.state.owned_host_bytes()
    }
}

#[derive(Clone)]
struct ProfiledAdmission {
    work: CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    source: ProfiledExternalRootSource,
    config: ExternalCpuConfig,
    job: ProfiledExternalJob,
    task_key: TaskKey,
    execution: ExecutionId,
    consumer: TaskConsumer,
    dispatch: Option<ProfiledExternalDispatch>,
    reservation: Option<FrontierExternalReservation>,
    binding: Option<CheckedProfiledExternalBinding>,
    completion: Option<Arc<ProfiledExternalCompletion>>,
    prepared_observation: Option<Observation>,
    published: Option<ObservationId>,
    consumed: Option<ObservationId>,
    feedback_published: bool,
    /// Exact ordinary after_profiled_external result, never minted by a reader.
    diagnostic_step_evidence: Option<ContractDigest>,
    publication_error: Option<PalsError>,
    terminal: bool,
    accounted: bool,
}

impl ProfiledAdmission {
    fn matches(&self, work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>) -> bool {
        self.work.frame == work.frame
            && self.work.original_first_move == work.original_first_move
            && self.work.original_deadline == work.original_deadline
            && self.work.input.stage == work.input.stage
            && self.work.input.question_digest == work.input.question_digest
            && self.work.input.view_seal == work.input.view_seal
            && Arc::ptr_eq(&self.work.input.question, &work.input.question)
            && Arc::ptr_eq(&self.work.input.view, &work.input.view)
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Preserve the actual root owner's original cleanup ceiling. This is
    /// metadata for closing the existing source, never a renewed admission or
    /// a change to submitted question / worker deadlines. The UCI owner passes
    /// its once-registered root boundary immediately after Frontier attachment.
    pub fn register_profiled_external_cleanup_deadline(
        &mut self,
        source: &ProfiledExternalRootSource,
        original: Instant,
    ) -> Result<(), PalsError> {
        if self.next_question != 1
            || self.root_ranking_pending
            || !self.profiled_external.entries.is_empty()
            || self.profiled_external.retirement_issued
            || self.profiled_external.original_cleanup_deadline.is_some()
            || original < self.limits.deadline
            || !self.engine.is_external()
            || source.authority() != self.authority
            || source.purpose() != ExternalEnginePurpose::ResearchChecker
            || source.clock().domain != self.deadline.clock
            || !self
                .rovent_frontier()
                .and_then(|frontier| frontier.external_source())
                .is_some_and(|actual| actual.same_source(source))
        {
            return Err(CheckerError::Invalid(
                "profiled cleanup boundary differs from original root admission",
            )
            .into());
        }
        self.profiled_external.original_cleanup_deadline = Some(original);
        Ok(())
    }

    /// Checks actual current-root ownership; an enum/config alone cannot
    /// authorize external assembly or replace the original Rules/view owner.
    pub fn validate_profiled_external_registration(
        &self,
        source: &ProfiledExternalRootSource,
        mode: rz_contracts::rovent::ExecutionMode,
        root_budget: ContractDigest,
    ) -> Result<(), PalsError> {
        source.validate_current()?;
        self.check_live()?;
        self.engine.validate_checker_namespace()?;
        let frontier = self
            .rovent_frontier()
            .ok_or(crate::pals::rovent_frontier::FrontierError::ForeignRoot)?;
        if !self.engine.is_external()
            || !self.engine.uses_model_resolver()
            || !mode.has_native_r()
            || frontier.registration().mode != mode
            || frontier.registration().root_budget != root_budget
            || frontier.registration().native_ledger.is_some()
            || source.authority() != self.authority
            || source.purpose() != ExternalEnginePurpose::ResearchChecker
            || source.clock().domain != self.deadline.clock
            || !frontier
                .external_source()
                .is_some_and(|actual| actual.same_source(source))
        {
            return Err(
                CheckerError::Invalid("profiled R differs from original external root").into(),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_profiled_external(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        config: &ExternalCpuConfig,
        source: &ProfiledExternalRootSource,
        id: ReciprocalQuestionId,
        execution: rz_contracts::ExecutionId,
        constraint: ExternalGoConstraint,
        soft_quantum: Duration,
        signal: CompletionSignal,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<AdaptiveProfiledExternalPreparation, PalsError> {
        self.check_work(work)?;
        // Check the same Store lineage as the eventual physical journal before
        // a worker can accept the go. The post-submit check remains necessary
        // for revocation and other changes during worker admission.
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
        config.validate()?;
        let frontier = self
            .rovent_frontier()
            .ok_or(crate::pals::rovent_frontier::FrontierError::ForeignRoot)?;
        let mode = frontier.registration().mode;
        let root_budget = frontier.registration().root_budget;
        self.validate_profiled_external_registration(source, mode, root_budget)?;
        if !super::external_commit::foreign_stage(work.input.stage)
            || self.engine.researcher.checker_registered_identity
                != CheckerIdentity::ExternalUci(config.identity.clone())
            || self.closed.contains(&work.frame)
            || self.engine.semantic.nodes[work.input.question.node]
                .terminal
                .is_some()
            || id.authority != self.authority
            || id.sequence == 0
            || execution.epoch != self.authority.epoch
            || execution.sequence == 0
        {
            return Err(
                CheckerError::Invalid("profiled R question namespace/stage differs").into(),
            );
        }
        let roots = super::external_commit::foreign_root_restriction(&work.input.question)?;
        if roots.is_some() && !source.profile().capabilities().searchmoves {
            return Ok(AdaptiveProfiledExternalPreparation::Unsupported(
                ProfiledExternalUnsupported::Searchmoves,
            ));
        }
        if matches!(constraint, ExternalGoConstraint::Combined { .. })
            && !source
                .profile()
                .capabilities()
                .combined_depth_nodes_movetime
        {
            return Ok(AdaptiveProfiledExternalPreparation::Unsupported(
                ProfiledExternalUnsupported::CombinedConstraint,
            ));
        }
        source
            .profile()
            .require_query(constraint, roots.is_some())?;
        if constraint
            .requested_depth()
            .is_some_and(|depth| depth > config.max_depth)
            || soft_quantum.is_zero()
            || soft_quantum > config.max_task_wall_time
        {
            return Err(
                CheckerError::Invalid("profiled question exceeds declared process limits").into(),
            );
        }
        work.input
            .question
            .position
            .snapshot()
            .uci_replay(config.max_prefix_plies)?;
        if let Some(original) = self
            .profiled_external
            .entries
            .get(&work.input.question_digest)
        {
            if !original.matches(work)
                || !original.source.same_source(source)
                || original.config != *config
                || (original.job.question.id != id
                    && self.profiled_reciprocal_question(work)? != Some(original.job.question))
                || original.job.execution != execution
                || original.job.constraint != constraint
                || original.job.soft_quantum != soft_quantum
                || original.terminal
            {
                return Err(
                    CheckerError::Invalid("profiled R preparation retry changed source").into(),
                );
            }
            return Ok(AdaptiveProfiledExternalPreparation::Ready(
                original.job.clone(),
            ));
        }
        // A hot observation and a closed semantic frame do not acknowledge
        // durable retirement of the original Work/view/dispatch/packet. Keep
        // every row until the whole-root retirement reaches its actual sink
        // acknowledgement. A long root exhausts this finite bound instead of
        // silently dropping evidence or renewing its allowance.
        reserve_profiled_question_slot(&mut self.profiled_external.entries)?;
        self.with_rovent_frontier(|frontier, _| {
            frontier.preflight_external_dispatch(source, clock, now)
        })?
        .ok_or(crate::pals::rovent_frontier::FrontierError::ForeignRoot)?;
        let peer = self.profiled_reciprocal_question(work)?;
        let anchor = if let Some(peer) = peer {
            peer.anchor
        } else {
            self.issue_rules_position(work.input.question.position.clone())?
                .export()
                .map_err(|_| RoleError::InvalidOutput)?
                .snapshot()
                .identity()
        };
        let sender = match work.input.stage {
            RefinementStage::CpuValidation | RefinementStage::RepairValidation => {
                SearchRecipient::Proposer
            }
            _ => SearchRecipient::Critic,
        };
        let question = peer.unwrap_or(QuestionEnvelope {
            id,
            parent: None,
            sender,
            recipient: SearchRecipient::Researcher,
            purpose: QuestionPurpose::TacticalReview,
            anchor,
            anchor_root_ply: ChessPly(
                u16::try_from(work.input.question.context.prefix.len())
                    .map_err(|_| PalsError::Capacity)?,
            ),
            original_first_move: Some(work.original_first_move),
            prefix_digest: ContractDigest(role_line_sha256(
                b"prefix",
                &work.input.question.context.prefix,
            )?),
            line_digest: None,
            line_offset: ChessPly(0),
            question_digest: work.input.question_digest,
            view: work.input.view_seal,
            evidence_revision: work.input.question.context.public_revision,
            root_budget,
            original_deadline: work.original_deadline,
        });
        if question.anchor != anchor
            || question.root_budget != root_budget
            || question.question_digest != work.input.question_digest
            || question.view != work.input.view_seal
            || question.original_deadline != work.original_deadline
            || question.recipient != SearchRecipient::Researcher
        {
            return Err(RoleError::InvalidOutput.into());
        }
        question
            .validate(mode, clock, now, work.input.view.token_budget())
            .map_err(|_| RoleError::InvalidOutput)?;
        let scope = ProfiledExternalScope {
            question: question.id,
            execution,
            owner: source.owner(),
        };
        let job = ProfiledExternalJob {
            question,
            mode,
            execution,
            owner: source.owner(),
            position: work.input.question.position.clone(),
            root_moves: roots.clone(),
            constraint,
            soft_quantum,
            context_token_budget: work.input.view.token_budget(),
            control: ControlMailbox::new(scope, signal),
        };
        let requested = ProfiledExternalConstraint::from(constraint);
        let condition = format!(
            "{};profiled-R/1;profile={:?};question={};requested={};deadline={}",
            self.engine.researcher.cpu_registered_condition,
            source.profile(),
            profiled_question_hex(work.input.question_digest),
            serde_json::to_string(&requested).map_err(|_| PalsError::InvalidConfig)?,
            self.deadline.at.0
        );
        let task_key = TaskKey {
            state: work.input.question.context.state,
            line: None,
            question: TaskQuestion::ProfiledExternal { requested },
            root_moves: roots
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .copied()
                .map(Move16::pack)
                .collect::<Result<_, _>>()?,
            model: 0,
            epoch: 0,
            value_identity: None,
            checker_identity: Some(CheckerIdentity::ExternalUci(config.identity.clone())),
            cpu_condition: Some(condition.clone()),
            profile: stable_id(source.profile().source_commit()),
            condition: stable_id(&condition),
            input_revision: work.input.question.context.public_revision,
            requested_depth: constraint.requested_depth().unwrap_or(0),
            node_budget: constraint.requested_nodes().unwrap_or(0),
        };
        self.engine.semantic.consumer_id = self
            .engine
            .semantic
            .consumer_id
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        let consumer = TaskConsumer {
            id: self.engine.semantic.consumer_id,
            situation: work.input.question.context.situation,
            revision: work.input.question.context.situation_revision,
            generation: self.engine.semantic.stores.generation(),
            deadline_tick: self.engine.tick_at(self.limits.deadline),
        };
        let admitted = self.engine.semantic.stores.hot_request_task(
            task_key.clone(),
            consumer,
            self.engine.tick_at(Instant::now()),
        )?;
        let store_execution = match admitted {
            TaskAdmission::Start(execution) => execution,
            TaskAdmission::Join(execution)
            | TaskAdmission::Reuse { execution, .. }
            | TaskAdmission::Resume { execution, .. } => {
                self.engine
                    .semantic
                    .stores
                    .tasks
                    .cancel_consumer(execution, consumer.id)?;
                if matches!(admitted, TaskAdmission::Resume { .. }) {
                    self.engine.semantic.stores.tasks.fail(execution)?;
                }
                return Err(CheckerError::Invalid(
                    "profiled physical question cannot reuse a different go",
                )
                .into());
            }
        };
        self.profiled_external.entries.insert(
            work.input.question_digest,
            ProfiledAdmission {
                work: work.clone(),
                source: source.clone(),
                config: config.clone(),
                job: job.clone(),
                task_key,
                execution: store_execution,
                consumer,
                dispatch: None,
                reservation: None,
                binding: None,
                completion: None,
                prepared_observation: None,
                published: None,
                consumed: None,
                feedback_published: false,
                diagnostic_step_evidence: None,
                publication_error: None,
                terminal: false,
                accounted: false,
            },
        );
        Ok(AdaptiveProfiledExternalPreparation::Ready(job))
    }

    pub fn profiled_external_admitted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        source: &ProfiledExternalRootSource,
        dispatch: &ProfiledExternalDispatch,
        clock: ClockDomain,
        now: MonotonicTick,
    ) -> Result<AdaptiveProfiledExternalAdmission, PalsError> {
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
        let admission = self
            .profiled_external
            .entries
            .get(&work.input.question_digest)
            .ok_or(RoleError::InvalidOutput)?
            .clone();
        if !admission.matches(work)
            || !admission.source.same_source(source)
            || dispatch.question() != admission.job.question
            || dispatch.scope() != admission.job.scope()
            || dispatch.constraint() != admission.job.constraint
            || dispatch.process_identity() != source.process_identity()
            || dispatch.root_restriction()
                != external_root_restriction_digest(admission.job.root_moves.as_deref())
        {
            return Err(CheckerError::Invalid(
                "profiled dispatch differs from immutable preparation",
            )
            .into());
        }
        if admission.dispatch.is_some() && admission.reservation.is_some() {
            return Ok(AdaptiveProfiledExternalAdmission {
                binding: admission.binding.ok_or(RoleError::InvalidOutput)?,
                reservation: admission.reservation.ok_or(RoleError::InvalidOutput)?,
            });
        }
        let binding = CheckedProfiledExternalBinding::attest(
            work,
            source,
            dispatch,
            admission.job.mode,
            clock,
            now,
        )
        .map_err(|_| RoleError::InvalidOutput)?;
        let tasks = self
            .counters
            .external_checker_tasks
            .checked_add(1)
            .ok_or(PalsError::Capacity)?;
        // Preserve actual mailbox ownership before any fallible projection.
        // A projection failure cannot reclassify this as an unsubmitted task.
        {
            let stored = self
                .profiled_external
                .entries
                .get_mut(&work.input.question_digest)
                .expect("checked original physical dispatch");
            stored.dispatch = Some(dispatch.clone());
            stored.binding = Some(binding.clone());
        }
        let reservation = self
            .with_rovent_frontier(|frontier, _| {
                frontier.record_observed_external_dispatch(source, dispatch)
            })?
            .ok_or(crate::pals::rovent_frontier::FrontierError::ForeignRoot)?;
        let stored = self
            .profiled_external
            .entries
            .get_mut(&work.input.question_digest)
            .expect("checked profiled admission");
        stored.dispatch = Some(dispatch.clone());
        stored.reservation = Some(reservation);
        stored.binding = Some(binding.clone());
        self.counters.external_checker_tasks = tasks;
        Ok(AdaptiveProfiledExternalAdmission {
            binding,
            reservation,
        })
    }

    pub fn profiled_external_work(&self) -> Option<FrontierExternalWork> {
        self.rovent_frontier()
            .and_then(|frontier| frontier.external_work())
    }
    pub fn profiled_external_retained_host_bytes(&self) -> Option<u64> {
        self.profiled_external.owned_host_bytes()
    }
    pub(super) fn profiled_external_protected_pins(&self) -> Result<StorePins, PalsError> {
        self.profiled_external
            .protected_pins(&self.engine.semantic.stores)
    }

    fn observe_profiled_packet(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        source: &ProfiledExternalRootSource,
        reservation: FrontierExternalReservation,
        completion: &Arc<ProfiledExternalCompletion>,
    ) -> Result<ProfiledAdmission, PalsError> {
        AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
        let admission = self
            .profiled_external
            .entries
            .get(&work.input.question_digest)
            .ok_or(RoleError::InvalidOutput)?
            .clone();
        let dispatch = admission
            .dispatch
            .as_ref()
            .ok_or(RoleError::InvalidOutput)?;
        if !admission.matches(work)
            || !admission.source.same_source(source)
            || admission.reservation != Some(reservation)
            || !completion.matches_submission(dispatch)
            || completion.question() != admission.job.question
            || !completion.controls().terminal
            || completion.controls().cancel_ack > completion.controls().cancel_sequence
            || completion.controls().yield_ack > completion.controls().yield_sequence
            || completion.controls().publish_ack > completion.controls().publish_sequence
            || completion.process_identity() != Some(source.process_identity())
            || admission
                .completion
                .as_ref()
                .is_some_and(|old| !Arc::ptr_eq(old, completion))
        {
            return Err(CheckerError::Invalid(
                "profiled terminal packet differs from actual dispatch",
            )
            .into());
        }
        self.with_rovent_frontier(|frontier, _| {
            frontier.record_external_completion(source, reservation, completion)
        })?
        .ok_or(crate::pals::rovent_frontier::FrontierError::ForeignRoot)?;
        // Retain the same packet before any raw append/consumption can fail.
        self.profiled_external
            .entries
            .get_mut(&work.input.question_digest)
            .expect("checked profiled packet")
            .completion = Some(Arc::clone(completion));
        // Physical terminal and logical Task lifetime are independent of a
        // later archive append. A late known packet cannot keep a Task active.
        if !admission.terminal {
            self.engine
                .semantic
                .stores
                .tasks
                .fail(admission.execution)?;
            self.profiled_external
                .entries
                .get_mut(&work.input.question_digest)
                .expect("checked task retirement")
                .terminal = true;
        }
        if !admission.accounted {
            if let Ok(outcome) = completion.result() {
                let reports = self
                    .counters
                    .external_checker_reports
                    .checked_add(1)
                    .ok_or(PalsError::Capacity)?;
                let nodes = self
                    .counters
                    .external_checker_nodes_observed
                    .checked_add(outcome.report.raw.work.nodes.unwrap_or(0))
                    .ok_or(PalsError::Capacity)?;
                self.counters.external_checker_reports = reports;
                self.counters.external_checker_nodes_observed = nodes;
                self.counters.external_checker_work_incomplete |=
                    outcome.report.raw.work.nodes.is_none()
                        || outcome.report.raw.work.qnodes.is_none()
                        || outcome.report.raw.work.tt_hits.is_none();
            } else {
                self.counters.external_checker_work_incomplete = true;
            }
            self.profiled_external
                .entries
                .get_mut(&work.input.question_digest)
                .expect("checked packet accounting")
                .accounted = true;
        }
        Ok(self.profiled_external.entries[&work.input.question_digest].clone())
    }

    fn profiled_raw_observation(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        admission: &ProfiledAdmission,
        completion: &Arc<ProfiledExternalCompletion>,
    ) -> Result<Observation, PalsError> {
        if let Some(original) = &admission.prepared_observation {
            return Ok(original.clone());
        }
        let outcome = completion.result().as_ref().map_err(Clone::clone)?;
        let profiled = &outcome.report;
        let raw = &profiled.raw;
        if completion.go_issue() != ExternalGoIssue::Issued
            || profiled.profile != admission.source.profile()
            || profiled.purpose != ExternalEnginePurpose::ResearchChecker
            || profiled.requested != admission.job.constraint
            || profiled.strict_native_nodes.is_some()
            || raw.identity != admission.config.identity
            || raw.request_id == 0
            || raw.requested_depth != admission.job.constraint.requested_depth().unwrap_or(0)
            || raw.perspective != admission.job.position.side_to_move()
            || raw.root_restricted != admission.job.root_moves.is_some()
        {
            return Err(
                CheckerError::Invalid("profiled raw report changed source/constraint").into(),
            );
        }
        // The complete raw report is retained independently. Every publication
        // stage must additionally meet the Store's exact bounded-PV contract;
        // validation/peer R must not be relabelled as discovery to obtain it.
        let projection_plies = self
            .engine
            .semantic
            .stores
            .hot_external_pv_projection_plies(raw.pv.len());
        let projected = &raw.pv[..projection_plies];
        let pv = if projected.is_empty() {
            None
        } else {
            self.check_work(work)?;
            admission.source.validate_current()?;
            let mut prefix = work.input.question.context.prefix.clone();
            prefix
                .try_reserve_exact(projected.len())
                .map_err(|_| PalsError::Capacity)?;
            prefix.extend_from_slice(projected);
            // This existing path handles every checked ply under the original
            // horizon, quota and deadline, including paths longer than one CPU
            // chunk. Failure retains any paid partial path and the completed
            // packet, but no partial Store observation is manufactured.
            let checked = match self.existing_checked_frontier_line(&prefix)? {
                Some(checked) => checked,
                None => self.checked_frontier_line(&prefix)?,
            };
            Some(self.engine.semantic.stores.hot_append_checked_cpu_pv(
                &admission.job.position,
                projected,
                &checked,
            )?)
        };
        let observation = Observation {
            state: admission.task_key.state,
            line: None,
            source: stable_id(&raw.identity.adapter_semantics),
            epoch: 0,
            value_identity: None,
            checker_identity: admission.task_key.checker_identity.clone(),
            checker_work: Some(raw.work),
            external_report: Some(Box::new(raw.clone())),
            model_value_identity: None,
            model_value_input: None,
            cpu_condition: admission.task_key.cpu_condition.clone(),
            cpu_pv: pv,
            scope: EvidenceScope::ProfiledExternalUci {
                requested: profiled.requested.into(),
                issued: profiled.issued.into(),
                reported_depth: raw.reported_depth,
                seldepth: raw.seldepth,
                bound: raw.bound,
            },
            score: RawScore::ExternalUci {
                value: raw.score,
                bound: raw.bound,
                perspective: raw.perspective,
                wdl_per_mille: raw.wdl_per_mille,
            },
            budget: raw.work.nodes.unwrap_or(0),
            kind: ObservationKind::ExternalCpuAnalysis,
            supersedes: None,
            execution: Some(admission.execution),
        };
        self.profiled_external
            .entries
            .get_mut(&work.input.question_digest)
            .expect("checked raw preparation")
            .prepared_observation = Some(observation.clone());
        Ok(observation)
    }

    pub fn prepare_profiled_external_publication(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        source: &ProfiledExternalRootSource,
        reservation: FrontierExternalReservation,
        completion: &Arc<ProfiledExternalCompletion>,
    ) -> Result<AdaptiveProfiledForeignPublication, PalsError> {
        let result = (|| {
            let admission = self.observe_profiled_packet(work, source, reservation, completion)?;
            if admission.published.is_some() || !self.engine.semantic.stores.archive_enabled() {
                return Err(CheckerError::Invalid(
                    "profiled permit needs an unpublished archive-on packet",
                )
                .into());
            }
            let observation = self.profiled_raw_observation(work, &admission, completion)?;
            let issued = ProfiledForeignGoIssuedProof::checked_worker_report(
                completion,
                admission.dispatch.as_ref().expect("checked dispatch"),
                source,
                &observation,
                self.limits.deadline,
            )?;
            Ok(AdaptiveProfiledForeignPublication {
                observation,
                issued,
            })
        })();
        self.retain_profiled_publication_error(work.input.question_digest, &result);
        result
    }

    pub fn commit_profiled_external(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        source: &ProfiledExternalRootSource,
        reservation: FrontierExternalReservation,
        completion: &Arc<ProfiledExternalCompletion>,
        consume_current: bool,
    ) -> Result<AdaptiveProfiledExternalCommit, PalsError> {
        let result = self.commit_profiled_external_inner(
            work,
            source,
            reservation,
            completion,
            consume_current,
            None,
        );
        self.retain_profiled_publication_error(work.input.question_digest, &result);
        result
    }

    pub fn commit_profiled_external_permitted(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        source: &ProfiledExternalRootSource,
        reservation: FrontierExternalReservation,
        completion: &Arc<ProfiledExternalCompletion>,
        consume_current: bool,
        permit: ProfiledForeignAppendPermit,
    ) -> Result<AdaptiveProfiledExternalCommit, PalsError> {
        let result = self.commit_profiled_external_inner(
            work,
            source,
            reservation,
            completion,
            consume_current,
            Some(permit),
        );
        self.retain_profiled_publication_error(work.input.question_digest, &result);
        result
    }

    fn retain_profiled_publication_error<T>(
        &mut self,
        question: ContractDigest,
        result: &Result<T, PalsError>,
    ) {
        if let Some(admission) = self.profiled_external.entries.get_mut(&question) {
            if admission.completion.is_some() {
                if let Err(error) = result {
                    admission.publication_error = Some(error.clone());
                } else if admission.feedback_published {
                    admission.publication_error = None;
                }
                // A raw-only drain or successful permit preparation does not
                // publish the refused feedback. Retain its original failure
                // for retirement instead of treating observation as recovery.
            }
        }
    }

    fn commit_profiled_external_inner(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        source: &ProfiledExternalRootSource,
        reservation: FrontierExternalReservation,
        completion: &Arc<ProfiledExternalCompletion>,
        consume_current: bool,
        permit: Option<ProfiledForeignAppendPermit>,
    ) -> Result<AdaptiveProfiledExternalCommit, PalsError> {
        let admission = self.observe_profiled_packet(work, source, reservation, completion)?;
        if let Err(error) = completion.result() {
            if !admission.terminal {
                self.engine
                    .semantic
                    .stores
                    .tasks
                    .fail(admission.execution)?;
            }
            self.profiled_external
                .entries
                .get_mut(&work.input.question_digest)
                .expect("checked failed packet")
                .terminal = true;
            return Err(error.clone().into());
        }
        let observation = if let Some(observation) = admission.published {
            if permit.is_some() {
                return Err(CheckerError::Invalid("profiled packet already published").into());
            }
            observation
        } else {
            if self.engine.semantic.stores.archive_enabled() && permit.is_none() {
                return Err(StoreError::ArchiveBudgetRequired.into());
            }
            let expected = self.profiled_raw_observation(work, &admission, completion)?;
            let observation = if let Some(permit) = permit {
                if permit.observation() != &expected
                    || permit.native_execution() != completion.scope().execution
                    || permit.question() != completion.question().id
                {
                    return Err(CheckerError::Invalid(
                        "profiled append permit belongs to another raw go",
                    )
                    .into());
                }
                self.engine
                    .semantic
                    .stores
                    .hot_append_profiled_foreign_permitted(permit)?
            } else {
                self.engine
                    .semantic
                    .stores
                    .hot_append_observation(expected)?
            };
            self.profiled_external
                .entries
                .get_mut(&work.input.question_digest)
                .expect("checked profiled append")
                .published = Some(observation);
            observation
        };
        self.engine
            .semantic
            .stores
            .hot_add_dependency(observation, admission.consumer.situation)?;
        self.engine
            .semantic
            .stores
            .hot_add_dependency(observation, self.engine.semantic.nodes[self.root].situation)?;
        self.engine.pin_archive_active_observation(observation)?;
        if !admission.terminal {
            // A foreign report is never a reusable native/completed-depth task.
            self.engine
                .semantic
                .stores
                .tasks
                .fail(admission.execution)?;
            self.profiled_external
                .entries
                .get_mut(&work.input.question_digest)
                .expect("checked settled raw")
                .terminal = true;
        }
        let outcome = completion
            .result()
            .as_ref()
            .expect("checked successful packet");
        let usable = matches!(
            outcome.report.raw.completion,
            ExternalCompletion::BestMove | ExternalCompletion::StoppedYield
        ) && outcome.report.raw.best_move.is_some();
        let current = consume_current
            && usable
            && self.check_work(work).is_ok()
            && source.validate_current().is_ok();
        let already = self.profiled_external.entries[&work.input.question_digest].consumed
            == Some(observation);
        if current && !already {
            let consumed_count = self
                .counters
                .consumed_external_checker_tasks
                .checked_add(1)
                .ok_or(PalsError::Capacity)?;
            match self
                .engine
                .semantic
                .stores
                .hot_consume_sealed_profiled_foreign_feedback(
                    admission.execution,
                    admission.consumer,
                    &admission.task_key,
                    &admission.job.position.snapshot(),
                    observation,
                    self.engine.tick_at(Instant::now()),
                ) {
                Ok(actual) if actual == observation => {}
                Ok(_) => {
                    return Err(StoreError::InvalidEvidence(
                        "profiled feedback consumed another observation",
                    )
                    .into());
                }
                Err(
                    StoreError::ExpiredConsumer
                    | StoreError::CancelledConsumer
                    | StoreError::StaleConsumer,
                ) => {
                    return Ok(AdaptiveProfiledExternalCommit {
                        observation,
                        consumed: false,
                    });
                }
                Err(error) => return Err(error.into()),
            }
            self.profiled_external
                .entries
                .get_mut(&work.input.question_digest)
                .expect("checked raw consumption")
                .consumed = Some(observation);
            self.counters.consumed_external_checker_tasks = consumed_count;
        }
        if current
            && !self.profiled_external.entries[&work.input.question_digest].feedback_published
        {
            if work.input.stage != RefinementStage::CpuDiscovery {
                // Retry this separate publication after an actual head use;
                // its failure must not silently become successful continuation.
                self.record_hot(
                    RecordKind::CpuVerification,
                    &admission.work.input.question.context.prefix,
                    Some(observation),
                )?;
            }
            self.profiled_external
                .entries
                .get_mut(&work.input.question_digest)
                .expect("checked feedback publication")
                .feedback_published = true;
        }
        Ok(AdaptiveProfiledExternalCommit {
            observation,
            consumed: current
                && (already
                    || self.profiled_external.entries[&work.input.question_digest].consumed
                        == Some(observation)),
        })
    }

    pub fn after_profiled_external(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        committed: &AdaptiveProfiledExternalCommit,
    ) -> Result<AdaptivePortStep, PalsError> {
        if !committed.consumed
            || !self.profiled_external_observation_consumed(work, committed.observation)
        {
            return Err(StoreError::InvalidConditions(
                "profiled raw-only packet cannot advance refinement",
            )
            .into());
        }
        let mut step = self.after_cpu(work, committed.observation, false)?;
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-profiled-R-raw-feedback/1\0");
        hash.update(work.input.question_digest.0);
        hash.update((committed.observation.0 as u64).to_le_bytes());
        step.evidence = ContractDigest(hash.finalize().into());
        // Observation only. The existing successful source/Store/transition
        // checks above remain the sole producer of this original evidence.
        if self.role_consumption_observation.is_some() {
            if let Some(admission) = self
                .profiled_external
                .entries
                .get_mut(&work.input.question_digest)
            {
                admission.diagnostic_step_evidence = Some(step.evidence);
            }
        }
        Ok(step)
    }

    /// Return an actual optional peer R observation to its retained accepted
    /// parent. This does not take the ordinary fixed refinement transition or
    /// issue a native CPU receipt, paused-stack reply, depth or mate proof.
    pub fn after_profiled_reciprocal_cpu(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        source: &ProfiledExternalRootSource,
        reservation: FrontierExternalReservation,
        packet: &Arc<ProfiledExternalCompletion>,
        committed: &AdaptiveProfiledExternalCommit,
    ) -> Result<CheckedProfiledExternalResponse, PalsError> {
        self.check_work(work)?;
        let question = self
            .profiled_reciprocal_question(work)?
            .ok_or(RoleError::InvalidOutput)?;
        let original = self
            .profiled_external
            .entries
            .get(&work.input.question_digest)
            .ok_or(RoleError::InvalidOutput)?;
        if !committed.consumed
            || !original.matches(work)
            || !original.source.same_source(source)
            || original.job.question != question
            || original.reservation != Some(reservation)
            || !original
                .completion
                .as_ref()
                .is_some_and(|same| Arc::ptr_eq(same, packet))
            || !self.profiled_external_observation_consumed(work, committed.observation)
        {
            return Err(StoreError::InvalidEvidence(
                "peer profiled R needs its exact consumed observation",
            )
            .into());
        }
        let mut hash = Sha256::new();
        hash.update(b"rz-rovent-consumed-profiled-peer-R-observation/1\0");
        hash.update(work.input.question_digest.0);
        hash.update(work.input.view_seal.view_digest.0);
        hash.update(original.job.execution.sequence.to_le_bytes());
        hash.update(question.id.sequence.to_le_bytes());
        hash.update((committed.observation.0 as u64).to_le_bytes());
        Ok(CheckedProfiledExternalResponse {
            frame: work.frame,
            stage: work.input.stage,
            first: work.original_first_move,
            deadline: work.original_deadline,
            question,
            execution: original.job.execution,
            input_digest: ContractDigest(work.input.view.digest()),
            public_context: work.input.public_context,
            evidence: ContractDigest(hash.finalize().into()),
        })
    }

    pub fn release_profiled_external_root(
        &mut self,
        source: &ProfiledExternalRootSource,
        fence: &ProfiledExternalCurrentRootFence,
    ) -> Result<ProfiledExternalRootRetirement, PalsError> {
        if !fence.source().same_source(source)
            || fence.authority() != self.authority
            || !fence.permits_reuse()
        {
            return Err(
                CheckerError::Invalid("profiled root lacks actual original drain fence").into(),
            );
        }
        self.retire_profiled_external_state(source, ProfiledRetirementBoundary::Root(fence.clone()))
    }

    /// Known process exit/drain is a close-only proof. It cannot synthesize a
    /// ready barrier, same-process reuse or a subsequent root admission.
    pub fn release_profiled_external_closed_root(
        &mut self,
        source: &ProfiledExternalRootSource,
        proof: &ProfiledExternalProcessCloseProof,
    ) -> Result<ProfiledExternalRootRetirement, PalsError> {
        if !proof.source().same_source(source)
            || proof.authority() != self.authority
            || proof.worker_owner() != source.owner()
            || proof.process_identity() != source.process_identity()
            || proof.clock().domain != source.clock().domain
            || proof.clock().origin != source.clock().origin
            || !proof.permits_semantic_closure()
        {
            return Err(
                CheckerError::Invalid("profiled root lacks actual process close proof").into(),
            );
        }
        self.retire_profiled_external_state(
            source,
            ProfiledRetirementBoundary::Process(proof.clone()),
        )
    }

    fn retire_profiled_external_state(
        &mut self,
        source: &ProfiledExternalRootSource,
        boundary: ProfiledRetirementBoundary,
    ) -> Result<ProfiledExternalRootRetirement, PalsError> {
        if self.profiled_external.retirement_issued
            || !self
                .rovent_frontier()
                .and_then(|f| f.external_source())
                .is_some_and(|actual| actual.same_source(source))
            || self.profiled_external.entries.values().any(|admission| {
                !admission.terminal
                    || (admission.dispatch.is_some() && admission.completion.is_none())
            })
            || self
                .profiled_external_work()
                .is_some_and(|work| work.outstanding_questions != 0)
        {
            return Err(
                CheckerError::Invalid("profiled root lacks actual original drain fence").into(),
            );
        }
        let pins = self.profiled_external_protected_pins();
        let state = std::mem::take(&mut self.profiled_external);
        self.profiled_external.retirement_issued = true;
        Ok(ProfiledExternalRootRetirement {
            identity: ProfiledExternalRetirementIdentity(Arc::new(())),
            boundary,
            authority: self.authority,
            source: source.clone(),
            owner: self.engine.semantic.stores.read_owner(),
            generation: self.engine.semantic.stores.generation(),
            pins,
            state,
        })
    }

    /// A definite pre-dispatch rejection fails only the untouched Store task.
    pub fn reject_profiled_external(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<(), PalsError> {
        let admission = self
            .profiled_external
            .entries
            .get_mut(&work.input.question_digest)
            .ok_or(RoleError::InvalidOutput)?;
        if !admission.matches(work)
            || admission.dispatch.is_some()
            || admission.completion.is_some()
        {
            return Err(
                CheckerError::Invalid("profiled rejection is not an unsubmitted question").into(),
            );
        }
        self.engine
            .semantic
            .stores
            .tasks
            .fail(admission.execution)?;
        admission.terminal = true;
        Ok(())
    }

    pub(super) fn profiled_external_feedback_still_authorized(
        &self,
        question: ContractDigest,
        observation: ObservationId,
    ) -> bool {
        self.profiled_external
            .entries
            .get(&question)
            .is_some_and(|admission| {
                admission.terminal
                    && admission.feedback_published
                    && admission.published == Some(observation)
                    && admission.consumed == Some(observation)
                    && admission.source.validate_current().is_ok()
                    && self
                        .engine
                        .semantic
                        .stores
                        .hot_validate_consumed_profiled_foreign_feedback(
                            admission.execution,
                            admission.consumer,
                            &admission.task_key,
                            &admission.job.position.snapshot(),
                            observation,
                            self.engine.tick_at(Instant::now()),
                        )
                        .is_ok()
            })
    }

    pub(super) fn profiled_external_observation_consumed(
        &self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        observation: ObservationId,
    ) -> bool {
        self.profiled_external
            .entries
            .get(&work.input.question_digest)
            .is_some_and(|admission| {
                admission.matches(work)
                    && self.profiled_external_feedback_still_authorized(
                        work.input.question_digest,
                        observation,
                    )
            })
    }
}
