//! Opt-in frozen-proposal intervention. A normal Ready offer is only the
//! original neutral checkpoint; it never pretends that every arm was Ready.
use super::*;
use crate::pals::coordinator::UtilityInterventionReadySource;
use crate::pals::review_context::ReviewQuerySequences;
use rz_contracts::pals::Move16 as ContractMove16;

pub const UTILITY_INTERVENTION_TARGET_DOMAIN: &str = "rz-rovent-utility-intervention-target/1";

/// Uses the same codec as the original submission builder. This historical
/// projection checks its actual digest but grants no live execution rights.
pub fn utility_intervention_arm_question_preimage(
    work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
) -> Result<AdaptiveQuestionDigestPreimage, PalsError> {
    let question = &work.input.question;
    let value = AdaptiveQuestionDigestPreimage {
        sequence: question.question_sequence,
        stage_code: question.stage as u8,
        view_digest: ContractDigest(work.input.view.digest()),
        view_seal_digest: ContractDigest(work.input.view.seal_digest()),
        focus_prefix_sha256: ContractDigest(question.context.focus_prefix_sha256()),
        goal_local_sha256: question.goal_local.as_ref().map(|origin| origin.digest()),
        legacy_full_records_sha256: question
            .legacy_full_records
            .as_ref()
            .map(|records| ContractDigest(records.digest())),
    };
    if question.stage != work.input.stage
        || question.question_digest != work.input.question_digest
        || value.digest() != work.input.question_digest
        || work.input.view_seal.authority != work.frame.authority
        || work.input.view_seal.view_digest != value.view_seal_digest
        || work.input.public_context.0 != work.input.view.record_scope_digest()
    {
        return Err(RoleError::InvalidOutput.into());
    }
    Ok(value)
}

/// Source codec fields, not another wire schema or a live admission proof.
pub struct UtilityInterventionTargetDigestPreimage {
    pub root_snapshot_digest: ContractDigest,
    pub record_scope_digest: ContractDigest,
    pub source_snapshot_digests: Vec<ContractDigest>,
}

/// Pure historical byte codec only. It neither validates a submitted View nor
/// reconstructs its Store pin, Ready source, Request or execution authority.
pub fn utility_intervention_historical_snapshot_digest(
    snapshot: &PositionSnapshot,
) -> Result<ContractDigest, PalsError> {
    Ok(ContractDigest(
        crate::pals::review_context::snapshot_digest(snapshot)?,
    ))
}

/// Borrowed, already decoded historical content. Callers remain responsible
/// for strict Rules replay, source/record relations, original reference bytes
/// and registered model identity; a resulting digest grants no live rights.
pub struct UtilityInterventionHistoricalTargetContent<'a> {
    pub root_snapshot: &'a PositionSnapshot,
    pub record_scope_digest: ContractDigest,
    pub target_record: u16,
    pub original_first_move: ContractMove16,
    pub reference_json: &'a [u8],
    pub model_identity: &'a crate::cpu_value::CpuValueIdentity,
    pub source_snapshots: &'a [PositionSnapshot],
}

/// Shared hash framing for actual capture and detached historical content.
/// No owner, generation, clock, Request, actor, or worker can be issued here.
pub fn utility_intervention_historical_target_digest(
    content: UtilityInterventionHistoricalTargetContent<'_>,
) -> Result<ContractDigest, PalsError> {
    let mut hash = Sha256::new();
    hash.update(UTILITY_INTERVENTION_TARGET_DOMAIN.as_bytes());
    hash.update([0]);
    hash.update(utility_intervention_historical_snapshot_digest(content.root_snapshot)?.0);
    hash.update(content.record_scope_digest.0);
    hash.update(content.target_record.to_le_bytes());
    hash.update(content.original_first_move.bits().to_le_bytes());
    utility_route_bytes(&mut hash, content.reference_json);
    utility_route_bytes(
        &mut hash,
        &serde_json::to_vec(content.model_identity).map_err(|_| RoleError::InvalidOutput)?,
    );
    hash.update((content.source_snapshots.len() as u64).to_le_bytes());
    for snapshot in content.source_snapshots {
        hash.update(utility_intervention_historical_snapshot_digest(snapshot)?.0);
    }
    Ok(ContractDigest(hash.finalize().into()))
}

pub fn utility_intervention_target_digest_preimage(
    input: &NnuecUtilityInputCapture,
) -> Result<UtilityInterventionTargetDigestPreimage, PalsError> {
    Ok(UtilityInterventionTargetDigestPreimage {
        root_snapshot_digest: utility_intervention_historical_snapshot_digest(
            input.root_snapshot(),
        )?,
        record_scope_digest: ContractDigest(input.provenance().record_scope_digest),
        source_snapshot_digests: input
            .source_snapshots()
            .iter()
            .map(utility_intervention_historical_snapshot_digest)
            .collect::<Result<_, _>>()?,
    })
}

/// The shared, pure digest codec for a detached receipt. The input is still an
/// actual actor capture; this projection does not recreate its live authority.
pub fn utility_intervention_target_digest(
    input: &NnuecUtilityInputCapture,
) -> Result<ContractDigest, PalsError> {
    utility_intervention_historical_target_digest(UtilityInterventionHistoricalTargetContent {
        root_snapshot: input.root_snapshot(),
        record_scope_digest: ContractDigest(input.provenance().record_scope_digest),
        target_record: input.target_record(),
        original_first_move: ContractMove16::encode(
            rz_contracts::Move::try_from(input.original_first_move())
                .map_err(|_| RoleError::InvalidOutput)?,
        ),
        reference_json: input.input_bytes(),
        model_identity: input.model_identity(),
        source_snapshots: input.source_snapshots(),
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UtilityInterventionUnavailable {
    UnsupportedMode,
    MissingRoleCredit,
    ProposalAlreadyAtLimit,
    TerminalProposal,
    NoOpponentDivergence,
    MissingNativeResearcher,
    MissingNativeCredit,
}

/// Bounded content checked before the original preparation becomes a Target.
/// This is not a Ready/worker grant and has no public construction path.
struct CheckedInterventionTarget {
    proposal: Vec<BoardMove>,
    unavailable: [Option<UtilityInterventionUnavailable>; 3],
    digest: ContractDigest,
}

/// This owner was bound through the original same-Request C preparation. Its
/// source work/view are retained until the intervention's physical retirement.
/// No deserializer or public boolean can construct it.
pub struct UtilityInterventionTarget<Q = AdaptivePortQuestion> {
    preparation: UtilityRoutePreparation,
    input: Arc<NnuecUtilityInputCapture>,
    source: UtilityInterventionReadySource<Q>,
    digest: ContractDigest,
    proposal: Vec<BoardMove>,
    unavailable: [Option<UtilityInterventionUnavailable>; 3],
}
impl<Q> UtilityInterventionTarget<Q> {
    pub fn source_frame(&self) -> FrameId {
        self.source.work().frame
    }
    pub fn source_work(&self) -> &CoordinatorWork<ReviewView, Q> {
        self.source.work()
    }
    pub fn input_capture(&self) -> &Arc<NnuecUtilityInputCapture> {
        &self.input
    }
    pub fn neutral_target_digest(&self) -> ContractDigest {
        self.digest
    }
    pub fn target_record(&self) -> u16 {
        self.preparation.target_record
    }
    pub fn record_scope_digest(&self) -> ContractDigest {
        ContractDigest(self.source.work().input.view.record_scope_digest())
    }
    pub fn first_move(&self) -> ContractMove16 {
        self.source.work().original_first_move
    }
    pub fn budget(&self) -> FrameBudget {
        self.source.work().budget
    }
    /// Original bounded witness set, including superseded owner publications.
    /// Iteration order is unspecified; compare the set, not HashSet traversal.
    pub fn baseline_witnesses(&self) -> impl Iterator<Item = &ContractDigest> {
        self.preparation.baseline.iter()
    }
    pub fn original_deadline(&self) -> Deadline {
        self.source.work().original_deadline
    }
    pub fn unsupported_reason(
        &self,
        recipient: UtilityRouteRecipient,
    ) -> Option<UtilityInterventionUnavailable> {
        self.unavailable[recipient.output_index() - 4]
    }
    /// A source-side probe, not an installed-worker receipt or work grant.
    /// The eventual role/CPU dispatcher still verifies actual capabilities.
    pub fn supported_recipients(&self) -> impl Iterator<Item = UtilityRouteRecipient> + '_ {
        [
            UtilityRouteRecipient::Proposer,
            UtilityRouteRecipient::Critic,
            UtilityRouteRecipient::Researcher,
        ]
        .into_iter()
        .filter(|recipient| self.unsupported_reason(*recipient).is_none())
    }
    fn matches_source(&self, work: &CoordinatorWork<ReviewView, Q>) -> bool {
        let source = self.source.work();
        source.frame == work.frame
            && source.original_first_move == work.original_first_move
            && source.original_deadline == work.original_deadline
            && source.budget == work.budget
            && source.input.stage == work.input.stage
            && source.input.question_digest == work.input.question_digest
            && source.input.view_seal == work.input.view_seal
            && source.input.public_context == work.input.public_context
            && Arc::ptr_eq(&source.input.question, &work.input.question)
            && Arc::ptr_eq(&source.input.view, &work.input.view)
    }
}

/// Actual arm query minted by this Search owner. The coordinator must consume
/// it once and replace its exact unexecuted Ready source, without new credit.
pub struct UtilityInterventionCandidate<Q = AdaptivePortQuestion> {
    target: UtilityInterventionTarget<Q>,
    input: FrameInput<ReviewView, Q>,
    recipient: UtilityRouteRecipient,
    started: Instant,
}
impl<Q> UtilityInterventionCandidate<Q> {
    pub(crate) fn matches_source_owner(&self, owner: &Arc<()>) -> bool {
        self.target.source.matches_owner(owner)
    }
    pub fn source_frame(&self) -> FrameId {
        self.target.source_frame()
    }
    pub fn input(&self) -> &FrameInput<ReviewView, Q> {
        &self.input
    }
    pub fn first_move(&self) -> ContractMove16 {
        self.target.first_move()
    }
    pub fn budget(&self) -> FrameBudget {
        self.target.budget()
    }
    pub fn original_deadline(&self) -> Deadline {
        self.target.original_deadline()
    }
    pub fn neutral_target_digest(&self) -> ContractDigest {
        self.target.digest
    }
    pub fn arm_query_digest(&self) -> ContractDigest {
        self.input.question_digest
    }
    pub fn recipient(&self) -> UtilityRouteRecipient {
        self.recipient
    }
    pub(crate) fn validates_source(&self, work: &CoordinatorWork<ReviewView, Q>) -> bool {
        self.target.matches_source(work)
            && self.input.view_seal.authority == work.frame.authority
            && self.input.view.generation() == work.input.view.generation()
            && self.input.view.root_state() == work.input.view.root_state()
            && self.input.view.record_scope_digest() == work.input.view.record_scope_digest()
            && matches!(
                (self.input.stage, self.recipient),
                (
                    RefinementStage::ProposalContinuation,
                    UtilityRouteRecipient::Proposer
                ) | (
                    RefinementStage::CriticDivergence,
                    UtilityRouteRecipient::Critic
                ) | (
                    RefinementStage::CpuValidation,
                    UtilityRouteRecipient::Researcher
                )
            )
    }
    pub(crate) fn admitted(self, frame: FrameId) -> UtilityInterventionAdmission<Q> {
        UtilityInterventionAdmission {
            target: self.target,
            input: self.input,
            recipient: self.recipient,
            frame,
            started: self.started,
        }
    }
}

/// Only the coordinator's real frame admission can issue this move-only cap.
pub struct UtilityInterventionAdmission<Q = AdaptivePortQuestion> {
    target: UtilityInterventionTarget<Q>,
    input: FrameInput<ReviewView, Q>,
    recipient: UtilityRouteRecipient,
    frame: FrameId,
    started: Instant,
}
impl<Q> UtilityInterventionAdmission<Q> {
    pub fn frame(&self) -> FrameId {
        self.frame
    }
    pub fn source_frame(&self) -> FrameId {
        self.target.source_frame()
    }
    pub fn input(&self) -> &FrameInput<ReviewView, Q> {
        &self.input
    }
    pub fn neutral_target_digest(&self) -> ContractDigest {
        self.target.digest
    }
    pub fn arm_query_digest(&self) -> ContractDigest {
        self.input.question_digest
    }
    pub fn input_capture(&self) -> &Arc<NnuecUtilityInputCapture> {
        &self.target.input
    }
    pub fn record_scope_digest(&self) -> ContractDigest {
        self.target.record_scope_digest()
    }
    pub fn first_move(&self) -> ContractMove16 {
        self.target.first_move()
    }
    pub fn budget(&self) -> FrameBudget {
        self.target.budget()
    }
    pub fn original_deadline(&self) -> Deadline {
        self.target.original_deadline()
    }
    pub fn recipient(&self) -> UtilityRouteRecipient {
        self.recipient
    }
    fn matches_work(&self, work: &CoordinatorWork<ReviewView, Q>) -> bool {
        self.frame == work.frame
            && self.first_move() == work.original_first_move
            && self.budget() == work.budget
            && self.original_deadline() == work.original_deadline
            && self.input.stage == work.input.stage
            && self.input.question_digest == work.input.question_digest
            && self.input.view_seal == work.input.view_seal
            && self.input.public_context == work.input.public_context
            && Arc::ptr_eq(&self.input.view, &work.input.view)
            && Arc::ptr_eq(&self.input.question, &work.input.question)
    }
}

pub struct UtilityInterventionBindRejection {
    pub admission: UtilityInterventionAdmission,
    pub error: PalsError,
}

pub struct UtilityInterventionTargetBindRejection {
    pub preparation: UtilityRoutePreparation,
    pub source: UtilityInterventionReadySource<AdaptivePortQuestion>,
    pub input: Arc<NnuecUtilityInputCapture>,
    pub error: PalsError,
}

pub(super) struct UtilityInterventionSource {
    source: UtilityInterventionWorkSource,
    arm: UtilityInterventionWorkSource,
    digest: ContractDigest,
    arm_request: Option<RequestIdentity>,
    arm_capture: Option<Arc<NnuecUtilityInputCapture>>,
}
/// The actual machine/worker/actor keep their strong pins until their existing
/// fences. An action stores exact weak identities, so a later pending receipt
/// cannot itself prevent the original Store from reaching pin zero.
struct UtilityInterventionWorkSource {
    frame: FrameId,
    first: ContractMove16,
    deadline: Deadline,
    budget: FrameBudget,
    stage: RefinementStage,
    question_digest: ContractDigest,
    view_seal: ReviewViewSeal,
    public_context: ContractDigest,
    question: Weak<AdaptivePortQuestion>,
    view: Weak<ReviewView>,
}
impl UtilityInterventionWorkSource {
    fn new(work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>) -> Self {
        Self {
            frame: work.frame,
            first: work.original_first_move,
            deadline: work.original_deadline,
            budget: work.budget,
            stage: work.input.stage,
            question_digest: work.input.question_digest,
            view_seal: work.input.view_seal,
            public_context: work.input.public_context,
            question: Arc::downgrade(&work.input.question),
            view: Arc::downgrade(&work.input.view),
        }
    }
    fn input(&self) -> Option<FrameInput<ReviewView, AdaptivePortQuestion>> {
        Some(FrameInput {
            stage: self.stage,
            question_digest: self.question_digest,
            question: self.question.upgrade()?,
            view_seal: self.view_seal,
            public_context: self.public_context,
            view: self.view.upgrade()?,
        })
    }
    fn work(&self) -> Option<CoordinatorWork<ReviewView, AdaptivePortQuestion>> {
        Some(CoordinatorWork {
            frame: self.frame,
            original_first_move: self.first,
            original_deadline: self.deadline,
            budget: self.budget,
            input: self.input()?,
        })
    }
    fn matches_work(&self, work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>) -> bool {
        self.frame == work.frame
            && self.first == work.original_first_move
            && self.deadline == work.original_deadline
            && self.budget == work.budget
            && self.stage == work.input.stage
            && self.question_digest == work.input.question_digest
            && self.view_seal == work.input.view_seal
            && self.public_context == work.input.public_context
            && self.view.ptr_eq(&Arc::downgrade(&work.input.view))
            && self.question.ptr_eq(&Arc::downgrade(&work.input.question))
    }
}
impl UtilityInterventionSource {
    pub(super) fn matches_arm_capture_lineage(
        &self,
        lineage: &crate::cpu_value::nnuec::v2::capture::NnuecCaptureLineage,
    ) -> bool {
        self.arm_capture
            .as_ref()
            .is_some_and(|capture| capture.matches_capture_lineage(lineage))
    }
    fn matches_arm_work(&self, work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>) -> bool {
        self.arm.matches_work(work)
    }
}

impl UtilityRouteAction {
    pub fn physical_input_captures(&self) -> impl Iterator<Item = &Arc<NnuecUtilityInputCapture>> {
        std::iter::once(&self.input).chain(self.intervention_arm_capture())
    }
    pub fn intervention_source_work(
        &self,
    ) -> Option<CoordinatorWork<ReviewView, AdaptivePortQuestion>> {
        self.intervention
            .as_ref()
            .and_then(|source| source.source.work())
    }
    pub fn intervention_arm_input(&self) -> Option<FrameInput<ReviewView, AdaptivePortQuestion>> {
        self.intervention
            .as_ref()
            .and_then(|source| source.arm.input())
    }
    pub fn intervention_neutral_target_digest(&self) -> Option<ContractDigest> {
        self.intervention.as_ref().map(|source| source.digest)
    }
    pub fn intervention_arm_capture(&self) -> Option<&Arc<NnuecUtilityInputCapture>> {
        self.intervention
            .as_ref()
            .and_then(|source| source.arm_capture.as_ref())
    }
}

impl UtilityRouteSemanticOutcome {
    pub fn physical_input_captures(&self) -> impl Iterator<Item = &Arc<NnuecUtilityInputCapture>> {
        self.action.physical_input_captures()
    }
    pub fn intervention_arm_capture(&self) -> Option<&Arc<NnuecUtilityInputCapture>> {
        self.action.intervention_arm_capture()
    }
    pub fn intervention_source_work(
        &self,
    ) -> Option<CoordinatorWork<ReviewView, AdaptivePortQuestion>> {
        self.action.intervention_source_work()
    }
    pub fn intervention_arm_input(&self) -> Option<FrameInput<ReviewView, AdaptivePortQuestion>> {
        self.action.intervention_arm_input()
    }
}

impl UtilityRouteUnresolvedAction {
    pub fn physical_input_captures(&self) -> impl Iterator<Item = &Arc<NnuecUtilityInputCapture>> {
        self.action.physical_input_captures()
    }
    pub fn intervention_arm_capture(&self) -> Option<&Arc<NnuecUtilityInputCapture>> {
        self.action.intervention_arm_capture()
    }
    pub fn intervention_source_work(
        &self,
    ) -> Option<CoordinatorWork<ReviewView, AdaptivePortQuestion>> {
        self.action.intervention_source_work()
    }
    pub fn intervention_arm_input(&self) -> Option<FrameInput<ReviewView, AdaptivePortQuestion>> {
        self.action.intervention_arm_input()
    }
}

impl<M: RoleModel> AdaptiveEnginePort<M> {
    /// Add the input-capture request to the original arm Request returned by
    /// prepare_cpu_nnuec_review. Its existing install seal remains unchanged.
    /// Only this exact Request can later bind its prepared capture.
    #[allow(clippy::result_large_err)]
    pub fn attach_utility_intervention_arm_request(
        &self,
        action: &mut UtilityRouteAction,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        request: Request,
    ) -> Result<Request, (AdaptiveNnuecPreparationError, Request)> {
        let checked = (|| -> Result<(), PalsError> {
            self.validate_utility_route_work(action, work)?;
            let source = action
                .intervention
                .as_ref()
                .ok_or(RoleError::InvalidOutput)?;
            let original = source.source.input().ok_or(RoleError::InvalidOutput)?;
            let provenance = request.provenance();
            if action.has_committed()
                || source.arm_request.is_some()
                || source.arm_capture.is_some()
                || action.preparation.id.frame != work.frame
                || action.preparation.id.question != work.input.question_digest
                || !source.matches_arm_work(work)
                || !request.matches_view(&work.input.view)
                || provenance.root != action.preparation.root
                || provenance.generation != action.preparation.generation
                || provenance.view_seal != work.input.view_seal.view_digest.0
                || provenance.full_view_digest != work.input.view.digest()
                || provenance.record_scope_digest != action.input.provenance().record_scope_digest
                || &provenance.model != action.input.model_identity()
                || self.engine.researcher.cpu_registered_value.as_ref() != Some(&provenance.model)
                || !action.input.validates(
                    &self.engine.semantic.stores,
                    action.preparation.root,
                    action.preparation.generation,
                    &original.view,
                )
            {
                return Err(RoleError::InvalidOutput.into());
            }
            Ok(())
        })();
        if let Err(error) = checked {
            return Err((error.into(), request));
        }
        let identity = request.identity();
        match request.with_utility_input_capture(action.input.target_record()) {
            Ok(request) => {
                action
                    .intervention
                    .as_mut()
                    .expect("checked intervention")
                    .arm_request = Some(identity);
                Ok(request)
            }
            Err((error, request)) => Err((error.into(), request)),
        }
    }

    /// P/C arms also keep their actual preparation terminal. This constructs a
    /// distinct real Request, without another cache prelude or actor epoch.
    pub fn prepare_utility_intervention_arm_request(
        &self,
        action: &mut UtilityRouteAction,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        model: &Model,
        cancel: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<Request, AdaptiveNnuecPreparationError> {
        self.validate_utility_route_work(action, work)?;
        if self.engine.researcher.cpu_registered_value.as_ref() != Some(model.identity()) {
            return Err(AdaptiveNnuecPreparationError::StateMismatch);
        }
        let request = Request::admit(
            model,
            Arc::clone(&work.input.view),
            &self.engine.semantic.stores,
            action.preparation.root,
            action.preparation.generation,
            work.input.question.deadline,
            cancel,
        )?;
        self.attach_utility_intervention_arm_request(action, work, request)
            .map_err(|(error, _unsubmitted_request)| error)
    }

    /// Bind only the capture from that original registered arm Request. Equal
    /// tensors from a third Request cannot authorize a researcher witness.
    #[allow(clippy::result_large_err)]
    pub fn bind_utility_intervention_arm_capture(
        &self,
        action: &mut UtilityRouteAction,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        input: Arc<NnuecUtilityInputCapture>,
    ) -> Result<(), (PalsError, Arc<NnuecUtilityInputCapture>)> {
        let checked = (|| -> Result<(), PalsError> {
            self.validate_utility_route_work(action, work)?;
            let source = action
                .intervention
                .as_ref()
                .ok_or(RoleError::InvalidOutput)?;
            let original = source.source.input().ok_or(RoleError::InvalidOutput)?;
            if action.has_committed()
                || source.arm_capture.is_some()
                || source
                    .arm_request
                    .as_ref()
                    .is_none_or(|request| !input.matches_request(request))
                || action.preparation.id.frame != work.frame
                || action.preparation.id.question != work.input.question_digest
                || !source.matches_arm_work(work)
                || !action.input.validates(
                    &self.engine.semantic.stores,
                    action.preparation.root,
                    action.preparation.generation,
                    &original.view,
                )
                || !input.validates(
                    &self.engine.semantic.stores,
                    action.preparation.root,
                    action.preparation.generation,
                    &work.input.view,
                )
                || input.target_record_ids() != action.input.target_record_ids()
                || input.input_bytes() != action.input.input_bytes()
                || input.model_identity() != action.input.model_identity()
                || utility_intervention_target_digest(&input)? != source.digest
            {
                return Err(RoleError::InvalidOutput.into());
            }
            Ok(())
        })();
        match checked {
            Ok(()) => {
                action
                    .intervention
                    .as_mut()
                    .expect("checked intervention")
                    .arm_capture = Some(input);
                Ok(())
            }
            Err(error) => Err((error, input)),
        }
    }

    /// Consume the original same-Request neutral preparation after its actor
    /// returned the actual immutable input. No NN utility forward is invoked.
    #[allow(clippy::result_large_err)]
    pub fn bind_utility_intervention_target(
        &self,
        preparation: UtilityRoutePreparation,
        source: UtilityInterventionReadySource<AdaptivePortQuestion>,
        input: Arc<NnuecUtilityInputCapture>,
    ) -> Result<UtilityInterventionTarget, UtilityInterventionTargetBindRejection> {
        let action = match self.bind_utility_route_input(preparation, source.work(), input) {
            Ok(action) => action,
            Err(rejected) => {
                return Err(UtilityInterventionTargetBindRejection {
                    preparation: rejected.preparation,
                    source,
                    input: rejected.input,
                    error: rejected.error,
                });
            }
        };
        let checked = self.check_intervention_target(&action, source.work());
        let CheckedInterventionTarget {
            proposal,
            unavailable,
            digest,
        } = match checked {
            Ok(value) => value,
            Err(error) => {
                return Err(UtilityInterventionTargetBindRejection {
                    preparation: action.preparation,
                    source,
                    input: action.input,
                    error,
                });
            }
        };
        Ok(UtilityInterventionTarget {
            preparation: action.preparation,
            input: action.input,
            source,
            digest,
            proposal,
            unavailable,
        })
    }

    fn check_intervention_target(
        &self,
        action: &UtilityRouteAction,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<CheckedInterventionTarget, PalsError> {
        let index = usize::from(action.preparation.target_record);
        let view = &work.input.view;
        let record = view.records().get(index).ok_or(RoleError::InvalidOutput)?;
        let metadata = view
            .records_metadata()
            .get(index)
            .ok_or(RoleError::InvalidOutput)?;
        if record.kind != RecordKind::Proposal
            || record.line.is_empty()
            || record.line.len() > 256
            || record.line.len() > self.engine.config.line_plies
            || metadata.anchor != 0
            || record.line.first() != Some(&action.preparation.first)
            || !view
                .anchors()
                .first()
                .is_some_and(|anchor| anchor.snapshot.same_state(action.input.root_snapshot()))
        {
            return Err(RoleError::Unsupported(
                "intervention requires an actual current-root proposal target",
            )
            .into());
        }
        let mut endpoint = Position::from_snapshot(
            action.input.root_snapshot(),
            rz_position::PositionLimits {
                max_history_positions: action
                    .input
                    .root_snapshot()
                    .known_history_len()
                    .checked_add(record.line.len())
                    // Rules terminal classification also checks one legal
                    // child with make/unmake; it is bounded scratch, not
                    // another proposal ply or a larger search allowance.
                    .and_then(|positions| positions.checked_add(1))
                    .ok_or(PalsError::Capacity)?
                    .max(1),
                ..rz_position::PositionLimits::default()
            },
        )?;
        let mut has_divergence = false;
        let mut first_terminal = false;
        let root_side = endpoint.side_to_move();
        for (ply, movement) in record.line.iter().enumerate() {
            has_divergence |= endpoint.side_to_move() != root_side
                && matches!(
                    endpoint.classify_position()?.play_status,
                    PlayStatus::Ongoing
                );
            endpoint.make_move(*movement)?;
            if ply == 0 {
                first_terminal = !matches!(
                    endpoint.classify_position()?.play_status,
                    PlayStatus::Ongoing
                );
            }
        }
        let terminal = !matches!(
            endpoint.classify_position()?.play_status,
            PlayStatus::Ongoing
        );
        let mut unavailable = [None; 3];
        let mode = action.preparation.mode;
        for recipient in [
            UtilityRouteRecipient::Proposer,
            UtilityRouteRecipient::Critic,
            UtilityRouteRecipient::Researcher,
        ] {
            if !mode.supports(recipient.search_recipient()) {
                unavailable[recipient.output_index() - 4] =
                    Some(UtilityInterventionUnavailable::UnsupportedMode);
            }
        }
        if action.preparation.budget.role_submissions_remaining == 0 {
            unavailable[0] = Some(UtilityInterventionUnavailable::MissingRoleCredit);
            unavailable[1] = Some(UtilityInterventionUnavailable::MissingRoleCredit);
        }
        // An explicit P intervention re-proposes from the original first
        // endpoint. A normal completed L-ply Proposal is still a valid target;
        // neither its neutral record nor its model input is shortened.
        if self.engine.config.line_plies <= 1 {
            unavailable[0] = Some(UtilityInterventionUnavailable::ProposalAlreadyAtLimit);
        }
        if first_terminal {
            unavailable[0] = Some(UtilityInterventionUnavailable::TerminalProposal);
        }
        if terminal {
            unavailable[2] = Some(UtilityInterventionUnavailable::TerminalProposal);
        }
        if !has_divergence {
            unavailable[1] = Some(UtilityInterventionUnavailable::NoOpponentDivergence);
        }
        if action.preparation.budget.original_cpu_credit == 0
            || self.limits.max_cpu_nodes <= self.counters.cpu_nodes
        {
            unavailable[2] = Some(UtilityInterventionUnavailable::MissingNativeCredit);
        }
        if self.engine.researcher.cpu_registered_value.as_ref()
            != Some(action.input.model_identity())
        {
            unavailable[2] = Some(UtilityInterventionUnavailable::MissingNativeResearcher);
        }
        let digest = utility_intervention_target_digest(&action.input)?;
        Ok(CheckedInterventionTarget {
            proposal: record.line.clone(),
            unavailable,
            digest,
        })
    }

    /// Mint a genuinely new policy-specific query over the original selected
    /// records. This does not manufacture an ordinary Ready offer or Repair.
    #[allow(clippy::result_large_err)]
    pub fn prepare_utility_intervention(
        &mut self,
        target: UtilityInterventionTarget,
        recipient: UtilityRouteRecipient,
    ) -> Result<UtilityInterventionCandidate, (PalsError, UtilityInterventionTarget)> {
        let started = Instant::now();
        let checked = (|| {
            self.check_work(target.source.work())?;
            if target.unsupported_reason(recipient).is_some() {
                return Err(PalsError::from(RoleError::Unsupported(
                    "frozen proposal intervention arm unavailable",
                )));
            }
            if !target.input.validates(
                &self.engine.semantic.stores,
                target.preparation.root,
                target.preparation.generation,
                &target.source.work().input.view,
            ) || self.validator_existing_witnesses()? != target.preparation.baseline
            {
                return Err(RoleError::InvalidOutput.into());
            }
            let (node, stage, prefix) = match recipient {
                UtilityRouteRecipient::Proposer => (
                    self.replay_node(&[target.preparation.first])?,
                    RefinementStage::ProposalContinuation,
                    vec![target.preparation.first],
                ),
                UtilityRouteRecipient::Critic => {
                    (self.root, RefinementStage::CriticDivergence, Vec::new())
                }
                UtilityRouteRecipient::Researcher => (
                    self.replay_node(&target.proposal)?,
                    RefinementStage::CpuValidation,
                    target.proposal.clone(),
                ),
            };
            let purpose = purpose_for_stage(stage);
            let divergences = if recipient == UtilityRouteRecipient::Critic {
                self.divergence_indices(&target.proposal)?
            } else {
                Vec::new()
            };
            let legal = if recipient == UtilityRouteRecipient::Critic {
                Vec::new()
            } else {
                self.engine.semantic.nodes[node].position.legal_moves()
            };
            let fork = self
                .engine
                .semantic
                .review_index
                .as_mut()
                .ok_or(PalsError::InvalidConfig)?
                .fork_query(
                    &self.engine.semantic.stores,
                    &target.source.work().input.view,
                    ReviewQuerySequences {
                        purpose,
                        prefix: &prefix,
                        proposal: &target.proposal,
                        counterexample: None,
                        legal: &legal,
                        divergences: &divergences,
                    },
                )?;
            self.input_with_purpose(
                node,
                stage,
                Some(target.preparation.first),
                prefix,
                target.proposal.clone(),
                None,
                None,
                None,
                false,
                Some(Arc::new(fork)),
                None,
                purpose,
            )
        })();
        match checked {
            Ok(input) => Ok(UtilityInterventionCandidate {
                target,
                input,
                recipient,
                started,
            }),
            Err(error) => Err((error, target)),
        }
    }

    #[allow(clippy::result_large_err)]
    pub fn bind_utility_intervention_route(
        &self,
        admission: UtilityInterventionAdmission,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<UtilityRouteAction, UtilityInterventionBindRejection> {
        let checked = (|| {
            self.check_work(work)?;
            AdaptivePortQuestion::validate_frontier_source(work, &self.engine.semantic.stores)?;
            let target = &admission.target;
            if !admission.matches_work(work)
                || !target
                    .preparation
                    .store
                    .matches(&self.engine.semantic.stores)
                || !target.input.validates(
                    &self.engine.semantic.stores,
                    target.preparation.root,
                    target.preparation.generation,
                    &target.source.work().input.view,
                )
                || self.validator_existing_witnesses()? != target.preparation.baseline
                || target.source.work().input.view.record_scope_digest()
                    != work.input.view.record_scope_digest()
            {
                return Err(PalsError::from(RoleError::InvalidOutput));
            }
            self.utility_route_search_scope(work)
        })();
        let search_scope = match checked {
            Ok(value) => value,
            Err(error) => return Err(UtilityInterventionBindRejection { admission, error }),
        };
        let UtilityInterventionAdmission {
            target,
            input: _arm,
            recipient,
            frame,
            started,
        } = admission;
        let UtilityInterventionTarget {
            mut preparation,
            input,
            source,
            digest,
            ..
        } = target;
        preparation.id = UtilityRouteActionId {
            owner: Arc::new(()),
            authority: self.authority,
            frame,
            question: work.input.question_digest,
            recipient,
        };
        preparation.source_view = Arc::downgrade(&work.input.view);
        preparation.source_query = Arc::downgrade(&work.input.question);
        preparation.view_seal = work.input.view_seal;
        preparation.full_view = ContractDigest(work.input.view.digest());
        preparation.first_observation = self
            .engine
            .semantic
            .stores
            .observations
            .next_observation_sequence();
        preparation.search_scope = search_scope;
        preparation.started = started;
        let mut action = new_utility_route_action(preparation, input);
        action.frozen_context = digest;
        action.intervention = Some(UtilityInterventionSource {
            source: UtilityInterventionWorkSource::new(source.work()),
            arm: UtilityInterventionWorkSource::new(work),
            digest,
            arm_request: None,
            arm_capture: None,
        });
        Ok(action)
    }
}

#[cfg(test)]
mod tests {
    // Production proposal publication with authored accepted role outputs,
    // followed by real tiny CPU NNUEC preparation. These component tests run
    // neither P/C neural inference nor an R search and
    // do not attest a manager's native role/installed-owner acceptance.
    use super::*;
    use crate::contract_time::InstantClock;
    use crate::cpu_value::nnuec::v2::review::{BoundNnuecReviewContext, PreparationActor};
    use crate::cpu_worker::CpuCreditLedger;
    use crate::pals::coordinator::{
        CoordinatorError, ProposerRankingPublished, RefinementMachine, StepEvidence,
    };
    use crate::pals::rovent_engine::RootMoveDomain;
    use crate::pals::rovent_frontier::{FrontierLimits, FrontierRegistration};
    use rz_contracts::MonotonicTick;
    use rz_contracts::rovent::{ExecutionMode, SearchGoal};
    use std::num::NonZeroU16;

    type Port = AdaptiveEnginePort<AdaptivePortModel>;
    type Work = CoordinatorWork<ReviewView, AdaptivePortQuestion>;
    type Machine = RefinementMachine<ReviewView, AdaptivePortQuestion>;

    struct SourceReplay {
        ranking: ProposerRankingPublished,
        initial: Work,
        ticket: RoleTicketEnvelope,
        completed_evidence: ContractDigest,
    }

    fn fixture() -> (
        Port,
        Machine,
        Work,
        UtilityInterventionTarget,
        Arc<BoundNnuecReviewContext>,
        PreparationActor,
        SourceReplay,
    ) {
        let value = LegalOrderRoleMock.value_identity().unwrap().clone();
        let model_key = ContractDigest(Sha256::digest(value.model.as_bytes()).into());
        let encoding_key = ContractDigest(Sha256::digest(value.encoding.as_bytes()).into());
        let namespace = AdaptiveModelNamespace::new(
            value.clone(),
            model_key,
            encoding_key,
            rz_contracts::PrecisionProfile::Fp32,
            0,
        )
        .unwrap();
        let model =
            AdaptivePortModel::new_with_namespace(value.model.clone(), Some(value), namespace)
                .unwrap();
        let mut port = super::super::super::port_tests::setup_with_model(
            Position::startpos(),
            2,
            ResolverPolicy::OwnRawRestricted,
            model,
        );
        port.engine.config.beam_width = 1;
        let clock = port.deadline.clock;
        let root_moves = Arc::new(RootMoveDomain::new(&port.rules, None).unwrap());
        let ledger = CpuCreditLedger::new(
            port.authority,
            port.limits.max_cpu_nodes,
            InstantClock::new(clock, Instant::now()),
        )
        .unwrap();
        let native_ledger = ledger.identity();
        let root_budget = ContractDigest([10; 32]);
        let registration = FrontierRegistration {
            authority: port.authority,
            mode: ExecutionMode::CpuPcr,
            original_deadline: port.deadline,
            root_budget,
            goal: SearchGoal {
                depth_plies: NonZeroU16::new(2),
                mate_moves: None,
            },
            nodes: Some(port.limits.max_cpu_nodes),
            selective_policy: ContractDigest([11; 32]),
            root_rules: Arc::clone(&port.rules),
            root_moves: Arc::clone(&root_moves),
            root_state: port.root_state_id(),
            model_namespace: Arc::new(port.model_namespace().unwrap().clone()),
            native_ledger: Some(native_ledger.clone()),
        };
        port.attach_rovent_frontier(
            registration,
            FrontierLimits {
                maximum_goal_plies: NonZeroU16::new(16).unwrap(),
                maximum_frames: 16,
                maximum_paths: 64,
                maximum_mate_nodes: 64,
                maximum_native_receipts: 16,
            },
            AdaptiveFrontierStartup {
                mode: ExecutionMode::CpuPcr,
                root_budget,
                root_moves: &root_moves,
                native_ledger: Some(&native_ledger),
            },
            clock,
            MonotonicTick(0),
        )
        .unwrap();
        port.prepare_rovent_root_rules(clock, MonotonicTick(0))
            .unwrap();
        // Follow the production make_root_candidate -> continuation ->
        // completed Proposal path. An authored short record could hide the
        // fact that ordinary proposals are published only at L/terminal.
        let (ranking, _registry) =
            super::super::super::port_tests::accepted_root_ranking(&mut port);
        let candidate = port.ranking_published(&ranking).unwrap().pop().unwrap();
        assert_eq!(candidate.input.stage, RefinementStage::ProposalContinuation);
        let mut machine = Machine::new(port.authority, port.deadline, 512).unwrap();
        let frame = machine
            .admit_candidate(
                &ranking,
                candidate.first_move,
                candidate.input.clone(),
                candidate.budget,
                clock,
                MonotonicTick(4),
            )
            .unwrap();
        port.root_candidate_admitted(frame, candidate.first_move, &candidate.input)
            .unwrap();
        let initial = machine.ready_work(frame, clock, MonotonicTick(4)).unwrap();
        let (mut ticket, mut output) = super::super::super::port_tests::output(&initial, 2);
        ticket.representation.model = model_key;
        ticket.representation.encoding = encoding_key;
        ticket.representation.frozen_epoch = 0;
        output.key = ticket.representation;
        machine
            .bind_role(frame, &ticket, clock, MonotonicTick(4))
            .unwrap();
        let running = machine
            .bound_work(frame, StepExecution::Role(ticket.id))
            .unwrap();
        let step = port.commit_role(&running, &ticket, &output, None).unwrap();
        port.role_consumed(&ticket, &running.input.view).unwrap();
        let step = port.materialize_step(step).unwrap();
        let completed_evidence = step.evidence;
        machine
            .resolve(
                StepEvidence {
                    frame,
                    stage: running.input.stage,
                    execution: StepExecution::Role(ticket.id),
                    submitted_question: running.input.question_digest,
                    submitted_view: running.input.view_seal,
                    completed_evidence: step.evidence,
                },
                step.transition,
                clock,
                MonotonicTick(4),
            )
            .unwrap();
        assert_eq!(port.counters.proposals, 1);
        let work = machine
            .ready_work(frame, port.deadline.clock, MonotonicTick(4))
            .unwrap();
        assert_eq!(work.input.stage, RefinementStage::CpuValidation);
        let line = work.input.question.proposal.clone();
        assert_eq!(
            line.len(),
            port.engine.config.line_plies,
            "ordinary completed L-ply Proposal"
        );
        let target = work
            .input
            .view
            .records()
            .iter()
            .position(|record| record.kind == RecordKind::Proposal && record.line == line)
            .expect("original selected proposal") as u16;
        let mut preparation = port.prepare_utility_route(&work, target).unwrap();
        let source = crate::pals::engine::preparing_root::tests::source();
        let model = Arc::clone(source.model());
        let request = Request::admit(
            &model,
            Arc::clone(&work.input.view),
            &port.engine.semantic.stores,
            port.root_state_id(),
            port.authority.root.0,
            port.limits.deadline,
            Arc::new(std::sync::atomic::AtomicBool::new(false)),
        )
        .unwrap();
        let request = preparation
            .attach_input_capture(request)
            .unwrap_or_else(|_| panic!("same original input request"));
        let actor = PreparationActor::spawn(model).unwrap();
        actor
            .try_submit(request)
            .unwrap_or_else(|_| panic!("one actual preparation"));
        let until = Instant::now() + Duration::from_secs(5);
        let bound = loop {
            if let Some(completion) = actor.try_complete().unwrap() {
                break Arc::new(completion.result.unwrap());
            }
            assert!(Instant::now() < until, "tiny original CPU preparation");
            std::thread::yield_now();
        };
        let input = Arc::clone(bound.utility_input_capture().unwrap());
        let source = machine
            .capture_intervention_source(frame, port.deadline.clock, MonotonicTick(0))
            .unwrap();
        let target = port
            .bind_utility_intervention_target(preparation, source, input)
            .unwrap_or_else(|rejected| panic!("actual same-Request target: {:?}", rejected.error));
        (
            port,
            machine,
            work,
            target,
            bound,
            actor,
            SourceReplay {
                ranking,
                initial,
                ticket,
                completed_evidence,
            },
        )
    }

    #[test]
    fn historical_target_codec_matches_actual_capture_without_new_authority_or_pins() {
        let (port, mut machine, work, target, _bound, actor, _replay) = fixture();
        let capture = target.input_capture();
        let before_pins = port.engine.semantic.stores.review_pin_count().unwrap();
        let metadata = utility_intervention_target_digest_preimage(capture).unwrap();
        assert_eq!(
            metadata.root_snapshot_digest,
            utility_intervention_historical_snapshot_digest(capture.root_snapshot()).unwrap()
        );
        let content = || UtilityInterventionHistoricalTargetContent {
            root_snapshot: capture.root_snapshot(),
            record_scope_digest: metadata.record_scope_digest,
            target_record: capture.target_record(),
            original_first_move: work.original_first_move,
            reference_json: capture.input_bytes(),
            model_identity: capture.model_identity(),
            source_snapshots: capture.source_snapshots(),
        };
        let historical = utility_intervention_historical_target_digest(content()).unwrap();
        assert_eq!(historical, target.neutral_target_digest());
        assert_eq!(
            historical,
            utility_intervention_target_digest(capture).unwrap()
        );
        let mut changed = content();
        changed.target_record += 1;
        assert_ne!(
            utility_intervention_historical_target_digest(changed).unwrap(),
            historical
        );
        let mut changed_bytes = capture.input_bytes().to_vec();
        changed_bytes.push(b' ');
        let mut changed = content();
        changed.reference_json = &changed_bytes;
        assert_ne!(
            utility_intervention_historical_target_digest(changed).unwrap(),
            historical,
            "raw reference bytes are never silently canonicalized"
        );
        assert_eq!(
            port.engine.semantic.stores.review_pin_count().unwrap(),
            before_pins
        );
        assert_eq!(machine.frames().count(), 1);
        assert_eq!(machine.intervention_source_count(), 0);
        machine.revoke();
        actor.shutdown().unwrap();
        machine.retired_after_worker_join();
    }

    #[test]
    fn intervention_preserves_neutral_scope_and_issues_distinct_actual_p_c_queries() {
        let mut scopes = Vec::new();
        for recipient in [
            UtilityRouteRecipient::Proposer,
            UtilityRouteRecipient::Critic,
        ] {
            let (mut port, mut machine, work, target, bound, actor, _replay) = fixture();
            let full_proposal = work.input.question.proposal.clone();
            assert_eq!(
                target.supported_recipients().collect::<Vec<_>>(),
                [
                    UtilityRouteRecipient::Proposer,
                    UtilityRouteRecipient::Critic
                ]
            );
            assert_eq!(
                target.unsupported_reason(UtilityRouteRecipient::Researcher),
                Some(UtilityInterventionUnavailable::MissingNativeResearcher)
            );
            let neutral = target.neutral_target_digest();
            let candidate = port
                .prepare_utility_intervention(target, recipient)
                .unwrap_or_else(|_| panic!("supported arm"));
            assert_eq!(candidate.budget(), work.budget);
            assert_eq!(candidate.original_deadline(), work.original_deadline);
            assert!(candidate.validates_source(&work));
            {
                // This never-submitted tamper copy tests the budget guard;
                // its fixture-only Arc owners must not outlive that check.
                // Actual machine/Bound/actor custody is unchanged below.
                let mut changed = work.clone();
                changed.budget.role_submissions_remaining += 1;
                assert!(!candidate.validates_source(&changed));
            }
            let (frame, admission) = machine
                .admit_intervention(candidate, port.deadline.clock, MonotonicTick(0))
                .unwrap_or_else(|_| panic!("actual source replacement"));
            assert_ne!(frame, work.frame);
            assert_eq!(machine.intervention_source_count(), 1);
            assert!(matches!(
                machine.ready_work(work.frame, port.deadline.clock, MonotonicTick(0)),
                Err(CoordinatorError::UnknownRequest)
            ));
            let arm = machine
                .ready_work(frame, port.deadline.clock, MonotonicTick(0))
                .unwrap();
            assert_eq!(arm.budget, work.budget);
            assert_eq!(
                arm.input.view.record_scope_digest(),
                work.input.view.record_scope_digest()
            );
            let preimage = utility_intervention_arm_question_preimage(&arm).unwrap();
            assert_eq!(preimage.digest(), arm.input.question_digest);
            assert_ne!(arm.input.question_digest, work.input.question_digest);
            assert_eq!(arm.input.view.query().proposal, full_proposal);
            assert_eq!(
                full_proposal.len(),
                port.engine.config.line_plies,
                "normal completed Proposal remains the full neutral target"
            );
            assert_eq!(
                arm.input.view.query().prefix,
                if recipient == UtilityRouteRecipient::Proposer {
                    vec![full_proposal[0]]
                } else {
                    Vec::new()
                },
                "only P restarts its bounded proposal from the original first endpoint"
            );
            assert_eq!(
                arm.input.view.query().counterexample,
                None,
                "no historical Repair synthesized"
            );
            let action = port
                .bind_utility_intervention_route(admission, &arm)
                .unwrap_or_else(|_| panic!("actual admitted arm binding"));
            assert_eq!(action.intervention_neutral_target_digest(), Some(neutral));
            assert_eq!(action.physical_input_captures().count(), 1);
            assert!(action.input().matches_bound(&bound));
            scopes.push((neutral, arm.input.question_digest, arm.input.stage));
            let source_view = Arc::downgrade(&work.input.view);
            let arm_view = Arc::downgrade(&arm.input.view);
            let historical = action.into_unresolved(UtilityRouteUnresolvedReason::PhysicalUnknown);
            drop(work);
            drop(arm);
            drop(bound);
            actor.shutdown().unwrap();
            // Cancellation is not a physical fence and must retain the source.
            machine.revoke();
            assert_eq!(machine.intervention_source_count(), 1);
            assert!(machine.frame(frame).is_some());
            assert!(source_view.upgrade().is_some());
            assert!(arm_view.upgrade().is_some());
            assert_eq!(historical.physical_input_captures().count(), 1);
            machine.retired_after_worker_join();
            assert_eq!(machine.intervention_source_count(), 0);
            assert!(source_view.upgrade().is_none());
            assert!(arm_view.upgrade().is_none());
            assert!(historical.intervention_source_work().is_none());
            assert!(historical.intervention_arm_input().is_none());
            assert_eq!(historical.physical_input_captures().count(), 1);
        }
        assert_eq!(scopes[0].0, scopes[1].0);
        assert_ne!(scopes[0].1, scopes[1].1);
        assert_eq!(scopes[0].2, RefinementStage::ProposalContinuation);
        assert_eq!(scopes[1].2, RefinementStage::CriticDivergence);
    }

    #[test]
    fn intervention_admission_closed_returns_original_candidate_and_retains_source() {
        let (mut port, mut machine, work, target, _bound, actor, _replay) = fixture();
        let candidate = port
            .prepare_utility_intervention(target, UtilityRouteRecipient::Critic)
            .unwrap_or_else(|_| panic!("actual C candidate"));
        let question = candidate.arm_query_digest();
        machine.revoke();
        let (error, candidate) = machine
            .admit_intervention(candidate, port.deadline.clock, MonotonicTick(0))
            .err()
            .expect("closed admission");
        assert_eq!(error, CoordinatorError::AdmissionClosed);
        assert_eq!(candidate.arm_query_digest(), question);
        assert_eq!(candidate.source_frame(), work.frame);
        assert!(machine.frame(work.frame).is_some());
        assert_eq!(machine.intervention_source_count(), 0);
        machine.retired_after_worker_join();
        actor.shutdown().unwrap();
    }

    #[test]
    fn intervention_rejects_another_machine_even_with_the_same_work_arcs_and_authority() {
        let (mut port, mut original, work, target, _bound, actor, replay) = fixture();
        let candidate = port
            .prepare_utility_intervention(target, UtilityRouteRecipient::Critic)
            .unwrap_or_else(|_| panic!("actual candidate"));
        let mut foreign = Machine::new(port.authority, port.deadline, 512).unwrap();
        let frame = foreign
            .admit_candidate(
                &replay.ranking,
                replay.initial.original_first_move,
                replay.initial.input.clone(),
                replay.initial.budget,
                port.deadline.clock,
                MonotonicTick(4),
            )
            .unwrap();
        foreign
            .bind_role(frame, &replay.ticket, port.deadline.clock, MonotonicTick(4))
            .unwrap();
        foreign
            .resolve(
                StepEvidence {
                    frame,
                    stage: replay.initial.input.stage,
                    execution: StepExecution::Role(replay.ticket.id),
                    submitted_question: replay.initial.input.question_digest,
                    submitted_view: replay.initial.input.view_seal,
                    completed_evidence: replay.completed_evidence,
                },
                FrameTransition::Continue(work.input.clone()),
                port.deadline.clock,
                MonotonicTick(4),
            )
            .unwrap();
        assert_eq!(
            frame, work.frame,
            "equal public frame metadata is insufficient"
        );
        let _foreign_cap = foreign
            .capture_intervention_source(frame, port.deadline.clock, MonotonicTick(0))
            .unwrap();
        let (error, candidate) = foreign
            .admit_intervention(candidate, port.deadline.clock, MonotonicTick(0))
            .err()
            .expect("different private machine owner");
        assert!(matches!(error, CoordinatorError::IdentityMismatch(_)));
        assert_eq!(foreign.intervention_source_count(), 0);
        assert!(foreign.frame(work.frame).is_some());
        original
            .admit_intervention(candidate, port.deadline.clock, MonotonicTick(0))
            .unwrap_or_else(|_| panic!("the original move-only cap remains usable"));
        assert!(
            original
                .ready_work(work.frame, port.deadline.clock, MonotonicTick(0))
                .is_err()
        );
        assert!(
            original
                .capture_intervention_source(work.frame, port.deadline.clock, MonotonicTick(0))
                .is_err()
        );
        original.revoke();
        foreign.revoke();
        actor.shutdown().unwrap();
        original.retired_after_worker_join();
        foreign.retired_after_worker_join();
    }

    #[test]
    fn intervention_expiry_keeps_original_cap_and_does_not_issue_a_frame() {
        let (mut port, mut machine, work, target, _bound, actor, _replay) = fixture();
        let candidate = port
            .prepare_utility_intervention(target, UtilityRouteRecipient::Critic)
            .unwrap_or_else(|_| panic!("actual C candidate"));
        let question = candidate.arm_query_digest();
        let (_, candidate) = machine
            .admit_intervention(candidate, port.deadline.clock, work.original_deadline.at)
            .err()
            .expect("original deadline enforced");
        assert_eq!(candidate.arm_query_digest(), question);
        assert_eq!(candidate.source_frame(), work.frame);
        assert_eq!(machine.intervention_source_count(), 0);
        assert_eq!(machine.frames().count(), 1);
        machine.revoke();
        actor.shutdown().unwrap();
        machine.retired_after_worker_join();
    }

    #[test]
    fn arm_hash_projection_rejects_changed_stage_and_uses_original_query_sequence() {
        let mut port = super::super::super::port_tests::setup(Position::startpos(), 4);
        let mut work = super::super::super::port_tests::work(
            &mut port,
            "e2e4",
            RefinementStage::ProposalContinuation,
            91,
        );
        let preimage = utility_intervention_arm_question_preimage(&work).unwrap();
        assert_ne!(preimage.sequence, work.frame.sequence);
        assert_eq!(preimage.digest(), work.input.question_digest);
        work.input.stage = RefinementStage::Repair;
        assert!(utility_intervention_arm_question_preimage(&work).is_err());
    }
}
