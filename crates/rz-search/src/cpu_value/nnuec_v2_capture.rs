//! Opt-in causal collection of actual native-R NNUEC readouts.
//!
//! A semantic owner reserves the original question/Bound before dispatch. The
//! actual CPU owner then binds its Job and immutable anchor. Only the existing
//! score call may publish a witness; these types never run another forward or
//! assign a training target. Numeric research inputs cannot construct them.

use super::review::BoundNnuecReviewContext;
use super::{Model, ReferenceInput};
use crate::contract_time::InstantClock;
use crate::cpu_value::nnuec::{NnuecLocal, NnuecOutput};
use crate::pals::coordinator::{CoordinatorWork, FrameId};
use crate::pals::engine::adaptive_port::AdaptivePortQuestion;
use crate::pals::review_context::ReviewView;
use crate::pals::store::PalsStores;
use rz_contracts::pals_adaptive::{CpuJobEnvelope, CpuWorkerId};
use rz_contracts::{Deadline, Digest, MonotonicTick};
use rz_position::{Position, PositionLimits, PositionSnapshot};
use sha2::{Digest as _, Sha256};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

#[path = "nnuec_v2_capture_lineage.rs"]
mod lineage;
pub use lineage::NnuecCaptureLineage;

const INPUT_DOMAIN: &[u8] = b"rz-nnuec-actual-r-readout-input/1\0";
const MAX_CAPTURE_RECORDS: u16 = 64;
const MAX_CAPTURE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CaptureError {
    Invalid(&'static str),
    Stale,
    Duplicate,
    Busy,
    Capacity,
    Poisoned,
    Rules,
    Input,
    Clock,
}

/// Required collection settings. No collection runs in the default CPU path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NnuecCaptureLimits {
    maximum_readouts: u16,
    every_nth_score: u32,
    maximum_input_bytes: usize,
}
impl NnuecCaptureLimits {
    pub fn maximum_readouts(&self) -> u16 {
        self.maximum_readouts
    }
    pub fn every_nth_score(&self) -> u32 {
        self.every_nth_score
    }
    /// Retained serialized numeric input bytes, not total/peak heap memory.
    pub fn maximum_input_bytes(&self) -> usize {
        self.maximum_input_bytes
    }
    pub fn new(
        maximum_readouts: u16,
        every_nth_score: u32,
        maximum_input_bytes: usize,
    ) -> Result<Self, CaptureError> {
        if !(1..=MAX_CAPTURE_RECORDS).contains(&maximum_readouts)
            || every_nth_score == 0
            || !(1..=MAX_CAPTURE_BYTES).contains(&maximum_input_bytes)
        {
            return Err(CaptureError::Invalid(
                "bounded readout sampling declaration",
            ));
        }
        Ok(Self {
            maximum_readouts,
            every_nth_score,
            maximum_input_bytes,
        })
    }
}

struct Submission {
    lineage: NnuecCaptureLineage,
    model: Arc<Model>,
    bound: Arc<BoundNnuecReviewContext>,
    frame: FrameId,
    question: Digest,
    capture_sequence: u64,
    limits: NnuecCaptureLimits,
    binding: Digest,
    original_deadline: Deadline,
    context_sha256: String,
    question_anchor: PositionSnapshot,
}
#[derive(Default)]
struct State {
    prepared_job: Option<(CpuJobEnvelope, CpuWorkerId)>,
    job: Option<(CpuJobEnvelope, CpuWorkerId)>,
    clock_origin: Option<Instant>,
    attempted: u64,
    selected: u64,
    overflow: u64,
    input_bytes: usize,
    preparation_time: Duration,
    error: Option<CaptureError>,
    readouts: Vec<NnuecCapturedReadout>,
    retired: bool,
}

/// Opaque original-submission custody; Clone shares one bounded observation
/// owner and cannot refill its limit or manufacture a second reservation.
#[derive(Clone)]
pub struct NnuecCaptureReservation {
    submission: Arc<Submission>,
    state: Arc<Mutex<State>>,
    retirement: Arc<OnceLock<NnuecCaptureRetirement>>,
}
impl NnuecCaptureReservation {
    /// Called only from the checked semantic-port factory, after its private
    /// current-root/Work validation. Public metadata alone is not a factory.
    pub(crate) fn reserve(
        model: Arc<Model>,
        bound: Arc<BoundNnuecReviewContext>,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
        stores: &PalsStores,
        capture_sequence: u64,
        limits: NnuecCaptureLimits,
    ) -> Result<Self, CaptureError> {
        AdaptivePortQuestion::validate_frontier_source(work, stores)
            .map_err(|_| CaptureError::Stale)?;
        let provenance = bound.provenance();
        if capture_sequence == 0
            || work.input.stage.role().is_some()
            || provenance.model != *model.identity()
            || provenance.generation != work.frame.authority.root.0
            || provenance.view_seal != work.input.view.seal_digest()
            || provenance.full_view_digest != work.input.view.digest()
            || !Arc::ptr_eq(bound.view(), &work.input.view)
            || !bound.validates(
                stores,
                provenance.root,
                provenance.generation,
                provenance.view_seal,
            )
        {
            return Err(CaptureError::Stale);
        }
        let binding = super::cpu::binding_digest(model.identity(), provenance)
            .map_err(|_| CaptureError::Input)?;
        let context_sha256 = bound.numeric_context_sha256().to_owned();
        Ok(Self {
            submission: Arc::new(Submission {
                lineage: NnuecCaptureLineage::new(
                    bound.instance_authority().clone(),
                    bound.request_identity(),
                    work.frame,
                    work.input.question_digest,
                    capture_sequence,
                    work.original_deadline,
                    binding,
                    Digest(work.input.view.digest()),
                    Digest(work.input.view.seal_digest()),
                ),
                model,
                bound,
                frame: work.frame,
                question: work.input.question_digest,
                capture_sequence,
                limits,
                binding,
                original_deadline: work.original_deadline,
                context_sha256,
                question_anchor: work.input.question.position.snapshot(),
            }),
            state: Arc::new(Mutex::new(State::default())),
            retirement: Arc::new(OnceLock::new()),
        })
    }

    pub fn lineage(&self) -> NnuecCaptureLineage {
        self.submission.lineage.clone()
    }
    pub fn matches_lineage(&self, lineage: &NnuecCaptureLineage) -> bool {
        self.submission.lineage.same_authority(lineage)
    }
    /// Original loaded-model instance readback, including before a Job exists.
    /// This grants no submission, worker activation, terminal or target right.
    pub fn matches_model(&self, model: &Arc<Model>) -> bool {
        Arc::ptr_eq(&self.submission.model, model)
    }

    /// Attach only to a prepared immutable Job. This reserves no native nodes
    /// and does not claim that the Job entered a worker or produced a value.
    pub(crate) fn prepare_job(
        &self,
        job: CpuJobEnvelope,
        owner: CpuWorkerId,
        anchor: &Position,
        clock: InstantClock,
    ) -> Result<(), CaptureError> {
        self.check_job(job, anchor, clock, self.submission.binding)?;
        let mut state = self.state.lock().map_err(|_| CaptureError::Poisoned)?;
        if state.retired {
            return Err(CaptureError::Stale);
        }
        match state.prepared_job {
            Some(original) if original != (job, owner) => return Err(CaptureError::Duplicate),
            Some(_) if state.clock_origin != Some(clock.origin()) => {
                return Err(CaptureError::Stale);
            }
            Some(_) => return Err(CaptureError::Duplicate),
            None => {
                state.prepared_job = Some((job, owner));
                state.clock_origin = Some(clock.origin());
            }
        }
        Ok(())
    }
    fn check_job(
        &self,
        job: CpuJobEnvelope,
        anchor: &Position,
        clock: InstantClock,
        actual_value_binding: Digest,
    ) -> Result<(), CaptureError> {
        if !self.submission.bound.authority_is_live()
            || job.id.authority != self.submission.frame.authority
            || job.question != self.submission.question
            || job.evaluation_namespace != actual_value_binding
            || actual_value_binding != self.submission.binding
            || job.state_identity.semantic
                != rz_position::contracts::rules_snapshot_semantic_digest(&anchor.snapshot())
            || job.state_identity.revision.0 != anchor.snapshot().revision()
            || !self
                .submission
                .question_anchor
                .same_state(&anchor.snapshot())
            || job.hard_deadline.clock != clock.domain()
            || job.hard_deadline.clock != self.submission.original_deadline.clock
            || job.hard_deadline.at > self.submission.original_deadline.at
        {
            return Err(CaptureError::Stale);
        }
        Ok(())
    }
    /// Only the actual worker invokes this after accepting the original Job.
    /// Binding is immutable across slices; retrying that exact binding does
    /// not refill sample credit. A foreign Job cannot consume the reservation.
    pub(crate) fn bind_job(
        &self,
        job: CpuJobEnvelope,
        owner: CpuWorkerId,
        anchor: &Position,
        clock: InstantClock,
        actual_value_binding: Digest,
    ) -> Result<NnuecActiveCapture, CaptureError> {
        self.check_job(job, anchor, clock, actual_value_binding)?;
        let mut state = self.state.lock().map_err(|_| CaptureError::Poisoned)?;
        if state.retired
            || state.prepared_job != Some((job, owner))
            || state.clock_origin != Some(clock.origin())
        {
            return Err(CaptureError::Stale);
        }
        match state.job {
            Some(original) if original != (job, owner) => return Err(CaptureError::Duplicate),
            Some(_) if state.clock_origin != Some(clock.origin()) => {
                return Err(CaptureError::Stale);
            }
            Some(_) => self.submission.lineage.activate(job, owner)?,
            None => {
                self.submission.lineage.activate(job, owner)?;
                state.job = Some((job, owner));
                state.clock_origin = Some(clock.origin());
            }
        }
        Ok(NnuecActiveCapture {
            reservation: self.clone(),
            job,
            owner,
            anchor: anchor.snapshot(),
            clock,
        })
    }
    pub fn retirement(&self) -> Option<&NnuecCaptureRetirement> {
        self.retirement.get()
    }
    /// The same OnceLock/submission owns this terminal, including a definite
    /// unqueued NotStarted. This does not assert worker activation or grant a
    /// semantic witness; those still require matches_lineage after bind_job.
    pub fn matches_retirement(&self, retirement: &NnuecCaptureRetirement) -> bool {
        self.retirement.get().is_some_and(|original| {
            std::ptr::eq(original, retirement)
                && Arc::ptr_eq(&self.submission, &retirement.submission)
                && original.job == retirement.job
                && original.owner == retirement.owner
        })
    }
    pub fn error(&self) -> Result<Option<CaptureError>, CaptureError> {
        self.state
            .try_lock()
            .map(|state| state.error.clone())
            .map_err(|error| match error {
                std::sync::TryLockError::WouldBlock => CaptureError::Busy,
                std::sync::TryLockError::Poisoned(_) => CaptureError::Poisoned,
            })
    }
    pub(crate) fn record_error(&self, error: CaptureError) {
        if let Ok(mut state) = self.state.lock()
            && !state.retired
        {
            state.error.get_or_insert(error);
        }
    }
    /// Actual worker terminal only. A Yield/report/progress is not retirement.
    pub(crate) fn retire(
        &self,
        job: CpuJobEnvelope,
        owner: CpuWorkerId,
        ending: NnuecCaptureEnding,
    ) -> Result<(), CaptureError> {
        let mut state = self.state.lock().map_err(|_| CaptureError::Poisoned)?;
        if state.retired
            || state.prepared_job != Some((job, owner))
            || (state.job != Some((job, owner)) && ending != NnuecCaptureEnding::NotStarted)
            || (ending == NnuecCaptureEnding::NotStarted && state.attempted != 0)
        {
            return Err(CaptureError::Duplicate);
        }
        state.retired = true;
        let receipt = NnuecCaptureRetirement {
            submission: Arc::clone(&self.submission),
            job,
            owner,
            ending,
            readouts: std::mem::take(&mut state.readouts),
            attempted: state.attempted,
            selected: state.selected,
            overflow: state.overflow,
            input_bytes: state.input_bytes,
            preparation_time: state.preparation_time,
            error: state.error.clone(),
        };
        self.retirement
            .set(receipt)
            .map_err(|_| CaptureError::Duplicate)
    }
}

/// Immutable per-Job scope carried explicitly by CpuSliceControl. No TLS,
/// global latest-context slot, native RequestId fabrication or shared budget.
#[derive(Clone)]
pub struct NnuecActiveCapture {
    reservation: NnuecCaptureReservation,
    job: CpuJobEnvelope,
    owner: CpuWorkerId,
    anchor: PositionSnapshot,
    clock: InstantClock,
}
impl NnuecActiveCapture {
    pub(crate) fn unsupported(&self) {
        if let Ok(mut state) = self.reservation.state.lock() {
            state.error = Some(CaptureError::Invalid(
                "evaluator has no causal NNUEC readout support",
            ));
        }
    }
    pub(crate) fn matches_slice(
        &self,
        job: rz_contracts::pals_adaptive::CpuJobId,
        owner: CpuWorkerId,
    ) -> bool {
        self.job.id == job && self.owner == owner
    }
    /// Consumes only the already-computed value. Failure is retained in the
    /// collection receipt, never replaced by fabricated utility/WDL labels.
    pub(super) fn observe(
        &self,
        model: &Arc<Model>,
        bound: &Arc<BoundNnuecReviewContext>,
        local: &NnuecLocal,
        position: &Position,
        numeric: &NnuecOutput,
    ) {
        let started = Instant::now();
        let Ok(mut state) = self.reservation.state.lock() else {
            return;
        };
        if state.retired || state.error.is_some() {
            return;
        }
        let attempt = state.attempted;
        let Some(next) = attempt.checked_add(1) else {
            state.error = Some(CaptureError::Capacity);
            return;
        };
        state.attempted = next;
        if attempt % u64::from(self.reservation.submission.limits.every_nth_score) != 0 {
            return;
        }
        if state.readouts.len() >= usize::from(self.reservation.submission.limits.maximum_readouts)
        {
            state.overflow = state.overflow.saturating_add(1);
            return;
        }
        state.selected = state.selected.saturating_add(1);
        let result = self.capture_one(model, bound, local, position, numeric, attempt, started);
        state.preparation_time = state.preparation_time.saturating_add(started.elapsed());
        match result {
            Ok(readout) => {
                let next_bytes = state.input_bytes.checked_add(readout.input_bytes);
                if next_bytes
                    .is_none_or(|n| n > self.reservation.submission.limits.maximum_input_bytes)
                {
                    state.overflow = state.overflow.saturating_add(1);
                    state.error = Some(CaptureError::Capacity);
                } else {
                    state.input_bytes = next_bytes.expect("bounded readout input bytes");
                    state.readouts.push(readout);
                }
            }
            Err(error) => {
                if error == CaptureError::Capacity {
                    state.overflow = state.overflow.saturating_add(1);
                }
                state.error = Some(error);
            }
        }
    }
    // Keep the original model, Bound, input and observation identities explicit.
    #[allow(clippy::too_many_arguments)]
    fn capture_one(
        &self,
        model: &Arc<Model>,
        bound: &Arc<BoundNnuecReviewContext>,
        local: &NnuecLocal,
        position: &Position,
        numeric: &NnuecOutput,
        readout_sequence: u64,
        observed_instant: Instant,
    ) -> Result<NnuecCapturedReadout, CaptureError> {
        let original = &self.reservation.submission;
        if !Arc::ptr_eq(model, &original.model)
            || !Arc::ptr_eq(bound, &original.bound)
            || !bound.authority_is_live()
            || local.anchor != self.anchor.position_identity()
        {
            return Err(CaptureError::Stale);
        }
        let config = &model.base.checkpoint.config;
        if numeric.schema != super::super::NNUEC_OUTPUT_SCHEMA
            || Some(&numeric.weights_sha256) != model.base.identity().weights_sha256.as_ref()
            || numeric
                .board_accumulators
                .iter()
                .any(|v| v.len() != config.board_width)
            || numeric.context_accumulator.len() != config.context_width
            || numeric.local_accumulator.len() != config.local_width
            || numeric.hidden.len() != config.hidden_width
            || !numeric
                .board_accumulators
                .iter()
                .flatten()
                .chain(&numeric.context_accumulator)
                .chain(&numeric.local_accumulator)
                .chain(&numeric.hidden)
                .chain(&numeric.raw_outputs)
                .chain(&numeric.wdl_logits)
                .chain(&numeric.wdl)
                .chain(std::iter::once(&numeric.value))
                .all(|v| v.is_finite())
        {
            return Err(CaptureError::Input);
        }
        // Reuse the actual Begin snapshot. Only selected observations replay
        // L; capture-off score calls allocate/copy no snapshot or history.
        let mut replay = Position::from_snapshot(&self.anchor, PositionLimits::default())
            .map_err(|_| CaptureError::Rules)?;
        for movement in local.moves() {
            replay
                .make_move(*movement)
                .map_err(|_| CaptureError::Rules)?;
        }
        if replay.position_identity() != position.position_identity() {
            return Err(CaptureError::Rules);
        }
        let input = model
            .reference_input(bound.context(), local, position)
            .map_err(|_| CaptureError::Input)?;
        let bytes = serde_json::to_vec(&input).map_err(|_| CaptureError::Input)?;
        if bytes.len() > original.limits.maximum_input_bytes {
            return Err(CaptureError::Capacity);
        }
        let mut hash = Sha256::new();
        hash.update(INPUT_DOMAIN);
        hash.update(&bytes);
        let input_sha256 = Digest(hash.finalize().into());
        let observed_at = self
            .clock
            .tick_at(observed_instant)
            .map_err(|_| CaptureError::Clock)?;
        self.job
            .hard_deadline
            .accepts(self.clock.domain(), observed_at)
            .map_err(|_| CaptureError::Stale)?;
        if !bound.authority_is_live() {
            return Err(CaptureError::Stale);
        }
        // Exactly the metadata boundary used by V2 Model::forward. The same
        // numeric values were already computed once with the admitted V2 C.
        // This does not convert/relabel an old V1 asset or execute a forward.
        let mut published_numeric = numeric.clone();
        published_numeric.schema = super::OUTPUT_SCHEMA.into();
        published_numeric.weights_sha256 = model
            .identity()
            .weights_sha256
            .clone()
            .ok_or(CaptureError::Input)?;
        Ok(NnuecCapturedReadout {
            submission: Arc::clone(original),
            job: self.job,
            owner: self.owner,
            readout_sequence,
            observed_at,
            input,
            input_sha256,
            input_bytes: bytes.len(),
            numeric: published_numeric,
            anchor: self.anchor.clone(),
            current: position.snapshot(),
            context_anchors: bound
                .view()
                .anchors()
                .iter()
                .map(|a| a.snapshot.clone())
                .collect(),
        })
    }
}

/// No public constructor/Deserialize: only an actual score with a bound Job
/// scope can issue this input/output observation. It contains no target.
pub struct NnuecCapturedReadout {
    submission: Arc<Submission>,
    job: CpuJobEnvelope,
    owner: CpuWorkerId,
    readout_sequence: u64,
    observed_at: MonotonicTick,
    input: ReferenceInput,
    input_sha256: Digest,
    input_bytes: usize,
    numeric: NnuecOutput,
    anchor: PositionSnapshot,
    current: PositionSnapshot,
    context_anchors: Vec<PositionSnapshot>,
}
impl NnuecCapturedReadout {
    pub fn matches_model(&self, model: &Arc<Model>) -> bool {
        Arc::ptr_eq(&self.submission.model, model)
    }
    pub fn context_source_snapshots(&self) -> &[PositionSnapshot] {
        self.submission.bound.context_source_snapshots()
    }
    pub fn job(&self) -> CpuJobEnvelope {
        self.job
    }
    pub fn owner(&self) -> CpuWorkerId {
        self.owner
    }
    pub fn frame(&self) -> FrameId {
        self.submission.frame
    }
    pub fn question_digest(&self) -> Digest {
        self.submission.question
    }
    pub fn capture_sequence(&self) -> u64 {
        self.submission.capture_sequence
    }
    pub fn readout_sequence(&self) -> u64 {
        self.readout_sequence
    }
    pub fn observed_at(&self) -> MonotonicTick {
        self.observed_at
    }
    pub fn input(&self) -> &ReferenceInput {
        &self.input
    }
    pub fn input_sha256(&self) -> Digest {
        self.input_sha256
    }
    pub fn numeric(&self) -> &NnuecOutput {
        &self.numeric
    }
    pub fn local_anchor(&self) -> &PositionSnapshot {
        &self.anchor
    }
    pub fn current_rules(&self) -> &PositionSnapshot {
        &self.current
    }
    pub fn context_anchors(&self) -> &[PositionSnapshot] {
        &self.context_anchors
    }
    pub fn provenance(&self) -> &super::review::Provenance {
        self.submission.bound.provenance()
    }
    pub fn context_sha256(&self) -> &str {
        &self.submission.context_sha256
    }
    /// The original immutable selected view, including local record IDs,
    /// relation metadata, anchors and checked rebase offsets. No new Store
    /// lookup or eligible-view consumption is performed here.
    pub fn context_view(&self) -> &Arc<ReviewView> {
        self.submission.bound.view()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NnuecCaptureEnding {
    CpuTerminal(crate::cpu::CpuCompletion),
    NotStarted,
    Failed,
    Unknown,
}

/// Raw, sealed terminal observation, not semantic acceptance or a WDL label.
/// The collector must join this same Job/Q to its original accepted retirement.
pub struct NnuecCaptureRetirement {
    submission: Arc<Submission>,
    job: CpuJobEnvelope,
    owner: CpuWorkerId,
    ending: NnuecCaptureEnding,
    readouts: Vec<NnuecCapturedReadout>,
    attempted: u64,
    selected: u64,
    overflow: u64,
    input_bytes: usize,
    preparation_time: Duration,
    error: Option<CaptureError>,
}
impl NnuecCaptureRetirement {
    /// Historical original-capture identity only. In particular, NotStarted
    /// keeps an unactivated lineage and cannot issue a semantic witness.
    pub fn lineage(&self) -> NnuecCaptureLineage {
        self.submission.lineage.clone()
    }
    /// Exact original activation, never public metadata equality. A known
    /// unstarted terminal supplies no actual worker/readout semantic witness.
    pub fn matches_lineage(&self, lineage: &NnuecCaptureLineage) -> bool {
        self.ending != NnuecCaptureEnding::NotStarted
            && self.submission.lineage.same_authority(lineage)
            && lineage.matches_activated_job(self.job, self.owner)
    }
    pub fn matches_model(&self, model: &Arc<Model>) -> bool {
        Arc::ptr_eq(&self.submission.model, model)
    }
    pub fn context_source_snapshots(&self) -> &[PositionSnapshot] {
        self.submission.bound.context_source_snapshots()
    }
    pub fn model_identity(&self) -> &crate::cpu_value::CpuValueIdentity {
        self.submission.model.identity()
    }
    pub fn provenance(&self) -> &super::review::Provenance {
        self.submission.bound.provenance()
    }
    pub fn context_sha256(&self) -> &str {
        &self.submission.context_sha256
    }
    pub fn context_view(&self) -> &Arc<ReviewView> {
        self.submission.bound.view()
    }
    pub fn limits(&self) -> NnuecCaptureLimits {
        self.submission.limits
    }
    pub fn original_deadline(&self) -> Deadline {
        self.submission.original_deadline
    }
    pub fn job(&self) -> CpuJobEnvelope {
        self.job
    }
    pub fn owner(&self) -> CpuWorkerId {
        self.owner
    }
    pub fn frame(&self) -> FrameId {
        self.submission.frame
    }
    pub fn question_digest(&self) -> Digest {
        self.submission.question
    }
    pub fn capture_sequence(&self) -> u64 {
        self.submission.capture_sequence
    }
    pub fn ending(&self) -> NnuecCaptureEnding {
        self.ending
    }
    pub fn readouts(&self) -> &[NnuecCapturedReadout] {
        &self.readouts
    }
    pub fn attempted(&self) -> u64 {
        self.attempted
    }
    pub fn selected(&self) -> u64 {
        self.selected
    }
    pub fn overflow(&self) -> u64 {
        self.overflow
    }
    pub fn input_bytes(&self) -> usize {
        self.input_bytes
    }
    /// Only sampled input/Rules preparation, not the complete collection
    /// overhead. input_bytes counts retained serialized inputs, not peak heap.
    pub fn preparation_time(&self) -> Duration {
        self.preparation_time
    }
    pub fn error(&self) -> Option<&CaptureError> {
        self.error.as_ref()
    }
}

#[cfg(test)]
#[path = "nnuec_v2_capture_tests.rs"]
mod tests;
