//! Actual native-R capture and semantic observations, separate from P/C
//! tickets, NN-input counters, CPU credit and complete-action cost targets.
use super::*;
use rz_search::cpu_value::nnuec::v2::capture::{
    NnuecCaptureEnding, NnuecCaptureLimits, NnuecCaptureReservation, NnuecCaptureRetirement,
};
use rz_search::cpu_value::nnuec::v2::review::BoundNnuecReviewContext;
use rz_search::pals::engine::adaptive_port::AdaptiveNnuecReviewSeal;
use rz_search::pals::engine::adaptive_port::{CpuSemanticStage, CpuSemanticWitness};

const MAX_PENDING_CAPTURES: usize = 64;
const MAX_SEMANTIC_OBSERVATIONS: usize = 256;

/// A bounded collection declaration. It supplies no Work, Bound, worker,
/// physical terminal, semantic-consumption or training-target authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveCpuCapturePlan {
    pub capture_sequence: u64,
    pub limits: NnuecCaptureLimits,
}

/// The caller joins its original Collection admission and loaded NNUEC source
/// in admitted(). Callbacks borrow original evidence or move small bounded
/// custody to the existing collector. They must not encode, evaluate, perform
/// archive I/O, or wait on a CPU/collector thread in this manager loop.
pub trait AdaptiveCpuSemanticObserver<Q>: Send {
    fn admitted(
        &mut self,
        admission: &RoventCollectionRootAdmission,
    ) -> Result<(), AdaptiveCollectionError>;
    fn capture_plan(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
    ) -> Result<Option<AdaptiveCpuCapturePlan>, AdaptiveCollectionError>;
    fn capture_reserved(
        &mut self,
        reservation: &NnuecCaptureReservation,
    ) -> Result<(), AdaptiveCollectionError>;
    fn capture_retired(
        &mut self,
        retirement: &NnuecCaptureRetirement,
    ) -> Result<(), AdaptiveCollectionError>;
    /// No Job was enqueued and no worker terminal capability exists. This is
    /// an explicit absence observation, never an empty successful retirement.
    fn capture_unsubmitted(
        &mut self,
        reservation: &NnuecCaptureReservation,
    ) -> Result<(), AdaptiveCollectionError>;
    fn semantic_witness(
        &mut self,
        witness: &CpuSemanticWitness,
    ) -> Result<(), AdaptiveCollectionError>;
}

pub type AdaptiveCpuSemanticObserverBox<Q> = Box<dyn AdaptiveCpuSemanticObserver<Q>>;

pub struct AdaptiveCpuCaptureCustody {
    frame: FrameId,
    question: Digest,
    reservation: NnuecCaptureReservation,
    job: Option<CpuJobEnvelope>,
    enqueued: bool,
    delivery_failed: bool,
}
impl AdaptiveCpuCaptureCustody {
    pub fn frame(&self) -> FrameId {
        self.frame
    }
    pub fn question(&self) -> Digest {
        self.question
    }
    pub fn reservation(&self) -> &NnuecCaptureReservation {
        &self.reservation
    }
    pub fn job(&self) -> Option<CpuJobEnvelope> {
        self.job
    }
    pub fn was_enqueued(&self) -> bool {
        self.enqueued
    }
}

/// Explicit post-fence transfer of the same observer and any unacknowledged
/// original capture. It may still own Bound/View pins and is not a pin-release
/// ACK, a CPU retirement, a successful collection, or permission for a new root.
pub struct AdaptiveCpuSemanticRetirement<Q> {
    state: CpuSemanticState<Q>,
}
pub struct AdaptiveCpuSemanticRetirementParts<Q> {
    pub admission: RoventCollectionRootAdmission,
    pub observer: AdaptiveCpuSemanticObserverBox<Q>,
    pub captures: Vec<AdaptiveCpuCaptureCustody>,
    /// Only the actual post-fence unsubmitted observer boundary populates
    /// this owner list. A missing worker terminal cannot construct it.
    pub unsubmitted: Vec<AdaptiveCpuCaptureCustody>,
    pub witnesses: Vec<CpuSemanticWitness>,
    pub delivered_witnesses: usize,
    /// Attempts, not a count of distinct facts or a zero-valued target.
    pub unrecorded_witness_attempts: u64,
    pub failures: Vec<AdaptiveCollectionError>,
}
impl<Q> AdaptiveCpuSemanticRetirement<Q> {
    pub fn admission(&self) -> &RoventCollectionRootAdmission {
        &self.state.admission
    }
    pub fn retained_captures(&self) -> &[AdaptiveCpuCaptureCustody] {
        &self.state.captures
    }
    pub fn unsubmitted_captures(&self) -> &[AdaptiveCpuCaptureCustody] {
        &self.state.unsubmitted
    }
    pub fn witnesses(&self) -> &[CpuSemanticWitness] {
        &self.state.witnesses
    }
    pub fn delivered_witnesses(&self) -> usize {
        self.state.delivered_witnesses
    }
    pub fn unrecorded_witness_attempts(&self) -> u64 {
        self.state.unrecorded_witness_attempts
    }
    pub fn failures(&self) -> &[AdaptiveCollectionError] {
        &self.state.failures
    }
    pub fn observer_mut(&mut self) -> &mut dyn AdaptiveCpuSemanticObserver<Q> {
        self.state.observer.as_mut()
    }
    pub fn into_parts(self) -> AdaptiveCpuSemanticRetirementParts<Q> {
        AdaptiveCpuSemanticRetirementParts {
            admission: self.state.admission,
            observer: self.state.observer,
            captures: self.state.captures,
            unsubmitted: self.state.unsubmitted,
            witnesses: self.state.witnesses,
            delivered_witnesses: self.state.delivered_witnesses,
            unrecorded_witness_attempts: self.state.unrecorded_witness_attempts,
            failures: self.state.failures,
        }
    }
}

pub(super) struct CpuSemanticState<Q> {
    admission: RoventCollectionRootAdmission,
    observer: AdaptiveCpuSemanticObserverBox<Q>,
    planned: Vec<(FrameId, Digest)>,
    sequences: Vec<u64>,
    captures: Vec<AdaptiveCpuCaptureCustody>,
    unsubmitted: Vec<AdaptiveCpuCaptureCustody>,
    witnesses: Vec<CpuSemanticWitness>,
    delivered_witnesses: usize,
    witness_delivery_failed: bool,
    unrecorded_witness_attempts: u64,
    failures: Vec<AdaptiveCollectionError>,
}

fn collection_error(code: &'static str, detail: impl Into<String>) -> AdaptiveCollectionError {
    AdaptiveCollectionError::new(AdaptiveCollectionStage::Settled, code, detail)
}

fn same_fact(left: &CpuSemanticWitness, right: &CpuSemanticWitness) -> bool {
    left.native_execution() == right.native_execution()
        && left.stage() == right.stage()
        && left.selected_pc_ticket() == right.selected_pc_ticket()
}

pub(super) fn promote_selected<Q, D: AdaptiveSearchDriver<Q>>(
    state: &mut CpuSemanticState<Q>,
    driver: &D,
    input: &FrameInput<ReviewView, Q>,
    accepted: &rz_search::pals::coordinator::AcceptedRoleStep,
    output: &RoleOutput,
) -> Vec<AdaptiveCollectionError> {
    let mut failures = Vec::new();
    for feedback in state.feedback() {
        if !input
            .view
            .records()
            .iter()
            .any(|record| record.cpu_observation == Some(feedback.observation()))
        {
            continue;
        }
        match driver.selected_cpu_pc_consumption(&feedback, input, accepted, output) {
            Ok(Some(witness)) => {
                if let Err(error) = state.append(witness) {
                    failures.push(error);
                }
            }
            Ok(None) => {}
            Err(error) => failures.push(collection_error(
                "CpuSelectedRecordRejected",
                format!("{error:?}"),
            )),
        }
    }
    failures
}

impl<Q> CpuSemanticState<Q> {
    pub(super) fn new(
        admission: RoventCollectionRootAdmission,
        mut observer: AdaptiveCpuSemanticObserverBox<Q>,
    ) -> Result<Self, AdaptiveCollectionError> {
        observer.admitted(&admission)?;
        Ok(Self {
            admission,
            observer,
            planned: Vec::new(),
            sequences: Vec::new(),
            captures: Vec::new(),
            unsubmitted: Vec::new(),
            witnesses: Vec::new(),
            delivered_witnesses: 0,
            witness_delivery_failed: false,
            unrecorded_witness_attempts: 0,
            failures: Vec::new(),
        })
    }
    pub(super) fn record_failure(&mut self, error: AdaptiveCollectionError) {
        if self.failures.len() < 64 {
            self.failures.push(error);
        }
    }
    pub(super) fn plan(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
    ) -> Result<Option<AdaptiveCpuCapturePlan>, AdaptiveCollectionError> {
        let key = (work.frame, work.input.question_digest);
        if self.planned.contains(&key) {
            return Err(collection_error(
                "CpuCaptureRepeatedQuestion",
                "the original question was already assigned its collection plan",
            ));
        }
        if self.planned.len() >= CPU_JOB_CAPACITY || self.captures.len() >= MAX_PENDING_CAPTURES {
            return Err(collection_error(
                "CpuCaptureCapacity",
                "original capture custody is full; no new plan was selected",
            ));
        }
        // Mark before calling the collector. Failure cannot select a new plan
        // or sequence for the same original question on a later retry.
        self.planned.push(key);
        let plan = self.observer.capture_plan(work)?;
        if let Some(plan) = plan {
            if plan.capture_sequence == 0 || self.sequences.contains(&plan.capture_sequence) {
                return Err(collection_error(
                    "CpuCaptureSequence",
                    "zero or repeated capture sequence in this original root",
                ));
            }
            self.sequences.push(plan.capture_sequence);
        }
        Ok(plan)
    }
    pub(super) fn reserved(
        &mut self,
        frame: FrameId,
        question: Digest,
        reservation: NnuecCaptureReservation,
    ) -> Result<(), AdaptiveCollectionError> {
        // Install original custody before an observer can fail. That failure
        // may not erase an actual reservation or its retained Bound/View.
        self.captures.push(AdaptiveCpuCaptureCustody {
            frame,
            question,
            reservation,
            job: None,
            enqueued: false,
            delivery_failed: false,
        });
        let held = self
            .captures
            .last_mut()
            .expect("installed original capture");
        let result = self.observer.capture_reserved(&held.reservation);
        held.delivery_failed = result.is_err();
        result
    }
    pub(super) fn enqueued(&mut self, job: CpuJobEnvelope) {
        if let Some(held) = self
            .captures
            .iter_mut()
            .find(|held| held.question == job.question)
        {
            held.job = Some(job);
            held.enqueued = true;
        }
    }
    pub(super) fn poll_retirements(&mut self) -> Vec<AdaptiveCollectionError> {
        let mut failures = Vec::new();
        let mut index = 0;
        while index < self.captures.len() {
            let held = &mut self.captures[index];
            if held.delivery_failed {
                index += 1;
                continue;
            }
            let Some(retirement) = held.reservation.retirement() else {
                index += 1;
                continue;
            };
            let same_original = match retirement.ending() {
                NnuecCaptureEnding::NotStarted => held.reservation.matches_retirement(retirement),
                _ => retirement.matches_lineage(&held.reservation.lineage()),
            };
            if !same_original || held.job.is_some_and(|job| job != retirement.job()) {
                held.delivery_failed = true;
                failures.push(collection_error(
                    "CpuCaptureRetirementMismatch",
                    "worker terminal differs from the original capture/Job",
                ));
                index += 1;
                continue;
            }
            match self.observer.capture_retired(retirement) {
                Ok(()) => {
                    self.captures.remove(index);
                }
                Err(error) => {
                    held.delivery_failed = true;
                    failures.push(error);
                    index += 1;
                }
            }
        }
        failures
    }
    pub(super) fn append(
        &mut self,
        witness: CpuSemanticWitness,
    ) -> Result<(), AdaptiveCollectionError> {
        if self
            .witnesses
            .iter()
            .any(|original| same_fact(original, &witness))
        {
            return Ok(());
        }
        if self.witnesses.len() >= MAX_SEMANTIC_OBSERVATIONS {
            self.unrecorded_witness_attempts = self.unrecorded_witness_attempts.saturating_add(1);
            return Err(collection_error(
                "CpuSemanticOutboxCapacity",
                "actual semantic fact was observed but its bounded outbox is full",
            ));
        }
        self.witnesses.push(witness);
        Ok(())
    }
    pub(super) fn feedback(&self) -> Vec<CpuSemanticWitness> {
        self.witnesses
            .iter()
            .filter(|witness| witness.stage() == CpuSemanticStage::SearchFeedbackConsumed)
            .cloned()
            .collect()
    }
    pub(super) fn deliver_witnesses(&mut self) -> Result<(), AdaptiveCollectionError> {
        if self.witness_delivery_failed {
            return Ok(());
        }
        while let Some(witness) = self.witnesses.get(self.delivered_witnesses) {
            if let Err(error) = self.observer.semantic_witness(witness) {
                self.witness_delivery_failed = true;
                return Err(error);
            }
            self.delivered_witnesses += 1;
        }
        Ok(())
    }
    pub(super) fn retire_unsubmitted(&mut self) -> Vec<AdaptiveCollectionError> {
        let mut failures = self.poll_retirements();
        let mut index = 0;
        while index < self.captures.len() {
            let held = &mut self.captures[index];
            if held.enqueued || held.delivery_failed {
                index += 1;
                continue;
            }
            match self.observer.capture_unsubmitted(&held.reservation) {
                Ok(()) => {
                    // Only this actual post-fence not-enqueued boundary
                    // preserves custody for the sidecar acknowledgement.
                    self.unsubmitted.push(self.captures.remove(index));
                }
                Err(error) => {
                    held.delivery_failed = true;
                    failures.push(error);
                    index += 1;
                }
            }
        }
        failures
    }
    pub(super) fn into_retirement(self) -> AdaptiveCpuSemanticRetirement<Q> {
        AdaptiveCpuSemanticRetirement { state: self }
    }
}

impl<Q, B, D> AdaptiveExecutionManager<Q, B, D>
where
    Q: Clone + Send + Sync + 'static,
    B: TicketNativeBackend<ReviewView, Q>,
    B::Output: ProposerRankingOutput,
    D: AdaptiveSearchDriver<Q>,
{
    pub fn set_cpu_semantic_observer(
        &mut self,
        observer: AdaptiveCpuSemanticObserverBox<Q>,
    ) -> Result<(), AdaptiveManagerError> {
        if self.root_started
            || self.phase != ManagerPhase::Running
            || self.nnuec.is_none()
            || self.cpu_semantic.is_some()
            || self.researcher_source_kind() != AdaptiveResearcherSourceKind::Native
            || self.registration.endpoints.execution_use != CpuExecutionUse::Collection
            // This observer captures the actual native R worker; the teacher
            // endpoint is selected independently and may be external.
            || self.registration.endpoints.cpu_r != CpuEndpoint::Own
        {
            return Err(AdaptiveManagerError::Invalid(
                "native R collection requires original idle NNUEC Collection root",
            ));
        }
        let admission = self
            .collection_admission
            .as_ref()
            .ok_or(AdaptiveManagerError::Invalid(
                "R collection has no original Collection admission",
            ))?;
        self.driver.validate_collection_admission(admission)?;
        self.cpu_semantic = Some(CpuSemanticState::new(admission.clone(), observer)?);
        Ok(())
    }

    /// Physical owner closure permits transfer, not dropping the returned
    /// Bound/View custody. The Collection caller must finish its original sink
    /// and release these captures before semantic-owner reuse can succeed.
    pub fn take_cpu_semantic_retirement(
        &mut self,
    ) -> Result<Option<AdaptiveCpuSemanticRetirement<Q>>, AdaptiveManagerError> {
        if self.cpu_semantic.is_none() {
            return Ok(None);
        }
        let receipt = self.shutdown_receipt()?;
        if self.phase != ManagerPhase::Finished
            || !(receipt.permits_root_reuse() || receipt.permits_archive_close())
        {
            return Err(AdaptiveManagerError::PhysicalUnknown);
        }
        let failures = self
            .cpu_semantic
            .as_mut()
            .expect("original R collector")
            .retire_unsubmitted();
        for failure in failures {
            self.record_cpu_semantic_failure(failure);
        }
        Ok(self
            .cpu_semantic
            .take()
            .map(CpuSemanticState::into_retirement))
    }

    pub(super) fn reserve_nnuec_job_capture(
        &mut self,
        work: &CoordinatorWork<ReviewView, Q>,
        seal: &AdaptiveNnuecReviewSeal,
        bound: &Arc<BoundNnuecReviewContext>,
    ) -> Result<Option<NnuecCaptureReservation>, AdaptiveManagerError> {
        let Some(state) = &mut self.cpu_semantic else {
            return Ok(None);
        };
        let plan = match state.plan(work) {
            Ok(plan) => plan,
            Err(error) => {
                state.record_failure(error.clone());
                return Err(error.into());
            }
        };
        let Some(plan) = plan else {
            return Ok(None);
        };
        let model = Arc::clone(
            &self
                .nnuec
                .as_ref()
                .ok_or(AdaptiveManagerError::Invalid(
                    "original NNUEC model missing from R capture",
                ))?
                .model,
        );
        let reservation = match self.driver.reserve_cpu_nnuec_capture(
            seal,
            model,
            Arc::clone(bound),
            plan.capture_sequence,
            plan.limits,
        ) {
            Ok(reservation) => reservation,
            Err(error) => {
                state.record_failure(collection_error(
                    "CpuCaptureSourceRejected",
                    format!("{error:?}"),
                ));
                return Err(error);
            }
        };
        if let Err(error) =
            state.reserved(work.frame, work.input.question_digest, reservation.clone())
        {
            state.record_failure(error.clone());
            return Err(error.into());
        }
        Ok(Some(reservation))
    }

    pub(super) fn record_cpu_semantic_failure(&mut self, failure: AdaptiveCollectionError) {
        if let Some(state) = &mut self.cpu_semantic {
            state.record_failure(failure.clone());
        }
        self.record_failure(failure.into());
    }

    pub(super) fn poll_cpu_semantic(&mut self) {
        let Some(state) = &mut self.cpu_semantic else {
            return;
        };
        let mut failures = state.poll_retirements();
        if let Err(error) = state.deliver_witnesses() {
            failures.push(error);
        }
        for failure in failures {
            self.record_cpu_semantic_failure(failure);
        }
    }

    pub(super) fn observe_cpu_semantic_use(
        &mut self,
        binding: &OwnBinding<Q>,
        report: &CpuSliceReport,
    ) {
        if self.cpu_semantic.is_none() {
            return;
        }
        let Some(capture) = binding
            .nnuec
            .as_ref()
            .and_then(|custody| custody.capture.as_ref())
        else {
            return;
        };
        let Some(settled) = binding.semantic_settlement.as_ref() else {
            self.record_cpu_semantic_failure(collection_error(
                "CpuSemanticSettlementMissing",
                "original settled slice custody is unavailable",
            ));
            return;
        };
        let witnesses =
            self.driver
                .cpu_semantic_witnesses(&binding.work, report, settled, &capture.lineage());
        let witnesses = match witnesses {
            Ok(witnesses) => witnesses,
            Err(error) => {
                self.record_cpu_semantic_failure(collection_error(
                    "CpuSemanticSourceRejected",
                    format!("{error:?}"),
                ));
                return;
            }
        };
        for witness in witnesses {
            if let Err(error) = self
                .cpu_semantic
                .as_mut()
                .expect("original R collector")
                .append(witness)
            {
                self.record_cpu_semantic_failure(error);
            }
        }
        self.poll_cpu_semantic();
    }

    /// The factory checks the actual selected record's CPU observation and
    /// exact accepted head-use tombstone. Ordinary, Fresh and GoalLocal inputs
    /// remain their original FrameInput; no ordinary Frame is synthesized.
    pub(super) fn observe_cpu_pc_head(
        &mut self,
        input: &FrameInput<ReviewView, Q>,
        accepted: &rz_search::pals::coordinator::AcceptedRoleStep,
        output: &RoleOutput,
    ) {
        let Some(state) = &self.cpu_semantic else {
            return;
        };
        let _ = state;
        let failures = promote_selected(
            self.cpu_semantic.as_mut().expect("original R collector"),
            &self.driver,
            input,
            accepted,
            output,
        );
        for failure in failures {
            self.record_cpu_semantic_failure(failure);
        }
        self.poll_cpu_semantic();
    }
}
