//! 실제 Collection의 R 캡처를 관리자 밖으로 넘기는 유한 관측 수신부다.
//!
//! 콜백은 원 reservation과 작은 semantic witness만 보존한다. Rules 재생,
//! 입력 직렬화, 파일 I/O, 모델 실행은 외부 collector의 책임이다. 실패한
//! 콜백의 원 cap은 관리자 퇴역 owner와 함께 회수하며 성공 자료로 바꾸지 않는다.

use super::{ArenaError, RoventCollectionRootAdmission, check, invalid};
use rz_contracts::Digest;
use rz_search::cpu_value::nnuec::v2::capture::{
    NnuecCaptureLimits, NnuecCaptureReservation, NnuecCaptureRetirement,
};
use rz_search::pals::coordinator::{CoordinatorWork, FrameId};
use rz_search::pals::engine::adaptive_port::{AdaptivePortQuestion, CpuSemanticWitness};
use rz_search::pals::review_context::ReviewView;
use rz_uci::pals_adaptive::{
    AdaptiveCollectionError, AdaptiveCollectionStage, AdaptiveCpuCapturePlan,
    AdaptiveCpuSemanticObserver, AdaptiveCpuSemanticRetirement,
};
use rz_uci::rovent_bootstrap::nnuec_source::LoadedRoventNnuecSource;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

const MAX_CAPTURES: usize = 64;
const MAX_WITNESSES: usize = 256;

/// 명시적인 root별 수신 한도다. input byte 한도는 보존한 직렬 입력에만
/// 적용되며 전체 heap·peak RAM 한도를 뜻하지 않는다.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoventRObserverLimits {
    maximum_captures: usize,
    maximum_witnesses: usize,
    capture: NnuecCaptureLimits,
}
impl RoventRObserverLimits {
    pub fn new(
        maximum_captures: usize,
        maximum_witnesses: usize,
        capture: NnuecCaptureLimits,
    ) -> Result<Self, ArenaError> {
        check(
            (1..=MAX_CAPTURES).contains(&maximum_captures)
                && (1..=MAX_WITNESSES).contains(&maximum_witnesses),
            "R observer capture/witness capacity is outside the registered bound",
        )?;
        Ok(Self {
            maximum_captures,
            maximum_witnesses,
            capture,
        })
    }
    pub fn maximum_captures(&self) -> usize {
        self.maximum_captures
    }
    pub fn maximum_witnesses(&self) -> usize {
        self.maximum_witnesses
    }
    pub fn capture_limits(&self) -> NnuecCaptureLimits {
        self.capture
    }
    pub fn maximum_serialized_input_bytes(&self) -> usize {
        // Both factors have small validated upper bounds (64 * 16 MiB).
        self.maximum_captures * self.capture.maximum_input_bytes()
    }
}

#[derive(Clone, Copy)]
struct CapturePlan {
    frame: FrameId,
    question: Digest,
    sequence: u64,
    full_view: Digest,
    view_seal: Digest,
    deadline: rz_contracts::Deadline,
}

/// 콜백의 수신 상태다. 이 표시는 worker 실행/완료/semantic 권한이 아니며,
/// 실제 terminal은 reservation의 원 OnceLock에서만 읽는다.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoventRCaptureDelivery {
    Reserved,
    TerminalObserved,
    UnsubmittedObserved,
    RecoveredAfterFence,
}

#[derive(Clone)]
pub struct RoventRCapturedReservation {
    original: NnuecCaptureReservation,
    delivery: RoventRCaptureDelivery,
}
impl RoventRCapturedReservation {
    pub fn reservation(&self) -> &NnuecCaptureReservation {
        &self.original
    }
    pub fn retirement(&self) -> Option<&NnuecCaptureRetirement> {
        self.original.retirement()
    }
    pub fn delivery(&self) -> RoventRCaptureDelivery {
        self.delivery
    }
}

struct Outbox {
    plans: Vec<CapturePlan>,
    captures: Vec<RoventRCapturedReservation>,
    witnesses: Vec<CpuSemanticWitness>,
}
struct Shared {
    admission: RoventCollectionRootAdmission,
    source: Arc<LoadedRoventNnuecSource>,
    limits: RoventRObserverLimits,
    sequence: Arc<AtomicU64>,
    // None means the actual post-fence receiver took custody. A late callback
    // cannot refill the outbox or acquire authority from a closed receiver.
    outbox: Mutex<Option<Outbox>>,
}

pub struct RoventRCollectionObserver {
    shared: Arc<Shared>,
}
/// 이 원 receiver는 root 종료 때 원 manager 퇴역과 함께 소비한다. Clone을
/// 제공하지 않아 서로 다른 수신자가 같은 outbox를 완료할 수 없다.
pub struct RoventRCollectionReceiver {
    shared: Arc<Shared>,
}

impl RoventRCollectionObserver {
    pub fn new(
        admission: RoventCollectionRootAdmission,
        source: Arc<LoadedRoventNnuecSource>,
        limits: RoventRObserverLimits,
        shared_process_sequence: Arc<AtomicU64>,
    ) -> Result<(Self, RoventRCollectionReceiver), ArenaError> {
        let lock = admission.native_origin().original_input_lock();
        let pin = lock
            .manifest
            .engine
            .nnuec
            .as_ref()
            .ok_or_else(|| invalid("R collection source has no original NNUEC declaration"))?;
        check(
            lock.manifest.purpose == rz_experiments::RoventPurpose::Collection
                && lock.manifest.engine.mode.has_native_r()
                && admission.external_source().is_none()
                && source.input_lock_digest() == admission.root_admission().startup_lock_digest()
                && source.weights_artifact() == &pin.weights
                && source.provenance_artifact() == &pin.provenance,
            "R observer requires original native Collection admission and loaded NNUEC source",
        )?;
        check(
            !matches!(
                shared_process_sequence.load(Ordering::Relaxed),
                0 | u64::MAX
            ),
            "R capture process sequence must start at a nonzero available value",
        )?;
        let shared = Arc::new(Shared {
            admission,
            source,
            limits,
            sequence: shared_process_sequence,
            outbox: Mutex::new(Some(Outbox {
                plans: Vec::with_capacity(limits.maximum_captures),
                captures: Vec::with_capacity(limits.maximum_captures),
                witnesses: Vec::with_capacity(limits.maximum_witnesses),
            })),
        });
        Ok((
            Self {
                shared: Arc::clone(&shared),
            },
            RoventRCollectionReceiver { shared },
        ))
    }
}

fn callback_error(code: &'static str, detail: &'static str) -> AdaptiveCollectionError {
    AdaptiveCollectionError::new(AdaptiveCollectionStage::Settled, code, detail)
}

fn next_sequence(sequence: &AtomicU64) -> Result<u64, AdaptiveCollectionError> {
    sequence
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            (value != 0).then(|| value.checked_add(1)).flatten()
        })
        .map_err(|_| callback_error("RObserverSequence", "R process capture sequence exhausted"))
}

fn same_admission(
    original: &RoventCollectionRootAdmission,
    candidate: &RoventCollectionRootAdmission,
) -> bool {
    Arc::ptr_eq(original.root_admission(), candidate.root_admission())
        && Arc::ptr_eq(original.rules(), candidate.rules())
        && Arc::ptr_eq(original.root_moves(), candidate.root_moves())
        && original.authority() == candidate.authority()
        && original.raw_sha256() == candidate.raw_sha256()
}

fn callback_outbox(
    shared: &Shared,
) -> Result<MutexGuard<'_, Option<Outbox>>, AdaptiveCollectionError> {
    let guard = shared.outbox.try_lock().map_err(|error| match error {
        TryLockError::WouldBlock => callback_error(
            "RObserverBusy",
            "R receiver is occupied; original evidence remains in manager custody",
        ),
        TryLockError::Poisoned(_) => callback_error(
            "RObserverPoisoned",
            "R receiver is poisoned; original evidence must be retained",
        ),
    })?;
    if guard.is_none() {
        return Err(callback_error(
            "RObserverClosed",
            "R receiver is already closed",
        ));
    }
    Ok(guard)
}

fn validate_reservation(
    shared: &Shared,
    outbox: &Outbox,
    reservation: &NnuecCaptureReservation,
) -> Result<(), AdaptiveCollectionError> {
    let lineage = reservation.lineage();
    if !outbox.plans.iter().any(|plan| {
        plan.frame == lineage.frame()
            && plan.question == lineage.question_digest()
            && plan.sequence == lineage.capture_sequence()
            && plan.full_view == lineage.full_view_digest()
            && plan.view_seal == lineage.view_seal_digest()
            && plan.deadline == lineage.original_deadline()
    }) {
        return Err(callback_error(
            "RObserverReservation",
            "original reservation does not match this receiver's selected question",
        ));
    }
    if let Some(retirement) = reservation.retirement()
        && (!retirement.matches_model(shared.source.model())
            || retirement.limits() != shared.limits.capture
            || retirement.job().id.authority != shared.admission.authority()
            || retirement.frame() != lineage.frame()
            || retirement.question_digest() != lineage.question_digest()
            || retirement.capture_sequence() != lineage.capture_sequence())
    {
        return Err(callback_error(
            "RObserverSource",
            "actual terminal differs from original loaded model/question/capture limits",
        ));
    }
    Ok(())
}

fn same_fact(left: &CpuSemanticWitness, right: &CpuSemanticWitness) -> bool {
    left.native_execution() == right.native_execution()
        && left.stage() == right.stage()
        && left.selected_pc_ticket() == right.selected_pc_ticket()
}

fn retain_witness(
    shared: &Shared,
    outbox: &mut Outbox,
    witness: &CpuSemanticWitness,
) -> Result<(), AdaptiveCollectionError> {
    if witness.authority() != shared.admission.authority()
        || !outbox.captures.iter().any(|held| {
            held.original.matches_lineage(witness.capture_lineage())
                && witness
                    .capture_lineage()
                    .matches_activated_job(witness.job(), witness.owner())
                && witness.frame() == witness.capture_lineage().frame()
                && witness.question() == witness.capture_lineage().question_digest()
        })
    {
        return Err(callback_error(
            "RObserverWitness",
            "semantic fact has no original activated capture in this receiver",
        ));
    }
    if let Some(original) = outbox.witnesses.iter().find(|row| same_fact(row, witness)) {
        if original.job() != witness.job()
            || original.owner() != witness.owner()
            || original.frame() != witness.frame()
            || original.question() != witness.question()
            || original.observation() != witness.observation()
            || original.store_execution() != witness.store_execution()
            || original.selected_record() != witness.selected_record()
            || original.selected_pc_output_digest() != witness.selected_pc_output_digest()
            || original.selected_full_view_digest() != witness.selected_full_view_digest()
        {
            return Err(callback_error(
                "RObserverConflictingWitness",
                "the same semantic occurrence was delivered with different original evidence",
            ));
        }
        return Ok(());
    }
    if outbox.witnesses.len() >= shared.limits.maximum_witnesses {
        return Err(callback_error(
            "RObserverWitnessCapacity",
            "R semantic outbox is full; manager retains the undelivered original witness",
        ));
    }
    outbox.witnesses.push(witness.clone());
    Ok(())
}

impl AdaptiveCpuSemanticObserver<AdaptivePortQuestion> for RoventRCollectionObserver {
    fn admitted(
        &mut self,
        admission: &RoventCollectionRootAdmission,
    ) -> Result<(), AdaptiveCollectionError> {
        if !same_admission(&self.shared.admission, admission) {
            return Err(callback_error(
                "RObserverAdmission",
                "R observer refuses another actual Collection root owner",
            ));
        }
        Ok(())
    }
    fn capture_plan(
        &mut self,
        work: &CoordinatorWork<ReviewView, AdaptivePortQuestion>,
    ) -> Result<Option<AdaptiveCpuCapturePlan>, AdaptiveCollectionError> {
        let root_deadline = self.shared.admission.root_admission().hard_deadline();
        if work.frame.authority != self.shared.admission.authority()
            || work.input.view_seal.authority != work.frame.authority
            || Digest(work.input.view.seal_digest()) != work.input.view_seal.view_digest
            || work.original_deadline.clock != root_deadline.clock
            || work.original_deadline.at > root_deadline.at
        {
            return Err(callback_error(
                "RObserverWork",
                "capture plan does not belong to the original root/view/deadline",
            ));
        }
        let mut guard = callback_outbox(&self.shared)?;
        let outbox = guard.as_mut().expect("checked open receiver");
        if outbox
            .plans
            .iter()
            .any(|plan| plan.frame == work.frame && plan.question == work.input.question_digest)
        {
            return Err(callback_error(
                "RObserverRepeatedQuestion",
                "the original R question already has its one capture plan",
            ));
        }
        if outbox.plans.len() >= self.shared.limits.maximum_captures {
            return Err(callback_error(
                "RObserverCaptureCapacity",
                "R original capture capacity exhausted; no replacement plan is minted",
            ));
        }
        let sequence = next_sequence(&self.shared.sequence)?;
        outbox.plans.push(CapturePlan {
            frame: work.frame,
            question: work.input.question_digest,
            sequence,
            full_view: Digest(work.input.view.digest()),
            view_seal: Digest(work.input.view.seal_digest()),
            deadline: work.original_deadline,
        });
        Ok(Some(AdaptiveCpuCapturePlan {
            capture_sequence: sequence,
            limits: self.shared.limits.capture,
        }))
    }
    fn capture_reserved(
        &mut self,
        reservation: &NnuecCaptureReservation,
    ) -> Result<(), AdaptiveCollectionError> {
        let mut guard = callback_outbox(&self.shared)?;
        let outbox = guard.as_mut().expect("checked open receiver");
        validate_reservation(&self.shared, outbox, reservation)?;
        if outbox
            .captures
            .iter()
            .any(|held| held.original.matches_lineage(&reservation.lineage()))
        {
            return Err(callback_error(
                "RObserverRepeatedReservation",
                "original capture repeated",
            ));
        }
        if outbox.captures.len() >= self.shared.limits.maximum_captures {
            return Err(callback_error(
                "RObserverCaptureCapacity",
                "R capture custody is full",
            ));
        }
        outbox.captures.push(RoventRCapturedReservation {
            original: reservation.clone(),
            delivery: RoventRCaptureDelivery::Reserved,
        });
        Ok(())
    }
    fn capture_retired(
        &mut self,
        retirement: &NnuecCaptureRetirement,
    ) -> Result<(), AdaptiveCollectionError> {
        let mut guard = callback_outbox(&self.shared)?;
        let outbox = guard.as_mut().expect("checked open receiver");
        let index = outbox
            .captures
            .iter()
            .position(|held| {
                held.original
                    .retirement()
                    .is_some_and(|original| std::ptr::eq(original, retirement))
            })
            .ok_or_else(|| {
                callback_error(
                    "RObserverTerminal",
                    "terminal is not the original reserved OnceLock",
                )
            })?;
        validate_reservation(&self.shared, outbox, &outbox.captures[index].original)?;
        if outbox.captures[index].delivery != RoventRCaptureDelivery::Reserved {
            return Err(callback_error(
                "RObserverRepeatedTerminal",
                "capture terminal already delivered",
            ));
        }
        outbox.captures[index].delivery = RoventRCaptureDelivery::TerminalObserved;
        Ok(())
    }
    fn capture_unsubmitted(
        &mut self,
        reservation: &NnuecCaptureReservation,
    ) -> Result<(), AdaptiveCollectionError> {
        let mut guard = callback_outbox(&self.shared)?;
        let outbox = guard.as_mut().expect("checked open receiver");
        let held = outbox
            .captures
            .iter_mut()
            .find(|held| held.original.matches_lineage(&reservation.lineage()))
            .ok_or_else(|| {
                callback_error(
                    "RObserverUnsubmitted",
                    "unsubmitted capture has no original reservation",
                )
            })?;
        if held.delivery != RoventRCaptureDelivery::Reserved || reservation.retirement().is_some() {
            return Err(callback_error(
                "RObserverUnsubmitted",
                "unsubmitted observation cannot overwrite an actual or delivered terminal",
            ));
        }
        held.delivery = RoventRCaptureDelivery::UnsubmittedObserved;
        Ok(())
    }
    fn semantic_witness(
        &mut self,
        witness: &CpuSemanticWitness,
    ) -> Result<(), AdaptiveCollectionError> {
        let mut guard = callback_outbox(&self.shared)?;
        retain_witness(
            &self.shared,
            guard.as_mut().expect("checked open receiver"),
            witness,
        )
    }
}

/// 실제 post-fence 회수 owner다. 이것을 성공 수집·pin 해제 ACK·학습 target으로
/// 해석하지 않는다. 원 manager 실패와 미수신 cap도 이 owner에 남는다.
pub struct RoventRCollectionClosure {
    admission: RoventCollectionRootAdmission,
    source: Arc<LoadedRoventNnuecSource>,
    captures: Vec<RoventRCapturedReservation>,
    witnesses: Vec<CpuSemanticWitness>,
    original: AdaptiveCpuSemanticRetirement<AdaptivePortQuestion>,
}
impl RoventRCollectionClosure {
    pub fn admission(&self) -> &RoventCollectionRootAdmission {
        &self.admission
    }
    pub fn nnuec_source(&self) -> &Arc<LoadedRoventNnuecSource> {
        &self.source
    }
    pub fn captures(&self) -> &[RoventRCapturedReservation] {
        &self.captures
    }
    pub fn witnesses(&self) -> &[CpuSemanticWitness] {
        &self.witnesses
    }
    pub fn original_retirement(&self) -> &AdaptiveCpuSemanticRetirement<AdaptivePortQuestion> {
        &self.original
    }
}

pub struct RoventRCollectionCloseRejected {
    pub error: ArenaError,
    pub receiver: RoventRCollectionReceiver,
    pub original: AdaptiveCpuSemanticRetirement<AdaptivePortQuestion>,
}

impl RoventRCollectionReceiver {
    /// 실제 manager가 physical fence 뒤에만 발급하는 owner를 받는다.
    /// 검증 실패 시에도 수신 cap과 미수신 manager cap을 모두 보존한다.
    // Rejection returns the original receiver, capture outbox and post-fence
    // retirement owner. Do not allocate a new error box or discard custody
    // while reconciling that bounded root's actual physical completion.
    #[allow(clippy::result_large_err)]
    pub fn finish_retaining(
        self,
        original: AdaptiveCpuSemanticRetirement<AdaptivePortQuestion>,
    ) -> Result<RoventRCollectionClosure, RoventRCollectionCloseRejected> {
        let recovered = self.reconcile(&original);
        match recovered {
            Ok(outbox) => Ok(RoventRCollectionClosure {
                admission: self.shared.admission.clone(),
                source: Arc::clone(&self.shared.source),
                captures: outbox.captures,
                witnesses: outbox.witnesses,
                original,
            }),
            Err(error) => Err(RoventRCollectionCloseRejected {
                error,
                receiver: self,
                original,
            }),
        }
    }

    fn reconcile(
        &self,
        original: &AdaptiveCpuSemanticRetirement<AdaptivePortQuestion>,
    ) -> Result<Outbox, ArenaError> {
        check(
            same_admission(&self.shared.admission, original.admission()),
            "R closure belongs to another actual Collection root owner",
        )?;
        // This is outside the manager. Poisoned custody remains reachable in
        // the returned receiver; do not consume a poisoned guard as success.
        let mut guard = self
            .shared
            .outbox
            .lock()
            .map_err(|_| invalid("R receiver custody poisoned"))?;
        let held = guard
            .as_ref()
            .ok_or_else(|| invalid("R receiver already closed"))?;
        // Stage reconciliation before taking any original owner. On error,
        // both the old outbox and the entire manager retirement are returned.
        let mut merged = Outbox {
            plans: held.plans.clone(),
            captures: held.captures.clone(),
            witnesses: held.witnesses.clone(),
        };
        for capture in original.retained_captures() {
            let reservation = capture.reservation();
            validate_reservation(&self.shared, &merged, reservation)
                .map_err(|error| invalid(error.to_string()))?;
            if let Some(terminal) = reservation.retirement() {
                check(
                    capture.job().is_none_or(|job| job == terminal.job()),
                    "manager capture Job differs from the actual original terminal",
                )?;
            }
            if !merged
                .captures
                .iter()
                .any(|row| row.original.matches_lineage(&reservation.lineage()))
            {
                check(
                    merged.captures.len() < self.shared.limits.maximum_captures,
                    "R closure capture recovery exceeds the registered bound",
                )?;
                merged.captures.push(RoventRCapturedReservation {
                    original: reservation.clone(),
                    delivery: RoventRCaptureDelivery::RecoveredAfterFence,
                });
            }
        }
        for capture in &merged.captures {
            validate_reservation(&self.shared, &merged, &capture.original)
                .map_err(|error| invalid(error.to_string()))?;
        }
        for witness in original.witnesses() {
            retain_witness(&self.shared, &mut merged, witness)
                .map_err(|error| invalid(error.to_string()))?;
        }
        let _original_outbox = guard.take().expect("checked original outbox");
        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_r_capture_limits_preserve_individual_and_total_input_bounds() {
        let capture = NnuecCaptureLimits::new(2, 7, 4096).unwrap();
        let limits = RoventRObserverLimits::new(3, 12, capture).unwrap();
        assert_eq!(limits.capture_limits(), capture);
        assert_eq!(limits.maximum_serialized_input_bytes(), 3 * 4096);
        assert!(RoventRObserverLimits::new(0, 12, capture).is_err());
        assert!(RoventRObserverLimits::new(65, 12, capture).is_err());
        assert!(RoventRObserverLimits::new(3, 0, capture).is_err());
        assert!(RoventRObserverLimits::new(3, 257, capture).is_err());
    }

    #[test]
    fn shared_process_capture_sequence_does_not_wrap_or_reuse_exhausted_value() {
        let sequence = Arc::new(AtomicU64::new(u64::MAX - 1));
        let another_root = Arc::clone(&sequence);
        assert_eq!(next_sequence(&sequence).unwrap(), u64::MAX - 1);
        let error = next_sequence(&another_root).unwrap_err();
        assert_eq!(error.code, "RObserverSequence");
        assert_eq!(sequence.load(Ordering::Relaxed), u64::MAX);
        assert!(next_sequence(&AtomicU64::new(0)).is_err());
    }
}
