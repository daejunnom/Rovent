//! Bounded custody for actual Own collection pump items.
//!
//! Reception is independent of publication and never establishes a native
//! fence or teacher retirement. The original receiver and items remain in the
//! same shared owner when its worker is stopped.
use crate::{ArenaError, rovent_collect::{RoventCollectionOwnerLockV2, RoventCorpusAssemblyLockV2, RoventRetiredLockV2Record, collection_hooks_core_v2::RoventCollectionCommitGateV2, teacher::{lock_v2_policy::{CommitTeacherPolicyFailureV2, PreparedTeacherPolicyLockV2}, lock_v2_row_pump::PumpItemV2, v2_session::SealedTeacherPolicyV2}}};
use rz_uci::rovent_bootstrap::RoventNnuecCollectionHandoffV2;
use std::{collections::VecDeque, sync::{Arc, Mutex, mpsc::{self, Receiver, SyncSender, TryRecvError}, atomic::{AtomicBool, Ordering}}, thread, time::{Duration, Instant}};

struct State {
    receiver: Receiver<PumpItemV2>,
    items: VecDeque<PumpItemV2>,
    maximum: usize,
    disconnected: bool,
    teacher_evidence: Vec<SealedTeacherPolicyV2>,
    commit_failures: Vec<Box<CommitTeacherPolicyFailureV2>>,
    failed_items: Vec<PumpItemV2>,
    overflow_item: Option<PumpItemV2>,
}

pub(super) enum OwnOfferOutcomeV2 { Pending, Offered, FailedRetained }
pub(crate) enum OwnPumpPollV2 { Received, Empty, Full, Disconnected }

pub(crate) struct OwnPumpConsumerV2 {
    state: Mutex<State>,
}

impl OwnPumpConsumerV2 {
    pub(crate) fn new(maximum: usize) -> (Arc<Self>, SyncSender<PumpItemV2>) {
        let (sender, receiver) = mpsc::sync_channel(16);
        (Arc::new(Self { state: Mutex::new(State { receiver, items: VecDeque::new(), maximum, disconnected: false, teacher_evidence: Vec::new(), commit_failures: Vec::new(), failed_items: Vec::new(), overflow_item: None }) }), sender)
    }

    /// Stops on full without removing the next original from the channel.
    pub(crate) fn poll(&self) -> Result<OwnPumpPollV2, ArenaError> {
        let mut state = self.state.lock().map_err(|_| ArenaError::Invalid("Own publisher custody poisoned".into()))?;
        if state.items.len().saturating_add(state.failed_items.len()).saturating_add(state.teacher_evidence.len()) >= state.maximum {
            if state.overflow_item.is_some() { return Ok(OwnPumpPollV2::Full); }
            return match state.receiver.try_recv() {
                Ok(item) => { state.overflow_item = Some(item); Ok(OwnPumpPollV2::Full) },
                Err(TryRecvError::Empty) => Ok(OwnPumpPollV2::Empty),
                Err(TryRecvError::Disconnected) => { state.disconnected = true; Ok(OwnPumpPollV2::Disconnected) },
            };
        }
        state.items.try_reserve(1).map_err(|_| ArenaError::Invalid("Own publisher item capacity".into()))?;
        match state.receiver.try_recv() {
            Ok(item) => { state.items.push_back(item); Ok(OwnPumpPollV2::Received) },
            Err(TryRecvError::Empty) => Ok(OwnPumpPollV2::Empty),
            Err(TryRecvError::Disconnected) => { state.disconnected = true; Ok(OwnPumpPollV2::Disconnected) },
        }
    }

    pub(crate) fn try_counts(&self) -> Result<Option<(usize, usize, usize, bool)>, ArenaError> {
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(ArenaError::Invalid("Own publisher custody poisoned".into())),
        };
        Ok(Some((state.items.len(), state.failed_items.len(), state.teacher_evidence.len(), state.disconnected)))
    }

    /// Copy diagnostic record projections without consuming the original rows,
    /// teacher evidence or failure objects. The projection cannot restore
    /// their in-process authority. Receiver contents must first be drained by
    /// the caller after the sender stops.
    pub(crate) fn try_snapshot_retained_rows(
        &self, maximum_bytes: usize,
    ) -> Result<Option<Vec<(&'static str, Vec<u8>)>>, ArenaError> {
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(ArenaError::Invalid("Own publisher custody poisoned".into())),
        };
        let mut projections = Vec::new();
        let mut total = 0usize;
        for item in state.items.iter().chain(state.failed_items.iter()).chain(state.overflow_item.iter()) {
            let (stage, row) = match item {
                PumpItemV2::Unlabeled { row } => ("unlabeled", row),
                PumpItemV2::Prepared { row, .. } => ("prepared", row),
                PumpItemV2::Committed { row, .. } => ("committed", row),
                PumpItemV2::Failed { row, .. } => ("failed", row),
            };
            let bytes = serde_json::to_vec(&row.record).map_err(|error| ArenaError::Invalid(error.to_string()))?;
            total = total.checked_add(bytes.len()).ok_or_else(|| ArenaError::Invalid("consumer projection size overflow".into()))?;
            if total > maximum_bytes { return Err(ArenaError::Invalid("consumer projections exceed report capacity".into())); }
            projections.try_reserve(1).map_err(|_| ArenaError::Invalid("consumer projection allocation".into()))?;
            projections.push((stage, bytes));
        }
        Ok(Some(projections))
    }

    /// This is only a custody check; the caller still verifies both worker
    /// joins, a final channel drain, the root fence and teacher aggregate.
    pub(super) fn try_has_failures(&self) -> Result<Option<bool>, ArenaError> {
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(ArenaError::Invalid("Own publisher custody poisoned".into())),
        };
        Ok(Some(!state.failed_items.is_empty() || !state.commit_failures.is_empty() || state.overflow_item.is_some()))
    }

    /// Attach a prepared teacher opinion and offer the exact original row.
    /// Failed items and rejected offers stay in this owner. Disk I/O and
    /// physical-retirement decisions are deliberately outside this lock.
    pub(super) fn try_offer_front(
        &self,
        owner: &RoventCollectionOwnerLockV2,
        assembly: &mut RoventCorpusAssemblyLockV2,
        gate: &RoventCollectionCommitGateV2<'_, RoventNnuecCollectionHandoffV2>,
        handoff: &mut RoventNnuecCollectionHandoffV2,
    ) -> Result<OwnOfferOutcomeV2, ArenaError> {
        self.try_offer_front_with(assembly, |row, evidence| {
            owner.try_with_fenced_own_producer_mut(row, gate, handoff,
                |producer, row| producer.attach_prepared_teacher_policy_v2(row, evidence))
        })
    }

    pub(super) fn try_offer_front_pc(
        &self,
        owner: &RoventCollectionOwnerLockV2,
        assembly: &mut RoventCorpusAssemblyLockV2,
        gate: &RoventCollectionCommitGateV2<'_, rz_uci::rovent_bootstrap::pc_session::RoventPcCollectionHandoffV2>,
        handoff: &mut rz_uci::rovent_bootstrap::pc_session::RoventPcCollectionHandoffV2,
    ) -> Result<OwnOfferOutcomeV2, ArenaError> {
        self.try_offer_front_with(assembly, |row, evidence| {
            owner.try_with_fenced_pc_producer_mut(row, gate, handoff,
                |producer, row| producer.attach_prepared_teacher_policy_v2(row, evidence))
        })
    }

    fn try_offer_front_with(
        &self,
        assembly: &mut RoventCorpusAssemblyLockV2,
        attach: impl FnOnce(&mut RoventRetiredLockV2Record, &mut Option<PreparedTeacherPolicyLockV2>)
            -> Result<Option<Result<SealedTeacherPolicyV2, Box<CommitTeacherPolicyFailureV2>>>, ArenaError>,
    ) -> Result<OwnOfferOutcomeV2, ArenaError> {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnOfferOutcomeV2::Pending),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(ArenaError::Invalid("Own publisher custody poisoned".into())),
        };
        if state.teacher_evidence.len() >= state.maximum { return Err(ArenaError::Invalid("Own teacher evidence capacity reached".into())); }
        if state.commit_failures.len() >= state.maximum { return Err(ArenaError::Invalid("Own teacher commit failure capacity reached".into())); }
        state.commit_failures.try_reserve(1).map_err(|_| ArenaError::Invalid("Own teacher commit failure capacity".into()))?;
        state.failed_items.try_reserve(1).map_err(|_| ArenaError::Invalid("Own failed row custody capacity".into()))?;
        if matches!(state.items.front(), Some(PumpItemV2::Failed { .. })) {
            if state.failed_items.len() >= state.maximum { return Err(ArenaError::Invalid("Own failed row custody capacity reached".into())); }
            state.failed_items.try_reserve(1).map_err(|_| ArenaError::Invalid("Own failed row custody capacity".into()))?;
            let original = state.items.pop_front().expect("failed front checked");
            state.failed_items.push(original);
            return Ok(OwnOfferOutcomeV2::FailedRetained);
        }
        let Some(item) = state.items.front_mut() else { return Ok(OwnOfferOutcomeV2::Pending); };
        match item {
            PumpItemV2::Failed { .. } => return Err(ArenaError::Invalid("Own teacher row failed; original evidence retained".into())),
            PumpItemV2::Prepared { evidence: Some(_), sealed: Some(_), .. } => return Err(ArenaError::Invalid("Own prepared row has conflicting teacher custody".into())),
            PumpItemV2::Prepared { row, evidence, sealed } if sealed.is_none() => {
                let Some(result) = attach(row, evidence)? else { return Ok(OwnOfferOutcomeV2::Pending); };
                match result {
                    Ok(original) => *sealed = Some(original),
                    Err(original) => {
                        state.commit_failures.push(original);
                        let row = state.items.pop_front().expect("commit failure front retained");
                        state.failed_items.push(row);
                        return Ok(OwnOfferOutcomeV2::FailedRetained);
                    }
                }
            }
            _ => {}
        }
        // Reserve before moving an original sealed evidence out of the item.
        state.teacher_evidence.try_reserve(1).map_err(|_| ArenaError::Invalid("Own teacher evidence capacity".into()))?;
        let item = state.items.pop_front().expect("front checked");
        match item {
            PumpItemV2::Unlabeled { row } => match assembly.offer(row) {
                Ok(()) => Ok(OwnOfferOutcomeV2::Offered),
                Err(original) => { let (error, row) = *original; state.items.push_front(PumpItemV2::Unlabeled { row }); Err(error) }
            },
            PumpItemV2::Prepared { row, evidence, sealed } => match assembly.offer(row) {
                Ok(()) => { if let Some(original) = sealed { state.teacher_evidence.push(original); } Ok(OwnOfferOutcomeV2::Offered) },
                Err(original) => { let (error, row) = *original; state.items.push_front(PumpItemV2::Prepared { row, evidence, sealed }); Err(error) }
            },
            PumpItemV2::Committed { row, sealed } => match assembly.offer(row) {
                Ok(()) => { state.teacher_evidence.push(sealed); Ok(OwnOfferOutcomeV2::Offered) },
                Err(original) => { let (error, row) = *original; state.items.push_front(PumpItemV2::Committed { row, sealed }); Err(error) }
            },
            original @ PumpItemV2::Failed { .. } => { state.items.push_front(original); Err(ArenaError::Invalid("Own teacher row failed; original evidence retained".into())) }
        }
    }

    /// The thread has an explicit deadline. Expiry leaves the original
    /// receiver and all already received items in this owner.
    pub(crate) fn run(&self, stop: &AtomicBool, until: Instant) -> Result<(), ArenaError> {
        while !stop.load(Ordering::Acquire) && Instant::now() < until {
            self.poll()?;
            thread::sleep(Duration::from_millis(2));
        }
        // Preserve any remainder in the original receiver; later explicit
        // draining is performed by the same owner.
        self.poll().map(|_| ())
    }
}
