//! Bounded custody for actual Own collection pump items.
//!
//! Reception is independent of publication and never establishes a native
//! fence or teacher retirement. The original receiver and items remain in the
//! same shared owner when its worker is stopped.
use crate::{ArenaError, rovent_collect::{RoventCollectionOwnerLockV2, RoventCorpusAssemblyLockV2, collection_hooks_core_v2::RoventCollectionCommitGateV2, teacher::{lock_v2_policy::CommitTeacherPolicyFailureV2, lock_v2_row_pump::PumpItemV2, v2_session::SealedTeacherPolicyV2}}};
use rz_uci::rovent_bootstrap::RoventNnuecCollectionHandoffV2;
use std::{collections::VecDeque, sync::{Arc, Mutex, mpsc::{self, Receiver, SyncSender, TryRecvError}, atomic::{AtomicBool, Ordering}}, thread, time::{Duration, Instant}};

struct State {
    receiver: Receiver<PumpItemV2>,
    items: VecDeque<PumpItemV2>,
    maximum: usize,
    disconnected: bool,
    teacher_evidence: Vec<SealedTeacherPolicyV2>,
    commit_failure: Option<Box<CommitTeacherPolicyFailureV2>>,
    failed_items: Vec<PumpItemV2>,
}

pub(super) enum OwnOfferOutcomeV2 { Pending, Offered, FailedRetained }

pub(super) struct OwnPumpConsumerV2 {
    state: Mutex<State>,
}

impl OwnPumpConsumerV2 {
    pub(super) fn new(maximum: usize) -> (Arc<Self>, SyncSender<PumpItemV2>) {
        let (sender, receiver) = mpsc::sync_channel(16);
        (Arc::new(Self { state: Mutex::new(State { receiver, items: VecDeque::new(), maximum, disconnected: false, teacher_evidence: Vec::new(), commit_failure: None, failed_items: Vec::new() }) }), sender)
    }

    /// Stops on full without removing the next original from the channel.
    pub(super) fn poll(&self) -> Result<(), ArenaError> {
        let mut state = self.state.lock().map_err(|_| ArenaError::Invalid("Own publisher custody poisoned".into()))?;
        if state.items.len().saturating_add(state.failed_items.len()).saturating_add(state.teacher_evidence.len()) >= state.maximum { return Ok(()); }
        state.items.try_reserve(1).map_err(|_| ArenaError::Invalid("Own publisher item capacity".into()))?;
        match state.receiver.try_recv() {
            Ok(item) => state.items.push_back(item),
            Err(TryRecvError::Empty) => {},
            Err(TryRecvError::Disconnected) => state.disconnected = true,
        }
        Ok(())
    }

    pub(super) fn try_counts(&self) -> Result<Option<(usize, usize, usize, bool)>, ArenaError> {
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(ArenaError::Invalid("Own publisher custody poisoned".into())),
        };
        Ok(Some((state.items.len(), state.failed_items.len(), state.teacher_evidence.len(), state.disconnected)))
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
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(OwnOfferOutcomeV2::Pending),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(ArenaError::Invalid("Own publisher custody poisoned".into())),
        };
        if state.commit_failure.is_some() { return Err(ArenaError::Invalid("Own original teacher commit failure retained".into())); }
        if state.teacher_evidence.len() >= state.maximum { return Err(ArenaError::Invalid("Own teacher evidence capacity reached".into())); }
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
                let Some(result) = owner.try_with_fenced_own_producer_mut(row, gate, handoff,
                    |producer, row| producer.attach_prepared_teacher_policy_v2(row, evidence))? else { return Ok(OwnOfferOutcomeV2::Pending); };
                match result {
                    Ok(original) => *sealed = Some(original),
                    Err(original) => { state.commit_failure = Some(original); return Err(ArenaError::Invalid("Own original teacher commit failure retained".into())); }
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
    pub(super) fn run(&self, stop: &AtomicBool, until: Instant) -> Result<(), ArenaError> {
        while !stop.load(Ordering::Acquire) && Instant::now() < until {
            self.poll()?;
            thread::sleep(Duration::from_millis(2));
        }
        // Preserve any remainder in the original receiver; later explicit
        // draining is performed by the same owner.
        self.poll()
    }
}
