//! Bounded custody for actual Own collection pump items.
//!
//! Reception is independent of publication and never establishes a native
//! fence or teacher retirement. The original receiver and items remain in the
//! same shared owner when its worker is stopped.
use crate::{ArenaError, rovent_collect::{RoventCollectionOwnerLockV2, RoventCorpusAssemblyLockV2, RoventRetiredLockV2Record, collection_hooks_core_v2::RoventCollectionCommitGateV2, teacher::{lock_v2_policy::{CommitTeacherPolicyFailureV2, PreparedTeacherPolicyLockV2}, lock_v2_row_pump::PumpItemV2, v2_session::SealedTeacherPolicyV2}}};
use rz_uci::rovent_bootstrap::RoventNnuecCollectionHandoffV2;
use std::{collections::VecDeque, sync::{Arc, Mutex, mpsc::{self, Receiver, SyncSender, TryRecvError}, atomic::{AtomicBool, Ordering}}, thread, time::{Duration, Instant}};

#[derive(serde::Serialize)]
struct FailureRowProjection<'a, T> { sequence: usize, stage: &'static str, record: &'a T }
struct FailureRowBuffer(Vec<u8>);
impl std::io::Write for FailureRowBuffer {
    fn write(&mut self, input: &[u8]) -> std::io::Result<usize> {
        const LIMIT: usize = rz_experiments::pals_data_v3::GENERATION_MAX_ROW_BYTES + 4096;
        if self.0.len().checked_add(input.len()).is_none_or(|n| n > LIMIT) {
            return Err(std::io::Error::other("failure row exceeds original row bound"));
        }
        self.0.try_reserve(input.len()).map_err(|_| std::io::Error::other("failure row allocation"))?;
        self.0.extend_from_slice(input); Ok(input.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

fn encode_failure_row<T: serde::Serialize>(sequence: usize, stage: &'static str, record: &T)
    -> Result<Vec<u8>, serde_json::Error> {
    let mut encoded = FailureRowBuffer(Vec::new());
    serde_json::to_writer(&mut encoded, &FailureRowProjection { sequence, stage, record })?;
    encoded.0.push(b'\n');
    Ok(encoded.0)
}

fn write_failure_row(output: &mut impl std::io::Write, bytes: &[u8], budget: &mut u64)
    -> std::io::Result<bool> {
    let size = bytes.len() as u64;
    if size > *budget { return Ok(false); }
    output.write_all(bytes)?;
    *budget -= size;
    Ok(true)
}

#[cfg(test)]
mod failure_stream_tests {
    use super::*;

    #[test]
    fn failure_rows_keep_identity_and_order_beyond_old_projection_limit() {
        let padding = "x".repeat(900_000);
        let mut stream = Vec::new();
        for sequence in 0..6 {
            let record = serde_json::json!({"question_id":format!("question-{sequence}"),"payload":&padding});
            stream.extend(encode_failure_row(sequence, "prepared", &record).unwrap());
        }
        assert!(stream.len() > 4 * 1024 * 1024);
        for (sequence, line) in stream.split(|byte| *byte == b'\n').filter(|line| !line.is_empty()).enumerate() {
            let value: serde_json::Value = serde_json::from_slice(line).unwrap();
            assert_eq!(value["sequence"], sequence);
            assert_eq!(value["stage"], "prepared");
            assert_eq!(value["record"]["question_id"], format!("question-{sequence}"));
        }
        let too_large = "x".repeat(rz_experiments::pals_data_v3::GENERATION_MAX_ROW_BYTES + 8192);
        assert!(encode_failure_row(0, "failed", &too_large).is_err());
        let mut output = Vec::new();
        let mut budget = 3;
        assert!(!write_failure_row(&mut output, b"abcd", &mut budget).unwrap());
        assert!(output.is_empty());
        assert_eq!(budget, 3);
        assert!(write_failure_row(&mut output, b"abc", &mut budget).unwrap());
        assert_eq!(output, b"abc");
        assert_eq!(budget, 0);
    }
}

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
    /// Stream original retained row projections without moving their owners.
    /// Only called after this same receiver's sender has joined. The bytes
    /// are failure evidence, never a corpus or a reconstructable authority.
    pub(super) fn try_stream_failure_rows(
        &self, output: &mut impl std::io::Write, budget: &mut u64, maximum_rows: usize,
    ) -> Result<Option<serde_json::Value>, ArenaError> {
        use sha2::{Digest as _, Sha256};
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(ArenaError::Invalid("failure stream consumer poisoned".into())),
        };
        let total = state.items.len().saturating_add(state.failed_items.len())
            .saturating_add(usize::from(state.overflow_item.is_some()));
        let mut sha = Sha256::new();
        let mut bytes = 0u64;
        let mut rows = 0usize;
        let mut reason = None;
        for item in state.items.iter().chain(state.failed_items.iter()).chain(state.overflow_item.iter()) {
            if rows >= maximum_rows { reason = Some("original_record_limit"); break; }
            let (stage, row) = match item {
                PumpItemV2::Unlabeled { row } => ("unlabeled", row),
                PumpItemV2::Prepared { row, .. } => ("prepared", row),
                PumpItemV2::Committed { row, .. } => ("committed", row),
                PumpItemV2::Failed { row, .. } => ("failed", row),
            };
            let encoded = match encode_failure_row(rows, stage, &row.record) {
                Ok(value) => value,
                Err(_) => { reason = Some("row_encoding_or_original_row_limit"); break; }
            };
            let size = encoded.len() as u64;
            if !write_failure_row(output, &encoded, budget)
                .map_err(|error| ArenaError::Io(format!("failure stream write: {error}")))? {
                reason = Some("original_output_byte_limit"); break;
            }
            sha.update(&encoded);
            bytes += size; rows += 1;
        }
        Ok(Some(serde_json::json!({"rows":rows,"bytes":bytes,"sha256":format!("{:x}",sha.finalize()),
            "omitted":total.saturating_sub(rows),"reason":reason,
            "scope":"retained_consumer_failure_projection_not_original_authority","training_eligible":false})))
    }
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

    /// Describe retained failures without moving their original rows or
    /// completion evidence. The receipt reports truncation explicitly.
    pub(crate) fn try_failure_observations(&self) -> Result<Option<serde_json::Value>, ArenaError> {
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(ArenaError::Invalid("Own publisher custody poisoned".into())),
        };
        let describe = |text: String| {
            let chars = text.chars().count();
            serde_json::json!({"text":text.chars().take(512).collect::<String>(),
                "original_chars":chars, "truncated":chars > 512})
        };
        let mut entries = Vec::new();
        let mut total = 0usize;
        for item in state.items.iter().chain(state.failed_items.iter()).chain(state.overflow_item.iter()) {
            if let PumpItemV2::Failed { row, error, evidence } = item {
                total += 1;
                if entries.len() < 32 {
                    entries.push(serde_json::json!({"error":describe(error.to_string()),
                        "evidence":describe(evidence.diagnostic()),
                        "original_question_id":row.record.0.input.question_id,
                        "original_job_id":row.record.0.input.job_id,
                        "original_slice_id":row.record.0.input.slice_id}));
                }
            }
        }
        let commits = state.commit_failures.iter().take(32).map(|failure| serde_json::json!({
            "error":describe(failure.error.to_string()), "panicked":failure.panic.is_some()
        })).collect::<Vec<_>>();
        Ok(Some(serde_json::json!({"count":total, "reported":entries, "omitted":total.saturating_sub(32),
            "commit_failure_count":state.commit_failures.len(), "commit_failures":commits,
            "commit_failures_omitted":state.commit_failures.len().saturating_sub(32)})))
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
        // A drained queue requires no new evidence or failure custody. This
        // remains a valid terminal observation even when the last row filled
        // the configured record capacity exactly.
        if state.items.is_empty() { return Ok(OwnOfferOutcomeV2::Pending); }
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
